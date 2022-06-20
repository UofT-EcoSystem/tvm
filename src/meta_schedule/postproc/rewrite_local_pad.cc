/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/meta_schedule/postproc.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <regex>
#include <utility>
#include <vector>

namespace tvm {
namespace tir {
namespace {

inline bool NameMatchesRegexPattern(const String& name, const std::string& pattern) {
  return std::regex_match(std::string(name), std::regex("^" + pattern + "$"));
}

/*!
 * \brief Analyze the read and write accesses of the body statements, used by `LocalPadder`.
 */
class StorageAccessAnalyzer {
 private:
  struct StorageType {
    enum { kGlobal = 0, kShared, kLocal, kOthers };
  };

  const StorageAccessAnalyzer& operator()(const Array<BufferRegion>& buffer_regions) {
    access_marker_ = std::vector<bool>(StorageType::kOthers + 1, false);
    for (const BufferRegion& buffer_region : buffer_regions) {
      SetStorageAccessMarker_(buffer_region->buffer);
    }
    return *this;
  }
  std::vector<bool> access_marker_;
  void SetStorageAccessMarker_(const Buffer& buf) {
    if (buf.scope() == "global") {
      access_marker_[StorageType::kGlobal] = true;
    } else if (buf.scope() == "shared") {
      access_marker_[StorageType::kShared] = true;
    } else if (buf.scope() == "local") {
      access_marker_[StorageType::kLocal] = true;
    } else {
      access_marker_[StorageType::kOthers] = true;
    }
  }
  bool NoAccesses_() const {
    return !(access_marker_[StorageType::kGlobal] || access_marker_[StorageType::kShared] ||
             access_marker_[StorageType::kLocal] || access_marker_[StorageType::kOthers]);
  }
  bool OnlyGlobalAccesses_() const {
    return !(access_marker_[StorageType::kShared] || access_marker_[StorageType::kLocal] ||
             access_marker_[StorageType::kOthers]) &&
           access_marker_[StorageType::kGlobal];
  }
  bool OnlyLocalOrSharedAccesses_() const {
    return !(access_marker_[StorageType::kGlobal] || access_marker_[StorageType::kOthers]) &&
           (access_marker_[StorageType::kShared] || access_marker_[StorageType::kLocal]);
  }

  friend class LocalPadder;
};

/*!
 * \brief Verify that all local variables are initialized to the same constant expression.
 */
class InitChecker : public StmtVisitor {
 private:
  void VisitStmt_(const BufferStoreNode* op) final {
    if (!inside_init_block_) {
      return StmtVisitor::VisitStmt_(op);
    }
    // Read the check the RHS values, make sure that they are the same constant for all the
    // initialization statements.
    CheckInitValue_<IntImmNode>(op->value);
    CheckInitValue_<FloatImmNode>(op->value);
    return StmtVisitor::VisitStmt_(op);
  }
  template<typename ImmNodeType>
  void CheckInitValue_(const PrimExpr& rhs) {
    if (const ImmNodeType* const rhs_val = rhs.as<ImmNodeType>()) {
      if (init_constexpr_.defined()) {
        if (const ImmNodeType* const init_val = init_constexpr_.as<ImmNodeType>()) {
          if (rhs_val->value != init_val->value) {
            init_with_single_constexpr_ = false;
          }
        } else {
          init_with_single_constexpr_ = false;
        }
      } else {
        init_with_single_constexpr_ = true;
        init_constexpr_ = rhs;
      }
    }
  }
  void operator()(const Stmt& stmt) {
    StmtVisitor::operator()(stmt);
    if (!init_with_single_constexpr_) {
      init_constexpr_ = PrimExpr();
    }
  }

  bool inside_init_block_ = false;
  bool init_with_single_constexpr_ = false;
  PrimExpr init_constexpr_;

  friend class LocalPadder;
};

/*!
 * \brief Split a predicate into inlinable and non-inlinable component.
 *
 *        We refer to "inlinable predicate" as
 *
 *            if (predicate) A = ...;
 *            ↓
 *            A = predicate ? ... : init_constexpr;
 *
 *        Note that not all predicates can be inlined. E.g., `threadIdx.x < 120` cannot be inlined,
 *        since doing so could lead to invalid memory accesses.
 */
class PredicateInliner : public ExprMutator {
 private:
  PrimExpr VisitExpr_(const AndNode* op) final {
    if (!op->a.as<AndNode>()) {
      if (!CanInlinePredicate_(op->a)) {
        non_inlinable_residual_ = non_inlinable_residual_.defined() ?
                                    And(non_inlinable_residual_, op->a) : op->a;
        return op->b;
      }
    }
    if (!op->b.as<AndNode>()) {
      if (!CanInlinePredicate_(op->b)) {
        non_inlinable_residual_ = non_inlinable_residual_.defined() ?
                                    And(non_inlinable_residual_, op->b) : op->b;
        return op->a;
      }
    }
    return ExprMutator::VisitExpr_(op);
  }
  PrimExpr VisitExpr_(const VarNode* op) final {
    if (!NameMatchesRegexPattern(op->name_hint, "((ax\\d_)+)fused_(\\d+)")) {
      predicate_inlinable_ = true;
    }
    return ExprMutator::VisitExpr_(op);
  }
  /*!
   * \brief Check if a predicate can be inlined. We cannot inline a predicate if it consists of
   *        `threadIdx.*` and serial iteration variables. 
   */
  bool CanInlinePredicate_(const PrimExpr& predicate) {
    predicate_inlinable_ = false;
    VisitExpr(predicate);
    return predicate_inlinable_;
  }

  PrimExpr non_inlinable_residual_;
  bool predicate_inlinable_ = false;

  friend class LocalPadder;
};

class LocalPadder : public StmtExprMutator {
 private:
  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    if (StorageAccessAnalyzer()(op->block->reads).NoAccesses_() &&
        StorageAccessAnalyzer()(op->block->writes).OnlyLocalOrSharedAccesses_()) {
      if (!NameMatchesRegexPattern(op->block->name_hint, "(.+)_init")) {
        LOG(WARNING) << "Treating " << op->block << " as an initialization block "
                        "as it only has local/shared memory writes";
      }
      init_checker_(op->block);
      // Remove all the predicates in the initialization step.
      return BlockRealize(op->iter_values, Bool(true),
                          Downcast<Block>(StmtExprMutator::VisitStmt(op->block)));
    } else if (StorageAccessAnalyzer()(op->block->reads).OnlyGlobalAccesses_() &&
               StorageAccessAnalyzer()(op->block->writes).OnlyLocalOrSharedAccesses_()) {
      if (!init_checker_.init_constexpr_.defined()) {
        return StmtExprMutator::VisitStmt_(op);
      }
      PredicateInliner predicate_inliner;
      predicate_stack_.push_back(predicate_inliner(op->predicate));
      enable_padding_ = true;
      Block op_block = Downcast<Block>(VisitStmt(op->block));
      enable_padding_ = false;
      predicate_stack_.pop_back();
      return BlockRealize(op->iter_values, predicate_inliner.non_inlinable_residual_, op_block);
    } else if (StorageAccessAnalyzer()(op->block->reads).OnlyLocalOrSharedAccesses_() &&
               StorageAccessAnalyzer()(op->block->writes).OnlyLocalOrSharedAccesses_()) {
      // Remove all the predicates in the compute step.
      return BlockRealize(op->iter_values, Bool(true),
                          Downcast<Block>(StmtExprMutator::VisitStmt(op->block)));
    }
    return StmtMutator::VisitStmt_(op);
  }
  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (!enable_padding_) {
      return StmtExprMutator::VisitStmt_(op);
    }
    return BufferStore(op->buffer, Select(ComposePredicate_(), op->value,
                                          ComposePaddedValue_(op->value->dtype)),
                       op->indices);
  }
  PrimExpr ComposePredicate_() const {
    PrimExpr predicate = Bool(true);
    for (PrimExpr expr : predicate_stack_) {
      predicate = And(predicate, expr);
    }
    return predicate;
  }
  PrimExpr ComposePaddedValue_(const DataType& dtype) const {
    if (init_checker_.init_constexpr_->dtype != dtype) {
      return Cast(dtype, init_checker_.init_constexpr_);
    }
    return init_checker_.init_constexpr_;
  }

  InitChecker init_checker_;
  std::vector<PrimExpr> predicate_stack_;
  bool enable_padding_ = false;
};

}  // anonymous namespace

static Stmt LocalPad(Stmt stmt) {
  // Skip the local padding optimization in the case when there is no single constant expression
  // used for initialization.
  LocalPadder local_padder;
  stmt = local_padder(std::move(stmt));
  return stmt;
}

namespace transform {

Pass LocalPad() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    PrimFuncNode* mutable_func = f.CopyOnWrite();
    mutable_func->body = LocalPad(std::move(mutable_func->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LocalPad", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LocalPad").set_body_typed(LocalPad);

}  // namespace transform
}  // namespace tir

namespace meta_schedule {
namespace {

class RewriteLocalPadNode : public PostprocNode {
 public:
  void InitializeWithTuneContext(const TuneContext& context) final {}
  bool Apply(const tir::Schedule& sch) final {
    tir::transform::Pass local_pad_pass = tir::transform::LocalPad();
    sch->state().get()->mod = local_pad_pass(sch->state().get()->mod);
    return true;
  }
  void VisitAttrs(tvm::AttrVisitor* v) {}
  static constexpr const char* _type_key = "meta_schedule.RewriteLocalPad";
  TVM_DECLARE_FINAL_OBJECT_INFO(RewriteLocalPadNode, PostprocNode);
};

}  // anonymous namespace

Postproc Postproc::RewriteLocalPad() {
  ObjectPtr<RewriteLocalPadNode> n = make_object<RewriteLocalPadNode>();
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(RewriteLocalPadNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocRewriteLocalPad")
    .set_body_typed(Postproc::RewriteLocalPad);

}  // namespace meta_schedule
}  // namespace tvm
