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
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <utility>
#include <vector>

namespace tvm {
namespace tir {
namespace transform {
namespace {

/*!
 * \brief Analyze the read and write accesses of the body statements, used by `LocalPadder`.
 */
class StorageAccessAnalyzer : private StmtExprVisitor {
 private:
  struct StorageType {
    enum { kGlobal = 0, kShared, kLocal, kOthers };
  };

  void VisitStmt_(const BufferStoreNode* op) final {
    write_marker_.SetStorageAccessMarker_(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const BufferLoadNode* op) final {
    read_marker_.SetStorageAccessMarker_(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }
  class AccessMarker {
   public:
    void SetStorageAccessMarker_(const Buffer& buf) {
      if (buf.scope() == "global") {
        bit_vector_[StorageType::kGlobal] = true;
      } else if (buf.scope() == "shared") {
        bit_vector_[StorageType::kShared] = true;
      } else if (buf.scope() == "local") {
        bit_vector_[StorageType::kLocal] = true;
      } else {
        bit_vector_[StorageType::kOthers] = true;
      }
    }
    bool NoAccesses() const {
      return !(bit_vector_[StorageType::kGlobal] || bit_vector_[StorageType::kShared] ||
               bit_vector_[StorageType::kLocal] || bit_vector_[StorageType::kOthers]);
    }
    bool OnlyGlobalAccesses() const {
      return !(bit_vector_[StorageType::kShared] || bit_vector_[StorageType::kLocal] ||
               bit_vector_[StorageType::kOthers]) &&
             bit_vector_[StorageType::kGlobal];
    }
    bool OnlyLocalOrSharedAccesses() const {
      return !(bit_vector_[StorageType::kGlobal] || bit_vector_[StorageType::kOthers]) &&
             (bit_vector_[StorageType::kShared] || bit_vector_[StorageType::kLocal]);
    }

   private:
    std::array<bool, StorageType::kOthers + 1> bit_vector_ = {false};
  };
  AccessMarker read_marker_, write_marker_;
  std::pair<AccessMarker, AccessMarker> Analyze_(const Stmt& stmt) {
    VisitStmt(stmt);
    return std::make_pair(read_marker_, write_marker_);
  }

  friend class LocalPadder;
};

/*!
 * \brief Verify that all local variables are initialized to the same constant expression.
 */
class InitChecker : public StmtVisitor {
 private:
  void VisitStmt_(const BufferStoreNode* op) final {
    // Read the check the RHS values, make sure that they are the same constant for all the
    // initialization statements.
    CheckInitValue_<IntImmNode>(op->value);
    CheckInitValue_<FloatImmNode>(op->value);
    return StmtVisitor::VisitStmt_(op);
  }
  template <typename ImmNodeType>
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
 *        Note that not all predicates can be inlined. For example, if a predicate is there to guard
 *        against out-of-boundary accesses to local/shared variables, then it cannot be inlined.
 */
class PredicateInliner : public StmtExprVisitor {
 private:
  explicit PredicateInliner(const Stmt& body_stmt) : body_stmt_(body_stmt) {}

#define VISIT_PREDICATE(OpType)                      \
  void VisitExpr_(const OpType##Node* op) final {    \
    OpType predicate = GetRef<OpType>(op);           \
    if (CanInlinePredicate_<OpType##Node>(op)) {     \
      inlinable_predicates_.push_back(predicate);    \
    } else {                                         \
      non_inlinable_residuals_.push_back(predicate); \
    }                                                \
  }
  VISIT_PREDICATE(LT)
  VISIT_PREDICATE(LE)
  VISIT_PREDICATE(GT)
  VISIT_PREDICATE(GE)
#undef VISIT_PREDICATE

  void VisitStmt_(const BufferStoreNode* op) final {
    if (op->indices.size() != 1) {
      return StmtVisitor::VisitStmt_(op);
    }
    CHECK(op->buffer.scope() == "shared" || op->buffer.scope() == "local");
    if (StructuralEqual()(op->indices[0], predicate_lhs_)) {
      predicate_inlinable_ = false;
    }
    return StmtVisitor::VisitStmt_(op);
  }
  /*!
   * \brief Check if a predicate can be inlined.
   */
  template <typename OpNodeType>
  bool CanInlinePredicate_(const OpNodeType* op) {
    predicate_inlinable_ = true;
    predicate_lhs_ = op->a;
    VisitStmt(body_stmt_);
    return predicate_inlinable_;
  }

  Stmt body_stmt_;
  std::vector<PrimExpr> inlinable_predicates_, non_inlinable_residuals_;
  bool predicate_inlinable_;
  PrimExpr predicate_lhs_;

  friend class LocalPadder;
};

class LocalPadder : public StmtExprMutator {
 private:
  Stmt VisitStmt_(const IfThenElseNode* op) final {
    if (op->else_case.defined()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    // Analyze the reads and writes of the body statements.
    StorageAccessAnalyzer::AccessMarker read_marker, write_marker;
    std::tie(read_marker, write_marker) = StorageAccessAnalyzer().Analyze_(op->then_case);

    if (read_marker.NoAccesses() && write_marker.OnlyLocalOrSharedAccesses()) {
      // In the case when there are no buffer reads and only local/shared buffer writes, remove the
      // predicates and obtain the buffer initialization values.
      init_checker_(op->then_case);
      if (!init_checker_.init_constexpr_.defined()) {
        return StmtExprMutator::VisitStmt_(op);
      }
      return StmtExprMutator::VisitStmt(op->then_case);
    } else if (read_marker.OnlyGlobalAccesses() && write_marker.OnlyLocalOrSharedAccesses()) {
      // In the case when there are global buffer reads and local/shared buffer writes, inline the
      // predicates as part of the buffer store statements.
      PredicateInliner predicate_inliner(op->then_case);
      predicate_inliner(op->condition);
      size_t predicate_stack_current_size = predicate_stack_.size();
      // Push the inlinable predicates on top of the stack.
      if (!predicate_inliner.inlinable_predicates_.empty()) {
        predicate_stack_.insert(predicate_stack_.end(),
                                predicate_inliner.inlinable_predicates_.begin(),
                                predicate_inliner.inlinable_predicates_.end());
      }

      enable_padding_ = true;
      // Update the body statements by inlining the predicates.
      Stmt inlined_body_stmt = VisitStmt(op->then_case);
      enable_padding_ = false;

      if (!predicate_inliner.inlinable_predicates_.empty()) {
        predicate_stack_.erase(predicate_stack_.begin() + predicate_stack_current_size,
                               predicate_stack_.end());
      }
      if (predicate_inliner.non_inlinable_residuals_.empty()) {
        return inlined_body_stmt;
      }
      return IfThenElse(FlattenPredicates_(predicate_inliner.non_inlinable_residuals_),
                        inlined_body_stmt);
    } else if (read_marker.OnlyLocalOrSharedAccesses() &&
               write_marker.OnlyLocalOrSharedAccesses()) {
      // In the case when there are global buffer reads and local/shared buffer writes, inline the
      // predicates as part of the buffer store statements.
      return StmtExprMutator::VisitStmt(op->then_case);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (!enable_padding_ || predicate_stack_.empty()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    PrimExpr store_predicate = FlattenPredicates_(predicate_stack_);
    return BufferStore(op->buffer,
                       Select(store_predicate, op->value, ComposePaddedValue_(op->value->dtype)),
                       op->indices);
  }
  PrimExpr ComposePaddedValue_(const DataType& dtype) const {
    if (init_checker_.init_constexpr_->dtype != dtype) {
      return Cast(dtype, init_checker_.init_constexpr_);
    }
    return init_checker_.init_constexpr_;
  }
  PrimExpr FlattenPredicates_(const Array<PrimExpr>& predicates) const {
    CHECK(!predicates.empty());
    PrimExpr ret = predicates.front();
    for (auto predicates_it = predicates.begin() + 1; predicates_it != predicates.end();
         ++predicates_it) {
      ret = ret && (*predicates_it);
    }
    return ret;
  }

  InitChecker init_checker_;
  std::vector<PrimExpr> predicate_stack_;
  bool enable_padding_ = false;
};

Stmt LocalPadTransform(Stmt stmt) {
  // Skip the local padding optimization in the case when there is no single constant expression
  // used for initialization.
  LocalPadder local_padder;
  stmt = local_padder(std::move(stmt));
  return stmt;
}

struct LocalPadConfigNode : public tvm::AttrsNode<LocalPadConfigNode> {
  bool enable;

  TVM_DECLARE_ATTRS(LocalPadConfigNode, "tir.transform.LocalPadConfig") {
    TVM_ATTR_FIELD(enable).describe("Enable local padding").set_default(false);
  }
};

class LocalPadConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LocalPadConfig, Attrs, LocalPadConfigNode);
};

TVM_REGISTER_NODE_TYPE(LocalPadConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.LocalPad", LocalPadConfig);

}  // anonymous namespace

Pass LocalPad() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    PrimFuncNode* mutable_func_node = f.CopyOnWrite();
    Optional<LocalPadConfig> cfg = ctx->GetConfig<LocalPadConfig>("tir.LocalPad");
    if (!cfg.defined()) {
      cfg = AttrsWithDefaultValues<LocalPadConfig>();
    }
    mutable_func_node->body = LocalPadTransform(std::move(mutable_func_node->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LocalPad", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LocalPad").set_body_typed(LocalPad);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
