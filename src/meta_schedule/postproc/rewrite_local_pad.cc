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

#include <tvm/arith/analyzer.h>
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

inline bool BlockNameMatchesRegexPattern(const String& block_name, const std::string& pattern) {
  return std::regex_match(std::string(block_name), std::regex(pattern));
}

enum class StorageType {
  kGlobal = 0,
  kShared,
  kLocal,
  kOthers
};

/*!
 * \brief Analyze the read and write accesses of the body statements, used by `LocalPadder`.
 */
class StorageAccessAnalyzer {
 private:
  StorageAccessAnalyzer& operator()(const Array<BufferRegion>& buffer_regions) {
    access_marker_ = std::vector<bool>(int(StorageType::kOthers) + 1, false);
    for (const BufferRegion& buffer_region : buffer_regions) {
      SetStorageAccessMarker_(buffer_region->buffer);
    }
    return *this;
  }
  std::vector<bool> access_marker_;
  void SetStorageAccessMarker_(const Buffer& buf) {
    if (buf.scope() == "global") {
      access_marker_[int(StorageType::kGlobal)] = true;
    } else if (buf.scope() == "shared") {
      access_marker_[int(StorageType::kShared)] = true;
    } else if (buf.scope() == "local") {
      access_marker_[int(StorageType::kLocal)] = true;
    } else {
      access_marker_[int(StorageType::kOthers)] = true;
    }
  }
  bool NoAccesses_() const {
    return !(access_marker_[int(StorageType::kGlobal)] ||
             access_marker_[int(StorageType::kShared)] ||
             access_marker_[int(StorageType::kLocal)] ||
             access_marker_[int(StorageType::kOthers)]);
  }
  bool OnlyGlobalAccesses_() const {
    return !(access_marker_[int(StorageType::kShared)] ||
             access_marker_[int(StorageType::kLocal)] ||
             access_marker_[int(StorageType::kOthers)]) &&
           access_marker_[int(StorageType::kGlobal)];
  }
  bool OnlyLocalOrSharedAccesses_() const {
    return !(access_marker_[int(StorageType::kGlobal)] ||
             access_marker_[int(StorageType::kOthers)]) &&
           (access_marker_[int(StorageType::kShared)] ||
            access_marker_[int(StorageType::kLocal)]);
  }

  friend class LocalPadInitChecker;
  friend class LocalPadder;
};

/*!
 * \brief Verify that all local variables are initialized to the same value.
 */
class LocalPadInitChecker : public StmtVisitor {
 private:
  void VisitStmt_(const BufferStoreNode* op) final {
    if (!inside_init_block_) {
      return StmtVisitor::VisitStmt_(op);
    }
    const PrimExpr& rhs = op->value;
    // Read the check the RHS values, make sure that they are the same constant for all the
    // initialization statements.
#define CHECK_INIT_VALUE(imm_node_type)                                                  \
    if (const imm_node_type* const rhs_val = rhs.as<imm_node_type>()) {                  \
      if (init_constexpr_.defined()) {                                                   \
        if (const imm_node_type* const init_val = init_constexpr_.as<imm_node_type>()) { \
          if (rhs_val->value != init_val->value) {                                       \
            init_with_single_constexpr_ = false;                                         \
          }                                                                              \
        } else {                                                                         \
          init_with_single_constexpr_ = false;                                           \
        }                                                                                \
      } else {                                                                           \
        init_with_single_constexpr_ = true;                                              \
        init_constexpr_ = rhs;                                                           \
      }                                                                                  \
    }

    CHECK_INIT_VALUE(IntImmNode)
    CHECK_INIT_VALUE(FloatImmNode)
    return StmtVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const BlockNode* op) final {
    // Detect the presence of an initialization block.
    if (BlockNameMatchesRegexPattern(op->name_hint, "^(.+)_init$")) {
      LOG(INFO) << op->reads << " and " << op->writes;


      inside_init_block_ = true;
      StmtVisitor::VisitStmt_(op);
      inside_init_block_ = false;
      return;
    }
    return StmtVisitor::VisitStmt_(op);
  }
 public:
  PrimExpr useSingleConstExpr() const {
    if (init_with_single_constexpr_) {
      return init_constexpr_;
    }
    return PrimExpr();
  }
 private:
  bool inside_init_block_ = false;
  bool init_with_single_constexpr_ = false;
  PrimExpr init_constexpr_;
};

class LocalPadder : public StmtExprMutator {
 public:
  explicit LocalPadder(const PrimExpr& init_constexpr) : init_constexpr_(init_constexpr) {}
 private:
  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    if (blockNameMatchesPattern(op->block->name_hint, "^(.+)_init$")) {
      // Remove all the predicates in the initialization step.
      return BlockRealize(op->iter_values, Bool(1), Downcast<Block>(VisitStmt(op->block)));
    }
    // Analyze the buffer read and write accesses.
    LocalPadStorageAccessAnalyzer access_analyzer;
    access_analyzer(op->block);
    return StmtMutator::VisitStmt_(op);
  }
  const PrimExpr init_constexpr_;
  PrimExpr inlinable_predicate_;  // We refer to "inlinable predicate" as
                                  //
                                  //     if (predicate) A = ...;
                                  //     ↓
                                  //     A = predicate ? ... : init_constexpr;
                                  //
                                  // Note that not all predicates can be inlined. E.g.,
                                  // `threadIdx.x < 120` cannot be inlined, since doing so could
                                  // lead to invalid memory accesses.
  arith::Analyzer analyzer_;
};

}  // anonymous namespace

static Stmt LocalPad(Stmt stmt) {
  LocalPadInitChecker init_checker;
  init_checker(stmt);
  PrimExpr init_constexpr = init_checker.useSingleConstExpr();
  // Skip the local padding optimization in the case when there is no single constant expression
  // used for initialization.
  if (!init_constexpr.defined()) {
    return stmt;
  }
  LocalPadder local_padder(init_constexpr);
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
