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

#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <array>
#include <utility>
#include <vector>

#include "../../runtime/thread_storage_scope.h"

namespace tvm {
namespace tir {
namespace transform {

/*!
 * \brief Analyze the read and write accesses of the body statements, used by `LocalPadder`.
 */
class StorageAccessAnalyzer : public StmtExprVisitor {
 private:
  enum class StorageType : int32_t { kGlobal = 0, kShared = 1, kLocal = 2, kOthers = 3 };

  void VisitStmt_(const BufferStoreNode* op) final {
    write_marker_.SetStorageAccessMarker(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const BufferLoadNode* op) final {
    read_marker_.SetStorageAccessMarker(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }
  class AccessMarker {
   public:
    void SetStorageAccessMarker(const Buffer& buf) {
      using runtime::StorageScope;

      if (StorageScope::Create(buf.scope()) == StorageScope::Create("global")) {
        bit_vector_[static_cast<int>(StorageType::kGlobal)] = true;
      } else if (StorageScope::Create(buf.scope()) == StorageScope::Create("shared")) {
        bit_vector_[static_cast<int>(StorageType::kShared)] = true;
      } else if (StorageScope::Create(buf.scope()) == StorageScope::Create("local")) {
        bit_vector_[static_cast<int>(StorageType::kLocal)] = true;
      } else {
        bit_vector_[static_cast<int>(StorageType::kOthers)] = true;
      }
    }
    bool NoAccesses() const {
      return !(bit_vector_[static_cast<int>(StorageType::kGlobal)] ||
               bit_vector_[static_cast<int>(StorageType::kShared)] ||
               bit_vector_[static_cast<int>(StorageType::kLocal)] ||
               bit_vector_[static_cast<int>(StorageType::kOthers)]);
    }
    bool OnlyGlobalAccesses() const {
      return !(bit_vector_[static_cast<int>(StorageType::kShared)] ||
               bit_vector_[static_cast<int>(StorageType::kLocal)] ||
               bit_vector_[static_cast<int>(StorageType::kOthers)]) &&
             bit_vector_[static_cast<int>(StorageType::kGlobal)];
    }
    bool OnlyLocalOrSharedAccesses() const {
      return !(bit_vector_[static_cast<int>(StorageType::kGlobal)] ||
               bit_vector_[static_cast<int>(StorageType::kOthers)]) &&
             (bit_vector_[static_cast<int>(StorageType::kShared)] ||
              bit_vector_[static_cast<int>(StorageType::kLocal)]);
    }

   private:
    std::array<bool, static_cast<int>(StorageType::kOthers) + 1> bit_vector_ = {false};
  };
  AccessMarker read_marker_, write_marker_;
  std::pair<AccessMarker, AccessMarker> Analyze(const Stmt& stmt) {
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
    CheckInitValue<IntImmNode>(op->value);
    CheckInitValue<FloatImmNode>(op->value);
    return StmtVisitor::VisitStmt_(op);
  }
  template <typename ImmNodeType>
  void CheckInitValue(const PrimExpr& rhs) {
    if (const ImmNodeType* rhs_val = rhs.as<ImmNodeType>()) {
      if (init_constexpr_) {
        if (const ImmNodeType* init_val = init_constexpr_.value().as<ImmNodeType>()) {
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
  Optional<PrimExpr> init_constexpr_;

  friend class LocalPadder;
};

/*!
 * \brief Split a predicate into inlinable and non-inlinable component.
 *
 *        We refer to "inlinable predicate" as
 *
 *            if (predicate) A = ...;
 *            |
 *            A = predicate ? ... : init_constexpr;
 *
 *        Note that not all predicates can be inlined. For example, if a predicate is there to guard
 *        against out-of-boundary accesses to local/shared variables, then it cannot be inlined.
 */
class PredicateInliner : public StmtExprVisitor {
 private:
  explicit PredicateInliner(const Stmt& body_stmt) : body_stmt_(body_stmt) {}

#define TVM_TIR_TRANSFORM_LOCAL_PAD_VISIT_PREDICATE(OpType) \
  void VisitExpr_(const OpType::ContainerType* op) final {  \
    OpType predicate = GetRef<OpType>(op);                  \
    if (CanInlinePredicate<OpType::ContainerType>(op)) {    \
      inlinable_predicates_.push_back(predicate);           \
    } else {                                                \
      non_inlinable_residuals_.push_back(predicate);        \
    }                                                       \
  }
  TVM_TIR_TRANSFORM_LOCAL_PAD_VISIT_PREDICATE(LT)
  TVM_TIR_TRANSFORM_LOCAL_PAD_VISIT_PREDICATE(LE)
  TVM_TIR_TRANSFORM_LOCAL_PAD_VISIT_PREDICATE(GT)
  TVM_TIR_TRANSFORM_LOCAL_PAD_VISIT_PREDICATE(GE)
#undef TVM_TIR_TRANSFORM_LOCAL_PAD_VISIT_PREDICATE

  void VisitStmt_(const BufferStoreNode* op) final {
    if (op->indices.size() != 1) {
      return StmtVisitor::VisitStmt_(op);
    }
    using runtime::StorageScope;
    CHECK(StorageScope::Create(op->buffer.scope()) == StorageScope::Create("shared") ||
          StorageScope::Create(op->buffer.scope()) == StorageScope::Create("local"));
    if (StructuralEqual()(op->indices[0], predicate_lhs_)) {
      predicate_inlinable_ = false;
    }
    return StmtVisitor::VisitStmt_(op);
  }
  /*!
   * \brief Check if a predicate can be inlined.
   */
  template <typename OpNodeType>
  bool CanInlinePredicate(const OpNodeType* op) {
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
    if (!is_no_op(op->else_case)) {
      return StmtExprMutator::VisitStmt_(op);
    }
    // Analyze the reads and writes of the body statements.
    StorageAccessAnalyzer::AccessMarker read_marker, write_marker;
    std::tie(read_marker, write_marker) = StorageAccessAnalyzer().Analyze(op->then_case);

    if (read_marker.NoAccesses() && write_marker.OnlyLocalOrSharedAccesses()) {
      // In the case when there are no buffer reads and only local/shared buffer writes, remove the
      // predicates and obtain the buffer initialization values.
      init_checker_(op->then_case);
      if (!init_checker_.init_constexpr_) {
        return StmtExprMutator::VisitStmt_(op);
      }
      return StmtExprMutator::VisitStmt(op->then_case);
    } else if (read_marker.OnlyGlobalAccesses() && write_marker.OnlyLocalOrSharedAccesses()) {
      if (!init_checker_.init_constexpr_) {
        return StmtExprMutator::VisitStmt_(op);
      }
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
      return IfThenElse(FlattenPredicates(predicate_inliner.non_inlinable_residuals_),
                        inlined_body_stmt);
    } else if (read_marker.OnlyLocalOrSharedAccesses() &&
               write_marker.OnlyLocalOrSharedAccesses()) {
      // In the case when there are global buffer reads and local/shared buffer writes, remove the
      // predicates.
      return StmtExprMutator::VisitStmt(op->then_case);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (!enable_padding_ || predicate_stack_.empty()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    PrimExpr store_predicate = FlattenPredicates(predicate_stack_);
    return BufferStore(op->buffer,
                       Select(store_predicate, op->value, ComposePaddedValue(op->value->dtype)),
                       op->indices);
  }
  PrimExpr ComposePaddedValue(const DataType& dtype) const {
    CHECK(init_checker_.init_constexpr_);
    if (init_checker_.init_constexpr_.value()->dtype != dtype) {
      return Cast(dtype, init_checker_.init_constexpr_.value());
    }
    return init_checker_.init_constexpr_.value();
  }
  PrimExpr FlattenPredicates(const Array<PrimExpr>& predicates) const {
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

Pass LocalPad(bool enable_local_pad) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    if (!enable_local_pad) {
      return f;
    }
    PrimFuncNode* mutable_func_node = f.CopyOnWrite();
    mutable_func_node->body = LocalPadTransform(std::move(mutable_func_node->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LocalPad", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LocalPad").set_body_typed(LocalPad);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
