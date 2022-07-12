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

#include <tvm/ir/type.h>
#include <tvm/tir/builtin.h>
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
  enum class RWMode { kRead, kWrite, kUnset };

  void VisitExpr_(const VarNode* op) final {
    if (rw_mode_ == RWMode::kRead) {
      read_vars_.insert(op);
      read_marker_.SetStorageAccessMarker(GetRef<Var>(op));
    }
    if (rw_mode_ == RWMode::kWrite) {
      write_vars_.insert(op);
      write_marker_.SetStorageAccessMarker(GetRef<Var>(op));
    }
  }
  class ReadScope {
   public:
    explicit ReadScope(StorageAccessAnalyzer* analyzer) : analyzer_(analyzer) {}
    void EnterWithScope() { analyzer_->rw_mode_ = RWMode::kRead; }
    void ExitWithScope() { analyzer_->rw_mode_ = RWMode::kUnset; }

   private:
    StorageAccessAnalyzer* analyzer_;
  };
  class WriteScope {
   public:
    explicit WriteScope(StorageAccessAnalyzer* analyzer) : analyzer_(analyzer) {}
    void EnterWithScope() { analyzer_->rw_mode_ = RWMode::kWrite; }
    void ExitWithScope() { analyzer_->rw_mode_ = RWMode::kUnset; }

   private:
    StorageAccessAnalyzer* analyzer_;
  };

  void VisitStmt_(const BufferStoreNode* op) final {
    {
      With<WriteScope> write_scope(this);
      VisitExpr(op->buffer->data);
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const BufferLoadNode* op) final {
    {
      With<ReadScope> read_scope(this);
      VisitExpr(op->buffer->data);
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  // Check opaque accesses within the WMMA instructions.
  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::tvm_fill_fragment())) {
      With<WriteScope> write_scope(this);
      VisitExpr(op->args[0]);
    } else if (op->op.same_as(builtin::tvm_load_matrix_sync()) ||
               op->op.same_as(builtin::tvm_store_matrix_sync())) {
      {
        With<WriteScope> write_scope(this);
        VisitExpr(op->args[0]);
      }
      {
        With<ReadScope> read_scope(this);
        VisitExpr(op->args[5]);
      }
    } else if (op->op.same_as(builtin::tvm_mma_sync()) ||
               op->op.same_as(builtin::tvm_bmma_sync())) {
      {
        With<WriteScope> write_scope(this);
        VisitExpr(op->args[0]);
      }
      {
        With<ReadScope> read_scope(this);
        VisitExpr(op->args[2]);
        VisitExpr(op->args[4]);
        VisitExpr(op->args[6]);
      }
    } else if (op->op.same_as(builtin::ptx_mma()) || op->op.same_as(builtin::ptx_mma_sp())) {
      {
        With<ReadScope> read_scope(this);
        VisitExpr(op->args[6]);
        VisitExpr(op->args[8]);
      }
      {
        With<WriteScope> write_scope(this);
        VisitExpr(op->args[10]);
      }
    } else if (op->op.same_as(builtin::ptx_ldmatrix())) {
      {
        With<WriteScope> write_scope(this);
        VisitExpr(op->args[3]);
      }
      {
        With<ReadScope> read_scope(this);
        VisitExpr(op->args[5]);
      }
    } else if (op->op.same_as(builtin::mma_store())) {
      {
        With<WriteScope> write_scope(this);
        VisitExpr(op->args[2]);
      }
      {
        With<ReadScope> read_scope(this);
        VisitExpr(op->args[3]);
      }
    } else if (op->op.same_as(builtin::mma_fill())) {
      With<WriteScope> write_scope(this);
      VisitExpr(op->args[1]);
    } else if (op->op.same_as(builtin::ptx_cp_async())) {
      {
        With<WriteScope> write_scope(this);
        VisitExpr(op->args[0]);
      }
      {
        With<ReadScope> read_scope(this);
        VisitExpr(op->args[0]);
      }
    } else {
      return StmtExprVisitor::VisitExpr_(op);
    }
  }
  class AccessMarker {
   public:
    void SetStorageAccessMarker(const Var& var) {
      using runtime::StorageScope;

      const PointerTypeNode* ptr_type = var->type_annotation.as<PointerTypeNode>();
      if (ptr_type == nullptr) {
        return;
      }
      if (StorageScope::Create(ptr_type->storage_scope) == StorageScope::Create("global")) {
        bit_vector_[static_cast<int>(StorageType::kGlobal)] = true;
      } else if (StorageScope::Create(ptr_type->storage_scope) == StorageScope::Create("shared")) {
        bit_vector_[static_cast<int>(StorageType::kShared)] = true;
      } else if (StorageScope::Create(ptr_type->storage_scope) == StorageScope::Create("local") ||
                 StorageScope::Create(ptr_type->storage_scope) ==
                     StorageScope::Create("wmma.matrix_a") ||
                 StorageScope::Create(ptr_type->storage_scope) ==
                     StorageScope::Create("wmma.matrix_b") ||
                 StorageScope::Create(ptr_type->storage_scope) ==
                     StorageScope::Create("wmma.accumulator")) {
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
    bool OnlyLocalAccesses() const {
      return !(bit_vector_[static_cast<int>(StorageType::kGlobal)] ||
               bit_vector_[static_cast<int>(StorageType::kShared)] ||
               bit_vector_[static_cast<int>(StorageType::kOthers)]) &&
             bit_vector_[static_cast<int>(StorageType::kLocal)];
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
  RWMode rw_mode_;
  AccessMarker read_marker_, write_marker_;
  std::multiset<const VarNode*> read_vars_, write_vars_;
  std::pair<AccessMarker, AccessMarker> Analyze(const Stmt& stmt) {
    VisitStmt(stmt);
    return std::make_pair(read_marker_, write_marker_);
  }

  friend class LocalPadder;
};

class LocalPadder : public StmtExprMutator {
 public:
  LocalPadder(std::multiset<const BlockRealizeNode*> blocks) : blocks_(blocks) {}

 private:
  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    if (is_one(op->predicate) || is_zero(op->predicate)) {
      return StmtExprMutator::VisitStmt_(op);
    }
    StorageAccessAnalyzer::AccessMarker read_marker, write_marker;
    std::tie(read_marker, write_marker) = StorageAccessAnalyzer().Analyze(op->block->body);

    // Remove and/or Inline the predicates, while preserving the correctness, where by "inline", we
    // refer to the following transformation:
    //
    //    if (predicate) A = ...;
    //    |
    //    A = predicate ? ... : init_constexpr;
    //
    // and by "correctness", we refer to:
    // - The padded value does not affect the computed results in the global memory.
    // - There is no out-of-boundary accesses.
    // - There is no race condition.

    if (!op->predicate.defined()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    // First, decompose the condition. Since a predicate is usually in the form of
    //
    //     a1 < c1 && a2 < c2 ...
    std::vector<PrimExpr> predicates = DecomposePredicate(op->predicate);

    std::vector<size_t> removable_indices;
    std::vector<size_t> inlinable_indices;
    std::vector<size_t> residual_indices;

    if (read_marker.NoAccesses() && write_marker.OnlyLocalAccesses()) {
      // No memory reads but only local writes, common pattern for initialization.
      //
      //    A_local[local_index] = 0.f;
      for (size_t i = 0; i < predicates.size(); ++i) {
        // In order to prove that `predicate` is directly removable, we have to show that if it is
        // evaluated to false, the same local memory location will be rejected by the one that is
        // structurally similar in the write-back block anyway.
        const PrimExpr& predicate = predicates[i];
      }
    }

    return StmtExprMutator::VisitStmt_(op);
  }
  std::vector<PrimExpr> DecomposePredicate(const PrimExpr& predicate) const {
    std::vector<PrimExpr> predicates;
    std::function<void(const PrimExpr&)> decompose;
    decompose = [&predicates, &decompose](const PrimExpr& cond) {
      if (const AndNode* op = cond.as<AndNode>()) {
        decompose(op->a);
        decompose(op->b);
      } else {
        predicates.push_back(cond);
      }
    };
    decompose(predicate);
    CHECK(!predicates.empty());
    return predicates;
  }

  std::multiset<const BlockRealizeNode*> blocks_;
  // Stmt VisitStmt_(const IfThenElseNode* op) final {
  //   if (!is_no_op(op->else_case)) {
  //     return StmtExprMutator::VisitStmt_(op);
  //   }
  //   // Analyze the reads and writes of the body statements.
  //   StorageAccessAnalyzer::AccessMarker read_marker, write_marker;
  //   std::tie(read_marker, write_marker) = StorageAccessAnalyzer().Analyze(op->then_case);

  //   // Remove and/or Inline the predicates, while preserving the correctness, where by "inline",
  //   we
  //   // refer to the following transformation:
  //   //
  //   //    if (predicate) A = ...;
  //   //    |
  //   //    A = predicate ? ... : init_constexpr;
  //   //
  //   // and by "correctness", we refer to:
  //   // - The padded value does not affect the computed results in the global memory.
  //   // - There is no out-of-boundary accesses.
  //   // - There is no race condition.

  //   // First, decompose the condition. Since a predicate is usually in the form of
  //   //
  //   //     a1 < c1 && a2 < c2 ...
  //   std::vector<PrimExpr> predicates = DecomposeCondition(op->condition);

  //   if (read_marker.NoAccesses() && write_marker.OnlyLocalAccesses()) {
  //     //

  //   } else if (read_marker.OnlyGlobalAccesses() && write_marker.OnlyLocalOrSharedAccesses()) {
  //     // if (!init_checker_.init_constexpr_) {
  //     //   return StmtExprMutator::VisitStmt_(op);
  //     // }
  //     // In the case when there are global buffer reads and local/shared buffer writes, inline
  //     the
  //     // predicates as part of the buffer store statements.
  //     // PredicateInliner predicate_inliner(op->then_case);
  //     predicate_inliner(op->condition);
  //     size_t predicate_stack_current_size = predicate_stack_.size();
  //     // Push the inlinable predicates on top of the stack.
  //     if (!predicate_inliner.inlinable_predicates_.empty()) {
  //       predicate_stack_.insert(predicate_stack_.end(),
  //                               predicate_inliner.inlinable_predicates_.begin(),
  //                               predicate_inliner.inlinable_predicates_.end());
  //     }

  //     enable_padding_ = true;
  //     // Update the body statements by inlining the predicates.
  //     Stmt inlined_body_stmt = VisitStmt(op->then_case);
  //     enable_padding_ = false;

  //     if (!predicate_inliner.inlinable_predicates_.empty()) {
  //       predicate_stack_.erase(predicate_stack_.begin() + predicate_stack_current_size,
  //                              predicate_stack_.end());
  //     }
  //     if (predicate_inliner.non_inlinable_residuals_.empty()) {
  //       return inlined_body_stmt;
  //     }
  //     return IfThenElse(FlattenPredicates(predicate_inliner.non_inlinable_residuals_),
  //                       inlined_body_stmt);
  //   } else if (read_marker.OnlyLocalOrSharedAccesses() &&
  //              write_marker.OnlyLocalOrSharedAccesses()) {
  //     // In the case when there are global buffer reads and local/shared buffer writes, remove
  //     the
  //     // predicates.
  //     return StmtExprMutator::VisitStmt(op->then_case);
  //   }
  //   return StmtExprMutator::VisitStmt_(op);
  // }
  // Stmt VisitStmt_(const BufferStoreNode* op) final {
  //   if (!enable_padding_ || predicate_stack_.empty()) {
  //     return StmtExprMutator::VisitStmt_(op);
  //   }
  //   PrimExpr store_predicate = FlattenPredicates(predicate_stack_);
  //   return BufferStore(op->buffer,
  //                      Select(store_predicate, op->value, ComposePaddedValue(op->value->dtype)),
  //                      op->indices);
  // }
  // std::vector<PrimExpr> DecomposeCondition(const PrimExpr& cond) const {
  //   std::vector<PrimExpr> predicates;
  //   std::function<void(const PrimExpr&)> visit_expr = [&predicates,
  //                                                      visit_expr](const PrimExpr& cond) {
  //     if (const AndNode* op = cond.as<AndNode>()) {
  //       visit_expr(op->a);
  //       visit_expr(op->b);
  //     }
  //     predicates.push_back(cond);
  //   };
  //   return predicates;
  // }
  // PrimExpr FlattenPredicates(const Array<PrimExpr>& predicates) const {
  //   CHECK(!predicates.empty());
  //   PrimExpr ret = predicates.front();
  //   for (auto predicates_it = predicates.begin() + 1; predicates_it != predicates.end();
  //        ++predicates_it) {
  //     ret = ret && (*predicates_it);
  //   }
  //   return ret;
  // }

  // bool init_verified_by_outer_stmts_ = false;
  // std::vector<PrimExpr> predicate_stack_;
  // bool enable_padding_ = false;
};

Stmt LocalPadTransform(Stmt stmt) {
  // Record all the blocks, used for tracing producer-consumer relationship.
  std::multiset<const BlockRealizeNode*> blocks;
  PostOrderVisit(stmt, [&blocks](const ObjectRef& obj_ref) {
    if (const BlockRealizeNode* op = obj_ref.as<BlockRealizeNode>()) {
      blocks.insert(op);
    }
  });
  LocalPadder local_padder(std::move(blocks));
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
