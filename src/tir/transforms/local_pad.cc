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

/*!
 * \file local_pad.cc
 */

#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <utility>


namespace tvm {
namespace tir {
namespace {

/**
 * \brief  Verify that all local variables are initialized to the same value.
 */
class LocalPadInitChecker : public StmtVisitor {
 private:
  void VisitStmt_(const BufferStoreNode* op) final {
    if (!inside_init_block_) {
      return StmtVisitor::VisitStmt_(op);
    }
    const PrimExpr& rhs = op->value;
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
    // detect the presence of an initialization block
    if (op->name_hint == "update_init") {
      inside_init_block_ = true;
      StmtVisitor::VisitStmt_(op);
      inside_init_block_ = false;
      return;
    }
    return StmtVisitor::VisitStmt_(op);
  }
 public:
  PrimExpr initWithSingleConstExpr() const {
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

}  // anonymous namespace

static Stmt LocalPad(Stmt stmt) {
  LocalPadInitChecker init_checker;
  init_checker(stmt);
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
}  // namespace tvm
