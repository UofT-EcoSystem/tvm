#pragma once

#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace arith {
class IntSet;
}  // namespace arith

namespace tir {

class DynShapeVarReplacer : public StmtExprMutator {
 private:
  std::function<PrimExpr(const VarNode*)> freplace_expr_;
 protected:
  PrimExpr VisitExpr_(const VarNode* op) final {
    if (freplace_expr_ == nullptr) {
      return StmtExprMutator::VisitExpr_(op);
    } else {
      return freplace_expr_(op);
    }
  }
 public:
  explicit DynShapeVarReplacer(std::function<PrimExpr(const VarNode*)> freplace_expr = nullptr)
      : freplace_expr_(freplace_expr) {}
};

arith::IntSet EvaluateRangeForAllWklInsts(const PrimExpr& expr, const Array<Var>& shape_vars,
                                          const Array<Array<IntImm>>& wkl_insts);

}  // namespace tir
}  // namespace tvm
