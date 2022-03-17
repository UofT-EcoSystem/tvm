#include <tvm/arith/analyzer.h>

#include "./dyn_shape_var.h"

namespace tvm {
namespace tir {

arith::IntSet EvaluateRangeForAllWklInsts(
    const PrimExpr& expr, const Array<Var>& shape_vars,
    const Array<Array<IntImm>>& wkl_insts) {
  int min = std::numeric_limits<int>().max(),
      max = std::numeric_limits<int>().min();
  arith::Analyzer analyzer;
  for (const Array<IntImm>& wkl_inst : wkl_insts) {
    DynShapeVarReplacer dyn_shape_var_replacer(
        [&shape_vars, &wkl_inst](const VarNode* op) -> PrimExpr {
          for (size_t i = 0; i < shape_vars.size(); ++i) {
            if (shape_vars[i]->name_hint == op->name_hint) {
              return wkl_inst[i];
            }
          }
          LOG(FATAL) << "DynShapeVar=" << GetRef<Var>(op)
                     << " has not been found in shape_vars";
          return GetRef<Var>(op);
        });
    const IntImmNode* const val =
        analyzer.Simplify(dyn_shape_var_replacer(expr)).as<IntImmNode>();
    CHECK(val != nullptr);
    min = val->value < min ? val->value : min;
    max = val->value > max ? val->value : max;
  }
  return arith::IntSet::Interval(Integer(min), Integer(max));
}

}  // namespace tir
}  // namespace tvm
