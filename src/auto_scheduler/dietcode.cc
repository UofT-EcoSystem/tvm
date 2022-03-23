#include <tvm/auto_scheduler/dietcode.h>

#include <tvm/driver/driver_api.h>
#include <tvm/ir/function.h>
#include <tvm/tir/transform.h>

#include "./utils.h"


namespace tvm {
namespace auto_scheduler {

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DietCodeDispatcherNode>(
      [](const ObjectRef& ref, ReprPrinter* p) {
        auto* node = static_cast<const DietCodeDispatcherNode*>(ref.get());
        p->stream << "DietCodeDispatcher("
                    << node->search_task << ", "
                    << node->GetStates() << ", "
                    << node->GetWklInstDispMap()
                  << ")";
      }
    );


Map<Integer, Integer> DietCodeDispatcherNode::GetWklInstDispMap() const {
  Map<Integer, Integer> wkl_inst_disp_map_;
  for (const auto& kv : wkl_inst_disp_map) {
    wkl_inst_disp_map_.Set(kv.first, kv.second);
  }
  return wkl_inst_disp_map_;
}

DietCodeDispatcher::DietCodeDispatcher(
    const SearchTask& search_task, std::vector<State>&& states,
    std::unordered_map<size_t, size_t>&& wkl_inst_disp_map) {
  ObjectPtr<DietCodeDispatcherNode> node = make_object<DietCodeDispatcherNode>();
  node->search_task = search_task;
  node->states = std::move(states);
  node->wkl_inst_disp_map = std::move(wkl_inst_disp_map);
  data_ = std::move(node);
}

std::pair<te::Schedule, Array<te::Tensor>>
DietCodeDispatcherNode::DispatchAndApplySteps(const int wkl_inst_id) const {
  const State& state = Dispatch(wkl_inst_id);
  return search_task->compute_dag.InstantiateAndApplySteps(
           state, search_task->shape_vars.value(),
           ToPrimExprArray(search_task->wkl_insts[wkl_inst_id])
         );
}

State DietCodeDispatcherNode::Dispatch(const int wkl_inst_id) const {
  return states[wkl_inst_disp_map.at(wkl_inst_id)];
}

TVM_REGISTER_GLOBAL("auto_scheduler.DispatcherStates")
    .set_body_typed([](const DietCodeDispatcher& dispatcher) {
      return dispatcher->GetStates();
    });


TVM_REGISTER_GLOBAL("auto_scheduler.DispatcherInstDispMap")
    .set_body_typed([](const DietCodeDispatcher& dispatcher) {
      return dispatcher->GetWklInstDispMap();
    });

}  // namespace auto_scheduler
}  // namespace tvm
