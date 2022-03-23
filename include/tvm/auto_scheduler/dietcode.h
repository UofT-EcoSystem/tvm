#pragma once

#include <tvm/auto_scheduler/search_task.h>
#include <tvm/runtime/object.h>
#include <tvm/ir/expr.h>


namespace tvm {
namespace auto_scheduler {

struct DietCodeDispatcherNode : Object {
  SearchTask search_task;
  std::vector<State> states;
  std::unordered_map<size_t, size_t> wkl_inst_disp_map;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("search_task", &search_task);
  }
  
  Array<State> GetStates() const { return Array<State>(states); }
  Map<Integer, Integer> GetWklInstDispMap() const;

  /**
   * @brief   Dispatch to an auto-scheduler state based on the given workload
   *          instance index and apply the transformation steps.
   * @return  A pair, 1st is the generated schedule, 2nd is the tensors
   * @sa      Dispatch
   */
  std::pair<te::Schedule, Array<te::Tensor>> DispatchAndApplySteps(const int wkl_inst_idx) const;
  /**
   * @brief   Dispatch to an auto-scheduler state based on the given workload
   *          instance index.
   * @return  The state to be dispatched to
   * @sa      DispatchAndApplySteps
   */
  State Dispatch(const int wkl_inst_idx) const;

  static constexpr const char* _type_key = "auto_scheduler.DietCodeDispatcher";
  TVM_DECLARE_FINAL_OBJECT_INFO(DietCodeDispatcherNode, Object);
};

class DietCodeDispatcher : public ObjectRef {
 public:
  /**
   * @brief DietCode Dispatcher that dispatches workload instances to
   *        auto-scheduler states
   * @param search_task        dynamic search task
   * @param states             auto-scheduler states
   * @param wkl_inst_disp_map  index mapping from auto-scheduler states to
   *                           workload instances
   */
  DietCodeDispatcher(const SearchTask& search_task, std::vector<State>&& states,
                     std::unordered_map<size_t, size_t>&& wkl_inst_disp_map);
  TVM_DEFINE_OBJECT_REF_METHODS(DietCodeDispatcher, ObjectRef,
                                DietCodeDispatcherNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DietCodeDispatcherNode);
};

}  // namespace auto_scheduler
}  // namespace tvm
