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
 * \file auto_scheduler/utils.cc
 * \brief Common utilities.
 */

#include "utils.h"

#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/auto_scheduler/search_task.h>

namespace tvm {
namespace auto_scheduler {

NullStream& NullStream::Global() {
  static NullStream stream;
  return stream;
}

int64_t FlopEstimator::AxisLengthProd(const Array<tir::IterVar>& axes) {
  int64_t ret = 1.0;
  for (const auto& x : axes) {
    if (const IntImmNode* imm = x->dom->extent.as<IntImmNode>()) {
      ret *= imm->value;
    } else {
      if (const IntImmNode* const imm =
            analyzer_.Simplify(replacer_(x->dom->extent)).as<IntImmNode>()) {
        ret *= imm->value;
      } else {
        return -1.0;
      }
    }
  }
  return ret;
}

double FlopEstimator::EstimateFlop(const Array<te::Operation>& ops) {
  double ret = 0;
  for (const auto& op : ops) {
    if (auto pop = op.as<te::ComputeOpNode>()) {
      if (pop->attrs.count("FLOP")) {
        // Use user-provided FLOP
        auto pint = pop->attrs["FLOP"].as<IntImmNode>();
        ICHECK(pint != nullptr);
        ret += pint->value;
      } else {
        // Estimate by parsing the compute body
        double num_element = AxisLengthProd(pop->axis);
        if (num_element == -1) {
          fail_ = true;
          break;
        }
        cur_type_code_ = pop->output_dtype(0).code();
        double op_per_element = 0;
        for (const auto& x : pop->body) {
          op_per_element += VisitExpr(x);
        }
        ret += num_element * op_per_element;
      }
    } else if (op->IsInstance<te::PlaceholderOpNode>()) {
      {}  // do nothing
    } else {
      LOG(FATAL) << "Invalid op type " << op;
    }
  }

  return fail_ ? -1 : ret;
}

double EstimateFlopsForWklInst(const ComputeDAG& compute_dag,
                               const Array<Var>& shape_vars,
                               const Array<IntImm>& wkl_insts) {
  CHECK(shape_vars.size() == wkl_insts.size());
  Map<String, IntImm> shape_var_value_map;
  for (size_t i = 0; i < shape_vars.size(); ++i) {
    shape_var_value_map.Set(shape_vars[i]->name_hint, wkl_insts[i]);
  }
  DynShapeVarReplacer replacer(
      [&shape_var_value_map](const VarNode* op) -> PrimExpr {
        auto shape_var_value_map_iter =
            shape_var_value_map.find(op->name_hint);
        if (shape_var_value_map_iter != shape_var_value_map.end()) {
          return (*shape_var_value_map_iter).second;
        }
        LOG(FATAL) << "Dynamic Axis Node " << GetRef<Var>(op)
                   << " has not been found in "
                   << shape_var_value_map;
        return GetRef<Var>(op);
      }
      );
  te::Schedule sch;
  Array<te::Tensor> tensors;
  std::tie(sch, tensors) = compute_dag.ApplySteps({});
  Array<te::Operation> sch_ops;
  for (const te::Stage& stage : sch->stages) {
    sch_ops.push_back(stage->op);
  }
  return FlopEstimator(replacer).EstimateFlop(sch_ops);
}

std::unordered_map<size_t, size_t>
TopKDispatcher::Dispatch(const std::vector<float>& scores,
                         const size_t num_states) {
  const size_t num_wkl_insts = scores.size() / num_states;
  std::unordered_map<size_t, size_t> disp_map_to_ret;

  size_t k = 0;
  // instance dispatch initial status
  std::unordered_set<size_t> inst_disp_init_remainder;

  for (size_t inst_id = 0; inst_id < num_wkl_insts; ++inst_id) {
    inst_disp_init_remainder.insert(inst_id);
  }

  struct ScoreboardItem {
    size_t state_id;
    float  score;
  };

  auto scoreboard_item_gt_cmp = [](const ScoreboardItem& LHS,
                                   const ScoreboardItem& RHS) {
    return LHS.score > RHS.score;
  };

  while (true) {
    ++k;  // increment the number of candidates selected
    disp_map_to_ret.clear();

    typedef std::vector<ScoreboardItem> Scoreboard;
    std::unordered_map<size_t, Scoreboard> inst_topK_candidates;

    for (size_t inst_id = 0; inst_id < num_wkl_insts; ++inst_id) {
      // ∀inst, pick its k most-preferred states
      Scoreboard& topK_candidates = inst_topK_candidates[inst_id];
      for (size_t state_id = 0; state_id < num_states; ++state_id) {
        const float score = scores[inst_id * num_states + state_id];
        if (topK_candidates.size() < k) {
          topK_candidates.push_back(ScoreboardItem{state_id, score});
        } else {
          if (score > topK_candidates.front().score) {
            std::pop_heap(topK_candidates.begin(), topK_candidates.end(),
                          scoreboard_item_gt_cmp);
            topK_candidates.back() = ScoreboardItem{state_id, score};
            std::push_heap(topK_candidates.begin(), topK_candidates.end(),
                           scoreboard_item_gt_cmp);
          }  // if (score > topK_candidates.front().score)
        }    // if (topK_candidates.size() < k)
      }      // for (state_id ∈ range(num_states))
    }        // for (inst_id ∈ range(num_wkl_insts))
    // Now that all the instances have their most-preferred states ready, we now
    // choose the minimum set that could cover all the candidates.

    // make a copy of the dispatch status
    std::unordered_set<size_t> inst_disp_remainder(inst_disp_init_remainder);
    std::unordered_set<size_t> selected_states;

    // keep iterating until all the iterators have been dispatched
    while (!inst_disp_remainder.empty()) {
      // count the number of votes per state [state_id → vote_cnt]
      std::unordered_map<size_t, float> votes;

      for (const size_t inst_id : inst_disp_remainder) {
        for (const ScoreboardItem& cand : inst_topK_candidates[inst_id]) {
          auto votes_it = votes.find(cand.state_id);
          if (votes_it == votes.end()) {
            votes[cand.state_id] = 0.;
          }
          votes[cand.state_id] += scores[inst_id * num_states + cand.state_id];
        }  // for (cand ∈ inst_topK_candidates[inst_id])
      }    // for (inst_id ∈ inst_disp_remainder)

      // pick the state_id with the maximum accumulated score
      const auto& votes_max_it = 
          std::max_element(votes.begin(), votes.end(),
                           [](const std::pair<size_t, float>& LHS,
                              const std::pair<size_t, float>& RHS)
                             -> bool {
                             return LHS.second < RHS.second;
                           });
      selected_states.insert(votes_max_it->first);

      std::unordered_set<size_t> inst_disp_remainder_copy = inst_disp_remainder;

      for (const size_t inst_id : inst_disp_remainder) {
        Scoreboard& topK_candidates = inst_topK_candidates[inst_id];

        Scoreboard::iterator topK_candidates_it;
        for (topK_candidates_it  = topK_candidates.begin();
             topK_candidates_it != topK_candidates.end();
             ++topK_candidates_it) {
          if (topK_candidates_it->state_id == votes_max_it->first) {
            break;
          }
        }
        if (topK_candidates_it != topK_candidates.end()) {
          inst_disp_remainder_copy.erase(inst_id);
          disp_map_to_ret[inst_id] = votes_max_it->first;
        }
      }    // for (inst_id ∈ inst_disp_remainder)
      inst_disp_remainder = std::move(inst_disp_remainder_copy);
    }  // while (!inst_disp_remainder.empty())

    if (selected_states.size() > max_num_states_) {
      LOG(WARNING) << "The number of selected states is greater than "
                   << max_num_states_ << ", hence is not valid";
    } else {
      break;
    }
  }
  LOG(INFO) << "k=" << k;
  return disp_map_to_ret;
}



std::tuple<std::unordered_map<size_t, size_t>,
           std::vector<State>,
           std::vector<float>,
           std::vector<float>>
TopKDispatcher::MapWklInstsToStates(const std::unordered_map<size_t, size_t>& raw_inst_id_disp_map,
                                    const std::vector<State>& candidate_states,
                                    const std::vector<float>& candidate_flops,
                                    const Array<Array<IntImm>>& wkl_insts,
                                    const std::vector<float>& adapted_candidate_flops) {
  std::vector<size_t> selected_candidate_state_ids;
  std::vector<float>  inst_predicted_flops(wkl_insts.size());
  std::unordered_map<size_t, size_t> inst_id_disp_map;

  // gather all the non-duplicate state_ids
  for (const std::pair<size_t, size_t>& inst_state_pair : raw_inst_id_disp_map) {
    auto iter = std::find(selected_candidate_state_ids.begin(),
                          selected_candidate_state_ids.end(),
                          inst_state_pair.second);
    if (iter != selected_candidate_state_ids.end()) {
      inst_id_disp_map[inst_state_pair.first] =
          std::distance(selected_candidate_state_ids.begin(), iter);
    } else {
      inst_id_disp_map[inst_state_pair.first] =
          selected_candidate_state_ids.size();
      selected_candidate_state_ids.push_back(inst_state_pair.second);
    }
    CHECK(inst_state_pair.first < wkl_insts.size());
    inst_predicted_flops[inst_state_pair.first] =
        adapted_candidate_flops[
          inst_state_pair.first * candidate_states.size() +
          inst_state_pair.second];
  }
  std::vector<State> selected_candidate_states;
  std::vector<float> selected_candidate_flops;

  for (const size_t state_id : selected_candidate_state_ids) {
    selected_candidate_states.push_back(candidate_states[state_id]);
    selected_candidate_flops .push_back(candidate_flops [state_id]);
  }

  return std::make_tuple(inst_id_disp_map,
                         selected_candidate_states,
                         selected_candidate_flops,
                         inst_predicted_flops);
}

}  // namespace auto_scheduler
}  // namespace tvm
