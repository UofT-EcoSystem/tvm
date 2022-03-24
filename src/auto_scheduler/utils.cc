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

}  // namespace auto_scheduler
}  // namespace tvm
