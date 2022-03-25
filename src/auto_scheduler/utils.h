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
 * \file auto_scheduler/utils.h
 * \brief Common utilities.
 */

#ifndef TVM_AUTO_SCHEDULER_UTILS_H_
#define TVM_AUTO_SCHEDULER_UTILS_H_

#include <dmlc/common.h>
#include <tvm/auto_scheduler/loop_state.h>
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>

#include "../tir/ir/dyn_shape_var.h"

#include <algorithm>
#include <deque>
#include <exception>
#include <future>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace std {

/*! \brief Hash function for std::pair */
template <typename T1, typename T2>
struct hash<std::pair<T1, T2>> {
  std::size_t operator()(const std::pair<T1, T2>& k) const {
    return ::dmlc::HashCombine(std::hash<T1>()(k.first), std::hash<T2>()(k.second));
  }
};

/*! \brief Hash function for std::tuple */
template <typename T1, typename T2, typename T3>
struct hash<std::tuple<T1, T2, T3>> {
  std::size_t operator()(const std::tuple<T1, T2, T3>& k) const {
    return ::dmlc::HashCombine(
        ::dmlc::HashCombine(std::hash<T1>()(std::get<0>(k)), std::hash<T2>()(std::get<1>(k))),
        std::hash<T3>()(std::get<2>(k)));
  }
};

}  // namespace std

namespace tvm {

inline size_t floor_by(const size_t a, const size_t b) {
  CHECK(b != 0);
  return (a + b - 1) / b * b;
}

inline size_t floor_div(const size_t a, const size_t b) {
  CHECK(b != 0);
  return (a + b - 1) / b;
}

namespace auto_scheduler {

inline size_t ceil_by(const size_t a, const size_t b) {
  CHECK(b != 0);
  return (a + b - 1) / b * b;
}

inline size_t ceil_div(const size_t a, const size_t b) {
  CHECK(b != 0);
  return (a + b - 1) / b;
}

/********** Utilities for Array, std::vector, std::string **********/
/*! \brief Get the first appearance index of elements in an Array */
template <typename T>
inline void GetIndices(const Array<T>& array, const Array<T>& to_locate, Array<Integer>* indices) {
  for (const auto& v : to_locate) {
    auto it = std::find(array.begin(), array.end(), v);
    if (it != array.end()) {
      indices->push_back(it - array.begin());
    } else {
      LOG(FATAL) << "Cannot find the item";
    }
  }
}

/*! \brief Get the first appearance index of an element in an Array */
template <typename T>
inline int GetIndex(const Array<T>& array, const T& to_locate) {
  for (size_t i = 0; i < array.size(); ++i) {
    if (array[i] == to_locate) {
      return i;
    }
  }
  LOG(FATAL) << "Cannot find the item";
  return -1;
}

/*! \brief Delete the item in a std::vector if it exists. */
template <typename T>
inline void FindAndDeleteItem(std::vector<T>* array, const T& to_delete) {
  auto iter = std::find(array->begin(), array->end(), to_delete);
  if (iter != array->end()) {
    array->erase(iter);
  }
}

/*! \brief Compute the product of all elements in a vector */
inline int64_t ElementProduct(const std::vector<int>& array) {
  int64_t ret = 1;
  for (auto x : array) {
    ret *= x;
  }
  return ret;
}

template<typename T>
inline std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec) {
  out << "[";
  for (const T& i : vec) {
    out << i << ", ";
  }
  out << "]";
  return out;
}

template<typename T>
inline std::ostream& operator<<(std::ostream& out, const std::vector<std::vector<T>>& mat) {
  out << "[";
  out << std::endl;
  for (const auto& vec : mat) {
    out << "  " << vec << std::endl;
  }
  out << "]";
  return out;
}

template<typename K, typename V, typename H, typename E>
inline std::ostream& operator<<(std::ostream& out, const std::unordered_map<K, V, H, E>& map) {
  out << "{" << std::endl;
  for (const auto& kv : map) {
    out << "  " << kv.first << " : " << kv.second << std::endl;
  }
  out << "}";
  return out;
}

template<typename T>
inline Array<PrimExpr> ToPrimExprArray(const Array<T>& a) {
  Array<PrimExpr> exprs;
  for (const T& t : a) {
    exprs.push_back(t);
  }
  return exprs;
}

template<typename T>
struct VectorStripper {
  typedef T type;
};

template<typename T>
struct VectorStripper<std::vector<T>> {
  typedef typename VectorStripper<T>::type type;
};

template<typename T>
inline std::tuple<std::vector<typename VectorStripper<T>::type>,
                  size_t,
                  std::vector<size_t>>
UnpackVector(const std::vector<T>& vec) {
  return {vec, vec.size(), {vec.size()}};
}

template<typename T>
inline std::tuple<std::vector<typename VectorStripper<T>::type>,
                  size_t,
                  std::vector<size_t>>
UnpackVector(const std::vector<std::vector<T>>& vec) {
  CHECK(vec.size() > 0);
  std::vector<typename VectorStripper<T>::type> flattened_subvec0,
                                                flattened_subvec;
  size_t subvec0_size, subvec_size;
  std::vector<size_t> subvec0_shape, subvec_shape;

  std::tie(flattened_subvec0, subvec0_size, subvec0_shape) = UnpackVector(vec[0]);

  for (size_t i = 1; i < vec.size(); ++i) {
    std::tie(flattened_subvec, subvec_size, subvec_shape) = UnpackVector(vec[i]);
    CHECK(subvec0_shape.size() == subvec_shape.size());
    for (size_t j = 0; j < subvec_shape.size(); ++j) {
      CHECK(subvec0_shape[j] == subvec_shape[j]);
    }
    flattened_subvec0.insert(flattened_subvec0.end(), flattened_subvec.begin(),
                             flattened_subvec.end());
    subvec0_size += subvec_size;
  }
  subvec0_shape.insert(subvec0_shape.begin(), vec.size());
  return {flattened_subvec0, subvec0_size, subvec0_shape};
}

template<typename T>
inline std::vector<int64_t> ToInt64(const std::vector<T>& vec) {
  std::vector<int64_t> int64_vec(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    int64_vec[i] = vec[i];
  }
  return int64_vec;
}

using ::tvm::runtime::NDArray;

template<typename T>
inline NDArray ToNDArray(const std::vector<T>& vec, std::vector<size_t> shape = {},
                         const DataType data_type = DataType::Float(32)) {
  std::vector<typename VectorStripper<T>::type> flattened_vec;
  size_t vec_size;
  std::vector<size_t> vec_shape;

  std::tie(flattened_vec, vec_size, vec_shape) = UnpackVector(vec);
  if (shape.empty()) {
    shape = vec_shape;
  }
  NDArray ret = NDArray::Empty(ToInt64(shape), data_type, Device{kDLCPU, 0});
  ret.CopyFromBytes(flattened_vec.data(), sizeof(int) * vec_size);
  return ret;
}

/*! \brief Move elements from multiple vectors to one vector */
template <typename T>
std::vector<T>& ConcatenateMove(std::vector<T>* out, std::vector<T>* in) {
  out->insert(out->end(), std::make_move_iterator(in->begin()), std::make_move_iterator(in->end()));
  return *out;
}

/*! \brief Move elements from multiple vectors to one vector */
template <typename T, typename... Args>
std::vector<T>& ConcatenateMove(std::vector<T>* out, std::vector<T>* first, Args... args) {
  ConcatenateMove(out, first);
  ConcatenateMove(out, args...);
  return *out;
}

/*! \brief Get a random permutation of integers [0, n-1] */
template <typename G>
void RandomPermutation(int n, std::vector<int>* out, G* gen) {
  out->assign(n, 0);
  std::iota(out->begin(), out->end(), 0);
  std::shuffle(out->begin(), out->end(), *gen);
}

/*! \brief Replace a sub-string to another sub-string in a string */
inline void StrReplace(std::string* base, const std::string& from, const std::string& to) {
  auto pos = base->find(from);
  while (pos != std::string::npos) {
    base->replace(pos, from.size(), to);
    pos = base->find(from, pos + to.size());
  }
}

/*! \brief Return whether two int arrays are elementwise-equal */
inline bool IntArrayEqual(const Array<PrimExpr>& arr1, const Array<PrimExpr>& arr2) {
  if (arr1.size() != arr2.size()) {
    return false;
  }

  for (size_t i = 0; i < arr1.size(); ++i) {
    std::ostringstream lhs_strout, rhs_strout;
    lhs_strout << arr1[i];
    rhs_strout << arr2[i];
    if (lhs_strout.str() != rhs_strout.str()) {
      return false;
    }
  }
  return true;
}

/********** Utilities for TVM Containers / ByteArray **********/
/*! \brief Compute mean of a FloatImm array */
inline double FloatArrayMean(const Array<PrimExpr>& float_array) {
  double sum = 0;
  if (float_array.empty()) {
    return 0.0;
  }
  std::vector<float> float_vector;

  for (const auto& x : float_array) {
    auto floatimm = x.as<tir::FloatImmNode>();
    ICHECK(floatimm != nullptr);
    sum += floatimm->value;
    float_vector.push_back(floatimm->value);
  }
  if (float_vector[0] < float_vector[float_vector.size() - 1] * 1.1) {
    return sum / float_array.size();
  } else {
    return float_vector[float_vector.size() - 1];
  }
}

/*! \brief Return whether a string starts with another substring */
inline bool StrStartsWith(const String& a, const String& b) {
  if (b.size() > a.size()) return false;
  return std::equal(a.c_str(), a.c_str() + b.size(), b.c_str());
}

/*! \brief Return whether a string ends with another substring */
inline bool StrEndsWith(const String& a, const String& b) {
  if (b.size() > a.size()) return false;
  return std::equal(a.c_str() + a.size() - b.size(), a.c_str() + a.size(), b.c_str());
}

/********** Other Utilities **********/
/*! \brief Get an int value from an Expr */
inline int64_t GetIntImm(const PrimExpr& expr) {
  auto pint = expr.as<IntImmNode>();
  ICHECK(pint != nullptr) << "Expect an IntImm but get " << expr;
  return pint->value;
}

/*! \brief Compute the product of the lengths of axes */
inline int64_t AxisLengthProd(const Array<tir::IterVar>& axes) {
  int64_t ret = 1.0;
  for (const auto& x : axes) {
    if (const IntImmNode* imm = x->dom->extent.as<IntImmNode>()) {
      ret *= imm->value;
    } else {
      return -1.0;
    }
  }
  return ret;
}

/*!
 * \brief Clean the name of an iterator or an op to make it valid in python code.
 * \param str The original name.
 * \param prefix The name prefix to differentiate the same name (e.g., the same iterator names).
 * \return The cleaned name.
 */
inline std::string CleanName(const std::string& str, const std::string& prefix = "") {
  std::string ret = str;
  StrReplace(&ret, ".", "_");
  StrReplace(&ret, "@", "_");
  StrReplace(&ret, "outer", "o");
  StrReplace(&ret, "inner", "i");
  if (prefix != "") {
    return prefix + "_" + ret;
  }
  return ret;
}

/*! \brief An empty output stream */
class NullStream : public std::ostream {
 public:
  NullStream() : std::ostream(nullptr) {}
  NullStream(const NullStream&) : std::ostream(nullptr) {}
  static NullStream& Global();
};

template <class T>
NullStream& operator<<(NullStream& os, const T& value) {
  return os;
}

/*! \brief Get std cout with verbose control */
inline std::ostream& StdCout(int verbose, int setting = 1) {
  return verbose >= setting ? std::cout : NullStream::Global();
}

/*! \brief Print multiple chars */
inline std::string Chars(const char& str, int times) {
  std::stringstream ret;
  for (int i = 0; i < times; ++i) {
    ret << str;
  }
  return ret.str();
}

/*! \brief Print the time elapsed */
inline void PrintTimeElapsed(std::chrono::time_point<std::chrono::high_resolution_clock> t_begin,
                             const std::string& info, int verbose) {
  double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now() - t_begin)
                        .count();
  StdCout(verbose) << "Time elapsed for " << info << ": " << std::fixed << std::setprecision(2)
                   << duration << " s" << std::endl;
}

/*!
 * \brief Parse shape and axis names from layout string
 */
inline void ParseKernelLayout(const String& layout, Array<PrimExpr>* shape,
                              std::vector<std::string>* axes) {
  int32_t factor = 0;
  std::string axis = "";
  for (char c : std::string(layout)) {
    if (c >= 'A' && c <= 'z') {
      axis += c;
      if (factor != 0) {
        shape->push_back(factor);
        factor = 0;
      }
    } else if (c >= '0' && c <= '9') {
      factor = factor * 10 + c - '0';
      if (!axis.empty()) {
        axes->push_back(axis);
        axis = "";
      }
    } else {
      LOG(FATAL) << "Invalid layout " << layout;
    }
  }
  if (!axis.empty()) {
    axes->push_back(axis);
  }
}

/*! \brief Get the base name before '_' of an axis */
inline std::string AxisBaseName(const std::string& str) { return str.substr(0, str.rfind("_")); }

class SyntheticExprReplacer : public StmtExprMutator {
 private:
  Map<ObjectRef, PrimExpr> expr_subst_map_;

  PrimExpr VisitExpr_(const ProducerLoadNode* op) override {
    auto producer_subst_map_it = producer_subst_map.find(op->producer);
    if (producer_subst_map_it != producer_subst_map.end()) {
      return ProducerLoad((*producer_subst_map_it).second,
                          op->indices);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

 public:
  Map<DataProducer, te::Tensor> producer_subst_map;

  SyntheticExprReplacer(const Map<ObjectRef, PrimExpr>& expr_subst_map)
      : expr_subst_map_(expr_subst_map) {}

  PrimExpr VisitExpr_(const VarNode* op) override final {
    auto expr_subst_map_it = expr_subst_map_.find(op->name_hint);
    if (expr_subst_map_it != expr_subst_map_.end()) {
      return (*expr_subst_map_it).second;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr(const PrimExpr& expr) override final {
    auto expr_subst_map_it = expr_subst_map_.find(expr);
    if (expr_subst_map_it != expr_subst_map_.end()) {
      return (*expr_subst_map_it).second;
    }
    std::ostringstream strout;
    strout << expr;
    std::string expr_str = strout.str();
    for (const std::pair<ObjectRef, PrimExpr>& kv : expr_subst_map_) {
      strout.str("");
      strout.clear();
      strout << kv.first;
      std::string k_str = strout.str();
      if (expr_str == k_str) {
        return kv.second;
      }
    }
    return StmtExprMutator::VisitExpr(expr);
  }
};

using te::Operation;
using namespace ::tvm::tir;

class FlopEstimator : public ExprFunctor<double(const PrimExpr& n)> {
 public:
  FlopEstimator(const DynShapeVarReplacer& replacer = DynShapeVarReplacer(nullptr))
      : replacer_(replacer) {}

 private:
  int64_t AxisLengthProd(const Array<tir::IterVar>& axes);

 public:
  double EstimateFlop(const Array<te::Operation>& ops);

  double VisitExpr_(const ReduceNode* op) final {
    uint64_t num_iter = 1;
    for (const auto& x : op->axis) {
      if (auto imm = x->dom->extent.as<IntImmNode>()) {
        num_iter *= imm->value;
      } else {
        if (const IntImmNode* const imm =
              analyzer_.Simplify(replacer_(x->dom->extent)).as<IntImmNode>()) {
          num_iter *= imm->value;
        } else {
          fail_ = true;
          num_iter = -1;
        }
      }
    }
    double body_flop = 0;
    for (size_t i = 0; i < op->combiner->result.size(); ++i) {
      body_flop += VisitExpr(op->combiner->result[i]);
      body_flop += VisitExpr(op->source[i]);
    }
    return num_iter * body_flop;
  }

  double VisitExpr_(const FloatImmNode* op) final { return 0.0; }
  double VisitExpr_(const IntImmNode* op) final { return 0.0; }
  double VisitExpr_(const ProducerLoadNode* op) final { return 0.0; }

  double VisitExpr_(const CastNode* op) final { return VisitExpr(op->value); }
  double VisitExpr_(const VarNode* op) final { return 0.0; }

  double VisitExpr_(const SelectNode* op) final {
    return VisitExpr(op->condition) +
           std::max(VisitExpr(op->true_value), VisitExpr(op->false_value));
  }

// Index calculations (e.g., the "i + j" expression in A[i + j]) are not counted in FLOPS.
#define VisitBinary(Node)                                                                     \
  double VisitExpr_(const Node* op) final {                                                   \
    double base = 1.0;                                                                        \
    if ((op->a->dtype.code() != cur_type_code_) && (op->b->dtype.code() != cur_type_code_)) { \
      base = 0.0;                                                                             \
    }                                                                                         \
    return base + VisitExpr(op->a) + VisitExpr(op->b);                                        \
  }

#define VisitUnary(Node)                                          \
  double VisitExpr_(const Node* op) final {                       \
    double base = op->dtype.code() == cur_type_code_ ? 1.0 : 0.0; \
    return base + VisitExpr(op->a);                               \
  }

  VisitBinary(AddNode);
  VisitBinary(SubNode);
  VisitBinary(MulNode);
  VisitBinary(DivNode);
  VisitBinary(ModNode);
  VisitBinary(FloorDivNode);
  VisitBinary(FloorModNode);
  VisitBinary(MaxNode);
  VisitBinary(MinNode);
  VisitBinary(EQNode);
  VisitBinary(NENode);
  VisitBinary(LTNode);
  VisitBinary(LENode);
  VisitBinary(GTNode);
  VisitBinary(GENode);
  VisitBinary(AndNode);
  VisitBinary(OrNode);
  VisitUnary(NotNode);

#undef VisitBinary
#undef VisitUnary

  double VisitExpr_(const CallNode* op) final {
    double ret = 0.0;
    for (const auto& x : op->args) {
      ret += VisitExpr(x);
    }
    return ret;
  }

  double VisitExprDefault_(const Object* op) final {
    fail_ = true;
    return -1.0;
  }

 private:
  arith::Analyzer analyzer_;
  DynShapeVarReplacer replacer_;
  bool fail_{false};
  int cur_type_code_;
};

struct TopKDispatcher {
 private:
  size_t max_num_states_;
 public:
  TopKDispatcher(const size_t max_num_states = 128)
      : max_num_states_(max_num_states) {}
  /**
   * @brief  Dispatch to workload instances according to adapted scores.
   */
  std::unordered_map<size_t, size_t>
  Dispatch(const std::vector<float>& scores, const size_t num_states);
  /**
   * @brief   Map workload instances to states.
   * @param   raw_wkl_inst_id_disp_maps
   * @param   candidate_states
   * @param   candidate_flops
   * @param   wkl_insts
   * @param   adapted_candidate_flops
   * @return  A tuple, 1st is the generated index mapping from workload
   *          instances to states, 2nd is the selected states, 3rd is the
   *          compute throughputs of those states, 4th is the 
   */
  std::tuple<std::unordered_map<size_t, size_t>,
             std::vector<State>,
             std::vector<float>,
             std::vector<float>>
  MapWklInstsToStates(const std::unordered_map<size_t, size_t>& raw_wkl_inst_id_disp_map,
                      const std::vector<State>& candidate_states,
                      const std::vector<float>& candidate_flops,
                      const Array<Array<IntImm>>& wkl_insts,
                      const std::vector<float>& adapted_candidate_flops);
};

class SearchTask;

void AdaptStateToWorkload(const SearchTask& task, const State& state,
                          const Array<IntImm>& shape_values,
                          const float score, float* const occupancy_penalty,
                          float* const padding_penalty,
                          float* const adapted_score
                          );

class ComputeDAG;

double EstimateFlopsForWklInst(const ComputeDAG& compute_dag,
                               const Array<Var>& shape_vars,
                               const Array<IntImm>& shape_values);

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_UTILS_H_
