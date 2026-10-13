/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file kernel_topk_pivot.h
 * @brief Kernel implementing a Vector k-largest value estimator.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"
#include "../tiling/tiling_topk.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Returns an estimation \f$\theta\f$ of the k-th largest value of an
 * input array \f$x\f$. The estimation always lower-bounds the k-largest value,
 * i.e., \f$ topk\_value(x) \geq \theta\f$.
 *
 * @tparam T Input data type. Default `half`.
 */
template <typename T = half>
class KernelPivotTopkEstimator {
  /// @brief Estimator has 32 buffers of length 32. Total length 1,024.
  constexpr static uint32_t ESTIMATOR_LEN = 32;
  constexpr static uint32_t MIN_VEC_SIZE = UB_ALIGNMENT / sizeof(T);

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Dimension of the input vectors.
   * @param [in] num_samples Number of samples of length 32.
   * @param [in] k_inner Parameter of inner-most top-k value. It must hold
   * that 1< k_inner < 32.
   * @param [in] k_outer Parameter of top-\f$k\_outer\f$ of top-\f$k\_inner\f$
   * values. It must hold that \f$ 1 < k_outer < num_samples \f$.
   */
  __aicore__ inline KernelPivotTopkEstimator(uint32_t vec_len,
                                             uint32_t num_samples,
                                             uint32_t k_inner, uint32_t k_outer)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        num_samples_(num_samples),
        k_inner_{k_inner},
        k_outer_{k_outer},
        tile_len_(num_samples * ESTIMATOR_LEN),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_len_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, vec_core_num_)),
        num_iters_(8) {
    ASCENDC_ASSERT(k_inner_ < 32, {
      KERNEL_LOG(KERNEL_ERROR,
                 "K-largest (top-k) value estimator supports k_inner at most "
                 "31. Got (k_inner_=%d, AIVs=%d)",
                 k_inner_, vec_core_num_);
    });
    ASCENDC_ASSERT(k_outer_ < 32, {
      KERNEL_LOG(KERNEL_ERROR,
                 "K-largest (top-k) value estimator supports k_outer_ at most "
                 "31. Got (k_outer_=%d, AIVs=%d)",
                 k_inner_, vec_core_num_);
    });
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to the input vector in global memory.
   * @param [in] vec_out Pointer to the output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR vec_out) {
    global_in_.SetGlobalBuffer((__gm__ T*)vec_in, vec_len_);
    global_out_.SetGlobalBuffer((__gm__ T*)vec_out, vec_core_num_);

    pipe_.InitBuffer(in_q_, 2, num_iters_ * tile_len_ * sizeof(T));
    pipe_.InitBuffer(out_q_, 1, vec_core_num_ * sizeof(T));

    // Sort32 writes into a Tuple 2 of (value, index). Each output element
    // of Sort32 is 8-bytes.
    constexpr uint32_t SORT_ELEM_SIZE = 8;
    pipe_.InitBuffer(concat_values_indices_buf_, tile_len_ * SORT_ELEM_SIZE);
    pipe_.InitBuffer(unused_indices_buf_, tile_len_ * sizeof(uint32_t));
    pipe_.InitBuffer(tmp_sorted_buf_, num_samples_ * ESTIMATOR_LEN * sizeof(T));
  }

  /**
   * @brief Run the kernel.
   *
   */
  __aicore__ inline void Process() {
    const uint32_t global_offset =
        GetBlockIdx() * tile_len_ * max_num_tiles_per_block_;

    // Calculate the k-largest estimate scalar and write back to GM.
    T max_estimate = std::numeric_limits<T>::min();
    copy::CopyGmToVec(in_q_, global_in_[global_offset], num_iters_ * tile_len_);
    LocalTensor<T> vec_in_lt = in_q_.DeQue<T>();
    for (uint32_t tile_idx = 0; tile_idx < num_iters_; tile_idx++) {
      const T estimate = ProcessTile(vec_in_lt[tile_idx * tile_len_]);
      max_estimate = scalar::Max(max_estimate, estimate);
    }
    in_q_.FreeTensor<T>(vec_in_lt);

    copy::CopyScalarToGm(global_out_[GetBlockIdx()], out_q_, max_estimate);
  }

 private:
  __aicore__ inline T ProcessTile(const LocalTensor<T>& vec_in_lt) {
    LocalTensor<T> concat_values_indices_lt =
        concat_values_indices_buf_.Get<T>();
    LocalTensor<uint32_t> unused_index_lt = unused_indices_buf_.Get<uint32_t>();
    LocalTensor<T> tmp_sorted_lt = tmp_sorted_buf_.Get<T>();

    // Sort32 bundles together the values (half/float) and indices
    // (uint32_t) into 8-bytes
    const int32_t repeats_of_32_elems = num_samples_;
    AscendC::Sort32<T>(concat_values_indices_lt, vec_in_lt, unused_index_lt,
                       repeats_of_32_elems);

    // Extracts the values and indices from 'sorted' (value, index) 8-bytes
    AscendC::Extract(tmp_sorted_lt, unused_index_lt, concat_values_indices_lt,
                     repeats_of_32_elems);

    // if (GetBlockIdx() == 0) {
    //     AscendC::DumpTensor(tmp_sorted_lt, 666, tmp_sorted_lt.GetSize());
    // }

    // Write back the (k_inner_)-th largest value into vec_in_lt
    for (int32_t i = 0; i < repeats_of_32_elems; i++) {
      const T current = tmp_sorted_lt(i * 32 + k_inner_ - 1);
      vec_in_lt.SetValue(i, current);
    }

    // Sort again (?!) the (k_inner_)-th largest value, like in the "median
    // of medians algorithm"
    AscendC::Sort32<T>(concat_values_indices_lt, vec_in_lt, unused_index_lt, 1);
    AscendC::Extract(tmp_sorted_lt, unused_index_lt, concat_values_indices_lt,
                     repeats_of_32_elems);

    // Estimate is the (k_outer_)- largest value of the (k_inner_)-th
    // largest values.
    const T estimate = tmp_sorted_lt(k_outer_ - 1);

    return estimate;
  }

  TPipe pipe_;

  TQue<QuePosition::VECIN, 2> in_q_;
  TQue<QuePosition::VECOUT, 1> out_q_;

  TBuf<QuePosition::VECCALC> concat_values_indices_buf_;
  TBuf<QuePosition::VECCALC> unused_indices_buf_;
  TBuf<QuePosition::VECCALC> tmp_sorted_buf_;

  GlobalTensor<T> global_in_;
  GlobalTensor<T> global_out_;

  const uint32_t vec_core_num_;
  const uint32_t vec_len_;
  const uint32_t num_samples_;
  const uint32_t k_inner_;
  const uint32_t k_outer_;
  const uint32_t tile_len_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
  const uint32_t num_iters_;
};

/**
 * @brief Run the `topk_pivot` kernel.
 *
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 * @tparam T Input data type.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] vec_len Dimension of the input vector.
 * @param [in] num_samples Number of samples of length 32.
 * @param [in] k_inner Parameter of inner-most top-k value. It must hold that
 * 1< k_inner < 32.
 * @param [in] k_outer Parameter of top-\f$k\_outer\f$ of top-\f$k\_inner\f$
 * values. It must hold that \f$ 1 < k_outer < num_samples \f$.
 */
template <bool ForceMixMode = true, typename T>
__aicore__ inline void run_topk_pivot(GM_ADDR vec_in, GM_ADDR vec_out,
                                      uint32_t vec_len, uint32_t num_samples,
                                      uint32_t k_inner, uint32_t k_outer) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIV {
    KernelPivotTopkEstimator<T> op(vec_len, num_samples, k_inner, k_outer);
    op.Init(vec_in, vec_out);
    op.Process();
  }
}

}  // namespace tcuscan
