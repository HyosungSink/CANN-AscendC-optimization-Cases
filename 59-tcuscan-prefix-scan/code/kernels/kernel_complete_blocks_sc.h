/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * @file kernel_complete_blocks_sc.h
 * @brief Kernel implementing a single AI-core cube-only scan.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Transforms block scans into a complete inclusive scan.
 */
class KernelCompleteBlocksSingleCore {
 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Number of elements in an input vector.
   * @param [in] tile_size Size of the block.
   */
  __aicore__ inline KernelCompleteBlocksSingleCore(uint32_t vec_len,
                                                   uint32_t tile_size)
      : vec_len_(vec_len),
        tile_size_(tile_size),
        num_tiles_(vec_len_ / tile_size_) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input Pointer to row-wise scan in global memory.
   * @param [in] output Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
    global_input_.SetGlobalBuffer((__gm__ float*)input);
    global_output_.SetGlobalBuffer((__gm__ float*)output);

    pipe.InitBuffer(vecin_q_, 1, tile_size_ * sizeof(float));
    pipe.InitBuffer(vecout_q_, 1, tile_size_ * sizeof(float));
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    float running_sum = 0.f;
    for (uint32_t idx = 0; idx < num_tiles_; ++idx) {
      sync::SyncGroup<sync::GroupSyncDirection::FULL>();
      if (GetBlockIdx() == 0) running_sum = VecIter(idx, running_sum);
    }
  }

 private:
  __aicore__ inline float VecIter(uint32_t iter_idx, float running_sum) {
    copy::CopyGmToVec(vecin_q_, global_input_[iter_idx * tile_size_]);
    running_sum = ReduceWithVec(running_sum);
    copy::CopyVecToGm<float>(global_output_[iter_idx * tile_size_], vecout_q_,
                             vecin_q_);
    return running_sum;
  }

  __aicore__ inline float ReduceWithVec(float running_sum) {
    const LocalTensor<float> vec_lt = vecin_q_.DeQue<float>();
    Adds(vec_lt, vec_lt, running_sum, tile_size_);
    running_sum = vec_lt.GetValue(tile_size_ - 1);
    vecin_q_.EnQue(vec_lt);
    return running_sum;
  }

  TPipe pipe;

  TQue<QuePosition::VECIN, 1> vecin_q_;
  TQue<QuePosition::VECOUT, 1> vecout_q_;

  GlobalTensor<float> global_input_;
  GlobalTensor<float> global_output_;

  const uint32_t vec_len_;
  const uint32_t tile_size_;
  const uint32_t num_tiles_;
};
}  // namespace tcuscan
