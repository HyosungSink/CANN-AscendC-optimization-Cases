/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_scan_vec_only.h
 * @brief Kernel implementing a vec-only scan using the CumSum API.
 */
#pragma once

#include "../tiling/tiling_scan_vec_only.h"
#include "kernel_pad.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Performs an inclusive scan on an input vector using the CumSum
 * vector-core API
 *
 * The algorithm utilizes only a single block.
 *
 * The algorithm splits the input vector into \f$N\f$ blocks and computes the
 * scan of each block. The algorithm then processes the blocks one after another
 * adding the last result from the previous block.
 */
class KernelScanVecOnly {
 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Number of elements in an input vector.
   * @param [in] tile_height Height of the tile: number of scans processed in
   * parallel by cumsum.
   * @param [in] tile_width Width of the tile: number of elements that
   * cumsum scans.
   */
  __aicore__ inline KernelScanVecOnly(uint32_t vec_len, uint32_t tile_height,
                                      uint32_t tile_width)
      : vec_len_(vec_len),
        tile_height_(tile_height),
        tile_width_(tile_width),
        tile_size_(tile_height * tile_width),
        num_tiles_(vec_len_ / tile_size_) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input Pointer to input vector in global memory.
   * @param [in] output Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
    global_input_.SetGlobalBuffer((__gm__ half*)input);
    global_output_.SetGlobalBuffer((__gm__ float*)output);

    pipe.InitBuffer(vecin_q_, 1, tile_size_ * sizeof(half));
    pipe.InitBuffer(vecout_q_, 1, tile_size_ * sizeof(float));
    pipe.InitBuffer(work_buf_, tile_size_ * sizeof(float));
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    float running_sum = 0.f;
    if (GetBlockIdx() != 0) return;
    for (uint32_t idx = 0; idx < num_tiles_; ++idx) {
      copy::CopyGmToVec<half>(vecin_q_, global_input_[idx * tile_size_]);
      ScanBlock();
      running_sum = CompleteBlock(running_sum);
      copy::CopyVecToGm<float>(global_output_[idx * tile_size_], vecout_q_,
                               work_buf_.Get<float>());
    }
  }

 private:
  __aicore__ inline void ScanBlock() {
    LocalTensor<half> in_lt = vecin_q_.DeQue<half>();
    LocalTensor<float> temp_lt = work_buf_.Get<float>();

    Cast(temp_lt, in_lt, RoundMode::CAST_NONE, tile_size_);
    vecin_q_.FreeTensor(in_lt);

    static constexpr CumSumConfig cumsum_config{true, false, false};
    const CumSumInfo cumsum_info{tile_height_, tile_width_};
    CumSum<float, cumsum_config>(temp_lt, temp_lt, temp_lt, cumsum_info);
  }

  __aicore__ inline float CompleteBlock(float initial_sum) {
    // Get 2 local tensors pointing to the same buffer: this is needed to
    // trick the compiler into considering operations on vec_buf1 and
    // vec_buf2 independent.
    const LocalTensor<float> vec_buf1 = work_buf_.Get<float>();
    const LocalTensor<float> vec_buf2 = work_buf_.Get<float>();

    sync::ScalarWaitForVec();

    uint32_t first_offset = 0;
    uint32_t second_offset = tile_width_;
    float first_sum = initial_sum;
    float second_sum = vec_buf1.GetValue(second_offset - 1) + first_sum;
    for (uint32_t i = 0; i < tile_height_; i += 2) {
      // The Adds instructions can be overlapped because they are
      // independent.
      Adds(vec_buf1[first_offset], vec_buf1[first_offset], first_sum,
           tile_width_);
      Adds(vec_buf2[second_offset], vec_buf2[second_offset], second_sum,
           tile_width_);
      first_offset += tile_width_ * 2;
      second_offset += tile_width_ * 2;
      first_sum = vec_buf2.GetValue(first_offset - 1);
      if (i + 2 < tile_height_)
        second_sum = vec_buf1.GetValue(second_offset - 1) + first_sum;
    }

    return first_sum;
  }

  TPipe pipe;

  TQue<QuePosition::VECIN, 1> vecin_q_;
  TQue<QuePosition::VECOUT, 1> vecout_q_;
  TBuf<QuePosition::VECCALC> work_buf_;

  GlobalTensor<half> global_input_;
  GlobalTensor<float> global_output_;

  const uint32_t vec_len_;
  const uint32_t tile_height_;
  const uint32_t tile_width_;
  const uint32_t tile_size_;
  const uint32_t num_tiles_;
};

}  // namespace tcuscan
