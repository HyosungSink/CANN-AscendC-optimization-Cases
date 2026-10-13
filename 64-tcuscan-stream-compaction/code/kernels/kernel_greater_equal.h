/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_greater_equal.h
 * @brief Kernel implementing a greater or equal operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Compares an array with a scalar with a greater or equal operation
 *
 * This algorithm compares an array with a scalar and it returns an array of
 * integers (int8_t). If an entry in the output array is 1 it means that the
 * corresponding value in the input array was greater or equal than the scalar
 * otherwise if 0.
 *
 */
template <typename InputT = int16_t>
class KernelGreaterEqual {
  constexpr static uint32_t BUFFER_NUM = 1;

 public:
  /**
   * @brief Class constructor.
   * @param [in] vec_len Dimension of the input vectors.
   * @param [in] tile_size Tile size.
   */
  __aicore__ inline KernelGreaterEqual(uint32_t vec_len, uint32_t tile_size)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        tile_size_(tile_size),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_size_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, vec_core_num_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] x Pointer to the input vector in global memory.
   * @param [in] y Pointer to the output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
    global_x_.SetGlobalBuffer((__gm__ InputT*)x, vec_len_);
    global_y_.SetGlobalBuffer((__gm__ uint8_t*)y, vec_len_);

    pipe.InitBuffer(in_q_, BUFFER_NUM, tile_size_ * sizeof(InputT));
    pipe.InitBuffer(out_q_, BUFFER_NUM, tile_size_ * sizeof(uint8_t));
    pipe.InitBuffer(x_buf_, tile_size_ * sizeof(InputT));
  }

  /**
   * @brief Run the kernel.
   *
   * @param [in] pivot Value to compare the input with.
   */
  __aicore__ inline void Process(InputT pivot) {
    uint32_t global_offset =
        GetBlockIdx() * tile_size_ * max_num_tiles_per_block_;
    const uint32_t num_tiles_to_process = tcuscan::scalar::GetWorkDistribution(
        vec_len_, tile_size_, vec_core_num_);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const bool full_tile = global_offset + tile_size_ <= vec_len_;
      const uint32_t num_elems_to_process =
          full_tile ? tile_size_ : vec_len_ - global_offset;

      copy::CopyGmToVec(in_q_, global_x_[global_offset], num_elems_to_process);
      ProcessTile(pivot, num_elems_to_process);
      copy::CopyVecToGm(global_y_[global_offset], out_q_, num_elems_to_process);
      global_offset += tile_size_;
    }
  }

 private:
  __aicore__ inline void ProcessTile(InputT pivot,
                                     uint32_t num_elems_to_process) {
    LocalTensor<InputT> x_lt = in_q_.DeQue<InputT>();
    const LocalTensor<uint8_t> y_lt = out_q_.AllocTensor<uint8_t>();
    // These buffers point to the same UB location but have different
    // types. Manual synchronization is needed!
    const LocalTensor<InputT> work_lt = x_buf_.Get<InputT>();

    compare::GreaterEqual<InputT, uint8_t>(y_lt, x_lt, pivot, work_lt,
                                           num_elems_to_process);

    out_q_.EnQue<uint8_t>(y_lt);
    in_q_.FreeTensor(x_lt);
  }

  TPipe pipe;

  TQue<QuePosition::VECIN, BUFFER_NUM> in_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> out_q_;
  TBuf<TPosition::VECCALC> x_buf_;

  GlobalTensor<InputT> global_x_;
  GlobalTensor<uint8_t> global_y_;

  const uint32_t vec_core_num_;
  const uint32_t vec_len_;
  const uint32_t tile_size_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
};

/**
 * @brief Run the `greater or equal` kernel.
 *
 * @tparam T Input data type.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] pivot Value to compare the input with.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] vec_len Dimension of the input vector.
 * @param [in] tile_size Tile size.
 */
template <typename T = half>
__aicore__ inline void run_greater_equal(GM_ADDR vec_in, T pivot,
                                         GM_ADDR vec_out, uint32_t vec_len,
                                         uint32_t tile_size) {
  if ASCEND_IS_AIV {
    KernelGreaterEqual<T> op(vec_len, tile_size);
    op.Init(vec_in, vec_out);
    op.Process(pivot);
  }
}

}  // namespace tcuscan
