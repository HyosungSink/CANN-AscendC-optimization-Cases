/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_unpad_batch.h
 * @brief Kernel implementing un-padding (or cropping) of a 2d tensor.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

namespace tcuscan {

/**
 * @brief Kernel that un-pads an ND tensor from GM to GM by reducing the inner
 * dimension's size
 *
 * The input inner dimension has length
 * `batch_len` and the number of batches are `num_batches`. Output tensor memory
 * length is `num_batches * new_batch_len`, with `new_batch_len < batch_len`.
 *
 * @tparam T Input data type
 */
template <typename T>
class KernelUnpadBatch {
  constexpr static uint32_t BUFFER_NUM = 2;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] batch_len Vector length of each batch.
   * @param [in] num_batches Number of batches.
   * @param [in] new_batch_len  Vector length of each batch after the
   * unpadding
   * @param [in] tile_size Tile size
   * to.
   */
  __aicore__ inline KernelUnpadBatch(uint32_t batch_len, uint32_t num_batches,
                                     uint32_t new_batch_len, uint32_t tile_size)
      : block_num_(GetBlockNum() * GetTaskRation()),
        batch_len_(batch_len),
        new_batch_len_(new_batch_len),
        num_batches_(num_batches),
        tile_size_(tile_size),
        num_tiles_per_batch_(scalar::CeilDiv(batch_len, tile_size)),
        num_batches_per_block_(scalar::GetBatchDistribution(
            num_batches_, block_num_, GetBlockIdx())) {
    ASCENDC_ASSERT(new_batch_len < batch_len, {
      KERNEL_LOG(KERNEL_ERROR,
                 "The new batch length should be smaller than the old one",
                 new_batch_len, batch_len);
    });
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] src Pointer to the source buffer in global memory.
   * @param [in] dst Pointer to the destination buffer in global memory.
   */
  __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst) {
    global_src_.SetGlobalBuffer((__gm__ T *)src, batch_len_ * num_batches_);
    global_dst_.SetGlobalBuffer((__gm__ T *)dst, new_batch_len_ * num_batches_);
    pipe_.InitBuffer(vec_in_q_, BUFFER_NUM, tile_size_ * sizeof(T));
    pipe_.InitBuffer(vec_out_q_, BUFFER_NUM, tile_size_ * sizeof(T));
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    uint32_t global_batch_offset =
        scalar::GetBatchOffset(num_batches_, block_num_);
    uint32_t global_in_offset = batch_len_ * global_batch_offset;
    uint32_t global_out_offset = new_batch_len_ * global_batch_offset;

    for (uint32_t batch_idx = 0; batch_idx < num_batches_per_block_;
         batch_idx++) {
      for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_batch_; tile_idx++) {
        const bool full_tile = (tile_idx + 1) * tile_size_ <= new_batch_len_;
        const uint32_t num_elems_to_copy =
            full_tile ? tile_size_ : new_batch_len_ - tile_idx * tile_size_;

        copy::CopyGmToVec(vec_in_q_, global_src_[global_in_offset],
                          num_elems_to_copy);
        copy::CopyVecToGm(global_dst_[global_out_offset], vec_out_q_, vec_in_q_,
                          num_elems_to_copy);

        global_in_offset += tile_size_;
        global_out_offset += num_elems_to_copy;
      }
    }
  }

 private:
  TPipe pipe_;

  GlobalTensor<T> global_src_;
  GlobalTensor<T> global_dst_;
  TQue<QuePosition::VECIN, BUFFER_NUM> vec_in_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> vec_out_q_;

  const uint32_t block_num_;
  const uint32_t batch_len_;
  const uint32_t new_batch_len_;
  const uint32_t num_batches_;
  const uint32_t tile_size_;
  const uint32_t num_tiles_per_batch_;
  const uint32_t num_batches_per_block_;
};

}  // namespace tcuscan
