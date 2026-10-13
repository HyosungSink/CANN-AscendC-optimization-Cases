/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_pad_batch.h
 * @brief Kernel implementing a Vector multi-dimensional padding kernel
 * operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Kernel that pads an ND tensor from GM to GM by padding the inner
 * dimension by an alignment legnth.
 *
 * The input inner dimension has length
 * `batch_len` and the number of batches are `num_batches`. The additional
 * padded elements are filled with zeros. Output tensor memory length is
 * `num_batches * AlignUp(batch_len, align_len)`. The tile length equals to the
 * aligned length.
 *
 * @tparam T Input data type
 */
template <typename T>
class KernelPadBatch {
  constexpr static uint32_t BUFFER_NUM = 2;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] batch_len Vector length of each batch.
   * @param [in] num_batches Number of batches.
   * @param [in] align_len Number of elements that the output will be aligned
   to.
   * @param [in] tile_size Tile size
   */
  __aicore__ inline KernelPadBatch(uint32_t batch_len, uint32_t num_batches,
                                   uint32_t align_len, uint32_t tile_size)
      : block_num_(GetBlockNum() * GetTaskRation()),
        batch_len_(batch_len),
        num_batches_(num_batches),
        align_len_(align_len),
        tile_size_(tile_size),
        pad_len_(batch_len % align_len_ ? align_len_ - batch_len % align_len_
                                        : 0),
        num_tiles_per_batch_(scalar::CeilDiv(batch_len, tile_size)),
        num_batches_per_block_(scalar::GetBatchDistribution(
            num_batches_, block_num_, GetBlockIdx())) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] src Pointer to the source buffer in global memory.
   * @param [in] dst Pointer to the destination buffer in global memory.
   */
  __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst) {
    global_src_.SetGlobalBuffer((__gm__ T*)src, batch_len_ * num_batches_);
    global_dst_.SetGlobalBuffer(
        (__gm__ T*)dst, scalar::AlignUp(batch_len_, align_len_) * num_batches_);
    pipe_.InitBuffer(vec_in_q_, BUFFER_NUM, tile_size_ * sizeof(T));
    pipe_.InitBuffer(vec_out_q_, BUFFER_NUM, tile_size_ * sizeof(T));
    pipe_.InitBuffer(vec_out_pad_q_, 1,
                     scalar::AlignUp(pad_len_, UB_ALIGNMENT) * sizeof(T));
    if constexpr (!duplicate::IsDuplicateSupported<T>)
      pipe_.InitBuffer(pad_h_buf_,
                       // Aligned to UB -> Aligned to sizeof(half)
                       scalar::AlignUp(pad_len_, UB_ALIGNMENT) * sizeof(T));
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    uint32_t global_batch_offset =
        scalar::GetBatchOffset(num_batches_, block_num_);
    uint32_t global_in_offset = batch_len_ * global_batch_offset;
    uint32_t global_out_offset =
        scalar::AlignUp(batch_len_, align_len_) * global_batch_offset;

    for (uint32_t batch_idx = 0; batch_idx < num_batches_per_block_;
         batch_idx++) {
      for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_batch_; tile_idx++) {
        const bool full_tile = (tile_idx + 1) * tile_size_ <= batch_len_;
        const uint32_t num_elems_to_copy =
            full_tile ? tile_size_ : batch_len_ - tile_idx * tile_size_;
        if (num_elems_to_copy) {
          copy::CopyGmToVec(vec_in_q_, global_src_[global_in_offset],
                            num_elems_to_copy);

          // Save the real data first.
          copy::CopyVecToGm(global_dst_[global_out_offset], vec_out_q_,
                            vec_in_q_, num_elems_to_copy);
        }
        global_in_offset += num_elems_to_copy;
        global_out_offset += num_elems_to_copy;
      }
      if (pad_len_) {
        if constexpr (!duplicate::IsDuplicateSupported<T>) {
          const LocalTensor<half> pad_h_lt = pad_h_buf_.Get<half>();

          Duplicate(pad_h_lt, (half)0, pad_h_lt.GetSize());

          const LocalTensor<T> pad_lt = pad_h_lt.template ReinterpretCast<T>();

          PipeBarrier<PIPE_V>();

          copy::CopyVecToGm(global_dst_[global_out_offset], vec_out_pad_q_,
                            pad_lt, pad_len_);
        } else {
          const LocalTensor<T> pad_lt =
              vec_out_pad_q_.template AllocTensor<T>();
          Duplicate(pad_lt, (T)0, pad_len_);
          vec_out_pad_q_.EnQue(pad_lt);
          copy::CopyVecToGm(global_dst_[global_out_offset], vec_out_pad_q_,
                            pad_len_);
        }
        global_out_offset += pad_len_;
      }
    }
  }

 private:
  TPipe pipe_;

  GlobalTensor<T> global_src_;
  GlobalTensor<T> global_dst_;
  TQue<QuePosition::VECIN, BUFFER_NUM> vec_in_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> vec_out_q_;
  TQue<QuePosition::VECOUT, 1> vec_out_pad_q_;
  TBuf<QuePosition::VECCALC> pad_h_buf_;

  const uint32_t block_num_;
  const uint32_t batch_len_;
  const uint32_t num_batches_;
  const uint32_t align_len_;
  const uint32_t tile_size_;
  const uint32_t pad_len_;
  const uint32_t num_tiles_per_batch_;
  const uint32_t num_batches_per_block_;
};

/**
 * @brief Run the `pad_batch` kernel.
 *
 * @tparam InputT Input data type
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to
 * `false` when running inside another mix mode kernel.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] batch_len Input vector length per batch.
 * @param [in] num_batches Number of batches.
 * @param [in] align_len Number of elements that the output will be aligned
 * @param [in] tile_size Tile size
 */
template <typename InputT, bool ForceMixMode = true>
__aicore__ inline void run_pad_batch(GM_ADDR vec_in, GM_ADDR vec_out,
                                     uint32_t batch_len, uint32_t num_batches,
                                     uint32_t align_len, uint32_t tile_size) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIV {
    KernelPadBatch<InputT> op(batch_len, num_batches, align_len, tile_size);
    op.Init(vec_in, vec_out);
    op.Process();
  }
}

}  // namespace tcuscan
