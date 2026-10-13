/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_pad.h
 * @brief Kernel implementing a pad operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Kernel that moves data from GM to GM and adds padding to it
 *
 * This kernel copies a vector to UB using DataCopyPad,
 * then it writes it back to GM as vector aligned to a given size. The
 * additional memory is filled with zeros.
 */
template <typename DataType>
class KernelPad {
 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] buf_len Number of elements in the buffer.
   * @param [in] align_len Number of elements that the output will be aligned
   * to.
   */
  __aicore__ inline KernelPad(uint32_t buf_len, uint32_t align_len)
      : block_num_(GetBlockNum() * GetTaskRation()),
        align_len_(align_len),
        buf_len_(buf_len),
        pad_len_(align_len_ - buf_len_ % align_len_),
        num_tiles_(scalar::CeilDiv(buf_len, align_len_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, block_num_)) {
    num_tiles_to_copy_ =
        tcuscan::scalar::GetWorkDistribution(buf_len_, align_len_, block_num_);
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] src Pointer to the source buffer in global memory.
   * @param [in] dst Pointer to the destination buffer in global memory.
   */
  __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst) {
    global_src_.SetGlobalBuffer((__gm__ DataType*)src, buf_len_);
    global_dst_.SetGlobalBuffer((__gm__ DataType*)dst,
                                scalar::AlignUp(buf_len_, align_len_));
    pipe.InitBuffer(vec_in_q_, buf_num_, align_len_ * sizeof(DataType));
    pipe.InitBuffer(vec_out_q_, buf_num_, align_len_ * sizeof(DataType));
    pipe.InitBuffer(vec_out_pad_q_, 1,
                    scalar::AlignUp(pad_len_, UB_ALIGNMENT) * sizeof(DataType));
    if constexpr (!duplicate::IsDuplicateSupported<DataType>)
      pipe.InitBuffer(
          pad_h_buf_,
          // Aligned to UB -> Aligned to sizeof(half)
          scalar::AlignUp(pad_len_, UB_ALIGNMENT) * sizeof(DataType));
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    uint32_t global_offset =
        GetBlockIdx() * align_len_ * max_num_tiles_per_block_;

    /* Aligned copies loop */
    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_copy_; tile_idx++) {
      const bool full_tile = global_offset + align_len_ <= buf_len_;
      const uint32_t num_elems_to_copy =
          full_tile ? align_len_ : buf_len_ - global_offset;
      copy::CopyGmToVec(vec_in_q_, global_src_[global_offset],
                        num_elems_to_copy);

      // Save the real data first.
      copy::CopyVecToGm(global_dst_[global_offset], vec_out_q_, vec_in_q_,
                        num_elems_to_copy);

      if (!full_tile) {
        if constexpr (!duplicate::IsDuplicateSupported<DataType>) {
          const LocalTensor<half> pad_h_lt = pad_h_buf_.Get<half>();

          Duplicate(pad_h_lt, (half)0, pad_h_lt.GetSize());

          const LocalTensor<DataType> pad_lt =
              pad_h_lt.template ReinterpretCast<DataType>();

          PipeBarrier<PIPE_V>();

          copy::CopyVecToGm(global_dst_[global_offset + num_elems_to_copy],
                            vec_out_pad_q_, pad_lt, pad_len_);
        } else {
          const LocalTensor<DataType> pad_lt =
              vec_out_pad_q_.template AllocTensor<DataType>();
          Duplicate(pad_lt, (DataType)0, pad_len_);
          vec_out_pad_q_.EnQue(pad_lt);
          copy::CopyVecToGm(global_dst_[global_offset + num_elems_to_copy],
                            vec_out_pad_q_, pad_len_);
        }
      }

      global_offset += align_len_;
    }
  }

 private:
  TPipe pipe;
  /// @brief Double buffering is enabled for all types but float/fp32, whose
  /// tiles (`align_len` elements) would otherwise overflow the UB.
  constexpr static uint32_t buf_num_ = std::is_same_v<DataType, float> ? 1 : 2;

  GlobalTensor<DataType> global_src_;
  GlobalTensor<DataType> global_dst_;
  TQue<QuePosition::VECIN, buf_num_> vec_in_q_;
  TQue<QuePosition::VECOUT, buf_num_> vec_out_q_;
  TQue<QuePosition::VECOUT, 1> vec_out_pad_q_;
  TBuf<QuePosition::VECCALC> pad_h_buf_;

  const uint32_t block_num_;
  const uint32_t align_len_;
  const uint32_t buf_len_;
  const uint32_t pad_len_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
  uint32_t num_tiles_to_copy_;
};

/**
 * @brief Run the `pad` kernel.
 *
 * @tparam DataType Data type of the input and output vectors.
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 *
 * @param [in] in Pointer to input vector.
 * @param [in] out Pointer to output vector.
 * @param [in] vec_len Dimension of the input (may be unaligned).
 * @param [in] align_len Length to be aligned to (has to be aligned to 32
 * bytes).
 */
template <typename DataType, bool ForceMixMode = true>
__aicore__ inline void run_pad_kernel(GM_ADDR in, GM_ADDR out, uint32_t vec_len,
                                      uint32_t align_len) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIV {
    KernelPad<DataType> op_pad(vec_len, align_len);
    op_pad.Init(in, out);
    op_pad.Process();
  }
}

}  // namespace tcuscan
