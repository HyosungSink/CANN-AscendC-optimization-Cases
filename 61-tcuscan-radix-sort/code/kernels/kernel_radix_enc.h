/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_radix_enc.h
 * @brief Kernel implementing the encoding/decoding step to enable floating
 * point sorting with radix sort.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Encodes/decodes fp16 values into a format that can be used to sort
 * them with (int16) radix-sort
 *
 * Radix sort correctly orders positive floating point values, but the order of
 * negative values is reversed. This kernel performs a bit-wise not on the
 * negative numbers so that they are correctly ordered, but preserves the value
 * of the sign bit. It preserves the sign bit because the int16 radix-sort
 * already correclty handles it, by reversing the last split operation.
 *
 * Running the kernel on already encoded values will decode it back to the
 * original values.
 */
class KernelRadixEnc {
  constexpr static uint32_t BUFFER_NUM = 1;

 public:
  /**
   * @brief Class constructor.
   * @param [in] vec_len Dimension of the input vectors.
   * @param [in] tile_size Tile size.
   */
  __aicore__ inline KernelRadixEnc(uint32_t vec_len, uint32_t tile_size)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        tile_size_(tile_size),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_size_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, vec_core_num_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] x Pointer to the input vector in global memory.
   * @param [in] y Pointer to the output matrix in global memory.
   */
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
    global_x_.SetGlobalBuffer((__gm__ half*)x, vec_len_);
    global_y_.SetGlobalBuffer((__gm__ half*)y, vec_len_);

    pipe.InitBuffer(in_q_, BUFFER_NUM, tile_size_ * sizeof(half));
    pipe.InitBuffer(out_q_, BUFFER_NUM, tile_size_ * sizeof(half));
    pipe.InitBuffer(xor_buf_, tile_size_ * sizeof(int16_t));
    pipe.InitBuffer(x_buf_, tile_size_ * sizeof(int16_t));
    pipe.InitBuffer(sign_int16_buf_, tile_size_ * sizeof(int16_t));
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    uint32_t global_offset =
        GetBlockIdx() * tile_size_ * max_num_tiles_per_block_;
    const uint32_t num_tiles_to_process = tcuscan::scalar::GetWorkDistribution(
        vec_len_, tile_size_, vec_core_num_);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const bool full_tile = global_offset + tile_size_ <= vec_len_;
      const uint32_t num_elems_to_process_ =
          full_tile ? tile_size_ : vec_len_ - global_offset;

      copy::CopyGmToVec(in_q_, global_x_[global_offset], num_elems_to_process_);
      RadixEncTile();
      copy::CopyVecToGm(global_y_[global_offset], out_q_,
                        num_elems_to_process_);
      global_offset += tile_size_;
    }
  }

 private:
  __aicore__ inline void RadixEncTile() {
    LocalTensor<half> x_lt = in_q_.DeQue<half>();
    const LocalTensor<half> y_lt = out_q_.AllocTensor<half>();
    // These buffers point to the same UB location but have different types.
    // Manual synchronization is needed!
    const LocalTensor<half> x_fp16_lt = x_buf_.Get<half>();
    const LocalTensor<int16_t> x_int16_lt =
        x_fp16_lt.template ReinterpretCast<int16_t>();

    const LocalTensor<int16_t> sign_int16_lt = sign_int16_buf_.Get<int16_t>();
    const LocalTensor<int16_t> xor_int16_lt = xor_buf_.Get<int16_t>();

    const uint32_t size = x_lt.GetSize();

    DataCopy(x_fp16_lt, x_lt, size);
    PipeBarrier<PIPE_V>();

    // Note: positive and negative refer to the sign of the initial value
    Not(x_int16_lt, x_int16_lt, size);

    // if positive sub -1
    // if negative sub 0
    ShiftRight(sign_int16_lt, x_int16_lt, (int16_t)15, size);
    Sub(x_int16_lt, x_int16_lt, sign_int16_lt, size);

    // if positive xor with 0000 0000 0000 0000 (no change)
    // if negative xor with 1000 0000 0000 0000 (flip sign bit)
    Not(sign_int16_lt, sign_int16_lt, size);
    ShiftLeft(sign_int16_lt, sign_int16_lt, (int16_t)15, size);
    Xor(xor_int16_lt, x_int16_lt, sign_int16_lt, size);

    // if positive multiply by -1 (equivalent to reversing all operations
    // so
    // far)
    // if negative multiply by 1 (no change to the applied operations)
    Not(sign_int16_lt, sign_int16_lt, size);
    ShiftRight(sign_int16_lt, sign_int16_lt, (int16_t)14, size);
    Mul(x_int16_lt, xor_int16_lt, sign_int16_lt, size);
    PipeBarrier<PIPE_V>();

    DataCopy(y_lt, x_fp16_lt, y_lt.GetSize());

    out_q_.EnQue<half>(y_lt);
    in_q_.FreeTensor(x_lt);
  }
  TPipe pipe;

  TQue<QuePosition::VECIN, BUFFER_NUM> in_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> out_q_;
  TBuf<QuePosition::VECCALC> xor_buf_;
  TBuf<QuePosition::VECCALC> x_buf_;
  TBuf<QuePosition::VECCALC> sign_int16_buf_;

  GlobalTensor<half> global_x_;
  GlobalTensor<half> global_y_;

  const uint32_t vec_core_num_;
  const uint32_t vec_len_;
  const uint32_t tile_size_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
};

/**
 * @brief Run the `radix encode` kernel.
 *
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 *
 * @param [in] x Pointer to the input vector.
 * @param [in] y Pointer ot the output vector.
 * @param [in] vec_len Dimension of the input vector.
 * @param [in] tile_size Tile size.
 */
template <bool ForceMixMode = true>
__aicore__ inline void run_radix_encode(GM_ADDR x, GM_ADDR y, uint32_t vec_len,
                                        uint32_t tile_size) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIV {
    KernelRadixEnc op(vec_len, tile_size);
    op.Init(x, y);
    op.Process();
  }
}

}  // namespace tcuscan
