/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_single_radix.h
 * @brief Kernel implementing the single radix retrieval.
 */
#pragma once

#include "kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Retrieves a bit at the given index from all input values.
 *
 * This kernel takes an input vector and produces a vector containing bits (each
 * bit encoded as a single `uint8_t` value) at the sepcific index from all the
 * input values. Bit index 0 corresponds to the LSB of a number.
 */
template <typename T = uint16_t>
class KernelSingleRadix {
  constexpr static uint32_t BUFFER_NUM = 2;

 public:
  /**
   * @brief Class constructor.
   * @param [in] vec_len Number of elements in the input vector.
   * @param [in] tile_size Tile size.
   * @param [in] bit_idx Index of the bits to retrieve.
   */
  __aicore__ inline KernelSingleRadix(uint32_t vec_len, uint32_t tile_size,
                                      uint16_t bit_idx)
      : block_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        tile_size_(tile_size),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_size_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, block_num_)),
        bit_idx_(bit_idx) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input Pointer to the input vector in global memory.
   * @param [in] output Pointer to the output matrix in global memory.
   */
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
    global_in_.SetGlobalBuffer((__gm__ T*)input, vec_len_);

    global_out_.SetGlobalBuffer((__gm__ uint8_t*)output, vec_len_);

    pipe.InitBuffer(in_q_, BUFFER_NUM, tile_size_ * sizeof(T));
    pipe.InitBuffer(out_q_, BUFFER_NUM, tile_size_ * sizeof(uint8_t));
    pipe.InitBuffer(bitmap_buf_, tile_size_ * sizeof(T));
    pipe.InitBuffer(shift_res_buf_, tile_size_ * sizeof(T));
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    uint32_t global_offset =
        GetBlockIdx() * tile_size_ * max_num_tiles_per_block_;
    const uint32_t num_tiles_to_process =
        tcuscan::scalar::GetWorkDistribution(vec_len_, tile_size_, block_num_);

    if (num_tiles_to_process == 0) {
      return;
    }

    LocalTensor<T> compare_lt = bitmap_buf_.Get<T>();
    const T compare_val = 1 << bit_idx_;
    Duplicate(compare_lt, compare_val, tile_size_);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const bool full_tile = global_offset + tile_size_ <= vec_len_;
      const uint32_t num_elems_to_process_ =
          full_tile ? tile_size_ : vec_len_ - global_offset;

      copy::CopyGmToVec(in_q_, global_in_[global_offset],
                        num_elems_to_process_);
      RadixTile(bit_idx_, compare_lt);
      copy::CopyVecToGm(global_out_[global_offset], out_q_,
                        num_elems_to_process_);
      global_offset += tile_size_;
    }
  }

 private:
  __aicore__ inline void RadixTile(uint16_t bit_level,
                                   const LocalTensor<T>& compare_lt) {
    LocalTensor<T> x_lt = in_q_.DeQue<T>();
    const uint32_t size = x_lt.GetSize();
    // These buffers point to the same UB location but have different types.
    // Manual synchronization is needed!
    const LocalTensor<T> out_uint_lt = shift_res_buf_.Get<T>(size);
    const LocalTensor<half> out_fp16_lt =
        out_uint_lt.template ReinterpretCast<half>();
    const LocalTensor<int16_t> out_int_lt =
        out_uint_lt.template ReinterpretCast<int16_t>();

    And(out_uint_lt, x_lt, compare_lt, size);
    ShiftRight(out_uint_lt, out_uint_lt, bit_level, size);

    PipeBarrier<PIPE_V>();

    Cast(out_fp16_lt, out_int_lt, RoundMode::CAST_NONE, size);

    const LocalTensor<uint8_t> out_8b_lt = out_q_.AllocTensor<uint8_t>();
    Cast(out_8b_lt, out_fp16_lt, RoundMode::CAST_NONE, size);

    out_q_.EnQue<uint8_t>(out_8b_lt);
    in_q_.FreeTensor(x_lt);
  }

  TPipe pipe;

  TQue<QuePosition::VECIN, BUFFER_NUM> in_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> out_q_;
  TBuf<TPosition::VECCALC> bitmap_buf_;
  TBuf<TPosition::VECCALC> shift_res_buf_;

  GlobalTensor<T> global_in_;
  GlobalTensor<uint8_t> global_out_;
  const uint32_t block_num_;
  const uint32_t vec_len_;
  const uint32_t tile_size_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
  const uint16_t bit_idx_;
};

/**
 * @brief Run the `single_radix` kernel.
 *
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to
 * `false` when running inside another mix mode kernel.
 *
 * @param [in] input Pointer to the input vector.
 * @param [in] output Pointer ot the output vector.
 * @param [in] vec_len Number of elements in the input vector.
 * @param [in] tile_size Tile size.
 * @param [in] bit_idx Index of the bits to retrieve.
 */
template <bool ForceMixMode = true>
__aicore__ inline void run_single_radix(GM_ADDR input, GM_ADDR output,
                                        uint32_t vec_len, uint32_t tile_size,
                                        uint16_t bit_idx) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIV {
    KernelSingleRadix op(vec_len, tile_size, bit_idx);
    op.Init(input, output);
    op.Process();
  }
}

}  // namespace tcuscan
