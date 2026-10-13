/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file kernel_where.h
 * @brief Kernel implementing a Vector where kernel operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Returns np.where(x >= pivot) for input tensor x and scalar `pivot`.
 *
 */
class KernelWhere {
  using MaskT = uint8_t;
  using PosT = int32_t;

  using PackedMaskT = uint8_t;

  template <typename U>
  using _GatherMaskT =
      typename std::conditional<sizeof(U) == 2, uint16_t, uint32_t>::type;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Number of elements in an input vector.
   * @param [in] tile_len Tile length.
   */
  __aicore__ inline KernelWhere(uint32_t vec_len, uint32_t tile_len)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        tile_len_(tile_len),
        mask_required_elems_(tile_len_ / IN_ELEMS_PER_MASK_ELEM),
        packed_mask_tile_size_(mask_required_elems_ < SMALLEST_MASK
                                   ? SMALLEST_MASK
                                   : mask_required_elems_),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_len_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, vec_core_num_)),
        max_num_elems_per_block_(tile_len_ * max_num_tiles_per_block_),
        num_elems_before_block_(GetBlockIdx() * max_num_elems_per_block_) {
    // GatherMask requires either 2- or 4-byte elements
    static_assert(sizeof(PosT) >= 2);
  }

 public:
  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] mask Pointer to mask vector in global memory.
   * @param [in] pos Pointer to position vector in global memory.
   * @param [in] output Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR mask, GM_ADDR pos, GM_ADDR output) {
    global_output_.SetGlobalBuffer((__gm__ PosT*)output, vec_len_);
    global_mask_.SetGlobalBuffer((__gm__ MaskT*)mask, vec_len_);
    global_pos_.SetGlobalBuffer((__gm__ PosT*)pos, vec_core_num_);

    pipe_.InitBuffer(vec_in_buf_, tile_len_ * sizeof(PosT));
    pipe_.InitBuffer(mask_in_q_, 1, tile_len_ * sizeof(MaskT));
    pipe_.InitBuffer(pos_in_q_, 1, tile_len_ * sizeof(PosT));
    pipe_.InitBuffer(mask_fp16_buf_, tile_len_ * sizeof(half));
    pipe_.InitBuffer(packed_mask_buf_,
                     packed_mask_tile_size_ * sizeof(PackedMaskT));
    pipe_.InitBuffer(gathered_out_q_, 2, tile_len_ * sizeof(PosT));
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    const uint32_t num_tiles_to_process = tcuscan::scalar::GetWorkDistribution(
        vec_len_, tile_len_, vec_core_num_);

    if (num_tiles_to_process == 0) {
      return;
    }

    uint32_t offset_within_block = 0;
    uint32_t output_offset = CalculatePartsOffsets();

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const uint32_t global_offset =
          num_elems_before_block_ + offset_within_block;

      const uint32_t num_elems_to_process =
          scalar::NextTileLen(tile_len_, global_offset, vec_len_);

      if (num_elems_to_process == 0) {
        return;
      }

      LoadAndConvertMask(global_offset, num_elems_to_process);
      // TODO (anastasios): generate arithmetic progression here.
      LocalTensor<PosT> input_lt = vec_in_buf_.Get<PosT>();

      AscendC::ArithProgression<PosT>(input_lt, global_offset, 1,
                                      num_elems_to_process);
      // Gather ones.
      const uint32_t num_gathered_elems =
          GatherAndStore(input_lt, gathered_out_q_,
                         global_output_[output_offset], num_elems_to_process);

      output_offset += num_gathered_elems;
      offset_within_block += num_elems_to_process;
    }
  }

 private:
  __aicore__ inline uint32_t CalculatePartsOffsets() {
    uint32_t output_offset = 0;
    if (num_elems_before_block_ > 0) {
      copy::CopyGmToVec(pos_in_q_, global_pos_, vec_core_num_);
      LocalTensor<PosT> pos_lt = pos_in_q_.DeQue<PosT>();

      output_offset =
          reduce::ReduceScalarAdd(pos_lt, static_cast<uint32_t>(GetBlockIdx()));
      pos_in_q_.FreeTensor(pos_lt);
    }
    return output_offset;
  }

  __aicore__ inline void LoadAndConvertMask(uint32_t global_offset,
                                            uint32_t num_elems_to_process) {
    copy::CopyGmToVec(mask_in_q_, global_mask_[global_offset],
                      num_elems_to_process);
    LocalTensor<MaskT> mask_lt = mask_in_q_.DeQue<MaskT>();

    // Cast mask to fp16 since compare instructions require fp16 or fp32
    // input.
    const LocalTensor<half> mask_fp16_lt = mask_fp16_buf_.Get<half>();
    Duplicate(mask_fp16_lt, static_cast<half>(0), tile_len_);
    Cast(mask_fp16_lt, mask_lt, RoundMode::CAST_NONE, num_elems_to_process);
    mask_in_q_.FreeTensor(mask_lt);

    // Create a packed mask in uint8 datatype.
    const LocalTensor<PackedMaskT> packed_mask_8b =
        packed_mask_buf_.Get<PackedMaskT>();
    CompareScalar(packed_mask_8b, mask_fp16_lt, static_cast<half>(1),
                  CMPMODE::EQ, tile_len_);
  }

  template <typename GatherT, int32_t _NumBuf>
  __aicore__ inline uint32_t GatherAndStore(
      const LocalTensor<GatherT>& input_lt,
      TQue<QuePosition::VECOUT, _NumBuf>& out_q,
      const GlobalTensor<GatherT>& global, uint32_t num_elems_to_process) {
    using GatherMaskT = _GatherMaskT<GatherT>;

    uint64_t num_gathered_elems = 0;
    {
      const LocalTensor<GatherMaskT> mask_lt =
          packed_mask_buf_.Get<GatherMaskT>();
      const LocalTensor<GatherT> output_lt =
          out_q.template AllocTensor<GatherT>();

      GatherMask(output_lt, input_lt, mask_lt, true, num_elems_to_process,
                 {1, 1, 8, 8}, num_gathered_elems);
      out_q.EnQue(output_lt);
    }

    {
      LocalTensor<GatherT> output_lt = out_q.template DeQue<GatherT>();

      if (num_gathered_elems == 0) {
        out_q.FreeTensor(output_lt);
        return static_cast<uint32_t>(0);
      }

      DataCopyExtParams params;
      params.blockCount = 1;
      params.blockLen = num_gathered_elems * sizeof(GatherT);
      params.srcStride = 0;
      params.dstStride = 0;
      DataCopyPad(global, output_lt, params);

      out_q.FreeTensor(output_lt);
      return static_cast<uint32_t>(num_gathered_elems);
    }
  }

  TPipe pipe_;

  GlobalTensor<PosT> global_output_;
  GlobalTensor<MaskT> global_mask_;
  GlobalTensor<PosT> global_pos_;

  TQue<QuePosition::VECIN, 1> mask_in_q_;
  TQue<QuePosition::VECIN, 1> pos_in_q_;
  TBuf<QuePosition::VECCALC> vec_in_buf_;
  TBuf<QuePosition::VECCALC> mask_fp16_buf_;
  TBuf<QuePosition::VECCALC> packed_mask_buf_;
  TQue<QuePosition::VECOUT, 2> gathered_out_q_;

  const uint32_t vec_core_num_;
  const uint32_t vec_len_;
  const uint32_t tile_len_;
  const uint32_t mask_required_elems_;
  const uint32_t packed_mask_tile_size_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
  const uint32_t max_num_elems_per_block_;
  const uint32_t num_elems_before_block_;

  constexpr static uint16_t SMALLEST_MASK =
      tcuscan::UB_ALIGNMENT / sizeof(PackedMaskT);
  constexpr static uint16_t IN_ELEMS_PER_MASK_ELEM = sizeof(PackedMaskT) * 8;
};

/**
 * @brief Run the `where` kernel.
 *
 * @tparam T Input data type
 *
 * @param [in] mask_in Pointer to the input mask vector.
 * @param [in] num_ones_per_block Point to number of mask ones per block.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] vec_len Dimension of the input vector.
 * @param [in] tile_len Tile length.
 */
__aicore__ inline void run_where(GM_ADDR mask_in, GM_ADDR num_ones_per_block,
                                 GM_ADDR vec_out, uint32_t vec_len,
                                 uint32_t tile_len) {
  exec_mode::EnableCubeCores();

  if ASCEND_IS_AIV {
    KernelWhere op(vec_len, tile_len);
    op.Init(mask_in, num_ones_per_block, vec_out);
    op.Process();
  }
}

}  // namespace tcuscan
