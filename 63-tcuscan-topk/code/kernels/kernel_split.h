/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_split.h
 * @brief Kernel implementing a split operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "kernel_arithmetic_progression.h"
#include "kernel_scan_multi_core.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Splits the input vector into two parts based on the binary mask.
 *
 * Kernel splits the input vector into two parts based on the binary mask. If
 * `zeros_first` is set, the elements for which the mask is `0` are followed by
 * the elements for which the mask is `1` in the output. If `zeros_first` is not
 * set, the elements with `1` come first. The split is stable, so the relative
 * order between the elements with the same value in mask is consistent between
 * the input and the output.
 *
 * If `WithIndices` is set to `true`, kernel also takes and splits the vector of
 * indices which data type is expected to be `int32_t`.
 *
 * Kernel also takes a position vector on the input which is an output of
 * inclusive scan on the binary input mask.
 *
 * The algorithm expects the mask datatype to be `int8_t` or `uint8_t`, the
 * positions to be `int32_t` and the input vector's elements to be at least 2
 * bytes.
 *
 * @tparam T Data type of the input vector.
 * @tparam WithIndices Indicates whether to calculate indices or not.
 */
template <typename T, bool WithIndices>
class KernelSplit {
  using MaskT = uint8_t;
  using PosT = int32_t;

  using PackedMaskT = uint8_t;

  using IndicesT = int32_t;

  template <typename _InputT>
  using _GatherMaskT =
      typename std::conditional<sizeof(_InputT) == 2, uint16_t, uint32_t>::type;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Number of elements in an input vector.
   * @param [in] tile_len Tile length.
   * @param [in] zeros_first Indicates whether the first elements in the
   * output vector are the ones with corresponding mask value set to zero or
   * one.
   */
  __aicore__ inline KernelSplit(uint32_t vec_len, uint32_t tile_len,
                                bool zeros_first)
      : vec_len_(vec_len),
        block_num_(GetBlockNum() * GetTaskRation()),
        tile_len_(tile_len),
        zeros_first_(zeros_first),
        mask_required_elems_(tile_len_ / IN_ELEMS_PER_MASK_ELEM),
        packed_mask_tile_size_(mask_required_elems_ < SMALLEST_MASK
                                   ? SMALLEST_MASK
                                   : mask_required_elems_),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_len_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, block_num_)),
        max_num_elems_per_block_(tile_len_ * max_num_tiles_per_block_),
        num_elems_before_block_(GetBlockIdx() * max_num_elems_per_block_) {
    // GatherMask requires either 2- or 4-byte elements
    static_assert(sizeof(T) >= 2);
  }

 private:
  __aicore__ inline void InitBase(GM_ADDR input, GM_ADDR mask, GM_ADDR pos,
                                  GM_ADDR output) {
    global_input_.SetGlobalBuffer((__gm__ T*)input, vec_len_);
    global_output_.SetGlobalBuffer((__gm__ T*)output, vec_len_);
    global_mask_.SetGlobalBuffer((__gm__ MaskT*)mask, vec_len_);
    global_pos_.SetGlobalBuffer((__gm__ PosT*)pos, block_num_);

    pipe.InitBuffer(vec_in_q_, 1, tile_len_ * sizeof(T));
    pipe.InitBuffer(mask_in_q_, 1, tile_len_ * sizeof(MaskT));
    pipe.InitBuffer(mask_fp16_buf_, tile_len_ * sizeof(half));
    pipe.InitBuffer(packed_mask_buf_,
                    packed_mask_tile_size_ * sizeof(PackedMaskT));
    pipe.InitBuffer(gathered_out_q_, 2, tile_len_ * sizeof(T));
    pipe.InitBuffer(pos_in_q_, 1, block_num_ * sizeof(PosT));
  }

 public:
  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input Pointer to input vector in global memory.
   * @param [in] mask Pointer to mask vector in global memory.
   * @param [in] pos Pointer to position vector in global memory.
   * @param [in] output Pointer to output vector in global memory.
   */
  template <bool _WithIndices = WithIndices,
            typename std::enable_if<!_WithIndices, int>::type = 0>
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR mask, GM_ADDR pos,
                              GM_ADDR output) {
    InitBase(input, mask, pos, output);
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input Pointer to input vector in global memory.
   * @param [in] mask Pointer to mask vector in global memory.
   * @param [in] pos Pointer to position vector in global memory.
   * @param [in] output Pointer to output vector in global memory.
   * @param [in] indices_in Pointer to the input indices vector in global
   * memory.
   * @param [in] indices_out Pointer to the output indices vector in global
   * memory.
   */
  template <bool _WithIndices = WithIndices,
            typename std::enable_if<_WithIndices, int>::type = 0>
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR mask, GM_ADDR pos,
                              GM_ADDR output, GM_ADDR indices_in,
                              GM_ADDR indices_out) {
    InitBase(input, mask, pos, output);

    global_ind_in_.SetGlobalBuffer((__gm__ IndicesT*)indices_in, vec_len_);
    global_ind_out_.SetGlobalBuffer((__gm__ IndicesT*)indices_out, vec_len_);
    pipe.InitBuffer(ind_in_q_, 1, tile_len_ * sizeof(IndicesT));
    pipe.InitBuffer(gathered_ind_q_, 2, tile_len_ * sizeof(IndicesT));
  }

  /**
   * @brief Run the kernel.
   */
  template <bool _WithIndices = WithIndices,
            typename std::enable_if<!_WithIndices, int>::type = 0>
  __aicore__ inline void Process() {
    const uint32_t num_tiles_to_process =
        tcuscan::scalar::GetWorkDistribution(vec_len_, tile_len_, block_num_);

    uint32_t offset_within_block = 0;
    uint32_t zeros_offset, ones_offset;
    CalculatePartsOffsets(num_elems_before_block_, zeros_offset, ones_offset);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const uint32_t global_offset =
          num_elems_before_block_ + offset_within_block;
      const uint32_t num_elems =
          scalar::NextTileLen(tile_len_, global_offset, vec_len_);

      LoadAndConvertMask(global_offset, num_elems);
      copy::CopyGmToVec(vec_in_q_, global_input_[global_offset], num_elems);

      LocalTensor<T> input_lt = vec_in_q_.DeQue<T>();

      // Gather ones.
      const uint32_t num_gathered_ones = GatherAndStore(
          input_lt, gathered_out_q_, global_output_[ones_offset], num_elems);
      ones_offset += num_gathered_ones;

      NegateMask();

      // Gather zeros.
      const uint32_t num_gathered_zeros = GatherAndStore(
          input_lt, gathered_out_q_, global_output_[zeros_offset], num_elems);
      zeros_offset += num_gathered_zeros;

      vec_in_q_.FreeTensor(input_lt);
      offset_within_block += num_elems;
    }
  }

  /**
   * @brief Run the kernel.
   */
  template <bool _WithIndices = WithIndices,
            typename std::enable_if<_WithIndices, int>::type = 0>
  __aicore__ inline void Process() {
    const uint32_t num_tiles_to_process =
        tcuscan::scalar::GetWorkDistribution(vec_len_, tile_len_, block_num_);

    uint32_t offset_within_block = 0;
    uint32_t zeros_offset, ones_offset;
    CalculatePartsOffsets(num_elems_before_block_, zeros_offset, ones_offset);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const uint32_t global_offset =
          num_elems_before_block_ + offset_within_block;

      const uint32_t num_elems =
          scalar::NextTileLen(tile_len_, global_offset, vec_len_);

      LoadAndConvertMask(global_offset, num_elems);

      copy::CopyGmToVec(vec_in_q_, global_input_[global_offset], num_elems);
      copy::CopyGmToVec(ind_in_q_, global_ind_in_[global_offset], num_elems);

      LocalTensor<T> input_lt = vec_in_q_.DeQue<T>();
      LocalTensor<IndicesT> ind_input_lt = ind_in_q_.DeQue<IndicesT>();
      // Gather ones
      const uint32_t num_gathered_ones = GatherAndStore(
          input_lt, gathered_out_q_, global_output_[ones_offset], num_elems);
      (void)GatherAndStore(ind_input_lt, gathered_ind_q_,
                           global_ind_out_[ones_offset], num_elems);
      ones_offset += num_gathered_ones;

      NegateMask();
      // Gather zeros.
      const uint32_t num_gathered_zeros = GatherAndStore(
          input_lt, gathered_out_q_, global_output_[zeros_offset], num_elems);
      (void)GatherAndStore(ind_input_lt, gathered_ind_q_,
                           global_ind_out_[zeros_offset], num_elems);
      zeros_offset += num_gathered_zeros;

      vec_in_q_.FreeTensor(input_lt);
      ind_in_q_.FreeTensor(ind_input_lt);
      offset_within_block += num_elems;
    }
  }

 private:
  __aicore__ inline void CalculatePartsOffsets(uint32_t global_offset,
                                               uint32_t& zeros_offset,
                                               uint32_t& ones_offset) {
    const uint32_t num_elems_before_tile = global_offset;
    uint32_t num_ones_before_tile = 0;

    copy::CopyGmToVec(pos_in_q_, global_pos_, block_num_);
    LocalTensor<PosT> pos_lt = pos_in_q_.DeQue<PosT>();
    if (GetBlockIdx() > 0)
      num_ones_before_tile =
          reduce::ReduceScalarAdd(pos_lt, static_cast<uint32_t>(GetBlockIdx()));
    const uint32_t total_num_ones =
        reduce::ReduceScalarAdd(pos_lt, static_cast<uint32_t>(block_num_));
    const uint32_t total_num_zeros = vec_len_ - total_num_ones;

    pos_in_q_.FreeTensor(pos_lt);

    const uint32_t num_zeros_before_tile =
        num_elems_before_tile - num_ones_before_tile;

    if (zeros_first_) {
      zeros_offset = num_zeros_before_tile;
      ones_offset = total_num_zeros + num_ones_before_tile;
    } else {
      zeros_offset = total_num_ones + num_zeros_before_tile;
      ones_offset = num_ones_before_tile;
    }
  }

  __aicore__ inline void LoadAndConvertMask(uint32_t global_offset,
                                            uint32_t num_elems) {
    copy::CopyGmToVec(mask_in_q_, global_mask_[global_offset], num_elems);
    LocalTensor<MaskT> mask_lt = mask_in_q_.DeQue<MaskT>();

    // Cast mask to fp16 since compare instructions require fp16 or fp32
    // input.
    // Load a whole mask tile even if not a full tile, GatherMask will
    // ignore the uninitialized part
    const LocalTensor<half> mask_fp16_lt = mask_fp16_buf_.Get<half>();
    // The assumption here is that if the output tensor is larger than
    // the input tensor, the operation will modify only some part of the
    // output tensor and there is no undefined behavior.
    Cast(mask_fp16_lt, mask_lt, RoundMode::CAST_NONE, mask_lt.GetSize());
    mask_in_q_.FreeTensor(mask_lt);

    // Create a packed mask in uint8 datatype.
    const LocalTensor<PackedMaskT> packed_mask_8b =
        packed_mask_buf_.Get<PackedMaskT>();
    CompareScalar(packed_mask_8b, mask_fp16_lt, static_cast<half>(1),
                  CMPMODE::EQ, mask_fp16_lt.GetSize());
  }

  template <typename GatherT, int32_t _NumBuf>
  __aicore__ inline uint32_t GatherAndStore(
      const LocalTensor<GatherT>& input_lt,
      TQue<QuePosition::VECOUT, _NumBuf>& out_q,
      const GlobalTensor<GatherT>& global, uint32_t num_elems) {
    using GatherMaskT = _GatherMaskT<GatherT>;

    uint64_t num_gathered_elems = 0;
    {
      const LocalTensor<GatherMaskT> mask_lt =
          packed_mask_buf_.Get<GatherMaskT>();
      const LocalTensor<GatherT> output_lt =
          out_q.template AllocTensor<GatherT>();

      GatherMask(output_lt, input_lt, mask_lt, true, num_elems, {1, 1, 8, 8},
                 num_gathered_elems);
      out_q.EnQue(output_lt);
    }

    {
      LocalTensor<GatherT> output_lt = out_q.template DeQue<GatherT>();
      if (num_gathered_elems == 0) {
        out_q.FreeTensor(output_lt);
        return 0;
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

  __aicore__ inline void NegateMask() {
    const LocalTensor<uint16_t> mask_lt = packed_mask_buf_.Get<uint16_t>();
    Not(mask_lt, mask_lt, mask_lt.GetSize());
  }

  TPipe pipe;

  GlobalTensor<T> global_input_;
  GlobalTensor<T> global_output_;
  GlobalTensor<MaskT> global_mask_;
  GlobalTensor<PosT> global_pos_;

  GlobalTensor<IndicesT> global_ind_in_;
  GlobalTensor<IndicesT> global_ind_out_;

  TQue<QuePosition::VECIN, 1> vec_in_q_;
  TQue<QuePosition::VECIN, 1> mask_in_q_;
  TQue<QuePosition::VECIN, 1> pos_in_q_;
  TBuf<QuePosition::VECCALC> mask_fp16_buf_;
  TBuf<QuePosition::VECCALC> packed_mask_buf_;
  TQue<QuePosition::VECOUT, 2> gathered_out_q_;

  TQue<QuePosition::VECIN, 1> ind_in_q_;
  TQue<QuePosition::VECOUT, 2> gathered_ind_q_;

  const uint32_t vec_len_;
  const uint32_t block_num_;
  const uint32_t tile_len_;
  const bool zeros_first_;
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

namespace split {

/**
 * @brief Calculate the workspace size for split.
 *
 * @tparam InputT Input data type.
 * @return Size of the workspace in bytes.
 */
__aicore__ inline uint32_t get_workspace_size() {
  return scalar::AlignUp(GetBlockNum() * GetTaskRation() * sizeof(uint32_t),
                         UB_ALIGNMENT);
}

}  // namespace split

/**
 * @brief Computes the reduction of a small vector.
 *
 * @tparam T Data type of the input vector.
 */
template <typename T>
class KernelReduce {
 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Number of elements in an input vector.
   */
  __aicore__ inline KernelReduce(uint32_t vec_len) : vec_len_(vec_len) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input Pointer to input vector in global memory.
   * @param [in] output Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
    global_input_.SetGlobalBuffer((__gm__ T*)input, vec_len_);
    global_output_.SetGlobalBuffer((__gm__ T*)output, vec_len_);
    pipe.InitBuffer(vec_in_q_, 1, vec_len_ * sizeof(T));
    pipe.InitBuffer(vec_out_q_, 1, sizeof(T));
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    if (GetBlockIdx() != 0) return;

    copy::CopyGmToVec(vec_in_q_, global_input_, vec_len_);
    LocalTensor<T> input_lt = vec_in_q_.DeQue<T>();
    T sum;
    if constexpr (reduce::IsAscendReduceSumSupported<T>) {
      ReduceSum(input_lt, input_lt, input_lt, vec_len_);
      sum = AscendC::GetAccVal<T>();
    } else {
      sum = reduce::ReduceScalarAdd<T>(input_lt, input_lt.GetSize());
    }
    vec_in_q_.FreeTensor(input_lt);

    copy::CopyScalarToGm(global_output_, vec_out_q_, sum);
  }

 private:
  TPipe pipe;

  GlobalTensor<T> global_input_;
  GlobalTensor<T> global_output_;

  TQue<QuePosition::VECIN, 1> vec_in_q_;
  TQue<QuePosition::VECOUT, 1> vec_out_q_;

  const uint32_t vec_len_;
};

/**
 * @brief Run the `split_uint16` kernel.
 *
 * @param [in] in Pointer to input vector.
 * @param [in] mask Pointer to mask vector.
 * @param [in] out Pointer to output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] vec_len Length of the input vector.
 * @param [in] tile_len Length of the tile processed in a single
 * iteration.
 * @param [in] zeros_first Indicates whether the first elements in the output
 * vector are the ones with corresponding mask value set to zero or one.
 */
__aicore__ inline void run_split_uint16(GM_ADDR in, GM_ADDR mask, GM_ADDR out,
                                        GM_ADDR workspace, uint32_t vec_len,
                                        uint32_t tile_len, bool zeros_first) {
  exec_mode::EnableCubeCores();

  GM_ADDR const sums = workspace;

  if ASCEND_IS_AIV {
    KernelReduceTiles<int8_t> op_reduce(tile_len, vec_len);
    op_reduce.Init(mask, sums);
    op_reduce.Process();
  }

  SyncAll<true /*isAIVOnly*/>();

  if ASCEND_IS_AIV {
    constexpr bool with_indices = false;
    KernelSplit<uint16_t, with_indices> op(vec_len, tile_len, zeros_first);
    op.Init(in, mask, sums, out);
    op.Process();
  }
}

/**
 * @brief Run the `split_ind_uint16` kernel.
 *
 * @param [in] in Pointer to input vector.
 * @param [in] mask Pointer to mask vector.
 * @param [in] indices_in Pointer to the input indices vector.
 * @param [in] out Pointer to output vector.
 * @param [in] indices_out Pointer to the output indices vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] vec_len Length of the input vector.
 * @param [in] tile_len Length of the tile processed in a single
 * iteration.
 * @param [in] zeros_first Indicates whether the first elements in the output
 * vector are the ones with corresponding mask value set to zero or one.
 * @return Number of ones in the mask.
 */
__aicore__ inline uint32_t run_split_ind_uint16(
    GM_ADDR in, GM_ADDR mask, GM_ADDR indices_in, GM_ADDR out,
    GM_ADDR indices_out, GM_ADDR workspace, uint32_t vec_len, uint32_t tile_len,
    bool zeros_first) {
  exec_mode::EnableCubeCores();

  GM_ADDR const sums = workspace;

  if ASCEND_IS_AIV {
    KernelReduceTiles<int8_t> op_reduce_tiles(tile_len, vec_len);
    op_reduce_tiles.Init(mask, sums);
    op_reduce_tiles.Process();
  }

  SyncAll<false /*isAIVOnly*/>();

  if ASCEND_IS_AIV {
    constexpr bool with_indices = true;
    KernelSplit<uint16_t, with_indices> op(vec_len, tile_len, zeros_first);
    op.Init(in, mask, sums, out, indices_in, indices_out);
    op.Process();
  }

  SyncAll<false /*isAIVOnly*/>();

  if ASCEND_IS_AIV {
    KernelReduce<uint32_t> op_reduce(GetBlockNum());
    op_reduce.Init(sums, sums);
    op_reduce.Process();
  }

  SyncAll<false /*isAIVOnly*/>();

  return scalar::GetGMValue<int32_t>(sums, 0, GetBlockNum());
}

}  // namespace tcuscan
