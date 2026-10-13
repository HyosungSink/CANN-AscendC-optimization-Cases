/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file kernel_mc_gather.h
 * @brief Kernel implementing an AIV `mc_gather` operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Performs multi-core AIV gather operation given an 1D vector
`values_in` and an 1D indices vector `idx_in`. Returns

  z_out = values_in[x for x in idx_in] (in numpy notation).

  The output vector has length `len(idx_in)`.

  Assumptions:
  1) Input indices vector `idx_in` has indices in increasing order.

 * @tparam DataType Input data type of values.
 * @tparam ValueTileSize (default: 32768) represents the maximum size handled by
the LocalBuffer that contains fetched values.
 */
template <typename DataType, uint32_t ValueTileSize = 32768>
class KernelMcGather {
  constexpr static uint32_t BUFFER_NUM = 1;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] idx_in_len Length of the input index vector.
   * @param [in] values_in_len Length of the input values_in_len vector.
   * @param [in] tile_len Length of the tile processed in a single iteration.
   */

  __aicore__ inline KernelMcGather(uint32_t idx_in_len, uint32_t values_in_len,
                                   uint32_t tile_len)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        values_in_len_(values_in_len),
        idx_in_len_(idx_in_len),
        tile_len_(tile_len),
        num_tiles_idx_(scalar::CeilDiv(idx_in_len, tile_len_)),
        max_num_tiles_per_block_idx_(
            scalar::CeilDiv(num_tiles_idx_, vec_core_num_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] values_in  Pointer to input values vector.
   * @param [in] idx_in Pointer to input column indices vector.
   * @param [in] z_out Pointer to output z vector.
   */
  __aicore__ inline void Init(GM_ADDR values_in, GM_ADDR idx_in,
                              GM_ADDR z_out) {
    global_values_.SetGlobalBuffer((__gm__ DataType*)values_in, values_in_len_);
    global_idx_.SetGlobalBuffer((__gm__ uint32_t*)idx_in, idx_in_len_);
    global_z_.SetGlobalBuffer((__gm__ DataType*)z_out, idx_in_len_);

    pipe.InitBuffer(values_q_gather_, BUFFER_NUM,
                    ValueTileSize * sizeof(DataType));
    pipe.InitBuffer(idx_q_, BUFFER_NUM, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(output_q_, BUFFER_NUM, tile_len_ * sizeof(DataType));

    pipe.InitBuffer(threshold_up_buf_, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(threshold_down_buf_, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(mask_buf_, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(mask_up_buf_, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(mask_down_buf_, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(gathered_mask_buf_, tile_len_ * sizeof(uint32_t));
  }

  /**
   * @brief Run the kernel.
   *
   *  This kernel tiles on the indexes and fetches from memory only the needed
   * values, putting them in a LocalTensor of maximum size "values_q_gather_"
   */
  __aicore__ inline void Process() {
    uint32_t gm_offset =
        GetBlockIdx() * tile_len_ * max_num_tiles_per_block_idx_;

    const uint32_t num_idx_tiles_to_process =
        tcuscan::scalar::GetWorkDistribution(idx_in_len_, tile_len_,
                                             vec_core_num_);
    for (uint32_t tile_idx = 0; tile_idx < num_idx_tiles_to_process;
         tile_idx++) {
      const bool is_full_tile = gm_offset + tile_len_ <= idx_in_len_;
      const uint32_t this_tile_len =
          is_full_tile ? tile_len_ : idx_in_len_ - gm_offset;
      copy::CopyGmToVec(idx_q_, global_idx_[gm_offset], this_tile_len);
      ProcessTile(gm_offset, this_tile_len);
      gm_offset += this_tile_len;
    }
  }

  /**
   * @brief Process tile performs the gather reducing the col_idx fetched
   * from GM by 1
   *
   * @param [in] output_gm_offset  Global memory output writing offset.
   * @param [in] this_tile_len the length of the current tile (always the
   * same as tile_len_, except possibly for the last tile)
   */
  __aicore__ inline void ProcessTile(uint32_t output_gm_offset,
                                     uint32_t this_tile_len) {
    HandleMultipleTiles(output_gm_offset, this_tile_len);
  }

  /**
   * @brief Handle a single input tile of `idx_in`.
   *
   * Pre-condition / Assumption: largest index of `idx_lt` tile is within the
   * range [start, end]. More precisely, the following invariant holds:
   *
   *  start <= idx_lt[i] <= end for all i.
   *
   * @param idx_lt Input indices tile.
   * @param output_gm_offset Output GM offset to start writing gathered values
   * @param start Index to start reading values from `values_in` GM arrray.
   * @param end Last index of reading `values_in` GM entries.
   * @param this_tile_len current tile length of `idx_in` (always the
   * same as tile_len_, except possibly for the last tile)
   */
  __aicore__ inline void HandleSingleTile(LocalTensor<uint32_t>& idx_lt,
                                          uint32_t output_gm_offset,
                                          uint32_t start, uint32_t end,
                                          uint32_t this_tile_len) {
    // Length of `values_in` tile
    const uint32_t values_tile_len = end - start + 1;

    LocalTensor<DataType> z_lt = output_q_.AllocTensor<DataType>();

    // 1. Read values from GM starting from offset `start` with length
    // values_tile_len
    copy::CopyGmToVec(values_q_gather_, global_values_[start], values_tile_len);

    // 2. Gather values using the indices of `idx_lt`.
    // Assumption: largest index of `idx_lt` tile is within the bounds of values
    // tile.
    LocalTensor<DataType> fetched_values = values_q_gather_.DeQue<DataType>();
    GatherWithOffset(z_lt, fetched_values, idx_lt, start, this_tile_len);
    values_q_gather_.FreeTensor<DataType>(fetched_values);
    output_q_.EnQue<DataType>(z_lt);

    // 3. Write gathered values to output GM.
    copy::CopyVecToGm(global_z_[output_gm_offset], output_q_, this_tile_len);
  }

  /**
   * @brief Handle the general case (multiple tiles) that requires several
   * `value_in` subtiles of length `tile_len_` computed through masking, and a
   * final subtile, possibly smaller than tile_len_
   *
   * @param [in] output_gm Output GM index to be written with gathered values
   * @param [in] this_tile_len the length of the current tile (always the
   * same as tile_len_, except possibly for the last tile)
   */
  __aicore__ inline void HandleMultipleTiles(uint32_t output_gm,
                                             uint32_t this_tile_len) {
    LocalTensor<uint32_t> idx_lt = idx_q_.DeQue<uint32_t>();

    // Track the range of indices contained in the tile: [start, end]
    const uint32_t start = idx_lt.GetValue(0);
    const uint32_t end = idx_lt.GetValue(this_tile_len - 1);

    uint32_t new_start = start;
    uint32_t gathered_count = 0;

    // As long as the range of indices is larger than the `value_in` maximum
    // tile size, process the indices tile into chunks
    while (end - new_start >= ValueTileSize) {
      const uint32_t threshold_down = idx_lt.GetValue(gathered_count);
      const uint32_t threshold_up = threshold_down + ValueTileSize;
      uint64_t gathered_size = 0;

      // Filter Indexes -> find idx subtile
      LocalTensor<uint32_t> gathered_idx_lt = FilterTileInterval(
          idx_lt, threshold_up, threshold_down, gathered_size, this_tile_len);

      new_start = gathered_idx_lt.GetValue(0);
      const uint32_t subtile_size = static_cast<uint32_t>(gathered_size);
      const uint32_t local_end = gathered_idx_lt.GetValue(subtile_size - 1);

      // Here, we have made sure that the gathered_idx_lt contains indices that
      // are contained in the range [new_start, local_end].
      HandleSingleTile(gathered_idx_lt, output_gm, new_start, local_end,
                       subtile_size);

      ///////////////////////////////////////////////////////////
      // Update addresses and prepare for next chunk iteration
      ///////////////////////////////////////////////////////////
      output_gm = output_gm + subtile_size;
      // Keep track of the number of gathered values
      gathered_count += gathered_size;
      // Update the offset to the new smallest index to handle
      new_start = idx_lt.GetValue(gathered_count);
    }

    // Here, all indices are handled, free the index tile
    idx_q_.FreeTensor<uint32_t>(idx_lt);

    /////////////////////////////////////
    // Handle any last remaining tile. //
    /////////////////////////////////////
    const uint32_t partial_tile_len = this_tile_len - gathered_count;
    // Copy tail of indices from GM
    copy::CopyGmToVec(idx_q_, global_idx_[output_gm], partial_tile_len);

    LocalTensor<uint32_t> sub_idx_lt = idx_q_.DeQue<uint32_t>();
    HandleSingleTile(sub_idx_lt, output_gm, new_start, end, partial_tile_len);
    idx_q_.FreeTensor<uint32_t>(sub_idx_lt);
  }

  /**
   * @brief Filters an input tensor based on a upper and lower threshold
   * values. The output is a sub_tensor of the input tensor containing
   * only values in the specified interval. The number of returned
   * elements is stored in the output parameter @p gathered_size.
   *
   * Example:
   *
   * Threshold_up 17
   * Threshold_down 10
   *
   * Input: 1 3 5 7 9 10 11 12 14 17 19 20
   *
   * Output: 11 12 14
   *
   *
   * @param idx_lt The input tensor whose elements are to be filtered.
   * @param threshold_up The upper value; elements must be lower than
   * this value.
   * @param threshold_down The lower value; elements must be bigger than
   * this value.
   * @param gathered_len A reference to a variable where the count of
   * (gathered) elements will be stored.
   * @param this_tile_len the length of the current tile (always the
   * same as tile_len_, except possibly for the last tile)
   * @return LocalTensor<uint32_t> A tensor containing the indices of elements
   * that meet the filtering criteria.
   */
  __aicore__ inline LocalTensor<uint32_t> FilterTileInterval(
      LocalTensor<uint32_t> idx_lt, uint32_t threshold_up,
      uint32_t threshold_down, uint64_t& gathered_len, uint32_t this_tile_len) {
    LocalTensor<uint32_t> threshold_up_lt = threshold_up_buf_.Get<uint32_t>();
    LocalTensor<uint32_t> threshold_down_lt =
        threshold_down_buf_.Get<uint32_t>();
    LocalTensor<uint8_t> mask_up_lt = mask_up_buf_.Get<uint8_t>();
    LocalTensor<uint8_t> mask_down_lt = mask_down_buf_.Get<uint8_t>();
    LocalTensor<uint16_t> mask_lt = mask_buf_.Get<uint16_t>();

    // Fill with threshold up/down the local tensors
    Duplicate<uint32_t>(threshold_up_lt, threshold_up, this_tile_len);
    Duplicate<uint32_t>(threshold_down_lt, threshold_down, this_tile_len);

    if (this_tile_len < tile_len_) {
      // If current tile length is small, make sure mask_[up/down]_lt are
      // zeroed, before applying AscendC::Compare
      Duplicate<uint16_t>(mask_up_lt.template ReinterpretCast<uint16_t>(), 0,
                          tile_len_);
      Duplicate<uint16_t>(mask_down_lt.template ReinterpretCast<uint16_t>(), 0,
                          tile_len_);
      Duplicate<uint16_t>(mask_lt, 0, tile_len_);
    }

    Compare(mask_up_lt, idx_lt.template ReinterpretCast<float>(),
            threshold_up_lt.template ReinterpretCast<float>(), CMPMODE::LT,
            tile_len_);  // Compares needs to handle the whole tile_len_. The
                         // AscendC::And below takes care of partial tiles.

    Compare(mask_down_lt, idx_lt.template ReinterpretCast<float>(),
            threshold_down_lt.template ReinterpretCast<float>(), CMPMODE::GE,
            tile_len_);  /// Compares needs to handle the whole tile_len_. The
                         /// AscendC::And below takes care of partial tiles.

    And(mask_lt, mask_up_lt.template ReinterpretCast<uint16_t>(),
        mask_down_lt.template ReinterpretCast<uint16_t>(), this_tile_len);

    LocalTensor<uint32_t> gathered_idx_lt = gathered_mask_buf_.Get<uint32_t>();
    GatherMask(gathered_idx_lt, idx_lt,
               mask_lt.template ReinterpretCast<uint32_t>(), true,
               this_tile_len, {1, 1, 8, 8}, gathered_len);

    return gathered_idx_lt;
  }

  /**
   * @brief Gathers elements from an input tensor using an index tensor
   * with an offset adjustment.
   *
   * @tparam U The type of the elements in the input and output
   * tensors.
   * @param out_tensor Output tensor that will contain the gather values.
   * @param in_values The local tensor from which elements will be
   * gathered.
   * @param idx_lt The local tensor containing indexes that will
   be adjusted and used for gathering.

   * @param offset The offset value to subtract from each index.
   * @param num_elem_to_process The number of elements to process.
   */
  __aicore__ inline void GatherWithOffset(LocalTensor<DataType>& out_tensor,
                                          LocalTensor<DataType> in_values,
                                          LocalTensor<uint32_t> idx_lt,
                                          int32_t offset,
                                          uint64_t num_elem_to_process) {
    LocalTensor<int32_t> idx_int32_lt =
        idx_lt.template ReinterpretCast<int32_t>();

    // Substract offset since gather must be done in local/tile "coordinates".
    AscendC::Adds(idx_int32_lt, idx_int32_lt, -offset, num_elem_to_process);
    // Multiple indices by sizeof(DataType) since Gather works on bytes
    AscendC::Muls(idx_int32_lt, idx_int32_lt,
                  static_cast<int32_t>(sizeof(DataType)), num_elem_to_process);
    // Perform the AscendC::Gather
    AscendC::Gather(out_tensor, in_values,
                    idx_int32_lt.template ReinterpretCast<uint32_t>(),
                    static_cast<uint32_t>(0), num_elem_to_process);
  }

 private:
  TPipe pipe;

  TQue<QuePosition::VECIN, BUFFER_NUM> values_q_gather_;
  TQue<QuePosition::VECIN, BUFFER_NUM> idx_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> output_q_;
  TBuf<QuePosition::VECCALC> threshold_up_buf_;
  TBuf<QuePosition::VECCALC> threshold_down_buf_;
  TBuf<QuePosition::VECCALC> mask_buf_;
  TBuf<QuePosition::VECCALC> mask_up_buf_;
  TBuf<QuePosition::VECCALC> mask_down_buf_;
  TBuf<QuePosition::VECCALC> gathered_mask_buf_;

  GlobalTensor<DataType> global_values_;
  GlobalTensor<uint32_t> global_idx_;
  GlobalTensor<DataType> global_z_;

  const uint32_t vec_core_num_;
  const uint32_t values_in_len_;
  const uint32_t idx_in_len_;
  const uint32_t tile_len_;
  const uint32_t num_tiles_idx_;
  const uint32_t max_num_tiles_per_block_idx_;
};

/**
 * @brief Run the `mc_gather` kernel.
 *
 * @tparam T Input data type.
 *
 * @param [in] values_in Pointer to input vector.
 * @param [in] idx_in Pointer to column indices input vector.
 * @param [in] z_out Pointer to output vector.
 * @param [in] idx_in_len length of the indices array
 * @param [in] val_in_len length of the values array
 * @param [in] tile_len Length of the tile processed in a single iteration.
 */

template <typename T>
__aicore__ inline void run_mc_gather(GM_ADDR values_in, GM_ADDR idx_in,
                                     GM_ADDR z_out, uint32_t idx_in_len,
                                     uint32_t val_in_len, uint32_t tile_len) {
  if ASCEND_IS_AIV {
    KernelMcGather<T> op(idx_in_len, val_in_len, tile_len);
    op.Init(values_in, idx_in, z_out);
    op.Process();
  }
}

}  // namespace tcuscan
