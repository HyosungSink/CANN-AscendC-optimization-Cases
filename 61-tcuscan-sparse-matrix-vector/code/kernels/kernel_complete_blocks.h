/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * @file kernel_complete_blocks.h
 * @brief Kernel implementing the broadcast scalar-vector addition step of
 * multi-core scans, known as down-sweep phase.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Finish the scan operation based on block-wise partial scans and
 * reduced blocks.
 *
 * This kernel takes outputs of `KernelBlockScan` and `KernelReduceTiles` and
 * finishes calculating the entire scan of the vector.
 *
 * Because of the limited vector core's support for data types, the input data
 * must be `half`, `int16_t`, `float` or `int32_t`.
 *
 * @tparam T Data type of the input and output vectors.
 * @tparam IsInclusive Indicates whether the scan is inclusive or exclusive.
 * @tparam BufferNum Number of buffers. If set to `2`, double-buffering is
 * enabled.
 */
template <typename T, bool IsInclusive = true, int32_t BufferNum = 2>
class KernelCompleteBlocks {
  constexpr static int32_t MIN_VEC_SIZE = UB_ALIGNMENT / sizeof(T);
  constexpr static uint32_t global_shift_ = IsInclusive ? 0 : 1;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Number of elements in an input vector.
   * @param [in] block_scan_len Length of block scans that are pre-computed
   * using `KernelBlockScan` or `KernelRowScan`.
   * @param [in] tile_len Length of kernel tiles.
   */
  __aicore__ inline KernelCompleteBlocks(uint32_t vec_len,
                                         uint32_t block_scan_len,
                                         uint32_t tile_len)
      : block_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        sums_len_(vec_len / block_scan_len),
        block_scan_len_(block_scan_len),
        tile_len_(tile_len),
        output_real_elems_(IsInclusive ? vec_len_ : vec_len_ - 1),
        max_num_tiles_per_block_(
            scalar::CeilDiv(vec_len_, tile_len_ * block_num_)),
        num_tiles_per_matrix_tile_(block_scan_len / tile_len) {
    constexpr bool IS_DT_SUPPORTED =
        std::is_same_v<T, float> || std::is_same_v<T, int32_t>;
    static_assert(IS_DT_SUPPORTED, "Unsupported data type.");
    ASCENDC_ASSERT((vec_len % block_scan_len == 0), {
      KERNEL_LOG(KERNEL_ERROR,
                 "Input length (%d) must be "
                 "divisible by the block scan length (%d)",
                 vec_len, block_scan_len);
    });

    ASCENDC_ASSERT((block_scan_len % tile_len == 0), {
      KERNEL_LOG(KERNEL_ERROR,
                 "Block scan length (%d) must be "
                 "divisible by tile length (%d)",
                 block_scan_len, tile_len);
    });
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input_rows Pointer to input vector in global memory.
   * @param [in] sums Pointer to vector with partial sums in global
   * memory.
   * @param [in] output Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR input_rows, GM_ADDR sums,
                              GM_ADDR output) {
    global_input_rows_.SetGlobalBuffer((__gm__ T*)input_rows, vec_len_);
    global_sums_.SetGlobalBuffer((__gm__ T*)sums, sums_len_);

    if constexpr (!IsInclusive) {
      // The very first element of the output where a zero value must go
      // in case of exclusive scan.
      global_output_first_elem_.SetGlobalBuffer((__gm__ T*)output, 1);
    }
    global_output_.SetGlobalBuffer((__gm__ T*)output + global_shift_,
                                   output_real_elems_);

    pipe.InitBuffer(vec_tile_in_q_, BufferNum, tile_len_ * sizeof(T));
    pipe.InitBuffer(vec_tile_out_q_, BufferNum, tile_len_ * sizeof(T));

    pipe.InitBuffer(sums_q_, 1, AlignUp(block_num_, MIN_VEC_SIZE) * sizeof(T));
  }

  /**
   * @brief Run the kernel - process all tiles.
   *
   * @param [in] starting_value Starting value added to all entries of the
   * output vector.
   */
  __aicore__ inline void Process(T starting_value = 0) {
    const T previous_sum = ScanReducedBlocks() + starting_value;
    AddSumsToMatmulTiles(previous_sum);
    if constexpr (!IsInclusive) {
      if (GetBlockIdx() == 0) {
        global_output_first_elem_(0) = starting_value;
        DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE,
                                 DcciDst::CACHELINE_OUT>(
            global_output_first_elem_);
      }
    }
  }

 private:
  /**
   * @brief Performs a scan on the block reduced values that is the output of
   * `KernelReduceTiles`.
   *
   * @return Returns the prefix sum value of index `GetBlockIdx()`.
   */
  __aicore__ inline T ScanReducedBlocks() {
    // Reduce the sums of all the previous tiles.
    copy::CopyGmToVec(sums_q_, global_sums_, sums_len_);
    LocalTensor<T> sums_lt = sums_q_.DeQue<T>();
    const T previous_sum = reduce::ReduceScalarAdd(sums_lt, GetBlockIdx());
    sums_q_.FreeTensor(sums_lt);
    return previous_sum;
  }

  __aicore__ inline void AddSumsToMatmulTiles(T previous_sum) {
    uint32_t global_offset =
        GetBlockIdx() * tile_len_ * max_num_tiles_per_block_;
    const uint32_t num_tiles_to_process =
        scalar::GetWorkDistribution(vec_len_, tile_len_, block_num_);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const bool full_tile = global_offset + tile_len_ <= output_real_elems_;
      const uint32_t num_elems_to_process =
          full_tile ? tile_len_ : output_real_elems_ - global_offset;

      copy::CopyGmToVec(vec_tile_in_q_, global_input_rows_[global_offset],
                        num_elems_to_process);

      const bool is_last_tile_in_block =
          (tile_idx + 1) % num_tiles_per_matrix_tile_ == 0;

      VectorAdds(previous_sum, num_elems_to_process, is_last_tile_in_block);

      copy::CopyVecToGm(global_output_[global_offset], vec_tile_out_q_,
                        num_elems_to_process);

      global_offset += tile_len_;
    }
  }

  __aicore__ inline void VectorAdds(T& block_level_running_sum,
                                    uint32_t num_elems_to_process,
                                    bool is_last_tile_in_block) {
    LocalTensor<T> vec_lt = vec_tile_in_q_.template DeQue<T>();
    const LocalTensor<T> vec_out_lt = vec_tile_out_q_.template AllocTensor<T>();
    DataCopy(vec_out_lt, vec_lt, vec_lt.GetSize());
    vec_tile_in_q_.FreeTensor(vec_lt);

    Adds(vec_out_lt, vec_out_lt, block_level_running_sum, num_elems_to_process);

    // If tile is the last in the block scan, update the running_sum value
    if (is_last_tile_in_block) {
      block_level_running_sum = vec_out_lt.GetValue(num_elems_to_process - 1);
    }
    vec_tile_out_q_.EnQue(vec_out_lt);
  }

  TPipe pipe;

  TQue<QuePosition::VECIN, BufferNum> vec_tile_in_q_;
  TQue<QuePosition::VECOUT, BufferNum> vec_tile_out_q_;
  TQue<QuePosition::VECIN, 1> sums_q_;

  GlobalTensor<T> global_input_rows_;
  GlobalTensor<T> global_sums_;
  GlobalTensor<T> global_output_;
  GlobalTensor<T> global_output_first_elem_;

  const uint32_t block_num_;
  const uint32_t vec_len_;
  const uint32_t sums_len_;
  const uint32_t block_scan_len_;
  const uint32_t tile_len_;
  const uint32_t output_real_elems_;
  const uint32_t max_num_tiles_per_block_;
  const uint32_t num_tiles_per_matrix_tile_;
};

}  // namespace tcuscan
