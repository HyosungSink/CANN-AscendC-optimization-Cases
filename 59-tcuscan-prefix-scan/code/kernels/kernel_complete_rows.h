/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_complete_rows.h
 * @brief Kernel implementing the down-sweep scan of broadcast add per block.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Finish the scan operation based on row-wise partial scans and reduced
 * tiles.
 *
 * This kernel takes outputs of `KernelRowScan` and `KernelReduceTiles` and
 * finishes calculating the entire scan of the vector.
 *
 * Because of the limited vector core's support for data types, the input data
 * must be `half`, `int16_t`, `float` or `int32_t`.
 *
 * @tparam T Data type of the input and output vectors.
 * @tparam IsInclusive Indicates whether the scan is inclusive or exclusive.
 */
template <typename T, bool IsInclusive = true>
class KernelCompleteRows {
  constexpr static int32_t MIN_VEC_SIZE = UB_ALIGNMENT / sizeof(T);
  /// @brief Maximum allowed tile size
  constexpr static uint32_t MAX_TILE_SIZE = 8192;
  /// @brief Double buffering is enabled for half/fp16
  constexpr static uint32_t BUFFER_NUM = std::is_same_v<T, float> ? 1 : 2;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] tile_width Size of the row used by `KernelRowScan`.
   * @param [in] tile_height Number of rows processed in a single iteration.
   * @param [in] vec_len Number of elements in an input vector.
   */
  __aicore__ inline KernelCompleteRows(uint32_t tile_width,
                                       uint32_t tile_height, uint32_t vec_len)
      : block_num_(GetBlockNum() * GetTaskRation()),
        tile_width_(tile_width),
        tile_height_(scalar::Min(tile_height,
                                 scalar::CeilDiv(MAX_TILE_SIZE, tile_width_))),
        tile_size_(tile_width_ * tile_height_),
        sums_len_(block_num_),
        vec_len_(vec_len),
        output_real_elems_(IsInclusive ? vec_len_ : vec_len_ - 1),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_size_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, block_num_)) {
    constexpr bool IS_DT_SUPPORTED =
        std::is_same_v<T, half> || std::is_same_v<T, int16_t> ||
        std::is_same_v<T, float> || std::is_same_v<T, int32_t>;
    static_assert(IS_DT_SUPPORTED, "Unsupported data type.");
    ASCENDC_ASSERT((MAX_TILE_SIZE % tile_width_ == 0), {
      KERNEL_LOG(KERNEL_ERROR,
                 "MAX_TILE_SIZE (8192) must be "
                 "divisible by the input tile width (%d)",
                 tile_width_);
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
    // The very first element of the output where a zero value must go
    // in case of exclusive scan.
    global_output_first_elem_.SetGlobalBuffer((__gm__ T*)output, 1);
    global_output_.SetGlobalBuffer((__gm__ T*)output + global_shift_,
                                   output_real_elems_);

    pipe.InitBuffer(vec_tile_q_, BUFFER_NUM, tile_size_ * sizeof(T));
    pipe.InitBuffer(vec_tile_out_q_, BUFFER_NUM, tile_size_ * sizeof(T));

    pipe.InitBuffer(sums_q_, 1, AlignUp(block_num_, MIN_VEC_SIZE) * sizeof(T));
  }

  /**
   * @brief Run the kernel - process all tiles.
   *
   * @param [in] starting_value Starting value added to all entries of the
   * output vector.
   */
  __aicore__ inline void Process(T starting_value = 0) {
    const T previous_sum = ScanReducedTiles() + starting_value;
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
  __aicore__ inline T ScanReducedTiles() {
    // Reduce the sums of all the previous tiles.
    copy::CopyGmToVec(sums_q_, global_sums_, sums_len_);
    LocalTensor<T> sums_lt = sums_q_.DeQue<T>();
    const T previous_sum = reduce::ReduceScalarAdd(sums_lt, GetBlockIdx());
    sums_q_.FreeTensor(sums_lt);
    return previous_sum;
  }

  __aicore__ inline void AddSumsToMatmulTiles(T previous_sum) {
    uint32_t global_offset =
        GetBlockIdx() * tile_size_ * max_num_tiles_per_block_;
    const uint32_t num_tiles_to_process =
        scalar::GetWorkDistribution(vec_len_, tile_size_, block_num_);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const bool full_tile = global_offset + tile_size_ <= output_real_elems_;
      const uint32_t num_elems_to_process =
          full_tile ? tile_size_ : output_real_elems_ - global_offset;

      copy::CopyGmToVec(vec_tile_q_, global_input_rows_[global_offset],
                        num_elems_to_process);
      previous_sum = VectorAdds(previous_sum);

      copy::CopyVecToGm(global_output_[global_offset], vec_tile_out_q_,
                        num_elems_to_process);

      global_offset += tile_size_;
    }
  }

  __aicore__ inline T VectorAdds(T running_sum) {
    LocalTensor<T> vec_lt = vec_tile_q_.template DeQue<T>();
    const LocalTensor<T> vec_out_lt = vec_tile_out_q_.template AllocTensor<T>();

    DataCopy(vec_out_lt, vec_lt, vec_lt.GetSize());
    vec_tile_q_.FreeTensor(vec_lt);

    uint32_t offset = 0;
    T accumulation = running_sum;
    for (uint32_t i = 0; i < tile_height_; i++) {
      Adds(vec_out_lt[offset], vec_out_lt[offset], accumulation, tile_width_);
      offset += tile_width_;
      accumulation = vec_out_lt.GetValue(offset - 1);
    }
    vec_tile_out_q_.EnQue(vec_out_lt);

    return accumulation;
  }

  TPipe pipe;

  TQue<QuePosition::VECIN, BUFFER_NUM> vec_tile_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> vec_tile_out_q_;

  TQue<QuePosition::VECIN, 1> sums_q_;

  GlobalTensor<T> global_input_rows_;
  GlobalTensor<T> global_sums_;
  GlobalTensor<T> global_output_;
  GlobalTensor<T> global_output_first_elem_;

  constexpr static uint32_t global_shift_ = IsInclusive ? 0 : 1;

  const uint32_t block_num_;
  const uint32_t tile_width_;
  const uint32_t tile_height_;
  const uint32_t tile_size_;
  const uint32_t sums_len_;

  const uint32_t vec_len_;
  const uint32_t output_real_elems_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
};

}  // namespace tcuscan
