/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file kernel_seg_sum_vec_revert.h
 * @brief Kernel implementing a revert speculation of segmented sum cube
 * operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Corrects the (speculative block scan) output of the `KernelRowScan` so
 * that the segmented sum of a vector is returned.
 *
 *
 * ### Example:
 * Given, vec_in = [1,2,3,4,5,6,7,8,9,10] and segm_ind_in = [0, 4, 7]
 * returns vec_out = (1+2+3+4, 5+6, 7+8+9+10) = (10, 11, 34)
 *
 * @tparam T Input data type. Supports `float` and `int32_t`.
 * @tparam SyncBefore If true, the `KernelSegSumVecRevert` (this kernel) waits
 * for the cube unit to send a synchronization signal after each matrix tile is
 * ready. Matrix tile has length `tile_len * tile_len`.
 * @tparam UseAtomicWrite If true, the output is written using atomic-add
 * semantics.
 */
template <typename T, bool SyncBefore = false, bool UseAtomicWrite = false>
class KernelSegSumVecRevert {
  constexpr static uint32_t BUFFER_NUM = 2;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Input vector length.
   * @param [in] num_segments Number of segments.
   * @param [in] tile_len Tile length.
   * @param [in] vec_start_offset Start offset of input data vector. Segment
   * values will be offset accordindly. Default value is `0`.
   * @param [in] alpha Scaling factor applied to every segment sum on write.
   * Default value is `1` (no scaling). Scaling at the output stage (instead of
   * scaling the input vector) keeps the running scan accumulator unscaled and
   * distributes over the atomic-add writes of overlapping blocks.
   */
  __aicore__ inline KernelSegSumVecRevert(uint32_t vec_len,
                                          uint32_t num_segments,
                                          uint32_t tile_len,
                                          uint32_t vec_start_offset = 0,
                                          T alpha = 1)
      : vec_len_(vec_len),
        num_segments_(num_segments),
        tile_len_(tile_len),
        matrix_tile_len_(tile_len * tile_len),
        num_tiles_(scalar::CeilDiv(vec_len_, matrix_tile_len_)),
        vec_start_offset_(vec_start_offset),
        alpha_(alpha) {
    constexpr bool IS_DT_SUPPORTED =
        std::is_same_v<T, float> || std::is_same_v<T, int32_t>;
    static_assert(IS_DT_SUPPORTED, "Unsupported data type.");
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to the input vector in global memory.
   * @param [in] segm_ind_in Pointer to the segment indices vector in global
   * memory.
   * @param [in] vec_out Pointer to the output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR segm_ind_in,
                              GM_ADDR vec_out) {
    global_in_.SetGlobalBuffer((__gm__ T*)vec_in + vec_start_offset_, vec_len_);
    global_segm_in_.SetGlobalBuffer((__gm__ uint32_t*)segm_ind_in,
                                    num_segments_);
    global_out_.SetGlobalBuffer((__gm__ T*)vec_out, num_segments_ + 1);

    pipe_.InitBuffer(in_q_, BUFFER_NUM, matrix_tile_len_ * sizeof(T));
    pipe_.InitBuffer(segm_q_, BUFFER_NUM, tile_len_ * sizeof(uint32_t));
    pipe_.InitBuffer(out_q_, 1, tile_len_ * sizeof(T));
  }

  /**
   * @brief Returns a local tensor that contains the next segment indices tile
   * (of length up to `tile_len_`)
   *
   * @return Segments index vector of tile length at most `tile_len_`.
   */
  __aicore__ inline LocalTensor<uint32_t> LoadNextSegmentTile() {
    const uint32_t next_tile_len =
        scalar::NextTileLen(tile_len_, segments_offset_, num_segments_);
    copy::CopyGmToVec(segm_q_, global_segm_in_[segments_offset_],
                      next_tile_len);

    return segm_q_.template DeQue<uint32_t>();
  }

  /**
   * @brief Prepare for output writing of the current value on the given
   * index.
   *
   * @param vec_out_lt Tile of output containing the segmented sums.
   * @param index Tile index to write on.
   * @param value Value to write.
   */
  __aicore__ inline void SafeOutWrite(LocalTensor<T>& vec_out_lt,
                                      uint32_t& index, const T value) {
    vec_out_lt.SetValue(index, value);
    index++;
    // Write tile to GM and "re-allocate" the output tile.
    if (index >= tile_len_) {
      out_q_.template EnQue<T>(vec_out_lt);
      if constexpr (UseAtomicWrite) {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SetAtomicAdd<T>();
      }
      copy::CopyVecToGm(global_out_[out_offset_], out_q_, tile_len_);
      if constexpr (UseAtomicWrite) {
        AscendC::SetAtomicNone();
      }
      out_offset_ += tile_len_;
      vec_out_lt = out_q_.template AllocTensor<T>();

      // "Local-coordinates" of local tensor index must be zeroed
      index = 0;
    }
  }

  /**
   * @brief Returns the index corresponding to the next segment end.
   *
   * Method follows the iterator pattern. At termination, returns `vec_len_`.
   *
   * @param [in] segm_ind_lt Tile containing the segment-ending indices.
   * @param [in] index Current "local" index within the @p segm_ind_lt tile.
   * @return Returns the next segment end.
   */
  __aicore__ inline uint32_t NextSegmEndIndex(
      LocalTensor<uint32_t>& segm_ind_lt, uint32_t& index) {
    index++;

    if (segments_offset_ + index >= num_segments_) {
      // Run out-of-segments, set the segment end to maximum
      // vector length.
      return vec_len_;
    } else if (index >= tile_len_) {
      // Run out-of-segments in the segment tiles. Just load the
      // next segment tile and update counters/indices
      segm_q_.template FreeTensor<uint32_t>(segm_ind_lt);
      segments_offset_ += tile_len_;
      segm_ind_lt = LoadNextSegmentTile();

      // "Local-coordinates" segment index must be zeroed and keep
      // track of the output vector location.
      index = 0;
      return static_cast<uint32_t>(segm_ind_lt.GetValue(0) - vec_start_offset_);
    } else {
      // Read next segment
      return static_cast<uint32_t>(segm_ind_lt.GetValue(index) -
                                   vec_start_offset_);
    }
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    if (GetSubBlockIdx() == 0) {
      PipelineProcessWithCube();
    } else {
      SyncWithCubeNoop();
    }
  }

 private:
  __aicore__ inline void SyncWithCubeNoop() {
    for (uint32_t tile_idx = 0; tile_idx < num_tiles_; tile_idx++) {
      if constexpr (SyncBefore) {
        sync::SyncGroup<sync::GroupSyncDirection::FULL>();
      }
    }
  }
  __aicore__ inline void PipelineProcessWithCube() {
    LocalTensor<uint32_t> segm_ind_lt = LoadNextSegmentTile();
    LocalTensor<T> vec_out_lt = out_q_.template AllocTensor<T>();

    T accumulation = 0;
    uint32_t global_in_offset = 0;
    uint32_t out_idx = 0;
    uint32_t segm_idx = 0;
    uint32_t segm_end = static_cast<uint32_t>(segm_ind_lt.GetValue(segm_idx) -
                                              vec_start_offset_);

    for (uint32_t matrix_tile_idx = 0; matrix_tile_idx < num_tiles_;
         matrix_tile_idx++) {
      if constexpr (SyncBefore) {
        sync::SyncGroup<sync::GroupSyncDirection::FULL>();
      }

      const uint32_t num_elems_to_copy =
          scalar::NextTileLen(matrix_tile_len_, global_in_offset, vec_len_);
      copy::CopyGmToVec(in_q_, global_in_[global_in_offset], num_elems_to_copy);
      LocalTensor<T> vec_in_lt = in_q_.template DeQue<T>();

      uint32_t matrix_tile_offset = 0;
      uint32_t num_elems_to_process =
          scalar::NextTileLen(tile_len_, matrix_tile_offset, num_elems_to_copy);

      while (num_elems_to_process > 0) {
        while (segm_end < global_in_offset + num_elems_to_process) {
          // Index of the segment end relative to the matrix tile
          const uint32_t matrix_tile_segm_end =
              matrix_tile_offset + segm_end - global_in_offset;

          // Last segment value. Zero if lies on tile boundary.
          const bool is_on_end_of_tile = segm_end == global_in_offset;
          const T delta = is_on_end_of_tile
                              ? 0
                              : vec_in_lt.GetValue(matrix_tile_segm_end - 1);
          SafeOutWrite(vec_out_lt, out_idx, alpha_ * (accumulation + delta));

          // Keep track of the value of the last segment's end to
          // subtract it from the next segment (recall scan
          // speculation within tile)
          accumulation = -delta;

          // Mutates both segm_ind_lt and segm_idx
          segm_end = NextSegmEndIndex(segm_ind_lt, segm_idx);
        }

        global_in_offset += num_elems_to_process;
        matrix_tile_offset += num_elems_to_process;

        accumulation += vec_in_lt.GetValue(matrix_tile_offset - 1);

        // Number of element to process in the next iteration
        num_elems_to_process = scalar::NextTileLen(
            tile_len_, matrix_tile_offset, num_elems_to_copy);
      }

      in_q_.template FreeTensor<T>(vec_in_lt);
    }

    segm_q_.template FreeTensor<uint32_t>(segm_ind_lt);

    // The last segment contains the remaining accumulation values.
    vec_out_lt.SetValue(out_idx, alpha_ * accumulation);

    const uint32_t tail_len = out_idx + 1;
    out_q_.template EnQue<T>(vec_out_lt);
    if constexpr (UseAtomicWrite) {
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::SetAtomicAdd<T>();
    }
    copy::CopyVecToGm(global_out_[out_offset_], out_q_, tail_len);
    if constexpr (UseAtomicWrite) {
      AscendC::SetAtomicNone();
    }
  }

  TPipe pipe_;

  TQue<QuePosition::VECIN, BUFFER_NUM> in_q_;
  TQue<QuePosition::VECIN, BUFFER_NUM> segm_q_;
  TQue<QuePosition::VECOUT, 1> out_q_;

  GlobalTensor<T> global_in_;
  GlobalTensor<uint32_t> global_segm_in_;
  GlobalTensor<T> global_out_;

  const uint32_t vec_len_;
  const uint32_t num_segments_;
  const uint32_t tile_len_;
  const uint32_t matrix_tile_len_;
  const uint32_t num_tiles_;
  const uint32_t vec_start_offset_;
  const T alpha_;

  uint32_t segments_offset_ = 0;
  uint32_t out_offset_ = 0;
};

}  // namespace tcuscan
