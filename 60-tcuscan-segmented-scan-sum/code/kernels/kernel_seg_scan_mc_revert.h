/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_seg_scan_mc_revert.h
 * @brief Kernel implementing an AIV segmented scan revertion kernel
 * operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Given a vector `vec_in` along with a segments indicator vector
 * `vec_f_in`, returns the following (Python notation)
 *
 * for segment_id in range(len(vec_diff_in)):
 *      vec_in = vec_in - (vec_f_in == segment_id + 1) * vec_diff_in[segment_id]
 *
 */
template <typename DataTypeT = float, typename InputFlagT = int32_t>
class KernelSegScanMcRevert {
  constexpr static uint32_t BUFFER_NUM = 2;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Dimension of the input vectors.
   * @param [in] num_segments Number of segments (length of `vec_diff_in`)
   * @param [in] tile_len Tile length.
   */
  __aicore__ inline KernelSegScanMcRevert(uint32_t vec_len,
                                          uint32_t num_segments,
                                          uint32_t tile_len)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        num_segments_(num_segments),
        tile_len_(tile_len),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_len_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, vec_core_num_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to the input data vector in global memory.
   * @param [in] vec_f_in Pointer to the input flag vector in global memory.
   * @param [in] vec_diff_in Pointer to the input diff vector in global
   * memory.
   * @param [in] vec_out Pointer to the output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR vec_f_in,
                              GM_ADDR vec_diff_in, GM_ADDR vec_out) {
    global_in_.SetGlobalBuffer((__gm__ DataTypeT*)vec_in, vec_len_);
    global_f_in_.SetGlobalBuffer((__gm__ DataTypeT*)vec_f_in, vec_len_);
    global_diff_in_.SetGlobalBuffer((__gm__ InputFlagT*)vec_diff_in,
                                    num_segments_);
    global_out_.SetGlobalBuffer((__gm__ DataTypeT*)vec_out, vec_len_);

    pipe.InitBuffer(in_q_, BUFFER_NUM, tile_len_ * sizeof(DataTypeT));
    pipe.InitBuffer(in_f_q_, BUFFER_NUM, tile_len_ * sizeof(InputFlagT));
    pipe.InitBuffer(in_diff_q_, BUFFER_NUM,
                    (tile_len_ + 1) * sizeof(DataTypeT));
    pipe.InitBuffer(out_q_, BUFFER_NUM, tile_len_ * sizeof(DataTypeT));

    pipe.InitBuffer(float_buf_, tile_len_ * sizeof(float));
    pipe.InitBuffer(mask_buf_, tile_len_ * sizeof(int32_t));
    pipe.InitBuffer(int16_buf_, tile_len_ * sizeof(int16_t));
  }

  /**
   * @brief Run the kernel.
   *
   */
  __aicore__ inline void Process() {
    uint32_t global_offset =
        GetBlockIdx() * tile_len_ * max_num_tiles_per_block_;
    const uint32_t num_tiles_to_process = tcuscan::scalar::GetWorkDistribution(
        vec_len_, tile_len_, vec_core_num_);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const bool full_tile = global_offset + tile_len_ <= vec_len_;
      const uint32_t num_elems_to_process_ =
          full_tile ? tile_len_ : vec_len_ - global_offset;

      copy::CopyGmToVec(in_q_, global_in_[global_offset],
                        num_elems_to_process_);

      copy::CopyGmToVec(in_f_q_, global_f_in_[global_offset],
                        num_elems_to_process_);

      ProcessTile();

      copy::CopyVecToGm(global_out_[global_offset], out_q_,
                        num_elems_to_process_);
      global_offset += tile_len_;
    }
  }

  /**
   * @brief Compare vector with a scalar using abs(x - scalar) and count
   * number of zeros
   *
   * @param dst Tensor containing 1 iff element equals to
   * `scalar`
   * @param src Input vector.
   * @param scalar Scalar to compare vector against.
   */

  __aicore__ inline void CustomCompareScalarEQ(LocalTensor<float>& dst,
                                               LocalTensor<int32_t>& src,
                                               int32_t scalar) {
    LocalTensor<int16_t> int16_lt = int16_buf_.Get<int16_t>();
    LocalTensor<uint16_t> uint16_lt =
        int16_lt.template ReinterpretCast<uint16_t>();
    LocalTensor<int32_t> mask_buf_lt = mask_buf_.Get<int32_t>();

    Adds(mask_buf_lt, src, -scalar, tile_len_);
    Cast(int16_lt, mask_buf_lt, RoundMode::CAST_NONE, tile_len_);
    Not(int16_lt, int16_lt, tile_len_);
    ShiftRight(uint16_lt, uint16_lt, FIFTEEN_, tile_len_);
    Cast(dst, int16_lt, RoundMode::CAST_NONE, tile_len_);
  }

  /**
   * @brief Masked vector substraction/difference.
   *
   * MaskedSub performs the following vector operation,
   * output = output - (mask == segment_id) * delta
   *
   * @param output Output local tensor
   * @param mask Scanned flag vector
   * @param segment_id The segment id
   * @param delta Correction value
   */
  __aicore__ inline void MaskedSub(LocalTensor<DataTypeT>& output,
                                   LocalTensor<int32_t> mask,
                                   int32_t segment_id, DataTypeT delta) {
    if (delta == 0) {
      return;
    }

    const uint32_t size = output.GetSize();
    LocalTensor<float> float_buf_lt = float_buf_.Get<float>();
    CustomCompareScalarEQ(float_buf_lt, mask, segment_id);
    Muls(float_buf_lt, float_buf_lt, delta, size);
    Sub(output, output, float_buf_lt, size);
  }

 private:
  __aicore__ inline void ProcessTile() {
    LocalTensor<DataTypeT> vec_in_lt = in_q_.DeQue<DataTypeT>();
    LocalTensor<InputFlagT> vec_f_in_lt = in_f_q_.DeQue<InputFlagT>();

    LocalTensor<DataTypeT> vec_out_lt = out_q_.AllocTensor<DataTypeT>();

    // Smallest and largest segment idx to process in this tile
    const int32_t smallestSegmIdx = vec_f_in_lt(0) > 0 ? vec_f_in_lt(0) - 1 : 0;

    const int32_t largestSegmIdx =
        static_cast<uint32_t>(vec_f_in_lt(vec_f_in_lt.GetSize() - 1)) >
                num_segments_
            ? (num_segments_ - 1)
            : vec_f_in_lt(vec_f_in_lt.GetSize() - 1);
    copy::CopyGmToVec(in_diff_q_, global_diff_in_[smallestSegmIdx],
                      largestSegmIdx - smallestSegmIdx + 1);

    LocalTensor<DataTypeT> vec_in_diff_lt = in_diff_q_.DeQue<DataTypeT>();

    DataCopy(vec_out_lt, vec_in_lt, tile_len_);

    DataTypeT previous_sum = 0;
    for (int32_t segm_idx = smallestSegmIdx; segm_idx <= largestSegmIdx;
         segm_idx++) {
      //  Get the last value of the previous segment
      const float delta = vec_in_diff_lt(segm_idx - smallestSegmIdx);
      MaskedSub(vec_out_lt, vec_f_in_lt, segm_idx + 1, delta - previous_sum);
      previous_sum = delta;
    }

    out_q_.EnQue<DataTypeT>(vec_out_lt);
    in_q_.FreeTensor<DataTypeT>(vec_in_lt);
    in_f_q_.FreeTensor<InputFlagT>(vec_f_in_lt);
    in_diff_q_.FreeTensor<>(vec_in_diff_lt);
  }

  TPipe pipe;

  TQue<QuePosition::VECIN, BUFFER_NUM> in_q_;
  TQue<QuePosition::VECIN, BUFFER_NUM> in_f_q_;
  TQue<QuePosition::VECIN, BUFFER_NUM> in_diff_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> out_q_;

  TBuf<QuePosition::VECCALC> float_buf_;
  TBuf<QuePosition::VECCALC> mask_buf_;
  TBuf<QuePosition::VECCALC> int16_buf_;

  GlobalTensor<DataTypeT> global_in_;
  GlobalTensor<DataTypeT> global_f_in_;
  GlobalTensor<InputFlagT> global_diff_in_;
  GlobalTensor<DataTypeT> global_out_;

  const uint32_t vec_core_num_;
  const uint32_t vec_len_;
  const uint32_t num_segments_;
  const uint32_t tile_len_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;

  constexpr static uint16_t FIFTEEN_ = 15;
  constexpr static float ZERO_FLOAT = 0;
};

/**
 * @brief Run the `seg_scan_mc_revert` kernel.
 *
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 *
 * @param [in] vec_in Pointer to the input data vector.
 * @param [in] vec_f_in Pointer to the input flag vector.
 * @param [in] vec_diff_in Pointer to the input difference vector vector.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] vec_len Dimension of the input vector.
 * @param [in] num_segments Number of segments.
 * @param [in] tile_len Tile length.
 */
template <bool ForceMixMode = true, typename DataTypeT>
__aicore__ inline void run_seg_scan_mc_revert(GM_ADDR vec_in, GM_ADDR vec_f_in,
                                              GM_ADDR vec_diff_in,
                                              GM_ADDR vec_out, uint32_t vec_len,
                                              uint32_t num_segments,
                                              uint32_t tile_len) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIV {
    KernelSegScanMcRevert<DataTypeT> op(vec_len, num_segments, tile_len);
    op.Init(vec_in, vec_f_in, vec_diff_in, vec_out);
    op.Process();
  }
}

}  // namespace tcuscan
