/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_seg_scan_vec_single_core.h
 * @brief Kernel implementing a segmented scan vector-only kernel operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Compute the segmented scan of an input vector and flag.
 *
 * The algorithm utilizes only a single vector core.
 *
 * @tparam InputT Data type of input and output vectors
 * @tparam FlagT Data type of flag vector
 */
template <typename InputT, typename FlagInT>
class KernelSegScanVecSingleCore {
  constexpr static uint32_t BUFFER_NUM = 1;
  using OutputT = tcuscan::cube_unit::CubeOutType_t<InputT>;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Input vector length.
   * @param [in] tile_len Tile length.
   */
  __aicore__ inline KernelSegScanVecSingleCore(uint32_t vec_len,
                                               uint16_t tile_len)
      : vec_len_(vec_len),
        tile_len_(tile_len),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_len_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to row-wise block scan of input in GM.
   * @param [in] flag_in Pointer to row-wise block scan of flags in GM.
   * @param [in] vec_out Pointer to output vector in GM.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR flag_in,
                              GM_ADDR vec_out) {
    global_input_.SetGlobalBuffer((__gm__ InputT*)vec_in, vec_len_);
    global_flag_.SetGlobalBuffer((__gm__ FlagInT*)flag_in, vec_len_);
    global_output_.SetGlobalBuffer((__gm__ OutputT*)vec_out, vec_len_);

    pipe.InitBuffer(vecin_input_q_, BUFFER_NUM, tile_len_ * sizeof(InputT));
    pipe.InitBuffer(vecin_flag_q_, BUFFER_NUM, tile_len_ * sizeof(FlagInT));
    pipe.InitBuffer(vecout_q_, BUFFER_NUM, tile_len_ * sizeof(OutputT));

    pipe.InitBuffer(tmp1_int16_buf_, tile_len_ * sizeof(int16_t));
    pipe.InitBuffer(tmp1_half_buf_, tile_len_ * sizeof(half));
    pipe.InitBuffer(tmp2_half_buf_, tile_len_ * sizeof(half));
    pipe.InitBuffer(tmp_get_start_float_buf_, tile_len_ * sizeof(float));
    pipe.InitBuffer(tmp_get_start_half_buf_, tile_len_ * sizeof(half));
    pipe.InitBuffer(tmp_float_buf_, tile_len_ * sizeof(float));
    pipe.InitBuffer(vec_calc_buf_, tile_len_ * sizeof(float));
    pipe.InitBuffer(flag_in_calc_buf_, tile_len_ * sizeof(float));
    pipe.InitBuffer(flag_calc_buf_, tile_len_ * sizeof(int32_t));
  }

  /**
   * @brief Run the kernel - process all tiles.
   */
  __aicore__ inline void Process() {
    uint32_t tile_idx = 0;
    OutputT running_sum = 0;
    if (GetBlockIdx() == 0 and GetSubBlockIdx() == 0) {
      // FIXME: uses only one vector core
      for (uint32_t i = 0; i < num_tiles_; i++) {
        VecIter(tile_idx, running_sum);
        tile_idx++;
      }
    }
  }

  /**
   * @brief Executes an iteration on tile with index `tile_idx`.
   *
   * @param tile_idx Tile index.
   * @param running_sum Running sum accumulation value of previous tile.
   */
  __aicore__ inline void VecIter(uint32_t tile_idx, OutputT& running_sum) {
    const bool is_last_tile = (tile_idx + 1) * tile_len_ < vec_len_;
    const uint32_t num_elems_to_process =
        is_last_tile ? tile_len_ : vec_len_ - tile_idx * tile_len_;

    copy::CopyGmToVec<InputT>(vecin_input_q_,
                              global_input_[tile_idx * tile_len_],
                              num_elems_to_process);
    copy::CopyGmToVec<FlagInT>(vecin_flag_q_,
                               global_flag_[tile_idx * tile_len_],
                               num_elems_to_process);

    SegmentedScanOnTile(running_sum, num_elems_to_process);

    copy::CopyVecToGm<OutputT>(global_output_[tile_idx * tile_len_], vecout_q_,
                               num_elems_to_process);
  }

  /**
   * @brief Computes the prefix sum using AscendC::CumSum API.
   *
   * @param out_lt Output local tensor.
   * @param in_lt Input local tensor.
   * @param scan_len Inner scan length.
   */
  __aicore__ inline void VecScan(LocalTensor<float> out_lt,
                                 LocalTensor<float> in_lt, uint32_t scan_len) {
    static constexpr CumSumConfig cumsum_config{true, false, false};
    // TODO (anastasios): Fix me. Scan must be on length s * s here.
    const CumSumInfo cumsum_info{1, scan_len};
    CumSum<float, cumsum_config>(out_lt, in_lt, in_lt, cumsum_info);
  }

  /**
   * @brief Computes the segmented scan on a tile.
   *
   * @param running_sum Accumulation value of last element of previous tile.
   * @param num_elems Number of element to process.
   */
  __aicore__ inline void SegmentedScanOnTile(OutputT& running_sum,
                                             uint32_t num_elems) {
    LocalTensor<InputT> input_vec_lt = vecin_input_q_.DeQue<InputT>();
    LocalTensor<FlagInT> flag_vec_lt = vecin_flag_q_.DeQue<FlagInT>();
    LocalTensor<OutputT> output_vec_lt = vecout_q_.AllocTensor<OutputT>();

    LocalTensor<OutputT> inplace_calc_vec = vec_calc_buf_.Get<OutputT>();

    LocalTensor<int32_t> flag_scanned_lt = flag_calc_buf_.Get<int32_t>();
    LocalTensor<float> in_flag_lt = flag_in_calc_buf_.Get<float>();

    // Scan the input tile of x (inplace)
    Cast(inplace_calc_vec, input_vec_lt, RoundMode::CAST_NONE, num_elems);
    VecScan(inplace_calc_vec, inplace_calc_vec, num_elems);

    // Cast input (int8) to (float) and scan f (inplace)
    Cast(input_vec_lt, flag_vec_lt, RoundMode::CAST_NONE, num_elems);
    Cast(in_flag_lt, input_vec_lt, RoundMode::CAST_NONE, num_elems);
    VecScan(in_flag_lt, in_flag_lt, num_elems);
    Cast(flag_scanned_lt, in_flag_lt, RoundMode::CAST_ROUND, num_elems);

    // Go over each segment and revert speculation, if necessary
    const int32_t num_segments = flag_scanned_lt(tile_len_ - 1);
    const int32_t first_flag = flag_scanned_lt(0);

    if (first_flag == static_cast<int32_t>(0)) {
      Adds(inplace_calc_vec, inplace_calc_vec, running_sum, num_elems);
    }

    PipeBarrier<PIPE_ALL>();
    for (int16_t seg_idx = 1; seg_idx <= num_segments; seg_idx++) {
      sync::ScalarWaitForVec();
      const float delta = GetDelta(inplace_calc_vec, flag_scanned_lt, seg_idx);

      FixSpecInPlace(inplace_calc_vec, flag_scanned_lt, seg_idx, delta);
    }

    if (num_elems == tile_len_) {
      sync::ScalarWaitForVec();
      running_sum = inplace_calc_vec(tile_len_ - 1);
    }

    DataCopy(output_vec_lt, inplace_calc_vec, num_elems);

    vecin_input_q_.FreeTensor<InputT>(input_vec_lt);
    vecin_flag_q_.FreeTensor<FlagInT>(flag_vec_lt);
    vecout_q_.EnQue<OutputT>(output_vec_lt);
  }

  /**
   * @brief Returns the segment delta correction value
   *
   * @param input Input scanned tile of data.
   * @param flag Input scanned tile of flags.
   * @param segment_id The id of the input segment.
   * @return Delta correction value
   */
  __aicore__ inline float GetDelta(const LocalTensor<OutputT>& input,
                                   const LocalTensor<int32_t>& flag,
                                   int16_t segment_id) {
    const int16_t segment_start_index =
        GetStartIndexOfSegment(flag, segment_id);

    const int16_t delta_index = segment_start_index - 1;
    const float delta = delta_index >= 0 ? input(delta_index) : 0;

    return delta;
  }

  /**
   * @brief Returns the start index of a segment given a `segment_id`.
   *
   * @param flag Tile of input scanned flag vector.
   * @param segment_id The id of the input segment.
   * @return Start index of the segment with segment_id.
   */
  __aicore__ inline int16_t GetStartIndexOfSegment(
      const LocalTensor<int32_t>& flag, int16_t segment_id) {
    const half threshold = static_cast<half>(static_cast<float>(segment_id) -
                                             static_cast<float>(0.1));

    LocalTensor<float> float_buf = tmp_get_start_float_buf_.Get<float>();
    LocalTensor<half> f_lt = tmp_get_start_half_buf_.Get<half>();

    const LocalTensor<uint16_t> f_uint16_lt =
        f_lt.template ReinterpretCast<uint16_t>();
    const LocalTensor<int16_t> f_int16_lt =
        f_lt.template ReinterpretCast<int16_t>();
    const LocalTensor<half> f_half_lt = f_lt.template ReinterpretCast<half>();

    const uint32_t size = f_lt.GetSize();

    Cast(float_buf, flag, RoundMode::CAST_NONE, size);
    Cast(f_lt, float_buf, RoundMode::CAST_NONE, size);
    Duplicate(float_buf, static_cast<float>(0), float_buf.GetSize());

    Muls(f_lt, f_lt, static_cast<half>(-1), size);

    Adds(f_lt, f_lt, threshold, size);

    PipeBarrier<PIPE_V>();

    Not(f_uint16_lt, f_uint16_lt, size);

    ShiftRight(f_uint16_lt, f_uint16_lt, (uint16_t)15, size);

    PipeBarrier<PIPE_V>();

    Cast(f_half_lt, f_int16_lt, RoundMode::CAST_NONE, size);
    Cast(float_buf, f_half_lt, RoundMode::CAST_NONE, size);
    PipeBarrier<PIPE_V>();

    ReduceSum(float_buf /*dst*/, float_buf /*src*/, float_buf /*work*/,
              float_buf.GetSize());

    return static_cast<int16_t>(AscendC::GetAccVal<float>());
  }

  /**
   * @brief Compare vector with a scalar using abs(x - scalar) and count
   * number of zeros
   *
   * @param dst Tile of output vector containing 1 iff element equals to
   * `scalar`
   * @param src Tile of input vector.
   * @param scalar Scalar to compare vector against.
   */
  __aicore__ inline void CustomCompareScalarEQ(LocalTensor<half>& dst,
                                               LocalTensor<int16_t>& src,
                                               int16_t scalar) {
    LocalTensor<half> tmp_half_lt = tmp1_half_buf_.Get<half>();
    LocalTensor<uint16_t> tmp_uint16_lt =
        tmp_half_lt.template ReinterpretCast<uint16_t>();
    LocalTensor<int16_t> tmp_int16_lt =
        tmp_half_lt.template ReinterpretCast<int16_t>();

    Cast(tmp_half_lt, src, RoundMode::CAST_NONE, tile_len_);
    Adds(tmp_half_lt, tmp_half_lt, static_cast<half>(-scalar), tile_len_);

    Not(tmp_uint16_lt, tmp_uint16_lt, tile_len_);

    ShiftRight(tmp_uint16_lt, tmp_uint16_lt, (uint16_t)15, tile_len_);

    PipeBarrier<PIPE_V>();

    Cast(dst, tmp_int16_lt, RoundMode::CAST_NONE, tile_len_);
  }

  /**
   * @brief Fixes the speculation in-place
   *
   * FixSpecInPlace performs the following vector operation,
   * output = output - (flag == segment_id) * delta
   *
   * @param output Output local tensor
   * @param flag Scanned flag vector
   * @param segment_id The segment id
   * @param delta Speculation correction value.
   */
  __aicore__ inline void FixSpecInPlace(LocalTensor<float>& output,
                                        LocalTensor<int32_t>& flag,
                                        int16_t segment_id, float delta) {
    if (delta == 0) {
      return;
    }

    LocalTensor<half> half_buf = tmp2_half_buf_.Get<half>();
    LocalTensor<int16_t> flag_int16_lt = tmp1_int16_buf_.Get<int16_t>();

    Cast(flag_int16_lt, flag, RoundMode::CAST_NONE, tile_len_);
    CustomCompareScalarEQ(half_buf, flag_int16_lt, segment_id);

    LocalTensor<float> float_buf = tmp_float_buf_.Get<float>();

    Cast(float_buf, half_buf, RoundMode::CAST_NONE, tile_len_);
    Muls(float_buf, float_buf, delta, tile_len_);
    Sub(output, output, float_buf, tile_len_);
  }

 private:
  TPipe pipe;

  TQue<QuePosition::VECIN, BUFFER_NUM> vecin_input_q_;
  TQue<QuePosition::VECIN, BUFFER_NUM> vecin_flag_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> vecout_q_;

  TBuf<QuePosition::VECCALC> tmp1_int16_buf_;
  TBuf<QuePosition::VECCALC> tmp1_half_buf_;
  TBuf<QuePosition::VECCALC> tmp2_half_buf_;
  TBuf<QuePosition::VECCALC> tmp_get_start_float_buf_;
  TBuf<QuePosition::VECCALC> tmp_get_start_half_buf_;
  TBuf<QuePosition::VECCALC> tmp_float_buf_;
  TBuf<QuePosition::VECCALC> vec_calc_buf_;
  TBuf<QuePosition::VECCALC> flag_in_calc_buf_;
  TBuf<QuePosition::VECCALC> flag_calc_buf_;

  GlobalTensor<InputT> global_input_;
  GlobalTensor<FlagInT> global_flag_;
  GlobalTensor<OutputT> global_output_;

  const uint32_t vec_len_;
  const uint32_t tile_len_;
  const uint32_t num_tiles_;
};

}  // namespace tcuscan
