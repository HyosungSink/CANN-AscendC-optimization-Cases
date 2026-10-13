/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file kernel_count_if.h
 * @brief Kernel implementing a Vector `count_if` kernel operation.
 */
#pragma once

#include <cmath>

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"
using namespace AscendC;

namespace tcuscan {

/**
 * @brief Returns `torch.count_nonzero(x <= pivot)` given an input vector.
 *
 * @tparam InputT Input data type.
 * @tparam ReduceBlocks if true the partial results of each core are accumulated
 * together, otherwise they are kept separate.
 */
template <typename InputT = half, bool ReduceBlocks = true>
class KernelCountIf {
  /// @brief Accumulation data type for the counts.
  using AccT = int32_t;
  using PackedMaskT = uint8_t;
  using GatherMaskT =
      typename std::conditional<sizeof(InputT) == 2, uint16_t, uint32_t>::type;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Dimension of the input vectors.
   * @param [in] tile_len Tile length.
   * @param [in] compare_mode Comparison enum of \c AscendC::CompareScalar.
   */
  __aicore__ inline KernelCountIf(uint32_t vec_len, uint32_t tile_len,
                                  CMPMODE compare_mode)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        compare_mode_(compare_mode),
        tile_len_(tile_len),
        vec_len_(vec_len),
        num_tiles_(scalar::CeilDiv(vec_len, tile_len)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, vec_core_num_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to the input vector in global memory.
   * @param [in] vec_out Pointer to the output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR vec_out) {
    global_input_.SetGlobalBuffer((__gm__ InputT*)vec_in, vec_len_);
    global_output_.SetGlobalBuffer((__gm__ AccT*)vec_out, vec_core_num_);

    pipe_.InitBuffer(in_q_, 1, tile_len_ * sizeof(InputT));

    pipe_.InitBuffer(work_buf_, tile_len_ * sizeof(InputT));
    pipe_.InitBuffer(packed_mask_buf_, tile_len_ * sizeof(PackedMaskT) / 8);
    pipe_.InitBuffer(out_q_, 1, MIN_VEC_SIZE * sizeof(AccT));
  }

  /**
   * @brief Run the kernel.
   *
   * @param [in] pivot Pivot on which to count the cardinality of the set {x_i <
   * pivot} where \f$ x_i \f$ are the input elements.
   *
   */
  __aicore__ inline void Process(InputT pivot) {
    const LocalTensor<InputT> work_lt = work_buf_.Get<InputT>();

    const uint32_t num_tiles_to_process = tcuscan::scalar::GetWorkDistribution(
        vec_len_, tile_len_, vec_core_num_);

    uint32_t global_offset =
        GetBlockIdx() * tile_len_ * max_num_tiles_per_block_;
    AccT sum = 0;

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const uint32_t num_elems =
          scalar::NextTileLen(tile_len_, global_offset, vec_len_);

      if (num_elems > 0) {
        copy::CopyGmToVec(
            in_q_, global_input_[global_offset], num_elems,
            MAP_CMPMODE_TO_BOUNDARY[static_cast<int>(compare_mode_)]);
        LocalTensor<InputT> input_lt = in_q_.template DeQue<InputT>();
        const uint32_t input_len =
            scalar::AlignUp(num_elems, UB_ALIGNMENT / sizeof(InputT));
        // pad the input
        if (num_elems < tile_len_) {
          Duplicate(input_lt[input_len],
                    MAP_CMPMODE_TO_BOUNDARY[static_cast<int>(compare_mode_)],
                    input_lt.GetSize() - input_len);
        }

        const LocalTensor<uint8_t> packed_mask_8b =
            packed_mask_buf_.Get<uint8_t>();
        const LocalTensor<GatherMaskT> mask_lt =
            packed_mask_8b.template ReinterpretCast<GatherMaskT>();

        CompareScalar(packed_mask_8b, input_lt, pivot, compare_mode_,
                      tile_len_);
        PipeBarrier<PIPE_ALL>();

        // Gather mask is used only to count the 1s in the mask; input and
        // output are ignored
        size_t num_gathered_elems = 0;
        GatherMask(work_lt, work_lt, mask_lt, true, input_len, {1, 1, 8, 8},
                   num_gathered_elems);
        // Ensure the vector pipe has written num_gathered_elems before the
        // scalar CPU reads it in the accumulation below.
        PipeBarrier<PIPE_V>();

        in_q_.FreeTensor(input_lt);
        sum += num_gathered_elems;
      }

      global_offset += num_elems;
    }

    if constexpr (ReduceBlocks)
      AtomicAddWrite(static_cast<int32_t>(sum));
    else {
      AscendC::PipeBarrier<PIPE_ALL>();
      copy::CopyScalarToGm(global_output_[GetBlockIdx()], out_q_, sum);
    }
  }

 private:
  __aicore__ inline void AtomicAddWrite(int32_t value) {
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SetAtomicAdd<int32_t>();
    copy::CopyScalarToGm(global_output_, out_q_, value);
    AscendC::SetAtomicNone();
  }

  TPipe pipe_;

  TQue<QuePosition::VECIN, 1> in_q_;
  TQue<QuePosition::VECOUT, 1> out_q_;

  TBuf<QuePosition::VECCALC> work_buf_;
  TBuf<QuePosition::VECCALC> packed_mask_buf_;

  GlobalTensor<InputT> global_input_;
  GlobalTensor<int32_t> global_output_;

  constexpr static int32_t MIN_VEC_SIZE = UB_ALIGNMENT / sizeof(AccT);
  constexpr static InputT P_INF = std::numeric_limits<InputT>::max();
  constexpr static InputT N_INF = std::is_same_v<InputT, half>
                                      ? fp16::FP16_LOWEST_NORMAL
                                      : std::numeric_limits<InputT>::lowest();

  // this array maps CMPMODE to a value that always returns false when compared
  // to the input.
  // Issue: NaN may behave unexpectedly with CompareScalar
  constexpr static InputT MAP_CMPMODE_TO_BOUNDARY[] = {N_INF, P_INF, NAN,
                                                       N_INF, P_INF, NAN};

  const uint32_t vec_core_num_;
  const CMPMODE compare_mode_;
  const uint32_t tile_len_;
  const uint32_t vec_len_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
};

/**
 * @brief Run the `count_if` kernel.
 *
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] vec_len Dimension of the input vector.
 * @param [in] tile_len Tile length.
 * @param [in] pivot Initial pivot.
 * @param [in] compare_mode Comparison enum of \c AscendC::CompareScalar.
 */
template <typename InputT>
__aicore__ inline void run_count_if(GM_ADDR vec_in, GM_ADDR vec_out,
                                    uint32_t vec_len, uint32_t tile_len,
                                    InputT pivot,
                                    AscendC::CMPMODE compare_mode) {
  if ASCEND_IS_AIV {
    KernelCountIf<InputT> op(vec_len, tile_len, compare_mode);
    op.Init(vec_in, vec_out);
    op.Process(pivot);
  }
}

}  // namespace tcuscan
