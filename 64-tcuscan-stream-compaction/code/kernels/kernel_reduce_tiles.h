/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_reduce_tiles.h
 * @brief Kernel implementing a tile reducer using the vector cores.
 *
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Performs add-reduce operation on input tiles.
 *
 * Divides an input vector into tiles of size `tile_size` and reduces each of
 * them separately using addition operation. Because of the UB alignment
 * constraints, each result scalar is represented by a 32 bytes vector, where
 * the first `sizeof(AccT)` bytes is the actual result and the rest are dummy
 * padding data.
 *
 * @tparam InputT Data type of the input vector.
 * @tparam BufferNum Number of buffers. If set to `2`, double-buffering is
 * enabled.
 */
template <typename InputT, int32_t BufferNum = 2>
class KernelReduceTiles {
  using IntermediateT = half;
  using AccT = cube_unit::CubeOutType_t<InputT>;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] tile_len Tile length.
   * @param [in] vec_len Number of elements in an input vector.
   */
  __aicore__ inline KernelReduceTiles(uint32_t tile_len, uint32_t vec_len)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        tile_len_(tile_len),
        vec_len_(vec_len),
        num_tiles_(scalar::CeilDiv(vec_len, tile_len)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, vec_core_num_)),
        red1_size_((tile_len_ / 16 > RED2_SIZE) ? tile_len_ / 16
                                                : tile_len_ / 4) {
    static_assert(IS_ACC_T_SUPPORTED, "Unsupported accumulator type.");
    static_assert(IS_IN_T_8BIT || !IS_IN_T_UNSIGNED, "Unsupported input type.");
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input Pointer to input vector in global memory.
   * @param [in] output Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
    global_input_.SetGlobalBuffer((__gm__ InputT*)input, vec_len_);
    global_output_.SetGlobalBuffer((__gm__ AccT*)output, vec_core_num_);

    pipe_.InitBuffer(vec_tile_input_q_, BufferNum, tile_len_ * sizeof(InputT));
    if constexpr (REQ_INTERMEDIATE_CAST) {
      pipe_.InitBuffer(vec_tile_intermediate_buf_,
                       tile_len_ * sizeof(IntermediateT));
    }
    pipe_.InitBuffer(vec_tile_q_, tile_len_ * sizeof(AccT));

    pipe_.InitBuffer(red1_buf_, red1_size_ * sizeof(AccT));
    pipe_.InitBuffer(red2_buf_, RED2_SIZE * sizeof(AccT));
    pipe_.InitBuffer(res_q_, 1, MIN_VEC_SIZE * sizeof(AccT));
  }

  /**
   * @brief Run the kernel - process all tiles.
   */
  __aicore__ inline void Process() {
    const LocalTensor<AccT> input_lt = vec_tile_q_.Get<AccT>();
    const uint32_t num_tiles_to_process = tcuscan::scalar::GetWorkDistribution(
        vec_len_, tile_len_, vec_core_num_);

    uint32_t global_offset =
        GetBlockIdx() * tile_len_ * max_num_tiles_per_block_;
    AccT sum = 0;

    if constexpr (reduce::IsAscendReduceSumSupported<AccT>) {
      for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
        const uint32_t num_elems =
            scalar::NextTileLen(tile_len_, global_offset, vec_len_);

        if (num_elems > 0) {
          LoadToAccT(input_lt, global_offset, num_elems);
          ReduceSum(input_lt, input_lt, input_lt, num_elems);
          sum += AscendC::GetAccVal<AccT>();
        }

        global_offset += num_elems;
      }
    } else {
      const LocalTensor<AccT> red1_lt = red1_buf_.Get<AccT>();

      const uint32_t num_elems =
          scalar::NextTileLen(tile_len_, global_offset, vec_len_);

      if (num_elems > 0) {
        LoadToAccT(input_lt, global_offset, num_elems);
        reduce::ReduceVecAdd<true /*AllocAcc*/, AccT>(red1_lt, input_lt,
                                                      num_elems);
      }

      global_offset += num_elems;

      for (uint32_t tile_idx = 1; tile_idx < num_tiles_to_process; tile_idx++) {
        const uint32_t num_elems =
            scalar::NextTileLen(tile_len_, global_offset, vec_len_);

        if (num_elems > 0) {
          LoadToAccT(input_lt, global_offset, num_elems);
          reduce::ReduceVecAdd<false /*AllocAcc*/, AccT>(red1_lt, input_lt,
                                                         num_elems);
        }

        global_offset += num_elems;
      }
      if (num_tiles_to_process > 0) {
        const LocalTensor<AccT> red2_lt = red2_buf_.Get<AccT>();
        reduce::ReduceVecAdd<true /*AllocAcc*/, AccT>(red2_lt, red1_lt,
                                                      red1_lt.GetSize());
        sum = reduce::ReduceScalarAdd<AccT>(red2_lt, red2_lt.GetSize());
      }
    }
    copy::CopyScalarToGm(global_output_[GetBlockIdx()], res_q_, sum);
  }

 private:
  __aicore__ inline void LoadToAccT(const LocalTensor<AccT>& dst_lt,
                                    uint32_t global_offset,
                                    uint32_t num_elems_to_process) {
    if (num_elems_to_process == 0) {
      return;
    }
    copy::CopyGmToVec(vec_tile_input_q_, global_input_[global_offset],
                      num_elems_to_process);
    LocalTensor<InputT> input_lt = vec_tile_input_q_.template DeQue<InputT>();
    const auto dst_size =
        scalar::AlignUp(num_elems_to_process, UB_ALIGNMENT / sizeof(InputT));

    if constexpr (REQ_INTERMEDIATE_CAST) {
      const LocalTensor<IntermediateT> intermediate_lt =
          vec_tile_intermediate_buf_.Get<IntermediateT>();
      Cast(intermediate_lt, input_lt, RoundMode::CAST_NONE, dst_size);

      constexpr auto cast_mode = std::is_same_v<AccT, float>
                                     ? RoundMode::CAST_NONE
                                     : RoundMode::CAST_RINT;
      Cast(dst_lt, intermediate_lt, cast_mode, dst_size);
    } else if constexpr (std::is_same_v<InputT, AccT>) {
      // `Cast` is a no-op when the source and destination types are identical,
      // so the accumulator tensor would be left uninitialized. Copy instead.
      DataCopy(dst_lt, input_lt, dst_size);
    } else {
      Cast(dst_lt, input_lt, RoundMode::CAST_NONE, dst_size);
    }

    vec_tile_input_q_.FreeTensor(input_lt);
  }

  TPipe pipe_;

  TQue<QuePosition::VECIN, BufferNum> vec_tile_input_q_;
  TBuf<QuePosition::VECCALC> vec_tile_intermediate_buf_;
  TBuf<QuePosition::VECCALC> vec_tile_q_;
  TBuf<QuePosition::VECCALC> red1_buf_;
  TBuf<QuePosition::VECCALC> red2_buf_;
  TQue<QuePosition::VECOUT, BufferNum> res_q_;

  GlobalTensor<InputT> global_input_;
  GlobalTensor<AccT> global_output_;

  constexpr static int32_t MIN_VEC_SIZE = UB_ALIGNMENT / sizeof(AccT);
  constexpr static bool IS_IN_T_8BIT =
      std::is_same_v<InputT, uint8_t> || std::is_same_v<InputT, int8_t>;
  // AscendC casts do not support unsigned integers in many
  // configurations.
  constexpr static bool IS_IN_T_UNSIGNED = std::is_same_v<InputT, uint8_t> ||
                                           std::is_same_v<InputT, uint16_t> ||
                                           std::is_same_v<InputT, uint32_t>;
  constexpr static bool IS_ACC_T_SUPPORTED =
      std::is_same_v<AccT, float> || std::is_same_v<AccT, int32_t>;
  constexpr static bool REQ_INTERMEDIATE_CAST = IS_IN_T_8BIT;

  const uint32_t vec_core_num_;
  const uint32_t tile_len_;
  const uint32_t vec_len_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;

  const int32_t red1_size_;
  constexpr static int32_t RED2_SIZE = 8;
};

}  // namespace tcuscan
