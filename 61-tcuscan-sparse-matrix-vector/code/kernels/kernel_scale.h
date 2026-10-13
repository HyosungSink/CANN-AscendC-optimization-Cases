/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * @file kernel_scale.h
 * @brief Kernel implementing an in-place vector scaling operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Performs the in-place multi-core AIV scaling `y = beta * y`.
 *
 * This is the `beta * y` term of the BLAS-style SpMV
 * \f$y \leftarrow \alpha A x + \beta y\f$. The SpMV kernels apply it as a
 * pre-pass over the output vector, before the (atomic-add) segment reduction
 * accumulates \f$\alpha A x\f$ on top of it.
 *
 * Following the BLAS convention, `beta == 0` overwrites @p y with zeros
 * *without reading it*, so an uninitialized (or NaN/Inf-carrying) output
 * vector is still valid input.
 *
 * @tparam T The data type of the scaled vector. Supports `float` and `half`.
 */
template <typename T>
class KernelScale {
  constexpr static uint32_t BUFFER_NUM = 2;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Length of the vector to scale.
   * @param [in] tile_len Length of the tile processed in a single iteration.
   */
  __aicore__ inline KernelScale(uint32_t vec_len, uint32_t tile_len)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        tile_len_(tile_len),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_len_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, vec_core_num_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in,out] vec_inout Pointer to the vector scaled in place.
   */
  __aicore__ inline void Init(GM_ADDR vec_inout) {
    global_.SetGlobalBuffer((__gm__ T*)vec_inout, vec_len_);

    pipe_.InitBuffer(in_q_, BUFFER_NUM, tile_len_ * sizeof(T));
    pipe_.InitBuffer(out_q_, BUFFER_NUM, tile_len_ * sizeof(T));
  }

  /**
   * @brief Run the kernel.
   *
   * @param [in] beta Scaling factor. A value of `0` zero-fills the vector
   * without reading it.
   */
  __aicore__ inline void Process(T beta) {
    const uint32_t num_tiles_to_process =
        scalar::GetWorkDistribution(vec_len_, tile_len_, vec_core_num_);
    const bool is_zero_fill = (beta == static_cast<T>(0));

    uint32_t gm_offset = GetBlockIdx() * tile_len_ * max_num_tiles_per_block_;
    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const bool is_full_tile = gm_offset + tile_len_ <= vec_len_;
      const uint32_t num_elems =
          is_full_tile ? tile_len_ : vec_len_ - gm_offset;

      LocalTensor<T> out_lt = out_q_.template AllocTensor<T>();
      if (is_zero_fill) {
        Duplicate<T>(out_lt, static_cast<T>(0), num_elems);
      } else {
        copy::CopyGmToVec(in_q_, global_[gm_offset], num_elems);
        LocalTensor<T> in_lt = in_q_.template DeQue<T>();
        Muls(out_lt, in_lt, beta, num_elems);
        in_q_.template FreeTensor<T>(in_lt);
      }
      out_q_.template EnQue<T>(out_lt);
      copy::CopyVecToGm(global_[gm_offset], out_q_, num_elems);

      gm_offset += tile_len_;
    }
  }

 private:
  TPipe pipe_;

  TQue<QuePosition::VECIN, BUFFER_NUM> in_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> out_q_;

  GlobalTensor<T> global_;

  const uint32_t vec_core_num_;
  const uint32_t vec_len_;
  const uint32_t tile_len_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
};

/// @brief Default tile length used by `run_scale_inplace`.
constexpr uint32_t SCALE_DEFAULT_TILE_LEN = 4096;

/**
 * @brief Run the in-place `y = beta * y` scaling kernel.
 *
 * No-op when `beta == 1`, so callers can invoke it unconditionally.
 *
 * @tparam T Scaled data type. Supports `half` and `float`.
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 *
 * @param [in,out] vec_inout Pointer to the vector scaled in place.
 * @param [in] vec_len Length of the vector to scale.
 * @param [in] beta Scaling factor.
 * @param [in] tile_len Length of the tile processed in a single iteration.
 *
 * @note The caller is responsible for synchronizing (`AscendC::SyncAll`)
 * between this pre-pass and any subsequent write to @p vec_inout.
 */
template <typename T, bool ForceMixMode = true>
__aicore__ inline void run_scale_inplace(
    GM_ADDR vec_inout, uint32_t vec_len, T beta,
    uint32_t tile_len = SCALE_DEFAULT_TILE_LEN) {
  if (beta == static_cast<T>(1) || vec_len == 0) {
    return;
  }

  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }

  if ASCEND_IS_AIV {
    static_assert(std::is_same_v<T, half> || std::is_same_v<T, float>,
                  "[scale] Unsupported input dtype");
    // Never tile wider than the vector itself, but keep the UB tile 32B
    // aligned so `InitBuffer` gets a well-formed size.
    constexpr uint32_t kUbBlock = UB_ALIGNMENT / sizeof(T);
    const uint32_t this_tile_len = scalar::AlignUp<uint32_t>(
        scalar::Min<uint32_t>(tile_len, vec_len), kUbBlock);

    KernelScale<T> op(vec_len, this_tile_len);
    op.Init(vec_inout);
    op.Process(beta);
  }
}

}  // namespace tcuscan
