/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_scan2p_single_core.h
 * @brief Kernel implementing a 2P scan kernel operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Row scan on two input vectors using matrix multiplications
 *
 * @tparam T1 Data type of first vector
 * @tparam T2 Data type of second vector
 * @tparam SyncAfter Synchronize cube tiles with vector cores if true
 */
template <typename T1 = half, typename T2 = int8_t, bool SyncAfter = false>
class KernelScan2PSingleCore {
  using OutputT1 = tcuscan::cube_unit::CubeOutType_t<T1>;
  using OutputT2 = tcuscan::cube_unit::CubeOutType_t<T2>;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] matmul_m_size Size of the M dimension of A matrix.
   * @param [in] matmul_k_size Size of the K dimension of A matrix.
   * @param [in] vec_len Number of elements in an input vector.
   */
  __aicore__ inline KernelScan2PSingleCore(uint16_t matmul_m_size,
                                           uint16_t matmul_k_size,
                                           uint32_t vec_len)
      : vec_len_(vec_len),
        M_(matmul_m_size),
        K_(matmul_k_size),
        N_(matmul_k_size),
        num_matrix_tiles_(vec_len_ / (M_ * K_)) {
    static_assert(tcuscan::cube_unit::IsCubeSupported<T1>,
                  "Unsupported input Cube dtype in first parameter. Please "
                  "use half.");
    static_assert(tcuscan::cube_unit::IsCubeSupported<T2>,
                  "Unsupported input Cube dtype in second parameter. "
                  "Please use int8_t.");

#ifdef ASCEND_CPU_DEBUG
    ASCENDC_ASSERT(matmul_k_size % tcuscan::GetFractlK<T1> == 0, {
      KERNEL_LOG(KERNEL_ERROR,
                 "Matrix multiplication inner K dimension must be "
                 "divisible by fractal dimension.");
    });
    ASCENDC_ASSERT(matmul_k_size % tcuscan::GetFractlK<T2> == 0, {
      KERNEL_LOG(KERNEL_ERROR,
                 "Matrix multiplication inner K dimension must be "
                 "divisible by fractal dimension.");
    });
#endif
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to input vector in global memory.
   * @param [in] vec_f_in Pointer to input vector in global memory.
   * @param [in] b_half Pointer to upper triangular all-ones half matrix in
   * GM.
   * @param [in] b_int8 Pointer to upper triangular all-ones int8_t matrix in
   * GM.
   * @param [in] vec_out Pointer to output vector in global memory.
   * @param [in] vec_f_out Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR vec_f_in, GM_ADDR b_half,
                              GM_ADDR b_int8, GM_ADDR vec_out,
                              GM_ADDR vec_f_out) {
    global_in_.SetGlobalBuffer((__gm__ T1*)vec_in, vec_len_);
    global_f_in_.SetGlobalBuffer((__gm__ T2*)vec_f_in, vec_len_);
    global_b_half_.SetGlobalBuffer((__gm__ half*)b_half, M_ * K_);
    global_b_int8_.SetGlobalBuffer((__gm__ int8_t*)b_int8, K_ * N_);
    global_out_.SetGlobalBuffer((__gm__ OutputT1*)vec_out, vec_len_);
    global_f_out_.SetGlobalBuffer((__gm__ OutputT2*)vec_f_out, vec_len_);

    // 16-bits: largest cube input hence 'uint16_t'
    pipe.InitBuffer(a1_q_, 1, a_cube_tile_size_ * sizeof(uint16_t));
    pipe.InitBuffer(a2_q_, 1, a_cube_tile_size_ * sizeof(uint16_t));
    pipe.InitBuffer(b1_q_, 1, b_cube_tile_size_ * sizeof(uint16_t));
    pipe.InitBuffer(b2_q_, 1, b_cube_tile_size_ * sizeof(uint16_t));

    pipe.InitBuffer(co1_q_, 1, c_cube_tile_size_ * sizeof(uint32_t));
  }

  /**
   * @brief Run the kernel - process all tiles.
   */
  __aicore__ inline void Process() {
    for (uint32_t idx = 0; idx < num_matrix_tiles_; ++idx) {
      CubeIter(idx);
      if constexpr (SyncAfter) {
        sync::SyncGroup<sync::GroupSyncDirection::FULL>();
      }
    }
  }

 private:
  __aicore__ inline void CubeIter(uint32_t iter_idx) {
    copy::CopyTransposedGmToL0B(b2_q_, b1_q_, global_b_half_, k_half_blocks_,
                                n_half_blocks_);

    copy::CopyGmToL1A(a1_q_, global_in_[iter_idx * a_cube_tile_size_],
                      m_half_blocks_, k_half_blocks_);

    copy::CopyL1ToL0A<T1, true>(a2_q_, a1_q_, m_half_blocks_, k_half_blocks_);

    cube_unit::Multiply<T1, false /* accumulate_c */, true /* free_a*/,
                        true /* free_b */>(a2_q_, b2_q_, co1_q_, M_, N_, K_);
    copy::CopyCL0ToGlobal(global_out_[iter_idx * c_cube_tile_size_], co1_q_, M_,
                          N_);

    copy::CopyTransposedGmToL0B(b2_q_, b1_q_, global_b_int8_, k_int8_blocks_,
                                n_int8_blocks_);

    copy::CopyGmToL1A(a1_q_, global_f_in_[iter_idx * a_cube_tile_size_],
                      m_int8_blocks_, k_int8_blocks_);

    copy::CopyL1ToL0A<T2, true>(a2_q_, a1_q_, m_int8_blocks_, k_int8_blocks_);

    cube_unit::Multiply<T2, false /* accumulate_c */, true /* free_a*/,
                        true /* free_b */>(a2_q_, b2_q_, co1_q_, M_, N_, K_);

    copy::CopyCL0ToGlobal(global_f_out_[iter_idx * c_cube_tile_size_], co1_q_,
                          M_, N_);
  }

  TPipe pipe;

  TQue<QuePosition::A1, 1> a1_q_;
  TQue<QuePosition::A2, 1> a2_q_;
  TQue<QuePosition::B1, 1> b1_q_;
  TQue<QuePosition::B2, 1> b2_q_;

  TQue<QuePosition::CO1, 1> co1_q_;

  GlobalTensor<T1> global_in_;
  GlobalTensor<T2> global_f_in_;
  GlobalTensor<half> global_b_half_;
  GlobalTensor<int8_t> global_b_int8_;
  GlobalTensor<OutputT1> global_out_;
  GlobalTensor<OutputT2> global_f_out_;

  const uint32_t vec_len_;
  const uint16_t M_;
  const uint16_t K_;
  const uint16_t N_;
  const uint32_t num_matrix_tiles_;

  const uint16_t m_half_blocks_ = M_ / tcuscan::GetFractalMN<half>();
  const uint16_t m_int8_blocks_ = M_ / tcuscan::GetFractalMN<int8_t>();
  const uint16_t k_half_blocks_ = K_ / tcuscan::GetFractalK<half>();
  const uint16_t k_int8_blocks_ = K_ / tcuscan::GetFractalK<int8_t>();
  const uint16_t n_half_blocks_ = N_ / tcuscan::GetFractalMN<half>();
  const uint16_t n_int8_blocks_ = N_ / tcuscan::GetFractalMN<int8_t>();

  const uint32_t a_cube_tile_size_ = M_ * K_;
  const uint32_t b_cube_tile_size_ = K_ * N_;
  const uint32_t c_cube_tile_size_ = M_ * N_;
};

/**
 * @brief Run the `scan2p_single_core` kernel.
 *
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_f_in Pointer to the input vector.
 * @param [in] b_half Pointer to the input vector.
 * @param [in] b_int8 Pointer to the input vector.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] vec_f_out Pointer ot the output vector.
 * @param [in] vec_len Dimension of the input vector.
 * @param [in] matmul_size Number of rows/columns of U_s.
 */
template <bool ForceMixMode = true>
__aicore__ inline void run_scan2p_single_core(GM_ADDR vec_in, GM_ADDR vec_f_in,
                                              GM_ADDR b_half, GM_ADDR b_int8,
                                              GM_ADDR vec_out,
                                              GM_ADDR vec_f_out,
                                              uint32_t vec_len,
                                              uint32_t matmul_size) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIC {
    KernelScan2PSingleCore<half, int8_t, /*SyncAfter*/ false> op(
        matmul_size, matmul_size, vec_len);
    op.Init(vec_in, vec_f_in, b_half, b_int8, vec_out, vec_f_out);
    op.Process();
  }
}

}  // namespace tcuscan
