/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_row_scan.h
 * @brief Kernel implementing row-wise scan using Cube unit.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "kernel_copy.h"
#include "kernel_pad_batch.h"
#include "kernel_unpad_batch.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Performs a "row-wise" inclusive scan on an input vector.
 *
 * The algorithm splits the input vector into chunks ("rows") of size
 * \f$K = \textit{matmul_k_size}\f$ and performs local inclusive scans on all
 * the chunks separately. Local scan is performed by transforming the problem
 * into matrix multiplication. Matrix \f$B\f$ is an upper triangular matrix of
 * size \f$K \times K\f$ with ones on the main diagonal and above. Matrix
 * \f$A\f$ is created from the input vector by reshaping it into a matrix with
 * \f$K\f$ dimension equal to \f$K\f$ and \f$M\f$ dimension equal to
 * \f$\textit{matmul_m_size}\f$. Then matrix \f$A\f$ is splitted into \f$M
 * \times K\f$ tiles that are multiplied by the same matrix \f$B\f$. The tiles
 * are equally distributed into different cube cores.
 *
 * The algorithm assumes that the length of the input vector is divisible by
 * \f$M K\f$.
 *
 * The algorithm is a first step in the full-scan calculation.
 *
 * The algorithm supports following data type configurations:
 *   - inputs: `int8_t`; output: `int32_t`;
 *   - inputs: `uint8_t`; output: `uint32_t`;
 *   - inputs: `half`; output: `float`.
 *
 * @tparam InputT Data type of the input vector.
 * @tparam SyncAfter If true, synchronize vector units with cube unit after each
 * matrix tile.
 */
template <typename InputT, bool SyncAfter = false>
class KernelRowScan {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<InputT>;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] matmul_k_size Size of the K dimension of A matrix.
   * @param [in] matmul_m_size Size of the M dimension of A matrix.
   * @param [in] vec_len Number of elements in an input vector.
   */
  __aicore__ inline KernelRowScan(uint32_t matmul_k_size,
                                  uint32_t matmul_m_size, uint32_t vec_len)
      : block_num_(GetBlockNum()),
        matmul_k_size_(matmul_k_size),
        matmul_m_size_(matmul_m_size),
        tile_size_(matmul_k_size * matmul_m_size),
        vec_len_(vec_len),
        num_tiles_(scalar::FloorDiv(vec_len, tile_size_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, block_num_)) {
    static_assert(
        tcuscan::cube_unit::IsCubeSupported<InputT>,
        "Unsupported input Cube dtype. Please use half, float, or int8_t.");
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input Pointer to input vector in global memory.
   * @param [in] b Pointer to upper triangular matrix filled with ones in
   * global memory.
   * @param [in] output Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR b, GM_ADDR output) {
    global_A_.SetGlobalBuffer((__gm__ InputT*)input, vec_len_);
    global_B_.SetGlobalBuffer((__gm__ InputT*)b);
    global_C_.SetGlobalBuffer((__gm__ OutputT*)output, vec_len_);

    pipe.InitBuffer(a1_q_, 1, a_cube_tile_size_ * sizeof(InputT));
    pipe.InitBuffer(a2_q_, 1, a_cube_tile_size_ * sizeof(InputT));
    pipe.InitBuffer(b1_q_, 1, b_cube_tile_size_ * sizeof(InputT));
    pipe.InitBuffer(b2_q_, 1, b_cube_tile_size_ * sizeof(InputT));

    pipe.InitBuffer(co1_q_, 1, c_cube_tile_size_ * sizeof(OutputT));
  }

  /**
   * @brief Run the kernel - process all tiles.
   */
  __aicore__ inline void Process() {
    // Load the B matrix only once from global memory
    const uint32_t num_tiles_to_process =
        tcuscan::scalar::GetWorkDistribution(vec_len_, tile_size_, block_num_);

    if (num_tiles_to_process == 0) {
      return;
    }

    LoadBToL0();
    for (uint32_t idx = 0; idx < num_tiles_to_process; ++idx) {
      CubeIter(idx);
      if constexpr (SyncAfter) {
        sync::SyncGroup<sync::GroupSyncDirection::FULL>();
      }
    }
    queue::FreeFromQ<InputT>(b2_q_);
  }

 private:
  __aicore__ inline void CubeIter(uint32_t iter_idx) {
    copy::CopyGmToL1A(
        a1_q_,
        global_A_[GetBlockIdx() * a_cube_tile_size_ * max_num_tiles_per_block_ +
                  iter_idx * a_cube_tile_size_],
        m_blocks_, k_blocks_);
    LoadA1ToA2();
    cube_unit::Multiply<InputT, false /* accumulate_c */, true /* free_a*/,
                        false /* free_b */>(a2_q_, b2_q_, co1_q_,
                                            matmul_m_size_, N_, K_);
    copy::CopyCL0ToGlobal(
        global_C_[GetBlockIdx() * c_cube_tile_size_ * max_num_tiles_per_block_ +
                  iter_idx * c_cube_tile_size_],
        co1_q_, matmul_m_size_, N_);
  }

  __aicore__ inline void LoadBToL0() {
    copy::CopyTransposedGmToL0B(b2_q_, b1_q_, global_B_, k_blocks_, n_blocks_);
  }

  __aicore__ inline void LoadA1ToA2() {
    copy::CopyL1ToL0A<InputT, true>(a2_q_, a1_q_, m_blocks_, k_blocks_);
  }

  TPipe pipe;

  TQue<QuePosition::A1, 1> a1_q_;
  TQue<QuePosition::A2, 1> a2_q_;
  TQue<QuePosition::B1, 1> b1_q_;
  TQue<QuePosition::B2, 1> b2_q_;

  TQue<QuePosition::CO1, 1> co1_q_;

  GlobalTensor<InputT> global_A_, global_B_;
  GlobalTensor<OutputT> global_C_;

  const uint32_t block_num_;
  const uint32_t matmul_k_size_;
  const uint32_t matmul_m_size_;
  const uint32_t tile_size_;
  const uint32_t vec_len_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;

  constexpr static uint32_t M_CUBE_BLOCK_SIZE = tcuscan::GetFractalMN<InputT>();
  constexpr static uint32_t N_CUBE_BLOCK_SIZE = tcuscan::GetFractalMN<InputT>();
  // Fractal size is 16x32 for 8-bit input data types instead of the standard
  // 16x16 one
  constexpr static uint32_t K_CUBE_BLOCK_SIZE = tcuscan::GetFractalK<InputT>();
  const uint32_t K_ = matmul_k_size_;
  const uint32_t N_ = matmul_k_size_;

  const uint32_t n_blocks_ = N_ / N_CUBE_BLOCK_SIZE;
  const uint32_t k_blocks_ = K_ / K_CUBE_BLOCK_SIZE;
  const uint32_t m_blocks_ = matmul_m_size_ / M_CUBE_BLOCK_SIZE;

  const uint32_t a_cube_tile_size_ = matmul_m_size_ * K_;
  const uint32_t b_cube_tile_size_ = K_ * N_;
  const uint32_t c_cube_tile_size_ = matmul_m_size_ * N_;
};

/**
 * @brief Run the row scan kernel with padding if needed.
 *
 * @tparam InputT Data type of the input vectors.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] upper_triangular Pointer to an upper-triangular matrix filled
 * with ones of size \f$\textit{matmul_size} \times \textit{matmul_size}\f$.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] vec_len Number of elements in one vector.
 * @param [in] batch_size Number of vectors in a batch.
 * @param [in] matmul_size Size of the matmul tile used in the kernel.
 * @param [in] workspace Pointer to a memory region used as workspace.
 */
template <typename InputT>
__aicore__ inline void run_row_scan_kernel(GM_ADDR input_vec,
                                           GM_ADDR upper_triangular,
                                           GM_ADDR output_vec, uint32_t vec_len,
                                           uint32_t batch_size,
                                           uint16_t matmul_size,
                                           GM_ADDR workspace) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<InputT>;

  if (vec_len == matmul_size && batch_size % matmul_size == 0) {
    if ASCEND_IS_AIC {
      KernelRowScan<InputT> op_cube(matmul_size, matmul_size,
                                    vec_len * batch_size);
      op_cube.Init(input_vec, upper_triangular, output_vec);
      op_cube.Process();
    }

  } else {
    const uint32_t vector_align_size = matmul_size;
    const uint32_t batch_align_size = matmul_size;
    const uint32_t aligned_vec_len =
        scalar::AlignUp(vec_len, vector_align_size);
    const uint32_t aligned_batch_size =
        scalar::AlignUp(batch_size, batch_align_size);

    const uint32_t padded_tensor_len = aligned_vec_len * aligned_batch_size;
    GM_ADDR const padded_input = workspace;
    GM_ADDR const padded_rowwise_scan =
        workspace + padded_tensor_len * sizeof(InputT);

    if (vec_len == matmul_size) {
      run_copy<OutputT, false /* ForceMixMode */>(
          input_vec, padded_input, batch_size * vec_len, matmul_size);
    } else {
      run_pad_batch<InputT, false>(input_vec, padded_input, vec_len, batch_size,
                                   vector_align_size, vector_align_size);
    }

    AscendC::SyncAll<false /*isAIVOnly*/>();
    sync::SyncGroup<sync::GroupSyncDirection::FULL>();
    PipeBarrier<PIPE_ALL>();

    if ASCEND_IS_AIC {
      KernelRowScan<InputT> op_cube(matmul_size, matmul_size,
                                    aligned_vec_len * aligned_batch_size);
      op_cube.Init(padded_input, upper_triangular, padded_rowwise_scan);
      op_cube.Process();
    }

    AscendC::SyncAll<false /*isAIVOnly*/>();
    sync::SyncGroup<sync::GroupSyncDirection::FULL>();
    PipeBarrier<PIPE_ALL>();

    if (vec_len == matmul_size) {
      run_copy<OutputT, false /* ForceMixMode */>(
          padded_rowwise_scan, output_vec, batch_size * vec_len, matmul_size);
    } else {
      if ASCEND_IS_AIV {
        KernelUnpadBatch<OutputT> op_vec(aligned_vec_len, batch_size, vec_len,
                                         matmul_size);
        op_vec.Init(padded_rowwise_scan, output_vec);
        op_vec.Process();
      }
    }
  }
}

}  // namespace tcuscan
