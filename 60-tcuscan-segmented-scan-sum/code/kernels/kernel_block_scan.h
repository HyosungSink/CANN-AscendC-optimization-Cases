/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file kernel_block_scan.h
 * @brief Kernel implementing block scan using multiple matrix multiplications.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Performs a block-wise inclusive scan on an input vector using the
 * "cube-only" algorithm.
 *
 * The algorithm assumes that the input vector length is divisible by the number
 * of blocks and the square of the tile size.
 *
 * The algorithm splits the input vector into \f$N \times N\f$ blocks (\f$N =
 * 128\f$). For input tile \f$A_i\f$, the result is calculated as \f$C_i = A_i U
 * + L A_i J\f$, where:
 * - \f$U\f$ is an upper triangular matrix of size \f$N \times N\f$ filled with
 * ones;
 * - \f$L\f$ is a strict lower triangular matrix of size \f$N \times N\f$ filled
 * with ones;
 * - \f$J\f$ is an all-ones matrix of size \f$N \times N\f$.
 *
 * The algorithm processes the blocks one after another. Computations on all
 * blocks are independent.
 *
 * Algorithm utilizes the cube in every iteration, the
 * sequence of operations to compute block \f$A_i\f$ (denoted for simplicity as
 * \f$A\f$) is as follow:
 * 1. \f$X_1 = A [1]\f$;
 * 2. \f$X_2 = A U\f$;
 * 3. \f$X_2 \mathrel{+}= L X_1\f$.
 *
 * @tparam T Input data type. Supported type is `half` / `fp16`.
 * @tparam SyncAfter If true, enables synchronization after each matrix tile
 * iteration with vector cores.
 */
template <typename T = half, bool SyncAfter = false>
class KernelBlockScan {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<T>;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Number of elements in an input vector.
   * @param [in] matrix_size Tile size of square matrices.
   */
  __aicore__ inline KernelBlockScan(uint32_t vec_len, uint32_t matrix_size)
      : vec_len_(vec_len),
        matmul_size_(matrix_size),
        num_tiles_(scalar::FloorDiv(vec_len, matmul_size_ * matmul_size_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, GetBlockNum())) {
    ASCENDC_ASSERT((vec_len_ % (matmul_size * matmul_size) == 0), {
      KERNEL_LOG(KERNEL_ERROR,
                 "The length of the input vector (%d) must be "
                 "divisible by the square of the tile size (%d)",
                 vec_len, matmul_size * matmul_size);
    });

    ASCENDC_ASSERT((num_tiles_ <= 0), {
      KERNEL_LOG(KERNEL_ERROR,
                 "Number of tiles is zero. Decrease the number of AI-core "
                 "blocks when running the kernel. Got block-num: (%d)",
                 GetBlockNum());
    });
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input Pointer to input vector in global memory.
   * @param [in] upper Pointer to upper triangular matrix filled with ones
   * in global memory.
   * @param [in] lower Pointer to strict lower triangular matrix filled
   * with ones.
   * @param [in] output Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR upper, GM_ADDR lower,
                              GM_ADDR output) {
    global_input_.SetGlobalBuffer(
        (__gm__ T*)input +
            (GetBlockIdx() * a_cube_tile_size_ * max_num_tiles_per_block_),
        a_cube_tile_size_ * max_num_tiles_per_block_);
    global_upper_.SetGlobalBuffer((__gm__ T*)upper, b_cube_tile_size_);
    global_lower_.SetGlobalBuffer((__gm__ T*)lower);
    global_output_.SetGlobalBuffer(
        (__gm__ OutputT*)output +
            (GetBlockIdx() * c_cube_tile_size_ * max_num_tiles_per_block_),
        c_cube_tile_size_ * max_num_tiles_per_block_);

    pipe.InitBuffer(a1_q_, 1, a_cube_tile_size_ * sizeof(T));
    pipe.InitBuffer(b1_1_q_, 1, b_cube_tile_size_ * sizeof(T));
    pipe.InitBuffer(b1_2_q_, 1, b_cube_tile_size_ * sizeof(T));

    pipe.InitBuffer(a2_1_q_, 1, a_cube_tile_size_ * sizeof(T));
    pipe.InitBuffer(a2_2_q_, 1, a_cube_tile_size_ * sizeof(T));

    pipe.InitBuffer(b2_1_q_, 1, b_cube_tile_size_ * sizeof(T));
    pipe.InitBuffer(b2_2_q_, 1, b_cube_tile_size_ * sizeof(T));

    pipe.InitBuffer(b1_buf_, b_cube_tile_size_ * sizeof(T));

    pipe.InitBuffer(co1_q_, 1, c_cube_tile_size_ * sizeof(OutputT));
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    const uint32_t num_tiles_to_process = tcuscan::scalar::GetWorkDistribution(
        vec_len_, matmul_size_ * matmul_size_, GetBlockNum());

    if (num_tiles_to_process == 0) {
      return;
    }

    // A1: L
    tcuscan::copy::CopyGmToL1A(a1_q_, global_lower_, m_blocks_, k_blocks_);

    // A2: L
    copy::CopyL1ToL0A<T, true>(a2_2_q_, a1_q_, m_blocks_, k_blocks_);

    // A2: L, A1: L, B1: [1]
    LocalTensor<T> all_ones_1_lt_ = b1_buf_.Get<T>();
    cube_unit::InitConstL1<T, TPosition::B1>(
        all_ones_1_lt_, 1, static_cast<uint16_t>(b_cube_tile_size_));

    // A2: L; B1: U; B2: [1]
    tcuscan::copy::CopyGmToL1B(b1_1_q_, global_upper_, k_blocks_, n_blocks_);

    copy::CopyL1ToL0B<T>(b2_1_q_, all_ones_1_lt_, k_blocks_, n_blocks_);

    for (uint32_t idx = 0; idx < num_tiles_to_process; ++idx) {
      // A1: A; A2: L; B1: U; B2: [1]
      copy::CopyGmToL1A(a1_q_, global_input_[idx * a_cube_tile_size_],
                        m_blocks_, k_blocks_);
      // A2: L, A; B1: U; B2: [1]
      copy::CopyL1ToL0A<T, true>(a2_1_q_, a1_q_, m_blocks_, k_blocks_);

      // X_1 = A x [1]
      // A2: L, A; B1: U; B2: [1]; C01: X_1
      cube_unit::Multiply<T, false /* accumulate_c */, false /* free_a*/,
                          false /* free_b */>(a2_1_q_, b2_1_q_, co1_q_, M_, N_,
                                              K_);

      // A2: L, A; B1: U, X_1; B2: [1]
      copy::CopyC01ToB1<T, float>(b1_2_q_, co1_q_, M_, N_);

      // A2: L, A; B1: U, X_1; B2: [1], U
      copy::CopyL1ToL0B<T, false>(b2_2_q_, b1_1_q_, k_blocks_, n_blocks_);

      // X_2 = A x U
      // A2: L; B1: U, X_1; B2: [1]; C01: X_2
      cube_unit::Multiply<T, false /* accumulate_c */, true /* free_a*/,
                          true /* free_b */>(a2_1_q_, b2_2_q_, co1_q_, M_, N_,
                                             K_);

      // A2: L; B1: U; B2: [1], X_1; C01: X_2
      copy::CopyL1ToL0B<T, true>(b2_2_q_, b1_2_q_, k_blocks_, n_blocks_);

      // X_2 += L x X_1
      // A2: L; B1: U; B2: [1]; C01: X_2
      cube_unit::Multiply<T, true /* accumulate_c */, false /* free_a*/,
                          true /* free_b */>(a2_2_q_, b2_2_q_, co1_q_, M_, N_,
                                             K_);

      // A2: L; B1: U; B2: [1]
      copy::CopyCL0ToGlobal(global_output_[idx * c_cube_tile_size_], co1_q_,
                            matmul_size_, N_);

      if constexpr (SyncAfter) {
        sync::SyncGroup<sync::GroupSyncDirection::FULL>();
      }
    }
    // B1: U; B2: [1];
    queue::FreeFromQ<T>(a2_2_q_);
    // B1: U
    queue::FreeFromQ<T>(b2_1_q_);
    // All empty
    queue::FreeFromQ<T>(b1_1_q_);
  }

 private:
  TPipe pipe;

  TQue<QuePosition::A1, 1> a1_q_;
  TQue<QuePosition::B1, 1> b1_1_q_;
  TQue<QuePosition::B1, 1> b1_2_q_;

  TQue<QuePosition::A2, 1> a2_1_q_;
  TQue<QuePosition::A2, 1> a2_2_q_;
  TQue<QuePosition::B2, 1> b2_1_q_;
  TQue<QuePosition::B2, 1> b2_2_q_;

  TQue<QuePosition::CO1, 1> co1_q_;

  TBuf<QuePosition::B1> b1_buf_;

  GlobalTensor<T> global_input_;
  GlobalTensor<T> global_upper_;
  GlobalTensor<T> global_lower_;
  GlobalTensor<OutputT> global_output_;

  const uint32_t vec_len_;
  const uint32_t matmul_size_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;

  const uint32_t K_ = matmul_size_;
  const uint32_t N_ = matmul_size_;
  const uint32_t M_ = matmul_size_;

  const uint32_t n_blocks_ = N_ / tcuscan::GetFractalMN<T>();
  const uint32_t k_blocks_ = K_ / tcuscan::GetFractalK<T>();
  const uint32_t m_blocks_ = M_ / tcuscan::GetFractalMN<T>();

  const uint32_t a_cube_tile_size_ = M_ * K_;
  const uint32_t b_cube_tile_size_ = K_ * N_;
  const uint32_t c_cube_tile_size_ = M_ * N_;
};

}  // namespace tcuscan
