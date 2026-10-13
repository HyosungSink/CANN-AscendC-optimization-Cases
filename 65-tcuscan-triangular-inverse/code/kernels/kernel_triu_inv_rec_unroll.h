/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_triu_inv_rec_unroll.h
 * @brief Kernel implementing a Vector XXX kernel operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

/**
 * @brief It takes as input a three dimensional tensor U, containing
 * block_dim strictly upper triangular matrices of size matrix_size *
 * matrix_size each. It prepares three new tensors:
 * U_neg: contains the negation -U_k, for each U_k in U.
 * I: contains block_dim identity matrices.
 * X: contains I+U_neg.
 *
 * @tparam InputT Data type of the input matrix.
 */
template <typename InputT>
class KernelPrepareIdentityAndMinusU {
 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] matrix_size Number of rows and columns of each matrix.
   * @param [in] block_dim Number of triangular matrices.
   */
  __aicore__ inline KernelPrepareIdentityAndMinusU(uint32_t matrix_size,
                                                   uint32_t block_dim)
      : matrix_size_(matrix_size),
        block_dim_(block_dim),
        global_offset_(
            tcuscan::scalar::FloorDiv(GetBlockIdx(), GetTaskRation()) *
            matrix_size_ * matrix_size_) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] U Pointer to the input strictly upper triang. matrix.
   * @param [in] U_neg Pointer to the matrix that will store -U.
   * @param [in] X Pointer to the matrix in GM, which will store I-U.
   * @param [in] I Pointer to GM, which will store I.
   */
  __aicore__ inline void Init(GM_ADDR U, GM_ADDR U_neg, GM_ADDR X, GM_ADDR I) {
    global_U_in_.SetGlobalBuffer((__gm__ InputT*)U);
    global_U_neg_out_.SetGlobalBuffer((__gm__ InputT*)U_neg);
    global_X_out_.SetGlobalBuffer((__gm__ InputT*)X);
    global_I_out_.SetGlobalBuffer((__gm__ InputT*)I);

    pipe_.InitBuffer(vecin_U_q_, 1, matrix_size_ * sizeof(InputT));
    pipe_.InitBuffer(vecout_X_q_, 1, matrix_size_ * sizeof(InputT));
    pipe_.InitBuffer(vecout_U_neg_q_, 1, matrix_size_ * sizeof(InputT));
    pipe_.InitBuffer(vecout_I_q_, 1, matrix_size_ * sizeof(InputT));
    pipe_.InitBuffer(work_buf_, matrix_size_ * sizeof(InputT));
    pipe_.InitBuffer(work_buf2_, matrix_size_ * sizeof(InputT));
  }

  /**
   * @brief Run the kernel - process all tiles.
   */
  __aicore__ inline void Process() {
    uint32_t chunk_size =
        tcuscan::scalar::CeilDiv(matrix_size_, GetTaskRation());
    uint32_t start = GetSubBlockIdx() * chunk_size;
    uint32_t end = start + chunk_size;
    for (uint32_t j = start; j < end; ++j) {
      VecIter(j);
    }
  }

 private:
  __aicore__ inline void VecIter(uint32_t j) {
    const uint32_t offset = global_offset_ + j * matrix_size_;
    tcuscan::copy::CopyGmToVec(vecin_U_q_, global_U_in_[offset]);

    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    LocalTensor<InputT> vec_U_lt = vecin_U_q_.DeQue<InputT>();
    LocalTensor<InputT> vec_ej_lt =
        work_buf_.Get<InputT>();  // j-th column of identity
    LocalTensor<InputT> vec_U_neg_lt = work_buf2_.Get<InputT>();
    Duplicate(vec_ej_lt, static_cast<InputT>(0), matrix_size_);
    vec_ej_lt.SetValue(j, 1);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Muls(vec_U_neg_lt, vec_U_lt, (InputT)-1, matrix_size_);

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Add(vec_U_lt, vec_ej_lt, vec_U_neg_lt, matrix_size_);
    AscendC::PipeBarrier<PIPE_ALL>();
    tcuscan::copy::CopyVecToGm<InputT>(global_I_out_[offset], vecout_I_q_,
                                       vec_ej_lt);
    tcuscan::copy::CopyVecToGm<InputT>(global_U_neg_out_[offset],
                                       vecout_U_neg_q_, vec_U_neg_lt);
    tcuscan::copy::CopyVecToGm<InputT>(global_X_out_[offset], vecout_X_q_,
                                       vec_U_lt);
    vecin_U_q_.FreeTensor<InputT>(vec_U_lt);
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
  }

  TPipe pipe_;

  TQue<QuePosition::VECIN, 1> vecin_U_q_;
  TQue<QuePosition::VECOUT, 1> vecout_I_q_;
  TQue<QuePosition::VECOUT, 1> vecout_X_q_;
  TQue<QuePosition::VECOUT, 1> vecout_U_neg_q_;
  TBuf<QuePosition::VECCALC> work_buf_;
  TBuf<QuePosition::VECCALC> work_buf2_;

  GlobalTensor<InputT> global_U_in_;
  GlobalTensor<InputT> global_U_neg_out_;
  GlobalTensor<InputT> global_X_out_;
  GlobalTensor<InputT> global_I_out_;

  const uint32_t matrix_size_;
  const uint32_t block_dim_;
  const uint32_t global_offset_;
};

/**
 * @brief Given as input the strictly upper triangular matrix U, the matrix I-U,
 * and an integer b, it returns a matrix which contains the inverses of the b*b
 * diagonal blocks of I+U. b must be greater than or equal to the smallest
 * fractal size for the given data type.
 *
 * @tparam InputT Data type of the input matrix.
 */
template <typename InputT>
class KernelInvTriuRecUnroll {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<InputT>;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] matrix_size Number of rows and columns of the matrix.
   */
  __aicore__ inline KernelInvTriuRecUnroll(uint32_t matrix_size)
      : matrix_size_(matrix_size),
        tile_len_(matrix_size_ * matrix_size_),
        fractal_size_(tcuscan::GetFractalMN<InputT>()),
        n_fractals_(matrix_size_ / fractal_size_),
        global_offset_(GetBlockIdx() * tile_len_) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] U Pointer to the input strictly upper triangular matrix.
   * @param [in] U_neg Pointer to -U.
   * @param [in] X Pointer to the matrix I-U.
   * @param [in] I Pointer to the matrix I.
   * @param [in] Z Pointer to the output matrix.
   */
  __aicore__ inline void Init(GM_ADDR U, GM_ADDR U_neg, GM_ADDR X, GM_ADDR I,
                              GM_ADDR Z) {
    global_U_in_.SetGlobalBuffer((__gm__ InputT*)U);
    global_U_neg_in_.SetGlobalBuffer((__gm__ InputT*)U_neg);
    global_X_in_.SetGlobalBuffer((__gm__ InputT*)X);
    global_I_in_.SetGlobalBuffer((__gm__ InputT*)I);
    global_Z_out_.SetGlobalBuffer((__gm__ OutputT*)Z);

    pipe_.InitBuffer(A1_1_q_, 1, tile_len_ * sizeof(InputT));
    pipe_.InitBuffer(A1_2_q_, 1, tile_len_ * sizeof(InputT));
    pipe_.InitBuffer(B1_1_q_, 1, tile_len_ * sizeof(InputT));
    pipe_.InitBuffer(B1_2_q_, 1, tile_len_ * sizeof(InputT));
    pipe_.InitBuffer(B1_X_q_, 1, tile_len_ * sizeof(InputT));
    pipe_.InitBuffer(CO1_q_, 1, tile_len_ * sizeof(OutputT));
    pipe_.InitBuffer(A2_q_, 1, tile_len_ * sizeof(InputT));
    pipe_.InitBuffer(B2_q_, 1, tile_len_ * sizeof(InputT));
  }
  /**
   * @brief Main method to compute the inverse.
   */
  __aicore__ inline void Process() {
    CopyIn();

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    ProcessInvTrick();

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    ProcessInvRecUnroll();

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    CopyOut();

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    FreeQueues();

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
  }
  /**
   * @brief Only the diagonal fractals of a global tensor in an L1A or L1B
   * queue.
   *
   * @tparam Pos Queue position in A1 or B1
   *
   * @param [in] local_tensor a local tensor in A1 or B1 queue
   * @param [in] global_tensor a global tensor containing the matrix to copy
   */
  template <TPosition Pos>
  __aicore__ inline void CopyDiagonalFractalsFromGmToL1(
      LocalTensor<InputT> local_tensor, GlobalTensor<InputT> global_tensor) {
    tcuscan::cube_unit::InitConstL1<InputT, Pos>(local_tensor, (InputT)0,
                                                 tile_len_);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    Nd2NzParams params;
    params.ndNum = matrix_size_ / fractal_size_;  // number of diagonal fractals
    params.nValue = fractal_size_;  // number of rows of each diagonal fractal
    params.dValue = fractal_size_;  // number of cols of each diagonal fractal
    params.srcNdMatrixStride =
        matrix_size_ * fractal_size_ +
        fractal_size_;  // total elements between starting addresses of
                        // adjacent fractals in the input (full) matrix
    params.srcDValue = matrix_size_;  // number of cols of the full matrix
    params.dstNzC0Stride = 1;         // Should not matter
    params.dstNzNStride = 1;  // Rows of independent fractals are consecutive
    params.dstNzMatrixStride =
        matrix_size_ * fractal_size_ + fractal_size_ * fractal_size_;
    AscendC::DataCopy(local_tensor, global_tensor[global_offset_], params);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
  }
  /**
   * @brief Copy data from GM to local tensors.
   */
  __aicore__ inline void CopyIn() {
    LocalTensor<InputT> a1_y_lt = A1_2_q_.template AllocTensor<InputT>();
    CopyDiagonalFractalsFromGmToL1<TPosition::A1>(a1_y_lt, global_U_in_);

    A1_2_q_.EnQue(a1_y_lt);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    LocalTensor<InputT> b1_y_lt = B1_2_q_.template AllocTensor<InputT>();
    CopyDiagonalFractalsFromGmToL1<TPosition::B1>(b1_y_lt, global_U_in_);

    B1_2_q_.EnQue(b1_y_lt);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    LocalTensor<InputT> a1_x_lt = A1_1_q_.template AllocTensor<InputT>();
    CopyDiagonalFractalsFromGmToL1<TPosition::A1>(a1_x_lt, global_X_in_);
    A1_1_q_.EnQue(a1_x_lt);

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    LocalTensor<InputT> b1_i_lt = B1_1_q_.template AllocTensor<InputT>();
    CopyDiagonalFractalsFromGmToL1<TPosition::B1>(b1_i_lt, global_I_in_);
    B1_1_q_.EnQue(b1_i_lt);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
  }
  /**
   * @brief Compute the inverses of the 16*16 diagonal blocks using the
   * "inv_trick" algorithm. In Python/NumPy notation it executes the
   * following:
   * X, Y = (I - U, U @ U)
   * for i in range(log2(b) - 1):
   *     X, Y = (X + X @ Y, Y @ Y) return X
   */
  __aicore__ inline void ProcessInvTrick() {
    tcuscan::exec_mode::AssertIsAIC();
    tcuscan::exec_mode::AssertIsAIC();
    for (uint32_t j = 1; j < fractal_size_ / 4; j *= 2) {
      SquareY();  // Y <- Y @ Y
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::DataSyncBarrier<MemDsbT::ALL>();
      UpdateX(false /* store_co1_to_b1 */);  // X <- X + X @ Y
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::DataSyncBarrier<MemDsbT::ALL>();
    }
    SquareY();  // Y <- Y @ Y
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    UpdateX(true /* store_co1_to_b1 */);  // X <- X + X @ Y
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
  }
  /**
   * @brief As the name says, it computes Y @ Y
   */
  __aicore__ inline void SquareY() {
    tcuscan::copy::CopyL1ToL0A<InputT, /* FreeSrc */ true>(
        A2_q_, A1_2_q_, n_fractals_, n_fractals_);
    tcuscan::copy::CopyL1ToL0B<InputT, /* FreeSrc */ true>(
        B2_q_, B1_2_q_, n_fractals_, n_fractals_);

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    tcuscan::cube_unit::Multiply<InputT, false /* accumulate_c */,
                                 true /*free_a */, true /* free_b */>(
        A2_q_, B2_q_, CO1_q_, matrix_size_, matrix_size_, matrix_size_);

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();

    tcuscan::copy::CopyC01ToB1<InputT, OutputT, 1, 1, false /* FreeSrc */>(
        B1_2_q_, CO1_q_, matrix_size_, matrix_size_);

    tcuscan::copy::CopyC01ToA1<InputT, OutputT, 1, 1, true /* FreeSrc */>(
        A1_2_q_, CO1_q_, matrix_size_, matrix_size_);
  }
  /**
   * @brief It updates the matrix X as X <- X + X @ Y
   *
   * @param [in] store_co1_to_b1 whether to also store the result of CO1
   * to B1, or only in A1
   */
  __aicore__ inline void UpdateX(bool store_co1_to_b1) {
    tcuscan::copy::CopyL1ToL0A<InputT, false>(A2_q_, A1_1_q_, n_fractals_,
                                              n_fractals_);
    tcuscan::copy::CopyL1ToL0B<InputT, false>(B2_q_, B1_1_q_, n_fractals_,
                                              n_fractals_);

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    tcuscan::cube_unit::Multiply<InputT, false /* accumulate_c */,
                                          true /* free_a */, true /* free_b
                                          */>(
            A2_q_, B2_q_, CO1_q_, matrix_size_, matrix_size_, matrix_size_);

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();

    tcuscan::copy::CopyL1ToL0B<InputT, false>(B2_q_, B1_2_q_, n_fractals_,
                                              n_fractals_);
    tcuscan::copy::CopyL1ToL0A<InputT, true>(A2_q_, A1_1_q_, n_fractals_,
                                             n_fractals_);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    tcuscan::cube_unit::Multiply<InputT, true /* accumulate_c */,
                                          true /* free_a */, true /* free_b
                                          */>(
            A2_q_, B2_q_, CO1_q_, matrix_size_, matrix_size_, matrix_size_);

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();

    if (store_co1_to_b1) {
      tcuscan::copy::CopyC01ToA1<InputT, OutputT, 1, 1,
                                 /* FreeSrc */ false>(
          A1_1_q_, CO1_q_, matrix_size_, matrix_size_);
      tcuscan::copy::CopyC01ToB1<InputT, OutputT, 1, 1,
                                 /* FreeSrc */ true>(
          B1_X_q_, CO1_q_, matrix_size_, matrix_size_);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::DataSyncBarrier<MemDsbT::ALL>();
      tcuscan::queue::FreeFromQ<InputT>(B1_2_q_);
    } else {
      tcuscan::copy::CopyC01ToA1<InputT, OutputT, 1, 1,
                                 /* FreeSrc */ true>(
          A1_1_q_, CO1_q_, matrix_size_, matrix_size_);
    }

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
  }

  /**
   * @brief ProcessInvRecUnroll() uses the output of ProcessInvTrick() to
   * compute the final inverse of the matrix I+U. Every iteration computes
   * the following update:
   *
   *     X <- LX * U_neg * RX + LX + RX,
   *
   * where LX/RX contain the even/odd diagonal blocks of X. The following
   * steps are used:
   * while block_size < matrix_size_:
   *
   *    1) CO1 <- I @ I
   *    2) A2 <- LX
   *    3) A1_X <- A2 @ B1_U_neg + CO1
   *    4) CO1 <- LX
   *    5) B2 <- RX
   *    6) CO1 <- A1_X @ B2 + CO1, where A1_X contains LX @ U_neg + I (step 3)
   *    7) [A1, B1] <- CO1
   *    8) double the block_size
   *
   * The reason top use this more complex iteration is because
   * ProcessInvTrick is very numerically unstable.
   */
  __aicore__ inline void ProcessInvRecUnroll() {
    LocalTensor<InputT> a1_i_lt = A1_2_q_.DeQue<InputT>();
    CopyDiagonalFractalsFromGmToL1<TPosition::A1>(a1_i_lt, global_I_in_);

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    A1_2_q_.EnQue(a1_i_lt);

    LocalTensor<InputT> b1_u_neg_lt = B1_2_q_.AllocTensor<InputT>();
    tcuscan::copy::CopyND2NZ(b1_u_neg_lt, global_U_neg_in_[global_offset_],
                             n_fractals_, n_fractals_);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    B1_2_q_.EnQue<InputT>(b1_u_neg_lt);

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    for (uint16_t block_size = 16; block_size < matrix_size_; block_size *= 2) {
      LoadIdentityInCO1();
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::DataSyncBarrier<MemDsbT::ALL>();
      LoadEvenBlocksOfXInA2(block_size);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::DataSyncBarrier<MemDsbT::ALL>();
      ComputeStep3();
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::DataSyncBarrier<MemDsbT::ALL>();
      LoadLXInCO1();
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::DataSyncBarrier<MemDsbT::ALL>();
      LoadOddBlocksOfXInB2(block_size);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::DataSyncBarrier<MemDsbT::ALL>();
      ComputeStep6(block_size);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::DataSyncBarrier<MemDsbT::ALL>();
    }
  }

  /**
   * @brief This method copies the even (0,2,4...) diagonal blocks of size
   * block_size * block_size of the matrix that is stored in A1_1_q_ to
   * the local tensor in the A2_q_. The two matrices have the same size.
   * The rest of the elements are set to zero.
   *
   * @param [in] block_size size of blocks, as documented above.
   */
  __aicore__ inline void LoadEvenBlocksOfXInA2(uint16_t block_size) {
    LocalTensor<InputT> A1_X_lt = A1_1_q_.DeQue<InputT>();
    LocalTensor<InputT> A2_LX_lt = A2_q_.template AllocTensor<InputT>();

    const uint16_t n_data_blocks = tile_len_ * sizeof(InputT) / (uint16_t)512;
    AscendC::InitConstValue(A2_LX_lt, {1, n_data_blocks, 0, 0});
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();

    uint16_t n_fractals_per_block = block_size / fractal_size_;
    uint16_t n_elems_per_fractal = fractal_size_ * fractal_size_;
    LoadData2DParams params;
    params.startIndex = 0;
    params.repeatTimes = matrix_size_ / (2 * block_size);
    params.srcStride =
        2 * n_fractals_ * n_fractals_per_block + 2 * n_fractals_per_block;
    params.sid = 0;  // Reserved parameter, always 0
    params.dstGap = params.srcStride - 1;
    params.addrMode = 0;  // Reserved parameter, always 0
    params.ifTranspose = false;
    for (uint16_t i = 0; i < n_fractals_per_block; ++i) {
      for (uint16_t j = 0; j < n_fractals_per_block; ++j) {
        uint16_t src_start_index =
            j * n_fractals_ * n_elems_per_fractal + i * n_elems_per_fractal;
        uint16_t dst_start_index =
            i * n_fractals_ * n_elems_per_fractal + j * n_elems_per_fractal;
        LoadData(A2_LX_lt[dst_start_index], A1_X_lt[src_start_index], params);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<MemDsbT::ALL>();
      }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    A1_1_q_.FreeTensor(A1_X_lt);
    A2_q_.EnQue(A2_LX_lt);
  }

  /**
   * @brief This method copies the odd (1,3,5...) diagonal blocks of size
   * block_size * block_size of the matrix that is stored in B1_X_q_ to
   * the local tensor in the B2_q_. The two matrices have the same size.
   * The rest of the elements are set to zero.
   *
   * @param [in] block_size size of blocks, as documented above.
   */
  __aicore__ inline void LoadOddBlocksOfXInB2(uint16_t block_size) {
    LocalTensor<InputT> B1_X_lt = B1_X_q_.template DeQue<InputT>();
    LocalTensor<InputT> B2_RX_lt = B2_q_.template AllocTensor<InputT>();

    const uint16_t n_data_blocks = tile_len_ * sizeof(InputT) / (uint16_t)512;
    AscendC::InitConstValue(B2_RX_lt, {1, n_data_blocks, 0, 0});
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();

    uint16_t n_fractals_per_block = block_size / fractal_size_;
    uint16_t n_elems_per_fractal = fractal_size_ * fractal_size_;
    LoadData2DParams params;
    params.startIndex = 0;
    params.repeatTimes = matrix_size_ / (2 * block_size);
    params.srcStride =
        2 * n_fractals_ * n_fractals_per_block + 2 * n_fractals_per_block;
    params.sid = 0;  // Reserved parameter, always 0
    params.dstGap = params.srcStride - 1;
    params.addrMode = 0;  // Reserved parameter, always 0
    params.ifTranspose = true;
    for (uint16_t i = 0; i < n_fractals_per_block; ++i) {
      for (uint16_t j = 0; j < n_fractals_per_block; ++j) {
        uint16_t src_start_index =
            n_fractals_ * n_fractals_per_block * n_elems_per_fractal +
            n_fractals_per_block * n_elems_per_fractal +
            j * n_fractals_ * n_elems_per_fractal + i * n_elems_per_fractal;
        uint16_t dst_start_index =
            n_fractals_ * n_fractals_per_block * n_elems_per_fractal +
            n_fractals_per_block * n_elems_per_fractal +
            i * n_fractals_ * n_elems_per_fractal + j * n_elems_per_fractal;
        LoadData(B2_RX_lt[dst_start_index], B1_X_lt[src_start_index], params);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<MemDsbT::ALL>();
      }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    B1_X_q_.FreeTensor(B1_X_lt);
    B2_q_.EnQue(B2_RX_lt);
  }

  /**
   * @brief Copies the identity matrix to B2 and A2 queues, and then
   * stores it in CO1 using a matrix multiplication.
   */
  __aicore__ inline void LoadIdentityInCO1() {
    tcuscan::copy::CopyL1ToL0A<InputT, false>(A2_q_, A1_2_q_, n_fractals_,
                                              n_fractals_);
    tcuscan::copy::CopyL1ToL0B<InputT, false>(B2_q_, B1_1_q_, n_fractals_,
                                              n_fractals_);

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    tcuscan::cube_unit::Multiply<InputT, false /* accumulate_c */,
                                 true /* free_a */, true /* free_b */>(
        A2_q_, B2_q_, CO1_q_, matrix_size_, matrix_size_, matrix_size_);

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
  }

  /**
   * @brief Step 3) A1_Y <- LX @ U_neg + I
   */
  __aicore__ inline void ComputeStep3() {
    tcuscan::copy::CopyL1ToL0B<InputT, /* FreeSrc */ false>(
        B2_q_, B1_2_q_, n_fractals_, n_fractals_);

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    tcuscan::cube_unit::Multiply<InputT, true /* accumulate_c */,
                                 false /* free_a */, true /* free_b */>(
        A2_q_, B2_q_, CO1_q_, matrix_size_, matrix_size_, matrix_size_);

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    tcuscan::copy::CopyC01ToA1<InputT, OutputT, 1, 1,
                               /* FreeSrc */ true>(A1_1_q_, CO1_q_,
                                                   matrix_size_, matrix_size_);
  }

  /**
   * @brief Loads the matrix LX to CO1, which contains the even blocks of
   * X
   */
  __aicore__ inline void LoadLXInCO1() {
    tcuscan::copy::CopyL1ToL0B<InputT, /* FreeSrc */ false>(
        B2_q_, B1_1_q_, n_fractals_, n_fractals_);

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    tcuscan::cube_unit::Multiply<InputT, false /* accumulate_c */,
                                 true /* free_a */, true /* free_b */>(
        A2_q_, B2_q_, CO1_q_, matrix_size_, matrix_size_,
        matrix_size_);  // CO1 contains LX

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
  }

  /**
   * @brief Step 6) CO1 <- A1 @ B1 + CO1
   *
   * @param [in] block_size size of blocks, as documented above.
   */
  __aicore__ inline void ComputeStep6(uint32_t block_size) {
    bool is_last_iter = (block_size == matrix_size_ / 2);
    tcuscan::copy::CopyL1ToL0A<InputT, /* FreeSrc */ true>(
        A2_q_, A1_1_q_, n_fractals_, n_fractals_);

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    tcuscan::cube_unit::Multiply<InputT, true /* accumulate_c */,
                                 true /* free_a */, true /* free_b */>(
        A2_q_, B2_q_, CO1_q_, matrix_size_, matrix_size_, matrix_size_);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();

    if (!is_last_iter) {
      tcuscan::copy::CopyC01ToA1<InputT, OutputT, 1, 1, false /* FreeSrc */>(
          A1_1_q_, CO1_q_, matrix_size_, matrix_size_);
      tcuscan::copy::CopyC01ToB1<InputT, OutputT, 1, true /* FreeSrc */>(
          B1_X_q_, CO1_q_, matrix_size_, matrix_size_);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::DataSyncBarrier<MemDsbT::ALL>();
    }
  }

  /**
   * @brief Copy output to GM.
   */
  __aicore__ inline void CopyOut() {
    tcuscan::exec_mode::AssertIsAIC();
    constexpr uint16_t fractal_size = tcuscan::GetFractalMN<OutputT>();

    LocalTensor<OutputT> lt = CO1_q_.template DeQue<OutputT>();
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    uint32_t height = matrix_size_;
    uint32_t width = matrix_size_;
    FixpipeParams<OutputT> params;
    params.cburstNum = height;
    params.burstLen = width * fractal_size * sizeof(OutputT) / 32;
    params.dstStride = height;

    Nz2NdParams nz2nd_params;
    nz2nd_params.nz2ndEn = true;
    nz2nd_params.originalNSize = height;
    params.nz2ndParams = nz2nd_params;

    Fixpipe(global_Z_out_[global_offset_], lt, params);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataSyncBarrier<MemDsbT::ALL>();

    CO1_q_.FreeTensor(lt);
  }
  /**
   * @brief Free queues
   */
  __aicore__ inline void FreeQueues() {
    // tcuscan::queue::FreeFromQ<InputT>(A1_1_q_);
    tcuscan::queue::FreeFromQ<InputT>(A1_2_q_);
    tcuscan::queue::FreeFromQ<InputT>(B1_1_q_);
    tcuscan::queue::FreeFromQ<InputT>(B1_2_q_);
    // tcuscan::queue::FreeFromQ<InputT>(B1_X_q_);
  }

 private:
  TPipe pipe_;

  TQue<QuePosition::A1, 1> A1_2_q_;
  TQue<QuePosition::A1, 1> A1_1_q_;
  TQue<QuePosition::B1, 1> B1_1_q_;
  TQue<QuePosition::B1, 1> B1_2_q_;

  TQue<QuePosition::B1, 1> B1_X_q_;
  TQue<QuePosition::CO1, 1> CO1_q_;

  TQue<QuePosition::A2, 1> A2_q_;
  TQue<QuePosition::B2, 1> B2_q_;

  GlobalTensor<InputT> global_U_in_;
  GlobalTensor<InputT> global_U_neg_in_;
  GlobalTensor<InputT> global_X_in_;
  GlobalTensor<InputT> global_I_in_;
  GlobalTensor<OutputT> global_Z_out_;

  const uint32_t matrix_size_;
  const uint32_t tile_len_;
  const uint32_t fractal_size_;
  const uint32_t n_fractals_;
  const uint32_t global_offset_;
};

/**
 * @brief Run the `triu_inv_rec_unroll` kernel.
 *
 * @tparam InputT Data type of the input matrices.
 *
 * @param [in] U Pointer to the input strictly upper triangular matrices.
 * @param [in] out_tensor Pointer to the inverses of I+U_k, for all U_k in U.
 * @param [in] matrix_size Number of rows and columns of the input matrices.
 * @param [in] block_dim Total number of upper triangular matrices.
 * @param [in] workspace Pointer to workspace.
 */
template <typename InputT>
__aicore__ inline void run_triu_inv_rec_unroll(GM_ADDR U, GM_ADDR out_tensor,
                                               uint32_t matrix_size,
                                               uint32_t block_dim,
                                               GM_ADDR workspace) {
  GM_ADDR const X = workspace;
  GM_ADDR const I =
      workspace + block_dim * matrix_size * matrix_size * sizeof(InputT);
  GM_ADDR const U_neg =
      workspace + 2 * block_dim * matrix_size * matrix_size * sizeof(InputT);
  if ASCEND_IS_AIV {
    KernelPrepareIdentityAndMinusU<InputT> op(matrix_size, block_dim);
    op.Init(U, U_neg, X, I);
    op.Process();
  }
  tcuscan::sync::SyncGroup<tcuscan::sync::GroupSyncDirection::FULL>();
  if ASCEND_IS_AIC {
    KernelInvTriuRecUnroll<InputT> op(matrix_size);
    op.Init(U, U_neg, X, I, out_tensor);
    op.Process();
  }
}
