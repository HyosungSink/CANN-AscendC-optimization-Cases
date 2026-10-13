/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tcuscan_utils.h
 * @brief Common building blocks for kernels.
 */

#pragma once
#include <limits>
#include <type_traits>

#include "ascendc_kernel_operator.h"

using namespace AscendC;

namespace tcuscan {

/// @brief Number of bytes for a required alignment in UB.
constexpr uint16_t UB_ALIGNMENT = 32;
/// @brief Number of bytes for the UB size (A2A3).
constexpr uint32_t UB_SIZE_BYTES = 190 * 1024;
/// @brief Number of bytes per data block.
constexpr uint16_t DATA_BLOCK_SIZE = 32;
/// Global memory allocation alignment, added only for performance.
constexpr int32_t GM_ALIGNMENT = 256;

/**
 * @brief Get the size of the fractal used internally by hardware along the
 * matrix K dimension.
 *
 * @tparam DataType Data type used for matrix multiplication.
 * @return The size of K dimension of the fractal.
 */
template <typename T>
constexpr __aicore__ inline uint16_t GetFractalK() {
  return 32 / sizeof(T);
}

/**
 * @brief Get the size of the fractal used internally by hardware along the
 * matrix M/N dimensions.
 *
 * @tparam DataType Data type used for matrix multiplication.
 * @return The size of both M and N dimensions of the fractal.
 */
template <typename T>
constexpr __aicore__ inline uint16_t GetFractalMN() {
  return 16;
}

/**
 * @brief Fill input with identity martix like `np.eye(matrix_size)`.
 *
 * It is assumed that the input local tensor has size at least \f$ matrix_size^2
 * \f$.
 *
 * @param [in] mat_lt Input local tensor of length \f$ matrix_size \times
 * matrix_size\f$ where the identity matrix will be written.
 * @param [in] matrix_size Matrix size.
 */
template <typename T>
__aicore__ inline void FillIdentity(const LocalTensor<T>& mat_lt,
                                    uint32_t matrix_size) {
  AscendC::Duplicate(mat_lt, static_cast<T>(0), matrix_size * matrix_size);

  // Set one on the main diagonal
  for (uint32_t i = 0; i < matrix_size; i++) {
    mat_lt.SetValue(i * matrix_size + i, static_cast<T>(1));
  }
}

/**
 * @brief Fill vector with the (column_index)-th column of the identity matrix.
 *
 * @param [in] lt Input vector of length \f$ matrix_size \f$.
 * @param [in] matrix_size Identity matrix size.
 * @param [in] column_index Column index to identity matrix.
 */
template <typename T>
__aicore__ inline void FillStandardVector(const LocalTensor<T>& lt,
                                          uint32_t matrix_size,
                                          uint32_t column_index) {
  // Write zeros on j-th column of input matrix
  Duplicate<T>(lt, static_cast<T>(0), matrix_size);

  // Write one on the diagonal (j,j)-th element of input matrix.
  lt.SetValue(column_index, static_cast<T>(1));
}

/**
 * @brief Fill diagonal entries like `np.filldiagonal(A, value)`.
 *
 * It is assumed that the input local tensor has size at least \f$ matrix_size^2
 * \f$.
 *
 * @tparam Data type.
 *
 * @param [in] mat_lt Input local tensor of length \f$ matrix_size \times
 * matrix_size\f$
 * @param [in] matrix_size Matrix size.
 * @param [in] value Value to fill in diagonals.
 */
template <typename T>
__aicore__ inline void FillDiagonal(const LocalTensor<T>& mat_lt,
                                    uint32_t matrix_size, T value) {
  for (uint32_t i = 0; i < matrix_size; i++) {
    mat_lt.SetValue(i * matrix_size + i, value);
  }
}

/**
 * @brief Copies tiling structure from global memory to registers.
 *
 * @tparam TilingT Structure representing kernel tiling parameters.
 * @param [in] tiling Pointer to the structure allocated in registers.
 * @param [in] tiling_global Pointer to the structure in global memory.
 */
template <typename TilingT>
__aicore__ inline void GetTilingData(TilingT* const tiling,
                                     GM_ADDR tiling_global) {
  uint32_t* const tiling_32b = reinterpret_cast<uint32_t*>(tiling);
  const __gm__ uint32_t* const tiling_global_32b =
      reinterpret_cast<__gm__ uint32_t*>(tiling_global);

  for (uint32_t i = 0; i < sizeof(TilingT) / sizeof(uint32_t); i++) {
    tiling_32b[i] = tiling_global_32b[i];
  }
}

namespace exec_mode {

/**
 * @brief Kernel for cube core that might be used to enable mix mode.
 *
 * This kernel does a single small cube core operation. It can be used to enable
 * mixed mode when only vector operations are used. This is useful when hw
 * synchronization is needed.
 *
 */
class KernelNoOpCube {
 public:
  /**
   * @brief Class constructor.
   *
   */
  __aicore__ inline KernelNoOpCube() {}
  /**
   * @brief Initialize buffers.
   *
   */
  __aicore__ inline void Init() {
    pipe.InitBuffer(a2_q_, 1, tile_size_ * sizeof(half));
    pipe.InitBuffer(b2_q_, 1, tile_size_ * sizeof(half));
    pipe.InitBuffer(co1_q_, 1, tile_size_ * sizeof(float));
  }
  /**
   * @brief Issue a small matrix multiplication instruction.
   *
   */
  __aicore__ inline void Process() {
    LocalTensor<half> a2_lt = a2_q_.AllocTensor<half>();
    LocalTensor<half> b2_lt = b2_q_.AllocTensor<half>();
    LocalTensor<float> co1_lt = co1_q_.AllocTensor<float>();
    Mmad(co1_lt, a2_lt, b2_lt,
         {tile_size_, tile_size_, tile_size_, false, 0, false, false, false});
    a2_q_.FreeTensor(a2_lt);
    b2_q_.FreeTensor(b2_lt);
    co1_q_.FreeTensor(co1_lt);
  }

 private:
  TPipe pipe;

  TQue<QuePosition::A2, 1> a2_q_;
  TQue<QuePosition::B2, 1> b2_q_;
  TQue<QuePosition::CO1, 1> co1_q_;

  constexpr static uint16_t tile_size_ = 32;
};

/**
 * @brief Kernel for vector core that might be used to enable mix mode.
 *
 * This kernel does a single small vector core operation. It can be used to
 * enable mixed mode when only cube operations are used. This is useful when hw
 * synchronization is needed.
 */
class KernelNoOpVector {
 public:
  /**
   * @brief Class constructor.
   *
   */
  __aicore__ inline KernelNoOpVector() {}
  /**
   * @brief Initialize buffers.
   *
   */
  __aicore__ inline void Init() {
    pipe.InitBuffer(vec_buf_, tile_size_ * sizeof(float));
  }
  /**
   * @brief Issue a small vector instruction.
   *
   */
  __aicore__ inline void Process() {
    const LocalTensor<float> lt = vec_buf_.Get<float>();
    Adds(lt, lt, 1.f, tile_size_);
  }

 private:
  TPipe pipe;

  TBuf<QuePosition::VECCALC> vec_buf_;
  constexpr static uint16_t tile_size_ = 32;
};

/**
 * @brief Executes a small instruction on cube cores.
 *
 * Might be used to enable mix mode in a vector-only kernel.
 */
__aicore__ inline void EnableCubeCores() {
  if ASCEND_IS_AIC {
    KernelNoOpCube op;
    op.Init();
    op.Process();
  }
}

/**
 * @brief Executes a small instruction on vector cores.
 *
 * Might be used to enable mix mode in a cube-only kernel.
 */
__aicore__ inline void EnableVectorCores() {
  if ASCEND_IS_AIV {
    KernelNoOpVector op;
    op.Init();
    op.Process();
  }
}

/**
 * @brief Makes the execution fail if the function is called from AIC core.
 *
 * Asserts only on CPU target, the check is done at runtime.
 */
__aicore__ inline void AssertIsAIV() {
#ifdef ASCENDC_CPU_DEBUG
  ASCENDC_ASSERT(g_coreType == AIV, {
    KERNEL_LOG(KERNEL_ERROR, "The function can only be called on AIV cores.");
  });
#endif
}

/**
 * @brief Makes the execution fail if the function is called from AIV core.
 *
 * Asserts only on CPU target, the check is done at runtime.
 */
__aicore__ inline void AssertIsAIC() {
#ifdef ASCENDC_CPU_DEBUG
  ASCENDC_ASSERT(g_coreType == AIC, {
    KERNEL_LOG(KERNEL_ERROR, "The function can only be called on AIC cores.");
  });
#endif
}

/**
 * @brief  Makes the execution fail if the LocalTensor is not in the position.
 *
 * Asserts only on CPU target, the check is done at runtime.
 *
 * @tparam T Local tensor data type.
 * @param lt Local tensor to check.
 * @param position Tensor position to check.
 */
// clang-format off
template <typename T>
__aicore__ inline void AssertLocalTensorIsIn([[maybe_unused]] const LocalTensor<T> &lt,
                                             [[maybe_unused]] TPosition position) {
#ifdef ASCENDC_CPU_DEBUG
    const TPosition pos = TPosition(lt.GetPosition());
    ASCENDC_ASSERT(pos == position, {
        KERNEL_LOG(KERNEL_ERROR,
                   "LocalTensor is not in correct Queue position.");
    });
#endif
}
// clang-format on

/**
 * @brief  Makes the execution fail if the LocalTensor is not in L1 (A1 or B1).
 *
 * Asserts only on CPU target, the check is done at runtime.
 *
 * @tparam T Local tensor data type.
 * @param lt Local tensor to check.
 */
// clang-format off
template <typename T>
__aicore__ inline void AssertLocalTensorIsInL1(
    [[maybe_unused]] const LocalTensor<T> &lt) {
#ifdef ASCENDC_CPU_DEBUG
    const TPosition pos = TPosition(lt.GetPosition());
    ASCENDC_ASSERT(pos == TPosition::A1 || pos == TPosition::B1, {
        KERNEL_LOG(KERNEL_ERROR, "LocalTensor is not in L1 (A1 or B1).");
    });
#endif
}
// clang-format on

/**
 * @brief  Makes the execution fail if the LocalTensor is not in L0 (A2 or B2).
 *
 * Asserts only on CPU target, the check is done at runtime.
 *
 * @tparam T Local tensor data type.
 * @param lt Local tensor to check.
 */
// clang-format off
template <typename T>
__aicore__ inline void AssertLocalTensorIsInL0(
    [[maybe_unused]] const LocalTensor<T> &lt) {
#ifdef ASCENDC_CPU_DEBUG
    const TPosition pos = TPosition(lt.GetPosition());
    ASCENDC_ASSERT(pos == TPosition::A2 || pos == TPosition::B2, {
        KERNEL_LOG(KERNEL_ERROR, "LocalTensor is not in L0 (A2 or B2).");
    });
#endif
}
// clang-format on

}  // namespace exec_mode

namespace copy {

/**
 * @brief Perform a copy from global tensor (ND layout) to local tensor (NZ
 * layout).
 *
 * @tparam DataType Data type of the tensors.
 * @tparam FractalHeight Height of the fractal used internally by hardware.
 * @tparam FractalWidth Width of the fractal used internally by hardware.
 *
 * @param [in] dst Destination local tensor.
 * @param [in] src Source global tensor.
 * @param [in] fractals_h Number of fractal patterns in the height dimension of
 * the input tensor.
 * @param [in] fractals_w Number of fractal patterns in the width dimension of
 * the input tensor.
 */
template <typename DataType, uint16_t FractalHeight = 16,
          uint16_t FractalWidth = 16>
__aicore__ inline void CopyND2NZ(const LocalTensor<DataType>& dst,
                                 const GlobalTensor<DataType>& src,
                                 uint16_t fractals_h, uint16_t fractals_w) {
  exec_mode::AssertLocalTensorIsInL1(dst);
  if (fractals_w == 1) {
    DataCopy(dst, src, dst.GetSize());
    return;
  }
  Nd2NzParams params;
  params.ndNum = 1;
  params.nValue = fractals_h * FractalHeight;
  params.dValue = fractals_w * FractalWidth;
  params.srcDValue = params.dValue;
  params.dstNzC0Stride = params.nValue;
  params.dstNzNStride = 1;
  // params.dstNzMatrixStride = 1; // This should have no effect when ndNum=1;
  DataCopy(dst, src, params);
}

/**
 * @brief Copy a tensor from CO1 to CO2 queue.
 *
 * @tparam DstDataType Data type of the destination tensor.
 * @tparam SrcDataType Data type of the source tensor.
 * @tparam DstNumBuffers Depth of the destination queue.
 * @tparam SrcNumBuffers Depth of the source queue.
 *
 * @param [in] dst_q Destination queue. The position must be CO2.
 * @param [in] src_q Source queue. The position must be CO1.
 * @param [in] num_fractals Number of fractals to copy. Usually fractal is 16x16
 * elements.
 */
template <typename DstDataType, typename SrcDataType, int32_t DstNumBuffers,
          int32_t SrcNumBuffers>
__aicore__ inline void CopyCL0ToL1(TQue<QuePosition::CO2, DstNumBuffers>& dst_q,
                                   TQue<QuePosition::CO1, SrcNumBuffers>& src_q,
                                   int32_t num_fractals) {
  exec_mode::AssertIsAIC();
  const LocalTensor<DstDataType> dst_lt =
      dst_q.template AllocTensor<DstDataType>();
  LocalTensor<SrcDataType> src_lt = src_q.template DeQue<SrcDataType>();

  DataCopyParams params;
  params.blockCount = 1;
  params.blockLen = num_fractals;
  DataCopyEnhancedParams enhancedParams;
  enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;
  DataCopy(dst_lt, src_lt, params, enhancedParams);

  dst_q.EnQue(dst_lt);
  src_q.FreeTensor(src_lt);
}

/**
 * @brief Copy data from tensor allocated in L0C to tensor allocated in L1.
 *
 * @tparam DstDataType Data type of the destination tensor.
 * @tparam SrcDataType Data type of the source tensor.
 *
 * @param [in] dst_lt Destination local tensor.
 * @param [in] src_lt Source local tensor.
 * @param [in] height Height of the matrix.
 * @param [in] width Width of the matrix.
 */
template <typename DstDataType, typename SrcDataType>
__aicore__ inline void CopyL0CToL1(const LocalTensor<DstDataType>& dst_lt,
                                   const LocalTensor<SrcDataType>& src_lt,
                                   uint32_t height, uint32_t width) {
  exec_mode::AssertIsAIC();
  exec_mode::AssertLocalTensorIsIn(src_lt, QuePosition::CO1);
  exec_mode::AssertLocalTensorIsInL1(dst_lt);
  static_assert(
      std::is_same_v<SrcDataType, float> && std::is_same_v<DstDataType, half>,
      "Unsupported data types. Please add support yourself.");

  FixpipeParamsV220 params;
  params.nSize = width;
  params.mSize = height;
  params.srcStride = height;
  params.dstStride = width;
  params.ndNum = 1;
  params.quantPre = QuantMode_t::F322F16;

  Fixpipe<DstDataType, SrcDataType, CFG_NZ>(dst_lt, src_lt, params);
}

/**
 * @brief Copy a tensor from CO1 queue to B1 queue.
 *
 * @tparam DstDataType Data type of the destination tensor.
 * @tparam SrcDataType Data type of the source tensor.
 * @tparam B1NumBuffers Depth of the B1 queue.
 * @tparam CO1NumBuffers Depth of the CO1 queue.
 * @tparam FreeSrc Whether to free the source tensor or not.
 *
 * @param [in] dst_q Destination queue. The position must be B1.
 * @param [in] src_q Source queue. The position must be CO1.
 * @param [in] height Height of the matrix.
 * @param [in] width Width of the matrix.
 */
template <typename DstDataType, typename SrcDataType, int32_t B1NumBuffers,
          int32_t C01NumBuffers, bool FreeSrc = true>
__aicore__ inline void CopyC01ToB1(TQue<QuePosition::B1, B1NumBuffers>& dst_q,
                                   TQue<QuePosition::CO1, C01NumBuffers>& src_q,
                                   uint32_t height, uint32_t width) {
  exec_mode::AssertIsAIC();
  LocalTensor<SrcDataType> src_lt = src_q.template DeQue<SrcDataType>();
  const LocalTensor<DstDataType> dst_lt =
      dst_q.template AllocTensor<DstDataType>();

  CopyL0CToL1<DstDataType, SrcDataType>(dst_lt, src_lt, height, width);
  if constexpr (FreeSrc) {
    src_q.FreeTensor(src_lt);
  } else {
    src_q.EnQue(src_lt);
  }
  dst_q.EnQue(dst_lt);
}

/**
 * @brief Copy a tensor from CO1 queue to B1 queue.
 *
 * @tparam DstDataType Data type of the destination tensor.
 * @tparam SrcDataType Data type of the source tensor.
 * @tparam CO1NumBuffers Depth of the CO1 queue.
 * @tparam FreeSrc Whether to free the source tensor or not.
 *
 * @param [in] dst_buf Destination buffer. The position must be B1.
 * @param [in] src_q Source queue. The position must be CO1.
 * @param [in] height Height of the matrix.
 * @param [in] width Width of the matrix.
 */
template <typename DstDataType, typename SrcDataType, int32_t C01NumBuffers,
          bool FreeSrc = true>
__aicore__ inline void CopyC01ToB1(TBuf<TPosition::B1>& dst_buf,
                                   TQue<QuePosition::CO1, C01NumBuffers>& src_q,
                                   uint32_t height, uint32_t width) {
  exec_mode::AssertIsAIC();
  LocalTensor<SrcDataType> src_lt = src_q.template DeQue<SrcDataType>();
  const LocalTensor<DstDataType> dst_lt = dst_buf.Get<DstDataType>();

  CopyL0CToL1<DstDataType, SrcDataType>(dst_lt, src_lt, height, width);
  if constexpr (FreeSrc) {
    src_q.FreeTensor(src_lt);
  } else {
    src_q.EnQue(src_lt);
  }
}

/**
 * @brief Copy a tensor from CO1 queue to A1 queue.
 *
 * @tparam DstDataType Data type of the destination tensor.
 * @tparam SrcDataType Data type of the source tensor.
 * @tparam A1NumBuffers Depth of the A1 queue.
 * @tparam CO1NumBuffers Depth of the CO1 queue.
 * @tparam FreeSrc Whether to free the source tensor or not.
 *
 * @param [in] dst_q Destination queue. The position must be A1.
 * @param [in] src_q Source queue. The position must be CO1.
 * @param [in] height Height of the matrix.
 * @param [in] width Width of the matrix.
 */
template <typename DstDataType, typename SrcDataType, int32_t A1NumBuffers,
          int32_t C01NumBuffers, bool FreeSrc = true>
__aicore__ inline void CopyC01ToA1(TQue<QuePosition::A1, A1NumBuffers>& dst_q,
                                   TQue<QuePosition::CO1, C01NumBuffers>& src_q,
                                   uint32_t height, uint32_t width) {
  exec_mode::AssertIsAIC();
  LocalTensor<SrcDataType> src_lt = src_q.template DeQue<SrcDataType>();
  const LocalTensor<DstDataType> dst_lt =
      dst_q.template AllocTensor<DstDataType>();

  CopyL0CToL1<DstDataType, SrcDataType>(dst_lt, src_lt, height, width);
  if constexpr (FreeSrc) {
    src_q.FreeTensor(src_lt);
  } else {
    src_q.EnQue(src_lt);
  }
  dst_q.EnQue(dst_lt);
}
/**
 * @brief Copy a tensor from CO1 queue to global memory.
 *
 * @tparam DataType Data type of the source tensor.
 * @tparam QNumBuffers Depth of the queue.
 *
 * @param [in] global Destination global tensor.
 * @param [in] src_q Source queue. The position must be CO1.
 * @param [in] height Height of the matrix.
 * @param [in] width Width of the matrix.
 */
template <typename DataType, int32_t QNumBuffers>
__aicore__ inline void CopyCL0ToGlobal(
    const GlobalTensor<DataType>& global,
    TQue<QuePosition::CO1, QNumBuffers>& src_q, uint32_t height,
    uint32_t width) {
  exec_mode::AssertIsAIC();
  constexpr uint16_t fractal_size = GetFractalMN<DataType>();

  LocalTensor<DataType> lt = src_q.template DeQue<DataType>();

  FixpipeParams<DataType> params;
  params.cburstNum = height;
  params.burstLen = width * fractal_size * sizeof(DataType) / DATA_BLOCK_SIZE;
  params.dstStride = height;

  Nz2NdParams nz2nd_params;
  nz2nd_params.nz2ndEn = true;
  nz2nd_params.originalNSize = height;
  params.nz2ndParams = nz2nd_params;

  Fixpipe(global, lt, params);

  src_q.FreeTensor(lt);
}

/**
 * @brief Copy a tensor from global memory to the local tensor in L1 memory.
 *
 * The queue's position must be either A1 or B1. The data layout is transformed
 * from ND to NZ.
 *
 * @tparam FractalHeight Height of the fractal used internally by hardware.
 * @tparam FractalWidth Width of the fractal used internally by hardware.
 * @tparam DataType Data type of the tensor.
 *
 * @param [in] local Destination local tensor. Must be alocated from either A1
 * or B1 queue.
 * @param [in] global Source global tensor.
 * @param [in] fractals_h Number of fractal patterns in the height dimension of
 * the input matrix.
 * @param [in] fractals_w Number of fractal patterns in the width dimension of
 * the input matrix.
 */
template <uint16_t FractalHeight, uint16_t FractalWidth, typename DataType>
__aicore__ inline void CopyGmToL1(const LocalTensor<DataType>& local,
                                  const GlobalTensor<DataType>& global,
                                  uint16_t fractals_h, uint16_t fractals_w) {
  exec_mode::AssertIsAIC();
  exec_mode::AssertLocalTensorIsInL1(local);
  tcuscan::copy::CopyND2NZ<DataType, FractalHeight, FractalWidth>(
      local, global, fractals_h, fractals_w);
}

/**
 * @brief Copy a tensor from global memory to the queue in A L1 memory.
 *
 * The data layout is transformed from ND to NZ.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam QNumBuffers Depth of the queue.
 *
 * @param [in] q Destination queue. The position must be A1.
 * @param [in] global Source global tensor.
 * @param [in] fractals_h Number of fractal patterns in the height dimension of
 * the input matrix.
 * @param [in] fractals_w Number of fractal patterns in the width dimension of
 * the input matrix.
 */
template <typename DataType, int32_t QNumBuffers>
__aicore__ inline void CopyGmToL1A(TQue<QuePosition::A1, QNumBuffers>& q,
                                   const GlobalTensor<DataType>& global,
                                   uint16_t fractals_h, uint16_t fractals_w) {
  exec_mode::AssertIsAIC();
  const LocalTensor<DataType> lt = q.template AllocTensor<DataType>();
  tcuscan::copy::CopyGmToL1<GetFractalMN<DataType>(), GetFractalK<DataType>()>(
      lt, global, fractals_h, fractals_w);
  q.EnQue(lt);
}

/**
 * @brief Copy a tensor from global memory to the queue in B L1 memory.
 *
 * The data layout is transformed from ND to NZ.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam QNumBuffers Depth of the queue.
 *
 * @param [in] q Destination queue. The position must be B1.
 * @param [in] global Source global tensor.
 * @param [in] fractals_h Number of fractal patterns in the height dimension of
 * the input matrix.
 * @param [in] fractals_w Number of fractal patterns in the width dimension of
 * the input matrix.
 */
template <typename DataType, int32_t QNumBuffers>
__aicore__ inline void CopyGmToL1B(TQue<QuePosition::B1, QNumBuffers>& q,
                                   const GlobalTensor<DataType>& global,
                                   uint16_t fractals_h, uint16_t fractals_w) {
  exec_mode::AssertIsAIC();
  const LocalTensor<DataType> lt = q.template AllocTensor<DataType>();
  tcuscan::copy::CopyGmToL1<GetFractalK<DataType>(), GetFractalMN<DataType>()>(
      lt, global, fractals_h, fractals_w);
  q.EnQue(lt);
}

/**
 * @brief Copy a tensor from global memory to the queue in B L0 memory.
 *
 * The data layout is transformed from column-major to NZ.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam QNumBuffersB2 Depth of the B2 queue.
 * @tparam QNumBuffersB1 Depth of the B1 queue.
 *
 * @param [in] b2_q Destination queue. The position must be B2.
 * @param [in] b1_q Intermediate queue. The position must be B1.
 * @param [in] global Source global tensor.
 * @param [in] fractals_h Number of fractal patterns in the height dimension of
 * the input matrix.
 * @param [in] fractals_w Number of fractal patterns in the width dimension of
 * the input matrix.
 */
template <typename DataType, int32_t QNumBuffersB2, int32_t QNumBuffersB1>
__aicore__ inline void CopyTransposedGmToL0B(
    TQue<QuePosition::B2, QNumBuffersB2>& b2_q,
    TQue<QuePosition::B1, QNumBuffersB1>& b1_q,
    const GlobalTensor<DataType>& global, uint16_t fractals_h,
    uint16_t fractals_w) {
  exec_mode::AssertIsAIC();
  // Copy with ND -> NZ transform.
  // But data in GM is transposed (column-major layout) so the resulting
  // layout is ZN (instead of NZ). And ZN is a layout `Mmad` expects for the B
  // operand.
  tcuscan::copy::CopyGmToL1B(b1_q, global, fractals_h, fractals_w);

  // Plain copy from L1 to L0B, because the layout is already correct.
  LocalTensor<DataType> src = b1_q.template DeQue<DataType>();
  const LocalTensor<DataType> dst = b2_q.template AllocTensor<DataType>();

  LoadData2dParams params;
  params.repeatTimes = fractals_w * fractals_h;
  params.srcStride = 1;
  params.ifTranspose = false;

  LoadData(dst, src, params);

  b1_q.FreeTensor(src);
  b2_q.EnQue(dst);
}

/**
 * @brief Copy a tensor from global memory to the queue in A L1 memory.
 *
 * The data layout is not transformed so the source data layout should already
 * be NZ.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam QNumBuffers Depth of the queue.
 *
 * @param [in] q Destination queue. The position must be A1.
 * @param [in] global Source global tensor.
 */
template <typename DataType, int32_t QNumBuffers>
__aicore__ inline void CopyPlainGmToL1A(TQue<QuePosition::A1, QNumBuffers>& q,
                                        const GlobalTensor<DataType>& global) {
  exec_mode::AssertIsAIC();
  const LocalTensor<DataType> lt = q.template AllocTensor<DataType>();
  DataCopy(lt, global, lt.GetSize());
  q.EnQue(lt);
}

/**
 * @brief Copy a tensor from global memory to the queue in B L1 memory.
 *
 * The data layout is not transformed so the source data layout should already
 * be NZ.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam QNumBuffers Depth of the queue.
 *
 * @param [in] q Destination queue. The position must be B1.
 * @param [in] global Source global tensor.
 */
template <typename DataType, int32_t QNumBuffers>
__aicore__ inline void CopyPlainGmToL1B(TQue<QuePosition::B1, QNumBuffers>& q,
                                        const GlobalTensor<DataType>& global) {
  exec_mode::AssertIsAIC();
  const LocalTensor<DataType> lt = q.template AllocTensor<DataType>();
  DataCopy(lt, global, lt.GetSize());
  q.EnQue(lt);
}

/**
 * @brief Copy data from tensor allocated in L1 to tensor allocated in L0.
 *
 * @tparam FractalHeight Height of the fractal used internally by hardware.
 * @tparam FractalWidth Width of the fractal used internally by hardware.
 * @tparam DataType Data type of the tensor.
 *
 * @param [in] dst Destination local tensor.
 * @param [in] src Source local tensor.
 * @param [in] fractals_h Number of fractal patterns in the height dimension of
 * the input tensor.
 * @param [in] fractals_w Number of fractal patterns in the width dimension of
 * the input tensor.
 * @param [in] transpose Indicates whether or not to transpose the matrix. If
 * set, the function performs an NZ to ZN transformation. Otherwise, it performs
 * NZ to ZZ one.
 */
template <uint16_t FractalHeight, uint16_t FractalWidth, typename DataType>
__aicore__ inline void CopyL1ToL0(const LocalTensor<DataType>& dst,
                                  const LocalTensor<DataType>& src,
                                  uint16_t fractals_h, uint16_t fractals_w,
                                  bool transpose) {
  exec_mode::AssertIsAIC();
  exec_mode::AssertLocalTensorIsInL1(src);
  exec_mode::AssertLocalTensorIsInL0(dst);

  int src_offset = 0;
  int dst_offset = 0;

  for (uint16_t i = 0; i < fractals_h; ++i) {
    LoadData2dParams params;
    params.repeatTimes = fractals_w;
    params.srcStride = fractals_h;
    params.ifTranspose = transpose;

    LoadData(dst[dst_offset], src[src_offset], params);

    src_offset += FractalWidth * FractalHeight;
    dst_offset += fractals_w * FractalWidth * FractalHeight;
  }
}

/**
 * @brief Copy a tensor from a given A L1 to A L0 queue.
 *
 * The function performs an NZ to ZZ layout transformation.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam FreeSrc Indicates whether the source tensor should be freed or
 * enqueued back.
 * @tparam L0NumBuffers Depth of the destination queue.
 * @tparam L1NumBuffers Depth of the source queue.
 *
 * @param [in] l0_q Destination queue. The position must be A2.
 * @param [in] l1_q Source queue. The position must be A1.
 * @param [in] fractals_h Number of fractal patterns in the height dimension of
 * the input matrix.
 * @param [in] fractals_w Number of fractal patterns in the width dimension of
 * the input matrix.
 */
template <typename DataType, bool FreeSrc, int32_t L0NumBuffers,
          int32_t L1NumBuffers>
__aicore__ inline void CopyL1ToL0A(TQue<QuePosition::A2, L0NumBuffers>& l0_q,
                                   TQue<QuePosition::A1, L1NumBuffers>& l1_q,
                                   uint16_t fractals_h, uint16_t fractals_w) {
  exec_mode::AssertIsAIC();
  const LocalTensor<DataType> l0_lt = l0_q.template AllocTensor<DataType>();
  LocalTensor<DataType> l1_lt = l1_q.template DeQue<DataType>();

  CopyL1ToL0<GetFractalMN<DataType>(), GetFractalK<DataType>()>(
      l0_lt, l1_lt, fractals_h, fractals_w, false /* transpose */);

  if constexpr (FreeSrc) {
    l1_q.FreeTensor(l1_lt);
  } else {
    l1_q.EnQue(l1_lt);
  }
  l0_q.EnQue(l0_lt);
}

/**
 * @brief Check if matrix B should be tranposed when moved from B L1 to B L0.
 *
 * Matrix tranpose is not supported for 8-bit data types.
 *
 * @tparam DataType Data type used for matrix multiplication.
 * @return Indicates whether matrix B should be transposed.
 */
template <typename T>
constexpr __aicore__ inline bool ShouldTransposeB() {
  if constexpr (sizeof(T) == 2) {
    return true;
  }
  return false;
}

/**
 * @brief Copy a tensor from a given B L1 to B L0 queue.
 *
 * The function performs an NZ to ZN layout transformation.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam FreeSrc Indicates whether the source tensor should be freed or
 * enqueued back.
 * @tparam L0NumBuffers Depth of the destination queue.
 * @tparam L1NumBuffers Depth of the source queue.
 *
 * @param [in] l0_q Destination queue. The position must be B2.
 * @param [in] l1_q Source queue. The position must be B1.
 * @param [in] fractals_h Number of fractal patterns in the height dimension of
 * the input matrix.
 * @param [in] fractals_w Number of fractal patterns in the width dimension of
 * the input matrix.
 */
template <typename DataType, bool FreeSrc, int32_t L0NumBuffers,
          int32_t L1NumBuffers>
__aicore__ inline void CopyL1ToL0B(TQue<QuePosition::B2, L0NumBuffers>& l0_q,
                                   TQue<QuePosition::B1, L1NumBuffers>& l1_q,
                                   uint16_t fractals_h, uint16_t fractals_w) {
  exec_mode::AssertIsAIC();
  LocalTensor<DataType> l1_lt = l1_q.template DeQue<DataType>();
  const LocalTensor<DataType> l0_lt = l0_q.template AllocTensor<DataType>();
  constexpr bool transpose = ShouldTransposeB<DataType>();
  CopyL1ToL0<GetFractalK<DataType>(), GetFractalMN<DataType>()>(
      l0_lt, l1_lt, fractals_h, fractals_w, transpose);

  if constexpr (FreeSrc) {
    l1_q.FreeTensor(l1_lt);
  } else {
    l1_q.EnQue(l1_lt);
  }
  l0_q.EnQue(l0_lt);
}

/**
 * @brief Copy an tensor from a local tensor in L1 to B L0 queue.
 *
 * The function performs an NZ to ZN layout transformation.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam L0NumBuffers Depth of the destination queue.
 * @tparam L1NumBuffers Depth of the source queue.
 *
 * @param [in] l0_q Destination queue. The position must be B2.
 * @param [in] l1_lt Source local tensor. The position must be B1.
 * @param [in] fractals_h Number of fractal patterns in the height dimension of
 * the input matrix.
 * @param [in] fractals_w Number of fractal patterns in the width dimension of
 * the input matrix.
 */
template <typename DataType, int32_t L0NumBuffers>
__aicore__ inline void CopyL1ToL0B(TQue<QuePosition::B2, L0NumBuffers>& l0_q,
                                   LocalTensor<DataType>& l1_lt,
                                   uint16_t fractals_h, uint16_t fractals_w) {
  exec_mode::AssertIsAIC();
  exec_mode::AssertLocalTensorIsInL1(l1_lt);
  const LocalTensor<DataType> l0_lt = l0_q.template AllocTensor<DataType>();
  constexpr bool transpose = ShouldTransposeB<DataType>();
  CopyL1ToL0<GetFractalK<DataType>(), GetFractalMN<DataType>()>(
      l0_lt, l1_lt, fractals_h, fractals_w, transpose);

  l0_q.EnQue(l0_lt);
}

/**
 * @brief Copy a tensor from global memory to the VECIN queue.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam QNumBuffers Depth of the queue.
 *
 * @param [in] q Destination queue. The position must be VECIN.
 * @param [in] global Source global tensor.
 * @param [in] num_elems Number of elements to load. By default, the function
 * loads all the elements for the local tensor obtained from the queue. If the
 * value is provided, the function will load \f$num_elems\f$ elements.
 * @param [in] pad_value Value use for padding when \f$num_elems\f$ is not
 * aligned to UB. It defaults to 0.
 */
template <typename DataType, int32_t QNumBuffers>
__aicore__ inline void CopyGmToVec(TQue<QuePosition::VECIN, QNumBuffers>& q,
                                   const GlobalTensor<DataType>& global,
                                   uint32_t num_elems = 0,
                                   DataType pad_value = 0) {
  exec_mode::AssertIsAIV();
  const LocalTensor<DataType> lt = q.template AllocTensor<DataType>();
  if (!num_elems) num_elems = lt.GetSize();
  if (num_elems % (UB_ALIGNMENT / sizeof(DataType)) == 0) {
    DataCopy(lt, global, num_elems);
  } else {
    const uint32_t align_len =
        static_cast<uint32_t>(UB_ALIGNMENT / sizeof(DataType));
    const uint8_t pad_len =
        static_cast<uint8_t>(align_len - num_elems % align_len);

    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = num_elems * sizeof(DataType);
    params.srcStride = 0;
    params.dstStride = 0;

    DataCopyPadExtParams<DataType> pad_params;
    pad_params.isPad = true;
    pad_params.leftPadding = 0;
    pad_params.rightPadding = pad_len;
    pad_params.paddingValue = static_cast<DataType>(pad_value);

    DataCopyPad(lt, global, params, pad_params);
  }
  q.EnQue(lt);
}

/**
 * @brief Copy a tensor from a given queue to global memory.
 *
 * The queue's position must be VECOUT.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam VecNumBuffers Depth of the queue.
 *
 * @param [in] global Destination global tensor.
 * @param [in] q Source queue. The position must be VECOUT.
 * @param [in] num_elems Number of elements to store. By default, the function
 * stores all the elements for the local tensor obtained from the queue. If the
 * value is provided, the function will store \f$num_elems\f$ elements.
 */
template <typename DataType, int32_t VecNumBuffers>
__aicore__ inline void CopyVecToGm(const GlobalTensor<DataType>& global,
                                   TQue<QuePosition::VECOUT, VecNumBuffers>& q,
                                   uint32_t num_elems = 0) {
  exec_mode::AssertIsAIV();
  LocalTensor<DataType> lt = q.template DeQue<DataType>();
  if (!num_elems) {
    num_elems = lt.GetSize();
  }

  if ((num_elems * sizeof(DataType)) % UB_ALIGNMENT == 0) {
    DataCopy(global, lt, num_elems);
  } else {
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = num_elems * sizeof(DataType);
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPad(global, lt, params);
  }
  q.FreeTensor(lt);
}

/**
 * @brief Copy a local tensor to global memory.
 *
 * The function first copies the data from the local tensor to
 * the provided VECOUT queue, and later from VECOUT queue to the global
 * memory.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam VecoutNumBuffers Depth of the `vecout_q` queue.
 *
 * @param [in] global Destination global tensor.
 * @param [in] vecout_q Queue used to hold an intermediate tensor. The position
 * must be VECOUT.
 * @param [in] lt Source tensor.
 * @param [in] num_elems Number of elements to store. By default, the function
 * stores all the elements of the local tensor. If the value is provided, the
 * function will store \f$num_elems\f$ elements.
 */
template <typename DataType, int32_t VecoutNumBuffers>
__aicore__ inline void CopyVecToGm(
    const GlobalTensor<DataType>& global,
    TQue<QuePosition::VECOUT, VecoutNumBuffers>& vecout_q,
    const LocalTensor<DataType>& lt, uint32_t num_elems = 0) {
  exec_mode::AssertIsAIV();
  // Any UB position -> VECOUT
  const uint32_t size =
      num_elems ? AlignUp(num_elems, UB_ALIGNMENT) : lt.GetSize();
  const LocalTensor<DataType> vecout_lt =
      vecout_q.template AllocTensor<DataType>();
  DataCopy(vecout_lt, lt, size);
  vecout_q.EnQue(vecout_lt);

  // VECOUT -> global
  tcuscan::copy::CopyVecToGm<DataType>(global, vecout_q, num_elems);
}

/**
 * @brief Copy a tensor from a given queue to global memory.
 *
 * The queue's position must be VECIN.
 * The function first copies the data from the input VECIN queue to
 * the provided VECOUT queue, and later from VECOUT queue to the global
 * memory.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam VecoutNumBuffers Depth of the `vecout_q` queue.
 * @tparam SrcNumBuffers Depth of the input queue.
 *
 * @param [in] global Destination global tensor.
 * @param [in] vecout_q Queue used to hold an intermediate tensor. The
 * position must be VECOUT.
 * @param [in] q Source queue. The position must be VECIN.
 * @param [in] num_elems Number of elements to store. By default, the function
 * stores all the elements for the local tensor obtained from the queue. If the
 * value is provided, the function will store \f$num_elems\f$ elements.
 */
template <typename DataType, int32_t VecoutNumBuffers, int32_t SrcNumBuffers>
__aicore__ inline void CopyVecToGm(
    const GlobalTensor<DataType>& global,
    TQue<QuePosition::VECOUT, VecoutNumBuffers>& vecout_q,
    TQue<QuePosition::VECIN, SrcNumBuffers>& q, uint32_t num_elems = 0) {
  exec_mode::AssertIsAIV();
  LocalTensor<DataType> lt = q.template DeQue<DataType>();
  tcuscan::copy::CopyVecToGm<DataType>(global, vecout_q, lt, num_elems);
  q.FreeTensor(lt);
}

/**
 * @brief Copy a scalar value to global memory.
 *
 * The function first stores a scalar in the local tensor allocated from the
 * VECOUT queue and then performs a data copy of that tensor to global memory.
 * Optionally, the scalar might be accumulated using atomic add operation.
 *
 * Limitation: The function writes more than a single element. Usually the
 * required size of the tensor from `vecout_q` must be at least 32 bytes. The
 * same number of bytes is written to global memory.
 *
 * @tparam DataType Data type of the scalar.
 * @tparam SrcNumBuffers Depth of the VECOUT queue.
 *
 * @param [in] global Destination global tensor.
 * @param [in] vecout_q Queue used to hold an intermediate tensor. The position
 * must be VECOUT.
 * @param [in] scalar Scalar value to be copied.
 */
template <typename DataType, int32_t QNumBuffers>
__aicore__ inline void CopyScalarToGm(
    const GlobalTensor<DataType>& global,
    TQue<QuePosition::VECOUT, QNumBuffers>& vecout_q, DataType scalar) {
  exec_mode::AssertIsAIV();
  {
    const LocalTensor<DataType> lt = vecout_q.template AllocTensor<DataType>();
    lt.SetValue(0, scalar);
    vecout_q.EnQue(lt);
  }
  {
    LocalTensor<DataType> lt = vecout_q.template DeQue<DataType>();
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = 1 * sizeof(DataType);
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPad(global, lt, params);
    vecout_q.FreeTensor(lt);
  }
}

}  // namespace copy

namespace queue {

/**
 * @brief Free a tensor from a given queue.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam Q Type of the queue.
 *
 * @param [in] q Queue from which to free a tensor.
 */
template <typename DataType, typename Q>
__aicore__ inline void FreeFromQ(Q& q) {
  LocalTensor<DataType> lt = q.template DeQue<DataType>();
  q.FreeTensor(lt);
}

namespace debug {

/**
 * @brief Free all the tensors from a given queue.
 *
 * @tparam DataType Data type of the tensor.
 * @tparam Q Type of the queue.
 *
 * @param [in] q Queue from which to free the tensors.
 */
template <typename DataType, typename Q>
__aicore__ inline void EmptyQ(Q& q) {
  while (q.HasTensorInQue()) {
    FreeFromQ<DataType>(q);
  }
}

}  // namespace debug

}  // namespace queue

namespace sync {

/**
 * @brief Used to specifies the direction of synchronization when synchronizing
 * cube and vectors within a single group.
 *
 * A single group consists of one cube core and two vector cores.
 *
 * Can be used to specify either the symmetric or asymetric synchronization.
 */
enum class GroupSyncDirection {
  /// Asymetric synchronization - cube continues execution only after vectors
  /// reach the synchronization point. Can be used when cube consumes the data
  /// produced by vectors from the same group.
  CUBE_WAIT_FOR_VEC,
  /// Asymetric synchronization - vectors continue execution only after cube
  /// reaches the synchronization point. Can be used when vectors consume the
  /// data produced by cube from the same group.
  VEC_WAIT_FOR_CUBE,
  /// Symmetric synchronization - execution continues after cubes and vectors
  /// synchronize at the same time
  FULL
};

/**
 * @brief Returns a synchronization config.
 *
 * @param [in] mode Synchronization mode.
 * @param [in] flag_id Flag to use for synchronization.
 * @return Synchronization config.
 */
__aicore__ inline int GetSyncConf(int mode, int flag_id) {
  return 1 | (mode << 4) | (flag_id << 8);
}

/**
 * @brief Synchronize cube and vector cores within a single group.
 *
 * @tparam Dir Direction of the synchronization.
 */
template <GroupSyncDirection Dir = GroupSyncDirection::FULL>
__aicore__ inline void SyncGroup() {
  const int mode = 2;

  if constexpr (Dir == GroupSyncDirection::CUBE_WAIT_FOR_VEC) {
    const int AIV_SET_FLAG_ID = 11;
    if ASCEND_IS_AIV {
      ffts_cross_core_sync(PIPE_MTE3, GetSyncConf(mode, AIV_SET_FLAG_ID));
    }
    if ASCEND_IS_AIC {
      wait_flag_dev(AIV_SET_FLAG_ID);
    }
    return;
  }
  if constexpr (Dir == GroupSyncDirection::VEC_WAIT_FOR_CUBE) {
    const int AIC_SET_FLAG_ID = 12;
    if ASCEND_IS_AIC {
      ffts_cross_core_sync(PIPE_FIX, GetSyncConf(mode, AIC_SET_FLAG_ID));
    }
    if ASCEND_IS_AIV {
      wait_flag_dev(AIC_SET_FLAG_ID);
    }
    return;
  }
  if constexpr (Dir == GroupSyncDirection::FULL) {
    const int AIV_SET_FLAG_ID = 11;
    const int AIC_SET_FLAG_ID = 12;
    if ASCEND_IS_AIV {
      ffts_cross_core_sync(PIPE_MTE3, GetSyncConf(mode, AIV_SET_FLAG_ID));
      wait_flag_dev(AIC_SET_FLAG_ID);
    }
    if ASCEND_IS_AIC {
      ffts_cross_core_sync(PIPE_FIX, GetSyncConf(mode, AIC_SET_FLAG_ID));
      wait_flag_dev(AIV_SET_FLAG_ID);
    }
    return;
  }
}

/**
 * @brief Makes scalar unit wait for vector unit.
 *
 * Scalar unit continues execution only after all previous vector operations
 * are finished.
 */
__aicore__ inline void ScalarWaitForVec() {
  const TEventID event_id = GetTPipePtr()->FetchEventID(HardEvent::V_S);
  SetFlag<HardEvent::V_S>(event_id);
  WaitFlag<HardEvent::V_S>(event_id);
}

}  // namespace sync

namespace cast {

/**
 * @brief Cast a tensor within a single queue.
 *
 * The function takes a tensor from the queue, casts it to a different data type
 * and puts it to the same queue. The size in bytes of the source and
 * destination data types must be the same.
 *
 * @tparam DstDataType Data type of the destination tensor.
 * @tparam SrcDataType Data type of the source tensor.
 * @tparam QPosition Position of the source queue. Must be either VECIN or
 * VECCALC.
 * @tparam QNumBuffers Depth of the source queue.
 *
 * @param [in] q Source queue. The position must be either VECIN or VECCALC.
 */
template <
    typename DstDataType, typename SrcDataType, QuePosition QPosition,
    int32_t QNumBuffers,
    typename std::enable_if<(QPosition == QuePosition::VECIN ||
                             QPosition == QuePosition::VECCALC) &&
                                (sizeof(DstDataType) == sizeof(SrcDataType)),
                            int>::type = 0>
__aicore__ inline void CastInPlace(TQue<QPosition, QNumBuffers>& q) {
  exec_mode::AssertIsAIV();
  const LocalTensor<SrcDataType> src = q.template DeQue<SrcDataType>();
  const LocalTensor<DstDataType> dst =
      src.template ReinterpretCast<DstDataType>();
  Cast(dst, src, RoundMode::CAST_NONE, src.GetSize());
  q.EnQue(dst);
}

}  // namespace cast

namespace data_cache {

/**
 * @brief Invalidates a single cacheline in data cache corresponding to the
 * global memory address.
 *
 * @tparam T Data type of the buffer in global memory.
 *
 * @param global [in] Global tensor which first values are invalidated in data
 * cache.
 */
template <typename T>
__aicore__ inline void InvalidateLine(const GlobalTensor<T>& global) {
  DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE,
                           DcciDst::CACHELINE_OUT>(global);
}
}  // namespace data_cache

namespace scalar {

/**
 * @brief Rounds an integral value up to the nearest multiple of a given
 * alignment.
 *
 * @tparam T Data type of the integral length.
 * @param [in] length Input length.
 * @param [in] alignment Alignment to use.
 * @return Aligned length.
 */
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
__aicore__ inline T AlignUp(T length, uint32_t alignment) {
  const T tail = length % alignment;
  if (!tail) {
    return length;
  }
  const T padding = alignment - tail;
  return length + padding;
}

/**
 * @brief Performs a division on two integral numbers and rounds the result up
 * to the nearest integer.
 *
 * @tparam T1 Data type of dividend.
 * @tparam T2 Data type of divisor.
 * @param [in] value Dividend.
 * @param [in] divisor Divisor.
 * @return Result of division.
 */
template <typename T1, typename T2,
          typename std::enable_if<std::is_integral<T1>::value &&
                                      std::is_integral<T2>::value,
                                  int>::type = 0>
__aicore__ inline T1 CeilDiv(T1 value, T2 divisor) {
  return (value + divisor - 1) / divisor;
}

/**
 * @brief Performs a division on two integral numbers and rounds the result down
 * to the nearest integer.
 *
 * @tparam T1 Data type of dividend.
 * @tparam T2 Data type of divisor.
 * @param [in] value Dividend.
 * @param [in] divisor Divisor.
 * @return Result of division.
 */
template <typename T1, typename T2,
          typename std::enable_if<std::is_integral<T1>::value &&
                                      std::is_integral<T2>::value,
                                  int>::type = 0>
__aicore__ inline T1 FloorDiv(T1 value, T2 divisor) {
  return value / divisor;
}

/**
 * @brief Returns the next tile length, given the global memory offset. The
 * returned tile length equals typically to `tile_len`, expect the last
 * iteration where the tile length is smaller than `tile_len`.
 *
 * @param tile_len Tile length
 * @param global_offset Global memory offset
 * @param length Total vector length
 * @return Length of "next" tile, given the current global memory offset.
 */
__aicore__ inline uint32_t NextTileLen(uint32_t tile_len,
                                       uint32_t global_offset,
                                       uint32_t length) {
  if (length <= global_offset) {
    return 0;
  }
  const bool full_tile = global_offset + tile_len <= length;
  const uint32_t num_elems_to_process =
      full_tile ? tile_len : length - global_offset;

  return num_elems_to_process;
}

/**
 * @brief Defines how the workload should be distributed among cores.
 *
 * The function returns the number of tiles to be processed by each block so
 * that the depth of execution is minimized. If the workload is not balanced it
 * will greedily assign as many tiles as possible starting from the first block,
 * but keeping the maximum depth optimal. If `vec_len` is not divisible by
 * `tile_size` the last tile will be smaller.
 *
 * @param [in] vec_len Size of the input vector.
 * @param [in] tile_size Tile size.
 * @param [in] block_n Number of blocks.
 *
 * @return Number of tiles assigned to the block calling the function.
 */
__aicore__ inline uint32_t GetWorkDistribution(uint32_t vec_len,
                                               uint32_t tile_size,
                                               uint32_t block_n) {
  const uint32_t num_tiles = scalar::CeilDiv(vec_len, tile_size);
  const uint32_t max_num_tiles_per_block = scalar::CeilDiv(num_tiles, block_n);
  uint32_t num_tiles_to_process = max_num_tiles_per_block;
  const int tiles_left =
      (int)num_tiles - (int)(GetBlockIdx() * max_num_tiles_per_block);

  if (tiles_left < 0) {
    num_tiles_to_process = 0;
  } else if (tiles_left < static_cast<int>(max_num_tiles_per_block)) {
    num_tiles_to_process = tiles_left;
  }
  return num_tiles_to_process;
}

/**
 * @brief Defines how the elements of a batch should be distributed among cores.
 *
 * The function returns the number of elements to be processed by each block so
 * that the depth of execution is minimized. If the workload is not balanced it
 * will assign one more element starting from the first block
 *
 * @param [in] batch_size Number of elements.
 * @param [in] block_n Number of blocks.
 * @param [in] core_idx Core index.
 *
 * @return Number of elements assigned to the block calling the function.
 */
__aicore__ inline uint32_t GetBatchDistribution(uint32_t batch_size,
                                                uint32_t block_n,
                                                uint32_t core_idx) {
  if (core_idx < batch_size % block_n)
    return scalar::CeilDiv(batch_size, block_n);
  else
    return scalar::FloorDiv(batch_size, block_n);
}

/**
 * @brief Returns how many batches are assigned to previous blocks using
 * `GetBatchDistribution`.
 *
 * The function returns the number of elements to be processed by each block so
 * that the depth of execution is minimized. If the workload is not balanced it
 * will assign one more element starting from the first block
 *
 * @param [in] batch_size Number of elements.
 * @param [in] block_n Number of blocks.
 *
 * @return Number of elements assigned to the block calling the function.
 */
__aicore__ inline uint32_t GetBatchOffset(uint32_t batch_size,
                                          uint32_t block_n) {
  uint32_t core_idx = GetBlockIdx();
  uint32_t offset = 0;

  for (uint32_t i = 0; i < core_idx; i++) {
    offset += GetBatchDistribution(batch_size, block_n, i);
  }

  return offset;
}

/**
 * @brief Reads a value from global memory.
 *
 * @tparam T Data type.
 * @param [in] addr Address in GM.
 * @param [in] offset Offset.
 * @param [in] vec_len Size of the global buffer.
 * @return Value from global memory.
 */
template <typename T>
__aicore__ inline T GetGMValue(GM_ADDR addr, uint32_t offset,
                               uint32_t vec_len) {
  ASCENDC_ASSERT(offset < vec_len, {
    KERNEL_LOG(KERNEL_ERROR,
               "GetGMValue is trying to access data out of bounds.");
  });

  GlobalTensor<T> gt;
  gt.SetGlobalBuffer((__gm__ T*)addr, vec_len);

  data_cache::InvalidateLine(gt);

  return gt.GetValue(offset);
}

/**
 * @brief Binary search (lower_bound) of @p value into a sorted GM array.
 *
 * Reproduces `torch.searchsorted(sorted, value, side='left')`: returns the
 * first index `j` such that `sorted[j] >= value` (equivalently, the count of
 * elements `< value`). The haystack is read one element at a time from global
 * memory via `GetGMValue`, so no Unified Buffer staging is required.
 *
 * @tparam T Data type of the sorted array and search value.
 *
 * @param [in] sorted Pointer to the sorted (ascending) array in global memory.
 * @param [in] len Number of elements in @p sorted.
 * @param [in] value Value to search for.
 * @return Insertion index in `[0, len]`.
 */
template <typename T>
__aicore__ inline uint32_t LowerBoundGM(GM_ADDR sorted, uint32_t len, T value) {
  uint32_t lo = 0;
  uint32_t hi = len;
  while (lo < hi) {
    const uint32_t mid = lo + (hi - lo) / 2;
    const T pivot = GetGMValue<T>(sorted, mid, len);
    if (pivot < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

/**
 * @brief Sets a scalar value to global memory.
 *
 * @tparam T Data type.
 *
 * @param [in] addr Address in GM.
 * @param [in] offset Offset.
 * @param [in] value Value to set.
 * @param [in] vec_len Size of the global buffer.
 */
template <typename T>
__aicore__ inline void SetGMValue(GM_ADDR addr, uint32_t offset, T value,
                                  uint32_t vec_len) {
  ASCENDC_ASSERT(offset < vec_len, {
    KERNEL_LOG(KERNEL_ERROR,
               "SetGMValue is trying to access data out of bounds.");
  });

  GlobalTensor<T> gt;
  gt.SetGlobalBuffer((__gm__ T*)addr, vec_len);
  gt.SetValue(offset, value);
  data_cache::InvalidateLine(gt);
}

/**
 * @brief Swaps two variables.
 *
 * @tparam DataT Data type.
 * @param [in] v1 First variables.
 * @param [in] v2 Second variables.
 *
 */
template <typename DataT>
__aicore__ inline void Swap(DataT& v1, DataT& v2) {
  DataT tmp = v1;
  v1 = v2;
  v2 = tmp;
}

/**
 * @brief Calculates minimum of two numbers.
 *
 * @tparam T Data type of the values.
 * @param [in] v1 First value.
 * @param [in] v2 Second value.
 * @return Minimum value.
 */
template <typename T>
__aicore__ inline T Min(T v1, T v2) {
  return v1 <= v2 ? v1 : v2;
}

/**
 * @brief Specialization since AI-Core does not support half comparisons.
 *
 * Avoids the compilation message: `error: half precision operation is not
 * allowed in aicore function`
 *
 * @param [in] v1 First value.
 * @param [in] v2 Second value.
 * @return Minimum value.
 */
template <>
__aicore__ inline half Min(half v1, half v2) {
  return static_cast<float>(v1) <= static_cast<float>(v2) ? v1 : v2;
}

/**
 * @brief Maximum of two values.
 *
 * @tparam T Data type of the values.
 * @param [in] v1 First value.
 * @param [in] v2 Second value.
 * @return Smaller value.
 */
template <typename T>
__aicore__ inline T Max(T v1, T v2) {
  return v1 >= v2 ? v1 : v2;
}

/**
 * @brief Specialization since AI-Core does not support half comparisons.
 *
 * Avoids the compilation message: `error: half precision operation is not
 * allowed in aicore function`
 *
 * @param [in] v1 First value.
 * @param [in] v2 Second value.
 * @return Smaller value.
 */
template <>
__aicore__ inline half Max(half v1, half v2) {
  return static_cast<float>(v1) >= static_cast<float>(v2) ? v1 : v2;
}

/**
 * @brief Rounds an integral value down to the nearest multiple of a given
 * alignment.
 *
 * @tparam T Integral data type of the length.
 * @param [in] length Input length.
 * @param [in] alignment Alignment to use.
 * @return Aligned value.
 */
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
__aicore__ inline T AlignDown(T length, uint32_t alignment) {
  const T tail = length % alignment;
  if (tail > 0) {
    return length - tail;
  }
  return length;
}

}  // namespace scalar

namespace reduce {

/**
 * @brief It is true if `DataType` is supported for the AscendC ReduceSum
 * instruction.
 *
 * @tparam DataType Data type to check.
 */
template <typename DataType>
constexpr bool IsAscendReduceSumSupported =
    std::is_same<DataType, half>::value || std::is_same<DataType, float>::value;

/**
 * @brief Reduce a tensor to a smaller tensor.
 *
 * The function takes the `src` tensor, divides it into parts of
 * size of the `acc` tensor, reduces the parts together using
 * element-wise addition and writes the result into `acc`.
 *
 * @tparam AllocateAcc Indicates whether the results should be accumulated or
 * written to the `acc` tensor.
 * @tparam DataType Data type of the tensors.
 *
 * @param [in] acc Accumulation tensor.
 * @param [in] src Source tensor.
 * @param [in] src_len Length of source tensor.
 */
template <bool AllocateAcc, typename DataType>
__aicore__ inline void ReduceVecAdd(const LocalTensor<DataType>& acc,
                                    const LocalTensor<DataType>& src,
                                    uint32_t src_len) {
  exec_mode::AssertIsAIV();
  const uint32_t acc_size = acc.GetSize();

  const uint32_t num_iters = tcuscan::scalar::FloorDiv(src_len, acc_size);
  const uint32_t tail_len = src_len - num_iters * acc_size;

  if constexpr (AllocateAcc) {
    Duplicate(acc, static_cast<DataType>(0), acc_size);
  }
  for (uint32_t i = 0; i < num_iters; i++) {
    Add(acc, src[i * acc_size], acc, acc_size);
  }
  if (tail_len > 0) {
    Add(acc, src[num_iters * acc_size], acc, tail_len);
  }
}

/**
 * @brief Reduce a tensor to scalar using add operation.
 *
 * @tparam DataType Data type of the tensor.
 *
 * @param [in] src Source tensor.
 * @param [in] size Number of elements to reduce.
 *
 * @return The result of reduction.
 */
template <typename DataType>
__aicore__ inline DataType ReduceScalarAdd(const LocalTensor<DataType>& src,
                                           uint32_t size) {
  DataType acc = 0;
  for (uint32_t i = 0; i < size; i++) {
    acc += src.GetValue(i);
  }
  return acc;
}

/**
 * @brief Reduce a tensor to a smaller tensor using bitwise or.
 *
 * The function takes the `src` tensor, divides it into parts of
 * size of the `acc` tensor, reduces the parts together using
 * element-wise or operation and writes the result into `acc`.
 *
 * @tparam AllocateAcc Indicates whether the results should be accumulated or
 * written to the `acc` tensor.
 * @tparam DataType Data type of the tensors.
 *
 * @param [in] acc Accumulation tensor.
 * @param [in] src Source tensor.
 */
template <bool AllocateAcc, typename DataType>
__aicore__ inline void ReduceVecOr(const LocalTensor<DataType>& acc,
                                   const LocalTensor<DataType>& src) {
  exec_mode::AssertIsAIV();
  const uint32_t src_size = src.GetSize();
  const uint32_t acc_size = acc.GetSize();

  if (src_size == acc_size) {
    DataCopy(acc, src, acc_size);
  } else {
    constexpr uint32_t i_start = AllocateAcc ? 2 : 0;
    const uint32_t i_end = src_size / acc_size;
    if constexpr (AllocateAcc) {
      Or(acc, src[0], src[acc_size], acc_size);
    }
    for (uint32_t i = i_start; i < i_end; i++) {
      Or(acc, src[i * acc_size], acc, acc_size);
    }
  }
}

/**
 * @brief Reduce a tensor to scalar using bitwise or operation.
 *
 * @tparam DataType Data type of the tensor.
 *
 * @param [in] src Source tensor.
 *
 * @return The result of reduction.
 */
template <typename DataType>
__aicore__ inline DataType ReduceScalarOr(const LocalTensor<DataType>& src) {
  exec_mode::AssertIsAIV();
  DataType acc = 0;
  for (uint32_t i = 0; i < src.GetSize(); i++) {
    acc |= src.GetValue(i);
  }
  return acc;
}

}  // namespace reduce

namespace duplicate {

/**
 * @brief It is true if `DataType` is supported for the Duplicate instruction.
 *
 * @tparam DataType Data type to check.
 */
template <typename DataType>
constexpr bool IsDuplicateSupported = std::is_same<DataType, uint16_t>::value ||
                                      std::is_same<DataType, int16_t>::value ||
                                      std::is_same<DataType, half>::value ||
                                      std::is_same<DataType, uint32_t>::value ||
                                      std::is_same<DataType, int32_t>::value ||
                                      std::is_same<DataType, float>::value ||
                                      std::is_same<DataType, bfloat16_t>::value;

}  // namespace duplicate

namespace cube_unit {

/**
 * @brief It is true if `DataType` is supported as input for the Cube unit.
 *
 * @tparam DataType Data type to check.
 */
template <typename DataType>
constexpr bool IsCubeSupported =
    std::is_same_v<DataType, half> || std::is_same_v<DataType, float> ||
    std::is_same_v<DataType, int8_t> || std::is_same_v<DataType, uint8_t>;

/**
 * @brief A type metafunction for Cube's input / output types. The following
 * type pairs are supported:(half, float), (float, float), (int8_t, int32_t),
 * (uint8_t, uint32_t)
 *
 * @tparam InputT Input cube type. Must be int8_t, uint8_t, half, or float.
 */
template <typename InputT>
struct CubeOutType {
  /// @brief Type
  using type = InputT;
};

/**
 * @brief Cube data type map int8_t -> int32_t.
 */
template <>
struct CubeOutType<int8_t> {
  /// @brief Type
  using type = int32_t;
};

/**
 * @brief Cube data type map uint8_t -> uint32_t.
 */
template <>
struct CubeOutType<uint8_t> {
  /// @brief Type
  using type = uint32_t;
};

/**
 * @brief Cube data type map half -> float.
 */
template <>
struct CubeOutType<half> {
  /// @brief Type
  using type = float;
};

/**
 * @brief Cube data type map float -> float.
 */
template <>
struct CubeOutType<float> {
  /// @brief Type
  using type = float;
};

/**
 * @brief Syntactic sugar for `CubeOutType`
 * @tparam T Input type
 */
template <typename T>
using CubeOutType_t = typename CubeOutType<T>::type;

/**
 * @brief Multiply two matrices given their queues.
 *
 * The function deques the input matrices from A2 and B2 and computes their
 * product using the cube unit. The output can be accumulated.
 * After execution the input matrices are either freed or enqued again if they
 * need to be reused.
 *
 * @tparam InputT Data type of the input matrices.
 * @tparam accumulate_c If true accumulates the output adding it to the matrix
 * in the CO1 queue.
 * @tparam free_a If true it frees the first input matrix, if false it enques it
 * to reuse it.
 * @tparam free_b If true it frees the second input matrix, if false it enques
 * it to reuse it.
 *
 * @param [in] q_a Input queue for the matrix A with position A2.
 * @param [in] q_b Input queue for the matrix B with position B2.
 * @param [in] q_c Output queue for the matrix C with position CO1.
 * @param [in] M Height of matrix A.
 * @param [in] N Width of matrix B.
 * @param [in] K Width of matrix A and hwight of matrix B.
 */
template <typename InputT, bool accumulate_c = false, bool free_a = true,
          bool free_b = true>
__aicore__ inline void Multiply(TQue<QuePosition::A2, 1>& q_a,
                                TQue<QuePosition::B2, 1>& q_b,
                                TQue<QuePosition::CO1, 1>& q_c, uint16_t M,
                                uint16_t N, uint16_t K) {
  exec_mode::AssertIsAIC();
  static_assert(IsCubeSupported<InputT>,
                "Unsupported input Cube dtype. Please use half or int8_t.");
  using OutputT = CubeOutType_t<InputT>;
  LocalTensor<InputT> a2_lt = q_a.DeQue<InputT>();
  LocalTensor<InputT> b2_lt = q_b.DeQue<InputT>();
  LocalTensor<OutputT> c1_lt =
      accumulate_c ? q_c.DeQue<OutputT>() : q_c.AllocTensor<OutputT>();

  Mmad(c1_lt, a2_lt, b2_lt, {M, N, K, accumulate_c, 0, false, false, false});

  q_c.EnQue<OutputT>(c1_lt);

  if constexpr (free_a) {
    q_a.FreeTensor(a2_lt);
  } else {
    q_a.EnQue(a2_lt);
  }

  if constexpr (free_b) {
    q_b.FreeTensor(b2_lt);
  } else {
    q_b.EnQue(b2_lt);
  }
}

/**
 * @brief Performs matrix multiplication C = A @ C (where C lies in L0C).
 *
 * The function deques the input matrices from A2 and L0C and computes their
 * product using the cube unit. The output can be accumulated.
 * After execution the input matrix A is either freed or enqued again.
 *
 * @tparam InputT Data type of the input matrices.
 * @tparam free_a If true it frees the input matrix A, otherwise it enques A
 * back to `q_a` to reuse it.
 *
 * @param [in] q_a Input queue where matrix A must be already unqueued.
 * @param [in] b1_q Intermediate empty L1 queue for enqueuing C from L0C.
 * @param [in] b2_q Intermediate empty L0B queue for enqueuing C from L1.
 * @param [in] c_q Output queue where C (+)= A @ C will be unqueued. Must be
 * already enqueued with C.
 * @param [in] M Height of matrix A.
 * @param [in] N Width of matrix C.
 * @param [in] K Width of matrix A and height of matrix C.
 */
template <typename InputT, bool free_a = true>
__aicore__ inline void MultiplyAWithC(TQue<QuePosition::A2, 1>& q_a,
                                      TQue<QuePosition::B1, 1>& b1_q,
                                      TQue<QuePosition::B2, 1>& b2_q,
                                      TQue<QuePosition::CO1, 1>& c_q,
                                      uint16_t M, uint16_t N, uint16_t K) {
  exec_mode::AssertIsAIC();
  static_assert(IsCubeSupported<InputT>,
                "Unsupported input Cube dtype. Please use half or int8_t.");
  using OutputT = CubeOutType_t<InputT>;
  LocalTensor<InputT> a2_lt = q_a.DeQue<InputT>();

  // Load C matrix from L0C into L0B.
  copy::CopyC01ToB1<InputT, OutputT /* Source */, 1, 1>(b1_q, c_q, M, N);
  const uint32_t k_blocks_ = M / tcuscan::GetFractalK<InputT>();
  const uint32_t n_blocks_ = N / tcuscan::GetFractalMN<InputT>();
  copy::CopyL1ToL0B<InputT, true /* free_src */>(b2_q, b1_q, k_blocks_,
                                                 n_blocks_);
  LocalTensor<InputT> b2_lt = b2_q.DeQue<InputT>();

  LocalTensor<OutputT> c1_lt = c_q.AllocTensor<OutputT>();

  Mmad(c1_lt, a2_lt, b2_lt,
       {M, N, K, false /* accumulate_c */, 0, false, false, false});

  c_q.EnQue<OutputT>(c1_lt);

  if constexpr (free_a) {
    q_a.FreeTensor(a2_lt);
  } else {
    q_a.EnQue(a2_lt);
  }

  b2_q.FreeTensor(b2_lt);
}

/**
 * @brief Initialize a local tensor that is allocated in Cube L1/L0
 * (A1/A2/B1/B2) with a constant value
 *
 * @tparam T Input type can be half or int8 for now.
 * @tparam Pos Input local tensor position.
 *
 * @param [in] local_tensor Input local tensor to populate.
 * @param [in] value Value to populate `local_tensor` with.
 * @param [in] len Length of the local tensor to populate starting from index 0.
 */
template <typename T, TPosition Pos>
__aicore__ inline void InitConstL1(const LocalTensor<T>& local_tensor, T value,
                                   uint16_t len) {
  exec_mode::AssertIsAIC();
  static_assert(Pos == TPosition::A1 || Pos == TPosition::B1 ||
                    Pos == TPosition::A2 || Pos == TPosition::B2,
                "Local tensor must be either in A1, B1, A2 or B2");
  // If Pos is A1/B1, the block size is 32B else 512B
  constexpr uint32_t block_size =
      (Pos == TPosition::A1 || Pos == TPosition::B1) ? 32 : 256;

  InitConstValueParams<T> params;
  params.repeatTimes = 1;
  params.blockNum = static_cast<uint16_t>(len * sizeof(T) / block_size);
  params.dstGap = 0;
  params.initValue = value;
  InitConstValue<T>(local_tensor, params);
}

/**
 * @brief Initialize a local tensor of dtype `int8_t/half` that is allocated in
 * Cube L1/L0 (A1/A2/B1/B2) with constant all-ones.
 *
 * @tparam T Input data type. Supports half/int8_t.
 * @tparam Pos Input local tensor position.
 *
 * @param lt Input local tensor to populate with all-ones.
 * @param len Length of the local tensor to populate starting from index 0.
 */
template <typename T, TPosition Pos>
__aicore__ inline void InitConstAllOnesL1(LocalTensor<T>& lt, uint16_t len) {
  static_assert(Pos == TPosition::A1 || Pos == TPosition::B1 ||
                    Pos == TPosition::A2 || Pos == TPosition::B2,
                "Local tensor must be either in A1, B1, A2 or B2");

  static_assert(std::is_same_v<T, half> or std::is_same_v<T, int8_t>,
                "Local tensor must have fp16 or int8_t dtype.");

  if constexpr (std::is_same_v<T, int8_t>) {
    ASCENDC_ASSERT((len % 2 == 0), {
      KERNEL_LOG(KERNEL_ERROR,
                 "The length of the local tensor (%d) must be "
                 "divisible by 2.",
                 len);
    });
    // Note: (int16_t)257 = 0x0101
    constexpr int16_t TWO_INT8_ONES = 257;
    LocalTensor<int16_t> int16_lt = lt.template ReinterpretCast<int16_t>();
    InitConstL1<int16_t, Pos>(int16_lt, TWO_INT8_ONES, len / 2);
  }
  if constexpr (std::is_same_v<T, half>) {
    InitConstL1<half, Pos>(lt, static_cast<half>(1), len);
  }
}

}  // namespace cube_unit

namespace compare {

/**
 * @brief Performs a less-than comparison between an input local tensor and a
 * pivot scalar, i.e., {x_i <= pivot}. Writes the 0/1 binary element-wise
 * outcome of the comparison in `out_lt`.
 *
 * @tparam T Input data type. Supports half, int16_t and uint16_t.
 * @tparam OutputT Output data type where binary outcome of comparison is
 * stored. Supports int8_t and uint8_t.
 *
 * @param [in] out_lt Output where the binary outcome of the comparison is
 * written.
 * @param [in] input_lt Input tensor.
 * @param [in] pivot Pivot value for less-than comparison.
 * @param [in] work_lt Temporary workspace to use for internal calculations.
 */
template <typename T, typename OutputT>
__aicore__ inline void LessThan(const LocalTensor<OutputT>& out_lt,
                                const LocalTensor<T>& input_lt, T pivot,
                                const LocalTensor<T>& work_lt) {
  static_assert(
      std::is_same_v<T, half> or std::is_same_v<T, int16_t> or
          std::is_same_v<T, uint16_t>,
      "LessThan supports only half, int16_t and uint16_t data types.");
  static_assert(
      std::is_same_v<OutputT, int8_t> or std::is_same_v<OutputT, uint8_t>,
      "LessThan supports only int8_t or uint8_t output data types.");

  const uint32_t size = work_lt.GetSize();

  // These buffers point to the same UB location but have different
  // types. Manual synchronization is needed!
  const LocalTensor<uint16_t> x_uint16_lt =
      work_lt.template ReinterpretCast<uint16_t>();
  const LocalTensor<int16_t> x_int16_lt =
      work_lt.template ReinterpretCast<int16_t>();
  const LocalTensor<half> x_half_lt = work_lt.template ReinterpretCast<half>();

  // Step 1. Perform work_lt = (pivot - x_i)
  AscendC::Muls<T>(work_lt, input_lt, static_cast<T>(-1), size);
  AscendC::Adds<T>(work_lt, work_lt, pivot, size);

  AscendC::PipeBarrier<PIPE_V>();

  // x_uint16_lt will be '1' iff (pivot - x_i) > 0, otherwise is zero.
  AscendC::Not(x_uint16_lt, x_uint16_lt, size);
  AscendC::ShiftRight(x_uint16_lt, x_uint16_lt, (uint16_t)15, size);

  AscendC::PipeBarrier<PIPE_V>();

  // Cast 0/1 values to OutputT local tensor.
  AscendC::Cast<half, int16_t>(x_half_lt, x_int16_lt, RoundMode::CAST_NONE,
                               size);
  AscendC::Cast<OutputT, half>(out_lt, x_half_lt, RoundMode::CAST_NONE, size);
}

/**
 * @brief Performs a greater-equal comparison between an input local tensor and
 * a pivot scalar, i.e., {x_i >= pivot}. Writes the 0/1 binary element-wise
 * outcome of the comparison in `out_lt`.
 *
 * @tparam T Input data type. Supports half, int16_t and uint16_t.
 * @tparam OutputT Output data type where binary outcome of comparison is
 * stored. Supports int8_t and uint8_t.
 *
 * @param [in] out_lt Output where the binary outcome of the comparison is
 * written.
 * @param [in] input_lt Input tensor.
 * @param [in] pivot Pivot value for less-than comparison.
 * @param [in] work_lt Temporary workspace to use for internal calculations.
 * @param [in] size Number of elements to process.
 */
template <typename T, typename OutputT>
__aicore__ inline void GreaterEqual(const LocalTensor<OutputT>& out_lt,
                                    const LocalTensor<T>& input_lt, T pivot,
                                    const LocalTensor<T>& work_lt,
                                    uint32_t size) {
  static_assert(
      std::is_same_v<T, half> or std::is_same_v<T, int16_t> or
          std::is_same_v<T, uint16_t>,
      "GreaterThan supports only half, int16_t and uint16_t data types.");
  static_assert(
      std::is_same_v<OutputT, int8_t> or std::is_same_v<OutputT, uint8_t>,
      "GreaterThan supports only int8_t or uint8_t output data types.");

  // These buffers point to the same UB location but have different
  // types. Manual synchronization is needed!
  const LocalTensor<uint16_t> x_uint16_lt =
      work_lt.template ReinterpretCast<uint16_t>();
  const LocalTensor<int16_t> x_int16_lt =
      work_lt.template ReinterpretCast<int16_t>();
  const LocalTensor<half> x_half_lt = work_lt.template ReinterpretCast<half>();

  // Step 1. Perform work_lt = (x_i - pivot)
  const float minus_pivot = -static_cast<float>(pivot);
  AscendC::DataCopy<T>(work_lt, input_lt, size);
  AscendC::Adds<T>(work_lt, work_lt, static_cast<T>(minus_pivot), size);

  AscendC::PipeBarrier<PIPE_V>();

  // x_uint16_lt will be '1' iff (x_i - pivot) > 0, otherwise is zero.
  AscendC::Not(x_uint16_lt, x_uint16_lt, size);
  AscendC::ShiftRight(x_uint16_lt, x_uint16_lt, (uint16_t)15, size);

  AscendC::PipeBarrier<PIPE_V>();

  // Cast 0/1 values to OutputT local tensor.
  AscendC::Cast<half, int16_t>(x_half_lt, x_int16_lt, RoundMode::CAST_NONE,
                               size);
  AscendC::Cast<OutputT, half>(out_lt, x_half_lt, RoundMode::CAST_NONE, size);
}

}  // namespace compare

namespace fp16 {
/// @brief Float number with the smallest aboslute value.
constexpr float FP16_MIN_NORMAL = std::numeric_limits<half>::min();
/// @brief Float number with the largest value.
constexpr float FP16_MAX_NORMAL = std::numeric_limits<half>::max();
/// @brief Float number with the most negative value.
constexpr float FP16_LOWEST_NORMAL = 0xfc00;
/// @brief Difference between float 1 and the next representable value.
constexpr float FP16_EPSILON = 0x1.0p-10;
/// @brief Float 1.
constexpr float FP16_ONE = 1.0f;
/// @brief Float 0.5.
constexpr float FP16_HALF = 0.5f;
/// @brief Float 0.
constexpr float FP16_ZERO = 0.0f;
/// @brief Next representable value after float 1.
constexpr float FP16_ONE_P_ULP = FP16_ONE + FP16_EPSILON;
/// @brief Increment ratio
constexpr float FP16_INC = FP16_ONE_P_ULP * FP16_EPSILON * FP16_HALF;

/**
 * @brief Computes the next representable value
 *
 * @param [in] val Current value.
 *
 * @return Next representable value after the current value.
 */
__aicore__ inline float next_after(float val) { return val * FP16_INC + val; }
}  // namespace fp16

}  // namespace tcuscan
