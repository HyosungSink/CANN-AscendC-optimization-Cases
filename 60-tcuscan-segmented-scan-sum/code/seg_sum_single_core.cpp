/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file seg_sum_single_core.cpp
 * @brief Entrypoint for segmented sum single core kernel operation.
 */

#include "kernels/constants.h"
#include "kernels/kernel_seg_sum_single_core.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_seg_sum_single_core.h"

using namespace AscendC;
using namespace tcuscan;

/**
 * @brief Run the `seg_sum_single_core` kernel with half/float16 dtype.
 *
 * The segment indices format follows the scipy Compressed Sparse Row Matrix
 * convention
 * (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] indptr Pointer to the segment indices vector. Use `indptr[1:-1]`
 * in Python.
 * @param [in] vec_out Pointer to the output segmented sum.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void seg_sum_single_core_fp16(
    GM_ADDR vec_in, GM_ADDR indptr, GM_ADDR vec_out, GM_ADDR workspace,
    GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::SegSumSingleCoreTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t num_segments = tiling.num_segments;
  const uint32_t matmul_size = tiling.tile_len;

  GM_ADDR const lower = load_tril_matrix<half>(matmul_size);

  tcuscan::run_seg_sum_single_core<half>(vec_in, lower, indptr, vec_out,
                                         workspace, vec_len, num_segments,
                                         matmul_size);
}

/**
 * @brief Run the `seg_sum_single_core` kernel with `int8_t` dtype.
 *
 * The segment indices format follows the scipy Compressed Sparse Row Matrix
 * convention
 * (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] indptr Pointer to the segment indices vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void seg_sum_single_core_int8(
    GM_ADDR vec_in, GM_ADDR indptr, GM_ADDR vec_out, GM_ADDR workspace,
    GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::SegSumSingleCoreTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t num_segments = tiling.num_segments;
  const uint32_t matmul_size = tiling.tile_len;

  GM_ADDR const lower = load_tril_matrix<int8_t>(matmul_size);

  tcuscan::run_seg_sum_single_core<int8_t>(vec_in, lower, indptr, vec_out,
                                           workspace, vec_len, num_segments,
                                           matmul_size);
}

/**
 * @brief Call the `seg_sum_single_core` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] indptr Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_seg_sum_single_core_fp16(
    uint32_t blockDim, void* stream, uint8_t* vec_in, uint8_t* indptr,
    uint8_t* vec_out, uint8_t* workspace, uint8_t* tiling_gm) {
  seg_sum_single_core_fp16<<<blockDim, nullptr, stream>>>(
      vec_in, indptr, vec_out, workspace, tiling_gm);
}

/**
 * @brief Call the `seg_sum_single_core` kernel for INT8 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] indptr Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_seg_sum_single_core_int8(
    uint32_t blockDim, void* stream, uint8_t* vec_in, uint8_t* indptr,
    uint8_t* vec_out, uint8_t* workspace, uint8_t* tiling_gm) {
  seg_sum_single_core_int8<<<blockDim, nullptr, stream>>>(
      vec_in, indptr, vec_out, workspace, tiling_gm);
}
