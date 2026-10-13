/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file scan_single_core.cpp
 * @brief Entrypoint for scan single core kernel operation.
 */

#include "kernels/constants.h"
#include "kernels/kernel_scan_single_core.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_scan_single_core.h"

/**
 * @brief Run the `scan_single_core` kernel with int8 dtype.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */

extern "C" __global__ __aicore__ void scan_single_core_int8(GM_ADDR vec_in,
                                                            GM_ADDR vec_out,
                                                            GM_ADDR workspace,
                                                            GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::SingleCoreScanTiling tiling;
  GetTilingData(&tiling, tilingGm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t matmul_size = tiling.matmul_size;
  const int32_t running_sum = tiling.running_sum.int_value;
  GM_ADDR const lower = load_tril_matrix<int8_t>(matmul_size);

  tcuscan::run_scan_single_core<int8_t>(vec_in, lower, vec_out, vec_len,
                                        matmul_size, workspace, running_sum);
}

/**
 * @brief Run the `scan_single_core` kernel with half/float16 dtype.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */

extern "C" __global__ __aicore__ void scan_single_core_fp16(GM_ADDR vec_in,
                                                            GM_ADDR vec_out,
                                                            GM_ADDR workspace,
                                                            GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::SingleCoreScanTiling tiling;
  GetTilingData(&tiling, tilingGm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t matmul_size = tiling.matmul_size;
  const float running_sum = tiling.running_sum.float_value;

  GM_ADDR const lower = load_tril_matrix<half>(matmul_size);

  tcuscan::run_scan_single_core<half>(vec_in, lower, vec_out, vec_len,
                                      matmul_size, workspace, running_sum);
}

/**
 * @brief Run the `scan_single_core` kernel with float/float32 dtype.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */

extern "C" __global__ __aicore__ void scan_single_core_fp32(GM_ADDR vec_in,
                                                            GM_ADDR vec_out,
                                                            GM_ADDR workspace,
                                                            GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::SingleCoreScanTiling tiling;
  GetTilingData(&tiling, tilingGm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t matmul_size = tiling.matmul_size;
  const float running_sum = tiling.running_sum.float_value;

  GM_ADDR const lower = load_tril_matrix<float>(matmul_size);

  tcuscan::run_scan_single_core<float>(vec_in, lower, vec_out, vec_len,
                                       matmul_size, workspace, running_sum);
}

/**
 * @brief Call the `scan_single_core` kernel for INT8 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" void launch_scan_single_core_int8(uint32_t blockDim, void* stream,
                                             uint8_t* vec_in, uint8_t* vec_out,
                                             uint8_t* workspace,
                                             uint8_t* tilingGm) {
  scan_single_core_int8<<<blockDim, nullptr, stream>>>(vec_in, vec_out,
                                                       workspace, tilingGm);
}

/**
 * @brief Call the `scan_single_core` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" void launch_scan_single_core_fp16(uint32_t blockDim, void* stream,
                                             uint8_t* vec_in, uint8_t* vec_out,
                                             uint8_t* workspace,
                                             uint8_t* tilingGm) {
  scan_single_core_fp16<<<blockDim, nullptr, stream>>>(vec_in, vec_out,
                                                       workspace, tilingGm);
}

/**
 * @brief Call the `scan_single_core` kernel for FP32 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" void launch_scan_single_core_fp32(uint32_t blockDim, void* stream,
                                             uint8_t* vec_in, uint8_t* vec_out,
                                             uint8_t* workspace,
                                             uint8_t* tilingGm) {
  scan_single_core_fp32<<<blockDim, nullptr, stream>>>(vec_in, vec_out,
                                                       workspace, tilingGm);
}
