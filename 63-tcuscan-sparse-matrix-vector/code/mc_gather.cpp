/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file mc_gather.cpp
 * @brief Kernel implementing the Vector MCGather kernel operation.
 */

#include "kernels/kernel_mc_gather.h"
#include "tiling/tiling_mc_gather.h"

/**
 * @brief Run the `mc_gather` kernel for dtype fp32.
 *
 * @param [in] values_in Pointer to input vector.
 * @param [in] indexes_in Pointer to column indices input vector.
 * @param [in] z_out Pointer to output vector.
 * @param [in] workspace Length of the input values vector.
 * @param [in] tiling_gm Length of the tile processed in a single iteration.
 */
extern "C" __global__ __aicore__ void mc_gather_fp32(GM_ADDR values_in,
                                                     GM_ADDR indexes_in,
                                                     GM_ADDR z_out,
                                                     GM_ADDR workspace,
                                                     GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::McGatherTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  tcuscan::run_mc_gather<float>(values_in, indexes_in, z_out, tiling.idx_len,
                                tiling.val_len, tiling.tile_len);
}

/**
 * @brief Run the `mc_gather` kernel for dtype fp32.
 *
 * @param [in] values_in Pointer to input vector.
 * @param [in] indexes_in Pointer to column indices input vector.
 * @param [in] z_out Pointer to output vector.
 * @param [in] workspace Length of the input values vector.
 * @param [in] tiling_gm Length of the tile processed in a single iteration.
 */
extern "C" __global__ __aicore__ void mc_gather_fp16(GM_ADDR values_in,
                                                     GM_ADDR indexes_in,
                                                     GM_ADDR z_out,
                                                     GM_ADDR workspace,
                                                     GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::McGatherTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  tcuscan::run_mc_gather<half>(values_in, indexes_in, z_out, tiling.idx_len,
                               tiling.val_len, tiling.tile_len);
}

/**
 * @brief Call the `mc_gather` kernel for FP32 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] values_in Pointer to an input buffer.
 * @param [in] indexes_in Pointer to an input buffer.
 * @param [in] z_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_mc_gather_fp32(uint32_t blockDim, void* stream,
                                      uint8_t* values_in, uint8_t* indexes_in,
                                      uint8_t* z_out, uint8_t* workspace,
                                      uint8_t* tiling_gm) {
  mc_gather_fp32<<<blockDim, nullptr, stream>>>(values_in, indexes_in, z_out,
                                                workspace, tiling_gm);
}

/**
 * @brief Call the `mc_gather` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] values_in Pointer to an input buffer.
 * @param [in] indexes_in Pointer to an input buffer.
 * @param [in] z_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_mc_gather_fp16(uint32_t blockDim, void* stream,
                                      uint8_t* values_in, uint8_t* indexes_in,
                                      uint8_t* z_out, uint8_t* workspace,
                                      uint8_t* tiling_gm) {
  mc_gather_fp16<<<blockDim, nullptr, stream>>>(values_in, indexes_in, z_out,
                                                workspace, tiling_gm);
}
