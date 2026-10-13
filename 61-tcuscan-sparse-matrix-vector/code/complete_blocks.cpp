/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file complete_blocks.cpp
 * @brief Kernel implementing a multi-core AIV complete blocks phase of block
 * scan.
 */

#include "kernels/kernel_complete_blocks.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_complete_blocks.h"

namespace tcuscan {

/**
 * @brief Run the multi core vector complete blocks kernel.
 *
 * @tparam T Input data type.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] input_sums Pointer to vector with partial block reductions
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
template <typename T>
__aicore__ inline void run_complete_blocks(GM_ADDR input_vec,
                                           GM_ADDR input_sums,
                                           GM_ADDR output_vec,
                                           GM_ADDR tiling_gm) {
  tcuscan::CompleteBlocksTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t num_blocks = tiling.num_blocks;
  const uint32_t tile_len = tiling.tile_len;

  if ASCEND_IS_AIV {
    const uint32_t block_len = vec_len / num_blocks;
    KernelCompleteBlocks<T> op_complete_blocks(vec_len, block_len, tile_len);
    op_complete_blocks.Init(input_vec, input_sums, output_vec);
    op_complete_blocks.Process();
  }
}

}  // namespace tcuscan

/**
 * @brief Run the multi core vector complete blocks kernel with dtype fp32.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] input_sums Pointer to vector with partial block reductions
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void complete_blocks_fp32(GM_ADDR input_vec,
                                                           GM_ADDR input_sums,
                                                           GM_ADDR output_vec,
                                                           GM_ADDR workspace,
                                                           GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  (void)workspace;
  tcuscan::run_complete_blocks<float>(input_vec, input_sums, output_vec,
                                      tiling_gm);
}

/**
 * @brief Run the `complete_blocks` kernel.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] input_sums Pointer to vector with partial block reductions
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void complete_blocks_int32(GM_ADDR input_vec,
                                                            GM_ADDR input_sums,
                                                            GM_ADDR output_vec,
                                                            GM_ADDR workspace,
                                                            GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  (void)workspace;
  tcuscan::run_complete_blocks<int32_t>(input_vec, input_sums, output_vec,
                                        tiling_gm);
}

/**
 * @brief Call the `complete_blocks` kernel for FP32 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] input_vec Pointer to an input buffer.
 * @param [in] input_sums Pointer to an input buffer.
 * @param [in] output_vec Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_complete_blocks_fp32(
    uint32_t blockDim, void* stream, uint8_t* input_vec, uint8_t* input_sums,
    uint8_t* output_vec, uint8_t* workspace, uint8_t* tiling_gm) {
  complete_blocks_fp32<<<blockDim, nullptr, stream>>>(
      input_vec, input_sums, output_vec, workspace, tiling_gm);
}

/**
 * @brief Call the `complete_blocks` kernel for INT32 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] input_vec Pointer to an input buffer.
 * @param [in] input_sums Pointer to an input buffer.
 * @param [in] output_vec Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_complete_blocks_int32(
    uint32_t blockDim, void* stream, uint8_t* input_vec, uint8_t* input_sums,
    uint8_t* output_vec, uint8_t* workspace, uint8_t* tiling_gm) {
  complete_blocks_int32<<<blockDim, nullptr, stream>>>(
      input_vec, input_sums, output_vec, workspace, tiling_gm);
}
