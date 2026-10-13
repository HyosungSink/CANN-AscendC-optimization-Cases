/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file cube_reduce.cpp
 * @brief Kernel implementing a multi-core MIX block reducer.
 */
#include "kernels/kernel_cube_reduce.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_cube_reduce.h"

/**
 * @ingroup reduce
 * @brief Run `cube_reduce` kernel with dtype fp16.
 *
 * @param [in] vec_in Pointer to an input vector.
 * @param [in] vec_out Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void cube_reduce_fp16(GM_ADDR vec_in,
                                                       GM_ADDR vec_out,
                                                       GM_ADDR workspace,
                                                       GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::CubeReduceTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t matmul_size = tiling.matmul_size;

  tcuscan::run_cube_reduce<half>(vec_in, vec_out, workspace, vec_len,
                                 matmul_size);
}

/**
 * @ingroup reduce
 * @brief Run `cube_reduce` kernel with dtype int8.
 *
 * @param [in] vec_in Pointer to an input vector.
 * @param [in] vec_out Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void cube_reduce_int8(GM_ADDR vec_in,
                                                       GM_ADDR vec_out,
                                                       GM_ADDR workspace,
                                                       GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::CubeReduceTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t matmul_size = tiling.matmul_size;

  tcuscan::run_cube_reduce<int8_t>(vec_in, vec_out, workspace, vec_len,
                                   matmul_size);
}

/**
 * @brief Call the `cube_reduce` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_cube_reduce_fp16(uint32_t blockDim, void* stream,
                                        uint8_t* vec_in, uint8_t* vec_out,
                                        uint8_t* workspace,
                                        uint8_t* tiling_gm) {
  cube_reduce_fp16<<<blockDim, nullptr, stream>>>(vec_in, vec_out, workspace,
                                                  tiling_gm);
}

/**
 * @brief Call the `cube_reduce` kernel for INT8 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_cube_reduce_int8(uint32_t blockDim, void* stream,
                                        uint8_t* vec_in, uint8_t* vec_out,
                                        uint8_t* workspace,
                                        uint8_t* tiling_gm) {
  cube_reduce_int8<<<blockDim, nullptr, stream>>>(vec_in, vec_out, workspace,
                                                  tiling_gm);
}
