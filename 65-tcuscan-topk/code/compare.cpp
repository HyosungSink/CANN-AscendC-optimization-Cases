/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file compare.cpp
 * @brief Comparison kernels (greater/less equal, etc).
 */

#include "kernels/kernel_count_if.h"
#include "kernels/kernel_greater_equal.h"
#include "kernels/kernel_less_or_equal.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_count_if.h"
#include "tiling/tiling_greater_equal.h"

using namespace AscendC;

/**
 * @brief Run the `count_if` kernel with input fp16.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling structure.
 */

extern "C" __global__ __aicore__ void count_if_fp16(GM_ADDR vec_in,
                                                    GM_ADDR vec_out,
                                                    GM_ADDR workspace,
                                                    GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  (void)workspace;
  tcuscan::CountIfTiling tiling;
  GetTilingData(&tiling, tiling_gm);
  const uint32_t vec_len = tiling.vec_len;
  const uint32_t tile_len = tiling.tile_len;
  const half pivot = static_cast<half>(tiling.pivot);
  const AscendC::CMPMODE compare_mode{tiling.compare_mode};

  tcuscan::run_count_if<half>(vec_in, vec_out, vec_len, tile_len, pivot,
                              compare_mode);
}

/**
 * @brief Greater-equal kernel for dtype fp16
 *
 * @param [in] vec_in Input data vector
 * @param [in] vec_out Output vector
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to tiling structure.
 */
extern "C" __global__ __aicore__ void greater_equal_fp16(GM_ADDR vec_in,
                                                         GM_ADDR vec_out,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  (void)workspace;
  tcuscan::GreaterEqualTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const half pivot = static_cast<half>(tiling.pivot);
  const uint32_t tile_len = tiling.tile_len;

  tcuscan::run_greater_equal<half>(vec_in, pivot, vec_out, vec_len, tile_len);
}

/**
 * @brief Less-equal kernel for dtype fp16
 *
 * @param [in] vec_in Input data vector
 * @param [in] vec_out Output vector
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to tiling structure.
 */
extern "C" __global__ __aicore__ void less_equal_fp16(GM_ADDR vec_in,
                                                      GM_ADDR vec_out,
                                                      GM_ADDR workspace,
                                                      GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  (void)workspace;
  tcuscan::GreaterEqualTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const half pivot = static_cast<half>(tiling.pivot);
  const uint32_t tile_len = tiling.tile_len;

  tcuscan::run_less_or_equal<true, half>(vec_in, vec_out, pivot, vec_len,
                                         tile_len);
}

/**
 * @brief Call the `count_if` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_count_if_fp16(uint32_t blockDim, void* stream,
                                     uint8_t* vec_in, uint8_t* vec_out,
                                     uint8_t* workspace, uint8_t* tiling_gm) {
  count_if_fp16<<<blockDim, nullptr, stream>>>(vec_in, vec_out, workspace,
                                               tiling_gm);
}

/**
 * @brief Call the `greater_equal` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_greater_equal_fp16(uint32_t blockDim, void* stream,
                                          uint8_t* vec_in, uint8_t* vec_out,
                                          uint8_t* workspace,
                                          uint8_t* tiling_gm) {
  greater_equal_fp16<<<blockDim, nullptr, stream>>>(vec_in, vec_out, workspace,
                                                    tiling_gm);
}

/**
 * @brief Call the `less_equal` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_less_equal_fp16(uint32_t blockDim, void* stream,
                                       uint8_t* vec_in, uint8_t* vec_out,
                                       uint8_t* workspace, uint8_t* tiling_gm) {
  less_equal_fp16<<<blockDim, nullptr, stream>>>(vec_in, vec_out, workspace,
                                                 tiling_gm);
}
