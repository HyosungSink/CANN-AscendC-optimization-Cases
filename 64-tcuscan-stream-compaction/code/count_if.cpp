/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file count_if.cpp
 * @brief Kernel implementing Vector `count_if` operation.
 */

#include "kernels/ascendc_kernel_operator.h"
#include "kernels/kernel_count_if.h"
#include "tiling/tiling_count_if.h"

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
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::CountIfTiling tiling;
  GetTilingData(&tiling, tiling_gm);
  const AscendC::CMPMODE mode =
      static_cast<AscendC::CMPMODE>(tiling.compare_mode);

  tcuscan::run_count_if<half>(vec_in, vec_out, tiling.vec_len, tiling.tile_len,
                              tiling.pivot, mode);
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
