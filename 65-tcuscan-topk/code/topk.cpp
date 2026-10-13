/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file topk.cpp
 * @brief Entrypoint for parallel topk kernel.
 */

#include "kernels/constants.h"
#include "kernels/kernel_topk.h"
#include "kernels/kernel_topk_pivot.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_topk.h"
#include "tiling/tiling_topk_pivot.h"

/**
 * @brief Run the `topk` (int16) kernel.
 *
 * @param [in] vec_in Pointer to input vector.
 * @param [in] vec_out Pointer to output vector.
 * @param [in] indices_out Pointer to output indices vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_ptr Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void topk_int16(GM_ADDR vec_in,
                                                 GM_ADDR vec_out,
                                                 GM_ADDR indices_out,
                                                 GM_ADDR workspace,
                                                 GM_ADDR tiling_ptr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::TopKTiling tiling;
  GetTilingData(&tiling, tiling_ptr);

  const int32_t x_min = tiling.x_min.value_i32;
  const int32_t x_max = tiling.x_max.value_i32;

  tcuscan::run_topk<int16_t>(vec_in, tiling.k, vec_out, indices_out, workspace,
                             x_min, x_max, tiling.num_elems,
                             tiling.vec_tile_size);
}

/**
 * @brief Run the `topk` (fp16) kernel.
 *
 * @param [in] vec_in Pointer to input vector.
 * @param [in] vec_out Pointer to output vector.
 * @param [in] indices_out Pointer to output indices vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_ptr Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void topk_fp16(GM_ADDR vec_in, GM_ADDR vec_out,
                                                GM_ADDR indices_out,
                                                GM_ADDR workspace,
                                                GM_ADDR tiling_ptr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::TopKTiling tiling;
  GetTilingData(&tiling, tiling_ptr);

  const half x_min = static_cast<half>(tiling.x_min.value_fp32);
  const half x_max = static_cast<half>(tiling.x_max.value_fp32);

  tcuscan::run_topk<half>(vec_in, tiling.k, vec_out, indices_out, workspace,
                          x_min, x_max, tiling.num_elems, tiling.vec_tile_size);
}

/**
 * @brief Run the `topk_pivot` (fp16) kernel.
 *
 * @param [in] vec_in Pointer to input vector.
 * @param [in] vec_out Pointer to output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_ptr Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void topk_pivot_fp16(GM_ADDR vec_in,
                                                      GM_ADDR vec_out,
                                                      GM_ADDR workspace,
                                                      GM_ADDR tiling_ptr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::TopKPivotTiling tiling;
  GetTilingData(&tiling, tiling_ptr);

  tcuscan::run_topk_pivot<false, half>(vec_in, vec_out, tiling.num_elems,
                                       tiling.num_samples, tiling.k_inner,
                                       tiling.k_outer);
}

/**
 * @brief Call the `topk` kernel for INT16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] indices_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_ptr Pointer to the tiling buffer.
 */
extern "C" void launch_topk_int16(uint32_t blockDim, void* stream,
                                  uint8_t* vec_in, uint8_t* vec_out,
                                  uint8_t* indices_out, uint8_t* workspace,
                                  uint8_t* tiling_ptr) {
  topk_int16<<<blockDim, nullptr, stream>>>(vec_in, vec_out, indices_out,
                                            workspace, tiling_ptr);
}

/**
 * @brief Call the `topk` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] indices_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_ptr Pointer to the tiling buffer.
 */
extern "C" void launch_topk_fp16(uint32_t blockDim, void* stream,
                                 uint8_t* vec_in, uint8_t* vec_out,
                                 uint8_t* indices_out, uint8_t* workspace,
                                 uint8_t* tiling_ptr) {
  topk_fp16<<<blockDim, nullptr, stream>>>(vec_in, vec_out, indices_out,
                                           workspace, tiling_ptr);
}

/**
 * @brief Call the `topk_pivot` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_ptr Pointer to the tiling buffer.
 */
extern "C" void launch_topk_pivot_fp16(uint32_t blockDim, void* stream,
                                       uint8_t* vec_in, uint8_t* vec_out,
                                       uint8_t* workspace,
                                       uint8_t* tiling_ptr) {
  topk_pivot_fp16<<<blockDim, nullptr, stream>>>(vec_in, vec_out, workspace,
                                                 tiling_ptr);
}
