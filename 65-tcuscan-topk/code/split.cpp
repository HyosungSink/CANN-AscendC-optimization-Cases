/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file split.cpp
 * @brief Entrypoint for parallel split kernel.
 */

#include "kernels/constants.h"
#include "kernels/kernel_split.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_split.h"

/**
 * @brief Run the `split_uint16` kernel.
 *
 * @param [in] in Pointer to input vector.
 * @param [in] mask Pointer to mask vector.
 * @param [in] out Pointer to output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to tiling vector.
 */
extern "C" __global__ __aicore__ void split_uint16(GM_ADDR in, GM_ADDR mask,
                                                   GM_ADDR out,
                                                   GM_ADDR workspace,
                                                   GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  constexpr bool zeros_first = false;
  tcuscan::SplitTiling tiling_data;
  GetTilingData(&tiling_data, tiling);

  tcuscan::run_split_uint16(in, mask, out, workspace, tiling_data.num_elems,
                            tiling_data.vec_tile_size, zeros_first);
}

/**
 * @brief Run the `split_ind_uint16` kernel.
 *
 * @param [in] vec_in Pointer to input vector.
 * @param [in] mask_in Pointer to mask vector.
 * @param [in] indices_in Pointer to input indices vector.
 * @param [in] vec_out Pointer to output vector.
 * @param [in] indices_out Pointer to output indices vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to tiling vector.
 */
extern "C" __global__ __aicore__ void split_ind_uint16(
    GM_ADDR vec_in, GM_ADDR mask_in, GM_ADDR indices_in, GM_ADDR vec_out,
    GM_ADDR indices_out, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  constexpr bool zeros_first = false;
  tcuscan::SplitTiling tiling_data;
  GetTilingData(&tiling_data, tiling);

  tcuscan::run_split_ind_uint16(vec_in, mask_in, indices_in, vec_out,
                                indices_out, workspace, tiling_data.num_elems,
                                tiling_data.vec_tile_size, zeros_first);
}

/**
 * @brief Call the `split` kernel for UINT16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] in Pointer to an input buffer.
 * @param [in] mask Pointer to an input buffer.
 * @param [in] out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to the tiling buffer.
 */
extern "C" void launch_split_uint16(uint32_t blockDim, void* stream,
                                    uint8_t* in, uint8_t* mask, uint8_t* out,
                                    uint8_t* workspace, uint8_t* tiling) {
  split_uint16<<<blockDim, nullptr, stream>>>(in, mask, out, workspace, tiling);
}

/**
 * @brief Call the `split_ind` kernel for UINT16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] mask_in Pointer to an input buffer.
 * @param [in] indices_in Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] indices_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to the tiling buffer.
 */
extern "C" void launch_split_ind_uint16(uint32_t blockDim, void* stream,
                                        uint8_t* vec_in, uint8_t* mask_in,
                                        uint8_t* indices_in, uint8_t* vec_out,
                                        uint8_t* indices_out,
                                        uint8_t* workspace, uint8_t* tiling) {
  split_ind_uint16<<<blockDim, nullptr, stream>>>(
      vec_in, mask_in, indices_in, vec_out, indices_out, workspace, tiling);
}
