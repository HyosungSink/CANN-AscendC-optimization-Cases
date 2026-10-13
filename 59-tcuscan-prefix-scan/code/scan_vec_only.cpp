/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file scan_vec_only.cpp
 * @brief Entrypoint for scan vector-only kernel operation.
 */

#include "kernels/kernel_scan_vec_only.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_scan_vec_only.h"

/**
 * @brief Run the scan vec only kernel.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void scan_vec_only_fp16(GM_ADDR input_vec,
                                                         GM_ADDR output_vec,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  (void)workspace;
  tcuscan::ScanVecOnlyTiling tiling_data;
  tcuscan::GetTilingData(&tiling_data, tiling);

  if ASCEND_IS_AIV {
    tcuscan::KernelScanVecOnly op(
        tiling_data.num_elems, tiling_data.tile_height, tiling_data.tile_width);
    op.Init(input_vec, output_vec);
    op.Process();
  }
}

/**
 * @brief Launch the scan vec only kernel.
 *
 * @param [in] blockDim The number of blocks to launch.
 * @param [in] stream The stream to launch the kernel on.
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to the tiling structure.
 */
extern "C" void launch_scan_vec_only_fp16(uint32_t blockDim, void* stream,
                                          GM_ADDR input_vec, GM_ADDR output_vec,
                                          GM_ADDR workspace, GM_ADDR tiling) {
  scan_vec_only_fp16<<<blockDim, nullptr, stream>>>(input_vec, output_vec,
                                                    workspace, tiling);
}
