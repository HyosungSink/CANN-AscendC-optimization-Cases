/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file gather_spmv.cpp
 * @brief Entrypoint for gather_spmv kernel operation.
 */

#include "kernels/kernel_gather_spmv.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_gather_spmv.h"

/**
 * @brief Gather SpMV kernel for CSR sparse matrices.
 *
 * @param values_in Input data vector.
 * @param cols_in Input column indices of CSR sparse format.
 * @param vec_out Output vector.
 * @param workspace Pointer to workspace.
 * @param tilingGm Pointer to tiling structure.
 */
extern "C" __global__ __aicore__ void gather_spmv(GM_ADDR values_in,
                                                  GM_ADDR cols_in,
                                                  GM_ADDR vec_out,
                                                  GM_ADDR workspace,
                                                  GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  (void)workspace;
  tcuscan::GatherSpmvTiling tiling;
  GetTilingData(&tiling, tilingGm);

  tcuscan::run_gather_spmv(values_in, cols_in, vec_out, tiling.idx_len,
                           tiling.val_len, tiling.tile_len, tiling.alpha);
}

/**
 * @brief Launch the `gather_spmv` kernel.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] values_in Pointer to an input buffer.
 * @param [in] cols_in Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" void launch_gather_spmv(uint32_t blockDim, void* stream,
                                   uint8_t* values_in, uint8_t* cols_in,
                                   uint8_t* vec_out, uint8_t* workspace,
                                   uint8_t* tilingGm) {
  gather_spmv<<<blockDim, nullptr, stream>>>(values_in, cols_in, vec_out,
                                             workspace, tilingGm);
}
