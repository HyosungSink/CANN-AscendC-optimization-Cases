/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file block_scan.cpp
 * @brief Kernel implementing a multi-core inclusive block scan.
 */

#include "kernels/constants.h"
#include "kernels/kernel_block_scan.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_block_scan.h"

namespace tcuscan {

/**
 * @brief Run the multi core inclusive block scan kernel
 *
 *
 * @tparam InputT Input data type.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] lower Pointer to upper-triangular matrix filled with ones.
 * @param [in] upper_strict Pointer to strict upper-triangular matrix filled
 * with ones
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
template <typename InputT>
__aicore__ inline void run_block_scan(GM_ADDR input_vec, GM_ADDR lower,
                                      GM_ADDR upper_strict, GM_ADDR output_vec,
                                      GM_ADDR tiling_gm) {
  BlockScanTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t matmul_size = tiling.matmul_size;

  if ASCEND_IS_AIC {
    KernelBlockScan<InputT> op_cube(vec_len, matmul_size);
    op_cube.Init(input_vec, lower, upper_strict, output_vec);
    op_cube.Process();
  }
}

}  // namespace tcuscan

/**
 * @brief Run the multi core inclusive block scan kernel with dtype fp16
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] lower Pointer to upper-triangular matrix filled with ones.
 * @param [in] upper_strict Pointer to strict upper-triangular matrix filled
 * with ones
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void block_scan_fp16(
    GM_ADDR input_vec, GM_ADDR lower, GM_ADDR upper_strict, GM_ADDR output_vec,
    GM_ADDR workspace, GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

  (void)workspace;
  tcuscan::run_block_scan<half>(input_vec, lower, upper_strict, output_vec,
                                tilingGm);
}

/**
 * @brief Call the `block_scan` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] input_vec Pointer to an input buffer.
 * @param [in] lower Pointer to an input buffer.
 * @param [in] upper_strict Pointer to an input buffer.
 * @param [in] output_vec Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" void launch_block_scan_fp16(uint32_t blockDim, void* stream,
                                       uint8_t* input_vec, uint8_t* lower,
                                       uint8_t* upper_strict,
                                       uint8_t* output_vec, uint8_t* workspace,
                                       uint8_t* tilingGm) {
  block_scan_fp16<<<blockDim, nullptr, stream>>>(
      input_vec, lower, upper_strict, output_vec, workspace, tilingGm);
}
