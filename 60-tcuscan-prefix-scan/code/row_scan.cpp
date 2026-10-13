/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file row_scan.cpp
 * @brief Kernel implementing a multi-core inclusive row scan.
 */

#include "kernels/constants.h"
#include "kernels/kernel_scan_multi_core.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_row_scan.h"

namespace tcuscan {

/**
 * @brief Run the multi core inclusive row scan kernel.
 *
 * @tparam InputT Input data type.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
template <typename InputT>
__aicore__ inline void run_row_scan(GM_ADDR input_vec, GM_ADDR output_vec,
                                    GM_ADDR tiling_gm) {
  tcuscan::RowScanTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t matmul_size = tiling.S;

  GM_ADDR const lower = load_tril_matrix<InputT>(matmul_size);

  if ASCEND_IS_AIC {
    KernelRowScan<InputT> op_cube(matmul_size, matmul_size, vec_len);
    op_cube.Init(input_vec, lower, output_vec);
    op_cube.Process();
  }
}

}  // namespace tcuscan

/**
 * @brief Run the multi core inclusive row scan kernel with dtype fp16
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void row_scan_fp16(GM_ADDR input_vec,
                                                    GM_ADDR output_vec,
                                                    GM_ADDR workspace,
                                                    GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

  (void)workspace;
  tcuscan::run_row_scan<half>(input_vec, output_vec, tilingGm);
}

/**
 * @brief Call the `row_scan` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] input_vec Pointer to an input buffer.
 * @param [in] output_vec Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" void launch_row_scan_fp16(uint32_t blockDim, void* stream,
                                     uint8_t* input_vec, uint8_t* output_vec,
                                     uint8_t* workspace, uint8_t* tilingGm) {
  row_scan_fp16<<<blockDim, nullptr, stream>>>(input_vec, output_vec, workspace,
                                               tilingGm);
}
