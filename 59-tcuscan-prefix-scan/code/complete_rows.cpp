/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file complete_rows.cpp
 * @brief Kernel implementing a multi-core AIV complete rows phase of MCSCAN.
 */

#include "kernels/kernel_complete_rows.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_complete_rows.h"

namespace tcuscan {

/**
 * @brief Run the multi core vector reduce tiles kernel.
 *
 * @tparam InputT Input data type.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] input_sums Pointer to vector with partial sums
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
template <typename InputT>
__aicore__ inline void run_complete_rows(GM_ADDR input_vec, GM_ADDR input_sums,
                                         GM_ADDR output_vec,
                                         GM_ADDR tiling_gm) {
  tcuscan::CompleteRowsTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t tile_width = tiling.tile_width;
  const uint32_t tile_height = tiling.tile_height;

  if ASCEND_IS_AIV {
    KernelCompleteRows<InputT> op_reduce(tile_width, tile_height, vec_len);
    op_reduce.Init(input_vec, input_sums, output_vec);
    op_reduce.Process();
  }
}

}  // namespace tcuscan

/**
 * @brief Run the multi core vector reduce tiles kernel with dtype fp32.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] input_sums Pointer to vector with partial sums
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void complete_rows_fp32(GM_ADDR input_vec,
                                                         GM_ADDR input_sums,
                                                         GM_ADDR output_vec,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  (void)workspace;
  tcuscan::run_complete_rows<float>(input_vec, input_sums, output_vec,
                                    tilingGm);
}

/**
 * @brief Run the multi core vector reduce tiles kernel with dtype int32.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] input_sums Pointer to vector with partial sums
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void complete_rows_int32(GM_ADDR input_vec,
                                                          GM_ADDR input_sums,
                                                          GM_ADDR output_vec,
                                                          GM_ADDR workspace,
                                                          GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  (void)workspace;
  tcuscan::run_complete_rows<int32_t>(input_vec, input_sums, output_vec,
                                      tilingGm);
}

/**
 * @brief Call the `complete_rows` kernel for FP32 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] input_vec Pointer to an input buffer.
 * @param [in] input_sums Pointer to an input buffer.
 * @param [in] output_vec Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" void launch_complete_rows_fp32(
    uint32_t blockDim, void* stream, uint8_t* input_vec, uint8_t* input_sums,
    uint8_t* output_vec, uint8_t* workspace, uint8_t* tilingGm) {
  complete_rows_fp32<<<blockDim, nullptr, stream>>>(
      input_vec, input_sums, output_vec, workspace, tilingGm);
}

/**
 * @brief Call the `complete_rows` kernel for INT32 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] input_vec Pointer to an input buffer.
 * @param [in] input_sums Pointer to an input buffer.
 * @param [in] output_vec Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" void launch_complete_rows_int32(
    uint32_t blockDim, void* stream, uint8_t* input_vec, uint8_t* input_sums,
    uint8_t* output_vec, uint8_t* workspace, uint8_t* tilingGm) {
  complete_rows_int32<<<blockDim, nullptr, stream>>>(
      input_vec, input_sums, output_vec, workspace, tilingGm);
}
