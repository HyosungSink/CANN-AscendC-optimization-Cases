/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tri_inv_col_sweep.cpp
 * @brief Entrypoint for vector-only `tri_inv_col_sweep` kernel.
 */

#include "kernels/kernel_tri_inv_col_sweep.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_tri_inv_col_sweep.h"

/**
 * @brief Run the `tri_inv_col_sweep` kernel on dtype fp16/half.
 *
 * @param [in] vec_in Pointer to input vector.
 * @param [in] vec_out Pointer to output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to tiling vector.
 */
extern "C" __global__ __aicore__ void tri_inv_col_sweep_fp16(
    GM_ADDR vec_in, GM_ADDR vec_out, GM_ADDR workspace, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  (void)workspace;
  tcuscan::TriInvColumnSweepTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t num_elems = tiling.num_elems;
  const uint32_t matrix_size = tiling.matrix_size;
  const uint32_t num_out_tiles = tiling.num_out_tiles;

  switch (matrix_size) {
    case 16:
      tcuscan::tri_inv_col_sweep<half, 16>(vec_in, vec_out, num_elems,
                                           num_out_tiles);
      break;
    case 32:
      tcuscan::tri_inv_col_sweep<half, 32>(vec_in, vec_out, num_elems,
                                           num_out_tiles);
      break;
    case 64:
      tcuscan::tri_inv_col_sweep<half, 64>(vec_in, vec_out, num_elems,
                                           num_out_tiles);
      break;
    case 128:
      tcuscan::tri_inv_col_sweep<half, 128>(vec_in, vec_out, num_elems,
                                            num_out_tiles);
      break;
    default:
      static_assert(
          "tri_inv_col_sweep_fp16: Invalid input matrix size. Supported sizes "
          "16,32,64,128.");
  }
}

/**
 * @brief Run the `tri_inv_col_sweep` kernel on dtype float32.
 *
 * @param [in] vec_in Pointer to input vector.
 * @param [in] vec_out Pointer to output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to tiling vector.
 */
extern "C" __global__ __aicore__ void tri_inv_col_sweep_fp32(
    GM_ADDR vec_in, GM_ADDR vec_out, GM_ADDR workspace, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  (void)workspace;
  tcuscan::TriInvColumnSweepTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t num_elems = tiling.num_elems;
  const uint32_t matrix_size = tiling.matrix_size;
  const uint32_t num_out_tiles = tiling.num_out_tiles;

  switch (matrix_size) {
    case 16:
      tcuscan::tri_inv_col_sweep<float, 16>(vec_in, vec_out, num_elems,
                                            num_out_tiles);
      break;
    case 32:
      tcuscan::tri_inv_col_sweep<float, 32>(vec_in, vec_out, num_elems,
                                            num_out_tiles);
      break;
    case 64:
      tcuscan::tri_inv_col_sweep<float, 64>(vec_in, vec_out, num_elems,
                                            num_out_tiles);
      break;
    case 128:
      tcuscan::tri_inv_col_sweep<float, 128>(vec_in, vec_out, num_elems,
                                             num_out_tiles);
      break;
    default:
      static_assert(
          "tri_inv_col_sweep_fp32: Invalid input matrix size. Supported sizes "
          "16,32,64,128.");
  }
}

/**
 * @brief Call the `tri_inv_col_sweep` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_tri_inv_col_sweep_fp16(uint32_t blockDim, void* stream,
                                              uint8_t* vec_in, uint8_t* vec_out,
                                              uint8_t* workspace,
                                              uint8_t* tiling_gm) {
  tri_inv_col_sweep_fp16<<<blockDim, nullptr, stream>>>(vec_in, vec_out,
                                                        workspace, tiling_gm);
}

/**
 * @brief Call the `tri_inv_col_sweep` kernel for FP32 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_tri_inv_col_sweep_fp32(uint32_t blockDim, void* stream,
                                              uint8_t* vec_in, uint8_t* vec_out,
                                              uint8_t* workspace,
                                              uint8_t* tiling_gm) {
  tri_inv_col_sweep_fp32<<<blockDim, nullptr, stream>>>(vec_in, vec_out,
                                                        workspace, tiling_gm);
}
