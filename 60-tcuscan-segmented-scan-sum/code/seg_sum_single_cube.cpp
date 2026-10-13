/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file seg_sum_single_cube.cpp
 * @brief Entrypoint for segmented sum single core kernel operation.
 */

#include "kernels/constants.h"
#include "kernels/kernel_block_scan.h"
#include "kernels/kernel_pad.h"
#include "kernels/kernel_seg_sum_cube_revert.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_seg_sum_single_cube.h"

using namespace AscendC;
using namespace tcuscan;

/**
 * @brief Run the `seg_sum_single_cube` kernel.
 *
 * @tparam T input data type
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] upper Pointer to an upper-triangular all-ones square matrix of
 * size \f$\textit{matmul_size}\f$.
 * @param [in] lower Pointer to an lower-triangular all-ones square matrix of
 * size \f$\textit{matmul_size}\f$.
 * @param [in] segm_ind_in Pointer to the segment indices vector.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] workspace Pointer to a memory region used as workspace.
 * @param [in] vec_len Input vector length.
 * @param [in] num_segments Number of segments.
 * @param [in] tile_len Tile length.
 */
template <typename T>
__aicore__ inline void run_seg_sum_single_cube(
    GM_ADDR vec_in, GM_ADDR upper, GM_ADDR lower, GM_ADDR segm_ind_in,
    GM_ADDR vec_out, GM_ADDR workspace, uint32_t vec_len, uint32_t num_segments,
    uint32_t tile_len) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<T>;

  const uint32_t align_size = tile_len * tile_len;
  const uint32_t padded_vec_len = scalar::AlignUp(vec_len, align_size);
  const uint32_t pad_size = padded_vec_len * sizeof(T);

  GM_ADDR const padded_input = workspace;
  GM_ADDR const spec_block_scan = workspace + pad_size;

  run_pad_kernel<T, false>(vec_in, padded_input, vec_len, align_size);

  tcuscan::sync::SyncGroup<sync::GroupSyncDirection::FULL>();

  if ASCEND_IS_AIC {
    KernelBlockScan<T, true> op_cube(padded_vec_len, tile_len);
    op_cube.Init(padded_input, upper, lower, spec_block_scan);
    op_cube.Process();
  }

  if ASCEND_IS_AIV {
    KernelSegSumCubeRevert<OutputT, true> op(vec_len, num_segments, tile_len);
    op.Init(spec_block_scan, segm_ind_in, vec_out);
    op.Process();
  }
}

/**
 * @brief Run the `seg_sum_single_cube` kernel with input type `fp16`.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] upper Pointer to an upper-triangular matrix filled
 * with ones of size \f$\textit{matmul_size} \times \textit{matmul_size}\f$.
 * @param [in] lower Pointer to an upper-triangular matrix filled
 * with ones of size \f$\textit{matmul_size} \times \textit{matmul_size}\f$.
 * @param [in] segm_ind_in Pointer to the segment indices vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to a memory region used as workspace.
 * @param [in] tiling Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void seg_sum_single_cube_fp16(
    GM_ADDR vec_in, GM_ADDR upper, GM_ADDR lower, GM_ADDR segm_ind_in,
    GM_ADDR vec_out, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::SegSumSingleCubeTiling t;
  GetTilingData(&t, tiling);

  const uint32_t vec_len = t.vec_len;
  const uint32_t num_segments = t.num_segments;
  const uint32_t tile_len = t.tile_len;

  run_seg_sum_single_cube<half>(vec_in, upper, lower, segm_ind_in, vec_out,
                                workspace, vec_len, num_segments, tile_len);
}

/**
 * @brief Call the `seg_sum_single_cube` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] upper Pointer to an input buffer.
 * @param [in] lower Pointer to an input buffer.
 * @param [in] segm_ind_in Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to the tiling buffer.
 */
extern "C" void launch_seg_sum_single_cube_fp16(
    uint32_t blockDim, void* stream, uint8_t* vec_in, uint8_t* upper,
    uint8_t* lower, uint8_t* segm_ind_in, uint8_t* vec_out, uint8_t* workspace,
    uint8_t* tiling) {
  seg_sum_single_cube_fp16<<<blockDim, nullptr, stream>>>(
      vec_in, upper, lower, segm_ind_in, vec_out, workspace, tiling);
}
