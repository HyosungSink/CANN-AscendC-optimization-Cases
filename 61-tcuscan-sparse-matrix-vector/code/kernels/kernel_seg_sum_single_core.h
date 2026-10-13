/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file kernel_seg_sum_single_core.h
 * @brief Kernel implementing a segmented sum single core operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "kernel_pad.h"
#include "kernel_row_scan.h"
#include "kernel_seg_sum_vec_revert.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {
/**
 * @brief Run the `seg_sum_single_core` kernel.
 *
 * @tparam T input data type
 * @tparam UseAtomicWrite If true, the output is written using atomic-add
 * semantics.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] upper Pointer to an upper-triangular all-ones square matrix of
 * size \f$\textit{matmul_size}\f$.
 * @param [in] segm_ind_in Pointer to the segment indices vector.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] workspace Pointer to a memory region used as workspace.
 * @param [in] vec_len Input vector length.
 * @param [in] num_segments Number of segments.
 * @param [in] tile_len Tile length.
 * @param [in] vec_start_offset Start offset of input data vector. Segment
 * values will be offset accordindly. Default value: `0`.
 */
template <typename T, bool UseAtomicWrite = false>
__aicore__ inline void run_seg_sum_single_core(
    GM_ADDR vec_in, GM_ADDR upper, GM_ADDR segm_ind_in, GM_ADDR vec_out,
    GM_ADDR workspace, uint32_t vec_len, uint32_t num_segments,
    uint32_t tile_len, uint32_t vec_start_offset = 0) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<T>;

  const uint32_t align_size = tile_len * tile_len;
  const uint32_t padded_vec_len = scalar::AlignUp(vec_len, align_size);
  const uint32_t pad_size = padded_vec_len * sizeof(T);

  GM_ADDR const padded_input = workspace;
  GM_ADDR const spec_block_scan = workspace + pad_size;

  run_pad_kernel<T, false>(vec_in, padded_input, vec_len, align_size);

  sync::SyncGroup<sync::GroupSyncDirection::FULL>();

  if ASCEND_IS_AIC {
    KernelRowScan<T, true> op_cube(tile_len, tile_len, padded_vec_len);
    op_cube.Init(padded_input, upper, spec_block_scan);
    op_cube.Process();
  }

  if ASCEND_IS_AIV {
    KernelSegSumVecRevert<OutputT, true, UseAtomicWrite> op(
        vec_len, num_segments, tile_len, vec_start_offset);
    op.Init(spec_block_scan, segm_ind_in, vec_out);
    op.Process();
  }
}

}  // namespace tcuscan
