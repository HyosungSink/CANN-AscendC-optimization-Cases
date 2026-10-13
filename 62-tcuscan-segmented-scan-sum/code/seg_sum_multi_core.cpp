/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file seg_sum_multi_core.cpp
 * @brief Entrypoint for segmented sum multi core kernel.
 */

#include "kernels/constants.h"
#include "kernels/kernel_pad.h"
#include "kernels/kernel_row_scan.h"
#include "kernels/kernel_seg_sum_vec_revert.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_seg_sum_multi_core.h"

using namespace AscendC;
using namespace tcuscan;

/**
 * @brief Run the `seg_sum_multi_core` kernel.
 *
 * @tparam T input data type
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] upper Pointer to an upper-triangular all-ones square matrix of
 * size \f$\textit{matmul_size}\f$.
 * @param [in] segm_ind_in Pointer to the segment indices vector.
 * @param [in] segm_offset_per_block Pointer to segment index offset per block.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] workspace Pointer to a memory region used as workspace.
 * @param [in] vec_len Input vector length.
 * @param [in] num_segments Number of segments.
 * @param [in] tile_len Tile length for matrix operations (matmul size).
 * @param [in] block_len Block length.
 */
template <typename T>
__aicore__ inline void run_seg_sum_multi_core(
    GM_ADDR vec_in, GM_ADDR upper, GM_ADDR segm_ind_in,
    GM_ADDR segm_offset_per_block, GM_ADDR vec_out, GM_ADDR workspace,
    uint32_t vec_len, uint32_t num_segments, uint32_t tile_len,
    uint32_t block_len) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<T>;

  const uint32_t align_size = tile_len * tile_len;
  const uint32_t padded_vec_len = scalar::AlignUp(vec_len, align_size);
  const uint32_t pad_size = padded_vec_len * sizeof(T);

  GM_ADDR const padded_input = workspace;
  GM_ADDR const spec_block_scan = workspace + pad_size;

  run_pad_kernel<T, false>(vec_in, padded_input, vec_len, align_size);

  sync::SyncGroup<sync::GroupSyncDirection::FULL>();
  AscendC::SyncAll<false /*isAIVOnly*/>();

  if ASCEND_IS_AIC {
    KernelRowScan<T> op_cube(tile_len, tile_len, padded_vec_len);
    op_cube.Init(padded_input, upper, spec_block_scan);
    op_cube.Process();
  }

  sync::SyncGroup<sync::GroupSyncDirection::FULL>();
  AscendC::SyncAll<false /*isAIVOnly*/>();
  AscendC::PipeBarrier<PIPE_ALL>();

  if ASCEND_IS_AIV {
    const uint32_t num_blocks = AscendC::GetBlockNum();

    // Use only 1 AIV core
    if (GetBlockIdx() % 2 == 1) {
      return;
    }

    // id is the id of each AI Core (2 AIVs and 1 AIC core)
    const auto id = GetBlockIdx() / GetTaskRation();
    int32_t segm_ind_offset =
        scalar::GetGMValue<int32_t>(segm_offset_per_block, id, num_blocks + 1);
    const int32_t next_offset = scalar::GetGMValue<int32_t>(
        segm_offset_per_block, id + 1, num_blocks + 1);
    const int32_t num_segments_per_block = next_offset - segm_ind_offset;

    // The boundaries of each segment must overlap
    if (id > 0) {
      segm_ind_offset--;
    }

    // Each AI Core group is responsible (offsets) starting from `block_len`
    const uint32_t block_vec_offset = id * block_len;
    if (block_vec_offset >= padded_vec_len) {
      return;
    }
    const bool is_overflow_block = block_vec_offset + block_len > vec_len;
    if (is_overflow_block) {
      block_len = vec_len - block_vec_offset;
    }

    KernelSegSumVecRevert<OutputT, false, true> op(
        block_len, num_segments_per_block, tile_len, block_vec_offset);
    op.Init(spec_block_scan, segm_ind_in + segm_ind_offset * sizeof(int32_t),
            vec_out + segm_ind_offset * sizeof(OutputT));
    op.Process();
  }
}

/**
 * @brief Run the `seg_sum_multi_core` kernel with half/float16 dtype.
 *
 * The segment indices format follows the scipy Compressed Sparse Row Matrix
 * convention
 * (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] indptr Pointer to the segment indices vector.
 * @param [in] segment_offsets Pointer to the segment offset per block.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void seg_sum_multi_core_fp16(
    GM_ADDR vec_in, GM_ADDR indptr, GM_ADDR segment_offsets, GM_ADDR vec_out,
    GM_ADDR workspace, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::SegSumMultiCoreTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t num_segments = tiling.num_segments;
  const uint32_t matmul_size = tiling.tile_len;
  const uint32_t block_len = tiling.block_len;

  GM_ADDR const lower = load_tril_matrix<half>(matmul_size);

  run_seg_sum_multi_core<half>(vec_in, lower, indptr, segment_offsets, vec_out,
                               workspace, vec_len, num_segments, matmul_size,
                               block_len);
}

/**
 * @brief Run the `seg_sum_multi_core` kernel with int8 dtype.
 *
 * The segment indices format follows the scipy Compressed Sparse Row Matrix
 * convention
 * (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] indptr Pointer to the segment indices vector.
 * @param [in] segment_offsets Pointer to the segment offset per block.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void seg_sum_multi_core_int8(
    GM_ADDR vec_in, GM_ADDR indptr, GM_ADDR segment_offsets, GM_ADDR vec_out,
    GM_ADDR workspace, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::SegSumMultiCoreTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t num_segments = tiling.num_segments;
  const uint32_t matmul_size = tiling.tile_len;
  const uint32_t block_len = tiling.block_len;

  GM_ADDR const lower = load_tril_matrix<int8_t>(matmul_size);

  run_seg_sum_multi_core<int8_t>(vec_in, lower, indptr, segment_offsets,
                                 vec_out, workspace, vec_len, num_segments,
                                 matmul_size, block_len);
}

/**
 * @brief Call the `seg_sum_multi_core` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] indptr Pointer to an input buffer.
 * @param [in] segment_offsets Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_seg_sum_multi_core_fp16(uint32_t blockDim, void* stream,
                                               uint8_t* vec_in, uint8_t* indptr,
                                               uint8_t* segment_offsets,
                                               uint8_t* vec_out,
                                               uint8_t* workspace,
                                               uint8_t* tiling_gm) {
  seg_sum_multi_core_fp16<<<blockDim, nullptr, stream>>>(
      vec_in, indptr, segment_offsets, vec_out, workspace, tiling_gm);
}

/**
 * @brief Call the `seg_sum_multi_core` kernel for INT8 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] indptr Pointer to an input buffer.
 * @param [in] segment_offsets Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_seg_sum_multi_core_int8(uint32_t blockDim, void* stream,
                                               uint8_t* vec_in, uint8_t* indptr,
                                               uint8_t* segment_offsets,
                                               uint8_t* vec_out,
                                               uint8_t* workspace,
                                               uint8_t* tiling_gm) {
  seg_sum_multi_core_int8<<<blockDim, nullptr, stream>>>(
      vec_in, indptr, segment_offsets, vec_out, workspace, tiling_gm);
}
