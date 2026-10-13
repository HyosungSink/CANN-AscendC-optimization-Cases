/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file spmv_v2_multi_cube.cpp
 * @brief Entrypoint for the multi-cube SpMV kernel.
 */

#include "kernels/constants.h"
#include "kernels/kernel_block_scan.h"
#include "kernels/kernel_csr_gather.h"
#include "kernels/kernel_scale.h"
#include "kernels/kernel_seg_sum_cube_revert.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_spmv.h"

using namespace AscendC;
using namespace tcuscan;

/**
 * @brief Run the multi-cube SpMV (Sparse Matrix-Vector multiplication) kernel.
 *
 * This is a multi-cube variant of `run_spmv_v2` (see `spmv.cpp`). The CSR
 * gather step is identical, but the prefix-scan is computed with the multi-cube
 * `KernelBlockScan` (distributing the matrix tiles across all cube cores) and
 * the segment reduction is performed by `KernelSegSumCubeRevert` using
 * atomic-add writes, mirroring `run_seg_sum_multi_cube` (see
 * `seg_sum_multi_cube.cpp`). The cube and vector stages stream tile-by-tile
 * through the `SyncAfter`/`SyncBefore` group synchronization instead of a
 * single full barrier.
 *
 * @tparam T input data type
 *
 * @param [in] vec_in Pointer to the input vector (sparse matrix non-zero
 * values).
 * @param [in] cols_in Pointer to the CSR column indices array.
 * @param [in] upper_in Pointer to an upper-triangular all-ones square matrix of
 * size \f$\textit{tile\_len} \times \textit{tile\_len}\f$.
 * @param [in] lower_in Pointer to a strict lower-triangular all-ones square
 * matrix of size \f$\textit{tile\_len} \times \textit{tile\_len}\f$.
 * @param [in] segm_ind_in Pointer to the full CSR row-pointer array.
 * @param [in] x_in Pointer to the dense input vector.
 * @param [out] vec_out Pointer to the output vector.
 * @param [in,out] workspace Pointer to a memory region used as workspace.
 * @param [in] vec_len Input vector length (number of non-zeros).
 * @param [in] num_segments Number of segments.
 * @param [in] x_len Length of the dense input vector.
 * @param [in] tile_len Tile size used for the matrix multiplication step.
 * @param [in] block_len Block length assigned to each AI Core group.
 * @param [in] alpha Scaling factor of the SpMV product, applied by the
 * segment reduction as it writes each segment sum.
 * @param [in] beta Scaling factor applied in-place to @p vec_out before the
 * segment reduction accumulates on top of it. Following the BLAS convention,
 * `beta == 0` overwrites @p vec_out without reading it.
 */
template <typename T>
__aicore__ inline void run_spmv_v2_multi_cube(
    GM_ADDR vec_in, GM_ADDR cols_in, GM_ADDR upper_in, GM_ADDR lower_in,
    GM_ADDR segm_ind_in, GM_ADDR x_in, GM_ADDR vec_out, GM_ADDR workspace,
    uint32_t vec_len, uint32_t num_segments, uint32_t x_len, uint32_t tile_len,
    uint32_t block_len, float alpha, float beta) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<T>;

  const uint32_t align_size = tile_len * tile_len;
  const uint32_t padded_vec_len = scalar::AlignUp(vec_len, align_size);
  const uint32_t pad_size = padded_vec_len * sizeof(T);

  GM_ADDR const csr_products_ws = workspace;
  GM_ADDR const spec_block_scan_ws = workspace + pad_size;

  // `y = beta * y` pre-pass. It must complete before `KernelSegSumCubeRevert`
  // atomically accumulates `alpha * A @ x` into `vec_out`; ordering is enforced
  // by the `SyncAll` before the block scan and the SyncGroup barriers between
  // `KernelBlockScan<..., SyncAfter=true>` and `KernelSegSumCubeRevert<...,
  // SyncBefore=true>`. No-op when `beta == 1`.
  run_scale_inplace<OutputT, false>(vec_out, num_segments,
                                    static_cast<OutputT>(beta));

  // Size the gather tile to the largest that fits in the vector core's Unified
  // Buffer. Per `KernelCSRGather::Init`, the UB footprint is
  // i.e. UB = x_len*sizeof(T) + tile*(BUFFER_NUM*(2*sizeof(T)+4) + 4), with
  // BUFFER_NUM == 2. Solve for the tile and floor it to the UB alignment.
  constexpr uint32_t kUbBudget = tcuscan::UB_SIZE_BYTES;
  constexpr uint32_t kTileByteCost =
      2 * (2 * sizeof(T) + sizeof(uint32_t)) + sizeof(uint32_t);
  const uint32_t x_bytes = x_len * sizeof(T);
  const uint32_t ub_bound_tile =
      x_bytes < kUbBudget ? (kUbBudget - x_bytes) / kTileByteCost : 0;
  // No point tiling larger than the whole padded vector; keep 32B UB alignment.
  const uint32_t csr_gather_tile_len = scalar::AlignDown<uint32_t>(
      scalar::Min<uint32_t>(ub_bound_tile, padded_vec_len),
      UB_ALIGNMENT / sizeof(T));
  run_csr_gather<T, false>(vec_in, cols_in, x_in, csr_products_ws, vec_len,
                           x_len, csr_gather_tile_len);

  sync::SyncGroup<sync::GroupSyncDirection::FULL>();
  AscendC::SyncAll<false /*isAIVOnly*/>();

  if ASCEND_IS_AIC {
    KernelBlockScan<T, true /* SyncAfter */> op_cube(padded_vec_len, tile_len);
    op_cube.Init(csr_products_ws, upper_in, lower_in, spec_block_scan_ws);
    op_cube.Process();
  }

  if ASCEND_IS_AIV {
    // id is the id of each AI Core group (2 AIVs and 1 AIC core)
    const auto id = GetBlockIdx() / GetTaskRation();

    // Fused searchsorted: each group derives its own two per-block segment
    // offsets by binary-searching the full indptr for its block boundaries
    // `sstart[id] = min(id * block_len, vec_len)`
    const uint32_t sstart_id = scalar::Min<uint32_t>(id * block_len, vec_len);
    const uint32_t sstart_next =
        scalar::Min<uint32_t>((id + 1) * block_len, vec_len);
    int32_t segm_ind_offset =
        static_cast<int32_t>(scalar::LowerBoundGM<int32_t>(
            segm_ind_in, num_segments + 1, sstart_id));
    const int32_t next_offset =
        static_cast<int32_t>(scalar::LowerBoundGM<int32_t>(
            segm_ind_in, num_segments + 1, sstart_next));
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

    KernelSegSumCubeRevert<OutputT, true /* SyncBefore */,
                           true /* UseAtomicWrite */>
        op(block_len, num_segments_per_block, tile_len, block_vec_offset,
           static_cast<OutputT>(alpha));
    op.Init(spec_block_scan_ws,
            segm_ind_in + (segm_ind_offset + 1) * sizeof(int32_t),
            vec_out + segm_ind_offset * sizeof(OutputT));
    op.Process();
  }
}

/**
 * @brief Run the multi-cube SpMV kernel with half/float16 dtype.
 *
 * The segment indices format follows the scipy Compressed Sparse Row Matrix
 * convention
 * (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).
 *
 * @param [in] vec_in Pointer to the sparse matrix non-zero values.
 * @param [in] cols_in Pointer to the CSR column indices array.
 * @param [in] upper Pointer to an upper-triangular all-ones matrix.
 * @param [in] lower Pointer to a strict lower-triangular all-ones matrix.
 * @param [in] indptr Pointer to the full CSR row-pointer array (`indptr`,
 * including the leading zero).
 * @param [in] x_in Pointer to the dense input vector.
 * @param [out] vec_out Pointer to the output vector.
 * @param [in,out] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void spmv_v2_multi_cube_fp16(
    GM_ADDR vec_in, GM_ADDR cols_in, GM_ADDR upper, GM_ADDR lower,
    GM_ADDR indptr, GM_ADDR x_in, GM_ADDR vec_out, GM_ADDR workspace,
    GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::SpMVTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.nnz;
  const uint32_t num_segments = tiling.num_segments;
  const uint32_t x_len = tiling.x_len;
  const uint32_t tile_len = tiling.tile_len;
  const uint32_t block_len = tiling.block_len;
  const float alpha = tiling.alpha;
  const float beta = tiling.beta;

  run_spmv_v2_multi_cube<half>(vec_in, cols_in, upper, lower, indptr, x_in,
                               vec_out, workspace, vec_len, num_segments, x_len,
                               tile_len, block_len, alpha, beta);
}

/**
 * @brief Call the `spmv_v2_multi_cube` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] cols_in Pointer to an input buffer.
 * @param [in] upper Pointer to an input buffer.
 * @param [in] lower Pointer to an input buffer.
 * @param [in] indptr Pointer to an input buffer.
 * @param [in] x_in Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_spmv_v2_multi_cube_fp16(
    uint32_t blockDim, void* stream, uint8_t* vec_in, uint8_t* cols_in,
    uint8_t* upper, uint8_t* lower, uint8_t* indptr, uint8_t* x_in,
    uint8_t* vec_out, uint8_t* workspace, uint8_t* tiling_gm) {
  spmv_v2_multi_cube_fp16<<<blockDim, nullptr, stream>>>(
      vec_in, cols_in, upper, lower, indptr, x_in, vec_out, workspace,
      tiling_gm);
}
