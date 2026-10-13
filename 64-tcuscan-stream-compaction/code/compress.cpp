/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file compress.cpp
 * @brief Multi-core compress approach.
 */

#include "kernels/kernel_compress.h"
#include "kernels/kernel_compress_ind.h"
#include "kernels/kernel_where.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_compress.h"
#include "tiling/tiling_where.h"

using namespace AscendC;

/**
 * @brief Compress kernel for dtype fp16
 *
 * @param [in] x Input data vector
 * @param [in] mask Input mask vector
 * @param [in] z Output vector
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to tiling structure.
 */
extern "C" __global__ __aicore__ void compress_fp16(GM_ADDR x, GM_ADDR mask,
                                                    GM_ADDR z,
                                                    GM_ADDR workspace,
                                                    GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::CompressTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t tile_len = tiling.tile_len;

  tcuscan::run_compress<half>(x, mask, z, workspace, vec_len, tile_len);
}

/**
 * @brief Compress kernel for dtype fp32
 *
 * @param [in] x Input data vector
 * @param [in] mask Input mask vector
 * @param [in] z Output vector
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to tiling structure.
 */
extern "C" __global__ __aicore__ void compress_fp32(GM_ADDR x, GM_ADDR mask,
                                                    GM_ADDR z,
                                                    GM_ADDR workspace,
                                                    GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::CompressTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t tile_len = tiling.tile_len;

  tcuscan::run_compress<float>(x, mask, z, workspace, vec_len, tile_len);
}

/**
 * @brief Compress with indices kernel for dtype fp16
 *
 * @param vec_in Input data vector
 * @param indices_in Input indices vector
 * @param mask Input mask vector
 * @param [in] num_ones_per_block Point to number of mask ones per block.
 * @param vec_out Output data vector
 * @param indices_out Output indices vector
 * @param workspace Pointer to workspace.
 * @param tiling_gm Pointer to tiling structure.
 */
extern "C" __global__ __aicore__ void compress_ind_fp16(
    GM_ADDR vec_in, GM_ADDR indices_in, GM_ADDR mask,
    GM_ADDR num_ones_per_block, GM_ADDR vec_out, GM_ADDR indices_out,
    GM_ADDR workspace, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::CompressTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t tile_len = tiling.tile_len;

  tcuscan::run_compress_ind<half>(vec_in, indices_in, mask, num_ones_per_block,
                                  vec_out, indices_out, vec_len, tile_len);
}

/**
 * @brief Compress with indices kernel for dtype fp32
 *
 * @param [in] vec_in Input data vector
 * @param [in] indices_in Input indices vector
 * @param [in] mask Input mask vector
 * @param [in] num_ones_per_block Point to number of mask ones per block.
 * @param [in] vec_out Output data vector
 * @param [in] indices_out Output indices vector
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to tiling structure.
 */
extern "C" __global__ __aicore__ void compress_ind_fp32(
    GM_ADDR vec_in, GM_ADDR indices_in, GM_ADDR mask,
    GM_ADDR num_ones_per_block, GM_ADDR vec_out, GM_ADDR indices_out,
    GM_ADDR workspace, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::CompressTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t tile_len = tiling.tile_len;

  tcuscan::run_compress_ind<float>(vec_in, indices_in, mask, num_ones_per_block,
                                   vec_out, indices_out, vec_len, tile_len);
}

/**
 * @brief Compress with indices kernel (no input enumerate indices) for dtype
 * fp16.
 *
 * @param [in] vec_in Input data vector
 * @param [in] mask Input mask vector
 * @param [in] num_ones_per_block Point to number of mask ones per block.
 * @param [in] vec_out Output data vector
 * @param [in] indices_out Output indices vector
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to tiling structure.
 */
extern "C" __global__ __aicore__ void compress_ind_no_arange_fp16(
    GM_ADDR vec_in, GM_ADDR mask, GM_ADDR num_ones_per_block, GM_ADDR vec_out,
    GM_ADDR indices_out, GM_ADDR workspace, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::CompressTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t tile_len = tiling.tile_len;

  tcuscan::run_compress_ind_no_arange<half>(vec_in, mask, num_ones_per_block,
                                            vec_out, indices_out, vec_len,
                                            tile_len);
}

/**
 * @brief Compress with indices kernel (no input enumerate indices) for dtype
 * fp32.
 *
 * @param [in] vec_in Input data vector
 * @param [in] mask Input mask vector
 * @param [in] num_ones_per_block Point to number of mask ones per block.
 * @param [in] vec_out Output data vector
 * @param [in] indices_out Output indices vector
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to tiling structure.
 */
extern "C" __global__ __aicore__ void compress_ind_no_arange_fp32(
    GM_ADDR vec_in, GM_ADDR mask, GM_ADDR num_ones_per_block, GM_ADDR vec_out,
    GM_ADDR indices_out, GM_ADDR workspace, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::CompressTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t tile_len = tiling.tile_len;

  tcuscan::run_compress_ind_no_arange<float>(vec_in, mask, num_ones_per_block,
                                             vec_out, indices_out, vec_len,
                                             tile_len);
}

/**
 * @brief Compress kernel with number of block mask sums (dtype fp16).
 *
 * @param [in] x Input data vector
 * @param [in] mask Input mask vector
 * @param [in] num_ones_per_block Input number of ones of mask per block.
 * @param [in] z Output vector
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to tiling structure.
 */
extern "C" __global__ __aicore__ void compress_with_sums_fp16(
    GM_ADDR x, GM_ADDR mask, GM_ADDR num_ones_per_block, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::CompressTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t tile_len = tiling.tile_len;
  const uint32_t block_len = tile_len * tile_len / 2;

  tcuscan::run_compress_with_num_ones<half>(x, mask, num_ones_per_block, z,
                                            vec_len, block_len);
}

/**
 * @brief Compress kernel with number of block mask sums (dtype fp32).
 *
 * @param [in] x Input data vector
 * @param [in] mask Input mask vector
 * @param [in] num_ones_per_block Input number of ones of mask per block.
 * @param [in] z Output vector
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to tiling structure.
 */
extern "C" __global__ __aicore__ void compress_with_sums_fp32(
    GM_ADDR x, GM_ADDR mask, GM_ADDR num_ones_per_block, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::CompressTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t tile_len = tiling.tile_len;
  const uint32_t block_len = tile_len * tile_len / 2;

  tcuscan::run_compress_with_num_ones<float>(x, mask, num_ones_per_block, z,
                                             vec_len, block_len);
}

/**
 * @brief Run the `where` kernel with dtype fp16.
 *
 * @param [in] mask_in Pointer to the input vector.
 * @param [in] num_ones_per_block Input number of ones of mask per block.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling structure.
 */

extern "C" __global__ __aicore__ void where_fp16(GM_ADDR mask_in,
                                                 GM_ADDR num_ones_per_block,
                                                 GM_ADDR vec_out,
                                                 GM_ADDR workspace,
                                                 GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::WhereTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t tile_len = tiling.tile_len;

  tcuscan::run_where(mask_in, num_ones_per_block, vec_out, vec_len, tile_len);
}

/**
 * @brief Call the `compress` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] x Pointer to an input buffer.
 * @param [in] mask Pointer to an input buffer.
 * @param [in] z Pointer to an input buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_compress_fp16(uint32_t blockDim, void* stream,
                                     uint8_t* x, uint8_t* mask, uint8_t* z,
                                     uint8_t* workspace, uint8_t* tiling_gm) {
  compress_fp16<<<blockDim, nullptr, stream>>>(x, mask, z, workspace,
                                               tiling_gm);
}

/**
 * @brief Call the `compress` kernel for FP32 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] x Pointer to an input buffer.
 * @param [in] mask Pointer to an input buffer.
 * @param [in] z Pointer to an input buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_compress_fp32(uint32_t blockDim, void* stream,
                                     uint8_t* x, uint8_t* mask, uint8_t* z,
                                     uint8_t* workspace, uint8_t* tiling_gm) {
  compress_fp32<<<blockDim, nullptr, stream>>>(x, mask, z, workspace,
                                               tiling_gm);
}

/**
 * @brief Call the `compress_ind` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] indices_in Pointer to an input buffer.
 * @param [in] mask Pointer to an input buffer.
 * @param [in] num_ones_per_block Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] indices_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_compress_ind_fp16(
    uint32_t blockDim, void* stream, uint8_t* vec_in, uint8_t* indices_in,
    uint8_t* mask, uint8_t* num_ones_per_block, uint8_t* vec_out,
    uint8_t* indices_out, uint8_t* workspace, uint8_t* tiling_gm) {
  compress_ind_fp16<<<blockDim, nullptr, stream>>>(
      vec_in, indices_in, mask, num_ones_per_block, vec_out, indices_out,
      workspace, tiling_gm);
}

/**
 * @brief Call the `compress_ind` kernel for FP32 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] indices_in Pointer to an input buffer.
 * @param [in] mask Pointer to an input buffer.
 * @param [in] num_ones_per_block Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] indices_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_compress_ind_fp32(
    uint32_t blockDim, void* stream, uint8_t* vec_in, uint8_t* indices_in,
    uint8_t* mask, uint8_t* num_ones_per_block, uint8_t* vec_out,
    uint8_t* indices_out, uint8_t* workspace, uint8_t* tiling_gm) {
  compress_ind_fp32<<<blockDim, nullptr, stream>>>(
      vec_in, indices_in, mask, num_ones_per_block, vec_out, indices_out,
      workspace, tiling_gm);
}

/**
 * @brief Call the `compress_ind_no_arange` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] mask Pointer to an input buffer.
 * @param [in] num_ones_per_block Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] indices_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_compress_ind_no_arange_fp16(
    uint32_t blockDim, void* stream, uint8_t* vec_in, uint8_t* mask,
    uint8_t* num_ones_per_block, uint8_t* vec_out, uint8_t* indices_out,
    uint8_t* workspace, uint8_t* tiling_gm) {
  compress_ind_no_arange_fp16<<<blockDim, nullptr, stream>>>(
      vec_in, mask, num_ones_per_block, vec_out, indices_out, workspace,
      tiling_gm);
}

/**
 * @brief Call the `compress_ind_no_arange` kernel for FP32 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] mask Pointer to an input buffer.
 * @param [in] num_ones_per_block Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] indices_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_compress_ind_no_arange_fp32(
    uint32_t blockDim, void* stream, uint8_t* vec_in, uint8_t* mask,
    uint8_t* num_ones_per_block, uint8_t* vec_out, uint8_t* indices_out,
    uint8_t* workspace, uint8_t* tiling_gm) {
  compress_ind_no_arange_fp32<<<blockDim, nullptr, stream>>>(
      vec_in, mask, num_ones_per_block, vec_out, indices_out, workspace,
      tiling_gm);
}

/**
 * @brief Call the `compress_with_sums` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] x Pointer to an input buffer.
 * @param [in] mask Pointer to an input buffer.
 * @param [in] num_ones_per_block Pointer to an input buffer.
 * @param [in] z Pointer to an input buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_compress_with_sums_fp16(uint32_t blockDim, void* stream,
                                               uint8_t* x, uint8_t* mask,
                                               uint8_t* num_ones_per_block,
                                               uint8_t* z, uint8_t* workspace,
                                               uint8_t* tiling_gm) {
  compress_with_sums_fp16<<<blockDim, nullptr, stream>>>(
      x, mask, num_ones_per_block, z, workspace, tiling_gm);
}

/**
 * @brief Call the `compress_with_sums` kernel for FP32 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] x Pointer to an input buffer.
 * @param [in] mask Pointer to an input buffer.
 * @param [in] num_ones_per_block Pointer to an input buffer.
 * @param [in] z Pointer to an input buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_compress_with_sums_fp32(uint32_t blockDim, void* stream,
                                               uint8_t* x, uint8_t* mask,
                                               uint8_t* num_ones_per_block,
                                               uint8_t* z, uint8_t* workspace,
                                               uint8_t* tiling_gm) {
  compress_with_sums_fp32<<<blockDim, nullptr, stream>>>(
      x, mask, num_ones_per_block, z, workspace, tiling_gm);
}

/**
 * @brief Call the `where` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] mask_in Pointer to an input buffer.
 * @param [in] num_ones_per_block Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_where_fp16(uint32_t blockDim, void* stream,
                                  uint8_t* mask_in, uint8_t* num_ones_per_block,
                                  uint8_t* vec_out, uint8_t* workspace,
                                  uint8_t* tiling_gm) {
  where_fp16<<<blockDim, nullptr, stream>>>(mask_in, num_ones_per_block,
                                            vec_out, workspace, tiling_gm);
}
