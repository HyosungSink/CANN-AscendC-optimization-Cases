/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file radix_sort.cpp
 * @brief Kernel implementing a radix sort operation for fp16 / int16.
 */

#include "kernels/constants.h"
#include "kernels/kernel_arithmetic_progression.h"
#include "kernels/kernel_radix_enc.h"
#include "kernels/kernel_single_radix.h"
#include "kernels/kernel_split.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_radix_sort.h"

__aicore__ inline void _radix_sort_iter(GM_ADDR in, GM_ADDR radices_ws,
                                        GM_ADDR out, GM_ADDR indices_in,
                                        GM_ADDR indices_out, GM_ADDR split_ws,
                                        uint32_t in_len, uint32_t vec_tile_len,
                                        uint16_t bit_idx, bool zeros_first) {
  tcuscan::run_single_radix<false>(in, radices_ws, in_len, vec_tile_len,
                                   bit_idx);
  SyncAll<false /*isAIVOnly*/>();
  tcuscan::run_split_ind_uint16(in, radices_ws, indices_in, out, indices_out,
                                split_ws, in_len, vec_tile_len, zeros_first);
}

/**
 * @brief Run the `radix_sort_fp16` kernel.
 *
 * @param [in] in Pointer to input vector.
 * @param [in] out Pointer to output vector.
 * @param [in] indices Pointer to output indices vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void radix_sort_fp16(GM_ADDR in, GM_ADDR out,
                                                      GM_ADDR indices,
                                                      GM_ADDR workspace,
                                                      GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  /// Indicates the order of sorting
  constexpr bool descending = false;
  tcuscan::RadixSortTiling tiling_data;
  GetTilingData(&tiling_data, tiling);

  // Arrays in workspace have to be aligned to their data type size. Therefore
  // we align the size of the radices array to 4 bytes, so that the
  // workspace that comes after starts at a valid address for int32_t.
  const uint32_t vec_len = tiling_data.vec_len;
  const uint32_t vec_tile_len = tiling_data.vec_tile_size;
  const uint32_t radices_size =
      scalar::AlignUp(vec_len * sizeof(uint8_t), sizeof(int32_t));
  const uint32_t split_ws_size = tcuscan::split::get_workspace_size();
  const uint32_t output_size = vec_len * sizeof(half);

  GM_ADDR const radices = workspace;
  GM_ADDR const split_workspace = radices + radices_size;
  GM_ADDR const out2 = split_workspace + split_ws_size;
  GM_ADDR const indices_ws = out2 + output_size;

  tcuscan::run_arithmetic_progression<int32_t, 0, 1, false /* ForceMixMode */>(
      indices, vec_len, tiling_data.vec_tile_size);
  SyncAll<false /*isAIVOnly*/>();

  tcuscan::run_radix_encode<false>(in, out, vec_len, vec_tile_len);

  bool are_zeros_first = !descending;

  GM_ADDR iter_in = out;
  GM_ADDR iter_out = out2;
  GM_ADDR indices_in = indices;
  GM_ADDR indices_out = indices_ws;
  for (int i = 0; i < 16; i++) {
    SyncAll<true /*isAIVOnly*/>();

    const bool is_last_iter = i == 15;
    if (is_last_iter) {
      are_zeros_first = !are_zeros_first;
    }
    _radix_sort_iter(iter_in, radices, iter_out, indices_in, indices_out,
                     split_workspace, vec_len, vec_tile_len, i,
                     are_zeros_first);
    GM_ADDR tmp = iter_in;
    iter_in = iter_out;
    iter_out = tmp;

    tmp = indices_in;
    indices_in = indices_out;
    indices_out = tmp;
  }

  SyncAll<true /*isAIVOnly*/>();

  // Encode again to obtain the initial values (enc(enc(x)) = x)
  tcuscan::run_radix_encode<false>(out, out, vec_len, vec_tile_len);
}

/**
 * @brief Run the `radix_sort_int16` kernel.
 *
 * @param [in] in Pointer to input vector.
 * @param [in] out Pointer to output vector.
 * @param [in] indices Pointer to output indices vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void radix_sort_int16(GM_ADDR in, GM_ADDR out,
                                                       GM_ADDR indices,
                                                       GM_ADDR workspace,
                                                       GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  /// Indicates the order of sorting
  constexpr bool descending = false;
  tcuscan::RadixSortTiling tiling_data;
  GetTilingData(&tiling_data, tiling);

  // Arrays in workspace have to be aligned to their data type size. Therefore
  // we align the size of the radices array to 4 bytes, so that the
  // workspace that comes after starts at a valid address for int32_t.
  const uint32_t vec_len = tiling_data.vec_len;
  const uint32_t vec_tile_len = tiling_data.vec_tile_size;
  const uint32_t radices_size =
      scalar::AlignUp(vec_len * sizeof(uint8_t), sizeof(int32_t));
  const uint32_t split_ws_size = tcuscan::split::get_workspace_size();
  const uint32_t output_size = vec_len * sizeof(uint16_t);

  GM_ADDR const radices = workspace;
  GM_ADDR const split_workspace = radices + radices_size;
  GM_ADDR const out2 = split_workspace + split_ws_size;
  GM_ADDR const indices_ws = out2 + output_size;

  tcuscan::run_arithmetic_progression<int32_t, 0, 1, false /* ForceMixMode */>(
      indices, vec_len, tiling_data.vec_tile_size);
  SyncAll<false /*isAIVOnly*/>();

  bool are_zeros_first = !descending;
  _radix_sort_iter(in, radices, out2, indices, indices_ws, split_workspace,
                   vec_len, vec_tile_len, 0, are_zeros_first);

  SyncAll<true /*isAIVOnly*/>();

  GM_ADDR iter_in = out2;
  GM_ADDR iter_out = out;
  GM_ADDR indices_in = indices_ws;
  GM_ADDR indices_out = indices;
  for (int i = 1; i < 16; i++) {
    const bool is_last_iter = i == 15;
    if (is_last_iter) {
      are_zeros_first = !are_zeros_first;
    }
    _radix_sort_iter(iter_in, radices, iter_out, indices_in, indices_out,
                     split_workspace, vec_len, vec_tile_len, i,
                     are_zeros_first);

    SyncAll<true /*isAIVOnly*/>();

    GM_ADDR tmp = iter_in;
    iter_in = iter_out;
    iter_out = tmp;

    tmp = indices_in;
    indices_in = indices_out;
    indices_out = tmp;
  }
}

/**
 * @brief Call the `radix_sort` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] in Pointer to an input buffer.
 * @param [in] out Pointer to an output buffer.
 * @param [in] indices Pointer to an input buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to the tiling buffer.
 */
extern "C" void launch_radix_sort_fp16(uint32_t blockDim, void* stream,
                                       uint8_t* in, uint8_t* out,
                                       uint8_t* indices, uint8_t* workspace,
                                       uint8_t* tiling) {
  radix_sort_fp16<<<blockDim, nullptr, stream>>>(in, out, indices, workspace,
                                                 tiling);
}

/**
 * @brief Call the `radix_sort` kernel for INT16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] in Pointer to an input buffer.
 * @param [in] out Pointer to an output buffer.
 * @param [in] indices Pointer to an input buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to the tiling buffer.
 */
extern "C" void launch_radix_sort_int16(uint32_t blockDim, void* stream,
                                        uint8_t* in, uint8_t* out,
                                        uint8_t* indices, uint8_t* workspace,
                                        uint8_t* tiling) {
  radix_sort_int16<<<blockDim, nullptr, stream>>>(in, out, indices, workspace,
                                                  tiling);
}
