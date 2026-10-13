/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_topk.h
 * @brief Kernel implementing the top-k operation.
 */
#pragma once

#include <cmath>

#include "ascendc_kernel_operator.h"
#include "kernel_arithmetic_progression.h"
#include "kernel_copy.h"
#include "kernel_less_or_equal.h"
#include "kernel_split.h"
#include "tcuscan_utils.h"
#include "../tiling/tiling_topk.h"

namespace tcuscan {

/**
 * @brief Run the `topk` kernel.
 *
 * @param [in] x Pointer to input vector.
 * @param [in] k Number of output elements.
 * @param [in] y Pointer to output vector.
 * @param [in] indices Pointer to output indices vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] x_min Minimum value of the input vector.
 * @param [in] x_max Maximum value of the input vector.
 * @param [in] vec_len Size of the input vector.
 * @param [in] tile_size Size of the vector tile.
 */
template <typename DataT>
__aicore__ inline void run_topk(GM_ADDR x, uint32_t k, GM_ADDR y,
                                GM_ADDR indices, GM_ADDR workspace, DataT x_min,
                                DataT x_max, uint32_t vec_len,
                                uint32_t tile_size) {
  const uint32_t mask_size =
      scalar::AlignUp(vec_len * sizeof(uint8_t), GM_ALIGNMENT);
  const uint32_t split_input_size =
      scalar::AlignUp(vec_len * sizeof(DataT), GM_ALIGNMENT);
  const uint32_t split_output_size = split_input_size;
  const uint32_t indices_ws_size =
      scalar::AlignUp(vec_len * sizeof(uint32_t) * 2, GM_ALIGNMENT);

  GM_ADDR const mask = workspace;
  GM_ADDR split_input = mask + mask_size;
  GM_ADDR split_output = split_input + split_input_size;
  GM_ADDR indices_in = split_output + split_output_size;
  GM_ADDR indices_out = indices_in + indices_ws_size / 2;
  GM_ADDR const split_ws = indices_out + indices_ws_size / 2;

  uint32_t offset = 0;
  uint32_t current_vec_len = vec_len;
  using _DataT = typename std::conditional<std::is_same_v<DataT, int16_t>,
                                           int16_t, float>::type;
  _DataT current_max = static_cast<_DataT>(x_max);
  _DataT current_min = static_cast<_DataT>(x_min);
  _DataT pivot;
  uint32_t current_k = k;

  bool first_iter = true;

  run_arithmetic_progression<int32_t, 0, 1, false /* ForceMixMode */>(
      indices_in, vec_len, tile_size);
  SyncAll<false /*isAIVOnly*/>();

  pivot = (current_min + current_max) / 2;
  uint16_t iters = 0;
  const int16_t MAX_ITERS = 10000;
  while (iters < MAX_ITERS) {
    iters++;
    uint32_t num_ones;
    if (first_iter) {
      run_less_or_equal<false, DataT>(x, mask, static_cast<DataT>(pivot),
                                      current_vec_len, tile_size);
      SyncAll<false /*isAIVOnly*/>();

      num_ones =
          run_split_ind_uint16(x, mask, indices_in, split_output, indices_out,
                               split_ws, current_vec_len, tile_size, true);
      first_iter = false;
    } else {
      run_less_or_equal<false, DataT>(split_input, mask,
                                      static_cast<DataT>(pivot),
                                      current_vec_len, tile_size);
      SyncAll<false /*isAIVOnly*/>();
      num_ones = run_split_ind_uint16(split_input, mask, indices_in,
                                      split_output, indices_out, split_ws,
                                      current_vec_len, tile_size, true);
    }

    SyncAll<false /*isAIVOnly*/>();

    const uint32_t L = current_vec_len - num_ones;

    if (current_k < L) {
      current_vec_len = L;
      current_min = pivot;
      pivot = (current_min + current_max) / 2;
    } else if (current_k == L) {
      current_min = current_max;
    } else if (current_k > L) {
      run_copy<DataT, false /* ForceMixMode */>(
          split_output, y + offset * sizeof(DataT), L, tile_size);
      run_copy<int32_t, false /* ForceMixMode */>(
          indices_out, indices + offset * sizeof(int32_t), L, tile_size);
      offset += L;
      current_max = pivot;

      split_output += L * sizeof(DataT);
      indices_out += L * sizeof(int32_t);

      current_k -= L;
      current_vec_len -= L;
      pivot = (current_min + current_max) / 2;
    }

    SyncAll<false /*isAIVOnly*/>();

    bool same_values;
    if constexpr (std::is_same_v<DataT, int16_t>) {
      same_values = current_min + 1 >= current_max;
    } else {
      same_values = fp16::next_after(current_min) >= current_max;
    }

    if (same_values) {
      run_copy<DataT, false /* ForceMixMode */>(
          split_output, y + offset * sizeof(DataT), current_k, tile_size);
      run_copy<int32_t, false /* ForceMixMode */>(
          indices_out, indices + offset * sizeof(int32_t), current_k,
          tile_size);

      return;
    }

    scalar::Swap(split_input, split_output);
    scalar::Swap(indices_in, indices_out);
  }
}

}  // namespace tcuscan
