/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tiling_complete_blocks.h
 * @brief Tiling structure for AIV complete blocks kernel.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Tiling struct of `complete_blocks` kernel.
 */
struct CompleteBlocksTiling {
  /// @brief Number of elements
  uint32_t num_elems;
  /// @brief Number of blocks that are prefix-summed.
  uint32_t num_blocks;
  /// @brief Length of kernel tile.
  uint32_t tile_len;
};
#pragma pack(pop)

}  // namespace tcuscan
