/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * @file tiling_greater_equal.h
 * @brief Tiling structure for greater-equal.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)

/**
 * @brief Greater-equal tiling parameter structure.
 */
struct GreaterEqualTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements in a single vector.
  uint32_t vec_len;
  /// @brief Size of the vector tiles.
  uint32_t tile_len;
  /// @brief Pivot for comparison.
  float pivot;
};
#pragma pack(pop)
}  // namespace tcuscan
