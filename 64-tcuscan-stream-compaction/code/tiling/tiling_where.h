/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * @file tiling_where.h
 * @brief Tiling structure for Vector `where` kernel operation.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Vector where kernel tiling parameter structure.
 */
struct WhereTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements.
  uint32_t vec_len;
  /// @brief Tiling length.
  uint32_t tile_len;
  /// @brief Pivot for comparison.
  float pivot;
};
#pragma pack(pop)

}  // namespace tcuscan
