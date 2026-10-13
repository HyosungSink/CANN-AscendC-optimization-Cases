

/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tiling_complete_rows.h
 * @brief Tiling structure for AIV complete rows kernel.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Tiling struct of `complete_rows` kernel.
 */
struct CompleteRowsTiling {
  /// @brief Number of elements
  uint32_t num_elems;
  /// @brief Tile width
  uint32_t tile_width;
  /// @brief Tile height
  uint32_t tile_height;
};
#pragma pack(pop)

}  // namespace tcuscan
