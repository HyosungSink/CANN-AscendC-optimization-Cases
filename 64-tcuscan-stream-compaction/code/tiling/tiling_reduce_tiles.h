

/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tiling_reduce_tiles.h
 * @brief Tiling structure for AIV reduce tiles kernel.
 */

#pragma once

#include <cstdint>
namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Tiling struct of `reduce_tiles` kernel.
 */
struct ReduceTilesTiling {
  /// @brief Number of elements
  uint32_t num_elems;
  /// @brief Length of tile
  uint32_t tile_len;
};
#pragma pack(pop)
}  // namespace tcuscan
