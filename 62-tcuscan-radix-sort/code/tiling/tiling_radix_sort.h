/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_radix_sort.h
 * @brief Tiling structure for radix sort.
 */

#pragma once

#include <cstdint>
namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Radix sort tiling parameter structure.
 */
struct RadixSortTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements in a single vector.
  uint32_t vec_len;
  /// @brief Size of the matmul tile.
  uint32_t matmul_size;
  /// @brief Size of the vector tiles.
  uint32_t vec_tile_size;
};
#pragma pack(pop)
}  // namespace tcuscan
