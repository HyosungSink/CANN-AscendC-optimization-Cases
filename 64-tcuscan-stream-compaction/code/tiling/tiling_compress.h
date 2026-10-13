/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_compress.h
 * @brief Tiling structure for compress kernel.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Compress tiling parameter structure.
 */
struct CompressTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements.
  uint32_t vec_len;
  /// @brief Tile length.
  uint32_t tile_len;
};
#pragma pack(pop)

}  // namespace tcuscan
