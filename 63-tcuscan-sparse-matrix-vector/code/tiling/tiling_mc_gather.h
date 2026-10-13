/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_mc_gather.h
 * @brief Tiling structure for Vector mcgather kernel operation.
 */

#pragma once

#include <cstdint>
namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Vector XXX kernel tiling parameter structure.
 */
struct McGatherTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements.
  uint32_t val_len;
  /// @brief Total number of input indices.
  uint32_t idx_len;
  /// @brief Tiling length.
  uint32_t tile_len;
};
#pragma pack(pop)
}  // namespace tcuscan
