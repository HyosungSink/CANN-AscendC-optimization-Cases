/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_gather_spmv.h
 * @brief Tiling structure for Vector gather_spmv kernel operation.
 */

#pragma once

#include <cstdint>
namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Vector gather_spmv kernel tiling parameter structure.
 */
struct GatherSpmvTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements.
  uint32_t val_len;
  /// @brief Total number of input indices.
  uint32_t idx_len;
  /// @brief Tiling length.
  uint32_t tile_len;
  /// @brief Scaling factor applied to the gathered values.
  float alpha;
};
#pragma pack(pop)
}  // namespace tcuscan
