/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tiling_count_if.h
 * @brief Tiling structure for Vector count_if kernel operation.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Vector count_if kernel tiling parameter structure.
 */
struct CountIfTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements.
  uint32_t vec_len;
  /// @brief Tiling length.
  uint32_t tile_len;
  /// @brief Input pivot.
  float pivot;
  /// @brief Comparison mode as in \c AscendC::CompareScalar.
  uint8_t compare_mode;
};
#pragma pack(pop)

}  // namespace tcuscan
