/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025 All rights reserved.
 *
 * @file tiling_seg_sum_single_core.h
 * @brief Tiling structure for segmented sum single core.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief `seg_sum_single_core` kernel tiling parameter structure.
 */
struct SegSumSingleCoreTiling {
  /// @brief Total number of input elements.
  uint32_t vec_len;
  /// @brief Total number of segments.
  uint32_t num_segments;
  /// @brief Tiling length.
  uint32_t tile_len;
};

#pragma pack(pop)

}  // namespace tcuscan
