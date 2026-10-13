/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tiling_seg_scan_mc_revert.h
 * @brief Tiling structure for AIV segmented scan revertion kernel operation.
 */

#pragma once

#include <cstdint>
namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief AIV segmented scan revertion kernel tiling parameter structure.
 */
struct SegScanMcRevertTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements.
  uint32_t num_elems;
  /// @brief Total number of segmented.
  uint32_t num_segments;
  /// @brief Tiling length.
  uint32_t tile_len;
};
#pragma pack(pop)
}  // namespace tcuscan
