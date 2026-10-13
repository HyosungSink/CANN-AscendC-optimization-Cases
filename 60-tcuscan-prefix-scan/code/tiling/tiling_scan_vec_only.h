/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_scan_vec_only.h
 * @brief Tiling structure for vector-only scan.
 */

#pragma once

#include <cstdint>
namespace tcuscan {

#pragma pack(push, 8)

/**
 * @brief Vector-only scan tiling parameter structure.
 */
struct ScanVecOnlyTiling {
  /// @brief Total number of input elements.
  uint32_t num_elems;
  /// @brief Height of the tile: number of scans processed in parallel by
  /// cumsum.
  uint32_t tile_height;
  /// @brief Width of the tile: number of elements that cumsum scans.
  uint32_t tile_width;
};
#pragma pack(pop)
}  // namespace tcuscan
