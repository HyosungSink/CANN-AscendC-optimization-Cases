/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_seg_scan_vec_single_core.h
 * @brief Tiling structure for segmented scan (vector-only) kernel operation.
 */

#pragma once

#include <cstdint>
namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Vector-only segmented scan kernel tiling parameter structure.
 */
struct SegScanVecSingleCoreTiling {
  /// @brief Total number of input elements.
  uint32_t num_elems;
  /// @brief Tiling length.
  uint32_t tile_len;
};
#pragma pack(pop)
}  // namespace tcuscan
