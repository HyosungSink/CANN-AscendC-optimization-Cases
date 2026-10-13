/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_seg_scan_single_core.h
 * @brief Tiling structure for segmented scan single core.
 */

#pragma once

#include <cstdint>
namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Segmented scan single core tiling parameter structure.
 */
struct SegScanSingleCoreTiling {
  /// @brief Total number of input elements.
  uint32_t vec_len;
  /// @brief Matrix tile size.
  uint32_t matmul_size;
};
#pragma pack(pop)
}  // namespace tcuscan
