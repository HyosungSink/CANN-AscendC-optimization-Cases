/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_scan_single_core.h
 * @brief Tiling structure for single core scan.
 */

#pragma once

#include <cstdint>
namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Single core scan tiling parameter structure.
 */
struct SingleCoreScanTiling {
  /// @brief Total number of input elements.
  uint32_t num_elems;
  /// @brief Size of the matmul tile used in the cube part.
  uint32_t matmul_size;
  /// @brief Starting sum for the scan operator. it can be either float or
  /// int32_t.
  union {
    int32_t int_value;
    float float_value;
  } running_sum;
};
#pragma pack(pop)
}  // namespace tcuscan
