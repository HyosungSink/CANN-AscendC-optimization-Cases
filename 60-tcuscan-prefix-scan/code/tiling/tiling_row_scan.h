

/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tiling_row_scan.h
 * @brief Tiling structure for MatMul Row Scan kernel.
 */

#pragma once

#include <cstdint>
namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief MatMul row scan tiling parameter structure. Matrix multiplication
 * C=AB, where B equals to U_s upper all-ones triangular matrix
 */
struct RowScanTiling {
  /// @brief Number of elements.
  uint32_t num_elems;
  /// @brief Block length of each row scan.
  uint32_t S;
};
#pragma pack(pop)
}  // namespace tcuscan
