

/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tiling_block_scan.h
 * @brief Tiling structure for block scan using multiple MatMuls.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief MatMul blocks scan tiling parameter structure.
 */
struct BlockScanTiling {
  /// @brief Number of elements.
  uint32_t num_elems;
  /// @brief Matmul size used. Each block has length `matmul_size *
  /// matmul_size`.
  uint32_t matmul_size;
};
#pragma pack(pop)

}  // namespace tcuscan
