/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_scan_multi_core.h
 * @brief Tiling structure for multi core scan.
 */

#pragma once

#include <cstdint>
namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Multi core scan tiling parameter structure.
 */
struct MultiCoreScanTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements.
  uint32_t vec_len;
  /// @brief Size of the matmul tile used in the cube part.
  uint32_t matmul_size;
  /// @brief L2 cache size between cores and global memory.
  uint64_t l2_cache_size;
};
#pragma pack(pop)
}  // namespace tcuscan
