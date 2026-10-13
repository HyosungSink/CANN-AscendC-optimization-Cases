/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_scan_batch.h
 * @brief Tiling structure for scan batch.
 */

#pragma once

#include <cstdint>
namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Scan batch tiling parameter structure.
 */
struct ScanBatchTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements in a single vector.
  uint32_t num_elems;
  /// @brief Number of vectors in a batch.
  uint32_t batch_size;
  /// @brief Size of the matmul tile used in the cube part.
  uint32_t matmul_size;
  /// @brief Vector to cube core ratio.
  uint32_t vec_cube_ratio;
};
#pragma pack(pop)
}  // namespace tcuscan
