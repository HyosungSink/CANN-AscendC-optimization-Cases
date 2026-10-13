/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_topk.h
 * @brief Tiling structure for topk.
 */

#pragma once

#include <cstdint>
namespace tcuscan {

/**
 * @brief TopK union type
 */
union TopKDtype {
  /// @brief The integer value.
  int32_t value_i32;
  /// @brief The float value.
  float value_fp32;
};

#pragma pack(push, 8)

/**
 * @brief Top-k tiling parameter structure.
 */
struct TopKTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements in a single vector.
  uint32_t num_elems;
  /// @brief Size of the matmul tile used in the cube part of scan.
  uint32_t matmul_size;
  /// @brief Size of the vector tiles.
  uint32_t vec_tile_size;
  /// @brief Minimum value of the input vector. 32-bits for i32 or fp32.
  union TopKDtype x_min;
  /// @brief Maximum value of the input vector. 32-bits for i32 or fp32.
  union TopKDtype x_max;
  /// @brief Number of topK items.
  uint32_t k;
};
#pragma pack(pop)
}  // namespace tcuscan
