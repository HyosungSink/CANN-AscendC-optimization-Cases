/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_csr_gather.h
 * @brief Tiling structure for CSR gather operation.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief `csr_gather` tiling parameter structure.
 */
struct CSRGatherTiling {
  /// @brief Total number of input value elements.
  uint32_t num_elems;
  /// @brief Total number of input x vector elements.
  uint32_t num_x_elems;
  /// @brief Width of the tile: length of the vectors processed by CumSum
  uint32_t tile_len;
  /// @brief Maximum number of `x` elements held in Unified Buffer at once.
  /// Determines the fast-path threshold and the chunk size used for larger `x`
  /// vectors.
  uint32_t x_tile_elems_max;
};

#pragma pack(pop)

}  // namespace tcuscan
