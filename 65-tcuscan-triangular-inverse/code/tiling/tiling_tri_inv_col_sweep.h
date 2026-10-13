
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tiling_tri_inv_col_sweep.h
 * @brief Tiling structure for Vector `tri_inv_col_sweep` kernel operation.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief `tri_inv_col_sweep` kernel tiling parameter structure.
 */
struct TriInvColumnSweepTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements.
  uint32_t num_elems;
  /// @brief Input matrix size.
  uint32_t matrix_size;
  /// @brief Number of output (matrix) tiles.
  uint32_t num_out_tiles;
};
#pragma pack(pop)

}  // namespace tcuscan
