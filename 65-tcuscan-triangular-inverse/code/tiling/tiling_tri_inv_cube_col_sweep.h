
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tiling_tri_inv_cube_col_sweep.h
 * @brief Tiling structure for cube-based column sweep matrix inversion.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief `KernelTriInvCubeColSweeep` tiling parameter structure.
 */
struct TriInvCubeColSweepTiling {
  /// @brief Number of blocks / batch dimension.
  uint32_t num_blocks;
  /// @brief Matrix size.
  uint32_t matrix_size;
  /// @brief Workspace circular buffer length.
  uint32_t ws_circular_buffer_len;
};
#pragma pack(pop)

}  // namespace tcuscan
