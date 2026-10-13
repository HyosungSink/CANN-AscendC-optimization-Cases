/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tiling_cube_reduce.h
 * @brief Tiling structure for Cube block reducer.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Cube reduce kernel tiling parameter structure.
 */
struct CubeReduceTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements.
  uint32_t vec_len;
  /// @brief Matrix multiplication size.
  uint32_t matmul_size;
};
#pragma pack(pop)

}  // namespace tcuscan
