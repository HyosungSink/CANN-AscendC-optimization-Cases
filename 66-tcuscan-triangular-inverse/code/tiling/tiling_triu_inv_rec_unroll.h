/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_triu_inv_rec_unroll.h
 * @brief Tiling structure for triu_inv_rec_unroll kernel operation.
 */

#pragma once

#include <cstdint>

#pragma pack(push, 8)
/**
 * @brief Tiling parameter structure.
 */
struct TriuInvRecUnrollTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Input matrix size.
  uint32_t matrix_size;
};
#pragma pack(pop)