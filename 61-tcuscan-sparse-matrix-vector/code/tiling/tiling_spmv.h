/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026 All rights reserved.
 *
 * @file tiling_spmv.h
 * @brief Tiling structure for CSR SpMV.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief `spmv` kernel tiling parameter structure.
 */
struct SpMVTiling {
  /// @brief Number of non-zeros elements.
  uint32_t nnz;
  /// @brief Total number of segments.
  uint32_t num_segments;
  /// @brief Length of the dense input vector.
  uint32_t x_len;
  /// @brief Tiling length.
  uint32_t tile_len;
  /// @brief Block length.
  uint32_t block_len;
  /// @brief Scaling factor applied to the SpMV product, i.e. the `alpha` of
  /// `y = alpha * A @ x + beta * y`.
  float alpha;
  /// @brief Scaling factor applied to the incoming output vector, i.e. the
  /// `beta` of `y = alpha * A @ x + beta * y`.
  float beta;
};

#pragma pack(pop)

}  // namespace tcuscan
