/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_topk_pivot.h
 * @brief Tiling structure for Vector K-largest value estimator.
 */

#pragma once

#include <cstdint>
namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Vector K-largest estimator kernel tiling parameter structure.
 */
struct TopKPivotTiling {
  /// @brief Total number of input elements.
  uint32_t num_elems;
  /// @brief Number of samples of length *exactly* 32.
  uint32_t num_samples;
  /// @brief Inner top-k parameter. Integer in range (1,32).
  uint32_t k_inner;
  /// @brief Top-K parameter. Integer in range (1, num_samples).
  uint32_t k_outer;
};
#pragma pack(pop)
}  // namespace tcuscan
