/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_gating_top_k_softmax_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_SOFTMAX_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_SOFTMAX_H_
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
enum MoeGatingTopKSoftmaxTilingKey
{
    MOE_GATING_SOFTMAX_FLOAT = 0,
    MOE_GATING_SOFTMAX_FLOAT_DOUBLE_BUFFER = 1,
    MOE_GATING_SOFTMAX_FLOAT16 = 2,
    MOE_GATING_SOFTMAX_FLOAT16_DOUBLE_BUFFER = 3,
    MOE_GATING_SOFTMAX_BF16 = 4,
    MOE_GATING_SOFTMAX_BF16_DOUBLE_BUFFER = 5,
    MOE_GATING_SOFTMAX_K_FULL_LOAD_FLOAT = 6,
    MOE_GATING_SOFTMAX_K_FULL_LOAD_FLOAT16 = 7,
    MOE_GATING_SOFTMAX_K_FULL_LOAD_BF16 = 8,
    MOE_GATING_SOFTMAX_PERF_FLOAT_COL_SMALLER_THAN_8 = 9,
    MOE_GATING_SOFTMAX_PERF_FLOAT16_COL_SMALLER_THAN_8 = 10,
    MOE_GATING_SOFTMAX_PERF_BF16_COL_SMALLER_THAN_8 = 11,
    MOE_GATING_SOFTMAX_PERF_FLOAT_COL_FROM_8_TO_64 = 12,
    MOE_GATING_SOFTMAX_PERF_FLOAT16_COL_FROM_8_TO_64 = 13,
    MOE_GATING_SOFTMAX_PERF_BF16_COL_FROM_8_TO_64 = 14,
    MOE_GATING_SOFTMAX_PERF_FLOAT_COL_BIGGER_THAN_64 = 15,
    MOE_GATING_SOFTMAX_PERF_FLOAT16_COL_BIGGER_THAN_64 = 16,
    MOE_GATING_SOFTMAX_PERF_BF16_COL_BIGGER_THAN_64 = 17
};

struct MoeGatingTopKSoftmaxCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_SOFTMAX_H_
