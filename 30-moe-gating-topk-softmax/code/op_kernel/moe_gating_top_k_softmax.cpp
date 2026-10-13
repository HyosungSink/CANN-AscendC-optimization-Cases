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
 * \file moe_gating_top_k_softmax.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "moe_gating_top_k_softmax_e_k_fullload.h"
#include "moe_gating_top_k_softmax_k_fullload.h"
#include "moe_gating_top_k_softmax_perf.h"

using namespace MoeGatingTopKSoftmax;

#define MOE_GATING_TOP_K_SOFTMAX_E_K_FULL_LOAD_IMPL(INPUT_TYPE, BUFFER_NUM)                            \
    do {                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKSoftmaxEKFullLoadTilingData, tiling_data_in, tiling); \
        const MoeGatingTopKSoftmaxEKFullLoadTilingData* __restrict tilingData = &tiling_data_in;       \
        MoeGatingTopKSoftmaxEKFullLoad<INPUT_TYPE, BUFFER_NUM> op;                                     \
        op.Init(gating, finished, out, indicesOut, sourceRowsOut, workspace, tilingData);              \
        op.Process();                                                                                  \
    } while (0)

#define MOE_GATING_TOP_K_SOFTMAX_K_FULL_LOAD_IMPL(INPUT_TYPE, BUFFER_NUM)                             \
    do {                                                                                              \
        GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKSoftmaxKFullLoadTilingData, tiling_data_in, tiling); \
        const MoeGatingTopKSoftmaxKFullLoadTilingData* __restrict tilingData = &tiling_data_in;       \
        MoeGatingTopKSoftmaxKFullLoad<INPUT_TYPE, BUFFER_NUM> op;                                     \
        op.Init(gating, finished, out, indicesOut, sourceRowsOut, workspace, tilingData);             \
        op.Process();                                                                                 \
    } while (0)

#define MOE_GATING_TOP_K_SOFTMAX_PERF_IMPL(INPUT_TYPE, COL_RANGE)                                \
    do {                                                                                         \
        GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKSoftmaxPerfTilingData, tiling_data_in, tiling); \
        const MoeGatingTopKSoftmaxPerfTilingData* __restrict tilingData = &tiling_data_in;       \
        MoeGatingTopKSoftmaxPerf<INPUT_TYPE, COL_RANGE> op;                                      \
        op.Init(gating, finished, out, indicesOut, sourceRowsOut, workspace, tilingData);        \
        op.Process(tilingData);                                                                  \
    } while (0)

extern "C" __global__ __aicore__ void moe_gating_top_k_softmax(
    GM_ADDR gating, GM_ADDR finished, GM_ADDR out, GM_ADDR indicesOut, GM_ADDR sourceRowsOut, GM_ADDR workspace,
    GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    if (TILING_KEY_IS(0)) {
        MOE_GATING_TOP_K_SOFTMAX_E_K_FULL_LOAD_IMPL(float, 1);
        return;
    } else if (TILING_KEY_IS(1)) {
        MOE_GATING_TOP_K_SOFTMAX_E_K_FULL_LOAD_IMPL(float, 2);
        return;
    } else if (TILING_KEY_IS(2)) {
        MOE_GATING_TOP_K_SOFTMAX_E_K_FULL_LOAD_IMPL(half, 1);
        return;
    } else if (TILING_KEY_IS(3)) {
        MOE_GATING_TOP_K_SOFTMAX_E_K_FULL_LOAD_IMPL(half, 2);
        return;
    } else if (TILING_KEY_IS(4)) {
        MOE_GATING_TOP_K_SOFTMAX_E_K_FULL_LOAD_IMPL(bfloat16_t, 1);
        return;
    } else if (TILING_KEY_IS(5)) {
        MOE_GATING_TOP_K_SOFTMAX_E_K_FULL_LOAD_IMPL(bfloat16_t, 2);
        return;
    } else if (TILING_KEY_IS(6)) {
        MOE_GATING_TOP_K_SOFTMAX_K_FULL_LOAD_IMPL(float, 2);
        return;
    } else if (TILING_KEY_IS(7)) {
        MOE_GATING_TOP_K_SOFTMAX_K_FULL_LOAD_IMPL(half, 2);
        return;
    } else if (TILING_KEY_IS(8)) {
        MOE_GATING_TOP_K_SOFTMAX_K_FULL_LOAD_IMPL(bfloat16_t, 2);
        return;
    } else if (TILING_KEY_IS(9)) {
        MOE_GATING_TOP_K_SOFTMAX_PERF_IMPL(float, ColRangeEnum::SMALLER_THAN_8);
        return;
    } else if (TILING_KEY_IS(10)) {
        MOE_GATING_TOP_K_SOFTMAX_PERF_IMPL(half, ColRangeEnum::SMALLER_THAN_8);
        return;
    } else if (TILING_KEY_IS(11)) {
        MOE_GATING_TOP_K_SOFTMAX_PERF_IMPL(bfloat16_t, ColRangeEnum::SMALLER_THAN_8);
        return;
    } else if (TILING_KEY_IS(12)) {
        MOE_GATING_TOP_K_SOFTMAX_PERF_IMPL(float, ColRangeEnum::FROM_8_TO_64);
        return;
    } else if (TILING_KEY_IS(13)) {
        MOE_GATING_TOP_K_SOFTMAX_PERF_IMPL(half, ColRangeEnum::FROM_8_TO_64);
        return;
    } else if (TILING_KEY_IS(14)) {
        MOE_GATING_TOP_K_SOFTMAX_PERF_IMPL(bfloat16_t, ColRangeEnum::FROM_8_TO_64);
        return;
    } else if (TILING_KEY_IS(15)) {
        MOE_GATING_TOP_K_SOFTMAX_PERF_IMPL(float, ColRangeEnum::BIGGER_THAN_64);
        return;
    } else if (TILING_KEY_IS(16)) {
        MOE_GATING_TOP_K_SOFTMAX_PERF_IMPL(half, ColRangeEnum::BIGGER_THAN_64);
        return;
    } else if (TILING_KEY_IS(17)) {
        MOE_GATING_TOP_K_SOFTMAX_PERF_IMPL(bfloat16_t, ColRangeEnum::BIGGER_THAN_64);
        return;
    }
    return;
}
