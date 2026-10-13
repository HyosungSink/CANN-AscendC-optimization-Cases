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
 * \file group_norm_silu.cpp
 * \brief
 */
#include "group_norm_silu_b16.h"
#include "group_norm_silu_b32.h"
#include "group_norm_silu_g16.h"
#include "group_norm_silu_g32.h"
#include "group_norm_silu_small_b16.h"
#include "group_norm_silu_small_b32.h"
#include "group_norm_silu_hw1_b16.h"
#include "group_norm_silu_hw1_b32.h"
#include "group_norm_silu_sd.h"

using namespace GroupNormSilu;

extern "C" __global__ __aicore__ void group_norm_silu(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR silu, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_VECTOR_CORE);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1011)) {
        GroupNormSilu::GroupNormSiluSmallB16<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1012)) {
        GroupNormSilu::GroupNormSiluSmallB16<DTYPE_X, float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(102)) {
        GroupNormSilu::GroupNormSiluSmallB32<float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1031)) {
        GroupNormSilu::GroupNormSiluB16<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1032)) {
        GroupNormSilu::GroupNormSiluB16<DTYPE_X, float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(104)) {
        GroupNormSilu::GroupNormSiluB32<float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1051)) {
        GroupNormSilu::GroupNormSiluG16<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1052)) {
        GroupNormSilu::GroupNormSiluG16<DTYPE_X, float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(106)) {
        GroupNormSilu::GroupNormSiluG32<float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1071)) {
        GroupNormSilu::GroupNormSiluHW1B16<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1072)) {
        GroupNormSilu::GroupNormSiluHW1B16<DTYPE_X, float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(108)) {
        GroupNormSilu::GroupNormSiluHW1B32<float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(109)) {
        GroupNormSilu::GroupNormSiluSD<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    }
}
