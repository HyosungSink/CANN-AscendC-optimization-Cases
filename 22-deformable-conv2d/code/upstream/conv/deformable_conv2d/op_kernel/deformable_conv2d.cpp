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
 * \file deformable_conv2d.cpp
 * \brief
 */

#include "deformable_conv2d_base.h"
#include "deformable_conv2d_base.cpp"

using namespace DeformableConv2dNS;

extern "C" __global__ __aicore__ void deformable_conv2d(GM_ADDR x, GM_ADDR weight, GM_ADDR offset, GM_ADDR bias,
    GM_ADDR out, GM_ADDR deform_out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    const DeformableConv2dTilingData *__restrict tiling_data = &tilingData;
    const TCubeTiling *__restrict mmTiling = &(tiling_data->mmTilingData);

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

#define INIT_AND_PROCESS                                                       \
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulObj, mmTiling); \
    op.Init(x, weight, offset, bias, out, deform_out, userWS, &tilingData);    \
    op.Process()

    if (TILING_KEY_IS(1)) {
        if constexpr (std::is_same_v<DTYPE_X, float>) {
            DeformableConv2dND<float> op;
            INIT_AND_PROCESS;
        }
        if constexpr (std::is_same_v<DTYPE_X, half>) {
            DeformableConv2dND<half> op;
            INIT_AND_PROCESS;
        }
        if constexpr (std::is_same_v<DTYPE_X, bfloat16_t>) {
            DeformableConv2dND<bfloat16_t> op;
            INIT_AND_PROCESS;
        }
    }
}
