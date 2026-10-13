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
 * \file rms_norm_grad.cpp
 * \brief
 */
#include "rms_norm_grad_split_d_high_precision.h"
#include "rms_norm_grad_split_n_high_precision.h"

extern "C" __global__ __aicore__ void rms_norm_grad(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    if (TILING_KEY_IS(1)) {
        RmsNormGradSplitNHighPrecision<DTYPE_DY, DTYPE_GAMMA> rms_norm_grad_split_n;
        rms_norm_grad_split_n.Init(dy, x, rstd, gamma, dx, dgamma, &tilingData, usrWorkspace);
        rms_norm_grad_split_n.Process();
    }
    if (TILING_KEY_IS(2)) {
        RmsNormGradSplitDHighPrecision<DTYPE_DY, DTYPE_GAMMA> rms_norm_grad_split_d;
        rms_norm_grad_split_d.Init(dy, x, rstd, gamma, dx, dgamma, &tilingData, usrWorkspace);
        rms_norm_grad_split_d.Process();
    }
}

#ifndef __CCE_KT_TEST__
void rms_norm_grad_do(
    uint32_t numBlocks, void* l2ctrl, void* stream, uint8_t* dy, uint8_t* x, uint8_t* rstd, uint8_t* gamma, uint8_t* dx,
    uint8_t* dgamma, uint8_t* workspace, uint8_t* tiling)
{
    rms_norm_grad<<<numBlocks, l2ctrl, stream>>>(dy, x, rstd, gamma, dx, dgamma, workspace, tiling);
}
#endif