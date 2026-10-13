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
 * \file dynamic_quant.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "dynamic_quant.h"
#include "dynamic_quant_single_row.h"
#include "dynamic_quant_multi_row.h"
#include "dynamic_quant_db.h"
#include "dynamic_quant_large_shape_opt.h"
#include "dynamic_quant_moe.h"
#include "dynamic_quant_moe_large_shape.h"

#ifdef __CCE_KT_TEST__
#define KERNEL_LINKAGE static
#else
#define KERNEL_LINKAGE extern "C"
#endif

using namespace AscendC;
using namespace DynamicQuantNDOpt;

#define INIT_AND_PROCESS                                                      \
    do {                                                                      \
        op.Init(x, smooth_scales, y, scale, nullptr, workSpace, &tilingData); \
        op.Process();                                                         \
    } while (0)

#define INIT_AND_PROCESS_MOE                                                               \
    do {                                                                                   \
        op.Init(x, smooth_scales, group_index, y, scale, nullptr, workSpace, &tilingData); \
        op.Process();                                                                      \
    } while (0)

KERNEL_LINKAGE __global__ __aicore__ void dynamic_quant(
    GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_index, GM_ADDR y, GM_ADDR scale, GM_ADDR workSpace, GM_ADDR tiling)
{
    if (x == nullptr || y == nullptr || scale == nullptr) {
        return;
    }

    GM_ADDR user1 = GetUserWorkspace(workSpace);
    if (user1 == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    if (GetBlockIdx() >= tilingData.coreNum) {
        return;
    }

    TPipe pipe;
    if (TILING_KEY_IS(0) || TILING_KEY_IS(1)) {
        DynamicQuant<DTYPE_X, DTYPE_Y> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2) || TILING_KEY_IS(3)) {
        DynamicQuantDbOpt<DTYPE_X, DTYPE_Y> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(6)) {
        DynamicQuantLargeShapeOpt<DTYPE_X, DTYPE_Y> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(7)) {
        DynamicQuantMoe<DTYPE_X, int32_t, DTYPE_Y> op(&pipe);
        INIT_AND_PROCESS_MOE;
    } else if (TILING_KEY_IS(8)) {
        DynamicQuantMoeLargeShape<DTYPE_X, DTYPE_X, int32_t, DTYPE_Y> op(&pipe);
        INIT_AND_PROCESS_MOE;
    } else if (TILING_KEY_IS(10)) {
        DynamicQuantMultiRow<DTYPE_X, DTYPE_Y> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(100)) {
        DynamicQuantSingleRow<DTYPE_X, DTYPE_Y> op(&pipe);
        INIT_AND_PROCESS;
    }
}