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
 * \file moe_init_routing_quant.cpp
 * \brief
 */
#include "moe_quant_gather_out.h"
#include "moe_quant_mrgsort_out.h"
#include "moe_quant_mrgsort.h"
#include "moe_quant_sort_multi_core.h"
#include "moe_quant_sort_one_core.h"
#include "moe_quant_src_to_dst_op.h"
#include "moe_quant_gather_out.h"
#include "moe_quant_gather_out_small_activate_row.h"

using namespace AscendC;
using namespace MoeInitRoutingQuant;
extern "C" __global__ __aicore__ void moe_init_routing_quant(
    GM_ADDR x,
    GM_ADDR rowIdx,
    GM_ADDR expertIdx,
    GM_ADDR expandedX,
    GM_ADDR expandedRowIdx,
    GM_ADDR expandedExpertIdx,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    auto t = &tilingData;
    TPipe sortPipe;
    // sort
    if (TILING_KEY_IS(1) || TILING_KEY_IS(3)) {
        MoeSortOneCore moeSortOneCore;
        moeSortOneCore.Init(expertIdx, rowIdx, expandedExpertIdx, userWS, t, &sortPipe);
        moeSortOneCore.Process();
    } else if (TILING_KEY_IS(2) || TILING_KEY_IS(4)) {
        MoeSortMultiCore moeSortMultiCore;
        moeSortMultiCore.Init(expertIdx, rowIdx, expandedExpertIdx, userWS, t, &sortPipe);
        moeSortMultiCore.Process();
    } else {
        return;
    }
    sortPipe.Destroy();
    TPipe srcToDstPipe;
    MoeSrcToDstOp moeSrcToDstOp;
    moeSrcToDstOp.Init(expandedRowIdx, userWS, t, &srcToDstPipe);
    moeSrcToDstOp.Process();
    srcToDstPipe.Destroy();

    TPipe gatherPipe;
    if (TILING_KEY_IS(1) || TILING_KEY_IS(2)) {
        MoeGatherOut<DTYPE_X> moeGatherOut;
        moeGatherOut.Init(x, expandedRowIdx, expandedX, t, &gatherPipe);
        moeGatherOut.Process();
    } else if (TILING_KEY_IS(3) || TILING_KEY_IS(4)) {
        MoeGatherOutSmallActiveRow<DTYPE_X> moeGatherOutSmallActiveRow;
        moeGatherOutSmallActiveRow.Init(x, userWS, expandedRowIdx, expandedX, t, &gatherPipe);
        moeGatherOutSmallActiveRow.Process();
    }
}
