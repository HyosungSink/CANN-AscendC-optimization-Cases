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
 * \file moe_distribute_combine_add_rms_norm.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#if __has_include("../moe_distribute_combine_v2/moe_distribute_combine_v2_tiling.h")
#include "../moe_distribute_combine_v2/moe_distribute_combine_v2.h"
#include "../moe_distribute_combine_v2/moe_distribute_combine_v2_tiling.h"
#include "../moe_distribute_combine_v2/moe_distribute_combine_v2_tiling_key.h"
#include "../moe_distribute_combine_v2/moe_distribute_combine_tiling.h"
#else
#include "../moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2.h"
#include "../moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2_tiling.h"
#include "../moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2_tiling_key.h"
#include "../moe_distribute_combine_v2/op_kernel/moe_distribute_combine_tiling.h"
#endif
using namespace AscendC;
using namespace MoeDistributeCombineV2Impl;
using namespace Mc2Tiling;

namespace {
template <CombineMC2TypeClass>
__aicore__ inline void ExecMoeDistributeCombineAddRmsNorm(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR epSendCount, GM_ADDR tpSendCount,
    GM_ADDR residualX, GM_ADDR gamma, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo,
    GM_ADDR oriX, GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR YOut,
    GM_ADDR dynamicScaleOut, GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe* pipePtr)
{
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);
    MoeDistributeCombineV2<CombineMC2TypeFunc> op;
    op.Init(
        nullptr, expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, residualX, gamma, scales, xActiveMask,
        sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, nullptr, YOut, dynamicScaleOut,
        XOut, workspaceGM, pipePtr, &tilingData);
    op.Process();
    }
}

/*
* A3 tilingkey说明
* 5位的十进制数
* 第1位（个位）：无意义占位使用
* 第2位（十位）：无意义占位使用
* 第3位（百位）：是否做tp域allgather:
*     0: 不做, 1: 做
+* 第4位（千位）：无实际意义
 * 第5位（万位）：无实际意义
 */

template<bool HasTp, uint8_t QuantMode, uint8_t LayeredMode, uint8_t ArchTag> __global__ __aicore__ void moe_distribute_combine_add_rms_norm(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR epSendCount, GM_ADDR scales, GM_ADDR residualX,
    GM_ADDR gamma, GM_ADDR tpSendCount, GM_ADDR xActiveMask, GM_ADDR activationScale, GM_ADDR weightScale,
    GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1, 
    GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR YOut, GM_ADDR dynamicScaleOut, GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineV2TilingData);
    TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
        ExecMoeDistributeCombineAddRmsNorm<DTYPE_EXPAND_X, DTYPE_X, int32_t, HasTp, QuantMode == TILINGKEY_INT8_QUANT, true>(
            expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, residualX, gamma, scales, xActiveMask,
            sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, YOut, dynamicScaleOut, XOut, workspaceGM, tilingGM, &pipe);
#endif
}