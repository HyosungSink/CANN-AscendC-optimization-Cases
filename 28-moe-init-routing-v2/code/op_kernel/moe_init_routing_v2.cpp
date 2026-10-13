/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_init_routing_v2.cpp
 * \brief
 */


#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/moe_v2_mrgsort_out.h"
#include "arch35/moe_v2_mrgsort.h"
#include "arch35/moe_v2_sort_multi_core.h"
#include "arch35/moe_v2_sort_one_core.h"
#include "arch35/moe_v2_expert_token_out_regbase.h"
#include "arch35/moe_v2_expert_token_out_simt.h"
#include "arch35/moe_v2_src_to_dst_op_simt.h"
#include "arch35/moe_v2_src_to_dst_with_capacity_simt.h"
#include "arch35/moe_v2_gather_out_for_simt.h"
#else
#include "moe_v2_mrgsort_out.h"
#include "moe_v2_mrgsort.h"
#include "moe_v2_sort_multi_core.h"
#include "moe_v2_sort_one_core.h"
#include "moe_v2_expert_token_out.h"
#include "moe_v2_src_to_dst_op.h"
#include "moe_v2_src_to_dst_with_capacity.h"
#include "moe_v2_gather_out.h"
#include "moe_v2_init_routing_fullload.h"
#endif

using namespace AscendC;
using namespace MoeInitRoutingV2;

extern "C" __global__ __aicore__ void moe_init_routing_v2(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX,
                                                          GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                                          GM_ADDR expertTokensBeforeCapacity, GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0); 
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

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    if (TILING_KEY_IS(10001) || TILING_KEY_IS(11001) || TILING_KEY_IS(10011) || TILING_KEY_IS(11011)) {
        TPipe sortPipe;
        MoeV2SortOneCore<DTYPE_EXPERT_IDX> op;
        op.Init(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, userWS, t, &sortPipe);
        op.Process();
        sortPipe.Destroy();
    } else if (TILING_KEY_IS(10002) || TILING_KEY_IS(11002) || TILING_KEY_IS(10012) || TILING_KEY_IS(11012)) {
        TPipe sortPipe;
        MoeV2SortMultiCore<DTYPE_EXPERT_IDX> op;
        op.Init(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, userWS, t, &sortPipe);
        op.Process();
        sortPipe.Destroy();
    }

    if (TILING_KEY_IS(11001) || TILING_KEY_IS(11002)) {
        if (t->expertTokensCountOrCumsumFlag != EXERPT_TOKENS_NONE) {
            TPipe expertTokenOutPipe;
            MoeV2ExpertTokenOutRegBase expertTokenOutOp;
            expertTokenOutOp.Init<MoeInitRoutingV2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                              expandedRowIdx, userWS, t, &expertTokenOutPipe);
            expertTokenOutOp.Process();
            expertTokenOutPipe.Destroy();
        }
    } else if (TILING_KEY_IS(11011) || TILING_KEY_IS(11012)) {
        TPipe expertTokenOutPipe;
        MoeV2ExpertTokenOutRegBase expertTokenOutOp;
        expertTokenOutOp.Init<MoeInitRoutingV2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                          expandedRowIdx, userWS, t, &expertTokenOutPipe);
        expertTokenOutOp.Process();
        expertTokenOutPipe.Destroy();

        TPipe expertTokenOutSimtPipe;
        MoeV2ExpertTokenOutSimt expertTokenOutOpSimt;
        expertTokenOutOpSimt.Init<MoeInitRoutingV2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                              expandedRowIdx, userWS, t, &expertTokenOutSimtPipe);
        expertTokenOutOpSimt.Process<false>();
        expertTokenOutSimtPipe.Destroy();
    } else if (TILING_KEY_IS(10001) || TILING_KEY_IS(10002)) {
        if (t->expertTokensCountOrCumsumFlag != EXERPT_TOKENS_NONE) {
            TPipe expertTokenOutPipe;
            MoeV2ExpertTokenOutSimt expertTokenOutOpSimt;
            expertTokenOutOpSimt.Init<MoeInitRoutingV2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                                  expandedRowIdx, userWS, t, &expertTokenOutPipe);
            expertTokenOutOpSimt.Process();
            expertTokenOutPipe.Destroy();
        }
    } else if (TILING_KEY_IS(10011) || TILING_KEY_IS(10012)) {
        TPipe expertTokenOutPipe;
        MoeV2ExpertTokenOutSimt expertTokenOutOpSimt;
        expertTokenOutOpSimt.Init<MoeInitRoutingV2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                              expandedRowIdx, userWS, t, &expertTokenOutPipe);
        expertTokenOutOpSimt.Process();
        expertTokenOutPipe.Destroy();
    }

    if (TILING_KEY_IS(10001) || TILING_KEY_IS(11001) || TILING_KEY_IS(10002) || TILING_KEY_IS(11002)) {
        MoeV2SrcToDstOpSimt srcToDstOpSimt;
        srcToDstOpSimt.Init<MoeInitRoutingV2TilingData>(expandedRowIdx, userWS, t);
        srcToDstOpSimt.Process();
    } else {
        MoeV2SrcToDstWithCapacitySimt<DTYPE_X, MoeInitRoutingV2TilingData> srcToDstWithCapacityOpSimt;
        srcToDstWithCapacityOpSimt.Init(expandedRowIdx, expandedX, userWS, t);
        srcToDstWithCapacityOpSimt.Process();
    }

    TPipe gatherPipe;
    MoeV2GatherOutSimt<DTYPE_X> gatherOpSimt;
    gatherOpSimt.Init(x, expandedRowIdx, expandedX, userWS, t, &gatherPipe);
    gatherOpSimt.Process();
    gatherPipe.Destroy();
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    if (TILING_KEY_IS(20000)) {
        TPipe sortPipe;
        MoeV2FullLoad<DTYPE_X> op;
        op.Init(x, expertIdx, expandedX, expandedRowIdx, expertTokensCountOrCumsum, userWS, t, &sortPipe);
        op.Process();
        sortPipe.Destroy();
        return;
    }

    if (TILING_KEY_IS(10001) || TILING_KEY_IS(10011)) {
        TPipe sortPipe;
        MoeV2SortOneCore op;
        op.Init<MoeInitRoutingV2TilingData>(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, userWS, t,
                                            &sortPipe);
        op.Process();
        op.ResetIO(expandedRowIdx, userWS);
        op.Process();
        sortPipe.Destroy();
    } else if (TILING_KEY_IS(10002) || TILING_KEY_IS(10012)) {
        TPipe sortPipe;
        MoeV2SortMultiCore op;
        op.Init<MoeInitRoutingV2TilingData>(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, userWS, t,
                                            &sortPipe);
        op.Process();
        op.ResetIO(expandedRowIdx, userWS);
        op.Process();
        sortPipe.Destroy();
    }

    if (TILING_KEY_IS(10001) || TILING_KEY_IS(10002)) {
        if (t->expertTokensCountOrCumsumFlag != EXERPT_TOKENS_NONE) {
            TPipe expertTokenOutPipe;
            MoeV2ExpertTokenOut expertTokenOutOp;
            expertTokenOutOp.Init<MoeInitRoutingV2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                              expandedRowIdx, userWS, t, &expertTokenOutPipe);
            expertTokenOutOp.Process();
            expertTokenOutPipe.Destroy();
        }

    } else if (TILING_KEY_IS(10011) || TILING_KEY_IS(10012)) {
        TPipe expertTokenOutPipe;
        MoeV2ExpertTokenOut expertTokenOutOp;
        expertTokenOutOp.Init<MoeInitRoutingV2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                          expandedRowIdx, userWS, t, &expertTokenOutPipe);
        expertTokenOutOp.Process();
        expertTokenOutPipe.Destroy();
    }
    TPipe gatherPipe;
    MoeV2GatherOut<DTYPE_X> gatherOp;
    gatherOp.Init(x, expandedRowIdx, expandedX, userWS, t, &gatherPipe);
    gatherOp.Process();
    gatherPipe.Destroy();
#else
    if (TILING_KEY_IS(20000)) {
        TPipe sortPipe;
        MoeV2FullLoad<DTYPE_X> op;
        op.Init(x, expertIdx, expandedX, expandedRowIdx, expertTokensCountOrCumsum, userWS, t, &sortPipe);
        op.Process();
        sortPipe.Destroy();
        return;
    }

    if (TILING_KEY_IS(10001) || TILING_KEY_IS(10011)) {
        TPipe sortPipe;
        MoeV2SortOneCore op;
        op.Init<MoeInitRoutingV2TilingData>(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, userWS, t,
                                            &sortPipe);
        op.Process();
        sortPipe.Destroy();
    } else if (TILING_KEY_IS(10002) || TILING_KEY_IS(10012)) {
        TPipe sortPipe;
        MoeV2SortMultiCore op;
        op.Init<MoeInitRoutingV2TilingData>(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, userWS, t,
                                            &sortPipe);
        op.Process();
        sortPipe.Destroy();
    }

    if (TILING_KEY_IS(10001) || TILING_KEY_IS(10002)) {
        if (t->expertTokensCountOrCumsumFlag != EXERPT_TOKENS_NONE) {
            TPipe expertTokenOutPipe;
            MoeV2ExpertTokenOut expertTokenOutOp;
            expertTokenOutOp.Init<MoeInitRoutingV2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                              expandedRowIdx, userWS, t, &expertTokenOutPipe);
            expertTokenOutOp.Process();
            expertTokenOutPipe.Destroy();
        }
        TPipe srcToDstPipe;
        MoeV2SrcToDstOp srcToDstOp;
        srcToDstOp.Init<MoeInitRoutingV2TilingData>(expandedRowIdx, userWS, t, &srcToDstPipe);
        srcToDstOp.Process();
        srcToDstPipe.Destroy();
    } else if (TILING_KEY_IS(10011) || TILING_KEY_IS(10012)) {
        TPipe expertTokenOutPipe;
        MoeV2ExpertTokenOut expertTokenOutOp;
        expertTokenOutOp.Init<MoeInitRoutingV2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                          expandedRowIdx, userWS, t, &expertTokenOutPipe);
        expertTokenOutOp.Process();
        expertTokenOutPipe.Destroy();

        TPipe srcToDstPipe;
        MoeV2SrcToDstWithCapacity<DTYPE_X, MoeInitRoutingV2TilingData> srcToDstWithCapacityOp;
        srcToDstWithCapacityOp.Init(expandedRowIdx, expandedX, userWS, t, &srcToDstPipe);
        srcToDstWithCapacityOp.Process();
        srcToDstPipe.Destroy();
    }

    TPipe gatherPipe;
    MoeV2GatherOut<DTYPE_X> gatherOp;
    gatherOp.Init(x, expandedRowIdx, expandedX, userWS, t, &gatherPipe);
    gatherOp.Process();
    gatherPipe.Destroy();
#endif
}
