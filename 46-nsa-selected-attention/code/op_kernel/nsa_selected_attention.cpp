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
 * \file nsa_selected_attention.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "nsa_selected_attention.h"

using namespace AscendC;

#ifdef __DAV_C220_CUBE__ // CUBE 实现

#define COPY_TILING_DATA(tiling)                                                                                       \
    GET_TILING_DATA_MEMBER(NsaSelectedAttentionTilingData, bmm1TilingData, bmm1TilingDataVar, tiling);                 \
    GET_TILING_DATA_MEMBER(NsaSelectedAttentionTilingData, bmm2TilingData, bmm2TilingDataVar, tiling);                 \
    const NsaSelectedAttentionTilingData *__restrict tilingData = nullptr;                                             \
    const TCubeTiling *__restrict bmm1tiling = &bmm1TilingDataVar;                                                     \
    const TCubeTiling *__restrict bmm2tiling = &bmm2TilingDataVar;

#define INVOKE_NSA_SELECTED_ATTENTION_GENERAL_OP_IMPL(templateClass, ...)                                              \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);                     \
    } while (0)

#else // VECTOR 实现

#define COPY_TILING_DATA(tiling)                                                                                       \
    GET_TILING_DATA_WITH_STRUCT(NsaSelectedAttentionTilingData, tilingDataIn, tiling);                                 \
    const NsaSelectedAttentionTilingData *__restrict tilingData = &tilingDataIn;                                       \
    const TCubeTiling *__restrict bmm1tiling = &(tilingData->bmm1TilingData);                                          \
    const TCubeTiling *__restrict bmm2tiling = &(tilingData->bmm2TilingData);

#define INVOKE_NSA_SELECTED_ATTENTION_GENERAL_OP_IMPL(templateClass, ...)                                              \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        templateClass<__VA_ARGS__> op;                                                                                 \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);                     \
        op.Init(query, key, value, topkIndices, attenMask, actualSeqQLen,                                              \
    actualSeqKvLen, softmaxMax, softmaxSum, attentionOut, workspace, tilingData, &tPipe);                              \
        op.Process();                                                                                                  \
    } while (0)

#endif

extern "C" __global__ __aicore__ void nsa_selected_attention(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR topkIndices,
    GM_ADDR attenMask, GM_ADDR actualSeqQLen, GM_ADDR actualSeqKvLen,
    GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR attentionOut, GM_ADDR workspace,
    GM_ADDR tiling)
{
    TPipe tPipe;
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    if (TILING_KEY_IS(0)) {
        INVOKE_NSA_SELECTED_ATTENTION_GENERAL_OP_IMPL(NsaSelectedAttention, half);
        return;
    } else if (TILING_KEY_IS(1)) {  // enable mask
        INVOKE_NSA_SELECTED_ATTENTION_GENERAL_OP_IMPL(NsaSelectedAttention, half, true);
        return;
    }
#endif
#if (ORIG_DTYPE_QUERY == DT_BF16)
    if (TILING_KEY_IS(0)) {
        INVOKE_NSA_SELECTED_ATTENTION_GENERAL_OP_IMPL(NsaSelectedAttention, bfloat16_t);
        return;
    } else if (TILING_KEY_IS(1)) {  // enable mask
        INVOKE_NSA_SELECTED_ATTENTION_GENERAL_OP_IMPL(NsaSelectedAttention, bfloat16_t, true);
        return;
    }
#endif
}
