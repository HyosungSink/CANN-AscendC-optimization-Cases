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
 * \file fused_floyd_attention.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "fused_floyd_attention_s1s2_bn2gs1.h"

using namespace AscendC;

#ifdef __DAV_C220_CUBE__ // CUBE 实现

#define COPY_TILING_DATA(tiling)                                                                                       \
    GET_TILING_DATA_MEMBER(FusedFloydAttentionGeneralTilingData, bmm1TilingData, bmm1TilingDataVar, tiling);           \
    GET_TILING_DATA_MEMBER(FusedFloydAttentionGeneralTilingData, bmm1k1TilingData, bmm1k1TilingDataVar, tiling);       \
    GET_TILING_DATA_MEMBER(FusedFloydAttentionGeneralTilingData, bmm2TilingData, bmm2TilingDataVar, tiling);           \
    GET_TILING_DATA_MEMBER(FusedFloydAttentionGeneralTilingData, bmm2v2TilingData, bmm2v2TilingDataVar, tiling);       \
    const FusedFloydAttentionGeneralTilingData *__restrict tilingData = nullptr;                                       \
    const TCubeTiling *__restrict bmm1tiling = &bmm1TilingDataVar;                                                     \
    const TCubeTiling *__restrict bmm1k1tiling = &bmm1k1TilingDataVar;                                                     \
    const TCubeTiling *__restrict bmm2tiling = &bmm2TilingDataVar;                                                     \
    const TCubeTiling *__restrict bmm2v2tiling = &bmm2v2TilingDataVar;

#define INVOKE_FA_GENERAL_OP_IMPL(templateClass, ...)                                                                  \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1k1, bmm1k1tiling, op.bmm2,         \
              bmm2tiling, op.bmm2v2, bmm2v2tiling);  \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,           \
                          bmm2tiling);                                                                                 \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling, op.bmm2Nz,           \
                          bmm2tiling);                                                                                \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(templateClass, ...)                                                          \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,           \
                          bmm2tiling);                                                                                 \
    } while (0)

#else // VECTOR 实现

#define COPY_TILING_DATA(tiling)                                                                                       \
    GET_TILING_DATA_WITH_STRUCT(FusedFloydAttentionGeneralTilingData, tilingDataIn, tiling);                           \
    const FusedFloydAttentionGeneralTilingData *__restrict tilingData = &tilingDataIn;                                 \
    const TCubeTiling *__restrict bmm1tiling = &(tilingData->bmm1TilingData);                                          \
    const TCubeTiling *__restrict bmm1k1tiling = &(tilingData->bmm1k1TilingData);                                      \
    const TCubeTiling *__restrict bmm2tiling = &(tilingData->bmm2TilingData);                                          \
    const TCubeTiling *__restrict bmm2v2tiling = &(tilingData->bmm2v2TilingData);

#define INVOKE_FA_GENERAL_OP_IMPL(templateClass, ...)                                                                  \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        templateClass<__VA_ARGS__> op;                                                                             \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1k1, bmm1k1tiling,              \
              op.bmm2, bmm2tiling, op.bmm2v2, bmm2v2tiling);                                                       \
        op.Init(query, key_1, value_1, key_2, value_2, attenMask, softmaxMax, softmaxSum,          \
                attentionOut, user, tilingData, &tPipe);                                               \
        op.Process();                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        templateClass<__VA_ARGS__> op;                                                                             \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,       \
                            bmm2tiling);                                                  \
        op.Init(query, key_1, value_1, key_2, value_2, attenMask, softmaxMax, softmaxSum,          \
                attentionOut, user, tilingData, &tPipe);                                               \
        op.Process();                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        templateClass<__VA_ARGS__> op;                                                                             \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling, op.bmm2Nz,       \
                            bmm2tiling);                                                                             \
        op.Init(query, key_1, value_1, key_2, value_2, attenMask, softmaxMax, softmaxSum,          \
                attentionOut, user, tilingData, &tPipe);                                               \
        op.Process();                                                                                              \
    } while (0)

#endif

extern "C" __global__ __aicore__ void
fused_floyd_attention(__gm__ uint8_t *query, __gm__ uint8_t *key_1, __gm__ uint8_t *value_1, __gm__ uint8_t *key_2, __gm__ uint8_t *value_2, 
                      __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *attentionOut,
                      __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    SetMaskNorm();

#if (ORIG_DTYPE_QUERY == DT_FLOAT16)             // 3
    if (TILING_KEY_IS(10000000001022030943UL)) {
        INVOKE_FA_GENERAL_OP_IMPL(FusedFloydAttentionS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, half, float, true, CubeFormat::ND,
                                  false);
        return;
    } else if (TILING_KEY_IS(10000000000022030943UL)) {
        INVOKE_FA_GENERAL_OP_IMPL(FusedFloydAttentionS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, half, float, true, CubeFormat::ND,
                                  false);
        return;
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16)             // 3
    if (TILING_KEY_IS(10000000001022030943UL)) {
        INVOKE_FA_GENERAL_OP_IMPL(FusedFloydAttentionS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, true, false, bfloat16_t, float, true, CubeFormat::ND,
                                  false);
        return;
    } else if (TILING_KEY_IS(10000000000022030943UL)) {
        INVOKE_FA_GENERAL_OP_IMPL(FusedFloydAttentionS1s2Bn2gs1, ImplModeEnum::AA_HIGH_PRECISION,
                                  LayOutTypeEnum::LAYOUT_BNSD, false, false, false, bfloat16_t, float, true, CubeFormat::ND,
                                  false);
        return;
    }
#endif

}
