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
 * \file flash_attention_score.cpp
 * \brief
 */

#ifdef KFC_L1_RESERVER_SIZE
#undef KFC_L1_RESERVER_SIZE
#define KFC_L1_RESERVER_SIZE 0
#else
#define KFC_L1_RESERVER_SIZE 0
#endif

#include "kernel_operator.h"
#include "arch32/flash_attention_score_empty_tensor.h"
#include "arch32/flash_attention_score_drop_mask_adapter.h"
#include "arch32/flash_attention_score_s1s2_bn2gs1.h"
#include "arch32/flash_attention_score_s1s2_bn2gs1_sab.h"
#include "arch32/flash_attention_score_s1_bn2gs1.h"
#include "arch32/flash_attention_score_bn2gs1s2_b.h"
#include "arch32/flash_attention_var_len_score.h"
#include "arch32/flash_attention_var_len_score_sab.h"
#include "arch32/flash_attention_score_template_tiling_key.h"
#include "arch32/flash_attention_score_tiling.h"

using namespace AscendC;

#ifdef __DAV_C220_CUBE__ // CUBE 实现

#define COPY_TILING_DATA(tiling)                                                                                       \
    GET_TILING_DATA_MEMBER(FlashAttentionScoreGeneralTilingData, bmm1TilingData, bmm1TilingDataVar, tiling);           \
    GET_TILING_DATA_MEMBER(FlashAttentionScoreGeneralTilingData, bmm2TilingData, bmm2TilingDataVar, tiling);           \
    const FlashAttentionScoreGeneralTilingData *__restrict tilingData = nullptr;                                       \
    const TCubeTiling *__restrict bmm1tiling = &bmm1TilingDataVar;                                                     \
    const TCubeTiling *__restrict bmm2tiling = &bmm2TilingDataVar;                                        

#define COPY_TILING_DATA_SAMEAB(tiling)                                                                                \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGeneralTilingData, tilingDataIn, tiling);                           \
    const FlashAttentionScoreGeneralTilingData *__restrict tilingData = &tilingDataIn;                                 \
    const TCubeTiling *__restrict bmm1tiling = &(tilingData->bmm1TilingData);                                          \
    const TCubeTiling *__restrict bmm2tiling = &(tilingData->bmm2TilingData);                                          

#define INVOKE_FA_GENERAL_OP_IMPL(templateClass, ...)                                                                  \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);                     \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_SAMEAB(templateClass, ...)                                                           \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA_SAMEAB(tiling);                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, sink,            \
                softmaxMax, softmaxSum,                                                                                \
                softmaxOut, attentionOut, user, tilingData, &tPipe);                                                   \
        op.Process();                                                                                                  \
    } while (0)


#define INVOKE_FA_GENERAL_OP_IMPL_WITH_MMPOLICY(templateClass, ...)                                                    \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA_SAMEAB(tiling);                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        FaTscm::FaTscmArray tscmArray[FaTscm::TSCM_BUF_NUM];                                                           \
        FaTscm::TscmGlobal = tscmArray;                                                                                \
        FaTscm::InitTscmBuffer(&tPipe, FaTscm::TscmGlobal);                                                            \
        op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, sink,            \
                softmaxMax, softmaxSum,                                                                                \
                softmaxOut, attentionOut, user, tilingData, &tPipe);                                                   \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,           \
                          bmm2tiling);                                                                                 \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ_SAMEAB(templateClass, ...)                                                    \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA_SAMEAB(tiling);                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, sink,            \
                softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                           \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ_WITH_MMPOLICY(templateClass, ...)                                             \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        matmul::GlobalL1Array l1Array[matmul::L1_BUF_NUM];                                                             \
        matmul::l1Global = l1Array;                                                                                    \
        matmul::InitL1Buffer(&tPipe, matmul::l1Global);                                                                \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,           \
                          bmm2tiling);                                                                                 \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        COPY_TILING_DATA(tiling);                                                                                      \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling, op.bmm2Nz,           \
                          bmm2tiling);                                                                                 \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(templateClass, ...)                                                          \
    do {                                                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                 \
        GET_TILING_DATA_MEMBER(FlashAttentionScoreGeneralTilingData, inputParams, inputParamsVar, tiling);             \
        if (inputParamsVar.needL1Carry) {                                                                              \
            COPY_TILING_DATA_SAMEAB(tiling);                                                                           \
            __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                        \
            op.UnpackInit(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, sink,  \
                          actualSeqLengths, actualSeqLengthsKv,                                                        \
                          softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                 \
            op.ProcessL1Carry();                                                                                       \
        } else {                                                                                                       \
            COPY_TILING_DATA(tiling);                                                                                  \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,       \
                              bmm2tiling);                                                                             \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(templateClass, ...)                                                   \
    do {                                                                                                               \
        COPY_TILING_DATA_SAMEAB(tiling);                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        templateClass<__VA_ARGS__> op;                                                                                 \
        op.UnpackInit(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, sink,      \
                      actualSeqLengths,                                                                                \
                      actualSeqLengthsKv, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe); \
        op.Process();                                                                                                  \
    } while (0)

#else // VECTOR 实现

#define COPY_TILING_DATA(tiling)                                                                                       \
    REGISTER_TILING_DEFAULT(FlashAttentionScoreGeneralTilingData);                                                     \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGeneralTilingData, tilingDataIn, tiling);                           \
    const FlashAttentionScoreGeneralTilingData *__restrict tilingData = &tilingDataIn;                                 \
    const TCubeTiling *__restrict bmm1tiling = &(tilingData->bmm1TilingData);                                          \
    const TCubeTiling *__restrict bmm2tiling = &(tilingData->bmm2TilingData);

#define INVOKE_FA_GENERAL_OP_IMPL(templateClass, ...)                                                                  \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Reset();                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);                 \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask,              \
                    sink, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                 \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);                 \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, sink,        \
                    softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                       \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_SAMEAB(templateClass, ...)                                                           \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Reset();                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask,              \
                    sink, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                 \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask,              \
                    sink, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                 \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_WITH_MMPOLICY(templateClass, ...)                                                    \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Reset();                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            FaTscm::FaTscmArray tscmArray[FaTscm::TSCM_BUF_NUM];                                                       \
            FaTscm::TscmGlobal = tscmArray;                                                                            \
            FaTscm::InitTscmBuffer(&tPipe, FaTscm::TscmGlobal);                                                        \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask,              \
                    sink, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                 \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            FaTscm::FaTscmArray tscmArray[FaTscm::TSCM_BUF_NUM];                                                       \
            FaTscm::TscmGlobal = tscmArray;                                                                            \
            FaTscm::InitTscmBuffer(&tPipe, FaTscm::TscmGlobal);                                                        \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask,              \
                    sink, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                 \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0) 

#define INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Destroy();                                                                                           \
            TPipe tPipeOp;                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipeOp, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,     \
                              bmm2tiling);                                                                             \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask,              \
                    sink, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipeOp);               \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,       \
                              bmm2tiling);                                                                             \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask,              \
                    sink, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                 \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ_SAMEAB(templateClass, ...)                                                    \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Destroy();                                                                                           \
            TPipe tPipeOp;                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask,              \
                    sink, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipeOp);               \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            op.Init(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask,              \
                    sink, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe);                 \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ_WITH_MMPOLICY(templateClass, ...)                                             \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Destroy();                                                                                           \
            TPipe tPipeOp;                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            matmul::GlobalL1Array l1Array[matmul::L1_BUF_NUM];                                                         \
            matmul::l1Global = l1Array;                                                                                \
            matmul::InitL1Buffer(&tPipe, matmul::l1Global);                                                            \
            REGIST_MATMUL_OBJ(&tPipeOp, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,     \
                              bmm2tiling);                                                                             \
            op.Init(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, sink, softmaxMax, softmaxSum,    \
                    softmaxOut, attentionOut, user, tilingData, &tPipeOp);                                             \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            matmul::GlobalL1Array l1Array[matmul::L1_BUF_NUM];                                                         \
            matmul::l1Global = l1Array;                                                                                \
            matmul::InitL1Buffer(&tPipe, matmul::l1Global);                                                             \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,       \
                              bmm2tiling);                                                                             \
            op.Init(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, sink, softmaxMax, softmaxSum,    \
                    softmaxOut, attentionOut, user, tilingData, &tPipe);                                               \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(templateClass, ...)                                                           \
    do {                                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        COPY_TILING_DATA(tiling);                                                                                      \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
            tPipe.Destroy();                                                                                           \
            TPipe tPipeOp;                                                                                             \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipeOp, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling, op.bmm2Nz,       \
                              bmm2tiling);                                                                             \
            op.Init(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, sink, softmaxMax, softmaxSum,    \
                    softmaxOut, attentionOut, user, tilingData, &tPipeOp);                                             \
            op.Process();                                                                                              \
        } else {                                                                                                       \
            templateClass<__VA_ARGS__> op;                                                                             \
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling, op.bmm2Nz,       \
                              bmm2tiling);                                                                             \
            op.Init(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, sink, softmaxMax, softmaxSum,    \
                    softmaxOut, attentionOut, user, tilingData, &tPipe);                                               \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(templateClass, ...)                                                          \
    do {                                                                                                               \
        COPY_TILING_DATA(tiling);                                                                                      \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
        }                                                                                                              \
        tPipe.Reset();                                                                                                 \
        templateClass<__VA_ARGS__> op;                                                                                 \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2,           \
                          bmm2tiling);                                                                                 \
        op.UnpackInit(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, sink,      \
                      actualSeqLengths,                                                                                \
                      actualSeqLengthsKv, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe); \
        if (tilingData->inputParams.needL1Carry) {                                                                     \
            op.ProcessL1Carry();                                                                                       \
        } else {                                                                                                       \
            op.Process();                                                                                              \
        }                                                                                                              \
    } while (0)

#define INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(templateClass, ...)                                                   \
    do {                                                                                                               \
        COPY_TILING_DATA(tiling);                                                                                      \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                            \
        if (tilingData->inputParams.needDropMaskOp) {                                                                  \
            FlashAttentionScoreDropMaskAdapter dropMaskAdapter;                                                        \
            dropMaskAdapter.Init(dropMask, user, tilingData, &tPipe);                                                  \
            dropMaskAdapter.Process();                                                                                 \
        }                                                                                                              \
        tPipe.Reset();                                                                                                 \
        templateClass<__VA_ARGS__> op;                                                                                 \
        op.UnpackInit(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, sink,      \
                      actualSeqLengths,                                                                                \
                      actualSeqLengthsKv, softmaxMax, softmaxSum, softmaxOut, attentionOut, user, tilingData, &tPipe); \
        op.Process();                                                                                                  \
    } while (0)

#endif

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
template<uint8_t KernelTypeKey, uint8_t UB0, uint8_t UB1, uint8_t Block, uint8_t ImplMode,
    uint8_t DataType, uint8_t Layout, uint8_t Bmm1Format, uint8_t Bmm2Source, uint8_t Sparse, uint8_t BigDoubleBuffer,
    bool HasDropOut, bool HasAttenMask, bool HasPse, bool EnableL1Reuse, bool HasRope,
    uint8_t MatmulPolicyType, uint8_t S1TemplateType, uint8_t S2TemplateType, uint8_t dTemplateSize>
__global__ __aicore__ void
flash_attention_score(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                      __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
                      __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
                      __gm__ uint8_t *qStartIdx, __gm__ uint8_t *kvStartIdx, __gm__ uint8_t *deqScaleQ,
                      __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *queryRope,
                      __gm__ uint8_t *keyRope, __gm__ uint8_t *sink, __gm__ uint8_t *pScale, __gm__ uint8_t *softmaxMax,
                      __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut,
                      __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    AscendC::SetMaskNorm();
    REGISTER_TILING_DEFAULT(FlashAttentionScoreGeneralTilingData);
    if constexpr (KernelTypeKey == 1) {
        REGISTER_TILING_FOR_TILINGKEY("(TILING_KEY_VAR == 0x1)", FlashAttentionScoreTilingData);
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreTilingData, tiling_data_in, tiling);
        const FlashAttentionScoreTilingData *__restrict tiling_data = &tiling_data_in;
        if (ORIG_DTYPE_QUERY == DT_FLOAT16) {
            FlashAttentionScoreEmptyTensor<half> op;
            op.Init(softmaxMax, softmaxSum, attentionOut, tiling_data);
            op.Process();
        } else if (ORIG_DTYPE_QUERY == DT_FLOAT) {
            FlashAttentionScoreEmptyTensor<float> op;
            op.Init(softmaxMax, softmaxSum, attentionOut, tiling_data);
            op.Process();
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        } else if (ORIG_DTYPE_QUERY == DT_BF16) {
            FlashAttentionScoreEmptyTensor<bfloat16_t> op;
            op.Init(softmaxMax, softmaxSum, attentionOut, tiling_data);
            op.Process();
#endif
        }
        return;
    }

    // TND sameAB 模版
    if constexpr (UB0 == 3 && UB1 == 9 && Block == 9 && Layout == 4)
    {
        if constexpr (DataType == 3) {
            if constexpr (MatmulPolicyType == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(FlashAttentionVarLenScoreSameAB,
                    LayOutTypeEnum::LAYOUT_TND, HasPse, HasAttenMask, HasDropOut,
                    half, float, CubeFormat::NZ, MmPolicyType::UNSPLITK, ImplModeEnum(ImplMode), HasRope);
            } else {
                INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(FlashAttentionVarLenScoreSameAB,
                    LayOutTypeEnum::LAYOUT_TND, HasPse, HasAttenMask, HasDropOut,
                    half, float, CubeFormat::NZ, MmPolicyType::NORMAL, ImplModeEnum(ImplMode), HasRope);
            }
        } else if constexpr (DataType == 2) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
            if constexpr (MatmulPolicyType == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(FlashAttentionVarLenScoreSameAB,
                    LayOutTypeEnum::LAYOUT_TND, HasPse, HasAttenMask, HasDropOut,
                    bfloat16_t, float, CubeFormat::NZ, MmPolicyType::UNSPLITK, ImplModeEnum(ImplMode), HasRope);
            } else {
                INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(FlashAttentionVarLenScoreSameAB,
                    LayOutTypeEnum::LAYOUT_TND, HasPse, HasAttenMask, HasDropOut,
                    bfloat16_t, float, CubeFormat::NZ, MmPolicyType::NORMAL, ImplModeEnum(ImplMode), HasRope);
            }
#endif
        } else if constexpr (DataType == 1){
            if constexpr (MatmulPolicyType == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(FlashAttentionVarLenScoreSameAB,
                    LayOutTypeEnum::LAYOUT_TND, HasPse, HasAttenMask, HasDropOut,
                    float, float, CubeFormat::NZ, MmPolicyType::UNSPLITK, ImplModeEnum(ImplMode), HasRope);
            } else {
                INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(FlashAttentionVarLenScoreSameAB,
                    LayOutTypeEnum::LAYOUT_TND, HasPse, HasAttenMask, HasDropOut,
                    float, float, CubeFormat::NZ, MmPolicyType::NORMAL, ImplModeEnum(ImplMode), HasRope);
            }
        }
    }

    // TND 模版
    if constexpr (UB0 == 3 && UB1 == 4 && Block == 9 && Layout == 4)
    {
        if constexpr (DataType == 3) {  
            INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum(ImplMode),
                LayOutTypeEnum::LAYOUT_TND, HasPse, HasAttenMask, HasDropOut,
                half, float, true, CubeFormat::NZ, HasRope);
        } else if constexpr (DataType == 2) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
            INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum(ImplMode),
                LayOutTypeEnum::LAYOUT_TND, HasPse, HasAttenMask, HasDropOut,
                bfloat16_t, float, true, CubeFormat::NZ, HasRope);
#endif
        } else if constexpr (DataType == 1) {
            INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttentionVarLenScore, ImplModeEnum(ImplMode),
                LayOutTypeEnum::LAYOUT_TND, HasPse, HasAttenMask, HasDropOut,
                float, float, true, CubeFormat::NZ, HasRope);
        }
    }

    // sameAB 模版
    if constexpr (UB0 == 3 && UB1 == 9 && Block == 9 && Layout != 4)
    {
        if constexpr (DataType == 3) {
            if constexpr (MatmulPolicyType == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_WITH_MMPOLICY(FlashAttentionScoreS1s2Bn2gs1SameAB, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), HasPse, HasAttenMask, HasDropOut,
                    half, float, CubeFormat::ND, MmPolicyType::UNSPLITK, HasRope);
            } else if constexpr (Bmm1Format == 0) {
                INVOKE_FA_GENERAL_OP_IMPL_SAMEAB(FlashAttentionScoreS1s2Bn2gs1SameAB, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), HasPse, HasAttenMask, HasDropOut,
                    half, float, CubeFormat::ND, MmPolicyType::NORMAL, HasRope);
            } else if constexpr (Bmm1Format == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ_SAMEAB(FlashAttentionScoreS1s2Bn2gs1SameAB, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), HasPse, HasAttenMask, HasDropOut,
                    half, float, CubeFormat::NZ, MmPolicyType::NORMAL, HasRope);
            }
        } else if (DataType == 2) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
            if constexpr (MatmulPolicyType == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_WITH_MMPOLICY(FlashAttentionScoreS1s2Bn2gs1SameAB, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), HasPse, HasAttenMask, HasDropOut,
                    bfloat16_t, float, CubeFormat::ND, MmPolicyType::UNSPLITK, HasRope);
            } else if constexpr (Bmm1Format == 0) {
                INVOKE_FA_GENERAL_OP_IMPL_SAMEAB(FlashAttentionScoreS1s2Bn2gs1SameAB, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), HasPse, HasAttenMask, HasDropOut,
                    bfloat16_t, float, CubeFormat::ND, MmPolicyType::NORMAL, HasRope);
            } else if constexpr (Bmm1Format == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ_SAMEAB(FlashAttentionScoreS1s2Bn2gs1SameAB, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout),  HasPse, HasAttenMask, HasDropOut,
                    bfloat16_t, float, CubeFormat::NZ, MmPolicyType::NORMAL, HasRope);
            }
#endif
        }
    }

    // s1s2 模版
    if constexpr (UB0 == 3 && UB1 == 4 && Block == 9 && Layout != 4)
    {
        if constexpr (DataType == 3) {
            if constexpr (Bmm1Format == 0) {
                INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    half, float, true, CubeFormat::ND, bool(EnableL1Reuse));
            } else {
                INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    half, float, true, CubeFormat::NZ, bool(EnableL1Reuse));
            }
        } else if constexpr (DataType == 2) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
            if constexpr (Bmm1Format == 0) {
                INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    bfloat16_t, float, true, CubeFormat::ND, bool(EnableL1Reuse));
            } else {
                INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    bfloat16_t, float, true, CubeFormat::NZ, bool(EnableL1Reuse));
            }
#endif
        } else if constexpr (DataType == 1) {
            if constexpr (Bmm1Format == 0) {
                INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    float, float, true, CubeFormat::ND, bool(EnableL1Reuse));
            } else {
                INVOKE_FA_GENERAL_OP_IMPL_BMM1NZ(FlashAttentionScoreS1s2Bn2gs1, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    float, float, true, CubeFormat::NZ, bool(EnableL1Reuse));
            }
        }
    }

    // S1模板
    if constexpr (UB0 == 3 && UB1 == 5 && Block == 9)
    {
        if constexpr (DataType == 3) {
            if constexpr (Bmm1Format == 0) {
                if constexpr (Bmm2Source > 0) {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum(ImplMode),
                        LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        half, float, true, CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ, bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
                } else {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum(ImplMode),
                        LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        half, float, true, CubeFormat::ND, TPosition::GM, CubeFormat::ND, bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
                }
            } else {
                if constexpr (Bmm2Source > 0) {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum(ImplMode),
                        LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        half, float, true, CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ, bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
                } else {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum(ImplMode),
                        LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        half, float, true, CubeFormat::NZ, TPosition::GM, CubeFormat::ND, bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
                }
            }
        } else if (DataType == 2) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
            if constexpr (Bmm1Format == 0) {
                if constexpr (Bmm2Source > 0) {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum(ImplMode),
                        LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        bfloat16_t, float, true, CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ, bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
                } else {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum(ImplMode),
                        LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        bfloat16_t, float, true, CubeFormat::ND, TPosition::GM, CubeFormat::ND, bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
                }
            } else {
                if constexpr (Bmm2Source > 0) {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum(ImplMode),
                        LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        bfloat16_t, float, true, CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ, bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
                } else {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum(ImplMode),
                        LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        bfloat16_t, float, true, CubeFormat::NZ, TPosition::GM, CubeFormat::ND, bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
                }
            }
#endif
        } else if (DataType == 1) {
            if constexpr (Bmm1Format == 0) {
                if constexpr (Bmm2Source > 0) {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum(ImplMode),
                        LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        float, float, true, CubeFormat::ND, TPosition::TSCM, CubeFormat::NZ, bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
                } else {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum(ImplMode),
                        LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        float, float, true, CubeFormat::ND, TPosition::GM, CubeFormat::ND, bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
                }
            } else {
                if constexpr (Bmm2Source > 0) {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum(ImplMode),
                        LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        float, float, true, CubeFormat::NZ, TPosition::TSCM, CubeFormat::NZ, bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
                } else {
                    INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreS1Bn2gs1, ImplModeEnum(ImplMode),
                        LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                        float, float, true, CubeFormat::NZ, TPosition::GM, CubeFormat::ND, bool(EnableL1Reuse),
                        STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
                }
            }
        }
    }

    // B模板
    if constexpr (UB0 == 9 && UB1 == 9 && Block == 0)
    {
        if constexpr (DataType == 3) {
            if constexpr (Layout == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    half, float, true, LayoutMode::BSNGD,
                    STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
            } else if constexpr (Layout == 2) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    half, float, true, LayoutMode::SBNGD,
                    STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
            } else if constexpr (Layout == 3) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    half, float, true, LayoutMode::BNGS1S2,
                    STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
            }
        } else if (DataType == 2) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
            if constexpr (Layout == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    bfloat16_t, float, true, LayoutMode::BSNGD,
                    STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
            } else if constexpr (Layout == 2) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    bfloat16_t, float, true, LayoutMode::SBNGD,
                    STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
            } else if constexpr (Layout == 3) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    bfloat16_t, float, true, LayoutMode::BNGS1S2,
                    STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
            }
#endif
        } else if (DataType == 1) {
            if constexpr (Layout == 1) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    float, float, true, LayoutMode::BSNGD,
                    STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
            } else if constexpr (Layout == 2) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    float, float, true, LayoutMode::SBNGD,
                    STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
            } else if constexpr (Layout == 3) {
                INVOKE_FA_GENERAL_OP_IMPL_BMM2NZ(FlashAttentionScoreBn2gs1s2B, ImplModeEnum(ImplMode),
                    LayOutTypeEnum(Layout), bool(HasPse), bool(HasAttenMask), bool(HasDropOut),
                    float, float, true, LayoutMode::BNGS1S2,
                    STemplateType(S1TemplateType * 16), STemplateType(S2TemplateType * 16), DTemplateType(dTemplateSize * 16));
            }
        }
    }

    return;
}
