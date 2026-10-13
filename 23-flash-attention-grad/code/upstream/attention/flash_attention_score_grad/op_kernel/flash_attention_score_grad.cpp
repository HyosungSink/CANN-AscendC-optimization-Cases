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
 * \file flash_attention_score_grad.cpp
 * \brief
 */
#ifdef KFC_L1_RESERVER_SIZE
#undef KFC_L1_RESERVER_SIZE
#define KFC_L1_RESERVER_SIZE 0 // only support Gm in and Gm out
#else
#define KFC_L1_RESERVER_SIZE 0 // only support Gm in and Gm out
#endif

#include "kernel_operator.h"
using namespace AscendC;

constexpr static const uint32_t BNGSD = 0;
constexpr static const uint32_t SBNGD = 1;
constexpr static const uint32_t BSNGD = 2;
constexpr static const uint32_t TND = 3;

#include "arch32/flash_attention_score_grad_tiling.h"
#include "arch32/flash_attention_score_grad_template_tiling_key.h"
#include "arch32/flash_attention_score_grad_constant_propagation.h"
#include "arch32/flash_attention_score_grad_empty_tensor.h"
#include "arch32/flash_attention_score_grad_post.h"
#include "arch32/flash_attention_score_grad_s1s2_bn2gs1s2.h"
#include "arch32/flash_attention_score_grad_pre.h"
#include "arch32/flash_attention_score_grad_sfmg.h"
#include "arch32/flash_attention_score_grad_s1s2_bn2.h"
#include "arch32/flash_attention_score_grad_ngs1s2_bn.h"
#include "arch32/flash_attention_score_grad_bngs1s2_b.h"
#include "arch32/flash_attention_score_grad_s1s2_bn2gs1s2_sab.h"
#include "arch32/flash_attention_score_grad_s1s2_bn2gs1s2_basic.h"
#include "arch32/flash_attention_score_grad_s1s2_bn2gs1s2_basic_det.h"

constexpr MatmulConfig MM_CFG_EXCEED = GetNormalConfig(true);
constexpr MatmulConfig MM_CFG_NORMAL = GetNormalConfig(false);
constexpr CubeFormat MM_NZ_OUT_FORMAT = CubeFormat::NZ;
constexpr CubeFormat MM_ND_OUT_FORMAT = CubeFormat::ND_ALIGN; // unused
constexpr CubeFormat MM_ND_OUT_NOALIGN = CubeFormat::ND;
constexpr uint64_t INPUT_NONE = 0;
constexpr uint64_t INPUT_EXIST = 1;
constexpr uint32_t INPUT_DISABLE = 0;
constexpr uint32_t INPUT_ENABLE = 1;

constexpr static uint32_t ND = 0;
constexpr static uint32_t NZ = 1;


#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(INPUT_TYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT, \
                                              MM2_OUT_FORMAT, TND_S1_PP)                                               \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, tiling_data_in, tiling_data);       \
        const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *__restrict tilingData = &tiling_data_in;                  \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                                       \
        const TCubeTiling *__restrict bmm2tiling = &(tilingData->mm2TilingData);                                       \
        const TCubeTiling *__restrict bmm3tiling = &(tilingData->mm3TilingData);                                       \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, true> opPre;      \
        opPre.Init(dq, dqRope, dk, dkRope, dv, drop_mask, user, tilingData, &pipeIn);                                  \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
                                                                                                                       \
        TPipe pipeBase;                                                                                                \
        FlashAttentionScoreGradS1s2Bn2gs1s2<INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,          \
                                            INPUT_LAYOUT, MM2_OUT_FORMAT, TND_S1_PP>                                   \
            op;                                                                                                        \
        REGIST_MATMUL_OBJ(&pipeBase, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm3, bmm2tiling, op.mm4,             \
                          bmm3tiling);                                                                                 \
        op.Init(key, keyRope, value, dy, query, queryRope, pse_shift, drop_mask, atten_mask, attention_in, softmax_max,\
                softmax_sum, prefix, actual_seq_qlen, actual_seq_kvlen, dq, dqRope, dk, dkRope, dv, dpse, user,        \
                tilingData, &pipeBase);                                                                                \
        op.Process();                                                                                                  \
        op.SyncALLCores();                                                                                             \
        pipeBase.Destroy();                                                                                            \
        TPipe pipePost;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, true, INPUT_LAYOUT,     \
                                    input_format>                                                                      \
            opPost;                                                                                                    \
        opPost.Init(dq, dqRope, dk, dkRope, dv, actual_seq_qlen, actual_seq_kvlen, dsink, user, tilingData, &pipePost);       \
        opPost.Process();                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_ROPE_IMPL(INPUT_TYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,          \
                                                   INPUT_LAYOUT, MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE)                  \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, tiling_data_in, tiling_data);       \
        const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *__restrict tilingData = &tiling_data_in;                  \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                                       \
        const TCubeTiling *__restrict bmm2tiling = &(tilingData->mm2TilingData);                                       \
        const TCubeTiling *__restrict bmm3tiling = &(tilingData->mm3TilingData);                                       \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, true, HAS_ROPE>   \
            opPre;                                                                                                     \
        opPre.Init(dq, dqRope, dk, dkRope, dv, drop_mask, user, tilingData, &pipeIn);                                  \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
                                                                                                                       \
        TPipe pipeBase;                                                                                                \
        FlashAttentionScoreGradS1s2Bn2gs1s2<INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,          \
                                            INPUT_LAYOUT, MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>                         \
            op;                                                                                                        \
        REGIST_MATMUL_OBJ(&pipeBase, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm3, bmm2tiling, op.mm4,             \
                          bmm3tiling);                                                                                 \
        op.Init(key, keyRope, value, dy, query, queryRope, pse_shift, drop_mask, atten_mask, attention_in, softmax_max,\
                softmax_sum, prefix, actual_seq_qlen, actual_seq_kvlen, dq, dqRope, dk, dkRope, dv, dpse, user,        \
                tilingData, &pipeBase);                                                                                \
        op.Process();                                                                                                  \
        op.SyncALLCores();                                                                                             \
        pipeBase.Destroy();                                                                                            \
        TPipe pipePost;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, true, INPUT_LAYOUT,     \
                                    input_format, HAS_ROPE>                                                            \
            opPost;                                                                                                    \
        opPost.Init(dq, dqRope, dk, dkRope, dv, actual_seq_qlen, actual_seq_kvlen, dsink, user, tilingData, &pipePost);\
        opPost.Process();                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_L1_CUSTOM_IMPL(INPUT_TYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP,             \
    MM_OUT_FORMAT, INPUT_LAYOUT, MM2_OUT_FORMAT, IS_DTM, S1TEMPLATETYPE, S2TEMPLATETYPE, DTEMPLATETYPE)                \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, tiling_data_in, tiling_data); \
        const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb *__restrict tilingData = &tiling_data_in;            \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, true> opPre;\
        opPre.Init(dq, dqRope, dk, dkRope, dv, drop_mask, user, tilingData, &pipeIn);                                  \
        opPre.Process();                                                                                               \
        pipeIn.Destroy();                                                                                              \
        if ASCEND_IS_AIV {                                                                                             \
            TPipe pipeSfmg;                                                                                            \
            FlashAttentionScoreGradSfmg<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb,        \
                INPUT_LAYOUT> opSfmg;                                                                                  \
            opSfmg.Init(dy, attention_in, actual_seq_qlen, dq, dk, dv, drop_mask, user, tilingData, &pipeSfmg);        \
            opSfmg.Process();                                                                                          \
            pipeSfmg.Destroy();                                                                                        \
        }                                                                                                              \
        TPipe pipeBase;                                                                                                \
        FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGType<INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP,           \
            MM_OUT_FORMAT, INPUT_LAYOUT, MM2_OUT_FORMAT, IS_DTM,                                                       \
            S1TEMPLATETYPE, S2TEMPLATETYPE, DTEMPLATETYPE>> op;                                                        \
        op.InitTscmBuffer(&pipeBase);                                                                                  \
        op.Init(key, keyRope, value, dy, query, queryRope, pse_shift, drop_mask, atten_mask, attention_in, softmax_max,\
                softmax_sum, sink, prefix, actual_seq_qlen, actual_seq_kvlen, dq, dqRope, dk, dkRope, dv, dpse, dsink, user,        \
                tilingData);                                                                                           \
        op.ProcessFirstMM();                                                                                           \
        op.InitBuffer(&pipeBase);                                                                                      \
        op.Process();                                                                                                  \
        op.SyncALLCores();                                                                                             \
        pipeBase.Destroy();                                                                                            \
        if ASCEND_IS_AIV {                                                                                             \
            TPipe pipePost;                                                                                            \
            constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                   \
            FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, true, INPUT_LAYOUT, input_format> opPost;\
            opPost.Init(dq, dqRope, dk, dkRope, dv, actual_seq_qlen, actual_seq_kvlen, dsink, user, tilingData, &pipePost);\
            opPost.Process();                                                                                          \
        }                                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_L1_CUSTOM_ROPE_IMPL(INPUT_TYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP,        \
    MM_OUT_FORMAT, INPUT_LAYOUT, MM2_OUT_FORMAT, IS_DTM, S1TEMPLATETYPE, S2TEMPLATETYPE, DTEMPLATETYPE, HAS_ROPE)      \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, tiling_data_in, tiling_data); \
        const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb *__restrict tilingData = &tiling_data_in;            \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, true,       \
             HAS_ROPE> opPre;                                                                                          \
        opPre.Init(dq, dqRope, dk, dkRope, dv, drop_mask, user, tilingData, &pipeIn);                                  \
        opPre.Process();                                                                                               \
        pipeIn.Destroy();                                                                                              \
        if ASCEND_IS_AIV {                                                                                             \
            TPipe pipeSfmg;                                                                                            \
            FlashAttentionScoreGradSfmg<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb,        \
                INPUT_LAYOUT> opSfmg;                                                                                  \
            opSfmg.Init(dy, attention_in, actual_seq_qlen, dq, dk, dv, drop_mask, user, tilingData, &pipeSfmg);        \
            opSfmg.Process();                                                                                          \
            pipeSfmg.Destroy();                                                                                        \
        }                                                                                                              \
        TPipe pipeBase;                                                                                                \
        FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGType<INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP,           \
            MM_OUT_FORMAT, INPUT_LAYOUT, MM2_OUT_FORMAT, IS_DTM,                                                       \
            S1TEMPLATETYPE, S2TEMPLATETYPE, DTEMPLATETYPE, HAS_ROPE>> op;                                              \
        op.InitTscmBuffer(&pipeBase);                                                                                  \
        op.Init(key, keyRope, value, dy, query, queryRope, pse_shift, drop_mask, atten_mask, attention_in, softmax_max,\
                softmax_sum, sink, prefix, actual_seq_qlen, actual_seq_kvlen, dq, dqRope, dk, dkRope, dv, dpse, dsink, user,        \
                tilingData);                                                                                     \
        op.ProcessFirstMM();                                                                                           \
        op.InitBuffer(&pipeBase);                                                                                      \
        op.Process();                                                                                                  \
        op.SyncALLCores();                                                                                             \
        pipeBase.Destroy();                                                                                            \
        if ASCEND_IS_AIV {                                                                                             \
            TPipe pipePost;                                                                                            \
            constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                   \
                FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, true, INPUT_LAYOUT, input_format, HAS_ROPE> opPost;\
            opPost.Init(dq, dqRope, dk, dkRope, dv, actual_seq_qlen, actual_seq_kvlen, dsink, user, tilingData, &pipePost);\
            opPost.Process();                                                                                          \
        }                                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(INPUT_TYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP,                       \
    MM_OUT_FORMAT, INPUT_LAYOUT, MM2_OUT_FORMAT, IS_DTM)                                                               \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, tiling_data_in, tiling_data); \
        const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb *__restrict tilingData = &tiling_data_in;            \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, true> opPre;\
        opPre.Init(dq, dqRope, dk, dkRope, dv, drop_mask, user, tilingData, &pipeIn);                                  \
        opPre.Process();                                                                                               \
        pipeIn.Destroy();                                                                                              \
        if ASCEND_IS_AIV {                                                                                             \
            TPipe pipeSfmg;                                                                                            \
            FlashAttentionScoreGradSfmg<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb,        \
                INPUT_LAYOUT> opSfmg;                                                                                  \
            opSfmg.Init(dy, attention_in, actual_seq_qlen, dq, dk, dv, drop_mask, user, tilingData, &pipeSfmg);        \
            opSfmg.Process();                                                                                          \
            pipeSfmg.Destroy();                                                                                        \
        }                                                                                                              \
        TPipe pipeBase;                                                                                                \
        FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGType<INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP,           \
            MM_OUT_FORMAT, INPUT_LAYOUT, MM2_OUT_FORMAT, IS_DTM>> op;                                                  \
        op.Init(key, keyRope, value, dy, query, queryRope, pse_shift, drop_mask, atten_mask, attention_in, softmax_max,\
                softmax_sum, sink, prefix, actual_seq_qlen, actual_seq_kvlen, dq, dqRope, dk, dkRope, dv, dpse, dsink, user,        \
                tilingData);                                                                                    \
        op.ProcessFirstMM();                                                                                           \
        op.InitBuffer(&pipeBase);                                                                                      \
        op.Process();                                                                                                  \
        op.SyncALLCores();                                                                                             \
        pipeBase.Destroy();                                                                                            \
        if ASCEND_IS_AIV {                                                                        \
            TPipe pipePost;                                                                                                \
            constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
            FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, true, INPUT_LAYOUT,input_format> opPost;\
            opPost.Init(dq, dqRope, dk, dkRope, dv, actual_seq_qlen, actual_seq_kvlen, dsink, user, tilingData, &pipePost);\
            opPost.Process();                                                                                              \
        }                                                                                                       \
    } while (0)


#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_ROPE_IMPL(INPUT_TYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP,                  \
    MM_OUT_FORMAT, INPUT_LAYOUT, MM2_OUT_FORMAT, IS_DTM, HAS_ROPE)                                                     \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, tiling_data_in, tiling_data); \
        const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb *__restrict tilingData = &tiling_data_in;            \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, true,       \
                                   HAS_ROPE> opPre;                                                                    \
        opPre.Init(dq, dqRope, dk, dkRope, dv, drop_mask, user, tilingData, &pipeIn);                                  \
        opPre.Process();                                                                                               \
        pipeIn.Destroy();                                                                                              \
        if ASCEND_IS_AIV {                                                                                             \
            TPipe pipeSfmg;                                                                                            \
            FlashAttentionScoreGradSfmg<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb,        \
                INPUT_LAYOUT> opSfmg;                                                                                  \
            opSfmg.Init(dy, attention_in, actual_seq_qlen, dq, dk, dv, drop_mask, user, tilingData, &pipeSfmg);        \
            opSfmg.Process();                                                                                          \
            pipeSfmg.Destroy();                                                                                        \
        }                                                                                                              \
        TPipe pipeBase;                                                                                                \
        FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGType<INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP,           \
            MM_OUT_FORMAT, INPUT_LAYOUT, MM2_OUT_FORMAT, IS_DTM, STemplateType::NotAligned, STemplateType::NotAligned, \
            DTemplateType::NotAligned, HAS_ROPE>> op;                                                                  \
        op.Init(key, keyRope, value, dy, query, queryRope, pse_shift, drop_mask, atten_mask, attention_in, softmax_max,\
                softmax_sum, sink, prefix, actual_seq_qlen, actual_seq_kvlen, dq, dqRope, dk, dkRope, dv, dpse, dsink, user,        \
                tilingData);                                                                                          \
        op.ProcessFirstMM();                                                                                           \
        op.InitBuffer(&pipeBase);                                                                                      \
        op.Process();                                                                                                  \
        op.SyncALLCores();                                                                                             \
        pipeBase.Destroy();                                                                                            \
        if ASCEND_IS_AIV {                                                                        \
            TPipe pipePost;                                                                                                \
            constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
            FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, true, INPUT_LAYOUT, input_format, HAS_ROPE> opPost;\
            opPost.Init(dq, dqRope, dk, dkRope, dv, actual_seq_qlen, actual_seq_kvlen, dsink, user, tilingData, &pipePost);\
            opPost.Process();                                                                                              \
        }                                                     \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(INPUT_TYPE, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,     \
                                         INPUT_LAYOUT, MM2_OUT_FORMAT)                                                 \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2, tiling_data_in, tiling_data);            \
        const FlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict tilingData = &tiling_data_in;                       \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                                       \
        const TCubeTiling *__restrict bmm31tiling = &(tilingData->mm31TilingData);                                     \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm4TilingData);                                       \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataS1s2Bn2, false> opPre;          \
        opPre.Init(dq, dqRope, dk, dkRope, dv, drop_mask, user, tilingData, &pipeIn);                                  \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeOp;                                                                                                  \
        FlashAttentionScoreGradS1s2Bn2<INPUT_TYPE, float, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG,             \
                                       DROPOUT_CFG, INPUT_LAYOUT, MM2_OUT_FORMAT, false, false>                        \
            op;                                                                                                        \
        REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.mm_12, bmm1tiling, op.mm_dkv, bmm4tiling, op.mm_dq,        \
                          bmm31tiling);                                                                                \
        op.Init(query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum,       \
                prefix, softmax_in, actual_seq_qlen, actual_seq_kvlen, attention_in, dq, dk, dv, dpse, user,           \
                tilingData, &pipeOp);                                                                                  \
        op.Process();                                                                                                  \
        pipeOp.Destroy();                                                                                              \
        TPipe pipeCast;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2, true, INPUT_LAYOUT,          \
        input_format> opCast;                                                                                          \
        opCast.Init(dq, dqRope, dk, dkRope, dv, actual_seq_qlen, actual_seq_kvlen, dsink, user, tilingData, &pipeCast);\
        opCast.Process();                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(INPUT_TYPE, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG,       \
                                                    DROPOUT_CFG, INPUT_LAYOUT, MM2_OUT_FORMAT)                         \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2, tiling_data_in, tiling_data);            \
        const FlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict tilingData = &tiling_data_in;                       \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                                       \
        const TCubeTiling *__restrict bmm31tiling = &(tilingData->mm31TilingData);                                     \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm4TilingData);                                       \
        FlashAttentionScoreGradS1s2Bn2<INPUT_TYPE, float, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG,             \
                                       DROPOUT_CFG, INPUT_LAYOUT, MM2_OUT_FORMAT, false, false>                        \
            op;                                                                                                        \
        REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm_12, bmm1tiling, op.mm_dkv, bmm4tiling, op.mm_dq,        \
                          bmm31tiling);                                                                                \
        op.Init(query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum,       \
                prefix, softmax_in, actual_seq_qlen, actual_seq_kvlen, attention_in, dq, dk, dv, dpse, user,           \
                tilingData, &pipeIn);                                                                                  \
        op.Process();                                                                                                  \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeCast;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2, true, INPUT_LAYOUT,          \
        input_format> opCast;                                                                                          \
        opCast.Init(dq, dqRope, dk, dkRope, dv, actual_seq_qlen, actual_seq_kvlen, dsink, user, tilingData, &pipeCast);\
        opCast.Process();                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2_POST2KERNEL_IMPL(INPUT_TYPE, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG,      \
                                                     DROPOUT_CFG, INPUT_LAYOUT, MM2_OUT_FORMAT)                        \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2, tiling_data_in, tiling_data);            \
        const FlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict tilingData = &tiling_data_in;                       \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                                       \
        const TCubeTiling *__restrict bmm31tiling = &(tilingData->mm31TilingData);                                     \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm4TilingData);                                       \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataS1s2Bn2, false> opPre;          \
        opPre.Init(dq, dqRope, dk, dkRope, dv, drop_mask, user, tilingData, &pipeIn);                                  \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeOp;                                                                                                  \
        FlashAttentionScoreGradS1s2Bn2<INPUT_TYPE, float, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG,             \
                                       DROPOUT_CFG, INPUT_LAYOUT, MM2_OUT_FORMAT, true, false>                         \
            op;                                                                                                        \
        REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.mm_12, bmm1tiling, op.mm_dkv, bmm4tiling, op.mm_dq,        \
                          bmm31tiling);                                                                                \
        op.Init(query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum,       \
                prefix, softmax_in, actual_seq_qlen, actual_seq_kvlen, attention_in, dq, dk, dv, dpse, user,           \
                tilingData, &pipeOp);                                                                                  \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_POST2KERNEL_IMPL(INPUT_TYPE, MM_CONFIG, CUBE_FORMAT, PSE_CFG,           \
                                                                ATTEN_MASK_CFG, DROPOUT_CFG, INPUT_LAYOUT,             \
                                                                MM2_OUT_FORMAT)                                        \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2, tiling_data_in, tiling_data);            \
        const FlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict tilingData = &tiling_data_in;                       \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                                       \
        const TCubeTiling *__restrict bmm31tiling = &(tilingData->mm31TilingData);                                     \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm4TilingData);                                       \
        FlashAttentionScoreGradS1s2Bn2<INPUT_TYPE, float, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG,             \
                                       DROPOUT_CFG, INPUT_LAYOUT, MM2_OUT_FORMAT, true, false>                         \
            op;                                                                                                        \
        REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm_12, bmm1tiling, op.mm_dkv, bmm4tiling, op.mm_dq,        \
                          bmm31tiling);                                                                                \
        op.Init(query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum,       \
                prefix, softmax_in, actual_seq_qlen, actual_seq_kvlen, attention_in, dq, dk, dv, dpse, user,           \
                tilingData, &pipeIn);                                                                                  \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROP_POSTIN_L1_IMPL(INPUT_TYPE, MM_CONFIG, CUBE_FORMAT, PSE_CFG,                \
                                                                ATTEN_MASK_CFG, DROPOUT_CFG, INPUT_LAYOUT,             \
                                                                MM2_OUT_FORMAT, L1_CUSTOM)                             \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2, tiling_data_in, tiling_data);            \
        const FlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict tilingData = &tiling_data_in;                       \
        FlashAttentionScoreGradS1s2Bn2<INPUT_TYPE, float, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG,             \
                                       DROPOUT_CFG, INPUT_LAYOUT, MM2_OUT_FORMAT, true, L1_CUSTOM>                     \
            op;                                                                                                        \
        op.Init(query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum,       \
                prefix, softmax_in, actual_seq_qlen, actual_seq_kvlen, attention_in, dq, dk, dv, dpse, user,           \
                tilingData, &pipeIn);                                                                                  \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(INPUT_TYPE, INPUT_LAYOUT, layout, MM_CONFIG, MM_OUT_FORMAT,             \
                                               MM2_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType)          \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradUbngs1s2BbTilingData, tiling_data_in, tiling_data);         \
        const FlashAttentionScoreGradUbngs1s2BbTilingData *__restrict tilingData = &tiling_data_in;                    \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradUbngs1s2BbTilingData, false> opPre;       \
        opPre.Init(dq, dqRope, dk, dkRope, dv, drop_mask, user, tilingData, &pipeIn);                                  \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeOp;                                                                                                  \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1AndMm2TilingData);                                 \
        const TCubeTiling *__restrict bmm3tiling = &(tilingData->mm31TilingData);                                      \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm32AndMm4TilingData);                                \
        FlashAttentionScoreGradUngs1s2Bb<INPUT_TYPE, float, MM_CONFIG, INPUT_LAYOUT, MM_OUT_FORMAT, MM2_OUT_FORMAT,    \
                                        s1TemplateType, s2TemplateType, dTemplateType> op;                             \
        REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm31, bmm3tiling,                      \
                          op.mm32, bmm4tiling, op.mm4, bmm4tiling);                                                    \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq,   \
                dk, dv, user, tilingData, &pipeOp);                                                                    \
        op.Process();                                                                                                  \
        pipeOp.Destroy();                                                                                              \
        TPipe pipeMuls;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradUbngs1s2BbTilingData, false,                    \
        layout, input_format> opMuls;                                                                                  \
        opMuls.Init(dq, dqRope, dk, dkRope, dv, actual_seq_qlen, actual_seq_kvlen, dsink, user, tilingData, &pipeMuls);       \
        opMuls.Process();                                                                                              \
        pipeMuls.Destroy();                                                                                            \
    } while (0)

#define INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(INPUT_TYPE, INPUT_LAYOUT, layout, MM_CONFIG, MM_OUT_FORMAT, MM2_OUT_FORMAT,  \
                                                                    s1TemplateType, s2TemplateType, dTemplateType)     \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataUngs1s2Bbn, tiling_data_in, tiling_data);         \
        const FlashAttentionScoreGradTilingDataUngs1s2Bbn *__restrict tilingData = &tiling_data_in;                    \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataUngs1s2Bbn, false> opPre;       \
        opPre.Init(dq, dqRope, dk, dkRope, dv, drop_mask, user, tilingData, &pipeIn);                                  \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeOp;                                                                                                  \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1AndMm2TilingData);                                 \
        const TCubeTiling *__restrict bmm3tiling = &(tilingData->mm31TilingData);                                      \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm32AndMm4TilingData);                                \
        FlashAttentionScoreGradUngs1s2Bbn<INPUT_TYPE, float, MM_CONFIG, true, INPUT_LAYOUT, MM_OUT_FORMAT,             \
                                          MM2_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType> op;           \
        REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm31, bmm3tiling,                      \
                          op.mm32, bmm4tiling, op.mm4, bmm4tiling);                                                    \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq,   \
                dk, dv, user, tilingData, &pipeOp);                                                                    \
        op.Process();                                                                                                  \
        pipeOp.Destroy();                                                                                              \
        TPipe pipeMuls;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataUngs1s2Bbn, false,                    \
        layout, input_format> opMuls;                                                                                  \
        opMuls.Init(dq, dqRope, dk, dkRope, dv, actual_seq_qlen, actual_seq_kvlen, dsink, user, tilingData, &pipeMuls);       \
        opMuls.Process();                                                                                              \
        pipeMuls.Destroy();                                                                                            \
    } while (0)

#define INVOKE_FAG_GENERAL_BASIC_IMPL(INPUT_TYPE)                                                                      \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionGradMlaTilingData, mla_tiling_data, tiling_data);                    \
        const FlashAttentionGradMlaTilingData *__restrict mlaTilingData = &mla_tiling_data;                            \
        FlashAttentionScoreGradBasic<INPUT_TYPE, FlashAttentionGradMlaTilingData> opMla;                               \
        pipeIn.Destroy();                                                                                              \
        opMla.Process(query, key ,value, dy, atten_mask, softmax_max, softmax_sum, attention_in,                       \
                      actual_seq_qlen, actual_seq_kvlen, dq, dk, dv, user, mlaTilingData);                             \
    } while (0)

#define INVOKE_FAG_DETERMINISTIC_BASIC_IMPL(INPUT_TYPE, SEQLEN_TYPE, DROP_ENABLE, DETERMINISTIC_ENABLE)                                                                      \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionGradBasicDetTilingData, det_tiling_data, tiling_data);                    \
        const FlashAttentionGradBasicDetTilingData *__restrict detTilingData = &det_tiling_data;                            \
        FlashAttentionScoreGradBasicDet<FAG_TYPE<INPUT_TYPE, FlashAttentionGradBasicDetTilingData, SEQLEN_TYPE, DROP_ENABLE, DETERMINISTIC_ENABLE>> opDet;                               \
        pipeIn.Destroy();                                                                                              \
        opDet.Process(query, key ,value, dy, drop_mask, atten_mask, softmax_max, softmax_sum, attention_in,                       \
                      actual_seq_qlen, actual_seq_kvlen, dq, dk, dv, user, detTilingData);                             \
    } while (0)


// implementation of kernel function
template<uint8_t UB0, uint8_t UB1, uint8_t Block, bool IsSameAB, uint8_t DataType, uint8_t Layout, uint8_t Sparse, uint8_t MatmulCfg, uint8_t Mm12IsNZOut,
    uint8_t Mm345IsNZOut, bool HasDropOut, bool HasPse, bool HasAttenMask, bool EnableL1Reuse, bool TNDS1Pingpong, uint8_t S1TemplateType,
    uint8_t S2TemplateType, uint8_t DTemplateType,bool IsDeterministic, bool HasRope>
__global__ __aicore__ void flash_attention_score_grad(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dy, __gm__ uint8_t *pse_shift,
    __gm__ uint8_t *drop_mask, __gm__ uint8_t *padding_mask, __gm__ uint8_t *atten_mask, __gm__ uint8_t *softmax_max,
    __gm__ uint8_t *softmax_sum, __gm__ uint8_t *softmax_in, __gm__ uint8_t *attention_in, __gm__ uint8_t *prefix,
    __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen, __gm__ uint8_t *q_start_idx, __gm__ uint8_t *kv_start_idx, 
    __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *deqScaleDy, __gm__ uint8_t *deqScaleO,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,  __gm__ uint8_t *sink, __gm__ uint8_t *dsScale, __gm__ uint8_t *pScale, 
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dpse,
    __gm__ uint8_t *dqRope, __gm__ uint8_t *dkRope, __gm__ uint8_t *dsink, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling_data)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipeIn;
    AscendC::SetMaskNorm();
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    REGISTER_TILING_DEFAULT(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2);

    if constexpr (UB0 == 0 && UB1 == 0 && Block == 0) {
        REGISTER_TILING_FOR_TILINGKEY("(TILING_KEY_VAR & 0x0)", FlashAttentionScoreGradTilingData);
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingData, tiling_data_in, tiling_data);
        const FlashAttentionScoreGradTilingData *__restrict empty_tensor_tiling_data = &tiling_data_in;
        #if (ORIG_DTYPE_QUERY == DT_FLOAT16)
            FlashAttentionScoreGradEmptyTensor<half> op;
            op.Init(dq, dk, dv, dpse, empty_tensor_tiling_data);
            op.Process();
        #elif (ORIG_DTYPE_QUERY == DT_FLOAT)
            FlashAttentionScoreGradEmptyTensor<float> op;
            op.Init(dq, dk, dv, dpse, empty_tensor_tiling_data);
            op.Process();
        #elif (ORIG_DTYPE_QUERY == DT_BF16)
            FlashAttentionScoreGradEmptyTensor<bfloat16_t> op;
            op.Init(dq, dk, dv, dpse, empty_tensor_tiling_data);
            op.Process();
        #endif
        return;
    }

    constexpr CubeFormat mm12Format = bool(Mm12IsNZOut) ? MM_NZ_OUT_FORMAT : MM_ND_OUT_NOALIGN;
    constexpr CubeFormat mm345Format = bool(Mm345IsNZOut) ? MM_NZ_OUT_FORMAT : MM_ND_OUT_NOALIGN;
    
    constexpr MatmulConfig mock_matmul_cfg = (MatmulCfg == 1) ? MM_CFG_EXCEED : MM_CFG_NORMAL;

    constexpr uint8_t mock_layout =
        Layout == 0 ? 2 :
        Layout == 2 ? 0 :
        Layout;
    
    #if (ORIG_DTYPE_QUERY == DT_FLOAT16)
        if constexpr (UB0 == 4 && UB1 == 3 && Block == 4 && IsSameAB == 1) { // sameab
            REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR & 0x1FFF = 0x1434", FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb);
            INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(half, HasAttenMask, HasPse, HasDropOut,
                                                         mm12Format, mock_layout, mm345Format, IsDeterministic);
        } else if constexpr (UB0 == 4 && UB1 == 3 && Block == 4 && IsSameAB == 0) { // s1s2
            INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, HasAttenMask, HasPse, HasDropOut,
                                            mm12Format, mock_layout, mm345Format, TNDS1Pingpong);
        } else if constexpr (UB0 == 4 && UB1 == 3 && Block == 1) { // bn2
            REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR & 0xFFF = 0x134", FlashAttentionScoreGradTilingDataS1s2Bn2);
            if constexpr (Layout == 3) {
                if constexpr (HasDropOut) {
                    if constexpr (MatmulCfg == 1) {
                        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, mm12Format, HasPse, HasAttenMask, INPUT_EXIST,
                                        TND, mm345Format);
                    } else {
                        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, mm12Format, HasPse, HasAttenMask, INPUT_EXIST,
                                        TND, mm345Format);
                    }
                } else {
                    if constexpr (MatmulCfg == 1) {
                        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, mm12Format, HasPse, HasAttenMask, INPUT_NONE,
                                        TND, mm345Format);
                    } else {
                        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, mm12Format, HasPse, HasAttenMask, INPUT_NONE,
                                        TND, mm345Format);
                    }
                }
            } else {
                if constexpr (HasDropOut) {
                    if constexpr (MatmulCfg == 1) {
                        INVOKE_FAG_GENERAL_S1S2_BN2_POST2KERNEL_IMPL(half, MM_CFG_EXCEED, mm12Format, HasPse, HasAttenMask,
                                            INPUT_EXIST, mock_layout, mm345Format);
                    } else {
                        INVOKE_FAG_GENERAL_S1S2_BN2_POST2KERNEL_IMPL(half, MM_CFG_NORMAL, mm12Format, HasPse, HasAttenMask,
                                            INPUT_EXIST, mock_layout, mm345Format);
                    }

                } else {
                    if constexpr (MatmulCfg == 1) {
                        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_POST2KERNEL_IMPL(half, MM_CFG_EXCEED, mm12Format, HasPse, HasAttenMask,
                                                                            INPUT_NONE, mock_layout, mm345Format);
                    } else {
                        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_POST2KERNEL_IMPL(half, MM_CFG_NORMAL, mm12Format, HasPse, HasAttenMask,
                                                                            INPUT_NONE, mock_layout, mm345Format);
                    }
                }
            }
        } else if constexpr (UB0 == 9 && UB1 == 9 && Block == 1) { // bn
            REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR & 0xFFF = 0x199", FlashAttentionScoreGradTilingDataUngs1s2Bbn);
            
            if constexpr (Layout == 0) {
                if constexpr (MatmulCfg == 1) {
                    INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::BSNGD, BSNGD, MM_CFG_EXCEED,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                } else {
                    INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                }
            } else if constexpr (Layout == 1) {
                if constexpr (MatmulCfg == 1) {
                    INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                } else {
                    INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                }
            } else if constexpr (Layout == 2) {
                if constexpr (MatmulCfg == 1) {
                    INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::BNGS1S2, BNGSD, MM_CFG_EXCEED,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                } else {
                    INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(half, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                }
            }
        } else if constexpr (UB0 == 9 && UB1 == 9 && Block == 0) { // b
            REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR & 0xFFF = 0x099", FlashAttentionScoreGradUbngs1s2BbTilingData);

            if constexpr (Layout == 0 || Layout == 3) {
                if constexpr (MatmulCfg == 1) {
                    INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BSNGD, BSNGD, MM_CFG_EXCEED,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                } else {
                    INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                }
            } else if constexpr (Layout == 1) {
                if constexpr (MatmulCfg == 1) {
                    INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                } else {
                    INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                }
            } else if constexpr (Layout == 2) {
                if constexpr (MatmulCfg == 1) {
                    INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BNGS1S2, BNGSD, MM_CFG_EXCEED,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                } else {
                    INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                }
            }
        } else if constexpr (UB0 == 9 && UB1 == 9 && Block == 9 && IsDeterministic == 1) { // basic det
            REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR & 0xFFF = 0x1999", FlashAttentionGradBasicDetTilingData);
            INVOKE_FAG_DETERMINISTIC_BASIC_IMPL(half, int64_t, false, true);
        } else if constexpr (UB0 == 9 && UB1 == 9 && Block == 9) { // basic
            REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR & 0xFFF = 0x999", FlashAttentionGradMlaTilingData);
            INVOKE_FAG_GENERAL_BASIC_IMPL(half);
        }
    #endif

    #if (ORIG_DTYPE_QUERY == DT_BF16)
        if constexpr (UB0 == 4 && UB1 == 3 && Block == 4 && IsSameAB == 1) { // sameab
            REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR & 0x1FFF = 0x1434", FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb);
            
            // sameab custom condition
            if constexpr (HasRope) {
                if constexpr(DTemplateType == 0) {
                    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_ROPE_IMPL(bfloat16_t, HasAttenMask, HasPse, HasDropOut,
                                                                mm12Format, mock_layout, mm345Format, IsDeterministic, INPUT_ENABLE);
                } else if constexpr (DTemplateType == 5) {
                    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_L1_CUSTOM_ROPE_IMPL(bfloat16_t, HasAttenMask, HasPse, HasDropOut, mm12Format, mock_layout,
                                        mm345Format, IsDeterministic, STemplateType::Aligned512, STemplateType::Aligned512, DTemplateType::Aligned128, INPUT_ENABLE);
                } else if constexpr (DTemplateType == 6) {
                    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_L1_CUSTOM_ROPE_IMPL(bfloat16_t, HasAttenMask, HasPse, HasDropOut, mm12Format, mock_layout,
                                        mm345Format, IsDeterministic, STemplateType::Aligned512, STemplateType::Aligned512, DTemplateType::Aligned192, INPUT_ENABLE);
                }
            } else {
                if constexpr(DTemplateType == 0) {
                    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL(bfloat16_t, HasAttenMask, HasPse, HasDropOut,
                                                                mm12Format, mock_layout, mm345Format, IsDeterministic);
                } else if constexpr(DTemplateType == 1) {
                    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_L1_CUSTOM_IMPL(bfloat16_t, HasAttenMask, HasPse, HasDropOut, mm12Format, mock_layout,
                                        mm345Format, IsDeterministic, STemplateType::Aligned512, STemplateType::Aligned512, DTemplateType::Aligned64);
                } else if constexpr (DTemplateType == 5) {
                    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_L1_CUSTOM_IMPL(bfloat16_t, HasAttenMask, HasPse, HasDropOut, mm12Format, mock_layout,
                                        mm345Format, IsDeterministic, STemplateType::Aligned512, STemplateType::Aligned512, DTemplateType::Aligned128);
                } else if constexpr (DTemplateType == 6) {
                    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_L1_CUSTOM_IMPL(bfloat16_t, HasAttenMask, HasPse, HasDropOut, mm12Format, mock_layout,
                                        mm345Format, IsDeterministic, STemplateType::Aligned512, STemplateType::Aligned512, DTemplateType::Aligned192);
                }
            }
        } else if constexpr (UB0 == 4 && UB1 == 3 && Block == 4 && IsSameAB == 0) { // s1/s2
            if constexpr (HasRope) {
                INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_ROPE_IMPL(bfloat16_t, HasAttenMask, HasPse, HasDropOut,
                                    mm12Format, mock_layout, mm345Format, TNDS1Pingpong, INPUT_ENABLE);
            } else {
                INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, HasAttenMask, HasPse, HasDropOut,
                                    mm12Format, mock_layout, mm345Format, TNDS1Pingpong);
            }
        } else if constexpr (UB0 == 4 && UB1 == 3 && Block == 1) { // bn2
            REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR & 0xFFF = 0x134", FlashAttentionScoreGradTilingDataS1s2Bn2);

            if constexpr (Layout == 3) {
                if constexpr (HasDropOut) {
                    if constexpr (MatmulCfg == 1) {
                        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, mm12Format, HasPse, HasAttenMask, INPUT_EXIST,
                                        TND, mm345Format);
                    } else {
                        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, mm12Format, HasPse, HasAttenMask, INPUT_EXIST,
                                        TND, mm345Format);
                    }
                } else {
                    if constexpr (MatmulCfg == 1) {
                        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, mm12Format, HasPse, HasAttenMask, INPUT_NONE,
                                        TND, mm345Format);
                    } else {
                        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, mm12Format, HasPse, HasAttenMask, INPUT_NONE,
                                        TND, mm345Format);
                    }
                }
            } else {
                if constexpr (EnableL1Reuse) {
                    if constexpr (MatmulCfg == 1) {
                        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROP_POSTIN_L1_IMPL(bfloat16_t, MM_CFG_EXCEED, mm12Format, HasPse, HasAttenMask,
                                                HasDropOut, mock_layout, mm345Format, INPUT_ENABLE);
                    } else {
                        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROP_POSTIN_L1_IMPL(bfloat16_t, MM_CFG_NORMAL, mm12Format, HasPse, HasAttenMask,
                                                HasDropOut, mock_layout, mm345Format, INPUT_ENABLE);
                    }
                } else {
                    if constexpr (HasDropOut) {
                        if constexpr (MatmulCfg == 1) {
                            INVOKE_FAG_GENERAL_S1S2_BN2_POST2KERNEL_IMPL(bfloat16_t, MM_CFG_EXCEED, mm12Format, HasPse, HasAttenMask,
                                                INPUT_EXIST, mock_layout, mm345Format);
                        } else {
                            INVOKE_FAG_GENERAL_S1S2_BN2_POST2KERNEL_IMPL(bfloat16_t, MM_CFG_NORMAL, mm12Format, HasPse, HasAttenMask,
                                                INPUT_EXIST, mock_layout, mm345Format);
                        }

                    } else {
                        if constexpr (MatmulCfg == 1) {
                            INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_POST2KERNEL_IMPL(bfloat16_t, MM_CFG_EXCEED, mm12Format, HasPse, HasAttenMask,
                                                                                INPUT_NONE, mock_layout, mm345Format);
                        } else {
                            INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_POST2KERNEL_IMPL(bfloat16_t, MM_CFG_NORMAL, mm12Format, HasPse, HasAttenMask,
                                                                                INPUT_NONE, mock_layout, mm345Format);
                        }
                    }
                }
            }

        } else if constexpr (UB0 == 9 && UB1 == 9 && Block == 1) { // bn
            REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR & 0xFFF = 0x199", FlashAttentionScoreGradTilingDataUngs1s2Bbn);
            
            if constexpr (Layout == 0) {
                if constexpr (MatmulCfg == 1) {
                    INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_EXCEED,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                } else {
                    // BSNGD bfloat16_t exists extra 5 TilingKey which should be dealed
                    if constexpr (S1TemplateType == 5) {
                        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned30, S2TemplateType::NotAligned30, DTemplateType::NotAligned72);
                    } else if constexpr (S1TemplateType == 4) {
                        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                        mm12Format, mm345Format, S1TemplateType::Aligned32, S2TemplateType::Aligned32, DTemplateType::NotAligned72);
                    } else if constexpr (S1TemplateType == 1) {
                        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                        mm12Format, mm345Format, S1TemplateType::Aligned48, S2TemplateType::Aligned48, DTemplateType::NotAligned72);
                    } else if constexpr (S1TemplateType == 3) {
                        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned60, S2TemplateType::NotAligned60, DTemplateType::NotAligned72);
                    } else if constexpr (S1TemplateType == 2) {
                        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                        mm12Format, mm345Format, S1TemplateType::Aligned64, S2TemplateType::Aligned64, DTemplateType::NotAligned72);
                    } else {
                        INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                    }
                }
            } else if constexpr (Layout == 1) {
                if constexpr (MatmulCfg == 1) {
                    INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                } else {
                    INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                }
            } else if constexpr (Layout == 2) {
                if constexpr (MatmulCfg == 1) {
                    INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::BNGS1S2, BNGSD, MM_CFG_EXCEED,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                } else {
                    INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(bfloat16_t, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                }
            }
        } else if constexpr (UB0 == 9 && UB1 == 9 && Block == 0) { // b
            REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR & 0xFFF = 0x099", FlashAttentionScoreGradUbngs1s2BbTilingData);

            if constexpr (Layout == 0 || Layout == 3) {
                if constexpr (MatmulCfg == 1) {
                    INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_EXCEED,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                } else {
                    // BSNGD bfloat16_t exists extra 3 TilingKey which should be dealed
                    if constexpr (S1TemplateType == 3 && DTemplateType == 3) {
                        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                            mm12Format, mm345Format, S1TemplateType::NotAligned15, S2TemplateType::NotAligned15, DTemplateType::NotAligned72);
                    } else if constexpr (S1TemplateType == 1 && DTemplateType == 3) {
                        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                            mm12Format, mm345Format, S1TemplateType::Aligned16, S2TemplateType::Aligned16, DTemplateType::NotAligned72);
                    } else if constexpr (S1TemplateType == 1 && DTemplateType == 4) {
                        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                            mm12Format, mm345Format, S1TemplateType::Aligned16, S2TemplateType::Aligned16, DTemplateType::NotAligned88);
                    } else {
                        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                            mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                    }
                }
            } else if constexpr (Layout == 1) {
                if constexpr (MatmulCfg == 1) {
                    INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_EXCEED,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                } else {
                    INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::SBNGD, SBNGD, MM_CFG_NORMAL,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                }
            } else if constexpr (Layout == 2) {
                if constexpr (MatmulCfg == 1) {
                    INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BNGS1S2, BNGSD, MM_CFG_EXCEED,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                } else {
                    INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BNGS1S2, BNGSD, MM_CFG_NORMAL,
                                        mm12Format, mm345Format, S1TemplateType::NotAligned, S2TemplateType::NotAligned, DTemplateType::NotAligned);
                }
            }
        } else if constexpr (UB0 == 9 && UB1 == 9 && Block == 9 && IsDeterministic == 1) { // basic det
            REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR & 0xFFF = 0x1999", FlashAttentionGradBasicDetTilingData);
            INVOKE_FAG_DETERMINISTIC_BASIC_IMPL(bfloat16_t, int64_t, false, true);
        } else if constexpr (UB0 == 9 && UB1 == 9 && Block == 9) { // basic
            REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR & 0xFFF = 0x999", FlashAttentionGradMlaTilingData);
            INVOKE_FAG_GENERAL_BASIC_IMPL(bfloat16_t);
        }
    #endif

    #if (ORIG_DTYPE_QUERY == DT_FLOAT)
        if constexpr (UB0 == 4 && UB1 == 3 && Block == 4 && IsSameAB == 0) { // s1s2
            INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, HasAttenMask, HasPse, HasDropOut,
                                            mm12Format, mock_layout, mm345Format, TNDS1Pingpong);
        } else if constexpr (UB0 == 4 && UB1 == 3 && Block == 1) { // bn2
            REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR & 0xFFF = 0x134", FlashAttentionScoreGradTilingDataS1s2Bn2);
            if constexpr (Layout == 3) {
                if constexpr (HasDropOut) {
                    if constexpr (MatmulCfg == 1) {
                        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, mm12Format, HasPse, HasAttenMask, INPUT_EXIST,
                                        TND, mm345Format);
                    } else {
                        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, mm12Format, HasPse, HasAttenMask, INPUT_EXIST,
                                        TND, mm345Format);
                    }
                } else {
                    if constexpr (MatmulCfg == 1) {
                        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, mm12Format, HasPse, HasAttenMask, INPUT_NONE,
                                        TND, mm345Format);
                    } else {
                        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, mm12Format, HasPse, HasAttenMask, INPUT_NONE,
                                        TND, mm345Format);
                    }
                }
            } else {
                if constexpr (EnableL1Reuse) {
                    if constexpr (MatmulCfg == 1) {
                        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROP_POSTIN_L1_IMPL(float, MM_CFG_EXCEED, mm12Format, HasPse, HasAttenMask,
                                                HasDropOut, mock_layout, mm345Format, INPUT_ENABLE);
                    } else {
                        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROP_POSTIN_L1_IMPL(float, MM_CFG_NORMAL, mm12Format, HasPse, HasAttenMask,
                                                HasDropOut, mock_layout, mm345Format, INPUT_ENABLE);
                    }
                } else {
                    if constexpr (HasDropOut) {
                        if constexpr (MatmulCfg == 1) {
                            INVOKE_FAG_GENERAL_S1S2_BN2_POST2KERNEL_IMPL(float, MM_CFG_EXCEED, mm12Format, HasPse, HasAttenMask,
                                                INPUT_EXIST, mock_layout, mm345Format);
                        } else {
                            INVOKE_FAG_GENERAL_S1S2_BN2_POST2KERNEL_IMPL(float, MM_CFG_NORMAL, mm12Format, HasPse, HasAttenMask,
                                                INPUT_EXIST, mock_layout, mm345Format);
                        }

                    } else {
                        if constexpr (MatmulCfg == 1) {
                            INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_POST2KERNEL_IMPL(float, MM_CFG_EXCEED, mm12Format, HasPse, HasAttenMask,
                                                                                INPUT_NONE, mock_layout, mm345Format);
                        } else {
                            INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_POST2KERNEL_IMPL(float, MM_CFG_NORMAL, mm12Format, HasPse, HasAttenMask,
                                                                                INPUT_NONE, mock_layout, mm345Format);
                        }
                    }
                }
            }
        } else if constexpr (UB0 == 9 && UB1 == 9 && Block == 1) { // bn
            REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR & 0xFFF = 0x199", FlashAttentionScoreGradTilingDataUngs1s2Bbn);
            INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL(float, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                    MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN, S1TemplateType::NotAligned,
                                    S2TemplateType::NotAligned, DTemplateType::NotAligned);
        } else if constexpr (UB0 == 9 && UB1 == 9 && Block == 0) { // b
            INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(float, LayoutMode::BSNGD, BSNGD, MM_CFG_NORMAL,
                                        MM_ND_OUT_NOALIGN, MM_ND_OUT_NOALIGN, S1TemplateType::NotAligned,
                                    S2TemplateType::NotAligned, DTemplateType::NotAligned);
        }
    #endif

    return;
}