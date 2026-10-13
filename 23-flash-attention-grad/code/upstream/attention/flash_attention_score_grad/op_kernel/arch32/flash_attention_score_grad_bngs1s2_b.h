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
 * \file flash_attention_score_grad_bngs1s2_b.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_BNGS1S2_B_H_
#define FLASH_ATTENTION_SCORE_GRAD_BNGS1S2_B_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "pse.h"
#include "dropmask.h"

constexpr uint32_t DROPOUT4BIT_LEN = 16;

using matmul::Matmul;
using matmul::MatmulType;

__aicore__ const constexpr MatmulConfig GetMm1Cfg41(S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType) {
    if (dTemplateType == DTemplateType::NotAligned72) {
        if (s1TemplateType == S1TemplateType::NotAligned15 && s2TemplateType == S2TemplateType::NotAligned15) {
            // FAG_CP_CFG_41_D72_S1S2_15_MM1_MM2;
            return MatmulConfig{true, false, false, 16, 16, 80, false, false, false, false,
                                0, 15, 15, 72, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        } else if (s1TemplateType == S1TemplateType::Aligned16 && s2TemplateType == S2TemplateType::Aligned16) {
            // FAG_CP_CFG_41_D72_S1S2_16_MM1_MM2;
            return MatmulConfig{true, false, false, 16, 16, 80, false, false, false, false,
                                0, 16, 16, 72, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        }
    } else if (dTemplateType == DTemplateType::NotAligned88) {
        if (s1TemplateType == S1TemplateType::Aligned16 && s2TemplateType == S2TemplateType::Aligned16) {
            // FAG_CP_CFG_41_D88_S1S2_16_MM1_MM2;
            return MatmulConfig{true, false, false, 16, 16, 96, false, false, false, false,
                                0, 16, 16, 88, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        }
    }
    return FAG_CP_CFG_31_41_DEFAULT;
}

__aicore__ const constexpr MatmulConfig GetMm2Cfg41(S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType) {
    if (dTemplateType == DTemplateType::NotAligned72) {
        if (s1TemplateType == S1TemplateType::NotAligned15 && s2TemplateType == S2TemplateType::NotAligned15) {
            // FAG_CP_CFG_41_D72_S1S2_15_MM31;
            return MatmulConfig{true, false, false, 16, 80, 16, false, false, false, false,
                                0, 15, 72, 15, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        } else if (s1TemplateType == S1TemplateType::Aligned16 && s2TemplateType == S2TemplateType::Aligned16) {
            // FAG_CP_CFG_41_D72_S1S2_16_MM31;
            return MatmulConfig{true, false, false, 16, 80, 16, false, false, false, false,
                                0, 16, 72, 16, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        }
    } else if (dTemplateType == DTemplateType::NotAligned88) {
        if (s1TemplateType == S1TemplateType::Aligned16 && s2TemplateType == S2TemplateType::Aligned16) {
            // FAG_CP_CFG_41_D88_S1S2_16_MM31;
            return MatmulConfig{true, false, false, 16, 96, 16, false, false, false, false,
                                0, 16, 88, 16, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        }
    }
    return FAG_CP_CFG_31_41_DEFAULT;
}

__aicore__ const constexpr MatmulConfig GetMm3Cfg41(S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType) {
    if (dTemplateType == DTemplateType::NotAligned72) {
        if (s1TemplateType == S1TemplateType::NotAligned15 && s2TemplateType == S2TemplateType::NotAligned15) {
            // FAG_CP_CFG_41_D72_S1S2_15_MM32_MM4;
            return MatmulConfig{true, false, false, 16, 80, 16, false, false, false, false,
                                0, 15, 72, 15, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        } else if (s1TemplateType == S1TemplateType::Aligned16 && s2TemplateType == S2TemplateType::Aligned16) {
            // FAG_CP_CFG_41_D72_S1S2_16_MM32_MM4;
            return MatmulConfig{true, false, false, 16, 80, 16, false, false, false, false,
                                0, 16, 72, 16, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        }
    } else if (dTemplateType == DTemplateType::NotAligned88) {
        if (s1TemplateType == S1TemplateType::Aligned16 && s2TemplateType == S2TemplateType::Aligned16) {
            // FAG_CP_CFG_41_D88_S1S2_16_MM32_MM4;
            return MatmulConfig{true, false, false, 16, 96, 16, false, false, false, false,
                                0, 16, 88, 16, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        }
    }
    return FAG_CP_CFG_31_41_DEFAULT;
}

// T1 for data, T2 for vecClc
template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout = LayoutMode::BNGS1S2,
          const CubeFormat MMPre_OUT_FORMAT = CubeFormat::ND, const CubeFormat MMNext_OUT_FORMAT = CubeFormat::ND,
          S1TemplateType s1TemplateType = S1TemplateType::NotAligned,
          S2TemplateType s2TemplateType = S2TemplateType::NotAligned,
          DTemplateType dTemplateType = DTemplateType::NotAligned>
class FlashAttentionScoreGradUngs1s2Bb {
public:
    __aicore__ inline FlashAttentionScoreGradUngs1s2Bb(){};
    __aicore__ inline void Init(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query,
                                __gm__ uint8_t *pse_shift, __gm__ uint8_t *drop_mask, __gm__ uint8_t *atten_mask,
                                __gm__ uint8_t *forward_res, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,
                                __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *workspace,
                                const FlashAttentionScoreGradUbngs1s2BbTilingData *__restrict ordTilingData,
                                TPipe *pipe_in);

    __aicore__ inline void Process();

    constexpr static MatmulConfig MM_1_2_CONFIG41 =
        GetMm1Cfg41(s1TemplateType, s2TemplateType, dTemplateType);
    constexpr static MatmulConfig MM_31_CONFIG41 =
        GetMm2Cfg41(s1TemplateType, s2TemplateType, dTemplateType);
    constexpr static MatmulConfig MM_32_4_CONFIG41 =
        GetMm3Cfg41(s1TemplateType, s2TemplateType, dTemplateType);

    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    using GmT1TrueLayout = MatmulType<TPosition::GM, CubeFormat::ND, T1, true, layout>;
    using GmT1FalseLayout = MatmulType<TPosition::GM, CubeFormat::ND, T1, false, layout>;
    using GmT1TrueBNSS = MatmulType<TPosition::GM, CubeFormat::ND, T1, true, LayoutMode::BNGS1S2>;
    using GmT1FalseBNSS = MatmulType<TPosition::GM, CubeFormat::ND, T1, false, LayoutMode::BNGS1S2>;

    using GmT2FalseBNSS = MatmulType<TPosition::GM, MMPre_OUT_FORMAT, T2, false, LayoutMode::BNGS1S2>;
    using GmT2FalseLayout = MatmulType<TPosition::GM, CubeFormat::ND, T2, false, layout>;
    using GmT2FalseNzBNSS = MatmulType<TPosition::GM, MMNext_OUT_FORMAT, T2, false, LayoutMode::BNGS1S2>;

    // matmul 1 2 config
    constexpr MatmulApiStaticTiling static CFG_NORM_TILING_MM1 = GetMatmulApiTiling<GmT1FalseLayout, GmT1TrueLayout, GmT2FalseBNSS, biasType>(MM_1_2_CONFIG41);
    Matmul<GmT1FalseLayout, GmT1TrueLayout, GmT2FalseBNSS, biasType, CFG_NORM_TILING_MM1> mm1;

    // matmul 31 config
    constexpr MatmulApiStaticTiling static CFG_NORM_TILING_MM31 = GetMatmulApiTiling<GmT1FalseBNSS, GmT1FalseLayout, GmT2FalseNzBNSS, biasType>(MM_31_CONFIG41);
    using modeTypeMmDq = typename AscendC::Conditional<
        (MMNext_OUT_FORMAT == CubeFormat::NZ),
        Matmul<GmT1FalseBNSS, GmT1FalseLayout, GmT2FalseNzBNSS, biasType, CFG_NORM_TILING_MM31>,
        Matmul<GmT1FalseBNSS, GmT1FalseLayout, GmT2FalseLayout, biasType, CFG_NORM_TILING_MM31>>::type;
    modeTypeMmDq mm31;

    // matmul 32 config
    constexpr MatmulApiStaticTiling static CFG_NORM_TILING_MM32 = GetMatmulApiTiling<GmT1TrueBNSS, GmT1FalseLayout, GmT2FalseNzBNSS, biasType>(MM_32_4_CONFIG41);
    using modeTypeMmDk = typename AscendC::Conditional<
        (MMNext_OUT_FORMAT == CubeFormat::NZ),
        Matmul<GmT1TrueBNSS, GmT1FalseLayout, GmT2FalseNzBNSS, biasType, CFG_NORM_TILING_MM32>,
        Matmul<GmT1TrueBNSS, GmT1FalseLayout, GmT2FalseLayout, biasType, CFG_NORM_TILING_MM32>>::type;
    modeTypeMmDk mm32;

    // matmul 4 config
    constexpr MatmulApiStaticTiling static CFG_NORM_TILING_MM4 = GetMatmulApiTiling<GmT1TrueBNSS, GmT1FalseLayout, GmT1FalseLayout, biasType>(MM_32_4_CONFIG41);
    Matmul<GmT1TrueBNSS, GmT1FalseLayout, GmT1FalseLayout, biasType, CFG_NORM_TILING_MM4> mm4;

protected:
    /* define the que */
    TQue<QuePosition::VECIN, 1> vecInQue1;
    TQue<QuePosition::VECIN, 1> vecInQue2;
    TQue<QuePosition::VECIN, 1> vecClc1;
    TQue<QuePosition::VECIN, 1> vecClc2;
    TQue<QuePosition::VECOUT, 1> vecOutQue1;
    TQue<QuePosition::VECIN, 1> softmaxGradQue;
    TQue<QuePosition::VECIN, 1> dropoutQue;
    TQue<QuePosition::VECIN, 1> maxSumQue;

    const FlashAttentionScoreGradUbngs1s2BbTilingData *__restrict ordTilingData_;

    TPipe *pipe;

    GlobalTensor<uint8_t> attenMaskU8Gm;
    GlobalTensor<T1> keyGm, valueGm, dxGm, queryGm, attenMaskGm, forwardResGm, pseGm;
    GlobalTensor<T1> dqGm, dkGm, dvGm;
    GlobalTensor<T2> dqWorkspaceGm, dkWorkspaceGm;

    GlobalTensor<float> softmaxMaxGm, softmaxSumGm;
    GlobalTensor<T2> workspaceGm;
    GlobalTensor<int32_t> syncGlobal;
    GlobalTensor<uint8_t> dropoutWorkspaceGm, dropMaskGm;

    GlobalTensor<T1> dropWorkspaceGm, mulWorkspaceGm;

    GlobalTensor<float> matmulResultBuffer1;
    GlobalTensor<float> matmulResultBuffer2;

    PseInfo pseInfo = {0};

    int64_t b;
    int64_t n;
    int64_t g;
    int64_t sQ;
    int64_t pseSq;
    uint32_t existPse;
    uint32_t pseShapeType;
    int64_t sKV;
    int64_t sKVAlign;
    int64_t sKVAlignByte;
    int64_t hQ;
    int64_t hKV;
    int64_t d;
    int64_t originalDAlign;
    int64_t attenMaskDimS2;
    float scaleValue;
    float keepProb;
    int64_t preTokens;
    int64_t nextTokens;
    int64_t headNum;
    uint32_t inputLayout; // 0:BSH 1:SBH 2:BNSD 3:BSND
    int64_t preTokensBlocks;
    int64_t nextTokensBlocks;
    uint32_t inputDType;
    uint32_t inputDTypeSize;
    uint32_t vecCalcDTypeSize;
    uint32_t hasAttenMask;
    uint32_t attenMaskShapeType;

    int64_t sKVAlignSize;
    int64_t bOut;
    int64_t apiClcQueueSize;
    int64_t usedCoreNum;

    int64_t bIn;
    uint32_t singleCoreBatchRange;
    uint32_t bCvInner;
    uint32_t bCvRatio;
    uint32_t syncLen;
    int64_t mm1WorkspaceLen;
    int64_t mm2WorkspaceLen;
    int64_t dqWorkspaceLen;
    int64_t dkWorkspaceLen;
    int64_t dropGmWorkspaceLen;
    int64_t mulGmWorkspaceLen;
    int64_t innerTmpBufSize;
    int64_t vecQueIn1Size;
    int64_t clcDInner;
    int64_t dSize;
    int64_t dInnerTail;
    int64_t dInnerTailAlign;

    int64_t subRange;
    int64_t subMask;
    int64_t subMaskTail;
    int64_t sKVAlignBlockNum;
    int64_t rightPadding;
    int64_t dstStride;

    int64_t innerReduceNum;
    int32_t innerMatResNum;
    int64_t maskInputNum;
    int64_t pseInputNum;

    uint32_t innerMatOutShape[2];
    uint32_t innerReduceShape[2];
    bool isDrop;
    int64_t dropoutWorkspaceLen;
    int64_t mBlockIdx;
    int64_t bInNGSq;

    int64_t pingPongDropOffset;
    int64_t pingPongMulOffset;
    uint32_t pingpongIdx = 1;
    uint32_t lastPingpongIdx = 0;
    uint32_t lastBCvInner = 0;

    DropMaskInfo dropMaskInfo = {0};

    constexpr static int64_t BEST_BASIC_BLOCK_SIZE = 64 * 128 * 4;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static uint32_t BIT_SIZE = 8;
    constexpr static int64_t PSE_BNSS = 0;
    constexpr static int64_t PSE_BN1S = 1;
    constexpr static int64_t PSE_1NSS = 2;
    // for nz
    constexpr static int64_t C0_SIZE = 16;
    constexpr static int64_t VEC_REPEAT = 8;
    constexpr static uint32_t CAL_BLOCK_NUM = 32 / sizeof(T2);
    constexpr static int64_t PER_COPY_MAX_NUM = 255;

    __aicore__ inline void FrontCompute(const int64_t &batchSqLoopOffset, const int64_t &batchSkvLoopOffset,
                                        const int64_t &dropMaskOffset, const int64_t &bCvMmOffset,
                                        const int64_t &bCvNzMmOffset, const int64_t &bCvIndex,
                                        const int64_t &currentBatchRange);

    __aicore__ inline void ReCompute(const int64_t &batchSqCLoopOffset, const int64_t &batchSkvCLoopOffset,
                                     const int64_t &batchSqLoopOffset, const int64_t &batchSkvLoopOffset,
                                     const int64_t &attenMaskOffset, const int64_t &dropMaskOffset,
                                     const int64_t &batchSoftmaxInputOffset, const int64_t &bCvMmOffset,
                                     const int64_t &bCvNzMmOffset, const int64_t &bCvIndex,
                                     const bool isCvTail, const int64_t &currentBatchRange);

    __aicore__ inline void ClcSub(LocalTensor<T2> &frontResInner, LocalTensor<T2> &dpResInner,
                                  LocalTensor<T2> &sftFrontResInner);

    __aicore__ inline void SetReClcShape(LocalTensor<T2> &mulResInner, LocalTensor<float> &maxInner,
                                         LocalTensor<float> &sumInner, LocalTensor<T2> &dvDropResInner);

    __aicore__ inline void CopyInSoftMax(LocalTensor<float> &maxInner, const GlobalTensor<float> &softmaxMaxGmIn,
                                         LocalTensor<float> &sumInner, const GlobalTensor<float> &softmaxSumGmIn);

    __aicore__ inline bool CopyInAttenMask(const int64_t &attenMaskOffset);

    __aicore__ inline void CalcAttenMaskOffset(int64_t &attenMaskOffset, const int64_t delta);

    __aicore__ inline void CalcCausalAttenMaskOffset(int64_t &attenMaskOffset, const int64_t delta);

    __aicore__ inline void ClcAttenMask(LocalTensor<T2> &mmResUb);

    __aicore__ inline void ClcSoftMax(LocalTensor<T2> &softmaxResInner, LocalTensor<T2> &reMatmulResInner,
                                      LocalTensor<float> &maxInner, LocalTensor<float> &sumInner);

    __aicore__ inline void ClcMm31(const GlobalTensor<T2> &tensorC, const GlobalTensor<T1> &tensorA,
                                   const GlobalTensor<T1> &tensorB, const uint32_t &currBCvInner, const bool &isSync);

    /* matmul 32 和 matmul31的区别在于输入TensorA是需要做Transpose的，且输出需要做G轴的reduce. */
    __aicore__ inline void ClcMm32(const GlobalTensor<T2> &tensorC, const GlobalTensor<T1> &tensorA,
                                   const GlobalTensor<T1> &tensorB, const uint32_t &currBCvInner, const bool &isSync);

    __aicore__ inline void ClcMm4(const GlobalTensor<T1> &tensorC, const GlobalTensor<T1> &tensorA,
                                  const GlobalTensor<T1> &tensorB, const uint32_t &currBCvInner, const bool &isSync);

    __aicore__ inline void Copy2Workspace(const int64_t &batchSqLoopOffset, const int64_t &batchSkvLoopOffset,
                                          LocalTensor<T2> &mulResInner, LocalTensor<T2> &dvDropResInner,
                                          const int64_t &bCvMmOffset);
    __aicore__ inline void ResetGm4Dkdv(const uint32_t &inputLayoutTmp, const int64_t &currBatchSkvLoopOffset,
                                        const uint32_t &currBCvInner, const int64_t &mmNzDkvOffset);
    __aicore__ inline void NZCopyIn(int64_t mmOffset, GlobalTensor<T2> &mmWspGm, LocalTensor<T2> &mmTensorCurr,
                                    int64_t sQ, int64_t sKVAlign);
    __aicore__ inline void NZ2ND(LocalTensor<T2> &mmTensorCurr, LocalTensor<T2> &tmpTensor, int64_t sQ,
                                 int64_t sKVAlign, int64_t srcBatchOffset, int64_t dstBatchOffset);
};

template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::Init(
    __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query, __gm__ uint8_t *pse_shift,
    __gm__ uint8_t *drop_mask, __gm__ uint8_t *atten_mask, __gm__ uint8_t *forward_res, __gm__ uint8_t *softmax_max,
    __gm__ uint8_t *softmax_sum, __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *workspace,
    const FlashAttentionScoreGradUbngs1s2BbTilingData *__restrict ordTilingData, TPipe *pipe_in)
{
    mBlockIdx = GetBlockIdx();
    keyGm.SetGlobalBuffer((__gm__ T1 *)key);
    valueGm.SetGlobalBuffer((__gm__ T1 *)value);
    dxGm.SetGlobalBuffer((__gm__ T1 *)dx);
    queryGm.SetGlobalBuffer((__gm__ T1 *)query);
    pseGm.SetGlobalBuffer((__gm__ T1 *)pse_shift);
    dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)drop_mask);
    attenMaskGm.SetGlobalBuffer((__gm__ T1 *)atten_mask);
    forwardResGm.SetGlobalBuffer((__gm__ T1 *)forward_res);
    attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t *)atten_mask);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmax_max);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmax_sum);
    dqGm.SetGlobalBuffer((__gm__ T1 *)dq);
    dkGm.SetGlobalBuffer((__gm__ T1 *)dk);
    dvGm.SetGlobalBuffer((__gm__ T1 *)dv);
    workspaceGm.SetGlobalBuffer((__gm__ T2 *)workspace);
    syncGlobal.SetGlobalBuffer((__gm__ int32_t *)workspace, 100 * 8);
    dropoutWorkspaceGm.SetGlobalBuffer((__gm__ uint8_t *)workspace + 3200);
    InitOutput<int32_t>(syncGlobal[GetBlockIdx() * 8], 8, 0);

    ordTilingData_ = ordTilingData;
    pipe = pipe_in;

    b = ordTilingData_->opInfo.b;
    n = ordTilingData_->opInfo.n;
    g = ordTilingData_->opInfo.g;
    sQ = ordTilingData_->opInfo.sQ;
    pseSq = ordTilingData_->opInfo.pseSq;
    existPse = ordTilingData_->opInfo.existPse;
    pseShapeType = ordTilingData_->opInfo.pseShapeType;
    sKV = ordTilingData_->opInfo.sKV;
    sKVAlign = ordTilingData_->opInfo.sKVAlign;
    sKVAlignByte = ordTilingData_->opInfo.sKVAlignByte;
    hQ = ordTilingData_->opInfo.hQ;
    hKV = ordTilingData_->opInfo.hKV;
    d = ordTilingData_->opInfo.d;
    originalDAlign = ordTilingData_->opInfo.originalDAlign;
    attenMaskDimS2 = ordTilingData_->opInfo.attenMaskS2Size;
    scaleValue = ordTilingData_->opInfo.scaleValue;
    keepProb = ordTilingData_->opInfo.keepProb;
    preTokens = ordTilingData_->opInfo.preTokens;
    nextTokens = ordTilingData_->opInfo.nextTokens;
    headNum = ordTilingData_->opInfo.headNum;
    inputLayout = ordTilingData_->opInfo.inputLayout;
    preTokensBlocks = ordTilingData_->opInfo.preTokensBlocks;
    nextTokensBlocks = ordTilingData_->opInfo.nextTokensBlocks;
    inputDType = ordTilingData_->opInfo.inputDType;
    inputDTypeSize = ordTilingData_->opInfo.inputDTypeSize;
    vecCalcDTypeSize = ordTilingData_->opInfo.vecCalcDTypeSize;
    attenMaskShapeType = ordTilingData_->opInfo.attenMaskShapeType;
    hasAttenMask = ordTilingData_->opInfo.hasAttenMask;
    sKVAlignSize = ordTilingData_->opInfo.sKVAlignSize;

    bOut = ordTilingData_->splitCoreParams.bOut;
    apiClcQueueSize = ordTilingData_->splitCoreParams.apiClcQueueSize;
    usedCoreNum = ordTilingData_->splitCoreParams.usedCoreNum;

    bIn = ordTilingData_->singleCoreParams.bIn;
    singleCoreBatchRange = ordTilingData_->singleCoreParams.singleCoreBatchRange;
    bCvInner = ordTilingData_->singleCoreParams.bCvInner;
    bCvRatio = ordTilingData_->singleCoreParams.bCvRatio;
    syncLen = ordTilingData_->opInfo.syncLen;
    mm1WorkspaceLen = ordTilingData_->opInfo.mm1WorkspaceLen;
    mm2WorkspaceLen = ordTilingData_->opInfo.mm2WorkspaceLen;
    dqWorkspaceLen = ordTilingData_->opInfo.dqWorkspaceLen;
    dkWorkspaceLen = ordTilingData_->opInfo.dkWorkspaceLen;
    dropGmWorkspaceLen = ordTilingData_->opInfo.dropGmWorkspaceLen;
    mulGmWorkspaceLen = ordTilingData_->opInfo.mulGmWorkspaceLen;

    pingPongDropOffset = dropGmWorkspaceLen / 2 / sizeof(T1);
    pingPongMulOffset = mulGmWorkspaceLen / 2 / sizeof(T1);

    innerTmpBufSize = ordTilingData_->singleCoreParams.innerTmpBufSize;
    vecQueIn1Size = ordTilingData_->singleCoreParams.vecQueIn1Size;

    clcDInner = ordTilingData_->singleCoreParams.clcDInner;
    dSize = ordTilingData_->singleCoreParams.dSize;
    dInnerTail = ordTilingData_->singleCoreParams.dInnerTail;
    dInnerTailAlign = ordTilingData_->singleCoreParams.dInnerTailAlign;

    subRange = ordTilingData_->singleCoreParams.subRange;
    subMask = ordTilingData_->singleCoreParams.subMask;
    subMaskTail = ordTilingData_->singleCoreParams.subMaskTail;
    sKVAlignBlockNum = ordTilingData_->singleCoreParams.sKVAlignBlockNum;
    rightPadding = ordTilingData_->singleCoreParams.rightPadding;
    dstStride = ordTilingData_->singleCoreParams.dstStride;

    dropoutWorkspaceLen = ordTilingData_->opInfo.dropoutWorkspaceLen;

    if (existPse != 0) {
        pseInfo.pseBSize = pseShapeType == PSE_1NSS ? 1 : b;
        pseInfo.pseShapeType = pseShapeType == PSE_1NSS ? PSE_BNSS : pseShapeType;
        pseInfo.s1BaseSize = sQ;
        pseInfo.s1Size = sQ;
        pseInfo.s2Size = sKV;
        pseInfo.gSize = g;
        pseInfo.n2G = n * g;
        pseInfo.s2RealSize = sKV;
        pseInfo.s2AlignedSize = sKVAlign;
        pseInfo.needCast = false;
    }

    int64_t matResNum = bCvInner * n * g * sQ * sKVAlign;
    int64_t matResNumOffset = matResNum * mBlockIdx;

    matmulResultBuffer1.SetGlobalBuffer((__gm__ T2 *)workspace + (syncLen + dropoutWorkspaceLen) / sizeof(T2) +
                                        matResNumOffset);
    matmulResultBuffer2.SetGlobalBuffer(
        (__gm__ T2 *)workspace + (syncLen + mm1WorkspaceLen + dropoutWorkspaceLen) / sizeof(T2) + matResNumOffset);

    int64_t usedWorkspaceLen = syncLen + dropoutWorkspaceLen + mm1WorkspaceLen + mm2WorkspaceLen;
    auto dqAddr = usedWorkspaceLen / sizeof(T2);
    auto dkAddr = dqAddr + dqWorkspaceLen / sizeof(T2);
    dqWorkspaceGm.SetGlobalBuffer((__gm__ T2 *)workspace + dqAddr);
    dkWorkspaceGm.SetGlobalBuffer((__gm__ T2 *)workspace + dkAddr);
    // 对于dropout mask其size是bIn * n * g * sQ * sKVAlignByte， sKVAlignByte这个值一定小于等于sKVAlignSize
    // 例如sKVALign是17，sKVAlignByte = 32， sKVAlignSize = 64
    // sKVAlign是32， sKVAlignByte = 32， sKVAlignSize = 64
    bInNGSq = bIn * n * g * sQ;
    if (pseShapeType == PSE_BNSS) {
        pseInputNum = bIn * n * g * pseSq * sKVAlign;
    } else if (pseShapeType == PSE_BN1S) {
        pseInputNum = bIn * n * g * 1 * sKVAlign;
    } else {
        pseInputNum = n * g * pseSq * sKVAlign;
    }
    auto bestBasicBlockSize = BEST_BASIC_BLOCK_SIZE;
    if (MMPre_OUT_FORMAT == CubeFormat::NZ) {
        bestBasicBlockSize = BEST_BASIC_BLOCK_SIZE / 4 * 3;
    }

    // queue len: 16k
    pipe->InitBuffer(vecInQue1, 1, vecQueIn1Size);
    pipe->InitBuffer(vecInQue2, 1, vecQueIn1Size);
    // buf len: 32k
    pipe->InitBuffer(vecClc1, 1, innerTmpBufSize);
    pipe->InitBuffer(vecClc2, 1, innerTmpBufSize);
    // 16k
    pipe->InitBuffer(softmaxGradQue, 1, vecQueIn1Size);
    // 8k
    pipe->InitBuffer(dropoutQue, 1, bestBasicBlockSize / 4);
    // 32k
    pipe->InitBuffer(maxSumQue, 1, bestBasicBlockSize);
    pipe->InitBuffer(vecOutQue1, 1, apiClcQueueSize);

    // drop workspace offset
    int64_t workspaceOffsets = (dkAddr * sizeof(T2) + dkWorkspaceLen);
    dropWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + workspaceOffsets / sizeof(T1) + matResNumOffset);


    // mul workspace offset
    workspaceOffsets = (workspaceOffsets + dropGmWorkspaceLen);
    mulWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + workspaceOffsets / sizeof(T1) + matResNumOffset);

    DropOutBitModeInit();

    bool boolMode = false;
    if (sKV % DROPOUT4BIT_LEN != 0) {
        boolMode = true;
    }

    isDrop = false;
    if (keepProb < 1 && dropoutWorkspaceLen > 0) {
        isDrop = true;

        // for compute dropout mask offset
        dropMaskInfo.n2G = n * g;
        dropMaskInfo.gSize = g;
        dropMaskInfo.s1Size = sQ;
        dropMaskInfo.s2Size = sKV;
        // for compute dropout mask
        dropMaskInfo.keepProb = keepProb;
        dropMaskInfo.boolMode = boolMode;
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ClcSub(
    LocalTensor<T2> &frontResInner, LocalTensor<T2> &dpResInner, LocalTensor<T2> &sftFrontResInner)
{
    // [m,n] - [m,8] -> [m,n] 按b轴的block数repeat，每个指令repeat算[m,8] - [m,8], subRange循环处理 超过mask情况
    for (int64_t batchIndex = 0; batchIndex < (bIn * n * g); ++batchIndex) {
        for (int32_t subIdx = 0; subIdx < subRange; subIdx++) {
            int64_t src0Offset = batchIndex * sQ * sKVAlign + subIdx * sKVAlign * BIT_SIZE;
            int64_t src1Offset = batchIndex * sQ * (32 / sizeof(T2)) + subIdx * subMask;
            if (subIdx == subRange - 1 && subMaskTail != 0) {
                Sub(frontResInner[src0Offset], dpResInner[src0Offset], sftFrontResInner[src1Offset], subMaskTail,
                    sKVAlignBlockNum, {(uint8_t)(sKVAlignBlockNum), (uint8_t)(sKVAlignBlockNum), 1, 1, 1, 0});
            } else {
                // dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride, src1_rep_stride
                Sub(frontResInner[src0Offset], dpResInner[src0Offset], sftFrontResInner[src1Offset], subMask,
                    sKVAlignBlockNum, {(uint8_t)(sKVAlignBlockNum), (uint8_t)(sKVAlignBlockNum), 1, 1, 1, 0});
            }
        }
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::SetReClcShape(
    LocalTensor<T2> &mulResInner, LocalTensor<float> &maxInner, LocalTensor<float> &sumInner,
    LocalTensor<T2> &dvDropResInner)
{
    mulResInner.SetShapeInfo(ShapeInfo(2, innerMatOutShape, DataFormat::ND));
    maxInner.SetShapeInfo(ShapeInfo(2, innerReduceShape, DataFormat::ND));
    sumInner.SetShapeInfo(ShapeInfo(2, innerReduceShape, DataFormat::ND));
    dvDropResInner.SetShapeInfo(ShapeInfo(2, innerMatOutShape, DataFormat::ND));
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::CopyInSoftMax(
    LocalTensor<float> &maxInner, const GlobalTensor<float> &softmaxMaxGmIn, LocalTensor<float> &sumInner,
    const GlobalTensor<float> &softmaxSumGmIn)
{
    DataCopy(maxInner, softmaxMaxGmIn, innerReduceNum);
    DataCopy(sumInner, softmaxSumGmIn, innerReduceNum);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline bool
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::CopyInAttenMask(
    const int64_t &attenMaskOffset)
{
    if (hasAttenMask != 1) {
        return false;
    }
    DataCopyExtParams copyParams;
    LocalTensor<uint8_t> attenMaskUb = vecInQue2.AllocTensor<uint8_t>();
    attenMaskUb.SetSize(bInNGSq * sKVAlignByte);
    copyParams.blockCount = sQ;
    copyParams.blockLen = sKV;
    copyParams.srcStride = attenMaskDimS2 - sKV;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    DataCopyPadExtParams<uint8_t> copyPadParams;
    copyPadParams.isPad = false;
    copyPadParams.leftPadding = 0;
    copyPadParams.rightPadding = 0;
    copyPadParams.paddingValue = 0;

    int64_t coe = (attenMaskShapeType == 1) ? (n * g) : 1;                                  // 1:B1SS
    int64_t stride = (attenMaskShapeType == 2 || attenMaskShapeType == 1) ? (sQ * sKV) : 0; // 1:B1SS 2:BNSS
    for (int64_t copyIndex = 0; copyIndex < bIn * n * g; ++copyIndex) {
        DataCopyPad(attenMaskUb[copyIndex * sQ * sKVAlignByte],
                    attenMaskU8Gm[attenMaskOffset + copyIndex / coe * stride], copyParams, copyPadParams);
    }
    vecInQue2.EnQue(attenMaskUb);
    return true;
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::CalcAttenMaskOffset(
    int64_t &attenMaskOffset, const int64_t delta)
{
    if (delta == 0) {
        attenMaskOffset = 0;
    } else if (delta < 0) {
        if (-delta > sQ) {
            attenMaskOffset = sQ;
        } else {
            attenMaskOffset = -delta;
        }
    } else {
        if (delta > sKV) {
            attenMaskOffset = sKV * attenMaskDimS2;
        } else {
            attenMaskOffset = delta * attenMaskDimS2;
        }
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::CalcCausalAttenMaskOffset(
    int64_t &attenMaskOffset, const int64_t delta)
{
    CalcAttenMaskOffset(attenMaskOffset, delta);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ClcAttenMask(
    LocalTensor<T2> &mmResUb)
{
    LocalTensor<uint8_t> attenMaskUb = vecInQue2.DeQue<uint8_t>();
    LocalTensor<uint8_t> ubWorkspace = vecOutQue1.AllocTensor<uint8_t>();

    T2 scalar;
    if constexpr (IsSameType<T2, float>::value) {
        uint32_t tmp = 0xFF7FFFFF;
        scalar = *((float *)&tmp);
    } else {
        uint16_t tmp = 0xFBFF;
        scalar = *((half *)&tmp);
    }
    SelectWithBytesMaskShapeInfo shapeInfo;
    shapeInfo.firstAxis = bInNGSq;
    shapeInfo.srcLastAxis = sKVAlign;
    shapeInfo.maskLastAxis = sKVAlignByte;
    attenMaskUb.SetSize(shapeInfo.firstAxis * shapeInfo.maskLastAxis);
    mmResUb.SetSize(shapeInfo.firstAxis * shapeInfo.srcLastAxis);
    SelectWithBytesMask(mmResUb, mmResUb, scalar, attenMaskUb, ubWorkspace, shapeInfo);

    vecOutQue1.FreeTensor(ubWorkspace);
    vecInQue2.FreeTensor(attenMaskUb);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ClcSoftMax(
    LocalTensor<T2> &softmaxResInner, LocalTensor<T2> &reMatmulResInner, LocalTensor<float> &maxInner,
    LocalTensor<float> &sumInner)
{
    LocalTensor<uint8_t> apiClcTensor = vecOutQue1.AllocTensor<uint8_t>();
    apiClcTensor.SetSize(apiClcQueueSize);
    bool isBasicBlock = ((bInNGSq) % 8 == 0) && (sKV % 64 == 0);
    if (isBasicBlock) {
        SimpleSoftMax<T2, true, true>(softmaxResInner, sumInner, maxInner, reMatmulResInner, apiClcTensor,
                                      ordTilingData_->softmaxTilingData);
    } else {
        SimpleSoftMax<T2, true, false>(softmaxResInner, sumInner, maxInner, reMatmulResInner, apiClcTensor,
                                       ordTilingData_->softmaxTilingData);
    }
    vecOutQue1.FreeTensor(apiClcTensor);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::FrontCompute(
    const int64_t &batchSqLoopOffset, const int64_t &batchSkvLoopOffset, const int64_t &dropMaskOffset,
    const int64_t &bCvMmOffset, const int64_t &bCvNzMmOffset, const int64_t &bCvIndex,
    const int64_t &currentBatchRange)
{
    uint32_t sftFrontResSize = bInNGSq * BLOCK_SIZE / sizeof(T2);
    LocalTensor<T2> sftFrontResInner = softmaxGradQue.AllocTensor<T2>();

    sftFrontResInner.SetSize(sftFrontResSize);
    uint32_t sftFrontResInnerShape[] = {static_cast<uint32_t>(sQ),
                                        static_cast<uint32_t>(bIn * n * g * (32 / sizeof(T2)))};
    sftFrontResInner.SetShapeInfo(ShapeInfo(2, sftFrontResInnerShape, DataFormat::ND));
    Duplicate<T2>(sftFrontResInner, 0.0, sftFrontResSize);
    AscendC::PipeBarrier<PIPE_V>();

    for (int64_t dSizeIdx = 0; dSizeIdx < dSize; dSizeIdx++) {
        int64_t dInner = (dSizeIdx == dSize - 1) ? dInnerTail : clcDInner;
        int64_t dInnerAlign = (dSizeIdx == dSize - 1) ? dInnerTailAlign : clcDInner;

        LocalTensor<T1> dxInner = vecInQue1.AllocTensor<T1>();
        LocalTensor<T1> frontResInner = vecInQue2.AllocTensor<T1>();
        uint32_t dxShape[2];
        dxShape[0] = bInNGSq;
        dxShape[1] = dInnerAlign;
        dxInner.SetSize(bInNGSq * dInnerAlign);
        dxInner.SetShapeInfo(ShapeInfo(2, dxShape, DataFormat::ND));
        frontResInner.SetSize(bInNGSq * dInnerAlign);
        frontResInner.SetShapeInfo(ShapeInfo(2, dxShape, DataFormat::ND));

        DataCopyExtParams copyParams;
        DataCopyPadExtParams<T1> copyPadParams;
        uint16_t dOffset = (dSizeIdx == dSize - 1) ? dSizeIdx * clcDInner : d - dInner;
        int64_t ubOffset = 0;
        int64_t gmOffset = 0;
        copyParams.blockCount = sQ;
        copyParams.blockLen = dInner * sizeof(T1);
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        copyPadParams.isPad = true;
        copyPadParams.leftPadding = 0;
        copyPadParams.rightPadding = (dInnerAlign - dInner);
        copyPadParams.paddingValue = 0;

        for (int64_t copyIndex = 0; copyIndex < bIn * n * g; copyIndex++) {
            int64_t bIdx = copyIndex / (n * g);
            int64_t nIdx = copyIndex % (n * g);
            if (inputLayout == 1) {
                // SBH
                ubOffset = copyIndex * sQ * dInnerAlign;
                gmOffset = copyIndex * d + dSizeIdx * clcDInner;
                copyParams.srcStride = ((b * n * g - 1) * d + dOffset) * sizeof(T1);
            } else if (inputLayout == 2) {
                // BNSD
                ubOffset = copyIndex * sQ * dInnerAlign;
                gmOffset = copyIndex * sQ * d + dSizeIdx * clcDInner;
                copyParams.srcStride = dOffset * sizeof(T1);
            } else {
                ubOffset = bIdx * n * g * sQ * dInnerAlign + nIdx * sQ * dInnerAlign;
                gmOffset = bIdx * n * g * sQ * d + nIdx * d + dSizeIdx * clcDInner;
                copyParams.srcStride = ((n * g - 1) * d + dOffset) * sizeof(T1);
            }
            DataCopyPad(dxInner[ubOffset], dxGm[batchSqLoopOffset + gmOffset], copyParams, copyPadParams);
            DataCopyPad(frontResInner[ubOffset], forwardResGm[batchSqLoopOffset + gmOffset],
                        copyParams, copyPadParams);
            vecInQue1.EnQue(dxInner);
            vecInQue1.DeQue<T1>();
            vecInQue2.EnQue(frontResInner);
            vecInQue2.DeQue<T1>();
        }
        vecInQue2.EnQue(frontResInner);
        vecInQue2.DeQue<T1>();

        bool isBasicBlock = (sQ % 8 == 0) && (dInnerAlign % 64 == 0);
        LocalTensor<uint8_t> apiClcTensor = vecOutQue1.AllocTensor<uint8_t>();
        apiClcTensor.SetSize(apiClcQueueSize);

        LocalTensor<T2> castedDxInner = vecClc1.AllocTensor<T2>();
        LocalTensor<T2> castedFrontResInner = vecClc2.AllocTensor<T2>();

        castedFrontResInner.SetSize(bInNGSq * dInnerAlign);
        castedFrontResInner.SetShapeInfo(ShapeInfo(2, dxShape, DataFormat::ND));
        castedDxInner.SetSize(bInNGSq * dInnerAlign);
        castedDxInner.SetShapeInfo(ShapeInfo(2, dxShape, DataFormat::ND));
        Cast(castedFrontResInner, frontResInner, RoundMode::CAST_NONE, bInNGSq * dInnerAlign);
        Cast(castedDxInner, dxInner, RoundMode::CAST_NONE, bInNGSq * dInnerAlign);
        AscendC::PipeBarrier<PIPE_V>();

        vecInQue1.FreeTensor(dxInner);
        vecInQue2.FreeTensor(frontResInner);

        LocalTensor<T2> softmaxTensor = vecInQue1.AllocTensor<T2>();

        if (isBasicBlock) {
            SoftmaxGradFront<T2, true>(softmaxTensor, castedFrontResInner, castedDxInner, apiClcTensor,
                                       this->ordTilingData_->softmaxGradTilingData);
        } else {
            SoftmaxGradFront<T2, false>(softmaxTensor, castedFrontResInner, castedDxInner, apiClcTensor,
                                        this->ordTilingData_->softmaxGradTilingData);
        }
        AscendC::PipeBarrier<PIPE_V>();
        vecClc1.FreeTensor(castedDxInner);
        vecClc2.FreeTensor(castedFrontResInner);
        Add(sftFrontResInner, softmaxTensor, sftFrontResInner, sftFrontResSize);
        AscendC::PipeBarrier<PIPE_V>();
        vecInQue1.FreeTensor(softmaxTensor);
        vecOutQue1.FreeTensor(apiClcTensor);
    }

    LocalTensor<T2> frontResInner1 = vecClc1.AllocTensor<T2>();
    LocalTensor<T2> &dpRes = frontResInner1;
    LocalTensor<T2> &mm1Res = frontResInner1;
    LocalTensor<uint8_t> dpMask;

    if (isDrop) {
        dpMask = dropoutQue.AllocTensor<uint8_t>();
        dpMask.SetSize(maskInputNum);
        CopyInDropMask<true>(dpMask, dropoutWorkspaceGm, dropMaskGm, this->dropMaskInfo);
        dropoutQue.EnQue(dpMask);
    }


    frontResInner1.SetShapeInfo(ShapeInfo(2, innerMatOutShape, DataFormat::ND));

    if (bCvIndex == 0) {
        if (inputLayout == 1 || inputLayout == 2) {
            mm1.WaitIterateBatch();
        } else if (currentBatchRange < 2) {
            // BSH只能同步
            // bmm1
            mm1.SetTail(sQ, sKV, d);
            mm1.SetTensorA(this->dxGm[batchSqLoopOffset]);
            mm1.SetTensorB(this->valueGm[batchSkvLoopOffset], true);
            for (uint32_t i = 0; i < bCvInner; i++) {
                mm1.SetTensorA(this->dxGm[batchSqLoopOffset + i * n * g * sQ * d]);
                mm1.SetTensorB(this->valueGm[batchSkvLoopOffset + i * n * sKV * d], true);
                if constexpr (MMPre_OUT_FORMAT == CubeFormat::NZ) {
                    mm1.template IterateBatch<true>(matmulResultBuffer1[i * n * g * sQ * sKVAlign], n * g, n, false);
                } else {
                    mm1.template IterateBatch<true>(matmulResultBuffer1[i * n * g * sQ * sKV], n * g, n, false);
                }
            }
        }
        mm1.End();
    }

    if constexpr (MMPre_OUT_FORMAT == CubeFormat::ND) {
        DataCopyExtParams intriParams;
        intriParams.blockCount = bInNGSq;
        intriParams.blockLen = sKV * vecCalcDTypeSize;
        intriParams.srcStride = 0;
        intriParams.dstStride = dstStride;
        intriParams.rsv = 0;

        DataCopyPadExtParams<T2> copyPadParams;
        copyPadParams.isPad = true;
        copyPadParams.leftPadding = 0;
        copyPadParams.rightPadding = rightPadding;
        copyPadParams.paddingValue = 0;

        DataCopyPad(mm1Res, matmulResultBuffer1[bCvMmOffset], intriParams, copyPadParams);
        vecClc1.EnQue(mm1Res);
        vecClc1.DeQue<T2>();
    } else {
        NZCopyIn(bCvNzMmOffset, matmulResultBuffer1, mm1Res, sQ, bIn * n * g *  sKVAlign);
        vecClc1.EnQue(mm1Res);
        vecClc1.DeQue<T2>();
        auto tmpTensor = vecOutQue1.AllocTensor<T2>();
        DataCopy(tmpTensor, mm1Res, sQ * bIn * n * g * sKVAlign + bIn * n * g * sKVAlign / C0_SIZE * VEC_REPEAT);
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t i = 0; i < bIn * n * g; i++) {
            int64_t sQLoop = (sQ + PER_COPY_MAX_NUM - 1) / PER_COPY_MAX_NUM;
            int64_t sQTail = sQ % PER_COPY_MAX_NUM;
            for(int64_t sQLoopIdx = 0; sQLoopIdx < sQLoop; sQLoopIdx++){
                int64_t sQReal = (sQLoopIdx == sQLoop - 1) ? sQTail : PER_COPY_MAX_NUM;
                sQReal = (sQReal == 0) ? PER_COPY_MAX_NUM : sQReal;
                int64_t srcBatchOffset = i * (sQ * sKVAlign + sKVAlign / C0_SIZE * VEC_REPEAT) + PER_COPY_MAX_NUM * sQLoopIdx * sKVAlign;
                int64_t dstBatchOffset = i * sQ * sKVAlign + PER_COPY_MAX_NUM * sQLoopIdx * sKVAlign;
                NZ2ND(mm1Res, tmpTensor, sQReal, sKVAlign, srcBatchOffset, dstBatchOffset);
            }
        }
        vecOutQue1.FreeTensor(tmpTensor);
    }

    if (isDrop) {
        dropoutQue.DeQue<T2>();
        // for compute dropout mask
        dropMaskInfo.firstAxis = bInNGSq;
        dropMaskInfo.lstAxis = sKVAlign;
        dropMaskInfo.maskLstAxis = sKVAlign;
        LocalTensor<uint8_t> apiClcTensor = vecOutQue1.AllocTensor<uint8_t>();
        ComputeDropMask<float, true>(dpRes, mm1Res, dpMask, apiClcTensor, this->dropMaskInfo);
        vecOutQue1.FreeTensor(apiClcTensor);

        AscendC::PipeBarrier<PIPE_V>();
        dropoutQue.FreeTensor(dpMask);
    }

    uint32_t tempInnerMatOutShape[2];
    tempInnerMatOutShape[0] = bInNGSq;
    tempInnerMatOutShape[1] = sKVAlign;
    mm1Res.SetShapeInfo(ShapeInfo(2, tempInnerMatOutShape, DataFormat::ND));

    ClcSub(frontResInner1, dpRes, sftFrontResInner);
    AscendC::PipeBarrier<PIPE_V>();

    vecClc1.FreeTensor(frontResInner1);

    softmaxGradQue.FreeTensor(sftFrontResInner);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ClcMm31(
    const GlobalTensor<T2> &tensorC, const GlobalTensor<T1> &tensorA, const GlobalTensor<T1> &tensorB,
    const uint32_t &currBCvInner, const bool &isSync)
{
    mm31.SetTensorA(tensorA, false);
    mm31.SetTensorB(tensorB, false);
    if (inputLayout == 1 || inputLayout == 2) {
        // SBH, BNSD
        if (isSync) {
          mm31.template IterateBatch<true>(tensorC, currBCvInner * n * g, currBCvInner * n, false);
        } else {
          mm31.template IterateBatch<false>(tensorC, currBCvInner * n * g, currBCvInner * n, false);
        }
    } else {
        // BSH
        for (uint32_t i = 0; i < currBCvInner; i++) {
            mm31.SetTensorA(tensorA[i * n * g * sQ * sKV]);
            mm31.SetTensorB(tensorB[i * n * sKV * d], false);
            if constexpr (MMNext_OUT_FORMAT == CubeFormat::NZ) {
                if (isSync) {
                  mm31.template IterateBatch<true>(tensorC[i * n * g * sQ * originalDAlign], n * g, n, false);
                } else {
                  mm31.template IterateBatch<false>(tensorC[i * n * g * sQ * originalDAlign], n * g, n, false);
                }
            } else {
                if (isSync) {
                  mm31.template IterateBatch<true>(tensorC[i * n * g * sQ * d], n * g, n, false);
                } else {
                  mm31.template IterateBatch<false>(tensorC[i * n * g * sQ * d], n * g, n, false);
                }
            }
        }
    }
    mm31.End();
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ClcMm32(
    const GlobalTensor<T2> &tensorC, const GlobalTensor<T1> &tensorA, const GlobalTensor<T1> &tensorB,
    const uint32_t &currBCvInner, const bool &isSync)
{
    mm32.SetTensorA(tensorA, true);
    mm32.SetTensorB(tensorB, false);
    if (inputLayout == 1 || inputLayout == 2) {
        // SBH, BNSD
        if (isSync) {
          mm32.template IterateBatch<true>(tensorC, currBCvInner * n * g, currBCvInner * n * g, false);
        } else {
          mm32.template IterateBatch<false>(tensorC, currBCvInner * n * g, currBCvInner * n * g, false);
        }
    } else {
        // BSH
        for (uint32_t i = 0; i < currBCvInner; i++) {
            mm32.SetTensorA(tensorA[i * n * g * sQ * sKV], true);
            mm32.SetTensorB(tensorB[i * n * g * sQ * d], false);
            if constexpr (MMNext_OUT_FORMAT == CubeFormat::NZ) {
                if (isSync) {
                  mm32.template IterateBatch<true>(tensorC[i * n * sKV * originalDAlign], n * g, n * g, false);
                } else {
                  mm32.template IterateBatch<false>(tensorC[i * n * sKV * originalDAlign], n * g, n * g, false);
                }
            } else {
                if (isSync) {
                  mm32.template IterateBatch<true>(tensorC[i * n * sKV * d], n * g, n * g, false);
                } else {
                  mm32.template IterateBatch<false>(tensorC[i * n * sKV * d], n * g, n * g, false);
                }
            }
        }
    }
    mm32.End();
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ClcMm4(
    const GlobalTensor<T1> &tensorC, const GlobalTensor<T1> &tensorA, const GlobalTensor<T1> &tensorB,
    const uint32_t &currBCvInner, const bool &isSync)
{
    mm4.SetTensorA(tensorA, true);
    mm4.SetTensorB(tensorB, false);
    if (inputLayout == 1 || inputLayout == 2) {
        // SBH, BNSD
        if (isSync) {
          mm4.template IterateBatch<true>(tensorC, currBCvInner * n * g, currBCvInner * n * g, false);
        } else {
          mm4.template IterateBatch<false>(tensorC, currBCvInner * n * g, currBCvInner * n * g, false);
        }
    } else {
        // BSH
        for (uint32_t i = 0; i < currBCvInner; i++) {
            mm4.SetTensorA(tensorA[i * n * g * sQ * sKV], true);
            mm4.SetTensorB(tensorB[i * n * g * sQ * d], false);
            if (isSync) {
              mm4.template IterateBatch<true>(tensorC[i * n * sKV * d], n * g, n * g, false);
            } else {
              mm4.template IterateBatch<false>(tensorC[i * n * sKV * d], n * g, n * g, false);
            }
        }
    }
    mm4.End();
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::NZCopyIn(
    int64_t mmOffset, GlobalTensor<T2> &mmWspGm, LocalTensor<T2> &mmTensorCurr, int64_t sQ, int64_t sKVAlign)
{
    DataCopyParams intriParams;
    intriParams.blockCount = sKVAlign / C0_SIZE;
    intriParams.blockLen = sQ * C0_SIZE / CAL_BLOCK_NUM;
    intriParams.srcStride = 0;
    intriParams.dstStride = 1;
    DataCopy(mmTensorCurr, mmWspGm[mmOffset], intriParams);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::NZ2ND(
    LocalTensor<T2> &mmTensorCurr, LocalTensor<T2> &tmpTensor, int64_t sQ, int64_t sKVAlign,
    int64_t srcBatchOffset, int64_t dstBatchOffset)
{
    CopyRepeatParams nz2ndParams;
    nz2ndParams.srcStride = sQ * C0_SIZE / CAL_BLOCK_NUM + 1;
    nz2ndParams.dstStride = C0_SIZE / CAL_BLOCK_NUM;
    nz2ndParams.srcRepeatSize = C0_SIZE / CAL_BLOCK_NUM;
    nz2ndParams.dstRepeatSize = sKVAlign / CAL_BLOCK_NUM;

    uint16_t c0Repeat = C0_SIZE / CAL_BLOCK_NUM;
    uint16_t c1Repeat = sKVAlign / C0_SIZE / VEC_REPEAT;
    uint16_t c1Remain = sKVAlign / C0_SIZE % VEC_REPEAT;
    uint16_t nRepeat = sQ;
    for (uint16_t i = 0; i < c0Repeat; ++i) {
        for (uint16_t j = 0; j < c1Repeat; ++j) {
            Copy(mmTensorCurr[dstBatchOffset + i * CAL_BLOCK_NUM + j * VEC_REPEAT * C0_SIZE],
                 tmpTensor[srcBatchOffset + i * CAL_BLOCK_NUM + j * VEC_REPEAT * (sQ * C0_SIZE + CAL_BLOCK_NUM)],
                 VEC_REPEAT * CAL_BLOCK_NUM, nRepeat, nz2ndParams);
        }
        if (c1Remain > 0) {
            Copy(mmTensorCurr[dstBatchOffset + i * CAL_BLOCK_NUM + c1Repeat * VEC_REPEAT * C0_SIZE],
                 tmpTensor[srcBatchOffset + i * CAL_BLOCK_NUM
                           + c1Repeat * VEC_REPEAT * (sQ * C0_SIZE + CAL_BLOCK_NUM)],
                 VEC_REPEAT * c1Remain, nRepeat, nz2ndParams);
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::Copy2Workspace(
    const int64_t &batchSqLoopOffset, const int64_t &batchSkvLoopOffset, LocalTensor<T2> &mulResInner,
    LocalTensor<T2> &dvDropResInner, const int64_t &bCvMmOffset)
{
    DataCopyExtParams intriParams;
    intriParams.blockCount = bInNGSq;
    intriParams.blockLen = sKV * inputDTypeSize;
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    intriParams.rsv = 0;

    LocalTensor<T1> castedMulResPad = vecOutQue1.AllocTensor<T1>();
    castedMulResPad.SetSize(innerMatResNum);
    Cast(castedMulResPad, mulResInner, RoundMode::CAST_ROUND, innerMatResNum);

    vecOutQue1.EnQue(castedMulResPad);
    vecOutQue1.DeQue<T1>();

    DataCopyPad(mulWorkspaceGm[pingpongIdx * pingPongMulOffset + bCvMmOffset], castedMulResPad, intriParams);
    vecOutQue1.FreeTensor(castedMulResPad);

    LocalTensor<T1> castedDvDropResPad = vecOutQue1.AllocTensor<T1>();
    castedDvDropResPad.SetSize(innerMatResNum);
    Cast(castedDvDropResPad, dvDropResInner, RoundMode::CAST_ROUND, innerMatResNum);

    vecOutQue1.EnQue(castedDvDropResPad);
    vecOutQue1.DeQue<T1>();

    DataCopyPad(dropWorkspaceGm[pingpongIdx * pingPongDropOffset + bCvMmOffset], castedDvDropResPad, intriParams);

    vecOutQue1.FreeTensor(castedDvDropResPad);
}


template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ReCompute(
    const int64_t &batchSqCLoopOffset, const int64_t &batchSkvCLoopOffset, const int64_t &batchSqLoopOffset,
    const int64_t &batchSkvLoopOffset, const int64_t &attenMaskOffset, const int64_t &dropMaskOffset,
    const int64_t &batchReduceOffset, const int64_t &bCvMmOffset, const int64_t &bCvNzMmOffset,
    const int64_t &bCvIndex, const bool isCvTail, const int64_t &currentBatchRange)
{
    LocalTensor<T2> subResInner = vecClc1.AllocTensor<T2>();
    LocalTensor<T2> &mulResInner = subResInner;
    LocalTensor<T2> dvDropResInner = vecClc2.AllocTensor<T2>();
    LocalTensor<T2> &reMatmulResInner = dvDropResInner;
    LocalTensor<T2> &softmaxResInner = dvDropResInner;
    LocalTensor<T2> &attenMaskResInner = dvDropResInner;

    bool clcAttenMask = false;
    if (existPse != 0) {
        LocalTensor<T1> pseUb = vecInQue2.AllocTensor<T1>();
        pseUb.SetSize(pseInputNum);
        auto noCastedPseUb = vecOutQue1.AllocTensor<T2>();
        noCastedPseUb.SetSize(0);
        PseCopyIn<T1, T2, LayOutTypeEnum::LAYOUT_BNSD, true>(noCastedPseUb, pseUb, this->pseGm, pseInfo);
        vecOutQue1.FreeTensor(noCastedPseUb);
        vecInQue2.EnQue(pseUb);
    }

    if (existPse == 0) {
        clcAttenMask = CopyInAttenMask(attenMaskOffset);
    }

    if (bCvIndex == 0) {
        if (inputLayout == 1 || inputLayout == 2) {
            mm1.WaitIterateBatch();
        } else if (currentBatchRange < 2) {
            // BSH 只支持同步
            // bmm2
            mm1.SetTail(sQ, sKV, d);
            mm1.SetTensorA(this->queryGm[batchSqLoopOffset]);
            mm1.SetTensorB(this->keyGm[batchSkvLoopOffset], true);
            for (uint32_t i = 0; i < bCvInner; i++) {
                mm1.SetTensorA(this->queryGm[batchSqLoopOffset + i * n * g * sQ * d]);
                mm1.SetTensorB(this->keyGm[batchSkvLoopOffset + i * n * sKV * d], true);
                if constexpr (MMPre_OUT_FORMAT == CubeFormat::NZ) {
                    mm1.template IterateBatch<true>(matmulResultBuffer2[i * n * g * sQ * sKVAlign], n * g, n, false);
                } else {
                    mm1.template IterateBatch<true>(matmulResultBuffer2[i * n * g * sQ * sKV], n * g, n, false);
                }
            }
        }
        mm1.End();
    }

    if constexpr (MMPre_OUT_FORMAT == CubeFormat::ND) {
        DataCopyExtParams intriParams;
        intriParams.blockCount = bInNGSq;
        intriParams.blockLen = sKV * vecCalcDTypeSize;
        intriParams.srcStride = 0;
        intriParams.dstStride = dstStride;
        intriParams.rsv = 0;

        DataCopyPadExtParams<T2> copyPadParams;
        copyPadParams.isPad = true;
        copyPadParams.leftPadding = 0;
        copyPadParams.rightPadding = rightPadding;
        copyPadParams.paddingValue = 0;

        DataCopyPad(reMatmulResInner, matmulResultBuffer2[bCvMmOffset], intriParams, copyPadParams);

        vecClc2.EnQue(reMatmulResInner);
        vecClc2.DeQue<T2>();
    } else {
        NZCopyIn(bCvNzMmOffset, matmulResultBuffer2, reMatmulResInner, sQ, bIn * n * g * sKVAlign);
        vecClc2.EnQue(reMatmulResInner);
        vecClc2.DeQue<T2>();
        auto tmpTensor = vecOutQue1.AllocTensor<T2>();
        DataCopy(tmpTensor, reMatmulResInner, sQ * bIn * n * g * sKVAlign + bIn * n * g * sKVAlign / C0_SIZE * VEC_REPEAT);
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t i = 0; i < bIn * n * g; i++) {
            int64_t sQLoop = (sQ + PER_COPY_MAX_NUM - 1) / PER_COPY_MAX_NUM;
            int64_t sQTail = sQ % PER_COPY_MAX_NUM;
            for(int64_t sQLoopIdx = 0; sQLoopIdx < sQLoop; sQLoopIdx++){
                int64_t sQReal = (sQLoopIdx == sQLoop - 1) ? sQTail : PER_COPY_MAX_NUM;
                sQReal = (sQReal == 0) ? PER_COPY_MAX_NUM : sQReal;
                int64_t srcBatchOffset = i * (sQ * sKVAlign + sKVAlign / C0_SIZE * VEC_REPEAT) + PER_COPY_MAX_NUM * sQLoopIdx * sKVAlign;
                int64_t dstBatchOffset = i * sQ * sKVAlign + PER_COPY_MAX_NUM * sQLoopIdx * sKVAlign;
                NZ2ND(reMatmulResInner, tmpTensor, sQReal, sKVAlign, srcBatchOffset, dstBatchOffset);
            }
        }
        vecOutQue1.FreeTensor(tmpTensor);
    }

    if (existPse != 0) {
        LocalTensor<T1> pseUb = vecInQue2.DeQue<T1>();
        uint32_t eleNum = pseInputNum;

        auto castedPseUb = vecOutQue1.AllocTensor<T2>();
        castedPseUb.SetSize(eleNum);
        Cast(castedPseUb, pseUb, RoundMode::CAST_NONE, eleNum);
        AscendC::PipeBarrier<PIPE_V>();
        if (pseShapeType == PSE_1NSS) {
            for (int64_t i = 0; i < bIn; i++) {
                LocalTensor<T2> pseRes = reMatmulResInner[i * eleNum];
                PseCompute<T2, true>(pseRes, castedPseUb, this->pseInfo);
            }
        } else if (pseShapeType == PSE_BN1S) {
            pseInfo.vec1S1RealSize = sQ;
            for (int64_t batchIndex = 0; batchIndex < bIn * n * g; ++batchIndex) {
                LocalTensor<T2> pseRes = reMatmulResInner[batchIndex * sQ * sKVAlign];
                LocalTensor<T2> pse = castedPseUb[batchIndex * sKVAlign];
                PseCompute<T2, true>(pseRes, pse, this->pseInfo);
            }
        } else {
            PseCompute<T2, true>(reMatmulResInner, castedPseUb, this->pseInfo);
        }
        vecOutQue1.FreeTensor(castedPseUb);
        vecInQue2.FreeTensor(pseUb);
        AscendC::PipeBarrier<PIPE_V>();
        clcAttenMask = CopyInAttenMask(attenMaskOffset);
    }
    LocalTensor<float> maxInner = maxSumQue.AllocTensor<float>();
    LocalTensor<float> sumInner = maxInner[bInNGSq * BLOCK_SIZE / sizeof(T2)];
    innerReduceNum = bInNGSq * BIT_SIZE;
    maxInner.SetSize(innerReduceNum);
    sumInner.SetSize(innerReduceNum);
    SetReClcShape(mulResInner, maxInner, sumInner, dvDropResInner);
    CopyInSoftMax(maxInner, softmaxMaxGm[batchReduceOffset], sumInner, softmaxSumGm[batchReduceOffset]);
    maxSumQue.EnQue(maxInner);
    maxSumQue.DeQue<float>();

    Muls(reMatmulResInner, reMatmulResInner, (T2)scaleValue, innerMatResNum);
    AscendC::PipeBarrier<PIPE_V>();
    if (clcAttenMask) {
        ClcAttenMask(reMatmulResInner);
        AscendC::PipeBarrier<PIPE_V>();
    }
    uint32_t tempInnerMatOutShape[2];
    tempInnerMatOutShape[0] = bInNGSq;
    tempInnerMatOutShape[1] = sKVAlign;
    dvDropResInner.SetShapeInfo(ShapeInfo(2, tempInnerMatOutShape, DataFormat::ND));
    ClcSoftMax(softmaxResInner, attenMaskResInner, maxInner, sumInner);
    AscendC::PipeBarrier<PIPE_V>();

    mulResInner.SetShapeInfo(ShapeInfo(2, tempInnerMatOutShape, DataFormat::ND));
    Mul(mulResInner, softmaxResInner, subResInner, innerMatResNum);
    AscendC::PipeBarrier<PIPE_ALL>();


    if (isDrop) {
        LocalTensor<uint8_t> dpMask = dropoutQue.AllocTensor<uint8_t>();
        dpMask.SetSize(maskInputNum);
        CopyInDropMask<true>(dpMask, dropoutWorkspaceGm, dropMaskGm, this->dropMaskInfo);
        dropoutQue.EnQue(dpMask);
        dropoutQue.DeQue<uint8_t>();

        // for compute dropout mask
        dropMaskInfo.firstAxis = bInNGSq;
        dropMaskInfo.lstAxis = sKVAlign;
        dropMaskInfo.maskLstAxis = sKVAlign;

        LocalTensor<uint8_t> apiClcTensor = vecOutQue1.AllocTensor<uint8_t>();
        ComputeDropMask<float, true>(softmaxResInner, softmaxResInner, dpMask, apiClcTensor, this->dropMaskInfo);
        vecOutQue1.FreeTensor(apiClcTensor);

        AscendC::PipeBarrier<PIPE_V>();
        dropoutQue.FreeTensor(dpMask);
    }

    maxSumQue.FreeTensor(maxInner);
    Copy2Workspace(batchSqLoopOffset, batchSkvLoopOffset, mulResInner, dvDropResInner, bCvMmOffset);
    vecClc1.FreeTensor(mulResInner);
    vecClc2.FreeTensor(dvDropResInner);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ResetGm4Dkdv(
    const uint32_t &inputLayoutTmp, const int64_t &currBatchSkvLoopOffset, const uint32_t &currBCvInner,
    const int64_t &mmNzDkvOffset)
{
    // 清除GM，否则pta连跑有问题
    if (inputLayoutTmp == 1) {
        // SBH
        for (int64_t i = 0; i < sKV; i++) {
            int64_t offset = currBatchSkvLoopOffset + i * b * n * 1 * d;
            int64_t num = currBCvInner * n * 1 * d;
            if constexpr (MMNext_OUT_FORMAT == CubeFormat::NZ) {
                int64_t nums = currBCvInner * n * 1 * sKV * originalDAlign;
                InitOutput<T2>(dkWorkspaceGm[mmNzDkvOffset], nums, 0);
            } else {
                InitOutput<T2>(dkWorkspaceGm[offset], num, 0);
            }
            InitOutput<T1>(dvGm[offset], num, 0);
        }
    } else {
        int64_t num = currBCvInner * n * 1 * sKV * d;
        if constexpr (MMNext_OUT_FORMAT == CubeFormat::NZ) {
            int64_t nums = currBCvInner * n * 1 * sKV * originalDAlign;
            InitOutput<T2>(dkWorkspaceGm[mmNzDkvOffset], nums, 0);
        } else {
            InitOutput<T2>(dkWorkspaceGm[currBatchSkvLoopOffset], num, 0);
        }
        InitOutput<T1>(dvGm[currBatchSkvLoopOffset], num, 0);
    }
}

// T1 INPUT_T, T2 CALC_T
template <typename T1, typename T2, const MatmulConfig &MM_CFG, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bb<T1, T2, MM_CFG, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::Process()
{
    if (g_coreType == AIV && mBlockIdx >= usedCoreNum) {
        SyncAll();
        return;
    }

    int64_t lastBatchSqLoopOffset = 0;
    int64_t lastBatchSkvLoopOffset = 0;
    int64_t lastMmNzDqOffset = 0;
    int64_t lastMmNzDkvOffset = 0;

    int64_t batchOffset = mBlockIdx * singleCoreBatchRange;
    int64_t currentBatchRange =
        singleCoreBatchRange < (bOut - batchOffset) ? singleCoreBatchRange : (bOut - batchOffset);

    for (int64_t batchIdx = 0; batchIdx < currentBatchRange; batchIdx++) {
        pingpongIdx = 1 - pingpongIdx;
        int64_t bIdx = batchOffset + batchIdx;
        int64_t bCvSqOffset = 0;
        int64_t bCvSkvOffset = 0;
        int64_t bCvMmOffset = 0;
        int64_t bCvNzMmOffset = 0;
        int64_t bCvDropMaskOffset = 0;
        int64_t bCvAttenMaskOffset = 0;

        int64_t batchSqLoopOffset = 0;
        int64_t batchSkvLoopOffset = 0;
        int64_t batchSoftmaxInputOffset = 0;

        int64_t mmNzDqOffset = 0;
        int64_t mmNzDkvOffset = 0;

        bool isCvTail = false;

        int64_t previousBatchCnt = (bIdx * bCvInner) * n * g;
        int64_t dropMaskOffset = previousBatchCnt * sQ * sKV;
        if (sKV % DROPOUT4BIT_LEN == 0) {
            dropMaskOffset = previousBatchCnt * sQ * sKV / 8;
        }

        int64_t attenMaskOffset = 0;
        if (hasAttenMask == 1) {
            int64_t compressMode = ordTilingData_->opInfo.attenMaskCompressMode;
            if (compressMode == 1) {
                CalcCausalAttenMaskOffset(attenMaskOffset, 0);
            } else if (compressMode == 2) {
                CalcCausalAttenMaskOffset(attenMaskOffset, sKV - sQ);
            } else if (attenMaskShapeType == 0) { // SS 11SS
                attenMaskOffset = 0;
            } else if (attenMaskShapeType == 1) { // B1SS
                attenMaskOffset = bIdx * bCvInner * sQ * sKV;
            } else { // BNSS
                attenMaskOffset = bIdx * bCvInner * n * g * sQ * sKV;
            }
        }
        batchSoftmaxInputOffset = previousBatchCnt * sQ * BIT_SIZE;
        if (inputLayout == 1) { // SBH即SBND
            batchSqLoopOffset = bIdx * bCvInner * n * g * d;
            batchSkvLoopOffset = bIdx * bCvInner * n * d;
        } else if (inputLayout == 2) { // BNSD
            batchSqLoopOffset = bIdx * bCvInner * n * g * sQ * d;
            batchSkvLoopOffset = bIdx * bCvInner * n * sKV * d;
        } else {
            batchSqLoopOffset = bIdx * sQ * bCvInner * n * g * d;
            batchSkvLoopOffset = bIdx * sKV * bCvInner * n * d;
        }

        if constexpr (MMNext_OUT_FORMAT == CubeFormat::NZ) {
            mmNzDqOffset = bIdx * bCvInner * n * g * sQ * originalDAlign;
            mmNzDkvOffset = bIdx * bCvInner * n * sKV * originalDAlign;
        }

        innerMatResNum = bInNGSq * sKVAlign;

        maskInputNum = bInNGSq * sKVAlignByte;
        innerReduceNum = bInNGSq * BIT_SIZE;

        innerMatOutShape[0] = bInNGSq;
        innerMatOutShape[1] = sKVAlign;
        innerReduceShape[0] = bInNGSq;
        innerReduceShape[1] = BIT_SIZE;

        uint32_t bCvInnerOffset = bIdx * bCvInner;
        bCvInner = bCvInner < (b - bIdx * bCvInner) ? bCvInner : (b - bIdx * bCvInner);
        uint32_t bCvLoop = (bCvInner + bIn - 1) / bIn;

        // 发射本轮bmm12
        if (inputLayout == 1 || inputLayout == 2) {
            // SBH, BNSD
            mm1.SetTensorA(this->dxGm[batchSqLoopOffset]);
            mm1.SetTensorB(this->valueGm[batchSkvLoopOffset], true);
            mm1.template IterateBatch<false, true>(matmulResultBuffer1, bCvInner * n * g, bCvInner * n, false);

            mm1.SetTensorA(this->queryGm[batchSqLoopOffset]);
            mm1.SetTensorB(this->keyGm[batchSkvLoopOffset], true);
            mm1.template IterateBatch<false, true>(matmulResultBuffer2, bCvInner * n * g, bCvInner * n, false);
        } else if (currentBatchRange > 1) {
            // 当循环大于1时，cv并行才可以显现效果，如果循环次数为1，保持原来的bmm12计算位置
            // BSH只能同步
            // bmm1
            mm1.SetTail(sQ, sKV, d);
            mm1.SetTensorA(this->dxGm[batchSqLoopOffset]);
            mm1.SetTensorB(this->valueGm[batchSkvLoopOffset], true);
            for (uint32_t i = 0; i < bCvInner; i++) {
                mm1.SetTensorA(this->dxGm[batchSqLoopOffset + i * n * g * sQ * d]);
                mm1.SetTensorB(this->valueGm[batchSkvLoopOffset + i * n * sKV * d], true);
                if constexpr (MMPre_OUT_FORMAT == CubeFormat::NZ) {
                    mm1.template IterateBatch<true>(matmulResultBuffer1[i * n * g * sQ * sKVAlign], n * g, n, false);
                } else {
                    mm1.template IterateBatch<true>(matmulResultBuffer1[i * n * g * sQ * sKV], n * g, n, false);
                }
            }
            // BSH 只支持同步
            // bmm2
            mm1.SetTail(sQ, sKV, d);
            mm1.SetTensorA(this->queryGm[batchSqLoopOffset]);
            mm1.SetTensorB(this->keyGm[batchSkvLoopOffset], true);
            for (uint32_t i = 0; i < bCvInner; i++) {
                mm1.SetTensorA(this->queryGm[batchSqLoopOffset + i * n * g * sQ * d]);
                mm1.SetTensorB(this->keyGm[batchSkvLoopOffset + i * n * sKV * d], true);
                if constexpr (MMPre_OUT_FORMAT == CubeFormat::NZ) {
                    mm1.template IterateBatch<true>(matmulResultBuffer2[i * n * g * sQ * sKVAlign], n * g, n, false);
                } else {
                    mm1.template IterateBatch<true>(matmulResultBuffer2[i * n * g * sQ * sKV], n * g, n, false);
                }
            }
        }
        mm1.End();

        // 发射上一轮bmm345(第一轮不做bmm345)
        if (batchIdx > 0) {
            // 清除GM，否则pta连跑有问题
            ResetGm4Dkdv(inputLayout, lastBatchSkvLoopOffset, lastBCvInner, lastMmNzDkvOffset);
            if constexpr (MMNext_OUT_FORMAT == CubeFormat::NZ) {
                ClcMm31(dqWorkspaceGm[lastMmNzDqOffset], mulWorkspaceGm[lastPingpongIdx * pingPongMulOffset],
                        keyGm[lastBatchSkvLoopOffset], lastBCvInner, false);
                ClcMm32(dkWorkspaceGm[lastMmNzDkvOffset], mulWorkspaceGm[lastPingpongIdx * pingPongMulOffset],
                        queryGm[lastBatchSqLoopOffset], lastBCvInner, false);
            } else {
                // 异步
                ClcMm31(dqWorkspaceGm[lastBatchSqLoopOffset], mulWorkspaceGm[lastPingpongIdx * pingPongMulOffset],
                        keyGm[lastBatchSkvLoopOffset], lastBCvInner, false);
                ClcMm32(dkWorkspaceGm[lastBatchSkvLoopOffset], mulWorkspaceGm[lastPingpongIdx * pingPongMulOffset],
                        queryGm[lastBatchSqLoopOffset], lastBCvInner, false);
            }
            ClcMm4(dvGm[lastBatchSkvLoopOffset], dropWorkspaceGm[lastPingpongIdx * pingPongDropOffset],
                   dxGm[lastBatchSqLoopOffset], lastBCvInner, false);
        }

        // 发射本轮 vec
        for (uint32_t bCvIndex = 0; bCvIndex < bCvLoop; bCvIndex++) {
            if (inputLayout == 1) { // SBH即SBND
                bCvSqOffset = bCvIndex * bIn * n * g * d;
                bCvSkvOffset = bCvIndex * bIn * n * g * d;
            } else if (inputLayout == 2) { // BNSD
                bCvSqOffset = bCvIndex * bInNGSq * d;
                bCvSkvOffset = bCvIndex * bIn * n * g * sKV * d;
            } else {
                bCvSqOffset = bCvIndex * bInNGSq * d;
                bCvSkvOffset = bCvIndex * bIn * n * g * sKV * d;
            }

            bCvMmOffset = (bCvIndex * bIn) * n * g * sQ * sKV;
            bCvNzMmOffset = bCvIndex * bIn * n * g * sQ * sKVAlign;
            bCvDropMaskOffset = (bCvIndex * bIn) * n * g * sQ * sKV;

            if (sKV % DROPOUT4BIT_LEN == 0) {
                bCvDropMaskOffset = (bCvIndex * bIn) * n * g * sQ * sKV / 8;
            }

            if (existPse != 0) {
                pseInfo.boIdx = bCvInnerOffset + bCvIndex * bIn;
                pseInfo.bSSOffset = pseInfo.boIdx * sQ * sKV;
                pseInfo.s2SizeAcc = pseInfo.boIdx * sKV;
            }

            if (isDrop) {
                // for compute dropout mask offset
                dropMaskInfo.bSSOffset = (bCvInnerOffset + bCvIndex * bIn) * sQ * sKV;
            }

            if (hasAttenMask == 1) {
                int64_t compressMode = ordTilingData_->opInfo.attenMaskCompressMode;
                if (compressMode == 1 || compressMode == 2 || attenMaskShapeType == 0) {
                    bCvAttenMaskOffset = 0;
                } else if (attenMaskShapeType == 1) { // B1SS
                    bCvAttenMaskOffset = bCvIndex * bIn * sQ * sKV;
                } else { // BNSS
                    bCvAttenMaskOffset = bCvIndex * bInNGSq * sKV;
                }
            }

            int64_t bCvSoftmaxInputOffset = bCvIndex * bInNGSq * BIT_SIZE;
            if (bCvIndex == bCvLoop - 1) {
                bIn = bCvInner - (bCvLoop - 1) * bIn;
                isCvTail = true;

                // tail时 重新赋值bIn相关参数
                bInNGSq = bIn * n * g * sQ;
                innerMatResNum = bInNGSq * sKVAlign;
                maskInputNum = bInNGSq * sKVAlignByte;
                innerReduceNum = bInNGSq * BIT_SIZE;
                if (pseShapeType == PSE_BNSS) {
                    pseInputNum = bIn * n * g * pseSq * sKVAlign;
                } else if (pseShapeType == PSE_BN1S) {
                    pseInputNum = bIn * n * g * 1 * sKVAlign;
                } else {
                    pseInputNum = n * g * pseSq * sKVAlign;
                }
                innerMatOutShape[0] = bInNGSq;
                innerMatOutShape[1] = sKVAlign;
                innerReduceShape[0] = bInNGSq;
                innerReduceShape[1] = 8;
            }

            if (existPse != 0) {
                pseInfo.blockCount = pseShapeType == PSE_1NSS ? 1 * n * g : bIn * n * g;
                pseInfo.vec1S1RealSize = pseShapeType == PSE_1NSS ? 1 * n * g * sQ : bIn * n * g * sQ;
            }

            if (isDrop) {
                // for copy in dropout mask
                dropMaskInfo.s1CopySize = bInNGSq;
                dropMaskInfo.s2CopySize = sKV;
                dropMaskInfo.s2TotalSize = sKV;
            }

            FrontCompute(batchSqLoopOffset + bCvSqOffset, batchSkvLoopOffset + bCvSkvOffset,
                         dropMaskOffset + bCvDropMaskOffset, bCvMmOffset, bCvNzMmOffset, bCvIndex, currentBatchRange);
            ReCompute(batchSqLoopOffset, batchSkvLoopOffset, batchSqLoopOffset + bCvSqOffset,
                      batchSkvLoopOffset + bCvSkvOffset, attenMaskOffset + bCvAttenMaskOffset,
                      dropMaskOffset + bCvDropMaskOffset, batchSoftmaxInputOffset + bCvSoftmaxInputOffset, bCvMmOffset,
                      bCvNzMmOffset, bCvIndex, isCvTail, currentBatchRange);
        }

        // 最后一个循环 直接发射本轮bmm345
        if (batchIdx + 1 >= currentBatchRange) {
          // 清除GM，否则pta连跑有问题
          ResetGm4Dkdv(inputLayout, batchSkvLoopOffset, bCvInner, mmNzDkvOffset);

            if constexpr (MMNext_OUT_FORMAT == CubeFormat::NZ) {
                ClcMm31(dqWorkspaceGm[mmNzDqOffset], mulWorkspaceGm[pingpongIdx * pingPongMulOffset],
                        keyGm[batchSkvLoopOffset], bCvInner, true);
                ClcMm32(dkWorkspaceGm[mmNzDkvOffset], mulWorkspaceGm[pingpongIdx * pingPongMulOffset],
                        queryGm[batchSqLoopOffset], bCvInner, true);
            } else {
                ClcMm31(dqWorkspaceGm[batchSqLoopOffset], mulWorkspaceGm[pingpongIdx * pingPongMulOffset],
                        keyGm[batchSkvLoopOffset], bCvInner, true);
                ClcMm32(dkWorkspaceGm[batchSkvLoopOffset], mulWorkspaceGm[pingpongIdx * pingPongMulOffset],
                        queryGm[batchSqLoopOffset], bCvInner, true);
            }
            ClcMm4(dvGm[batchSkvLoopOffset], dropWorkspaceGm[pingpongIdx * pingPongDropOffset],
                   dxGm[batchSqLoopOffset], bCvInner, true);
        }
        // 备份本轮bmm345地址
        lastBCvInner = bCvInner;
        lastBatchSqLoopOffset = batchSqLoopOffset;
        lastBatchSkvLoopOffset = batchSkvLoopOffset;
        if constexpr (MMNext_OUT_FORMAT == CubeFormat::NZ) {
            lastMmNzDqOffset = mmNzDqOffset;
            lastMmNzDkvOffset = mmNzDkvOffset;
        }
        lastPingpongIdx = pingpongIdx;
    }
    SyncAll();
}

#endif // FLASH_ATTENTION_SCORE_GRAD_BNGS1S2_B_H_
