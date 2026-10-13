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
 * \file flash_attention_score_grad_ngs1s2_bn.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_NGS1S2_BN_H_
#define FLASH_ATTENTION_SCORE_GRAD_NGS1S2_BN_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "pse.h"
#include "dropmask.h"

constexpr uint32_t BN_DROPOUT4BIT_LEN = 16;

using matmul::Matmul;
using matmul::MatmulType;

__aicore__ const constexpr MatmulConfig GetMm1Cfg31(S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType) {
    if (dTemplateType == DTemplateType::NotAligned72) {
        if (s1TemplateType == S1TemplateType::Aligned48 && s2TemplateType == S2TemplateType::Aligned48) {
            // FAG_CP_CFG_31_D72_S1S2_48_MM1_MM2;
            return MatmulConfig{true, false, false, 48, 48, 80, false, false, false, false,
                                0, 48, 48, 72, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        } else if (s1TemplateType == S1TemplateType::NotAligned60 && s2TemplateType == S2TemplateType::NotAligned60) {
            // FAG_CP_CFG_31_D72_S1S2_60_MM1_MM2;
            return MatmulConfig{true, false, false, 64, 64, 80, false, false, false, false,
                                0, 60, 60, 72, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        } else if (s1TemplateType == S1TemplateType::Aligned64 && s2TemplateType == S2TemplateType::Aligned64) {
            // FAG_CP_CFG_31_D72_S1S2_64_MM1_MM2;
            return MatmulConfig{true, false, false, 64, 64, 80, false, false, false, false,
                                0, 64, 64, 72, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        } else if (s1TemplateType == S1TemplateType::NotAligned30 && s2TemplateType == S2TemplateType::NotAligned30) {
            // FAG_CP_CFG_31_D72_S1S2_30_MM1_MM2;
            return MatmulConfig{true, false, false, 32, 32, 80, false, false, false, false,
                                0, 30, 30, 72, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        } else if (s1TemplateType == S1TemplateType::Aligned32 && s2TemplateType == S2TemplateType::Aligned32) {
            // FAG_CP_CFG_31_D72_S1S2_32_MM1_MM2;
            return MatmulConfig{true, false, false, 32, 32, 80, false, false, false, false,
                                0, 32, 32, 72, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        }
    }
    return FAG_CP_CFG_31_41_DEFAULT;
}

__aicore__ const constexpr MatmulConfig GetMm2Cfg31(S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType) {
    if (dTemplateType == DTemplateType::NotAligned72) {
        if (s1TemplateType == S1TemplateType::Aligned48 && s2TemplateType == S2TemplateType::Aligned48) {
            // FAG_CP_CFG_31_D72_S1S2_48_MM31;
            return MatmulConfig{true, false, false, 48, 80, 48, false, false, false, false,
                                0, 48, 72, 48, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        } else if (s1TemplateType == S1TemplateType::NotAligned60 && s2TemplateType == S2TemplateType::NotAligned60) {
            // FAG_CP_CFG_31_D72_S1S2_60_MM31;
            return MatmulConfig{true, false, false, 64, 80, 64, false, false, false, false,
                                0, 60, 72, 60, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        } else if (s1TemplateType == S1TemplateType::Aligned64 && s2TemplateType == S2TemplateType::Aligned64) {
            // FAG_CP_CFG_31_D72_S1S2_64_MM31;
            return MatmulConfig{true, false, false, 64, 80, 64, false, false, false, false,
                                0, 64, 72, 64, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        } else if (s1TemplateType == S1TemplateType::NotAligned30 && s2TemplateType == S2TemplateType::NotAligned30) {
            // FAG_CP_CFG_31_D72_S1S2_30_MM31;
            return MatmulConfig{true, false, false, 32, 80, 32, false, false, false, false,
                                0, 30, 72, 30, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        } else if (s1TemplateType == S1TemplateType::Aligned32 && s2TemplateType == S2TemplateType::Aligned32) {
            // FAG_CP_CFG_31_D72_S1S2_32_MM31;
            return MatmulConfig{true, false, false, 32, 80, 32, false, false, false, false,
                                0, 32, 72, 32, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        }
    }
    return FAG_CP_CFG_31_41_DEFAULT;
}

__aicore__ const constexpr MatmulConfig GetMm3Cfg31(S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType) {
    if (dTemplateType == DTemplateType::NotAligned72) {
        if (s1TemplateType == S1TemplateType::Aligned48 && s2TemplateType == S2TemplateType::Aligned48) {
            // FAG_CP_CFG_31_D72_S1S2_48_MM32_MM4;
            return MatmulConfig{true, false, false, 48, 80, 48, false, false, false, false,
                                0, 48, 72, 48, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        } else if (s1TemplateType == S1TemplateType::NotAligned60 && s2TemplateType == S2TemplateType::NotAligned60) {
            // FAG_CP_CFG_31_D72_S1S2_60_MM32_MM4;
            return MatmulConfig{true, false, false, 64, 80, 64, false, false, false, false,
                                0, 60, 72, 60, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        } else if (s1TemplateType == S1TemplateType::Aligned64 && s2TemplateType == S2TemplateType::Aligned64) {
            // FAG_CP_CFG_31_D72_S1S2_64_MM32_MM4;
            return MatmulConfig{true, false, false, 64, 80, 64, false, false, false, false,
                                0, 64, 72, 64, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        } else if (s1TemplateType == S1TemplateType::NotAligned30 && s2TemplateType == S2TemplateType::NotAligned30) {
            // FAG_CP_CFG_31_D72_S1S2_30_MM32_MM4;
            return MatmulConfig{true, false, false, 32, 80, 32, false, false, false, false,
                                0, 30, 72, 30, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        } else if (s1TemplateType == S1TemplateType::Aligned32 && s2TemplateType == S2TemplateType::Aligned32) {
            // FAG_CP_CFG_31_D72_S1S2_32_MM32_MM4;
            return MatmulConfig{true, false, false, 32, 80, 32, false, false, false, false,
                                0, 32, 72, 32, 0, 0, 0, 0, true, false, false, false, false, true,
                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};
        }
    }
    return FAG_CP_CFG_31_41_DEFAULT;
}

#if __CCE_KT_TEST__
#define MIX_LOG1(...)                                                                                                  \
    do {                                                                                                               \
        if (mBlockIdx == 0) {                                                                                          \
            MIX_LOG(__VA_ARGS__);                                                                                      \
        }                                                                                                              \
    } while (0)

#else
#define MIX_LOG1(format, ...)
#endif

// T1 for data, T2 for vecClc
template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16 = false,
          LayoutMode layout = LayoutMode::BNGS1S2, const CubeFormat MMPre_OUT_FORMAT = CubeFormat::ND,
          const CubeFormat MMNext_OUT_FORMAT = CubeFormat::ND,
          S1TemplateType s1TemplateType = S1TemplateType::NotAligned,
          S2TemplateType s2TemplateType = S2TemplateType::NotAligned,
          DTemplateType dTemplateType = DTemplateType::NotAligned>
class FlashAttentionScoreGradUngs1s2Bbn {
public:
    __aicore__ inline FlashAttentionScoreGradUngs1s2Bbn(){};
    __aicore__ inline void Init(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query,
                                __gm__ uint8_t *pse_shift, __gm__ uint8_t *drop_mask, __gm__ uint8_t *atten_mask,
                                __gm__ uint8_t *forward_res, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,
                                __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *workspace,
                                const FlashAttentionScoreGradTilingDataUngs1s2Bbn *__restrict ordTilingData,
                                TPipe *pipe_in);

    __aicore__ inline void Process();

    constexpr static MatmulConfig MM_1_2_CONFIG31 =
        GetMm1Cfg31(s1TemplateType, s2TemplateType, dTemplateType);
    constexpr static MatmulConfig MM_31_CONFIG31 =
        GetMm2Cfg31(s1TemplateType, s2TemplateType, dTemplateType);
    constexpr static MatmulConfig MM_32_4_CONFIG31 =
        GetMm3Cfg31(s1TemplateType, s2TemplateType, dTemplateType);

    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    using GmT1TrueLayout = MatmulType<TPosition::GM, CubeFormat::ND, T1, true, layout>;
    using GmT1FalseLayout = MatmulType<TPosition::GM, CubeFormat::ND, T1, false, layout>;

    using GmT1TrueBNSS = MatmulType<TPosition::GM, CubeFormat::ND, T1, true, LayoutMode::BNGS1S2>;
    using GmT1FalseBNSS = MatmulType<TPosition::GM, CubeFormat::ND, T1, false, LayoutMode::BNGS1S2>;

    using GmT2FalseBNSS = MatmulType<TPosition::GM, MMPre_OUT_FORMAT, T2, false, LayoutMode::BNGS1S2>;

    using GmT2FalseBNSS0 = MatmulType<TPosition::GM, MMNext_OUT_FORMAT, T2, false, LayoutMode::BNGS1S2>;
    using GmT2FalseBNSS1 = MatmulType<TPosition::GM, CubeFormat::ND, T2, false, layout>;

    using GmT1FalseBNS2 = MatmulType<TPosition::GM, CubeFormat::ND, T1, false, layout>;

    // matmul 1 2 config
    constexpr MatmulApiStaticTiling static CFG_NORM_TILING_MM1 = GetMatmulApiTiling<GmT1FalseLayout, GmT1TrueLayout, GmT2FalseBNSS, biasType>(MM_1_2_CONFIG31);
    Matmul<GmT1FalseLayout, GmT1TrueLayout, GmT2FalseBNSS, biasType, CFG_NORM_TILING_MM1> mm1;

    // matmul 31 config
    constexpr MatmulApiStaticTiling static CFG_NORM_TILING_MM31 = GetMatmulApiTiling<GmT1FalseBNSS, GmT1FalseLayout, GmT2FalseBNSS0, biasType>(MM_31_CONFIG31);
    using modeTypeMmDq = typename AscendC::Conditional<
        (MMNext_OUT_FORMAT == CubeFormat::NZ),
        Matmul<GmT1FalseBNSS, GmT1FalseLayout, GmT2FalseBNSS0, biasType, CFG_NORM_TILING_MM31>,
        Matmul<GmT1FalseBNSS, GmT1FalseLayout, GmT2FalseBNSS1, biasType, CFG_NORM_TILING_MM31>>::type;
    modeTypeMmDq mm31;

    // matmul 32 config
    constexpr MatmulApiStaticTiling static CFG_NORM_TILING_MM32 = GetMatmulApiTiling<GmT1TrueBNSS, GmT1FalseLayout, GmT2FalseBNSS0, biasType>(MM_32_4_CONFIG31);
    using modeTypeMmDk = typename AscendC::Conditional<
        (MMNext_OUT_FORMAT == CubeFormat::NZ),
        Matmul<GmT1TrueBNSS, GmT1FalseLayout, GmT2FalseBNSS0, biasType, CFG_NORM_TILING_MM32>,
        Matmul<GmT1TrueBNSS, GmT1FalseLayout, GmT2FalseBNSS1, biasType, CFG_NORM_TILING_MM32>>::type;
    modeTypeMmDk mm32;

    // matmul 4 config
    constexpr MatmulApiStaticTiling static CFG_NORM_TILING_MM4 = GetMatmulApiTiling<GmT1TrueBNSS, GmT1FalseLayout, GmT1FalseBNS2, biasType>(MM_32_4_CONFIG31);
    Matmul<GmT1TrueBNSS, GmT1FalseLayout, GmT1FalseBNS2, biasType, CFG_NORM_TILING_MM4> mm4;

protected:
    /* define the que */
    TQue<QuePosition::VECIN, 1> vecInQue1;
    TQue<QuePosition::VECIN, 1> vecInQue2;
    TBuf<> vecClc1;
    TBuf<> vecClc2;
    TBuf<> vecCast;
    TQue<QuePosition::VECOUT, 1> apiClcQue;

    const FlashAttentionScoreGradTilingDataUngs1s2Bbn *__restrict ordTilingData_;

    TPipe *pipe;

    GlobalTensor<uint8_t> attenMaskU8Gm;
    GlobalTensor<T1> keyGm, valueGm, dxGm, queryGm, attenMaskGm, forwardResGm, pseGm;
    GlobalTensor<T1> dqGm, dkGm, dvGm;
    GlobalTensor<T2> dqWorkspaceGm, dkWorkspaceGm;

    GlobalTensor<float> softmaxMaxGm, softmaxSumGm;
    GlobalTensor<T2> workspaceGm;
    GlobalTensor<uint8_t> dropoutWorkspaceGm, dropMaskGm;

    GlobalTensor<T1> dropWorkSpaceGm, mulWorkSpaceGm;

    // matmal1/matmal2 result buffer
    GlobalTensor<float> matmulResultBuffer1;
    GlobalTensor<float> matmulResultBuffer2;

    PseInfo pseInfo = {0};

    int64_t b;
    int64_t n;
    int64_t g;
    int64_t sQ;
    int64_t pseSq;
    int64_t sKV;
    int64_t sKVAlign;
    int64_t sKVAlignVec;
    int64_t sKVAlignByte;
    int64_t sKVStride;
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
    uint32_t inputLayout; // 0:BSH 1:SBH 2:BNSD
    int64_t preTokensBlocks;
    int64_t nextTokensBlocks;
    uint32_t inputDType;
    uint32_t inputDTypeSize;
    uint32_t vecCalcDTypeSize;
    uint32_t hasAttenMask;
    uint32_t attenMaskShapeType;
    uint32_t elementPerBlock;
    uint32_t pseShapeType;

    int64_t totalBatch;
    int64_t sKVAlignSize;
    int64_t nOut;
    int64_t apiClcQueueSize;
    int64_t usedCoreNum;

    int64_t nIn;
    int64_t nInTail;
    int64_t oriNIn;
    uint32_t singleCoreBatchRange;
    uint32_t singleCoreBatchRangeTail;
    int64_t nCvInner;
    int64_t mm1WorkspaceLen;
    int64_t mm2WorkspaceLen;
    int64_t dqWorkspaceLen;
    int64_t dkWorkspaceLen;
    int64_t dropGmWorkspaceLen;
    int64_t mulGmWorkspaceLen;
    int64_t innerTmpBufSize;
    int64_t vecCastSize;
    int64_t splitedDAlign;
    int64_t dRange;

    int64_t subRange;
    int64_t subMask;
    int64_t subMaskTail;
    int64_t sKVAlignBlockNumVec;

    int32_t innerReduceNum;
    int32_t innerMaskNum;
    int32_t innerMatResNum;
    int64_t innerMatResNumVec;
    int64_t maskInputNum;

    uint32_t softmaxGradInputShape[2];
    uint32_t innerMatOutShape[2];
    uint32_t maskInputShape[2];
    uint32_t innerReduceShape[2];
    bool isDrop;
    int64_t dropoutWorkspaceLen;
    int64_t mBlockIdx;

    int64_t previousBatchCnt;
    const int64_t FP32_PER_BLOCK_NUM = 8;
    // for nz
    constexpr static int64_t C0_SIZE = 16;
    constexpr static int64_t VEC_REPEAT = 8;
    constexpr static uint32_t CAL_BLOCK_NUM = 32 / sizeof(T2);
    constexpr static uint32_t BLOCK = 32;
    constexpr static int64_t PER_COPY_MAX_NUM = 255;

    int64_t pingPongDropOffset;
    int64_t pingPongMulOffset;
    uint32_t pingpongIdx = 1;
    uint32_t lastPingpongIdx = 0;
    int64_t lastNCvInner = 0;

    DropMaskInfo dropMaskInfo = {0};
    __aicore__ inline void FrontCompute(const int64_t &batchSqLoopOffset, const int64_t &batchSkvLoopOffset,
                                        const int64_t &dropMaskOffset, const int64_t &nCvIndex);

    __aicore__ inline void ReCompute(const int64_t &batchSqLoopOffset, const int64_t &batchSkvLoopOffset,
                                     const int64_t &attenMaskOffset, const int64_t &dropMaskOffset,
                                     const int64_t &batchSoftmaxInputOffset, const int64_t &nCvIndex);

    __aicore__ inline void SetFrontClcShape(LocalTensor<T2> &sftFrontResInner, LocalTensor<T2> &frontResInner,
                                            LocalTensor<uint8_t> &dpMaskInner);

    __aicore__ inline void CopyInSoftMaxGrad(LocalTensor<T1> &dxInner, const GlobalTensor<T1> &dxGmIn,
                                             LocalTensor<T1> &forwardResInner, const GlobalTensor<T1> &forwardResGmIn,
                                             int64_t thisDAlign, int64_t thisD, int64_t dAlignOffset);

    __aicore__ inline void ClcSoftMaxGrad(LocalTensor<T2> &softmaxGradOutUb, LocalTensor<T1> &dxUb,
                                          LocalTensor<T1> &attentionInUb, int64_t thisDAlign);

    __aicore__ inline void HandleSoftMaxGrad(const int64_t &batchSqLoopOffset, LocalTensor<T2> &softmaxGradOutUb);

    __aicore__ inline void ClcSoftMaxGradSplitD(LocalTensor<T2> &softmaxGradOutUb, LocalTensor<T1> &dxUb,
                                                LocalTensor<T1> &attentionInUb, int64_t thisDAlign);

    __aicore__ inline void HandleSoftMaxGradSplitD(const int64_t &batchSqLoopOffset, LocalTensor<T2> &softmaxGradOutUb);

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

    __aicore__ inline void ClcMm1(const int64_t &batchSqLoopOffset, const int64_t &batchSkvLoopOffset);

    __aicore__ inline void ClcMm2(const int64_t &batchSqLoopOffset, const int64_t &batchSkvLoopOffset);

    __aicore__ inline void ClcMm31(const int64_t &cvMulOffset, const int64_t &batchSqLoopOffset,
                                   const int64_t &batchSkvLoopOffset, const bool &isSync);
    /* matmul 32 和 matmul31的区别在于输入TensorA是需要做Transpose的，且输出需要做G轴的reduce. */
    __aicore__ inline void ClcMm32(const int64_t &cvMulOffset, const int64_t &batchSqLoopOffset,
                                   const int64_t &batchSkvLoopOffset, const bool &isSync);

    __aicore__ inline void ClcMm4(const int64_t &cvMulOffset, const int64_t &batchSqLoopOffset,
                                  const int64_t &batchSkvLoopOffset, const bool &isSync);

    __aicore__ inline void CopyToWorkspace(const int64_t &batchSqLoopOffset, const int64_t &batchSkvLoopOffset,
                                           LocalTensor<T2> &mulResInner, LocalTensor<T2> &dvDropResInner,
                                           const int64_t &nCvIndex);

    __aicore__ inline void Mm12OutProcess(GlobalTensor<T2> &matmulOutGm, LocalTensor<T2> &outLocalTensor, const int64_t nCvIndex);
};


#if __CCE_KT_TEST__
template <typename T1> inline void Ngs1s2BnPrint(const GlobalTensor<T1> global, uint32_t len1, uint32_t printLen)
{
    std::ostringstream os_;
    os_.str("");
    uint32_t blockNum = ONE_BLK_SIZE / sizeof(T1);
    const int32_t width = 4;

    for (uint32_t i = len1; i < printLen; i++) {
        if (i % blockNum == 0) {
            os_ << std::setw(width) << std::setfill('0') << (i / blockNum) * blockNum << " : ";
        }

        if constexpr (std::is_same<T1, float>::value) {
            os_ << global.GetValue(i) << " ";
        } else if constexpr (std::is_same<T1, uint8_t>::value) {
            os_ << ('0' + (uint8_t)(global.GetValue(i)) - '0') << " ";
        } else {
            os_ << global.GetValue(i).ToFloat() << " ";
        }

        if ((i + 1) % blockNum == 0) {
            os_ << std::endl;
        }
    }
    os_ << std::endl;
    std::cout << os_.str();
}

template <typename T1> inline void Ngs1s2BnPrint(const LocalTensor<T1> global, uint32_t len1, uint32_t printLen)
{
    std::ostringstream os_;
    os_.str("");
    uint32_t blockNum = ONE_BLK_SIZE / sizeof(T1);
    const int32_t width = 4;

    for (uint32_t i = len1; i < printLen; i++) {
        if (i % blockNum == 0) {
            os_ << std::setw(width) << std::setfill('0') << (i / blockNum) * blockNum << " : ";
        }

        if constexpr (std::is_same<T1, float>::value) {
            os_ << global.GetValue(i) << " ";
        } else if constexpr (std::is_same<T1, uint8_t>::value) {
            os_ << ('0' + (uint8_t)(global.GetValue(i)) - '0') << " ";
        } else {
            os_ << global.GetValue(i).ToFloat() << " ";
        }

        if ((i + 1) % blockNum == 0) {
            os_ << std::endl;
        }
    }
    os_ << std::endl;
    std::cout << os_.str();
}
template <typename T1>
inline void Ngs1s2BnRightPad(const LocalTensor<T1> global, int64_t dim1, int64_t dim2, uint32_t padNum, T1 value)
{
    for (int64_t i = 0; i < dim1; i++) {
        for (int64_t j = dim2 - padNum; j < dim2; j++) {
            global.SetValue(i * dim2 + j, value);
        }
    }
}
#endif // __CCE_KT_TEST__

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::Init(
    __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query, __gm__ uint8_t *pse_shift,
    __gm__ uint8_t *drop_mask, __gm__ uint8_t *atten_mask, __gm__ uint8_t *forward_res, __gm__ uint8_t *softmax_max,
    __gm__ uint8_t *softmax_sum, __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *workspace,
    const FlashAttentionScoreGradTilingDataUngs1s2Bbn *__restrict ordTilingData, TPipe *pipe_in)
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
    dropoutWorkspaceGm.SetGlobalBuffer((__gm__ uint8_t *)workspace);

    ordTilingData_ = ordTilingData;
    pipe = pipe_in;

    b = ordTilingData_->opInfo.b;
    n = ordTilingData_->opInfo.n;
    g = ordTilingData_->opInfo.g;
    sQ = ordTilingData_->opInfo.sQ;
    pseSq = ordTilingData_->opInfo.pseSq;
    pseShapeType = ordTilingData_->opInfo.pseShapeType;
    sKV = ordTilingData_->opInfo.sKV;
    sKVAlign = ordTilingData_->opInfo.sKVAlign;
    sKVAlignVec = ordTilingData_->opInfo.sKVAlignVec;
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
    elementPerBlock = ordTilingData_->opInfo.elementPerBlock;
    sKVAlignSize = ordTilingData_->opInfo.sKVAlignSize;

    totalBatch = ordTilingData_->splitCoreParams.totalBatch;
    nOut = ordTilingData_->splitCoreParams.nOut;
    apiClcQueueSize = ordTilingData_->splitCoreParams.apiClcQueueSize;
    usedCoreNum = ordTilingData_->splitCoreParams.usedCoreNum;

    nIn = ordTilingData_->singleCoreParams.nIn;
    nInTail = ordTilingData_->singleCoreParams.nInTail;
    oriNIn = ordTilingData_->singleCoreParams.nIn;
    singleCoreBatchRange = ordTilingData_->singleCoreParams.singleCoreBatchRange;
    singleCoreBatchRangeTail = ordTilingData_->singleCoreParams.singleCoreBatchRangeTail;
    nCvInner = ordTilingData_->singleCoreParams.nCvInner;
    mm1WorkspaceLen = ordTilingData_->opInfo.mm1WorkspaceLen;
    mm2WorkspaceLen = ordTilingData_->opInfo.mm2WorkspaceLen;
    dqWorkspaceLen = ordTilingData_->opInfo.dqWorkspaceLen;
    dkWorkspaceLen = ordTilingData_->opInfo.dkWorkspaceLen;
    dropGmWorkspaceLen = ordTilingData_->opInfo.dropGmWorkspaceLen;
    mulGmWorkspaceLen = ordTilingData_->opInfo.mulGmWorkspaceLen;
    pingPongDropOffset = dropGmWorkspaceLen / 2 / sizeof(T1);
    pingPongMulOffset = mulGmWorkspaceLen / 2 / sizeof(T1);
    innerTmpBufSize = ordTilingData_->singleCoreParams.innerTmpBufSize;
    vecCastSize = ordTilingData_->singleCoreParams.vecCastSize;
    splitedDAlign = ordTilingData_->singleCoreParams.splitedDAlign;
    dRange = ordTilingData_->singleCoreParams.dRange;

    subRange = ordTilingData_->singleCoreParams.subRange;
    subMask = ordTilingData_->singleCoreParams.subMask;
    subMaskTail = ordTilingData_->singleCoreParams.subMaskTail;
    sKVAlignBlockNumVec = ordTilingData_->singleCoreParams.sKVAlignBlockNumVec;
    sKVStride = (sKVAlignVec - sKV) / FP32_PER_BLOCK_NUM;

    dropoutWorkspaceLen = ordTilingData_->opInfo.dropoutWorkspaceLen;

    if (pseSq != 0) {
        pseInfo.pseShapeType = pseShapeType == 2 ? 0 : pseShapeType;
        pseInfo.pseBSize = pseShapeType == 2 ? 1 : b;
        pseInfo.s1BaseSize = sQ;
        pseInfo.gSize = g;
        pseInfo.s1Size = sQ;
        pseInfo.s2Size = sKV;
        pseInfo.n2G = n * g;
        pseInfo.s2RealSize = sKV;
        pseInfo.s2AlignedSize = sKVAlign;
        pseInfo.needCast = false;
    }

    int64_t mm1WorkspaceOffest = dropoutWorkspaceLen / sizeof(T2);
    int64_t mm2WorkspaceOffest = (dropoutWorkspaceLen + mm1WorkspaceLen) / sizeof(T2);

    int64_t batchMatRstNum = nCvInner * g * sQ * sKVAlignVec * mBlockIdx;

    matmulResultBuffer1.SetGlobalBuffer((__gm__ T2 *)workspace + mm1WorkspaceOffest + batchMatRstNum);
    matmulResultBuffer2.SetGlobalBuffer((__gm__ T2 *)workspace + mm2WorkspaceOffest + batchMatRstNum);

    int64_t usedWorkspaceLen = dropoutWorkspaceLen + mm1WorkspaceLen + mm2WorkspaceLen;
    auto dqAddr = usedWorkspaceLen / sizeof(T2);
    auto dkAddr = dqAddr + dqWorkspaceLen / sizeof(T2);
    dqWorkspaceGm.SetGlobalBuffer((__gm__ T2 *)workspace + dqAddr);
    dkWorkspaceGm.SetGlobalBuffer((__gm__ T2 *)workspace + dkAddr);
    // 对于dropout mask其size是nIn * g * sQ * sKVAlignByte， sKVAlignByte这个值一定小于等于sKVAlignSize
    // 例如sKVALign是17，sKVAlignByte = 32， sKVAlignSize = 64
    // sKVAlign是32， sKVAlignByte = 32， sKVAlignSize = 64
    int64_t pseAndMaskInputSize = nIn * g * sQ * sKVAlignSize;
    pipe->InitBuffer(vecInQue1, 2, ordTilingData_->singleCoreParams.vecQueIn1Size);
    pipe->InitBuffer(vecInQue2, 1, pseAndMaskInputSize);
    // Bf16场景下vecClc1和vecClc2会被softmax的输入的Cast复用
    // tmpBufSize在计算时会取splitedDAlign和sKvAlign的较大值
    pipe->InitBuffer(vecClc1, innerTmpBufSize);
    pipe->InitBuffer(vecClc2, innerTmpBufSize);
    // D切分才需要申请额外的vecCast用于attenin的cast
    if (vecCastSize > 0) {
        pipe->InitBuffer(vecCast, vecCastSize);
    }
    pipe->InitBuffer(apiClcQue, 1, apiClcQueueSize);

    // drop workspace offset
    int64_t workspaceOffsets = dkAddr * sizeof(T2) + dkWorkspaceLen;
    dropWorkSpaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + workspaceOffsets / sizeof(T1));

    // mul workspace offset
    workspaceOffsets = workspaceOffsets + dropGmWorkspaceLen;
    mulWorkSpaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + workspaceOffsets / sizeof(T1));

    DropOutBitModeInit();

    bool boolMode = true;
    if (sKV % BN_DROPOUT4BIT_LEN == 0) {
        boolMode = false;
    }

    isDrop = false;
    if (keepProb < 1 && dropoutWorkspaceLen > 0) {
        isDrop = true;

        // for compute dropout mask offset
        dropMaskInfo.n2G = n * g;
        dropMaskInfo.gSize = g;
        dropMaskInfo.s1Size = sQ;
        dropMaskInfo.s2Size = sKV;

        // for copy in dropout mask
        dropMaskInfo.s2CopySize = sKV;
        dropMaskInfo.s2TotalSize = sKV;

        // for compute dropout mask
        dropMaskInfo.keepProb = keepProb;
        dropMaskInfo.boolMode = boolMode;

        // for compute dropout mask
        dropMaskInfo.lstAxis = sKVAlignVec;
        dropMaskInfo.maskLstAxis = sKVAlignVec;
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::CopyInSoftMaxGrad(
    LocalTensor<T1> &dxInner, const GlobalTensor<T1> &dxGmIn, LocalTensor<T1> &forwardResInner,
    const GlobalTensor<T1> &forwardResGmIn, int64_t thisDAlign, int64_t thisD, int64_t dAlignOffset)
{
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<T1> copyPadParams;
    int64_t ubOffset = 0;
    int64_t gmOffset = 0;
    copyParams.blockCount = sQ;
    copyParams.blockLen = thisD * sizeof(T1);
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    copyPadParams.isPad = true;
    copyPadParams.leftPadding = 0;
    copyPadParams.rightPadding = (thisDAlign - thisD);
    copyPadParams.paddingValue = 0;

    for (int64_t copyIndex = 0; copyIndex < static_cast<int64_t>(nIn * g); ++copyIndex) {
        if (inputLayout == 1) { // SBH SBND
            gmOffset = copyIndex * d + dAlignOffset;
            copyParams.srcStride = (b * hQ - thisD) * sizeof(T1);
        } else if (inputLayout == 2) { // BNSD
            gmOffset = copyIndex * sQ * d + dAlignOffset;
            copyParams.srcStride = (d - thisD) * sizeof(T1);
        } else { // BSH BSDN
            gmOffset = copyIndex * d + dAlignOffset;
            copyParams.srcStride = (hQ - thisD) * sizeof(T1);
        }

        DataCopyPad(dxInner[ubOffset], dxGmIn[gmOffset], copyParams, copyPadParams);
        DataCopyPad(forwardResInner[ubOffset], forwardResGmIn[gmOffset], copyParams, copyPadParams);
        ubOffset += sQ * thisDAlign;
    }
    vecInQue1.EnQue(forwardResInner); // enque,等待数据copy，dxUb和dxUb同一个que的，共用一个enque操作
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::SetFrontClcShape(
    LocalTensor<T2> &sftFrontResUb, LocalTensor<T2> &frontResUb, LocalTensor<uint8_t> &dpMaskUb)
{
    uint32_t innerFrontReClcShape[2] = {static_cast<uint32_t>(sQ), static_cast<uint32_t>(nIn * g * (32 / sizeof(T2)))};
    sftFrontResUb.SetShapeInfo(ShapeInfo(2, innerFrontReClcShape, DataFormat::ND));
    frontResUb.SetShapeInfo(ShapeInfo(2, innerMatOutShape, DataFormat::ND));
    dpMaskUb.SetShapeInfo(ShapeInfo(2, innerMatOutShape, DataFormat::ND));
}

/* D不切分 直接复用softmaxGradOutUb结果内存，计算的结果放在softmaxGradOutUb */
template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ClcSoftMaxGrad(
    LocalTensor<T2> &softmaxGradOutUb, LocalTensor<T1> &dxUb, LocalTensor<T1> &attentionInUb, int64_t thisDAlign)
{
    vecInQue1.DeQue<T1>(); // deque一把等待数据copy完，dxUb和dxUb同一个que的，共用一个deque操作
    auto softmaxGradInputNum = softmaxGradInputShape[0] * thisDAlign;
    bool isBasicBlock = (sQ % 8 == 0) && (thisDAlign % 64 == 0);
    LocalTensor<uint8_t> apiClcTensor = apiClcQue.AllocTensor<uint8_t>();
    apiClcTensor.SetSize(apiClcQueueSize);

    if constexpr (IS_BF16) {
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<T2> castedAttentionIn = vecClc2.Get<T2>(softmaxGradInputNum); // 临时借用softmaxGradOutUb内存
        LocalTensor<T2> castedDx = vecClc1.Get<T2>(softmaxGradInputNum);
        castedAttentionIn.SetShapeInfo(ShapeInfo(2, softmaxGradInputShape, DataFormat::ND));
        castedDx.SetShapeInfo(ShapeInfo(2, softmaxGradInputShape, DataFormat::ND));
        Cast(castedAttentionIn, attentionInUb, RoundMode::CAST_NONE, softmaxGradInputNum);
        Cast(castedDx, dxUb, RoundMode::CAST_NONE, softmaxGradInputNum);
        AscendC::PipeBarrier<PIPE_V>();

        if (isBasicBlock) {
            SoftmaxGradFront<T2, true>(softmaxGradOutUb, castedAttentionIn, castedDx, apiClcTensor,
                                       ordTilingData_->softmaxGradTilingData);
        } else {
            SoftmaxGradFront<T2, false>(softmaxGradOutUb, castedAttentionIn, castedDx, apiClcTensor,
                                        ordTilingData_->softmaxGradTilingData);
        }
    } else {
        if (isBasicBlock) {
            SoftmaxGradFront<T2, true>(softmaxGradOutUb, attentionInUb, dxUb, apiClcTensor,
                                       ordTilingData_->softmaxGradTilingData);
        } else {
            SoftmaxGradFront<T2, false>(softmaxGradOutUb, attentionInUb, dxUb, apiClcTensor,
                                        ordTilingData_->softmaxGradTilingData);
        }
    }

    apiClcQue.FreeTensor(apiClcTensor);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::HandleSoftMaxGrad(
    const int64_t &batchSqLoopOffset, LocalTensor<T2> &softmaxGradOutUb)
{
    LocalTensor<T1> attentionInUb = vecInQue1.AllocTensor<T1>();
    LocalTensor<T1> dxUb = vecInQue1.AllocTensor<T1>();
    int64_t softmaxGradInputNum = softmaxGradInputShape[0] * originalDAlign;
    dxUb.SetSize(softmaxGradInputNum);
    attentionInUb.SetSize(softmaxGradInputNum);

    softmaxGradInputShape[1] = originalDAlign;
    dxUb.SetShapeInfo(ShapeInfo(2, softmaxGradInputShape, DataFormat::ND));
    attentionInUb.SetShapeInfo(ShapeInfo(2, softmaxGradInputShape, DataFormat::ND));

    CopyInSoftMaxGrad(dxUb, dxGm[batchSqLoopOffset], attentionInUb, forwardResGm[batchSqLoopOffset], originalDAlign, d,
                      0);

    ClcSoftMaxGrad(softmaxGradOutUb, dxUb, attentionInUb, originalDAlign);

    vecInQue1.FreeTensor(attentionInUb);
    vecInQue1.FreeTensor(dxUb);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ClcSoftMaxGradSplitD(
    LocalTensor<T2> &softmaxGradOutUb, LocalTensor<T1> &dxUb, LocalTensor<T1> &attentionInUb, int64_t thisDAlign)
{
    vecInQue1.DeQue<T1>(); // deque一吧等待数据copyin完，dxUb和dxUb同一个que的，共用一个deque操作
    auto softmaxGradInputNum = softmaxGradInputShape[0] * thisDAlign;
    bool isBasicBlock = (sQ % 8 == 0) && (thisDAlign % 64 == 0);
    LocalTensor<uint8_t> apiClcTensor = apiClcQue.AllocTensor<uint8_t>(); // apiClcTensor可以外提
    apiClcTensor.SetSize(apiClcQueueSize);

    /* 这里复用了Calc1，这时mm还没有GetTensorC，Clc1处于闲置状态 */
    LocalTensor<T2> softmaxGradTempRes = vecClc1.Get<T2>(softmaxGradOutUb.GetSize());
    if constexpr (IS_BF16) {
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<T2> castedAttentionIn = vecCast.Get<T2>(softmaxGradInputNum);
        LocalTensor<T2> castedDx = vecClc1.Get<T2>(softmaxGradInputNum);
        castedAttentionIn.SetShapeInfo(ShapeInfo(2, softmaxGradInputShape, DataFormat::ND));
        castedDx.SetShapeInfo(ShapeInfo(2, softmaxGradInputShape, DataFormat::ND));
        Cast(castedAttentionIn, attentionInUb, RoundMode::CAST_NONE, softmaxGradInputNum);
        Cast(castedDx, dxUb, RoundMode::CAST_NONE, softmaxGradInputNum);
        AscendC::PipeBarrier<PIPE_V>();

        if (isBasicBlock) {
            SoftmaxGradFront<T2, true>(softmaxGradTempRes, castedAttentionIn, castedDx, apiClcTensor,
                                       ordTilingData_->softmaxGradTilingData);
        } else {
            SoftmaxGradFront<T2, false>(softmaxGradTempRes, castedAttentionIn, castedDx, apiClcTensor,
                                        ordTilingData_->softmaxGradTilingData);
        }
    } else {
        if (isBasicBlock) {
            SoftmaxGradFront<T2, true>(softmaxGradOutUb, attentionInUb, dxUb, apiClcTensor,
                                       ordTilingData_->softmaxGradTilingData);
        } else {
            SoftmaxGradFront<T2, false>(softmaxGradOutUb, attentionInUb, dxUb, apiClcTensor,
                                        ordTilingData_->softmaxGradTilingData);
        }
    }

    AscendC::PipeBarrier<PIPE_V>();
    Add(softmaxGradOutUb, softmaxGradOutUb, softmaxGradTempRes, softmaxGradOutUb.GetSize());
    AscendC::PipeBarrier<PIPE_V>();

    apiClcQue.FreeTensor(apiClcTensor);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::HandleSoftMaxGradSplitD(
    const int64_t &batchSqLoopOffset, LocalTensor<T2> &softmaxGradOutUb)
{
    int64_t softmaxGradInputNum = 0;
    // 清空softmaxGradOutUb
    Duplicate<T2>(softmaxGradOutUb, 0, softmaxGradOutUb.GetSize());
    AscendC::PipeBarrier<PIPE_V>();
    for (int64_t i = 0; i < dRange; ++i) {
        LocalTensor<T1> attentionInUb = vecInQue1.AllocTensor<T1>(); // 必须要放在循环内，做MTE和v之间的pipe等待
        LocalTensor<T1> dxUb = vecInQue1.AllocTensor<T1>();
        auto dAlignOffset = i * splitedDAlign;
        auto remainDAlign = originalDAlign - dAlignOffset;
        int64_t thisDAlign = 0;
        int64_t thisD = 0;
        if (i == dRange - 1) {
            thisDAlign = remainDAlign;
            thisD = d - dAlignOffset;
        } else {
            /* 非尾块都是对齐的 */
            thisDAlign = splitedDAlign;
            thisD = splitedDAlign;
        }
        softmaxGradInputNum = softmaxGradInputShape[0] * thisDAlign;
        dxUb.SetSize(softmaxGradInputNum);
        attentionInUb.SetSize(softmaxGradInputNum);

        softmaxGradInputShape[1] = thisDAlign;
        dxUb.SetShapeInfo(ShapeInfo(2, softmaxGradInputShape, DataFormat::ND));
        attentionInUb.SetShapeInfo(ShapeInfo(2, softmaxGradInputShape, DataFormat::ND));

        CopyInSoftMaxGrad(dxUb, dxGm[batchSqLoopOffset], attentionInUb, forwardResGm[batchSqLoopOffset], thisDAlign,
                          thisD, dAlignOffset);
        ClcSoftMaxGradSplitD(softmaxGradOutUb, dxUb, attentionInUb, thisDAlign);
        vecInQue1.FreeTensor(attentionInUb);
        vecInQue1.FreeTensor(dxUb);
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ClcSub(
    LocalTensor<T2> &frontResInner, LocalTensor<T2> &dpResInner, LocalTensor<T2> &sftFrontResInner)
{
    // [m,n] - [m,16] -> [m,n] 按n轴的block数repeat，每个指令repeat算[m,16] - [m,16], subRange循环处理 超过mask情况
    for (int64_t batchIndex = 0; batchIndex < nIn * g; ++batchIndex) {
        for (int64_t subIdx = 0; subIdx < subRange; ++subIdx) {
            int64_t src0Offset = batchIndex * sQ * sKVAlignVec + subIdx * sKVAlignVec * 8;
            int64_t src1Offset = batchIndex * sQ * (32 / sizeof(T2)) + subIdx * subMask;
            if (subIdx == subRange - 1 && subMaskTail != 0) {
                Sub(frontResInner[src0Offset], dpResInner[src0Offset], sftFrontResInner[src1Offset], subMaskTail,
                    sKVAlignBlockNumVec, {(uint8_t)(sKVAlignBlockNumVec), (uint8_t)(sKVAlignBlockNumVec), 1, 1, 1, 0});
            } else {
                // dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride, src1_rep_stride
                Sub(frontResInner[src0Offset], dpResInner[src0Offset], sftFrontResInner[src1Offset], subMask,
                    sKVAlignBlockNumVec, {(uint8_t)(sKVAlignBlockNumVec), (uint8_t)(sKVAlignBlockNumVec), 1, 1, 1, 0});
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::SetReClcShape(
    LocalTensor<T2> &mulResInner, LocalTensor<float> &maxInner, LocalTensor<float> &sumInner,
    LocalTensor<T2> &dvDropResInner)
{
    mulResInner.SetShapeInfo(ShapeInfo(2, innerMatOutShape, DataFormat::ND));
    maxInner.SetShapeInfo(ShapeInfo(2, innerReduceShape, DataFormat::ND));
    sumInner.SetShapeInfo(ShapeInfo(2, innerReduceShape, DataFormat::ND));
    dvDropResInner.SetShapeInfo(ShapeInfo(2, innerMatOutShape, DataFormat::ND));
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::CopyInSoftMax(
    LocalTensor<float> &maxInner, const GlobalTensor<float> &softmaxMaxGmIn, LocalTensor<float> &sumInner,
    const GlobalTensor<float> &softmaxSumGmIn)
{
    // innerReduceNum是*8 *4（数据类型size），已经32bytes对齐，符合DataCopy的接口要求，不需要在做16对齐
    DataCopy(maxInner, softmaxMaxGmIn, innerReduceNum);
    DataCopy(sumInner, softmaxSumGmIn, innerReduceNum);
    vecInQue1.EnQue(sumInner);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline bool
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::CopyInAttenMask(
    const int64_t &attenMaskOffset)
{
    if (hasAttenMask != 1) {
        return false;
    }
    DataCopyParams copyParams;
    LocalTensor<uint8_t> attenMaskUb = vecInQue2.AllocTensor<uint8_t>();
    attenMaskUb.SetSize(nIn * g * sQ * sKVAlignByte);
    copyParams.blockCount = sQ;
    copyParams.blockLen = sKV;
    copyParams.srcStride = attenMaskDimS2 - sKV;
    copyParams.dstStride = 0;

    int64_t stride = attenMaskShapeType == 2 ? (sQ * sKV) : 0; // 2:BNSS
    for (int64_t copyIndex = 0; copyIndex < nIn * g; ++copyIndex) {
        DataCopyPad(attenMaskUb[copyIndex * sQ * sKVAlignByte], attenMaskU8Gm[attenMaskOffset + copyIndex * stride],
                    copyParams, {false, 0, 0, 0});
    }
    vecInQue2.EnQue(attenMaskUb);
    return true;
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ClcAttenMask(
    LocalTensor<T2> &mmResUb)
{
    LocalTensor<uint8_t> attenMaskUb = vecInQue2.DeQue<uint8_t>();
    LocalTensor<uint8_t> ubWorkspace = apiClcQue.AllocTensor<uint8_t>();

    T2 scalar;
    if constexpr (IsSameType<T2, float>::value) {
        uint32_t tmp = 0xFF7FFFFF;
        scalar = *((float *)&tmp);
    } else {
        uint16_t tmp = 0xFBFF;
        scalar = *((half *)&tmp);
    }
    SelectWithBytesMaskShapeInfo shapeInfo;
    shapeInfo.firstAxis = nIn * g * sQ;
    shapeInfo.srcLastAxis = sKVAlignVec;
    shapeInfo.maskLastAxis = sKVAlignByte;
    attenMaskUb.SetSize(shapeInfo.firstAxis * shapeInfo.maskLastAxis); // 需要设置，否则cpu跑的时候会报size不匹配的错误
    mmResUb.SetSize(shapeInfo.firstAxis * shapeInfo.srcLastAxis);
    SelectWithBytesMask(mmResUb, mmResUb, scalar, attenMaskUb, ubWorkspace, shapeInfo);

    apiClcQue.FreeTensor(ubWorkspace);
    vecInQue2.FreeTensor(attenMaskUb);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ClcSoftMax(
    LocalTensor<T2> &softmaxResInner, LocalTensor<T2> &reMatmulResInner, LocalTensor<float> &maxInner,
    LocalTensor<float> &sumInner)
{
    vecInQue1.DeQue<float>();
    LocalTensor<uint8_t> apiClcTensor = apiClcQue.AllocTensor<uint8_t>();
    apiClcTensor.SetSize(apiClcQueueSize);
    bool isBasicBlock = ((nIn * g * sQ) % 8 == 0) && (sKV % 64 == 0);
    if (isBasicBlock) {
        SimpleSoftMax<T2, true, true>(softmaxResInner, sumInner, maxInner, reMatmulResInner, apiClcTensor,
                                      ordTilingData_->softmaxTilingData);
    } else {
        SimpleSoftMax<T2, true, false>(softmaxResInner, sumInner, maxInner, reMatmulResInner, apiClcTensor,
                                       ordTilingData_->softmaxTilingData);
    }
    AscendC::PipeBarrier<PIPE_V>();
    apiClcQue.FreeTensor(apiClcTensor);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::FrontCompute(
    const int64_t &batchSqLoopOffset, const int64_t &batchSkvLoopOffset, const int64_t &dropMaskOffset,
    const int64_t &nCvIndex)
{
    LocalTensor<T2> softmaxGradOutUb = vecClc2.Get<T2>(nIn * g * sQ * 32 / sizeof(T2)); // FlashSoftmaxGrad的计算结果
    LocalTensor<T2> frontResUb = vecClc1.Get<T2>(innerMatResNum);                       // Sub的计算结果
    LocalTensor<T2> &dpResUb = frontResUb;                               // Dropout计算结果 共用内存
    LocalTensor<T2> &dpMatmmulResUb = frontResUb;                        // BMM1计算结果 共用内存
    LocalTensor<uint8_t> dpMaskInner = vecInQue2.AllocTensor<uint8_t>(); // drop_mask的输入

    dpMaskInner.SetSize(maskInputNum);
    SetFrontClcShape(softmaxGradOutUb, frontResUb, dpMaskInner);

    if (isDrop) {
        CopyInDropMask<true>(dpMaskInner, dropoutWorkspaceGm, dropMaskGm, this->dropMaskInfo);
        vecInQue2.EnQue(dpMaskInner);
    }

    if (vecCastSize == 0) { // 不切D
        HandleSoftMaxGrad(batchSqLoopOffset, softmaxGradOutUb);
    } else {
        HandleSoftMaxGradSplitD(batchSqLoopOffset, softmaxGradOutUb);
    }

    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);

    Mm12OutProcess(matmulResultBuffer1, dpMatmmulResUb, nCvIndex);


    if (isDrop) {
        AscendC::PipeBarrier<PIPE_V>();
        // for compute dropout mask
        dropMaskInfo.firstAxis = nIn * g * sQ;
        vecInQue2.DeQue<uint8_t>();
        LocalTensor<uint8_t> apiClcTensor = apiClcQue.AllocTensor<uint8_t>();
        apiClcTensor.SetSize(ordTilingData_->splitCoreParams.apiClcQueueSize);
        ComputeDropMask<float, true>(dpResUb, dpMatmmulResUb, dpMaskInner, apiClcTensor, this->dropMaskInfo);
        AscendC::PipeBarrier<PIPE_V>();
        apiClcQue.FreeTensor(apiClcTensor);
    }

    AscendC::PipeBarrier<PIPE_V>();
    ClcSub(frontResUb, dpMatmmulResUb, softmaxGradOutUb);
    event_t v2WaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    AscendC::SetFlag<HardEvent::V_MTE2>(v2WaitMte2);
    AscendC::WaitFlag<HardEvent::V_MTE2>(v2WaitMte2);
    vecInQue2.FreeTensor(dpMaskInner);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::CopyToWorkspace(
    const int64_t &batchSqLoopOffset, const int64_t &batchSkvLoopOffset, LocalTensor<T2> &mulResInner,
    LocalTensor<T2> &dvDropResInner, const int64_t &nCvIndex)
{
    int64_t offset = previousBatchCnt * sQ * sKV + nCvIndex * oriNIn * g * sQ * sKV;
    if constexpr (IS_BF16) {
        LocalTensor<T1> castedMulResPad = vecInQue1.AllocTensor<T1>();
        AscendC::PipeBarrier<PIPE_V>();
        castedMulResPad.SetSize(innerMatResNum);
        Cast(castedMulResPad, mulResInner, RoundMode::CAST_ROUND, innerMatResNum);
        event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        AscendC::SetFlag<HardEvent::V_MTE3>(mte3WaitV);
        AscendC::WaitFlag<HardEvent::V_MTE3>(mte3WaitV);

        DataCopyPad(mulWorkSpaceGm[pingpongIdx * pingPongMulOffset + offset], castedMulResPad,
                    {static_cast<uint16_t>(nIn * g * sQ), static_cast<uint16_t>(sKV * inputDTypeSize), 0, 0});

        LocalTensor<T1> castedDvDropResPad = vecInQue1.AllocTensor<T1>();
        castedDvDropResPad.SetSize(innerMatResNum);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(castedDvDropResPad, dvDropResInner, RoundMode::CAST_ROUND, innerMatResNum);
        AscendC::SetFlag<HardEvent::V_MTE3>(mte3WaitV);
        AscendC::WaitFlag<HardEvent::V_MTE3>(mte3WaitV);

        DataCopyPad(dropWorkSpaceGm[pingpongIdx * pingPongDropOffset + offset], castedDvDropResPad,
                    {static_cast<uint16_t>(nIn * g * sQ), static_cast<uint16_t>(sKV * inputDTypeSize), 0, 0});
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

        vecInQue1.FreeTensor(castedMulResPad);
        vecInQue1.FreeTensor(castedDvDropResPad);
    } else {
        DataCopyPad(mulWorkSpaceGm[pingpongIdx * pingPongMulOffset + offset], mulResInner,
                    {static_cast<uint16_t>(nIn * g * sQ), static_cast<uint16_t>(sKV * inputDTypeSize), 0, 0});
        AscendC::PipeBarrier<PIPE_V>();
        DataCopyPad(dropWorkSpaceGm[pingpongIdx * pingPongDropOffset + offset], dvDropResInner,
                    {static_cast<uint16_t>(nIn * g * sQ), static_cast<uint16_t>(sKV * inputDTypeSize), 0, 0});
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::CalcAttenMaskOffset(
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

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::CalcCausalAttenMaskOffset(
    int64_t &attenMaskOffset, const int64_t delta)
{
    CalcAttenMaskOffset(attenMaskOffset, delta);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT, S1TemplateType s1TemplateType,
          S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType,
                                  s2TemplateType, dTemplateType>::Mm12OutProcess(GlobalTensor<T2> &matmulOutGm,
                                                                                 LocalTensor<T2> &outLocalTensor,
                                                                                 const int64_t nCvIndex)
{
    DataCopyParams intriParams;
    event_t mte2WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

    if (nCvIndex == 0) {
        mm1.WaitIterateBatch();
        mm1.End();
    }
    if constexpr (MMPre_OUT_FORMAT == CubeFormat::ND) {
        intriParams.blockCount = static_cast<uint16_t>(nIn * g * sQ);
        intriParams.blockLen = static_cast<uint16_t>(sKV * sizeof(T2));
        intriParams.srcStride = 0;
        intriParams.dstStride = sKVStride;
        DataCopyPad(outLocalTensor, matmulOutGm[nCvIndex * oriNIn * g * sQ * sKV], intriParams,
                    {true, 0, static_cast<uint8_t>(sKVAlignVec - sKV - sKVStride * FP32_PER_BLOCK_NUM), 0});

        AscendC::SetFlag<HardEvent::MTE2_V>(mte2WaitV);
        AscendC::WaitFlag<HardEvent::MTE2_V>(mte2WaitV);
    } else {
        // NZ copy in
        auto tmpTensor = vecInQue1.AllocTensor<T2>();
        DataCopy(tmpTensor, matmulOutGm[nCvIndex * oriNIn * g * sQ * sKVAlign], oriNIn * g * sKVAlign * sQ);
        AscendC::SetFlag<HardEvent::MTE2_V>(mte2WaitV);
        AscendC::WaitFlag<HardEvent::MTE2_V>(mte2WaitV);

        // NZ to ND
        intriParams.blockCount = sQ;
        intriParams.blockLen = C0_SIZE * sizeof(T2) / BLOCK;
        intriParams.srcStride = 0;
        intriParams.dstStride = (sKVAlign - C0_SIZE) * sizeof(T2) / BLOCK;
        uint32_t loop = sKVAlign / C0_SIZE;
        for (uint64_t i = 0; i < nIn * g; ++i) {
            uint64_t batchOffset = i * sQ * sKVAlign;
            uint64_t inOffset = 0;
            uint64_t outOffset = 0;
            for (uint64_t j = 0; j < loop; ++j) {
                DataCopy(outLocalTensor[batchOffset + outOffset], tmpTensor[batchOffset + inOffset], intriParams);
                inOffset += C0_SIZE * sQ;
                outOffset += C0_SIZE;
            }
        }

        vecInQue1.FreeTensor(tmpTensor);
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ReCompute(
    const int64_t &batchSqLoopOffset, const int64_t &batchSkvLoopOffset, const int64_t &attenMaskOffset,
    const int64_t &dropMaskOffset, const int64_t &batchSoftmaxInputOffset, const int64_t &nCvIndex)
{
    LocalTensor<T2> subResInner = vecClc1.Get<T2>(innerMatResNum);
    LocalTensor<T2> &mulResInner = subResInner;
    LocalTensor<T2> dvDropResInner = vecClc2.Get<T2>(innerMatResNum);

    LocalTensor<T2> &reMatmulResInner = dvDropResInner;
    LocalTensor<T2> &softmaxResInner = dvDropResInner;
    LocalTensor<T2> &attenMaskResInner = dvDropResInner;

    bool clcAttenMask = false;
    if (pseSq != 0) {
        LocalTensor<T1> pseUb = vecInQue2.AllocTensor<T1>();
        pseUb.SetSize(nIn * g * pseSq * sKVAlign);
        auto noCastedPseUb = vecInQue1.AllocTensor<T2>();
        noCastedPseUb.SetSize(0);
        PseCopyIn<T1, T2, LayOutTypeEnum::LAYOUT_BNSD, true>(noCastedPseUb, pseUb, this->pseGm, this->pseInfo);
        vecInQue1.FreeTensor(noCastedPseUb);
        vecInQue2.EnQue(pseUb);
    }

    if (pseSq == 0) {
        clcAttenMask = CopyInAttenMask(attenMaskOffset);
    }

    Mm12OutProcess(matmulResultBuffer2,reMatmulResInner, nCvIndex);


    if (pseSq != 0) {
        LocalTensor<T1> pseUb = vecInQue2.DeQue<T1>();
        uint32_t eleNum = reMatmulResInner.GetSize();
        auto castedPseUb = vecInQue1.AllocTensor<T2>();
        castedPseUb.SetSize(eleNum);
        Cast(castedPseUb, pseUb, RoundMode::CAST_NONE, eleNum);
        AscendC::PipeBarrier<PIPE_V>();
        if (pseShapeType == 1) {
            pseInfo.vec1S1RealSize = sQ;
            for (int64_t batchIndex = 0; batchIndex < nIn * g; ++batchIndex) {
                LocalTensor<T2> pseRes = reMatmulResInner[batchIndex * sQ * sKVAlign];
                LocalTensor<T2> pse = castedPseUb[batchIndex * sKVAlign];
                PseCompute<T2, true>(pseRes, pse, this->pseInfo);
            }
        } else {
            PseCompute<T2, true>(reMatmulResInner, castedPseUb, this->pseInfo);
        }
        vecInQue1.FreeTensor(castedPseUb);
        vecInQue2.FreeTensor(pseUb);
        AscendC::PipeBarrier<PIPE_V>();
        clcAttenMask = CopyInAttenMask(attenMaskOffset);
    }
    LocalTensor<float> maxInner = vecInQue1.AllocTensor<float>();
    LocalTensor<float> sumInner = vecInQue1.AllocTensor<float>();
    maxInner.SetSize(innerReduceNum);
    sumInner.SetSize(innerReduceNum);
    SetReClcShape(mulResInner, maxInner, sumInner, dvDropResInner);
    CopyInSoftMax(maxInner, softmaxMaxGm[batchSoftmaxInputOffset], sumInner, softmaxSumGm[batchSoftmaxInputOffset]);
    event_t sWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    AscendC::SetFlag<HardEvent::MTE2_S>(sWaitMte2);
    AscendC::WaitFlag<HardEvent::MTE2_S>(sWaitMte2);
    Muls(reMatmulResInner, reMatmulResInner, (T2)scaleValue, innerMatResNumVec);
    AscendC::PipeBarrier<PIPE_V>();
    if (clcAttenMask) {
        ClcAttenMask(reMatmulResInner);
        AscendC::PipeBarrier<PIPE_V>();
    }
    uint32_t tempInnerMatOutShape[2];
    tempInnerMatOutShape[0] = nIn * g * sQ;
    tempInnerMatOutShape[1] = sKVAlignVec;
    dvDropResInner.SetShapeInfo(ShapeInfo(2, tempInnerMatOutShape, DataFormat::ND));

    ClcSoftMax(softmaxResInner, attenMaskResInner, maxInner, sumInner);

    mulResInner.SetShapeInfo(ShapeInfo(2, tempInnerMatOutShape, DataFormat::ND));
    AscendC::PipeBarrier<PIPE_V>();

    Mul(mulResInner, softmaxResInner, subResInner, innerMatResNumVec);
    AscendC::PipeBarrier<PIPE_V>();

    if (isDrop) {
        LocalTensor<uint8_t> dpMask = vecInQue2.AllocTensor<uint8_t>();
        CopyInDropMask<true>(dpMask, dropoutWorkspaceGm, dropMaskGm, this->dropMaskInfo);
        vecInQue2.EnQue(dpMask);
        vecInQue2.DeQue<uint8_t>();

        // for compute dropout mask
        dropMaskInfo.firstAxis = nIn * g * sQ;
        LocalTensor<uint8_t> apiClcTensor = apiClcQue.AllocTensor<uint8_t>();
        apiClcTensor.SetSize(ordTilingData_->splitCoreParams.apiClcQueueSize);
        ComputeDropMask<float, true>(softmaxResInner, softmaxResInner, dpMask, apiClcTensor, this->dropMaskInfo);

        AscendC::PipeBarrier<PIPE_V>();
        apiClcQue.FreeTensor(apiClcTensor);
        vecInQue2.FreeTensor(dpMask);
    }
    vecInQue1.FreeTensor(maxInner);
    vecInQue1.FreeTensor(sumInner);
    CopyToWorkspace(batchSqLoopOffset, batchSkvLoopOffset, mulResInner, dvDropResInner, nCvIndex);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ClcMm1(
    const int64_t &batchSqLoopOffset, const int64_t &batchSkvLoopOffset)
{
    mm1.SetTensorA(dxGm[batchSqLoopOffset]);
    mm1.SetTensorB(valueGm[batchSkvLoopOffset], true);
    mm1.template IterateBatch<false, true>(matmulResultBuffer1, nCvInner * g, nCvInner, false);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ClcMm2(
    const int64_t &batchSqLoopOffset, const int64_t &batchSkvLoopOffset)
{
    mm1.SetTensorA(queryGm[batchSqLoopOffset]);
    mm1.SetTensorB(keyGm[batchSkvLoopOffset], true);
    mm1.template IterateBatch<false, true>(matmulResultBuffer2, nCvInner * g, nCvInner, false);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ClcMm31(
    const int64_t &cvMulOffset, const int64_t &batchSqLoopOffset,
    const int64_t &batchSkvLoopOffset, const bool &isSync)
{
    mm31.SetTail(sQ, d, -1);
    mm31.SetTensorA(mulWorkSpaceGm[lastPingpongIdx * pingPongMulOffset + cvMulOffset]);
    mm31.SetTensorB(keyGm[batchSkvLoopOffset]);
    if (isSync) {
      mm31.template IterateBatch<true>(dqWorkspaceGm[batchSqLoopOffset], lastNCvInner * g, lastNCvInner, false);
    } else {
      mm31.template IterateBatch<false>(dqWorkspaceGm[batchSqLoopOffset], lastNCvInner * g, lastNCvInner, false);
    }
    mm31.End();
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ClcMm32(
    const int64_t &cvMulOffset, const int64_t &batchSqLoopOffset,
    const int64_t &batchSkvLoopOffset, const bool &isSync)
{
    mm32.SetTail(sKV, -1, sQ);
    mm32.SetTensorA(mulWorkSpaceGm[lastPingpongIdx * pingPongMulOffset + cvMulOffset], true);
    mm32.SetTensorB(queryGm[batchSqLoopOffset]);
    if (isSync) {
      mm32.template IterateBatch<true>(dkWorkspaceGm[batchSkvLoopOffset], lastNCvInner * g, lastNCvInner * g, false);
    } else {
      mm32.template IterateBatch<false>(dkWorkspaceGm[batchSkvLoopOffset], lastNCvInner * g, lastNCvInner * g, false);
    }
    mm32.End();
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::ClcMm4(
    const int64_t &cvMulOffset, const int64_t &batchSqLoopOffset,
    const int64_t &batchSkvLoopOffset, const bool &isSync)
{
    mm4.SetTensorA(dropWorkSpaceGm[lastPingpongIdx * pingPongDropOffset + cvMulOffset], true);
    mm4.SetTensorB(dxGm[batchSqLoopOffset]);
    if (isSync) {
      mm4.template IterateBatch<true>(dvGm[batchSkvLoopOffset], lastNCvInner * g, lastNCvInner * g, false);
    } else {
      mm4.template IterateBatch<false>(dvGm[batchSkvLoopOffset], lastNCvInner * g, lastNCvInner * g, false);
    }
    mm4.End();
}

// T1 INPUT_T, T2 CALC_T
template <typename T1, typename T2, const MatmulConfig &MM_CFG, const bool IS_BF16, LayoutMode layout,
          const CubeFormat MMPre_OUT_FORMAT, const CubeFormat MMNext_OUT_FORMAT,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ inline void
    FlashAttentionScoreGradUngs1s2Bbn<T1, T2, MM_CFG, IS_BF16, layout, MMPre_OUT_FORMAT, MMNext_OUT_FORMAT, s1TemplateType, s2TemplateType, dTemplateType>::Process()
{
    if (g_coreType == AIV && mBlockIdx >= usedCoreNum) {
        SyncAll();
        return;
    }

    int64_t lastCvMulOffset = 0;
    int64_t lastBatchSqLoopOffset = 0;
    int64_t lastBatchSkvLoopOffset = 0;
    int64_t lastMmNzDqOffset = 0;
    int64_t lastMmNzDkvOffset = 0;

    int64_t batchOffset = mBlockIdx * singleCoreBatchRange;
    int64_t currentBatchRange =
        singleCoreBatchRange < (totalBatch - batchOffset) ? singleCoreBatchRange : (totalBatch - batchOffset);

    for (int64_t batchIdx = 0; batchIdx < currentBatchRange; ++batchIdx) {
        pingpongIdx = 1 - pingpongIdx;
        int64_t totalBatch = batchOffset + batchIdx;
        int64_t bIdx = totalBatch / nOut;
        int64_t nIdx = totalBatch % nOut;
        int64_t batchSqLoopOffset = 0;
        int64_t batchSkvLoopOffset = 0;
        int64_t mmNzDqOffset = 0;
        int64_t mmNzDkvOffset = 0;

        previousBatchCnt = (bIdx * n + nIdx * nCvInner) * g;
        int64_t dropMaskOffset = previousBatchCnt * sQ * sKV;
        int64_t attenMaskOffset = 0;
        if (hasAttenMask == 1) {
            int64_t compressMode = ordTilingData_->opInfo.attenMaskCompressMode;
            // compressMode == 0: no atten_mask compress
            if (compressMode == 1) {
                CalcCausalAttenMaskOffset(attenMaskOffset, 0);
            } else if (compressMode == 2) {
                CalcCausalAttenMaskOffset(attenMaskOffset, sKV - sQ);
            } else if (attenMaskShapeType == 0) { // SS
                attenMaskOffset = 0;
            } else if (attenMaskShapeType == 1) { // B1SS
                attenMaskOffset = bIdx * sQ * sKV;
            } else { // BNSS
                attenMaskOffset = previousBatchCnt * sQ * sKV;
            }
        }
        int64_t batchSoftmaxInputOffset = previousBatchCnt * sQ * 8;
        int64_t cvSqOffest = g * d;
        int64_t cvSkvOffest = d;
        if (inputLayout == 0 || inputLayout == 3) { // BSH or BSND
            batchSqLoopOffset = bIdx * sQ * n * g * d + nIdx * nCvInner * cvSqOffest;
            batchSkvLoopOffset = bIdx * sKV * n * d + nIdx * nCvInner * cvSkvOffest;
        } else if (inputLayout == 1) { // SBH即SBND
            batchSqLoopOffset = bIdx * n * g * d + nIdx * nCvInner * cvSqOffest;
            batchSkvLoopOffset = bIdx * n * d + nIdx * nCvInner * cvSkvOffest;
        } else if (inputLayout == 2) { // BNSD
            cvSqOffest = g * sQ * d;
            cvSkvOffest = sKV * d;
            batchSqLoopOffset = bIdx * n * g * sQ * d + nIdx * nCvInner * cvSqOffest;
            batchSkvLoopOffset = bIdx * n * sKV * d + nIdx * nCvInner * cvSkvOffest;
        } else {
            return;
        }

        if constexpr (MMNext_OUT_FORMAT == CubeFormat::NZ) {
            mmNzDqOffset = bIdx * n * g * sQ * originalDAlign + nIdx * nCvInner * g * sQ * originalDAlign;
            mmNzDkvOffset = bIdx * n * sKV * originalDAlign + nIdx * nCvInner * sKV * originalDAlign;
        }

        int64_t cvMulOffset = previousBatchCnt * sQ * sKV;
        // 原逻辑是计算这里
        int64_t nCvInnerOffsetDrop = nIdx * nCvInner;

        nIn = ordTilingData_->singleCoreParams.nIn;
        nCvInner = ordTilingData_->singleCoreParams.nCvInner;

        nIn = nIn < (n - nIdx * nCvInner) ? nIn : (n - nIdx * nCvInner); // nCvInnerTail < nIn
        uint32_t nCvInnerOffset = nIdx * nCvInner;
        nCvInner = nCvInner < (n - nIdx * nCvInner) ? nCvInner : (n - nIdx * nCvInner); // nCvInnerTail
        int64_t nCvLoop = (nCvInner + nIn - 1) / nIn;

        auto nInGSq = nIn * g * sQ;
        innerMatResNum = nInGSq * sKVAlign;
        innerMatResNumVec = nInGSq * sKVAlignVec;
        maskInputNum = nInGSq * sKVAlignByte;
        innerReduceNum = nInGSq * 8;

        softmaxGradInputShape[0] = nIn * g * sQ;

        innerMatOutShape[0] = nInGSq;
        innerMatOutShape[1] = sKVAlignVec;
        innerReduceShape[0] = nInGSq;
        innerReduceShape[1] = 8;

        // 发射本轮bmm12
        ClcMm1(batchSqLoopOffset, batchSkvLoopOffset);
        ClcMm2(batchSqLoopOffset, batchSkvLoopOffset);

        if (batchIdx > 0) {
          // 清workspace，否则pta连跑失败
            if (inputLayout == 1) {
                // SBH
                for (int i = 0; i < sKV; i++) {
                    int64_t offset = lastBatchSkvLoopOffset + i * b * n * 1 * d;
                    int64_t num = lastNCvInner * 1 * d;
                    if constexpr (MMNext_OUT_FORMAT == CubeFormat::NZ) {
                        int64_t nums = lastNCvInner * 1 * sKV * originalDAlign;
                        InitOutput<T2>(dkWorkspaceGm[lastMmNzDkvOffset], nums, 0);
                    } else {
                        InitOutput<T2>(dkWorkspaceGm[offset], num, 0);
                    }
                    InitOutput<T1>(dvGm[offset], num, 0);
                }
            } else if (inputLayout == 2) {
                // BNSD
                int64_t num = lastNCvInner * 1 * sKV * d;
                if constexpr (MMNext_OUT_FORMAT == CubeFormat::NZ) {
                    int64_t nums = lastNCvInner * 1 * sKV * originalDAlign;
                    InitOutput<T2>(dkWorkspaceGm[lastMmNzDkvOffset], nums, 0);
                } else {
                    InitOutput<T2>(dkWorkspaceGm[lastBatchSkvLoopOffset], num, 0);
                }
                InitOutput<T1>(dvGm[lastBatchSkvLoopOffset], num, 0);
            } else {
                // BSH BSND
                for (int i = 0; i < sKV; i++) {
                    int64_t offset = lastBatchSkvLoopOffset + i * n * 1 * d;
                    int64_t num = lastNCvInner * 1 * d;
                    if constexpr (MMNext_OUT_FORMAT == CubeFormat::NZ) {
                        int64_t nums = lastNCvInner * 1 * sKV * originalDAlign;
                        InitOutput<T2>(dkWorkspaceGm[lastMmNzDkvOffset], nums, 0);
                    } else {
                        InitOutput<T2>(dkWorkspaceGm[offset], num, 0);
                    }
                    InitOutput<T1>(dvGm[offset], num, 0);
                }
            }

            // 异步
            if constexpr (MMNext_OUT_FORMAT == CubeFormat::NZ) {
                ClcMm31(lastCvMulOffset, lastMmNzDqOffset, lastBatchSkvLoopOffset, false);
                ClcMm32(lastCvMulOffset, lastBatchSqLoopOffset, lastMmNzDkvOffset, false);
            } else {
                ClcMm31(lastCvMulOffset, lastBatchSqLoopOffset, lastBatchSkvLoopOffset, false);
                ClcMm32(lastCvMulOffset, lastBatchSqLoopOffset, lastBatchSkvLoopOffset, false);
            }
            ClcMm4(lastCvMulOffset, lastBatchSqLoopOffset, lastBatchSkvLoopOffset, false);
        }
        // 发射本轮 vec
        for (int64_t nCvIndex = 0; nCvIndex < nCvLoop; ++nCvIndex) {
            int64_t nCvSqOffset = nIn * cvSqOffest;
            int64_t nCvSkvOffset = nIn * cvSkvOffest;
            int64_t nCvSoftmaxOffset = nIn * g * sQ * 8;
            int64_t nCvDropOffset = nIn * g * sQ * sKV;

            if (isDrop) {
                // for compute dropout mask offset
                dropMaskInfo.bSSOffset = bIdx * sQ * sKV;
                dropMaskInfo.n2OutIdx = (nCvInnerOffsetDrop + nCvIndex * nIn);
            }

            int64_t nCvAttenMaskOffset = attenMaskShapeType == 2 ? nCvDropOffset : 0; // 2:BNSS
            if (pseSq != 0) {
                pseInfo.boIdx = bIdx;
                pseInfo.n2oIdx = nCvInnerOffset + nCvIndex * nIn;
                pseInfo.bSSOffset = pseInfo.boIdx * sQ * sKV;
                pseInfo.s2SizeAcc = pseInfo.boIdx * sKV;
            }
            nIn = nIn < (nCvInner - nIn * nCvIndex) ? nIn : (nCvInner - nIn * nCvIndex); // nCvInnerTail > nIn

            if (nCvIndex == nCvLoop - 1 && nIn < ordTilingData_->singleCoreParams.nIn) {
                nInGSq = nIn * g * sQ;
                innerMatResNum = nInGSq * sKVAlign;
                innerMatResNumVec = nInGSq * sKVAlignVec;
                maskInputNum = nInGSq * sKVAlignByte;
                innerReduceNum = nInGSq * 8;

                softmaxGradInputShape[0] = nIn * g * sQ;

                innerMatOutShape[0] = nInGSq;
                innerMatOutShape[1] = sKVAlignVec;
                innerReduceShape[0] = nInGSq;
                innerReduceShape[1] = 8;
            }

            if (pseSq != 0) {
                pseInfo.blockCount = nIn * g;
                pseInfo.vec1S1RealSize = nIn * g * sQ;
            }

            int64_t cvBatchSqOffset = batchSqLoopOffset + nCvIndex * nCvSqOffset;
            int64_t cvBatchSkvOffset = batchSkvLoopOffset + nCvIndex * nCvSkvOffset;
            int64_t cvDropMaskOffset = dropMaskOffset + nCvIndex * nCvDropOffset;
            int64_t cvAttenMaskOffset = attenMaskOffset + nCvIndex * nCvAttenMaskOffset;
            if (sKV % BN_DROPOUT4BIT_LEN == 0) {
                cvDropMaskOffset = cvDropMaskOffset / 8;
            }

            if (isDrop) {
                // for copy in dropout mask
                dropMaskInfo.s1CopySize = nIn * g * sQ;
            }

            // FrontCompute 中 WaitIterateBatch value dy
            FrontCompute(cvBatchSqOffset, cvBatchSkvOffset, cvDropMaskOffset, nCvIndex);
            // ReCompute 中 WaitIterateBatch key query
            ReCompute(cvBatchSqOffset, cvBatchSkvOffset, cvAttenMaskOffset, cvDropMaskOffset,
                      batchSoftmaxInputOffset + nCvIndex * nCvSoftmaxOffset, nCvIndex);
        }

        // 最后一个循环 直接发射本轮bmm345
        if (batchIdx + 1 >= currentBatchRange) {
            // 清workspace，否则pta连跑失败
            if (inputLayout == 1) {
                // SBH
                for (int i = 0; i < sKV; i++) {
                    int64_t offset = batchSkvLoopOffset + i * b * n * 1 * d;
                    int64_t num = nCvInner * 1 * d;
                    if constexpr (MMNext_OUT_FORMAT == CubeFormat::NZ) {
                        int64_t nums = nCvInner * 1 * sKV * originalDAlign;
                        InitOutput<T2>(dkWorkspaceGm[mmNzDkvOffset], nums, 0);
                    } else {
                        InitOutput<T2>(dkWorkspaceGm[offset], num, 0);
                    }
                    InitOutput<T1>(dvGm[offset], num, 0);
                }
            } else if (inputLayout == 2) {
                // BNSD
                int64_t num = nCvInner * 1 * sKV * d;
                if constexpr (MMNext_OUT_FORMAT == CubeFormat::NZ) {
                    int64_t nums = nCvInner * 1 * sKV * originalDAlign;
                    InitOutput<T2>(dkWorkspaceGm[mmNzDkvOffset], nums, 0);
                } else {
                    InitOutput<T2>(dkWorkspaceGm[batchSkvLoopOffset], num, 0);
                }
                InitOutput<T1>(dvGm[batchSkvLoopOffset], num, 0);
            } else {
                // BSH BSND
                for (int i = 0; i < sKV; i++) {
                    int64_t offset = batchSkvLoopOffset + i * n * 1 * d;
                    int64_t num = nCvInner * 1 * d;
                    if constexpr (MMNext_OUT_FORMAT == CubeFormat::NZ) {
                        int64_t nums = nCvInner * 1 * sKV * originalDAlign;
                        InitOutput<T2>(dkWorkspaceGm[mmNzDkvOffset], nums, 0);
                    } else {
                        InitOutput<T2>(dkWorkspaceGm[offset], num, 0);
                    }
                    InitOutput<T1>(dvGm[offset], num, 0);
                }
            }

            lastNCvInner = nCvInner;
            lastPingpongIdx = pingpongIdx;
            // 同步
            if constexpr (MMNext_OUT_FORMAT == CubeFormat::NZ) {
                ClcMm31(cvMulOffset, mmNzDqOffset, batchSkvLoopOffset, true);
                ClcMm32(cvMulOffset, batchSqLoopOffset, mmNzDkvOffset, true);
            } else {
                ClcMm31(cvMulOffset, batchSqLoopOffset, batchSkvLoopOffset, true);
                ClcMm32(cvMulOffset, batchSqLoopOffset, batchSkvLoopOffset, true);
            }
            ClcMm4(cvMulOffset, batchSqLoopOffset, batchSkvLoopOffset, true);
        }

        // 备份本轮bmm345地址
        lastNCvInner = nCvInner;
        lastCvMulOffset = cvMulOffset;
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

#endif // FLASH_ATTENTION_SCORE_GRAD_NGS1S2_BN_H_
