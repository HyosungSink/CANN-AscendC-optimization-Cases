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
 * \file flash_attention_score_bn2gs1s2_b.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_BN2GS1S2_B_H
#define FLASH_ATTENTION_SCORE_BN2GS1S2_B_H
#define FA_BN2GS1S2B_FUNCTION_TEMPLATE                                                                                 \
    template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop,              \
              typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout, STemplateType s1TemplateType,        \
              STemplateType s2TemplateType, DTemplateType dTemplateType>
#define FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE                                                                          \
    implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, layout, s1TemplateType, s2TemplateType, \
    dTemplateType

#include "../../common/include/kernel/util.h"
#include "../../common/include/kernel/dropmask.h"
#include "flash_attention_score_common.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "../../common/include/kernel/pse.h"

struct SplitBExtraInfo {
    int64_t boIdx;
    int64_t biN2GoIdx;
    int64_t s1oIdx;
    int64_t taskId;
    int64_t vecS1BaseSize;    // S1基本块，保证vecS1BaseSize * S2 <= 32K
    int64_t vecS1TailSize;    // S1尾块
    int64_t s2AlignSize;      // S2 基本块大小
    int64_t s2AlignBlockSize; // S2 基本块32对齐大小
    int64_t s1Vec2BaseSize;
    int64_t s1Vec2BaseTailSize;
    int64_t s1Vec2OuterSize;
    int64_t qCoreOffset;
    int64_t softmaxCopyOutLimit;
    int64_t softmaxCopyOutSize;
    int64_t softmaxOutOffset = 0;
};

struct SplitBNz2NdInfo {
    int64_t ndFirstAxisRealSize;
    int64_t ndFirstAxisLoopSize;
    int64_t ndLastAxis;
    int64_t bmmResOffset;
};

template <STemplateType s1TemplateType, STemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ const constexpr MatmulConfig GetBn2gs1s2BMm1Config()
{
    constexpr uint32_t baseM = static_cast<uint32_t>(s1TemplateType);
    constexpr uint32_t baseN = static_cast<uint32_t>(s2TemplateType);
    constexpr uint32_t baseK = static_cast<uint32_t>(dTemplateType);
    return MatmulConfig{true,  false, false, baseM, baseN,
                        baseK, false, false, false, false,
                        0,     0,     0,     0,     0,
                        0,     0,     0,     true,  false,
                        false, false, false, true,  BatchMode::BATCH_LESS_THAN_L1,
                        true,  true,  false, false, false,
                        false};
}

template <STemplateType s1TemplateType, STemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ const constexpr MatmulConfig GetBn2gs1s2BMm2Config()
{
    constexpr uint32_t baseM = static_cast<uint32_t>(s1TemplateType);
    constexpr uint32_t baseN = static_cast<uint32_t>(dTemplateType);
    constexpr uint32_t baseK = static_cast<uint32_t>(s2TemplateType);
    return MatmulConfig{true,  false, false, baseM, baseN,
                        baseK, false, false, false, false,
                        0,     0,     0,     0,     0,
                        0,     0,     0,     true,  false,
                        false, false, false, true,  BatchMode::BATCH_LESS_THAN_L1,
                        true,  true,  false, false, false,
                        false};
}

// INPUT_T - means data type for input
// T       - means data type when calc
template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T = INPUT_T, bool isBasicBlock = false, LayoutMode layout = LayoutMode::BNGS1S2,
          STemplateType s1TemplateType = STemplateType::S_TEMPLATE_UNKNOW,
          STemplateType s2TemplateType = STemplateType::S_TEMPLATE_UNKNOW,
          DTemplateType dTemplateType = DTemplateType::D_TEMPLATE_UNKNOW>
class FlashAttentionScoreBn2gs1s2B {
public:
    __aicore__ inline FlashAttentionScoreBn2gs1s2B(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                                __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *prefix,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *sink, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // define batchmatmul
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, layout>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true, layout>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float, false, LayoutMode::BNGS1S2>;
    using c1Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false, LayoutMode::BNGS1S2>;
    constexpr static MatmulConfig mm1Config = GetBn2gs1s2BMm1Config<s1TemplateType, s2TemplateType, dTemplateType>();
    constexpr static MatmulApiStaticTiling stcMm1Tiling =
        matmul::GetMatmulApiTiling<a1Type, b1Type, c1Type, bias1Type>(mm1Config);
    using modeTypeMm1 = typename AscendC::Conditional<
        (s1TemplateType == STemplateType::S_TEMPLATE_UNKNOW),
        matmul::Matmul<a1Type, b1Type, c1Type, bias1Type>,
        matmul::Matmul<a1Type, b1Type, c1Type, bias1Type, stcMm1Tiling>>::type;
    modeTypeMm1 bmm1;

    using a2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::BNGS1S2>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, layout>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, float, false, layout>;
    using c2Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false, LayoutMode::BNGS1S2>;
    constexpr static MatmulConfig mm2Config = GetBn2gs1s2BMm2Config<s1TemplateType, s2TemplateType, dTemplateType>();
    constexpr static MatmulApiStaticTiling stcMm2Tiling =
        matmul::GetMatmulApiTiling<a2Type, b2Type, c2Type, bias2Type>(mm2Config);
    using modeTypeMm2 = typename AscendC::Conditional<
        (s1TemplateType == STemplateType::S_TEMPLATE_UNKNOW),
        matmul::Matmul<a2Type, b2Type, c2Type, bias2Type>,
        matmul::Matmul<a2Type, b2Type, c2Type, bias2Type, stcMm2Tiling>>::type;
    modeTypeMm2 bmm2;

    using c2NzType = MatmulType<TPosition::GM, CubeFormat::NZ, T, false, LayoutMode::BNGS1S2>;
    using modeTypeMm2Nz = typename AscendC::Conditional<
        (s1TemplateType == STemplateType::S_TEMPLATE_UNKNOW),
        matmul::Matmul<a2Type, b2Type, c2NzType, bias2Type>,
        matmul::Matmul<a2Type, b2Type, c2NzType, bias2Type, stcMm2Tiling>>::type;
    modeTypeMm2Nz bmm2Nz;

protected:
    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                     __gm__ uint8_t *pse, __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask,
                                     __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, __gm__ uint8_t *sink, __gm__ uint8_t *softmaxMax,
                                     __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut,
                                     __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                     const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void WaitBmm1Result();
    template <typename T2, const auto &TILING>
    __aicore__ inline void WaitBmm2Result(matmul::Matmul<a2Type, b2Type, T2, bias2Type, TILING> &bmm2);
    __aicore__ inline void IterateBmm1(SplitBExtraInfo &extraInfo);
    template <typename T2, const auto &TILING>
    __aicore__ inline void IterateBmm2(SplitBExtraInfo &extraInfo,
                                       matmul::Matmul<a2Type, b2Type, T2, bias2Type, TILING> &bmm2);
    __aicore__ inline void NzToNd(SplitBNz2NdInfo &nz2NdInfo, const GlobalTensor<T> &bmmResGm,
                                  LocalTensor<T> &tempUb, LocalTensor<T> &bmmResUb);
    __aicore__ inline void AtenmaskBoolCopyIn(LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor,
                                              int64_t offset, SplitBExtraInfo &extraInfo, int32_t s2Size,
                                              int64_t totalS2Size);
    __aicore__ inline int64_t ComputeQCoreOffset(SplitBExtraInfo &extraInfo);
    __aicore__ inline int64_t ComputeKVCoreOffset(SplitBExtraInfo &extraInfo);
    __aicore__ inline void SetExtraInfo(SplitBExtraInfo &extraInfo, int64_t taskId, int64_t multiCoreInnerIdx);
    __aicore__ inline void SetTiling(const FlashAttentionScoreGeneralTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void CalBatchSize();
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void RefreshConstexpr();
    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);
    __aicore__ inline void ProcessVec1(SplitBExtraInfo &extraInfo);
    __aicore__ inline void ProcessVec2(SplitBExtraInfo &extraInfo);
    __aicore__ inline void CopyInAttenMask(SplitBExtraInfo &extraInfo, int64_t maskOffset);
    __aicore__ inline int64_t ComputeAttenMaskOffset(SplitBExtraInfo &extraInfo);
    __aicore__ inline int64_t ComputeOffsetForNoCompress(SplitBExtraInfo &extraInfo);
    __aicore__ inline void GetBmm1Result(SplitBExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb, int64_t loopIdx);
    __aicore__ inline void ComputeAttenMask(SplitBExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb,
                                            const uint8_t maskType);
    __aicore__ inline void SoftMaxCompute(SplitBExtraInfo &extraInfo, LocalTensor<T> &srcTensor, int64_t loopIdx);
    __aicore__ inline void Bmm2ResultDiv(SplitBExtraInfo &extraInfo, int64_t vec2S1Idx, LocalTensor<T> &bmm2Res,
                                         LocalTensor<T> &sumTensor, int64_t s1Vec2BaseSize);
    __aicore__ inline void Bmm2DataCopyOut(SplitBExtraInfo &extraInfo, int64_t vec2S1Idx, LocalTensor<T> &bmm2Res,
                                           LocalTensor<INPUT_T> &attentionOut);
    __aicore__ inline int64_t ComputeOffsetForCausal(const int64_t &delta, const uint32_t &s1BaseSize,
                                                     const uint32_t &s2BaseSize, const uint32_t &attenMaskS2Size);
    __aicore__ inline int64_t ComputeOffsetForPrefixRectangle(const int64_t &delta, const uint32_t &s2BaseSize,
                                                              const uint32_t &attenMaskS2Size);

    // 构建dataCopyTranspose参数
    CopyTransposeTiling dataCopyTiling;

    uint32_t s1BaseSize;
    uint32_t s1BaseTailSize;
    uint32_t s2BaseSize;
    uint32_t dSize;
    uint32_t dBaseSize;
    uint32_t s1Size;
    uint32_t s2Size;

    // sparse 用函数
    __aicore__ inline void GetS1LoopRange(int64_t &multiCoreInnerOffset, int64_t &multiCoreInnerLimit);
    __aicore__ inline void GetS2LoopRange();

    // sparse 用参数
    int64_t tensorABatchSize;
    int64_t tensorBBatchSize;

    // 资源分配
    TBuf<> maskTBufPing;      // 11K
    TBuf<> maskTBufPong;      // 11K
    TBuf<> pseTBuf;           // 16K
    TBuf<> stage1PingBuf;     // 32K
    TBuf<> stage1PongBuf;     // 32K
    TBuf<> softmaxSumPingBuf; // 8K
    TBuf<> softmaxSumPongBuf; // 8K
    TBuf<> softmaxMaxPingBuf; // 8K
    TBuf<> softmaxMaxPongBuf; // 8K
    TBuf<> softmaxExpBuf;     // 8K
    TBuf<> commonTBuf;        // 32k
    TBuf<> vecOut;            // 16k

    GlobalTensor<T> mm1ResPing;
    GlobalTensor<T> mm1ResPong;
    GlobalTensor<T> mm2ResPing;
    GlobalTensor<T> mm2ResPong;
    GlobalTensor<half> pseAlibiGm;
    GlobalTensor<INPUT_T> stage1ResPing;
    GlobalTensor<INPUT_T> stage1ResPong;
    GlobalTensor<uint8_t> dropoutWorkspaceGm;

    // 轴的乘积
    int64_t gS1D;
    int64_t n2GS1D;
    int64_t n2S2D;
    int64_t s1S2;
    int64_t gS1S2;
    int64_t n2GS1S2;
    int64_t gS1;
    int64_t n2GS1;
    int64_t gD;
    int64_t n2D;
    int64_t n2G;
    int64_t n2GD;
    int64_t bN2G;
    int64_t bN2GD;
    int64_t gS2;
    int64_t n2GS2;
    int64_t biN2G;
    int64_t biN2GS1D;
    int64_t biN2GS1;
    int64_t biN2GD;
    int64_t biN2S2D;
    int64_t biN2D;
    int64_t biN2GS1S2;
    int64_t biN2GS2;
    int64_t bBaseSize;
    int64_t s1OuterSize;
    int64_t s1Vec2BaseSize;
    int64_t s1Vec2BaseTailSize;
    int64_t s1Vec2OuterSize;
    int64_t dSizeAlign16;
    int64_t softmaxBufSize = 256;
    uint32_t negativeIntScalar = NEGATIVE_MIN_VAULE_FP32;
    constexpr static int32_t repeatMaxBytes = 256;
    constexpr static int32_t repeatMaxTimes = 255;
    int32_t repeatMaxSize;
    T negativeFloatScalar;
    T positiveFloatScalar;

    uint8_t attenMaskCompressMode;
    bool hasSink;

    int32_t blockIdx;
    const FlashAttentionScoreGeneralTilingData *__restrict tilingData;
    int64_t boIdx;
    int64_t currentN1Idx;

    TPipe *pipe;

    GlobalTensor<INPUT_T> queryGm;
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<INPUT_T> pseGm;
    __gm__ uint8_t *pseSlope;
    GM_ADDR prefixNAddr;
    GlobalTensor<INPUT_T> valueGm;
    GlobalTensor<INPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<uint8_t> dropMaskGm;
    GlobalTensor<uint8_t> attenMaskGmInt;
    GlobalTensor<float> sinkGm;

    bool dropMaskUnAligned;

    int64_t attenMaskOffsetPre;

    PseInfo pseInfo = {0};

    DropMaskInfo dropMaskInfo = {0};
};

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::Init(
                                __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                                __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *prefix,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *sink, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe)
{
    this->InitInput(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, sink, softmaxMax, softmaxSum,
                    softmaxOut, attentionOut, workspace, tiling, tPipe); // gm设置

    this->ComputeConstexpr();
    this->CalBatchSize();
    this->InitBuffer();
    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    DropOutBitModeInit(apiTmpBuffer);
    if (this->blockIdx < this->tilingData->multiCoreParams.coreNum) {
        LocalTensor<half> pseHelpBuffer = this->stage1PingBuf.template Get<half>();
        PseInnerAlibiCreate<hasPse>(this->pseAlibiGm, pseHelpBuffer, this->pseInfo);
    }
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::SetTiling(
    const FlashAttentionScoreGeneralTilingData *__restrict tilingData)
{
    // copy base params
    this->tilingData = tilingData;
};

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::InitInput(
                                                __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                                __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
                                                __gm__ uint8_t *paddingMask, __gm__ uint8_t *prefix,
                                                __gm__ uint8_t *attenMask, __gm__ uint8_t *sink, __gm__ uint8_t *softmaxMax,
                                                __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut,
                                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                                const FlashAttentionScoreGeneralTilingData *__restrict tiling,
                                                TPipe *tPipe)
{
    this->blockIdx = GetBlockIdx();
    this->pipe = tPipe;
    this->repeatMaxSize = this->repeatMaxBytes / sizeof(T);
    this->SetTiling(tiling);

    this->queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    this->pseGm.SetGlobalBuffer((__gm__ INPUT_T *)pse);
    this->pseSlope = pse;
    this->prefixNAddr = prefix;
    this->dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)dropMask);
    this->attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
    this->softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    this->softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    this->sinkGm.SetGlobalBuffer((__gm__ float *)sink);
    this->attentionOutGm.SetGlobalBuffer((__gm__ INPUT_T *)attentionOut);

    int64_t mm1ResultSize = this->tilingData->coreParams.bBaseSize * this->tilingData->inputParams.n2Size *
                            this->tilingData->inputParams.gSize * this->tilingData->inputParams.s1Size *
                            this->tilingData->coreParams.s2BaseSize * sizeof(T) * (sizeof(INPUT_T) / 2);
    int64_t mm2ResultSize = this->tilingData->coreParams.bBaseSize * this->tilingData->inputParams.n2Size *
                            this->tilingData->inputParams.gSize * this->tilingData->inputParams.s1Size *
                            this->tilingData->coreParams.dBaseSize * sizeof(T);

    int64_t pseInnerAlibiSize = this->tilingData->coreParams.pseAlibiBaseS1 *
                                this->tilingData->coreParams.pseAlibiBaseS2 * sizeof(half);
    // 512对齐
    int64_t mm1Offset = CeilDiv(mm1ResultSize, 512) * 512 / (sizeof(INPUT_T) / 2); // 与tiling算法保持一致
    int64_t mm2Offset = CeilDiv(mm2ResultSize, 512) * 512;

    int64_t mm1TotalOffset = mm1Offset * 2;
    int64_t mm2TotalOffset = mm2Offset * 2;
    int64_t pseAlibiOffset = CeilDiv(pseInnerAlibiSize, 512) * 512;

    uint64_t dropmaskWorkSpaceLen = this->tilingData->dropmaskParams.shapeTotalSize;
    dropmaskWorkSpaceLen = CeilDiv(dropmaskWorkSpaceLen, 512) * 512;
    uint64_t mm1ResPingAddr = (dropmaskWorkSpaceLen +
                               this->blockIdx * (mm1TotalOffset + mm2TotalOffset + pseAlibiOffset)) / sizeof(T);
    uint64_t mm1ResPongAddr = mm1ResPingAddr + mm1Offset / sizeof(T);  // 间隔1份bmm1Result空间
    uint64_t mm2ResPingAddr = mm1ResPongAddr + mm1Offset / sizeof(T);  // 间隔1份bmm1Result空间
    uint64_t mm2ResPongAddr = mm2ResPingAddr + mm2Offset / sizeof(T);  // 间隔1份bmm2Result空间
    uint64_t pseAlibiAddr = mm2ResPongAddr + mm2Offset / sizeof(T);    // 间隔1份bmm2Result空间

    // FP32场景，stage1Result不与bmm1Result共用空间，需要占用2倍mm1Offset空间
    if constexpr (IsSameType<INPUT_T, float>::value) {
        mm1ResPingAddr = (dropmaskWorkSpaceLen +
                          this->blockIdx * (mm1TotalOffset * 2 + mm2TotalOffset + pseAlibiOffset)) / sizeof(T);
        mm1ResPongAddr = mm1ResPingAddr + mm1Offset / sizeof(T);      // 间隔1份bmm1Result空间
        mm2ResPingAddr = mm1ResPongAddr + mm1Offset / sizeof(T) * 3;  // 间隔1份bmm1Result空间和2份stage1Result空间
        mm2ResPongAddr = mm2ResPingAddr + mm2Offset / sizeof(T);      // 间隔1份bmm2Result空间
        pseAlibiAddr = mm2ResPongAddr + mm2Offset / sizeof(T);        // 间隔1份bmm2Result空间
    }

    // bmm1Result，占用2倍mm1Offset空间
    this->mm1ResPing.SetGlobalBuffer((__gm__ T *)workspace + mm1ResPingAddr);
    this->mm1ResPong.SetGlobalBuffer((__gm__ T *)workspace + mm1ResPongAddr);

    // stage1Result，不占用/占用2倍mm1Offset空间
    if constexpr (!IsSameType<T, INPUT_T>::value) {
        this->stage1ResPing.SetGlobalBuffer((__gm__ INPUT_T *)workspace + (mm1ResPingAddr * 2));
        this->stage1ResPong.SetGlobalBuffer((__gm__ INPUT_T *)workspace + (mm1ResPongAddr * 2));
    } else {
        this->stage1ResPing.SetGlobalBuffer((__gm__ INPUT_T *)workspace + mm1ResPingAddr + mm1Offset / sizeof(T) * 2);
        this->stage1ResPong.SetGlobalBuffer((__gm__ INPUT_T *)workspace + mm1ResPongAddr + mm1Offset / sizeof(T) * 2);
    }

    // bmm2Result，占用2倍mm2Offset空间
    this->mm2ResPing.SetGlobalBuffer((__gm__ T *)workspace + mm2ResPingAddr);
    this->mm2ResPong.SetGlobalBuffer((__gm__ T *)workspace + mm2ResPongAddr);

    this->pseAlibiGm.SetGlobalBuffer((__gm__ half*)workspace + pseAlibiAddr * 2);
    // dropout workspace
    dropoutWorkspaceGm.SetGlobalBuffer((__gm__ uint8_t *)workspace);

    this->dropMaskUnAligned = this->tilingData->inputParams.needDropMaskOp == 1;
    if constexpr (IsSameType<T, half>::value) {
        this->negativeIntScalar = NEGATIVE_MIN_VAULE_FP16;
    }
    GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);

    this->attenMaskCompressMode = this->tilingData->inputParams.attenMaskCompressMode;
    this->hasSink = this->tilingData->inputParams.needSinkOp;
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::ComputeConstexpr()
{
    // 计算轴的乘积
    this->s1S2 = this->tilingData->inputParams.s1Size * this->tilingData->inputParams.s2Size;
    this->gS1D = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s1Size *
                 this->tilingData->inputParams.dSize;
    this->dBaseSize = this->tilingData->coreParams.dBaseSize;
    this->s1Size = this->tilingData->inputParams.s1Size;
    this->s2Size = this->tilingData->inputParams.s2Size;
    this->dSize = this->tilingData->inputParams.dSize;
    this->dSizeAlign16 = CeilDiv(this->tilingData->inputParams.dSize, 16) * 16;
    this->bBaseSize = this->tilingData->coreParams.bBaseSize;
    this->gS1S2 = this->tilingData->inputParams.gSize * this->s1S2;
    this->gD = this->tilingData->inputParams.gSize * this->tilingData->inputParams.dSize;
    this->gS1 = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s1Size;
    this->gS2 = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s2Size;
    this->n2D = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.dSize;
    this->n2G = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize;
    this->n2GD = this->tilingData->inputParams.n2Size * this->gD;
    this->bN2G = this->tilingData->inputParams.bSize * this->tilingData->inputParams.n2Size *
                 this->tilingData->inputParams.gSize;
    this->bN2GD = this->bN2G * this->dSize;
    this->n2GS1D = this->tilingData->inputParams.n2Size * this->gS1D;
    this->n2S2D = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.s2Size *
                  this->tilingData->inputParams.dSize;
    this->n2GS1S2 = this->tilingData->inputParams.n2Size * this->gS1S2;
    this->n2GS1 = this->tilingData->inputParams.n2Size * this->gS1;
    this->n2GS2 = this->tilingData->inputParams.n2Size * this->gS2;
    this->s1BaseSize = this->tilingData->coreParams.s1BaseSize;
    this->s2BaseSize = this->tilingData->coreParams.s2BaseSize; // 16对齐的S2
    if constexpr (hasPse == true) {
        this->pseInfo.pseBSize = this->tilingData->inputParams.pseBSize;
        this->pseInfo.s1Size = this->s1Size;
        this->pseInfo.s2Size = this->s2Size;
        this->pseInfo.n2G = this->n2G;
        this->pseInfo.s2RealSize = this->s2Size;
        this->pseInfo.pseShapeType = this->tilingData->inputParams.pseShapeType;
        this->pseInfo.pseType = this->tilingData->inputParams.pseType;
        this->pseInfo.pseAlibiBaseS1 = this->tilingData->coreParams.pseAlibiBaseS1;
        this->pseInfo.pseAlibiBaseS2 = this->tilingData->coreParams.pseAlibiBaseS2;
        this->pseInfo.qStartIdx = this->tilingData->inputParams.qStartIdx;
        this->pseInfo.kvStartIdx = this->tilingData->inputParams.kvStartIdx;
        this->pseInfo.pseEndogenous = (this->pseInfo.pseType == (int64_t)PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            this->pseInfo.pseType == (int64_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) ? true : false;
    }
    if constexpr (hasDrop == true) {
        this->dropMaskInfo.s1Size = this->s1Size;
        this->dropMaskInfo.s2Size = this->s2Size;
        this->dropMaskInfo.n2G = this->n2G;
    }
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::RefreshConstexpr()
{
    if (this->blockIdx == this->tilingData->multiCoreParams.coreNum - 1 &&
        this->boIdx == this->tilingData->coreParams.bOuterSize - 1) {
        this->bBaseSize = this->tilingData->coreParams.bBaseTailSize;
    }
    this->biN2G = this->bBaseSize * this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize;
    this->biN2GS1D = this->bBaseSize * this->n2GS1D;
    this->biN2GD = this->bBaseSize * this->n2GD;
    this->biN2S2D = this->bBaseSize * this->n2S2D;
    this->biN2D = this->bBaseSize * this->n2D;
    this->biN2GS1S2 = this->bBaseSize * this->n2GS1S2;
    this->biN2GS2 = this->bBaseSize * this->n2GS2;
    this->biN2GS1 = this->bBaseSize * this->n2GS1;
    this->s1OuterSize = this->tilingData->coreParams.s1OuterSize;
    this->s1BaseSize = this->tilingData->coreParams.s1BaseSize;
    this->s1BaseTailSize = this->tilingData->coreParams.s1BaseTailSize;
    this->s1Vec2BaseSize = this->tilingData->coreParams.s1Vec2BaseSize;
    this->s1Vec2BaseTailSize = this->tilingData->coreParams.s1Vec2BaseTailSize;
    this->s1Vec2OuterSize = this->tilingData->coreParams.s1Vec2OuterSize;
    this->tensorABatchSize =
        this->bBaseSize * this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize;
    this->tensorBBatchSize = this->bBaseSize * this->tilingData->inputParams.n2Size;
    if constexpr (hasPse == true) {
        this->pseInfo.bSSOffset = this->bBaseSize * this->s1S2;
        this->pseInfo.s2SizeAcc = this->bBaseSize * this->s2Size;
        this->pseInfo.s1BaseSize = this->s1BaseSize;
    }
    if constexpr (hasDrop == true) {
        this->dropMaskInfo.bSSOffset = this->bBaseSize * this->s1S2;
        this->dropMaskInfo.s1BaseSize = this->s1BaseSize;
    }
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::CalBatchSize()
{
    this->tensorABatchSize = this->bBaseSize * this->tilingData->inputParams.n2Size *
                             this->tilingData->inputParams.gSize;
    this->tensorBBatchSize = this->bBaseSize * this->tilingData->inputParams.n2Size;
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::InitBuffer()
{
    this->pipe->InitBuffer(this->maskTBufPing, 11 * 1024);             // 可以给attenmask 11k
    this->pipe->InitBuffer(this->maskTBufPong, 16 * 1024);             // 可以给dropoutmask和pse 16k
    this->pipe->InitBuffer(this->pseTBuf, 16384); // pse 16k

    this->pipe->InitBuffer(this->stage1PingBuf, 8 * 1024 * sizeof(T)); // t.a 32k
    this->pipe->InitBuffer(this->stage1PongBuf, 8 * 1024 * sizeof(T)); // i.a 32k
    this->pipe->InitBuffer(this->commonTBuf, 64 * 128 * sizeof(T));    // t.b 32k

    this->pipe->InitBuffer(this->softmaxSumPingBuf, this->softmaxBufSize * blockBytes); // 8k max
    this->pipe->InitBuffer(this->softmaxSumPongBuf, this->softmaxBufSize * blockBytes); // 8k max
    this->pipe->InitBuffer(this->softmaxMaxPingBuf, this->softmaxBufSize * blockBytes); // 8k max
    this->pipe->InitBuffer(this->softmaxMaxPongBuf, this->softmaxBufSize * blockBytes); // 8k max
    this->pipe->InitBuffer(this->softmaxExpBuf, blockBytes);      // 当前模版exp未使用，分配32B即可

    // 存放vector的输出，为Cast成fp16之后的结果
    this->pipe->InitBuffer(this->vecOut, 16384); // 16k
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::Process()
{
    int64_t multiCoreInnerOffset = this->blockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize;
    if (this->tilingData->multiCoreParams.totalSize < multiCoreInnerLimit) {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
    }

    SplitBExtraInfo extraInfo[3];
    int64_t taskId = 0;
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    bool notSecondLast = true;
    bool notLast = true;
    multiCoreInnerLimit += 2;
    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
         multiCoreInnerIdx++) {
        this->boIdx = multiCoreInnerIdx;
        if (multiCoreInnerIdx == multiCoreInnerLimit - 2) {
            notSecondLast = false;
        } else if (multiCoreInnerIdx == multiCoreInnerLimit - 1) {
            notLast = false;
        }
        bool notLastTwoLoop = notSecondLast && notLast;
        RefreshConstexpr();

        if (taskId >= 1 && notLast) {
            WaitBmm1Result();
        }

        if (notLastTwoLoop) {
            this->SetExtraInfo(extraInfo[taskId % 3], taskId, multiCoreInnerIdx);
            this->IterateBmm1(extraInfo[taskId % 3]);
        }

        if (taskId > 0 && notLast) {
            this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }

        if (taskId > 1) {
            if (likely(this->dSizeAlign16 == this->dSize)) {
                WaitBmm2Result(this->bmm2);
            } else {
                WaitBmm2Result(this->bmm2Nz);
            }
        }

        if (taskId > 0 && notLast) {
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            if (likely(this->dSizeAlign16 == this->dSize)) {
                this->IterateBmm2(extraInfo[(taskId + 2) % 3], this->bmm2);
            } else {
                this->IterateBmm2(extraInfo[(taskId + 2) % 3], this->bmm2Nz);
            }
        }

        if (taskId > 1) {
            this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
        }
        taskId++;
    }
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::WaitBmm1Result()
{
    bmm1.WaitIterateBatch();
    bmm1.End();
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::SetExtraInfo(
    SplitBExtraInfo &extraInfo, int64_t taskId, int64_t multiCoreInnerIdx)
{
    extraInfo.boIdx = multiCoreInnerIdx;
    extraInfo.taskId = taskId;
    extraInfo.vecS1BaseSize = this->s1BaseSize;
    extraInfo.vecS1TailSize = this->s1BaseTailSize;
    extraInfo.s2AlignSize = this->s2BaseSize;
    extraInfo.s2AlignBlockSize = CeilDiv(this->s2BaseSize, blockBytes) * blockBytes;
    extraInfo.s1Vec2BaseSize = this->s1Vec2BaseSize;
    extraInfo.s1Vec2BaseTailSize = this->s1Vec2BaseTailSize;
    extraInfo.s1Vec2OuterSize = this->s1Vec2OuterSize;
    extraInfo.softmaxCopyOutLimit = this->softmaxBufSize / extraInfo.vecS1BaseSize;
    extraInfo.softmaxCopyOutSize = Min(this->s1OuterSize, extraInfo.softmaxCopyOutLimit) * this->s1BaseSize;
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::IterateBmm1(SplitBExtraInfo &extraInfo)
{
    int64_t qCoreOffset = this->ComputeQCoreOffset(extraInfo);
    int64_t kvCoreOffset = this->ComputeKVCoreOffset(extraInfo);
    extraInfo.qCoreOffset = qCoreOffset;
    this->bmm1.SetTensorA(this->queryGm[qCoreOffset]);
    this->bmm1.SetTensorB(this->keyGm[kvCoreOffset], true);
    if (extraInfo.taskId % 2 == 0) {
        this->bmm1.template IterateBatch<false, true>(this->mm1ResPing, this->tensorABatchSize, this->tensorBBatchSize,
                                                      false);
    } else {
        this->bmm1.template IterateBatch<false, true>(this->mm1ResPong, this->tensorABatchSize, this->tensorBBatchSize,
                                                      false);
    }
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
template <typename T2, const auto &TILING>
__aicore__ inline void FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::WaitBmm2Result(
    matmul::Matmul<a2Type, b2Type, T2, bias2Type, TILING> &bmm2)
{
    bmm2.WaitIterateBatch();
    bmm2.End();
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
template <typename T2, const auto &TILING>
__aicore__ inline void FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::IterateBmm2(
    SplitBExtraInfo &extraInfo, matmul::Matmul<a2Type, b2Type, T2, bias2Type, TILING> &bmm2)
{
    int64_t kvCoreOffset = this->ComputeKVCoreOffset(extraInfo);

    if (extraInfo.taskId % 2 == 0) {
        bmm2.SetTensorA(this->stage1ResPing);
    } else {
        bmm2.SetTensorA(this->stage1ResPong);
    }
    bmm2.SetTensorB(this->valueGm[kvCoreOffset]);
    if (extraInfo.taskId % 2 == 0) {
        bmm2.template IterateBatch<false, true>(this->mm2ResPing, this->tensorABatchSize, this->tensorBBatchSize,
                                                false);
    } else {
        bmm2.template IterateBatch<false, true>(this->mm2ResPong, this->tensorABatchSize, this->tensorBBatchSize,
                                                false);
    }
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::ComputeQCoreOffset(SplitBExtraInfo &extraInfo)
{
    // 计算gm上的offset
    int64_t qBOffset = 0;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        // BSH/BNGS1S2
        qBOffset = extraInfo.boIdx * this->biN2GS1D;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        // SBH/SBNGD
        qBOffset = extraInfo.boIdx * this->biN2GD;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        // BNGSD
        qBOffset = extraInfo.boIdx * this->biN2GS1D;
    }

    return qBOffset;
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::ComputeKVCoreOffset(SplitBExtraInfo &extraInfo)
{
    // 计算gm上的offset
    int64_t kvBOffset = 0;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        // BSH/BSND
        kvBOffset = extraInfo.boIdx * this->biN2S2D;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        // SBH/SBND
        kvBOffset = extraInfo.boIdx * this->biN2D;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        // BNSD
        kvBOffset = extraInfo.boIdx * this->biN2S2D;
    }

    return kvBOffset;
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::GetBmm1Result(
    SplitBExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb, int64_t loopIdx)
{
    int32_t dtypeSize = sizeof(T);
    int32_t s2Align8 = (this->s2Size + 7) / 8 * 8;
    if (extraInfo.s2AlignSize == this->s2Size) {
        // 16对齐场景，直接DataCopy提升性能
        if (extraInfo.taskId % 2 == 0) {
            DataCopy(bmm1ResUb,
                     this->mm1ResPing[extraInfo.biN2GoIdx * this->s1Size * this->s2Size +
                                      loopIdx * this->s1BaseSize * this->s2Size],
                     extraInfo.vecS1BaseSize * this->s2Size);
        } else {
            DataCopy(bmm1ResUb,
                     this->mm1ResPong[extraInfo.biN2GoIdx * this->s1Size * this->s2Size +
                                      loopIdx * this->s1BaseSize * this->s2Size],
                     extraInfo.vecS1BaseSize * this->s2Size);
        }
    } else {
        DataCopyParams dataCopyParams;
        DataCopyPadParams dataCopyPadParams;
        dataCopyParams.blockCount = extraInfo.vecS1BaseSize;
        dataCopyParams.dstStride = 0;
        dataCopyParams.srcStride = 0;
        dataCopyParams.blockLen = this->s2Size * dtypeSize;
        dataCopyPadParams.rightPadding = extraInfo.s2AlignSize - this->s2Size;
        dataCopyPadParams.paddingValue = 0;
        if (dataCopyPadParams.rightPadding > blockSize) {
            // 8对齐场景，内部vector需要16对齐，我们在data copy的时候需要手动补0
            dataCopyPadParams.rightPadding -= blockSize;
            dataCopyParams.dstStride = 1;
            Duplicate<T>(bmm1ResUb[s2Align8], 0, blockSize, extraInfo.vecS1BaseSize, 0,
                         extraInfo.s2AlignSize * sizeof(T) / blockBytes);
        }
        if (extraInfo.taskId % 2 == 0) {
            DataCopyPad(bmm1ResUb,
                        this->mm1ResPing[extraInfo.biN2GoIdx * this->s1Size * this->s2Size +
                                         loopIdx * this->s1BaseSize * this->s2Size],
                        dataCopyParams, dataCopyPadParams);
        } else {
            DataCopyPad(bmm1ResUb,
                        this->mm1ResPong[extraInfo.biN2GoIdx * this->s1Size * this->s2Size +
                                         loopIdx * this->s1BaseSize * this->s2Size],
                        dataCopyParams, dataCopyPadParams);
        }
    }
    uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(extraInfo.vecS1BaseSize),
                                 static_cast<uint32_t>(extraInfo.s2AlignSize)};
    bmm1ResUb.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, DataFormat::ND));
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::ProcessVec1(SplitBExtraInfo &extraInfo)
{
    LocalTensor<T> stage1PingTensor = this->stage1PingBuf.template Get<T>(); // t.a 32k
    LocalTensor<T> stage1PongTensor = this->stage1PongBuf.template Get<T>(); // i.a 32k
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t eventIdVToMte2A = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdVToMte2B = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventIdPseDropVToMte2A;
    event_t eventIdPseDropVToMte2B;
    if constexpr (hasPse == true && hasDrop == true && !IsSameType<INPUT_T, T>::value) {
        eventIdPseDropVToMte2A = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        if (!this->pseInfo.pseEndogenous) {
            eventIdPseDropVToMte2B = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        }
    }
    uint32_t loopIdxNew = 0;
    for (uint32_t biN2GoIdx = 0; biN2GoIdx < this->biN2G; biN2GoIdx++) {
        extraInfo.biN2GoIdx = biN2GoIdx;
        extraInfo.softmaxOutOffset = 0;
        this->currentN1Idx = biN2GoIdx % (this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize);
        for (uint32_t loopIdx = 0; loopIdx < this->s1OuterSize; loopIdx++, loopIdxNew++) {
            extraInfo.vecS1BaseSize = this->s1BaseSize;
            if (loopIdx == this->s1OuterSize - 1) {
                extraInfo.vecS1BaseSize = extraInfo.vecS1TailSize;
            }
            extraInfo.s1oIdx = loopIdx;

            if (loopIdxNew > 0) {
                AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
            }

            // FP32场景，需要等待vec1上一轮输出搬完
            if constexpr (IsSameType<INPUT_T, float>::value) {
                event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }
            if constexpr (hasPse == true) {
                this->pseInfo.s2AlignedSize = extraInfo.s2AlignSize;
                this->pseInfo.vec1S1RealSize = extraInfo.vecS1BaseSize;
                this->pseInfo.vec1S1BaseSize = extraInfo.vecS1BaseSize;
                this->pseInfo.boIdx = extraInfo.boIdx;
                this->pseInfo.goIdx = extraInfo.biN2GoIdx;
                this->pseInfo.s1oIdx = extraInfo.s1oIdx;
                this->pseInfo.bSSOffset = this->pseInfo.boIdx * this->bBaseSize * this->s1S2;
                this->pseInfo.s2SizeAcc = this->pseInfo.boIdx * this->bBaseSize * this->s2Size;
                this->pseInfo.needCast = true;
                stage1PingTensor.SetSize(extraInfo.vecS1BaseSize * extraInfo.s2AlignSize);
                if constexpr (hasDrop == true && !IsSameType<INPUT_T, T>::value) {
                    if (loopIdxNew > 0) {
                        AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdPseDropVToMte2A);
                    }
                }
                if (this->pseInfo.pseEndogenous) {
                    LocalTensor<half> pseUb = this->maskTBufPong.template Get<half>();
                    PseSlopeCopyIn<T, hasPse>(stage1PingTensor, pseUb, this->pseSlope, this->pseAlibiGm, this->pseInfo);
                } else {
                    LocalTensor<INPUT_T> pseUb = this->maskTBufPong.template Get<INPUT_T>();
                    PseCopyIn<INPUT_T, T, layOutType, hasPse>(stage1PingTensor, pseUb, this->pseGm, this->pseInfo);
                    // FP32场景，需要等PSE输入搬完再启动计算
                    if constexpr (IsSameType<INPUT_T, float>::value) {
                        AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                        AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                    }
                    if constexpr (hasDrop == true && !IsSameType<INPUT_T, T>::value) {
                        AscendC::SetFlag<HardEvent::V_MTE2>(eventIdPseDropVToMte2B);
                    }
                }
            }
            this->GetBmm1Result(extraInfo, stage1PongTensor, loopIdx);
            AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            if (this->tilingData->inputParams.pseType != (uint32_t)PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(stage1PongTensor, stage1PongTensor, static_cast<T>(this->tilingData->inputParams.scaleValue),
                     extraInfo.vecS1BaseSize * extraInfo.s2AlignSize);
            }
            if constexpr (hasPse == true) {
                AscendC::PipeBarrier<PIPE_V>();
                PseCompute<T, hasPse>(stage1PongTensor, stage1PingTensor, this->pseInfo);
            }
            this->CopyInAttenMask(extraInfo, -1);
            if (this->tilingData->inputParams.pseType == (uint32_t)PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                AscendC::PipeBarrier<PIPE_V>();
                Muls(stage1PingTensor, stage1PongTensor, static_cast<T>(this->tilingData->inputParams.scaleValue),
                     extraInfo.vecS1BaseSize * extraInfo.s2AlignSize);
            } else {
                AscendC::PipeBarrier<PIPE_V>();
                Muls(stage1PingTensor, stage1PongTensor, static_cast<T>(1.0),
                     extraInfo.vecS1BaseSize * extraInfo.s2AlignSize);
            }
            AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            if constexpr (hasAtten == true) {
                if (this->attenMaskCompressMode != static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) {
                    this->ComputeAttenMask(extraInfo, stage1PingTensor, 0);
                }
                if (this->attenMaskCompressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
                    event_t eventIdVToMte2Tmp = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                    AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2Tmp);
                    AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2Tmp);
                    this->CopyInAttenMask(extraInfo, this->attenMaskOffsetPre);
                    event_t eventIdMte2ToVTmp = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVTmp);
                    AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVTmp);
                    this->ComputeAttenMask(extraInfo, stage1PingTensor, 1);
                }
                if (this->attenMaskCompressMode == static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) {
                    event_t eventIdMte3ToMte2 =
                        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                    AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                    this->CopyInAttenMask(extraInfo, this->attenMaskOffsetPre);
                    int32_t maskTotalNum =
                        extraInfo.vecS1BaseSize * extraInfo.s2AlignBlockSize / 2; // 除2数据量按照uint16类型折半
                    event_t eventIdMte2ToVTmp = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVTmp);
                    AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVTmp);
                    LocalTensor<uint8_t> attenMaskCasualUb = this->maskTBufPing.template Get<uint8_t>();
                    LocalTensor<uint8_t> attenMaskPrefixUb = this->pseTBuf.template Get<uint8_t>();
                    auto attenMaskCasualTmp = attenMaskCasualUb.ReinterpretCast<uint16_t>();
                    auto attenMaskPrefixUbTmp = attenMaskPrefixUb.ReinterpretCast<uint16_t>();
                    And(attenMaskCasualTmp, attenMaskCasualTmp, attenMaskPrefixUbTmp, maskTotalNum);
                    AscendC::PipeBarrier<PIPE_V>();
                    attenMaskCasualUb = attenMaskCasualTmp.ReinterpretCast<uint8_t>();
                    this->ComputeAttenMask(extraInfo, stage1PingTensor, 0);
                }
            }
            if (loopIdxNew < this->biN2G * this->s1OuterSize - 1 && !hasSink) {
                AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
            }

            if (loopIdxNew > 0) {
                AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
            }

            if constexpr (hasDrop == true) {
                LocalTensor<uint8_t> dropMaskUb = this->maskTBufPong.template Get<uint8_t>();
                this->dropMaskInfo.splitS1BaseSize = extraInfo.vecS1BaseSize;
                this->dropMaskInfo.gOutIdx = extraInfo.biN2GoIdx;
                this->dropMaskInfo.s1OutIdx = extraInfo.s1oIdx;
                this->dropMaskInfo.bSSOffset = extraInfo.boIdx * this->bBaseSize * this->s1S2;
                this->dropMaskInfo.s1CopySize = static_cast<uint32_t>(extraInfo.vecS1BaseSize);
                this->dropMaskInfo.s2CopySize = this->s2Size;
                this->dropMaskInfo.s2TotalSize = static_cast<int64_t>(this->s2Size);
                this->dropMaskInfo.boolMode = this->dropMaskUnAligned;
                if constexpr (hasPse == true && !IsSameType<INPUT_T, T>::value) {
                    if (!this->pseInfo.pseEndogenous) {
                        AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdPseDropVToMte2B);
                    }
                }
                CopyInDropMask<hasDrop>(dropMaskUb, dropoutWorkspaceGm, this->dropMaskGm, this->dropMaskInfo);
            }

            this->SoftMaxCompute(extraInfo, stage1PingTensor, loopIdx);
            if (loopIdxNew < this->biN2G * this->s1OuterSize - 1 && hasSink) {
                AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
            }
            if constexpr (hasDrop == true) {
                LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
                LocalTensor<uint8_t> dropMaskUb = this->maskTBufPong.template Get<uint8_t>();
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                this->dropMaskInfo.firstAxis = static_cast<uint32_t>(extraInfo.vecS1BaseSize);
                this->dropMaskInfo.lstAxis = static_cast<uint32_t>(extraInfo.s2AlignSize);
                this->dropMaskInfo.maskLstAxis = this->s2Size;
                this->dropMaskInfo.keepProb = this->tilingData->inputParams.keepProb;
                ComputeDropMask<T, hasDrop>(stage1PingTensor, stage1PingTensor, dropMaskUb, apiTmpBuffer,
                                            this->dropMaskInfo);
                if constexpr (hasPse == true && !IsSameType<INPUT_T, T>::value) {
                    if (loopIdxNew < this->biN2G * this->s1OuterSize - 1) {
                        AscendC::SetFlag<HardEvent::V_MTE2>(eventIdPseDropVToMte2A);
                    }
                }
            }

            if (loopIdxNew < this->biN2G * this->s1OuterSize - 1) {
                AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
            }
            AscendC::PipeBarrier<PIPE_V>();

            if (loopIdxNew > 0) {
                AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            }

            if constexpr (!IsSameType<T, INPUT_T>::value) {
                LocalTensor<INPUT_T> stage1CastTensor = this->vecOut.template Get<INPUT_T>();
                Cast(stage1CastTensor, stage1PingTensor, RoundMode::CAST_ROUND,
                     extraInfo.vecS1BaseSize * extraInfo.s2AlignSize);
                AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                DataCopyParams dataCopyParams;
                DataCopyPadParams dataCopyPadParams;
                dataCopyParams.blockCount = extraInfo.vecS1BaseSize;
                dataCopyParams.dstStride = 0;
                dataCopyParams.srcStride = 0;
                dataCopyParams.blockLen = this->s2Size * sizeof(INPUT_T);
                if (extraInfo.taskId % 2 == 0) {
                    DataCopyPad(this->stage1ResPing[biN2GoIdx * this->s1Size * this->s2Size +
                                                    loopIdx * this->s1BaseSize * this->s2Size],
                                stage1CastTensor, dataCopyParams);
                } else {
                    DataCopyPad(this->stage1ResPong[biN2GoIdx * this->s1Size * this->s2Size +
                                                    loopIdx * this->s1BaseSize * this->s2Size],
                                stage1CastTensor, dataCopyParams);
                }
            } else {
                AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                DataCopyParams dataCopyParams;
                DataCopyPadParams dataCopyPadParams;
                dataCopyParams.blockCount = extraInfo.vecS1BaseSize;
                dataCopyParams.dstStride = 0;
                dataCopyParams.srcStride = (extraInfo.s2AlignSize - this->s2Size) / 8;
                dataCopyParams.blockLen = this->s2Size * sizeof(INPUT_T);
                if (extraInfo.taskId % 2 == 0) {
                    DataCopyPad(this->stage1ResPing[biN2GoIdx * this->s1Size * this->s2Size +
                                                    loopIdx * this->s1BaseSize * this->s2Size],
                                stage1PingTensor, dataCopyParams);
                } else {
                    DataCopyPad(this->stage1ResPong[biN2GoIdx * this->s1Size * this->s2Size +
                                                    loopIdx * this->s1BaseSize * this->s2Size],
                                stage1PingTensor, dataCopyParams);
                }
            }
            if (loopIdxNew < this->biN2G * this->s1OuterSize - 1) {
                AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            }
        }
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2A);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2B);
    if constexpr (hasPse == true && hasDrop == true && !IsSameType<INPUT_T, T>::value) {
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdPseDropVToMte2A);
        if (!this->pseInfo.pseEndogenous) {
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdPseDropVToMte2B);
        }
    }
    return;
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::NzToNd(
    SplitBNz2NdInfo &nz2NdInfo, const GlobalTensor<T> &bmmResGm, LocalTensor<T> &tempUb, LocalTensor<T> &bmmResUb)
{
    // 1.将bmm1结果由GM搬至UB，每块数据在UB上间隔1个block，防止BANK冲突
    DataCopyParams dataCopyParams;
    int64_t nzFirstAxis = CeilDiv(nz2NdInfo.ndLastAxis, 16L);
    dataCopyParams.blockCount = nzFirstAxis;
    dataCopyParams.blockLen = nz2NdInfo.ndFirstAxisLoopSize * 2;
    dataCopyParams.srcStride = (nz2NdInfo.ndFirstAxisRealSize - nz2NdInfo.ndFirstAxisLoopSize) * 2;
    dataCopyParams.dstStride = 1;
    int64_t innerLoop = nzFirstAxis / 8L;
    int64_t innerRemain = nzFirstAxis % 8L;

    CopyRepeatParams repeatParams;
    repeatParams.srcStride = nz2NdInfo.ndFirstAxisLoopSize * 2 + 1;
    repeatParams.dstStride = 2;
    repeatParams.srcRepeatSize = 2;
    repeatParams.dstRepeatSize = nz2NdInfo.ndLastAxis / 8;
    int32_t outerLoop = nz2NdInfo.ndFirstAxisLoopSize / this->repeatMaxTimes;
    int32_t outerRemain = nz2NdInfo.ndFirstAxisLoopSize % this->repeatMaxTimes;
    int32_t outerBmmOffset = this->repeatMaxTimes * nz2NdInfo.ndLastAxis;
    int32_t outerTempOffset = this->repeatMaxTimes * 16;
    int64_t offsetJ = 128 * nz2NdInfo.ndFirstAxisLoopSize + 64;
    DataCopy(tempUb, bmmResGm[nz2NdInfo.bmmResOffset], dataCopyParams);

    // 2.使用vcopy进行transpose，[S2/16, vec1S1Base * 16 + 8] -> [vec1S1Base, S2/16 , 16]
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    for (int64_t outerIndex = 0; outerIndex < outerLoop; ++outerIndex) {
        for (int64_t i = 0; i < 2; ++i) {
            for (int64_t j = 0; j < innerLoop; ++j) {
                Copy(bmmResUb[outerIndex * outerBmmOffset + j * 128 + i * 8],
                     tempUb[outerIndex * outerTempOffset + j * offsetJ + i * 8], this->repeatMaxSize,
                     this->repeatMaxTimes, repeatParams);
            }
            if (likely(innerRemain)) {
                Copy(bmmResUb[outerIndex * outerBmmOffset + innerLoop * 128 + i * 8],
                     tempUb[outerIndex * outerTempOffset + innerLoop * offsetJ + i * 8], innerRemain * 8,
                     this->repeatMaxTimes, repeatParams);
            }
        }
    }
    if (likely(outerRemain)) {
        for (int64_t i = 0; i < 2; ++i) {
            for (int64_t j = 0; j < innerLoop; ++j) {
                Copy(bmmResUb[outerLoop * outerBmmOffset + j * 128 + i * 8],
                     tempUb[outerLoop * outerTempOffset + j * offsetJ + i * 8], this->repeatMaxSize, outerRemain,
                     repeatParams);
            }
            if (likely(innerRemain)) {
                Copy(bmmResUb[outerLoop * outerBmmOffset + innerLoop * 128 + i * 8],
                     tempUb[outerLoop * outerTempOffset + innerLoop * offsetJ + i * 8], innerRemain * 8,
                     outerRemain, repeatParams);
            }
        }
    }
    uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(nz2NdInfo.ndFirstAxisLoopSize),
                                 static_cast<uint32_t>(nz2NdInfo.ndLastAxis)};
    bmmResUb.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, DataFormat::ND));
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::ProcessVec2(SplitBExtraInfo &extraInfo)
{
    int64_t dAlign8 = (this->dSize + 7) / 8 * 8;
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    for (uint32_t biN2GoIdx = 0; biN2GoIdx < this->biN2G; ++biN2GoIdx) {
        extraInfo.biN2GoIdx = biN2GoIdx;
        // 获取缓存bmm2的计算结果
        LocalTensor<T> bmm2ResPingUb = this->stage1PingBuf.template Get<T>();
        LocalTensor<T> bmm2ResPongUb = this->stage1PongBuf.template Get<T>();
        LocalTensor<T> softmaxSumPingUb = this->pseTBuf.template Get<float>();
        LocalTensor<T> softmaxSumPongUb = this->vecOut.template Get<float>();

        // pseTBuf和vecOut同时作为Softmax输入和最后attention out输出的ub
        LocalTensor<INPUT_T> attentionOutPingUb = this->pseTBuf.template Get<INPUT_T>();
        LocalTensor<INPUT_T> attentionOutPongUb = this->vecOut.template Get<INPUT_T>();

        int64_t bOffset = extraInfo.boIdx * this->biN2GS1 * fp32BaseSize;
        int64_t biN2GOffset = extraInfo.biN2GoIdx * this->s1Size * this->dSize;
        int64_t biN2GSumOffset = extraInfo.biN2GoIdx * this->s1Size * fp32BaseSize;
        int64_t sumGmOffset = bOffset + biN2GSumOffset;
        for (int64_t s1oIdx = 0; s1oIdx < extraInfo.s1Vec2OuterSize; ++s1oIdx) {
            int64_t sumGmRealOffset = 0;
            extraInfo.s1Vec2BaseSize = this->s1Vec2BaseSize;
            if (s1oIdx == extraInfo.s1Vec2OuterSize - 1) {
                extraInfo.s1Vec2BaseSize = extraInfo.s1Vec2BaseTailSize;
            }
            int64_t sumInnerSize = extraInfo.s1Vec2BaseSize * fp32BaseSize;
            int64_t mm2ResCalcSize = extraInfo.s1Vec2BaseSize * this->dSizeAlign16;
            int64_t mm2ResOffset = biN2GOffset + s1oIdx * this->s1Vec2BaseSize * this->dSize;
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            sumGmRealOffset = sumGmOffset + s1oIdx * this->s1Vec2BaseSize * fp32BaseSize;
            if constexpr (IsSameType<INPUT_T, float>::value) {
                bmm2ResPingUb.SetSize(mm2ResCalcSize);
                bmm2ResPongUb.SetSize(mm2ResCalcSize);
            }
            if (extraInfo.taskId % 2 == 0) {
                if (likely(this->dSizeAlign16 == this->dSize)) {
                    DataCopy(bmm2ResPingUb, this->mm2ResPing[mm2ResOffset], mm2ResCalcSize);
                } else {
                    SplitBNz2NdInfo nz2NdInfo;
                    nz2NdInfo.ndFirstAxisRealSize = this->s1Size;
                    nz2NdInfo.ndFirstAxisLoopSize = extraInfo.s1Vec2BaseSize;
                    nz2NdInfo.ndLastAxis = this->dSizeAlign16;
                    nz2NdInfo.bmmResOffset = extraInfo.biN2GoIdx * this->s1Size * this->dSizeAlign16 +
                                             s1oIdx * this->s1Vec2BaseSize * 16;
                    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                    AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
                    AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
                    NzToNd(nz2NdInfo, this->mm2ResPing, bmm2ResPongUb, bmm2ResPingUb);
                }
                DataCopy(softmaxSumPingUb, this->softmaxSumGm[sumGmRealOffset], sumInnerSize);
                Bmm2ResultDiv(extraInfo, s1oIdx, bmm2ResPingUb, softmaxSumPingUb, extraInfo.s1Vec2BaseSize);
                Bmm2DataCopyOut(extraInfo, s1oIdx, bmm2ResPingUb, attentionOutPingUb);
            } else {
                if (likely(this->dSizeAlign16 == this->dSize)) {
                    DataCopy(bmm2ResPongUb, this->mm2ResPong[mm2ResOffset], mm2ResCalcSize);
                } else {
                    SplitBNz2NdInfo nz2NdInfo;
                    nz2NdInfo.ndFirstAxisRealSize = this->s1Size;
                    nz2NdInfo.ndFirstAxisLoopSize = extraInfo.s1Vec2BaseSize;
                    nz2NdInfo.ndLastAxis = this->dSizeAlign16;
                    nz2NdInfo.bmmResOffset = extraInfo.biN2GoIdx * this->s1Size * this->dSizeAlign16 +
                                             s1oIdx * this->s1Vec2BaseSize * 16;
                    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                    AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
                    AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
                    NzToNd(nz2NdInfo, this->mm2ResPong, bmm2ResPingUb, bmm2ResPongUb);
                }
                DataCopy(softmaxSumPongUb, this->softmaxSumGm[sumGmRealOffset], sumInnerSize);
                Bmm2ResultDiv(extraInfo, s1oIdx, bmm2ResPongUb, softmaxSumPongUb, extraInfo.s1Vec2BaseSize);
                Bmm2DataCopyOut(extraInfo, s1oIdx, bmm2ResPongUb, attentionOutPongUb);
            }
        }
    }
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    return;
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::Bmm2ResultDiv(
    SplitBExtraInfo &extraInfo, int64_t vec2S1Idx, LocalTensor<T> &bmm2Res, LocalTensor<T> &sumTensor,
    int64_t vec2S1BaseSize)
{
    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src0RepStride = this->dSizeAlign16 / blockSize;
    repeatParams.src1BlkStride = 0;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = this->dSizeAlign16 / blockSize;
    int32_t s1OuterLoop = vec2S1BaseSize / repeatMaxTimes;
    int32_t s1OuterRemain = vec2S1BaseSize % repeatMaxTimes;
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<T, half>::value) {
        LocalTensor<T> sumCastTensor;
        if (extraInfo.taskId % 2 == 0) {
            sumCastTensor = this->pseTBuf.template Get<T>();
        } else {
            sumCastTensor = this->vecOut.template Get<T>();
        }
        Cast(sumCastTensor, sumTensor, RoundMode::CAST_ROUND, sumTensor.GetSize());
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t i = 0; i < s1OuterLoop; i++) {
            int32_t innerLoop = this->dSizeAlign16 / repeatMaxSize;
            int32_t innerRemain = this->dSizeAlign16 % repeatMaxSize;
            int64_t s1OuterOffset = i * repeatMaxTimes * this->dSizeAlign16;
            int64_t sumOffset = i * repeatMaxTimes * 8;
            for (int32_t j = 0; j < innerLoop; j++) {
                Div(bmm2Res[s1OuterOffset + j * repeatMaxSize], bmm2Res[s1OuterOffset + j * repeatMaxSize],
                    sumCastTensor[sumOffset], repeatMaxSize, repeatMaxTimes, repeatParams);
            }
            if (likely(innerRemain)) {
                Div(bmm2Res[s1OuterOffset + innerLoop * repeatMaxSize],
                    bmm2Res[s1OuterOffset + innerLoop * repeatMaxSize], sumCastTensor[sumOffset], innerRemain,
                    repeatMaxTimes, repeatParams);
            }
        }
        if (likely(s1OuterRemain)) {
            int32_t innerLoop = this->dSizeAlign16 / repeatMaxSize;
            int32_t innerRemain = this->dSizeAlign16 % repeatMaxSize;
            int64_t s1OuterOffset = s1OuterLoop * repeatMaxTimes * this->dSizeAlign16;
            int64_t sumOffset = s1OuterLoop * repeatMaxTimes * 8;
            for (int32_t j = 0; j < innerLoop; j++) {
                Div(bmm2Res[s1OuterOffset + j * repeatMaxSize], bmm2Res[s1OuterOffset + j * repeatMaxSize],
                    sumCastTensor[sumOffset], repeatMaxSize, s1OuterRemain, repeatParams);
            }
            if (likely(innerRemain)) {
                Div(bmm2Res[s1OuterOffset + innerLoop * repeatMaxSize],
                    bmm2Res[s1OuterOffset + innerLoop * repeatMaxSize], sumCastTensor[sumOffset], innerRemain,
                    s1OuterRemain, repeatParams);
            }
        }
    } else {
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        for (int32_t i = 0; i < s1OuterLoop; i++) {
            int32_t innerLoop = this->dSizeAlign16 / repeatMaxSize;
            int32_t innerRemain = this->dSizeAlign16 % repeatMaxSize;
            int64_t s1OuterOffset = i * repeatMaxTimes * this->dSizeAlign16;
            int64_t sumOffset = i * repeatMaxTimes * 8;
            for (int32_t j = 0; j < innerLoop; j++) {
                Div(bmm2Res[s1OuterOffset + j * repeatMaxSize], bmm2Res[s1OuterOffset + j * repeatMaxSize],
                    sumTensor[sumOffset], repeatMaxSize, repeatMaxTimes, repeatParams);
            }
            if (likely(innerRemain)) {
                Div(bmm2Res[s1OuterOffset + innerLoop * repeatMaxSize],
                    bmm2Res[s1OuterOffset + innerLoop * repeatMaxSize], sumTensor[sumOffset], innerRemain,
                    repeatMaxTimes, repeatParams);
            }
        }
        if (likely(s1OuterRemain)) {
            int32_t innerLoop = this->dSizeAlign16 / repeatMaxSize;
            int32_t innerRemain = this->dSizeAlign16 % repeatMaxSize;
            int64_t s1OuterOffset = s1OuterLoop * repeatMaxTimes * this->dSizeAlign16;
            int64_t sumOffset = s1OuterLoop * repeatMaxTimes * 8;
            for (int32_t j = 0; j < innerLoop; j++) {
                Div(bmm2Res[s1OuterOffset + j * repeatMaxSize], bmm2Res[s1OuterOffset + j * repeatMaxSize],
                    sumTensor[sumOffset], repeatMaxSize, s1OuterRemain, repeatParams);
            }
            if (likely(innerRemain)) {
                Div(bmm2Res[s1OuterOffset + innerLoop * repeatMaxSize],
                    bmm2Res[s1OuterOffset + innerLoop * repeatMaxSize], sumTensor[sumOffset], innerRemain,
                    s1OuterRemain, repeatParams);
            }
        }
    }
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::Bmm2DataCopyOut(
    SplitBExtraInfo &extraInfo, int64_t vec2S1Idx, LocalTensor<T> &bmm2Res, LocalTensor<INPUT_T> &attentionOut)
{
    uint32_t calcSize = bmm2Res.GetSize();
    AscendC::PipeBarrier<PIPE_V>();

    if constexpr (!IsSameType<INPUT_T, T>::value) {
        Cast(attentionOut, bmm2Res, RoundMode::CAST_ROUND, calcSize);
    } else {
        DataCopy(attentionOut, bmm2Res, calcSize);
    }

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = this->dSize * sizeof(INPUT_T);
    dataCopyParams.srcStride = 0;
    int64_t dstStride = 0;
    if constexpr (IsSameType<INPUT_T, float>::value) {
        if (this->dSizeAlign16 - this->dSize >= blockSize) {
            dataCopyParams.srcStride = 1;
        }
    }
    int64_t attenOutOffset = this->dSize;
    int64_t biOffset = (extraInfo.biN2GoIdx / this->n2G) * this->n2GS1D;
    int64_t n2GOffset = this->s1Size * this->dSize;
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        attenOutOffset = this->n2GD;
        n2GOffset = this->dSize;
        dstStride = (this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize - 1) * this->dSize *
                    sizeof(INPUT_T);
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        attenOutOffset = this->bN2GD;
        n2GOffset = this->dSize;
        biOffset = (extraInfo.biN2GoIdx / this->n2G) * this->n2GD;
        dstStride = (this->tilingData->inputParams.bSize * this->tilingData->inputParams.n2Size *
                         this->tilingData->inputParams.gSize -
                     1) *
                    this->dSize * sizeof(INPUT_T);
    }
    int64_t attOutBaseOffset = extraInfo.qCoreOffset + biOffset + (extraInfo.biN2GoIdx % this->n2G) * n2GOffset +
                               vec2S1Idx * this->s1Vec2BaseSize * attenOutOffset;
    // dataCopyParams.dstStride类型定义uint16_t，65535是其最大值
    if (likely(dstStride <= 65535)) {
        dataCopyParams.blockCount = extraInfo.s1Vec2BaseSize;
        dataCopyParams.dstStride = static_cast<uint16_t>(dstStride);
        DataCopyPad(this->attentionOutGm[attOutBaseOffset], attentionOut, dataCopyParams);
    } else {
        dataCopyParams.blockCount = 1;
        dataCopyParams.dstStride = 0;
        int64_t datacopyOffset = this->dSize;
        if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
            datacopyOffset = this->n2GD;
        } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
            datacopyOffset = this->bN2G * this->dSize;
        }

        for (uint32_t i = 0; i < extraInfo.s1Vec2BaseSize; i++) {
            DataCopyPad(this->attentionOutGm[attOutBaseOffset + i * datacopyOffset],
                        attentionOut[i * this->dSizeAlign16], dataCopyParams);
        }
    }
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::CopyInAttenMask(SplitBExtraInfo &extraInfo,
                                                                                     int64_t maskOffset)
{
    if constexpr (hasAtten == true) {
        LocalTensor<uint8_t> attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
        if (this->attenMaskCompressMode == static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE) &&
            maskOffset != -1) {
            AscendC::PipeBarrier<PIPE_V>();
            attenMaskUb = this->pseTBuf.template Get<uint8_t>();
        }
        if (maskOffset == -1) {
            maskOffset = this->ComputeAttenMaskOffset(extraInfo);
        }
        this->AtenmaskBoolCopyIn(attenMaskUb, this->attenMaskGmInt, maskOffset, extraInfo, this->s2Size,
                                 this->tilingData->inputParams.attenMaskS2Size);
        return;
    }
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline int64_t FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::ComputeOffsetForCausal(
    const int64_t &delta, const uint32_t &s1BaseSize, const uint32_t &s2BaseSize, const uint32_t &attenMaskS2Size)
{
    if constexpr (hasAtten == true) {
        if (delta <= 0) {
            return Min(-1 * delta, s1BaseSize);
        } else {
            return Min(delta, s2BaseSize) * attenMaskS2Size;
        }
    }
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::ComputeOffsetForPrefixRectangle(
    const int64_t &delta, const uint32_t &s2BaseSize, const uint32_t &attenMaskS2Size)
{
    if constexpr (hasAtten == true) {
        // attenMask S1 is same to S2
        if (delta <= 0) {
            return attenMaskS2Size * attenMaskS2Size + attenMaskS2Size / 2; // 2048 * 2048 + 1024
        } else if (delta > s2BaseSize) {
            return attenMaskS2Size * attenMaskS2Size; // 2048 * 2048 + 0
        } else {
            return attenMaskS2Size * attenMaskS2Size + attenMaskS2Size / 2 - delta; // 2048 * 2048 + (1024 - delta)
        }
    }
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::ComputeAttenMaskOffset(SplitBExtraInfo &extraInfo)
{
    if constexpr (hasAtten == true) {
        int64_t deltaCausalOrNext = 0;
        int64_t s1Offset = extraInfo.s1oIdx * this->s1BaseSize;
        int64_t s2Offset = 0;
        int64_t deltaN = static_cast<int64_t>(this->s1Size) - static_cast<int64_t>(this->s2Size);
        int64_t deltaPre = 0;
        if (this->attenMaskCompressMode == static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
            return this->ComputeOffsetForNoCompress(extraInfo);
        } else if (this->attenMaskCompressMode == static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE)) {
            deltaCausalOrNext = s1Offset - s2Offset;
        } else if (this->attenMaskCompressMode == static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
            deltaCausalOrNext = s1Offset - s2Offset - deltaN;
        } else if (this->attenMaskCompressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
            deltaPre = s1Offset - s2Offset - this->tilingData->inputParams.preTokens - 1;
            this->attenMaskOffsetPre = this->ComputeOffsetForCausal(deltaPre, this->s1BaseSize, this->s2Size,
                                                                    this->tilingData->inputParams.attenMaskS2Size);
            deltaCausalOrNext = s1Offset - s2Offset + this->tilingData->inputParams.nextTokens;
        } else if (this->attenMaskCompressMode == static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) {
            deltaCausalOrNext = s1Offset - s2Offset - deltaN;
            deltaPre = (this->s1Size + ((__gm__ int64_t *)this->prefixNAddr)[extraInfo.boIdx] > this->s2Size) ?
                           (((__gm__ int64_t *)this->prefixNAddr)[extraInfo.boIdx] - s2Offset) : 0;
            this->attenMaskOffsetPre =
                this->ComputeOffsetForPrefixRectangle(deltaPre, this->s2Size,
                                                      this->tilingData->inputParams.attenMaskS2Size);
            if (this->blockIdx + extraInfo.vecS1BaseSize < prefixAttenMaskDownHeight) { // in case of out of bound
                this->attenMaskOffsetPre += this->tilingData->inputParams.attenMaskS2Size * this->blockIdx;
            }
        } else {
            return 0;
        }
        return this->ComputeOffsetForCausal(deltaCausalOrNext, this->s1BaseSize, this->s2Size,
                                            this->tilingData->inputParams.attenMaskS2Size);
    }
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::ComputeOffsetForNoCompress(
    SplitBExtraInfo &extraInfo)
{
    if constexpr (hasAtten == true) {
        int64_t bOffset = 0;
        int64_t s1Offset = 0;
        int64_t biN2GOffset = 0;
        if (this->tilingData->inputParams.attenMaskShapeType == 0) { // 0: (B,N2,G,S1,S2)
            bOffset = extraInfo.boIdx * this->biN2GS1S2;
            s1Offset = extraInfo.s1oIdx * this->s1BaseSize * this->s2Size;
            biN2GOffset = extraInfo.biN2GoIdx * this->s1Size * this->s2Size;
        } else if (this->tilingData->inputParams.attenMaskShapeType == 1) { // 1: (B,1,1,S1,S2)
            bOffset = extraInfo.boIdx * this->bBaseSize * this->s1S2;
            s1Offset = extraInfo.s1oIdx * this->s1BaseSize * this->s2Size;
            biN2GOffset = extraInfo.biN2GoIdx /
                          (this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize) * this->s1Size *
                          this->s2Size;
        } else if (this->tilingData->inputParams.attenMaskShapeType == 2) { // 2: (1,1,1,s1,s2)
            s1Offset = extraInfo.s1oIdx * this->s1BaseSize * this->s2Size;
        }
        return bOffset + s1Offset + biN2GOffset;
    }
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::AtenmaskBoolCopyIn(
    LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor, int64_t offset, SplitBExtraInfo &extraInfo,
    int32_t s2Size, int64_t totalS2Size)
{
    extraInfo.s2AlignBlockSize = CeilDiv(s2Size, blockBytes) * blockBytes;
    uint32_t shapeArray[] = {static_cast<uint32_t>(extraInfo.vecS1BaseSize),
                             static_cast<uint32_t>(extraInfo.s2AlignBlockSize)};
    dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    dstTensor.SetSize(extraInfo.vecS1BaseSize * extraInfo.s2AlignBlockSize);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = extraInfo.vecS1BaseSize;
    dataCopyParams.dstStride = 0;
    if (s2Size % blockBytes == 0) {
        dataCopyParams.blockLen = extraInfo.s2AlignBlockSize / blockBytes;
        dataCopyParams.srcStride = (totalS2Size - extraInfo.s2AlignBlockSize) / blockBytes;
        DataCopy(dstTensor, srcTensor[offset], dataCopyParams);
    } else {
        dataCopyParams.blockLen = s2Size;
        dataCopyParams.srcStride = totalS2Size - s2Size;
        DataCopyPadParams dataCopyPadParams;
        dataCopyPadParams.isPad = true;
        dataCopyPadParams.rightPadding = extraInfo.s2AlignBlockSize - s2Size;
        dataCopyPadParams.paddingValue = 1;
        DataCopyPad(dstTensor, srcTensor[offset], dataCopyParams, dataCopyPadParams);
    }
}

FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::ComputeAttenMask(
    SplitBExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb, const uint8_t maskType)
{
    if constexpr (hasAtten == true) {
        LocalTensor<uint8_t> attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
        uint32_t shapeArray[] = {static_cast<uint32_t>(extraInfo.vecS1BaseSize),
                                 static_cast<uint32_t>(extraInfo.s2AlignBlockSize)};
        attenMaskUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        attenMaskUb.SetSize(extraInfo.vecS1BaseSize * extraInfo.s2AlignBlockSize);
        bmm1ResUb.SetSize(extraInfo.vecS1BaseSize * extraInfo.s2AlignSize);
        LocalTensor<uint8_t> apiTmpBuffer = commonTBuf.template Get<uint8_t>();
        SelectWithBytesMaskShapeInfo shapeInfo;
        shapeInfo.firstAxis = extraInfo.vecS1BaseSize;
        shapeInfo.srcLastAxis = extraInfo.s2AlignSize;
        shapeInfo.maskLastAxis = extraInfo.s2AlignBlockSize;

        if (maskType == 0) {
            SelectWithBytesMask(bmm1ResUb, bmm1ResUb, this->negativeFloatScalar, attenMaskUb, apiTmpBuffer, shapeInfo);
        } else {
            SelectWithBytesMask(bmm1ResUb, this->negativeFloatScalar, bmm1ResUb, attenMaskUb, apiTmpBuffer, shapeInfo);
        }
        return;
    }
}


FA_BN2GS1S2B_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreBn2gs1s2B<FA_BN2GS1S2B_FUNCTION_PARAMS_TEMPLATE>::SoftMaxCompute(
    SplitBExtraInfo &extraInfo, LocalTensor<T> &srcTensor, int64_t loopIdx)
{
    uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(extraInfo.vecS1BaseSize),
                                 static_cast<uint32_t>(extraInfo.s2AlignSize)};
    uint32_t bmm1ResUbOriShape[] = {static_cast<uint32_t>(extraInfo.vecS1BaseSize),
                                    static_cast<uint32_t>(this->s2Size)};
    srcTensor.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, 2, bmm1ResUbOriShape, DataFormat::ND));

    uint32_t maxSumShape[] = {static_cast<uint32_t>(extraInfo.vecS1BaseSize), static_cast<uint32_t>(fp32BaseSize)};
    LocalTensor<T> sumUb;
    LocalTensor<T> maxUb;
    int64_t sumOffset = (loopIdx % extraInfo.softmaxCopyOutLimit) * this->s1BaseSize * fp32BaseSize;
    if (extraInfo.boIdx % 2 == 0) {
        sumUb = this->softmaxSumPingBuf.template Get<T>()[sumOffset];
        maxUb = this->softmaxMaxPingBuf.template Get<T>()[sumOffset];
    } else {
        sumUb = this->softmaxSumPongBuf.template Get<T>()[sumOffset];
        maxUb = this->softmaxMaxPongBuf.template Get<T>()[sumOffset];
    }

    sumUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    maxUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));

    if (hasSink) {
        float inMax = sinkGm.GetValue(this->currentN1Idx);
        float inSum = 1.0;
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        Duplicate(sumUb, static_cast<T>(inSum), extraInfo.vecS1BaseSize * fp32BaseSize);
        Duplicate(maxUb, static_cast<T>(inMax), extraInfo.vecS1BaseSize * fp32BaseSize);
    }

    uint32_t expShape[] = {static_cast<uint32_t>(extraInfo.vecS1BaseSize), static_cast<uint32_t>(fp32BaseSize)};
    LocalTensor<T> expUb;
    expUb = this->softmaxExpBuf.template Get<T>()[0];
    expUb.SetShapeInfo(ShapeInfo(2, expShape, DataFormat::ND));

    LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
    AscendC::PipeBarrier<PIPE_V>();
    SoftMaxTiling softmaxFlashTilingData;
    if (IsBasicBlockInSoftMax(extraInfo.vecS1BaseSize, this->s2Size)) {
        if (hasSink){
            expUb = this->maskTBufPing.template Get<T>()[0];
            expUb.SetShapeInfo(ShapeInfo(2, expShape, DataFormat::ND));
            SoftmaxFlashV2<T, true, true, true>(srcTensor, sumUb, maxUb, srcTensor, expUb, sumUb, maxUb, apiTmpBuffer,
                                        softmaxFlashTilingData);
        }else{
            SoftmaxFlashV2<T, false, true, true>(srcTensor, sumUb, maxUb, srcTensor, expUb, sumUb, maxUb, apiTmpBuffer,
                                        softmaxFlashTilingData);
        }
    } else {
        if (hasSink){
            expUb = this->maskTBufPing.template Get<T>()[0];
            expUb.SetShapeInfo(ShapeInfo(2, expShape, DataFormat::ND));
            SoftmaxFlashV2<T, true, true, false>(srcTensor, sumUb, maxUb, srcTensor, expUb, sumUb, maxUb, apiTmpBuffer,
                                                softmaxFlashTilingData);
        }else{
            SoftmaxFlashV2<T, false, true, false>(srcTensor, sumUb, maxUb, srcTensor, expUb, sumUb, maxUb, apiTmpBuffer,
                                                softmaxFlashTilingData);
        }
    }

    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION) {
        SoftMaxShapeInfo softmaxShapeInfo{static_cast<uint32_t>(extraInfo.vecS1BaseSize),
                                          static_cast<uint32_t>(extraInfo.s2AlignSize),
                                          static_cast<uint32_t>(extraInfo.vecS1BaseSize), this->s2Size};
        AdjustSoftMaxRes<T, T>(srcTensor, maxUb, this->negativeIntScalar, 0.0, softmaxShapeInfo);
        if (loopIdx == this->s1OuterSize - 1 || (loopIdx + 1) % extraInfo.softmaxCopyOutLimit == 0) {
            LocalTensor<T> maxTensor;
            LocalTensor<T> sumTensor;
            if (extraInfo.boIdx % 2 == 0) {
                maxTensor = this->softmaxMaxPingBuf.template Get<T>();
                sumTensor = this->softmaxSumPingBuf.template Get<T>();
            } else {
                maxTensor = this->softmaxMaxPongBuf.template Get<T>();
                sumTensor = this->softmaxSumPongBuf.template Get<T>();
            }
            uint32_t currS1Size = static_cast<uint32_t>(Min(extraInfo.softmaxCopyOutSize * fp32BaseSize,
                                                            this->s1Size * fp32BaseSize - extraInfo.softmaxOutOffset));
            currS1Size = CeilDiv(currS1Size, fp32BaseSize);
            SoftMaxShapeInfo softmaxShapeInfo{currS1Size, static_cast<uint32_t>(fp32BaseSize), currS1Size,
                                              static_cast<uint32_t>(fp32BaseSize)};
            AdjustSoftMaxRes<T, T>(sumTensor, maxTensor, this->negativeIntScalar, this->positiveFloatScalar,
                                   softmaxShapeInfo);
        }
    }

    if (loopIdx == this->s1OuterSize - 1 || (loopIdx + 1) % extraInfo.softmaxCopyOutLimit == 0) {
        int64_t bOffset = extraInfo.boIdx * this->biN2GS1 * fp32BaseSize;
        int64_t biN2GOffset = extraInfo.biN2GoIdx * this->s1Size * fp32BaseSize;
        int64_t vS1Offset = extraInfo.softmaxOutOffset;

        int64_t gmOffset = bOffset + biN2GOffset + vS1Offset;
        int64_t calculateSize =
            Min(extraInfo.softmaxCopyOutSize * fp32BaseSize, this->s1Size * fp32BaseSize - extraInfo.softmaxOutOffset);
        LocalTensor<float> sumUbStart;
        LocalTensor<float> maxUbStart;
        if (extraInfo.boIdx % 2 == 0) {
            sumUbStart = this->softmaxSumPingBuf.template Get<float>()[0];
            maxUbStart = this->softmaxMaxPingBuf.template Get<float>()[0];
        } else {
            sumUbStart = this->softmaxSumPongBuf.template Get<float>()[0];
            maxUbStart = this->softmaxMaxPongBuf.template Get<float>()[0];
        }

        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopy(this->softmaxSumGm[gmOffset], sumUbStart, calculateSize);
        DataCopy(this->softmaxMaxGm[gmOffset], maxUbStart, calculateSize);
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        extraInfo.softmaxOutOffset += calculateSize;
    }
}

#endif // FLASH_ATTENTION_SCORE_B_H
