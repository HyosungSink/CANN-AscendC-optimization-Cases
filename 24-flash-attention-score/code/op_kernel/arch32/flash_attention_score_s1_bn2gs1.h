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
 * \file flash_attention_score_s1_bn2gs1.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_S1_BN2GS1_H
#define FLASH_ATTENTION_SCORE_S1_BN2GS1_H
#define FA_S1BN2GS1_FUNCTION_TEMPLATE                                                                                  \
    template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop,              \
              typename INPUT_T, typename T, bool isBasicBlock, CubeFormat bmm1Format, TPosition bmm2Source,            \
              CubeFormat bmm2SourceFormat, bool enableL1Reuse, STemplateType s1TemplateType,                           \
              STemplateType s2TemplateType, DTemplateType dTemplateType>
#define FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE                                                                           \
    implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format, bmm2Source,                 \
        bmm2SourceFormat, enableL1Reuse, s1TemplateType, s2TemplateType, dTemplateType

#include "../../common/include/kernel/util.h"
#include "../../common/include/kernel/dropmask.h"
#include "flash_attention_score_common.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "stdarg.h"
#include "../../common/include/kernel/pse.h"

using matmul::MatmulType;

struct SplitS1dExtraInfo {
    int64_t s2StartIdx;
    int64_t s2EndIdx;

    int64_t s1oIdx;
    int64_t boIdx;
    int64_t n2oIdx;
    int64_t goIdx;
    int64_t taskId;
    int8_t taskIdMod2;
    int64_t s1RealSize;
    int64_t s2RealSize;
    int64_t s2RealSizeAlign8;
    int64_t s2RealSizeAlign16;
    int64_t s2RealSizeAlign32;
    int64_t s2RealSizeAlign64 = 0;
    uint64_t duplicateMask = 0;
    int64_t s2RealSizeFloorAlign8 = 0;

    int64_t vecS1BaseSize;
    int64_t vecS1TailSize;
    int64_t vec2S1BaseSize;
    int64_t vec2S1TailSize;
    int64_t realSplitN;
    int64_t softmaxOutOffset = 0;
    int64_t s1BaseTimesS2Align16;
    int64_t s1BaseTimesS2Align64;
    int64_t qCoreOffset;
    int64_t multiCoreInnerIdx;
    int64_t softmaxCopyOutLimit;
    int64_t softmaxCopyOutS1Size;
    bool lastNotPair;
};

template <STemplateType s1TemplateType, STemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ const constexpr MatmulConfig GetS1Bn2gs1Mm1Config()
{
    constexpr uint32_t baseM = static_cast<uint32_t>(s1TemplateType);
    constexpr uint32_t baseN = static_cast<uint32_t>(s2TemplateType);
    constexpr uint32_t baseK = static_cast<uint32_t>(dTemplateType);
    return MatmulConfig{true, false, false, baseM, baseN, baseK, true};
}

template <STemplateType s1TemplateType, STemplateType s2TemplateType, DTemplateType dTemplateType>
__aicore__ const constexpr MatmulConfig GetS1Bn2gs1Mm2Config()
{
    constexpr uint32_t baseM = static_cast<uint32_t>(s1TemplateType);
    constexpr uint32_t baseN = static_cast<uint32_t>(dTemplateType);
    constexpr uint32_t baseK = static_cast<uint32_t>(s2TemplateType);
    return MatmulConfig{true, false, false, baseM, baseN, baseK, true};
}

// INPUT_T - means data type for input
// T       - means data type when calc
template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T = INPUT_T, bool isBasicBlock = false, CubeFormat bmm1Format = CubeFormat::ND,
          TPosition bmm2Source = TPosition::GM, CubeFormat bmm2SourceFormat = CubeFormat::ND,
          bool enableL1Reuse = false, STemplateType s1TemplateType = STemplateType::S_TEMPLATE_UNKNOW,
          STemplateType s2TemplateType = STemplateType::S_TEMPLATE_UNKNOW,
          DTemplateType dTemplateType = DTemplateType::D_TEMPLATE_UNKNOW>
class FlashAttentionScoreS1Bn2gs1 {
public:
    __aicore__ inline FlashAttentionScoreS1Bn2gs1(){};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                                __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *prefix,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *sink, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // define matmul
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true, LayoutMode::NONE, enableL1Reuse>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c1Type = MatmulType<TPosition::GM, bmm1Format, T>;
    constexpr static MatmulConfig mm1Config = GetS1Bn2gs1Mm1Config<s1TemplateType, s2TemplateType, dTemplateType>();
    constexpr static MatmulApiStaticTiling stcMm1Tiling =
        matmul::GetMatmulApiTiling<a1Type, b1Type, c1Type, bias1Type>(mm1Config);
    using modeTypeMm1 =
        typename AscendC::Conditional<(s1TemplateType == STemplateType::S_TEMPLATE_UNKNOW),
                                      matmul::Matmul<a1Type, b1Type, c1Type, bias1Type, GetMmCfg(enableL1Reuse)>,
                                      matmul::Matmul<a1Type, b1Type, c1Type, bias1Type, stcMm1Tiling>>::type;
    modeTypeMm1 bmm1;

    // define batchmatmul
    using a2Type = typename AscendC::Conditional<(IsSameType<T, INPUT_T>::value == true),
                                                 MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>,
                                                 MatmulType<TPosition::GM, bmm1Format, INPUT_T>>::type;
    using b2Type = MatmulType<bmm2Source, bmm2SourceFormat, INPUT_T, false, LayoutMode::NONE, enableL1Reuse>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c2Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    constexpr static MatmulConfig mm2Config = GetS1Bn2gs1Mm2Config<s1TemplateType, s2TemplateType, dTemplateType>();
    constexpr static MatmulApiStaticTiling stcMm2Tiling =
        matmul::GetMatmulApiTiling<a2Type, b2Type, c2Type, bias2Type>(mm2Config);
    using modeTypeMm2 =
        typename AscendC::Conditional<(s1TemplateType == STemplateType::S_TEMPLATE_UNKNOW),
                                      matmul::Matmul<a2Type, b2Type, c2Type, bias2Type, GetMmCfg(enableL1Reuse)>,
                                      matmul::Matmul<a2Type, b2Type, c2Type, bias2Type, stcMm2Tiling>>::type;
    modeTypeMm2 bmm2;

    using c2NzType = MatmulType<TPosition::GM, CubeFormat::NZ, float>;
    constexpr static MatmulApiStaticTiling stcMm2NzTiling =
        matmul::GetMatmulApiTiling<a2Type, b2Type, c2NzType, bias2Type>(mm2Config);
    using modeTypeMm2Nz =
        typename AscendC::Conditional<(s1TemplateType == STemplateType::S_TEMPLATE_UNKNOW),
                                      matmul::Matmul<a2Type, b2Type, c2NzType, bias2Type, GetMmCfg(enableL1Reuse)>,
                                      matmul::Matmul<a2Type, b2Type, c2NzType, bias2Type, stcMm2NzTiling>>::type;
    modeTypeMm2Nz bmm2Nz;

protected:
    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                     __gm__ uint8_t *pse, __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask,
                                     __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, __gm__ uint8_t *sink, __gm__ uint8_t *softmaxMax,
                                     __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut,
                                     __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                     const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void WaitBmm1Result();
    template <typename T2, typename T3, const auto &TILING>
    __aicore__ inline void WaitBmm2Result(matmul::Matmul<T2, b2Type, T3, bias2Type, TILING> &bmm2);
    template <typename T2, typename T3, const auto &TILING>
    __aicore__ inline void IterateBmm2(SplitS1dExtraInfo &extraInfo,
                                       matmul::Matmul<T2, b2Type, T3, bias2Type, TILING> &bmm2);
    __aicore__ inline void SetExtraInfo(SplitS1dExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount,
                                        int64_t s2LoopLimit, int64_t multiCoreInnerIdx, bool lastNotPair);
    __aicore__ inline void SetTiling(const FlashAttentionScoreGeneralTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);
    __aicore__ inline void IterateBmm1(SplitS1dExtraInfo &extraInfo);
    __aicore__ inline void Bmm1SetTensorA(SplitS1dExtraInfo &extraInfo);
    __aicore__ inline void ComputeBmm1Tail(SplitS1dExtraInfo &extraInfo);
    __aicore__ inline void SetBmm1TensorB(SplitS1dExtraInfo &extraInfo);
    __aicore__ inline void Bmm2ResultDiv(SplitS1dExtraInfo &extraInfo, int64_t vec2S1Idx, LocalTensor<T> &bmm2Res,
                                         LocalTensor<T> &sumTensor, int32_t vec2S1RealSize);
    __aicore__ inline void LoopDivVec2(LocalTensor<T> &bmm2Res, LocalTensor<T> &sumTensor, int32_t s1OuterLoop,
                                       int32_t s1OuterRemain, BinaryRepeatParams repeatParams);
    __aicore__ inline void Bmm2DataCopyOut(SplitS1dExtraInfo &extraInfo, int64_t vec2S1Idx, int64_t divCastCalcSize,
                                           LocalTensor<T> &bmm2Res, LocalTensor<INPUT_T> &attentionOut);
    __aicore__ inline void ProcessVec2(SplitS1dExtraInfo &extraInfo);
    __aicore__ inline void ProcessVec1(SplitS1dExtraInfo &extraInfo);
    __aicore__ inline void CopyInAttenMask(SplitS1dExtraInfo &extraInfo, int64_t loopIdx, int64_t maskOffset);
    __aicore__ inline int64_t ComputeAttenMaskOffset(SplitS1dExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline int64_t ComputeOffsetForNoCompress(SplitS1dExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline void NdToNz(SplitS1dExtraInfo &extraInfo, LocalTensor<INPUT_T> &nzResUb,
                                  LocalTensor<INPUT_T> &vec1ResUb, int64_t loopIdx);
    __aicore__ inline void GetBmm1Result(SplitS1dExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb, int64_t loopIdx);
    __aicore__ inline void ComputeAttenMask(SplitS1dExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb,
                                            const uint8_t maskType);
    __aicore__ inline void SoftMaxCompute(SplitS1dExtraInfo &extraInfo, LocalTensor<T> &srcTensor, int64_t loopIdx);
    __aicore__ inline void Bmm2SetTensor();
    __aicore__ inline void CopyToL1(const GlobalTensor<INPUT_T> &gmTensor, LocalTensor<INPUT_T> &localTensor,
                                    int64_t row, int64_t column, uint16_t srcDValue);

    uint32_t s1BaseSize;
    uint32_t s2BaseSize;
    uint32_t dSize;
    int64_t dSizeAlign16;
    // sparse 用函数
    __aicore__ inline void GetS1LoopRange(int64_t &multiCoreInnerOffset, int64_t &multiCoreInnerLimit);
    __aicore__ inline void GetS2LoopRange();

    __aicore__ inline int64_t ComputeOffsetForCausal(const int64_t &delta, const uint32_t &s1BaseSize,
                                                     const uint32_t &s2BaseSize, const uint32_t &attenMaskS2Size);
    __aicore__ inline int64_t ComputeOffsetForPrefixRectangle(const int64_t &delta, const uint32_t &s2BaseSize,
                                                              const uint32_t &attenMaskS2Size);
    __aicore__ inline int32_t Align(int32_t shape);

    // sparse 用参数
    int64_t s2StartIdx;
    int64_t s2EndIdx;

    int64_t qCoreOffset;
    int64_t mm1Ka;
    int64_t mm1Kb;
    int64_t mm2Kb;
    int64_t srcDValue;

    int64_t bmm2BOffsetLast = 0;
    int64_t bmm2N2OffsetLast = 0;
    int64_t lastVec1S1RealSize = 0;
    int64_t lastVec2S1RealSize = 0;

    // 资源分配
    TBuf<> maskTBufPing;  // 11K
    TBuf<> maskTBufPong;  // 11K
    TBuf<> pseTBuf;       // 16K
    TBuf<> stage1PingBuf; // 32K
    TBuf<> stage1PongBuf; // 32K
    TBuf<> vecOut;        // 16K
    TBuf<> softmaxSumBuf; // 8K
    TBuf<> softmaxMaxBuf; // 8K

    TBuf<> softmaxExpBuf; // 8K 这个buff并不使用，只是为了传递给softmaxV2接口

    TBuf<> commonTBuf; // 32K common的复用空间

    GlobalTensor<T> mm1Res[2];
    GlobalTensor<INPUT_T> stage1Res[2];
    GlobalTensor<half> pseAlibiGm;
    GlobalTensor<T> mm2Res[2];
    TSCM<TPosition::GM, 1> valueScm;
    LocalTensor<INPUT_T> valueScmTensor;

    // 轴的乘积
    int64_t gS1o;
    int64_t n2GS1o;
    int64_t s1D;
    int64_t gS1D;
    int64_t n2GS1D;
    int64_t s2D;
    int64_t n2S2D;
    int64_t s1S2;
    int64_t gS1S2;
    int64_t n2GS1S2;
    int64_t gS1;
    int64_t n2GS1;
    int64_t gD;
    int64_t n2D;
    int64_t bN2D;
    int64_t n2G;
    int64_t n2GD;
    int64_t bN2GD;
    int64_t gS2;

    // s2base * N之后的长度
    int64_t s1BaseS2;
    bool hasSink;

    int64_t s1BaseN2GD;
    int64_t s1BaseBN2GD;
    int64_t s1BaseD;
    int64_t bN2G;
    int64_t n2GS2;

    int64_t softmaxBufSize = 256;
    uint32_t negativeIntScalar = NEGATIVE_MIN_VAULE_FP32;
    T negativeFloatScalar;
    T positiveFloatScalar;

    AttenMaskComputeMode attenMaskComputeMode = AttenMaskComputeMode::NORMAL_MODE;

    int32_t blockIdx;

    const FlashAttentionScoreGeneralTilingData *__restrict tilingData;

    int64_t boIdx;
    int64_t n2oIdx;
    int64_t goIdx;
    int64_t s1oIdx;

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

    // 资源分配

    constexpr static int32_t blockBytes = 32;
    constexpr static int32_t byteBitRatio = 8;
    constexpr static int32_t fp32BaseSize = 8;
    constexpr static int32_t repeatMaxBytes = 256;
    constexpr static int32_t repeatMaxTimes = 255;

    bool dropMaskUnAligned;
    int32_t repeatMaxSize;

    // 0级接口的block间隔范围需要满足32B对齐
    constexpr static int32_t blockSize = blockBytes / sizeof(T);

    int64_t attenMaskBN2GS1S2 = 0;
    int64_t attenMaskBS1S2 = 1;
    int64_t attenMaskS1S2 = 2;
    int64_t attenMaskTT = 99;
    int64_t attenMaskOffsetPre = 0;

    PseInfo pseInfo = {0};

    DropMaskInfo dropMaskInfo = {0};
};

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, __gm__ uint8_t *sink, __gm__ uint8_t *softmaxMax,
    __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe)
{
    this->InitInput(query, key, value, pse, dropMask, paddingMask, prefix, attenMask, sink, softmaxMax, softmaxSum,
                    softmaxOut, attentionOut, workspace, tiling, tPipe); // gm设置

    this->ComputeConstexpr();
    this->InitBuffer();
    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    DropOutBitModeInit(apiTmpBuffer);
    if (this->blockIdx < this->tilingData->multiCoreParams.coreNum) {
        LocalTensor<half> pseHelpBuffer = this->stage1PingBuf.template Get<half>();
        PseInnerAlibiCreate<hasPse>(this->pseAlibiGm, pseHelpBuffer, this->pseInfo);
    }
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::InitInput(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, __gm__ uint8_t *sink, __gm__ uint8_t *softmaxMax,
    __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe)
{
    this->blockIdx = GetBlockIdx();
    this->pipe = tPipe;
    this->repeatMaxSize = this->repeatMaxBytes / sizeof(T);
    this->SetTiling(tiling);

    // init global buffer
    this->queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    this->pseGm.SetGlobalBuffer((__gm__ INPUT_T *)pse);
    this->pseSlope = pse;
    this->prefixNAddr = prefix;

    this->dropMaskUnAligned = this->tilingData->inputParams.needDropMaskOp == 1;
    if (this->dropMaskUnAligned) {
        this->dropMaskGm.SetGlobalBuffer(workspace);
        if constexpr (hasDrop == true) {
            workspace += CeilDiv(this->tilingData->dropmaskParams.shapeTotalSize, 512) * 512;
        }
    } else {
        this->dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)dropMask);
    }
    this->attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
    this->softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    this->softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    this->sinkGm.SetGlobalBuffer((__gm__ float *)sink);
    this->attentionOutGm.SetGlobalBuffer((__gm__ INPUT_T *)attentionOut);

    // 补齐到512， 统一按T处理
    // S1BaseSize是配比之后的S1BaseSize值
    int64_t mmNRatioOffset = 0;
    int64_t stage1WorkspaceOffset = 0;
    int64_t s2SizeAlign16 = CeilDiv(this->tilingData->inputParams.s2Size, 16) * 16;
    mmNRatioOffset = CeilDiv(s1BaseSize * s2SizeAlign16, 256) * 256 * sizeof(T);
    stage1WorkspaceOffset = mmNRatioOffset;

    // FP32下的稀疏场景bmm1与stage1的workspace不复用
    if constexpr (IsSameType<INPUT_T, float>::value) {
        stage1WorkspaceOffset = mmNRatioOffset * sizeof(INPUT_T) / 2;
    }

    // d一定会对齐，这里考虑开乒乓，申请两份workspace
    int64_t bmm2ResultOffset = CeilDiv(s1BaseSize * this->dSizeAlign16, 256) * 256 * sizeof(T);
    int64_t pseInnerAlibiSize = this->tilingData->coreParams.pseAlibiBaseS1 * this->tilingData->coreParams.pseAlibiBaseS2 * sizeof(half);
    int64_t pseAlibiOffset =  CeilDiv(pseInnerAlibiSize, 512) * 512;
    int64_t totalOffset = this->blockIdx * (mmNRatioOffset * 2 + stage1WorkspaceOffset + 2 * bmm2ResultOffset + pseAlibiOffset);

    // bmm1Result， 占用mmNRatioOffset空间
    this->mm1Res[0].SetGlobalBuffer((__gm__ T *)(workspace + totalOffset));
    this->mm1Res[1].SetGlobalBuffer((__gm__ T *)(workspace + totalOffset + mmNRatioOffset));
    // stage1Result， 占用mmNRatioOffset空间
    this->stage1Res[0].SetGlobalBuffer((__gm__ INPUT_T *)(workspace + totalOffset + 2 * mmNRatioOffset));
    this->stage1Res[1].SetGlobalBuffer(
        (__gm__ INPUT_T *)(workspace + totalOffset + 2 * mmNRatioOffset + stage1WorkspaceOffset / 2));

    // bmm2Result, 占用s1BaseSize * D的空间
    this->mm2Res[0].SetGlobalBuffer((__gm__ T *)(workspace + totalOffset + mmNRatioOffset * 2 + stage1WorkspaceOffset));
    this->mm2Res[1].SetGlobalBuffer(
        (__gm__ T *)(workspace + totalOffset + mmNRatioOffset * 2 + stage1WorkspaceOffset + bmm2ResultOffset));

    uint64_t pseAlibiAddr = totalOffset + mmNRatioOffset * 2 + stage1WorkspaceOffset + 2 * bmm2ResultOffset;
    this->pseAlibiGm.SetGlobalBuffer((__gm__ half*)(workspace + pseAlibiAddr));

    if constexpr (IsSameType<T, half>::value) {
        this->negativeIntScalar = NEGATIVE_MIN_VAULE_FP16;
    }
    GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
    this->hasSink = this->tilingData->inputParams.needSinkOp;
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline int64_t FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::ComputeOffsetForCausal(
    const int64_t &delta, const uint32_t &s1BaseSize, const uint32_t &s2BaseSize, const uint32_t &attenMaskS2Size)
{
    if (delta <= 0) {
        return Min(-1 * delta, s1BaseSize);
    } else {
        return Min(delta, s2BaseSize) * attenMaskS2Size;
    }
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::ComputeOffsetForPrefixRectangle(
    const int64_t &delta, const uint32_t &s2BaseSize, const uint32_t &attenMaskS2Size)
{
    // attenMask S1 is same to S2
    if (delta <= 0) {
        return attenMaskS2Size * attenMaskS2Size + attenMaskS2Size / 2; // 2048 * 2048 + 1024
    } else if (delta > s2BaseSize) {
        return attenMaskS2Size * attenMaskS2Size; // 2048 * 2048 + 0
    } else {
        return attenMaskS2Size * attenMaskS2Size + attenMaskS2Size / 2 - delta + 1; // 2048 * 2048 + (1024 - delta)
    }
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline int32_t FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::Align(int32_t shape)
{
    int32_t alignFactor = 16;
    int32_t alignedSize = CeilDiv(shape, alignFactor) * alignFactor;
    return alignedSize;
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::SetTiling(
    const FlashAttentionScoreGeneralTilingData *__restrict tilingData)
{
    // copy base params
    this->tilingData = tilingData;
    this->s1BaseSize = this->tilingData->coreParams.s1BaseSize;
    this->s2BaseSize = this->tilingData->coreParams.s2BaseSize;
    this->dSize = this->tilingData->inputParams.dSize;
    this->dSizeAlign16 = CeilDiv(this->tilingData->inputParams.dSize, 16) * 16;
};

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::InitBuffer()
{
    // 按可用UB空间192k计算，当S2取1024时，S1最大取8，因此UB空间计算的基本块取8 * 1024
    // 8 * 1024 * 1 = 8k，但由于非对齐pad需要额外空间，因此设置11k
    this->pipe->InitBuffer(this->maskTBufPing, 11264); // attenmask 11k
    this->pipe->InitBuffer(this->maskTBufPong, 11264); // dropoutmask 11k

    // 8 * 1024 * 2 = 16k
    this->pipe->InitBuffer(this->pseTBuf, 16384); // pse 16k

    // 8 * 1024 * 4 = 32k
    this->pipe->InitBuffer(this->stage1PingBuf, 8192 * sizeof(T)); // t.a 32k
    this->pipe->InitBuffer(this->stage1PongBuf, 9216 * sizeof(T)); // i.a 36k
    this->pipe->InitBuffer(this->commonTBuf, 8192 * sizeof(T));    // t.b 32k

    // 8 * 1024 * 2 = 16k，存放vec1 Cast成fp16之后的结果, 支持NdToNz后需要增加2K防止bank冲突
    this->pipe->InitBuffer(this->vecOut, 18432);
    // 根据UB剩余空间大小，分配sum和max各8k，无ping pong，exp额外占用32B，共16KB + 32B
    this->pipe->InitBuffer(this->softmaxSumBuf, this->softmaxBufSize * this->blockBytes);
    this->pipe->InitBuffer(this->softmaxMaxBuf, this->softmaxBufSize * this->blockBytes);
    this->pipe->InitBuffer(this->softmaxExpBuf, this->blockBytes);
    if constexpr (bmm2Source == TPosition::TSCM) {
        this->pipe->InitBuffer(valueScm, 1, this->tilingData->inputParams.alignedS2 * dSize * sizeof(INPUT_T));
    }
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::ComputeConstexpr()
{
    // 计算轴的乘积
    this->s1D = this->tilingData->inputParams.s1Size * dSize;
    this->s2D = this->tilingData->inputParams.s2Size * dSize;
    this->gD = this->tilingData->inputParams.gSize * dSize;
    this->n2D = this->tilingData->inputParams.n2Size * dSize;
    this->s1S2 = this->tilingData->inputParams.s1Size * this->tilingData->inputParams.s2Size;
    this->gS1 = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s1Size;
    this->n2G = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize;
    this->gS1o = this->tilingData->inputParams.gSize * this->tilingData->coreParams.s1OuterSize;

    this->bN2D = this->tilingData->inputParams.bSize * n2D;
    this->n2GS1o = this->tilingData->inputParams.n2Size * this->gS1o;
    this->gS1D = this->tilingData->inputParams.gSize * this->s1D;
    this->n2S2D = this->tilingData->inputParams.n2Size * this->s2D;
    this->n2GD = this->tilingData->inputParams.n2Size * this->gD;
    this->bN2GD = this->tilingData->inputParams.bSize * n2GD;
    this->gS1S2 = this->tilingData->inputParams.gSize * this->s1S2;
    this->n2GS1 = this->tilingData->inputParams.n2Size * this->gS1;

    this->n2GS1D = this->tilingData->inputParams.n2Size * this->gS1D;
    this->n2GS1S2 = this->tilingData->inputParams.n2Size * this->gS1S2;

    // 计算切分轴的乘积
    this->s1BaseS2 = s1BaseSize * this->tilingData->inputParams.s2Size;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        // BSH/BSNGD
        this->s1BaseN2GD = s1BaseSize * this->n2GD;
        mm1Ka = n2GD;
        mm1Kb = n2D;
        mm2Kb = this->n2D;
        srcDValue = n2D;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        // SBH/SBNGD
        this->bN2G = this->tilingData->inputParams.bSize * this->n2G;
        mm1Ka = bN2GD;
        mm1Kb = bN2D;
        mm2Kb = this->bN2D;
        srcDValue = bN2D;
        this->s1BaseBN2GD = s1BaseSize * this->tilingData->inputParams.bSize * this->n2GD;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        // BNSD
        this->s1BaseD = s1BaseSize * dSize;
        mm1Ka = dSize;
        mm1Kb = dSize;
        mm2Kb = dSize;
        srcDValue = dSize;
    }

    if (this->tilingData->inputParams.pseShapeType == pse1S2) {
        this->gS2 = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s2Size;
        this->n2GS2 = this->tilingData->inputParams.n2Size * this->gS2;
    }
    if constexpr (hasPse == true) {
        this->pseInfo.s2Size = this->tilingData->inputParams.s2Size;
        this->pseInfo.s1Size = this->tilingData->inputParams.s1Size;
        this->pseInfo.gSize = this->tilingData->inputParams.gSize;
        this->pseInfo.pseShapeType = this->tilingData->inputParams.pseShapeType;
        this->pseInfo.n2G = this->n2G;
        this->pseInfo.pseBSize = this->tilingData->inputParams.pseBSize;
        this->pseInfo.s1BaseSize = this->s1BaseSize;
        this->pseInfo.pseType = this->tilingData->inputParams.pseType;
        this->pseInfo.pseAlibiBaseS1 = this->tilingData->coreParams.pseAlibiBaseS1;
        this->pseInfo.pseAlibiBaseS2 = this->tilingData->coreParams.pseAlibiBaseS2;
        this->pseInfo.qStartIdx = this->tilingData->inputParams.qStartIdx;
        this->pseInfo.kvStartIdx = this->tilingData->inputParams.kvStartIdx;
    }
    if constexpr (hasDrop == true) {
        this->dropMaskInfo.s2Size = this->tilingData->inputParams.s2Size;
        this->dropMaskInfo.s1Size = this->tilingData->inputParams.s1Size;
        this->dropMaskInfo.gSize = this->tilingData->inputParams.gSize;
        this->dropMaskInfo.n2G = this->n2G;
        this->dropMaskInfo.s1BaseSize = this->s1BaseSize;
    }
}

// sparse functions
FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::GetS1LoopRange(int64_t &multiCoreInnerOffset,
                                                                                  int64_t &multiCoreInnerLimit)
{
    // 计算sparse场景下s1的循环范围
    if (this->tilingData->inputParams.sparseType > 0) {
        if constexpr (enableL1Reuse) {
            // sparse场景下负载均衡后每个核获取的结果
            if (this->blockIdx % 2 == 0) {
                multiCoreInnerOffset = this->tilingData->multiCoreParams.sparseStartIdx[this->blockIdx];
                if (likely((this->tilingData->multiCoreParams.coreNum - 2) > this->blockIdx)) {
                    multiCoreInnerLimit = this->tilingData->multiCoreParams.sparseStartIdx[this->blockIdx + 2];
                } else {
                    multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
                }
            } else {
                multiCoreInnerOffset = this->tilingData->multiCoreParams.sparseStartIdx[this->blockIdx - 1];
                if (likely((this->tilingData->multiCoreParams.coreNum - 1) > this->blockIdx)) {
                    multiCoreInnerLimit = this->tilingData->multiCoreParams.sparseStartIdx[this->blockIdx + 1];
                } else {
                    multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
                }
            }
        } else {
            // sparse场景下负载均衡后每个核获取的结果
            multiCoreInnerOffset = this->tilingData->multiCoreParams.sparseStartIdx[this->blockIdx];
            if (likely((this->tilingData->multiCoreParams.coreNum - 1) > this->blockIdx)) {
                multiCoreInnerLimit = this->tilingData->multiCoreParams.sparseStartIdx[this->blockIdx + 1];
            } else {
                multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
            }
        }
    }
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::GetS2LoopRange()
{
    // 计算S2的循环范围相关参数: 后续可 使用static_cast<uint32_t>优化scale性能
    if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::CAUSAL)) { // 下三角
        this->s2StartIdx = 0;
        this->s2EndIdx = Min((this->s1oIdx + 1) * s1BaseSize, this->tilingData->inputParams.s2Size);
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND)) {
        // 对角线往外扩散场景, s1和s2可能不同
        // 如果存在bit模式的dropmask输入，则s2StartIdx需要向下8对齐
        if (hasDrop && !this->tilingData->inputParams.needDropMaskOp) {
            this->s2StartIdx = Max(this->s1oIdx * this->tilingData->coreParams.s1BaseSize -
                                       this->tilingData->coreParams.s1SparseValidSize,
                                   0) /
                               8 * 8;
        } else {
            this->s2StartIdx = Max(this->s1oIdx * this->tilingData->coreParams.s1BaseSize -
                                       this->tilingData->coreParams.s1SparseValidSize,
                                   0);
        }
        this->s2EndIdx = Min((this->s1oIdx + 1) * s1BaseSize + this->tilingData->coreParams.s2SparseValidSize,
                             this->tilingData->inputParams.s2Size);
        // s1baseSize行都无效时, 将startIdx设置为0, endIdx设置为S2Size
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min(this->tilingData->inputParams.s2Size, 128L);
        }
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::PREFIX)) {
        this->s2StartIdx = 0;
        this->s2EndIdx = Max(s1BaseSize * (this->s1oIdx + 1) - this->tilingData->inputParams.s1Size +
                                 this->tilingData->inputParams.s2Size,
                             ((__gm__ int64_t *)this->prefixNAddr)[this->boIdx]);
        this->s2EndIdx = CeilDiv(this->s2EndIdx, s2BaseSize) * s2BaseSize;
        this->s2EndIdx = Min(this->s2EndIdx, this->tilingData->inputParams.s2Size);
        // s1baseSize行都无效时, 将startIdx设置为0, endIdx设置为S2Size
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min(this->tilingData->inputParams.s2Size, 128L);
        }
    } else { // 其它场景, 如无attention mask
        this->s2StartIdx = 0;
        this->s2EndIdx = this->tilingData->inputParams.s2Size;
    }
    return;
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::Bmm2ResultDiv(
    SplitS1dExtraInfo &extraInfo, int64_t vec2S1Idx, LocalTensor<T> &bmm2Res, LocalTensor<T> &sumTensor,
    int32_t vec2S1RealSize)
{
    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src0RepStride = this->dSizeAlign16 / this->blockSize;
    repeatParams.src1BlkStride = 0;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = this->dSizeAlign16 / this->blockSize;
    int32_t s1OuterLoop = vec2S1RealSize / this->repeatMaxTimes;
    int32_t s1OuterRemain = vec2S1RealSize % this->repeatMaxTimes;

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
        LoopDivVec2(bmm2Res, sumCastTensor, s1OuterLoop, s1OuterRemain, repeatParams);
    } else {
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        LoopDivVec2(bmm2Res, sumTensor, s1OuterLoop, s1OuterRemain, repeatParams);
    }
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::LoopDivVec2(
    LocalTensor<T> &bmm2Res, LocalTensor<T> &sumTensor, int32_t s1OuterLoop, int32_t s1OuterRemain,
    BinaryRepeatParams repeatParams)
{
    int64_t bmm2LoopMaxSize = this->repeatMaxTimes * this->dSizeAlign16;
    int64_t softmaxSumMaxSize = this->repeatMaxTimes * 8;
    int32_t innerLoop = this->dSizeAlign16 / this->repeatMaxSize;
    int32_t innerRemain = this->dSizeAlign16 % this->repeatMaxSize;
    LocalTensor<T> bmm2ResPongUb = this->stage1PongBuf.template Get<T>();
    for (int32_t i = 0; i < s1OuterLoop; ++i) {
        auto bmm2ResTmp = bmm2Res[i * bmm2LoopMaxSize];
        auto bmm2ResDstTmp = bmm2ResPongUb[i * bmm2LoopMaxSize];
        for (int32_t j = 0; j < innerLoop; ++j) {
            Div(bmm2ResDstTmp[j * this->repeatMaxSize], bmm2ResTmp[j * this->repeatMaxSize],
                sumTensor[i * softmaxSumMaxSize], this->repeatMaxSize, this->repeatMaxTimes, repeatParams);
        }
        if (likely(innerRemain)) {
            Div(bmm2ResDstTmp[innerLoop * this->repeatMaxSize], bmm2ResTmp[innerLoop * this->repeatMaxSize],
                sumTensor[i * softmaxSumMaxSize], innerRemain, this->repeatMaxTimes, repeatParams);
        }
    }
    if (likely(s1OuterRemain)) {
        auto bmm2ResTmp = bmm2Res[s1OuterLoop * bmm2LoopMaxSize];
        auto bmm2ResDstRemain = bmm2ResPongUb[s1OuterLoop * bmm2LoopMaxSize];
        for (int32_t j = 0; j < innerLoop; ++j) {
            Div(bmm2ResDstRemain[j * this->repeatMaxSize], bmm2ResTmp[j * this->repeatMaxSize],
                sumTensor[s1OuterLoop * softmaxSumMaxSize], this->repeatMaxSize, s1OuterRemain, repeatParams);
        }
        if (likely(innerRemain)) {
            Div(bmm2ResDstRemain[innerLoop * this->repeatMaxSize], bmm2ResTmp[innerLoop * this->repeatMaxSize],
                sumTensor[s1OuterLoop * softmaxSumMaxSize], innerRemain, s1OuterRemain, repeatParams);
        }
    }
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::Bmm2DataCopyOut(
    SplitS1dExtraInfo &extraInfo, int64_t vec2S1Idx, int64_t divCastCalcSize, LocalTensor<T> &bmm2Res,
    LocalTensor<INPUT_T> &attentionOut)
{
    int32_t calcSize = bmm2Res.GetSize();
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
    dataCopyParams.blockLen = dSize * sizeof(INPUT_T);
    dataCopyParams.srcStride = 0;
    if constexpr (IsSameType<INPUT_T, float>::value) {
        if (this->dSizeAlign16 - this->dSize >= this->blockSize) {
            dataCopyParams.srcStride = 1;
        }
    }
    int64_t dstStride = 0;
    int64_t attenOutOffset = dSize;
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        attenOutOffset = this->n2GD;
        dstStride = (this->n2G - 1) * dataCopyParams.blockLen;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        attenOutOffset = this->bN2GD;
        dstStride = (this->bN2G - 1) * dataCopyParams.blockLen;
    }

    // dataCopyParams.dstStride类型定义uint16_t，65535是其最大值
    int64_t gmOffset = extraInfo.qCoreOffset + vec2S1Idx * extraInfo.vec2S1BaseSize * attenOutOffset;
    if (likely(dstStride <= 65535)) {
        dataCopyParams.blockCount = divCastCalcSize;
        dataCopyParams.dstStride = static_cast<uint16_t>(dstStride);
        DataCopyPad(this->attentionOutGm[gmOffset], attentionOut, dataCopyParams);
    } else {
        dataCopyParams.blockCount = 1;
        dataCopyParams.dstStride = 0;
        int64_t datacopyOffset = dSize;
        if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
            datacopyOffset = this->n2GD;
        } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
            datacopyOffset = this->bN2G * dSize;
        }

        for (uint32_t i = 0; i < divCastCalcSize; ++i) {
            DataCopyPad(this->attentionOutGm[gmOffset + i * datacopyOffset], attentionOut[i * this->dSizeAlign16], dataCopyParams);
        }
    }
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::ProcessVec2(SplitS1dExtraInfo &extraInfo)
{
    if constexpr (enableL1Reuse) {
        if (extraInfo.lastNotPair) {
            return;
        }
    }

    // 获取缓存bmm2的计算结果
    LocalTensor<T> bmm2ResPingUb = this->stage1PingBuf.template Get<T>();
    LocalTensor<T> bmm2ResPongUb = this->stage1PongBuf.template Get<T>();

    LocalTensor<T> softmaxSumPingUb = this->pseTBuf.template Get<float>();

    // 使用common buf作为最后的输出的ping和pong，可以在common buf往外搬出的时候，用pseBuf向内搬运
    // 最后的输出结果是16KB，fp16，ping和pong分别是8192个fp32数
    LocalTensor<INPUT_T> attentionOutUb[2];
    attentionOutUb[0] = this->commonTBuf.template Get<INPUT_T>();
    if constexpr (IsSameType<INPUT_T, float>::value) {
        attentionOutUb[1] = this->commonTBuf.template Get<INPUT_T>()[4096];
    } else {
        attentionOutUb[1] = this->commonTBuf.template Get<INPUT_T>()[8192];
    }
    int64_t vec2LoopLimit = CeilDiv(extraInfo.s1RealSize, extraInfo.vec2S1BaseSize);
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    int64_t bOffset = extraInfo.boIdx * this->n2GS1 * this->fp32BaseSize;
    int64_t n2Offset = extraInfo.n2oIdx * this->gS1 * this->fp32BaseSize;
    int64_t gOffset = extraInfo.goIdx * this->tilingData->inputParams.s1Size * this->fp32BaseSize;
    int64_t s1Offset = extraInfo.s1oIdx * s1BaseSize * this->fp32BaseSize;
    int64_t sumGmOffset = bOffset + n2Offset + gOffset + s1Offset;
    int64_t vec2S1DNormalSize = extraInfo.vec2S1BaseSize * this->dSize;
    int64_t vec2S1DNormalSize1 = extraInfo.vec2S1BaseSize * this->dSizeAlign16;
    int64_t totalSize = extraInfo.s1RealSize * this->dSize;
    int64_t totalSize1 = extraInfo.s1RealSize * this->dSizeAlign16;
    for (int64_t vec2S1Idx = 0; vec2S1Idx < vec2LoopLimit; ++vec2S1Idx) {
        int64_t divCastCalcSize = 0;
        int64_t divCastCalcSize1 = 0;
        int64_t mm2ResOffset = 0;
        if (extraInfo.s1RealSize < extraInfo.vec2S1BaseSize) {
            divCastCalcSize = totalSize;
            divCastCalcSize1 = extraInfo.s1RealSize * this->dSizeAlign16;
            mm2ResOffset = 0;
        } else {
            mm2ResOffset = vec2S1Idx * vec2S1DNormalSize;
            divCastCalcSize = Min(vec2S1DNormalSize, totalSize - mm2ResOffset);
            divCastCalcSize1 = Min(vec2S1DNormalSize1, totalSize1 - vec2S1Idx * vec2S1DNormalSize1);
        }

        int64_t s1RealSizeInLoop = divCastCalcSize1 / this->dSizeAlign16;
        int64_t sumInnerSize = s1RealSizeInLoop * this->fp32BaseSize;
        int64_t sumGmOffsetLoop = sumGmOffset + vec2S1Idx * extraInfo.vec2S1BaseSize * this->fp32BaseSize;
        int64_t dAlign8 = (this->dSize + 7) / 8 * 8;
        if (vec2S1Idx > 0) {
            AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        }
        if (likely(this->dSizeAlign16 == this->dSize)) {
            DataCopy(bmm2ResPingUb, this->mm2Res[extraInfo.taskIdMod2][mm2ResOffset], divCastCalcSize);
        } else {
            Nz2NdInfo nz2NdInfo;
            nz2NdInfo.ndFirstAxisRealSize = extraInfo.s1RealSize;
            nz2NdInfo.ndFirstAxisBaseSize = extraInfo.vec2S1BaseSize;
            nz2NdInfo.ndFirstAxisLoopSize = s1RealSizeInLoop;
            nz2NdInfo.ndLastAxis = this->dSizeAlign16;
            nz2NdInfo.loopIdx = vec2S1Idx;
            event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
            AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
            NzToNd(nz2NdInfo, this->mm2Res[extraInfo.taskIdMod2], bmm2ResPongUb, bmm2ResPingUb);
            divCastCalcSize = divCastCalcSize1;
        }

        DataCopy(softmaxSumPingUb, this->softmaxSumGm[sumGmOffsetLoop], sumInnerSize);
        bmm2ResPongUb.SetSize(divCastCalcSize);
        Bmm2ResultDiv(extraInfo, vec2S1Idx, bmm2ResPingUb, softmaxSumPingUb, s1RealSizeInLoop);
        if (vec2S1Idx < vec2LoopLimit - 1) {
            AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        }
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        Bmm2DataCopyOut(extraInfo, vec2S1Idx, s1RealSizeInLoop, bmm2ResPongUb, attentionOutUb[extraInfo.taskIdMod2]);
    }
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    return;
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::Process()
{
    // 确定核内切分起点
    int64_t multiCoreInnerOffset = this->blockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    if constexpr (enableL1Reuse) {
        multiCoreInnerOffset = this->blockIdx / 2 * this->tilingData->multiCoreParams.splitFactorSize;
    }

    int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize;
    if (this->tilingData->multiCoreParams.totalSize < multiCoreInnerLimit) {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
    }
    // 计算sparse场景下s1的循环范围
    this->GetS1LoopRange(multiCoreInnerOffset, multiCoreInnerLimit);
    SplitS1dExtraInfo extraInfo[3];
    int64_t taskId = 0;
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    bool notSecondLast = true;
    bool notLast = true;
    multiCoreInnerLimit += 2;
    bool unPair = false;
    bool lastNotPair = false;
    if constexpr (enableL1Reuse) {
        multiCoreInnerLimit += 2;
        if ((multiCoreInnerLimit - multiCoreInnerOffset) % 2) {
            multiCoreInnerLimit += 1;
            unPair = true;
        }
    }

    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
         ++multiCoreInnerIdx) {
        if constexpr (enableL1Reuse) {
            if (multiCoreInnerIdx == multiCoreInnerLimit - 4) {
                notSecondLast = false;
            } else if (multiCoreInnerIdx == multiCoreInnerLimit - 2) {
                notLast = false;
            }
        } else {
            if (multiCoreInnerIdx == multiCoreInnerLimit - 2) {
                notSecondLast = false;
            } else if (multiCoreInnerIdx == multiCoreInnerLimit - 1) {
                notLast = false;
            }
        }
        bool notLastTwoLoop = notSecondLast && notLast;

        if constexpr (enableL1Reuse) {
            if ((multiCoreInnerIdx - multiCoreInnerOffset) % 2 != this->blockIdx % 2) {
                continue;
            }
        }
        if (multiCoreInnerIdx + 1 == multiCoreInnerLimit - 4) {
            lastNotPair = unPair;
        }
        this->ComputeAxisIdx(multiCoreInnerIdx);

        // s2轴循环计数, 支持sparse和非sparse场景
        this->GetS2LoopRange();
        if (taskId >= 1 && notLast) {
            WaitBmm1Result();
        }
        if (notLastTwoLoop) {
            if constexpr (enableL1Reuse) {
                this->SetExtraInfo(extraInfo[taskId % 3], taskId, 0, 0, multiCoreInnerIdx / 2, lastNotPair);
            } else {
                this->SetExtraInfo(extraInfo[taskId % 3], taskId, 0, 0, multiCoreInnerIdx, false);
            }
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
        ++taskId;
    }
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::ComputeAxisIdx(int64_t multiCoreInnerIdx)
{
    // 计算轴的idx
    this->boIdx = multiCoreInnerIdx / this->n2GS1o;
    this->n2oIdx = multiCoreInnerIdx % this->n2GS1o / this->gS1o;
    this->goIdx = multiCoreInnerIdx % this->gS1o / this->tilingData->coreParams.s1OuterSize;
    this->s1oIdx = multiCoreInnerIdx % this->tilingData->coreParams.s1OuterSize;
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::WaitBmm1Result()
{
    this->bmm1.WaitIterateAll();
    this->bmm1.End();
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::SetExtraInfo(
    SplitS1dExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx,
    bool lastNotPair)
{
    extraInfo.s2StartIdx = this->s2StartIdx;
    extraInfo.s2EndIdx = this->s2EndIdx;
    extraInfo.s1oIdx = this->s1oIdx;
    extraInfo.boIdx = this->boIdx;
    extraInfo.n2oIdx = this->n2oIdx;
    extraInfo.goIdx = this->goIdx;
    extraInfo.taskId = taskId;
    extraInfo.taskIdMod2 = taskId % 2;
    extraInfo.multiCoreInnerIdx = multiCoreInnerIdx;
    extraInfo.softmaxOutOffset = 0;
    if constexpr (enableL1Reuse) {
        extraInfo.lastNotPair = lastNotPair;
    }

    // s1尾块, 配比
    if (extraInfo.s1oIdx == this->tilingData->coreParams.s1OuterSize - 1) {
        extraInfo.s1RealSize = this->tilingData->coreParams.s1BaseTailSize;
    } else {
        extraInfo.s1RealSize = s1BaseSize;
    }

    // 第二阶段，在64 * 128个数的基础上根据dSize重新计算基本块，同时也要考虑D小的时候；
    extraInfo.vec2S1BaseSize = this->tilingData->coreParams.s1Vec2BaseSize;
    this->ComputeBmm1Tail(extraInfo);
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::ComputeBmm1Tail(SplitS1dExtraInfo &extraInfo)
{
    extraInfo.s2RealSize = extraInfo.s2EndIdx - extraInfo.s2StartIdx;
    extraInfo.s2RealSizeAlign8 = CeilDiv(extraInfo.s2RealSize, 8) * 8;
    extraInfo.s2RealSizeAlign16 = this->Align(extraInfo.s2RealSize);
    extraInfo.s2RealSizeAlign32 = CeilDiv(extraInfo.s2RealSize, 32) * 32;
    extraInfo.s2RealSizeAlign64 = CeilDiv(extraInfo.s2RealSize, 64) * 64;
    extraInfo.s2RealSizeFloorAlign8 = extraInfo.s2RealSize / 8 * 8;
    extraInfo.vecS1BaseSize = 1024 / extraInfo.s2RealSizeAlign64 * 8; // 基本块为8 * 1024大小
    // s1 尽量用8对齐来提升softmax的性能
    extraInfo.vecS1BaseSize = Min(extraInfo.vecS1BaseSize, this->s1BaseSize);
    extraInfo.vecS1BaseSize = Min(extraInfo.vecS1BaseSize, this->softmaxBufSize);
    extraInfo.s1BaseTimesS2Align16 = extraInfo.vecS1BaseSize * extraInfo.s2RealSizeAlign16;
    extraInfo.s1BaseTimesS2Align64 = extraInfo.vecS1BaseSize * extraInfo.s2RealSizeAlign64;
    extraInfo.duplicateMask = (((uint64_t)1 << (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSizeFloorAlign8)) - 1) &
        (~(((uint64_t)1 << (extraInfo.s2RealSize - extraInfo.s2RealSizeFloorAlign8)) - 1));
    if (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSizeFloorAlign8 == 64) {
        extraInfo.duplicateMask = 0xffffffffffffffff &
            (~(((uint64_t)1 << (extraInfo.s2RealSize - extraInfo.s2RealSizeFloorAlign8)) - 1));
    }
    // vec计算循环次数
    extraInfo.realSplitN = CeilDiv(extraInfo.s1RealSize, extraInfo.vecS1BaseSize);

    // 放满softmaxBuf需要的最大循环次数
    extraInfo.softmaxCopyOutLimit = this->softmaxBufSize / extraInfo.vecS1BaseSize;
    // 每次需要拷贝出去的softmaxSum和Max的数据量， 当S1RealSize比较小的时候，每次只需要拷贝realSplitN分的vecS1BaseSize
    // 尾块场景在SoftmaxCompute函数中处理。
    extraInfo.softmaxCopyOutS1Size = Min(extraInfo.realSplitN, extraInfo.softmaxCopyOutLimit) * extraInfo.vecS1BaseSize;
    return;
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::IterateBmm1(SplitS1dExtraInfo &extraInfo)
{
    if (extraInfo.s1RealSize != lastVec1S1RealSize || this->tilingData->inputParams.sparseType > 0) {
        this->bmm1.SetOrgShape(extraInfo.s1RealSize, mm1Kb, mm1Ka, mm1Kb, extraInfo.s2RealSize);
        lastVec1S1RealSize = extraInfo.s1RealSize;
    }
    this->Bmm1SetTensorA(extraInfo);
    this->SetBmm1TensorB(extraInfo);
    if constexpr (enableL1Reuse) {
        this->bmm1.template IterateAll<false>(this->mm1Res[extraInfo.taskIdMod2], false, false, true,
                                              extraInfo.lastNotPair);
    } else {
        this->bmm1.template IterateAll<false>(this->mm1Res[extraInfo.taskIdMod2], false, false, true);
    }
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::Bmm1SetTensorA(SplitS1dExtraInfo &extraInfo)
{
    // 计算gm上的offset
    int64_t bOffset = 0;

    // s1需要考虑inner轴的影响
    int64_t s1Offset = 0;

    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        // BSH/BSNGD
        bOffset = extraInfo.boIdx * this->n2GS1D;
        s1Offset = extraInfo.s1oIdx * this->s1BaseN2GD;
        n2Offset = extraInfo.n2oIdx * this->gD;
        gOffset = extraInfo.goIdx * dSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        // SBH/SBNGD
        s1Offset = extraInfo.s1oIdx * this->s1BaseBN2GD;
        bOffset = extraInfo.boIdx * this->n2GD;
        n2Offset = extraInfo.n2oIdx * this->gD;
        gOffset = extraInfo.goIdx * dSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        // bnsd
        bOffset = extraInfo.boIdx * this->n2GS1D;
        n2Offset = extraInfo.n2oIdx * this->gS1D;
        gOffset = extraInfo.goIdx * this->s1D;
        s1Offset = extraInfo.s1oIdx * this->s1BaseD;
    }
    this->qCoreOffset = bOffset + n2Offset + gOffset + s1Offset;
    extraInfo.qCoreOffset = this->qCoreOffset;
    this->bmm1.SetTensorA(this->queryGm[this->qCoreOffset]);
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::CopyToL1(
    const GlobalTensor<INPUT_T> &gmTensor, LocalTensor<INPUT_T> &localTensor, int64_t row, int64_t column,
    uint16_t srcDValue)
{
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = row;
    nd2nzParams.dValue = column;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.srcDValue =
        srcDValue; // 源操作数同一nd矩阵的相邻行起始地址间的偏移, 取值范围: [1, 65535]，单位: element
    // dstNzC0Stride: ND转换到NZ格式后, 源操作数中的一行会转换为目的操作数的多行. dstNzC0Stride表示, 目的nz矩阵中,
    // 来自源操作数同一行的多行数据相邻行起始地址间的偏移, 取值范围: dstNzC0Stride∈[1, 16384], 单位:C0_SIZE(32B)
    nd2nzParams.dstNzC0Stride = CeilDiv(row, 16L) * 16;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
    DataCopy(localTensor, gmTensor, nd2nzParams);
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::SetBmm1TensorB(SplitS1dExtraInfo &extraInfo)
{
    // 计算gm上的offset
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        // BSH/BSND
        bOffset = extraInfo.boIdx * this->n2S2D;
        s2Offset = extraInfo.s2StartIdx * this->n2D;
        n2Offset = extraInfo.n2oIdx * dSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        // SBH/SBND
        s2Offset = extraInfo.s2StartIdx * this->bN2D;
        bOffset = extraInfo.boIdx * this->n2D;
        n2Offset = extraInfo.n2oIdx * dSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        // BNSD
        bOffset = extraInfo.boIdx * this->n2S2D;
        n2Offset = extraInfo.n2oIdx * this->s2D;
        s2Offset = extraInfo.s2StartIdx * dSize;
    }
    int64_t kCoreOffset = bOffset + n2Offset + s2Offset;
    this->bmm1.SetTensorB(this->keyGm[kCoreOffset], true);
    this->bmm1.SetTail(extraInfo.s1RealSize, extraInfo.s2RealSize, this->tilingData->inputParams.dSize);
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::ProcessVec1(SplitS1dExtraInfo &extraInfo)
{
    if constexpr (enableL1Reuse) {
        if (extraInfo.lastNotPair) {
            return;
        }
    }

    LocalTensor<T> stage1PingTensor = this->stage1PingBuf.template Get<T>(); // t.a 32k
    LocalTensor<T> stage1PongTensor = this->stage1PongBuf.template Get<T>(); // i.a 32k
    LocalTensor<T> &actualUseTensor = bmm1Format == CubeFormat::ND ? stage1PongTensor : stage1PingTensor;
    LocalTensor<T> commonTBuf = this->commonTBuf.template Get<T>(); // t.b 32k
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t eventIdVToMte2A = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdVToMte2B = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    for (uint32_t loopIdx = 0; loopIdx < extraInfo.realSplitN; ++loopIdx) {
        extraInfo.vecS1TailSize = extraInfo.vecS1BaseSize;
        if (loopIdx == extraInfo.realSplitN - 1) {
            extraInfo.vecS1TailSize = extraInfo.s1RealSize - loopIdx * extraInfo.vecS1BaseSize;
        }
        if (loopIdx > 0) {
            AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
        }
        if constexpr (IsSameType<INPUT_T, float>::value) {
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        if constexpr (hasPse == true) {
            this->pseInfo.loopIdx = loopIdx;
            this->pseInfo.vec1S1RealSize = extraInfo.vecS1TailSize;
            this->pseInfo.s2RealSize = extraInfo.s2RealSize;
            this->pseInfo.s2AlignedSize = extraInfo.s2RealSizeAlign64;
            this->pseInfo.s2StartIdx = extraInfo.s2StartIdx;
            this->pseInfo.bSSOffset = extraInfo.boIdx * this->s1S2;
            this->pseInfo.s2SizeAcc = extraInfo.boIdx * pseInfo.s2Size;
            this->pseInfo.boIdx = extraInfo.boIdx;
            this->pseInfo.n2oIdx = extraInfo.n2oIdx;
            this->pseInfo.goIdx = extraInfo.goIdx;
            this->pseInfo.s1oIdx = extraInfo.s1oIdx;
            this->pseInfo.vec1S1BaseSize = extraInfo.vecS1BaseSize;
            this->pseInfo.needCast = true;

            if (this->pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                this->pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                LocalTensor<half> pseUb = this->pseTBuf.template Get<half>();
                PseSlopeCopyIn<T, hasPse>(commonTBuf, pseUb, this->pseSlope, this->pseAlibiGm, this->pseInfo, 64);
            } else {
                LocalTensor<INPUT_T> pseUb = this->pseTBuf.template Get<INPUT_T>();
                PseCopyIn<INPUT_T, T, layOutType, hasPse>(commonTBuf, pseUb, this->pseGm, this->pseInfo, 64);
                if constexpr (IsSameType<INPUT_T, float>::value) {
                    AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                    AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                }
            }
        }
        this->GetBmm1Result(extraInfo, actualUseTensor, loopIdx);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        if (this->tilingData->inputParams.pseType != (uint32_t)PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            Muls(stage1PingTensor, actualUseTensor, static_cast<T>(this->tilingData->inputParams.scaleValue),
                 extraInfo.vecS1TailSize * extraInfo.s2RealSizeAlign64);
        }
        if constexpr (hasPse == true) {
            AscendC::PipeBarrier<PIPE_V>();
            PseCompute<T, hasPse>(this->tilingData->inputParams.pseType != (uint32_t)PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE ? stage1PingTensor : actualUseTensor, commonTBuf, this->pseInfo);
        }
        this->CopyInAttenMask(extraInfo, loopIdx, -1);

        if (this->tilingData->inputParams.pseType == (uint32_t)PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            AscendC::PipeBarrier<PIPE_V>();
            Muls(stage1PingTensor, actualUseTensor, static_cast<T>(this->tilingData->inputParams.scaleValue),
                 extraInfo.vecS1TailSize * extraInfo.s2RealSizeAlign64);
        }
        AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        if constexpr (hasAtten == true) {
            if (this->attenMaskComputeMode != AttenMaskComputeMode::PREFIX_COMPUTE_MODE) {
                this->ComputeAttenMask(extraInfo, stage1PingTensor, 0);
            }

            if (this->tilingData->inputParams.attenMaskCompressMode ==
                static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
                event_t eventIdVToMte2Tmp = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2Tmp);
                AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2Tmp);
                this->CopyInAttenMask(extraInfo, loopIdx, this->attenMaskOffsetPre);
                event_t eventIdMte2ToVTmp = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVTmp);
                AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVTmp);
                this->ComputeAttenMask(extraInfo, stage1PingTensor, 1);
            }

            if (this->attenMaskComputeMode == AttenMaskComputeMode::PREFIX_COMPUTE_MODE) {
                event_t eventIdVToMte2Tmp = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2Tmp);
                AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2Tmp);

                this->CopyInAttenMask(extraInfo, loopIdx, -2);
                event_t eventIdMte2ToVTmp = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVTmp);
                AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVTmp);

                LocalTensor<uint8_t> attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
                LocalTensor<uint8_t> attenMaskUbTmp = this->pseTBuf.template Get<uint8_t>();

                auto attenmaskTensorTmp = attenMaskUb.ReinterpretCast<uint16_t>();
                auto attenmaskPreTensorTmp = attenMaskUbTmp.ReinterpretCast<uint16_t>();

                int32_t maskNum = extraInfo.vecS1TailSize * extraInfo.s2RealSizeAlign64 / 2;
                And(attenmaskTensorTmp, attenmaskTensorTmp, attenmaskPreTensorTmp, maskNum);

                attenMaskUb = attenmaskTensorTmp.ReinterpretCast<uint8_t>();
                AscendC::PipeBarrier<PIPE_V>();

                this->ComputeAttenMask(extraInfo, stage1PingTensor, 0);
            }
        }


        if (loopIdx < extraInfo.realSplitN - 1 && !hasSink) {
            AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
        }
        if (loopIdx > 0) {
            AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
        }

        if constexpr (hasDrop == true) {
            LocalTensor<uint8_t> dropMaskUb = this->maskTBufPong.template Get<uint8_t>();
            this->dropMaskInfo.s1InnerIdx = loopIdx;
            this->dropMaskInfo.s2StartIdx = extraInfo.s2StartIdx;
            this->dropMaskInfo.bSSOffset = extraInfo.boIdx * this->s1S2;
            this->dropMaskInfo.n2OutIdx = extraInfo.n2oIdx;
            this->dropMaskInfo.gOutIdx = extraInfo.goIdx;
            this->dropMaskInfo.s1OutIdx = extraInfo.s1oIdx;
            this->dropMaskInfo.splitS1BaseSize = extraInfo.vecS1BaseSize;
            this->dropMaskInfo.s1CopySize = static_cast<uint32_t>(extraInfo.vecS1TailSize);
            this->dropMaskInfo.s2CopySize = static_cast<uint32_t>(extraInfo.s2RealSize);
            this->dropMaskInfo.s2TotalSize = this->tilingData->inputParams.s2Size;
            this->dropMaskInfo.boolMode = this->dropMaskUnAligned;
            CopyInDropMask<hasDrop>(dropMaskUb, this->dropMaskGm, this->dropMaskGm, this->dropMaskInfo, 64);
        }

        this->SoftMaxCompute(extraInfo, stage1PingTensor, loopIdx);
        if (loopIdx < extraInfo.realSplitN - 1 && hasSink) {
            AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
        }

        if constexpr (hasDrop == true) {
            LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
            LocalTensor<uint8_t> dropMaskUb = this->maskTBufPong.template Get<uint8_t>();
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            this->dropMaskInfo.firstAxis = static_cast<uint32_t>(extraInfo.vecS1TailSize);
            this->dropMaskInfo.lstAxis = static_cast<uint32_t>(extraInfo.s2RealSizeAlign64);
            this->dropMaskInfo.maskLstAxis = this->dropMaskInfo.lstAxis;
            this->dropMaskInfo.keepProb = this->tilingData->inputParams.keepProb;
            ComputeDropMask<T, hasDrop>(stage1PingTensor, stage1PingTensor, dropMaskUb, apiTmpBuffer,
                                        this->dropMaskInfo);
        }

        if (loopIdx < extraInfo.realSplitN - 1) {
            AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
        }

        if (loopIdx > 0) {
            AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV); // 可以调整到DataCopy附近就好
        }
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (!IsSameType<T, INPUT_T>::value) {
            if constexpr (bmm1Format == CubeFormat::NZ) {
                LocalTensor<INPUT_T> stage1NzTensor = this->vecOut.template Get<INPUT_T>();
                LocalTensor<INPUT_T> stage1CastTensor = this->stage1PingBuf.template Get<INPUT_T>();
                Cast(stage1CastTensor, stage1PingTensor, RoundMode::CAST_ROUND,
                     extraInfo.vecS1TailSize * extraInfo.s2RealSizeAlign64);
                NdToNz(extraInfo, stage1NzTensor, stage1CastTensor, loopIdx);
            } else {
                LocalTensor<INPUT_T> stage1CastTensor = this->vecOut.template Get<INPUT_T>();
                Cast(stage1CastTensor, stage1PingTensor, RoundMode::CAST_ROUND,
                     extraInfo.vecS1TailSize * extraInfo.s2RealSizeAlign64);
                AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                DataCopyParams dataCopyParams;
                dataCopyParams.blockCount = extraInfo.vecS1TailSize;
                dataCopyParams.blockLen = extraInfo.s2RealSizeAlign16 / 16;
                dataCopyParams.srcStride = (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSizeAlign16) / 16;
                dataCopyParams.dstStride = 0;
                DataCopy(this->stage1Res[extraInfo.taskIdMod2][loopIdx * extraInfo.s1BaseTimesS2Align16],
                         stage1CastTensor, dataCopyParams);
                AscendC::PipeBarrier<PIPE_V>();
            }
        } else {
            AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = extraInfo.vecS1TailSize;
            dataCopyParams.blockLen = extraInfo.s2RealSizeAlign16 / 8;
            dataCopyParams.srcStride = (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSizeAlign16) / 8;
            dataCopyParams.dstStride = 0;
            DataCopy(this->stage1Res[extraInfo.taskIdMod2][loopIdx * extraInfo.s1BaseTimesS2Align16], stage1PingTensor,
                     dataCopyParams);
        }
        if (loopIdx < extraInfo.realSplitN - 1) {
            AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        }
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2A);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2B);
    return;
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void
FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::CopyInAttenMask(SplitS1dExtraInfo &extraInfo,
                                                                                   int64_t loopIdx, int64_t maskOffset)
{
    if constexpr (hasAtten == true) {
        LocalTensor<uint8_t> attenMaskUb;
        // init
        if (maskOffset == -1) {
            maskOffset = this->ComputeAttenMaskOffset(extraInfo, loopIdx);
            if (this->attenMaskComputeMode == AttenMaskComputeMode::PREFIX_N_COMPUTE_MODE) {
                maskOffset = this->attenMaskOffsetPre;
            }
            attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
            // prefix compress
        } else if (maskOffset == -2) {
            attenMaskUb = this->pseTBuf.template Get<uint8_t>();
            maskOffset = this->attenMaskOffsetPre;
            // band compress
        } else {
            attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
        }
        BoolCopyIn(attenMaskUb, this->attenMaskGmInt, maskOffset, extraInfo.vecS1TailSize, extraInfo.s2RealSize,
                   this->tilingData->inputParams.attenMaskS2Size, 64);
        return;
    }
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline int64_t
FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::ComputeAttenMaskOffset(SplitS1dExtraInfo &extraInfo,
                                                                                          int64_t loopIdx)
{
    if constexpr (hasAtten == true) {
        int64_t delta = 0;
        if (this->tilingData->inputParams.attenMaskCompressMode ==
            static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
            return this->ComputeOffsetForNoCompress(extraInfo, loopIdx);
        } else if (this->tilingData->inputParams.attenMaskCompressMode ==
                   static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE)) {
            delta = extraInfo.s1oIdx * s1BaseSize + loopIdx * extraInfo.vecS1BaseSize;
        } else if (this->tilingData->inputParams.attenMaskCompressMode ==
                   static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
            delta = extraInfo.s1oIdx * s1BaseSize + loopIdx * extraInfo.vecS1BaseSize +
                    this->tilingData->inputParams.s2Size - this->tilingData->inputParams.s1Size;
        } else if (this->tilingData->inputParams.attenMaskCompressMode ==
                   static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
            int64_t deltaPre = extraInfo.s1oIdx * s1BaseSize + loopIdx * extraInfo.vecS1BaseSize -
                               extraInfo.s2StartIdx - this->tilingData->inputParams.preTokens - 1;
            this->attenMaskOffsetPre =
                this->ComputeOffsetForCausal(deltaPre, extraInfo.vecS1BaseSize, this->tilingData->inputParams.s2Size,
                                             this->tilingData->inputParams.attenMaskS2Size);
            delta = extraInfo.s1oIdx * s1BaseSize + loopIdx * extraInfo.vecS1BaseSize - extraInfo.s2StartIdx +
                    this->tilingData->inputParams.nextTokens;
        } else if (this->tilingData->inputParams.attenMaskCompressMode ==
                   static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) {
            int64_t s1VOffset = extraInfo.s1oIdx * s1BaseSize + loopIdx * extraInfo.vecS1BaseSize;
            int64_t s1s2Sub = this->tilingData->inputParams.s1Size - this->tilingData->inputParams.s2Size;
            int64_t prefixNTmp = ((__gm__ int64_t *)this->prefixNAddr)[extraInfo.boIdx];
            delta = s1VOffset - s1s2Sub;

            this->attenMaskOffsetPre = this->ComputeOffsetForPrefixRectangle(
                prefixNTmp + 1, this->tilingData->inputParams.s2Size, this->tilingData->inputParams.attenMaskS2Size);
            if (this->blockIdx + extraInfo.vecS1TailSize < prefixAttenMaskDownHeight) { // in case of out of bound
                this->attenMaskOffsetPre += this->tilingData->inputParams.attenMaskS2Size * this->blockIdx;
            }
            int64_t intersectionX = s1s2Sub + prefixNTmp;
            if (s1VOffset > intersectionX) {
                this->attenMaskComputeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
            } else if (s1VOffset + extraInfo.vecS1BaseSize < intersectionX) {
                this->attenMaskComputeMode = AttenMaskComputeMode::PREFIX_N_COMPUTE_MODE;
            } else {
                this->attenMaskComputeMode = AttenMaskComputeMode::PREFIX_COMPUTE_MODE;
            }
        } else {
            return 0;
        }
        return this->ComputeOffsetForCausal(delta, extraInfo.vecS1BaseSize, this->tilingData->inputParams.s2Size,
                                            this->tilingData->inputParams.attenMaskS2Size);
    }
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline int64_t FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::ComputeOffsetForNoCompress(
    SplitS1dExtraInfo &extraInfo, int64_t loopIdx)
{
    if constexpr (hasAtten == true) {
        int64_t bOffset = 0;
        int64_t n2Offset = 0;
        int64_t gOffset = 0;
        int64_t s1Offset = extraInfo.s1oIdx * this->s1BaseS2 +
                           loopIdx * extraInfo.vecS1BaseSize * this->tilingData->inputParams.s2Size;
        int64_t s2Offset = extraInfo.s2StartIdx;
        if (this->tilingData->inputParams.attenMaskShapeType == this->attenMaskBN2GS1S2) {
            bOffset = extraInfo.boIdx * this->n2GS1S2;
            n2Offset = extraInfo.n2oIdx * this->gS1S2;
            gOffset = extraInfo.goIdx * this->s1S2;
        } else if (this->tilingData->inputParams.attenMaskShapeType == this->attenMaskBS1S2) {
            bOffset = extraInfo.boIdx * this->s1S2;
        }
        return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
    }
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::NdToNz(
    SplitS1dExtraInfo &extraInfo, LocalTensor<INPUT_T> &nzResUb, LocalTensor<INPUT_T> &vec1ResUb, int64_t loopIdx)
{
    auto nzResUbTmp = nzResUb.template ReinterpretCast<half>();
    auto vec1ResUbTmp = vec1ResUb.template ReinterpretCast<half>();
    int64_t s21 = CeilDiv(extraInfo.s2RealSizeAlign16, 16L);
    int64_t s2InnerLoop = s21 / 8L;
    int64_t s2InnerRemain = s21 % 8L;
    int64_t repeatMaxSize16 = this->repeatMaxBytes / sizeof(INPUT_T);
    CopyRepeatParams repeatParams;
    repeatParams.srcStride = 1;
    repeatParams.dstStride = extraInfo.vecS1TailSize + 1; // 防止bank冲突，所以+1
    repeatParams.srcRepeatSize = extraInfo.s2RealSizeAlign64 / 16;
    repeatParams.dstRepeatSize = 1;
    int32_t s1OuterLoop = extraInfo.vecS1TailSize / this->repeatMaxTimes;
    int32_t s1OuterRemain = extraInfo.vecS1TailSize % this->repeatMaxTimes;
    int32_t s1OuterBmm1Offset = this->repeatMaxTimes * extraInfo.s2RealSizeAlign64;
    int32_t s1OuterTempOffset = this->repeatMaxTimes * 16;
    int64_t vecS1BaseSizeTime16 = extraInfo.vecS1BaseSize * 16;
    int64_t offsetJ = 8 * extraInfo.vecS1TailSize * 16 + 128;
    AscendC::PipeBarrier<PIPE_V>();
    for (int64_t outerIndex = 0; outerIndex < s1OuterLoop; ++ outerIndex) {
        for (int64_t j = 0; j < s2InnerLoop; ++j) {
            Copy(nzResUbTmp[outerIndex * s1OuterTempOffset + j * offsetJ],
                 vec1ResUbTmp[outerIndex * s1OuterBmm1Offset + j * 128],
                 repeatMaxSize16, this->repeatMaxTimes, repeatParams);
        }
        if (s2InnerRemain) {
            Copy(nzResUbTmp[outerIndex * s1OuterTempOffset + s2InnerLoop * offsetJ],
                 vec1ResUbTmp[outerIndex * s1OuterBmm1Offset + s2InnerLoop * 128],
                 s2InnerRemain * 16, this->repeatMaxTimes, repeatParams);
        }
    }

    if (s1OuterRemain) {
        for (int64_t j = 0; j < s2InnerLoop; ++j) {
            Copy(nzResUbTmp[s1OuterLoop * s1OuterTempOffset + j * offsetJ],
                 vec1ResUbTmp[s1OuterLoop * s1OuterBmm1Offset + j * 128],
                 repeatMaxSize16, s1OuterRemain, repeatParams);
        }
        if (s2InnerRemain) {
            Copy(nzResUbTmp[s1OuterLoop * s1OuterTempOffset + s2InnerLoop * offsetJ],
                 vec1ResUbTmp[s1OuterLoop * s1OuterBmm1Offset + s2InnerLoop * 128],
                 s2InnerRemain * 16, s1OuterRemain, repeatParams);
        }
    }

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = s21;
    dataCopyParams.blockLen = extraInfo.vecS1TailSize;
    dataCopyParams.srcStride = 1;
    dataCopyParams.dstStride = this->Align(extraInfo.s1RealSize) - extraInfo.vecS1TailSize;
    DataCopy(this->stage1Res[extraInfo.taskIdMod2][loopIdx * vecS1BaseSizeTime16], nzResUb, dataCopyParams);
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::GetBmm1Result(
    SplitS1dExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb, int64_t loopIdx)
{
    if constexpr (bmm1Format == CubeFormat::NZ) {
        Nz2NdInfo nz2NdInfo;
        nz2NdInfo.ndFirstAxisRealSize = extraInfo.s1RealSize;
        nz2NdInfo.ndFirstAxisBaseSize = extraInfo.vecS1BaseSize;
        nz2NdInfo.ndFirstAxisLoopSize = extraInfo.vecS1TailSize;
        nz2NdInfo.ndLastAxis = extraInfo.s2RealSizeAlign64;
        nz2NdInfo.loopIdx = loopIdx;
        LocalTensor<T> tempUb = this->stage1PongBuf.template Get<T>();
        NzToNd(nz2NdInfo, this->mm1Res[extraInfo.taskIdMod2], tempUb, bmm1ResUb);
        return;
    } else {
        if (extraInfo.s2RealSizeAlign64 == extraInfo.s2RealSize) {
            // 16对齐场景，使用DataCopy提升性能
            DataCopy(bmm1ResUb, this->mm1Res[extraInfo.taskIdMod2][loopIdx * extraInfo.s1BaseTimesS2Align64],
                     extraInfo.vecS1TailSize * extraInfo.s2RealSizeAlign64);
        } else {
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = extraInfo.vecS1TailSize;
            dataCopyParams.blockLen = extraInfo.s2RealSize * sizeof(T);
            dataCopyParams.dstStride = (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSize) / 8;
            dataCopyParams.srcStride = 0;

            DataCopyPadParams padParams;
            padParams.isPad = false;
            DataCopyPad(bmm1ResUb,
                        this->mm1Res[extraInfo.taskIdMod2][loopIdx * extraInfo.vecS1BaseSize * extraInfo.s2RealSize],
                        dataCopyParams, padParams);
        }
        uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(extraInfo.vecS1TailSize),
                                     static_cast<uint32_t>(extraInfo.s2RealSizeAlign64)};
        bmm1ResUb.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, DataFormat::ND));
    }
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::ComputeAttenMask(
    SplitS1dExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb, const uint8_t maskType)
{
    if constexpr (hasAtten == true) {
        LocalTensor<uint8_t> attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
        int32_t alignedS2Size = extraInfo.s2RealSizeAlign64;
        uint32_t shapeArray[] = {static_cast<uint32_t>(extraInfo.vecS1TailSize), static_cast<uint32_t>(alignedS2Size)};
        attenMaskUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        attenMaskUb.SetSize(extraInfo.vecS1TailSize * alignedS2Size);
        bmm1ResUb.SetSize(extraInfo.vecS1TailSize * extraInfo.s2RealSizeAlign64);
        LocalTensor<uint8_t> apiTmpBuffer = commonTBuf.template Get<uint8_t>();
        SelectWithBytesMaskShapeInfo shapeInfo;
        shapeInfo.firstAxis = extraInfo.vecS1TailSize;
        shapeInfo.srcLastAxis = extraInfo.s2RealSizeAlign64;
        shapeInfo.maskLastAxis = alignedS2Size;

        if (maskType == 0) {
            SelectWithBytesMask(bmm1ResUb, bmm1ResUb, this->negativeFloatScalar, attenMaskUb, apiTmpBuffer, shapeInfo);
        } else {
            SelectWithBytesMask(bmm1ResUb, this->negativeFloatScalar, bmm1ResUb, attenMaskUb, apiTmpBuffer, shapeInfo);
        }
        return;
    }
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::SoftMaxCompute(
    SplitS1dExtraInfo &extraInfo, LocalTensor<T> &srcTensor, int64_t loopIdx)
{
    int64_t vecS1TailSize = CeilDiv(extraInfo.vecS1TailSize, 8) * 8;
    uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(vecS1TailSize),
                                 static_cast<uint32_t>(extraInfo.s2RealSizeAlign64)};
    uint32_t bmm1ResOriginShape[] = {static_cast<uint32_t>(vecS1TailSize),
                                     static_cast<uint32_t>(extraInfo.s2RealSizeAlign64)};

    if constexpr (hasAtten == false) {
        uint64_t mask[2] = {extraInfo.duplicateMask, 0};
        Duplicate<T>(srcTensor[extraInfo.s2RealSizeFloorAlign8], this->negativeFloatScalar, mask, vecS1TailSize,
                     1, extraInfo.s2RealSizeAlign64 * sizeof(T) / this->blockBytes);
    } else if (this->tilingData->inputParams.attenMaskCompressMode !=
               static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE) || extraInfo.s2RealSize < 64) {
        uint64_t mask[2] = {extraInfo.duplicateMask, 0};
        Duplicate<T>(srcTensor[extraInfo.s2RealSizeFloorAlign8], this->negativeFloatScalar, mask, vecS1TailSize,
                     1, extraInfo.s2RealSizeAlign64 * sizeof(T) / this->blockBytes);
    }

    srcTensor.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, 2, bmm1ResOriginShape, DataFormat::ND));
    srcTensor.SetSize(vecS1TailSize * extraInfo.s2RealSizeAlign64);

    uint32_t maxSumShape[] = {static_cast<uint32_t>(vecS1TailSize),
                              static_cast<uint32_t>(this->fp32BaseSize)};
    LocalTensor<T> sumUb;
    LocalTensor<T> maxUb;
    // 当S1RealSize大于256时，sumUb和maxUb的8K不足以存放整个S1RealSize的数据量，需要循环存放。
    // 需要将realSplitN循环的loopIndex取模 softmaxCopyOutLimit
    int64_t sumOffset = (loopIdx % extraInfo.softmaxCopyOutLimit) * extraInfo.vecS1BaseSize * this->fp32BaseSize;
    sumUb = this->softmaxSumBuf.template Get<T>()[sumOffset];
    maxUb = this->softmaxMaxBuf.template Get<T>()[sumOffset];

    sumUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    maxUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));

    if (hasSink) {
        float inMax = sinkGm.GetValue(extraInfo.n2oIdx * this->tilingData->inputParams.gSize + extraInfo.goIdx);
        float inSum = 1.0;
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        Duplicate(sumUb, static_cast<T>(inSum), vecS1TailSize * this->fp32BaseSize);
        Duplicate(maxUb, static_cast<T>(inMax), vecS1TailSize * this->fp32BaseSize);
    }

    uint32_t expShape[] = {static_cast<uint32_t>(vecS1TailSize),
                           static_cast<uint32_t>(this->blockBytes / sizeof(T))};
    LocalTensor<T> expUb = this->softmaxExpBuf.template Get<T>()[0];
    expUb.SetShapeInfo(ShapeInfo(2, expShape, DataFormat::ND));

    LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
    AscendC::PipeBarrier<PIPE_V>();

    SoftMaxTiling newTiling = AscendC::SoftMaxFlashV2TilingFuncImpl(vecS1TailSize,
                                                                    extraInfo.s2RealSizeAlign64, sizeof(T),
                                                                    sizeof(T),
                                                                    apiTmpBuffer.GetSize() / sizeof(T),
                                                                    false, true);
    if (hasSink){
        expUb = this->maskTBufPing.template Get<T>()[0];
        expUb.SetShapeInfo(ShapeInfo(2, expShape, DataFormat::ND));
        SoftmaxFlashV2<T, true, true, true, false, SOFTMAX_DEFAULT_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                    sumUb, maxUb, apiTmpBuffer, newTiling);
    }else{
        SoftmaxFlashV2<T, false, true, true, false, SOFTMAX_DEFAULT_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                    sumUb, maxUb, apiTmpBuffer, newTiling);
    }


    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION) {
        SoftMaxShapeInfo softmaxShapeInfo{
            static_cast<uint32_t>(vecS1TailSize), static_cast<uint32_t>(extraInfo.s2RealSizeAlign64),
            static_cast<uint32_t>(vecS1TailSize), static_cast<uint32_t>(extraInfo.s2RealSizeAlign64)};
        AdjustSoftMaxRes<T, T>(srcTensor, maxUb, this->negativeIntScalar, 0.0, softmaxShapeInfo);
        if (loopIdx == extraInfo.realSplitN - 1 || (loopIdx + 1) % extraInfo.softmaxCopyOutLimit == 0) {
            LocalTensor<T> maxTensor;
            LocalTensor<T> sumTensor;
            maxTensor = this->softmaxMaxBuf.template Get<T>();
            sumTensor = this->softmaxSumBuf.template Get<T>();
            uint32_t currS1Size = static_cast<uint32_t>(
                Min(extraInfo.softmaxCopyOutS1Size, extraInfo.s1RealSize - extraInfo.softmaxOutOffset));
            SoftMaxShapeInfo softmaxFullShapeInfo{currS1Size, static_cast<uint32_t>(this->fp32BaseSize), currS1Size,
                                                  static_cast<uint32_t>(this->fp32BaseSize)};
            AdjustSoftMaxRes<T, T>(sumTensor, maxTensor, this->negativeIntScalar, this->positiveFloatScalar,
                                   softmaxFullShapeInfo);
        }
    }

    // softmax sum ub为8K，最大可以支持S1到256，如果S1方向有配比，这里会积攒到256 * 8个数后统一会写到GM
    // 如果不足256 * 8个数，那么写的时候的vs1Offset偏移默认为0
    if (loopIdx == extraInfo.realSplitN - 1 || (loopIdx + 1) % extraInfo.softmaxCopyOutLimit == 0) {
        int64_t bOffset = extraInfo.boIdx * this->n2GS1 * this->fp32BaseSize;
        int64_t n2Offset = extraInfo.n2oIdx * this->gS1 * this->fp32BaseSize;
        int64_t gOffset = extraInfo.goIdx * this->tilingData->inputParams.s1Size * this->fp32BaseSize;
        int64_t s1Offset = extraInfo.s1oIdx * s1BaseSize * this->fp32BaseSize;
        int64_t vs1Offset = extraInfo.softmaxOutOffset * this->fp32BaseSize;
        int64_t gmOffset = bOffset + n2Offset + gOffset + s1Offset + vs1Offset;

        // 每次计算的数据量需要考虑尾块，其中extraInfo.softmaxCopyOutS1Size 已经考虑了SplitN较小的场景
        int64_t calculateEleNum =
            Min(extraInfo.softmaxCopyOutS1Size, extraInfo.s1RealSize - extraInfo.softmaxOutOffset);
        int64_t calculateSize = calculateEleNum * this->fp32BaseSize;

        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        // 每次向外copy的时候都从Ub的起始位置开始copy
        LocalTensor<float> sumUbStart;
        LocalTensor<float> maxUbStart;
        sumUbStart = this->softmaxSumBuf.template Get<float>()[0];
        maxUbStart = this->softmaxMaxBuf.template Get<float>()[0];

        DataCopy(this->softmaxSumGm[gmOffset], sumUbStart, calculateSize);
        DataCopy(this->softmaxMaxGm[gmOffset], maxUbStart, calculateSize);

        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        extraInfo.softmaxOutOffset += calculateEleNum;
    }
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
template <typename T2, typename T3, const auto &TILING>
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::WaitBmm2Result(
    matmul::Matmul<T2, b2Type, T3, bias2Type, TILING> &bmm2)
{
    bmm2.WaitIterateAll();
    bmm2.End();
}

FA_S1BN2GS1_FUNCTION_TEMPLATE
template <typename T2, typename T3, const auto &TILING>
__aicore__ inline void FlashAttentionScoreS1Bn2gs1<FA_S1BN2GS1_FUNCTION_PARAMS_TEMPLATE>::IterateBmm2(
    SplitS1dExtraInfo &extraInfo, matmul::Matmul<T2, b2Type, T3, bias2Type, TILING> &bmm2)
{
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        // BSH/BSND
        bOffset = extraInfo.boIdx * this->n2S2D;
        s2Offset = extraInfo.s2StartIdx * this->n2D;
        n2Offset = extraInfo.n2oIdx * dSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        // SBH/SBND
        s2Offset = extraInfo.s2StartIdx * this->bN2D;
        bOffset = extraInfo.boIdx * this->n2D;
        n2Offset = extraInfo.n2oIdx * dSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        // BNSD
        bOffset = extraInfo.boIdx * this->n2S2D;
        n2Offset = extraInfo.n2oIdx * this->s2D;
        s2Offset = extraInfo.s2StartIdx * dSize;
    }
    int64_t vCoreOffset = bOffset + n2Offset + s2Offset;
    if (extraInfo.s1RealSize != lastVec2S1RealSize || this->tilingData->inputParams.sparseType > 0) {
        bmm2.SetOrgShape(extraInfo.s1RealSize, mm2Kb, extraInfo.s2RealSizeAlign16, mm2Kb, dSize);
        lastVec2S1RealSize = extraInfo.s1RealSize;
    }
    bmm2.SetTensorA(this->stage1Res[extraInfo.taskIdMod2]);

    if constexpr (bmm2Source == TPosition::TSCM) {
        valueScmTensor = valueScm.AllocTensor<INPUT_T>();
        if (extraInfo.taskId == 0 || n2Offset != bmm2N2OffsetLast || bOffset != bmm2BOffsetLast) {
            bmm2N2OffsetLast = n2Offset;
            bmm2BOffsetLast = bOffset;
            CopyToL1(this->valueGm[vCoreOffset], valueScmTensor, this->tilingData->inputParams.s2Size, dSize,
                     srcDValue);
        }
        int64_t vCoreOffsetL1 = s2Offset;
        bmm2.SetTensorB(valueScmTensor[vCoreOffsetL1]);
        bmm2.SetTail(extraInfo.s1RealSize, this->tilingData->inputParams.dSize, extraInfo.s2RealSize);
        valueScm.FreeTensor(valueScmTensor);
    } else {
        bmm2.SetTensorB(this->valueGm[vCoreOffset]);
    }

    // 通过extraInfo.s2RealSize来限制一次计算的大小
    bmm2.SetTail(extraInfo.s1RealSize, this->tilingData->inputParams.dSize, extraInfo.s2RealSize);
    if constexpr (enableL1Reuse) {
        bmm2.template IterateAll<false>(this->mm2Res[extraInfo.taskIdMod2], false, false, true,
                                              extraInfo.lastNotPair);
    } else {
        bmm2.template IterateAll<false>(this->mm2Res[extraInfo.taskIdMod2], false, false, true);
    }
}

#endif // FLASH_ATTENTION_SCORE_S1_BN2GS1_H
