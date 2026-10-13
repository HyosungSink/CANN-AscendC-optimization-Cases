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
 * \file fused_floyd_attention_s1s2_bn2gs1.h
 * \brief
 */

#ifndef FUSED_FLOYD_ATTENTION_S1S2_BN2GS1_H
#define FUSED_FLOYD_ATTENTION_S1S2_BN2GS1_H

#include "util.h"
#include "dropmask.h"
#include "fused_floyd_attention_common.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "pse.h"

using matmul::MatmulType;

struct SplitExtraInfo {
    int64_t s2StartIdx;
    int64_t s2EndIdx;
    int64_t s2LoopCount;
    int64_t s1oIdx;
    int64_t boIdx;
    int64_t n2oIdx;
    int64_t atten_boIdx;
    int64_t goIdx;
    int64_t taskId;
    int8_t taskIdMod2;
    int8_t multiCoreInnerIdxMod2;
    int8_t needNz2Nd;
    bool lastNotPair;
    int32_t s1RealSize;
    int32_t s2RealSize;
    int32_t n2RealSize;
    int32_t s2AlignedSize;
    int32_t vec1S1BaseSize;
    int32_t vec1S1RealSize;
    int32_t vec2S1BaseSize;
    int32_t vec2S1RealSize;
    int32_t realSplitN;
    int32_t s2LoopLimit;
    int64_t multiCoreInnerIdx;
    int64_t qCoreOffset;
    int64_t pseS2ComputeSize;
    int64_t s1SizeDelta;
    int64_t s1SizeAcc;
    int64_t attenB1SSOffset;
    int64_t attenMaskS2Size;
    int64_t s1Size;
    int64_t s2Size;
    int64_t softmaxMaxOffset;
    int64_t k1CoreOffset;
};

constexpr int64_t GM_DOUBLE_BUFFER = 2;
constexpr int64_t INVALID_OFFSET = INT64_MIN;
constexpr int64_t SPLIT_S2_SIZE_LIMIT = 1024;
constexpr AscendC::SoftmaxConfig SOFTMAX_DEFAULT_CFG = {false};


__aicore__ inline void AttenMaskBoolCopyIn(LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor,
    int64_t srcOffset, uint32_t s1Size, uint32_t s2Size, int64_t totalS2Size, int64_t attenMaskShapeType=0)
{
    uint32_t alignedS2Size = CeilDiv(s2Size, blockBytes) * blockBytes;
    uint32_t shapeArray[] = {s1Size, alignedS2Size};
    dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND)); // 2:dimNum
    dstTensor.SetSize(s1Size * alignedS2Size);
    DataCopyParams dataCopyParams;
    if (attenMaskShapeType == 0) {
        dataCopyParams.blockCount = s1Size;
    } else if (attenMaskShapeType == 3) { // 3:B_N2_1_1_S2
        dataCopyParams.blockCount = 1;
    }
    dataCopyParams.dstStride = 0;
    if (totalS2Size % blockBytes == 0) {
        dataCopyParams.blockLen = alignedS2Size / blockBytes;
        dataCopyParams.srcStride = (totalS2Size - alignedS2Size) / blockBytes;
        DataCopy(dstTensor, srcTensor[srcOffset], dataCopyParams);
        if (attenMaskShapeType == 3) { // 3:B_N2_1_1_S2
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            uint32_t logStep = 0;
            for (uint32_t tmp = 1; tmp < s1Size; tmp <<= 1) ++logStep;
            /* 3. 逐级“自身翻倍”广播 */
            uint32_t curRows = 1;                      // 当前已经准备好的行数
            uint32_t rowBytes = alignedS2Size;         // 每行字节数
            for (uint32_t step = 0; step < logStep; ++step) {
                /* 把 [0, curRows) 行拷贝到 [curRows, 2*curRows) 行 */
                DataCopyParams bc;
                bc.blockCount = curRows;               // 拷贝 curRows 行
                bc.blockLen   = rowBytes / blockBytes; // 每行 block 数
                bc.dstStride  = 0;                     // 目的行之间连续
                bc.srcStride  = 0;                     // 源行之间也连续
                AscendC::PipeBarrier<PIPE_V>();
                DataCopy(dstTensor[curRows * rowBytes], // 目的首地址
                        dstTensor,                     // 源首地址
                        bc);
                curRows <<= 1;                          // 行数翻倍
            }
        }
    } else {
        dataCopyParams.blockLen = s2Size;
        dataCopyParams.srcStride = totalS2Size - s2Size;
        DataCopyPadParams dataCopyPadParams;
        dataCopyPadParams.isPad = true;
        dataCopyPadParams.rightPadding = alignedS2Size - s2Size;
        dataCopyPadParams.paddingValue = 0;
        DataCopyPad(dstTensor, srcTensor[srcOffset], dataCopyParams, dataCopyPadParams);
    }
}


__aicore__ const constexpr MatmulConfig &GetMmCfg(bool enableL1Reuse)
{
    if (enableL1Reuse) {
        return CFG_IBSHARE_EXCEED;
    } else {
        return CFG_EXCEED;
    }
}

// INPUT_T - means data type for input
// T       - means data type when calc
template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T = INPUT_T, bool isBasicBlock = false, CubeFormat bmm1Format = CubeFormat::ND,
          bool enableL1Reuse = false>
class FusedFloydAttentionS1s2Bn2gs1 {
public:
    __aicore__ inline FusedFloydAttentionS1s2Bn2gs1(){};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *key1, 
                                __gm__ uint8_t *value1, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const FusedFloydAttentionGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // define matmul
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::NORMAL>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true, LayoutMode::NORMAL>;
    using c1Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false, LayoutMode::NORMAL>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    constexpr static MatmulConfig MM_CFG_STD = GetNormalConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1);
    constexpr static MatmulConfig MM_CFG_LL1 = GetNormalConfig(false, false, false, BatchMode::BATCH_LARGE_THAN_L1);

    matmul::Matmul<a1Type, b1Type, c1Type, bias1Type, MM_CFG_STD> bmm1;
    matmul::Matmul<a1Type, a1Type, c1Type, bias1Type, MM_CFG_LL1> bmm2;

    using a3Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::BSNGD>;
    using b3Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true, LayoutMode::BSNGD>;
    using c3Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false, LayoutMode::BSNGD>;

    constexpr static MatmulConfig LIMIT_CFG = GetNormalConfig(true);
    matmul::Matmul<a3Type, b3Type, c3Type, bias1Type, LIMIT_CFG> bmm1k1;
    matmul::Matmul<a3Type, a3Type, c3Type, bias1Type, LIMIT_CFG> bmm2v2;

protected:
    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                     __gm__ uint8_t *key1, __gm__ uint8_t *value1,
                                     __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
                                     __gm__ uint8_t *softmaxSum,
                                     __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                     const FusedFloydAttentionGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void WaitBmm2Result();
    __aicore__ inline void IterateBmm2(SplitExtraInfo &extraInfo);
    __aicore__ inline void IterateBmm2forValue1(SplitExtraInfo &extraInfo);
    __aicore__ inline void SetExtraInfo(SplitExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount,
                                        int64_t s2LoopLimit, int64_t multiCoreInnerIdx, bool lastNotPair);
    __aicore__ inline void SetTiling(const FusedFloydAttentionGeneralTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void GetBNIdx(const int64_t &multiCoreInnerOffset, const int64_t &multiCoreInnerLimit);
    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);
    template <typename T2, const MatmulConfig &MM_CFG>
    __aicore__ inline void IterateBmm1k1(SplitExtraInfo &extraInfo,
                                      matmul::Matmul<a3Type, b3Type, T2, bias1Type, MM_CFG> &bmm1k1);
    template <typename T2, const MatmulConfig &MM_CFG>
    __aicore__ inline void IterateBmm1(SplitExtraInfo &extraInfo,
                                       matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG> &bmm1);
    template <typename T2, const MatmulConfig &MM_CFG>
    __aicore__ inline void Bmm1SetTensorA(SplitExtraInfo &extraInfo,
                                          matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG> &bmm1);
    template <typename T2, const MatmulConfig &MM_CFG>
    __aicore__ inline void SetBmm1TensorB(SplitExtraInfo &extraInfo,
                                          matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG> &bmm1, int64_t loopN2);

    __aicore__ inline void ComputeBmm1Tail(SplitExtraInfo &extraInfo);
    __aicore__ inline void ProcessVec1(SplitExtraInfo &extraInfo);
    __aicore__ inline void CopyInAttenMask(SplitExtraInfo &extraInfo, int64_t loopIdx, int64_t maskOffset, int64_t loopN2,
                                           bool secondTime = false);
    __aicore__ inline int64_t ComputeOffsetForNoCompress(SplitExtraInfo &extraInfo, int64_t loopIdx, int64_t loopN2);
    __aicore__ inline void GetBmm1Result(SplitExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb, int64_t loopIdx, int64_t loopN2);
    __aicore__ inline void ComputeAttenMask(SelectWithBytesMaskShapeInfo &shapeInfo, LocalTensor<T> &bmm1ResUb,
                                            LocalTensor<uint8_t> &maskUb, const uint8_t maskType, event_t vWaitMte2);

    __aicore__ inline void SoftMaxCompute(SplitExtraInfo &extraInfo, LocalTensor<T> &srcTensor, int64_t loopIdx, int64_t loopN2);
    __aicore__ inline void SoftMaxCheckResCompress(SplitExtraInfo &extraInfo, int64_t vec1S1realSplitN);
    __aicore__ inline void InvalidLineSplitS2Process(SplitExtraInfo &extraInfo, LocalTensor<T> &srcTensor,
                                                     LocalTensor<T> &maxUb, int64_t loopIdx);
    __aicore__ inline void ProcessVec2(SplitExtraInfo &extraInfo);
    __aicore__ inline void Bmm2ResultMul(SplitExtraInfo &extraInfo, LocalTensor<T> &bmm2ResUb, int64_t s1oIdx);
    __aicore__ inline void Bmm2ResultDiv(SplitExtraInfo &extraInfo, int64_t s1oIdx);
    __aicore__ inline void Bmm2DataCopyOut(SplitExtraInfo &extraInfo, int64_t s1oIdx, int64_t mm2ResCalcSize, int64_t loopN2);
    __aicore__ inline void SoftmaxDataCopyOut(SplitExtraInfo &extraInfo, int64_t loopN2);

    uint32_t s1BaseSize;
    uint32_t s2BaseSize;
    uint32_t n2BaseSize;
    uint32_t n2BaseTailSize;
    uint32_t bmm1Num;
    uint32_t bmmv2Num;

    uint32_t dSize;
    int64_t dSizeAlign16;
    int64_t s1Size;
    int64_t s2Size;
    int64_t s1OuterSize;

    // sparse 用参数
    int64_t s2StartIdx;
    int64_t s2EndIdx;
    int64_t nextS2EndIdx;

    // BNG 外循环
    int64_t bngStartIdx;
    int64_t bngEndIdx;

    // s2方向的尾块，包含N:1配比
    int64_t lastS2RealSize = INVALID_OFFSET;
    int64_t bmm2LastS2RealSize = INVALID_OFFSET;

    // L1Reuse场景vector核是否是奇数核
    int64_t l1ReuseBlockMod2 = 0;

    int64_t qCoreOffset;

    // 资源分配
    TBuf<> maskTBufPing;
    TBuf<> maskTBufPong;
    TBuf<> pseTBuf;
    TBuf<> stage1PingBuf;
    TBuf<> stage1PongBuf;
    TBuf<> stage2TBuf;
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2];
    TBuf<> softmaxMaxBuf;
    TBuf<> commonTBuf; // common的复用空间

    LocalTensor<T> softmaxExpUb;
    GlobalTensor<T> mm1Res[2];
    GlobalTensor<T> mm2Res[2];
    GlobalTensor<T> vec2Res[2];
    GlobalTensor<INPUT_T> stage1Res[2];
    GlobalTensor<half> pseAlibiGm;

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

    // s2base*N之后的长度
    int64_t s2BaseNratioSize;
    int64_t s1BaseS2;

    int64_t s2BaseN2D;
    int64_t s2BaseBN2D;
    int64_t s1BaseN2GD;
    int64_t s1BaseBN2GD;
    int64_t s1BaseD;
    int64_t s2BaseNratioD;
    int64_t s2BaseNratioN2D;
    int64_t s2BaseNratioBN2D;
    int64_t bN2G;
    int64_t n2GS2;
    int64_t s2SizeSum;

    int64_t mm1Ka;
    int64_t mm1Kb;
    int64_t mm2Kb;
    // 当splitN大于16时，需要修改softMaxCheckRes数据类型
    uint16_t softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
    uint32_t negativeIntScalar = NEGATIVE_MIN_VAULE_FP32;
    T negativeFloatScalar;
    T positiveFloatScalar;

    AttenMaskComputeMode attenMaskComputeMode = AttenMaskComputeMode::NORMAL_MODE;

    int32_t blockIdx;
    const FusedFloydAttentionGeneralTilingData *__restrict tilingData;

    int64_t boIdx;
    int64_t n2oIdx;
    int64_t goIdx;
    int64_t s1oIdx;

    int64_t atten_boIdx;
    int64_t attenMaskShapeType;
    TPipe *pipe;

    GlobalTensor<INPUT_T> queryGm;
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<INPUT_T> keyGm1;
    GlobalTensor<INPUT_T> valueGm;
    GlobalTensor<INPUT_T> valueGm1;
    GlobalTensor<INPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<uint8_t> attenMaskGmInt;

};

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                                   __gm__ uint8_t *key1, __gm__ uint8_t *value1, 
                                                   __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
                                                   __gm__ uint8_t *softmaxSum,
                                                   __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                                   const FusedFloydAttentionGeneralTilingData *__restrict tiling,
                                                   TPipe *tPipe)
{
    this->InitInput(query, key, value, key1, value1, attenMask, softmaxMax, softmaxSum,
                    attentionOut, workspace, tiling, tPipe); // gm设置

    this->ComputeConstexpr();
    this->InitBuffer();
    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    DropOutBitModeInit(apiTmpBuffer);
    if (this->blockIdx < this->tilingData->multiCoreParams.coreNum) {
        LocalTensor<half> pseHelpBuffer = this->stage1PingBuf.template Get<half>();
    }

}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key,
                                                        __gm__ uint8_t *value, __gm__ uint8_t *key1,
                                                        __gm__ uint8_t *value1, __gm__ uint8_t *attenMask,
                                                        __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                                        __gm__ uint8_t *attentionOut,
                                                        __gm__ uint8_t *workspace,
                                                        const FusedFloydAttentionGeneralTilingData *__restrict tiling,
                                                        TPipe *tPipe)
{
    this->blockIdx = GetBlockIdx();
    this->pipe = tPipe;
    this->SetTiling(tiling);


    // init global buffer
    this->queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    this->keyGm1.SetGlobalBuffer((__gm__ INPUT_T *)key1);
    this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    this->valueGm1.SetGlobalBuffer((__gm__ INPUT_T *)value1);
    this->attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
    this->softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    this->softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    this->attentionOutGm.SetGlobalBuffer((__gm__ INPUT_T *)attentionOut);

    // 补齐到512， 统一按T处理
    int64_t mm1ResultSize = n2BaseSize * s1BaseSize * s2BaseSize;
    int64_t mmNRatioOffset = CeilDiv(mm1ResultSize * this->tilingData->coreParams.nRatio, 128) * 128 * sizeof(T);
    int64_t mm2ResultSize = n2BaseSize * s1BaseSize * dSizeAlign16;
    int64_t mm2Offset = CeilDiv(mm2ResultSize, 128) * 128 * 4;
    int64_t bmm1AndVec1Ratio = GM_DOUBLE_BUFFER;
    int64_t vector1OffsetPing = 0;
    int64_t vector1OffsetPong = mmNRatioOffset;

    // NZND场景，stage1Result不与bmm1Result共用空间，需要占用1倍mmNRatioOffset空间
    if constexpr (bmm1Format == CubeFormat::NZ) {
        if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
            vector1OffsetPing = mmNRatioOffset * GM_DOUBLE_BUFFER;
            vector1OffsetPong = vector1OffsetPing + mmNRatioOffset / 2;
            bmm1AndVec1Ratio = GM_DOUBLE_BUFFER + 1;
        } else {
            if (this->tilingData->inputParams.s2Size % 64 != 0) {
                vector1OffsetPing = mmNRatioOffset * GM_DOUBLE_BUFFER;
                vector1OffsetPong = vector1OffsetPing + mmNRatioOffset / 2;
                bmm1AndVec1Ratio = GM_DOUBLE_BUFFER + 1;
            }
        }
    }

    // FP32场景，stage1Result不与bmm1Result共用空间，需要占用2倍mmNRatioOffset空间
    if constexpr (IsSameType<INPUT_T, float>::value) {
        vector1OffsetPing = mmNRatioOffset * GM_DOUBLE_BUFFER;
        vector1OffsetPong = vector1OffsetPing + mmNRatioOffset;
        bmm1AndVec1Ratio = GM_DOUBLE_BUFFER + 2;
    }

    int64_t totalOffset = mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * GM_DOUBLE_BUFFER;
    if (dSizeAlign16 > 64) {
        totalOffset = mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 2 * GM_DOUBLE_BUFFER;
    }

    // bmm1Result，占用2倍mmNRatioOffset空间
    this->mm1Res[0].SetGlobalBuffer((__gm__ T *)(workspace + this->blockIdx * totalOffset));
    this->mm1Res[1].SetGlobalBuffer((__gm__ T *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset));

    // stage1Result，不占用/占用1倍/占用2倍mmNRatioOffset空间
    this->stage1Res[0].SetGlobalBuffer(
        (__gm__ INPUT_T *)(workspace + this->blockIdx * totalOffset + vector1OffsetPing));
    this->stage1Res[1].SetGlobalBuffer(
        (__gm__ INPUT_T *)(workspace + this->blockIdx * totalOffset + vector1OffsetPong));

    // bmm2Result，占用2倍mmOffset空间
    this->mm2Res[0].SetGlobalBuffer(
        (__gm__ T *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio));
    this->mm2Res[1].SetGlobalBuffer(
        (__gm__ T *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset));

    // vec2阶段，占用2倍mmOffset空间，仅在D轴大于64的情况下出现
    if (dSizeAlign16 > 64) {
        this->vec2Res[0].SetGlobalBuffer(
            (__gm__ T *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 2));
        this->vec2Res[1].SetGlobalBuffer(
            (__gm__ T *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 3));
    }
    if constexpr (IsSameType<T, half>::value) {
        this->negativeIntScalar = NEGATIVE_MIN_VAULE_FP16;
    }
    GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::SetTiling(const FusedFloydAttentionGeneralTilingData
                                                            *__restrict tilingData)
{
    // copy base params
    this->tilingData = tilingData;
    this->s1BaseSize = this->tilingData->coreParams.s1BaseSize;
    this->s2BaseSize = this->tilingData->coreParams.s2BaseSize;
    this->n2BaseSize = this->tilingData->coreParams.n2BaseSize;
    this->n2BaseTailSize = this->tilingData->coreParams.n2BaseTailSize;

    this->bmm1Num = this->tilingData->coreParams.bmm1Num;
    this->bmmv2Num = this->tilingData->coreParams.bmmv2Num;

    this->dSize = this->tilingData->inputParams.dSize;
    this->dSizeAlign16 = CeilDiv(this->tilingData->inputParams.dSize, 16) * 16;
};

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                     isBasicBlock, bmm1Format, enableL1Reuse>::InitBuffer()
{
    uint64_t stage1Size = 8 * 1024;
    uint64_t stage1AttenSize = 9 * 1024;
    uint64_t stage1PongSize = 35 * 1024;
    uint64_t stage2Size = 64 * 128;
    uint64_t maskTBufPongSize = 16 * 1024;

    // 可选输入的buffer空间，保持和stage1处理的size一致
    this->pipe->InitBuffer(this->maskTBufPing, stage1AttenSize); // 可以给attenmask 9k
    this->pipe->InitBuffer(this->maskTBufPong, maskTBufPongSize); // 可以给dropoutmask 16k
    this->pipe->InitBuffer(this->pseTBuf, 16384); // pse 16k

    this->pipe->InitBuffer(this->stage1PingBuf, stage2Size * sizeof(T)); // t.a 32k
    this->pipe->InitBuffer(this->stage2TBuf, stage2Size * sizeof(T));    // t.c 32k
    this->pipe->InitBuffer(this->commonTBuf, stage2Size * sizeof(T));    // t.b 32k

    this->pipe->InitBuffer(this->softmaxSumBuf[0], s1BaseSize * blockBytes); // 4k
    this->pipe->InitBuffer(this->softmaxSumBuf[1], s1BaseSize * blockBytes); // 4k
    this->pipe->InitBuffer(this->softmaxMaxBuf, s1BaseSize * blockBytes);    // 4k
    this->pipe->InitBuffer(this->softmaxExpBuf[0], s1BaseSize * blockBytes); // 4k
    this->pipe->InitBuffer(this->softmaxExpBuf[1], s1BaseSize * blockBytes); // 4k
    if constexpr (bmm1Format == CubeFormat::NZ) {
        this->pipe->InitBuffer(this->stage1PongBuf, stage1PongSize); // i.a 35k
    } else {
        this->pipe->InitBuffer(this->stage1PongBuf, stage1Size * sizeof(T)); // i.a 32k
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                     isBasicBlock, bmm1Format, enableL1Reuse>::ComputeConstexpr()
{
    // 计算轴的乘积
    if constexpr (enableL1Reuse) {
        this->s1OuterSize = this->tilingData->coreParams.s1OuterSize;
    }
    this->s1D = this->tilingData->inputParams.s1Size * dSize;
    this->s2D = this->tilingData->inputParams.s2Size * dSize;
    this->gD = this->tilingData->inputParams.gSize * dSize;
    this->n2D = this->tilingData->inputParams.n2Size * dSize;
    this->s1S2 = this->tilingData->inputParams.s1Size * this->tilingData->inputParams.s2Size;
    this->gS1 = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s1Size;
    this->n2G = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize;
    this->gS1o = this->tilingData->inputParams.gSize * this->tilingData->coreParams.s1OuterSize;

    this->bN2D = this->tilingData->inputParams.bSize * n2D;
    this->n2GS1o = this->tilingData->coreParams.n2OuterSize * this->gS1o;
    this->gS1D = this->tilingData->inputParams.gSize * this->s1D;
    this->n2S2D = this->tilingData->inputParams.n2Size * this->s2D;
    this->n2GD = this->tilingData->inputParams.n2Size * this->gD;
    this->bN2GD = this->tilingData->inputParams.bSize * n2GD;
    this->gS1S2 = this->tilingData->inputParams.gSize * this->s1S2;
    this->n2GS1 = this->tilingData->inputParams.n2Size * this->gS1;

    this->n2GS1D = this->tilingData->inputParams.n2Size * this->gS1D;
    this->n2GS1S2 = this->tilingData->inputParams.n2Size * this->gS1S2;

    // 计算切分轴的乘积
    this->s2BaseN2D = this->s2BaseSize * this->n2D;
    this->s2BaseNratioSize = this->s2BaseSize * this->tilingData->coreParams.nRatio;
    this->s1BaseS2 = this->s1BaseSize * this->tilingData->inputParams.s2Size;
    this->s1BaseD = this->s1BaseSize * this->dSize;
    this->s2BaseNratioD = this->s2BaseNratioSize * this->dSize;
    this->mm1Ka = this->dSize;
    this->mm1Kb = this->dSize;
    this->mm2Kb = this->dSize;
    this->attenMaskShapeType = this->tilingData->inputParams.attenMaskShapeType;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                     isBasicBlock, bmm1Format, enableL1Reuse>::Process()
{
    // 确定核内切分起点
    int64_t multiCoreInnerOffset = this->blockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    if constexpr (enableL1Reuse) {
        this->l1ReuseBlockMod2 = this->blockIdx % 2;
        multiCoreInnerOffset = this->blockIdx / 2 * this->tilingData->multiCoreParams.splitFactorSize;
    }
    int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize;
    if (this->tilingData->multiCoreParams.totalSize < multiCoreInnerLimit) {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
    }

    SplitExtraInfo extraInfo[3];
    int64_t taskId = 0;
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    bool notSecondLast = true;
    bool notLast = true;
    multiCoreInnerLimit += 2;

    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
            multiCoreInnerIdx++) {
        if (multiCoreInnerIdx == multiCoreInnerLimit - 2) {
            notSecondLast = false;
        } else if (multiCoreInnerIdx == multiCoreInnerLimit - 1) {
            notLast = false;
        }

        int64_t s2LoopLimit;
        this->softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
        bool notLastTwoLoop = notSecondLast && notLast;
        if (notLastTwoLoop) {
            this->ComputeAxisIdx(multiCoreInnerIdx);

            this->s2StartIdx = 0;
            this->s2EndIdx = this->s2Size;

            s2LoopLimit = CeilDiv(this->s2EndIdx - this->s2StartIdx, s2BaseNratioSize) - 1;
        } else {
            s2LoopLimit = 0;
        }
        
        for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
            if (notLastTwoLoop) {
                this->SetExtraInfo(extraInfo[taskId % 3], taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx,
                                    false);

                this->Bmm1SetTensorA(extraInfo[taskId % 3], this->bmm1);
                this->IterateBmm1k1(extraInfo[taskId % 3], this->bmm1k1);
                this->IterateBmm1(extraInfo[taskId % 3], this->bmm1);
            }

            if (taskId > 0 && notLast) {
                this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }

            if (taskId > 1) {
                // 对应extraInfo[(i+1)%3]
                WaitBmm2Result();
            }

            if (taskId > 0 && notLast) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                this->IterateBmm2forValue1(extraInfo[(taskId + 2) % 3]);
                this->IterateBmm2(extraInfo[(taskId + 2) % 3]);
            }

            if (taskId > 1) {
                this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
            }
            taskId++;
        }
    }
};

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::GetBNIdx(const int64_t &multiCoreInnerOffset,
                                                       const int64_t &multiCoreInnerLimit)
{
    bngStartIdx = multiCoreInnerOffset / this->tilingData->coreParams.s1OuterSize;
    bngEndIdx = CeilDiv(multiCoreInnerLimit, this->tilingData->coreParams.s1OuterSize);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::ComputeAxisIdx(int64_t multiCoreInnerIdx)
{
    // 计算轴的idx
    this->boIdx = multiCoreInnerIdx / this->n2GS1o;
    this->n2oIdx = multiCoreInnerIdx % this->n2GS1o / this->gS1o;
    this->goIdx = multiCoreInnerIdx % this->gS1o / this->tilingData->coreParams.s1OuterSize;
    this->s1oIdx = multiCoreInnerIdx % this->tilingData->coreParams.s1OuterSize;
    int64_t h =  this->tilingData->inputParams.bSize / this->tilingData->inputParams.attenbSize;
    
    this->atten_boIdx = multiCoreInnerIdx / this->n2GS1o / h;
    this->s1Size = this->tilingData->inputParams.s1Size;
    this->s2Size = this->tilingData->inputParams.s2Size;
}


template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::SetExtraInfo(SplitExtraInfo &extraInfo, int64_t taskId,
                                                           int64_t s2LoopCount, int64_t s2LoopLimit,
                                                           int64_t multiCoreInnerIdx, bool lastNotPair)
{
    extraInfo.s2StartIdx = this->s2StartIdx;
    extraInfo.s2EndIdx = this->s2EndIdx;
    extraInfo.s2LoopCount = s2LoopCount;
    extraInfo.s1oIdx = this->s1oIdx;
    extraInfo.boIdx = this->boIdx;
    extraInfo.n2oIdx = this->n2oIdx;
    extraInfo.atten_boIdx = this->atten_boIdx;
    extraInfo.goIdx = this->goIdx;
    extraInfo.taskId = taskId;
    extraInfo.taskIdMod2 = taskId % 2;
    extraInfo.s2LoopLimit = s2LoopLimit;
    extraInfo.multiCoreInnerIdx = multiCoreInnerIdx;
    extraInfo.multiCoreInnerIdxMod2 = multiCoreInnerIdx % 2;
    extraInfo.s1Size = this->tilingData->inputParams.s1Size;
    extraInfo.s2Size = this->tilingData->inputParams.s2Size;
    extraInfo.attenB1SSOffset = extraInfo.boIdx * this->s1S2;
    extraInfo.attenMaskS2Size = this->tilingData->inputParams.attenMaskS2Size;
    extraInfo.s1SizeAcc = extraInfo.boIdx * extraInfo.s1Size;
    if constexpr (enableL1Reuse) {
        extraInfo.lastNotPair = lastNotPair;
    }

    extraInfo.s1RealSize = Min(s1BaseSize, this->tilingData->inputParams.s1Size - extraInfo.s1oIdx * s1BaseSize);
    
    this->ComputeBmm1Tail(extraInfo);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::ComputeBmm1Tail(SplitExtraInfo &extraInfo)
{
    if (this->tilingData->inputParams.s1Size < (extraInfo.s1oIdx + 1) * this->s1BaseSize) {
        extraInfo.s1RealSize = this->tilingData->inputParams.s1Size - extraInfo.s1oIdx * this->s1BaseSize;
    }
    extraInfo.s2RealSize = this->s2BaseNratioSize;
    extraInfo.s2AlignedSize = extraInfo.s2RealSize;
    if (extraInfo.s2StartIdx + (extraInfo.s2LoopCount + 1) * extraInfo.s2RealSize > extraInfo.s2EndIdx) {
        extraInfo.s2RealSize = extraInfo.s2EndIdx - extraInfo.s2LoopCount * extraInfo.s2RealSize - extraInfo.s2StartIdx;
        extraInfo.s2AlignedSize = Align(extraInfo.s2RealSize);
    }

    if(extraInfo.n2oIdx == this->tilingData->coreParams.n2OuterSize-1){
        extraInfo.n2RealSize = this->n2BaseTailSize;
    }else{
        extraInfo.n2RealSize = this->n2BaseSize;
    }

    if(extraInfo.s2Size <= 1024){
        extraInfo.vec1S1BaseSize = Min(s2BaseNratioSize / extraInfo.s2AlignedSize * 8, extraInfo.s1RealSize);
    }else{
        extraInfo.vec1S1BaseSize = 4;
    }
    extraInfo.realSplitN = CeilDiv(extraInfo.s1RealSize, extraInfo.vec1S1BaseSize);

    if (dSizeAlign16 > 64) {
        extraInfo.vec2S1BaseSize = 64 * 128 / dSizeAlign16;
    } else {
        extraInfo.vec2S1BaseSize = extraInfo.s1RealSize;
    }
    if constexpr (bmm1Format == CubeFormat::NZ) {
        extraInfo.needNz2Nd = (extraInfo.s2RealSize % 64 == 0) ? 0 : 1;
    } else {
        extraInfo.needNz2Nd = 0;
    }
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
template <typename T2, const MatmulConfig &MM_CFG>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::IterateBmm1(SplitExtraInfo &extraInfo,
                                                          matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG> &bmm1)
{
    uint32_t loops = CeilDiv(extraInfo.n2RealSize, this->bmm1Num);
    int64_t matrixStrideA = dSize * this->s1Size;

    for (uint32_t idx = 0; idx < loops; ++idx) {
        uint32_t batchSizes = idx * this->bmm1Num;
        int64_t batchOffsetD = batchSizes * this->s1D;
        bmm1.SetTensorA(this->queryGm[extraInfo.qCoreOffset + batchOffsetD]);
        SetBmm1TensorB(extraInfo, bmm1, batchSizes);

        int64_t outOffset = batchSizes * extraInfo.s2RealSize * extraInfo.s1RealSize;
        uint32_t realBatchNum = Min(this->bmm1Num, extraInfo.n2RealSize - batchSizes);
        bmm1.IterateBatch(this->mm1Res[extraInfo.taskIdMod2][outOffset], realBatchNum, realBatchNum, false, matrixStrideA, 0, 0, false, 1);
    }

    bmm1.End();
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
template <typename T2, const MatmulConfig &MM_CFG>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::IterateBmm1k1(SplitExtraInfo &extraInfo,
                              matmul::Matmul<a3Type, b3Type, T2, bias1Type, MM_CFG> &bmm1mix)
{
    uint32_t loops = (extraInfo.s1RealSize + bmmv2Num - 1) / bmmv2Num;
    uint32_t loop_rem = extraInfo.s1RealSize % bmmv2Num;
    uint32_t realBatchNum = bmmv2Num;

    bmm1mix.SetTail(extraInfo.n2RealSize, -1, -1);
    for (uint32_t idx = 0; idx < loops; ++idx) {
        if (idx == loops - 1 && loop_rem != 0)
        {
            realBatchNum = loop_rem;
        }

        int64_t batchOffsetD = bmmv2Num * idx * this->dSize;
        bmm1mix.SetTensorA(this->queryGm[extraInfo.qCoreOffset + batchOffsetD]);

        int64_t kCoreOffset = extraInfo.k1CoreOffset + batchOffsetD;
        bmm1mix.SetTensorB(this->keyGm1[kCoreOffset], true);
        bmm1mix.IterateBatch(this->mm1Res[extraInfo.taskIdMod2][bmmv2Num * idx * extraInfo.s2RealSize], realBatchNum, realBatchNum, false);
    }

    bmm1mix.End();
}


template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
template <typename T2, const MatmulConfig &MM_CFG>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::Bmm1SetTensorA(SplitExtraInfo &extraInfo,
                                                             matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG>
                                                                 &bmm1)
{
    // 计算gm上的offset
    int64_t bOffset = 0;

    // s1需要考虑inner轴的影响
    int64_t s1Offset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    // BNSD
    bOffset = extraInfo.boIdx * this->n2GS1D;
    n2Offset = extraInfo.n2oIdx * this->gS1D * this->n2BaseSize;
    gOffset = extraInfo.goIdx * this->s1D;
    s1Offset = extraInfo.s1oIdx * this->s1BaseD;
    this->qCoreOffset = bOffset + n2Offset + gOffset + s1Offset;
    extraInfo.qCoreOffset = this->qCoreOffset;

    // BNSD
    int64_t bs1Offset = extraInfo.boIdx * this->s1D * this->s2Size;
    int64_t s2Offset = extraInfo.s2LoopCount * this->s2BaseNratioSize * this->s1D; 
    extraInfo.k1CoreOffset = bs1Offset + s2Offset + s1Offset;
}



template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
template <typename T2, const MatmulConfig &MM_CFG>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::SetBmm1TensorB(SplitExtraInfo &extraInfo,
                                                             matmul::Matmul<a1Type, b1Type, T2, bias1Type, MM_CFG> &bmm1,
                                                             int64_t loopN2
                                                            )
{
    // 计算gm上的offset
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    // BNSD
    bOffset = extraInfo.boIdx * this->n2S2D;
    n2Offset = (extraInfo.n2oIdx * this->n2BaseSize + loopN2) * this->s2D;
    s2Offset = extraInfo.s2StartIdx * dSize + extraInfo.s2LoopCount * this->s2BaseNratioD;
    int64_t kCoreOffset = bOffset + n2Offset + s2Offset;
    bmm1.SetTensorB(this->keyGm[kCoreOffset], true);
    bmm1.SetTail(extraInfo.s1RealSize, extraInfo.s2RealSize);
}



template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::ProcessVec1(SplitExtraInfo &extraInfo)
{
    if constexpr (enableL1Reuse) {
        if (extraInfo.lastNotPair) {
            return;
        }
    }
    LocalTensor<T> stage1PingTensor = this->stage1PingBuf.template Get<T>(); // t.a 32k
    LocalTensor<T> stage1PongTensor = this->stage1PongBuf.template Get<T>(); // i.a 32k
    LocalTensor<T> &actualUseTensor = (extraInfo.needNz2Nd == 0) ? stage1PongTensor : stage1PingTensor;
    LocalTensor<T> commonTBuf = this->commonTBuf.template Get<T>(); // t.b 32k

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t eventIdVToMte2A = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdVToMte2B = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdVToMte2C = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventIdDropMte3ToMte2;
    extraInfo.vec1S1RealSize = extraInfo.vec1S1BaseSize;


    for (int32_t loopN2 = 0; loopN2 < extraInfo.n2RealSize; ++loopN2) {

    for (int32_t loopIdx = 0; loopIdx < extraInfo.realSplitN; loopIdx++) {
        if (loopIdx == extraInfo.realSplitN - 1) {
            extraInfo.vec1S1RealSize = extraInfo.s1RealSize - loopIdx * extraInfo.vec1S1BaseSize;
        }
        if (loopIdx > 0) {
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
        } else {
            if constexpr (IsSameType<T, INPUT_T>::value == false && layOutType == LayOutTypeEnum::LAYOUT_TND) {
                event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            }
        }

        // FP32场景，需要等待vec1上一轮输出搬完
        if constexpr (IsSameType<INPUT_T, float>::value) {
            event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        this->GetBmm1Result(extraInfo, actualUseTensor, loopIdx, loopN2);

        // mul需要等bmm结果搬完
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        this->CopyInAttenMask(extraInfo, loopIdx, -1, loopN2);

        AscendC::PipeBarrier<PIPE_V>();
        Muls(stage1PingTensor, actualUseTensor, static_cast<T>(this->tilingData->inputParams.scaleValue),
        extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize);

        if constexpr (hasAtten) {
            SelectWithBytesMaskShapeInfo shapeInfo;
            shapeInfo.firstAxis = extraInfo.vec1S1RealSize;
            shapeInfo.srcLastAxis = extraInfo.s2AlignedSize;
            shapeInfo.maskLastAxis = CeilDiv(extraInfo.s2RealSize, blockBytes) * blockBytes;
            stage1PingTensor.SetSize(extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize);

            uint8_t maskType = 0;
            LocalTensor<uint8_t> attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
            this->ComputeAttenMask(shapeInfo, stage1PingTensor, attenMaskUb, maskType, eventIdMte2ToV);
        }
        if (loopIdx < extraInfo.realSplitN - 1) {
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
        }

        if (loopIdx > 0) {
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
        }

        this->SoftMaxCompute(extraInfo, stage1PingTensor, loopIdx, loopN2);
        if (loopIdx < extraInfo.realSplitN - 1) {
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
        }

        if (loopIdx > 0) {
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        }
        AscendC::PipeBarrier<PIPE_V>();

        LocalTensor<INPUT_T> stage1CastTensor;
        stage1CastTensor = this->pseTBuf.template Get<INPUT_T>();
        Cast(stage1CastTensor, stage1PingTensor, RoundMode::CAST_ROUND,
                extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        // 注意这里如果S1有尾块
        int64_t vecS1S2Align = extraInfo.vec1S1BaseSize * extraInfo.s2AlignedSize;
        int64_t n2LoopIOffset = (loopN2 * extraInfo.realSplitN + loopIdx) * vecS1S2Align;
        DataCopy(
            this->stage1Res[extraInfo.taskIdMod2][n2LoopIOffset],
            stage1CastTensor, extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize);
        
        if (loopIdx < extraInfo.realSplitN - 1) {
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        }
    }

    if (extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
        SoftmaxDataCopyOut(extraInfo, loopN2);
    }

    }

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2A);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2B);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2C);
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::CopyInAttenMask(SplitExtraInfo &extraInfo, int64_t loopIdx,
                                                              int64_t maskOffset, int64_t loopN2, bool secondTime)
{
    if constexpr (hasAtten == true) {
        LocalTensor<uint8_t> attenMaskUb;
        if (secondTime) {
            attenMaskUb = this->pseTBuf.template Get<uint8_t>();
        } else {
            attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
        }
        if (maskOffset == -1) {
            maskOffset = this->ComputeOffsetForNoCompress(extraInfo, loopIdx, loopN2);
        }
        int64_t s2StrideSize = this->tilingData->inputParams.attenMaskS2Size;
        AttenMaskBoolCopyIn(attenMaskUb, this->attenMaskGmInt, maskOffset, extraInfo.vec1S1RealSize, extraInfo.s2RealSize,
                   s2StrideSize, attenMaskShapeType);
        return;
    }
}


template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline int64_t
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::ComputeOffsetForNoCompress(SplitExtraInfo &extraInfo, int64_t loopIdx, int64_t loopN2)
{
    if constexpr (hasAtten == true) {
        int64_t bOffset = 0;
        int64_t n2Offset = 0;
        int64_t gOffset = 0;
        int64_t s1Offset = extraInfo.s1oIdx * this->s1BaseSize * extraInfo.s2Size +
                           loopIdx * extraInfo.vec1S1BaseSize * extraInfo.s2Size;
        int64_t s2Offset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * s2BaseNratioSize;

        if (attenMaskShapeType == 0) {
            bOffset = extraInfo.attenB1SSOffset * this->n2G;
            n2Offset = (extraInfo.n2oIdx * this->n2BaseSize + loopN2) * this->tilingData->inputParams.gSize * extraInfo.s1Size * extraInfo.s2Size;
            gOffset = extraInfo.goIdx * extraInfo.s1Size * extraInfo.s2Size;
        } else if (attenMaskShapeType == 3) {
            bOffset = extraInfo.atten_boIdx * this->n2G * this->tilingData->inputParams.s2Size;
            n2Offset = (extraInfo.n2oIdx * this->n2BaseSize+ loopN2) * extraInfo.s2Size;
            s1Offset = 0;
        }
        return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::GetBmm1Result(SplitExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb,
                                                            int64_t loopIdx, int64_t loopN2)
{
    int64_t n2BaseOffset = loopN2 * extraInfo.s1RealSize * extraInfo.s2RealSize;
    if constexpr (bmm1Format == CubeFormat::NZ) {
        if (extraInfo.needNz2Nd == 1) {
            Nz2NdInfo nz2NdInfo;
            nz2NdInfo.ndFirstAxisRealSize = extraInfo.s1RealSize;
            nz2NdInfo.ndFirstAxisBaseSize = extraInfo.vec1S1BaseSize;
            nz2NdInfo.ndFirstAxisLoopSize = extraInfo.vec1S1RealSize;
            nz2NdInfo.ndLastAxis = extraInfo.s2AlignedSize;
            nz2NdInfo.loopIdx = loopIdx;
            LocalTensor<T> tempUb = this->stage1PongBuf.template Get<T>();
            NzToNd(nz2NdInfo, this->mm1Res[extraInfo.taskIdMod2], tempUb, bmm1ResUb);
            return;
        }
    }
    if (likely(extraInfo.s2AlignedSize == extraInfo.s2RealSize)) {
        DataCopy2D(bmm1ResUb,
                   this->mm1Res[extraInfo.taskIdMod2][n2BaseOffset + loopIdx * extraInfo.vec1S1BaseSize * extraInfo.s2RealSize],
                   extraInfo.vec1S1RealSize, extraInfo.s2RealSize, extraInfo.s2RealSize);

    } else {
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = extraInfo.vec1S1RealSize;
        dataCopyParams.blockLen = extraInfo.s2RealSize * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPadParams dataCopyPadParams;
        dataCopyPadParams.isPad = true;
        dataCopyPadParams.rightPadding = extraInfo.s2AlignedSize - extraInfo.s2RealSize;
        if (dataCopyPadParams.rightPadding > blockSize) {
            dataCopyPadParams.rightPadding -= blockSize;
            dataCopyParams.dstStride = 1;
            int32_t s2BlockAlignedSize = CeilDiv(extraInfo.s2RealSize, blockSize) * blockSize;
            Duplicate<T>(bmm1ResUb[s2BlockAlignedSize], 0, blockSize, extraInfo.vec1S1RealSize, 0,
                         extraInfo.s2AlignedSize * sizeof(T) / blockBytes);
        }
        dataCopyPadParams.paddingValue = 0;
        DataCopyPad(bmm1ResUb,
                    this->mm1Res[extraInfo.taskIdMod2][n2BaseOffset + loopIdx * extraInfo.vec1S1BaseSize * extraInfo.s2RealSize],
                    dataCopyParams, dataCopyPadParams);
    }
    uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                                 static_cast<uint32_t>(extraInfo.s2AlignedSize)};
    bmm1ResUb.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, DataFormat::ND));
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::ComputeAttenMask(SelectWithBytesMaskShapeInfo &shapeInfo,
                                                               LocalTensor<T> &bmm1ResUb,
                                                               LocalTensor<uint8_t> &attenMaskUb,
                                                               const uint8_t maskType, event_t vWaitMte2)
{
    if constexpr (hasAtten == true) {
        LocalTensor<uint8_t> apiTmpBuffer = commonTBuf.template Get<uint8_t>();
        attenMaskUb.SetSize(shapeInfo.firstAxis * shapeInfo.maskLastAxis);
        SetFlag<HardEvent::MTE2_V>(vWaitMte2);
        WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        if (maskType == 0) {
            SelectWithBytesMask(bmm1ResUb, bmm1ResUb, this->negativeFloatScalar, attenMaskUb, apiTmpBuffer, shapeInfo);
        } else {
            SelectWithBytesMask(bmm1ResUb, this->negativeFloatScalar, bmm1ResUb, attenMaskUb, apiTmpBuffer, shapeInfo);
        }
        return;
    }
}


template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::SoftMaxCheckResCompress(SplitExtraInfo &extraInfo,
                                                                      int64_t vec1S1realSplitN)
{
    if (unlikely(extraInfo.realSplitN == 1)) {
        bool res = IsIncludeInvalidLine(this->softMaxCheckRes, vec1S1realSplitN);
        UpdateSoftMaxCheckRes(this->softMaxCheckRes, 0, res);
    } else {
        int64_t avg = CeilDiv(vec1S1realSplitN, extraInfo.realSplitN);
        for (int64_t i = 0; i < extraInfo.realSplitN; ++i) {
            int64_t endIdx = Min((i + 1) * avg, vec1S1realSplitN) - 1;
            if (IsIncludeInvalidLine(this->softMaxCheckRes, endIdx, i * avg)) {
                UpdateSoftMaxCheckRes(this->softMaxCheckRes, i, true);
            } else {
                UpdateSoftMaxCheckRes(this->softMaxCheckRes, i, false);
            }
        }
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::InvalidLineSplitS2Process(SplitExtraInfo &extraInfo,
                                                                        LocalTensor<T> &srcTensor,
                                                                        LocalTensor<T> &maxUb, int64_t loopIdx)
{
    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION) {
        if (loopIdx == 0 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
            int64_t vec1S1BaseSize = Min(8, extraInfo.s1RealSize);
            int64_t vec1S1realSplitN = CeilDiv(extraInfo.s1RealSize, vec1S1BaseSize);
            if (extraInfo.realSplitN != vec1S1realSplitN) {
                this->SoftMaxCheckResCompress(extraInfo, vec1S1realSplitN);
            }
        }

        if (hasInvalidLine(this->softMaxCheckRes, loopIdx)) {
            SoftMaxShapeInfo softmaxShapeInfo{
                static_cast<uint32_t>(extraInfo.vec1S1RealSize), static_cast<uint32_t>(extraInfo.s2AlignedSize),
                static_cast<uint32_t>(extraInfo.vec1S1RealSize), static_cast<uint32_t>(extraInfo.s2RealSize)};
            bool res = AdjustSoftMaxRes<T, T>(srcTensor, maxUb, this->negativeIntScalar, 0.0, softmaxShapeInfo);
            UpdateSoftMaxCheckRes(this->softMaxCheckRes, loopIdx, res);
        }

        if (loopIdx == extraInfo.realSplitN - 1 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit &&
            IsIncludeInvalidLine(this->softMaxCheckRes, extraInfo.realSplitN)) {
            LocalTensor<T> maxTensor = this->softmaxMaxBuf.template Get<T>();
            LocalTensor<T> sumTensor;
            sumTensor = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<T>();

            SoftMaxShapeInfo softmaxShapeInfo{
                static_cast<uint32_t>(extraInfo.s1RealSize), static_cast<uint32_t>(fp32BaseSize),
                static_cast<uint32_t>(extraInfo.s1RealSize), static_cast<uint32_t>(fp32BaseSize)};
            AdjustSoftMaxRes<T, T>(sumTensor, maxTensor, this->negativeIntScalar, this->positiveFloatScalar,
                                   softmaxShapeInfo);
        }
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::SoftMaxCompute(SplitExtraInfo &extraInfo, LocalTensor<T> &srcTensor,
                                                             int64_t loopIdx, int64_t loopN2)
{
    uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                                 static_cast<uint32_t>(extraInfo.s2AlignedSize)};
    uint32_t bmm1ResUbOrgShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                                    static_cast<uint32_t>(extraInfo.s2RealSize)};
    srcTensor.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, 2, bmm1ResUbOrgShape, DataFormat::ND));

    uint32_t maxSumShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize), static_cast<uint32_t>(fp32BaseSize)};
    LocalTensor<T> sumUb;
    sumUb = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2]
                .template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * fp32BaseSize];
    LocalTensor<T> maxUb = this->softmaxMaxBuf.template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * fp32BaseSize];

    sumUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    maxUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    LocalTensor<T> expUb;
    expUb = this->softmaxExpBuf[extraInfo.taskIdMod2]
                .template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * blockBytes / sizeof(T)];

    expUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
    AscendC::PipeBarrier<PIPE_V>();
    if (unlikely(extraInfo.s2LoopCount == 0)) {
        if (IsBasicBlockInSoftMax(extraInfo.vec1S1RealSize, extraInfo.s2RealSize)) {
            SoftMaxTiling newTiling = AscendC::SoftMaxFlashV2TilingFuncImpl(extraInfo.vec1S1RealSize,
                                                                            extraInfo.s2AlignedSize, sizeof(T),
                                                                            sizeof(T),
                                                                            apiTmpBuffer.GetSize() / sizeof(T),
                                                                            false, true);
            SoftmaxFlashV2<T, false, false, true, false, SOFTMAX_DEFAULT_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                             sumUb, maxUb, apiTmpBuffer, newTiling);
        } else {
            SoftMaxTiling newTiling = AscendC::SoftMaxFlashV2TilingFuncImpl(extraInfo.vec1S1RealSize,
                                                                            extraInfo.s2AlignedSize, sizeof(T),
                                                                            sizeof(T),
                                                                            apiTmpBuffer.GetSize() / sizeof(T),
                                                                            false, false);
            SoftmaxFlashV2<T, false, false, false, false, SOFTMAX_DEFAULT_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                             sumUb, maxUb, apiTmpBuffer, newTiling);
        }
    } else {
        if (IsBasicBlockInSoftMax(extraInfo.vec1S1RealSize, extraInfo.s2RealSize)) {
            SoftMaxTiling newTiling = AscendC::SoftMaxFlashV2TilingFuncImpl(extraInfo.vec1S1RealSize,
                                                                            extraInfo.s2AlignedSize, sizeof(T),
                                                                            sizeof(T),
                                                                            apiTmpBuffer.GetSize() / sizeof(T),
                                                                            true, true);
            SoftmaxFlashV2<T, true, false, true, false, SOFTMAX_DEFAULT_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                             sumUb, maxUb, apiTmpBuffer, newTiling);
        } else {
            SoftMaxTiling newTiling = AscendC::SoftMaxFlashV2TilingFuncImpl(extraInfo.vec1S1RealSize,
                                                                            extraInfo.s2AlignedSize, sizeof(T),
                                                                            sizeof(T),
                                                                            apiTmpBuffer.GetSize() / sizeof(T),
                                                                            true, false);
            SoftmaxFlashV2<T, true, false, false, false, SOFTMAX_DEFAULT_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                             sumUb, maxUb, apiTmpBuffer, newTiling);
        }
    }

    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION) {
        if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
            if (extraInfo.s2Size > SPLIT_S2_SIZE_LIMIT) {
                this->InvalidLineSplitS2Process(extraInfo, srcTensor, maxUb, loopIdx);
            } else {
                SoftMaxShapeInfo softmaxShapeInfo{
                    static_cast<uint32_t>(extraInfo.vec1S1RealSize), static_cast<uint32_t>(extraInfo.s2AlignedSize),
                    static_cast<uint32_t>(extraInfo.vec1S1RealSize), static_cast<uint32_t>(extraInfo.s2RealSize)};
                AdjustSoftMaxRes<T, T>(srcTensor, maxUb, this->negativeIntScalar, 0.0, softmaxShapeInfo);
                if (loopIdx == extraInfo.realSplitN - 1 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
                    LocalTensor<T> maxTensor = this->softmaxMaxBuf.template Get<T>();
                    LocalTensor<T> sumTensor;
                    sumTensor = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<T>();

                    SoftMaxShapeInfo softmaxFullShapeInfo{
                        static_cast<uint32_t>(extraInfo.s1RealSize), static_cast<uint32_t>(fp32BaseSize),
                        static_cast<uint32_t>(extraInfo.s1RealSize), static_cast<uint32_t>(fp32BaseSize)};
                    AdjustSoftMaxRes<T, T>(sumTensor, maxTensor, this->negativeIntScalar, this->positiveFloatScalar,
                                           softmaxFullShapeInfo);
                }
            }
        } else {
            this->InvalidLineSplitS2Process(extraInfo, srcTensor, maxUb, loopIdx);
        }
    }

    if (loopIdx == extraInfo.realSplitN - 1 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
        extraInfo.softmaxMaxOffset =
            (extraInfo.s1SizeAcc * this->n2G +
             extraInfo.n2oIdx * this->n2BaseSize * this->tilingData->inputParams.gSize * extraInfo.s1Size +
             extraInfo.goIdx * extraInfo.s1Size + extraInfo.s1oIdx * static_cast<int64_t>(s1BaseSize)) *
            static_cast<int64_t>(fp32BaseSize);
        int64_t calculateSize = extraInfo.s1RealSize * fp32BaseSize;
        LocalTensor<float> maxTensor = this->softmaxMaxBuf.template Get<float>();
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopy(this->softmaxMaxGm[extraInfo.softmaxMaxOffset + loopN2 * extraInfo.s1Size * fp32BaseSize], maxTensor, calculateSize);
    }
}


template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                     isBasicBlock, bmm1Format, enableL1Reuse>::WaitBmm2Result()
{
    this->bmm2.WaitIterateBatch();
    this->bmm2.End();
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::IterateBmm2(SplitExtraInfo &extraInfo)
{
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    // BNSD
    bOffset = extraInfo.boIdx * this->n2S2D;
    n2Offset = extraInfo.n2oIdx * this->s2D * this->n2BaseSize;
    s2Offset = extraInfo.s2StartIdx * dSize + extraInfo.s2LoopCount * s2BaseNratioSize * dSize;

    int64_t vCoreOffset = bOffset + n2Offset + s2Offset;

    uint32_t batchNum = extraInfo.n2RealSize;
    this->bmm2.SetTensorA(this->stage1Res[extraInfo.taskIdMod2]);
    this->bmm2.SetTensorB(this->valueGm[vCoreOffset]);
    this->bmm2.template IterateBatch<false, true>(this->mm2Res[extraInfo.taskIdMod2], batchNum, batchNum, false, 0, 0, 0, false, 1);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::IterateBmm2forValue1(SplitExtraInfo &extraInfo)
{
    uint32_t loops = (extraInfo.s1RealSize + bmmv2Num - 1) / bmmv2Num;
    uint32_t loop_rem = extraInfo.s1RealSize % bmmv2Num;
    uint32_t realBatchNum = bmmv2Num;

    this->bmm2v2.SetTail(extraInfo.n2RealSize, -1, -1);
    for (uint32_t idx = 0; idx < loops; ++idx)
    {
        if (idx == loops - 1 && loop_rem != 0)
        {
            realBatchNum = loop_rem;
        }

        int64_t batchOffsetD = bmmv2Num * idx * this->dSize;
        this->bmm2v2.SetTensorA(this->stage1Res[extraInfo.taskIdMod2][bmmv2Num * idx * extraInfo.s2AlignedSize]);
        this->bmm2v2.SetTensorB(this->valueGm1[extraInfo.k1CoreOffset + batchOffsetD]);
        this->bmm2v2.IterateBatch(this->mm2Res[extraInfo.taskIdMod2][batchOffsetD], realBatchNum, realBatchNum, false);
    }

    this->bmm2v2.End();
}


template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::ProcessVec2(SplitExtraInfo &extraInfo)
{
    if constexpr (enableL1Reuse) {
        if (extraInfo.lastNotPair) {
            return;
        }
    }
    // 获取缓存bmm2的计算结果
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
    LocalTensor<T> stage2BufTensor = this->commonTBuf.template Get<T>();
    int64_t vec2LoopLimit = CeilDiv(extraInfo.s1RealSize, extraInfo.vec2S1BaseSize);

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    extraInfo.vec2S1RealSize = extraInfo.vec2S1BaseSize;

    
    for (int64_t loopN2 = 0; loopN2 < extraInfo.n2RealSize; ++loopN2) {
        LocalTensor<float> sumUb;
        sumUb = softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].Get<float>();

        // s1Size 没有对齐
        int64_t vec2S1N2Offset = loopN2 * extraInfo.s1Size * fp32BaseSize;
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        DataCopy(sumUb, this->softmaxSumGm[extraInfo.softmaxMaxOffset + vec2S1N2Offset],  extraInfo.s1RealSize * fp32BaseSize);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    for (int64_t s1oIdx = 0; s1oIdx < vec2LoopLimit; s1oIdx++) {
        if (s1oIdx == vec2LoopLimit - 1) {
            extraInfo.vec2S1RealSize = extraInfo.s1RealSize - s1oIdx * extraInfo.vec2S1BaseSize;
        }
        int64_t mm2ResCalcSize = extraInfo.vec2S1RealSize * dSize;
        int64_t n2Offset = loopN2 * extraInfo.s1RealSize * dSize;
        int64_t mm2ResOffset = s1oIdx * extraInfo.vec2S1BaseSize * dSize;
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        int64_t dAlign8 = (this->dSize + 7) / 8 * 8;

        if (likely(this->dSizeAlign16 == this->dSize)) {
            DataCopy(stage2BufTensor, this->mm2Res[extraInfo.taskIdMod2][n2Offset + mm2ResOffset], mm2ResCalcSize);
        } else {
            DataCopyParams dataCopyParams;
            DataCopyPadParams dataCopyPadParams;
            dataCopyParams.blockCount = extraInfo.vec2S1RealSize;
            dataCopyParams.dstStride = 0;
            dataCopyParams.srcStride = 0;
            dataCopyParams.blockLen = this->dSize * 4;
            dataCopyPadParams.rightPadding = this->dSizeAlign16 - this->dSize;
            dataCopyPadParams.paddingValue = 0;
            if (dataCopyPadParams.rightPadding > blockSize) {
                // 8对齐场景，内部vector需要16对齐，我们在data copy的时候需要手动补0
                dataCopyPadParams.rightPadding -= blockSize;
                dataCopyParams.dstStride = 1;
                Duplicate<T>(stage2BufTensor[dAlign8], 0, blockSize, extraInfo.vec2S1RealSize, 0,
                                this->dSizeAlign16 * sizeof(T) / blockBytes);
            }
            DataCopyPad(stage2BufTensor, this->mm2Res[extraInfo.taskIdMod2][n2Offset + mm2ResOffset], dataCopyParams,
                        dataCopyPadParams);
            mm2ResCalcSize = extraInfo.vec2S1RealSize * dSizeAlign16;
            mm2ResOffset = s1oIdx * extraInfo.vec2S1BaseSize * dSizeAlign16;
        }

        if (vec2LoopLimit > 1) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            DataCopy(bmm2ResUb, this->vec2Res[extraInfo.multiCoreInnerIdxMod2][mm2ResOffset], mm2ResCalcSize);
        }

        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        if (unlikely(extraInfo.s2LoopCount == 0)) {
            DataCopy(bmm2ResUb, stage2BufTensor, mm2ResCalcSize);
        } else {
            this->Bmm2ResultMul(extraInfo, bmm2ResUb, s1oIdx);
            AscendC::PipeBarrier<PIPE_V>();
            Add(bmm2ResUb, bmm2ResUb, stage2BufTensor, mm2ResCalcSize);
        }

        if (extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
            Bmm2ResultDiv(extraInfo, s1oIdx);
            Bmm2DataCopyOut(extraInfo, s1oIdx, mm2ResCalcSize, loopN2);
            event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        } else if (vec2LoopLimit > 1) {
            SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            DataCopy(this->vec2Res[extraInfo.multiCoreInnerIdxMod2][mm2ResOffset], bmm2ResUb, mm2ResCalcSize);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
    }

    }
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::Bmm2ResultMul(SplitExtraInfo &extraInfo, LocalTensor<T> &bmm2ResUb,
                                                            int64_t s1oIdx)
{
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<T> expUb;
    expUb = softmaxExpBuf[extraInfo.taskIdMod2].Get<T>();

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 0;
    repeatParams.src0RepStride = 1;
    repeatParams.src1RepStride = dSizeAlign16 / blockSize;
    repeatParams.dstRepStride = dSizeAlign16 / blockSize;

    // s1长度可能会超过255限制，修改成双重循环
    // 根据一次最多计算的byte数量，对bmm2Res分组mul
    int32_t loop = dSizeAlign16 / repeatMaxSize;
    int32_t remain = dSizeAlign16 % repeatMaxSize;
    for (int i = 0; i < loop; i++) {
        Mul(bmm2ResUb[i * repeatMaxSize], expUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], bmm2ResUb[i * repeatMaxSize],
            repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
    }
    if (likely(remain)) {
        Mul(bmm2ResUb[loop * repeatMaxSize], expUb[s1oIdx * extraInfo.vec2S1BaseSize * 8],
            bmm2ResUb[loop * repeatMaxSize], remain, extraInfo.vec2S1RealSize, repeatParams);
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::Bmm2ResultDiv(SplitExtraInfo &extraInfo, int64_t s1oIdx)
{
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src0RepStride = dSizeAlign16 / blockSize;
    repeatParams.src1BlkStride = 0;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = dSizeAlign16 / blockSize;
    int32_t loop = dSizeAlign16 / repeatMaxSize;
    int32_t remain = dSizeAlign16 % repeatMaxSize;

    LocalTensor<float> sumUb;
    sumUb = softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].Get<float>();

    int32_t calcSize = sumUb.GetSize();
    // 用optionalInputQueue的queue
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<T, half>::value) {
        LocalTensor<float> commonBufTensor = this->commonTBuf.template Get<float>();
        Copy(commonBufTensor, sumUb, 64, calcSize / 64, {2, 1, 16, 8});
        Copy(commonBufTensor[8], sumUb, 64, calcSize / 64, {2, 1, 16, 8});
        LocalTensor<T> sumCastTensor = softmaxExpBuf[0].Get<T>(this->tilingData->tensorSizeParams.softmaxExpUbSize);

        Cast(sumCastTensor, commonBufTensor, RoundMode::CAST_ROUND, 2 * sumUb.GetSize());
        for (int i = 0; i < loop; i++) {
            Div(bmm2ResUb[i * repeatMaxSize], bmm2ResUb[i * repeatMaxSize], sumCastTensor, repeatMaxSize,
                extraInfo.vec2S1RealSize, repeatParams);
        }
        if (likely(remain)) {
            Div(bmm2ResUb[loop * repeatMaxSize], bmm2ResUb[loop * repeatMaxSize], sumCastTensor, remain,
                extraInfo.vec2S1RealSize, repeatParams);
        }
    } else {
        for (int i = 0; i < loop; i++) {
            Div(bmm2ResUb[i * repeatMaxSize], bmm2ResUb[i * repeatMaxSize],
                sumUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
        }
        if (likely(remain)) {
            Div(bmm2ResUb[loop * repeatMaxSize], bmm2ResUb[loop * repeatMaxSize],
                sumUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], remain, extraInfo.vec2S1RealSize, repeatParams);
        }
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::Bmm2DataCopyOut(SplitExtraInfo &extraInfo, int64_t s1oIdx,
                                                              int64_t mm2ResCalcSize, int64_t loopN2)
{
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
    LocalTensor<INPUT_T> attenOut = this->stage2TBuf.template Get<INPUT_T>();
    bmm2ResUb.SetSize(mm2ResCalcSize);
    AscendC::PipeBarrier<PIPE_V>();

    if constexpr (!IsSameType<INPUT_T, T>::value) {
        Cast(attenOut, bmm2ResUb, RoundMode::CAST_ROUND, mm2ResCalcSize);
    } else {
        DataCopy(attenOut, bmm2ResUb, mm2ResCalcSize);
    }

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = this->dSize * sizeof(INPUT_T);
    dataCopyParams.srcStride = 0;
    if constexpr (IsSameType<INPUT_T, float>::value) {
        if (this->dSizeAlign16 - this->dSize >= blockSize) {
            dataCopyParams.srcStride = 1;
        }
    }
    int64_t dstStride = 0;
    int64_t attenOutOffset = this->dSize;
    int64_t datacopyOffset = this->dSize;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        datacopyOffset = this->n2GD;
        attenOutOffset = this->n2GD;
        dstStride = (this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize - 1) * this->dSize *
                    sizeof(INPUT_T);
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        datacopyOffset = this->bN2G * this->dSize;
        attenOutOffset = this->bN2GD;
        dstStride = (this->tilingData->inputParams.bSize * this->tilingData->inputParams.n2Size *
                         this->tilingData->inputParams.gSize -
                     1) *
                    this->dSize * sizeof(INPUT_T);
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        datacopyOffset = this->n2GD;
        attenOutOffset = this->n2GD;
        dstStride = (this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize - 1) * this->dSize *
                    sizeof(INPUT_T);
    }

    int64_t n2Offset = loopN2 * this->gS1D;

    // dataCopyParams.dstStride类型定义uint16_t，65535是其最大值
    if (likely(dstStride <= 65535)) {
        dataCopyParams.blockCount = extraInfo.vec2S1RealSize;
        dataCopyParams.dstStride = static_cast<uint16_t>(dstStride);
        DataCopyPad(this->attentionOutGm[extraInfo.qCoreOffset + n2Offset + s1oIdx * extraInfo.vec2S1BaseSize * attenOutOffset],
                    attenOut, dataCopyParams);
    } else {
        dataCopyParams.blockCount = 1;
        dataCopyParams.dstStride = 0;

        for (int32_t i = 0; i < extraInfo.vec2S1RealSize; i++) {
            DataCopyPad(this->attentionOutGm[extraInfo.qCoreOffset + n2Offset +
                                             s1oIdx * extraInfo.vec2S1BaseSize * attenOutOffset + i * datacopyOffset],
                        attenOut[i * this->dSizeAlign16], dataCopyParams);
        }
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool enableL1Reuse>
__aicore__ inline void
FusedFloydAttentionS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock, bmm1Format,
                              enableL1Reuse>::SoftmaxDataCopyOut(SplitExtraInfo &extraInfo, int64_t loopN2)
{
    int64_t vec2S1N2Offset = loopN2 * extraInfo.s1Size * fp32BaseSize;
    LocalTensor<float> sumTensor;
    sumTensor = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<float>();
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopy(this->softmaxSumGm[extraInfo.softmaxMaxOffset + vec2S1N2Offset], sumTensor,
             extraInfo.s1RealSize * fp32BaseSize);
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

#endif // FUSED_FLOYD_ATTENTION_S1S2_BN2GS1_H
