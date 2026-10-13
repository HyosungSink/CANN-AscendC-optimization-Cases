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
 * \file group_norm_silu_b16.h
 * \brief
 */

#ifndef GROUP_NORM_SILU_B16_H_
#define GROUP_NORM_SILU_B16_H_

#include "group_norm_silu_base.h"

namespace GroupNormSilu {
using namespace AscendC;

template <typename T1, typename T2>
class GroupNormSiluB16 : public GroupNormSiluBase<T1>
{
public:
    __aicore__ inline GroupNormSiluB16(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR silu, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
        const GroupNormSiluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <bool isAlign = true>
    __aicore__ inline void CopyInX(const int64_t xOffset, const int64_t copyNum);
    __aicore__ inline void CopyInGammaAndBeta();
    __aicore__ inline void CastGammaAndBeta();
    __aicore__ inline void ComputeSumX(const int64_t& num);
    __aicore__ inline void CastMeanAndRstd(const int64_t& copyNum);
    __aicore__ inline void ComputeSilu(const int64_t& loopNum);
    __aicore__ inline void ComputeForMeanAlign(
        LocalTensor<T1>& meanOut, LocalTensor<T1>& rstdOut, const int64_t& groups);
    __aicore__ inline void ComputeOneLoop(const int64_t& loopNum);
    __aicore__ inline void ComputeMultipleLoop(const int64_t& loopNum);
    __aicore__ inline void AccumulateXandX2OneLoop(const int64_t& outIdx);
    __aicore__ inline void AccumulateXandX2MultipleLoop(const int64_t& outIdx);
    __aicore__ inline void CalcSilu(const float& scale, const float& bias, const int64_t& calcNum);
    __aicore__ inline void CopyOutY(const int64_t& yOffset, const int64_t& copyNum);
    __aicore__ inline void CopyOutYWithPad(
        const float& scale, const float& bias, const int64_t& yOffset, const int64_t& copyNum);
    __aicore__ inline void CopyOutMeanAndRstd(const int64_t& copyNum);
    __aicore__ inline void ProcessPerCore(const int64_t& loopNum);
    constexpr static int32_t bufferNum = 2;
    constexpr static float negativeOne = -1.0;
    constexpr static float scalarOne = 1.0;
    constexpr static int64_t blockSize = 32;
    constexpr static int64_t elementsPerBlock = blockSize / sizeof(T1);
    constexpr static int64_t elementsPerBlock32 = 8;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueGamma;
    TQue<QuePosition::VECIN, 1> inQueueBeta;
    TQue<QuePosition::VECOUT, bufferNum> outQueueSilu;
    TQue<QuePosition::VECOUT, 1> outQueueMean;
    TQue<QuePosition::VECOUT, 1> outQueueRstd;

    TBuf<QuePosition::VECCALC> xBuf32;
    TBuf<QuePosition::VECCALC> x1Buf32;
    TBuf<QuePosition::VECCALC> x2Buf32;
    TBuf<QuePosition::VECCALC> gammaBuf32;
    TBuf<QuePosition::VECCALC> betaBuf32;
    TBuf<QuePosition::VECCALC> meanBuf32;
    TBuf<QuePosition::VECCALC> rstdBuf32;
    TBuf<QuePosition::VECCALC> tmpTensor;

    GlobalTensor<T1> xGm;
    GlobalTensor<T2> gammaGm, betaGm;
    GlobalTensor<T1> siluGm, meanGm, rstdGm;

    int32_t blockIdx = 0;
    int64_t blockOffset = 0;
    int64_t gmXOffset = 0;
    float numRec = 0;
    int64_t shapeCAlign = 0;
    int64_t numPerCoreAlign = 0;
    int64_t loopTailAlign = 0;
    int64_t innerLoopTailAlign = 0;
    // tiling params
    const GroupNormSiluTilingData* tiling;
};

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluB16<T1, T2>::Init(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR silu, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
    const GroupNormSiluTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    tiling = tilingData;
    xGm.SetGlobalBuffer((__gm__ T1*)x);
    gammaGm.SetGlobalBuffer((__gm__ T2*)gamma);
    betaGm.SetGlobalBuffer((__gm__ T2*)beta);
    siluGm.SetGlobalBuffer((__gm__ T1*)silu);
    meanGm.SetGlobalBuffer((__gm__ T1*)mean);
    rstdGm.SetGlobalBuffer((__gm__ T1*)rstd);

    shapeCAlign = this->CeilDiv(tiling->shapeC, elementsPerBlock) * elementsPerBlock;
    numPerCoreAlign = this->CeilDiv(tiling->numPerCore, elementsPerBlock) * elementsPerBlock;
    loopTailAlign = this->CeilDiv(tiling->loopTail, elementsPerBlock) * elementsPerBlock;
    innerLoopTailAlign = this->CeilDiv(tiling->innerLoopTail, elementsPerBlock) * elementsPerBlock;

    pipe.InitBuffer(xBuf32, tiling->processSize * sizeof(float));
    pipe.InitBuffer(x1Buf32, tiling->processSize * sizeof(float));
    pipe.InitBuffer(x2Buf32, tiling->processSize * sizeof(float));
    pipe.InitBuffer(inQueueX, bufferNum, tiling->processSize * sizeof(T1));
    pipe.InitBuffer(outQueueSilu, bufferNum, tiling->processSize * sizeof(T1));
    pipe.InitBuffer(inQueueGamma, 1, shapeCAlign * sizeof(T2));
    pipe.InitBuffer(inQueueBeta, 1, shapeCAlign * sizeof(T2));
    if (sizeof(T2) != sizeof(float)) {
        pipe.InitBuffer(gammaBuf32, shapeCAlign * sizeof(float));
        pipe.InitBuffer(betaBuf32, shapeCAlign * sizeof(float));
    }
    pipe.InitBuffer(outQueueMean, 1, numPerCoreAlign * sizeof(T1));
    pipe.InitBuffer(outQueueRstd, 1, numPerCoreAlign * sizeof(T1));
    pipe.InitBuffer(meanBuf32, numPerCoreAlign * sizeof(float));
    pipe.InitBuffer(rstdBuf32, numPerCoreAlign * sizeof(float));
    pipe.InitBuffer(tmpTensor, elementsPerBlock32 * sizeof(float));
    gmXOffset = blockIdx * tiling->numPerCore * tiling->elemNum;
    numRec = float(1.0) / float(tiling->elemNum);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluB16<T1, T2>::Process()
{
    if (blockIdx >= tiling->realCoreNum) {
        return;
    }

    if (blockIdx == tiling->realCoreNum - 1) { // process last core
        ProcessPerCore(tiling->numLastCore);
    } else {
        ProcessPerCore(tiling->numPerCore);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluB16<T1, T2>::ProcessPerCore(const int64_t& loopNum)
{
    CopyInGammaAndBeta();
    if constexpr (sizeof(T2) != sizeof(float)) {
        CastGammaAndBeta();
    }
    ComputeSilu(loopNum);
    CastMeanAndRstd(loopNum);
    CopyOutMeanAndRstd(loopNum);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluB16<T1, T2>::CopyInGammaAndBeta()
{
    LocalTensor<T2> gamma = inQueueGamma.AllocTensor<T2>();
    this->template CopyInData<T2, false>(gamma, gammaGm, tiling->shapeC);
    inQueueGamma.EnQue(gamma);
    LocalTensor<T2> beta = inQueueBeta.AllocTensor<T2>();
    this->template CopyInData<T2, false>(beta, betaGm, tiling->shapeC);
    inQueueBeta.EnQue(beta);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluB16<T1, T2>::CastGammaAndBeta()
{
    LocalTensor<T2> gammaUb = inQueueGamma.DeQue<T2>();
    LocalTensor<T2> betaUb = inQueueBeta.DeQue<T2>();
    Cast(gammaBuf32.Get<float>(), gammaUb, RoundMode::CAST_NONE, tiling->shapeC);
    Cast(betaBuf32.Get<float>(), betaUb, RoundMode::CAST_NONE, tiling->shapeC);
    inQueueGamma.FreeTensor(gammaUb);
    inQueueBeta.FreeTensor(betaUb);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluB16<T1, T2>::ComputeSilu(const int64_t& loopNum)
{
    if (tiling->loopNum == 1) {
        ComputeOneLoop(loopNum);
    } else {
        ComputeMultipleLoop(loopNum);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluB16<T1, T2>::ComputeOneLoop(const int64_t& loopNum)
{
    LocalTensor<float> x2Ub32 = x2Buf32.Get<float>();
    LocalTensor<float> tmpMean = tmpTensor.Get<float>();
    LocalTensor<float> meanUb = meanBuf32.Get<float>();
    LocalTensor<float> rstdUb = rstdBuf32.Get<float>();
    LocalTensor<float> gammaUb32;
    LocalTensor<float> betaUb32;
    if (sizeof(T2) != sizeof(float)) {
        gammaUb32 = gammaBuf32.Get<float>();
        betaUb32 = betaBuf32.Get<float>();
    } else {
        gammaUb32 = inQueueGamma.DeQue<float>();
        betaUb32 = inQueueBeta.DeQue<float>();
    }
    for (int64_t outIdx = 0; outIdx < loopNum; outIdx++) {
        AccumulateXandX2OneLoop(outIdx);
        // calc var and rstd;
        float mean = tmpMean.GetValue(0);
        float rstd = x2Ub32.GetValue(0);
        mean = mean * numRec;
        rstd = float(1.0) / sqrt(rstd * numRec - mean * mean + tiling->epsilon);
        meanUb.SetValue(outIdx, mean);
        rstdUb.SetValue(outIdx, rstd);
        event_t eventS2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventS2V);
        WaitFlag<HardEvent::S_V>(eventS2V);
        // normalize x and calc silu
        int64_t groupIdx = ((blockIdx * tiling->numPerCore + outIdx) % tiling->numGroups);
        for (int64_t dIdx = 0; dIdx < tiling->shapeD; dIdx++) {
            int64_t cIdx = groupIdx * tiling->shapeD + dIdx;
            int64_t xOffset = outIdx * tiling->elemNum + dIdx * tiling->hwNum;
            float gamma = gammaUb32.GetValue(cIdx);
            float beta = betaUb32.GetValue(cIdx);
            float scale = rstd * gamma;
            float bias = -scale * mean + beta;
            CopyInX<false>(xOffset, tiling->innerLoopTail);
            CalcSilu(scale, bias, tiling->innerLoopTail);
            CopyOutYWithPad(scale, bias, xOffset, tiling->innerLoopTail);
        }
    }
    if (sizeof(T2) == sizeof(float)) {
        inQueueGamma.FreeTensor(gammaUb32);
        inQueueBeta.FreeTensor(betaUb32);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluB16<T1, T2>::ComputeMultipleLoop(const int64_t& loopNum)
{
    LocalTensor<float> gammaUb32;
    LocalTensor<float> betaUb32;
    if (sizeof(T2) != sizeof(float)) {
        gammaUb32 = gammaBuf32.Get<float>();
        betaUb32 = betaBuf32.Get<float>();
    } else {
        gammaUb32 = inQueueGamma.DeQue<float>();
        betaUb32 = inQueueBeta.DeQue<float>();
    }
    for (int64_t outIdx = 0; outIdx < loopNum; outIdx++) {
        AccumulateXandX2MultipleLoop(outIdx);
        // calc var and rstd;
        float mean = tmpTensor.Get<float>().GetValue(0);
        float rstd = x2Buf32.Get<float>().GetValue(0);
        mean = mean * numRec;
        rstd = float(1.0) / sqrt(rstd * numRec - mean * mean + tiling->epsilon);
        meanBuf32.Get<float>().SetValue(outIdx, mean);
        rstdBuf32.Get<float>().SetValue(outIdx, rstd);
        // normalize x and calc silu
        int64_t groupIdx = ((blockIdx * tiling->numPerCore + outIdx) % tiling->numGroups);
        for (int64_t dIdx = 0; dIdx < tiling->shapeD; dIdx++) {
            int64_t cIdx = groupIdx * tiling->shapeD + dIdx;
            float gamma = gammaUb32.GetValue(cIdx);
            float beta = betaUb32.GetValue(cIdx);
            float scale = rstd * gamma;
            float bias = -scale * mean + beta;
            for (int64_t innerIdx = 0; innerIdx < tiling->innerLoopNum - 1; innerIdx++) {
                int64_t xOffset = outIdx * tiling->elemNum + dIdx * tiling->hwNum + innerIdx * tiling->processSize;
                CopyInX(xOffset, tiling->processSize);
                CalcSilu(scale, bias, tiling->processSize);
                CopyOutY(xOffset, tiling->processSize);
            }
            int64_t tailOffset =
                outIdx * tiling->elemNum + dIdx * tiling->hwNum + (tiling->innerLoopNum - 1) * tiling->processSize;
            CopyInX<false>(tailOffset, tiling->innerLoopTail);
            CalcSilu(scale, bias, tiling->innerLoopTail);
            CopyOutYWithPad(scale, bias, tailOffset, tiling->innerLoopTail);
        }
    }
    if (sizeof(T2) == sizeof(float)) {
        inQueueGamma.FreeTensor(gammaUb32);
        inQueueBeta.FreeTensor(betaUb32);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluB16<T1, T2>::AccumulateXandX2OneLoop(const int64_t& outIdx)
{
    LocalTensor<float> xUb32 = xBuf32.Get<float>();
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    LocalTensor<float> x2Ub32 = x2Buf32.Get<float>();
    LocalTensor<float> tmpMean = tmpTensor.Get<float>();

    CopyInX<false>(outIdx * tiling->elemNum, tiling->loopTail);
    LocalTensor<T1> xUb = inQueueX.DeQue<T1>();
    Cast(x1Ub32, xUb, RoundMode::CAST_NONE, tiling->loopTail);
    PipeBarrier<PIPE_V>();
    inQueueX.FreeTensor(xUb);
    Mul(x2Ub32, x1Ub32, x1Ub32, tiling->loopTail);
    // accumulate x and x^2
    ReduceSum<float>(tmpMean, x1Ub32, xUb32, tiling->loopTail);
    ReduceSum<float>(x2Ub32, x2Ub32, xUb32, tiling->loopTail);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluB16<T1, T2>::AccumulateXandX2MultipleLoop(const int64_t& outIdx)
{
    LocalTensor<float> xUb32 = xBuf32.Get<float>();
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    LocalTensor<float> x2Ub32 = x2Buf32.Get<float>();
    LocalTensor<float> tmpMean = tmpTensor.Get<float>();
    // process the first loop
    CopyInX(outIdx * tiling->elemNum, tiling->processSize);
    LocalTensor<T1> xUb = inQueueX.DeQue<T1>();
    Cast(x1Ub32, xUb, RoundMode::CAST_NONE, tiling->processSize);
    inQueueX.FreeTensor(xUb);
    Mul(x2Ub32, x1Ub32, x1Ub32, tiling->processSize);
    // process the middle loops
    for (int64_t innerIdx = 1; innerIdx < tiling->loopNum - 1; innerIdx++) {
        CopyInX(outIdx * tiling->elemNum + innerIdx * tiling->processSize, tiling->processSize);
        ComputeSumX(tiling->processSize);
    }
    // process the last loop
    if (tiling->loopNum > 1) { // process tail data
        CopyInX<false>(outIdx * tiling->elemNum + (tiling->loopNum - 1) * tiling->processSize, tiling->loopTail);
        ComputeSumX(tiling->loopTail);
    }
    // accumulate x and x^2
    ReduceSum<float>(tmpMean, x1Ub32, xUb32, tiling->processSize);
    ReduceSum<float>(x2Ub32, x2Ub32, xUb32, tiling->processSize);
}

template <typename T1, typename T2>
template <bool isAlign>
__aicore__ inline void GroupNormSiluB16<T1, T2>::CopyInX(const int64_t xOffset, const int64_t copyNum)
{
    LocalTensor<T1> xUb = inQueueX.AllocTensor<T1>();
    this->template CopyInData<T1, isAlign>(xUb, xGm[gmXOffset + xOffset], copyNum);
    inQueueX.EnQue(xUb);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluB16<T1, T2>::ComputeSumX(const int64_t& num)
{
    LocalTensor<T1> xUb = inQueueX.DeQue<T1>();
    LocalTensor<float> xUb32 = xBuf32.Get<float>();
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    LocalTensor<float> x2Ub32 = x2Buf32.Get<float>();

    Cast(xUb32, xUb, RoundMode::CAST_NONE, num);
    inQueueX.FreeTensor(xUb);
    Add(x1Ub32, x1Ub32, xUb32, num);
    Mul(xUb32, xUb32, xUb32, num);
    Add(x2Ub32, x2Ub32, xUb32, num);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluB16<T1, T2>::CalcSilu(const float& scale, const float& bias, const int64_t& calcNum)
{
    LocalTensor<T1> xUb = inQueueX.DeQue<T1>();
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    LocalTensor<T1> outSilu = outQueueSilu.AllocTensor<T1>();
    LocalTensor<float> x2Ub32 = x2Buf32.Get<float>();

    Cast(x1Ub32, xUb, RoundMode::CAST_NONE, calcNum);
    PipeBarrier<PIPE_V>();
    inQueueX.FreeTensor(xUb);
    // normalize x
    Muls(x1Ub32, x1Ub32, scale, calcNum);
    PipeBarrier<PIPE_V>();
    Adds(x1Ub32, x1Ub32, bias, calcNum);
    PipeBarrier<PIPE_V>();
    // calc silu
    if (tiling->activateSilu) {
        Muls(x2Ub32, x1Ub32, negativeOne, calcNum);
        PipeBarrier<PIPE_V>();
        Exp(x2Ub32, x2Ub32, calcNum);
        PipeBarrier<PIPE_V>();
        Adds(x2Ub32, x2Ub32, scalarOne, calcNum);
        PipeBarrier<PIPE_V>();
        Div(x1Ub32, x1Ub32, x2Ub32, calcNum);
        PipeBarrier<PIPE_V>();
    }
    Cast(outSilu, x1Ub32, this->GetRoundMode(), calcNum);
    PipeBarrier<PIPE_V>();
    outQueueSilu.EnQue(outSilu);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluB16<T1, T2>::CopyOutY(const int64_t& yOffset, const int64_t& copyNum)
{
    LocalTensor<T1> outSilu = outQueueSilu.DeQue<T1>();
    DataCopy(siluGm[gmXOffset + yOffset], outSilu, copyNum);
    outQueueSilu.FreeTensor(outSilu);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluB16<T1, T2>::CopyOutYWithPad(
    const float& scale, const float& bias, const int64_t& yOffset, const int64_t& copyNum)
{
    LocalTensor<T1> outSilu = outQueueSilu.DeQue<T1>();
#if __CCE_AICORE__ == 220
    uint16_t dataCount = static_cast<uint16_t>(copyNum);
    uint16_t blockCount = 1;
    uint16_t blockLen = dataCount * sizeof(T1);
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    DataCopyParams dataCopyParams{blockCount, blockLen, srcStride, dstStride};
    DataCopyPad(siluGm[gmXOffset + yOffset], outSilu, dataCopyParams);
    outQueueSilu.FreeTensor(outSilu);
#else
    if (tiling->hwNum % elementsPerBlock == 0) {
        DataCopy(siluGm[gmXOffset + yOffset], outSilu, copyNum);
        outQueueSilu.FreeTensor(outSilu);
    } else {
        if (copyNum >= elementsPerBlock) {
            DataCopy(siluGm[gmXOffset + yOffset], outSilu, copyNum);
        }
        outQueueSilu.FreeTensor(outSilu);
        CopyInX(yOffset + copyNum - elementsPerBlock, elementsPerBlock);
        CalcSilu(scale, bias, elementsPerBlock);
        CopyOutY(yOffset + copyNum - elementsPerBlock, elementsPerBlock);
    }
#endif
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluB16<T1, T2>::CastMeanAndRstd(const int64_t& copyNum)
{
    LocalTensor<float> meanUb = meanBuf32.Get<float>();
    LocalTensor<float> rstdUb = rstdBuf32.Get<float>();

    LocalTensor<T1> meanOut = outQueueMean.AllocTensor<T1>();
    Cast(meanOut, meanUb, this->GetRoundMode(), copyNum);
    outQueueMean.EnQue(meanOut);
    LocalTensor<T1> rstdOut = outQueueRstd.AllocTensor<T1>();
    Cast(rstdOut, rstdUb, this->GetRoundMode(), copyNum);
    outQueueRstd.EnQue(rstdOut);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluB16<T1, T2>::CopyOutMeanAndRstd(const int64_t& copyNum)
{
    LocalTensor<T1> meanOut = outQueueMean.DeQue<T1>();
    LocalTensor<T1> rstdOut = outQueueRstd.DeQue<T1>();
#if __CCE_AICORE__ == 220
    // when support DataCopyPad, use DataCopyPad
    uint16_t dataCount = static_cast<uint16_t>(copyNum);
    uint16_t blockCount = 1;
    uint16_t blockLen = dataCount * sizeof(T1);
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    DataCopyParams dataCopyParams{blockCount, blockLen, srcStride, dstStride};
    DataCopyPad(meanGm[blockIdx * tiling->numPerCore], meanOut, dataCopyParams);
    DataCopyPad(rstdGm[blockIdx * tiling->numPerCore], rstdOut, dataCopyParams);
#endif
    outQueueMean.FreeTensor(meanOut);
    outQueueRstd.FreeTensor(rstdOut);
}
} // namespace GroupNormSilu

#endif // GROUP_NORM_SILU_B16_H_