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
 * \file group_norm_silu_b32.h
 * \brief
 */

#ifndef GROUP_NORM_SILU_B32_H_
#define GROUP_NORM_SILU_B32_H_

#include "group_norm_silu_base.h"

namespace GroupNormSilu {
using namespace AscendC;

template <typename T>
class GroupNormSiluB32 : public GroupNormSiluBase<T>
{
public:
    __aicore__ inline GroupNormSiluB32(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR silu, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
        const GroupNormSiluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInXFirst(const int64_t& xOffset, const int64_t& copyNum);
    template <bool isAlign = true>
    __aicore__ inline void CopyInX(const int64_t xOffset, const int64_t copyNum);
    __aicore__ inline void CopyInGammaAndBeta();
    __aicore__ inline void ComputeSumX(LocalTensor<T>& x2Ub32, const int64_t& num);
    __aicore__ inline void ComputeSilu(const int64_t& loopNum);
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
    constexpr static int64_t elementsPerBlock = blockSize / sizeof(T);
    constexpr static int64_t mask = 64;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueGamma;
    TQue<QuePosition::VECIN, 1> inQueueBeta;
    TQue<QuePosition::VECOUT, bufferNum> outQueueSilu;
    TQue<QuePosition::VECOUT, 1> outQueueMean;
    TQue<QuePosition::VECOUT, 1> outQueueRstd;

    TBuf<QuePosition::VECCALC> x1Buf32;
    TBuf<QuePosition::VECCALC> workLocal;
    TBuf<QuePosition::VECCALC> tmpTensor;

    GlobalTensor<T> xGm, gammaGm, betaGm;
    GlobalTensor<T> siluGm, meanGm, rstdGm;

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

template <typename T>
__aicore__ inline void GroupNormSiluB32<T>::Init(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR silu, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
    const GroupNormSiluTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    tiling = tilingData;
    xGm.SetGlobalBuffer((__gm__ T*)x);
    gammaGm.SetGlobalBuffer((__gm__ T*)gamma);
    betaGm.SetGlobalBuffer((__gm__ T*)beta);
    siluGm.SetGlobalBuffer((__gm__ T*)silu);
    meanGm.SetGlobalBuffer((__gm__ T*)mean);
    rstdGm.SetGlobalBuffer((__gm__ T*)rstd);

    shapeCAlign = this->CeilDiv(tiling->shapeC, elementsPerBlock) * elementsPerBlock;
    numPerCoreAlign = this->CeilDiv(tiling->numPerCore, elementsPerBlock) * elementsPerBlock;
    loopTailAlign = this->CeilDiv(tiling->loopTail, elementsPerBlock) * elementsPerBlock;
    innerLoopTailAlign = this->CeilDiv(tiling->innerLoopTail, elementsPerBlock) * elementsPerBlock;

    pipe.InitBuffer(x1Buf32, tiling->processSize * sizeof(T));
    pipe.InitBuffer(inQueueX, bufferNum, tiling->processSize * sizeof(T));
    pipe.InitBuffer(outQueueSilu, bufferNum, tiling->processSize * sizeof(T));
    pipe.InitBuffer(inQueueGamma, 1, shapeCAlign * sizeof(T));
    pipe.InitBuffer(inQueueBeta, 1, shapeCAlign * sizeof(T));
    pipe.InitBuffer(workLocal, tiling->processSize / mask * sizeof(T));
    pipe.InitBuffer(outQueueMean, 1, numPerCoreAlign * sizeof(T));
    pipe.InitBuffer(outQueueRstd, 1, numPerCoreAlign * sizeof(T));
    pipe.InitBuffer(tmpTensor, elementsPerBlock * sizeof(T));
    gmXOffset = blockIdx * tiling->numPerCore * tiling->elemNum;
    numRec = float(1.0) / float(tiling->elemNum);
}

template <typename T>
__aicore__ inline void GroupNormSiluB32<T>::Process()
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

template <typename T>
__aicore__ inline void GroupNormSiluB32<T>::ProcessPerCore(const int64_t& loopNum)
{
    CopyInGammaAndBeta();
    ComputeSilu(loopNum);
    CopyOutMeanAndRstd(loopNum);
}

template <typename T>
__aicore__ inline void GroupNormSiluB32<T>::CopyInGammaAndBeta()
{
    LocalTensor<T> gamma = inQueueGamma.AllocTensor<T>();
    this->template CopyInData<T, false>(gamma, gammaGm, tiling->shapeC);
    inQueueGamma.EnQue(gamma);

    LocalTensor<T> beta = inQueueBeta.AllocTensor<T>();
    this->template CopyInData<T, false>(beta, betaGm, tiling->shapeC);
    inQueueBeta.EnQue(beta);
}

template <typename T>
__aicore__ inline void GroupNormSiluB32<T>::ComputeSilu(const int64_t& loopNum)
{
    if (tiling->loopNum == 1) {
        ComputeOneLoop(loopNum);
    } else {
        ComputeMultipleLoop(loopNum);
    }
}

template <typename T>
__aicore__ inline void GroupNormSiluB32<T>::ComputeOneLoop(const int64_t& loopNum)
{
    LocalTensor<float> meanUb = outQueueMean.AllocTensor<T>();
    LocalTensor<float> rstdUb = outQueueRstd.AllocTensor<T>();
    LocalTensor<float> gammaUb32 = inQueueGamma.DeQue<T>();
    LocalTensor<float> betaUb32 = inQueueBeta.DeQue<T>();

    for (int64_t outIdx = 0; outIdx < loopNum; outIdx++) {
        AccumulateXandX2OneLoop(outIdx);
        // calc var and rstd;
        float mean = tmpTensor.Get<float>().GetValue(0);
        float rstd = x1Buf32.Get<float>().GetValue(0);
        mean = mean * numRec;
        rstd = float(1.0) / sqrt(rstd * numRec - mean * mean + tiling->epsilon);
        meanUb.SetValue(outIdx, mean);
        rstdUb.SetValue(outIdx, rstd);
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
    outQueueMean.EnQue(meanUb);
    outQueueRstd.EnQue(rstdUb);
    inQueueGamma.FreeTensor(gammaUb32);
    inQueueBeta.FreeTensor(betaUb32);
}

template <typename T>
__aicore__ inline void GroupNormSiluB32<T>::ComputeMultipleLoop(const int64_t& loopNum)
{
    LocalTensor<float> meanUb = outQueueMean.AllocTensor<T>();
    LocalTensor<float> rstdUb = outQueueRstd.AllocTensor<T>();
    LocalTensor<float> gammaUb32 = inQueueGamma.DeQue<T>();
    LocalTensor<float> betaUb32 = inQueueBeta.DeQue<T>();

    for (int64_t outIdx = 0; outIdx < loopNum; outIdx++) {
        AccumulateXandX2MultipleLoop(outIdx);
        // calc var and rstd;
        float mean = tmpTensor.Get<float>().GetValue(0);
        float rstd = x1Buf32.Get<float>().GetValue(0);
        mean = mean * numRec;
        rstd = float(1.0) / sqrt(rstd * numRec - mean * mean + tiling->epsilon);
        meanUb.SetValue(outIdx, mean);
        rstdUb.SetValue(outIdx, rstd);
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
    outQueueMean.EnQue(meanUb);
    outQueueRstd.EnQue(rstdUb);
    inQueueGamma.FreeTensor(gammaUb32);
    inQueueBeta.FreeTensor(betaUb32);
}

template <typename T>
__aicore__ inline void GroupNormSiluB32<T>::AccumulateXandX2OneLoop(const int64_t& outIdx)
{
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    LocalTensor<float> workLocalUb = workLocal.Get<float>();
    LocalTensor<float> tmpMean = tmpTensor.Get<float>();

    CopyInX<false>(outIdx * tiling->elemNum, tiling->loopTail);

    LocalTensor<T> xUb = inQueueX.DeQue<T>();
    Mul(x1Ub32, xUb, xUb, tiling->loopTail);

    // accumulate x and x^2
    ReduceSum<float>(tmpMean, xUb, workLocalUb, tiling->loopTail);
    ReduceSum<float>(x1Ub32, x1Ub32, workLocalUb, tiling->loopTail);
    inQueueX.FreeTensor(xUb);
}

template <typename T>
__aicore__ inline void GroupNormSiluB32<T>::AccumulateXandX2MultipleLoop(const int64_t& outIdx)
{
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    LocalTensor<float> workLocalUb = workLocal.Get<float>();
    LocalTensor<T> x2Ub32 = outQueueSilu.AllocTensor<T>();
    LocalTensor<float> tmpMean = tmpTensor.Get<float>();

    // process the first loop
    CopyInXFirst(outIdx * tiling->elemNum, tiling->processSize);
    Mul(x2Ub32, x1Ub32, x1Ub32, tiling->processSize);
    // process the middle loops
    for (int64_t innerIdx = 1; innerIdx < tiling->loopNum - 1; innerIdx++) {
        CopyInX(outIdx * tiling->elemNum + innerIdx * tiling->processSize, tiling->processSize);
        ComputeSumX(x2Ub32, tiling->processSize);
    }
    // process the last loop
    if (tiling->loopNum > 1) { // process tail data
        CopyInX<false>(outIdx * tiling->elemNum + (tiling->loopNum - 1) * tiling->processSize, tiling->loopTail);
        ComputeSumX(x2Ub32, tiling->loopTail);
    }
    // accumulate x and x^2
    ReduceSum<float>(tmpMean, x1Ub32, workLocalUb, tiling->processSize);
    ReduceSum<float>(x1Ub32, x2Ub32, workLocalUb, tiling->processSize);
    outQueueSilu.FreeTensor(x2Ub32);
}

template <typename T>
__aicore__ inline void GroupNormSiluB32<T>::CopyInXFirst(const int64_t& xOffset, const int64_t& copyNum)
{
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    DataCopy(x1Ub32, xGm[gmXOffset + xOffset], copyNum);
    event_t eventMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
}

template <typename T>
template <bool isAlign>
__aicore__ inline void GroupNormSiluB32<T>::CopyInX(const int64_t xOffset, const int64_t copyNum)
{
    LocalTensor<T> xUb = inQueueX.AllocTensor<T>();
    this->template CopyInData<T, isAlign>(xUb, xGm[gmXOffset + xOffset], copyNum);
    inQueueX.EnQue(xUb);
}

template <typename T>
__aicore__ inline void GroupNormSiluB32<T>::ComputeSumX(LocalTensor<T>& x2Ub32, const int64_t& num)
{
    LocalTensor<T> xUb = inQueueX.DeQue<T>();
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();

    Add(x1Ub32, x1Ub32, xUb, num);
    Mul(xUb, xUb, xUb, num);
    Add(x2Ub32, x2Ub32, xUb, num);
    inQueueX.FreeTensor(xUb);
}

template <typename T>
__aicore__ inline void GroupNormSiluB32<T>::CalcSilu(const float& scale, const float& bias, const int64_t& calcNum)
{
    LocalTensor<T> xUb = inQueueX.DeQue<T>();
    LocalTensor<T> outSilu = outQueueSilu.AllocTensor<T>();
    // normalize x
    Muls(xUb, xUb, scale, calcNum);
    PipeBarrier<PIPE_V>();
    // calc silu
    if (tiling->activateSilu) {
        Adds(xUb, xUb, bias, calcNum);
        PipeBarrier<PIPE_V>();
        Muls(outSilu, xUb, negativeOne, calcNum);
        PipeBarrier<PIPE_V>();
        Exp(outSilu, outSilu, calcNum);
        PipeBarrier<PIPE_V>();
        Adds(outSilu, outSilu, scalarOne, calcNum);
        PipeBarrier<PIPE_V>();
        Div(outSilu, xUb, outSilu, calcNum);
        PipeBarrier<PIPE_V>();
    } else {
        Adds(outSilu, xUb, bias, calcNum);
        PipeBarrier<PIPE_V>();
    }

    outQueueSilu.EnQue(outSilu);
    inQueueX.FreeTensor(xUb);
}

template <typename T>
__aicore__ inline void GroupNormSiluB32<T>::CopyOutY(const int64_t& yOffset, const int64_t& copyNum)
{
    LocalTensor<T> outSilu = outQueueSilu.DeQue<T>();
    DataCopy(siluGm[gmXOffset + yOffset], outSilu, copyNum);
    outQueueSilu.FreeTensor(outSilu);
}

template <typename T>
__aicore__ inline void GroupNormSiluB32<T>::CopyOutYWithPad(
    const float& scale, const float& bias, const int64_t& yOffset, const int64_t& copyNum)
{
    LocalTensor<T> outSilu = outQueueSilu.DeQue<T>();
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    uint16_t dataCount = static_cast<uint16_t>(copyNum);
    uint16_t blockCount = 1;
    uint16_t blockLen = dataCount * sizeof(T);
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

template <typename T>
__aicore__ inline void GroupNormSiluB32<T>::CopyOutMeanAndRstd(const int64_t& copyNum)
{
    LocalTensor<T> meanOut = outQueueMean.DeQue<T>();
    LocalTensor<T> rstdOut = outQueueRstd.DeQue<T>();
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    // when support DataCopyPad, use DataCopyPad
    uint16_t dataCount = static_cast<uint16_t>(copyNum);
    uint16_t blockCount = 1;
    uint16_t blockLen = dataCount * sizeof(T);
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

#endif // GROUP_NORM_SILU_B32_H_