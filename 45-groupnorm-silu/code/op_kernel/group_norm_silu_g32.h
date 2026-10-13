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
 * \file group_norm_silu_g32.h
 * \brief
 */
#ifndef GROUP_NORM_SILU_G32_H_
#define GROUP_NORM_SILU_G32_H_

#include "group_norm_silu_base.h"

namespace GroupNormSilu {
using namespace AscendC;

template <typename T>
class GroupNormSiluG32 : public GroupNormSiluBase<T>
{
public:
    __aicore__ inline GroupNormSiluG32(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
        const GroupNormSiluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    // function
    template <bool isAlign = true>
    __aicore__ inline void CopyInX(const int64_t startAddr, const int64_t copyNum);
    __aicore__ inline void ComputeSum(LocalTensor<T>& x2Buf, const int64_t& processNum);
    __aicore__ inline void ReduceSumXAndX2(const int64_t& groupIdx, const int64_t& reduseNum);
    __aicore__ inline void CopyInGammaBeta(const int64_t& startAddr);
    __aicore__ inline void CalculateGroupNormSilu(const float& scale, const float& bias, const int64_t& computeNum);
    __aicore__ inline void ComputeGroupNormSilu(
        const int64_t& groupIdx, LocalTensor<T>& meanUb, LocalTensor<T>& rstdUb);
    __aicore__ inline void ProcessPerCore(const int64_t& groups);
    __aicore__ inline void ProcessGroupNormSilu(const int64_t& offset, const int64_t& groups);
    __aicore__ inline void CopyOutY(const int64_t& startAddr, const int64_t& copyNum);
    __aicore__ inline void CopyOutYWithPad(
        const float& scale, const float& bias, const int64_t& startAddr, const int64_t& copyNum);
    __aicore__ inline void CopyOutMeanAndRstd(const int64_t& startAddr, const int64_t& copyNum);
    __aicore__ inline void CopyOutMeanAndRstdAlign(const int64_t& startAddr);
    // constant
    constexpr static int32_t bufferNum = 2;
    constexpr static float negativeOne = static_cast<float>(-1.0);
    constexpr static float scalarZero = 0.0;
    constexpr static float scalarOne = 1.0;
    constexpr static int64_t processSize = 8192;
    constexpr static int64_t elementsPerBlock = 8;
    constexpr static int64_t mask = 64;
    constexpr static int64_t innerNumPerCore = 2048;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueGamma;
    TQue<QuePosition::VECIN, 1> inQueueBeta;
    TQue<QuePosition::VECOUT, bufferNum> outQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueMean;
    TQue<QuePosition::VECOUT, 1> outQueueRstd;

    GlobalTensor<T> xGm, gammaGm, betaGm;
    GlobalTensor<T> yGm, meanGm, rstdGm;

    TBuf<QuePosition::VECCALC> tempX;
    TBuf<QuePosition::VECCALC> workLocal;
    TBuf<QuePosition::VECCALC> tmpTensor;

    int32_t blockIdx = 0;
    float numRec = 0;
    int64_t reduseSumNum = 0;
    int64_t gmOffset = 0;
    int64_t numPerCoreLoop = 0;
    int64_t numPerCoreTail = 0;

    // tiling params
    const GroupNormSiluTilingData* tiling;
};

template <typename T>
__aicore__ inline void GroupNormSiluG32<T>::Init(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
    const GroupNormSiluTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    tiling = tilingData;
    xGm.SetGlobalBuffer((__gm__ T*)x);
    gammaGm.SetGlobalBuffer((__gm__ T*)gamma);
    betaGm.SetGlobalBuffer((__gm__ T*)beta);
    yGm.SetGlobalBuffer((__gm__ T*)y);
    meanGm.SetGlobalBuffer((__gm__ T*)mean);
    rstdGm.SetGlobalBuffer((__gm__ T*)rstd);

    pipe.InitBuffer(inQueueX, bufferNum, tiling->processSize * sizeof(T));
    pipe.InitBuffer(inQueueGamma, 1, elementsPerBlock * sizeof(T));
    pipe.InitBuffer(inQueueBeta, 1, elementsPerBlock * sizeof(T));
    pipe.InitBuffer(outQueueY, bufferNum, tiling->processSize * sizeof(T));
    pipe.InitBuffer(outQueueMean, 1, innerNumPerCore * sizeof(T));
    pipe.InitBuffer(outQueueRstd, 1, innerNumPerCore * sizeof(T));
    pipe.InitBuffer(tempX, tiling->processSize * sizeof(float));
    pipe.InitBuffer(workLocal, tiling->processSize / mask * sizeof(T));

    // one block
    pipe.InitBuffer(tmpTensor, elementsPerBlock * sizeof(float));
    numRec = float(scalarOne) / float(tiling->elemNum);
    if (tiling->loopNum > 1) {
        reduseSumNum = tiling->processSize;
    } else {
        reduseSumNum = tiling->loopTail;
    }
    gmOffset = blockIdx * tiling->numPerCore * tiling->elemNum;
}

template <typename T>
__aicore__ inline void GroupNormSiluG32<T>::Process()
{
    if (blockIdx == tiling->realCoreNum - 1) { // process last core
        ProcessPerCore(tiling->numLastCore);
    } else if (blockIdx < tiling->realCoreNum - 1) {
        ProcessPerCore(tiling->numPerCore);
    } else {
        return;
    }
}

template <typename T>
__aicore__ inline void GroupNormSiluG32<T>::ProcessPerCore(const int64_t& pgroups)
{
    numPerCoreLoop = pgroups / innerNumPerCore;
    numPerCoreTail = pgroups - numPerCoreLoop * innerNumPerCore;
    for (int64_t loop = 0; loop < numPerCoreLoop; loop++) {
        int64_t groupOffset = loop * innerNumPerCore;
        ProcessGroupNormSilu(groupOffset, innerNumPerCore);
        CopyOutMeanAndRstdAlign(groupOffset);
    }
    if (numPerCoreTail > 0) {
        int64_t groupTailOffsetN = numPerCoreLoop * innerNumPerCore;
        ProcessGroupNormSilu(groupTailOffsetN, numPerCoreTail);
        CopyOutMeanAndRstd(groupTailOffsetN, numPerCoreTail);
    }
}

template <typename T>
__aicore__ inline void GroupNormSiluG32<T>::ProcessGroupNormSilu(const int64_t& offset, const int64_t& groups)
{
    LocalTensor<float> meanUb = outQueueMean.AllocTensor<T>();
    LocalTensor<float> rstdUb = outQueueRstd.AllocTensor<T>();
    for (int64_t groupIdx = 0; groupIdx < groups; groupIdx++) {
        int64_t groupStartIdx = offset + groupIdx;
        // accumulate and reduceSum x && x^2
        ReduceSumXAndX2(groupStartIdx, reduseSumNum);
        // calculate GroupNormSilu
        ComputeGroupNormSilu(groupStartIdx, meanUb, rstdUb);
    }
    outQueueMean.EnQue(meanUb);
    outQueueRstd.EnQue(rstdUb);
}

template <typename T>
__aicore__ inline void GroupNormSiluG32<T>::ReduceSumXAndX2(const int64_t& groupIdx, const int64_t& reduseNum)
{
    // first init x x^2
    LocalTensor<float> xBuf = tempX.Get<float>();
    LocalTensor<T> x2Buf = outQueueY.AllocTensor<T>();
    Duplicate<float>(xBuf, scalarZero, tiling->processSize);
    Duplicate<float>(x2Buf, scalarZero, tiling->processSize);
    int64_t StartAddr = gmOffset + groupIdx * tiling->elemNum;
    // middle
    for (int loopIdx = 0; loopIdx < tiling->loopNum - 1; loopIdx++) {
        int64_t loopStartAddr = StartAddr + loopIdx * tiling->processSize;
        CopyInX(loopStartAddr, tiling->processSize);
        ComputeSum(x2Buf, tiling->processSize);
    }
    // tail
    int64_t tailStartAddr = StartAddr + (tiling->loopNum - 1) * tiling->processSize;
    CopyInX<false>(tailStartAddr, tiling->loopTail);
    ComputeSum(x2Buf, tiling->loopTail);

    // reduseSum(x)
    ReduceSum<float>(tmpTensor.Get<float>(), xBuf, workLocal.Get<float>(), reduseNum);
    // reduseSum(x^2)
    ReduceSum<float>(xBuf, x2Buf, workLocal.Get<float>(), reduseNum);
    outQueueY.FreeTensor(x2Buf);
}

template <typename T>
template <bool isAlign>
__aicore__ inline void GroupNormSiluG32<T>::CopyInX(const int64_t startAddr, const int64_t copyNum)
{
    LocalTensor<T> xUb = inQueueX.AllocTensor<T>();
    this->template CopyInData<T, isAlign>(xUb, xGm[startAddr], copyNum);
    inQueueX.EnQue(xUb);
}

template <typename T>
__aicore__ inline void GroupNormSiluG32<T>::ComputeSum(LocalTensor<T>& x2Buf, const int64_t& processNum)
{
    LocalTensor<T> xUb = inQueueX.DeQue<T>();
    // add x
    Add(tempX.Get<float>(), tempX.Get<float>(), xUb, processNum);
    // x^2
    Mul(xUb, xUb, xUb, processNum);
    PipeBarrier<PIPE_V>();
    // add x^2
    Add(x2Buf, x2Buf, xUb, processNum);
    inQueueX.FreeTensor(xUb);
}

template <typename T>
__aicore__ inline void GroupNormSiluG32<T>::ComputeGroupNormSilu(
    const int64_t& groupIdx, LocalTensor<T>& meanUb, LocalTensor<T>& rstdUb)
{
    LocalTensor<float> x2Ub = tempX.Get<float>();
    LocalTensor<float> tmpMean = tmpTensor.Get<float>();
    // calculate mean && variance
    float mean = tmpMean.GetValue(0);
    float rstd = x2Ub.GetValue(0);
    mean = mean * numRec;
    rstd = float(1.0) / sqrt(rstd * numRec - mean * mean + tiling->epsilon);
    meanUb.SetValue(groupIdx % innerNumPerCore, mean);
    rstdUb.SetValue(groupIdx % innerNumPerCore, rstd);
    // calculating y  y = (x- mean) * rstd * gamma + beta
    for (int64_t dIdx = 0; dIdx < tiling->shapeD; dIdx++) {
        // copy gamma && beta
        int64_t subStartAddr = ((blockIdx * tiling->numPerCore + groupIdx) * tiling->shapeD + dIdx) % tiling->shapeC;
        CopyInGammaBeta(subStartAddr);
        LocalTensor<float> gammaUb = inQueueGamma.DeQue<T>();
        LocalTensor<float> betaUb = inQueueBeta.DeQue<T>();
        float gamma = gammaUb.GetValue(0);
        float beta = betaUb.GetValue(0);
        float scale = rstd * gamma;
        float bias = scale * mean * negativeOne + beta;
        inQueueGamma.FreeTensor(gammaUb);
        inQueueBeta.FreeTensor(betaUb);
        int64_t hwStartAddr = gmOffset + groupIdx * tiling->elemNum + dIdx * tiling->hwNum;
        for (int64_t innerLoop = 0; innerLoop < tiling->innerLoopNum - 1; innerLoop++) {
            int64_t hwInnerAddr = hwStartAddr + innerLoop * tiling->processSize;
            CopyInX(hwInnerAddr, tiling->processSize);
            CalculateGroupNormSilu(scale, bias, tiling->processSize);
            CopyOutY(hwInnerAddr, tiling->processSize);
        }
        int64_t hwTailAddrG32 = hwStartAddr + (tiling->innerLoopNum - 1) * tiling->processSize;
        CopyInX<false>(hwTailAddrG32, tiling->innerLoopTail);
        CalculateGroupNormSilu(scale, bias, tiling->innerLoopTail);
        CopyOutYWithPad(scale, bias, hwTailAddrG32, tiling->innerLoopTail);
    }
}

template <typename T>
__aicore__ inline void GroupNormSiluG32<T>::CopyInGammaBeta(const int64_t& startAddr)
{
    LocalTensor<T> gamma = inQueueGamma.AllocTensor<T>();
    this->template CopyInData<T, false>(gamma, gammaGm[startAddr], 1);
    event_t eventMte2ToS1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventMte2ToS1);
    WaitFlag<HardEvent::MTE2_S>(eventMte2ToS1);
    inQueueGamma.EnQue(gamma);

    LocalTensor<T> beta = inQueueBeta.AllocTensor<T>();
    this->template CopyInData<T, false>(beta, betaGm[startAddr], 1);
    event_t eventMte2ToS2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventMte2ToS2);
    WaitFlag<HardEvent::MTE2_S>(eventMte2ToS2);
    inQueueBeta.EnQue(beta);
}

template <typename T>
__aicore__ inline void GroupNormSiluG32<T>::CalculateGroupNormSilu(
    const float& scale, const float& bias, const int64_t& computeNum)
{
    LocalTensor<float> xUb = inQueueX.DeQue<T>();
    LocalTensor<T> outY = outQueueY.AllocTensor<T>();
    Muls(xUb, xUb, scale, computeNum);
    PipeBarrier<PIPE_V>();
    if (tiling->activateSilu == 1) {
        Adds(xUb, xUb, bias, computeNum);
        PipeBarrier<PIPE_V>();
        Muls(outY, xUb, negativeOne, computeNum);
        PipeBarrier<PIPE_V>();
        Exp(outY, outY, computeNum);
        PipeBarrier<PIPE_V>();
        Adds(outY, outY, scalarOne, computeNum);
        PipeBarrier<PIPE_V>();
        Div(outY, xUb, outY, computeNum);
        PipeBarrier<PIPE_V>();
    } else {
        Adds(outY, xUb, bias, computeNum);
        PipeBarrier<PIPE_V>();
    }
    outQueueY.EnQue(outY);
    inQueueX.FreeTensor(xUb);
}

template <typename T>
__aicore__ inline void GroupNormSiluG32<T>::CopyOutY(const int64_t& startAddr, const int64_t& copyNum)
{
    LocalTensor<T> outY = outQueueY.DeQue<T>();
    DataCopy(yGm[startAddr], outY, copyNum);
    outQueueY.FreeTensor(outY);
}

template <typename T>
__aicore__ inline void GroupNormSiluG32<T>::CopyOutYWithPad(
    const float& scale, const float& bias, const int64_t& startAddr, const int64_t& copyNum)
{
    LocalTensor<T> outY = outQueueY.DeQue<T>();
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    uint16_t dataCount = static_cast<uint16_t>(copyNum);
    uint16_t blockCount = 1;
    uint16_t blockLen = dataCount * sizeof(T);
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    DataCopyParams dataCopyParams{blockCount, blockLen, srcStride, dstStride};
    DataCopyPad(yGm[startAddr], outY, dataCopyParams);
    outQueueY.FreeTensor(outY);
#else
    if (tiling->hwNum % elementsPerBlock == 0) {
        DataCopy(yGm[startAddr], outY, copyNum);
        outQueueY.FreeTensor(outY);
    } else {
        if (tiling->hwNum < elementsPerBlock) {
            DataCopy(yGm[startAddr], outY, elementsPerBlock);
            outQueueY.FreeTensor(outY);
        } else {
            if (copyNum >= elementsPerBlock) {
                DataCopy(yGm[startAddr], outY, copyNum);
            }
            outQueueY.FreeTensor(outY);
            CopyInX(startAddr + copyNum - elementsPerBlock, elementsPerBlock);
            CalculateGroupNormSilu(scale, bias, elementsPerBlock);
            CopyOutY(startAddr + copyNum - elementsPerBlock, elementsPerBlock);
        }
    }
#endif
}

template <typename T>
__aicore__ inline void GroupNormSiluG32<T>::CopyOutMeanAndRstd(const int64_t& startAddr, const int64_t& copyNum)
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
    DataCopyPad(meanGm[blockIdx * tiling->numPerCore + startAddr], meanOut, dataCopyParams);
    DataCopyPad(rstdGm[blockIdx * tiling->numPerCore + startAddr], rstdOut, dataCopyParams);
#endif
    outQueueMean.FreeTensor(meanOut);
    outQueueRstd.FreeTensor(rstdOut);
}

template <typename T>
__aicore__ inline void GroupNormSiluG32<T>::CopyOutMeanAndRstdAlign(const int64_t& startAddr)
{
    LocalTensor<T> meanOut = outQueueMean.DeQue<T>();
    LocalTensor<T> rstdOut = outQueueRstd.DeQue<T>();
    DataCopy(meanGm[blockIdx * tiling->numPerCore + startAddr], meanOut, innerNumPerCore);
    DataCopy(rstdGm[blockIdx * tiling->numPerCore + startAddr], rstdOut, innerNumPerCore);
    outQueueMean.FreeTensor(meanOut);
    outQueueRstd.FreeTensor(rstdOut);
}
} // namespace GroupNormSilu

#endif // GROUP_NORM_SILU_G32_H_