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
 * \file group_norm_silu_hw1_b32.h
 * \brief
 */

#ifndef GROUP_NORM_SILU_HW1_B32_H_
#define GROUP_NORM_SILU_HW1_B32_H_

#include "group_norm_silu_base.h"

namespace GroupNormSilu {
using namespace AscendC;

template <typename T>
class GroupNormSiluHW1B32 : public GroupNormSiluBase<T>
{
public:
    __aicore__ inline GroupNormSiluHW1B32(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
        const GroupNormSiluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    // function
    template <bool isAlign = true>
    __aicore__ inline void CopyInX(const int64_t startAddr, const int64_t copyNum);
    __aicore__ inline void CopyInXPad(const int64_t startAddr, const int64_t copyNum);
    __aicore__ inline void ComputeSum(LocalTensor<T>& x2Buf, const int64_t& processNum);
    __aicore__ inline void ReduceSumXAndX2(const int64_t& groupIdx, const int64_t& reduseNum);
    template <bool isAlign = true>
    __aicore__ inline void CopyInGammaBeta(const int64_t startAddr, const int64_t processNum);
    __aicore__ inline void CalculateGroupNormSilu(const float& mean, const float& rstd, const int64_t& computeNum);
    __aicore__ inline void ComputeGroupNormSilu(
        const int64_t& groupIdx, LocalTensor<T>& meanUb, LocalTensor<T>& rstdUb);
    __aicore__ inline void ProcessPerCore(const int64_t& groups);
    __aicore__ inline void ProcessYWithEqualC(const int64_t& groups);
    __aicore__ inline void CalcSilu(LocalTensor<T>& betaUb, const int64_t& computeNum);
    __aicore__ inline void ProcessMeanWithEqualC(const int64_t& groups);
    __aicore__ inline void ProcessRstdWithEqualC(const int64_t& groups);
    __aicore__ inline void ProcessGroupNormSilu(const int64_t& offset, const int64_t& groups);
    __aicore__ inline void ComputeForMeanAlign(
        LocalTensor<T>& meanOut, LocalTensor<T>& rstdOut, const int64_t& offset, const int64_t& groups);
    __aicore__ inline void CopyOutY(const int64_t& startAddr, const int64_t& copyNum);
    __aicore__ inline void CopyOutYWithPad(
        const float& mean, const float& rstd, const int64_t& startAddr, const int64_t& copyNum);
    __aicore__ inline void CopyOutMeanAndRstd(const int64_t& startAddr, const int64_t& copyNum);
    __aicore__ inline void CopyOutMeanAndRstdAlign(const int64_t& startAddr);
    // constant
    constexpr static float negativeOne = static_cast<float>(-1.0);
    constexpr static float scalarZero = 0.0;
    constexpr static float scalarOne = 1.0;
    constexpr static int64_t processSize = 8192;
    constexpr static int64_t elementsPerBlock = 8;
    constexpr static int64_t mask = 64;
    constexpr static int64_t innerNumPerCore = 2048;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueGamma;
    TQue<QuePosition::VECIN, 1> inQueueBeta;
    TQue<QuePosition::VECOUT, 1> outQueueY;
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
__aicore__ inline void GroupNormSiluHW1B32<T>::Init(
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

    pipe.InitBuffer(inQueueX, 1, tiling->processSize * sizeof(T));
    pipe.InitBuffer(inQueueGamma, 1, tiling->processSize * sizeof(T));
    pipe.InitBuffer(inQueueBeta, 1, tiling->processSize * sizeof(T));
    pipe.InitBuffer(outQueueY, 1, tiling->processSize * sizeof(T));
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
}

template <typename T>
__aicore__ inline void GroupNormSiluHW1B32<T>::Process()
{
    if (blockIdx < tiling->realCoreNum - 1) {
        ProcessPerCore(tiling->numPerCore);
    } else if (blockIdx == tiling->realCoreNum - 1) { // process last core
        ProcessPerCore(tiling->numLastCore);
    } else {
        return;
    }
}

template <typename T>
__aicore__ inline void GroupNormSiluHW1B32<T>::ProcessPerCore(const int64_t& groups)
{
    if (tiling->numGroups == tiling->shapeC) {
        gmOffset = blockIdx * tiling->numPerCore * tiling->shapeC;
        ProcessYWithEqualC(groups);
        ProcessMeanWithEqualC(groups);
        ProcessRstdWithEqualC(groups);
    } else {
        gmOffset = blockIdx * tiling->numPerCore * tiling->elemNum;
        numPerCoreLoop = groups / innerNumPerCore;
        numPerCoreTail = groups - numPerCoreLoop * innerNumPerCore;
        for (int64_t loopIdx = 0; loopIdx < numPerCoreLoop; loopIdx++) {
            int64_t groupOffset = loopIdx * innerNumPerCore;
            ProcessGroupNormSilu(groupOffset, innerNumPerCore);
            CopyOutMeanAndRstdAlign(groupOffset);
        }
        if (numPerCoreTail > 0) {
            int64_t groupTailOffset = numPerCoreLoop * innerNumPerCore;
            ProcessGroupNormSilu(groupTailOffset, numPerCoreTail);
            CopyOutMeanAndRstd(groupTailOffset, numPerCoreTail);
        }
    }
}

template <typename T>
__aicore__ inline void GroupNormSiluHW1B32<T>::ProcessYWithEqualC(const int64_t& groups)
{
    int64_t loopC = tiling->shapeC / processSize;
    int64_t cTail = tiling->shapeC - loopC * processSize;
    for (int64_t gIdx = 0; gIdx < groups; gIdx++) {
        PipeBarrier<PIPE_ALL>();
        for (int64_t cIdx = 0; cIdx < loopC; cIdx++) {
            PipeBarrier<PIPE_ALL>();
            LocalTensor<T> xUb = outQueueY.AllocTensor<T>();
            PipeBarrier<PIPE_ALL>();
            DataCopy(xUb, betaGm[cIdx * processSize], processSize);
            PipeBarrier<PIPE_ALL>();
            outQueueY.EnQue(xUb);
            PipeBarrier<PIPE_ALL>();
            LocalTensor<T> xUb1 = outQueueY.DeQue<T>();
            PipeBarrier<PIPE_ALL>();
            CalcSilu(xUb1, processSize);
            PipeBarrier<PIPE_ALL>();
            DataCopy(yGm[gmOffset + gIdx * tiling->shapeC + cIdx * processSize], xUb1, processSize);
            PipeBarrier<PIPE_ALL>();
            outQueueY.FreeTensor(xUb1);
        }
        if (cTail > 0) {
            LocalTensor<T> xUb = outQueueY.AllocTensor<T>();
            this->template CopyInData<T, false>(xUb, betaGm[loopC * processSize], cTail);
            PipeBarrier<PIPE_ALL>();
            outQueueY.EnQue(xUb);
            PipeBarrier<PIPE_ALL>();
            LocalTensor<T> xUb1 = outQueueY.DeQue<T>();
            PipeBarrier<PIPE_ALL>();
            CalcSilu(xUb1, cTail);
            PipeBarrier<PIPE_ALL>();
            this->template CopyOutData<T, false>(
                yGm[gmOffset + gIdx * tiling->shapeC + loopC * processSize], xUb1, cTail);
            PipeBarrier<PIPE_ALL>();
            outQueueY.FreeTensor(xUb1);
        }
    }
}

template <typename T>
__aicore__ inline void GroupNormSiluHW1B32<T>::CalcSilu(LocalTensor<T>& betaUb, const int64_t& computeNum)
{
    if (tiling->activateSilu) {
        LocalTensor<T> gammaUb = inQueueGamma.AllocTensor<T>();
        Muls(gammaUb, betaUb, negativeOne, computeNum);
        PipeBarrier<PIPE_V>();
        Exp(gammaUb, gammaUb, computeNum);
        PipeBarrier<PIPE_V>();
        Adds(gammaUb, gammaUb, scalarOne, computeNum);
        PipeBarrier<PIPE_V>();
        Div(betaUb, betaUb, gammaUb, computeNum);
        PipeBarrier<PIPE_V>();
        inQueueGamma.FreeTensor(gammaUb);
    }
}

template <typename T>
__aicore__ inline void GroupNormSiluHW1B32<T>::ProcessMeanWithEqualC(const int64_t& groups)
{
    int64_t loopC = groups * tiling->shapeC / processSize;
    int64_t cTail = groups * tiling->shapeC - loopC * processSize;
    for (int64_t cIdx = 0; cIdx < loopC; cIdx++) {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<T> xUb = inQueueX.AllocTensor<T>();
        DataCopy(xUb, xGm[gmOffset + cIdx * processSize], processSize);
        PipeBarrier<PIPE_ALL>();
        inQueueX.EnQue(xUb);
        LocalTensor<T> xUb1 = inQueueX.DeQue<T>();
        PipeBarrier<PIPE_ALL>();
        DataCopy(meanGm[gmOffset + cIdx * processSize], xUb1, processSize);
        PipeBarrier<PIPE_ALL>();
        inQueueX.FreeTensor(xUb1);
    }
    if (cTail > 0) {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<T> xUb = inQueueX.AllocTensor<T>();
        this->template CopyInData<T, false>(xUb, xGm[gmOffset + loopC * processSize], cTail);
        PipeBarrier<PIPE_ALL>();
        inQueueX.EnQue(xUb);
        LocalTensor<T> xUb1 = inQueueX.DeQue<T>();
        PipeBarrier<PIPE_ALL>();
        this->template CopyOutData<T, false>(meanGm[gmOffset + loopC * processSize], xUb1, cTail);
        PipeBarrier<PIPE_ALL>();
        inQueueX.FreeTensor(xUb1);
    }
}

template <typename T>
__aicore__ inline void GroupNormSiluHW1B32<T>::ProcessRstdWithEqualC(const int64_t& groups)
{
    int64_t loopC = groups * tiling->shapeC / processSize;
    int64_t cTail = groups * tiling->shapeC - loopC * processSize;
    Duplicate<T>(tempX.Get<T>(), float(1.0) / sqrt(tiling->epsilon), processSize);
    PipeBarrier<PIPE_ALL>();
    for (int64_t cIdx = 0; cIdx < loopC; cIdx++) {
        PipeBarrier<PIPE_ALL>();
        DataCopy(rstdGm[gmOffset + cIdx * processSize], tempX.Get<T>(), processSize);
    }
    if (cTail > 0) {
        PipeBarrier<PIPE_ALL>();
        this->template CopyOutData<T, false>(rstdGm[gmOffset + loopC * processSize], tempX.Get<T>(), cTail);
    }
}

template <typename T>
__aicore__ inline void GroupNormSiluHW1B32<T>::ProcessGroupNormSilu(const int64_t& offset, const int64_t& groups)
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
__aicore__ inline void GroupNormSiluHW1B32<T>::ComputeForMeanAlign(
    LocalTensor<T>& meanOut, LocalTensor<T>& rstdOut, const int64_t& offset, const int64_t& groups)
{
    LocalTensor<float> x2Ub = tempX.Get<float>();
    LocalTensor<float> tmpMean = tmpTensor.Get<float>();
    int64_t sumMeanNum = (tiling->realCoreNum - 1) * tiling->numPerCore + tiling->numLastCore;
    for (int64_t groupIdx = 0; groupIdx < elementsPerBlock - groups; groupIdx++) {
        int64_t groupStartIdx = offset + groups + groupIdx;
        if (blockIdx * tiling->numPerCore + groupStartIdx >= sumMeanNum) {
            break;
        }
        // accumulate and reduceSum x && x^2
        ReduceSumXAndX2(groupStartIdx, reduseSumNum);
        // calculate mean && variance
        float mean = tmpMean.GetValue(0);
        float rstd = x2Ub.GetValue(0);
        mean = mean * numRec;
        rstd = float(1.0) / sqrt(rstd * numRec - mean * mean + tiling->epsilon);
        meanOut.SetValue(groups + groupIdx, mean);
        rstdOut.SetValue(groups + groupIdx, rstd);
    }
}

template <typename T>
__aicore__ inline void GroupNormSiluHW1B32<T>::ReduceSumXAndX2(const int64_t& groupIdx, const int64_t& reduseNum)
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
__aicore__ inline void GroupNormSiluHW1B32<T>::CopyInX(const int64_t startAddr, const int64_t copyNum)
{
    LocalTensor<T> xUb = inQueueX.AllocTensor<T>();
    this->template CopyInData<T, isAlign>(xUb, xGm[startAddr], copyNum);
    inQueueX.EnQue(xUb);
}

template <typename T>
__aicore__ inline void GroupNormSiluHW1B32<T>::ComputeSum(LocalTensor<T>& x2Buf, const int64_t& processNum)
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
__aicore__ inline void GroupNormSiluHW1B32<T>::ComputeGroupNormSilu(
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
    int64_t StartAddr = gmOffset + groupIdx * tiling->elemNum;
    int64_t gammeStartAddr = ((blockIdx * tiling->numPerCore + groupIdx) % tiling->numGroups) * tiling->shapeD;
    // middle
    for (int loopIdx = 0; loopIdx < tiling->loopNum - 1; loopIdx++) {
        int64_t loopStartAddr = StartAddr + loopIdx * tiling->processSize;
        int64_t loopgammeStartAddr = gammeStartAddr + loopIdx * tiling->processSize;
        CopyInGammaBeta(loopgammeStartAddr, tiling->processSize);
        CopyInX(loopStartAddr, tiling->processSize);
        CalculateGroupNormSilu(mean, rstd, tiling->processSize);
        CopyOutY(loopStartAddr, tiling->processSize);
    }
    // tail
    int64_t tailStartAddr = StartAddr + (tiling->loopNum - 1) * tiling->processSize;
    int64_t tailGammeStartAddr = gammeStartAddr + (tiling->loopNum - 1) * tiling->processSize;
    CopyInGammaBeta<false>(tailGammeStartAddr, tiling->loopTail);
    CopyInX<false>(tailStartAddr, tiling->loopTail);
    CalculateGroupNormSilu(mean, rstd, tiling->loopTail);
    CopyOutYWithPad(mean, rstd, tailStartAddr, tiling->loopTail);
}

template <typename T>
template <bool isAlign>
__aicore__ inline void GroupNormSiluHW1B32<T>::CopyInGammaBeta(const int64_t startAddr, const int64_t processNum)
{
    LocalTensor<T> gamma = inQueueGamma.AllocTensor<T>();
    this->template CopyInData<T, isAlign>(gamma, gammaGm[startAddr], processNum);
    inQueueGamma.EnQue(gamma);

    LocalTensor<T> beta = inQueueBeta.AllocTensor<T>();
    this->template CopyInData<T, isAlign>(beta, betaGm[startAddr], processNum);
    inQueueBeta.EnQue(beta);
}

template <typename T>
__aicore__ inline void GroupNormSiluHW1B32<T>::CalculateGroupNormSilu(
    const float& mean, const float& rstd, const int64_t& computeNum)
{
    LocalTensor<T> xUb = inQueueX.DeQue<T>();
    LocalTensor<T> gammaUb = inQueueGamma.DeQue<T>();
    LocalTensor<T> betaUb = inQueueBeta.DeQue<T>();
    LocalTensor<T> outY = outQueueY.AllocTensor<T>();

    Adds(xUb, xUb, -mean, computeNum);
    PipeBarrier<PIPE_V>();
    Muls(xUb, xUb, rstd, computeNum);
    PipeBarrier<PIPE_V>();
    Mul(xUb, xUb, gammaUb, computeNum);
    PipeBarrier<PIPE_V>();
    if (tiling->activateSilu) {
        Add(xUb, xUb, betaUb, computeNum);
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
        Add(outY, xUb, betaUb, computeNum);
        PipeBarrier<PIPE_V>();
    }
    outQueueY.EnQue(outY);
    inQueueX.FreeTensor(xUb);
    inQueueGamma.FreeTensor(gammaUb);
    inQueueBeta.FreeTensor(betaUb);
}

template <typename T>
__aicore__ inline void GroupNormSiluHW1B32<T>::CopyOutY(const int64_t& startAddr, const int64_t& copyNum)
{
    LocalTensor<T> outY = outQueueY.DeQue<T>();
    DataCopy(yGm[startAddr], outY, copyNum);
    outQueueY.FreeTensor(outY);
}

template <typename T>
__aicore__ inline void GroupNormSiluHW1B32<T>::CopyOutYWithPad(
    const float& mean, const float& rstd, const int64_t& startAddr, const int64_t& copyNum)
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
            DataCopy(yGm[startAddr], outY, copyNum);
            outQueueY.FreeTensor(outY);
            CopyInX(startAddr + copyNum - elementsPerBlock, elementsPerBlock);
            CalculateGroupNormSilu(mean, rstd, elementsPerBlock);
            CopyOutY(startAddr + copyNum - elementsPerBlock, elementsPerBlock);
        }
    }
#endif
}

template <typename T>
__aicore__ inline void GroupNormSiluHW1B32<T>::CopyOutMeanAndRstd(const int64_t& startAddr, const int64_t& copyNum)
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
#else
    // when not support DataCopyPad, use DataCopy, copyNum must be align process
    if (copyNum % elementsPerBlock == 0) {
        DataCopy(meanGm[blockIdx * tiling->numPerCore + startAddr], meanOut, copyNum);
        DataCopy(rstdGm[blockIdx * tiling->numPerCore + startAddr], rstdOut, copyNum);
    } else if (copyNum > elementsPerBlock) {
        // DataCopy will round down by 32B when copyNum is not align
        DataCopy(meanGm[blockIdx * tiling->numPerCore + startAddr], meanOut, copyNum);
        DataCopy(rstdGm[blockIdx * tiling->numPerCore + startAddr], rstdOut, copyNum);
        LocalTensor<T> meanTmp = tmpTensor.Get<T>();
        LocalTensor<T> rstdTmp = workLocal.Get<T>();
        // copy out last loop overlap
        for (int64_t idx = 0; idx < elementsPerBlock; idx++) {
            meanTmp.SetValue(idx, meanOut.GetValue(copyNum - elementsPerBlock + idx));
            rstdTmp.SetValue(idx, rstdOut.GetValue(copyNum - elementsPerBlock + idx));
        }
        DataCopy(
            meanGm[blockIdx * tiling->numPerCore + startAddr + copyNum - elementsPerBlock], meanTmp, elementsPerBlock);
        DataCopy(
            rstdGm[blockIdx * tiling->numPerCore + startAddr + copyNum - elementsPerBlock], rstdTmp, elementsPerBlock);
    } else {
        // copyNum is less than elementsPerBlock
        ComputeForMeanAlign(meanOut, rstdOut, startAddr, copyNum);
        DataCopy(meanGm[blockIdx * tiling->numPerCore + startAddr], meanOut, elementsPerBlock);
        DataCopy(rstdGm[blockIdx * tiling->numPerCore + startAddr], rstdOut, elementsPerBlock);
    }
#endif
    outQueueMean.FreeTensor(meanOut);
    outQueueRstd.FreeTensor(rstdOut);
}

template <typename T>
__aicore__ inline void GroupNormSiluHW1B32<T>::CopyOutMeanAndRstdAlign(const int64_t& startOffset)
{
    LocalTensor<T> meanOut = outQueueMean.DeQue<T>();
    LocalTensor<T> rstdOut = outQueueRstd.DeQue<T>();
    DataCopy(meanGm[blockIdx * tiling->numPerCore + startOffset], meanOut, innerNumPerCore);
    DataCopy(rstdGm[blockIdx * tiling->numPerCore + startOffset], rstdOut, innerNumPerCore);
    outQueueMean.FreeTensor(meanOut);
    outQueueRstd.FreeTensor(rstdOut);
}
} // namespace GroupNormSilu

#endif // GROUP_NORM_SILU_HW1_B32_H_