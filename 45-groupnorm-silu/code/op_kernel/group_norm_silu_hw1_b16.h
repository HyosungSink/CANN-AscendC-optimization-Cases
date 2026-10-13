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
 * \file group_norm_silu_hw1_b16.h
 * \brief
 */

#ifndef GROUP_NORM_SILU_HW1_B16_H_
#define GROUP_NORM_SILU_HW1_B16_H_

#include "group_norm_silu_base.h"

namespace GroupNormSilu {
using namespace AscendC;

template <typename T1, typename T2>
class GroupNormSiluHW1B16 : public GroupNormSiluBase<T1>
{
public:
    __aicore__ inline GroupNormSiluHW1B16(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
        const GroupNormSiluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    // function
    template <bool isAlign = true>
    __aicore__ inline void CopyInX(const int64_t startAddr, const int64_t copyNum);
    __aicore__ inline void CopyInXPad(const int64_t startAddr, const int64_t copyNum);
    __aicore__ inline void ComputeSum(const int64_t& processNum);
    __aicore__ inline void AccumulateXAndX2(const int64_t& groupIdx);
    __aicore__ inline void ReduceSumXAndX2(const int64_t& reduseNum);
    template <bool isAlign = true>
    __aicore__ inline void CopyInGammaBeta16(const int64_t startAddr, const int64_t copyNum);
    template <bool isAlign = true>
    __aicore__ inline void CopyInGammaBeta32(const int64_t startAddr, const int64_t copyNum);
    __aicore__ inline void CalculateGroupNormSilu(const float& mean, const float& rstd, const int64_t& computeNum);
    __aicore__ inline void ComputeGroupNormSilu(
        const int64_t& groupIdx, LocalTensor<float>& meanUb, LocalTensor<float>& rstdUb);
    __aicore__ inline void ProcessPerCore(const int64_t& groups);
    __aicore__ inline void ProcessYWithEqualC(const int64_t& groups);
    __aicore__ inline void ProcessYWithSameType(const int64_t& groups, const int64_t& loopC, const int64_t& cTail);
    __aicore__ inline void ProcessYWithdiffType(const int64_t& groups, const int64_t& loopC, const int64_t& cTail);
    __aicore__ inline void CalcSiluWithCast(LocalTensor<T1>& betaUb, const int64_t& computeNum);
    __aicore__ inline void CalcSilu(const int64_t& computeNum);
    __aicore__ inline void ProcessMeanWithEqualC(const int64_t& groups);
    __aicore__ inline void ProcessRstdWithEqualC(const int64_t& groups);
    __aicore__ inline void ProcessGroupNormSilu(const int64_t& offset, const int64_t& groups);
    __aicore__ inline void ComputeForMeanAlign(
        LocalTensor<T1>& meanOut, LocalTensor<T1>& rstdOut, const int64_t& offset, const int64_t& groups);
    __aicore__ inline void CopyOutY(const int64_t& startAddr, const int64_t& copyNum);
    __aicore__ inline void CopyOutYWithPad(
        const float& mean, const float& rstd, const int64_t& startAddr, const int64_t& copyNum);
    __aicore__ inline void CastMeanAndRstd(
        LocalTensor<float>& meanUb, LocalTensor<float>& rstdUb, const int64_t& copyNum);
    __aicore__ inline void CopyOutMeanAndRstd(const int64_t& startAddr, const int64_t& copyNum);
    __aicore__ inline void CopyOutMeanAndRstdAlign(const int64_t& startAddr);
    // constant
    constexpr static float negativeOne = static_cast<float>(-1.0);
    constexpr static float scalarZero = 0.0;
    constexpr static float scalarOne = 1.0;
    constexpr static int64_t processSize = 8192;
    constexpr static int64_t elementsPerBlockT1 = 32 / sizeof(T1);
    constexpr static int64_t elementsPerBlockT2 = 32 / sizeof(T2);
    constexpr static int64_t elementsPerBlockFloat = 8;
    constexpr static int64_t innerNumPerCore = 32;

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueGamma;
    TQue<QuePosition::VECIN, 1> inQueueBeta;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueMean;
    TQue<QuePosition::VECOUT, 1> outQueueRstd;

    GlobalTensor<T1> xGm;
    GlobalTensor<T2> gammaGm, betaGm;
    GlobalTensor<float> gammaGmFloat, betaGmFloat;
    GlobalTensor<T1> yGm, meanGm, rstdGm;

    TBuf<QuePosition::VECCALC> xBuf32;
    TBuf<QuePosition::VECCALC> x1Buf32;
    TBuf<QuePosition::VECCALC> x2Buf32;
    TBuf<QuePosition::VECCALC> gammaBuf32;
    TBuf<QuePosition::VECCALC> betaBuf32;
    TBuf<QuePosition::VECCALC> meanBuf32;
    TBuf<QuePosition::VECCALC> rstdBuf32;
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

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::Init(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
    const GroupNormSiluTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    tiling = tilingData;
    xGm.SetGlobalBuffer((__gm__ T1*)x);
    if (sizeof(T2) == sizeof(float)) {
        gammaGmFloat.SetGlobalBuffer((__gm__ float*)gamma);
        betaGmFloat.SetGlobalBuffer((__gm__ float*)beta);
    } else {
        gammaGm.SetGlobalBuffer((__gm__ T2*)gamma);
        betaGm.SetGlobalBuffer((__gm__ T2*)beta);
    }
    yGm.SetGlobalBuffer((__gm__ T1*)y);
    meanGm.SetGlobalBuffer((__gm__ T1*)mean);
    rstdGm.SetGlobalBuffer((__gm__ T1*)rstd);

    pipe.InitBuffer(inQueueX, 1, tiling->processSize * sizeof(T1));
    pipe.InitBuffer(inQueueGamma, 1, tiling->processSize * sizeof(T1));
    pipe.InitBuffer(inQueueBeta, 1, tiling->processSize * sizeof(T1));
    pipe.InitBuffer(outQueueY, 1, tiling->processSize * sizeof(T1));
    pipe.InitBuffer(outQueueMean, 1, innerNumPerCore * sizeof(T1));
    pipe.InitBuffer(outQueueRstd, 1, innerNumPerCore * sizeof(T1));

    pipe.InitBuffer(xBuf32, tiling->processSize * sizeof(float));
    pipe.InitBuffer(x1Buf32, tiling->processSize * sizeof(float));
    pipe.InitBuffer(x2Buf32, tiling->processSize * sizeof(float));

    pipe.InitBuffer(meanBuf32, innerNumPerCore * sizeof(float));
    pipe.InitBuffer(rstdBuf32, innerNumPerCore * sizeof(float));
    pipe.InitBuffer(gammaBuf32, elementsPerBlockFloat * sizeof(float));
    pipe.InitBuffer(betaBuf32, elementsPerBlockFloat * sizeof(float));

    // one block
    pipe.InitBuffer(tmpTensor, elementsPerBlockFloat * sizeof(float));
    numRec = float(scalarOne) / float(tiling->elemNum);
    if (tiling->loopNum > 1) {
        reduseSumNum = tiling->processSize;
    } else {
        reduseSumNum = tiling->loopTail;
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::Process()
{
    if (blockIdx < tiling->realCoreNum - 1) {
        ProcessPerCore(tiling->numPerCore);
    } else if (blockIdx == tiling->realCoreNum - 1) { // process last core
        ProcessPerCore(tiling->numLastCore);
    } else {
        return;
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::ProcessPerCore(const int64_t& groups)
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

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::ProcessYWithEqualC(const int64_t& groups)
{
    int64_t loopC = tiling->shapeC / processSize;
    int64_t cTail = tiling->shapeC - loopC * processSize;
    if (sizeof(T1) == sizeof(T2)) {
        ProcessYWithSameType(groups, loopC, cTail);
    } else {
        ProcessYWithdiffType(groups, loopC, cTail);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::ProcessYWithSameType(
    const int64_t& groups, const int64_t& loopC, const int64_t& cTail)
{
    int64_t cTailAlign = this->CeilDiv(cTail, elementsPerBlockT1) * elementsPerBlockT1;
    for (int64_t gIdx = 0; gIdx < groups; gIdx++) {
        PipeBarrier<PIPE_ALL>();
        for (int64_t cIdx = 0; cIdx < loopC; cIdx++) {
            PipeBarrier<PIPE_ALL>();
            LocalTensor<T2> xUb = outQueueY.AllocTensor<T2>();
            DataCopy(xUb, betaGm[cIdx * processSize], processSize);
            PipeBarrier<PIPE_ALL>();
            outQueueY.EnQue(xUb);
            LocalTensor<T1> xUb1 = outQueueY.DeQue<T1>();
            PipeBarrier<PIPE_ALL>();
            CalcSiluWithCast(xUb1, processSize);
            PipeBarrier<PIPE_ALL>();
            DataCopy(yGm[gmOffset + gIdx * tiling->shapeC + cIdx * processSize], xUb1, processSize);
            PipeBarrier<PIPE_ALL>();
            outQueueY.FreeTensor(xUb1);
        }
        if (cTail > 0) {
            LocalTensor<T2> xUb = outQueueY.AllocTensor<T2>();
            this->template CopyInData<T2, false>(xUb, betaGm[loopC * processSize], cTail);
            PipeBarrier<PIPE_ALL>();
            outQueueY.EnQue(xUb);
            LocalTensor<T1> xUb1 = outQueueY.DeQue<T1>();
            PipeBarrier<PIPE_ALL>();
            CalcSiluWithCast(xUb1, cTailAlign);
            PipeBarrier<PIPE_ALL>();
            this->template CopyOutData<T1, false>(
                yGm[gmOffset + gIdx * tiling->shapeC + loopC * processSize], xUb1, cTail);
            PipeBarrier<PIPE_ALL>();
            outQueueY.FreeTensor(xUb1);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::ProcessYWithdiffType(
    const int64_t& groups, const int64_t& loopC, const int64_t& cTail)
{
    int64_t cTailAlign = this->CeilDiv(cTail, elementsPerBlockT1) * elementsPerBlockT1;
    for (int64_t gIdx = 0; gIdx < groups; gIdx++) {
        PipeBarrier<PIPE_ALL>();
        for (int64_t cIdx = 0; cIdx < loopC; cIdx++) {
            PipeBarrier<PIPE_ALL>();
            DataCopy(xBuf32.Get<float>(), betaGmFloat[cIdx * processSize], processSize);
            PipeBarrier<PIPE_ALL>();
            CalcSilu(processSize);
            PipeBarrier<PIPE_ALL>();
            LocalTensor<T1> xUb = outQueueY.AllocTensor<T1>();
            Cast(xUb, xBuf32.Get<float>(), this->GetRoundMode(), processSize);
            PipeBarrier<PIPE_ALL>();
            outQueueY.EnQue(xUb);
            LocalTensor<T1> xUb1 = outQueueY.DeQue<T1>();
            PipeBarrier<PIPE_ALL>();
            DataCopy(yGm[gmOffset + gIdx * tiling->shapeC + cIdx * processSize], xUb1, processSize);
            PipeBarrier<PIPE_ALL>();
            outQueueY.FreeTensor(xUb1);
        }
        if (cTail > 0) {
            this->template CopyInData<float, false>(xBuf32.Get<float>(), betaGmFloat[loopC * processSize], cTail);
            PipeBarrier<PIPE_ALL>();
            CalcSilu(cTailAlign);
            PipeBarrier<PIPE_ALL>();
            LocalTensor<T1> xUb = outQueueY.AllocTensor<T1>();
            Cast(xUb, xBuf32.Get<float>(), this->GetRoundMode(), cTailAlign);
            PipeBarrier<PIPE_ALL>();
            outQueueY.EnQue(xUb);
            LocalTensor<T1> xUb1 = outQueueY.DeQue<T1>();
            PipeBarrier<PIPE_ALL>();
            this->template CopyOutData<T1, false>(
                yGm[gmOffset + gIdx * tiling->shapeC + loopC * processSize], xUb1, cTail);
            PipeBarrier<PIPE_ALL>();
            outQueueY.FreeTensor(xUb1);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::CalcSiluWithCast(LocalTensor<T1>& betaUb, const int64_t& computeNum)
{
    if (tiling->activateSilu) {
        Cast(xBuf32.Get<float>(), betaUb, RoundMode::CAST_NONE, computeNum);
        PipeBarrier<PIPE_V>();
        Muls(x2Buf32.Get<float>(), xBuf32.Get<float>(), negativeOne, computeNum);
        PipeBarrier<PIPE_V>();
        Exp(x2Buf32.Get<float>(), x2Buf32.Get<float>(), computeNum);
        PipeBarrier<PIPE_V>();
        Adds(x2Buf32.Get<float>(), x2Buf32.Get<float>(), scalarOne, computeNum);
        PipeBarrier<PIPE_V>();
        Div(xBuf32.Get<float>(), xBuf32.Get<float>(), x2Buf32.Get<float>(), computeNum);
        PipeBarrier<PIPE_V>();
        Cast(betaUb, xBuf32.Get<float>(), this->GetRoundMode(), computeNum);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::CalcSilu(const int64_t& computeNum)
{
    if (tiling->activateSilu) {
        Muls(x2Buf32.Get<float>(), xBuf32.Get<float>(), negativeOne, computeNum);
        PipeBarrier<PIPE_V>();
        Exp(x2Buf32.Get<float>(), x2Buf32.Get<float>(), computeNum);
        PipeBarrier<PIPE_V>();
        Adds(x2Buf32.Get<float>(), x2Buf32.Get<float>(), scalarOne, computeNum);
        PipeBarrier<PIPE_V>();
        Div(xBuf32.Get<float>(), xBuf32.Get<float>(), x2Buf32.Get<float>(), computeNum);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::ProcessMeanWithEqualC(const int64_t& groups)
{
    int64_t loopC = groups * tiling->shapeC / processSize;
    int64_t cTail = groups * tiling->shapeC - loopC * processSize;
    for (int64_t cIdx = 0; cIdx < loopC; cIdx++) {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<T1> meanUb = inQueueGamma.AllocTensor<T1>();
        DataCopy(meanUb, xGm[gmOffset + cIdx * processSize], processSize);
        PipeBarrier<PIPE_ALL>();
        inQueueGamma.EnQue(meanUb);
        LocalTensor<T1> meanUb1 = inQueueGamma.DeQue<T1>();
        PipeBarrier<PIPE_ALL>();
        DataCopy(meanGm[gmOffset + cIdx * processSize], meanUb1, processSize);
        PipeBarrier<PIPE_ALL>();
        inQueueGamma.FreeTensor(meanUb1);
    }
    if (cTail > 0) {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<T1> meanUb = inQueueGamma.AllocTensor<T1>();
        PipeBarrier<PIPE_ALL>();
        this->template CopyInData<T1, false>(meanUb, xGm[gmOffset + loopC * processSize], cTail);
        PipeBarrier<PIPE_ALL>();
        inQueueGamma.EnQue(meanUb);
        LocalTensor<T1> meanUb1 = inQueueGamma.DeQue<T1>();
        PipeBarrier<PIPE_ALL>();
        this->template CopyOutData<T1, false>(meanGm[gmOffset + loopC * processSize], meanUb1, cTail);
        PipeBarrier<PIPE_ALL>();
        inQueueGamma.FreeTensor(meanUb1);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::ProcessRstdWithEqualC(const int64_t& groups)
{
    int64_t loopC = groups * tiling->shapeC / processSize;
    int64_t cTail = groups * tiling->shapeC - loopC * processSize;
    LocalTensor<T1> rstdUb = inQueueBeta.AllocTensor<T1>();
    Duplicate<float>(x1Buf32.Get<float>(), float(1.0) / sqrt(tiling->epsilon), processSize);
    PipeBarrier<PIPE_ALL>();
    Cast(rstdUb, x1Buf32.Get<float>(), this->GetRoundMode(), processSize);
    PipeBarrier<PIPE_ALL>();
    for (int64_t cIdx = 0; cIdx < loopC; cIdx++) {
        PipeBarrier<PIPE_ALL>();
        DataCopy(rstdGm[gmOffset + cIdx * processSize], rstdUb, processSize);
    }
    if (cTail > 0) {
        PipeBarrier<PIPE_ALL>();
        this->template CopyOutData<T1, false>(rstdGm[gmOffset + loopC * processSize], rstdUb, cTail);
    }
    inQueueBeta.FreeTensor(rstdUb);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::ProcessGroupNormSilu(const int64_t& offset, const int64_t& groups)
{
    LocalTensor<float> meanUb = meanBuf32.Get<float>();
    LocalTensor<float> rstdUb = rstdBuf32.Get<float>();
    for (int64_t groupIdx = 0; groupIdx < groups; groupIdx++) {
        int64_t groupStartIdx = offset + groupIdx;
        // accumulate x && x^2
        AccumulateXAndX2(groupStartIdx);
        // reduceSum x && x^2
        ReduceSumXAndX2(reduseSumNum);
        // calculate GroupNormSilu
        ComputeGroupNormSilu(groupStartIdx, meanUb, rstdUb);
    }
    CastMeanAndRstd(meanUb, rstdUb, groups);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::ComputeForMeanAlign(
    LocalTensor<T1>& meanOut, LocalTensor<T1>& rstdOut, const int64_t& offset, const int64_t& groups)
{
    LocalTensor<float> meanUb = meanBuf32.Get<float>();
    LocalTensor<float> rstdUb = rstdBuf32.Get<float>();
    LocalTensor<float> x2Ub32 = x2Buf32.Get<float>();
    LocalTensor<float> tmpMean = tmpTensor.Get<float>();
    int64_t castNum = groups;
    int64_t sumMeanNum = (tiling->realCoreNum - 1) * tiling->numPerCore + tiling->numLastCore;
    for (int64_t groupIdx = 0; groupIdx < elementsPerBlockT1 - groups; groupIdx++) {
        int64_t groupStartIdx = offset + groups + groupIdx;
        if (blockIdx * tiling->numPerCore + groupStartIdx >= sumMeanNum) {
            break;
        }
        // accumulate x && x^2
        AccumulateXAndX2(groupStartIdx);
        // reduceSum x && x^2
        ReduceSumXAndX2(reduseSumNum);
        PipeBarrier<PIPE_V>();
        // calculate mean && variance
        float mean = tmpMean.GetValue(0);
        float rstd = x2Ub32.GetValue(0);
        mean = mean * numRec;
        rstd = float(1.0) / sqrt(rstd * numRec - mean * mean + tiling->epsilon);
        meanUb.SetValue(groups + groupIdx, mean);
        rstdUb.SetValue(groups + groupIdx, rstd);
        castNum++;
    }
    event_t eventS2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventS2V);
    WaitFlag<HardEvent::S_V>(eventS2V);
    Cast(meanOut, meanUb, this->GetRoundMode(), castNum);
    Cast(rstdOut, rstdUb, this->GetRoundMode(), castNum);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::AccumulateXAndX2(const int64_t& groupIdx)
{
    // first init x1Buf32  x2Buf32
    Duplicate<float>(x1Buf32.Get<float>(), scalarZero, tiling->processSize);
    Duplicate<float>(x2Buf32.Get<float>(), scalarZero, tiling->processSize);
    int64_t StartAddr = gmOffset + groupIdx * tiling->elemNum;
    // middle
    for (int loopIdx = 0; loopIdx < tiling->loopNum - 1; loopIdx++) {
        int64_t loopStartAddr = StartAddr + loopIdx * tiling->processSize;
        CopyInX(loopStartAddr, tiling->processSize);
        ComputeSum(tiling->processSize);
    }
    // tail
    int64_t tailStartAddr = StartAddr + (tiling->loopNum - 1) * tiling->processSize;
    CopyInX<false>(tailStartAddr, tiling->loopTail);
    ComputeSum(tiling->loopTail);
}

template <typename T1, typename T2>
template <bool isAlign>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::CopyInX(const int64_t startAddr, const int64_t copyNum)
{
    LocalTensor<T1> xUb = inQueueX.AllocTensor<T1>();
    this->template CopyInData<T1, isAlign>(xUb, xGm[startAddr], copyNum);
    inQueueX.EnQue(xUb);
    LocalTensor<T1> xUb1 = inQueueX.DeQue<T1>();
    Cast(xBuf32.Get<float>(), xUb1, RoundMode::CAST_NONE, copyNum);
    PipeBarrier<PIPE_V>();
    inQueueX.FreeTensor(xUb1);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::CopyInXPad(const int64_t startAddr, const int64_t copyNum)
{
    LocalTensor<T1> xUb = inQueueX.AllocTensor<T1>();
    DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
    copyParams.blockLen = copyNum * sizeof(T1);
    DataCopyPadExtParams<T1> padParams = {false, 0, 0, 0};
    DataCopyPad(xUb, xGm[startAddr], copyParams, padParams);
    inQueueX.EnQue(xUb);
    LocalTensor<T1> xUb1 = inQueueX.DeQue<T1>();
    Cast(xBuf32.Get<float>(), xUb1, RoundMode::CAST_NONE, copyNum);
    PipeBarrier<PIPE_V>();
    inQueueX.FreeTensor(xUb1);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::ComputeSum(const int64_t& processNum)
{
    // add x
    Add(x1Buf32.Get<float>(), x1Buf32.Get<float>(), xBuf32.Get<float>(), processNum);
    // x^2
    Mul(xBuf32.Get<float>(), xBuf32.Get<float>(), xBuf32.Get<float>(), processNum);
    PipeBarrier<PIPE_V>();
    // add x^2
    Add(x2Buf32.Get<float>(), x2Buf32.Get<float>(), xBuf32.Get<float>(), processNum);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::ReduceSumXAndX2(const int64_t& reduseNum)
{
    // reduseSum(x)
    ReduceSum<float>(tmpTensor.Get<float>(), x1Buf32.Get<float>(), xBuf32.Get<float>(), reduseNum);
    // reduseSum(x^2)
    ReduceSum<float>(x2Buf32.Get<float>(), x2Buf32.Get<float>(), xBuf32.Get<float>(), reduseNum);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::ComputeGroupNormSilu(
    const int64_t& groupIdx, LocalTensor<float>& meanUb, LocalTensor<float>& rstdUb)
{
    PipeBarrier<PIPE_V>();
    LocalTensor<float> x2Ub32 = x2Buf32.Get<float>();
    LocalTensor<float> tmpMean = tmpTensor.Get<float>();
    // calculate mean && variance
    float mean = tmpMean.GetValue(0);
    float rstd = x2Ub32.GetValue(0);
    mean = mean * numRec;
    rstd = float(1.0) / sqrt(rstd * numRec - mean * mean + tiling->epsilon);
    meanUb.SetValue(groupIdx % innerNumPerCore, mean);
    rstdUb.SetValue(groupIdx % innerNumPerCore, rstd);

    int64_t StartAddr = gmOffset + groupIdx * tiling->elemNum;
    int64_t gammeStartAddr = ((blockIdx * tiling->numPerCore + groupIdx) % tiling->numGroups) * tiling->shapeD;
    for (int loopIdx = 0; loopIdx < tiling->loopNum - 1; loopIdx++) {
        int64_t loopStartAddr = StartAddr + loopIdx * tiling->processSize;
        int64_t loopgammeStartAddr = gammeStartAddr + loopIdx * tiling->processSize;
        if (sizeof(T2) != sizeof(float)) {
            CopyInGammaBeta16(loopgammeStartAddr, tiling->processSize);
        } else {
            CopyInGammaBeta32(loopgammeStartAddr, tiling->processSize);
        }
        CopyInX(loopStartAddr, tiling->processSize);
        CalculateGroupNormSilu(mean, rstd, tiling->processSize);
        CopyOutY(loopStartAddr, tiling->processSize);
    }
    // tail
    int64_t tailStartAddr = StartAddr + (tiling->loopNum - 1) * tiling->processSize;
    int64_t tailGammeStartAddr = gammeStartAddr + (tiling->loopNum - 1) * tiling->processSize;
    if (sizeof(T2) != sizeof(float)) {
        CopyInGammaBeta16<false>(tailGammeStartAddr, tiling->loopTail);
    } else {
        CopyInGammaBeta32<false>(tailGammeStartAddr, tiling->loopTail);
    }
    CopyInX<false>(tailStartAddr, tiling->loopTail);
    CalculateGroupNormSilu(mean, rstd, tiling->loopTail);
    CopyOutYWithPad(mean, rstd, tailStartAddr, tiling->loopTail);
}

template <typename T1, typename T2>
template <bool isAlign>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::CopyInGammaBeta16(const int64_t startAddr, const int64_t copyNum)
{
    int64_t copyNumAlign = this->CeilDiv(copyNum, elementsPerBlockT1) * elementsPerBlockT1;
    LocalTensor<T2> gamma = inQueueGamma.AllocTensor<T2>();
    this->template CopyInData<T2, isAlign>(gamma, gammaGm[startAddr], copyNum);
    inQueueGamma.EnQue(gamma);
    LocalTensor<T2> gammaUb = inQueueGamma.DeQue<T2>();

    LocalTensor<T2> beta = inQueueBeta.AllocTensor<T2>();
    this->template CopyInData<T2, isAlign>(beta, betaGm[startAddr], copyNum);
    inQueueBeta.EnQue(beta);
    LocalTensor<T2> betaUb = inQueueBeta.DeQue<T2>();
    // cast gamma beta
    Cast(x1Buf32.Get<float>(), gammaUb, RoundMode::CAST_NONE, copyNumAlign);
    PipeBarrier<PIPE_V>();
    Cast(x2Buf32.Get<float>(), betaUb, RoundMode::CAST_NONE, copyNumAlign);
    PipeBarrier<PIPE_V>();
    inQueueGamma.FreeTensor(gammaUb);
    inQueueBeta.FreeTensor(betaUb);
}

template <typename T1, typename T2>
template <bool isAlign>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::CopyInGammaBeta32(const int64_t startAddr, const int64_t copyNum)
{
    int64_t copyNumAlign = this->CeilDiv(copyNum, elementsPerBlockT1) * elementsPerBlockT1;
    this->template CopyInData<float, isAlign>(x1Buf32.Get<float>(), gammaGmFloat[startAddr], copyNum);
    event_t eventMte2ToS1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventMte2ToS1);
    WaitFlag<HardEvent::MTE2_S>(eventMte2ToS1);

    this->template CopyInData<float, isAlign>(x2Buf32.Get<float>(), betaGmFloat[startAddr], copyNum);
    event_t eventMte2ToS2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventMte2ToS2);
    WaitFlag<HardEvent::MTE2_S>(eventMte2ToS2);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::CalculateGroupNormSilu(
    const float& mean, const float& rstd, const int64_t& computeNum)
{
    LocalTensor<float> xUb32 = xBuf32.Get<float>();
    LocalTensor<T1> outY = outQueueY.AllocTensor<T1>();

    Adds(xUb32, xUb32, -mean, computeNum);
    PipeBarrier<PIPE_V>();
    Muls(xUb32, xUb32, rstd, computeNum);
    PipeBarrier<PIPE_V>();
    Mul(xUb32, xUb32, x1Buf32.Get<float>(), computeNum);
    PipeBarrier<PIPE_V>();
    Add(xUb32, xUb32, x2Buf32.Get<float>(), computeNum);
    PipeBarrier<PIPE_V>();
    if (tiling->activateSilu) {
        Muls(x1Buf32.Get<float>(), xUb32, negativeOne, computeNum);
        PipeBarrier<PIPE_V>();
        Exp(x1Buf32.Get<float>(), x1Buf32.Get<float>(), computeNum);
        PipeBarrier<PIPE_V>();
        Adds(x1Buf32.Get<float>(), x1Buf32.Get<float>(), scalarOne, computeNum);
        PipeBarrier<PIPE_V>();
        Div(xUb32, xUb32, x1Buf32.Get<float>(), computeNum);
        PipeBarrier<PIPE_V>();
    }
    Cast(outY, xUb32, this->GetRoundMode(), computeNum);
    PipeBarrier<PIPE_V>();
    outQueueY.EnQue(outY);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::CopyOutY(const int64_t& startAddr, const int64_t& copyNum)
{
    LocalTensor<T1> outY = outQueueY.DeQue<T1>();
    DataCopy(yGm[startAddr], outY, copyNum);
    outQueueY.FreeTensor(outY);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::CopyOutYWithPad(
    const float& mean, const float& rstd, const int64_t& startAddr, const int64_t& copyNum)
{
    LocalTensor<T1> outY = outQueueY.DeQue<T1>();
    LocalTensor<T1> tmpOut = tmpTensor.Get<T1>();
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    uint16_t dataCount = static_cast<uint16_t>(copyNum);
    uint16_t blockCount = 1;
    uint16_t blockLen = dataCount * sizeof(T1);
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    DataCopyParams dataCopyParams{blockCount, blockLen, srcStride, dstStride};
    DataCopyPad(yGm[startAddr], outY, dataCopyParams);
    outQueueY.FreeTensor(outY);
#else
    if (tiling->hwNum % elementsPerBlockT1 == 0) {
        DataCopy(yGm[startAddr], outY, copyNum);
        outQueueY.FreeTensor(outY);
    } else {
        if (tiling->hwNum < elementsPerBlockT1) {
            DataCopy(yGm[startAddr], outY, elementsPerBlockT1);
            outQueueY.FreeTensor(outY);
        } else {
            DataCopy(yGm[startAddr], outY, copyNum);
            outQueueY.FreeTensor(outY);
            CopyInX(startAddr + copyNum - elementsPerBlockT1, elementsPerBlockT1);
            CalculateGroupNormSilu(mean, rstd, elementsPerBlockT1);
            CopyOutY(startAddr + copyNum - elementsPerBlockT1, elementsPerBlockT1);
        }
    }
#endif
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::CastMeanAndRstd(
    LocalTensor<float>& meanUb, LocalTensor<float>& rstdUb, const int64_t& copyNum)
{
    LocalTensor<T1> meanOut = outQueueMean.AllocTensor<T1>();
    Cast(meanOut, meanUb, this->GetRoundMode(), copyNum);
    outQueueMean.EnQue(meanOut);

    LocalTensor<T1> rstdOut = outQueueRstd.AllocTensor<T1>();
    Cast(rstdOut, rstdUb, this->GetRoundMode(), copyNum);
    outQueueRstd.EnQue(rstdOut);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::CopyOutMeanAndRstd(const int64_t& startAddr, const int64_t& copyNum)
{
    LocalTensor<T1> meanOut = outQueueMean.DeQue<T1>();
    LocalTensor<T1> rstdOut = outQueueRstd.DeQue<T1>();
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    // when support DataCopyPad, use DataCopyPad
    uint16_t dataCount = static_cast<uint16_t>(copyNum);
    uint16_t blockCount = 1;
    uint16_t blockLen = dataCount * sizeof(T1);
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    DataCopyParams dataCopyParams{blockCount, blockLen, srcStride, dstStride};
    DataCopyPad(meanGm[blockIdx * tiling->numPerCore + startAddr], meanOut, dataCopyParams);
    DataCopyPad(rstdGm[blockIdx * tiling->numPerCore + startAddr], rstdOut, dataCopyParams);
#else
    // when not support DataCopyPad, use DataCopy, copyNum must be align process
    if (copyNum % elementsPerBlockT1 == 0) {
        DataCopy(meanGm[blockIdx * tiling->numPerCore + startAddr], meanOut, copyNum);
        DataCopy(rstdGm[blockIdx * tiling->numPerCore + startAddr], rstdOut, copyNum);
    } else if (copyNum > elementsPerBlockT1) {
        // DataCopy will round down by 32B when copyNum is not align
        DataCopy(meanGm[blockIdx * tiling->numPerCore + startAddr], meanOut, copyNum);
        DataCopy(rstdGm[blockIdx * tiling->numPerCore + startAddr], rstdOut, copyNum);
        LocalTensor<T1> meanTmp = gammaBuf32.Get<T1>();
        LocalTensor<T1> rstdTmp = betaBuf32.Get<T1>();
        // copy out last loop overlap
        for (int64_t idx = 0; idx < elementsPerBlockT1; idx++) {
            meanTmp.SetValue(idx, meanOut.GetValue(copyNum - elementsPerBlockT1 + idx));
            rstdTmp.SetValue(idx, rstdOut.GetValue(copyNum - elementsPerBlockT1 + idx));
        }
        DataCopy(
            meanGm[blockIdx * tiling->numPerCore + startAddr + copyNum - elementsPerBlockT1], meanTmp,
            elementsPerBlockT1);
        DataCopy(
            rstdGm[blockIdx * tiling->numPerCore + startAddr + copyNum - elementsPerBlockT1], rstdTmp,
            elementsPerBlockT1);
    } else {
        // copyNum is less than elementsPerBlockT1
        ComputeForMeanAlign(meanOut, rstdOut, startAddr, copyNum);
        DataCopy(meanGm[blockIdx * tiling->numPerCore + startAddr], meanOut, elementsPerBlockT1);
        DataCopy(rstdGm[blockIdx * tiling->numPerCore + startAddr], rstdOut, elementsPerBlockT1);
    }
#endif
    outQueueMean.FreeTensor(meanOut);
    outQueueRstd.FreeTensor(rstdOut);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluHW1B16<T1, T2>::CopyOutMeanAndRstdAlign(const int64_t& startAddr)
{
    LocalTensor<T1> meanOut = outQueueMean.DeQue<T1>();
    LocalTensor<T1> rstdOut = outQueueRstd.DeQue<T1>();
    DataCopy(meanGm[blockIdx * tiling->numPerCore + startAddr], meanOut, innerNumPerCore);
    DataCopy(rstdGm[blockIdx * tiling->numPerCore + startAddr], rstdOut, innerNumPerCore);
    outQueueMean.FreeTensor(meanOut);
    outQueueRstd.FreeTensor(rstdOut);
}

} // namespace GroupNormSilu

#endif // GROUP_NORM_SILU_HW1_B16_H_
