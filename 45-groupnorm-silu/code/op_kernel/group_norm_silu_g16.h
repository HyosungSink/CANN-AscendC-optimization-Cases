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
 * \file group_norm_silu_g16.h
 * \brief
 */
#ifndef GROUP_NORM_SILU_G16_H_
#define GROUP_NORM_SILU_G16_H_

#include "group_norm_silu_base.h"

namespace GroupNormSilu {
using namespace AscendC;

template <typename T1, typename T2>
class GroupNormSiluG16 : public GroupNormSiluBase<T1>
{
public:
    __aicore__ inline GroupNormSiluG16(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
        const GroupNormSiluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    // function
    template <bool isAlign = true>
    __aicore__ inline void CopyInX(const int64_t startAddr, const int64_t copyNum);
    __aicore__ inline void ComputeSum(const int64_t& processNum);
    __aicore__ inline void AccumulateXAndX2(const int64_t& groupIdx);
    __aicore__ inline void ReduceSumXAndX2(const int64_t& reduseNum);
    __aicore__ inline void CopyInGammaBeta16(const int64_t& startAddr);
    __aicore__ inline void CopyInGammaBeta32(const int64_t& startAddr);
    __aicore__ inline void CalculateGroupNormSilu(const float& scale, const float& bias, const int64_t& computeNum);
    __aicore__ inline void ComputeGroupNormSilu(
        const int64_t& groupIdx, LocalTensor<float>& meanUb, LocalTensor<float>& rstdUb);
    __aicore__ inline void ProcessPerCore(const int64_t& groups);
    __aicore__ inline void ProcessGroupNormSilu(const int64_t& offset, const int64_t& groups);
    __aicore__ inline void CopyOutY(const int64_t& startAddr, const int64_t& copyNum);
    __aicore__ inline void CopyOutYWithPad(
        const float& scale, const float& bias, const int64_t& startAddr, const int64_t& copyNum);
    __aicore__ inline void CastMeanAndRstd(
        LocalTensor<float>& meanUb, LocalTensor<float>& rstdUb, const int64_t& copyNum);
    __aicore__ inline void CopyOutMeanAndRstd(const int64_t& startAddr, const int64_t& copyNum);
    __aicore__ inline void CopyOutMeanAndRstdAlign(const int64_t& startAddr);
    // constant
    constexpr static int32_t bufferNum = 2;
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
    TQue<QuePosition::VECIN, bufferNum> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueGamma;
    TQue<QuePosition::VECIN, 1> inQueueBeta;
    TQue<QuePosition::VECOUT, bufferNum> outQueueY;
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
__aicore__ inline void GroupNormSiluG16<T1, T2>::Init(
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

    pipe.InitBuffer(inQueueX, bufferNum, tiling->processSize * sizeof(T1));
    pipe.InitBuffer(inQueueGamma, 1, elementsPerBlockT2 * sizeof(T2));
    pipe.InitBuffer(inQueueBeta, 1, elementsPerBlockT2 * sizeof(T2));
    pipe.InitBuffer(outQueueY, bufferNum, tiling->processSize * sizeof(T1));
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
    gmOffset = blockIdx * tiling->numPerCore * tiling->elemNum;
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluG16<T1, T2>::Process()
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
__aicore__ inline void GroupNormSiluG16<T1, T2>::ProcessPerCore(const int64_t& groups)
{
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

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluG16<T1, T2>::ProcessGroupNormSilu(const int64_t& offset, const int64_t& groups)
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
__aicore__ inline void GroupNormSiluG16<T1, T2>::AccumulateXAndX2(const int64_t& groupIdx)
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
__aicore__ inline void GroupNormSiluG16<T1, T2>::CopyInX(const int64_t startAddr, const int64_t copyNum)
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
__aicore__ inline void GroupNormSiluG16<T1, T2>::ComputeSum(const int64_t& processNum)
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
__aicore__ inline void GroupNormSiluG16<T1, T2>::ReduceSumXAndX2(const int64_t& reduseNum)
{
    // reduseSum(x)
    ReduceSum<float>(tmpTensor.Get<float>(), x1Buf32.Get<float>(), xBuf32.Get<float>(), reduseNum);
    // reduseSum(x^2)
    ReduceSum<float>(x2Buf32.Get<float>(), x2Buf32.Get<float>(), xBuf32.Get<float>(), reduseNum);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluG16<T1, T2>::ComputeGroupNormSilu(
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
    // calculating y  y = (x- mean) * rstd * gamma + beta
    for (int64_t dIdx = 0; dIdx < tiling->shapeD; dIdx++) {
        PipeBarrier<PIPE_V>();
        // copy gamma && beta
        int64_t subStartAddr = ((blockIdx * tiling->numPerCore + groupIdx) * tiling->shapeD + dIdx) % tiling->shapeC;
        if (sizeof(T2) != sizeof(float)) {
            CopyInGammaBeta16(subStartAddr);
        } else {
            CopyInGammaBeta32(subStartAddr);
        }
        float gamma = gammaBuf32.Get<float>().GetValue(0);
        float beta = betaBuf32.Get<float>().GetValue(0);
        float scale = rstd * gamma;
        float bias = -scale * mean + beta;
        int64_t hwStartAddr = gmOffset + groupIdx * tiling->elemNum + dIdx * tiling->hwNum;
        for (int64_t innerLoop = 0; innerLoop < tiling->innerLoopNum - 1; innerLoop++) {
            int64_t hwInnerAddr = hwStartAddr + innerLoop * tiling->processSize;
            CopyInX(hwInnerAddr, tiling->processSize);
            CalculateGroupNormSilu(scale, bias, tiling->processSize);
            CopyOutY(hwInnerAddr, tiling->processSize);
        }
        int64_t hwTailAddrG16 = hwStartAddr + (tiling->innerLoopNum - 1) * tiling->processSize;
        CopyInX<false>(hwTailAddrG16, tiling->innerLoopTail);
        CalculateGroupNormSilu(scale, bias, tiling->innerLoopTail);
        CopyOutYWithPad(scale, bias, hwTailAddrG16, tiling->innerLoopTail);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluG16<T1, T2>::CopyInGammaBeta16(const int64_t& startAddr)
{
    LocalTensor<T2> gamma = inQueueGamma.AllocTensor<T2>();
    this->template CopyInData<T2, false>(gamma, gammaGm[startAddr], 1);
    inQueueGamma.EnQue(gamma);
    LocalTensor<T2> gammaUb = inQueueGamma.DeQue<T2>();

    LocalTensor<T2> beta = inQueueBeta.AllocTensor<T2>();
    this->template CopyInData<T2, false>(beta, betaGm[startAddr], 1);
    inQueueBeta.EnQue(beta);
    LocalTensor<T2> betaUb = inQueueBeta.DeQue<T2>();
    // cast gamma beta
    Cast(gammaBuf32.Get<float>(), gammaUb, RoundMode::CAST_NONE, 1);
    event_t eventV2S1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventV2S1);
    WaitFlag<HardEvent::V_S>(eventV2S1);
    Cast(betaBuf32.Get<float>(), betaUb, RoundMode::CAST_NONE, 1);
    event_t eventV2S2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventV2S2);
    WaitFlag<HardEvent::V_S>(eventV2S2);
    inQueueGamma.FreeTensor(gammaUb);
    inQueueBeta.FreeTensor(betaUb);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluG16<T1, T2>::CopyInGammaBeta32(const int64_t& startAddr)
{
    this->template CopyInData<float, false>(gammaBuf32.Get<float>(), gammaGmFloat[startAddr], 1);
    event_t eventMte2ToS1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventMte2ToS1);
    WaitFlag<HardEvent::MTE2_S>(eventMte2ToS1);

    this->template CopyInData<float, false>(betaBuf32.Get<float>(), betaGmFloat[startAddr], 1);
    event_t eventMte2ToS2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventMte2ToS2);
    WaitFlag<HardEvent::MTE2_S>(eventMte2ToS2);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluG16<T1, T2>::CalculateGroupNormSilu(
    const float& scale, const float& bias, const int64_t& computeNum)
{
    LocalTensor<float> xUb32 = xBuf32.Get<float>();
    LocalTensor<T1> outY = outQueueY.AllocTensor<T1>();
    Muls(xUb32, xUb32, scale, computeNum);
    PipeBarrier<PIPE_V>();
    Adds(xUb32, xUb32, bias, computeNum);
    PipeBarrier<PIPE_V>();
    if (tiling->activateSilu == 1) {
        LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
        Muls(x1Ub32, xUb32, negativeOne, computeNum);
        PipeBarrier<PIPE_V>();
        Exp(x1Ub32, x1Ub32, computeNum);
        PipeBarrier<PIPE_V>();
        Adds(x1Ub32, x1Ub32, scalarOne, computeNum);
        PipeBarrier<PIPE_V>();
        Div(xUb32, xUb32, x1Ub32, computeNum);
        PipeBarrier<PIPE_V>();
    }
    Cast(outY, xUb32, this->GetRoundMode(), computeNum);
    PipeBarrier<PIPE_V>();
    outQueueY.EnQue(outY);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluG16<T1, T2>::CopyOutY(const int64_t& startAddr, const int64_t& copyNum)
{
    LocalTensor<T1> outY = outQueueY.DeQue<T1>();
    DataCopy(yGm[startAddr], outY, copyNum);
    outQueueY.FreeTensor(outY);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluG16<T1, T2>::CopyOutYWithPad(
    const float& scale, const float& bias, const int64_t& startAddr, const int64_t& copyNum)
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
            if (copyNum >= elementsPerBlockT1) {
                DataCopy(yGm[startAddr], outY, copyNum);
            }
            outQueueY.FreeTensor(outY);
            CopyInX(startAddr + copyNum - elementsPerBlockT1, elementsPerBlockT1);
            CalculateGroupNormSilu(scale, bias, elementsPerBlockT1);
            CopyOutY(startAddr + copyNum - elementsPerBlockT1, elementsPerBlockT1);
        }
    }
#endif
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluG16<T1, T2>::CastMeanAndRstd(
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
__aicore__ inline void GroupNormSiluG16<T1, T2>::CopyOutMeanAndRstd(const int64_t& startAddr, const int64_t& copyNum)
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
#endif
    outQueueMean.FreeTensor(meanOut);
    outQueueRstd.FreeTensor(rstdOut);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluG16<T1, T2>::CopyOutMeanAndRstdAlign(const int64_t& startAddr)
{
    LocalTensor<T1> meanOut = outQueueMean.DeQue<T1>();
    LocalTensor<T1> rstdOut = outQueueRstd.DeQue<T1>();
    DataCopy(meanGm[blockIdx * tiling->numPerCore + startAddr], meanOut, innerNumPerCore);
    DataCopy(rstdGm[blockIdx * tiling->numPerCore + startAddr], rstdOut, innerNumPerCore);
    outQueueMean.FreeTensor(meanOut);
    outQueueRstd.FreeTensor(rstdOut);
}

} // namespace GroupNormSilu

#endif // GROUP_NORM_SILU_G16_H_
