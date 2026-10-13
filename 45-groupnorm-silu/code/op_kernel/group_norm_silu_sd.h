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
 * \file group_norm_silu_sd.h
 * \brief
 */
#ifndef GROUP_NORM_SILU_SD_H_
#define GROUP_NORM_SILU_SD_H_

#include "group_norm_silu_base.h"

namespace GroupNormSilu {
using namespace AscendC;

template <typename T1, typename T2>
class GroupNormSiluSD : public GroupNormSiluBase<T1>
{
public:
    __aicore__ inline GroupNormSiluSD(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR silu, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
        const GroupNormSiluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CastGammaAndBeta(const int64_t& groups, const int64_t& startAddr);
    __aicore__ inline void ComputeSilu(const int64_t& groups, const int64_t& startAddr);
    __aicore__ inline void ReduceXandX2(const int64_t& startAddr);
    __aicore__ inline void CopyInX(const int64_t& startAddr, const int64_t& copyNum);
    __aicore__ inline void SumXandX2(const int64_t& num);
    __aicore__ inline void NormaLizeX(const int64_t& xIdx, const float& scale, const float& bias, const int64_t& num);
    __aicore__ inline void CalcSilu(const int64_t& calcNum);
    __aicore__ inline void ComputeYSmall(
        const int64_t& cIdx, const float& rstd, const float& mean, const int64_t& gIdxs);
    __aicore__ inline void ComputeYBig(const int64_t& cIdx, const float& rstd, const float& mean, const int64_t& num);
    __aicore__ inline void CopyOutY(const int64_t& startAddr, const int64_t& copyNum);
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

    TBuf<QuePosition::VECCALC> xBuf32;
    TBuf<QuePosition::VECCALC> x1Buf32;
    TBuf<QuePosition::VECCALC> x2Buf32;
    TBuf<QuePosition::VECCALC> gammaBuf32;
    TBuf<QuePosition::VECCALC> betaBuf32;
    TBuf<QuePosition::VECCALC> tmpTensor;

    GlobalTensor<T1> xGm;
    GlobalTensor<T2> gammaGm, betaGm;
    GlobalTensor<T1> siluGm;

    int32_t blockIdx = 0;
    int64_t blockOffset = 0;
    float numRec = 0;
    const GroupNormSiluTilingData* tiling;
};

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSD<T1, T2>::Init(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR silu, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
    const GroupNormSiluTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    tiling = tilingData;
    xGm.SetGlobalBuffer((__gm__ T1*)x);
    gammaGm.SetGlobalBuffer((__gm__ T2*)gamma);
    betaGm.SetGlobalBuffer((__gm__ T2*)beta);
    siluGm.SetGlobalBuffer((__gm__ T1*)silu);

    pipe.InitBuffer(xBuf32, tiling->processSize * sizeof(float));
    pipe.InitBuffer(x1Buf32, tiling->processSize * sizeof(float));
    pipe.InitBuffer(x2Buf32, tiling->processSize * sizeof(float));
    pipe.InitBuffer(inQueueX, bufferNum, tiling->processSize * sizeof(T1));
    pipe.InitBuffer(outQueueSilu, bufferNum, tiling->processSize * sizeof(T1));
    pipe.InitBuffer(inQueueGamma, 1, tiling->numGroups * sizeof(T2));
    pipe.InitBuffer(inQueueBeta, 1, tiling->numGroups * sizeof(T2));
    pipe.InitBuffer(gammaBuf32, tiling->numGroups * sizeof(float));
    pipe.InitBuffer(betaBuf32, tiling->numGroups * sizeof(float));
    pipe.InitBuffer(tmpTensor, elementsPerBlock32 * sizeof(float));
    numRec = float(1.0) / float(tiling->elemNum);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSD<T1, T2>::Process()
{
    if (blockIdx >= tiling->realCoreNum || blockIdx < 0) {
        return;
    } else if (blockIdx < tiling->numLastCore) {
        int64_t gammaStartAddr = blockIdx * (tiling->numPerCore + 1) * tiling->shapeD;
        CastGammaAndBeta(tiling->numPerCore + 1, gammaStartAddr);
        int64_t xStartAddr = gammaStartAddr * tiling->hwNum;
        ComputeSilu(tiling->numPerCore + 1, xStartAddr);
    } else {
        int64_t gammaStartAddr =
            (tiling->numLastCore * (tiling->numPerCore + 1) + (blockIdx - tiling->numLastCore) * tiling->numPerCore) *
            tiling->shapeD;
        CastGammaAndBeta(tiling->numPerCore, gammaStartAddr);
        int64_t xStartAddr = gammaStartAddr * tiling->hwNum;
        ComputeSilu(tiling->numPerCore, xStartAddr);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSD<T1, T2>::CastGammaAndBeta(const int64_t& groups, const int64_t& startAddr)
{
    LocalTensor<T2> gamma = inQueueGamma.AllocTensor<T2>();
    DataCopy(gamma, gammaGm[startAddr], tiling->numGroups);
    inQueueGamma.EnQue(gamma);

    LocalTensor<T2> beta = inQueueBeta.AllocTensor<T2>();
    DataCopy(beta, betaGm[startAddr], tiling->numGroups);
    inQueueBeta.EnQue(beta);

    LocalTensor<T2> gammaUb = inQueueGamma.DeQue<T2>();
    LocalTensor<T2> betaUb = inQueueBeta.DeQue<T2>();
    Cast(gammaBuf32.Get<float>(), gammaUb, RoundMode::CAST_NONE, groups * tiling->shapeD);
    Cast(betaBuf32.Get<float>(), betaUb, RoundMode::CAST_NONE, groups * tiling->shapeD);
    inQueueGamma.FreeTensor(gammaUb);
    inQueueBeta.FreeTensor(betaUb);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSD<T1, T2>::ComputeSilu(const int64_t& groups, const int64_t& startAddr)
{
    for (int64_t group = 0; group < groups; group++) {
        int64_t xStartAddr = startAddr + group * tiling->elemNum;
        ReduceXandX2(xStartAddr);
        // calc var and rstd;
        float mean = tmpTensor.Get<float>().GetValue(0);
        float rstd = x2Buf32.Get<float>().GetValue(0);
        mean = mean * numRec;
        rstd = float(1.0) / sqrt(rstd * numRec - mean * mean + tiling->epsilon);
        // normalize x and calc silu
        int64_t gammaIdx = group * tiling->shapeD;
        if (tiling->processSize >= tiling->hwNum) {
            for (int64_t dIdx = 0; dIdx < tiling->loopNum; dIdx++) {
                int64_t cIdx = gammaIdx + dIdx * tiling->innerLoopNum;
                CopyInX(xStartAddr + dIdx * tiling->processSize, tiling->processSize);
                ComputeYSmall(cIdx, rstd, mean, tiling->innerLoopNum);
                CopyOutY(xStartAddr + dIdx * tiling->processSize, tiling->processSize);
            }
            // process tail data
            if (tiling->loopTail > 0) {
                CopyInX(xStartAddr + tiling->loopNum * tiling->processSize, tiling->loopTail);
                ComputeYSmall(
                    gammaIdx + tiling->loopNum * tiling->innerLoopNum, rstd, mean,
                    tiling->shapeD - tiling->loopNum * tiling->innerLoopNum);
                CopyOutY(xStartAddr + tiling->loopNum * tiling->processSize, tiling->loopTail);
            }
        } else {
            for (int64_t cIdx = 0; cIdx < tiling->shapeD; cIdx++) {
                int64_t betaIdx = gammaIdx + cIdx;
                int64_t xStartAddrInner = xStartAddr + cIdx * tiling->hwNum;
                for (int64_t dIdx = 0; dIdx < tiling->innerLoopNum; dIdx++) {
                    CopyInX(xStartAddrInner + dIdx * tiling->processSize, tiling->processSize);
                    ComputeYBig(betaIdx, rstd, mean, tiling->processSize);
                    CopyOutY(xStartAddrInner + dIdx * tiling->processSize, tiling->processSize);
                }
                if (tiling->innerLoopTail > 0) {
                    CopyInX(xStartAddrInner + tiling->innerLoopNum * tiling->processSize, tiling->innerLoopTail);
                    ComputeYBig(betaIdx, rstd, mean, tiling->innerLoopTail);
                    CopyOutY(xStartAddrInner + tiling->innerLoopNum * tiling->processSize, tiling->innerLoopTail);
                }
            }
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSD<T1, T2>::ReduceXandX2(const int64_t& startAddr)
{
    LocalTensor<float> xUb32 = xBuf32.Get<float>();
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    LocalTensor<float> x2Ub32 = x2Buf32.Get<float>();
    LocalTensor<float> tmpMean = tmpTensor.Get<float>();

    if (tiling->loopNum > 0) {
        // process the first loop
        CopyInX(startAddr, tiling->processSize);
        LocalTensor<T1> xUb = inQueueX.DeQue<T1>();
        Cast(x1Ub32, xUb, RoundMode::CAST_NONE, tiling->processSize);
        inQueueX.FreeTensor(xUb);
        Mul(x2Ub32, x1Ub32, x1Ub32, tiling->processSize);
        // process the middle loops
        for (int64_t innerIdx = 1; innerIdx < tiling->loopNum; innerIdx++) {
            CopyInX(startAddr + innerIdx * tiling->processSize, tiling->processSize);
            SumXandX2(tiling->processSize);
        }
        // process tail data
        if (tiling->loopTail > 0) {
            CopyInX(startAddr + tiling->loopNum * tiling->processSize, tiling->loopTail);
            SumXandX2(tiling->loopTail);
        }
        // accumulate x and x^2
        ReduceSum<float>(tmpMean, x1Ub32, xUb32, tiling->processSize);
        ReduceSum<float>(x2Ub32, x2Ub32, xUb32, tiling->processSize);
    } else {
        // process the first loop
        CopyInX(startAddr, tiling->loopTail);
        LocalTensor<T1> xUb = inQueueX.DeQue<T1>();
        Cast(x1Ub32, xUb, RoundMode::CAST_NONE, tiling->loopTail);
        inQueueX.FreeTensor(xUb);
        Mul(x2Ub32, x1Ub32, x1Ub32, tiling->loopTail);
        // accumulate x and x^2
        ReduceSum<float>(tmpMean, x1Ub32, xUb32, tiling->loopTail);
        ReduceSum<float>(x2Ub32, x2Ub32, xUb32, tiling->loopTail);
    }
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSD<T1, T2>::CopyInX(const int64_t& startAddr, const int64_t& copyNum)
{
    LocalTensor<T1> xUb = inQueueX.AllocTensor<T1>();
    DataCopy(xUb, xGm[startAddr], copyNum);
    inQueueX.EnQue(xUb);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSD<T1, T2>::SumXandX2(const int64_t& num)
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
__aicore__ inline void GroupNormSiluSD<T1, T2>::NormaLizeX(
    const int64_t& xIdx, const float& scale, const float& bias, const int64_t& num)
{
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    // normalize x
    Muls(x1Ub32[xIdx], x1Ub32[xIdx], scale, num);
    PipeBarrier<PIPE_V>();
    Adds(x1Ub32[xIdx], x1Ub32[xIdx], bias, num);
    PipeBarrier<PIPE_V>();
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSD<T1, T2>::CalcSilu(const int64_t& calcNum)
{
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    LocalTensor<float> x2Ub32 = x2Buf32.Get<float>();
    // calc silu
    Muls(x2Ub32, x1Ub32, negativeOne, calcNum);
    PipeBarrier<PIPE_V>();
    Exp(x2Ub32, x2Ub32, calcNum);
    PipeBarrier<PIPE_V>();
    Adds(x2Ub32, x2Ub32, scalarOne, calcNum);
    PipeBarrier<PIPE_V>();
    Div(x1Ub32, x1Ub32, x2Ub32, calcNum);
    PipeBarrier<PIPE_V>();
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSD<T1, T2>::ComputeYSmall(
    const int64_t& cIdx, const float& rstd, const float& mean, const int64_t& gIdxs)
{
    LocalTensor<T1> xUb = inQueueX.DeQue<T1>();
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    LocalTensor<T1> outSilu = outQueueSilu.AllocTensor<T1>();
    LocalTensor<float> gammaUb32 = gammaBuf32.Get<float>();
    LocalTensor<float> betaUb32 = betaBuf32.Get<float>();

    Cast(x1Ub32, xUb, RoundMode::CAST_NONE, gIdxs * tiling->hwNum);
    PipeBarrier<PIPE_V>();
    inQueueX.FreeTensor(xUb);
    for (int64_t gIdx = 0; gIdx < gIdxs; gIdx++) {
        float gamma = gammaUb32.GetValue(cIdx + gIdx);
        float beta = betaUb32.GetValue(cIdx + gIdx);
        float scale = rstd * gamma;
        float bias = -scale * mean + beta;
        NormaLizeX(gIdx * tiling->hwNum, scale, bias, tiling->hwNum);
    }
    if (tiling->activateSilu) {
        CalcSilu(gIdxs * tiling->hwNum);
    }
    Cast(outSilu, x1Ub32, this->GetRoundMode(), gIdxs * tiling->hwNum);
    PipeBarrier<PIPE_V>();
    outQueueSilu.EnQue(outSilu);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSD<T1, T2>::ComputeYBig(
    const int64_t& cIdx, const float& rstd, const float& mean, const int64_t& num)
{
    LocalTensor<T1> xUb = inQueueX.DeQue<T1>();
    LocalTensor<float> x1Ub32 = x1Buf32.Get<float>();
    LocalTensor<T1> outSilu = outQueueSilu.AllocTensor<T1>();
    LocalTensor<float> gammaUb32 = gammaBuf32.Get<float>();
    LocalTensor<float> betaUb32 = betaBuf32.Get<float>();

    Cast(x1Ub32, xUb, RoundMode::CAST_NONE, num);
    PipeBarrier<PIPE_V>();
    inQueueX.FreeTensor(xUb);
    float gamma = gammaUb32.GetValue(cIdx);
    float beta = betaUb32.GetValue(cIdx);
    float scale = rstd * gamma;
    float bias = -scale * mean + beta;
    NormaLizeX(0, scale, bias, num);
    if (tiling->activateSilu) {
        CalcSilu(num);
    }
    Cast(outSilu, x1Ub32, this->GetRoundMode(), num);
    PipeBarrier<PIPE_V>();
    outQueueSilu.EnQue(outSilu);
}

template <typename T1, typename T2>
__aicore__ inline void GroupNormSiluSD<T1, T2>::CopyOutY(const int64_t& startAddr, const int64_t& copyNum)
{
    LocalTensor<T1> outSilu = outQueueSilu.DeQue<T1>();
    DataCopy(siluGm[startAddr], outSilu, copyNum);
    outQueueSilu.FreeTensor(outSilu);
}

} // namespace GroupNormSilu

#endif // GROUP_NORM_SILU_SD_H_