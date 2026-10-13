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
 * \file rms_norm_grad_whole_reduce_n.h
 * \brief
 */
#ifndef RMS_NORM_GRAD_WHOLE_REDUCE_N_H
#define RMS_NORM_GRAD_WHOLE_REDUCE_N_H
#include "rms_norm_grad_common.h"

template <typename T>
class RmsNormGradWholeReduceN {
public:
    __aicore__ inline RmsNormGradWholeReduceN(){};
    __aicore__ inline void Init(
        GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma,
        const RmsNormGradTilingData* tiling);
    __aicore__ inline void Process();

protected:
    TPipe pipe;
    GlobalTensor<T> dyGm, gammaGm, dxGm, xGm;
    GlobalTensor<float> dgammaGm, rstdGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueDY, inQueX, inQueRstd, inQueGamma;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueDX, outQueDgamma;
    TBuf<TPosition::VECCALC> ndBufFp32Buf1, ndBufFp32Buf2, ndBufFp32Buf3;

    // tilingdata
    uint32_t rowVal;
    uint32_t colVal;
    float avgFactor;
    uint32_t dataType;
    uint32_t blockFactor;
    uint32_t ubFactor;
    uint32_t coreCalcNum;
    uint32_t coreCalcTail;
    uint32_t numBlocks;
    uint32_t ubCalcNum;
    uint32_t ubCalcTail;
    uint32_t ubCalcLoop;
    uint32_t ubCalcTailNum;
    uint32_t ubCalcTailTail;
    uint32_t ubCalcTailLoop;
    uint32_t fixedOutput{0};

    uint32_t blockIdx;
    uint32_t alignLen;
    uint32_t coreOffset;
    uint32_t ubFactorAlign;
    uint32_t rstdLen;
    uint32_t bufferLenSize;
    uint32_t bufferNum;

    uint64_t fp32Mask = ONE_REPEAT_BYTE_SIZE / sizeof(float);

    __aicore__ inline void InitData(const RmsNormGradTilingData* tiling);
    __aicore__ inline void CopyGammaIn();
    __aicore__ inline void CopyIn(uint32_t loopIdx, uint32_t calcLen);
    __aicore__ inline void CopyOut(uint32_t loopIdx, uint32_t calcLen);
    __aicore__ inline void CopyDgammaOut();
    __aicore__ inline void CopyDgammaOutInOrder();
    __aicore__ inline void Compute(uint32_t calcLen, LocalTensor<T>& gammaUb, LocalTensor<float>& dgamma);
    __aicore__ inline void ComputeMain(
        LocalTensor<T>& x, LocalTensor<T>& dx, LocalTensor<T>& dy, LocalTensor<float>& rstd, LocalTensor<T>& gamma,
        LocalTensor<float>& dgamma, uint32_t calcLen);
};

template <typename T>
__aicore__ inline void RmsNormGradWholeReduceN<T>::Init(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma, const RmsNormGradTilingData* tiling)
{
    InitData(tiling);

    blockIdx = GetBlockIdx();
    if (blockIdx < numBlocks - 1) {
        coreOffset = blockFactor;
    } else {
        coreOffset = coreCalcTail > 0 ? coreCalcTail : blockFactor;
    }
    dyGm.SetGlobalBuffer((__gm__ T*)dy + blockIdx * blockFactor * colVal, coreOffset * colVal);
    xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx * blockFactor * colVal, coreOffset * colVal);
    rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockIdx * blockFactor, coreOffset);
    gammaGm.SetGlobalBuffer((__gm__ T*)gamma, colVal);
    dxGm.SetGlobalBuffer((__gm__ T*)dx + blockIdx * blockFactor * colVal, coreOffset * colVal);
    dgammaGm.SetGlobalBuffer((__gm__ float*)dgamma, colVal);

    ubFactorAlign = ubFactor * colVal;
    rstdLen = (ubFactor + alignLen - 1) / alignLen * alignLen;
    bufferLenSize = ubFactorAlign * sizeof(T);
    pipe.InitBuffer(inQueDY, bufferNum, bufferLenSize);
    pipe.InitBuffer(inQueX, bufferNum, bufferLenSize);
    pipe.InitBuffer(inQueRstd, bufferNum, rstdLen * sizeof(float));
    pipe.InitBuffer(inQueGamma, BUFFER_NUM, colVal * sizeof(T));
    pipe.InitBuffer(outQueDX, bufferNum, bufferLenSize);
    pipe.InitBuffer(outQueDgamma, BUFFER_NUM, colVal * sizeof(float));

    uint32_t ubFactorLen = (ubFactor + alignLen - 1) / alignLen * alignLen;
    uint32_t ubFactorAlignLen = ubFactorAlign * sizeof(float);
    pipe.InitBuffer(ndBufFp32Buf1, ubFactorAlignLen);
    if constexpr (is_same<T, half>::value) {
        pipe.InitBuffer(ndBufFp32Buf2, ubFactorAlignLen);
        pipe.InitBuffer(ndBufFp32Buf3, ubFactorAlignLen);
    }

    if (blockIdx == 0) {
        LocalTensor<float> initGmZeros = ndBufFp32Buf1.Get<float>();
        Duplicate(initGmZeros, 0.0f, colVal);
        PipeBarrier<PIPE_V>();
        DataCopy(dgammaGm, initGmZeros, colVal);
        PipeBarrier<PIPE_V>();
        fixed_output_sync[0] = 0;
    }
    SyncAll();
}

template <typename T>
__aicore__ inline void RmsNormGradWholeReduceN<T>::InitData(const RmsNormGradTilingData* tiling)
{
    rowVal = tiling->row;
    colVal = tiling->col;
    avgFactor = tiling->avg_factor;
    dataType = tiling->data_type;
    blockFactor = tiling->block_factor;
    ubFactor = tiling->ub_factor;
    coreCalcNum = tiling->core_calc_num;
    coreCalcTail = tiling->core_calc_tail;
    numBlocks = tiling->block_dim;
    ubCalcNum = tiling->ub_calc_num;
    ubCalcTail = tiling->ub_calc_tail;
    ubCalcLoop = tiling->ub_calc_loop;
    ubCalcTailNum = tiling->ub_calc_tail_num;
    ubCalcTailTail = tiling->ub_calc_tail_tail;
    ubCalcTailLoop = tiling->ub_calc_tail_loop;
    fixedOutput = tiling->fixed_output;

    if constexpr (is_same<T, half>::value) {
        alignLen = ALIGN_16;
        bufferNum = BUFFER_NUM;
    } else if constexpr (is_same<T, float>::value) {
        alignLen = ALIGN_32;
        bufferNum = BUFFER_NUM;
    }
}

template <typename T>
__aicore__ inline void RmsNormGradWholeReduceN<T>::Process()
{
    CopyGammaIn();
    LocalTensor<T> gammaUb = inQueGamma.DeQue<T>();
    LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
    Duplicate(dgamma, 0.0f, colVal);
    if (coreCalcTail == 0) {
        for (uint32_t i = 0; i < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); i++) {
            CopyIn(i, ubCalcNum);
            Compute(ubCalcNum, gammaUb, dgamma);
            CopyOut(i, ubCalcNum);
        }
        if (ubCalcTail != 0) {
            CopyIn(ubCalcLoop - 1, ubCalcTail);
            Compute(ubCalcTail, gammaUb, dgamma);
            CopyOut(ubCalcLoop - 1, ubCalcTail);
        }
    } else {
        if (blockIdx < numBlocks - 1) {
            for (uint32_t i = 0; i < (ubCalcTail == 0 ? ubCalcLoop : ubCalcLoop - 1); i++) {
                CopyIn(i, ubCalcNum);
                Compute(ubCalcNum, gammaUb, dgamma);
                CopyOut(i, ubCalcNum);
            }
            if (ubCalcTail != 0) {
                CopyIn(ubCalcLoop - 1, ubCalcTail);
                Compute(ubCalcTail, gammaUb, dgamma);
                CopyOut(ubCalcLoop - 1, ubCalcTail);
            }
        } else {
            for (uint32_t i = 0; i < (ubCalcTailTail == 0 ? ubCalcTailLoop : ubCalcTailLoop - 1); i++) {
                CopyIn(i, ubCalcTailNum);
                Compute(ubCalcTailNum, gammaUb, dgamma);
                CopyOut(i, ubCalcTailNum);
            }
            if (ubCalcTailTail != 0) {
                CopyIn(ubCalcTailLoop - 1, ubCalcTailTail);
                Compute(ubCalcTailTail, gammaUb, dgamma);
                CopyOut(ubCalcTailLoop - 1, ubCalcTailTail);
            }
        }
    }
    inQueGamma.FreeTensor(gammaUb);
    outQueDgamma.EnQue(dgamma);
    if (fixedOutput == 1) {
        CopyDgammaOutInOrder();
    } else {
        CopyDgammaOut();
    }
}

template <typename T>
__aicore__ inline void RmsNormGradWholeReduceN<T>::CopyGammaIn()
{
    LocalTensor<T> gammaUb = inQueGamma.AllocTensor<T>();
    DataCopy(gammaUb, gammaGm, colVal);
    inQueGamma.EnQue(gammaUb);
}

template <typename T>
__aicore__ inline void RmsNormGradWholeReduceN<T>::CopyIn(uint32_t loopIdx, uint32_t calcLen)
{
    LocalTensor<float> rstd = inQueRstd.AllocTensor<float>();
    DataCopy(rstd, rstdGm[loopIdx * ubFactor], (calcLen + alignLen - 1) / alignLen * alignLen);
    inQueRstd.EnQue(rstd);
    LocalTensor<T> x = inQueX.AllocTensor<T>();
    DataCopy(x, xGm[loopIdx * ubFactor * colVal], calcLen * colVal);
    inQueX.EnQue(x);
    LocalTensor<T> dy = inQueDY.AllocTensor<T>();
    DataCopy(dy, dyGm[loopIdx * ubFactor * colVal], calcLen * colVal);
    inQueDY.EnQue(dy);
}

template <typename T>
__aicore__ inline void RmsNormGradWholeReduceN<T>::CopyOut(uint32_t loopIdx, uint32_t calcLen)
{
    LocalTensor<T> dx = outQueDX.DeQue<T>();
    DataCopy(dxGm[loopIdx * ubFactor * colVal], dx, calcLen * colVal);
    outQueDX.FreeTensor(dx);
}

template <typename T>
__aicore__ inline void RmsNormGradWholeReduceN<T>::CopyDgammaOut()
{
    LocalTensor<float> dgammaOut = outQueDgamma.DeQue<float>();
    SetAtomicAdd<float>();
    DataCopy(dgammaGm, dgammaOut, colVal);
    SetAtomicNone();
    outQueDgamma.FreeTensor(dgammaOut);
}

template <typename T>
__aicore__ inline void RmsNormGradWholeReduceN<T>::CopyDgammaOutInOrder()
{
    int32_t whileMax = 2147483647;
    for (int32_t count = 0; count < whileMax; count++) {
        if (fixed_output_sync[0] == blockIdx) {
            whileMax = 0;
        }
    }
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> dgammaOut = outQueDgamma.DeQue<float>();
    SetAtomicAdd<float>();
    DataCopy(dgammaGm, dgammaOut, colVal);
    SetAtomicNone();
    outQueDgamma.FreeTensor(dgammaOut);
    fixed_output_sync[0] = fixed_output_sync[0] + 1;
}

template <typename T>
__aicore__ inline void RmsNormGradWholeReduceN<T>::Compute(
    uint32_t calcLen, LocalTensor<T>& gammaUb, LocalTensor<float>& dgamma)
{
    LocalTensor<T> xUb = inQueX.DeQue<T>();
    LocalTensor<T> dyUb = inQueDY.DeQue<T>();
    LocalTensor<float> rstdUb = inQueRstd.DeQue<float>();
    LocalTensor<T> dxUb = outQueDX.AllocTensor<T>();
    ComputeMain(xUb, dxUb, dyUb, rstdUb, gammaUb, dgamma, calcLen);
    outQueDX.EnQue(dxUb);
}

template <>
__aicore__ inline void RmsNormGradWholeReduceN<half>::ComputeMain(
    LocalTensor<half>& x, LocalTensor<half>& dx, LocalTensor<half>& dy, LocalTensor<float>& rstd,
    LocalTensor<half>& gamma, LocalTensor<float>& dgamma, uint32_t calcLen)
{
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> dySum = ndBufFp32Buf1.Get<float>();
    LocalTensor<float> tmp32Buf = ndBufFp32Buf3.Get<float>();
    LocalTensor<float> tmp32Buf2 = ndBufFp32Buf2.Get<float>();
    Cast(tmp32Buf2, x, RoundMode::CAST_NONE, colVal * calcLen);
    Duplicate(dySum, 0.0f, colVal * calcLen);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < calcLen; i++) {
        float rstdValue = rstd.GetValue(i);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        Muls(tmp32Buf2[i * colVal], tmp32Buf2[i * colVal], rstdValue, colVal);
        PipeBarrier<PIPE_V>();
        Cast(tmp32Buf[i * colVal], gamma, RoundMode::CAST_NONE, colVal);
        PipeBarrier<PIPE_V>();
        Cast(dySum[i * colVal], dy[i * colVal], RoundMode::CAST_NONE, colVal);
        PipeBarrier<PIPE_V>();
        Mul(tmp32Buf[i * colVal], tmp32Buf[i * colVal], dySum[i * colVal], colVal);
        PipeBarrier<PIPE_V>();
        if (i == calcLen - 1) {
            inQueRstd.FreeTensor(rstd);
        }
        Mul(dySum[i * colVal], tmp32Buf[i * colVal], tmp32Buf2[i * colVal], colVal);
        PipeBarrier<PIPE_V>();
        uint32_t colSplitNum = colVal;
        uint32_t colSplitTailNum = 0;
        while (colSplitNum >= 32) {
            colSplitTailNum = colSplitNum % 16;
            if (colSplitTailNum != 0) {
                colSplitNum = colSplitNum - colSplitTailNum;
                Add(dySum[i * colVal], dySum[i * colVal], dySum[i * colVal + colSplitNum], colSplitTailNum);
                PipeBarrier<PIPE_V>();
            }
            Add(dySum[i * colVal], dySum[i * colVal], dySum[i * colVal + colSplitNum / 2], colSplitNum / 2);
            PipeBarrier<PIPE_V>();
            colSplitNum = colSplitNum / 2;
        }
        WholeReduceSum(dySum[i * colVal], dySum[i * colVal], colSplitNum, 1, 1, 1, 8);
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float dySumVal = dySum.GetValue(i * colVal) * avgFactor;
        event_t eventSV2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV2);
        WaitFlag<HardEvent::S_V>(eventSV2);
        Muls(dySum[i * colVal], tmp32Buf2[i * colVal], dySumVal, colVal);
        PipeBarrier<PIPE_V>();
        Sub(dySum[i * colVal], tmp32Buf[i * colVal], dySum[i * colVal], colVal);
        PipeBarrier<PIPE_V>();
        Muls(dySum[i * colVal], dySum[i * colVal], rstdValue, colVal);
        PipeBarrier<PIPE_V>();
    }
    Cast(dx, dySum, RoundMode::CAST_NONE, colVal * calcLen);
    Cast(tmp32Buf, dy, RoundMode::CAST_NONE, colVal * calcLen);
    PipeBarrier<PIPE_V>();
    Mul(tmp32Buf, tmp32Buf2, tmp32Buf, colVal * calcLen);
    PipeBarrier<PIPE_V>();
    inQueX.FreeTensor(x);
    inQueDY.FreeTensor(dy);
    Duplicate(dySum, 0.0f, colVal);
    PipeBarrier<PIPE_V>();
    uint32_t calcHalfRemain = calcLen % 2;
    uint32_t calcHalf = calcLen / 2;
    if (calcHalfRemain != 0) {
        Add(dySum, dySum, tmp32Buf[calcHalf * colVal * 2], colVal);
        PipeBarrier<PIPE_V>();
    }
    while (calcHalf > 0) {
        Add(tmp32Buf, tmp32Buf, tmp32Buf[calcHalf * colVal], calcHalf * colVal);
        PipeBarrier<PIPE_V>();
        calcHalfRemain = calcHalf % 2;
        calcHalf = calcHalf / 2;
        if (calcHalfRemain != 0) {
            Add(dySum, dySum, tmp32Buf[calcHalf * colVal * 2], colVal);
            PipeBarrier<PIPE_V>();
        }
    }
    Add(dgamma, dySum, dgamma, colVal);
    PipeBarrier<PIPE_V>();
}

template <>
__aicore__ inline void RmsNormGradWholeReduceN<float>::ComputeMain(
    LocalTensor<float>& x, LocalTensor<float>& dx, LocalTensor<float>& dy, LocalTensor<float>& rstd,
    LocalTensor<float>& gamma, LocalTensor<float>& dgamma, uint32_t calcLen)
{
    LocalTensor<float> tmp_buf = ndBufFp32Buf1.Get<float>(calcLen * colVal);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t i = 0; i < calcLen; i++) {
        float rstdValue = rstd.GetValue(i);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        // grad_y = dy * g
        Mul(tmp_buf[i * colVal], dy[i * colVal], gamma, colVal);
        // y = x * rstd
        Muls(x[i * colVal], x[i * colVal], rstdValue, colVal);
        PipeBarrier<PIPE_V>();
        Mul(dy[i * colVal], dy[i * colVal], x[i * colVal], colVal);
        PipeBarrier<PIPE_V>();
    }
    Duplicate(dx, 0.0f, colVal);
    PipeBarrier<PIPE_V>();
    uint32_t calcHalfRemain = calcLen % 2;
    uint32_t calcHalf = calcLen / 2;
    if (calcHalfRemain != 0) {
        Add(dx, dx, dy[calcHalf * colVal * 2], colVal);
        PipeBarrier<PIPE_V>();
    }
    while (calcHalf > 0) {
        Add(dy, dy, dy[calcHalf * colVal], calcHalf * colVal);
        PipeBarrier<PIPE_V>();
        calcHalfRemain = calcHalf % 2;
        calcHalf = calcHalf / 2;
        if (calcHalfRemain != 0) {
            Add(dx, dx, dy[calcHalf * colVal * 2], colVal);
            PipeBarrier<PIPE_V>();
        }
    }
    Add(dgamma, dx, dgamma, colVal);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < calcLen; i++) {
        // mean = sum(grad_y * y) * avg_factor
        Duplicate(dy[i * colVal], 0.0f, colVal);
        PipeBarrier<PIPE_V>();
        Mul(dy[i * colVal], tmp_buf[i * colVal], x[i * colVal], colVal);
        PipeBarrier<PIPE_V>();
        uint32_t colSplitNum = colVal;
        uint32_t colSplitTailNum = 0;
        while (colSplitNum >= 32) {
            colSplitTailNum = colSplitNum % 16;
            if (colSplitTailNum != 0) {
                colSplitNum = colSplitNum - colSplitTailNum;
                Add(dy[i * colVal], dy[i * colVal], dy[i * colVal + colSplitNum], colSplitTailNum);
                PipeBarrier<PIPE_V>();
            }
            Add(dy[i * colVal], dy[i * colVal], dy[i * colVal + colSplitNum / 2], colSplitNum / 2);
            PipeBarrier<PIPE_V>();
            colSplitNum = colSplitNum / 2;
        }
        WholeReduceSum(dy[i * colVal], dy[i * colVal], colSplitNum, 1, 1, 1, 8);
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float dySumVal = dy.GetValue(i * colVal) * avgFactor;
        event_t eventSV2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV2);
        WaitFlag<HardEvent::S_V>(eventSV2);
        Muls(x[i * colVal], x[i * colVal], dySumVal, colVal);
        PipeBarrier<PIPE_V>();
        Sub(tmp_buf[i * colVal], tmp_buf[i * colVal], x[i * colVal], colVal);
        event_t eventVS2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS2);
        WaitFlag<HardEvent::V_S>(eventVS2);
        float rstdValue = rstd.GetValue(i);
        event_t eventSV3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV3);
        WaitFlag<HardEvent::S_V>(eventSV3);
        if (i == calcLen - 1) {
            inQueX.FreeTensor(x);
            inQueDY.FreeTensor(dy);
            inQueRstd.FreeTensor(rstd);
        }
        Muls(dx[i * colVal], tmp_buf[i * colVal], rstdValue, colVal);
        PipeBarrier<PIPE_V>();
    }
}

#endif // RMS_NORM_GRAD_WHOLE_REDUCE_N_H