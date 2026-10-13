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
 * \file add_rms_norm_quant.h
 * \brief add rms norm quant file
 */
#ifndef _ADD_RMS_NORM_QUANT_H_
#define _ADD_RMS_NORM_QUANT_H_
#include "add_rms_norm_quant_base.h"

using namespace AscendC;
using namespace AddRmsNormQuantBase;

template <typename TX, typename TScale, typename TOffset, bool RN, bool A, bool PT>
class KernelAddRmsNormQuant {
public:
    __aicore__ inline KernelAddRmsNormQuant(TPipe* pipe)
    {
        Ppipe = pipe;
    }
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zero_points1,
        GM_ADDR zero_points2, GM_ADDR beta, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR res_out,
        const AddRMSNormQuantTilingData* tilingData)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        this->numRow = tilingData->numRow;
        this->numCol = tilingData->numCol;
        this->blockFactor = tilingData->blockFactor;
        this->rowFactor = tilingData->rowFactor;
        this->ubFactor = tilingData->ubFactor;
        this->epsilon = tilingData->epsilon;
        this->avgFactor = (float)1.0 / numCol;
        this->hasZeroPoints1 = tilingData->hasZeroPoints1;
        this->hasBeta = tilingData->hasBeta;
        this->divMode = tilingData->divMode;
        this->hasScales2 = tilingData->hasScales2 && !PT;
        this->hasZeroPoints2 = tilingData->hasZeroPoints2 && !PT;
        blockIdx_ = GetBlockIdx();
        if (blockIdx_ < GetBlockNum() - 1) {
            this->rowWork = blockFactor;
        } else if (blockIdx_ == GetBlockNum() - 1) {
            this->rowWork = numRow - (GetBlockNum() - 1) * blockFactor;
        } else {
        }
        // get start index for current core, core parallel
        x1Gm.SetGlobalBuffer((__gm__ TX*)x1 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        x2Gm.SetGlobalBuffer((__gm__ TX*)x2 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ TX*)gamma, numCol);
        scales1Gm.SetGlobalBuffer((__gm__ TScale*)scales1, numCol);
        // out
        y1Gm.SetGlobalBuffer((__gm__ int8_t*)y1 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        y2Gm.SetGlobalBuffer((__gm__ int8_t*)y2 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        xGm.SetGlobalBuffer((__gm__ TX*)x + blockIdx_ * blockFactor * numCol, rowWork * numCol);

        // pipe alloc memory to queue, the unit is Bytes
        Ppipe->InitBuffer(inQueueX, BUFFER_NUM, ubFactor * sizeof(TX));
        Ppipe->InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(TX));
        Ppipe->InitBuffer(scales1Buf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(outQueueY1, BUFFER_NUM, ubFactor * sizeof(TX));

        if constexpr (is_same<TX, half>::value || is_same<TX, bfloat16_t>::value) {
            Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        }
        Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
        initOptionalParams(scales2, zero_points1, zero_points2, beta, res_out);
    }

    __aicore__ inline void initOptionalParams(GM_ADDR scales2, GM_ADDR zero_points1, GM_ADDR zero_points2, GM_ADDR beta, GM_ADDR res_out)
    {
        if (hasBeta) {
            betaGm.SetGlobalBuffer((__gm__ TX*)beta, numCol);
            Ppipe->InitBuffer(inQueueBeta, BUFFER_NUM, ubFactor * sizeof(TX));
        }
        if (hasZeroPoints1) {
            zeroPoints1Gm.SetGlobalBuffer((__gm__ TOffset*)zero_points1, numCol);
            Ppipe->InitBuffer(zeroPoints1Buf, ubFactor * sizeof(int32_t));
        }
        if (hasScales2) {
            Ppipe->InitBuffer(tmpBuf, ubFactor * sizeof(float));
            scales2Gm.SetGlobalBuffer((__gm__ TScale*)scales2, numCol);
            Ppipe->InitBuffer(outQueueY2, BUFFER_NUM, ubFactor * sizeof(TX));
            Ppipe->InitBuffer(scales2Buf, ubFactor * sizeof(float));
            if (hasZeroPoints2) {
                zeroPoints2Gm.SetGlobalBuffer((__gm__ TOffset*)zero_points2, numCol);
                Ppipe->InitBuffer(zeroPoints2Buf, ubFactor * sizeof(int32_t));
            }
        }
        if (RN) {
            resOutGm.SetGlobalBuffer((__gm__ TX*)res_out + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        }
    }

    __aicore__ inline void Process()
    {
        CopyInGammaBeta();
        LocalTensor<TX> gammaLocal = inQueueGamma.DeQue<TX>();
        LocalTensor<TX> betaLocal;
        if (hasBeta) {
            betaLocal = inQueueBeta.DeQue<TX>();
        }

        uint32_t iOMax = AddRmsNormQuantBase::CeilDiv(rowWork, rowFactor);
        uint32_t rowTail = rowWork - (iOMax - 1) * rowFactor;

        for (uint32_t iO = 0; iO < iOMax - 1; iO++) {
            computeBefore(iO, rowFactor, gammaLocal, betaLocal);
        }
        computeBefore(iOMax - 1, rowTail, gammaLocal, betaLocal);
        inQueueGamma.FreeTensor(gammaLocal);
        if (hasBeta) {
            inQueueBeta.FreeTensor(betaLocal);
        }
    }

    __aicore__ inline void computeBefore(
        uint32_t iO, uint32_t calcRowNum, LocalTensor<TX>& gammaLocal, LocalTensor<TX>& betaLocal = nullptr)
    {
        for (uint32_t iI = 0; iI < calcRowNum; iI++) {
            uint32_t gmBias = (iO * rowFactor + iI) * numCol;
            CopyIn(gmBias);
            Compute(iI, gammaLocal, gmBias, betaLocal);
            CopyOutY(gmBias);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t gmBias)
    {
        LocalTensor<TX> x1LocalIn = inQueueX.AllocTensor<TX>();
        LocalTensor<TX> x2Local = sqxBuf.Get<TX>();
        LocalTensor<TX> xLocal = outQueueY1.AllocTensor<TX>();

        if constexpr (is_same<TX, half>::value || is_same<TX, bfloat16_t>::value) {
            x2Local = x2Local[ubFactor];
        }

        DataCopyCustom<TX>(x1LocalIn, x1Gm[gmBias], numCol);
        DataCopyCustom<TX>(x2Local, x2Gm[gmBias], numCol);
        inQueueX.EnQue(x1LocalIn);
        auto x1Local = inQueueX.DeQue<TX>();
        LocalTensor<float> x1Fp32Local = xFp32Buf.Get<float>();
        if constexpr (is_same<TX, half>::value && !PT) {
            Add(xLocal, x1Local, x2Local, numCol);
            PipeBarrier<PIPE_V>();
            Cast(x1Fp32Local, xLocal, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
        } else if constexpr (is_same<TX, bfloat16_t>::value || PT) {
            LocalTensor<float> x2_fp32 = sqxBuf.Get<float>();
            Cast(x1Fp32Local, x1Local, RoundMode::CAST_NONE, numCol);
            Cast(x2_fp32, x2Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(x1Fp32Local, x1Fp32Local, x2_fp32, numCol);
            PipeBarrier<PIPE_V>();
            #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
            Cast(xLocal, x1Fp32Local, RoundMode::CAST_NONE, numCol);
            #else
            Cast(xLocal, x1Fp32Local, RoundMode::CAST_RINT, numCol);
            #endif
            PipeBarrier<PIPE_V>();
            // 有beta的bf16场景，做一个x-> bf16 -> fp32的转换
            if (hasBeta && !PT) {
                Cast(x1Fp32Local, xLocal, RoundMode::CAST_NONE, numCol);
                PipeBarrier<PIPE_V>();
            }
        } else {
            Add(x1Local, x1Local, x2Local, numCol);
            PipeBarrier<PIPE_V>();
            Adds(xLocal, x1Local, (float)0, numCol);
        }
        inQueueX.FreeTensor(x1Local);

        // CopyOut x1 + x2
        if constexpr (A) {
            outQueueY1.EnQue(xLocal);
            auto xOut = outQueueY1.DeQue<TX>();
            DataCopyCustom<TX>(xGm[gmBias], xOut, numCol);
            outQueueY1.FreeTensor(xOut);
        } else {
            outQueueY1.FreeTensor(xLocal);
        }
    }

    __aicore__ inline void CopyInGammaBeta()
    {
        LocalTensor<TX> gammaLocal = inQueueGamma.AllocTensor<TX>();
        DataCopyCustom<TX>(gammaLocal, gammaGm, numCol);
        inQueueGamma.EnQue(gammaLocal);

        if (hasBeta) {
            LocalTensor<TX> betaLocal = inQueueBeta.AllocTensor<TX>();
            DataCopyCustom<TX>(betaLocal, betaGm, numCol);
            inQueueBeta.EnQue(betaLocal);
        }

        AddRmsNormQuantBase::CopyInScales<TScale>(scales1Buf, scales1Gm, numCol, ubFactor);
        AddRmsNormQuantBase::CopyInZeroPoints<TOffset>(zeroPoints1Buf, zeroPoints1Gm, numCol, ubFactor, hasZeroPoints1);
        if (hasScales2) {
            AddRmsNormQuantBase::CopyInScales<TScale>(scales2Buf, scales2Gm, numCol, ubFactor);
            AddRmsNormQuantBase::CopyInZeroPoints<TOffset>(
                zeroPoints2Buf, zeroPoints2Gm, numCol, ubFactor, hasZeroPoints2);
        }
    }

    __aicore__ inline void Compute(
        uint32_t inner_progress, LocalTensor<TX> gammaLocal, uint32_t gmBias, LocalTensor<TX> betaLocal=nullptr)
    {
        LocalTensor<float> xFp32Local = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduceLocal = reduceFp32Buf.Get<float>();

        Mul(sqx, xFp32Local, xFp32Local, numCol);
        PipeBarrier<PIPE_V>();

        Muls(sqx, sqx, avgFactor, numCol);
        PipeBarrier<PIPE_V>();

        ReduceSumCustom(sqx, sqx, reduceLocal, numCol);
        PipeBarrier<PIPE_V>();

        Adds(sqx, sqx, epsilon, 1);
        PipeBarrier<PIPE_V>();

        Sqrt(sqx, sqx, 1);
        PipeBarrier<PIPE_V>();
        Duplicate(reduceLocal, ONE, 1);
        PipeBarrier<PIPE_V>();
        Div(sqx, reduceLocal, sqx, 1);
        PipeBarrier<PIPE_V>();
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float rstdValue = sqx.GetValue(0);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        Muls(xFp32Local, xFp32Local, rstdValue, numCol);
        PipeBarrier<PIPE_V>();

        ComputePart2(gammaLocal, gmBias, xFp32Local, sqx, betaLocal);
    }

    __aicore__ inline void ComputePart2(
        LocalTensor<TX> gammaLocal, uint32_t gmBias, LocalTensor<float> xFp32Local, LocalTensor<float> sqx,
        LocalTensor<TX> betaLocal=nullptr) {
        if constexpr (is_same<TX, half>::value && !PT) {
            LocalTensor<half> xFp16Cast = sqxBuf.Get<half>();
            Cast(xFp16Cast, xFp32Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Mul(xFp16Cast, gammaLocal, xFp16Cast, numCol);
            PipeBarrier<PIPE_V>();
            Cast(xFp32Local, xFp16Cast, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
        } else { // bfloat16
            Cast(sqx, gammaLocal, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Mul(xFp32Local, sqx, xFp32Local, numCol);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (RN) {
            LocalTensor<TX> rnLocal = outQueueY1.AllocTensor<TX>();
            #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
            Cast(rnLocal, xFp32Local, RoundMode::CAST_NONE, numCol);
            #else
            Cast(rnLocal, xFp32Local, RoundMode::CAST_RINT, numCol);
            #endif
            outQueueY1.EnQue(rnLocal);
            auto rnOut = outQueueY1.DeQue<TX>();
            DataCopyCustom<TX>(resOutGm[gmBias], rnOut, numCol);
            outQueueY1.FreeTensor(rnOut);
        }
        if (hasBeta) {
            PipeBarrier<PIPE_MTE2>();
            LocalTensor<float> betaFp32 = sqxBuf.Get<float>();
            Cast(betaFp32, betaLocal, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(xFp32Local, xFp32Local, betaFp32, numCol);
            PipeBarrier<PIPE_V>();
        }
        doQuant(xFp32Local);
    }

    __aicore__ inline void doQuant(LocalTensor<float> xFp32Local)
    {
        event_t event_v_mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(event_v_mte);
        WaitFlag<HardEvent::V_MTE2>(event_v_mte);
        // do quant2
        if (hasScales2) {
            LocalTensor<float> tmpFp32 = tmpBuf.Get<float>();
            AddRmsNormQuantBase::doScales(tmpFp32, xFp32Local, scales2Buf, divMode, numCol);
            AddRmsNormQuantBase::doZeroPoints(tmpFp32, zeroPoints2Buf, numCol, hasZeroPoints2);
            LocalTensor<int8_t> y2Local = outQueueY2.AllocTensor<int8_t>();
            RoundFloat2Int8(y2Local, tmpFp32, numCol);
            outQueueY2.EnQue<int8_t>(y2Local);
        }

        // do quant1
        if constexpr (PT) {
            float scale;
            if constexpr (is_same<TScale, bfloat16_t>::value) {
                scale = 1.0f / ToFloat(scales1Gm.GetValue(0));
            } else {
                scale = 1.0f / scales1Gm.GetValue(0);
            }
            Muls(xFp32Local, xFp32Local, scale, numCol);
            PipeBarrier<PIPE_V>();
        } else {
            AddRmsNormQuantBase::doScales(xFp32Local, xFp32Local, scales1Buf, divMode, numCol);
        }
        if (hasZeroPoints1 && PT) {
            LocalTensor<float> zeroPoints1Fp32 = zeroPoints1Buf.Get<float>();
            int32_t eventIDVToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIDVToS);
            WaitFlag<HardEvent::V_S>(eventIDVToS);
            Adds(xFp32Local, xFp32Local, zeroPoints1Fp32.GetValue(0), numCol);
            PipeBarrier<PIPE_V>();
        } else {
            AddRmsNormQuantBase::doZeroPoints(xFp32Local, zeroPoints1Buf, numCol, hasZeroPoints1);
        }
        LocalTensor<int8_t> y1Local = outQueueY1.AllocTensor<int8_t>();
        RoundFloat2Int8(y1Local, xFp32Local, numCol);
        outQueueY1.EnQue<int8_t>(y1Local);
    }

    __aicore__ inline void CopyOutY(uint32_t progress)
    {
        LocalTensor<int8_t> y1Local = outQueueY1.DeQue<int8_t>();
        DataCopyCustom<int8_t>(y1Gm[progress], y1Local, numCol);
        outQueueY1.FreeTensor(y1Local);

        if (hasScales2) {
            LocalTensor<int8_t> y2Local = outQueueY2.DeQue<int8_t>();
            DataCopyCustom<int8_t>(y2Gm[progress], y2Local, numCol);
            outQueueY2.FreeTensor(y2Local);
        }
    }

private:
    TPipe* Ppipe = nullptr;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGamma;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueBeta;
    // create queues for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY1;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY2;

    TBuf<TPosition::VECCALC> scales1Buf;
    TBuf<TPosition::VECCALC> zeroPoints1Buf;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> sqxBuf;
    TBuf<TPosition::VECCALC> reduceFp32Buf;
    TBuf<TPosition::VECCALC> scales2Buf;
    TBuf<TPosition::VECCALC> zeroPoints2Buf;
    TBuf<TPosition::VECCALC> tmpBuf;

    GlobalTensor<TX> x1Gm;
    GlobalTensor<TX> x2Gm;
    GlobalTensor<TX> gammaGm;
    GlobalTensor<TX> betaGm;
    GlobalTensor<TScale> scales1Gm;
    GlobalTensor<TScale> scales2Gm;
    GlobalTensor<TOffset> zeroPoints1Gm;
    GlobalTensor<TOffset> zeroPoints2Gm;
    GlobalTensor<int8_t> y1Gm;
    GlobalTensor<int8_t> y2Gm;
    GlobalTensor<TX> xGm;
    GlobalTensor<TX> resOutGm;

    uint32_t numRow;
    uint32_t numCol;
    uint32_t blockFactor; // number of calculations rows on each core
    uint32_t rowFactor;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    uint32_t hasZeroPoints1 = 0;
    uint32_t hasBeta = 0;
    uint32_t divMode = 1;
    uint32_t hasScales2 = 0;
    uint32_t hasZeroPoints2 = 0;
    int32_t blockIdx_;
    uint32_t rowWork = 1;
};
#endif // ADD_RMS_NORM_QUANT_H_