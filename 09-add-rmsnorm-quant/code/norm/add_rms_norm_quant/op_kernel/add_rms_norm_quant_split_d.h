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
 * \file add_rms_norm_quant_split_d.h
 * \brief add rms norm quant split d h file
 */
#ifndef _ADD_RMS_NORM_QUANT_SPLIT_D_H_
#define _ADD_RMS_NORM_QUANT_SPLIT_D_H_
#include "add_rms_norm_quant_base.h"

using namespace AscendC;
using namespace AddRmsNormQuantBase;

template <typename TX, typename TScale, typename TOffset, bool RN, bool A, bool PT>
class KernelAddRmsNormQuantSplitD {
public:
    __aicore__ inline KernelAddRmsNormQuantSplitD(TPipe* pipe)
    {
        Ppipe = pipe;
    }
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zero_points1,
        GM_ADDR zero_points2, GM_ADDR beta, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR res_out,
        const AddRMSNormQuantTilingData* tilingData)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim1 can not be zero!");
        this->numCol = tilingData->numCol;
        this->numRow = tilingData->numRow;
        this->rowFactor = tilingData->rowFactor;
        this->blockFactor = tilingData->blockFactor;
        this->epsilon = tilingData->epsilon;
        this->ubFactor = tilingData->ubFactor;
        this->hasZeroPoints1 = tilingData->hasZeroPoints1;
        this->avgFactor = (float)1.0 / numCol;
        this->divMode = tilingData->divMode;
        this->hasBeta = tilingData->hasBeta;
        this->hasZeroPoints2 = tilingData->hasZeroPoints2 && !PT;
        this->hasScales2 = tilingData->hasScales2 && !PT;

        blockIdx_v1 = GetBlockIdx();
        if (blockIdx_v1 < GetBlockNum() - 1) {
            this->rowWork = blockFactor;
        } else if (blockIdx_v1 == GetBlockNum() - 1) {
            this->rowWork = numRow - (GetBlockNum() - 1) * blockFactor;
        } else {
        }
        // get start index for current core, core parallel
        x1Gm.SetGlobalBuffer((__gm__ TX*)x1 + blockIdx_v1 * blockFactor * numCol, rowWork * numCol);
        x2Gm.SetGlobalBuffer((__gm__ TX*)x2 + blockIdx_v1 * blockFactor * numCol, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ TX*)gamma, numCol);
        scales1Gm.SetGlobalBuffer((__gm__ TScale*)scales1, numCol);
        if (hasZeroPoints1) {
            zeroPoints1Gm.SetGlobalBuffer((__gm__ TOffset*)zero_points1, numCol);
        }
        // out
        y1Gm.SetGlobalBuffer((__gm__ int8_t*)y1 + blockIdx_v1 * blockFactor * numCol, rowWork * numCol);
        y2Gm.SetGlobalBuffer((__gm__ int8_t*)y2 + blockIdx_v1 * blockFactor * numCol, rowWork * numCol);
        xGm.SetGlobalBuffer((__gm__ TX*)x + blockIdx_v1 * blockFactor * numCol, rowWork * numCol);
        if constexpr (RN) {
            resOutGm.SetGlobalBuffer((__gm__ TX*)res_out + blockIdx_v1 * blockFactor * numCol, rowWork * numCol);
        }

        // pipe alloc memory to queue, the unit is Bytes.
        // We need 2 buffers here for both x1 and x2.
        Ppipe->InitBuffer(inQueueX, BUFFER_NUM, 2 * ubFactor * sizeof(TX));
        Ppipe->InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(TX));
        Ppipe->InitBuffer(outQueueY1, BUFFER_NUM, ubFactor * sizeof(TX));
        Ppipe->InitBuffer(scales1Buf, ubFactor * sizeof(float));
        if (hasZeroPoints1) {
            Ppipe->InitBuffer(zeroPoints1Buf, ubFactor * sizeof(int32_t));
        }
        if constexpr (is_same<TX, half>::value || is_same<TX, bfloat16_t>::value) {
            Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        }
        Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));

        Ppipe->InitBuffer(sumBufSplit, rowFactor * NUM_PER_BLK_FP32 * sizeof(float));
        Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
        Ppipe->InitBuffer(rstdBuf, rowFactor * sizeof(float));
        initOptionalParams(scales2, zero_points1, zero_points2, beta);
    }

    __aicore__ inline void initOptionalParams(GM_ADDR scales2, GM_ADDR zero_points1, GM_ADDR zero_points2, GM_ADDR beta)
    {
        if (hasBeta) {
            betaGm.SetGlobalBuffer((__gm__ TX*)beta, numCol);
            Ppipe->InitBuffer(inQueueBeta, BUFFER_NUM, ubFactor * sizeof(TX));
        }

        if (hasScales2) {
            scales2Gm.SetGlobalBuffer((__gm__ TScale*)scales2, numCol);
            Ppipe->InitBuffer(outQueueY2, BUFFER_NUM, ubFactor * sizeof(TX));
            Ppipe->InitBuffer(scales2Buf, ubFactor * sizeof(float));
            Ppipe->InitBuffer(tmpBufSplit, ubFactor * sizeof(float));
            if (hasZeroPoints2) {
                zeroPoints2Gm.SetGlobalBuffer((__gm__ TOffset*)zero_points2, numCol);
                Ppipe->InitBuffer(zeroPoints2Buf, ubFactor * sizeof(int32_t));
            }
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t iOMax = AddRmsNormQuantBase::CeilDiv(rowWork, rowFactor);
        uint32_t rowTail = rowWork - (iOMax - 1) * rowFactor;
        uint32_t jMax = AddRmsNormQuantBase::CeilDiv(numCol, ubFactor);
        uint32_t colTail = numCol - (jMax - 1) * ubFactor;
        for (uint32_t iO = 0; iO < iOMax - 1; iO++) {
            SubProcess(iO, rowFactor, jMax, colTail);
        }
        SubProcess(iOMax - 1, rowTail, jMax, colTail);
    }

    __aicore__ inline void SubProcess(uint32_t iO, uint32_t calcRowNum, uint32_t jMax, uint32_t colTail)
    {
        LocalTensor<float> sumLocal = sumBufSplit.Get<float>();

        LocalTensor<float> rstdLocal = rstdBuf.Get<float>();
        Duplicate(rstdLocal, (float)0.0, calcRowNum);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < jMax - 1; j++) {
            ComputeFormer(iO, calcRowNum, j, rstdLocal, sumLocal, ubFactor);
        }
        // do tail
        ComputeFormer(iO, calcRowNum, jMax - 1, rstdLocal, sumLocal, colTail);
        ComputeRstd(rstdLocal, calcRowNum);

        for (uint32_t j = 0; j < jMax - 1; j++) {
            ComputeLatter(iO, calcRowNum, j, rstdLocal, ubFactor);
        }
        ComputeLatter(iO, calcRowNum, jMax - 1, rstdLocal, colTail);
    }

private:
    __aicore__ inline void CopyInAndAdd(uint32_t iIdx, uint32_t jIdx, uint32_t num)
    {
        LocalTensor<TX> x1x2In = inQueueX.AllocTensor<TX>();
        LocalTensor<TX> x1In = x1x2In[0];
        LocalTensor<TX> x2In = x1x2In[ubFactor];
        DataCopyCustom<TX>(x1In, x1Gm[iIdx * numCol + jIdx * ubFactor], num);
        DataCopyCustom<TX>(x2In, x2Gm[iIdx * numCol + jIdx * ubFactor], num);
        inQueueX.EnQue(x1x2In);
        LocalTensor<TX> x1x2Local = inQueueX.DeQue<TX>();

        auto x1Local = x1x2Local[0];
        auto x2Local = x1x2Local[ubFactor];

        LocalTensor<TX> xLocal = outQueueY1.AllocTensor<TX>();
        LocalTensor<float> x1Fp32Local = xFp32Buf.Get<float>();

        if constexpr (is_same<TX, half>::value && !PT) {
            Add(xLocal, x1Local, x2Local, num);
            PipeBarrier<PIPE_V>();
            Cast(x1Fp32Local, xLocal, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
            // x1+x2 saved in x1Fp32Local
        } else if constexpr (is_same<TX, bfloat16_t>::value || PT) {
            LocalTensor<float> x2Fp32Local = x1x2Local.template ReinterpretCast<float>();

            Cast(x1Fp32Local, x1Local, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
            Cast(x2Fp32Local, x2Local, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();

            Add(x1Fp32Local, x1Fp32Local, x2Fp32Local, num);
            PipeBarrier<PIPE_V>();
            #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
            Cast(xLocal, x1Fp32Local, RoundMode::CAST_NONE, num);
            #else
            Cast(xLocal, x1Fp32Local, RoundMode::CAST_RINT, num);
            #endif
            PipeBarrier<PIPE_V>();
            // x1+x2 saved in x1Fp32Local
            // 有beta的bf16场景，做一个x-> bf16 -> fp32的转换
            if (hasBeta && !PT) {
                Cast(x1Fp32Local, xLocal, RoundMode::CAST_NONE, num);
                PipeBarrier<PIPE_V>();
            }
        } else {
            Add(x1Local, x1Local, x2Local, num);
            PipeBarrier<PIPE_V>();
            Adds(xLocal, x1Local, (float)0.0, num);
            // x1+x2 saved in inQueueX
        }
        inQueueX.FreeTensor(x1x2Local);

        // copy out to workspace && x_out
        if constexpr (A) {
            outQueueY1.EnQue(xLocal);
            auto x_out = outQueueY1.DeQue<TX>();
            DataCopyCustom<TX>(xGm[iIdx * numCol + jIdx * ubFactor], x_out, num);
            outQueueY1.FreeTensor(x_out);
        } else {
            outQueueY1.FreeTensor(xLocal);
        }
    }

    __aicore__ inline void ComputeFormer(
        uint32_t iOIdx, uint32_t calcRowNum, uint32_t jIdx, LocalTensor<float>& rstdLocal, LocalTensor<float>& sumLocal,
        uint32_t num)
    {
        for (uint32_t i_i = 0; i_i < calcRowNum; i_i++) {
            CopyInAndAdd(iOIdx * rowFactor + i_i, jIdx, num);
            ComputeSum(i_i, sumLocal, num);
        }
        BlockReduceSumFP32(sumLocal, sumLocal, calcRowNum * NUM_PER_BLK_FP32);
        Add(rstdLocal, rstdLocal, sumLocal, calcRowNum);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeSum(uint32_t iIIdx, LocalTensor<float>& sumLocal, uint32_t num)
    {
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
        if constexpr (is_same<TX, half>::value || is_same<TX, bfloat16_t>::value) {
            LocalTensor<float> xFp32Local = xFp32Buf.Get<float>();
            PipeBarrier<PIPE_V>();
            Mul(sqx, xFp32Local, xFp32Local, num);
        } else {
            LocalTensor<TX> xLocal = inQueueX.AllocTensor<float>();
            PipeBarrier<PIPE_V>();
            Mul(sqx, xLocal, xLocal, num);
            inQueueX.FreeTensor(xLocal);
        }
        PipeBarrier<PIPE_V>();
        Muls(sqx, sqx, avgFactor, num);
        PipeBarrier<PIPE_V>();
        // 8 means 8 fp32 pre block
        ReduceSumFP32ToBlock(sumLocal[iIIdx * 8], sqx, reduce_buf_local, num);
    }

    __aicore__ inline void ComputeRstd(LocalTensor<float> rstdLocal, uint32_t num)
    {
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
        Adds(rstdLocal, rstdLocal, epsilon, num);
        PipeBarrier<PIPE_V>();
        Sqrt(rstdLocal, rstdLocal, num);
        Duplicate(reduce_buf_local, ONE, num);
        PipeBarrier<PIPE_V>();
        Div(rstdLocal, reduce_buf_local, rstdLocal, num);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeLatter(
        uint32_t iOIdx, uint32_t calcRowNum, uint32_t jIdx, LocalTensor<float>& rstdLocal, uint32_t num)
    {
        CopyInGammaBeta(jIdx, num);

        LocalTensor<TX> gammaLocal = inQueueGamma.DeQue<TX>();
        LocalTensor<TX> betaLocal;
        if (hasBeta) {
            betaLocal = inQueueBeta.DeQue<TX>();
        }
        for (uint32_t i_i = 0; i_i < calcRowNum; i_i++) {
            CopyInAndAdd(iOIdx * rowFactor + i_i, jIdx, num);
            uint32_t computeParams[3] = {num, iOIdx * rowFactor + i_i, jIdx};
            ComputeY(i_i, gammaLocal, rstdLocal, computeParams, betaLocal);
            CopyOutY(iOIdx * rowFactor + i_i, jIdx, num);
        }
        inQueueGamma.FreeTensor(gammaLocal);
        if (hasBeta) {
            inQueueBeta.FreeTensor(betaLocal);
        }
    }

    __aicore__ inline void CopyInGammaBeta(uint32_t jIdx, uint32_t num)
    {
        LocalTensor<TX> gammaLocal = inQueueGamma.AllocTensor<TX>();
        DataCopyCustom<TX>(gammaLocal, gammaGm[jIdx * ubFactor], num);
        inQueueGamma.EnQue(gammaLocal);

        if (hasBeta) {
            LocalTensor<TX> betaLocal = inQueueBeta.AllocTensor<TX>();
            DataCopyCustom<TX>(betaLocal, betaGm[jIdx * ubFactor], num);
            inQueueBeta.EnQue(betaLocal);
        }

        AddRmsNormQuantBase::CopyInScales<TScale>(scales1Buf, scales1Gm[jIdx * ubFactor], num, ubFactor);
        if (PT) {
            AddRmsNormQuantBase::CopyInZeroPoints<TOffset>(zeroPoints1Buf, zeroPoints1Gm, num, ubFactor, hasZeroPoints1);
        } else {
            AddRmsNormQuantBase::CopyInZeroPoints<TOffset>(zeroPoints1Buf, zeroPoints1Gm[jIdx * ubFactor], num, ubFactor, hasZeroPoints1);
        }
        if (hasScales2) {
            AddRmsNormQuantBase::CopyInScales<TScale>(scales2Buf, scales2Gm[jIdx * ubFactor], num, ubFactor);
            AddRmsNormQuantBase::CopyInZeroPoints<TOffset>(
                zeroPoints2Buf, zeroPoints2Gm[jIdx * ubFactor], num, ubFactor, hasZeroPoints2);
        }
    }

    __aicore__ inline void ComputeY(uint32_t iIIdx, LocalTensor<TX>& gammaLocal, LocalTensor<float>& rstdLocal,
        uint32_t *computeParams, LocalTensor<TX>& betaLocal=nullptr)
    {
        uint32_t num = computeParams[0];
        uint32_t iIdx = computeParams[1];
        uint32_t jIdx = computeParams[2];
        LocalTensor<float> xFp32Local = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float rstdValue = rstdLocal.GetValue(iIIdx);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        PipeBarrier<PIPE_V>();
        Muls(xFp32Local, xFp32Local, rstdValue, num);
        PipeBarrier<PIPE_V>();

        if constexpr (is_same<TX, half>::value && !PT) {
            LocalTensor<half> xFp16Cast = sqxBuf.Get<half>();
            Cast(xFp16Cast, xFp32Local, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
            Mul(xFp16Cast, gammaLocal, xFp16Cast, num);
            PipeBarrier<PIPE_V>();
            Cast(xFp32Local, xFp16Cast, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
        } else { // bfloat16
            Cast(sqx, gammaLocal, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
            Mul(xFp32Local, sqx, xFp32Local, num);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (RN) {
            LocalTensor<TX> rnLocal = outQueueY1.AllocTensor<TX>();
            doCast(rnLocal, xFp32Local, num);
            outQueueY1.EnQue(rnLocal);
            auto rnOut = outQueueY1.DeQue<TX>();
            DataCopyCustom<TX>(resOutGm[iIdx * numCol + jIdx * ubFactor], rnOut, num);
            outQueueY1.FreeTensor(rnOut);
        }
        if (hasBeta) {
            PipeBarrier<PIPE_MTE2>();
            Cast(sqx, betaLocal, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
            Add(xFp32Local, xFp32Local, sqx, num);
            PipeBarrier<PIPE_V>();
        }
        doQuant(xFp32Local, num);
    }

    __aicore__ inline void doCast(LocalTensor<TX>& rnLocal, LocalTensor<float>& xFp32Local, uint32_t num)
    {
        #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
        Cast(rnLocal, xFp32Local, RoundMode::CAST_NONE, num);
        #else
        Cast(rnLocal, xFp32Local, RoundMode::CAST_RINT, num);
        #endif
    }

    __aicore__ inline void doQuant(LocalTensor<float> xFp32Local, uint32_t num)
    {
        if (hasScales2) {
            LocalTensor<float> tmpFp32 = tmpBufSplit.Get<float>();
            AddRmsNormQuantBase::doScales(tmpFp32, xFp32Local, scales2Buf, divMode, num);
            AddRmsNormQuantBase::doZeroPoints(tmpFp32, zeroPoints2Buf, num, hasZeroPoints2);
            LocalTensor<int8_t> y2Local = outQueueY2.AllocTensor<int8_t>();
            RoundFloat2Int8(y2Local, tmpFp32, num);
            outQueueY2.EnQue<int8_t>(y2Local);
        }

        if constexpr (PT) {
            float scale;
            if constexpr (is_same<TScale, bfloat16_t>::value) {
                scale = 1.0f / ToFloat(scales1Gm.GetValue(0));
            } else {
                scale = 1.0f / scales1Gm.GetValue(0);
            }
            Muls(xFp32Local, xFp32Local, scale, num);
            PipeBarrier<PIPE_V>();
        } else {
            AddRmsNormQuantBase::doScales(xFp32Local, xFp32Local, scales1Buf, divMode, num);
        }
        if (hasZeroPoints1 && PT) {
            LocalTensor<float> zeroPoints1Fp32 = zeroPoints1Buf.Get<float>();
            int32_t eventIDVToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIDVToS);
            WaitFlag<HardEvent::V_S>(eventIDVToS);
            Adds(xFp32Local, xFp32Local, zeroPoints1Fp32.GetValue(0), num);
            PipeBarrier<PIPE_V>();
        } else {
            AddRmsNormQuantBase::doZeroPoints(xFp32Local, zeroPoints1Buf, num, hasZeroPoints1);
        }
        LocalTensor<int8_t> y1Local = outQueueY1.AllocTensor<int8_t>();
        RoundFloat2Int8(y1Local, xFp32Local, num);
        outQueueY1.EnQue<int8_t>(y1Local);
    }

    __aicore__ inline void CopyOutY(uint32_t iIdx, uint32_t jIdx, uint32_t num)
    {
        LocalTensor<int8_t> y1Local = outQueueY1.DeQue<int8_t>();
        PipeBarrier<PIPE_ALL>();
        DataCopyCustom<int8_t>(y1Gm[iIdx * numCol + jIdx * ubFactor], y1Local, num);
        PipeBarrier<PIPE_ALL>();
        outQueueY1.FreeTensor(y1Local);

        if (hasScales2) {
            LocalTensor<int8_t> y2Local = outQueueY2.DeQue<int8_t>();
            PipeBarrier<PIPE_ALL>();
            DataCopyCustom<int8_t>(y2Gm[iIdx * numCol + jIdx * ubFactor], y2Local, num);
            PipeBarrier<PIPE_ALL>();
            outQueueY2.FreeTensor(y2Local);
        }
    }

private:
    TPipe* Ppipe = nullptr;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGamma;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueBeta;
    // create queues for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY1;

    TBuf<TPosition::VECCALC> scales1Buf;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> zeroPoints1Buf;
    TBuf<TPosition::VECCALC> sqxBuf;
    TBuf<TPosition::VECCALC> tmpBufSplit;
    TBuf<TPosition::VECCALC> sumBufSplit;
    TBuf<TPosition::VECCALC> reduceFp32Buf;
    TBuf<TPosition::VECCALC> rstdBuf;
    TBuf<TPosition::VECCALC> scales2Buf;
    TBuf<TPosition::VECCALC> zeroPoints2Buf;

    GlobalTensor<TX> x2Gm;
    GlobalTensor<TX> x1Gm;
    GlobalTensor<TX> gammaGm;
    GlobalTensor<TX> betaGm;
    GlobalTensor<TScale> scales2Gm;
    GlobalTensor<TScale> scales1Gm;
    GlobalTensor<TOffset> zeroPoints1Gm;
    GlobalTensor<int8_t> y1Gm;
    GlobalTensor<TOffset> zeroPoints2Gm;
    GlobalTensor<int8_t> y2Gm;
    GlobalTensor<TX> xGm;
    GlobalTensor<TX> resOutGm;

    uint32_t numCol;
    uint32_t numRow;
    uint32_t blockFactor; // number of calculations rows on each core
    uint32_t ubFactor;
    uint32_t rowFactor;
    float epsilon;
    float avgFactor;
    uint32_t hasZeroPoints1 = 0;
    uint32_t hasBeta = 0;
    uint32_t hasScales2 = 0;
    uint32_t divMode = 1;
    uint32_t hasZeroPoints2 = 0;
    int32_t blockIdx_v1;
    uint32_t rowWork = 1;

    int tempbufNum;
};
#endif // _ADD_RMS_NORM_QUANT_SPLIT_D_H_