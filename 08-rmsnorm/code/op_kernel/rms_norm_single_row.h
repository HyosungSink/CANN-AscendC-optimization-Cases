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
 * \file rms_norm_single_row.h
 * \brief rms norm single row file
 */
#ifndef RMS_NORM_SINGLE_ROW_H_
#define RMS_NORM_SINGLE_ROW_H_
#include "rms_norm_base.h"

namespace RmsNorm {
using namespace AscendC;
using namespace RmsNorm;

template <typename T, typename T_GAMMA>
class KernelRmsNormSingleRow {
public:
    __aicore__ inline KernelRmsNormSingleRow()
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, const RMSNormTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        InitVar(tiling);
        blockIdx_ = GetBlockIdx();
        if (blockIdx_ != GetBlockNum() - 1) {
            rowWork = blockFactor;
        } else {
            rowWork = numRow - (GetBlockNum() - 1) * blockFactor;
        }

        xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ T_GAMMA*)gamma, numCol);
        yGm.SetGlobalBuffer((__gm__ T*)y + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockIdx_ * blockFactor, blockFactor);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, ubFactor * sizeof(T_GAMMA));
        pipe.InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(T_GAMMA));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, ubFactor * sizeof(T));
        pipe.InitBuffer(outQueueRstd, BUFFER_NUM, rowFactor * sizeof(float));
        if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
            pipe.InitBuffer(xBufFp32, ubFactor * sizeof(float));
        }
        pipe.InitBuffer(reduceBufFp32, NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void InitVar(const RMSNormTilingData* tiling)
    {
        numRow = tiling->num_row;
        numCol = tiling->num_col;
        blockFactor = tiling->block_factor;
        ubFactor = tiling->ub_factor;
        rowFactor = NUM_PER_REP_FP32;
        epsilon = tiling->epsilon;
        avgFactor = (numCol != 0) ? ((float)1.0 / numCol) : 0;
    }

    __aicore__ inline void Process()
    {
        // only support mix dtype
        if constexpr (IS_MIX_DTYPE) {
            CopyInGamma();
            LocalTensor<T_GAMMA> gammaLocal = inQueueGamma.DeQue<T_GAMMA>();

            uint32_t indexOuterMax = CeilDiv(rowWork, rowFactor);
            uint32_t rowTail = rowWork - (indexOuterMax - 1) * rowFactor;

            for (uint32_t indexOuter = 0; indexOuter < indexOuterMax - 1; indexOuter++) {
                SubProcess(indexOuter, rowFactor, gammaLocal);
            }
            SubProcess(indexOuterMax - 1, rowTail, gammaLocal);
            inQueueGamma.FreeTensor(gammaLocal);
        }
    }

    __aicore__ inline void SubProcess(uint32_t indexOuter, uint32_t calcRowNum, LocalTensor<T_GAMMA>& gammaLocal)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        for (uint32_t indexInner = 0; indexInner < calcRowNum; indexInner++) {
            uint32_t gmBias = (indexOuter * rowFactor + indexInner) * numCol;
            CopyIn(gmBias);
            ComputeMixDtype(indexInner, gammaLocal, rstdLocal);
            CopyOutY(gmBias);
        }
        outQueueRstd.EnQue<float>(rstdLocal);
        CopyOutRstd(indexOuter, calcRowNum);
    }

private:
    __aicore__ inline void CopyIn(uint32_t gmBias)
    {
        // Copy in x
        LocalTensor<T_GAMMA> xLocal = inQueueX.AllocTensor<T_GAMMA>();
        auto xLocalHalf = xLocal.template ReinterpretCast<T>();
        DataCopyCustom<T>(xLocalHalf, xGm[gmBias], numCol);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T_GAMMA> gammaLocal = inQueueGamma.AllocTensor<T_GAMMA>();
        DataCopyCustom<T_GAMMA>(gammaLocal, gammaGm, numCol);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void ComputeMixDtype(
        uint32_t innerProgress, LocalTensor<T_GAMMA> gammaLocal, LocalTensor<float> rstdLocal)
    {
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();

        // 1. Cast x and Cal x^2
        LocalTensor<T_GAMMA> xLocal = inQueueX.DeQue<T_GAMMA>();
        auto xLocalHalf = xLocal.template ReinterpretCast<T>();
        Cast(xLocalFp32, xLocalHalf, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        Mul(xLocal, xLocalFp32, xLocalFp32, numCol);
        PipeBarrier<PIPE_V>();

        // 2. Rstd = 1 / sqrt(1 / reduceDim * reducesum(x^2) + eps)
        float reduceOut = ReduceSumHalfInterval(xLocal, numCol);
        inQueueX.FreeTensor(xLocal);
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float rstdValue = 1 / sqrt(reduceOut * avgFactor + epsilon);
        rstdLocal.SetValue(innerProgress, rstdValue);
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);

        // 3. Y = x * rstd * gamma
        Muls(xLocalFp32, xLocalFp32, rstdValue, numCol);
        PipeBarrier<PIPE_V>();
        Mul(xLocalFp32, xLocalFp32, gammaLocal, numCol);
        PipeBarrier<PIPE_V>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        if constexpr (is_same<T, half>::value) {
            Cast(yLocal, xLocalFp32, RoundMode::CAST_NONE, numCol);
        } else {
            Cast(yLocal, xLocalFp32, RoundMode::CAST_RINT, numCol);
        }
        PipeBarrier<PIPE_V>();
        outQueueY.EnQue<T>(yLocal);
    }

    __aicore__ inline void CopyOutY(uint32_t progress)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyCustom<T>(yGm[progress], yLocal, numCol);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(uint32_t outerProgress, uint32_t num)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        DataCopyCustom<float>(rstdGm[outerProgress * rowFactor], rstdLocal, num);
#endif
        outQueueRstd.FreeTensor(rstdLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueGamma;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY, outQueueRstd;

    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> reduceBufFp32;
    GlobalTensor<T> xGm, yGm;
    GlobalTensor<T_GAMMA> gammaGm;
    GlobalTensor<float> rstdGm;

    uint32_t numRow;
    uint32_t numCol;
    uint32_t blockFactor;
    uint32_t rowFactor;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    int32_t blockIdx_;
    uint32_t rowWork;
};
} // namespace RmsNorm
#endif // _RMS_NORM_SINGLE_ROW_H_
