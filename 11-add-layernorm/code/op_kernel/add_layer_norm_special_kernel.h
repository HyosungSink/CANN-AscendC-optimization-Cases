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
 * \file add_layer_norm_special_kernel.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_SPECIAL_KERNEL_H_
#define ADD_LAYER_NORM_SPECIAL_KERNEL_H_

#include "add_layer_norm_base.h"

template <typename T_X1, typename T_X2, typename T_GAMMA, typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddLayerNormBetterUB {
#define IS_ADDITIONAL_OUTPUT_ENABLE ((TILING_KEY % 1000) / 100 == 1)
#define IS_BIAS_PRESENT ((TILING_KEY % 10) == 1)
#define IS_BIAS_BROADCAST ((TILING_KEY % 10) == 2)

public:
    __aicore__ inline KernelAddLayerNormBetterUB(TPipe* pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline uint32_t CEIL_DIV(uint32_t x1, uint32_t y)
    {
        if (y > 0) {
            return (x1 + y - 1) / y;
        }
        return 0;
    }

    __aicore__ inline uint32_t ROUND_UP32(uint32_t x)
    {
        return (x + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }

    __aicore__ inline uint32_t MAX(uint32_t x, uint32_t y)
    {
        return x > y ? x : y;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return x < y ? x : y;
    }

    __aicore__ inline void InitVar(
        uint32_t num_core_, uint32_t num_Last_dim_, uint32_t num_first_dim_, uint32_t nl_first_dim_per_core_,
        uint32_t l_first_dim_per_core_, uint32_t first_dim_per_time_, uint32_t last_dim_per_time_, float eps_,
        float aveNum_, uint32_t col_move_cnt_, uint32_t col_tail_)
    {
        numCore = num_core_;
        numLastDim = num_Last_dim_;
        numFirstDim = num_first_dim_;
        nlFirstDimPerCore = nl_first_dim_per_core_;
        lFirstDimPerCore = l_first_dim_per_core_;
        firstDimPerTime = first_dim_per_time_;
        lastDimPerTime = last_dim_per_time_;
        aveNum = aveNum_;
        eps = eps_;
        colMoveCnt = col_move_cnt_;
        colTail = col_tail_;
    }

    __aicore__ inline void Init(
        __gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* gamma, __gm__ uint8_t* beta, __gm__ uint8_t* bias,
        __gm__ uint8_t* y, __gm__ uint8_t* mean, __gm__ uint8_t* rstd, __gm__ uint8_t* x, __gm__ uint8_t* workspace,
        uint32_t num_core_, uint32_t num_Last_dim_, uint32_t num_first_dim_, uint32_t nl_first_dim_per_core_,
        uint32_t l_first_dim_per_core_, uint32_t first_dim_per_time_, uint32_t last_dim_per_time_, float eps_,
        float aveNum_, uint32_t col_move_cnt_, uint32_t col_tail_, uint32_t workspace_size)
    {
        InitVar(
            num_core_, num_Last_dim_, num_first_dim_, nl_first_dim_per_core_, l_first_dim_per_core_,
            first_dim_per_time_, last_dim_per_time_, eps_, aveNum_, col_move_cnt_, col_tail_);
        if (block_idx != numCore - 1) {
            rowWork = nlFirstDimPerCore;
            rowStep = firstDimPerTime;
        } else {
            rowWork = lFirstDimPerCore;
            rowStep = MIN(firstDimPerTime, rowWork);
        }
        rowTail_ = (rowWork % rowStep == 0) ? rowStep : (rowWork % rowStep);
        gmOffset_ = nlFirstDimPerCore * numLastDim;
        x1Gm.SetGlobalBuffer((__gm__ T*)(x1) + block_idx * gmOffset_);
        x2Gm.SetGlobalBuffer((__gm__ T*)(x2) + block_idx * gmOffset_);
        if constexpr (IS_BIAS_PRESENT) {
            biasGm.SetGlobalBuffer((__gm__ T*)(bias) + block_idx * gmOffset_);
        } else if constexpr (IS_BIAS_BROADCAST) {
            biasGm.SetGlobalBuffer((__gm__ T*)bias);
        }
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma);
        betaGm.SetGlobalBuffer((__gm__ T*)beta);
        yGm.SetGlobalBuffer((__gm__ T*)(y) + block_idx * gmOffset_);
        // mean/rstd always output fp32
        meanGm.SetGlobalBuffer((__gm__ float*)mean + block_idx * nlFirstDimPerCore);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + block_idx * nlFirstDimPerCore);
        xGm.SetGlobalBuffer((__gm__ T*)(x) + block_idx * gmOffset_);

        numLastDimAligned = numLastDim;
        if (ROUND_UP32(numLastDim * sizeof(T)) != numLastDim * sizeof(T)) {
            lastDimPad = true;
            numLastDimAligned = ROUND_UP32(numLastDim * sizeof(T)) / sizeof(T);
        }

        Ppipe->InitBuffer(x1x2Que, BUFFER_NUM, ROUND_UP32(2 * rowStep * numLastDimAligned * sizeof(T)));
        Ppipe->InitBuffer(yQue, BUFFER_NUM, ROUND_UP32(rowStep * numLastDimAligned * sizeof(T)));

        Ppipe->InitBuffer(betaBuf, ROUND_UP32(numLastDimAligned * sizeof(float)));
        Ppipe->InitBuffer(gammaBuf, ROUND_UP32(numLastDimAligned * sizeof(float)));

        Ppipe->InitBuffer(xBufFp32, ROUND_UP32(rowStep * numLastDimAligned * sizeof(float)));
        Ppipe->InitBuffer(yBufFp32, ROUND_UP32(rowStep * numLastDimAligned * sizeof(float)));

        if constexpr (IS_BIAS_BROADCAST) {
            Ppipe->InitBuffer(biasBuf, ROUND_UP32(numLastDim * sizeof(T)));
        }

#if OUTPUT_MEAN_RSTD == 1
        Ppipe->InitBuffer(rstdQue, BUFFER_NUM, ROUND_UP32(rowStep * sizeof(float)));
        Ppipe->InitBuffer(meanQue, BUFFER_NUM, ROUND_UP32(rowStep * sizeof(float)));
#endif
    }

    __aicore__ inline void Process()
    {
        DataCopyPadParams padParams;
        int32_t rowMoveCnt = CEIL_DIV(rowWork, rowStep);

        if (lastDimPad) {
            padParams.isPad = true;
            padParams.paddingValue = 0;
            padParams.rightPadding = numLastDimAligned - numLastDim;
        }

        LocalTensor<float> betaLocal = betaBuf.template Get<float>();
        LocalTensor<float> gammaLocal = gammaBuf.template Get<float>();

        if constexpr (is_same<float, T>::value) {
            DataCopyEx(betaLocal, betaGm, numLastDim);
            DataCopyEx(gammaLocal, gammaGm, numLastDim);
        } else {
            auto betaLocalHalf = betaLocal.ReinterpretCast<T>();
            auto gammaLocalHalf = gammaLocal.ReinterpretCast<T>();
            DataCopyEx(betaLocalHalf[numLastDimAligned], betaGm, numLastDim);
            DataCopyEx(gammaLocalHalf[numLastDimAligned], gammaGm, numLastDim);
        }

        LocalTensor<T> biasLocal;
        if constexpr (IS_BIAS_BROADCAST) {
            biasLocal = biasBuf.template Get<T>();
            DataCopyEx(biasLocal, biasGm, numLastDim);
        }

        uint32_t gmOffset = 0;
        auto elementCount = numLastDimAligned * rowStep;

        {
            LocalTensor<T> x1x2LocalIn = x1x2Que.template AllocTensor<T>();
            DataCopyEx(x1x2LocalIn[0], x1Gm[gmOffset], numLastDim, rowStep, padParams);
            DataCopyEx(x1x2LocalIn[elementCount], x2Gm[gmOffset], numLastDim, rowStep, padParams);
            x1x2Que.EnQue(x1x2LocalIn);
            auto x1x2Local = x1x2Que.template DeQue<T>();

            if constexpr (!is_same<T, float>::value) {
                Cast(gammaLocal, gammaLocal.ReinterpretCast<T>()[numLastDimAligned], RoundMode::CAST_NONE, numLastDim);
                Cast(betaLocal, betaLocal.ReinterpretCast<T>()[numLastDimAligned], RoundMode::CAST_NONE, numLastDim);
            }

            if constexpr (IS_BIAS_BROADCAST) {
                CopyInAndAddBroadCast(0, rowStep, biasLocal, x1x2Local, elementCount);
            }
            CopyOutAdditionalOutput(0, rowStep);
            precisionCompute(rowStep, gammaLocal, betaLocal, x1x2Local, elementCount);
            CopyOut(0, rowStep);
            gmOffset += rowStep * numLastDim;
        }
        for (int32_t rowIdx = 1; rowIdx < rowMoveCnt - 1; ++rowIdx) {
            LocalTensor<T> x1x2LocalIn = x1x2Que.template AllocTensor<T>();
            DataCopyEx(x1x2LocalIn[0], x1Gm[gmOffset], numLastDim, rowStep, padParams);
            DataCopyEx(x1x2LocalIn[elementCount], x2Gm[gmOffset], numLastDim, rowStep, padParams);
            x1x2Que.EnQue(x1x2LocalIn);
            auto x1x2Local = x1x2Que.template DeQue<T>();

            if constexpr (IS_BIAS_BROADCAST) {
                CopyInAndAddBroadCast(rowIdx, rowStep, biasLocal, x1x2Local, elementCount);
            }
            CopyOutAdditionalOutput(rowIdx, rowStep);
            precisionCompute(rowStep, gammaLocal, betaLocal, x1x2Local, elementCount);
            CopyOut(rowIdx, rowStep);
            gmOffset += rowStep * numLastDim;
        }
        {
            auto rowIdx = rowMoveCnt - 1;
            if (rowIdx > 0) {
                elementCount = numLastDimAligned * rowTail_;

                LocalTensor<T> x1x2LocalIn = x1x2Que.template AllocTensor<T>();
                DataCopyEx(x1x2LocalIn[0], x1Gm[gmOffset], numLastDim, rowTail_, padParams);
                DataCopyEx(x1x2LocalIn[elementCount], x2Gm[gmOffset], numLastDim, rowTail_, padParams);
                x1x2Que.EnQue(x1x2LocalIn);
                auto x1x2Local = x1x2Que.template DeQue<T>();

                if constexpr (IS_BIAS_BROADCAST) {
                    CopyInAndAddBroadCast(rowIdx, rowTail_, biasLocal, x1x2Local, elementCount);
                }
                CopyOutAdditionalOutput(rowIdx, rowTail_);
                precisionCompute(rowTail_, gammaLocal, betaLocal, x1x2Local, elementCount);
                CopyOut(rowIdx, rowTail_);
            }
        }
    }

private:
    __aicore__ inline void CopyInAndAddBroadCast(
        int32_t procId, int32_t rowCount, LocalTensor<T>& biasLocal, LocalTensor<T>& x1x2Local, uint32_t elementCount)
    {
        LocalTensor<float> addBufLocal = xBufFp32.Get<float>();
        LocalTensor<float> yBufLocalSpecial = yBufFp32.Get<float>();

        auto x1LocalSpecial = x1x2Local[0];
        auto x2LocalSpecial = x1x2Local[elementCount];

        // Use add as
        if constexpr (is_same<float, T>::value) {
            Add(addBufLocal, x2LocalSpecial, x1LocalSpecial, elementCount);
            PipeBarrier<PIPE_V>();
            for (int i = 0; i < rowCount; i++) {
                Add(addBufLocal[i * numLastDimAligned], biasLocal, addBufLocal[i * numLastDimAligned], numLastDim);
            }
            PipeBarrier<PIPE_V>();
        } else {
            Cast(addBufLocal, x1LocalSpecial, RoundMode::CAST_NONE, elementCount);
            Cast(yBufLocalSpecial, x2LocalSpecial, RoundMode::CAST_NONE, elementCount);
            PipeBarrier<PIPE_V>();
            Add(yBufLocalSpecial, addBufLocal, yBufLocalSpecial, elementCount);
            Cast(x1x2Local.template ReinterpretCast<float>(), biasLocal, RoundMode::CAST_NONE, numLastDim);
            PipeBarrier<PIPE_V>();
            for (int i = 0; i < rowCount; i++) {
                Add(addBufLocal[i * numLastDimAligned], x1x2Local.template ReinterpretCast<float>(),
                    yBufLocalSpecial[i * numLastDimAligned], numLastDim);
            }
            PipeBarrier<PIPE_V>();
        }
        x1x2Que.FreeTensor(x1x2Local);
    }

    __aicore__ inline void precisionCompute(
        int32_t nums, LocalTensor<float>& gammaLocal, LocalTensor<float>& betaLocal, LocalTensor<T>& x_out,
        uint32_t elementCount)
    {
#if OUTPUT_MEAN_RSTD == 1
        LocalTensor<float> meanLocal = meanQue.template AllocTensor<float>();
        LocalTensor<float> rstdLocal = rstdQue.template AllocTensor<float>();
#endif
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();

        Muls(yLocalFp32, xLocalFp32, aveNum, elementCount);
        PipeBarrier<PIPE_V>();

        // Reduce#1 for E(x)
        for (int32_t rid = 0; rid < nums; ++rid) {
            auto aveLocalTemp = ReduceSumFP32(yLocalFp32[rid * numLastDimAligned], numLastDim);
#if OUTPUT_MEAN_RSTD == 1
            meanLocal.SetValue(rid, aveLocalTemp);
#endif
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Adds(
                yLocalFp32[rid * numLastDimAligned], xLocalFp32[rid * numLastDimAligned], aveLocalTemp * -1,
                numLastDim);
        }
        PipeBarrier<PIPE_V>();

        Mul(xLocalFp32, yLocalFp32, yLocalFp32, elementCount);
        PipeBarrier<PIPE_V>();
        Muls(xLocalFp32, xLocalFp32, aveNum, elementCount);
        PipeBarrier<PIPE_V>();

        // Reduce#2 for Var(x)
        for (int32_t rid = 0; rid < nums; ++rid) {
            float varLocalTemp = ReduceSumFP32(xLocalFp32[rid * numLastDimAligned], numLastDim);
            float rstdLocalTemp = 1 / sqrt(varLocalTemp + eps);
#if OUTPUT_MEAN_RSTD == 1
            rstdLocal.SetValue(rid, rstdLocalTemp);
#endif
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Muls(xLocalFp32[rid * numLastDimAligned], yLocalFp32[rid * numLastDimAligned], rstdLocalTemp, numLastDim);
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<T> yLocal = yQue.template AllocTensor<T>();

        if constexpr (!is_same<T, float>::value) {
            for (int32_t rid = 0; rid < nums; ++rid) {
                FusedMulAdd(xLocalFp32[rid * numLastDimAligned], gammaLocal, betaLocal, numLastDim);
            }
            PipeBarrier<PIPE_V>();
            if constexpr (is_same<T, half>::value) {
                Cast(yLocal, xLocalFp32, RoundMode::CAST_NONE, elementCount);
            } else {
                Cast(yLocal, xLocalFp32, RoundMode::CAST_RINT, elementCount);
            }
        } else {
            for (int32_t rid = 0; rid < nums; ++rid) {
                FusedMulAdd(xLocalFp32[rid * numLastDimAligned], gammaLocal, betaLocal, numLastDim);
            }
            PipeBarrier<PIPE_V>();
            Adds(yLocal, xLocalFp32, (float)0.0, elementCount);
        }

#if OUTPUT_MEAN_RSTD == 1
        meanQue.EnQue(meanLocal);
        rstdQue.EnQue(rstdLocal);
#endif
        yQue.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t rowIdx, int32_t rowCount)
    {
        LocalTensor<T> res = yQue.template DeQue<T>();
        uint32_t gmOffset = rowIdx * rowStep * numLastDim;
        DataCopyEx(yGm[gmOffset], res, numLastDim, rowCount);
        yQue.FreeTensor(res);

#if OUTPUT_MEAN_RSTD == 1
        uint32_t gmOffsetMean = rowIdx * rowStep;
        LocalTensor<float> mean = meanQue.template DeQue<float>();
        LocalTensor<float> rstd = rstdQue.template DeQue<float>();
        DataCopyEx(meanGm[gmOffsetMean], mean, rowCount);
        DataCopyEx(rstdGm[gmOffsetMean], rstd, rowCount);
        meanQue.FreeTensor(mean);
        rstdQue.FreeTensor(rstd);
#endif
    }

    __aicore__ inline void CopyOutAdditionalOutput(int32_t procId, int32_t rowCount)
    {
        if constexpr (IS_ADDITIONAL_OUTPUT_ENABLE) {
            LocalTensor<float> addBufLocalSpecial = xBufFp32.Get<float>();
            uint32_t gmOffsetSpecial = procId * rowStep * numLastDim;
            auto elementCountSpecial = numLastDimAligned * rowCount;
            auto xLocalSpecial = yQue.template AllocTensor<T>();
            if constexpr (is_same<T, float>::value) {
                Adds(xLocalSpecial, addBufLocalSpecial, ZERO, elementCountSpecial);
            } else if constexpr (is_same<T, half>::value) {
                Cast(xLocalSpecial, addBufLocalSpecial, RoundMode::CAST_NONE, elementCountSpecial);
            } else {
                Cast(xLocalSpecial, addBufLocalSpecial, RoundMode::CAST_RINT, elementCountSpecial);
            }
            PipeBarrier<PIPE_V>();
            yQue.template EnQue<T>(xLocalSpecial);
            auto xSpecial = yQue.template DeQue<T>();

            DataCopyEx(xGm[gmOffsetSpecial], xSpecial, numLastDim, rowCount);
            yQue.FreeTensor(xSpecial);
        }
    }

private:
    TPipe* Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1x2Que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQue;
#if OUTPUT_MEAN_RSTD == 1
    TQue<QuePosition::VECOUT, BUFFER_NUM> meanQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rstdQue;
#endif
    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;

    TBuf<TPosition::VECCALC> gammaBuf;
    TBuf<TPosition::VECCALC> betaBuf;
    TBuf<TPosition::VECCALC> biasBuf;

    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> betaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<float> meanGm;
    GlobalTensor<float> rstdGm;
    uint32_t numCore;
    uint32_t numFirstDim;
    uint32_t numLastDim;
    uint32_t rowStep;
    uint32_t rowWork;
    uint32_t gmOffset_;
    uint32_t rowTail_;
    uint32_t colTail;
    uint32_t colMoveCnt;
    uint32_t firstDimPerTime;
    uint32_t lastDimPerTime;
    uint32_t nlFirstDimPerCore;
    uint32_t lFirstDimPerCore;
    float eps;
    float aveNum;
    bool lastDimPad = false;
    size_t numLastDimAligned;
};

#endif // __ADD_LAYER_NORM_SPECIAL_KERNEL_H_