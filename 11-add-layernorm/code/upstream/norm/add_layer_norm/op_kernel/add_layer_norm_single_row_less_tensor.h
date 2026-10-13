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
 * \file add_layer_norm_single_row_less_tensor.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_SINGLE_ROW_LESS_TENSOR_H
#define ADD_LAYER_NORM_SINGLE_ROW_LESS_TENSOR_H

#include "add_layer_norm_base.h"

template <typename T_X1, typename T_X2, typename T_GAMMA, typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddLayerNormSingleRowLessTensor {
#define IS_ADDITIONAL_OUTPUT_ENABLE ((TILING_KEY % 1000) / 100 == 1)
#define IS_NO_BIAS ((TILING_KEY % 10) == 0)
#define IS_BIAS_PRESENT ((TILING_KEY % 10) == 1)
#define IS_BIAS_BROADCAST ((TILING_KEY % 10) == 2)
#define IS_SINGLE_ROW_LESS_TENSOR_CASE ((TILING_KEY % 100) / 10 == 9)
#define IS_CAST_BEFORE_ADD (!is_same<T_X1, T_X2>::value)
#define IS_X1_NEEDCAST ((!is_same<T_X1, float>::value) && IS_CAST_BEFORE_ADD)
#define IS_X2_NEEDCAST ((!is_same<T_X2, float>::value) && IS_CAST_BEFORE_ADD)
#define IS_GAMMA_FP32 (is_same<T_GAMMA, float>::value)
#define IS_X1_X2_ALL_FP32 ((is_same<T_X1, float>::value) && (is_same<T_X2, float>::value))
#define IS_X_B16_GAMMA_B32 ((!IS_CAST_BEFORE_ADD) && (!is_same<T_X1, float>::value) && (IS_GAMMA_FP32))

public:
    __aicore__ inline KernelAddLayerNormSingleRowLessTensor(TPipe* pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline uint32_t CEIL_DIV(uint32_t x3, uint32_t y)
    {
        if (y > 0) {
            return (x3 + y - 1) / y;
        }
        return 0;
    }

    __aicore__ inline uint32_t ROUND_UP32(uint32_t x)
    {
        return (x + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return x < y ? x : y;
    }

    __aicore__ inline void Init(
        __gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* gamma, __gm__ uint8_t* beta, __gm__ uint8_t* bias,
        __gm__ uint8_t* y, __gm__ uint8_t* mean, __gm__ uint8_t* rstd, __gm__ uint8_t* x, __gm__ uint8_t* workspace,
        uint32_t numCore_, uint32_t numLastDim_, uint32_t numFirstDim_, uint32_t nlFirstDimPerCore_,
        uint32_t lFirstDimPerCore_, uint32_t firstDimPerTime_, uint32_t lastDimPerTime_, float eps_, float aveNum_,
        uint32_t colMoveCnt_, uint32_t colTail_, uint32_t workspace_size)
    {
        numCore = numCore_;
        numLastDim = numLastDim_;
        numFirstDim = numFirstDim_;
        nlFirstDimPerCore = nlFirstDimPerCore_;
        lFirstDimPerCore = lFirstDimPerCore_;
        firstDimPerTime = firstDimPerTime_;
        lastDimPerTime = lastDimPerTime_;
        aveNum = aveNum_;
        eps = eps_;
        colMoveCnt = colMoveCnt_;
        colTail = colTail_;
        if (block_idx != numCore - 1) {
            rowWork = nlFirstDimPerCore;
            rowStep = firstDimPerTime;
        } else {
            rowWork = lFirstDimPerCore;
            rowStep = MIN(firstDimPerTime, rowWork);
        }
        rowTail_ = (rowWork % rowStep == 0) ? rowStep : (rowWork % rowStep);

        InitInputGMBuffer(x1, x2, gamma, beta, bias);
        InitOutputGMBuffer(y, mean, rstd, x);
        workspaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspace_size);

        numLastDimAligned = numLastDim;
        if (ROUND_UP32(numLastDim * sizeof(T)) != numLastDim * sizeof(T)) {
            lastDimPad = true;
            numLastDimAligned = ROUND_UP32(numLastDim * sizeof(T)) / sizeof(T);
        }
        if constexpr (IS_X1_NEEDCAST || IS_X2_NEEDCAST) {
            numLastDimAlignedMixDtype = numLastDim;
            if (ROUND_UP32(numLastDim * sizeof(half)) != numLastDim * sizeof(half)) {
                lastDimPadMixDtype = true;
                numLastDimAlignedMixDtype = ROUND_UP32(numLastDim * sizeof(half)) / sizeof(half);
            }
        }

        InitUBBuffer();
    }

    __aicore__ inline void InitInputGMBuffer(
        __gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* gamma, __gm__ uint8_t* beta, __gm__ uint8_t* bias)
    {
        uint32_t gmOffset_ = nlFirstDimPerCore * numLastDim;
        x1Gm.SetGlobalBuffer((__gm__ T_X1*)(x1) + block_idx * gmOffset_);
        x2Gm.SetGlobalBuffer((__gm__ T_X2*)(x2) + block_idx * gmOffset_);
        if constexpr (IS_BIAS_PRESENT) {
            biasGm.SetGlobalBuffer((__gm__ T*)(bias) + block_idx * gmOffset_);
        } else if constexpr (IS_BIAS_BROADCAST) {
            biasGm.SetGlobalBuffer((__gm__ T*)bias);
        }
        gammaGm.SetGlobalBuffer((__gm__ T_GAMMA*)gamma);
        betaGm.SetGlobalBuffer((__gm__ T_GAMMA*)beta);
    }

    __aicore__ inline void InitOutputGMBuffer(
        __gm__ uint8_t* y, __gm__ uint8_t* mean, __gm__ uint8_t* rstd, __gm__ uint8_t* x)
    {
        uint32_t gmOffset_ = nlFirstDimPerCore * numLastDim;
        yGm.SetGlobalBuffer((__gm__ T*)(y) + block_idx * gmOffset_);
        // mean/rstd always output fp32
        meanGm.SetGlobalBuffer((__gm__ float*)mean + block_idx * nlFirstDimPerCore);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + block_idx * nlFirstDimPerCore);
        xGm.SetGlobalBuffer((__gm__ T*)(x) + block_idx * gmOffset_);
    }

    __aicore__ inline void InitUBBuffer()
    {
        Ppipe->InitBuffer(inputOutputQue, BUFFER_NUM, ROUND_UP32(numLastDim * sizeof(T)));
        if constexpr (IS_X_B16_GAMMA_B32) {
            Ppipe->InitBuffer(tmpQueFp32, BUFFER_NUM, ROUND_UP32(numLastDim * sizeof(float)));
        }
        Ppipe->InitBuffer(xBufFp32, ROUND_UP32(numLastDim * sizeof(float)));
        Ppipe->InitBuffer(yBufFp32, ROUND_UP32(numLastDim * sizeof(float)));
#if OUTPUT_MEAN_RSTD == 1
        Ppipe->InitBuffer(meanQue, BUFFER_NUM, ROUND_UP32(rowStep * sizeof(float)));
        Ppipe->InitBuffer(rstdQue, BUFFER_NUM, ROUND_UP32(rowStep * sizeof(float)));
#endif
    }

    __aicore__ inline void Process()
    {
        int32_t rowMoveCnt = CEIL_DIV(rowWork, rowStep);

        for (int32_t rowIdx = 0; rowIdx < rowMoveCnt; ++rowIdx) {
            uint32_t gmOffset = rowIdx * rowStep * numLastDim;
            CopyInAdd(gmOffset, numLastDim);
            if constexpr (IS_ADDITIONAL_OUTPUT_ENABLE) {
                CopyOutX(gmOffset, numLastDim);
            }
            CopyInGammaOneRow();
            ComputeFirstPart(); // compute mean rstd and part of y
            CopyInBetaOneRow();
            ComputeSecondPart(); // compute y
            CopyOut(rowIdx, 1);
        }
    }

private:
    template <typename T_NOCAST, typename T_NEEDCAST>
    __aicore__ inline void CopyInAddWithCast(
        GlobalTensor<T_NOCAST>& xNoCastGm, GlobalTensor<T_NEEDCAST>& xNeedCastGm, uint32_t gmOffset, uint32_t size)
    {
        event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        auto xBufLocal = xBufFp32.Get<float>();
        auto yBufLocal = yBufFp32.Get<float>();

        // 1. x1/x2 datacopy to ub together and cast
        LocalTensor<T_NOCAST> xNoCastLocalIn = inputOutputQue.template AllocTensor<T_NOCAST>();
        auto tmpLocal = xBufLocal.template ReinterpretCast<T_NEEDCAST>();
        DataCopyEx(tmpLocal, xNeedCastGm[gmOffset], size);

        SetFlag<HardEvent::MTE2_V>(eventMTE2V);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V);
        Cast(yBufLocal, tmpLocal, RoundMode::CAST_NONE, size); // cast together with MTE2
        DataCopyEx(xNoCastLocalIn, xNoCastGm[gmOffset], size);
        inputOutputQue.EnQue(xNoCastLocalIn);

        // 2. add x1x2
        LocalTensor<T_NOCAST> xNoCastLocal = inputOutputQue.template DeQue<T_NOCAST>();
        PipeBarrier<PIPE_V>();
        Add(xBufLocal, yBufLocal, xNoCastLocal, size);
        inputOutputQue.FreeTensor(xNoCastLocal);
    }

    __aicore__ inline void CopyInAddWithoutCast(uint32_t gmOffset, uint32_t size)
    {
        auto xBufLocal = xBufFp32.Get<float>();

        // 1. x1/x2 datacopy to ub together
        LocalTensor<T> xLocalIn = inputOutputQue.template AllocTensor<T>();
        DataCopyEx(xLocalIn, x1Gm[gmOffset], size);
        DataCopyEx(xBufLocal, x2Gm[gmOffset], size);
        inputOutputQue.EnQue(xLocalIn);

        // 2. add x1x2
        LocalTensor<T> xLocal = inputOutputQue.template DeQue<T>();
        Add(xBufLocal, xBufLocal, xLocal, size);
        inputOutputQue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyInAddAllCast(uint32_t gmOffset, uint32_t size)
    {
        event_t eventMTE3MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

        auto xBufLocal = xBufFp32.Get<float>();

        // 1. x1/x2 datacopy to ub together
        LocalTensor<T> inputLocalIn = inputOutputQue.template AllocTensor<T>();
        LocalTensor<float> tmpLocalIn = tmpQueFp32.template AllocTensor<float>();
        auto tmpLocalInHalf = tmpLocalIn.ReinterpretCast<T>();
        DataCopyEx(inputLocalIn, x1Gm[gmOffset], size);
        DataCopyEx(tmpLocalInHalf, x2Gm[gmOffset], size);
        inputOutputQue.EnQue(inputLocalIn);
        tmpQueFp32.EnQue(tmpLocalIn);

        // 2. fp32 add x1/x2
        LocalTensor<T> inputLocal = inputOutputQue.template DeQue<T>();
        LocalTensor<float> tmpLocal = tmpQueFp32.template DeQue<float>();
        auto tmpLocalHalf = tmpLocal.ReinterpretCast<T>();
        Cast(xBufLocal, tmpLocalHalf, RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        Cast(tmpLocal, inputLocal, RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        Add(xBufLocal, xBufLocal, tmpLocal, size);
        PipeBarrier<PIPE_V>();

        // 3. cast x_sum
        if constexpr (is_same<T, half>::value) {
            Cast(inputLocal, xBufLocal, RoundMode::CAST_NONE, size);
        } else {
            Cast(inputLocal, xBufLocal, RoundMode::CAST_RINT, size);
        }
        PipeBarrier<PIPE_V>();
        inputOutputQue.EnQue(inputLocal);
        tmpQueFp32.FreeTensor(tmpLocal);
    }

    __aicore__ inline void CopyInAddBiasAllCast(uint32_t gmOffset, uint32_t size)
    {
        event_t eventMTE3MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

        auto xBufLocal = xBufFp32.Get<float>();
        auto yBufLocal = yBufFp32.Get<float>();
        auto xLocalHalf = xBufLocal.ReinterpretCast<T>();
        auto yLocalHalf = yBufLocal.ReinterpretCast<T>();

        // 1. x2/bias datacopy to ub together
        LocalTensor<T> inputLocalIn = inputOutputQue.template AllocTensor<T>();
        LocalTensor<float> tmpLocalIn = tmpQueFp32.template AllocTensor<float>();
        auto tmpLocalInHalf = tmpLocalIn.ReinterpretCast<T>();
        DataCopyEx(inputLocalIn, biasGm, size);
        DataCopyEx(tmpLocalInHalf, x2Gm[gmOffset], size);
        inputOutputQue.EnQue(inputLocalIn);
        tmpQueFp32.EnQue(tmpLocalIn);

        // 2. fp32 add x2/bias
        LocalTensor<T> inputLocal = inputOutputQue.template DeQue<T>();
        LocalTensor<float> tmpLocal = tmpQueFp32.template DeQue<float>();
        auto tmpLocalHalf = tmpLocal.ReinterpretCast<T>();
        Cast(xBufLocal, tmpLocalHalf, RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        Cast(tmpLocal, inputLocal, RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        Add(xBufLocal, xBufLocal, tmpLocal, size);
        PipeBarrier<PIPE_V>();
        inputOutputQue.FreeTensor(inputLocal);
        tmpQueFp32.FreeTensor(tmpLocal);

        // 3. x1 datacopy to ub
        SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
        DataCopyEx(yLocalHalf, x1Gm[gmOffset], size);
        SetFlag<HardEvent::MTE2_V>(eventMTE2V);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V);

        // 4. fp32 add x1 to x2/bias
        LocalTensor<T> xOutLocal = inputOutputQue.template AllocTensor<T>();
        LocalTensor<float> tmpLocalIn2 = tmpQueFp32.template AllocTensor<float>();
        tmpQueFp32.EnQue(tmpLocalIn2);
        LocalTensor<float> tmpLocal2 = tmpQueFp32.template DeQue<float>();
        Cast(tmpLocal2, yLocalHalf, RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        Add(xBufLocal, xBufLocal, tmpLocal2, size);
        PipeBarrier<PIPE_V>();
        if constexpr (is_same<T, half>::value) {
            Cast(xOutLocal, xBufLocal, RoundMode::CAST_NONE, size);
        } else {
            Cast(xOutLocal, xBufLocal, RoundMode::CAST_RINT, size);
        }
        PipeBarrier<PIPE_V>();
        inputOutputQue.EnQue(xOutLocal);
    }

    __aicore__ inline void CopyInAdd(uint32_t gmOffset, uint32_t size)
    {
        if constexpr (IS_X1_NEEDCAST) {
            CopyInAddWithCast<T_X2, T_X1>(x2Gm, x1Gm, gmOffset, size);
        } else if constexpr (IS_X2_NEEDCAST) {
            CopyInAddWithCast<T_X1, T_X2>(x1Gm, x2Gm, gmOffset, size);
        } else if constexpr (IS_X1_X2_ALL_FP32) {
            CopyInAddWithoutCast(gmOffset, size);
        } else if constexpr (IS_X_B16_GAMMA_B32 && IS_BIAS_BROADCAST) {
            CopyInAddBiasAllCast(gmOffset, size);
        } else if constexpr (IS_X_B16_GAMMA_B32 && IS_NO_BIAS) {
            CopyInAddAllCast(gmOffset, size);
        }
    }

    __aicore__ inline void CopyOutX(uint32_t gmOffset, uint32_t size)
    {
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

        if constexpr (is_same<T, float>::value) {
            auto addBufLocal = xBufFp32.Get<float>();
            SetFlag<HardEvent::V_MTE3>(eventVMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
            DataCopyEx(xGm[gmOffset], addBufLocal, size);
        } else {
            event_t eventMTE3MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            LocalTensor<float> xOutLocal = inputOutputQue.template DeQue<float>();
            auto xOutLocalHalf = xOutLocal.ReinterpretCast<T>();
            SetFlag<HardEvent::V_MTE3>(eventVMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
            DataCopyEx(xGm[gmOffset], xOutLocalHalf, size);
            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
            inputOutputQue.FreeTensor(xOutLocal);
        }
    }

    __aicore__ inline void CopyInGammaOneRow()
    {
        if constexpr (is_same<T, float>::value) { // T_GAMMA and T is float
            LocalTensor<T> gammaLocal = inputOutputQue.template AllocTensor<T>();
            DataCopyEx(gammaLocal, gammaGm, numLastDim);
            inputOutputQue.EnQue(gammaLocal);
        } else if constexpr (IS_X_B16_GAMMA_B32) { // T_GAMMA is float not equal T
            LocalTensor<float> gammaLocalIn = tmpQueFp32.template AllocTensor<float>();
            DataCopyEx(gammaLocalIn, gammaGm, numLastDim);
            tmpQueFp32.EnQue(gammaLocalIn);
        }
    }

    __aicore__ inline void CopyInBetaOneRow()
    {
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        LocalTensor<T_GAMMA> betaLocal = yBufFp32.Get<float>();

        SetFlag<HardEvent::V_MTE2>(eventVMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
        DataCopyEx(betaLocal, betaGm, numLastDim);
    }

    __aicore__ inline void CopyOut(int32_t rowIdx, int32_t row_count)
    {
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));

        auto yLocalFp32 = yBufFp32.Get<float>();
        uint32_t gmOffset = rowIdx * rowStep * numLastDim;

        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        if constexpr (is_same<T, float>::value) {
            DataCopyEx(yGm[gmOffset], yLocalFp32, numLastDim, row_count);
        } else if constexpr (IS_X_B16_GAMMA_B32) {
            auto yLocalFp32Half = yLocalFp32.ReinterpretCast<T>();
            DataCopyEx(yGm[gmOffset], yLocalFp32Half, numLastDim, row_count);
        }
        SetFlag<HardEvent::MTE3_V>(eventMTE3V);
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V);

#if OUTPUT_MEAN_RSTD == 1
        uint32_t gm_offset_mean = rowIdx * rowStep;
        LocalTensor<float> mean = meanQue.template DeQue<float>();
        LocalTensor<float> rstd = rstdQue.template DeQue<float>();
        DataCopyEx(meanGm[gm_offset_mean], mean, row_count);
        DataCopyEx(rstdGm[gm_offset_mean], rstd, row_count);
        meanQue.FreeTensor(mean);
        rstdQue.FreeTensor(rstd);
#endif
    }

    __aicore__ inline void ComputeFirstPart()
    {
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));

#if OUTPUT_MEAN_RSTD == 1
        LocalTensor<float> meanLocal = meanQue.template AllocTensor<float>();
        LocalTensor<float> rstdLocal = rstdQue.template AllocTensor<float>();
#endif
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();

        // 1. mean process: 1/N * x_sum
        Muls(yLocalFp32, xLocalFp32, aveNum, numLastDim);
        PipeBarrier<PIPE_V>();
        // 2. mean end: reduce(1/N * x_sum)
        float meanLocalTemp = ReduceSumFP32(yLocalFp32, numLastDim);
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);

        // 3. rstd process: x - mean
        Adds(yLocalFp32, xLocalFp32, meanLocalTemp * -1, numLastDim);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::MTE3_V>(eventMTE3V); // need make sure xout MTE3 finish.
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
        // 4. rstd process: (x - mean) ^ 2
        Mul(xLocalFp32, yLocalFp32, yLocalFp32, numLastDim);
        PipeBarrier<PIPE_V>();
        // 5. rstd process: reduce( 1 / N * (x - mean) ^ 2 )
        Muls(xLocalFp32, xLocalFp32, aveNum, numLastDim);
        PipeBarrier<PIPE_V>();
        float varLocalTemp = ReduceSumFP32(xLocalFp32, numLastDim);
        // 6. rstd end: 1 / sqrt( 1 / N * reduce( (x - mean) ^ 2 ) )
        float rstdLocalTemp = 1 / sqrt(varLocalTemp + eps);

#if OUTPUT_MEAN_RSTD == 1
        meanLocal.SetValue(0, meanLocalTemp);
        rstdLocal.SetValue(0, rstdLocalTemp);
#endif
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        // 7. y process: (x - mean) / rstd
        Muls(xLocalFp32, yLocalFp32, rstdLocalTemp, numLastDim);
        PipeBarrier<PIPE_V>();

#if OUTPUT_MEAN_RSTD == 1
        meanQue.EnQue(meanLocal);
        rstdQue.EnQue(rstdLocal);
#endif
    }

    __aicore__ inline void ComputeSecondPart()
    {
        event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));

        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> gammaLocal;
        if constexpr (is_same<T, float>::value) {
            gammaLocal = inputOutputQue.template DeQue<T>();
        } else if constexpr (IS_X_B16_GAMMA_B32) {
            gammaLocal = tmpQueFp32.template DeQue<float>();
        }
        auto yLocalFp32 = yBufFp32.Get<float>();

        Mul(xLocalFp32, xLocalFp32, gammaLocal, numLastDim);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::MTE2_V>(eventMTE2V); // unuse deque, need make sure MTE2 finish.
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V);
        if constexpr (is_same<T, float>::value) {
            Add(yLocalFp32, xLocalFp32, yLocalFp32, numLastDim);
        } else if constexpr (is_same<T, half>::value) {
            Add(xLocalFp32, xLocalFp32, yLocalFp32, numLastDim);
            PipeBarrier<PIPE_V>();
            Cast(yLocalFp32.ReinterpretCast<T>(), xLocalFp32, RoundMode::CAST_NONE, numLastDim);
        } else {
            Add(xLocalFp32, xLocalFp32, yLocalFp32, numLastDim);
            PipeBarrier<PIPE_V>();
            Cast(yLocalFp32.ReinterpretCast<T>(), xLocalFp32, RoundMode::CAST_RINT, numLastDim);
        }
        PipeBarrier<PIPE_V>();

        if constexpr (IS_X_B16_GAMMA_B32) {
            tmpQueFp32.FreeTensor(gammaLocal);
        } else {
            inputOutputQue.FreeTensor(gammaLocal);
        }
    }

private:
    TPipe* Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputOutputQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> tmpQueFp32;
#if OUTPUT_MEAN_RSTD == 1
    TQue<QuePosition::VECOUT, BUFFER_NUM> meanQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rstdQue;
#endif
    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;
    GlobalTensor<T_X1> x1Gm;
    GlobalTensor<T_X2> x2Gm;
    GlobalTensor<T_GAMMA> gammaGm;
    GlobalTensor<T_GAMMA> betaGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> xGm;
    GlobalTensor<float> meanGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<float> workspaceGm;
    uint32_t numCore;
    uint32_t numFirstDim;
    uint32_t numLastDim;
    uint32_t rowStep;
    uint32_t rowWork;
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
    bool lastDimPadMixDtype = false;
    size_t numLastDimAlignedMixDtype;
};

#endif
