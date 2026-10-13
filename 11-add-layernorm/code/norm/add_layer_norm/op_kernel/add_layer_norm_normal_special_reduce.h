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
 * \file add_layer_norm_normal_special_reduce.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_NORMAL_SPECIAL_REDUCE_H_
#define ADD_LAYER_NORM_NORMAL_SPECIAL_REDUCE_H_

#include "add_layer_norm_base.h"

template <typename T_X1, typename T_X2, typename T_GAMMA, typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddLayerNormNormalSpecialReduce {
#define IS_ADDITIONAL_OUTPUT_ENABLE ((TILING_KEY % 1000) / 100 == 1)
#define IS_NORMAL_SPECIAL_REDUCE_BIG_N_CASE ((TILING_KEY % 100) / 10 == 8)
#define IS_BIAS_BROADCAST ((TILING_KEY % 10) == 2)

public:
    __aicore__ inline KernelAddLayerNormNormalSpecialReduce(TPipe* pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline uint32_t ROUND_UP32(uint32_t x)
    {
        return (x + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }

    __aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
    {
        if (y > 0) {
            return (x + y - 1) / y;
        }
        return 0;
    }

    __aicore__ inline uint32_t BlockAlign(uint32_t x, uint32_t blockElem)
    {
        if (blockElem > 0) {
            return (x + blockElem - 1) / blockElem * blockElem;
        }
        return 0;
    }

    __aicore__ inline uint32_t MIN(uint32_t a, uint32_t b)
    {
        return a < b ? a : b;
    }

    __aicore__ inline uint32_t MAX(uint32_t a, uint32_t b)
    {
        return a > b ? a : b;
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
        notLastFirstDimPerCore = nlFirstDimPerCore_;
        lFirstDimPerCore = lFirstDimPerCore_;
        firstDimPerTime = firstDimPerTime_;
        lastDimPerTime = lastDimPerTime_;
        aveNum = aveNum_;
        eps = eps_;
        colMoveCnt = colMoveCnt_;
        colTail = colTail_;
        if (block_idx != numCore - 1) {
            rowWork = notLastFirstDimPerCore;
            rowStep = firstDimPerTime;
        } else {
            rowWork = lFirstDimPerCore;
            rowStep = MIN(firstDimPerTime, rowWork);
        }
        rowTail_ = (rowWork % rowStep == 0) ? rowStep : (rowWork % rowStep);
        gmOffset_ = notLastFirstDimPerCore * numLastDim;
        x1Gm.SetGlobalBuffer((__gm__ T*)(x1) + block_idx * gmOffset_);
        x2Gm.SetGlobalBuffer((__gm__ T*)(x2) + block_idx * gmOffset_);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma);
        betaGm.SetGlobalBuffer((__gm__ T*)beta);
        yGm.SetGlobalBuffer((__gm__ T*)(y) + block_idx * gmOffset_);
        // mean/rstd always output fp32
        meanGm.SetGlobalBuffer((__gm__ float*)mean + block_idx * notLastFirstDimPerCore);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + block_idx * notLastFirstDimPerCore);
        xGm.SetGlobalBuffer((__gm__ T*)(x) + block_idx * gmOffset_);
        if constexpr (IS_BIAS_BROADCAST) {
            biasGm.SetGlobalBuffer((__gm__ T*)bias);
        }

        numLastDimAligned = numLastDim;
        if (ROUND_UP32(numLastDim * sizeof(T)) != numLastDim * sizeof(T)) {
            lastDimPad = true;
            numLastDimAligned = ROUND_UP32(numLastDim * sizeof(T)) / sizeof(T);
        }

        Ppipe->InitBuffer(x1x2Que, BUFFER_NUM, ROUND_UP32(2 * rowStep * numLastDimAligned * sizeof(T)));
        Ppipe->InitBuffer(yQue, BUFFER_NUM, ROUND_UP32(rowStep * numLastDimAligned * sizeof(T)));
        Ppipe->InitBuffer(betaBuf, ROUND_UP32(numLastDimAligned * sizeof(float)));
        Ppipe->InitBuffer(gammaBuf, ROUND_UP32(numLastDimAligned * sizeof(float)));
        if constexpr (IS_BIAS_BROADCAST) {
            Ppipe->InitBuffer(biasBuf, ROUND_UP32(numLastDim * sizeof(T)));
        }

        Ppipe->InitBuffer(xBufFp32, ROUND_UP32(rowStep * numLastDimAligned * sizeof(float)));
        Ppipe->InitBuffer(zBufFp32, ROUND_UP32(rowStep * numLastDimAligned * sizeof(float)));
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        uint32_t brcbRowStep = BlockAlign(rowStep, BRCB_BROADCAST_NUMBER);
        Ppipe->InitBuffer(yBufFp32, ROUND_UP32(brcbRowStep * ELEM_PER_REP_FP32 * sizeof(float)));
#else
        Ppipe->InitBuffer(yBufFp32, ROUND_UP32(rowStep * ELEM_PER_REP_FP32 * sizeof(float)));
#endif

#if __CCE_AICORE__ != 220
        Ppipe->InitBuffer(orBufINT16, 16 * sizeof(int16_t)); // one block

        if constexpr (IS_NORMAL_SPECIAL_REDUCE_BIG_N_CASE) {
            Ppipe->InitBuffer(transposeSrcBuf, ROUND_UP32(16 * 16 * 8 * sizeof(half)));
            Ppipe->InitBuffer(transposeDstBuf, ROUND_UP32(16 * 16 * 8 * sizeof(half)));
        }
#endif

#if OUTPUT_MEAN_RSTD == 1
        Ppipe->InitBuffer(meanQue, BUFFER_NUM, ROUND_UP32(rowStep * sizeof(float)));
        Ppipe->InitBuffer(rstdQue, BUFFER_NUM, ROUND_UP32(rowStep * sizeof(float)));
#endif
    }

    __aicore__ inline void Process()
    {
        int32_t rowMoveCnt = CEIL_DIV(rowWork, rowStep);

        DataCopyPadParams padParams;
        if (lastDimPad) {
            padParams.isPad = true;
            padParams.paddingValue = 0;
            padParams.rightPadding = numLastDimAligned - numLastDim;
        }

        LocalTensor<float> betaLocalReduce = betaBuf.template Get<float>();
        LocalTensor<float> gammaLocalReduce = gammaBuf.template Get<float>();

        if constexpr (is_same<float, T>::value) {
            DataCopyEx(betaLocalReduce, betaGm, numLastDim);
            DataCopyEx(gammaLocalReduce, gammaGm, numLastDim);
        } else {
            auto betaLocalHalf = betaLocalReduce.ReinterpretCast<T>();
            auto gammaLocalHalf = gammaLocalReduce.ReinterpretCast<T>();
            DataCopyEx(betaLocalHalf[numLastDimAligned], betaGm, numLastDim);
            DataCopyEx(gammaLocalHalf[numLastDimAligned], gammaGm, numLastDim);
        }

        LocalTensor<T> biasLocal;
        if constexpr (IS_BIAS_BROADCAST) {
            biasLocal = biasBuf.template Get<T>();
            DataCopyEx(biasLocal, biasGm, numLastDim);
        }

        uint32_t gmOffset = 0;
        uint32_t elementCount = numLastDimAligned * rowStep;

        {
            LocalTensor<T> x1x2LocalIn = x1x2Que.template AllocTensor<T>();
            DataCopyEx(x1x2LocalIn[0], x1Gm[gmOffset], numLastDim, rowStep, padParams);
            DataCopyEx(x1x2LocalIn[elementCount], x2Gm[gmOffset], numLastDim, rowStep, padParams);
            x1x2Que.EnQue(x1x2LocalIn);
            auto x1x2Local = x1x2Que.template DeQue<T>();

            if constexpr (!is_same<T, float>::value) {
                Cast(gammaLocalReduce, gammaLocalReduce.ReinterpretCast<T>()[numLastDimAligned], RoundMode::CAST_NONE, numLastDim);
                Cast(betaLocalReduce, betaLocalReduce.ReinterpretCast<T>()[numLastDimAligned], RoundMode::CAST_NONE, numLastDim);
            }

            if constexpr (IS_BIAS_BROADCAST) {
                CopyInAndAddBroadCast(rowStep, elementCount, biasLocal, x1x2Local);
            } else {
                CopyIn(rowStep, elementCount, x1x2Local, padParams);
            }
            CopyOutAdditionalOutput(0, rowStep);
            if constexpr (IS_NORMAL_SPECIAL_REDUCE_BIG_N_CASE) {
                PrecisionComputeBigN(rowStep, gammaLocalReduce, betaLocalReduce);
            } else {
                PrecisionCompute(rowStep, gammaLocalReduce, betaLocalReduce, elementCount);
            }
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
                CopyInAndAddBroadCast(rowStep, elementCount, biasLocal, x1x2Local);
            } else {
                CopyIn(rowStep, elementCount, x1x2Local, padParams);
            }

            CopyOutAdditionalOutput(rowIdx, rowStep);
            if constexpr (IS_NORMAL_SPECIAL_REDUCE_BIG_N_CASE) {
                PrecisionComputeBigN(rowStep, gammaLocalReduce, betaLocalReduce);
            } else {
                PrecisionCompute(rowStep, gammaLocalReduce, betaLocalReduce, elementCount);
            }
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
                    CopyInAndAddBroadCast(rowTail_, elementCount, biasLocal, x1x2Local);
                } else {
                    CopyIn(rowTail_, elementCount, x1x2Local, padParams);
                }

                CopyOutAdditionalOutput(rowIdx, rowTail_);
                if constexpr (IS_NORMAL_SPECIAL_REDUCE_BIG_N_CASE) {
                    PrecisionComputeBigN(rowTail_, gammaLocalReduce, betaLocalReduce);
                } else {
                    PrecisionCompute(rowTail_, gammaLocalReduce, betaLocalReduce, elementCount);
                }
                CopyOut(rowIdx, rowTail_);
            }
        }
    }

private:
    __aicore__ inline void CopyInAndAddBroadCast(
        int32_t rowCount, uint32_t elementCount, LocalTensor<T>& biasLocal, LocalTensor<T>& x1x2Local)
    {
        LocalTensor<float> addBufLocal = zBufFp32.Get<float>();
        LocalTensor<float> xBufLocal = xBufFp32.Get<float>();

        auto x1Local = x1x2Local[0];
        auto x2Local = x1x2Local[elementCount];

        // Use add as
        if constexpr (is_same<float, T>::value) {
            Add(addBufLocal, x2Local, x1Local, elementCount);
            PipeBarrier<PIPE_V>();
            for (int i = 0; i < rowCount; i++) {
                Add(addBufLocal[i * numLastDimAligned], biasLocal, addBufLocal[i * numLastDimAligned], numLastDim);
            }
            PipeBarrier<PIPE_V>();
        } else {
            Cast(addBufLocal, x1Local, RoundMode::CAST_NONE, elementCount);
            Cast(xBufLocal, x2Local, RoundMode::CAST_NONE, elementCount);
            PipeBarrier<PIPE_V>();
            Add(xBufLocal, addBufLocal, xBufLocal, elementCount);
            Cast(x1x2Local.template ReinterpretCast<float>(), biasLocal, RoundMode::CAST_NONE, numLastDim);
            PipeBarrier<PIPE_V>();
            for (int i = 0; i < rowCount; i++) {
                Add(addBufLocal[i * numLastDimAligned], x1x2Local.template ReinterpretCast<float>(),
                    xBufLocal[i * numLastDimAligned], numLastDim);
            }
            PipeBarrier<PIPE_V>();
        }
        x1x2Que.FreeTensor(x1x2Local);
    }

    __aicore__ inline void CopyIn(
        int32_t rowCount, uint32_t elementCount, LocalTensor<T>& x1x2Local, const DataCopyPadParams& padParams = {})
    {
        auto x1Local = x1x2Local[0];
        auto x2Local = x1x2Local[elementCount];

        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> addBufLocal = zBufFp32.Get<float>();

        // Use add as
        if constexpr (is_same<float, T>::value) {
            Add(addBufLocal, x2Local, x1Local, elementCount);
            PipeBarrier<PIPE_V>();
        } else {
            Cast(addBufLocal, x1Local, RoundMode::CAST_NONE, elementCount);
            Cast(xLocalFp32, x2Local, RoundMode::CAST_NONE, elementCount);
            PipeBarrier<PIPE_V>();
            Add(addBufLocal, addBufLocal, xLocalFp32, elementCount);
            PipeBarrier<PIPE_V>();
        }
        x1x2Que.FreeTensor(x1x2Local);
    }

    __aicore__ inline void CopyOutAdditionalOutput(int32_t procId, int32_t rowCount)
    {
        if constexpr (IS_ADDITIONAL_OUTPUT_ENABLE) {
            LocalTensor<float> addBufLocal = zBufFp32.Get<float>();
            uint32_t gmOffset = procId * rowStep * numLastDim;
            auto elementCount = numLastDimAligned * rowCount;
            auto xLocal = yQue.template AllocTensor<T>();
            if constexpr (is_same<T, float>::value) {
                Adds(xLocal, addBufLocal, ZERO, elementCount);
            } else if constexpr (is_same<T, half>::value) {
                Cast(xLocal, addBufLocal, RoundMode::CAST_NONE, elementCount);
            } else {
                Cast(xLocal, addBufLocal, RoundMode::CAST_RINT, elementCount);
            }
            PipeBarrier<PIPE_V>();
            yQue.template EnQue<T>(xLocal);

            auto x = yQue.template DeQue<T>();
            DataCopyEx(xGm[gmOffset], x, numLastDim, rowCount);
            yQue.FreeTensor(x);
        }
    }

    __aicore__ inline void PrecisionCompute(
        int32_t nums, LocalTensor<float>& gammaLocal, LocalTensor<float>& betaLocal, uint32_t elementCount)
    {
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));

#if OUTPUT_MEAN_RSTD == 1
        LocalTensor<float> meanLocal = meanQue.template AllocTensor<float>();
        LocalTensor<float> rstdLocal = rstdQue.template AllocTensor<float>();
#endif
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>(); // for reduce
        LocalTensor<float> zLocalFp32 = zBufFp32.Get<float>();

        // 1.1. mean process: 1/N * x_sum
        Muls(xLocalFp32, zLocalFp32, aveNum, elementCount);
        // 1.2. init buffer for reduce
        Duplicate(yLocalFp32, ZERO, nums * ELEM_PER_REP_FP32);
        PipeBarrier<PIPE_V>();

        // 2. mean end: reduce(1/N * x_sum)
        const uint32_t tailCount = numLastDim % ELEM_PER_REP_FP32;
        const uint32_t repeat = nums; // repeat < 255 * 8 = 2040
        const uint8_t repStride = numLastDimAligned / FLOAT_BLOCK_ELEM;
        ReduceSumForSmallReduceDim(
            xLocalFp32, xLocalFp32, yLocalFp32, numLastDimAligned, numLastDim, tailCount, repeat, repStride);

        // 3. rstd process: x - mean
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        for (int32_t rid = 0; rid < nums; ++rid) {
            auto roundOffset = rid * numLastDimAligned;

            auto meanTemp = xLocalFp32.GetValue(rid);
#if OUTPUT_MEAN_RSTD == 1
            meanLocal.SetValue(rid, meanTemp);
#endif
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            Adds(zLocalFp32[roundOffset], zLocalFp32[roundOffset], meanTemp * -1, numLastDim);
        }
        PipeBarrier<PIPE_V>();

        // 4. rstd process: (x - mean) ^ 2
        Mul(xLocalFp32, zLocalFp32, zLocalFp32, elementCount);
        PipeBarrier<PIPE_V>();

        // 5. rstd process: reduce( 1 / N * (x - mean) ^ 2 )
        Muls(xLocalFp32, xLocalFp32, aveNum, elementCount);
        PipeBarrier<PIPE_V>();
        Duplicate(yLocalFp32, ZERO, nums * ELEM_PER_REP_FP32);
        PipeBarrier<PIPE_V>();
        ReduceSumForSmallReduceDim(
            xLocalFp32, xLocalFp32, yLocalFp32, numLastDimAligned, numLastDim, tailCount, repeat, repStride);
        PipeBarrier<PIPE_V>();

        // 6. rstd end: 1 / sqrt( 1 / N * reduce( (x - mean) ^ 2 ) )
        Adds(xLocalFp32, xLocalFp32, eps, repeat);
        PipeBarrier<PIPE_V>();
        Sqrt(xLocalFp32, xLocalFp32, repeat);
        Duplicate(yLocalFp32, float(1), repeat);
        PipeBarrier<PIPE_V>();
        Div(xLocalFp32, yLocalFp32, xLocalFp32, repeat);

        // 7. y process: (x - mean) / rstd
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        for (int32_t rid = 0; rid < nums; ++rid) {
            auto roundOffset = rid * numLastDimAligned;

            float rstdTmp = xLocalFp32.GetValue(rid);
#if OUTPUT_MEAN_RSTD == 1
            rstdLocal.SetValue(rid, rstdTmp);
#endif
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            Muls(zLocalFp32[roundOffset], zLocalFp32[roundOffset], rstdTmp, numLastDim);
        }
        PipeBarrier<PIPE_V>();

        // 8. y = (x - mean) / rstd * beta + gamma
        LocalTensor<T> yLocal = yQue.template AllocTensor<T>();
        if constexpr (!is_same<T, float>::value) {
            for (int32_t rid = 0; rid < nums; ++rid) {
                Mul(zLocalFp32[rid * numLastDimAligned], zLocalFp32[rid * numLastDimAligned], gammaLocal, numLastDim);
            }
            PipeBarrier<PIPE_V>();
            for (int32_t rid = 0; rid < nums; ++rid) {
                Add(zLocalFp32[rid * numLastDimAligned], zLocalFp32[rid * numLastDimAligned], betaLocal, numLastDim);
            }
            PipeBarrier<PIPE_V>();

            if constexpr (is_same<T, half>::value) {
                Cast(yLocal, zLocalFp32, RoundMode::CAST_NONE, elementCount);
            } else {
                Cast(yLocal, zLocalFp32, RoundMode::CAST_RINT, elementCount);
            }
        } else {
            for (int32_t rid = 0; rid < nums; ++rid) {
                Mul(zLocalFp32[rid * numLastDimAligned], zLocalFp32[rid * numLastDimAligned], gammaLocal, numLastDim);
            }
            PipeBarrier<PIPE_V>();
            for (int32_t rid = 0; rid < nums; ++rid) {
                Add(yLocal[rid * numLastDimAligned], zLocalFp32[rid * numLastDimAligned], betaLocal, numLastDim);
            }
            PipeBarrier<PIPE_V>();
        }

#if OUTPUT_MEAN_RSTD == 1
        meanQue.EnQue(meanLocal);
        rstdQue.EnQue(rstdLocal);
#endif
        yQue.EnQue(yLocal);
    }

    __aicore__ inline void PrecisionComputeBigN(
        int32_t nums, LocalTensor<float>& gammaLocal, LocalTensor<float>& betaLocal)
    {
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        PrecisionComputeBigNBrcb(nums, gammaLocal, betaLocal);
#else
        precisionComputeBigNTranspose(nums, gammaLocal, betaLocal);
#endif
    }

    __aicore__ inline void precisionComputeBigNTranspose(
        int32_t nums, LocalTensor<float>& gammaLocal, LocalTensor<float>& betaLocal)
    {
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));

        LocalTensor<int16_t> orOffsetINT16 = orBufINT16.Get<int16_t>();

        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<float> zLocalFp32 = zBufFp32.Get<float>();

        LocalTensor<float> transposeSrcLocal = transposeSrcBuf.Get<float>();
        LocalTensor<float> transposeDstLocal = transposeDstBuf.Get<float>();

        int32_t elementNum = numLastDimAligned * nums;

        // 1.1. mean process: 1/N * x_sum
        Muls(xLocalFp32, zLocalFp32, aveNum, elementNum);
        // 1.2. init buffer for reduce
        Duplicate(yLocalFp32, ZERO, nums * ELEM_PER_REP_FP32);
        PipeBarrier<PIPE_V>();

        // 2.1. reducesum
        const uint32_t forCount = numLastDim / ELEM_PER_REP_FP32;
        const uint32_t tailCount = numLastDim % ELEM_PER_REP_FP32;
        const uint32_t repeat = nums; // repeat < 255 * 8 = 2040
        const uint8_t repStride = numLastDimAligned / FLOAT_BLOCK_ELEM;
        ReduceSumForSmallReduceDim(
            xLocalFp32, xLocalFp32, yLocalFp32, numLastDimAligned, numLastDim, tailCount, repeat, repStride);
        PipeBarrier<PIPE_V>();

        // 2.2. broadcast reducesum value
        uint64_t srcLocalList[16];
        uint64_t dstLocalList[16];
        InitVAForTransposeAscendC((__ubuf__ half *)transposeDstLocal.GetPhyAddr(), (__ubuf__ half *)transposeSrcLocal.GetPhyAddr(), dstLocalList, srcLocalList);
        CCEBroadCastShortAscendC(xLocalFp32.ReinterpretCast<int16_t>(), xLocalFp32.ReinterpretCast<float>(),
                        transposeDstLocal.ReinterpretCast<int16_t>(), transposeSrcLocal.ReinterpretCast<int16_t>(),
                        orOffsetINT16.ReinterpretCast<int16_t>(), dstLocalList, srcLocalList,
                        forCount, tailCount, repeat, repStride);
        PipeBarrier<PIPE_V>();

        // 3. rstd process: x - mean
        Sub(zLocalFp32, zLocalFp32, xLocalFp32, elementNum);
        PipeBarrier<PIPE_V>();

        // 4. rstd process: (x - mean) ^ 2
        Mul(xLocalFp32, zLocalFp32, zLocalFp32, elementNum);
        PipeBarrier<PIPE_V>();

        // 5. rstd process: reduce( 1 / N * (x - mean) ^ 2 )
        Muls(xLocalFp32, xLocalFp32, aveNum, elementNum);
        Duplicate(yLocalFp32, ZERO, nums * ELEM_PER_REP_FP32);
        PipeBarrier<PIPE_V>();
        ReduceSumForSmallReduceDim(
            xLocalFp32, xLocalFp32, yLocalFp32, numLastDimAligned, numLastDim, tailCount, repeat, repStride);
        PipeBarrier<PIPE_V>();

        // 6. rstd end: 1 / sqrt( 1 / N * reduce( (x - mean) ^ 2 ) )
        Adds(xLocalFp32, xLocalFp32, eps, nums);
        PipeBarrier<PIPE_V>();
        Sqrt(xLocalFp32, xLocalFp32, nums);
        Duplicate(yLocalFp32, (float)1, nums);
        PipeBarrier<PIPE_V>();
        Div(xLocalFp32, yLocalFp32, xLocalFp32, nums);
        PipeBarrier<PIPE_V>();

        // 7. broadcast reducesum value
        CCEBroadCastShortAscendC(xLocalFp32.ReinterpretCast<int16_t>(), xLocalFp32.ReinterpretCast<float>(),
                      transposeDstLocal.ReinterpretCast<int16_t>(), transposeSrcLocal.ReinterpretCast<int16_t>(),
                      orOffsetINT16.ReinterpretCast<int16_t>(), dstLocalList, srcLocalList,
                      forCount, tailCount, repeat, repStride);
        PipeBarrier<PIPE_V>();
        Mul(zLocalFp32, zLocalFp32, xLocalFp32, elementNum);
        PipeBarrier<PIPE_V>();

        // 8. y = (x - mean) / rstd * beta + gamma
        LocalTensor<T> yLocal = yQue.template AllocTensor<T>();
        if constexpr (!is_same<T, float>::value) {
            Level0MulFp32Short(zLocalFp32, gammaLocal, zLocalFp32, numLastDimAligned, nums, numLastDim);
            PipeBarrier<PIPE_V>();
            Level0AddFp32Short(zLocalFp32, betaLocal, zLocalFp32, numLastDimAligned, nums, numLastDim);
            PipeBarrier<PIPE_V>();

            if constexpr (is_same<T, half>::value) {
                Cast(yLocal, zLocalFp32, RoundMode::CAST_NONE, elementNum);
            } else {
                Cast(yLocal, zLocalFp32, RoundMode::CAST_RINT, elementNum);
            }
            PipeBarrier<PIPE_V>();
        } else {
            Level0MulFp32Short(yLocal, gammaLocal, zLocalFp32, numLastDimAligned, nums, numLastDim);
            PipeBarrier<PIPE_V>();
            Level0AddFp32Short(yLocal, betaLocal, yLocal, numLastDimAligned, nums, numLastDim);
            PipeBarrier<PIPE_V>();
        }

        yQue.EnQue(yLocal);
    }

#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    __aicore__ inline void PrecisionComputeBigNBrcb(
        int32_t nums, LocalTensor<float>& gammaLocal, LocalTensor<float>& betaLocal)
    {
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));

        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<float> zLocalFp32 = zBufFp32.Get<float>();

        LocalTensor<float> meanLocal = meanQue.template AllocTensor<float>();
        LocalTensor<float> rstdLocal = rstdQue.template AllocTensor<float>();

        int32_t elementNum = numLastDimAligned * nums;

        // 1.1. mean process: 1/N * x_sum
        Muls(xLocalFp32, zLocalFp32, aveNum, elementNum);
        // 1.2. init buffer for reduce
        Duplicate(yLocalFp32, ZERO, nums * ELEM_PER_REP_FP32);
        PipeBarrier<PIPE_V>();

        // 2.1. reducesum
        const uint32_t tailCount = numLastDim % ELEM_PER_REP_FP32;
        const uint32_t repeat = nums; // repeat < 255 * 8 = 2040
        const uint8_t repStride = numLastDimAligned / FLOAT_BLOCK_ELEM;
        ReduceSumForSmallReduceDim(
            meanLocal, xLocalFp32, yLocalFp32, numLastDimAligned, numLastDim, tailCount, repeat, repStride);
        PipeBarrier<PIPE_V>();

        // 2.2. broadcast reducesum value
        const uint32_t broadcastDim = BROADCAST_ND_DIM_NUM;
        const uint32_t broadcastAxis = BROADCAST_ND_LAST_INDEX;
        uint32_t dstShape[broadcastDim] = {(uint32_t)nums, (uint32_t)numLastDimAligned};
        uint32_t srcShape[broadcastDim] = {(uint32_t)nums, 1};
        auto sharedTmpBuffer = yLocalFp32.ReinterpretCast<uint8_t>();
        BroadCast<float, broadcastDim, broadcastAxis>(xLocalFp32, meanLocal, dstShape, srcShape, sharedTmpBuffer);
        PipeBarrier<PIPE_V>();

        // 3. rstd process: x - mean
        Sub(zLocalFp32, zLocalFp32, xLocalFp32, elementNum);
        PipeBarrier<PIPE_V>();

        // 4. rstd process: (x - mean) ^ 2
        Mul(xLocalFp32, zLocalFp32, zLocalFp32, elementNum);
        PipeBarrier<PIPE_V>();

        // 5. rstd process: reduce( 1 / N * (x - mean) ^ 2 )
        Muls(xLocalFp32, xLocalFp32, aveNum, elementNum);
        Duplicate(yLocalFp32, ZERO, nums * ELEM_PER_REP_FP32);
        PipeBarrier<PIPE_V>();
        ReduceSumForSmallReduceDim(
            rstdLocal, xLocalFp32, yLocalFp32, numLastDimAligned, numLastDim, tailCount, repeat, repStride);
        PipeBarrier<PIPE_V>();

        // 6. rstd end: 1 / sqrt( 1 / N * reduce( (x - mean) ^ 2 ) )
        Adds(rstdLocal, rstdLocal, eps, nums);
        PipeBarrier<PIPE_V>();
        Sqrt(rstdLocal, rstdLocal, nums);
        Duplicate(yLocalFp32, (float)1, nums);
        PipeBarrier<PIPE_V>();
        Div(rstdLocal, yLocalFp32, rstdLocal, nums);
        PipeBarrier<PIPE_V>();

        // 7. broadcast reducesum value
        BroadCast<float, broadcastDim, broadcastAxis>(xLocalFp32, rstdLocal, dstShape, srcShape, sharedTmpBuffer);
        PipeBarrier<PIPE_V>();
        Mul(zLocalFp32, zLocalFp32, xLocalFp32, elementNum);
        PipeBarrier<PIPE_V>();

        // 8. y = (x - mean) / rstd * beta + gamma
        LocalTensor<T> yLocalBigN = yQue.template AllocTensor<T>();
        if constexpr (!is_same<T, float>::value) {
            Level0MulFp32Short(zLocalFp32, gammaLocal, zLocalFp32, numLastDimAligned, nums, numLastDim);
            PipeBarrier<PIPE_V>();
            Level0AddFp32Short(zLocalFp32, betaLocal, zLocalFp32, numLastDimAligned, nums, numLastDim);
            PipeBarrier<PIPE_V>();

            if constexpr (is_same<T, half>::value) {
                Cast(yLocalBigN, zLocalFp32, RoundMode::CAST_NONE, elementNum);
            } else {
                Cast(yLocalBigN, zLocalFp32, RoundMode::CAST_RINT, elementNum);
            }
            PipeBarrier<PIPE_V>();
        } else {
            Level0MulFp32Short(yLocalBigN, gammaLocal, zLocalFp32, numLastDimAligned, nums, numLastDim);
            PipeBarrier<PIPE_V>();
            Level0AddFp32Short(yLocalBigN, betaLocal, yLocalBigN, numLastDimAligned, nums, numLastDim);
            PipeBarrier<PIPE_V>();
        }

        meanQue.EnQue(meanLocal);
        rstdQue.EnQue(rstdLocal);
        yQue.EnQue(yLocalBigN);
    }
#endif

    __aicore__ inline void CopyOut(int32_t rowIdx, int32_t rowCount)
    {
        LocalTensor<T> resReduce = yQue.template DeQue<T>();
        uint32_t gmOffsetReduce = rowIdx * rowStep * numLastDim;
        DataCopyEx(yGm[gmOffsetReduce], resReduce, numLastDim, rowCount);
        yQue.FreeTensor(resReduce);

#if OUTPUT_MEAN_RSTD == 1
        uint32_t gmOffsetMeanReduce = rowIdx * rowStep;
        LocalTensor<float> meanReduce = meanQue.template DeQue<float>();
        LocalTensor<float> rstdReduce = rstdQue.template DeQue<float>();
        DataCopyEx(meanGm[gmOffsetMeanReduce], meanReduce, rowCount);
        DataCopyEx(rstdGm[gmOffsetMeanReduce], rstdReduce, rowCount);
        meanQue.FreeTensor(meanReduce);
        rstdQue.FreeTensor(rstdReduce);
#endif
    }

private:
    TPipe* Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1x2Que;
    TBuf<TPosition::VECCALC> gammaBuf;
    TBuf<TPosition::VECCALC> betaBuf;
    TBuf<TPosition::VECCALC> biasBuf;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQue; // (x1 + x2) reuse this que
#if OUTPUT_MEAN_RSTD == 1
    TQue<QuePosition::VECOUT, BUFFER_NUM> meanQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rstdQue;
#endif

    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;
    TBuf<TPosition::VECCALC> zBufFp32;

    TBuf<TPosition::VECCALC> orBufINT16;
    TBuf<TPosition::VECCALC> transposeSrcBuf;
    TBuf<TPosition::VECCALC> transposeDstBuf;

    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> betaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<float> meanGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<float> workspaceGm;
    uint32_t numCore;
    uint32_t numLastDim;
    uint32_t numFirstDim;
    uint32_t rowStep;
    uint32_t rowWork;
    uint32_t gmOffset_;
    uint32_t rowTail_;
    uint32_t colTail;
    uint32_t colMoveCnt;
    uint32_t firstDimPerTime;
    uint32_t lastDimPerTime;
    uint32_t notLastFirstDimPerCore;
    uint32_t lFirstDimPerCore;
    float eps;
    float aveNum;
    bool lastDimPad = false;
    size_t numLastDimAligned;
    size_t numLastDimAlignedFp32;
};

#endif // __ADD_LAYER_NORM_NORMAL_SPECIAL_REDUCE_H_
