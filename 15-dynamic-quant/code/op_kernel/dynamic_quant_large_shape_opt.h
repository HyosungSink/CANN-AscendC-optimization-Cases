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
 * \file dynamic_quant_large_shape_opt.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_LARGE_SHAPE_OPT_H
#define DYNAMIC_QUANT_LARGE_SHAPE_OPT_H

#include "dynamic_quant_base.h"

namespace DynamicQuantNDOpt {
using namespace AscendC;

#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
template <typename T1, typename yDtype>
class DynamicQuantLargeShapeOpt : public DynamicQuantBase
{
public:
    __aicore__ inline DynamicQuantLargeShapeOpt(TPipe* pipe)
    {
        pPipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR workSpace,
        const DynamicQuantTilingData* __restrict tilingData)
    {
        ParseTilingData(tilingData);
        InitLargeShapeParams(offset);
        InitLargeShapeBaseBuffer();
        InitAndSetBuffer(x, smooth_scales, y, scale, offset);
    }

    __aicore__ inline void Process()
    {
        if (isAsymmetrical) {
            ProcessAsymmetrical();
        } else {
            ProcessSymmetrical();
        }
    }

    __aicore__ inline void ProcessSymmetrical()
    {
        uint32_t offsetBase = 0;
        uint32_t srcOffset = 0;
        float maxUpdateValue = 0.0;
        float scale;
        CopyInByEle(0, 0, tilingData_.innerLoopEle, 0);
        LocalTensor<float> scaleLocal = scaleQueue.AllocTensor<float>();
        for (uint32_t i = 0; i < loopCnt; i++) {
            maxUpdateValue = MIN_FLOAT_VALUE;
            offsetBase = i * tilingData_.rowLen;
            // 计算每一行的最大值
            for (uint32_t innerLoopIndex = 0; innerLoopIndex < tilingData_.innerLoopTimes; innerLoopIndex++) {
                srcOffset = offsetBase + innerLoopIndex * tilingData_.innerLoopEle;
                if (innerLoopIndex == tilingData_.innerLoopTimes - 1) {
                    if (tilingData_.innerLoopTail == 0) {
                        CopyInByEle(offsetBase, 0, tilingData_.innerLoopEle, 0);
                    } else {
                        CopyInByEle(srcOffset + tilingData_.innerLoopEle, innerLoopIndex + 1, tilingData_.innerLoopTail, rightPadding);
                    }
                } else {
                    CopyInByEle(srcOffset + tilingData_.innerLoopEle, innerLoopIndex + 1, tilingData_.innerLoopEle, 0);
                }
                ComputeGetMaxValue(tilingData_.innerLoopEle, maxUpdateValue);
            }
            // 如果核内切的有尾块
            if (tilingData_.innerLoopTail > 0) {
                srcOffset = offsetBase + tilingData_.innerLoopTimes * tilingData_.innerLoopEle;
                CopyInByEle(offsetBase, 0, tilingData_.innerLoopEle, 0);
                ComputeGetMaxValue(tilingData_.innerLoopTail, maxUpdateValue);
            }
            GetScale(maxUpdateValue, scale);
            scaleLocal.SetValue(i % scaleMaxLength, 1 / scale);
            // 量化计算
            SymmetricalQuant(offsetBase, scale, i);

            if ((i +1) % scaleMaxLength == 0 || i == loopCnt - 1) {
                scaleQueue.EnQue<float>(scaleLocal);
                if (i == loopCnt - 1) {
                    CopyScalesOut(loopCnt - scaleIdx * scaleMaxLength, scaleIdx * scaleMaxLength);
                } else {
                    CopyScalesOut(scaleMaxLength, scaleIdx * scaleMaxLength);
                    scaleIdx++;
                    scaleLocal = scaleQueue.AllocTensor<float>();
                }
            }
        }
    }

    __aicore__ inline void SymmetricalQuant(uint32_t offsetBase, float scale, uint32_t idx)
    {
        uint32_t srcOffset = 0;
        for (uint32_t innerLoopIndex = 0; innerLoopIndex < tilingData_.innerLoopTimes; innerLoopIndex++) {
            srcOffset = offsetBase + innerLoopIndex * tilingData_.innerLoopEle;
            if (innerLoopIndex == tilingData_.innerLoopTimes - 1) {
                if (tilingData_.innerLoopTail == 0 && idx < loopCnt - 1) {
                    CopyInByEle((idx + 1) * tilingData_.rowLen, 0, tilingData_.innerLoopEle, 0);
                } else if (tilingData_.innerLoopTail != 0) {
                    CopyInByEle(srcOffset + tilingData_.innerLoopEle, innerLoopIndex + 1, tilingData_.innerLoopTail, rightPadding);
                }
            } else {
                CopyInByEle(srcOffset + tilingData_.innerLoopEle, innerLoopIndex + 1, tilingData_.innerLoopEle, 0);
            }
            Compute(scale, 0.0, tilingData_.innerLoopEle);
            CopyOut(srcOffset, tilingData_.innerLoopEle);
        }
        if (tilingData_.innerLoopTail > 0) {
            srcOffset = offsetBase + tilingData_.innerLoopTimes * tilingData_.innerLoopEle;
            if (idx < loopCnt - 1) {
                CopyInByEle((idx + 1) * tilingData_.rowLen, 0, tilingData_.innerLoopEle, 0);
            }
            Compute(scale, 0.0, tilingData_.innerLoopTail);
            CopyOut(srcOffset, tilingData_.innerLoopTail);
        }
    }

    __aicore__ inline void ProcessAsymmetrical()
    {
        uint32_t offsetBase = 0;
        uint32_t srcOffset = 0;
        float maxUpdateValue = 0.0;
        float minUpdateValue = 0.0;
        float scale, offset;

        LocalTensor<float> scaleLocal = scaleQueue.AllocTensor<float>();
        LocalTensor<float> offsetLocal = offsetQueue.AllocTensor<float>();
        for (uint32_t i = 0; i < loopCnt; i++) {
            maxUpdateValue = MIN_FLOAT_VALUE;
            minUpdateValue = MAX_FLOAT_VALUE;
            offsetBase = i * tilingData_.rowLen;
            // 计算每一行的最大值以及最小值
            for (uint32_t innerLoopIndex = 0; innerLoopIndex < tilingData_.innerLoopTimes; innerLoopIndex++) {
                srcOffset = offsetBase + innerLoopIndex * tilingData_.innerLoopEle;
                ComputeMaxAndMinValue(
                    srcOffset, innerLoopIndex, tilingData_.innerLoopEle, 0, maxUpdateValue, minUpdateValue);
            }
            // 如果核内切的有尾块
            if (tilingData_.innerLoopTail > 0) {
                srcOffset = offsetBase + tilingData_.innerLoopTimes * tilingData_.innerLoopEle;
                ComputeMaxAndMinValue(
                    srcOffset, tilingData_.innerLoopTimes, tilingData_.innerLoopTail, rightPadding, maxUpdateValue,
                    minUpdateValue);
            }

            GetScaleAndOffset(maxUpdateValue, minUpdateValue, scale, offset);
            scaleLocal.SetValue(i % scaleMaxLength, scale);
            offsetLocal.SetValue(i % scaleMaxLength, offset);

            // 量化计算
            for (uint32_t innerLoopIndex = 0; innerLoopIndex < tilingData_.innerLoopTimes; innerLoopIndex++) {
                srcOffset = offsetBase + innerLoopIndex * tilingData_.innerLoopEle;
                QuantCompute(srcOffset, innerLoopIndex, scale, offset, tilingData_.innerLoopEle, 0);
            }
            // 如果核内切的有尾块
            if (tilingData_.innerLoopTail > 0) {
                srcOffset = offsetBase + tilingData_.innerLoopTimes * tilingData_.innerLoopEle;
                QuantCompute(srcOffset, tilingData_.innerLoopTimes, scale, offset, tilingData_.innerLoopTail, rightPadding);
            }
            if ((i +1) % scaleMaxLength == 0 || i == loopCnt - 1) {
                scaleQueue.EnQue<float>(scaleLocal);
                offsetQueue.EnQue<float>(offsetLocal);

                if (i == loopCnt - 1) {
                    CopyScalesOut(loopCnt - scaleIdx * scaleMaxLength, scaleIdx * scaleMaxLength);
                    CopyOffsetOut(loopCnt - scaleIdx * scaleMaxLength, scaleIdx * scaleMaxLength);
                } else {
                    CopyScalesOut(scaleMaxLength, scaleIdx * scaleMaxLength);
                    CopyOffsetOut(scaleMaxLength, scaleIdx * scaleMaxLength);
                    scaleIdx++;
                    scaleLocal = scaleQueue.AllocTensor<float>();
                    offsetLocal = offsetQueue.AllocTensor<float>();
                }
            }
        }
    }

private:
    __aicore__ inline void InitAndSetBuffer(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset)
    {
        scaleMaxLength = tilingData_.ubSize - UB_RESERVED_LENGTH - THIRTY_TWO - tilingData_.innerLoopEle * sizeof(float) * DOUBLE;
        if (tilingData_.hasSmooth) {
            smoothGm.SetGlobalBuffer((__gm__ T1*)smooth_scales);
            pPipe->InitBuffer(smoothQueue, DOUBLE_BUFFER_NUM, tilingData_.innerLoopEle * sizeof(T1));
            scaleMaxLength -= tilingData_.innerLoopEle * sizeof(T1) * DOUBLE_BUFFER_NUM;
        }
        if (blockIdx < tilingData_.headCoreNum) {
            inGm.SetGlobalBuffer((__gm__ T1*)x + blockIdx * static_cast<int64_t>(lenHead), lenHead);
            outGm.SetGlobalBuffer((__gm__ yDtype*)y + blockIdx * static_cast<int64_t>(outLenHead), outLenHead);
            scaleGm.SetGlobalBuffer((__gm__ float*)scale + blockIdx * rowPerHeadCore, rowPerHeadCore);
            if (isAsymmetrical) {
                offsetGm.SetGlobalBuffer((__gm__ float*)offset + blockIdx * rowPerHeadCore, rowPerHeadCore);
            }
        } else {
            inGm.SetGlobalBuffer(
                (__gm__ T1*)x + tilingData_.headCoreNum * static_cast<int64_t>(lenHead) + (blockIdx - tilingData_.headCoreNum) * static_cast<int64_t>(lenTail),
                lenTail);
            outGm.SetGlobalBuffer(
                (__gm__ yDtype*)y + tilingData_.headCoreNum * static_cast<int64_t>(outLenHead) +
                    (blockIdx - tilingData_.headCoreNum) * static_cast<int64_t>(outLenTail),
                outLenTail);
            scaleGm.SetGlobalBuffer(
                (__gm__ float*)scale + tilingData_.headCoreNum * rowPerHeadCore +
                    (blockIdx - tilingData_.headCoreNum) * rowPerTailCore,
                rowPerTailCore);
            if (isAsymmetrical) {
                offsetGm.SetGlobalBuffer(
                    (__gm__ float*)offset + tilingData_.headCoreNum * rowPerHeadCore +
                        (blockIdx - tilingData_.headCoreNum) * rowPerTailCore,
                    rowPerTailCore);
            }
        }
        scaleMaxLength -= (tilingData_.innerLoopEle * sizeof(T1) + tilingData_.innerLoopEle * sizeof(yDtype)) * DOUBLE_BUFFER_NUM;
        scaleMaxLength = scaleMaxLength / THIRTY_TWO * THIRTY_TWO / sizeof(float);
        if (isAsymmetrical) {
            scaleMaxLength /= DOUBLE;
            pPipe->InitBuffer(offsetQueue, BUFFER_NUM, GetMin(sizeFloatLen, scaleMaxLength) * sizeof(float));
        }
        // innerLoopEle已经保证了32Bytes对齐
        pPipe->InitBuffer(inQueue, DOUBLE_BUFFER_NUM, tilingData_.innerLoopEle * sizeof(T1));
        pPipe->InitBuffer(outQueue, DOUBLE_BUFFER_NUM, tilingData_.innerLoopEle * sizeof(yDtype));
        pPipe->InitBuffer(scaleQueue, BUFFER_NUM, GetMin(sizeFloatLen, scaleMaxLength) * sizeof(float));
    }

    __aicore__ inline void CopyInByEle(uint32_t offset, uint32_t loopIndex, uint32_t elementNum, uint8_t rightPadding)
    {
        LocalTensor<T1> inLocal = inQueue.AllocTensor<T1>();
        DataCopyParams copyParams{1, (uint16_t)(elementNum * sizeof(T1)), 0, 0};
        DataCopyPadParams padParams{true, 0, rightPadding, 0};
        DataCopyPad(inLocal, inGm[offset], copyParams, padParams);

        inQueue.EnQue(inLocal);
        if (tilingData_.hasSmooth) {
            LocalTensor<T1> smoothLocal = smoothQueue.AllocTensor<T1>();
            // 一次拷贝不完
            DataCopyParams copyParams1{1, (uint16_t)(elementNum * sizeof(T1)), 0, 0};
            DataCopyPadParams padParams1{true, 0, rightPadding, 0};
            DataCopyPad(smoothLocal, smoothGm[loopIndex * tilingData_.innerLoopEle], copyParams1, padParams1);
            smoothQueue.EnQue(smoothLocal);
        }
    }

    __aicore__ inline void CopyScalesOut(uint32_t element, uint32_t offset)
    {
        LocalTensor<float> scaleLocal = scaleQueue.DeQue<float>();
        DataCopyParams copyParams{1, (uint16_t)(element * sizeof(float)), 0, 0};
        DataCopyPad(scaleGm[offset], scaleLocal, copyParams);
        scaleQueue.FreeTensor(scaleLocal);
    }

    __aicore__ inline void CopyOffsetOut(uint32_t element, uint32_t offset)
    {
        LocalTensor<float> offsetLocal = offsetQueue.DeQue<float>();
        DataCopyParams copyParams{1, (uint16_t)(element * sizeof(float)), 0, 0};
        DataCopyPad(offsetGm[offset], offsetLocal, copyParams);
        offsetQueue.FreeTensor(offsetLocal);
    }

    __aicore__ inline void QuantCompute(
        uint32_t offset, uint32_t loopIndex, float scale, float offsetOut, uint32_t elementNum, uint8_t rightPadding)
    {
        CopyInByEle(offset, loopIndex, elementNum, rightPadding);
        Compute(scale, offsetOut, elementNum);
        CopyOut(offset, elementNum);
    }

    __aicore__ inline void ComputeMaxValue(
        uint32_t offset, uint32_t loopIndex, uint32_t elementNum, uint8_t rightPadding, float& maxUpdateValue)
    {
        CopyInByEle(offset, loopIndex, elementNum, rightPadding);
        ComputeGetMaxValue(elementNum, maxUpdateValue);
    }

    __aicore__ inline void ComputeTail(uint32_t elementNum, float& maxUpdateValue)
    {
        float scale = 0.0f;
        LocalTensor<T1> smoothScalsLocal;
        LocalTensor<T1> inLocal = inQueue.DeQue<T1>();
        LocalTensor<float> tempFp32 = tempCastUb.Get<float>();
        LocalTensor<yDtype> outLocal = outQueue.AllocTensor<yDtype>();
        AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();
        LocalTensor<int32_t> tempCastInt32 = fp32_buf_.Get<int32_t>();
        Cast(tempFp32, inLocal, RoundMode::CAST_NONE, elementNum);
        PipeBarrier<PIPE_V>();
        if (tilingData_.hasSmooth) {
            smoothScalsLocal = smoothQueue.DeQue<T1>();
            Cast(temp, smoothScalsLocal, RoundMode::CAST_NONE, elementNum);
            PipeBarrier<PIPE_V>();
            Mul(tempFp32, tempFp32, temp, elementNum);
            PipeBarrier<PIPE_V>();
            smoothQueue.FreeTensor(smoothScalsLocal);
        }
        Abs(temp, tempFp32, elementNum);
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(temp, elementNum);
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float maxValue = temp.GetValue(0);
        maxUpdateValue = (maxUpdateValue > maxValue) ? maxUpdateValue : maxValue;
        GetScale(maxUpdateValue, scale);
        Muls(tempFp32, tempFp32, scale, elementNum);
        PipeBarrier<PIPE_V>();
        Cast(tempCastInt32, tempFp32, RoundMode::CAST_RINT, elementNum);
        PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<half> tempHalfCast = temp.ReinterpretCast<half>();
        SetDeqScale(static_cast<half>(1.0));
        PipeBarrier<PIPE_V>();
        Cast(tempHalfCast, tempCastInt32, RoundMode::CAST_ROUND, elementNum);
        PipeBarrier<PIPE_V>();
        Cast(outLocal, tempHalfCast, RoundMode::CAST_TRUNC, elementNum);
        outQueue.EnQue<yDtype>(outLocal);
        inQueue.FreeTensor(inLocal);
    }

    __aicore__ inline void ComputeGetMaxValue(uint32_t elementNum, float& maxUpdateValue)
    {
        LocalTensor<T1> smoothLocal;
        LocalTensor<T1> inLocal = inQueue.DeQue<T1>();
        LocalTensor<float> tempFp32 = tempCastUb.Get<float>();
        AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();

        Cast(tempFp32, inLocal, RoundMode::CAST_NONE, elementNum);
        PipeBarrier<PIPE_V>();
        if (tilingData_.hasSmooth) {
            smoothLocal = smoothQueue.DeQue<T1>();
            Cast(temp, smoothLocal, RoundMode::CAST_NONE, elementNum);
            PipeBarrier<PIPE_V>();
            Mul(tempFp32, tempFp32, temp, elementNum);
            PipeBarrier<PIPE_V>();
            smoothQueue.FreeTensor(smoothLocal);
        }
        Abs(temp, tempFp32, elementNum);
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(temp, elementNum);
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float maxValue = temp.GetValue(0);
        maxUpdateValue = (maxUpdateValue > maxValue) ? maxUpdateValue : maxValue;
        inQueue.FreeTensor(inLocal);
    }

    __aicore__ inline void ComputeMaxAndMinValue(
        uint32_t offset, uint32_t loopIndex, uint32_t elementNum, uint8_t rightPadding, float& maxUpdateValue,
        float& minUpdataValue)
    {
        CopyInByEle(offset, loopIndex, elementNum, rightPadding);
        ComputeGetMaxAndMinValue(elementNum, maxUpdateValue, minUpdataValue);
    }

    __aicore__ inline void ComputeGetMaxAndMinValue(uint32_t elementNum, float& maxUpdateValue, float& minUpdateValue)
    {
        LocalTensor<T1> smoothLocal;
        LocalTensor<T1> inLocal = inQueue.DeQue<T1>();
        LocalTensor<float> tempFp32 = tempCastUb.Get<float>();
        AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();

        Cast(tempFp32, inLocal, RoundMode::CAST_NONE, elementNum);
        PipeBarrier<PIPE_V>();
        if (tilingData_.hasSmooth) {
            smoothLocal = smoothQueue.DeQue<T1>();
            Cast(temp, smoothLocal, RoundMode::CAST_NONE, elementNum);
            PipeBarrier<PIPE_V>();
            Mul(tempFp32, tempFp32, temp, elementNum);
            PipeBarrier<PIPE_V>();
            smoothQueue.FreeTensor(smoothLocal);
        }
        ReduceMax(temp, tempFp32, temp, elementNum, false);
        PipeBarrier<PIPE_V>();
        float maxValue = temp.GetValue(0);
        maxUpdateValue = (maxUpdateValue > maxValue) ? maxUpdateValue : maxValue;
        ReduceMin(temp, tempFp32, temp, elementNum, false);
        PipeBarrier<PIPE_V>();
        float minValue = temp.GetValue(0);
        minUpdateValue = (minUpdateValue > minValue) ? minValue : minUpdateValue;
        inQueue.FreeTensor(inLocal);
    }

    __aicore__ inline void Compute(float scale, float offset, uint32_t elementNum)
    {
        LocalTensor<T1> smoothScalsLocal;
        LocalTensor<T1> inLocal = inQueue.DeQue<T1>();
        LocalTensor<float> tempFp32 = tempCastUb.Get<float>();
        LocalTensor<yDtype> outLocal = outQueue.AllocTensor<yDtype>();
        AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();
        LocalTensor<int32_t> tempInt32 = fp32_buf_.Get<int32_t>();

        Cast(tempFp32, inLocal, RoundMode::CAST_NONE, elementNum);
        PipeBarrier<PIPE_V>();
        if (tilingData_.hasSmooth) {
            smoothScalsLocal = smoothQueue.DeQue<T1>();
            Cast(temp, smoothScalsLocal, RoundMode::CAST_NONE, elementNum);
            PipeBarrier<PIPE_V>();
            Mul(tempFp32, tempFp32, temp, elementNum);
            PipeBarrier<PIPE_V>();
            smoothQueue.FreeTensor(smoothScalsLocal);
        }
        if (isAsymmetrical) {
            Muls(tempFp32, tempFp32, 1 / scale, elementNum);
            PipeBarrier<PIPE_V>();
            Adds(tempFp32, tempFp32, offset, elementNum);
        } else {
            Muls(tempFp32, tempFp32, scale, elementNum);
        }
        PipeBarrier<PIPE_V>();
        Cast(tempInt32, tempFp32, RoundMode::CAST_RINT, elementNum);
        PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<half> tempHalfCast = temp.ReinterpretCast<half>();
        SetDeqScale(static_cast<half>(1.0));
        PipeBarrier<PIPE_V>();
        Cast(tempHalfCast, tempInt32, RoundMode::CAST_ROUND, elementNum);
        PipeBarrier<PIPE_V>();
        Cast(outLocal, tempHalfCast, RoundMode::CAST_TRUNC, elementNum);
        outQueue.EnQue<yDtype>(outLocal);
        inQueue.FreeTensor(inLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t element)
    {
        LocalTensor<yDtype> outLocal = outQueue.DeQue<yDtype>();
        DataCopyExtParams copyExtParams{(uint16_t)1, (uint16_t)(element * sizeof(yDtype)), 0, 0, 0};
        if constexpr (IsSameType<yDtype, int4b_t>::value) {
            copyExtParams.blockLen = copyExtParams.blockLen >> 1;
            uint32_t index = offset;
            DataCopyPad(outGm[index], outLocal, copyExtParams);
        } else {
            DataCopyPad(outGm[offset], outLocal, copyExtParams);
        }
        outQueue.FreeTensor(outLocal);
    }

private:
    /* ascendc variable */
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> smoothQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> scaleQueue, offsetQueue;

    /* global memory address */
    GlobalTensor<T1> inGm;
    GlobalTensor<T1> smoothGm;
    GlobalTensor<yDtype> outGm;
    GlobalTensor<float> scaleGm, offsetGm;

    int64_t scaleMaxLength = 0;
    int64_t scaleIdx = 0;
};
#endif
} // namespace DynamicQuantNDOpt
#endif // DYNAMIC_QUANT_LARGE_SHAPE_OPT_H