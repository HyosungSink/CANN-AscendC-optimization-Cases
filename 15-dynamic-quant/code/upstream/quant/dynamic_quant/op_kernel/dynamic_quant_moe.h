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
 * \file dynamic_quant_moe.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_MOE_H
#define DYNAMIC_QUANT_MOE_H

#include "dynamic_quant_base.h"

namespace DynamicQuantNDOpt {
using namespace AscendC;

#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
template <typename T1, typename T3, typename yDtype>
class DynamicQuantMoe : public DynamicQuantBase
{
public:
    __aicore__ inline DynamicQuantMoe(TPipe* pipe)
    {
        pPipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_index, GM_ADDR y, GM_ADDR scale, GM_ADDR offset,
        GM_ADDR workSpace, const DynamicQuantTilingData* __restrict tilingData)
    {
        ParseTilingData(tilingData);
        InitParams(offset);
        InitBaseBuffer();
        InitAndSetBuffer(x, smooth_scales, group_index, y, scale, offset);
    }

    __aicore__ inline void Process()
    {
        uint32_t smoothIndex = 0;
        T3 offsetRow = 0;

        GroupCopyIn();
        groupLocal = groupQueue.DeQue<T3>();

        DuplicateConst();
        for (uint32_t i = 0; i < loopCnt; i++) {
            offsetRow = baseRow + i * multiRowNum;
            LoopProcess(multiRowNum, offsetRow, i, smoothIndex);
        }

        if (remainRow > 0) {
            offsetRow = baseRow + loopCnt * multiRowNum;
            LoopProcess(remainRow, offsetRow, loopCnt, smoothIndex);
        }
        groupQueue.FreeTensor(groupLocal);
    }

private:
    __aicore__ inline void InitAndSetBuffer(
        GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_index, GM_ADDR y, GM_ADDR scale, GM_ADDR offset)
    {
        smoothGm.SetGlobalBuffer((__gm__ T1*)smooth_scales);
        pPipe->InitBuffer(smooth_buf_, sizeHalfLen * sizeof(T1));
        pPipe->InitBuffer(smoothQueue, BUFFER_NUM, sizeHalfLen * sizeof(float));

        groupGm.SetGlobalBuffer((__gm__ T3*)group_index);
        pPipe->InitBuffer(groupQueue, BUFFER_NUM, tilingData_.alignGroupNum * sizeof(T3));

        if (blockIdx < tilingData_.headCoreNum) {
            baseRow = blockIdx * rowPerHeadCore;
            inGm.SetGlobalBuffer((__gm__ T1*)x + blockIdx * lenHead, lenHead);
            outGm.SetGlobalBuffer((__gm__ yDtype*)y + blockIdx * outLenHead, outLenHead);
            scaleGm.SetGlobalBuffer((__gm__ float*)scale + baseRow, rowPerHeadCore);
            if (isAsymmetrical) {
                offsetGm.SetGlobalBuffer((__gm__ float*)offset + baseRow, rowPerHeadCore);
            }
        } else {
            baseRow = tilingData_.headCoreNum * rowPerHeadCore + (blockIdx - tilingData_.headCoreNum) * rowPerTailCore;
            inGm.SetGlobalBuffer(
                (__gm__ T1*)x + tilingData_.headCoreNum * lenHead + (blockIdx - tilingData_.headCoreNum) * lenTail,
                lenTail);
            outGm.SetGlobalBuffer(
                (__gm__ yDtype*)y + tilingData_.headCoreNum * outLenHead +
                    (blockIdx - tilingData_.headCoreNum) * outLenTail,
                outLenTail);
            scaleGm.SetGlobalBuffer((__gm__ float*)scale + baseRow, rowPerTailCore);
            if (isAsymmetrical) {
                offsetGm.SetGlobalBuffer((__gm__ float*)offset + baseRow, rowPerTailCore);
            }
        }
        if (isAsymmetrical) {
            pPipe->InitBuffer(offsetQueue, BUFFER_NUM, sizeFloatLen * sizeof(float));
        }
        pPipe->InitBuffer(inQueue, BUFFER_NUM, lenMultiRow * sizeof(T1));
        pPipe->InitBuffer(outQueue, BUFFER_NUM, outLen * sizeof(yDtype));
        pPipe->InitBuffer(scaleQueue, BUFFER_NUM, sizeFloatLen * sizeof(float));
    }

    __aicore__ inline void GroupCopyIn()
    {
        LocalTensor<T3> groupLocal = groupQueue.AllocTensor<T3>();
        uint8_t rightPadding = tilingData_.alignGroupNum - tilingData_.groupNum;
        DataCopyParams copyParams{1, (uint16_t)(tilingData_.groupNum * sizeof(T3)), 0, 0};
        DataCopyPadParams padParams{true, 0, rightPadding, 0};
        DataCopyPad(groupLocal, groupGm, copyParams, padParams);
        groupQueue.EnQue(groupLocal);
    }

    __aicore__ inline uint32_t GetSmoothIndex(T3 realRowNum, T3& groupNum, uint32_t smoothIndex)
    {
        // 获取符合条件的smooth_scales的index
        for (size_t index = smoothIndex; index < tilingData_.groupNum; index++) {
            groupNum = groupLocal.GetValue(index);
            if (groupNum >= realRowNum) {
                return index;
            }
        }
        return smoothIndex;
    }

    __aicore__ inline void LoopProcess(uint32_t multiRow, T3 offsetRow, uint32_t loopNum, uint32_t& smoothIndex)
    {
        uint32_t smoothOffset = smoothIndex * tilingData_.rowLen;
        CopyIn(multiRow, loopNum, smoothOffset);
        if (isAsymmetrical) {
            ComputeAsymmetric(multiRow, offsetRow, smoothIndex);
        } else {
            Compute(multiRow, offsetRow, smoothIndex);
        }
        CopyOut(multiRow, loopNum);
    }

    __aicore__ inline void SmoothCopyIn(uint32_t offset)
    {
        LocalTensor<T1> smoothLocal = smooth_buf_.Get<T1>();
        if (isPad) {
            DataCopyParams copyParams{1, (uint16_t)(tilingData_.rowLen * sizeof(T1)), 0, 0};
            DataCopyPadParams padParams{true, 0, rightPadding, 0};
            DataCopyPad(smoothLocal, smoothGm[offset], copyParams, padParams);
        } else {
            DataCopy(smoothLocal, smoothGm[offset], tilingData_.rowLen);
        }
        PipeBarrier<PIPE_ALL>();
        LocalTensor<float> smoothTemp = smoothQueue.AllocTensor<float>();
        Cast(smoothTemp, smoothLocal, RoundMode::CAST_NONE, tilingData_.rowLen);
        smoothQueue.EnQue(smoothTemp);
    }

    __aicore__ inline void CopyIn(uint32_t multiRow, uint32_t loopNum, uint32_t smoothOffset)
    {
        LocalTensor<T1> inLocal = inQueue.AllocTensor<T1>();
        if (isPad) {
            DataCopyParams copyParams{(uint16_t)multiRow, (uint16_t)(tilingData_.rowLen * sizeof(half)), 0, 0};
            DataCopyPadParams padParams{true, 0, rightPadding, 0};
            DataCopyPad(inLocal, inGm[loopNum * lenGMMultiRow], copyParams, padParams);
        } else {
            DataCopy(inLocal, inGm[loopNum * lenGMMultiRow], multiRow * tilingData_.rowLen);
        }
        inQueue.EnQue(inLocal);
        SmoothCopyIn(smoothOffset);
    }

    __aicore__ inline void Compute(uint32_t multiRow, T3 offsetRow, uint32_t& smoothIndex)
    {
        uint32_t index = 0;
        uint32_t smoothOffset = 0;
        T3 realRowNum = 0;
        T3 groupValue = groupLocal.GetValue(smoothIndex);

        LocalTensor<float> scaleLocal = scaleQueue.AllocTensor<float>();
        LocalTensor<T1> inLocal = inQueue.DeQue<T1>();
        LocalTensor<float> smoothLocal = smoothQueue.DeQue<float>();
        LocalTensor<float> tempFp32 = tempCastUb.Get<float>();
        LocalTensor<yDtype> outLocal = outQueue.AllocTensor<yDtype>();
        AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();
        LocalTensor<int32_t> tempInt32 = fp32_buf_.Get<int32_t>();
        auto tempHalf = temp.ReinterpretCast<half>();

        for (int32_t i = 0; i < multiRow; i++) {
            index = i * sizeHalfLen;
            Cast(tempFp32, inLocal[index], RoundMode::CAST_NONE, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();

            realRowNum = offsetRow + i + 1;
            if (groupValue < realRowNum) {
                smoothQueue.FreeTensor(smoothLocal);
                smoothIndex = GetSmoothIndex(realRowNum, groupValue, smoothIndex + 1);
                smoothOffset = smoothIndex * tilingData_.rowLen;
                SmoothCopyIn(smoothOffset);
                smoothLocal = smoothQueue.DeQue<float>();
            }
            Mul(tempFp32, tempFp32, smoothLocal, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            Abs(temp, tempFp32, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            ReduceMax(temp, temp, temp, tilingData_.rowLen, false);
            PipeBarrier<PIPE_V>();

            Div(temp, constScale, temp, MAX_VALUE_NUM);
            event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(event_v_s);
            WaitFlag<HardEvent::V_S>(event_v_s);
            float scale = temp.GetValue(0);
            scaleLocal.SetValue(i, 1 / scale);
            Muls(tempFp32, tempFp32, scale, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            Cast(tempInt32, tempFp32, RoundMode::CAST_RINT, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            SetDeqScale(static_cast<half>(1.0));
            PipeBarrier<PIPE_V>();
            Cast(tempHalf, tempInt32, RoundMode::CAST_ROUND, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            Cast(outLocal[i * outAlignLen], tempHalf, RoundMode::CAST_TRUNC, tilingData_.rowLen);
        }
        outQueue.EnQue<yDtype>(outLocal);
        scaleQueue.EnQue<float>(scaleLocal);
        inQueue.FreeTensor(inLocal);
        smoothQueue.FreeTensor(smoothLocal);
    }

    __aicore__ inline void ComputeAsymmetric(uint32_t multiRow, T3 offsetRow, uint32_t& smoothIndex)
    {
        uint32_t index = 0;
        uint32_t smoothOffset = 0;
        float max_value, min_value;
        float back_scale;
        float scale, offset;
        T3 realRowNum = 0;
        T3 groupValue = groupLocal.GetValue(smoothIndex);

        LocalTensor<float> scaleLocal = scaleQueue.AllocTensor<float>();
        LocalTensor<float> offsetLocal = offsetQueue.AllocTensor<float>();
        LocalTensor<T1> inLocal = inQueue.DeQue<T1>();
        LocalTensor<float> smoothLocal = smoothQueue.DeQue<float>();
        LocalTensor<float> tempFp32 = tempCastUb.Get<float>();
        LocalTensor<yDtype> outLocal = outQueue.AllocTensor<yDtype>();
        AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();
        LocalTensor<int32_t> tempInt32 = fp32_buf_.Get<int32_t>();
        auto tempHalfCast = temp.ReinterpretCast<half>();

        for (int32_t i = 0; i < multiRow; i++) {
            index = i * sizeHalfLen;
            Cast(tempFp32, inLocal[index], RoundMode::CAST_NONE, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();

            realRowNum = offsetRow + i + 1;
            if (groupValue < realRowNum) {
                smoothQueue.FreeTensor(smoothLocal);
                smoothIndex = GetSmoothIndex(realRowNum, groupValue, smoothIndex + 1);
                smoothOffset = smoothIndex * tilingData_.rowLen;
                SmoothCopyIn(smoothOffset);
                smoothLocal = smoothQueue.DeQue<float>();
            }
            Mul(tempFp32, tempFp32, smoothLocal, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            ReduceMax(temp, tempFp32, temp, tilingData_.rowLen, false);
            PipeBarrier<PIPE_V>();
            max_value = temp.GetValue(0);
            ReduceMin(temp, tempFp32, temp, tilingData_.rowLen, false);
            PipeBarrier<PIPE_V>();
            min_value = temp.GetValue(0);
            GetScaleAndOffset(max_value, min_value, scale, offset);
            back_scale = 1 / scale;
            scaleLocal.SetValue(i, scale);
            offsetLocal.SetValue(i, offset);
            Muls(tempFp32, tempFp32, back_scale, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            Adds(tempFp32, tempFp32, offset, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            Cast(tempInt32, tempFp32, RoundMode::CAST_RINT, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            SetDeqScale(static_cast<half>(1.0));
            PipeBarrier<PIPE_V>();
            Cast(tempHalfCast, tempInt32, RoundMode::CAST_ROUND, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            Cast(outLocal[i * outAlignLen], tempHalfCast, RoundMode::CAST_TRUNC, tilingData_.rowLen);
        }
        outQueue.EnQue<yDtype>(outLocal);
        scaleQueue.EnQue<float>(scaleLocal);
        offsetQueue.EnQue<float>(offsetLocal);
        inQueue.FreeTensor(inLocal);
        smoothQueue.FreeTensor(smoothLocal);
    }

    __aicore__ inline void CopyOut(uint32_t multiRow, uint32_t loopCount)
    {
        LocalTensor<yDtype> outLocal = outQueue.DeQue<yDtype>();
        LocalTensor<float> scaleLocal = scaleQueue.DeQue<float>();
        LocalTensor<float> offsetLocal;
        if (isAsymmetrical) {
            offsetLocal = offsetQueue.DeQue<float>();
        }
        DataCopyExtParams copyOutParams{(uint16_t)multiRow, (uint16_t)(tilingData_.rowLen * sizeof(yDtype)), 0, 0, 0};
        if constexpr (IsSameType<yDtype, int4b_t>::value) {
            copyOutParams.blockLen = copyOutParams.blockLen >> 1;
            uint32_t index = loopCount * lenGMMultiRow;
            DataCopyPad(outGm[index], outLocal, copyOutParams);
        } else {
            DataCopyPad(outGm[loopCount * lenGMMultiRow], outLocal, copyOutParams);
        }

        DataCopyParams copyParams1{1, (uint16_t)(multiRow * sizeof(float)), 0, 0};
        DataCopyPad(scaleGm[loopCount * multiRowNum], scaleLocal, copyParams1);
        if (isAsymmetrical) {
            DataCopyPad(offsetGm[loopCount * multiRowNum], offsetLocal, copyParams1);
            offsetQueue.FreeTensor(offsetLocal);
        }
        outQueue.FreeTensor(outLocal);
        scaleQueue.FreeTensor(scaleLocal);
    }

private:
    /* ascendc variable */
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> smoothQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> groupQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue, scaleQueue, offsetQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> smooth_buf_;

    /* global memory address */
    GlobalTensor<T1> inGm;
    GlobalTensor<T1> smoothGm;
    GlobalTensor<T3> groupGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> offsetGm;
    GlobalTensor<yDtype> outGm;

    /* variable */
    T3 baseRow = 0; // 记录开始处理的行数
    LocalTensor<T3> groupLocal;
};
#endif
} // namespace DynamicQuantNDOpt
#endif // DYNAMIC_QUANT_MOE_H
