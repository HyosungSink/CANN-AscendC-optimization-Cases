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
 * \file dynamic_quant.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_H
#define DYNAMIC_QUANT_H

#include "dynamic_quant_base.h"

namespace DynamicQuantNDOpt {
using namespace AscendC;

#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
template <typename xDtype, typename yDtype>
class DynamicQuant : public DynamicQuantBase
{
public:
    __aicore__ inline DynamicQuant(TPipe* pipe)
    {
        pPipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR workSpace,
        const DynamicQuantTilingData* __restrict tilingData)
    {
        ParseTilingData(tilingData);
        InitParams(offset);
        InitBaseBuffer();
        InitAndSetBuffer(x, smooth_scales, y, scale, offset);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<xDtype> smoothHalfLocal;
        LocalTensor<float> smoothLocal;
        // copy smooth from GM to UB
        if (tilingData_.hasSmooth) {
            smoothLocal = smooth_buf_.Get<float>();
            SmoothCopyIn();
            smoothHalfLocal = smoothQueue.DeQue<xDtype>();
            PipeBarrier<PIPE_V>();
            Cast(smoothLocal, smoothHalfLocal, RoundMode::CAST_NONE, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
        }

        DuplicateConst();
        for (int32_t i = 0; i < loopCnt; i++) {
            LoopProcess(smoothLocal, multiRowNum, i);
        }

        if (remainRow > 0) {
            LoopProcess(smoothLocal, remainRow, loopCnt);
        }

        if (tilingData_.hasSmooth) {
            smoothQueue.FreeTensor(smoothHalfLocal);
        }
    }

private:
    __aicore__ inline void InitAndSetBuffer(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset)
    {
        if (tilingData_.hasSmooth) {
            smoothGm.SetGlobalBuffer((__gm__ xDtype*)smooth_scales);
            pPipe->InitBuffer(smooth_buf_, sizeHalfLen * sizeof(float));
            pPipe->InitBuffer(smoothQueue, BUFFER_NUM, sizeHalfLen * sizeof(xDtype));
        }
        if (blockIdx < tilingData_.headCoreNum) {
            inGm.SetGlobalBuffer((__gm__ xDtype*)x + blockIdx * lenHead, lenHead);
            outGm.SetGlobalBuffer((__gm__ yDtype*)y + blockIdx * outLenHead, outLenHead);
            scaleGm.SetGlobalBuffer((__gm__ float*)scale + blockIdx * rowPerHeadCore, rowPerHeadCore);
            if (isAsymmetrical) {
                offsetGm.SetGlobalBuffer((__gm__ float*)offset + blockIdx * rowPerHeadCore, rowPerHeadCore);
            }
        } else {
            inGm.SetGlobalBuffer(
                (__gm__ xDtype*)x + tilingData_.headCoreNum * lenHead + (blockIdx - tilingData_.headCoreNum) * lenTail,
                lenTail);
            outGm.SetGlobalBuffer(
                (__gm__ yDtype*)y + tilingData_.headCoreNum * outLenHead +
                    (blockIdx - tilingData_.headCoreNum) * outLenTail,
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
        if (isAsymmetrical) {
            pPipe->InitBuffer(offsetQueue, BUFFER_NUM, sizeFloatLen * sizeof(float));
        }
        pPipe->InitBuffer(inQueue, BUFFER_NUM, lenMultiRow * sizeof(xDtype));
        pPipe->InitBuffer(outQueue, BUFFER_NUM, outLen * sizeof(yDtype));
        pPipe->InitBuffer(scaleQueue, BUFFER_NUM, sizeFloatLen * sizeof(float));
    }

    __aicore__ inline void LoopProcess(const LocalTensor<float>& smoothLocal, int32_t multiRow, int32_t loopNum)
    {
        CopyIn(multiRow, loopNum);
        if (isAsymmetrical) {
            ComputAsymmetric(smoothLocal, multiRow);
        } else {
            Compute(smoothLocal, multiRow);
        }
        CopyOut(multiRow, loopNum);
    }

    __aicore__ inline void SmoothCopyIn()
    {
        LocalTensor<xDtype> smoothLocal = smoothQueue.AllocTensor<xDtype>();
        if (isPad) {
            DataCopyParams copyParams{1, (uint16_t)(tilingData_.rowLen * sizeof(xDtype)), 0, 0};
            DataCopyPadParams padParams{true, 0, rightPadding, 0};
            DataCopyPad(smoothLocal, smoothGm, copyParams, padParams);
        } else {
            DataCopy(smoothLocal, smoothGm, tilingData_.rowLen);
        }
        smoothQueue.EnQue(smoothLocal);
    }

    __aicore__ inline void CopyIn(int32_t multiRow, int32_t loopNum)
    {
        LocalTensor<xDtype> inLocal = inQueue.AllocTensor<xDtype>();
        if (isPad) {
            DataCopyParams copyParams{(uint16_t)multiRow, (uint16_t)(tilingData_.rowLen * sizeof(xDtype)), 0, 0};
            DataCopyPadParams padParams{true, 0, rightPadding, 0};
            DataCopyPad(inLocal, inGm[loopNum * lenGMMultiRow], copyParams, padParams);
        } else {
            DataCopy(inLocal, inGm[loopNum * lenGMMultiRow], multiRow * tilingData_.rowLen);
        }
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute(const LocalTensor<float>& smoothLocal, int32_t multiRow)
    {
        uint32_t index = 0;
        LocalTensor<float> scaleLocal = scaleQueue.AllocTensor<float>();
        LocalTensor<xDtype> inLocal = inQueue.DeQue<xDtype>();
        LocalTensor<float> tempFp32 = tempCastUb.Get<float>();
        LocalTensor<yDtype> outLocal = outQueue.AllocTensor<yDtype>();
        AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();
        LocalTensor<int32_t> tempInt32 = fp32_buf_.Get<int32_t>();
        auto tempHalf = temp.ReinterpretCast<half>();

        for (int32_t i = 0; i < multiRow; i++) {
            index = i * sizeHalfLen;
            // x fp16->fp32
            Cast(tempFp32, inLocal[index], RoundMode::CAST_NONE, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            if (tilingData_.hasSmooth) {
                Mul(tempFp32, tempFp32, smoothLocal, tilingData_.rowLen);
                PipeBarrier<PIPE_V>();
            }
            Abs(temp, tempFp32, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            ReduceMaxInplace(temp, tilingData_.rowLen);
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
    }

    __aicore__ inline void ComputAsymmetric(const LocalTensor<float>& smoothLocal, int32_t multiRow)
    {
        uint32_t index = 0;
        float max_value, min_value;
        float scale, offset;
        float back_scale;
        LocalTensor<float> scaleLocal = scaleQueue.AllocTensor<float>();
        LocalTensor<float> offsetLocal = offsetQueue.AllocTensor<float>();
        LocalTensor<xDtype> inLocal = inQueue.DeQue<xDtype>();
        LocalTensor<float> tempFp32 = tempCastUb.Get<float>();
        LocalTensor<yDtype> outLocal = outQueue.AllocTensor<yDtype>();
        AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();
        LocalTensor<int32_t> tempInt32 = fp32_buf_.Get<int32_t>();
        auto tempHalf = temp.ReinterpretCast<half>();
        for (int32_t i = 0; i < multiRow; i++) {
            index = i * sizeHalfLen;
            // x fp16->fp32
            Cast(tempFp32, inLocal[index], RoundMode::CAST_NONE, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            if (tilingData_.hasSmooth) {
                Mul(tempFp32, tempFp32, smoothLocal, tilingData_.rowLen);
                PipeBarrier<PIPE_V>();
            }
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
            Cast(tempHalf, tempInt32, RoundMode::CAST_ROUND, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            Cast(outLocal[i * outAlignLen], tempHalf, RoundMode::CAST_TRUNC, tilingData_.rowLen);
        }
        outQueue.EnQue<yDtype>(outLocal);
        scaleQueue.EnQue<float>(scaleLocal);
        offsetQueue.EnQue<float>(offsetLocal);
        inQueue.FreeTensor(inLocal);
    }

    __aicore__ inline void CopyOut(int32_t multiRow, int32_t loopCount)
    {
        LocalTensor<yDtype> outLocal = outQueue.DeQue<yDtype>();
        LocalTensor<float> scaleLocal = scaleQueue.DeQue<float>();
        LocalTensor<float> offsetLocal;
        if (isAsymmetrical) {
            offsetLocal = offsetQueue.DeQue<float>();
        }
        DataCopyExtParams copyParams{(uint16_t)multiRow, (uint16_t)(tilingData_.rowLen * sizeof(yDtype)), 0, 0, 0};
        if constexpr (IsSameType<yDtype, int4b_t>::value) {
            copyParams.blockLen = copyParams.blockLen >> 1;
            uint32_t index = loopCount * lenGMMultiRow;
            DataCopyPad(outGm[index], outLocal, copyParams);
        } else {
            DataCopyPad(outGm[loopCount * lenGMMultiRow], outLocal, copyParams);
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
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue, scaleQueue, offsetQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> smooth_buf_;

    /* global memory address */
    GlobalTensor<xDtype> inGm, smoothGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> offsetGm;
    GlobalTensor<yDtype> outGm;
};
#endif
} // namespace DynamicQuantNDOpt
#endif // DYNAMIC_QUANT
