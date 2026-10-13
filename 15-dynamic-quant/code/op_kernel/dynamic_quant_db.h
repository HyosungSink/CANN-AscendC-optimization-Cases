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
 * \file dynamic_quant_db.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_DB_H
#define DYNAMIC_QUANT_DB_H

#include "dynamic_quant_base.h"

namespace DynamicQuantNDOpt {
using namespace AscendC;

#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
template <typename xDtype, typename yDtype>
class DynamicQuantDbOpt : public DynamicQuantBase
{
public:
    __aicore__ inline DynamicQuantDbOpt(TPipe* pipe)
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
        DuplicateConst();
        // copy smooth from GM to UB
        if (tilingData_.hasSmooth) {
            smoothLocal = smooth_buf_.Get<float>();
            SmoothCopyIn();
            smoothHalfLocal = smoothQueue.DeQue<xDtype>();
            PipeBarrier<PIPE_V>();
            Cast(smoothLocal, smoothHalfLocal, RoundMode::CAST_NONE, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
        }
        CopyIn(multiRowNum, 0);
        for (uint32_t i = 1; i < loopCnt; i++) {
            CopyIn(multiRowNum, i);
            ProcessMultiRow(smoothLocal, multiRowNum, i - 1);
        }

        if (remainRow > 0) {
            CopyIn(remainRow, loopCnt);
        }

        ProcessMultiRow(smoothLocal, multiRowNum, loopCnt - 1);

        if (remainRow > 0) {
            ProcessMultiRow(smoothLocal, remainRow, loopCnt);
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
            pPipe->InitBuffer(offsetBuf, sizeFloatLen * sizeof(float));
        }
        pPipe->InitBuffer(inQueue, DOUBLE_BUFFER_NUM, lenMultiRow * sizeof(xDtype));
        pPipe->InitBuffer(outBuf, outLen * sizeof(yDtype));
        pPipe->InitBuffer(scaleBuf, sizeFloatLen * sizeof(float));
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

    __aicore__ inline void CopyIn(uint32_t multiRow, uint32_t loopNum)
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

    __aicore__ inline void ProcessMultiRow(const LocalTensor<float>& smoothLocal, uint32_t multiRow, uint32_t loopNum)
    {
        if (isAsymmetrical) {
            ProcessAsymmetric(smoothLocal, multiRow, loopNum);
        } else {
            ProcessSymmetric(smoothLocal, multiRow, loopNum);
        }
    }

    __aicore__ inline void ProcessAsymmetric(const LocalTensor<float>& smoothLocal, uint32_t multiRow, uint32_t loopNum)
    {
        float max_value, min_value, back_scale;
        float scale, offset;

        LocalTensor<xDtype> inLocal = inQueue.DeQue<xDtype>();
        LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
        LocalTensor<float> offsetLocal = offsetBuf.Get<float>();
        LocalTensor<float> tempCast = tempCastUb.Get<float>();
        LocalTensor<yDtype> outLocal = outBuf.AllocTensor<yDtype>();
        LocalTensor<float> temp = fp32_buf_.Get<float>();
        AscendC::LocalTensor<float> tempInt32 = fp32_buf_.Get<float>();
        auto tempHalfCast = temp.ReinterpretCast<half>();

        event_t event_mte3_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(event_mte3_s);
        event_t event_mte3_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(event_mte3_v);
        for (uint32_t i = 0; i < multiRow; i++) {
            ComputeRowMinMax(i, inLocal, tempCast, temp, smoothLocal, max_value, min_value);
            
            if constexpr (IsSameType<yDtype, int4b_t>::value) {
                scale = GetMax((max_value - min_value) / DYNAMIC_QUANT_INT4_SCALE, DYNAMIC_QUANT_EPSINON);
                offset = DYNAMIC_QUANT_INT4_OFFSET - max_value / scale;
            } else {
                scale = GetMax((max_value - min_value) / DYNAMIC_QUANT_INT8_SCALE, DYNAMIC_QUANT_EPSINON);
                offset = DYNAMIC_QUANT_INT8_OFFSET - max_value / scale;
            }
            back_scale = 1 / scale;
            if (unlikely(i == 0)) {
                WaitFlag<HardEvent::MTE3_S>(event_mte3_s);
            }
            scaleLocal.SetValue(i, scale);
            offsetLocal.SetValue(i, offset);
            Muls(tempCast, tempCast, back_scale, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            Adds(tempCast, tempCast, offset, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            Cast(tempInt32, tempCast, RoundMode::CAST_RINT, tilingData_.rowLen);
            SetDeqScale(static_cast<half>(1.0));
            PipeBarrier<PIPE_V>();
            Cast(tempHalfCast, tempInt32, RoundMode::CAST_ROUND, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            if (unlikely(i == 0)) {
                WaitFlag<HardEvent::MTE3_V>(event_mte3_v);
            }
            Cast(outLocal[i * outAlignLen], tempHalfCast, RoundMode::CAST_TRUNC, tilingData_.rowLen);
        }
        inQueue.FreeTensor(inLocal);

        CopyOutData(multiRow, loopNum, outLocal, scaleLocal, &offsetLocal);
    }

    __aicore__ inline void ComputeRowMinMax(uint32_t rowIndex, const LocalTensor<xDtype>& inLocal, 
                                           LocalTensor<float>& tempCast, LocalTensor<float>& temp, 
                                           const LocalTensor<float>& smoothLocal, float& max_value, 
                                           float& min_value)
    {
        // cast to fp32
        Cast(tempCast, inLocal[rowIndex * sizeHalfLen], RoundMode::CAST_NONE, tilingData_.rowLen);
        PipeBarrier<PIPE_V>();
        if (tilingData_.hasSmooth) {
            Mul(tempCast, tempCast, smoothLocal, sizeHalfLen);
            PipeBarrier<PIPE_V>();
        }
        ReduceMax(temp, tempCast, temp, tilingData_.rowLen, false);
        event_t event_v_s_max = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s_max);
        WaitFlag<HardEvent::V_S>(event_v_s_max);
        max_value = temp.GetValue(0);
        ReduceMin(temp, tempCast, temp, tilingData_.rowLen, false);
        event_t event_v_s_min = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s_min);
        WaitFlag<HardEvent::V_S>(event_v_s_min);
        min_value = temp.GetValue(0);
    }

    __aicore__ inline void CopyOutData(uint32_t multiRow, uint32_t loopNum, const LocalTensor<yDtype>& outLocal,
                                        const LocalTensor<float>& scaleLocal, const LocalTensor<float>* offsetLocal)
    {
        event_t event_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event_v_mte3);
        WaitFlag<HardEvent::V_MTE3>(event_v_mte3);
        DataCopyExtParams copyOutParams{(uint16_t)multiRow, (uint16_t)(tilingData_.rowLen * sizeof(yDtype)), 0, 0, 0};
        if constexpr (IsSameType<yDtype, int4b_t>::value) {
            copyOutParams.blockLen = copyOutParams.blockLen >> 1;
            uint32_t index = loopNum * lenGMMultiRow;
            DataCopyPad(outGm[index], outLocal, copyOutParams);
        } else {
            DataCopyPad(outGm[loopNum * lenGMMultiRow], outLocal, copyOutParams);
        }
        DataCopyParams copyParams{1, (uint16_t)(multiRow * sizeof(float)), 0, 0};
        DataCopyPad(scaleGm[loopNum * multiRowNum], scaleLocal, copyParams);
        if (offsetLocal != nullptr) {
            DataCopyPad(offsetGm[loopNum * multiRowNum], *offsetLocal, copyParams);
        }
    }

    __aicore__ inline void ProcessSymmetric(const LocalTensor<float>& smoothLocal, uint32_t multiRow, uint32_t loopNum)
    {
        uint32_t index = 0;
        LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
        LocalTensor<xDtype> inLocal = inQueue.DeQue<xDtype>();
        LocalTensor<float> tempCast = tempCastUb.Get<float>();
        LocalTensor<yDtype> outLocal = outBuf.template Get<yDtype>();
        AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();
        AscendC::LocalTensor<int16_t> tempInt16 = fp32_buf_.Get<int16_t>();
        auto tempHalf = temp.ReinterpretCast<half>();

        event_t event_mte3_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(event_mte3_v);
        for (int32_t i = 0; i < multiRow; i++) {
            index = i * sizeHalfLen;
            ComputeRowMax(inLocal[index], tempCast, temp, smoothLocal);
            QuantizeRow(tempCast, temp, tempInt16, tempHalf, scaleLocal, outLocal, i, event_mte3_v);
        }
        inQueue.FreeTensor(inLocal);

        CopyOutData(multiRow, loopNum, outLocal, scaleLocal, nullptr);
    }

    __aicore__ inline void ComputeRowMax(const LocalTensor<xDtype>& inLocal, LocalTensor<float>& tempCast,
                                         LocalTensor<float>& temp, const LocalTensor<float>& smoothLocal)
    {
        if constexpr (IsSameType<xDtype, bfloat16_t>::value) {
            Cast(tempCast, inLocal, RoundMode::CAST_NONE, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            if (tilingData_.hasSmooth) {
                Mul(tempCast, tempCast, smoothLocal, sizeHalfLen);
                PipeBarrier<PIPE_V>();
            }
            Abs(temp, tempCast, tilingData_.rowLen);
            PipeBarrier<PIPE_V>();
            ReduceMaxInplace(temp, tilingData_.rowLen);
        } else {
            Cast(tempCast, inLocal, RoundMode::CAST_NONE, tilingData_.rowLen);
            if (tilingData_.hasSmooth) {
                PipeBarrier<PIPE_V>();
                Mul(tempCast, tempCast, smoothLocal, sizeHalfLen);
                PipeBarrier<PIPE_V>();
                Abs(temp, tempCast, tilingData_.rowLen);
                PipeBarrier<PIPE_V>();
                ReduceMaxInplace(temp, tilingData_.rowLen);
            } else {
                Abs(inLocal, inLocal, tilingData_.rowLen);
                PipeBarrier<PIPE_V>();
                ReduceMaxInplace(inLocal, tilingData_.rowLen);
                PipeBarrier<PIPE_V>();
                Cast(temp, inLocal, RoundMode::CAST_NONE, tilingData_.rowLen);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void QuantizeRow(LocalTensor<float>& tempCast, LocalTensor<float>& temp,
                                        LocalTensor<int16_t>& tempInt16, LocalTensor<half>& tempHalf,
                                        LocalTensor<float>& scaleLocal, LocalTensor<yDtype>& outLocal,
                                        int32_t rowIndex, event_t event_mte3_v)
    {
        Brcb(brcbTmp, temp, 1, {1, 8});
        PipeBarrier<PIPE_V>();
        Div(quantScaleTmp, constScale, brcbTmp, MAX_VALUE_NUM);
        PipeBarrier<PIPE_V>();
        Mul(tempCast, tempCast, quantScaleTmp, 64, (tilingData_.rowLen + 63) >> 6, {1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        Cast(tempInt16, tempCast, RoundMode::CAST_RINT, tilingData_.rowLen);
        PipeBarrier<PIPE_V>();
        Cast(tempHalf, tempInt16, RoundMode::CAST_ROUND, tilingData_.rowLen);
        PipeBarrier<PIPE_V>();
        if (unlikely(rowIndex == 0)) {
            WaitFlag<HardEvent::MTE3_V>(event_mte3_v);
        }
        uint64_t mask[DOUBLE] = { 1ULL<<(rowIndex & 7), 0ULL};
        Mul(scaleLocal[rowIndex & ~7], brcbTmp, constInvScale, mask, 1, {1, 1, 0, 8, 8, 0});
        Cast(outLocal[rowIndex * outAlignLen], tempHalf, RoundMode::CAST_TRUNC, tilingData_.rowLen);
    }

private:
    /* ascendc variable */
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> smoothQueue;
    TBuf<> outBuf;
    TBuf<> scaleBuf;
    TBuf<> offsetBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> smooth_buf_;

    /* global memory address */
    GlobalTensor<xDtype> inGm, smoothGm;
    GlobalTensor<yDtype> outGm;
    GlobalTensor<float> scaleGm, offsetGm;
};
#endif
} // namespace DynamicQuantNDOpt
#endif // DYNAMIC_QUANT_DB_H