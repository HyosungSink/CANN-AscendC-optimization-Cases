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
 * \file dynamic_quant_multi_row.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_MULTI_ROW_H
#define DYNAMIC_QUANT_MULTI_ROW_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace DynamicQuantNDOpt {
using namespace AscendC;

template <typename xDtype, typename yDtype>
class DynamicQuantMultiRow {
public:
    __aicore__ inline DynamicQuantMultiRow(TPipe* pipe)
    {
        pPipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR workSpace,
        const DynamicQuantTilingData* __restrict tilingData)
    {
        blockIdx_ = GetBlockIdx();
        ParseTilingData(tilingData);

        SetupGlobalBuffers(x, smooth_scales, scale);
        AllocateLocalBuffers();
        SetupOutputBufferAndConstants(y);
    }

    __aicore__ inline void Process()
    {
        if (hasSmooth) {
            CopyInSmooth();
        }
        ProcessQuant();
    }

private:
    constexpr static int32_t DIMENSION_2 = 2;

private:
    template <typename T1, typename T2>
    __aicore__ inline auto CeilDiv(T1 x, T2 y)
    {
        return y != 0 ? (x + y - 1) / y : x;
    };

    template <typename T1, typename T2>
    __aicore__ inline auto CeilAlign(T1 x, T2 y)
    {
        return y != 0 ? CeilDiv(x, y) * y : x;
    };

    __aicore__ inline void ParseTilingData(const DynamicQuantTilingData* tilingData)
    {
        rowLen = tilingData->rowLen;
        headCoreNum = tilingData->headCoreNum;
        headPerCoreRow = tilingData->rowPerHeadCore;
        singleLoopRows = tilingData->multiRowNumHeadCore;
        hasSmooth = tilingData->hasSmooth;
        ubSize = tilingData->ubSize;

        if (blockIdx_ < headCoreNum) {
            totalRows = headPerCoreRow;
            startRow = blockIdx_ * headPerCoreRow;
        } else {
            totalRows = headPerCoreRow - 1;
            startRow = blockIdx_ * totalRows + headCoreNum;
        }
        singleLoopEle = singleLoopRows * rowLen;
    }

    __aicore__ inline void SetupGlobalBuffers(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR scale)
    {
        inGm.SetGlobalBuffer((__gm__ xDtype*)x + static_cast<int64_t>(startRow) * rowLen);
        scaleGm.SetGlobalBuffer((__gm__ float*)scale + startRow);
        if (hasSmooth) {
            smoothGm.SetGlobalBuffer((__gm__ xDtype*)smooth_scales);
        }
    }

    __aicore__ inline void AllocateLocalBuffers()
    {
        pPipe->InitBuffer(bufQueue, ubSize - UB_RESERVED_SIZE);
        LocalTensor<uint8_t> tmp = bufQueue.Get<uint8_t>();
        uint32_t offsetBytes = 0;

        uint32_t singleLoopEleAlign = CeilAlign(singleLoopEle, ALIGN_FACTOR_32);
        uint32_t rowLenAlign = CeilAlign(rowLen, ALIGN_FACTOR_32);
        inLocal = tmp.ReinterpretCast<xDtype>();
        offsetBytes += singleLoopEleAlign * sizeof(xDtype);
        if (hasSmooth) {
            smoothLocal = tmp[offsetBytes].ReinterpretCast<float>();
            offsetBytes += rowLenAlign * sizeof(float);
        }

        castTmp = tmp[offsetBytes].ReinterpretCast<float>();
        offsetBytes += singleLoopEleAlign * sizeof(float);
        absTmp = tmp[offsetBytes].ReinterpretCast<float>();
        offsetBytes += singleLoopEleAlign * sizeof(float);

        outLocal = tmp[offsetBytes].ReinterpretCast<yDtype>();
        offsetBytes += singleLoopEleAlign * sizeof(yDtype);
        scaleLocal = tmp[offsetBytes].ReinterpretCast<float>();
        offsetBytes += SCALE_BUFFER_SIZE * sizeof(float);
        scaleTmp = tmp[offsetBytes].ReinterpretCast<float>();
        offsetBytes += SCALE_BUFFER_SIZE * sizeof(float);
        quantScaleTmp = tmp[offsetBytes].ReinterpretCast<float>();
        offsetBytes += SCALE_BUFFER_SIZE * sizeof(float);

        constScale = tmp[offsetBytes].ReinterpretCast<float>();
        offsetBytes += CONST_SCALE_ELEMENTS * sizeof(float);
        constInvScale = tmp[offsetBytes].ReinterpretCast<float>();
    }

    __aicore__ inline void SetupOutputBufferAndConstants(GM_ADDR y)
    {
        if constexpr (IsSameType<yDtype, int4b_t>::value) {
            outGm.SetGlobalBuffer((__gm__ yDtype*)y + (static_cast<int64_t>(startRow) * rowLen >> 1));
            Duplicate<float>(constScale, DYNAMIC_QUANT_INT4_SYM_SCALE, CONST_SCALE_ELEMENTS);
            Duplicate<float>(constInvScale, float(1) / DYNAMIC_QUANT_INT4_SYM_SCALE, CONST_SCALE_ELEMENTS);
        } else {
            outGm.SetGlobalBuffer((__gm__ yDtype*)y + static_cast<int64_t>(startRow) * rowLen);
            Duplicate<float>(constScale, DYNAMIC_QUANT_INT8_SYM_SCALE, CONST_SCALE_ELEMENTS);
            Duplicate<float>(constInvScale, float(1) / DYNAMIC_QUANT_INT8_SYM_SCALE, CONST_SCALE_ELEMENTS);
        }
    }

    __aicore__ inline void CopyInSmooth()
    {
        LocalTensor<xDtype> inSmooth =
            smoothLocal.template ReinterpretCast<xDtype>()[CeilAlign(rowLen, ALIGN_FACTOR_16)];
        DataCopyExtParams copyInParamsSmooth{1, static_cast<uint32_t>(rowLen * sizeof(xDtype)), 0, 0, 0};
        DataCopyPadExtParams<xDtype> padParamsSmooth{false, 0, 0, 0};
        DataCopyPad(inSmooth, smoothGm, copyInParamsSmooth, padParamsSmooth);

        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        Cast(smoothLocal, inSmooth, RoundMode::CAST_NONE, rowLen);
    }

    __aicore__ inline void ProcessQuant()
    {
        uint32_t loopCnt = CeilDiv(totalRows, singleLoopRows);
        uint32_t srcShape1[DIMENSION_2] = {singleLoopRows, 1};
        uint32_t dstShape1[DIMENSION_2] = {singleLoopRows, rowLen};
        DataCopyExtParams copyInParams{1, 1, 0, 0, 0};
        DataCopyPadExtParams<xDtype> padParams{false, 0, 0, 0};
        event_t event_mte3_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        uint32_t ubRows = singleLoopRows;
        uint32_t ubNums = singleLoopEle;
        copyInParams.blockLen = singleLoopEle * sizeof(xDtype);
        for (uint32_t idx = 0; idx < loopCnt; idx++) {
            if (idx == loopCnt - 1) {
                ubRows = totalRows - idx * singleLoopRows;
                ubNums = ubRows * rowLen;
                srcShape1[0] = ubRows;
                dstShape1[0] = ubRows;
                copyInParams.blockLen = ubNums * sizeof(xDtype);
            }
            CopyInAndApplySmooth(inGm[idx * singleLoopEle], copyInParams, padParams, ubRows, ubNums);
            ComputeQuantization(ubRows, ubNums, srcShape1, dstShape1);
            QuantizeAndWriteOut(idx, ubRows, ubNums, event_mte3_v);
        }
    }

    __aicore__ inline void CopyInAndApplySmooth(
        GlobalTensor<xDtype> srcGm, DataCopyExtParams& copyInParams, DataCopyPadExtParams<xDtype>& padParams,
        uint32_t ubRows, uint32_t ubNums)
    {
        DataCopyPad(inLocal, srcGm, copyInParams, padParams);

        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        Cast(castTmp, inLocal, RoundMode::CAST_NONE, ubNums);
        PipeBarrier<PIPE_V>();
        event_t eventId1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId1);
        WaitFlag<HardEvent::V_MTE2>(eventId1);

        if (hasSmooth) {
            if (ubRows > 1) {
                uint32_t srcShape[DIMENSION_2] = {1, rowLen};
                uint32_t dstShape[DIMENSION_2] = {ubRows, rowLen};
                BroadCast<float, DIMENSION_2, 0>(absTmp, smoothLocal, dstShape, srcShape);
                PipeBarrier<PIPE_V>();
                Mul(castTmp, castTmp, absTmp, ubNums);
            } else {
                Mul(castTmp, castTmp, smoothLocal, ubNums);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeQuantization(
        uint32_t ubRows, uint32_t ubNums, uint32_t srcShape1[DIMENSION_2], uint32_t dstShape1[DIMENSION_2])
    {
        Abs(absTmp, castTmp, ubNums);
        PipeBarrier<PIPE_V>();
        ComputeReduceMax(ubRows);
        PipeBarrier<PIPE_V>();

        Div(quantScaleTmp, constScale, scaleTmp, BLOCK_SIZE_64, CeilDiv(ubRows, BLOCK_SIZE_64),
            {1, 0, 1, STRIDE_8, 0, STRIDE_8});
        PipeBarrier<PIPE_V>();

        BroadCast<float, DIMENSION_2, 1>(absTmp, quantScaleTmp, dstShape1, srcShape1);
        PipeBarrier<PIPE_V>();

        Mul(absTmp, castTmp, absTmp, ubNums);
        PipeBarrier<PIPE_V>();
        Cast(absTmp.ReinterpretCast<int16_t>(), absTmp, RoundMode::CAST_RINT, ubNums);
        PipeBarrier<PIPE_V>();
        Cast(absTmp.ReinterpretCast<half>(), absTmp.ReinterpretCast<int16_t>(), RoundMode::CAST_ROUND, ubNums);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void QuantizeAndWriteOut(uint32_t idx, uint32_t ubRows, uint32_t ubNums, event_t event_mte3_v)
    {
        if (idx != 0) {
            SetFlag<HardEvent::MTE3_V>(event_mte3_v);
            WaitFlag<HardEvent::MTE3_V>(event_mte3_v);
        }
        Mul(scaleLocal, scaleTmp, constInvScale, BLOCK_SIZE_64, CeilDiv(ubRows, BLOCK_SIZE_64),
            {1, 1, 0, STRIDE_8, STRIDE_8, 0});
        Cast(outLocal, absTmp.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, ubNums);
        PipeBarrier<PIPE_V>();
        event_t event_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event_v_mte3);
        WaitFlag<HardEvent::V_MTE3>(event_v_mte3);
        DataCopyParams outScaleParams{1, static_cast<uint16_t>(ubRows * sizeof(float)), 0, 0};
        DataCopyPad(scaleGm[idx * singleLoopRows], scaleLocal, outScaleParams);

        DataCopyParams copyOutParams{1, static_cast<uint16_t>(ubNums * sizeof(yDtype)), 0, 0};
        if constexpr (IsSameType<yDtype, int4b_t>::value) {
            copyOutParams.blockLen = copyOutParams.blockLen >> 1;
        }
        DataCopyPad(outGm[idx * singleLoopEle], outLocal, copyOutParams);
    }
    __aicore__ inline void ComputeReduceMax(uint32_t ubRows)
    {
        LocalTensor<float> calcTensor;
        BinaryRepeatParams repeatParams{1, 1, 1, 0, STRIDE_8, 0};

        for (uint32_t i = 0; i < ubRows; i++) {
            calcTensor = absTmp[i * rowLen];
            uint32_t repeatTimes = rowLen / BLOCK_SIZE_64;
            uint32_t remainElements = rowLen % BLOCK_SIZE_64;

            if (remainElements > 0) {
                Max(calcTensor, calcTensor, calcTensor[repeatTimes * BLOCK_SIZE_64], remainElements, 1, repeatParams);
                PipeBarrier<PIPE_V>();
            }

            if (repeatTimes > 1) {
                Max(calcTensor, calcTensor[BLOCK_SIZE_64], calcTensor, BLOCK_SIZE_64, repeatTimes - 1, repeatParams);
                PipeBarrier<PIPE_V>();
            }
        }
        WholeReduceMax(scaleTmp, absTmp, BLOCK_SIZE_64, ubRows, 1, 1, rowLen / STRIDE_8, ReduceOrder::ORDER_ONLY_VALUE);
    }

private:
    constexpr static int32_t UB_RESERVED_SIZE = 1024;
    constexpr static int32_t ALIGN_FACTOR_32 = 32;
    constexpr static int32_t ALIGN_FACTOR_16 = 16;
    constexpr static int32_t SCALE_BUFFER_SIZE = 192;
    constexpr static int32_t CONST_SCALE_ELEMENTS = 8;
    constexpr static int32_t BLOCK_SIZE_64 = 64;
    constexpr static int32_t STRIDE_8 = 8;

    uint32_t blockIdx_ = 0;
    uint32_t rowLen = 0;
    uint32_t headCoreNum = 0;    // head核数量
    uint32_t headPerCoreRow = 0; // head核处理的行数
    uint32_t singleLoopRows = 0; // 每次ub最多载入的行数
    uint32_t startRow = 0;       // 当前核开始处理的行数
    uint32_t totalRows = 0;      // 当前核计算的总行数
    bool hasSmooth = false;
    uint32_t ubSize = 0;

    uint32_t singleLoopEle = 0;

    TPipe* pPipe = nullptr;

    TBuf<QuePosition::VECCALC> bufQueue;

    /* global memory address */
    GlobalTensor<xDtype> inGm;
    GlobalTensor<xDtype> smoothGm;
    GlobalTensor<yDtype> outGm;
    GlobalTensor<float> scaleGm;
    LocalTensor<xDtype> inLocal;
    LocalTensor<float> smoothLocal;
    LocalTensor<yDtype> outLocal;
    LocalTensor<float> castTmp, absTmp, scaleTmp, quantScaleTmp, brcbQuantScaleTmp;
    LocalTensor<float> scaleLocal;
    LocalTensor<float> constScale, constInvScale;
};
} // namespace DynamicQuantNDOpt
#endif // DYNAMIC_QUANT_MULTI_ROW_H
