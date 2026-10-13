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
 * \file dynamic_quant_single_row.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_SINGLE_ROW_H
#define DYNAMIC_QUANT_SINGLE_ROW_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace DynamicQuantNDOpt {
using namespace AscendC;

template <typename xDtype, typename yDtype>
class DynamicQuantSingleRow {
public:
    __aicore__ inline DynamicQuantSingleRow(TPipe* pipe)
    {
        pPipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR workSpace,
        const DynamicQuantTilingData* __restrict tilingData)
    {
        tilingD_ = tilingData;
        blockIdx_ = GetBlockIdx();

        maxHandleNum_ = tilingD_->multiRowNumHeadCore * tilingD_->rowLen;
        uint32_t outQueLen =
            maxHandleNum_ * sizeof(yDtype) + tilingD_->multiRowNumHeadCore * sizeof(float) + BLOCK_SIZE;
        rowsAlignBlock_ =
            (tilingD_->multiRowNumHeadCore + BLOCK_ELEM_F32 - 1) / BLOCK_ELEM_F32 * BLOCK_ELEM_F32 * BLOCK_SIZE;
        uint32_t tmpBufLens = maxHandleNum_ * sizeof(float) + rowsAlignBlock_ * CONST_2;
        if (tilingD_->hasSmooth) {
            tmpBufLens = tmpBufLens + tilingD_->rowLen * sizeof(float);
            smoothF32Offset_ = maxHandleNum_ + rowsAlignBlock_ * CONST_2 / sizeof(float);
        }
        pPipe->InitBuffer(inQueue_, 1, maxHandleNum_ * sizeof(xDtype) * CONST_2);
        pPipe->InitBuffer(outQueue_, 1, outQueLen);
        pPipe->InitBuffer(tmpBufNew_, tmpBufLens);

        blkRowOffset_ = blockIdx_ * tilingD_->rowPerHeadCore;

        inGm.SetGlobalBuffer((__gm__ xDtype*)x + blkRowOffset_ * tilingD_->rowLen);
        outGm.SetGlobalBuffer((__gm__ yDtype*)y + blkRowOffset_ * tilingD_->rowLen);
        scaleGm.SetGlobalBuffer((__gm__ float*)scale + blkRowOffset_);

        if (tilingD_->hasSmooth) {
            smoothGm.SetGlobalBuffer((__gm__ xDtype*)smooth_scales);
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (IsSameType<xDtype, bfloat16_t>::value) {
            ProcessBf16();
        } else {
            if (tilingD_->hasSmooth) {
                ProcessBf16();
            } else {
                ProcessFp16();
            }
        }
    }

    __aicore__ inline void CopyInSmooth(LocalTensor<float>& smoothLocalFp32)
    {
        LocalTensor<xDtype> smoothLocal = smoothLocalFp32.template ReinterpretCast<xDtype>();
        DataCopyPadExtParams<xDtype> padParamsSmooth{false, 0, 0, 0};
        DataCopyExtParams copyInParamsSmooth{1, 1, 0, 0, 0};
        copyInParamsSmooth.blockLen = tilingD_->rowLen * sizeof(xDtype);
        DataCopyPad(smoothLocal[tilingD_->rowLen], smoothGm, copyInParamsSmooth, padParamsSmooth);

        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        Cast(smoothLocalFp32, smoothLocal[tilingD_->rowLen], RoundMode::CAST_NONE, tilingD_->rowLen);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessBf16()
    {
        LocalTensor<float> inputCast = tmpBufNew_.Get<float>();
        LocalTensor<int32_t> inputInt32 = inputCast.template ReinterpretCast<int32_t>();
        LocalTensor<float> temp = inputCast[maxHandleNum_];
        LocalTensor<float> temp1 = temp[rowsAlignBlock_ / sizeof(float)];

        LocalTensor<float> smoothLocalFp32;
        if (tilingD_->hasSmooth) {
            smoothLocalFp32 = inputCast[smoothF32Offset_];
            CopyInSmooth(smoothLocalFp32);
        }

        uint32_t blkRows = (blockIdx_ >= tilingD_->headCoreNum) ? tilingD_->rowPerTailCore : tilingD_->rowPerHeadCore;
        uint32_t loopCnt = (blkRows + tilingD_->multiRowNumHeadCore - 1) / tilingD_->multiRowNumHeadCore;
        uint32_t tailUbRows = blkRows - (loopCnt - 1) * tilingD_->multiRowNumHeadCore;

        DataCopyExtParams copyInParams{1, 1, 0, 0, 0};
        DataCopyPadExtParams<xDtype> padParams{false, 0, 0, 0};
        for (uint32_t idx = 0; idx < loopCnt; idx++) {
            uint32_t ubRows = (idx == loopCnt - 1) ? tailUbRows : tilingD_->multiRowNumHeadCore;
            uint32_t ubNums = ubRows * tilingD_->rowLen;

            LocalTensor<xDtype> inLocal = inQueue_.AllocTensor<xDtype>();
            copyInParams.blockLen = ubNums * sizeof(xDtype);
            DataCopyPad(inLocal[maxHandleNum_], inGm[idx * maxHandleNum_], copyInParams, padParams);

            inQueue_.EnQue(inLocal);
            inLocal = inQueue_.DeQue<xDtype>();
            LocalTensor<float> inLocalFp32 = inLocal.template ReinterpretCast<float>();

            Cast(inLocalFp32, inLocal[maxHandleNum_], RoundMode::CAST_NONE, ubNums);
            PipeBarrier<PIPE_V>();
            if (tilingD_->hasSmooth) {
                // [1, rowLen] -> [ubRows, rowLen]
                SetMaskCount();
                SetVectorMask<float, MaskMode::COUNTER>(tilingD_->rowLen);
                Copy<float, false>(
                    inputCast, smoothLocalFp32, AscendC::MASK_PLACEHOLDER, ubRows,
                    {1, 1, static_cast<uint16_t>(tilingD_->rowLen / BLOCK_ELEM_F32), 0});
                SetMaskNorm();
                ResetMask();
                PipeBarrier<PIPE_V>();
                Mul(inLocalFp32, inLocalFp32, inputCast, ubNums);
                PipeBarrier<PIPE_V>();
            }
            Abs(inputCast, inLocalFp32, ubNums);
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < ubRows; ++i) {
                ComputeReduceMax(inputCast[i * tilingD_->rowLen], tilingD_->rowLen);
            }
            PipeBarrier<PIPE_V>();
            WholeReduceMax(
                inputCast, inputCast, MASK_NUM_T32, ubRows, 1, 1, tilingD_->rowLen / MASK_NUM_T32 * MASK_BLK_STRIDE,
                ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();

            LocalTensor<yDtype> outLocal = outQueue_.AllocTensor<yDtype>();
            LocalTensor<float> outScaleLocal = outLocal[maxHandleNum_].template ReinterpretCast<float>();

            // max/127 = max*(1.0/127)
            Muls(outScaleLocal, inputCast, DYNAMIC_QUANT_FACTOR, ubRows);
            PipeBarrier<PIPE_V>();

            // 127/max
            Duplicate<float>(temp1, DYNAMIC_QUANT_MAX, ubRows);
            PipeBarrier<PIPE_V>();
            Div(temp1, temp1, inputCast, ubRows);
            PipeBarrier<PIPE_V>();

            // [ubRows] -> [ubRows, Block]
            int64_t blockCount = (ubRows + BLOCK_ELEM_F32 - 1) / BLOCK_ELEM_F32;
            Brcb(temp, temp1, blockCount, {1, MASK_BLK_STRIDE});
            PipeBarrier<PIPE_V>();
            // [ubRows, Block] -> [ubRows, tilingD_->rowLen]
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(tilingD_->rowLen);
            Copy<float, false>(
                inputCast, temp, AscendC::MASK_PLACEHOLDER, ubRows,
                {1, 0, static_cast<uint16_t>(tilingD_->rowLen / BLOCK_ELEM_F32), 1});
            SetMaskNorm();
            ResetMask();
            PipeBarrier<PIPE_V>();
            // 改成乘法 Div(inputCast, inLocalFp32, inputCast, ubNums);
            Mul(inputCast, inLocalFp32, inputCast, ubNums);
            PipeBarrier<PIPE_V>();

            inQueue_.FreeTensor(inLocal);

            Cast(inputInt32, inputCast, RoundMode::CAST_RINT, ubNums);
            PipeBarrier<PIPE_V>();
            SetDeqScale(static_cast<half>(1.0));
            PipeBarrier<PIPE_V>();
            Cast(inputCast.ReinterpretCast<half>(), inputInt32, RoundMode::CAST_ROUND, ubNums);
            PipeBarrier<PIPE_V>();
            Cast(outLocal, inputCast.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, ubNums);
            PipeBarrier<PIPE_V>();

            outQueue_.EnQue(outLocal);
            outLocal = outQueue_.DeQue<yDtype>();
            outScaleLocal = outLocal[maxHandleNum_].template ReinterpretCast<float>();

            DataCopyParams outScaleParams{1, static_cast<uint16_t>(ubRows * sizeof(float)), 0, 0};
            DataCopyPad(scaleGm[idx * tilingD_->multiRowNumHeadCore], outScaleLocal, outScaleParams);

            DataCopyParams copyOutParams{1, static_cast<uint16_t>(ubNums * sizeof(yDtype)), 0, 0};
            DataCopyPad(outGm[idx * maxHandleNum_], outLocal, copyOutParams);
            outQueue_.FreeTensor(outLocal);
        }
    }

    __aicore__ inline void ProcessFp16()
    {
        LocalTensor<float> inputCast = tmpBufNew_.Get<float>();
        LocalTensor<int32_t> inputInt32 = inputCast.template ReinterpretCast<int32_t>();
        LocalTensor<float> temp = inputCast[maxHandleNum_];
        LocalTensor<float> temp1 = temp[rowsAlignBlock_ / sizeof(float)];

        uint32_t blkRows = (blockIdx_ >= tilingD_->headCoreNum) ? tilingD_->rowPerTailCore : tilingD_->rowPerHeadCore;
        uint32_t loopCnt = (blkRows + tilingD_->multiRowNumHeadCore - 1) / tilingD_->multiRowNumHeadCore;
        uint32_t tailUbRows = blkRows - (loopCnt - 1) * tilingD_->multiRowNumHeadCore;

        DataCopyExtParams copyInParams{1, 1, 0, 0, 0};
        DataCopyPadExtParams<xDtype> padParams{false, 0, 0, 0};
        for (uint32_t idx = 0; idx < loopCnt; idx++) {
            uint32_t ubRows = (idx == loopCnt - 1) ? tailUbRows : tilingD_->multiRowNumHeadCore;
            uint32_t ubNums = ubRows * tilingD_->rowLen;

            LocalTensor<xDtype> inLocal = inQueue_.AllocTensor<xDtype>();
            copyInParams.blockLen = ubNums * sizeof(xDtype);
            DataCopyPad(inLocal, inGm[idx * maxHandleNum_], copyInParams, padParams);

            inQueue_.EnQue(inLocal);
            inLocal = inQueue_.DeQue<xDtype>();

            Cast(inputCast, inLocal, RoundMode::CAST_NONE, ubNums);
            PipeBarrier<PIPE_V>();
            Abs(inLocal, inLocal, ubNums);
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < ubRows; ++i) {
                ComputeReduceMax(inLocal[i * tilingD_->rowLen], tilingD_->rowLen);
            }
            PipeBarrier<PIPE_V>();
            WholeReduceMax(
                inLocal, inLocal, ELEM_PER_REP_HALF, ubRows, 1, 1,
                tilingD_->rowLen / ELEM_PER_REP_HALF * MASK_BLK_STRIDE, ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();

            LocalTensor<yDtype> outLocal = outQueue_.AllocTensor<yDtype>();
            LocalTensor<float> outScaleLocal = outLocal[maxHandleNum_].template ReinterpretCast<float>();

            Cast(outScaleLocal, inLocal, RoundMode::CAST_NONE, ubRows);
            PipeBarrier<PIPE_V>();

            // 127/max
            Duplicate<float>(temp1, DYNAMIC_QUANT_MAX, ubRows);
            PipeBarrier<PIPE_V>();
            Div(temp1, temp1, outScaleLocal, ubRows);
            PipeBarrier<PIPE_V>();

            Muls(outScaleLocal, outScaleLocal, DYNAMIC_QUANT_FACTOR, ubRows);
            PipeBarrier<PIPE_V>();

            LocalTensor<float> inLocalFp32 = inLocal.template ReinterpretCast<float>();

            int64_t blockCount = (ubRows + BLOCK_ELEM_F32 - 1) / BLOCK_ELEM_F32;
            Brcb(temp, temp1, blockCount, {1, MASK_BLK_STRIDE});
            PipeBarrier<PIPE_V>();
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(tilingD_->rowLen);
            Copy<float, false>(
                inLocalFp32, temp, AscendC::MASK_PLACEHOLDER, ubRows,
                {1, 0, static_cast<uint16_t>(tilingD_->rowLen / BLOCK_ELEM_F32), 1});
            SetMaskNorm();
            ResetMask();
            PipeBarrier<PIPE_V>();
            // 改成乘法 Div(inputCast, inputCast, inLocalFp32, ubNums);
            Mul(inputCast, inputCast, inLocalFp32, ubNums);
            PipeBarrier<PIPE_V>();

            inQueue_.FreeTensor(inLocal);

            Cast(inputInt32, inputCast, RoundMode::CAST_RINT, ubNums);
            PipeBarrier<PIPE_V>();
            SetDeqScale(static_cast<half>(1.0));
            PipeBarrier<PIPE_V>();
            Cast(inputCast.ReinterpretCast<half>(), inputInt32, RoundMode::CAST_ROUND, ubNums);
            PipeBarrier<PIPE_V>();
            Cast(outLocal, inputCast.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, ubNums);
            PipeBarrier<PIPE_V>();

            outQueue_.EnQue(outLocal);
            outLocal = outQueue_.DeQue<yDtype>();
            outScaleLocal = outLocal[maxHandleNum_].template ReinterpretCast<float>();

            DataCopyParams outScaleParams{1, static_cast<uint16_t>(ubRows * sizeof(float)), 0, 0};
            DataCopyPad(scaleGm[idx * tilingD_->multiRowNumHeadCore], outScaleLocal, outScaleParams);

            DataCopyParams copyOutParams{1, static_cast<uint16_t>(ubNums * sizeof(yDtype)), 0, 0};
            DataCopyPad(outGm[idx * maxHandleNum_], outLocal, copyOutParams);
            outQueue_.FreeTensor(outLocal);
        }
    }

    template <typename T>
    __aicore__ inline void ComputeReduceMax(const LocalTensor<T>& tempRes, int32_t calCount)
    {
        constexpr uint32_t elePerRepeat = REPEAT_BYTES / sizeof(T);
        uint32_t vectorCycles = calCount / elePerRepeat;
        uint32_t remainElements = calCount % elePerRepeat;

        BinaryRepeatParams repeatParams{1, 1, 1, 0, MASK_BLK_STRIDE, 0};

        if (vectorCycles > 0 && remainElements > 0) {
            Max(tempRes, tempRes, tempRes[vectorCycles * elePerRepeat], remainElements, 1, repeatParams);
            PipeBarrier<PIPE_V>();
        }

        if (vectorCycles > 1) {
            Max(tempRes, tempRes[elePerRepeat], tempRes, elePerRepeat, vectorCycles - 1, repeatParams);
            PipeBarrier<PIPE_V>();
        }
    }

private:
    constexpr static int32_t CONST_2 = 2;
    constexpr static int32_t REPEAT_BYTES = 256;
    constexpr static int32_t ELEM_PER_REP_HALF = 128;
    constexpr static int32_t MASK_NUM_T32 = 64;
    constexpr static int32_t MASK_BLK_STRIDE = 8;
    constexpr static float DYNAMIC_QUANT_FACTOR = 1.0 / 127.0;
    constexpr static float DYNAMIC_QUANT_MAX = 127.0;
    constexpr static int32_t BLOCK_ELEM_F32 = 8;
    constexpr static int32_t BLOCK_SIZE = 32;

    uint32_t blockIdx_ = 0;
    uint32_t blkRowOffset_ = 0;
    uint32_t maxHandleNum_ = 0;
    uint32_t smoothF32Offset_ = 0;
    uint32_t rowsAlignBlock_ = 0;

    TPipe* pPipe = nullptr;
    const DynamicQuantTilingData* tilingD_ = nullptr;

    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    AscendC::TBuf<> tmpBufNew_;

    /* global memory address */
    GlobalTensor<xDtype> inGm;
    GlobalTensor<xDtype> smoothGm;
    GlobalTensor<yDtype> outGm;
    GlobalTensor<float> scaleGm;
};
} // namespace DynamicQuantNDOpt
#endif // DYNAMIC_QUANT_SINGLE_ROW_H
