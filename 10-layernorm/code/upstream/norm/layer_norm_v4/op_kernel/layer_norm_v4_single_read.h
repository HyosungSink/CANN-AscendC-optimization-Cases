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
 * \file layer_norm_v4_single_read.h
 * \brief
 */

#ifndef LAYER_NORM_V4_SINGLE_READ_H
#define LAYER_NORM_V4_SINGLE_READ_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"

namespace LayerNormV4 {
using namespace AscendC;

constexpr int32_t SINGLE_CONST_TWO = 2;
constexpr int32_t SINGLE_CONST_THREE = 3;
constexpr int32_t SINGLE_CONST_FOUR = 4;
constexpr int32_t SINGLE_CONST_32 = 32;

template <typename Tfm, typename Tweight>
class LayerNormV4SingleRead {
public:
    __aicore__ inline LayerNormV4SingleRead()
    {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
        const LayerNormV4TilingDataSingleRead* __restrict tilingData)
    {
        // load tiling data
        colSize = tilingData->colSize;
        rowSize = tilingData->rowSize;
        numBlocks = tilingData->numBlocks;
        eps = tilingData->eps;
        rowAlign = tilingData->rowAlign;
        nRow = tilingData->nRow;
        coefficient = tilingData->coefficient;
        tailNRow = tilingData->tailNRow;
        tailLoop = tilingData->tailLoop;
        loopCount = tilingData->loopCount;
        tileLength = tilingData->tileLength;
        nullptrGamma = tilingData->nullptrGamma;
        blockLength = tilingData->blockLength;
        nullptrBeta = tilingData->nullptrBeta;

        // calculate xGm and paramGm offset and size
        uint32_t xGmOffset = 0;
        uint32_t xGmSize = 0;
        uint32_t paramGmOffset = 0;
        uint32_t paramGmSize = 0;
        // calculate xGm offset
        if (GetBlockIdx() < tailLoop) {
            xGmOffset = (loopCount + 1) * nRow * rowSize * GetBlockIdx();
        } else {
            xGmOffset = loopCount * nRow * rowSize * GetBlockIdx() + nRow * rowSize * tailLoop;
        }
        // calculate xGm size
        if (GetBlockIdx() < tailLoop) {
            xGmSize = (loopCount + 1) * blockLength;
        } else if (GetBlockIdx() < numBlocks - 1) {
            xGmSize = loopCount * blockLength;
        } else {
            xGmSize = loopCount * blockLength + tailNRow * rowSize;
        }
        // calculate paramGm offset
        if (GetBlockIdx() < tailLoop) {
            paramGmOffset = (loopCount + 1) * nRow * GetBlockIdx();
        } else {
            paramGmOffset = loopCount * nRow * GetBlockIdx() + nRow * tailLoop;
        }
        // calculate paramGm size
        if (GetBlockIdx() < tailLoop) {
            paramGmSize = (loopCount + 1) * nRow;
        } else if (GetBlockIdx() < numBlocks - 1) {
            paramGmSize = loopCount * nRow;
        } else {
            paramGmSize = loopCount * nRow + tailNRow;
        }

        // set global buffer
        xGm.SetGlobalBuffer((__gm__ Tfm*)x + xGmOffset, xGmSize);
        yGm.SetGlobalBuffer((__gm__ Tfm*)y + xGmOffset, xGmSize);
        meanGm.SetGlobalBuffer((__gm__ float*)mean + paramGmOffset, paramGmSize);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + paramGmOffset, paramGmSize);
        gammaGm.SetGlobalBuffer((__gm__ Tweight*)gamma, rowSize);
        betaGm.SetGlobalBuffer((__gm__ Tweight*)beta, rowSize);

        // pipe init buffer
        pipe.InitBuffer(inQueueX, 1, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueMean, 1, SINGLE_CONST_32 * sizeof(float));
        pipe.InitBuffer(outQueueRstd, 1, SINGLE_CONST_32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t Count = loopCount;
        if (GetBlockIdx() < tailLoop) {
            Count += 1;
        }
        // do main baisc block
        for (uint32_t loopIdx = 0; loopIdx < Count; ++loopIdx) {
            uint32_t currentBlockOffset = loopIdx * blockLength;
            uint32_t currentParamOffset = loopIdx * nRow;
            ProcessBasicBlock(nRow, currentBlockOffset, currentParamOffset);
        }

        // do tail baisc block
        if (tailNRow > 0 && GetBlockIdx() == (numBlocks - 1)) {
            uint32_t currentBlockOffset = Count * blockLength;
            uint32_t currentParamOffset = Count * nRow;
            ProcessBasicBlock(tailNRow, currentBlockOffset, currentParamOffset);
        }
    }

private:
    __aicore__ inline void ProcessBasicBlock(uint32_t nRow, uint32_t currentBlockOffset, uint32_t currentParamOffset)
    {
        // allocate local tensor
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> meanLocal = outQueueMean.AllocTensor<float>();
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();

        // load xGm to xLocal
        {
            DataCopyExtParams dataCopyParams;
            DataCopyPadExtParams<Tfm> padParams;
            dataCopyParams.blockCount = nRow;
            dataCopyParams.blockLen = rowSize * sizeof(Tfm);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            padParams.isPad = false;
            DataCopyPad(
                xLocal.ReinterpretCast<Tfm>()[(sizeof(Tfm) == SINGLE_CONST_TWO) * tileLength], xGm[currentBlockOffset], dataCopyParams,
                padParams);
        }
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<float>();

        // cast xLocal to float
        if (sizeof(Tfm) == SINGLE_CONST_TWO) {
            Cast(xLocal, xLocal.ReinterpretCast<Tfm>()[tileLength], RoundMode::CAST_NONE, tileLength);
            PipeBarrier<PIPE_V>();
        }

        // calculate x * coefficient
        SetMaskNorm();
        Muls(yLocal, xLocal, coefficient, tileLength);
        PipeBarrier<PIPE_V>();

        // calculate mean row-by-row
        for (uint32_t rowIdx = 0; rowIdx < nRow; ++rowIdx) {
            uint32_t currentRowOffset = rowIdx * rowAlign;
            ReduceSum(yLocal[currentRowOffset], yLocal[currentRowOffset], yLocal[currentRowOffset], rowSize);
            acc_val = GetAccVal();
            value = *reinterpret_cast<float*>(&acc_val);
            meanLocal.SetValue(rowIdx, static_cast<float>(value));
            Adds(yLocal[currentRowOffset], xLocal[currentRowOffset], -value, rowSize);
        }

        // store meanLocal to meanGm
        {
            event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
            WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = nRow * sizeof(float);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(meanGm[currentParamOffset], meanLocal, dataCopyParams);
        }
        outQueueMean.FreeTensor(meanLocal);

        // calculate square and muls coefficient
        PipeBarrier<PIPE_V>();
        Mul(xLocal, yLocal, yLocal, tileLength);
        PipeBarrier<PIPE_V>();
        Muls(xLocal, xLocal, coefficient, tileLength);
        PipeBarrier<PIPE_V>();

        // set load weight data copy pad params
        DataCopyExtParams dataCopyParams;
        DataCopyPadExtParams<Tweight> padParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = rowSize * sizeof(Tweight);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        padParams.isPad = false;

        // process rstd row-by-row
        for (uint32_t rowIdx = 0; rowIdx < nRow; ++rowIdx) {
            uint32_t currentRowOffset = rowIdx * rowAlign;
            ReduceSum(xLocal[currentRowOffset], xLocal[currentRowOffset], xLocal[currentRowOffset], rowSize);
            acc_val = GetAccVal();
            value = *reinterpret_cast<float*>(&acc_val);
            float rstdValue = static_cast<float>(1.0) / sqrt(value + static_cast<float>(eps));
            rstdLocal.SetValue(rowIdx, rstdValue);

            // the 0th row has been processed, load gamma to the 0th row
            if (!nullptrGamma && rowIdx == 0) {
                event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                DataCopyPad(
                    xLocal.ReinterpretCast<Tweight>()[(sizeof(Tweight) == SINGLE_CONST_TWO) * rowAlign], gammaGm, dataCopyParams,
                    padParams);
            }

            // the 1st row has been processed, load gamma to the 1st row
            if (!nullptrBeta && rowIdx == 1) {
                event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                DataCopyPad(
                    xLocal.ReinterpretCast<Tweight>()[(sizeof(Tweight) == SINGLE_CONST_TWO) * SINGLE_CONST_TWO * rowAlign + rowAlign], betaGm,
                    dataCopyParams, padParams);
            }

            Muls(yLocal[currentRowOffset], yLocal[currentRowOffset], rstdValue, rowSize);

            // if sizeof(Tweight) == 2, wait for gamma loaded, cast gamma to float
            if (!nullptrGamma && rowIdx == 0 && sizeof(Tweight) == SINGLE_CONST_TWO) {
                event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                Cast(xLocal, xLocal.ReinterpretCast<Tweight>()[rowAlign], RoundMode::CAST_NONE, rowAlign);
            }
            // if sizeof(Tweight) == 2, wait for beta loaded, cast beta to float
            if (!nullptrBeta && rowIdx == 1 && sizeof(Tweight) == SINGLE_CONST_TWO) {
                event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                Cast(xLocal[rowAlign], xLocal.ReinterpretCast<Tweight>()[rowAlign * SINGLE_CONST_THREE], RoundMode::CAST_NONE, rowAlign);
            }
        }

        // store rstdLocal to rstdGm
        {
            event_t eventIdSToMte3V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventIdSToMte3V1);
            WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3V1);
            DataCopyExtParams dataCopyParams;
            dataCopyParams.dstStride = 0;
            dataCopyParams.blockLen = nRow * sizeof(float);
            dataCopyParams.srcStride = 0;
            dataCopyParams.blockCount = 1;
            DataCopyPad(rstdGm[currentParamOffset], rstdLocal, dataCopyParams);
        }
        outQueueRstd.FreeTensor(rstdLocal);

        if ((!nullptrGamma || !nullptrBeta) && sizeof(Tweight) == SINGLE_CONST_FOUR) {
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        } else {
            PipeBarrier<PIPE_V>();
        }

        // calculate y = x * gamma
        if (!nullptrGamma) {
            for (uint32_t rowIdx = 0; rowIdx < nRow; ++rowIdx) {
                uint32_t currentRowOffset = rowIdx * rowAlign;
                Mul(yLocal[currentRowOffset], yLocal[currentRowOffset], xLocal, rowSize);
            }
        }

        // load beta if necessary
        if (!nullptrBeta && nRow == 1) {
            event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            DataCopyPad(
                xLocal.ReinterpretCast<Tweight>()[(sizeof(Tweight) == SINGLE_CONST_TWO) * rowAlign], betaGm, dataCopyParams,
                padParams);
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            if (sizeof(Tweight) == SINGLE_CONST_TWO) {
                Cast(xLocal, xLocal.ReinterpretCast<Tweight>()[rowAlign], RoundMode::CAST_NONE, rowAlign);
                PipeBarrier<PIPE_V>();
            }
        } else {
            PipeBarrier<PIPE_V>();
        }

        // calculate y = y + beta
        if (!nullptrBeta) {
            for (uint32_t rowIdx = 0; rowIdx < nRow; ++rowIdx) {
                uint32_t currentRowOffset = rowIdx * rowAlign;
                Add(yLocal[currentRowOffset], yLocal[currentRowOffset], xLocal[(nRow > 1) * rowAlign], rowSize);
            }
        }
        inQueueX.FreeTensor(xLocal);

        // cast xLocal to Tfm
        if (sizeof(Tfm) == SINGLE_CONST_TWO) {
            PipeBarrier<PIPE_V>();
            if (std::is_same<Tfm, bfloat16_t>::value) {
                Cast(yLocal.ReinterpretCast<Tfm>(), yLocal, RoundMode::CAST_ROUND, tileLength);
            }
            if (std::is_same<Tfm, half>::value) {
                Cast(yLocal.ReinterpretCast<Tfm>(), yLocal, RoundMode::CAST_NONE, tileLength);
            }
        }

        // store yLocal to yGm
        outQueueY.EnQue(yLocal);
        yLocal = outQueueY.DeQue<float>();
        {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = nRow;
            dataCopyParams.blockLen = rowSize * sizeof(Tfm);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(yGm[currentBlockOffset], yLocal.ReinterpretCast<Tfm>(), dataCopyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY, outQueueMean, outQueueRstd;

    GlobalTensor<Tfm> yGm;
    GlobalTensor<Tfm> xGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<Tweight> gammaGm;
    GlobalTensor<Tweight> betaGm;
    GlobalTensor<float> meanGm;

    float value = 0.0;
    uint64_t acc_val = 0;

    // tilingData
    uint32_t numBlocks = 0;
    uint32_t colSize = 0;
    uint32_t rowSize = 0;
    float eps = 0.0;
    float coefficient = 0.0;
    uint32_t rowAlign = 0;
    uint32_t nRow = 0;
    uint32_t tailNRow = 0;
    uint32_t loopCount = 0;
    uint32_t tailLoop = 0;
    uint32_t tileLength = 0;
    uint32_t blockLength = 0;
    uint32_t nullptrGamma = 0;
    uint32_t nullptrBeta = 0;
};

} // namespace LayerNormV4

#endif // LAYER_NORM_V4_SINGLE_READ_H
