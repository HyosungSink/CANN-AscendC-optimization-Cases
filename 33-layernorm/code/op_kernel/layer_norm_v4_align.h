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
 * \file layer_norm_v4_align.h
 * \brief
 */

#ifndef LAYER_NORM_V4_ALIGN_H
#define LAYER_NORM_V4_ALIGN_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
#include "layer_norm_v4_common.h"

namespace LayerNormV4 {
using namespace AscendC;

template <typename TfmAlign, typename Tweight>
class LayerNormCustomAlign {
static constexpr int32_t FLOAT16_SIZES = 2;
static constexpr int32_t FLOAT32_SIZE = 4;
static constexpr int32_t NUM_THREE = 3;
static constexpr int32_t NUM_EIGHT = 8;
static constexpr int32_t NUM_SEVEN = 7;
public:
    __aicore__ inline LayerNormCustomAlign()
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd,
        GM_ADDR workspace, const LayerNormV4MergeNTilingData *__restrict tilingData)
    {
        InitTiling(tilingData);
        // calculate xGm and paramGm offset and size
        uint32_t xGmOffset = 0;
        uint32_t xGmSize = 0;
        uint32_t paramGmSize = 0;
        uint32_t paramGmOffset = 0;
        uint32_t reulstGmSize = 0;

        // calculate xGm offset、xGm size
        if (GetBlockIdx() < formerNum) {
            blockLength = nRow * rowSize;
            uint32_t formerTileLength = loopCount * nRow * rowSize  + formerRemainNum * rowSize;
            xGmOffset = formerTileLength * GetBlockIdx();
            xGmSize = formerTileLength;
            // calculate paramGm offset、size
            paramGmOffset = (loopCount * nRow + formerRemainNum) * GetBlockIdx();
            paramGmSize = loopCount * nRow + formerRemainNum;
            remainTileLength = formerRemainNum * rowAlign;
            reulstGmSize = (nRow + NUM_SEVEN) / NUM_EIGHT * NUM_EIGHT;;
        } else {
            blockLength = tailNRow * rowSize;
            uint32_t tailTileLength = tailLoop * blockLength + tailRemainNum * rowSize;
            xGmOffset = (loopCount * nRow * rowSize  + formerRemainNum * rowSize) * formerNum + tailTileLength * (GetBlockIdx() - formerNum);
            xGmSize = tailTileLength;
            // calculate paramGm offset、size
            paramGmOffset = (loopCount * nRow + formerRemainNum) * formerNum + (tailLoop * tailNRow + tailRemainNum) * (GetBlockIdx() - formerNum);
            paramGmSize = tailLoop * tailNRow + tailRemainNum;
            tileLength = tailNRow * rowAlign;
            remainTileLength = tailRemainNum * rowAlign;
            reulstGmSize = (tailNRow + NUM_SEVEN) / NUM_EIGHT * NUM_EIGHT;
        }

        // set global buffer
        xGm.SetGlobalBuffer((__gm__ TfmAlign *)x + xGmOffset, xGmSize);
        yGm.SetGlobalBuffer((__gm__ TfmAlign *)y + xGmOffset, xGmSize);
        meanGm.SetGlobalBuffer((__gm__ float *)mean + paramGmOffset, paramGmSize);
        rstdGm.SetGlobalBuffer((__gm__ float *)rstd + paramGmOffset, paramGmSize);
        gammaGm.SetGlobalBuffer((__gm__ Tweight *)gamma, rowSize);
        betaGm.SetGlobalBuffer((__gm__ Tweight *)beta, rowSize);

        // pipe init buffer
        pipe.InitBuffer(inQueueX, 1, tilingData->blockLength * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, tilingData->blockLength * sizeof(float));
        pipe.InitBuffer(outQueueMean, 1, reulstGmSize * sizeof(float));
        pipe.InitBuffer(outQueueRstd, 1, reulstGmSize * sizeof(float));
    }

    __aicore__ inline void InitTiling(const LayerNormV4MergeNTilingData *__restrict tilingData) {
        // load tiling data
        numBlocks = tilingData->numBlocks;
        colSize = tilingData->colSize;
        coefficient = tilingData->coefficient;
        rowAlign = tilingData->rowAlign;
        nRow = tilingData->nRow;
        rowSize = tilingData->rowSize;
        eps = tilingData->eps;
        tailLoop = tilingData->tailLoop;
        tileLength = tilingData->tileLength;
        tailNRow = tilingData->tailNRow;
        loopCount = tilingData->loopCount;
        nullptrGamma = tilingData->nullptrGamma;
        nullptrBeta = tilingData->nullptrBeta;    
        blockLength = tilingData->tileLength;
        tailNum = tilingData->tailNum;
        formerNum = tilingData->formerNum;
        tailRemainNum = tilingData->tailRemainNum;
        formerRemainNum = tilingData->formerRemainNum;
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() < formerNum) {
            uint32_t currentBlockOffsetAlign = 0;
            uint32_t currentParamOffset = 0;
            uint32_t count = loopCount;
            for (uint32_t loopIdx = 0; loopIdx < count; ++loopIdx) {
                currentBlockOffsetAlign = loopIdx * blockLength;
                currentParamOffset = loopIdx * nRow;
                ProcessBasicBlock(nRow, currentBlockOffsetAlign, currentParamOffset);
            }

            if (formerRemainNum > 0) {
                currentBlockOffsetAlign = currentBlockOffsetAlign + blockLength;
                currentParamOffset = currentParamOffset + nRow;
                tileLength = remainTileLength;
                ProcessBasicBlock(formerRemainNum, currentBlockOffsetAlign, currentParamOffset);
            }
        }else {
            uint32_t count = tailLoop;
            uint32_t currentBlockOffsetAlign = 0;
            uint32_t currentParamOffset = 0;
            for (uint32_t loopIdx = 0; loopIdx < count; ++loopIdx) {
                currentBlockOffsetAlign = loopIdx * blockLength;
                currentParamOffset = loopIdx * tailNRow;
                ProcessBasicBlock(tailNRow, currentBlockOffsetAlign, currentParamOffset);
            }
            if (tailRemainNum > 0) {
                currentBlockOffsetAlign = currentBlockOffsetAlign + blockLength;
                currentParamOffset = currentParamOffset + tailNRow;
                tileLength = remainTileLength;
                ProcessBasicBlock(tailRemainNum, currentBlockOffsetAlign, currentParamOffset);
            }
        }
    }

private:
    __aicore__ inline void DataCopyInX(LocalTensor<float>& dstTensor, GlobalTensor<TfmAlign> srcTensor, DataCopyStruct dataCopyStruct, uint32_t tileLength) {
        // load srcTensor to dstTensor
        DataCopyInContiguous(dstTensor.ReinterpretCast<TfmAlign>(), srcTensor, dataCopyStruct, tileLength);
        inQueueX.EnQue(dstTensor);
        dstTensor = inQueueX.DeQue<float>();

        // cast dstTensor to float
        if constexpr (sizeof(TfmAlign) == FLOAT16_SIZES) {
            Cast(dstTensor, dstTensor.ReinterpretCast<TfmAlign>()[tileLength], RoundMode::CAST_NONE, tileLength);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void DataCopyOutY(GlobalTensor<TfmAlign> dstTensor, LocalTensor<float>& srcTensor, DataCopyStruct dataCopyStruct) {
        // cast xLocal to TfmAlign
        if constexpr (sizeof(TfmAlign) == FLOAT16_SIZES) {
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<TfmAlign, bfloat16_t>::value) {
                Cast(srcTensor.ReinterpretCast<TfmAlign>(), srcTensor, RoundMode::CAST_ROUND, tileLength);
            }
            if constexpr (std::is_same<TfmAlign, half>::value) {
                Cast(srcTensor.ReinterpretCast<TfmAlign>(), srcTensor, RoundMode::CAST_NONE, tileLength);
            }
        }

        // store srcTensor to dstTensor
        outQueueY.EnQue(srcTensor);
        srcTensor = outQueueY.DeQue<float>();
        DataCopyOutContiguous(dstTensor, srcTensor.ReinterpretCast<TfmAlign>(), dataCopyStruct);
        outQueueY.FreeTensor(srcTensor);
    }

    __aicore__ inline void BeforeProcessBasicBlock(uint32_t rowNum, uint32_t currentBlockOffset, uint32_t currentParamOffset) {
        DataCopyStruct dataCopyStruct1{rowAlign == rowSize ? 1 : rowNum, static_cast<uint32_t>(rowAlign == rowSize ? rowNum * rowSize * sizeof(TfmAlign) : rowSize * sizeof(TfmAlign)), 0, 0, rowAlign != rowSize, 0, static_cast<uint8_t>(rowAlign - rowSize)};
        DataCopyInX(xLocal, xGm[currentBlockOffset], dataCopyStruct1, tileLength);
        // calculate x * coefficient
        Muls(yLocal, xLocal, coefficient, tileLength);
        PipeBarrier<PIPE_V>();

        // calculate mean row-by-row
        for (uint32_t rowIdx = 0; rowIdx < rowNum; ++rowIdx) {
            uint32_t currentRowOffset = rowIdx * rowAlign;
            ReduceSum<float>(yLocal[currentRowOffset], yLocal[currentRowOffset], yLocal[currentRowOffset], rowSize);
            SetEvtFlag<HardEvent::V_S>();
            float value = yLocal.GetValue(currentRowOffset);
            SetEvtFlag<HardEvent::S_V>();
            Adds(yLocal[currentRowOffset], xLocal[currentRowOffset], -value, rowSize);
            meanLocal.SetValue(rowIdx, static_cast<float>(value));
        }

        // store meanLocal to meanGm
        SetEvtFlag<HardEvent::S_MTE3>();
        DataCopyStruct dataCopyStruct2{1, static_cast<uint32_t>(rowNum * sizeof(float)), 0, 0, false, 0, 0};
        DataCopyOutContiguous(meanGm[currentParamOffset], meanLocal, dataCopyStruct2);
        outQueueMean.FreeTensor(meanLocal);

        // calculate square and muls coefficient
        PipeBarrier<PIPE_V>();
        Mul(xLocal, yLocal, yLocal, tileLength);
        PipeBarrier<PIPE_V>();
        Muls(xLocal, xLocal, coefficient, tileLength);
        PipeBarrier<PIPE_V>();

        // process rstd row-by-row
        for (uint32_t rowIdx = 0; rowIdx < rowNum; ++rowIdx) {
            uint32_t currentRowOffset = rowIdx * rowAlign;
            ReduceSum<float>(xLocal[currentRowOffset], xLocal[currentRowOffset], xLocal[currentRowOffset], rowSize);
            SetEvtFlag<HardEvent::S_MTE3>();
            float value = xLocal.GetValue(currentRowOffset);
            float rstdValue = static_cast<float>(1.0) / sqrt(value + static_cast<float>(eps));
            SetEvtFlag<HardEvent::S_V>();
            Muls(yLocal[currentRowOffset], yLocal[currentRowOffset], rstdValue, rowSize);
            rstdLocal.SetValue(rowIdx, rstdValue);
        }

        // store rstdLocal to rstdGm
        SetEvtFlag<HardEvent::S_MTE3>();
        DataCopyStruct dataCopyStruct4{1, static_cast<uint32_t>(rowNum * sizeof(float)), 0, 0, false, 0, 0};
        DataCopyOutContiguous(rstdGm[currentParamOffset], rstdLocal, dataCopyStruct4);

        // the 0th row has been processed, load gamma to the 0th row
        if (!nullptrGamma) {
            SetEvtFlag<HardEvent::V_MTE2>();
            DataCopyStruct dataCopyStruct3{1, static_cast<uint32_t>(rowSize * sizeof(Tweight)), 0, 0, false, 0, 0};
            DataCopyInContiguous(xLocal.ReinterpretCast<Tweight>(), gammaGm, dataCopyStruct3, rowAlign);
            SetEvtFlag<HardEvent::MTE2_V>();

            // if sizeof(Tweight) == 2, wait for gamma loaded, cast gamma to float
            if constexpr (sizeof(Tweight) == FLOAT16_SIZES) {
                Cast(xLocal, xLocal.ReinterpretCast<Tweight>()[rowAlign], RoundMode::CAST_NONE, rowAlign);
                PipeBarrier<PIPE_V>();
            }

            // calculate y = x * gamma
            for (uint32_t rowIdx = 0; rowIdx < rowNum; ++rowIdx) {
                uint32_t currentRowOffset = rowIdx * rowAlign;
                Mul(yLocal[currentRowOffset], yLocal[currentRowOffset], xLocal, rowSize);
            }
        }
    }

    __aicore__ inline void ProcessBasicBlock(uint32_t rowNum, uint32_t currentBlockOffset, uint32_t currentParamOffset)
    {
        // allocate local tensor
        yLocal = outQueueY.AllocTensor<float>();
        xLocal = inQueueX.AllocTensor<float>();
        rstdLocal = outQueueRstd.AllocTensor<float>();
        meanLocal = outQueueMean.AllocTensor<float>();
        BeforeProcessBasicBlock(rowNum, currentBlockOffset, currentParamOffset);
        outQueueRstd.FreeTensor(rstdLocal);
        DataCopyStruct dataCopyStruct3{1, static_cast<uint32_t>(rowSize * sizeof(Tweight)), 0, 0, false, 0, 0};
        // the 1st row has been processed, load gamma to the 1st row
        // if sizeof(Tweight) == 2, wait for beta loaded, cast beta to float
        if (!nullptrBeta) {
            if (rowNum >= NUM_TWO) {
                SetEvtFlag<HardEvent::V_MTE2>();
                DataCopyInContiguous(xLocal.ReinterpretCast<Tweight>()[rowAlign], betaGm, dataCopyStruct3, FLOAT16_SIZES * rowAlign);
                SetEvtFlag<HardEvent::MTE2_V>();
            }

            if (rowNum >= NUM_TWO && sizeof(Tweight) == FLOAT16_SIZES) {
                Cast(xLocal[rowAlign], xLocal.ReinterpretCast<Tweight>()[rowAlign * NUM_THREE], RoundMode::CAST_NONE, rowAlign);
                PipeBarrier<PIPE_V>();
            }

            // load beta if necessary
            if (rowNum == 1) {
                SetEvtFlag<HardEvent::V_MTE2>();
                DataCopyInContiguous(xLocal.ReinterpretCast<Tweight>(), betaGm, dataCopyStruct3, rowAlign);
                SetEvtFlag<HardEvent::MTE2_V>();
                if (sizeof(Tweight) == FLOAT16_SIZES) {
                    Cast(xLocal, xLocal.ReinterpretCast<Tweight>()[rowAlign], RoundMode::CAST_NONE, rowAlign);
                    PipeBarrier<PIPE_V>();
                }
            }

            // calculate y = y + beta
            for (uint32_t rowIdx = 0; rowIdx < rowNum; ++rowIdx) {
                uint32_t currentRowOffset = rowIdx * rowAlign;
                Add(yLocal[currentRowOffset], yLocal[currentRowOffset], xLocal[(rowNum > 1) * rowAlign], rowSize);
            }
        }
        inQueueX.FreeTensor(xLocal);

        DataCopyStruct dataCopyStruct5{rowAlign == rowSize ? 1 : rowNum, static_cast<uint32_t>(rowAlign == rowSize ? rowNum * rowSize * sizeof(TfmAlign) : rowSize * sizeof(TfmAlign)), 0, 0, false, 0, 0};
        DataCopyOutY(yGm[currentBlockOffset], yLocal, dataCopyStruct5);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueMean;
    TQue<QuePosition::VECOUT, 1> outQueueRstd;

    GlobalTensor<TfmAlign> xGm;
    GlobalTensor<TfmAlign> yGm;
    GlobalTensor<Tweight> gammaGm;
    GlobalTensor<Tweight> betaGm;
    GlobalTensor<float> meanGm;
    GlobalTensor<float> rstdGm;

    LocalTensor<float> yLocal;
    LocalTensor<float> xLocal;
    LocalTensor<float> rstdLocal;
    LocalTensor<float> meanLocal;

    uint64_t acc_val = 0;
    float value = 0.0;

    // tilingData
    uint32_t numBlocks = 0;
    uint32_t colSize = 0;
    uint32_t rowSize = 0;
    float eps = 0.0;
    uint32_t rowAlign = 0;
    float coefficient = 0.0;
    uint32_t tailNRow = 0;
    uint32_t nRow = 0;
    uint32_t tailLoop = 0;
    uint32_t loopCount = 0;
    uint32_t remainTileLength = 0;
    uint32_t tileLength = 0;
    uint32_t nullptrGamma = 0;
    uint32_t blockLength = 0;
    uint32_t formerNum = 0;
    uint32_t nullptrBeta = 0;    
    uint32_t tailNum = 0;
    uint32_t formerRemainNum = 0;
    uint32_t tailRemainNum = 0;
};

}  // namespace LayerNormCustom

#endif  // LAYER_NORM_CUSTOM_ALIGN_H