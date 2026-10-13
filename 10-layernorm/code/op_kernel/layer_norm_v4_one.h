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
* \file layer_norm_v4_one.h
* \brief
*/

#ifndef LAYER_NORM_V4_ONE_H
#define LAYER_NORM_V4_ONE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
#include "layer_norm_v4_common.h"

namespace LayerNormV4 {
using namespace AscendC;

template <typename TfmOne, typename Tweight>
class LayerNormCustomOne {
static constexpr int32_t BLOCK_ALIGN_SIZE = 64;
static constexpr int32_t FLOAT32_SIZE = 4;
static constexpr int32_t BLOCK_SIZE = 32;
static constexpr int32_t FLOAT_TO_BLOCK = 8;
static constexpr int32_t NUM_THREE = 3;
static constexpr int32_t NUM_EIGHT = 8;

public:
    __aicore__ inline LayerNormCustomOne()
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd,
        GM_ADDR workspace, const LayerNormV4MergeNTilingData *__restrict tilingData)
    {
        InitTiling(tilingData);
        // calculate xGm and paramGm offset and size
        uint32_t xGmSize = 0;
        uint32_t xGmOffset = 0;
        uint32_t paramGmSize = 0;
        uint32_t paramGmOffset = 0;
        uint32_t reulstGmSize = 0;
        uint32_t alignNum = NUM_EIGHT;
        if constexpr (sizeof(TfmOne) == FLOAT16_SIZE) {
            alignNum = NUM_EIGHT * NUM_TWO;
        }

        // calculate xGm offset、xGm size
        if (GetBlockIdx() < formerNum) {
            BigCoreInit(xGmOffset, xGmSize, paramGmOffset, paramGmSize, reulstGmSize);
        } else {
            SmallCoreInit(xGmOffset, xGmSize, paramGmOffset, paramGmSize, reulstGmSize);
        }
        // pipe init buffer
        pipe.InitBuffer(inQueueX, 1, tilingData->blockLength * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, tilingData->blockLength * sizeof(float));

        // set global buffer
        xGm.SetGlobalBuffer((__gm__ TfmOne *)x + xGmOffset, xGmSize);
        yGm.SetGlobalBuffer((__gm__ TfmOne *)y + xGmOffset, xGmSize);
        meanGm.SetGlobalBuffer((__gm__ float *)mean + paramGmOffset, paramGmSize);
        rstdGm.SetGlobalBuffer((__gm__ float *)rstd + paramGmOffset, paramGmSize);
        gammaGm.SetGlobalBuffer((__gm__ Tweight *)gamma, rowAlign);
        betaGm.SetGlobalBuffer((__gm__ Tweight *)beta, rowAlign);
    }

    __aicore__ inline void BigCoreInit(uint32_t& xGmOffset, uint32_t& xGmSize, uint32_t& paramGmOffset,
            uint32_t& paramGmSize, uint32_t& reulstGmSize) {
        uint32_t alignNum = NUM_EIGHT;
        if constexpr (sizeof(TfmOne) == FLOAT16_SIZE) {
            alignNum = NUM_EIGHT * NUM_TWO;
        }
        blockLength = nRow * rowSize;
        uint32_t formerBlockLength = loopCount * blockLength  + formerRemainNum * rowSize;
        xGmOffset = formerBlockLength * GetBlockIdx();
        xGmSize = formerBlockLength;
        // calculate paramGm offset、size
        paramGmOffset = (loopCount * nRow + formerRemainNum) * GetBlockIdx();
        paramGmSize = loopCount * nRow + formerRemainNum;
        if (loopCount < alignNum) {
            remainTileLength = 0;
            nRow = loopCount * nRow + formerRemainNum;
            reulstGmSize = nRow;
            blockLength = nRow;
            tileLength = (nRow + alignNum - 1) / alignNum * alignNum;
            loopCount = 1;
            formerRemainNum = 0;
        } else {
            formerRemainNum = (loopCount % alignNum) * nRow * rowSize + formerRemainNum;
            loopCount = loopCount / alignNum;
            remainTileLength = (formerRemainNum + alignNum - 1) / alignNum * alignNum;
            nRow = nRow * alignNum;
            tileLength = nRow;
            reulstGmSize = nRow;
            blockLength = nRow;
        }
    }

    __aicore__ inline void SmallCoreInit(uint32_t& xGmOffset, uint32_t& xGmSize, uint32_t& paramGmOffset,
            uint32_t& paramGmSize, uint32_t& reulstGmSize) {
        uint32_t alignNum = NUM_EIGHT;
        if constexpr (sizeof(TfmOne) == FLOAT16_SIZE) {
            alignNum = NUM_EIGHT * NUM_TWO;
        }
        blockLength = tailNRow * rowSize;
        uint32_t tailTileLength = tailLoop * blockLength + tailRemainNum * rowSize;
        xGmOffset = (loopCount * nRow * rowSize  + formerRemainNum * rowSize) * formerNum + tailTileLength * (GetBlockIdx() - formerNum);
        xGmSize = tailTileLength;

        paramGmSize = tailLoop * tailNRow + tailRemainNum;
        // calculate paramGm offset、size
        paramGmOffset = (loopCount * nRow + formerRemainNum) * formerNum + (paramGmSize) * (GetBlockIdx() - formerNum);
        if (tailLoop < alignNum) {
            tailNRow = tailNRow * tailLoop + tailRemainNum;
            tileLength = (tailNRow + alignNum - 1)  / alignNum * alignNum;
            remainTileLength = 0;
            reulstGmSize = tailNRow;
            blockLength = tailNRow;
            tailRemainNum = 0;
            tailLoop = 1;
        } else {
            tailRemainNum = (tailLoop % alignNum) * tailNRow + tailRemainNum;
            tailLoop = tailLoop / alignNum;
            tailNRow = tailNRow * alignNum;
            tileLength = tailNRow;
            remainTileLength = (tailRemainNum + alignNum - 1) / alignNum * alignNum;
            reulstGmSize = tailNRow;
            blockLength = tailNRow;
        }
    }

    __aicore__ inline void InitTiling(const LayerNormV4MergeNTilingData *__restrict tilingData) {
        // load tiling data
        colSize = tilingData->colSize;
        numBlocks = tilingData->numBlocks;
        rowSize = tilingData->rowSize;
        eps = tilingData->eps;
        coefficient = tilingData->coefficient;
        rowAlign = tilingData->rowAlign;
        tailNRow = tilingData->tailNRow;
        nRow = tilingData->nRow;
        nullptrGamma = tilingData->nullptrGamma;
        nullptrBeta = tilingData->nullptrBeta;   
        loopCount = tilingData->loopCount;
        tileLength = tilingData->tileLength;
        tailLoop = tilingData->tailLoop;
        formerNum = tilingData->formerNum;
        formerRemainNum = tilingData->formerRemainNum;
        tailNum = tilingData->tailNum;
        tailRemainNum = tilingData->tailRemainNum;
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() < formerNum) {
            uint32_t currentBlockOffsetOne = 0;
            uint32_t currentParamOffset = 0;
            uint32_t count = loopCount;
            for (uint32_t loopIdx = 0; loopIdx < count; ++loopIdx) {
                currentBlockOffsetOne = loopIdx * blockLength;
                currentParamOffset = loopIdx * nRow;
                ProcessBasicBlock(nRow, currentBlockOffsetOne, currentParamOffset);
            }

            if (formerRemainNum > 0) {
                currentBlockOffsetOne = currentBlockOffsetOne + blockLength;
                currentParamOffset = currentParamOffset + nRow;
                tileLength = remainTileLength;
                ProcessBasicBlock(formerRemainNum, currentBlockOffsetOne, currentParamOffset);
            }
        } else {
            uint32_t count = tailLoop;
            uint32_t currentBlockOffsetOne = 0;
            uint32_t currentParamOffset = 0;
            for (uint32_t loopIdx = 0; loopIdx < count; ++loopIdx) {
                currentBlockOffsetOne = loopIdx * blockLength;
                currentParamOffset = loopIdx * tailNRow;
                ProcessBasicBlock(tailNRow, currentBlockOffsetOne, currentParamOffset);
            }

            if (tailRemainNum > 0) {
                currentBlockOffsetOne = currentBlockOffsetOne + blockLength;
                currentParamOffset = currentParamOffset + tailNRow;
                tileLength = remainTileLength;
                ProcessBasicBlock(tailRemainNum, currentBlockOffsetOne, currentParamOffset);
            }
        }
    }

private:
    __aicore__ inline void ProcessBasicBlock(uint32_t rowNum, uint32_t currentBlockOffset, uint32_t currentParamOffset)
    {
        // allocate local tensor
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // load xGm to xLocal
        DataCopyStruct dataCopyStruct{1, static_cast<uint32_t>(rowNum * rowSize * sizeof(TfmOne)), 0, 0, false, 0, static_cast<uint8_t>(rowAlign - rowSize)};
        SetEvtFlag<HardEvent::MTE3_MTE2>();
        DataCopyInContiguous<TfmOne>(xLocal.ReinterpretCast<TfmOne>(), xGm[currentBlockOffset], dataCopyStruct, tileLength);
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<float>();

        // cast xLocal to float
        if constexpr (sizeof(TfmOne) == FLOAT16_SIZE) {
            Cast(xLocal, xLocal.ReinterpretCast<TfmOne>()[tileLength], RoundMode::CAST_NONE, tileLength);
            PipeBarrier<PIPE_V>();
        }

        // store meanLocal to meanGm
        SetEvtFlag<HardEvent::V_MTE3>();
        DataCopyStruct dataCopyStruct1{1, static_cast<uint32_t>(rowNum * sizeof(float)), 0, 0, false, 0, 0};
        DataCopyOutContiguous(meanGm[currentParamOffset], xLocal, dataCopyStruct1);

        // the 1st row has been processed, load gamma to the 1st row
        if (nullptrBeta) {
            SetEvtFlag<HardEvent::MTE3_V>();
            Duplicate(xLocal.ReinterpretCast<Tweight>(), static_cast<Tweight>(0.0f), rowNum);
        } else {
            SetEvtFlag<HardEvent::MTE3_MTE2>();
            DataCopyStruct dataCopyStruct2{1, static_cast<uint32_t>(rowSize * sizeof(Tweight)), 0, 0, true, 0, static_cast<uint8_t>(rowAlign - rowSize)};
            DataCopyInContiguous(xLocal.ReinterpretCast<Tweight>(), betaGm, dataCopyStruct2, 0);

            SetEvtFlag<HardEvent::MTE2_S>();
            Tweight betaValue = xLocal.ReinterpretCast<Tweight>().GetValue(0);
            SetEvtFlag<HardEvent::S_V>();
            Duplicate(xLocal.ReinterpretCast<Tweight>(), betaValue, rowNum);
        }
        PipeBarrier<PIPE_V>();
        if constexpr (!std::is_same<TfmOne, Tweight>::value) {
            Cast(xLocal.ReinterpretCast<TfmOne>(),xLocal.ReinterpretCast<Tweight>(), RoundMode::CAST_NONE, rowNum);
            PipeBarrier<PIPE_V>();
        }

        float rstdValue = static_cast<float>(1.0) / sqrt(static_cast<float>(eps));
        Duplicate(yLocal, rstdValue, rowNum);

        // store rstdLocal to rstdGm
        SetEvtFlag<HardEvent::V_MTE3>();
        DataCopyStruct dataCopyStruct3{1, static_cast<uint32_t>(rowNum * sizeof(float)), 0, 0, false, 0, 0};
        DataCopyOutContiguous(rstdGm[currentParamOffset], yLocal, dataCopyStruct3);

        // store yLocal to yGm
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<float>();
        DataCopyStruct dataCopyStruct4{1, static_cast<uint32_t>(rowNum * rowSize * sizeof(TfmOne)), 0, 0, false, 0, 0};
        DataCopyOutContiguous(yGm[currentBlockOffset], xLocal.ReinterpretCast<TfmOne>(), dataCopyStruct4);
        outQueueY.FreeTensor(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;

    GlobalTensor<TfmOne> yGm;
    GlobalTensor<TfmOne> xGm;
    GlobalTensor<Tweight> gammaGm;
    GlobalTensor<Tweight> betaGm;
    GlobalTensor<float> meanGm;
    GlobalTensor<float> rstdGm;

    uint64_t acc_val = 0;
    float value = 0.0;

    // tilingData
    uint32_t numBlocks = 0;
    uint32_t rowSize = 0;
    uint32_t colSize = 0;
    float eps = 0.0;
    float coefficient = 0.0;
    uint32_t rowAlign = 0;
    uint32_t nRow = 0;
    uint32_t formerNum = 0;
    uint32_t tailNRow = 0;
    uint32_t loopCount = 0;
    uint32_t tailLoop = 0;
    uint32_t tileLength = 0;
    uint32_t remainTileLength = 0;
    uint32_t blockLength = 0;   
    uint32_t formerRemainNum = 0;
    uint32_t tailRemainNum = 0;
    uint32_t rowParts = 0;
    uint32_t tailNum = 0;
    uint32_t rowTailOffset = 0;
    uint32_t nullptrGamma = 0;
    uint32_t nullptrBeta = 0;    
    uint32_t rowTailMask = 0;
};

}  // namespace LayerNormCustomOne

#endif  // LAYER_NORM_CUSTOM_ONE_H