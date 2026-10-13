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
* \file layer_norm_v4_align_limit.h
* \brief
*/

#ifndef LAYER_NORM_V4_ALIGN_LIMIT_H
#define LAYER_NORM_V4_ALIGN_LIMIT_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
#include "layer_norm_v4_common.h"

namespace LayerNormV4 {
using namespace AscendC;

template <typename Tfm, typename Tweight>
class LayerNormCustomAlignLimit {
static constexpr int32_t BLOCK_ALIGN_SIZE = 64;
static constexpr int32_t MAX_REPEAT_TIMES = 255;
static constexpr int32_t ALIGN_MAX_REPEAT_TIMES = 224;
static constexpr int32_t SPECIAL_COL = 24;
static constexpr int32_t FLOAT32_SIZE = 4;
static constexpr int32_t NUM_EIGHT = 8;
static constexpr int32_t NUM_SEVEN = 7;
static constexpr int32_t BLOCK_SIZE = 32;
static constexpr int32_t FLOAT_TO_BLOCK = 8;
static constexpr int32_t BLOCK_TO_REPEAT = 8;
static constexpr int32_t NUM_THREE = 3;
static constexpr int32_t REPEAT_ALIGN_LIMIT = 32;
static constexpr int32_t HALF_REPEAT_ALIGN_LIMIT = 16;

struct ComputeStruct {
uint32_t mask = 0;
int32_t repeatTimes = 0;  // 行数
uint8_t repStrid = 0;
uint32_t rowCycleNum = 0;
uint32_t rowCycleTailNum = 0;
uint32_t rowAlignOffset = 0;
uint32_t blockOffset = 0;
};

public:
    __aicore__ inline LayerNormCustomAlignLimit()
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd,
        GM_ADDR workspace, const LayerNormV4MergeNTilingData *__restrict tilingData)
    {
        InitTiling(tilingData);
        // calculate xGm and paramGm offset and size
        uint32_t xGmOffset = 0;
        uint32_t xGmSize = 0;
        uint32_t paramGmOffset = 0;
        uint32_t paramGmSize = 0;
        uint32_t reulstGmSize = 0;

        // calculate xGm offset、xGm size
        if (GetBlockIdx() < formerNum) {
            blockLength = nRow * rowSize;
            uint32_t formerBlockLength = loopCount * blockLength  + formerRemainNum * rowSize;
            xGmOffset = formerBlockLength * GetBlockIdx();
            xGmSize = formerBlockLength;
            // calculate paramGm offset、size
            paramGmOffset = (loopCount * nRow + formerRemainNum) * GetBlockIdx();
            paramGmSize = loopCount * nRow + formerRemainNum;
            remainTileLength = formerRemainNum * rowAlign;
            reulstGmSize = (nRow + NUM_SEVEN) / NUM_EIGHT * NUM_EIGHT;
        } else {
            blockLength = tailNRow * rowSize;
            uint32_t tailTileLength = tailLoop * blockLength + tailRemainNum * rowSize;
            xGmOffset = (loopCount * nRow * rowSize  + formerRemainNum * rowSize) * formerNum + tailTileLength * (GetBlockIdx() - formerNum);
            xGmSize = tailTileLength;

            paramGmSize = tailLoop * tailNRow + tailRemainNum;
            // calculate paramGm offset、size
            paramGmOffset = (loopCount * nRow + formerRemainNum) * formerNum + (paramGmSize) * (GetBlockIdx() - formerNum);
            tileLength = tailNRow * rowAlign;
            remainTileLength = tailRemainNum * rowAlign;
            reulstGmSize = (tailNRow + NUM_SEVEN) / NUM_EIGHT * NUM_EIGHT;
        }

        // set global buffer
        xGm.SetGlobalBuffer((__gm__ Tfm *)x + xGmOffset, xGmSize);
        yGm.SetGlobalBuffer((__gm__ Tfm *)y + xGmOffset, xGmSize);
        meanGm.SetGlobalBuffer((__gm__ float *)mean + paramGmOffset, paramGmSize);
        rstdGm.SetGlobalBuffer((__gm__ float *)rstd + paramGmOffset, paramGmSize);
        gammaGm.SetGlobalBuffer((__gm__ Tweight *)gamma, rowAlign);
        betaGm.SetGlobalBuffer((__gm__ Tweight *)beta, rowAlign);

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
        rowSize = tilingData->rowSize;
        eps = tilingData->eps;
        coefficient = tilingData->coefficient;
        rowAlign = tilingData->rowAlign;
        nRow = tilingData->nRow;
        tailNRow = tilingData->tailNRow;
        loopCount = tilingData->loopCount;
        tailLoop = tilingData->tailLoop;
        tileLength = tilingData->tileLength;
        blockLength = tilingData->tileLength;
        nullptrGamma = tilingData->nullptrGamma;
        nullptrBeta = tilingData->nullptrBeta;    
        formerNum = tilingData->formerNum;
        formerRemainNum = tilingData->formerRemainNum;
        tailNum = tilingData->tailNum;
        tailRemainNum = tilingData->tailRemainNum;

        rowParts = (rowAlign + BLOCK_ALIGN_SIZE - 1) / BLOCK_ALIGN_SIZE;
        rowTailOffset = (rowParts - 1) * BLOCK_ALIGN_SIZE;
        rowTailMask = rowSize % BLOCK_ALIGN_SIZE == 0 ? BLOCK_ALIGN_SIZE : rowSize % BLOCK_ALIGN_SIZE;
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() < formerNum) {
            uint32_t currentBlockOffset = 0;
            uint32_t currentParamOffset = 0;
            uint32_t count = loopCount;
            for (uint32_t loopIdx = 0; loopIdx < count; ++loopIdx) {
                currentBlockOffset = loopIdx * blockLength;
                currentParamOffset = loopIdx * nRow;
                ProcessBasicBlock(nRow, currentBlockOffset, currentParamOffset);
            }

            if (formerRemainNum > 0) {
                currentBlockOffset = currentBlockOffset + blockLength;
                currentParamOffset = currentParamOffset + nRow;
                tileLength = remainTileLength;
                ProcessBasicBlock(formerRemainNum, currentBlockOffset, currentParamOffset);
            }
        } else {
            uint32_t count = tailLoop;
            uint32_t currentBlockOffset = 0;
            uint32_t currentParamOffset = 0;
            for (uint32_t loopIdx = 0; loopIdx < count; ++loopIdx) {
                currentBlockOffset = loopIdx * blockLength;
                currentParamOffset = loopIdx * tailNRow;
                ProcessBasicBlock(tailNRow, currentBlockOffset, currentParamOffset);
            }

            if (tailRemainNum > 0) {
                currentBlockOffset = currentBlockOffset + blockLength;
                currentParamOffset = currentParamOffset + tailNRow;
                tileLength = remainTileLength;
                ProcessBasicBlock(tailRemainNum, currentBlockOffset, currentParamOffset);
            }
        }
    }

private:
    __aicore__ inline void ReduceSumWithModeDifTensor(LocalTensor<float> dstTensor, LocalTensor<float> srcTensor, uint32_t repeat, uint32_t mask, uint32_t dstRepStride, uint32_t srcBlkStride, uint32_t srcRepStride) {
        uint32_t blockNum = ((repeat * rowAlign) + NUM_SEVEN) / NUM_EIGHT;
        uint32_t repeatNum = (blockNum + NUM_SEVEN) / NUM_EIGHT;
        uint32_t num = repeatNum / MAX_REPEAT_TIMES;
        uint32_t reduceNum = repeat / ALIGN_MAX_REPEAT_TIMES;
        for(uint32_t i = 0; i < reduceNum; ++i) {
            WholeReduceSum<float>(dstTensor[i * ALIGN_MAX_REPEAT_TIMES], srcTensor[i * ALIGN_MAX_REPEAT_TIMES * rowAlign], mask, ALIGN_MAX_REPEAT_TIMES, dstRepStride, srcBlkStride, srcRepStride);
        }   
        WholeReduceSum<float>(dstTensor[reduceNum * ALIGN_MAX_REPEAT_TIMES], srcTensor[reduceNum * ALIGN_MAX_REPEAT_TIMES * rowAlign], mask, repeat % ALIGN_MAX_REPEAT_TIMES, dstRepStride, srcBlkStride, srcRepStride);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceSumWithModeSameTensor(LocalTensor<float> dstTensor, LocalTensor<float> srcTensor, uint32_t repeat, uint32_t mask, uint32_t dstRepStride, uint32_t srcBlkStride, uint32_t srcRepStride) {
        uint32_t blockNum = ((repeat * rowAlign) + NUM_SEVEN) / NUM_EIGHT;
        uint32_t repeatNum = (blockNum + NUM_SEVEN) / NUM_EIGHT;
        uint32_t num = repeatNum / MAX_REPEAT_TIMES;
        uint32_t reduceNum = repeat / ALIGN_MAX_REPEAT_TIMES;
        for(uint32_t i = 0; i < reduceNum; ++i) {
            WholeReduceSum<float>(dstTensor[i * ALIGN_MAX_REPEAT_TIMES], srcTensor[i * ALIGN_MAX_REPEAT_TIMES * rowAlign], mask, ALIGN_MAX_REPEAT_TIMES, dstRepStride, srcBlkStride, srcRepStride);
            PipeBarrier<PIPE_V>();
        }   
        WholeReduceSum<float>(dstTensor[reduceNum * ALIGN_MAX_REPEAT_TIMES], srcTensor[reduceNum * ALIGN_MAX_REPEAT_TIMES * rowAlign], mask, repeat % ALIGN_MAX_REPEAT_TIMES, dstRepStride, srcBlkStride, srcRepStride);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MultiBrcb(LocalTensor<float> dstTensor, LocalTensor<float> srcTensor, uint32_t brcbRepeat) {
        uint32_t num = brcbRepeat / MAX_REPEAT_TIMES;
        for(uint32_t i = 0; i < num; ++i) {
            Brcb(dstTensor[i * MAX_REPEAT_TIMES * NUM_EIGHT * NUM_EIGHT], srcTensor[i * MAX_REPEAT_TIMES * NUM_EIGHT], MAX_REPEAT_TIMES, {1,FLOAT_TO_BLOCK});
        }   
        Brcb(dstTensor[num * MAX_REPEAT_TIMES * NUM_EIGHT * NUM_EIGHT], srcTensor[num * MAX_REPEAT_TIMES * NUM_EIGHT], brcbRepeat % MAX_REPEAT_TIMES, {1,FLOAT_TO_BLOCK});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MultiSelfAdd64(LocalTensor<float> dstTensor, ComputeStruct computeStruct) {
        for(uint32_t i = 1; i < rowParts - 1; ++i){
            Add(dstTensor, dstTensor[i * BLOCK_ALIGN_SIZE], dstTensor, BLOCK_ALIGN_SIZE, computeStruct.repeatTimes, {1,1,1,computeStruct.repStrid,computeStruct.repStrid,computeStruct.repStrid});
            PipeBarrier<PIPE_V>();
        }

        for(uint32_t i = 0; i < computeStruct.rowCycleNum; ++i){
            uint32_t dstOffset = i * rowAlign * MAX_REPEAT_TIMES;
            if (rowParts > 1) {
                Add(dstTensor[dstOffset], dstTensor[rowTailOffset + dstOffset], dstTensor[dstOffset], rowTailMask, MAX_REPEAT_TIMES, {1,1,1,computeStruct.repStrid,computeStruct.repStrid,computeStruct.repStrid});
                PipeBarrier<PIPE_V>();
            }
        }
        if (rowParts > 1) {
            Add(dstTensor[computeStruct.rowAlignOffset], dstTensor[rowTailOffset + computeStruct.rowAlignOffset], dstTensor[computeStruct.rowAlignOffset], rowTailMask, computeStruct.rowCycleTailNum, {1,1,1,computeStruct.repStrid,computeStruct.repStrid,computeStruct.repStrid});
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void MultiAdd(LocalTensor<float> dstTensor, LocalTensor<float> srcTensor, ComputeStruct computeStruct) {
        for(uint32_t i = 0; i < computeStruct.rowCycleNum; ++i){
            uint32_t dstOffset = i * rowAlign * MAX_REPEAT_TIMES;
            uint32_t srcOffset = i * NUM_EIGHT * MAX_REPEAT_TIMES;
            for(uint32_t j = 0; j < rowParts - 1; ++j){
                uint32_t offset = j * BLOCK_ALIGN_SIZE;
                Add(dstTensor[offset + dstOffset], srcTensor[srcOffset], dstTensor[offset + dstOffset], BLOCK_ALIGN_SIZE, MAX_REPEAT_TIMES, {1, 0, 1, computeStruct.repStrid, 1, computeStruct.repStrid});
            }
            Add(dstTensor[rowTailOffset + dstOffset], srcTensor[srcOffset], dstTensor[rowTailOffset + dstOffset], rowTailMask, MAX_REPEAT_TIMES, {1, 0, 1, computeStruct.repStrid, 1, computeStruct.repStrid});
        }
        
        for(uint32_t j = 0; j < rowParts - 1; ++j){
            uint32_t offset = j * BLOCK_ALIGN_SIZE;
            Add(dstTensor[offset + computeStruct.rowAlignOffset], srcTensor[computeStruct.blockOffset], dstTensor[offset + computeStruct.rowAlignOffset], BLOCK_ALIGN_SIZE, computeStruct.rowCycleTailNum, {1, 0, 1, computeStruct.repStrid, 1, computeStruct.repStrid});
        }
        Add(dstTensor[rowTailOffset + computeStruct.rowAlignOffset], srcTensor[computeStruct.blockOffset], dstTensor[rowTailOffset + computeStruct.rowAlignOffset], rowTailMask, computeStruct.rowCycleTailNum, {1, 0, 1, computeStruct.repStrid, 1, computeStruct.repStrid});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MultiMul(LocalTensor<float> dstTensor, LocalTensor<float> srcTensor, ComputeStruct computeStruct) {
        for(uint32_t i = 0; i < computeStruct.rowCycleNum; ++i){
            uint32_t dstOffset = i * rowAlign * MAX_REPEAT_TIMES;
            uint32_t srcOffset = i * NUM_EIGHT * MAX_REPEAT_TIMES;
            for(uint32_t j = 0; j < rowParts - 1; ++j){
                uint32_t offset = j * BLOCK_ALIGN_SIZE;
                Mul(dstTensor[offset + dstOffset], srcTensor[srcOffset], dstTensor[offset + dstOffset], BLOCK_ALIGN_SIZE, MAX_REPEAT_TIMES, {1, 0, 1, computeStruct.repStrid, 1, computeStruct.repStrid});
            }
            Mul(dstTensor[rowTailOffset + dstOffset], srcTensor[srcOffset], dstTensor[rowTailOffset + dstOffset], rowTailMask, MAX_REPEAT_TIMES, {1, 0, 1, computeStruct.repStrid, 1, computeStruct.repStrid});
        }
        
        for(uint32_t j = 0; j < rowParts - 1; ++j){
            uint32_t offset = j * BLOCK_ALIGN_SIZE;
            Mul(dstTensor[offset + computeStruct.rowAlignOffset], srcTensor[computeStruct.blockOffset], dstTensor[offset + computeStruct.rowAlignOffset], BLOCK_ALIGN_SIZE, computeStruct.rowCycleTailNum, {1, 0, 1, computeStruct.repStrid, 1, computeStruct.repStrid});
        }

        Mul(dstTensor[rowTailOffset + computeStruct.rowAlignOffset], srcTensor[computeStruct.blockOffset], dstTensor[rowTailOffset + computeStruct.rowAlignOffset], rowTailMask, computeStruct.rowCycleTailNum, {1, 0, 1, computeStruct.repStrid, 1, computeStruct.repStrid});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MultiMulAdd(LocalTensor<float> dstTensor, LocalTensor<float> srcTensor, ComputeStruct computeStruct) {
        // calculate y = x * gamma
        for(uint32_t i = 0; i < computeStruct.rowCycleNum; ++i){
            uint32_t dstOffset = i * rowAlign * MAX_REPEAT_TIMES;
            for(uint32_t j = 0; j < rowParts - 1; ++j){
                uint32_t offset = j * BLOCK_ALIGN_SIZE;
                Mul(dstTensor[offset + dstOffset], srcTensor[offset], dstTensor[offset + dstOffset], BLOCK_ALIGN_SIZE, MAX_REPEAT_TIMES, {1, 1, 1, computeStruct.repStrid, 0, computeStruct.repStrid});
                PipeBarrier<PIPE_V>();
                Add(dstTensor[offset + dstOffset], srcTensor[offset + rowAlign], dstTensor[offset + dstOffset], BLOCK_ALIGN_SIZE, MAX_REPEAT_TIMES, {1, 1, 1, computeStruct.repStrid, 0, computeStruct.repStrid});
            }
            Mul(dstTensor[rowTailOffset + dstOffset], srcTensor[rowTailOffset], dstTensor[rowTailOffset + dstOffset], rowTailMask, MAX_REPEAT_TIMES, {1, 1, 1, computeStruct.repStrid, 0, computeStruct.repStrid});
            PipeBarrier<PIPE_V>();
            Add(dstTensor[rowTailOffset + dstOffset], srcTensor[rowTailOffset + rowAlign], dstTensor[rowTailOffset + dstOffset], rowTailMask, MAX_REPEAT_TIMES, {1, 1, 1, computeStruct.repStrid, 0, computeStruct.repStrid});
        }
        for(uint32_t i = 0; i < rowParts - 1; ++i){
            uint32_t offset = i * BLOCK_ALIGN_SIZE;
            Mul(dstTensor[offset + computeStruct.rowAlignOffset], srcTensor[offset], dstTensor[offset + computeStruct.rowAlignOffset], BLOCK_ALIGN_SIZE, computeStruct.rowCycleTailNum, {1, 1, 1, computeStruct.repStrid, 0, computeStruct.repStrid});
            PipeBarrier<PIPE_V>();
            Add(dstTensor[offset + computeStruct.rowAlignOffset], srcTensor[offset + rowAlign], dstTensor[offset + computeStruct.rowAlignOffset], BLOCK_ALIGN_SIZE, computeStruct.rowCycleTailNum, {1, 1, 1, computeStruct.repStrid, 0, computeStruct.repStrid});
        }
        Mul(dstTensor[rowTailOffset + computeStruct.rowAlignOffset], srcTensor[rowTailOffset], dstTensor[rowTailOffset + computeStruct.rowAlignOffset], rowTailMask, computeStruct.rowCycleTailNum, {1, 1, 1, computeStruct.repStrid, 0, computeStruct.repStrid});
        PipeBarrier<PIPE_V>();
        Add(dstTensor[rowTailOffset + computeStruct.rowAlignOffset], srcTensor[rowTailOffset + rowAlign], dstTensor[rowTailOffset + computeStruct.rowAlignOffset], rowTailMask, computeStruct.rowCycleTailNum, {1, 1, 1, computeStruct.repStrid, 0, computeStruct.repStrid});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DataCopyInX(LocalTensor<float>& dstTensor, GlobalTensor<Tfm> srcTensor, DataCopyStruct dataCopyStruct, uint32_t tileLength) {
        // load srcTensor to dstTensor
        DataCopyInContiguous(dstTensor.ReinterpretCast<Tfm>(), srcTensor, dataCopyStruct, tileLength);
        inQueueX.EnQue(dstTensor);
        dstTensor = inQueueX.DeQue<float>();

        // cast dstTensor to float
        if constexpr (sizeof(Tfm) == FLOAT16_SIZE) {
            Cast(dstTensor, dstTensor.ReinterpretCast<Tfm>()[tileLength], RoundMode::CAST_NONE, tileLength);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void DataCopyOutY(GlobalTensor<Tfm> dstTensor, LocalTensor<float>& srcTensor, DataCopyStruct dataCopyStruct) {
        // cast xLocal to Tfm
        if constexpr (sizeof(Tfm) == FLOAT16_SIZE) {
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<Tfm, bfloat16_t>::value) {
                Cast(srcTensor.ReinterpretCast<Tfm>(), srcTensor, RoundMode::CAST_ROUND, tileLength);
            }
            if constexpr (std::is_same<Tfm, half>::value) {
                Cast(srcTensor.ReinterpretCast<Tfm>(), srcTensor, RoundMode::CAST_NONE, tileLength);
            }
        }

        // store srcTensor to dstTensor
        outQueueY.EnQue(srcTensor);
        srcTensor = outQueueY.DeQue<float>();
        DataCopyOutContiguous(dstTensor, srcTensor.ReinterpretCast<Tfm>(), dataCopyStruct);
        outQueueY.FreeTensor(srcTensor);
    }

    __aicore__ inline void processCompute(LocalTensor<float>& xLocal, LocalTensor<float>& yLocal, LocalTensor<float>& rstdLocal, ComputeStruct computeStruct, uint32_t brcbRepeat) {
        int32_t repeat = computeStruct.repeatTimes;   //行数，多行，一行的时候根据长度确认需要多少个元素
        int32_t dstRepStride = 1;   //目的操作数相邻迭代间的地址步长，以一个repeat归约后的长度为单位
        int32_t srcBlkStride = 1;   //单次迭代内datablock的地址步长
        int32_t srcRepStride = computeStruct.repStrid;   //源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目
        Adds(yLocal, xLocal, static_cast<float>(0), tileLength);
        PipeBarrier<PIPE_V>();
        // calculate square and muls coefficient (x-u)^2
        Mul(xLocal, yLocal, yLocal, tileLength);
        PipeBarrier<PIPE_V>();
        // calculate * (x-u)^2
        Muls(xLocal, xLocal, coefficient, tileLength);
        PipeBarrier<PIPE_V>();
        MultiSelfAdd64(xLocal, computeStruct);

        // repeat行数，多行
        //目的操作数相邻迭代间的地址步长，以一个repeat归约后的长度为单位
        //单次迭代内datablock的地址步长
        //源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目
        ReduceSumWithModeSameTensor(xLocal, xLocal, repeat, computeStruct.mask, 1, srcBlkStride, srcRepStride);
        PipeBarrier<PIPE_V>();

        // variance + epsilon
        Adds(xLocal, xLocal, static_cast<float>(eps), tileLength);
        PipeBarrier<PIPE_V>();

        // Sqrt (variance + epsilon)
        Sqrt(xLocal, xLocal, tileLength);
        PipeBarrier<PIPE_V>();

        // xLocal 前面计算出来的rowNum个连续rstd值，对齐32后，偏移后置为1
        uint32_t rowNumAlign = (computeStruct.repeatTimes + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        Duplicate(xLocal[rowNumAlign], static_cast<float>(1.0), rowNumAlign);
        PipeBarrier<PIPE_V>();
        // 用1/sqrt(variance + epsilon)^2
        Div(rstdLocal, xLocal[rowNumAlign], xLocal, computeStruct.repeatTimes);
        PipeBarrier<PIPE_V>();

        MultiBrcb(xLocal, rstdLocal, brcbRepeat);
        PipeBarrier<PIPE_V>();
        MultiMul(yLocal, xLocal, computeStruct);
    }

    __aicore__ inline void CopyInGamma(LocalTensor<float>& xLocal, DataCopyStruct dataCopyStruct) {
        if (!nullptrGamma) {
            // the 0th row has been processed, load gamma to the 0th row
            SetEvtFlag<HardEvent::V_MTE2>();
            // set load weight data copy pad params
            
            DataCopyInContiguous(xLocal.ReinterpretCast<Tweight>(), gammaGm, dataCopyStruct, rowAlign);
            SetEvtFlag<HardEvent::MTE2_V>();
            // if sizeof(Tweight) == 2, wait for gamma loaded, cast gamma to float
            if constexpr (sizeof(Tweight) == FLOAT16_SIZE) {
                Cast(xLocal, xLocal.ReinterpretCast<Tweight>()[rowAlign], RoundMode::CAST_NONE, rowAlign);
            }
        } else {
            Duplicate(xLocal, 1.0f, rowAlign);
        }
    }

    __aicore__ inline void ProcessBasicBlock(uint32_t rowNum, uint32_t currentBlockOffset, uint32_t currentParamOffset)
    {
        // allocate local tensor
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> meanLocal = outQueueMean.AllocTensor<float>();
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();

        DataCopyStruct dataCopyStruct1{rowAlign == rowSize ? 1 : rowNum, static_cast<uint32_t>(rowAlign == rowSize ? rowNum * rowSize * sizeof(Tfm) : rowSize * sizeof(Tfm)), 0, 0, rowAlign != rowSize, 0, static_cast<uint8_t>(rowAlign - rowSize)};
        DataCopyInX(xLocal, xGm[currentBlockOffset], dataCopyStruct1, tileLength);

        // calculate x * coefficient
        Muls(yLocal, xLocal, coefficient, tileLength);
        PipeBarrier<PIPE_V>();
        ComputeStruct computeStruct{rowAlign < BLOCK_ALIGN_SIZE ? rowTailMask : BLOCK_ALIGN_SIZE, static_cast<int32_t>(rowNum), static_cast<uint8_t>(rowAlign * FLOAT32_SIZE / BLOCK_SIZE), rowNum / MAX_REPEAT_TIMES,
                rowNum % MAX_REPEAT_TIMES, rowNum / MAX_REPEAT_TIMES * rowAlign * MAX_REPEAT_TIMES, rowNum / MAX_REPEAT_TIMES * NUM_EIGHT * MAX_REPEAT_TIMES};
        // for每次偏移 需要偏移一个mask块，
        MultiSelfAdd64(yLocal, computeStruct);
        
        ReduceSumWithModeDifTensor(meanLocal, yLocal, rowNum, computeStruct.mask, 1, 1, computeStruct.repStrid);

        // Brcb 
        uint32_t brcbRepeat = (rowNum + FLOAT_TO_BLOCK - 1) / FLOAT_TO_BLOCK;
        MultiBrcb(yLocal, meanLocal, brcbRepeat);
        // -mean
        Muls(yLocal, yLocal, static_cast<float>(-1), rowNum * FLOAT_TO_BLOCK);
        PipeBarrier<PIPE_V>();
        MultiAdd(xLocal, yLocal, computeStruct);

        // store meanLocal to meanGm
        SetEvtFlag<HardEvent::V_MTE3>();
        DataCopyStruct dataCopyStruct2{1, static_cast<uint32_t>(rowNum * sizeof(float)), 0, 0, false, 0, 0};
        DataCopyOutContiguous(meanGm[currentParamOffset], meanLocal, dataCopyStruct2);
        outQueueMean.FreeTensor(meanLocal);

        processCompute(xLocal, yLocal, rstdLocal, computeStruct, brcbRepeat);

        // store rstdLocal to rstdGm
        SetEvtFlag<HardEvent::V_MTE3>();
        DataCopyStruct dataCopyStruct4{1, static_cast<uint32_t>(rowNum * sizeof(float)), 0, 0, false, 0, 0};
        DataCopyOutContiguous(rstdGm[currentParamOffset], rstdLocal, dataCopyStruct4);
        outQueueRstd.FreeTensor(rstdLocal);

        DataCopyStruct dataCopyStruct3{1, static_cast<uint32_t>(rowSize * sizeof(Tweight)), 0, 0, rowAlign != rowSize, 0, static_cast<uint8_t>(rowAlign - rowSize)};
        CopyInGamma(xLocal, dataCopyStruct3);

        if (!nullptrBeta) {
            // the 1st row has been processed, load gamma to the 1st row
            SetEvtFlag<HardEvent::V_MTE2>();
            DataCopyInContiguous(xLocal.ReinterpretCast<Tweight>()[rowAlign], betaGm, dataCopyStruct3, FLOAT16_SIZE * rowAlign);
            SetEvtFlag<HardEvent::MTE2_V>();
            // if sizeof(Tweight) == 2, wait for beta loaded, cast beta to float
            if constexpr (sizeof(Tweight) == FLOAT16_SIZE) {
                Cast(xLocal[rowAlign], xLocal.ReinterpretCast<Tweight>()[rowAlign * NUM_THREE], RoundMode::CAST_NONE, rowAlign);
            }
        } else {
            Duplicate(xLocal[rowAlign], 0.0f, rowAlign);
        }

        PipeBarrier<PIPE_V>();
        MultiMulAdd(yLocal, xLocal, computeStruct);

        inQueueX.FreeTensor(xLocal);

        DataCopyStruct dataCopyStruct5{rowAlign == rowSize ? 1 : rowNum, static_cast<uint32_t>(rowAlign == rowSize ? rowNum * rowSize * sizeof(Tfm) : rowSize * sizeof(Tfm)), 0, 0, false, 0, 0};
        DataCopyOutY(yGm[currentBlockOffset], yLocal, dataCopyStruct5);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueMean;
    TQue<QuePosition::VECOUT, 1> outQueueRstd;

    GlobalTensor<Tweight> gammaGm;
    GlobalTensor<Tweight> betaGm;
    GlobalTensor<Tfm> xGm;
    GlobalTensor<Tfm> yGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<float> meanGm;

    uint64_t acc_val = 0;
    float value = 0.0;

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
    uint32_t remainTileLength = 0;
    uint32_t blockLength = 0;
    uint32_t nullptrGamma = 0;
    uint32_t nullptrBeta = 0;    
    uint32_t formerNum = 0;
    uint32_t formerRemainNum = 0;
    uint32_t tailNum = 0;
    uint32_t tailRemainNum = 0;
    uint32_t rowParts = 0;
    uint32_t rowTailOffset = 0;
    uint32_t rowTailMask = 0;
};

}  // namespace LayerNormCustomAlignLimit

#endif  // LAYER_NORM_CUSTOM_ALIGN_LIMIT_H