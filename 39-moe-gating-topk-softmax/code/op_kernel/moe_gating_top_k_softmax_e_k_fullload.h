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
 * \file moe_gating_top_k_softmax_e_k_fullload.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_SOFTMAX_E_K_FULLOAD
#define MOE_GATING_TOP_K_SOFTMAX_E_K_FULLOAD

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace MoeGatingTopKSoftmax {
using namespace AscendC;

constexpr int32_t ALIGNMENT_SIZE = 32;
constexpr int64_t REPEAT_MAX = 255;
constexpr int64_t COUNTINUOUS_DATA = 256;
constexpr int64_t FLOAT_BYTES = 4;
constexpr int64_t FLOAT_MASK = 64;
constexpr int64_t ALIGN_FACTOR = 32;

template <typename T, int32_t bufferNum>
class MoeGatingTopKSoftmaxEKFullLoad
{
public:
    __aicore__ inline MoeGatingTopKSoftmaxEKFullLoad(){};
    __aicore__ inline void Init(
        GM_ADDR gating, GM_ADDR finished, GM_ADDR out, GM_ADDR indicesOut, GM_ADDR sourceRowsOut, GM_ADDR workspace,
        const MoeGatingTopKSoftmaxEKFullLoadTilingData* __restrict tilingData)
    {
        ParesTiling(tilingData);
        //   计算核块大小，获取当前核的起始索引
        int64_t formerblockLength = blockFormer * col;
        int64_t blockLength = (GetBlockIdx() != blockNum - 1) ? formerblockLength : blockTail * col;
        gatingTensorGM.SetGlobalBuffer((__gm__ T*)gating + formerblockLength * GetBlockIdx(), blockLength);
        if (finished != nullptr) {
            exitFinished = true;
            int64_t blockLengthFinished = (GetBlockIdx() != blockNum - 1) ? blockFormer : blockTail;
            finishedTensorGM.SetGlobalBuffer((__gm__ bool*)finished + blockFormer * GetBlockIdx(), blockLengthFinished);
        }

        int64_t outFormerBlockLength = blockFormer * k;
        int64_t outBlockLength = (GetBlockIdx() != blockNum - 1) ? outFormerBlockLength : blockTail * k;
        outTensorGM.SetGlobalBuffer((__gm__ T*)out + outFormerBlockLength * GetBlockIdx(), outBlockLength);
        indicesOutTensorGM.SetGlobalBuffer(
            (__gm__ int32_t*)indicesOut + outFormerBlockLength * GetBlockIdx(), outBlockLength);
        sourceRowsOutTensorGM.SetGlobalBuffer(
            (__gm__ int32_t*)sourceRowsOut + outFormerBlockLength * GetBlockIdx(), outBlockLength);

        if constexpr (IsSameType<T, bfloat16_t>::value) {
            pipe.InitBuffer(gatingQueue, bufferNum, ubFormer * colAlign * sizeof(float));
        } else {
            pipe.InitBuffer(gatingQueue, bufferNum, ubFormer * colAlign * sizeof(T));
        }

        pipe.InitBuffer(
            finishedQueue, bufferNum, (ubFormer * sizeof(bool) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * ALIGNMENT_SIZE);
        pipe.InitBuffer(
            outQueue, bufferNum, ubFormer * ((k * sizeof(T) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE) * ALIGNMENT_SIZE);
        pipe.InitBuffer(indicesOutQueue, bufferNum, ubFormer * kAlignB32);
        pipe.InitBuffer(sourceRowsOutQueue, bufferNum, ubFormer * colAlign * sizeof(int32_t));
        pipe.InitBuffer(oneDimTensorUb, colAlign * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        int64_t ubLoopCount;
        int32_t curRowNum = ubFormer;
        SoftMaxTiling* softmaxTilingData = &formerSoftmaxTilingData;
        TopkTiling* topKTilingData = &formerTopkTilingData;
        ubLoopCount = (GetBlockIdx() == blockNum - 1) ? ubLoopOfTailBlock : ubLoopOfFormerBlock;

        for (int64_t i = 0; i < ubLoopCount - 1; i++) {
            CopyIn(i, ubFormer);
            Compute(i, ubFormer, softmaxTilingData, topKTilingData);
            CopyOut(i, ubFormer);
        }

        if (GetBlockIdx() < blockNum - 1) {
            curRowNum = ubTailOfFormerBlock;
            softmaxTilingData = &formerBlockTailSoftmaxTilingData;
            topKTilingData = &formerBlockTailTopkTilingData;
        } else {
            curRowNum = ubTailOfTailBlock;
            softmaxTilingData = &tailBlockTailSoftmaxTilingData;
            topKTilingData = &tailBlockTailTopkTilingData;
        }
        CopyIn(ubLoopCount - 1, curRowNum);
        Compute(ubLoopCount - 1, curRowNum, softmaxTilingData, topKTilingData);
        CopyOut(ubLoopCount - 1, curRowNum);
    }

private:
    __aicore__ inline void CastComputeAlignmentBefore(
        LocalTensor<float>& dst, LocalTensor<bfloat16_t>& src, const int32_t rowCount, const int32_t colCount)
    {
        int32_t repeatTimesOneRow = (colCount * sizeof(float) + COUNTINUOUS_DATA - 1) / COUNTINUOUS_DATA;
        int32_t colCountAlign = (colCount + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * ALIGNMENT_SIZE;
        UnaryRepeatParams repeatParams;
        if (rowCount > repeatTimesOneRow - 1 ||
            (colCount + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * sizeof(float) > REPEAT_MAX) {
            for (int32_t i = 0; i < rowCount; i++) {
                repeatParams.dstBlkStride = 1;
                repeatParams.srcBlkStride = 1;
                repeatParams.dstRepStride = COUNTINUOUS_DATA / ALIGNMENT_SIZE;
                repeatParams.srcRepStride = (COUNTINUOUS_DATA / FLOAT_BYTES) * sizeof(bfloat16_t) / ALIGNMENT_SIZE;
                Cast(
                    dst[i * ((colCount + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * ALIGNMENT_SIZE)],
                    src[i * ((colCount + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * ALIGNMENT_SIZE)], RoundMode::CAST_NONE,
                    FLOAT_MASK, repeatTimesOneRow - 1, repeatParams);
            }

            repeatParams.dstBlkStride = 1;
            repeatParams.srcBlkStride = 1;
            repeatParams.dstRepStride = (colCount + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * sizeof(float);
            repeatParams.srcRepStride = (colCount + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * sizeof(bfloat16_t);
            int32_t mask = colCount - FLOAT_MASK * (repeatTimesOneRow - 1);
            if ((colCount + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * sizeof(float) <= REPEAT_MAX) {
                Cast(
                    dst[(repeatTimesOneRow - 1) * COUNTINUOUS_DATA / FLOAT_BYTES],
                    src[(repeatTimesOneRow - 1) * COUNTINUOUS_DATA / FLOAT_BYTES], RoundMode::CAST_NONE, mask, rowCount,
                    repeatParams);
            } else {
                for (int32_t i = 0; i < rowCount; i++) {
                    Cast(
                        dst[(repeatTimesOneRow - 1) * COUNTINUOUS_DATA / FLOAT_BYTES + i * colCountAlign],
                        src[(repeatTimesOneRow - 1) * COUNTINUOUS_DATA / FLOAT_BYTES + i * colCountAlign],
                        RoundMode::CAST_NONE, mask, 1, {1, 1, 1, 1});
                }
            }
        } else {
            for (int32_t i = 0; i < repeatTimesOneRow - 1; i++) {
                repeatParams.dstBlkStride = 1;
                repeatParams.srcBlkStride = 1;
                repeatParams.dstRepStride = (colCount + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * sizeof(float);
                repeatParams.srcRepStride = (colCount + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * sizeof(bfloat16_t);
                Cast(
                    dst[i * (COUNTINUOUS_DATA / FLOAT_BYTES)], src[i * (COUNTINUOUS_DATA / FLOAT_BYTES)],
                    RoundMode::CAST_NONE, FLOAT_MASK, rowCount, repeatParams);
            }

            int32_t mask = colCount - FLOAT_MASK * (repeatTimesOneRow - 1);
            Cast(
                dst[(repeatTimesOneRow - 1) * (COUNTINUOUS_DATA / FLOAT_BYTES)],
                src[(repeatTimesOneRow - 1) * (COUNTINUOUS_DATA / FLOAT_BYTES)], RoundMode::CAST_NONE, mask, rowCount,
                repeatParams);
        }
    }
    __aicore__ inline void CastComputeAlignmentAfter(
        LocalTensor<bfloat16_t>& dst, LocalTensor<float>& src, const int32_t rowCount, const int32_t colCount)
    {
        int32_t repeatTimesOneRow = (colCount * sizeof(float) + COUNTINUOUS_DATA - 1) / COUNTINUOUS_DATA;

        UnaryRepeatParams repeatParams;
        if (rowCount > repeatTimesOneRow - 1) {
            for (int32_t i = 0; i < rowCount; i++) {
                repeatParams.dstBlkStride = 1;
                repeatParams.srcBlkStride = 1;
                repeatParams.dstRepStride = (COUNTINUOUS_DATA / FLOAT_BYTES) * sizeof(bfloat16_t) / ALIGNMENT_SIZE;
                repeatParams.srcRepStride = COUNTINUOUS_DATA / ALIGNMENT_SIZE;
                Cast(
                    dst[i * ((colCount * sizeof(bfloat16_t) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * ALIGNMENT_SIZE) /
                        sizeof(bfloat16_t)],
                    src[i * ((colCount * sizeof(float) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE * ALIGNMENT_SIZE) /
                        sizeof(float)],
                    RoundMode::CAST_ROUND, FLOAT_MASK, repeatTimesOneRow - 1, repeatParams);
            }

            repeatParams.dstBlkStride = 1;
            repeatParams.srcBlkStride = 1;
            repeatParams.dstRepStride = (colCount * sizeof(bfloat16_t) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE;
            repeatParams.srcRepStride = (colCount * sizeof(float) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE;
            int32_t mask =
                FLOAT_MASK - (((repeatTimesOneRow * COUNTINUOUS_DATA) - (colCount * FLOAT_BYTES)) / FLOAT_BYTES);
            Cast(
                dst[(repeatTimesOneRow - 1) * (COUNTINUOUS_DATA / FLOAT_BYTES)],
                src[(repeatTimesOneRow - 1) * (COUNTINUOUS_DATA / FLOAT_BYTES)], RoundMode::CAST_ROUND, mask, rowCount,
                repeatParams);
        } else {
            for (int32_t i = 0; i < repeatTimesOneRow - 1; i++) {
                repeatParams.dstBlkStride = 1;
                repeatParams.srcBlkStride = 1;
                repeatParams.dstRepStride = (colCount * sizeof(bfloat16_t) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE;
                repeatParams.srcRepStride = (colCount * sizeof(float) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE;
                Cast(
                    dst[i * (COUNTINUOUS_DATA / FLOAT_BYTES)], src[i * (COUNTINUOUS_DATA / FLOAT_BYTES)],
                    RoundMode::CAST_ROUND, FLOAT_MASK, rowCount, repeatParams);
            }

            int32_t mask =
                FLOAT_MASK - (((repeatTimesOneRow * COUNTINUOUS_DATA) - (colCount * FLOAT_BYTES)) / FLOAT_BYTES);
            Cast(
                dst[(repeatTimesOneRow - 1) * (COUNTINUOUS_DATA / FLOAT_BYTES)],
                src[(repeatTimesOneRow - 1) * (COUNTINUOUS_DATA / FLOAT_BYTES)], RoundMode::CAST_ROUND, mask, rowCount,
                repeatParams);
        }
    }

    template <typename U>
    __aicore__ inline void VectorDup(LocalTensor<U>& dst, const int32_t rowCount, const int32_t colCount)
    {
        if (colAlign - colCount != 0) {
            // 当对齐后大小与实际大小不一致，需要将 colCount到colAlign之间的数据掩成-1
            uint64_t mask[2] = {
                (((uint64_t)1 << (colAlign - colCount)) - 1) << (ALIGN_FACTOR - (colAlign - colCount)), 0};
            if (colAlign * sizeof(U) / ALIGNMENT_SIZE <= REPEAT_MAX) {
                Duplicate(
                    dst[colAlign - ALIGN_FACTOR], static_cast<U>(-1), mask, rowCount, 1,
                    colAlign * sizeof(U) / ALIGNMENT_SIZE);
            } else {
                for (int32_t i = 0; i < rowCount; i++) {
                    Duplicate(dst[i * colAlign + colAlign - ALIGN_FACTOR], static_cast<U>(-1), mask, 1, 1, 1);
                }
            }
        }
    }

    __aicore__ inline void AddsComputeAlignment(
        LocalTensor<int32_t>& dst, LocalTensor<int32_t>& src, const int32_t rowCount, const int32_t colCount)
    {
        int32_t repeatTimesOneRow = (colCount * FLOAT_BYTES + COUNTINUOUS_DATA - 1) / COUNTINUOUS_DATA;
        int32_t countOneRow32B = (colCount * FLOAT_BYTES + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE;

        UnaryRepeatParams repeatParams;

        for (int32_t i = 1; i < rowCount; i++) {
            repeatParams.dstBlkStride = 1;
            repeatParams.srcBlkStride = 1;
            repeatParams.dstRepStride = COUNTINUOUS_DATA / ALIGNMENT_SIZE;
            repeatParams.srcRepStride = COUNTINUOUS_DATA / ALIGNMENT_SIZE;

            if ((repeatTimesOneRow - 1) > 0) {
                Adds(
                    dst[i * ((countOneRow32B * ALIGNMENT_SIZE) / FLOAT_BYTES)], src[0], i, FLOAT_MASK,
                    repeatTimesOneRow - 1, repeatParams);
            }
            int32_t mask =
                FLOAT_MASK - (((repeatTimesOneRow * COUNTINUOUS_DATA) - (colCount * FLOAT_BYTES)) / FLOAT_BYTES);
            Adds(
                dst[i * (countOneRow32B * ALIGNMENT_SIZE / FLOAT_BYTES) +
                    (repeatTimesOneRow - 1) * (COUNTINUOUS_DATA / FLOAT_BYTES)],
                src[(repeatTimesOneRow - 1) * (COUNTINUOUS_DATA / FLOAT_BYTES)], i, mask, 1, repeatParams);
        }
    }

    __aicore__ inline void CopyIn(int32_t progress, int32_t curRowsNum)
    {
        LocalTensor<T> gatingLocal = gatingQueue.template AllocTensor<T>();

        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = curRowsNum;
        intriParams.blockLen = col * sizeof(T);
        intriParams.srcStride = 0;
        intriParams.dstStride =
            (colAlign * sizeof(T) - (((col * sizeof(T) + ALIGNMENT_SIZE - 1) / ALIGNMENT_SIZE) * ALIGNMENT_SIZE)) /
            ALIGNMENT_SIZE;

        DataCopyPad(gatingLocal, gatingTensorGM[ubFormer * col * progress], intriParams, padParams);
        gatingQueue.EnQue(gatingLocal);

        if (exitFinished) {
            LocalTensor<bool> finishedLocal = finishedQueue.template AllocTensor<bool>();
            DataCopyParams intriParamsFinished;
            intriParamsFinished.blockCount = 1;
            intriParamsFinished.blockLen = curRowsNum * sizeof(bool);
            intriParamsFinished.srcStride = 0;
            intriParamsFinished.dstStride = 0;

            DataCopyPad(finishedLocal, finishedTensorGM[ubFormer * progress], intriParamsFinished, padParams);
            finishedQueue.EnQue(finishedLocal);
        }
    }

    __aicore__ inline void Compute(
        int32_t progress, int32_t curRowsNum, const SoftMaxTiling* softmaxTilingData, const TopkTiling* topKTilingData)
    {
        SoftMaxShapeInfo softmaxShapeInfoData;
        softmaxShapeInfoData.srcK = colAlign;
        softmaxShapeInfoData.srcM = curRowsNum;
        softmaxShapeInfoData.oriSrcK = col;
        softmaxShapeInfoData.oriSrcM = curRowsNum;
        TopKInfo topKInfoData;
        topKInfoData.outter = curRowsNum;
        topKInfoData.inner = colAlign;
        topKInfoData.n = col;

        LocalTensor<T> gatingLocal = gatingQueue.template DeQue<T>();
        LocalTensor<bool> finishedLocal;
        LocalTensor<int32_t> srcIndexLocal;
        if (exitFinished) {
            finishedLocal = finishedQueue.template DeQue<bool>();
        }

        LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();
        LocalTensor<int32_t> indicesOutLocal = indicesOutQueue.template AllocTensor<int32_t>();
        LocalTensor<int32_t> oneDimTensor = oneDimTensorUb.template Get<int32_t>();

        if constexpr (IsSameType<T, bfloat16_t>::value) {
            LocalTensor<float> CastOutBuffer = sourceRowsOutQueue.template AllocTensor<float>();
            CastComputeAlignmentBefore(CastOutBuffer, gatingLocal, curRowsNum, col);
            gatingQueue.FreeTensor(gatingLocal);
            PipeBarrier<PIPE_V>();

            LocalTensor<float> softmaxOutBuffer = gatingQueue.template AllocTensor<float>();
            SoftMax<float, true, false>(softmaxOutBuffer, CastOutBuffer, *softmaxTilingData, softmaxShapeInfoData);
            PipeBarrier<PIPE_V>();

            VectorDup<float>(softmaxOutBuffer, curRowsNum, col);
            PipeBarrier<PIPE_V>();

            ArithProgression(oneDimTensor, 0, 1, colAlign);
            PipeBarrier<PIPE_V>();

            if (exitFinished) {
                TopK<float, true, true, true, TopKMode::TOPK_NORMAL>(
                    CastOutBuffer, indicesOutLocal, softmaxOutBuffer, oneDimTensor, finishedLocal, k, *topKTilingData,
                    topKInfoData, true);
            } else {
                TopK<float, true, false, true, TopKMode::TOPK_NORMAL>(
                    CastOutBuffer, indicesOutLocal, softmaxOutBuffer, oneDimTensor, finishedLocal, k, *topKTilingData,
                    topKInfoData, true);
            }
            PipeBarrier<PIPE_V>();

            CastComputeAlignmentAfter(outLocal, CastOutBuffer, curRowsNum, k);
            sourceRowsOutQueue.FreeTensor(CastOutBuffer);
            gatingQueue.FreeTensor(softmaxOutBuffer);
        } else {
            LocalTensor<T> softmaxOutBuffer = sourceRowsOutQueue.template AllocTensor<T>();
            SoftMax<T, true, false>(softmaxOutBuffer, gatingLocal, *softmaxTilingData, softmaxShapeInfoData);
            gatingQueue.FreeTensor(gatingLocal);
            PipeBarrier<PIPE_V>();

            VectorDup<T>(softmaxOutBuffer, curRowsNum, col);
            PipeBarrier<PIPE_V>();

            ArithProgression(oneDimTensor, 0, 1, colAlign);
            PipeBarrier<PIPE_V>();

            if (exitFinished) {
                TopK<T, true, true, true, TopKMode::TOPK_NORMAL>(
                    outLocal, indicesOutLocal, softmaxOutBuffer, oneDimTensor, finishedLocal, k, *topKTilingData,
                    topKInfoData, true);
            } else {
                TopK<T, true, false, true, TopKMode::TOPK_NORMAL>(
                    outLocal, indicesOutLocal, softmaxOutBuffer, oneDimTensor, finishedLocal, k, *topKTilingData,
                    topKInfoData, true);
            }
            PipeBarrier<PIPE_V>();
            sourceRowsOutQueue.FreeTensor(softmaxOutBuffer);
        }
        outQueue.EnQue(outLocal);
        indicesOutQueue.EnQue(indicesOutLocal);

        LocalTensor<T> newOutLocal = outQueue.template DeQue<T>();
        LocalTensor<int32_t> newIndicesOutLocal = indicesOutQueue.template DeQue<int32_t>();

        int64_t gmIndex = ubFormer * k * progress;
        DataCopyParams intriParams;
        intriParams.blockCount = curRowsNum;
        intriParams.blockLen = k * sizeof(T);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        DataCopyPad(outTensorGM[gmIndex], newOutLocal, intriParams);
        intriParams.blockLen = k * sizeof(int32_t);
        DataCopyPad(indicesOutTensorGM[gmIndex], newIndicesOutLocal, intriParams);
        outQueue.FreeTensor(newOutLocal);
        indicesOutQueue.FreeTensor(newIndicesOutLocal);

        if (exitFinished) {
            finishedQueue.FreeTensor(finishedLocal);
        }

        LocalTensor<int32_t> sourceRowsOutLocal = sourceRowsOutQueue.template AllocTensor<int32_t>();
        auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventID);
        WaitFlag<HardEvent::V_S>(eventID);
        ArithProgression(
            sourceRowsOutLocal, static_cast<int32_t>(GetBlockIdx() * blockFormer + progress * ubFormer),
            static_cast<int32_t>(row), k);
        PipeBarrier<PIPE_V>();
        AddsComputeAlignment(sourceRowsOutLocal, sourceRowsOutLocal, curRowsNum, k);
        sourceRowsOutQueue.EnQue(sourceRowsOutLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, int32_t curRowsNum)
    {
        DataCopyParams intriParams;
        intriParams.blockCount = curRowsNum;
        intriParams.blockLen = k * sizeof(int32_t);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        LocalTensor<int32_t> sourceRowsOutLocal = sourceRowsOutQueue.template DeQue<int32_t>();

        DataCopyPad(sourceRowsOutTensorGM[ubFormer * k * progress], sourceRowsOutLocal, intriParams);
        sourceRowsOutQueue.FreeTensor(sourceRowsOutLocal);
    }

    __aicore__ inline void ParesTiling(const MoeGatingTopKSoftmaxEKFullLoadTilingData* __restrict tilingData)
    {
        tilingKey = tilingData->tilingKey;
        row = tilingData->row;
        col = tilingData->col;
        colAlign = tilingData->colAlign;
        k = tilingData->k;
        kAlignB16 = tilingData->kAlignB16;
        kAlignB32 = tilingData->kAlignB32;
        blockNum = tilingData->blockNum;
        blockFormer = tilingData->blockFormer;
        blockTail = tilingData->blockTail;
        ubLoopOfFormerBlock = tilingData->ubLoopOfFormerBlock;
        ubLoopOfTailBlock = tilingData->ubLoopOfTailBlock;
        ubFormer = tilingData->ubFormer;
        ubTailOfFormerBlock = tilingData->ubTailOfFormerBlock;
        ubTailOfTailBlock = tilingData->ubTailOfTailBlock;
        formerSoftmaxTilingData = tilingData->formerSoftmaxTilingData;
        formerBlockTailSoftmaxTilingData = tilingData->formerBlockTailSoftmaxTilingData;
        tailBlockTailSoftmaxTilingData = tilingData->tailBlockTailSoftmaxTilingData;
        formerTopkTilingData = tilingData->formerTopkTilingData;
        formerBlockTailTopkTilingData = tilingData->formerBlockTailTopkTilingData;
        tailBlockTailTopkTilingData = tilingData->tailBlockTailTopkTilingData;
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> gatingQueue, finishedQueue;
    TQue<QuePosition::VECOUT, bufferNum> outQueue, indicesOutQueue, sourceRowsOutQueue;

    TBuf<> oneDimTensorUb;

    GlobalTensor<T> gatingTensorGM;
    GlobalTensor<bool> finishedTensorGM;
    GlobalTensor<T> outTensorGM;
    GlobalTensor<int32_t> indicesOutTensorGM;
    GlobalTensor<int32_t> sourceRowsOutTensorGM;

    uint32_t tilingKey;
    uint32_t row;
    uint32_t col;
    uint32_t colAlign;
    uint32_t k;
    uint32_t kAlignB16;
    uint32_t kAlignB32;
    uint32_t blockNum;
    uint32_t blockFormer;
    uint32_t blockTail;
    uint32_t ubLoopOfFormerBlock;
    uint32_t ubLoopOfTailBlock;
    uint32_t ubFormer;
    uint32_t ubTailOfFormerBlock;
    uint32_t ubTailOfTailBlock;
    SoftMaxTiling formerSoftmaxTilingData;
    SoftMaxTiling formerBlockTailSoftmaxTilingData;
    SoftMaxTiling tailBlockTailSoftmaxTilingData;
    TopkTiling formerTopkTilingData;
    TopkTiling formerBlockTailTopkTilingData;
    TopkTiling tailBlockTailTopkTilingData;

    bool exitFinished{false};
};
} // namespace MoeGatingTopKSoftmax
#endif
