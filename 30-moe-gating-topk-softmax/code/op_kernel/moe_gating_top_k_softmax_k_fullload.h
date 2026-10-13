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
 * \file moe_gating_top_k_softmax_k_fullload.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_SOFTMAX_K_FULLLOAD
#define MOE_GATING_TOP_K_SOFTMAX_K_FULLLOAD

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace MoeGatingTopKSoftmax {
using namespace AscendC;

template <typename T, int32_t bufferNum>
class MoeGatingTopKSoftmaxKFullLoad
{
public:
    __aicore__ inline MoeGatingTopKSoftmaxKFullLoad(){};
    __aicore__ inline void Init(
        GM_ADDR gating, GM_ADDR finished, GM_ADDR out, GM_ADDR indicesOut, GM_ADDR sourceRowsOut, GM_ADDR workspace,
        const MoeGatingTopKSoftmaxKFullLoadTilingData* __restrict tilingData)
    {
        ParesTiling(tilingData);
        //  计算核块大小，获取当前核的起始索引
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
            pipe.InitBuffer(gatingQueue, bufferNum, ubFormerAlign * sizeof(float));
            pipe.InitBuffer(sourceRowsOutQueue, bufferNum, ubFormerAlign * sizeof(int32_t));
            pipe.InitBuffer(outQueue, bufferNum, (ubFormerAlign + kAlign) * sizeof(float));
        } else {
            pipe.InitBuffer(gatingQueue, bufferNum, ubFormerAlign * sizeof(T));
            pipe.InitBuffer(sourceRowsOutQueue, bufferNum, kAlign * sizeof(int32_t));
            pipe.InitBuffer(outQueue, bufferNum, (ubFormerAlign + kAlign) * sizeof(T));
        }
        pipe.InitBuffer(softmaxMaxTensorUb, ALIGNMENT_SIZE);
        pipe.InitBuffer(softmaxSumTensorUb, ALIGNMENT_SIZE);
        pipe.InitBuffer(expMaxTensorUb, ALIGNMENT_SIZE);
        pipe.InitBuffer(finishedQueue, bufferNum, ALIGNMENT_SIZE);
        pipe.InitBuffer(indicesOutQueue, bufferNum, (ubFormerAlign + kAlign) * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        int64_t ubLoopCount;
        int32_t curRowNum = ubFormer;

        ubLoopCount = (GetBlockIdx() < blockNum - 1) ? blockFormer : blockTail;

        int32_t curCol;
        for (int64_t rowIdx = 0; rowIdx < ubLoopCount; rowIdx++) {
            LocalTensor<int32_t> indicesOutLocal = indicesOutQueue.template AllocTensor<int32_t>();
            if (exitFinished) {
                CopyInFinished(rowIdx);
            }

            LocalTensor<bool> finishedLocal;
            if (exitFinished) {
                finishedLocal = finishedQueue.template DeQue<bool>();
            }

            if constexpr (IsSameType<T, bfloat16_t>::value) {
                LocalTensor<float> softmaxSumTensor = softmaxSumTensorUb.template Get<float>();
                LocalTensor<float> outLocal = outQueue.template AllocTensor<float>();

                for (int64_t ubIdx = 0; ubIdx < ubLoop; ubIdx++) {
                    curCol = (ubIdx < ubLoop - 1) ? ubFormer : ubTail;
                    CopyInGating(rowIdx, ubIdx, curCol);
                    ComputeBF16<float>(
                        rowIdx, ubIdx, curCol, 1, outLocal, indicesOutLocal, softmaxSumTensor, finishedLocal);
                }
                ComputeOutBF16<float>(outLocal, indicesOutLocal, softmaxSumTensor, finishedLocal, rowIdx);
            } else {
                LocalTensor<T> softmaxSumTensor = softmaxSumTensorUb.template Get<T>();
                LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();
                for (int64_t ubIdx = 0; ubIdx < ubLoop; ubIdx++) {
                    curCol = (ubIdx < ubLoop - 1) ? ubFormer : ubTail;
                    CopyInGating(rowIdx, ubIdx, curCol);
                    ComputeNotBF16<T>(
                        rowIdx, ubIdx, curCol, 1, outLocal, indicesOutLocal, softmaxSumTensor, finishedLocal);
                }
                ComputeOutNotBF16<T>(outLocal, indicesOutLocal, softmaxSumTensor, finishedLocal, rowIdx);
            }
            CopyOut(rowIdx);
            ComputeRowOut(rowIdx);
            CopyRowOut(rowIdx);
        }
    }

private:
    __aicore__ inline int64_t Align(const int64_t elementNum)
    {
        return (elementNum + ALIGN_FACTOR - 1) / ALIGN_FACTOR * ALIGN_FACTOR;
    }

    template <typename U>
    __aicore__ inline void VectorDup(LocalTensor<U>& dst, const int32_t colCount, const int32_t colCountAlign)
    {
        if (colCountAlign - colCount != 0) {
            // 当对齐后大小与实际大小不一致，需要将 colCount到colAlign之间的数据掩成-1
            uint64_t mask[2] = {
                (((uint64_t)1 << (colCountAlign - colCount)) - 1) << (ALIGN_FACTOR - (colCountAlign - colCount)), 0};
            Duplicate(dst[(kAlign + colCountAlign) - ALIGN_FACTOR], static_cast<U>(-1), mask, 1, 1, 1);
        }
    }

    __aicore__ inline void CopyInFinished(int32_t progress)
    {
        LocalTensor<bool> finishedLocal = finishedQueue.template AllocTensor<bool>();

        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = 1 * sizeof(bool);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        DataCopyPad(finishedLocal, finishedTensorGM[progress], intriParams, padParams);
        finishedQueue.EnQue(finishedLocal);
    }

    __aicore__ inline void CopyInGating(int32_t progress, int32_t ubIdx, int32_t curCol)
    {
        LocalTensor<T> gatingLocal = gatingQueue.template AllocTensor<T>();

        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = curCol * sizeof(T);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        DataCopyPad(gatingLocal, gatingTensorGM[col * progress + ubFormer * ubIdx], intriParams, padParams);
        gatingQueue.EnQue(gatingLocal);
    }

    template <typename U>
    __aicore__ inline void ComputeBF16(
        int32_t progress, int32_t ubIdx, int32_t curCol, int32_t curRowsNum, LocalTensor<U>& outLocal,
        LocalTensor<int32_t>& indicesOutLocal, LocalTensor<U>& softmaxSumTensor, LocalTensor<bool>& finishedLocal)
    {
        SoftMaxShapeInfo softmaxShapeInfoData;
        softmaxShapeInfoData.srcK = Align(curCol);
        softmaxShapeInfoData.srcM = curRowsNum;
        softmaxShapeInfoData.oriSrcK = curCol;
        softmaxShapeInfoData.oriSrcM = curRowsNum;
        TopKInfo topKInfoData;
        topKInfoData.outter = curRowsNum;
        topKInfoData.inner = (ubIdx < ubLoop - 1) ? kAlign + ubFormerAlign : kAlign + ubTailAlign;
        topKInfoData.n = (ubIdx < ubLoop - 1) ? kAlign + ubFormer : kAlign + ubTail;

        LocalTensor<T> gatingLocal = gatingQueue.template DeQue<T>();

        SoftMaxTiling* softmaxTilingData = (ubIdx < ubLoop - 1) ? &ubFormerSoftmaxTilingData : &ubTailSoftmaxTilingData;
        TopkTiling* topKTilingData = (ubIdx < ubLoop - 1) ? &topkFormerTilingData : &topkTailTilingData;

        LocalTensor<float> CastOutBuffer = sourceRowsOutQueue.template AllocTensor<float>();
        // curcol haven't dirty data
        Cast(CastOutBuffer, gatingLocal, RoundMode::CAST_NONE, curCol);
        gatingQueue.FreeTensor(gatingLocal);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> softmaxMaxTensor = softmaxMaxTensorUb.template Get<float>();
        LocalTensor<float> expMaxTensor = expMaxTensorUb.template Get<float>();
        LocalTensor<float> inExpSumTensor;
        LocalTensor<float> inMaxTensor;

        if (ubIdx == 0) {
            Duplicate(outLocal, static_cast<U>(-1), kAlign);
            PipeBarrier<PIPE_V>();
            SoftmaxFlashV2<float, false, false, false, false>(
                outLocal[kAlign], softmaxSumTensor, softmaxMaxTensor, CastOutBuffer, expMaxTensor, inExpSumTensor,
                inMaxTensor, *softmaxTilingData, softmaxShapeInfoData);
        } else {
            SoftmaxFlashV2<float, true, false, false, false>(
                outLocal[kAlign], softmaxSumTensor, softmaxMaxTensor, CastOutBuffer, expMaxTensor, softmaxSumTensor,
                softmaxMaxTensor, *softmaxTilingData, softmaxShapeInfoData);
            auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventID);
            WaitFlag<HardEvent::V_S>(eventID);
            Muls<float>(outLocal, outLocal, expMaxTensor.GetValue(0), kAlign);
        }
        PipeBarrier<PIPE_V>();
        sourceRowsOutQueue.FreeTensor(CastOutBuffer);

        int32_t curColAlign = (ubIdx < ubLoop - 1) ? ubFormerAlign : ubTailAlign;
        VectorDup<float>(outLocal, curCol, curColAlign);
        PipeBarrier<PIPE_V>();
        int32_t firstValue = ubIdx * ubFormer;
        ArithProgression(indicesOutLocal[kAlign], firstValue, 1, curColAlign);
        PipeBarrier<PIPE_V>();

        TopK<float, true, false, false, TopKMode::TOPK_NORMAL>(
            outLocal, indicesOutLocal, outLocal, indicesOutLocal, finishedLocal, kAlign, *topKTilingData, topKInfoData,
            true);
        PipeBarrier<PIPE_V>();

        return;
    }

    template <typename U>
    __aicore__ inline void ComputeNotBF16(
        int32_t progress, int32_t ubIdx, int32_t curCol, int32_t curRowsNum, LocalTensor<T>& outLocal,
        LocalTensor<int32_t>& indicesOutLocal, LocalTensor<U>& softmaxSumTensor, LocalTensor<bool>& finishedLocal)
    {
        SoftMaxShapeInfo softmaxShapeInfoData;
        softmaxShapeInfoData.srcK = Align(curCol);
        softmaxShapeInfoData.srcM = curRowsNum;
        softmaxShapeInfoData.oriSrcK = curCol;
        softmaxShapeInfoData.oriSrcM = curRowsNum;
        TopKInfo topKInfoData;
        topKInfoData.outter = curRowsNum;
        topKInfoData.inner = (ubIdx < ubLoop - 1) ? kAlign + ubFormerAlign : kAlign + ubTailAlign;
        topKInfoData.n = (ubIdx < ubLoop - 1) ? kAlign + ubFormer : kAlign + ubTail;

        LocalTensor<T> gatingLocal = gatingQueue.template DeQue<T>();

        SoftMaxTiling* softmaxTilingData = (ubIdx < ubLoop - 1) ? &ubFormerSoftmaxTilingData : &ubTailSoftmaxTilingData;
        TopkTiling* topKTilingData = (ubIdx < ubLoop - 1) ? &topkFormerTilingData : &topkTailTilingData;

        LocalTensor<T> softmaxMaxTensor = softmaxMaxTensorUb.template Get<T>();
        LocalTensor<T> expMaxTensor = expMaxTensorUb.template Get<T>();
        LocalTensor<T> inExpSumTensor;
        LocalTensor<T> inMaxTensor;

        if (ubIdx == 0) {
            Duplicate(outLocal, static_cast<U>(-1), kAlign);
            PipeBarrier<PIPE_V>();
            SoftmaxFlashV2<T, false, false, false, false>(
                outLocal[kAlign], softmaxSumTensor, softmaxMaxTensor, gatingLocal, expMaxTensor, inExpSumTensor,
                inMaxTensor, *softmaxTilingData, softmaxShapeInfoData);
        } else {
            SoftmaxFlashV2<T, true, false, false, false>(
                outLocal[kAlign], softmaxSumTensor, softmaxMaxTensor, gatingLocal, expMaxTensor, softmaxSumTensor,
                softmaxMaxTensor, *softmaxTilingData, softmaxShapeInfoData);
            auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventID);
            WaitFlag<HardEvent::V_S>(eventID);
            Muls<T>(outLocal, outLocal, expMaxTensor.GetValue(0), kAlign);
        }
        PipeBarrier<PIPE_V>();

        gatingQueue.FreeTensor(gatingLocal);

        int32_t curColAlign = (ubIdx < ubLoop - 1) ? ubFormerAlign : ubTailAlign;
        VectorDup<T>(outLocal, curCol, curColAlign);
        PipeBarrier<PIPE_V>();

        int32_t firstValue = ubIdx * ubFormer;
        ArithProgression(indicesOutLocal[kAlign], firstValue, 1, curColAlign);
        PipeBarrier<PIPE_V>();

        TopK<T, true, false, false, TopKMode::TOPK_NORMAL>(
            outLocal, indicesOutLocal, outLocal, indicesOutLocal, finishedLocal, kAlign, *topKTilingData, topKInfoData,
            true);
        PipeBarrier<PIPE_V>();

        return;
    }

    template <typename U>
    __aicore__ inline void ComputeOutBF16(
        LocalTensor<U>& outLocal, LocalTensor<int32_t>& indicesOutLocal, LocalTensor<U>& softmaxSumTensor,
        LocalTensor<bool>& finishedLocal, int32_t rowIdx)
    {
        auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventID);
        WaitFlag<HardEvent::V_S>(eventID);
        // if softmax reslut is nan，indicesOutLocal is 0 - (k-1)
        if (((softmaxSumTensor.template ReinterpretCast<int32_t>().GetValue(0) & 0x7f800000) == 0x7f800000) &&
            (softmaxSumTensor.template ReinterpretCast<int32_t>().GetValue(0) & 0x7fffff) > 0) {
            ArithProgression(indicesOutLocal, 0, 1, k);
        }
        PipeBarrier<PIPE_V>();

        if (exitFinished) {
            if (finishedLocal.GetValue(0)) {
                Duplicate(indicesOutLocal, static_cast<int32_t>(col), k);
            }
            finishedQueue.FreeTensor(finishedLocal);
        }
        PipeBarrier<PIPE_V>();

        Muls<U>(
            outLocal, outLocal, static_cast<U>(static_cast<U>(1.0) / static_cast<U>(softmaxSumTensor.GetValue(0))), k);
        PipeBarrier<PIPE_V>();

        Cast(outLocal.template ReinterpretCast<bfloat16_t>(), outLocal, RoundMode::CAST_ROUND, k);
        PipeBarrier<PIPE_V>();

        outQueue.EnQue(outLocal);
        indicesOutQueue.EnQue(indicesOutLocal);
    }

    template <typename U>
    __aicore__ inline void ComputeOutNotBF16(
        LocalTensor<T>& outLocal, LocalTensor<int32_t>& indicesOutLocal, LocalTensor<U>& softmaxSumTensor,
        LocalTensor<bool>& finishedLocal, int32_t rowIdx)
    {
        auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventID);
        WaitFlag<HardEvent::V_S>(eventID);

        // if softmax reslut is nan，indicesOutLocal is 0 - (k-1)
        if constexpr (IsSameType<U, float>::value) {
            // float32
            if (((softmaxSumTensor.template ReinterpretCast<int32_t>().GetValue(0) & 0x7f800000) == 0x7f800000) &&
                (softmaxSumTensor.template ReinterpretCast<int32_t>().GetValue(0) & 0x7fffff) > 0) {
                ArithProgression(indicesOutLocal, 0, 1, k);
            }
        } else if constexpr (IsSameType<U, half>::value) {
            // float16
            if (((softmaxSumTensor.template ReinterpretCast<int16_t>().GetValue(0) & 0x7f80) == 0x7f80) &&
                (softmaxSumTensor.template ReinterpretCast<int16_t>().GetValue(0) & 0x7fff) > 0) {
                ArithProgression(indicesOutLocal, 0, 1, k);
            }
        }
        PipeBarrier<PIPE_V>();

        if (exitFinished) {
            if (finishedLocal.GetValue(0)) {
                Duplicate(indicesOutLocal, static_cast<int32_t>(col), k);
            }
            finishedQueue.FreeTensor(finishedLocal);
        }
        PipeBarrier<PIPE_V>();

        Muls<T>(
            outLocal, outLocal,
            static_cast<T>(static_cast<float>(1.0) / static_cast<float>(softmaxSumTensor.GetValue(0))), k);
        PipeBarrier<PIPE_V>();

        outQueue.EnQue(outLocal);
        indicesOutQueue.EnQue(indicesOutLocal);
    }

    __aicore__ inline void ComputeRowOut(int32_t progress)
    {
        LocalTensor<int32_t> sourceRowsOutLocal = sourceRowsOutQueue.template AllocTensor<int32_t>();

        ArithProgression(
            sourceRowsOutLocal, static_cast<int32_t>(GetBlockIdx() * blockFormer + progress), static_cast<int32_t>(row),
            k);
        PipeBarrier<PIPE_V>();
        sourceRowsOutQueue.EnQue(sourceRowsOutLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<T> newOutLocal = outQueue.template DeQue<T>();
        LocalTensor<int32_t> newIndicesOutLocal = indicesOutQueue.template DeQue<int32_t>();

        int64_t gmIndex = k * progress;
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = k * sizeof(T);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        DataCopyPad(outTensorGM[gmIndex], newOutLocal, intriParams);
        intriParams.blockLen = k * sizeof(int32_t);
        DataCopyPad(indicesOutTensorGM[gmIndex], newIndicesOutLocal, intriParams);
        outQueue.FreeTensor(newOutLocal);
        indicesOutQueue.FreeTensor(newIndicesOutLocal);
    }

    __aicore__ inline void CopyRowOut(int32_t progress)
    {
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = k * sizeof(int32_t);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        LocalTensor<int32_t> sourceRowsOutLocal = sourceRowsOutQueue.template DeQue<int32_t>();

        DataCopyPad(sourceRowsOutTensorGM[k * progress], sourceRowsOutLocal, intriParams);
        sourceRowsOutQueue.FreeTensor(sourceRowsOutLocal);
    }

    __aicore__ inline void ParesTiling(const MoeGatingTopKSoftmaxKFullLoadTilingData* __restrict tilingData)
    {
        tilingKey = tilingData->tilingKey;
        row = tilingData->row;
        col = tilingData->col;
        k = tilingData->k;
        kAlign = tilingData->kAlign;
        blockNum = tilingData->blockNum;
        blockFormer = tilingData->blockFormer;
        blockTail = tilingData->blockTail;
        ubLoop = tilingData->ubLoop;
        ubFormer = tilingData->ubFormer;
        ubFormerAlign = tilingData->ubFormerAlign;
        ubTail = tilingData->ubTail;
        ubTailAlign = tilingData->ubTailAlign;
        ubFormerSoftmaxTilingData = tilingData->ubFormerSoftmaxTilingData;
        ubTailSoftmaxTilingData = tilingData->ubTailSoftmaxTilingData;
        topkFormerTilingData = tilingData->topkFormerTilingData;
        topkTailTilingData = tilingData->topkTailTilingData;
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> gatingQueue, finishedQueue;
    TQue<QuePosition::VECOUT, bufferNum> outQueue, indicesOutQueue, sourceRowsOutQueue;

    TBuf<> softmaxMaxTensorUb;
    TBuf<> softmaxSumTensorUb;
    TBuf<> expMaxTensorUb;

    GlobalTensor<T> gatingTensorGM;
    GlobalTensor<bool> finishedTensorGM;
    GlobalTensor<T> outTensorGM;
    GlobalTensor<int32_t> indicesOutTensorGM;
    GlobalTensor<int32_t> sourceRowsOutTensorGM;

    uint32_t tilingKey;
    uint32_t row;
    uint32_t col;
    uint32_t k;
    uint32_t kAlign;
    uint32_t blockNum;
    uint32_t blockFormer;
    uint32_t blockTail;
    uint32_t ubLoop;
    uint32_t ubFormer;
    uint32_t ubFormerAlign;
    uint32_t ubTail;
    uint32_t ubTailAlign;
    SoftMaxTiling ubFormerSoftmaxTilingData;
    SoftMaxTiling ubTailSoftmaxTilingData;
    TopkTiling topkFormerTilingData;
    TopkTiling topkTailTilingData;

    bool exitFinished{false};
};
} // namespace MoeGatingTopKSoftmax
#endif
