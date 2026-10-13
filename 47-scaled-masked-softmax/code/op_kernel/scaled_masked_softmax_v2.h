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
 * \file scaled_masked_softmax_v2.h
 * \brief
 */

#ifndef SCALED_MASKED_SOFTMAX_V2_KERNEL_H
#define SCALED_MASKED_SOFTMAX_V2_KERNEL_H

#include "kernel_operator.h"

namespace AscendC {
constexpr float MASK_VAL = -10000.0f;

template <typename T>
class ScaledMaskedSoftmaxV2
{
public:
    struct MaskOffset {
        uint64_t batchOffset = 0;
        uint64_t channelOffset = 0;
        uint64_t lineOffset = 0;
        __aicore__ inline void NextChannel(uint64_t channelNum)
        {
            if (channelNum == 0) {
                batchOffset = 0;
                channelOffset = 0;
                lineOffset = 0;
                return;
            }
            channelOffset = (channelOffset + 1) % channelNum;
            if (channelOffset == 0) {
                batchOffset++;
            }
            lineOffset = 0;
        }
        __aicore__ inline uint64_t GetOffset(uint64_t realBatch, uint64_t realChannel, uint64_t realLine)
        {
            return batchOffset * realBatch + channelOffset * realChannel + lineOffset * realLine;
        }
    };

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mask, GM_ADDR y, const ScaledMaskedSoftmaxV2TilingData& tilingData)
    {
        this->blockIdx = GetBlockIdx();
        this->coreOffset = (this->blockIdx >= tilingData.headCoreNum ? tilingData.headCoreNum : this->blockIdx) *
                               (tilingData.lineHeadCore * tilingData.width) +
                           (this->blockIdx >= tilingData.headCoreNum ? this->blockIdx - tilingData.headCoreNum : 0) *
                               (tilingData.lineTailCore * tilingData.width);
        this->startLineOffset = this->coreOffset / tilingData.width;
        gmX.SetGlobalBuffer((__gm__ T*)x + this->coreOffset);
        gmY.SetGlobalBuffer((__gm__ T*)y + this->coreOffset);

        uint64_t bufSize = tilingData.padLineNum * tilingData.lineHeadIter;
        pipe.InitBuffer(inQueueX, 1, bufSize * sizeof(T));
        pipe.InitBuffer(outQueueY, 1, bufSize * sizeof(T));

        pipe.InitBuffer(bufX, bufSize * sizeof(float));
        uint64_t tmpBufSize = 32 * 1024; // 高阶api复用空间大小 32K
        if (bufSize > tmpBufSize) {
            tmpBufSize = 64 * 1024; // 高阶api复用空间大小 64 * 1024
        }
        pipe.InitBuffer(bufShare, tmpBufSize);

        gmMask.SetGlobalBuffer((__gm__ bool*)mask);

        pipe.InitBuffer(inQueueMask, 1, tilingData.alignedMaskWidth * tilingData.lineHeadIter * sizeof(bool));

        this->tilingData = tilingData;
        this->scale = tilingData.scale;

        this->xBatch = tilingData.batch;
        this->xChannel = tilingData.channel;
        this->xHeight = tilingData.height;
        this->xWidth = tilingData.width;

        this->linePerBatch = this->xChannel * this->xHeight;
        this->linesPerIter = tilingData.lineHeadIter;
        this->linesLastIter =
            this->blockIdx >= tilingData.headCoreNum ? tilingData.lineLastTailIter : tilingData.lineLastHeadIter;

        this->maskBatchOffset = tilingData.nStep * tilingData.maskChannel * this->xHeight * this->xWidth;
        this->maskChannelOffset = tilingData.cStep * this->xHeight * this->xWidth;
        this->gmOffsetPerIdx = this->xWidth * this->linesPerIter;
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= tilingData.coreNum) {
            return;
        }

        uint64_t loop = this->blockIdx >= tilingData.headCoreNum ? tilingData.iterTailCore : tilingData.iterHeadCore;
        uint64_t linePerIter = this->linesPerIter;
        for (uint64_t idx = 0u; idx < loop; ++idx) {
            // tail iter
            if (idx == loop - 1) {
                linePerIter = this->linesLastIter;
            }
            this->elePerIter = tilingData.padLineNum * linePerIter;
            CopyIn(idx, linePerIter);
            Compute(idx, linePerIter);
            CopyOut(idx, linePerIter);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t idx, uint64_t linePerIter)
    {
        CopyMaskIn(idx, linePerIter);
        CopyXIn(idx, linePerIter);
    }

    __aicore__ inline void CopyXIn(uint64_t idx, uint64_t linePerIter)
    {
        LocalTensor<T> xTensor = inQueueX.AllocTensor<T>();
        DataCopyExtParams params = {
            static_cast<uint16_t>(linePerIter), static_cast<uint32_t>(tilingData.width * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> extParams = {true, 0, static_cast<uint8_t>(tilingData.paddingNum), 0.0};
        DataCopyPad(xTensor, gmX[idx * gmOffsetPerIdx], params, extParams);
        inQueueX.EnQue(xTensor);
    }

    __aicore__ inline void CopyMaskIn(uint64_t idx, uint64_t linePerIter)
    {
        uint64_t curBatch = 0u;
        uint64_t curChannel = 0u;
        uint64_t curLineInChannel = 0u;
        uint64_t endBatch = 0u;
        uint64_t endChannel = 0u;
        uint64_t endLineInChannel = 0u;

        LocalTensor<bool> maskTensor = inQueueMask.AllocTensor<bool>();
        uint64_t curLine = this->startLineOffset + idx * this->linesPerIter;
        CalcCurPos(curLine, curBatch, curChannel, curLineInChannel);
        CalcEndPos(curLine, linePerIter, endBatch, endChannel, endLineInChannel);

        MaskOffset maskOffset = {curBatch, curChannel, curLineInChannel};
        uint64_t lineCnt = 0u;
        if (curBatch == endBatch) {
            if (curChannel == endChannel) {
                MoveMaskLines(maskTensor, lineCnt, maskOffset, linePerIter);
            } else {
                MoveMaskLines(maskTensor, lineCnt, maskOffset, this->xHeight - curLineInChannel);
                maskOffset.NextChannel(this->xChannel);
                for (uint64_t i = curChannel + 1; i < endChannel; ++i) {
                    MoveMaskLines(maskTensor, lineCnt, maskOffset, this->xHeight);
                    maskOffset.NextChannel(this->xChannel);
                }
                MoveMaskLines(maskTensor, lineCnt, maskOffset, endLineInChannel);
            }
        } else {
            // first channel
            MoveMaskLines(maskTensor, lineCnt, maskOffset, this->xHeight - curLineInChannel);
            maskOffset.NextChannel(this->xChannel);
            // first batch other channels
            for (uint64_t i = curChannel + 1; i < this->xChannel; ++i) {
                MoveMaskLines(maskTensor, lineCnt, maskOffset, this->xHeight);
                maskOffset.NextChannel(this->xChannel);
            }
            // whole batches loop
            for (uint64_t i = curBatch + 1; i < endBatch; ++i) {
                for (uint64_t j = 0; j < this->xChannel; ++j) {
                    MoveMaskLines(maskTensor, lineCnt, maskOffset, this->xHeight);
                    maskOffset.NextChannel(this->xChannel);
                }
            }
            // last batch other channels
            for (uint64_t i = 0; i < endChannel; ++i) {
                MoveMaskLines(maskTensor, lineCnt, maskOffset, this->xHeight);
                maskOffset.NextChannel(this->xChannel);
            }
            // last batch last channel
            MoveMaskLines(maskTensor, lineCnt, maskOffset, endLineInChannel);
        }
        inQueueMask.EnQue(maskTensor);
    }

    __aicore__ inline void MoveMaskLines(
        LocalTensor<bool>& maskTensor, uint64_t& lineCnt, MaskOffset& maskOffset, uint64_t lines)
    {
        uint64_t gmOffset = maskOffset.GetOffset(maskBatchOffset, maskChannelOffset, tilingData.maskWidth);
        DataCopyExtParams paramsMask = {
            static_cast<uint16_t>(lines), static_cast<uint32_t>(tilingData.maskWidth * sizeof(bool)), 0, 0, 0};
        DataCopyPadExtParams<bool> extParamsMask = {true, 0, static_cast<uint8_t>(tilingData.alignedMaskPadding), 0};
        DataCopyPad(maskTensor[lineCnt * tilingData.alignedMaskWidth], gmMask[gmOffset], paramsMask, extParamsMask);
        lineCnt += lines;
    }

    __aicore__ inline void CalcCurPos(
        uint64_t curLine, uint64_t& curBatch, uint64_t& curChannel, uint64_t& curLineInChannel)
    {
        curBatch = curLine / this->linePerBatch;
        curChannel = (curLine - curBatch * this->linePerBatch) / this->xHeight;
        curLineInChannel = curLine % this->xHeight;
    }

    __aicore__ inline void CalcEndPos(
        uint64_t curLine, uint64_t lines, uint64_t& endBatch, uint64_t& endChannel, uint64_t& endLineInChannel)
    {
        // 为了计算batch和channel的时候不进位 减一
        uint64_t endLine = curLine + lines - 1;
        endBatch = endLine / this->linePerBatch;
        endChannel = (endLine - endBatch * this->linePerBatch) / this->xHeight;
        // 计算行数末尾位置需要补充上1
        endLineInChannel = endLine % this->xHeight + 1;
    }

    __aicore__ inline void Compute(uint64_t idx, uint64_t linePerIter)
    {
        LocalTensor<T> xTensor = inQueueX.DeQue<T>();
        LocalTensor<bool> maskTensor = inQueueMask.DeQue<bool>();
        scaledMaskedX = bufX.Get<float>();
        sharedBuffer = bufShare.Get<uint8_t>();

        AscendC::SelectWithBytesMaskShapeInfo selectShapeInfo;
        selectShapeInfo.firstAxis = linePerIter;
        selectShapeInfo.srcLastAxis = tilingData.padLineNum;
        selectShapeInfo.maskLastAxis = tilingData.alignedMaskWidth;
        scaledMaskedX.SetSize(selectShapeInfo.firstAxis * selectShapeInfo.srcLastAxis);
        xTensor.SetSize(selectShapeInfo.firstAxis * selectShapeInfo.srcLastAxis);
        maskTensor.SetSize(selectShapeInfo.firstAxis * selectShapeInfo.maskLastAxis);
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            Cast(scaledMaskedX, xTensor, RoundMode::CAST_NONE, this->elePerIter);
            Muls(scaledMaskedX, scaledMaskedX, static_cast<float>(scale), this->elePerIter);
            AscendC::SelectWithBytesMask(
                scaledMaskedX, scaledMaskedX, MASK_VAL, maskTensor, sharedBuffer, selectShapeInfo);
        } else {
            Muls(xTensor, xTensor, static_cast<T>(scale), this->elePerIter);
            if constexpr (std::is_same<T, half>::value) {
                AscendC::SelectWithBytesMask(
                    xTensor, xTensor, static_cast<T>(MASK_VAL), maskTensor, sharedBuffer, selectShapeInfo);
                Cast(scaledMaskedX, xTensor, RoundMode::CAST_NONE, this->elePerIter);
            } else {
                AscendC::SelectWithBytesMask(
                    scaledMaskedX, xTensor, MASK_VAL, maskTensor, sharedBuffer, selectShapeInfo);
            }
        }

        LocalTensor<T> yTensor = outQueueY.AllocTensor<T>();
        yTensor.SetSize(selectShapeInfo.firstAxis * selectShapeInfo.srcLastAxis);
        if constexpr (!std::is_same<T, float>::value) {
            SoftmaxX(scaledMaskedX, scaledMaskedX, sharedBuffer, linePerIter);
            Cast(yTensor, scaledMaskedX, RoundMode::CAST_RINT, this->elePerIter);
        } else {
            SoftmaxX(yTensor, scaledMaskedX, sharedBuffer, linePerIter);
        }

        inQueueX.FreeTensor(xTensor);
        inQueueMask.FreeTensor(maskTensor);
        outQueueY.EnQue(yTensor);
    }

    __aicore__ inline void SoftmaxX(
        LocalTensor<float>& dstTensor, LocalTensor<float>& srcTensor, LocalTensor<uint8_t> sharedBuffer, uint64_t lines)
    {
        SoftMaxTiling softmaxTilingData = tilingData.softmaxTilingData;
        SoftMaxShapeInfo softmaxShapeInfoData = {
            static_cast<uint32_t>(lines),
            static_cast<uint32_t>(tilingData.padLineNum),
            static_cast<uint32_t>(lines),
            static_cast<uint32_t>(tilingData.width),
        };

        SoftMax<float, false, false>(dstTensor, srcTensor, sharedBuffer, softmaxTilingData, softmaxShapeInfoData);
    }

    __aicore__ inline void CopyOut(uint64_t idx, uint64_t linePerIter)
    {
        LocalTensor<T> yTensor = outQueueY.DeQue<T>();
        DataCopyExtParams params = {
            static_cast<uint16_t>(linePerIter), static_cast<uint32_t>(this->xWidth * sizeof(T)), 0, 0, 0};
        DataCopyPad(gmY[idx * gmOffsetPerIdx], yTensor, params);
        outQueueY.FreeTensor(yTensor);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueMask;

    TQue<QuePosition::VECOUT, 1> outQueueY;

    GlobalTensor<T> gmX;
    GlobalTensor<T> gmY;
    GlobalTensor<bool> gmMask;

    LocalTensor<float> scaledMaskedX;
    LocalTensor<uint8_t> sharedBuffer;

    TBuf<TPosition::VECCALC> bufX;
    TBuf<TPosition::VECCALC> bufShare;

    ScaledMaskedSoftmaxV2TilingData tilingData;

    float scale;
    uint64_t blockIdx;
    uint64_t xBatch;
    uint64_t xChannel;
    uint64_t xHeight;
    uint64_t xWidth;

    uint64_t linePerBatch;
    uint64_t coreOffset;
    uint64_t startLineOffset;
    uint64_t linesPerIter;
    uint64_t linesLastIter;
    uint64_t elePerIter;

    uint64_t maskBatchOffset;
    uint64_t maskChannelOffset;
    uint64_t gmOffsetPerIdx;
};
} // namespace AscendC

#endif // SCALED_MASKED_SOFTMAX_V2_KERNEL_H