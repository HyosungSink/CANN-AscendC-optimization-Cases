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
 * \file deformable_conv2d_base.cpp
 * \brief
 */
#include "deformable_conv2d_base.h"

using namespace DeformableConv2dNS;

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::Init(GM_ADDR input, GM_ADDR weight, GM_ADDR offset, GM_ADDR bias,
                                                   GM_ADDR out, GM_ADDR deform_out, GM_ADDR workspace,
                                                   DeformableConv2dTilingData* tilingData)
{
    if constexpr (std::is_same_v<T, float>) {
        blockNum = FLOAT_BLOCK_NUM;
    } else {
        blockNum = HALF_BLOCK_NUM;
    }
    ParseTilingData(tilingData);

    pipe.InitBuffer(xyBuf, IMAGE_DIM * DATA_COUNT * sizeof(float));
    pipe.InitBuffer(offsetBuf, OFFSET_C * DATA_COUNT * sizeof(float));
    pipe.InitBuffer(xyFloorBuf, IMAGE_DIM * DATA_COUNT * sizeof(float));
    pipe.InitBuffer(xyCeilBuf, IMAGE_DIM * DATA_COUNT * sizeof(float));
    pipe.InitBuffer(weightBuf, WEIGHT_NUM * DATA_COUNT * sizeof(float));
    pipe.InitBuffer(indexBuf, WEIGHT_NUM * DATA_COUNT * sizeof(int32_t));
    pipe.InitBuffer(maskBuf, MASK_NUM * MASK_SIZE * sizeof(uint8_t));
    pipe.InitBuffer(inputBuf, INPUT_DOUBLE * WEIGHT_NUM * MAX_INPUT_SIZE * sizeof(float));
    pipe.InitBuffer(bilinearBuf, MAX_INPUT_SIZE * sizeof(float));
    pipe.InitBuffer(srcOffsetBuf, MAX_INPUT_SIZE * sizeof(int32_t));
    pipe.InitBuffer(gatherBuf, MAX_INPUT_SIZE * sizeof(T));

    inputGm.SetGlobalBuffer((__gm__ T*)input);
    weightGm.SetGlobalBuffer((__gm__ T*)weight);
    offsetGm.SetGlobalBuffer((__gm__ T*)offset);
    outGm.SetGlobalBuffer((__gm__ T*)out);
    deformOutGm.SetGlobalBuffer((__gm__ T*)deform_out);
    convInputGm.SetGlobalBuffer((__gm__ T*)workspace);
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::CalculateStandard(SlideRange range)
{
    int64_t strideOffsetH = range.ohIdx * strideH - padTop;
    int64_t strideOffsetW = range.owStart * strideW - padLeft;

    int64_t idx = 0;
    int64_t idy = range.dataCount;
    for (int64_t ow = range.owStart; ow < range.owEnd; ow++) {
        for (int64_t g = range.groupStart; g < range.groupEnd; g++) {
            for (int64_t kh = 0; kh < kH; kh++) {
                for (int64_t kw = 0; kw < kW; kw++) {
                    xyTensor.SetValue(idx, static_cast<float>(strideOffsetW + kw * dilationW));
                    xyTensor.SetValue(idy, static_cast<float>(strideOffsetH + kh * dilationH));
                    idx++;
                    idy++;
                }
            }
        }
        strideOffsetW += strideW;
    }
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::AdjustStandard(SlideRange range)
{
    Adds(xyTensor, xyTensor, static_cast<float>((range.owStart - range.lastStartW) * strideW), range.dataCount);
    Adds(xyTensor[range.dataCount], xyTensor[range.dataCount],
         static_cast<float>((range.ohIdx - range.lastStartH) * strideH), range.dataCount);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::CopyInOffset(SlideRange range)
{
    int64_t offsetIndex =
        (range.nIdx * outSize + range.ohIdx * outW + range.owStart) * OFFSET_C * deformableGroups * kSize;
    int64_t xOffset = offsetIndex + range.groupStart * kSize;
    int64_t yOffset = xOffset + deformableGroups * kSize;
    int64_t maskOffset = yOffset + deformableGroups * kSize;
    int64_t inOffsetY = DATA_COUNT;
    int64_t inOffsetM = DATA_COUNT * IMAGE_DIM;
    if (range.groupKernelSize % blockNum == 0) {
        inOffsetY = range.dataCount;
        inOffsetM = range.xyDataCount;
    }

    uint32_t srcBlockLen = range.groupKernelSize * sizeof(T);
    uint32_t srcStride = OFFSET_C * deformableGroups * kSize * sizeof(T) - srcBlockLen;
    DataCopyExtParams copyInParams{static_cast<uint16_t>(range.owActualLen), srcBlockLen, srcStride, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(offsetTensor, offsetGm[xOffset], copyInParams, padParams);
    DataCopyPad(offsetTensor[inOffsetY], offsetGm[yOffset], copyInParams, padParams);
    DataCopyPad(offsetTensor[inOffsetM], offsetGm[maskOffset], copyInParams, padParams);

    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    if (range.groupKernelSize % blockNum != 0) {
        uint16_t src0RepeatStride = CeilA2B(range.groupKernelSize, blockNum);
        GatherMaskParams maskParams{1, static_cast<uint16_t>(range.owActualLen), src0RepeatStride, 0};
        uint64_t rsvdCnt = 0;
        GatherMask(offsetTensor, offsetTensor, ALL_PATTERN, true, range.groupKernelSize, maskParams, rsvdCnt);
        GatherMask(offsetTensor[range.dataCount], offsetTensor[inOffsetY], ALL_PATTERN, true, range.groupKernelSize,
                   maskParams, rsvdCnt);
        GatherMask(offsetTensor[range.xyDataCount], offsetTensor[inOffsetM], ALL_PATTERN, true, range.groupKernelSize,
                   maskParams, rsvdCnt);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::CalculateWeight(SlideRange range)
{
    if constexpr (std::is_same_v<T, float>) {
        Add(xyOffsetTensor, xyTensor, offsetTensor, range.xyDataCount);
        PipeBarrier<PIPE_V>();
    } else {
        Cast(xyOffsetTensor, offsetTensor, RoundMode::CAST_NONE, range.xyDataCount + range.dataCount);
        PipeBarrier<PIPE_V>();
        Add(xyOffsetTensor, xyTensor, xyOffsetTensor, range.xyDataCount);
        PipeBarrier<PIPE_V>();
    }
    Cast(xyIntFloorTensor, xyOffsetTensor, RoundMode::CAST_FLOOR, range.xyDataCount);
    PipeBarrier<PIPE_V>();
    Adds(xyIntCeilTensor, xyIntFloorTensor, static_cast<int32_t>(1), range.xyDataCount);
    PipeBarrier<PIPE_V>();
    Muls(ltIndexTensor, xyIntFloorTensor[range.dataCount], intInW, range.dataCount);  // y0*inW
    Muls(rtIndexTensor, xyIntFloorTensor[range.dataCount], intInW, range.dataCount);  // y0*inW
    Muls(lbIndexTensor, xyIntCeilTensor[range.dataCount], intInW, range.dataCount);   // y1*inW
    Muls(rbIndexTensor, xyIntCeilTensor[range.dataCount], intInW, range.dataCount);   // y1*inW
    PipeBarrier<PIPE_V>();
    Add(ltIndexTensor, ltIndexTensor, xyIntFloorTensor, range.dataCount);  // y0*inW + x0
    Add(rtIndexTensor, rtIndexTensor, xyIntCeilTensor, range.dataCount);   // y0*inW + x1
    Add(lbIndexTensor, lbIndexTensor, xyIntFloorTensor, range.dataCount);  // y1*inW + x0
    Add(rbIndexTensor, rbIndexTensor, xyIntCeilTensor, range.dataCount);   // y1*inW + x1
    PipeBarrier<PIPE_V>();

    Cast(xyFloorTensor, xyIntFloorTensor, RoundMode::CAST_NONE, range.xyDataCount);
    Cast(xyCeilTensor, xyIntCeilTensor, RoundMode::CAST_NONE, range.xyDataCount);
    PipeBarrier<PIPE_V>();
    CompareScalar(ltMaskLocal0, xyFloorTensor, 0.0f, CMPMODE::GE, range.dataCount);                       // 0 <= x0
    CompareScalar(ltMaskLocal1, xyFloorTensor, floatInW, CMPMODE::LT, range.dataCount);                   // x0 < inW
    CompareScalar(lbMaskLocal0, xyFloorTensor[range.dataCount], 0.0f, CMPMODE::GE, range.dataCount);      // 0 <= y0
    CompareScalar(lbMaskLocal1, xyFloorTensor[range.dataCount], floatInH, CMPMODE::LT, range.dataCount);  // y0 < inH
    CompareScalar(rtMaskLocal0, xyCeilTensor, 0.0f, CMPMODE::GE, range.dataCount);                        // 0 <= x1
    CompareScalar(rtMaskLocal1, xyCeilTensor, floatInW, CMPMODE::LT, range.dataCount);                    // x1 < inW
    CompareScalar(rbMaskLocal0, xyCeilTensor[range.dataCount], 0.0f, CMPMODE::GE, range.dataCount);       // 0 <= y1
    CompareScalar(rbMaskLocal1, xyCeilTensor[range.dataCount], floatInH, CMPMODE::LT, range.dataCount);   // y1 < inH
    PipeBarrier<PIPE_V>();
    And(ltMaskLocal0Tmp, ltMaskLocal0Tmp, ltMaskLocal1Tmp, AND_SIZE);  // 0 <= x0 < inW
    And(lbMaskLocal0Tmp, lbMaskLocal0Tmp, lbMaskLocal1Tmp, AND_SIZE);  // 0 <= y0 < inH
    And(rtMaskLocal0Tmp, rtMaskLocal0Tmp, rtMaskLocal1Tmp, AND_SIZE);  // 0 <= x1 < inW
    And(rbMaskLocal0Tmp, rbMaskLocal0Tmp, rbMaskLocal1Tmp, AND_SIZE);  // 0 <= y1 < inH
    PipeBarrier<PIPE_V>();
    And(ltMaskLocal1Tmp, ltMaskLocal0Tmp, lbMaskLocal0Tmp, AND_SIZE);  // 0 <= x0 < inW && 0 <= y0 < inH
    And(lbMaskLocal1Tmp, ltMaskLocal0Tmp, rbMaskLocal0Tmp, AND_SIZE);  // 0 <= x0 < inW && 0 <= y1 < inH
    And(rtMaskLocal1Tmp, rtMaskLocal0Tmp, lbMaskLocal0Tmp, AND_SIZE);  // 0 <= x1 < inW && 0 <= y0 < inH
    And(rbMaskLocal1Tmp, rtMaskLocal0Tmp, rbMaskLocal0Tmp, AND_SIZE);  // 0 <= x1 < inW && 0 <= y1 < inH
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::ProcessZero(SlideRange range)
{
    Sub(xyFloorTensor, xyOffsetTensor, xyFloorTensor, range.xyDataCount);
    Sub(xyCeilTensor, xyCeilTensor, xyOffsetTensor, range.xyDataCount);
    PipeBarrier<PIPE_V>();
    Mul(ltWeightTensor, xyCeilTensor, xyCeilTensor[range.dataCount], range.dataCount);
    Mul(lbWeightTensor, xyCeilTensor, xyFloorTensor[range.dataCount], range.dataCount);
    Mul(rtWeightTensor, xyFloorTensor, xyCeilTensor[range.dataCount], range.dataCount);
    Mul(rbWeightTensor, xyFloorTensor, xyFloorTensor[range.dataCount], range.dataCount);
    PipeBarrier<PIPE_V>();
    Select(ltIndexFloat, ltMaskLocal1, ltIndexFloat, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, range.dataCount);
    Select(lbIndexFloat, lbMaskLocal1, lbIndexFloat, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, range.dataCount);
    Select(rtIndexFloat, rtMaskLocal1, rtIndexFloat, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, range.dataCount);
    Select(rbIndexFloat, rbMaskLocal1, rbIndexFloat, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, range.dataCount);
    Select(ltWeightTensor, ltMaskLocal1, ltWeightTensor, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, range.dataCount);
    Select(lbWeightTensor, lbMaskLocal1, lbWeightTensor, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, range.dataCount);
    Select(rtWeightTensor, rtMaskLocal1, rtWeightTensor, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, range.dataCount);
    Select(rbWeightTensor, rbMaskLocal1, rbWeightTensor, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, range.dataCount);
    PipeBarrier<PIPE_V>();
    Mul(ltWeightTensor, ltWeightTensor, xyOffsetTensor[range.xyDataCount], range.dataCount);
    Mul(lbWeightTensor, lbWeightTensor, xyOffsetTensor[range.xyDataCount], range.dataCount);
    Mul(rtWeightTensor, rtWeightTensor, xyOffsetTensor[range.xyDataCount], range.dataCount);
    Mul(rbWeightTensor, rbWeightTensor, xyOffsetTensor[range.xyDataCount], range.dataCount);
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::BilinearInterpSmallC(SlideRange range)
{
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    int64_t baseOffset = range.nIdx * inH * inW * inC;
    int64_t idx = 0;
    int64_t inputOffset = 0;
    for (int64_t ow = range.owStart; ow < range.owEnd; ow++) {
        for (int64_t kh = 0; kh < kH; kh++) {
            for (int64_t kw = 0; kw < kW; kw++) {
                CopyInSmallC(inputOffset, baseOffset, idx);
                idx++;
                inputOffset += totalC;
            }
        }
    }
    CalculateBilinearSmallC(range);

    // deform(n, outH * kH, outW * kW, inC), conv(n, outH, groups, outW, kH * kW, inC_conv_group)
    int64_t deformOffset0 = (range.nIdx * outH + range.ohIdx) * outW * kSize * inC;
    int64_t convOffset0 = deformOffset0 + range.owStart * kSize * inC_conv_group;
    deformOffset0 += range.owStart * kW * inC;
    int64_t outOffset = 0;
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    for (int64_t ow = range.owStart; ow < range.owEnd; ow++) {
        DeformOutSmallC(deformOffset0, outOffset);
        outOffset += kSize * inC;
        deformOffset0 += kW * inC;
    }

    uint32_t srcBlockLen = range.owActualLen * kSize * inC_conv_group * sizeof(T);
    int64_t ceilConvC = CeilA2B(range.owActualLen * kSize * inC_conv_group, blockNum) * blockNum;
    if (groups > 1) {
        for (int groupIdx = 0; groupIdx < groups; groupIdx++) {
            Gather(gatherTensor[groupIdx * ceilConvC], deformOutTensor, srcOffsetLocal,
                   groupIdx * inC_conv_group * sizeof(T), range.owActualLen * kSize * inC_conv_group);
        }
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        uint32_t dstStride = outW * kSize * inC_conv_group * sizeof(T) - srcBlockLen;
        DataCopyExtParams copyOutParams{static_cast<uint16_t>(groups), srcBlockLen, 0, dstStride, 0};
        DataCopyPad(convInputGm[convOffset0], gatherTensor, copyOutParams);
    } else {
        DataCopyExtParams copyOutParams{1, srcBlockLen, 0, 0, 0};
        DataCopyPad(convInputGm[convOffset0], deformOutTensor, copyOutParams);
    }
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::CopyInSmallC(int64_t inputOffset, int64_t gmOffset, int64_t idx)
{
    int32_t ltIndex = ltIndexTensor.GetValue(idx);
    int32_t rtIndex = rtIndexTensor.GetValue(idx);
    int32_t lbIndex = lbIndexTensor.GetValue(idx);
    int32_t rbIndex = rbIndexTensor.GetValue(idx);
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if (ltIndex >= 0 && rtIndex >= 0 && lbIndex >= 0 && rbIndex >= 0) {
        // 一次性拷入
        uint32_t srcStride = inW * lenC - doubleLenC;
        DataCopyExtParams copyInParams{2, doubleLenC, srcStride, 0, 0};
        DataCopyPad(inputTensor[inputOffset], inputGm[gmOffset + ltIndex * inC], copyInParams, padParams);
        return;
    }

    // 分两次拷入
    if (ltIndex >= 0 || rtIndex >= 0) {
        if (ltIndex >= 0) {
            DataCopyExtParams copyInParams{1, (rtIndex >= 0) ? doubleLenC : lenC, 0, 0, 0};
            DataCopyPad(inputTensor[inputOffset], inputGm[gmOffset + ltIndex * inC], copyInParams, padParams);
        } else {
            DataCopyExtParams copyInParams{1, lenC, 0, 0, 0};
            DataCopyPad(inputTensor[inputOffset + inC], inputGm[gmOffset + rtIndex * inC], copyInParams, padParams);
        }
    }
    if (lbIndex >= 0 || rbIndex >= 0) {
        if (lbIndex >= 0) {
            DataCopyExtParams copyInParams{1, (rbIndex >= 0) ? doubleLenC : lenC, 0, 0, 0};
            DataCopyPad(inputTensor[inputOffset + lbOffset], inputGm[gmOffset + lbIndex * inC], copyInParams,
                        padParams);
        } else {
            DataCopyExtParams copyInParams{1, lenC, 0, 0, 0};
            DataCopyPad(inputTensor[inputOffset + rbOffset], inputGm[gmOffset + rbIndex * inC], copyInParams,
                        padParams);
        }
    }
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::CalculateBilinearSmallC(SlideRange range)
{
    uint32_t calCount = range.owActualLen * kSize;
    uint32_t batchCount = calCount * inC;
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    uint16_t src0RepeatStride = totalC / blockNum;
    GatherMaskParams maskParams{1, static_cast<uint16_t>(calCount), src0RepeatStride, 0};
    uint64_t rsvdCnt = 0;
    GatherMask(ltInputTensor, inputTensor, ALL_PATTERN, true, inC, maskParams, rsvdCnt);
    GatherMask(rtInputTensor, inputTensor[inC], ALL_PATTERN, true, inC, maskParams, rsvdCnt);
    GatherMask(lbInputTensor, inputTensor[lbOffset], ALL_PATTERN, true, inC, maskParams, rsvdCnt);
    GatherMask(rbInputTensor, inputTensor[rbOffset], ALL_PATTERN, true, inC, maskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();

    uint32_t srcShape[2] = {calCount, 1};
    uint32_t dstShape[2] = {calCount, static_cast<uint32_t>(inC)};
    PipeBarrier<PIPE_V>();
    BroadCast<float, 2, 1>(ltWeightTmp, ltWeightTensor, dstShape, srcShape);
    BroadCast<float, 2, 1>(rtWeightTmp, rtWeightTensor, dstShape, srcShape);
    BroadCast<float, 2, 1>(lbWeightTmp, lbWeightTensor, dstShape, srcShape);
    BroadCast<float, 2, 1>(rbWeightTmp, rbWeightTensor, dstShape, srcShape);
    if constexpr (!std::is_same_v<T, float>) {
        Cast(ltInputFloat, ltInputTensor, RoundMode::CAST_NONE, batchCount);
        Cast(rtInputFloat, rtInputTensor, RoundMode::CAST_NONE, batchCount);
        Cast(lbInputFloat, lbInputTensor, RoundMode::CAST_NONE, batchCount);
        Cast(rbInputFloat, rbInputTensor, RoundMode::CAST_NONE, batchCount);
    }
    PipeBarrier<PIPE_V>();
    Mul(ltInputFloat, ltInputFloat, ltWeightTmp, batchCount);
    Mul(rtInputFloat, rtInputFloat, rtWeightTmp, batchCount);
    Mul(lbInputFloat, lbInputFloat, lbWeightTmp, batchCount);
    Mul(rbInputFloat, rbInputFloat, rbWeightTmp, batchCount);
    PipeBarrier<PIPE_V>();
    Duplicate(inputTensor, static_cast<T>(0), WEIGHT_NUM * MAX_INPUT_SIZE);
    Add(ltInputFloat, ltInputFloat, rtInputFloat, batchCount);
    Add(lbInputFloat, lbInputFloat, rbInputFloat, batchCount);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    Add(bilinearTensor, ltInputFloat, lbInputFloat, batchCount);
    if constexpr (!std::is_same_v<T, float>) {
        PipeBarrier<PIPE_V>();
        Cast(deformOutTensor, bilinearTensor, RoundMode::CAST_RINT, batchCount);
    }
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::DeformOutSmallC(int64_t deformOutOffset, int64_t outOffset)
{
    uint32_t srcBlockLen = kW * lenC;
    uint32_t dstStride = outW * kW * lenC - srcBlockLen;
    DataCopyExtParams copyOutParams{static_cast<uint16_t>(kH), srcBlockLen, 0, dstStride, 0};
    DataCopyPad(deformOutGm[deformOutOffset], deformOutTensor[outOffset], copyOutParams);
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::BilinearInterp(SlideRange range)
{
    int64_t baseOffset = range.nIdx * inH * inW * inC;
    // deform(n, outH * kH, outW * kW, inC), conv(n, outH, groups, outW, kH * kW, inC_conv_group)
    int64_t deformOffset0 = (range.nIdx * outH + range.ohIdx) * outW * kSize * inC;
    int64_t convOffset0 = deformOffset0 + range.owStart * kSize * inC_conv_group;
    deformOffset0 += range.owStart * kW * inC + range.groupStart * inC_per_group;

    int64_t idx = 0;
    for (int64_t ow = range.owStart; ow < range.owEnd; ow++) {
        int64_t deformOffset1 = deformOffset0;
        int64_t convOffset1 = convOffset0;
        for (int64_t g = range.groupStart; g < range.groupEnd; g++) {
            int64_t deformOffset2 = deformOffset1;
            int64_t convOffset2 = convOffset1;
            int64_t gmOffset = baseOffset + g * inC_per_group;
            for (int64_t kh = 0; kh < kH; kh++) {
                for (int64_t kw = 0; kw < kW; kw++) {
                    ComputeEntireC(gmOffset, idx, deformOffset2 + kw * inC, convOffset2 + kw * inC_conv_group);
                    idx++;
                }
                deformOffset2 += outW * kW * inC;
                convOffset2 += kW * inC_conv_group;
            }
            deformOffset1 += inC_per_group;
        }
        deformOffset0 += kW * inC;
        convOffset0 += kSize * inC_conv_group;
    }
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::ComputeEntireC(int64_t gmOffset, int64_t idx, int64_t deformOutOffset,
                                                             int64_t convInputOffset)
{
    for (int64_t c = 0; c < cLoop; c++) {
        int64_t cStart = c * MAX_INPUT_SIZE;
        int64_t cLen = Min(MAX_INPUT_SIZE, inC_per_group - cStart);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        CopyInAndCalculate(gmOffset, idx, cLen);
        if constexpr (!std::is_same_v<T, float>) {
            Cast(deformOutTensor, bilinearTensor, RoundMode::CAST_RINT, cLen);
            PipeBarrier<PIPE_V>();
        }
        CopyOut(deformOutOffset, convInputOffset, cLen);

        deformOutOffset += MAX_INPUT_SIZE;
        gmOffset += MAX_INPUT_SIZE;
    }
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::CopyInAndCalculate(int64_t gmOffset, int64_t idx, int64_t cLen)
{
    int32_t ltIndex = ltIndexTensor.GetValue(idx);
    int32_t rtIndex = rtIndexTensor.GetValue(idx);
    int32_t lbIndex = lbIndexTensor.GetValue(idx);
    int32_t rbIndex = rbIndexTensor.GetValue(idx);

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(cLen * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    // 分四次拷入
    if (ltIndex >= 0) {
        DataCopyPad(ltInputTensor, inputGm[gmOffset + ltIndex * inC], copyInParams, padParams);
    }
    if (rtIndex >= 0) {
        DataCopyPad(rtInputTensor, inputGm[gmOffset + rtIndex * inC], copyInParams, padParams);
    }
    if (lbIndex >= 0) {
        DataCopyPad(lbInputTensor, inputGm[gmOffset + lbIndex * inC], copyInParams, padParams);
    }
    if (rbIndex >= 0) {
        DataCopyPad(rbInputTensor, inputGm[gmOffset + rbIndex * inC], copyInParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    if constexpr (!std::is_same_v<T, float>) {
        Cast(ltInputFloat, ltInputTensor, RoundMode::CAST_NONE, cLen);
        Cast(rtInputFloat, rtInputTensor, RoundMode::CAST_NONE, cLen);
        Cast(lbInputFloat, lbInputTensor, RoundMode::CAST_NONE, cLen);
        Cast(rbInputFloat, rbInputTensor, RoundMode::CAST_NONE, cLen);
    }
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    Duplicate(bilinearTensor, static_cast<float>(0), cLen);
    if (ltIndex >= 0) {
        PipeBarrier<PIPE_V>();
        Axpy(bilinearTensor, ltInputFloat, ltWeightTensor.GetValue(idx), cLen);
    }
    if (rtIndex >= 0) {
        PipeBarrier<PIPE_V>();
        Axpy(bilinearTensor, rtInputFloat, rtWeightTensor.GetValue(idx), cLen);
    }
    if (lbIndex >= 0) {
        PipeBarrier<PIPE_V>();
        Axpy(bilinearTensor, lbInputFloat, lbWeightTensor.GetValue(idx), cLen);
    }
    if (rbIndex >= 0) {
        PipeBarrier<PIPE_V>();
        Axpy(bilinearTensor, rbInputFloat, rbWeightTensor.GetValue(idx), cLen);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::CopyOut(int64_t deformOutOffset, int64_t convInputOffset, int64_t cLen)
{
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(cLen * sizeof(T)), 0, 0, 0};
    DataCopyPad(deformOutGm[deformOutOffset], deformOutTensor, copyOutParams);

    int64_t cStart = deformOutOffset % inC;
    int64_t groupStart = cStart / inC_conv_group;
    int64_t groupOffset = cStart % inC_conv_group;
    int64_t outLenC = 0;

    int64_t offset = convInputOffset + groupStart * outW * kSize * inC_conv_group + groupOffset;
    int64_t outLen = Min(cLen, inC_conv_group - groupOffset);
    DataCopyExtParams convCopyParams{1, static_cast<uint32_t>(outLen * sizeof(T)), 0, 0, 0};
    DataCopyPad(convInputGm[offset], deformOutTensor, convCopyParams);
    outLenC += outLen;
    groupStart++;
    for (int64_t groupIdx = groupStart; outLenC < cLen; groupIdx++) {
        int64_t offset = convInputOffset + groupIdx * outW * kSize * inC_conv_group;
        int64_t outLen = Min(cLen - outLenC, inC_conv_group);
        DataCopyExtParams convCopyParams{1, static_cast<uint32_t>(outLen * sizeof(T)), 0, 0, 0};
        if (outLenC % blockNum == 0) {
            DataCopyPad(convInputGm[offset], deformOutTensor[outLenC], convCopyParams);
        } else {
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            Gather(gatherTensor, deformOutTensor, srcOffsetLocal, outLenC * sizeof(T), outLen);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            DataCopyPad(convInputGm[offset], gatherTensor, convCopyParams);
        }
        outLenC += outLen;
    }
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::Conv2d(SlideRange range)
{
    int64_t aOffset = 0;
    int64_t bOffset = (range.nIdx * outH + range.ohIdx) * groups * outW * kSize * inC_conv_group;
    int64_t cOffset = (range.nIdx * outH + range.ohIdx) * outC * outW;
    for (int64_t groupIdx = 0; groupIdx < groups; groupIdx++) {
        matmulObj.SetTensorA(weightGm[aOffset], false);
        matmulObj.SetTensorB(convInputGm[bOffset], true);
        matmulObj.IterateAll(outGm[cOffset], 0);
        aOffset += outC_conv_group * kSize * inC_conv_group;
        bOffset += outW * kSize * inC_conv_group;
        cOffset += outC_conv_group * outW;
    }
}
