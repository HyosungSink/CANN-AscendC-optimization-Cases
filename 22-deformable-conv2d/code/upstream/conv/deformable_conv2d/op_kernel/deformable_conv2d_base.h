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
 * \file deformable_conv2d_base.h
 * \brief
 */
#ifndef DEFORMABLE_CONV2D_BASE_H
#define DEFORMABLE_CONV2D_BASE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace DeformableConv2dNS
{
using namespace AscendC;

constexpr MatmulConfig MDL_CFG = GetNormalConfig();
constexpr int32_t IMAGE_DIM = 2;
constexpr int32_t WEIGHT_NUM = 4;
constexpr int32_t OFFSET_C = 3;
constexpr int32_t DATA_COUNT = 2048;
constexpr int32_t MAX_INPUT_SIZE = 1024;
constexpr int32_t BRCB_STRIDE = 8;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t HALF_BLOCK_NUM = 16;
constexpr int32_t FLOAT_BLOCK_NUM = 8;
constexpr int32_t MASK_NUM = 8;
constexpr int32_t MASK_SIZE = 256;
constexpr int32_t AND_SIZE = 128;
constexpr int32_t DATA_ALIGN = 64;
constexpr int32_t INPUT_DOUBLE = 2;
constexpr uint8_t ALL_PATTERN = 7;

struct SlideRange {
    int64_t nIdx;
    int64_t ohIdx;
    int64_t owStart;
    int64_t owEnd;
    int64_t owActualLen;

    int64_t groupStart;
    int64_t groupEnd;
    int64_t groupActualLen;
    int64_t groupKernelSize;
    int64_t dataCount;
    int64_t xyDataCount;

    int64_t lastStartW;
    int64_t lastStartH;
    int64_t lastLenW;
    int64_t lastLenGroup;
};

template <typename T>
class DeformableConv2dND
{
public:
    TPipe pipe;
    SlideRange range;
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG>
        matmulObj;

    __aicore__ inline DeformableConv2dND(){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight, GM_ADDR offset, GM_ADDR bias, GM_ADDR out,
                                GM_ADDR deform_out, GM_ADDR workspace, DeformableConv2dTilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        return (b != 0) ? (a + b - 1) / b : a;
    };

    template <typename T1, typename T2>
    __aicore__ inline T1 Min(T1 a, T2 b)
    {
        return (a < b) ? a : b;
    };

    __aicore__ inline void InitTensor();
    __aicore__ inline void InitOffset();
    __aicore__ inline void InitEventId();
    __aicore__ inline void ReleaseEventId();
    __aicore__ inline void ParseTilingData(DeformableConv2dTilingData* tilingData);
    __aicore__ inline void CopyInOffset(SlideRange range);
    __aicore__ inline void CalculateStandard(SlideRange range);
    __aicore__ inline void AdjustStandard(SlideRange range);
    __aicore__ inline void CalculateWeight(SlideRange range);
    __aicore__ inline void ProcessZero(SlideRange range);
    __aicore__ inline void BilinearInterp(SlideRange range);
    __aicore__ inline void ComputeEntireC(int64_t gmOffset, int64_t idx, int64_t deformOutOffset,
                                          int64_t convInputOffset);
    __aicore__ inline void CopyInAndCalculate(int64_t gmOffset, int64_t idx, int64_t cLen);
    __aicore__ inline void CopyOut(int64_t deformOutOffset, int64_t convInputOffset, int64_t cLen);
    __aicore__ inline void Conv2d(SlideRange range);

    __aicore__ inline void BilinearInterpSmallC(SlideRange range);
    __aicore__ inline void CopyInSmallC(int64_t inputOffset, int64_t gmOffset, int64_t idx);
    __aicore__ inline void CalculateBilinearSmallC(SlideRange range);
    __aicore__ inline void DeformOutSmallC(int64_t deformOutOffset, int64_t outOffset);

private:
    TBuf<TPosition::VECCALC> xyBuf;       // 2 * 8K
    TBuf<TPosition::VECCALC> offsetBuf;   // 3 * 8K
    TBuf<TPosition::VECCALC> xyFloorBuf;  // 2 * 8K
    TBuf<TPosition::VECCALC> xyCeilBuf;   // 2 * 8K
    TBuf<TPosition::VECCALC> weightBuf;   // 4 * 8K
    TBuf<TPosition::VECCALC> indexBuf;    // 4 * 8K
    TBuf<TPosition::VECCALC> maskBuf;     // 2K
    // 138K
    TBuf<TPosition::VECCALC> inputBuf;      // 2 * 4 * 4K
    TBuf<TPosition::VECCALC> bilinearBuf;   // 4K
    TBuf<TPosition::VECCALC> srcOffsetBuf;  // 4K
    TBuf<TPosition::VECCALC> gatherBuf;     // 4K
    // 44K

    GlobalTensor<T> inputGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> offsetGm;
    GlobalTensor<T> outGm;
    GlobalTensor<T> deformOutGm;
    GlobalTensor<T> convInputGm;

    LocalTensor<float> xyTensor;
    LocalTensor<T> offsetTensor;
    LocalTensor<float> xyOffsetTensor;
    LocalTensor<int32_t> xyIntFloorTensor;
    LocalTensor<float> xyFloorTensor;
    LocalTensor<int32_t> xyIntCeilTensor;
    LocalTensor<float> xyCeilTensor;
    LocalTensor<float> ltWeightTensor;
    LocalTensor<float> lbWeightTensor;
    LocalTensor<float> rtWeightTensor;
    LocalTensor<float> rbWeightTensor;

    LocalTensor<float> ltIndexFloat;
    LocalTensor<float> rtIndexFloat;
    LocalTensor<float> lbIndexFloat;
    LocalTensor<float> rbIndexFloat;
    LocalTensor<int32_t> ltIndexTensor;
    LocalTensor<int32_t> rtIndexTensor;
    LocalTensor<int32_t> lbIndexTensor;
    LocalTensor<int32_t> rbIndexTensor;

    LocalTensor<uint8_t> ltMaskLocal0;
    LocalTensor<uint8_t> lbMaskLocal0;
    LocalTensor<uint8_t> rtMaskLocal0;
    LocalTensor<uint8_t> rbMaskLocal0;
    LocalTensor<uint8_t> ltMaskLocal1;
    LocalTensor<uint8_t> lbMaskLocal1;
    LocalTensor<uint8_t> rtMaskLocal1;
    LocalTensor<uint8_t> rbMaskLocal1;
    LocalTensor<uint16_t> ltMaskLocal0Tmp;
    LocalTensor<uint16_t> lbMaskLocal0Tmp;
    LocalTensor<uint16_t> rtMaskLocal0Tmp;
    LocalTensor<uint16_t> rbMaskLocal0Tmp;
    LocalTensor<uint16_t> ltMaskLocal1Tmp;
    LocalTensor<uint16_t> lbMaskLocal1Tmp;
    LocalTensor<uint16_t> rtMaskLocal1Tmp;
    LocalTensor<uint16_t> rbMaskLocal1Tmp;

    LocalTensor<T> ltInputTensor;
    LocalTensor<T> rtInputTensor;
    LocalTensor<T> lbInputTensor;
    LocalTensor<T> rbInputTensor;
    LocalTensor<T> inputTensor;
    LocalTensor<float> ltInputFloat;
    LocalTensor<float> rtInputFloat;
    LocalTensor<float> lbInputFloat;
    LocalTensor<float> rbInputFloat;
    LocalTensor<float> ltWeightTmp;
    LocalTensor<float> rtWeightTmp;
    LocalTensor<float> lbWeightTmp;
    LocalTensor<float> rbWeightTmp;

    LocalTensor<float> bilinearTensor;
    LocalTensor<T> deformOutTensor;
    LocalTensor<int32_t> srcOffsetTensor;
    LocalTensor<uint32_t> srcOffsetLocal;
    LocalTensor<T> gatherTensor;

    event_t eventIdVToMte2;
    event_t eventIdMte2ToV;
    event_t eventIdVToMte3;
    event_t eventIdVToS;
    event_t eventIdSToV;
    event_t eventIdMte3ToV;

    int32_t blockNum = 0;

    int64_t n = 0;
    int64_t inC = 0;
    int64_t inH = 0;
    int64_t inW = 0;
    int64_t outC = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t outSize = 0;
    int64_t kH = 0;
    int64_t kW = 0;
    int64_t kSize = 0;

    int64_t padTop = 0;
    int64_t padLeft = 0;
    int64_t strideH = 0;
    int64_t strideW = 0;
    int64_t dilationH = 0;
    int64_t dilationW = 0;
    int64_t deformableGroups = 0;
    int64_t groups = 0;

    int64_t slideSizeW = 0;
    int64_t groupLen = 0;

    int64_t slideStart = 0;
    int64_t slideEnd = 0;
    int64_t inC_per_group = 0;
    int64_t cLoop = 0;
    int64_t inC_conv_group = 0;
    int64_t outC_conv_group = 0;
    int64_t calW = 0;

    int32_t intInH = 0;
    int32_t intInW = 0;
    float floatInH = 0.0f;
    float floatInW = 0.0f;

    uint32_t lenC = 0;
    uint32_t doubleLenC = 0;
    uint32_t lbOffset = 0;
    uint32_t rbOffset = 0;
    uint32_t totalC = 0;
};

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::Process()
{
    if (slideStart >= slideEnd) {
        return;
    }

    InitTensor();
    InitOffset();
    InitEventId();
    for (int64_t slideIdx = slideStart; slideIdx < slideEnd; slideIdx++) {
        range.nIdx = slideIdx / outH;
        range.ohIdx = slideIdx % outH;
        for (int64_t ow = 0; ow < outW; ow += slideSizeW) {
            range.owStart = ow;
            range.owEnd = Min(range.owStart + slideSizeW, outW);
            range.owActualLen = range.owEnd - range.owStart;
            for (int64_t g = 0; g < deformableGroups; g += groupLen) {
                range.groupStart = g;
                range.groupEnd = Min(range.groupStart + groupLen, deformableGroups);
                range.groupActualLen = range.groupEnd - range.groupStart;
                range.groupKernelSize = range.groupActualLen * kSize;
                range.dataCount = CeilA2B(range.owActualLen * range.groupKernelSize, DATA_ALIGN) * DATA_ALIGN;
                range.xyDataCount = IMAGE_DIM * range.dataCount;
                // 拷入x-y偏移
                CopyInOffset(range);
                // 获取标准卷积的原图像下标
                if (slideIdx == slideStart && range.owStart == 0 && range.groupStart == 0) {
                    CalculateStandard(range);
                } else if (range.owActualLen != range.lastLenW || range.groupActualLen != range.lastLenGroup) {
                    CalculateStandard(range);
                } else {
                    AdjustStandard(range);
                }
                // 计算周围四个点的权重
                CalculateWeight(range);
                ProcessZero(range);
                // 双线性插值
                if (calW > 0) {
                    BilinearInterpSmallC(range);
                } else {
                    BilinearInterp(range);
                }
                range.lastStartH = range.ohIdx;
                range.lastStartW = range.owStart;
                range.lastLenW = range.owActualLen;
                range.lastLenGroup = range.groupActualLen;
            }
        }
        Conv2d(range);
    }
    ReleaseEventId();
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::ParseTilingData(DeformableConv2dTilingData* tilingData)
{
    n = tilingData->n;
    inC = tilingData->inC;
    inH = tilingData->inH;
    inW = tilingData->inW;
    intInH = static_cast<int32_t>(inH);
    intInW = static_cast<int32_t>(inW);
    floatInH = static_cast<float>(inH);
    floatInW = static_cast<float>(inW);
    outC = tilingData->outC;
    outH = tilingData->outH;
    outW = tilingData->outW;
    outSize = outH * outW;
    kH = tilingData->kH;
    kW = tilingData->kW;
    kSize = kH * kW;

    padTop = tilingData->padTop;
    padLeft = tilingData->padLeft;
    strideH = tilingData->strideH;
    strideW = tilingData->strideW;
    dilationH = tilingData->dilationH;
    dilationW = tilingData->dilationW;
    deformableGroups = tilingData->deformableGroups;
    groups = tilingData->groups;
    slideSizeW = tilingData->slideSizeW;
    groupLen = tilingData->groupLen;

    inC_per_group = inC / deformableGroups;
    cLoop = (inC_per_group > MAX_INPUT_SIZE) ? CeilA2B(inC_per_group, MAX_INPUT_SIZE) : 1;
    inC_conv_group = inC / groups;
    outC_conv_group = outC / groups;
    lenC = inC * sizeof(T);
    doubleLenC = INPUT_DOUBLE * lenC;
    lbOffset = IMAGE_DIM * inC;
    rbOffset = OFFSET_C * inC;
    totalC = WEIGHT_NUM * inC;

    if (deformableGroups == 1 && inC % blockNum == 0 && kSize * inC <= MAX_INPUT_SIZE) {
        calW = Min(MAX_INPUT_SIZE / (kSize * inC), slideSizeW);
        if (CeilA2B(calW * kSize * inC_conv_group, blockNum) * blockNum * groups <= MAX_INPUT_SIZE) {
            slideSizeW = calW;
        } else {
            calW = 0;
        }
    }

    int64_t singleVecNum = tilingData->singleVecNum;
    int64_t tailVecNum = tilingData->tailVecNum;
    int32_t blockIdx = GetBlockIdx();
    slideStart = singleVecNum * blockIdx + (blockIdx < tailVecNum ? blockIdx : tailVecNum);
    slideEnd = slideStart + singleVecNum + (blockIdx < tailVecNum ? 1 : 0);
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::InitTensor()
{
    xyTensor = xyBuf.Get<float>();
    offsetTensor = offsetBuf.Get<T>();
    xyOffsetTensor = offsetBuf.Get<float>();
    xyIntFloorTensor = xyFloorBuf.Get<int32_t>();
    xyFloorTensor = xyFloorBuf.Get<float>();
    xyIntCeilTensor = xyCeilBuf.Get<int32_t>();
    xyCeilTensor = xyCeilBuf.Get<float>();
    ltWeightTensor = weightBuf.Get<float>();
    lbWeightTensor = ltWeightTensor[DATA_COUNT];
    rtWeightTensor = lbWeightTensor[DATA_COUNT];
    rbWeightTensor = rtWeightTensor[DATA_COUNT];

    ltIndexFloat = indexBuf.Get<float>();
    rtIndexFloat = ltIndexFloat[DATA_COUNT];
    lbIndexFloat = rtIndexFloat[DATA_COUNT];
    rbIndexFloat = lbIndexFloat[DATA_COUNT];
    ltIndexTensor = indexBuf.Get<int32_t>();
    rtIndexTensor = ltIndexTensor[DATA_COUNT];
    lbIndexTensor = rtIndexTensor[DATA_COUNT];
    rbIndexTensor = lbIndexTensor[DATA_COUNT];

    ltMaskLocal0 = maskBuf.Get<uint8_t>();
    lbMaskLocal0 = ltMaskLocal0[MASK_SIZE];
    rtMaskLocal0 = lbMaskLocal0[MASK_SIZE];
    rbMaskLocal0 = rtMaskLocal0[MASK_SIZE];
    ltMaskLocal1 = rbMaskLocal0[MASK_SIZE];
    lbMaskLocal1 = ltMaskLocal1[MASK_SIZE];
    rtMaskLocal1 = lbMaskLocal1[MASK_SIZE];
    rbMaskLocal1 = rtMaskLocal1[MASK_SIZE];

    ltInputTensor = inputBuf.Get<T>();
    rtInputTensor = ltInputTensor[MAX_INPUT_SIZE];
    lbInputTensor = rtInputTensor[MAX_INPUT_SIZE];
    rbInputTensor = lbInputTensor[MAX_INPUT_SIZE];
    inputTensor = rbInputTensor[MAX_INPUT_SIZE];
    ltInputFloat = inputBuf.Get<float>();
    rtInputFloat = ltInputFloat[MAX_INPUT_SIZE];
    lbInputFloat = rtInputFloat[MAX_INPUT_SIZE];
    rbInputFloat = lbInputFloat[MAX_INPUT_SIZE];

    bilinearTensor = bilinearBuf.Get<float>();
    deformOutTensor = bilinearBuf.Get<T>();
    srcOffsetTensor = srcOffsetBuf.Get<int32_t>();
    srcOffsetLocal = srcOffsetTensor.ReinterpretCast<uint32_t>();
    gatherTensor = gatherBuf.Get<T>();
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::InitOffset()
{
    ltMaskLocal0Tmp = ltMaskLocal0.ReinterpretCast<uint16_t>();
    lbMaskLocal0Tmp = lbMaskLocal0.ReinterpretCast<uint16_t>();
    rtMaskLocal0Tmp = rtMaskLocal0.ReinterpretCast<uint16_t>();
    rbMaskLocal0Tmp = rbMaskLocal0.ReinterpretCast<uint16_t>();
    ltMaskLocal1Tmp = ltMaskLocal1.ReinterpretCast<uint16_t>();
    lbMaskLocal1Tmp = lbMaskLocal1.ReinterpretCast<uint16_t>();
    rtMaskLocal1Tmp = rtMaskLocal1.ReinterpretCast<uint16_t>();
    rbMaskLocal1Tmp = rbMaskLocal1.ReinterpretCast<uint16_t>();
    ltWeightTmp = rbInputFloat[MAX_INPUT_SIZE];
    rtWeightTmp = ltWeightTmp[MAX_INPUT_SIZE];
    lbWeightTmp = rtWeightTmp[MAX_INPUT_SIZE];
    rbWeightTmp = lbWeightTmp[MAX_INPUT_SIZE];

    if (calW > 0) {
        Duplicate(inputTensor, static_cast<T>(0), WEIGHT_NUM * MAX_INPUT_SIZE);
        ArithProgression(srcOffsetTensor, static_cast<int32_t>(0), static_cast<int32_t>(sizeof(T)), MAX_INPUT_SIZE);
        PipeBarrier<PIPE_V>();
        uint16_t src0RepeatStride = inC / blockNum;
        GatherMaskParams maskParams{1, static_cast<uint16_t>(calW * kSize), src0RepeatStride, 0};
        uint64_t rsvdCnt = 0;
        GatherMask(srcOffsetTensor, srcOffsetTensor, ALL_PATTERN, true, inC_conv_group, maskParams, rsvdCnt);
        PipeBarrier<PIPE_V>();
    } else {
        ArithProgression(srcOffsetTensor, static_cast<int32_t>(0), static_cast<int32_t>(sizeof(T)), MAX_INPUT_SIZE);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::InitEventId()
{
    eventIdVToMte2 = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE2>());
    eventIdMte2ToV = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
    eventIdVToMte3 = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());
    eventIdVToS = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_S>());
    eventIdSToV = static_cast<event_t>(pipe.AllocEventID<HardEvent::S_V>());
    eventIdMte3ToV = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
}

template <typename T>
__aicore__ inline void DeformableConv2dND<T>::ReleaseEventId()
{
    pipe.ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2);
    pipe.ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    pipe.ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
    pipe.ReleaseEventID<HardEvent::V_S>(eventIdVToS);
    pipe.ReleaseEventID<HardEvent::S_V>(eventIdSToV);
    pipe.ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
}
}  // namespace DeformableConv2dNS

#endif  // DEFORMABLE_CONV2D_BASE_H
