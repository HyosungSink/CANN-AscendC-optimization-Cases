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
 * \file rotate_interleaved_split_bs.h
 * \brief
 */
#ifndef ROTATE_INTERLEAVED_SPLIT_BS_H
#define ROTATE_INTERLEAVED_SPLIT_BS_H
#include "rotate_interleaved_common.h"

namespace RotateInterleavedN {
using namespace AscendC;

template <typename T>
class InterleavedSplitBS {
public:
    __aicore__ inline InterleavedSplitBS(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                const RotaryPositionEmbeddingTilingData &tiling, TPipe *pipe);
    __aicore__ inline void Process();

protected:
    GlobalTensor<T> xGm;
    GlobalTensor<T> cosGm;
    GlobalTensor<T> sinGm;
    GlobalTensor<T> yGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueCos;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueY;
    TBuf<TPosition::VECCALC> tmpFp32Buf1;
    TBuf<TPosition::VECCALC> tmpFp32Buf2;
    TBuf<TPosition::VECCALC> tmpFp32Buf3;
    TBuf<TPosition::VECCALC> gatherOffsetBuf;

    // tilingdata
    uint64_t batchSize;
    uint64_t seqLen;
    uint64_t numHeads;
    uint64_t headDim;
    uint64_t frontCoreNum;
    uint64_t tailCoreNum;
    uint64_t coreCalcNum;
    uint64_t coreCalcTail;
    uint64_t ubCalcBNum;
    uint64_t ubCalcBLoop;
    uint64_t ubCalcBTail;

    // init tmp data
    uint32_t blockIdx;
    uint32_t ubCalcSeqLoop;
    uint64_t ioOffset;
    uint64_t triOffset;
    uint64_t bufferBsndSize;
    uint64_t bufferSdSize;
    uint64_t bufferNdSize;
    uint64_t bufferLenSize;
    uint64_t gatherOffsetLenSize;

    __aicore__ inline void InitData(const RotaryPositionEmbeddingTilingData &tiling);
    __aicore__ inline void CopyInX(LocalTensor<T> &x, uint32_t seqIdx, uint32_t batchIdx, uint32_t calcLen);
    __aicore__ inline void CopyInCos(LocalTensor<T> &cos, uint32_t seqIdx, uint32_t calcLen);
    __aicore__ inline void CopyInSin(LocalTensor<T> &sin, uint32_t seqIdx, uint32_t calcLen);
    __aicore__ inline void CopyOut(uint32_t seqIdx, uint32_t batchIdx, uint32_t calcLen);
    __aicore__ inline void Compute(uint32_t seqIdx, uint32_t batchIdx, LocalTensor<uint32_t> &gatherOffsetCast,
                                   uint32_t calcLen);
    __aicore__ inline void ComputeCastFp32(uint32_t seqIdx, uint32_t batchIdx, LocalTensor<uint32_t> &gatherOffsetCast,
                                           uint32_t calcLen);
};

template <typename T>
__aicore__ inline void InterleavedSplitBS<T>::Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                                   const RotaryPositionEmbeddingTilingData &tiling, TPipe *pipe)
{
    InitData(tiling);

    blockIdx = GetBlockIdx();
    bufferSdSize = seqLen * headDim;
    bufferNdSize = numHeads * headDim;

    if (blockIdx < frontCoreNum) {
        ubCalcSeqLoop = coreCalcNum;
        ioOffset = blockIdx * coreCalcNum * bufferNdSize;
        triOffset = blockIdx * coreCalcNum * headDim;
    } else if (coreCalcTail != 0) {
        ubCalcSeqLoop = coreCalcTail;
        ioOffset = frontCoreNum * coreCalcNum * bufferNdSize + (blockIdx - frontCoreNum) * coreCalcTail * bufferNdSize;
        triOffset = frontCoreNum * coreCalcNum * headDim + (blockIdx - frontCoreNum) * coreCalcTail * headDim;
    }

    bufferBsndSize = batchSize * seqLen * bufferNdSize;
    xGm.SetGlobalBuffer((__gm__ T *)x + ioOffset, bufferBsndSize);
    yGm.SetGlobalBuffer((__gm__ T *)y + ioOffset, bufferBsndSize);
    cosGm.SetGlobalBuffer((__gm__ T *)cos + triOffset, bufferSdSize);
    sinGm.SetGlobalBuffer((__gm__ T *)sin + triOffset, bufferSdSize);

    bufferLenSize = ubCalcBNum * bufferNdSize * sizeof(T);
    pipe->InitBuffer(inQueX, BUFFER_NUM, bufferLenSize);
    pipe->InitBuffer(inQueCos, BUFFER_NUM, bufferLenSize);
    pipe->InitBuffer(outQueY, BUFFER_NUM, bufferLenSize);

    if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
        bufferLenSize = ubCalcBNum * bufferNdSize * sizeof(float);
        pipe->InitBuffer(tmpFp32Buf1, bufferLenSize);
        pipe->InitBuffer(tmpFp32Buf2, bufferLenSize);
        pipe->InitBuffer(tmpFp32Buf3, bufferLenSize);
    }

    gatherOffsetLenSize = bufferNdSize * sizeof(int32_t);
    pipe->InitBuffer(gatherOffsetBuf, gatherOffsetLenSize);
}

template <typename T>
__aicore__ inline void InterleavedSplitBS<T>::InitData(const RotaryPositionEmbeddingTilingData &tiling)
{
    const RopeInterleavedParams &rotateInterleavedTiling = tiling.ropeInterleavedParams;
    batchSize = rotateInterleavedTiling.batchSize;
    seqLen = rotateInterleavedTiling.seqLen;
    numHeads = rotateInterleavedTiling.numHeads;
    headDim = rotateInterleavedTiling.headDim;
    frontCoreNum = rotateInterleavedTiling.frontCoreNum;
    tailCoreNum = rotateInterleavedTiling.tailCoreNum;
    coreCalcNum = rotateInterleavedTiling.coreCalcNum;
    coreCalcTail = rotateInterleavedTiling.coreCalcTail;
    ubCalcBNum = rotateInterleavedTiling.ubCalcBNum;
    ubCalcBLoop = rotateInterleavedTiling.ubCalcBLoop;
    ubCalcBTail = rotateInterleavedTiling.ubCalcBTail;
}

template <typename T>
__aicore__ inline void InterleavedSplitBS<T>::CopyInX(LocalTensor<T> &x, uint32_t seqIdx, uint32_t batchIdx,
                                                      uint32_t calcLen)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = calcLen;
    dataCopyParams.blockLen = bufferNdSize * sizeof(T);
    dataCopyParams.srcStride = (seqLen - 1) * bufferNdSize * sizeof(T);
    dataCopyParams.dstStride = 0;
    DataCopyPad(x, xGm[batchIdx * ubCalcBNum * seqLen * bufferNdSize + seqIdx * bufferNdSize], dataCopyParams,
                {false, 0, 0, 0});
    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
}

template <typename T>
__aicore__ inline void InterleavedSplitBS<T>::CopyInCos(LocalTensor<T> &cos, uint32_t seqIdx, uint32_t calcLen)
{
    DataCopyExtParams bsDataCopyTriParams;
    bsDataCopyTriParams.blockCount = 1;
    bsDataCopyTriParams.blockLen = headDim * sizeof(T);
    bsDataCopyTriParams.srcStride = 0;
    bsDataCopyTriParams.dstStride = 0;
    DataCopyPad(cos, cosGm[seqIdx * headDim], bsDataCopyTriParams, {false, 0, 0, 0});
    event_t eventId2MTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId2MTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventId2MTE2ToV);
    BroadCastTriToB1nd(cos, calcLen, numHeads, headDim);
}

template <typename T>
__aicore__ inline void InterleavedSplitBS<T>::CopyInSin(LocalTensor<T> &sin, uint32_t seqIdx, uint32_t calcLen)
{
    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);

    DataCopyExtParams bsDataCopyTriParams;
    bsDataCopyTriParams.blockCount = 1;
    bsDataCopyTriParams.blockLen = headDim * sizeof(T);
    bsDataCopyTriParams.srcStride = 0;
    bsDataCopyTriParams.dstStride = 0;
    DataCopyPad(sin, sinGm[seqIdx * headDim], bsDataCopyTriParams, {false, 0, 0, 0});
    event_t eventId3MTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId3MTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventId3MTE2ToV);
    BroadCastTriToB1nd(sin, calcLen, numHeads, headDim);
}

template <typename T>
__aicore__ inline void InterleavedSplitBS<T>::CopyOut(uint32_t seqIdx, uint32_t batchIdx, uint32_t calcLen)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = calcLen;
    dataCopyParams.blockLen = bufferNdSize * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = (seqLen - 1) * bufferNdSize * sizeof(T);
    LocalTensor<T> y = outQueY.DeQue<T>();
    DataCopyPad(yGm[batchIdx * ubCalcBNum * seqLen * bufferNdSize + seqIdx * bufferNdSize], y, dataCopyParams);
    outQueY.FreeTensor(y);
}

template <typename T>
__aicore__ inline void InterleavedSplitBS<T>::Process()
{
    LocalTensor<int32_t> gatherOffset = gatherOffsetBuf.Get<int32_t>();
    SetGatherSrcOffset(gatherOffset, headDim * numHeads, static_cast<int32_t>(sizeof(float)));
    LocalTensor<uint32_t> gatherOffsetCast = gatherOffset.ReinterpretCast<uint32_t>();

    if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
        for (uint32_t i = 0; i < ubCalcSeqLoop; ++i) {
            for (uint32_t j = 0; j < (ubCalcBTail == 0 ? ubCalcBLoop : ubCalcBLoop - 1); ++j) {
                ComputeCastFp32(i, j, gatherOffsetCast, ubCalcBNum);
                CopyOut(i, j, ubCalcBNum);
            }
            if (ubCalcBTail != 0) {
                ComputeCastFp32(i, ubCalcBLoop - 1, gatherOffsetCast, ubCalcBTail);
                CopyOut(i, ubCalcBLoop - 1, ubCalcBTail);
            }
        }
    } else {
        for (uint32_t i = 0; i < ubCalcSeqLoop; ++i) {
            for (uint32_t j = 0; j < (ubCalcBTail == 0 ? ubCalcBLoop : ubCalcBLoop - 1); ++j) {
                Compute(i, j, gatherOffsetCast, ubCalcBNum);
                CopyOut(i, j, ubCalcBNum);
            }
            if (ubCalcBTail != 0) {
                Compute(i, ubCalcBLoop - 1, gatherOffsetCast, ubCalcBTail);
                CopyOut(i, ubCalcBLoop - 1, ubCalcBTail);
            }
        }
    }
}

template <typename T>
__aicore__ inline void InterleavedSplitBS<T>::Compute(uint32_t seqIdx, uint32_t batchIdx,
                                                      LocalTensor<uint32_t> &gatherOffsetCast, uint32_t calcLen)
{
    uint64_t calcTotalNum = calcLen * bufferNdSize;

    LocalTensor<T> x = inQueX.AllocTensor<T>();
    CopyInX(x, seqIdx, batchIdx, calcLen);

    LocalTensor<T> cos = inQueCos.AllocTensor<T>();
    CopyInCos(cos, seqIdx, calcLen);

    LocalTensor<T> y = outQueY.AllocTensor<T>();
    Mul(y, x, cos, calcTotalNum);
    for (uint32_t i = 0; i < calcLen; ++i) {
        Gather(x[i * bufferNdSize], x[i * bufferNdSize], gatherOffsetCast, 0, bufferNdSize);
    }

    CopyInSin(cos, seqIdx, calcLen);

    Mul(x, x, cos, calcTotalNum);
    inQueCos.FreeTensor(cos);
    InterleavedInversion(x, calcTotalNum);
    Add(y, y, x, calcTotalNum);
    inQueX.FreeTensor(x);
    outQueY.EnQue(y);
}

template <typename T>
__aicore__ inline void InterleavedSplitBS<T>::ComputeCastFp32(uint32_t seqIdx, uint32_t batchIdx,
                                                              LocalTensor<uint32_t> &gatherOffsetCast, uint32_t calcLen)
{
    uint64_t calcTotalNum = calcLen * bufferNdSize;

    LocalTensor<T> x = inQueX.AllocTensor<T>();
    CopyInX(x, seqIdx, batchIdx, calcLen);
    LocalTensor<float> tmp32BSBuf1 = tmpFp32Buf1.Get<float>();
    Cast(tmp32BSBuf1, x, RoundMode::CAST_NONE, calcTotalNum);
    inQueX.FreeTensor(x);

    LocalTensor<T> cos = inQueCos.AllocTensor<T>();
    CopyInCos(cos, seqIdx, calcLen);
    LocalTensor<float> tmp32Buf2 = tmpFp32Buf2.Get<float>();
    Cast(tmp32Buf2, cos, RoundMode::CAST_NONE, calcTotalNum);

    LocalTensor<float> tmp32Buf3 = tmpFp32Buf3.Get<float>();
    Mul(tmp32Buf3, tmp32BSBuf1, tmp32Buf2, calcTotalNum);

    for (uint32_t i = 0; i < calcLen; ++i) {
        Gather(tmp32BSBuf1[i * bufferNdSize], tmp32BSBuf1[i * bufferNdSize], gatherOffsetCast, 0, bufferNdSize);
    }

    CopyInSin(cos, seqIdx, calcLen);
    Cast(tmp32Buf2, cos, RoundMode::CAST_NONE, calcTotalNum);
    inQueCos.FreeTensor(cos);

    Mul(tmp32BSBuf1, tmp32BSBuf1, tmp32Buf2, calcTotalNum);
    InterleavedInversion(tmp32BSBuf1, calcTotalNum);
    Add(tmp32Buf3, tmp32Buf3, tmp32BSBuf1, calcTotalNum);

    LocalTensor<T> y = outQueY.AllocTensor<T>();
    Cast(y, tmp32Buf3, RoundMode::CAST_RINT, calcTotalNum);
    outQueY.EnQue(y);
}

} // namespace RotateInterleavedN

#endif // ROTATE_INTERLEAVED_SPLIT_BS_H
