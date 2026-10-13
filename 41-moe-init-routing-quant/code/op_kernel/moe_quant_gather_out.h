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
 * \file moe_quant_gather_out.h
 * \brief
 */
#ifndef MOE_QUANT_GATHER_OUT_H
#define MOE_QUANT_GATHER_OUT_H

#include "moe_quant_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingQuant {
using namespace AscendC;

constexpr int64_t BUFFER_NUM = 1;

template <typename T>
class MoeGatherOut
{
public:
    __aicore__ inline MoeGatherOut(){};
    __aicore__ inline void Init(
        GM_ADDR inputActivations, GM_ADDR expandSrcToDstRow, GM_ADDR expandedActivations,
        const MoeInitRoutingQuantTilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInIndices(int64_t progress, int64_t kProcess);
    __aicore__ inline void CopyIn(int64_t progress, int64_t colsProgress);
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut(int64_t progress, int64_t colsProgress, LocalTensor<int32_t>& indicesLocal);
    __aicore__ inline void UpdataOffset(int64_t progress, int64_t colsProgress);
    __aicore__ inline void ParseTilingData(const MoeInitRoutingQuantTilingData* tilingData);

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputActivationsCopyInQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> expandSrcToDstRowCopyInQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> inputActivationsCopyOutQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> floatQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> halfQueue;

    GlobalTensor<T> inputActivationsGm;
    GlobalTensor<int32_t> expandSrcToDstRowGm;
    GlobalTensor<int8_t> expandedActivationsGm;

    const QuantGatherOutComputeTilingData* gatherOutTilingData;

    int64_t needCoreNum;
    int64_t blockIdx;
    int64_t cols;
    int64_t n;
    int64_t k;
    int64_t activateRows;
    float scale;
    float offset;

    int64_t currentLoopRows;
    int64_t coreRows;
    int64_t perLoopRows;
    int64_t lastLoopRows;
    int64_t maxColsOneLoop;
    int64_t coreK;
    int64_t perLoopK;
    int64_t lastLoopK;
    int64_t kTileLength;

    int64_t inputOffset;
    int64_t indicesOffset;
    int64_t colsTileLength;

    int64_t inputActivationsStartIndex;
    int64_t expandSrcToDstRowIndex;
};

template <typename T>
__aicore__ inline void MoeGatherOut<T>::UpdataOffset(int64_t progress, int64_t colsProgress)
{
    this->inputOffset = progress * this->perLoopRows * this->cols + colsProgress * maxColsOneLoop;
}

template <typename T>
__aicore__ inline void MoeGatherOut<T>::CopyInIndices(int64_t progress, int64_t kProcess)
{
    this->indicesOffset = progress * this->perLoopRows + kProcess * this->perLoopK * this->n;
    LocalTensor<int32_t> indicesLocal = expandSrcToDstRowCopyInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{
        static_cast<uint16_t>(kTileLength), static_cast<uint32_t>(this->currentLoopRows * sizeof(int32_t)),
        static_cast<uint32_t>((this->n - this->currentLoopRows) * sizeof(int32_t)), 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, expandSrcToDstRowGm[indicesOffset], dataCopyParams, dataCopyPadParams);
    expandSrcToDstRowCopyInQueue.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOut<T>::CopyIn(int64_t progress, int64_t colsProgress)
{
    LocalTensor<T> inLocal = inputActivationsCopyInQueue.AllocTensor<T>();

    uint32_t dstStride =
        (Align(this->colsTileLength, sizeof(int8_t)) * sizeof(T) - this->colsTileLength * sizeof(T)) / BLOCK_BYTES;
    DataCopyExtParams dataCopyParams{
        static_cast<uint16_t>(this->currentLoopRows), static_cast<uint32_t>(this->colsTileLength * sizeof(T)),
        static_cast<uint32_t>((this->cols - this->colsTileLength) * sizeof(T)), dstStride, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(inLocal, inputActivationsGm[inputOffset], dataCopyParams, dataCopyPadParams);
    inputActivationsCopyInQueue.EnQue<T>(inLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOut<T>::Compute()
{
    LocalTensor<T> inLocal = inputActivationsCopyInQueue.DeQue<T>();
    LocalTensor<int8_t> outLocal = inputActivationsCopyOutQueue.AllocTensor<int8_t>();
    LocalTensor<float> floatLocal = floatQueue.AllocTensor<float>();
    LocalTensor<half> halfLocal = halfQueue.AllocTensor<half>();
    uint32_t elements = this->currentLoopRows * Align(this->colsTileLength, sizeof(int8_t));
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        Cast(floatLocal, inLocal, RoundMode::CAST_NONE, elements);
        Cast(halfLocal, floatLocal, RoundMode::CAST_NONE, elements);
        Muls(halfLocal, halfLocal, static_cast<half>(this->scale), elements);
        Adds(halfLocal, halfLocal, static_cast<half>(this->offset), elements);
        LocalTensor<int32_t> intLocal = floatLocal.ReinterpretCast<int32_t>();
        Cast(intLocal, halfLocal, RoundMode::CAST_RINT, elements);
        SetDeqScale((half)1.000000e+00f);
        Cast(halfLocal, intLocal, RoundMode::CAST_RINT, elements);
        Cast(outLocal, halfLocal, RoundMode::CAST_RINT, elements);
    } else if constexpr (IsSameType<T, float>::value) {
        Cast(halfLocal, inLocal, RoundMode::CAST_NONE, elements);
        Muls(halfLocal, halfLocal, static_cast<half>(this->scale), elements);
        Adds(halfLocal, halfLocal, static_cast<half>(this->offset), elements);
        Cast(outLocal, halfLocal, RoundMode::CAST_RINT, elements);
    } else {
        Muls(inLocal, inLocal, static_cast<T>(this->scale), elements);
        Adds(inLocal, inLocal, static_cast<T>(this->offset), elements);
        Cast(outLocal, inLocal, RoundMode::CAST_RINT, elements);
    }
    inputActivationsCopyOutQueue.EnQue(outLocal);
    inputActivationsCopyInQueue.FreeTensor(inLocal);
    floatQueue.FreeTensor(floatLocal);
    halfQueue.FreeTensor(halfLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOut<T>::CopyOut(
    int64_t progress, int64_t colsProgress, LocalTensor<int32_t>& indicesLocal)
{
    LocalTensor<int8_t> outLocal = inputActivationsCopyOutQueue.DeQue<int8_t>();
    int64_t colsOffset = colsProgress * maxColsOneLoop;
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = this->colsTileLength * sizeof(int8_t);

    int64_t outOffset;
    uint32_t outIndex;
    int64_t inOffset;
    int64_t indicesFactor = Align(this->currentLoopRows, sizeof(int32_t));
    int64_t inFactor = Align(this->colsTileLength, sizeof(int8_t));
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

    for (int64_t idx = 0; idx < this->currentLoopRows; idx++) {
        inOffset = idx * inFactor;
        for (int64_t j = 0; j < kTileLength; j++) {
            outIndex = indicesLocal.GetValue(idx + j * indicesFactor);
            if (outIndex < this->activateRows) {
                outOffset = outIndex * this->cols + colsOffset;
                event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
                WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
                DataCopyPad(expandedActivationsGm[outOffset], outLocal[inOffset], intriParams);
            }
        }
    }
    inputActivationsCopyOutQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOut<T>::ParseTilingData(
    const MoeInitRoutingQuantTilingData* tilingData)
{
    this->blockIdx = GetBlockIdx();
    this->gatherOutTilingData = &(tilingData->gatherOutComputeParamsOp);

    this->needCoreNum = this->gatherOutTilingData->needCoreNum;
    this->activateRows = this->gatherOutTilingData->activateRows;
    this->cols = tilingData->cols;
    this->n = tilingData->n;
    this->k = tilingData->k;
    this->scale = tilingData->scale;
    this->offset = tilingData->offset;

    this->maxColsOneLoop = this->gatherOutTilingData->maxColsOneLoop;

    if (this->blockIdx == this->gatherOutTilingData->needCoreNum - 1) {
        this->coreRows = this->gatherOutTilingData->lastCoreRows;
        this->perLoopRows = this->gatherOutTilingData->lastCorePerLoopRows;
        this->lastLoopRows = this->gatherOutTilingData->lastCoreLastLoopRows;
        this->coreK = this->gatherOutTilingData->lastCoreK;
        this->perLoopK = this->gatherOutTilingData->lastCorePerLoopK;
        this->lastLoopK = this->gatherOutTilingData->lastCoreLastLoopK;
    } else {
        this->coreRows = this->gatherOutTilingData->perCoreRows;
        this->perLoopRows = this->gatherOutTilingData->perCorePerLoopRows;
        this->lastLoopRows = this->gatherOutTilingData->perCoreLastLoopRows;
        this->coreK = this->gatherOutTilingData->perCoreK;
        this->perLoopK = this->gatherOutTilingData->perCorePerLoopK;
        this->lastLoopK = this->gatherOutTilingData->perCoreLastLoopK;
    }
}

template <typename T>
__aicore__ inline void MoeGatherOut<T>::Init(
    GM_ADDR inputActivations, GM_ADDR expandSrcToDstRow, GM_ADDR expandedActivations,
    const MoeInitRoutingQuantTilingData* tilingData, TPipe* tPipe)
{
    this->pipe = tPipe;
    ParseTilingData(tilingData);

    if (this->gatherOutTilingData->splitFlag == SPLIT_N) {
        inputActivationsGm.SetGlobalBuffer(
            (__gm__ T*)inputActivations + this->blockIdx * this->gatherOutTilingData->perCoreRows * this->cols,
            this->coreRows * this->cols);
        expandSrcToDstRowGm.SetGlobalBuffer(
            (__gm__ int32_t*)expandSrcToDstRow + this->blockIdx * this->gatherOutTilingData->perCoreRows,
            tilingData->n * tilingData->k);
    } else if (this->gatherOutTilingData->splitFlag == SPLIT_K) {
        inputActivationsGm.SetGlobalBuffer((__gm__ T*)inputActivations, this->coreRows * this->cols);
        expandSrcToDstRowGm.SetGlobalBuffer(
            (__gm__ int32_t*)expandSrcToDstRow + this->blockIdx * this->gatherOutTilingData->perCoreK * this->n,
            tilingData->n * tilingData->k);
    }

    expandedActivationsGm.SetGlobalBuffer((__gm__ int8_t*)expandedActivations, this->n * this->k * this->cols);
    pipe->InitBuffer(inputActivationsCopyInQueue, BUFFER_NUM,
        this->perLoopRows * Align(this->maxColsOneLoop, sizeof(int8_t)) * sizeof(T));
    pipe->InitBuffer(
        floatQueue, BUFFER_NUM, this->perLoopRows * Align(this->maxColsOneLoop, sizeof(int8_t)) * sizeof(float));
    pipe->InitBuffer(
        halfQueue, BUFFER_NUM, this->perLoopRows * Align(this->maxColsOneLoop, sizeof(int8_t)) * sizeof(half));
    pipe->InitBuffer(
        inputActivationsCopyOutQueue, BUFFER_NUM, this->perLoopRows * AlignBytes(this->maxColsOneLoop, sizeof(T)));
    pipe->InitBuffer(
        expandSrcToDstRowCopyInQueue, BUFFER_NUM, this->perLoopK * AlignBytes(this->perLoopRows, sizeof(int32_t)));
}

template <typename T>
__aicore__ inline void MoeGatherOut<T>::Process()
{
    if (this->blockIdx < this->needCoreNum) {
        int64_t loops = (coreRows + perLoopRows - 1) / perLoopRows;
        int64_t colsLoops = Ceil(cols, maxColsOneLoop);
        int64_t tailCols = cols - (colsLoops - 1) * maxColsOneLoop;
        int64_t kLoops = Ceil(this->coreK, this->perLoopK);
        int64_t tailK = coreK - (kLoops - 1) * perLoopK;
        kTileLength = perLoopK;

        for (int64_t kLoop = 0; kLoop < kLoops; kLoop++) {
            if (kLoop == kLoops - 1) {
                kTileLength = tailK;
            }
            currentLoopRows = perLoopRows;
            for (int64_t loop = 0; loop < loops - 1; loop++) {
                colsTileLength = maxColsOneLoop;

                CopyInIndices(loop, kLoop);
                LocalTensor<int32_t> indicesLocal = expandSrcToDstRowCopyInQueue.DeQue<int32_t>();
                for (int64_t colsLoop = 0; colsLoop < colsLoops - 1; colsLoop++) {
                    UpdataOffset(loop, colsLoop);
                    CopyIn(loop, colsLoop);
                    Compute();
                    CopyOut(loop, colsLoop, indicesLocal);
                }

                colsTileLength = tailCols;
                UpdataOffset(loop, colsLoops - 1);
                CopyIn(loop, colsLoops - 1);
                Compute();
                CopyOut(loop, colsLoops - 1, indicesLocal);
                expandSrcToDstRowCopyInQueue.FreeTensor(indicesLocal);
            }

            currentLoopRows = lastLoopRows;
            colsTileLength = maxColsOneLoop;

            CopyInIndices(loops - 1, kLoop);
            LocalTensor<int32_t> indicesLocal = expandSrcToDstRowCopyInQueue.DeQue<int32_t>();
            for (int64_t colsLoop = 0; colsLoop < colsLoops - 1; colsLoop++) {
                UpdataOffset(loops - 1, colsLoop);
                CopyIn(loops - 1, colsLoop);
                Compute();
                CopyOut(loops - 1, colsLoop, indicesLocal);
            }

            colsTileLength = tailCols;
            UpdataOffset(loops - 1, colsLoops - 1);
            CopyIn(loops - 1, colsLoops - 1);
            Compute();
            CopyOut(loops - 1, colsLoops - 1, indicesLocal);
            expandSrcToDstRowCopyInQueue.FreeTensor(indicesLocal);
        }
    }
}
} // namespace MoeInitRoutingQuant
#endif // MOE_QUANT_GATHER_OUT_H
