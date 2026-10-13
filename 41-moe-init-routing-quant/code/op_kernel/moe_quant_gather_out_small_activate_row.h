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
 * \file moe_quant_gather_out_small_activate_row.h
 * \brief
 */
#ifndef MOE_QUANT_GATHER_OUT_SMALL_ACTIVATE_ROW_H
#define MOE_QUANT_GATHER_OUT_SMALL_ACTIVATE_ROW_H

#include "moe_quant_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingQuant {
using namespace AscendC;

template <typename T>
class MoeGatherOutSmallActiveRow
{
public:
    __aicore__ inline MoeGatherOutSmallActiveRow(){};
    __aicore__ inline void Init(
        GM_ADDR inputActivations, GM_ADDR workspace, GM_ADDR expandSrcToDstRow, GM_ADDR expandedActivations,
        const MoeInitRoutingQuantTilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn(int64_t loop, int64_t row, int64_t colsLoop, LocalTensor<int32_t> indicesLocal);
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut(int64_t loop, int64_t row, int64_t colsLoop);

private:
    __aicore__ inline void CopyInIndices(int64_t progress);

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputActivationsCopyInQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> inputActivationsCopyOutQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> floatQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> halfQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> expandDstToSrcRowCopyInQueue;

    GlobalTensor<T> inputActivationsGm;
    GlobalTensor<int8_t> expandedActivationsGm;
    GlobalTensor<int32_t> expandDstToSrcRowGm;

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
    int64_t colsTileLength;

    int64_t indicesOffset;
    int64_t inputOffset;
    int64_t outOffset;
};

template <typename T>
__aicore__ inline void MoeGatherOutSmallActiveRow<T>::CopyInIndices(int64_t progress)
{
    this->indicesOffset = progress * this->perLoopRows;
    LocalTensor<int32_t> indicesLocal = expandDstToSrcRowCopyInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(this->currentLoopRows * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, expandDstToSrcRowGm[indicesOffset], dataCopyParams, dataCopyPadParams);

    expandDstToSrcRowCopyInQueue.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutSmallActiveRow<T>::Init(
    GM_ADDR inputActivations, GM_ADDR workspace, GM_ADDR expandSrcToDstRow, GM_ADDR expandedActivations,
    const MoeInitRoutingQuantTilingData* tilingData, TPipe* tPipe)
{
    this->pipe = tPipe;
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
    } else {
        this->coreRows = this->gatherOutTilingData->perCoreRows;
        this->perLoopRows = this->gatherOutTilingData->perCorePerLoopRows;
        this->lastLoopRows = this->gatherOutTilingData->perCoreLastLoopRows;
    }

    inputActivationsGm.SetGlobalBuffer((__gm__ T*)inputActivations, this->coreRows * this->cols);

    expandedActivationsGm.SetGlobalBuffer(
        (__gm__ int8_t*)expandedActivations + this->blockIdx * this->gatherOutTilingData->perCoreRows * this->cols,
        tilingData->n * tilingData->k * this->cols);

    expandDstToSrcRowGm.SetGlobalBuffer(
        (__gm__ int32_t*)workspace + this->blockIdx * this->gatherOutTilingData->perCoreRows,
        Align(this->coreRows, sizeof(int32_t)));

    pipe->InitBuffer(inputActivationsCopyInQueue, BUFFER_NUM, AlignBytes(this->maxColsOneLoop, sizeof(T)));
    pipe->InitBuffer(expandDstToSrcRowCopyInQueue, BUFFER_NUM, AlignBytes(this->perLoopRows, sizeof(int32_t)));

    pipe->InitBuffer(floatQueue, BUFFER_NUM, AlignBytes(this->maxColsOneLoop, sizeof(float)));
    pipe->InitBuffer(halfQueue, BUFFER_NUM, AlignBytes(this->maxColsOneLoop, sizeof(half)));
    pipe->InitBuffer(inputActivationsCopyOutQueue, BUFFER_NUM, AlignBytes(this->maxColsOneLoop, sizeof(int8_t)));
}

template <typename T>
__aicore__ inline void MoeGatherOutSmallActiveRow<T>::CopyIn(
    int64_t loop, int64_t row, int64_t colsLoop, LocalTensor<int32_t> indicesLocal)
{
    LocalTensor<T> inLocal = inputActivationsCopyInQueue.AllocTensor<T>();
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    inputOffset = indicesLocal.GetValue(row) % this->n * cols + colsLoop * maxColsOneLoop;
    event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
    WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);

    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = this->colsTileLength * sizeof(T);
    DataCopyPadParams dataCopyPadParams;
    DataCopyPad(inLocal, inputActivationsGm[inputOffset], intriParams, dataCopyPadParams);
    inputActivationsCopyInQueue.EnQue<T>(inLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutSmallActiveRow<T>::Compute()
{
    LocalTensor<int8_t> outLocal = inputActivationsCopyOutQueue.AllocTensor<int8_t>();
    LocalTensor<T> inLocal = inputActivationsCopyInQueue.DeQue<T>();
    LocalTensor<half> halfLocal = halfQueue.AllocTensor<half>();
    LocalTensor<float> floatLocal = floatQueue.AllocTensor<float>();
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        Cast(floatLocal, inLocal, RoundMode::CAST_NONE, this->colsTileLength);
        Cast(halfLocal, floatLocal, RoundMode::CAST_NONE, this->colsTileLength);
        Muls(halfLocal, halfLocal, static_cast<half>(this->scale), this->colsTileLength);
        Adds(halfLocal, halfLocal, static_cast<half>(this->offset), this->colsTileLength);
        LocalTensor<int32_t> intLocal = floatLocal.ReinterpretCast<int32_t>();
        Cast(intLocal, halfLocal, RoundMode::CAST_RINT, this->colsTileLength);
        SetDeqScale((half)1.000000e+00f);
        Cast(halfLocal, intLocal, RoundMode::CAST_RINT, this->colsTileLength);
        Cast(outLocal, halfLocal, RoundMode::CAST_RINT, this->colsTileLength);
    } else if constexpr (IsSameType<T, float>::value) {
        Cast(halfLocal, inLocal, RoundMode::CAST_NONE, this->colsTileLength);
        Muls(halfLocal, halfLocal, static_cast<half>(this->scale), this->colsTileLength);
        Adds(halfLocal, halfLocal, static_cast<half>(this->offset), this->colsTileLength);

        Cast(outLocal, halfLocal, RoundMode::CAST_RINT, this->colsTileLength);
    } else {
        Muls(inLocal, inLocal, static_cast<T>(this->scale), this->colsTileLength);
        Adds(inLocal, inLocal, static_cast<T>(this->offset), this->colsTileLength);
        Cast(outLocal, inLocal, RoundMode::CAST_RINT, this->colsTileLength);
    }
    inputActivationsCopyOutQueue.EnQue<int8_t>(outLocal);
    inputActivationsCopyInQueue.FreeTensor(inLocal);
    halfQueue.FreeTensor(halfLocal);
    floatQueue.FreeTensor(floatLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutSmallActiveRow<T>::CopyOut(int64_t loop, int64_t row, int64_t colsLoop)
{
    LocalTensor<int8_t> outLocal = inputActivationsCopyOutQueue.DeQue<int8_t>();
    outOffset = (loop * perLoopRows + row) * cols + colsLoop * maxColsOneLoop;
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = this->colsTileLength * sizeof(int8_t);
    DataCopyPad(expandedActivationsGm[outOffset], outLocal, intriParams);
    inputActivationsCopyOutQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeGatherOutSmallActiveRow<T>::Process()
{
    if (this->blockIdx < this->needCoreNum) {
        int64_t loops = (coreRows + perLoopRows - 1) / perLoopRows;
        int64_t colsLoops = Ceil(cols, maxColsOneLoop);
        int64_t tailCols = cols - (colsLoops - 1) * maxColsOneLoop;
        currentLoopRows = perLoopRows;
        for (int64_t loop = 0; loop < loops; loop++) {
            CopyInIndices(loop);
            LocalTensor<int32_t> indicesLocal = expandDstToSrcRowCopyInQueue.DeQue<int32_t>();
            if (loop == loops - 1) {
                currentLoopRows = lastLoopRows;
            }
            for (int64_t row = 0; row < currentLoopRows; row++) {
                colsTileLength = maxColsOneLoop;
                for (int64_t colsLoop = 0; colsLoop < colsLoops; colsLoop++) {
                    if (colsLoop == colsLoops - 1) {
                        colsTileLength = tailCols;
                    }
                    CopyIn(loop, row, colsLoop, indicesLocal);
                    Compute();
                    CopyOut(loop, row, colsLoop);
                }
            }
            expandDstToSrcRowCopyInQueue.FreeTensor(indicesLocal);
        }
    }
}
} // namespace MoeInitRoutingQuant
#endif // MOE_QUANT_GATHER_OUT_SMALL_ACTIVATE_ROW_H
