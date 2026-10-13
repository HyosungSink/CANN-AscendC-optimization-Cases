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
 * \file moe_sort_one_core.h
 * \brief
 */
#ifndef MOE_SORT_ONE_CORE_H
#define MOE_SORT_ONE_CORE_H

#include "moe_sort_base.h"
#include "moe_mrgsort.h"
namespace MoeTokenPermute {
using namespace AscendC;

template <typename T>
class MoeSortOneCore : public MoeSortBase
{
public:
    __aicore__ inline MoeSortOneCore(){};
    __aicore__ inline void Init(
        GM_ADDR expertForSourceRow, GM_ADDR sortedExpertForSourceRow, GM_ADDR workspace,
        const MoeTokenPermuteTilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void SortCompute();
    __aicore__ inline void CopyOut();

private:
    int64_t sortNum;
    TBuf<TPosition::VECCALC> indexBuffer;
    LocalTensor<int32_t> indexLocal;
    GlobalTensor<T> expertForSourceRowGm;
    GlobalTensor<int32_t> sortedExpertForSourceRowGm;
};

template <typename T>
__aicore__ inline void MoeSortOneCore<T>::CopyIn()
{
    LocalTensor<T> inLocal = sortDataCopyInQueue.AllocTensor<T>();
    DataCopyExtParams dataCopyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(this->totalLength * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, (int)0};
    if constexpr (IsSameType<T, int64_t>::value) {
        DataCopyB64(inLocal, expertForSourceRowGm, dataCopyParams, dataCopyPadParams);
    } else {
        DataCopyPad(inLocal, expertForSourceRowGm, dataCopyParams, dataCopyPadParams);
    }
    sortDataCopyInQueue.EnQue(inLocal);
    PipeBarrier<PIPE_V>();
    ArithProgression<int32_t>(indexLocal, static_cast<int32_t>(0), static_cast<int32_t>(1), this->totalLength);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void MoeSortOneCore<T>::SortCompute()
{
    LocalTensor<T> inLocal = sortDataCopyInQueue.DeQue<T>();
    LocalTensor<T> expertForSourceRowLocal = inLocal[0];
    LocalTensor<float> expertForSourceRowLocalFp32;
    expertForSourceRowLocalFp32 = expertForSourceRowLocal.template ReinterpretCast<float>();

    LocalTensor<int32_t> expertForSourceRowLocalInt32 = expertForSourceRowLocal.template ReinterpretCast<int32_t>();

    Cast(expertForSourceRowLocalFp32, expertForSourceRowLocal, RoundMode::CAST_ROUND, this->tileLength);

    PipeBarrier<PIPE_V>();

    Muls(expertForSourceRowLocalFp32, expertForSourceRowLocalFp32, (float)-1, this->tileLength);
    PipeBarrier<PIPE_V>();

    int64_t duplicateNum = this->totalLength % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = this->totalLength - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertForSourceRowLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
    }

    LocalTensor<float> concatLocal;
    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();

    Concat(concatLocal, expertForSourceRowLocalFp32, outLocal, this->sortNum / ONE_REPEAT_SORT_NUM);

    LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortNum));
    LocalTensor<uint32_t> sourceRowLocal;
    sourceRowLocal = indexLocal.ReinterpretCast<uint32_t>();

    Sort<float, true>(sortedLocal, concatLocal, sourceRowLocal, outLocal, this->sortNum / ONE_REPEAT_SORT_NUM);

    LocalTensor<float> sortedExpertForSourceRowLocal = outLocal[0];
    LocalTensor<uint32_t> expandDstToSrcRowLocal;
    expandDstToSrcRowLocal = outLocal[this->sortNum].template ReinterpretCast<uint32_t>();
    Extract(sortedExpertForSourceRowLocal, expandDstToSrcRowLocal, sortedLocal, this->sortNum / ONE_REPEAT_SORT_NUM);
    // for sort :Muls(sortedExpertForSourceRowLocal, sortedExpertForSourceRowLocal, (float)-1, this->tileLength);

    // for sort :LocalTensor<int32_t> expertForSourceRowLocalInt32;
    // for sort :expertForSourceRowLocalInt32 = sortedExpertForSourceRowLocal.ReinterpretCast<int32_t>();
    // for sort :Cast(expertForSourceRowLocalInt32, sortedExpertForSourceRowLocal, RoundMode::CAST_ROUND,
    // this->tileLength);
    sortDataCopyOutQueue.EnQue<float>(outLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
}

template <typename T>
__aicore__ inline void MoeSortOneCore<T>::CopyOut()
{
    LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.DeQue<int32_t>();
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = this->totalLength * sizeof(int32_t);
    DataCopyPad(sortedExpertForSourceRowGm, outLocal[this->sortNum], intriParams);
    // for sort :DataCopyPad(expandDstToSrcRowGm, outLocal[this->sortNum], intriParams);
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeSortOneCore<T>::Init(
    GM_ADDR expertForSourceRow, GM_ADDR sortedExpertForSourceRow, GM_ADDR workspace,
    const MoeTokenPermuteTilingData* tilingData, TPipe* tPipe)
{
    this->tileLength = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
    this->sortNum = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    this->totalLength = tilingData->n * tilingData->topK;
    this->coreNum = tilingData->coreNum;
    this->pipe = tPipe;

    expertForSourceRowGm.SetGlobalBuffer((__gm__ T*)expertForSourceRow, this->tileLength);
    sortedExpertForSourceRowGm.SetGlobalBuffer((__gm__ int32_t*)sortedExpertForSourceRow, this->tileLength);
    expandDstToSrcRowGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace), this->tileLength);

    // key and value
    int64_t kvFactor = 2;

    int64_t indexBufferSize = this->sortNum * sizeof(int32_t);
    int64_t sortDataBufferSize = indexBufferSize * sizeof(int64_t) / sizeof(int32_t);

    int64_t buffSize = indexBufferSize * kvFactor;
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, sortDataBufferSize);
    pipe->InitBuffer(indexBuffer, indexBufferSize);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, buffSize);
    pipe->InitBuffer(sortedBuffer, buffSize);
    indexLocal = indexBuffer.Get<int32_t>();
}

template <typename T>
__aicore__ inline void MoeSortOneCore<T>::Process()
{
    if (GetBlockIdx() < 1) {
        CopyIn();
        SortCompute();
        CopyOut();
    }
    AscendC::SyncAll();
}
} // namespace MoeTokenPermute
#endif // MOE_SORT_ONE_CORE_H