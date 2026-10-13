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
 * \file moe_quant_sort_one_core.h
 * \brief
 */
#ifndef MOE_QUANT_SORT_ONE_CORE_H
#define MOE_QUANT_SORT_ONE_CORE_H

#include "moe_quant_mrgsort.h"

namespace MoeInitRoutingQuant {
using namespace AscendC;

class MoeSortOneCore : public MoeSortBase
{
public:
    __aicore__ inline MoeSortOneCore(){};
    __aicore__ inline void Init(
        GM_ADDR expertForSourceRow, GM_ADDR sourceRow, GM_ADDR sortedExpertForSourceRow, GM_ADDR workspace,
        const MoeInitRoutingQuantTilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void SortCompute();
    __aicore__ inline void CopyOut();

private:
    int64_t sortNum;
};

__aicore__ inline void MoeSortOneCore::CopyIn()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(this->totalLength * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(inLocal[0], expertForSourceRowGm, dataCopyParams, dataCopyPadParams);
    DataCopyPad(inLocal[this->sortNum], sourceRowGm, dataCopyParams, dataCopyPadParams);
    sortDataCopyInQueue.EnQue(inLocal);
}

__aicore__ inline void MoeSortOneCore::SortCompute()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
    LocalTensor<int32_t> expertForSourceRowLocal = inLocal[0];
    LocalTensor<float> expertForSourceRowLocalFp32;
    expertForSourceRowLocalFp32 = expertForSourceRowLocal.ReinterpretCast<float>();
    Cast(expertForSourceRowLocalFp32, expertForSourceRowLocal, RoundMode::CAST_ROUND, this->tileLength);
    Muls(expertForSourceRowLocalFp32, expertForSourceRowLocalFp32, (float)-1, this->tileLength);
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
    LocalTensor<float> tempTensor = tempBuffer.Get<float>(GetSortLen<float>(this->sortNum));
    Concat(concatLocal, expertForSourceRowLocalFp32, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);

    LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortNum));
    LocalTensor<uint32_t> sourceRowLocal;
    sourceRowLocal = inLocal[this->sortNum].ReinterpretCast<uint32_t>();
    Sort<float, true>(sortedLocal, concatLocal, sourceRowLocal, tempTensor, this->sortNum / ONE_REPEAT_SORT_NUM);

    LocalTensor<float> outLocal = sortDataCopyOutQueue.AllocTensor<float>();
    LocalTensor<float> sortedExpertForSourceRowLocal = outLocal[0];
    LocalTensor<uint32_t> expandDstToSrcRowLocal;
    expandDstToSrcRowLocal = outLocal[this->sortNum].ReinterpretCast<uint32_t>();
    Extract(sortedExpertForSourceRowLocal, expandDstToSrcRowLocal, sortedLocal, this->sortNum / ONE_REPEAT_SORT_NUM);
    Muls(sortedExpertForSourceRowLocal, sortedExpertForSourceRowLocal, (float)-1, this->tileLength);
    LocalTensor<int32_t> expertForSourceRowLocalInt32;
    expertForSourceRowLocalInt32 = sortedExpertForSourceRowLocal.ReinterpretCast<int32_t>();
    Cast(expertForSourceRowLocalInt32, sortedExpertForSourceRowLocal, RoundMode::CAST_ROUND, this->tileLength);
    sortDataCopyOutQueue.EnQue<float>(outLocal);
    sortDataCopyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void MoeSortOneCore::CopyOut()
{
    LocalTensor<int32_t> outLocal = sortDataCopyOutQueue.DeQue<int32_t>();
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = this->totalLength * sizeof(int32_t);
    DataCopyPad(sortedExpertForSourceRowGm, outLocal[0], intriParams);
    DataCopyPad(expandDstToSrcRowGm, outLocal[this->sortNum], intriParams);
    sortDataCopyOutQueue.FreeTensor(outLocal);
}

__aicore__ inline void MoeSortOneCore::Init(
    GM_ADDR expertForSourceRow, GM_ADDR sourceRow, GM_ADDR sortedExpertForSourceRow, GM_ADDR workspace,
    const MoeInitRoutingQuantTilingData* tilingData, TPipe* tPipe)
{
    this->tileLength = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
    this->sortNum = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    this->totalLength = tilingData->n * tilingData->k;
    this->coreNum = tilingData->coreNum;
    this->pipe = tPipe;

    expertForSourceRowGm.SetGlobalBuffer((__gm__ int32_t*)expertForSourceRow, this->tileLength);
    sourceRowGm.SetGlobalBuffer((__gm__ int32_t*)sourceRow, this->tileLength);
    sortedExpertForSourceRowGm.SetGlobalBuffer((__gm__ int32_t*)sortedExpertForSourceRow, this->tileLength);
    expandDstToSrcRowGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace), this->tileLength);

    int64_t coreNum = GetBlockNum();

    // key and value
    int64_t kvFactor = 2;
    int64_t buffSize = this->sortNum * sizeof(int32_t) * kvFactor;
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize);
    pipe->InitBuffer(sortDataCopyOutQueue, bufferNum, buffSize);
    pipe->InitBuffer(tempBuffer, buffSize);
    pipe->InitBuffer(sortedBuffer, buffSize);
}

__aicore__ inline void MoeSortOneCore::Process()
{
    if (GetBlockIdx() < 1) {
        CopyIn();
        SortCompute();
        CopyOut();
    }
    this->SyncAll();
}
} // namespace MoeInitRoutingQuant
#endif // MOE_QUANT_SORT_ONE_CORE_H