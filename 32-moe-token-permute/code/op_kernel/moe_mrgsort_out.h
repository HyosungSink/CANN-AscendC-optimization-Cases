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
 * \file moe_mrgsort_out.h
 * \brief
 */
#ifndef MOE_MRGSORT_OUT_H
#define MOE_MRGSORT_OUT_H

#include "moe_mrgsort.h"
#include "kernel_operator.h"

namespace MoeTokenPermute {
using namespace AscendC;
template <typename T, typename T2>
class MoeMrgsortOut
{
public:
    __aicore__ inline MoeMrgsortOut(){};
    __aicore__ inline void Init(MoeMrgsortParam* param, TPipe* tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void SetInput(GlobalTensor<float>& gmInput, LocalTensor<float>& ubInput);
    __aicore__ inline void SetOutput(
        GlobalTensor<T>& gmOutput1, GlobalTensor<T2>& gmOutput2, LocalTensor<float>& ubOutput1,
        LocalTensor<float>& ubOutput2);
    __aicore__ inline void SetBuffer(LocalTensor<float>& tempBuffer);

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void UpdateMrgParam();
    __aicore__ inline void MrgsortCompute();
    __aicore__ inline void UpdateSortInfo();
    __aicore__ inline void Extract();
    __aicore__ inline void CopyOut();
    __aicore__ inline void ClearCache();

private:
    MoeMrgsortParam* param = nullptr;

    GlobalTensor<float> gmInputs[4];
    GlobalTensor<T> gmOutput1;
    GlobalTensor<T2> gmOutput2;

    LocalTensor<float> ubInputs[4];
    LocalTensor<float> tempBuffer;
    LocalTensor<T2> ubOutputCast2;
    LocalTensor<float> tmpUbInputs[4];

    // for extract
    LocalTensor<float> ubOutput1;
    LocalTensor<uint32_t> ubOutput2;

    // for copy out
    LocalTensor<T> ubOutputCast1;

    int64_t listNum{0};
    int64_t remainListNum{0};
    int64_t outOffset{0};
    int64_t offsets[4];
    int64_t listRemainElements[4];
    int64_t lengths[4];
    int64_t allRemainElements{0};
    int64_t curLoopSortedNum{0};

    // for MrgSort
    uint16_t validBitTail;
    uint16_t elementCountListTail[4];
    uint32_t listSortedNums[4];

    event_t eventIdMte3ToMte2;
    event_t eventIdVToMte3;
    event_t eventIdMte2ToV;
};

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::ClearCache()
{
    this->listNum = 0;
    this->allRemainElements = 0;
    this->outOffset = 0;
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::SetInput(GlobalTensor<float>& gmInput, LocalTensor<float>& ubInput)
{
    this->gmInputs[listNum] = gmInput;
    this->ubInputs[listNum] = ubInput;
    this->listNum += 1;
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::SetOutput(
    GlobalTensor<T>& gmOutput1, GlobalTensor<T2>& gmOutput2, LocalTensor<float>& ubOutput1,
    LocalTensor<float>& ubOutput2)
{
    this->gmOutput1 = gmOutput1;
    this->ubOutput1 = ubOutput1;
    this->ubOutputCast1 = ubOutput1.ReinterpretCast<T>();

    this->gmOutput2 = gmOutput2;
    this->ubOutput2 = ubOutput2.ReinterpretCast<uint32_t>();
    this->ubOutputCast2 = ubOutput2.ReinterpretCast<int32_t>();
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::SetBuffer(LocalTensor<float>& tempBuffer)
{
    this->tempBuffer = tempBuffer;
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::UpdateMrgParam()
{
    if (this->remainListNum == MERGE_LIST_TWO) {
        elementCountListTail[MERGE_LIST_IDX_TWO] = 0;
        elementCountListTail[MERGE_LIST_IDX_THREE] = 0;
        validBitTail = 0b0011;
    } else if (this->remainListNum == MERGE_LIST_THREE) {
        elementCountListTail[MERGE_LIST_IDX_THREE] = 0;
        validBitTail = 0b0111;
    } else if (this->remainListNum == MERGE_LIST_FOUR) {
        validBitTail = 0b1111;
    } else {
        validBitTail = 0b0001;
    }
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::CopyIn()
{
    this->remainListNum = 0;
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    for (int64_t i = 0, j = 0; i < listNum; i++) {
        lengths[i] = Min(param->oneLoopMaxElements, listRemainElements[i]);
        if (lengths[i] > 0) {
            DataCopy(
                this->ubInputs[i], this->gmInputs[i][offsets[i]], Align(GetSortLen<float>(lengths[i]), sizeof(float)));
            tmpUbInputs[j] = this->ubInputs[i];
            elementCountListTail[j] = lengths[i];
            this->remainListNum += 1;
            j++;
        }
    }
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::MrgsortCompute()
{
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    if (this->remainListNum == MERGE_LIST_TWO) {
        MrgSortSrcList sortListTail = MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[0], tmpUbInputs[0]);
        MrgSort<float, true>(this->tempBuffer, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);
    } else if (this->remainListNum == MERGE_LIST_THREE) {
        MrgSortSrcList sortListTail =
            MrgSortSrcList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[MERGE_LIST_IDX_TWO], tmpUbInputs[0]);
        MrgSort<float, true>(this->tempBuffer, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);
    } else if (this->remainListNum == MERGE_LIST_FOUR) {
        MrgSortSrcList sortListTail = MrgSortSrcList(
            tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[MERGE_LIST_IDX_TWO], tmpUbInputs[MERGE_LIST_IDX_THREE]);
        MrgSort<float, true>(this->tempBuffer, sortListTail, elementCountListTail, listSortedNums, validBitTail, 1);
    } else {
        DataCopy(
            this->tempBuffer, this->tmpUbInputs[0], Align(GetSortLen<float>(elementCountListTail[0]), sizeof(float)));
        listSortedNums[0] = elementCountListTail[0];
    }
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::UpdateSortInfo()
{
    curLoopSortedNum = 0;
    for (int64_t i = 0, j = 0; i < listNum; i++) {
        if (lengths[i] > 0) {
            // update remain size
            listRemainElements[i] -= listSortedNums[j];
            allRemainElements -= listSortedNums[j];
            // update offset
            offsets[i] += GetSortOffset<float>(listSortedNums[j]);
            // update current loop sorted nums
            curLoopSortedNum += listSortedNums[j];
            j += 1;
        }
    }
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::Extract()
{
    AscendC::Extract(this->ubOutput1, this->ubOutput2, this->tempBuffer, Ceil(curLoopSortedNum, ONE_REPEAT_SORT_NUM));
    // for sort: Muls(this->ubOutput1, this->ubOutput1, (float)-1, Align(curLoopSortedNum, sizeof(float)));
    // for sort: Cast(this->ubOutputCast1, this->ubOutput1, RoundMode::CAST_ROUND, Align(curLoopSortedNum,
    // sizeof(float)));
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::CopyOut()
{
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = curLoopSortedNum * sizeof(int32_t);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    LocalTensor<int32_t> ubOutputCast = this->ubOutput2.template ReinterpretCast<int32_t>();
    DataCopyPad(this->gmOutput1[outOffset], ubOutputCast, intriParams);
    outOffset += curLoopSortedNum;
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::Init(MoeMrgsortParam* param, TPipe* tPipe)
{
    this->param = param;
    this->allRemainElements = 0;
    eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

    for (int64_t i = 0; i < listNum; i++) {
        offsets[i] = GetSortOffset<float>(param->perListElements * i);
        if (i == listNum - 1) {
            listRemainElements[i] = param->lastListElements;
        } else {
            listRemainElements[i] = param->perListElements;
        }
        allRemainElements += listRemainElements[i];
    }
}

template <typename T, typename T2>
__aicore__ inline void MoeMrgsortOut<T, T2>::Process()
{
    for (; allRemainElements > 0;) {
        CopyIn();
        UpdateMrgParam();
        MrgsortCompute();
        UpdateSortInfo();
        Extract();
        CopyOut();
    }
    ClearCache();
}
} // namespace MoeTokenPermute
#endif // MOE_MRGSORT_OUT_H