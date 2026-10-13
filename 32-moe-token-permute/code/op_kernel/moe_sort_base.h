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
 * \file moe_sort_base.h
 * \brief
 */
#ifndef MOE_SORT_BASE_H
#define MOE_SORT_BASE_H

#include "kernel_operator.h"

namespace MoeTokenPermute {
using namespace AscendC;

template <typename T>
__aicore__ inline void GetBaseArithProgressionSupportInt32(
    const LocalTensor<T>& dstLocal, const T firstValue, const T diffValue, const int32_t count)
{
    for (int i = 0; i < count; i++) {
        dstLocal.SetValue(
            i, static_cast<T>(firstValue) + static_cast<T>(diffValue) * static_cast<T>(i)); // 数据有可能会截断。
    }
}

template <typename T>
__aicore__ inline void ArithProgressionSupportInt32(
    const LocalTensor<T>& dstLocal, const T firstValue, const T diffValue, const int32_t count)
{
    struct UnaryRepeatParams addsParamsStride1(1, 1, 1, 1);
    struct UnaryRepeatParams addsParamsStride8(1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);

    constexpr int32_t BLOCK_NUM = (ONE_BLK_SIZE / sizeof(T));
    constexpr int32_t REPEAT_NUM = (ONE_REPEAT_BYTE_SIZE / sizeof(T));
    if (count > BLOCK_NUM) {
        // Generates a basic arithmetic sequence of the BLOCK_NUM length for filling in subsequent arithmetic sequences.
        GetBaseArithProgressionSupportInt32<T>(dstLocal, firstValue, diffValue, BLOCK_NUM);
        auto eventIdSToV = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        if (count > REPEAT_NUM) {
            // broadcast from 1 block size to 8 block size
            SetVectorMask<T>(0, (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(BLOCK_NUM)) - 1));
            PipeBarrier<PIPE_V>();
            for (int i = 0; i < DEFAULT_BLK_NUM - 1; i++) {
                Adds<T, false>(
                    dstLocal[(i + 1) * BLOCK_NUM], dstLocal[i * BLOCK_NUM],
                    static_cast<T>(static_cast<float>(diffValue) * static_cast<float>(BLOCK_NUM)), MASK_PLACEHOLDER,
                    (uint16_t)1, addsParamsStride1);
                PipeBarrier<PIPE_V>();
            }
            int32_t repeat = count / REPEAT_NUM;
            int32_t tail = count % REPEAT_NUM;
            ResetMask();
            PipeBarrier<PIPE_V>();
            // Fills the following arithmetic progression with 8 block size arithmetic progressions
            for (int i = 0; i < repeat - 1; i++) {
                Adds<T, false>(
                    dstLocal[(i + 1) * REPEAT_NUM], dstLocal[i * REPEAT_NUM],
                    static_cast<T>(static_cast<float>(diffValue) * static_cast<float>(REPEAT_NUM)), MASK_PLACEHOLDER,
                    (uint16_t)1, addsParamsStride8);
                PipeBarrier<PIPE_V>();
            }
            if (tail > 0) {
                int32_t tail_aligned = (tail + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM;
                SetVectorMask<T>(tail_aligned);
                PipeBarrier<PIPE_V>();
                Adds<T, false>(
                    dstLocal[repeat * REPEAT_NUM], dstLocal[(repeat - 1) * REPEAT_NUM],
                    static_cast<T>(static_cast<float>(diffValue) * static_cast<float>(REPEAT_NUM)), MASK_PLACEHOLDER,
                    (uint16_t)1, addsParamsStride8);
                PipeBarrier<PIPE_V>();
            }
        } else {
            // Fills the following arithmetic progression
            int32_t countAligned = (count + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM;
            int32_t repeat = countAligned / BLOCK_NUM;
            SetVectorMask<T>(0, (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(BLOCK_NUM)) - 1));
            PipeBarrier<PIPE_V>();
            for (int i = 0; i < repeat - 1; i++) {
                Adds<T, false>(
                    dstLocal[(i + 1) * BLOCK_NUM], dstLocal[i * BLOCK_NUM],
                    static_cast<T>(static_cast<float>(diffValue) * static_cast<float>(BLOCK_NUM)), MASK_PLACEHOLDER,
                    (uint16_t)1, addsParamsStride1);
                PipeBarrier<PIPE_V>();
            }
        }
    } else {
        // When the length is less than BLOCK_NUM, the arithmetic sequence is generated directly by using a scalar.
        GetBaseArithProgressionSupportInt32<T>(dstLocal, firstValue, diffValue, count);
    }
}

template <typename T>
__aicore__ inline __inout_pipe__(MTE2) void DataCopyB64(
    const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal, const DataCopyExtParams& dataCopyParams,
    const DataCopyPadExtParams<int32_t>& padParams)
{
    DataCopyPadGm2UBImpl(
        (__ubuf__ int32_t*)dstLocal.GetPhyAddr(), (__gm__ int32_t*)srcGlobal.GetPhyAddr(), dataCopyParams, padParams);
}

template <typename T>
__aicore__ inline __inout_pipe__(MTE3) void DataCopyB64(
    const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal, const DataCopyExtParams& dataCopyParams)
{
    DataCopyPadUB2GMImpl(
        (__gm__ int32_t*)dstGlobal.GetPhyAddr(), (__ubuf__ int32_t*)srcLocal.GetPhyAddr(), dataCopyParams);
}

class MoeSortBase
{
public:
    __aicore__ inline MoeSortBase(){};

protected:
    __aicore__ inline void CleanWSCache();
    __aicore__ inline void SyncAll();

protected:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> sortDataCopyInQueue;
    TQue<QuePosition::VECOUT, 1> sortDataCopyOutQueue;
    TBuf<TPosition::VECCALC> tempBuffer;
    TBuf<TPosition::VECCALC> sortedBuffer;

    GlobalTensor<int32_t> sourceRowGm;
    GlobalTensor<int32_t> sortedExpertForSourceRowGm;
    GlobalTensor<int32_t> expandDstToSrcRowGm;

    int64_t tileLength;
    int64_t bufferNum = 1;
    int64_t totalLength;
    int64_t coreNum;

    static constexpr int64_t SYNC_GM_NUM = 2;
    static constexpr int64_t WORK_GM_NUM = 2;
    static constexpr int64_t DST_BLK_STRIDE = 1;
    static constexpr int64_t DST_REP_STRIDE = 8;
};

__aicore__ inline void MoeSortBase::SyncAll()
{
    if (coreNum == 1) {
        return;
    }
    AscendC::SyncAll();
}

} // namespace MoeTokenPermute
#endif // MOE_SORT_BASE_H