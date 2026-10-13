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
 * \file add_rms_norm_quant_base.h
 * \brief add rms norm quant base file
 */
#ifndef _ADD_RMS_NORM_QUANT_BASE_H_
#define _ADD_RMS_NORM_QUANT_BASE_H_

#include "kernel_operator.h"
#include "reduce_common.h"

namespace AddRmsNormQuantBase {
using namespace AscendC;

__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 220 && __CCE_AICORE__ != 310 && __CCE_AICORE__ != 220
#define bfloat16_t int16_t
#endif
constexpr int32_t NUM_PER_BLK_FP32 = 8;
constexpr int32_t BUFFER_NUM = 1;        // tensor num for each queue
constexpr int32_t NUM_PER_REP_FP32 = 64; // ONE_REPEAT_BYTE_SIZE / sizeof(float);
constexpr int32_t BLOCK_SIZE = 32;
constexpr uint32_t ONCE_VECTOR_SIZE = 256;
constexpr float ONE = 1;
constexpr int32_t SECOND_LOOP = 2;
constexpr int32_t HALf_INTERVAL = 2;

constexpr uint32_t V_LENGTH = GetVRegSize() / sizeof(float);
constexpr uint64_t ALIGN_512_FACTOR = 512;
constexpr uint64_t ALIGN_32_FACTOR = 32;
constexpr int32_t CONST_FACTOR_2 = 2;
constexpr uint32_t SUM_COUNT = 2;

template <typename T>
__aicore__ inline T CeilDiv(T x, T y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

template <typename Tp4, Tp4 v>
struct integral_constant {
    static constexpr Tp4 value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

__aicore__ inline void ReduceSumFP32(
    const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local_v1, const LocalTensor<float>& work_local,
    int32_t count)
{
    // count need smaller than 255 repeat
    uint64_t mask = NUM_PER_REP_FP32;
    int32_t tailCount = count % NUM_PER_REP_FP32;
    int32_t repeatTimes = count / NUM_PER_REP_FP32;
    int32_t bodyCount = repeatTimes * NUM_PER_REP_FP32;
    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = ONE_REPEAT_BYTE_SIZE / ONE_BLK_SIZE;
    repeatParams.src1BlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1RepStride = 0;
    repeatParams.dstRepStride = 0;
    repeatParams.dstBlkStride = 1;
    Duplicate(work_local, ZERO, NUM_PER_REP_FP32);
    PipeBarrier<PIPE_V>();
    if (likely(repeatTimes > 0)) {
        Add(work_local, src_local_v1, work_local, mask, repeatTimes, repeatParams);
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailCount != 0)) {
        Add(work_local, src_local_v1[bodyCount], work_local, tailCount, 1, repeatParams);
        PipeBarrier<PIPE_V>();
    }
    AscendCUtils::SetMask<float>(NUM_PER_REP_FP32);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        WholeReduceSum<float, false>(dst_local, work_local, NUM_PER_REP_FP32, 1, 0, 1, 0);
    }
#else
    WholeReduceSum<float, false>(dst_local, work_local, NUM_PER_REP_FP32, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ReduceSumCustom(
    const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local, const LocalTensor<float>& work_local,
    int32_t count)
{
    ReduceSumFP32(dst_local, src_local, work_local, count);
}
__aicore__ inline void ReduceSumFP32ToBlock(
    const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local, const LocalTensor<float>& work_local,
    int32_t count)
{
    // count need smaller than 255 repeat
    uint64_t mask = NUM_PER_REP_FP32;
    int32_t repeatTimesBlock = count / NUM_PER_REP_FP32;
    int32_t tailCount = count % NUM_PER_REP_FP32;
    int32_t bodyCount = repeatTimesBlock * NUM_PER_REP_FP32;
    BinaryRepeatParams repeatParamsBlock;
    repeatParamsBlock.src0RepStride = ONCE_VECTOR_SIZE / BLOCK_SIZE;
    repeatParamsBlock.src0BlkStride = 1;
    repeatParamsBlock.src1RepStride = 0;
    repeatParamsBlock.src1BlkStride = 1;
    repeatParamsBlock.dstRepStride = 0;
    repeatParamsBlock.dstBlkStride = 1;
    Duplicate(work_local, ZERO, NUM_PER_REP_FP32);
    PipeBarrier<PIPE_V>();
    if (likely(repeatTimesBlock > 0)) {
        Add(work_local, src_local, work_local, mask, repeatTimesBlock, repeatParamsBlock);
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailCount != 0)) {
        Add(work_local, src_local[bodyCount], work_local, tailCount, 1, repeatParamsBlock);
        PipeBarrier<PIPE_V>();
    }
    BlockReduceSum(dst_local, work_local, 1, mask, 1, 1, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void BlockReduceSumFP32(
    const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local, int32_t count)
{
    // count need multiple of 8
    int32_t repeatTimes = count / NUM_PER_REP_FP32;
    int32_t tailCount = count % NUM_PER_REP_FP32;
    int32_t dstAddr = repeatTimes * 8;
    int32_t srcAddr = repeatTimes * NUM_PER_REP_FP32;
    if (likely(repeatTimes > 0)) {
        BlockReduceSum(dst_local, src_local, repeatTimes, NUM_PER_REP_FP32, 1, 1, DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
    }
    if (tailCount != 0) {
        BlockReduceSum(dst_local[dstAddr], src_local[srcAddr], 1, tailCount, 1, 1, DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void doScales(
    LocalTensor<float> quantLocal, LocalTensor<float> yLocal, TBuf<TPosition::VECCALC>& scalesBuf, uint32_t divMode,
    uint32_t num)
{
    LocalTensor<float> scalesLocal = scalesBuf.Get<float>();
    if (divMode) {
        Div(quantLocal, yLocal, scalesLocal, num);
        PipeBarrier<PIPE_V>();
    } else {
        Mul(quantLocal, yLocal, scalesLocal, num);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void doZeroPoints(
    LocalTensor<float> yLocal, TBuf<TPosition::VECCALC>& zeroPointsBuf, uint32_t num, uint32_t hasZeroPoints)
{
    if (!hasZeroPoints) {
        return;
    }
    LocalTensor<float> zeroPointsFp32 = zeroPointsBuf.Get<float>();
    Add(yLocal, yLocal, zeroPointsFp32, num);
    PipeBarrier<PIPE_V>();
}

template <typename T, typename U, typename R>
__aicore__ inline void DataCopyCustom(const U& dstTensor, const R& srcTensor, const uint32_t count)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    DataCopyParams copyParams;
    copyParams.blockLen = count * sizeof(T);
    copyParams.blockCount = 1;
    if constexpr (is_same<U, AscendC::LocalTensor<T>>::value) {
        DataCopyPadParams padParams;
        DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
    } else {
        DataCopyPad(dstTensor, srcTensor, copyParams);
    }
#else
    // only support count greater than 32byte
    int32_t numPerBlock = ONE_BLK_SIZE / sizeof(T);
    if (count % numPerBlock == 0) {
        DataCopy(dstTensor, srcTensor, count);
    } else {
        if constexpr (is_same<U, AscendC::LocalTensor<T>>::value) {
            int32_t num = AlignUp(count, numPerBlock);
            DataCopy(dstTensor, srcTensor, num);
        } else {
            if (count < numPerBlock) {
                DataCopy(dstTensor, srcTensor, numPerBlock);
            } else {
                int32_t num = count / numPerBlock * numPerBlock;
                DataCopy(dstTensor, srcTensor, num);
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                for (int32_t i = 0; i < numPerBlock; i++) {
                    T tensorValue = srcTensor.GetValue(count - numPerBlock + i);
                    srcTensor.SetValue(i, tensorValue);
                }
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopy(dstTensor[count - numPerBlock], srcTensor, numPerBlock);
            }
        }
    }
#endif
}

__aicore__ inline void RoundFloat2Int8(LocalTensor<int8_t>& dstTensor, LocalTensor<float>& srcTensor, int32_t size)
{
    Cast(srcTensor.ReinterpretCast<int32_t>(), srcTensor, RoundMode::CAST_RINT, size);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(srcTensor.ReinterpretCast<half>(), srcTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, size);
    PipeBarrier<PIPE_V>();
    Cast(dstTensor, srcTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, size);
}

template <typename TScale>
__aicore__ inline void CopyInScales(
    TBuf<TPosition::VECCALC>& scalesBuf, const GlobalTensor<TScale>& scalesGm, uint32_t num, uint32_t ubFactor)
{
    LocalTensor<float> scales1Local = scalesBuf.Get<float>();
    if constexpr (is_same<TScale, float>::value) {
        DataCopyCustom<float>(scales1Local, scalesGm, num);
    } else { // bfloat16
        LocalTensor<bfloat16_t> scales1Bf16 = scalesBuf.Get<bfloat16_t>()[ubFactor];
        DataCopyCustom<bfloat16_t>(scales1Bf16, scalesGm, num);
        event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);
        Cast(scales1Local, scales1Bf16, RoundMode::CAST_NONE, num);
        PipeBarrier<PIPE_V>();
    }
}

template <typename TOffset>
__aicore__ inline void CopyInZeroPoints(
    TBuf<TPosition::VECCALC>& zeroPointsBuf, const GlobalTensor<TOffset>& zeroPointsGm, uint32_t num, uint32_t ubFactor,
    uint32_t hasZeroPoints)
{
    if (!hasZeroPoints) {
        return;
    }
    LocalTensor<float> zeroPointsFp32 = zeroPointsBuf.Get<float>();
    if constexpr (is_same<TOffset, int32_t>::value) {
        LocalTensor<int32_t> zeroPointsInt32 = zeroPointsBuf.Get<int32_t>();
        DataCopyCustom<int32_t>(zeroPointsInt32, zeroPointsGm, num);
        event_t eventMte2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2V2);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V2);
        Cast(zeroPointsFp32, zeroPointsInt32, RoundMode::CAST_NONE, num);
        PipeBarrier<PIPE_V>();
    } else {
        LocalTensor<bfloat16_t> zeroPointsBf16 = zeroPointsBuf.Get<bfloat16_t>()[ubFactor];
        DataCopyCustom<bfloat16_t>(zeroPointsBf16, zeroPointsGm, num);
        event_t eventMte2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2V2);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V2);
        Cast(zeroPointsFp32, zeroPointsBf16, RoundMode::CAST_NONE, num);
        PipeBarrier<PIPE_V>();
    }
}
} // namespace AddRmsNormQuantBase
#endif // _ADD_RMS_NORM_QUANT_BASE_H_
