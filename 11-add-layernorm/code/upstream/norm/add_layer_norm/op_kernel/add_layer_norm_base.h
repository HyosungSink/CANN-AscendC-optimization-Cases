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
 * \file add_layer_norm_base.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_BASE_H_
#define ADD_LAYER_NORM_BASE_H_

#include "kernel_operator.h"
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
#endif

using namespace AscendC;
static constexpr float ZERO = 0;
constexpr uint32_t FLOAT_BLOCK_ELEM = 8;
constexpr int32_t ELEM_PER_REP_FP32 = 64; // ONE_REPEAT_BYTE_SIZE / sizeof(float)
constexpr int32_t ELEM_PER_REP_FP16 = 128;
constexpr uint32_t MAX_REP_NUM = 255;
constexpr uint32_t BROADCAST_ND_DIM_NUM = 2;    // only support 1 or 2
constexpr uint32_t BROADCAST_ND_LAST_INDEX = 1; // only support 0 or 1
constexpr uint32_t LAYER_NUM_TWO = 2;

#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
#define OUTPUT_MEAN_RSTD 1
#define SUPPORT_BF16 1
#else
#define OUTPUT_MEAN_RSTD 0
#define SUPPORT_BF16 0
#endif

template <typename Tp2, Tp2 v>
struct integral_constant {
    static constexpr Tp2 value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {
};
template <typename Tp3>
struct is_same<Tp3, Tp3> : public true_type {
};

template <typename T, template <typename U> typename R, template <typename U> typename S>
__aicore__ inline void DataCopyEx(
    const R<T>& dst, const S<T>& src, const uint32_t len, const uint32_t count = 1,
    const DataCopyPadParams& padParams = {})
{
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    DataCopyParams copyParams;
    copyParams.blockLen = len * sizeof(T);
    copyParams.blockCount = count;
    if constexpr (is_same<R<T>, AscendC::LocalTensor<T>>::value) {
        DataCopyPad(dst, src, copyParams, padParams);
    } else {
        DataCopyPad(dst, src, copyParams);
    }
#else
    auto elementCount = len * count;
    int32_t numPerBlock = ONE_BLK_SIZE / sizeof(T);
    if (elementCount % numPerBlock == 0) {
        DataCopy(dst, src, elementCount);
    } else {
        if constexpr (is_same<R<T>, AscendC::LocalTensor<T>>::value) {
            auto num = AlignUp(elementCount, numPerBlock);
            DataCopy(dst, src, num);
        } else {
            int32_t num = elementCount / numPerBlock * numPerBlock;
            DataCopy(dst, src, num);
            if (elementCount != num) {
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                for (int32_t i = 0; i < numPerBlock; i++) {
                    auto tensorValue = src.GetValue(elementCount - numPerBlock + i);
                    src.SetValue(i, tensorValue);
                }
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopy(dst[elementCount - numPerBlock], src, numPerBlock);
            }
        }
    }
#endif
}

/*
 * only support count <= 255 * 64 = 16320
 */
__aicore__ inline float ReduceSumFP32(const LocalTensor<float>& src_local, int32_t count)
{
    int32_t elementNumPerRep = ELEM_PER_REP_FP32;
    int32_t repeatTimes = count / elementNumPerRep;
    int32_t tailCount = count % elementNumPerRep;
    int32_t bodyCount = repeatTimes * elementNumPerRep;
#ifdef __CCE_KT_TEST__
    assert(count <= MAX_REPEAT_TIMES * elementNumPerRep);
#endif
    float value = 0.0;
    ResetMask();
    ReduceSum(src_local, src_local, src_local, count);
    value = GetAccVal<float>();
    return value;
}

__aicore__ inline void ReduceSumShort(
    const LocalTensor<float>& dst_local3, const LocalTensor<float>& src_local, const LocalTensor<float>& tmp_local,
    int32_t align_len, int32_t data_len, int32_t repeat)
{
    int32_t elementNum = ONE_BLK_SIZE / sizeof(float);
    int32_t maxRepeat = ELEM_PER_REP_FP32;
    int32_t tailCount = data_len % elementNum;
    uint32_t index1 = 0;
    uint8_t repStride = align_len / ONE_BLK_FLOAT_NUM;

    int32_t repeatTimesV1 = repeat / elementNum;
    int32_t bodyCount = repeatTimesV1 * elementNum;
    int32_t repeatTail = repeat % elementNum * elementNum;

    Duplicate<float>(tmp_local, ZERO, repeat * elementNum);
    PipeBarrier<PIPE_V>();
    for (index1 = 0; index1 + elementNum <= data_len; index1 += elementNum) {
        Add(tmp_local, tmp_local, src_local[index1], elementNum, repeat, {1, 1, 1, 1, 1, repStride});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailCount != 0)) {
        Add(tmp_local, tmp_local, src_local[index1], tailCount, repeat, {1, 1, 1, 1, 1, repStride});
    }
    PipeBarrier<PIPE_V>();
    if (repeatTimesV1 != 0) {
        BlockReduceSum<float>(dst_local3, tmp_local, repeatTimesV1, maxRepeat, 1, 1, elementNum);
    }
    if (repeatTail != 0) {
        BlockReduceSum<float>(dst_local3[bodyCount], tmp_local[bodyCount * elementNum], 1, repeatTail, 1, 1, elementNum);
    }
}

__aicore__ inline void ReduceSumForSmallReduceDimPreRepeat(
    const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal, const LocalTensor<float>& tmpLocal,
    const uint32_t elemNum, const uint32_t numLastDim, const uint32_t tailCount, const uint32_t repeat,
    const uint8_t repStride)
{
    uint32_t elemIndex = 0;
    for (; elemIndex + ELEM_PER_REP_FP32 <= numLastDim; elemIndex += ELEM_PER_REP_FP32) {
        Add(tmpLocal, srcLocal[elemIndex], tmpLocal, elemNum, repeat,
            {1, 1, 1, FLOAT_BLOCK_ELEM, repStride, FLOAT_BLOCK_ELEM});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailCount != 0)) {
        Add(tmpLocal, srcLocal[elemIndex], tmpLocal, tailCount, repeat,
            {1, 1, 1, FLOAT_BLOCK_ELEM, repStride, FLOAT_BLOCK_ELEM});
    }
    PipeBarrier<PIPE_V>();
    AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32); // set mask = 64
    WholeReduceSum<float, false>(dstLocal, tmpLocal, MASK_PLACEHOLDER, repeat, 1, 1, FLOAT_BLOCK_ELEM);
}

/*
 * reduce dim form (N, D) to (N, 1)
 * this reduce sum is for small reduce dim.
 */
__aicore__ inline void ReduceSumForSmallReduceDim(
    const LocalTensor<float>& dstLocal1, const LocalTensor<float>& srcLocal, const LocalTensor<float>& tmpLocal,
    const uint32_t numLastDimAligned, const uint32_t numLastDim, const uint32_t tailCount, const uint32_t repeat,
    const uint8_t repStride)
{
    uint32_t repeatTimes = repeat / MAX_REP_NUM;
    if (repeatTimes == 0) {
        ReduceSumForSmallReduceDimPreRepeat(
            dstLocal1, srcLocal, tmpLocal, ELEM_PER_REP_FP32, numLastDim, tailCount, repeat, repStride);
    } else {
        uint32_t repTailNum = repeat % MAX_REP_NUM;
        uint32_t repIndex = 0;
        uint32_t repElem;
        for (; repIndex + MAX_REP_NUM <= repeat; repIndex += MAX_REP_NUM) {
            ReduceSumForSmallReduceDimPreRepeat(
                dstLocal1[repIndex], srcLocal[repIndex * numLastDimAligned], tmpLocal[repIndex * ELEM_PER_REP_FP32],
                ELEM_PER_REP_FP32, numLastDim, tailCount, MAX_REP_NUM, repStride);
        }
        if (repTailNum != 0) {
            ReduceSumForSmallReduceDimPreRepeat(
                dstLocal1[repIndex], srcLocal[repIndex * numLastDimAligned], tmpLocal[repIndex * ELEM_PER_REP_FP32],
                ELEM_PER_REP_FP32, numLastDim, tailCount, repTailNum, repStride);
        }
    }
}

__aicore__ inline void InitVAForTransposeAscendC(__ubuf__ half *transposeDstAddr, __ubuf__ half *transposeSrcAddr, uint64_t dstLocalList[16], uint64_t srcLocalList[16]) {
  for (int i = 0; i < 16; ++i) {
    dstLocalList[i] = (uint64_t)(transposeDstAddr + i * 128);
    srcLocalList[i] = (uint64_t)(transposeSrcAddr + (i % 2 ? 16 : 0) + (i / 2) * 256);
  }
}

/*
 * for reduce dim > 64, need run InitVAForTranspose first.
 */
__aicore__ inline void CCEBroadCastShortAscendC(const LocalTensor<int16_t>& dstAddr, const LocalTensor<float>& srcAddr,
                                         const LocalTensor<int16_t>& transposeDstAddr, const LocalTensor<int16_t>& transposeSrcAddr,
                                         const LocalTensor<int16_t>& orOffsetINT16Addr, uint64_t *dstLocalList, uint64_t *srcLocalList,
                                         const uint32_t forCount, const uint32_t tailCount, const uint32_t repeat, const uint8_t repStride)
{
  SetMaskNorm();
  SetVectorMask<int16_t, MaskMode::NORMAL>(0x0, 0xffff);
  Duplicate<int16_t, false>(orOffsetINT16Addr, (int16_t)0, MASK_PLACEHOLDER, 1, (uint16_t)0, (uint8_t)0);    // all zero
  ResetMask();

  DataCopy<float>(transposeSrcAddr.ReinterpretCast<float>(), srcAddr, repStride * FLOAT_BLOCK_ELEM);
  PipeBarrier<PIPE_V>();

  Transpose(transposeDstAddr.ReinterpretCast<uint16_t>(), transposeSrcAddr.ReinterpretCast<uint16_t>());

  PipeBarrier<PIPE_V>();

  BinaryRepeatParams repeatParams;
  repeatParams.src0RepStride = 0;
  repeatParams.src0BlkStride = 1;
  repeatParams.src1RepStride = 0;
  repeatParams.src1BlkStride = 0;
  repeatParams.dstRepStride = FLOAT_BLOCK_ELEM * LAYER_NUM_TWO;
  repeatParams.dstBlkStride = 1;
  SetVectorMask<int16_t, MaskMode::NORMAL>(ELEM_PER_REP_FP16);
  Or<int16_t, false>(transposeSrcAddr, transposeDstAddr,
      orOffsetINT16Addr, MASK_PLACEHOLDER, FLOAT_BLOCK_ELEM, repeatParams);
  Or<int16_t, false>(transposeSrcAddr[(int64_t)ELEM_PER_REP_FP16],
      transposeDstAddr[(int64_t)ELEM_PER_REP_FP16],
      orOffsetINT16Addr, MASK_PLACEHOLDER, FLOAT_BLOCK_ELEM, repeatParams);

  PipeBarrier<PIPE_V>();
  TransDataTo5HDParams transDataParams;
  transDataParams.repeatTimes = FLOAT_BLOCK_ELEM;
  transDataParams.srcRepStride = LAYER_NUM_TWO;
  transDataParams.dstRepStride = 1;
  TransDataTo5HD<int16_t>(dstLocalList, srcLocalList, transDataParams);
  PipeBarrier<PIPE_V>();
  BinaryRepeatParams repeatParams1;
  repeatParams1.src0RepStride = 1;
  repeatParams1.src0BlkStride = 0;
  repeatParams1.src1RepStride = 0;
  repeatParams1.src1BlkStride = 0;
  repeatParams1.dstRepStride = repStride;
  repeatParams1.dstBlkStride = 1;
  for (int64_t forIndex = 0; forIndex < (int64_t)forCount; ++forIndex) {
    Or<int16_t, false>(dstAddr[forIndex * (int64_t)ELEM_PER_REP_FP16],
        transposeDstAddr,
        orOffsetINT16Addr, MASK_PLACEHOLDER,
        repeat, repeatParams1);
  }
  if (tailCount != 0) {
    SetVectorMask<int16_t, MaskMode::NORMAL>(tailCount * LAYER_NUM_TWO);
    Or<int16_t, false>(dstAddr[forCount * (int64_t)ELEM_PER_REP_FP16],
        transposeDstAddr,
        orOffsetINT16Addr, MASK_PLACEHOLDER,
        repeat, repeatParams1);
  }
}

__aicore__ inline void Level0MulFp32Short(
    const LocalTensor<float>& dstLocal, const LocalTensor<float>& src0Local, const LocalTensor<float>& src1Local,
    uint32_t alignElem, uint32_t repeat, uint32_t processElem)
{
    uint32_t maxElemFp32 = ELEM_PER_REP_FP32;
    uint8_t repStride = alignElem / FLOAT_BLOCK_ELEM;
    uint32_t tailCount = processElem % maxElemFp32;

    uint32_t repeatTimes = repeat / MAX_REP_NUM;

    uint32_t index = 0;
    uint32_t elemIndex = 0;
    if (likely(repeatTimes == 0)) {
        elemIndex = 0;
        for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
            Mul(dstLocal[elemIndex], src0Local[elemIndex], src1Local[elemIndex], maxElemFp32, repeat,
                {1, 1, 1, repStride, 0, repStride});
        }
        if (tailCount != 0) {
            Mul(dstLocal[elemIndex], src0Local[elemIndex], src1Local[elemIndex], tailCount, repeat,
                {1, 1, 1, repStride, 0, repStride});
        }
    } else {
        uint32_t repTailNum = repeat % MAX_REP_NUM;
        uint32_t repIndex = 0;
        uint32_t repElem;
        for (; repIndex + MAX_REP_NUM <= repeat; repIndex += MAX_REP_NUM) {
            elemIndex = 0;
            repElem = repIndex * alignElem;
            for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
                index = repElem + elemIndex;
                Mul(dstLocal[elemIndex], src0Local[index], src1Local[elemIndex], maxElemFp32, MAX_REP_NUM,
                    {1, 1, 1, repStride, 0, repStride});
            }
            if (tailCount != 0) {
                index = repElem + elemIndex;
                Mul(dstLocal[elemIndex], src0Local[index], src1Local[elemIndex], tailCount, MAX_REP_NUM,
                    {1, 1, 1, repStride, 0, repStride});
            }
        }
        if (repTailNum != 0) {
            elemIndex = 0;
            for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
                index = repElem + elemIndex;
                Mul(dstLocal[elemIndex], src0Local[index], src1Local[elemIndex], maxElemFp32, repTailNum,
                    {1, 1, 1, repStride, 0, repStride});
            }
            if (tailCount != 0) {
                index = repElem + elemIndex;
                Mul(dstLocal[elemIndex], src0Local[index], src1Local[elemIndex], tailCount, repTailNum,
                    {1, 1, 1, repStride, 0, repStride});
            }
        }
    }
}

__aicore__ inline void Level0AddFp32Short(
    const LocalTensor<float>& dstLocal, const LocalTensor<float>& src0Local, const LocalTensor<float>& src1Local,
    uint32_t alignElem, uint32_t repeat, uint32_t processElem)
{
    uint32_t addMaxElemFp32 = ELEM_PER_REP_FP32;
    uint8_t addRepStride = alignElem / FLOAT_BLOCK_ELEM;
    uint32_t addTailCount = processElem % addMaxElemFp32;

    uint32_t addRepeatTimes = repeat / MAX_REP_NUM;

    uint32_t addIndex = 0;
    uint32_t addElemIndex = 0;
    if (likely(addRepeatTimes == 0)) {
        addElemIndex = 0;
        for (; addElemIndex + addMaxElemFp32 <= processElem; addElemIndex += addMaxElemFp32) {
            Add(dstLocal[addElemIndex], src0Local[addElemIndex], src1Local[addElemIndex], addMaxElemFp32, repeat,
                {1, 1, 1, addRepStride, 0, addRepStride});
        }
        if (addTailCount != 0) {
            Add(dstLocal[addElemIndex], src0Local[addElemIndex], src1Local[addElemIndex], addTailCount, repeat,
                {1, 1, 1, addRepStride, 0, addRepStride});
        }
    } else {
        uint32_t addRepTailNum = repeat % MAX_REP_NUM;
        uint32_t addRepIndex = 0;
        uint32_t addRepElem;
        for (; addRepIndex + MAX_REP_NUM <= repeat; addRepIndex += MAX_REP_NUM) {
            addElemIndex = 0;
            addRepElem = addRepIndex * alignElem;
            for (; addElemIndex + addMaxElemFp32 <= processElem; addElemIndex += addMaxElemFp32) {
                addIndex = addRepElem + addElemIndex;
                Add(dstLocal[addElemIndex], src0Local[addIndex], src1Local[addElemIndex], addMaxElemFp32, MAX_REP_NUM,
                    {1, 1, 1, addRepStride, 0, addRepStride});
            }
            if (addTailCount != 0) {
                addIndex = addRepElem + addElemIndex;
                Add(dstLocal[addElemIndex], src0Local[addIndex], src1Local[addElemIndex], addTailCount, MAX_REP_NUM,
                    {1, 1, 1, addRepStride, 0, addRepStride});
            }
        }
        if (addRepTailNum != 0) {
            addElemIndex = 0;
            for (; addElemIndex + addMaxElemFp32 <= processElem; addElemIndex += addMaxElemFp32) {
                addIndex = addRepElem + addElemIndex;
                Add(dstLocal[addElemIndex], src0Local[addIndex], src1Local[addElemIndex], addMaxElemFp32, addRepTailNum,
                    {1, 1, 1, addRepStride, 0, addRepStride});
            }
            if (addTailCount != 0) {
                addIndex = addRepElem + addElemIndex;
                Add(dstLocal[addElemIndex], src0Local[addIndex], src1Local[addElemIndex], addTailCount, addRepTailNum,
                    {1, 1, 1, addRepStride, 0, addRepStride});
            }
        }
    }
}

#endif // __ADD_LAYER_NORM_BASE_H_
