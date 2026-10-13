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
 * \file rms_norm_grad_common.h
 * \brief RmsNormGrad Common File
 */
#ifndef RMS_NORM_GRAD_BASE_H
#define RMS_NORM_GRAD_BASE_H
#include "kernel_operator.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
#include "reduce_common.h"

using namespace AscendC;
static volatile __gm__ uint32_t fixed_output_sync[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM_DB = 2;
constexpr uint32_t FLOAT_DTYPE = 0;
constexpr uint32_t FLOAT16_DTYPE = 1;
constexpr uint32_t BFLOAT16_DTYPE = 2;
constexpr uint32_t ALIGN_32 = 8;
constexpr uint32_t ALIGN_16 = 16;
constexpr uint32_t CORE_NUM = 50;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t EACH_CORE_HANDLE_NUM = BLOCK_SIZE / sizeof(int32_t);
const int32_t CAL_ONE_BLOCK_FP32 = 8;
const uint32_t REDUCESUMTHRESHOLD = 64;
const uint32_t SMALLD_THRESHOLD = 640;
constexpr uint32_t ROW_FACTOR_SPLIT_D = 32;
constexpr int32_t DIM_NUM = 2;
constexpr int32_t DIM_N = 0;
constexpr int32_t DIM_D = 1;

constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t ROW_TEMPLATE = 180;
constexpr int64_t COL_TEMPLATE = 64;  // 按64列进行分核
constexpr int64_t UB_SIZE = 180 * 1024 + 2 * DOUBLE_BUFFER * COL_TEMPLATE * sizeof(float);
constexpr int64_t FLOAT_ALIGN = 8;
constexpr uint64_t B32_REPEAT_STRIDE = 8;

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};

namespace RmsNormGrad{
struct deterministic_struct {
    LocalTensor<float> buffer1_;
    LocalTensor<float> buffer2_;
    GlobalTensor<float> workspaceGmOri_;
    GlobalTensor<float> dgammaGm_;
};
}
using namespace RmsNormGrad;

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

__aicore__ inline void ReduceSumFP32(
    uint32_t idx, LocalTensor<float>& dst_local, const LocalTensor<float>& src_local,
    const LocalTensor<float>& work_local, int32_t count, uint32_t col_val_align)
{
    if (g_coreType == AIV) {
        int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(float);
        uint64_t mask = elementNumPerRep;
        int32_t repeatTimes = count / elementNumPerRep;
        int32_t tailCount = count % elementNumPerRep;
        int32_t bodyCount = repeatTimes * elementNumPerRep;
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = ONE_REPEAT_BYTE_SIZE / ONE_BLK_SIZE;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = 0;
        repeatParams.dstBlkStride = 1;
        Duplicate(work_local, 0.0f, elementNumPerRep);
        PipeBarrier<PIPE_V>();
        int32_t start_addr = idx * col_val_align;
        if (likely(repeatTimes > 0)) {
            Add(work_local, src_local[start_addr], work_local, mask, repeatTimes, repeatParams);
            PipeBarrier<PIPE_V>();
        }
        if (unlikely(tailCount != 0)) {
            Add(work_local, src_local[start_addr + bodyCount], work_local, tailCount, 1, repeatParams);
            PipeBarrier<PIPE_V>();
        }
        AscendCUtils::SetMask<float>(elementNumPerRep);
        ReduceSum(dst_local[start_addr], work_local, work_local, elementNumPerRep);
        PipeBarrier<PIPE_V>();
    }
}

/*
 * only support count <= 255 * 64 = 16320
 */
__aicore__ inline float ReduceSumFP32_V2(const LocalTensor<float>& src_local, int32_t count)
{
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(float);
    int32_t repeatTimes = count / elementNumPerRep;
    int32_t tailCount = count % elementNumPerRep;
    int32_t bodyCount = repeatTimes * elementNumPerRep;
#ifdef __CCE_KT_TEST__
    assert(count <= MAX_REPEAT_TIMES * elementNumPerRep);
#endif
    float value = 0.0;
    if (g_coreType == AIV) {
        if (likely(repeatTimes > 0)) {
            AscendCUtils::SetMask<float>(elementNumPerRep);
            ReduceSum(src_local, src_local, src_local, elementNumPerRep);
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventId);
            WaitFlag<HardEvent::V_S>(eventId);
#ifdef __CCE_KT_TEST__
            uint64_t acc_val = GetAccVal();
#else
            uint64_t acc_val = GetAccVal();
#endif
            value = *reinterpret_cast<float*>(&acc_val);
        }
        if (unlikely(tailCount != 0)) {
            AscendCUtils::SetMask<float>(tailCount);
            ReduceSum(src_local[bodyCount], src_local[bodyCount], src_local[bodyCount], tailCount);
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventId);
            WaitFlag<HardEvent::V_S>(eventId);
#ifdef __CCE_KT_TEST__
            uint64_t acc_val = GetAccVal();
#else
            uint64_t acc_val = GetAccVal();
#endif
            value += *reinterpret_cast<float*>(&acc_val);
        }
    }
    return value;
}

template <typename T>
__aicore__ inline void Cast2FloatIf(LocalTensor<float>& castLocal, uint32_t srcOffset, uint32_t calcCount)
{
    if constexpr (!is_same<T, float>::value) {
        LocalTensor<T> castLocalB16 = castLocal.ReinterpretCast<T>();
        Cast(castLocal, castLocalB16[srcOffset], RoundMode::CAST_NONE, calcCount);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline uint32_t ROUND_UP(uint32_t x, uint32_t block_number)
{
    if (block_number > 0) {
        return (x + block_number - 1) / block_number * block_number;
    }
    return 0;
}

template <typename T>
__aicore__ inline void DataCopyCustom(
    GlobalTensor<T>& dstTensor, LocalTensor<T>& srcTensor, const uint32_t dstOffset, const uint32_t srcOffset,
    const uint32_t blockCount)
{
    uint32_t alignLen_ = ALIGN_32;
    if constexpr (!is_same<T, float>::value) {
        alignLen_ = ALIGN_16;
    }
    uint32_t calcLenAlign32 = (blockCount / alignLen_) * alignLen_;
    if (calcLenAlign32 > 0) {
        DataCopy(dstTensor[dstOffset], srcTensor[srcOffset], calcLenAlign32);
    }
    uint32_t calcLenTail32 = blockCount % alignLen_;

    if (calcLenTail32 > 0) {
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        for (uint32_t i = 0; i < calcLenTail32; i++) {
            dstTensor.SetValue(dstOffset + calcLenAlign32 + i, srcTensor.GetValue(srcOffset + calcLenAlign32 + i));
        }
        DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(dstTensor);
        PipeBarrier<PIPE_ALL>();
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    }
}

template <typename T>
__aicore__ inline void InitGmZero(
    GlobalTensor<T>& outGm, TBuf<TPosition::VECCALC>& TmpZeroTBuf, const uint32_t zeroLen, const uint32_t outOffset)
{
    uint32_t alignLen_ = BLOCK_SIZE / sizeof(T);
    LocalTensor<T> temp_zero_tensor = TmpZeroTBuf.Get<T>();

    Duplicate(temp_zero_tensor, (T)0.0, zeroLen);
    PipeBarrier<PIPE_ALL>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(outGm[outOffset], temp_zero_tensor, ROUND_UP(zeroLen, alignLen_));
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);

    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void dCopyIn(int64_t colIndex, int64_t colSize, int64_t rowSize,int64_t colAlignV_, deterministic_struct& deterministicStruct) {
    uint8_t rightPad = 0;
    bool isPad = false;
    int64_t colSizeMod = colSize % FLOAT_ALIGN;
    // 尾核补齐对齐
    if (colSizeMod != 0) {
        rightPad = FLOAT_ALIGN - colSizeMod;
        isPad = true;
    } 
#if __CCE_AICORE__ == 220
    if ASCEND_IS_AIV {
#endif
    DataCopyParams intriParams;
    intriParams.blockCount = rowSize;
    intriParams.blockLen   = (colSize + FLOAT_ALIGN - 1) / FLOAT_ALIGN;
    intriParams.srcStride  = (colAlignV_ - colSize) / FLOAT_ALIGN;
    intriParams.dstStride  = (COL_TEMPLATE - (colSize + rightPad)) / FLOAT_ALIGN;
    TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    SetFlag<HardEvent::V_MTE2>(eventID);
    WaitFlag<HardEvent::V_MTE2>(eventID);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventID);
    int64_t offset = colIndex * COL_TEMPLATE;
    DataCopy(deterministicStruct.buffer1_, deterministicStruct.workspaceGmOri_[offset], intriParams);
#if __CCE_AICORE__ == 220
    }
#endif
}

__aicore__ inline void dCompute(int64_t colIndex, int64_t rowIndex, int64_t colSize, int64_t rowSize, deterministic_struct& deterministicStruct) {
    int64_t colSizeMod = colSize % FLOAT_ALIGN;
    int64_t colSizeAlign = colSize;
    // 尾核补齐对齐
    if (colSizeMod != 0) {
        colSizeAlign += FLOAT_ALIGN - colSizeMod;
    } 
    TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventId);
    TEventID eventId1 = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::MTE3_V>(eventId1);
    WaitFlag<HardEvent::MTE3_V>(eventId1);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventId1);
    Duplicate(deterministicStruct.buffer2_, static_cast<float>(0.0), COL_TEMPLATE);
    PipeBarrier<PIPE_V>();
    uint64_t maskVal = colSize;
    uint8_t repeatTimes = MAX_REPEAT_TIMES;
    BinaryRepeatParams binaryRepeatParams;
    binaryRepeatParams.dstBlkStride = 1;
    binaryRepeatParams.src0BlkStride = 1;
    binaryRepeatParams.src1BlkStride = 1;
    binaryRepeatParams.dstRepStride = 0;
    binaryRepeatParams.src0RepStride = B32_REPEAT_STRIDE;
    binaryRepeatParams.src1RepStride = 0;
    int64_t rowRepeatTimes = (rowSize + MAX_REPEAT_TIMES - 1) / MAX_REPEAT_TIMES;
    for (int64_t i = 0; i < rowRepeatTimes; i++) {
        if (i == rowRepeatTimes - 1) {
            repeatTimes = rowSize - (rowRepeatTimes - 1) * MAX_REPEAT_TIMES;
        }
        Add(deterministicStruct.buffer2_,deterministicStruct.buffer1_[i * MAX_REPEAT_TIMES],deterministicStruct.buffer2_,maskVal,repeatTimes,binaryRepeatParams);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void dCopyOut(int64_t colIndex, int64_t colSize, deterministic_struct& deterministicStruct)
{
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = colSize * sizeof(float);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::V_MTE3>(eventID);
    WaitFlag<HardEvent::V_MTE3>(eventID);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventID);
    int64_t offset = colIndex * COL_TEMPLATE;
    DataCopyPad(deterministicStruct.dgammaGm_[offset], deterministicStruct.buffer2_, intriParams);
}

__aicore__ inline void FinalProcessDeterministic(int64_t tcolAlignV, int64_t tblockNum, int64_t tcol, deterministic_struct& deterministicStruct) {
    int64_t colAlignV_ = tcolAlignV;
    int64_t row_ = tblockNum;
    int64_t col_ = tcol;
    uint32_t coreNum = tblockNum;
    int64_t colcycleCount = (colAlignV_ + COL_TEMPLATE - 1) / COL_TEMPLATE;
    int64_t colcyclePerBlockCount = (colcycleCount + coreNum - 1) / coreNum;
    int64_t rowcycleCount = (row_ + ROW_TEMPLATE - 1) / ROW_TEMPLATE;
    int64_t colSize = COL_TEMPLATE;
    int64_t taskId = 0;
    for (int64_t blocktaskId = 0; blocktaskId < colcyclePerBlockCount; blocktaskId++) {
        taskId = blocktaskId * coreNum + GetBlockIdx();
        if (taskId < colcycleCount) {
            if (taskId == colcycleCount - 1) {
                colSize = col_ - COL_TEMPLATE * taskId;
            }
            dCopyIn(taskId,colSize,row_,colAlignV_,deterministicStruct);
            dCompute(taskId,0,colSize,row_,deterministicStruct);
            dCopyOut(taskId,colSize,deterministicStruct); 
        } else {
            break;
        }
    }
}

#endif // RMS_NORM_GRAD_BASE_H