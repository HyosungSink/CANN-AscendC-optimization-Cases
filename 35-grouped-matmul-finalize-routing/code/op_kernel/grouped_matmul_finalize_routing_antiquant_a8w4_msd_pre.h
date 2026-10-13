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
 * \file grouped_matmul_finalize_routing_antiquant_a8w4_msd_pre.h
 * \brief
 */

#ifndef ASCENDC_GROUPED_MATMUL_FINALIZE_ROUTING_ANTIQUANT_A8W4_MSD_PRE_H
#define ASCENDC_GROUPED_MATMUL_FINALIZE_ROUTING_ANTIQUANT_A8W4_MSD_PRE_H

#include "kernel_operator.h"
namespace GroupedMatmulFinalizeRouting {
using namespace AscendC;

constexpr int TWO = 2;
constexpr int EIGHT = 8;
constexpr float ONE_EIGHT = 0.0625;
constexpr int BUFFER_NUM_A8W4_PRE = 1;
constexpr size_t LEN_128 = 128;     // 16bit operator
constexpr int DATA_BLOCK_SIZE_32 = 32;
constexpr int ROW_SUM_REPEAT_LEN = 256;

struct MMPreInitParams {
    GM_ADDR x;
    GM_ADDR y;
    GM_ADDR groupList;
    GM_ADDR workspace;
};

template <bool groupListType_>
struct ParamW4A8Pre {
    static const bool groupListType = groupListType_;
};

template <class P>
class GMMA8W4PreProcess {
public:
    __aicore__ inline GMMA8W4PreProcess(){};
    __aicore__ inline void Init(MMPreInitParams& preInitParams, const GroupMatmulFRTilingData& tilingData, TPipe *pipe);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessLoop();
    __aicore__ inline void ProcessPerToken(uint32_t xloop);
    __aicore__ inline void ComputeTotalGroup();
private:
    TQue<QuePosition::VECIN, 1> vecInQueueX;
    TQue<QuePosition::VECIN, 1> vecInQueueXBak;
    TQue<QuePosition::VECOUT, 1> vecOutQueueA1;
    TQue<QuePosition::VECOUT, 1> vecOutQueueA2;
    TQue<QuePosition::VECOUT, 1> vecOutQueueA3;
    TQue<QuePosition::VECOUT, 1> vecOutQueue0F;
    TQue<QuePosition::VECIN, 1> vecInQueueG;
    TQue<QuePosition::VECIN, 1> vecInQueueGF;
    TQue<QuePosition::VECIN, 1> vecInWork;
    TQue<QuePosition::VECOUT, 1> vecOutQueueRowSum;
    TBuf<TPosition::VECCALC> sharedVkBuff;

    LocalTensor<int8_t> xTensor;
    LocalTensor<half> xHighHalfTensor;
    LocalTensor<half> xLowHalfTensor;
    LocalTensor<half> xLowHalfTensor2;
    LocalTensor<int4b_t> xHighI4Tensor;
    LocalTensor<int4b_t> xLowI4Tensor;
    LocalTensor<int16_t> xLowI16Tensor;
    LocalTensor<int64_t> groupListTensor;
    LocalTensor<float> groupListFTensor;
    LocalTensor<float> workTensor;
    LocalTensor<float> xHighFloatTensor;
    LocalTensor<float> xRowSumTensor;
    GlobalTensor<int8_t> xGm;
    GlobalTensor<int8_t> yGm;
    GlobalTensor<int64_t> groupListGm;
    GlobalTensor<float> xRowSumGm;
    uint32_t vK;
    uint32_t vKAlign;
    uint32_t totalGroup{0};
    uint32_t blockDim;
    uint32_t coreId;
    uint32_t currentNum;
    uint32_t startNum;
    uint32_t groupNum;
    uint32_t xRowSumRepeatNum;
    uint32_t withOffset;
    half oneEight;
    size_t lenVk;
    size_t lastLenVk;
};

template <class P>
__aicore__ inline void GMMA8W4PreProcess<P>::Init(MMPreInitParams& preInitParams, const GroupMatmulFRTilingData& tilingData, TPipe *pipe){
    xGm.SetGlobalBuffer((__gm__ int8_t *)preInitParams.x);
    yGm.SetGlobalBuffer((__gm__ int8_t *)preInitParams.y);
    groupListGm.SetGlobalBuffer((__gm__ int64_t *)preInitParams.groupList);

    vK = tilingData.k;
    withOffset = tilingData.withOffset;
    lenVk = (vK / 2) / 128;   // align to 128
    lastLenVk = (vK % ROW_SUM_REPEAT_LEN) / 2;
    oneEight = static_cast<half>(ONE_EIGHT);
    pipe->InitBuffer(vecInQueueX, BUFFER_NUM_A8W4_PRE, vK * sizeof(int8_t)); // 2KB
    pipe->InitBuffer(vecOutQueueA1, BUFFER_NUM_A8W4_PRE, vK * sizeof(int4b_t)); // 1KB
    pipe->InitBuffer(vecOutQueueA2, BUFFER_NUM_A8W4_PRE, vK * sizeof(int4b_t)); // 1KB
    pipe->InitBuffer(vecOutQueueA3, BUFFER_NUM_A8W4_PRE, vK * sizeof(half)); // 4 KB

    constexpr int BUFFER_SIZE_256B = 128 * sizeof(int16_t);
    pipe->InitBuffer(vecOutQueue0F, BUFFER_NUM_A8W4_PRE, BUFFER_SIZE_256B); // 256 B
    pipe->InitBuffer(sharedVkBuff, vK * sizeof(float)); // 256 B
    groupNum = static_cast<uint32_t>(tilingData.groupNum);
    pipe->InitBuffer(vecInQueueG, BUFFER_NUM_A8W4_PRE, Ceil(groupNum * sizeof(int64_t), DATA_BLOCK_SIZE_32) * DATA_BLOCK_SIZE_32);
    pipe->InitBuffer(vecInQueueGF, BUFFER_NUM_A8W4_PRE, Ceil(groupNum * sizeof(float), DATA_BLOCK_SIZE_32) * DATA_BLOCK_SIZE_32);
    pipe->InitBuffer(vecInWork, BUFFER_NUM_A8W4_PRE, BUFFER_SIZE_256B * sizeof(float));
    if (withOffset == uint32_t(1)) {
        xRowSumGm.SetGlobalBuffer((__gm__ float *)preInitParams.workspace);
        xRowSumRepeatNum = (vK * sizeof(float) + ROW_SUM_REPEAT_LEN - 1) / ROW_SUM_REPEAT_LEN;
        pipe->InitBuffer(vecOutQueueRowSum, BUFFER_NUM_A8W4_PRE, 1 * sizeof(float));
    }
    startNum = GetBlockIdx();
    blockDim = GetBlockNum() * GetTaskRation();
}

template <class P>
__aicore__ inline void GMMA8W4PreProcess<P>::ProcessPerToken(uint32_t xloop)
{
    uint64_t startAddr = xloop * vK;
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    DataCopy(xTensor, xGm[startAddr], vK);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(xHighHalfTensor, xTensor, AscendC::RoundMode::CAST_NONE, vK);
    PipeBarrier<PIPE_V>();
    if (withOffset == uint32_t(1)) {
        Cast(xHighFloatTensor, xHighHalfTensor, AscendC::RoundMode::CAST_NONE, vK);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        ReduceSum(xRowSumTensor, xHighFloatTensor, workTensor, vK);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);

        DataCopyExtParams xRowSumParams{1, static_cast<uint32_t>(1 * sizeof(float)), 0, 0, 0};
        DataCopyPad(xRowSumGm[xloop], xRowSumTensor, xRowSumParams);
        AscendC::PipeBarrier<PIPE_V>();
    }
    // INT8中高4bit转为int4
    Muls(xHighHalfTensor, xHighHalfTensor, oneEight, vK);
    PipeBarrier<PIPE_V>();
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    Cast(xHighI4Tensor, xHighHalfTensor, AscendC::RoundMode::CAST_FLOOR, vK);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(yGm[startAddr], xHighI4Tensor.ReinterpretCast<int8_t>(), vK / 2);  // 2: size int8 -> int4
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    // INT8中低4bit转为int4
    And(xLowHalfTensor.ReinterpretCast<int16_t>(), xTensor.ReinterpretCast<int16_t>(),
            xLowI16Tensor, LEN_128, lenVk, {1, 1, 1, 8, 8, 0});
    if (lastLenVk > 0) {
        And(xLowHalfTensor[lenVk * LEN_128].ReinterpretCast<int16_t>(), xTensor[lenVk * LEN_128 * TWO].ReinterpretCast<int16_t>(),
                xLowI16Tensor, LEN_128, lastLenVk, {1, 1, 1, 8, 8, 0});
    }
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    Cast(xLowHalfTensor2.ReinterpretCast<half>(), xLowHalfTensor.ReinterpretCast<int8_t>(), AscendC::RoundMode::CAST_NONE, vK);
    PipeBarrier<PIPE_V>();
    // uint4转为int4,需要减8
    const half MINUS_EIGHT = static_cast<half>(-8);     // shrink 0~15 to -8~7
    Adds(xHighHalfTensor, xLowHalfTensor2, MINUS_EIGHT, vK);
    PipeBarrier<PIPE_V>();
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    Cast(xLowI4Tensor, xHighHalfTensor.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, vK);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    DataCopy(yGm[startAddr + vK / TWO], xLowI4Tensor.ReinterpretCast<int8_t>(), vK / TWO);                          
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

template <class P>
__aicore__ inline void GMMA8W4PreProcess<P>::ProcessLoop()
{
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    for(uint32_t xloop = startNum; xloop < totalGroup; xloop += blockDim) {
        ProcessPerToken(xloop);
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    SyncAll<false>();
}

template <class P>
__aicore__ inline void GMMA8W4PreProcess<P>::ComputeTotalGroup(){
    if constexpr (P::groupListType) {
        totalGroup = static_cast<int32_t>(groupListGm.GetValue(groupNum - 1));
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        return;
    }
    groupListTensor = vecInQueueG.AllocTensor<int64_t>();
    groupListFTensor = vecInQueueGF.AllocTensor<float>();
    workTensor = vecInWork.AllocTensor<float>();
    DataCopyParams dataCopyParams{1, static_cast<uint16_t>(groupNum * sizeof(int64_t)), 0, 0};
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyPad(groupListTensor, groupListGm, dataCopyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(groupListFTensor, groupListTensor, AscendC::RoundMode::CAST_ROUND, groupNum);
    PipeBarrier<PIPE_V>();
    ReduceSum(groupListFTensor, groupListFTensor, workTensor, groupNum);
    PipeBarrier<PIPE_V>();
    Cast(groupListTensor, groupListFTensor, AscendC::RoundMode::CAST_ROUND, 1);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    totalGroup = groupListTensor.GetValue(0);
}

template <class P>
__aicore__ inline void GMMA8W4PreProcess<P>::Process()
{
    constexpr int32_t MASK = 128;
    xTensor = vecInQueueX.AllocTensor<int8_t>();
    xHighI4Tensor = vecOutQueueA1.AllocTensor<int4b_t>();
    xLowI4Tensor = vecOutQueueA2.AllocTensor<int4b_t>();
    xHighHalfTensor = vecOutQueueA3.AllocTensor<half>();
    const uint32_t xLowHalfOffset = vK * sizeof(half);

    xLowHalfTensor = sharedVkBuff.GetWithOffset<half>(xLowHalfOffset, 0);
    xLowHalfTensor2 = sharedVkBuff.GetWithOffset<half>(xLowHalfOffset, xLowHalfOffset);
    xLowI16Tensor = vecOutQueue0F.AllocTensor<int16_t>();
    if (withOffset == uint32_t(1)) {
        xHighFloatTensor = sharedVkBuff.GetWithOffset<float>(vK * sizeof(float), 0);
        xRowSumTensor = vecOutQueueRowSum.AllocTensor<float>();
    }
    Duplicate(xLowI16Tensor, static_cast<int16_t>(0x0F0F), MASK);   // get rid of high 4 bits in every int8
    //先计算要处理的group数
    ComputeTotalGroup();
    ProcessLoop();
    vecInQueueX.FreeTensor(xTensor);
    vecOutQueueA1.FreeTensor(xHighI4Tensor);
    vecOutQueueA2.FreeTensor(xLowI4Tensor);
    vecOutQueueA3.FreeTensor(xHighHalfTensor);
    vecOutQueue0F.FreeTensor(xLowI16Tensor);
    if constexpr (!P::groupListType) {
        vecInQueueG.FreeTensor(groupListTensor);
        vecInQueueGF.FreeTensor(groupListFTensor);
        vecInWork.FreeTensor(workTensor);
    }
    if (withOffset == uint32_t(1)) {
        vecOutQueueRowSum.FreeTensor(xRowSumTensor);
    }
}

} // namespace
#endif