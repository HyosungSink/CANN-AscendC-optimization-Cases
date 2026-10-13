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
 * \file vecOp.h
 * \brief
 */
#ifndef _VEC_OP_H_
#define _VEC_OP_H_

#include "kernel_operator.h"

template <typename IN_TYPE, class TILING_TYPE>
class VecOp {
public:
    __aicore__ inline VecOp(){};
    __aicore__ inline void Init(GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR atten_mask, GM_ADDR actual_seq_qlen,
                                GM_ADDR actual_seq_kvlen, GM_ADDR workspace,
                                const TILING_TYPE *__restrict ordTilingData, TPipe *pipe_in);
    __aicore__ inline void Process(const VecAddrInfo &addrs);

private:
    __aicore__ inline void GetSeqQlenKvlenByBidx(int64_t bIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen);
    __aicore__ inline void CopyInAttenMaskBool(LocalTensor<uint8_t> &dstTensor, int64_t attenMaskOffset,
                                               uint32_t s1Extend, uint32_t s2Extend);
    __aicore__ inline void CalcAttenMaskBool(LocalTensor<float> &dstTensor, LocalTensor<uint8_t> srcTensor,
                                             uint32_t s1Extend, uint32_t s2SrcExtend,
                                             uint32_t s2MaskExtend = BASE_BLOCK_LENGTH, uint8_t maskType = 0);
    __aicore__ inline void CopyInSoftMax(LocalTensor<float> &dstTensor, uint32_t s1Extend, uint32_t softMaxOffset);
    __aicore__ inline void CalcSoftMax(LocalTensor<float> &dstTensor, LocalTensor<float> &src0Tensor,
                                       LocalTensor<float> &src1Tensor, uint32_t s1Extend, uint32_t s2Extend,
                                       uint32_t s2ExtendAlign, const SoftMaxTiling &tiling);
    __aicore__ inline void SubGrapA(int64_t curIdx, const VecBlockInfo &blockInfo, event_t mte2WaitMte3A);
    __aicore__ inline void SubGrapB(int64_t curIdx, const VecBlockInfo &blockInfo, event_t mte2WaitMte3B);

protected:
    TPipe *pipe;
    TBuf<> unifiedBuffer;

    GlobalTensor<uint8_t> attenMaskU8Gm;
    GlobalTensor<float> mm1WorkspaceGm;
    GlobalTensor<float> mm2WorkspaceGm;
    GlobalTensor<IN_TYPE> dropWorkSpaceGm;
    GlobalTensor<IN_TYPE> mulWorkSpaceGm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm;
    GlobalTensor<float> sfmgWorkspaceGm;

    constexpr static uint32_t DTYPE_FACTOR = sizeof(float) / sizeof(IN_TYPE);
    constexpr static uint32_t cal_block_num = 32 / sizeof(float);
    constexpr static uint32_t cal_repeat_num = 256 / sizeof(float);
    constexpr static uint32_t input_block_num = 32 / sizeof(IN_TYPE);
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint32_t INPUT_NUMS = 2;
    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static int64_t C0_SIZE = 16;
    constexpr static int64_t VEC_REPEAT = 8;

    constexpr static uint32_t T2Begin = 0;
    constexpr static uint32_t T1Begin = 33 * 1024;
    constexpr static uint32_t BoolBegin = 50 * 1024;
    constexpr static uint32_t U8Begin = 58 * 1024;
    constexpr static uint32_t T2BlockBegin = 66 * 1024;

    constexpr static uint32_t DbBegin = 74 * 1024;
    constexpr static int64_t TMP_UB_OFFSET = 148 * 1024;
    constexpr static int64_t SFMG_UB_OFFSET = (148 + 33) * 1024;
    constexpr static int64_t TMP_UB_SIZE = 33 * 1024;
    constexpr static int64_t SFMG_UB_SIZE = 8 * 1024;
    constexpr static int64_t TOTAL_SIZE = 189 * 1024;

    constexpr static uint32_t AttenMaskDimS2 = 2048;

    uint32_t blockIdx;
    uint32_t cubeBlockIdx;
    uint32_t subIdx;
    uint32_t sparseMode;

    int32_t taskId = 0;
    int32_t pingpongIdx = 0;
    int32_t blockLen = 0;

    // org shape info
    int64_t n2;
    int64_t g;
    int64_t actualS1Len;
    int64_t actualS2Len;

    bool tndSoftmaxIn;
    float scaleValue;

    int32_t cubeBaseMN;

    int32_t s1VecSize;
    int32_t s2VecSize;
    constexpr static int32_t BASE_BLOCK_LENGTH = 128;
    constexpr static int32_t S1_CUBESIZE = BASE_BLOCK_LENGTH;
    constexpr static int32_t S2_CUBESIZE = BASE_BLOCK_LENGTH;

    int32_t s1Extend;
    int32_t s2Extend;
    int32_t s2ExtendAlign;
    int32_t s1CubeExtend;
    int32_t s2CubeExtend;

    int32_t curS1Idx;
    int32_t curS2Idx;

    // offset
    int32_t sfmgOffset = 0;
    int32_t softMaxOffset = 0;

    int64_t copyInOffset = 0;
    int64_t copyOutOffset = 0;
    DataCopyParams copyInParam;
    DataCopyParams copyOutParam;

    int64_t batch;
    int64_t headNum;
    int64_t headDim;
    int64_t coreNum;
    int64_t mixCoreNum;
    struct CubeAddrInfo cubeAddrInfo[2];
    struct VecAddrInfo vecAddrInfo;
    GM_ADDR actual_seq_qlen_addr;
    GM_ADDR actual_seq_kvlen_addr;
    const TILING_TYPE *__restrict tilingData;
};

template <typename IN_TYPE, class TILING_TYPE>
__aicore__ inline void VecOp<IN_TYPE, TILING_TYPE>::Init(GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR atten_mask,
                                                         GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                                         GM_ADDR workspace, const TILING_TYPE *__restrict ordTilingData,
                                                         TPipe *pipe_in)
{
    // ub分配
    tilingData = ordTilingData;
    pipe = pipe_in;
    blockIdx = GetBlockIdx();
    cubeBlockIdx = blockIdx / 2;
    subIdx = blockIdx % 2;
    curS1Idx = subIdx;
    curS2Idx = 0;
    cubeBaseMN = 16 * 128 * 128;

    batch = tilingData->mlaTensorTilingData.b;
    headNum = tilingData->mlaTensorTilingData.n2 * tilingData->mlaTensorTilingData.g;
    headDim = tilingData->mlaTensorTilingData.d;
    coreNum = tilingData->mlaTensorTilingData.coreNum;
    n2 = tilingData->mlaTensorTilingData.n2;
    g = tilingData->mlaTensorTilingData.g;

    tndSoftmaxIn = tilingData->mlaTensorTilingData.tndSoftmaxIn == 1 ? true : false;
    scaleValue = tilingData->mlaTensorTilingData.scaleValue;
    mixCoreNum = (coreNum + 1) / 2;
    actual_seq_qlen_addr = actual_seq_qlen;
    actual_seq_kvlen_addr = actual_seq_kvlen;
    pipe->InitBuffer(unifiedBuffer, TOTAL_SIZE);

    // global tensor
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmax_max);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmax_sum);
    attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t *)atten_mask);
    mm1WorkspaceGm.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->mlaTensorTilingData.mm1WorkspaceOffset));
    mm2WorkspaceGm.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->mlaTensorTilingData.mm2WorkspaceOffset));
    mulWorkSpaceGm.SetGlobalBuffer((__gm__ IN_TYPE *)(workspace + tilingData->mlaTensorTilingData.mm1WorkspaceOffset));
    dropWorkSpaceGm.SetGlobalBuffer((__gm__ IN_TYPE *)(workspace + tilingData->mlaTensorTilingData.mm2WorkspaceOffset));
    sfmgWorkspaceGm.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->mlaTensorTilingData.sfmgWorkspaceOffset));
}

template <typename IN_TYPE, class TILING_TYPE>
__aicore__ inline void VecOp<IN_TYPE, TILING_TYPE>::Process(const VecAddrInfo &addrs)
{
    taskId = addrs.taskId;
    pingpongIdx = taskId % 2;
    blockLen = addrs.blockLength;
    event_t mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    sparseMode = tilingData->mlaTensorTilingData.sparseMode;
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3));
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3));
    if (taskId == 0 && sparseMode != 0) {
        LocalTensor<uint8_t> attenMaskUbuint8 =
            unifiedBuffer.GetWithOffset<uint8_t>(16 * 1024 / sizeof(uint8_t), BoolBegin);
        CopyInAttenMaskBool(attenMaskUbuint8, 0, BASE_BLOCK_LENGTH, BASE_BLOCK_LENGTH);
    }
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < blockLen; ++i) {
        auto &blockInfo = addrs.VecBlkInfo[i];
        ///////////////////////////////////////////////////////////////
        // do scalar calculate
        ///////////////////////////////////////////////////////////////
        GetSeqQlenKvlenByBidx(blockInfo.batchIdx, actualS1Len, actualS2Len);
        s1CubeExtend = blockInfo.lengthy;
        s2CubeExtend = BASE_BLOCK_LENGTH;
        // split info
        s1VecSize = (s1CubeExtend + 1) / 2;
        s2VecSize = BASE_BLOCK_LENGTH;
        s1Extend = subIdx ? s1CubeExtend - s1VecSize : s1VecSize;
        s2Extend = blockInfo.lengthx;
        s2ExtendAlign = (s2Extend + 15) / 16 * 16;
        // offset
        sfmgOffset = 0;
        if (blockInfo.batchIdx > 0) {
            sfmgOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[blockInfo.batchIdx - 1] * n2 * g * 8;
        }
        sfmgOffset += ((blockInfo.n2Idx * g + blockInfo.gIdx) * actualS1Len + blockInfo.S1Idx * S1_CUBESIZE +
                       curS1Idx * s1VecSize) *
                      8;

        if (tndSoftmaxIn) {
            int64_t innerRowOffsetLeft =
                unlikely(blockInfo.batchIdx == 0) ?
                    0 :
                    ((__gm__ int64_t *)actual_seq_qlen_addr)[blockInfo.batchIdx - 1] * 32 / sizeof(float);
            int64_t originInnerBatchOffset = ((blockInfo.n2Idx * g + blockInfo.gIdx) * actualS1Len +
                                              blockInfo.S1Idx * S1_CUBESIZE + curS1Idx * s1VecSize) *
                                             32 / sizeof(float);
            softMaxOffset = ((((__gm__ int64_t *)actual_seq_qlen_addr)[batch - 1] * 32 / sizeof(float)) *
                                 (blockInfo.n2Idx * g + blockInfo.gIdx) +
                             innerRowOffsetLeft + originInnerBatchOffset % (actualS1Len * 32 / sizeof(float)));
        } else {
            softMaxOffset = sfmgOffset;
        }

        // copyIn cube_workspace params
        copyInOffset = cubeBlockIdx * cubeBaseMN * 2 + pingpongIdx * cubeBaseMN + blockInfo.offset +
                       curS1Idx * s1VecSize * s2CubeExtend;
        copyInParam = {static_cast<uint16_t>(s1Extend), static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                       static_cast<uint16_t>((s2CubeExtend - s2ExtendAlign) * sizeof(float)), 0};
        // copyOut cube_workspace params
        copyOutOffset = 2 * ((cubeBlockIdx * cubeBaseMN * 2 + pingpongIdx * cubeBaseMN + blockInfo.offset) +
                             (curS1Idx * s1VecSize * s2CubeExtend));
        copyOutParam = {static_cast<uint16_t>(s1Extend), static_cast<uint16_t>(s2ExtendAlign * sizeof(IN_TYPE)), 0,
                        static_cast<uint16_t>((s2CubeExtend - s2ExtendAlign) * sizeof(IN_TYPE))};
        ///////////////////////////////////////////////////////////////
        // do vector calculate
        ///////////////////////////////////////////////////////////////
        event_t mte2WaitMte3A = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        event_t mte2WaitMte3B = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        SubGrapA(i, blockInfo, mte2WaitMte3A);
        SubGrapB(i, blockInfo, mte2WaitMte3B);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(mte2WaitMte3A);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(mte2WaitMte3B);
    }
}

template <typename IN_TYPE, class TILING_TYPE>
__aicore__ inline void VecOp<IN_TYPE, TILING_TYPE>::GetSeqQlenKvlenByBidx(int64_t bIdx, int64_t &actualSeqQlen,
                                                                          int64_t &actualSeqKvlen)
{
    if (unlikely(bIdx == 0)) {
        actualSeqQlen = ((__gm__ int64_t *)actual_seq_qlen_addr)[0];
        actualSeqKvlen = ((__gm__ int64_t *)actual_seq_kvlen_addr)[0];
    } else {
        actualSeqQlen =
            ((__gm__ int64_t *)actual_seq_qlen_addr)[bIdx] - ((__gm__ int64_t *)actual_seq_qlen_addr)[bIdx - 1];
        actualSeqKvlen =
            ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIdx] - ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIdx - 1];
    }
    return;
}

template <typename IN_TYPE, class TILING_TYPE>
__aicore__ inline void VecOp<IN_TYPE, TILING_TYPE>::CopyInAttenMaskBool(LocalTensor<uint8_t> &dstTensor,
                                                                        int64_t attenMaskOffset, uint32_t s1Extend,
                                                                        uint32_t s2Extend)
{
    AscendC::DataCopyExtParams intriParams;
    intriParams.blockCount = s1Extend;
    intriParams.blockLen = s2Extend * sizeof(uint8_t);
    intriParams.srcStride = (AttenMaskDimS2 - s2Extend) * sizeof(uint8_t);
    intriParams.dstStride = 0;
    intriParams.rsv = 0;

    DataCopyPad(dstTensor, attenMaskU8Gm[attenMaskOffset], intriParams, {false, 0, 0, 0});
}

template <typename IN_TYPE, class TILING_TYPE>
__aicore__ inline void VecOp<IN_TYPE, TILING_TYPE>::CalcAttenMaskBool(LocalTensor<float> &dstTensor,
                                                                      LocalTensor<uint8_t> srcTensor, uint32_t s1Extend,
                                                                      uint32_t s2SrcExtend, uint32_t s2MaskExtend,
                                                                      uint8_t maskType)
{
    LocalTensor<uint8_t> tmpUbBuffer =
        unifiedBuffer.GetWithOffset<uint8_t>(TMP_UB_SIZE / sizeof(uint8_t), TMP_UB_OFFSET);
    float scalar;
    if constexpr (IsSameType<float, float>::value) {
        uint32_t tmp = 0xFF7FFFFF;
        scalar = *((float *)&tmp);
    } else {
        uint16_t tmp = 0xFBFF;
        scalar = *((IN_TYPE *)&tmp);
    }
    SelectWithBytesMaskShapeInfo info;
    info.firstAxis = s1Extend;
    info.srcLastAxis = s2SrcExtend;
    info.maskLastAxis = s2MaskExtend;
    dstTensor.SetSize(info.firstAxis * info.srcLastAxis);
    srcTensor.SetSize(info.firstAxis * info.maskLastAxis);
    SelectWithBytesMask<float, uint8_t, false>(dstTensor, dstTensor, scalar, srcTensor, tmpUbBuffer, info);
}

template <typename IN_TYPE, class TILING_TYPE>
__aicore__ inline void VecOp<IN_TYPE, TILING_TYPE>::CopyInSoftMax(LocalTensor<float> &dstTensor, uint32_t s1Extend,
                                                                  uint32_t softMaxOffset)
{
    DataCopyPad(dstTensor, softmaxSumGm[softMaxOffset], {1, static_cast<uint16_t>(s1Extend * BLOCK_SIZE), 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(dstTensor[64 * 8], softmaxMaxGm[softMaxOffset], {1, static_cast<uint16_t>(s1Extend * BLOCK_SIZE), 0, 0},
                {false, 0, 0, 0});
}

template <typename IN_TYPE, class TILING_TYPE>
__aicore__ inline void
VecOp<IN_TYPE, TILING_TYPE>::CalcSoftMax(LocalTensor<float> &dstTensor, LocalTensor<float> &src0Tensor,
                                         LocalTensor<float> &src1Tensor, uint32_t s1Extend, uint32_t s2Extend,
                                         uint32_t s2ExtendAlign, const SoftMaxTiling &tiling)
{
    bool isBasicBlock = (s1Extend % 8 == 0) && (s2Extend % 64 == 0);
    if (isBasicBlock) {
        LocalTensor<uint8_t> sharedTmp =
            unifiedBuffer.GetWithOffset<uint8_t>(TMP_UB_SIZE / sizeof(uint8_t), TMP_UB_OFFSET);
        uint32_t shapeArray1[2];
        shapeArray1[0] = s1Extend;
        shapeArray1[1] = s2Extend;
        dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray1, AscendC::DataFormat::ND));
        src0Tensor.SetShapeInfo(ShapeInfo(2, shapeArray1, AscendC::DataFormat::ND));
        SimpleSoftMax<float, false, true>(dstTensor, src1Tensor, src1Tensor[64 * 8], src0Tensor, sharedTmp, tiling);
    } else {
        LocalTensor<float> vecOutBuffer =
            unifiedBuffer.GetWithOffset<float>(TMP_UB_SIZE / sizeof(float), TMP_UB_OFFSET);
        uint32_t sub_block_count = (s2Extend + cal_repeat_num - 1) / cal_repeat_num;
        for (uint32_t subIdx = 0; subIdx < sub_block_count; subIdx++) {
            uint32_t subMaskCount =
                (subIdx == sub_block_count - 1) ? (s2Extend - subIdx * cal_repeat_num) : cal_repeat_num;
            Sub(dstTensor[subIdx * cal_repeat_num], src0Tensor[subIdx * cal_repeat_num], src1Tensor[64 * 8],
                subMaskCount, s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8), 1});
            AscendC::PipeBarrier<PIPE_V>();
            Exp(vecOutBuffer[subIdx * cal_repeat_num], dstTensor[subIdx * cal_repeat_num], subMaskCount, s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8)});
            AscendC::PipeBarrier<PIPE_V>();
            Div(dstTensor[subIdx * cal_repeat_num], vecOutBuffer[subIdx * cal_repeat_num], src1Tensor, subMaskCount,
                s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8), 1});
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

template <typename IN_TYPE, class TILING_TYPE>
__aicore__ inline void VecOp<IN_TYPE, TILING_TYPE>::SubGrapA(int64_t curIdx, const VecBlockInfo &blockInfo,
                                                             event_t mte2WaitMte3A)
{
    uint32_t ubBufferOffset = 0;
    if (curIdx > 0) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3A));
    }
    LocalTensor<float> vecInBuffer3 =
        unifiedBuffer.GetWithOffset<float>(8 * 1024 / sizeof(float), ubBufferOffset + T2BlockBegin);

    CopyInSoftMax(vecInBuffer3, s1Extend, softMaxOffset);
    LocalTensor<float> vecClc2Buffer =
        unifiedBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), ubBufferOffset + T2Begin);
    DataCopyPad(vecClc2Buffer, mm2WorkspaceGm[copyInOffset], copyInParam, {false, 0, 0, 0});
    event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    AscendC::SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    AscendC::WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    AscendC::PipeBarrier<PIPE_V>();
    Muls(vecClc2Buffer, vecClc2Buffer, scaleValue, s1Extend * s2ExtendAlign);

    AscendC::PipeBarrier<PIPE_V>();
    if (sparseMode != 0) {
        LocalTensor<uint8_t> attenMaskUbuint8 =
            unifiedBuffer.GetWithOffset<uint8_t>(16 * 1024 / sizeof(uint8_t), ubBufferOffset + BoolBegin);
        if (blockInfo.S1Idx == blockInfo.S2Idx) {
            CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8[curS1Idx * s1VecSize * BASE_BLOCK_LENGTH], s1Extend,
                              s2ExtendAlign, S2_CUBESIZE, 0);
        }
    }
    ///////////////////////////////////////////////////////////////
    // simpleSoftMax
    ///////////////////////////////////////////////////////////////
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<float> simpleSoftmaxResBuf = unifiedBuffer.GetWithOffset<float>(33 * 1024 / sizeof(float), DbBegin);
    CalcSoftMax(simpleSoftmaxResBuf, vecClc2Buffer, vecInBuffer3, s1Extend, s2Extend, s2ExtendAlign,
                tilingData->mlaTensorTilingData.softmaxTilingData);
    LocalTensor<float> vecDropBuffer = simpleSoftmaxResBuf;
    ///////////////////////////////////////////////////////////////
    // cast fp322bf16
    ///////////////////////////////////////////////////////////////
    LocalTensor<IN_TYPE> vecCopyOutBuffer =
        unifiedBuffer.GetWithOffset<IN_TYPE>(17 * 1024 / sizeof(IN_TYPE), ubBufferOffset + T1Begin);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(vecCopyOutBuffer, vecDropBuffer, RoundMode::CAST_ROUND, s1Extend * s2ExtendAlign);
    event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    AscendC::SetFlag<HardEvent::V_MTE3>(static_cast<int32_t>(mte3WaitV));
    AscendC::WaitFlag<HardEvent::V_MTE3>(static_cast<int32_t>(mte3WaitV));
    DataCopyPad(dropWorkSpaceGm[copyOutOffset], vecCopyOutBuffer, copyOutParam);
    if (curIdx < blockLen - 1) {
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3A));
    }
}

template <typename IN_TYPE, class TILING_TYPE>
__aicore__ inline void VecOp<IN_TYPE, TILING_TYPE>::SubGrapB(int64_t curIdx, const VecBlockInfo &blockInfo,
                                                             event_t mte2WaitMte3B)
{
    uint32_t ubBufferOffset = DbBegin;
    if (curIdx > 0) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3B));
    }
    // copyIn sfmg
    LocalTensor<float> sfmgClc3 = unifiedBuffer.GetWithOffset<float>(SFMG_UB_SIZE / sizeof(float), SFMG_UB_OFFSET);
    DataCopy(sfmgClc3, sfmgWorkspaceGm[sfmgOffset], s1Extend * 8);
    LocalTensor<float> vecClc1Buffer =
        unifiedBuffer.GetWithOffset<float>(33 * 1024 / sizeof(float), ubBufferOffset + T1Begin);

    // copyIn cube result
    DataCopyPad(vecClc1Buffer, mm1WorkspaceGm[copyInOffset], copyInParam, {false, 0, 0, 0});
    event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    AscendC::SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    AscendC::WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    ///////////////////////////////////////////////////////////////
    // sub
    ///////////////////////////////////////////////////////////////
    uint32_t sub_block_cout = (s2ExtendAlign + cal_repeat_num - 1) / cal_repeat_num;
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t subIdx = 0; subIdx < sub_block_cout; subIdx++) {
        uint32_t subMaskCout =
            (subIdx == sub_block_cout - 1) ? (s2ExtendAlign - subIdx * cal_repeat_num) : cal_repeat_num;
        Sub(vecClc1Buffer[subIdx * cal_repeat_num], vecClc1Buffer[subIdx * cal_repeat_num], sfmgClc3, subMaskCout,
            s1Extend,
            {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
             static_cast<uint8_t>(s2ExtendAlign / 8), 1});
    }
    ///////////////////////////////////////////////////////////////
    // mul
    ///////////////////////////////////////////////////////////////
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<float> simpleSoftmaxResBuf = unifiedBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), DbBegin);
    Mul(vecClc1Buffer, vecClc1Buffer, simpleSoftmaxResBuf, s1Extend * s2ExtendAlign);
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<IN_TYPE> vecCopyOutBuffer =
        unifiedBuffer.GetWithOffset<IN_TYPE>(17 * 1024 / sizeof(IN_TYPE), ubBufferOffset + T1Begin);
    Cast(vecCopyOutBuffer, vecClc1Buffer, RoundMode::CAST_ROUND, s1Extend * s2ExtendAlign);
    event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    AscendC::SetFlag<HardEvent::V_MTE3>(static_cast<int32_t>(mte3WaitV));
    AscendC::WaitFlag<HardEvent::V_MTE3>(static_cast<int32_t>(mte3WaitV));
    // dyv = dp -> ds
    DataCopyPad(mulWorkSpaceGm[copyOutOffset], vecCopyOutBuffer, copyOutParam);
    if (curIdx < blockLen - 1) {
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3B));
    }
}
#endif
