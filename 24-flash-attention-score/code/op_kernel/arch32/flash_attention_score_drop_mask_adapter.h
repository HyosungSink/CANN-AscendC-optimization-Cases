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
 * \file flash_attention_score_drop_mask_adapter.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_DROP_MASK_ADAPTER_H
#define FLASH_ATTENTION_SCORE_DROP_MASK_ADAPTER_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

class FlashAttentionScoreDropMaskAdapter {
public:
    __aicore__ inline FlashAttentionScoreDropMaskAdapter()
    {
    }
    __aicore__ inline void Init(__gm__ uint8_t *dropMask, __gm__ uint8_t *workspace,
                                const FlashAttentionScoreGeneralTilingData *__restrict tiling, AscendC::TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn(int64_t offset, int32_t calSize);
    __aicore__ inline void Compute(int32_t calSize, AscendC::LocalTensor<half> &dropMaskSelSrc,
                                   const AscendC::BinaryRepeatParams &binaryRepeatParams);
    __aicore__ inline void CopyOut(int64_t offset, int32_t calSize);
    __aicore__ inline void SyncAllCores();

    template <typename T1, typename T2> __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    };

    template <typename T1, typename T2> __aicore__ inline T1 Min(T1 a, T2 b)
    {
        return (a > b) ? (b) : (a);
    };

    int32_t blockIdx;
    AscendC::TPipe *pipe;
    const FlashAttentionScoreGeneralTilingData *__restrict tilingData;

    AscendC::GlobalTensor<uint8_t> dropMaskGm;
    AscendC::GlobalTensor<uint8_t> outputGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> dropMaskInputQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> dropMaskOutputQueue;
    AscendC::TBuf<> dropMaskSelSrcTBuf;
    AscendC::TBuf<> dropMaskSelResTBuf;

    constexpr static int32_t SELECT_MAX_MASK = 128;
    constexpr static int32_t SELECT_MAX_SRC_STRIDE = 16;
    constexpr static int32_t SELECT_MAX_REPEAT = 255;
    constexpr static int32_t SELECT_MAX_REPEAT_MASK = SELECT_MAX_MASK * SELECT_MAX_REPEAT;
};

__aicore__ inline void
FlashAttentionScoreDropMaskAdapter::Init(__gm__ uint8_t *dropMask, __gm__ uint8_t *workspace,
                                         const FlashAttentionScoreGeneralTilingData *__restrict tiling,
                                         AscendC::TPipe *tPipe)
{
    pipe = tPipe;
    tilingData = tiling;
    blockIdx = AscendC::GetBlockIdx();

    dropMaskGm.SetGlobalBuffer(dropMask);
    outputGm.SetGlobalBuffer(workspace, CeilDiv(tiling->dropmaskParams.shapeTotalSize, 512) * 512);
    pipe->InitBuffer(dropMaskInputQueue, 1, tiling->dropmaskParams.baseUbCalSize / AscendC::ONE_BYTE_BIT_SIZE);
    pipe->InitBuffer(dropMaskOutputQueue, 1, tiling->dropmaskParams.baseUbCalSize);
    pipe->InitBuffer(dropMaskSelSrcTBuf, tiling->dropmaskParams.baseUbCalSize * sizeof(half));
    pipe->InitBuffer(dropMaskSelResTBuf, tiling->dropmaskParams.baseUbCalSize * sizeof(half));
}

__aicore__ inline void FlashAttentionScoreDropMaskAdapter::Process()
{
    // 输入非对齐时，bit[gm] -> select(fp16)[ub] -> bool[ub] -> bool[gm]
    // 单次ub计算量为x个元素，空间占用：x/8 * 1 [1个 uint8]+ 2x * 2[2个fp16,select的src和res] + x * 1 [1个uint8]
    int64_t multiCoreFactorSize = tilingData->dropmaskParams.multiCoreFactorSize; // BN2GS1S2 / 单次UB计算量 / coreNum
    int64_t multiCoreTotalSize = tilingData->dropmaskParams.multiCoreTotalSize; // 绑多核轴总量，BN2GS1S2 / 单次UB计算量
    int64_t coreOffset = multiCoreFactorSize * blockIdx;
    int64_t shapeTotalSize = tilingData->dropmaskParams.shapeTotalSize; // BN2GS1S2
    int64_t singleCoreCalNum = Min(multiCoreFactorSize, multiCoreTotalSize - coreOffset);
    if (singleCoreCalNum <= 0) {
        SyncAllCores();
        return;
    }

    int32_t baseUbCalSize = tilingData->dropmaskParams.baseUbCalSize;
    AscendC::LocalTensor<half> dropMaskSelSrc = dropMaskSelSrcTBuf.template Get<half>();
    AscendC::Duplicate<half>(dropMaskSelSrc, 1.0, baseUbCalSize);

    AscendC::BinaryRepeatParams binaryRepeatParams;
    binaryRepeatParams.src0BlkStride = 1;
    binaryRepeatParams.src0RepStride = 0;
    binaryRepeatParams.src1BlkStride = 1;
    binaryRepeatParams.src1RepStride = 0;
    binaryRepeatParams.dstBlkStride = 1;
    binaryRepeatParams.dstRepStride = 8; // 8: 256B / 32B
    AscendC::PipeBarrier<PIPE_V>();
    for (int64_t loop = 0; loop < singleCoreCalNum; ++loop) {
        int64_t totalOffset = coreOffset + loop;
        int32_t realBaseUbCalSize = baseUbCalSize;
        if (unlikely((totalOffset + 1) * baseUbCalSize > shapeTotalSize)) {
            realBaseUbCalSize = static_cast<int32_t>(shapeTotalSize - totalOffset * baseUbCalSize);
        }

        CopyIn(totalOffset * baseUbCalSize, realBaseUbCalSize);
        Compute(realBaseUbCalSize, dropMaskSelSrc, binaryRepeatParams);
        CopyOut(totalOffset * baseUbCalSize, realBaseUbCalSize);
    }

    SyncAllCores();
}

__aicore__ inline void FlashAttentionScoreDropMaskAdapter::CopyIn(int64_t offset, int32_t calSize)
{
    AscendC::LocalTensor<uint8_t> dropMaskUb = dropMaskInputQueue.template AllocTensor<uint8_t>();
    AscendC::DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.dstStride = 0;
    dataCopyParams.blockLen = CeilDiv(calSize, AscendC::ONE_BYTE_BIT_SIZE);
    dataCopyParams.srcStride = 0;
    AscendC::DataCopyPadParams dataCopyPadParams;
    DataCopyPad(dropMaskUb, dropMaskGm[offset >> 3], dataCopyParams, dataCopyPadParams); // 右移3位表示除8

    dropMaskInputQueue.EnQue(dropMaskUb);
}

__aicore__ inline void FlashAttentionScoreDropMaskAdapter::Compute(int32_t calSize, 
                    AscendC::LocalTensor<half> &dropMaskSelSrc, const AscendC::BinaryRepeatParams &binaryRepeatParams)
{
    AscendC::LocalTensor<half> dropMaskSelRes = dropMaskSelResTBuf.template Get<half>();
    AscendC::LocalTensor<uint8_t> dropMaskUb = dropMaskInputQueue.template DeQue<uint8_t>();
    AscendC::PipeBarrier<PIPE_V>();

    int32_t loop = calSize / SELECT_MAX_REPEAT_MASK;
    for (int32_t idx = 0; idx < loop; ++idx) {
        Select(dropMaskSelRes[SELECT_MAX_MASK * idx], dropMaskUb[SELECT_MAX_SRC_STRIDE * idx], dropMaskSelSrc,
               static_cast<half>(0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, SELECT_MAX_MASK, SELECT_MAX_REPEAT,
               binaryRepeatParams);
    }

    int32_t repeat = CeilDiv(calSize, SELECT_MAX_MASK) - SELECT_MAX_REPEAT * loop;
    Select(dropMaskSelRes[SELECT_MAX_MASK * loop], dropMaskUb[SELECT_MAX_SRC_STRIDE * loop], dropMaskSelSrc,
           static_cast<half>(0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, Min(calSize, SELECT_MAX_MASK), repeat,
           binaryRepeatParams);

    AscendC::PipeBarrier<PIPE_V>();
    dropMaskInputQueue.FreeTensor(dropMaskUb);

    AscendC::LocalTensor<uint8_t> dropMaskOutUb = dropMaskOutputQueue.template AllocTensor<uint8_t>();
    Cast(dropMaskOutUb, dropMaskSelRes, AscendC::RoundMode::CAST_NONE, calSize);
    dropMaskOutputQueue.EnQue(dropMaskOutUb);
}

__aicore__ inline void FlashAttentionScoreDropMaskAdapter::CopyOut(int64_t offset, int32_t calSize)
{
    AscendC::LocalTensor<uint8_t> dropMaskOutUb = dropMaskOutputQueue.template DeQue<uint8_t>();
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = calSize;
    dataCopyParams.dstStride = 0;
    dataCopyParams.srcStride = 0;
    DataCopyPad(outputGm[offset], dropMaskOutUb, dataCopyParams);
    dropMaskOutputQueue.FreeTensor(dropMaskOutUb);
}

__aicore__ inline void FlashAttentionScoreDropMaskAdapter::SyncAllCores()
{
    AscendC::SyncAll();
}

#endif // FLASH_ATTENTION_SCORE_DROP_MASK_ADAPTER_H
