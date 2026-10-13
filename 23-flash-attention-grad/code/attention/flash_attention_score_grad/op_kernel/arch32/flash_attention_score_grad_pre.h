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
 * \file flash_attention_score_grad_pre.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_PRE_KERNEL_H_
#define FLASH_ATTENTION_SCORE_GRAD_PRE_KERNEL_H_

#include "kernel_operator.h"

template <typename T1, typename T2, class TILING_TYPE, const bool INIT_OUTPUT = true, const uint32_t HAS_ROPE = 0>
class FlashAttentionScoreGradPre {
public:
    __aicore__ inline FlashAttentionScoreGradPre(){};
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dqRope, __gm__ uint8_t *dk, __gm__ uint8_t *dkRope, __gm__ uint8_t *dv, __gm__ uint8_t *drop_mask,
                                __gm__ uint8_t *workspace, const TILING_TYPE *ordTilingData, TPipe *pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void SyncALLCores();

    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> helpQue;
    TQue<QuePosition::VECIN, 1> inputQue;
    TQue<QuePosition::VECIN, 1> castQue;
    TQue<QuePosition::VECOUT, 1> outQue;

    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, dsinksumWorkSpaceGm, dvGm;
    GlobalTensor<uint8_t> maskWorkSpaceGm;
    GlobalTensor<uint8_t> drop_maskGm;
    GlobalTensor<float> dqRopeWorkSpaceGm;
    GlobalTensor<float> dkRopeWorkSpaceGm;

    const TILING_TYPE *TilingData;
    constexpr static uint32_t HELP_LEN = 256;
    constexpr static uint32_t BIT8 = 8;
    constexpr static uint32_t NUMBER_8 = 8;
    constexpr static uint32_t B16_VECTOR_MASK = 128;
    constexpr static uint32_t ENABLE = 1;

    uint32_t cBlockIdx;
    // query
    uint32_t ubBaseSize;
    uint32_t qPreBlockFactor;
    uint32_t qPreBlockTotal;
    uint32_t qPreBlockTail;
    uint32_t qRopePreBlockFactor;
    uint32_t qRopePreBlockTotal;
    uint32_t qRopePreBlockTail;
    uint32_t kvPreBlockFactor;
    uint32_t kvPreBlockTotal;
    uint32_t kvPreBlockTail;
    uint32_t kRopePreBlockFactor;
    uint32_t kRopePreBlockTotal;
    uint32_t kRopePreBlockTail;

    int64_t qSizeAlign;
    int64_t kvSizeAlign;
    int64_t qRopeSizeAlign;
    int64_t kRopeSizeAlign;

    int64_t initdqSize;
    int64_t dqOffset;
    int64_t initdkSize;
    int64_t dkvOffset;
    int64_t initdqRopeSize;
    int64_t dqRopeOffset;
    int64_t initdkRopeSize;
    int64_t dkRopeOffset;
    bool isDropBoolMode;
    uint32_t maskUsedCoreNum;
    uint32_t maskUBProcessNum;
    uint32_t maskTailUBProcessNum;
    uint32_t maskUBLoop;

    DataCopyParams copyParams;
    BinaryRepeatParams repParams;
    half padValue{1.0};
};

template <typename T1, typename T2, class TILING_TYPE, const bool INIT_OUTPUT, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradPre<T1, T2, TILING_TYPE, INIT_OUTPUT, HAS_ROPE>::Init(
    __gm__ uint8_t *dq, __gm__ uint8_t *dqRope, __gm__ uint8_t *dk, __gm__ uint8_t *dkRope, __gm__ uint8_t *dv, __gm__ uint8_t *drop_mask, __gm__ uint8_t *workspace,
    const TILING_TYPE *ordTilingData, TPipe *pipe_in)
{
    cBlockIdx = GetBlockIdx();

    TilingData = ordTilingData;
    pipe = pipe_in;

    maskUsedCoreNum = TilingData->preTilingData.maskCoreNum;

    drop_maskGm.SetGlobalBuffer((__gm__ uint8_t *)drop_mask);
    dvGm.SetGlobalBuffer((__gm__ float *)dv);

    if constexpr (INIT_OUTPUT) {
        // tiling_data
        qPreBlockFactor = TilingData->preTilingData.qPreBlockFactor;
        qPreBlockTotal = TilingData->preTilingData.qPreBlockTotal;
        qPreBlockTail = TilingData->preTilingData.qPreBlockTail;
        qSizeAlign = TilingData->postTilingData.qSizeAlign;
        kvPreBlockFactor = TilingData->preTilingData.kvPreBlockFactor;
        kvPreBlockTotal = TilingData->preTilingData.kvPreBlockTotal;
        kvPreBlockTail = TilingData->preTilingData.kvPreBlockTail;
        kvSizeAlign = TilingData->postTilingData.kvSizeAlign;
        if constexpr (HAS_ROPE == ENABLE) {
            qRopePreBlockFactor = TilingData->preTilingData.qRopePreBlockFactor;
            qRopePreBlockTotal = TilingData->preTilingData.qRopePreBlockTotal;
            qRopePreBlockTail = TilingData->preTilingData.qRopePreBlockTail;
            qRopeSizeAlign = TilingData->postTilingData.qRopeSizeAlign;
            kRopePreBlockFactor = TilingData->preTilingData.kRopePreBlockFactor;
            kRopePreBlockTotal = TilingData->preTilingData.kRopePreBlockTotal;
            kRopePreBlockTail = TilingData->preTilingData.kRopePreBlockTail;
            kRopeSizeAlign = TilingData->postTilingData.kRopeSizeAlign;
        }
        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  TilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
        if constexpr (HAS_ROPE == ENABLE) {
            dqRopeWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                    TilingData->postTilingData.dqRopeWorkSpaceOffset / sizeof(float));
        }
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  TilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
        if constexpr (HAS_ROPE == ENABLE) {
            dkRopeWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                    TilingData->postTilingData.dkRopeWorkSpaceOffset / sizeof(float));
        }
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  TilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));
        dsinksumWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                TilingData->postTilingData.dsinksumWorkSpaceOffset / sizeof(float));

        initdqSize = cBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
        dqOffset = ((int64_t)cBlockIdx) * qPreBlockFactor;
        initdkSize = cBlockIdx == kvPreBlockTotal - 1 ? kvPreBlockTail : kvPreBlockFactor;
        dkvOffset = ((int64_t)cBlockIdx) * kvPreBlockFactor;
        initdqRopeSize = cBlockIdx == qRopePreBlockTotal - 1 ? qRopePreBlockTail : qRopePreBlockFactor;
        dqRopeOffset = ((int64_t)cBlockIdx) * qRopePreBlockFactor;
        initdkRopeSize = cBlockIdx == kRopePreBlockTotal - 1 ? kRopePreBlockTail : kRopePreBlockFactor;
        dkRopeOffset = ((int64_t)cBlockIdx) * kRopePreBlockFactor;
    }

    // dropMask params init
    isDropBoolMode = TilingData->preTilingData.dropoutIsDivisibleBy8 == 0 ? true : false;

    if (isDropBoolMode) {
        maskWorkSpaceGm.SetGlobalBuffer((__gm__ uint8_t *)workspace + TilingData->preTilingData.dropBeginAddr);

        pipe->InitBuffer(helpQue, 1, HELP_LEN);
        pipe->InitBuffer(inputQue, 1, TilingData->preTilingData.inputBufferLen);
        pipe->InitBuffer(castQue, 1, TilingData->preTilingData.castBufferLen);
        pipe->InitBuffer(outQue, 1, TilingData->preTilingData.outputBufferLen);

        // reset params
        repParams.src0BlkStride = 1;
        repParams.src0RepStride = 0;
        repParams.src1BlkStride = 0;
        repParams.src1RepStride = 0;
        repParams.dstBlkStride = 1;
        repParams.dstRepStride = NUMBER_8;

        copyParams.blockCount = 1;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
    }
}


template <typename T1, typename T2, class TILING_TYPE, const bool INIT_OUTPUT, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradPre<T1, T2, TILING_TYPE, INIT_OUTPUT, HAS_ROPE>::Process()
{
    // process clear dq dk dv workspace
    if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.qPreBlockTotal) {
        if constexpr (INIT_OUTPUT) {
            InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
        }
    }

    if constexpr (HAS_ROPE == ENABLE) {
        if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.qRopePreBlockTotal) {
            if constexpr (INIT_OUTPUT) {
                InitOutput<float>(dqRopeWorkSpaceGm[dqRopeOffset], initdqRopeSize, 0);
            }
        }
    }

    if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.kvPreBlockTotal) {
        if constexpr (INIT_OUTPUT) {
            InitOutput<float>(dkWorkSpaceGm[dkvOffset], initdkSize, 0);
            if constexpr (IsSameType<T1, float>::value) {
                InitOutput<float>(dvGm[dkvOffset], initdkSize, 0);
            } else {
                InitOutput<float>(dvWorkSpaceGm[dkvOffset], initdkSize, 0);
            }
        }
    }

    if constexpr (HAS_ROPE == ENABLE) {
        if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.kRopePreBlockTotal) {
            if constexpr (INIT_OUTPUT) {
                InitOutput<float>(dkRopeWorkSpaceGm[dkRopeOffset], initdkRopeSize, 0);
            }
        }
    }

    if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.maskCoreNum) {
        if (!isDropBoolMode) {
            return;
        }
        maskUBLoop = TilingData->preTilingData.maskSingleCoreLoop;
        maskTailUBProcessNum = TilingData->preTilingData.maskLastLoopNum;
        if (unlikely(cBlockIdx == maskUsedCoreNum - 1)) {
            maskUBLoop = TilingData->preTilingData.maskTailCoreLoop;
            maskTailUBProcessNum = TilingData->preTilingData.maskTailCoreLastLoopNum;
        }

        // malloc tensor filled by 1.0
        auto helpTensor = helpQue.AllocTensor<half>();
        Duplicate<half>(helpTensor, padValue, HELP_LEN / sizeof(half));
        AscendC::PipeBarrier<PIPE_V>();

        int64_t outputAddr = cBlockIdx * TilingData->preTilingData.maskSingleCoreNum;
        int64_t inputAddr = cBlockIdx * TilingData->preTilingData.maskSingleCoreNum / BIT8;

        // process
        for (int64_t idx = 0; idx < maskUBLoop; idx++) {
            maskUBProcessNum = TilingData->preTilingData.singleUBProcessNum;
            int64_t outputOffset = idx * maskUBProcessNum;
            int64_t inputOffset = idx * maskUBProcessNum / BIT8;
            if (unlikely(idx == maskUBLoop - 1)) {
                maskUBProcessNum = maskTailUBProcessNum;
            }

            // copyIn
            auto inputTensor = inputQue.AllocTensor<uint8_t>();
            copyParams.blockLen = maskUBProcessNum / BIT8;
            DataCopyPad(inputTensor, drop_maskGm[inputAddr + inputOffset], copyParams, {false, 0, 0, 0});
            inputQue.EnQue(inputTensor);
            inputQue.DeQue<uint8_t>();

            // select
            auto castTensor = castQue.AllocTensor<half>();
            uint8_t selectRepeat = (maskUBProcessNum + B16_VECTOR_MASK - 1) / B16_VECTOR_MASK;
            Select(castTensor, inputTensor, helpTensor, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, B16_VECTOR_MASK,
                   selectRepeat, repParams);
            AscendC::PipeBarrier<PIPE_V>();
            inputQue.FreeTensor(inputTensor);

            // cast
            auto outputTensor = outQue.AllocTensor<uint8_t>();
            Cast(outputTensor, castTensor, RoundMode::CAST_ROUND, maskUBProcessNum);
            castQue.FreeTensor(castTensor);

            // copyOut
            outQue.EnQue(outputTensor);
            outQue.DeQue<uint8_t>();
            copyParams.blockLen = maskUBProcessNum;
            DataCopyPad(maskWorkSpaceGm[outputAddr + outputOffset], outputTensor, copyParams);
            outQue.FreeTensor(outputTensor);
        }
        helpQue.FreeTensor(helpTensor);
    }
}

template <typename T1, typename T2, class TILING_TYPE, const bool INIT_OUTPUT, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradPre<T1, T2, TILING_TYPE, INIT_OUTPUT, HAS_ROPE>::SyncALLCores()
{
    SyncAll();
}

// Partial Specialize for FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb
template <typename T1, typename T2, const bool INIT_OUTPUT, const uint32_t HAS_ROPE>
class FlashAttentionScoreGradPre<T1, T2, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, INIT_OUTPUT, HAS_ROPE> {
public:
    __aicore__ inline FlashAttentionScoreGradPre(){}
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dqRope, __gm__ uint8_t *dk, __gm__ uint8_t *dkRope, __gm__ uint8_t *dv, __gm__ uint8_t *drop_mask,
                                __gm__ uint8_t *workspace, const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb *ordTilingData, TPipe *pipe_in)
    {
        cBlockIdx = GetBlockIdx();

        TilingData = ordTilingData;
        pipe = pipe_in;

        maskUsedCoreNum = TilingData->preTilingData.maskCoreNum;

        drop_maskGm.SetGlobalBuffer((__gm__ uint8_t *)drop_mask);
        dvGm.SetGlobalBuffer((__gm__ float *)dv);

        if constexpr (INIT_OUTPUT) {
            // tiling_data
            qPreBlockFactor = TilingData->preTilingData.qPreBlockFactor;
            qPreBlockTotal = TilingData->preTilingData.qPreBlockTotal;
            qPreBlockTail = TilingData->preTilingData.qPreBlockTail;
            qSizeAlign = TilingData->postTilingData.qSizeAlign;
            kvPreBlockFactor = TilingData->preTilingData.kvPreBlockFactor;
            kvPreBlockTotal = TilingData->preTilingData.kvPreBlockTotal;
            kvPreBlockTail = TilingData->preTilingData.kvPreBlockTail;
            kvSizeAlign = TilingData->postTilingData.kvSizeAlign;

            if constexpr (HAS_ROPE == ENABLE) {
                qRopePreBlockFactor = TilingData->preTilingData.qRopePreBlockFactor;
                qRopePreBlockTotal = TilingData->preTilingData.qRopePreBlockTotal;
                qRopePreBlockTail = TilingData->preTilingData.qRopePreBlockTail;
                qRopeSizeAlign = TilingData->postTilingData.qRopeSizeAlign;
                kRopePreBlockFactor = TilingData->preTilingData.kRopePreBlockFactor;
                kRopePreBlockTotal = TilingData->preTilingData.kRopePreBlockTotal;
                kRopePreBlockTail = TilingData->preTilingData.kRopePreBlockTail;
                kRopeSizeAlign = TilingData->postTilingData.kRopeSizeAlign;
            }

            vPreBlockFactor = TilingData->preTilingData.vPreBlockFactor;
            vPreBlockTotal = TilingData->preTilingData.vPreBlockTotal;
            vPreBlockTail = TilingData->preTilingData.vPreBlockTail;
            vSizeAlign = TilingData->postTilingData.vSizeAlign;

            dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                    TilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
            if constexpr (HAS_ROPE == ENABLE) {
                dqRopeWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                        TilingData->postTilingData.dqRopeWorkSpaceOffset / sizeof(float));
            }
            dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                    TilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
            if constexpr (HAS_ROPE == ENABLE) {
                dkRopeWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                        TilingData->postTilingData.dkRopeWorkSpaceOffset / sizeof(float));
            }
            dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                    TilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));
            dsinksumWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                TilingData->postTilingData.dsinksumWorkSpaceOffset / sizeof(float));

            initdqRopeSize = cBlockIdx == qRopePreBlockTotal - 1 ? qRopePreBlockTail : qRopePreBlockFactor;
            dqRopeOffset = ((int64_t)cBlockIdx) * qRopePreBlockFactor;
            initdkRopeSize = cBlockIdx == kRopePreBlockTotal - 1 ? kRopePreBlockTail : kRopePreBlockFactor;
            dkRopeOffset = ((int64_t)cBlockIdx) * kRopePreBlockFactor;
                                                
            initdqSize = cBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
            dqOffset = ((int64_t)cBlockIdx) * qPreBlockFactor;
            initdkSize = cBlockIdx == kvPreBlockTotal - 1 ? kvPreBlockTail : kvPreBlockFactor;
            dkvOffset = ((int64_t)cBlockIdx) * kvPreBlockFactor;
            initdvSize = cBlockIdx == vPreBlockTotal - 1 ? vPreBlockTail : vPreBlockFactor;
            dvOffset = ((int64_t)cBlockIdx) * vPreBlockFactor;
        }

        // dropMask params init
        isDropBoolMode = TilingData->preTilingData.dropoutIsDivisibleBy8 == 0 ? true : false;

        if (isDropBoolMode) {
            maskWorkSpaceGm.SetGlobalBuffer((__gm__ uint8_t *)workspace + TilingData->preTilingData.dropBeginAddr);

            pipe->InitBuffer(helpQue, 1, HELP_LEN);
            pipe->InitBuffer(inputQue, 1, TilingData->preTilingData.inputBufferLen);
            pipe->InitBuffer(castQue, 1, TilingData->preTilingData.castBufferLen);
            pipe->InitBuffer(outQue, 1, TilingData->preTilingData.outputBufferLen);

            // reset params
            repParams.src0BlkStride = 1;
            repParams.src0RepStride = 0;
            repParams.src1BlkStride = 0;
            repParams.src1RepStride = 0;
            repParams.dstBlkStride = 1;
            repParams.dstRepStride = NUMBER_8;

            copyParams.blockCount = 1;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
        }
    }
    __aicore__ inline void Process()
    {
        // process clear dq dk dv workspace
        if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.qPreBlockTotal) {
            if constexpr (INIT_OUTPUT) {
                InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
            }
        }

        if constexpr (HAS_ROPE == ENABLE) {
            if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.qRopePreBlockTotal) {
                if constexpr (INIT_OUTPUT) {
                    InitOutput<float>(dqRopeWorkSpaceGm[dqRopeOffset], initdqRopeSize, 0);
                }
            }
        }

        if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.kvPreBlockTotal) {
            if constexpr (INIT_OUTPUT) {
                InitOutput<float>(dkWorkSpaceGm[dkvOffset], initdkSize, 0);

            }
        }

        if constexpr (HAS_ROPE == ENABLE) {
            if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.kRopePreBlockTotal) {
                if constexpr (INIT_OUTPUT) {
                    InitOutput<float>(dkRopeWorkSpaceGm[dkRopeOffset], initdkRopeSize, 0);
                }
            }
        }

        if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.vPreBlockTotal) {
            if constexpr (INIT_OUTPUT) {
                if constexpr (IsSameType<T1, float>::value) {
                    InitOutput<float>(dvGm[dvOffset], initdvSize, 0);
                } else {
                    InitOutput<float>(dvWorkSpaceGm[dvOffset], initdvSize, 0);
                }
            }
        }

        if (unlikely(TilingData->s1s2BNGS1S2BaseParams.sink == 1)) {
            if (g_coreType == AIV && cBlockIdx == 0) {
                int s1Pad = (TilingData->postTilingData.s1 + 255) / 256 * 256;
                int s2Pad = (TilingData->postTilingData.s2 + 255) / 256 * 256; 
                size_t dsinksumSize = TilingData->postTilingData.b * TilingData->postTilingData.n2 * TilingData->postTilingData.g * s1Pad * s2Pad / TilingData->postTilingData.baseMN;
                InitOutput<float>(dsinksumWorkSpaceGm, dsinksumSize, 0);
            }
        }

        if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.maskCoreNum) {
            if (!isDropBoolMode) {
                return;
            }
            maskUBLoop = TilingData->preTilingData.maskSingleCoreLoop;
            maskTailUBProcessNum = TilingData->preTilingData.maskLastLoopNum;
            if (unlikely(cBlockIdx == maskUsedCoreNum - 1)) {
                maskUBLoop = TilingData->preTilingData.maskTailCoreLoop;
                maskTailUBProcessNum = TilingData->preTilingData.maskTailCoreLastLoopNum;
            }

            // malloc tensor filled by 1.0
            auto helpTensor = helpQue.AllocTensor<half>();
            Duplicate<half>(helpTensor, padValue, HELP_LEN / sizeof(half));
            AscendC::PipeBarrier<PIPE_V>();

            int64_t outputAddr = cBlockIdx * TilingData->preTilingData.maskSingleCoreNum;
            int64_t inputAddr = cBlockIdx * TilingData->preTilingData.maskSingleCoreNum / BIT8;

            // process
            for (int64_t idx = 0; idx < maskUBLoop; idx++) {
                maskUBProcessNum = TilingData->preTilingData.singleUBProcessNum;
                int64_t outputOffset = idx * maskUBProcessNum;
                int64_t inputOffset = idx * maskUBProcessNum / BIT8;
                if (unlikely(idx == maskUBLoop - 1)) {
                    maskUBProcessNum = maskTailUBProcessNum;
                }

                // copyIn
                auto inputTensor = inputQue.AllocTensor<uint8_t>();
                copyParams.blockLen = maskUBProcessNum / BIT8;
                DataCopyPad(inputTensor, drop_maskGm[inputAddr + inputOffset], copyParams, {false, 0, 0, 0});
                inputQue.EnQue(inputTensor);
                inputQue.DeQue<uint8_t>();

                // select
                auto castTensor = castQue.AllocTensor<half>();
                uint8_t selectRepeat = (maskUBProcessNum + B16_VECTOR_MASK - 1) / B16_VECTOR_MASK;
                Select(castTensor, inputTensor, helpTensor, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, B16_VECTOR_MASK,
                    selectRepeat, repParams);
                AscendC::PipeBarrier<PIPE_V>();
                inputQue.FreeTensor(inputTensor);

                // cast
                auto outputTensor = outQue.AllocTensor<uint8_t>();
                Cast(outputTensor, castTensor, RoundMode::CAST_ROUND, maskUBProcessNum);
                castQue.FreeTensor(castTensor);

                // copyOut
                outQue.EnQue(outputTensor);
                outQue.DeQue<uint8_t>();
                copyParams.blockLen = maskUBProcessNum;
                DataCopyPad(maskWorkSpaceGm[outputAddr + outputOffset], outputTensor, copyParams);
                outQue.FreeTensor(outputTensor);
            }
            helpQue.FreeTensor(helpTensor);
        }
    }
    __aicore__ inline void SyncALLCores()
    {
        SyncAll();
    }

    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> helpQue;
    TQue<QuePosition::VECIN, 1> inputQue;
    TQue<QuePosition::VECIN, 1> castQue;
    TQue<QuePosition::VECOUT, 1> outQue;

    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, dsinksumWorkSpaceGm, dvGm;
    GlobalTensor<uint8_t> maskWorkSpaceGm;
    GlobalTensor<uint8_t> drop_maskGm;
    GlobalTensor<float> dqRopeWorkSpaceGm, dkRopeWorkSpaceGm;

    const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb *TilingData;
    constexpr static uint32_t HELP_LEN = 256;
    constexpr static uint32_t BIT8 = 8;
    constexpr static uint32_t NUMBER_8 = 8;
    constexpr static uint32_t B16_VECTOR_MASK = 128;
    constexpr static uint32_t ENABLE = 1;

    uint32_t cBlockIdx;
    // query
    uint32_t ubBaseSize;
    uint32_t qPreBlockFactor;
    uint32_t qPreBlockTotal;
    uint32_t qPreBlockTail;
    uint32_t qRopePreBlockFactor;
    uint32_t qRopePreBlockTotal;
    uint32_t qRopePreBlockTail;
    uint32_t kvPreBlockFactor;
    uint32_t kvPreBlockTotal;
    uint32_t kvPreBlockTail;
    uint32_t kRopePreBlockFactor;
    uint32_t kRopePreBlockTotal;
    uint32_t kRopePreBlockTail;

    uint32_t vPreBlockFactor;
    uint32_t vPreBlockTotal;
    uint32_t vPreBlockTail;

    int64_t qSizeAlign;
    int64_t kvSizeAlign;
    int64_t qRopeSizeAlign;
    int64_t kRopeSizeAlign;
    int64_t vSizeAlign;

    int64_t initdqSize;
    int64_t dqOffset;
    int64_t initdkSize;
    int64_t dkvOffset;
    int64_t initdqRopeSize;
    int64_t dqRopeOffset;
    int64_t initdkRopeSize;
    int64_t dkRopeOffset;
    int64_t initdvSize;
    int64_t dvOffset;

    bool isDropBoolMode;
    uint32_t maskUsedCoreNum;
    uint32_t maskUBProcessNum;
    uint32_t maskTailUBProcessNum;
    uint32_t maskUBLoop;

    DataCopyParams copyParams;
    BinaryRepeatParams repParams;
    half padValue{1.0};
};

// // EXACT the same with FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb except for class name.
// // Partial Specialize for FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2
template <typename T1, typename T2, const bool INIT_OUTPUT, const uint32_t HAS_ROPE>
class FlashAttentionScoreGradPre<T1, T2, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, INIT_OUTPUT, HAS_ROPE> {
public:
    __aicore__ inline FlashAttentionScoreGradPre(){}
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dqRope, __gm__ uint8_t *dk, __gm__ uint8_t *dkRope, __gm__ uint8_t *dv, __gm__ uint8_t *drop_mask,
                                __gm__ uint8_t *workspace, const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *ordTilingData, TPipe *pipe_in)
    {
        cBlockIdx = GetBlockIdx();

        TilingData = ordTilingData;
        pipe = pipe_in;

        maskUsedCoreNum = TilingData->preTilingData.maskCoreNum;

        drop_maskGm.SetGlobalBuffer((__gm__ uint8_t *)drop_mask);
        dvGm.SetGlobalBuffer((__gm__ float *)dv);

        if constexpr (INIT_OUTPUT) {
            // tiling_data
            qPreBlockFactor = TilingData->preTilingData.qPreBlockFactor;
            qPreBlockTotal = TilingData->preTilingData.qPreBlockTotal;
            qPreBlockTail = TilingData->preTilingData.qPreBlockTail;
            qSizeAlign = TilingData->postTilingData.qSizeAlign;
            kvPreBlockFactor = TilingData->preTilingData.kvPreBlockFactor;
            kvPreBlockTotal = TilingData->preTilingData.kvPreBlockTotal;
            kvPreBlockTail = TilingData->preTilingData.kvPreBlockTail;
            kvSizeAlign = TilingData->postTilingData.kvSizeAlign;

            if constexpr (HAS_ROPE == ENABLE) {
                qRopePreBlockFactor = TilingData->preTilingData.qRopePreBlockFactor;
                qRopePreBlockTotal = TilingData->preTilingData.qRopePreBlockTotal;
                qRopePreBlockTail = TilingData->preTilingData.qRopePreBlockTail;
                qRopeSizeAlign = TilingData->postTilingData.qRopeSizeAlign;
                kRopePreBlockFactor = TilingData->preTilingData.kRopePreBlockFactor;
                kRopePreBlockTotal = TilingData->preTilingData.kRopePreBlockTotal;
                kRopePreBlockTail = TilingData->preTilingData.kRopePreBlockTail;
                kRopeSizeAlign = TilingData->postTilingData.kRopeSizeAlign;
            }

            vPreBlockFactor = TilingData->preTilingData.vPreBlockFactor;
            vPreBlockTotal = TilingData->preTilingData.vPreBlockTotal;
            vPreBlockTail = TilingData->preTilingData.vPreBlockTail;
            vSizeAlign = TilingData->postTilingData.vSizeAlign;

            dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                    TilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
            if constexpr (HAS_ROPE == ENABLE) {
                dqRopeWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                        TilingData->postTilingData.dqRopeWorkSpaceOffset / sizeof(float));
            }
            dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                    TilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
            if constexpr (HAS_ROPE == ENABLE) {
                dkRopeWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                        TilingData->postTilingData.dkRopeWorkSpaceOffset / sizeof(float));
            }
            dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                    TilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));

            dsinksumWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                            TilingData->postTilingData.dsinksumWorkSpaceOffset / sizeof(float));

            initdqRopeSize = cBlockIdx == qRopePreBlockTotal - 1 ? qRopePreBlockTail : qRopePreBlockFactor;
            dqRopeOffset = ((int64_t)cBlockIdx) * qRopePreBlockFactor;
            initdkRopeSize = cBlockIdx == kRopePreBlockTotal - 1 ? kRopePreBlockTail : kRopePreBlockFactor;
            dkRopeOffset = ((int64_t)cBlockIdx) * kRopePreBlockFactor;

            initdqSize = cBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
            dqOffset = ((int64_t)cBlockIdx) * qPreBlockFactor;
            initdkSize = cBlockIdx == kvPreBlockTotal - 1 ? kvPreBlockTail : kvPreBlockFactor;
            dkvOffset = ((int64_t)cBlockIdx) * kvPreBlockFactor;
            initdvSize = cBlockIdx == vPreBlockTotal - 1 ? vPreBlockTail : vPreBlockFactor;
            dvOffset = ((int64_t)cBlockIdx) * vPreBlockFactor;
        }

        // dropMask params init
        isDropBoolMode = TilingData->preTilingData.dropoutIsDivisibleBy8 == 0 ? true : false;

        if (isDropBoolMode) {
            maskWorkSpaceGm.SetGlobalBuffer((__gm__ uint8_t *)workspace + TilingData->preTilingData.dropBeginAddr);

            pipe->InitBuffer(helpQue, 1, HELP_LEN);
            pipe->InitBuffer(inputQue, 1, TilingData->preTilingData.inputBufferLen);
            pipe->InitBuffer(castQue, 1, TilingData->preTilingData.castBufferLen);
            pipe->InitBuffer(outQue, 1, TilingData->preTilingData.outputBufferLen);

            // reset params
            repParams.src0BlkStride = 1;
            repParams.src0RepStride = 0;
            repParams.src1BlkStride = 0;
            repParams.src1RepStride = 0;
            repParams.dstBlkStride = 1;
            repParams.dstRepStride = NUMBER_8;

            copyParams.blockCount = 1;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
        }
    }
    __aicore__ inline void Process()
    {
        // process clear dq dk dv workspace
        if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.qPreBlockTotal) {
            if constexpr (INIT_OUTPUT) {
                InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
            }
        }

        if constexpr (HAS_ROPE == ENABLE) {
            if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.qRopePreBlockTotal) {
                if constexpr (INIT_OUTPUT) {
                    InitOutput<float>(dqRopeWorkSpaceGm[dqRopeOffset], initdqRopeSize, 0);
                }
            }
        }

        if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.kvPreBlockTotal) {
            if constexpr (INIT_OUTPUT) {
                InitOutput<float>(dkWorkSpaceGm[dkvOffset], initdkSize, 0);

            }
        }

        if constexpr (HAS_ROPE == ENABLE) {
            if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.kRopePreBlockTotal) {
                if constexpr (INIT_OUTPUT) {
                    InitOutput<float>(dkRopeWorkSpaceGm[dkRopeOffset], initdkRopeSize, 0);
                }
            }
        }

        if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.vPreBlockTotal) {
            if constexpr (INIT_OUTPUT) {
                if constexpr (IsSameType<T1, float>::value) {
                    InitOutput<float>(dvGm[dvOffset], initdvSize, 0);
                } else {
                    InitOutput<float>(dvWorkSpaceGm[dvOffset], initdvSize, 0);
                }
            }
        }

        if (g_coreType == AIV && cBlockIdx < TilingData->preTilingData.maskCoreNum) {
            if (!isDropBoolMode) {
                return;
            }
            maskUBLoop = TilingData->preTilingData.maskSingleCoreLoop;
            maskTailUBProcessNum = TilingData->preTilingData.maskLastLoopNum;
            if (unlikely(cBlockIdx == maskUsedCoreNum - 1)) {
                maskUBLoop = TilingData->preTilingData.maskTailCoreLoop;
                maskTailUBProcessNum = TilingData->preTilingData.maskTailCoreLastLoopNum;
            }

            // malloc tensor filled by 1.0
            auto helpTensor = helpQue.AllocTensor<half>();
            Duplicate<half>(helpTensor, padValue, HELP_LEN / sizeof(half));
            AscendC::PipeBarrier<PIPE_V>();

            int64_t outputAddr = cBlockIdx * TilingData->preTilingData.maskSingleCoreNum;
            int64_t inputAddr = cBlockIdx * TilingData->preTilingData.maskSingleCoreNum / BIT8;

            // process
            for (int64_t idx = 0; idx < maskUBLoop; idx++) {
                maskUBProcessNum = TilingData->preTilingData.singleUBProcessNum;
                int64_t outputOffset = idx * maskUBProcessNum;
                int64_t inputOffset = idx * maskUBProcessNum / BIT8;
                if (unlikely(idx == maskUBLoop - 1)) {
                    maskUBProcessNum = maskTailUBProcessNum;
                }

                // copyIn
                auto inputTensor = inputQue.AllocTensor<uint8_t>();
                copyParams.blockLen = maskUBProcessNum / BIT8;
                DataCopyPad(inputTensor, drop_maskGm[inputAddr + inputOffset], copyParams, {false, 0, 0, 0});
                inputQue.EnQue(inputTensor);
                inputQue.DeQue<uint8_t>();

                // select
                auto castTensor = castQue.AllocTensor<half>();
                uint8_t selectRepeat = (maskUBProcessNum + B16_VECTOR_MASK - 1) / B16_VECTOR_MASK;
                Select(castTensor, inputTensor, helpTensor, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, B16_VECTOR_MASK,
                    selectRepeat, repParams);
                AscendC::PipeBarrier<PIPE_V>();
                inputQue.FreeTensor(inputTensor);

                // cast
                auto outputTensor = outQue.AllocTensor<uint8_t>();
                Cast(outputTensor, castTensor, RoundMode::CAST_ROUND, maskUBProcessNum);
                castQue.FreeTensor(castTensor);

                // copyOut
                outQue.EnQue(outputTensor);
                outQue.DeQue<uint8_t>();
                copyParams.blockLen = maskUBProcessNum;
                DataCopyPad(maskWorkSpaceGm[outputAddr + outputOffset], outputTensor, copyParams);
                outQue.FreeTensor(outputTensor);
            }
            helpQue.FreeTensor(helpTensor);
        }
    }
    __aicore__ inline void SyncALLCores()
    {
        SyncAll();
    }

    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> helpQue;
    TQue<QuePosition::VECIN, 1> inputQue;
    TQue<QuePosition::VECIN, 1> castQue;
    TQue<QuePosition::VECOUT, 1> outQue;

    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, dsinksumWorkSpaceGm, dvGm;
    GlobalTensor<uint8_t> maskWorkSpaceGm;
    GlobalTensor<uint8_t> drop_maskGm;
    GlobalTensor<float> dqRopeWorkSpaceGm, dkRopeWorkSpaceGm;

    const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *TilingData;
    constexpr static uint32_t HELP_LEN = 256;
    constexpr static uint32_t BIT8 = 8;
    constexpr static uint32_t NUMBER_8 = 8;
    constexpr static uint32_t B16_VECTOR_MASK = 128;
    constexpr static uint32_t ENABLE = 1;

    uint32_t cBlockIdx;
    // query
    uint32_t ubBaseSize;
    uint32_t qPreBlockFactor;
    uint32_t qPreBlockTotal;
    uint32_t qPreBlockTail;
    uint32_t qRopePreBlockFactor;
    uint32_t qRopePreBlockTotal;
    uint32_t qRopePreBlockTail;
    uint32_t kvPreBlockFactor;
    uint32_t kvPreBlockTotal;
    uint32_t kvPreBlockTail;
    uint32_t kRopePreBlockFactor;
    uint32_t kRopePreBlockTotal;
    uint32_t kRopePreBlockTail;

    uint32_t vPreBlockFactor;
    uint32_t vPreBlockTotal;
    uint32_t vPreBlockTail;

    int64_t qSizeAlign;
    int64_t kvSizeAlign;
    int64_t qRopeSizeAlign;
    int64_t kRopeSizeAlign;
    int64_t vSizeAlign;

    int64_t initdqSize;
    int64_t dqOffset;
    int64_t initdkSize;
    int64_t dkvOffset;
    int64_t initdqRopeSize;
    int64_t dqRopeOffset;
    int64_t initdkRopeSize;
    int64_t dkRopeOffset;
    int64_t initdvSize;
    int64_t dvOffset;

    bool isDropBoolMode;
    uint32_t maskUsedCoreNum;
    uint32_t maskUBProcessNum;
    uint32_t maskTailUBProcessNum;
    uint32_t maskUBLoop;

    DataCopyParams copyParams;
    BinaryRepeatParams repParams;
    half padValue{1.0};
};

#endif // FLASH_ATTENTION_SCORE_GRAD_PRE_KERNEL_H_