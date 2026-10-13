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
 * \file flash_attention_score_grad_post.h
 * \brief common post process
 */

#ifndef _FLASH_ATTENTION_SCORE_GRAD_POST_H_
#define _FLASH_ATTENTION_SCORE_GRAD_POST_H_

#include "kernel_operator.h"
#include "util.h"
#include "flash_attention_score_grad_tiling.h"

using AscendC::CopyRepeatParams;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;

template <typename OUT_TYPE, class TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const uint32_t HAS_ROPE = 0>
class FlashAttentionScoreGradPost {
public:
    __aicore__ inline FlashAttentionScoreGradPost(){};
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dqRope, __gm__ uint8_t *dk, __gm__ uint8_t *dkRope, __gm__ uint8_t *dv,
                         __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen, __gm__ uint8_t *dsink,
                         __gm__ uint8_t *workspace, const TILING_TYPE *__restrict ordTilingData, TPipe *pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void InitIndex(uint64_t startIdx, int64_t curG, int64_t &curS, GM_ADDR seqS, int64_t d, int64_t dAlign);
    __aicore__ inline void NZ2ND(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, uint64_t sLen,
                          uint64_t ubOffset, uint64_t srcUbOffset);
    __aicore__ inline void NZVecClc(GlobalTensor<float> srcGm, GlobalTensor<OUT_TYPE> dstGm, uint64_t dataSize,
                             GM_ADDR seqS, int64_t curG, int64_t &curS, bool needMuls, int64_t flag, int64_t d, int64_t dAlign);
    __aicore__ inline void NZProcess();
    __aicore__ inline void ComputeDataCopyOffset(int64_t curG, int64_t &curS, int64_t d, int64_t dAlign);

    constexpr static uint32_t BUFFER_NUM = 1;
    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    // NZ buffer
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePing;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePong;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueuePing;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueuePong;
    TBuf<> tmpBufPing;
    TBuf<> tmpBufPong;

    AscendC::GlobalTensor<OUT_TYPE> dqGm, dkGm, dvGm;
    AscendC::GlobalTensor<OUT_TYPE> dqRopeGm;
    AscendC::GlobalTensor<OUT_TYPE> dkRopeGm;
    // input
    AscendC::GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    AscendC::GlobalTensor<float> dqRopeWorkSpaceGm;
    AscendC::GlobalTensor<float> dkRopeWorkSpaceGm;

    const TILING_TYPE *__restrict tilingData;
    constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint64_t C0_SIZE = 16;
    constexpr static uint64_t VEC_REPEAT = 8;
    constexpr static uint32_t cal_block_num = 32 / sizeof(float);
    constexpr static uint32_t ENABLE = 1;

    int64_t usedCoreNum;
    int64_t cBlockIdx;
    // query
    int64_t ubBaseSize;
    uint32_t nzReservedSize;
    int64_t qPostBlockFactor;
    uint64_t qPostBlockTotal;
    int64_t qPostBaseNum;
    int64_t qPostTailNum;
    uint64_t qSizeAlign;
    int64_t kvPostBlockFactor;
    uint64_t kvPostBlockTotal;
    int64_t kvPostBaseNum;
    int64_t kvPostTailNum;
    uint64_t kvSizeAlign;

    int64_t qRopePostBlockFactor;
    uint64_t qRopePostBlockTotal;
    int64_t qRopePostBaseNum;
    int64_t qRopePostTailNum;
    uint64_t qRopeSizeAlign;
    int64_t kRopePostBlockFactor;
    uint64_t kRopePostBlockTotal;
    int64_t kRopePostBaseNum;
    int64_t kRopePostTailNum;
    uint64_t kRopeSizeAlign;

    // org shape info
    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t dAlign;
    int64_t rope_d = 0;
    int64_t rope_dAlign = 0;
    constexpr static uint32_t BNGSD = 0;
    constexpr static uint32_t SBNGD = 1;
    constexpr static uint32_t BSNGD = 2;
    constexpr static uint32_t TND = 3;

    constexpr static uint32_t ND = 0;
    constexpr static uint32_t NZ = 1;

    GM_ADDR actual_seq_qlen_addr;
    GM_ADDR actual_seq_kvlen_addr;

    uint64_t bIdx;
    uint64_t nIdx;
    uint64_t sIdx;

    uint64_t scrOffsetBase = 0;
    uint64_t dstOffsetBase = 0;
    uint64_t copyInSrcOffset = 0;
    uint64_t copyOutDstOffset = 0;
    uint64_t copyOutDstStride = 0;
};

template <typename OUT_TYPE, class TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT, HAS_ROPE>::Init(
    __gm__ uint8_t *dq, __gm__ uint8_t *dqRope, __gm__ uint8_t *dk, __gm__ uint8_t *dkRope, __gm__ uint8_t *dv, __gm__ uint8_t *actual_seq_qlen,
    __gm__ uint8_t *actual_seq_kvlen,__gm__ uint8_t *dsink, __gm__ uint8_t *workspace, const TILING_TYPE *__restrict ordTilingData,
    TPipe *pipe_in)
{
    cBlockIdx = GetBlockIdx();

    tilingData = ordTilingData;
    pipe = pipe_in;

    dqGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dq);
    if constexpr (HAS_ROPE == ENABLE) {
        dqRopeGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dqRope);
    }
    dkGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dk);
    if constexpr (HAS_ROPE == ENABLE) {
        dkRopeGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dkRope);
    }
    dvGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dv);

    // tiling_data
    usedCoreNum = tilingData->postTilingData.coreNum;
    ubBaseSize = tilingData->postTilingData.postUbBaseSize;
    nzReservedSize = tilingData->postTilingData.nzReservedSize;
    qPostBlockFactor = tilingData->postTilingData.qPostBlockFactor;
    qPostBlockTotal = tilingData->postTilingData.qPostBlockTotal;
    qPostBaseNum = tilingData->postTilingData.qPostBaseNum;
    qPostTailNum = tilingData->postTilingData.qPostTailNum;
    kvPostBlockFactor = tilingData->postTilingData.kvPostBlockFactor;
    kvPostBlockTotal = tilingData->postTilingData.kvPostBlockTotal;
    kvPostBaseNum = tilingData->postTilingData.kvPostBaseNum;
    kvPostTailNum = tilingData->postTilingData.kvPostTailNum;
    qSizeAlign = tilingData->postTilingData.qSizeAlign;
    kvSizeAlign = tilingData->postTilingData.kvSizeAlign;

    if constexpr (HAS_ROPE == ENABLE) {
        qRopePostBlockFactor = tilingData->postTilingData.qRopePostBlockFactor;
        qRopePostBlockTotal = tilingData->postTilingData.qRopePostBlockTotal;
        qRopePostBaseNum = tilingData->postTilingData.qRopePostBaseNum;
        qRopePostTailNum = tilingData->postTilingData.qRopePostTailNum;
        kRopePostBlockFactor = tilingData->postTilingData.kRopePostBlockFactor;
        kRopePostBlockTotal = tilingData->postTilingData.kRopePostBlockTotal;
        kRopePostBaseNum = tilingData->postTilingData.kRopePostBaseNum;
        kRopePostTailNum = tilingData->postTilingData.kRopePostTailNum;
        qRopeSizeAlign = tilingData->postTilingData.qRopeSizeAlign;
        kRopeSizeAlign = tilingData->postTilingData.kRopeSizeAlign;
    }
    if constexpr (INPUT_FORMAT == NZ) {
        b = tilingData->postTilingData.b;
        n2 = tilingData->postTilingData.n2;
        g = tilingData->postTilingData.g;
        s1 = tilingData->postTilingData.s1;
        s2 = tilingData->postTilingData.s2;
        d = tilingData->postTilingData.d;
        dAlign = (d + 15) / 16 * 16;
        actual_seq_qlen_addr = actual_seq_qlen;
        actual_seq_kvlen_addr = actual_seq_kvlen;
        if constexpr (HAS_ROPE == ENABLE) {
            rope_d = tilingData->postTilingData.rope_d;
            rope_dAlign = (rope_d + 15) / 16 * 16;
        }
    }

    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  tilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
    if constexpr (HAS_ROPE == ENABLE) {
        dqRopeWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                          tilingData->postTilingData.dqRopeWorkSpaceOffset / sizeof(float));
    }
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  tilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
    if constexpr (HAS_ROPE == ENABLE) {
        dkRopeWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                          tilingData->postTilingData.dkRopeWorkSpaceOffset / sizeof(float));
    }
    if constexpr (CAST_DV) {
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                      tilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));
    }

    if constexpr (INPUT_FORMAT == NZ) {
        pipe->InitBuffer(inQueuePing, 1, ubBaseSize * 2 + nzReservedSize);
        pipe->InitBuffer(inQueuePong, 1, ubBaseSize * 2 + nzReservedSize);
        pipe->InitBuffer(outQueuePing, 1, ubBaseSize);
        pipe->InitBuffer(outQueuePong, 1, ubBaseSize);
        pipe->InitBuffer(tmpBufPing, ubBaseSize * 2);
        pipe->InitBuffer(tmpBufPong, ubBaseSize * 2);
    } else {
        pipe->InitBuffer(inQueue, 1, ubBaseSize * 2);
        pipe->InitBuffer(outQueue, 1, ubBaseSize);
    }
}

template <typename OUT_TYPE, class TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT, HAS_ROPE>::InitIndex(
    uint64_t startIdx, int64_t curG, int64_t &curS, GM_ADDR seqS, int64_t d, int64_t dAlign)
{
    if constexpr (LAYOUT == TND) {
        uint64_t totalLen = 0;
        for (int64_t bDimIdx = 0; bDimIdx < b; bDimIdx++) {
            totalLen = n2 * curG * ((__gm__ int64_t *)seqS)[bDimIdx] * d;
            if (totalLen > startIdx) {
                bIdx = bDimIdx;
                curS = (bIdx == 0) ? ((__gm__ int64_t *)seqS)[bIdx] :
                                     (((__gm__ int64_t *)seqS)[bIdx] - ((__gm__ int64_t *)seqS)[bIdx - 1]);
                uint64_t bTail = startIdx - (totalLen - n2 * curG * curS * d);
                nIdx = bTail / (curS * d);
                uint64_t nTail = bTail % (curS * d);
                sIdx = nTail / d;
                break;
            }
        }
        // 计算输入、输出的offset
        dstOffsetBase = totalLen - n2 * curG * curS * d;
        scrOffsetBase = dstOffsetBase / d * dAlign;

        copyInSrcOffset = scrOffsetBase + nIdx * curS * dAlign + sIdx * C0_SIZE;
        copyOutDstOffset = dstOffsetBase + (sIdx * n2 * curG + nIdx) * d;
    } else { // 补充offset 计算
        bIdx = startIdx / (n2 * curG * curS * d);
        uint64_t bTail = startIdx % (n2 * curG * curS * d);
        nIdx = bTail / (curS * d);
        uint64_t nTail = bTail % (curS * d);
        sIdx = nTail / d;
        ComputeDataCopyOffset(curG, curS, d, dAlign);
    }
}


template <typename OUT_TYPE, class TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT, HAS_ROPE>::ComputeDataCopyOffset(int64_t curG, int64_t &curS, int64_t d, int64_t dAlign)
{
    // src BNSD
    scrOffsetBase = bIdx * n2 * curS * curG * dAlign;
    copyInSrcOffset = scrOffsetBase + nIdx * curS * dAlign + sIdx * C0_SIZE;

    if constexpr (LAYOUT == BSNGD) {
        // BSND
        copyOutDstStride = n2 * curG * d - d;
        copyOutDstOffset = ((bIdx * curS + sIdx ) * n2 * curG + nIdx) * d;
    } else if constexpr (LAYOUT == SBNGD) {
        // SBND
        copyOutDstStride = b * n2 * curG * d - d;
        copyOutDstOffset = (( sIdx * b + bIdx ) * n2 * curG + nIdx ) * d;
    } else if constexpr (LAYOUT == BNGSD) {
        // BNSD
        copyOutDstStride = 0;
        copyOutDstOffset = ((bIdx * n2 * curG + nIdx )* curS + sIdx) * d;
    }
}

template <typename OUT_TYPE, class TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT, HAS_ROPE>::NZ2ND(
    LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, uint64_t sLen, uint64_t ubOffset,
    uint64_t srcUbOffset)
{
    /*
    Func:
    将NZ转为ND
    */
    CopyRepeatParams nz2ndParams;
    nz2ndParams.srcStride = sLen * C0_SIZE / cal_block_num + 1;
    nz2ndParams.dstStride = C0_SIZE / cal_block_num;
    nz2ndParams.srcRepeatSize = C0_SIZE / cal_block_num;
    nz2ndParams.dstRepeatSize = dAlign / cal_block_num;

    uint16_t c0_repeat = C0_SIZE / cal_block_num;
    uint16_t c1_repeat = dAlign / C0_SIZE / VEC_REPEAT;
    uint16_t c1_remain = dAlign / C0_SIZE % VEC_REPEAT;
    uint16_t n_repeat = sLen;
    for (uint16_t i = 0; i < c0_repeat; ++i) {
        for (uint16_t j = 0; j < c1_repeat; ++j) {
            Copy(dstTensor[ubOffset + i * cal_block_num + j * VEC_REPEAT * C0_SIZE],
                 srcTensor[srcUbOffset + i * cal_block_num + j * VEC_REPEAT * (sLen * C0_SIZE + cal_block_num)],
                 VEC_REPEAT * cal_block_num, n_repeat, nz2ndParams);
        }
        if (c1_remain > 0) {
            Copy(dstTensor[ubOffset + i * cal_block_num + c1_repeat * VEC_REPEAT * C0_SIZE],
                 srcTensor[srcUbOffset + i * cal_block_num + c1_repeat * VEC_REPEAT * (sLen * C0_SIZE + cal_block_num)],
                 VEC_REPEAT * c1_remain, n_repeat, nz2ndParams);
        }
    }
}

template <typename OUT_TYPE, class TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT, HAS_ROPE>::NZVecClc(
    GlobalTensor<float> srcGm, GlobalTensor<OUT_TYPE> dstGm, uint64_t dataSize, GM_ADDR seqS, int64_t curG,
    int64_t &curS, bool needMuls, int64_t flag, int64_t d, int64_t dAlign)
{
    if (dataSize == 0) {
        return;
    }

    
    event_t mte2WaitVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t mte2WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCommon;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueCommon;
    TBuf<> tmpBufCommon;
    event_t curEventId;
    if (flag) {
        inQueueCommon = inQueuePing;
        outQueueCommon = outQueuePing;
        tmpBufCommon = tmpBufPing;
        curEventId = mte2WaitVPing;
    } else {
        inQueueCommon = inQueuePong;
        outQueueCommon = outQueuePong;
        tmpBufCommon = tmpBufPong;
        curEventId = mte2WaitVPong;
    }

    LocalTensor<float> vecIn = inQueueCommon.template AllocTensor<float>();
    LocalTensor<float> tmpTensor = tmpBufCommon.template Get<float>();
    LocalTensor<OUT_TYPE> vecOut = outQueueCommon.template AllocTensor<OUT_TYPE>();

    uint32_t sClcSize = dataSize / d;

    uint64_t sLen = (sIdx + sClcSize) > curS ? (curS - sIdx) : sClcSize;
    sLen = sLen > 255 ? 255 : sLen;
    uint64_t dataLen = sLen * dAlign;

    uint64_t ubOffset = 0;
    uint64_t inUbOffset = 0;

    while (sClcSize > 0) {
        // Nz copy In
        AscendC::DataCopyExtParams intriParams;
        intriParams.blockCount = dAlign / C0_SIZE;
        intriParams.blockLen = sLen * C0_SIZE * sizeof(float);
        intriParams.srcStride = curS * C0_SIZE * sizeof(float) - intriParams.blockLen;
        intriParams.dstStride = 1; // 间隔一个block，防止bank冲突
        intriParams.rsv = 0;
        DataCopyPad(vecIn[inUbOffset], srcGm[copyInSrcOffset], intriParams, {false, 0, 0, 0});
        sClcSize = sClcSize - sLen;

        inQueueCommon.EnQue(vecIn);
        inQueueCommon.template DeQue<float>();

        if constexpr (!AscendC::IsSameType<OUT_TYPE, float>::value) {
            NZ2ND(tmpTensor, vecIn, sLen, ubOffset, inUbOffset);
        } else {
            NZ2ND(vecOut, vecIn, sLen, ubOffset, inUbOffset);
        }

        if (sClcSize <= 0) {
            inQueueCommon.FreeTensor(vecIn);
        }

        AscendC::PipeBarrier<PIPE_V>();
        if (needMuls) {

            if constexpr (!AscendC::IsSameType<OUT_TYPE, float>::value) {
                Muls(tmpTensor[ubOffset], tmpTensor[ubOffset], (float)tilingData->postTilingData.scaleValue,
                 sLen * dAlign);
            } else {
                Muls(vecOut[ubOffset], vecOut[ubOffset], (float)tilingData->postTilingData.scaleValue,
                 sLen * dAlign);
            }

            AscendC::PipeBarrier<PIPE_V>();
        }

        if constexpr (!AscendC::IsSameType<OUT_TYPE, float>::value) {
            Cast(vecOut[ubOffset], tmpTensor[ubOffset], RoundMode::CAST_ROUND, sLen * dAlign);
            AscendC::PipeBarrier<PIPE_V>();
        }

        outQueueCommon.EnQue(vecOut);
        outQueueCommon.template DeQue<OUT_TYPE>();

        if constexpr (LAYOUT == TND) {
            DataCopyPad(dstGm[copyOutDstOffset], vecOut[ubOffset],
                    {static_cast<uint16_t>(dataLen / dAlign), static_cast<uint32_t>(d * sizeof(OUT_TYPE)), 0,
                    static_cast<uint32_t>((n2 * curG * d - d) * sizeof(OUT_TYPE)), 0});
        } else {
            DataCopyPad(dstGm[copyOutDstOffset], vecOut[ubOffset],
                    {static_cast<uint16_t>(dataLen / dAlign), static_cast<uint32_t>(d * sizeof(OUT_TYPE)), 0,
                    static_cast<uint32_t>(copyOutDstStride * sizeof(OUT_TYPE)), 0});
        }


        if (sLen + sIdx < curS) {
            sIdx += sLen;
        } else if (nIdx == n2 * curG - 1) {
            sIdx = 0;
            nIdx = 0;
            bIdx++;
            if constexpr (LAYOUT == TND) {
                scrOffsetBase += curS * n2 * curG * dAlign;
                dstOffsetBase += curS * n2 * curG * d;
                if (bIdx < b) {
                    curS = ((__gm__ int64_t *)seqS)[bIdx] - ((__gm__ int64_t *)seqS)[bIdx - 1];
                } else {
                    curS = 0;
                }
            }
        } else {
            sIdx = 0;
            nIdx++;
        }
        if constexpr (LAYOUT == TND) {
            copyInSrcOffset = scrOffsetBase + nIdx * curS * dAlign + sIdx * C0_SIZE;
            copyOutDstOffset = dstOffsetBase + (sIdx * n2 * curG + nIdx) * d;
        } else {
            ComputeDataCopyOffset(curG, curS, d, dAlign);
        }
        ubOffset += dataLen;
        inUbOffset += dataLen + dAlign / C0_SIZE * cal_block_num;
        sLen = sClcSize > curS ? curS : sClcSize;
        sLen = sLen > (curS - sIdx) ? (curS - sIdx) : sLen;
        sLen = sLen > 255 ? 255 : sLen;
        dataLen = sLen * dAlign;
        if ((sLen > 0) && (inUbOffset + dataLen + dAlign / C0_SIZE * cal_block_num) * sizeof(float) >
                              ubBaseSize * 2 + nzReservedSize) {
            inUbOffset = 0;
            AscendC::SetFlag<HardEvent::V_MTE2>(static_cast<int32_t>(curEventId));
            AscendC::WaitFlag<HardEvent::V_MTE2>(static_cast<int32_t>(curEventId));
        }
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(mte2WaitVPing);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(mte2WaitVPong);
    outQueueCommon.FreeTensor(vecOut);
}

template <typename OUT_TYPE, class TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT, HAS_ROPE>::NZProcess()
{
    uint64_t qBegin = cBlockIdx * qPostBlockFactor * qPostBaseNum;
    uint64_t qEnd = (cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum;
    if (((cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum) > qPostBlockTotal) {
        qEnd = qPostBlockTotal;
    }

    InitIndex(qBegin, g, s1, actual_seq_qlen_addr, d, dAlign);


    for (uint64_t i = qBegin; i < qEnd; i = i + 2 * qPostBaseNum) {
        uint64_t dataSize = i + qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
        NZVecClc(dqWorkSpaceGm, dqGm, dataSize, actual_seq_qlen_addr, g, s1, true, 0, d, dAlign);
        uint64_t dataSize1 = i + 2 * qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
        dataSize1 = i + qPostBaseNum >= qPostBlockTotal ? 0 : dataSize1;
        NZVecClc(dqWorkSpaceGm, dqGm, dataSize1, actual_seq_qlen_addr, g, s1, true, 1, d, dAlign);
    }

    if constexpr (HAS_ROPE == ENABLE) {
        uint64_t qRopeBegin = cBlockIdx * qRopePostBlockFactor * qRopePostBaseNum;
        uint64_t qRopeEnd = (cBlockIdx + 1) * qRopePostBlockFactor * qRopePostBaseNum;
        if (((cBlockIdx + 1) * qRopePostBlockFactor * qRopePostBaseNum) > qRopePostBlockTotal) {
            qRopeEnd = qRopePostBlockTotal;
        }

        InitIndex(qRopeBegin, g, s1, actual_seq_qlen_addr, rope_d, rope_dAlign);

        for (uint64_t i = qRopeBegin; i < qRopeEnd; i = i + 2 * qRopePostBaseNum) {
            uint64_t dataSize = i + qRopePostBaseNum < qRopePostBlockTotal ? qRopePostBaseNum : qRopePostTailNum;
            NZVecClc(dqRopeWorkSpaceGm, dqRopeGm, dataSize, actual_seq_qlen_addr, g, s1, true, 0, rope_d, rope_dAlign);
            uint64_t dataSize1 = i + 2 * qRopePostBaseNum < qRopePostBlockTotal ? qRopePostBaseNum : qRopePostTailNum;
            dataSize1 = i + qRopePostBaseNum >= qRopePostBlockTotal ? 0 : dataSize1;
            NZVecClc(dqRopeWorkSpaceGm, dqRopeGm, dataSize1, actual_seq_qlen_addr, g, s1, true, 1, rope_d, rope_dAlign);
        }
    }
    // init k
    uint64_t kvBegin = cBlockIdx * kvPostBlockFactor * kvPostBaseNum;
    uint64_t kvEnd = (cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum;
    if (((cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum) > kvPostBlockTotal) {
        kvEnd = kvPostBlockTotal;
    }
    InitIndex(kvBegin, 1, s2, actual_seq_kvlen_addr, d, dAlign);
    for (uint64_t i = kvBegin; i < kvEnd; i = i + 2 * kvPostBaseNum) {
        uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
        NZVecClc(dkWorkSpaceGm, dkGm, dataSize, actual_seq_kvlen_addr, 1, s2, true, 0, d, dAlign);
        uint64_t dataSize1 = i + 2 * kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
        dataSize1 = i + kvPostBaseNum >= kvPostBlockTotal ? 0 : dataSize1;
        NZVecClc(dkWorkSpaceGm, dkGm, dataSize1, actual_seq_kvlen_addr, 1, s2, true, 1, d, dAlign);
    }

    if constexpr (HAS_ROPE == ENABLE) {
        // init kRope
        uint64_t kRopeBegin = cBlockIdx * kRopePostBlockFactor * kRopePostBaseNum;
        uint64_t kRopeEnd = (cBlockIdx + 1) * kRopePostBlockFactor * kRopePostBaseNum;
        if (((cBlockIdx + 1) * kRopePostBlockFactor * kRopePostBaseNum) > kRopePostBlockTotal) {
            kRopeEnd = kRopePostBlockTotal;
        }
        InitIndex(kRopeBegin, 1, s2, actual_seq_kvlen_addr, rope_d, rope_dAlign);
        for (uint64_t i = kRopeBegin; i < kRopeEnd; i = i + 2 * kRopePostBaseNum) {
            uint64_t dataSize = i + kRopePostBaseNum < kRopePostBlockTotal ? kRopePostBaseNum : kRopePostTailNum;
            NZVecClc(dkRopeWorkSpaceGm, dkRopeGm, dataSize, actual_seq_kvlen_addr, 1, s2, true, 0, rope_d, rope_dAlign);
            uint64_t dataSize1 = i + 2 * kRopePostBaseNum < kRopePostBlockTotal ? kRopePostBaseNum : kRopePostTailNum;
            dataSize1 = i + kRopePostBaseNum >= kRopePostBlockTotal ? 0 : dataSize1;
            NZVecClc(dkRopeWorkSpaceGm, dkRopeGm, dataSize1, actual_seq_kvlen_addr, 1, s2, true, 1, rope_d, rope_dAlign);
        }
    }

    // init v
    if constexpr (CAST_DV) {
        InitIndex(kvBegin, 1, s2, actual_seq_kvlen_addr, d, dAlign);
        for (uint64_t i = kvBegin; i < kvEnd; i = i + 2 * kvPostBaseNum) {
            uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
            NZVecClc(dvWorkSpaceGm, dvGm, dataSize, actual_seq_kvlen_addr, 1, s2, false, 0, d, dAlign);
            uint64_t dataSize1 = i + 2 * kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
            dataSize1 = i + kvPostBaseNum >= kvPostBlockTotal ? 0 : dataSize1;
            NZVecClc(dvWorkSpaceGm, dvGm, dataSize1, actual_seq_kvlen_addr, 1, s2, false, 1, d, dAlign);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

template <typename OUT_TYPE, class TILING_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradPost<OUT_TYPE, TILING_TYPE, CAST_DV, LAYOUT, INPUT_FORMAT, HAS_ROPE>::Process()
{
    if constexpr (INPUT_FORMAT == NZ) {
        NZProcess();
        return;
    }
    // init q
    uint64_t qBegin = cBlockIdx * qPostBlockFactor * qPostBaseNum;
    uint64_t qEnd = (cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum;

    if (((cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum) > qPostBlockTotal) {
        qEnd = qPostBlockTotal;
    }
    for (uint64_t i = qBegin; i < qEnd; i = i + qPostBaseNum) {
        AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
        AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
        uint64_t dataSize = i + qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
        DataCopy(vecIn, dqWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
        inQueue.EnQue(vecIn);
        inQueue.template DeQue<float>();
        if constexpr (AscendC::IsSameType<OUT_TYPE, float>::value) {
            Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();
            DataCopy(dqGm[i], vecOut, (dataSize + 7) / 8 * 8); // dataSize(fp16) align 32B
        } else {
            Muls(vecIn, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();
            DataCopy(dqGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
        }
        inQueue.FreeTensor(vecIn);
        outQueue.FreeTensor(vecOut);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    
    if constexpr (HAS_ROPE == ENABLE) {
        // init qRope
        uint64_t qRopeBegin = cBlockIdx * qRopePostBlockFactor * qRopePostBaseNum;
        uint64_t qRopeEnd = (cBlockIdx + 1) * qRopePostBlockFactor * qRopePostBaseNum;

        if (((cBlockIdx + 1) * qRopePostBlockFactor * qRopePostBaseNum) > qRopePostBlockTotal) {
            qRopeEnd = qRopePostBlockTotal;
        }
        for (uint64_t i = qRopeBegin; i < qRopeEnd; i = i + qRopePostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
            AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
            uint64_t dataSize = i + qRopePostBaseNum < qRopePostBlockTotal ? qRopePostBaseNum : qRopePostTailNum;
            DataCopy(vecIn, dqRopeWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
            inQueue.EnQue(vecIn);
            inQueue.template DeQue<float>();
            if constexpr (AscendC::IsSameType<OUT_TYPE, float>::value) {
                Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                outQueue.EnQue(vecOut);
                outQueue.template DeQue<OUT_TYPE>();
                DataCopy(dqRopeGm[i], vecOut, (dataSize + 7) / 8 * 8); // dataSize(fp16) align 32B
            } else {
                Muls(vecIn, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                AscendC::PipeBarrier<PIPE_V>();
                Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
                outQueue.EnQue(vecOut);
                outQueue.template DeQue<OUT_TYPE>();
                DataCopy(dqRopeGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
            }
            inQueue.FreeTensor(vecIn);
            outQueue.FreeTensor(vecOut);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    // init k
    uint64_t kvBegin = cBlockIdx * kvPostBlockFactor * kvPostBaseNum;
    uint64_t kvEnd = (cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum;
    if (((cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum) > kvPostBlockTotal) {
        kvEnd = kvPostBlockTotal;
    }

    for (uint64_t i = kvBegin; i < kvEnd; i = i + kvPostBaseNum) {
        AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
        AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
        uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
        DataCopy(vecIn, dkWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
        inQueue.EnQue(vecIn);
        inQueue.template DeQue<float>();
        if constexpr (AscendC::IsSameType<OUT_TYPE, float>::value) {
            Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();
            DataCopy(dkGm[i], vecOut, (dataSize + 7) / 8 * 8); // dataSize(fp16) align 32B
        } else {
            Muls(vecIn, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();
            DataCopy(dkGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
        }
        inQueue.FreeTensor(vecIn);
        outQueue.FreeTensor(vecOut);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    if constexpr (HAS_ROPE == ENABLE) {
        // init kRope
        uint64_t kRopeBegin = cBlockIdx * kRopePostBlockFactor * kRopePostBaseNum;
        uint64_t kRopeEnd = (cBlockIdx + 1) * kRopePostBlockFactor * kRopePostBaseNum;
        if (((cBlockIdx + 1) * kRopePostBlockFactor * kRopePostBaseNum) > kRopePostBlockTotal) {
            kRopeEnd = kRopePostBlockTotal;
        }

        for (uint64_t i = kRopeBegin; i < kRopeEnd; i = i + kRopePostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
            AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
            uint64_t dataSize = i + kRopePostBaseNum < kRopePostBlockTotal ? kRopePostBaseNum : kRopePostTailNum;
            DataCopy(vecIn, dkRopeWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
            inQueue.EnQue(vecIn);
            inQueue.template DeQue<float>();
            if constexpr (AscendC::IsSameType<OUT_TYPE, float>::value) {
                Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                outQueue.EnQue(vecOut);
                outQueue.template DeQue<OUT_TYPE>();
                DataCopy(dkRopeGm[i], vecOut, (dataSize + 7) / 8 * 8); // dataSize(fp16) align 32B
            } else {
                Muls(vecIn, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                AscendC::PipeBarrier<PIPE_V>();
                Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
                outQueue.EnQue(vecOut);
                outQueue.template DeQue<OUT_TYPE>();
                DataCopy(dkRopeGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
            }
            inQueue.FreeTensor(vecIn);
            outQueue.FreeTensor(vecOut);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // init v

    if constexpr (CAST_DV && !AscendC::IsSameType<OUT_TYPE, float>::value) {
        for (uint64_t i = kvBegin; i < kvEnd; i = i + kvPostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
            AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
            uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
            DataCopy(vecIn, dvWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
            inQueue.EnQue(vecIn);
            inQueue.template DeQue<float>();
            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
            outQueue.EnQue(vecOut);
            outQueue.template DeQue<OUT_TYPE>();
            DataCopy(dvGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
            inQueue.FreeTensor(vecIn);
            outQueue.FreeTensor(vecOut);
        }
    }
}

// Partial Specialize for FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb
template <typename OUT_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const uint32_t HAS_ROPE>
class FlashAttentionScoreGradPost<OUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb, CAST_DV, LAYOUT,
          INPUT_FORMAT, HAS_ROPE> {
public:
    __aicore__ inline FlashAttentionScoreGradPost(){}
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dqRope, __gm__ uint8_t *dk, __gm__ uint8_t *dkRope, __gm__ uint8_t *dv,
                         __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen, __gm__ uint8_t *dsink, 
                         __gm__ uint8_t *workspace, const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb *__restrict ordTilingData, TPipe *pipe_in)
    {
        cBlockIdx = GetBlockIdx();

        tilingData = ordTilingData;
        pipe = pipe_in;

        dqGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dq);
        if constexpr (HAS_ROPE == ENABLE) {
            dqRopeGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dqRope);
        }
        dkGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dk);
        if constexpr (HAS_ROPE == ENABLE) {
            dkRopeGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dkRope);
        }
        dvGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dv);

        dsinkGm.SetGlobalBuffer((__gm__ float *)dsink);


        // tiling_data
        usedCoreNum = tilingData->postTilingData.coreNum;
        ubBaseSize = tilingData->postTilingData.postUbBaseSize;
        nzReservedSize = tilingData->postTilingData.nzReservedSize;
        qPostBlockFactor = tilingData->postTilingData.qPostBlockFactor;
        qPostBlockTotal = tilingData->postTilingData.qPostBlockTotal;
        qPostBaseNum = tilingData->postTilingData.qPostBaseNum;
        qPostTailNum = tilingData->postTilingData.qPostTailNum;
        kvPostBlockFactor = tilingData->postTilingData.kvPostBlockFactor;
        kvPostBlockTotal = tilingData->postTilingData.kvPostBlockTotal;
        kvPostBaseNum = tilingData->postTilingData.kvPostBaseNum;
        kvPostTailNum = tilingData->postTilingData.kvPostTailNum;
        vPostBlockFactor = tilingData->postTilingData.vPostBlockFactor;
        vPostBlockTotal = tilingData->postTilingData.vPostBlockTotal;
        vPostBaseNum = tilingData->postTilingData.vPostBaseNum;
        vPostTailNum = tilingData->postTilingData.vPostTailNum;
        qSizeAlign = tilingData->postTilingData.qSizeAlign;
        kvSizeAlign = tilingData->postTilingData.kvSizeAlign;
        vSizeAlign = tilingData->postTilingData.vSizeAlign;

        if constexpr (HAS_ROPE == ENABLE) {
            qRopePostBlockFactor = tilingData->postTilingData.qRopePostBlockFactor;
            qRopePostBlockTotal = tilingData->postTilingData.qRopePostBlockTotal;
            qRopePostBaseNum = tilingData->postTilingData.qRopePostBaseNum;
            qRopePostTailNum = tilingData->postTilingData.qRopePostTailNum;
            kRopePostBlockFactor = tilingData->postTilingData.kRopePostBlockFactor;
            kRopePostBlockTotal = tilingData->postTilingData.kRopePostBlockTotal;
            kRopePostBaseNum = tilingData->postTilingData.kRopePostBaseNum;
            kRopePostTailNum = tilingData->postTilingData.kRopePostTailNum;
            qRopeSizeAlign = tilingData->postTilingData.qRopeSizeAlign;
            kRopeSizeAlign = tilingData->postTilingData.kRopeSizeAlign;
        }

        if constexpr (INPUT_FORMAT == NZ) {
            b = tilingData->postTilingData.b;
            n2 = tilingData->postTilingData.n2;
            g = tilingData->postTilingData.g;
            s1 = tilingData->postTilingData.s1;
            s2 = tilingData->postTilingData.s2;
            d = tilingData->postTilingData.d;
            dAlign = (d + 15) / 16 * 16;
            value_d = tilingData->postTilingData.value_d;
            value_dAlign = (value_d + 15) / 16 * 16;
            actual_seq_qlen_addr = actual_seq_qlen;
            actual_seq_kvlen_addr = actual_seq_kvlen;
            if constexpr (HAS_ROPE == ENABLE) {
                rope_d = tilingData->postTilingData.rope_d;
                rope_dAlign = (rope_d + 15) / 16 * 16;
            }
        }

        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                    tilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
        if constexpr (HAS_ROPE == ENABLE) {
            dqRopeWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                                tilingData->postTilingData.dqRopeWorkSpaceOffset / sizeof(float));
        }
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                    tilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
        if constexpr (HAS_ROPE == ENABLE) {
            dkRopeWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                                tilingData->postTilingData.dkRopeWorkSpaceOffset / sizeof(float));
        }
        if constexpr (CAST_DV) {
            dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                        tilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));
        }
        dsinksumWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                    tilingData->postTilingData.dsinksumWorkSpaceOffset / sizeof(float));
        dsinksumDataSizeGm.SetGlobalBuffer((__gm__ uint32_t *)workspace +
                    tilingData->postTilingData.dsinksumDataSizeOffset / sizeof(uint32_t));

        if constexpr (INPUT_FORMAT == NZ) {
            pipe->InitBuffer(inQueuePing, 1, ubBaseSize * 2 + nzReservedSize);
            pipe->InitBuffer(inQueuePong, 1, ubBaseSize * 2 + nzReservedSize);
            pipe->InitBuffer(outQueuePing, 1, ubBaseSize);
            pipe->InitBuffer(outQueuePong, 1, ubBaseSize);
            pipe->InitBuffer(tmpBufPing, ubBaseSize * 2);
            pipe->InitBuffer(tmpBufPong, ubBaseSize * 2);
        } else {
            pipe->InitBuffer(inQueue, 1, ubBaseSize * 2);
            pipe->InitBuffer(outQueue, 1, ubBaseSize);
        }
    }
    __aicore__ inline void InitIndex(uint64_t startIdx, int64_t curG, int64_t &curS, GM_ADDR seqS, int64_t d, int64_t dAlign)
    {
        if constexpr (LAYOUT == TND) {
            uint64_t totalLen = 0;
            for (int64_t bDimIdx = 0; bDimIdx < b; bDimIdx++) {
                totalLen = n2 * curG * ((__gm__ int64_t *)seqS)[bDimIdx] * d;
                if (totalLen > startIdx) {
                    bIdx = bDimIdx;
                    curS = (bIdx == 0) ? ((__gm__ int64_t *)seqS)[bIdx] :
                                        (((__gm__ int64_t *)seqS)[bIdx] - ((__gm__ int64_t *)seqS)[bIdx - 1]);
                    uint64_t bTail = startIdx - (totalLen - n2 * curG * curS * d);
                    nIdx = bTail / (curS * d);
                    uint64_t nTail = bTail % (curS * d);
                    if (d != 0) {
                        sIdx = nTail / d;
                    } else {
                        // 处理d为0的情况，这里假设sIdx为0，可以根据实际情况调整
                        sIdx = 0;
                    }
                    break;
                }
            }
            // 计算输入、输出的offset
            dstOffsetBase = totalLen - n2 * curG * curS * d;
            if (d != 0) {
                scrOffsetBase = dstOffsetBase / d * dAlign;
            } else {
                // 处理d为0的情况，这里假设scrOffsetBase为0，可以根据实际情况调整
                scrOffsetBase = 0;
            }

            copyInSrcOffset = scrOffsetBase + nIdx * curS * dAlign + sIdx * C0_SIZE;
            copyOutDstOffset = dstOffsetBase + (sIdx * n2 * curG + nIdx) * d;
        } else { // 补充offset 计算
            bIdx = startIdx / (n2 * curG * curS * d);
            uint64_t bTail = startIdx % (n2 * curG * curS * d);
            nIdx = bTail / (curS * d);
            uint64_t nTail = bTail % (curS * d);
            sIdx = nTail / (d);
            ComputeDataCopyOffset(curG, curS, d, dAlign);
        }
    }
    __aicore__ inline void Process()
    {
        if constexpr (INPUT_FORMAT == NZ) {
            NZProcess();
            return;
        }
        // init q
        uint64_t qBegin = cBlockIdx * qPostBlockFactor * qPostBaseNum;
        uint64_t qEnd = (cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum;

        if (((cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum) > qPostBlockTotal) {
            qEnd = qPostBlockTotal;
        }
        for (uint64_t i = qBegin; i < qEnd; i = i + qPostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
            AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
            uint64_t dataSize = i + qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
            DataCopy(vecIn, dqWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
            inQueue.EnQue(vecIn);
            inQueue.template DeQue<float>();
            if constexpr (AscendC::IsSameType<OUT_TYPE, float>::value) {
                Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                outQueue.EnQue(vecOut);
                outQueue.template DeQue<OUT_TYPE>();
                DataCopy(dqGm[i], vecOut, (dataSize + 7) / 8 * 8); // dataSize(fp16) align 32B
            } else {
                Muls(vecIn, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                AscendC::PipeBarrier<PIPE_V>();
                Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
                outQueue.EnQue(vecOut);
                outQueue.template DeQue<OUT_TYPE>();
                DataCopy(dqGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
            }
            inQueue.FreeTensor(vecIn);
            outQueue.FreeTensor(vecOut);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        if constexpr (HAS_ROPE == ENABLE) {
            // init qRope
            uint64_t qRopeBegin = cBlockIdx * qRopePostBlockFactor * qRopePostBaseNum;
            uint64_t qRopeEnd = (cBlockIdx + 1) * qRopePostBlockFactor * qRopePostBaseNum;
    
            if (((cBlockIdx + 1) * qRopePostBlockFactor * qRopePostBaseNum) > qRopePostBlockTotal) {
                qRopeEnd = qRopePostBlockTotal;
            }
            for (uint64_t i = qRopeBegin; i < qRopeEnd; i = i + qRopePostBaseNum) {
                AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
                AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
                uint64_t dataSize = i + qRopePostBaseNum < qRopePostBlockTotal ? qRopePostBaseNum : qRopePostTailNum;
                DataCopy(vecIn, dqRopeWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
                inQueue.EnQue(vecIn);
                inQueue.template DeQue<float>();
                if constexpr (AscendC::IsSameType<OUT_TYPE, float>::value) {
                    Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                    outQueue.EnQue(vecOut);
                    outQueue.template DeQue<OUT_TYPE>();
                    DataCopy(dqRopeGm[i], vecOut, (dataSize + 7) / 8 * 8); // dataSize(fp16) align 32B
                } else {
                    Muls(vecIn, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                    AscendC::PipeBarrier<PIPE_V>();
                    Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
                    outQueue.EnQue(vecOut);
                    outQueue.template DeQue<OUT_TYPE>();
                    DataCopy(dqRopeGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
                }
                inQueue.FreeTensor(vecIn);
                outQueue.FreeTensor(vecOut);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        // init k
        uint64_t kvBegin = cBlockIdx * kvPostBlockFactor * kvPostBaseNum;
        uint64_t kvEnd = (cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum;
        if (((cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum) > kvPostBlockTotal) {
            kvEnd = kvPostBlockTotal;
        }

        for (uint64_t i = kvBegin; i < kvEnd; i = i + kvPostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
            AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
            uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
            DataCopy(vecIn, dkWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
            inQueue.EnQue(vecIn);
            inQueue.template DeQue<float>();
            if constexpr (AscendC::IsSameType<OUT_TYPE, float>::value) {
                Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                outQueue.EnQue(vecOut);
                outQueue.template DeQue<OUT_TYPE>();
                DataCopy(dkGm[i], vecOut, (dataSize + 7) / 8 * 8); // dataSize(fp16) align 32B
            } else {
                Muls(vecIn, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                AscendC::PipeBarrier<PIPE_V>();
                Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
                outQueue.EnQue(vecOut);
                outQueue.template DeQue<OUT_TYPE>();
                DataCopy(dkGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
            }
            inQueue.FreeTensor(vecIn);
            outQueue.FreeTensor(vecOut);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        if constexpr (HAS_ROPE == ENABLE) {
            // init kRope
            uint64_t kRopeBegin = cBlockIdx * kRopePostBlockFactor * kRopePostBaseNum;
            uint64_t kRopeEnd = (cBlockIdx + 1) * kRopePostBlockFactor * kRopePostBaseNum;
            if (((cBlockIdx + 1) * kRopePostBlockFactor * kRopePostBaseNum) > kRopePostBlockTotal) {
                kRopeEnd = kRopePostBlockTotal;
            }
    
            for (uint64_t i = kRopeBegin; i < kRopeEnd; i = i + kRopePostBaseNum) {
                AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
                AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
                uint64_t dataSize = i + kRopePostBaseNum < kRopePostBlockTotal ? kRopePostBaseNum : kRopePostTailNum;
                DataCopy(vecIn, dkRopeWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
                inQueue.EnQue(vecIn);
                inQueue.template DeQue<float>();
                if constexpr (AscendC::IsSameType<OUT_TYPE, float>::value) {
                    Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                    outQueue.EnQue(vecOut);
                    outQueue.template DeQue<OUT_TYPE>();
                    DataCopy(dkRopeGm[i], vecOut, (dataSize + 7) / 8 * 8); // dataSize(fp16) align 32B
                } else {
                    Muls(vecIn, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                    AscendC::PipeBarrier<PIPE_V>();
                    Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
                    outQueue.EnQue(vecOut);
                    outQueue.template DeQue<OUT_TYPE>();
                    DataCopy(dkRopeGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
                }
                inQueue.FreeTensor(vecIn);
                outQueue.FreeTensor(vecOut);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        // init v
        uint64_t vBegin = cBlockIdx * vPostBlockFactor * vPostBaseNum;
        uint64_t vEnd = (cBlockIdx + 1) * vPostBlockFactor * vPostBaseNum;
        if (((cBlockIdx + 1) * vPostBlockFactor * vPostBaseNum) > vPostBlockTotal) {
            vEnd = vPostBlockTotal;
        }
        if constexpr (CAST_DV && !AscendC::IsSameType<OUT_TYPE, float>::value) {
            for (uint64_t i = vBegin; i < vEnd; i = i + vPostBaseNum) {
                AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
                AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
                uint64_t dataSize = i + vPostBaseNum < vPostBlockTotal ? vPostBaseNum : vPostTailNum;
                DataCopy(vecIn, dvWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
                inQueue.EnQue(vecIn);
                inQueue.template DeQue<float>();
                Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
                outQueue.EnQue(vecOut);
                outQueue.template DeQue<OUT_TYPE>();
                DataCopy(dvGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
                inQueue.FreeTensor(vecIn);
                outQueue.FreeTensor(vecOut);
            }
        }

        // reduce dsinksum

        if (unlikely(tilingData->s1s2BNGS1S2BaseParams.sink == 1)) {
            AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
            AscendC::LocalTensor<float> vecOut = outQueue.template AllocTensor<float>();
            int s1Pad = (tilingData->postTilingData.s1 + 255) / 256 * 256;
            int s2Pad = (tilingData->postTilingData.s2 + 255) / 256 * 256;
            int dataSizePerN1 = tilingData->postTilingData.b * s1Pad * s2Pad / tilingData->postTilingData.baseMN;
            int N1 = tilingData->postTilingData.n2 * tilingData->postTilingData.g;
            int dsinkSumMaxCopyTimePerN1 = (dataSizePerN1 * sizeof(float) + 2 * ubBaseSize -1) / ( 2 * ubBaseSize);
            int tailDsinkSumSize = (dataSizePerN1 * sizeof(float)) % (2 * ubBaseSize);
            int copyOffset = 0;
            for (int n1temp = 0; n1temp < N1; n1temp++) {
                float dsinkCalc = 0.0;
                for (int dsinkSumDataCopyLoop = 0; dsinkSumDataCopyLoop < dsinkSumMaxCopyTimePerN1; dsinkSumDataCopyLoop++) {
                    int currentCopyDataSize = (dsinkSumDataCopyLoop ==  dsinkSumMaxCopyTimePerN1-1) ? tailDsinkSumSize : 2 * ubBaseSize;
                    int itemNum = currentCopyDataSize / sizeof(float);

                    DataCopyExtParams copyParams;
                    copyParams.blockCount = 1;
                    copyParams.blockLen = currentCopyDataSize;
                    copyParams.srcStride = 0;
                    copyParams.dstStride = 0;
                    copyParams.rsv = 0;
                    DataCopyPadExtParams<float> copyPadParams;
                    copyPadParams.isPad = ((itemNum % 8) == 0) ? false : true;
                    copyPadParams.leftPadding = 0;
                    copyPadParams.rightPadding = 8 - (itemNum % 8);
                    copyPadParams.paddingValue = 0.0;
                    AscendC::PipeBarrier<PIPE_ALL>();
                    DataCopyPad(vecIn, dsinksumWorkSpaceGm[copyOffset], copyParams, copyPadParams);
                    AscendC::PipeBarrier<PIPE_ALL>();

                    inQueue.EnQue(vecIn);
                    inQueue.template DeQue<float>();


                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::ReduceSum<float>(vecOut, vecIn[0], vecIn[0], itemNum);
                    AscendC::PipeBarrier<PIPE_ALL>();

                    outQueue.EnQue(vecOut);
                    outQueue.template DeQue<float>();

                    AscendC::PipeBarrier<PIPE_ALL>();
                    dsinkCalc += vecOut.GetValue(0);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    copyOffset += itemNum;
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                dsinkGm.SetValue(n1temp, -dsinkCalc);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            inQueue.FreeTensor(vecIn);
            outQueue.FreeTensor(vecOut);
        }
    }
    __aicore__ inline void NZ2ND(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, uint64_t sLen,
                          uint64_t ubOffset, uint64_t srcUbOffset, int64_t dAlign)
    {
        /*
        Func:
        将NZ转为ND
        */
        CopyRepeatParams nz2ndParams;
        nz2ndParams.srcStride = sLen * C0_SIZE / cal_block_num + 1;
        nz2ndParams.dstStride = C0_SIZE / cal_block_num;
        nz2ndParams.srcRepeatSize = C0_SIZE / cal_block_num;
        nz2ndParams.dstRepeatSize = dAlign / cal_block_num;

        uint16_t c0_repeat = C0_SIZE / cal_block_num;
        uint16_t c1_repeat = dAlign / C0_SIZE / VEC_REPEAT;
        uint16_t c1_remain = dAlign / C0_SIZE % VEC_REPEAT;
        uint16_t n_repeat = sLen;
        for (uint16_t i = 0; i < c0_repeat; ++i) {
            for (uint16_t j = 0; j < c1_repeat; ++j) {
                Copy(dstTensor[ubOffset + i * cal_block_num + j * VEC_REPEAT * C0_SIZE],
                    srcTensor[srcUbOffset + i * cal_block_num + j * VEC_REPEAT * (sLen * C0_SIZE + cal_block_num)],
                    VEC_REPEAT * cal_block_num, n_repeat, nz2ndParams);
            }
            if (c1_remain > 0) {
                Copy(dstTensor[ubOffset + i * cal_block_num + c1_repeat * VEC_REPEAT * C0_SIZE],
                    srcTensor[srcUbOffset + i * cal_block_num + c1_repeat * VEC_REPEAT * (sLen * C0_SIZE + cal_block_num)],
                    VEC_REPEAT * c1_remain, n_repeat, nz2ndParams);
            }
        }
    }
    __aicore__ inline void NZVecClc(GlobalTensor<float> srcGm, GlobalTensor<OUT_TYPE> dstGm, uint64_t dataSize,
                             GM_ADDR seqS, int64_t curG, int64_t &curS, bool needMuls, int64_t flag, int64_t d, int64_t dAlign)
    {
        if (dataSize == 0) {
            return;
        }
        event_t mte2WaitVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        event_t mte2WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCommon;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueCommon;
        TBuf<> tmpBufCommon;
        event_t curEventId;
        if (flag) {
            inQueueCommon = inQueuePing;
            outQueueCommon = outQueuePing;
            tmpBufCommon = tmpBufPing;
            curEventId = mte2WaitVPing;
        } else {
            inQueueCommon = inQueuePong;
            outQueueCommon = outQueuePong;
            tmpBufCommon = tmpBufPong;
            curEventId = mte2WaitVPong;
        }

        LocalTensor<float> vecIn = inQueueCommon.template AllocTensor<float>();
        LocalTensor<float> tmpTensor = tmpBufCommon.template Get<float>();
        LocalTensor<OUT_TYPE> vecOut = outQueueCommon.template AllocTensor<OUT_TYPE>();

        uint32_t sClcSize = dataSize / (d);

        uint64_t sLen = (sIdx + sClcSize) > curS ? (curS - sIdx) : sClcSize;
        sLen = sLen > 255 ? 255 : sLen;
        uint64_t dataLen = sLen * dAlign;

        uint64_t ubOffset = 0;
        uint64_t inUbOffset = 0;

        while (sClcSize > 0) {
            // Nz copy In
            AscendC::DataCopyExtParams intriParams;
            intriParams.blockCount = dAlign / C0_SIZE;
            intriParams.blockLen = sLen * C0_SIZE * sizeof(float);
            intriParams.srcStride = curS * C0_SIZE * sizeof(float) - intriParams.blockLen;
            intriParams.dstStride = 1; // 间隔一个block，防止bank冲突
            intriParams.rsv = 0;
            DataCopyPad(vecIn[inUbOffset], srcGm[copyInSrcOffset], intriParams, {false, 0, 0, 0});
            sClcSize = sClcSize - sLen;

            inQueueCommon.EnQue(vecIn);
            inQueueCommon.template DeQue<float>();

            if constexpr (!AscendC::IsSameType<OUT_TYPE, float>::value) {
                NZ2ND(tmpTensor, vecIn, sLen, ubOffset, inUbOffset, dAlign);
            } else {
                NZ2ND(vecOut, vecIn, sLen, ubOffset, inUbOffset, dAlign);
            }

            if (sClcSize <= 0) {
                inQueueCommon.FreeTensor(vecIn);
            }

            AscendC::PipeBarrier<PIPE_V>();
            if (needMuls) {
                if constexpr (!AscendC::IsSameType<OUT_TYPE, float>::value) {
                    Muls(tmpTensor[ubOffset], tmpTensor[ubOffset], (float)tilingData->postTilingData.scaleValue,
                    sLen * dAlign);
                } else {
                    Muls(vecOut[ubOffset], vecOut[ubOffset], (float)tilingData->postTilingData.scaleValue,
                    sLen * dAlign);
                }

                AscendC::PipeBarrier<PIPE_V>();
            }

            if constexpr (!AscendC::IsSameType<OUT_TYPE, float>::value) {
                Cast(vecOut[ubOffset], tmpTensor[ubOffset], RoundMode::CAST_ROUND, sLen * dAlign);
                AscendC::PipeBarrier<PIPE_V>();
            }

            outQueueCommon.EnQue(vecOut);
            outQueueCommon.template DeQue<OUT_TYPE>();

            if constexpr (LAYOUT == TND) {
                DataCopyPad(dstGm[copyOutDstOffset], vecOut[ubOffset],
                        {static_cast<uint16_t>(dataLen / (dAlign)), static_cast<uint32_t>(d * sizeof(OUT_TYPE)), 0,
                        static_cast<uint32_t>((n2 * curG * d - d) * sizeof(OUT_TYPE)), 0});
            } else {
                DataCopyPad(dstGm[copyOutDstOffset], vecOut[ubOffset],
                        {static_cast<uint16_t>(dataLen / (dAlign)), static_cast<uint32_t>(d * sizeof(OUT_TYPE)), 0,
                        static_cast<uint32_t>(copyOutDstStride * sizeof(OUT_TYPE)), 0});
            }
            if (sLen + sIdx < curS) {
                sIdx += sLen;
            } else if (nIdx == n2 * curG - 1) {
                sIdx = 0;
                nIdx = 0;
                bIdx++;
                if constexpr (LAYOUT == TND) {
                    scrOffsetBase += curS * n2 * curG * dAlign;
                    dstOffsetBase += curS * n2 * curG * d;
                    if (bIdx < b) {
                        curS = ((__gm__ int64_t *)seqS)[bIdx] - ((__gm__ int64_t *)seqS)[bIdx - 1];
                    } else {
                        curS = 0;
                    }
                }
            } else {
                sIdx = 0;
                nIdx++;
            }
            if constexpr (LAYOUT == TND) {
                copyInSrcOffset = scrOffsetBase + nIdx * curS * dAlign + sIdx * C0_SIZE;
                copyOutDstOffset = dstOffsetBase + (sIdx * n2 * curG + nIdx) * d;
            } else {
                ComputeDataCopyOffset(curG, curS, d, dAlign);
            }
            ubOffset += dataLen;
            inUbOffset += dataLen + dAlign / C0_SIZE * cal_block_num;
            sLen = sClcSize > curS ? curS : sClcSize;
            sLen = sLen > (curS - sIdx) ? (curS - sIdx) : sLen;
            sLen = sLen > 255 ? 255 : sLen;
            dataLen = sLen * dAlign;
            if ((sLen > 0) && (inUbOffset + dataLen + dAlign / C0_SIZE * cal_block_num) * sizeof(float) >
                                ubBaseSize * 2 + nzReservedSize) {
                inUbOffset = 0;
                AscendC::SetFlag<HardEvent::V_MTE2>(static_cast<int32_t>(curEventId));
                AscendC::WaitFlag<HardEvent::V_MTE2>(static_cast<int32_t>(curEventId));
            }
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(mte2WaitVPing);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(mte2WaitVPong);
        outQueueCommon.FreeTensor(vecOut);
    }
    __aicore__ inline void NZProcess()
    {
        uint64_t qBegin = cBlockIdx * qPostBlockFactor * qPostBaseNum;
        uint64_t qEnd = (cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum;
        if (((cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum) > qPostBlockTotal) {
            qEnd = qPostBlockTotal;
        }

        InitIndex(qBegin, g, s1, actual_seq_qlen_addr, d, dAlign);

        for (uint64_t i = qBegin; i < qEnd; i = i + 2 * qPostBaseNum) {
            uint64_t dataSize = i + qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
            NZVecClc(dqWorkSpaceGm, dqGm, dataSize, actual_seq_qlen_addr, g, s1, true, 0, d, dAlign);
            uint64_t dataSize1 = i + 2 * qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
            dataSize1 = i + qPostBaseNum >= qPostBlockTotal ? 0 : dataSize1;
            NZVecClc(dqWorkSpaceGm, dqGm, dataSize1, actual_seq_qlen_addr, g, s1, true, 1, d, dAlign);
        }

        if constexpr (HAS_ROPE == ENABLE) {
            uint64_t qRopeBegin = cBlockIdx * qRopePostBlockFactor * qRopePostBaseNum;
            uint64_t qRopeEnd = (cBlockIdx + 1) * qRopePostBlockFactor * qRopePostBaseNum;
            if (((cBlockIdx + 1) * qRopePostBlockFactor * qRopePostBaseNum) > qRopePostBlockTotal) {
                qRopeEnd = qRopePostBlockTotal;
            }
    
            InitIndex(qRopeBegin, g, s1, actual_seq_qlen_addr, rope_d, rope_dAlign);
    
            for (uint64_t i = qRopeBegin; i < qRopeEnd; i = i + 2 * qRopePostBaseNum) {
                uint64_t dataSize = i + qRopePostBaseNum < qRopePostBlockTotal ? qRopePostBaseNum : qRopePostTailNum;
                NZVecClc(dqRopeWorkSpaceGm, dqRopeGm, dataSize, actual_seq_qlen_addr, g, s1, true, 0, rope_d, rope_dAlign);
                uint64_t dataSize1 = i + 2 * qRopePostBaseNum < qRopePostBlockTotal ? qRopePostBaseNum : qRopePostTailNum;
                dataSize1 = i + qRopePostBaseNum >= qRopePostBlockTotal ? 0 : dataSize1;
                NZVecClc(dqRopeWorkSpaceGm, dqRopeGm, dataSize1, actual_seq_qlen_addr, g, s1, true, 1, rope_d, rope_dAlign);
            }
        }

        // init k
        uint64_t kvBegin = cBlockIdx * kvPostBlockFactor * kvPostBaseNum;
        uint64_t kvEnd = (cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum;
        if (((cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum) > kvPostBlockTotal) {
            kvEnd = kvPostBlockTotal;
        }
        InitIndex(kvBegin, 1, s2, actual_seq_kvlen_addr, d, dAlign);
        for (uint64_t i = kvBegin; i < kvEnd; i = i + 2 * kvPostBaseNum) {
            uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
            NZVecClc(dkWorkSpaceGm, dkGm, dataSize, actual_seq_kvlen_addr, 1, s2, true, 0, d, dAlign);
            uint64_t dataSize1 = i + 2 * kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
            dataSize1 = i + kvPostBaseNum >= kvPostBlockTotal ? 0 : dataSize1;
            NZVecClc(dkWorkSpaceGm, dkGm, dataSize1, actual_seq_kvlen_addr, 1, s2, true, 1, d, dAlign);
        }

        if constexpr (HAS_ROPE == ENABLE) {
            // init kRope
            uint64_t kRopeBegin = cBlockIdx * kRopePostBlockFactor * kRopePostBaseNum;
            uint64_t kRopeEnd = (cBlockIdx + 1) * kRopePostBlockFactor * kRopePostBaseNum;
            if (((cBlockIdx + 1) * kRopePostBlockFactor * kRopePostBaseNum) > kRopePostBlockTotal) {
                kRopeEnd = kRopePostBlockTotal;
            }
            InitIndex(kRopeBegin, 1, s2, actual_seq_kvlen_addr, rope_d, rope_dAlign);
            for (uint64_t i = kRopeBegin; i < kRopeEnd; i = i + 2 * kRopePostBaseNum) {
                uint64_t dataSize = i + kRopePostBaseNum < kRopePostBlockTotal ? kRopePostBaseNum : kRopePostTailNum;
                NZVecClc(dkRopeWorkSpaceGm, dkRopeGm, dataSize, actual_seq_kvlen_addr, 1, s2, true, 0, rope_d, rope_dAlign);
                uint64_t dataSize1 = i + 2 * kRopePostBaseNum < kRopePostBlockTotal ? kRopePostBaseNum : kRopePostTailNum;
                dataSize1 = i + kRopePostBaseNum >= kRopePostBlockTotal ? 0 : dataSize1;
                NZVecClc(dkRopeWorkSpaceGm, dkRopeGm, dataSize1, actual_seq_kvlen_addr, 1, s2, true, 1, rope_d, rope_dAlign);
            }
        }

        // init v
        uint64_t vBegin = cBlockIdx * vPostBlockFactor * vPostBaseNum;
        uint64_t vEnd = (cBlockIdx + 1) * vPostBlockFactor * vPostBaseNum;
        if (((cBlockIdx + 1) * vPostBlockFactor * vPostBaseNum) > vPostBlockTotal) {
            vEnd = vPostBlockTotal;
        }
        if constexpr (CAST_DV) {
            InitIndex(vBegin, 1, s2, actual_seq_kvlen_addr, value_d, value_dAlign);
            for (uint64_t i = vBegin; i < vEnd; i = i + 2 * vPostBaseNum) {
                uint64_t dataSize = i + vPostBaseNum < vPostBlockTotal ? vPostBaseNum : vPostTailNum;
                NZVecClc(dvWorkSpaceGm, dvGm, dataSize, actual_seq_kvlen_addr, 1, s2, false, 0, value_d, value_dAlign);
                uint64_t dataSize1 = i + 2 * vPostBaseNum < vPostBlockTotal ? vPostBaseNum : vPostTailNum;
                dataSize1 = i + vPostBaseNum >= vPostBlockTotal ? 0 : dataSize1;
                NZVecClc(dvWorkSpaceGm, dvGm, dataSize1, actual_seq_kvlen_addr, 1, s2, false, 1, value_d, value_dAlign);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        if (unlikely(tilingData->s1s2BNGS1S2BaseParams.sink == 1)) {
            AscendC::LocalTensor<float> vecIn = inQueuePing.template AllocTensor<float>();
            AscendC::LocalTensor<float> vecOut = outQueuePing.template AllocTensor<float>();

            int s1Pad = (tilingData->postTilingData.s1 + 255) / 256 * 256;
            int s2Pad = (tilingData->postTilingData.s2 + 255) / 256 * 256;
            int dataSizePerN1 = tilingData->postTilingData.b * s1Pad * s2Pad / tilingData->postTilingData.baseMN;
            int N1 = tilingData->postTilingData.n2 * tilingData->postTilingData.g;
            int dsinkSumMaxCopyTimePerN1 = (dataSizePerN1 * sizeof(float) + 2 * ubBaseSize -1) / ( 2 * ubBaseSize);
            int tailDsinkSumSize = (dataSizePerN1 * sizeof(float)) % (2 * ubBaseSize);
            int copyOffset = 0;

            for (int n1temp = 0; n1temp < N1; n1temp++) {
                float dsinkCalc = 0.0;
                for (int dsinkSumDataCopyLoop = 0; dsinkSumDataCopyLoop < dsinkSumMaxCopyTimePerN1; dsinkSumDataCopyLoop++) {
                    int currentCopyDataSize = (dsinkSumDataCopyLoop ==  dsinkSumMaxCopyTimePerN1-1) ? tailDsinkSumSize : 2 * ubBaseSize;
                    int itemNum = currentCopyDataSize / sizeof(float);

                    DataCopyExtParams copyParams;
                    copyParams.blockCount = 1;
                    copyParams.blockLen = currentCopyDataSize;
                    copyParams.srcStride = 0;
                    copyParams.dstStride = 0;
                    copyParams.rsv = 0;
                    DataCopyPadExtParams<float> copyPadParams;
                    copyPadParams.isPad = ((itemNum % 8) == 0) ? false : true;
                    copyPadParams.leftPadding = 0;
                    copyPadParams.rightPadding = 8 - (itemNum % 8);
                    copyPadParams.paddingValue = 0.0;
                    AscendC::PipeBarrier<PIPE_ALL>();
                    DataCopyPad(vecIn, dsinksumWorkSpaceGm[copyOffset], copyParams, copyPadParams);
                    AscendC::PipeBarrier<PIPE_ALL>();

                    inQueuePing.EnQue(vecIn);
                    inQueuePing.template DeQue<float>();


                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::ReduceSum<float>(vecOut, vecIn[0], vecIn[0], itemNum);
                    AscendC::PipeBarrier<PIPE_ALL>();

                    outQueuePing.EnQue(vecOut);
                    outQueuePing.template DeQue<float>();

                    AscendC::PipeBarrier<PIPE_ALL>();
                    dsinkCalc += vecOut.GetValue(0);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    copyOffset += itemNum;
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                dsinkGm.SetValue(n1temp, -dsinkCalc);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            inQueuePing.FreeTensor(vecIn);
            outQueuePing.FreeTensor(vecOut);
        }
    }
    __aicore__ inline void ComputeDataCopyOffset(int64_t curG, int64_t &curS, int64_t d, int64_t dAlign)
    {
        // src BNSD
        scrOffsetBase = bIdx * n2 * curS * curG * dAlign;
        copyInSrcOffset = scrOffsetBase + nIdx * curS * dAlign + sIdx * C0_SIZE;

        if constexpr (LAYOUT == BSNGD) {
            // BSND
            copyOutDstStride = n2 * curG * d - d;
            copyOutDstOffset = ((bIdx * curS + sIdx ) * n2 * curG + nIdx) * d;
        } else if constexpr (LAYOUT == SBNGD) {
            // SBND
            copyOutDstStride = b * n2 * curG * d - d;
            copyOutDstOffset = (( sIdx * b + bIdx ) * n2 * curG + nIdx ) * d;
        } else if constexpr (LAYOUT == BNGSD) {
            // BNSD
            copyOutDstStride = 0;
            copyOutDstOffset = ((bIdx * n2 * curG + nIdx )* curS + sIdx) * d;
        }
    }

    constexpr static uint32_t BUFFER_NUM = 1;
    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    // NZ buffer
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePing;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePong;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueuePing;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueuePong;
    TBuf<> tmpBufPing;
    TBuf<> tmpBufPong;

    AscendC::GlobalTensor<OUT_TYPE> dqGm, dkGm, dvGm;
    AscendC::GlobalTensor<float> dsinkGm;

    AscendC::GlobalTensor<OUT_TYPE> dqRopeGm, dkRopeGm;
    // input
    AscendC::GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    AscendC::GlobalTensor<float> dqRopeWorkSpaceGm, dkRopeWorkSpaceGm;
    AscendC::GlobalTensor<float> dsinksumWorkSpaceGm;
    AscendC::GlobalTensor<uint32_t> dsinksumDataSizeGm;

    const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb *__restrict tilingData;
    constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint64_t C0_SIZE = 16;
    constexpr static uint64_t VEC_REPEAT = 8;
    constexpr static uint32_t cal_block_num = 32 / sizeof(float);
    constexpr static uint32_t ENABLE = 1;

    int64_t usedCoreNum;
    int64_t cBlockIdx;
    // query
    int64_t ubBaseSize;
    uint32_t nzReservedSize;
    int64_t qPostBlockFactor;
    uint64_t qPostBlockTotal;
    int64_t qPostBaseNum;
    int64_t qPostTailNum;
    uint64_t qSizeAlign;
    int64_t kvPostBlockFactor;
    uint64_t kvPostBlockTotal;
    int64_t kvPostBaseNum;
    int64_t kvPostTailNum;
    uint64_t kvSizeAlign;

    int64_t qRopePostBlockFactor;
    uint64_t qRopePostBlockTotal;
    int64_t qRopePostBaseNum;
    int64_t qRopePostTailNum;
    uint64_t qRopeSizeAlign;
    int64_t kRopePostBlockFactor;
    uint64_t kRopePostBlockTotal;
    int64_t kRopePostBaseNum;
    int64_t kRopePostTailNum;
    uint64_t kRopeSizeAlign;

    int64_t vPostBlockFactor;
    uint64_t vPostBlockTotal;
    int64_t vPostBaseNum;
    int64_t vPostTailNum;
    uint64_t vSizeAlign;

    // org shape info
    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t dAlign;
    int64_t value_d;
    int64_t value_dAlign;
    int64_t rope_d;
    int64_t rope_dAlign;
    constexpr static uint32_t BNGSD = 0;
    constexpr static uint32_t SBNGD = 1;
    constexpr static uint32_t BSNGD = 2;
    constexpr static uint32_t TND = 3;

    constexpr static uint32_t ND = 0;
    constexpr static uint32_t NZ = 1;

    GM_ADDR actual_seq_qlen_addr;
    GM_ADDR actual_seq_kvlen_addr;

    uint64_t bIdx;
    uint64_t nIdx;
    uint64_t sIdx;

    uint64_t scrOffsetBase = 0;
    uint64_t dstOffsetBase = 0;
    uint64_t copyInSrcOffset = 0;
    uint64_t copyOutDstOffset = 0;
    uint64_t copyOutDstStride = 0;
};

// EXACT the same with FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb except for class name.
// Partial Specialize for FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2
template <typename OUT_TYPE, const bool CAST_DV, const uint32_t LAYOUT,
          const uint32_t INPUT_FORMAT, const uint32_t HAS_ROPE>
class FlashAttentionScoreGradPost<OUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, CAST_DV, LAYOUT,
          INPUT_FORMAT, HAS_ROPE> {
public:
    __aicore__ inline FlashAttentionScoreGradPost(){}
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dqRope, __gm__ uint8_t *dk, __gm__ uint8_t *dkRope, __gm__ uint8_t *dv,
                         __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen,__gm__ uint8_t *dsink,
                         __gm__ uint8_t *workspace, const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *__restrict ordTilingData, TPipe *pipe_in)
    {
        cBlockIdx = GetBlockIdx();

        tilingData = ordTilingData;
        pipe = pipe_in;

        dqGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dq);
        if constexpr (HAS_ROPE == ENABLE) {
            dqRopeGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dqRope);
        }
        dkGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dk);
        if constexpr (HAS_ROPE == ENABLE) {
            dkRopeGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dkRope);
        }
        dvGm.SetGlobalBuffer((__gm__ OUT_TYPE *)dv);

        // tiling_data
        usedCoreNum = tilingData->postTilingData.coreNum;
        ubBaseSize = tilingData->postTilingData.postUbBaseSize;
        nzReservedSize = tilingData->postTilingData.nzReservedSize;
        qPostBlockFactor = tilingData->postTilingData.qPostBlockFactor;
        qPostBlockTotal = tilingData->postTilingData.qPostBlockTotal;
        qPostBaseNum = tilingData->postTilingData.qPostBaseNum;
        qPostTailNum = tilingData->postTilingData.qPostTailNum;
        kvPostBlockFactor = tilingData->postTilingData.kvPostBlockFactor;
        kvPostBlockTotal = tilingData->postTilingData.kvPostBlockTotal;
        kvPostBaseNum = tilingData->postTilingData.kvPostBaseNum;
        kvPostTailNum = tilingData->postTilingData.kvPostTailNum;
        vPostBlockFactor = tilingData->postTilingData.vPostBlockFactor;
        vPostBlockTotal = tilingData->postTilingData.vPostBlockTotal;
        vPostBaseNum = tilingData->postTilingData.vPostBaseNum;
        vPostTailNum = tilingData->postTilingData.vPostTailNum;
        qSizeAlign = tilingData->postTilingData.qSizeAlign;
        kvSizeAlign = tilingData->postTilingData.kvSizeAlign;
        vSizeAlign = tilingData->postTilingData.vSizeAlign;

        if constexpr (HAS_ROPE == ENABLE) {
            qRopePostBlockFactor = tilingData->postTilingData.qRopePostBlockFactor;
            qRopePostBlockTotal = tilingData->postTilingData.qRopePostBlockTotal;
            qRopePostBaseNum = tilingData->postTilingData.qRopePostBaseNum;
            qRopePostTailNum = tilingData->postTilingData.qRopePostTailNum;
            kRopePostBlockFactor = tilingData->postTilingData.kRopePostBlockFactor;
            kRopePostBlockTotal = tilingData->postTilingData.kRopePostBlockTotal;
            kRopePostBaseNum = tilingData->postTilingData.kRopePostBaseNum;
            kRopePostTailNum = tilingData->postTilingData.kRopePostTailNum;
            qRopeSizeAlign = tilingData->postTilingData.qRopeSizeAlign;
            kRopeSizeAlign = tilingData->postTilingData.kRopeSizeAlign;
        }

        if constexpr (INPUT_FORMAT == NZ) {
            b = tilingData->postTilingData.b;
            n2 = tilingData->postTilingData.n2;
            g = tilingData->postTilingData.g;
            s1 = tilingData->postTilingData.s1;
            s2 = tilingData->postTilingData.s2;
            d = tilingData->postTilingData.d;
            dAlign = (d + 15) / 16 * 16;
            value_d = tilingData->postTilingData.value_d;
            value_dAlign = (value_d + 15) / 16 * 16;
            actual_seq_qlen_addr = actual_seq_qlen;
            actual_seq_kvlen_addr = actual_seq_kvlen;
            if constexpr (HAS_ROPE == ENABLE) {
                rope_d = tilingData->postTilingData.rope_d;
                rope_dAlign = (rope_d + 15) / 16 * 16;
            }
        }

        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                    tilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
        if constexpr (HAS_ROPE == ENABLE) {
            dqRopeWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                                tilingData->postTilingData.dqRopeWorkSpaceOffset / sizeof(float));
        }
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                    tilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
        if constexpr (HAS_ROPE == ENABLE) {
            dkRopeWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                                tilingData->postTilingData.dkRopeWorkSpaceOffset / sizeof(float));
        }
        if constexpr (CAST_DV) {
            dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                        tilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));
        }

        if constexpr (INPUT_FORMAT == NZ) {
            pipe->InitBuffer(inQueuePing, 1, ubBaseSize * 2 + nzReservedSize);
            pipe->InitBuffer(inQueuePong, 1, ubBaseSize * 2 + nzReservedSize);
            pipe->InitBuffer(outQueuePing, 1, ubBaseSize);
            pipe->InitBuffer(outQueuePong, 1, ubBaseSize);
            pipe->InitBuffer(tmpBufPing, ubBaseSize * 2);
            pipe->InitBuffer(tmpBufPong, ubBaseSize * 2);
        } else {
            pipe->InitBuffer(inQueue, 1, ubBaseSize * 2);
            pipe->InitBuffer(outQueue, 1, ubBaseSize);
        }
    }
    __aicore__ inline void InitIndex(uint64_t startIdx, int64_t curG, int64_t &curS, GM_ADDR seqS, int64_t d, int64_t dAlign)
    {
        if constexpr (LAYOUT == TND) {
            uint64_t totalLen = 0;
            for (int64_t bDimIdx = 0; bDimIdx < b; bDimIdx++) {
                totalLen = n2 * curG * ((__gm__ int64_t *)seqS)[bDimIdx] * d;
                if (totalLen > startIdx) {
                    bIdx = bDimIdx;
                    curS = (bIdx == 0) ? ((__gm__ int64_t *)seqS)[bIdx] :
                                        (((__gm__ int64_t *)seqS)[bIdx] - ((__gm__ int64_t *)seqS)[bIdx - 1]);
                    uint64_t bTail = startIdx - (totalLen - n2 * curG * curS * d);
                    nIdx = bTail / (curS * d);
                    uint64_t nTail = bTail % (curS * d);
                    sIdx = nTail / (d);
                    break;
                }
            }
            // 计算输入、输出的offset
            dstOffsetBase = totalLen - n2 * curG * curS * d;
            scrOffsetBase = dstOffsetBase / (d) * dAlign;

            copyInSrcOffset = scrOffsetBase + nIdx * curS * dAlign + sIdx * C0_SIZE;
            copyOutDstOffset = dstOffsetBase + (sIdx * n2 * curG + nIdx) * d;
        } else { // 补充offset 计算
            bIdx = startIdx / (n2 * curG * curS * d);
            uint64_t bTail = startIdx % (n2 * curG * curS * d);
            nIdx = bTail / (curS * d);
            uint64_t nTail = bTail % (curS * d);
            sIdx = nTail / (d);
            ComputeDataCopyOffset(curG, curS, d, dAlign);
        }
    }
    __aicore__ inline void Process()
    {
        if constexpr (INPUT_FORMAT == NZ) {
            NZProcess();
            return;
        }
        // init q
        uint64_t qBegin = cBlockIdx * qPostBlockFactor * qPostBaseNum;
        uint64_t qEnd = (cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum;

        if (((cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum) > qPostBlockTotal) {
            qEnd = qPostBlockTotal;
        }
        for (uint64_t i = qBegin; i < qEnd; i = i + qPostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
            AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
            uint64_t dataSize = i + qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
            DataCopy(vecIn, dqWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
            inQueue.EnQue(vecIn);
            inQueue.template DeQue<float>();
            if constexpr (AscendC::IsSameType<OUT_TYPE, float>::value) {
                Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                outQueue.EnQue(vecOut);
                outQueue.template DeQue<OUT_TYPE>();
                DataCopy(dqGm[i], vecOut, (dataSize + 7) / 8 * 8); // dataSize(fp16) align 32B
            } else {
                Muls(vecIn, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                AscendC::PipeBarrier<PIPE_V>();
                Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
                outQueue.EnQue(vecOut);
                outQueue.template DeQue<OUT_TYPE>();
                DataCopy(dqGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
            }
            inQueue.FreeTensor(vecIn);
            outQueue.FreeTensor(vecOut);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        if constexpr (HAS_ROPE == ENABLE) {
            // init qRope
            uint64_t qRopeBegin = cBlockIdx * qRopePostBlockFactor * qRopePostBaseNum;
            uint64_t qRopeEnd = (cBlockIdx + 1) * qRopePostBlockFactor * qRopePostBaseNum;
    
            if (((cBlockIdx + 1) * qRopePostBlockFactor * qRopePostBaseNum) > qRopePostBlockTotal) {
                qRopeEnd = qRopePostBlockTotal;
            }
            for (uint64_t i = qRopeBegin; i < qRopeEnd; i = i + qRopePostBaseNum) {
                AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
                AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
                uint64_t dataSize = i + qRopePostBaseNum < qRopePostBlockTotal ? qRopePostBaseNum : qRopePostTailNum;
                DataCopy(vecIn, dqRopeWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
                inQueue.EnQue(vecIn);
                inQueue.template DeQue<float>();
                if constexpr (AscendC::IsSameType<OUT_TYPE, float>::value) {
                    Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                    outQueue.EnQue(vecOut);
                    outQueue.template DeQue<OUT_TYPE>();
                    DataCopy(dqRopeGm[i], vecOut, (dataSize + 7) / 8 * 8); // dataSize(fp16) align 32B
                } else {
                    Muls(vecIn, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                    AscendC::PipeBarrier<PIPE_V>();
                    Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
                    outQueue.EnQue(vecOut);
                    outQueue.template DeQue<OUT_TYPE>();
                    DataCopy(dqRopeGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
                }
                inQueue.FreeTensor(vecIn);
                outQueue.FreeTensor(vecOut);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        // init k
        uint64_t kvBegin = cBlockIdx * kvPostBlockFactor * kvPostBaseNum;
        uint64_t kvEnd = (cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum;
        if (((cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum) > kvPostBlockTotal) {
            kvEnd = kvPostBlockTotal;
        }

        for (uint64_t i = kvBegin; i < kvEnd; i = i + kvPostBaseNum) {
            AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
            AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
            uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
            DataCopy(vecIn, dkWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
            inQueue.EnQue(vecIn);
            inQueue.template DeQue<float>();
            if constexpr (AscendC::IsSameType<OUT_TYPE, float>::value) {
                Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                outQueue.EnQue(vecOut);
                outQueue.template DeQue<OUT_TYPE>();
                DataCopy(dkGm[i], vecOut, (dataSize + 7) / 8 * 8); // dataSize(fp16) align 32B
            } else {
                Muls(vecIn, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                AscendC::PipeBarrier<PIPE_V>();
                Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
                outQueue.EnQue(vecOut);
                outQueue.template DeQue<OUT_TYPE>();
                DataCopy(dkGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
            }
            inQueue.FreeTensor(vecIn);
            outQueue.FreeTensor(vecOut);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        if constexpr (HAS_ROPE == ENABLE) {
            // init kRope
            uint64_t kRopeBegin = cBlockIdx * kRopePostBlockFactor * kRopePostBaseNum;
            uint64_t kRopeEnd = (cBlockIdx + 1) * kRopePostBlockFactor * kRopePostBaseNum;
            if (((cBlockIdx + 1) * kRopePostBlockFactor * kRopePostBaseNum) > kRopePostBlockTotal) {
                kRopeEnd = kRopePostBlockTotal;
            }
    
            for (uint64_t i = kRopeBegin; i < kRopeEnd; i = i + kRopePostBaseNum) {
                AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
                AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
                uint64_t dataSize = i + kRopePostBaseNum < kRopePostBlockTotal ? kRopePostBaseNum : kRopePostTailNum;
                DataCopy(vecIn, dkRopeWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
                inQueue.EnQue(vecIn);
                inQueue.template DeQue<float>();
                if constexpr (AscendC::IsSameType<OUT_TYPE, float>::value) {
                    Muls(vecOut, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                    outQueue.EnQue(vecOut);
                    outQueue.template DeQue<OUT_TYPE>();
                    DataCopy(dkRopeGm[i], vecOut, (dataSize + 7) / 8 * 8); // dataSize(fp16) align 32B
                } else {
                    Muls(vecIn, vecIn, (float)tilingData->postTilingData.scaleValue, dataSize);
                    AscendC::PipeBarrier<PIPE_V>();
                    Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
                    outQueue.EnQue(vecOut);
                    outQueue.template DeQue<OUT_TYPE>();
                    DataCopy(dkRopeGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
                }
                inQueue.FreeTensor(vecIn);
                outQueue.FreeTensor(vecOut);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        // init v
        uint64_t vBegin = cBlockIdx * vPostBlockFactor * vPostBaseNum;
        uint64_t vEnd = (cBlockIdx + 1) * vPostBlockFactor * vPostBaseNum;
        if (((cBlockIdx + 1) * vPostBlockFactor * vPostBaseNum) > vPostBlockTotal) {
            vEnd = vPostBlockTotal;
        }
        if constexpr (CAST_DV && !AscendC::IsSameType<OUT_TYPE, float>::value) {
            for (uint64_t i = vBegin; i < vEnd; i = i + vPostBaseNum) {
                AscendC::LocalTensor<float> vecIn = inQueue.template AllocTensor<float>();
                AscendC::LocalTensor<OUT_TYPE> vecOut = outQueue.template AllocTensor<OUT_TYPE>();
                uint64_t dataSize = i + vPostBaseNum < vPostBlockTotal ? vPostBaseNum : vPostTailNum;
                DataCopy(vecIn, dvWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
                inQueue.EnQue(vecIn);
                inQueue.template DeQue<float>();
                Cast(vecOut, vecIn, AscendC::RoundMode::CAST_ROUND, dataSize);
                outQueue.EnQue(vecOut);
                outQueue.template DeQue<OUT_TYPE>();
                DataCopy(dvGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
                inQueue.FreeTensor(vecIn);
                outQueue.FreeTensor(vecOut);
            }
        }
    }
    __aicore__ inline void NZ2ND(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, uint64_t sLen,
                          uint64_t ubOffset, uint64_t srcUbOffset, int64_t dAlign)
    {
        /*
        Func:
        将NZ转为ND
        */
        CopyRepeatParams nz2ndParams;
        nz2ndParams.srcStride = sLen * C0_SIZE / cal_block_num + 1;
        nz2ndParams.dstStride = C0_SIZE / cal_block_num;
        nz2ndParams.srcRepeatSize = C0_SIZE / cal_block_num;
        nz2ndParams.dstRepeatSize = dAlign / cal_block_num;

        uint16_t c0_repeat = C0_SIZE / cal_block_num;
        uint16_t c1_repeat = dAlign / C0_SIZE / VEC_REPEAT;
        uint16_t c1_remain = dAlign / C0_SIZE % VEC_REPEAT;
        uint16_t n_repeat = sLen;
        for (uint16_t i = 0; i < c0_repeat; ++i) {
            for (uint16_t j = 0; j < c1_repeat; ++j) {
                Copy(dstTensor[ubOffset + i * cal_block_num + j * VEC_REPEAT * C0_SIZE],
                    srcTensor[srcUbOffset + i * cal_block_num + j * VEC_REPEAT * (sLen * C0_SIZE + cal_block_num)],
                    VEC_REPEAT * cal_block_num, n_repeat, nz2ndParams);
            }
            if (c1_remain > 0) {
                Copy(dstTensor[ubOffset + i * cal_block_num + c1_repeat * VEC_REPEAT * C0_SIZE],
                    srcTensor[srcUbOffset + i * cal_block_num + c1_repeat * VEC_REPEAT * (sLen * C0_SIZE + cal_block_num)],
                    VEC_REPEAT * c1_remain, n_repeat, nz2ndParams);
            }
        }
    }
    __aicore__ inline void NZVecClc(GlobalTensor<float> srcGm, GlobalTensor<OUT_TYPE> dstGm, uint64_t dataSize,
                             GM_ADDR seqS, int64_t curG, int64_t &curS, bool needMuls, int64_t flag, int64_t d, int64_t dAlign)
    {
        if (dataSize == 0) {
            return;
        }
        event_t mte2WaitVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        event_t mte2WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCommon;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueCommon;
        TBuf<> tmpBufCommon;
        event_t curEventId;
        if (flag) {
            inQueueCommon = inQueuePing;
            outQueueCommon = outQueuePing;
            tmpBufCommon = tmpBufPing;
            curEventId = mte2WaitVPing;
        } else {
            inQueueCommon = inQueuePong;
            outQueueCommon = outQueuePong;
            tmpBufCommon = tmpBufPong;
            curEventId = mte2WaitVPong;
        }

        LocalTensor<float> vecIn = inQueueCommon.template AllocTensor<float>();
        LocalTensor<float> tmpTensor = tmpBufCommon.template Get<float>();
        LocalTensor<OUT_TYPE> vecOut = outQueueCommon.template AllocTensor<OUT_TYPE>();

        uint32_t sClcSize = dataSize / (d);

        uint64_t sLen = (sIdx + sClcSize) > curS ? (curS - sIdx) : sClcSize;
        sLen = sLen > 255 ? 255 : sLen;
        uint64_t dataLen = sLen * dAlign;

        uint64_t ubOffset = 0;
        uint64_t inUbOffset = 0;

        while (sClcSize > 0) {
            // Nz copy In
            AscendC::DataCopyExtParams intriParams;
            intriParams.blockCount = dAlign / C0_SIZE;
            intriParams.blockLen = sLen * C0_SIZE * sizeof(float);
            intriParams.srcStride = curS * C0_SIZE * sizeof(float) - intriParams.blockLen;
            intriParams.dstStride = 1; // 间隔一个block，防止bank冲突
            intriParams.rsv = 0;
            DataCopyPad(vecIn[inUbOffset], srcGm[copyInSrcOffset], intriParams, {false, 0, 0, 0});
            sClcSize = sClcSize - sLen;

            inQueueCommon.EnQue(vecIn);
            inQueueCommon.template DeQue<float>();

            if constexpr (!AscendC::IsSameType<OUT_TYPE, float>::value) {
                NZ2ND(tmpTensor, vecIn, sLen, ubOffset, inUbOffset, dAlign);
            } else {
                NZ2ND(vecOut, vecIn, sLen, ubOffset, inUbOffset, dAlign);
            }

            if (sClcSize <= 0) {
                inQueueCommon.FreeTensor(vecIn);
            }

            AscendC::PipeBarrier<PIPE_V>();
            if (needMuls) {
                if constexpr (!AscendC::IsSameType<OUT_TYPE, float>::value) {
                    Muls(tmpTensor[ubOffset], tmpTensor[ubOffset], (float)tilingData->postTilingData.scaleValue,
                    sLen * dAlign);
                } else {
                    Muls(vecOut[ubOffset], vecOut[ubOffset], (float)tilingData->postTilingData.scaleValue,
                    sLen * dAlign);
                }

                AscendC::PipeBarrier<PIPE_V>();
            }

            if constexpr (!AscendC::IsSameType<OUT_TYPE, float>::value) {
                Cast(vecOut[ubOffset], tmpTensor[ubOffset], RoundMode::CAST_ROUND, sLen * dAlign);
                AscendC::PipeBarrier<PIPE_V>();
            }

            outQueueCommon.EnQue(vecOut);
            outQueueCommon.template DeQue<OUT_TYPE>();

            if constexpr (LAYOUT == TND) {
                DataCopyPad(dstGm[copyOutDstOffset], vecOut[ubOffset],
                        {static_cast<uint16_t>(dataLen / (dAlign)), static_cast<uint32_t>(d * sizeof(OUT_TYPE)), 0,
                        static_cast<uint32_t>((n2 * curG * d - d) * sizeof(OUT_TYPE)), 0});
            } else {
                DataCopyPad(dstGm[copyOutDstOffset], vecOut[ubOffset],
                        {static_cast<uint16_t>(dataLen / (dAlign)), static_cast<uint32_t>(d * sizeof(OUT_TYPE)), 0,
                        static_cast<uint32_t>(copyOutDstStride * sizeof(OUT_TYPE)), 0});
            }
            if (sLen + sIdx < curS) {
                sIdx += sLen;
            } else if (nIdx == n2 * curG - 1) {
                sIdx = 0;
                nIdx = 0;
                bIdx++;
                if constexpr (LAYOUT == TND) {
                    scrOffsetBase += curS * n2 * curG * dAlign;
                    dstOffsetBase += curS * n2 * curG * d;
                    if (bIdx < b) {
                        curS = ((__gm__ int64_t *)seqS)[bIdx] - ((__gm__ int64_t *)seqS)[bIdx - 1];
                    } else {
                        curS = 0;
                    }
                }
            } else {
                sIdx = 0;
                nIdx++;
            }
            if constexpr (LAYOUT == TND) {
                copyInSrcOffset = scrOffsetBase + nIdx * curS * dAlign + sIdx * C0_SIZE;
                copyOutDstOffset = dstOffsetBase + (sIdx * n2 * curG + nIdx) * d;
            } else {
                ComputeDataCopyOffset(curG, curS, d, dAlign);
            }
            ubOffset += dataLen;
            inUbOffset += dataLen + dAlign / C0_SIZE * cal_block_num;
            sLen = sClcSize > curS ? curS : sClcSize;
            sLen = sLen > (curS - sIdx) ? (curS - sIdx) : sLen;
            sLen = sLen > 255 ? 255 : sLen;
            dataLen = sLen * dAlign;
            if ((sLen > 0) && (inUbOffset + dataLen + dAlign / C0_SIZE * cal_block_num) * sizeof(float) >
                                ubBaseSize * 2 + nzReservedSize) {
                inUbOffset = 0;
                AscendC::SetFlag<HardEvent::V_MTE2>(static_cast<int32_t>(curEventId));
                AscendC::WaitFlag<HardEvent::V_MTE2>(static_cast<int32_t>(curEventId));
            }
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(mte2WaitVPing);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(mte2WaitVPong);
        outQueueCommon.FreeTensor(vecOut);
    }
    __aicore__ inline void NZProcess()
    {
        uint64_t qBegin = cBlockIdx * qPostBlockFactor * qPostBaseNum;
        uint64_t qEnd = (cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum;
        if (((cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum) > qPostBlockTotal) {
            qEnd = qPostBlockTotal;
        }

        InitIndex(qBegin, g, s1, actual_seq_qlen_addr, d, dAlign);

        for (uint64_t i = qBegin; i < qEnd; i = i + 2 * qPostBaseNum) {
            uint64_t dataSize = i + qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
            NZVecClc(dqWorkSpaceGm, dqGm, dataSize, actual_seq_qlen_addr, g, s1, true, 0, d, dAlign);
            uint64_t dataSize1 = i + 2 * qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
            dataSize1 = i + qPostBaseNum >= qPostBlockTotal ? 0 : dataSize1;
            NZVecClc(dqWorkSpaceGm, dqGm, dataSize1, actual_seq_qlen_addr, g, s1, true, 1, d, dAlign);
        }

        if constexpr (HAS_ROPE == ENABLE) {
            uint64_t qRopeBegin = cBlockIdx * qRopePostBlockFactor * qRopePostBaseNum;
            uint64_t qRopeEnd = (cBlockIdx + 1) * qRopePostBlockFactor * qRopePostBaseNum;
            if (((cBlockIdx + 1) * qRopePostBlockFactor * qRopePostBaseNum) > qRopePostBlockTotal) {
                qRopeEnd = qRopePostBlockTotal;
            }
    
            InitIndex(qRopeBegin, g, s1, actual_seq_qlen_addr, rope_d, rope_dAlign);
    
            for (uint64_t i = qRopeBegin; i < qRopeEnd; i = i + 2 * qRopePostBaseNum) {
                uint64_t dataSize = i + qRopePostBaseNum < qRopePostBlockTotal ? qRopePostBaseNum : qRopePostTailNum;
                NZVecClc(dqRopeWorkSpaceGm, dqRopeGm, dataSize, actual_seq_qlen_addr, g, s1, true, 0, rope_d, rope_dAlign);
                uint64_t dataSize1 = i + 2 * qRopePostBaseNum < qRopePostBlockTotal ? qRopePostBaseNum : qRopePostTailNum;
                dataSize1 = i + qRopePostBaseNum >= qRopePostBlockTotal ? 0 : dataSize1;
                NZVecClc(dqRopeWorkSpaceGm, dqRopeGm, dataSize1, actual_seq_qlen_addr, g, s1, true, 1, rope_d, rope_dAlign);
            }
        }

        // init k
        uint64_t kvBegin = cBlockIdx * kvPostBlockFactor * kvPostBaseNum;
        uint64_t kvEnd = (cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum;
        if (((cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum) > kvPostBlockTotal) {
            kvEnd = kvPostBlockTotal;
        }
        InitIndex(kvBegin, 1, s2, actual_seq_kvlen_addr, d, dAlign);
        for (uint64_t i = kvBegin; i < kvEnd; i = i + 2 * kvPostBaseNum) {
            uint64_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
            NZVecClc(dkWorkSpaceGm, dkGm, dataSize, actual_seq_kvlen_addr, 1, s2, true, 0, d, dAlign);
            uint64_t dataSize1 = i + 2 * kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
            dataSize1 = i + kvPostBaseNum >= kvPostBlockTotal ? 0 : dataSize1;
            NZVecClc(dkWorkSpaceGm, dkGm, dataSize1, actual_seq_kvlen_addr, 1, s2, true, 1, d, dAlign);
        }

        if constexpr (HAS_ROPE == ENABLE) {
            // init kRope
            uint64_t kRopeBegin = cBlockIdx * kRopePostBlockFactor * kRopePostBaseNum;
            uint64_t kRopeEnd = (cBlockIdx + 1) * kRopePostBlockFactor * kRopePostBaseNum;
            if (((cBlockIdx + 1) * kRopePostBlockFactor * kRopePostBaseNum) > kRopePostBlockTotal) {
                kRopeEnd = kRopePostBlockTotal;
            }
            InitIndex(kRopeBegin, 1, s2, actual_seq_kvlen_addr, rope_d, rope_dAlign);
            for (uint64_t i = kRopeBegin; i < kRopeEnd; i = i + 2 * kRopePostBaseNum) {
                uint64_t dataSize = i + kRopePostBaseNum < kRopePostBlockTotal ? kRopePostBaseNum : kRopePostTailNum;
                NZVecClc(dkRopeWorkSpaceGm, dkRopeGm, dataSize, actual_seq_kvlen_addr, 1, s2, true, 0, rope_d, rope_dAlign);
                uint64_t dataSize1 = i + 2 * kRopePostBaseNum < kRopePostBlockTotal ? kRopePostBaseNum : kRopePostTailNum;
                dataSize1 = i + kRopePostBaseNum >= kRopePostBlockTotal ? 0 : dataSize1;
                NZVecClc(dkRopeWorkSpaceGm, dkRopeGm, dataSize1, actual_seq_kvlen_addr, 1, s2, true, 1, rope_d, rope_dAlign);
            }
        }

        // init v
        uint64_t vBegin = cBlockIdx * vPostBlockFactor * vPostBaseNum;
        uint64_t vEnd = (cBlockIdx + 1) * vPostBlockFactor * vPostBaseNum;
        if (((cBlockIdx + 1) * vPostBlockFactor * vPostBaseNum) > vPostBlockTotal) {
            vEnd = vPostBlockTotal;
        }
        if constexpr (CAST_DV) {
            InitIndex(vBegin, 1, s2, actual_seq_kvlen_addr, value_d, value_dAlign);
            for (uint64_t i = vBegin; i < vEnd; i = i + 2 * vPostBaseNum) {
                uint64_t dataSize = i + vPostBaseNum < vPostBlockTotal ? vPostBaseNum : vPostTailNum;
                NZVecClc(dvWorkSpaceGm, dvGm, dataSize, actual_seq_kvlen_addr, 1, s2, false, 0, value_d, value_dAlign);
                uint64_t dataSize1 = i + 2 * vPostBaseNum < vPostBlockTotal ? vPostBaseNum : vPostTailNum;
                dataSize1 = i + vPostBaseNum >= vPostBlockTotal ? 0 : dataSize1;
                NZVecClc(dvWorkSpaceGm, dvGm, dataSize1, actual_seq_kvlen_addr, 1, s2, false, 1, value_d, value_dAlign);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void ComputeDataCopyOffset(int64_t curG, int64_t &curS, int64_t d, int64_t dAlign)
    {
        // src BNSD
        scrOffsetBase = bIdx * n2 * curS * curG * dAlign;
        copyInSrcOffset = scrOffsetBase + nIdx * curS * dAlign + sIdx * C0_SIZE;

        if constexpr (LAYOUT == BSNGD) {
            // BSND
            copyOutDstStride = n2 * curG * d - d;
            copyOutDstOffset = ((bIdx * curS + sIdx ) * n2 * curG + nIdx) * d;
        } else if constexpr (LAYOUT == SBNGD) {
            // SBND
            copyOutDstStride = b * n2 * curG * d - d;
            copyOutDstOffset = (( sIdx * b + bIdx ) * n2 * curG + nIdx ) * d;
        } else if constexpr (LAYOUT == BNGSD) {
            // BNSD
            copyOutDstStride = 0;
            copyOutDstOffset = ((bIdx * n2 * curG + nIdx )* curS + sIdx) * d;
        }
    }

    constexpr static uint32_t BUFFER_NUM = 1;
    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    // NZ buffer
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePing;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePong;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueuePing;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueuePong;
    TBuf<> tmpBufPing;
    TBuf<> tmpBufPong;

    AscendC::GlobalTensor<OUT_TYPE> dqGm, dkGm, dvGm;
    AscendC::GlobalTensor<OUT_TYPE> dqRopeGm, dkRopeGm;
    // input
    AscendC::GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    AscendC::GlobalTensor<float> dqRopeWorkSpaceGm, dkRopeWorkSpaceGm;

    const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *__restrict tilingData;
    constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint64_t C0_SIZE = 16;
    constexpr static uint64_t VEC_REPEAT = 8;
    constexpr static uint32_t cal_block_num = 32 / sizeof(float);
    constexpr static uint32_t ENABLE = 1;

    int64_t usedCoreNum;
    int64_t cBlockIdx;
    // query
    int64_t ubBaseSize;
    uint32_t nzReservedSize;
    int64_t qPostBlockFactor;
    uint64_t qPostBlockTotal;
    int64_t qPostBaseNum;
    int64_t qPostTailNum;
    uint64_t qSizeAlign;
    int64_t kvPostBlockFactor;
    uint64_t kvPostBlockTotal;
    int64_t kvPostBaseNum;
    int64_t kvPostTailNum;
    uint64_t kvSizeAlign;

    int64_t qRopePostBlockFactor;
    uint64_t qRopePostBlockTotal;
    int64_t qRopePostBaseNum;
    int64_t qRopePostTailNum;
    uint64_t qRopeSizeAlign;
    int64_t kRopePostBlockFactor;
    uint64_t kRopePostBlockTotal;
    int64_t kRopePostBaseNum;
    int64_t kRopePostTailNum;
    uint64_t kRopeSizeAlign;

    int64_t vPostBlockFactor;
    uint64_t vPostBlockTotal;
    int64_t vPostBaseNum;
    int64_t vPostTailNum;
    uint64_t vSizeAlign;

    // org shape info
    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t dAlign;
    int64_t value_d;
    int64_t value_dAlign;
    int64_t rope_d = 0;
    int64_t rope_dAlign = 0;
    constexpr static uint32_t BNGSD = 0;
    constexpr static uint32_t SBNGD = 1;
    constexpr static uint32_t BSNGD = 2;
    constexpr static uint32_t TND = 3;

    constexpr static uint32_t ND = 0;
    constexpr static uint32_t NZ = 1;

    GM_ADDR actual_seq_qlen_addr;
    GM_ADDR actual_seq_kvlen_addr;

    uint64_t bIdx;
    uint64_t nIdx;
    uint64_t sIdx;

    uint64_t scrOffsetBase = 0;
    uint64_t dstOffsetBase = 0;
    uint64_t copyInSrcOffset = 0;
    uint64_t copyOutDstOffset = 0;
    uint64_t copyOutDstStride = 0;
};

#endif // _FLASH_ATTENTION_SCORE_GRAD_POST_H_
