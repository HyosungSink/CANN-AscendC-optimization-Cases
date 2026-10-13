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
 * \file flash_attention_score_grad_sfmg.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_SFMG_KERNEL_H_
#define FLASH_ATTENTION_SCORE_GRAD_SFMG_KERNEL_H_

#include "kernel_operator.h"

template <typename T1, typename T2, typename TILING_TYPE, const uint32_t INPUT_LAYOUT>
class FlashAttentionScoreGradSfmg {
public:
    __aicore__ inline FlashAttentionScoreGradSfmg(){};
    __aicore__ inline void Init(__gm__ uint8_t *dy, __gm__ uint8_t *attenIn, __gm__ uint8_t *actual_seq_qlen,
                                __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *drop_mask,
                                __gm__ uint8_t *workspace, const TILING_TYPE *ordTilingData, TPipe *pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void SyncALLCores();
    __aicore__ inline void InitIndex(int64_t startIdx, int64_t &curS, GM_ADDR seqS);
    __aicore__ inline void CopyInSfmg(int64_t leftNburst, int64_t &curS, GM_ADDR seqS);
    __aicore__ inline void DoCopyIn(int64_t curS, int64_t curNBurst, int64_t dstOffset, GM_ADDR seqS);

protected:
    constexpr static uint32_t BNGSD = 0;
    constexpr static uint32_t SBNGD = 1;
    constexpr static uint32_t BSNGD = 2;
    constexpr static uint32_t TND = 3;
    constexpr static int64_t BLOCK_BYTE_SIZE = 32;
    constexpr static int64_t BLOCK_SIZE = 8;
    constexpr static int64_t SFMG_HIGH_PERF_N_FACTOR = 8;
    constexpr static int64_t SFMG_HIGH_PERF_D_FACTOR = 64;

    TPipe *pipe;
    uint32_t cBlockIdx;

    const TILING_TYPE *TilingData;

    GlobalTensor<T2> sfmgWorkspaceGm;
    GlobalTensor<T1> dyGm;
    GlobalTensor<T1> attenInGm;
    TQue<QuePosition::VECIN, 1> input1Que, input2Que;
    TBuf<> cast1Buf, cast2Buf, tmpBuf;
    TQue<QuePosition::VECOUT, 1> out1Que;

    int64_t b;
    int64_t n1;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t dAlign;
    GM_ADDR actual_seq_qlen_addr;

    int64_t bIdx = 0;
    int64_t nIdx = 0;
    int64_t sIdx = 0;

    int64_t dstOffset = 0;
    int64_t transpse_stride = 0;

    LocalTensor<T1> input1Buf;
    LocalTensor<T1> input2Buf;
    LocalTensor<T2> outputBuf;
};

template <typename T1, typename T2, typename TILING_TYPE, const uint32_t INPUT_LAYOUT>
__aicore__ inline void
FlashAttentionScoreGradSfmg<T1, T2, TILING_TYPE, INPUT_LAYOUT>::Init(
    __gm__ uint8_t *dy, __gm__ uint8_t *attenIn, __gm__ uint8_t *actual_seq_qlen,
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *drop_mask, __gm__ uint8_t *workspace,
    const TILING_TYPE *orgTilingData, TPipe *pipe_in)
{
    cBlockIdx = GetBlockIdx();
    TilingData = orgTilingData;
    pipe = pipe_in;
    dyGm.SetGlobalBuffer((__gm__ T1 *)dy);
    attenInGm.SetGlobalBuffer((__gm__ T1 *)attenIn);
    sfmgWorkspaceGm.SetGlobalBuffer((__gm__ T2 *)workspace + TilingData->preSfmgTilingData.sfmgPreBeginAddr / sizeof(T2));

    pipe->InitBuffer(input1Que, 1, TilingData->preSfmgTilingData.inputBufferLen); // 24K
    pipe->InitBuffer(input2Que, 1, TilingData->preSfmgTilingData.inputBufferLen); // 24K
    pipe->InitBuffer(cast1Buf, TilingData->preSfmgTilingData.castBufferLen); // 48K
    pipe->InitBuffer(cast2Buf, TilingData->preSfmgTilingData.castBufferLen); // 48K
    pipe->InitBuffer(out1Que, 1, TilingData->preSfmgTilingData.outputBufferLen);
    pipe->InitBuffer(tmpBuf, TilingData->preSfmgTilingData.tempBufferLen); // 40K - outputBufferLen

    b = TilingData->preSfmgTilingData.b;
    n1 = TilingData->preSfmgTilingData.n2 * TilingData->preSfmgTilingData.g;
    s1 = TilingData->preSfmgTilingData.s1;
    d = TilingData->preSfmgTilingData.value_d;
    int64_t blockNums = BLOCK_BYTE_SIZE / sizeof(T1);
    dAlign = (d + blockNums - 1) / blockNums * blockNums;
    actual_seq_qlen_addr = actual_seq_qlen;

    if constexpr(INPUT_LAYOUT == TND) {
        transpse_stride = (n1 * d - d) * sizeof(T1);
    } else if constexpr(INPUT_LAYOUT == BNGSD){
        transpse_stride = 0;
    } else if constexpr(INPUT_LAYOUT == BSNGD){
        transpse_stride = (n1 * d - d) * sizeof(T1);
    } else if constexpr(INPUT_LAYOUT == SBNGD){
        transpse_stride = (b * n1 * d - d) * sizeof(T1);
    }
}

template <typename T1, typename T2, typename TILING_TYPE, const uint32_t INPUT_LAYOUT>
__aicore__ inline void FlashAttentionScoreGradSfmg<T1, T2, TILING_TYPE, INPUT_LAYOUT>::InitIndex(
    int64_t startIdx, int64_t& curS, GM_ADDR seqS)
{
    if constexpr (INPUT_LAYOUT == TND) {
        int64_t totalLen = 0;
        for (int64_t bDimIdx = bIdx; bDimIdx < b; bDimIdx++) {
            totalLen = n1 * ((__gm__ int64_t *)seqS)[bDimIdx] * d;
            if (totalLen > startIdx) {
                bIdx = bDimIdx;
                curS = (bIdx == 0) ? ((__gm__ int64_t *)seqS)[bIdx] :
                                     (((__gm__ int64_t *)seqS)[bIdx] - ((__gm__ int64_t *)seqS)[bIdx - 1]);
                int64_t bTail = startIdx - (totalLen - n1 * curS * d);
                nIdx = bTail / (curS * d);
                int64_t nTail = bTail % (curS * d);
                sIdx = nTail / d;
                break;
            }
        }
    } else {
        bIdx = startIdx / (n1 * s1 * d);
        int64_t bTail = startIdx % (n1 * s1 * d);
        nIdx = bTail / (s1 * d);
        int64_t nTail = bTail % (s1 * d);
        sIdx = nTail / d;
    }
}

template <typename T1, typename T2, typename TILING_TYPE, const uint32_t INPUT_LAYOUT>
__aicore__ inline void FlashAttentionScoreGradSfmg<T1, T2, TILING_TYPE, INPUT_LAYOUT>::DoCopyIn(
    int64_t curS, int64_t curNBurst, int64_t dstOffset, GM_ADDR seqS)
{
    int64_t srcOffset = 0;
    if constexpr (INPUT_LAYOUT == TND) {
        int64_t bOffset = bIdx == 0 ? 0 : n1 * ((__gm__ int64_t *)seqS)[bIdx - 1] * d;
        srcOffset = bOffset + (sIdx * n1 + nIdx) * d;
    } else {
        if constexpr (INPUT_LAYOUT == BNGSD) {
            srcOffset = bIdx * ( n1 * s1 * d) + nIdx * (s1 * d) + sIdx * d;
        } else if constexpr (INPUT_LAYOUT == BSNGD) {
            srcOffset = bIdx * (s1 * n1 * d) + sIdx * (n1 * d) + nIdx * d;
        } else if constexpr (INPUT_LAYOUT == SBNGD) {
            srcOffset = sIdx * (b * n1 * d) + bIdx * (n1 * d) + nIdx * d;
        }
    }

    DataCopyPad(input1Buf[dstOffset], dyGm[srcOffset],
                {static_cast<uint16_t>(curNBurst), static_cast<uint32_t>(d * sizeof(T1)),
                static_cast<uint32_t>(transpse_stride), 0, 0},
                {true, 0, static_cast<uint8_t>((dAlign - d)), 0});
    DataCopyPad(input2Buf[dstOffset], attenInGm[srcOffset],
                {static_cast<uint16_t>(curNBurst), static_cast<uint32_t>(d * sizeof(T1)),
                static_cast<uint32_t>(transpse_stride), 0, 0},
                {true, 0, static_cast<uint8_t>((dAlign - d)), 0});
}

template <typename T1, typename T2, typename TILING_TYPE, const uint32_t INPUT_LAYOUT>
__aicore__ inline void FlashAttentionScoreGradSfmg<T1, T2, TILING_TYPE, INPUT_LAYOUT>::CopyInSfmg(
    int64_t leftNburst, int64_t &curS, GM_ADDR seqS)
{
    int64_t dstOffset = 0;
    while (leftNburst > 0) {
        int64_t curNburst = 0;
        if (curS - sIdx < leftNburst) { // 需要借N或借B
            curNburst = curS - sIdx;
            DoCopyIn(curS, curNburst, dstOffset, seqS);
            leftNburst = leftNburst - curNburst;
            sIdx = 0;
            if (nIdx < n1 - 1) { // 需要借N
                nIdx += 1;
            } else {
                nIdx = 0;
                if (bIdx < b - 1) { // 需要借B
                    bIdx += 1;
                    if constexpr (INPUT_LAYOUT == TND) {
                        curS = ((__gm__ int64_t *)seqS)[bIdx] - ((__gm__ int64_t *)seqS)[bIdx - 1];
                    } else {
                        curS = s1;
                    }
                } else { // 没有轴可以借了，end
                    leftNburst = 0;
                }
            }
        } else {  // 当前S够用
            curNburst = leftNburst;
            DoCopyIn(curS, curNburst, dstOffset, seqS);
            sIdx = sIdx + leftNburst;
            leftNburst = 0;
        }
        dstOffset = dstOffset + curNburst * dAlign;
    }
}

template <typename T1, typename T2, typename TILING_TYPE, const uint32_t INPUT_LAYOUT>
__aicore__ inline void FlashAttentionScoreGradSfmg<T1, T2, TILING_TYPE, INPUT_LAYOUT>::Process()
{
    AscendC::PipeBarrier<PIPE_ALL>(); // 去掉pre和sfmg之间的SyncALL，这里需要增加pipeALL

    uint32_t usedCoreNums = TilingData->preSfmgTilingData.usedCoreNum;
    if (cBlockIdx < usedCoreNums) {
        LocalTensor<uint8_t> tempBuf = tmpBuf.Get<uint8_t>();
        LocalTensor<T2> sfmgClc1 = cast1Buf.Get<T2>();
        LocalTensor<T2> sfmgClc2 = cast2Buf.Get<T2>();

        int64_t singleCoreLoopTimes = TilingData->preSfmgTilingData.normalCoreLoopTimes;
        int64_t singleCoreLastLoopNBurstNum = TilingData->preSfmgTilingData.normalCoreLastLoopNBurstNum; // 普通单核最后一次loop处理多少个D
        if (cBlockIdx == usedCoreNums - 1) {
            singleCoreLoopTimes = TilingData->preSfmgTilingData.tailCoreLoopTimes;
            singleCoreLastLoopNBurstNum = TilingData->preSfmgTilingData.tailCoreLastLoopNBurstNum;
        }

        int64_t startIdx = cBlockIdx * TilingData->preSfmgTilingData.normalCoreNBurstNums;
        int64_t nBurst = TilingData->preSfmgTilingData.singleLoopNBurstNum;
        int64_t curS = s1;

        for (int64_t i = 0; i < singleCoreLoopTimes; i++) {
            if (i == singleCoreLoopTimes - 1) {
                nBurst = singleCoreLastLoopNBurstNum;
            }

            // copyIn
            if (i == 0) {
                input1Buf = input1Que.AllocTensor<T1>();
                input2Buf = input2Que.AllocTensor<T1>();
                InitIndex((startIdx + i * TilingData->preSfmgTilingData.singleLoopNBurstNum) * d,
                           curS, actual_seq_qlen_addr);
                CopyInSfmg(nBurst, curS, actual_seq_qlen_addr);
            }

            // cast 1
            input1Que.EnQue(input1Buf);
            input1Que.DeQue<T1>();
            int64_t calcSize = nBurst * dAlign;
            Cast(sfmgClc1, input1Buf, RoundMode::CAST_NONE, calcSize);
            AscendC::PipeBarrier<PIPE_V>();
            input1Que.FreeTensor(input1Buf);

            // cast 2
            input2Que.EnQue(input2Buf);
            input2Que.DeQue<T1>();
            Cast(sfmgClc2, input2Buf, RoundMode::CAST_NONE, calcSize);
            AscendC::PipeBarrier<PIPE_V>();
            input2Que.FreeTensor(input2Buf);

            // pre copyIn next nBurst
            if (i < singleCoreLoopTimes - 1) {
                int64_t nextNBurst = i == singleCoreLoopTimes - 2 ? singleCoreLastLoopNBurstNum : nBurst;
                input1Buf = input1Que.AllocTensor<T1>();
                input2Buf = input2Que.AllocTensor<T1>();
                InitIndex((startIdx + (i + 1) * TilingData->preSfmgTilingData.singleLoopNBurstNum) * d,
                           curS, actual_seq_qlen_addr);
                CopyInSfmg(nextNBurst, curS, actual_seq_qlen_addr);
            }

            // sfmg
            outputBuf = out1Que.AllocTensor<T2>();
            Duplicate<T2>(outputBuf, 0.0, nBurst * 8);
            AscendC::PipeBarrier<PIPE_V>();

            uint32_t shapeArray[] = {static_cast<uint32_t>(nBurst), static_cast<uint32_t>(dAlign)};
            sfmgClc1.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
            sfmgClc2.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
            uint32_t shapeArray1[] = {static_cast<uint32_t>(nBurst), BLOCK_BYTE_SIZE / sizeof(T2)};
            outputBuf.SetShapeInfo(ShapeInfo(2, shapeArray1, DataFormat::ND));

            bool isBasicBlock = (nBurst % SFMG_HIGH_PERF_N_FACTOR == 0) && (dAlign % SFMG_HIGH_PERF_D_FACTOR == 0);
            if (likely(isBasicBlock)) {
                SoftmaxGradFront<float, true>(outputBuf, sfmgClc1, sfmgClc2, tempBuf, TilingData->softmaxGradTilingData);
            } else {
                SoftmaxGradFront<float, false>(outputBuf, sfmgClc1, sfmgClc2, tempBuf, TilingData->softmaxGradTilingData);
            }
            AscendC::PipeBarrier<PIPE_V>();

            // copyOut
            out1Que.EnQue(outputBuf);
            out1Que.DeQue<T2>();
            int64_t sfmgOutputOffset = (startIdx + i * TilingData->preSfmgTilingData.singleLoopNBurstNum) * BLOCK_SIZE;
            DataCopy(sfmgWorkspaceGm[sfmgOutputOffset], outputBuf, nBurst * BLOCK_SIZE);
            out1Que.FreeTensor(outputBuf);
        }
    }
}

template <typename T1, typename T2, typename TILING_TYPE, const uint32_t INPUT_LAYOUT>
__aicore__ inline void FlashAttentionScoreGradSfmg<T1, T2, TILING_TYPE, INPUT_LAYOUT>::SyncALLCores()
{
    SyncAll();
}

#endif // FLASH_ATTENTION_SCORE_GRAD_SFMG_KERNEL_H_
