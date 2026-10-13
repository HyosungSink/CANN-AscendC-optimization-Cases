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
 * \file nsa_selected_attention.h
 * \brief
 */

#ifndef NSA_SELECTED_ATTENTION_H
#define NSA_SELECTED_ATTENTION_H
#include "nsa_selected_attention_common.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using matmul::MatmulType;
using namespace AscendC;

struct SplitExtraInfo {
    int64_t bs1oIdx;
    int64_t boIdx;
    int64_t n2oIdx;
    int64_t goIdx;

    int64_t selectedS2RealSize;

    int64_t qCoreOffset;
    int64_t attentionOutGmOffset;
};

template <typename INPUT_T, bool ATTEN_ENABLE = false>
class NsaSelectedAttention {
public:
    __aicore__ inline NsaSelectedAttention(){};

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                    GM_ADDR topkIndices, GM_ADDR attenMask,
                                    GM_ADDR actualSeqQLen, GM_ADDR actualSeqKvLen,
                                    GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                    GM_ADDR attentionOut, GM_ADDR workspace,
                                    const NsaSelectedAttentionTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // define matmul1
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true, LayoutMode::NONE, false>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    matmul::Matmul<a1Type, b1Type, c1Type, bias1Type, CFG_EXCEED> bmm1;

    // define matmul2
    using a2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true, LayoutMode::NONE, false>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    matmul::Matmul<a2Type, b2Type, c2Type, bias2Type, CFG_EXCEED> bmm2;

protected:
    __aicore__ inline void InitIOGm(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR topkIndices,
                                    GM_ADDR attenMask, GM_ADDR actualSeqQLen, GM_ADDR actualSeqKvLen,
                                    GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR attentionOut);
    __aicore__ inline void SetTiling(const NsaSelectedAttentionTilingData *__restrict tiling);
    __aicore__ inline void InitWorkspace(GM_ADDR workspace);
    __aicore__ inline void InitBuffer();

    __aicore__ inline void GetIndices();
    __aicore__ inline void CopyInSeq();
    __aicore__ inline void CalcRealBS();
    __aicore__ inline void SelectAndGatherK();
    __aicore__ inline void SelectAndGatherV();

    __aicore__ inline void IterateBmm1(SplitExtraInfo &extraInfo);
    __aicore__ inline void IterateBmm2(SplitExtraInfo &extraInfo);
    __aicore__ inline void WaitBmm1Result();
    __aicore__ inline void WaitBmm2Result();
    __aicore__ inline void SetBmm1TensorA(SplitExtraInfo &extraInfo);
    __aicore__ inline void SetBmm1TensorB(SplitExtraInfo &extraInfo);

    __aicore__ inline void SetExtraInfo(SplitExtraInfo &extraInfo);
    __aicore__ inline void GetBmm1Result(SplitExtraInfo &extraInfo, uint32_t bmmResOffset, uint32_t currentProcessSize);
    __aicore__ inline void ProcessSoftmax(SplitExtraInfo &extraInfo);
    __aicore__ inline void SoftmaxDataCopyOut(SplitExtraInfo &extraInfo, int64_t s1oIdx);
    __aicore__ inline void GetShiftLeftMask(uint32_t totalSize, uint32_t shiftCount, uint64_t &mask);
    __aicore__ inline void DuplicateWithLoop(size_t selectedBlockIdx, uint32_t saveNum, uint32_t currentProcessSize,
                                             uint64_t mask[]);
    __aicore__ inline void HandleMask(uint32_t currentProcessSize);

    // 轴的乘积
    int64_t n2G;
    int64_t gD;
    int64_t gD2;
    int64_t n2GD;
    int64_t n2GD2;
    int64_t s1n2GD2;
    int64_t bN2GD2;
    int64_t gSize;

    int64_t currentCoreS1Len;  // 当前核处理的S1长度
    int64_t startBS1Offset; // 当前核位于BS1的起始位置，单位：1
    int64_t endBS1Offset;   // 当前核位于BS1的结束位置，单位：1
    int64_t currentBStartOffset;   // 当前B索引下BS1的初始位置，单位：1
    int64_t currentBS2StartOffset;   // 当前B索引下S1的长度，单位：1
    int64_t currentBS1Len;   // 当前B索引下S1的长度，单位：1
    int64_t currentS1BlockIdx; // 当前S1的块位置，单位：selectedBlockSize
    int64_t maxBlockNum;  // 当前处理的块数
    int64_t savedBS1Len; // 记录当前B索引下BS1最大长度

    uint64_t selectedBlockSize;
    uint64_t selectedBlockCount;
    uint64_t selectedLength;
    uint32_t dSize;
    uint32_t d2Size;
    uint32_t coreNum;
    uint32_t aivCoreNum;
    float scaleValue;
    uint64_t frontCoreNum;
    uint64_t frontCoreS1Len;
    int32_t softmaxReduceSize = 8;
    int32_t softmaxLocalWsSize;
    uint32_t maxAndSumOutBlockSize;
    uint32_t alignedBlockCount;
    uint32_t selectedKCopyRound;
    uint32_t selectedKCopySize;
    uint32_t selectedKCopySizeTail;
    uint32_t selectedVCopyRound;
    uint32_t selectedVCopySize;
    uint32_t selectedVCopySizeTail;
    uint32_t softmaxProcessRound;
    uint32_t softmaxProcessSize;
    uint32_t softmaxProcessSizeTail;

    float negativeFloatScalar;
    float positiveFloatScalar;

    int32_t blockIdx;
    const NsaSelectedAttentionTilingData *__restrict tilingData;
    // gather模块 变量
    uint32_t dimK;
    uint32_t dimV;
    uint32_t b;
    uint32_t n2;
    uint32_t s2;
    uint32_t s1;
    uint64_t currentBS1;
    uint64_t currentB;
    uint64_t currentS1;
    uint64_t currentN2;
    uint64_t topkIndicesGmOffset;
    uint64_t selectKeyGmOffset;
    uint64_t selectValueGmOffset;
    int64_t maxBlockSize;
    float negativeScalar;
    float positiveScalar;

    TPipe *pipe;

    // input
    GlobalTensor<INPUT_T> queryGm;
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<INPUT_T> valueGm;
    GlobalTensor<int64_t> actualSeqQLenGm;
    GlobalTensor<int64_t> actualSeqKvLenGm;
    GlobalTensor<int32_t> topkIndicesGm;
    GlobalTensor<uint8_t> attenMaskGmInt;
    // output
    GlobalTensor<INPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    // workspace
    GlobalTensor<INPUT_T> selectedKRes;
    GlobalTensor<INPUT_T> selectedVRes;
    GlobalTensor<float> mm1Res;
    GlobalTensor<INPUT_T> softmaxRes;

    LocalTensor<int32_t> topKIndicesBufferUB;
    LocalTensor<INPUT_T> selectKBufferUB;
    LocalTensor<INPUT_T> selectVBufferUB;
    LocalTensor<float> bmm1ResUb;
    LocalTensor<float> sumUb;
    LocalTensor<float> maxUb;
    LocalTensor<uint8_t> softmaxLocalWorkspace;
    LocalTensor<INPUT_T> softmaxResUb;
    LocalTensor<int64_t> actualSeqQLenUB;
    LocalTensor<int64_t> actualSeqKvLenUB;

    TBuf<TPosition::VECCALC> vecQue;

    constexpr static uint32_t SELECTED_KV_MAX_ELE_NUM = 24 * 1024 / sizeof(INPUT_T);
    constexpr static uint32_t SOFTMAX_MAX_ELE_NUM = 128 * 1024 / sizeof(float);
    constexpr static uint32_t UL_BIT_COUNT = 64;
    constexpr static uint32_t NUM_256 = 256;
};

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::Init(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR topkIndices,
    GM_ADDR attenMask, GM_ADDR actualSeqQLen, GM_ADDR actualSeqKvLen,
    GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR attentionOut, GM_ADDR workspace,
    const NsaSelectedAttentionTilingData *__restrict tiling, TPipe *tPipe)
{
    this->blockIdx = GetBlockIdx();
    this->pipe = tPipe;
    this->SetTiling(tiling);

    this->InitIOGm(query, key, value, topkIndices, attenMask, actualSeqQLen,
    actualSeqKvLen, softmaxMax, softmaxSum, attentionOut);
    this->InitWorkspace(workspace);
    this->InitBuffer();
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::SetTiling(
                                        const NsaSelectedAttentionTilingData *__restrict tiling)
{
    // copy base params
    this->tilingData = tiling;
    this->gSize = this->tilingData->inputParams.gSize;
    this->dSize = this->tilingData->inputParams.dSize;
    this->d2Size = this->tilingData->inputParams.d2Size;
    this->selectedBlockSize = this->tilingData->inputParams.selectedBlockSize;
    this->selectedBlockCount = this->tilingData->inputParams.selectedBlockCount;
    this->selectedLength = this->selectedBlockSize * this->selectedBlockCount;
    this->coreNum = this->tilingData->multiCoreParams.coreNum;
    this->aivCoreNum = this->coreNum;
    this->scaleValue = this->tilingData->inputParams.scaleValue;
    this->softmaxLocalWsSize = this->tilingData->multiCoreParams.localWorkSpaceSize;
    // gather 变量
    this->dimK = this->tilingData->inputParams.dSize;
    this->dimV = this->tilingData->inputParams.d2Size;
    this->b = this->tilingData->inputParams.bSize;
    this->n2 = this->tilingData->inputParams.n2Size;
    this->s2 = this->tilingData->inputParams.s2Size;
    this->s1 = this->tilingData->inputParams.s1Size;
    // 轴的乘积
    this->gD = this->tilingData->inputParams.gSize * this->tilingData->inputParams.dSize;
    this->n2G = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize;
    this->n2GD = this->n2G * this->tilingData->inputParams.dSize;
    this->gD2 = this->tilingData->inputParams.gSize * this->tilingData->inputParams.d2Size;
    this->n2GD2 = this->tilingData->inputParams.n2Size * this->gD2;
    this->s1n2GD2 = this->tilingData->inputParams.s1Size * this->gD2; // TND 场景S1后续会不同
    this->bN2GD2 = this->tilingData->inputParams.bSize * this->n2GD2;
    this->maxAndSumOutBlockSize = static_cast<uint32_t>(this->gSize) * SOFTMAX_REDUCE_SIZE;
    // BS1切分
    int64_t s1NumPerHeadCore = this->tilingData->multiCoreParams.s1NumPerHeadCore;
    int64_t s1NumPerTailCore = this->tilingData->multiCoreParams.s1NumPerTailCore;
    int64_t headCoreNum = this->tilingData->multiCoreParams.headCoreNum;
    int64_t tailCoreOffset = s1NumPerHeadCore * headCoreNum;
    bool isTailCore = headCoreNum <= this->blockIdx;
    this->currentCoreS1Len = isTailCore ? s1NumPerTailCore : s1NumPerHeadCore;
    this->startBS1Offset = isTailCore ? tailCoreOffset + (this->blockIdx - headCoreNum) * s1NumPerTailCore :
                                        this->blockIdx * s1NumPerHeadCore;
    this->endBS1Offset = this->startBS1Offset + this->currentCoreS1Len;

    alignedBlockCount = CeilDiv(selectedBlockCount, FP32_DATA_BLOCK) * FP32_DATA_BLOCK;
    if (selectedBlockSize * dimK <= SELECTED_KV_MAX_ELE_NUM) {
        selectedKCopyRound = 1;
        selectedKCopySize = selectedKCopySizeTail = selectedBlockSize;
    } else {
        selectedKCopySize = SELECTED_KV_MAX_ELE_NUM / dimK;
        selectedKCopyRound = (selectedBlockSize + selectedKCopySize - 1) / selectedKCopySize;
        selectedKCopySizeTail = selectedBlockSize - (selectedKCopyRound - 1) * selectedKCopySize;
    }
    if (selectedBlockSize * dimV <= SELECTED_KV_MAX_ELE_NUM) {
        selectedVCopyRound = 1;
        selectedVCopySize = selectedVCopySizeTail = selectedBlockSize;
    } else {
        selectedVCopySize = SELECTED_KV_MAX_ELE_NUM / dimV;
        selectedVCopyRound = (selectedBlockSize + selectedVCopySize - 1) / selectedVCopySize;
        selectedVCopySizeTail = selectedBlockSize - (selectedVCopyRound - 1) * selectedVCopySize;
    }
    if (gSize * selectedLength <= SOFTMAX_MAX_ELE_NUM) {
        softmaxProcessRound = 1;
        softmaxProcessSize = softmaxProcessSizeTail = gSize;
    } else {
        softmaxProcessSize = SOFTMAX_MAX_ELE_NUM / selectedLength;
        softmaxProcessRound = (gSize + softmaxProcessSize - 1) / softmaxProcessSize;
        softmaxProcessSizeTail = gSize - (softmaxProcessRound - 1) * softmaxProcessSize;
    }
};

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::InitIOGm(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR topkIndices,
    GM_ADDR attenMask, GM_ADDR actualSeqQLen, GM_ADDR actualSeqKvLen,
    GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR attentionOut)
{
    this->queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    this->topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topkIndices);
    this->attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
    this->actualSeqQLenGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqQLen);
    this->actualSeqKvLenGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqKvLen);

    this->softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    this->softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    this->attentionOutGm.SetGlobalBuffer((__gm__ INPUT_T *)attentionOut);
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::InitWorkspace(GM_ADDR workspace)
{
    uint64_t selectedDataSize = this->selectedLength * sizeof(INPUT_T);
    uint64_t selectedKSize = this->dSize * selectedDataSize * this->aivCoreNum;
    uint64_t selectedVSize = this->d2Size * selectedDataSize * this->aivCoreNum;

    uint64_t selectedKCoreOffset = this->dSize * selectedDataSize * this->blockIdx;
    uint64_t selectedVCoreOffset = this->d2Size * selectedDataSize * this->blockIdx;
    uint64_t mm1ResCoreOffset = this->gSize * this->selectedLength * sizeof(float) * this->blockIdx;
    
    this->selectedKRes.SetGlobalBuffer((__gm__ INPUT_T *)(workspace + selectedKCoreOffset));
    this->selectedVRes.SetGlobalBuffer((__gm__ INPUT_T *)(workspace + selectedKSize + selectedVCoreOffset));
    this->mm1Res.SetGlobalBuffer((__gm__ float *)(workspace + selectedKSize + selectedVSize + mm1ResCoreOffset));
    this->softmaxRes.SetGlobalBuffer((__gm__ INPUT_T *)(workspace + selectedKSize + selectedVSize + mm1ResCoreOffset));
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::InitBuffer()
{
    this->pipe->InitBuffer(vecQue, 191 * 1024);
    uint32_t ubOffset = 0;

    // selected indices calc ub
    this->topKIndicesBufferUB = vecQue.GetWithOffset<int32_t>(alignedBlockCount, ubOffset);
    ubOffset += alignedBlockCount * sizeof(int32_t);

    this->selectVBufferUB = this->selectKBufferUB = vecQue.GetWithOffset<INPUT_T>(SELECTED_KV_MAX_ELE_NUM, ubOffset);
    ubOffset += SELECTED_KV_MAX_ELE_NUM * sizeof(INPUT_T);  // 24K

    // softmax calc ub
    // softmaxResUb, bmm1ResUb UB复用
    this->softmaxResUb = vecQue.GetWithOffset<INPUT_T>(SOFTMAX_MAX_ELE_NUM, ubOffset);
    this->bmm1ResUb = vecQue.GetWithOffset<float>(SOFTMAX_MAX_ELE_NUM, ubOffset);
    ubOffset += SOFTMAX_MAX_ELE_NUM * sizeof(float);  // 128K

    this->sumUb = vecQue.GetWithOffset<float>(this->gSize * SOFTMAX_REDUCE_SIZE, ubOffset);
    ubOffset += this->gSize * SOFTMAX_REDUCE_SIZE * sizeof(float);

    this->maxUb = vecQue.GetWithOffset<float>(this->gSize * SOFTMAX_REDUCE_SIZE, ubOffset);
    ubOffset += this->gSize * SOFTMAX_REDUCE_SIZE * sizeof(float);

    this->softmaxLocalWorkspace = vecQue.GetWithOffset<uint8_t>(this->softmaxLocalWsSize, ubOffset);
    ubOffset += this->softmaxLocalWsSize * sizeof(uint8_t);

    int64_t AlignedB = CeilDiv(this->b, S64_DATA_BLOCK) * S64_DATA_BLOCK;
    this->actualSeqQLenUB = vecQue.GetWithOffset<int64_t>(AlignedB, ubOffset);
    ubOffset += AlignedB * sizeof(int64_t);

    this->actualSeqKvLenUB = vecQue.GetWithOffset<int64_t>(AlignedB, ubOffset);
    ubOffset += AlignedB * sizeof(int64_t);
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::Process()
{
    if (this->blockIdx >= this->aivCoreNum) {
        return;
    }
    SplitExtraInfo extraInfo;
    GetExtremeValue(this->negativeScalar, this->positiveScalar);
    this->CopyInSeq();
    bool firstLoop = true;
    this->currentB = 0;
    this->currentS1 = 0;
    this->savedBS1Len = 0;
    for (this->currentBS1 = this->startBS1Offset; this->currentBS1 < this->endBS1Offset; this->currentBS1++) {
        this->CalcRealBS();
        for (this->currentN2 = 0; this->currentN2 < this->n2; this->currentN2++) {
            this->GetIndices();
            this->SelectAndGatherK();
            this->SetExtraInfo(extraInfo);
            if (!firstLoop) {
                this->WaitBmm2Result();
            } else {
                firstLoop = false;
            }
            this->IterateBmm1(extraInfo);
            this->SelectAndGatherV();
            this->WaitBmm1Result();
            this->ProcessSoftmax(extraInfo);
            this->IterateBmm2(extraInfo);
        }
    }
    if (!firstLoop) {
        this->WaitBmm2Result();
    }
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::CopyInSeq()
{
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = this->b * sizeof(int64_t);
    DataCopyPadExtParams<int64_t> padParams;
    DataCopyPad(this->actualSeqQLenUB, this->actualSeqQLenGm, copyParams, padParams);
    DataCopyPad(this->actualSeqKvLenUB, this->actualSeqKvLenGm, copyParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_S>(eventIdMte2ToS);
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::CalcRealBS()
{
    // currentBS1 未超过currentB索引下BS1最大长度,不需要更新currentB索引
    if (this->savedBS1Len > this->currentBS1) {
        this->currentS1 = this->currentBS1 - this->currentBStartOffset;
        this->maxBlockNum = CeilDiv(this->currentS1, this->selectedBlockSize);
        this->maxBlockNum = this->maxBlockNum < this->selectedBlockCount ? this->maxBlockNum : this->selectedBlockCount;
        return;
    }
    // 更新currentB索引
    for (size_t index = this->currentB; index < this->b; ++index) {
        this->savedBS1Len = this->actualSeqQLenUB.GetValue(index);
        if (this->savedBS1Len > this->currentBS1) {
            this->currentB = index;
            this->currentBStartOffset = index == 0 ? 0 : this->actualSeqQLenUB.GetValue(index - 1);
            this->currentS1 = this->currentBS1 - this->currentBStartOffset;
            this->currentBS1Len = this->savedBS1Len - this->currentBStartOffset;
            this->currentBS2StartOffset = index == 0 ? 0 : this->actualSeqKvLenUB.GetValue(index - 1);
            this->maxBlockNum = CeilDiv(this->currentS1, this->selectedBlockSize);
            this->maxBlockNum = this->maxBlockNum < this->selectedBlockCount ? this->maxBlockNum : this->selectedBlockCount;
            return;
        }
    }
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::GetIndices()
{
    this->topkIndicesGmOffset = static_cast<uint64_t>(
        this->currentBS1 * this->n2 * this->selectedBlockCount +
        this->currentN2 * this->selectedBlockCount);
    DataCopy(this->topKIndicesBufferUB, this->topkIndicesGm[this->topkIndicesGmOffset], alignedBlockCount);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::SelectAndGatherK()
{
    this->selectKeyGmOffset = 0;
    uint64_t selectKTmpOffset = 0;
    uint64_t selectKOffset = this->currentBS2StartOffset * this->n2 * this->dimK;
    uint64_t copySize = 0;
    DataCopyParams copyParams;
    copyParams.blockCount = 0;
    copyParams.blockLen = this->dimK / DATA_BLOCK;
    copyParams.srcStride = (this->n2 - 1) * this->dimK / DATA_BLOCK;
    copyParams.dstStride = 0;

    for (size_t index = 0; index < this->selectedBlockCount; ++index) {
        uint64_t topkIndex = static_cast<uint64_t>(this->topKIndicesBufferUB.GetValue(index));
        selectKTmpOffset = static_cast<uint64_t>(selectKOffset + topkIndex * this->selectedBlockSize * this->n2 * this->dimK + this->currentN2 * this->dimK);

        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        for (uint32_t copyRound = 0; copyRound < selectedKCopyRound; ++copyRound) {
            if (copyRound + 1 == selectedKCopyRound) {
                copyParams.blockCount = selectedKCopySizeTail;
            } else {
                copyParams.blockCount = selectedKCopySize;
            }
            copySize = copyParams.blockCount * this->dimK;
            DataCopy(this->selectKBufferUB, this->keyGm[selectKTmpOffset], copyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopy(this->selectedKRes[this->selectKeyGmOffset], this->selectKBufferUB, copySize);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            this->selectKeyGmOffset += copySize;
            selectKTmpOffset += selectedKCopySize * this->n2 * this->dimK;
        }
    }
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::SelectAndGatherV()
{
    this->selectValueGmOffset = 0;
    uint64_t selectVTmpOffset = 0;
    uint64_t selectVOffset = this->currentBS2StartOffset * this->n2 * this->dimV;
    uint64_t copySize = 0;
    DataCopyParams copyParams;
    copyParams.blockCount = 0;
    copyParams.blockLen = this->dimV / DATA_BLOCK;
    copyParams.srcStride = (this->n2 - 1) * this->dimV / DATA_BLOCK;
    copyParams.dstStride = 0;
    for (size_t index = 0; index < this->selectedBlockCount; ++index) {
        uint64_t topkIndex = static_cast<uint64_t>(this->topKIndicesBufferUB.GetValue(index));
        selectVTmpOffset = static_cast<uint64_t>(selectVOffset + topkIndex * this->selectedBlockSize * this->n2 * this->dimV + this->currentN2 * this->dimV);

        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);

        for (uint32_t copyRound = 0; copyRound < selectedVCopyRound; ++copyRound) {
            if (copyRound + 1 == selectedVCopyRound) {
                copyParams.blockCount = selectedVCopySizeTail;
            } else {
                copyParams.blockCount = selectedVCopySize;
            }
            copySize = copyParams.blockCount * this->dimV;
            DataCopy(this->selectVBufferUB, this->valueGm[selectVTmpOffset], copyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

            DataCopy(this->selectedVRes[this->selectValueGmOffset], this->selectVBufferUB, copySize);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            this->selectValueGmOffset += copySize;
            selectVTmpOffset += selectedVCopySize * this->n2 * this->dimV;
        }
    }
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::ProcessSoftmax(SplitExtraInfo &extraInfo)
{
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    for (uint32_t processRound = 0, currentProcessSize = softmaxProcessSize, bmmResOffset = 0, reduceOffset = 0;
         processRound < softmaxProcessRound; ++processRound) {
        if (processRound + 1 == softmaxProcessRound) {
            currentProcessSize = softmaxProcessSizeTail;
        }
        if (processRound > 0) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        // 获取mm1结果并做mask
        this->GetBmm1Result(extraInfo, bmmResOffset, currentProcessSize);

        // mul需要等mm结果搬完
        uint64_t copyLength = currentProcessSize * this->selectedLength;
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Muls(bmm1ResUb, bmm1ResUb, scaleValue, copyLength);
        AscendC::PipeBarrier<PIPE_V>();

        // softmax API
        AscendC::SoftMaxShapeInfo softmaxShapeInfo = {
            static_cast<uint32_t>(currentProcessSize),
            static_cast<uint32_t>(this->selectedLength),
            static_cast<uint32_t>(currentProcessSize),
            static_cast<uint32_t>(this->selectedLength)};
        SoftMax(bmm1ResUb,
                sumUb[reduceOffset],
                maxUb[reduceOffset],
                bmm1ResUb,
                softmaxLocalWorkspace,
                this->tilingData->softmaxTilingData,
                softmaxShapeInfo);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(softmaxResUb, bmm1ResUb, RoundMode::CAST_ROUND, copyLength);

        // softmax res --> workspace
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopy(this->softmaxRes[bmmResOffset], softmaxResUb, copyLength);
        bmmResOffset += softmaxProcessSize * selectedLength;
        reduceOffset += softmaxProcessSize * SOFTMAX_REDUCE_SIZE;
    }

    // softmax Max/Sum --> GM
    uint64_t maxAndSumOutOffset = this->currentBS1 * this->n2G * SOFTMAX_REDUCE_SIZE +
                                  this->currentN2 * this->maxAndSumOutBlockSize;
    DataCopy(this->softmaxMaxGm[maxAndSumOutOffset], maxUb, this->maxAndSumOutBlockSize);
    DataCopy(this->softmaxSumGm[maxAndSumOutOffset], sumUb, this->maxAndSumOutBlockSize);

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::GetShiftLeftMask(
    uint32_t totalSize, uint32_t shiftCount, uint64_t &mask)
{
    mask = (~0ULL) << shiftCount;
    if (totalSize < UL_BIT_COUNT) {
        mask &= ((1ULL << totalSize) - 1);
    }
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::DuplicateWithLoop(
    size_t selectedBlockIdx, uint32_t saveNum, uint32_t currentProcessSize, uint64_t mask[])
{
    if (saveNum >= UL_BIT_COUNT) {
        GetShiftLeftMask(this->selectedBlockSize - UL_BIT_COUNT, saveNum - UL_BIT_COUNT, mask[0]);
        for (uint32_t round = 0, offset = selectedBlockIdx * this->selectedBlockSize + UL_BIT_COUNT;
             round < currentProcessSize; ++round, offset += this->selectedLength) {
            AscendC::Duplicate(this->bmm1ResUb[offset], this->negativeScalar, mask, 1, 1, 0);
        }
    } else {
        GetShiftLeftMask(this->selectedBlockSize, saveNum, mask[0]);
        for (uint32_t round = 0, offset = selectedBlockIdx * this->selectedBlockSize;
             round < currentProcessSize; ++round, offset += this->selectedLength) {
            AscendC::Duplicate(this->bmm1ResUb[offset], this->negativeScalar, mask, 1, 1, 0);
        }
        if (selectedBlockSize > UL_BIT_COUNT) {
            for (uint32_t round = 0, offset = selectedBlockIdx * this->selectedBlockSize + UL_BIT_COUNT;
                 round < currentProcessSize; ++round, offset += this->selectedLength) {
                AscendC::Duplicate(this->bmm1ResUb[offset], this->negativeScalar,
                                   this->selectedBlockSize - UL_BIT_COUNT);
            }
        }
    }
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::HandleMask(uint32_t currentProcessSize)
{
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    uint8_t dstStride = this->selectedLength / FP32_DATA_BLOCK;
    // when block size > 64, total bytes in one block will exceed 256
    // when selectedLength >= 2048, dstStride will be >= 256, can not be saved in uint8_t
    bool useLoop = this->selectedBlockSize > UL_BIT_COUNT || (this->selectedLength >= FP32_DATA_BLOCK * NUM_256);
    uint64_t diagBlockIdx = this->currentS1 / this->selectedBlockSize;
    int32_t currentBlockIdx = 0;
    uint64_t saveNum = this->currentS1 % this->selectedBlockSize + 1;
    uint64_t mask[1];
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    for (size_t selectedBlockIdx = 0; selectedBlockIdx < this->selectedBlockCount; ++selectedBlockIdx) {
         currentBlockIdx = this->topKIndicesBufferUB.GetValue(selectedBlockIdx);
        if (currentBlockIdx == diagBlockIdx) {
            if (saveNum == this->selectedBlockSize) {
                continue;
            }
            if (useLoop) {
                DuplicateWithLoop(selectedBlockIdx, saveNum, currentProcessSize, mask);
            } else {
                GetShiftLeftMask(this->selectedBlockSize, saveNum, mask[0]);
                AscendC::Duplicate(this->bmm1ResUb[selectedBlockIdx * this->selectedBlockSize],
                                   this->negativeScalar, mask, currentProcessSize, 1, dstStride);
            }
            AscendC::PipeBarrier<PIPE_V>();
        } else if (currentBlockIdx > diagBlockIdx) {
            if (useLoop) {
                for (uint32_t round = 0, offset = selectedBlockIdx * this->selectedBlockSize;
                     round < currentProcessSize; ++round, offset += this->selectedLength) {
                    AscendC::Duplicate(this->bmm1ResUb[offset], this->negativeScalar, this->selectedBlockSize);
                }
            } else {
                AscendC::Duplicate(this->bmm1ResUb[selectedBlockIdx * this->selectedBlockSize],
                                   this->negativeScalar, this->selectedBlockSize, currentProcessSize, 1, dstStride);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(eventIdMte2ToS);
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::GetBmm1Result(
    SplitExtraInfo &extraInfo, uint32_t bmmResOffset, uint32_t currentProcessSize)
{
    DataCopy(this->bmm1ResUb, this->mm1Res[bmmResOffset], currentProcessSize * this->selectedLength);
    // mask 
    if constexpr (ATTEN_ENABLE) {
        HandleMask(currentProcessSize);
    }
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::SetExtraInfo(SplitExtraInfo &extraInfo)
{
    extraInfo.bs1oIdx = this->currentBS1;
    extraInfo.n2oIdx = this->currentN2;

    extraInfo.selectedS2RealSize = this->selectedLength;
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::IterateBmm1(SplitExtraInfo &extraInfo)
{
    this->SetBmm1TensorA(extraInfo);
    this->SetBmm1TensorB(extraInfo);

    this->bmm1.template IterateAll<false>(this->mm1Res, false, false, true);
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::SetBmm1TensorA(SplitExtraInfo &extraInfo)
{
    // 计算gm上的offset
    int64_t bs1Offset = 0;
    int64_t n2Offset = 0;

    bs1Offset = extraInfo.bs1oIdx * this->n2GD;
    n2Offset = extraInfo.n2oIdx * this->gD;
    extraInfo.qCoreOffset = bs1Offset + n2Offset;
    this->bmm1.SetTensorA(this->queryGm[extraInfo.qCoreOffset]);
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::SetBmm1TensorB(SplitExtraInfo &extraInfo)
{
    this->bmm1.SetTensorB(this->selectedKRes, true);
    this->bmm1.SetTail(this->tilingData->inputParams.gSize, extraInfo.selectedS2RealSize);
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::WaitBmm1Result()
{
    this->bmm1.WaitIterateAll();
    this->bmm1.End();
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::IterateBmm2(SplitExtraInfo &extraInfo)
{
    // 计算gm上的offset
    int64_t bs1Offset = 0;
    int64_t n2Offset = 0;

    bs1Offset = extraInfo.bs1oIdx * this->n2GD2;
    n2Offset = extraInfo.n2oIdx * this->gD2;
    extraInfo.attentionOutGmOffset = bs1Offset + n2Offset;

    this->bmm2.SetTensorA(this->softmaxRes);
    this->bmm2.SetTensorB(this->selectedVRes);
    this->bmm2.SetTail(this->tilingData->inputParams.gSize, this->d2Size, extraInfo.selectedS2RealSize);

    this->bmm2.template IterateAll<false>(this->attentionOutGm[extraInfo.attentionOutGmOffset], false, false, true);
}

template <typename INPUT_T, bool ATTEN_ENABLE>
__aicore__ inline void NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>::WaitBmm2Result()
{
    this->bmm2.WaitIterateAll();
    this->bmm2.End();
}

#endif // NSA_SELECTED_ATTENTION_H

