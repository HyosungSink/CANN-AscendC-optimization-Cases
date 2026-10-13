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
 * \file flash_attention_var_len_score.h
 * \brief
 */

#ifndef FLASH_ATTENTION_VAR_LEN_SCORE_H
#define FLASH_ATTENTION_VAR_LEN_SCORE_H

#include "flash_attention_score_s1s2_bn2gs1.h"

// INPUT_T - means data type for input
// T       - means data type when calc
#define Q_EVENT0  EVENT_ID2
#define KV_EVENT0 EVENT_ID4
#define P_EVENT0  EVENT_ID6

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T = INPUT_T, bool isBasicBlock = false, CubeFormat bmm1Format = CubeFormat::NZ, bool hasRope = false>
class FlashAttentionVarLenScore
    : public FlashAttentionScoreS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                           bmm1Format, false, hasRope> {
public:
    __aicore__ inline FlashAttentionVarLenScore(){};

    __aicore__ inline void
    UnpackInit(__gm__ uint8_t *query, __gm__ uint8_t *queryRope, __gm__ uint8_t *key, __gm__ uint8_t *keyRope,
               __gm__ uint8_t *value, __gm__ uint8_t *pse,
               __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, __gm__ uint8_t *sink,
               __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *softmaxMax,
               __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut,
               __gm__ uint8_t *workspace, const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);

    __aicore__ inline void Process();
    __aicore__ inline void ProcessL1Carry();

    using a1TypeL1Carry = MatmulType<TPosition::A1, CubeFormat::NZ, INPUT_T>;
    using b1TypeL1Carry = MatmulType<TPosition::B1, CubeFormat::NZ, INPUT_T, true>;
    using bias1Type = typename FlashAttentionScoreS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                                             bmm1Format, false, hasRope>::bias1Type;
    using c1Type = typename FlashAttentionScoreS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                                          bmm1Format, false, hasRope>::c1Type;
    matmul::MatmulImpl<a1TypeL1Carry, b1TypeL1Carry, c1Type, bias1Type, CFG_EXCEED> bmm1L1Carry;

    using c1NzType = typename FlashAttentionScoreS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                                            bmm1Format, false, hasRope>::c1NzType;                                                                          
    matmul::MatmulImpl<a1TypeL1Carry, b1TypeL1Carry, c1NzType, bias1Type, CFG_EXCEED> bmm1NzL1Carry;

    using a2TypeL1Carry = MatmulType<TPosition::A1, CubeFormat::NZ, INPUT_T>;
    using b2TypeL1Carry = MatmulType<TPosition::B1, CubeFormat::NZ, INPUT_T>;
    using bias2Type = typename FlashAttentionScoreS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                                             bmm1Format, false, hasRope>::bias2Type;
    using c2NzType = typename FlashAttentionScoreS1s2Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                                                            bmm1Format, false, hasRope>::c2NzType;
    matmul::MatmulImpl<a2TypeL1Carry, b2TypeL1Carry, c2NzType, bias2Type, CFG_MDL_EXCEED> bmm2L1Carry;   

protected:
    __aicore__ inline void ComputeConstexpr();

    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);

    __aicore__ inline void ComputeBmm1Tail(SplitExtraInfo &extraInfo);

    __aicore__ inline void SetExtraInfo(SplitExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount,
                                        int64_t s2LoopLimit, int64_t multiCoreInnerIdx);

    __aicore__ inline void CalS1OuterSize(int64_t offset);

    __aicore__ inline void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen);

    __aicore__ inline void GetS2LoopRange();

    __aicore__ inline void CalcQCoreOffset(const SplitExtraInfo &extraInfo);

    __aicore__ inline void CalcKCoreOffset(const SplitExtraInfo &extraInfo);

    __aicore__ inline void CalcVCoreOffset(const SplitExtraInfo &extraInfo);

    __aicore__ inline void CalcQRopeCoreOffset(const SplitExtraInfo &extraInfo);

    __aicore__ inline void CalcKRopeCoreOffset(const SplitExtraInfo &extraInfo);

    template <typename T2>
    __aicore__ inline void ComputeMm1(const SplitExtraInfo &extraInfo, 
                                      matmul::MatmulImpl<a1TypeL1Carry, b1TypeL1Carry, T2, bias1Type, CFG_EXCEED> &bmm1, int32_t cubeSubIdx);

    __aicore__ inline void ComputeMm2(const SplitExtraInfo &extraInfo, int32_t cubeSubIdx);

    __aicore__ inline void CopyInMm1AToL1(LocalTensor<INPUT_T> &l1Tensor, const SplitExtraInfo &extraInfo);

    __aicore__ inline void CopyInMm1BToL1(LocalTensor<INPUT_T> &l1Tensor, const SplitExtraInfo &extraInfo,
                                          uint32_t subNid, uint32_t subNSize);

    __aicore__ inline void CopyInMm2AToL1(LocalTensor<INPUT_T> &l1Tensor, const SplitExtraInfo &extraInfo,
                                          uint32_t subKid, uint32_t subKSize, int32_t cubeSubIdx);

    __aicore__ inline void CopyInMm2BToL1(LocalTensor<INPUT_T> &l1Tensor, const SplitExtraInfo &extraInfo,
                                          uint32_t subKid, uint32_t subKSize); 

    __aicore__ inline void CopyGmToL1(LocalTensor<INPUT_T> &l1Tensor, GlobalTensor<INPUT_T> &gmSrcTensor,
                                      uint32_t srcN, uint32_t srcD, uint32_t srcDstride); 

    __aicore__ inline void PairComputeAxisIdx(int64_t multiCoreInnerIdx);

    __aicore__ inline void PairSetExtraInfo(SplitExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount,
                                            int64_t s2LoopLimit, int64_t multiCoreInnerIdx);

    __aicore__ inline void PairCalS1OuterSize(int64_t offset);     

    __aicore__ inline void PairGetS1LoopRange(int64_t &multiCoreInnerOffset, int64_t &multiCoreInnerLimit);

    __aicore__ inline void PairGetS2LoopRange();       
                                                                                                                                                           
    // Unpack 用参数
    GM_ADDR actualSeqQlenAddr;
    GM_ADDR actualSeqKvlenAddr;
    GM_ADDR sinkAddr;
    GM_ADDR prefixNAddr;
    uint64_t s1OuterSizeAcc;
    uint64_t s1SizeAcc;
    uint64_t s2SizeAcc;
    uint64_t attenB1SSOffset;
    uint64_t softmaxPingPongCnt = 0;
    GlobalTensor<int64_t> qListGm;
    GlobalTensor<int64_t> kvListGm;
    GlobalTensor<float> sinkGm;

    static constexpr uint32_t S1_BASE_SIZE_L1CARRY_MAX = 128;
    static constexpr uint32_t D_SIZE_L1CARRY_MAX = 256;
    static constexpr uint32_t D2_SIZE_L1CARRY_MAX = D_SIZE_L1CARRY_MAX;
    static constexpr uint32_t N_SPLIT_SIZE = 256;
    static constexpr uint32_t K_SPLIT_SIZE = N_SPLIT_SIZE;
    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_C1_V1_FLAG[2][2] = {{9, 10}, {11, 12}};
    static constexpr uint64_t SYNC_V1_C2_FLAG[2][2] = {{5, 6}, {7, 8}};
    static constexpr uint64_t SYNC_C2_V2_FLAG[2][2] = {{1, 2}, {3, 4}};
    static constexpr uint32_t L1_Q_SIZE = (S1_BASE_SIZE_L1CARRY_MAX * D_SIZE_L1CARRY_MAX * sizeof(INPUT_T));
    static constexpr uint32_t L1_KV_SIZE = (D_SIZE_L1CARRY_MAX * N_SPLIT_SIZE * sizeof(INPUT_T));
    static constexpr uint32_t L1_P_SIZE = (S1_BASE_SIZE_L1CARRY_MAX * K_SPLIT_SIZE * sizeof(INPUT_T));

    TBuf<TPosition::A1> queryBufL1;
    LocalTensor<INPUT_T> qL1Tensor;
    TBuf<TPosition::B1> kvBufL1;
    LocalTensor<INPUT_T> kvL1Tensor;
    TBuf<TPosition::A1> pBufL1;
    LocalTensor<INPUT_T> pL1Tensor;
    int64_t kCoreOffset;
    int64_t vCoreOffset;
    int64_t kRopeCoreOffset;
    uint32_t kvL1BufIter = 0;
    uint32_t pL1BufIter = 0;

    int64_t pairBoIdx;
    int64_t pairN2oIdx;
    int64_t pairGoIdx;
    int64_t pairS1oIdx;
    int64_t pairS1OuterSizeAcc;
    int64_t pairS1SizeAcc;
    int64_t pairS2SizeAcc;
    int64_t pairAttenB1SSOffset;
    int64_t pairS2StartIdx;
    int64_t pairS2EndIdx;
    int32_t pairBlockIdx;
};

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::UnpackInit(
    __gm__ uint8_t *query, __gm__ uint8_t *queryRope, __gm__ uint8_t *key, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, __gm__ uint8_t *sink, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe)
{
    this->InitInput(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, sink, softmaxMax, softmaxSum,
                    softmaxOut, attentionOut, workspace, tiling, tPipe); // gm设置
    if ASCEND_IS_AIC {
        if (this->tilingData->inputParams.needL1Carry) {
            this->bmm1L1Carry.SetSubBlockIdx(0);
            this->bmm1L1Carry.Init(&tiling->bmm1TilingData, tPipe);
            this->bmm1NzL1Carry.SetSubBlockIdx(0);
            this->bmm1NzL1Carry.Init(&tiling->bmm1TilingData, tPipe);
            this->bmm2L1Carry.SetSubBlockIdx(0);
            this->bmm2L1Carry.Init(&tiling->bmm2TilingData, tPipe);
        }
        this->pairBlockIdx = this->blockIdx + 1;
    } else {
        this->pairBlockIdx = this->blockIdx & 1 ? this->blockIdx - 1 : this->blockIdx + 1;
    }
    this->ComputeConstexpr();
    qListGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengths);
    kvListGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsKv);
    sinkGm.SetGlobalBuffer((__gm__ float *)sink);
    // 初始化unpack
    actualSeqQlenAddr = actualSeqLengths;
    actualSeqKvlenAddr = actualSeqLengthsKv;
    sinkAddr = sink;
    int64_t actualS1Len;
    int64_t actualS2Len;
    for (int64_t i = 0; i < this->tilingData->inputParams.bSize; ++i) {
        GetSeqQlenKvlenByBoidx(i, actualS1Len, actualS2Len);
        if (actualS2Len <= 0 && actualS1Len != 0) {
            int64_t accumSize = (i == 0) ? 0 : ((__gm__ int64_t *)actualSeqQlenAddr)[i - 1];
            if (actualS1Len < 0 && accumSize > 0) {
                actualS1Len = this->s1Size - accumSize;
                int64_t frontCoreNum = actualS1Len % this->tilingData->multiCoreParams.coreNum;
                int64_t splitFactor = frontCoreNum > 0 ? 1 : 0;
                int64_t s1SizeInner = (actualS1Len / this->tilingData->multiCoreParams.coreNum);
                int64_t innerOffset1 = (s1SizeInner + splitFactor) * (this->blockIdx >= frontCoreNum ?
                                       frontCoreNum : this->blockIdx);
                int64_t innerOffset2 = s1SizeInner * (this->blockIdx >= frontCoreNum ?
                                       this->blockIdx - frontCoreNum : 0);
                accumSize = accumSize + innerOffset1 + innerOffset2;
                actualS1Len = s1SizeInner + (this->blockIdx >= frontCoreNum ? 0 : splitFactor);
            }
            AscendC::InitOutput<INPUT_T>(this->attentionOutGm[accumSize * this->n2GD2],
                                         actualS1Len * this->n2GD2, static_cast<INPUT_T>(0.0));
            AscendC::InitOutput<float>(this->softmaxMaxGm[accumSize * this->n2G * 8],
                                       actualS1Len * this->n2G * 8, static_cast<float>(0.0));
            AscendC::InitOutput<float>(this->softmaxSumGm[accumSize * this->n2G * 8],
                                       actualS1Len * this->n2G * 8, static_cast<float>(0.0));
        }
    }
    this->InitBuffer();
    if (this->tilingData->inputParams.needL1Carry) {
        this->pipe->InitBuffer(this->queryBufL1, L1_Q_SIZE * 2);
        qL1Tensor = this->queryBufL1.template Get<INPUT_T>();
        this->pipe->InitBuffer(this->kvBufL1, L1_KV_SIZE * 2);
        kvL1Tensor = this->kvBufL1.template Get<INPUT_T>();
        this->pipe->InitBuffer(this->pBufL1, L1_P_SIZE * 2);
        pL1Tensor = this->pBufL1.template Get<INPUT_T>();
    }
    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    DropOutBitModeInit(apiTmpBuffer);
    if (this->blockIdx < this->tilingData->multiCoreParams.coreNum) {
        LocalTensor<half> pseHelpBuffer = this->stage1PingBuf.template Get<half>();
        PseInnerAlibiCreate<hasPse>(this->pseAlibiGm, pseHelpBuffer, this->pseInfo);
    }
    prefixNAddr = prefix;
    this->s2SizeSum = ((__gm__ uint64_t *)actualSeqLengthsKv)[this->tilingData->inputParams.bSize - 1];
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::ComputeConstexpr()
{
    this->gD = this->tilingData->inputParams.gSize * this->dSize;
    this->n2D = this->tilingData->inputParams.n2Size * this->dSize;
    this->n2G = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize;
    this->n2GD = this->tilingData->inputParams.n2Size * this->gD;
    this->gD2 = this->tilingData->inputParams.gSize * this->d2Size;
    this->n2D2 = this->tilingData->inputParams.n2Size * this->d2Size;
    this->n2GD2 = this->tilingData->inputParams.n2Size * this->gD2;
    // 计算切分轴的乘积
    this->s2BaseN2D = this->s2BaseSize * this->n2D;
    this->s2BaseNratioSize = this->s2BaseSize * this->tilingData->coreParams.nRatio;
    this->s1Size = this->tilingData->inputParams.s1Size;
    // layout(bs)ND
    this->s1BaseN2GD = this->s1BaseSize * this->n2GD;
    this->s1BaseN2GD2 = this->s1BaseSize * this->n2GD2;
    this->s2BaseNratioN2D = this->s2BaseN2D * this->tilingData->coreParams.nRatio;
    if constexpr (hasRope == true) {
        this->gDRope = this->tilingData->inputParams.gSize * this->dRopeSize;
        this->n2DRope = this->tilingData->inputParams.n2Size * this->dRopeSize;
        this->n2GDRope = this->tilingData->inputParams.n2Size * this->gDRope;
        this->s2BaseN2DRope = this->s2BaseSize * this->n2DRope;
        this->s1BaseN2GDRope = this->s1BaseSize * this->n2GDRope;
        this->s2BaseNratioN2DRope = this->s2BaseN2DRope * this->tilingData->coreParams.nRatio;
        this->mm1Ka2 = this->n2GDRope;
        this->mm1Kb2 = this->n2DRope;
    }
    // layout(bs)ND
    this->mm1Ka1 = this->n2GD;
    this->mm1Kb1 = this->n2D;
    this->mm1Kb = this->mm1Kb1 + this->mm1Kb2;
    this->mm2Kb = this->n2D2;

    if constexpr (hasDrop == true) {
        this->dropMaskInfo.s2BaseNratioSize = this->s2BaseNratioSize;
        this->dropMaskInfo.s1BaseSize = this->s1BaseSize;
        this->dropMaskInfo.n2G = this->n2G;
        this->dropMaskInfo.gSize = this->tilingData->inputParams.gSize;
    }
    if constexpr (hasPse == true) {
        this->pseInfo.gSize = this->tilingData->inputParams.gSize;
        this->pseInfo.n2G = this->n2G;
        this->pseInfo.kvStartIdx = this->tilingData->inputParams.kvStartIdx;
        this->pseInfo.pseAlibiBaseS1 = this->tilingData->coreParams.pseAlibiBaseS1;
        this->pseInfo.pseAlibiBaseS2 = this->tilingData->coreParams.pseAlibiBaseS2;
        this->pseInfo.pseBSize = this->tilingData->inputParams.pseBSize;
        this->pseInfo.pseEncodeType = (uint32_t)this->tilingData->inputParams.pseEncodeType;
        this->pseInfo.pseShapeType = this->tilingData->inputParams.pseShapeType;
        this->pseInfo.pseType = this->tilingData->inputParams.pseType;
        this->pseInfo.pseS1Size = this->tilingData->inputParams.pseS1Size;
        this->pseInfo.pseS2Size = this->tilingData->inputParams.pseS2Size;
        this->pseInfo.qStartIdx = this->tilingData->inputParams.qStartIdx;
        this->pseInfo.s1BaseSize = this->s1BaseSize;
        this->pseInfo.s2BaseNratioSize = this->s2BaseNratioSize;
    }
    if (this->s1BaseSize > 128) {
        this->softmaxReduceSize = 1;
    }
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void
FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T, isBasicBlock,
                          bmm1Format, hasRope>::GetSeqQlenKvlenByBoidx(int64_t boIdx,
                                                                       int64_t &actualSeqQlen,
                                                                       int64_t &actualSeqKvlen)
{
    if (boIdx == 0) {
        actualSeqQlen = qListGm.GetValue(0);
        actualSeqKvlen = kvListGm.GetValue(0);
    } else {
        actualSeqQlen = qListGm.GetValue(boIdx) - qListGm.GetValue(boIdx - 1);
        actualSeqKvlen = kvListGm.GetValue(boIdx) - kvListGm.GetValue(boIdx - 1);
    }
    return;
}

// 初始化s1方向上的累加值
template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::CalS1OuterSize(int64_t offset)
{
    int64_t actualS1Outersize = 0;
    // 用于取actualS1Len下标
    this->boIdx = 0;
    this->s1OuterSizeAcc = 0;
    this->attenB1SSOffset = 0;
    this->s1SizeAcc = 0;
    this->s2SizeAcc = 0;
    int64_t actualS1Len;
    int64_t actualS2Len;
    for (int64_t i = 0; i < this->tilingData->inputParams.bSize; i++) {
        GetSeqQlenKvlenByBoidx(i, actualS1Len, actualS2Len);
        actualS1Outersize += (CeilDiv(actualS1Len, this->s1BaseSize) * this->n2G);
        if (offset >= actualS1Outersize) {
            this->s1OuterSizeAcc = actualS1Outersize;
            this->s1SizeAcc += actualS1Len;
            this->s2SizeAcc += actualS2Len;
            this->attenB1SSOffset += actualS1Len * actualS2Len;
            this->boIdx++;
        } else {
            break;
        }
    }
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::ComputeAxisIdx(int64_t multiCoreInnerIdx)
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
    int64_t actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(actualS1Len, this->s1BaseSize) * this->n2G);
    while (multiCoreInnerIdx >= actualS1Outersize) {
        this->s1OuterSizeAcc = actualS1Outersize;
        this->s1SizeAcc += actualS1Len;
        this->s2SizeAcc += actualS2Len;
        this->attenB1SSOffset += actualS1Len * actualS2Len;
        this->boIdx++;
        GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
        actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(actualS1Len, this->s1BaseSize) * this->n2G);
    }
    // 计算轴的idx
    int64_t tmpS1Outersize = CeilDiv(actualS1Len, this->s1BaseSize);
    actualS1Outersize = multiCoreInnerIdx - this->s1OuterSizeAcc;
    this->n2oIdx = actualS1Outersize / tmpS1Outersize / this->tilingData->inputParams.gSize;
    this->goIdx = actualS1Outersize / tmpS1Outersize % this->tilingData->inputParams.gSize;
    this->s1oIdx = actualS1Outersize % tmpS1Outersize;
    GetSeqQlenKvlenByBoidx(this->boIdx, this->s1Size, this->s2Size);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::GetS2LoopRange()
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
    if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::CAUSAL)) { // 下三角
        this->s2StartIdx = 0;
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->s1BaseSize, actualS2Len);
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::RIGHT_DOWN_CAUSAL)) {
        this->s2StartIdx = 0;
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->s1BaseSize + actualS2Len - actualS1Len, actualS2Len);
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND)) {
        this->s2StartIdx = Max(
            this->s1oIdx * this->tilingData->coreParams.s1BaseSize - this->tilingData->coreParams.s1SparseValidSize, 0);
        this->s2EndIdx =
            Min((this->s1oIdx + 1) * this->s1BaseSize + this->tilingData->coreParams.s2SparseValidSize, actualS2Len);
        // s1baseSize行都无效时，需要将startIdx设置为0，,endIdx设置为S2realSize
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = actualS2Len;
        }
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND_COMPRESS)) {
        this->s2StartIdx = Max(this->s1oIdx * this->tilingData->coreParams.s1BaseSize - actualS1Len +
                                   Max(actualS2Len - this->tilingData->inputParams.preTokens, 0),
                               0);
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->s1BaseSize + actualS2Len -
                                 Max(actualS1Len - this->tilingData->inputParams.nextTokens, 0),
                             actualS2Len);
        // s1baseSize行都无效时，需要将startIdx设置为0，,endIdx设置为S2realSize
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = actualS2Len;
        }
    } else if (this->tilingData->inputParams.sparseType ==
               static_cast<uint8_t>(SparseModeEnum::RIGHT_DOWN_CAUSAL_BAND)) {
        if (this->boIdx == this->tilingData->inputParams.bandIndex) {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->s1BaseSize + actualS2Len +
                                     this->tilingData->inputParams.nextTokens - actualS1Len,
                                 actualS2Len);
        } else {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->s1BaseSize + actualS2Len - actualS1Len, actualS2Len);
        }
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND_LEFT_UP_CAUSAL)) {
        if (this->boIdx == this->tilingData->inputParams.bandIndex) {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->s1BaseSize + actualS2Len -
                                     Max(actualS1Len - this->tilingData->inputParams.nextTokens, 0),
                                 actualS2Len);
        } else {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->s1BaseSize, actualS2Len);
        }
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::PREFIX)) {
        this->s2StartIdx = 0;
        this->s2EndIdx =
            Max((this->s1oIdx + 1) * this->s1BaseSize - actualS1Len + actualS2Len,
                ((__gm__ int64_t *)prefixNAddr)[this->boIdx]);
        this->s2EndIdx = Min(this->s2EndIdx, this->s2Size);
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = actualS2Len;
        }
    } else { // 其它场景全计算
        this->s2StartIdx = 0;
        this->s2EndIdx = actualS2Len;
    }
    this->s2StartIdx = this->s2StartIdx / 8 * 8;
    this->s2EndIdx = Min(CeilDiv(this->s2EndIdx, 8) * 8, actualS2Len);
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::Process()
{
    // 确定核内切分起点
    int64_t multiCoreInnerOffset = this->blockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize;
    if (this->tilingData->multiCoreParams.totalSize < multiCoreInnerLimit) {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
    }
    // 计算sparse场景下s1的循环范围
    this->GetS1LoopRange(multiCoreInnerOffset, multiCoreInnerLimit);
    // 初始化AxisIdx
    this->CalS1OuterSize(multiCoreInnerOffset);

    SplitExtraInfo extraInfo[3];
    int64_t taskId = 0;
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
         ++multiCoreInnerIdx) {
        this->softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
        this->ComputeAxisIdx(multiCoreInnerIdx);
        // s2轴循环计数, 支持sparse和非sparse场景
        this->GetS2LoopRange();
        int64_t s2LoopLimit = CeilDiv(this->s2EndIdx - this->s2StartIdx, this->s2BaseNratioSize) - 1;
        for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
            if (s2LoopCount == 0) {
                this->softmaxPingPongCnt++;
            }
            if (taskId > 0) {
                // 对应extraInfo[(i+2)%3]
                this->WaitBmm1Result(extraInfo[(taskId + 2) % 3]);
            }
            this->SetExtraInfo(extraInfo[taskId % 3], taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);

            if (extraInfo[taskId % 3].needNz2Nd == 1) {
                this->IterateBmm1(extraInfo[taskId % 3], this->bmm1Nz);
            } else {
                this->IterateBmm1(extraInfo[taskId % 3], this->bmm1);
            }

            if (taskId > 0) {
                this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }

            if (taskId > 1) {
                // 对应extraInfo[(i+1)%3]
                this->WaitBmm2Result();
            }

            if (taskId > 0) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                this->IterateBmm2(extraInfo[(taskId + 2) % 3]);
            }

            if (taskId > 1) {
                this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
            }
            taskId++;
        }
    }
    if (taskId >= 1) {
        // 对应extraInfo[(i+2)%3]
        this->softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
        this->WaitBmm1Result(extraInfo[(taskId + 2) % 3]);
        this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        if (taskId > 1) {
            // 对应extraInfo[(i+1)%3]
            this->WaitBmm2Result();
        }
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        this->IterateBmm2(extraInfo[(taskId + 2) % 3]);
        if (taskId > 1) {
            this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
        }
    }
    taskId++;
    if (taskId >= 2) {
        // 对应extraInfo[(i+1)%3]
        this->WaitBmm2Result();
        this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
    }
};

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::ProcessL1Carry()
{
    // 确定核内切分起点
    int64_t multiCoreInnerOffset = this->blockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize;
    if (this->tilingData->multiCoreParams.totalSize < multiCoreInnerLimit) {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
    }
    // 计算sparse场景下s1的循环范围
    this->GetS1LoopRange(multiCoreInnerOffset, multiCoreInnerLimit);
    // 初始化AxisIdx
    this->CalS1OuterSize(multiCoreInnerOffset);

    SplitExtraInfo extraInfo[3];
    SplitExtraInfo pairExtraInfo[3];

    int64_t pairMultiCoreInnerOffset = this->pairBlockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    int64_t pairMultiCoreInnerLimit = pairMultiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize;
    if (this->tilingData->multiCoreParams.totalSize < pairMultiCoreInnerLimit) {
        pairMultiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
    }
    this->PairGetS1LoopRange(pairMultiCoreInnerOffset, pairMultiCoreInnerLimit);
    this->PairCalS1OuterSize(pairMultiCoreInnerOffset);

    int64_t realMultiCoreInnerLimit = multiCoreInnerLimit;
    multiCoreInnerLimit = (pairMultiCoreInnerLimit - pairMultiCoreInnerOffset) > (multiCoreInnerLimit - multiCoreInnerOffset) ? multiCoreInnerOffset + pairMultiCoreInnerLimit - pairMultiCoreInnerOffset : multiCoreInnerLimit;
    int64_t pairMultiCoreInnerIdx = pairMultiCoreInnerOffset;

    int64_t taskId = 0;
    int64_t pairTaskId = 0;
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    if ASCEND_IS_AIC {
        SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + 1);
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(P_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + 1);
        SetFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + 1);
    }
    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
         ++multiCoreInnerIdx) {
        this->softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
        int64_t s2LoopLimit = 0;
        int64_t pairS2LoopLimit = 0;
        if (multiCoreInnerIdx < realMultiCoreInnerLimit) {
            this->ComputeAxisIdx(multiCoreInnerIdx);
            // s2轴循环计数, 支持sparse和非sparse场景
            this->GetS2LoopRange();
            s2LoopLimit = CeilDiv(this->s2EndIdx - this->s2StartIdx, this->s2BaseNratioSize) - 1;
        }
        int64_t realS2LoopLimit = s2LoopLimit;
        if (pairMultiCoreInnerIdx < pairMultiCoreInnerLimit) {
            this->PairComputeAxisIdx(pairMultiCoreInnerIdx);
            this->PairGetS2LoopRange();
            pairS2LoopLimit = CeilDiv(this->pairS2EndIdx - this->pairS2StartIdx, this->s2BaseNratioSize) - 1;
            s2LoopLimit = pairS2LoopLimit > s2LoopLimit ? pairS2LoopLimit : s2LoopLimit;
        }
        for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
            bool needExec = multiCoreInnerIdx < realMultiCoreInnerLimit && s2LoopCount <= realS2LoopLimit;
            bool needPair = pairMultiCoreInnerIdx < pairMultiCoreInnerLimit && s2LoopCount <= pairS2LoopLimit;
            if (s2LoopCount == 0 && needExec) {
                this->softmaxPingPongCnt++;
            }
            if ASCEND_IS_AIV {
                if (pairTaskId > 0 && needPair) {
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V1_C2_FLAG[(pairTaskId + 1) & 1][!(this->blockIdx & 1)]);
                }
                if (taskId > 0 && needExec) {
                    CrossCoreWaitFlag(SYNC_C1_V1_FLAG[(taskId + 1) & 1][this->blockIdx & 1]);
                }
            }
            if (needExec) {
                this->SetExtraInfo(extraInfo[taskId % 3], taskId, s2LoopCount, realS2LoopLimit, multiCoreInnerIdx);
            }
            if ASCEND_IS_AIC {
                if (needExec) {
                    if (extraInfo[taskId % 3].needNz2Nd == 1) {
                        this->ComputeMm1(extraInfo[taskId % 3], this->bmm1NzL1Carry, 0);
                    } else {
                        this->ComputeMm1(extraInfo[taskId % 3], this->bmm1L1Carry, 0);
                    }
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG[taskId & 1][0]);
                }
                if (needPair) {
                    this->PairSetExtraInfo(pairExtraInfo[pairTaskId % 3], pairTaskId, s2LoopCount, pairS2LoopLimit, pairMultiCoreInnerIdx);
                     if (pairExtraInfo[pairTaskId % 3].needNz2Nd == 1) {
                        this->ComputeMm1(pairExtraInfo[pairTaskId % 3], this->bmm1NzL1Carry, 1);
                    } else {
                        this->ComputeMm1(pairExtraInfo[pairTaskId % 3], this->bmm1L1Carry, 1);
                    }
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG[pairTaskId & 1][1]);                   
                }
            }
            if ASCEND_IS_AIV {
                if (taskId > 0 && needExec) {
                    this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
                    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V1_C2_FLAG[(taskId + 1) & 1][this->blockIdx & 1]);
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                }
                if (pairTaskId > 0 && needPair) {
                    CrossCoreWaitFlag(SYNC_C1_V1_FLAG[(pairTaskId + 1) & 1][!(this->blockIdx & 1)]);
                }
                if (taskId > 1 && needExec) {
                    CrossCoreWaitFlag(SYNC_C2_V2_FLAG[taskId & 1][this->blockIdx & 1]);
                    this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
                }
                if (pairTaskId > 1 && needPair) {
                    CrossCoreWaitFlag(SYNC_C2_V2_FLAG[pairTaskId & 1][!(this->blockIdx & 1)]);
                }
            }
            if ASCEND_IS_AIC {
                if (taskId > 0 && needExec) {
                    CrossCoreWaitFlag(SYNC_V1_C2_FLAG[(taskId + 1) & 1][0]);
                    this->ComputeMm2(extraInfo[(taskId + 2) % 3], 0);
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C2_V2_FLAG[(taskId + 1) & 1][0]);
                }
                if (pairTaskId > 0 && needPair) {
                    CrossCoreWaitFlag(SYNC_V1_C2_FLAG[(pairTaskId + 1) & 1][1]);
                    this->ComputeMm2(pairExtraInfo[(pairTaskId + 2) % 3], 1);
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C2_V2_FLAG[(pairTaskId + 1) & 1][1]);                    
                }
            }
            if (needExec) {
                taskId++;
            }
            if (needPair) {
                pairTaskId++;
            }
        }
        pairMultiCoreInnerIdx++;
    }
    if (pairTaskId >= 1) {
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V1_C2_FLAG[(pairTaskId + 1) & 1][!(this->blockIdx & 1)]);
        }
    }
    if (taskId >= 1) {
        // 对应extraInfo[(i+2)%3]
        this->softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(SYNC_C1_V1_FLAG[(taskId + 1) & 1][this->blockIdx & 1]);
            this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V1_C2_FLAG[(taskId + 1) & 1][this->blockIdx & 1]);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag(SYNC_V1_C2_FLAG[(taskId + 1) & 1][0]);
            this->ComputeMm2(extraInfo[(taskId + 2) % 3], 0);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C2_V2_FLAG[(taskId + 1) & 1][0]);
        }
        if (taskId > 1) {
            if ASCEND_IS_AIV {
                CrossCoreWaitFlag(SYNC_C2_V2_FLAG[taskId & 1][this->blockIdx & 1]);
                this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
            }
        }
    }
    if (pairTaskId >= 1) {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag(SYNC_V1_C2_FLAG[(pairTaskId + 1) & 1][1]);
            this->ComputeMm2(pairExtraInfo[(pairTaskId + 2) % 3], 1);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C2_V2_FLAG[(pairTaskId + 1) & 1][1]);               
        }
    }
    taskId++;
    if (taskId >= 2) {
        if ASCEND_IS_AIV {
            // 对应extraInfo[(i+1)%3]
            CrossCoreWaitFlag(SYNC_C2_V2_FLAG[taskId & 1][this->blockIdx & 1]);
            this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
        }
    }
    if ASCEND_IS_AIC {
        WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + 1);
        if (taskId >= 2) {
            WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (this->kvL1BufIter % 2));
            WaitFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + (this->pL1BufIter % 2));
            WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (this->kvL1BufIter % 2) ^ 1);
            WaitFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + (this->pL1BufIter % 2) ^ 1);
        } else {
            WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
            WaitFlag<HardEvent::MTE1_MTE2>(P_EVENT0);
            WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + 1);
            WaitFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + 1);
        }
    }   
};

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::SetExtraInfo(SplitExtraInfo &extraInfo,
                                                                                                  int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    extraInfo.s2StartIdx = this->s2StartIdx;
    extraInfo.s2EndIdx = this->s2EndIdx;
    extraInfo.s2LoopCount = s2LoopCount;
    extraInfo.s1oIdx = this->s1oIdx;
    extraInfo.boIdx = this->boIdx;
    extraInfo.n2oIdx = this->n2oIdx;
    extraInfo.goIdx = this->goIdx;
    extraInfo.taskId = taskId;
    extraInfo.taskIdMod2 = taskId % 2;
    extraInfo.s2LoopLimit = s2LoopLimit;
    extraInfo.multiCoreInnerIdx = multiCoreInnerIdx;
    extraInfo.multiCoreInnerIdxMod2 = this->softmaxPingPongCnt % 2;
    extraInfo.s1SizeAcc = s1SizeAcc;
    extraInfo.s2SizeAcc = s2SizeAcc;
    extraInfo.attenB1SSOffset = attenB1SSOffset;
    extraInfo.attenMaskS2Size = extraInfo.s2Size;
    if (this->tilingData->inputParams.attenMaskShapeType == attenMaskS1S2) {
        extraInfo.attenMaskS2Size = this->tilingData->inputParams.s2Size;
    } else if (this->tilingData->inputParams.attenMaskShapeType == attenMaskTT) {
        extraInfo.attenMaskS2Size = this->s2SizeSum;
    }
    // band compress mode
    if (this->tilingData->inputParams.attenMaskCompressMode !=
        static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
        extraInfo.attenMaskS2Size = this->tilingData->inputParams.attenMaskS2Size;
    }
    GetSeqQlenKvlenByBoidx(extraInfo.boIdx, extraInfo.s1Size, extraInfo.s2Size);
    this->ComputeBmm1Tail(extraInfo);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::ComputeBmm1Tail(SplitExtraInfo &extraInfo)
{
    extraInfo.s1RealSize = this->s1BaseSize;
    if (extraInfo.s1Size < (extraInfo.s1oIdx + 1) * this->s1BaseSize) {
        extraInfo.s1RealSize = extraInfo.s1Size - extraInfo.s1oIdx * this->s1BaseSize;
    }
    extraInfo.s2RealSize = this->s2BaseNratioSize;
    extraInfo.s2AlignedSize = extraInfo.s2RealSize;
    if (extraInfo.s2StartIdx + (extraInfo.s2LoopCount + 1) * extraInfo.s2RealSize > extraInfo.s2EndIdx) {
        extraInfo.s2RealSize = extraInfo.s2EndIdx - extraInfo.s2LoopCount * extraInfo.s2RealSize - extraInfo.s2StartIdx;
        extraInfo.s2AlignedSize = Align(extraInfo.s2RealSize);
    }
    // In scenes where s2 is less than 8, when traversing s1basesize for computation, there is a memory issue.
    // Therefore, it's changed to compute once
    if (unlikely(extraInfo.s2RealSize < fp32BaseSize)) {
        extraInfo.vec1S1BaseSize = extraInfo.s1RealSize;
        extraInfo.realSplitN = 1;
    } else {
        extraInfo.vec1S1BaseSize = Min(1024 / extraInfo.s2AlignedSize * 8, extraInfo.s1RealSize); // Maximize the ub
        extraInfo.realSplitN = CeilDiv(extraInfo.s1RealSize, extraInfo.vec1S1BaseSize);
    }
    if (this->dSizeAlign16 > 64) {
        extraInfo.vec2S1BaseSize = 1024 / this->dSizeAlign16 * 8;
    } else {
        extraInfo.vec2S1BaseSize = Min(128, extraInfo.s1RealSize);
    }
    extraInfo.needNz2Nd = (extraInfo.s2RealSize % 64 == 0) ? 0 : 1;
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::CalcQCoreOffset(const SplitExtraInfo &extraInfo)
{
    int64_t bOffset = extraInfo.s1SizeAcc * this->n2GD;
    int64_t s1Offset = extraInfo.s1oIdx * this->s1BaseN2GD;
    int64_t n2Offset = extraInfo.n2oIdx * this->gD;
    int64_t gOffset = extraInfo.goIdx * this->dSize;
    this->qCoreOffset = bOffset + n2Offset + gOffset + s1Offset;
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::CalcKCoreOffset(const SplitExtraInfo &extraInfo)
{
    int64_t bOffset = extraInfo.s2SizeAcc * this->n2D;
    int64_t s2Offset = extraInfo.s2StartIdx * this->n2D + extraInfo.s2LoopCount * this->s2BaseNratioN2D;
    int64_t n2Offset = extraInfo.n2oIdx * this->dSize;
    this->kCoreOffset = bOffset + n2Offset + s2Offset;
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::CalcVCoreOffset(const SplitExtraInfo &extraInfo)
{
    int64_t bOffset = extraInfo.s2SizeAcc * this->n2D2;
    int64_t s2Offset = extraInfo.s2StartIdx * this->n2D2 + extraInfo.s2LoopCount * this->s2BaseNratioSize * this->n2D2;
    int64_t n2Offset = extraInfo.n2oIdx * this->d2Size;
    this->vCoreOffset = bOffset + n2Offset + s2Offset;
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
template <typename T2>          
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::ComputeMm1(const SplitExtraInfo &extraInfo,
                                                 matmul::MatmulImpl<a1TypeL1Carry, b1TypeL1Carry, T2, bias1Type, CFG_EXCEED> &bmm1, int32_t cubeSubIdx)
{
    bmm1.SetOrgShape(extraInfo.s1RealSize, this->mm1Kb, this->mm1Ka1, this->mm1Kb1, extraInfo.s2RealSize);
    LocalTensor<INPUT_T> qTensor = qL1Tensor[cubeSubIdx * L1_Q_SIZE / sizeof(INPUT_T)];
    if (extraInfo.s2LoopCount == 0) { // first loop
        WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + cubeSubIdx);
        CopyInMm1AToL1(qTensor, extraInfo);
        SetFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + cubeSubIdx);
        WaitFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + cubeSubIdx);
    }
    uint32_t nLoops = (extraInfo.s2RealSize + N_SPLIT_SIZE - 1) / N_SPLIT_SIZE;
    uint32_t nTail = extraInfo.s2RealSize - (nLoops - 1) * N_SPLIT_SIZE;
    uint32_t subNSizeAct = N_SPLIT_SIZE;
    for (uint32_t n = 0; n < nLoops; n++) {
        if (n == nLoops - 1) {
            subNSizeAct = nTail;
        }
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % 2));
        LocalTensor<INPUT_T> kTensor = kvL1Tensor[(kvL1BufIter % 2) * (L1_KV_SIZE / sizeof(INPUT_T))];
        CopyInMm1BToL1(kTensor, extraInfo, n, subNSizeAct); // 拷贝 256 * 192
        SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % 2));
        WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % 2));
        bmm1.SetTensorA(qTensor);
        bmm1.SetTensorB(kTensor, true);
        bmm1.SetTail(extraInfo.s1RealSize, subNSizeAct);
        if (extraInfo.needNz2Nd == 1) {
            bmm1.template IterateAll<false>(this->mm1Res[extraInfo.taskIdMod2][cubeSubIdx * this->totalOffset / sizeof(T) + extraInfo.s1RealSize * n * N_SPLIT_SIZE]);
        }else{
            bmm1.template IterateAll<false>(this->mm1Res[extraInfo.taskIdMod2][cubeSubIdx * this->totalOffset / sizeof(T) + n * N_SPLIT_SIZE]);
        }
        bmm1.End();
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % 2));
        kvL1BufIter++;
    }
    if (extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
        SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + cubeSubIdx);
    }    
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::ComputeMm2(const SplitExtraInfo &extraInfo, int32_t cubeSubIdx)
{
    this->bmm2L1Carry.SetOrgShape(extraInfo.s1RealSize, this->mm2Kb, extraInfo.s2AlignedSize, this->mm2Kb, this->d2Size);
    // 切K
    uint32_t kLoops = (extraInfo.s2RealSize + K_SPLIT_SIZE - 1) / K_SPLIT_SIZE;
    uint32_t kTail = extraInfo.s2RealSize - (kLoops - 1) * K_SPLIT_SIZE;
    uint32_t subKSizeAct = K_SPLIT_SIZE;
    uint32_t subKSizeActAligned = CeilDiv(subKSizeAct, 16) * 16;
    for (uint32_t k = 0; k < kLoops; k++) {
        if (k == kLoops - 1) {
            subKSizeAct = kTail;
            subKSizeActAligned = CeilDiv(kTail, 16) * 16;
        }
        WaitFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + (pL1BufIter % 2));
        LocalTensor<INPUT_T> pTensor = pL1Tensor[(pL1BufIter % 2) * (L1_P_SIZE / sizeof(INPUT_T))];
        CopyInMm2AToL1(pTensor, extraInfo, k, subKSizeActAligned, cubeSubIdx); // s1 * K_SPLIT_SIZE
        SetFlag<HardEvent::MTE2_MTE1>(P_EVENT0 + (pL1BufIter % 2));
        WaitFlag<HardEvent::MTE2_MTE1>(P_EVENT0 + (pL1BufIter % 2));

        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % 2));
        LocalTensor vTensor = kvL1Tensor[(kvL1BufIter % 2) * (L1_KV_SIZE / sizeof(INPUT_T))];
        CopyInMm2BToL1(vTensor, extraInfo, k, subKSizeAct); // K_SPLIT_SIZE * dSize
        SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % 2));
        WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % 2));
        this->bmm2L1Carry.SetTensorA(pTensor);
        this->bmm2L1Carry.SetTensorB(vTensor);
        this->bmm2L1Carry.SetTail(extraInfo.s1RealSize, this->d2Size, subKSizeAct);
        this->bmm2L1Carry.template Iterate(k != 0);
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % 2));
        SetFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + (pL1BufIter % 2));
        kvL1BufIter++;
        pL1BufIter++;
    }    
    this->bmm2L1Carry.template GetTensorC(this->mm2Res[extraInfo.taskIdMod2][cubeSubIdx * this->totalOffset / sizeof(T)]);
    this->bmm2L1Carry.End();
}         

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::CopyInMm1AToL1(LocalTensor<INPUT_T> &l1Tensor,
                                                                                                    const SplitExtraInfo &extraInfo)
{
    CalcQCoreOffset(extraInfo);
    auto srcGm = this->queryGm[this->qCoreOffset];
    CopyGmToL1(l1Tensor, srcGm, extraInfo.s1RealSize, this->dSize, this->n2GD);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::CopyInMm1BToL1(LocalTensor<INPUT_T> &l1Tensor,
                                                                                                    const SplitExtraInfo &extraInfo, uint32_t subNid, uint32_t subNSize)
{
    CalcKCoreOffset(extraInfo);
    auto srcGm = this->keyGm[this->kCoreOffset + subNid * N_SPLIT_SIZE * this->n2D];
    CopyGmToL1(l1Tensor, srcGm, subNSize, this->dSize, this->n2D);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::CopyInMm2AToL1(LocalTensor<INPUT_T> &l1Tensor,
                                                                                                    const SplitExtraInfo &extraInfo, uint32_t subKid, uint32_t subKSize, int32_t cubeSubIdx)
{
    int64_t s1RealSizeAligned = CeilDiv(extraInfo.s1RealSize, 16) * 16; 
    auto srcGm = this->stage1Res[extraInfo.taskIdMod2][cubeSubIdx * this->totalOffset / sizeof(INPUT_T) + s1RealSizeAligned * subKid * K_SPLIT_SIZE];        
    DataCopy(l1Tensor, srcGm, s1RealSizeAligned * subKSize);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::CopyInMm2BToL1(LocalTensor<INPUT_T> &l1Tensor,
                                                                                                    const SplitExtraInfo &extraInfo, uint32_t subKid, uint32_t subKSize)
{
    CalcVCoreOffset(extraInfo);
    auto srcGm = this->valueGm[this->vCoreOffset + subKid * K_SPLIT_SIZE * this->n2D2];
    CopyGmToL1(l1Tensor, srcGm, subKSize, this->d2Size, this->n2D2);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::CopyGmToL1(LocalTensor<INPUT_T> &l1Tensor, GlobalTensor<INPUT_T> &gmSrcTensor,
                                                                                                uint32_t srcN, uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // 行数
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = CeilDiv(srcN, 16) * 16; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    AscendC::DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::PairGetS1LoopRange(int64_t &multiCoreInnerOffset, int64_t &multiCoreInnerLimit)
{
    multiCoreInnerOffset = this->tilingData->multiCoreParams.sparseStartIdx[this->pairBlockIdx];
    if (likely((this->tilingData->multiCoreParams.coreNum - 1) > this->pairBlockIdx)) {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.sparseStartIdx[this->pairBlockIdx + 1];
    } else {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
    }    
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::PairCalS1OuterSize(int64_t offset)
{
    int64_t actualS1Outersize = 0;
    // 用于取actualS1Len下标
    this->pairBoIdx = 0;
    this->pairS1OuterSizeAcc = 0;
    this->pairAttenB1SSOffset = 0;
    this->pairS1SizeAcc = 0;
    this->pairS2SizeAcc = 0;
    int64_t actualS1Len;
    int64_t actualS2Len;
    for (int64_t i = 0; i < this->tilingData->inputParams.bSize; i++) {
        GetSeqQlenKvlenByBoidx(i, actualS1Len, actualS2Len);
        actualS1Outersize += (CeilDiv(actualS1Len, this->s1BaseSize) * this->n2G);
        if (offset >= actualS1Outersize) {
            this->pairS1OuterSizeAcc = actualS1Outersize;
            this->pairS1SizeAcc += actualS1Len;
            this->pairS2SizeAcc += actualS2Len;
            this->pairAttenB1SSOffset += actualS1Len * actualS2Len;
            this->pairBoIdx++;
        } else {
            break;
        }
    }
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::PairComputeAxisIdx(int64_t multiCoreInnerIdx)
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    GetSeqQlenKvlenByBoidx(this->pairBoIdx, actualS1Len, actualS2Len);
    int64_t actualS1Outersize = this->pairS1OuterSizeAcc + (CeilDiv(actualS1Len, this->s1BaseSize) * this->n2G);
    while (multiCoreInnerIdx >= actualS1Outersize) {
        this->pairS1OuterSizeAcc = actualS1Outersize;
        this->pairS1SizeAcc += actualS1Len;
        this->pairS2SizeAcc += actualS2Len;
        this->pairAttenB1SSOffset += actualS1Len * actualS2Len;
        this->pairBoIdx++;
        GetSeqQlenKvlenByBoidx(this->pairBoIdx, actualS1Len, actualS2Len);
        actualS1Outersize = this->pairS1OuterSizeAcc + (CeilDiv(actualS1Len, this->s1BaseSize) * this->n2G);
    }
    // 计算轴的idx
    int64_t tmpS1Outersize = CeilDiv(actualS1Len, this->s1BaseSize);
    actualS1Outersize = multiCoreInnerIdx - this->pairS1OuterSizeAcc;
    this->pairN2oIdx = actualS1Outersize / tmpS1Outersize / this->tilingData->inputParams.gSize;
    this->pairGoIdx = actualS1Outersize / tmpS1Outersize % this->tilingData->inputParams.gSize;
    this->pairS1oIdx = actualS1Outersize % tmpS1Outersize;
    GetSeqQlenKvlenByBoidx(this->pairBoIdx, this->s1Size, this->s2Size);
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::PairGetS2LoopRange()
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    GetSeqQlenKvlenByBoidx(this->pairBoIdx, actualS1Len, actualS2Len);
    if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::CAUSAL)) { // 下三角
        this->pairS2StartIdx = 0;
        this->pairS2EndIdx = Min((this->pairS1oIdx + 1) * this->s1BaseSize, actualS2Len);
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::RIGHT_DOWN_CAUSAL)) {
        this->pairS2StartIdx = 0;
        this->pairS2EndIdx = Min((this->pairS1oIdx + 1) * this->s1BaseSize + actualS2Len - actualS1Len, actualS2Len);
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND)) {
        this->pairS2StartIdx = Max(
            this->pairS1oIdx * this->tilingData->coreParams.s1BaseSize - this->tilingData->coreParams.s1SparseValidSize, 0);
        this->pairS2EndIdx =
            Min((this->pairS1oIdx + 1) * this->s1BaseSize + this->tilingData->coreParams.s2SparseValidSize, actualS2Len);
        // s1baseSize行都无效时，需要将startIdx设置为0，,endIdx设置为S2realSize
        if (this->pairS2EndIdx - this->pairS2StartIdx <= 0) {
            this->pairS2StartIdx = 0;
            this->pairS2EndIdx = actualS2Len;
        }
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND_COMPRESS)) {
        this->pairS2StartIdx = Max(this->pairS1oIdx * this->tilingData->coreParams.s1BaseSize - actualS1Len +
                                   Max(actualS2Len - this->tilingData->inputParams.preTokens, 0),
                               0);
        this->pairS2EndIdx = Min((this->pairS1oIdx + 1) * this->s1BaseSize + actualS2Len -
                                 Max(actualS1Len - this->tilingData->inputParams.nextTokens, 0),
                             actualS2Len);
        // s1baseSize行都无效时，需要将startIdx设置为0，,endIdx设置为S2realSize
        if (this->pairS2EndIdx - this->pairS2StartIdx <= 0) {
            this->pairS2StartIdx = 0;
            this->pairS2EndIdx = actualS2Len;
        }
    } else if (this->tilingData->inputParams.sparseType ==
               static_cast<uint8_t>(SparseModeEnum::RIGHT_DOWN_CAUSAL_BAND)) {
        if (this->pairBoIdx == this->tilingData->inputParams.bandIndex) {
            this->pairS2StartIdx = 0;
            this->pairS2EndIdx = Min((this->pairS1oIdx + 1) * this->s1BaseSize + actualS2Len +
                                     this->tilingData->inputParams.nextTokens - actualS1Len,
                                 actualS2Len);
        } else {
            this->pairS2StartIdx = 0;
            this->pairS2EndIdx = Min((this->pairS1oIdx + 1) * this->s1BaseSize + actualS2Len - actualS1Len, actualS2Len);
        }
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND_LEFT_UP_CAUSAL)) {
        if (this->pairBoIdx == this->tilingData->inputParams.bandIndex) {
            this->pairS2StartIdx = 0;
            this->pairS2EndIdx = Min((this->pairS1oIdx + 1) * this->s1BaseSize + actualS2Len -
                                     Max(actualS1Len - this->tilingData->inputParams.nextTokens, 0),
                                 actualS2Len);
        } else {
            this->pairS2StartIdx = 0;
            this->pairS2EndIdx = Min((this->pairS1oIdx + 1) * this->s1BaseSize, actualS2Len);
        }
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::PREFIX)) {
        this->pairS2StartIdx = 0;
        this->pairS2EndIdx =
            Max((this->pairS1oIdx + 1) * this->s1BaseSize - actualS1Len + actualS2Len,
                ((__gm__ int64_t *)prefixNAddr)[this->pairBoIdx]);
        this->pairS2EndIdx = Min(this->pairS2EndIdx, this->s2Size);
        if (this->pairS2EndIdx - this->pairS2StartIdx <= 0) {
            this->pairS2StartIdx = 0;
            this->pairS2EndIdx = actualS2Len;
        }
    } else { // 其它场景全计算
        this->pairS2StartIdx = 0;
        this->pairS2EndIdx = actualS2Len;
    }
    this->pairS2StartIdx = this->pairS2StartIdx / 8 * 8;
    this->pairS2EndIdx = Min(CeilDiv(this->pairS2EndIdx, 8) * 8, actualS2Len);
    return;
}

template <ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, bool isBasicBlock, CubeFormat bmm1Format, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScore<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                 isBasicBlock, bmm1Format, hasRope>::PairSetExtraInfo(SplitExtraInfo &extraInfo,
                                                                                                      int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    extraInfo.s2StartIdx = this->pairS2StartIdx;
    extraInfo.s2EndIdx = this->pairS2EndIdx;
    extraInfo.s2LoopCount = s2LoopCount;
    extraInfo.s1oIdx = this->pairS1oIdx;
    extraInfo.boIdx = this->pairBoIdx;
    extraInfo.n2oIdx = this->pairN2oIdx;
    extraInfo.goIdx = this->pairGoIdx;
    extraInfo.taskId = taskId;
    extraInfo.taskIdMod2 = taskId % 2;
    extraInfo.s2LoopLimit = s2LoopLimit;
    extraInfo.multiCoreInnerIdx = multiCoreInnerIdx;
    extraInfo.multiCoreInnerIdxMod2 = this->softmaxPingPongCnt % 2;
    extraInfo.s1SizeAcc = this->pairS1SizeAcc;
    extraInfo.s2SizeAcc = this->pairS2SizeAcc;
    extraInfo.attenB1SSOffset = this->pairAttenB1SSOffset;
    extraInfo.attenMaskS2Size = extraInfo.s2Size;
    GetSeqQlenKvlenByBoidx(extraInfo.boIdx, extraInfo.s1Size, extraInfo.s2Size);
    this->ComputeBmm1Tail(extraInfo);
}
#endif // FLASH_ATTENTION_VAR_LEN_SCORE_H
