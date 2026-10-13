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
 * \file flash_attention_var_len_score_sab.h
 * \brief
 */

#ifndef FLASH_ATTENTION_VAR_LEN_SCORE_SAB_H
#define FLASH_ATTENTION_VAR_LEN_SCORE_SAB_H

#include "flash_attention_score_s1s2_bn2gs1_sab.h"

// INPUT_T - means data type for input
// T       - means data type when calc
#define Q_EVENT0  EVENT_ID2
#define KV_EVENT0 EVENT_ID4
#define P_EVENT0  EVENT_ID6
template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T, typename T = INPUT_T,
          CubeFormat bmm1Format = CubeFormat::NZ, MmPolicyType mmPolicyType = MmPolicyType::NORMAL,
          ImplModeEnum implMode = ImplModeEnum::AA_HIGH_PRECISION, bool hasRope = false>
class FlashAttentionVarLenScoreSameAB
    : public FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                           bmm1Format, mmPolicyType, hasRope> {
public:
    __aicore__ inline FlashAttentionVarLenScoreSameAB(){};

    __aicore__ inline void
    UnpackInit(__gm__ uint8_t *query, __gm__ uint8_t *queryRope, __gm__ uint8_t *key, __gm__ uint8_t *keyRope,
               __gm__ uint8_t *value, __gm__ uint8_t *pse,
               __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, __gm__ uint8_t *sink,
               __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *softmaxMax,
               __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut,
               __gm__ uint8_t *workspace, const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);

    __aicore__ inline void Process();

    static constexpr MatmulConfig CFG_CUSTOM_EXCEED = (mmPolicyType == MmPolicyType::UNSPLITK) ? CFG_DIS_UNIT_FLAG_EXCEED : CFG_EXCEED;

     // define matmul
    using a1TypeL1Carry = MatmulType<TPosition::A1, CubeFormat::NZ, INPUT_T, false, LayoutMode::NONE, true>;
    using b1TypeL1Carry = MatmulType<TPosition::B1, CubeFormat::NZ, INPUT_T, true,  LayoutMode::NONE, true>;
    using bias1TypeL1Carry = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c1TypeL1Carry = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using mm1TypeL1Carry = matmul::MatmulImpl<a1TypeL1Carry, b1TypeL1Carry, c1TypeL1Carry, bias1TypeL1Carry, CFG_CUSTOM_EXCEED>;
    mm1TypeL1Carry bmm1L1Carry;

    using c1NzTypeL1Carry = MatmulType<TPosition::GM, CubeFormat::NZ, T>;
    using mm1NzTypeL1Carry = matmul::MatmulImpl<a1TypeL1Carry, b1TypeL1Carry, c1NzTypeL1Carry, bias1TypeL1Carry, CFG_CUSTOM_EXCEED>;
    mm1NzTypeL1Carry bmm1NzL1Carry;

    // define batchmatmul
    using a2TypeL1Carry = MatmulType<TPosition::A1, CubeFormat::NZ, INPUT_T, false, LayoutMode::NONE, true>;
    using b2TypeL1Carry = MatmulType<TPosition::B1, CubeFormat::NZ, INPUT_T, false, LayoutMode::NONE, true>;
    using bias2TypeL1Carry = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c2NzTypeL1Carry = MatmulType<TPosition::GM, CubeFormat::NZ, T>;
    using mm2TypeL1Carry = matmul::MatmulImpl<a2TypeL1Carry, b2TypeL1Carry, c2NzTypeL1Carry, bias2TypeL1Carry, CFG_MDL_EXCEED>;
    mm2TypeL1Carry bmm2L1Carry;
    
protected:
    __aicore__ inline void ComputeConstexpr();

    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);

    __aicore__ inline void ComputeBmm1Tail(SplitSameABExtraInfo &extraInfo);

    __aicore__ inline void SetExtraInfo(SplitSameABExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount,
                                        int64_t s2LoopLimit, int64_t multiCoreInnerIdx);

    __aicore__ inline void CalS1OuterSize(int64_t offset);

    __aicore__ inline void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen);

    __aicore__ inline void GetS2LoopRange();
    template <typename T2>
    __aicore__ inline void ComputeMm1(SplitSameABExtraInfo &extraInfo, 
                                      matmul::MatmulImpl<a1TypeL1Carry, b1TypeL1Carry, T2, bias1TypeL1Carry, CFG_CUSTOM_EXCEED> &bmm1);
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<INPUT_T> &l1Tensor, SplitSameABExtraInfo &extraInfo);
    __aicore__ inline void CopyInMm1BToL1(LocalTensor<INPUT_T> &bL1Tensor, const SplitSameABExtraInfo &extraInfo, 
                                          uint32_t subNid, uint32_t subNSize, uint32_t nSplitSize);
    __aicore__ inline void ComputeMm2(SplitSameABExtraInfo &extraInfo,
                                      matmul::MatmulImpl<a2TypeL1Carry, b2TypeL1Carry, c2NzTypeL1Carry, bias2TypeL1Carry, CFG_MDL_EXCEED> &bmm2);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<INPUT_T> &al1Tensor, SplitSameABExtraInfo &extraInfo, 
                                          uint32_t subKid, uint32_t subKSize, uint32_t kSplitSize);
    __aicore__ inline void CopyInMm2BToL1(LocalTensor<INPUT_T> &bL1Tensor,  
                                          uint32_t subKid, uint32_t subKSize, uint32_t kSplitSize, uint32_t nLoopIdx, bool isLastN);
    __aicore__ inline void CopyGmToL1(LocalTensor<INPUT_T> &l1Tensor, GlobalTensor<INPUT_T> &gmSrcTensor, uint32_t srcN, 
                                      uint32_t srcD, uint32_t srcDstride);
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
    uint64_t SYNC_C1_V1_FLAG[3] = {4, 5, 6};
    uint64_t SYNC_V1_C2_FLAG[3] = {7, 8, 9};
    uint64_t SYNC_C2_V2_FLAG[3] = {1, 2, 3};
    uint32_t L1_Q_SIZE = (192 * 256 * sizeof(INPUT_T));
    uint32_t L1_KV_SIZE = (256 * 192 * sizeof(INPUT_T));
    uint32_t L1_P_SIZE = (256 * 192 * sizeof(INPUT_T));
    uint32_t L0C_SIZE = (256 * 192 * sizeof(T));
    TBuf<TPosition::A1> queryBufL1;
    LocalTensor<INPUT_T> qL1Tensor;
    TBuf<TPosition::B1> kvBufL1;
    LocalTensor<INPUT_T> kvL1Tensor;
    TBuf<TPosition::A1> pBufL1;
    LocalTensor<INPUT_T> pL1Tensor;
    // L0C
    TBuf<TPosition::CO1> oBufL0C;
    LocalTensor<T> oL0CTensor;
    int64_t kCoreOffset;
    int64_t vCoreOffset;
    uint32_t kvL1BufIter = 0;
    uint32_t pL1BufIter = 0;
    bool needL1Carry = false;
};

template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                       bmm1Format, mmPolicyType, implMode, hasRope>::UnpackInit(
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
        if constexpr (hasRope == false) { // Only enable L1Carry feature when has no rope.
            if (this->cubeS1BaseSize <= 256 && 
               (this->dSize == 192 || this->dSize == 128 || this->dSize == 88 || this->dSize == 80)) {
                this->needL1Carry = true;
                this->bmm1L1Carry .SetSubBlockIdx(0);
                this->bmm1L1Carry .Init(&tiling->bmm1TilingData, tPipe);
                this->bmm1NzL1Carry .SetSubBlockIdx(0);
                this->bmm1NzL1Carry .Init(&tiling->bmm1TilingData, tPipe);
                this->bmm2L1Carry .SetSubBlockIdx(0);
                this->bmm2L1Carry .Init(&tiling->bmm2TilingData, tPipe);
            }
        }
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
                actualS1Len = this->s1Size -accumSize;
                int64_t frontCoreNum = actualS1Len % this->tilingData->multiCoreParams.coreNum;
                int64_t splitFactor = frontCoreNum > 0 ? 1 : 0;
                int64_t s1SizeInner = (actualS1Len / this->tilingData->multiCoreParams.coreNum);
                int64_t innerOffset1 = (s1SizeInner + splitFactor) * (this->vecBlockIdx >= frontCoreNum ?
                                       frontCoreNum : this->vecBlockIdx);
                int64_t innerOffset2 = s1SizeInner * (this->vecBlockIdx >= frontCoreNum ?
                                       this->vecBlockIdx - frontCoreNum : 0);
                accumSize = accumSize + innerOffset1 + innerOffset2;
                actualS1Len = s1SizeInner + (this->vecBlockIdx >= frontCoreNum ? 0 : splitFactor);
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
    if (this->needL1Carry) {
        this->pipe->InitBuffer(this->queryBufL1, L1_Q_SIZE);
        qL1Tensor = this->queryBufL1.template Get<INPUT_T>();
        this->pipe->InitBuffer(this->kvBufL1, L1_KV_SIZE * 2);
        kvL1Tensor = this->kvBufL1.template Get<INPUT_T>();
        this->pipe->InitBuffer(this->pBufL1, L1_P_SIZE * 2);
        pL1Tensor = this->pBufL1.template Get<INPUT_T>();
        this->pipe->InitBuffer(this->oBufL0C, L0C_SIZE);
        oL0CTensor = this->oBufL0C.template Get<T>();
    }
    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    DropOutBitModeInit(apiTmpBuffer);
    if (this->vecBlockIdx < this->tilingData->multiCoreParams.coreNum) {
        LocalTensor<half> pseHelpBuffer = this->stage1PingBuf.template Get<half>();
        PseInnerAlibiCreate<hasPse>(this->pseAlibiGm, pseHelpBuffer, this->pseInfo);
    }
    prefixNAddr = prefix;
    this->s2SizeSum = ((__gm__ uint64_t *)actualSeqLengthsKv)[this->tilingData->inputParams.bSize - 1];
    return;
}

template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                       bmm1Format, mmPolicyType, implMode, hasRope>::ComputeConstexpr()
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
    this->s1BaseN2GD = this->cubeS1BaseSize * this->n2GD;
    this->s1BaseN2GD2 = this->cubeS1BaseSize * this->n2GD2;
    this->s2BaseNratioN2D = this->s2BaseN2D * this->tilingData->coreParams.nRatio;

    if constexpr (hasRope == true) {
        this->gDRope = this->tilingData->inputParams.gSize * this->dRopeSize;
        this->n2DRope = this->tilingData->inputParams.n2Size * this->dRopeSize;
        this->n2GDRope = this->tilingData->inputParams.n2Size * this->gDRope;
        this->s2BaseN2DRope = this->s2BaseSize * this->n2DRope;
        this->s1BaseN2GDRope = this->cubeS1BaseSize * this->n2GDRope;
        this->s2BaseNratioN2DRope = this->s2BaseN2DRope * this->tilingData->coreParams.nRatio;
        
        this->mm1Ka2 = this->n2GDRope;
        this->mm1Kb2 = this->n2DRope;
    }

    // layout(bs)ND
    this->mm1Ka1 = this->n2GD;
    this->mm1Kb1 = this->n2D;
    this->mm1Kb = this->mm1Kb1 + this->mm1Kb2;
    this->mm2Kb = this->n2D2;
    if constexpr (hasPse == true) {
        this->pseInfo.gSize = this->tilingData->inputParams.gSize;
        this->pseInfo.pseShapeType = this->tilingData->inputParams.pseShapeType;
        this->pseInfo.n2G = this->n2G;
        this->pseInfo.pseBSize = this->tilingData->inputParams.pseBSize;
        this->pseInfo.s1BaseSize = this->cubeS1BaseSize;
        this->pseInfo.pseS1Size = this->tilingData->inputParams.pseS1Size;
        this->pseInfo.pseS2Size = this->tilingData->inputParams.pseS2Size;
        this->pseInfo.s2BaseNratioSize = this->s2BaseNratioSize;
        this->pseInfo.pseEncodeType = (uint32_t)this->tilingData->inputParams.pseEncodeType;
        this->pseInfo.pseType = this->tilingData->inputParams.pseType;
        this->pseInfo.pseAlibiBaseS1 = this->tilingData->coreParams.pseAlibiBaseS1;
        this->pseInfo.pseAlibiBaseS2 = this->tilingData->coreParams.pseAlibiBaseS2;
        this->pseInfo.qStartIdx = this->tilingData->inputParams.qStartIdx;
        this->pseInfo.kvStartIdx = this->tilingData->inputParams.kvStartIdx;
    }

    if constexpr (hasDrop == true) {
        this->dropMaskInfo.gSize = this->tilingData->inputParams.gSize;
        this->dropMaskInfo.n2G = this->n2G;
        this->dropMaskInfo.s1BaseSize = this->cubeS1BaseSize;
        this->dropMaskInfo.s2BaseNratioSize = this->s2BaseNratioSize;
    }
}

template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void
FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, implMode, hasRope>::GetSeqQlenKvlenByBoidx(int64_t boIdx,
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
template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                       bmm1Format, mmPolicyType, implMode, hasRope>::CalS1OuterSize(int64_t offset)
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
        actualS1Outersize += (CeilDiv(actualS1Len, this->cubeS1BaseSize) * this->n2G);
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

template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                       bmm1Format, mmPolicyType, implMode, hasRope>::ComputeAxisIdx(int64_t multiCoreInnerIdx)
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
    int64_t actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(actualS1Len, this->cubeS1BaseSize) * this->n2G);
    while (multiCoreInnerIdx >= actualS1Outersize) {
        this->s1OuterSizeAcc = actualS1Outersize;
        this->s1SizeAcc += actualS1Len;
        this->s2SizeAcc += actualS2Len;
        this->attenB1SSOffset += actualS1Len * actualS2Len;
        this->boIdx++;
        GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
        actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(actualS1Len, this->cubeS1BaseSize) * this->n2G);
    }
    // 计算轴的idx
    int64_t tmpS1Outersize = CeilDiv(actualS1Len, this->cubeS1BaseSize);
    actualS1Outersize = multiCoreInnerIdx - this->s1OuterSizeAcc;
    this->n2oIdx = actualS1Outersize / tmpS1Outersize / this->tilingData->inputParams.gSize;
    this->goIdx = actualS1Outersize / tmpS1Outersize % this->tilingData->inputParams.gSize;
    this->s1oIdx = actualS1Outersize % tmpS1Outersize;
    GetSeqQlenKvlenByBoidx(this->boIdx, this->s1Size, this->s2Size);
}

template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                       bmm1Format, mmPolicyType, implMode, hasRope>::GetS2LoopRange()
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
    if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::CAUSAL)) { // 下三角
        this->s2StartIdx = 0;
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize, actualS2Len);
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::RIGHT_DOWN_CAUSAL)) {
        this->s2StartIdx = 0;
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len - actualS1Len, actualS2Len);
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND)) {
        this->s2StartIdx = Max(
            this->s1oIdx * this->cubeS1BaseSize - this->tilingData->coreParams.s1SparseValidSize, 0);
        this->s2EndIdx =
            Min((this->s1oIdx + 1) * this->cubeS1BaseSize + this->tilingData->coreParams.s2SparseValidSize, actualS2Len);
        // s1baseSize行都无效时，需要将startIdx设置为0，,endIdx设置为S2realSize
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = actualS2Len;
        }
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND_COMPRESS)) {
        this->s2StartIdx = Max(this->s1oIdx * this->cubeS1BaseSize - actualS1Len +
                                   Max(actualS2Len - this->tilingData->inputParams.preTokens, 0),
                               0);
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len -
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
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len +
                                     this->tilingData->inputParams.nextTokens - actualS1Len,
                                 actualS2Len);
        } else {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len - actualS1Len, actualS2Len);
        }
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND_LEFT_UP_CAUSAL)) {
        if (this->boIdx == this->tilingData->inputParams.bandIndex) {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len -
                                     Max(actualS1Len - this->tilingData->inputParams.nextTokens, 0),
                                 actualS2Len);
        } else {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize, actualS2Len);
        }
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::PREFIX)) {
        this->s2StartIdx = 0;
        this->s2EndIdx =
            Max((this->s1oIdx + 1) * this->cubeS1BaseSize - actualS1Len + actualS2Len,
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

template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                       bmm1Format, mmPolicyType, implMode, hasRope>::Process()
{
    // 确定核内切分起点
    int64_t multiCoreInnerOffset = this->cubeBlockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize;
    if (this->tilingData->multiCoreParams.totalSize < multiCoreInnerLimit) {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
    }
    // 计算sparse场景下s1的循环范围
    this->GetS1LoopRange(multiCoreInnerOffset, multiCoreInnerLimit);
    // 初始化AxisIdx
    this->CalS1OuterSize(multiCoreInnerOffset);

    SplitSameABExtraInfo extraInfo[3];
    int64_t taskId = 0;
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    if ASCEND_IS_AIC {
        if (this->needL1Carry) {
            SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
            SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
            SetFlag<HardEvent::MTE1_MTE2>(P_EVENT0);
            SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + 1);
            SetFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + 1);
        }
    }
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
                if ASCEND_IS_AIV {
                    CrossCoreWaitFlag(SYNC_C1_V1_FLAG[(taskId + 2) % 3]);
                }
            }
            this->SetExtraInfo(extraInfo[taskId % 3], taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);

            if (extraInfo[taskId % 3].needNz2Nd == 1) {
                if ASCEND_IS_AIC { 
                    if (this->needL1Carry) {
                        this->ComputeMm1(extraInfo[taskId % 3], this->bmm1NzL1Carry);
                    } else {
                        this->IterateBmm1(extraInfo[taskId % 3], this->bmm1Nz);
                    }
                    CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C1_V1_FLAG[taskId % 3]);
                }  
            } else {    
                if ASCEND_IS_AIC {  
                    if (this->needL1Carry) {
                        this->ComputeMm1(extraInfo[taskId % 3], this->bmm1L1Carry);
                    } else {
                        this->IterateBmm1(extraInfo[taskId % 3], this->bmm1);
                    }                    
                    CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C1_V1_FLAG[taskId % 3]);
                }  
            }

            if (taskId > 0) {
                if ASCEND_IS_AIV {
                    this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
                    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);                
                    CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V1_C2_FLAG[(taskId + 2) % 3]);
                }       
            }
            if (taskId > 1) {
                if ASCEND_IS_AIV {
                    CrossCoreWaitFlag(SYNC_C2_V2_FLAG[(taskId + 1) % 3]);
                }
            }

            if (taskId > 0) {
                if ASCEND_IS_AIV {
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                }
                
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag(SYNC_V1_C2_FLAG[(taskId + 2) % 3]);
                    if (this->needL1Carry) {
                        this->ComputeMm2(extraInfo[(taskId + 2) % 3], this->bmm2L1Carry);
                    } else {
                        this->IterateBmm2(extraInfo[(taskId + 2) % 3], this->bmm2);
                    }
                    CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C2_V2_FLAG[(taskId + 2) % 3]);
                }      
            }
            
            if (taskId > 1) {
                if ASCEND_IS_AIV {
                    this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
                }
            }
            taskId++;
        }
    }
    if (taskId >= 1) {
        this->softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(SYNC_C1_V1_FLAG[(taskId + 2) % 3]);
            this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V1_C2_FLAG[(taskId + 2) % 3]);
        }
        if (taskId > 1) {
            if ASCEND_IS_AIV {
                CrossCoreWaitFlag(SYNC_C2_V2_FLAG[(taskId + 1) % 3]);
            }
        }
        if ASCEND_IS_AIV {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag(SYNC_V1_C2_FLAG[(taskId + 2) % 3]);
            if (this->needL1Carry) {
                this->ComputeMm2(extraInfo[(taskId + 2) % 3], this->bmm2L1Carry);
            } else {
                this->IterateBmm2(extraInfo[(taskId + 2) % 3], this->bmm2);
            }
            CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C2_V2_FLAG[(taskId + 2) % 3]);
        }     
        if (taskId > 1) {
            if ASCEND_IS_AIV {
                this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
            }
        }
    }
    taskId++;
    if (taskId >= 2) {
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(SYNC_C2_V2_FLAG[(taskId + 1) % 3]);
            this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
        }
    }
    if ASCEND_IS_AIC {
        if (this->needL1Carry) {
            if (taskId >= 2) {
                WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
                WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (this->kvL1BufIter % 2));
                WaitFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + (this->pL1BufIter % 2));
                WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (this->kvL1BufIter % 2) ^ 1);
                WaitFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + (this->pL1BufIter % 2) ^ 1);
            } else {
                WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
                WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
                WaitFlag<HardEvent::MTE1_MTE2>(P_EVENT0);
                WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + 1);
                WaitFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + 1);
            }
        }
    }
};

template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                       bmm1Format, mmPolicyType, implMode, hasRope>::SetExtraInfo(
    SplitSameABExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
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
    extraInfo.cubeS1RealSize = Min(this->cubeS1BaseSize, extraInfo.s1Size - extraInfo.s1oIdx * this->cubeS1BaseSize);
    this->ComputeBmm1Tail(extraInfo);
}

template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                       bmm1Format, mmPolicyType, implMode, hasRope>::ComputeBmm1Tail(SplitSameABExtraInfo &extraInfo)
{
    extraInfo.s1RealSize = (extraInfo.cubeS1RealSize + 1) / 2;
    extraInfo.vecCoreOffset = this->cubeSubIdx * extraInfo.s1RealSize;
    if (this->cubeSubIdx == 1) {
        extraInfo.s1RealSize = extraInfo.cubeS1RealSize - extraInfo.s1RealSize;
    }
    extraInfo.s2RealSize = this->s2BaseNratioSize;
    extraInfo.s2AlignedSize = extraInfo.s2RealSize;
    bool isS2Tail = false;
    if (extraInfo.s2StartIdx + (extraInfo.s2LoopCount + 1) * extraInfo.s2RealSize >= extraInfo.s2EndIdx) {
        extraInfo.s2RealSize = extraInfo.s2EndIdx - extraInfo.s2LoopCount * extraInfo.s2RealSize - extraInfo.s2StartIdx;
        extraInfo.s2AlignedSize = Align(extraInfo.s2RealSize);
        isS2Tail = true;
    }

    extraInfo.s2RealSizeAlign64 = extraInfo.s2AlignedSize;
    extraInfo.s2RealSizeFloorAlign8 = extraInfo.s2AlignedSize;
    extraInfo.s2LastLoop = false;
    if (isS2Tail) {
        extraInfo.s2RealSizeAlign64 = CeilDiv(extraInfo.s2RealSize, 64) * 64;
        extraInfo.s2RealSizeFloorAlign8 = extraInfo.s2RealSize / 8 * 8;
        extraInfo.s2LastLoop = true;
        extraInfo.duplicateMask =
            (((uint64_t)1 << (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSizeFloorAlign8)) - 1) &
            (~(((uint64_t)1 << (extraInfo.s2RealSize - extraInfo.s2RealSizeFloorAlign8)) - 1));
        if (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSizeFloorAlign8 == 64) {
            extraInfo.duplicateMask =
                0xffffffffffffffff & (~(((uint64_t)1 << (extraInfo.s2RealSize - extraInfo.s2RealSizeFloorAlign8)) - 1));
        }
    }

    // In scenes where s2 is less than 8, when traversing s1basesize for computation, there is a memory issue.
    // Therefore, it's changed to compute once
    if (unlikely(extraInfo.s2RealSize < fp32BaseSize)) {
        extraInfo.vec1S1BaseSize = extraInfo.s1RealSize;
        extraInfo.realSplitN = 1;
    } else {
        extraInfo.vec1S1BaseSize = Min(1024 / extraInfo.s2RealSizeAlign64 * 8, extraInfo.s1RealSize); // Maximize the ub
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

template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
template <typename T2>
__aicore__ inline void
FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, implMode, hasRope>::ComputeMm1(SplitSameABExtraInfo &extraInfo, 
                                    matmul::MatmulImpl<a1TypeL1Carry, b1TypeL1Carry, T2, bias1TypeL1Carry, CFG_CUSTOM_EXCEED> &bmm1) {

    bmm1.SetOrgShape(extraInfo.cubeS1RealSize, this->mm1Kb, this->mm1Ka1, this->mm1Kb1, extraInfo.s2RealSize);

    LocalTensor<INPUT_T> qTensor = qL1Tensor;

    if (extraInfo.s2LoopCount == 0) { // first loop
        WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
        CopyInMm1AToL1(qTensor, extraInfo);
        SetFlag<HardEvent::MTE2_MTE1>(Q_EVENT0);
        WaitFlag<HardEvent::MTE2_MTE1>(Q_EVENT0);
    }
    
    // 计算gm上的offset
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;

    bOffset = extraInfo.s2SizeAcc * this->n2D;
    s2Offset = extraInfo.s2StartIdx * this->n2D + extraInfo.s2LoopCount * this->s2BaseNratioN2D;
    n2Offset = extraInfo.n2oIdx * this->dSize;

    this->kCoreOffset = bOffset + n2Offset + s2Offset;

    const uint32_t nSplitSize = 256; // n方向切分
    uint32_t nloops = (extraInfo.s2RealSize + nSplitSize - 1) / nSplitSize;
    uint32_t nTail = extraInfo.s2RealSize - (nloops - 1) * nSplitSize;
    uint32_t subNSizeAct = nSplitSize;

    for (uint32_t n = 0; n < nloops; n++) {
        if (n == nloops - 1) {
            subNSizeAct = nTail;
        }
 
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % 2));
        LocalTensor<INPUT_T> kTensor = kvL1Tensor[(kvL1BufIter % 2) * (L1_KV_SIZE / sizeof(INPUT_T))];
 
        CopyInMm1BToL1(kTensor, extraInfo, n, subNSizeAct, nSplitSize); // 拷贝 256 * 192

        SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % 2));
        WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % 2));
        bmm1.SetTensorA(qTensor);
        bmm1.SetTensorB(kTensor, true);
        bmm1.SetTail(extraInfo.cubeS1RealSize, subNSizeAct);

        if(extraInfo.needNz2Nd == 1){
            bmm1.template IterateAll<false>(this->mm1Res[extraInfo.taskIdMod2][extraInfo.cubeS1RealSize * n * nSplitSize]);
        }else{
            bmm1.template IterateAll<false>(this->mm1Res[extraInfo.taskIdMod2][n * nSplitSize]);
        }
        bmm1.End();
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % 2));
        kvL1BufIter++;
    }
    if (extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
        SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
    }
}

template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                       bmm1Format, mmPolicyType, implMode, hasRope>::ComputeMm2(SplitSameABExtraInfo &extraInfo,
                                                            matmul::MatmulImpl<a2TypeL1Carry, b2TypeL1Carry, c2NzTypeL1Carry, bias2TypeL1Carry, CFG_MDL_EXCEED> &bmm2)
{
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;

    bOffset = extraInfo.s2SizeAcc * this->n2D2;
    s2Offset = extraInfo.s2StartIdx * this->n2D2 + extraInfo.s2LoopCount * this->s2BaseNratioSize * this->n2D2;
    n2Offset = extraInfo.n2oIdx * this->d2Size;

    this->vCoreOffset = bOffset + n2Offset + s2Offset;

    // 切K
    uint32_t kSplitSize = 192;
    uint32_t kloops = (extraInfo.s2RealSize + kSplitSize - 1) / kSplitSize;
    uint32_t kTail = extraInfo.s2RealSize - (kloops - 1) * kSplitSize;
    uint32_t subKSizeAct = kSplitSize;
    uint32_t subKSizeActAligned = CeilDiv(subKSizeAct, 16) * 16;

    // 切N
    uint32_t nSplitSize = 128;
    uint32_t nLoops = CeilDiv(this->d2Size, nSplitSize);
    uint32_t nTail = this->d2Size - (nLoops - 1) * nSplitSize;
    uint32_t subNSizeAct = nSplitSize;

    LocalTensor<T> oTensor = oL0CTensor;

    uint32_t tmpTNDkvL1BufIter = kvL1BufIter;
    uint32_t tmpTNDpL1BufIter = pL1BufIter;

    // nloop 按照n的循环控制每次从L0C放入不同的GM的地址
    for (auto n = 0; n < nLoops; ++n) {
        bool isLastN = n == nLoops - 1;
        if (isLastN) {
            subNSizeAct = nTail;
        }
        kvL1BufIter = tmpTNDkvL1BufIter;
        pL1BufIter = tmpTNDpL1BufIter;
        subKSizeAct = kSplitSize;
        subKSizeActAligned = CeilDiv(subKSizeAct, 16) * 16;
        for (uint32_t k = 0; k < kloops; k++) {
            if (k == kloops - 1) {
                subKSizeAct = kTail;
                subKSizeActAligned = CeilDiv(kTail, 16) * 16;
            }
            
            WaitFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + (pL1BufIter % 2));
            LocalTensor<INPUT_T> pTensor = pL1Tensor[(pL1BufIter % 2) * (L1_P_SIZE / sizeof(INPUT_T))];
            CopyInMm2AToL1(pTensor, extraInfo, k, subKSizeActAligned, kSplitSize); // s1 * kSplitSize
            
            SetFlag<HardEvent::MTE2_MTE1>(P_EVENT0 + (pL1BufIter % 2));
            WaitFlag<HardEvent::MTE2_MTE1>(P_EVENT0 + (pL1BufIter % 2));
            
            WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % 2));
            LocalTensor vTensor = kvL1Tensor[(kvL1BufIter % 2) * (L1_KV_SIZE / sizeof(INPUT_T))];
            CopyInMm2BToL1(vTensor, k, subKSizeAct, kSplitSize, n, isLastN); // kSplitSize * dSize
            
            SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % 2));
            WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % 2));

            bmm2.SetOrgShape(extraInfo.cubeS1RealSize, this->d2Size, subKSizeActAligned, subKSizeActAligned, this->d2Size);
            bmm2.SetTail(extraInfo.cubeS1RealSize, subNSizeAct, subKSizeAct);

            bmm2.SetTensorA(pTensor);
            bmm2.SetTensorB(vTensor);
            
            bmm2.template Iterate(k!=0, oTensor);
            
            SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % 2));
            SetFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + (pL1BufIter % 2));
            kvL1BufIter++;
            pL1BufIter++;
        }

        bmm2.template GetTensorC(this->mm2Res[extraInfo.taskIdMod2][extraInfo.cubeS1RealSize * nSplitSize * n]);
    }
    bmm2.End();
}

template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void 
FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, implMode, hasRope>::CopyInMm1AToL1(LocalTensor<INPUT_T> &l1Tensor,
                                                                                    SplitSameABExtraInfo &extraInfo)
{
    // 计算gm上的offset
    int64_t bOffset = 0;

    // s1需要考虑inner轴的影响
    int64_t s1Offset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    bOffset = extraInfo.s1SizeAcc * this->n2GD;
    s1Offset = extraInfo.s1oIdx * this->s1BaseN2GD;
    n2Offset = extraInfo.n2oIdx * this->gD;
    gOffset = extraInfo.goIdx * this->dSize;
    
    this->qCoreOffset = bOffset + n2Offset + gOffset + s1Offset;
    extraInfo.qCoreOffset = this->qCoreOffset;
    auto srcGm = this->queryGm[extraInfo.qCoreOffset];
    CopyGmToL1(l1Tensor, srcGm, extraInfo.cubeS1RealSize, this->dSize, this->n2GD);
}

template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void 
FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, implMode, hasRope>::CopyInMm1BToL1(LocalTensor<INPUT_T> &bL1Tensor,
                                                                                const SplitSameABExtraInfo &extraInfo, uint32_t subNid, uint32_t subNSize, uint32_t nSplitSize)
{
    auto srcGm = this->keyGm[this->kCoreOffset + subNid * nSplitSize * this->n2D];
    CopyGmToL1(bL1Tensor, srcGm, subNSize, this->dSize, this->n2D);
}

template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void 
FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, implMode, hasRope>::CopyInMm2AToL1(LocalTensor<INPUT_T> &al1Tensor,
                                                                                    SplitSameABExtraInfo &extraInfo, uint32_t subKid, uint32_t subKSize, uint32_t kSplitSize)
{
    int64_t cubeS1RealSizeAligned = CeilDiv(extraInfo.cubeS1RealSize, 16) * 16;
    auto srcGm = this->stage1Res[extraInfo.taskIdMod2][cubeS1RealSizeAligned*subKid*kSplitSize];
    DataCopy(al1Tensor, srcGm, cubeS1RealSizeAligned*subKSize);
}

template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void 
FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, implMode, hasRope>::CopyInMm2BToL1(LocalTensor<INPUT_T> &bL1Tensor,
                                                                                uint32_t subKid, uint32_t subKSize, uint32_t kSplitSize,
                                                                                uint32_t nLoopIdx, bool isLastN)
{
    uint32_t nSplitSize = 128;
    uint32_t nStartOffset = nLoopIdx * nSplitSize;
    auto srcGm = this->valueGm[this->vCoreOffset + subKid * kSplitSize * this->n2D2 + nStartOffset];
    if (isLastN) {
        nSplitSize = this->d2Size - nLoopIdx * nSplitSize;
    }
    CopyGmToL1(bL1Tensor, srcGm, subKSize, nSplitSize, this->n2D2);
}

template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void 
FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, implMode, hasRope>::CopyGmToL1(LocalTensor<INPUT_T> &l1Tensor, GlobalTensor<INPUT_T> &gmSrcTensor,
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

#endif // FLASH_ATTENTION_VAR_LEN_SCORE_SAB_H
