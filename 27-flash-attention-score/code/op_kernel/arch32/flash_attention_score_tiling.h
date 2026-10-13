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
 * \file flash_attention_score_tiling.h
 * \brief
 */

#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

class BaseParams {
public:
    // 基础参数
    uint32_t batchSize;
    uint32_t headNumSize;
    uint32_t seqSizeQ;
    uint32_t seqSizeK;
    uint32_t seqSizeV;
    uint32_t seqInnerSize;
    uint32_t headSize;
    float keepProb;
    float scaleValue;
    uint32_t preTockens;
    uint32_t nextTockens;
    uint32_t attenMaskSOuter;
    uint32_t seqBaseSize;
    uint32_t seqBaseRange;
    uint32_t seqBasePadValue;
    uint32_t seqSizeAlign;
    uint32_t transType;
    uint32_t pseShapeType;
    uint32_t seqInnerSizeAlign;
    uint32_t attenMaskTypeIdx;
    uint32_t attenBatchSize;
    uint32_t attenHeadNum;
    uint32_t isSD9k;
    uint32_t remain;

    // ========================
    // Getter & Setter 方法
    // ========================

    uint32_t get_batchSize() const { return batchSize; }
    void set_batchSize(uint32_t value) { this->batchSize = value; }

    uint32_t get_headNumSize() const { return headNumSize; }
    void set_headNumSize(uint32_t value) { this->headNumSize = value; }

    uint32_t get_seqSizeQ() const { return seqSizeQ; }
    void set_seqSizeQ(uint32_t value) { this->seqSizeQ = value; }

    uint32_t get_seqSizeK() const { return seqSizeK; }
    void set_seqSizeK(uint32_t value) { this->seqSizeK = value; }

    uint32_t get_seqSizeV() const { return seqSizeV; }
    void set_seqSizeV(uint32_t value) { this->seqSizeV = value; }

    uint32_t get_seqInnerSize() const { return seqInnerSize; }
    void set_seqInnerSize(uint32_t value) { this->seqInnerSize = value; }

    uint32_t get_headSize() const { return headSize; }
    void set_headSize(uint32_t value) { this->headSize = value; }

    float get_keepProb() const { return keepProb; }
    void set_keepProb(float value) { this->keepProb = value; }

    float get_scaleValue() const { return scaleValue; }
    void set_scaleValue(float value) { this->scaleValue = value; }

    uint32_t get_preTockens() const { return preTockens; }
    void set_preTockens(uint32_t value) { this->preTockens = value; }

    uint32_t get_nextTockens() const { return nextTockens; }
    void set_nextTockens(uint32_t value) { this->nextTockens = value; }

    uint32_t get_attenMaskSOuter() const { return attenMaskSOuter; }
    void set_attenMaskSOuter(uint32_t value) { this->attenMaskSOuter = value; }

    uint32_t get_seqBaseSize() const { return seqBaseSize; }
    void set_seqBaseSize(uint32_t value) { this->seqBaseSize = value; }

    uint32_t get_seqBaseRange() const { return seqBaseRange; }
    void set_seqBaseRange(uint32_t value) { this->seqBaseRange = value; }

    uint32_t get_seqBasePadValue() const { return seqBasePadValue; }
    void set_seqBasePadValue(uint32_t value) { this->seqBasePadValue = value; }

    uint32_t get_seqSizeAlign() const { return seqSizeAlign; }
    void set_seqSizeAlign(uint32_t value) { this->seqSizeAlign = value; }

    uint32_t get_transType() const { return transType; }
    void set_transType(uint32_t value) { this->transType = value; }

    uint32_t get_pseShapeType() const { return pseShapeType; }
    void set_pseShapeType(uint32_t value) { this->pseShapeType = value; }

    uint32_t get_seqInnerSizeAlign() const { return seqInnerSizeAlign; }
    void set_seqInnerSizeAlign(uint32_t value) { this->seqInnerSizeAlign = value; }

    uint32_t get_attenMaskTypeIdx() const { return attenMaskTypeIdx; }
    void set_attenMaskTypeIdx(uint32_t value) { this->attenMaskTypeIdx = value; }

    uint32_t get_attenBatchSize() const { return attenBatchSize; }
    void set_attenBatchSize(uint32_t value) { this->attenBatchSize = value; }

    uint32_t get_attenHeadNum() const { return attenHeadNum; }
    void set_attenHeadNum(uint32_t value) { this->attenHeadNum = value; }

    uint32_t get_isSD9k() const { return isSD9k; }
    void set_isSD9k(uint32_t value) { this->isSD9k = value; }

    uint32_t get_remain() const { return remain; }
    void set_remain(uint32_t value) { this->remain = value; }

    void reset()
    {
        set_batchSize(0);
        set_headNumSize(0);
        set_seqSizeQ(0);
        set_seqSizeK(0);
        set_seqSizeV(0);
        set_seqInnerSize(0);
        set_headSize(0);
        set_keepProb(0);
        set_scaleValue(0);
        set_preTockens(0);
        set_nextTockens(0);
        set_attenMaskSOuter(0);
        set_seqBaseSize(0);
        set_seqBaseRange(0);
        set_seqBasePadValue(0);
        set_seqSizeAlign(0);
        set_transType(0);
        set_pseShapeType(0);
        set_seqInnerSizeAlign(0);
        set_attenMaskTypeIdx(0);
        set_attenBatchSize(0);
        set_attenHeadNum(0);
        set_isSD9k(0);
    }

};

class AttentionScoreCoreParams {
public:
    // 基础参数
    uint32_t coreNum;
    uint32_t loopBatchNum;
    uint32_t loopHeadNum;
    uint32_t loopBatchNumTail;
    uint32_t loopHeadNumTail;
    uint32_t b1;
    uint32_t n1;
    uint32_t s1;
    uint32_t singleCoreBatchSize;
    uint32_t singleCoreBatchSizeTail;
    uint32_t singleCoreSeqSize;
    uint32_t singleCoreHeadNumSize;
    uint32_t singleCoreHeadNumSizeTail;
    uint32_t singleCoreSeqSizeTail;
    uint32_t singleCoreHeadSize;

    // 数组字段（固定长度 48）
    uint32_t seqList[48];
    uint32_t eachCoreSeqSize[48];

    // 其他参数
    uint32_t formerNum;
    uint32_t tailNum;
    uint32_t formerHeadNum;
    uint32_t tailHeadNum;
    uint32_t singleCoreDataSize;
    uint32_t bnRange;
    uint32_t bnRangeTail;

    // ========================
    // Getter & Setter 方法
    // ========================

    uint32_t get_coreNum() const { return coreNum; }
    void set_coreNum(uint32_t value) { this->coreNum = value; }

    uint32_t get_loopBatchNum() const { return loopBatchNum; }
    void set_loopBatchNum(uint32_t value) { this->loopBatchNum = value; }

    uint32_t get_loopHeadNum() const { return loopHeadNum; }
    void set_loopHeadNum(uint32_t value) { this->loopHeadNum = value; }

    uint32_t get_loopBatchNumTail() const { return loopBatchNumTail; }
    void set_loopBatchNumTail(uint32_t value) { this->loopBatchNumTail = value; }

    uint32_t get_loopHeadNumTail() const { return loopHeadNumTail; }
    void set_loopHeadNumTail(uint32_t value) { this->loopHeadNumTail = value; }

    uint32_t get_b1() const { return b1; }
    void set_b1(uint32_t value) { this->b1 = value; }

    uint32_t get_n1() const { return n1; }
    void set_n1(uint32_t value) { this->n1 = value; }

    uint32_t get_s1() const { return s1; }
    void set_s1(uint32_t value) { this->s1 = value; }

    uint32_t get_singleCoreBatchSize() const { return singleCoreBatchSize; }
    void set_singleCoreBatchSize(uint32_t value) { this->singleCoreBatchSize = value; }

    uint32_t get_singleCoreBatchSizeTail() const { return singleCoreBatchSizeTail; }
    void set_singleCoreBatchSizeTail(uint32_t value) { this->singleCoreBatchSizeTail = value; }

    uint32_t get_singleCoreSeqSize() const { return singleCoreSeqSize; }
    void set_singleCoreSeqSize(uint32_t value) { this->singleCoreSeqSize = value; }

    uint32_t get_singleCoreHeadNumSize() const { return singleCoreHeadNumSize; }
    void set_singleCoreHeadNumSize(uint32_t value) { this->singleCoreHeadNumSize = value; }

    uint32_t get_singleCoreHeadNumSizeTail() const { return singleCoreHeadNumSizeTail; }
    void set_singleCoreHeadNumSizeTail(uint32_t value) { this->singleCoreHeadNumSizeTail = value; }

    uint32_t get_singleCoreSeqSizeTail() const { return singleCoreSeqSizeTail; }
    void set_singleCoreSeqSizeTail(uint32_t value) { this->singleCoreSeqSizeTail = value; }

    uint32_t get_singleCoreHeadSize() const { return singleCoreHeadSize; }
    void set_singleCoreHeadSize(uint32_t value) { this->singleCoreHeadSize = value; }

    const uint32_t* get_seqList() const { return seqList; }
    void set_seqList(const uint32_t* values) {
        for (int i = 0; i < 48; ++i) {
            seqList[i] = values[i];
        }
    }

    const uint32_t* get_eachCoreSeqSize() const { return eachCoreSeqSize; }
    void set_eachCoreSeqSize(const uint32_t* values) {
        for (int i = 0; i < 48; ++i) {
            eachCoreSeqSize[i] = values[i];
        }
    }

    uint32_t get_formerNum() const { return formerNum; }
    void set_formerNum(uint32_t value) { this->formerNum = value; }

    uint32_t get_tailNum() const { return tailNum; }
    void set_tailNum(uint32_t value) { this->tailNum = value; }

    uint32_t get_formerHeadNum() const { return formerHeadNum; }
    void set_formerHeadNum(uint32_t value) { this->formerHeadNum = value; }

    uint32_t get_tailHeadNum() const { return tailHeadNum; }
    void set_tailHeadNum(uint32_t value) { this->tailHeadNum = value; }

    uint32_t get_singleCoreDataSize() const { return singleCoreDataSize; }
    void set_singleCoreDataSize(uint32_t value) { this->singleCoreDataSize = value; }

    uint32_t get_bnRange() const { return bnRange; }
    void set_bnRange(uint32_t value) { this->bnRange = value; }

    uint32_t get_bnRangeTail() const { return bnRangeTail; }
    void set_bnRangeTail(uint32_t value) { this->bnRangeTail = value; }

    void reset()
    {
        set_coreNum(0);
        set_loopBatchNum(0);
        set_loopHeadNum(0);
        set_loopBatchNumTail(0);
        set_loopHeadNumTail(0);
        set_b1(0);
        set_n1(0);
        set_s1(0);
        set_singleCoreBatchSize(0);
        set_singleCoreBatchSizeTail(0);
        set_singleCoreSeqSize(0);
        set_singleCoreHeadNumSize(0);
        set_singleCoreHeadNumSizeTail(0);
        set_singleCoreSeqSizeTail(0);
        set_singleCoreHeadSize(0);
        for (int i = 0; i < 48; ++i) {
            seqList[i] = 0;
        }
        for (int i = 0; i < 48; ++i) {
            eachCoreSeqSize[i] = 0;
        }
        set_formerNum(0);
        set_tailNum(0);
        set_formerHeadNum(0);
        set_tailHeadNum(0);
        set_singleCoreDataSize(0);
        set_bnRange(0);
        set_bnRangeTail(0);
    }
};

class AttentionScoreSingleCoreParams {
public:
    // 基础参数
    uint32_t eachCoreBatchNum;
    uint32_t sOuterLoopTimes;
    uint32_t sInnerLoopTimes;
    uint32_t headRange;
    uint32_t singleProcessBatchSize;
    uint32_t singleProcessHeadNumSize;
    uint32_t singleProcessSOuterSize;
    uint32_t singleProcessSOuterSizeTail;
    uint32_t singleProcessSInnerSize;
    uint32_t singleProcessSInnerSizeTail;
    uint32_t singleProcessHeadSize;

    // 数组字段（固定长度 48）
    uint32_t sOuterLoopTimesList[48];

    // 尾部核心相关参数
    uint32_t tailCoreSingleProcessSOuterSize;
    uint32_t tailCoreSingleProcessSOuterSizeTail;
    uint32_t lastCoreSingleProcessSOuterSize;
    uint32_t lastCoreSingleProcessSOuterSizeTail;
    uint32_t tailCoreSOuterLoopTimes;
    uint32_t lastCoreSOuterLoopTimes;
    uint32_t tailCoreSingleProcessSInnerSize;
    uint32_t tailCoreSInnerLoopTimes;
    uint32_t tailCoreSingleProcessSInnerSizeTail;

    // ========================
    // Getter & Setter 方法
    // ========================

    uint32_t get_eachCoreBatchNum() const { return eachCoreBatchNum; }
    void set_eachCoreBatchNum(uint32_t value) { this->eachCoreBatchNum = value; }

    const uint32_t* get_sOuterLoopTimesList() const { return sOuterLoopTimesList; }
    void set_sOuterLoopTimesList(const uint32_t* values) {
        for (int i = 0; i < 48; ++i) {
            sOuterLoopTimesList[i] = values[i];
        }
    }

    uint32_t get_sOuterLoopTimes() const { return sOuterLoopTimes; }
    void set_sOuterLoopTimes(uint32_t value) { this->sOuterLoopTimes = value; }

    uint32_t get_sInnerLoopTimes() const { return sInnerLoopTimes; }
    void set_sInnerLoopTimes(uint32_t value) { this->sInnerLoopTimes = value; }

    uint32_t get_headRange() const { return headRange; }
    void set_headRange(uint32_t value) { this->headRange = value; }

    uint32_t get_singleProcessBatchSize() const { return singleProcessBatchSize; }
    void set_singleProcessBatchSize(uint32_t value) { this->singleProcessBatchSize = value; }

    uint32_t get_singleProcessHeadNumSize() const { return singleProcessHeadNumSize; }
    void set_singleProcessHeadNumSize(uint32_t value) { this->singleProcessHeadNumSize = value; }

    uint32_t get_singleProcessSOuterSize() const { return singleProcessSOuterSize; }
    void set_singleProcessSOuterSize(uint32_t value) { this->singleProcessSOuterSize = value; }

    uint32_t get_singleProcessSOuterSizeTail() const { return singleProcessSOuterSizeTail; }
    void set_singleProcessSOuterSizeTail(uint32_t value) { this->singleProcessSOuterSizeTail = value; }

    uint32_t get_singleProcessSInnerSize() const { return singleProcessSInnerSize; }
    void set_singleProcessSInnerSize(uint32_t value) { this->singleProcessSInnerSize = value; }

    uint32_t get_singleProcessSInnerSizeTail() const { return singleProcessSInnerSizeTail; }
    void set_singleProcessSInnerSizeTail(uint32_t value) { this->singleProcessSInnerSizeTail = value; }

    uint32_t get_singleProcessHeadSize() const { return singleProcessHeadSize; }
    void set_singleProcessHeadSize(uint32_t value) { this->singleProcessHeadSize = value; }

    uint32_t get_tailCoreSingleProcessSOuterSize() const { return tailCoreSingleProcessSOuterSize; }
    void set_tailCoreSingleProcessSOuterSize(uint32_t value) {
        this->tailCoreSingleProcessSOuterSize = value;
    }

    uint32_t get_tailCoreSingleProcessSOuterSizeTail() const { return tailCoreSingleProcessSOuterSizeTail; }
    void set_tailCoreSingleProcessSOuterSizeTail(uint32_t value) {
        this->tailCoreSingleProcessSOuterSizeTail = value;
    }

    uint32_t get_lastCoreSingleProcessSOuterSize() const { return lastCoreSingleProcessSOuterSize; }
    void set_lastCoreSingleProcessSOuterSize(uint32_t value) {
        this->lastCoreSingleProcessSOuterSize = value;
    }

    uint32_t get_lastCoreSingleProcessSOuterSizeTail() const { return lastCoreSingleProcessSOuterSizeTail; }
    void set_lastCoreSingleProcessSOuterSizeTail(uint32_t value) {
        this->lastCoreSingleProcessSOuterSizeTail = value;
    }

    uint32_t get_tailCoreSOuterLoopTimes() const { return tailCoreSOuterLoopTimes; }
    void set_tailCoreSOuterLoopTimes(uint32_t value) {
        this->tailCoreSOuterLoopTimes = value;
    }

    uint32_t get_lastCoreSOuterLoopTimes() const { return lastCoreSOuterLoopTimes; }
    void set_lastCoreSOuterLoopTimes(uint32_t value) {
        this->lastCoreSOuterLoopTimes = value;
    }

    uint32_t get_tailCoreSingleProcessSInnerSize() const { return tailCoreSingleProcessSInnerSize; }
    void set_tailCoreSingleProcessSInnerSize(uint32_t value) {
        this->tailCoreSingleProcessSInnerSize = value;
    }

    uint32_t get_tailCoreSInnerLoopTimes() const { return tailCoreSInnerLoopTimes; }
    void set_tailCoreSInnerLoopTimes(uint32_t value) {
        this->tailCoreSInnerLoopTimes = value;
    }

    uint32_t get_tailCoreSingleProcessSInnerSizeTail() const { return tailCoreSingleProcessSInnerSizeTail; }
    void set_tailCoreSingleProcessSInnerSizeTail(uint32_t value) {
        this->tailCoreSingleProcessSInnerSizeTail = value;
    }

    void reset()
    {
        set_eachCoreBatchNum(0);
        for (int i = 0; i < 48; ++i) {
            sOuterLoopTimesList[i] = 0;
        }
        set_sOuterLoopTimes(0);
        set_sInnerLoopTimes(0);
        set_headRange(0);
        set_singleProcessBatchSize(0);
        set_singleProcessHeadNumSize(0);
        set_singleProcessSOuterSize(0);
        set_singleProcessSOuterSizeTail(0);
        set_singleProcessSInnerSize(0);
        set_singleProcessSInnerSizeTail(0);
        set_singleProcessHeadSize(0);
        set_tailCoreSingleProcessSOuterSize(0);
        set_tailCoreSingleProcessSOuterSizeTail(0);
        set_lastCoreSingleProcessSOuterSize(0);
        set_lastCoreSingleProcessSOuterSizeTail(0);
        set_tailCoreSOuterLoopTimes(0);
        set_lastCoreSOuterLoopTimes(0);
        set_tailCoreSingleProcessSInnerSize(0);
        set_tailCoreSInnerLoopTimes(0);
        set_tailCoreSingleProcessSInnerSizeTail(0);
    }
};

class AttentionScoreSingleCoreTensorSize {
public:
    // ======== 核心内存大小参数 ========
    uint32_t mmResUbSize;
    uint32_t paddingMaskUbSize;
    uint32_t attenMaskUbSize;
    uint32_t pseUbSize;
    uint32_t maskSize;
    uint32_t softmaxMaxSize;
    uint32_t softmaxSumSize;
    uint32_t softmaxExpSize;
    uint32_t spmTmpSize;
    uint32_t scmTmpSize;
    uint32_t apiTmpUbSize;
    uint32_t apiTmpUbSizeTail;
    uint32_t mmResInUbSize;
    uint32_t mmResInUbSizeTail;
    uint32_t bmm2ResUbSize;
    uint32_t tmpMMResBmm2PreUbSize;
    uint32_t tmpSoftmaxBmm2UbSize;
    uint32_t mmResUbSizeTailLoop;
    uint32_t attenMaskUbSizeTailLoop;
    uint32_t maskSizeTailLoop;

    // ======== 尾核（tail core）专用参数 ========
    uint32_t tailCoreMMResUbSize;
    uint32_t tailCorePaddingMaskUbSize;
    uint32_t tailCoreAttenMaskUbSize;
    uint32_t tailCorePseUbSize;
    uint32_t tailCoreMaskSize;
    uint32_t tailCoreSoftmaxMaxSize;
    uint32_t tailCoreSoftmaxSumSize;
    uint32_t tailCoreSoftmaxExpSize;
    uint32_t tailCoreSpmTmpSize;
    uint32_t tailCoreScmTmpSize;
    uint32_t tailCoreBmm2ResUbSize;
    uint32_t tailCoreTmpMMResBmm2PreUbSize;
    uint32_t tailCoreTmpSoftmaxBmm2UbSize;
    uint32_t tailCoreMMResUbSizeTailLoop;
    uint32_t tailCoreAttenMaskUbSizeTailLoop;
    uint32_t tailCoreMaskSizeTailLoop;

    // ======== 临时缓冲区 ========
    uint32_t softmaxTmpBuffer;
    uint32_t dropOutTmpBuffer;

    // ========================
    // Getter & Setter 方法
    // ========================

    uint32_t get_mmResUbSize() const { return mmResUbSize; }
    void set_mmResUbSize(uint32_t value) { this->mmResUbSize = value; }

    uint32_t get_paddingMaskUbSize() const { return paddingMaskUbSize; }
    void set_paddingMaskUbSize(uint32_t value) { this->paddingMaskUbSize = value; }

    uint32_t get_attenMaskUbSize() const { return attenMaskUbSize; }
    void set_attenMaskUbSize(uint32_t value) { this->attenMaskUbSize = value; }

    uint32_t get_pseUbSize() const { return pseUbSize; }
    void set_pseUbSize(uint32_t value) { this->pseUbSize = value; }

    uint32_t get_maskSize() const { return maskSize; }
    void set_maskSize(uint32_t value) { this->maskSize = value; }

    uint32_t get_softmaxMaxSize() const { return softmaxMaxSize; }
    void set_softmaxMaxSize(uint32_t value) { this->softmaxMaxSize = value; }

    uint32_t get_softmaxSumSize() const { return softmaxSumSize; }
    void set_softmaxSumSize(uint32_t value) { this->softmaxSumSize = value; }

    uint32_t get_softmaxExpSize() const { return softmaxExpSize; }
    void set_softmaxExpSize(uint32_t value) { this->softmaxExpSize = value; }

    uint32_t get_spmTmpSize() const { return spmTmpSize; }
    void set_spmTmpSize(uint32_t value) { this->spmTmpSize = value; }

    uint32_t get_scmTmpSize() const { return scmTmpSize; }
    void set_scmTmpSize(uint32_t value) { this->scmTmpSize = value; }

    uint32_t get_apiTmpUbSize() const { return apiTmpUbSize; }
    void set_apiTmpUbSize(uint32_t value) { this->apiTmpUbSize = value; }

    uint32_t get_apiTmpUbSizeTail() const { return apiTmpUbSizeTail; }
    void set_apiTmpUbSizeTail(uint32_t value) { this->apiTmpUbSizeTail = value; }

    uint32_t get_mmResInUbSize() const { return mmResInUbSize; }
    void set_mmResInUbSize(uint32_t value) { this->mmResInUbSize = value; }

    uint32_t get_mmResInUbSizeTail() const { return mmResInUbSizeTail; }
    void set_mmResInUbSizeTail(uint32_t value) { this->mmResInUbSizeTail = value; }

    uint32_t get_bmm2ResUbSize() const { return bmm2ResUbSize; }
    void set_bmm2ResUbSize(uint32_t value) { this->bmm2ResUbSize = value; }

    uint32_t get_tmpMMResBmm2PreUbSize() const { return tmpMMResBmm2PreUbSize; }
    void set_tmpMMResBmm2PreUbSize(uint32_t value) { this->tmpMMResBmm2PreUbSize = value; }

    uint32_t get_tmpSoftmaxBmm2UbSize() const { return tmpSoftmaxBmm2UbSize; }
    void set_tmpSoftmaxBmm2UbSize(uint32_t value) { this->tmpSoftmaxBmm2UbSize = value; }

    uint32_t get_mmResUbSizeTailLoop() const { return mmResUbSizeTailLoop; }
    void set_mmResUbSizeTailLoop(uint32_t value) { this->mmResUbSizeTailLoop = value; }

    uint32_t get_attenMaskUbSizeTailLoop() const { return attenMaskUbSizeTailLoop; }
    void set_attenMaskUbSizeTailLoop(uint32_t value) { this->attenMaskUbSizeTailLoop = value; }

    uint32_t get_maskSizeTailLoop() const { return maskSizeTailLoop; }
    void set_maskSizeTailLoop(uint32_t value) { this->maskSizeTailLoop = value; }

    // --- Tail Core Fields ---
    uint32_t get_tailCoreMMResUbSize() const { return tailCoreMMResUbSize; }
    void set_tailCoreMMResUbSize(uint32_t value) { this->tailCoreMMResUbSize = value; }

    uint32_t get_tailCorePaddingMaskUbSize() const { return tailCorePaddingMaskUbSize; }
    void set_tailCorePaddingMaskUbSize(uint32_t value) { this->tailCorePaddingMaskUbSize = value; }

    uint32_t get_tailCoreAttenMaskUbSize() const { return tailCoreAttenMaskUbSize; }
    void set_tailCoreAttenMaskUbSize(uint32_t value) { this->tailCoreAttenMaskUbSize = value; }

    uint32_t get_tailCorePseUbSize() const { return tailCorePseUbSize; }
    void set_tailCorePseUbSize(uint32_t value) { this->tailCorePseUbSize = value; }

    uint32_t get_tailCoreMaskSize() const { return tailCoreMaskSize; }
    void set_tailCoreMaskSize(uint32_t value) { this->tailCoreMaskSize = value; }

    uint32_t get_tailCoreSoftmaxMaxSize() const { return tailCoreSoftmaxMaxSize; }
    void set_tailCoreSoftmaxMaxSize(uint32_t value) { this->tailCoreSoftmaxMaxSize = value; }

    uint32_t get_tailCoreSoftmaxSumSize() const { return tailCoreSoftmaxSumSize; }
    void set_tailCoreSoftmaxSumSize(uint32_t value) { this->tailCoreSoftmaxSumSize = value; }

    uint32_t get_tailCoreSoftmaxExpSize() const { return tailCoreSoftmaxExpSize; }
    void set_tailCoreSoftmaxExpSize(uint32_t value) { this->tailCoreSoftmaxExpSize = value; }

    uint32_t get_tailCoreSpmTmpSize() const { return tailCoreSpmTmpSize; }
    void set_tailCoreSpmTmpSize(uint32_t value) { this->tailCoreSpmTmpSize = value; }

    uint32_t get_tailCoreScmTmpSize() const { return tailCoreScmTmpSize; }
    void set_tailCoreScmTmpSize(uint32_t value) { this->tailCoreScmTmpSize = value; }

    uint32_t get_tailCoreBmm2ResUbSize() const { return tailCoreBmm2ResUbSize; }
    void set_tailCoreBmm2ResUbSize(uint32_t value) { this->tailCoreBmm2ResUbSize = value; }

    uint32_t get_tailCoreTmpMMResBmm2PreUbSize() const { return tailCoreTmpMMResBmm2PreUbSize; }
    void set_tailCoreTmpMMResBmm2PreUbSize(uint32_t value) { this->tailCoreTmpMMResBmm2PreUbSize = value; }

    uint32_t get_tailCoreTmpSoftmaxBmm2UbSize() const { return tailCoreTmpSoftmaxBmm2UbSize; }
    void set_tailCoreTmpSoftmaxBmm2UbSize(uint32_t value) { this->tailCoreTmpSoftmaxBmm2UbSize = value; }

    uint32_t get_tailCoreMMResUbSizeTailLoop() const { return tailCoreMMResUbSizeTailLoop; }
    void set_tailCoreMMResUbSizeTailLoop(uint32_t value) { this->tailCoreMMResUbSizeTailLoop = value; }

    uint32_t get_tailCoreAttenMaskUbSizeTailLoop() const { return tailCoreAttenMaskUbSizeTailLoop; }
    void set_tailCoreAttenMaskUbSizeTailLoop(uint32_t value) { this->tailCoreAttenMaskUbSizeTailLoop = value; }

    uint32_t get_tailCoreMaskSizeTailLoop() const { return tailCoreMaskSizeTailLoop; }
    void set_tailCoreMaskSizeTailLoop(uint32_t value) { this->tailCoreMaskSizeTailLoop = value; }

    // --- Temporary Buffers ---
    uint32_t get_softmaxTmpBuffer() const { return softmaxTmpBuffer; }
    void set_softmaxTmpBuffer(uint32_t value) { this->softmaxTmpBuffer = value; }

    uint32_t get_dropOutTmpBuffer() const { return dropOutTmpBuffer; }
    void set_dropOutTmpBuffer(uint32_t value) { this->dropOutTmpBuffer = value; }

    void reset()
    {
        set_mmResUbSize(0);
        set_paddingMaskUbSize(0);
        set_attenMaskUbSize(0);
        set_pseUbSize(0);
        set_maskSize(0);
        set_softmaxMaxSize(0);
        set_softmaxSumSize(0);
        set_softmaxExpSize(0);
        set_spmTmpSize(0);
        set_scmTmpSize(0);
        set_apiTmpUbSize(0);
        set_apiTmpUbSizeTail(0);
        set_mmResInUbSize(0);
        set_mmResInUbSizeTail(0);
        set_bmm2ResUbSize(0);
        set_tmpMMResBmm2PreUbSize(0);
        set_tmpSoftmaxBmm2UbSize(0);
        set_mmResUbSizeTailLoop(0);
        set_attenMaskUbSizeTailLoop(0);
        set_maskSizeTailLoop(0);
        set_tailCoreMMResUbSize(0);
        set_tailCorePaddingMaskUbSize(0);
        set_tailCoreAttenMaskUbSize(0);
        set_tailCorePseUbSize(0);
        set_tailCoreMaskSize(0);
        set_tailCoreSoftmaxMaxSize(0);
        set_tailCoreSoftmaxSumSize(0);
        set_tailCoreSoftmaxExpSize(0);
        set_tailCoreSpmTmpSize(0);
        set_tailCoreScmTmpSize(0);
        set_tailCoreBmm2ResUbSize(0);
        set_tailCoreTmpMMResBmm2PreUbSize(0);
        set_tailCoreTmpSoftmaxBmm2UbSize(0);
        set_tailCoreMMResUbSizeTailLoop(0);
        set_tailCoreAttenMaskUbSizeTailLoop(0);
        set_tailCoreMaskSizeTailLoop(0);
        set_softmaxTmpBuffer(0);
        set_dropOutTmpBuffer(0);
    }
};

class AttentionScoreOffestStrideParams {
public:
    // === 核心参数 ===
    uint32_t typeByteNum;
    uint32_t matmulHead;

    // ========================
    // Getter & Setter 方法
    // ========================

    uint32_t get_typeByteNum() const { return typeByteNum; }
    void set_typeByteNum(uint32_t value) { this->typeByteNum = value; }

    uint32_t get_matmulHead() const { return matmulHead; }
    void set_matmulHead(uint32_t value) { this->matmulHead = value; }

    void reset()
    {
        set_typeByteNum(0);
        set_matmulHead(0);
    }
};

class FlashAttentionScoreEmptyInputTilingData {
public:
    // === 核心参数 ===
    uint32_t coreNum;
    uint32_t attentionOutFormerNum;
    uint32_t attentionOutTailNum;
    uint32_t softmaxMaxFormerNum;
    uint32_t softmaxMaxTailNum;
    uint32_t reserved;

    // === 内存大小（单位：byte）===
    uint64_t attentionOutSingleCoreDataSize;
    uint64_t attentionOutTailCoreDataSize;
    uint64_t softmaxMaxSingleCoreDataSize;
    uint64_t softmaxMaxTailCoreDataSize;
    uint64_t attentionOutLastCoreDataSize;
    uint64_t attentionOutLastCoreIndex;

    // ========================
    // Getter & Setter 方法
    // ========================

    uint32_t get_coreNum() const { return coreNum; }
    void set_coreNum(uint32_t value) { this->coreNum = value; }

    uint32_t get_attentionOutFormerNum() const { return attentionOutFormerNum; }
    void set_attentionOutFormerNum(uint32_t value) { this->attentionOutFormerNum = value; }

    uint32_t get_attentionOutTailNum() const { return attentionOutTailNum; }
    void set_attentionOutTailNum(uint32_t value) { this->attentionOutTailNum = value; }

    uint32_t get_softmaxMaxFormerNum() const { return softmaxMaxFormerNum; }
    void set_softmaxMaxFormerNum(uint32_t value) { this->softmaxMaxFormerNum = value; }

    uint32_t get_softmaxMaxTailNum() const { return softmaxMaxTailNum; }
    void set_softmaxMaxTailNum(uint32_t value) { this->softmaxMaxTailNum = value; }

    uint32_t get_reserved() const { return reserved; }
    void set_reserved(uint32_t value) { this->reserved = value; }

    uint64_t get_attentionOutSingleCoreDataSize() const { return attentionOutSingleCoreDataSize; }
    void set_attentionOutSingleCoreDataSize(uint64_t value) { this->attentionOutSingleCoreDataSize = value; }

    uint64_t get_attentionOutTailCoreDataSize() const { return attentionOutTailCoreDataSize; }
    void set_attentionOutTailCoreDataSize(uint64_t value) { this->attentionOutTailCoreDataSize = value; }

    uint64_t get_softmaxMaxSingleCoreDataSize() const { return softmaxMaxSingleCoreDataSize; }
    void set_softmaxMaxSingleCoreDataSize(uint64_t value) { this->softmaxMaxSingleCoreDataSize = value; }

    uint64_t get_softmaxMaxTailCoreDataSize() const { return softmaxMaxTailCoreDataSize; }
    void set_softmaxMaxTailCoreDataSize(uint64_t value) { this->softmaxMaxTailCoreDataSize = value; }

    uint64_t get_attentionOutLastCoreDataSize() const { return attentionOutLastCoreDataSize; }
    void set_attentionOutLastCoreDataSize(uint64_t value) { this->attentionOutLastCoreDataSize = value; }

    uint64_t get_attentionOutLastCoreIndex() const { return attentionOutLastCoreIndex; }
    void set_attentionOutLastCoreIndex(uint64_t value) { this->attentionOutLastCoreIndex = value; }

    void reset()
    {
        set_coreNum(0);
        set_attentionOutFormerNum(0);
        set_attentionOutTailNum(0);
        set_softmaxMaxFormerNum(0);
        set_softmaxMaxTailNum(0);
        set_reserved(0);
        set_attentionOutSingleCoreDataSize(0);
        set_attentionOutTailCoreDataSize(0);
        set_softmaxMaxSingleCoreDataSize(0);
        set_softmaxMaxTailCoreDataSize(0);
        set_attentionOutLastCoreDataSize(0);
        set_attentionOutLastCoreIndex(0);
    }
};

class InputParams {
public:
    // 基础参数
    int64_t bSize = 0;
    int64_t n2Size = 0;
    int64_t gSize = 0;
    int64_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t alignedS2 = 0;
    int64_t dSize = 0;
    float keepProb = 0;
    float scaleValue = 0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    int64_t pseS1Size = 0;
    int64_t pseS2Size = 0;
    uint32_t pseBSize = 0;
    uint32_t bandIndex = 0; // 13
    uint8_t layoutType = 0;
    uint8_t pseShapeType = 0;
    uint8_t attenMaskShapeType = 0;
    uint8_t attenMaskDataType = 0;
    uint8_t attenMaskCompressMode = 0;
    uint8_t implMode = 0;
    uint8_t sparseType = 0;
    uint8_t needDropMaskOp = 0;
    uint8_t pseEncodeType = 0;
    uint8_t rsv = 0;
    uint8_t needL1Carry = 0;
    uint8_t remainPH[1] = {0}; 
    uint16_t remain = 0;
    uint8_t attenMaskS2SizePH[2] = {0}; // 15
    uint32_t attenMaskS2Size = 0;
    uint32_t pseType = 0; // 16
    uint32_t rsv1 = 0;
    uint8_t qStartIdxPH[4] = {0};
    int64_t qStartIdx = 0;
    int64_t kvStartIdx = 0;
    uint8_t tndSoftmaxOut = 0;
    uint8_t needSinkOp = 0;
    uint8_t InputParamsPH[6] = {0};

    // ========================
    // Getter & Setter 方法
    // ========================

    int64_t get_bSize() const { return bSize; }
    void set_bSize(int64_t value) { this->bSize = value; }

    int64_t get_n2Size() const { return n2Size; }
    void set_n2Size(int64_t value) { this->n2Size = value; }

    int64_t get_gSize() const { return gSize; }
    void set_gSize(int64_t value) { this->gSize = value; }

    int64_t get_s1Size() const { return s1Size; }
    void set_s1Size(int64_t value) { this->s1Size = value; }

    int64_t get_s2Size() const { return s2Size; }
    void set_s2Size(int64_t value) { this->s2Size = value; }

    int64_t get_alignedS2() const { return alignedS2; }
    void set_alignedS2(int64_t value) { this->alignedS2 = value; }

    int64_t get_dSize() const { return dSize; }
    void set_dSize(int64_t value) { this->dSize = value; }

    float get_keepProb() const { return keepProb; }
    void set_keepProb(float value) { this->keepProb = value; }

    float get_scaleValue() const { return scaleValue; }
    void set_scaleValue(float value) { this->scaleValue = value; }

    int64_t get_preTokens() const { return preTokens; }
    void set_preTokens(int64_t value) { this->preTokens = value; }

    int64_t get_nextTokens() const { return nextTokens; }
    void set_nextTokens(int64_t value) { this->nextTokens = value; }

    int64_t get_pseS1Size() const { return pseS1Size; }
    void set_pseS1Size(int64_t value) { this->pseS1Size = value; }

    int64_t get_pseS2Size() const { return pseS2Size; }
    void set_pseS2Size(int64_t value) { this->pseS2Size = value; }

    uint32_t get_pseBSize() const { return pseBSize; }
    void set_pseBSize(uint32_t value) { this->pseBSize = value; }

    uint32_t get_bandIndex() const { return bandIndex; }
    void set_bandIndex(uint32_t value) { this->bandIndex = value; }

    uint8_t get_layoutType() const { return layoutType; }
    void set_layoutType(uint8_t value) { this->layoutType = value; }

    uint8_t get_pseShapeType() const { return pseShapeType; }
    void set_pseShapeType(uint8_t value) { this->pseShapeType = value; }

    uint8_t get_attenMaskShapeType() const { return attenMaskShapeType; }
    void set_attenMaskShapeType(uint8_t value) { this->attenMaskShapeType = value; }

    uint8_t get_attenMaskDataType() const { return attenMaskDataType; }
    void set_attenMaskDataType(uint8_t value) { this->attenMaskDataType = value; }

    uint8_t get_attenMaskCompressMode() const { return attenMaskCompressMode; }
    void set_attenMaskCompressMode(uint8_t value) { this->attenMaskCompressMode = value; }

    uint8_t get_implMode() const { return implMode; }
    void set_implMode(uint8_t value) { this->implMode = value; }

    uint8_t get_sparseType() const { return sparseType; }
    void set_sparseType(uint8_t value) { this->sparseType = value; }

    uint8_t get_needDropMaskOp() const { return needDropMaskOp; }
    void set_needDropMaskOp(uint8_t value) { this->needDropMaskOp = value; }

    uint8_t get_needSinkOp() const { return needSinkOp; }
    void set_needSinkOp(uint8_t value) { this->needSinkOp = value; }

    uint8_t get_pseEncodeType() const { return pseEncodeType; }
    void set_pseEncodeType(uint8_t value) { this->pseEncodeType = value; }

    uint8_t get_rsv() const { return rsv; }
    void set_rsv(uint8_t value) { this->rsv = value; }

    uint8_t get_needL1Carry() const { return needL1Carry; }
    void set_needL1Carry(uint8_t value) { this->needL1Carry = value; }

    uint16_t get_remain() const { return remain; }
    void set_remain(uint16_t value) { this->remain = value; }

    uint32_t get_attenMaskS2Size() const { return attenMaskS2Size; }
    void set_attenMaskS2Size(uint32_t value) { this->attenMaskS2Size = value; }

    uint32_t get_pseType() const { return pseType; }
    void set_pseType(uint32_t value) { this->pseType = value; }

    uint32_t get_rsv1() const { return rsv1; }
    void set_rsv1(uint32_t value) { this->rsv1 = value; }

    int64_t get_qStartIdx() const { return qStartIdx; }
    void set_qStartIdx(int64_t value) { this->qStartIdx = value; }

    int64_t get_kvStartIdx() const { return kvStartIdx; }
    void set_kvStartIdx(int64_t value) { this->kvStartIdx = value; }

    uint8_t get_tndSoftmaxOut() const { return tndSoftmaxOut; }
    void set_tndSoftmaxOut(uint8_t value) { this->tndSoftmaxOut = value; }

    void reset()
    {
        set_bSize(0);
        set_n2Size(0);
        set_gSize(0);
        set_s1Size(0);
        set_s2Size(0);
        set_alignedS2(0);
        set_dSize(0);
        set_keepProb(0);
        set_scaleValue(0);
        set_preTokens(0);
        set_nextTokens(0);
        set_pseS1Size(0);
        set_pseS2Size(0);
        set_pseBSize(0);
        set_bandIndex(0); // 13
        set_layoutType(0);
        set_pseShapeType(0);
        set_attenMaskShapeType(0);
        set_attenMaskDataType(0);
        set_attenMaskCompressMode(0);
        set_implMode(0);
        set_sparseType(0);
        set_needDropMaskOp(0);
        set_pseEncodeType(0);
        set_rsv(0);
        set_needL1Carry(0);
        set_remain(0);
        set_attenMaskS2Size(0);
        set_pseType(0); // 16
        set_rsv1(0);
        set_qStartIdx(0);
        set_kvStartIdx(0);
        set_tndSoftmaxOut(0);
        set_needSinkOp(0);
    }
};

class MultiCoreParams {
public:
    int32_t coreNum = 0;
    int32_t reserve = 0;
    int64_t totalSize = 0;
    int64_t splitFactorSize = 0;
    int64_t splitFactorTailSize = 0;
    int64_t sparseStartIdx[48] = {};

    // ========================
    // Getter & Setter 方法
    // ========================

    int32_t get_coreNum() const { return coreNum; }
    void set_coreNum(int32_t value) { this->coreNum = value; }

    int32_t get_reserve() const { return reserve; }
    void set_reserve(int32_t value) { this->reserve = value; }

    int64_t get_totalSize() const { return totalSize; }
    void set_totalSize(int64_t value) { this->totalSize = value; }

    int64_t get_splitFactorSize() const { return splitFactorSize; }
    void set_splitFactorSize(int64_t value) { this->splitFactorSize = value; }

    int64_t get_splitFactorTailSize() const { return splitFactorTailSize; }
    void set_splitFactorTailSize(int64_t value) { this->splitFactorTailSize = value; }

    int64_t* get_sparseStartIdx()  { return sparseStartIdx; }
    void set_sparseStartIdx(const int64_t* values) {
        for (int i = 0; i < 48; ++i) {
            sparseStartIdx[i] = values[i];
        }
    }

    void reset()
    {
        set_coreNum(0);
        set_reserve(0);
        set_totalSize(0);
        set_splitFactorSize(0);
        set_splitFactorTailSize(0);
        for (int64_t idx = 0; idx < 48; ++idx) {
            sparseStartIdx[idx] = 0;
        }
    }
};

class CoreParams {
public:
    int32_t s1BaseSize = 0;
    int32_t s1BaseTailSize = 0;
    int64_t s1OuterSize = 0;
    int32_t s1Vec2BaseSize = 0;
    int32_t s1Vec2BaseTailSize = 0;
    int64_t s1Vec2OuterSize = 0;
    int32_t s2BaseSize = 0;
    int32_t s2BaseTailSize = 0;
    int64_t s2OuterSize = 0;
    int32_t dBaseSize = 0;
    int32_t dBaseTailSize = 0;
    int64_t dOuterSize = 0;
    int32_t bBaseSize = 0;
    int32_t bBaseTailSize = 0;
    int64_t bOuterSize = 0;
    int32_t n2BaseSize = 0;
    int32_t n2BaseTailSize = 0;
    int64_t n2OuterSize = 0;
    int32_t gBaseSize = 0;
    int32_t gBaseTailSize = 0;
    int64_t gOuterSize = 0;
    int32_t nRatio = 0;
    int32_t dRopeSize = 0;
    int32_t d2Size = 0;
    uint8_t s1SparseValidSizePH[4] = {};
    int64_t s1SparseValidSize = 0;
    int64_t s2SparseValidSize = 0;
    int64_t pseAlibiBaseS1 = 0;
    int64_t pseAlibiBaseS2 = 0;

    // ========================
    // Getter & Setter 方法
    // ========================

    int32_t get_s1BaseSize() const { return s1BaseSize; }
    void set_s1BaseSize(int32_t value) { this->s1BaseSize = value; }

    int32_t get_s1BaseTailSize() const { return s1BaseTailSize; }
    void set_s1BaseTailSize(int32_t value) { this->s1BaseTailSize = value; }

    int64_t get_s1OuterSize() const { return s1OuterSize; }
    void set_s1OuterSize(int64_t value) { this->s1OuterSize = value; }

    int32_t get_s1Vec2BaseSize() const { return s1Vec2BaseSize; }
    void set_s1Vec2BaseSize(int32_t value) { this->s1Vec2BaseSize = value; }

    int32_t get_s1Vec2BaseTailSize() const { return s1Vec2BaseTailSize; }
    void set_s1Vec2BaseTailSize(int32_t value) { this->s1Vec2BaseTailSize = value; }

    int64_t get_s1Vec2OuterSize() const { return s1Vec2OuterSize; }
    void set_s1Vec2OuterSize(int64_t value) { this->s1Vec2OuterSize = value; }

    int32_t get_s2BaseSize() const { return s2BaseSize; }
    void set_s2BaseSize(int32_t value) { this->s2BaseSize = value; }

    int32_t get_s2BaseTailSize() const { return s2BaseTailSize; }
    void set_s2BaseTailSize(int32_t value) { this->s2BaseTailSize = value; }

    int64_t get_s2OuterSize() const { return s2OuterSize; }
    void set_s2OuterSize(int64_t value) { this->s2OuterSize = value; }

    int32_t get_dBaseSize() const { return dBaseSize; }
    void set_dBaseSize(int32_t value) { this->dBaseSize = value; }

    int32_t get_dBaseTailSize() const { return dBaseTailSize; }
    void set_dBaseTailSize(int32_t value) { this->dBaseTailSize = value; }

    int64_t get_dOuterSize() const { return dOuterSize; }
    void set_dOuterSize(int64_t value) { this->dOuterSize = value; }

    int32_t get_bBaseSize() const { return bBaseSize; }
    void set_bBaseSize(int32_t value) { this->bBaseSize = value; }

    int32_t get_bBaseTailSize() const { return bBaseTailSize; }
    void set_bBaseTailSize(int32_t value) { this->bBaseTailSize = value; }

    int64_t get_bOuterSize() const { return bOuterSize; }
    void set_bOuterSize(int64_t value) { this->bOuterSize = value; }

    int32_t get_n2BaseSize() const { return n2BaseSize; }
    void set_n2BaseSize(int32_t value) { this->n2BaseSize = value; }

    int32_t get_n2BaseTailSize() const { return n2BaseTailSize; }
    void set_n2BaseTailSize(int32_t value) { this->n2BaseTailSize = value; }

    int64_t get_n2OuterSize() const { return n2OuterSize; }
    void set_n2OuterSize(int64_t value) { this->n2OuterSize = value; }

    int32_t get_gBaseSize() const { return gBaseSize; }
    void set_gBaseSize(int32_t value) { this->gBaseSize = value; }

    int32_t get_gBaseTailSize() const { return gBaseTailSize; }
    void set_gBaseTailSize(int32_t value) { this->gBaseTailSize = value; }

    int64_t get_gOuterSize() const { return gOuterSize; }
    void set_gOuterSize(int64_t value) { this->gOuterSize = value; }

    int32_t get_nRatio() const { return nRatio; }
    void set_nRatio(int32_t value) { this->nRatio = value; }

    int32_t get_dRopeSize() const { return dRopeSize; }
    void set_dRopeSize(int32_t value) { this->dRopeSize = value; }

    int32_t get_d2Size() const { return d2Size; }
    void set_d2Size(int32_t value) { this->d2Size = value; }

    int64_t get_s1SparseValidSize() const { return s1SparseValidSize; }
    void set_s1SparseValidSize(int64_t value) { this->s1SparseValidSize = value; }

    int64_t get_s2SparseValidSize() const { return s2SparseValidSize; }
    void set_s2SparseValidSize(int64_t value) { this->s2SparseValidSize = value; }

    int64_t get_pseAlibiBaseS1() const { return pseAlibiBaseS1; }
    void set_pseAlibiBaseS1(int64_t value) { this->pseAlibiBaseS1 = value; }

    int64_t get_pseAlibiBaseS2() const { return pseAlibiBaseS2; }
    void set_pseAlibiBaseS2(int64_t value) { this->pseAlibiBaseS2 = value; }

    void reset()
    {
        set_s1BaseSize(0);
        set_s1BaseTailSize(0);
        set_s1OuterSize(0);
        set_s1Vec2BaseSize(0);
        set_s1Vec2BaseTailSize(0);
        set_s1Vec2OuterSize(0);
        set_s2BaseSize(0);
        set_s2BaseTailSize(0);
        set_s2OuterSize(0);
        set_dBaseSize(0);
        set_dBaseTailSize(0);
        set_dOuterSize(0);
        set_bBaseSize(0);
        set_bBaseTailSize(0);
        set_bOuterSize(0);
        set_n2BaseSize(0);
        set_n2BaseTailSize(0);
        set_n2OuterSize(0);
        set_gBaseSize(0);
        set_gBaseTailSize(0);
        set_gOuterSize(0);
        set_nRatio(0);
        set_dRopeSize(0);
        set_d2Size(0);
        set_s1SparseValidSize(0);
        set_s2SparseValidSize(0);
        set_pseAlibiBaseS1(0);
        set_pseAlibiBaseS2(0);
    }
};

class TensorSizeParams {
public:
    int32_t bmm1ResUbSize = 0;
    int32_t attenMaskUbSize = 0;
    int32_t pseUbSize = 0;
    int32_t dropMaskUbSize = 0;
    int32_t castUbSize = 0;
    int32_t softmaxMaxUbSize = 0;
    int32_t softmaxSumUbSize = 0;
    int32_t softmaxExpUbSize = 0;
    int32_t apiTmpBufferBytes = 0;
    int32_t bmm2ResUbSize = 0;
    int32_t inputQueBytes = 0;
    int32_t outputQueBytes = 0;
    int32_t tmpBufBytes = 0;
    int32_t softmaxMaxOffsetBytes = 0;
    int32_t softmaxSumOffsetBytes = 0;
    int32_t maxSumApiOffsetBytes = 0;
    int32_t customSoftmaxApiOffsetBytes = 0;
    int32_t pseTbufOffsetBytes = 0;
    int32_t dropoutApiOffsetBytes = 0;
    int32_t maxSumApiSize = 0;
    int32_t customSoftmaxApiSize = 0;
    int32_t dropoutApiSize = 0;
    int32_t attenMaskApiSize = 0;
    int32_t attenMaskApiOffsetBytes = 0;
    int32_t bmm1ProcessTInStage2Size = 0;
    int32_t bmm1ProcessTInStage2OffsetBytes = 0;
    int32_t wkspSection1OffsetBytes = 0;
    int32_t wkspSection2OffsetBytes = 0;

    // ========================
    // Getter & Setter 方法
    // ========================

    int32_t get_bmm1ResUbSize() const { return bmm1ResUbSize; }
    void set_bmm1ResUbSize(int32_t value) { this->bmm1ResUbSize = value; }

    int32_t get_attenMaskUbSize() const { return attenMaskUbSize; }
    void set_attenMaskUbSize(int32_t value) { this->attenMaskUbSize = value; }

    int32_t get_pseUbSize() const { return pseUbSize; }
    void set_pseUbSize(int32_t value) { this->pseUbSize = value; }

    int32_t get_dropMaskUbSize() const { return dropMaskUbSize; }
    void set_dropMaskUbSize(int32_t value) { this->dropMaskUbSize = value; }

    int32_t get_castUbSize() const { return castUbSize; }
    void set_castUbSize(int32_t value) { this->castUbSize = value; }

    int32_t get_softmaxMaxUbSize() const { return softmaxMaxUbSize; }
    void set_softmaxMaxUbSize(int32_t value) { this->softmaxMaxUbSize = value; }

    int32_t get_softmaxSumUbSize() const { return softmaxSumUbSize; }
    void set_softmaxSumUbSize(int32_t value) { this->softmaxSumUbSize = value; }

    int32_t get_softmaxExpUbSize() const { return softmaxExpUbSize; }
    void set_softmaxExpUbSize(int32_t value) { this->softmaxExpUbSize = value; }

    int32_t get_apiTmpBufferBytes() const { return apiTmpBufferBytes; }
    void set_apiTmpBufferBytes(int32_t value) { this->apiTmpBufferBytes = value; }

    int32_t get_bmm2ResUbSize() const { return bmm2ResUbSize; }
    void set_bmm2ResUbSize(int32_t value) { this->bmm2ResUbSize = value; }

    int32_t get_inputQueBytes() const { return inputQueBytes; }
    void set_inputQueBytes(int32_t value) { this->inputQueBytes = value; }

    int32_t get_outputQueBytes() const { return outputQueBytes; }
    void set_outputQueBytes(int32_t value) { this->outputQueBytes = value; }

    int32_t get_tmpBufBytes() const { return tmpBufBytes; }
    void set_tmpBufBytes(int32_t value) { this->tmpBufBytes = value; }

    int32_t get_softmaxMaxOffsetBytes() const { return softmaxMaxOffsetBytes; }
    void set_softmaxMaxOffsetBytes(int32_t value) { this->softmaxMaxOffsetBytes = value; }

    int32_t get_softmaxSumOffsetBytes() const { return softmaxSumOffsetBytes; }
    void set_softmaxSumOffsetBytes(int32_t value) { this->softmaxSumOffsetBytes = value; }

    int32_t get_maxSumApiOffsetBytes() const { return maxSumApiOffsetBytes; }
    void set_maxSumApiOffsetBytes(int32_t value) { this->maxSumApiOffsetBytes = value; }

    int32_t get_customSoftmaxApiOffsetBytes() const { return customSoftmaxApiOffsetBytes; }
    void set_customSoftmaxApiOffsetBytes(int32_t value) { this->customSoftmaxApiOffsetBytes = value; }

    int32_t get_pseTbufOffsetBytes() const { return pseTbufOffsetBytes; }
    void set_pseTbufOffsetBytes(int32_t value) { this->pseTbufOffsetBytes = value; }

    int32_t get_dropoutApiOffsetBytes() const { return dropoutApiOffsetBytes; }
    void set_dropoutApiOffsetBytes(int32_t value) { this->dropoutApiOffsetBytes = value; }

    int32_t get_maxSumApiSize() const { return maxSumApiSize; }
    void set_maxSumApiSize(int32_t value) { this->maxSumApiSize = value; }

    int32_t get_customSoftmaxApiSize() const { return customSoftmaxApiSize; }
    void set_customSoftmaxApiSize(int32_t value) { this->customSoftmaxApiSize = value; }

    int32_t get_dropoutApiSize() const { return dropoutApiSize; }
    void set_dropoutApiSize(int32_t value) { this->dropoutApiSize = value; }

    int32_t get_attenMaskApiSize() const { return attenMaskApiSize; }
    void set_attenMaskApiSize(int32_t value) { this->attenMaskApiSize = value; }

    int32_t get_attenMaskApiOffsetBytes() const { return attenMaskApiOffsetBytes; }
    void set_attenMaskApiOffsetBytes(int32_t value) { this->attenMaskApiOffsetBytes = value; }

    int32_t get_bmm1ProcessTInStage2Size() const { return bmm1ProcessTInStage2Size; }
    void set_bmm1ProcessTInStage2Size(int32_t value) { this->bmm1ProcessTInStage2Size = value; }

    int32_t get_bmm1ProcessTInStage2OffsetBytes() const { return bmm1ProcessTInStage2OffsetBytes; }
    void set_bmm1ProcessTInStage2OffsetBytes(int32_t value) { this->bmm1ProcessTInStage2OffsetBytes = value; }

    int32_t get_wkspSection1OffsetBytes() const { return wkspSection1OffsetBytes; }
    void set_wkspSection1OffsetBytes(int32_t value) { this->wkspSection1OffsetBytes = value; }

    int32_t get_wkspSection2OffsetBytes() const { return wkspSection2OffsetBytes; }
    void set_wkspSection2OffsetBytes(int32_t value) { this->wkspSection2OffsetBytes = value; }

    void reset()
    {
        set_bmm1ResUbSize(0);
        set_attenMaskUbSize(0);
        set_pseUbSize(0);
        set_dropMaskUbSize(0);
        set_castUbSize(0);
        set_softmaxMaxUbSize(0);
        set_softmaxSumUbSize(0);
        set_softmaxExpUbSize(0);
        set_apiTmpBufferBytes(0);
        set_bmm2ResUbSize(0);
        set_inputQueBytes(0);
        set_outputQueBytes(0);
        set_tmpBufBytes(0);
        set_softmaxMaxOffsetBytes(0);
        set_softmaxSumOffsetBytes(0);
        set_maxSumApiOffsetBytes(0);
        set_customSoftmaxApiOffsetBytes(0);
        set_pseTbufOffsetBytes(0);
        set_dropoutApiOffsetBytes(0);
        set_maxSumApiSize(0);
        set_customSoftmaxApiSize(0);
        set_dropoutApiSize(0);
        set_attenMaskApiSize(0);
        set_attenMaskApiOffsetBytes(0);
        set_bmm1ProcessTInStage2Size(0);
        set_bmm1ProcessTInStage2OffsetBytes(0);
        set_wkspSection1OffsetBytes(0);
        set_wkspSection2OffsetBytes(0);
    }
};

class MaxSumTiling {
public:
    // 源张量维度（输入）
    uint32_t srcM = 0;
    uint32_t srcK = 0;
    uint32_t srcSize = 0;

    // 输出最大维度（目标）
    uint32_t outMaxM = 0;
    uint32_t outMaxK = 0;
    uint32_t outMaxSize = 0;

    // 分片参数（用于 tiling 策略）
    uint32_t splitM = 0;
    uint32_t splitK = 0;
    uint32_t splitSize = 0;

    // 还原/归约维度（reduce 阶段使用）
    uint32_t reduceM = 0;
    uint32_t reduceK = 0;
    uint32_t reduceSize = 0;

    // 范围与尾部处理参数（tail handling）
    uint32_t rangeM = 0;
    uint32_t tailM = 0;
    uint32_t tailSplitSize = 0;
    uint32_t tailReduceSize = 0;

    // ========================
    // Getter & Setter 方法
    // ========================

    uint32_t get_srcM() const { return srcM; }
    void set_srcM(uint32_t value) { this->srcM = value; }

    uint32_t get_srcK() const { return srcK; }
    void set_srcK(uint32_t value) { this->srcK = value; }

    uint32_t get_srcSize() const { return srcSize; }
    void set_srcSize(uint32_t value) { this->srcSize = value; }

    uint32_t get_outMaxM() const { return outMaxM; }
    void set_outMaxM(uint32_t value) { this->outMaxM = value; }

    uint32_t get_outMaxK() const { return outMaxK; }
    void set_outMaxK(uint32_t value) { this->outMaxK = value; }

    uint32_t get_outMaxSize() const { return outMaxSize; }
    void set_outMaxSize(uint32_t value) { this->outMaxSize = value; }

    uint32_t get_splitM() const { return splitM; }
    void set_splitM(uint32_t value) { this->splitM = value; }

    uint32_t get_splitK() const { return splitK; }
    void set_splitK(uint32_t value) { this->splitK = value; }

    uint32_t get_splitSize() const { return splitSize; }
    void set_splitSize(uint32_t value) { this->splitSize = value; }

    uint32_t get_reduceM() const { return reduceM; }
    void set_reduceM(uint32_t value) { this->reduceM = value; }

    uint32_t get_reduceK() const { return reduceK; }
    void set_reduceK(uint32_t value) { this->reduceK = value; }

    uint32_t get_reduceSize() const { return reduceSize; }
    void set_reduceSize(uint32_t value) { this->reduceSize = value; }

    uint32_t get_rangeM() const { return rangeM; }
    void set_rangeM(uint32_t value) { this->rangeM = value; }

    uint32_t get_tailM() const { return tailM; }
    void set_tailM(uint32_t value) { this->tailM = value; }

    uint32_t get_tailSplitSize() const { return tailSplitSize; }
    void set_tailSplitSize(uint32_t value) { this->tailSplitSize = value; }

    uint32_t get_tailReduceSize() const { return tailReduceSize; }
    void set_tailReduceSize(uint32_t value) { this->tailReduceSize = value; }
};

class DropmaskParams {
public:
    // 基础参数：控制多核分块的基本配置
    int32_t multiCoreFactorSize = 0;
    int32_t baseUbCalSize = 0;

    // 总大小计算：用于内存分配与调度决策
    int64_t multiCoreTotalSize = 0;
    int64_t shapeTotalSize = 0;

    int32_t get_multiCoreFactorSize() const { return multiCoreFactorSize; }
    void set_multiCoreFactorSize(int32_t value) {
        this->multiCoreFactorSize = value;
    }

    int32_t get_baseUbCalSize() const { return baseUbCalSize; }
    void set_baseUbCalSize(int32_t value) {
        this->baseUbCalSize = value;
    }

    int64_t get_multiCoreTotalSize() const { return multiCoreTotalSize; }
    void set_multiCoreTotalSize(int64_t value) {
        this->multiCoreTotalSize = value;
    }

    int64_t get_shapeTotalSize() const { return shapeTotalSize; }
    void set_shapeTotalSize(int64_t value) {
        this->shapeTotalSize = value;
    }

    void reset()
    {
        set_multiCoreFactorSize(0);
        set_baseUbCalSize(0);
        set_multiCoreTotalSize(0);
        set_shapeTotalSize(0);
    }
};

class CopyTransposeTiling
{
public:
    uint32_t dstShapeB = 0;
    uint32_t dstShapeN = 0;
    uint32_t dstShapeS = 0;
    uint32_t dstShapeHN = 0;
    uint32_t dstShapeH = 0;
    uint32_t srcShapeB = 0;
    uint32_t srcShapeN = 0;
    uint32_t srcShapeS = 0;
    uint32_t srcShapeHN = 0;
    uint32_t originalShapeNLen = 0;
    uint32_t shapeSHValue = 0;
    uint32_t shapeNsValue = 0;
    uint32_t shapeNsnValue = 0;
    uint32_t invalidParamCopyTransposeTiling = 0;
    uint32_t shapeBHValue = 0;
    uint32_t paramsAlign = 0;

    uint32_t get_dstShapeB() const { return dstShapeB; }
    void set_dstShapeB(uint32_t value) { this->dstShapeB = value; }

    uint32_t get_dstShapeN() const { return dstShapeN; }
    void set_dstShapeN(uint32_t value) { this->dstShapeN = value; }

    uint32_t get_dstShapeS() const { return dstShapeS; }
    void set_dstShapeS(uint32_t value) { this->dstShapeS = value; }

    uint32_t get_dstShapeHN() const { return dstShapeHN; }
    void set_dstShapeHN(uint32_t value) { this->dstShapeHN = value; }

    uint32_t get_dstShapeH() const { return dstShapeH; }
    void set_dstShapeH(uint32_t value) { this->dstShapeH = value; }

    uint32_t get_srcShapeB() const { return srcShapeB; }
    void set_srcShapeB(uint32_t value) { this->srcShapeB = value; }

    uint32_t get_srcShapeN() const { return srcShapeN; }
    void set_srcShapeN(uint32_t value) { this->srcShapeN = value; }

    uint32_t get_srcShapeS() const { return srcShapeS; }
    void set_srcShapeS(uint32_t value) { this->srcShapeS = value; }

    uint32_t get_srcShapeHN() const { return srcShapeHN; }
    void set_srcShapeHN(uint32_t value) { this->srcShapeHN = value; }

    uint32_t get_originalShapeNLen() const { return originalShapeNLen; }
    void set_originalShapeNLen(uint32_t value) { this->originalShapeNLen = value; }

    uint32_t get_shapeSHValue() const { return shapeSHValue; }
    void set_shapeSHValue(uint32_t value) { this->shapeSHValue = value; }

    uint32_t get_shapeNsValue() const { return shapeNsValue; }
    void set_shapeNsValue(uint32_t value) { this->shapeNsValue = value; }

    uint32_t get_shapeNsnValue() const { return shapeNsnValue; }
    void set_shapeNsnValue(uint32_t value) { this->shapeNsnValue = value; }

    uint32_t get_invalidParamCopyTransposeTiling() const { return invalidParamCopyTransposeTiling; }
    void set_invalidParamCopyTransposeTiling(uint32_t value) { this->shapeNsnValue = value; }

    uint32_t get_shapeBHValue() const { return shapeBHValue; }
    void set_shapeBHValue(uint32_t value) { this->shapeBHValue = value; }

    uint32_t get_paramsAlign() const { return paramsAlign; }
    void set_paramsAlign(uint32_t value) { this->paramsAlign = value; }

    void GetDataCopyTransposeTiling(const int64_t dst0, const int64_t dst1, const int64_t dst2, const int64_t dst3,
                                    const int64_t src0, const int64_t src1, const int64_t src2, const int64_t src3,
                                    const uint32_t typeSize)
    {
        this->set_dstShapeB(dst0);
        this->set_dstShapeN(dst1);
        this->set_dstShapeS(dst2);
        this->set_dstShapeH(dst3);
        this->set_dstShapeHN(this->get_dstShapeH() / this->get_dstShapeN());

        this->set_srcShapeB(src0);
        this->set_srcShapeN(src1);
        this->set_srcShapeS(src2);
        this->set_srcShapeHN(src3);
        this->set_originalShapeNLen(this->get_srcShapeHN() * typeSize);
        this->set_shapeSHValue(this->get_dstShapeS() * this->get_dstShapeH());
        this->set_shapeNsValue(this->get_dstShapeN() * this->get_dstShapeS());
        this->set_shapeNsnValue(this->get_dstShapeN() * this->get_srcShapeS() * this->get_srcShapeN());
        this->set_shapeBHValue(this->get_dstShapeB() * this->get_dstShapeH());
    }

    void reset()
    {
        set_dstShapeB(0);
        set_dstShapeN(0);
        set_dstShapeS(0);
        set_dstShapeHN(0);
        set_dstShapeH(0);
        set_srcShapeB(0);
        set_srcShapeN(0);
        set_srcShapeS(0);
        set_srcShapeHN(0);
        set_originalShapeNLen(0);
        set_shapeSHValue(0);
        set_shapeNsValue(0);
        set_shapeNsnValue(0);
        set_invalidParamCopyTransposeTiling(0);
        set_shapeBHValue(0);
        set_paramsAlign(0);
    }
};

class FlashAttentionScoreTilingData {
public:
    BaseParams baseParams;
    AttentionScoreCoreParams attentionScoreCoreParams;
    AttentionScoreSingleCoreParams attentionScoreSingleCoreParams;
    AttentionScoreSingleCoreTensorSize attentionScoreSingleCoreTensorSize;
    AttentionScoreOffestStrideParams attentionScoreOffestStrideParams;
    TCubeTiling mmTilingData;
    TCubeTiling bmm1TilingData;
    TCubeTiling bmm2TilingData;
    TCubeTiling bmm1TilingDataTailCore;
    TCubeTiling bmm2TilingDataTailCore;
    SoftMaxTiling softmaxTilingData;
    SoftMaxTiling softmaxFlashTilingData;
    SoftMaxTiling softmaxTilingDataTailCore;
    SoftMaxTiling softmaxFlashTilingDataTailCore;
    CopyTransposeTiling transposeTilingData;
    CopyTransposeTiling transposeTilingDataTailCore;
    FlashAttentionScoreEmptyInputTilingData emptyInputTilingData;

    void reset()
    {
        baseParams.reset();
        attentionScoreCoreParams.reset();
        attentionScoreSingleCoreParams.reset();
        attentionScoreSingleCoreTensorSize.reset();
        attentionScoreOffestStrideParams.reset();
        transposeTilingData.reset();
        transposeTilingDataTailCore.reset();
        emptyInputTilingData.reset();
    }
};

class FlashAttentionScoreGeneralTilingData {
public:
    InputParams inputParams; // 20 *8 = 160
    MultiCoreParams multiCoreParams; // 52 * 8 = 416
    CoreParams coreParams; // 20 * 8 = 160
    TensorSizeParams tensorSizeParams; // 28 * 4byte = 112
    TCubeTiling bmm1TilingData; // 50 *4 = 200
    TCubeTiling bmm2TilingData; // 200
    SoftMaxTiling softmaxFlashTilingData;   // 16*4 = 64
    CopyTransposeTiling transposeTilingData;  // 16 * 4byte = 64
    CopyTransposeTiling transposeTilingDataTailCore; // 64
    DropmaskParams dropmaskParams;  // 2 * 4byte + 2 * 8byte = 24

    void reset()
    {
        inputParams.reset();
        multiCoreParams.reset();
        coreParams.reset();
        tensorSizeParams.reset();
        transposeTilingData.reset();
        transposeTilingDataTailCore.reset();
    }
};
// }  // namespace optiling