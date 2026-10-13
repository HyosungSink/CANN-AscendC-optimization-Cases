/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_TILING_H 
#define FLASH_ATTENTION_SCORE_GRAD_TILING_H

#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

class FlashAttentionScoreGradS1S2BNGS1S2SABBaseParams {
public:
    int64_t b = 0;
    int64_t n2 = 0;
    int64_t g = 0;
    int64_t s1 = 0;
    int64_t s2 = 0;
    int64_t d = 0;
    int64_t rope_d = 0;
    int64_t value_d = 0;
    float scaleValue = 0;
    float keepProb = 0;
    int64_t s1Token = 0;
    int64_t s2Token = 0;
    uint32_t sparseMode = 0;
    uint32_t isSparse = 0;
    int64_t attenMaskS2Size = 0;
    uint32_t coreNum = 0;
    uint32_t attenMaskCompressMode = 0;
    int64_t qStartIdx = 0;
    int64_t kvStartIdx = 0;
    int64_t pseAlibiBaseS1 = 0;
    int64_t pseAlibiBaseS2 = 0;
    uint32_t pseType = 0;
    uint32_t pseOptional = 0;
    uint32_t pseShapeType = 0;
    uint32_t pseDtype = 0;
    uint32_t attenMaskOptional = 0;
    uint32_t attenMaskDtype = 0;
    uint32_t attenMaskShapeType = 0;
    uint32_t pad = 0;
    uint8_t tndSoftmaxIn = 0;
    uint32_t sink = 0;
    uint8_t FlashAttentionScoreGradS1S2BNGS1S2SABBaseParamsPH[3] = {};

    int64_t get_b() const
    {
        return b;
    }
    void set_b(int64_t b_val)
    {
        this->b = b_val;
    }

    int64_t get_n2() const
    {
        return n2;
    }
    void set_n2(int64_t n2_val)
    {
        this->n2 = n2_val;
    }

    int64_t get_g() const
    {
        return g;
    }
    void set_g(int64_t g_val)
    {
        this->g = g_val;
    }

    int64_t get_s1() const
    {
        return s1;
    }
    void set_s1(int64_t s1_val)
    {
        this->s1 = s1_val;
    }

    int64_t get_s2() const
    {
        return s2;
    }
    void set_s2(int64_t s2_val)
    {
        this->s2 = s2_val;
    }

    int64_t get_d() const
    {
        return d;
    }
    void set_d(int64_t d_val)
    {
        this->d = d_val;
    }

    int64_t get_rope_d() const
    {
        return rope_d;
    }
    void set_rope_d(int64_t rope_d_val)
    {
        this->rope_d = rope_d_val;
    }

    int64_t get_value_d() const
    {
        return value_d;
    }
    void set_value_d(int64_t value_d_val)
    {
        this->value_d = value_d_val;
    }

    float get_scaleValue() const
    {
        return scaleValue;
    }
    void set_scaleValue(float scaleValue_val)
    {
        this->scaleValue = scaleValue_val;
    }

    float get_keepProb() const
    {
        return keepProb;
    }
    void set_keepProb(float keepProb_val)
    {
        this->keepProb = keepProb_val;
    }

    int64_t get_s1Token() const
    {
        return s1Token;
    }
    void set_s1Token(int64_t s1Token_val)
    {
        this->s1Token = s1Token_val;
    }

    int64_t get_s2Token() const
    {
        return s2Token;
    }
    void set_s2Token(int64_t s2Token_val)
    {
        this->s2Token = s2Token_val;
    }

    uint32_t get_sparseMode() const
    {
        return sparseMode;
    }
    void set_sparseMode(uint32_t sparseMode_val)
    {
        this->sparseMode = sparseMode_val;
    }

    uint32_t get_isSparse() const
    {
        return isSparse;
    }
    void set_isSparse(uint32_t isSparse_val)
    {
        this->isSparse = isSparse_val;
    }

    int64_t get_attenMaskS2Size() const
    {
        return attenMaskS2Size;
    }
    void set_attenMaskS2Size(int64_t attenMaskS2Size_val)
    {
        this->attenMaskS2Size = attenMaskS2Size_val;
    }

    uint32_t get_coreNum() const
    {
        return coreNum;
    }
    void set_coreNum(uint32_t coreNum_val)
    {
        this->coreNum = coreNum_val;
    }

    uint32_t get_attenMaskCompressMode() const
    {
        return attenMaskCompressMode;
    }
    void set_attenMaskCompressMode(uint32_t attenMaskCompressMode_val)
    {
        this->attenMaskCompressMode = attenMaskCompressMode_val;
    }

    int64_t get_qStartIdx() const
    {
        return qStartIdx;
    }
    void set_qStartIdx(int64_t qStartIdx_val)
    {
        this->qStartIdx = qStartIdx_val;
    }

    int64_t get_kvStartIdx() const
    {
        return kvStartIdx;
    }
    void set_kvStartIdx(int64_t kvStartIdx_val)
    {
        this->kvStartIdx = kvStartIdx_val;
    }

    int64_t get_pseAlibiBaseS1() const
    {
        return pseAlibiBaseS1;
    }
    void set_pseAlibiBaseS1(int64_t pseAlibiBaseS1_val)
    {
        this->pseAlibiBaseS1 = pseAlibiBaseS1_val;
    }

    int64_t get_pseAlibiBaseS2() const
    {
        return pseAlibiBaseS2;
    }
    void set_pseAlibiBaseS2(int64_t pseAlibiBaseS2_val)
    {
        this->pseAlibiBaseS2 = pseAlibiBaseS2_val;
    }

    uint32_t get_pseType() const
    {
        return pseType;
    }
    void set_pseType(uint32_t pseType_val)
    {
        this->pseType = pseType_val;
    }

    uint32_t get_pseOptional() const
    {
        return pseOptional;
    }
    void set_pseOptional(uint32_t pseOptional_val)
    {
        this->pseOptional = pseOptional_val;
    }

    uint32_t get_pseShapeType() const
    {
        return pseShapeType;
    }
    void set_pseShapeType(uint32_t pseShapeType_val)
    {
        this->pseShapeType = pseShapeType_val;
    }

    uint32_t get_pseDtype() const
    {
        return pseDtype;
    }
    void set_pseDtype(uint32_t pseDtype_val)
    {
        this->pseDtype = pseDtype_val;
    }

    uint32_t get_attenMaskOptional() const
    {
        return attenMaskOptional;
    }
    void set_attenMaskOptional(uint32_t attenMaskOptional_val)
    {
        this->attenMaskOptional = attenMaskOptional_val;
    }

    uint32_t get_attenMaskDtype() const
    {
        return attenMaskDtype;
    }
    void set_attenMaskDtype(uint32_t attenMaskDtype_val)
    {
        this->attenMaskDtype = attenMaskDtype_val;
    }

    uint32_t get_attenMaskShapeType() const
    {
        return attenMaskShapeType;
    }
    void set_attenMaskShapeType(uint32_t attenMaskShapeType_val)
    {
        this->attenMaskShapeType = attenMaskShapeType_val;
    }

    uint32_t get_pad() const
    {
        return pad;
    }
    void set_pad(uint32_t pad_val)
    {
        this->pad = pad_val;
    }

    uint8_t get_tndSoftmaxIn() const
    {
        return tndSoftmaxIn;
    }
    void set_tndSoftmaxIn(uint8_t tndSoftmaxIn_val)
    {
        this->tndSoftmaxIn = tndSoftmaxIn_val;
    }

    uint32_t get_sink() const
    {
        return sink;
    }

    void set_sink(uint32_t sink_val)
    {
        this->sink = sink_val;
    }

    void reset()
    {
        b = 0;
        n2 = 0;
        g = 0;
        s1 = 0;
        s2 = 0;
        d = 0;
        rope_d = 0;
        value_d = 0;
        scaleValue = 0;
        keepProb = 0;
        s1Token = 0;
        s2Token = 0;
        sparseMode = 0;
        isSparse = 0;
        attenMaskS2Size = 0;
        coreNum = 0;
        attenMaskCompressMode = 0;
        qStartIdx = 0;
        kvStartIdx = 0;
        pseAlibiBaseS1 = 0;
        pseAlibiBaseS2 = 0;
        pseType = 0;
        pseOptional = 0;
        pseShapeType = 0;
        pseDtype = 0;
        attenMaskOptional = 0;
        attenMaskDtype = 0;
        attenMaskShapeType = 0;
        pad = 0;
        tndSoftmaxIn = 0;
        sink = 0;
    }
};

class FlashAttentionScoreGradS1S2BNGS1S2SABSplitCoreParams {
public:
    int64_t s1Outer = 0;
    uint32_t s1Inner = 0;
    uint32_t s1CvInner = 0;
    uint32_t s1Tail = 0;
    uint32_t s1CvTail = 0;
    int64_t s2Outer = 0;
    uint32_t s2Inner = 0;
    uint32_t s2CvInner = 0;
    uint32_t s2Tail = 0;
    uint32_t baseMN = 0;
    uint32_t blockOuter = 0;
    uint8_t bandIdxPH[4] = {};
    int64_t bandIdx = 0;

    int64_t get_s1Outer() const
    {
        return s1Outer;
    }
    void set_s1Outer(int64_t s1Outer_val)
    {
        this->s1Outer = s1Outer_val;
    }

    uint32_t get_s1Inner() const
    {
        return s1Inner;
    }
    void set_s1Inner(uint32_t s1Inner_val)
    {
        this->s1Inner = s1Inner_val;
    }

    uint32_t get_s1CvInner() const
    {
        return s1CvInner;
    }
    void set_s1CvInner(uint32_t s1CvInner_val)
    {
        this->s1CvInner = s1CvInner_val;
    }

    uint32_t get_s1Tail() const
    {
        return s1Tail;
    }
    void set_s1Tail(uint32_t s1Tail_val)
    {
        this->s1Tail = s1Tail_val;
    }

    uint32_t get_s1CvTail() const
    {
        return s1CvTail;
    }
    void set_s1CvTail(uint32_t s1CvTail_val)
    {
        this->s1CvTail = s1CvTail_val;
    }

    int64_t get_s2Outer() const
    {
        return s2Outer;
    }
    void set_s2Outer(int64_t s2Outer_val)
    {
        this->s2Outer = s2Outer_val;
    }

    uint32_t get_s2Inner() const
    {
        return s2Inner;
    }
    void set_s2Inner(uint32_t s2Inner_val)
    {
        this->s2Inner = s2Inner_val;
    }

    uint32_t get_s2CvInner() const
    {
        return s2CvInner;
    }
    void set_s2CvInner(uint32_t s2CvInner_val)
    {
        this->s2CvInner = s2CvInner_val;
    }

    uint32_t get_s2Tail() const
    {
        return s2Tail;
    }
    void set_s2Tail(uint32_t s2Tail_val)
    {
        this->s2Tail = s2Tail_val;
    }

    uint32_t get_baseMN() const
    {
        return baseMN;
    }
    void set_baseMN(uint32_t baseMN_val)
    {
        this->baseMN = baseMN_val;
    }

    uint32_t get_blockOuter() const
    {
        return blockOuter;
    }
    void set_blockOuter(uint32_t blockOuter_val)
    {
        this->blockOuter = blockOuter_val;
    }

    uint32_t get_bandIdx() const
    {
        return bandIdx;
    }
    void set_bandIdx(uint32_t bandIdx_val)
    {
        this->bandIdx = bandIdx_val;
    }

    void reset()
    {
        s1Outer = 0;
        s1Inner = 0;
        s1CvInner = 0;
        s1Tail = 0;
        s1CvTail = 0;
        s2Outer = 0;
        s2Inner = 0;
        s2CvInner = 0;
        s2Tail = 0;
        baseMN = 0;
        blockOuter = 0;
        bandIdx = 0;
    }
};

class PreParams {
public:
    uint64_t maskPreBlockTotal = 0;
    uint64_t maskSingleCoreNum = 0;
    uint32_t qPreBlockFactor = 0;
    uint32_t qPreBlockTotal = 0;
    uint32_t qPreBlockTail = 0;
    uint32_t qRopePreBlockFactor = 0;
    uint32_t qRopePreBlockTotal = 0;
    uint32_t qRopePreBlockTail = 0;
    uint32_t kvPreBlockFactor = 0;
    uint32_t kvPreBlockTotal = 0;
    uint32_t kvPreBlockTail = 0;
    uint32_t kRopePreBlockFactor = 0;
    uint32_t kRopePreBlockTotal = 0;
    uint32_t kRopePreBlockTail = 0;
    uint32_t vPreBlockFactor = 0;
    uint32_t vPreBlockTotal = 0;
    uint32_t vPreBlockTail = 0;
    uint32_t maskCoreNum = 0;
    uint32_t castBufferLen = 0;
    uint32_t outputBufferLen = 0;
    uint32_t inputBufferLen = 0;
    uint32_t singleUBProcessNum = 0;
    uint32_t maskSingleCoreLoop = 0;
    uint32_t maskLastLoopNum = 0;
    uint32_t maskTailCoreLoop = 0;
    uint32_t maskTailCoreLastLoopNum = 0;
    uint32_t dropoutIsDivisibleBy8 = 0;
    uint8_t dropBeginAddrPH[4] = {};
    uint64_t dropBeginAddr = 0;
    uint32_t reserved = 0;
    uint8_t PreParamsPH[4] = {};

    uint64_t get_maskPreBlockTotal() const
    {
        return maskPreBlockTotal;
    }
    void set_maskPreBlockTotal(uint64_t maskPreBlockTotal_val)
    {
        this->maskPreBlockTotal = maskPreBlockTotal_val;
    }

    uint64_t get_maskSingleCoreNum() const
    {
        return maskSingleCoreNum;
    }
    void set_maskSingleCoreNum(uint64_t maskSingleCoreNum_val)
    {
        this->maskSingleCoreNum = maskSingleCoreNum_val;
    }

    uint32_t get_qPreBlockFactor() const
    {
        return qPreBlockFactor;
    }
    void set_qPreBlockFactor(uint32_t qPreBlockFactor_val)
    {
        this->qPreBlockFactor = qPreBlockFactor_val;
    }

    uint32_t get_qPreBlockTotal() const
    {
        return qPreBlockTotal;
    }
    void set_qPreBlockTotal(uint32_t qPreBlockTotal_val)
    {
        this->qPreBlockTotal = qPreBlockTotal_val;
    }

    uint32_t get_qPreBlockTail() const
    {
        return qPreBlockTail;
    }
    void set_qPreBlockTail(uint32_t qPreBlockTail_val)
    {
        this->qPreBlockTail = qPreBlockTail_val;
    }

    uint32_t get_qRopePreBlockFactor() const
    {
        return qRopePreBlockFactor;
    }
    void set_qRopePreBlockFactor(uint32_t qRopePreBlockFactor_val)
    {
        this->qRopePreBlockFactor = qRopePreBlockFactor_val;
    }

    uint32_t get_qRopePreBlockTotal() const
    {
        return qRopePreBlockTotal;
    }
    void set_qRopePreBlockTotal(uint32_t qRopePreBlockTotal_val)
    {
        this->qRopePreBlockTotal = qRopePreBlockTotal_val;
    }

    uint32_t get_qRopePreBlockTail() const
    {
        return qRopePreBlockTail;
    }
    void set_qRopePreBlockTail(uint32_t qRopePreBlockTail_val)
    {
        this->qRopePreBlockTail = qRopePreBlockTail_val;
    }

    uint32_t get_kvPreBlockFactor() const
    {
        return kvPreBlockFactor;
    }
    void set_kvPreBlockFactor(uint32_t kvPreBlockFactor_val)
    {
        this->kvPreBlockFactor = kvPreBlockFactor_val;
    }

    uint32_t get_kvPreBlockTotal() const
    {
        return kvPreBlockTotal;
    }
    void set_kvPreBlockTotal(uint32_t kvPreBlockTotal_val)
    {
        this->kvPreBlockTotal = kvPreBlockTotal_val;
    }

    uint32_t get_kvPreBlockTail() const
    {
        return kvPreBlockTail;
    }
    void set_kvPreBlockTail(uint32_t kvPreBlockTail_val)
    {
        this->kvPreBlockTail = kvPreBlockTail_val;
    }

    uint32_t get_kRopePreBlockFactor() const
    {
        return kRopePreBlockFactor;
    }
    void set_kRopePreBlockFactor(uint32_t kRopePreBlockFactor_val)
    {
        this->kRopePreBlockFactor = kRopePreBlockFactor_val;
    }

    uint32_t get_kRopePreBlockTotal() const
    {
        return kRopePreBlockTotal;
    }
    void set_kRopePreBlockTotal(uint32_t kRopePreBlockTotal_val)
    {
        this->kRopePreBlockTotal = kRopePreBlockTotal_val;
    }

    uint32_t get_kRopePreBlockTail() const
    {
        return kRopePreBlockTail;
    }
    void set_kRopePreBlockTail(uint32_t kRopePreBlockTail_val)
    {
        this->kRopePreBlockTail = kRopePreBlockTail_val;
    }

    uint32_t get_vPreBlockFactor() const
    {
        return vPreBlockFactor;
    }
    void set_vPreBlockFactor(uint32_t vPreBlockFactor_val)
    {
        this->vPreBlockFactor = vPreBlockFactor_val;
    }

    uint32_t get_vPreBlockTotal() const
    {
        return vPreBlockTotal;
    }
    void set_vPreBlockTotal(uint32_t vPreBlockTotal_val)
    {
        this->vPreBlockTotal = vPreBlockTotal_val;
    }

    uint32_t get_vPreBlockTail() const
    {
        return vPreBlockTail;
    }
    void set_vPreBlockTail(uint32_t vPreBlockTail_val)
    {
        this->vPreBlockTail = vPreBlockTail_val;
    }

    uint32_t get_maskCoreNum() const
    {
        return maskCoreNum;
    }
    void set_maskCoreNum(uint32_t maskCoreNum_val)
    {
        this->maskCoreNum = maskCoreNum_val;
    }

    uint32_t get_castBufferLen() const
    {
        return castBufferLen;
    }
    void set_castBufferLen(uint32_t castBufferLen_val)
    {
        this->castBufferLen = castBufferLen_val;
    }

    uint32_t get_outputBufferLen() const
    {
        return outputBufferLen;
    }
    void set_outputBufferLen(uint32_t outputBufferLen_val)
    {
        this->outputBufferLen = outputBufferLen_val;
    }

    uint32_t get_inputBufferLen() const
    {
        return inputBufferLen;
    }
    void set_inputBufferLen(uint32_t inputBufferLen_val)
    {
        this->inputBufferLen = inputBufferLen_val;
    }

    uint32_t get_singleUBProcessNum() const
    {
        return singleUBProcessNum;
    }
    void set_singleUBProcessNum(uint32_t singleUBProcessNum_val)
    {
        this->singleUBProcessNum = singleUBProcessNum_val;
    }

    uint32_t get_maskSingleCoreLoop() const
    {
        return maskSingleCoreLoop;
    }
    void set_maskSingleCoreLoop(uint32_t maskSingleCoreLoop_val)
    {
        this->maskSingleCoreLoop = maskSingleCoreLoop_val;
    }

    uint32_t get_maskLastLoopNum() const
    {
        return maskLastLoopNum;
    }
    void set_maskLastLoopNum(uint32_t maskLastLoopNum_val)
    {
        this->maskLastLoopNum = maskLastLoopNum_val;
    }

    uint32_t get_maskTailCoreLoop() const
    {
        return maskTailCoreLoop;
    }
    void set_maskTailCoreLoop(uint32_t maskTailCoreLoop_val)
    {
        this->maskTailCoreLoop = maskTailCoreLoop_val;
    }

    uint32_t get_maskTailCoreLastLoopNum() const
    {
        return maskTailCoreLastLoopNum;
    }
    void set_maskTailCoreLastLoopNum(uint32_t maskTailCoreLastLoopNum_val)
    {
        this->maskTailCoreLastLoopNum = maskTailCoreLastLoopNum_val;
    }

    uint32_t get_dropoutIsDivisibleBy8() const
    {
        return dropoutIsDivisibleBy8;
    }
    void set_dropoutIsDivisibleBy8(uint32_t dropoutIsDivisibleBy8_val)
    {
        this->dropoutIsDivisibleBy8 = dropoutIsDivisibleBy8_val;
    }

    uint64_t get_dropBeginAddr() const
    {
        return dropBeginAddr;
    }
    void set_dropBeginAddr(uint64_t dropBeginAddr_val)
    {
        this->dropBeginAddr = dropBeginAddr_val;
    }

    uint32_t get_reserved() const
    {
        return reserved;
    }
    void set_reserved(uint32_t reserved_val)
    {
        this->reserved = reserved_val;
    }
    void reset()
    {
        maskPreBlockTotal = 0;
        maskSingleCoreNum = 0;
        qPreBlockFactor = 0;
        qPreBlockTotal = 0;
        qPreBlockTail = 0;
        qRopePreBlockFactor = 0;
        qRopePreBlockTotal = 0;
        qRopePreBlockTail = 0;
        kvPreBlockFactor = 0;
        kvPreBlockTotal = 0;
        kvPreBlockTail = 0;
        kRopePreBlockFactor = 0;
        kRopePreBlockTotal = 0;
        kRopePreBlockTail = 0;
        vPreBlockFactor = 0;
        vPreBlockTotal = 0;
        vPreBlockTail = 0;
        maskCoreNum = 0;
        castBufferLen = 0;
        outputBufferLen = 0;
        inputBufferLen = 0;
        singleUBProcessNum = 0;
        maskSingleCoreLoop = 0;
        maskLastLoopNum = 0;
        maskTailCoreLoop = 0;
        maskTailCoreLastLoopNum = 0;
        dropoutIsDivisibleBy8 = 0;
        dropBeginAddr = 0;
        reserved = 0;
    }
};


class PreSfmgParams {
public:
    uint32_t usedCoreNum = 0;
    uint32_t inputBufferLen = 0;
    uint32_t castBufferLen = 0;
    uint32_t outputBufferLen = 0;
    uint32_t tempBufferLen = 0;
    uint32_t rsv1 = 0;
    int64_t singleLoopNBurstNum = 0;
    int64_t normalCoreLoopTimes = 0;
    int64_t tailCoreLoopTimes = 0;
    int64_t normalCoreLastLoopNBurstNum = 0;
    int64_t tailCoreLastLoopNBurstNum = 0;
    int64_t normalCoreNBurstNums = 0;
    int64_t sfmgPreBeginAddr = 0;
    int64_t b = 0;
    int64_t n2 = 0;
    int64_t g = 0;
    int64_t s1 = 0;
    int64_t d = 0;
    int64_t rope_d = 0;
    int64_t value_d = 0;

    uint32_t get_usedCoreNum() const
    {
        return usedCoreNum;
    }
    void set_usedCoreNum(uint32_t usedCoreNum_val)
    {
        this->usedCoreNum = usedCoreNum_val;
    }

    uint32_t get_inputBufferLen() const
    {
        return inputBufferLen;
    }
    void set_inputBufferLen(uint32_t inputBufferLen_val)
    {
        this->inputBufferLen = inputBufferLen_val;
    }

    uint32_t get_castBufferLen() const
    {
        return castBufferLen;
    }
    void set_castBufferLen(uint32_t castBufferLen_val)
    {
        this->castBufferLen = castBufferLen_val;
    }

    uint32_t get_outputBufferLen() const
    {
        return outputBufferLen;
    }
    void set_outputBufferLen(uint32_t outputBufferLen_val)
    {
        this->outputBufferLen = outputBufferLen_val;
    }

    uint32_t get_tempBufferLen() const
    {
        return tempBufferLen;
    }
    void set_tempBufferLen(uint32_t tempBufferLen_val)
    {
        this->tempBufferLen = tempBufferLen_val;
    }

    uint32_t get_rsv1() const
    {
        return rsv1;
    }
    void set_rsv1(uint32_t rsv1_val)
    {
        this->rsv1 = rsv1_val;
    }

    int64_t get_singleLoopNBurstNum() const
    {
        return singleLoopNBurstNum;
    }
    void set_singleLoopNBurstNum(int64_t singleLoopNBurstNum_val)
    {
        this->singleLoopNBurstNum = singleLoopNBurstNum_val;
    }

    int64_t get_normalCoreLoopTimes() const
    {
        return normalCoreLoopTimes;
    }
    void set_normalCoreLoopTimes(int64_t normalCoreLoopTimes_val)
    {
        this->normalCoreLoopTimes = normalCoreLoopTimes_val;
    }

    int64_t get_tailCoreLoopTimes() const
    {
        return tailCoreLoopTimes;
    }
    void set_tailCoreLoopTimes(int64_t tailCoreLoopTimes_val)
    {
        this->tailCoreLoopTimes = tailCoreLoopTimes_val;
    }

    int64_t get_normalCoreLastLoopNBurstNum() const
    {
        return normalCoreLastLoopNBurstNum;
    }
    void set_normalCoreLastLoopNBurstNum(int64_t normalCoreLastLoopNBurstNum_val)
    {
        this->normalCoreLastLoopNBurstNum = normalCoreLastLoopNBurstNum_val;
    }

    int64_t get_tailCoreLastLoopNBurstNum() const
    {
        return tailCoreLastLoopNBurstNum;
    }
    void set_tailCoreLastLoopNBurstNum(int64_t tailCoreLastLoopNBurstNum_val)
    {
        this->tailCoreLastLoopNBurstNum = tailCoreLastLoopNBurstNum_val;
    }

    int64_t get_normalCoreNBurstNums() const
    {
        return normalCoreNBurstNums;
    }
    void set_normalCoreNBurstNums(int64_t normalCoreNBurstNums_val)
    {
        this->normalCoreNBurstNums = normalCoreNBurstNums_val;
    }

    int64_t get_sfmgPreBeginAddr() const
    {
        return sfmgPreBeginAddr;
    }
    void set_sfmgPreBeginAddr(int64_t sfmgPreBeginAddr_val)
    {
        this->sfmgPreBeginAddr = sfmgPreBeginAddr_val;
    }

    int64_t get_b() const
    {
        return b;
    }
    void set_b(int64_t b_val)
    {
        this->b = b_val;
    }

    int64_t get_n2() const
    {
        return n2;
    }
    void set_n2(int64_t n2_val)
    {
        this->n2 = n2_val;
    }

    int64_t get_g() const
    {
        return g;
    }
    void set_g(int64_t g_val)
    {
        this->g = g_val;
    }

    int64_t get_s1() const
    {
        return s1;
    }
    void set_s1(int64_t s1_val)
    {
        this->s1 = s1_val;
    }

    int64_t get_d() const
    {
        return d;
    }
    void set_d(int64_t d_val)
    {
        this->d = d_val;
    }

    int64_t get_rope_d() const
    {
        return rope_d;
    }
    void set_rope_d(int64_t rope_d_val)
    {
        this->rope_d = rope_d_val;
    }

    int64_t get_value_d() const
    {
        return value_d;
    }
    void set_value_d(int64_t value_d_val)
    {
        this->value_d = value_d_val;
    }

    void reset()
    {
        usedCoreNum = 0;
        inputBufferLen = 0;
        castBufferLen = 0;
        outputBufferLen = 0;
        tempBufferLen = 0;
        rsv1 = 0;
        singleLoopNBurstNum = 0;
        normalCoreLoopTimes = 0;
        tailCoreLoopTimes = 0;
        normalCoreLastLoopNBurstNum = 0;
        tailCoreLastLoopNBurstNum = 0;
        normalCoreNBurstNums = 0;
        sfmgPreBeginAddr = 0;
        b = 0;
        n2 = 0;
        g = 0;
        s1 = 0;
        d = 0;
        rope_d = 0;
        value_d = 0;
    }
};

class PostParams {
public:
    uint32_t coreNum = 0;
    float scaleValue = 0;
    uint32_t postUbBaseSize = 0;
    uint32_t postMaxDataSize = 0;
    uint32_t nzReservedSize = 0;
    uint32_t qPostBlockFactor = 0;
    uint64_t qPostBlockTotal = 0;
    uint32_t qPostBaseNum = 0;
    uint32_t qPostTailNum = 0;
    uint32_t qRopePostBlockFactor = 0;
    uint8_t qRopePostBlockTotalPH[4] = {};
    uint64_t qRopePostBlockTotal = 0;
    uint32_t qRopePostBaseNum = 0;
    uint32_t qRopePostTailNum = 0;
    uint32_t kvPostBlockFactor = 0;
    uint32_t kvPostBlockTotal = 0;
    uint32_t kvPostBaseNum = 0;
    uint32_t kvPostTailNum = 0;
    uint32_t kRopePostBlockFactor = 0;
    uint32_t kRopePostBlockTotal = 0;
    uint32_t kRopePostBaseNum = 0;
    uint32_t kRopePostTailNum = 0;
    uint32_t vPostBlockFactor = 0;
    uint32_t vPostBlockTotal = 0;
    uint32_t vPostBaseNum = 0;
    uint32_t vPostTailNum = 0;
    uint64_t qSizeAlign = 0;
    uint64_t kvSizeAlign = 0;
    uint64_t qRopeSizeAlign = 0;
    uint64_t kRopeSizeAlign = 0;
    uint64_t vSizeAlign = 0;
    uint64_t dqWorkSpaceOffset = 0;
    uint64_t dqRopeWorkSpaceOffset = 0;
    uint64_t dkWorkSpaceOffset = 0;
    uint64_t dkRopeWorkSpaceOffset = 0;
    uint64_t dvWorkSpaceOffset = 0;
    int64_t b = 0;
    int64_t n2 = 0;
    int64_t g = 0;
    int64_t s1 = 0;
    int64_t s2 = 0;
    int64_t d = 0;
    int64_t rope_d = 0;
    int64_t value_d = 0;
    uint64_t dsinksumWorkSpaceOffset = 0;
    uint64_t dsinksumDataSizeOffset = 0;
    uint32_t baseMN = 0;
    uint32_t sinkDataSize = 0;

    uint32_t get_coreNum() const
    {
        return coreNum;
    }
    void set_coreNum(uint32_t coreNum_val)
    {
        this->coreNum = coreNum_val;
    }

    float get_scaleValue() const
    {
        return scaleValue;
    }
    void set_scaleValue(float scaleValue_val)
    {
        this->scaleValue = scaleValue_val;
    }

    uint32_t get_postUbBaseSize() const
    {
        return postUbBaseSize;
    }
    void set_postUbBaseSize(uint32_t postUbBaseSize_val)
    {
        this->postUbBaseSize = postUbBaseSize_val;
    }

    uint32_t get_postMaxDataSize() const
    {
        return postMaxDataSize;
    }
    void set_postMaxDataSize(uint32_t postMaxDataSize_val)
    {
        this->postMaxDataSize = postMaxDataSize_val;
    }

    uint32_t get_nzReservedSize() const
    {
        return nzReservedSize;
    }
    void set_nzReservedSize(uint32_t nzReservedSize_val)
    {
        this->nzReservedSize = nzReservedSize_val;
    }

    uint32_t get_qPostBlockFactor() const
    {
        return qPostBlockFactor;
    }
    void set_qPostBlockFactor(uint32_t qPostBlockFactor_val)
    {
        this->qPostBlockFactor = qPostBlockFactor_val;
    }

    uint64_t get_qPostBlockTotal() const
    {
        return qPostBlockTotal;
    }
    void set_qPostBlockTotal(uint64_t qPostBlockTotal_val)
    {
        this->qPostBlockTotal = qPostBlockTotal_val;
    }

    uint32_t get_qPostBaseNum() const
    {
        return qPostBaseNum;
    }
    void set_qPostBaseNum(uint32_t qPostBaseNum_val)
    {
        this->qPostBaseNum = qPostBaseNum_val;
    }

    uint32_t get_qPostTailNum() const
    {
        return qPostTailNum;
    }
    void set_qPostTailNum(uint32_t qPostTailNum_val)
    {
        this->qPostTailNum = qPostTailNum_val;
    }

    uint32_t get_qRopePostBlockFactor() const
    {
        return qRopePostBlockFactor;
    }
    void set_qRopePostBlockFactor(uint32_t qRopePostBlockFactor_val)
    {
        this->qRopePostBlockFactor = qRopePostBlockFactor_val;
    }

    uint64_t get_qRopePostBlockTotal() const
    {
        return qRopePostBlockTotal;
    }
    void set_qRopePostBlockTotal(uint64_t qRopePostBlockTotal_val)
    {
        this->qRopePostBlockTotal = qRopePostBlockTotal_val;
    }

    uint32_t get_qRopePostBaseNum() const
    {
        return qRopePostBaseNum;
    }
    void set_qRopePostBaseNum(uint32_t qRopePostBaseNum_val)
    {
        this->qRopePostBaseNum = qRopePostBaseNum_val;
    }

    uint32_t get_qRopePostTailNum() const
    {
        return qRopePostTailNum;
    }
    void set_qRopePostTailNum(uint32_t qRopePostTailNum_val)
    {
        this->qRopePostTailNum = qRopePostTailNum_val;
    }

    uint32_t get_kvPostBlockFactor() const
    {
        return kvPostBlockFactor;
    }
    void set_kvPostBlockFactor(uint32_t kvPostBlockFactor_val)
    {
        this->kvPostBlockFactor = kvPostBlockFactor_val;
    }

    uint32_t get_kvPostBlockTotal() const
    {
        return kvPostBlockTotal;
    }
    void set_kvPostBlockTotal(uint32_t kvPostBlockTotal_val)
    {
        this->kvPostBlockTotal = kvPostBlockTotal_val;
    }

    uint32_t get_kvPostBaseNum() const
    {
        return kvPostBaseNum;
    }
    void set_kvPostBaseNum(uint32_t kvPostBaseNum_val)
    {
        this->kvPostBaseNum = kvPostBaseNum_val;
    }

    uint32_t get_kvPostTailNum() const
    {
        return kvPostTailNum;
    }
    void set_kvPostTailNum(uint32_t kvPostTailNum_val)
    {
        this->kvPostTailNum = kvPostTailNum_val;
    }

    uint32_t get_kRopePostBlockFactor() const
    {
        return kRopePostBlockFactor;
    }
    void set_kRopePostBlockFactor(uint32_t kRopePostBlockFactor_val)
    {
        this->kRopePostBlockFactor = kRopePostBlockFactor_val;
    }

    uint32_t get_kRopePostBlockTotal() const
    {
        return kRopePostBlockTotal;
    }
    void set_kRopePostBlockTotal(uint32_t kRopePostBlockTotal_val)
    {
        this->kRopePostBlockTotal = kRopePostBlockTotal_val;
    }

    uint32_t get_kRopePostBaseNum() const
    {
        return kRopePostBaseNum;
    }
    void set_kRopePostBaseNum(uint32_t kRopePostBaseNum_val)
    {
        this->kRopePostBaseNum = kRopePostBaseNum_val;
    }

    uint32_t get_kRopePostTailNum() const
    {
        return kRopePostTailNum;
    }
    void set_kRopePostTailNum(uint32_t kRopePostTailNum_val)
    {
        this->kRopePostTailNum = kRopePostTailNum_val;
    }

    uint32_t get_vPostBlockFactor() const
    {
        return vPostBlockFactor;
    }
    void set_vPostBlockFactor(uint32_t vPostBlockFactor_val)
    {
        this->vPostBlockFactor = vPostBlockFactor_val;
    }

    uint32_t get_vPostBlockTotal() const
    {
        return vPostBlockTotal;
    }
    void set_vPostBlockTotal(uint32_t vPostBlockTotal_val)
    {
        this->vPostBlockTotal = vPostBlockTotal_val;
    }

    uint32_t get_vPostBaseNum() const
    {
        return vPostBaseNum;
    }
    void set_vPostBaseNum(uint32_t vPostBaseNum_val)
    {
        this->vPostBaseNum = vPostBaseNum_val;
    }

    uint32_t get_vPostTailNum() const
    {
        return vPostTailNum;
    }
    void set_vPostTailNum(uint32_t vPostTailNum_val)
    {
        this->vPostTailNum = vPostTailNum_val;
    }

    uint64_t get_qSizeAlign() const
    {
        return qSizeAlign;
    }
    void set_qSizeAlign(uint64_t qSizeAlign_val)
    {
        this->qSizeAlign = qSizeAlign_val;
    }

    uint64_t get_kvSizeAlign() const
    {
        return kvSizeAlign;
    }
    void set_kvSizeAlign(uint64_t kvSizeAlign_val)
    {
        this->kvSizeAlign = kvSizeAlign_val;
    }

    uint64_t get_qRopeSizeAlign() const
    {
        return qRopeSizeAlign;
    }
    void set_qRopeSizeAlign(uint64_t qRopeSizeAlign_val)
    {
        this->qRopeSizeAlign = qRopeSizeAlign_val;
    }

    uint64_t get_kRopeSizeAlign() const
    {
        return kRopeSizeAlign;
    }
    void set_kRopeSizeAlign(uint64_t kRopeSizeAlign_val)
    {
        this->kRopeSizeAlign = kRopeSizeAlign_val;
    }

    uint64_t get_vSizeAlign() const
    {
        return vSizeAlign;
    }
    void set_vSizeAlign(uint64_t vSizeAlign_val)
    {
        this->vSizeAlign = vSizeAlign_val;
    }

    uint64_t get_dqWorkSpaceOffset() const
    {
        return dqWorkSpaceOffset;
    }
    void set_dqWorkSpaceOffset(uint64_t dqWorkSpaceOffset_val)
    {
        this->dqWorkSpaceOffset = dqWorkSpaceOffset_val;
    }

    uint64_t get_dqRopeWorkSpaceOffset() const
    {
        return dqRopeWorkSpaceOffset;
    }
    void set_dqRopeWorkSpaceOffset(uint64_t dqRopeWorkSpaceOffset_val)
    {
        this->dqRopeWorkSpaceOffset = dqRopeWorkSpaceOffset_val;
    }

    uint64_t get_dkWorkSpaceOffset() const
    {
        return dkWorkSpaceOffset;
    }
    void set_dkWorkSpaceOffset(uint64_t dkWorkSpaceOffset_val)
    {
        this->dkWorkSpaceOffset = dkWorkSpaceOffset_val;
    }

    uint64_t get_dkRopeWorkSpaceOffset() const
    {
        return dkRopeWorkSpaceOffset;
    }
    void set_dkRopeWorkSpaceOffset(uint64_t dkRopeWorkSpaceOffset_val)
    {
        this->dkRopeWorkSpaceOffset = dkRopeWorkSpaceOffset_val;
    }

    uint64_t get_dvWorkSpaceOffset() const
    {
        return dvWorkSpaceOffset;
    }
    void set_dvWorkSpaceOffset(uint64_t dvWorkSpaceOffset_val)
    {
        this->dvWorkSpaceOffset = dvWorkSpaceOffset_val;
    }

    int64_t get_b() const
    {
        return b;
    }
    void set_b(int64_t b_val)
    {
        this->b = b_val;
    }

    int64_t get_n2() const
    {
        return n2;
    }
    void set_n2(int64_t n2_val)
    {
        this->n2 = n2_val;
    }

    int64_t get_g() const
    {
        return g;
    }
    void set_g(int64_t g_val)
    {
        this->g = g_val;
    }

    int64_t get_s1() const
    {
        return s1;
    }
    void set_s1(int64_t s1_val)
    {
        this->s1 = s1_val;
    }

    int64_t get_s2() const
    {
        return s2;
    }
    void set_s2(int64_t s2_val)
    {
        this->s2 = s2_val;
    }

    int64_t get_d() const
    {
        return d;
    }
    void set_d(int64_t d_val)
    {
        this->d = d_val;
    }

    int64_t get_rope_d() const
    {
        return rope_d;
    }
    void set_rope_d(int64_t rope_d_val)
    {
        this->rope_d = rope_d_val;
    }

    int64_t get_value_d() const
    {
        return value_d;
    }
    void set_value_d(int64_t value_d_val)
    {
        this->value_d = value_d_val;
    }

    uint64_t get_dsinksumWorkSpaceOffset() const
    {
        return this->dsinksumWorkSpaceOffset;
    }
    void set_dsinksumWorkSpaceOffset(uint64_t dsinksumWorkSpaceOffset_val)
    {
        this->dsinksumWorkSpaceOffset = dsinksumWorkSpaceOffset_val;
    }
    uint64_t get_dsinksumDataSizeOffset() const
    {
        return this->dsinksumDataSizeOffset;
    }
    void set_dsinksumDataSizeOffset(uint64_t dsinksumDataSizeOffset_val)
    {
        this->dsinksumDataSizeOffset = dsinksumDataSizeOffset_val;
    }
    uint32_t get_baseMN() const
    {
        return this->baseMN;
    }
    void set_baseMN(uint32_t baseMN_val)
    {
        this->baseMN = baseMN_val;
    }

    uint32_t get_sinkDataSize() const
    {
        return this->sinkDataSize;
    }
    void set_sinkDataSize(uint32_t sinkDataSize_val)
    {
        this->sinkDataSize = sinkDataSize_val;
    }

    void reset()
    {
        coreNum = 0;
        scaleValue = 0.0f;
        postUbBaseSize = 0;
        postMaxDataSize = 0;
        nzReservedSize = 0;
        qPostBlockFactor = 0;
        qPostBlockTotal = 0;
        qPostBaseNum = 0;
        qPostTailNum = 0;
        qRopePostBlockFactor = 0;
        qRopePostBlockTotal = 0;
        qRopePostBaseNum = 0;
        qRopePostTailNum = 0;
        kvPostBlockFactor = 0;
        kvPostBlockTotal = 0;
        kvPostBaseNum = 0;
        kvPostTailNum = 0;
        kRopePostBlockFactor = 0;
        kRopePostBlockTotal = 0;
        kRopePostBaseNum = 0;
        kRopePostTailNum = 0;
        vPostBlockFactor = 0;
        vPostBlockTotal = 0;
        vPostBaseNum = 0;
        vPostTailNum = 0;
        qSizeAlign = 0;
        kvSizeAlign = 0;
        qRopeSizeAlign = 0;
        kRopeSizeAlign = 0;
        vSizeAlign = 0;
        dqWorkSpaceOffset = 0;
        dqRopeWorkSpaceOffset = 0;
        dkWorkSpaceOffset = 0;
        dkRopeWorkSpaceOffset = 0;
        dvWorkSpaceOffset = 0;
        b = 0;
        n2 = 0;
        g = 0;
        s1 = 0;
        s2 = 0;
        d = 0;
        rope_d = 0;
        value_d = 0;
        dsinksumWorkSpaceOffset = 0;
        dsinksumDataSizeOffset = 0;
        baseMN = 0;
        sinkDataSize = 0;
    }
};

class FlashAttentionScoreGradS1S2BNGS1S2BaseParams {
public:
    int64_t b = 0;
    int64_t n2 = 0;
    int64_t g = 0;
    int64_t s1 = 0;
    int64_t s2 = 0;
    int64_t d = 0;
    int64_t rope_d = 0;
    int64_t value_d = 0;
    float scaleValue = 0;
    float keepProb = 0;
    int64_t s1Token = 0;
    int64_t s2Token = 0;
    uint32_t sparseMode = 0;
    uint32_t isSparse = 0;
    int64_t attenMaskS2Size = 0;
    uint32_t coreNum = 0;
    uint32_t attenMaskCompressMode = 0;
    int64_t qStartIdx = 0;
    int64_t kvStartIdx = 0;
    int64_t pseAlibiBaseS1 = 0;
    int64_t pseAlibiBaseS2 = 0;
    uint32_t pseType = 0;
    uint32_t pseOptional = 0;
    uint32_t pseShapeType = 0;
    uint32_t pseDtype = 0;
    uint32_t attenMaskOptional = 0;
    uint32_t attenMaskDtype = 0;
    uint32_t attenMaskShapeType = 0;
    uint32_t pad = 0;
    uint8_t tndSoftmaxIn = 0;
    uint8_t FlashAttentionScoreGradS1S2BNGS1S2BaseParamsPH[7] = {};

    int64_t get_b() const
    {
        return b;
    }
    void set_b(int64_t b_val)
    {
        this->b = b_val;
    }

    int64_t get_n2() const
    {
        return n2;
    }
    void set_n2(int64_t n2_val)
    {
        this->n2 = n2_val;
    }

    int64_t get_g() const
    {
        return g;
    }
    void set_g(int64_t g_val)
    {
        this->g = g_val;
    }

    int64_t get_s1() const
    {
        return s1;
    }
    void set_s1(int64_t s1_val)
    {
        this->s1 = s1_val;
    }

    int64_t get_s2() const
    {
        return s2;
    }
    void set_s2(int64_t s2_val)
    {
        this->s2 = s2_val;
    }

    int64_t get_d() const
    {
        return d;
    }
    void set_d(int64_t d_val)
    {
        this->d = d_val;
    }

    int64_t get_rope_d() const
    {
        return rope_d;
    }
    void set_rope_d(int64_t rope_d_val)
    {
        this->rope_d = rope_d_val;
    }

    int64_t get_value_d() const
    {
        return value_d;
    }
    void set_value_d(int64_t value_d_val)
    {
        this->value_d = value_d_val;
    }

    float get_scaleValue() const
    {
        return scaleValue;
    }
    void set_scaleValue(float scaleValue_val)
    {
        this->scaleValue = scaleValue_val;
    }

    float get_keepProb() const
    {
        return keepProb;
    }
    void set_keepProb(float keepProb_val)
    {
        this->keepProb = keepProb_val;
    }

    int64_t get_s1Token() const
    {
        return s1Token;
    }
    void set_s1Token(int64_t s1Token_val)
    {
        this->s1Token = s1Token_val;
    }

    int64_t get_s2Token() const
    {
        return s2Token;
    }
    void set_s2Token(int64_t s2Token_val)
    {
        this->s2Token = s2Token_val;
    }

    uint32_t get_sparseMode() const
    {
        return sparseMode;
    }
    void set_sparseMode(uint32_t sparseMode_val)
    {
        this->sparseMode = sparseMode_val;
    }

    uint32_t get_isSparse() const
    {
        return isSparse;
    }
    void set_isSparse(uint32_t isSparse_val)
    {
        this->isSparse = isSparse_val;
    }

    int64_t get_attenMaskS2Size() const
    {
        return attenMaskS2Size;
    }
    void set_attenMaskS2Size(int64_t attenMaskS2Size_val)
    {
        this->attenMaskS2Size = attenMaskS2Size_val;
    }

    uint32_t get_coreNum() const
    {
        return coreNum;
    }
    void set_coreNum(uint32_t coreNum_val)
    {
        this->coreNum = coreNum_val;
    }

    uint32_t get_attenMaskCompressMode() const
    {
        return attenMaskCompressMode;
    }
    void set_attenMaskCompressMode(uint32_t attenMaskCompressMode_val)
    {
        this->attenMaskCompressMode = attenMaskCompressMode_val;
    }

    int64_t get_qStartIdx() const
    {
        return qStartIdx;
    }
    void set_qStartIdx(int64_t qStartIdx_val)
    {
        this->qStartIdx = qStartIdx_val;
    }

    int64_t get_kvStartIdx() const
    {
        return kvStartIdx;
    }
    void set_kvStartIdx(int64_t kvStartIdx_val)
    {
        this->kvStartIdx = kvStartIdx_val;
    }

    int64_t get_pseAlibiBaseS1() const
    {
        return pseAlibiBaseS1;
    }
    void set_pseAlibiBaseS1(int64_t pseAlibiBaseS1_val)
    {
        this->pseAlibiBaseS1 = pseAlibiBaseS1_val;
    }

    int64_t get_pseAlibiBaseS2() const
    {
        return pseAlibiBaseS2;
    }
    void set_pseAlibiBaseS2(int64_t pseAlibiBaseS2_val)
    {
        this->pseAlibiBaseS2 = pseAlibiBaseS2_val;
    }

    uint32_t get_pseType() const
    {
        return pseType;
    }
    void set_pseType(uint32_t pseType_val)
    {
        this->pseType = pseType_val;
    }

    uint32_t get_pseOptional() const
    {
        return pseOptional;
    }
    void set_pseOptional(uint32_t pseOptional_val)
    {
        this->pseOptional = pseOptional_val;
    }

    uint32_t get_pseShapeType() const
    {
        return pseShapeType;
    }
    void set_pseShapeType(uint32_t pseShapeType_val)
    {
        this->pseShapeType = pseShapeType_val;
    }

    uint32_t get_pseDtype() const
    {
        return pseDtype;
    }
    void set_pseDtype(uint32_t pseDtype_val)
    {
        this->pseDtype = pseDtype_val;
    }

    uint32_t get_attenMaskOptional() const
    {
        return attenMaskOptional;
    }
    void set_attenMaskOptional(uint32_t attenMaskOptional_val)
    {
        this->attenMaskOptional = attenMaskOptional_val;
    }

    uint32_t get_attenMaskDtype() const
    {
        return attenMaskDtype;
    }
    void set_attenMaskDtype(uint32_t attenMaskDtype_val)
    {
        this->attenMaskDtype = attenMaskDtype_val;
    }

    uint32_t get_attenMaskShapeType() const
    {
        return attenMaskShapeType;
    }
    void set_attenMaskShapeType(uint32_t attenMaskShapeType_val)
    {
        this->attenMaskShapeType = attenMaskShapeType_val;
    }

    uint32_t get_pad() const
    {
        return pad;
    }
    void set_pad(uint32_t pad_val)
    {
        this->pad = pad_val;
    }

    uint8_t get_tndSoftmaxIn() const
    {
        return tndSoftmaxIn;
    }
    void set_tndSoftmaxIn(uint8_t tndSoftmaxIn_val)
    {
        this->tndSoftmaxIn = tndSoftmaxIn_val;
    }
    void reset()
    {
        b = 0;
        n2 = 0;
        g = 0;
        s1 = 0;
        s2 = 0;
        d = 0;
        rope_d = 0;
        value_d = 0;
        scaleValue = 0.0f;
        keepProb = 0.0f;
        s1Token = 0;
        s2Token = 0;
        sparseMode = 0;
        isSparse = 0;
        attenMaskS2Size = 0;
        coreNum = 0;
        attenMaskCompressMode = 0;
        qStartIdx = 0;
        kvStartIdx = 0;
        pseAlibiBaseS1 = 0;
        pseAlibiBaseS2 = 0;
        pseType = 0;
        pseOptional = 0;
        pseShapeType = 0;
        pseDtype = 0;
        attenMaskOptional = 0;
        attenMaskDtype = 0;
        attenMaskShapeType = 0;
        pad = 0;
        tndSoftmaxIn = 0;
    }
};


class FlashAttentionScoreGradS1S2BNGS1S2SplitCoreParams {
public:
    int64_t s1Outer = 0;
    uint32_t s1CvRatio = 0;
    uint32_t s1Inner = 0;
    uint32_t s1CvInner = 0;
    uint32_t s1Tail = 0;
    uint32_t s1CvTail = 0;
    uint8_t s2OuterPH[4] = {};
    int64_t s2Outer = 0;
    uint32_t s2CvRatio = 0;
    uint32_t s2Inner = 0;
    uint32_t s2Tail = 0;
    uint32_t baseMN = 0;
    uint32_t sfmgdOuter = 0;
    uint32_t sfmgdFactor = 0;
    uint32_t sfmgdTail = 0;
    uint32_t blockOuter = 0;
    int64_t bandIdx = 0;

    int64_t get_s1Outer() const
    {
        return s1Outer;
    }
    void set_s1Outer(int64_t s1Outer_val)
    {
        this->s1Outer = s1Outer_val;
    }

    uint32_t get_s1CvRatio() const
    {
        return s1CvRatio;
    }
    void set_s1CvRatio(uint32_t s1CvRatio_val)
    {
        this->s1CvRatio = s1CvRatio_val;
    }

    uint32_t get_s1Inner() const
    {
        return s1Inner;
    }
    void set_s1Inner(uint32_t s1Inner_val)
    {
        this->s1Inner = s1Inner_val;
    }

    uint32_t get_s1CvInner() const
    {
        return s1CvInner;
    }
    void set_s1CvInner(uint32_t s1CvInner_val)
    {
        this->s1CvInner = s1CvInner_val;
    }

    uint32_t get_s1Tail() const
    {
        return s1Tail;
    }
    void set_s1Tail(uint32_t s1Tail_val)
    {
        this->s1Tail = s1Tail_val;
    }

    uint32_t get_s1CvTail() const
    {
        return s1CvTail;
    }
    void set_s1CvTail(uint32_t s1CvTail_val)
    {
        this->s1CvTail = s1CvTail_val;
    }

    int64_t get_s2Outer() const
    {
        return s2Outer;
    }
    void set_s2Outer(int64_t s2Outer_val)
    {
        this->s2Outer = s2Outer_val;
    }

    uint32_t get_s2CvRatio() const
    {
        return s2CvRatio;
    }
    void set_s2CvRatio(uint32_t s2CvRatio_val)
    {
        this->s2CvRatio = s2CvRatio_val;
    }

    uint32_t get_s2Inner() const
    {
        return s2Inner;
    }
    void set_s2Inner(uint32_t s2Inner_val)
    {
        this->s2Inner = s2Inner_val;
    }

    uint32_t get_s2Tail() const
    {
        return s2Tail;
    }
    void set_s2Tail(uint32_t s2Tail_val)
    {
        this->s2Tail = s2Tail_val;
    }

    uint32_t get_baseMN() const
    {
        return baseMN;
    }
    void set_baseMN(uint32_t baseMN_val)
    {
        this->baseMN = baseMN_val;
    }

    uint32_t get_sfmgdOuter() const
    {
        return sfmgdOuter;
    }
    void set_sfmgdOuter(uint32_t sfmgdOuter_val)
    {
        this->sfmgdOuter = sfmgdOuter_val;
    }

    uint32_t get_sfmgdFactor() const
    {
        return sfmgdFactor;
    }
    void set_sfmgdFactor(uint32_t sfmgdFactor_val)
    {
        this->sfmgdFactor = sfmgdFactor_val;
    }

    uint32_t get_sfmgdTail() const
    {
        return sfmgdTail;
    }
    void set_sfmgdTail(uint32_t sfmgdTail_val)
    {
        this->sfmgdTail = sfmgdTail_val;
    }

    uint32_t get_blockOuter() const
    {
        return blockOuter;
    }
    void set_blockOuter(uint32_t blockOuter_val)
    {
        this->blockOuter = blockOuter_val;
    }

    int64_t get_bandIdx() const
    {
        return bandIdx;
    }
    void set_bandIdx(int64_t bandIdx_val)
    {
        this->bandIdx = bandIdx_val;
    }

    void reset()
    {
        s1Outer = 0;
        s1CvRatio = 0;
        s1Inner = 0;
        s1CvInner = 0;
        s1Tail = 0;
        s1CvTail = 0;
        s2Outer = 0;
        s2CvRatio = 0;
        s2Inner = 0;
        s2Tail = 0;
        baseMN = 0;
        sfmgdOuter = 0;
        sfmgdFactor = 0;
        sfmgdTail = 0;
        blockOuter = 0;
        bandIdx = 0;
    }
};


class BlockNumListParams {
public:
    int64_t blockStarts[50] = {};
    int64_t blockEnds[50] = {};

    const int64_t *get_blockStarts() const
    {
        return blockStarts;
    }
    void set_blockStarts(int64_t blockStarts_val[50])
    {
        for (int i = 0; i < 50; i++) {
            this->blockStarts[i] = blockStarts_val[i];
        }
    }

    const int64_t *get_blockEnds() const
    {
        return blockEnds;
    }
    void set_blockEnds(int64_t blockEnds_val[50])
    {
        for (int i = 0; i < 50; i++) {
            this->blockEnds[i] = blockEnds_val[i];
        }
    }
    void reset()
    {
        for (int i = 0; i < 50; i++) {
            blockStarts[i] = 0;
            blockEnds[i] = 0;
        }
    }
};

class FlashAttentionScoreGradBaseParamsS1s2Bn2 {
public:
    uint32_t usedCoreNum = 0;
    uint32_t formerCoreNum = 0;
    uint32_t formerCoreProcessNNum = 0;
    uint32_t remainCoreProcessNNum = 0;
    int64_t B = 0;
    int64_t N2 = 0;
    int64_t S1 = 0;
    int64_t S2 = 0;
    int64_t G = 0;
    int64_t D = 0;
    float scaleValue = 0;
    float keepProb = 0;
    uint32_t layout = 0;
    uint8_t preTokensPH[4] = {};
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    uint32_t isSparse = 0;
    uint32_t maskDataType = 0;
    uint32_t maskShapeType = 0;
    uint32_t pseShapeType = 0;
    uint32_t pseType = 0;
    uint32_t unpadEmptyInput = 0;
    int64_t qStartIdx = 0;
    int64_t kvStartIdx = 0;
    int64_t pseAlibiBaseS1 = 0;
    int64_t pseAlibiBaseS2 = 0;
    uint32_t bandIdx = 0;
    uint32_t existAttenMask = 0;
    uint32_t attenMaskCompressMode = 0;
    uint32_t attenMaskS2Size = 0;
    uint32_t inputBufferLen = 0;
    uint32_t helpBufferLen = 0;
    uint32_t mm1WorkspaceLen = 0;
    uint32_t mm2WorkspaceLen = 0;
    uint32_t mm4InputWorkspaceLen = 0;
    uint32_t mm3InputWorkspaceLen = 0;
    uint32_t sparseMode = 0;
    uint32_t castUsedCoreNum = 0;
    int64_t dqWorkspaceLen = 0;
    int64_t dkWorkspaceLen = 0;
    int64_t dvWorkspaceLen = 0;
    int64_t dropoutWorkspaceLen = 0;

    uint32_t get_usedCoreNum() const
    {
        return usedCoreNum;
    }
    void set_usedCoreNum(uint32_t usedCoreNum_val)
    {
        this->usedCoreNum = usedCoreNum_val;
    }

    uint32_t get_formerCoreNum() const
    {
        return formerCoreNum;
    }
    void set_formerCoreNum(uint32_t formerCoreNum_val)
    {
        this->formerCoreNum = formerCoreNum_val;
    }

    uint32_t get_formerCoreProcessNNum() const
    {
        return formerCoreProcessNNum;
    }
    void set_formerCoreProcessNNum(uint32_t formerCoreProcessNNum_val)
    {
        this->formerCoreProcessNNum = formerCoreProcessNNum_val;
    }

    uint32_t get_remainCoreProcessNNum() const
    {
        return remainCoreProcessNNum;
    }
    void set_remainCoreProcessNNum(uint32_t remainCoreProcessNNum_val)
    {
        this->remainCoreProcessNNum = remainCoreProcessNNum_val;
    }

    int64_t get_B() const
    {
        return B;
    }
    void set_B(int64_t B_val)
    {
        this->B = B_val;
    }

    int64_t get_N2() const
    {
        return N2;
    }
    void set_N2(int64_t N2_val)
    {
        this->N2 = N2_val;
    }

    int64_t get_S1() const
    {
        return S1;
    }
    void set_S1(int64_t S1_val)
    {
        this->S1 = S1_val;
    }

    int64_t get_S2() const
    {
        return S2;
    }
    void set_S2(int64_t S2_val)
    {
        this->S2 = S2_val;
    }

    int64_t get_G() const
    {
        return G;
    }
    void set_G(int64_t G_val)
    {
        this->G = G_val;
    }

    int64_t get_D() const
    {
        return D;
    }
    void set_D(int64_t D_val)
    {
        this->D = D_val;
    }

    float get_scaleValue() const
    {
        return scaleValue;
    }
    void set_scaleValue(float scaleValue_val)
    {
        this->scaleValue = scaleValue_val;
    }

    float get_keepProb() const
    {
        return keepProb;
    }
    void set_keepProb(float keepProb_val)
    {
        this->keepProb = keepProb_val;
    }

    uint32_t get_layout() const
    {
        return layout;
    }
    void set_layout(uint32_t layout_val)
    {
        this->layout = layout_val;
    }

    int64_t get_preTokens() const
    {
        return preTokens;
    }
    void set_preTokens(int64_t preTokens_val)
    {
        this->preTokens = preTokens_val;
    }

    int64_t get_nextTokens() const
    {
        return nextTokens;
    }
    void set_nextTokens(int64_t nextTokens_val)
    {
        this->nextTokens = nextTokens_val;
    }

    uint32_t get_isSparse() const
    {
        return isSparse;
    }
    void set_isSparse(uint32_t isSparse_val)
    {
        this->isSparse = isSparse_val;
    }

    uint32_t get_maskDataType() const
    {
        return maskDataType;
    }
    void set_maskDataType(uint32_t maskDataType_val)
    {
        this->maskDataType = maskDataType_val;
    }

    uint32_t get_maskShapeType() const
    {
        return maskShapeType;
    }
    void set_maskShapeType(uint32_t maskShapeType_val)
    {
        this->maskShapeType = maskShapeType_val;
    }

    uint32_t get_pseShapeType() const
    {
        return pseShapeType;
    }
    void set_pseShapeType(uint32_t pseShapeType_val)
    {
        this->pseShapeType = pseShapeType_val;
    }

    uint32_t get_pseType() const
    {
        return pseType;
    }
    void set_pseType(uint32_t pseType_val)
    {
        this->pseType = pseType_val;
    }

    uint32_t get_unpadEmptyInput() const
    {
        return unpadEmptyInput;
    }
    void set_unpadEmptyInput(uint32_t unpadEmptyInput_val)
    {
        this->unpadEmptyInput = unpadEmptyInput_val;
    }

    int64_t get_qStartIdx() const
    {
        return qStartIdx;
    }
    void set_qStartIdx(int64_t qStartIdx_val)
    {
        this->qStartIdx = qStartIdx_val;
    }

    int64_t get_kvStartIdx() const
    {
        return kvStartIdx;
    }
    void set_kvStartIdx(int64_t kvStartIdx_val)
    {
        this->kvStartIdx = kvStartIdx_val;
    }

    int64_t get_pseAlibiBaseS1() const
    {
        return pseAlibiBaseS1;
    }
    void set_pseAlibiBaseS1(int64_t pseAlibiBaseS1_val)
    {
        this->pseAlibiBaseS1 = pseAlibiBaseS1_val;
    }

    int64_t get_pseAlibiBaseS2() const
    {
        return pseAlibiBaseS2;
    }
    void set_pseAlibiBaseS2(int64_t pseAlibiBaseS2_val)
    {
        this->pseAlibiBaseS2 = pseAlibiBaseS2_val;
    }

    uint32_t get_bandIdx() const
    {
        return bandIdx;
    }
    void set_bandIdx(uint32_t bandIdx_val)
    {
        this->bandIdx = bandIdx_val;
    }

    uint32_t get_existAttenMask() const
    {
        return existAttenMask;
    }
    void set_existAttenMask(uint32_t existAttenMask_val)
    {
        this->existAttenMask = existAttenMask_val;
    }

    uint32_t get_attenMaskCompressMode() const
    {
        return attenMaskCompressMode;
    }
    void set_attenMaskCompressMode(uint32_t attenMaskCompressMode_val)
    {
        this->attenMaskCompressMode = attenMaskCompressMode_val;
    }

    uint32_t get_attenMaskS2Size() const
    {
        return attenMaskS2Size;
    }
    void set_attenMaskS2Size(uint32_t attenMaskS2Size_val)
    {
        this->attenMaskS2Size = attenMaskS2Size_val;
    }

    uint32_t get_inputBufferLen() const
    {
        return inputBufferLen;
    }
    void set_inputBufferLen(uint32_t inputBufferLen_val)
    {
        this->inputBufferLen = inputBufferLen_val;
    }

    uint32_t get_helpBufferLen() const
    {
        return helpBufferLen;
    }
    void set_helpBufferLen(uint32_t helpBufferLen_val)
    {
        this->helpBufferLen = helpBufferLen_val;
    }

    uint32_t get_mm1WorkspaceLen() const
    {
        return mm1WorkspaceLen;
    }
    void set_mm1WorkspaceLen(uint32_t mm1WorkspaceLen_val)
    {
        this->mm1WorkspaceLen = mm1WorkspaceLen_val;
    }

    uint32_t get_mm2WorkspaceLen() const
    {
        return mm2WorkspaceLen;
    }
    void set_mm2WorkspaceLen(uint32_t mm2WorkspaceLen_val)
    {
        this->mm2WorkspaceLen = mm2WorkspaceLen_val;
    }

    uint32_t get_mm4InputWorkspaceLen() const
    {
        return mm4InputWorkspaceLen;
    }
    void set_mm4InputWorkspaceLen(uint32_t mm4InputWorkspaceLen_val)
    {
        this->mm4InputWorkspaceLen = mm4InputWorkspaceLen_val;
    }

    uint32_t get_mm3InputWorkspaceLen() const
    {
        return mm3InputWorkspaceLen;
    }
    void set_mm3InputWorkspaceLen(uint32_t mm3InputWorkspaceLen_val)
    {
        this->mm3InputWorkspaceLen = mm3InputWorkspaceLen_val;
    }

    uint32_t get_sparseMode() const
    {
        return sparseMode;
    }
    void set_sparseMode(uint32_t sparseMode_val)
    {
        this->sparseMode = sparseMode_val;
    }

    uint32_t get_castUsedCoreNum() const
    {
        return castUsedCoreNum;
    }
    void set_castUsedCoreNum(uint32_t castUsedCoreNum_val)
    {
        this->castUsedCoreNum = castUsedCoreNum_val;
    }

    int64_t get_dqWorkspaceLen() const
    {
        return dqWorkspaceLen;
    }
    void set_dqWorkspaceLen(int64_t dqWorkspaceLen_val)
    {
        this->dqWorkspaceLen = dqWorkspaceLen_val;
    }

    int64_t get_dkWorkspaceLen() const
    {
        return dkWorkspaceLen;
    }
    void set_dkWorkspaceLen(int64_t dkWorkspaceLen_val)
    {
        this->dkWorkspaceLen = dkWorkspaceLen_val;
    }

    int64_t get_dvWorkspaceLen() const
    {
        return dvWorkspaceLen;
    }
    void set_dvWorkspaceLen(int64_t dvWorkspaceLen_val)
    {
        this->dvWorkspaceLen = dvWorkspaceLen_val;
    }

    int64_t get_dropoutWorkspaceLen() const
    {
        return dropoutWorkspaceLen;
    }
    void set_dropoutWorkspaceLen(int64_t dropoutWorkspaceLen_val)
    {
        this->dropoutWorkspaceLen = dropoutWorkspaceLen_val;
    }
    void reset()
    {
        usedCoreNum = 0;
        formerCoreNum = 0;
        formerCoreProcessNNum = 0;
        remainCoreProcessNNum = 0;
        B = 0;
        N2 = 0;
        S1 = 0;
        S2 = 0;
        G = 0;
        D = 0;
        scaleValue = 0.0f;
        keepProb = 0.0f;
        layout = 0;
        preTokens = 0;
        nextTokens = 0;
        isSparse = 0;
        maskDataType = 0;
        maskShapeType = 0;
        pseShapeType = 0;
        pseType = 0;
        unpadEmptyInput = 0;
        qStartIdx = 0;
        kvStartIdx = 0;
        pseAlibiBaseS1 = 0;
        pseAlibiBaseS2 = 0;
        bandIdx = 0;
        existAttenMask = 0;
        attenMaskCompressMode = 0;
        attenMaskS2Size = 0;
        inputBufferLen = 0;
        helpBufferLen = 0;
        mm1WorkspaceLen = 0;
        mm2WorkspaceLen = 0;
        mm4InputWorkspaceLen = 0;
        mm3InputWorkspaceLen = 0;
        sparseMode = 0;
        castUsedCoreNum = 0;
        dqWorkspaceLen = 0;
        dkWorkspaceLen = 0;
        dvWorkspaceLen = 0;
        dropoutWorkspaceLen = 0;
    }
};

class SplitCoreParamsS1s2Bn2 {
public:
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t singleN = 0;
    uint32_t singleM = 0;
    uint32_t s1OuterOuter = 0;
    uint32_t s2OuterOuter = 0;
    uint32_t dInner = 0;
    uint32_t SFTBaseM = 0;
    uint32_t SFTSingleM = 0;
    uint32_t placeholderX2 = 0;

    uint32_t get_baseM() const
    {
        return baseM;
    }
    void set_baseM(uint32_t baseM_val)
    {
        this->baseM = baseM_val;
    }

    uint32_t get_baseN() const
    {
        return baseN;
    }
    void set_baseN(uint32_t baseN_val)
    {
        this->baseN = baseN_val;
    }

    uint32_t get_singleN() const
    {
        return singleN;
    }
    void set_singleN(uint32_t singleN_val)
    {
        this->singleN = singleN_val;
    }

    uint32_t get_singleM() const
    {
        return singleM;
    }
    void set_singleM(uint32_t singleM_val)
    {
        this->singleM = singleM_val;
    }

    uint32_t get_s1OuterOuter() const
    {
        return s1OuterOuter;
    }
    void set_s1OuterOuter(uint32_t s1OuterOuter_val)
    {
        this->s1OuterOuter = s1OuterOuter_val;
    }

    uint32_t get_s2OuterOuter() const
    {
        return s2OuterOuter;
    }
    void set_s2OuterOuter(uint32_t s2OuterOuter_val)
    {
        this->s2OuterOuter = s2OuterOuter_val;
    }

    uint32_t get_dInner() const
    {
        return dInner;
    }
    void set_dInner(uint32_t dInner_val)
    {
        this->dInner = dInner_val;
    }

    uint32_t get_SFTBaseM() const
    {
        return SFTBaseM;
    }
    void set_SFTBaseM(uint32_t SFTBaseM_val)
    {
        this->SFTBaseM = SFTBaseM_val;
    }

    uint32_t get_SFTSingleM() const
    {
        return SFTSingleM;
    }
    void set_SFTSingleM(uint32_t SFTSingleM_val)
    {
        this->SFTSingleM = SFTSingleM_val;
    }

    uint32_t get_placeholderX2() const
    {
        return placeholderX2;
    }
    void set_placeholderX2(uint32_t placeholderX2_val)
    {
        this->placeholderX2 = placeholderX2_val;
    }
    void reset()
    {
        baseM = 0;
        baseN = 0;
        singleN = 0;
        singleM = 0;
        s1OuterOuter = 0;
        s2OuterOuter = 0;
        dInner = 0;
        SFTBaseM = 0;
        SFTSingleM = 0;
        placeholderX2 = 0;
    }
};

class DeterministicTndSplitCoreS1s2Bn2 {
public:
    int64_t bN2idxStarts[50] = {};
    int64_t bN2idxEnds[50] = {};

    const int64_t *get_bN2idxStarts() const
    {
        return bN2idxStarts;
    }
    void set_bN2idxStarts(int64_t bN2idxStarts_val[50])
    {
        for (int i = 0; i < 50; i++) {
            this->bN2idxStarts[i] = bN2idxStarts_val[i];
        }
    }

    const int64_t *get_bN2idxEnds() const
    {
        return bN2idxEnds;
    }
    void set_bN2idxEnds(int64_t bN2idxEnds_val[50])
    {
        for (int i = 0; i < 50; i++) {
            this->bN2idxEnds[i] = bN2idxEnds_val[i];
        }
    }
    void reset()
    {
        for (int i = 0; i < 50; i++) {
            bN2idxStarts[i] = 0;
            bN2idxEnds[i] = 0;
        }
    }
};

class FlashAttentionScoreGradShapeAttrParams1 {
public:
    int64_t b = 0;
    int64_t n = 0;
    int64_t g = 0;
    int64_t sQ = 0;
    int64_t sKV = 0;
    int64_t sKVAlign = 0;
    int64_t sKVAlignSize = 0;
    int64_t sKVAlignVec = 0;
    int64_t sKVAlignSizeVec = 0;
    int64_t sKVAlignByte = 0;
    int64_t hQ = 0;
    int64_t hKV = 0;
    int64_t d = 0;
    int64_t originalDAlign = 0;
    float scaleValue = 0;
    float keepProb = 0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    int64_t headNum = 0;
    uint32_t inputLayout = 0;
    uint8_t preTokensBlocksPH[4] = {};
    int64_t preTokensBlocks = 0;
    int64_t nextTokensBlocks = 0;
    uint32_t inputDType = 0;
    uint32_t inputDTypeSize = 0;
    uint32_t vecCalcDTypeSize = 0;
    uint32_t pseSq = 0;
    uint32_t pseShapeType = 0;
    uint32_t attenMaskShapeType = 0;
    uint32_t hasAttenMask = 0;
    uint32_t attenMaskCompressMode = 0;
    int64_t attenMaskS2Size = 0;
    uint32_t elementPerBlock = 0;
    uint32_t precisionMode = 0;
    uint32_t resv = 0;
    uint8_t mm1WorkspaceLenPH[4] = {};
    int64_t mm1WorkspaceLen = 0;
    int64_t mm2WorkspaceLen = 0;
    int64_t dqWorkspaceLen = 0;
    int64_t dkWorkspaceLen = 0;
    int64_t dropGmWorkspaceLen = 0;
    int64_t mulGmWorkspaceLen = 0;
    int64_t dropoutWorkspaceLen = 0;

    int64_t get_b() const
    {
        return b;
    }
    void set_b(int64_t b_val)
    {
        this->b = b_val;
    }

    int64_t get_n() const
    {
        return n;
    }
    void set_n(int64_t n_val)
    {
        this->n = n_val;
    }

    int64_t get_g() const
    {
        return g;
    }
    void set_g(int64_t g_val)
    {
        this->g = g_val;
    }

    int64_t get_sQ() const
    {
        return sQ;
    }
    void set_sQ(int64_t sQ_val)
    {
        this->sQ = sQ_val;
    }

    int64_t get_sKV() const
    {
        return sKV;
    }
    void set_sKV(int64_t sKV_val)
    {
        this->sKV = sKV_val;
    }

    int64_t get_sKVAlign() const
    {
        return sKVAlign;
    }
    void set_sKVAlign(int64_t sKVAlign_val)
    {
        this->sKVAlign = sKVAlign_val;
    }

    int64_t get_sKVAlignSize() const
    {
        return sKVAlignSize;
    }
    void set_sKVAlignSize(int64_t sKVAlignSize_val)
    {
        this->sKVAlignSize = sKVAlignSize_val;
    }

    int64_t get_sKVAlignVec() const
    {
        return sKVAlignVec;
    }
    void set_sKVAlignVec(int64_t sKVAlignVec_val)
    {
        this->sKVAlignVec = sKVAlignVec_val;
    }

    int64_t get_sKVAlignSizeVec() const
    {
        return sKVAlignSizeVec;
    }
    void set_sKVAlignSizeVec(int64_t sKVAlignSizeVec_val)
    {
        this->sKVAlignSizeVec = sKVAlignSizeVec_val;
    }

    int64_t get_sKVAlignByte() const
    {
        return sKVAlignByte;
    }
    void set_sKVAlignByte(int64_t sKVAlignByte_val)
    {
        this->sKVAlignByte = sKVAlignByte_val;
    }

    int64_t get_hQ() const
    {
        return hQ;
    }
    void set_hQ(int64_t hQ_val)
    {
        this->hQ = hQ_val;
    }

    int64_t get_hKV() const
    {
        return hKV;
    }
    void set_hKV(int64_t hKV_val)
    {
        this->hKV = hKV_val;
    }

    int64_t get_d() const
    {
        return d;
    }
    void set_d(int64_t d_val)
    {
        this->d = d_val;
    }

    int64_t get_originalDAlign() const
    {
        return originalDAlign;
    }
    void set_originalDAlign(int64_t originalDAlign_val)
    {
        this->originalDAlign = originalDAlign_val;
    }

    float get_scaleValue() const
    {
        return scaleValue;
    }
    void set_scaleValue(float scaleValue_val)
    {
        this->scaleValue = scaleValue_val;
    }

    float get_keepProb() const
    {
        return keepProb;
    }
    void set_keepProb(float keepProb_val)
    {
        this->keepProb = keepProb_val;
    }

    int64_t get_preTokens() const
    {
        return preTokens;
    }
    void set_preTokens(int64_t preTokens_val)
    {
        this->preTokens = preTokens_val;
    }

    int64_t get_nextTokens() const
    {
        return nextTokens;
    }
    void set_nextTokens(int64_t nextTokens_val)
    {
        this->nextTokens = nextTokens_val;
    }

    int64_t get_headNum() const
    {
        return headNum;
    }
    void set_headNum(int64_t headNum_val)
    {
        this->headNum = headNum_val;
    }

    uint32_t get_inputLayout() const
    {
        return inputLayout;
    }
    void set_inputLayout(uint32_t inputLayout_val)
    {
        this->inputLayout = inputLayout_val;
    }

    int64_t get_preTokensBlocks() const
    {
        return preTokensBlocks;
    }
    void set_preTokensBlocks(int64_t preTokensBlocks_val)
    {
        this->preTokensBlocks = preTokensBlocks_val;
    }

    int64_t get_nextTokensBlocks() const
    {
        return nextTokensBlocks;
    }
    void set_nextTokensBlocks(int64_t nextTokensBlocks_val)
    {
        this->nextTokensBlocks = nextTokensBlocks_val;
    }

    uint32_t get_inputDType() const
    {
        return inputDType;
    }
    void set_inputDType(uint32_t inputDType_val)
    {
        this->inputDType = inputDType_val;
    }

    uint32_t get_inputDTypeSize() const
    {
        return inputDTypeSize;
    }
    void set_inputDTypeSize(uint32_t inputDTypeSize_val)
    {
        this->inputDTypeSize = inputDTypeSize_val;
    }

    uint32_t get_vecCalcDTypeSize() const
    {
        return vecCalcDTypeSize;
    }
    void set_vecCalcDTypeSize(uint32_t vecCalcDTypeSize_val)
    {
        this->vecCalcDTypeSize = vecCalcDTypeSize_val;
    }

    uint32_t get_pseSq() const
    {
        return pseSq;
    }
    void set_pseSq(uint32_t pseSq_val)
    {
        this->pseSq = pseSq_val;
    }

    uint32_t get_pseShapeType() const
    {
        return pseShapeType;
    }
    void set_pseShapeType(uint32_t pseShapeType_val)
    {
        this->pseShapeType = pseShapeType_val;
    }

    uint32_t get_attenMaskShapeType() const
    {
        return attenMaskShapeType;
    }
    void set_attenMaskShapeType(uint32_t attenMaskShapeType_val)
    {
        this->attenMaskShapeType = attenMaskShapeType_val;
    }

    uint32_t get_hasAttenMask() const
    {
        return hasAttenMask;
    }
    void set_hasAttenMask(uint32_t hasAttenMask_val)
    {
        this->hasAttenMask = hasAttenMask_val;
    }

    uint32_t get_attenMaskCompressMode() const
    {
        return attenMaskCompressMode;
    }
    void set_attenMaskCompressMode(uint32_t attenMaskCompressMode_val)
    {
        this->attenMaskCompressMode = attenMaskCompressMode_val;
    }

    int64_t get_attenMaskS2Size() const
    {
        return attenMaskS2Size;
    }
    void set_attenMaskS2Size(int64_t attenMaskS2Size_val)
    {
        this->attenMaskS2Size = attenMaskS2Size_val;
    }

    uint32_t get_elementPerBlock() const
    {
        return elementPerBlock;
    }
    void set_elementPerBlock(uint32_t elementPerBlock_val)
    {
        this->elementPerBlock = elementPerBlock_val;
    }

    uint32_t get_precisionMode() const
    {
        return precisionMode;
    }
    void set_precisionMode(uint32_t precisionMode_val)
    {
        this->precisionMode = precisionMode_val;
    }

    uint32_t get_resv() const
    {
        return resv;
    }
    void set_resv(uint32_t resv_val)
    {
        this->resv = resv_val;
    }

    int64_t get_mm1WorkspaceLen() const
    {
        return mm1WorkspaceLen;
    }
    void set_mm1WorkspaceLen(int64_t mm1WorkspaceLen_val)
    {
        this->mm1WorkspaceLen = mm1WorkspaceLen_val;
    }

    int64_t get_mm2WorkspaceLen() const
    {
        return mm2WorkspaceLen;
    }
    void set_mm2WorkspaceLen(int64_t mm2WorkspaceLen_val)
    {
        this->mm2WorkspaceLen = mm2WorkspaceLen_val;
    }

    int64_t get_dqWorkspaceLen() const
    {
        return dqWorkspaceLen;
    }
    void set_dqWorkspaceLen(int64_t dqWorkspaceLen_val)
    {
        this->dqWorkspaceLen = dqWorkspaceLen_val;
    }

    int64_t get_dkWorkspaceLen() const
    {
        return dkWorkspaceLen;
    }
    void set_dkWorkspaceLen(int64_t dkWorkspaceLen_val)
    {
        this->dkWorkspaceLen = dkWorkspaceLen_val;
    }

    int64_t get_dropGmWorkspaceLen() const
    {
        return dropGmWorkspaceLen;
    }
    void set_dropGmWorkspaceLen(int64_t dropGmWorkspaceLen_val)
    {
        this->dropGmWorkspaceLen = dropGmWorkspaceLen_val;
    }

    int64_t get_mulGmWorkspaceLen() const
    {
        return mulGmWorkspaceLen;
    }
    void set_mulGmWorkspaceLen(int64_t mulGmWorkspaceLen_val)
    {
        this->mulGmWorkspaceLen = mulGmWorkspaceLen_val;
    }

    int64_t get_dropoutWorkspaceLen() const
    {
        return dropoutWorkspaceLen;
    }
    void set_dropoutWorkspaceLen(int64_t dropoutWorkspaceLen_val)
    {
        this->dropoutWorkspaceLen = dropoutWorkspaceLen_val;
    }

    void reset()
    {
        b = 0;
        n = 0;
        g = 0;
        sQ = 0;
        sKV = 0;
        sKVAlign = 0;
        sKVAlignSize = 0;
        sKVAlignVec = 0;
        sKVAlignSizeVec = 0;
        sKVAlignByte = 0;
        hQ = 0;
        hKV = 0;
        d = 0;
        originalDAlign = 0;
        scaleValue = 0.0f;
        keepProb = 0.0f;
        preTokens = 0;
        nextTokens = 0;
        headNum = 0;
        inputLayout = 0;
        preTokensBlocks = 0;
        nextTokensBlocks = 0;
        inputDType = 0;
        inputDTypeSize = 0;
        vecCalcDTypeSize = 0;
        pseSq = 0;
        pseShapeType = 0;
        attenMaskShapeType = 0;
        hasAttenMask = 0;
        attenMaskCompressMode = 0;
        attenMaskS2Size = 0;
        elementPerBlock = 0;
        precisionMode = 0;
        resv = 0;
        mm1WorkspaceLen = 0;
        mm2WorkspaceLen = 0;
        dqWorkspaceLen = 0;
        dkWorkspaceLen = 0;
        dropGmWorkspaceLen = 0;
        mulGmWorkspaceLen = 0;
        dropoutWorkspaceLen = 0;
    }
};


class SplitNSplitCoreParams {
public:
    int64_t totalBatch = 0;
    int64_t nOut = 0;
    int64_t apiClcQueueSize = 0;
    int64_t mm1ResSize = 0;
    uint32_t usedCoreNum = 0;
    uint32_t reserved = 0;

    int64_t get_totalBatch() const
    {
        return totalBatch;
    }
    void set_totalBatch(int64_t totalBatch_val)
    {
        this->totalBatch = totalBatch_val;
    }

    int64_t get_nOut() const
    {
        return nOut;
    }
    void set_nOut(int64_t nOut_val)
    {
        this->nOut = nOut_val;
    }

    int64_t get_apiClcQueueSize() const
    {
        return apiClcQueueSize;
    }
    void set_apiClcQueueSize(int64_t apiClcQueueSize_val)
    {
        this->apiClcQueueSize = apiClcQueueSize_val;
    }

    int64_t get_mm1ResSize() const
    {
        return mm1ResSize;
    }
    void set_mm1ResSize(int64_t mm1ResSize_val)
    {
        this->mm1ResSize = mm1ResSize_val;
    }

    uint32_t get_usedCoreNum() const
    {
        return usedCoreNum;
    }
    void set_usedCoreNum(uint32_t usedCoreNum_val)
    {
        this->usedCoreNum = usedCoreNum_val;
    }

    uint32_t get_reserved() const
    {
        return reserved;
    }
    void set_reserved(uint32_t reserved_val)
    {
        this->reserved = reserved_val;
    }

    void reset()
    {
        totalBatch = 0;
        nOut = 0;
        apiClcQueueSize = 0;
        mm1ResSize = 0;
        usedCoreNum = 0;
        reserved = 0;
    }
};


class SplitNSingleCoreParams {
public:
    int64_t nIn = 0;
    int64_t nInTail = 0;
    uint32_t singleCoreBatchRange = 0;
    uint32_t singleCoreBatchRangeTail = 0;
    int64_t nCvInner = 0;
    int64_t innerTmpBufSize = 0;
    int64_t vecCastSize = 0;
    int64_t splitedDAlign = 0;
    int64_t dRange = 0;
    int64_t vecQueIn1Size = 0;
    int64_t subRange = 0;
    int64_t subMask = 0;
    int64_t subMaskTail = 0;
    int64_t sKVAlignBlockNumVec = 0;

    // Getters and Setters
    int64_t get_nIn() const
    {
        return nIn;
    }
    void set_nIn(int64_t nIn_val)
    {
        this->nIn = nIn_val;
    }

    int64_t get_nInTail() const
    {
        return nInTail;
    }
    void set_nInTail(int64_t nInTail_val)
    {
        this->nInTail = nInTail_val;
    }

    uint32_t get_singleCoreBatchRange() const
    {
        return singleCoreBatchRange;
    }
    void set_singleCoreBatchRange(uint32_t singleCoreBatchRange_val)
    {
        this->singleCoreBatchRange = singleCoreBatchRange_val;
    }

    uint32_t get_singleCoreBatchRangeTail() const
    {
        return singleCoreBatchRangeTail;
    }
    void set_singleCoreBatchRangeTail(uint32_t singleCoreBatchRangeTail_val)
    {
        this->singleCoreBatchRangeTail = singleCoreBatchRangeTail_val;
    }

    int64_t get_nCvInner() const
    {
        return nCvInner;
    }
    void set_nCvInner(int64_t nCvInner_val)
    {
        this->nCvInner = nCvInner_val;
    }

    int64_t get_innerTmpBufSize() const
    {
        return innerTmpBufSize;
    }
    void set_innerTmpBufSize(int64_t innerTmpBufSize_val)
    {
        this->innerTmpBufSize = innerTmpBufSize_val;
    }

    int64_t get_vecCastSize() const
    {
        return vecCastSize;
    }
    void set_vecCastSize(int64_t vecCastSize_val)
    {
        this->vecCastSize = vecCastSize_val;
    }

    int64_t get_splitedDAlign() const
    {
        return splitedDAlign;
    }
    void set_splitedDAlign(int64_t splitedDAlign_val)
    {
        this->splitedDAlign = splitedDAlign_val;
    }

    int64_t get_dRange() const
    {
        return dRange;
    }
    void set_dRange(int64_t dRange_val)
    {
        this->dRange = dRange_val;
    }

    int64_t get_vecQueIn1Size() const
    {
        return vecQueIn1Size;
    }
    void set_vecQueIn1Size(int64_t vecQueIn1Size_val)
    {
        this->vecQueIn1Size = vecQueIn1Size_val;
    }

    int64_t get_subRange() const
    {
        return subRange;
    }
    void set_subRange(int64_t subRange_val)
    {
        this->subRange = subRange_val;
    }

    int64_t get_subMask() const
    {
        return subMask;
    }
    void set_subMask(int64_t subMask_val)
    {
        this->subMask = subMask_val;
    }

    int64_t get_subMaskTail() const
    {
        return subMaskTail;
    }
    void set_subMaskTail(int64_t subMaskTail_val)
    {
        this->subMaskTail = subMaskTail_val;
    }

    int64_t get_sKVAlignBlockNumVec() const
    {
        return sKVAlignBlockNumVec;
    }
    void set_sKVAlignBlockNumVec(int64_t sKVAlignBlockNumVec_val)
    {
        this->sKVAlignBlockNumVec = sKVAlignBlockNumVec_val;
    }

    void reset()
    {
        nIn = 0;
        nInTail = 0;
        singleCoreBatchRange = 0;
        singleCoreBatchRangeTail = 0;
        nCvInner = 0;
        innerTmpBufSize = 0;
        vecCastSize = 0;
        splitedDAlign = 0;
        dRange = 0;
        vecQueIn1Size = 0;
        subRange = 0;
        subMask = 0;
        subMaskTail = 0;
        sKVAlignBlockNumVec = 0;
    }
};

class FlashAttentionScoreGradShapeAttrParamsForB {
public:
    int64_t b = 0;
    int64_t n = 0;
    int64_t g = 0;
    int64_t sQ = 0;
    int64_t sKV = 0;
    int64_t sKVAlign = 0;
    int64_t sKVAlignSize = 0;
    int64_t sKVAlignByte = 0;
    int64_t hQ = 0;
    int64_t hKV = 0;
    int64_t d = 0;
    int64_t originalDAlign = 0;
    float scaleValue = 0;
    float keepProb = 0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    int64_t headNum = 0;
    uint32_t inputLayout = 0;
    uint8_t preTokensBlocksPH[4] = {};
    int64_t preTokensBlocks = 0;
    int64_t nextTokensBlocks = 0;
    uint32_t inputDType = 0;
    uint8_t inputDTypeSizePH[4] = {};
    int64_t inputDTypeSize = 0;
    uint32_t vecCalcDTypeSize = 0;
    uint32_t pseSq = 0;
    uint32_t existPse = 0;
    uint32_t pseShapeType = 0;
    uint32_t attenMaskShapeType = 0;
    uint32_t hasAttenMask = 0;
    uint32_t attenMaskCompressMode = 0;
    uint8_t attenMaskS2SizePH[4] = {};
    int64_t attenMaskS2Size = 0;
    uint32_t precisionMode = 0;
    uint32_t syncLen = 0;
    int64_t mm1WorkspaceLen = 0;
    int64_t mm2WorkspaceLen = 0;
    int64_t dqWorkspaceLen = 0;
    int64_t dkWorkspaceLen = 0;
    int64_t dropGmWorkspaceLen = 0;
    int64_t mulGmWorkspaceLen = 0;
    int64_t dropoutWorkspaceLen = 0;
    uint32_t placeholder = 0;
    uint8_t FlashAttentionScoreGradShapeAttrParamsForBPH[4] = {};

    // b
    int64_t get_b() const
    {
        return b;
    }
    void set_b(int64_t b_val)
    {
        this->b = b_val;
    }

    // n
    int64_t get_n() const
    {
        return n;
    }
    void set_n(int64_t n_val)
    {
        this->n = n_val;
    }

    // g
    int64_t get_g() const
    {
        return g;
    }
    void set_g(int64_t g_val)
    {
        this->g = g_val;
    }

    // sQ
    int64_t get_sQ() const
    {
        return sQ;
    }
    void set_sQ(int64_t sQ_val)
    {
        this->sQ = sQ_val;
    }

    // sKV
    int64_t get_sKV() const
    {
        return sKV;
    }
    void set_sKV(int64_t sKV_val)
    {
        this->sKV = sKV_val;
    }

    // sKVAlign
    int64_t get_sKVAlign() const
    {
        return sKVAlign;
    }
    void set_sKVAlign(int64_t sKVAlign_val)
    {
        this->sKVAlign = sKVAlign_val;
    }

    // sKVAlignSize
    int64_t get_sKVAlignSize() const
    {
        return sKVAlignSize;
    }
    void set_sKVAlignSize(int64_t sKVAlignSize_val)
    {
        this->sKVAlignSize = sKVAlignSize_val;
    }

    // sKVAlignByte
    int64_t get_sKVAlignByte() const
    {
        return sKVAlignByte;
    }
    void set_sKVAlignByte(int64_t sKVAlignByte_val)
    {
        this->sKVAlignByte = sKVAlignByte_val;
    }

    // hQ
    int64_t get_hQ() const
    {
        return hQ;
    }
    void set_hQ(int64_t hQ_val)
    {
        this->hQ = hQ_val;
    }

    // hKV
    int64_t get_hKV() const
    {
        return hKV;
    }
    void set_hKV(int64_t hKV_val)
    {
        this->hKV = hKV_val;
    }

    // d
    int64_t get_d() const
    {
        return d;
    }
    void set_d(int64_t d_val)
    {
        this->d = d_val;
    }

    // originalDAlign
    int64_t get_originalDAlign() const
    {
        return originalDAlign;
    }
    void set_originalDAlign(int64_t originalDAlign_val)
    {
        this->originalDAlign = originalDAlign_val;
    }

    // scaleValue
    float get_scaleValue() const
    {
        return scaleValue;
    }
    void set_scaleValue(float scaleValue_val)
    {
        this->scaleValue = scaleValue_val;
    }

    // keepProb
    float get_keepProb() const
    {
        return keepProb;
    }
    void set_keepProb(float keepProb_val)
    {
        this->keepProb = keepProb_val;
    }

    // preTokens
    int64_t get_preTokens() const
    {
        return preTokens;
    }
    void set_preTokens(int64_t preTokens_val)
    {
        this->preTokens = preTokens_val;
    }

    // nextTokens
    int64_t get_nextTokens() const
    {
        return nextTokens;
    }
    void set_nextTokens(int64_t nextTokens_val)
    {
        this->nextTokens = nextTokens_val;
    }

    // headNum
    int64_t get_headNum() const
    {
        return headNum;
    }
    void set_headNum(int64_t headNum_val)
    {
        this->headNum = headNum_val;
    }

    // inputLayout
    uint32_t get_inputLayout() const
    {
        return inputLayout;
    }
    void set_inputLayout(uint32_t inputLayout_val)
    {
        this->inputLayout = inputLayout_val;
    }

    // preTokensBlocks
    int64_t get_preTokensBlocks() const
    {
        return preTokensBlocks;
    }
    void set_preTokensBlocks(int64_t preTokensBlocks_val)
    {
        this->preTokensBlocks = preTokensBlocks_val;
    }

    // nextTokensBlocks
    int64_t get_nextTokensBlocks() const
    {
        return nextTokensBlocks;
    }
    void set_nextTokensBlocks(int64_t nextTokensBlocks_val)
    {
        this->nextTokensBlocks = nextTokensBlocks_val;
    }

    // inputDType
    uint32_t get_inputDType() const
    {
        return inputDType;
    }
    void set_inputDType(uint32_t inputDType_val)
    {
        this->inputDType = inputDType_val;
    }

    // inputDTypeSize
    int64_t get_inputDTypeSize() const
    {
        return inputDTypeSize;
    }
    void set_inputDTypeSize(int64_t inputDTypeSize_val)
    {
        this->inputDTypeSize = inputDTypeSize_val;
    }

    // vecCalcDTypeSize
    uint32_t get_vecCalcDTypeSize() const
    {
        return vecCalcDTypeSize;
    }
    void set_vecCalcDTypeSize(uint32_t vecCalcDTypeSize_val)
    {
        this->vecCalcDTypeSize = vecCalcDTypeSize_val;
    }

    // pseSq
    uint32_t get_pseSq() const
    {
        return pseSq;
    }
    void set_pseSq(uint32_t pseSq_val)
    {
        this->pseSq = pseSq_val;
    }

    // existPse
    uint32_t get_existPse() const
    {
        return existPse;
    }
    void set_existPse(uint32_t existPse_val)
    {
        this->existPse = existPse_val;
    }

    // pseShapeType
    uint32_t get_pseShapeType() const
    {
        return pseShapeType;
    }
    void set_pseShapeType(uint32_t pseShapeType_val)
    {
        this->pseShapeType = pseShapeType_val;
    }

    // attenMaskShapeType
    uint32_t get_attenMaskShapeType() const
    {
        return attenMaskShapeType;
    }
    void set_attenMaskShapeType(uint32_t attenMaskShapeType_val)
    {
        this->attenMaskShapeType = attenMaskShapeType_val;
    }

    // hasAttenMask
    uint32_t get_hasAttenMask() const
    {
        return hasAttenMask;
    }
    void set_hasAttenMask(uint32_t hasAttenMask_val)
    {
        this->hasAttenMask = hasAttenMask_val;
    }

    // attenMaskCompressMode
    uint32_t get_attenMaskCompressMode() const
    {
        return attenMaskCompressMode;
    }
    void set_attenMaskCompressMode(uint32_t attenMaskCompressMode_val)
    {
        this->attenMaskCompressMode = attenMaskCompressMode_val;
    }

    // attenMaskS2Size
    int64_t get_attenMaskS2Size() const
    {
        return attenMaskS2Size;
    }
    void set_attenMaskS2Size(int64_t attenMaskS2Size_val)
    {
        this->attenMaskS2Size = attenMaskS2Size_val;
    }

    // precisionMode
    uint32_t get_precisionMode() const
    {
        return precisionMode;
    }
    void set_precisionMode(uint32_t precisionMode_val)
    {
        this->precisionMode = precisionMode_val;
    }

    // syncLen
    uint32_t get_syncLen() const
    {
        return syncLen;
    }
    void set_syncLen(uint32_t syncLen_val)
    {
        this->syncLen = syncLen_val;
    }

    // mm1WorkspaceLen
    int64_t get_mm1WorkspaceLen() const
    {
        return mm1WorkspaceLen;
    }
    void set_mm1WorkspaceLen(int64_t mm1WorkspaceLen_val)
    {
        this->mm1WorkspaceLen = mm1WorkspaceLen_val;
    }

    // mm2WorkspaceLen
    int64_t get_mm2WorkspaceLen() const
    {
        return mm2WorkspaceLen;
    }
    void set_mm2WorkspaceLen(int64_t mm2WorkspaceLen_val)
    {
        this->mm2WorkspaceLen = mm2WorkspaceLen_val;
    }

    // dqWorkspaceLen
    int64_t get_dqWorkspaceLen() const
    {
        return dqWorkspaceLen;
    }
    void set_dqWorkspaceLen(int64_t dqWorkspaceLen_val)
    {
        this->dqWorkspaceLen = dqWorkspaceLen_val;
    }

    // dkWorkspaceLen
    int64_t get_dkWorkspaceLen() const
    {
        return dkWorkspaceLen;
    }
    void set_dkWorkspaceLen(int64_t dkWorkspaceLen_val)
    {
        this->dkWorkspaceLen = dkWorkspaceLen_val;
    }

    // dropGmWorkspaceLen
    int64_t get_dropGmWorkspaceLen() const
    {
        return dropGmWorkspaceLen;
    }
    void set_dropGmWorkspaceLen(int64_t dropGmWorkspaceLen_val)
    {
        this->dropGmWorkspaceLen = dropGmWorkspaceLen_val;
    }

    // mulGmWorkspaceLen
    int64_t get_mulGmWorkspaceLen() const
    {
        return mulGmWorkspaceLen;
    }
    void set_mulGmWorkspaceLen(int64_t mulGmWorkspaceLen_val)
    {
        this->mulGmWorkspaceLen = mulGmWorkspaceLen_val;
    }

    // dropoutWorkspaceLen
    int64_t get_dropoutWorkspaceLen() const
    {
        return dropoutWorkspaceLen;
    }
    void set_dropoutWorkspaceLen(int64_t dropoutWorkspaceLen_val)
    {
        this->dropoutWorkspaceLen = dropoutWorkspaceLen_val;
    }

    // placeholder
    uint32_t get_placeholder() const
    {
        return placeholder;
    }
    void set_placeholder(uint32_t placeholder_val)
    {
        this->placeholder = placeholder_val;
    }

    void reset()
    {
        b = 0;
        n = 0;
        g = 0;
        sQ = 0;
        sKV = 0;
        sKVAlign = 0;
        sKVAlignSize = 0;
        sKVAlignByte = 0;
        hQ = 0;
        hKV = 0;
        d = 0;
        originalDAlign = 0;
        scaleValue = 0.0f;
        keepProb = 0.0f;
        preTokens = 0;
        nextTokens = 0;
        headNum = 0;
        inputLayout = 0;
        preTokensBlocks = 0;
        nextTokensBlocks = 0;
        inputDType = 0;
        inputDTypeSize = 0;
        vecCalcDTypeSize = 0;
        pseSq = 0;
        existPse = 0;
        pseShapeType = 0;
        attenMaskShapeType = 0;
        hasAttenMask = 0;
        attenMaskCompressMode = 0;
        attenMaskS2Size = 0;
        precisionMode = 0;
        syncLen = 0;
        mm1WorkspaceLen = 0;
        mm2WorkspaceLen = 0;
        dqWorkspaceLen = 0;
        dkWorkspaceLen = 0;
        dropGmWorkspaceLen = 0;
        mulGmWorkspaceLen = 0;
        dropoutWorkspaceLen = 0;
        placeholder = 0;
    }
};

class SplitBSplitCoreParams {
public:
    int64_t bOut = 0;
    int64_t apiClcQueueSize = 0;
    uint32_t usedCoreNum = 0;
    uint32_t reserved = 0;

    int64_t get_bOut() const
    {
        return bOut;
    }
    void set_bOut(int64_t bOut_val)
    {
        this->bOut = bOut_val;
    }

    int64_t get_apiClcQueueSize() const
    {
        return apiClcQueueSize;
    }
    void set_apiClcQueueSize(int64_t apiClcQueueSize_val)
    {
        this->apiClcQueueSize = apiClcQueueSize_val;
    }

    uint32_t get_usedCoreNum() const
    {
        return usedCoreNum;
    }
    void set_usedCoreNum(uint32_t usedCoreNum_val)
    {
        this->usedCoreNum = usedCoreNum_val;
    }

    uint32_t get_reserved() const
    {
        return reserved;
    }
    void set_reserved(uint32_t reserved_val)
    {
        this->reserved = reserved_val;
    }

    void reset()
    {
        bOut = 0;
        apiClcQueueSize = 0;
        usedCoreNum = 0;
        reserved = 0;
    }
};

class SplitBSingleCoreParams {
public:
    int64_t bIn = 0;
    uint32_t singleCoreBatchRange = 0;
    uint32_t bCvInner = 0;
    uint32_t bCvRatio = 0;
    uint8_t innerTmpBufSizePH[4] = {};
    int64_t innerTmpBufSize = 0;
    int64_t clcDInner = 0;
    int64_t dSize = 0;
    int64_t dInnerTail = 0;
    int64_t dInnerTailAlign = 0;
    int64_t vecQueIn1Size = 0;
    int64_t subRange = 0;
    int64_t subMask = 0;
    int64_t subMaskTail = 0;
    int64_t sKVAlignBlockNum = 0;
    int64_t rightPadding = 0;
    int64_t dstStride = 0;

    int64_t get_bIn() const
    {
        return bIn;
    }
    void set_bIn(int64_t bIn_val)
    {
        this->bIn = bIn_val;
    }

    uint32_t get_singleCoreBatchRange() const
    {
        return singleCoreBatchRange;
    }
    void set_singleCoreBatchRange(uint32_t singleCoreBatchRange_val)
    {
        this->singleCoreBatchRange = singleCoreBatchRange_val;
    }

    uint32_t get_bCvInner() const
    {
        return bCvInner;
    }
    void set_bCvInner(uint32_t bCvInner_val)
    {
        this->bCvInner = bCvInner_val;
    }

    uint32_t get_bCvRatio() const
    {
        return bCvRatio;
    }
    void set_bCvRatio(uint32_t bCvRatio_val)
    {
        this->bCvRatio = bCvRatio_val;
    }

    int64_t get_innerTmpBufSize() const
    {
        return innerTmpBufSize;
    }
    void set_innerTmpBufSize(int64_t innerTmpBufSize_val)
    {
        this->innerTmpBufSize = innerTmpBufSize_val;
    }

    int64_t get_clcDInner() const
    {
        return clcDInner;
    }
    void set_clcDInner(int64_t clcDInner_val)
    {
        this->clcDInner = clcDInner_val;
    }

    int64_t get_dSize() const
    {
        return dSize;
    }
    void set_dSize(int64_t dSize_val)
    {
        this->dSize = dSize_val;
    }

    int64_t get_dInnerTail() const
    {
        return dInnerTail;
    }
    void set_dInnerTail(int64_t dInnerTail_val)
    {
        this->dInnerTail = dInnerTail_val;
    }

    int64_t get_dInnerTailAlign() const
    {
        return dInnerTailAlign;
    }
    void set_dInnerTailAlign(int64_t dInnerTailAlign_val)
    {
        this->dInnerTailAlign = dInnerTailAlign_val;
    }

    int64_t get_vecQueIn1Size() const
    {
        return vecQueIn1Size;
    }
    void set_vecQueIn1Size(int64_t vecQueIn1Size_val)
    {
        this->vecQueIn1Size = vecQueIn1Size_val;
    }

    int64_t get_subRange() const
    {
        return subRange;
    }
    void set_subRange(int64_t subRange_val)
    {
        this->subRange = subRange_val;
    }

    int64_t get_subMask() const
    {
        return subMask;
    }
    void set_subMask(int64_t subMask_val)
    {
        this->subMask = subMask_val;
    }

    int64_t get_subMaskTail() const
    {
        return subMaskTail;
    }
    void set_subMaskTail(int64_t subMaskTail_val)
    {
        this->subMaskTail = subMaskTail_val;
    }

    int64_t get_sKVAlignBlockNum() const
    {
        return sKVAlignBlockNum;
    }
    void set_sKVAlignBlockNum(int64_t sKVAlignBlockNum_val)
    {
        this->sKVAlignBlockNum = sKVAlignBlockNum_val;
    }

    int64_t get_rightPadding() const
    {
        return rightPadding;
    }
    void set_rightPadding(int64_t rightPadding_val)
    {
        this->rightPadding = rightPadding_val;
    }

    int64_t get_dstStride() const
    {
        return dstStride;
    }
    void set_dstStride(int64_t dstStride_val)
    {
        this->dstStride = dstStride_val;
    }

    void reset()
    {
        bIn = 0;
        singleCoreBatchRange = 0;
        bCvInner = 0;
        bCvRatio = 0;
        innerTmpBufSize = 0;
        clcDInner = 0;
        dSize = 0;
        dInnerTail = 0;
        dInnerTailAlign = 0;
        vecQueIn1Size = 0;
        subRange = 0;
        subMask = 0;
        subMaskTail = 0;
        sKVAlignBlockNum = 0;
        rightPadding = 0;
        dstStride = 0;
    }
};

class EmptyTensorTilingData {
public:
    uint32_t formerDqNum = 0;
    uint32_t formerDkNum = 0;
    uint32_t formerDvNum = 0;
    uint32_t formerDpseNum = 0;
    uint32_t res = 0;
    uint8_t singleCoreDqNumPH[4] = {};
    uint64_t singleCoreDqNum = 0;
    uint64_t tailCoreDqNum = 0;
    uint64_t singleCoreDkNum = 0;
    uint64_t tailCoreDkNum = 0;
    uint64_t singleCoreDvNum = 0;
    uint64_t tailCoreDvNum = 0;
    uint64_t singleCoreDpseNum = 0;
    uint64_t tailCoreDpseNum = 0;

    // formerDqNum
    uint32_t get_formerDqNum() const
    {
        return formerDqNum;
    }
    void set_formerDqNum(uint32_t formerDqNum_val)
    {
        this->formerDqNum = formerDqNum_val;
    }

    // formerDkNum
    uint32_t get_formerDkNum() const
    {
        return formerDkNum;
    }
    void set_formerDkNum(uint32_t formerDkNum_val)
    {
        this->formerDkNum = formerDkNum_val;
    }

    // formerDvNum
    uint32_t get_formerDvNum() const
    {
        return formerDvNum;
    }
    void set_formerDvNum(uint32_t formerDvNum_val)
    {
        this->formerDvNum = formerDvNum_val;
    }

    // formerDpseNum
    uint32_t get_formerDpseNum() const
    {
        return formerDpseNum;
    }
    void set_formerDpseNum(uint32_t formerDpseNum_val)
    {
        this->formerDpseNum = formerDpseNum_val;
    }

    // res
    uint32_t get_res() const
    {
        return res;
    }
    void set_res(uint32_t res_val)
    {
        this->res = res_val;
    }

    // singleCoreDqNum
    uint64_t get_singleCoreDqNum() const
    {
        return singleCoreDqNum;
    }
    void set_singleCoreDqNum(uint64_t singleCoreDqNum_val)
    {
        this->singleCoreDqNum = singleCoreDqNum_val;
    }

    // tailCoreDqNum
    uint64_t get_tailCoreDqNum() const
    {
        return tailCoreDqNum;
    }
    void set_tailCoreDqNum(uint64_t tailCoreDqNum_val)
    {
        this->tailCoreDqNum = tailCoreDqNum_val;
    }

    // singleCoreDkNum
    uint64_t get_singleCoreDkNum() const
    {
        return singleCoreDkNum;
    }
    void set_singleCoreDkNum(uint64_t singleCoreDkNum_val)
    {
        this->singleCoreDkNum = singleCoreDkNum_val;
    }

    // tailCoreDkNum
    uint64_t get_tailCoreDkNum() const
    {
        return tailCoreDkNum;
    }
    void set_tailCoreDkNum(uint64_t tailCoreDkNum_val)
    {
        this->tailCoreDkNum = tailCoreDkNum_val;
    }

    // singleCoreDvNum
    uint64_t get_singleCoreDvNum() const
    {
        return singleCoreDvNum;
    }
    void set_singleCoreDvNum(uint64_t singleCoreDvNum_val)
    {
        this->singleCoreDvNum = singleCoreDvNum_val;
    }

    // tailCoreDvNum
    uint64_t get_tailCoreDvNum() const
    {
        return tailCoreDvNum;
    }
    void set_tailCoreDvNum(uint64_t tailCoreDvNum_val)
    {
        this->tailCoreDvNum = tailCoreDvNum_val;
    }

    // singleCoreDpseNum
    uint64_t get_singleCoreDpseNum() const
    {
        return singleCoreDpseNum;
    }
    void set_singleCoreDpseNum(uint64_t singleCoreDpseNum_val)
    {
        this->singleCoreDpseNum = singleCoreDpseNum_val;
    }

    // tailCoreDpseNum
    uint64_t get_tailCoreDpseNum() const
    {
        return tailCoreDpseNum;
    }
    void set_tailCoreDpseNum(uint64_t tailCoreDpseNum_val)
    {
        this->tailCoreDpseNum = tailCoreDpseNum_val;
    }

    void reset()
    {
        formerDqNum = 0;
        formerDkNum = 0;
        formerDvNum = 0;
        formerDpseNum = 0;
        res = 0;
        singleCoreDqNum = 0;
        tailCoreDqNum = 0;
        singleCoreDkNum = 0;
        tailCoreDkNum = 0;
        singleCoreDvNum = 0;
        tailCoreDvNum = 0;
        singleCoreDpseNum = 0;
        tailCoreDpseNum = 0;
    }
};


class MLATensorTilingData {
public:
    uint32_t coreNum = 0;
    float scaleValue = 0;
    int64_t b = 0;
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t n2 = 0;
    int64_t g = 0;
    int64_t d = 0;
    int64_t qSize = 0;
    int64_t kvSize = 0;
    int64_t sfmgSize = 0;
    uint32_t sparseMode = 0;
    uint8_t dqWorkSpaceOffsetPH[4] = {};
    int64_t dqWorkSpaceOffset = 0;
    int64_t dkWorkSpaceOffset = 0;
    int64_t dvWorkSpaceOffset = 0;
    int64_t sfmgWorkspaceOffset = 0;
    int64_t mm1WorkspaceOffset = 0;
    int64_t mm2WorkspaceOffset = 0;
    int64_t pWorkspaceOffset = 0;
    int64_t dsWorkspaceOffset = 0;
    uint8_t tndSoftmaxIn = 0;
    uint8_t softmaxTilingDataPH[7] = {};
    SoftMaxTiling softmaxTilingData;
    SoftMaxTiling softmaxGradTilingData;

    uint32_t get_coreNum() const
    {
        return coreNum;
    }
    void set_coreNum(uint32_t coreNum_val)
    {
        this->coreNum = coreNum_val;
    }

    float get_scaleValue() const
    {
        return scaleValue;
    }
    void set_scaleValue(float scaleValue_val)
    {
        this->scaleValue = scaleValue_val;
    }

    uint64_t get_b() const
    {
        return b;
    }
    void set_b(uint64_t b_val)
    {
        this->b = b_val;
    }

    uint64_t get_t1() const
    {
        return t1;
    }
    void set_t1(uint64_t t1_val)
    {
        this->t1 = t1_val;
    }

    uint64_t get_t2() const
    {
        return t2;
    }
    void set_t2(uint64_t t2_val)
    {
        this->t2 = t2_val;
    }

    uint64_t get_n2() const
    {
        return n2;
    }
    void set_n2(uint64_t n2_val)
    {
        this->n2 = n2_val;
    }

    uint64_t get_g() const
    {
        return g;
    }
    void set_g(uint64_t g_val)
    {
        this->g = g_val;
    }

    uint64_t get_d() const
    {
        return d;
    }
    void set_d(uint64_t d_val)
    {
        this->d = d_val;
    }

    uint64_t get_qSize() const
    {
        return qSize;
    }
    void set_qSize(uint64_t qSize_val)
    {
        this->qSize = qSize_val;
    }

    uint64_t get_kvSize() const
    {
        return kvSize;
    }
    void set_kvSize(uint64_t kvSize_val)
    {
        this->kvSize = kvSize_val;
    }

    uint64_t get_sfmgSize() const
    {
        return sfmgSize;
    }
    void set_sfmgSize(uint64_t sfmgSize_val)
    {
        this->sfmgSize = sfmgSize_val;
    }

    uint32_t get_sparseMode() const
    {
        return sparseMode;
    }
    void set_sparseMode(uint32_t sparseMode_val)
    {
        this->sparseMode = sparseMode_val;
    }

    uint64_t get_dqWorkSpaceOffset() const
    {
        return dqWorkSpaceOffset;
    }
    void set_dqWorkSpaceOffset(uint64_t dqWorkSpaceOffset_val)
    {
        this->dqWorkSpaceOffset = dqWorkSpaceOffset_val;
    }

    uint64_t get_dkWorkSpaceOffset() const
    {
        return dkWorkSpaceOffset;
    }
    void set_dkWorkSpaceOffset(uint64_t dkWorkSpaceOffset_val)
    {
        this->dkWorkSpaceOffset = dkWorkSpaceOffset_val;
    }

    uint64_t get_dvWorkSpaceOffset() const
    {
        return dvWorkSpaceOffset;
    }
    void set_dvWorkSpaceOffset(uint64_t dvWorkSpaceOffset_val)
    {
        this->dvWorkSpaceOffset = dvWorkSpaceOffset_val;
    }

    uint64_t get_sfmgWorkspaceOffset() const
    {
        return sfmgWorkspaceOffset;
    }
    void set_sfmgWorkspaceOffset(uint64_t sfmgWorkspaceOffset_val)
    {
        this->sfmgWorkspaceOffset = sfmgWorkspaceOffset_val;
    }

    uint64_t get_mm1WorkspaceOffset() const
    {
        return mm1WorkspaceOffset;
    }
    void set_mm1WorkspaceOffset(uint64_t mm1WorkspaceOffset_val)
    {
        this->mm1WorkspaceOffset = mm1WorkspaceOffset_val;
    }

    uint64_t get_mm2WorkspaceOffset() const
    {
        return mm2WorkspaceOffset;
    }
    void set_mm2WorkspaceOffset(uint64_t mm2WorkspaceOffset_val)
    {
        this->mm2WorkspaceOffset = mm2WorkspaceOffset_val;
    }

    uint64_t get_pWorkspaceOffset() const
    {
        return pWorkspaceOffset;
    }
    void set_pWorkspaceOffset(uint64_t pWorkspaceOffset_val)
    {
        this->pWorkspaceOffset = pWorkspaceOffset_val;
    }
    uint64_t get_dsWorkspaceOffset() const
    {
        return dsWorkspaceOffset;
    }
    void set_dsWorkspaceOffset(uint64_t dsWorkspaceOffset_val)
    {
        this->dsWorkspaceOffset = dsWorkspaceOffset_val;
    }

    uint8_t get_tndSoftmaxIn() const
    {
        return tndSoftmaxIn;
    }
    void set_tndSoftmaxIn(uint8_t tndSoftmaxIn_val)
    {
        this->tndSoftmaxIn = tndSoftmaxIn_val;
    }
};

class BasicDetTensorTilingData {
public:
    uint32_t coreNum = 0;
    float scaleValue = 0;
    int64_t b = 0;
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t s1 = 0;
    int64_t s2 = 0;
    int64_t n2 = 0;
    int64_t g = 0;
    int64_t d = 0;
    int64_t qSize = 0;
    int64_t kvSize = 0;
    int64_t sfmgSize = 0;
    float keepProb = 0;
    int64_t preTockens = 0;
    int64_t nextTockens = 0;
    uint32_t dqPostAbsorb = 0;
    uint32_t sparseMode = 0;
    uint32_t layout = 0;
    uint8_t dqWorkSpaceOffsetPH[4] = {};
    int64_t dqWorkSpaceOffset = 0;
    int64_t dkWorkSpaceOffset = 0;
    int64_t dvWorkSpaceOffset = 0;
    int64_t sfmgWorkspaceOffset = 0;
    int64_t mm1WorkspaceOffset = 0;
    int64_t mm2WorkspaceOffset = 0;
    int64_t pWorkspaceOffset = 0;
    int64_t dsWorkspaceOffset = 0;
    int64_t dqDetWorkspaceOffset = 0;
    int64_t dkDetWorkspaceOffset = 0;
    int64_t dvDetWorkspaceOffset = 0;
    uint8_t tndSoftmaxIn = 0;
    uint8_t softmaxTilingDataPH[7] = {};
    SoftMaxTiling softmaxTilingData;
    SoftMaxTiling softmaxGradTilingData;

    uint32_t get_coreNum() const
    {
        return coreNum;
    }
    void set_coreNum(uint32_t coreNum_val)
    {
        this->coreNum = coreNum_val;
    }

    float get_scaleValue() const
    {
        return scaleValue;
    }
    void set_scaleValue(float scaleValue_val)
    {
        this->scaleValue = scaleValue_val;
    }

    // keepProb
    float get_keepProb() const
    {
        return keepProb;
    }
    void set_keepProb(float keepProb_val)
    {
        this->keepProb = keepProb_val;
    }

    // preTockens
    int64_t get_preTockens() const
    {
        return preTockens;
    }
    void set_preTockens(int64_t preTockens_val)
    {
        this->preTockens = preTockens_val;
    }

    // nextTockens
    int64_t get_nextTockens() const
    {
        return nextTockens;
    }
    void set_nextTockens(int64_t nextTockens_val)
    {
        this->nextTockens = nextTockens_val;
    }

    // dqPostAbsorb
    uint32_t get_dqPostAbsorb() const
    {
        return dqPostAbsorb;
    }
    void set_dqPostAbsorb(uint32_t dqPostAbsorb_val)
    {
        this->dqPostAbsorb = dqPostAbsorb_val;
    }

    uint64_t get_b() const
    {
        return b;
    }
    void set_b(uint64_t b_val)
    {
        this->b = b_val;
    }

    uint64_t get_t1() const
    {
        return t1;
    }
    void set_t1(uint64_t t1_val)
    {
        this->t1 = t1_val;
    }

    uint64_t get_t2() const
    {
        return t2;
    }
    void set_t2(uint64_t t2_val)
    {
        this->t2 = t2_val;
    }
    uint64_t get_s1() const
    {
        return s1;
    }
    void set_s1(uint64_t s1_val)
    {
        this->s1 = s1_val;
    }

    uint64_t get_s2() const
    {
        return s2;
    }
    void set_s2(uint64_t s2_val)
    {
        this->s2 = s2_val;
    }

    uint64_t get_n2() const
    {
        return n2;
    }
    void set_n2(uint64_t n2_val)
    {
        this->n2 = n2_val;
    }

    uint64_t get_g() const
    {
        return g;
    }
    void set_g(uint64_t g_val)
    {
        this->g = g_val;
    }

    uint64_t get_d() const
    {
        return d;
    }
    void set_d(uint64_t d_val)
    {
        this->d = d_val;
    }

    uint64_t get_qSize() const
    {
        return qSize;
    }
    void set_qSize(uint64_t qSize_val)
    {
        this->qSize = qSize_val;
    }

    uint64_t get_kvSize() const
    {
        return kvSize;
    }
    void set_kvSize(uint64_t kvSize_val)
    {
        this->kvSize = kvSize_val;
    }

    uint64_t get_sfmgSize() const
    {
        return sfmgSize;
    }
    void set_sfmgSize(uint64_t sfmgSize_val)
    {
        this->sfmgSize = sfmgSize_val;
    }

    uint32_t get_sparseMode() const
    {
        return sparseMode;
    }
    void set_sparseMode(uint32_t sparseMode_val)
    {
        this->sparseMode = sparseMode_val;
    }

    uint32_t get_layout() const
    {
        return layout;
    }
    void set_layout(uint32_t layout_val)
    {
        this->layout = layout_val;
    }

    uint64_t get_dqWorkSpaceOffset() const
    {
        return dqWorkSpaceOffset;
    }
    void set_dqWorkSpaceOffset(uint64_t dqWorkSpaceOffset_val)
    {
        this->dqWorkSpaceOffset = dqWorkSpaceOffset_val;
    }

    uint64_t get_dkWorkSpaceOffset() const
    {
        return dkWorkSpaceOffset;
    }
    void set_dkWorkSpaceOffset(uint64_t dkWorkSpaceOffset_val)
    {
        this->dkWorkSpaceOffset = dkWorkSpaceOffset_val;
    }

    uint64_t get_dvWorkSpaceOffset() const
    {
        return dvWorkSpaceOffset;
    }
    void set_dvWorkSpaceOffset(uint64_t dvWorkSpaceOffset_val)
    {
        this->dvWorkSpaceOffset = dvWorkSpaceOffset_val;
    }

    uint64_t get_sfmgWorkspaceOffset() const
    {
        return sfmgWorkspaceOffset;
    }
    void set_sfmgWorkspaceOffset(uint64_t sfmgWorkspaceOffset_val)
    {
        this->sfmgWorkspaceOffset = sfmgWorkspaceOffset_val;
    }

    uint64_t get_mm1WorkspaceOffset() const
    {
        return mm1WorkspaceOffset;
    }
    void set_mm1WorkspaceOffset(uint64_t mm1WorkspaceOffset_val)
    {
        this->mm1WorkspaceOffset = mm1WorkspaceOffset_val;
    }

    uint64_t get_mm2WorkspaceOffset() const
    {
        return mm2WorkspaceOffset;
    }
    void set_mm2WorkspaceOffset(uint64_t mm2WorkspaceOffset_val)
    {
        this->mm2WorkspaceOffset = mm2WorkspaceOffset_val;
    }

    uint64_t get_pWorkspaceOffset() const
    {
        return pWorkspaceOffset;
    }
    void set_pWorkspaceOffset(uint64_t pWorkspaceOffset_val)
    {
        this->pWorkspaceOffset = pWorkspaceOffset_val;
    }

    uint64_t get_dsWorkspaceOffset() const
    {
        return dsWorkspaceOffset;
    }
    void set_dsWorkspaceOffset(uint64_t dsWorkspaceOffset_val)
    {
        this->dsWorkspaceOffset = dsWorkspaceOffset_val;
    }

    uint64_t get_dqDetWorkspaceOffset() const
    {
        return dqDetWorkspaceOffset;
    }
    void set_dqDetWorkspaceOffset(uint64_t dqDetWorkspaceOffset_val)
    {
        this->dqDetWorkspaceOffset = dqDetWorkspaceOffset_val;
    }

    uint64_t get_dkDetWorkspaceOffset() const
    {
        return dkDetWorkspaceOffset;
    }
    void set_dkDetWorkspaceOffset(uint64_t dkDetWorkspaceOffset_val)
    {
        this->dkDetWorkspaceOffset = dkDetWorkspaceOffset_val;
    }

    uint64_t get_dvDetWorkspaceOffset() const
    {
        return dvDetWorkspaceOffset;
    }
    void set_dvDetWorkspaceOffset(uint64_t dvDetWorkspaceOffset_val)
    {
        this->dvDetWorkspaceOffset = dvDetWorkspaceOffset_val;
    }

    uint8_t get_tndSoftmaxIn() const
    {
        return tndSoftmaxIn;
    }
    void set_tndSoftmaxIn(uint8_t tndSoftmaxIn_val)
    {
        this->tndSoftmaxIn = tndSoftmaxIn_val;
    }
};

// begin def of all tiling struct
class FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb {
public:
    FlashAttentionScoreGradS1S2BNGS1S2SABBaseParams s1s2BNGS1S2BaseParams;
    FlashAttentionScoreGradS1S2BNGS1S2SABSplitCoreParams s1s2BNGS1S2SplitCoreParams;
    TCubeTiling mm1TilingData;
    TCubeTiling mm2TilingData;
    TCubeTiling mm3TilingData;
    SoftMaxTiling softmaxTilingData;
    SoftMaxTiling softmaxGradTilingData;
    PreParams preTilingData;
    PreSfmgParams preSfmgTilingData;
    PostParams postTilingData;

    void reset()
    {
        s1s2BNGS1S2BaseParams.reset();
        s1s2BNGS1S2SplitCoreParams.reset();
        preTilingData.reset();
        preSfmgTilingData.reset();
        postTilingData.reset();
    }
};

class FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 {
public:
    FlashAttentionScoreGradS1S2BNGS1S2BaseParams s1s2BNGS1S2BaseParams;
    FlashAttentionScoreGradS1S2BNGS1S2SplitCoreParams s1s2BNGS1S2SplitCoreParams;
    TCubeTiling mm1TilingData;
    TCubeTiling mm2TilingData;
    TCubeTiling mm3TilingData;
    SoftMaxTiling softmaxTilingData;
    SoftMaxTiling softmaxGradTilingData;
    BlockNumListParams s1s2BNGS1S2BlockNumList;
    PreParams preTilingData;
    PostParams postTilingData;

    void reset()
    {
        s1s2BNGS1S2BaseParams.reset();
        s1s2BNGS1S2SplitCoreParams.reset();
        s1s2BNGS1S2BlockNumList.reset();
        preTilingData.reset();
        postTilingData.reset();
    }
};

class FlashAttentionScoreGradTilingDataS1s2Bn2 {
public:
    FlashAttentionScoreGradBaseParamsS1s2Bn2 opInfo;
    SplitCoreParamsS1s2Bn2 splitCoreParams;
    PreParams preTilingData;
    PostParams postTilingData;
    DeterministicTndSplitCoreS1s2Bn2 tndSplitCoreParams;
    TCubeTiling mm1TilingData;
    TCubeTiling mm31TilingData;
    TCubeTiling mm4TilingData;
    SoftMaxTiling softmaxTilingData;
    SoftMaxTiling softmaxGradTilingData;

    void reset()
    {
        opInfo.reset();
        splitCoreParams.reset();
        preTilingData.reset();
        postTilingData.reset();
        tndSplitCoreParams.reset();
    }
};

class FlashAttentionScoreGradTilingDataUngs1s2Bbn {
public:
    FlashAttentionScoreGradShapeAttrParams1 opInfo;
    SplitNSplitCoreParams splitCoreParams;
    SplitNSingleCoreParams singleCoreParams;
    PreParams preTilingData;
    PostParams postTilingData;
    TCubeTiling mm1AndMm2TilingData;
    TCubeTiling mm31TilingData;
    TCubeTiling mm32AndMm4TilingData;
    SoftMaxTiling softmaxTilingData;
    SoftMaxTiling softmaxGradTilingData;

    void reset()
    {
        opInfo.reset();
        splitCoreParams.reset();
        singleCoreParams.reset();
        preTilingData.reset();
        postTilingData.reset();
    }
};

class FlashAttentionScoreGradUbngs1s2BbTilingData {
public:
    FlashAttentionScoreGradShapeAttrParamsForB opInfo;
    SplitBSplitCoreParams splitCoreParams;
    SplitBSingleCoreParams singleCoreParams;
    PreParams preTilingData;
    PostParams postTilingData;
    TCubeTiling mm1AndMm2TilingData;
    TCubeTiling mm31TilingData;
    TCubeTiling mm32AndMm4TilingData;
    SoftMaxTiling softmaxTilingData;
    SoftMaxTiling softmaxGradTilingData;

    void reset()
    {
        opInfo.reset();
        splitCoreParams.reset();
        singleCoreParams.reset();
        preTilingData.reset();
        postTilingData.reset();
    }
};

class FlashAttentionScoreGradTilingData {
public:
    EmptyTensorTilingData emptyTensorTilingData;
};

class FlashAttentionGradMlaTilingData {
public:
    MLATensorTilingData mlaTensorTilingData;
};

class FlashAttentionGradBasicDetTilingData {
public:
    BasicDetTensorTilingData basicDetTensorTilingData;
};
#endif