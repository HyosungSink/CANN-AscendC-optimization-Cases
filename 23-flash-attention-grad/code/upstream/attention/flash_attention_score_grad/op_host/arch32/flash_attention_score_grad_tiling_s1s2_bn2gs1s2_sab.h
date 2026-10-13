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
 * \file flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.h
 * \brief
 */

#pragma once

#include "flash_attention_score_grad_tiling_common.h"
#include "tiling_base/tiling_base.h"


using namespace Ops::Transformer::OpTiling;

namespace optiling {

struct SameAbFuzzyBaseInfoParams { // 频繁使用的基础参数
    int64_t coreNum;
    int64_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0cSize;

    int64_t b;
    int64_t n1;
    int64_t n2;
    int64_t s1;
    int64_t s2;
    int64_t g;
    int64_t d;          // query, key dim
    int64_t rope_d = 0; // query_rope, key_rope dim
    int64_t value_d;    // value dim
    uint32_t s1Align = 0;
    int64_t s1Outer;
    uint32_t s1Inner;
    uint32_t s1CvInner;
    uint32_t s2CvInner;
    uint32_t s1Tail;
    uint32_t s1CvTail;
    uint32_t s2Align;
    int64_t s2Outer;
    uint32_t s2Inner;
    uint32_t s2Tail;
    uint32_t s2CvTail;
    int64_t sfmgNormalAxisSize = 0;
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t sumS1S2Product = 0;

    int64_t pseType = PSE_OUTER_ADD_MUL_TYPE;
    int64_t pseAlibiBaseS1 = 0;
    int64_t pseAlibiBaseS2 = 0;
    int64_t qStartIdx = 0;
    int64_t kvStartIdx = 0;
    uint32_t pseOptional;
    uint32_t pseShapeType = 0;
    uint32_t pseDtype = 0;

    uint32_t queryType;
    uint32_t attenMaskOptional;
    uint32_t attenMaskShapeType = 0;
    uint32_t attenMaskDtype = 0;
    uint32_t attenMaskCompressMode;
    int64_t attenMaskS1Size = 0;
    int64_t attenMaskS2Size = 0;
    uint32_t dropoutIsDivisibleBy8 = 0;
    uint32_t layoutType;
    float scaleValue;
    float keepProb;
    int64_t bandIdx;

    uint32_t dataTypeSize;
    uint32_t dataBlockNum;
    uint32_t calTypeSize;
    uint32_t calBlockNum;
    uint32_t blockNum;
    int64_t s1Token;
    int64_t s2Token;
    uint32_t blockOuter;
    int64_t blockFactor;

    int64_t qSize; // 元素个数: tokens*head_num*group_ratio*head_dim
    int64_t qRopeSize = 0;
    int64_t kvSize;
    int64_t kRopeSize = 0;
    int64_t vSize;
    int64_t qSizeAlign;
    int64_t qRopeSizeAlign = 0;
    int64_t kvSizeAlign;
    int64_t kRopeSizeAlign = 0;
    int64_t vSizeAlign;
    int64_t dropMaskSize;
    uint32_t sink = 0; // 判断是否有sink，默认没有传入

    uint32_t baseMN;
    uint32_t sparseMode;
    std::vector<int64_t> prefixN;

    std::vector<int64_t> actualSeqQlen;
    std::vector<int64_t> actualSeqKvlen;

    bool isSparse;
    bool isBf16;
    bool mm1IsNZOut;
    bool mm2IsNZOut;
    bool isDeterministic;
    bool enableL1Custom{false};

    bool tndSoftmaxIn;
    uint32_t tmpBufferSize = 0;

    TilingDataType mode;
};

class FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb : public TilingBaseClass {
public:
    explicit FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb(gert::TilingContext *context) : TilingBaseClass(context)
    {
        tilingData->reset();
    }
    FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb *tilingData = context_->GetTilingData<FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb>();

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus DoSplit();
    void SetMatmulTilingBufferInfo(AscendC::tiling::TCubeTiling* mmTiling);
    ge::graphStatus DoPreTiling();
    ge::graphStatus DoPostTiling();
    void DetermineMode();
    virtual void SetQKVStartIdx();
    ge::graphStatus CheckAttenMaskShape();
    ge::graphStatus ProcessPseInfo(const char *inputLayout);
    ge::graphStatus ProcessDropInfo(const char *inputLayout);
    ge::graphStatus ProcessPseNormal(const char *inputLayout);
    ge::graphStatus ProcessSparseModeInfo();
    ge::graphStatus ProcessTokensInfo();
    void ProcessSinkInfo();
    ge::graphStatus SaveToTilingData();
    int64_t FindBandIdx();
    bool SetSparseParams();
    void PrintShapeInfo();
    ge::graphStatus GetBaseShapeInfo();
    void DoPreSfmgTiling();
    void AdjustCvInner();
    void ChooseL1Custom();
    float CalculateMaskRatio();

private:
    SameAbFuzzyBaseInfoParams fBaseParams;
};

class FlashAttentionScoreGradTilingSameABDeterministic : public FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb {
public:
    explicit FlashAttentionScoreGradTilingSameABDeterministic(gert::TilingContext *context)
        : FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb(context)
    {
    }

protected:
    bool IsCapable() override;
};

} // namespace optiling
