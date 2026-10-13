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
 * \file flash_attention_score_grad_tiling_s1s2_bn2gs1s2.h
 * \brief
 */

#pragma once

#include "flash_attention_score_grad_tiling_common.h"
#include "tiling_base/tiling_base.h"

using namespace Ops::Transformer::OpTiling;

namespace optiling {

constexpr uint32_t CORE_LIST_NUM = 50;
constexpr uint32_t ARRAY_LENGTH = 3;
struct FuzzyBaseInfoParams { // 频繁使用的基础参数
    int64_t coreNum = 0;
    int64_t aicNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0cSize = 0;

    int64_t b = 0;
    int64_t n1 = 0;
    int64_t n2 = 0;
    int64_t s1 = 0;
    int64_t s2 = 0;
    int64_t g = 0;
    int64_t d = 0;
    int64_t rope_d = 0;
    int64_t value_d = 0;
    uint32_t s1Align = 0;
    int64_t s1Outer = 0;
    uint32_t s1Inner = 0;
    uint32_t s1CvInner = 0;
    uint32_t s1Tail = 0;
    uint32_t s1CvTail = 0;
    uint32_t s2Align = 0;
    int64_t s2Outer = 0;
    uint32_t s1CvRatio = 1;
    uint32_t s2CvRatio = 1;
    uint32_t mm1CalRatio = 1;
    uint32_t cvS2Inner = 0;
    uint32_t s2Inner = 0;
    uint32_t s2Tail = 0;
    uint32_t s2CvTail = 0;
    uint32_t sfmgdOuter = 0;
    uint32_t sfmgdInner = 0;
    uint32_t sfmgdTail = 0;
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t sumS1S2Product = 0;

    int64_t pseType = PSE_OUTER_ADD_MUL_TYPE;
    int64_t pseAlibiBaseS1 = 0;
    int64_t pseAlibiBaseS2 = 0;
    int64_t qStartIdx = 0;
    int64_t kvStartIdx = 0;
    uint32_t pseOptional = 0;
    uint32_t pseShapeType = 0;
    uint32_t pseDtype = 0;

    uint32_t queryType = 0;
    uint32_t attenMaskOptional = 0;
    uint32_t attenMaskShapeType = 0;
    uint32_t attenMaskDtype = 0;
    uint32_t attenMaskCompressMode = 0;
    int64_t attenMaskS1Size = 0;
    int64_t attenMaskS2Size = 0;
    uint32_t dropoutIsDivisibleBy8 = 0;
    uint32_t layoutType = 0;
    float scaleValue = 0;
    float keepProb = 0;
    int64_t bandIdx = 0;

    uint32_t dataTypeSize = 0;
    uint32_t dataBlockNum = 0;
    uint32_t calTypeSize = 0;
    uint32_t calBlockNum = 0;
    uint32_t blockNum = 0;
    int64_t s1Token = 0;
    int64_t s2Token = 0;
    uint32_t blockOuter = 0;
    int64_t blockFactor = 0;

    int64_t qSize = 0;
    int64_t kvSize = 0;
    int64_t qRopeSize = 0;
    int64_t kRopeSize = 0;
    int64_t vSize = 0;
    int64_t qSizeAlign = 0;
    int64_t kvSizeAlign = 0;
    int64_t vSizeAlign = 0;
    int64_t qRopeSizeAlign = 0;
    int64_t kRopeSizeAlign = 0;
    int64_t dropMaskSize = 0;

    uint32_t baseMN = 0;
    int64_t blockStarts[CORE_LIST_NUM] = {0};
    int64_t blockEnds[CORE_LIST_NUM] = {0};
    uint32_t sparseMode = 0;
    std::vector<int64_t> prefixN;

    std::vector<int64_t> actualSeqQlen;
    std::vector<int64_t> actualSeqKvlen;

    bool isSparse = false;
    bool isBf16 = false;
    bool mm1IsNZOut = false;
    bool mm2IsNZOut = false;
    bool isSplitBalance = false;
    bool isTndS1PingPong = false;

    bool tndSoftmaxIn;
    uint32_t tmpBufferSize = 0;

    TilingDataType mode;
};

class FlashAttentionScoreGradTilingS1s2Bn2gs1s2 : public TilingBaseClass {
public:
    explicit FlashAttentionScoreGradTilingS1s2Bn2gs1s2(gert::TilingContext *context) : TilingBaseClass(context)
    {
        tilingData->reset();
    }
    FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *tilingData = context_->GetTilingData<FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2>();

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
    ge::graphStatus DoSparse();
    bool CheckFuzzyArgsLegal(uint32_t s1Inner, uint32_t s2Inner);
    std::tuple<uint32_t, uint32_t, uint32_t> FuzzyForBestSplit();
    void SetMatmulTilingBufferInfo(AscendC::tiling::TCubeTiling* mmTiling);
    ge::graphStatus GetSparseBlockInfo();
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
    ge::graphStatus SaveToTilingData();
    ge::graphStatus GetSparsePrefixBlockInfo();
    ge::graphStatus GetSparseUnpadBlockInfo();
    int64_t FindBandIdx();

    bool CheckForDichotomy(std::vector<int64_t> &nums, int64_t x, int64_t m);
    int64_t GetSplitArrayMinMaxSum(std::vector<int64_t> &s1s2WeightNums, int64_t coreNum);
    void FillEveryBlockInfo(std::vector<std::vector<int64_t>> &everyBlockInfo);
    void FillS1s2WeightVec(std::vector<std::vector<int64_t>> &everyBlockInfo, std::vector<int64_t> &s1s2Weight);
    bool CoreLoadBalancingForUnpad(int64_t (&blockStarts)[CORE_LIST_NUM], int64_t (&blockEnds)[CORE_LIST_NUM],
                                   int64_t splitCounter);

    void FillBlockInfo(std::vector<std::vector<std::vector<int64_t>>> &calculatedBlockInfo,
                       std::vector<std::vector<int64_t>> &totalBlockInfo);

    bool CheckPrefixNExist(const int64_t bIdx, const int64_t prefixN,
                           std::vector<std::vector<std::pair<int64_t, int64_t>>> &s1ValidIdx);
    void GetCommS1S2OuterInfo(const int64_t s1, const int64_t s2, int64_t (*parseInfo)[ARRAY_LENGTH]);
    void GetParseS1S2OuterInfo(int64_t (*parseInfo)[ARRAY_LENGTH]);
    bool SetSparseParams();
    void GetCommS1S2OuterInfo(const int64_t prefixN, std::vector<std::pair<int64_t, int64_t>> &s1ValidIdx);
    void PrintShapeInfo();
    ge::graphStatus GetBaseShapeInfo();
    ge::graphStatus ProcessTndToBsh();
    bool tnd2bsh = false;

private:
    FuzzyBaseInfoParams fBaseParams;
};

} // namespace optiling
