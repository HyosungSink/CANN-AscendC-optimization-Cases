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
 * \file flash_attention_score_grad_tiling_s1s2_bn2gs1s2_basic.h
 * \brief
 */

#pragma once

#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_type.h"
#include "flash_attention_score_grad_tiling_common.h"

namespace optiling {
struct MlaBaseInfoParams {
    uint32_t coreNum;
    int64_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0cSize;
    uint32_t queryType;
    uint64_t l2CacheSize{0};
    float scaleValue;
    int64_t b;
    int64_t t1;
    int64_t t2;
    int64_t n1;
    int64_t n2;
    int64_t g;
    int64_t d; // the headdim of query or key
    int64_t qSize;
    int64_t s1;
    int64_t s2;
    int64_t dv; // the headdim of value
    uint32_t sparseMode;
    float keepProb;
    const char *inputLayout;
    bool eaqualActSeqLen{true};
    bool pseEnable{false};
    bool dropMskEnable{false};
    bool attenMskEnable{false};

    int64_t dqWorkSpaceOffset;
    int64_t dkWorkSpaceOffset;
    int64_t dvWorkSpaceOffset;
    int64_t sfmgWorkspaceOffset;
    int64_t mm1WorkspaceOffset;
    int64_t mm2WorkspaceOffset;
    int64_t pWorkspaceOffset;
    int64_t dssWorkspaceOffset;
    std::vector<int64_t> actualSeqQlen;
    std::vector<int64_t> actualSeqKvlen;
};

struct MultiHeadLatentAttentionGradCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    int64_t coreNum;
};

class FlashAttentionScoreGraTilingMla : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit FlashAttentionScoreGraTilingMla(gert::TilingContext *context) : TilingBaseClass(context)
    {
        
    }
    FlashAttentionGradMlaTilingData *tilingData = context_->GetTilingData<FlashAttentionGradMlaTilingData>();

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus SetBaseInfo();
    ge::graphStatus SetAttrsInfo();
    bool IsShapeCapable();
    bool IsAttenMskCapable();

private:
    MlaBaseInfoParams fBaseParams;
};

} // namespace optiling