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
 * \file rotary_position_embedding.cc
 * \brief
 */
#include "rotary_position_embedding_tiling.h"
#include "rope_rotate_matrix_tiling.h"
#include "rope_rotate_half_tiling.h"
#include "rope_interleaved_tiling.h"
#include "register/op_def_registry.h"
#include "log/log.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
constexpr uint32_t MODE_ATTR_IDX = 0;

ge::graphStatus RotaryPosEmbeddingMembaseTilingClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        socVersion_ = ascendcPlatform.GetSocVersion();
        aicoreParams_.ubSize = ubSizePlatForm;
    } else {
        auto compileInfoPtr = reinterpret_cast<const RotaryPositionEmbeddingCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.numBlocks = compileInfoPtr->numBlocks;
        socVersion_ = compileInfoPtr->socVersion;
    }
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus RotaryPosEmbeddingMembaseTilingClass::GetShapeAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const uint32_t inputMode = *(attrs->GetAttrPointer<uint32_t>(MODE_ATTR_IDX));
    OP_LOGI(context_->GetNodeName(), "[mode]: %d", inputMode);
    inputMode_ = inputMode;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4RotaryPositionEmbedding(gert::TilingContext *context)
{
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForRotaryPositionEmbedding(gert::TilingParseContext *context)
{
    OP_LOGD(context, "TilingPrepareForRotaryPositionEmbedding context success");
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(RotaryPositionEmbedding, RopeRotateMatrixTilingClass, 49999);
REGISTER_OPS_TILING_TEMPLATE(RotaryPositionEmbedding, RopeRotateHalfTilingClass, 50000);
REGISTER_OPS_TILING_TEMPLATE(RotaryPositionEmbedding, RopeInterLeavedTilingClass, 60000);

IMPL_OP_OPTILING(RotaryPositionEmbedding)
    .Tiling(Tiling4RotaryPositionEmbedding)
    .TilingParse<RotaryPositionEmbeddingCompileInfo>(TilingPrepareForRotaryPositionEmbedding);
} // namespace optiling
