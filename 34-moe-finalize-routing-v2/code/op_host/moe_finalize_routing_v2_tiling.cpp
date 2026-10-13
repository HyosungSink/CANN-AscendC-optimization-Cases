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
 * \file moe_finalize_routing_v2_tiling.cpp
 * \brief
 */
#include "moe_finalize_routing_v2_tiling.h"

using namespace std;
using namespace ge;

namespace optiling {

static constexpr size_t WORKSPACE_RESERVED = static_cast<int64_t>(16 * 1024 * 1024);

ge::graphStatus MoeFinalizeRoutingTilingV2::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();
    return DoGetPlatformInfo();
}

ge::graphStatus MoeFinalizeRoutingTilingV2::GetShapeAttrsInfo()
{
    return DoGetShapeAttrsInfo();
}

uint64_t MoeFinalizeRoutingTilingV2::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus MoeFinalizeRoutingTilingV2::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_RESERVED;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingTilingV2::PostTiling()
{
    DoPostTiling();
    uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(usedCoreNum_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingTilingV2::DoOpTiling()
{
    OP_CHECK_IF(
        (CalcOpTiling() != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "CalcOpTiling failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CalcTilingKey() != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "CalcTilingKey failed."), return ge::GRAPH_FAILED);

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingTilingV2::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForMoeFinalizeRoutingV2(gert::TilingContext* context)
{
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForMoeFinalizeRoutingV2(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeFinalizeRoutingV2)
    .Tiling(TilingForMoeFinalizeRoutingV2)
    .TilingParse<MoeFinalizeRoutingCompileInfoV2>(TilingPrepareForMoeFinalizeRoutingV2);

} // namespace optiling