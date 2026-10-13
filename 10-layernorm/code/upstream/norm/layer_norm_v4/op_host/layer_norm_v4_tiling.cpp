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
 * \file layer_norm_v4_tiling.cpp
 * \brief
 */

#include "op_host/tiling_util.h"
#include "layer_norm_v4_tiling.h"

namespace optiling {
using namespace Ops::NN::OpTiling;

static ge::graphStatus Tiling4LayerNormV4(gert::TilingContext* context)
{
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4CompileInfo(gert::TilingParseContext* context, LayerNormV4CompileInfo* compileInfo)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    compileInfo->isAscend310P = npuArch == NpuArch::DAV_2002;
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0),
        OP_LOGE(context, "Get core num failed, core num: %u", static_cast<uint32_t>(compileInfo->coreNum)),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0),
        OP_LOGE(context, "Get ub size failed, ub size: %u", static_cast<uint32_t>(compileInfo->ubSizePlatForm)),
        return ge::GRAPH_FAILED);
    compileInfo->isRegBase =
        (IsRegbaseSocVersion(context) ||
         npuArch == NpuArch::DAV_5102) ? true : false;
    compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
    compileInfo->vectorLength = Ops::Base::GetVRegSize(context);
    OP_LOGD(context, "TilingPrepare4LayerNormV4 exit.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4LayerNormV4(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4LayerNormV4 enter.");
    auto compileInfo = context->GetCompiledInfo<LayerNormV4CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return TilingPrepare4CompileInfo(context, compileInfo);
}

IMPL_OP_OPTILING(LayerNormV4).Tiling(Tiling4LayerNormV4).TilingParse<LayerNormV4CompileInfo>(TilingPrepare4LayerNormV4);

} // namespace optiling
