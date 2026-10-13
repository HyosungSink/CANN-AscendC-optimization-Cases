/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qkv_rms_norm_rope_cache_tiling_base.cpp
 * \brief
 */
#include "qkv_rms_norm_rope_cache_tiling.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "../common/include/tiling_base/tiling_util.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {
using namespace Ops::Transformer::OpTiling;

std::tuple<int64_t, int64_t, int64_t, int64_t> QkvRmsNormRopeCacheTilingBase::GetShapeTuple(
    const gert::TilingContext* context, const int64_t index)
{
    const gert::StorageShape* shapePtr = context->GetInputShape(index);
    OP_CHECK_IF(shapePtr == nullptr, OP_LOGE(context->GetNodeName(), "Shape is nullptr."), return std::make_tuple(0, 0, 0, 0));
    // check shape length is DIM_SIZE
    OP_CHECK_IF(
        shapePtr->GetStorageShape().GetDimNum() != DIM_SIZE, OP_LOGE(context->GetNodeName(), "the number of dimensions must be 4."),
        return std::make_tuple(0, 0, 0, 0));
    return std::make_tuple(
        shapePtr->GetStorageShape().GetDim(DIM_ZERO), shapePtr->GetStorageShape().GetDim(DIM_ONE),
        shapePtr->GetStorageShape().GetDim(DIM_TWO), shapePtr->GetStorageShape().GetDim(DIM_THREE));
}

std::tuple<int64_t, int64_t> QkvRmsNormRopeCacheTilingBase::GetShapeTupleOfTH(
    const gert::TilingContext* context, const int64_t index)
{
    const gert::StorageShape* shapePtr = context->GetInputShape(index);
    OP_CHECK_IF(shapePtr == nullptr, OP_LOGE(context->GetNodeName(), "Shape is nullptr."), return std::make_tuple(0, 0));
    // check shape length is DIM_SIZE
    OP_CHECK_IF(
        shapePtr->GetStorageShape().GetDimNum() != DIM_TWO, OP_LOGE(context->GetNodeName(), "the number of dimensions must be 2."),
        return std::make_tuple(0, 0));
    return std::make_tuple(
        shapePtr->GetStorageShape().GetDim(SHAPE_IDX_BS), shapePtr->GetStorageShape().GetDim(SHAPE_IDX_ND));
}

ge::graphStatus QkvRmsNormRopeCacheTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<QkvRmsNormRopeCacheCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"),
                      return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
        OP_LOGD(context_->GetNodeName(), "GetPlatformInfo : platformInfo == nullptr");
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
        isRegbase_ = IsRegbaseSocVersion(context_);
        OP_LOGD(context_->GetNodeName(), "GetPlatformInfo : platformInfo != nullptr");
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t QkvRmsNormRopeCacheTilingBase::GetTilingKey() const
{
    return tilingKey_;
}
ge::graphStatus Tiling4QkvRmsNormRopeCache(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4QkvRmsNormRopeCache running.");
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
    OP_LOGD(context, "Tiling4QkvRmsNormRopeCache exit.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4QkvRmsNormRopeCache(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4QkvRmsNormRopeCache running.");
    auto compileInfo = context->GetCompiledInfo<QkvRmsNormRopeCacheCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0),
        OP_LOGE(
            context->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);
    
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_CHECK_IF(
        (compileInfo->ubSize <= 0),
        OP_LOGE(
            context->GetNodeName(), "ubSize must be greater than 0."),
        return ge::GRAPH_FAILED);
    
    OP_LOGD(context, "coreNum: %ld, ubSize: %ld", compileInfo->coreNum, compileInfo->ubSize);
    OP_LOGD(context, "TilingPrepare4QkvRmsNormRopeCache success.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QkvRmsNormRopeCache)
    .Tiling(Tiling4QkvRmsNormRopeCache)
    .TilingParse<QkvRmsNormRopeCacheCompileInfo>(TilingPrepare4QkvRmsNormRopeCache);
} // namespace optiling
