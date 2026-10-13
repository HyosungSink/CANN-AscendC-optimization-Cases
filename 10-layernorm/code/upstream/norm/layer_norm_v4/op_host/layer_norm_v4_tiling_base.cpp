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
 * \file layer_norm_v4_tiling_base.cpp
 * \brief
 */

#include "layer_norm_v4_tiling.h"
#include "exe_graph/runtime/shape.h"

namespace optiling {
constexpr size_t K_INPUT_IDX_X = 0;
constexpr size_t K_INPUT_IDX_NORM_SHAPE = 1;
constexpr size_t K_INPUT_IDX_GAMMA = 2;
constexpr size_t K_INPUT_IDX_BETA = 3;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t SIZE_OF_FLOAT = 4;
constexpr uint64_t SIZE_OF_HALF = 2;
constexpr uint64_t BASE_WSP_SIZE = 32;

static const std::unordered_map<ge::DataType, uint64_t> DTYPE_SIZE_MAP{
    {ge::DataType::DT_FLOAT, 4}, {ge::DataType::DT_FLOAT16, 2}, {ge::DataType::DT_BF16, 2}};

int64_t GetDTypeKey(ge::DataType tensorDtype, ge::DataType paramDtype)
{
    constexpr static int64_t LN_TENSOR_KEY_WEIGHT = 10;

    auto GetKeyForDType = [](ge::DataType dtype) -> int64_t {
        switch (dtype) {
            case ge::DT_FLOAT:
                return 0;
            case ge::DT_FLOAT16:
                return 1;
            case ge::DT_BF16:
                return 2;
            default:
                return -1;
        }
    };

    int64_t tensorKey = GetKeyForDType(tensorDtype);
    int64_t paramKey = GetKeyForDType(paramDtype);

    return tensorKey * LN_TENSOR_KEY_WEIGHT + paramKey;
}

bool LayerNormV4TilingBase::IsCapable()
{
    return true;
}

ge::graphStatus LayerNormV4TilingBase::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4TilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t LayerNormV4TilingBase::GetTilingKey() const
{
    return 0;
}

ge::graphStatus GetCommonPlatformInfo(
    gert::TilingContext* context, const LayerNormV4CompileInfo* compileInfo, ParamsLayerNomrV4& commonParams)
{
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context, "compile info is null"), return ge::GRAPH_FAILED);
    commonParams.coreNum = compileInfo->coreNum;
    commonParams.ubSizePlatForm = compileInfo->ubSizePlatForm;
    commonParams.blockSize = compileInfo->blockSize;
    commonParams.isAscend310P = compileInfo->isAscend310P;
    commonParams.isRegBase = compileInfo->isRegBase;
    commonParams.vlFp32 = compileInfo->vectorLength / sizeof(float);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4TilingBase::GetPlatformInfo()
{
    const LayerNormV4CompileInfo* compileInfo = context_->GetCompileInfo<LayerNormV4CompileInfo>();
    return GetCommonPlatformInfo(context_, compileInfo, commonParams);
}

ge::graphStatus GetCommonShapeAttrsInfo(
    gert::TilingContext* context, uint64_t colSize, uint64_t rowSize, ParamsLayerNomrV4& commonParams)
{
    OP_CHECK_IF(
        commonParams.colSize <= 0,
        OP_LOGE(context, "colSize must be greater than 0, colSize: %u", static_cast<uint32_t>(commonParams.colSize)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        commonParams.rowSize <= 0,
        OP_LOGE(context, "rowSize must be greater than 0, rowSize: %u", static_cast<uint32_t>(commonParams.rowSize)),
        return ge::GRAPH_FAILED);

    commonParams.colSize = colSize;
    commonParams.rowSize = rowSize;
    commonParams.coefficient = static_cast<float>(1.0) / static_cast<float>(commonParams.rowSize);
    uint64_t alignment = 16;
    if (DTYPE_SIZE_MAP.find(commonParams.tensorDtype) != DTYPE_SIZE_MAP.end()) {
        alignment = BLOCK_SIZE / DTYPE_SIZE_MAP.at(commonParams.tensorDtype);
    } else {
        OP_LOGE(context, "x dtype must be in float32, float16, bfloat16.");
        return ge::GRAPH_FAILED;
    }
    commonParams.rowAlign = (commonParams.rowSize + alignment - 1) / alignment * alignment;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4TilingBase::GetShapeAttrsInfo()
{
    std::string opType(context_->GetNodeType());
    commonParams.tensorDtype = context_->GetInputDesc(K_INPUT_IDX_X)->GetDataType();
    commonParams.paramDtype = ge::DT_FLOAT;
    auto gammaDesc = context_->GetOptionalInputDesc(K_INPUT_IDX_GAMMA);
    auto betaDesc = context_->GetOptionalInputDesc(K_INPUT_IDX_BETA);
    if (gammaDesc != nullptr) {
        commonParams.paramDtype = gammaDesc->GetDataType();
    } else if (betaDesc != nullptr) {
        commonParams.paramDtype = betaDesc->GetDataType();
    }
    commonParams.gammaNullPtr = (gammaDesc == nullptr ? 1 : 0);
    commonParams.betaNullPtr = (betaDesc == nullptr ? 1 : 0);
    commonParams.dtypeKey = GetDTypeKey(commonParams.tensorDtype, commonParams.paramDtype);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    if (attrs->GetFloat(0) != nullptr) {
        commonParams.eps = *(attrs->GetFloat(0));
    } else {
        OP_LOGE(context_->GetNodeName(), "eps is nullptr");
        return ge::GRAPH_FAILED;
    }
    
    uint64_t normalizedShapeLen;
    const gert::Shape xShape = context_->GetInputShape(K_INPUT_IDX_X)->GetStorageShape();
    const gert::Shape normalizedShape = context_->GetInputShape(K_INPUT_IDX_NORM_SHAPE)->GetStorageShape();
    OP_CHECK_IF(
        normalizedShape.GetDimNum() > 1,
        OP_LOGE(
            context_->GetNodeName(), "normalizedShape dim num must be 1, dim num: %u",
            static_cast<uint32_t>(normalizedShape.GetDimNum())),
        return ge::GRAPH_FAILED);
    normalizedShapeLen = normalizedShape.IsScalar() ? 1 : normalizedShape.GetDim(0);
    OP_CHECK_IF(
        static_cast<uint64_t>(xShape.GetDimNum()) < normalizedShapeLen,
        OP_LOGE(
            context_->GetNodeName(),
            "normalizedShape dim num must be less than xShape dim num, xShape dim num: %u, "
            "normalizedShape dim num: %u",
            static_cast<uint32_t>(xShape.GetDimNum()), static_cast<uint32_t>(normalizedShapeLen)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        static_cast<int64_t>(normalizedShapeLen) < 0,
        OP_LOGE(
            context_->GetNodeName(), "normalizedShapeLen must be greater than 0, normalizedShapeLen: %u",
            static_cast<uint32_t>(normalizedShapeLen)),
        return ge::GRAPH_FAILED);

    // fuse axis
    uint64_t colSize = 1;
    uint64_t rowSize = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        if (i < xShape.GetDimNum() - normalizedShapeLen) {
            colSize *= xShape.GetDim(i);
        } else {
            rowSize *= xShape.GetDim(i);
        }
    }
    return GetCommonShapeAttrsInfo(context_, colSize, rowSize, commonParams);
}

ge::graphStatus LayerNormV4TilingBase::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = BASE_WSP_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4TilingBase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
