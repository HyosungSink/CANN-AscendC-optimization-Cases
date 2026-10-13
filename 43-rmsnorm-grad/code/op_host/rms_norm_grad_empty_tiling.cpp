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
 * \file rms_norm_grad_empty_tiling.cpp
 * \brief RmsNormGrad empty tiling file
 */

#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"
#include "rms_norm_grad_tiling.h"

namespace optiling {
static constexpr uint32_t MIN_WORKSPACE_SIZE = 16 * 1024 * 1024;
static constexpr int64_t INPUT_NUM = 4;

static constexpr int64_t INPUT_INDEX_0 = 0;
static constexpr int64_t INPUT_INDEX_1 = 1;
static constexpr int64_t INPUT_INDEX_2 = 2;
static constexpr int64_t INPUT_INDEX_3 = 3;
static constexpr int64_t TWO = 2;
static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t FLOATBYTESIZE = 4;
static constexpr int64_t MAX_CORE_COLS = 8192;
static constexpr uint32_t EMPTY_TENSOR_KEY = 8000;
static const gert::Shape g_vec_1_shape = {1};

const std::string OP_NAME = "RmsNormGrad";


ge::graphStatus RmsNormGradEmptyTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo(); // 判断条件修改
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = ubSizePlatform;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradEmptyTiling::CheckShapeAllPositive(gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(
            shape.GetDim(i) < 0,
            OP_LOGE(
                context_->GetNodeName(), "Dim %lu of input should be positive, but actual %ld.", i, shape.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradEmptyTiling::CheckShapesEqual(gert::Shape& shape0, gert::Shape& shape1)
{
    OP_CHECK_IF(
        shape0.GetDimNum() != shape1.GetDimNum(),
        OP_LOGE(
            context_->GetNodeName(), "DimNum of shapes are not equal: %lu vs %lu", shape0.GetDimNum(),
            shape1.GetDimNum()),
        return ge::GRAPH_FAILED);

    for (size_t i = 0; i < shape0.GetDimNum(); i++) {
        OP_CHECK_IF(
            shape0.GetDim(i) != shape1.GetDim(i),
            OP_LOGE(
                context_->GetNodeName(), "Dim %lu of shapes are not equal: %ld vs %ld", i, shape0.GetDim(i),
                shape1.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

const gert::Shape& RmsNormGradEmptyTiling::EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}

void RmsNormGradEmptyTiling::CalcRowsAndCols(gert::Shape& gammaShape)
{
    rows_ = 0;
    cols_ = 1;
    for (size_t i = 0; i < gammaShape.GetDimNum(); i++) {
        cols_ *= gammaShape.GetDim(i);
    }
}

ge::graphStatus RmsNormGradEmptyTiling::CheckInputsShape()
{
    // check dy
    auto inputShape = context_->GetInputShape(INPUT_INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape0 = EnsureNotScalar(inputShape->GetStorageShape());
    if (CheckShapeAllPositive(storageShape0) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "dy shape contains negative.");
        return ge::GRAPH_FAILED;
    }

    // check x
    inputShape = context_->GetInputShape(INPUT_INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape1 = EnsureNotScalar(inputShape->GetStorageShape());
    if (CheckShapeAllPositive(storageShape1) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "x shape contains negative.");
        return ge::GRAPH_FAILED;
    }

    // check shapes of input0 and input1 are equal
    if (CheckShapesEqual(storageShape0, storageShape1) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Shapes of x1 and x2 are not equal.");
        return ge::GRAPH_FAILED;
    }

    // check rstd
    inputShape = context_->GetInputShape(INPUT_INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape2 = EnsureNotScalar(inputShape->GetStorageShape());
    if (CheckShapeAllPositive(storageShape2) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "gamma shape contains negative.");
        return ge::GRAPH_FAILED;
    }

    // check gamma
    inputShape = context_->GetInputShape(INPUT_INDEX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto storageShape3 = EnsureNotScalar(inputShape->GetStorageShape());
    if (CheckShapeAllPositive(storageShape3) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "beta shape contains negative.");
        return ge::GRAPH_FAILED;
    }

    CalcRowsAndCols(storageShape3);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradEmptyTiling::CheckInputsDtypeAndFormat()
{
    for (int i = 0; i < INPUT_NUM; i++) {
        auto inputDesc = context_->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
        // check format
        auto format = inputDesc->GetFormat().GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGE(context_->GetNodeName(), "Input %d only supports ND format.", i);
            return ge::GRAPH_FAILED;
        }
        // check dtype
        auto dtype = inputDesc->GetDataType();
        if (dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_BF16 && dtype != ge::DataType::DT_FLOAT) {
            OP_LOGE(context_->GetNodeName(), "Input %d only supports float16, bfloat16, float32.", i);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradEmptyTiling::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE(OP_NAME, "Tiling context is null");
        return ge::GRAPH_FAILED;
    }

    // check inputs shape
    if (CheckInputsShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Inputs shape invalid.");
        return ge::GRAPH_FAILED;
    }
    // check inputs dtype and format
    if (CheckInputsDtypeAndFormat() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Inputs dtype/format invalid.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool RmsNormGradEmptyTiling::IsCapable()
{
    return true;
}

uint64_t RmsNormGradEmptyTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus RmsNormGradEmptyTiling::PostTiling()
{
    context_->SetBlockDim(aivCoreNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradEmptyTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradEmptyTiling::GetWorkspaceSize()
{
    workspaceSize_ = MIN_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormGradEmptyTiling::CalcUsedCoreNumGamma()
{
    if (cols_ <= MAX_CORE_COLS) {
        // 单核计算单colset
        usedCoreNumDG_ = 1;
        colsPerCoreDG_ = cols_;
        colsLastCoreDG_ = cols_;
        coreUbBlockCount_ = 0;
        tailUbCols_ = colsPerCoreDG_;
        colsPerUBDG_ = colsPerCoreDG_;
        lastCoreBlockCount_ = 0;
        lastCoreTailUbCols_ = colsLastCoreDG_;
    } else {
        // 多核计算
        ge::graphStatus result = ge::GRAPH_SUCCESS;
        usedCoreNumDG_ = aivCoreNum_;
        colsPerCoreDG_ = Ops::Base::CeilDiv(cols_, usedCoreNumDG_);
        colsPerUBDG_ = colsPerCoreDG_;
        colsLastCoreDG_ = cols_ - colsPerCoreDG_ * (usedCoreNumDG_ - 1);
        result = CalcTilingDataDgamma();
        return result;
    }
    return ge::GRAPH_SUCCESS;
}

int32_t RmsNormGradEmptyTiling::NearestLowerPowerOfTwo(int32_t temp)
{
    int64_t power = 0;
    int32_t powerofTwoValueDG = 0;
    while (power <= temp) {
        powerofTwoValueDG += 1;
        power = std::pow(TWO, powerofTwoValueDG);
    }
    return powerofTwoValueDG - 1;
}

ge::graphStatus RmsNormGradEmptyTiling::CalcTilingDataDgamma()
{
    if (static_cast<uint64_t>(ubSize_) >= BUFFER_NUM * (colsPerCoreDG_ * FLOATBYTESIZE)) {
        // full-load
        coreUbBlockCount_ = 0;
        tailUbCols_ = colsPerCoreDG_;
        lastCoreBlockCount_ = 0;
        lastCoreTailUbCols_ = colsLastCoreDG_;
    } else {
        //UB最多存放多少列
        //colsPerUBDG_ = ubSize_ / (BUFFER_NUM * FLOATBYTESIZE);
        int maxRowsNumDG_ = ubSize_ / (BUFFER_NUM * FLOATBYTESIZE);
        colsPerUBDG_ = std::pow(TWO, NearestLowerPowerOfTwo(maxRowsNumDG_));
        OP_CHECK_IF(
            maxRowsNumDG_ <= 0, OP_LOGE(context_->GetNodeName(), "The maxRowsNumDG_ size is neg: %d.", maxRowsNumDG_),
            return ge::GRAPH_FAILED);
        coreUbBlockCount_ = (colsPerCoreDG_ + colsPerUBDG_ -1) / colsPerUBDG_ - 1;
        tailUbCols_ = colsPerCoreDG_ - colsPerUBDG_ * coreUbBlockCount_;
        if (colsPerUBDG_ > colsLastCoreDG_) {
            lastCoreBlockCount_ = 0;
            lastCoreTailUbCols_ = colsLastCoreDG_;
        } else {
            lastCoreBlockCount_ = (colsLastCoreDG_ + colsPerUBDG_ -1) / colsPerUBDG_ - 1;
            lastCoreTailUbCols_ = colsLastCoreDG_ - colsPerUBDG_ * lastCoreBlockCount_;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void RmsNormGradEmptyTiling::LogTilingResult()
{
    OP_LOGI(OP_NAME, "rows: %ld, cols_: %ld", rows_, cols_);
}

ge::graphStatus RmsNormGradEmptyTiling::DoOpTiling()
{
    tilingKey_ = EMPTY_TENSOR_KEY;
    // split cores
    ge::graphStatus result = CalcUsedCoreNumGamma();
    if (result != ge::GRAPH_SUCCESS) {
        return result;
    }

    tilingData_.set_colsPerUBDG(colsPerUBDG_);
    tilingData_.set_tailUbCols(tailUbCols_);
    tilingData_.set_coreUbBlockCount(coreUbBlockCount_);
    tilingData_.set_lastCoreBlockCount(lastCoreBlockCount_);
    tilingData_.set_lastCoreTailUbCols(lastCoreTailUbCols_);
    tilingData_.set_ubSize(ubSize_);
    tilingData_.set_cols(cols_);
    tilingData_.set_usedCoreNumDG(usedCoreNumDG_);
    tilingData_.set_colsPerCoreDG(colsPerCoreDG_);
    tilingData_.set_colsLastCoreDG(colsLastCoreDG_);
    LogTilingResult();
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling