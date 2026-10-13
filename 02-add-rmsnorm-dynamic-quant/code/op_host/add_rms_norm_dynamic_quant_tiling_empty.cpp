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
 * \file add_rms_norm_dynamic_quant_tiling_arch35.cpp
 * \brief
 */

#include "add_rms_norm_dynamic_quant_tiling.h"
#include "../norm_common/op_host/norm_tiling_check_common.h"

namespace optiling {
using namespace NormCheck;

static constexpr uint64_t X1_INDEX = 0;
static constexpr uint64_t X2_INDEX = 1;
static constexpr uint64_t GAMMA_INDEX = 2;
static constexpr uint64_t SMOOTH_SCALE1_INDEX = 3;
static constexpr uint64_t SMOOTH_SCALE2_INDEX = 4;
static constexpr uint64_t Y1_INDEX = 0;
static constexpr uint64_t Y2_INDEX = 1;
static constexpr uint64_t X_INDEX = 2;
static constexpr uint64_t SCALE1_INDEX = 3;
static constexpr uint64_t SCALE2_INDEX = 4;

static constexpr uint32_t TWO = 2;
static constexpr uint32_t MAX_DIM_CNT = 8;
static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t FLOATBYTESIZE = 4;
static constexpr uint32_t MIN_WORKSPACE_SIZE = 16 * 1024 * 1024;

static constexpr uint64_t PER_CORE_MAX_SIZE = 4096;
static constexpr uint32_t EMPTY_TENSOR_KEY = 500;

const std::string OP_NAME = "AddRmsNormDynamicQuant";

ge::graphStatus AddRmsNormDynamicQuantEmptyTiling::CheckDtypeVaild(
    ge::DataType& srcDtype, std::vector<ge::DataType>& supportDtypeList, string srcName)
{
    for (const auto& supportedDtype : supportDtypeList) {
        if (supportedDtype == srcDtype) {
            return ge::GRAPH_SUCCESS;
        }
    }
    OP_LOGE(
        nodeName_.c_str(), "Dtype check invalid, %s dtype is %s, not in supportDtypeList.", srcName.c_str(),
        Ops::Base::ToString(srcDtype).c_str());
    return ge::GRAPH_FAILED;
}

bool AddRmsNormDynamicQuantEmptyTiling::CheckShapeNull()
{
    OP_LOGD(nodeName_.c_str(), "Enter AddRmsNormDynamicQuantEmptyTiling CheckShapeNull.");

    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    const gert::StorageShape* xShape = context_->GetOutputShape(X_INDEX);

    OP_CHECK_IF(
        (nullptr == x1Shape) || (nullptr == x2Shape) || (nullptr == gammaShape) || (nullptr == xShape), , return false);
    return true;
}

bool AddRmsNormDynamicQuantEmptyTiling::CheckOptionalInput()
{
    OP_LOGD(nodeName_.c_str(), "Enter AddRmsNormDynamicQuantEmptyTiling CheckOptionalInput.");
    const gert::StorageShape* smoothScale1Shape = context_->GetOptionalInputShape(SMOOTH_SCALE1_INDEX);
    const gert::StorageShape* smoothScale2Shape = context_->GetOptionalInputShape(SMOOTH_SCALE2_INDEX);

    if (smoothScale1Shape != nullptr) {
        hasSmoothScale1_ = true;
    }
    if (smoothScale2Shape != nullptr) {
        hasSmoothScale2_ = true;
    }

    OP_CHECK_IF(
        !hasSmoothScale1_ && hasSmoothScale2_,
        OP_LOGE(nodeName_.c_str(), "When input smoothScale2, must input smoothScale1."), return false);
    return true;
}

bool AddRmsNormDynamicQuantEmptyTiling::CheckInputShapeDim()
{
    OP_LOGD(nodeName_.c_str(), "Enter AddRmsNormDynamicQuantEmptyTiling CheckInputShapeDim.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    auto x1InputShape = optiling::EnsureNotScalar(x1Shape->GetStorageShape());
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);
    auto x2InputShape = optiling::EnsureNotScalar(x2Shape->GetStorageShape());

    // Not support zero shape.
    size_t x1DimNum = x1InputShape.GetDimNum();
    size_t x2DimNum = x2InputShape.GetDimNum();
    OP_CHECK_IF(
        (x1DimNum > MAX_DIM_CNT) || (x2DimNum > MAX_DIM_CNT),
        OP_LOGE(nodeName_.c_str(), "Input x1/x2 dim should not bigger than %u.", MAX_DIM_CNT), return false);
    OP_CHECK_IF(!CheckDimBiggerZero(x1Shape, x1DimNum, nodeName_, "x1"), , return false);
    OP_CHECK_IF(!CheckDimBiggerZero(x2Shape, x2DimNum, nodeName_, "x2"), , return false);
    return true;
}

bool AddRmsNormDynamicQuantEmptyTiling::CheckInputShapeValue()
{
    OP_LOGD(nodeName_.c_str(), "Enter AddRmsNormDynamicQuantEmptyTiling CheckInputShapeValue.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    const gert::StorageShape* smoothScale1Shape = context_->GetOptionalInputShape(SMOOTH_SCALE1_INDEX);
    const gert::StorageShape* smoothScale2Shape = context_->GetOptionalInputShape(SMOOTH_SCALE2_INDEX);
    const gert::StorageShape* xShape = context_->GetOutputShape(X_INDEX);

    // Check x1&x2&y1&y2&x's shape should be equal
    if (!NormCheck::CheckShapeSame(x1Shape, x2Shape, nodeName_, "x1", "x2")) {
        return false;
    };
    if (!NormCheck::CheckShapeSame(x1Shape, xShape, nodeName_, "x1", "x")) {
        return false;
    };

    // Check smoothScale&scale's shape should be equal
    if (hasSmoothScale1_ &&
        !NormCheck::CheckShapeSame(gammaShape, smoothScale1Shape, nodeName_, "gamma", "smoothScale1")) {
        return false;
    };
    if (hasSmoothScale2_ &&
        !NormCheck::CheckShapeSame(gammaShape, smoothScale2Shape, nodeName_, "gamma", "smoothScale2")) {
        return false;
    };

    // Check gamma should be last dim of x
    // Check scale should be not last dim of x
    auto gammaInputShape = optiling::EnsureNotScalar(gammaShape->GetStorageShape());
    if ((1 == gammaInputShape.GetDimNum()) &&
        !NormCheck::CheckShapeBC(x1Shape, gammaShape, nodeName_, "x1", "gamma", true)) {
        return false;
    };

    return true;
}

bool AddRmsNormDynamicQuantEmptyTiling::CheckInputDtype()
{
    OP_LOGD(nodeName_.c_str(), "Enter AddRmsNormDynamicQuantEmptyTiling CheckInputDtype.");
    std::vector<ge::DataType> supportedXGammaDtypes = {ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16};
    std::vector<ge::DataType> supportedSmoothScaleDtypes = {ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16};

    ge::DataType x1Dtype = context_->GetInputTensor(X1_INDEX)->GetDataType();
    ge::DataType x2Dtype = context_->GetInputTensor(X2_INDEX)->GetDataType();
    ge::DataType gammaDtype = context_->GetInputTensor(GAMMA_INDEX)->GetDataType();
    ge::DataType smoothScale1Dtype = ge::DT_FLOAT;
    ge::DataType smoothScale2Dtype = ge::DT_FLOAT;
    if (hasSmoothScale1_) {
        smoothScale1Dtype = context_->GetOptionalInputTensor(SMOOTH_SCALE1_INDEX)->GetDataType();
    }
    if (hasSmoothScale2_) {
        smoothScale2Dtype = context_->GetOptionalInputTensor(SMOOTH_SCALE2_INDEX)->GetDataType();
    }
    if ((x1Dtype != x2Dtype) || (x1Dtype != gammaDtype)) {
        OP_LOGE(nodeName_.c_str(), "Input x1/gamma/xout dtype should be equal.");
        return false;
    }
    if (hasSmoothScale1_ && (x1Dtype != smoothScale1Dtype)) {
        OP_LOGE(nodeName_.c_str(), "Input smoothScale1/x1 dtype should be equal.");
        return false;
    }
    if (hasSmoothScale2_ && (x1Dtype != smoothScale2Dtype)) {
        OP_LOGE(nodeName_.c_str(), "Input smoothScale2/x1 dtype should be equal.");
        return false;
    }

    return true;
}

ge::graphStatus AddRmsNormDynamicQuantEmptyTiling::SetInputParams()
{
    OP_LOGD(nodeName_.c_str(), "Enter AddRmsNormDynamicQuantEmptyTiling SetInputParams.");
    // Set input dim
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    auto x1InputShape = optiling::EnsureNotScalar(x1Shape->GetStorageShape());
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    auto gammaInputShape = optiling::EnsureNotScalar(gammaShape->GetStorageShape());

    size_t x1DimNum = x1InputShape.GetDimNum();
    size_t gammaDimNum = gammaInputShape.GetDimNum();
    uint64_t numM = 1;
    uint64_t numN = 0;
    for (size_t i = 0; i < x1DimNum - gammaDimNum; i++) {
        numM *= x1InputShape.GetDim(i);
    }
    
    numM_ = numM;
    numN_ = numN;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicQuantEmptyTiling::GetShapeAttrsInfo()
{
    OP_LOGD(nodeName_.c_str(), "Enter AddRmsNormDynamicQuantEmptyTiling GetShapeAttrsInfo.");
    OP_CHECK_IF(
        !CheckShapeNull(), OP_LOGE(nodeName_.c_str(), "The not optional input is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckOptionalInput(), OP_LOGE(nodeName_.c_str(), "The optional input is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckInputShapeDim(), OP_LOGE(nodeName_.c_str(), "The input shape dim is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckInputShapeValue(), OP_LOGE(nodeName_.c_str(), "The input shape relationship is invalid."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputDtype(), OP_LOGE(nodeName_.c_str(), "The input dtype is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ge::GRAPH_SUCCESS != SetInputParams(), OP_LOGE(nodeName_.c_str(), "Set input shape failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicQuantEmptyTiling::GetPlatformInfo()
{
    OP_LOGD(nodeName_.c_str(), "Enter AddRmsNormDynamicQuantEmptyTiling GetPlatformInfo.");
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = ubSizePlatform;
    }
    return ge::GRAPH_SUCCESS;
}

bool AddRmsNormDynamicQuantEmptyTiling::IsCapable()
{
    return true;
}

ge::graphStatus AddRmsNormDynamicQuantEmptyTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicQuantEmptyTiling::GetWorkspaceSize()
{
    workspaceSize_ = MIN_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicQuantEmptyTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t AddRmsNormDynamicQuantEmptyTiling::GetTilingKey() const
{
    uint64_t tilingKey = 500;
    return tilingKey;
}

uint64_t AddRmsNormDynamicQuantEmptyTiling::NearestLowerPowerOfTwo(int32_t tmp)
{
    uint64_t power = 0;
    uint64_t powerofTwoValue = 0;
    while (power <= tmp) {
        powerofTwoValue += 1;
        power = std::pow(TWO, powerofTwoValue);
    }
    return powerofTwoValue - 1;
}

ge::graphStatus AddRmsNormDynamicQuantEmptyTiling::CalcTilingData()
{
    if (static_cast<int64_t>(ubSize_) >= BUFFER_NUM * (mPerCore_ * FLOATBYTESIZE)) {
        coreUbBlockCount_ = 0;
        mTailUb_ = mPerCore_;
        lastCoreBlockCount_ = 0;
        mlastCoreTailUb_ = mLastCore_;
    } else {
        uint64_t maxRowsNum_ = ubSize_ / (BUFFER_NUM * FLOATBYTESIZE);
        mPerUB_ = std::pow(TWO, NearestLowerPowerOfTwo(maxRowsNum_));
        coreUbBlockCount_ = (mPerCore_ + mPerUB_ -1) / mPerUB_ - 1;
        mTailUb_ = mPerCore_ - mPerUB_ * coreUbBlockCount_;
        if (mPerUB_ > mLastCore_) {
            lastCoreBlockCount_ = 0;
            mlastCoreTailUb_ = mLastCore_;
        } else {
            lastCoreBlockCount_ = (mLastCore_ + mPerUB_ -1) / mPerUB_ - 1;
            mlastCoreTailUb_ = mLastCore_ - mPerUB_ * lastCoreBlockCount_;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void AddRmsNormDynamicQuantEmptyTiling::CalcUsedCoreNum()
{
    if (numM_ <= PER_CORE_MAX_SIZE) {
        usedCoreNum_ = 1;
        mPerCore_ = numM_;
        mLastCore_ = numM_;
        coreUbBlockCount_ = 0;
        mTailUb_ = mPerCore_;
        mPerUB_ = mPerCore_;
        lastCoreBlockCount_ = 0;
        mlastCoreTailUb_ = mLastCore_;
    } else {
        ge::graphStatus result = ge::GRAPH_SUCCESS;
        usedCoreNum_ = aivCoreNum_;
        mPerCore_ = Ops::Base::CeilDiv(numM_, usedCoreNum_);
        mPerUB_ = mPerCore_;
        mLastCore_ = numM_ - mPerCore_ * (usedCoreNum_ - 1);
        result = CalcTilingData();
    }
}

void AddRmsNormDynamicQuantEmptyTiling::LogTilingResult()
{
    OP_LOGI(OP_NAME, "numN: %ld, numM: %ld", numN_, numM_);
}

ge::graphStatus AddRmsNormDynamicQuantEmptyTiling::DoOpTiling()
{
    tilingKey_ = EMPTY_TENSOR_KEY;
    // split cores
    CalcUsedCoreNum();

    tilingData_.set_mPerUB(mPerUB_);
    tilingData_.set_mTailUb(mTailUb_);
    tilingData_.set_coreUbBlockCount(coreUbBlockCount_);
    tilingData_.set_lastCoreBlockCount(lastCoreBlockCount_);
    tilingData_.set_mlastCoreTailUb(mlastCoreTailUb_);
    tilingData_.set_ubSize(ubSize_);
    tilingData_.set_numM(numM_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_mPerCore(mPerCore_);
    tilingData_.set_mLastCore(mLastCore_);
    tilingData_.set_hasSmoothScale1(hasSmoothScale1_);
    tilingData_.set_hasSmoothScale2(hasSmoothScale2_);
    LogTilingResult();
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
