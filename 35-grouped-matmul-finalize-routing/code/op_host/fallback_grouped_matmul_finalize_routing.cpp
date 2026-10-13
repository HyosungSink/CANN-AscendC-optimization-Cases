/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>

#include "../common/include/fallback/fallback.h"
#include "../common/include/fallback/fallback_comm.h"
#include "log/log.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;

constexpr size_t INDEX_INPUT_X = 0;
constexpr size_t INDEX_INPUT_WEIGHT = 1;
constexpr size_t INDEX_INPUT_WEIGHT_SCALE = 2;
constexpr size_t INDEX_INPUT_BIAS = 3;
constexpr size_t INDEX_INPUT_X_SCALE = 4;
constexpr size_t INDEX_INPUT_GROUP_LIST = 5;
constexpr size_t INDEX_INPUT_SHARED_INPUT = 6;
constexpr size_t INDEX_INPUT_LOGIT = 7;
constexpr size_t INDEX_INPUT_ROW_INDEX = 8;
constexpr size_t INDEX_INPUT_OFFSET = 9;
constexpr size_t INDEX_OUTPUT_Y = 0;
constexpr size_t INDEX_ATTR_DTYPE = 0;
constexpr size_t INDEX_ATTR_SHAREDINPUT_WEIGHT = 1;
constexpr size_t INDEX_ATTR_SHAREDINPUT_OFFSET = 2;
constexpr size_t INDEX_ATTR_TRANSPOSE_X = 3;
constexpr size_t INDEX_ATTR_TRANSPOSE_WEIGHT = 4;
constexpr size_t INDEX_ATTR_GROUP_LIST_TYPE = 6;
constexpr size_t MXFP_TPYEM_SCALE_DIM_NUM = 4;
constexpr size_t TYPEM_SCALE_K_DIM_INDEX = 1;
constexpr size_t TYPEM_SCALE_N_DIM_INDEX = 2;
constexpr size_t DIM_TWO = 2;


static bool SetShapeAndStrideForMXFPTypeMForGMMFinalizeRouting(std::vector<int64_t> &viewShape,
                                                               std::vector<int64_t> &strides)
{
    OP_CHECK_IF(viewShape.size() < MXFP_TPYEM_SCALE_DIM_NUM,
                OP_LOGE("GroupedMatmulFinalizeRouting aclnnfallback",
                        "When split m, the dim num should be greater than 3 in mx quant mode, but the actual is %zu.",
                        viewShape.size()),
                return false);
    // for GMMFinalizeRouting weightScale, when it is transposed, should swap the dim1 and dim2
    // swap strides
    std::swap(strides[TYPEM_SCALE_K_DIM_INDEX], strides[TYPEM_SCALE_N_DIM_INDEX]);
    // swap viewShape
    std::swap(viewShape[TYPEM_SCALE_K_DIM_INDEX], viewShape[TYPEM_SCALE_N_DIM_INDEX]);
    return true;
}

static bool SetShapeAndStrideForLastTwoDimForGMMFinalizeRouting(std::vector<int64_t> &viewShape,
                                                                std::vector<int64_t> &strides)
{
    OP_CHECK_IF(viewShape.size() < DIM_TWO,
                OP_LOGE("GroupedMatmulFinalizeRouting aclnnfallback",
                        "The dim num should be greater than 2, but the actual is %zu.", viewShape.size()),
                return false);
    // when tensor is transposed, last two dims in strides and viewShape should swap
    auto dimM = viewShape.size() - DIM_TWO;
    auto dimN = viewShape.size() - 1;
    std::swap(strides[dimM], strides[dimN]);
    // swap viewShape
    std::swap(viewShape[dimM], viewShape[dimN]);
    return true;
}

static void InitShapeAndStrideForGMMFinalizeRouting(const gert::Shape &originShape, std::vector<int64_t> &viewShape,
                                                    std::vector<int64_t> &strides)
{
    for (size_t i = 0; i < originShape.GetDimNum(); ++i) {
        viewShape.push_back(originShape.GetDim(i));
    }
    strides.resize(viewShape.size(), 1);
    // compute the strides of contiguous tensor
    for (int64_t i = viewShape.size() - DIM_TWO; i >= 0; i--) {
        strides[i] = viewShape[i + 1] * strides[i + 1];
    }
}

static bool IsFP8OrFP4BitsDataTypeForGMMFinalizeRouting(ge::DataType dataType)
{
    return dataType == ge::DataType::DT_FLOAT8_E4M3FN || dataType == ge::DataType::DT_FLOAT8_E5M2 ||
           dataType == ge::DataType::DT_FLOAT8_E8M0 || dataType == ge::DataType::DT_FLOAT4_E2M1;
}


static inline aclDataType ToAclDataTypeForGMMFinalizeRouting(ge::DataType dtype)
{
    static const std::vector<DataType> GMMWsiglu_CONVERT_TO_ACL_DataType_LIST = {
        ge::DataType::DT_FLOAT8_E4M3FN, ge::DataType::DT_FLOAT8_E5M2, ge::DataType::DT_FLOAT8_E8M0, ge::DataType::DT_FLOAT4_E2M1};
    auto iter =
        std::find(GMMWsiglu_CONVERT_TO_ACL_DataType_LIST.begin(), GMMWsiglu_CONVERT_TO_ACL_DataType_LIST.end(), dtype);
    if (iter == GMMWsiglu_CONVERT_TO_ACL_DataType_LIST.end()) {
        return aclDataType::ACL_DT_UNDEFINED;
    }
    return static_cast<aclDataType>(dtype);
}

static inline aclTensor *GeTensor2AclTensor(const gert::Tensor *geTensor, bool enableTranspose, size_t index,
                                            bool enableNZ = false)
{
    OP_CHECK_IF(geTensor == nullptr, OP_LOGE("GroupedMatmulFinalizeRouting aclnnfallback", "The geTensor nullptr"),
                return nullptr);
    auto storageShape = geTensor->GetStorageShape();
    if (storageShape.GetDimNum() <= 1) {
        return ConvertType(geTensor);
    }
    std::vector<int64_t> storageShapeVec;
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        storageShapeVec.push_back(storageShape.GetDim(i));
    }

    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    OP_CHECK_IF(aclCreateTensor == nullptr,
                OP_LOGE("GroupedMatmulFinalizeRouting aclnnfallback", "The aclCreateTensor nullptr"), return nullptr);

    void *deviceAddr = const_cast<void *>(geTensor->GetAddr());
    // convert data type
    auto dataTypeGE = geTensor->GetDataType();
    aclDataType dataType = ACL_DT_UNDEFINED;
    if (IsFP8OrFP4BitsDataTypeForGMMFinalizeRouting(dataTypeGE)) {
        dataType = ToAclDataTypeForGMMFinalizeRouting(dataTypeGE);
    } else {
        dataType = ToAclDataType(dataTypeGE);
    }
    // convert view shape
    const gert::Shape &origin_shape = geTensor->GetOriginShape();
    std::vector<int64_t> viewShape;
    std::vector<int64_t> strides;
    InitShapeAndStrideForGMMFinalizeRouting(origin_shape, viewShape, strides);
    if (enableTranspose) {
        if (dataTypeGE == ge::DataType::DT_FLOAT8_E8M0) {
            if (index == INDEX_INPUT_WEIGHT_SCALE) {
                OP_CHECK_IF(!SetShapeAndStrideForMXFPTypeMForGMMFinalizeRouting(viewShape, strides),
                            OP_LOGE("GroupedMatmulFinalizeRouting aclnnfallback",
                                    "SetShapeAndStrideForMXFPTypeMForGMMFinalizeRouting failed"), return nullptr);
            }
        } else {
            OP_CHECK_IF(!SetShapeAndStrideForLastTwoDimForGMMFinalizeRouting(viewShape, strides),
                        OP_LOGE("GroupedMatmulFinalizeRouting aclnnfallback",
                                "SetShapeAndStrideForLastTwoDimForGMMFinalizeRouting failed"), return nullptr);
        }
    }
    auto aclFormat = aclFormat::ACL_FORMAT_ND;
    if (enableNZ && GetPrimaryFormat(geTensor->GetStorageFormat()) == ge::Format::FORMAT_FRACTAL_NZ) {
        aclFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    }
    aclTensor *out = aclCreateTensor(viewShape.data(), viewShape.size(), dataType, strides.data(), 0, aclFormat,
                                     storageShapeVec.data(), storageShapeVec.size(), deviceAddr);
    OP_CHECK_IF(out == nullptr, OP_LOGE("GroupedMatmulFinalizeRouting aclnnfallback",
                "The GeTensor2AclTensor out nullptr"), return nullptr);
    return out;
}

static void PrepareAclTensor(const OpExecuteContext *host_api_ctx, const aclTensor *&tensor, size_t index,
                                    bool enableTranspose, bool enableNZ)
{
    if (index == INDEX_INPUT_X || index == INDEX_INPUT_WEIGHT) {
        tensor = GeTensor2AclTensor(host_api_ctx->GetInputTensor(index), enableTranspose, index, enableNZ);
    } else {
        tensor = GeTensor2AclTensor(host_api_ctx->GetOptionalInputTensor(index), enableTranspose, index, enableNZ);
    }
    if (tensor == nullptr) {
        OP_LOGE("GroupedMatmulFinalizeRouting aclnnfallback", "PrepareAclTensor failed");
    }
    return;
}

static void PrepareOutputTensor(const OpExecuteContext *host_api_ctx, const gert::Tensor *&tensor, size_t index)
{
    tensor = host_api_ctx->GetOutputTensor(index);
    if (tensor == nullptr) {
        OP_LOGE("GroupedMatmulFinalizeRouting aclnnfallback", "PrepareOutputTensor failed");
    }
    return;
}

static graphStatus GroupedMatmulFinalizeRoutingExecuteFunc(OpExecuteContext *host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("GroupedMatmulFinalizeRouting aclnnfallback", "The host_api_ctx is null"), return GRAPH_FAILED);
    auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("GroupedMatmulFinalizeRouting aclnnfallback", "The attrs is null"), return GRAPH_FAILED);
    const int64_t* dtypeGe = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DTYPE);
    OP_CHECK_IF(dtypeGe == nullptr, OP_LOGE("GroupedMatmulFinalizeRouting aclnnfallback", "The dtypeGe is null"), return GRAPH_FAILED);
    const float* sharedInputWeightGe = attrs->GetAttrPointer<float>(INDEX_ATTR_SHAREDINPUT_WEIGHT);
    OP_CHECK_IF(sharedInputWeightGe == nullptr, OP_LOGE("GroupedMatmulFinalizeRouting aclnnfallback", "The sharedInputWeightGe is null"), return GRAPH_FAILED);
    const int64_t* sharedInputOffsetGe = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SHAREDINPUT_OFFSET);
    OP_CHECK_IF(sharedInputOffsetGe == nullptr, OP_LOGE("GroupedMatmulFinalizeRouting aclnnfallback", "The sharedInputOffsetGe is null"), return GRAPH_FAILED);
    const bool*  transXGe = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANSPOSE_X);
    OP_CHECK_IF(transXGe == nullptr, OP_LOGE("aclnnfallback", "transXGe is null"), return GRAPH_FAILED);
    const bool*  transWeightGe = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANSPOSE_WEIGHT);
    OP_CHECK_IF(transWeightGe == nullptr, OP_LOGE("aclnnfallback", "transWeightGe is null"), return GRAPH_FAILED);
    const int64_t* groupListTypeGe = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_GROUP_LIST_TYPE);
    OP_CHECK_IF(groupListTypeGe == nullptr, OP_LOGE("GroupedMatmulFinalizeRouting aclnnfallback", "The groupListTypeGe is null"), return GRAPH_FAILED);

    const aclTensor *aclTensorX = nullptr;
    PrepareAclTensor(host_api_ctx, aclTensorX, INDEX_INPUT_X, false, false);

    const aclTensor *aclTensorWeight = nullptr;
    PrepareAclTensor(host_api_ctx, aclTensorWeight, INDEX_INPUT_WEIGHT, false, false);

    const aclTensor *aclTensorWeightScale = nullptr;
    PrepareAclTensor(host_api_ctx, aclTensorWeightScale, INDEX_INPUT_WEIGHT_SCALE, false, false);

    const aclTensor *aclTensorBias = nullptr;
    PrepareAclTensor(host_api_ctx, aclTensorBias, INDEX_INPUT_BIAS, false, false);

    const aclTensor *aclTensorXScale = nullptr;
    PrepareAclTensor(host_api_ctx, aclTensorXScale, INDEX_INPUT_X_SCALE, false, false);

    const aclTensor *aclTensorGroupList = nullptr;
    PrepareAclTensor(host_api_ctx, aclTensorGroupList, INDEX_INPUT_GROUP_LIST, false, false);

    const aclTensor *aclTensorSharedInput = nullptr;
    PrepareAclTensor(host_api_ctx, aclTensorSharedInput, INDEX_INPUT_SHARED_INPUT, false, false);

    const aclTensor *aclTensorLogit = nullptr;
    PrepareAclTensor(host_api_ctx, aclTensorLogit, INDEX_INPUT_LOGIT, false, false);

    const aclTensor *aclTensorRowIndex = nullptr;
    PrepareAclTensor(host_api_ctx, aclTensorRowIndex, INDEX_INPUT_ROW_INDEX, false, false);

    const aclTensor *aclTensorOffset = nullptr;
    PrepareAclTensor(host_api_ctx, aclTensorOffset, INDEX_INPUT_OFFSET, false, false);

    const gert::Tensor *geTensorY = nullptr;
    PrepareOutputTensor(host_api_ctx, geTensorY, INDEX_OUTPUT_Y);

    aclTensor *aclTensorAntiQuantScaleOptional = nullptr;
    aclTensor *aclTensorAntiQunatOffsetOptional = nullptr;
    aclIntArray *tuningConfig = nullptr;

    // execute opapi
    auto apiRet = EXEC_OPAPI_CMD(
        aclnnGroupedMatmulFinalizeRoutingV3, aclTensorX, aclTensorWeight, aclTensorWeightScale, aclTensorBias,
        aclTensorOffset, aclTensorAntiQuantScaleOptional, aclTensorAntiQunatOffsetOptional, aclTensorXScale,
        aclTensorGroupList, aclTensorSharedInput, aclTensorLogit, aclTensorRowIndex, *dtypeGe, *sharedInputWeightGe,
        *sharedInputOffsetGe, *transXGe, *transWeightGe, *groupListTypeGe, tuningConfig, geTensorY);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("GroupedMatmulFinalizeRouting aclnnfallback",
                "The apiRet failed:%u", apiRet), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

IMPL_OP(GroupedMatmulFinalizeRouting).OpExecuteFunc(GroupedMatmulFinalizeRoutingExecuteFunc);

} // namespace fallback

#ifdef __cplusplus
}
#endif
