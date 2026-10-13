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

#include "fallback/fallback.h"
#include "fallback/fallback_comm.h"
#include "log/log.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;

constexpr size_t INDEX_INPUT_X = 0UL;
constexpr size_t INDEX_INPUT_X_SCALE = 1UL;
constexpr size_t INDEX_INPUT_GROUP_LIST = 2UL;
constexpr size_t INDEX_INPUT_WEIGHT = 3UL;
constexpr size_t INDEX_INPUT_WEIGHT_SCALE = 4UL;
constexpr size_t INDEX_INPUT_WEIGHT_ASSIST_MATRIX = 5UL;
constexpr size_t INDEX_INPUT_BIAS = 6UL;
constexpr size_t INDEX_INPUT_SMOOTH_SCALE = 7UL;
constexpr size_t INDEX_OUTPUT_Y = 0UL;
constexpr size_t INDEX_OUTPUT_Y_SCALE = 1UL;
constexpr size_t INDEX_ATTR_DEQUANT_MODE = 0UL;
constexpr size_t INDEX_ATTR_DEQUANT_DTYPE = 1UL;
constexpr size_t INDEX_ATTR_QUANT_MODE = 2UL;
constexpr size_t INDEX_ATTR_TRANSPOSE_WEIGHT = 4UL;
constexpr size_t INDEX_ATTR_GROUP_LIST_TYPE = 5UL;
constexpr size_t INDEX_ATTR_TUNING_CONFIG = 6UL;

constexpr size_t MXFP_TPYEM_SCALE_DIM_NUM = 4UL;
constexpr size_t TYPEM_SCALE_K_DIM_INDEX = 1UL;
constexpr size_t TYPEM_SCALE_N_DIM_INDEX = 2UL;
constexpr size_t DIM_TWO = 2UL;


static bool SetShapeAndStrideForMXFPTypeMForGMMSwigluQuantV2(std::vector<int64_t> &viewShape, std::vector<int64_t> &strides) {
    OP_CHECK_IF(viewShape.size() < MXFP_TPYEM_SCALE_DIM_NUM,
                OP_LOGE("GroupedMatmulSwigluQuantV2 aclnnfallback",
                        "When split m, the dim num should be greater than 3 in mx quant mode, but the actual is %zu.",
                        viewShape.size()),
                return false);
    // for GMMSwigluQuantV2 weightScale, when it is transposed, should swap the dim1 and dim2
    // swap strides
    std::swap(strides[TYPEM_SCALE_K_DIM_INDEX], strides[TYPEM_SCALE_N_DIM_INDEX]);
    // swap viewShape
    std::swap(viewShape[TYPEM_SCALE_K_DIM_INDEX], viewShape[TYPEM_SCALE_N_DIM_INDEX]);
    return true;
}

static bool SetShapeAndStrideForLastTwoDimForGMMSwigluQuantV2(std::vector<int64_t> &viewShape, std::vector<int64_t> &strides)
{
    OP_CHECK_IF(viewShape.size() < DIM_TWO,
                OP_LOGE("GroupedMatmulSwigluQuantV2 aclnnfallback", "The dim num should be greater than 2, but the actual is %zu.",
                        viewShape.size()),
                return false);
    // when tensor is transposed, last two dims in strides and viewShape should swap
    auto dimM = viewShape.size() - 2;
    auto dimN = viewShape.size() - 1;
    std::swap(strides[dimM], strides[dimN]);
    // swap viewShape
    std::swap(viewShape[dimM], viewShape[dimN]);
    return true;
}

static void InitShapeAndStrideForGMMSwigluQuantV2(const gert::Shape &originShape, std::vector<int64_t> &viewShape,
                                     std::vector<int64_t> &strides)
{
    for (size_t i = 0; i < originShape.GetDimNum(); ++i) {
        viewShape.push_back(originShape.GetDim(i));
    }
    strides.resize(viewShape.size(), 1);
    // compute the strides of contiguous tensor
    for (int64_t i = viewShape.size() - 2; i >= 0; i--) {
        strides[i] = viewShape[i + 1] * strides[i + 1];
    }
}

static bool IsFP8FP4BitsDataTypeForGMMSwigluQuantV2(ge::DataType dataType)
{
    return dataType == ge::DataType::DT_FLOAT8_E4M3FN || dataType == ge::DataType::DT_FLOAT8_E5M2
           || dataType == ge::DataType::DT_FLOAT8_E8M0 || dataType == ge::DataType::DT_FLOAT4_E2M1
           || dataType == ge::DataType::DT_INT8 || dataType == ge::DataType::DT_HIFLOAT8;
}


static inline aclDataType ToAclDataTypeForGMMSwigluQuantV2(ge::DataType dtype)
{
    static const std::vector<DataType> GMMWsiglu_CONVERT_TO_ACL_DataType_LIST = {
        ge::DataType::DT_FLOAT8_E4M3FN, ge::DataType::DT_FLOAT8_E5M2, ge::DataType::DT_FLOAT8_E8M0,
        ge::DataType::DT_FLOAT4_E2M1, ge::DataType::DT_INT8, ge::DataType::DT_HIFLOAT8};
    auto iter =
        std::find(GMMWsiglu_CONVERT_TO_ACL_DataType_LIST.begin(), GMMWsiglu_CONVERT_TO_ACL_DataType_LIST.end(), dtype);
    if (iter == GMMWsiglu_CONVERT_TO_ACL_DataType_LIST.end()) {
        return aclDataType::ACL_DT_UNDEFINED;
    }
    return static_cast<aclDataType>(dtype);
}

static inline aclTensorList* ConvertType(aclTensorList* geTensorList) {
    return geTensorList;
}

static inline aclTensor *GeTensor2AclTensor(const gert::Tensor *geTensor, bool enableTranspose, size_t index,
                                            bool enableNZ = false)
{
    OP_CHECK_IF(geTensor == nullptr, OP_LOGE("GroupedMatmulSwigluQuantV2 aclnnfallback", "The geTensor nullptr"), return nullptr);
    auto storageShape = geTensor->GetStorageShape();
    if (storageShape.GetDimNum() <= 1) {
        return ConvertType(geTensor);
    }
    std::vector<int64_t> storageShapeVec;
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        storageShapeVec.push_back(storageShape.GetDim(i));
    }

    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    OP_CHECK_IF(aclCreateTensor == nullptr, OP_LOGE("GroupedMatmulSwigluQuantV2 aclnnfallback", "The aclCreateTensor nullptr"), return nullptr);

    void* deviceAddr = const_cast<void*>(geTensor->GetAddr());
    // convert data type
    auto dataTypeGE = geTensor->GetDataType();
    aclDataType dataType = ACL_DT_UNDEFINED;
    if (IsFP8FP4BitsDataTypeForGMMSwigluQuantV2(dataTypeGE)) {
        dataType = ToAclDataTypeForGMMSwigluQuantV2(dataTypeGE);
    } else {
        dataType = ToAclDataType(dataTypeGE);
    }
    // convert view shape
    const gert::Shape &origin_shape = geTensor->GetOriginShape();
    std::vector<int64_t> viewShape;
    std::vector<int64_t> strides;
    InitShapeAndStrideForGMMSwigluQuantV2(origin_shape, viewShape, strides);
    if (enableTranspose) {
        if (dataTypeGE == ge::DataType::DT_FLOAT8_E8M0) {
            if (index == INDEX_INPUT_WEIGHT_SCALE) {
                OP_CHECK_IF(!SetShapeAndStrideForMXFPTypeMForGMMSwigluQuantV2(viewShape, strides),
                        OP_LOGE("GroupedMatmulSwigluQuantV2 aclnnfallback", "SetShapeAndStrideForMXFPTypeMForGMMSwigluQuantV2 failed"),
                        return nullptr);
            }
        } else {
            OP_CHECK_IF(!SetShapeAndStrideForLastTwoDimForGMMSwigluQuantV2(viewShape, strides),
                    OP_LOGE("GroupedMatmulSwigluQuantV2 aclnnfallback", "SetShapeAndStrideForLastTwoDimForGMMSwigluQuantV2 failed"),
                    return nullptr);
        }
    }
    auto aclFormat = aclFormat::ACL_FORMAT_ND;
    if (enableNZ && GetPrimaryFormat(geTensor->GetStorageFormat()) == ge::Format::FORMAT_FRACTAL_NZ) {
        aclFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    }
    aclTensor* out = aclCreateTensor(viewShape.data(), viewShape.size(), dataType, strides.data(),
                                    0, aclFormat, storageShapeVec.data(), storageShapeVec.size(), deviceAddr);
    OP_CHECK_IF(out == nullptr, OP_LOGE("GroupedMatmulSwigluQuantV2 aclnnfallback", "The GeTensor2AclTensor out nullptr"), return nullptr);
    return out;
}

static graphStatus PrepareAclTensor(const OpExecuteContext *host_api_ctx, const aclTensor *&tensor, size_t index,
                                    bool enableTranspose, bool enableNZ)
{
    if (index != INDEX_INPUT_SMOOTH_SCALE && index != INDEX_INPUT_BIAS) {
        tensor = GeTensor2AclTensor(host_api_ctx->GetInputTensor(index), enableTranspose, index, enableNZ);
    } else {
        tensor = GeTensor2AclTensor(host_api_ctx->GetOptionalInputTensor(index), enableTranspose, index, enableNZ);
    }
    OP_CHECK_IF(tensor == nullptr, OP_LOGE("GroupedMatmulSwigluQuantV2 aclnnfallback", "PrepareAclTensor fialed"), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus PrepareAclTensorVector(const OpExecuteContext *host_api_ctx, std::vector<const aclTensor *> &tensorVector,
                                          size_t index, bool enableTranspose, bool enableNZ)
{
    size_t cnt = 0;
    while (true) {
        auto inputGe = host_api_ctx->GetDynamicInputTensor(index, cnt);
        // GetDynamicInputTensor return nullptr, indicating that there are no more input tensors, and loop will safely exit
        if (inputGe == nullptr) {
            break;
        }
        auto inputAcl = GeTensor2AclTensor(inputGe, enableTranspose, index, enableNZ);
        OP_CHECK_IF(inputAcl == nullptr, OP_LOGE("GroupedMatmulSwigluQuantV2 aclnnfallback", "PrepareAclTensorVector fialed"), return GRAPH_FAILED);
        tensorVector.push_back(inputAcl);
        cnt++;
    }
    return GRAPH_SUCCESS;
}

static graphStatus PrepareOutputTensor(const OpExecuteContext *host_api_ctx,
                                             const gert::Tensor* &tensor, size_t index)
{
    tensor = host_api_ctx->GetOutputTensor(index);
    OP_CHECK_IF(tensor == nullptr, OP_LOGE("GroupedMatmulSwigluQuantV2 aclnnfallback", "PrepareOutputTensor fialed"), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus GroupedMatmulSwigluQuantV2ExecuteFunc(OpExecuteContext* host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("GroupedMatmulSwigluQuantV2 aclnnfallback", "The host_api_ctx is null"), return GRAPH_FAILED);
    auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("GroupedMatmulSwigluQuantV2 aclnnfallback", "The attrs is null"), return GRAPH_FAILED);
    const int64_t* dequantModeGe = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DEQUANT_MODE);
    OP_CHECK_IF(dequantModeGe == nullptr, OP_LOGE("GroupedMatmulSwigluQuantV2 aclnnfallback", "The dequantModeGe is null"), return GRAPH_FAILED);
    const int64_t* dequantDtypeGe = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DEQUANT_DTYPE);
    OP_CHECK_IF(dequantDtypeGe == nullptr, OP_LOGE("GroupedMatmulSwigluQuantV2 aclnnfallback", "The dequantDtypeGe is null"), return GRAPH_FAILED);
    const int64_t* quantModeGe = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_QUANT_MODE);
    OP_CHECK_IF(quantModeGe == nullptr, OP_LOGE("GroupedMatmulSwigluQuantV2 aclnnfallback", "The quantModeGe is null"), return GRAPH_FAILED);
    const int64_t* groupListTypeGe = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_GROUP_LIST_TYPE);
    OP_CHECK_IF(groupListTypeGe == nullptr, OP_LOGE("GroupedMatmulSwigluQuantV2 aclnnfallback", "The groupListTypeGe is null"), return GRAPH_FAILED);
    const bool*  transWeightGe = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANSPOSE_WEIGHT);
    OP_CHECK_IF(transWeightGe == nullptr, OP_LOGE("aclnnfallback", "transWeightGe is null"), return GRAPH_FAILED);
    
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    OP_CHECK_IF(aclCreateTensorList == nullptr,
              OP_LOGE("GroupedMatmulSwigluQuantV2 aclnnfallback", "Get opapi func aclCreateTensorList failed"), return GRAPH_FAILED);
    
    const aclTensor *aclTensorX = nullptr;
    PrepareAclTensor(host_api_ctx, aclTensorX, INDEX_INPUT_X, false, false);

    const aclTensor *aclTensorXScale = nullptr;
    PrepareAclTensor(host_api_ctx, aclTensorXScale, INDEX_INPUT_X_SCALE, false, false);

    const aclTensor *aclTensorGroupList = nullptr;
    PrepareAclTensor(host_api_ctx, aclTensorGroupList, INDEX_INPUT_GROUP_LIST, false, false);

    std::vector<const aclTensor*> aclTensorVectorWeight;
    PrepareAclTensorVector(host_api_ctx, aclTensorVectorWeight, INDEX_INPUT_WEIGHT, *transWeightGe, false);
    auto aclTensorListWeight = aclCreateTensorList(aclTensorVectorWeight.data(), aclTensorVectorWeight.size());

    std::vector<const aclTensor*> aclTensorVectorWeightScale;
    if(*quantModeGe == 2){
        PrepareAclTensorVector(host_api_ctx, aclTensorVectorWeightScale, INDEX_INPUT_WEIGHT_SCALE, *transWeightGe, false);
    }
    else{
        PrepareAclTensorVector(host_api_ctx, aclTensorVectorWeightScale, INDEX_INPUT_WEIGHT_SCALE, false, false);
    }
    auto aclTensorListWeightScale = aclCreateTensorList(aclTensorVectorWeightScale.data(), aclTensorVectorWeightScale.size());

    const gert::Tensor* geTensorY = nullptr;
    PrepareOutputTensor(host_api_ctx, geTensorY, INDEX_OUTPUT_Y);

    const gert::Tensor* geTensorYScale = nullptr;
    PrepareOutputTensor(host_api_ctx, geTensorYScale, INDEX_OUTPUT_Y_SCALE);
    
    aclTensorList* biasTensorOptional = nullptr;
    aclTensorList* smoothScaleTensorOptional = nullptr;
    aclIntArray* tuningConfig = nullptr;
    aclTensorList* aclTensorListWeightAssistMatrix = nullptr;

    // execute opapi
    auto api_ret = EXEC_OPAPI_CMD(aclnnGroupedMatmulSwigluQuantV2, aclTensorX, aclTensorListWeight, aclTensorListWeightScale,
                                  aclTensorListWeightAssistMatrix, biasTensorOptional, aclTensorXScale, smoothScaleTensorOptional, aclTensorGroupList, 
                                  *dequantModeGe, *dequantDtypeGe, *quantModeGe, *groupListTypeGe, tuningConfig,
                                  geTensorY, geTensorYScale);
    OP_CHECK_IF(api_ret != GRAPH_SUCCESS, OP_LOGE("GroupedMatmulSwigluQuantV2 aclnnfallback", "The api_ret failed:%u", api_ret),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

IMPL_OP(GroupedMatmulSwigluQuantV2).OpExecuteFunc(GroupedMatmulSwigluQuantV2ExecuteFunc);

}  // namespace fallback

#ifdef __cplusplus
}
#endif
