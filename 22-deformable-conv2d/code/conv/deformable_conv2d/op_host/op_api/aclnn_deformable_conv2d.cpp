/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_deformable_conv2d.h"
#include "deformable_conv2d.h"
#include "level0/add.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "../../../convolution_forward/op_host/op_api/convolution.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace
{
struct DeformableConv2dInputTensor {
    const aclTensor* input;
    const aclTensor* weight;
    const aclTensor* offset;
    const aclTensor* biasOptional;
};

struct ConvolutionOutput {
    aclTensor* out;
    aclTensor* deformOutOptional;
};

struct ConvolutionResult {
    const aclTensor* out;
    const aclTensor* deformOutOptional;
};

struct DeformableConv2dParams {
    const aclIntArray* kernelSize;
    const aclIntArray* stride;
    const aclIntArray* padding;
    const aclIntArray* dilation;
    const int64_t groups;
    const int64_t deformableGroups;
    const bool modulated;
    const bool deformOutMask;
};

static const int64_t MAX_KERNEL_SIZE = 2048;
static const int64_t MAX_MATMUL_K = 65535;
static const int64_t THREE_NUM = 3;
static const int64_t INDEX_ZERO = 0;
static const int64_t INDEX_ONE = 1;
static const int64_t INDEX_TWO = 2;
static const int64_t INDEX_THREE = 3;
static const int32_t INT_MAX_VALUE = 2147483647;
static size_t KERNEL_ARRAY_DIM_SIZE = 2;
static size_t STRIDE_ARRAY_DIM_SIZE = 4;
static size_t PADDING_ARRAY_DIM_SIZE = 4;
static size_t DILATION_ARRAY_DIM_SIZE = 4;
static size_t DIM_FOUR = 4;
static size_t DIM_ONE = 1;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_BF16};

static aclnnStatus InputTransProcess(const aclTensor*& inputTensor, const string& tensorName, aclOpExecutor* executor)
{
    // API输入预处理 l0 InputTensor -> l0op::Contiguous -> l0op::Transpose -> inputTensor
    inputTensor = l0op::Contiguous(inputTensor, executor);
    OP_CHECK(inputTensor != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input preprocess failed, %s with Contiguous return nullptr.",
                     tensorName.c_str()),
             return ACLNN_ERR_INNER_NULLPTR);

    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch != NpuArch::DAV_3510 || tensorName != "weight") {
        int64_t valuePerm[4] = {0, 2, 3, 1};
        auto perm = executor->AllocIntArray(valuePerm, 4);
        CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
        inputTensor = l0op::Transpose(inputTensor, perm, executor);
        OP_CHECK(inputTensor != nullptr,
                OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input preprocess failed, %s with Transpose return nullptr.",
                        tensorName.c_str()),
                return ACLNN_ERR_INNER_NULLPTR);
    }

    if (curArch == NpuArch::DAV_3510 && tensorName != "weight") {
        const_cast<aclTensor *>(inputTensor)->SetStorageFormat(Format::FORMAT_NHWC);
        const_cast<aclTensor *>(inputTensor)->SetOriginalFormat(Format::FORMAT_NHWC);
        const_cast<aclTensor *>(inputTensor)->SetViewFormat(Format::FORMAT_NHWC);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus InputProcess(const aclTensor*& inputTensor, const string& tensorName, aclOpExecutor* executor)
{
    // API输入预处理 l0 InputTensor -> l0op::Contiguous -> inputTensor
    inputTensor = l0op::Contiguous(inputTensor, executor);
    OP_CHECK(inputTensor != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input preprocess failed, %s with Contiguous return nullptr.",
                     tensorName.c_str()),
             return ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus BiasProcess(DeformableConv2dInputTensor& inputTensor, ConvolutionResult& resultTensor,
                               aclOpExecutor* executor)
{
    if (inputTensor.biasOptional != nullptr) {
        op::Shape biasShape = inputTensor.biasOptional->GetViewShape();
        int64_t biasLength = biasShape.GetDim(0);
        inputTensor.biasOptional = l0op::Reshape(inputTensor.biasOptional, {1, 1, biasLength, 1}, executor);
        CHECK_RET(inputTensor.biasOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        resultTensor.out = l0op::Add(resultTensor.out, inputTensor.biasOptional, executor);
        CHECK_RET(resultTensor.out != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ResultViewProcess(DeformableConv2dInputTensor& inputTensor, ConvolutionResult& resultTensor,
                                     ConvolutionOutput& outputTensor, aclOpExecutor* executor)
{
    auto dtype = outputTensor.out->GetDataType();
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch != NpuArch::DAV_3510) {
        auto res = BiasProcess(inputTensor, resultTensor, executor);
        CHECK_RET(res == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        if (dtype == op::DataType::DT_BF16 || dtype == op::DataType::DT_FLOAT16) {
            resultTensor.out = l0op::Cast(resultTensor.out, dtype, executor);
            CHECK_RET(resultTensor.out != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        int64_t outPerm[4] = {0, 2, 1, 3};
        auto outPermArray = executor->AllocIntArray(outPerm, 4);
        CHECK_RET(outPermArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        resultTensor.out = l0op::Transpose(resultTensor.out, outPermArray, executor);
        OP_CHECK(resultTensor.out != nullptr,
                OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output viewprocess failed, out with Transpose return nullptr."),
                return ACLNN_ERR_INNER_NULLPTR);
    }
    auto resultOut = l0op::ViewCopy(resultTensor.out, outputTensor.out, executor);
    OP_CHECK(resultOut != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output viewprocess failed, out with ViewCopy return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);

    if (outputTensor.deformOutOptional != nullptr) {
        if (curArch != NpuArch::DAV_3510) {
            if (dtype == op::DataType::DT_BF16 || dtype == op::DataType::DT_FLOAT16) {
                resultTensor.deformOutOptional = l0op::Cast(resultTensor.deformOutOptional, dtype, executor);
                CHECK_RET(resultTensor.deformOutOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
            }
            int64_t deformOutPerm[4] = {0, 3, 1, 2};
            auto perm = executor->AllocIntArray(deformOutPerm, 4);
            CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
            resultTensor.deformOutOptional = l0op::Transpose(resultTensor.deformOutOptional, perm, executor);
            OP_CHECK(resultTensor.deformOutOptional != nullptr,
                    OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                            "The output viewprocess failed, deformableOut with Transpose return nullptr."),
                    return ACLNN_ERR_INNER_NULLPTR);
        }
        auto resultDeformableOut =
            l0op::ViewCopy(resultTensor.deformOutOptional, outputTensor.deformOutOptional, executor);
        OP_CHECK(resultDeformableOut != nullptr,
                 OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                         "The output viewprocess failed, deformableOut with ViewCopy return nullptr."),
                 return ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclIntArray* ResetParams(int64_t newValue, aclOpExecutor* executor) {
    int64_t data[DIM_FOUR];
    data[INDEX_ZERO] = newValue;
    data[INDEX_ONE] = newValue;
    data[INDEX_TWO] = newValue;
    data[INDEX_THREE] = newValue;
    aclIntArray *newArray = executor->AllocIntArray(data, DIM_FOUR);
    return newArray;
}

static aclIntArray* ParamsTranspose(const aclIntArray* inputArray, aclOpExecutor* executor) {
    int64_t data[DIM_FOUR];
    data[INDEX_ZERO] = (*inputArray)[INDEX_ZERO];
    data[INDEX_ONE] = (*inputArray)[INDEX_TWO];
    data[INDEX_TWO] = (*inputArray)[INDEX_THREE];
    data[INDEX_THREE] = (*inputArray)[INDEX_ONE];
    aclIntArray *newArray = executor->AllocIntArray(data, DIM_FOUR);
    return newArray;
}

static aclnnStatus DeformableConv2dV2(DeformableConv2dInputTensor& inputTensor, ConvolutionOutput& outputTensor,
                                     DeformableConv2dParams& params, aclOpExecutor* executor)
{
    params.stride = ParamsTranspose(params.stride, executor);
    params.dilation = ParamsTranspose(params.dilation, executor);
    OP_CHECK_NULL(params.stride, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(params.dilation, return ACLNN_ERR_INNER_NULLPTR);
    auto deformOut = l0op::DeformableOffsetsNHWC(inputTensor.input, inputTensor.offset, params.kernelSize,
                                                    outputTensor.out->GetDataType(), params.stride,
                                                    params.padding, params.dilation,
                                                    params.deformableGroups, params.modulated, executor);
    int64_t deformOutPerm[4] = {0, 3, 1, 2};
    auto perm = executor->AllocIntArray(deformOutPerm, 4);
    CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
    deformOut = l0op::Transpose(deformOut, perm, executor);
    OP_CHECK(deformOut != nullptr,
                OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                        "DeformableOut with Transpose return nullptr."),
                return ACLNN_ERR_INNER_NULLPTR);
    const_cast<aclTensor *>(deformOut)->SetStorageFormat(Format::FORMAT_NCHW);
    const_cast<aclTensor *>(deformOut)->SetOriginalFormat(Format::FORMAT_NCHW);
    const_cast<aclTensor *>(deformOut)->SetViewFormat(Format::FORMAT_NCHW);
    params.padding = ResetParams(0, executor);
    params.dilation = ResetParams(1, executor);
    OP_CHECK_NULL(params.padding, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(params.dilation, return ACLNN_ERR_INNER_NULLPTR);
    int64_t newStride[KERNEL_ARRAY_DIM_SIZE];
    newStride[INDEX_ZERO] = (*params.kernelSize)[INDEX_ZERO];
    newStride[INDEX_ONE] = (*params.kernelSize)[INDEX_ONE];
    aclIntArray *newStrideArray = executor->AllocIntArray(newStride, KERNEL_ARRAY_DIM_SIZE);
    OP_CHECK_NULL(newStrideArray, return ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* tmpOut = l0op::Conv2dV2NCHW(deformOut, inputTensor.weight, inputTensor.biasOptional,
                                                    outputTensor.out->GetDataType(), newStrideArray,
                                                    params.padding, params.dilation, params.groups,
                                                    false, executor);
    ConvolutionResult resultTensor = {tmpOut, deformOut};
    auto result = ResultViewProcess(inputTensor, resultTensor, outputTensor, executor);
    CHECK_RET(result == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    OP_LOGD("After CalculateDeformableConv2d");
    return ACLNN_SUCCESS;
}

static aclnnStatus CalculateDeformableConv2d(DeformableConv2dInputTensor& inputTensor, ConvolutionOutput& outputTensor,
                                             DeformableConv2dParams& params, aclOpExecutor* executor)
{
    CHECK_RET(InputTransProcess(inputTensor.input, "input", executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(InputTransProcess(inputTensor.offset, "offset", executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(InputTransProcess(inputTensor.weight, "weight", executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    if (inputTensor.biasOptional != nullptr) {
        CHECK_RET(InputProcess(inputTensor.biasOptional, "biasOptional", executor) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
    }

    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_3510) {
        return DeformableConv2dV2(inputTensor, outputTensor, params, executor);
    }

    auto dtype = inputTensor.input->GetDataType();
    if (dtype == op::DataType::DT_BF16 || dtype == op::DataType::DT_FLOAT16) {
        inputTensor.input = l0op::Cast(inputTensor.input, op::DataType::DT_FLOAT, executor);
        CHECK_RET(inputTensor.input != nullptr, ACLNN_ERR_INNER_NULLPTR);
        inputTensor.offset = l0op::Cast(inputTensor.offset, op::DataType::DT_FLOAT, executor);
        CHECK_RET(inputTensor.offset != nullptr, ACLNN_ERR_INNER_NULLPTR);
        inputTensor.weight = l0op::Cast(inputTensor.weight, op::DataType::DT_FLOAT, executor);
        CHECK_RET(inputTensor.weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (inputTensor.biasOptional != nullptr) {
            inputTensor.biasOptional = l0op::Cast(inputTensor.biasOptional, op::DataType::DT_FLOAT, executor);
            CHECK_RET(inputTensor.biasOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    auto deformableResult =
        l0op::DeformableConv2d(inputTensor.input, inputTensor.weight, inputTensor.offset, inputTensor.biasOptional,
                               params.kernelSize, params.stride, params.padding, params.dilation, params.groups,
                               params.deformableGroups, params.modulated, executor);
    const aclTensor* tmpOut = std::get<0>(deformableResult);
    const aclTensor* tmpDeformableOut = std::get<1>(deformableResult);
    CHECK_RET(tmpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(tmpDeformableOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    ConvolutionResult resultTensor = {tmpOut, tmpDeformableOut};
    auto res = ResultViewProcess(inputTensor, resultTensor, outputTensor, executor);
    CHECK_RET(res == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    OP_LOGD("After CalculateDeformableConv2d");
    return ACLNN_SUCCESS;
}

static bool CheckNotNull(DeformableConv2dInputTensor& inputTensor, ConvolutionOutput& outputTensor,
                         DeformableConv2dParams& params)
{
    OP_CHECK_NULL(inputTensor.input, return false);
    OP_CHECK_NULL(inputTensor.offset, return false);
    OP_CHECK_NULL(inputTensor.weight, return false);
    OP_CHECK_NULL(outputTensor.out, return false);
    OP_CHECK_NULL(params.kernelSize, return false);
    OP_CHECK_NULL(params.stride, return false);
    OP_CHECK_NULL(params.padding, return false);
    OP_CHECK_NULL(params.dilation, return false);
    return true;
}

static bool CheckDtypeValid(DeformableConv2dInputTensor& inputTensor, ConvolutionOutput& outputTensor)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.input, DTYPE_SUPPORT_LIST, return false);
    if (inputTensor.biasOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(inputTensor.biasOptional, inputTensor.input->GetDataType(), return false);
    }
    if (outputTensor.deformOutOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(outputTensor.deformOutOptional, inputTensor.input->GetDataType(), return false);
    }
    OP_CHECK_DTYPE_NOT_MATCH(inputTensor.offset, inputTensor.input->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(inputTensor.weight, inputTensor.input->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(outputTensor.out, inputTensor.input->GetDataType(), return false);
    return true;
}

static bool CheckFormat(DeformableConv2dInputTensor& inputTensor, ConvolutionOutput& outputTensor)
{
    // 如果输入格式不满足格式要求，记录日志，直接报错
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_3510) {
        OP_CHECK(inputTensor.input->GetStorageFormat() == Format::FORMAT_NCHW,
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x Format only support NCHW or ND."), return false);
    } else {
        OP_CHECK(inputTensor.input->GetStorageFormat() == Format::FORMAT_NCHW ||
                    inputTensor.input->GetStorageFormat() == Format::FORMAT_ND,
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x Format only support NCHW or ND."), return false);
    }

    OP_CHECK(inputTensor.offset->GetStorageFormat() == inputTensor.input->GetStorageFormat(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of offset and x should be equal."), return false);

    OP_CHECK(inputTensor.weight->GetStorageFormat() == inputTensor.input->GetStorageFormat(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of weight and x should be equal."), return false);

    if (inputTensor.biasOptional != nullptr) {
        OP_CHECK(inputTensor.biasOptional->GetStorageFormat() == Format::FORMAT_ND,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "biasOptional Format only support ND."), return false);
    }

    OP_CHECK(outputTensor.out->GetStorageFormat() == inputTensor.input->GetStorageFormat(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of out and x should be equal."), return false);

    if (outputTensor.deformOutOptional != nullptr) {
        OP_CHECK(outputTensor.deformOutOptional->GetStorageFormat() == inputTensor.input->GetStorageFormat(),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of deformOutOptional and x should be equal."), return false);
    }
    return true;
}

static bool CheckAttrs(DeformableConv2dParams& params)
{
    OP_CHECK(params.kernelSize->Size() == KERNEL_ARRAY_DIM_SIZE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kernelSize length should be 2."), return false);
    OP_CHECK(params.stride->Size() == STRIDE_ARRAY_DIM_SIZE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stride length should be 4."), return false);
    OP_CHECK(params.padding->Size() == PADDING_ARRAY_DIM_SIZE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding length should be 4."), return false);
    OP_CHECK(params.dilation->Size() == DILATION_ARRAY_DIM_SIZE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilation length should be 4."), return false);

    int64_t kH = (*params.kernelSize)[INDEX_ZERO];
    int64_t kW = (*params.kernelSize)[INDEX_ONE];
    OP_CHECK(kH > 0 && kW > 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kernelSize should greater than 0."), return false);
    OP_CHECK(kH * kW <= MAX_KERNEL_SIZE, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kH * kW should not exceed 2048."),
             return false);

    OP_CHECK((*params.stride)[INDEX_ZERO] == 1 && (*params.stride)[INDEX_ONE] == 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stride[0]、stride[1] should be 1."), return false);
    OP_CHECK((*params.stride)[INDEX_TWO] > 0 && (*params.stride)[INDEX_THREE] > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stride[2]、stride[3] should greater than 0."), return false);

    OP_CHECK((*params.dilation)[INDEX_ZERO] == 1 && (*params.dilation)[INDEX_ONE] == 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilation[0]、dilation[1] should be 1."), return false);
    OP_CHECK((*params.dilation)[INDEX_TWO] > 0 && (*params.dilation)[INDEX_THREE] > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilation[2]、dilation[3] should greater than 0."), return false);

    OP_CHECK(params.groups > 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groups should greater than 0."), return false);
    OP_CHECK(params.deformableGroups > 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "deformableGroups should greater than 0."),
             return false);
    OP_CHECK(params.modulated == true, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "modulated should be true."), return false);
    return true;
}

static bool CheckDimension(DeformableConv2dInputTensor& inputTensor, ConvolutionOutput& outputTensor)
{
    OP_CHECK_WRONG_DIMENSION(inputTensor.input, DIM_FOUR, return false);
    OP_CHECK_WRONG_DIMENSION(inputTensor.weight, DIM_FOUR, return false);
    OP_CHECK_WRONG_DIMENSION(inputTensor.offset, DIM_FOUR, return false);
    if (inputTensor.biasOptional != nullptr) {
        OP_CHECK_WRONG_DIMENSION(inputTensor.biasOptional, DIM_ONE, return false);
    }
    OP_CHECK_WRONG_DIMENSION(outputTensor.out, DIM_FOUR, return false);
    if (outputTensor.deformOutOptional != nullptr) {
        OP_CHECK_WRONG_DIMENSION(outputTensor.deformOutOptional, DIM_FOUR, return false);
    }
    return true;
}

static bool CheckShape(DeformableConv2dInputTensor& inputTensor, ConvolutionOutput& outputTensor, 
                       DeformableConv2dParams& params)
{
    int64_t n = inputTensor.input->GetViewShape()[INDEX_ZERO];
    int64_t inC = inputTensor.input->GetViewShape()[INDEX_ONE];
    int64_t inH = inputTensor.input->GetViewShape()[INDEX_TWO];
    int64_t inW = inputTensor.input->GetViewShape()[INDEX_THREE];
    int64_t inSize = inH * inW;
    int64_t outC = inputTensor.weight->GetViewShape()[INDEX_ZERO];
    OP_CHECK(inC % params.deformableGroups == 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inC needs to be divisible by deformableGroups."), return false);
    OP_CHECK(inC % params.groups == 0 && outC % params.groups == 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Both inC and outC needs to be divisible by groups."), return false);
    OP_CHECK(inSize <= INT_MAX_VALUE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inH multiplied by inW should not exceed 2147483647."), return false);

    int64_t kH = (*params.kernelSize)[INDEX_ZERO];
    int64_t kW = (*params.kernelSize)[INDEX_ONE];
    int64_t padTop = (*params.padding)[INDEX_ZERO];
    int64_t padBottom = (*params.padding)[INDEX_ONE];
    int64_t padLeft = (*params.padding)[INDEX_TWO];
    int64_t padRight = (*params.padding)[INDEX_THREE];
    int64_t dilationH = (*params.dilation)[INDEX_TWO];
    int64_t dilationW = (*params.dilation)[INDEX_THREE];
    int64_t strideH = (*params.stride)[INDEX_TWO];
    int64_t strideW = (*params.stride)[INDEX_THREE];

    int64_t outH = (inH + padTop + padBottom - (kH - 1) * dilationH - 1) / strideH + 1;
    int64_t outW = (inW + padLeft + padRight - (kW - 1) * dilationW - 1) / strideW + 1;

    op::Shape weightExpectShape = {outC, inC / params.groups, kH, kW};
    op::Shape offsetExpectShape = {n, THREE_NUM * params.deformableGroups * kH * kW, outH, outW};
    op::Shape outExpectShape = {n, outC, outH, outW};
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(inputTensor.weight, weightExpectShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(inputTensor.offset, offsetExpectShape, return false);
    if (inputTensor.biasOptional != nullptr) {
        op::Shape biasExpectShape = {outC};
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(inputTensor.biasOptional, biasExpectShape, return false);
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputTensor.out, outExpectShape, return false);
    if (outputTensor.deformOutOptional != nullptr) {
        op::Shape deformOutExpectShape = {n, inC, outH * kH, outW * kW};
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputTensor.deformOutOptional, deformOutExpectShape, return false);
    }
    int64_t matmulK = kH * kW * inC / params.groups;
    OP_CHECK(matmulK <= MAX_MATMUL_K, 
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kH multiplied by kW multiplied by inC divided by groups should not exceed 65535."), return false);
    return true;
}

static aclnnStatus CheckParams(DeformableConv2dInputTensor& inputTensor, ConvolutionOutput& outputTensor,
                               DeformableConv2dParams& params)
{
    // 1. 检查参数是否为空指针
    CHECK_COND(CheckNotNull(inputTensor, outputTensor, params), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_COND(CheckDtypeValid(inputTensor, outputTensor), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");

    // 3. 检查数据格式是否支持
    CHECK_COND(CheckFormat(inputTensor, outputTensor), ACLNN_ERR_PARAM_INVALID, "CheckFormat failed!");

    // 4. 检查Params是否支持
    CHECK_COND(CheckAttrs(params), ACLNN_ERR_PARAM_INVALID, "CheckAttrs failed!");

    // 5. 检查Shape是否支持
    CHECK_COND(CheckDimension(inputTensor, outputTensor), ACLNN_ERR_PARAM_INVALID, "CheckDimension failed!");
    CHECK_COND(CheckShape(inputTensor, outputTensor, params), ACLNN_ERR_PARAM_INVALID, "CheckShape failed!");

    return ACLNN_SUCCESS;
}
};  // namespace

aclnnStatus aclnnDeformableConv2dGetWorkspaceSize(const aclTensor* x, const aclTensor* weight, const aclTensor* offset,
                                                  const aclTensor* biasOptional, const aclIntArray* kernelSize,
                                                  const aclIntArray* stride, const aclIntArray* padding,
                                                  const aclIntArray* dilation, int64_t groups, int64_t deformableGroups,
                                                  bool modulated, aclTensor* out, aclTensor* deformOutOptional,
                                                  uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnDeformableConv2d,
                   DFX_IN(x, weight, offset, biasOptional, kernelSize, stride, padding, dilation, groups,
                          deformableGroups, modulated),
                   DFX_OUT(out, deformOutOptional));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    DeformableConv2dInputTensor inputTensor = {x, weight, offset, biasOptional};
    ConvolutionOutput outputTensor = {out, deformOutOptional};
    bool deformOutMask = (deformOutOptional != nullptr);
    DeformableConv2dParams params = {kernelSize, stride,           padding,   dilation,
                                     groups,     deformableGroups, modulated, deformOutMask};

    // 固定写法，参数检查
    auto ret = CheckParams(inputTensor, outputTensor, params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor直接返回
    if (x->IsEmpty() || weight->IsEmpty() || offset->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    OP_LOGD("begin stride is: %s", params.stride->ToString().GetString());
    OP_LOGD("begin padding is: %s", params.padding->ToString().GetString());
    OP_LOGD("begin dilation is: %s", params.dilation->ToString().GetString());
    OP_LOGD("begin kernelSize is: %s", params.kernelSize->ToString().GetString());
    OP_LOGD("Entering CalculateDeformableConv2d");
    auto ret1 = CalculateDeformableConv2d(inputTensor, outputTensor, params, uniqueExecutor.get());
    CHECK_RET(ret1 == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDeformableConv2d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDeformableConv2d);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
