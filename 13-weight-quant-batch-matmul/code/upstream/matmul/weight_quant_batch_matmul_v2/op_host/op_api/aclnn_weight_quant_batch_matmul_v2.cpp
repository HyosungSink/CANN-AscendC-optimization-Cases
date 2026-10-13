/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_weight_quant_batch_matmul_v2.h"
#include "aclnn_weight_quant_batch_matmul_v3.h"
#include "aclnn_weight_quant_batch_matmul_nz.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "weight_quant_batch_matmul_v2.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"

#include <limits>
#include "graph/types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
using Ops::NN::IsTransposeLastTwoDims;
using Ops::NN::SwapLastTwoDimValue;
using TupleTensor = std::tuple<
    const aclTensor*&, const aclTensor*&, const aclTensor*&, const aclTensor*&, const aclTensor*&, const aclTensor*&>;
using TupleAttr = std::tuple<int&, bool&, bool&>;

static constexpr int INDEX_X_IN_MANDTORY_TUPLE = 0;
static constexpr int INDEX_WEIGHT_IN_MANDTORY_TUPLE = 1;
static constexpr int INDEX_WEIGHT_BAK_IN_MANDTORY_TUPLE = 2;
static constexpr int INDEX_ANTIQUANT_SCALE_IN_MANDTORY_TUPLE = 3;
static constexpr int INDEX_Y_OUT_MANDTORY_TUPLE = 4;
static constexpr int INDEX_ANTIQUANT_SCALE_REF_IN_MANDTORY_TUPLE = 5;
static constexpr int INDEX_ANTIQUANT_OFFSET_IN_OPTIONAL_TUPLE = 0;
static constexpr int INDEX_QUANT_SCALE_IN_OPTIONAL_TUPLE = 1;
static constexpr int INDEX_QUANT_SCALE_BAK_IN_OPTIONAL_TUPLE = 2;
static constexpr int INDEX_QUANT_OFFSET_IN_OPTIONAL_TUPLE = 3;
static constexpr int INDEX_BIAS_IN_OPTIONAL_TUPLE = 4;
static constexpr int INDEX_ANTIQUANT_GROUPSIZE_IN_ATTR_TUPLE = 0;
static constexpr int INDEX_TRANSPOSE_X_IN_ATTR_TUPLE = 1;
static constexpr int INDEX_TRANSPOSE_WEIGHT_IN_ATTR_TUPLE = 2;

namespace {
const size_t ANTIQUANT_DIM_MAX_VALUE = 2;
const size_t WEIGHT_K_OFFSET = 2;
const size_t BIAS_DIM_MAX_VALUE = 2;
const uint64_t INPUT_DIM_MIN_VALUE = 2;
const uint64_t INPUT_DIM_MAX_VALUE = 2;
const uint64_t INPUT_DIM_MAX_VALUE_BATCH = 6;
const uint64_t OPTIONAL_INPUT_DIM_MIN_VALUE = 1;
const uint64_t OPTIONAL_INPUT_DIM_MAX_VALUE = 2;
const int ANTIQUANT_GROUP_SIZE_MIN_VALUE = 32;
const uint64_t INT4_NUMS_IN_INT32 = 8;
const int64_t M_K_N_MAX_VALUE = 65535;
const int64_t MAX_MK_VALUE = 512000000;
const int64_t SECOND_LAST_DIM = 2;
const int64_t ANTIQUANT_GRP_SIZE64 = 64;
const int64_t ANTIQUANT_GRP_SIZE128 = 128;
const int64_t N_ALIGN_VALUE = 64;
const uint64_t INPUT_DIM_WITHOUT_BATCH = 2;

static const std::initializer_list<DataType> ASCEND910B_AQSCALE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_UINT64, DataType::DT_INT64};
static const std::initializer_list<DataType> ASCEND910B_X_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::initializer_list<DataType> ASCEND310P_X_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16};
static const std::initializer_list<DataType> ASCEND910B_WEIGHT_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT8, DataType::DT_INT4};
static const std::initializer_list<DataType> DAV3510_WEIGHT_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT8,          DataType::DT_INT4,
    DataType::DT_FLOAT8_E4M3FN, DataType::DT_HIFLOAT8, DataType::DT_FLOAT4_E2M1};
static const std::initializer_list<DataType> DAV3510_ANTIQUANT_SCALE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<DataType> ASCEND310P_WEIGHT_DTYPE_SUPPORT_LIST = {DataType::DT_INT8};
static const std::initializer_list<DataType> ASCEND910B_Y_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT8};
static const std::initializer_list<DataType> DAV3510_Y_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::initializer_list<DataType> ASCEND310P_Y_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16};
static const std::initializer_list<DataType> EMPTY_LIST = {};
static const std::vector<uint64_t> DIM_RANGE_WITHOUT_BATCH = {INPUT_DIM_MIN_VALUE, INPUT_DIM_MAX_VALUE};
static const std::vector<uint64_t> DIM_RANGE_WITH_BATCH = {INPUT_DIM_MIN_VALUE, INPUT_DIM_MAX_VALUE_BATCH};
static const std::vector<uint64_t> DIM_RANGE_OPTIONAL_INPUT = {
    OPTIONAL_INPUT_DIM_MIN_VALUE, OPTIONAL_INPUT_DIM_MAX_VALUE};
static const std::vector<uint64_t> BIAS_DIM_RANGE_OPTIONAL_INPUT = {1, INPUT_DIM_MAX_VALUE_BATCH};
static inline const std::initializer_list<DataType>& GetAntiQuantScaleDtypeSupportList()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    NpuArch npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    switch (npuArch) {
        case NpuArch::DAV_2201:
            return ASCEND910B_AQSCALE_DTYPE_SUPPORT_LIST;
        case NpuArch::DAV_2002:
            return ASCEND310P_X_DTYPE_SUPPORT_LIST;
        case NpuArch::DAV_3510:
            return DAV3510_ANTIQUANT_SCALE_DTYPE_SUPPORT_LIST;
        default: {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
            return EMPTY_LIST;
        }
    }
}

static inline const std::initializer_list<DataType>& GetXDtypeSupportList()
{
    NpuArch npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (npuArch) {
        case NpuArch::DAV_2201:
        case NpuArch::DAV_3510:
            return ASCEND910B_X_DTYPE_SUPPORT_LIST;
        case NpuArch::DAV_2002:
            return ASCEND310P_X_DTYPE_SUPPORT_LIST;
        default: {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
            return EMPTY_LIST;
        }
    }
}

static inline const std::initializer_list<DataType>& GetYDtypeSupportList()
{
    NpuArch npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (npuArch) {
        case NpuArch::DAV_2201:
            return ASCEND910B_Y_DTYPE_SUPPORT_LIST;
        case NpuArch::DAV_3510:
            return DAV3510_Y_DTYPE_SUPPORT_LIST;
        case NpuArch::DAV_2002:
            return ASCEND310P_Y_DTYPE_SUPPORT_LIST;
        default: {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
            return EMPTY_LIST;
        }
    }
}

static inline const std::initializer_list<DataType>& GetWeightDtypeSupportList()
{
    NpuArch npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (npuArch) {
        case NpuArch::DAV_2201:
            return ASCEND910B_WEIGHT_DTYPE_SUPPORT_LIST;
        case NpuArch::DAV_3510:
            return DAV3510_WEIGHT_DTYPE_SUPPORT_LIST;
        case NpuArch::DAV_2002:
            return ASCEND310P_WEIGHT_DTYPE_SUPPORT_LIST;
        default: {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
            return EMPTY_LIST;
        }
    }
}

// 获取broadcast shape
static inline op::Shape GetBroadcastShape(const aclTensor* tensor)
{
    op::Shape shape;
    size_t dimNum = tensor->GetViewShape().GetDimNum();
    size_t loopDims = dimNum - INPUT_DIM_WITHOUT_BATCH;
    for (size_t idx = 0; idx < loopDims; idx++) {
        int64_t tmpVal = tensor->GetViewShape().GetDim(idx);
        shape.AppendDim(tmpVal);
    }
    if (shape.GetDimNum() == 0) {
        shape.AppendDim(1);
    }
    return shape;
}

static const aclTensor* SetTensorToNDFormat(const aclTensor* input)
{
    const aclTensor* output = nullptr;
    if (input == nullptr) {
        return output;
    }
    if (input->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ) {
        OP_LOGD("weightQuantBatchMatmul set tensor to ND format.");
        output = l0op::ReFormat(input, op::Format::FORMAT_ND);
    } else {
        output = input;
    }
    return output;
}

static bool CheckShapeValid(const aclTensor* x, const aclTensor* weight, const aclTensor* out)
{
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND310P) {
        return true;
    }
    op::Shape broadcastShape;
    auto xDimNum = x->GetViewShape().GetDimNum();
    auto weightDimNum = weight->GetViewShape().GetDimNum();
    if (xDimNum == INPUT_DIM_WITHOUT_BATCH && weightDimNum == INPUT_DIM_WITHOUT_BATCH) {
        return true;
    }
    auto xBroadcastShape = GetBroadcastShape(x);
    auto weightBroadcastShape = GetBroadcastShape(weight);
    auto outBroadcastShape = GetBroadcastShape(out);
    auto outDimNum = outBroadcastShape.GetDimNum();
    OP_CHECK(
        BroadcastInferShape(xBroadcastShape, weightBroadcastShape, broadcastShape),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(x->GetViewShape()).GetString(),
            op::ToString(weight->GetViewShape()).GetString()),
        return false);
    OP_CHECK(
        broadcastShape.GetDimNum() == outDimNum,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "%s and %s can't broadcast to %s.", op::ToString(x->GetViewShape()).GetString(),
            op::ToString(weight->GetViewShape()).GetString(), op::ToString(out->GetViewShape()).GetString()),
        return false);
    for (size_t i = 0; i < outDimNum; i++) {
        OP_CHECK(
            outBroadcastShape.GetDim(i) == broadcastShape.GetDim(i),
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Output dim %ld is not equal to infered output dim %ld at shape index %zu.",
                outBroadcastShape.GetDim(i), broadcastShape.GetDim(i), i),
            return false);
    }
    return true;
}

static bool IsFormatSupport(const aclTensor* input, Format format, const std::string& inputName)
{
    if (input != nullptr && input->GetStorageFormat() != format) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "%s's format should be ND. actual is [%s].", inputName.c_str(),
            op::ToString(input->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool IsDimSupport(const aclTensor* input, const std::vector<uint64_t>& dimRange, const std::string& inputName)
{
    if (input != nullptr &&
        (input->GetViewShape().GetDimNum() < dimRange[0] || input->GetViewShape().GetDimNum() > dimRange[1])) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "%s's dim should be in range [%lu, %lu]. actual is [%zu].", inputName.c_str(),
            dimRange[0], dimRange[1], input->GetViewShape().GetDimNum());
        return false;
    }
    return true;
}

static bool CheckAntiquantGroupSize(const aclTensor* x, int antiquantGroupSize)
{
    // antiquantGroupSize为默认值0或者antiquantGroupSize%32 == 0并且antiquantGroupSize在[32, k-1]范围内
    if (antiquantGroupSize == 0) {
        return true;
    }

    OP_CHECK(
        antiquantGroupSize >= 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cannot support group_size[%d] less than 0.", antiquantGroupSize),
        return false);

    OP_CHECK(
        antiquantGroupSize % ANTIQUANT_GROUP_SIZE_MIN_VALUE == 0,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "antiquantGroupSize should be an integer multiple of 32, actual is %d.",
            antiquantGroupSize),
        return false);

    OP_CHECK(
        antiquantGroupSize >= ANTIQUANT_GROUP_SIZE_MIN_VALUE,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "antiquantGroupSize should not be smaller than %d, actual is %d.",
            ANTIQUANT_GROUP_SIZE_MIN_VALUE, antiquantGroupSize),
        return false);

    size_t kDimX = x->GetViewShape().GetDimNum() - 1;
    int64_t kX = x->GetViewShape().GetDim(kDimX);
    OP_CHECK(
        static_cast<int64_t>(antiquantGroupSize) + 1 <= kX,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "antiquantGroupSize should not be greater than k-1, which is %ld."
            "actual antiquantGroupSize is %d.",
            (kX - 1), antiquantGroupSize),
        return false);
    return true;
}

static aclnnStatus CheckInnerPrecise(int innerPrecise)
{
    // innerPrecise取值只能为1或0. 0:高精度 1:高性能
    if (innerPrecise != 0 && innerPrecise != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cannot support innerPrecise[%d]。only support 0 or 1.", innerPrecise);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static bool IsViewShapeSame(const aclTensor* tensorA, const aclTensor* tensorB)
{
    if (tensorA != nullptr && tensorB != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL(tensorA, tensorB, return false);
    }
    return true;
}

static int64_t GetWeightK(const aclTensor* weight)
{
    size_t kDimWeight = weight->GetViewShape().GetDimNum() - WEIGHT_K_OFFSET;
    int64_t kWeight = weight->GetViewShape().GetDim(kDimWeight);
    if (weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ &&
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
        kWeight = weight->GetViewShape().GetDim(kDimWeight + 1);
    }
    return kWeight;
}

static int64_t GetWeightN(const aclTensor* weight)
{
    size_t kDimWeight = weight->GetViewShape().GetDimNum() - WEIGHT_K_OFFSET;
    int64_t nWeight = weight->GetViewShape().GetDim(kDimWeight + 1);
    if (weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ &&
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
        nWeight = weight->GetViewShape().GetDim(kDimWeight);
    }
    return nWeight;
}

static bool CheckMicroScalingCondition(
    const aclTensor* weight, const aclTensor* antiquantScale, const aclTensor* antiquantOffsetOptional,
    int antiquantGroupSize)
{
    bool microScalingFlag = (weight->GetDataType() == DataType::DT_FLOAT4_E2M1) &&
                            (antiquantScale->GetDataType() == DataType::DT_FLOAT8_E8M0);
    if (microScalingFlag) {
        OP_CHECK(
            antiquantGroupSize == ANTIQUANT_GROUP_SIZE_MIN_VALUE,
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Invalid groupSize [%d], only support groupSize 32 on fp4 micro-scaling condition.",
                antiquantGroupSize),
            return false);
        OP_CHECK(
            antiquantOffsetOptional == nullptr,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "antiquantOffset is not supported on fp4 micro-scaling condition."),
            return false);
    }

    return true;
}

static bool CheckAntiquantParamShape(
    const aclTensor* weight, const aclTensor* antiquantScale, const aclTensor* antiquantOffsetOptional,
    int antiquantGroupSize)
{
    int64_t kWeight = GetWeightK(weight);
    int64_t nWeight = GetWeightN(weight);
    size_t nDimAntiquantScale = antiquantScale->GetViewShape().GetDimNum() - 1;
    int64_t nAntiquantScale = antiquantScale->GetViewShape().GetDim(nDimAntiquantScale);
    int64_t kAntiquantScale = 0;
    // 仅在310P weightNz场景，aclnn接口的weight为（n, k）输入
    bool isWeightNk = weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ &&
                      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
    // 非per_group场景antiquantScale支持(1),(n),(1,n), 且group size应该为0
    if (antiquantScale->GetViewShape().GetDimNum() == 1) {
        if (antiquantGroupSize > 0 || (nAntiquantScale != 1 && nAntiquantScale != nWeight)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "antiquantScale shape must be [1] or [%ld] and group size is 0 "
                "when the size is 1, but it is %s and group size is [%d]",
                nWeight, op::ToString(antiquantScale->GetViewShape()).GetString(), antiquantGroupSize);
            return false;
        }
    }
    if (antiquantScale->GetViewShape().GetDimNum() == ANTIQUANT_DIM_MAX_VALUE) {
        kAntiquantScale = isWeightNk ? antiquantScale->GetViewShape().GetDim(nDimAntiquantScale) :
                                       antiquantScale->GetViewShape().GetDim(nDimAntiquantScale - 1);
        nAntiquantScale = isWeightNk ? antiquantScale->GetViewShape().GetDim(nDimAntiquantScale - 1) : nAntiquantScale;
        if (antiquantGroupSize == 0 && (kAntiquantScale != 1 || (nAntiquantScale != nWeight && nAntiquantScale != 1))) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "antiquantScale shape must be [%ld, %ld] or [1, 1]"
                "when the size is 2 and without group size, but it is %s",
                (isWeightNk ? nWeight : static_cast<long int>(1)), (isWeightNk ? static_cast<long int>(1) : nWeight),
                op::ToString(antiquantScale->GetViewShape()).GetString());
            return false;
        }
        if (antiquantGroupSize > 0) {
            int64_t kAntiquantScaleValue =
                (kWeight + static_cast<int64_t>(antiquantGroupSize - 1)) / static_cast<int64_t>(antiquantGroupSize);
            if (kAntiquantScale != kAntiquantScaleValue || nAntiquantScale != nWeight) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "antiquantScale shape must be [%ld, %ld] "
                    "when the size is 2 and with group size, but it is %s",
                    (isWeightNk ? nWeight : kAntiquantScaleValue), (isWeightNk ? kAntiquantScaleValue : nWeight),
                    op::ToString(antiquantScale->GetViewShape()).GetString());
                return false;
            }
        }
    }
    if (!IsViewShapeSame(antiquantOffsetOptional, antiquantScale)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "The ViewShape of antiquant offset must same with scale when it is not null, "
            "but they are %s and %s",
            op::ToString(antiquantOffsetOptional->GetViewShape()).GetString(),
            op::ToString(antiquantScale->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static bool CheckQuantParamShape(
    const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional, int64_t nWeight)
{
    if (quantScaleOptional == nullptr) {
        return true;
    }
    // quantScaleOptional支持(0),(1),(n),(1,n), quantOffsetOptional需和quantScaleOptional一致
    if (quantScaleOptional->GetViewShape().GetDimNum() == OPTIONAL_INPUT_DIM_MIN_VALUE) {
        if (quantScaleOptional->GetViewShape().GetDim(0) != 0 && quantScaleOptional->GetViewShape().GetDim(0) != 1 &&
            quantScaleOptional->GetViewShape().GetDim(0) != nWeight) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "when quantScaleOptional's shape size is 1, it's shape should be [%d], [%d] or [%ld], actual is "
                "%s",
                0, 1, nWeight, op::ToString(quantScaleOptional->GetViewShape()).GetString());
            return false;
        }
    }
    if (quantScaleOptional->GetViewShape().GetDimNum() == OPTIONAL_INPUT_DIM_MAX_VALUE) {
        if (quantScaleOptional->GetViewShape().GetDim(0) != 1 ||
            quantScaleOptional->GetViewShape().GetDim(1) != nWeight) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "when quantScaleOptional's shape size is 2, it's shape should be [%d, %ld], actual is %s.", 1, nWeight,
                op::ToString(quantScaleOptional->GetViewShape()).GetString());
            return false;
        }
    }
    if (!IsViewShapeSame(quantOffsetOptional, quantScaleOptional)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "expected tensor for quantOffsetOptional to have same size as tensor for "
            "quantScaleOptional, but %s does not equal %s.",
            op::ToString(quantOffsetOptional->GetViewShape()).GetString(),
            op::ToString(quantScaleOptional->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static bool CheckBiasShape(const aclTensor* biasOptional, int64_t nWeight)
{
    if (biasOptional == nullptr) {
        return true;
    }
    // 支持(0),（n）或者(1,n)
    if (biasOptional->GetViewShape().GetDimNum() == 1 && biasOptional->GetViewShape().GetDim(0) != 0 &&
        biasOptional->GetViewShape().GetDim(0) != nWeight) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "when bias's shape size is 1, it's shape should be [%d], or [%ld], actual is %s.",
            0, nWeight, op::ToString(biasOptional->GetViewShape()).GetString());
        return false;
    }
    if (biasOptional->GetViewShape().GetDimNum() == BIAS_DIM_MAX_VALUE &&
        (biasOptional->GetViewShape().GetDim(0) != 1 || biasOptional->GetViewShape().GetDim(1) != nWeight)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "when bias's shape size is 2, it's shape should be [%d, %ld], actual is %s.", 1,
            nWeight, op::ToString(biasOptional->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static bool CheckXWeightDimValue(const aclTensor* weight, int64_t kX, int64_t mX, bool transposeX, bool transposeWeight)
{
    int64_t kWeight = GetWeightK(weight);
    int64_t nWeight = GetWeightN(weight);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND310P &&
        op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510 &&
        (kX > M_K_N_MAX_VALUE || nWeight > M_K_N_MAX_VALUE || (transposeX && (mX > M_K_N_MAX_VALUE)))) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "k,n shouldn't be larger than %ld, actual k is %ld, n is %ld. When x is transposed, "
            "m shouldn't be larger than %ld, actual m is %ld.",
            M_K_N_MAX_VALUE, kX, nWeight, M_K_N_MAX_VALUE, mX);
        return false;
    }

    if (weight->GetDataType() == DataType::DT_INT8) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510 &&
            weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ &&
            (kX > M_K_N_MAX_VALUE || nWeight > M_K_N_MAX_VALUE)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "k,n shouldn't be larger than %ld, actual k is %ld, n is %ld. ",
                M_K_N_MAX_VALUE, kX, nWeight);
            return false;
        }
    }

    if (weight->GetDataType() == DataType::DT_INT4) {
        if (transposeWeight && ((static_cast<uint64_t>(kWeight) & 1) != 0)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "In the int4 scenario, if weight is transposed, k[%ld] should be an even number.", kWeight);
            return false;
        }
        if (!transposeWeight && ((static_cast<uint64_t>(nWeight) & 1) != 0)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "In the int4 scenario, if weight is not transposed, n[%ld] should be an even number ", nWeight);
            return false;
        }

        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510 &&
            weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ && transposeWeight) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "DAV3510 does not support w4 tranB when weight's dtype is FORMAT_FRACTAL_NZ!");
            return false;
        }
    }

    return true;
}

static bool CheckXWeight(const aclTensor* x, const aclTensor* weight, bool transposeWeight)
{
    size_t kDimX = x->GetViewShape().GetDimNum() - 1;
    int64_t kX = x->GetViewShape().GetDim(kDimX);
    int64_t mX = x->GetViewShape().GetDim(kDimX - 1);
    int64_t kWeight = GetWeightK(weight);
    int64_t nWeight = GetWeightN(weight);
    bool transposeX = IsTransposeLastTwoDims(x);

    if (kX != kWeight) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "x's k and weight's k should be equal, actual x'k is %ld, weight's k is %ld.", kX,
            kWeight);
        return false;
    }

    if (kX < 1 || mX < 1 || nWeight < 1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "m,k,n shouldn't be smaller than %d, actual m is %ld, k is %ld, n is %ld.", 1, mX,
            kX, nWeight);
        return false;
    }

    return CheckXWeightDimValue(weight, kX, mX, transposeX, transposeWeight);
}

bool CheckMultiplyOverflow(int64_t a, int64_t b)
{
    if (a == 0 || b == 0) {
        return false;
    }
    if (a > 0 && b > 0) {
        return a > std::numeric_limits<int64_t>::max() / b;
    }
    return false;
}

static bool CheckXWeightY(const aclTensor* x, const aclTensor* weight, const aclTensor* y, int antiquantGroupSize)
{
    size_t kDimX = x->GetViewShape().GetDimNum() - 1;
    int64_t kX = x->GetViewShape().GetDim(kDimX);
    int64_t mX = x->GetViewShape().GetDim(kDimX - 1);
    int64_t nWeight = GetWeightN(weight);
    size_t nDimY = y->GetViewShape().GetDimNum() - 1;
    int64_t mY = y->GetViewShape().GetDim(nDimY - 1);
    int64_t nY = y->GetViewShape().GetDim(nDimY);

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P && antiquantGroupSize > 0) {
        OP_CHECK(
            !CheckMultiplyOverflow(mX, kX),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "m * k (%ld * %ld) is overflow of int64.", mX, kX),
        return false);
        if (mX * kX > MAX_MK_VALUE) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "m*k shouldn't be larger than %ld, actual m is %ld, k is %ld.", MAX_MK_VALUE,
                mX, kX);
            return false;
        }
    }

    if (mY != mX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y's m and x's m should be equal, actual y's m is %ld, x's m is %ld.", mY, mX);
        return false;
    }

    if (nY != nWeight) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "y's n and weight's n should be equal, actual y's n is %ld, weight's n is %ld.",
            nY, nWeight);
        return false;
    }
    return true;
}

static bool CheckShapeForPerGrp(const aclTensor* weight, int antiquantGroupSize)
{
    int64_t kWeight = GetWeightK(weight);
    int64_t nWeight = GetWeightN(weight);
    // DAV3510 nz没有该限制
    if ((antiquantGroupSize != 0) && (kWeight % antiquantGroupSize) != 0 &&
        op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "when weight's dtype is [int4], weight's format is [FRACTAL_NZ], antiquantGroupSize is larger than 0,"
            "k should be an integer multiple of antiquantGroupSize, "
            "actual k is [%ld], antiquantGroupSize is [%d].",
            kWeight, antiquantGroupSize);
        return false;
    }
    if ((nWeight % N_ALIGN_VALUE) != 0 && op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "when weight's dtype is [int4], weight's format is [FRACTAL_NZ], antiquantGroupSize is larger than 0,"
            "n should be an integer multiple of %ld, actual n is [%ld].",
            N_ALIGN_VALUE, nWeight);
        return false;
    }
    return true;
}

static bool CheckValForWeightInt4Nz(
    const aclTensor* x, const aclTensor* weight, bool transposeWeight, int antiquantGroupSize,
    const aclTensor* antiquantScale)
{
    if ((weight->GetDataType() != DataType::DT_INT4) || (weight->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ)) {
        return true;
    }
    NpuArch npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    // per-group场景
    if (antiquantGroupSize > 0) {
        bool transposeX = IsTransposeLastTwoDims(x);
        if (transposeX || transposeWeight) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "when weight's dtype is [int4], weight's format is [FRACTAL_NZ], and antiquantGroupSize is "
                "larger than 0, x and weight should not be transposed, "
                "actual x is [%s], weight is [%s].",
                (transposeX ? "transposed" : "not transposed"), (transposeWeight ? "transposed" : "not transposed"));
            return false;
        }
        if ((antiquantGroupSize != ANTIQUANT_GRP_SIZE128) && (antiquantGroupSize != ANTIQUANT_GRP_SIZE64) &&
            (npuArch != NpuArch::DAV_3510)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "when weight's dtype is [int4], weight's format is [FRACTAL_NZ], antiquantGroupSize should be "
                "%ld or %ld, but actual antiquantGroupSize is [%d].",
                ANTIQUANT_GRP_SIZE128, ANTIQUANT_GRP_SIZE64, antiquantGroupSize);
            return false;
        }
        CHECK_RET(CheckShapeForPerGrp(weight, antiquantGroupSize), false);
    } else {
        // per-tensor场景
        if (antiquantScale->GetViewShape().GetShapeSize() == 1) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "when weight's dtype is [int4], weight's format is [FRACTAL_NZ], "
                "antiquant mode should not be per-tensor.");
            return false;
        }
        // per-channel场景
        if (!transposeWeight && (npuArch != NpuArch::DAV_3510)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "when weight's dtype is [int4], weight's format is [FRACTAL_NZ], and antiquantGroupSize is 0, "
                "weight should be transposed.");
            return false;
        }
    }

    return true;
}

static bool CheckValForWeightInt8Nz(
    const aclTensor* x, const aclTensor* weight, int antiquantGroupSize,
    const aclTensor* antiquantScale)
{
    if ((weight->GetDataType() != DataType::DT_INT8) || (weight->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ) ||
        op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        return true;
    }

    // per-group场景
    if (antiquantGroupSize > 0) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "when weight's dtype is [int8], weight's format is [FRACTAL_NZ], only support per-channel antiquant mode, "
            "antiquantGroupSize should be 0, but actual antiquantGroupSize is [%d].",
            antiquantGroupSize);
        return false;
    }

    // per-tensor场景
    if (antiquantScale->GetViewShape().GetShapeSize() == 1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "when weight's dtype is [int8], weight's format is [FRACTAL_NZ], "
            "antiquant mode should not be per-tensor.");
        return false;
    } 
    
    // per-channel场景
    if (IsTransposeLastTwoDims(x)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "when weight's dtype is [int8], weight's format is [FRACTAL_NZ], and antiquantGroupSize is 0, "
            "x should not be transposed.");
        return false;
    }

    return true;
}

static bool AdvancedWeightParamsCheck(
    const aclTensor* x, const aclTensor* weight, int antiquantGroupSize, bool transposeWeight, const aclTensor* y)
{
    return CheckXWeight(x, weight, transposeWeight) && CheckXWeightY(x, weight, y, antiquantGroupSize) &&
           CheckShapeValid(x, weight, y);
}

static bool AdvancedShapeParamsCheck(
    const aclTensor* weight, const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional,
    const aclTensor* biasOptional)
{
    int64_t nWeight = GetWeightN(weight);

    return CheckQuantParamShape(quantScaleOptional, quantOffsetOptional, nWeight) &&
           CheckBiasShape(biasOptional, nWeight);
}

static bool AdvancedAntiquantParamsCheck(
    const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale,
    const aclTensor* antiquantOffsetOptional, int antiquantGroupSize)
{
    return CheckAntiquantGroupSize(x, antiquantGroupSize) &&
           CheckAntiquantParamShape(weight, antiquantScale, antiquantOffsetOptional, antiquantGroupSize) &&
           CheckMicroScalingCondition(weight, antiquantScale, antiquantOffsetOptional, antiquantGroupSize);
}

static bool AdvancedQuantParamsNullCheckForWeightInt4Nz(
    const aclTensor* weight, const aclTensor* antiquantScale, const aclTensor* quantScaleOptional,
    const aclTensor* quantOffsetOptional, bool transposeWeight)
{
    if ((weight->GetDataType() != DataType::DT_INT4) || (weight->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ)) {
        return true;
    }

    if ((quantScaleOptional != nullptr) || (quantOffsetOptional != nullptr)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "when weight's dtype is [int4], weight's format is [FRACTAL_NZ], y's dtype should not be [int8].");
        return false;
    }

    OP_CHECK(
        !(weight->GetDataType() == DataType::DT_FLOAT4_E2M1 &&
          antiquantScale->GetDataType() == DataType::DT_FLOAT8_E8M0 &&
          weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ && transposeWeight),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Only supported transposeWeight false on fp4 micro-scaling condition."),
        return false);

    return true;
}

static bool AdvancedParamsCheck(
    const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale, int antiquantGroupSize,
    bool transposeWeight)
{
    return CheckValForWeightInt4Nz(x, weight, transposeWeight, antiquantGroupSize, antiquantScale) &&
           CheckValForWeightInt8Nz(x, weight, antiquantGroupSize, antiquantScale);
}

static aclnnStatus CheckSocValid()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    NpuArch npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    switch (npuArch) {
        case NpuArch::DAV_2201:
        case NpuArch::DAV_3510:
        case NpuArch::DAV_2002:
            break;
        default: {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
            return ACLNN_ERR_RUNTIME_ERROR;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckNzSocValid()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    NpuArch npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    switch (npuArch) {
        case NpuArch::DAV_3510:
            break;
        default: {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "aclnn nz support for %s is not implemented", op::ToString(socVersion).GetString());
            return ACLNN_ERR_RUNTIME_ERROR;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus IsNzFormat(const aclTensor* weight)
{
    return weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ ||
           weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ_C0_2;
}

static bool CheckNotNull(
    const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale, const aclTensor* y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(antiquantScale, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static bool CheckOptionalNotNull(const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
        if (quantScaleOptional != nullptr || quantOffsetOptional != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Current Soc do not support quantScaleOptional or quantOffsetOptional");
            return false;
        }
    }
    return true;
}

static bool CheckAntiquantForFixpipe(const aclTensor* antiquantScale, const aclTensor* antiquantOffsetOptional)
{
    if (antiquantOffsetOptional->GetDataType() == DataType::DT_INT32) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "antiquantOffset's dtype only support DT_FLOAT16 and DT_BF16, "
                "actual antiquantOffset's dtype is DT_INT32.");
            return false;
        }

        if (antiquantScale->GetDataType() != DataType::DT_UINT64 &&
            antiquantScale->GetDataType() != DataType::DT_INT64) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "antiquantScale's dtype must be DT_UINT64 or DT_INT64 when antiquantOffset's dtype is DT_INT32, "
                "actual antiquantScale's dtype is [%s].",
                op::ToString(antiquantScale->GetDataType()).GetString());
            return false;
        }
    }
    return true;
}

static bool CheckXYDtypeValidForFixpipe(
    const aclTensor* x, const aclTensor* antiquantOffsetOptional, const aclTensor* weight, const aclTensor* y)
{
    if (x->GetDataType() != DataType::DT_FLOAT16) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "x's dtype must be DT_FLOAT16 when antiquantScale's dtype is DT_UINT64/DT_INT64, actual x's dtype is [%s].",
            op::ToString(x->GetDataType()).GetString());
        return false;
    }
    if (weight->GetDataType() != DataType::DT_INT8) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "weight's dtype must be DT_INT8 when antiquantScale's dtype is DT_UINT64/DT_INT64,"
            "actual weight's dtype is [%s].",
            op::ToString(weight->GetDataType()).GetString());
        return false;
    }
    if (antiquantOffsetOptional != nullptr) {
        if (antiquantOffsetOptional->GetDataType() != DataType::DT_INT32) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "antiquantOffset's dtype must be DT_INT32 when antiquantScale's dtype is DT_UINT64/DT_INT64, "
                "actual antiquantOffset's dtype is [%s].",
                op::ToString(antiquantOffsetOptional->GetDataType()).GetString());
            return false;
        }
    }
    if (y->GetDataType() != DataType::DT_FLOAT16) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "y's dtype should be [DT_FLOAT16] when antiquantScale's dtype is [DT_UINT64/DT_INT64], actual y's "
            "dtype is [%s].",
            op::ToString(y->GetDataType()).GetString());
        return false;
    }
    return true;
}

static bool CheckXDtypeValidForNormal(
    const aclTensor* x, const aclTensor* antiquantScale, const aclTensor* antiquantOffsetOptional)
{
    if (antiquantScale->GetDataType() != DataType::DT_FLOAT8_E8M0 &&
        x->GetDataType() != antiquantScale->GetDataType()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "x's dtype must be same as antiquantScale's dtype, actual x's "
            "dtype is [%s], antiquantScale's dtype is [%s].",
            op::ToString(x->GetDataType()).GetString(), op::ToString(antiquantScale->GetDataType()).GetString());
        return false;
    }

    if (antiquantOffsetOptional != nullptr && antiquantOffsetOptional->GetDataType() != antiquantScale->GetDataType()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "antiquantOffsetOptional's dtype must be same as antiquantScale's dtype, actual antiquantOffsetOptional's "
            "dtype is [%s], antiquantScale's dtype is [%s].",
            op::ToString(antiquantOffsetOptional->GetDataType()).GetString(),
            op::ToString(antiquantScale->GetDataType()).GetString());
        return false;
    }
    return true;
}

static bool CheckBiasWithXBf16(const aclTensor* weight, const aclTensor* biasOptional){
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        if (weight->GetDataType() == DataType::DT_FLOAT4_E2M1 || weight->GetDataType() == DataType::DT_FLOAT8_E4M3FN ||
            weight->GetDataType() == DataType::DT_HIFLOAT8) {
            if (biasOptional->GetDataType() != DataType::DT_BF16) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID, "biasOptional's dtype should be [DT_BF16], actual is [%s].",
                    op::ToString(biasOptional->GetDataType()).GetString());
                return false;
            }
        } else if (
            weight->GetDataType() == DataType::DT_INT8 && weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
            if (biasOptional->GetDataType() != DataType::DT_FLOAT) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID, "biasOptional's dtype should be [DT_FLOAT], actual is [%s].",
                    op::ToString(biasOptional->GetDataType()).GetString());
                return false;
            }
        } else {
            if (biasOptional->GetDataType() != DataType::DT_BF16 && biasOptional->GetDataType() != DataType::DT_FLOAT) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID, "biasOptional's dtype should be [DT_FLOAT]/[DT_BF16], actual is [%s].",
                    op::ToString(biasOptional->GetDataType()).GetString());
                return false;
            }
        }
    } else {
        if (biasOptional->GetDataType() != DataType::DT_FLOAT) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "biasOptional's dtype should be [DT_FLOAT], actual is [%s].",
                op::ToString(biasOptional->GetDataType()).GetString());
            return false;
        }
    }

    return true;
}

static bool CheckBiasDtypeValid(const aclTensor* x, const aclTensor* weight, const aclTensor* biasOptional)
{
    if (biasOptional == nullptr) {
        return true;
    }

    if (x->GetDataType() == DataType::DT_FLOAT16 && biasOptional->GetDataType() != DataType::DT_FLOAT16) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "When x dtype is [DF_FLOAT16], biasOptional's dtype should be [DT_FLOAT16], actual is [%s].",
            op::ToString(biasOptional->GetDataType()).GetString());
        return false;
    }
    if (x->GetDataType() == DataType::DT_BF16) {
        CHECK_RET(CheckBiasWithXBf16(weight, biasOptional), false);
    }
    return true;
}

static bool CheckScaleDtypeValid(
    const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale,
    const aclTensor* antiquantOffsetOptional, const aclTensor* y)
{
    if (antiquantOffsetOptional != nullptr) {
        CHECK_RET(CheckAntiquantForFixpipe(antiquantScale, antiquantOffsetOptional), false);
    }

    if ((antiquantScale->GetDataType() == DataType::DT_UINT64) ||
        (antiquantScale->GetDataType() == DataType::DT_INT64)) {
        CHECK_RET(CheckXYDtypeValidForFixpipe(x, antiquantOffsetOptional, weight, y), false);
    } else {
        CHECK_RET(CheckXDtypeValidForNormal(x, antiquantScale, antiquantOffsetOptional), false);
    }

    return true;
}

static bool CheckDtypeValid(
    const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale, const aclTensor* quantScaleOptional,
    const aclTensor* y)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, GetXDtypeSupportList(), return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(weight, GetWeightDtypeSupportList(), return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(antiquantScale, GetAntiQuantScaleDtypeSupportList(), return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, GetYDtypeSupportList(), return false);

    if (quantScaleOptional != nullptr && y->GetDataType() != DataType::DT_INT8) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "when quantScaleOptional is not null, y's dtype should be [DT_INT8], actual is [%s].",
            op::ToString(y->GetDataType()).GetString());
        return false;
    }
    if (quantScaleOptional == nullptr && y->GetDataType() != x->GetDataType()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "when quantScaleOptional is null, y's dtype should be same as x [%s], actual is [%s].",
            op::ToString(x->GetDataType()).GetString(), op::ToString(y->GetDataType()).GetString());
        return false;
    }

    if (weight != nullptr && weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ &&
        op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        if (weight->GetDataType() != DataType::DT_INT4 && weight->GetDataType() != DataType::DT_FLOAT4_E2M1 &&
            weight->GetDataType() != DataType::DT_INT8) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "when weight's format is FRACTAL_NZ, weight's dtype only support DT_INT4、DT_FLOAT4_E2M1 and DT_INT8, "
                "actual is [%s].",
                op::ToString(weight->GetDataType()).GetString());
            return false;
        }
    }
    return true;
}

static bool CheckOptionalFormatValid(
    const aclTensor* antiquantOffsetOptional, const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional)
{
    CHECK_RET(IsFormatSupport(antiquantOffsetOptional, Format::FORMAT_ND, "antiquantOffsetOptional"), false);
    CHECK_RET(IsFormatSupport(quantScaleOptional, Format::FORMAT_ND, "quantScaleOptional"), false);
    CHECK_RET(IsFormatSupport(quantOffsetOptional, Format::FORMAT_ND, "quantOffsetOptional"), false);
    return true;
}

static bool CheckFormatValid(
    const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale, const aclTensor* biasOptional)
{
    x = SetTensorToNDFormat(x);
    CHECK_RET(IsFormatSupport(x, Format::FORMAT_ND, "x"), false);
    weight = SetTensorToNDFormat(weight);
    if (weight != nullptr && weight->GetStorageFormat() != Format::FORMAT_ND &&
        weight->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "weight's format should be ND or FRACTAL_NZ. actual is [%s].",
            op::ToString(weight->GetStorageFormat()).GetString());
        return false;
    }

    CHECK_RET(IsFormatSupport(antiquantScale, Format::FORMAT_ND, "antiquantScale"), false);
    biasOptional = SetTensorToNDFormat(biasOptional);
    CHECK_RET(IsFormatSupport(biasOptional, Format::FORMAT_ND, "biasOptional"), false);
    return true;
}

static bool CheckDimValid(const aclTensor* x, const aclTensor* weight, const aclTensor* y)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
        CHECK_RET(IsDimSupport(x, DIM_RANGE_WITH_BATCH, "x"), false);
        CHECK_RET(IsDimSupport(weight, DIM_RANGE_WITH_BATCH, "weight"), false);
        CHECK_RET(IsDimSupport(y, DIM_RANGE_WITH_BATCH, "y"), false);
    } else {
        CHECK_RET(IsDimSupport(x, DIM_RANGE_WITHOUT_BATCH, "x"), false);
        CHECK_RET(IsDimSupport(weight, DIM_RANGE_WITHOUT_BATCH, "weight"), false);
        CHECK_RET(IsDimSupport(y, DIM_RANGE_WITHOUT_BATCH, "y"), false);
    }
    return true;
}

static bool CheckBiasOptionalDimValid(const aclTensor* biasOptional)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
        CHECK_RET(IsDimSupport(biasOptional, BIAS_DIM_RANGE_OPTIONAL_INPUT, "biasOptional"), false);
    } else {
        CHECK_RET(IsDimSupport(biasOptional, DIM_RANGE_OPTIONAL_INPUT, "biasOptional"), false);
    }
    return true;
}

static bool CheckDimRangeOptionalInputValid(
    const aclTensor* antiquantScale, const aclTensor* antiquantOffsetOptional, const aclTensor* quantScaleOptional,
    const aclTensor* quantOffsetOptional)
{
    CHECK_RET(IsDimSupport(antiquantScale, DIM_RANGE_OPTIONAL_INPUT, "antiquantScale"), false);
    CHECK_RET(IsDimSupport(antiquantOffsetOptional, DIM_RANGE_OPTIONAL_INPUT, "antiquantOffsetOptional"), false);
    CHECK_RET(IsDimSupport(quantScaleOptional, DIM_RANGE_OPTIONAL_INPUT, "quantScaleOptional"), false);
    CHECK_RET(IsDimSupport(quantOffsetOptional, DIM_RANGE_OPTIONAL_INPUT, "quantOffsetOptional"), false);
    return true;
}

static bool CheckQuantScale(const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional)
{
    if (quantScaleOptional == nullptr && quantOffsetOptional != nullptr) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "The sence quantScaleOptional is null and quantOffsetOptional is't null can't support.");
        return false;
    }
    if (quantScaleOptional == nullptr) {
        return true;
    }

    if (quantScaleOptional->GetDataType() != DataType::DT_UINT64) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "The type of quantScaleOptional only supports [UINT64], actual is [%s]. You should call the "
            "aclnnTransQuantParam() "
            "interface.",
            op::ToString(quantScaleOptional->GetDataType()).GetString());
        return false;
    }
    return true;
}

static aclnnStatus AntiQuantScaleAndOffsetContiguousCheck(const aclTensor* antiquantScale,
                                                          const aclTensor* antiquantOffsetOptional,
                                                          bool transposeWeight) {
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        if (antiquantScale->GetViewShape().GetDimNum() == ANTIQUANT_DIM_MAX_VALUE &&
            antiquantScale->GetViewShape().GetDim(0) != 1 && antiquantScale->GetViewShape().GetDim(1) != 1) {
            OP_CHECK(
                (transposeWeight && !IsContiguous(antiquantScale)) ||
                    (!transposeWeight && IsContiguous(antiquantScale)),
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "when weight is contiguous or transpose last two dims, antiquantScale tensor should be "
                    "consistent with it."),
                return ACLNN_ERR_PARAM_INVALID);
        }

        if (antiquantOffsetOptional != nullptr &&
            antiquantOffsetOptional->GetViewShape().GetDimNum() == ANTIQUANT_DIM_MAX_VALUE &&
            antiquantOffsetOptional->GetViewShape().GetDim(0) != 1 &&
            antiquantOffsetOptional->GetViewShape().GetDim(1) != 1) {
            OP_CHECK(
                (transposeWeight && !IsContiguous(antiquantOffsetOptional)) ||
                    (!transposeWeight && IsContiguous(antiquantOffsetOptional)),
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "when weight is contiguous or transpose last two dims, antiquantOffsetOptional tensor should "
                    "be consistent with it."),
                return ACLNN_ERR_PARAM_INVALID);
        }
    } else {
        OP_CHECK(
            transposeWeight || IsContiguous(antiquantScale),
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "only support antiquantScale tensor is contiguous or transpose last two dims."),
            return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK(
            antiquantOffsetOptional == nullptr || transposeWeight ||
                (!transposeWeight && IsContiguous(antiquantOffsetOptional)),
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "only support antiquantOffsetOptional tensor is contiguous or transpose last two dims."),
            return ACLNN_ERR_PARAM_INVALID);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus ContiguousCheck(
    const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale,
    const aclTensor* antiquantOffsetOptional, const aclTensor* y)
{
    bool transposeX = IsTransposeLastTwoDims(x);
    OP_CHECK(
        transposeX || IsContiguous(x),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support x tensor is contiguous or transpose last two dims."),
        return ACLNN_ERR_PARAM_INVALID);

    bool transposeWeight = IsTransposeLastTwoDims(weight);
    OP_CHECK(
        transposeWeight || IsContiguous(weight),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support weight tensor is contiguous or transpose last two dims."),
        return ACLNN_ERR_PARAM_INVALID);

    if (weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ &&
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
        OP_CHECK(
            IsContiguous(antiquantScale),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support antiquantScale tensor is contiguous."),
            return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK(
            antiquantOffsetOptional == nullptr || IsContiguous(antiquantOffsetOptional),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support antiquantOffsetOptional tensor is contiguous."),
            return ACLNN_ERR_PARAM_INVALID);
    } else {
        aclnnStatus antiQuantRes =
            AntiQuantScaleAndOffsetContiguousCheck(antiquantScale, antiquantOffsetOptional, transposeWeight);
        CHECK_RET(antiQuantRes == ACLNN_SUCCESS, antiQuantRes);
    }

    OP_CHECK(
        IsContiguous(y), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support y tensor is contiguous."),
        return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus OptionalContiguousCheck(
    const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional, const aclTensor* biasOptional)
{
    OP_CHECK(
        quantScaleOptional == nullptr || IsContiguous(quantScaleOptional),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support quantScaleOptional tensor is contiguous."),
        return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(
        quantOffsetOptional == nullptr || IsContiguous(quantOffsetOptional),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support quantOffsetOptional tensor is contiguous."),
        return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK(
        biasOptional == nullptr || IsContiguous(biasOptional),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support biasOptional tensor is contiguous."),
        return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus BasicParamsScaleCheck(
    const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale,
    const aclTensor* antiquantOffsetOptional, const aclTensor* y)
{
    CHECK_RET(CheckScaleDtypeValid(x, weight, antiquantScale, antiquantOffsetOptional, y), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus BasicParamsBiasCheck(
    const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale, const aclTensor* biasOptional)
{
    CHECK_RET(CheckBiasDtypeValid(x, weight, biasOptional), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckFormatValid(x, weight, antiquantScale, biasOptional), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckBiasOptionalDimValid(biasOptional), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus BasicOptionalParamsCheck(
    const aclTensor* antiquantScale, const aclTensor* antiquantOffsetOptional, const aclTensor* quantScaleOptional,
    const aclTensor* quantOffsetOptional)
{
    CHECK_RET(
        CheckOptionalFormatValid(antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional),
        ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(
        CheckDimRangeOptionalInputValid(
            antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional),
        ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckQuantScale(quantScaleOptional, quantOffsetOptional), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus BasicParamsCheck(
    const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale, const aclTensor* quantScaleOptional,
    const aclTensor* y)
{
    CHECK_RET(CheckDtypeValid(x, weight, antiquantScale, quantScaleOptional, y), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckDimValid(x, weight, y), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static inline bool CreateTransposedView(const aclTensor*& contiguousTensor, aclOpExecutor* executor)
{
    // 创建一个连续的Tensor，其ViewShape是输入Tensor ViewShape后两维度的转置
    if (contiguousTensor == nullptr || contiguousTensor->GetViewShape().GetDimNum() == 1) {
        return true;
    }
    contiguousTensor = executor->CreateView(
        contiguousTensor, SwapLastTwoDimValue(contiguousTensor->GetViewShape()), contiguousTensor->GetViewOffset());
    CHECK_RET(contiguousTensor != nullptr, false);
    return true;
}

static inline bool CreateContiguous(const aclTensor*& contiguousTensor, aclOpExecutor* executor)
{
    // 根据输入Tensor的ViewShape，创建一个连续的Tensor
    if (contiguousTensor == nullptr) {
        return true;
    }
    contiguousTensor = l0op::Contiguous(contiguousTensor, executor);
    CHECK_RET(contiguousTensor != nullptr, false);
    return true;
}

static inline bool TensorContiguousProcess(const aclTensor*& contiguousTensor, bool transpose, aclOpExecutor* executor)
{
    // 对于支持带Transpose的非连续输入Tensor做连续处理
    if (transpose) {
        return CreateTransposedView(contiguousTensor, executor);
    } else {
        return CreateContiguous(contiguousTensor, executor);
    }
}

static aclIntArray* GetPerm(int64_t dim, aclOpExecutor* executor)
{
    std::vector<int64_t> valuePerm(dim);
    for (int64_t i = 0; i < dim; i++) {
        valuePerm[i] = i;
    }

    // Transpose操作仅针对最后两个维度
    std::swap(valuePerm[dim - 1], valuePerm[dim - 2]);
    return executor->AllocIntArray(valuePerm.data(), dim);
}

static aclnnStatus TransposeAndTransDataForInputs(
    const aclTensor*& weight, const aclTensor*& antiquantScale, const aclTensor*& antiquantOffsetOptional,
    bool& transposeWeight, aclOpExecutor* executor)
{
    if (weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        return ACLNN_SUCCESS;
    }
    if (!transposeWeight) {
        // 在310P时kernel侧只支持weight(n, k)输入，当(k,
        // n)输入时需要对输入的weight以及antiquantScale、antiquantOffset做转置
        auto perm = GetPerm(weight->GetViewShape().GetDimNum(), executor);
        CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
        weight = l0op::Transpose(weight, perm, executor);
        CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
        //  当antiquantScale的shape维度为1时不需要做转置
        if (antiquantScale->GetViewShape().GetDimNum() > 1) {
            perm = GetPerm(antiquantScale->GetViewShape().GetDimNum(), executor);
            CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
            antiquantScale = l0op::Transpose(antiquantScale, perm, executor);
            CHECK_RET(antiquantScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        // 当antiquantOffset的shape维度为1时不需要做转置
        if (antiquantOffsetOptional != nullptr && antiquantOffsetOptional->GetViewShape().GetDimNum() > 1) {
            perm = GetPerm(antiquantOffsetOptional->GetViewShape().GetDimNum(), executor);
            CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
            antiquantOffsetOptional = l0op::Transpose(antiquantOffsetOptional, perm, executor);
            CHECK_RET(antiquantOffsetOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        transposeWeight = !transposeWeight;
    }
    weight = l0op::TransData(weight, Format::FORMAT_FRACTAL_NZ, 0, executor);
    CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus TransposeAndTransDataForInputsGroup(
    const aclTensor*& weight, const aclTensor*& antiquantScale, const aclTensor*& antiquantOffsetOptional,
    bool& transposeWeight, aclOpExecutor* executor)
{
    if (weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ || transposeWeight) {
        // 在310P时PER_GROUP场景，当weight(n,k)输入时需要对输入的antiquantScale、antiquantOffset做转置
        if (antiquantScale->GetViewShape().GetDimNum() > 1) {
            auto perm = GetPerm(antiquantScale->GetViewShape().GetDimNum(), executor);
            CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
            antiquantScale = l0op::Transpose(antiquantScale, perm, executor);
            CHECK_RET(antiquantScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        if (antiquantOffsetOptional != nullptr && antiquantScale->GetViewShape().GetDimNum() > 1) {
            auto perm = GetPerm(antiquantOffsetOptional->GetViewShape().GetDimNum(), executor);
            CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
            antiquantOffsetOptional = l0op::Transpose(antiquantOffsetOptional, perm, executor);
            CHECK_RET(antiquantOffsetOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    if (weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        return ACLNN_SUCCESS;
    }
    if (!transposeWeight) {
        // 在310P时kernel侧只支持weight(n, k)输入，当(k, n)输入时需要对输入的weight做转置
        auto perm = GetPerm(weight->GetViewShape().GetDimNum(), executor);
        CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
        weight = l0op::Transpose(weight, perm, executor);
        CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
        transposeWeight = !transposeWeight;
    }
    weight = l0op::TransData(weight, Format::FORMAT_FRACTAL_NZ, 0, executor);
    CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

bool IsTransLastTwoDims(const aclTensor* tensor)
{
    // 相对于公共仓接口区别于，输入shape仅支持2维，在tensor输入shape为（1, 1）时返回true
    if (tensor->GetViewShape().GetDimNum() != 2) {
        return false;
    }
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - 2;
    if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        return true;
    }
    return false;
}

static bool SetShapeStrideForNZ(const aclTensor* weight, aclTensor* weightTemp, bool transposeWeight)
{
    if (!transposeWeight) {
        op::Strides newStrides = weight->GetViewStrides();
        auto strideSize = newStrides.size();
        OP_CHECK(
            strideSize >= DIM_RANGE_WITHOUT_BATCH.front() && strideSize <= DIM_RANGE_WITHOUT_BATCH.back(),
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "The dim of weight's strides should be in range [%lu, %lu]. actual is [%lu].",
                DIM_RANGE_WITHOUT_BATCH.front(), DIM_RANGE_WITHOUT_BATCH.back(), strideSize),
            return false);
        // 当transposeWeight=false时，viewStride的倒数第二维要放大8倍， 即(n/8，1) -> (n, 1)
        newStrides[strideSize - SECOND_LAST_DIM] *= INT4_NUMS_IN_INT32;
        weightTemp->SetViewStrides(newStrides);
    }
    auto newOriginalShape =
        transposeWeight ? SwapLastTwoDimValue(weightTemp->GetViewShape()) : weightTemp->GetViewShape();
    weightTemp->SetOriginalShape(newOriginalShape);
    // storageShape的倒数第一维要放大8倍， 即(n/64,k/16,16,8) -> (n/64,k/16,16,64)
    auto storageShape = weight->GetStorageShape();
    auto storageShapeDim = storageShape.GetDimNum();
    storageShape[storageShapeDim - 1] *= INT4_NUMS_IN_INT32;
    weightTemp->SetStorageShape(storageShape);
    return true;
}

static bool CastQuantScaleOptionalToUint64(
    const aclTensor* quantScaleOptional, const aclTensor*& tensorQuantScaleOptional, aclOpExecutor* executor)
{
    auto quantScaleOptionalTemp = executor->CreateView(
        quantScaleOptional, quantScaleOptional->GetViewShape(), quantScaleOptional->GetViewOffset());
    CHECK_RET(quantScaleOptionalTemp != nullptr, false);
    quantScaleOptionalTemp->SetDataType(DataType::DT_UINT64);
    OP_LOGD("The correction of quantScale's datatype from int64 to uint64 is completed.");
    tensorQuantScaleOptional = quantScaleOptionalTemp;
    return true;
}

static aclnnStatus ModifyTensorDtype(
    const aclTensor*& tensorRef, aclTensor* tensorMod, DataType dtype, aclOpExecutor* executor)
{
    auto tensorTmp = tensorMod == nullptr ?
                         executor->CreateView(tensorRef, tensorRef->GetViewShape(), tensorRef->GetViewOffset()) :
                         tensorMod;
    CHECK_RET(tensorTmp != nullptr, ACLNN_ERR_INNER_NULLPTR);
    tensorTmp->SetDataType(dtype);
    tensorRef = tensorTmp;
    return ACLNN_SUCCESS;
}

static aclnnStatus SetNZC0FormatToNZFormat(aclTensor* input) {
    if (input->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ_C0_2) {
        input->SetViewFormat(op::Format::FORMAT_ND);
        input->SetOriginalFormat(op::Format::FORMAT_ND);
        input->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus PackedWeightPreProcess(
    const aclTensor* weight, const aclTensor*& tensorWeight, aclOpExecutor* executor)
{
    CHECK_RET(IsDimSupport(weight, DIM_RANGE_WITHOUT_BATCH, "weight"), ACLNN_ERR_PARAM_INVALID);
    auto viewShape = weight->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    bool transposeWeight = IsTransLastTwoDims(weight);
    if (transposeWeight) {
        // 2含义：当transposeWeIsTransLastTwoDimsight=true时，shape的倒数第2维要放大8倍， 即(k/8, n) -> (k, n)
        viewShape[viewShapeDim - 2] = viewShape[viewShapeDim - 2] * INT4_NUMS_IN_INT32;
    } else {
        // 当transposeWeight=false时，shape的最后一维要放大8倍， 即(k, n/8) -> (k, n)
        viewShape[viewShapeDim - 1] = viewShape[viewShapeDim - 1] * INT4_NUMS_IN_INT32;
    }
    auto weightTemp = executor->CreateView(weight, viewShape, weight->GetViewOffset());
    CHECK_RET(weightTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (transposeWeight) {
        op::Strides newStrides = weight->GetViewStrides();
        auto strideSize = newStrides.size();
        OP_CHECK(
            strideSize >= DIM_RANGE_WITHOUT_BATCH.front() && strideSize <= DIM_RANGE_WITHOUT_BATCH.back(),
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "The dim of weight's strides should be in range [%lu, %lu]. actual is [%lu].",
                DIM_RANGE_WITHOUT_BATCH.front(), DIM_RANGE_WITHOUT_BATCH.back(), strideSize),
            return ACLNN_ERR_PARAM_INVALID);
        // 当transposeWeight=true时，strides的最后一维要放大8倍， 即(1, k/8) -> (1, k)
        newStrides[strideSize - 1] *= INT4_NUMS_IN_INT32;
        weightTemp->SetViewStrides(newStrides);
    }
    if (IsNzFormat(weight)) {
        CHECK_RET(SetShapeStrideForNZ(weight, weightTemp, transposeWeight), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(SetNZC0FormatToNZFormat(weightTemp) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    }
    if (weight->GetDataType() == DataType::DT_INT32) {
        CHECK_RET(
            ModifyTensorDtype(tensorWeight, weightTemp, DataType::DT_INT4, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_PARAM_INVALID);
        OP_LOGD("The conversion of weight from int32 to int4 is completed.");
    }
    if (weight->GetDataType() == DataType::DT_FLOAT) {
        CHECK_RET(
            ModifyTensorDtype(tensorWeight, weightTemp, DataType::DT_FLOAT4_E2M1, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_PARAM_INVALID);
        OP_LOGD("The conversion of weight from fp32 to fp4 is completed.");
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus TensorPreProcess(TupleTensor mandatoryTensors, TupleTensor optionalTensors, aclOpExecutor* executor)
{
    auto& weight = std::get<INDEX_WEIGHT_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& tensorWeight = std::get<INDEX_WEIGHT_BAK_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& antiquantScaleRef = std::get<INDEX_ANTIQUANT_SCALE_REF_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& quantScaleOptional = std::get<INDEX_QUANT_SCALE_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto& tensorQuantScaleOptional = std::get<INDEX_QUANT_SCALE_BAK_IN_OPTIONAL_TUPLE>(optionalTensors);
    // 将int32的输入weight dtype修改为int4。同时ViewShape,ViewStrides也从int32修改为int4所对应的。
    // 采用float32承载float4_e2m1数据，对于float32采用相同处理流程
    if (weight->GetDataType() == DataType::DT_INT32 ||
        (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510 &&
         weight->GetDataType() == DataType::DT_FLOAT)) {
        CHECK_RET(PackedWeightPreProcess(weight, tensorWeight, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    } else if (
        weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ &&
        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
         op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510)) {
        // aclnn转Nz接口调用：
        //     昇腾910B AI处理器/昇腾910_93 AI处理器 weight为nz且带transpose时，weight originshape应该为(n, k)实际为(k,
        //     n) 昇腾910B AI处理器/昇腾910_93 AI处理器 weight为nz且不带transpose时，weight originshape应该为(k,
        //     n)实际为(k, n)
        // torch转Nz接口+aclCreateTensor调用：
        //     昇腾910B AI处理器/昇腾910_93 AI处理器 weight为nz且带transpose时，weight originshape应该为(n,
        //     k)实际为storageshape(k//32, n//16，16, 32) 昇腾910B AI处理器/昇腾910_93 AI处理器
        //     weight为nz且不带transpose时，weight originshape应该为(k, n)实际为storageshape(n//32, k//16，16, 32)
        auto weightTemp = executor->CreateView(weight, weight->GetViewShape(), weight->GetViewOffset());
        CHECK_RET(weightTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);
        weightTemp->SetViewStrides(weight->GetViewStrides());
        weightTemp->SetStorageShape(weight->GetStorageShape());
        auto newOriginalShape =
            IsTransposeLastTwoDims(weight) ? SwapLastTwoDimValue(weight->GetViewShape()) : weight->GetViewShape();
        weightTemp->SetOriginalShape(newOriginalShape);
        tensorWeight = weightTemp;
        OP_LOGD("The correction of weight's Originalshape is completed.");
    }

    if (quantScaleOptional != nullptr && quantScaleOptional->GetDataType() == DataType::DT_INT64) {
        bool ret = CastQuantScaleOptionalToUint64(quantScaleOptional, tensorQuantScaleOptional, executor);
        CHECK_RET(ret, ACLNN_ERR_INNER_NULLPTR);
    }

    // microscaling场景，采用uint8承载float8_e8m0数据，此处需修正antiquantScale dtype
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510 &&
        weight->GetDataType() == DataType::DT_FLOAT && antiquantScaleRef->GetDataType() == DataType::DT_UINT8) {
        CHECK_RET(
            ModifyTensorDtype(antiquantScaleRef, nullptr, DataType::DT_FLOAT8_E8M0, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_PARAM_INVALID);
        OP_LOGD("The conversion of antiquantScale from uint8 to fp8 is completed.");
    }

    return ACLNN_SUCCESS;
}

aclnnStatus CheckContiguous(
    const aclTensor*& x, const aclTensor*& weight, const aclTensor*& antiquantScale,
    const aclTensor*& antiquantOffsetOptional, const aclTensor*& quantScaleOptional,
    const aclTensor*& quantOffsetOptional, const aclTensor*& biasOptional, const int& antiquantGroupSize,
    bool& transposeX, bool& transposeWeight, aclOpExecutor* executor)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_CHECK(
            !transposeX && IsContiguous(x),
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "DAV_3510 not support x transpose and only support x tensor is contiguous."),
            return ACLNN_ERR_PARAM_INVALID);
    }
    CHECK_RET(TensorContiguousProcess(x, transposeX, executor), ACLNN_ERR_INNER_NULLPTR);

    if (weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ &&
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
        // 在310P输入weight为Nz格式时，输入的weight、antiquantScale、antiquantOffset必须连续。
        // 由于weight在输入前已经做过连续处理，而且StorageShape修改为FractalNz的格式，若连续处理会影响到StorageShape
        CHECK_RET(CreateContiguous(antiquantScale, executor), ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(CreateContiguous(antiquantOffsetOptional, executor), ACLNN_ERR_INNER_NULLPTR);
    } else {
        if (weight->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ) {
            CHECK_RET(TensorContiguousProcess(weight, transposeWeight, executor), ACLNN_ERR_INNER_NULLPTR);
        }
        CHECK_RET(TensorContiguousProcess(antiquantScale, transposeWeight, executor), ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(TensorContiguousProcess(antiquantOffsetOptional, transposeWeight, executor), ACLNN_ERR_INNER_NULLPTR);
    }
    // 对于不支持非连续输入的Tensor做连续处理
    CHECK_RET(CreateContiguous(quantScaleOptional, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CreateContiguous(quantOffsetOptional, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CreateContiguous(biasOptional, executor), ACLNN_ERR_INNER_NULLPTR);

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
        if (antiquantGroupSize > 0) {
            auto ret = TransposeAndTransDataForInputsGroup(
                weight, antiquantScale, antiquantOffsetOptional, transposeWeight, executor);
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
        } else {
            auto ret = TransposeAndTransDataForInputs(
                weight, antiquantScale, antiquantOffsetOptional, transposeWeight, executor);
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(
    TupleTensor mandatoryTensors, TupleTensor optionalTensor, int& antiquantGroupSize, bool& transposeWeight)
{
    auto& x = std::get<INDEX_X_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& tensorWeight = std::get<INDEX_WEIGHT_BAK_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& antiquantScale = std::get<INDEX_ANTIQUANT_SCALE_REF_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& y = std::get<INDEX_Y_OUT_MANDTORY_TUPLE>(mandatoryTensors);
    auto& antiquantOffsetOptional = std::get<INDEX_ANTIQUANT_OFFSET_IN_OPTIONAL_TUPLE>(optionalTensor);
    auto& tensorQuantScaleOptional = std::get<INDEX_QUANT_SCALE_BAK_IN_OPTIONAL_TUPLE>(optionalTensor);
    auto& quantOffsetOptional = std::get<INDEX_QUANT_OFFSET_IN_OPTIONAL_TUPLE>(optionalTensor);
    auto& biasOptional = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensor);

    aclnnStatus res = BasicParamsCheck(x, tensorWeight, antiquantScale, tensorQuantScaleOptional, y);
    CHECK_RET(res == ACLNN_SUCCESS, res);

    res = BasicParamsScaleCheck(x, tensorWeight, antiquantScale, antiquantOffsetOptional, y);
    CHECK_RET(res == ACLNN_SUCCESS, res);

    res = BasicParamsBiasCheck(x, tensorWeight, antiquantScale, biasOptional);
    CHECK_RET(res == ACLNN_SUCCESS, res);

    res = BasicOptionalParamsCheck(
        antiquantScale, antiquantOffsetOptional, tensorQuantScaleOptional, quantOffsetOptional);
    CHECK_RET(res == ACLNN_SUCCESS, res);

    bool advanceCheckRes = AdvancedWeightParamsCheck(x, tensorWeight, antiquantGroupSize, transposeWeight, y);
    CHECK_RET(advanceCheckRes, ACLNN_ERR_PARAM_INVALID);

    advanceCheckRes =
        AdvancedAntiquantParamsCheck(x, tensorWeight, antiquantScale, antiquantOffsetOptional, antiquantGroupSize);
    CHECK_RET(advanceCheckRes, ACLNN_ERR_PARAM_INVALID);

    advanceCheckRes =
        AdvancedShapeParamsCheck(tensorWeight, tensorQuantScaleOptional, quantOffsetOptional, biasOptional);
    CHECK_RET(advanceCheckRes, ACLNN_ERR_PARAM_INVALID);

    advanceCheckRes = AdvancedQuantParamsNullCheckForWeightInt4Nz(
        tensorWeight, antiquantScale, tensorQuantScaleOptional, quantOffsetOptional, transposeWeight);
    CHECK_RET(advanceCheckRes, ACLNN_ERR_PARAM_INVALID);

    advanceCheckRes = AdvancedParamsCheck(x, tensorWeight, antiquantScale, antiquantGroupSize, transposeWeight);
    CHECK_RET(advanceCheckRes, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnWeightQuantBatchMatmulGetWorkspaceSizeCommonProcess(
    TupleTensor mandatoryTensors, TupleTensor optionalTensors, TupleAttr attrs, aclOpExecutor* executor)
{
    auto& x = std::get<INDEX_X_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& weight = std::get<INDEX_WEIGHT_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& tensorWeight = std::get<INDEX_WEIGHT_BAK_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& antiquantScale = std::get<INDEX_ANTIQUANT_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& y = std::get<INDEX_Y_OUT_MANDTORY_TUPLE>(mandatoryTensors);
    auto& antiquantScaleRef = std::get<INDEX_ANTIQUANT_SCALE_REF_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& antiquantOffsetOptional = std::get<INDEX_ANTIQUANT_OFFSET_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto& quantScaleOptional = std::get<INDEX_QUANT_SCALE_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto& tensorQuantScaleOptional = std::get<INDEX_QUANT_SCALE_BAK_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto& quantOffsetOptional = std::get<INDEX_QUANT_OFFSET_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto& biasOptional = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto& antiquantGroupSize = std::get<INDEX_ANTIQUANT_GROUPSIZE_IN_ATTR_TUPLE>(attrs);
    auto& transposeX = std::get<INDEX_TRANSPOSE_X_IN_ATTR_TUPLE>(attrs);
    auto& transposeWeight = std::get<INDEX_TRANSPOSE_WEIGHT_IN_ATTR_TUPLE>(attrs);

    aclnnStatus res = ContiguousCheck(x, weight, antiquantScale, antiquantOffsetOptional, y);
    CHECK_RET(res == ACLNN_SUCCESS, res);

    res = OptionalContiguousCheck(quantScaleOptional, quantOffsetOptional, biasOptional);
    CHECK_RET(res == ACLNN_SUCCESS, res);

    res = TensorPreProcess(mandatoryTensors, optionalTensors, executor);
    CHECK_RET(res == ACLNN_SUCCESS, res);

    transposeX = IsTransposeLastTwoDims(x);
    transposeWeight = (weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ &&
                       GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) ?
                          true :
                          IsTransposeLastTwoDims(tensorWeight);
    OP_LOGD("transposeX is %d, transposeWeight is %d", transposeX, transposeWeight);
    res = CheckParams(mandatoryTensors, optionalTensors, antiquantGroupSize, transposeWeight);
    CHECK_RET(res == ACLNN_SUCCESS, res);

    res = CheckContiguous(
        x, tensorWeight, antiquantScaleRef, antiquantOffsetOptional, tensorQuantScaleOptional, quantOffsetOptional,
        biasOptional, antiquantGroupSize, transposeX, transposeWeight, executor);
    CHECK_RET(res == ACLNN_SUCCESS, res);

    return ACLNN_SUCCESS;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
    const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale,
    const aclTensor* antiquantOffsetOptional, const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional,
    const aclTensor* biasOptional, int antiquantGroupSize, const aclTensor* y, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnWeightQuantBatchMatmulV2,
        DFX_IN(
            x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional, biasOptional),
        DFX_OUT(y));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    aclnnStatus socRes = CheckSocValid();
    CHECK_RET(socRes == ACLNN_SUCCESS, socRes);
    CHECK_RET(CheckNotNull(x, weight, antiquantScale, y), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckOptionalNotNull(quantScaleOptional, quantOffsetOptional), ACLNN_ERR_PARAM_NULLPTR);
    const aclTensor* tensorWeight = weight;
    const aclTensor* antiquantScaleRef = antiquantScale;
    const aclTensor* tensorQuantScaleOptional = quantScaleOptional;
    const aclTensor* reservedRef = nullptr;
    bool transposeX = IsTransposeLastTwoDims(x);
    bool transposeWeight = (weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ &&
                            GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) ?
                               true :
                               IsTransposeLastTwoDims(tensorWeight);
    aclnnStatus res = AclnnWeightQuantBatchMatmulGetWorkspaceSizeCommonProcess(
        std::tie(x, weight, tensorWeight, antiquantScale, y, antiquantScaleRef),
        std::tie(
            antiquantOffsetOptional, quantScaleOptional, tensorQuantScaleOptional, quantOffsetOptional, biasOptional,
            reservedRef),
        std::tie(antiquantGroupSize, transposeX, transposeWeight), uniqueExecutor.get());
    CHECK_RET(res == ACLNN_SUCCESS, res);

    // dtype=-1表示输出dtype和输入x dtype一致
    int64_t dtype = tensorQuantScaleOptional != nullptr ? static_cast<int64_t>(y->GetDataType()) : -1;
    auto result = l0op::WeightQuantBatchMatmulV2(
        x, tensorWeight, antiquantScaleRef, antiquantOffsetOptional, tensorQuantScaleOptional, quantOffsetOptional,
        biasOptional, transposeX, transposeWeight, antiquantGroupSize, dtype, 0,
        uniqueExecutor.get()); // 0:v2接口innerPrecise参数默认传0
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_INVALID);

    auto viewCopyResult = l0op::ViewCopy(result, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_PARAM_INVALID);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
    const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale,
    const aclTensor* antiquantOffsetOptional, const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional,
    const aclTensor* biasOptional, int antiquantGroupSize, int innerPrecise, const aclTensor* y,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnWeightQuantBatchMatmulV3,
        DFX_IN(
            x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional, biasOptional),
        DFX_OUT(y));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    aclnnStatus socRes = CheckSocValid();
    CHECK_RET(socRes == ACLNN_SUCCESS, socRes);
    CHECK_RET(CheckNotNull(x, weight, antiquantScale, y), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckOptionalNotNull(quantScaleOptional, quantOffsetOptional), ACLNN_ERR_PARAM_NULLPTR);

    const aclTensor* tensorWeight = weight;
    const aclTensor* antiquantScaleRef = antiquantScale;
    const aclTensor* tensorQuantScaleOptional = quantScaleOptional;
    const aclTensor* reservedRef = nullptr;
    bool transposeX = IsTransposeLastTwoDims(x);
    bool transposeWeight = (weight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ &&
                            GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) ?
                               true :
                               IsTransposeLastTwoDims(tensorWeight);
    aclnnStatus res = CheckInnerPrecise(innerPrecise);
    CHECK_RET(res == ACLNN_SUCCESS, res);
    res = AclnnWeightQuantBatchMatmulGetWorkspaceSizeCommonProcess(
        std::tie(x, weight, tensorWeight, antiquantScale, y, antiquantScaleRef),
        std::tie(
            antiquantOffsetOptional, quantScaleOptional, tensorQuantScaleOptional, quantOffsetOptional, biasOptional,
            reservedRef),
        std::tie(antiquantGroupSize, transposeX, transposeWeight), uniqueExecutor.get());
    CHECK_RET(res == ACLNN_SUCCESS, res);

    // dtype=-1表示输出dtype和输入x dtype一致
    int64_t dtype = tensorQuantScaleOptional != nullptr ? static_cast<int64_t>(y->GetDataType()) : -1;
    auto result = l0op::WeightQuantBatchMatmulV2(
        x, tensorWeight, antiquantScaleRef, antiquantOffsetOptional, tensorQuantScaleOptional, quantOffsetOptional,
        biasOptional, transposeX, transposeWeight, antiquantGroupSize, dtype, innerPrecise, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_INVALID);

    auto viewCopyResult = l0op::ViewCopy(result, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_PARAM_INVALID);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnWeightQuantBatchMatmulNzGetWorkspaceSize(
    const aclTensor* x, const aclTensor* weight, const aclTensor* antiquantScale,
    const aclTensor* antiquantOffsetOptional, const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional,
    const aclTensor* biasOptional, int antiquantGroupSize, const aclTensor* y, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnWeightQuantBatchMatmulNz,
        DFX_IN(
            x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional, biasOptional),
        DFX_OUT(y));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    aclnnStatus socRes = CheckNzSocValid();
    CHECK_RET(socRes == ACLNN_SUCCESS, socRes);
    CHECK_RET(CheckNotNull(x, weight, antiquantScale, y), ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(
        IsNzFormat(weight), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support weight tensor is FRACTAL_NZ."),
        return ACLNN_ERR_PARAM_INVALID);
    const aclTensor* tensorWeight = weight;
    const aclTensor* antiquantScaleRef = antiquantScale;
    const aclTensor* tensorQuantScaleOptional = quantScaleOptional;
    const aclTensor* reservedRef = nullptr;
    bool transposeX = IsTransposeLastTwoDims(x);
    bool transposeWeight = IsTransposeLastTwoDims(tensorWeight);
    aclnnStatus processRes = AclnnWeightQuantBatchMatmulGetWorkspaceSizeCommonProcess(
        std::tie(x, weight, tensorWeight, antiquantScale, y, antiquantScaleRef),
        std::tie(
            antiquantOffsetOptional, quantScaleOptional, tensorQuantScaleOptional, quantOffsetOptional, biasOptional,
            reservedRef),
        std::tie(antiquantGroupSize, transposeX, transposeWeight), uniqueExecutor.get());
    CHECK_RET(processRes == ACLNN_SUCCESS, processRes);

    // dtype=-1表示输出dtype和输入x dtype一致
    int64_t dtype = tensorQuantScaleOptional != nullptr ? static_cast<int64_t>(y->GetDataType()) : -1;
    auto l0Result = l0op::WeightQuantBatchMatmulV2(
        x, tensorWeight, antiquantScaleRef, antiquantOffsetOptional, tensorQuantScaleOptional, quantOffsetOptional,
        biasOptional, transposeX, transposeWeight, antiquantGroupSize, dtype, 0,
        uniqueExecutor.get()); // 0:v2接口innerPrecise参数默认传0
    CHECK_RET(l0Result != nullptr, ACLNN_ERR_PARAM_INVALID);

    auto viewCopyResult = l0op::ViewCopy(l0Result, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_PARAM_INVALID);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnWeightQuantBatchMatmulV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnWeightQuantBatchMatmulV2)
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnWeightQuantBatchMatmulV3(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnWeightQuantBatchMatmulV3)
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnWeightQuantBatchMatmulNz(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnWeightQuantBatchMatmulNz)
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif