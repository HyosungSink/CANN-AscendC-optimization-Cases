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
 * \file quant_matmul_checker.cpp
 * \brief
 */
#include "quant_matmul_checker.h"

using namespace op;
using namespace ge;

namespace {
static constexpr int INDEX_X1_IN_INPUT_TUPLE = 0;
static constexpr int INDEX_X2_IN_INPUT_TUPLE = 1;
static constexpr int INDEX_Y_OFFSET_IN_QUANT_TUPLE = 5;
static constexpr int INDEX_BIAS_IN_QUANT_TUPLE = 6;
static constexpr int INDEX_GROUP_SIZE_IN_QUANT_TUPLE = 7;
static constexpr int INDEX_INTERFACE_TYPE_IN_QUANT_TUPLE = 8;
static constexpr int INDEX_X1_SCALE_IN_QUANT_TUPLE = 0;
static constexpr int INDEX_X2_SCALE_IN_QUANT_TUPLE = 1;
static constexpr int INDEX_Y_SCALE_IN_QUANT_TUPLE = 2;
static constexpr int INDEX_X1_OFFSET_IN_QUANT_TUPLE = 3;
static constexpr int INDEX_X2_OFFSET_IN_QUANT_TUPLE = 4;

static const int X2_FIXED_DIM_NUM_A4W4 = 2;
static const int64_t INT4_NUMS_IN_INT8 = 2;
static const int64_t MIN_DIM_NUM_ND = 2;
static const size_t MX_SCALE_DIM = 3;
static const size_t PENULTIMATE_DIM = 2;
static const size_t BATCH_TAILENDER_DIM = 3;
static const size_t NZ_K1_INDEX = 3;
static const size_t NZ_K1_INDEX_TRANS = 4;
static const int64_t NZ_K0_VALUE_INT8_INT4 = 16;
static const int64_t NZ_K0_VALUE_INT8_TRANS = 32;
static const int64_t NZ_K0_VALUE_INT4_TRANS = 64;
static constexpr int64_t OUTPUT_INFER_FAIL = -1L;
static const int64_t PERGROUP_GROUP_SIZE = 32;
static const int64_t PERGROUP_GROUPSIZEK_SIZE = 256;
static const int64_t MXFP_DIVISOR_SIZE = 64;
static const int64_t MXFP_MULTI_BASE_SIZE = 2;
static const uint64_t PERBLOCK_BLOCK_SIZE = 128;
static const int32_t GROUP_M_OFFSET = 32;
static const int32_t GROUP_N_OFFSET = 16;
static const uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
static const int64_t MICRO_SCALING_ALIGN_NUM = 2;

const std::map<int64_t, std::string> X1_NAME{{3, "x1"}, {4, "x1"}, {5, "x1"}};

const std::map<int64_t, std::string> X2_NAME{{3, "x2"}, {4, "x2"}, {5, "x2"}};

const std::map<int64_t, std::string> OUT_NAME{{3, "out"}, {4, "out"}, {5, "out"}};

const std::map<int64_t, std::string> BIAS_NAME{{3, "bias"}, {4, "bias"}, {5, "bias"}};

const std::map<int64_t, std::string> X1SCALE_NAME{
    {3, "pertokenScaleOptional"}, {4, "pertokenScaleOptional"}, {5, "x1Scale(pertokenScale)"}};

const std::map<int64_t, std::string> X2SCALE_NAME{{3, "scale"}, {4, "scale"}, {5, "x2Scale(scale)"}};

const std::map<int64_t, std::string> X2OFFSET_NAME{{3, "offset"}, {4, "offset"}, {5, "x2Offset(offset)"}};

static const std::initializer_list<op::DataType> X1_X2_L0C2OUT_PERTOKEN_SUPPORT_LIST = {op::DataType::DT_INT4,
                                                                                        op::DataType::DT_INT8};
static const std::initializer_list<op::DataType> UINT64_X2_SCALE_TYPE_SUPPORT_LIST = {op::DataType::DT_UINT64,
                                                                                      op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> X2_INT8_X2_SCALE_TYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT64, op::DataType::DT_INT64, op::DataType::DT_FLOAT, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> INT8_OUT_TYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8, op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> HIF8_OUT_TYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> FP8_OUT_TYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> PERTOKEN_FP16_OUT_BIAS_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> PERTOKEN_BF16_OUT_X2SCALE_SUPPORT_LIST = {
    op::DataType::DT_BF16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> PERTOKEN_BF16_OUT_BIAS_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_BF16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> PERTOKEN_OUT_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                                   op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> BF16_OUT_X2SCALE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                  op::DataType::DT_BF16};                                                                              
static const std::initializer_list<op::DataType> BIAS_TYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_BF16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> FP8_AND_HIF8_COMMON_OUT_TYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> FOUR_BIT_FLOAT_INPUT_LIST = {op::DataType::DT_FLOAT4_E2M1};
static const std::initializer_list<op::DataType> EIGHT_BIT_FLOAT_INPUT_LIST = {op::DataType::DT_FLOAT8_E4M3FN,
                                                                               op::DataType::DT_FLOAT8_E5M2};
static const std::initializer_list<op::DataType> INT32_OUT_X2SCALE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
 	                                                                               op::DataType::DT_BF16};
static inline std::string GetInputName(const std::map<int64_t, std::string> &inputNameMap, int64_t interfaceType)
{
    std::string inputName;
    auto iter = inputNameMap.find(interfaceType);
    if (iter != inputNameMap.end()) {
        inputName = iter->second;
    }
    return inputName;
}

static inline bool OpCheckDtypeNotSupport(int64_t interfaceType, const std::map<int64_t, std::string> &inputNameMap,
                                          const aclTensor *tensor,
                                          const std::initializer_list<op::DataType> &supportList)
{
    if (!CheckType(tensor->GetDataType(), supportList)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor %s not implemented for %s, should be in dtype support list %s.",
                GetInputName(inputNameMap, interfaceType).c_str(),
                op::ToString(tensor->GetDataType()).GetString(), op::ToString(supportList).GetString());
        return false;
    }
    return true;
}

static inline bool OpCheckDtypeNotMatch(int64_t interfaceType, const std::map<int64_t, std::string> &inputNameMap,
                                        const aclTensor *tensor, DataType expectedDtype)
{
    if (tensor->GetDataType() != expectedDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor %s expected dtype is %s but found %s.",
                GetInputName(inputNameMap, interfaceType).c_str(),
                op::ToString(expectedDtype).GetString(), op::ToString(tensor->GetDataType()).GetString());
        return false;
    }
    return true;
}

static inline bool OpCheckWrongDimension(int64_t interfaceType, const std::map<int64_t, std::string> &inputNameMap,
                                         const aclTensor *tensor, size_t expectedDimNum)
{
    if (tensor->GetViewShape().GetDimNum() != expectedDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected %zu dimension input, but got %s with shape %s.",
                expectedDimNum, GetInputName(inputNameMap, interfaceType).c_str(),
                op::ToString(tensor->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static inline std::string RemoveDtInDtype(std::string str) {
    std::string target = "DT_";
    std::string replacement = "";
    size_t pos = str.find(target);
    while (pos != std::string::npos) {
        str.replace(pos, target.length(), replacement);
        pos = str.find(target, pos + replacement.length());
    }
    return str;
}

static inline bool IsInt8Input(const aclTensor *x1, const aclTensor *x2) {
    return x1->GetDataType() == op::DataType::DT_INT8 && x2->GetDataType() == op::DataType::DT_INT8;
}

static inline bool IsInt4Input(const aclTensor *x1, const aclTensor *x2) {
    return x1->GetDataType() == op::DataType::DT_INT4 && x2->GetDataType() == op::DataType::DT_INT4;
}

static inline bool IsInt32Input(const aclTensor *x1, const aclTensor *x2) {
    return x1->GetDataType() == op::DataType::DT_INT32 && x2->GetDataType() == op::DataType::DT_INT32;
}

static inline bool IsIntInput(const aclTensor *x1, const aclTensor *x2) {
    return IsInt8Input(x1, x2) || IsInt4Input(x1, x2) || IsInt32Input(x1, x2);
}

static inline bool IsHif8Input(const aclTensor *x1, const aclTensor *x2) {
    return x1->GetDataType() == op::DataType::DT_HIFLOAT8 && x2->GetDataType() == op::DataType::DT_HIFLOAT8;
}

static inline bool IsMicroScaling(const aclTensor *x1Scale, const aclTensor *x2Scale) {
    if (x1Scale == nullptr) {
        return false;
    }
    return x1Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0 &&
           x2Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0;
}

static inline bool IsFp8Input(const aclTensor *x1, const aclTensor *x2) {
    return (std::find(EIGHT_BIT_FLOAT_INPUT_LIST.begin(), EIGHT_BIT_FLOAT_INPUT_LIST.end(), x1->GetDataType()) !=
            EIGHT_BIT_FLOAT_INPUT_LIST.end()) &&
            (std::find(EIGHT_BIT_FLOAT_INPUT_LIST.begin(), EIGHT_BIT_FLOAT_INPUT_LIST.end(), x2->GetDataType()) !=
            EIGHT_BIT_FLOAT_INPUT_LIST.end());
}

static inline bool IsPerblock(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                              const aclTensor *x2Scale)
{
    if (x1Scale == nullptr) {
        return false;
    }
    return (
        (IsInt8Input(x1, x2) || IsFp8Input(x1, x2) || IsHif8Input(x1, x2)) &&
        x1Scale->GetDataType() == op::DataType::DT_FLOAT && x1Scale->GetViewShape().GetDimNum() > 1 &&
        x2Scale->GetDataType() == op::DataType::DT_FLOAT && x2Scale->GetViewShape().GetDimNum() > 1);
}

static inline bool IsFp4Input(const aclTensor *x1, const aclTensor *x2) {
    return (std::find(FOUR_BIT_FLOAT_INPUT_LIST.begin(), FOUR_BIT_FLOAT_INPUT_LIST.end(), x1->GetDataType()) !=
            FOUR_BIT_FLOAT_INPUT_LIST.end()) &&
            (std::find(FOUR_BIT_FLOAT_INPUT_LIST.begin(), FOUR_BIT_FLOAT_INPUT_LIST.end(), x2->GetDataType()) !=
            FOUR_BIT_FLOAT_INPUT_LIST.end());
}

template <typename T>
static inline auto CeilDiv(T x, T y) -> T {
  if (y != 0 && x != 0) {
    const T quotient = x / y;
    return (x % y != 0) ? (quotient + 1) : quotient;
  }
  return x;
}

template <typename T>
static inline auto CeilAlign(T x, T y) -> T {
  return CeilDiv(x, y) * y;
}

template <typename T>
static inline bool IsAligned(T num, T factor)
{
    if (factor == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The divisor cannot be zero.");
        return false;
    }
    return num % factor == 0 && num > 0;
}
}

void QuantMatmulChecker::Init()
{
    socVersion_ = GetCurrentPlatformInfo().GetSocVersion();
    npuArch_ = op::GetCurrentPlatformInfo().GetCurNpuArch();
    x1_ = std::get<INDEX_X1_IN_INPUT_TUPLE>(inputTensors_);
    x2_ = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors_);
    x1Scale_ = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors_);
    x2Scale_ = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors_);
    yScale_ = std::get<INDEX_Y_SCALE_IN_QUANT_TUPLE>(quantTensors_);
    x1Offset_ = std::get<INDEX_X1_OFFSET_IN_QUANT_TUPLE>(quantTensors_);
    x2Offset_ = std::get<INDEX_X2_OFFSET_IN_QUANT_TUPLE>(quantTensors_);
    yOffset_ = std::get<INDEX_Y_OFFSET_IN_QUANT_TUPLE>(quantTensors_);
    bias_ = std::get<INDEX_BIAS_IN_QUANT_TUPLE>(quantTensors_);
    groupSize_ = std::get<INDEX_GROUP_SIZE_IN_QUANT_TUPLE>(quantTensors_);
    transposeX1_ = std::get<INDEX_X1_IN_INPUT_TUPLE>(boolsTrans_);
    transposeX2_ = std::get<INDEX_X2_IN_INPUT_TUPLE>(boolsTrans_);
    interfaceType_ = std::get<INDEX_INTERFACE_TYPE_IN_QUANT_TUPLE>(quantTensors_);
    if (out_ != nullptr) {
        outType_ = out_->GetDataType();
    }
    if (x2Scale_ != nullptr) {
        x2ScaleType_ = x2Scale_->GetDataType();
    }
    if (x1_ != nullptr && x2_ != nullptr) {
        GetX1X2DimValue();
        isA4W4_ = IsInt4Input(x1_, x2_);
    }
}

bool QuantMatmulChecker::IsA4W4PergroupNonSymmetric(const uint64_t groupSizeK) const
{
    if (x1Scale_ == nullptr || x2Scale_ == nullptr || x2Offset_ == nullptr) {
        return false;
    }

    if (x1Scale_->GetDataType() != op::DataType::DT_FLOAT || x1Scale_->GetViewShape().GetDimNum() <= 1) {
        return false;
    }

    if (x2Scale_->GetDataType() != op::DataType::DT_FLOAT || x2Scale_->GetViewShape().GetDimNum() <= 1) {
        return false;
    }

    if (x2Offset_->GetDataType() != op::DataType::DT_FLOAT16 || x2Offset_->GetViewShape().GetDimNum() <= 1) {
        return false;
    }

    auto bias = std::get<INDEX_BIAS_IN_QUANT_TUPLE>(quantTensors_);
    if (bias != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias is currently not supported.");
        return false;
    }

    int64_t m = (transposeX1_ ? x1_->GetViewShape().GetDim(1) : x1_->GetViewShape().GetDim(0));
    int64_t k = (transposeX2_ ? x2_->GetViewShape().GetDim(1) : x2_->GetViewShape().GetDim(0));
    int64_t n = (transposeX2_ ? x2_->GetViewShape().GetDim(0) : x2_->GetViewShape().GetDim(1));

    int32_t ALIGN1024 = 1024;
    int32_t ALIGN256 = 256;
    if (k % ALIGN1024 != 0 || n % ALIGN256 != 0) {
        return false;
    }

    return (
        IsInt4Input(x1_, x2_) && (x1Scale_->GetViewShape().GetDim(0) == m) &&
        (x1Scale_->GetViewShape().GetDim(1) == 1) &&
        (x2Scale_->GetViewShape().GetDim(0) == CeilDiv(k, static_cast<int64_t>(groupSizeK))) &&
        (x2Scale_->GetViewShape().GetDim(1) == n) &&
        (x2Offset_->GetViewShape().GetDim(0) == CeilDiv(k, static_cast<int64_t>(groupSizeK))) &&
        (x2Offset_->GetViewShape().GetDim(1) == n));
}

std::string QuantMatmulChecker::GetX1ScaleName() const
{
    return GetInputName(X1SCALE_NAME, interfaceType_);
}

std::string QuantMatmulChecker::GetX2ScaleName() const
{
    return GetInputName(X2SCALE_NAME, interfaceType_);
}

std::string QuantMatmulChecker::GetX2OffsetName() const
{
    return GetInputName(X2OFFSET_NAME, interfaceType_);
}

bool QuantMatmulChecker::CheckEmptyTensor() const
{
    // scale, out和可选参数已在CheckShape函数校验，无需再次校验空tensor场景。
    if (x1_->IsEmpty() || x2_->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AclnnQuantMatmul do not support empty tensor currently.");
        return false;
    }
    return true;
}

void QuantMatmulChecker::GetX1X2DimValue()
{
    auto x1DimNum = x1_->GetViewShape().GetDimNum();
    auto x2DimNum = x2_->GetViewShape().GetDimNum();
    if (x1DimNum >= MIN_DIM_NUM_ND && x2DimNum >= MIN_DIM_NUM_ND) {
        const op::Shape x1Shape = x1_->GetViewShape();
        const op::Shape x2Shape = x2_->GetViewShape();
        x1KDim_ = transposeX1_ ? x1Shape[x1DimNum - PENULTIMATE_DIM] : x1Shape[x1DimNum - 1];
        x1MDim_ = transposeX1_ ? x1Shape[x1DimNum - 1] : x1Shape[x1DimNum - PENULTIMATE_DIM];
        x2KDim_ = transposeX2_ ? x2Shape[x2DimNum - 1] : x2Shape[x2DimNum - PENULTIMATE_DIM];
        x2NDim_ = transposeX2_ ? x2Shape[x2DimNum - PENULTIMATE_DIM] : x2Shape[x2DimNum - 1];
    }
}

bool QuantMatmulChecker::CheckShapeForWeightNz() const
{
    const op::Shape x2Shape = x2_->GetStorageShape();
    auto x2DimNum = x2_->GetStorageShape().GetDimNum();
    int64_t x2K1Dim = transposeX2_ ? x2Shape[x2DimNum - NZ_K1_INDEX_TRANS] : x2Shape[x2DimNum - NZ_K1_INDEX];
    int64_t nz_k0_value_trans = (x1_->GetDataType() == op::DataType::DT_INT32 || x1_->GetDataType() == op::DataType::DT_INT4) ? 
        NZ_K0_VALUE_INT4_TRANS : NZ_K0_VALUE_INT8_TRANS;
    int64_t aligneValue = transposeX2_ ? nz_k0_value_trans : NZ_K0_VALUE_INT8_INT4;
    int64_t alignedX1K = ((x1KDim_ + aligneValue - 1) / aligneValue) * aligneValue;
    if (alignedX1K != x2K1Dim * aligneValue) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "AlignedK1 should match with k1 times aligneValue. AlignedX1K: %ld, k1 times aligneValue: %ld.",
                alignedX1K, x2K1Dim * aligneValue);
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckKDimValueFp4Fp8WeightNZMicroScaling() const
{
    // fp4内轴偶数校验
    if (IsFp4Input(x1_, x2_)) {
        if (x1KDim_ % MICRO_SCALING_ALIGN_NUM != 0 || x2KDim_ % MICRO_SCALING_ALIGN_NUM != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In mx quantification and x1 and x2 are FLOAT4, \
matrix multiplication is (batchX1, m, k) @ transpose(batchX2, n, k) = (batch, m, n). \
The k-dimension of x1 and x2 should both be even number, actual they are %ld and %ld.", x1KDim_, x2KDim_);
            return false;
        }
    }
    if (CeilDiv(x1KDim_, PERGROUP_GROUP_SIZE) % MICRO_SCALING_ALIGN_NUM != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In mx quantification, when x1 and x2 are FLOAT4, matrix multiplication is  (batchX1, m, k) @ \
transpose(batchX2, n, k) = (batch, m, n). The k-dimension should satisfy that ceil(k, 32) is an even \
number. Actual k-dimension is %ld and ceil(k, 32) = %ld.",
                x1KDim_, CeilDiv(x1KDim_, PERGROUP_GROUP_SIZE));
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckDimValueMicroScaling() const
{
    auto x1ScaleMDimIndex = transposeX1_ ? 1 : 0;
    auto x2ScaleNDimIndex = transposeX2_ ? 0 : 1;
    if (x1Scale_->GetViewShape().GetDim(x1ScaleMDimIndex) != x1MDim_ ||
        x2Scale_->GetViewShape().GetDim(x2ScaleNDimIndex) != x2NDim_) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In mx quantification, the m-dimension of x1 and %s should be same, actual m of x1 is %ld, m of \
x1Scale is %ld, the n-dimension of x2 and %s should be same, actual n of x2 is %ld, n of x2Scale is %ld.",
                GetX1ScaleName().c_str(), x1MDim_, x1Scale_->GetViewShape().GetDim(x1ScaleMDimIndex),
                GetX2ScaleName().c_str(), x2NDim_, x2Scale_->GetViewShape().GetDim(x2ScaleNDimIndex));
        return false;
    }
    auto x1ScaleKDimIndex = transposeX1_ ? 0 : 1;
    auto x2ScaleKDimIndex = transposeX2_ ? 1 : 0;
    if (CeilDiv(x1KDim_, MXFP_DIVISOR_SIZE) != x1Scale_->GetViewShape().GetDim(x1ScaleKDimIndex) ||
        CeilDiv(x2KDim_, MXFP_DIVISOR_SIZE) != x2Scale_->GetViewShape().GetDim(x2ScaleKDimIndex)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In mx quantification, the k-dimension of x1 and %s need to satisfy ceil(k_x1, 64) = k_x1Scale, and \
the k-dimension of x2 and %s need to satisfy ceil(k_x2, 64) = k_x2Scale, actual k_x1 is %ld, k_x1Scale is %ld, \
actual k_x2 is %ld, k_x2Scale is %ld.",
                GetX1ScaleName().c_str(), GetX2ScaleName().c_str(),
                x1KDim_, x1Scale_->GetViewShape().GetDim(x1ScaleKDimIndex),
                x2KDim_, x2Scale_->GetViewShape().GetDim(x2ScaleKDimIndex));
        return false;
    }
    if (x1Scale_->GetViewShape().GetDim(MX_SCALE_DIM - 1) != MXFP_MULTI_BASE_SIZE ||
        x2Scale_->GetViewShape().GetDim(MX_SCALE_DIM - 1) != MXFP_MULTI_BASE_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "In mx quantification, the last dimension of %s and %s should be 2, \
actual last dimension of %s is %ld, actual last dimension of %s is %ld.",
                GetX1ScaleName().c_str(), GetX2ScaleName().c_str(),
                GetX1ScaleName().c_str(), x1Scale_->GetViewShape().GetDim(MX_SCALE_DIM - 1),
                GetX2ScaleName().c_str(), x2Scale_->GetViewShape().GetDim(MX_SCALE_DIM - 1));
        return false;
    }
    if (IsFp4Input(x1_, x2_)) {
        CHECK_RET(CheckKDimValueFp4Fp8WeightNZMicroScaling(), false);
    }
    return true;
}

bool QuantMatmulChecker::CheckGroupSizeShape(uint64_t groupSizeM) const
{
    auto x1DimNum = x1_->GetViewShape().GetDimNum();
    auto x2DimNum = x2_->GetViewShape().GetDimNum();
    auto x1Shape = x1_->GetViewShape();
    auto x1ScaShape = x1Scale_->GetViewShape();
    auto x2Shape = x2_->GetViewShape();
    auto x2ScaShape = x2Scale_->GetViewShape();
    auto mDimNum = x1DimNum - (transposeX1_ ? 1 : PENULTIMATE_DIM);
    auto x1KDimNum = x1DimNum - (transposeX1_ ? PENULTIMATE_DIM : 1);
    auto nDimNum = x2DimNum - (transposeX2_ ? PENULTIMATE_DIM : 1);
    auto x2KDimNum = x2DimNum - (transposeX2_ ? 1 : PENULTIMATE_DIM);
    // m dim
    if (CeilDiv(x1Shape.GetDim(mDimNum), static_cast<int64_t>(groupSizeM)) != x1ScaShape.GetDim(mDimNum)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When quantification mode is G-B or B-B quantification, \
matrix multiplication is (batchx1, m, k) @ (batchX2, k, n) = (batch, m, n), the m-dimension of x1Scale(pertoken_scale) \
and x1 need to satisfy the relationship: ceil(m_x1, groupSizeM) == m_x1Scale, actual m_x1 is %ld, m_x1Scale is %ld, \
groupSizeM is %lu.", x1Shape.GetDim(mDimNum), x1ScaShape.GetDim(mDimNum), groupSizeM);
        return false;
    }

    if (CeilDiv(x1Shape.GetDim(x1KDimNum), static_cast<int64_t>(PERBLOCK_BLOCK_SIZE)) !=
        x1ScaShape.GetDim(x1KDimNum)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When quantification mode is G-B or B-B quantification, \
matrix multiplication is (batchx1, m, k) @ (batchX2, k, n) = (batch, m, n), the k-dimension of x1Scale(pertoken_scale) \
and x1 need to satisfy the relationship: ceil(k_x1, 128) == k_x1Scale, actual k_x1 is %ld, k_x1Scale is %ld.",
                x1Shape.GetDim(x1KDimNum), x1ScaShape.GetDim(x1KDimNum));
        return false;
    }
    if (CeilDiv(x2Shape.GetDim(nDimNum), static_cast<int64_t>(PERBLOCK_BLOCK_SIZE)) != x2ScaShape.GetDim(nDimNum)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When quantification mode is G-B or B-B quantification, \
matrix multiplication is (batchx1, m, k) @ (batchX2, k, n) = (batch, m, n), the n-dimension of x2Scale(scale) and \
x2 need to satisfy the relationship: ceil(n_x2, 128) == n_x2Scale, actual n_x2 is %ld, n_x2Scale is %ld.",
               x2Shape.GetDim(nDimNum), x2ScaShape.GetDim(nDimNum));
        return false;
    }

    if (CeilDiv(x2Shape.GetDim(x2KDimNum), static_cast<int64_t>(PERBLOCK_BLOCK_SIZE)) !=
        x2ScaShape.GetDim(x2KDimNum)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When quantification mode is G-B or B-B quantification, \
matrix multiplication is (batchx1, m, k) @ (batchX2, k, n) = (batch, m, n), the k-dimension of x2Scale(scale) and \
x2 need to satisfy the relationship: ceil(k_x2, 128) == k_x2Scale, actual k_x2 is %ld, k_x2Scale is %ld.",
               x2Shape.GetDim(x2KDimNum), x2ScaShape.GetDim(x2KDimNum));
        return false;
    }
    return true;
}

bool QuantMatmulChecker::ReCalcGroupSize(int64_t inputSize, int64_t scaleSize, int64_t &groupSize,
                                         const char *dimensionName) const
{
    if (scaleSize == 0ULL) {
        std::string scaleName = strcmp(dimensionName, "n") == 0 ? "x2Scale(scale)" : "x1Scale(pertokenScale)";
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The %s dimension of %s is 0, invalid shape!", dimensionName, scaleName.c_str());
        return false;
    }
    if (groupSize == 0ULL) {
        if (inputSize % scaleSize != 0UL) {
            std::string scaleName = "x1Scale(pertokenScale)";
            std::string inputName = "x1";
            if (strcmp(dimensionName, "n") == 0) {
                scaleName = "x2Scale(scale)";
                inputName = "x2";
            }
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The groupSize in %s dimension is 0 and the %s dimension of %s [%ld] is not divisible by \
the %s dimension of %s [%ld], the real groupSize in %s dimension can not be inferred.",
                    dimensionName, dimensionName, inputName.c_str(), inputSize, dimensionName, scaleName.c_str(), scaleSize, dimensionName);
            return false;
        }
        groupSize = inputSize / scaleSize;
    }
    return true;
}

bool QuantMatmulChecker::InferGroupSize(int64_t &groupSize)
{
    auto &x1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(inputTensors_);
    auto &x2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors_);
    auto &x1Scale = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors_);
    auto &x2Scale = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors_);
    // when x1Scale and x2Scale dim num is less than 2, groupsize not used
    if (x1Scale == nullptr || x1Scale->GetViewShape().GetDimNum() < 2 || x2Scale->GetViewShape().GetDimNum() < 2) {
        return true;
    }
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    auto x1ScaleDimNum = x1Scale->GetViewShape().GetDimNum();
    auto x2ScaleDimNum = x2Scale->GetViewShape().GetDimNum();
    auto transX1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(boolsTrans_);
    auto transX2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(boolsTrans_);
    int64_t groupSizeK = static_cast<int64_t>(static_cast<uint64_t>(groupSize) & GROUP_MNK_BIT_SIZE);
    int64_t groupSizeN = static_cast<int64_t>((static_cast<uint64_t>(groupSize) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE);
    int64_t groupSizeM = static_cast<int64_t>((static_cast<uint64_t>(groupSize) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE);
    auto inputSizeM = transX1 ? x1->GetViewShape().GetDim(x1DimNum - 1) : x1->GetViewShape().GetDim(x1DimNum - PENULTIMATE_DIM);
    auto scaleSizeM = 0;
    if (IsMicroScaling(x1Scale, x2Scale)) {
        scaleSizeM = x1Scale->GetViewShape().GetDim(transX1 ? 1 : 0);
    } else {
        scaleSizeM = transX1 ? x1Scale->GetViewShape().GetDim(x1ScaleDimNum - 1)
                             : x1Scale->GetViewShape().GetDim(x1ScaleDimNum - PENULTIMATE_DIM);
    }
    CHECK_RET(ReCalcGroupSize(inputSizeM, scaleSizeM, groupSizeM, "m"), false);
    auto inputSizeK = transX1 ? x1->GetViewShape().GetDim(x1DimNum - PENULTIMATE_DIM) : x1->GetViewShape().GetDim(x1DimNum - 1);
    auto scaleSizeK = 0;
    if (IsMicroScaling(x1Scale, x2Scale)) {
        scaleSizeK = x1Scale->GetViewShape().GetDim(transX1 ? 0 : 1) * 2; //when scale type is e8m0, scalex1 shape is [m, k/2, 2] or [k/2, m, 2]
    } else {
        scaleSizeK = transX1 ? x1Scale->GetViewShape().GetDim(x1ScaleDimNum - PENULTIMATE_DIM)
                             : x1Scale->GetViewShape().GetDim(x1ScaleDimNum - 1);
    }
    CHECK_RET(ReCalcGroupSize(inputSizeK, scaleSizeK, groupSizeK, "k"), false);
    auto inputSizeN = transX2 ? x2->GetViewShape().GetDim(x2DimNum - PENULTIMATE_DIM) : x2->GetViewShape().GetDim(x2DimNum - 1);
    auto scaleSizeN = 0;
    if (IsMicroScaling(x1Scale, x2Scale)) {
        scaleSizeN = x2Scale->GetViewShape().GetDim(transX2 ? 0 : 1);
    } else {
        scaleSizeN = transX2 ? x2Scale->GetViewShape().GetDim(x2ScaleDimNum - PENULTIMATE_DIM)
                             : x2Scale->GetViewShape().GetDim(x2ScaleDimNum - 1);
    }
    CHECK_RET(ReCalcGroupSize(inputSizeN, scaleSizeN, groupSizeN, "n"), false);
    OP_LOGD("Infered groupSize: groupSizeM: %lu, groupSizeN: %lu, groupSizeK: %lu.", groupSizeM, groupSizeN, groupSizeK);
    groupSize = static_cast<int64_t>((groupSizeM << GROUP_M_OFFSET) | (groupSizeN << GROUP_N_OFFSET) | groupSizeK);
    return true;
}

bool QuantMatmulChecker::CheckDimValuePerblock() const
{
    auto x1DimNum = x1_->GetViewShape().GetDimNum();
    auto x2DimNum = x2_->GetViewShape().GetDimNum();
    auto x1ScaleDimNum = x1Scale_->GetViewShape().GetDimNum();
    auto x2ScaleDimNum = x2Scale_->GetViewShape().GetDimNum();
    uint64_t groupSizeM = std::max((static_cast<uint64_t>(groupSize_) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE, 1UL);
    if (x1ScaleDimNum != x1DimNum || x2ScaleDimNum != x2DimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The dimension of %s should be same as the dimension of x1, the dimension of %s should be same as the \
dimension of x2. Actual dimensions of x1 and x2 are (%zu, %zu), dimensions of %s and %s are (%zu, %zu).",
                GetX1ScaleName().c_str(), GetX2ScaleName().c_str(), x1DimNum, x2DimNum, GetX1ScaleName().c_str(),
                GetX2ScaleName().c_str(), x1ScaleDimNum, x2ScaleDimNum);
        return false;
    }
    for (int64_t index = x1DimNum - BATCH_TAILENDER_DIM; index >= 0; --index) {
        if (x1Scale_->GetViewShape().GetDim(index) != x1_->GetViewShape().GetDim(index)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Batch value of %s should be same as batch value of x1, actual batch of x1 and %s is (%ld, %ld).",
                    GetX1ScaleName().c_str(), GetX1ScaleName().c_str(), x1_->GetViewShape().GetDim(index),
                    x1Scale_->GetViewShape().GetDim(index));
            return false;
        }
    }
    for (int64_t index = x2DimNum - BATCH_TAILENDER_DIM; index >= 0; --index) {
        if (x2Scale_->GetViewShape().GetDim(index) != x2_->GetViewShape().GetDim(index)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Batch value of %s should be same as batch value of x2, actual batch of x2 and %s is (%ld, %ld).",
                    GetX2ScaleName().c_str(), GetX2ScaleName().c_str(), x2_->GetViewShape().GetDim(index),
                    x2Scale_->GetViewShape().GetDim(index));
            return false;
        }
    }
    return CheckGroupSizeShape(groupSizeM);
}

bool QuantMatmulChecker::CheckDimValuePertokenDoubleScale() const
{
    auto x1ScaleDim0Size = x1Scale_->GetViewShape().GetDim(0);
    if (x1MDim_ == 1) {
        if (x1ScaleDim0Size != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When %s, %s both have only one dimension, the shape of %s should be (1) or (m), m represents the \
size of m dimension of x1, which is 1. Actual shape of %s: (%ld).",
                    GetX1ScaleName().c_str(), GetX2ScaleName().c_str(), GetX1ScaleName().c_str(),
                    GetX1ScaleName().c_str(), x1ScaleDim0Size);
            return false;
        }
    } else {
        if (x1ScaleDim0Size == 1 && !IsInt8Input(x1_, x2_)) { // double scale
            if (x2Scale_->GetViewShape().GetDim(0) != 1 && x2Scale_->GetViewShape().GetDim(0) != x2NDim_) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "When the shape of %s is (1), %s should also be (1) or (n), where n is the size of x2's n \
dimension, but actual shape is (%ld).",
                        GetX1ScaleName().c_str(), GetX2ScaleName().c_str(), x2Scale_->GetViewShape().GetDim(0));
                return false;
            }
        } else {
            if (x1ScaleDim0Size!= x1MDim_) { // pertoken
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "When %s, %s both have only one dimension, the shape of %s should be (m), m represents \
the size of m dimension of x1, which is %ld. Actual shape of %s: (%ld).",
                        GetX1ScaleName().c_str(), GetX2ScaleName().c_str(), GetX1ScaleName().c_str(), x1MDim_,
                        GetX1ScaleName().c_str(), x1ScaleDim0Size);
                return false;
            }
        }
    }
    return true;
}

bool QuantMatmulChecker::CheckDimValue() const
{
    if (x1Scale_ != nullptr) {
        if (IsMicroScaling(x1Scale_, x2Scale_)) {
            CHECK_RET(CheckDimValueMicroScaling(), false);
            return true;
        } else if (IsPerblock(x1_, x2_, x1Scale_, x2Scale_)) {
            CHECK_RET(CheckDimValuePerblock(), false);
            return true;
        } else {
            CHECK_RET(CheckDimValuePertokenDoubleScale(), false);
        }
    }
    if (ge::GetPrimaryFormat(x2_->GetStorageFormat()) == Format::FORMAT_FRACTAL_NZ && (x2KDim_ == 1 || x2NDim_ == 1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
                "It is not supported to convert x2 to private format(FRACTAL_NZ) when the last two dimensions of x2 have 1");
        return false;
    }

    const uint64_t groupSizeK = static_cast<uint64_t>(groupSize_ & GROUP_MNK_BIT_SIZE);
    bool isA4W4PergroupNonSymmetric = IsA4W4PergroupNonSymmetric(groupSizeK);
    if (!isA4W4PergroupNonSymmetric && x2Scale_->GetViewShape().GetDim(0) != x2NDim_ &&
        x2Scale_->GetViewShape().GetDim(0) != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When quantification of x2 is pertensor or perchannel quant, \
the size of %s's last dimension should equal to the size of x2's n dimension %ld or 1, but actual is %ld.",
                GetX2ScaleName().c_str(), x2NDim_, x2Scale_->GetViewShape().GetDim(0));
        return false;
    }

    if (x2Offset_ != nullptr) {
        if (!isA4W4PergroupNonSymmetric) {
            CHECK_RET(OpCheckWrongDimension(interfaceType_, X2OFFSET_NAME, x2Offset_, 1UL), false);
            if (x2Offset_->GetViewShape().GetDim(0) != x2NDim_ && x2Offset_->GetViewShape().GetDim(0) != 1) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "The size of %s's 1st dimension should equal to the size x2's n dimension %ld or 1, \
    but actual is %ld.",
                    GetX2OffsetName().c_str(), x2NDim_, x2Offset_->GetViewShape().GetDim(0));
                return false;
            }
        }
    }
    return true;
}

int64_t QuantMatmulChecker::InferOutputShape(std::vector<int64_t> &batchRecord) const
{
    int64_t inferedOutbatchValue = 1;
    auto x1DimNum = x1_->GetViewShape().GetDimNum();
    auto x2DimNum = x2_->GetViewShape().GetDimNum();
    auto outDimNum = std::max(x1DimNum, x2DimNum);
    size_t vaildOffset = outDimNum - std::min(x1DimNum, x2DimNum);
    auto &longShapeTensor = x1DimNum > x2DimNum ? x1_ : x2_;
    auto &shortShapeTensor = x1DimNum > x2DimNum ? x2_ : x1_;
    for (size_t i = 0; i < outDimNum - PENULTIMATE_DIM; i++) {
        auto longDimVal = longShapeTensor->GetViewShape().GetDim(i);
        auto shortDimVal = i < vaildOffset ? 1 : shortShapeTensor->GetViewShape().GetDim(i - vaildOffset);
        if (shortDimVal > 1 && longDimVal > 1 && shortDimVal != longDimVal) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Current short dim value %ld and long dim value %ld are not supported for broadcasting.",
                    shortDimVal, longDimVal);
            return OUTPUT_INFER_FAIL;
        }
        int64_t curBatchValue = static_cast<int64_t>(std::max(shortDimVal, longDimVal));
        inferedOutbatchValue = inferedOutbatchValue * curBatchValue;
        batchRecord.push_back(curBatchValue);
    }
    return inferedOutbatchValue;
}

bool QuantMatmulChecker::CheckBiasShape(const std::vector<int64_t> &batchRecord, int64_t inferedOutbatchValue) const
{
    auto biasDimNum = bias_->GetViewShape().GetDimNum();
    // 3 is bias with batch dim-num
    if (biasDimNum != 3 && biasDimNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor bias's dimension should equal 3 or 1, but it is %zu.", biasDimNum);
        return false;
    }
    auto biasFirstDim = bias_->GetViewShape().GetDim(0);
    if (biasDimNum == 1) {
        OP_CHECK(biasFirstDim == x2NDim_,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of bias's 1st dimension should be equal to the size of \
    x2's n dimension %ld, but it is %ld.",
                         x2NDim_, biasFirstDim),
                 return false);
        return true;
    }
    auto biasSecondDim = bias_->GetViewShape().GetDim(1);
    // 2 is bias last dim index
    auto biasThirdDim = bias_->GetViewShape().GetDim(2);
    // output batch need to be only 1 dim when bias dim is 3
    if (batchRecord.size() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When bias's dimension is 3, infered out batch dimension should be 1, but infered out batch \
dimension is %zu.",
                batchRecord.size());
        return false;
    }
    OP_CHECK(biasSecondDim == 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of bias's 2nd dimension should be equal to 1, \
but it is %ld.",
                     biasSecondDim),
             return false);
    OP_CHECK(biasThirdDim == x2NDim_,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of bias's last dimension should be equal to the size of x2's \
n dimension %ld, but actually is %ld.",
                     x2NDim_, biasThirdDim), return false);
    OP_CHECK(biasFirstDim == inferedOutbatchValue,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The size of bias's 1st dimension should be equal to the size of out batch dimension, \
but it is %ld and infered out batch dimension's size is %ld.",
                biasFirstDim, inferedOutbatchValue), return false);
    return true;
}

bool QuantMatmulChecker::CheckOutShape(bool twoDimMatmulCaseFlag, const std::vector<int64_t> &batchRecord) const
{
    auto outDimNum = out_->GetViewShape().GetDimNum();
    int64_t outMDim = out_->GetViewShape().GetDim(outDimNum - PENULTIMATE_DIM);
    int64_t outNDim = out_->GetViewShape().GetDim(outDimNum - 1);
    size_t inferedOutDimNum = batchRecord.size() + 2;
    // x1 and x2 are 2 dim and out is 3 dim speical case
    if (outMDim == 1 && inferedOutDimNum == 2 && outDimNum == 3 && twoDimMatmulCaseFlag) {
        outDimNum -= 1;
        outMDim = out_->GetViewShape().GetDim(0);
    }
    if (inferedOutDimNum != outDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Infered output dimension should be equal to actual out dimension. Infered: %zu, actual: %zu.",
                inferedOutDimNum, outDimNum);
        return false;
    }
    OP_CHECK(outMDim == x1MDim_,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Out's 1st dimension size should be equal to x1's m dimension size, \
but out 1st dimension size is %ld, the size of x1's m dimension is %ld.",
                     outMDim, x1MDim_), return false);
    OP_CHECK(outNDim == x2NDim_,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Out 2nd dimension should be equal to x2 n dimension, but out 2nd dimension size  is %ld, \
the size of x2's n dimension is %ld.",
                     outNDim, x2NDim_), return false);
    for (size_t i = 0; i < outDimNum - PENULTIMATE_DIM; i++) {
        OP_CHECK(out_->GetViewShape().GetDim(i) == batchRecord[i],
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "Output dimension should be equal to inferred output dimension. Output dimension is %ld, \
infered output dimension is %ld, shape index is %zu.",
                         out_->GetViewShape().GetDim(i), batchRecord[i], i),
                 return false);
    }
    return true;
}

bool QuantMatmulChecker::CheckShapeInt4() const
{
    if (!IsAligned<int64_t>(x1KDim_, INT4_NUMS_IN_INT8)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The size of x1's k dimension should be a positive even number \
in a4w4 senario, but now it is %ld.", x1KDim_);
        return false;
    }
    if (transposeX2_ && !IsAligned<int64_t>(x2KDim_, INT4_NUMS_IN_INT8)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The size of x2's k dimension should be a positive even number \
when transposeX2 is true in a4w4 senario, but now it is %ld.",
                x2KDim_);
        return false;
    }
    if (!transposeX2_ && !IsAligned<int64_t>(x2NDim_, INT4_NUMS_IN_INT8)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The size of x2's n dimension should be a positive even number \
when transposeX2 is false in a4w4 senario, but now it is %ld.",
                x2NDim_);
        return false;
    }
    if (transposeX1_) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "TransposeX1 should be false in a4w4 senario, but now is true.");
        return false;
    }
    if (x2_->GetViewShape().GetDimNum() != X2_FIXED_DIM_NUM_A4W4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x2's dimension should be 2 in a4w4, but is %zu.",
                x2_->GetViewShape().GetDimNum());
        return false;
    }
    if (bias_ != nullptr && bias_->GetViewShape().GetDimNum() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input bias's dimension should be 1 in a4w4, but is %zu.",
                bias_->GetViewShape().GetDimNum());
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckScaleDimRange() const
{
    if (x1Scale_ != nullptr) {
        auto x1ScaleDim = x1Scale_->GetViewShape().GetDimNum();
        auto x2ScaleDim = x2Scale_->GetViewShape().GetDimNum();
        const uint64_t groupSizeK = static_cast<uint64_t>(groupSize_ & GROUP_MNK_BIT_SIZE);
        bool isA4W4PergroupNonSymmetric = IsA4W4PergroupNonSymmetric(groupSizeK);
        if (IsMicroScaling(x1Scale_, x2Scale_)) {
            if (x1ScaleDim != MX_SCALE_DIM) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In mx quantification, \
the dimension of %s should be %zu. Actual dimension of %s: %zu.",
                        GetX1ScaleName().c_str(), MX_SCALE_DIM, GetX1ScaleName().c_str(), x1ScaleDim);
                return false;
            }
            if (x2ScaleDim != MX_SCALE_DIM) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In mx quantification, \
the dimension of %s should be %zu. Actual dimension of %s: %zu.",
                        GetX2ScaleName().c_str(), MX_SCALE_DIM, GetX2ScaleName().c_str(), x2ScaleDim);
                return false;
            }
        } else if (isA4W4PergroupNonSymmetric) {
            return true;
        } else {
            bool isPerblock = IsPerblock(x1_, x2_, x1Scale_, x2Scale_);
            if ((isPerblock && x1ScaleDim != x1_->GetViewShape().GetDimNum()) || (!isPerblock && x1ScaleDim != 1)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When quantification mode is G-B or B-B quantification, \
the dimension of %s and x1 should be same. For other quantification mode, the dimension of %s should be 1, actual \
dimension of %s is %zu, x1 is %zu", GetX1ScaleName().c_str(), GetX1ScaleName().c_str(), GetX1ScaleName().c_str(),
                        x1ScaleDim, x1_->GetViewShape().GetDimNum());
                return false;
            }
            if ((isPerblock && x2ScaleDim != x2_->GetViewShape().GetDimNum()) || (!isPerblock && x2ScaleDim != 1)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When quantification mode is G-B or B-B quantification, \
the dimension of %s and x1 should be same. For other quantification mode, the dimension of %s should be 1, actual \
that of %s is %zu, x1 is %zu",  GetX2ScaleName().c_str(),  GetX2ScaleName().c_str(),  GetX2ScaleName().c_str(),
                        x2ScaleDim, x2_->GetViewShape().GetDimNum());
                return false;
            }
        }
    } else { // pertensor / perchannel
        OP_CHECK_WRONG_DIMENSION(x2Scale_, 1, return false);
    }
    OP_LOGD("QuantMatmul check dimension range success.");
    return true;
}

bool QuantMatmulChecker::CheckGroupSize() const
{
    if (npuArch_ != NpuArch::DAV_3510) {
        return true;
    }
    uint64_t groupSizeM = (static_cast<uint64_t>(groupSize_) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(groupSize_) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeK = static_cast<uint64_t>(groupSize_) & GROUP_MNK_BIT_SIZE;
    if (IsMicroScaling(x1Scale_, x2Scale_)) {
        if (groupSizeK != static_cast<uint64_t>(PERGROUP_GROUP_SIZE) || groupSizeM != 1ULL || groupSizeN != 1ULL) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Unsupported groupSize. In mx quantification, input or infered groupSize should be \
4295032864(for torch api, group_sizes should be [1, 1, 32]). Actual groupSize: %lu(for torch api \
group_sizes is [%lu, %lu, %lu]).",
                    groupSize_, groupSizeM, groupSizeN, groupSizeK);
            return false;
        }
    } else if (IsPerblock(x1_, x2_, x1Scale_, x2Scale_)) {
        if (groupSizeK != PERBLOCK_BLOCK_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Unsupported groupSize. When quantification mode is G-B or B-B quantification, input or infered \
groupSizeK(for torch api, group_size[2]) should be 128. Actual groupSizeK: %lu, groupSizeK = groupSize & 0xFFFF.",
                    groupSizeK);
            return false;
        }
        if (groupSizeN != PERBLOCK_BLOCK_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Unsupported groupSize. When quantification mode is G-B or B-B quantification, input or infered \
groupSizeN(for torch api, group_size[1]) should be 128. Actual groupSizeN: %lu, groupSizeN = (groupSize >> 16) & 0xFFFF.",
                    groupSizeN);
            return false;
        }
        if (groupSizeM != PERBLOCK_BLOCK_SIZE && groupSizeM != 1UL) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Unsupported groupSize. When quantification mode is G-B or B-B quantification, input or infered \
groupSizeM(for torch api, group_size[0]) \
should be 128 or 1. Actual groupSizeM: %lu, groupSizeM = (groupSize >> 32) & 0xFFFF.", groupSizeM);
            return false;
        }
    } else if (groupSize_ != 0UL) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Unsupported groupSize. When quantification mode is not G-B or B-B or mx quantification, \
groupSize should be 0(torch api group_sizes should be [0, 0, 0] or None). \
Actual groupSize: %lu(torch api group_sizes is [%lu, %lu, %lu]).",
                groupSize_, groupSizeM, groupSizeN, groupSizeK);
        return false;
    }
    OP_LOGD("QuantMatmul check group_size success.");
    return true;
}

bool QuantMatmulChecker::CheckShape() const
{
    auto x1DimNum = x1_->GetViewShape().GetDimNum();
    auto x2DimNum = x2_->GetViewShape().GetDimNum();

    CHECK_RET(CheckScaleDimRange(), false);
    CHECK_RET(CheckGroupSize(), false);
    if (isA4W4_ && !CheckShapeInt4()) {
        return false;
    }
    OP_CHECK(x1KDim_ == x2KDim_,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input the size of x1's k dimension and size of x2's k dimension \
should be same, but x1's is %ld, x2's is %ld.",
                     x1KDim_, x2KDim_), return false);

    if (ge::GetPrimaryFormat(x2_->GetStorageFormat()) == Format::FORMAT_FRACTAL_NZ) {
        CHECK_RET(CheckShapeForWeightNz(), false);
    }
    CHECK_RET(CheckDimValue(), false);

    std::vector<int64_t> batchRecord;
    int64_t inferedOutbatchValue = InferOutputShape(batchRecord);
    if (inferedOutbatchValue == OUTPUT_INFER_FAIL) {
        return false;
    }
    if (bias_ != nullptr) {
        if (!CheckBiasShape(batchRecord, inferedOutbatchValue)) {
            return false;
        }
    }
    bool twoDimMatmulCaseFlag = x1DimNum == x2DimNum && x2DimNum == 2;
    CHECK_RET(CheckOutShape(twoDimMatmulCaseFlag, batchRecord), false);
    return true;
}

bool QuantMatmulChecker::CheckFormatInt4() const
{
    if (x1_->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x1 only support ND format in a4w4 scenario, but now is %s.",
                op::ToString(x1_->GetStorageFormat()).GetString());
        return false;
    }
    if (isWeightNz_ && x2_->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x2 only support NZ in a4w4 weightNz scenario, but now is %s.",
                op::ToString(x2_->GetStorageFormat()).GetString());
        return false;
    } else if (!isWeightNz_ && x2_->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x2 only support ND in a4w4 scenario, but now is %s.",
                op::ToString(x2_->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckDtype4WeightNz() const
{
    if (!(x1_->GetDataType() == op::DataType::DT_INT8 || (x1_->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN &&
                                                          x2_->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When format of x2 is FRACTAL_NZ, input dtype must be INT8/FLOAT8_E4M3FN, actual x1 is %s, x2 is %s.",
                op::ToString(x1_->GetDataType()).GetString(), op::ToString(x2_->GetDataType()).GetString());
        return false;
    }

    if ((x1_->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN &&
         x2_->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN) &&
        !(x1Scale_->GetDataType() == op::DataType::DT_FLOAT8_E8M0 &&
          x2Scale_->GetDataType() == op::DataType::DT_FLOAT8_E8M0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When format of x2 is FRACTAL_NZ and input dtype is FLOAT8_E4M3FN, x1Scale and x2Scale are \
FLOAT8_E8M0, actual x1Scale is %s, x2Scale is %s.",
                op::ToString(x1Scale_->GetDataType()).GetString(), op::ToString(x2Scale_->GetDataType()).GetString());
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckFormat() const
{
    auto x2StorageFormat = ge::GetPrimaryFormat(x2_->GetStorageFormat());
    if (x1Scale_ != nullptr && npuArch_ == NpuArch::DAV_2002) {
        OP_LOGD("QuantMatmul pertoken mode in Arch 2002: x1 should be FORMAT_FRACTAL_NZ");
    } else {
        CHECK_RET(x1_->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    if (npuArch_ == NpuArch::DAV_2002) {
        CHECK_RET(x2_->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ, false);
    } else {
        CHECK_RET(x2StorageFormat == op::Format::FORMAT_ND || x2StorageFormat == op::Format::FORMAT_FRACTAL_NZ, false);
    }
    if (x1Scale_ != nullptr) {
        CHECK_RET(x1Scale_->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    CHECK_RET(x2Scale_->GetStorageFormat() == op::Format::FORMAT_ND, false);
    if (x2Offset_ != nullptr) {
        CHECK_RET(x2Offset_->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    if (bias_ != nullptr) {
        CHECK_RET(bias_->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    if (isA4W4_) {
        CHECK_RET(CheckFormatInt4(), false);
    }
    return true;
}

bool QuantMatmulChecker::CheckL0c2outPertensorPerchannel() const
{
    if (!(x2Scale_->GetDataType() == op::DataType::DT_FLOAT || x2Scale_->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When out dtype is INT32, %s dtype should be FLOAT or BF16, actual dtype is %s.",
                GetX2ScaleName().c_str(), op::ToString(x2Scale_->GetDataType()).GetString());
        return false;
    }
    if (bias_ != nullptr && bias_->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When out dtype is INT32, bias dtype should be INT32, actual dtype is %s.",
                op::ToString(bias_->GetDataType()).GetString());
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckL0c2outAndL0c2ubPertensorPerchannel() const
{
    CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X1_NAME, x1_, op::DataType::DT_INT8), false);
    CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2_NAME, x2_, op::DataType::DT_INT8), false);
    if (outType_ == op::DataType::DT_INT32) {
        CHECK_RET(CheckL0c2outPertensorPerchannel(), false);
    } else if (outType_ == op::DataType::DT_BF16) {
        CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME, x2Scale_,BF16_OUT_X2SCALE_SUPPORT_LIST), false);
        if (groupSize_ != 0UL) {
            CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2SCALE_NAME, x2Scale_, op::DataType::DT_BF16), false);
        }
        if (bias_ != nullptr) {
            CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, BIAS_NAME, bias_, BIAS_TYPE_SUPPORT_LIST), false);
        }
        if (x2Offset_ != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When out dtype is BFLOAT16, %s must be null, but it is not null.", GetX2OffsetName().c_str());
            return false;
        }
    } else if (outType_ == op::DataType::DT_INT8) {
        CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME,
                                         x2Scale_, UINT64_X2_SCALE_TYPE_SUPPORT_LIST), false);
        if (bias_ != nullptr) {
            CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, BIAS_NAME, bias_, op::DataType::DT_INT32), false);
        }
        if (x2Offset_ != nullptr) {
            CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2OFFSET_NAME, x2Offset_, op::DataType::DT_FLOAT), false);
        }
    } else if (outType_ == op::DataType::DT_FLOAT16) {
        CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME,
                                         x2Scale_, UINT64_X2_SCALE_TYPE_SUPPORT_LIST), false);
        if (bias_ != nullptr) {
            CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, BIAS_NAME, bias_, op::DataType::DT_INT32), false);
        }
        if (x2Offset_ != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When out dtype is FLOAT16, %s must be null, but it is not null.", GetX2OffsetName().c_str());
            return false;
        }
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid out dtype: %s.",
                RemoveDtInDtype(op::ToString(outType_).GetString()).c_str());
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckOnlyL0c2outPertoken() const
{
    CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, X2_NAME, x2_, X1_X2_L0C2OUT_PERTOKEN_SUPPORT_LIST), false);
    CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X1SCALE_NAME, x1Scale_, op::DataType::DT_FLOAT), false);
    if (outType_ == op::DataType::DT_BF16) {
        CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME, x2Scale_,
                                         PERTOKEN_BF16_OUT_X2SCALE_SUPPORT_LIST), false);
        if (bias_ != nullptr) {
            CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, BIAS_NAME,
                                             bias_, PERTOKEN_BF16_OUT_BIAS_SUPPORT_LIST), false);
        }
    } else if (outType_ == op::DataType::DT_FLOAT16) {
        CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2SCALE_NAME, x2Scale_, op::DataType::DT_FLOAT), false);
        if (bias_ != nullptr) {
            CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, BIAS_NAME,
                                             bias_, PERTOKEN_FP16_OUT_BIAS_SUPPORT_LIST), false);
        }
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When x1/x2 are INT8/INT4 and %s, %s are FLOAT, output dtype should be BF16/FLOAT16, but actual \
dtype is %s.",
                GetX1ScaleName().c_str(), GetX2ScaleName().c_str(),
                RemoveDtInDtype(op::ToString(outType_).GetString()).c_str());
        return false;
    }
    if (x2Offset_ != nullptr) {
        if (x1_->GetDataType() == op::DataType::DT_INT8 && x2_->GetDataType() == op::DataType::DT_INT8 &&
            x1Scale_->GetDataType() != op::DataType::DT_FLOAT && x2Scale_->GetDataType() != op::DataType::DT_FLOAT) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Input %s is not supported when x1, x2 are INT8 and %s, %s are FLOAT.",
                GetX2OffsetName().c_str(), GetX1ScaleName().c_str(), GetX2ScaleName().c_str());
            return false;
        }
    }
    return true;
}

bool QuantMatmulChecker::CheckDtypeValidOnOnlyL0c2outForA4W4() const
{
    if (x1_->GetDataType() != op::DataType::DT_INT4 || x2_->GetDataType() != op::DataType::DT_INT4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x1 x2 dtype should be INT4 in a4w4 scenario, actual dtype is %s %s.",
                op::ToString(x1_->GetDataType()).GetString(), op::ToString(x2_->GetDataType()).GetString());
        return false;
    }
    if (x2Scale_->GetDataType() != op::DataType::DT_UINT64 && x2Scale_->GetDataType() != op::DataType::DT_INT64) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Input %s dtype should be UINT64 or INT64 in a4w4 without %s scenario, actual dtype is %s.",
            GetX2ScaleName().c_str(), GetX1ScaleName().c_str(), op::ToString(x2Scale_->GetDataType()).GetString());
        return false;
    }
    if (bias_ != nullptr && bias_->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Bias dtype should be INT32 in a4w4 without %s scenario, actual dtype is %s.",
            GetX1ScaleName().c_str(), op::ToString(bias_->GetDataType()).GetString());
        return false;
    }
    if (out_->GetDataType() != op::DataType::DT_FLOAT16 && out_->GetDataType() != op::DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output dtype should be FLOAT16 or BF16 in a4w4 scenario, actual dtype is %s.",
                op::ToString(out_->GetDataType()).GetString());
        return false;
    }
    return true;
}

aclnnStatus QuantMatmulChecker::CheckDtypeOnlyL0c2ub() const
{
    if (isA4W4_) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                "QuantBatchMatmul support for %s is not implemented in a4w4 senario.",
                op::ToString(socVersion_).GetString());
    }
    if (x1Scale_ != nullptr) {
 	    CHECK_RET(CheckOnlyL0c2ubPertoken(), ACLNN_ERR_PARAM_INVALID);
 	} else {
 	    CHECK_RET(CheckL0c2outAndL0c2ubPertensorPerchannel(), ACLNN_ERR_PARAM_INVALID);
 	}
    return ACLNN_SUCCESS;
}

aclnnStatus QuantMatmulChecker::CheckDtypeOnlyL0c2out() const
{
    if (isA4W4_ && x1Scale_ == nullptr) { // pertensor/perchannel
        CHECK_RET(CheckDtypeValidOnOnlyL0c2outForA4W4(), ACLNN_ERR_PARAM_INVALID);
    } else if (!isA4W4_ && x1Scale_ == nullptr) { // pertensor/perchannel
        CHECK_RET(CheckL0c2outAndL0c2ubPertensorPerchannel(), ACLNN_ERR_PARAM_INVALID);
    } else if (x1_->GetDataType() == op::DataType::DT_INT8 || (isA4W4_ && x1Scale_ != nullptr)) { // pertoken
        CHECK_RET(CheckOnlyL0c2outPertoken(), ACLNN_ERR_PARAM_INVALID);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unexpected quantification.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

bool QuantMatmulChecker::CheckL0c2outOrL0c2ubPertensorPerchannel4Int8Input() const
{
    if (outType_ == op::DataType::DT_BF16) {
        CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME, x2Scale_,
                                         X2_INT8_X2_SCALE_TYPE_SUPPORT_LIST), false);
    } else if (outType_ == op::DataType::DT_INT8 || outType_ == op::DataType::DT_FLOAT16) {
        CHECK_RET(
            OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME, x2Scale_, UINT64_X2_SCALE_TYPE_SUPPORT_LIST), false);
    } else if (outType_ == op::DataType::DT_INT32) {
        CHECK_RET(
            OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME, x2Scale_, INT32_OUT_X2SCALE_SUPPORT_LIST), false);
    }
    if (bias_ != nullptr) {
        if (outType_ == op::DataType::DT_INT32 && bias_->GetDataType() != op::DataType::DT_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,"The dtype of bias_ should be nullptr or INT32 when out is INT32.");
            return false;
        }
        if (outType_ == op::DataType::DT_BF16 &&
            (x2ScaleType_ == op::DataType::DT_FLOAT || x2ScaleType_ == op::DataType::DT_BF16)) {
            CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, BIAS_NAME, bias_, BIAS_TYPE_SUPPORT_LIST), false);
        }
        if ((outType_ == op::DataType::DT_INT8 || outType_ == op::DataType::DT_FLOAT16 ||
             outType_ == op::DataType::DT_BF16) &&
            (x2ScaleType_ == op::DataType::DT_UINT64 || x2ScaleType_ == op::DataType::DT_INT64)) {
            CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, BIAS_NAME, bias_, op::DataType::DT_INT32), false);
        }
    }
    if (x2Offset_ != nullptr) {
        if (outType_ != op::DataType::DT_INT8) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Unsupported %s. When x1, x2 are INT8 and out is not INT8, %s should be nullptr.",
                    GetX2OffsetName().c_str(), GetX2OffsetName().c_str());
            return false;
        }
        CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2OFFSET_NAME, x2Offset_, op::DataType::DT_FLOAT), false);
    }
    CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, OUT_NAME, out_, INT8_OUT_TYPE_SUPPORT_LIST), false);
    return true;
}

bool QuantMatmulChecker::CheckL0c2outOrL0c2ubPertensorPerchannel() const
{
    if (outType_ == op::DataType::DT_INT32 && (x1_->GetDataType() != op::DataType::DT_INT8 || x2_->GetDataType() != op::DataType::DT_INT8)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When out is INT32, x1 and x2 should be INT8.");
        return false;
    }
    if (IsInt8Input(x1_, x2_)) {
        CHECK_RET(CheckL0c2outOrL0c2ubPertensorPerchannel4Int8Input(), false);
    } else if (IsHif8Input(x1_, x2_) || IsFp8Input(x1_, x2_)) {
        CHECK_RET(
            OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME, x2Scale_, UINT64_X2_SCALE_TYPE_SUPPORT_LIST), false);
        if (bias_ != nullptr) {
            CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, BIAS_NAME, bias_, op::DataType::DT_FLOAT), false);
        }
        if (x2Offset_ != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s is not supported when x1, x2 are FLOAT8/HIFLOAT8.",
                    GetX2OffsetName().c_str());
            return false;
        }
        if (IsHif8Input(x1_, x2_)) {
            CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, OUT_NAME, out_, HIF8_OUT_TYPE_SUPPORT_LIST), false);
        } else {
            CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, OUT_NAME, out_, FP8_OUT_TYPE_SUPPORT_LIST), false);
        }
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When %s does not exist, the dtypes of x1 and x2 should be consistent, and they should be one out \
of INT8/FLOAT8/HIFLOAT8. Actual x1 dtype: %s, x2 dtype: %s.",
                GetX1ScaleName().c_str(), RemoveDtInDtype(op::ToString(x1_->GetDataType()).GetString()).c_str(),
                RemoveDtInDtype(op::ToString(x2_->GetDataType()).GetString()).c_str());
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckMicroScaling() const
{
    if (!IsFp4Input(x1_, x2_) && !IsFp8Input(x1_, x2_)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Invalid x1 or x2 dtype. When %s and %s are FLOAT8_E8M0, the dtype of x1 and x2 must be FLOAT4 or \
FLOAT8. Actual x1 dtype: %s, x2 dtype: %s.",
                GetX1ScaleName().c_str(), GetX2ScaleName().c_str(),
                RemoveDtInDtype(op::ToString(x1_->GetDataType()).GetString()).c_str(),
                RemoveDtInDtype(op::ToString(x2_->GetDataType()).GetString()).c_str());
        return false;
    }
    if (bias_ != nullptr) {
        CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, BIAS_NAME, bias_, op::DataType::DT_FLOAT), false);
    }
    CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, OUT_NAME, out_, FP8_AND_HIF8_COMMON_OUT_TYPE_SUPPORT_LIST), false);
    if (x2Offset_ != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Unsupported %s. When %s and %s are FLOAT8_E8M0, %s should be nullptr.",
                GetX2OffsetName().c_str(), GetX1ScaleName().c_str(), GetX2ScaleName().c_str(),
                GetX2OffsetName().c_str());
        return false;
    }
    if (IsFp4Input(x1_, x2_) && (transposeX1_ || !transposeX2_)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Unsupported transpose. When %s and %s are FLOAT8_E8M0 and x1 and x2 are FLOAT4, transposeX1 should be \
false and transposeX2 should be true. Actual transposeX1: %d, transposeX2: %d.",
                GetX1ScaleName().c_str(), GetX2ScaleName().c_str(), transposeX1_, transposeX2_);
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckL0c2outOrL0c2ubPertoken() const
{
    CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X1_NAME, x1_, op::DataType::DT_INT8), false);
    CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2_NAME, x2_, op::DataType::DT_INT8), false);
    CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, OUT_NAME, out_, PERTOKEN_OUT_TYPE_SUPPORT_LIST), false);
    CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X1SCALE_NAME, x1Scale_, op::DataType::DT_FLOAT), false);

    if (x2Offset_ != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input %s is not supported when x1, x2 are INT8 with %s.",
                GetX2OffsetName().c_str(),
                GetX1ScaleName().c_str());
        return false;
    }

    if (outType_ == op::DataType::DT_BF16) {
        CHECK_RET(
            OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME, x2Scale_, PERTOKEN_BF16_OUT_X2SCALE_SUPPORT_LIST),
            false);
        if (bias_ != nullptr) {
            CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, BIAS_NAME, bias_, PERTOKEN_BF16_OUT_BIAS_SUPPORT_LIST),
                      false);
        }
    } else if (outType_ == op::DataType::DT_FLOAT16) {
        CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2SCALE_NAME, x2Scale_, op::DataType::DT_FLOAT), false);
        if (bias_ != nullptr) {
            CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, BIAS_NAME, bias_, PERTOKEN_FP16_OUT_BIAS_SUPPORT_LIST),
                      false);
        }
    }

    return true;
}

bool QuantMatmulChecker::CheckDoubleScaleAndFp8Hif8PertokenPerblock() const
{
    CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X1SCALE_NAME, x1Scale_, op::DataType::DT_FLOAT), false);
    CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2SCALE_NAME, x2Scale_, op::DataType::DT_FLOAT), false);
    CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, OUT_NAME, out_, FP8_AND_HIF8_COMMON_OUT_TYPE_SUPPORT_LIST), false);
    if (IsPerblock(x1_, x2_, x1Scale_, x2Scale_)) {
        if (bias_ != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input bias is unsupported in perblock scenarion.");
            return false;
        }
    }
    if (bias_ != nullptr) {
        if (x1Scale_->GetViewShape().GetDim(0) == 1L && x1MDim_ != 1L &&
            x2Scale_->GetViewShape().GetDim(0) == x2NDim_ && x2NDim_ != 1L) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input bias is unsupported.");
            return false;
        } else {
            CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, BIAS_NAME, bias_, op::DataType::DT_FLOAT), false);
        }
    }
    if (x2Offset_ != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Input %s is not supported when x1, x2 are FLOAT8/HIFLOAT8 and %s, %s are FLOAT.",
                GetX2OffsetName().c_str(), GetX1ScaleName().c_str(), GetX2ScaleName().c_str());
        return false;
    }
    return true;
}

aclnnStatus QuantMatmulChecker::CheckDtypeL0c2outOrL0c2ub() const
{
    if (ge::GetPrimaryFormat(x2_->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ && !CheckDtype4WeightNz()) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (isA4W4_) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                "QuantBatchMatmul support for %s is not implemented in a4w4 senario.",
                op::ToString(socVersion_).GetString());
    }
    if (outType_ == op::DataType::DT_INT32 && x1Scale_ != nullptr) { 
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,"The pertokenScaleOptional/x1Scale(pertokenScale) should be nullptr when out is INT32."); 
        return ACLNN_ERR_PARAM_INVALID; 
    }
    if (x1Scale_ == nullptr) { // pertensor/perchannel
        CHECK_RET(CheckL0c2outOrL0c2ubPertensorPerchannel(), ACLNN_ERR_PARAM_INVALID);
    } else if (IsMicroScaling(x1Scale_, x2Scale_)) { // micro scaling
        CHECK_RET(CheckMicroScaling(), ACLNN_ERR_PARAM_INVALID);
    } else if (IsInt8Input(x1_, x2_)) { // pertoken
        CHECK_RET(CheckL0c2outOrL0c2ubPertoken(), ACLNN_ERR_PARAM_INVALID);
    } else if (IsHif8Input(x1_, x2_) ||
               IsFp8Input(x1_, x2_)) {  // double scale, pertensor-perchannel, fp8/hif8 pertoken, perblock/pertile
        CHECK_RET(CheckDoubleScaleAndFp8Hif8PertokenPerblock(), ACLNN_ERR_PARAM_INVALID);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Unexpected quantification. x1 dtype is %s and x2 dtype is %s, %s %s nullptr.",
                RemoveDtInDtype(op::ToString(x1_->GetDataType()).GetString()).c_str(),
                RemoveDtInDtype(op::ToString(x2_->GetDataType()).GetString()).c_str(),
                GetX1ScaleName().c_str(), x1Scale_ == nullptr ? "is" : "is not");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

 bool QuantMatmulChecker::CheckOnlyL0c2ubPertoken() const
 	 {
 	     CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X1_NAME, x1_, op::DataType::DT_INT8), false);
 	     CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2_NAME, x2_, op::DataType::DT_INT8), false);
 	     CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X1SCALE_NAME, x1Scale_, op::DataType::DT_FLOAT), false);
 	     if (outType_ == op::DataType::DT_FLOAT16) {
 	         CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2SCALE_NAME,
 	                                          x2Scale_, op::DataType::DT_FLOAT), false);
 	         if (bias_ != nullptr) {
 	             CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, BIAS_NAME,
 	                                              bias_,  op::DataType::DT_INT32), false);
 	         }
 	         if (x2Offset_ != nullptr) {
 	             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
 	                     "When out dtype is FLOAT16, %s must be null, but it is not null.", GetX2OffsetName().c_str());
 	             return false;
 	         }
 	     } else {
 	         OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When pertokenScaleOptional is not nullptr, out dtype should be FLOAT16, actual dtype is: %s.",
 	                 RemoveDtInDtype(op::ToString(outType_).GetString()).c_str());
 	         return false;
 	     }
 	     return true;
 	 }

aclnnStatus QuantMatmulChecker::CheckDtype() const
{
    // 5 represents the aclnnQuantMatmulV5 interface
    if (!IsIntInput(x1_, x2_) && interfaceType_ != 5) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Interface aclnnQuantMatmulV3/V4 only support int8/int4/int32 input, \
but got %s and %s.",
                op::ToString(x1_->GetDataType()).GetString(), op::ToString(x2_->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    switch (npuArch_) {
        case NpuArch::DAV_2201:
            CHECK_RET(CheckDtypeOnlyL0c2out() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
            break;
        case NpuArch::DAV_3510:
            CHECK_RET(CheckDtypeL0c2outOrL0c2ub() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
            break;
        case NpuArch::DAV_2002:
            CHECK_RET(CheckDtypeOnlyL0c2ub() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
            break;
        default:
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "AclnnQuantMatmul do not support this platform: %s.",
                    op::ToString(socVersion_).GetString());
            return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus QuantMatmulChecker::CheckParams() const
{
    // 1. 分平台校验dtype
    aclnnStatus dtypeRet = CheckDtype();
    CHECK_RET(dtypeRet == ACLNN_SUCCESS, dtypeRet);

    // 2. 检查shape是否符合要求
    CHECK_RET(CheckShape(), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查format是否符合要求
    CHECK_RET(CheckFormat(), ACLNN_ERR_PARAM_INVALID);

    // 4. 空Tensor处理逻辑
    CHECK_RET(CheckEmptyTensor(), ACLNN_ERR_PARAM_INVALID);

    OP_LOGD("QuantMatmul check params success.");
    return ACLNN_SUCCESS;
}
