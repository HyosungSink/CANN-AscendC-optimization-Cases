/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>

#include "aclnn_grouped_matmul_finalize_routing_weight_nz.h"
#include "aclnn_grouped_matmul_finalize_routing_weight_nz_v2.h"
#include "aclnn_grouped_matmul_finalize_routing_v3.h"
#include "aclnn_grouped_matmul_finalize_routing_v2.h"
#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "grouped_matmul_finalize_routing.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "grouped_matmul_finalize_routing_MX_checker.h"
#include "../../../grouped_matmul/op_host/op_api/grouped_matmul_950_checker.h"

using namespace op;
using namespace GmmFinalizeRouting;
#ifdef __cplusplus
extern "C" {
#endif


namespace {

static constexpr int INDEX_X1_IN_MANDTORY_TUPLE = 0;
static constexpr int INDEX_X2_IN_MANDTORY_TUPLE = 1;
static constexpr int INDEX_SCALE_IN_OPTIONAL_TUPLE = 0;
static constexpr int INDEX_BIAS_IN_OPTIONAL_TUPLE = 1;
static constexpr int INDEX_PERTOKEN_IN_OPTIONAL_TUPLE = 2;
static constexpr int INDEX_GROUPLIST_IN_OPTIONAL_TUPLE = 3;
static constexpr int INDEX_SHAREINPUT_IN_OPTIONAL_TUPLE = 4;
static constexpr int INDEX_LOGIT_IN_OPTIONAL_TUPLE = 5;
static constexpr int INDEX_ROWINDEX_IN_OPTIONAL_TUPLE = 6;
static constexpr int INDEX_OFFSET_IN_OPTIONAL_TUPLE = 7;
static constexpr int INDEX_TUNINGCONFIG_IN_OPTIONAL_TUPLE = 8;
static constexpr int INDEX_GROUPLISTTYPE_IN_OPTIONAL_TUPLE = 2;
static constexpr int INDEX_OUT_IN_TUPLE = 2;
static constexpr int LAST_SECOND_DIM_INDEX = 2;
static constexpr int SCALE_DIM = 2;
static constexpr int W4A8_SCALE_DIM = 3;
static constexpr int MX_SCALE_DIM = 4;
static constexpr int MX_PERTOKEN_SCALE_DIM = 3;
static constexpr size_t MM_DIM = 2;

static const int ONE_DIM_NUM = 1;
static const int TWO_DIM_NUM = 2;
static const int THREE_DIM_NUM = 3;

static const int MIN_DIM_NUM_ND = 2;
static const int MAX_DIM_NUM_ND = 6;
static const int MIN_DIM_NUM_NZ = 4;
static const int MAX_DIM_NUM_NZ = 8;
static const int PENULTIMATE_DIM = 2;
static const int NZ_K1_INDEX = 3;
static const int NZ_K1_INDEX_TRANS = 4;
static const int NZ_STORAGE_PENULTIMATE_DIM = 16;
static const int NZ_STORAGE_LAST_DIM = 32;
static const int NZ_K0_VALUE_INT8 = 16;
static const int ND_K0_VALUE_INT8 = 64;
static const int NZ_K0_VALUE_INT8_TRANS = 32;
static const int NZ_K0_VALUE_INT4 = 16;
static const int NZ_K0_VALUE_INT4_TRANS = 64;
static const int ND_N_VALUE_ALIGN = 8;
static const int NZ_N_VALUE_ALIGN = 64;
static const int ND_K_VALUE_ALIGN = 64;

static const int64_t K_VALUE_2048 = 2048;
static const int64_t K_VALUE_128 = 128;

static const int64_t N_VALUE_7168 = 7168;
static const int64_t N_VALUE_256 = 256;
static const int64_t N_VALUE_64 = 64;

static const int64_t PER_INT4_IN_U32 = 8;
static const int64_t PER_INT4_IN_U8 = 2;

static const std::initializer_list<op::DataType> IN_TYPE_SUPPORT_LIST = {op::DataType::DT_INT8, op::DataType::DT_INT4};
static const std::initializer_list<op::DataType> OUT_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> SCALE_TYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT64, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> BIAS_TYPE_SUPPORT_LIST = { op::DataType::DT_BF16 };
static const std::initializer_list<op::DataType> OFFSET_TYPE_SUPPORT_LIST = { op::DataType::DT_FLOAT };
static const std::initializer_list<op::DataType> PERTOKEN_SCALE_TYPE_SUPPORT_LIST = { op::DataType::DT_FLOAT };
static const std::initializer_list<op::DataType> GROUP_LIST_TYPE_SUPPORT_LIST = { op::DataType::DT_INT64 };
static const std::initializer_list<op::DataType> SHARED_INPUT_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16 };
static const std::initializer_list<op::DataType> LOGIT_TYPE_SUPPORT_LIST = { op::DataType::DT_FLOAT };
static const std::initializer_list<op::DataType> ROW_INDEX_TYPE_SUPPORT_LIST = { op::DataType::DT_INT64,op::DataType::DT_INT32};

// w4a8 support dtype
static const std::initializer_list<op::DataType> W4A8_BIAS_TYPE_SUPPORT_LIST = { op::DataType::DT_FLOAT };
static const std::initializer_list<op::DataType> W4A8_IN1_TYPE_SUPPORT_LIST = { op::DataType::DT_INT8 };
static const std::initializer_list<op::DataType> W4A8_IN2_TYPE_SUPPORT_LIST = { op::DataType::DT_INT4 };
static const std::initializer_list<op::DataType> W4A8_OUT_TYPE_SUPPORT_LIST = { op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> W4A8_SCALE_TYPE_SUPPORT_LIST = { op::DataType::DT_INT64 };
static const std::initializer_list<op::DataType> W4A8_ROW_INDEX_TYPE_SUPPORT_LIST = { op::DataType::DT_INT64 };


// CheckW4orW8 Params
struct CheckW4orW8DimParams {
    int64_t x2EDim;
    int64_t x2NDim;
    int64_t x1MDim;
    const aclTensor* scale;
    const aclTensor* offset;
    const aclTensor* bias;
};

// CheckSupportScene Params
struct CheckSupportSceneParams {
    const aclTensor* x;
    const aclTensor* w;
    const aclTensor* scaleOptional;
    const aclTensor* pertokenScaleOptional;
    const aclTensor* groupListOptional;
    const aclTensor* sharedInputOptional;
    const aclTensor* logitOptional;
    const aclTensor* rowIndexOptional;
    int64_t dtype;
};


static inline bool CheckNotNull(const GroupedMatmulParams &params)
{
    OP_CHECK_NULL(params.x1, return false);
    OP_CHECK_NULL(params.x2, return false);
    OP_CHECK_NULL(params.out, return false);
    return true;
}

static inline bool CheckDtypeValid(const GroupedMatmulParams &params)
{
    if (params.x2->GetDataType() == DataType::DT_INT4) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.x1, W4A8_IN1_TYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(params.x2, W4A8_IN2_TYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(params.scale, W4A8_SCALE_TYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(params.rowIndex, W4A8_ROW_INDEX_TYPE_SUPPORT_LIST, return false);
        if (params.offset != nullptr) {
            OP_CHECK_DTYPE_NOT_SUPPORT(params.offset, OFFSET_TYPE_SUPPORT_LIST, return false);
        }

        if (params.bias != nullptr) {
            OP_CHECK_DTYPE_NOT_SUPPORT(params.bias, W4A8_BIAS_TYPE_SUPPORT_LIST, return false);
        }
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.x1, IN_TYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(params.x2, IN_TYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(params.scale, SCALE_TYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(params.rowIndex, ROW_INDEX_TYPE_SUPPORT_LIST, return false);
        if (params.bias != nullptr) {
            OP_CHECK_DTYPE_NOT_SUPPORT(params.bias, BIAS_TYPE_SUPPORT_LIST, return false);
        }
    }

    if (params.pertokenScaleOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.pertokenScaleOptional, PERTOKEN_SCALE_TYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(params.groupList, GROUP_LIST_TYPE_SUPPORT_LIST, return false);
    if (params.shareInput != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.shareInput, SHARED_INPUT_TYPE_SUPPORT_LIST, return false);
    }
    if (params.logit != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.logit, LOGIT_TYPE_SUPPORT_LIST, return false);
    }
    
    OP_CHECK_DTYPE_NOT_SUPPORT(params.out, OUT_TYPE_SUPPORT_LIST, return false);
    if (params.shareInput != nullptr && params.logit != nullptr && params.out->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "when shareInput is not null and logit is not null, y's dtype should be [DT_FLOAT], actual is [%s].",
            op::ToString(params.out->GetDataType()).GetString());
        return false;
    }
    if (params.x2->GetDataType() == DataType::DT_INT4) {
        return true;
    }

    // 无芯片差异的公共校验
    if (params.x1->GetDataType() != params.x2->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 and x2 dtype should be same, actual x1 dtype is %s and x2 dtype is %s.",
            op::ToString(params.x1->GetDataType()).GetString(), op::ToString(params.x2->GetDataType()).GetString());
        return false;
    }

    return true;
}

static bool IsFormatSupport(const aclTensor *input, Format format, const std::string &inputName)
{
    if (input != nullptr && input->GetStorageFormat() != format) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s's format should be ND. actual is [%s].", inputName.c_str(),
            op::ToString(input->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckFormat(const GroupedMatmulParams &params)
{
    CHECK_RET(IsFormatSupport(params.x1, Format::FORMAT_ND, "x"), false);
    CHECK_RET(IsFormatSupport(params.scale, Format::FORMAT_ND, "scale"), false);
    CHECK_RET(IsFormatSupport(params.bias, Format::FORMAT_ND, "bias"), false);
    CHECK_RET(IsFormatSupport(params.pertokenScaleOptional, Format::FORMAT_ND, "pertokenScaleOptional"), false);
    CHECK_RET(IsFormatSupport(params.groupList, Format::FORMAT_ND, "groupList"), false);
    CHECK_RET(IsFormatSupport(params.shareInput, Format::FORMAT_ND, "shareInput"), false);
    CHECK_RET(IsFormatSupport(params.logit, Format::FORMAT_ND, "logit"), false);
    CHECK_RET(IsFormatSupport(params.rowIndex, Format::FORMAT_ND, "rowIndex"), false);
    CHECK_RET(IsFormatSupport(params.offset, Format::FORMAT_ND, "offset"), false);
    CHECK_RET(IsFormatSupport(params.out, Format::FORMAT_ND, "out"), false);
    if (params.x2 != nullptr && params.x2->GetStorageFormat() != Format::FORMAT_ND &&
        params.x2->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2's format should be ND or FRACTAL_NZ. actual is [%s].",
            op::ToString(params.x2->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckDimRange(const GroupedMatmulParams &params)
{
    auto x2StorageFormat = ge::GetPrimaryFormat(params.x2->GetStorageFormat());
    int64_t x2MaxDimNum = x2StorageFormat == op::Format::FORMAT_FRACTAL_NZ ? MAX_DIM_NUM_NZ + 1 : MAX_DIM_NUM_ND + 1;
    int64_t x2MinDimNum = x2StorageFormat == op::Format::FORMAT_FRACTAL_NZ ? MIN_DIM_NUM_NZ + 1 : MIN_DIM_NUM_ND + 1;
    int64_t x2DimNum = params.x2->GetStorageShape().GetDimNum();
    CHECK_RET(x2DimNum >= x2MinDimNum && x2DimNum <= x2MaxDimNum, false);
    OP_CHECK_MIN_DIM(params.x1, MIN_DIM_NUM_ND, return false);
    OP_CHECK_MIN_DIM(params.out, MIN_DIM_NUM_ND, return false);

    if (CheckType(params.x2->GetDataType(), X_WEIGHT_TYPE_SUPPORT_LIST_MX)) {
        OP_CHECK_WRONG_DIMENSION(params.scale, MX_SCALE_DIM, return false);
        if (params.bias != nullptr) {
            OP_CHECK_WRONG_DIMENSION(params.bias, TWO_DIM_NUM, return false);
        }
    } else if (params.x2->GetDataType() == DataType::DT_INT4) {
        OP_CHECK_WRONG_DIMENSION(params.scale, W4A8_SCALE_DIM, return false);
        if (params.offset != nullptr) {
            OP_CHECK_WRONG_DIMENSION(params.offset, W4A8_SCALE_DIM, return false);
        }
        if (params.bias != nullptr) {
            OP_CHECK_WRONG_DIMENSION(params.bias, TWO_DIM_NUM, return false);
        }
    } else {
        OP_CHECK_WRONG_DIMENSION(params.scale, SCALE_DIM, return false);
    }

    if (CheckType(params.x1->GetDataType(), X_WEIGHT_TYPE_SUPPORT_LIST_MX)) {
        OP_CHECK_WRONG_DIMENSION(params.pertokenScaleOptional, MX_PERTOKEN_SCALE_DIM, return false);
    } else if (params.pertokenScaleOptional != nullptr) {
        OP_CHECK_WRONG_DIMENSION(params.pertokenScaleOptional, 1, return false);
    }

    OP_CHECK_WRONG_DIMENSION(params.groupList, 1, return false);
    if (params.shareInput != nullptr) {
        OP_CHECK_WRONG_DIMENSION(params.shareInput, 2, return false);
    }
    if (params.logit != nullptr) {
        OP_CHECK_WRONG_DIMENSION(params.logit, 1, return false);
    }
    OP_CHECK_WRONG_DIMENSION(params.rowIndex, 1, return false);
    OP_LOGD("GroupedMatmulFinalizeRouting check dim-num range success");
    return true;
}

static inline std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> GetX1X2DimValue(const aclTensor *x1,
    const aclTensor *x2, bool transposeX1, bool transposeX2)
{
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    if (!(x1DimNum == TWO_DIM_NUM && x2DimNum == THREE_DIM_NUM)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input x Dims is %zu, weight Dims is %zu.", x1DimNum, x2DimNum);
    }

    const op::Shape x1Shape = x1->GetViewShape();
    const op::Shape x2Shape = x2->GetViewShape();
    int64_t x1KDim = transposeX1 ? x1Shape[x1DimNum - PENULTIMATE_DIM] : x1Shape[x1DimNum - 1];
    int64_t x1MDim = transposeX1 ? x1Shape[x1DimNum - 1] : x1Shape[x1DimNum - PENULTIMATE_DIM];
    int64_t x2EDim = x2Shape[0]; // E
    int64_t x2KDim = transposeX2 ? x2Shape[x2DimNum - 1] : x2Shape[x2DimNum - PENULTIMATE_DIM];
    int64_t x2NDim = transposeX2 ? x2Shape[x2DimNum - PENULTIMATE_DIM] : x2Shape[x2DimNum - 1];
    return std::tie(x1KDim, x1MDim, x2EDim, x2KDim, x2NDim);
}

static inline bool CheckW4orW8DimValue(const CheckW4orW8DimParams& params, bool isWeightInt4)
{
    if (!isWeightInt4) {
        if (!(params.scale->GetViewShape().GetDim(0) == params.x2EDim && params.scale->GetViewShape().GetDim(1) == params.x2NDim)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Scale fisrt dim should equal to weight EDim %lld, but actual is %lld.\
            Scale last dim should equal to weight NDim %lld or 1, but actual is %lld.",
                params.x2EDim, params.scale->GetViewShape().GetDim(0), params.x2NDim, params.scale->GetViewShape().GetDim(1));
            return false;
        }
        return true;
    } else {
        if (!(params.scale->GetViewShape().GetDim(0) == params.x2EDim
              && params.scale->GetViewShape().GetDim(ONE_DIM_NUM) == ONE_DIM_NUM
              && params.scale->GetViewShape().GetDim(TWO_DIM_NUM) == params.x2NDim)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Scale fisrt dim should equal to weight EDim %lld, but actual is %lld,"
                                             " Scale second dim should be 1, but actual is %lld,"
                                             " Scale last dim should equal to weight NDim %lld or 1, but actual is %lld.",
                    params.x2EDim, params.scale->GetViewShape().GetDim(0),
                    params.scale->GetViewShape().GetDim(ONE_DIM_NUM), params.x2NDim, params.scale->GetViewShape().GetDim(TWO_DIM_NUM));
            return false;
        }
        if ((params.offset != nullptr) && 
                !(params.offset->GetViewShape().GetDim(0) == params.x2EDim
                  && params.offset->GetViewShape().GetDim(TWO_DIM_NUM) == params.x2NDim 
                  && params.offset->GetViewShape().GetDim(ONE_DIM_NUM) == ONE_DIM_NUM)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "w4a8 offset's dim value should equal scale,"
                    " which is (%lld, %lld, %lld)",
                    params.offset->GetViewShape().GetDim(0),
                    params.offset->GetViewShape().GetDim(ONE_DIM_NUM), params.offset->GetViewShape().GetDim(TWO_DIM_NUM));
            return false;
            }
        if (!(params.bias->GetViewShape().GetDim(0) == params.x2EDim
            && params.bias->GetViewShape().GetDim(ONE_DIM_NUM) == params.x2NDim)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "w4a8 bias's dim value should be (e, n),"
                    " which is (%lld, %lld)",
                    params.bias->GetViewShape().GetDim(0), params.bias->GetViewShape().GetDim(ONE_DIM_NUM));
            return false;
        }
        return true;
    }
}

static inline bool CheckDimValue(GroupedMatmulParams &params, int64_t x2EDim, int64_t x2NDim, int64_t x1MDim,
    int64_t outputBS, int64_t shareInputOffset, bool isWeightInt4)
{
    CheckW4orW8DimParams dimParams{x2EDim, x2NDim, x1MDim, params.scale, params.offset, params.bias};
    if(!(CheckW4orW8DimValue(dimParams, isWeightInt4))){
        return false;
    }

    if ((params.pertokenScaleOptional != nullptr) && (!(params.pertokenScaleOptional->GetViewShape().GetDim(0) == x1MDim))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pertokenScale fisrt dim should equal to x MDim %lld, but actual is %lld.",
            x1MDim, params.pertokenScaleOptional->GetViewShape().GetDim(0));
        return false;
    }

    if (!(params.groupList->GetViewShape().GetDim(0) == x2EDim)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groupList fisrt dim should equal to weight EDim %lld, but actual is %lld.", x2EDim,
            params.groupList->GetViewShape().GetDim(0));
        return false;
    }

    if (params.shareInput != nullptr &&
        !(params.shareInput->GetViewShape().GetDim(0) <= outputBS && params.shareInput->GetViewShape().GetDim(1) == x2NDim)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "shareInput fisrt dim should less than or equal to outputBS %lld, but actual is %lld.\
        shareInput last dim should equal to weight NDim %lld, but actual is %lld.",
            outputBS, params.shareInput->GetViewShape().GetDim(0), x2NDim, params.shareInput->GetViewShape().GetDim(1));
        return false;
    }

    if (params.logit != nullptr && !(params.logit->GetViewShape().GetDim(0) == x1MDim)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "logit fisrt dim should equal to x MDim %lld, but actual is %lld.",
            x1MDim, params.logit->GetViewShape().GetDim(0));
        return false;
    }

    if (!(params.rowIndex->GetViewShape().GetDim(0) == x1MDim)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rowIndex fisrt dim should equal to x MDim %lld, but actual is %lld.",
            x1MDim, params.rowIndex->GetViewShape().GetDim(0));
        return false;
    }

    OP_CHECK(outputBS <= x1MDim,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outputBS should be less than or equal to x MDim, but outputBS is %lld, x MDim is %lld.", outputBS,
        x1MDim),
        return false);

    auto shareInputDim = 0;
    if (params.shareInput != nullptr) {
        shareInputDim = params.shareInput->GetViewShape().GetDim(0);
    }

    OP_CHECK((shareInputOffset + shareInputDim) <= outputBS,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "shareInputOffset add shareInputDim should be less than or equal to outputBS, but shareInputOffset is %lld, weight is %lld.",
        shareInputOffset + shareInputDim, outputBS),
        return false);

    return true;
}

static inline bool CheckShapeForWeightNz(const aclTensor *x1, const aclTensor *x2, bool transposeX1, bool transposeX2)
{
    const op::Shape x1Shape = x1->GetViewShape();
    const op::Shape x2Shape = x2->GetStorageShape();
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetStorageShape().GetDimNum();
    int64_t x1KDim = transposeX1 ? x1Shape[x1DimNum - PENULTIMATE_DIM] : x1Shape[x1DimNum - 1];
    int64_t x2K1Dim = transposeX2 ? x2Shape[x2DimNum - NZ_K1_INDEX_TRANS] : x2Shape[x2DimNum - NZ_K1_INDEX];
    int64_t aligneValue = x2->GetDataType() == DataType::DT_INT4 ?
                            (transposeX2 ? NZ_K0_VALUE_INT4_TRANS : NZ_K0_VALUE_INT4) :
                            (transposeX2 ? NZ_K0_VALUE_INT8_TRANS : NZ_K0_VALUE_INT8);
    int64_t alignedX1K = ((x1KDim + aligneValue - 1) / aligneValue) * aligneValue;
    if (alignedX1K != x2K1Dim * aligneValue) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "AlignedK1 value %lld is not matched with k1 value times aligneValue, which is %lld.", alignedX1K,
            x2K1Dim * aligneValue);
        return false;
    }
    return true;
}

static bool CheckInputViewShape(const op::Shape &viewShape)
{
    uint64_t viewDimMutltiply = 1;
    uint64_t viewDimNum = viewShape.GetDimNum();
    for (uint64_t i = 0; i < viewDimNum; i++) {
        viewDimMutltiply *= viewShape[i];
    }

    return viewDimMutltiply == 0;
}

static inline bool CheckShape(GroupedMatmulParams &params)
{
    auto shareInputOffset = params.shareInputOffset;
    int64_t groupListType = params.groupListType;
    bool transposeX1 = params.transposeX1;
    bool transposeX2 = params.transposeX2;

    int64_t x1KDim, x1MDim, x2EDim, x2KDim, x2NDim;
    std::tie(x1KDim, x1MDim, x2EDim, x2KDim, x2NDim) = GetX1X2DimValue(params.x1, params.x2, transposeX1, transposeX2);
    bool isWeightInt4 = (params.x2->GetDataType() == DataType::DT_INT4);
    bool isWeightNz = ge::GetPrimaryFormat(params.x2->GetStorageFormat()) == Format::FORMAT_FRACTAL_NZ;
    if (isWeightNz) {
        CHECK_RET(CheckShapeForWeightNz(params.x1, params.x2, transposeX1, transposeX2), false);
    }

    OP_CHECK(x1KDim == x2KDim, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 k dim and x2 k dim should be same,"
                " but x1 is %lld, x2 is %lld.", x1KDim, x2KDim), return false);

    if (!isWeightInt4) {
        OP_CHECK(((x2KDim % NZ_K0_VALUE_INT8 == 0) && (x2NDim % NZ_K0_VALUE_INT8_TRANS == 0) && (x2NDim >= N_VALUE_256)),
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "k dim %lld and n dim %lld do not support."
                " only support k mod 16 equal 0, n mod 32 equal 0 and n bigger or equal 256", x2KDim, x2NDim), return false);
    } else {
        int n_align = isWeightNz ? NZ_N_VALUE_ALIGN : ND_N_VALUE_ALIGN;
        OP_CHECK(((x2NDim % n_align == 0) && (x2KDim % ND_K_VALUE_ALIGN == 0) && (x2NDim > N_VALUE_64) && (x2KDim > K_VALUE_128)),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Weight k dim %lld and n dim %lld do not support."
            " In W4A8 mode, only support k mod 64 equal 0, n mod %d equal 0, k should bigger than 128, n should bigger than 64.", x2KDim, x2NDim, n_align),
            return false);
    }

    int64_t outDimNum = params.out->GetViewShape().GetDimNum();
    OP_CHECK(outDimNum == 2, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Out dim num should be equal to 2, but is %lld.", outDimNum), return false);
    int64_t outputBS = params.out->GetViewShape().GetDim(outDimNum - PENULTIMATE_DIM);
    int64_t outNDim = params.out->GetViewShape().GetDim(outDimNum - 1);

    OP_CHECK(outputBS <= x1MDim && outputBS >= 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Out 1st dim should be less than or equal to x MDim , but out 1st dim is %lld", outputBS), return false);
    OP_CHECK(outNDim == x2NDim, OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "Out 2nd dim should be equal to weight NDim, but out 2nd dim is %lld, weight NDim is %lld.", outNDim, x2NDim), return false);
    OP_CHECK(shareInputOffset >= 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sharedInputOffset should bigger than or equal to 0"), return false);
    OP_CHECK(groupListType == 0 || groupListType == 1, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groupListType should be 0 or 1, but is %lld", groupListType), return false);

    CHECK_RET(CheckDimValue(params, x2EDim, x2NDim, x1MDim, outputBS, shareInputOffset, isWeightInt4), false);

    if (params.pertokenScaleOptional != nullptr && CheckInputViewShape(params.pertokenScaleOptional->GetViewShape())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GroupedMatmulFinalizeRouting do not support pertokenScaleOptional size is 0.");
        return false;
    }

    if (CheckInputViewShape(params.x1->GetViewShape()) || CheckInputViewShape(params.x2->GetViewShape()) || CheckInputViewShape(params.scale->GetViewShape())
         || CheckInputViewShape(params.groupList->GetViewShape()) || (params.shareInput != nullptr && CheckInputViewShape(params.shareInput->GetViewShape()))
         || (params.logit != nullptr && CheckInputViewShape(params.logit->GetViewShape())) || CheckInputViewShape(params.rowIndex->GetViewShape())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GroupedMatmulFinalizeRouting WeightNz do not support input tensor size is 0.");
        return false;
    }
    
    return true;
}

static inline bool CheckEmptyTensor(const GroupedMatmulParams &params)
{
    // scale, out和可选参数已在CheckShape函数校验，无需再次校验空tensor场景。
    if (params.x1->IsEmpty() || params.x2->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GroupedMatmulFinalizeRouting not support to process empty tensor currently.");
        return false;
    }
    return true;
}

static inline bool CheckTuningConfig(const GroupedMatmulParams &params)
{
    const op::Shape x1Shape = params.x1->GetViewShape();
    auto tuningConfig = params.tuningConfig;

    if (tuningConfig != nullptr && tuningConfig->Size() > 0) {
        auto tuningConfigVal = (*tuningConfig)[0];
        if (tuningConfigVal < 0 || tuningConfigVal > x1Shape[0]) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GroupedMatmulFinalizeRouting Invalid tuningConfigOptional (%lld)! It should"
            " be a non-negative num and smaller than (%lld)", tuningConfigVal, x1Shape[0]);
            return false;
        }
    }

    return true;
}

static aclnnStatus CheckParams(GroupedMatmulParams &params)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        GmmFinalizeRouting::AclnnGroupedMatmulFinalizeRoutingDAV3510Checker checker;
        aclnnStatus status = checker.CheckParams(params);
        CHECK_RET(status == ACLNN_SUCCESS, status);
    } else {
        // 1. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
        CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);
        // 2. 检查shape是否符合要求
        CHECK_RET(CheckShape(params), ACLNN_ERR_PARAM_INVALID);
        // 3. 检查format是否符合要求
        CHECK_RET(CheckFormat(params), ACLNN_ERR_PARAM_INVALID);
        // 4. 空Tensor处理逻辑
        CHECK_RET(CheckEmptyTensor(params), ACLNN_ERR_PARAM_INVALID);
        // 5. tuningConfig 逻辑校验
        CHECK_RET(CheckTuningConfig(params), ACLNN_ERR_PARAM_INVALID);
    }

    OP_LOGD("GroupedMatmulFinalizeRouting check params success.");
    return ACLNN_SUCCESS;
}

static const aclTensor *SetTensorToNDFormat(const aclTensor *input)
{
    OP_LOGD("GroupedMatmulFinalizeRouting set tensor to ND format.");
    auto formatTensor = const_cast<aclTensor*>(input);
    if (input->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) {
        formatTensor->SetViewFormat(op::Format::FORMAT_ND);
        formatTensor->SetOriginalFormat(op::Format::FORMAT_ND);
        formatTensor->SetStorageFormat(op::Format::FORMAT_ND);
    }
    return formatTensor;
}

static bool IsLastTwoDimsTranspose(const aclTensor *tensor) {
    // 当输入tensor的shape小于2或者大于6的时候，返回错误
    if (tensor->GetViewShape().GetDimNum() < 2 || tensor->GetViewShape().GetDimNum() > 6) {
         return false;
    }
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - 2;
    // BMM 场景下，Batch维度的stride需要等于 N, D 的乘积
    if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        int64_t tmpNxD = tensor->GetViewShape().GetDim(dim1) * tensor->GetViewShape().GetDim(dim2);
        // 多batch连续，3是batch索引
        for (int64_t batchDim = tensor->GetViewShape().GetDimNum() - 3; batchDim >= 0; batchDim--) {
            if (tensor->GetViewStrides()[batchDim] != tmpNxD) {
                return false;
            }
            tmpNxD *= tensor->GetViewShape().GetDim(batchDim);
        }
        if (tensor->GetViewShape().GetDim(dim1) == 1 && tensor->GetViewShape().GetDim(dim2) == 1) {
            return false;
        }
        return true;
    }
    return false;
}

static bool IsTransposeForMxScale(const aclTensor *tensor) {
    // 当输入tensor的shape不为4维的时候，返回错误
    if (tensor->GetViewShape().GetDimNum() != FOUR_DIM) {
         return false;
    }
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 2; // 倒数第二维
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - 3; // 倒数第三维
    // 在Mx量化的场景下，最后一维固定的维度为2
    if (tensor->GetViewStrides()[dim2] == 2 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2) * 2) {
        int64_t tmpNxD = tensor->GetViewShape().GetDim(dim1) * tensor->GetViewShape().GetDim(dim2) * 2;
        // 从倒数第四维开始，检查前面的维度是否连续
        for (int64_t batchDim = tensor->GetViewShape().GetDimNum() - 4; batchDim >= 0; batchDim--) {
            if (tensor->GetViewStrides()[batchDim] != tmpNxD) {
                return false;
            }
            tmpNxD *= tensor->GetViewShape().GetDim(batchDim);
        }
        if (tensor->GetViewShape().GetDim(dim1) == 1 && tensor->GetViewShape().GetDim(dim2) == 1) {
            return false;
        }
        return true;
    }
    return false;
}

static op::Shape SwapLastTwoDimValue(const op::Shape tensorShape)
{
    op::Shape swapedShape = tensorShape;
    int64_t dimNum = tensorShape.GetDimNum();
    if (static_cast<size_t>(dimNum) >= MM_DIM) {
        int64_t lastDim = tensorShape.GetDim(dimNum - 1);
        // dimNum - 1, 这里1指的是取最后一维的dim值。dimNum - 2, 这里2指的是取倒数第二维的dim值
        swapedShape.SetDim(dimNum - 1, tensorShape.GetDim(dimNum - 2));
        // dimNum - 2, 这里2指的是取倒数第二维的dim值
        swapedShape.SetDim(dimNum - 2, lastDim);
    }
    else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimNum is not supported , which is %lld.", dimNum);
    }
    return swapedShape;
}

static op::Shape SwapLastSecondAndThirdDimValue(const op::Shape& tensorShape)
{
    op::Shape swapedShape = tensorShape;
    int64_t dimNum = tensorShape.GetDimNum();
    int64_t lastSecondDim = tensorShape.GetDim(dimNum - 2);
    // dimNum - 2, 这里1指的是取倒数第二维的dim值。dimNum - 3, 这里3指的是取倒数第三维的dim值
    swapedShape.SetDim(dimNum - 2, tensorShape.GetDim(dimNum - 3));
    // dimNum - 3, 这里3指的是取倒数第三维的dim值
    swapedShape.SetDim(dimNum - 3, lastSecondDim);

    return swapedShape;
}


static inline bool TransposeTensorContiguousProcess(const aclTensor *&contiguousTensor, bool &transpose, aclOpExecutor *executor)
{
    if (contiguousTensor == nullptr || contiguousTensor->GetViewShape().GetDimNum() == 1) {
        OP_LOGD("GroupedMatmulFinalizeRouting no need to do contiguous process.");
        return true;
    }

    auto transposeFlag = IsLastTwoDimsTranspose(contiguousTensor);
    // swap tensor if its viewshape not satisfy request shape without adding a transpose node
    if (transposeFlag) {
        contiguousTensor = executor->CreateView(contiguousTensor, SwapLastTwoDimValue(contiguousTensor->GetViewShape()),
            contiguousTensor->GetViewOffset());
        transpose = !transpose;
    } else {
        contiguousTensor = l0op::Contiguous(contiguousTensor, executor);
    }
    CHECK_RET(contiguousTensor != nullptr, false);
    return true;
}

static inline bool TransposeTensorContiguousProcessForMx(const aclTensor *&contiguousTensor, bool &transpose, aclOpExecutor *executor)
{
    if (contiguousTensor == nullptr || contiguousTensor->GetViewShape().GetDimNum() == 1) {
        OP_LOGD("GroupedMatmulFinalizeRouting no need to do contiguous process.");
        return true;
    }

    auto transposeFlag = IsLastTwoDimsTranspose(contiguousTensor);
    // swap tensor if its viewshape not satisfy request shape without adding a transpose node
    if (transposeFlag) {
        contiguousTensor = executor->CreateView(contiguousTensor, SwapLastTwoDimValue(contiguousTensor->GetViewShape()),
            contiguousTensor->GetViewOffset());
        transpose = true;
    } else {
        contiguousTensor = l0op::Contiguous(contiguousTensor, executor);
    }
    CHECK_RET(contiguousTensor != nullptr, false);
    return true;
}

static inline bool TransposeTensorContiguousProcessForMXScale(const aclTensor *&contiguousTensor, bool &transpose, aclOpExecutor *executor)
{   
    // 检查MX scale是不是四维
    if (contiguousTensor == nullptr || contiguousTensor->GetViewShape().GetDimNum() != FOUR_DIM) {
        OP_LOGD("GroupedMatmulFinalizeRouting  scale no need to do contiguous process.");
        return true;
    }
    auto transposeFlag = IsTransposeForMxScale(contiguousTensor);
    // swap tensor if its viewshape not satisfy request shape without adding a transpose node
    if (transposeFlag) {
        contiguousTensor = executor->CreateView(contiguousTensor, SwapLastSecondAndThirdDimValue(contiguousTensor->GetViewShape()),
            contiguousTensor->GetViewOffset());
        transpose = true;
    } else {
        contiguousTensor = l0op::Contiguous(contiguousTensor, executor);
    }
    CHECK_RET(contiguousTensor != nullptr, false);
    return true;
}


static inline bool TensorContiguousProcess(const aclTensor *&contiguousTensor, aclOpExecutor *executor)
{
    if (contiguousTensor == nullptr) {
        OP_LOGD("GroupedMatmulFinalizeRouting no need to do contiguous process.");
        return true;
    }

    contiguousTensor = l0op::Contiguous(contiguousTensor, executor);

    CHECK_RET(contiguousTensor != nullptr, false);
    return true;
}

static aclnnStatus SpecialOutputProcess(const aclTensor *x1, const aclTensor *x2, const aclTensor *out,
    const aclTensor *&matmulRet, aclOpExecutor *executor)
{
    // we have to reshape for case which x1 and x2 are 2 dims and out is 3 dims, otherwise, viewcopy will fail
    OP_LOGD("GroupedMatmulFinalizeRouting enter SpecialOutputProcess func.");
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    auto outShape = out->GetViewShape();
    auto outDimNum = outShape.GetDimNum();
    int64_t outMDim = outShape.GetDim(outDimNum - 2);
    // speical case : x1 and x2 are 2 dim, output is 3 dim, have to reshape matmul result, otherwise viewcopy will fail.
    if (x1DimNum == 2 && outDimNum == 3 && outMDim == 1 && x2DimNum == 2) {
        matmulRet = l0op::Reshape(matmulRet, outShape, executor);
    }
    CHECK_RET(matmulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static const aclTensor *GetNDFormat(const aclTensor *input)
{
    const aclTensor *reformatedInput = input;
    if (input != nullptr) {
        reformatedInput = SetTensorToNDFormat(input);
    }
    return reformatedInput;
}

static aclnnStatus WeightNZCaseProcess(const aclTensor *&x2, bool &transposeX2, aclOpExecutor *executor)
{
    // if weight is already in nz format, no need to set contiguous
    if (ge::GetPrimaryFormat(x2->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            if (transposeX2 == false) {
                CHECK_RET(TransposeTensorContiguousProcessForMx(x2, transposeX2, executor), ACLNN_ERR_INNER_NULLPTR);
            }
        } 
    } else {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            if (transposeX2 == false) {
                CHECK_RET(TransposeTensorContiguousProcessForMx(x2, transposeX2, executor), ACLNN_ERR_INNER_NULLPTR);
            }
        } else {
            CHECK_RET(TransposeTensorContiguousProcess(x2, transposeX2, executor), ACLNN_ERR_INNER_NULLPTR);
        }
    }
    x2->SetOriginalShape(x2->GetViewShape());
    return ACLNN_SUCCESS;
}

static aclnnStatus WeightNZCaseProcessForMXScale(const aclTensor *&x2, bool &transposeX2, aclOpExecutor *executor)
{
    // if weight is already in nz format, no need to set contiguous
    if (ge::GetPrimaryFormat(x2->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GroupedMatmulFinalizeRoutingWeightV3: Scale only soupports ND format, but current format is FORMAT_FRACTAL_NZ.");
        return ACLNN_ERR_PARAM_INVALID;
    } else {
        CHECK_RET(TransposeTensorContiguousProcessForMXScale(x2, transposeX2, executor), ACLNN_ERR_INNER_NULLPTR);
    }
    x2->SetOriginalShape(x2->GetViewShape());
    return ACLNN_SUCCESS;
}

static aclnnStatus PostMatmulCalcProcess(const aclTensor *matmulRet, const GroupedMatmulParams &params,
    aclOpExecutor *executor)
{
    CHECK_RET(matmulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(SpecialOutputProcess(params.x1, params.x2, params.out, matmulRet, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(matmulRet, params.out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

inline static int64_t CeilDiv(int64_t value, int64_t factor)
{
    int64_t valueNum = 0;
    if (factor == 0) {
        return valueNum;
    }
    if (value % factor == 0) {
        valueNum = value / factor;
    } else {
        valueNum = value / factor + 1;
    }
    return valueNum;
}

static aclnnStatus PreMatmulCalcProcess(GroupedMatmulParams &params, aclOpExecutor *executor)
{
    auto &x1 = params.x1;
    auto &x2 = params.x2;
    bool &transposeX1 = params.transposeX1;
    bool &transposeX2 = params.transposeX2;
    auto &scale = params.scale;
    
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(TransposeTensorContiguousProcess(x1, transposeX1, executor), ACLNN_ERR_INNER_NULLPTR);
    auto ret = WeightNZCaseProcess(x2, transposeX2, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (scale != nullptr && CheckType(x1->GetDataType(), X_WEIGHT_TYPE_SUPPORT_LIST_MX) && CheckType(scale->GetDataType(), SCALE_TYPE_SUPPORT_LIST_MX)) {
        bool transposescale = false;
        ret = WeightNZCaseProcessForMXScale(scale, transposescale, executor);
        CHECK_RET(transposeX2 == transposescale, ret);
    }

    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        CHECK_RET(CheckDimRange(params), ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus aclnnGroupedMatmulFinalizeRoutingGetWorkspaceSizeCommonProcess(GroupedMatmulParams &params, aclOpExecutor *executor)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        auto x1MDim = params.x1->GetViewShape().GetDim(0);
        auto x2NIndex = params.x2->GetViewShape().GetDimNum() - (params.transposeX2 ? PENULTIMATE_DIM : 1);
        auto x2NDim = params.x2->GetViewShape().GetDim(x2NIndex);
        if (x1MDim == 0 || x2NDim == 0) {
            return ACLNN_SUCCESS;
        }
    }
    auto ret = PreMatmulCalcProcess(params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // shareInput格式转换
    if (params.shareInput != nullptr) {
        CHECK_RET(TensorContiguousProcess(params.shareInput, executor), ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(TensorContiguousProcess(params.x1, executor), ACLNN_ERR_INNER_NULLPTR);
    if (ge::GetPrimaryFormat(params.x2->GetStorageFormat()) != op::Format::FORMAT_FRACTAL_NZ) {
        CHECK_RET(TensorContiguousProcess(params.x2, executor), ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(TensorContiguousProcess(params.scale, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(TensorContiguousProcess(params.offset, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(TensorContiguousProcess(params.bias, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(TensorContiguousProcess(params.rowIndex, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(TensorContiguousProcess(params.groupList, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(TensorContiguousProcess(params.logit, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(TensorContiguousProcess(params.pertokenScaleOptional, executor), ACLNN_ERR_INNER_NULLPTR);
    auto reformatedX1 = SetTensorToNDFormat(params.x1);
    const aclTensor *reformatedX2 = SetTensorToNDFormat(params.x2);
    const aclTensor *reformatedScale = GetNDFormat(params.scale);
    const aclTensor *reformatedBias = GetNDFormat(params.bias);
    const aclTensor *reformatedPertokenScaleOptional = GetNDFormat(params.pertokenScaleOptional);
    const aclTensor *reformatedGroupList = GetNDFormat(params.groupList);
    const aclTensor *reformatedShareInput = GetNDFormat(params.shareInput);
    const aclTensor *reformatedLogit = GetNDFormat(params.logit);
    const aclTensor *reformatedRowIndex = GetNDFormat(params.rowIndex);
    const aclTensor *reformatedOffset = GetNDFormat(params.offset);
    GroupedMatmulParams params2 = params;
    params2.x1 = reformatedX1;
    params2.x2 = reformatedX2;
    params2.scale = reformatedScale;
    params2.bias = reformatedBias;
    params2.pertokenScaleOptional = reformatedPertokenScaleOptional;
    params2.groupList = reformatedGroupList;
    params2.shareInput = reformatedShareInput;
    params2.logit = reformatedLogit;
    params2.rowIndex = reformatedRowIndex;
    params2.offset = reformatedOffset;
    
    ret = CheckParams(params2);
    
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    int64_t outDimNum = params.out->GetViewShape().GetDimNum();
    int64_t outputBS = params.out->GetViewShape().GetDim(outDimNum - PENULTIMATE_DIM);
    
    // 调用l0算子GroupedMatmulFinalizeRouting进行计算，包含infershape
    auto matmulRet = l0op::GroupedMatmulFinalizeRouting(reformatedX1, params2.x2, params2.scale, reformatedBias,
        reformatedPertokenScaleOptional, reformatedGroupList, reformatedShareInput, reformatedLogit, reformatedRowIndex,
        reformatedOffset, 0, params.shareInputWeight, params.shareInputOffset, params.transposeX1, params.transposeX2, outputBS, params.groupListType, params.tuningConfig, executor);
    ret = PostMatmulCalcProcess(matmulRet, params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}
}

namespace {
static op::Shape GetWeightNzShape(const aclTensor *input, bool transpose)
{
    bool isWeightInt4 = input->GetDataType() == DataType::DT_INT4;
    int64_t viewDimNum = input->GetViewShape().GetDimNum();
    uint64_t k = transpose ? input->GetViewShape().GetDim(viewDimNum - 1) :
                             input->GetViewShape().GetDim(viewDimNum - LAST_SECOND_DIM_INDEX);
    uint64_t n = transpose ? input->GetViewShape().GetDim(viewDimNum - LAST_SECOND_DIM_INDEX) :
                             input->GetViewShape().GetDim(viewDimNum - 1);

    uint64_t k0 = transpose ? (isWeightInt4 ? NZ_K0_VALUE_INT4_TRANS : NZ_K0_VALUE_INT8_TRANS) :
                              (isWeightInt4 ? NZ_K0_VALUE_INT4 : NZ_K0_VALUE_INT8);
    uint64_t n0 = transpose ? (isWeightInt4 ? NZ_K0_VALUE_INT4 : NZ_K0_VALUE_INT8) :
                              (isWeightInt4 ? NZ_K0_VALUE_INT4_TRANS : NZ_K0_VALUE_INT8_TRANS);
    uint64_t k1 = CeilDiv(k, k0);
    uint64_t n1 = CeilDiv(n, n0);

    op::Shape weightNzShape;
    for (uint64_t i = 0; i < static_cast<uint64_t>(viewDimNum - LAST_SECOND_DIM_INDEX); i++) {
        weightNzShape.AppendDim(input->GetViewShape().GetDim(i));
    }
    if (transpose) {
        weightNzShape.AppendDim(k1);
        weightNzShape.AppendDim(n1);
    } else {
        weightNzShape.AppendDim(n1);
        weightNzShape.AppendDim(k1);
    }
    weightNzShape.AppendDim(NZ_STORAGE_PENULTIMATE_DIM);
    weightNzShape.AppendDim(isWeightInt4 ? NZ_STORAGE_LAST_DIM * PER_INT4_IN_U8 : NZ_STORAGE_LAST_DIM);
    return weightNzShape;
}

static bool CheckWeightNzStorageShape(const op::Shape &nzShape, const op::Shape &storageShape)
{
    uint64_t nzDimMultiply = 1;
    uint64_t nzDimNum = nzShape.GetDimNum();
    for (uint64_t i = 0; i < nzDimNum; i++) {
        nzDimMultiply *= nzShape[i];
    }

    uint64_t storageDimMultiply = 1;
    uint64_t storageDimNum = storageShape.GetDimNum();
    for (uint64_t i = 0; i < storageDimNum; i++) {
        storageDimMultiply *= storageShape[i];
    }

    return nzDimMultiply == storageDimMultiply;
}

static inline aclnnStatus CheckWeightNzFormat(const aclTensor *x1, const aclTensor *x2, bool transposeX2)
{
    // only support x1 ND, x2 NZ
    if (x1->GetStorageFormat() != op::Format::FORMAT_ND || x2->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GroupedMatmulFinalizeRoutingWeightNzV2 do not support x1's format[%s],"
                "x2's format[%s]. Only support x1: ND, x2: NZ", op::ToString(x1->GetStorageFormat()).GetString(),
                op::ToString(x2->GetStorageFormat()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // check x2(NZ)'s shape
    op::Shape weightNzShape = GetWeightNzShape(x2, transposeX2);
    if (!CheckWeightNzStorageShape(weightNzShape, x2->GetStorageShape())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "GroupedMatmulFinalizeRoutingWeightNzV2 x2'format only support NZ, but now x2's format \
                is not NZ(Ascend affinity format). aclnnCalculateMatmulWeightSizeV2 and aclnnTransMatmulWeight can be \
                used to convert the input format from ND to Ascend affinity format.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckSupportScene(const CheckSupportSceneParams& params, bool transposeX, bool transposeW)
{
    // 支持sharedInput输入为空, 不支持logit为空
    auto scene = params.x != nullptr && params.w != nullptr && params.scaleOptional != nullptr && params.logitOptional != nullptr
        && params.groupListOptional != nullptr && params.rowIndexOptional != nullptr;

    if (!scene) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "GroupedMatmulFinalizeRoutingWeightNz do not support input nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    if (params.dtype != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GroupedMatmulFinalizeRoutingWeightNz dtype must be 0, but is %lld.", params.dtype);
        return ACLNN_ERR_PARAM_INVALID;
    }

    int64_t viewDimNum = params.w->GetViewShape().GetDimNum();
    if (viewDimNum < MIN_DIM_NUM_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "GroupedMatmulFinalizeRoutingWeightNz w's view dimNum should greater than 1, but is %lld.", viewDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (!(transposeX == false && transposeW == false)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GroupedMatmulFinalizeRoutingWeightNz transpose should be false");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // check input format
    auto ret0 = CheckWeightNzFormat(params.x, params.w, transposeW);
    CHECK_RET(ret0 == ACLNN_SUCCESS, ret0);
    return ACLNN_SUCCESS;
}
}

aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNzGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
    const aclTensor *scale, const aclTensor *bias, const aclTensor *pertokenScaleOptional, const aclTensor *groupList,
    const aclTensor *sharedInput, const aclTensor *logit, const aclTensor *rowIndex, int64_t dtype,
    float sharedInputWeight, int64_t sharedInputOffset, bool transposeX1, bool transposeX2, int64_t groupListType,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnGroupedMatmulFinalizeRoutingWeightNz,
        DFX_IN(x1, x2, scale, bias, pertokenScaleOptional, groupList, sharedInput, logit, rowIndex, dtype,
        sharedInputWeight, sharedInputOffset, transposeX1, transposeX2, groupListType),
        DFX_OUT(out));
    
    CheckSupportSceneParams supportSceneParams{x1, x2, scale, pertokenScaleOptional, groupList, sharedInput,
                                               logit, rowIndex, dtype};
    auto ret0 = CheckSupportScene(supportSceneParams, transposeX1, transposeX2);
    CHECK_RET(ret0 == ACLNN_SUCCESS, ret0);
    const aclTensor* unused = nullptr;
    const aclIntArray* unusedTuningConfig = nullptr;
    auto uniqueExecutor = CREATE_EXECUTOR();
    GroupedMatmulParams params = GroupedMatmulParamsBuilder::Create(x1, x2, out)
        .SetScale(scale)
        .SetBias(bias)
        .SetPertokenScale(pertokenScaleOptional)
        .SetGroupList(groupList)
        .SetShareInput(sharedInput)
        .SetLogit(logit)
        .SetRowIndex(rowIndex)
        .SetOffset(unused)
        .SetTuningConfig(unusedTuningConfig)
        .SetNumbers(sharedInputWeight, sharedInputOffset, groupListType)
        .SetTranspose(transposeX1, transposeX2)
        .Build();
    auto ret = aclnnGroupedMatmulFinalizeRoutingGetWorkspaceSizeCommonProcess(params, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulFinalizeRoutingWeightNz);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
    const aclTensor *scale, const aclTensor *bias, const aclTensor *offsetOptional,
    const aclTensor *antiquantScaleOptional, const aclTensor *antiquantOffsetOptional,
    const aclTensor *pertokenScaleOptional, const aclTensor *groupList, const aclTensor *sharedInput,
    const aclTensor *logit, const aclTensor *rowIndex, int64_t dtype, float sharedInputWeight,
    int64_t sharedInputOffset, bool transposeX1, bool transposeX2, int64_t groupListType,
    const aclIntArray *tuningConfigOptional, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnGroupedMatmulFinalizeRoutingWeightNzV2,
        DFX_IN(x1, x2, scale, bias, pertokenScaleOptional, groupList, sharedInput,
        logit, rowIndex, dtype, sharedInputWeight, sharedInputOffset, transposeX1, transposeX2,
        groupListType),
        DFX_OUT(out));
    (void) antiquantScaleOptional;
    (void) antiquantOffsetOptional;
    auto viewShape = x2->GetViewShape();
    auto uniqueExecutor = CREATE_EXECUTOR();
    // unpack int32 to int4
    auto tmpWeight = uniqueExecutor.get()->CreateView(x2, viewShape, x2->GetViewOffset());
    if(op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510){
        auto storageShape = x2->GetStorageShape();
        if (tmpWeight->GetDataType() == DataType::DT_INT32) {
            tmpWeight->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
            auto viewShapeDim = viewShape.GetDimNum();
            viewShape[viewShapeDim - 1] *= PER_INT4_IN_U32;
            auto storageShapeDim = storageShape.GetDimNum();
            // The following line adjusts the storage shape because we have a few
            // checks that put some requirements on the storage shape and the view shape,
            // e.g., the function 'CheckWeightNzStorageShape'.
            //
            // HACK: Right now we hard code the value of the last dim as
            // 'NZ_STORAGE_LAST_DIM * PER_INT4_IN_U8' (which is 64), instead of
            // 'storageShape[storageShapeDim - 1] *= PER_INT4_IN_U32' because as of
            // torch_npu 7.1.0, the function 'npu_convert_weight_to_int4pack' does
            // not support 3D tensor. So in the ascend-vllm project, the following
            // procedures are used to generate the int4 weight tensor in NZ format:
            //
            //   - Pack two int4 of (E, K, N/2) as an int8 (E, K, N/2)
            //   - 'npu_format_cast' the int8 tensor to NZ format ('npu_format_cast'
            //     gives wrong results for int32 here because C0 is 8)
            //   - '.view(torch.int32)' to change the view shape to (E, K, N/8)
            //     and the data type to int32.
            //
            // Therefore, the storage shape of the final tensor does not necessarily
            // matches the data type, int32. That is why we hard code the value here.
            // Fortunately, this is not so bad because the existing checks will verify
            // the new storage shape here to some extent. For example, 'CheckWeightNzStorageShape'
            // ensures that the two shapes still match.
            //
            // In the future, when we settle on a canonical way to handle NZ int4 tensors
            // in torch_npu, we should update the following line accordingly.
            storageShape[storageShapeDim - 1] = NZ_STORAGE_LAST_DIM * PER_INT4_IN_U8;
            tmpWeight->SetViewShape(viewShape);
            tmpWeight->SetDataType(DataType::DT_INT4);
        }
        tmpWeight->SetStorageShape(storageShape);

        if (tmpWeight->GetDataType() == DataType::DT_INT4 && pertokenScaleOptional == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                    "GroupedMatmulFinalizeRoutingWeightNz does not support nullptr for pertokenScale.");
            return ACLNN_ERR_PARAM_NULLPTR;
        }

        CheckSupportSceneParams sceneParams{x1,    tmpWeight, scale, pertokenScaleOptional, groupList, sharedInput,
                                            logit, rowIndex,  dtype};
        auto ret0 = CheckSupportScene(sceneParams, transposeX1, transposeX2);
        CHECK_RET(ret0 == ACLNN_SUCCESS, ret0);
    }
    GroupedMatmulParams params = GroupedMatmulParamsBuilder::Create(x1, tmpWeight, out)
        .SetScale(scale)
        .SetBias(bias)
        .SetPertokenScale(pertokenScaleOptional)
        .SetGroupList(groupList)
        .SetShareInput(sharedInput)
        .SetLogit(logit)
        .SetRowIndex(rowIndex).SetOffset(offsetOptional)
        .SetTuningConfig(tuningConfigOptional)
        .SetNumbers(sharedInputWeight, sharedInputOffset, groupListType)
        .SetTranspose(transposeX1, transposeX2)
        .Build();
    auto ret = aclnnGroupedMatmulFinalizeRoutingGetWorkspaceSizeCommonProcess(params, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNzV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulFinalizeRoutingWeightNzV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

/* weight ND interface*/
aclnnStatus aclnnGroupedMatmulFinalizeRoutingGetWorkspaceSize(const aclTensor *x1, aclTensor *x2,
    const aclTensor *scale, const aclTensor *bias, const aclTensor *pertokenScaleOptional, const aclTensor *groupList,
    const aclTensor *sharedInput, const aclTensor *logit, const aclTensor *rowIndex, int64_t dtype,
    float sharedInputWeight, int64_t sharedInputOffset, bool transposeX1, bool transposeX2, int64_t groupListType,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnGroupedMatmulFinalizeRouting,
        DFX_IN(x1, x2, scale, bias, pertokenScaleOptional, groupList, sharedInput, logit, rowIndex, dtype,
        sharedInputWeight, sharedInputOffset, transposeX1, transposeX2, groupListType), DFX_OUT(out));
    auto scene = x1 != nullptr && x2 != nullptr && scale != nullptr && pertokenScaleOptional != nullptr &&
        groupList != nullptr && logit != nullptr && rowIndex != nullptr && bias != nullptr;
    if (!scene) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "GroupedMatmulFinalizeRouting weightNd do not support input nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    int64_t viewDimNum = x2->GetViewShape().GetDimNum();
    if (dtype != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GroupedMatmulFinalizeRouting weightNd dtype must be 0, but is %lld.", dtype);
        return ACLNN_ERR_PARAM_INVALID;
    } else if (viewDimNum < MIN_DIM_NUM_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GroupedMatmulFinalizeRouting weightNd x2's view dimNum should greater than 1, but is %lld.", viewDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    } else if (!(transposeX1 == false && transposeX2 == false)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GroupedMatmulFinalizeRouting weightNd transpose should be false");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // unpack int32 to int4
    auto tmpWeight = x2;
    if (tmpWeight->GetDataType() == DataType::DT_INT32) {
        op::Shape weightShape = tmpWeight->GetViewShape();
        auto viewShapeDim = weightShape.GetDimNum();
        weightShape[viewShapeDim - 1] = weightShape[viewShapeDim - 1] * PER_INT4_IN_U32;
        tmpWeight->SetViewShape(weightShape);
        tmpWeight->SetDataType(DataType::DT_INT4);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GroupedMatmulFinalizeRouting weightNd weight type should be INT_32, but now is %s",
            op::ToString(tmpWeight->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // aclnnGroupedMatmulFinalizeRoutingND
    auto uniqueExecutorND = CREATE_EXECUTOR();
    const aclTensor* unusedND = nullptr;
    const aclIntArray* unusedTuningConfig = nullptr;
    GroupedMatmulParams params = GroupedMatmulParamsBuilder::Create(x1, x2, out)
        .SetScale(scale).SetBias(bias)
        .SetPertokenScale(pertokenScaleOptional).SetGroupList(groupList)
        .SetShareInput(sharedInput).SetLogit(logit)
        .SetRowIndex(rowIndex).SetOffset(unusedND)
        .SetTuningConfig(unusedTuningConfig)
        .SetNumbers(sharedInputWeight, sharedInputOffset, groupListType)
        .SetTranspose(transposeX1, transposeX2).Build();
    auto ret = aclnnGroupedMatmulFinalizeRoutingGetWorkspaceSizeCommonProcess(params, uniqueExecutorND.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutorND->GetWorkspaceSize();
    uniqueExecutorND.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulFinalizeRouting(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulFinalizeRouting);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnGroupedMatmulFinalizeRoutingV2GetWorkspaceSize(const aclTensor *x1, aclTensor *x2,
    const aclTensor *scaleOptional, const aclTensor *biasOptional, const aclTensor *offsetOptional, const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional, const aclTensor *pertokenScaleOptional, const aclTensor *groupListOptional,
    const aclTensor *sharedInputOptional, const aclTensor *logitOptional, const aclTensor *rowIndexOptional, int64_t dtype,
    float sharedInputWeight, int64_t sharedInputOffset, bool transposeX1, bool transposeX2, int64_t groupListType,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnGroupedMatmulFinalizeRoutingV2,
        DFX_IN(x1, x2, scaleOptional, biasOptional, pertokenScaleOptional, groupListOptional, sharedInputOptional, logitOptional, rowIndexOptional, dtype,
        sharedInputWeight, sharedInputOffset, transposeX1, transposeX2, groupListType), DFX_OUT(out));
    auto scene = x1 != nullptr && x2 != nullptr && scaleOptional != nullptr && pertokenScaleOptional != nullptr &&
        groupListOptional != nullptr && logitOptional != nullptr && rowIndexOptional != nullptr && biasOptional != nullptr
        && antiquantScaleOptional == nullptr && antiquantOffsetOptional == nullptr;
    if (!scene) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "GroupedMatmulFinalizeRoutingV2 weightNd do not support input nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    int64_t viewDimNum = x2->GetViewShape().GetDimNum();
    if (dtype != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GroupedMatmulFinalizeRoutingV2 weightNd dtype must be 0, but is %lld.", dtype);
        return ACLNN_ERR_PARAM_INVALID;
    } else if (viewDimNum < MIN_DIM_NUM_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GroupedMatmulFinalizeRoutingV2 weightNd x2's view dimNum should greater than 1, but is %lld.", viewDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    } else if (!(transposeX1 == false && transposeX2 == false)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GroupedMatmulFinalizeRoutingV2 weightNd transpose should be false");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // unpack int32 to int4
    auto tmpWeightV2 = x2;
    if (tmpWeightV2->GetDataType() == DataType::DT_INT32) {
        op::Shape weightShapeV2 = tmpWeightV2->GetViewShape();
        auto viewShapeDimV2 = weightShapeV2.GetDimNum();
        weightShapeV2[viewShapeDimV2 - 1] = weightShapeV2[viewShapeDimV2 - 1] * PER_INT4_IN_U32;
        tmpWeightV2->SetViewShape(weightShapeV2);
        tmpWeightV2->SetDataType(DataType::DT_INT4);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GroupedMatmulFinalizeRoutingV2 weightNd weight type should be INT_32, but now is %s",
            op::ToString(tmpWeightV2->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    const aclIntArray* unusedTuningConfig = nullptr;
    GroupedMatmulParams params = GroupedMatmulParamsBuilder::Create(x1, x2, out)
        .SetScale(scaleOptional).SetBias(biasOptional)
        .SetPertokenScale(pertokenScaleOptional).SetGroupList(groupListOptional)
        .SetShareInput(sharedInputOptional).SetLogit(logitOptional)
        .SetRowIndex(rowIndexOptional).SetOffset(offsetOptional)
        .SetTuningConfig(unusedTuningConfig)
        .SetNumbers(sharedInputWeight, sharedInputOffset, groupListType)
        .SetTranspose(transposeX1, transposeX2).Build();
    auto ret1 = aclnnGroupedMatmulFinalizeRoutingGetWorkspaceSizeCommonProcess(params, uniqueExecutor.get());
    CHECK_RET(ret1 == ACLNN_SUCCESS, ret1);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulFinalizeRoutingV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulFinalizeRoutingV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

static inline aclnnStatus CheckSupportSceneforV3(const aclTensor *x1, aclTensor *x2, const aclTensor *scaleOptional,
                                                 const aclTensor *groupListOptional,
                                                 const aclTensor *pertokenScaleOptional, const aclTensor *logitOptional,
                                                 const aclTensor *rowIndexOptional,
                                                 const aclTensor *antiquantScaleOptional,
                                                 const aclTensor *antiquantOffsetOptional)
{
    // 支持sharedInput输入为空, 不支持logit为空
    auto scene = x1 != nullptr && x2 != nullptr && scaleOptional != nullptr && groupListOptional != nullptr &&
                 pertokenScaleOptional != nullptr && logitOptional != nullptr && rowIndexOptional != nullptr &&
                 antiquantScaleOptional == nullptr && antiquantOffsetOptional == nullptr;

    if (scene) {
        return ACLNN_SUCCESS;
    }

    std::string errMsg = "aclnnGroupedMatmulFinalizeRoutingV3 invalid input paramater configuration.";

    if (x1 == nullptr) {
        errMsg += " Required parameter 'x1' is null.";
    }
    if (x2 == nullptr) {
        errMsg += " Required parameter 'x2' is null.";
    }
    if (scaleOptional == nullptr) {
        errMsg += " Required parameter 'scale' is null.";
    }
    if (groupListOptional == nullptr) {
        errMsg += " Required parameter 'groupList' is null.";
    }
    if (pertokenScaleOptional == nullptr) {
        errMsg += " Required parameter 'pertokenScale' is null.";
    }
    if (logitOptional == nullptr) {
        errMsg += " Required parameter 'logit' is null.";
    }
    if (rowIndexOptional == nullptr) {
        errMsg += " Required parameter 'rowIndex' is null.";
    }
    if (antiquantScaleOptional != nullptr) {
        errMsg += " Required parameter 'antiquantScale' must be null.";
    }
    if (antiquantOffsetOptional != nullptr) {
        errMsg += " Required parameter 'antiquantOffset' must be null.";
    }
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s", errMsg.c_str());
    return ACLNN_ERR_PARAM_NULLPTR;
}
aclnnStatus aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize(const aclTensor *x1, aclTensor *x2,
    const aclTensor *scaleOptional, const aclTensor *biasOptional, const aclTensor *offsetOptional, const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional, const aclTensor *pertokenScaleOptional, const aclTensor *groupListOptional,
    const aclTensor *sharedInputOptional, const aclTensor *logitOptional, const aclTensor *rowIndexOptional, int64_t dtype,
    float sharedInputWeight, int64_t sharedInputOffset, bool transposeX1, bool transposeX2, int64_t groupListType, const aclIntArray *tuningConfigOptional,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnGroupedMatmulFinalizeRoutingV3,
        DFX_IN(x1, x2, scaleOptional, biasOptional, pertokenScaleOptional, groupListOptional, sharedInputOptional, logitOptional, rowIndexOptional, dtype,
        sharedInputWeight, sharedInputOffset, transposeX1, transposeX2, groupListType), DFX_OUT(out));
    
    auto ret0 = CheckSupportSceneforV3(x1, x2, scaleOptional, groupListOptional, pertokenScaleOptional, logitOptional,
                                       rowIndexOptional, antiquantScaleOptional, antiquantOffsetOptional);
    CHECK_RET(ret0 == ACLNN_SUCCESS, ret0);

    int64_t viewDimNum = x2->GetViewShape().GetDimNum();
    if (dtype != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnGroupedMatmulFinalizeRoutingV3 weightNd dtype must be 0, but is %lld.", dtype);
        return ACLNN_ERR_PARAM_INVALID;
    } else if (viewDimNum < MIN_DIM_NUM_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnGroupedMatmulFinalizeRoutingV3 weightNd x2's view dimNum should greater than 1, but is %lld.", viewDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    } else if (!((transposeX1 == false && transposeX2 == false) ||
                 (transposeX1 == false && transposeX2 == true &&
                  op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnGroupedMatmulFinalizeRoutingV3: valid transpose config: transposeX1=%s,transposeX2=%s",
                transposeX1 ? "true" : "false", transposeX2 ? "true" : "false");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // unpack int32 to int4
    auto tmpWeightV3 = x2;
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        if (tmpWeightV3->GetDataType() == DataType::DT_INT32) {
            op::Shape weightShapeV3 = tmpWeightV3->GetViewShape();
            auto viewShapeDimV2 = weightShapeV3.GetDimNum();
            weightShapeV3[viewShapeDimV2 - 1] = weightShapeV3[viewShapeDimV2 - 1] * PER_INT4_IN_U32;
            tmpWeightV3->SetViewShape(weightShapeV3);
            tmpWeightV3->SetDataType(DataType::DT_INT4);
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "aclnnGroupedMatmulFinalizeRoutingV3 weightNd weight type should be INT_32, but now "
                    "is %s",
                    op::ToString(tmpWeightV3->GetDataType()).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    bool isMXValid = CheckType(x1->GetDataType(), X_WEIGHT_TYPE_SUPPORT_LIST_MX) &&
                     CheckType(tmpWeightV3->GetDataType(), X_WEIGHT_TYPE_SUPPORT_LIST_MX);
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510 && !isMXValid) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnGroupedMatmulFinalizeRoutingV3 weightNd: Invalid dtype combination."
                "Expected: x1 and x2 both in [FLOAT8_E5M2, FLOAT8_E4M3FN, "
                "FLOAT4_E2M1] for mx."
                "But got x1=%s, x2=%s",
                op::ToString(x1->GetDataType()).GetString(), op::ToString(tmpWeightV3->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
        auto uniqueExecutor = CREATE_EXECUTOR();
    GroupedMatmulParams params = GroupedMatmulParamsBuilder::Create(x1, x2, out)
        .SetScale(scaleOptional).SetBias(biasOptional)
        .SetPertokenScale(pertokenScaleOptional).SetGroupList(groupListOptional)
        .SetShareInput(sharedInputOptional).SetLogit(logitOptional)
        .SetRowIndex(rowIndexOptional).SetOffset(offsetOptional)
        .SetTuningConfig(tuningConfigOptional)
        .SetNumbers(sharedInputWeight, sharedInputOffset, groupListType)
        .SetTranspose(transposeX1, transposeX2).Build();
    auto ret = aclnnGroupedMatmulFinalizeRoutingGetWorkspaceSizeCommonProcess(params, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulFinalizeRoutingV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulFinalizeRoutingV3);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif