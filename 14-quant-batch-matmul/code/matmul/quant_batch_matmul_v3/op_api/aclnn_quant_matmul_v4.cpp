/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_quant_matmul_v4.h"
#include <dlfcn.h>
#include "aclnn_quant_matmul_v3.h"
#include "aclnn_quant_matmul_weight_nz.h"
#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "quant_matmul_v3.h"
#include "matmul/quant_batch_matmul_v4/op_host/op_api/quant_matmul_v4.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "util/math_util.h"
#include "quant_matmul_checker.h"

using namespace op;
using Ops::NN::SwapLastTwoDimValue;
using Ops::NN::IsTransposeLastTwoDims;
using Ops::Base::CeilDiv;
using TupleTensor = std::tuple<const aclTensor *, const aclTensor *, const aclTensor *>;
using TupleOptional = std::tuple<const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
                                 const aclTensor *, const int64_t &>;
using TupleInput = std::tuple<const aclTensor *, const aclTensor *>;
using TupleQuant = std::tuple<const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
                              const aclTensor *, const aclTensor *, const aclTensor *, const int64_t &, const int64_t &>;
using TupleAttr = std::tuple<bool, bool>;

namespace {
static constexpr int INDEX_X1_IN_MANDTORY_TUPLE = 0;
static constexpr int INDEX_X2_IN_MANDTORY_TUPLE = 1;
static constexpr int INDEX_SCALE_IN_MANDTORY_TUPLE = 2;
static constexpr int INDEX_OFFSET_IN_OPTIONAL_TUPLE = 0;
static constexpr int INDEX_PERTOKEN_IN_OPTIONAL_TUPLE = 1;
static constexpr int INDEX_BIAS_IN_OPTIONAL_TUPLE = 2;
static constexpr int INDEX_Y_SCALE_IN_OPTIONAL_TUPLE = 3;
static constexpr int INDEX_Y_OFFSET_IN_OPTIONAL_TUPLE = 4;
static constexpr int INDEX_GROUP_SIZE_IN_OPTIONAL_TUPLE = 5;
static constexpr int INDEX_OUT_IN_TUPLE = 2;
static constexpr int INDEX_ISA4W4_IN_BOOL_TUPLE = 2;
static constexpr size_t LAST_SECOND_DIM_INDEX = 2;

static const int MIN_DIM_NUM_ND = 2;
static const int MAX_DIM_NUM_ND = 6;
static const int MIN_DIM_NUM_NZ = 4;
static const int MAX_DIM_NUM_NZ = 8;
static const int PENULTIMATE_DIM = 2;
static const int NZ_K1_INDEX = 3;
static const int NZ_K1_INDEX_TRANS = 4;
static const int NZ_STORAGE_PENULTIMATE_DIM = 16;
static const int NZ_STORAGE_LAST_DIM = 32;
static const int64_t NZ_K0_VALUE_BMM_BLOCK_NUM = 16;
static const int64_t NZ_K0_VALUE_INT32_TRANS = 8;
static const int64_t NZ_K0_VALUE_INT8_TRANS = 32;
static const int64_t NZ_K0_VALUE_INT4_TRANS = 64;
static constexpr int64_t OUTPUT_INFER_FAIL = -1L;
static const int64_t LAST_AXIS_LIMIT = 65535;
static const int X2_FIXED_DIM_NUM_A4W4 = 2;
static const int64_t INT4_NUMS_IN_INT8 = 2;
static const int64_t INT4_NUMS_IN_INT32 = 8;
static const int64_t INNER_SIZE_MULTIPLE = 64;
static const int64_t K_VALUE = 3696;
static const int64_t N_VALUE = 8192;
static const int64_t M_RANGE1_LEFT = 128;
static const int64_t M_RANGE1_RIGHT = 512;
static const int32_t CORE_NUM_20 = 20;
static const int64_t SUPPORTED_GROUP_SIZE = 32;
static constexpr uint64_t B4_PER_B32 = 8UL;
static const int64_t SUPPORTED_K_ALIGN_NUM = 64;
static const int64_t SUPPORTED_N_ALIGN_NUM = 64;
static const size_t MAX_DIM_VALUE = 2;
static const size_t MX_SCALE_DIM_VALUE = 3;
static const uint64_t GROUP_M_OFFSET = 32;
static const uint64_t GROUP_N_OFFSET = 16;
static const uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
static const size_t MX_SCALE_MAX_DIM = 3;
static const size_t MX_SCALE_DIM_NUM = 3;
static const int64_t MAX_SHAPE_SIZE_A8W4_INT = 29576;
static const int64_t PPMATMUL_PRIORITY_M = 1024;
static const int64_t NO_BATCH_DIM_SUM = 2;

static const std::initializer_list<op::DataType> IN_TYPE_SUPPORT_LIST = {op::DataType::DT_INT4,
                                                                         op::DataType::DT_INT8};
static const std::initializer_list<op::DataType> INT4_TYPE_SUPPORT_LIST = {op::DataType::DT_INT4,
                                                                         op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> OUT_TYPE_SUPPORT_LIST = {op::DataType::DT_INT8,
                                                                          op::DataType::DT_FLOAT16,
                                                                          op::DataType::DT_BF16,
                                                                          op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> SCALE_TYPE_SUPPORT_LIST = {op::DataType::DT_UINT64,
                                                                            op::DataType::DT_BF16,
                                                                            op::DataType::DT_INT64,
                                                                            op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> BIAS_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32,
                                                                           op::DataType::DT_BF16,
                                                                           op::DataType::DT_FLOAT16,
                                                                           op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> Y_SCALE_SUPPORT_LIST = {op::DataType::DT_UINT64};

static inline bool isA8W4Float(const aclTensor* x1, const aclTensor* x2)
{
    return x1->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN &&
           (x2->GetDataType() == op::DataType::DT_FLOAT || x2->GetDataType() == op::DataType::DT_FLOAT4_E2M1);
}

static inline bool isA8W4Int(const aclTensor* x1, const aclTensor* x2)
{
    return x1->GetDataType() == op::DataType::DT_INT8 &&
           (x2->GetDataType() == op::DataType::DT_INT4 || x2->GetDataType() == op::DataType::DT_INT32);
}

static inline bool isMxfp8Nz(const aclTensor* x1, const aclTensor* x2, const aclTensor* scale)
{
    return x1->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN && x2->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN
     && scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0;
}

static inline bool isA8W4Msd(const aclTensor* x1, const aclTensor* x2, const aclTensor* scale,
    const aclTensor* pertokenScale)
{
    if (x1->GetDataType() != op::DataType::DT_INT8) {
        return false;
    }

    if (std::find(INT4_TYPE_SUPPORT_LIST.begin(), INT4_TYPE_SUPPORT_LIST.end(),
        x2->GetDataType()) == INT4_TYPE_SUPPORT_LIST.end()) {
        return false;
    }

    if (scale->GetDataType() != op::DataType::DT_UINT64) {
        return false;
    }

    if (pertokenScale == nullptr || pertokenScale->GetDataType() != op::DataType::DT_FLOAT) {
        return false;
    }

    return true;
}

static inline bool CheckNotNull(TupleTensor mandatoryTensors, const aclTensor *out) {
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(scale, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValidOnOnlyL0c2ub(TupleTensor mandatoryTensors, TupleOptional optionalTensors,
                                               const aclTensor *out)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);

    if (x1->GetDataType() != op::DataType::DT_INT8 || x2->GetDataType() != op::DataType::DT_INT8) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x1 and x2 dtype should be INT8, actual dtype are %s and %s",
                op::ToString(x1->GetDataType()).GetString(), op::ToString(x2->GetDataType()).GetString());
        return false;
    }
    if (!(scale->GetDataType() == op::DataType::DT_UINT64 || scale->GetDataType() == op::DataType::DT_INT64)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Scale dtype should be UINT64 or INT64, actual dtype is %s",
                op::ToString(scale->GetDataType()).GetString());
        return false;
    }
    if (bias != nullptr && bias->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias dtype should be INT32, actual dtype is %s",
                op::ToString(bias->GetDataType()).GetString());
        return false;
    }
    if (pertokenScaleOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "PertokenScaleOptional should be null");
        return false;
    }
    if (!(out->GetDataType() == op::DataType::DT_INT8 || out->GetDataType() == op::DataType::DT_FLOAT16)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output dtype should be INT8 or FLOAT16, actual dtype is %s",
                op::ToString(out->GetDataType()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckDtypeValidOnOnlyL0c2ubPertoken(TupleTensor mandatoryTensors, TupleOptional optionalTensors,
                                               const aclTensor *out)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    if (x1->GetDataType() != op::DataType::DT_INT8 || x2->GetDataType() != op::DataType::DT_INT8) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x1 and x2 dtype should be INT8, actual dtype are %s and %s",
                op::ToString(x1->GetDataType()).GetString(), op::ToString(x2->GetDataType()).GetString());
        return false;
    }
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    if (scale->GetDataType() != op::DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Scale dtype should be FLOAT, actual dtype is %s",
                op::ToString(scale->GetDataType()).GetString());
        return false;
    }
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    if (bias != nullptr && bias->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias dtype should be INT32, actual dtype is %s",
                op::ToString(bias->GetDataType()).GetString());
        return false;
    }
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    if (pertokenScaleOptional->GetDataType() != op::DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "PertokenScaleOptional should be FLOAT, actual dtype is %s",
                op::ToString(scale->GetDataType()).GetString());
        return false;
    }
    if (out->GetDataType() != op::DataType::DT_FLOAT16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output dtype should be FLOAT16, actual dtype is %s",
                op::ToString(out->GetDataType()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckDtypeValidOnOnlyL0c2outForA4W4(TupleTensor mandatoryTensors, TupleOptional optionalTensors,
                                                       const aclTensor *out)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    if (isA8W4Msd(x1, x2, scale, pertokenScaleOptional)) {
        return true;
    }

    if (x1->GetDataType() != op::DataType::DT_INT4 || x2->GetDataType() != op::DataType::DT_INT4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Iutput x1 x2 dtype should be INT4 in a4w4 scenario, actual dtype is %s %s.",
                op::ToString(x1->GetDataType()).GetString(), op::ToString(x2->GetDataType()).GetString());
        return false;
    }
    if (pertokenScaleOptional == nullptr) {
        if (scale->GetDataType() != op::DataType::DT_UINT64 && scale->GetDataType() != op::DataType::DT_INT64) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Scale dtype should be UINT64 or INT64 in a4w4 without pertoken scale scenario, actual dtype is %s.",
                    op::ToString(scale->GetDataType()).GetString());
            return false;
        }
        if (bias != nullptr && bias->GetDataType() != op::DataType::DT_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias dtype should be INT32 in a4w4 without pertoken scale scenario, actual dtype is %s",
                    op::ToString(bias->GetDataType()).GetString());
            return false;
        }
    }
    if (out->GetDataType() != op::DataType::DT_FLOAT16 && out->GetDataType() != op::DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output dtype should be FLOAT16 or BF16 in a4w4 scenario, actual dtype is %s.",
                op::ToString(out->GetDataType()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckDtypeValidOnOnlyL0c2outForPertoken(TupleTensor mandatoryTensors, TupleOptional optionalTensors,
                                                           const aclTensor *out)
{
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    if (pertokenScaleOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(pertokenScaleOptional, op::DataType::DT_FLOAT, return false);
        if (bias != nullptr && bias->GetDataType() == op::DataType::DT_FLOAT16 &&
            out->GetDataType() != op::DataType::DT_FLOAT16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When pertokenScaleOptional is not nullptr, bias dtype is FLOAT16, out dtype should be FLOAT16, \
actual dtype is %s.", op::ToString(out->GetDataType()).GetString());
            return false;
        }
        if (out->GetDataType() != op::DataType::DT_FLOAT16 && out->GetDataType() != op::DataType::DT_BF16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When pertokenScaleOptional is not nullptr, out dtype should be FLOAT16 or BF16, actual dtype is %s.",
                    op::ToString(out->GetDataType()).GetString());
            return false;
        }
        if (out->GetDataType() == op::DataType::DT_FLOAT16 && (scale->GetDataType() != op::DataType::DT_FLOAT
            && scale->GetDataType() != op::DataType::DT_UINT64)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When pertokenScaleOptional is not nullptr, out dtype is FLOAT16, scale dtype should be FLOAT \
or UINT64, actual dtype is %s.", op::ToString(scale->GetDataType()).GetString());
            return false;
        }
    } else {
        if (bias != nullptr && bias->GetDataType() == op::DataType::DT_FLOAT16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When pertokenScaleOptional is not nullptr, bias dtype should not be FLOAT16.");
            return false;
        }
        if (bias != nullptr && bias->GetDataType() == op::DataType::DT_FLOAT &&
            out->GetDataType() != op::DataType::DT_BF16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When pertokenScaleOptional is nullptr and bias dtype is FLOAT, out dtype should be BF16, \
actual dtype is %s.", op::ToString(out->GetDataType()).GetString());
            return false;
        }
        if ((out->GetDataType() == op::DataType::DT_INT8 || out->GetDataType() == op::DataType::DT_FLOAT16) &&
            scale->GetDataType() == op::DataType::DT_FLOAT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When pertokenScaleOptional is nullptr and output dtype is INT8 or FLOAT16, \
scale dtype should not be FLOAT.");
            return false;
        }
    }
    return true;
}

static inline bool CheckDtypeValidInBf16OutScenario(TupleTensor mandatoryTensors)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    bool isA8W4 = isA8W4Int(x1, x2) || isA8W4Float(x1, x2);
    bool passScaleCheckA8W4 = isA8W4 && !(scale->GetDataType() == op::DataType::DT_BF16 ||
        scale->GetDataType() == op::DataType::DT_FLOAT || scale->GetDataType() == op::DataType::DT_UINT64);
    if (passScaleCheckA8W4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When out dtype is BF16, scale dtype should be BF16/FLOAT/UINT64 in A8W4 scenario, actual dtype is %s.",
                op::ToString(scale->GetDataType()).GetString());
        return false;
    }
    bool passScaleCheckOthers = !isA8W4 && !(scale->GetDataType() == op::DataType::DT_BF16 || scale->GetDataType() == op::DataType::DT_FLOAT);
    if (passScaleCheckOthers) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When out dtype is BF16, scale dtype should be BF16/FLOAT in A4W4/A8W8 scenario, actual dtype is %s.",
                op::ToString(scale->GetDataType()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckDtypeValidOnOnlyL0c2outForUnclassified(TupleTensor mandatoryTensors,
                                                               TupleOptional optionalTensors, const aclTensor *out)
{
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    if (bias != nullptr && bias->GetDataType() == op::DataType::DT_BF16 &&
        out->GetDataType() != op::DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When bias dtype is BF16, out dtype should be BF16, actual dtype is %s.",
                op::ToString(out->GetDataType()).GetString());
        return false;
    }
    if (scale->GetDataType() == op::DataType::DT_BF16 && out->GetDataType() != op::DataType::DT_BF16 &&
        out->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When scale dtype is BF16, out dtype should be BF16 or INT32, actual dtype is %s",
                op::ToString(out->GetDataType()).GetString());
        return false;
    }
    if (out->GetDataType() == op::DataType::DT_BF16 && !CheckDtypeValidInBf16OutScenario(mandatoryTensors)) {
        return false;
    }
    if (out->GetDataType() == op::DataType::DT_INT8 &&
        !(scale->GetDataType() == op::DataType::DT_INT64 || scale->GetDataType() == op::DataType::DT_UINT64)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When out dtype is INT8, scale dtype should be INT64 or UINT64, actual dtype is %s.",
                op::ToString(scale->GetDataType()).GetString());
        return false;
    }
    if (out->GetDataType() == op::DataType::DT_INT32 &&
        !(scale->GetDataType() == op::DataType::DT_FLOAT || scale->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When out dtype is INT32, scale dtype should be FLOAT or BF16, actual dtype is %s.",
                op::ToString(scale->GetDataType()).GetString());
        return false;
    }
    if (out->GetDataType() == op::DataType::DT_INT32 && bias != nullptr &&
        bias->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When out dtype is INT32, bias dtype should be INT32, actual dtype is %s.",
                op::ToString(bias->GetDataType()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckDtypeValidOnOnlyL0c2out(TupleTensor mandatoryTensors, TupleOptional optionalTensors,
                                                const aclTensor *out, bool isA4W4)
{
    // 对A4W4场景/非A4W4场景进行校验
    if (isA4W4 && !CheckDtypeValidOnOnlyL0c2outForA4W4(mandatoryTensors, optionalTensors, out)) {
        return false;
    }
    if (!CheckDtypeValidOnOnlyL0c2outForUnclassified(mandatoryTensors, optionalTensors, out)) {
        return false;
    }
    if (!CheckDtypeValidOnOnlyL0c2outForPertoken(mandatoryTensors, optionalTensors, out)) {
        return false;
    }
    return true;
}

static inline bool CheckDtypeValid(TupleTensor mandatoryTensors, TupleOptional optionalTensors, const aclTensor *out,
                                   bool isA4W4)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto pertokenScale = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto offset = std::get<INDEX_OFFSET_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, IN_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, IN_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(scale, SCALE_TYPE_SUPPORT_LIST, return false);
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, BIAS_TYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_TYPE_SUPPORT_LIST, return false);

    // 无芯片差异的公共校验
    if (!isA8W4Msd(x1, x2, scale, pertokenScale) && x1->GetDataType() != x2->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In non-A8W4 case, x1 and x2 dtype should be same, \
        actual x1 dtype is %s and x2 dtype is %s.",
                op::ToString(x1->GetDataType()).GetString(), op::ToString(x2->GetDataType()).GetString());
        return false;
    }

    if (offset != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(offset, op::DataType::DT_FLOAT, return false);
        // offset only exists if out is int8
        if (out->GetDataType() != op::DataType::DT_INT8) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Offset only exists when out dtype is INT8, actual dtype is %s.",
                    op::ToString(out->GetDataType()).GetString());
            return false;
        }
    }
    // 区分芯片校验
    if ((GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) &&
        ((pertokenScale == nullptr && !CheckDtypeValidOnOnlyL0c2ub(mandatoryTensors, optionalTensors, out)) ||
         (pertokenScale != nullptr && !CheckDtypeValidOnOnlyL0c2ubPertoken(mandatoryTensors, optionalTensors, out)))) {
        return false;
    } else if ((GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
                GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) &&
               !CheckDtypeValidOnOnlyL0c2out(mandatoryTensors, optionalTensors, out, isA4W4)) {
        return false;
    }
    return true;
}

static inline bool CheckFormatInt4(const aclTensor *x1, const aclTensor *x2) {
    if (x1->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 only support ND format in a4w4 scenario, but now is %s.",
            op::ToString(x1->GetStorageFormat()).GetString());
        return false;
    }
    if (x2->GetStorageFormat() != op::Format::FORMAT_ND && x2->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2 only support ND/NZ in a4w4 scenario, but now is %s.",
            op::ToString(x2->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckFormat(TupleTensor mandatoryTensors, TupleOptional optionalTensors, bool isA4W4) {
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto offset = std::get<INDEX_OFFSET_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto x1StorageFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(x1->GetStorageFormat()));
    auto x2StorageFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat()));
    bool formatValid = ((x1StorageFormat == op::Format::FORMAT_ND) ||
                        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P &&
                         pertokenScaleOptional != nullptr && x1StorageFormat == op::Format::FORMAT_FRACTAL_NZ)) &&
                       (x2StorageFormat == op::Format::FORMAT_ND || x2StorageFormat == op::Format::FORMAT_FRACTAL_NZ) &&
                       scale->GetStorageFormat() == op::Format::FORMAT_ND;
    if (offset != nullptr) {
        formatValid = formatValid && offset->GetStorageFormat() == op::Format::FORMAT_ND;
    }
    if (pertokenScaleOptional != nullptr) {
        formatValid = formatValid && pertokenScaleOptional->GetStorageFormat() == op::Format::FORMAT_ND;
    }
    if (bias != nullptr) {
        formatValid = formatValid && bias->GetStorageFormat() == op::Format::FORMAT_ND;
    }
    if (isA4W4) {
        formatValid = formatValid && CheckFormatInt4(x1, x2);
    }
    return formatValid;
}

static inline bool CheckDimRange(const aclTensor *x1, const aclTensor *x2, const aclTensor *scale,
                                 const aclTensor *out) {
    auto x2StorageFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat()));
    int64_t x2MaxDimNum = x2StorageFormat == op::Format::FORMAT_FRACTAL_NZ ? MAX_DIM_NUM_NZ : MAX_DIM_NUM_ND;
    int64_t x2MinDimNum = x2StorageFormat == op::Format::FORMAT_FRACTAL_NZ ? MIN_DIM_NUM_NZ : MIN_DIM_NUM_ND;
    int64_t x2DimNum = x2->GetStorageShape().GetDimNum();
    CHECK_RET(x2DimNum >= x2MinDimNum && x2DimNum <= x2MaxDimNum, false);
    OP_CHECK_MIN_DIM(x1, MIN_DIM_NUM_ND, return false);
    OP_CHECK_MIN_DIM(out, MIN_DIM_NUM_ND, return false);
    OP_CHECK_MAX_DIM(x1, MAX_DIM_NUM_ND, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_NUM_ND, return false);
    size_t expectScaleDim = isMxfp8Nz(x1, x2, scale) ? MX_SCALE_DIM_NUM : 1;
    OP_CHECK_WRONG_DIMENSION(scale, expectScaleDim, return false);
    OP_LOGD("QuantMatmul check dim-num range success");
    return true;
}

static int64_t InferOutputShape(const aclTensor *x1, const aclTensor *x2, std::vector<int64_t> &batchRecord) {
    int64_t inferedOutbatchValue = 1;
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    auto outDimNum = std::max(x1DimNum, x2DimNum);
    auto &longShapeTensor = x1DimNum > x2DimNum ? x1 : x2;
    auto &shortShapeTensor = x1DimNum > x2DimNum ? x2 : x1;
    size_t vaildOffset = outDimNum - std::min(x1DimNum, x2DimNum);
    for (size_t i = 0; i < outDimNum - PENULTIMATE_DIM; i++) {
        auto shortDimValue = i < vaildOffset ? 1 : shortShapeTensor->GetViewShape().GetDim(i - vaildOffset);
        auto longDimValue = longShapeTensor->GetViewShape().GetDim(i);
        if (shortDimValue > 1 && longDimValue > 1 && shortDimValue != longDimValue) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Current short dim value %ld and long dim value %ld are not supported for broadcasting.",
                    shortDimValue, longDimValue);
            return OUTPUT_INFER_FAIL;
        }
        int64_t curBatchValue = static_cast<int64_t>(std::max(shortDimValue, longDimValue));
        inferedOutbatchValue = inferedOutbatchValue * curBatchValue;
        batchRecord.push_back(curBatchValue);
    }
    return inferedOutbatchValue;
}

static inline bool CheckBiasShape(const aclTensor *bias, int64_t x2NDim, const std::vector<int64_t> batchRecord,
                                  int64_t inferedOutbatchValue) {
    auto biasDimNum = bias->GetViewShape().GetDimNum();
    // 3 is bias with batch dim-num
    if (biasDimNum != 3 && biasDimNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Bias dim-num should equal 3 or 1, but it is %zu.", biasDimNum);
        return false;
    }
    auto biasFirstDim = bias->GetViewShape().GetDim(0);
    if (biasDimNum == 1) {
        OP_CHECK(biasFirstDim == x2NDim,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias 1st dim should be equal to x2 n dim %ld, but it is %ld.",
                         x2NDim, biasFirstDim),
                 return false);
        return true;
    }
    auto biasSecondDim = bias->GetViewShape().GetDim(1);
    // 2 is bias last dim index
    auto biasThirdDim = bias->GetViewShape().GetDim(2);
    // output batch need to be only 1 dim when bias dim is 3
    if (batchRecord.size() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When bias dim-num is 3, infered out batch dim-num should be 1, but infered out batch dim-num is %zu.",
                batchRecord.size());
                return false;
    }
    OP_CHECK(biasFirstDim == inferedOutbatchValue,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Bias 1st dim should be equal to out batch dim, but it is %ld and infered out batch dim is %ld.",
                biasFirstDim, inferedOutbatchValue), return false);
    OP_CHECK(biasSecondDim == 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias 2nd dim should be equal to 1, but it is %ld.", biasFirstDim),
             return false);
    OP_CHECK(biasThirdDim == x2NDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias last dim should be equal to x2 n dim %ld, but actually is %ld.",
                     x2NDim, biasThirdDim), return false);
    return true;
}

static inline bool CheckOutShape(const aclTensor *out, bool twoDimMatmulCaseFlag, int64_t x1MDim,
                                 int64_t x2NDim, const std::vector<int64_t> &batchRecord) {
    auto outDimNum = out->GetViewShape().GetDimNum();
    int64_t outMDim = out->GetViewShape().GetDim(outDimNum - PENULTIMATE_DIM);
    int64_t outNDim = out->GetViewShape().GetDim(outDimNum - 1);
    size_t inferedOutDimNum = batchRecord.size() + 2;
    // x1 and x2 are 2 dim and out is 3 dim speical case
    if (outMDim == 1 && inferedOutDimNum == 2 && outDimNum == 3 && twoDimMatmulCaseFlag) {
        outDimNum -= 1;
        outMDim = out->GetViewShape().GetDim(0);
    }
    if (inferedOutDimNum != outDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Infered output dim-num %zu is not equal to actual out dim-num %zu.",
                inferedOutDimNum, outDimNum);
        return false;
    }
    OP_CHECK(outMDim == x1MDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Out 1st dim should be equal to x1 m dim, but out 1st dim is %ld, x1 m dim is %ld.",
                     outMDim, x1MDim), return false);
    OP_CHECK(outNDim == x2NDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Out 2nd dim should be equal to x2 n dim, but out 2nd dim is %ld, x2 n dim is %ld.",
                     outNDim, x2NDim), return false);
    for (size_t i = 0; i < outDimNum - PENULTIMATE_DIM; i++) {
        OP_CHECK(out->GetViewShape().GetDim(i) == batchRecord[i],
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "Output dim %ld is not equal to infered output dim %ld at shape index %zu.",
                         out->GetViewShape().GetDim(i), batchRecord[i], i), return false);
    }
    return true;
}

static inline std::tuple<int64_t, int64_t, int64_t, int64_t> GetX1X2DimValue(const aclTensor *x1, const aclTensor *x2,
                                                                             bool transposeX1, bool transposeX2) {
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    const op::Shape x1Shape = x1->GetViewShape();
    const op::Shape x2Shape = x2->GetViewShape();
    int64_t x1KDim = transposeX1 ? x1Shape[x1DimNum - PENULTIMATE_DIM] : x1Shape[x1DimNum - 1];
    int64_t x1MDim = transposeX1 ? x1Shape[x1DimNum - 1] : x1Shape[x1DimNum - PENULTIMATE_DIM];
    int64_t x2KDim = transposeX2 ? x2Shape[x2DimNum - 1] : x2Shape[x2DimNum - PENULTIMATE_DIM];
    int64_t x2NDim = transposeX2 ? x2Shape[x2DimNum - PENULTIMATE_DIM] : x2Shape[x2DimNum - 1];
    return std::tie(x1KDim, x1MDim, x2KDim, x2NDim);
}

static inline bool CheckDimValue(const aclTensor *scale, const aclTensor *offset,
                                 const aclTensor *pertokenScaleOptional, int64_t x2NDim, int64_t x1MDim) {
    if (scale->GetViewShape().GetDim(0) != x2NDim && scale->GetViewShape().GetDim(0) != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Scale last dim should equal to x2 n dim %ld or 1, but actual is %ld.",
                x2NDim, scale->GetViewShape().GetDim(0));
        return false;
    }

    if (offset != nullptr) {
        OP_CHECK_WRONG_DIMENSION(offset, 1, return false);
        if (offset->GetViewShape().GetDim(0) != x2NDim && offset->GetViewShape().GetDim(0) != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Offset 1st dim should equal to x2 n dim %ld or 1, but actual is %ld.",
                    x2NDim, offset->GetViewShape().GetDim(0));
            return false;
        }
    }

    if (pertokenScaleOptional != nullptr) {
        OP_CHECK_WRONG_DIMENSION(pertokenScaleOptional, 1, return false);
        if (pertokenScaleOptional->GetViewShape().GetDim(0) != x1MDim) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "PertokenScaleOptional 1st dim should be equal to x1 m dim %ld or 1, but actually is %ld.",
                    x1MDim, pertokenScaleOptional->GetViewShape().GetDim(0));
            return false;
        }
    }
    return true;
}

static inline bool MaxDimCheck(int64_t x1DimNum, int64_t x2DimNum, const op::Shape x1Shape, const op::Shape x2Shape) {
    OP_CHECK(x1Shape[x1DimNum - 1] <= LAST_AXIS_LIMIT && x2Shape[x2DimNum - 1] <= LAST_AXIS_LIMIT,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 last dim or x2 last dim is larger than 65535, x1 is %ld, x2 is %ld.",
                x1Shape[x1DimNum - 1], x2Shape[x2DimNum - 1]),
        return false);
    return true;
}

static inline int64_t SelectNzK0Value(op::DataType dataType) {
    switch (dataType) {
        case op::DataType::DT_INT4:
            return NZ_K0_VALUE_INT4_TRANS;
        case op::DataType::DT_INT32:
            return NZ_K0_VALUE_INT32_TRANS;
        default:
            return NZ_K0_VALUE_INT8_TRANS;
    }
}

static inline bool CheckShapeForWeightNz(const aclTensor *x1, const aclTensor *x2, bool transposeX1, bool transposeX2) {
    const op::Shape x1Shape = x1->GetViewShape();
    const op::Shape x2Shape = x2->GetStorageShape();
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetStorageShape().GetDimNum();
    int64_t x1KDim = transposeX1 ? x1Shape[x1DimNum - PENULTIMATE_DIM] : x1Shape[x1DimNum - 1];
    int64_t x2K1Dim = transposeX2 ? x2Shape[x2DimNum - NZ_K1_INDEX_TRANS] : x2Shape[x2DimNum - NZ_K1_INDEX];
    int64_t nz_k0_value_trans = SelectNzK0Value(x2->GetDataType());
    int64_t roundValue = transposeX2 ? nz_k0_value_trans : NZ_K0_VALUE_BMM_BLOCK_NUM;
    int64_t x1KDimRound = ((x1KDim + roundValue - 1) / roundValue) * roundValue;
    if (x1KDimRound != x2K1Dim * roundValue) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "AlignedK1 value %ld is not matched with k1 value times roundValue, which is %ld.",
            x1KDimRound, x2K1Dim * roundValue);
        return false;
    }
    return true;
}

template <typename T>
static inline bool IsAligned(T num, T factor)
{
    if (factor == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The divisor cannot be zero.");
        return false;
    }
    return num > 0 && num % factor == 0;
}

static inline bool CheckShapeInt4(const aclTensor *x1, const aclTensor *x2, bool transposeX1, bool transposeX2,
                                  const aclTensor *bias)
{
    int64_t x1KDim, x1MDim, x2KDim, x2NDim;
    std::tie(x1KDim, x1MDim, x2KDim, x2NDim) = GetX1X2DimValue(x1, x2, transposeX1, transposeX2);
    if (!IsAligned<int64_t>(x1KDim, INT4_NUMS_IN_INT8)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "x1_k should be a positive even number in a4w4/a8w4 senario, but now x1_k is %ld.", x1KDim);
        return false;
    }
    if (transposeX2 && !IsAligned<int64_t>(x2KDim, INT4_NUMS_IN_INT8)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "x2_k should be a positive even number when transposeX2 is true in a4w4 senario, but now x2_k is %ld.", x2KDim);
        return false;
    }
    if (isA8W4Int(x1, x2) && x1KDim > MAX_SHAPE_SIZE_A8W4_INT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The k-dim must belong to [1, %ld], which is %ld", MAX_SHAPE_SIZE_A8W4_INT, x1KDim);
        return false;
    }
    if (!transposeX2 && !IsAligned<int64_t>(x2NDim, INT4_NUMS_IN_INT8)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "x2_n should be a positive even number when transposeX2 is false in a4w4 senario, but now x2_n is %ld.", x2NDim);
        return false;
    }
    if (transposeX1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "TransposeX1 should be false in a4w4/a8w4 senario, but now is true.");
        return false;
    }
    if (x2->GetViewShape().GetDimNum() != X2_FIXED_DIM_NUM_A4W4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2 should be 2-d in a4w4/a8w4, but is %zu.", x2->GetViewShape().GetDimNum());
        return false;
    }
    if (bias != nullptr && bias->GetViewShape().GetDimNum() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias should be 1-d in a4w4/a8w4, but is %zu.", bias->GetViewShape().GetDimNum());
        return false;
    }
    return true;
}

static inline bool CheckShape(TupleTensor &mandatoryTensors, TupleOptional &optionalTensors,
                             TupleAttr &boolsTrans, bool isA4W4, const aclTensor *out) {
    auto transposeX1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(boolsTrans);
    auto transposeX2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(boolsTrans);
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto offset = std::get<INDEX_OFFSET_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    const op::Shape x1Shape = x1->GetViewShape();
    const op::Shape x2Shape = x2->GetViewShape();
    int64_t x1KDim;
    int64_t x1MDim;
    int64_t x2KDim;
    int64_t x2NDim;
    std::tie(x1KDim, x1MDim, x2KDim, x2NDim) = GetX1X2DimValue(x1, x2, transposeX1, transposeX2);

    if ((isA4W4 || isA8W4Msd(x1, x2, scale, pertokenScaleOptional)) && !CheckShapeInt4(x1, x2, transposeX1, transposeX2, bias)) {
        return false;
    }

    OP_CHECK(x1KDim == x2KDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 k dim and x2 k dim should be same, but x1 is %ld, x2 is %ld.",
                     x1KDim, x2KDim), return false);

    CHECK_RET(MaxDimCheck(x1DimNum, x2DimNum, x1Shape, x2Shape), false);

    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat())) == Format::FORMAT_FRACTAL_NZ) {
        CHECK_RET(CheckShapeForWeightNz(x1, x2, transposeX1, transposeX2), false);
    }

    CHECK_RET(CheckDimValue(scale, offset, pertokenScaleOptional, x2NDim, x1MDim), false);

    std::vector<int64_t> batchRecord;
    int64_t inferedOutbatchValue = InferOutputShape(x1, x2, batchRecord);
    if (inferedOutbatchValue == OUTPUT_INFER_FAIL) {
        return false;
    }
    if (bias != nullptr) {
        if (!CheckBiasShape(bias, x2NDim, batchRecord, inferedOutbatchValue)) {
            return false;
        }
    }
    bool twoDimMatmulCaseFlag = x1DimNum == x2DimNum && x2DimNum == 2;
    CHECK_RET(CheckOutShape(out, twoDimMatmulCaseFlag, x1MDim, x2NDim, batchRecord), false);
    return true;
}

static inline bool CheckEmptyTensor(TupleTensor mandatoryTensors) {
    // scale, out和可选参数已在CheckShape函数校验，无需再次校验空tensor场景。
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    if (x1->IsEmpty() || x2->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QuantMatmul not support to process empty tensor currently.");
        return false;
    }
    return true;
}

static inline bool IsMicroScaling(const aclTensor *x1Scale, const aclTensor *x2Scale) {
    if (x1Scale == nullptr) {
        return false;
    }
    return x1Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0 &&
           x2Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0;
}

static bool CheckA8W4Dtype(const TupleTensor &mandatoryTensors, const TupleOptional &optionalTensors) {
    auto x2Scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x1Scale = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto yScale = std::get<INDEX_Y_SCALE_IN_OPTIONAL_TUPLE>(optionalTensors);

    if (x1Scale == nullptr) {
        // A8W4 mode
        if (bias != nullptr || yScale == nullptr) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "In A8W4 t-cg quant mode, bias must be null and yScale cannot be null.");
            return false;
        }
        if (x2Scale == nullptr || x2Scale->GetDataType() != DataType::DT_BF16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In A8W4 mode, x2Scale must be bf16.");
            return false;            
        }

        OP_CHECK_DTYPE_NOT_SUPPORT(yScale, Y_SCALE_SUPPORT_LIST, return false);
    } else if (IsMicroScaling(x1Scale, x2Scale)) {
        // MxA8W4 mode
        if (bias != nullptr && bias->GetDataType() != DataType::DT_BF16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In A8W4 mx quant mode, bias dtype must be bfloat16.");
            return false;
        }
        if (yScale != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In A8W4 mx quant mode, yScale must be null.");
            return false;
        }
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unexpected quant mode, A8W4 only support t-cg and mx quant mode!");
        return false;
    }
    return true;
}

static inline bool CheckA8W4Format(const TupleTensor &mandatoryTensors, const TupleOptional &optionalTensors,
                                   const aclTensor *out) {
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2Scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x1Scale = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto yScale = std::get<INDEX_Y_SCALE_IN_OPTIONAL_TUPLE>(optionalTensors);
    CHECK_RET(x1->GetStorageFormat() == op::Format::FORMAT_ND, false);
    CHECK_RET(x2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ, false);
    CHECK_RET(x2Scale->GetStorageFormat() == op::Format::FORMAT_ND, false);

    if (x1Scale != nullptr) {
        CHECK_RET(x1Scale->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    if (bias != nullptr) {
        CHECK_RET(bias->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    if (yScale != nullptr) {
        CHECK_RET(yScale->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    CHECK_RET(out->GetStorageFormat() == op::Format::FORMAT_ND, false);
    return true;
}

static inline bool CheckA8W4ScaleX1Shape(
    const TupleOptional& optionalTensors, const TupleTensor& mandatoryTensors, int64_t groupDimM, int64_t groupDimK)
{
    auto x1Scale = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto x2Scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    if (IsMicroScaling(x1Scale, x2Scale)) {
        // 2：x1Scale 形状为（m, groupDimK / 2, 2）
        if (x1Scale->GetViewShape().GetDim(0) != groupDimM || x1Scale->GetViewShape().GetDim(1) != groupDimK / 2 ||
            x1Scale->GetViewShape().GetDim(2) != 2) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "The x1scale shape must be [%ld, %ld, 2], which are [%ld, %ld, %d]", groupDimM,
                groupDimK, x1Scale->GetViewShape().GetDim(0), x1Scale->GetViewShape().GetDim(1),
                x1Scale->GetViewShape().GetDim(2));
            return false;
        }
    }
    return true;
}

static inline bool CheckA8W4ScaleX2Shape(
    const TupleOptional& optionalTensors, const TupleTensor& mandatoryTensors, int64_t groupDimN, int64_t groupDimK,
    bool transposeX2)
{
    auto x1Scale = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto x2Scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    int64_t x2ScaleNDim = transposeX2 ? x2Scale->GetViewShape().GetDim(0) : x2Scale->GetViewShape().GetDim(1);
    int64_t x2ScaleGroupDim = transposeX2 ? x2Scale->GetViewShape().GetDim(1) : x2Scale->GetViewShape().GetDim(0);
    if (IsMicroScaling(x1Scale, x2Scale)) {
        // 2： x2Scale形状：（n, k / 32 / 2, 2）
        if (x2ScaleNDim != groupDimN || x2ScaleGroupDim != groupDimK / 2 || x2Scale->GetViewShape().GetDim(2) != 2) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "The x2scale shape must be [%ld, %ld, 2], which are [%ld, %ld, %ld]",
                groupDimN, groupDimK / 2, x2ScaleNDim, x2ScaleGroupDim, x2Scale->GetViewShape().GetDim(2));
            return false;
        }
    } else if (x2ScaleNDim != groupDimN || x2ScaleGroupDim != groupDimK) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The x2scale shape must be [%ld, %ld], which are [%ld, %ld]", groupDimN, groupDimK,
            x2ScaleNDim, x2ScaleGroupDim);
        return false;
    }
    return true;
}

static inline bool CheckA8W4OutAndBiasShape(const TupleOptional& optionalTensors, int64_t x1MDim, int64_t x2NDim,
                                            const aclTensor* out)
{
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto yScale = std::get<INDEX_Y_SCALE_IN_OPTIONAL_TUPLE>(optionalTensors);
    if (bias != nullptr) {
        if (bias->GetViewShape().GetDim(0) != 1 || bias->GetViewShape().GetDim(1) != x2NDim) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The bias shape must be [1, %ld], which are [%ld, %ld]", x2NDim,
                    bias->GetViewShape().GetDim(0), bias->GetViewShape().GetDim(1));
            return false;
        }
    }

    if (out->GetViewShape().GetDim(0) != x1MDim || out->GetViewShape().GetDim(1) != x2NDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The output shape must be [%ld, %ld], which are [%ld, %ld]", x1MDim, x2NDim,
                out->GetViewShape().GetDim(0), out->GetViewShape().GetDim(1));
        return false;
    }

    if (yScale != nullptr) {
        if (yScale->GetViewShape().GetDim(1) != x2NDim || yScale->GetViewShape().GetDim(0) != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The yScale shape must be [1, %ld], which are [%ld, %ld]",
                    x2NDim, yScale->GetViewShape().GetDim(0), yScale->GetViewShape().GetDim(1));
            return false;
        }
    }
    return true;
}

static inline bool CheckA8W4X1X2Shape(int64_t x1KDim, int64_t x2KDim, int64_t x2NDim) {
    // CHECK x1KDim
    if (x1KDim <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the k-dim must be at least 1, which is %ld", x1KDim);
        return false;
    }
    if (x2NDim <= 0) { // A8W4Float
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the n-dim must be at least 1, which is %ld", x2NDim);
        return false;
    }

    if (x1KDim != x2KDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the k dim of x1 and x2 must be same, which are %ld and %ld", x1KDim, x2KDim);
        return false;
    }
    if (x1KDim % SUPPORTED_K_ALIGN_NUM != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the k dim must be align to %ld, which is %ld", SUPPORTED_K_ALIGN_NUM, x1KDim);
        return false;
    }
    if (x2NDim % SUPPORTED_N_ALIGN_NUM != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "A8W4 NZ the n dim must be align to %ld, which is %ld", SUPPORTED_N_ALIGN_NUM,
                x2NDim);
        return false;
    }
    return true;
}

static inline bool CheckA8W4Shape(const TupleTensor &mandatoryTensors, const TupleOptional &optionalTensors,
                                  const TupleAttr &boolsTrans, const aclTensor *out) {
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    bool transposeX1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(boolsTrans);
    bool transposeX2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(boolsTrans);

    auto x1DimNum = x1->GetViewShape().GetDimNum();
    const op::Shape x1Shape = x1->GetViewShape();
    int64_t x1KDim = transposeX1 ? x1Shape[x1DimNum - PENULTIMATE_DIM] : x1Shape[x1DimNum - 1];
    int64_t x1MDim = transposeX1 ? x1Shape[x1DimNum - 1] : x1Shape[x1DimNum - PENULTIMATE_DIM];

    const op::Shape x2Shape = x2->GetViewShape();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    int64_t x2KDim = transposeX2 ? x2Shape[x2DimNum - 1] : x2Shape[x2DimNum - PENULTIMATE_DIM];
    int64_t x2NDim = transposeX2 ? x2Shape[x2DimNum - PENULTIMATE_DIM] : x2Shape[x2DimNum - 1];
    if (x1MDim <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the m-dim must at least 1, which is %ld", x1MDim);
        return false;
    }
    CHECK_RET(CheckA8W4X1X2Shape(x1KDim, x2KDim, x2NDim), false);
    int64_t groupDimK = (x2KDim + SUPPORTED_GROUP_SIZE - 1) / SUPPORTED_GROUP_SIZE;
    int64_t groupDimM = x1MDim;
    int64_t groupDimN = x2NDim;
    CHECK_RET(CheckA8W4ScaleX1Shape(optionalTensors, mandatoryTensors, groupDimM, groupDimK), false);
    CHECK_RET(CheckA8W4ScaleX2Shape(optionalTensors, mandatoryTensors, groupDimN, groupDimK, transposeX2), false);
    CHECK_RET(CheckA8W4OutAndBiasShape(optionalTensors, x1MDim, x2NDim, out), false);
    return true;
}

static inline aclnnStatus CheckParamsA8W4Float(const TupleTensor &mandatoryTensors, const TupleOptional &optionalTensors,
                                          const TupleAttr &boolsTrans, const aclTensor *out) {
    // 1. 校验dtype是否符合要求
    CHECK_RET(CheckA8W4Dtype(mandatoryTensors, optionalTensors), ACLNN_ERR_PARAM_INVALID);
    // 2. 检查format是否符合要求
    CHECK_RET(CheckA8W4Format(mandatoryTensors, optionalTensors, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查shape是否符合要求
    CHECK_RET(CheckA8W4Shape(mandatoryTensors, optionalTensors, boolsTrans, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParamsDAV3510(TupleTensor mandatoryTensors, TupleOptional optionalTensors, TupleAttr boolsTrans,
                                     const aclTensor *out) {
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);                                    
    if (isA8W4Float(x1, x2)) {
        return CheckParamsA8W4Float(mandatoryTensors, optionalTensors, boolsTrans, out);
    }
    const TupleInput inputTensors = std::tie(std::get<0>(mandatoryTensors), std::get<1>(mandatoryTensors));
    const aclTensor *yScale = nullptr;
    const aclTensor *x1Offset = nullptr;
    const int64_t groupSize = std::get<INDEX_GROUP_SIZE_IN_OPTIONAL_TUPLE>(optionalTensors);
    // 5 represents the aclnnQuantMatmulV5 interface
    const int64_t interfaceType = 5;
    const TupleQuant quantTensors =
        std::tie(std::get<1>(optionalTensors), std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors), yScale,
                    x1Offset, std::get<0>(optionalTensors), x1Offset,
                    std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors), groupSize, interfaceType);

    int64_t groupSizeReal = groupSize;
    auto& scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    if (isMxfp8Nz(x1, x2, scale)) {
        QuantMatmulChecker qmmV3Checker(inputTensors, quantTensors, boolsTrans, out);
        qmmV3Checker.Init();
        CHECK_RET(qmmV3Checker.InferGroupSize(groupSizeReal), ACLNN_ERR_PARAM_INVALID);
        OP_LOGD("Infer groupSize success. groupSize: %ld.", groupSizeReal);
    }
    const TupleQuant quantTuples =
        std::tie(std::get<1>(optionalTensors), std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors), yScale,
                    x1Offset, std::get<0>(optionalTensors), x1Offset,
                    std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors), groupSizeReal, interfaceType);

    QuantMatmulChecker qmmV3Checker(inputTensors, quantTuples, boolsTrans, out);
    qmmV3Checker.Init();
    return qmmV3Checker.CheckParams();
}

static bool IsFormatNZ(const aclTensor* tensor) {
    return ge::GetPrimaryFormat(tensor->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ ||
           ge::GetPrimaryFormat(tensor->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ_C0_4 ||
           ge::GetPrimaryFormat(tensor->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ_C0_32;
}

static aclnnStatus CheckWeightNzParamsDAV3510(const aclTensor *x1, const aclTensor *x2, const aclTensor *out)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        return ACLNN_SUCCESS;
    }

    if (x1 == nullptr || x2 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 and x2 can not be null");
        return ACLNN_ERR_PARAM_INVALID;        
    }

    if (isA8W4Float(x1, x2)) {
        if (!IsFormatNZ(x2)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "in A8W4 case, Format of x2 must be FRACTAL_NZ or FRACTAL_NZ_C0_4 or FRACTAL_NZ_C0_32, actual is %s.",
                op::ToString(x2->GetStorageFormat()).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (out->GetDataType() != op::DataType::DT_BF16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Data type of out only support bfloat16, actual is %s.",
                    op::ToString(out->GetDataType()).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat())) != Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of x2 must be FRACTAL_NZ, actual is %s.",
                op::ToString(x2->GetStorageFormat()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 对于torch的场景，NZ情况两维某一维度为1的场景无法正确判断是否转置，资料呈现不支持此种输入
    int64_t dim1 = x2->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = x2->GetViewShape().GetDimNum() - PENULTIMATE_DIM;
    if (x2->GetViewShape().GetDim(dim2) == 1 || x2->GetViewShape().GetDim(dim1) == 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of x2 must be FRACTAL_NZ, actual is %s.",
                op::ToString(x2->GetStorageFormat()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    OP_LOGD("QuantMatmulWeightNz check params success.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(TupleTensor mandatoryTensors, TupleOptional optionalTensors, TupleAttr boolsTrans,
                               bool isA4W4, const aclTensor *out) {
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        return CheckParamsDAV3510(mandatoryTensors, optionalTensors, boolsTrans, out);
    } else {
        // 1. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
        CHECK_RET(CheckDtypeValid(mandatoryTensors, optionalTensors, out, isA4W4), ACLNN_ERR_PARAM_INVALID);

        // 2. 检查shape是否符合要求
        CHECK_RET(CheckShape(mandatoryTensors, optionalTensors, boolsTrans, isA4W4, out), ACLNN_ERR_PARAM_INVALID);

        // 3. 检查format是否符合要求
        CHECK_RET(CheckFormat(mandatoryTensors, optionalTensors, isA4W4), ACLNN_ERR_PARAM_INVALID);

        // 4. 空Tensor处理逻辑
        CHECK_RET(CheckEmptyTensor(mandatoryTensors), ACLNN_ERR_PARAM_INVALID);
    }
    OP_LOGD("QuantMatmul check params success.");
    return ACLNN_SUCCESS;
}

static bool CheckSpecialCase(const aclTensor* tensor, int64_t firstLastDim, int64_t secondLastDim)
{
    if ((tensor->GetViewShape().GetDim(firstLastDim) == tensor->GetViewShape().GetDim(secondLastDim))
        && (tensor->GetViewShape().GetDim(secondLastDim) == 1))
        {
            OP_LOGD("QuantMatmul special case, no need to set transpose attr value.");
            return true;
        }
    return false;
}

static bool GetTransposeAttrValue(const aclTensor *tensor, bool transpose, bool checkSpecialCase = true) {
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - PENULTIMATE_DIM;
    // check if tensor is contiguous layout
    if (tensor->GetViewStrides()[dim2] == 1 && (tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2))) {
        OP_LOGD("QuantMatmul GetTransposeAttrValue, find tensor is not contiguous.");
        const_cast<aclTensor *>(tensor)->SetViewShape(SwapLastTwoDimValue(tensor->GetViewShape()));
        // 如果不需要校验特殊case，则直接返回
        if(!checkSpecialCase) {
            return !transpose;
        }
        if (!CheckSpecialCase(tensor, dim1, dim2)) {
            return !transpose;
        }
    }
    return transpose;
}

static const aclTensor* SetTensorToNDFormat(const aclTensor *input) {
    OP_LOGD("QuantMatmul set tensor to ND format.");
    auto formatTensor = const_cast<aclTensor *>(input);
    if (input->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) {
        formatTensor->SetViewFormat(op::Format::FORMAT_ND);
        formatTensor->SetOriginalFormat(op::Format::FORMAT_ND);
        formatTensor->SetStorageFormat(op::Format::FORMAT_ND);
    }
    return formatTensor;
}

static aclIntArray* GetPerm(int64_t dim, aclOpExecutor* executor) {
    CHECK_RET(dim >= MIN_DIM_NUM_ND, nullptr);
    std::vector<int64_t> valuePerm(dim);
    for (int64_t i = 0; i < dim; i++) {
        valuePerm[i] = i;
    }
    std::swap(valuePerm[dim - 1], valuePerm[dim - PENULTIMATE_DIM]);
    return executor->AllocIntArray(valuePerm.data(), dim);
}

static aclnnStatus TransposeAndTransDataForInputs(const aclTensor *&x1, const aclTensor *&x2, bool& transposeX1,
                                                  bool& transposeX2, aclOpExecutor* executor) {
    if (transposeX1) {
        auto perm = GetPerm(x1->GetViewShape().GetDimNum(), executor);
        CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
        x1 = l0op::Transpose(x1, perm, executor);
        CHECK_RET(x1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        transposeX1 = !transposeX1;
    }
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat())) == Format::FORMAT_FRACTAL_NZ) {
        return ACLNN_SUCCESS;
    }
    x2 = SetTensorToNDFormat(x2);
    if (!transposeX2) {
        auto perm = GetPerm(x2->GetViewShape().GetDimNum(), executor);
        CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
        x2 = l0op::Transpose(x2, perm, executor);
        CHECK_RET(x2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        transposeX2 = !transposeX2;
    }
    x2 = l0op::TransData(x2, Format::FORMAT_FRACTAL_NZ, 0, executor);
    CHECK_RET(x2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus TransdataForX1(const aclTensor *&inputTensor, aclOpExecutor* executor)
{
    OP_LOGD("QuantMatmul enter TransdataForX1 func.");
    inputTensor = l0op::Contiguous(inputTensor, executor);
    OP_CHECK(inputTensor != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
            "The function Contiguous() return nullptr, which causes function TransdataForX1() to fail."),
        return ACLNN_ERR_INNER_NULLPTR);
    inputTensor = l0op::TransData(inputTensor, Format::FORMAT_FRACTAL_NZ, 0, executor);
    OP_CHECK(inputTensor != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                "The function TransData() return nullptr, which causes function TransdataForX1() to fail."),
            return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static inline bool TensorContiguousProcess(const aclTensor *&contiguousTensor, bool &transpose, aclOpExecutor *executor) {
    if (contiguousTensor == nullptr) {
        OP_LOGD("QuantMatmul no need to do contiguous process.");
        return true;
    }
    bool isNZTensor = static_cast<ge::Format>(
        ge::GetPrimaryFormat(contiguousTensor->GetStorageFormat())) == op::Format::FORMAT_FRACTAL_NZ;
    auto storageShape = contiguousTensor->GetStorageShape();
    auto transposeFlag = IsTransposeLastTwoDims(contiguousTensor);
    // swap tensor if its viewshape not satisfy request shape without adding a transpose node
    if (transposeFlag) {
        contiguousTensor = executor->CreateView(contiguousTensor, SwapLastTwoDimValue(contiguousTensor->GetViewShape()),
                                                contiguousTensor->GetViewOffset());
        transpose = !transpose;
    } else {
        contiguousTensor = l0op::Contiguous(contiguousTensor, executor);
    }
    if (isNZTensor) {
        contiguousTensor->SetStorageShape(storageShape); //对NZ的场景需要用原NZshape刷新
    }
    CHECK_RET(contiguousTensor != nullptr, false);
    return true;
}

static aclnnStatus SpecialOutputProcess(const aclTensor *x1, const aclTensor *x2, const aclTensor *out,
                                        const aclTensor *&matmulRet, aclOpExecutor* executor) {
    // we have to reshape for case which x1 and x2 are 2 dims and out is 3 dims, otherwise, viewcopy will fail
    OP_LOGD("QuantMatmul enter SpecialOutputProcess func.");
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

static aclnnStatus CheckSupportSocVersion(bool isA4W4) {
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    NpuArch npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (isA4W4) {
        // a4w4 support 910B 910_93，其余暂不支持
        switch (npuArch) {
            case NpuArch::DAV_2201:
                break;
            default: {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                        "QuantBatchMatmul support for %s is not implemented in a4w4 senario.",
                        op::ToString(socVersion).GetString());
                return ACLNN_ERR_RUNTIME_ERROR;
            }
        }
    } else {
        switch (npuArch) {
            case NpuArch::DAV_2201:
            case NpuArch::DAV_3510:
            case NpuArch::DAV_2002:
                break;
            default: {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                        "QuantBatchMatmul support for %s is not implemented in a8w8 senario.",
                        op::ToString(socVersion).GetString());
                return ACLNN_ERR_RUNTIME_ERROR;
            }
        }
    }
    return ACLNN_SUCCESS;
}

static const aclTensor* GetNDFormat(const aclTensor *input) {
    const aclTensor* reformatedInput = input;
    if (input != nullptr) {
        reformatedInput = SetTensorToNDFormat(input);
    }
    return reformatedInput;
}

static aclTensor* ConvertTensorToInt4(const aclTensor* input, aclOpExecutor* executor)
{
    // 将int32的输入dtype修改为int4, 同时ViewShape和ViewStrides也从int32修改为int4所对应的。
    auto viewShape = input->GetViewShape();
    auto storageShape = input->GetStorageShape();
    auto viewShapeDim = viewShape.GetDimNum();
    viewShape[viewShapeDim - 1] = viewShape[viewShapeDim - 1] * INT4_NUMS_IN_INT32;
    auto inputTemp = executor->CreateView(input, viewShape, input->GetViewOffset());
    inputTemp->SetDataType(DataType::DT_INT4);
    if (input->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
        storageShape[storageShape.GetDimNum() - 1] = NZ_K0_VALUE_INT4_TRANS;
        storageShape[storageShape.GetDimNum() - MIN_DIM_NUM_NZ] = (viewShape[viewShapeDim - 1] +
            NZ_K0_VALUE_INT4_TRANS - 1) / NZ_K0_VALUE_INT4_TRANS;
        inputTemp->SetStorageShape(storageShape);
    }
    OP_LOGD("The conversion from int32 to int4 is completed.");
    return inputTemp;
}

static void InputPreProcessA4W4(const aclTensor *&x1, const aclTensor *&x2, bool &isA4W4, aclOpExecutor *executor)
{
    if (x1->GetDataType() == DataType::DT_INT32) {
        isA4W4 = true;
        x1 = ConvertTensorToInt4(x1, executor);
    }
    if (x2->GetDataType() == DataType::DT_INT32) {
        isA4W4 = true;
        x2 = ConvertTensorToInt4(x2, executor);
    }
    isA4W4 = isA4W4 || (x1->GetDataType() == DataType::DT_INT4 || x2->GetDataType() == DataType::DT_INT4);
}

static aclnnStatus WeightNZCaseProcess(const aclTensor *&x2, bool &transposeX2, aclOpExecutor *executor) {
    auto viewShape = x2->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    bool isNotOneDim = viewShapeDim >= PENULTIMATE_DIM && viewShape[viewShapeDim - 1] != 1 &&
                       viewShape[viewShapeDim - PENULTIMATE_DIM] != 1;
    auto formatX2 = static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat()));
    // if plateform is not DAV3510 and weight is already in nz format, no need to set contiguous
    if (formatX2 != op::Format::FORMAT_FRACTAL_NZ ||
        (isNotOneDim && op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510)) {
        CHECK_RET(TensorContiguousProcess(x2, transposeX2, executor), ACLNN_ERR_INNER_NULLPTR);
    }
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat())) == op::Format::FORMAT_FRACTAL_NZ) {
        x2->SetOriginalShape(x2->GetViewShape());
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus A4W4CaseProcess(const aclTensor *&x1, const aclTensor *&x2, bool &isA4W4, aclOpExecutor *executor) {
    InputPreProcessA4W4(x1, x2, isA4W4, executor);
    CHECK_RET(CheckSupportSocVersion(isA4W4) != ACLNN_ERR_RUNTIME_ERROR, ACLNN_ERR_RUNTIME_ERROR);
    return ACLNN_SUCCESS;
}

static aclnnStatus PostMatmulCalcProcess(const aclTensor *matmulRet, TupleTensor mandatoryTensors,
                                         aclOpExecutor *executor) {
    CHECK_RET(matmulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto out = std::get<INDEX_OUT_IN_TUPLE>(mandatoryTensors);
    CHECK_RET(SpecialOutputProcess(x1, x2, out, matmulRet, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(matmulRet, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static inline bool CheckInputAttrExistence(const TupleAttr &boolsTrans, const TupleTensor &mandatoryTensors,
                                           const TupleOptional &optionalTensors)
{
    auto &x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    int64_t groupSize = std::get<INDEX_GROUP_SIZE_IN_OPTIONAL_TUPLE>(optionalTensors);
    bool transposeX1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(boolsTrans);
    bool transposeX2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(boolsTrans);
    if (transposeX1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Only support transposeX1 is false.");
        return false;
    }

    bool isX2Nz = ge::GetPrimaryFormat(x2->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ;
    if (!isX2Nz) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "A8W4 x2 only support nz format.");
        return false;       
    }

    auto &x1Scale = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto &x2Scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    if (x1Scale == nullptr && transposeX2) {
        // A8W4 mode
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "A8W4 NZ T-CG quantization mode only support transposeX2 is false.");
        return false;
    } else if (IsMicroScaling(x1Scale, x2Scale)  && !transposeX2) {
            // MxA8W4 mode
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "A8W4 NZ mx quantization mode only support transposeX2 is true.");
        return false;
    }
    uint64_t groupSizeK = static_cast<uint64_t>(groupSize) & GROUP_MNK_BIT_SIZE;
    if (groupSizeK != SUPPORTED_GROUP_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "A8W4 only support groupSizeK equal to 32.");
        return false;
    }
    return true;
}

static inline bool CheckDimRangeA8W4(const TupleTensor& mandatoryTensors, const TupleOptional& optionalTensors,
                                     const aclTensor* out) {
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);

    if (x1->GetViewShape().GetDimNum() != MAX_DIM_VALUE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of x1 should be 2. Actual x1 dimension is: %zu.",
                x1->GetViewShape().GetDimNum());
        return false;
    }
    if (x2->GetViewShape().GetDimNum() != MAX_DIM_VALUE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of x2 should be 2. Actual x2 dimension is: %zu.",
                x2->GetViewShape().GetDimNum());
        return false;
    }
    if (bias != nullptr && bias->GetViewShape().GetDimNum() != MAX_DIM_VALUE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of bias should be 2. Actual bias dimension is: %zu.",
                bias->GetViewShape().GetDimNum());
        return false;
    }
    if (out->GetViewShape().GetDimNum() != MAX_DIM_VALUE) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The dimension of out should be 2. Actual out dimension is: %zu.",
            out->GetViewShape().GetDimNum());
        return false;
    }
    OP_LOGD("QuantMatmul check dimension range success.");
    return true;
}

static inline bool CheckScaleDimRangeA8W4(const TupleTensor& mandatoryTensors, const TupleOptional& optionalTensors)
{
    auto x2Scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x1Scale = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto yScale = std::get<INDEX_Y_SCALE_IN_OPTIONAL_TUPLE>(optionalTensors);

    if (IsMicroScaling(x1Scale, x2Scale)) {
        if (x1Scale->GetViewShape().GetDimNum() != MX_SCALE_DIM_VALUE) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "The dimension of x1Scale should be 3. Actual x1Scale dimension is: %zu.",
                x1Scale->GetViewShape().GetDimNum());
            return false;
        }
        if (x2Scale != nullptr && x2Scale->GetViewShape().GetDimNum() != MX_SCALE_DIM_VALUE) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "The dimension of x2Scale should be 3. Actual x2Scale dimension is: %zu.",
                x2Scale->GetViewShape().GetDimNum());
            return false;
        }
    } else {
        if (x1Scale != nullptr && x1Scale->GetViewShape().GetDimNum() != MAX_DIM_VALUE) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "The dimension of x1Scale should be 2. Actual x1Scale dimension is: %zu.",
                x1Scale->GetViewShape().GetDimNum());
            return false;
        }
        if (x2Scale !=nullptr && x2Scale->GetViewShape().GetDimNum() != MAX_DIM_VALUE) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "The dimension of x2Scale should be 2. Actual x2Scale dimension is: %zu.",
                x2Scale->GetViewShape().GetDimNum());
            return false;
        }
    }
    if (yScale != nullptr && yScale->GetViewShape().GetDimNum() != MAX_DIM_VALUE) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The dimension of yScale should be 2. Actual yScale dimension is: %zu.",
            yScale->GetViewShape().GetDimNum());
        return false;
    }
    OP_LOGD("QuantMatmul check scale dimension range success.");
    return true;
}

static inline bool MxScaleContiguousProcess(const aclTensor*& mxScaleTensor, aclOpExecutor* executor)
{
    if (mxScaleTensor == nullptr || mxScaleTensor->GetViewShape().GetDimNum() < MX_SCALE_MAX_DIM) {
        OP_LOGD("MX scale no need to do contiguous process.");
        return true;
    }
    auto transposeFlag = false;
    int64_t dimNum = mxScaleTensor->GetViewShape().GetDimNum();
    int64_t lastDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 1);
    int64_t lastSecondDim = mxScaleTensor->GetViewShape().GetDim(dimNum - PENULTIMATE_DIM);
    int64_t lastThirdDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 3); // 3: 倒数第3维
    if (mxScaleTensor->GetViewStrides()[dimNum - 3] == lastDim &&            // 3： 倒数第3维
        mxScaleTensor->GetViewStrides()[dimNum - PENULTIMATE_DIM] == lastDim * lastThirdDim) {
        int64_t tmpNxD = lastDim * lastSecondDim * lastThirdDim;
        transposeFlag = true;
        // 4：batch维度从倒数第4维起
        for (int64_t batchDim = dimNum - 4; batchDim >= 0; batchDim--) {
            if (mxScaleTensor->GetViewStrides()[batchDim] != tmpNxD) {
                transposeFlag = false;
                break;
            }
            tmpNxD *= mxScaleTensor->GetViewShape().GetDim(batchDim);
        }
        if (lastSecondDim == 1 && lastThirdDim == 1) {
            transposeFlag = false;
        }
    }
    if (transposeFlag) {
        op::Shape swapedShape = mxScaleTensor->GetViewShape();
        swapedShape.SetDim(dimNum - PENULTIMATE_DIM, lastThirdDim);
        swapedShape.SetDim(dimNum - 3, lastSecondDim); // 3： 倒数第3维
        mxScaleTensor = executor->CreateView(mxScaleTensor, swapedShape, mxScaleTensor->GetViewOffset());
    } else {
        mxScaleTensor = l0op::Contiguous(mxScaleTensor, executor);
    }
    CHECK_RET(mxScaleTensor != nullptr, false);
    return true;
}

static aclnnStatus PreMatmulCalcProcess(TupleTensor &mandatoryTensors, TupleOptional &optionalTensors,
                                        TupleAttr &boolsTrans, bool &isA4W4, const aclTensor *out,
                                        aclOpExecutor *executor) {
    auto &x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto &x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto &scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto &perTokenScale = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    bool &transposeX1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(boolsTrans);
    bool &transposeX2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(boolsTrans);
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    CHECK_RET(CheckNotNull(std::tie(x1, x2, scale), out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(TensorContiguousProcess(x1, transposeX1, executor), ACLNN_ERR_INNER_NULLPTR);
    if (perTokenScale != nullptr) {
 	    if (IsMicroScaling(perTokenScale, scale)) {
 	        CHECK_RET(MxScaleContiguousProcess(scale, executor), ACLNN_ERR_INNER_NULLPTR);
 	        CHECK_RET(MxScaleContiguousProcess(perTokenScale, executor), ACLNN_ERR_INNER_NULLPTR);
 	    }
 	}
    auto ret = WeightNZCaseProcess(x2, transposeX2, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (isA8W4Float(x1, x2)) {
        bool tempTransposeValue = false;
        CHECK_RET(TensorContiguousProcess(perTokenScale, tempTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(TensorContiguousProcess(scale, tempTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(
            CheckInputAttrExistence(boolsTrans, mandatoryTensors, optionalTensors),
            ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckDimRangeA8W4(mandatoryTensors, optionalTensors, out), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckScaleDimRangeA8W4(mandatoryTensors, optionalTensors), ACLNN_ERR_PARAM_INVALID);
    } else {
        CHECK_RET(CheckDimRange(x1, x2, scale, out), ACLNN_ERR_PARAM_INVALID);
    }
    A4W4CaseProcess(x1, x2, isA4W4, executor);
    return ACLNN_SUCCESS;
}

static void GetDtypeAndTranspose(TupleTensor mandatoryTensors, int64_t &dtype, bool &transposeX1,
                                 bool &transposeX2) {
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto out = std::get<INDEX_OUT_IN_TUPLE>(mandatoryTensors);
    dtype = static_cast<int64_t> (out->GetDataType());
    transposeX1 = GetTransposeAttrValue(x1, transposeX1, true);
    transposeX2 = GetTransposeAttrValue(x2, transposeX2, true);
    OP_LOGD("QuantMatmul attr transposeX1 is %d, transposeX2 is %d.", transposeX1, transposeX2);
}

static aclTensor* ProcessScaleTensor(const aclTensor *scale) {
    auto castedScale = const_cast<aclTensor*>(scale);
    if (castedScale->GetDataType() == op::DataType::DT_INT64) {
        castedScale->SetDataType(op::DataType::DT_UINT64);
    }
    return castedScale;
}

static bool IsX1Transdata(const aclTensor *x1, const aclTensor *x2, int64_t dtype, bool transposeX1, bool transposeX2)
{
    if (transposeX1 == true || transposeX2 == true) {
        return false;
    }
    if (x1->GetStorageFormat() != op::Format::FORMAT_ND || x2->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) {
        return false;
    }
    if (x1->GetDataType() != op::DataType::DT_INT8) {
        return false;
    }
    if (dtype != static_cast<int>(op::DataType::DT_FLOAT16) && dtype != static_cast<int>(op::DataType::DT_BF16) &&
        dtype != static_cast<int>(op::DataType::DT_INT32)) {
        return false;
    }
    // innersize待校验
    Shape x1Shape = x1->GetViewShape();
    int64_t x1DimNum = x1Shape.GetDimNum();
    Shape x2Shape = x2->GetOriginalShape();
    int64_t x2DimNum = x2Shape.GetDimNum();
    if (x1DimNum != MIN_DIM_NUM_ND || x2DimNum != MIN_DIM_NUM_ND) {
        return false;
    }
    int64_t m = transposeX1 ? x1Shape.GetDim(x1DimNum - 1) : x1Shape.GetDim(x1DimNum - 2);
    int64_t k = transposeX1 ? x1Shape.GetDim(x1DimNum - 2) : x1Shape.GetDim(x1DimNum - 1);
    int64_t n = transposeX2 ? x2Shape.GetDim(x2DimNum - 2) : x2Shape.GetDim(x2DimNum - 1);
    int64_t innerSize = x1Shape.GetDim(x1DimNum - 1);
    // m校验
    bool isSupportedM = false;
    if ((m > M_RANGE1_LEFT && m <= M_RANGE1_RIGHT)) {
        isSupportedM = true;
    }
    if (innerSize % INNER_SIZE_MULTIPLE == 0 || !isSupportedM || k != K_VALUE || n != N_VALUE) {
        return false;
    }
    return true;
}

static void A8W4ProcessYScaleTensor(const aclTensor *x1Scale, const aclTensor *yScale) {
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        // A8W4场景输入的INT64转为UINT64
        if (x1Scale == nullptr && yScale != nullptr && yScale->GetDataType() == op::DataType::DT_INT64) {
            auto castYScale = const_cast<aclTensor*>(yScale);
            castYScale->SetDataType(op::DataType::DT_UINT64);
            yScale = castYScale;
            OP_LOGD("The conversion from INT64 to UINT64 has been completed.");
        }
    }
}

static inline bool A8W4ValidGroupSize(uint64_t groupSizeM, uint64_t groupSizeN) {
    return (groupSizeM == 0 && groupSizeN == 0) || (groupSizeM == 1 && groupSizeN == 1);
}

static inline bool A8W4InferGroupSize(int64_t& groupSize) {
    uint64_t groupSizeK = static_cast<uint64_t>(groupSize) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(groupSize) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (static_cast<uint64_t>(groupSize) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    if (!A8W4ValidGroupSize(groupSizeM, groupSizeN)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "The valid groupSizeM and groupSizeN must both be 0 or 1, actual groupSizeM [%lu], groupSizeN [%lu]!",
            groupSizeM, groupSizeN);
        return false;
    }

    OP_LOGD(
        "A8W4 after Infered groupSize: groupSizeM: %lu, groupSizeN: %lu, groupSizeK: %lu.", groupSizeM, groupSizeN,
        groupSizeK);
    groupSize = groupSizeK;
    return true;
}

static aclnnStatus TensorPreProcess(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* yScale, int64_t& groupSize)
{
    bool isA8W4F = isA8W4Float(x1, x2);
    if (isA8W4F) {
        A8W4ProcessYScaleTensor(x1Scale, yScale);
        CHECK_RET(A8W4InferGroupSize(groupSize), ACLNN_ERR_PARAM_INVALID);
        OP_LOGD("Infer groupSize success. groupSize: %ld.", groupSize);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus SetReformtedX2(const aclTensor *&reformatedX1, const aclTensor *&reformatedX2, bool& transposeX1,
                                       bool& transposeX2, aclOpExecutor* executor) {
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
        auto ret = TransposeAndTransDataForInputs(reformatedX1, reformatedX2, transposeX1, transposeX2,
                                             executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    } else {
        reformatedX2 = SetTensorToNDFormat(reformatedX2);
    }
    return ACLNN_SUCCESS;
}

static inline aclnnStatus TransdataX1Process(bool isX1TransdataFlag, const aclTensor *&reformatedX1,
                                             aclOpExecutor *executor, bool isPpMatmul)
{
    auto socLongVersion = GetCurrentPlatformInfo().GetSocLongVersion();
    bool checkSocLongVersion =
        (socLongVersion == "Ascend910B3" || socLongVersion == "Ascend910B4" || socLongVersion == "Ascend910B4-1");
    auto coreNum = static_cast<int32_t>(GetCurrentPlatformInfo().GetCubeCoreNum());
    if ((isX1TransdataFlag && checkSocLongVersion && coreNum == CORE_NUM_20) || isPpMatmul) {
        auto ret = TransdataForX1(reformatedX1, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    return ACLNN_SUCCESS;
}


static aclnnStatus aclnnQuantMatmulGetWorkspaceSizeCommonProcess(TupleTensor mandatoryTensors,
                                                                 TupleOptional optionalTensors,
                                                                 TupleAttr boolsTrans, const aclTensor *out,
                                                                 aclOpExecutor *executor) {
    auto &x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto &x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto &scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto &offset = std::get<INDEX_OFFSET_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto &pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto &bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto &yScale = std::get<INDEX_Y_SCALE_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto &yOffset = std::get<INDEX_Y_OFFSET_IN_OPTIONAL_TUPLE>(optionalTensors);
    int64_t groupSize = std::get<INDEX_GROUP_SIZE_IN_OPTIONAL_TUPLE>(optionalTensors);
    bool &transposeX1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(boolsTrans);
    bool &transposeX2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(boolsTrans);
    bool isA8W4F = isA8W4Float(x1, x2);
    bool isA8W4I = isA8W4Int(x1, x2);
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        auto x1DimNum = x1->GetViewShape().GetDimNum();
        auto inputSizeM = transposeX1 ? x1->GetViewShape().GetDim(x1DimNum - 1) :
                                        x1->GetViewShape().GetDim(x1DimNum - PENULTIMATE_DIM);
        auto x2DimNum = x2->GetViewShape().GetDimNum();
        auto inputSizeN = transposeX2 ? x2->GetViewShape().GetDim(x2DimNum - PENULTIMATE_DIM) :
                                        x2->GetViewShape().GetDim(x2DimNum - 1);
        if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat())) == Format::FORMAT_FRACTAL_NZ) {
            if (inputSizeM == 0) {
                OP_LOGD("aclnnV4 nz m=0");
                return ACLNN_SUCCESS;
            }
        } else {
            if (inputSizeM == 0 || inputSizeN == 0) {
                OP_LOGD("aclnnV4 nd m/n=0");
                return ACLNN_SUCCESS;
            }
        }
    }
    auto ret = TensorPreProcess(x1, x2, pertokenScaleOptional, yScale, groupSize);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    bool isA4W4 = false;
    ret = PreMatmulCalcProcess(mandatoryTensors, optionalTensors, boolsTrans, isA4W4, out, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    bool biasTransposeValue = false;
    CHECK_RET(TensorContiguousProcess(bias, biasTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);
    bool scaleTransposeValue = false;
 	CHECK_RET(TensorContiguousProcess(scale, scaleTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);
 	bool offsetTransposeValue = false;
 	CHECK_RET(TensorContiguousProcess(offset, offsetTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);
 	bool perTokenScaleTransposeValue = false;
 	CHECK_RET(TensorContiguousProcess(pertokenScaleOptional, perTokenScaleTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);
    auto reformatedX1 = SetTensorToNDFormat(x1);
    const aclTensor* reformatedX2 = x2;
    ret = SetReformtedX2(reformatedX1, reformatedX2, transposeX1, transposeX2, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor* reformatedScale = GetNDFormat(scale);
    const aclTensor* reformatedpertokenScaleOptional = GetNDFormat(pertokenScaleOptional);
    const aclTensor* reformatedBias = GetNDFormat(bias);
    const aclTensor* reformatedYScale = GetNDFormat(yScale);

    ret = CheckParams(std::tie(reformatedX1, reformatedX2, reformatedScale),
                      std::tie(offset, reformatedpertokenScaleOptional, reformatedBias, reformatedYScale, yOffset, groupSize),
                      std::tie(transposeX1, transposeX2), isA4W4, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto castedScale = ProcessScaleTensor(reformatedScale);
    int64_t dtype = 0;
    GetDtypeAndTranspose(std::tie(reformatedX1, reformatedX2, out), dtype, transposeX1, transposeX2);
    bool isX1TransdataFlag = IsX1Transdata(reformatedX1, reformatedX2, dtype, transposeX1, transposeX2);
    auto inputAShape = reformatedX1->GetViewShape();
    uint32_t M = inputAShape.GetDimNum() == NO_BATCH_DIM_SUM ? inputAShape[0] : inputAShape[1];
    auto socLongVersion = GetCurrentPlatformInfo().GetSocLongVersion();
    bool isPpMatmul =
        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P &&
         ((M >= PPMATMUL_PRIORITY_M && bias != nullptr && !transposeX1 && transposeX2 && dtype != DataType::DT_BF16) ||
          (pertokenScaleOptional != nullptr && !pertokenScaleOptional->IsEmpty())));
    ret = TransdataX1Process(isX1TransdataFlag, reformatedX1, executor, isPpMatmul);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor *matmulRet = nullptr;
    if (isA8W4F || isA8W4I) {
        // 调用l0算子QuantBatchMatmulV4进行计算
        matmulRet = l0op::QuantBatchMatmulV4(
            reformatedX1, reformatedX2, reformatedBias, reformatedpertokenScaleOptional, castedScale, reformatedYScale,
            nullptr, nullptr, yOffset, nullptr, dtype, -1, transposeX1, transposeX2, groupSize, executor);
    } else {
        // 调用l0算子QuantBatchMatmulV3进行计算
        matmulRet = l0op::QuantBatchMatmulV3(reformatedX1, reformatedX2, castedScale, offset, reformatedBias,
                                             reformatedpertokenScaleOptional, dtype, transposeX1, transposeX2,
                                             groupSize, executor);
    }

    if (isPpMatmul) {
        CHECK_RET(matmulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const aclTensor *matmulNdRet = nullptr;
        matmulNdRet = l0op::TransData(matmulRet, Format::FORMAT_ND, 0, executor);

        CHECK_RET(PostMatmulCalcProcess(matmulNdRet, std::tie(x1, x2, out), executor) == ACLNN_SUCCESS, ret);
    } else {
        CHECK_RET(PostMatmulCalcProcess(matmulRet, std::tie(x1, x2, out), executor) == ACLNN_SUCCESS, ret);
    }
    return ACLNN_SUCCESS;
}
}

aclnnStatus aclnnQuantMatmulV3GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *scale,
                                               const aclTensor *offset, const aclTensor *bias, bool transposeX1,
                                               bool transposeX2, const aclTensor *out, uint64_t *workspaceSize,
                                               aclOpExecutor **executor) {
    DEPRECATED_API_WARN_ONCE("aclnnQuantMatmulV3GetWorkspaceSize", "December 2026", "aclnnQuantMatmulV5GetWorkspaceSize");
    L2_DFX_PHASE_1(aclnnQuantMatmulV3, DFX_IN(x1, x2, scale, offset, bias), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    const aclTensor *tempPtr = nullptr;
    const aclTensor *tempYScalePtr = nullptr;
    const aclTensor *tempYOffsetPtr = nullptr;
    int64_t groupSize = 0;
    auto ret = aclnnQuantMatmulGetWorkspaceSizeCommonProcess(std::tie(x1, x2, scale),
                                                             std::tie(offset, tempPtr, bias, tempYScalePtr, tempYOffsetPtr, groupSize),
                                                             std::tie(transposeX1, transposeX2),
                                                             out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmulV4GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *scale,
                                               const aclTensor *offset, const aclTensor *pertokenScaleOptional,
                                               const aclTensor *bias, bool transposeX1, bool transposeX2,
                                               const aclTensor *out, uint64_t *workspaceSize,
                                               aclOpExecutor **executor) {
    DEPRECATED_API_WARN_ONCE("aclnnQuantMatmulV4GetWorkspaceSize", "December 2026", "aclnnQuantMatmulV5GetWorkspaceSize");
    L2_DFX_PHASE_1(aclnnQuantMatmulV4, DFX_IN(x1, x2, scale, offset, pertokenScaleOptional, bias), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    const aclTensor *tempYScalePtr = nullptr;
    const aclTensor *tempYOffsetPtr = nullptr;
    int64_t groupSize = 0;
    auto ret = aclnnQuantMatmulGetWorkspaceSizeCommonProcess(
        std::tie(x1, x2, scale), std::tie(offset, pertokenScaleOptional, bias, tempYScalePtr, tempYOffsetPtr, groupSize),
        std::tie(transposeX1, transposeX2), out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

namespace {
static op::Shape GetWeightNzShape(const aclTensor *input, bool transpose)
{
    size_t viewDimNum = input->GetViewShape().GetDimNum();
    int64_t k = transpose ? input->GetViewShape().GetDim(viewDimNum - 1)
        : input->GetViewShape().GetDim(viewDimNum - LAST_SECOND_DIM_INDEX);
    int64_t n = transpose ? input->GetViewShape().GetDim(viewDimNum - LAST_SECOND_DIM_INDEX)
        : input->GetViewShape().GetDim(viewDimNum - 1);

    int64_t nz_k0_value_trans = SelectNzK0Value(input->GetDataType());
    int64_t k1 = transpose ? CeilDiv(k, nz_k0_value_trans) : CeilDiv(k, NZ_K0_VALUE_BMM_BLOCK_NUM);
    int64_t n1 = transpose ? CeilDiv(n, NZ_K0_VALUE_BMM_BLOCK_NUM) : CeilDiv(n, nz_k0_value_trans);

    op::Shape weightNzShape;
    for (size_t i = 0; i < viewDimNum - LAST_SECOND_DIM_INDEX; i++) {
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
    weightNzShape.AppendDim(nz_k0_value_trans);
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

static const aclTensor *SetTensorToNZFormat(const aclTensor *input, op::Shape &shape, aclOpExecutor *executor)
{
    auto formatTensor = executor->CreateView(input, shape, input->GetViewOffset());
    formatTensor->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    formatTensor->SetOriginalFormat(op::Format::FORMAT_ND);
    formatTensor->SetViewShape(input->GetViewShape());
    return formatTensor;
}

bool checkNotSupportParam(
    TupleTensor mandatoryTensors, const aclTensor* pertokenScale, const aclTensor* yScale, const aclTensor* x1Offset,
    const aclTensor* yOffset, int64_t groupSize)
{
    auto& x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);

    if (x1Offset != nullptr && x1Offset->GetViewShape().GetShapeSize() != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Current version do not support x1Offset.");
        return false;
    }

    if (yOffset != nullptr && yOffset->GetViewShape().GetShapeSize() != 0
        && !isA8W4Msd(x1, x2, scale, pertokenScale)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Current version do not support yOffset.");
        return false;
    }

    if (!(isA8W4Float(x1, x2) || isMxfp8Nz(x1, x2, scale))) {
        if (yScale != nullptr && yScale->GetViewShape().GetShapeSize() != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Current version do not support yScale.");
            return false;
        }

        if (groupSize != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Current version do not support groupSize.");
            return false;
        }
    }

    return true;
}

static void SetStorageShapeForNZ(aclTensor* tensor)
{
    // storageShape的倒数第一维要放大8倍， 比如(n/32,k/16,16,4) -> (n/32,k/16,16,32)
    auto storageShape = tensor->GetStorageShape();
    auto storageShapeDim = storageShape.GetDimNum();
    storageShape[storageShapeDim - 1] *= B4_PER_B32;
    tensor->SetStorageShape(storageShape);
}

static void UnpackB32ToB4(const aclTensor* tensorB32)
{
    DataType b32Dtype = tensorB32->GetDataType();
    DataType b4Dtype = DataType::DT_INT4;
    if (b32Dtype == DataType::DT_FLOAT) {
        b4Dtype = DataType::DT_FLOAT4_E2M1;
    }

    OP_LOGD("Unpack from %s to %s start.", op::ToString(b32Dtype).GetString(), op::ToString(b4Dtype).GetString());
    auto tensorB4 = const_cast<aclTensor*>(tensorB32);
    op::Shape tensorShape = tensorB4->GetViewShape();
    op::Strides newStride = tensorB4->GetViewStrides();
    auto viewShapeDim = tensorShape.GetDimNum();
    bool transposeTensor = false;
    auto changeDimIdx = viewShapeDim - 1;
    // 轴大于等于2才判断是否转置
    if (viewShapeDim >= 2 && IsTransposeLastTwoDims(tensorB4)) {
        transposeTensor = true;
        // 转置场景扩大倒数第2维
        changeDimIdx = viewShapeDim - 2;
    }
    tensorShape[changeDimIdx] = tensorShape[changeDimIdx] * B4_PER_B32;
    tensorB4->SetViewShape(tensorShape);
    tensorB4->SetDataType(b4Dtype);
    if (IsFormatNZ(tensorB4)) {
        SetStorageShapeForNZ(tensorB4);
    }

    if (transposeTensor) {
        auto strideSize = newStride.size();
        // 转置场景，B32承载B4时strides缩小了8倍，需要放大， 即（k*n/8, 1，k/8）->(k*n, 1, k)
        newStride[strideSize - 1] *= B4_PER_B32;
        tensorB4->SetViewStrides(newStride);
    }
    OP_LOGD("Current tensor transpose status: %d.", transposeTensor);
    OP_LOGD("Unpack from %s to %s finished.", op::ToString(b32Dtype).GetString(), op::ToString(b4Dtype).GetString());
}
} // namespace

static aclnnStatus modifyScaleStorageShape(const aclTensor *scale) {
    auto scaleShape = scale->GetViewShape();
    auto scaleStorageShape = scale->GetStorageShape();    
    int64_t dimNum = scaleStorageShape.GetDimNum();
    // 1维的storage shape需要修正为2维的viewShape，
    if (dimNum == 1) {
        uint64_t viewShapeMultiply = 1;
        uint64_t viewShapeDimNum = scaleShape.GetDimNum();
        for (uint64_t i = 0; i < viewShapeDimNum; i++) {
            viewShapeMultiply *= scaleShape[i];
        }

        uint64_t storageShapeMultiply = 1;
        storageShapeMultiply *= scaleStorageShape[0];
        if (viewShapeMultiply != storageShapeMultiply) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The product of view shape dimensions %ld does not equal the product of storage shape dimensions %ld .",
                viewShapeMultiply, storageShapeMultiply);
            return ACLNN_ERR_PARAM_INVALID;
        }

        scale->SetStorageShape(scaleShape);
        OP_LOGD("modify storage shape to view shape finish.");
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus preProcessTensor(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale)
{
    if (x1->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN && x2->GetDataType() == op::DataType::DT_FLOAT) {
        UnpackB32ToB4(x2);
        // mx场景下修正x1_scale和x2_scale的1维的storage shape
        if (x1Scale != nullptr) {
            auto ret = modifyScaleStorageShape(x1Scale);
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
        }

        if (x2Scale != nullptr) {
            auto ret = modifyScaleStorageShape(x2Scale);
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
        }
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmulWeightNzGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                                                     const aclTensor *x2Scale, const aclTensor *yScale,
                                                     const aclTensor *x1Offset, const aclTensor *x2Offset,
                                                     const aclTensor *yOffset, const aclTensor *bias, bool transposeX1,
                                                     bool transposeX2, int64_t groupSize, aclTensor *out,
                                                     uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnQuantMatmulWeightNz,
                   DFX_IN(x1, x2, x1Scale, x2Scale, yScale, x1Offset, x2Offset, yOffset, bias, transposeX1, transposeX2,
                          groupSize),
                   DFX_OUT(out));

    if (!checkNotSupportParam(std::tie(x1, x2, x2Scale), x1Scale, yScale, x1Offset, yOffset, groupSize)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto ret = CheckWeightNzParamsDAV3510(x1, x2, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (x2 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QuantMatmul WeightNz do not support x2 is nullptr.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    int64_t viewDimNum = x2->GetViewShape().GetDimNum();
    if(viewDimNum < MIN_DIM_NUM_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2's view dimNum should greater than 1, but is %ld.", viewDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 修改传入tensor的format和shape
    ret = preProcessTensor(x1, x2, x1Scale, x2Scale);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (!isA8W4Int(x1, x2)) {
        transposeX2 = GetTransposeAttrValue(x2, transposeX2, false);
    }

    op::Shape weightNzShape = GetWeightNzShape(x2, transposeX2);
    if (!CheckWeightNzStorageShape(weightNzShape, x2->GetStorageShape())) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "x2'format only support NZ, but now x2's format is not NZ(Ascend affinity format). \
aclnnCalculateMatmulWeightSizeV2 and aclnnTransMatmulWeight can be used to convert the input format from ND to Ascend \
affinity format.");
      return ACLNN_ERR_PARAM_INVALID;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    x2 = SetTensorToNZFormat(x2, weightNzShape, uniqueExecutor.get());
    ret = aclnnQuantMatmulGetWorkspaceSizeCommonProcess(
        std::tie(x1, x2, x2Scale), std::tie(x2Offset, x1Scale, bias, yScale, yOffset, groupSize),
        std::tie(transposeX1, transposeX2), out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmulV3(void *workspace, uint64_t workspaceSize,
                               aclOpExecutor *executor, aclrtStream stream) {
    DEPRECATED_API_WARN_ONCE("aclnnQuantMatmulV3", "December 2026", "aclnnQuantMatmulV5");
    L2_DFX_PHASE_2(aclnnQuantMatmulV3);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnQuantMatmulV4(void *workspace, uint64_t workspaceSize,
                               aclOpExecutor *executor, aclrtStream stream) {
    DEPRECATED_API_WARN_ONCE("aclnnQuantMatmulV4", "December 2026", "aclnnQuantMatmulV5");
    L2_DFX_PHASE_2(aclnnQuantMatmulV4);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnQuantMatmulWeightNz(void *workspace, uint64_t workspaceSize,
                               aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnQuantMatmulWeightNz);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}