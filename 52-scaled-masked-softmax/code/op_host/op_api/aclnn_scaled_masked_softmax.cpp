/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_scaled_masked_softmax.h"

#include "../../common/stub/op_api/aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "../../common/inc/op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int32_t INPUT_DIM_NUM = 4;
constexpr int32_t D_LIMIT = 4096;
constexpr int32_t D_LIMIT_D = 8192;

static const std::initializer_list<op::DataType> SOFTMAX_X_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> MASK_DTYPE_SUPPORT_LIST = {op::DataType::DT_BOOL};

extern aclnnStatus aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize(
    const aclTensor* x, const aclTensor* mask, double scale, bool fixedTriuMask, aclTensor* y, uint64_t* workspaceSize,
    aclOpExecutor** executor);

extern aclnnStatus aclnnInnerScaledMaskedSoftmaxV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

static bool CheckNotNull(const aclTensor* x, const aclTensor* mask, aclTensor* y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(mask, return false);
    OP_CHECK_NULL(y, return false);

    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* mask, const aclTensor* y)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, SOFTMAX_X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(mask, MASK_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, SOFTMAX_X_DTYPE_SUPPORT_LIST, return false);

    OP_CHECK_DTYPE_NOT_SAME(x, y, return false);

    return true;
}

static bool CheckTensorDim(const aclTensor* x, const aclTensor* mask, const aclTensor* y)
{
    OP_CHECK_WRONG_DIMENSION(x, INPUT_DIM_NUM, return false);
    OP_CHECK_WRONG_DIMENSION(mask, INPUT_DIM_NUM, return false);
    OP_CHECK_WRONG_DIMENSION(y, INPUT_DIM_NUM, return false);

    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* mask)
{
    if ((x->GetViewShape().GetDim(DIM_0) != mask->GetViewShape().GetDim(DIM_0)) &&
        (mask->GetViewShape().GetDim(DIM_0) != 1)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "When x%s and mask%s are different in the first dimensions, the broadcast relation needs to be satisfied",
            op::ToString(x->GetViewShape()).GetString(), op::ToString(mask->GetViewShape()).GetString());
        return false;
    }
    if ((x->GetViewShape().GetDim(DIM_1) != mask->GetViewShape().GetDim(DIM_1)) &&
        (mask->GetViewShape().GetDim(DIM_1) != 1)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "When x%s and mask%s are different in the second dimensions, the broadcast relation needs to be satisfied",
            op::ToString(x->GetViewShape()).GetString(), op::ToString(mask->GetViewShape()).GetString());
        return false;
    }

    if ((x->GetViewShape().GetDim(DIM_2) != mask->GetViewShape().GetDim(DIM_2)) ||
        (x->GetViewShape().GetDim(DIM_3) != mask->GetViewShape().GetDim(DIM_3))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x and mask should be same in dim3 and dim4");
        return false;
    }

    int32_t dDimLimit = D_LIMIT;


    if ((x->GetViewShape().GetDim(DIM_3) <= 0) || (x->GetViewShape().GetDim(DIM_3) > dDimLimit)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected x and mask dim4 in range of (0, 4096].");
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* mask, aclTensor* y)
{
    CHECK_RET(CheckNotNull(x, mask, y), ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(CheckDtypeValid(x, mask, y), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckTensorDim(x, mask, y), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShape(x, mask), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnScaledMaskedSoftmaxGetWorkspaceSize(
    const aclTensor* x, const aclTensor* mask, double scale, bool fixedTriuMask, aclTensor* y, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    auto ret_param = CheckParams(x, mask, y);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);
    if (fixedTriuMask) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the param fixedTriuMask only suppport false.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto result = aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize(x, mask, scale, false, y, workspaceSize, executor);
    return result;
}

aclnnStatus aclnnScaledMaskedSoftmax(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnInnerScaledMaskedSoftmaxV2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif