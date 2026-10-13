/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn/aclnn_base.h"
#include "op_api/op_api_def.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"

#include "add_rms_norm_quant.h"
#include "aclnn_add_rms_norm_quant.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr int IDX_INT8_QUANT = 2;
constexpr int IDX_HIFLOAT8_QUANT = 34;
constexpr int IDX_FLOAT8_E5M2_QUANT = 35;
constexpr int IDX_FLOAT8_E4M3FN_QUANT = 36;
static std::map<op::DataType, int> dstTypeMapQuant = {{op::DataType::DT_INT8, IDX_INT8_QUANT}, 
       {op::DataType::DT_HIFLOAT8, IDX_HIFLOAT8_QUANT}, {op::DataType::DT_FLOAT8_E5M2, IDX_FLOAT8_E5M2_QUANT}, 
       {op::DataType::DT_FLOAT8_E4M3FN, IDX_FLOAT8_E4M3FN_QUANT}};
namespace AddRmsNormQuantACLNN {
constexpr int IDX_0 = 0;
constexpr int IDX_1 = 1;
constexpr int IDX_2 = 2;
const size_t MIN_SUPPORT_DIMS_NUMS = 1;

static const std::initializer_list<op::DataType> REGBASE_DTYPE_SUPPORT_LIST_X_SCALE = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> REGBASE_DTYPE_SUPPORT_LIST_ZEROPOINT = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> REGBASE_DTYPE_SUPPORT_LIST_Y = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_HIFLOAT8, op::DataType::DT_INT8};

static bool CheckNotNull(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* scales1, aclTensor* y1Out,
    aclTensor* y2Out, const aclTensor* xOut)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(gamma, return false);
    OP_CHECK_NULL(scales1, return false);
    OP_CHECK_NULL(y1Out, return false);
    OP_CHECK_NULL(y2Out, return false);
    OP_CHECK_NULL(xOut, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* scales1,
    const aclTensor* scales2Optional, const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional,
    aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, REGBASE_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, REGBASE_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gamma, REGBASE_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(scales1, REGBASE_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    if (nullptr != scales2Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scales2Optional, REGBASE_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    }
    if (nullptr != zeroPoints1Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(zeroPoints1Optional, REGBASE_DTYPE_SUPPORT_LIST_ZEROPOINT, return false);
    }
    if (nullptr != zeroPoints2Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(zeroPoints2Optional, REGBASE_DTYPE_SUPPORT_LIST_ZEROPOINT, return false);
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(y1Out, REGBASE_DTYPE_SUPPORT_LIST_Y, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y2Out, REGBASE_DTYPE_SUPPORT_LIST_Y, return false); // Mandatory output
    OP_CHECK_DTYPE_NOT_SAME(y1Out, y2Out, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(xOut, REGBASE_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    return true;
}

static bool CheckShapeDim(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* scales1,
    const aclTensor* scales2Optional, const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional,
    aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut)
{
    OP_CHECK_MAX_DIM(x1, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(x2, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(gamma, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(scales1, MAX_SUPPORT_DIMS_NUMS, return false);

    OP_CHECK_MIN_DIM(x1, MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(x2, MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(gamma, MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(scales1, MIN_SUPPORT_DIMS_NUMS, return false);

    CHECK_RET(CheckDims(x1), false);
    CHECK_RET(CheckDims(x2), false);
    CHECK_RET(CheckDims(gamma), false);
    CHECK_RET(CheckDims(scales1), false);

    if (nullptr != scales2Optional) {
        OP_CHECK_MAX_DIM(scales2Optional, MAX_SUPPORT_DIMS_NUMS, return false);
        OP_CHECK_MIN_DIM(scales2Optional, MIN_SUPPORT_DIMS_NUMS, return false);
        CHECK_RET(CheckDims(scales2Optional), false);
    }
    if (nullptr != zeroPoints1Optional) {
        OP_CHECK_MAX_DIM(zeroPoints1Optional, MAX_SUPPORT_DIMS_NUMS, return false);
        OP_CHECK_MIN_DIM(zeroPoints1Optional, MIN_SUPPORT_DIMS_NUMS, return false);
        CHECK_RET(CheckDims(zeroPoints1Optional), false);
    }
    if (nullptr != zeroPoints2Optional) {
        OP_CHECK_MAX_DIM(zeroPoints2Optional, MAX_SUPPORT_DIMS_NUMS, return false);
        OP_CHECK_MIN_DIM(zeroPoints2Optional, MIN_SUPPORT_DIMS_NUMS, return false);
        CHECK_RET(CheckDims(zeroPoints2Optional), false);
    }
    OP_CHECK_MAX_DIM(y1Out, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(y2Out, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(xOut, MAX_SUPPORT_DIMS_NUMS, return false);

    OP_CHECK_MIN_DIM(y1Out, MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(y2Out, MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(xOut, MIN_SUPPORT_DIMS_NUMS, return false);

    CHECK_RET(CheckDims(y1Out), false);
    CHECK_RET(CheckDims(y2Out), false);
    CHECK_RET(CheckDims(xOut), false);
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* scales1,
    const aclTensor* scales2Optional, const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional,
    aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut)
{
    // 1. 检查必选输入/输出是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, gamma, scales1, y1Out, y2Out, xOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入/输出的数据类型是否合法
    CHECK_RET(
        CheckDtypeValid(
            x1, x2, gamma, scales1, scales2Optional, zeroPoints1Optional, zeroPoints2Optional, y1Out, y2Out, xOut),
        ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入/输出的shape大小
    CHECK_RET(
        CheckShapeDim(
            x1, x2, gamma, scales1, scales2Optional, zeroPoints1Optional, zeroPoints2Optional, y1Out, y2Out, xOut),
        ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}
} // namespace AddRmsNormQuantACLNN

aclnnStatus ComputeAddRmsNormQuant(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* scales1,
    const aclTensor* scales2Optional, const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional,
    int64_t axis, double epsilon, bool divMode, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut,
    aclOpExecutor* executor)
{
    aclTensor* y1ComputeOut = nullptr;
    aclTensor* y2ComputeOut = nullptr;
    aclTensor* xComputeOut = nullptr;
    bool isDual = (nullptr != scales2Optional);
    
    int dstType = dstTypeMapQuant[y1Out->GetDataType()];

    auto addRmsNormQuantOuts = l0op::AddRmsNormQuant(
        x1, x2, gamma, scales1, scales2Optional, zeroPoints1Optional, zeroPoints2Optional, nullptr, axis, epsilon,
        divMode, dstType, executor);
    y1ComputeOut = std::get<AddRmsNormQuantACLNN::IDX_0>(addRmsNormQuantOuts);
    y2ComputeOut = std::get<AddRmsNormQuantACLNN::IDX_1>(addRmsNormQuantOuts);
    xComputeOut = std::get<AddRmsNormQuantACLNN::IDX_2>(addRmsNormQuantOuts);

    // 不支持空Tensor
    CHECK_RET(y1ComputeOut != nullptr && y2ComputeOut != nullptr && xComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将 y1ComputeOut 结果拷贝到 y1 上
    auto viewCopyY1Result = l0op::ViewCopy(y1ComputeOut, y1Out, executor);
    CHECK_RET(viewCopyY1Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (isDual) {
        // 将 y2ComputeOut 结果拷贝到 y2 上
        auto viewCopyY2Result = l0op::ViewCopy(y2ComputeOut, y2Out, executor);
        CHECK_RET(viewCopyY2Result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 将 xComputeOut 结果拷贝到 x 上
    auto viewCopyXResult = l0op::ViewCopy(xComputeOut, xOut, executor);
    CHECK_RET(viewCopyXResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

const aclTensor* GetTensorContiguous(const aclTensor* opt, aclOpExecutor* executor)
{
    if (nullptr == opt) {
        return nullptr;
    }
    return l0op::Contiguous(opt, executor);
}

aclnnStatus aclnnAddRmsNormQuantGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* scales1,
    const aclTensor* scales2Optional, const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional,
    int64_t axis, double epsilon, bool divMode, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("Enter aclnnAddRmsNormQuantGetWorkspaceSize.");
    L2_DFX_PHASE_1(
        aclnnAddRmsNormQuant,
        DFX_IN(
            x1, x2, gamma, scales1, scales2Optional, zeroPoints1Optional, zeroPoints2Optional, axis, epsilon, divMode),
        DFX_OUT(y1Out, y2Out, xOut));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = AddRmsNormQuantACLNN::CheckParams(
        x1, x2, gamma, scales1, scales2Optional, zeroPoints1Optional, zeroPoints2Optional, y1Out, y2Out, xOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 支持空tensor
    bool hasEmptyTensor = x1->IsEmpty() || gamma->IsEmpty() || y2Out->IsEmpty();
    if (hasEmptyTensor) {
        OP_LOGW("Got empty tensor in aclnnAddRmsNormQuant!");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入转换成连续的tensor，可选输入不做判空校验
    auto x1ContQuant = l0op::Contiguous(x1, uniqueExecutor.get());
    auto x2ContQuant = l0op::Contiguous(x2, uniqueExecutor.get());
    auto gammaContQuant = l0op::Contiguous(gamma, uniqueExecutor.get());

    CHECK_RET(x1ContQuant != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(x2ContQuant != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gammaContQuant != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto s1Cont = GetTensorContiguous(scales1, uniqueExecutor.get());
    auto s2Cont = GetTensorContiguous(scales2Optional, uniqueExecutor.get());
    auto z1Cont = GetTensorContiguous(zeroPoints1Optional, uniqueExecutor.get());
    auto z2Cont = GetTensorContiguous(zeroPoints2Optional, uniqueExecutor.get());

    ret = ComputeAddRmsNormQuant(
        x1ContQuant, x2ContQuant, gammaContQuant, s1Cont, s2Cont, z1Cont, z2Cont, axis, epsilon, divMode, y1Out, y2Out, xOut,
        uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    OP_LOGD("Finish aclnnAddRmsNormQuantGetWorkspaceSize.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddRmsNormQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddRmsNormQuant);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
