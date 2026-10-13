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
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "../../common/inc/op_api/aclnn_util.h"

#include "../../common/stub/op_api/aclnn_kernels/cast.h"
#include "../../common/stub/op_api/aclnn_kernels/common/op_error_check.h"
#include "../../common/stub/op_api/aclnn_kernels/contiguous.h"

#include "add_rms_norm_dynamic_quant.h"
#include "aclnn_add_rms_norm_dynamic_quant.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr int IDX_INT8 = 2;
constexpr int IDX_HIFLOAT8 = 34;
constexpr int IDX_FLOAT8_E5M2 = 35;
constexpr int IDX_FLOAT8_E4M3FN = 36;
static std::map<op::DataType, int> dstTypeMap = {{op::DataType::DT_INT8, IDX_INT8}, 
       {op::DataType::DT_HIFLOAT8, IDX_HIFLOAT8}, {op::DataType::DT_FLOAT8_E5M2, IDX_FLOAT8_E5M2}, 
       {op::DataType::DT_FLOAT8_E4M3FN, IDX_FLOAT8_E4M3FN}};

namespace AddRmsNormDynamicQuantACLNN {
constexpr int IDX_0 = 0;
constexpr int IDX_1 = 1;
constexpr int IDX_2 = 2;
constexpr int IDX_3 = 3;
constexpr int IDX_4 = 4;

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> REGBASE_DTYPE_SUPPORT_LIST_Y = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_HIFLOAT8, op::DataType::DT_INT8};
static bool CheckNotNull(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, aclTensor* y1Out, aclTensor* y2Out,
    const aclTensor* xOut, const aclTensor* scale1Out, const aclTensor* scale2Out)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(gamma, return false);
    OP_CHECK_NULL(y1Out, return false);
    OP_CHECK_NULL(y2Out, return false);
    OP_CHECK_NULL(xOut, return false);
    OP_CHECK_NULL(scale1Out, return false);
    OP_CHECK_NULL(scale2Out, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, const aclTensor* y1Out, const aclTensor* y2Out, const aclTensor* xOut,
    const aclTensor* scale1Out, const aclTensor* scale2Out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gamma, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false);

    if (nullptr != smoothScale1Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(smoothScale1Optional, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    }
    if (nullptr != smoothScale2Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(smoothScale2Optional, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(y1Out, REGBASE_DTYPE_SUPPORT_LIST_Y, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y2Out, REGBASE_DTYPE_SUPPORT_LIST_Y, return false); // Mandatory output
    OP_CHECK_DTYPE_NOT_SAME(y1Out, y2Out, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(xOut, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false);

    OP_CHECK_DTYPE_NOT_MATCH(scale1Out, op::DataType::DT_FLOAT, return false);
    OP_CHECK_DTYPE_NOT_MATCH(scale2Out, op::DataType::DT_FLOAT, return false);

    return true;
}

static bool CheckShapeValid(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional)
{
    // 检查 x1 和 x2 shape 是否一致
    auto x1Shape = x1->GetViewShape();
    auto x2Shape = x2->GetViewShape();
    if (x1Shape != x2Shape) {
        OP_LOGE(ACLNN_ERR_INNER_TILING_ERROR, "x1 and x2 shape mismatch.");
        return false;
    }

    // 检查 gamma shape 是否为1维
    auto gammaShape = gamma->GetViewShape();
    if (gammaShape.GetDimNum() != 1) {
        OP_LOGE(ACLNN_ERR_INNER_TILING_ERROR, "gamma shape dim num should be 1.");
        return false;
    }

    // 检查 gammaShape 是否与 x1 最后一维一致
    auto x1DimNum = x1Shape.GetDimNum();
    if (x1Shape.GetDim(x1DimNum - 1) != gammaShape.GetDim(0)) {
        OP_LOGE(ACLNN_ERR_INNER_TILING_ERROR, "gammaShape isn't consistent with the last dimension of x1.");
        return false;
    }

    // 检查 smoothScale shape 是否与 gamma 一致
    if (smoothScale1Optional != nullptr) {
        auto smooth1Shape = smoothScale1Optional->GetViewShape();
        if (smooth1Shape != gammaShape) {
            OP_LOGE(ACLNN_ERR_INNER_TILING_ERROR, "smoothScale1 shape mismatch with gamma.");
            return false;
        }
    }
    if (smoothScale2Optional != nullptr) {
        auto smooth2Shape = smoothScale2Optional->GetViewShape();
        if (smooth2Shape != gammaShape) {
            OP_LOGE(ACLNN_ERR_INNER_TILING_ERROR, "smoothScale2 shape mismatch with gamma.");
            return false;
        }
    }

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut, aclTensor* scale1Out,
    aclTensor* scale2Out)
{
    // 1. 检查必选输入/输出是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, gamma, y1Out, y2Out, xOut, scale1Out, scale2Out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入/输出的数据类型是否合法
    CHECK_RET(
        CheckDtypeValid(
            x1, x2, gamma, smoothScale1Optional, smoothScale2Optional, y1Out, y2Out, xOut, scale1Out, scale2Out),
        ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入 shape 是否合法
    CHECK_RET(
        CheckShapeValid(x1, x2, gamma, smoothScale1Optional, smoothScale2Optional),
        ACLNN_ERR_INNER_TILING_ERROR);

    return ACLNN_SUCCESS;
}
} // namespace AddRmsNormDynamicQuantACLNN

aclnnStatus ComputeAddRmsNormDynamicQuant(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, double epsilon, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut,
    aclTensor* scale1Out, aclTensor* scale2Out, aclOpExecutor* executor)
{
    aclTensor* y1ComputeOut = nullptr;
    aclTensor* y2ComputeOut = nullptr;
    aclTensor* xComputeOut = nullptr;

    int dstType = dstTypeMap[y1Out->GetDataType()];

    auto addRmsNormQuantOuts = l0op::AddRmsNormDynamicQuant(
        x1, x2, gamma, smoothScale1Optional, smoothScale2Optional, nullptr, epsilon, nullptr, dstType, scale1Out,
        scale2Out, executor);
    y1ComputeOut = std::get<AddRmsNormDynamicQuantACLNN::IDX_0>(addRmsNormQuantOuts);
    y2ComputeOut = std::get<AddRmsNormDynamicQuantACLNN::IDX_1>(addRmsNormQuantOuts);
    xComputeOut = std::get<AddRmsNormDynamicQuantACLNN::IDX_2>(addRmsNormQuantOuts);

    // 不支持空Tensor
    CHECK_RET(y1ComputeOut != nullptr && y2ComputeOut != nullptr && xComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将 y1ComputeOut 结果拷贝到 y1 上
    auto viewCopyY1Result = l0op::ViewCopy(y1ComputeOut, y1Out, executor);
    CHECK_RET(viewCopyY1Result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 将 xComputeOut 结果拷贝到 x 上
    auto viewCopyXResult = l0op::ViewCopy(xComputeOut, xOut, executor);
    CHECK_RET(viewCopyXResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyY2Result = l0op::ViewCopy(y2ComputeOut, y2Out, executor);
    CHECK_RET(viewCopyY2Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

const aclTensor* ContiguousX(const aclTensor* opt, aclOpExecutor* executor)
{
    if (nullptr == opt) {
        return nullptr;
    }
    return l0op::Contiguous(opt, executor);
}

aclnnStatus aclnnAddRmsNormDynamicQuantGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, double epsilon, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut,
    aclTensor* scale1Out, aclTensor* scale2Out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("Enter aclnnAddRmsNormDynamicQuantGetWorkspaceSize.");
    L2_DFX_PHASE_1(
        aclnnAddRmsNormDynamicQuant, DFX_IN(x1, x2, gamma, smoothScale1Optional, smoothScale2Optional, epsilon),
        DFX_OUT(y1Out, y2Out, xOut, scale1Out, scale2Out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = AddRmsNormDynamicQuantACLNN::CheckParams(
        x1, x2, gamma, smoothScale1Optional, smoothScale2Optional, y1Out, y2Out, xOut, scale1Out, scale2Out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 支持空tensor
    bool hasEmptyTensor = x1->IsEmpty() || gamma->IsEmpty() || y2Out->IsEmpty();
    bool hasReduceEmptyTensor = gamma->IsEmpty();
    // 非reduce轴为0处理
    if (hasEmptyTensor) {
        OP_LOGW("Got empty tensor in aclnnAddRmsNormQuant!");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入转换成连续的tensor，可选输入不做判空校验
    auto x1Cont = l0op::Contiguous(x1, uniqueExecutor.get());
    auto x2Cont = l0op::Contiguous(x2, uniqueExecutor.get());
    auto gammaCont = l0op::Contiguous(gamma, uniqueExecutor.get());

    CHECK_RET(gammaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(x2Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(x1Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto s1Cont = ContiguousX(smoothScale1Optional, uniqueExecutor.get());
    auto s2Cont = ContiguousX(smoothScale2Optional, uniqueExecutor.get());

    ret = ComputeAddRmsNormDynamicQuant(
        x1Cont, x2Cont, gammaCont, s1Cont, s2Cont, epsilon, y1Out, y2Out, xOut, scale1Out, scale2Out,
        uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    OP_LOGD("Finish aclnnAddRmsNormQuantGetWorkspaceSize.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddRmsNormDynamicQuant(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddRmsNormDynamicQuant);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
