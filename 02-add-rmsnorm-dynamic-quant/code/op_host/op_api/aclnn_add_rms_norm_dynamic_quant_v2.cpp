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
#include "opdev/op_dfx.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

#include "../../common/stub/op_api/aclnn_kernels/cast.h"
#include "../../common/stub/op_api/aclnn_kernels/common/op_error_check.h"
#include "../../common/stub/op_api/aclnn_kernels/contiguous.h"

#include "add_rms_norm_dynamic_quant.h"
#include "aclnn_add_rms_norm_dynamic_quant_v2.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace AddRmsNormDynamicQuantV2ACLNN {
constexpr int IDX_0 = 0;
constexpr int IDX_1 = 1;
constexpr int IDX_2 = 2;
constexpr int IDX_3 = 3;
constexpr int IDX_4 = 4;
constexpr int OUTPUT_MASK_LEN = 2;
static constexpr int64_t INT4_NUMS_IN_INT32_SPACE = 8;

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST_Y_SCALE = {
    op::DataType::DT_INT8, op::DataType::DT_INT4, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool checkLastDimCompatibility(const aclTensor* outTensor, int64_t xLastDim)
{
    auto outDtype = outTensor->GetDataType();
    auto outShape = outTensor->GetViewShape();

    OP_LOGD("checkLastDimCompatibility,c_shape:%s,xlastdim %d", op::ToString(outShape).GetString(),xLastDim);

    if (outDtype == op::DataType::DT_INT8|| outDtype == op::DataType::DT_INT4){   
      return true;
    } else {
        bool isInt32 = (outDtype == op::DataType::DT_INT32);
        if (isInt32) {
            int64_t outDimNum = static_cast<int64_t>(outShape.GetDimNum());
            if (outDimNum == 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y1 out tensor has no dimensions.");
                return false;
            }
            int64_t outLastDim = outShape.GetDim(outDimNum - 1);
            OP_CHECK(
                xLastDim == outLastDim * AddRmsNormDynamicQuantV2ACLNN::INT4_NUMS_IN_INT32_SPACE,
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "For INT32 output, output last dim must be 1/8 of input last dim,"
                    " Input1 last dim is (%ld), Output last dim is (%ld).",
                    xLastDim, outLastDim),
                return false);
        }
        return true;
    }
}

static bool CheckDtypeValid(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, const aclTensor* betaOptional, const aclTensor* y1Out,
    const aclTensor* y2Out, const aclTensor* xOut, const aclTensor* scale1Out, const aclTensor* scale2Out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gamma, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    if (nullptr != betaOptional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(betaOptional, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false); // 校验可选beta
    }

    if (nullptr != smoothScale1Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(smoothScale1Optional, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    }

    if (nullptr != smoothScale2Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(smoothScale2Optional, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(y2Out, ASCEND910B_DTYPE_SUPPORT_LIST_Y_SCALE, return false); // Mandatory output
    OP_CHECK_DTYPE_NOT_SUPPORT(y1Out, ASCEND910B_DTYPE_SUPPORT_LIST_Y_SCALE, return false);

    int64_t xDimNum = static_cast<int64_t>(x1->GetViewShape().GetDimNum());
    int64_t xLastDim = x1->GetViewShape().GetDim(xDimNum - 1);
    if (!checkLastDimCompatibility(y1Out, xLastDim)) {
        return false;
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(xOut, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false);

    OP_CHECK_DTYPE_NOT_MATCH(scale1Out, op::DataType::DT_FLOAT, return false);
    OP_CHECK_DTYPE_NOT_MATCH(scale2Out, op::DataType::DT_FLOAT, return false);

    return true;
}

static bool CheckFlag(
    const aclTensor* smoothScale1Optional, const aclTensor* smoothScale2Optional, const aclBoolArray* outputMask)
{
    if (outputMask != nullptr && outputMask->Size() == OUTPUT_MASK_LEN) {
        // 只能为nullptr或者长度为2的数组
        bool outquant1 = (*outputMask)[0];
        bool outquant2 = (*outputMask)[1];
        if (smoothScale1Optional != nullptr && !outquant1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SmoothScale1Optional is not nullptr but outputMask[0] is False.");
            return false;
        }
        if (smoothScale2Optional != nullptr && !outquant2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SmoothScale2Optional is not nullptr but outputMask[1] is False.");
            return false;
        }
        // 不能两个全部为false
        if (!outquant1 && !outquant2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support both outputMask[0] and outputMask[1] are False.");
            return false;
        }
    } else if (outputMask != nullptr && outputMask->Size() != OUTPUT_MASK_LEN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The length of outputMask must be 2, bug got %zu.", outputMask->Size());
        return false;
    } else {
        // 老场景不支持只有smooth2
        if (smoothScale1Optional == nullptr && smoothScale2Optional != nullptr) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "When outputMask is unavailable, it is not supported only smoothScale2Optional without "
                "smoothScale1Optional.");
            return false;
        }
    }
    return true;
}

static bool CheckNotNull(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, aclTensor* y1Out, aclTensor* y2Out,
    const aclTensor* xOut, const aclTensor* scale1Out, const aclTensor* scale2Out)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(gamma, return false);
    OP_CHECK_NULL(y2Out, return false);
    OP_CHECK_NULL(y1Out, return false);
    OP_CHECK_NULL(xOut, return false);
    OP_CHECK_NULL(scale1Out, return false);
    OP_CHECK_NULL(scale2Out, return false);
    return true;
}
static aclnnStatus CheckParams(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, const aclTensor* betaOptional, const aclBoolArray* outputMask,
    aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut, aclTensor* scale1Out, aclTensor* scale2Out)
{
    // 1. 检查必选输入/输出是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, gamma, y1Out, y2Out, xOut, scale1Out, scale2Out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入/输出的数据类型是否合法
    CHECK_RET(
        CheckDtypeValid(
            x1, x2, gamma, smoothScale1Optional, smoothScale2Optional, betaOptional, y1Out, y2Out, xOut, scale1Out,
            scale2Out),
        ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckFlag(smoothScale1Optional, smoothScale2Optional, outputMask), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace AddRmsNormDynamicQuantV2ACLNN

aclnnStatus AddRmsNormDynamicQuantV2Int42Int32PackedTensor(
    const aclTensor* y, const aclTensor*& outTensor, aclOpExecutor* executor)
{
    // if outType is int32, pack output
    auto viewShape = y->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    viewShape[viewShapeDim - 1] /= AddRmsNormDynamicQuantV2ACLNN::INT4_NUMS_IN_INT32_SPACE;
    auto outTemp = executor->CreateView(y, viewShape, y->GetViewOffset());
    CHECK_RET(outTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);

    outTemp->SetDataType(DataType::DT_INT32);
    outTensor = outTemp;
    OP_LOGD("AddRmsNormDynamicQuantV2ACLNN output real dtype is int4, pack to int32 to out.");

    return ACLNN_SUCCESS;
}

aclnnStatus ComputeAddRmsNormDynamicQuantV2(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, const aclTensor* betaOptional, double epsilon,
    const aclBoolArray* outputMask, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut, aclTensor* scale1Out,
    aclTensor* scale2Out, aclOpExecutor* executor)
{
    aclTensor* y1ComputeOut = nullptr;
    aclTensor* y2ComputeOut = nullptr;
    aclTensor* xComputeOut = nullptr;
    int32_t yType = y1Out->GetDataType();
    if (yType == op::DataType::DT_INT32) {
        yType = op::DataType::DT_INT4;
    }
    auto addRmsNormQuantOuts = l0op::AddRmsNormDynamicQuant(
        x1, x2, gamma, smoothScale1Optional, smoothScale2Optional, betaOptional, epsilon, outputMask, yType, scale1Out,
        scale2Out, executor);
    y1ComputeOut = std::get<AddRmsNormDynamicQuantV2ACLNN::IDX_0>(addRmsNormQuantOuts);
    y2ComputeOut = std::get<AddRmsNormDynamicQuantV2ACLNN::IDX_1>(addRmsNormQuantOuts);
    xComputeOut = std::get<AddRmsNormDynamicQuantV2ACLNN::IDX_2>(addRmsNormQuantOuts);

    const aclTensor* out1Tensor = y1ComputeOut;
    const aclTensor* out2Tensor = y2ComputeOut;

    // 不支持空Tensor
    CHECK_RET(y1ComputeOut != nullptr && y2ComputeOut != nullptr && xComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (yType == op::DataType::DT_INT4) {
        bool processOut1 = (outputMask == nullptr) || (*outputMask)[0];
        bool processOut2 = (outputMask == nullptr) || (*outputMask)[1];
        if (processOut1) {
            auto ret = AddRmsNormDynamicQuantV2Int42Int32PackedTensor(y1ComputeOut, out1Tensor, executor);
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
            auto viewCopyY1Result = l0op::ViewCopy(out1Tensor, y1Out, executor);
            CHECK_RET(viewCopyY1Result != nullptr, ACLNN_ERR_INNER_NULLPTR);       
        }

        if (processOut2) {
            auto ret = AddRmsNormDynamicQuantV2Int42Int32PackedTensor(y2ComputeOut, out2Tensor, executor);
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
            auto viewCopyY2Result = l0op::ViewCopy(out2Tensor, y2Out, executor);
            CHECK_RET(viewCopyY2Result != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    } else {
        // 将 y1ComputeOut 的结果拷贝到 y1 上
        auto viewCopyY1Result = l0op::ViewCopy(y1ComputeOut, y1Out, executor);
        CHECK_RET(viewCopyY1Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将 y2ComputeOut 的结果拷贝到 y2 上
        auto viewCopyY2Result = l0op::ViewCopy(y2ComputeOut, y2Out, executor);
        CHECK_RET(viewCopyY2Result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 将 xComputeOut 的结果拷贝到 x 上
    auto viewCopyXResult = l0op::ViewCopy(xComputeOut, xOut, executor);
    CHECK_RET(viewCopyXResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

const aclTensor* ContiguousTensor(const aclTensor* opt, aclOpExecutor* executor)
{
    if (nullptr == opt) {
        return nullptr;
    }
    return l0op::Contiguous(opt, executor);
}

aclnnStatus aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, const aclTensor* betaOptional, double epsilon,
    const aclBoolArray* outputMask, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut, aclTensor* scale1Out,
    aclTensor* scale2Out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("Enter aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize.");
    L2_DFX_PHASE_1(
        aclnnAddRmsNormDynamicQuantV2,
        DFX_IN(x1, x2, gamma, smoothScale1Optional, smoothScale2Optional, betaOptional, epsilon, outputMask),
        DFX_OUT(y1Out, y2Out, xOut, scale1Out, scale2Out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = AddRmsNormDynamicQuantV2ACLNN::CheckParams(
        x1, x2, gamma, smoothScale1Optional, smoothScale2Optional, betaOptional, outputMask, y1Out, y2Out, xOut,
        scale1Out, scale2Out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 支持空tensor
    bool hasEmptyTensor = x1->IsEmpty() || gamma->IsEmpty() || y2Out->IsEmpty();
    if (hasEmptyTensor) {
        OP_LOGW("Got empty tensor in aclnnAddRmsNormQuantV2!");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入转换成连续的tensor，可选输入不做判空校验
    auto x1Cont = l0op::Contiguous(x1, uniqueExecutor.get());
    auto x2Cont = l0op::Contiguous(x2, uniqueExecutor.get());
    auto gammaCont = l0op::Contiguous(gamma, uniqueExecutor.get());

    CHECK_RET(x1Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(x2Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gammaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto s1Cont = ContiguousTensor(smoothScale1Optional, uniqueExecutor.get());
    auto s2Cont = ContiguousTensor(smoothScale2Optional, uniqueExecutor.get());
    auto zCont = ContiguousTensor(betaOptional, uniqueExecutor.get());

    ret = ComputeAddRmsNormDynamicQuantV2(
        x1Cont, x2Cont, gammaCont, s1Cont, s2Cont, zCont, epsilon, outputMask, y1Out, y2Out, xOut, scale1Out, scale2Out,
        uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    OP_LOGD("Finish aclnnAddRmsNormQuantV2GetWorkspaceSize.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddRmsNormDynamicQuantV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddRmsNormDynamicQuantV2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
