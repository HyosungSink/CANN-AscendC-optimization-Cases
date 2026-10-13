/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include "aclnn_dynamic_quant.h"
#include "quant/dynamic_quant_v2/op_host/op_api/dynamic_quant_v2.h"
#include "aclnn_dynamic_quant_v3.h"
#include "dynamic_quant.h"
#include "quant/dynamic_quant_v2/op_host/op_api/aclnn_dynamic_quant_v2.h"
#include "level0/fault_injection.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

#include <dlfcn.h>

#include <new>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "op_api/aclnn_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif
namespace {
static constexpr int64_t INT4_NUMS_IN_INT32_SPACE = 8;
static constexpr int64_t INT4_NUMS_IN_INT8_SPACE = 2;
static constexpr int64_t NUM_TWO = 2;
static constexpr int64_t DIM_MAX = 9;
using DtypeCheck = std::initializer_list<op::DataType>;

static const std::initializer_list<DataType> EMPTY_LIST = {};

static const std::initializer_list<op::DataType> INPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> INPUT_DTYPE_SUPPORT_LIST_FP16 = {op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> GROUP_INDEX_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT4, op::DataType::DT_INT8, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST_INT8 = {op::DataType::DT_INT8};

static const std::initializer_list<op::DataType> INPUT_DTYPE_EMPTY_LIST = {};

static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST_INT_WITH_FP_QUANT = {
    op::DataType::DT_INT4,     op::DataType::DT_INT8,        op::DataType::DT_INT32,
    op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN};

struct DynamicQuantParams {
    const aclTensor* x = nullptr;
    const aclTensor* smoothScales = nullptr;
    const aclTensor* groupIndex = nullptr;
    int64_t dstType;
    bool isSymmetrical;
    const char* quantMode;
    const aclTensor* y = nullptr;
    const aclTensor* scale = nullptr;
    const aclTensor* offset = nullptr;
};

static inline const std::initializer_list<op::DataType>& GetDtypeSupportListByNpuArch()
{
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        return INPUT_DTYPE_SUPPORT_LIST;
    }
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    switch (npuArch) {
        case NpuArch::DAV_2201: {
            return INPUT_DTYPE_SUPPORT_LIST;
        }
        case NpuArch::DAV_1001:
        case NpuArch::DAV_2002: {
            return INPUT_DTYPE_SUPPORT_LIST_FP16;
        }
        default: {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %u is not implemented", static_cast<uint32_t>(npuArch));
            return EMPTY_LIST;
        }
    }
}

static inline const std::initializer_list<op::DataType>& GetOutputSupportListByNpuArch()
{
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        return OUTPUT_DTYPE_SUPPORT_LIST_INT_WITH_FP_QUANT;
    }
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    switch (npuArch) {
        case NpuArch::DAV_2201: {
            return OUTPUT_DTYPE_SUPPORT_LIST;
        }
        case NpuArch::DAV_1001:
        case NpuArch::DAV_2002: {
            return OUTPUT_DTYPE_SUPPORT_LIST_INT8;
        }
        default: {
            OP_LOGE(
                ACLNN_ERR_RUNTIME_ERROR, "API aclnnDynamicQuant support for %u is not implemented",
                static_cast<uint32_t>(npuArch));
            return EMPTY_LIST;
        }
    }
}

static aclnnStatus CheckNotNull(const DynamicQuantParams& dynamicQuantParams)
{
    CHECK_COND(dynamicQuantParams.x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x must not be nullptr.");
    CHECK_COND(dynamicQuantParams.y != nullptr, ACLNN_ERR_PARAM_NULLPTR, "y must not be nullptr.");
    CHECK_COND(dynamicQuantParams.scale != nullptr, ACLNN_ERR_PARAM_NULLPTR, "scale must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOpDim(const op::Shape& shape1, const op::Shape& shape2, uint32_t shape1Dim, uint32_t shape2Dim)
{
    if (shape1Dim != shape2Dim) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    for (uint32_t i = 0; i < shape1Dim; i++) {
        if (shape1.GetDim(i) != shape2.GetDim(i)) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInt32OutputShape(
    const op::Shape& shape1, const op::Shape& shape2, uint32_t shape1Dim, uint32_t shape2Dim)
{
    if (shape1Dim != shape2Dim) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    int64_t dimInput = shape1.GetDim(shape1Dim - 1);
    int64_t dimOutput = shape2.GetDim(shape2Dim - 1);
    // check last dim
    if (dimInput != (dimOutput * INT4_NUMS_IN_INT32_SPACE)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "If output dtype is int32, "
            "the last dimension of output must be 1/8 of the last dimension of input, "
            "the last dim of input is (%ld), the last dim of output is (%ld).",
            dimInput, dimOutput);
        return ACLNN_ERR_PARAM_INVALID;
    }

    for (int64_t i = 0; i < shape1Dim - 1; i++) {
        if (shape1.GetDim(i) != shape2.GetDim(i)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "The (%ld)-th dim of input must be the same as (%ld)-th dim of output, "
                "where input is (%ld) and output is (%ld).",
                i, i, shape1.GetDim(i), shape2.GetDim(i));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckAttr(const DynamicQuantParams& dynamicQuantParams)
{
    if (dynamicQuantParams.isSymmetrical && dynamicQuantParams.offset != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "If isSymmetrical is True, offset must be nullptr.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!dynamicQuantParams.isSymmetrical && dynamicQuantParams.offset == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "If isSymmetrical is False, offset can't be nullptr.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const std::string mode = std::string(dynamicQuantParams.quantMode);
    if (mode != "pertensor" && mode != "pertoken" && mode != "perchannel") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantMode only support pertensor, pertoken and perchannel.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (mode == "pertensor" && dynamicQuantParams.groupIndex != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "If quantMode is pertensor, groupIndexOptional must be nullptr.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (mode == "perchannel" && dynamicQuantParams.groupIndex != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "If quantMode is perchannel, groupIndexOptional must be nullptr.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const DynamicQuantParams& dynamicQuantParams)
{
    auto xDimNum = dynamicQuantParams.x->GetViewShape().GetDimNum();
    int64_t xLastDimInput = dynamicQuantParams.x->GetViewShape().GetDim(xDimNum - 1);
    CHECK_COND(xDimNum > 1, ACLNN_ERR_PARAM_INVALID, "The dimNum[%lu] of x should be greater than 1", xDimNum);
    CHECK_COND(xDimNum < DIM_MAX, ACLNN_ERR_PARAM_INVALID, "The dimNum[%lu] of x should be less than 9", xDimNum);
    int64_t xLastToSecondDimInput = dynamicQuantParams.x->GetViewShape().GetDim(xDimNum - NUM_TWO);
    const std::string mode = std::string(dynamicQuantParams.quantMode);

    if (dynamicQuantParams.smoothScales) {
        if (dynamicQuantParams.groupIndex) {
            auto groupDimNum = dynamicQuantParams.groupIndex->GetViewShape().GetDimNum();
            CHECK_COND(
                groupDimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dimNum[%lu] of group_indexs should be equal to one.",
                groupDimNum);
        } else {
            auto smoothDimNum = dynamicQuantParams.smoothScales->GetViewShape().GetDimNum();
            CHECK_COND(
                smoothDimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dimNum[%lu] of smooth_scales should be equal to one.",
                smoothDimNum);
            if (mode == "perchannel") {
                CHECK_COND(
                    dynamicQuantParams.smoothScales->GetViewShape().GetDim(smoothDimNum - 1) == xLastToSecondDimInput,
                    ACLNN_ERR_PARAM_INVALID,
                    "If quantMode is perchannel, the second to last dim[%ld] of x and the last dim[%ld] of smooth_scales shoule be equal.",
                    xLastToSecondDimInput, dynamicQuantParams.smoothScales->GetViewShape().GetDim(smoothDimNum - 1));
            } else {
                CHECK_COND(
                    dynamicQuantParams.smoothScales->GetViewShape().GetDim(smoothDimNum - 1) == xLastDimInput,
                    ACLNN_ERR_PARAM_INVALID, "The last dim[%ld] of x and the dim[%ld] of smooth_scales shoule be equal.",
                    xLastDimInput, dynamicQuantParams.smoothScales->GetViewShape().GetDim(smoothDimNum - 1));
            }
        }
    }

    auto yDtype = dynamicQuantParams.y->GetDataType();
    auto yDimNum = dynamicQuantParams.y->GetViewShape().GetDimNum();
    if (yDtype != op::DataType::DT_INT32) {
        CHECK_COND(
            CheckOpDim(dynamicQuantParams.x->GetViewShape(), dynamicQuantParams.y->GetViewShape(), xDimNum, yDimNum) ==
                ACLNN_SUCCESS,
            ACLNN_ERR_PARAM_INVALID, "The shapes of x and y are inconsistent.");
    }

    // check y dtype int32
    if (yDtype == op::DataType::DT_INT32) {
        CHECK_RET(
            CheckInt32OutputShape(
                dynamicQuantParams.x->GetViewShape(), dynamicQuantParams.y->GetViewShape(), xDimNum, yDimNum) ==
                ACLNN_SUCCESS,
            ACLNN_ERR_PARAM_INVALID);
    }

    // check y dtype int4
    if (yDtype == op::DataType::DT_INT4) {
        if (xLastDimInput % INT4_NUMS_IN_INT8_SPACE) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "If y dtype is int4, the last dim of x must be divisible by 2, but the last dim is (%ld).",
                xLastDimInput);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    // check scale shape
    auto scaleNum = dynamicQuantParams.scale->GetViewShape().GetDimNum();
    if (mode == "pertoken") {
        CHECK_COND(
            CheckOpDim(
                dynamicQuantParams.x->GetViewShape(), dynamicQuantParams.scale->GetViewShape(), xDimNum - 1,
                scaleNum) == ACLNN_SUCCESS,
            ACLNN_ERR_PARAM_INVALID,
            "The shapes of x and scale should be consistent except for the last dimension of x.");
    } else if (mode == "pertensor") {
        CHECK_COND(
            scaleNum == 1, ACLNN_ERR_PARAM_INVALID, "If quantMode is pertensor, the shape of scale must be (1,).");
        CHECK_COND(
            dynamicQuantParams.scale->GetViewShape().GetDim(0) == 1, ACLNN_ERR_PARAM_INVALID,
            "If quantMode is pertensor, the shape of scale must be (1,).");
    } else if (mode == "perchannel") {
        CHECK_COND(
            CheckOpDim(
                dynamicQuantParams.x->GetViewShape(), dynamicQuantParams.scale->GetViewShape(), xDimNum - NUM_TWO,
                scaleNum - 1) == ACLNN_SUCCESS,
            ACLNN_ERR_PARAM_INVALID,
            "If quantMode is perchannel, the shapes of x excluding the last two dimensions "
            "must be the same as the shape of scale excluding the last dimension.");
        CHECK_COND(
            (dynamicQuantParams.scale->GetViewShape().GetDim(scaleNum - 1) == 
                dynamicQuantParams.x->GetViewShape().GetDim(xDimNum - 1)),
            ACLNN_ERR_PARAM_INVALID,
            "If quantMode is perchannel, the last dim of scale must be equal to the last dim of x.");
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantMode is invalid, must be pertoken, pertensor or perchannel.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // check offset shape
    if (dynamicQuantParams.offset) {
        OP_CHECK_SHAPE_NOT_EQUAL(dynamicQuantParams.offset, dynamicQuantParams.scale, return ACLNN_ERR_PARAM_INVALID);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(const DynamicQuantParams& dynamicQuantParams)
{
    // 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListByNpuArch();
    OP_CHECK_DTYPE_NOT_SUPPORT(dynamicQuantParams.x, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
    if (dynamicQuantParams.smoothScales) {
        // 检查smooth的数据类型是否在add算子的支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(dynamicQuantParams.smoothScales, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
        CHECK_COND(
            dynamicQuantParams.x->GetDataType() == dynamicQuantParams.smoothScales->GetDataType(),
            ACLNN_ERR_PARAM_INVALID, "The dtypes of x and smooth_scales are not equal.");
        if (dynamicQuantParams.groupIndex) {
            OP_CHECK_DTYPE_NOT_SUPPORT(
                dynamicQuantParams.groupIndex, GROUP_INDEX_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
        }
    }
    const std::initializer_list<op::DataType> outputSupportList = GetOutputSupportListByNpuArch();
    OP_CHECK_DTYPE_NOT_SUPPORT(dynamicQuantParams.y, outputSupportList, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(dynamicQuantParams.scale, op::DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);
    if (dynamicQuantParams.offset) {
        OP_CHECK_DTYPE_NOT_MATCH(dynamicQuantParams.offset, op::DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

inline static aclnnStatus CheckParams(DynamicQuantParams& dynamicQuantParams)
{
    CHECK_COND(
        CheckNotNull(dynamicQuantParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
        "One of the required inputs is nullptr.");
    CHECK_RET(CheckAttr(dynamicQuantParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(CheckDtype(dynamicQuantParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid dtype.");
    CHECK_RET(CheckShape(dynamicQuantParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus Int42Int32PackedTensor(const aclTensor* out, const aclTensor*& outTensor, aclOpExecutor* executor)
{
    auto viewShape = out->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    viewShape[viewShapeDim - 1] /= INT4_NUMS_IN_INT32_SPACE;
    auto outTemp = executor->CreateView(out, viewShape, out->GetViewOffset());
    CHECK_RET(outTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);
    outTemp->SetDataType(DataType::DT_INT32);
    outTensor = outTemp;
    return ACLNN_SUCCESS;
}
}; // namespace

aclnnStatus InputsContiguous(
    const aclTensor* tensor, const aclTensor*& reformatedTensor, const std::string& tensorName, aclOpExecutor* executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    reformatedTensor = l0op::Contiguous(tensor, executor);
    CHECK_COND(reformatedTensor != nullptr, ACLNN_ERR_INNER_NULLPTR, "%s Contiguous failed.", tensorName.c_str());
    return ACLNN_SUCCESS;
}

aclnnStatus GetDynamicQuantResultByL0Api(
    DynamicQuantParams& dynamicQuantParams, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    aclnnStatus ret = CheckParams(dynamicQuantParams);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (dynamicQuantParams.x->IsEmpty() || dynamicQuantParams.y->IsEmpty() || dynamicQuantParams.scale->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* x = nullptr;
    const aclTensor* smoothScales = nullptr;
    const aclTensor* groupIndex = nullptr;
    const aclTensor* outputTensor = nullptr;
    const aclTensor* y = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    ret = InputsContiguous(dynamicQuantParams.x, x, "x", uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = InputsContiguous(dynamicQuantParams.smoothScales, smoothScales, "smooth_scales", uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = InputsContiguous(dynamicQuantParams.groupIndex, groupIndex, "group_index", uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 输出dtype的int32是int4伪装，实际需要按照int4计算
    auto yDtype = dynamicQuantParams.y->GetDataType();
    if (yDtype == op::DataType::DT_INT32) {
        yDtype = op::DataType::DT_INT4;
        OP_LOGD("op dynamicquant real output is int4.");
    }

    if (Ops::NN::AclnnUtil::IsRegbase()) {
        // DynamicQuantV2 consolidates and remains backward compatible with DynamicQuant in socVersion ASCEND950
        auto dynamicQuantV2Result = l0op::DynamicQuantV2(
            x, smoothScales, groupIndex, yDtype, dynamicQuantParams.isSymmetrical, dynamicQuantParams.quantMode,
            uniqueExecutor.get());
        y = std::get<0>(dynamicQuantV2Result);
        outputTensor = std::get<0>(dynamicQuantV2Result);
        scale = std::get<1>(dynamicQuantV2Result);
        offset = std::get<2>(dynamicQuantV2Result); // 2 is offset index
        CHECK_RET(y != nullptr && scale != nullptr && offset != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (dynamicQuantParams.offset) {
            auto viewCopyOffsetResult = l0op::ViewCopy(offset, dynamicQuantParams.offset, uniqueExecutor.get());
            CHECK_RET(viewCopyOffsetResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    } else {
        bool isSymmetrical = dynamicQuantParams.offset == nullptr ? true : false;
        if (isSymmetrical) {
            auto dynamicQuantResult = l0op::DynamicQuant(x, smoothScales, groupIndex, yDtype, uniqueExecutor.get());
            y = std::get<0>(dynamicQuantResult);
            outputTensor = std::get<0>(dynamicQuantResult);
            scale = std::get<1>(dynamicQuantResult);
            CHECK_RET(y != nullptr && scale != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else {
            auto dynamicQuantV2Result =
                l0op::DynamicQuantV2(x, smoothScales, groupIndex, yDtype, false, "pertoken", uniqueExecutor.get());
            y = std::get<0>(dynamicQuantV2Result);
            outputTensor = std::get<0>(dynamicQuantV2Result);
            scale = std::get<1>(dynamicQuantV2Result);
            offset = std::get<2>(dynamicQuantV2Result); // 2 is offset index
            CHECK_RET(y != nullptr && scale != nullptr && offset != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto viewCopyOffsetResult = l0op::ViewCopy(offset, dynamicQuantParams.offset, uniqueExecutor.get());
            CHECK_RET(viewCopyOffsetResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    // 针对pta调用方式，由于pta不支持int4，故当输出是int4时，pta那边会将dstType转换成DT_INT32
    if (dynamicQuantParams.y->GetDataType() == op::DataType::DT_INT32) {
        OP_LOGD("DynamicQuantParams.dstType is DT_INT32");
        // 如果输出是Int4，需要转成Int32，8个Int4输出拼成1个Int32
        ret = Int42Int32PackedTensor(outputTensor, y, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyYResult = l0op::ViewCopy(y, dynamicQuantParams.y, uniqueExecutor.get());
    CHECK_RET(viewCopyYResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyScaleResult = l0op::ViewCopy(scale, dynamicQuantParams.scale, uniqueExecutor.get());
    CHECK_RET(viewCopyScaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDynamicQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* smoothScalesOptional, const aclTensor* yOut, const aclTensor* scaleOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnDynamicQuant, DFX_IN(x, smoothScalesOptional), DFX_OUT(yOut, scaleOut));
    DynamicQuantParams dynamicQuantParams{x,      smoothScalesOptional, nullptr, 2, true, "pertoken", yOut, scaleOut,
                                          nullptr};
    return GetDynamicQuantResultByL0Api(dynamicQuantParams, workspaceSize, executor);
}

aclnnStatus aclnnDynamicQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDynamicQuant);
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in DynamicQuant launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDynamicQuantV2GetWorkspaceSize(
    const aclTensor* x, const aclTensor* smoothScalesOptional, const aclTensor* groupIndexOptional, int64_t dstType,
    const aclTensor* yOut, const aclTensor* scaleOut, const aclTensor* offsetOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnDynamicQuantV2, DFX_IN(x, smoothScalesOptional, groupIndexOptional, dstType),
        DFX_OUT(yOut, scaleOut, offsetOut));

    bool isSymmetrical = (offsetOut == nullptr);
    const char* quantMode = "pertoken";
    DynamicQuantParams dynamicQuantParams{
        x, smoothScalesOptional, groupIndexOptional, dstType, isSymmetrical, quantMode, yOut, scaleOut, offsetOut};
    return GetDynamicQuantResultByL0Api(dynamicQuantParams, workspaceSize, executor);
}

aclnnStatus aclnnDynamicQuantV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDynamicQuantV2);
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in DynamicQuant launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDynamicQuantV3GetWorkspaceSize(
    const aclTensor* x, const aclTensor* smoothScalesOptional, const aclTensor* groupIndexOptional, int64_t dstType,
    bool isSymmetrical, const char* quantMode, const aclTensor* yOut, const aclTensor* scaleOut,
    const aclTensor* offsetOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnDynamicQuantV3, DFX_IN(x, smoothScalesOptional, groupIndexOptional, dstType, isSymmetrical, quantMode),
        DFX_OUT(yOut, scaleOut, offsetOut));
    if (!Ops::NN::AclnnUtil::IsRegbase()) {
        OP_LOGE(ACLNN_ERR_INNER, "aclnnDynamicQuantV3 only support socVersion Ascend950");
        return ACLNN_ERR_INNER;
    }
    DynamicQuantParams dynamicQuantParams{
        x, smoothScalesOptional, groupIndexOptional, dstType, isSymmetrical, quantMode, yOut, scaleOut, offsetOut};
    return GetDynamicQuantResultByL0Api(dynamicQuantParams, workspaceSize, executor);
}

aclnnStatus aclnnDynamicQuantV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDynamicQuantV3);
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in DynamicQuant launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
