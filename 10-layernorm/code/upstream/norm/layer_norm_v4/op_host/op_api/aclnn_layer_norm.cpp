/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_util.h"
#include "norm/norm_common/op_host/op_api/norm_tensor_util.h"
#include "level0/fill.h"
#include "layer_norm_v3.h"
#include "layer_norm_v4.h"
#include "aclnn_layer_norm.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM_LEN = 8;
constexpr size_t LEAST_NORMALIZED_SHAPE_LEN = 1;
constexpr size_t LAYER_NORM_OUT_NUM = 3;
constexpr size_t Y_INDEX = 0;
constexpr size_t MEAN_INDEX = 1;
constexpr size_t RSTD_INDEX = 2;
constexpr int32_t HIGH_PRECISION = 0;
constexpr int32_t HIGH_PERFORMANCE = 1;
constexpr int32_t KEEP_FP16 = 2;
constexpr int64_t MIN_V4_REDUCE_AXIS = 1024;
constexpr int64_t MAX_V4_REDUCE_AXIS = 4096;
constexpr int64_t B16_BLOCK_ALIGN_NUM = 16;
constexpr int64_t B32_BLOCK_ALIGN_NUM = 8;
constexpr int64_t V4_TRANSPOSE_REDUCE_AXIS_LIMIT = 64;
constexpr int64_t V4_TRANSPOSE_310P_REDUCE_AXIS_MAX = 40;
constexpr int64_t V4_TRANSPOSE_310P_REDUCE_AXIS_MIN = 20;
constexpr int64_t LN_DIM_ONE = 1;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase(curArch)) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}

inline static bool CheckNotNull(const aclTensor* input, const aclIntArray* normalizedShape, const aclTensor* out)
{
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(normalizedShape, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckInputDtype(const aclTensor* input, const aclTensor* weightOptional, const aclTensor* biasOptional)
{
    const auto& supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(input, supportList, return false);
    if (weightOptional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(weightOptional, supportList, return false);
        if (weightOptional->GetDataType() != DataType::DT_FLOAT) {
            OP_CHECK_DTYPE_NOT_SAME(weightOptional, input, return false);
        }
    }
    if (biasOptional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(biasOptional, supportList, return false);
        if (biasOptional->GetDataType() != DataType::DT_FLOAT) {
            OP_CHECK_DTYPE_NOT_SAME(biasOptional, input, return false);
        }
    }
    if (weightOptional && biasOptional) {
        OP_CHECK_DTYPE_NOT_SAME(weightOptional, biasOptional, return false);
    }
    return true;
}

static bool CheckOutputDtype(const aclTensor* out, const aclTensor* meanOutOptional, const aclTensor* rstdOutOptional)
{
    const auto& supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
    if (meanOutOptional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(meanOutOptional, supportList, return false);
    }
    if (rstdOutOptional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(rstdOutOptional, supportList, return false);
    }
    return true;
}

inline static bool CheckArrayLen(const aclIntArray* normalizedShape)
{
    if (normalizedShape->Size() > MAX_DIM_LEN) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected aclnnLayerNorm normalizedShape dim [%zu] to not be greater than [%zu] but check failed.",
            normalizedShape->Size(), MAX_DIM_LEN);
        return false;
    }
    return true;
}

static bool CheckLen(
    const aclTensor* input, const aclIntArray* normalizedShape, const aclTensor* weightOptional,
    const aclTensor* biasOptional, const aclTensor* out, const aclTensor* meanOutOptional,
    const aclTensor* rstdOutOptional)
{
    OP_CHECK_MAX_DIM(input, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);
    if (weightOptional) {
        OP_CHECK_MAX_DIM(weightOptional, MAX_DIM_LEN, return false);
    }
    if (biasOptional) {
        OP_CHECK_MAX_DIM(biasOptional, MAX_DIM_LEN, return false);
    }
    if (meanOutOptional) {
        OP_CHECK_MAX_DIM(meanOutOptional, MAX_DIM_LEN, return false);
    }
    if (rstdOutOptional) {
        OP_CHECK_MAX_DIM(rstdOutOptional, MAX_DIM_LEN, return false);
    }
    return CheckArrayLen(normalizedShape);
}

static bool CheckShape(
    const aclTensor* input, const aclIntArray* normalizedShape, const aclTensor* weightOptional,
    const aclTensor* biasOptional, const aclTensor* out, const aclTensor* meanOutOptional,
    const aclTensor* rstdOutOptional)
{
    // 1.检查入参维度是否小于8维
    if (!CheckLen(input, normalizedShape, weightOptional, biasOptional, out, meanOutOptional, rstdOutOptional)) {
        return false;
    }
    // 2.检查normalizedShape的长度是否大于等于1
    if (normalizedShape->Size() < LEAST_NORMALIZED_SHAPE_LEN) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected aclnnLayerNorm normalizedShape len [%zu] to be greater than [%zu] but check failed.",
            normalizedShape->Size(), LEAST_NORMALIZED_SHAPE_LEN);
        return false;
    }
    // 3.检查input维度是否不小于normalizedShape的长度
    OP_CHECK_MIN_DIM(input, normalizedShape->Size(), return false);
    // 4.校验weight存在时与normalizedShape长度是否相同
    if (weightOptional) {
        OP_CHECK_WRONG_DIMENSION(weightOptional, normalizedShape->Size(), return false);
    }
    // 5.校验bias存在时与normalizedShape长度是否相同
    if (biasOptional) {
        OP_CHECK_WRONG_DIMENSION(biasOptional, normalizedShape->Size(), return false);
    }

    // 6.检查输入与normalizedShape间的关系
    auto inputShape = input->GetViewShape();
    const size_t beginAxis = inputShape.GetDimNum() - normalizedShape->Size();
    for (size_t index = 0; index < normalizedShape->Size(); index++) {
        // 6.1 校验input与normalizedShape间shape是否满足右对齐相等
        int64_t normDim = *(normalizedShape->GetData() + index);
        int64_t inputDim = inputShape.GetDim(index + beginAxis);
        if (normDim != inputDim) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Expected normalized index [%zu] shape [%ld] be equal to input index [%zu] shape [%ld], but failed.",
                index, normDim, index + beginAxis, inputDim);
            return false;
        }
        // 6.2 校验weight存在时与normalizedShape是否相等
        if (weightOptional) {
            int64_t weightDim = weightOptional->GetViewShape().GetDim(index);
            if (normDim != weightDim) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "Expected normalized index [%zu] shape [%ld] be equal to weight index [%zu] shape [%ld], but "
                    "failed.",
                    index, normDim, index, weightDim);
                return false;
            }
        }
        // 6.3 校验bias存在时与normalizedShape是否相等
        if (biasOptional) {
            int64_t biasDim = biasOptional->GetViewShape().GetDim(index);
            if (normDim != biasDim) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "Expected normalized index [%zu] shape [%ld] be equal to bias index [%zu] shape [%ld], but failed.",
                    index, normDim, index, biasDim);
                return false;
            }
        }
    }

    // 7.校验三个输出的shape
    OP_CHECK_SHAPE_NOT_EQUAL(input, out, return false);
    return true;
}

static bool CheckImplMode(
    const aclTensor* input, const aclTensor* weightOptional, const aclTensor* biasOptional, int32_t implMode)
{
    if (implMode != HIGH_PRECISION && implMode != HIGH_PERFORMANCE && implMode != KEEP_FP16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected implMode to be in [0, 1, 2], but now got [%d].", implMode);
        return false;
    }
    if (implMode == KEEP_FP16) {
        if (input->GetDataType() != DataType::DT_FLOAT16 ||
            (weightOptional && weightOptional->GetDataType() != DataType::DT_FLOAT16) ||
            (biasOptional && biasOptional->GetDataType() != DataType::DT_FLOAT16)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "KEEP_FP16 only support all input dtype float16!");
            return false;
        }
    }
    return true;
}

inline static bool IsV4SocCheck(const aclTensor* input, const aclTensor* weight, int64_t reduceAxis)
{
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        return true;
    }
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    bool v4SingleReadTemplateSup = input->GetDataType() == weight->GetDataType() &&
                                   input->GetDataType() == DataType::DT_BF16 &&
                                   (reduceAxis == MIN_V4_REDUCE_AXIS || reduceAxis == MAX_V4_REDUCE_AXIS) &&
                                   (curArch == NpuArch::DAV_2201);
    bool v4TransposeTemplate310pSup =
        (input->GetDataType() == DataType::DT_FLOAT16) &&
        (reduceAxis == V4_TRANSPOSE_310P_REDUCE_AXIS_MIN || reduceAxis == V4_TRANSPOSE_310P_REDUCE_AXIS_MAX) &&
        (curArch == NpuArch::DAV_2002);
    int64_t inputBlockAlign = (input->GetDataType() == DataType::DT_FLOAT ? B32_BLOCK_ALIGN_NUM : B16_BLOCK_ALIGN_NUM);
    bool v4TransposeTemplate910bSup = ((reduceAxis % inputBlockAlign) != 0) &&
                                      (reduceAxis < V4_TRANSPOSE_REDUCE_AXIS_LIMIT) &&
                                      (curArch == NpuArch::DAV_2201);
    return (v4SingleReadTemplateSup || v4TransposeTemplate310pSup || v4TransposeTemplate910bSup);
}

static aclnnStatus CheckParamsWithImplMode(
    const aclTensor* input, const aclIntArray* normalizedShape, const aclTensor* weightOptional,
    const aclTensor* biasOptional, const aclTensor* out, const aclTensor* meanOutOptional,
    const aclTensor* rstdOutOptional, int32_t implMode)
{
    // 1. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckInputDtype(input, weightOptional, biasOptional), ACLNN_ERR_PARAM_INVALID);
    // 2. 检查输出的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckOutputDtype(out, meanOutOptional, rstdOutOptional), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查input, weight，bias与normalizedShape间的shape关系
    CHECK_RET(
        CheckShape(input, normalizedShape, weightOptional, biasOptional, out, meanOutOptional, rstdOutOptional),
        ACLNN_ERR_PARAM_INVALID);
    // 4. 检查implMode是否合法
    CHECK_RET(CheckImplMode(input, weightOptional, biasOptional, implMode), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLayerNormWithImplModeGetWorkspaceSize(
    const aclTensor* input, const aclIntArray* normalizedShape, const aclTensor* weightOptional,
    const aclTensor* biasOptional, double eps, aclTensor* out, aclTensor* meanOutOptional, aclTensor* rstdOutOptional,
    int32_t implMode, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnLayerNormWithImplMode, DFX_IN(input, normalizedShape, weightOptional, biasOptional, eps, implMode),
        DFX_OUT(out, meanOutOptional, rstdOutOptional));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 检查参数是否为空指针
    CHECK_RET(CheckNotNull(input, normalizedShape, out), ACLNN_ERR_PARAM_NULLPTR);

    // 固定写法，参数检查
    auto ret = CheckParamsWithImplMode(
        input, normalizedShape, weightOptional, biasOptional, out, meanOutOptional, rstdOutOptional, implMode);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 根据input_shape和normalizedShape的关系获取非reduce轴和reduce轴的shape
    auto inputShape = input->GetViewShape();
    const size_t inputDim = inputShape.GetDimNum();
    const size_t normDim = normalizedShape->Size();
    const size_t beginAxis = inputDim - normDim;
    int64_t M = 1;
    int64_t N = 1;
    for (size_t index = 0; index < beginAxis; index++) {
        M *= inputShape.GetDim(index);
    }
    for (size_t index = beginAxis; index < inputDim; index++) {
        N *= inputShape.GetDim(index);
    }

    // 空tensor场景处理，区分reduce轴是否为0
    if (M == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (N == 0) {
        if (meanOutOptional) {
            ret = op::ProcessEmptyTensorWithValue(meanOutOptional, 0, uniqueExecutor.get());
            CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        }
        if (rstdOutOptional) {
            ret = op::ProcessEmptyTensorWithValue(
                rstdOutOptional, std::numeric_limits<float>::quiet_NaN(), uniqueExecutor.get());
            CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        }
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // 固定写法，将输入转换成连续的tensor
    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 判定设备类型，后续做不同处理
    bool forwardV4Compute = Ops::NN::AclnnUtil::IsRegbase();

    // 构造新的weightContiguous
    const aclTensor* weightContiguous = nullptr;
    if (weightOptional) {
        weightContiguous = l0op::Contiguous(weightOptional, uniqueExecutor.get());
        CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        if (!forwardV4Compute) {
            // 非目标设备按照原逻辑执行
            auto weightTensor = uniqueExecutor.get()->ConvertToTensor(normalizedShape, DataType::DT_INT64);
            aclScalar* scalarOne = uniqueExecutor.get()->AllocScalar(1);
            auto weightOptionalDtype = biasOptional ? biasOptional->GetDataType() : inputContiguous->GetDataType();
            auto oneTensor = uniqueExecutor.get()->ConvertToTensor(scalarOne, weightOptionalDtype);
            weightContiguous = l0op::Fill(weightTensor, oneTensor, normalizedShape, uniqueExecutor.get());
            CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    // 构造新的biasContiguous
    const aclTensor* biasContiguous = nullptr;
    if (biasOptional) {
        biasContiguous = l0op::Contiguous(biasOptional, uniqueExecutor.get());
        CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        if (!forwardV4Compute) {
            // 非目标设备按照原逻辑执行
            auto biasTensor = uniqueExecutor.get()->ConvertToTensor(normalizedShape, DataType::DT_INT64);
            aclScalar* scalarZero = uniqueExecutor.get()->AllocScalar(0);
            auto biasOptionalDtype = weightOptional ? weightOptional->GetDataType() : inputContiguous->GetDataType();
            auto zeroTensor = uniqueExecutor.get()->ConvertToTensor(scalarZero, biasOptionalDtype);
            biasContiguous = l0op::Fill(biasTensor, zeroTensor, normalizedShape, uniqueExecutor.get());
            CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    // 进行LayerNorm计算，根据规格决定使用LayerNormV4或LayerNormV3算子
    if (!forwardV4Compute) {
        forwardV4Compute = IsV4SocCheck(inputContiguous, weightContiguous, N);
    }
    std::array<aclTensor*, LAYER_NORM_OUT_NUM> layerNormOut = {nullptr, nullptr, nullptr};

    if (forwardV4Compute) {
        l0op::LayerNormV4Params V4_params;
        V4_params.input = inputContiguous;
        V4_params.normalizedShape = normalizedShape;
        V4_params.weight = weightContiguous;
        V4_params.bias = biasContiguous;

        layerNormOut = l0op::LayerNormV4(V4_params, eps, uniqueExecutor.get());
    } else {
        l0op::LayerNormV3WithImplModeparams V3_params;
        V3_params.input = inputContiguous;
        V3_params.weight = weightContiguous;
        V3_params.bias = biasContiguous;
        V3_params.eps = eps;
        V3_params.implMode = implMode;
        V3_params.beginAxis = static_cast<int64_t>(beginAxis);
        layerNormOut = l0op::LayerNormV3WithImplMode(V3_params, uniqueExecutor.get());
    }

    // 处理第一个输出
    auto outRes = layerNormOut[Y_INDEX];
    CHECK_RET(outRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outCast = l0op::Cast(outRes, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(outCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outViewCopy = l0op::ViewCopy(outCast, out, uniqueExecutor.get());
    CHECK_RET(outViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 处理第二个输出
    if (meanOutOptional) {
        auto meanRes = layerNormOut[MEAN_INDEX];
        CHECK_RET(meanRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto meanCast = l0op::Cast(meanRes, meanOutOptional->GetDataType(), uniqueExecutor.get());
        CHECK_RET(meanCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto meanViewCopy = l0op::ViewCopy(meanCast, meanOutOptional, uniqueExecutor.get());
        CHECK_RET(meanViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 处理第三个输出
    if (rstdOutOptional) {
        auto rstdRes = layerNormOut[RSTD_INDEX];
        CHECK_RET(rstdRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto rstdCast = l0op::Cast(rstdRes, rstdOutOptional->GetDataType(), uniqueExecutor.get());
        CHECK_RET(rstdCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto rstdViewCopy = l0op::ViewCopy(rstdCast, rstdOutOptional, uniqueExecutor.get());
        CHECK_RET(rstdViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLayerNormGetWorkspaceSize(
    const aclTensor* input, const aclIntArray* normalizedShape, const aclTensor* weightOptional,
    const aclTensor* biasOptional, double eps, aclTensor* out, aclTensor* meanOutOptional, aclTensor* rstdOutOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    int32_t implMode = HIGH_PRECISION;
    return aclnnLayerNormWithImplModeGetWorkspaceSize(
        input, normalizedShape, weightOptional, biasOptional, eps, out, meanOutOptional, rstdOutOptional, implMode,
        workspaceSize, executor);
}

aclnnStatus aclnnLayerNormWithImplMode(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLayerNormWithImplMode);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnLayerNorm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLayerNorm);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
