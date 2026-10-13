/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "opdev/data_type_utils.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/cast.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "norm/norm_common/op_host/op_api/norm_tensor_util.h"
#include "level0/fill.h"
#include "layer_norm_v4.h"
#include "aclnn_layer_norm.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM_LEN = 8;
constexpr size_t LEAST_NORMALIZED_SHAPE_LEN = 1;
constexpr size_t Y_INDEX = 0;
constexpr size_t LAYER_NORM_OUT_NUM = 3;
constexpr size_t MEAN_INDEX = 1;
constexpr int32_t HIGH_PRECISION = 0;
constexpr size_t RSTD_INDEX = 2;
constexpr int32_t HIGH_PERFORMANCE = 1;
constexpr int64_t MIN_V4_REDUCE_AXIS = 1024;
constexpr int64_t MAX_V4_REDUCE_AXIS = 4096;
constexpr int64_t B16_BLOCK_ALIGN_NUM = 16;
constexpr int64_t B32_BLOCK_ALIGN_NUM = 8;
constexpr int32_t KEEP_FP16 = 2;
constexpr int64_t V4_TRANSPOSE_310P_REDUCE_AXIS_MAX = 40;
constexpr int64_t V4_TRANSPOSE_REDUCE_AXIS_LIMIT = 64;
constexpr int64_t LN_DIM_ONE = 1;
constexpr int64_t V4_TRANSPOSE_310P_REDUCE_AXIS_MIN = 20;

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
}

struct LayerNormTensor {
    const aclTensor* input;
    const aclTensor* weightOptional;
    const aclTensor* biasOptional;
    double eps;
    aclTensor* out;
    aclTensor* meanOutOptional;
    aclTensor* rstdOutOptional;
};

static bool CheckInputDtype(const aclTensor* input, const aclTensor* fastWeightOptional, const aclTensor* biasOptional)
{
    const auto& supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(input, supportList, return false);
    if (fastWeightOptional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(fastWeightOptional, supportList, return false);
        if (fastWeightOptional->GetDataType() != DataType::DT_FLOAT) {
            OP_CHECK_DTYPE_NOT_SAME(fastWeightOptional, input, return false);
        }
    }
    if (fastWeightOptional && biasOptional) {
        OP_CHECK_DTYPE_NOT_SAME(fastWeightOptional, biasOptional, return false);
    }
    if (biasOptional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(biasOptional, supportList, return false);
        if (biasOptional->GetDataType() != DataType::DT_FLOAT) {
            OP_CHECK_DTYPE_NOT_SAME(biasOptional, input, return false);
        }
    }
    return true;
}

static bool CheckFastOutputDtype(const aclTensor* out, const aclTensor* meanOutOptional, const aclTensor* rstdOutOptional)
{
    const auto& supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
    if (rstdOutOptional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(rstdOutOptional, supportList, return false);
    }
    if (meanOutOptional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(meanOutOptional, supportList, return false);
    }
    return true;
}

inline static bool CheckArrayLen(const aclIntArray* normalizedShape)
{
    if (normalizedShape->Size() > MAX_DIM_LEN) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected aclnnFastLayerNorm normalizedShape dim [%zu] to not be greater than [%zu] but check failed.",
            normalizedShape->Size(), MAX_DIM_LEN);
        return false;
    }
    return true;
}

inline static bool CheckFastNotNull(const aclTensor* input, const aclIntArray* normalizedShape, const aclTensor* out)
{
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(normalizedShape, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckLen(
    const aclTensor* input, const aclIntArray* normalizedShape, const aclTensor* weightOptional,
    const aclTensor* biasOptional, const aclTensor* out, const aclTensor* meanOutOptional,
    const aclTensor* rstdOutOptional)
{
    OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(input, MAX_DIM_LEN, return false);
    if (weightOptional) {
        OP_CHECK_MAX_DIM(weightOptional, MAX_DIM_LEN, return false);
    }
    if (rstdOutOptional) {
        OP_CHECK_MAX_DIM(rstdOutOptional, MAX_DIM_LEN, return false);
    }
    if (meanOutOptional) {
        OP_CHECK_MAX_DIM(meanOutOptional, MAX_DIM_LEN, return false);
    }
    if (biasOptional) {
        OP_CHECK_MAX_DIM(biasOptional, MAX_DIM_LEN, return false);
    }
    return CheckArrayLen(normalizedShape);
}

static bool CheckNormalizeShape(
    const aclTensor* input, const aclIntArray* fastNormalizedShape, const aclTensor* weightOptional, const aclTensor* biasOptional)
{
    // 6.检查输入与fastNormalizedShape间的关系
    auto inputShape = input->GetViewShape();
    const size_t beginAxis = inputShape.GetDimNum() - fastNormalizedShape->Size();
    for (size_t index = 0; index < fastNormalizedShape->Size(); index++) {
        // 6.1 校验input与fastNormalizedShape间shape是否满足右对齐相等
        int64_t fastNormDim = *(fastNormalizedShape->GetData() + index);
        int64_t inputDim = inputShape.GetDim(index + beginAxis);
        if (fastNormDim != inputDim) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Expected normalized index [%zu] shape [%ld] be equal to input index [%zu] shape [%ld], but failed.",
                index, fastNormDim, index + beginAxis, inputDim);
            return false;
        }
        // 6.2 校验weight存在时与fastNormalizedShape是否相等
        if (weightOptional) {
            int64_t weightDim = weightOptional->GetViewShape().GetDim(index);
            if (fastNormDim != weightDim) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "Expected normalized index [%zu] shape [%ld] be equal to weight index [%zu] shape [%ld], but "
                    "failed.",
                    index, fastNormDim, index, weightDim);
                return false;
            }
        }
        // 6.3 校验bias存在时与fastNormalizedShape是否相等
        if (biasOptional) {
            int64_t biasDim = biasOptional->GetViewShape().GetDim(index);
            if (fastNormDim != biasDim) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "Expected normalized index [%zu] shape [%ld] be equal to bias index [%zu] shape [%ld], but failed.",
                    index, fastNormDim, index, biasDim);
                return false;
            }
        }
    }
    return true;
}

static bool CheckShape(
    const aclTensor* input, const aclIntArray* normalizedShape, const aclTensor* weightOptional,
    const aclTensor* biasOptional, const aclTensor* out, const aclTensor* meanOutOptional,
    const aclTensor* rstdOutOptional)
{
    // 1.检查normalizedShape的长度是否大于等于1
    if (normalizedShape->Size() < LEAST_NORMALIZED_SHAPE_LEN) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected aclnnFastLayerNorm normalizedShape len [%zu] to be greater than [%zu] but check failed.",
            normalizedShape->Size(), LEAST_NORMALIZED_SHAPE_LEN);
        return false;
    }
    // 2.检查入参维度是否小于8维
    if (!CheckLen(input, normalizedShape, weightOptional, biasOptional, out, meanOutOptional, rstdOutOptional)) {
        return false;
    }
    // 3.校验weight存在时与normalizedShape长度是否相同
    if (weightOptional) {
        OP_CHECK_WRONG_DIMENSION(weightOptional, normalizedShape->Size(), return false);
    }
    // 4.检查input维度是否不小于normalizedShape的长度
    if (input->GetViewShape().GetDimNum() < static_cast<size_t>(normalizedShape->Size())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The input tensor's dimensions must not be smaller than normalizedShape's size.");
        return false;
    }
    // 5.校验bias存在时与normalizedShape长度是否相同
    if (biasOptional) {
        OP_CHECK_WRONG_DIMENSION(biasOptional, normalizedShape->Size(), return false);
    }

    if (!CheckNormalizeShape(input, normalizedShape, weightOptional, biasOptional)) {
        return false;
    }

    // 7.校验三个输出的shape
    OP_CHECK_SHAPE_NOT_EQUAL(input, out, return false);
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* input, const aclIntArray* normalizedShape, const aclTensor* weightOptional,
    const aclTensor* biasOptional, const aclTensor* out, const aclTensor* meanOutOptional,
    const aclTensor* rstdOutOptional)
{
    // 1. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckInputDtype(input, weightOptional, biasOptional), ACLNN_ERR_PARAM_INVALID);
    // 2. 检查输出的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckFastOutputDtype(out, meanOutOptional, rstdOutOptional), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查input, weight，bias与normalizedShape间的shape关系
    CHECK_RET(
        CheckShape(input, normalizedShape, weightOptional, biasOptional, out, meanOutOptional, rstdOutOptional),
        ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus doFastLayerNormInput(
    LayerNormTensor layerNormTensor, const aclIntArray* normalizedShape, const aclTensor*& weightContiguous, const aclTensor*& biasContiguous, aclOpExecutor* executor)
{
    if (layerNormTensor.weightOptional) {
        weightContiguous = l0op::Contiguous(layerNormTensor.weightOptional, executor);
        CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // 非目标设备按照原逻辑执行
        auto weightTensor = executor->ConvertToTensor(normalizedShape, DataType::DT_INT64);
        aclScalar* scalarOne = executor->AllocScalar(1);
        auto weightOptionalDtype = layerNormTensor.biasOptional ? layerNormTensor.biasOptional->GetDataType() : layerNormTensor.input->GetDataType();
        auto oneTensor = executor->ConvertToTensor(scalarOne, weightOptionalDtype);
        weightContiguous = l0op::Fill(weightTensor, oneTensor, normalizedShape, executor);
        CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (layerNormTensor.biasOptional) {
        biasContiguous = l0op::Contiguous(layerNormTensor.biasOptional, executor);
        CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // 非目标设备按照原逻辑执行
        auto biasTensor = executor->ConvertToTensor(normalizedShape, DataType::DT_INT64);
        aclScalar* scalarZero = executor->AllocScalar(0);
        auto biasOptionalDtype = layerNormTensor.weightOptional ? layerNormTensor.weightOptional->GetDataType() : layerNormTensor.input->GetDataType();
        auto zeroTensor = executor->ConvertToTensor(scalarZero, biasOptionalDtype);
        biasContiguous = l0op::Fill(biasTensor, zeroTensor, normalizedShape, executor);
        CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus doFastLayerNorm(
    LayerNormTensor layerNormTensor, const aclIntArray* normalizedShape, aclOpExecutor* executor)
{
    // 固定写法，将输入转换成连续的tensor
    auto inputContiguous = l0op::Contiguous(layerNormTensor.input, executor);
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 构造新的weightContiguous
    const aclTensor* weightContiguous;

    // 构造新的biasContiguous
    const aclTensor* biasContiguous;

    auto ret = doFastLayerNormInput(layerNormTensor, normalizedShape, weightContiguous, biasContiguous, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    std::array<aclTensor*, LAYER_NORM_OUT_NUM> fastLayerNormOut = {nullptr, nullptr, nullptr};

    l0op::LayerNormV4Params V4_params = {inputContiguous, normalizedShape, weightContiguous, biasContiguous};

    fastLayerNormOut = l0op::LayerNormV4(V4_params, layerNormTensor.eps, executor);

    // 处理第一个输出
    auto outRes = fastLayerNormOut[Y_INDEX];
    CHECK_RET(outRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outViewCopy = l0op::ViewCopy(outRes, layerNormTensor.out, executor);
    CHECK_RET(outViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 处理第三个输出
    if (layerNormTensor.rstdOutOptional) {
        auto rstdRes = fastLayerNormOut[RSTD_INDEX];
        CHECK_RET(rstdRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto rstdCast = l0op::Cast(rstdRes, layerNormTensor.rstdOutOptional->GetDataType(), executor);
        CHECK_RET(rstdCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto rstdViewCopy = l0op::ViewCopy(rstdCast, layerNormTensor.rstdOutOptional, executor);
        CHECK_RET(rstdViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 处理第二个输出
    if (layerNormTensor.meanOutOptional) {
        auto meanRes = fastLayerNormOut[MEAN_INDEX];
        CHECK_RET(meanRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto meanCast = l0op::Cast(meanRes, layerNormTensor.meanOutOptional->GetDataType(), executor);
        CHECK_RET(meanCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto meanViewCopy = l0op::ViewCopy(meanCast, layerNormTensor.meanOutOptional, executor);
        CHECK_RET(meanViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFastLayerNormGetWorkspaceSize(
    const aclTensor* input, const aclIntArray* normalizedShape, const aclTensor* weightOptional,
    const aclTensor* biasOptional, double eps, aclTensor* out, aclTensor* meanOutOptional, aclTensor* rstdOutOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnFastLayerNorm, DFX_IN(input, normalizedShape, weightOptional, biasOptional, eps),
        DFX_OUT(out, meanOutOptional, rstdOutOptional));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 检查参数是否为空指针
    CHECK_RET(CheckFastNotNull(input, normalizedShape, out), ACLNN_ERR_PARAM_NULLPTR);

    // 固定写法，参数检查
    auto ret1 = CheckParams(
        input, normalizedShape, weightOptional, biasOptional, out, meanOutOptional, rstdOutOptional);
    CHECK_RET(ret1 == ACLNN_SUCCESS, ret1);

    // 根据input_shape和normalizedShape的关系获取非reduce轴和reduce轴的shape
    auto inputShape = input->GetViewShape();
    const size_t inputDim = inputShape.GetDimNum();
    const size_t normDim = normalizedShape->Size();
    const size_t beginAxis = inputDim - normDim;
    int64_t M = 1;
    int64_t N = 1;
    for (size_t index = beginAxis; index < inputDim; index++) {
        N *= inputShape.GetDim(index);
    }
    for (size_t index = 0; index < beginAxis; index++) {
        M *= inputShape.GetDim(index);
    }

    // 空tensor场景处理，区分reduce轴是否为0
    if (N == 0) {
        if (meanOutOptional) {
            ret1 = op::ProcessEmptyTensorWithValue(meanOutOptional, 0, uniqueExecutor.get());
            CHECK_RET(ret1 == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        }
        if (rstdOutOptional) {
            ret1 = op::ProcessEmptyTensorWithValue(
                rstdOutOptional, std::numeric_limits<float>::quiet_NaN(), uniqueExecutor.get());
            CHECK_RET(ret1 == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        }
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (M == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    LayerNormTensor layerNormTensor = {input, weightOptional, biasOptional, eps, out, meanOutOptional, rstdOutOptional};
    auto ret2 = doFastLayerNorm(layerNormTensor, normalizedShape, uniqueExecutor.get());
    CHECK_RET(ret2 == ACLNN_SUCCESS, ret2);
    
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFastLayerNorm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFastLayerNorm);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
