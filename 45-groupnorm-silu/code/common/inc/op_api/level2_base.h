/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LEVEL2_BASE_H_
#define LEVEL2_BASE_H_

#include <algorithm>
#include "op_api_def.h"
#include "aclnn_util.h"
#include "aclnn/aclnn_base.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace op {

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LISTS = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};
static const int64_t UPSAMPLE_EXPECT_SIZE = 3;
static const int64_t UPSAMPLE_DIM_LIMIT = 5;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t DIM_FOUR = 4;

// 检查1个输入和1个输出是否是空指针
inline static bool CheckNotNull2Tensor(const aclTensor* t0, const aclTensor* t1)
{
    OP_CHECK_NULL(t0, return false);
    OP_CHECK_NULL(t1, return false);

    return true;
}
inline static bool CheckNotNull3Tensor(const aclTensor* t0, const aclTensor* t1, const aclTensor* t2)
{
    // 检查输入是否是空指针
    OP_CHECK_NULL(t0, return false);
    OP_CHECK_NULL(t1, return false);
    // 检查输入是否是空指针
    OP_CHECK_NULL(t2, return false);

    return true;
}
// 检查3个输入和1个输出是否是空指针
inline static bool CheckNotNull4Tensor(const aclTensor* t0, const aclTensor* t1, const aclTensor* t2, const aclTensor* t3)
{
    // 检查输入是否是空指针
    OP_CHECK_NULL(t0, return false);
    OP_CHECK_NULL(t1, return false);
    OP_CHECK_NULL(t2, return false);
    // 检查输入是否是空指针
    OP_CHECK_NULL(t3, return false);

    return true;
}

inline static bool CheckNotNull2In1Out(
    const aclTensor* gradOutput, const aclTensor* input, const aclTensor* grid, const aclTensor* inputGrad,
    const aclTensor* gridGrad)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(grid, return false);
    OP_CHECK_NULL(inputGrad, return false);
    OP_CHECK_NULL(gridGrad, return false);
    return true;
}

inline static bool CheckNotNull2In2Out(
    const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize, const aclTensor* gradInput)
{
    OP_CHECK_NULL(gradOut, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(inputSize, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

/**
 * 1. 1个输入1个输出
 * 2. 输入输出的shape必须一致
 * 3. 输入的维度必须小于等于8
 */
inline static bool CheckSameShape1In1Out(const aclTensor* self, const aclTensor* out)
{
    // self和out的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    // self的维度必须小于等于8
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

    return true;
}

inline static bool CheckShapeCumMinMax(const aclTensor* self, const aclTensor* valuesOut, const aclTensor* indicesOut)
{
    // 所有输入的维度都不能超过8
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

    // self和valuesOut、indicesOut的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, valuesOut, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(self, indicesOut, return false);
    return true;
}

inline static bool CheckSameShapeNotlimit1In1Out(const aclTensor* self, const aclTensor* out)
{
    // self和out的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    return true;
}

inline static bool CheckShapeNotlimitDim1In1Out(const aclTensor* self)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}

// 检查1个输入和1个输出的数据类型是否在算子的支持列表内
static inline bool CheckDtypeValid1In1Out(
    const aclTensor* self, const aclTensor* out, const std::initializer_list<op::DataType>& dtypeSupportList,
    const std::initializer_list<op::DataType>& dtypeOutList)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
    // 检查输出的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeOutList, return false);

    return true;
}

// 检查1个输入和1个输出的数据类型是否在算子的支持列表内
static inline bool CheckDtypeValid1In1OutScalar(
    const aclTensor* self, const aclScalar* attrScalar, const aclTensor* out,
    const std::initializer_list<op::DataType>& dtypeSupportList,
    const std::initializer_list<op::DataType>& attrScalarSupportList,
    const std::initializer_list<op::DataType>& dtypeOutList)
{
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
    // 检查attr的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(attrScalar, attrScalarSupportList, return false);
    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeOutList, return false);

    // 检查self和out的数据类型是否一致
    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

    return true;
}

// 检查1个输入和1个输出的数据类型是否在算子的支持列表内
static inline bool CheckDtypeValid1In1OutTensor(
    const aclTensor* self, const aclTensor* attrTensor, const aclTensor* out,
    const std::initializer_list<op::DataType>& dtypeSupportList,
    const std::initializer_list<op::DataType>& attrTensorSupportList,
    const std::initializer_list<op::DataType>& dtypeOutList)
{
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
    // 检查attr的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(attrTensor, attrTensorSupportList, return false);
    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeOutList, return false);

    // 检查self和out的数据类型是否一致
    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

    return true;
}

static inline bool CheckDtypeValid1In1OutMatch(
    const aclTensor* self, const aclTensor* valueOut, const aclTensor* inverseOut, const aclTensor* countsOut,
    const std::initializer_list<op::DataType>& dtypeSupportList)
{
    // 检查self的数据类型是否在UniqueConsecutive算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);

    // 检查inverseOut数据类型
    OP_CHECK_DTYPE_NOT_MATCH(inverseOut, op::DataType::DT_INT64, return false);

    // 检查countsOut数据类型
    OP_CHECK_DTYPE_NOT_MATCH(countsOut, op::DataType::DT_INT64, return false);

    // 检查self与valueOut数据类型一致
    OP_CHECK_DTYPE_NOT_MATCH(valueOut, self->GetDataType(), return false);

    // 检查inverseOut与countsOut数据类型一致
    OP_CHECK_DTYPE_NOT_MATCH(inverseOut, countsOut->GetDataType(), return false);
    return true;
}

/**
 * l1: DAV_2201 芯片，该算子支持的数据类型列表
 * l2: 其他芯片，该算子支持的数据类型列表
 */
inline static const std::initializer_list<DataType>& GetDtypeSupportListV1(
    const std::initializer_list<op::DataType>& l1, const std::initializer_list<op::DataType>& l2)
{
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        return l1;
    } else {
        return l2;
    }
}

/**
 * l1: DAV_2201 和 DAV_3510芯片，该算子支持的数据类型列表
 * l2: 其他芯片，该算子支持的数据类型列表
 */
inline static const std::initializer_list<DataType>& GetDtypeSupportListV2(
    const std::initializer_list<op::DataType>& l1, const std::initializer_list<op::DataType>& l2)
{
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        return l1;
    } else {
        return l2;
    }
}

inline static const std::initializer_list<op::DataType> GetDtypeSupportListV3(
    const std::initializer_list<op::DataType>& l1, const std::initializer_list<op::DataType>& l2)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    switch (npuArch) {

        case NpuArch::DAV_2201: {
            return l1;
        }
        default: {
            // 如果既不是1971也不是1980，先暂且默认当做1971处理
            return l1;
        }
    }
}

static inline aclnnStatus CheckArrayDataAvgPoolBackWard(
    const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding)
{
    for (uint64_t i = 0; i < kernelSize->Size(); i++) {
        auto size = (*kernelSize)[i];
        if (size <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kernelSize [%lu] data [%li] is less than or equal to 0", i, size);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    for (uint64_t i = 0; i < stride->Size(); i++) {
        auto size = (*stride)[i];
        if (size <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stride [%lu] data [%li] is less than or equal to 0", i, size);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (kernelSize->Size() != padding->Size()) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    for (uint64_t i = 0; i < kernelSize->Size(); i++) {
        auto halfKsize = (*kernelSize)[i] / 2;
        auto padSize = (*padding)[i];
        if (halfKsize < padSize) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "padding [%lu] data [%li] should less than kernelSize / 2 [%ld]", i, padSize,
                halfKsize);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (padSize < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding [%lu] data [%li] is less than 0", i, padSize);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

inline static bool CheckPaddingValidAvgPool2D(const aclIntArray* kernelSize, const aclIntArray* padding)
{
    const int64_t kKernelSizeHIdx = 0;
    const int64_t kKernelSizeWIdx = 1;
    const int64_t kPaddingUpIdx = 0;
    const int64_t kPaddingLeftIdx = 1;
    const int64_t kernelH = (*kernelSize)[kKernelSizeHIdx];
    const int64_t kernelPaddingSize = 1;
    const int64_t MULTIPLIER = 2;
    // 1表示kernelSize长度为1
    const int64_t kernelW =
        kernelSize->Size() == kernelPaddingSize ? (*kernelSize)[kKernelSizeHIdx] : (*kernelSize)[kKernelSizeWIdx];
    OP_CHECK(padding->Size() != 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding is empty"), return false);
    const int64_t paddingH = (*padding)[kPaddingUpIdx];
    const int64_t paddingW =
        padding->Size() == kernelPaddingSize ? (*padding)[kPaddingUpIdx] : (*padding)[kPaddingLeftIdx];
    // MULTIPLIER 表示paddingH不能大于kernelH的一半，下同
    if (kernelH < MULTIPLIER * paddingH) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "value of paddingH should be at most half of kernelH. Actual: paddingH is [%ld],kernelH is [%ld].",
            paddingH, kernelH);
        return false;
    }
    if (kernelW < MULTIPLIER * paddingW) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "value of paddingW should be at most half of kernelW. Actual: paddingW is [%ld],"
            "kernelW is [%ld].",
            paddingW, kernelW);
        return false;
    }
    return true;
}

static inline bool CheckDtypeValid1Out1In(const aclTensor* gradOut, const aclTensor* gradInput)
{
    // 检查gradOut的数据类型是否在ResizeNearestNeighborV2Grad算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, DTYPE_SUPPORT_LISTS, return false);
    // 检查gradInput的数据类型是否与gradOut一致
    OP_CHECK_DTYPE_NOT_MATCH(gradOut, gradInput->GetDataType(), return false);
    return true;
}

inline static bool CheckUpsampleShape(const aclTensor* gradOut, const aclIntArray* outputSize, const aclIntArray* inputSize)
{
    size_t outputSizeNum = outputSize->Size();
    size_t inputSizeNum = inputSize->Size();
    OP_CHECK_WRONG_DIMENSION(gradOut, UPSAMPLE_DIM_LIMIT, return false);
    OP_CHECK(
        outputSizeNum == UPSAMPLE_EXPECT_SIZE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected output_size equals to 3, but got size %zu", outputSizeNum),
        return false);

    OP_CHECK(
        inputSizeNum == UPSAMPLE_DIM_LIMIT,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected input_size equals to 5, but got size %zu", inputSizeNum),
        return false);
    return true;
}

static inline bool CheckPromoteTypeGeluBackward(const aclTensor* gradOutput, const aclTensor* self, op::DataType &promoteType,
    const std::initializer_list<DataType> dtypeSupportList)
{
    // 检查self和gradOutput能否做数据类型推导
    promoteType = op::PromoteType(gradOutput->GetDataType(), self->GetDataType());
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "GradOutput dtype %s and Self dtype %s can not be promoted.",
            op::ToString(gradOutput->GetDataType()).GetString(), op::ToString(self->GetDataType()).GetString());
        return false;
    }

    if (((!(GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201)) && (promoteType == op::DataType::DT_BF16))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dtype of gelu is not support bfloat16 in current socversion.");
        return false;
    }

    // 检查promoteType的数据类型是否在支持列表内
    if (!CheckType(promoteType, dtypeSupportList)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "promoteType dtype %s should be in dtype support list %s.",
            op::ToString(promoteType).GetString(), op::ToString(dtypeSupportList).GetString());
        return false;
    }
    return true;
}

// 循环判断size是否相等
inline static bool CheckSizeLoop(size_t dimNum, Shape gradOutShape, FVector<int64_t>& fullOutputSize)
{
    for (size_t i = 0; i < dimNum; ++i) {
        if (gradOutShape.GetDim(i) != fullOutputSize[i]) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Expected grad_output to have the same shape as output;"
                " output.size(%zu) = %ld but got grad_output.size(%zu) = %ld",
                i, fullOutputSize[i], i, gradOutShape.GetDim(i));
            return false;
        }
    }
    return true;
}

static inline bool CheckInpuNullTensorMaxUnPool3D(const aclTensor* self)
{
    auto inputShape = self->GetViewShape();
    size_t dimNum = inputShape.GetDimNum();

    for (size_t i = 1; i < dimNum; ++i) {
        if (inputShape.GetDim(i) <= 0) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "max_unpool3d(): Expected input to have non-zero size for non-batch dimensions, "
                "but got %zu with dimension %zu being empty.",
                dimNum, i);
            return false;
        }
    }
    return true;
}

static inline bool CheckIntArrayShapeMaxUnPool3D(
    const aclTensor* self, const aclIntArray* outputSize, const aclIntArray* stride, const aclIntArray* padding)
{
    constexpr size_t NCDHW_DIM_NUM = 5;
    constexpr int64_t EXPECT_SIZE = 3;

    OP_CHECK(
        outputSize->Size() == EXPECT_SIZE,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "There should be exactly three elements (depth, height, width) in "
            "output_size, but got %zu elements",
            outputSize->Size()),
        return false);
    OP_CHECK(
        stride->Size() == EXPECT_SIZE,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "There should be exactly three elements (depth, height, width) in "
            "stride, but got %zu elements",
            stride->Size()),
        return false);
    OP_CHECK(
        padding->Size() == EXPECT_SIZE,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "There should be exactly three elements (depth, height, width) in "
            "padding, but got %zu elements",
            padding->Size()),
        return false);

    for (size_t i = 0; i < EXPECT_SIZE; ++i) {
        if ((*stride)[i] <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "strides[%zu] shoule be greater than zero, but got %ld.", i, (*stride)[i]);
            return false;
        }
        if ((*outputSize)[i] <= 0) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "outputSize[%zu] should be greater than zero, but got %ld.", i,
                (*outputSize)[i]);
            return false;
        }
    }

    int64_t dimD = self->GetViewShape().GetDim(DIM_ONE);
    int64_t dimH = self->GetViewShape().GetDim(DIM_TWO);
    int64_t dimW = self->GetViewShape().GetDim(DIM_THREE);

    if (self->GetViewShape().GetDimNum() == NCDHW_DIM_NUM) {
        dimD = self->GetViewShape().GetDim(DIM_TWO);
        dimH = self->GetViewShape().GetDim(DIM_THREE);
        dimW = self->GetViewShape().GetDim(DIM_FOUR);
    }

    OP_CHECK(
        ((*outputSize)[DIM_ZERO] * (*outputSize)[DIM_ONE] * (*outputSize)[DIM_TWO]) >= (dimD * dimH * dimW),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "The output dimensions are of size %ld x %ld x %ld, "
            "should be greater than or equal to self of size %ld x %ld x %ld.",
            (*outputSize)[DIM_ZERO], (*outputSize)[DIM_ONE], (*outputSize)[DIM_TWO], dimD, dimH, dimW),
        return false);
    return true;
}

inline static void CalculateValuesSliceUpper(const aclTensor* self, int64_t& dim, int64_t& start, int64_t& end)
{
    if (dim < 0) {
        dim = dim + self->GetViewShape().GetDimNum();
    }
    if (start < 0) {
        start = start + self->GetViewShape().GetDim(dim);
    }
    if (end < 0) {
        end = end + self->GetViewShape().GetDim(dim);
    }
}

inline static void CalculateValuesSliceLower(const aclTensor* self, int64_t& dim, int64_t& start, int64_t& end)
{
    if (start < 0) {
        start = 0;
    } else if (start >= self->GetViewShape().GetDim(dim)) {
        start = self->GetViewShape().GetDim(dim);
    }
    if (end < start) {
        end = start;
    } else if (end >= self->GetViewShape().GetDim(dim)) {
        end = self->GetViewShape().GetDim(dim);
    }
}

static inline bool CheckDimValueWithUnique(const aclTensor* self, int64_t dim)
{
    int64_t dimSize = self->GetViewShape().GetDimNum();
    int64_t dimMin = std::min(-1 * dimSize, dimSize - 1);
    int64_t dimMax = std::max(-1 * dimSize, dimSize - 1);
    if (dim > dimMax || dim < dimMin) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The param of dim must be in the range of [%ld, %ld], but got %ld.", dimMin,
            dimMax, dim);
        return false;
    }
    return true;
}

} // namespace op
#ifdef __cplusplus
}
#endif
#endif // LEVEL2_BASE_H_
