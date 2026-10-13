/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LEVEL2_BASE_CALCULATION_H_
#define LEVEL2_BASE_CALCULATION_H_

#include <bitset>
#include "op_api_def.h"
#include "aclnn/aclnn_base.h"
#include "opdev/shape_utils.h"
#include "../../stub/op_api/level0/fill.h"
#include "../../stub/op_api/aclnn_kernels/reshape.h"
#include "../../stub/op_api/aclnn_kernels/common/op_error_check.h"
#include "level2_base.h"


#ifdef __cplusplus
extern "C" {
#endif

namespace op {
static inline bool CheckBroadcastLogAddExp(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    // 检查self和other是否能broadcast，如可以求broadcast之后的shape
    OP_CHECK_BROADCAST(self, other, return false);
    op::Shape broadcastShape;
    BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape);

    // 检查self和other经过broadcast后的shape是否与out的shape一致
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape, return false);
    return true;
}

// 检查传入的reduction数值是否在可选范围内
static inline FVector<int64_t> FillTensorGetTmpDim(const aclTensor* out)
{
    FVector<int64_t> tmp;
    if (out->GetViewShape().GetDimNum() != 0) {
        size_t dimNum = out->GetViewShape().GetDimNum();
        for (size_t idx = 0; idx < dimNum; idx++) {
            int64_t tmpVal = out->GetViewShape().GetDim(idx);
            tmp.push_back(tmpVal);
        }
    } else {
        tmp.push_back(1);
    }
    return tmp;
}

static inline const aclTensor* FillTensorWithFalseValueWithInf(aclTensor* out, bool val, aclOpExecutor* executor)
{
    FVector<int64_t> tmp = FillTensorGetTmpDim(out);
    const aclTensor* dims = executor->ConvertToTensor(tmp.data(), tmp.size(), op::ToOpDataType(ACL_INT64));
    aclIntArray* shapeArray = executor->AllocIntArray(tmp.data(), tmp.size());
    FVector<bool> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());
    if (dims == nullptr || shapeArray == nullptr || valTensor == nullptr) {
        return nullptr;
    }
    return l0op::Fill(dims, valTensor, shapeArray, executor);
}

static inline aclIntArray* GetTensorShapeActivation(const aclTensor* x, aclOpExecutor* executor)
{
    auto shape = x->GetViewShape();
    int64_t dimSize = x->GetViewShape().GetDimNum();

    std::vector<int64_t> valuePerm(dimSize);
    for (int i = 0; i < dimSize; i++) {
        valuePerm[i] = shape[i];
    }
    auto perm = executor->AllocIntArray(valuePerm.data(), dimSize);
    return perm;
}

static inline bool CheckSocVersionIsSupportBf16Activation(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static inline bool CheckDtypeValidActivation(
    const aclTensor* self, const aclTensor* out, const std::initializer_list<op::DataType>& supportList)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    bool bf16flag = CheckSocVersionIsSupportBf16Activation();
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (!bf16flag && self->GetDataType() == op::DataType::DT_BF16) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self dtype %s is unsupported by the current SOC version [%s].",
            op::ToString(self->GetDataType()).GetString(), op::ToString(socVersion).GetString());
        return false;
    }
    return true;
}

static inline const aclTensor* ReshapeLongTensorActivation(
    const aclTensor* x, aclOpExecutor* executor, int originalDimSize, aclIntArray* valuePerm = nullptr)
{
    int64_t dimSize = x->GetViewShape().GetDimNum();
    if (static_cast<int64_t>(originalDimSize) == dimSize && dimSize <= static_cast<int64_t>(MAX_SUPPORT_DIMS_NUMS)) {
        return x;
    }

    auto reshapeSelf = l0op::Reshape(x, valuePerm, executor);
    return reshapeSelf;
}

static inline const aclTensor* ReshapeSelfValueGetActivation(
    const aclTensor* self, size_t dimSize, const aclTensor* selfContiguous, UniqueExecutor& uniqueExecutor)
{
    auto reshapeSelf = selfContiguous;
    auto shapeOri = self->GetViewShape();
    if (dimSize > MAX_SUPPORT_DIMS_NUMS) {
        int64_t AllDimValue = 1;
        for (size_t i = 0; i < dimSize; i++) {
            AllDimValue *= shapeOri[i];
        }
        int64_t AllDim[1] = {AllDimValue};
        auto shape1d = (uniqueExecutor)->AllocIntArray(AllDim, 1);
        reshapeSelf = ReshapeLongTensorActivation(selfContiguous, uniqueExecutor.get(), dimSize, shape1d);
    }
    return reshapeSelf;
}

static inline FVector<int64_t> FillScalarGetShapeValue(const aclTensor* out)
{
    FVector<int64_t> shape;
    size_t dimNum = out->GetViewShape().GetDimNum();
    for (size_t idx = 0; idx < dimNum; idx++) {
        int64_t tmpVal = out->GetViewShape().GetDim(idx);
        shape.push_back(tmpVal);
    }
    return shape;
}

static inline aclnnStatus CheckFillScalarShapeStdAndVar(aclTensor* out, float val, aclOpExecutor* executor)
{
    FVector<int64_t> shape = FillScalarGetShapeValue(out);
    auto dims = executor->ConvertToTensor(shape.data(), shape.size(), op::DataType::DT_INT64);
    CHECK_RET(dims != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto shapeArray = executor->AllocIntArray(shape.data(), shape.size());
    CHECK_RET(shapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

    FVector<float> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());
    CHECK_RET(valTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto fillOut = l0op::Fill(dims, valTensor, shapeArray, executor);
    CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(fillOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static inline uint64_t GetPosDimWithStd(int64_t dim, int64_t selfdimNum)
{
    if (selfdimNum <= 0) {
        selfdimNum = 1;
    }
    return dim >= 0 ? dim : dim + selfdimNum;
}

static inline void ExpectShapeInferWithDimMask(
    const op::Shape& selfShape, const aclIntArray* dim, bool keepDim, op::Shape& expectShape)
{
    bitset<MAX_MASK_LEN64> dimMask = bitset<MAX_MASK_LEN64>();

    // dim为空时，所有轴都视为mask，与竞品一致
    if (dim->Size() == 0) {
        dimMask.flip();
    }
    for (size_t i = 0; i < dim->Size(); i++) {
        int64_t index = GetPosDimWithStd(dim->operator[](i), selfShape.GetDimNum());
        // 前序已校验，此处dim不会重复
        dimMask.set(index);
    }

    for (size_t i = 0; i < selfShape.GetDimNum(); i++) {
        if (!dimMask[i]) {
            expectShape.AppendDim(selfShape.GetDim(i));
        } else if (keepDim) {
            // dim为空时 所有轴reduce
            expectShape.AppendDim(1);
        }
    }
}

static inline int64_t CalcShapeProdStdAndVarMean(const aclTensor* self, const aclIntArray* dim)
{
    auto selfViewShape = self->GetViewShape();
    int64_t shapeProd = 1;
    if (selfViewShape.GetDimNum() != 0) {
        // dim为all reduce
        if (dim->Size() == 0) {
            for (size_t i = 0; i < selfViewShape.GetDimNum(); i++) {
                shapeProd *= selfViewShape.GetDim(i);
            }
        } else {
            for (size_t i = 0; i < dim->Size(); i++) {
                shapeProd *= selfViewShape.GetDim(dim->operator[](i));
            }
        }
    }
    return shapeProd;
}

static inline aclIntArray* CalcDimWithVar(const aclTensor* self, aclOpExecutor* executor)
{
    FVector<int64_t> dimVector;
    auto selfViewShape = self->GetViewShape();
    size_t selfDimNum = selfViewShape.GetDimNum();
    for (size_t i = 0; i < selfDimNum; i++) {
        dimVector.push_back(static_cast<int64_t>(i));
    }
    return executor->AllocIntArray(dimVector.data(), dimVector.size());
}

static inline op::Shape ReduceShapeGetWithVar(const aclTensor* self, const aclIntArray* dim, bool keepdim)
{
    auto selfShape = self->GetViewShape();
    int64_t selfDimNum = selfShape.GetDimNum();
    bool reduceArray[MAX_SUPPORT_DIMS_NUMS] = {false, false, false, false, false, false, false, false};
    int64_t axis;

    op::Shape reduceShape;
    reduceShape.SetDimNum(0);
    if (dim != nullptr && dim->Size() != 0) {
        for (int64_t i = 0; i < static_cast<int64_t>(dim->Size()); i++) {
            axis = dim->operator[](i);
            if (axis < 0) {
                axis = axis + selfDimNum;
            }
            reduceArray[axis] = true;
        }
        for (int64_t i = 0; i < selfDimNum; i++) {
            if (!reduceArray[i]) {
                reduceShape.AppendDim(selfShape.GetDim(i));
            } else if (keepdim) {
                reduceShape.AppendDim(1);
            }
        }
    } else if (keepdim) {
        for (int64_t i = 0; i < selfDimNum; i++) {
            reduceShape.AppendDim(1);
        }
    }
    return reduceShape;
}

} // namespace op
#ifdef __cplusplus
}
#endif
#endif // LEVEL2_BASE_CALCULATION_H_