/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LEVEL2_BASE_LOSS_H_
#define LEVEL2_BASE_LOSS_H_

#include "../../stub/op_api/aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "opdev/shape_utils.h"
#include "../../stub/op_api/level0/fill.h"
#include "../../stub/op_api/aclnn_kernels/reshape.h"
#include "level2_base.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace op
{
// 针对reduction!='none'且self为空tensor场景，对out按照给定值进行填充
inline static aclnnStatus CheckFillScalarLoss(aclTensor* out, float val, aclOpExecutor* executor)
{
    FVector<int64_t> tmp = {1};
    auto dims = executor->ConvertToTensor(tmp.data(), tmp.size(), op::DataType::DT_INT64);
    auto shapeArray = executor->AllocIntArray(tmp.data(), tmp.size());
    CHECK_RET(shapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

    FVector<float> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());
    auto fillOut = l0op::Fill(dims, valTensor, shapeArray, executor);
    CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(fillOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

inline static aclIntArray* GetBroadcastShapeLossBackward(const op::Shape broadcastShape, aclOpExecutor* executor)
{
    int64_t tensorSize = static_cast<int64_t>(broadcastShape.GetDimNum());
    std::vector<int64_t> tensorShape(tensorSize);
    for (int i = 0; i < tensorSize; i++) {
        tensorShape[i] = broadcastShape[i];
    }
    return executor->AllocIntArray(tensorShape.data(), tensorSize);
}

inline static bool CheckDtypeValidMseLoss(const aclTensor* self, const aclTensor* target, const aclTensor* out,
                                          const std::initializer_list<op::DataType>& l1,
                                          const std::initializer_list<op::DataType>& l2)
{
    auto supportList = GetDtypeSupportListV2(l1, l2);
    // 检查self和target做数据类型推导后的数据类型是否在支持列表内
    op::DataType promoteType = op::PromoteType(self->GetDataType(), target->GetDataType());
    if (!CheckType(promoteType, supportList)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected self dtype [%s] and target dtype [%s] to be promotable but check failed.",
                ToString(self->GetDataType()).GetString(), ToString(target->GetDataType()).GetString());
        return false;
    }

    // 检查self, target和out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(target, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
    return true;
}

inline static bool CheckReductionMseLoss(int64_t reduction, int64_t REDUCTION_SUM_NUM, int64_t REDUCTION_NONE_NUM)
{
    if (reduction > REDUCTION_SUM_NUM || reduction < REDUCTION_NONE_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected reduction to be between 0 and 2, but got %ld.", reduction);
        return false;
    }
    return true;
}

}  // namespace op
#ifdef __cplusplus
}
#endif
#endif  // LEVEL2_BASE_LOSS_H_