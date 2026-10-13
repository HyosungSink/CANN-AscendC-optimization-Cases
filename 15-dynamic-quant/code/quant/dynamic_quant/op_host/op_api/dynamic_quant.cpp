/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dynamic_quant.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(DynamicQuant);

static op::Shape GetOutPutShape(const aclTensor* x)
{
    op::Shape outputShape;
    size_t dimNum = x->GetViewShape().GetDimNum();
    for (size_t i = 0; i < dimNum - 1; i++) {
        outputShape.AppendDim(x->GetViewShape().GetDim(i));
    }
    return outputShape;
}

std::tuple<aclTensor*, aclTensor*> DynamicQuant(
    const aclTensor* x, const aclTensor* smoothScalesOptional, const aclTensor* groupIndexsOptional, int32_t dstType,
    aclOpExecutor* executor)
{
    L0_DFX(DynamicQuant, x, smoothScalesOptional, groupIndexsOptional);
    auto yOut = executor->AllocTensor(
        x->GetStorageShape(), x->GetViewShape(), op::DataType(dstType), x->GetStorageFormat(), x->GetOriginalFormat());

    auto scaleShape = GetOutPutShape(x);
    auto scaleOut = executor->AllocTensor(scaleShape, op::DataType::DT_FLOAT);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        DynamicQuant, OP_INPUT(x, smoothScalesOptional, groupIndexsOptional), OP_OUTPUT(yOut, scaleOut),
        OP_ATTR(dstType));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "DynamicQuant launch kernel failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tuple<aclTensor*, aclTensor*>(yOut, scaleOut);
}

} // namespace l0op