/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "add_rms_norm_quant.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
constexpr int IDX_INT8 = 2;

OP_TYPE_REGISTER(AddRmsNormQuant);

const std::array<aclTensor*, ADD_RMS_NORM_QUANT_OUT_NUM> AddRmsNormQuantInner(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* scales1,
    const aclTensor* scales2Optional, const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional,
    const aclTensor* betaOptional, int64_t axis, double epsilon, bool divMode, int dstType, aclOpExecutor* executor) 
{
    bool isDual = (nullptr != scales2Optional);
    Shape dummyShape({1});

    auto y1Out = executor->AllocTensor(x1->GetViewShape(), op::DataType(dstType), op::Format::FORMAT_ND);
    auto xOut = executor->AllocTensor(x1->GetViewShape(), x1->GetDataType(), op::Format::FORMAT_ND);
    auto y2Out = (isDual) ? executor->AllocTensor(x1->GetViewShape(), op::DataType(dstType), op::Format::FORMAT_ND) :
                            executor->AllocTensor(dummyShape, op::DataType(dstType), op::Format::FORMAT_ND);
    OP_LOGD(
        "y1Out=[%s], y2Out=[%s], xOut=[%s].", op::ToString(y1Out->GetViewShape()).GetString(),
        op::ToString(y2Out->GetViewShape()).GetString(), op::ToString(xOut->GetViewShape()).GetString());

    ADD_TO_LAUNCHER_LIST_AICORE(
        AddRmsNormQuant,
        OP_INPUT(x1, x2, gamma, scales1, scales2Optional, zeroPoints1Optional, zeroPoints2Optional, betaOptional),
        OP_OUTPUT(y1Out, y2Out, xOut), OP_ATTR(axis, static_cast<float>(epsilon), divMode, dstType));
    return {y1Out, y2Out, xOut};
}

const std::array<aclTensor*, ADD_RMS_NORM_QUANT_OUT_NUM> AddRmsNormQuant(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* scales1,
    const aclTensor* scales2Optional, const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional,
    const aclTensor* betaOptional, int64_t axis, double epsilon, bool divMode, aclOpExecutor* executor)
{
    L0_DFX(
        AddRmsNormQuant, x1, x2, gamma, scales1, scales2Optional, zeroPoints1Optional, zeroPoints2Optional,
        betaOptional, axis, epsilon, divMode);
    OP_LOGD("AddRmsNormQuant L0_DFX.");

    int dstType = IDX_INT8;
    return AddRmsNormQuantInner(x1, x2, gamma, scales1, scales2Optional, zeroPoints1Optional, zeroPoints2Optional, 
                                betaOptional, axis, epsilon,divMode, dstType, executor);
}

const std::array<aclTensor*, ADD_RMS_NORM_QUANT_OUT_NUM> AddRmsNormQuant(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* scales1,
    const aclTensor* scales2Optional, const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional,
    const aclTensor* betaOptional, int64_t axis, double epsilon, bool divMode, int dstType, aclOpExecutor* executor)
{
    L0_DFX(
        AddRmsNormQuant, x1, x2, gamma, scales1, scales2Optional, zeroPoints1Optional, zeroPoints2Optional,
        betaOptional, axis, epsilon, divMode, dstType);
    OP_LOGD("AddRmsNormQuant L0_DFX.");

    return AddRmsNormQuantInner(x1, x2, gamma, scales1, scales2Optional, zeroPoints1Optional, zeroPoints2Optional, 
                                betaOptional, axis, epsilon,divMode, dstType, executor);
}

OP_TYPE_REGISTER(AddRmsNormQuantV2);
const std::array<aclTensor*, ADD_RMS_NORM_QUANT_OUT_NUM_V2> AddRmsNormQuantV2(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* bias,
    const aclTensor* scales1, const aclTensor* scales2Optional,
    const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional, aclTensor* resOut, aclTensor* xOut,
    int64_t axis, double epsilon, bool divMode, aclOpExecutor* executor)
{
    OP_LOGD("AddRmsNormQuantV2 L0 Start.");
    bool isDual = (nullptr != scales2Optional);
    Shape dummyShape({1});

    L0_DFX(AddRmsNormQuantV2, x1, x2, gamma, bias, scales1, scales2Optional,
           zeroPoints1Optional, zeroPoints2Optional, axis, epsilon, divMode);
    OP_LOGD("AddRmsNormQuantV2 L0_DFX.");

    auto y1Out = executor->AllocTensor(x1->GetViewShape(), DataType::DT_INT8, op::Format::FORMAT_ND);

    resOut = executor->AllocTensor(x1->GetViewShape(), x1->GetDataType(), op::Format::FORMAT_ND);
    OP_LOGD("resOut shape=[%s].", op::ToString(resOut->GetViewShape()).GetString());

    xOut = executor->AllocTensor(x1->GetViewShape(), x1->GetDataType(), op::Format::FORMAT_ND);
    OP_LOGD("xOut shape=[%s].", op::ToString(xOut->GetViewShape()).GetString());

    auto y2Out = (isDual) ? executor->AllocTensor(x1->GetViewShape(), DataType::DT_INT8, op::Format::FORMAT_ND)
                          : executor->AllocTensor(dummyShape, DataType::DT_INT8, op::Format::FORMAT_ND);
    OP_LOGD("y1Out=[%s], y2Out=[%s].", op::ToString(y1Out->GetViewShape()).GetString(),
                                                  op::ToString(y2Out->GetViewShape()).GetString());

    ADD_TO_LAUNCHER_LIST_AICORE(AddRmsNormQuantV2,
                                OP_INPUT(x1, x2, gamma, scales1, scales2Optional,
                                         zeroPoints1Optional, zeroPoints2Optional, bias),
                                OP_OUTPUT(y1Out, y2Out, xOut, resOut),
                                OP_ATTR(axis, static_cast<float>(epsilon), divMode));
    OP_LOGI("AddRmsNormQuantV2 Launch finish.");

    return {y1Out, y2Out, xOut, resOut};
}

} // namespace l0op
