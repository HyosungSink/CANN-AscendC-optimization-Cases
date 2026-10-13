/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file deformable_conv2d.cpp
 * \brief
 */
#include "deformable_conv2d.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op
{
OP_TYPE_REGISTER(DeformableConv2d);
OP_TYPE_REGISTER(DeformableOffsets);

static const int64_t INDEX_ZERO = 0;
static const int64_t INDEX_ONE = 1;
static const int64_t INDEX_TWO = 2;
static const int64_t INDEX_THREE = 3;

const std::tuple<aclTensor*, aclTensor*> DeformableConv2d(const aclTensor* x, const aclTensor* weight,
                                                          const aclTensor* offset, const aclTensor* biasOptional,
                                                          const aclIntArray* kernelSize, const aclIntArray* stride,
                                                          const aclIntArray* padding, const aclIntArray* dilation,
                                                          int64_t groups, int64_t deformableGroups, bool modulated,
                                                          aclOpExecutor* executor)
{
    L0_DFX(DeformableConv2d, x, weight, offset, biasOptional, kernelSize, stride, padding, dilation, groups,
           deformableGroups, modulated);
    op::Shape normalOutput = offset->GetViewShape();
    normalOutput.SetDim(INDEX_ONE, offset->GetViewShape().GetDim(INDEX_ONE));
    normalOutput.SetDim(INDEX_TWO, weight->GetViewShape().GetDim(INDEX_ZERO));
    normalOutput.SetDim(INDEX_THREE, offset->GetViewShape().GetDim(INDEX_TWO));
    auto out = executor->AllocTensor(normalOutput, x->GetDataType(), x->GetStorageFormat());

    op::Shape deformableOutput = x->GetViewShape();
    deformableOutput.SetDim(INDEX_ONE, offset->GetViewShape().GetDim(INDEX_ONE) * (*kernelSize)[0]);
    deformableOutput.SetDim(INDEX_TWO, offset->GetViewShape().GetDim(INDEX_TWO) * (*kernelSize)[1]);
    deformableOutput.SetDim(INDEX_THREE, x->GetViewShape().GetDim(INDEX_THREE));
    auto deformOut = executor->AllocTensor(deformableOutput, x->GetDataType(), x->GetStorageFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        DeformableConv2d, OP_INPUT(x, weight, offset, biasOptional), OP_OUTPUT(out, deformOut),
        OP_ATTR(kernelSize, stride, padding, dilation, groups, deformableGroups, modulated));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "DeformableConv2dAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tuple<aclTensor*, aclTensor*>(out, deformOut);
}

const aclTensor *DeformableOffsetsNHWC(const aclTensor *x, const aclTensor *offset, const aclIntArray *kernelSize,
                                       op::DataType outputDtype, const aclIntArray *stride,const aclIntArray *padding,
                                       const aclIntArray *dilation, int64_t deformableGroups, bool modulated,
                                       aclOpExecutor* executor)
{
    L0_DFX(DeformableOffsetsNHWC, x, offset, kernelSize, outputDtype, stride, padding, dilation, deformableGroups,
        modulated);
    if (x->GetStorageFormat() != Format::FORMAT_NHWC || offset->GetStorageFormat() != Format::FORMAT_NHWC) {
        OP_LOGE(ACLNN_ERR_INNER, "L0 func only support inputs with NHWC format.");
        return nullptr;
    }
    ge::AscendString originalFormat = op::ToString(x->GetOriginalFormat());
    const char *dataFormat = originalFormat.GetString();
    auto output = executor->AllocTensor(outputDtype, x->GetStorageFormat(), x->GetOriginalFormat());
    auto ret = INFER_SHAPE(DeformableOffsets, OP_INPUT(x, offset), OP_OUTPUT(output),
        OP_ATTR(stride, padding, kernelSize, dilation, dataFormat, deformableGroups, modulated));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
        output = nullptr;
        return nullptr;
    }

    ADD_TO_LAUNCHER_LIST_AICORE(DeformableOffsets, OP_INPUT(x, offset), OP_OUTPUT(output),
        OP_ATTR(stride, padding, kernelSize, dilation, dataFormat, deformableGroups, modulated));
    return output;
}
}  // namespace l0op
