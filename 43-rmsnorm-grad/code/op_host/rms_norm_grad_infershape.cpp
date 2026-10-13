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
 * \file rms_norm_grad_infershape.cpp
 * \brief
 */

#include "op_common/log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShape4RmsNormGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4RmsNormGrad.");
    const gert::Shape* x_shape = context->GetInputShape(1);
    const gert::Shape* gamma_shape = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);

    // get output shapes
    gert::Shape* dx_shape = context->GetOutputShape(0);
    gert::Shape* dgamma_shape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, dx_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, dgamma_shape);
    *dx_shape = *x_shape;
    *dgamma_shape = *gamma_shape;

    OP_LOGD(context, "End to do InferShape4RmsNormGrad.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4RmsNormGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4RmsNormGrad");
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, DT_FLOAT);
    OP_LOGD(context, "End to do InferDataType4RmsNormGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RmsNormGrad).InferShape(InferShape4RmsNormGrad).InferDataType(InferDataType4RmsNormGrad);
} // namespace ops