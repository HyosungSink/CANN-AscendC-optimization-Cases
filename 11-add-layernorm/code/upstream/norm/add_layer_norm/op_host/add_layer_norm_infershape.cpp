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
 * \file add_layer_norm_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

static constexpr int INPUT_NODE_NUM = 4;
static constexpr int OUTPUT_NODE_NUM = 4;
static constexpr int INPUT_NODE_OPTIONAL_NUM = 5;
static constexpr int X1_IDX = 0;
static constexpr int X2_IDX = 1;
static constexpr int GAMMA_IDX = 2;
static constexpr int BETA_IDX = 3;
static constexpr int Y_IDX = 0;
static constexpr int MEAN_IDX = 1;
static constexpr int RSTD_IDX = 2;
static constexpr int X_IDX = 3;

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4AddLayerNorm(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("InferShape4AddLayerNorm", "Context is nullptr, check failed.");
        return GRAPH_FAILED;
    }
    if ((!(context->GetComputeNodeInputNum() == INPUT_NODE_NUM ||
           context->GetComputeNodeInputNum() == INPUT_NODE_OPTIONAL_NUM)) ||
        context->GetComputeNodeOutputNum() != OUTPUT_NODE_NUM) {
        return GRAPH_FAILED;
    }
    const gert::Shape* x1_shape = context->GetInputShape(X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    const gert::Shape* gamma_shape = context->GetInputShape(GAMMA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);
    gert::Shape* y_shape = context->GetOutputShape(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    gert::Shape* mean_shape = context->GetOutputShape(MEAN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, mean_shape);
    gert::Shape* rstd_shape = context->GetOutputShape(RSTD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, rstd_shape);
    gert::Shape* x_shape = context->GetOutputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    *y_shape = *x1_shape;
    *x_shape = *x1_shape;
    if (Ops::Base::IsUnknownRank(*x1_shape)) {
        *mean_shape = *x1_shape;
        *rstd_shape = *x1_shape;
        OP_LOGI(context, "End to do InferShape4AddLayerNorm, x1 shape is [-2].");
        return GRAPH_SUCCESS;
    }
    if (Ops::Base::IsUnknownRank(*gamma_shape)) {
        *mean_shape = *gamma_shape;
        *rstd_shape = *gamma_shape;
        OP_LOGI(context, "End to do InferShape4AddLayerNorm, gamma shape is [-2].");
        return GRAPH_SUCCESS;
    }
    if (x1_shape->GetDimNum() < gamma_shape->GetDimNum()) {
        OP_LOGE(
            context, "Dim num of x1 should be no less than dim num of gamma.");
        return GRAPH_FAILED;
    }
    auto shape(*x1_shape);
    shape.SetDim(shape.GetDimNum() - gamma_shape->GetDimNum(), 1);
    *mean_shape = shape;
    *rstd_shape = shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4AddLayerNorm(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4AddLayerNorm");
    if (context->GetInputDataType(X1_IDX) == context->GetInputDataType(X2_IDX)) {
        context->SetOutputDataType(Y_IDX, context->GetInputDataType(X1_IDX));
        context->SetOutputDataType(X_IDX, context->GetInputDataType(X1_IDX));
    } else {
        context->SetOutputDataType(Y_IDX, DT_FLOAT);
        context->SetOutputDataType(X_IDX, DT_FLOAT);
    }
    context->SetOutputDataType(MEAN_IDX, DT_FLOAT);
    context->SetOutputDataType(RSTD_IDX, DT_FLOAT);
    OP_LOGD(context, "End to do InferDataType4AddLayerNorm");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AddLayerNorm).InferShape(InferShape4AddLayerNorm).InferDataType(InferDataType4AddLayerNorm);
IMPL_OP_INFERSHAPE(InplaceAddLayerNorm).InferShape(InferShape4AddLayerNorm).InferDataType(InferDataType4AddLayerNorm);
} // namespace ops