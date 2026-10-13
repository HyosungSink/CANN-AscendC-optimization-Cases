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
 * \file scaled_masked_softmax_v2_infershape.cpp
 * \brief
 */
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "../common/inc/error_util.h"

using namespace gert;
namespace {
static constexpr size_t IDX_IN_X = 0;
static constexpr size_t IDX_IN_MASK = 1;
static constexpr size_t IDX_OUT_Y = 0;

static ge::graphStatus ScaledMaskedSoftmaxV2InferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do ScaledMaskedSoftmaxV2InferShape");

    // get input shapes
    const gert::Shape* xShape = context->GetInputShape(IDX_IN_X);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* maskShape = context->GetInputShape(IDX_IN_MASK);
    OPS_CHECK_NULL_WITH_CONTEXT(context, maskShape);
    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(IDX_OUT_Y);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);

    *yShape = *xShape;

    OP_LOGD(context->GetNodeName(), "End to do ScaledMaskedSoftmaxV2InferShape");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ScaledMaskedSoftmaxV2InferDtype(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "ScaledMaskedSoftmaxV2InferDtype enter");
    // Get input tout
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto inputDtype = context->GetInputDataType(IDX_IN_X);
    context->SetOutputDataType(IDX_OUT_Y, inputDtype);

    OP_LOGD(context->GetNodeName(), "ScaledMaskedSoftmaxV2InferDtype end");

    return ge::GRAPH_SUCCESS;
}

} // namespace

namespace ops {
IMPL_OP_INFERSHAPE(ScaledMaskedSoftmaxV2)
    .InferShape(ScaledMaskedSoftmaxV2InferShape)
    .InferDataType(ScaledMaskedSoftmaxV2InferDtype);
}
