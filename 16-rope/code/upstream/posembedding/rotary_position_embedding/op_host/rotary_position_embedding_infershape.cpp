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
 * \file rotary_position_embedding.cc
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;
static constexpr size_t INPUT_X_INDEX = 0;
static constexpr size_t INPUT_COS_INDEX = 1;
static constexpr size_t INPUT_SIN_INDEX = 2;
static constexpr size_t OUTPUT_Y_INDEX = 0;

namespace ops {
static ge::graphStatus InferShapeForRotaryPositionEmbedding(gert::InferShapeContext *context)
{
    OP_LOGD(context, "Begin to do InferShapeForRotaryPositionEmbedding.");
    const gert::Shape *xShape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *cosShape = context->GetInputShape(INPUT_COS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cosShape);
    const gert::Shape *sinShape = context->GetInputShape(INPUT_SIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sinShape);
    gert::Shape *yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    *yShape = *xShape;

    OP_LOGD(context, "End to do InferShapeForRotaryPositionEmbedding.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForRotaryPositionEmbedding(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(OUTPUT_Y_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RotaryPositionEmbedding)
    .InferShape(InferShapeForRotaryPositionEmbedding)
    .InferDataType(InferDataTypeForRotaryPositionEmbedding);
} // namespace ops
