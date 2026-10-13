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
 * \file fused_floyd_attention_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {

constexpr int FLA_SOFTMAXMAX_F32_DIM0SHAPE = 8;
static const uint64_t DIM_NUM_5 = 5;

ge::graphStatus InferShapeFusedFloydAttention(gert::InferShapeContext *context)
{
    OP_LOGI(context, "Enter FusedFloydAttention runtime infershape impl.");

    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    auto attrs = context->GetAttrs();
    const auto *queryDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    int64_t shapeB = 1;
    int64_t shapeS = 1;
    int64_t shapeH = 1;

    // BHNSD
    shapeB = queryShape->GetDim(0);
    shapeH = queryShape->GetDim(1);
    auto headNum = queryShape->GetDim(2); // 2: BHNSD中的N
    shapeS = queryShape->GetDim(3); // 3: BHNSD中的S

    // softmaxMax, fp32: (B, H, N, S, 8)
    gert::Shape *softmaxMaxShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxMaxShape);

    // 0, 1, 2, 3, 4 : dim idx
    softmaxMaxShape->SetDimNum(DIM_NUM_5);
    softmaxMaxShape->SetDim(0, shapeB);
    softmaxMaxShape->SetDim(1, shapeH);
    softmaxMaxShape->SetDim(2, headNum);
    softmaxMaxShape->SetDim(3, shapeS);
    softmaxMaxShape->SetDim(4, FLA_SOFTMAXMAX_F32_DIM0SHAPE);

    // softmaxSum, shape same as softmaxMax
    gert::Shape *softmaxSumShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxSumShape);
    *softmaxSumShape = *softmaxMaxShape;


    gert::Shape *attentionOutShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    *attentionOutShape = *queryShape;

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeFusedFloydAttention(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dtype = context->GetInputDataType(0);
    // softmax_max, outidx:0
    context->SetOutputDataType(0, DT_FLOAT);
    // softmax_sum, outidx:1
    context->SetOutputDataType(1, DT_FLOAT);
    // attention_out, outidx:2
    context->SetOutputDataType(2, dtype);
    return GRAPH_SUCCESS;
}

IMPL_OP(FusedFloydAttention).InferShape(InferShapeFusedFloydAttention).InferDataType(InferDataTypeFusedFloydAttention);

} // namespace ops
