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
 * \file nsa_selected_attention_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {

constexpr int NSA_SOFTMAXMAX_F32_DIM0SHAPE = 8;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_2 = 2;
constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr size_t KEY_INPUT_INDEX = 1;
constexpr size_t VALUE_INPUT_INDEX = 2;
constexpr size_t TOPK_INDICES_INPUT_INDEX = 3;
constexpr size_t INPUT_LAYOUT_ATTR_INDEX = 3;

ge::graphStatus InferShapeNsaSelectedAttention(gert::InferShapeContext *context)
{
    OP_LOGI(context, "Enter NsaSelectedAttention runtime infershape impl.");
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::Shape *valueShape = context->GetInputShape(VALUE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);
    const gert::Shape *topkIndicesShape = context->GetInputShape(TOPK_INDICES_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIndicesShape);
    auto attrs = context->GetAttrs();
    const auto *queryDesc = context->GetInputDesc(QUERY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char *inputLayout = attrs->GetAttrPointer<char>(INPUT_LAYOUT_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayout);
    std::string inputLayoutStr = std::string(inputLayout);
    for (auto &c : inputLayoutStr) {
        c = toupper(c);
    }
    if (inputLayoutStr != "TND") {
        OP_LOGE(context, "The inputLayout should be TND(case-insensitive), but got %s.",
                inputLayoutStr.c_str());
        return GRAPH_FAILED;
    }

    int64_t shapeT = 0;
    shapeT = queryShape->GetDim(0);
    auto headNum = attrs->GetInt(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, headNum);

    // softmaxMax, fp32: (T, N, 8)
    gert::Shape *softmaxMaxShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxMaxShape);
    softmaxMaxShape->SetDimNum(DIM_NUM_3);
    softmaxMaxShape->SetDim(0, shapeT);
    softmaxMaxShape->SetDim(1, *headNum);
    softmaxMaxShape->SetDim(DIM_NUM_2, NSA_SOFTMAXMAX_F32_DIM0SHAPE);

    // softmaxSum, shape same as softmaxMax
    gert::Shape *softmaxSumShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxSumShape);
    *softmaxSumShape = *softmaxMaxShape;

    gert::Shape *attentionOutShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    *attentionOutShape = *queryShape;
    //TND
    auto shapeD2 = valueShape->GetDim(2);
    attentionOutShape->SetDim(2, shapeD2);
    return GRAPH_SUCCESS;
}    

ge::graphStatus InferDataTypeNsaSelectedAttention(gert::InferDataTypeContext *context)
{   
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto inputDataType = context->GetInputDataType(0);
    // softmax_max, outidx:0
    context->SetOutputDataType(0, ge::DT_FLOAT);
    // softmax_sum, outidx:1
    context->SetOutputDataType(1, ge::DT_FLOAT);
    // attention_out, outidx:3
    context->SetOutputDataType(2, inputDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NsaSelectedAttention).InferShape(InferShapeNsaSelectedAttention).InferDataType(InferDataTypeNsaSelectedAttention);
} // namespace ops
