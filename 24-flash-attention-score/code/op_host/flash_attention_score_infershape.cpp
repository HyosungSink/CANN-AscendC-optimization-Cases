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
 * \file flash_attention_score_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {

static const uint64_t DIM_NUM_2 = 2;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_4 = 4;
static const uint64_t INDEX_2 = 2;
static const uint64_t INDEX_3 = 3;
static const uint64_t INDEX_HEAD_NUM = 4;
static const uint64_t INDEX_LAYOUT = 5;
static const uint64_t INDEX_OUTDTYPE = 11;
constexpr int FLA_SOFTMAXMAX_F32_DIM0SHAPE = 8;
constexpr int FLA_FP8_SOFTMAXMAX_F32_DIM0SHAPE = 1;

ge::graphStatus InferShapeFlashAttentionScore(gert::InferShapeContext *context)
{
    OP_LOGI(context, "Start enter FlashAttentionScore runtime infershape impl.");
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const auto *queryDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);

    const gert::Shape *keyShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::Shape *valueShape = context->GetInputShape(INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto headNum = attrs->GetInt(INDEX_HEAD_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, headNum);
    const char *inputLayout = attrs->GetAttrPointer<char>(INDEX_LAYOUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayout);

    std::string inputLayoutStr = std::string(inputLayout);
    for (auto &c : inputLayoutStr) {
        c = toupper(c);
    }
    if (inputLayoutStr != "BSH" && inputLayoutStr != "BSND" && inputLayoutStr != "SBH" &&  inputLayoutStr != "BNSD" &&
        inputLayoutStr != "TND") {
        OP_LOGE(context, "The inputLayout should be BSH/SBH/BSND/BNSD/TND(case-insensitive), but got %s.",
                  inputLayoutStr.c_str());
        return GRAPH_FAILED;
    }

    int64_t shapeB = 1;
    int64_t shapeS = 1;
    int64_t shapeT = 0;
     if (inputLayoutStr == "SBH") {
        shapeB = queryShape->GetDim(1);
        shapeS = queryShape->GetDim(0);
    } else if (inputLayoutStr == "TND") {
        shapeT = queryShape->GetDim(0);
    } else if (inputLayoutStr == "BSND" || inputLayoutStr == "BSH" ) {
        shapeB = queryShape->GetDim(0);
        shapeS = queryShape->GetDim(1);
    } else {
        // BNSD
        shapeB = queryShape->GetDim(0);
        shapeS = queryShape->GetDim(DIM_NUM_2);
    }
    OP_LOGI(context, "InferShape info: B=%ld, N=%ld, T=%ld, S=%ld, inputLayout=%s, dtype=%s", shapeB, *headNum, shapeT,
              shapeS, inputLayoutStr.c_str(), ge::TypeUtils::DataTypeToSerialString(queryDesc->GetDataType()).c_str());

    // softmaxMax, fp32: (B, N, S, 8)
    gert::Shape *softmaxMaxShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxMaxShape);
    // softmaxSum, shape same as softmaxMax
    gert::Shape *softmaxSumShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxSumShape);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto dtype = inputDesc->GetDataType();
    if (dtype == DT_HIFLOAT8) {
        softmaxMaxShape->SetDimNum(DIM_NUM_4);
        softmaxMaxShape->SetDim(0, shapeB);
        softmaxMaxShape->SetDim(1, *headNum);
        softmaxMaxShape->SetDim(DIM_NUM_2, shapeS);
        softmaxMaxShape->SetDim(DIM_NUM_3, FLA_FP8_SOFTMAXMAX_F32_DIM0SHAPE);
    } else {
        if (inputLayoutStr == "TND") {
            softmaxMaxShape->SetDimNum(DIM_NUM_3);
            softmaxMaxShape->SetDim(0, shapeT);
            softmaxMaxShape->SetDim(1, *headNum);
            softmaxMaxShape->SetDim(DIM_NUM_2, FLA_SOFTMAXMAX_F32_DIM0SHAPE);
        } else {
            // 0, 1, 2, 3, 4 : dim idx
            softmaxMaxShape->SetDimNum(DIM_NUM_4);
            softmaxMaxShape->SetDim(0, shapeB);
            softmaxMaxShape->SetDim(1, *headNum);
            softmaxMaxShape->SetDim(DIM_NUM_2, shapeS);
            softmaxMaxShape->SetDim(DIM_NUM_3, FLA_SOFTMAXMAX_F32_DIM0SHAPE);
        }
    }
    *softmaxSumShape = *softmaxMaxShape;

    // softmaxOut, shape: (B, N, S, S)
    gert::Shape *softmaxOutShape = context->GetOutputShape(INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxOutShape);
    // 0, 1, 2, 3, 4 : dim idx
    softmaxOutShape->SetDimNum(DIM_NUM_4);
    softmaxOutShape->SetDim(DIM_NUM_3, 0);
    softmaxOutShape->SetDim(DIM_NUM_2, 0);
    softmaxOutShape->SetDim(1, 0);
    softmaxOutShape->SetDim(0, 0);

    gert::Shape *attentionOutShape = context->GetOutputShape(INDEX_3);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    *attentionOutShape = *queryShape;
    if (inputLayoutStr == "BSND" || inputLayoutStr == "BNSD") {
        auto shapeD2 = valueShape->GetDim(3);
        attentionOutShape->SetDim(3, shapeD2);
    } else if (inputLayoutStr == "BSH" || inputLayoutStr == "SBH" ) {
        auto N1 = *headNum;
        if (N1 == 0) {
            attentionOutShape->SetDim(DIM_NUM_2, 0);
            return GRAPH_SUCCESS;
        }
        auto h1 =  queryShape->GetDim(DIM_NUM_2);
        auto D1 = h1 / N1;
        if (D1 == 0) {
            attentionOutShape->SetDim(DIM_NUM_2, 0);
            return GRAPH_SUCCESS;
        }
        auto h2 =  keyShape->GetDim(DIM_NUM_2);
        auto N2 = h2 / D1;
        if (N2 == 0) {
            attentionOutShape->SetDim(DIM_NUM_2, N1 * D1);
            return GRAPH_SUCCESS;
        }
        auto h3 =  valueShape->GetDim(DIM_NUM_2);
        auto D2 = h3 / N2;
        attentionOutShape->SetDim(DIM_NUM_2, N1 * D2);
    } else{
        //TND
        auto shapeD2 = valueShape->GetDim(DIM_NUM_2);
        attentionOutShape->SetDim(DIM_NUM_2, shapeD2);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeFlashAttentionScore(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dtype = context->GetInputDataType(0);
    // softmax_max, outidx:0
    context->SetOutputDataType(0, DT_FLOAT);
    // softmax_sum, outidx:1
    context->SetOutputDataType(1, DT_FLOAT);

    if (dtype == DT_FLOAT8_E5M2 || dtype == DT_FLOAT8_E4M3FN || dtype == DT_HIFLOAT8) {
        auto attrs = context->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

        auto outDtypePtr = attrs->GetAttrPointer<int64_t>(INDEX_OUTDTYPE);
        int64_t outDtype = *outDtypePtr;
        // softmax_out, outidx:2
        context->SetOutputDataType(INDEX_2, ge::DT_BF16);
        // attention_out, outidx:3
        context->SetOutputDataType(INDEX_3, ge::DT_BF16);
        return GRAPH_SUCCESS;
    }

    // softmax_out, outidx:2
    context->SetOutputDataType(INDEX_2, dtype);
    // attention_out, outidx:3
    context->SetOutputDataType(INDEX_3, dtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FlashAttentionScore).InferShape(InferShapeFlashAttentionScore).InferDataType(InferDataTypeFlashAttentionScore);

} // namespace ops
