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
 * \file add_rms_norm_quant_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

static constexpr int INPUT_X1_IDX = 0;
static constexpr int INPUT_SCALE2_IDX = 4;
static constexpr int OUTPUT_Y1_IDX = 0;
static constexpr int OUTPUT_Y2_IDX = 1;
static constexpr int OUTPUT_X_IDX = 2;
static constexpr int ATTR_INDEX_OF_DST_TYPE = 3;

using namespace ge;

namespace ops {
static const std::initializer_list<ge::DataType> OUT_TYPE_LIST = {
    DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN};
static ge::graphStatus InferShape4AddRmsNormQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4AddRmsNormQuant");

    // get input shapes
    const gert::Shape* x1Shape = context->GetInputShape(INPUT_X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);

    // get output shapes
    gert::Shape* y1Shape = context->GetOutputShape(OUTPUT_Y1_IDX);
    gert::Shape* y2Shape = context->GetOutputShape(OUTPUT_Y2_IDX);
    gert::Shape* xShape = context->GetOutputShape(OUTPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    *y1Shape = *x1Shape;
    *xShape = *x1Shape;

    const gert::Shape* scale2Shape = context->GetOptionalInputShape(INPUT_SCALE2_IDX);
    if (nullptr != scale2Shape && (scale2Shape->GetDimNum() != 0)) {
        *y2Shape = *x1Shape;
    } else {
        *y2Shape = gert::Shape({1});
    }

    OP_LOGD(context, "End to do InferShape4AddRmsNormQuant");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4AddRmsNormQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4AddRmsNormQuant");
    ge::DataType yDtype = ge::DT_INT8;
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int32_t* pDstDtype = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_TYPE);
        if (pDstDtype != nullptr) {
            int32_t dstDtype = *pDstDtype;
            yDtype = static_cast<ge::DataType>(dstDtype);
            OP_CHECK_IF(
                std::find(OUT_TYPE_LIST.begin(), OUT_TYPE_LIST.end(), yDtype) == OUT_TYPE_LIST.end(),
                OP_LOGE(context,
                    "attr dst_type only support 2(int8), 34(hifloat8), 35(float8_e5m2), 36(float8_e4m3fn)"),
                return ge::GRAPH_FAILED);
        }
    }
    context->SetOutputDataType(OUTPUT_Y1_IDX, yDtype);
    context->SetOutputDataType(OUTPUT_Y2_IDX, yDtype);
    context->SetOutputDataType(OUTPUT_X_IDX, context->GetInputDataType(INPUT_X1_IDX));
    OP_LOGD(context, "End to do InferDataType4AddRmsNormQuant");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AddRmsNormQuant).InferShape(InferShape4AddRmsNormQuant).InferDataType(InferDataType4AddRmsNormQuant);
} // namespace ops
