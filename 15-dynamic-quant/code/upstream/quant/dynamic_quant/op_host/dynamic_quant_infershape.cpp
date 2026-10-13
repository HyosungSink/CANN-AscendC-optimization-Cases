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
 * \file dynamic_quant_infershape.cpp.
 * \brief
 */
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "log/log.h"
#include "util/math_util.h"

using namespace ge;
namespace ops {
static const size_t ATTR_INDEX_OF_DST_TYPE = 0;
static const std::initializer_list<ge::DataType> OUT_TYPE_LIST = {
    DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN};
static constexpr uint32_t OUTPUT_NUM_DYNAMIC_QUANT = 2;

static ge::graphStatus CheckComputeNodeNumMerged(gert::InferShapeContext* context)
{
    // check input and output number
    if (context->GetComputeNodeOutputNum() != OUTPUT_NUM_DYNAMIC_QUANT) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeCheck(gert::InferShapeContext* context)
{
    if (context == nullptr || CheckComputeNodeNumMerged(context) == GRAPH_FAILED) {
        return GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < OUTPUT_NUM_DYNAMIC_QUANT; ++i) {
        if (context->GetOutputShape(i) == nullptr) {
            return GRAPH_FAILED;
        }
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus DynamicQuantInferShape(gert::InferShapeContext* context)
{
    if (InferShapeCheck(context) == GRAPH_FAILED) {
        return GRAPH_FAILED;
    }

    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    gert::Shape* scaleShape = context->GetOutputShape(1);
    *yShape = *xShape;
    scaleShape->SetDimNum(xShape->GetDimNum() - 1);
    for (uint32_t i = 0; i < xShape->GetDimNum() - 1; ++i) {
        scaleShape->SetDim(i, xShape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus DynamicQuantInferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    OP_LOGD(context, "DynamicQuantInferDataType begin");
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
                    "attr dst_type only support 2(int8), 29(int4), 34(hifloat8), 35(float8_e5m2), 36(float8_e4m3fn)"),
                return ge::GRAPH_FAILED);
        }
    }
    context->SetOutputDataType(0, yDtype);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    OP_LOGD(context, "DynamicQuantInferDataType end");
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(DynamicQuant).InferShape(DynamicQuantInferShape).InferDataType(DynamicQuantInferDataType);
} // namespace ops
