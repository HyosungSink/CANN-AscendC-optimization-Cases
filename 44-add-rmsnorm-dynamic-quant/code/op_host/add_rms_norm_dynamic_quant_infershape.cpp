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
 * \file add_rms_norm_dynamic_quant_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"

static constexpr int X1_IDX = 0;
static constexpr int X2_IDX = 1;
static constexpr int GAMMA_IDX = 2;
static constexpr int SMOOTH1_IDX = 3;
static constexpr int SMOOTH2_IDX = 4;

static constexpr int Y1_IDX = 0;
static constexpr int Y2_IDX = 1;
static constexpr int X_IDX = 2;
static constexpr int OUT_SCALE1_IDX = 3;
static constexpr int OUT_SCALE2_IDX = 4;
static constexpr int ATTR_INDEX_OF_DST_TYPE = 2;

using namespace ge;
using namespace Ops::Base;

namespace ops {
static const std::initializer_list<ge::DataType> OUT_TYPE_LIST = {
    DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_INT4};
static bool InferReduceShape(const gert::Shape* xShape, const gert::Shape* gammaShape, gert::Shape* reduceShape)
{
    size_t gammaDimNum = gammaShape->GetDimNum();
    size_t xDimNum = xShape->GetDimNum();
    if (xDimNum < gammaDimNum) {
        return false;
    }
    int64_t xDimValue1 = 0;
    reduceShape->SetDimNum(xDimNum - gammaDimNum);
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        xDimValue1 = xShape->GetDim(i);
        reduceShape->SetDim(i, xDimValue1);
        OP_LOGI(
            "InferShape4AddRmsNormDynamicQuant InferReduceShape", "reduceShape[%zu] = [%zu]", i,
            reduceShape->GetDim(i));
    }
    return true;
}

static bool CheckOptionalShapeExisting(const gert::Shape* smoothShape)
{
    OP_CHECK_IF(nullptr == smoothShape, OP_LOGD("CheckOptionalShapeExisting", "Get nullptr smoothShape"), return false);
    return true;
}

static ge::graphStatus InferShape4AddRmsNormDynamicQuant(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Begin to do InferShape4AddRmsNormDynamicQuant");

    // get input shapes
    const gert::Shape* x1Shape = context->GetInputShape(X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* gammaShape = context->GetInputShape(GAMMA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    // get output shapes
    gert::Shape* y1Shape = context->GetOutputShape(Y1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1Shape);
    gert::Shape* y2Shape = context->GetOutputShape(Y2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2Shape);
    gert::Shape* xShape = context->GetOutputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* outScale1Shape = context->GetOutputShape(OUT_SCALE1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outScale1Shape);
    gert::Shape* outScale2Shape = context->GetOutputShape(OUT_SCALE2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outScale2Shape);

    *y1Shape = *x1Shape;
    *xShape = *x1Shape;

    const gert::Shape* smooth1Shape = context->GetOptionalInputShape(SMOOTH1_IDX);
    bool smooth1Exist = CheckOptionalShapeExisting(smooth1Shape);
    const gert::Shape* smooth2Shape = context->GetOptionalInputShape(SMOOTH2_IDX);
    bool smooth2Exist = CheckOptionalShapeExisting(smooth2Shape);
    bool isOnlyExistSmooth2 = (!smooth1Exist) && smooth2Exist;
    OP_CHECK_IF(isOnlyExistSmooth2, OP_LOGE(context, "Dynamic AddRmsNormDynamicQuant Not support only have scale2."),
                return GRAPH_FAILED);

    // unknown rank
    if (IsUnknownRank(*x1Shape) || IsUnknownRank(*gammaShape)) {
        SetUnknownRank(*outScale1Shape);
        if (smooth2Exist) {
            *outScale2Shape = *outScale1Shape;
            *y2Shape = *x1Shape;
        } else {
            *y2Shape = gert::Shape({1});
            *outScale2Shape = gert::Shape({1});
        }
        OP_LOGI(context, "End to do InferShape4AddRmsNormDynamicQuant with unknown rank.");
        return GRAPH_SUCCESS;
    }

    auto ret = InferReduceShape(x1Shape, gammaShape, outScale1Shape);
    OP_CHECK_IF(!ret, OP_LOGE(context, "Dynamic AddRmsNormDynamicQuant Not support gammaDimNum > xDimNum."),
                return GRAPH_FAILED);
    if (smooth2Exist) {
        *outScale2Shape = *outScale1Shape;
        *y2Shape = *x1Shape;
    } else {
        *y2Shape = gert::Shape({1});
        *outScale2Shape = gert::Shape({1});
    }
    OP_LOGI(context, "End to do InferShape4AddRmsNormDynamicQuant");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4AddRmsNormDynamicQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4AddRmsNormDynamicQuant");
    ge::DataType yDtype = ge::DT_INT8;
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int32_t* pDstDtype = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_TYPE);
        if (pDstDtype != nullptr) {
            int32_t dstDtype = *pDstDtype;
            yDtype = static_cast<ge::DataType>(dstDtype);
            OP_CHECK_IF(
                std::find(OUT_TYPE_LIST.begin(), OUT_TYPE_LIST.end(), yDtype) == OUT_TYPE_LIST.end(),
                OP_LOGE(
                    context,
                    "attr dst_type only support 2(int8), 3(int4), 34(hifloat8), 35(float8_e5m2), 36(float8_e4m3fn)"),
                return ge::GRAPH_FAILED);
        }
    }
    context->SetOutputDataType(Y1_IDX, yDtype);
    context->SetOutputDataType(Y2_IDX, yDtype);
    context->SetOutputDataType(X_IDX, context->GetInputDataType(X1_IDX));
    context->SetOutputDataType(OUT_SCALE1_IDX, DT_FLOAT);
    context->SetOutputDataType(OUT_SCALE2_IDX, DT_FLOAT);
    OP_LOGD(context, "End to do InferDataType4AddRmsNormDynamicQuant");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AddRmsNormDynamicQuant)
    .InferShape(InferShape4AddRmsNormDynamicQuant)
    .InferDataType(InferDataType4AddRmsNormDynamicQuant);
} // namespace ops
