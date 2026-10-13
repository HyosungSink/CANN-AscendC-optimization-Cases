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
 * \file dequant_swiglu_quant_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "graph/utils/type_utils.h"
#include "util/shape_util.h"
#include "log/log.h"
#include "util/math_util.h"

using namespace ge;
namespace ops {
constexpr size_t INPUT_IDX_X = 0;
constexpr size_t OUTPUT_IDX_Y = 0;
constexpr size_t OUTPUT_IDX_SCALE = 1;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t INDEX_ATTR_DST_TYPE = 2;
constexpr int64_t INDEX_ATTR_ACTIVATE_DIM = 4;
static const std::initializer_list<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2,
                                                                        ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                                                                        ge::DT_INT8, ge::DT_HIFLOAT8};

graphStatus InferShape4DequantSwigluQuant(gert::InferShapeContext* context) {
  OP_LOGD(context, "Begin to do InferShape4DequantSwigluQuant.");

  const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
  OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
  gert::Shape* yShape = context->GetOutputShape(OUTPUT_IDX_Y);
  OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
  gert::Shape* scaleShape = context->GetOutputShape(OUTPUT_IDX_SCALE);
  OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);

  *yShape = *xShape;
  *scaleShape = *xShape;

  OP_CHECK_IF(Ops::Base::IsUnknownRank(*xShape),
           OP_LOGD(context, "End to do InferShape4DequantSwigluQuant, inputx is [-2]."),
           return GRAPH_SUCCESS);

  auto attrsPtr = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
  const int32_t *activateDim = attrsPtr->GetAttrPointer<int32_t>(INDEX_ATTR_ACTIVATE_DIM);
  const int32_t activateDimNum = (activateDim == nullptr) ? -1 : *activateDim;

  // 将切分轴转换为正数
  int64_t xShapeRank = static_cast<int64_t>(xShape->GetDimNum());
  size_t selectDim = (activateDimNum >= 0) ? static_cast<size_t>(activateDimNum) : static_cast<size_t>(activateDimNum + xShapeRank);

  // 设置Y的shape
  yShape->SetDim(selectDim, xShape->GetDim(selectDim) / NUM_TWO);
  // 设置Scale的shape
  scaleShape->SetDimNum(xShapeRank - 1);
  scaleShape->SetDim(selectDim, xShape->GetDim(selectDim) / NUM_TWO);

  OP_LOGD(context, "End to do InferShape4DequantSwigluQuant");
  return ge::GRAPH_SUCCESS;
}

graphStatus InferDtype4DequantSwigluQuant(gert::InferDataTypeContext* context) {
  OP_LOGD(context, "InferDtype4DequantSwigluQuant enter");

  auto attrsPtr = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
  const int32_t *dstDtype = attrsPtr->GetAttrPointer<int32_t>(INDEX_ATTR_DST_TYPE);
  const int32_t dstDtypeNum = (dstDtype == nullptr) ? 2 : *dstDtype;

  ge::DataType outDtype = static_cast<ge::DataType>(dstDtypeNum);
  OP_CHECK_IF(std::find(Y_SUPPORT_DTYPE_SET.begin(), Y_SUPPORT_DTYPE_SET.end(), outDtype) == Y_SUPPORT_DTYPE_SET.end(),
           OP_LOGE(context, "dst_type is illegal, only supports 2(INT8) 40(FLOAT4_E2M1), 41(FLOAT4_E1M2), 35(FLOAT8E5M2), 36(FLOAT8E4M3), 34(HiFloat8)"),
           return ge::GRAPH_FAILED);

  context->SetOutputDataType(OUTPUT_IDX_Y, outDtype);
  context->SetOutputDataType(OUTPUT_IDX_SCALE, DT_FLOAT);
  OP_LOGD(context, "InferDtype4DequantSwigluQuant end");

  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DequantSwigluQuant)
    .InferShape(InferShape4DequantSwigluQuant)
    .InferDataType(InferDtype4DequantSwigluQuant);
}  // namespace ops
