/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mla_prolog_v3_infershape.h"

using namespace ge;

namespace ops {

ge::graphStatus GetMlaPrologV3ShapeDim(const gert::InferShapeContext *context, MlaPrologProtoShapeParam &shapeParam)
{
    auto apiRet = GetMlaPrologShapeDim(context, shapeParam);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "Context get input shape failed"), return ge::GRAPH_FAILED);
    auto weightDqShape = context->GetRequiredInputShape(WEIGHT_DQ_INDEX);  // (He, Hcq)
    OP_CHECK_NULL_WITH_CONTEXT(context, weightDqShape);
    shapeParam.Hcq = weightDqShape->GetDim(DIM_INDEX_1);
    return GRAPH_SUCCESS;
}

ge::graphStatus SetMlaPrologV3ShapeDim(const MlaPrologProtoShapeParam &shapeParam, gert::InferShapeContext *context)
{
    auto apiRet = SetMlaPrologShapeDim(shapeParam, context);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "SetMlaPrologShapeDim failed"), return ge::GRAPH_FAILED);

    // set output shape
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // Get attribute pointers and dereference once
    const int *weightQuantModePtr = attrs->GetAttrPointer<int>(ATTR_WEIGHT_QUANT_MODE_FLAG_INDEX);
    const int weightQuantMode = (weightQuantModePtr == nullptr) ? 0 : *weightQuantModePtr;
    const int *kvQuantModePtr = attrs->GetAttrPointer<int>(ATTR_KV_QUANT_MODE_FLAG_INDEX);
    const int kvQuantMode = (kvQuantModePtr == nullptr) ? 0 : *kvQuantModePtr;

    // dequantScaleQNope: (B*S, N ,1) | (T, N, 1). (1) if not enabled
    auto dequantScaleQNopeShape = context->GetOutputShape(DEQUANT_SCALE_Q_NOPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dequantScaleQNopeShape);

    if ((weightQuantMode == WEIGHT_QUANT_MODE_FULL_QUANT && kvQuantMode == KV_QUANT_MODE_PER_TENSOR) ||
        (weightQuantMode == WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT && kvQuantMode == KV_QUANT_MODE_PER_TENSOR)) {
        dequantScaleQNopeShape->SetDimNum(DIM_NUM_3);                   // (B*S, N, 1) | (T, N, 1)
        dequantScaleQNopeShape->SetDim(DIM_INDEX_0, shapeParam.isBsMerge ? shapeParam.T : shapeParam.B * shapeParam.S);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_1, shapeParam.N);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_2, DIM_NUM_1);                 // 1: Fix dim 1
    } else {
        dequantScaleQNopeShape->SetDimNum(DIM_NUM_1);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_0, DIM_NUM_0);
    }

    // queryNorm
    gert::Shape *queryNormShape = context->GetOutputShape(QUERY_NORM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryNormShape);
    gert::Shape *dequantScaleQNormShape = context->GetOutputShape(DEQUANT_SCALE_Q_NORM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dequantScaleQNormShape);

    const bool *queryNormFlagPtr = attrs->GetAttrPointer<bool>(ATTR_QUERY_NORM_FLAG_INDEX);
    const bool queryNormFlag = (queryNormFlagPtr == nullptr) ? 0 : *queryNormFlagPtr;

    if (queryNormFlag) {
        if (shapeParam.isBsMerge) {
            // [T, Hcq]
            queryNormShape->SetDimNum(DIM_NUM_2);
            queryNormShape->SetDim(DIM_INDEX_0, shapeParam.T);
            queryNormShape->SetDim(DIM_INDEX_1, shapeParam.Hcq);
        } else {
            // [B, S, Hcq]
            queryNormShape->SetDimNum(DIM_NUM_3);
            queryNormShape->SetDim(DIM_INDEX_0, shapeParam.B);
            queryNormShape->SetDim(DIM_INDEX_1, shapeParam.S);
            queryNormShape->SetDim(DIM_INDEX_2, shapeParam.Hcq);
        }

        auto weightUqQrDesc = context->GetInputDesc(WEIGHT_UQ_QR_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, weightUqQrDesc);

        if (weightQuantMode == WEIGHT_QUANT_MODE_NO_QUANT) {
            dequantScaleQNormShape->SetDimNum(DIM_NUM_1);
            dequantScaleQNormShape->SetDim(DIM_INDEX_0, DIM_NUM_0);
        } else if (weightQuantMode == WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT) {
            dequantScaleQNormShape->SetDimNum(DIM_NUM_2);
            dequantScaleQNormShape->SetDim(DIM_INDEX_0, shapeParam.isBsMerge ? shapeParam.T : shapeParam.B * shapeParam.S);
            dequantScaleQNormShape->SetDim(DIM_INDEX_1, shapeParam.Hcq / FP8_E4M3_BLOCK_SIZE);
        } else {
            dequantScaleQNormShape->SetDimNum(DIM_NUM_2);
            dequantScaleQNormShape->SetDim(DIM_INDEX_0, shapeParam.isBsMerge ? shapeParam.T : shapeParam.B * shapeParam.S);
            dequantScaleQNormShape->SetDim(DIM_INDEX_1, DIM_NUM_1);
        }
    } else {
        queryNormShape->SetDimNum(DIM_NUM_1);
        queryNormShape->SetDim(DIM_INDEX_0, DIM_NUM_0);
        dequantScaleQNormShape->SetDimNum(DIM_NUM_1);
        dequantScaleQNormShape->SetDim(DIM_INDEX_0, DIM_NUM_0);
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeMlaPrologV3(gert::InferShapeContext *context) {
    OP_LOGI(context->GetNodeName(), "Enter MlaPrologV3 infershape impl.");

    MlaPrologProtoShapeParam shapeParam {};
    auto apiRet = GetMlaPrologV3ShapeDim(context, shapeParam);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "Context get input shape failed"), return ge::GRAPH_FAILED);

    apiRet = SetMlaPrologV3ShapeDim(shapeParam, context);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "Context set output shape failed"), return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeMlaPrologV3(gert::InferDataTypeContext *context)
{
    OP_LOGI(context->GetNodeName(), "Enter MlaPrologV3 infershape impl.");

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // Get attribute pointers and dereference once
    const int *weightQuantModePtr = attrs->GetAttrPointer<int>(ATTR_WEIGHT_QUANT_MODE_FLAG_INDEX);
    const int weightQuantMode = (weightQuantModePtr == nullptr) ? 0 : *weightQuantModePtr;
    const int *kvQuantModePtr = attrs->GetAttrPointer<int>(ATTR_KV_QUANT_MODE_FLAG_INDEX);
    const int kvQuantMode = (kvQuantModePtr == nullptr) ? 0 : *kvQuantModePtr;

    // mxfp8 quant
    if (weightQuantMode == WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT) {
        bool isMxfp8FullQuant = (context->GetRequiredInputDataType(TOKEN_X_INDEX) == ge::DT_FLOAT8_E4M3FN &&
            context->GetOptionalInputDataType(QUANT_SCALE_CKV_INDEX) != ge::DT_UNDEFINED);

        context->SetOutputDataType(QUERY_INDEX, (isMxfp8FullQuant) ? context->GetRequiredInputDataType(WEIGHT_DKV_KR_INDEX) : context->GetRequiredInputDataType(WEIGHT_UK_INDEX));
        context->SetOutputDataType(QUERY_ROPE_INDEX, context->GetRequiredInputDataType(WEIGHT_UK_INDEX));
        context->SetOutputDataType(KV_CACHE_OUT_INDEX, context->GetRequiredInputDataType(KV_CACHE_INDEX_V3));
        context->SetOutputDataType(KR_CACHE_OUT_INDEX, context->GetRequiredInputDataType(KR_CACHE_INDEX_V3));
        context->SetOutputDataType(DEQUANT_SCALE_Q_NOPE_INDEX, ge::DT_FLOAT);
        context->SetOutputDataType(QUERY_NORM_INDEX, context->GetRequiredInputDataType(WEIGHT_UQ_QR_INDEX));
        context->SetOutputDataType(DEQUANT_SCALE_Q_NORM_INDEX, ge::DT_FLOAT8_E8M0);
    } else {
        context->SetOutputDataType(QUERY_INDEX, context->GetRequiredInputDataType(WEIGHT_UK_INDEX));
        context->SetOutputDataType(QUERY_ROPE_INDEX, context->GetRequiredInputDataType(WEIGHT_UK_INDEX));
        context->SetOutputDataType(KV_CACHE_OUT_INDEX, context->GetRequiredInputDataType(KV_CACHE_INDEX_V3));
        context->SetOutputDataType(KR_CACHE_OUT_INDEX, context->GetRequiredInputDataType(KR_CACHE_INDEX_V3));

        // full quant
        bool isQuantQuery = (weightQuantMode == WEIGHT_QUANT_MODE_FULL_QUANT && kvQuantMode == KV_QUANT_MODE_PER_TENSOR);

        context->SetOutputDataType(QUERY_INDEX, isQuantQuery ? ge::DT_INT8 : ge::DT_BF16);
        context->SetOutputDataType(DEQUANT_SCALE_Q_NOPE_INDEX, ge::DT_FLOAT);

        if (weightQuantMode == WEIGHT_QUANT_MODE_NO_QUANT) {
            context->SetOutputDataType(QUERY_NORM_INDEX, ge::DT_BF16);
        } else {
            context->SetOutputDataType(QUERY_NORM_INDEX, ge::DT_INT8);
        }
        context->SetOutputDataType(DEQUANT_SCALE_Q_NORM_INDEX, ge::DT_FLOAT);
    }

  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MlaPrologV3).InferShape(InferShapeMlaPrologV3).InferDataType(InferDataTypeMlaPrologV3);
}  // namespace ops