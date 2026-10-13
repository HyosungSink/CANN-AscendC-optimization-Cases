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
 * \file weight_quant_batch_matmul_v2_infershape.cpp
 * \brief
 */
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "error_util.h"

namespace {
constexpr size_t MINIMUM_SHAPE_SIZE = 2UL;
constexpr size_t MAXNUM_SHAPE_SIZE = 6UL;
constexpr int64_t INT4_NUMS_IN_INT32 = 8UL;
constexpr int64_t UNKNOWN_DIM_NUM = -2;
const char* NODE_NAME = "WeightQuantBatchMatmulV2";

static bool BroadcastBatchDim(const int64_t dimA, const int64_t dimB, int64_t& dim)
{
    if (dimA > 1 && dimB > 1) {
        if (dimA != dimB) {
            return false;
        }
        dim = dimA;
        return true;
    }
    dim = std::max(dimA, dimB);
    return true;
}

bool CheckIsUnknownDimNum(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == UNKNOWN_DIM_NUM;
}

static ge::graphStatus SetBroadcastShape(const gert::Shape* xShape, const gert::Shape* weightShape, gert::Shape* yShape)
{
    size_t numDimA = xShape->GetDimNum();
    size_t numDimB = weightShape->GetDimNum();
    size_t outDimNum = std::max(numDimA, numDimB);
    int64_t broadcastDim = 0;
    if (numDimA == numDimB) {
        for (size_t i = 0; i < outDimNum - MINIMUM_SHAPE_SIZE; i++) {
            OP_CHECK_IF(
                !BroadcastBatchDim(xShape->GetDim(i), weightShape->GetDim(i), broadcastDim),
                OP_LOGE(
                    NODE_NAME, "batch of xShape and batch of wShape can't broadcast, x shape is %s, weight shape is %s",
                    Ops::Base::ToString(*xShape).c_str(), Ops::Base::ToString(*weightShape).c_str()),
                return ge::GRAPH_FAILED);
            yShape->SetDim(i, broadcastDim);
        }
    } else {
        auto longerShape = numDimA > numDimB ? xShape : weightShape;
        auto smallShape = numDimA < numDimB ? xShape : weightShape;
        uint64_t batchOffset = longerShape->GetDimNum() - smallShape->GetDimNum();
        for (size_t i = 0; i < batchOffset; i++) {
            yShape->SetDim(i, longerShape->GetDim(i));
        }
        for (size_t i = batchOffset; i < outDimNum - MINIMUM_SHAPE_SIZE; i++) {
            OP_CHECK_IF(
                !BroadcastBatchDim(longerShape->GetDim(i), smallShape->GetDim(i - batchOffset), broadcastDim),
                OP_LOGE(
                    NODE_NAME, "batch of xShape and batch of wShape can't broadcast, x shape is %s, weight shape is %s",
                    Ops::Base::ToString(*xShape).c_str(), Ops::Base::ToString(*weightShape).c_str()),
                return ge::GRAPH_FAILED);
            yShape->SetDim(i, broadcastDim);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ShapeCheckAndInfer(
    const gert::Shape* xShape, const gert::Shape* weightShape, const ge::DataType weightDtype, gert::Shape* yShape,
    const bool transposeX, const bool transposeWeight)
{
    size_t numDimA = xShape->GetDimNum();
    size_t numDimB = weightShape->GetDimNum();
    OP_CHECK_IF(
        std::min(numDimA, numDimB) < MINIMUM_SHAPE_SIZE || std::min(numDimA, numDimB) > MAXNUM_SHAPE_SIZE,
        OP_LOGE(
            NODE_NAME, "The shape of x and weight must >= 2 and <=6, which are %s and %s",
            Ops::Base::ToString(*xShape).c_str(), Ops::Base::ToString(*weightShape).c_str()),
        return ge::GRAPH_FAILED);
    size_t mIdx = transposeX ? 1UL : 2UL;
    size_t kaIdx = transposeX ? 2UL : 1UL;
    size_t kbIdx = transposeWeight ? 1UL : 2UL;
    size_t nIdx = transposeWeight ? 2UL : 1UL;
    int64_t xK = xShape->GetDim(numDimA - kaIdx);
    int64_t weightK = weightShape->GetDim(numDimB - kbIdx);
    int64_t weightN = weightShape->GetDim(numDimB - nIdx);
    // weight int32输入图融合前infershape weight输入情况： weight前存在transpose节点：x （m, k), weight (k/8, n);
    //                                                   weight前不存在transpose节点：x （m, k), weight (k, n/8)。
    // weight int32输入图融合后infershape weight输入最后一维大小为实际值除以8。
    // 采用float32承载float4_e2m1数据，shape推到逻辑同int32
    if ((weightDtype == ge::DT_INT32 || weightDtype == ge::DT_FLOAT) && xK > 0 && weightK > 0) {
        bool transWeightInt32 = false;
        if (!transposeX && !transposeWeight) {
            transWeightInt32 = (weightK * INT4_NUMS_IN_INT32 == xK);
        }
        if (transposeWeight || transWeightInt32) {
            weightK *= INT4_NUMS_IN_INT32;
        } else {
            weightN *= INT4_NUMS_IN_INT32;
        }
    }
    OP_CHECK_IF(
        xK != weightK && xK > 0 && weightK > 0, OP_LOGE(NODE_NAME, "Ka[%ld] != Kb[%ld]", xK, weightK),
        return ge::GRAPH_FAILED);
    size_t outDimNum = std::max(numDimA, numDimB);
    yShape->SetDimNum(outDimNum);
    yShape->SetDim(outDimNum - 2, xShape->GetDim(numDimA - mIdx)); // 2:设置yShape m
    yShape->SetDim(outDimNum - 1, weightN);                        // 1:设置yShape n
    return SetBroadcastShape(xShape, weightShape, yShape);
}

static ge::graphStatus InferShapeForWeightQuantBatchMatmulV2(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "InferShapeForWeightQuantBatchMatmulV2 begin");
    auto xShape = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto weightShape = context->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    auto weightDesc = context->GetInputDesc(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, weightDesc);
    auto weightDtype = weightDesc->GetDataType();
    auto yShape = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* transposeX = attrs->GetAttrPointer<bool>(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, transposeX);
    const bool* transposeWeight = attrs->GetAttrPointer<bool>(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, transposeWeight);
    if (CheckIsUnknownDimNum(*xShape) || CheckIsUnknownDimNum(*weightShape)) {
        yShape->SetDimNum(1);
        yShape->SetDim(0, UNKNOWN_DIM_NUM);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGD(
        context->GetNodeName(),
        "x_shape: %s, weight_shape: %s, transpose_x: %d, transpose_weight: %d, weight_dtype: %s",
        Ops::Base::ToString(*xShape).c_str(), Ops::Base::ToString(*weightShape).c_str(), *transposeX, *transposeWeight,
        ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString());

    OP_CHECK_IF(
        ShapeCheckAndInfer(xShape, weightShape, weightDtype, yShape, *transposeX, *transposeWeight) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "The Shape Check failed "), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

} // namespace

namespace Ops::NN::MatMul {
IMPL_OP_INFERSHAPE(WeightQuantBatchMatmulV2).InferShape(InferShapeForWeightQuantBatchMatmulV2);
}
