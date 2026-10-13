/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_init_routing_quant_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"


using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t X_IDX = 0;
static constexpr int64_t ROW_IDX = 1;
static constexpr int64_t EXPERT_IDX = 2;
static constexpr int64_t EXPENDED_X_IDX = 0;
static constexpr int64_t EXPENDED_ROW_IDX = 1;
static constexpr int64_t EXPENDED_EXPERT_IDX = 2;

static bool isSameDim(int64_t dim1, int64_t dim2)
{
    if (dim1 == NEG_ONE || dim2 == NEG_ONE) {
        return true;
    }
    return dim1 == dim2;
}

static ge::graphStatus CheckInputShape(
    const gert::InferShapeContext* context, const gert::Shape* xShape, const gert::Shape* rowIdxShape,
    const gert::Shape* expertIdxShape)
{
    int64_t x_n = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(0);
    int64_t cols = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(DIM_ONE);
    if (x_n < NEG_ONE || cols < NEG_ONE) {
        OP_LOGE(context->GetNodeName(), "Invalid x shape, shape is %s.", Ops::Base::ToString(*xShape).c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t row_idx_n = rowIdxShape->GetDimNum() == 1U ? NEG_ONE : rowIdxShape->GetDim(0);
    int64_t row_idx_k = rowIdxShape->GetDimNum() == 1U ? NEG_ONE : rowIdxShape->GetDim(DIM_ONE);
    if (row_idx_n < NEG_ONE || row_idx_k < NEG_ONE) {
        OP_LOGE(context->GetNodeName(), "Invalid row_idx shape, shape is %s.", Ops::Base::ToString(*rowIdxShape).c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t expert_idx_n = expertIdxShape->GetDimNum() == 1U ? NEG_ONE : expertIdxShape->GetDim(0);
    int64_t expert_idx_k = expertIdxShape->GetDimNum() == 1U ? NEG_ONE : expertIdxShape->GetDim(DIM_ONE);
    if (expert_idx_n < NEG_ONE || expert_idx_k < NEG_ONE) {
        OP_LOGE(
            context->GetNodeName(), "Invalid expert_idx shape, shape is %s.",
            Ops::Base::ToString(*expertIdxShape).c_str());
        return ge::GRAPH_FAILED;
    }

    if (!isSameDim(x_n, row_idx_n) || !isSameDim(x_n, expert_idx_n) || !isSameDim(row_idx_n, expert_idx_n)) {
        OP_LOGE(context->GetNodeName(), "Invalid input shape, the first dim of x, row_idx and expert_idx should be same.");
        return ge::GRAPH_FAILED;
    }

    if (!isSameDim(row_idx_k, expert_idx_k)) {
        OP_LOGE(context->GetNodeName(), "Invalid input shape, the second dim of row_idx and expert_idx should be same.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputDimsAndAttr(
    const gert::InferShapeContext* context, const gert::Shape* xShape, const gert::Shape* rowIdxShape,
    const gert::Shape* expertIdxShape, const int64_t activeNum)
{
    if (xShape->GetDimNum() == 1U) {
        if (xShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OP_LOGE(
                context->GetNodeName(), "The dynamic dim of x should be -2, current is %s.",
                Ops::Base::ToString(*xShape).c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (xShape->GetDimNum() != DIM_TWO) {
        OP_LOGE(
            context->GetNodeName(), "The dim of x should be 2 or dynamic, current is %s.",
            Ops::Base::ToString(*xShape).c_str());
        return ge::GRAPH_FAILED;
    }

    if (rowIdxShape->GetDimNum() == 1U) {
        if (rowIdxShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OP_LOGE(
                context->GetNodeName(), "The dynamic dim of row_idx should be -2, current is %s.",
                Ops::Base::ToString(*rowIdxShape).c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (rowIdxShape->GetDimNum() != DIM_TWO) {
        OP_LOGE(
            context->GetNodeName(), "The dim of row_idx should be 2 or dynamic, current is %s.",
            Ops::Base::ToString(*rowIdxShape).c_str());
        return ge::GRAPH_FAILED;
    }

    if (expertIdxShape->GetDimNum() == 1U) {
        if (expertIdxShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OP_LOGE(
                context->GetNodeName(), "The dynamic dim of expert_idx should be -2, current is %s.",
                Ops::Base::ToString(*expertIdxShape).c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (expertIdxShape->GetDimNum() != DIM_TWO) {
        OP_LOGE(
            context->GetNodeName(), "The dim of expert_idx should be 2 or dynamic, current is %s.",
            Ops::Base::ToString(*expertIdxShape).c_str());
        return ge::GRAPH_FAILED;
    }

    if (activeNum < 0) {
        OP_LOGE(context->GetNodeName(), "active_num must be a non-negative number.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static void ShowInputShapeInfo(
    const gert::InferShapeContext* context, const gert::Shape* xShape, const gert::Shape* rowIdxShape,
    const gert::Shape* expertIdxShape, const int64_t activeNum)
{
    OP_LOGD(context->GetNodeName(), "x's shape is: %s.", Ops::Base::ToString(*xShape).c_str());
    OP_LOGD(context->GetNodeName(), "row_idx's shape is: %s.", Ops::Base::ToString(*rowIdxShape).c_str());
    OP_LOGD(context->GetNodeName(), "expert_idx's shape is: %s.", Ops::Base::ToString(*expertIdxShape).c_str());
    OP_LOGD(context->GetNodeName(), "activeNum is: %ld.", activeNum);
}

static void ShowOutputShapeInfo(
    const gert::InferShapeContext* context, const gert::Shape* expandedXShape, const gert::Shape* expandedRowIdx,
    const gert::Shape* expandedExpertIdxShape)
{
    OP_LOGD(
        context->GetNodeName(), "expanded_x's shape is: %s after infershape.", Ops::Base::ToString(*expandedXShape).c_str());
    OP_LOGD(
        context->GetNodeName(), "expanded_row_idx's shape is: %s after infershape.",
        Ops::Base::ToString(*expandedRowIdx).c_str());
    OP_LOGD(
        context->GetNodeName(), "expanded_expert_idx's shape is: %s after infershape.",
        Ops::Base::ToString(*expandedExpertIdxShape).c_str());
}

static ge::graphStatus InferShape4MoeInitRoutingQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeInitRoutingQuantInfershape.");
    // 获取输入shape
    const gert::Shape* xShape = context->GetInputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* rowIdxShape = context->GetInputShape(ROW_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, rowIdxShape);
    const gert::Shape* expertIdxShape = context->GetInputShape(EXPERT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertIdxShape);
    gert::Shape* expandedXShape = context->GetOutputShape(EXPENDED_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedXShape);
    gert::Shape* expandedRowIdx = context->GetOutputShape(EXPENDED_ROW_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedRowIdx);
    gert::Shape* expandedExpertIdxShape = context->GetOutputShape(EXPENDED_EXPERT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedExpertIdxShape);
    // 获取attr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* activeNumPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, activeNumPtr);
    const int64_t activeNum = *activeNumPtr;
    ShowInputShapeInfo(context, xShape, rowIdxShape, expertIdxShape, activeNum);

    // 参数校验
    if (CheckInputDimsAndAttr(context, xShape, rowIdxShape, expertIdxShape, activeNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckInputShape(context, xShape, rowIdxShape, expertIdxShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t x_n = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(0);
    int64_t cols = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(DIM_ONE);

    int64_t row_idx_n = rowIdxShape->GetDimNum() == 1U ? NEG_ONE : rowIdxShape->GetDim(0);
    int64_t row_idx_k = rowIdxShape->GetDimNum() == 1U ? NEG_ONE : rowIdxShape->GetDim(DIM_ONE);

    int64_t expert_idx_n = expertIdxShape->GetDimNum() == 1U ? NEG_ONE : expertIdxShape->GetDim(0);
    int64_t expert_idx_k = expertIdxShape->GetDimNum() == 1U ? NEG_ONE : expertIdxShape->GetDim(DIM_ONE);

    int64_t n = x_n > row_idx_n ? (x_n > expert_idx_n ? x_n : expert_idx_n) :
                                  (row_idx_n > expert_idx_n ? row_idx_n : expert_idx_n);
    int64_t k = std::max(row_idx_k, expert_idx_k);
    int64_t outActiveNum = (n == NEG_ONE || k == NEG_ONE) ? NEG_ONE : (std::min(n, activeNum) * k);
    int64_t expertForSourceRowNum = (n == NEG_ONE || k == NEG_ONE) ? NEG_ONE : (n * k);

    expandedXShape->SetDimNum(DIM_TWO);
    expandedXShape->SetDim(0U, outActiveNum);
    expandedXShape->SetDim(DIM_ONE, cols);

    expandedRowIdx->SetDimNum(DIM_ONE);
    expandedRowIdx->SetDim(0U, expertForSourceRowNum);

    expandedExpertIdxShape->SetDimNum(DIM_ONE);
    expandedExpertIdxShape->SetDim(0U, expertForSourceRowNum);

    ShowOutputShapeInfo(context, expandedXShape, expandedRowIdx, expandedExpertIdxShape);
    OP_LOGD(context->GetNodeName(), "End to do MoeInitRoutingQuantInfershape.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeInitRoutingQant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeInitRoutingQuntInferDataType.");
    context->SetOutputDataType(EXPENDED_X_IDX, ge::DT_INT8);
    context->SetOutputDataType(EXPENDED_ROW_IDX, ge::DT_INT32);
    context->SetOutputDataType(EXPENDED_EXPERT_IDX, ge::DT_INT32);
    OP_LOGD(context->GetNodeName(), "End to do MoeInitRoutingQuntInferDataType.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeInitRoutingQuant)
    .InferShape(InferShape4MoeInitRoutingQuant)
    .InferDataType(InferDataType4MoeInitRoutingQant);
} // namespace ops
