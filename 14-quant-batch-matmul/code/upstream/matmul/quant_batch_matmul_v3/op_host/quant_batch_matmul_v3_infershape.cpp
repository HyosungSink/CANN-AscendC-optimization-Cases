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
 * \file quant_batch_matmul_v3_infershape.cpp
 * \brief
 */
#include "common/op_host/matmul_common_infershape.h"
#include "runtime/infer_datatype_context.h"

namespace {
constexpr size_t QUANT_BATCH_MATMUL_V3_MAX_SHAPE_SIZE = 6;
constexpr size_t QUANT_BATCH_MATMUL_V3_MIN_SHAPE_SIZE = 2;
static ge::graphStatus InferShapeForQuantBatchMatmulV3(gert::InferShapeContext* context)
{
    auto shape_x1 = context->GetInputShape(0);
    auto shape_x2 = context->GetInputShape(1);
    auto scale = context->GetInputShape(2);
    auto dim_a = shape_x1->GetDimNum();
    auto dim_b = shape_x2->GetDimNum();
    if (scale == nullptr) {
        OP_LOGE(context->GetNodeName(), "[InferShape] scale is null");
        return ge::GRAPH_FAILED;
    }
    bool any_unknow_rank = Ops::NN::CheckIsUnknownDimNum(*shape_x1) || Ops::NN::CheckIsUnknownDimNum(*shape_x2);
    if (!any_unknow_rank &&
        (dim_a < QUANT_BATCH_MATMUL_V3_MIN_SHAPE_SIZE || dim_a > QUANT_BATCH_MATMUL_V3_MAX_SHAPE_SIZE ||
         dim_b < QUANT_BATCH_MATMUL_V3_MIN_SHAPE_SIZE || dim_b > QUANT_BATCH_MATMUL_V3_MAX_SHAPE_SIZE)) {
        OP_LOGE(context->GetNodeName(), "[InferShape] The shape can only be in the range of 2 to 6.");
        return ge::GRAPH_FAILED;
    }
    // first transpose attr is transpose_x1, its index is 1 and bias input tensor index is 4
    return Ops::NN::InferShapeForBatchMatMul(context, 1, 4UL);
}

// 由于编译期InferShapeRange在InferShape之后，因此该函数无需重复InferShape的dtype校验
static ge::graphStatus InferShapeRangeForQuantBatchMatmulV3(gert::InferShapeRangeContext *context) {
  // first transpose attr is transpose_x1, its index is 1 and bias input tensor index is 4
    return Ops::NN::InferShapeRangeForBatchMatMul(context, 1, 4);
}

} // namespace
namespace Ops::NN::MatMul {
IMPL_OP_INFERSHAPE(QuantBatchMatmulV3)
    .InferShape(InferShapeForQuantBatchMatmulV3)
    .InferShapeRange(InferShapeRangeForQuantBatchMatmulV3);
}
