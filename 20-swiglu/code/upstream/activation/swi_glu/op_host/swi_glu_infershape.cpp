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
 * \file swi_glu_infer.cc
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "util/shape_util.h"

using namespace ge;

namespace {
constexpr size_t GLU_IN_X = 0;
constexpr size_t GLU_OUT_Y = 0;
constexpr size_t GLU_ATTR_DIM = 0;
const size_t SPLIT_NUM = 2;
}  // namespace

namespace ops {
static ge::graphStatus InferShapeForSwiGlu(gert::InferShapeContext* context) {
    auto x_shape = context->GetInputShape(GLU_IN_X);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    auto y_shape = context->GetOutputShape(GLU_OUT_Y);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto split_dim_ptr = attrs->GetAttrPointer<int64_t>(GLU_ATTR_DIM);
    OPS_CHECK_NULL_WITH_CONTEXT(context, split_dim_ptr);

    auto split_dim = *split_dim_ptr;
    if (split_dim < 0) {
      split_dim += x_shape->GetDimNum();
    }

    if (Ops::Base::IsUnknownRank(*x_shape)) {
        OP_LOGD("SwiGlu", "inputs have unknown rank, no need check.");
        *y_shape = *x_shape;
        return ge::GRAPH_SUCCESS;
    }

    if (split_dim < 0 || split_dim >= static_cast<int64_t>(x_shape->GetDimNum())) {
      OP_LOGE("SwiGlu", "The value of attr [dim] must be in the range [-%zu, %zu], but got [%ld].",
                  x_shape->GetDimNum(), x_shape->GetDimNum() - 1, split_dim);
      return GRAPH_FAILED;
    }

    *y_shape = *x_shape;
    // dynamic shape
    if (x_shape->GetDim(split_dim) == -1) {
        return ge::GRAPH_SUCCESS;
    }
    if (x_shape->GetDim(split_dim) < 0 || x_shape->GetDim(split_dim) % SPLIT_NUM != 0) {
        OP_LOGE("SwiGlu", "The shape [%s] is not divisible by 2.", Ops::Base::ToString(*x_shape).c_str());
        return ge::GRAPH_FAILED;
    }

    y_shape->SetDim(split_dim, x_shape->GetDim(split_dim) / SPLIT_NUM);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSwiGlu(gert::InferDataTypeContext *context) {
    const ge::DataType dtype = context->GetInputDataType(0);
    ge::graphStatus ret = context->SetOutputDataType(0, dtype);
    return ret;
}

IMPL_OP_INFERSHAPE(SwiGlu).InferShape(InferShapeForSwiGlu).InferDataType(InferDataTypeForSwiGlu);
}  // namespace ops
