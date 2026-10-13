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
 * \file ffn_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShapeFFN(gert::InferShapeContext *context)
{
    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFFN(gert::InferDataTypeContext *context)
{
    auto input_x_dtype = context->GetInputDataType(0);
    if (input_x_dtype == ge::DT_INT8) {
        auto attrs = context->GetAttrs();
        const int64_t *output_dtype = attrs->GetInt(2);
        if (output_dtype != nullptr && *output_dtype == 1) {
            context->SetOutputDataType(0, ge::DT_BF16);
        } else {
            context->SetOutputDataType(0, ge::DT_FLOAT16);
        }
    } else {
        context->SetOutputDataType(0, input_x_dtype);
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FFN).InferShape(InferShapeFFN).InferDataType(InferDataTypeFFN);
} // namespace ops
