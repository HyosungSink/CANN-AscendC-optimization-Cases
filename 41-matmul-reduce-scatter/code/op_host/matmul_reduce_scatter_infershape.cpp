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
 * \file matmul_reduce_scatter_infershape.cpp
 * \brief
 */
#include "../common/inc/mc2_log.h"
#include "../common/inc/mc2_common_infershape.h"
#include "../common/inc/context_util.h"
#include "../common/inc/mc2_hcom_topo_info.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeMatmulReduceScatter(gert::InferShapeContext* context)
{
    OP_LOGE_IF(
        InferMatmulReduceScatterCommon(context) != GRAPH_SUCCESS, GRAPH_FAILED, context->GetNodeName(),
        "infer shape excute failed.");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMatmulReduceScatter(gert::InferDataTypeContext* context)
{
    auto d_type = context->GetInputDataType(0);
    context->SetOutputDataType(0, d_type);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MatmulReduceScatter)
    .InferShape(InferShapeMatmulReduceScatter)
    .InferDataType(InferDataTypeMatmulReduceScatter);
} // namespace ops
