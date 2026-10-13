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
 * \file group_norm_silu_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {
static constexpr size_t GROUPNORMSILU_IDX_IN_X = 0;
static constexpr size_t GROUPNORMSILU_IDX_OUT_Y = 0;
static constexpr size_t GROUPNORMSILU_IDX_OUT_MEAN = 1;
static constexpr size_t GROUPNORMSILU_IDX_OUT_VAR = 2;
static constexpr size_t NUMGROUPS_IDX = 0;
static constexpr size_t N_IDX = 0;

static ge::graphStatus GroupNormSiluInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do GroupNormSiluInferShape");

    // get input shapes
    const gert::Shape* x_shape = context->GetInputShape(GROUPNORMSILU_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    // get output shapes
    gert::Shape* y_shape = context->GetOutputShape(GROUPNORMSILU_IDX_OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    gert::Shape* mean_shape = context->GetOutputShape(GROUPNORMSILU_IDX_OUT_MEAN);
    OP_CHECK_NULL_WITH_CONTEXT(context, mean_shape);
    gert::Shape* var_shape = context->GetOutputShape(GROUPNORMSILU_IDX_OUT_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, var_shape);

    *y_shape = *x_shape;
    mean_shape->SetDimNum(0);

    // process attr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* num_groups = attrs->GetAttrPointer<int64_t>(NUMGROUPS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, num_groups);

    // update mean and var shape
    const int64_t n_dim = x_shape->GetDim(N_IDX);
    mean_shape->AppendDim(n_dim);
    mean_shape->AppendDim(*num_groups);
    *var_shape = *mean_shape;

    OP_LOGD(context, "End to do GroupNormSiluInferShape");
    return ge::GRAPH_SUCCESS;
}

static graphStatus GroupNormSiluInferDtype(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "GroupNormSiluInferDtype enter");

    // Get input tout
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto inputDtype = context->GetInputDataType(GROUPNORMSILU_IDX_IN_X);
    context->SetOutputDataType(GROUPNORMSILU_IDX_OUT_Y, inputDtype);

    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    OP_CHECK_IF(
        (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
         ge::GRAPH_SUCCESS),
        OP_LOGE(context, "Cannot get platform info!"), return ge::GRAPH_FAILED);
    OP_LOGD(context, "soc version is %s", platform_info.str_info.short_soc_version.c_str());
    if (platform_info.str_info.short_soc_version == "Ascend950") {
        auto gammaDtype = context->GetOptionalInputDataType(GROUPNORMSILU_IDX_OUT_MEAN);
        auto betaDtype = context->GetOptionalInputDataType(GROUPNORMSILU_IDX_OUT_VAR);
        bool validGammaDtype = ((gammaDtype != ge::DT_UNDEFINED) && (gammaDtype != ge::DT_MAX));
        bool validBetaDtype = ((betaDtype != ge::DT_UNDEFINED) && (betaDtype != ge::DT_MAX));
        if (validGammaDtype && validBetaDtype) {
            OP_CHECK_IF(
                gammaDtype != betaDtype, OP_LOGE(context, "gammaDtype not same as betaDtype"),
                return ge::GRAPH_FAILED);
        }
        if (validGammaDtype) {
            context->SetOutputDataType(GROUPNORMSILU_IDX_OUT_MEAN, gammaDtype);
            context->SetOutputDataType(GROUPNORMSILU_IDX_OUT_VAR, gammaDtype);
        } else if (validBetaDtype) {
            context->SetOutputDataType(GROUPNORMSILU_IDX_OUT_MEAN, betaDtype);
            context->SetOutputDataType(GROUPNORMSILU_IDX_OUT_VAR, betaDtype);
        } else {
            context->SetOutputDataType(GROUPNORMSILU_IDX_OUT_MEAN, inputDtype);
            context->SetOutputDataType(GROUPNORMSILU_IDX_OUT_VAR, inputDtype);
        }
    } else {
        context->SetOutputDataType(GROUPNORMSILU_IDX_OUT_MEAN, inputDtype);
        context->SetOutputDataType(GROUPNORMSILU_IDX_OUT_VAR, inputDtype);
    }

    OP_LOGD(context, "GroupNormSiluInferDtype end");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupNormSilu).InferShape(GroupNormSiluInferShape).InferDataType(GroupNormSiluInferDtype);
} // namespace ops
