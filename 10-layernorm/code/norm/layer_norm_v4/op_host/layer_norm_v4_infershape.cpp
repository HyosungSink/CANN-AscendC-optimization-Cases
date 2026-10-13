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
 * \file layer_norm_v4_infershape.cpp
 * \brief
 */

#include "op_common/log/log.h"
#include "error_util.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
constexpr size_t INPUT_IDX_X = 0;
constexpr size_t INPUT_IDX_NORM_SHAPE = 1;
constexpr size_t OUTPUT_IDX_Y = 0;
constexpr size_t OUTPUT_IDX_MEAN = 1;
constexpr size_t OUTPUT_IDX_RSTD = 2;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2LL;
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;

static inline bool IsUnknownRank(const gert::Shape* check_shape)
{
    return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}

inline ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape* output_shape)
{
    OP_CHECK_IF(
        output_shape == nullptr, OP_LOGD("SetAllUnknownDim", "the output_shape is nullptr, return unsuccess"),
        return ge::GRAPH_FAILED);
    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
    }
    OP_LOGD("SetAllUnknownDim", "set all dim = -1, output = %s", Ops::Base::ToString(*output_shape).c_str());
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferShape4LayerNormV4(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Begin to do InferShape4LayerNormV4.");

    const gert::Shape* x_shape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* y_shape = context->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    gert::Shape* mean_shape = context->GetOutputShape(OUTPUT_IDX_MEAN);
    OP_CHECK_NULL_WITH_CONTEXT(context, mean_shape);
    gert::Shape* rstd_shape = context->GetOutputShape(OUTPUT_IDX_RSTD);
    OP_CHECK_NULL_WITH_CONTEXT(context, rstd_shape);

    *y_shape = *x_shape;
    *mean_shape = *x_shape;
    *rstd_shape = *x_shape;
    OP_CHECK_IF(
        IsUnknownRank(x_shape), OP_LOGI(context, "End to do InferShape4LayerNormV4, inputx is [-2]."),
        return GRAPH_SUCCESS);

    const gert::Shape* norm_shape = context->GetInputShape(INPUT_IDX_NORM_SHAPE);
    OP_CHECK_IF(
        norm_shape->GetDimNum() > 1, OP_LOGE(context, "Shape of norm_shape should be 1 dimensions!"),
        return GRAPH_FAILED);

    int64_t norm_shape_len = norm_shape->IsScalar() ? 1 : norm_shape->GetDim(0);
    if (norm_shape_len < 0) {
        OP_CHECK_IF(
            SetAllUnknownDim(x_shape->GetDimNum(), mean_shape) != GRAPH_SUCCESS,
            OP_LOGE(context, "do InferShape4LayerNormV4 failed!"), return GRAPH_FAILED);
        *rstd_shape = *mean_shape;
        OP_LOGI(context, "End to do InferShape4LayerNormV4, norm_shape is unknown.");
        return GRAPH_SUCCESS;
    }

    OP_CHECK_IF(
        static_cast<int64_t>(x_shape->GetDimNum()) < norm_shape_len,
        OP_LOGE(context, "norm_shape_len must be <= xshape rank!"), return GRAPH_FAILED);

    int64_t begin_norm_axis_val = x_shape->GetDimNum() - norm_shape_len;
    for (size_t i = 0; i < x_shape->GetDimNum(); ++i) {
        if (static_cast<int64_t>(i) >= begin_norm_axis_val) {
            mean_shape->SetDim(i, 1);
            rstd_shape->SetDim(i, 1);
        } else {
            mean_shape->SetDim(i, x_shape->GetDim(i));
            rstd_shape->SetDim(i, x_shape->GetDim(i));
        }
    }

    OP_LOGD(context, "End to do InferShape4LayerNorm.");
    return GRAPH_SUCCESS;
}

static graphStatus InferDtype4LayerNormV4(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "InferDtype4LayerNormV4 enter");

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto input_dtype = context->GetInputDataType(INPUT_IDX_X);

    context->SetOutputDataType(OUTPUT_IDX_Y, input_dtype);

    OP_LOGD(context, "set output dtype: %s", Ops::Base::ToString(input_dtype).c_str());
    context->SetOutputDataType(OUTPUT_IDX_MEAN, DT_FLOAT);
    context->SetOutputDataType(OUTPUT_IDX_RSTD, DT_FLOAT);
    OP_LOGD(context, "InferDtype4LayerNormV4 end");

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRange4LayerNormV4(gert::InferShapeRangeContext* context)
{
    OP_LOGD(context, "InferShapeRange4LayerNormV4 enter");
    auto x_shape_range = context->GetInputShapeRange(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape_range);
    auto norm_shape_range = context->GetInputShapeRange(INPUT_IDX_NORM_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, norm_shape_range);
    auto y_shape_range = context->GetOutputShapeRange(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape_range);
    auto mean_shape_range = context->GetOutputShapeRange(OUTPUT_IDX_MEAN);
    OP_CHECK_NULL_WITH_CONTEXT(context, mean_shape_range);
    auto rstd_shape_range = context->GetOutputShapeRange(OUTPUT_IDX_RSTD);
    OP_CHECK_NULL_WITH_CONTEXT(context, rstd_shape_range);

    OP_LOGD(
        context, "InferShapeRange4LayerNormV4 x_shape_range->GetMin() = %s",
        Ops::Base::ToString(*x_shape_range->GetMin()).c_str());
    OP_LOGD(
        context, "InferShapeRange4LayerNormV4 x_shape_range->GetMax() = %s",
        Ops::Base::ToString(*x_shape_range->GetMax()).c_str());
    OP_LOGD(
        context, "InferShapeRange4LayerNormV4 norm_shape_range->GetMin() = %s",
        Ops::Base::ToString(*norm_shape_range->GetMin()).c_str());
    OP_LOGD(
        context, "InferShapeRange4LayerNormV4 norm_shape_range->GetMax() = %s",
        Ops::Base::ToString(*norm_shape_range->GetMax()).c_str());

    bool is_need_update_y_range = y_shape_range->GetMax() != nullptr && y_shape_range->GetMin() != nullptr;
    bool is_need_update_mean_range = mean_shape_range->GetMax() != nullptr && mean_shape_range->GetMin() != nullptr;

    size_t output_shape_dim_num = x_shape_range->GetMax()->GetDimNum();

    auto norm_shape_max_shape = norm_shape_range->GetMax();
    int64_t norm_shape_max_len = norm_shape_max_shape->GetDimNum() == 0 ? 1 : norm_shape_max_shape->GetDim(0);
    auto norm_shape_min_shape = norm_shape_range->GetMin();
    int64_t norm_shape_min_len = norm_shape_min_shape->GetDimNum() == 0 ? 1 : norm_shape_min_shape->GetDim(0);

    norm_shape_max_len = norm_shape_max_len == -1 ? output_shape_dim_num : norm_shape_max_len;
    norm_shape_min_len = norm_shape_min_len == -1 ? 0 : norm_shape_min_len;
    OP_CHECK_IF(
        static_cast<int64_t>(norm_shape_max_len) > static_cast<int64_t>(output_shape_dim_num),
        OP_LOGE(context, "norm_shape_len must be <= xshape rank!"), return GRAPH_FAILED);
    OP_CHECK_IF(
        static_cast<int64_t>(norm_shape_min_len) > static_cast<int64_t>(output_shape_dim_num),
        OP_LOGE(context, "norm_shape_len must be <= xshape rank!"), return GRAPH_FAILED);
    if (is_need_update_y_range) {
        y_shape_range->GetMax()->SetDimNum(output_shape_dim_num);
        y_shape_range->GetMin()->SetDimNum(output_shape_dim_num);

        for (size_t i = 0U; i < output_shape_dim_num; i++) {
            y_shape_range->GetMin()->SetDim(i, x_shape_range->GetMin()->GetDim(i));
            y_shape_range->GetMax()->SetDim(i, x_shape_range->GetMax()->GetDim(i));
        }
    }

    if (is_need_update_mean_range) {
        mean_shape_range->GetMax()->SetDimNum(output_shape_dim_num);
        mean_shape_range->GetMin()->SetDimNum(output_shape_dim_num);
        rstd_shape_range->GetMax()->SetDimNum(output_shape_dim_num);
        rstd_shape_range->GetMin()->SetDimNum(output_shape_dim_num);
        for (size_t i = 0U; i < output_shape_dim_num; i++) {
            mean_shape_range->GetMax()->SetDim(i, x_shape_range->GetMax()->GetDim(i));
            mean_shape_range->GetMin()->SetDim(i, x_shape_range->GetMin()->GetDim(i));
            rstd_shape_range->GetMax()->SetDim(i, x_shape_range->GetMax()->GetDim(i));
            rstd_shape_range->GetMin()->SetDim(i, x_shape_range->GetMin()->GetDim(i));
            if (static_cast<int64_t>(i) >= static_cast<int64_t>(output_shape_dim_num) - norm_shape_min_len) {
                mean_shape_range->GetMax()->SetDim(i, 1);
                mean_shape_range->GetMin()->SetDim(i, 1);
                rstd_shape_range->GetMax()->SetDim(i, 1);
                rstd_shape_range->GetMin()->SetDim(i, 1);
            } else if (static_cast<int64_t>(i) >= static_cast<int64_t>(output_shape_dim_num) - norm_shape_max_len) {
                mean_shape_range->GetMin()->SetDim(i, 0);
                rstd_shape_range->GetMin()->SetDim(i, 0);
            }
        }
    }
    OP_LOGD(context, "InferShapeRange4LayerNormV4 end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LayerNormV4)
    .InferShape(InferShape4LayerNormV4)
    .InferDataType(InferDtype4LayerNormV4)
    .InferShapeRange(InferShapeRange4LayerNormV4);
} // namespace ops
