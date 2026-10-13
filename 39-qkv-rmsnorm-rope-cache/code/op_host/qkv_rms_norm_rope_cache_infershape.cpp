/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qkv_rms_norm_rope_cache_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
using namespace ge;
namespace ops {
// Tensor Indices
constexpr int64_t QKV_INDEX = 0;
// In-place inputs
constexpr int64_t Q_OUT_INDEX = 6;
constexpr int64_t K_CACHE_INDEX = 7;  
constexpr int64_t V_CACHE_INDEX = 8;    
constexpr int64_t IS_OUTPUT_QKV_IDX = 4;

constexpr size_t OUTPUT_IDX_Q_OUT = 0;
constexpr size_t OUTPUT_IDX_K_CACHE = 1;
constexpr size_t OUTPUT_IDX_V_CACHE = 2;
constexpr size_t OUTPUT_IDX_Q_OUT_PROTO = 3;
constexpr size_t OUTPUT_IDX_K_CACHE_PROTO = 4;
constexpr size_t OUTPUT_IDX_V_CACHE_PROTO = 5;

constexpr size_t EXPECT_DIM_NUM = 2;
constexpr size_t Q_EXPECT_DIM_NUM = 2;
constexpr size_t KV_EXPECT_DIM_NUM = 4;
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;

constexpr int64_t DIM_ZERO = 0;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;

inline ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape* output_shape)
{
    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
    }
    return ge::GRAPH_SUCCESS;
}

graphStatus InferShape4QkvRmsNormRopeCache(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Begin to do InferShape4QkvRmsNormRopeCache.");

    const gert::Shape* qkvInputShape = context->GetInputShape(QKV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qkvInputShape);
    const gert::Shape* qOutInputShape = context->GetInputShape(Q_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qOutInputShape);
    const gert::Shape* kCacheInputShape = context->GetInputShape(K_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheInputShape);
    const gert::Shape* vCacheInputShape = context->GetInputShape(V_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, vCacheInputShape);

    gert::Shape* qOutShape = context->GetOutputShape(OUTPUT_IDX_Q_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, qOutShape);
    gert::Shape* kCacheShape = context->GetOutputShape(OUTPUT_IDX_K_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheShape);
    gert::Shape* vCacheShape = context->GetOutputShape(OUTPUT_IDX_V_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, vCacheShape);
    gert::Shape* qOutProtoShape = context->GetOutputShape(OUTPUT_IDX_Q_OUT_PROTO);
    OP_CHECK_NULL_WITH_CONTEXT(context, qOutProtoShape);
    gert::Shape* kCacheProtoShape = context->GetOutputShape(OUTPUT_IDX_K_CACHE_PROTO);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheProtoShape);
    gert::Shape* vCacheProtoShape = context->GetOutputShape(OUTPUT_IDX_V_CACHE_PROTO);
    OP_CHECK_NULL_WITH_CONTEXT(context, vCacheProtoShape);
    
    int64_t qOutDimSize = qOutInputShape->GetDimNum();
    int64_t qkvDimSize = qkvInputShape->GetDimNum();
    
    auto attrs = context->GetAttrs();
    const bool* isOutputQkv = attrs->GetBool(IS_OUTPUT_QKV_IDX);
    bool tmp = isOutputQkv == nullptr ? false : *isOutputQkv;
    
    *qOutShape = *qOutInputShape;
    *kCacheShape = *kCacheInputShape;
    *vCacheShape = *vCacheInputShape;
    if(!tmp){
        return GRAPH_SUCCESS;
    }
    *qOutProtoShape = *qOutInputShape;
    *kCacheProtoShape = *qkvInputShape;
    *vCacheProtoShape = *qkvInputShape;

    // -2 场景将q_out_proto可选输出设置为[-1, -1]
    if (Ops::Base::IsUnknownRank(*qOutInputShape)) {
        SetAllUnknownDim(Q_EXPECT_DIM_NUM, qOutProtoShape);
        // 非-2场景校验输入shape为2维
    } else {
        if (qOutDimSize != Q_EXPECT_DIM_NUM) {
            OP_LOGE(context, "don't support q_out dimSize != 2 , infershape failed");
            return ge::GRAPH_FAILED;
        }
    }

    // -2 场景将k_cache_proto, v_cache_proto可选输出设置为[-1, -1]
    if (Ops::Base::IsUnknownRank(*qkvInputShape)) {
        SetAllUnknownDim(EXPECT_DIM_NUM, kCacheProtoShape);
        SetAllUnknownDim(EXPECT_DIM_NUM, vCacheProtoShape);
        // 非-2场景校验输入shape为2维
    } else {
        if (qkvDimSize != Q_EXPECT_DIM_NUM) {
            OP_LOGE(context, "don't support qkv dimSize != 2 , infershape failed");
            return ge::GRAPH_FAILED;
        }
    }
    
    if (!Ops::Base::IsUnknownRank(*kCacheInputShape)) {
        kCacheProtoShape->SetDim(DIM_ONE, kCacheInputShape->GetDim(DIM_ONE) * kCacheInputShape->GetDim(DIM_THREE));
    }
    if (!Ops::Base::IsUnknownRank(*vCacheInputShape)) {
        vCacheProtoShape->SetDim(DIM_ONE, vCacheInputShape->GetDim(DIM_ONE) * vCacheInputShape->GetDim(DIM_THREE));
    }

    OP_LOGD(context, "End to do InferShape4QkvRmsNormRopeCache.");
    return GRAPH_SUCCESS;
}

graphStatus InferDtype4QkvRmsNormRopeCache(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "InferShape4QkvRmsNormRopeCache enter");

    auto qkv_input_dtype = context->GetInputDataType(QKV_INDEX);
    auto q_input_dtype = context->GetInputDataType(Q_OUT_INDEX);
    auto k_input_dtype = context->GetInputDataType(K_CACHE_INDEX);
    auto v_input_dtype = context->GetInputDataType(V_CACHE_INDEX);

    context->SetOutputDataType(OUTPUT_IDX_Q_OUT, q_input_dtype);
    context->SetOutputDataType(OUTPUT_IDX_K_CACHE, k_input_dtype);
    context->SetOutputDataType(OUTPUT_IDX_V_CACHE, v_input_dtype);
    context->SetOutputDataType(OUTPUT_IDX_Q_OUT_PROTO, qkv_input_dtype);
    context->SetOutputDataType(OUTPUT_IDX_K_CACHE_PROTO, qkv_input_dtype);
    context->SetOutputDataType(OUTPUT_IDX_V_CACHE_PROTO, qkv_input_dtype);

    OP_LOGD(context, "InferDtype4KvRmsNormRopeCache end");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QkvRmsNormRopeCache)
    .InferShape(InferShape4QkvRmsNormRopeCache)
    .InferDataType(InferDtype4QkvRmsNormRopeCache);
} // namespace ops