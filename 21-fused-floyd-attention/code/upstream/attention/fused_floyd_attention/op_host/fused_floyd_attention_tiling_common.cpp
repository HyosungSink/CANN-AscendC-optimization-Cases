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
 * \file fused_floyd_attention_tiling_common.cpp
 * \brief
 */

#include "fused_floyd_attention_tiling_common.h"

namespace optiling {
namespace FFA {
uint32_t FloydCalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum)
{
    uint32_t ration;
    if (aicCoreNum == 0 || aivCoreNum == 0 || aicCoreNum > aivCoreNum) {
        return sliceNum;
    }
    ration = aivCoreNum / aicCoreNum;
    return (sliceNum + (ration - 1)) / ration;
}

ge::graphStatus CheckAttentionMaskShape(gert::TilingContext *context, int64_t b, int64_t n, int64_t k)
{
    auto attentionMaskShape = context->GetOptionalInputShape(ATTENTION_MASK_INPUT_INDEX);
    if (attentionMaskShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto attentionMaskDimNum = attentionMaskShape->GetStorageShape().GetDimNum();
    if (attentionMaskDimNum != SUPPORT_DIM_NUM) { // attentionMask only support 5 dimensions
        OP_LOGE(context, "The shape of atten_mask is invalid, got %lu dimensions", attentionMaskDimNum);
        return ge::GRAPH_FAILED;
    }
    auto dim0 = attentionMaskShape->GetStorageShape().GetDim(0); // 0:b
    auto dim1 = attentionMaskShape->GetStorageShape().GetDim(1); // 1:1
    auto dim2 = attentionMaskShape->GetStorageShape().GetDim(2); // 2:n
    auto dim3 = attentionMaskShape->GetStorageShape().GetDim(3); // 3:1
    auto dim4 = attentionMaskShape->GetStorageShape().GetDim(4); // 4:k

    OP_CHECK_IF((dim0 != b || dim1 != 1 || dim2 != n || dim3 != 1 || dim4 != k),
                OP_LOGE(context,
                        "The shape of atten_mask is invalid, got (%ld,%ld,%ld,%ld,%ld), should be (%ld,1,%ld,1,%ld)",
                        dim0, dim1, dim2, dim3, dim4, b, n, k),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSupportShape(gert::TilingContext *context)
{
    const gert::StorageShape *key1Shape = context->GetInputShape(KEY1_INPUT_INDEX);
    const gert::StorageShape *value1Shape = context->GetInputShape(VALUE1_INPUT_INDEX);
    const gert::StorageShape *key2Shape = context->GetInputShape(KEY2_INPUT_INDEX);
    const gert::StorageShape *value2Shape = context->GetInputShape(VALUE2_INPUT_INDEX);

    OP_CHECK_IF(!CheckSameShape(key1Shape, value1Shape),
                OP_LOGE(context, "key1's shape and value1's shape is different"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckSameShape(key2Shape, value2Shape),
                OP_LOGE(context, "key2's shape and value2's shape is different"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckInputShapeValid(gert::TilingContext *context, int64_t b, int64_t n, int64_t s1, int64_t s2,
                                     int64_t s3, int64_t d)
{
    auto isShapeInValid = (b == 0 || n == 0 || s1 == 0 || s2 == 0 || s3 == 0 || d == 0);
    OP_CHECK_IF(isShapeInValid,
                OP_LOGE(context, "input shape error, got 0 in bhmnkd(%ld,%ld,%ld,%ld,%ld,%ld)", b, n, s1, s2, s3, d),
                return ge::GRAPH_FAILED);

    auto ret = CheckSupportShape(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckAttentionMaskShape(context, b, s1, s3);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

bool CheckSameShape(const gert::StorageShape *aShape, const gert::StorageShape *bShape) {
    OP_CHECK_IF((aShape == nullptr) || (bShape == nullptr),
               OP_LOGW("fused_floyd_attention_tiling_common", "aShape or bShape is nullptr."),
               return false);
    uint32_t dimSizeA = aShape->GetStorageShape().GetDimNum();
    uint32_t dimSizeB = bShape->GetStorageShape().GetDimNum();
    if (dimSizeA != SUPPORT_DIM_NUM || dimSizeA != dimSizeB) {
        return false;
    }

    for (uint32_t i = 0; i < dimSizeA; i++) {
        auto dimNumA = aShape->GetStorageShape().GetDim(i);
        auto dimNumB = bShape->GetStorageShape().GetDim(i);
        if (dimNumA != dimNumB) {
            return false;
        }
    }
    return true;
}

ge::graphStatus CheckAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto scaleValuePtr = attrs->GetAttrPointer<float>(0);
    size_t *workspaces = context->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(context, scaleValuePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckBaseInput(gert::TilingContext *context)
{
    auto &queryShape = context->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto &keyShape = context->GetInputShape(KEY1_INPUT_INDEX)->GetStorageShape();
    int64_t b = queryShape.GetDim(0);
    int64_t n = queryShape.GetDim(1);
    int64_t s1 = queryShape.GetDim(2);
    int64_t s2 = queryShape.GetDim(3);
    int64_t d = queryShape.GetDim(4);
    int64_t s3 = keyShape.GetDim(3);
    OP_CHECK_IF((b != keyShape.GetDim(0) || n != keyShape.GetDim(1) || d != keyShape.GetDim(4)),
                OP_LOGE(context, "query or key1 shape is invalid"), return ge::GRAPH_FAILED);

    OP_CHECK_IF((s1 % ALIGNED_NUM_16 != 0 || s2 % ALIGNED_NUM_128 != 0),
                OP_LOGE(context, "The third dimension of the query only supports 16-byte alignment, and the fourth "
                                 "dimension only supports 128-byte alignment"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(s3 % ALIGNED_NUM_128 != 0,
                OP_LOGE(context, "The fourth dimension of the key1 only supports 128-byte alignment"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((d != 32 && d != 64 && d != 128), OP_LOGE(context, "Headdim only supports 32/64/128."),
                return ge::GRAPH_FAILED);

    if (CheckAttrs(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckInputShapeValid(context, b, n, s1, s2, s3, d) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace FFA
} // namespace optiling
