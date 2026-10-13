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
 * \file nsa_compress_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "platform/platform_info.h"
#include "log/log.h"

using namespace ge;
namespace ops {

constexpr size_t INPUT_INPUT_INDEX = 0;
constexpr size_t ACTSEQLEN_INPUT_INDEX = 2;
constexpr size_t COMPRESSBLOCKSIZE_ATTRS_INDEX = 1;
constexpr size_t COMPRESSSTRIDE_ATTRS_INDEX = 2;
constexpr size_t OUTPUT_INDEX = 0;

static ge::graphStatus InferShapeNsaCompress(gert::InferShapeContext *context)
{
    OP_LOGI(context, "Enter NsaCompress runtime infershape impl.");
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *inputShape = context->GetInputShape(INPUT_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    uint32_t headNums = inputShape->GetDim(1);
    uint32_t headDims = inputShape->GetDim(2);

    const gert::Shape *actSeqLenShape = context->GetInputShape(ACTSEQLEN_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, actSeqLenShape);
    uint32_t batchSize = actSeqLenShape->GetDim(0);

    const gert::Tensor *actSeqLenTensor = context->GetOptionalInputTensor(ACTSEQLEN_INPUT_INDEX);
    const int64_t *actualSeqLen = actSeqLenTensor->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context, actualSeqLen);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    auto compressBlockSizePtr = attrs->GetAttrPointer<int64_t>(COMPRESSBLOCKSIZE_ATTRS_INDEX);
    uint32_t compressBlockSize = *compressBlockSizePtr;

    auto compressStridePtr = attrs->GetAttrPointer<int64_t>(COMPRESSSTRIDE_ATTRS_INDEX);
    uint32_t compressStride = *compressStridePtr;

    int64_t compressKvNum = 0;
    int64_t preSeqLen = 0;
    for (size_t i = 0; i < batchSize; i++) {
        int64_t cur_seq_len = actualSeqLen[i] - preSeqLen;
        if (cur_seq_len >= compressBlockSize) {
            compressKvNum += (cur_seq_len - compressBlockSize + compressStride) / compressStride;
        }
        preSeqLen += cur_seq_len;
    }
    gert::Shape out = gert::Shape({compressKvNum, headNums, headDims});

    gert::Shape *yShape = context->GetOutputShape(OUTPUT_INDEX);
    *yShape = out;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeNsaCompress(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(INPUT_INPUT_INDEX);
    context->SetOutputDataType(OUTPUT_INDEX, inputDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NsaCompress).InferShape(InferShapeNsaCompress).InferDataType(InferDataTypeNsaCompress);


} // namespace ops
