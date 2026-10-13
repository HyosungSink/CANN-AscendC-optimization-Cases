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
 * \file nsa_compress_tiling.cpp
 * \brief
 */
#include "nsa_compress_tiling.h"
#include "nsa_compress_tiling_common.h"

#include <climits>
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "../common/include/err/ops_err.h"
#include "../common/include/tiling_base/tiling_base.h"
using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;
namespace optiling {

static ge::graphStatus CheckParams(const gert::TilingContext *context)
{
    if (context->GetInputShape(INPUT_INPUT_INDEX) != nullptr && context->GetInputShape(WEIGHT_INPUT_INDEX) != nullptr &&
        context->GetAttrs() != nullptr) {
        // shape=[T, N, D]
        auto &inputShape = context->GetInputShape(INPUT_INPUT_INDEX)->GetStorageShape();
        // shape=[compressBlockSize, N]
        auto &weightShape = context->GetInputShape(WEIGHT_INPUT_INDEX)->GetStorageShape();
        auto actSeqLenTensor = context->GetOptionalInputTensor(ACT_SEQ_LEN_INPUT_INDEX);
        auto &actSeqLenShape = actSeqLenTensor->GetShape().GetStorageShape();
        const char *inputLayout = context->GetAttrs()->GetAttrPointer<char>(INPUTLAYOUT_ATTRS_INDEX);
        const int64_t inputCompressBlockSize =
            *context->GetAttrs()->GetAttrPointer<int64_t>(COMPRESS_BLOCK_SIZE_ATTRS_INDEX);
        const int64_t inputCompressStride = *context->GetAttrs()->GetAttrPointer<int64_t>(COMPRESS_STRIDE_ATTRS_INDEX);

        const int64_t actseqlenType = *context->GetAttrs()->GetAttrPointer<int64_t>(ACT_SEQ_LEN_TYPE_ATTRS_INDEX);

        OP_CHECK_IF((inputLayout[0] != 'T' || inputLayout[1] != 'N' || inputLayout[2] != 'D'),
                   OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "The inputLayout currently only supports the 'TND' format"),
                   return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            (inputShape.GetDim(1) != weightShape.GetDim(1)),
            OPS_REPORT_VECTOR_INNER_ERR(
                context->GetNodeName(),
                    "input.shape[1] must equal weight.shape[1], but got input.shape[1]=%ld, weight.shape[1]=%ld",
                inputShape.GetDim(1), weightShape.GetDim(1)),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF((inputShape.GetDim(2) % 16 != 0),
                   OPS_REPORT_VECTOR_INNER_ERR(
                       context->GetNodeName(), "input.shape[2] must be a multiple of 16, but got input.shape[2]=%ld",
                       inputShape.GetDim(2)),
                   return ge::GRAPH_FAILED);

        OP_CHECK_IF((weightShape.GetDim(0) != inputCompressBlockSize),
                   OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                               "weight.shape[0] must equal compressBlockSize, but got "
                                               "weight.shape[0]=%ld, compressBlockSize=%ld",
                                               weightShape.GetDim(0), inputCompressBlockSize),
                   return ge::GRAPH_FAILED);

        OP_CHECK_IF((inputCompressBlockSize % 16 != 0),
                   OPS_REPORT_VECTOR_INNER_ERR(
                       context->GetNodeName(), "compressBlockSize must be a multiple of 16, but got compressBlockSize=%ld",
                       inputCompressBlockSize),
                   return ge::GRAPH_FAILED);

        OP_CHECK_IF((inputCompressStride % 16 != 0),
                   OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                               "compressStride must be a multiple of 16, but got compressStride=%ld",
                                               inputCompressStride),
                   return ge::GRAPH_FAILED);

        OP_CHECK_IF((inputCompressBlockSize < inputCompressStride),
                   OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                               "compressStride can not greater than compressBlockSize, but got "
                                               "compressBlockSize=%ld, compressStride=%ld",
                                               inputCompressBlockSize, inputCompressStride),
                   return ge::GRAPH_FAILED);

        OP_CHECK_IF((actseqlenType != 0),
                   OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "actseqlenType only support 0, but got actseqlenType=%ld",
                                               actseqlenType),
                   return ge::GRAPH_FAILED);

        const int64_t *actSeqLenValue = actSeqLenTensor->GetData<int64_t>();
        uint32_t batchSize = actSeqLenShape.GetDim(Zero);
        int64_t preSeqLen = 0;

        for (uint32_t i = 0; i < batchSize; ++i) {
            int64_t valueI = actSeqLenValue[i];
            OP_CHECK_IF((valueI < preSeqLen),
                   OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "actSeqLenOptional currently only supports the prefix sum format and requires values to be greater than 0, but actSeqLenOptional[%u]=%ld contains an invalid value",
                                               i, valueI),
                   return ge::GRAPH_FAILED);
            preSeqLen = valueI;
        }
        OP_CHECK_IF((preSeqLen != inputShape.GetDim(0)),
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "input.shape[0] must equal actSeqLenOptional[-1], but got "
                                               "input.shape[0]=%ld, actSeqLenOptional[-1]=%ld",
                                        inputShape.GetDim(0), preSeqLen),
            return ge::GRAPH_FAILED);

        return ge::SUCCESS;
    }
    OP_LOGW(context, "fail to get shape or attr from context");
    return ge::GRAPH_FAILED;
}

static bool IsEmptyInput(gert::TilingContext *context)
{
    auto inputShape = context->GetInputShape(INPUT_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto weightShape = context->GetInputShape(WEIGHT_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    auto actSeqLenTensor = context->GetOptionalInputTensor(ACT_SEQ_LEN_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, actSeqLenTensor);
    int64_t inputShapeSize = inputShape->GetStorageShape().GetShapeSize();
    int64_t weightShapeSize = weightShape->GetStorageShape().GetShapeSize();
    if ((inputShapeSize == 0 || weightShapeSize == 0)) {
        return true;
    }
    auto &actSeqLenShape = actSeqLenTensor->GetShape().GetStorageShape();

    OP_CHECK_IF(actSeqLenShape.GetDimNum() != 1,
                OP_LOGE(context->GetNodeName(),
                "NsaCompress actSeqLenShape is invalid %lu %ld", actSeqLenShape.GetDimNum(), actSeqLenShape.GetDim(0)),
                return ge::GRAPH_FAILED);
    const int64_t *actSeqLenValue = actSeqLenTensor->GetData<int64_t>();
    OP_CHECK_IF(actSeqLenValue == nullptr,
                OP_LOGE(context->GetNodeName(),
                "NsaCompress actSeqLenTensor data is null pointer"),
                return ge::GRAPH_FAILED);

    return false;
}

ASCENDC_EXTERN_C ge::graphStatus TilingNsaCompress(gert::TilingContext *context)
{
    if (IsEmptyInput(context)) {
        return ge::GRAPH_FAILED;
    }
    if (CheckParams(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto resultCode = TilingRegistry::GetInstance().DoTilingImpl(context);
    return resultCode;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForNsaCompress(gert::TilingParseContext *context)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                OP_LOGE(context->GetNodeName(),
                "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<NsaCompressCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                OP_LOGE(context->GetNodeName(),
                "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);

    OP_CHECK_IF((compileInfoPtr->aivNum == 0 || compileInfoPtr->ubSize == 0),
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "platform info is invalid, aivNum=%u, ubSize=%lu",
                                           compileInfoPtr->aivNum, compileInfoPtr->ubSize),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NsaCompress)
    .Tiling(TilingNsaCompress)
    .TilingInputsDataDependency({ACT_SEQ_LEN_INPUT_INDEX})
    .TilingParse<NsaCompressCompileInfo>(TilingPrepareForNsaCompress); // regist into the framework
} // namespace optiling
