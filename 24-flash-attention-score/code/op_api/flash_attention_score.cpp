/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FlashAttentionScore);

const std::array<const aclTensor *, 4> FlashAttentionScore(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclTensor *sinkOptional, const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional, const aclTensor *dScaleQOptional, const aclTensor *dScaleKOptional,
    const aclTensor *dScaleVOptional, const aclTensor *pScaleOptional, const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional, double scaleValue, double keepProb, int64_t preTockens,
    int64_t nextTockens, int64_t headNum, const char *inputLayout, int64_t innerPrecise,
    int64_t sparseMode, int64_t pseType, int64_t seed, int64_t offset, int64_t outDtype, const char *softmaxOutLayout,
    aclOpExecutor *executor)
{
    L0_DFX(FlashAttentionScore, query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional,
           attenMaskOptional, sinkOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional,
           kvStartIdxOptional, dScaleQOptional, dScaleKOptional, dScaleVOptional, pScaleOptional, queryRopeOptional,
           keyRopeOptional, scaleValue, keepProb, preTockens, nextTockens, headNum, inputLayout, innerPrecise, sparseMode,
           pseType, seed, offset, outDtype, softmaxOutLayout);

    if (realShiftOptional == nullptr) {
        realShiftOptional = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (dropMaskOptional == nullptr) {
        dropMaskOptional = executor->AllocTensor(DataType::DT_UINT8, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (paddingMaskOptional == nullptr) {
        paddingMaskOptional = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (attenMaskOptional == nullptr) {
        attenMaskOptional = executor->AllocTensor(DataType::DT_BOOL, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (dScaleQOptional == nullptr) {
        dScaleQOptional = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (dScaleKOptional == nullptr) {
        dScaleKOptional = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (dScaleVOptional == nullptr) {
        dScaleVOptional = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (pScaleOptional == nullptr) {
        pScaleOptional = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor *prefixOptionalTensor = nullptr;
    if (prefixOptional) {
        prefixOptionalTensor = executor->ConvertToTensor(prefixOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(prefixOptionalTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(prefixOptionalTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(prefixOptionalTensor)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        prefixOptionalTensor = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor *actualSeqQLen = nullptr;
    if (actualSeqQLenOptional) {
        actualSeqQLen = executor->ConvertToTensor(actualSeqQLenOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualSeqQLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqQLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqQLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualSeqQLen = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor *actualSeqKvLen = nullptr;
    if (actualSeqKvLenOptional) {
        actualSeqKvLen = executor->ConvertToTensor(actualSeqKvLenOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualSeqKvLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqKvLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqKvLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualSeqKvLen = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor *qStartIdxOptionalTensor = nullptr;
    if (qStartIdxOptional) {
        qStartIdxOptionalTensor = executor->ConvertToTensor(qStartIdxOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(qStartIdxOptionalTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(qStartIdxOptionalTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(qStartIdxOptionalTensor)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        qStartIdxOptionalTensor = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor *kvStartIdxOptionalTensor = nullptr;
    if (kvStartIdxOptional) {
        kvStartIdxOptionalTensor = executor->ConvertToTensor(kvStartIdxOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(kvStartIdxOptionalTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(kvStartIdxOptionalTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(kvStartIdxOptionalTensor)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        kvStartIdxOptionalTensor = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    auto softmaxMaxOut = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxSumOut = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    DataType outputDtype = query->GetDataType();
    if (query->GetDataType() == DataType::DT_FLOAT8_E4M3FN ||
        query->GetDataType() == DataType::DT_FLOAT8_E5M2 ||
        query->GetDataType() == DataType::DT_HIFLOAT8) {
        if (outDtype == 0) {
            outputDtype = DataType::DT_FLOAT16;
         } else {
            outputDtype = DataType::DT_BF16;
        }
    }
    auto softmaxOutOut = executor->AllocTensor(outputDtype, Format::FORMAT_ND, Format::FORMAT_ND);
    auto attentionOutOut = executor->AllocTensor(outputDtype, Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(FlashAttentionScore,
                           OP_INPUT(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional,
                                    attenMaskOptional, prefixOptionalTensor, actualSeqQLen, actualSeqKvLen,
                                    qStartIdxOptionalTensor, kvStartIdxOptionalTensor, dScaleQOptional, dScaleKOptional,
                                    dScaleVOptional, queryRopeOptional, keyRopeOptional, sinkOptional, pScaleOptional),
                           OP_OUTPUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut),
                           OP_ATTR(static_cast<float>(scaleValue), static_cast<float>(keepProb),
                                   preTockens, nextTockens, headNum, inputLayout, innerPrecise,
                                   sparseMode, pseType, seed, offset, outDtype, softmaxOutLayout));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FlashAttentionScore InferShape failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        FlashAttentionScore,
        OP_INPUT(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
                 prefixOptionalTensor, actualSeqQLen, actualSeqKvLen, qStartIdxOptionalTensor, kvStartIdxOptionalTensor,
                 dScaleQOptional, dScaleKOptional, dScaleVOptional, queryRopeOptional, keyRopeOptional,
                 sinkOptional, pScaleOptional),
        OP_OUTPUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut),
        OP_ATTR(static_cast<float>(scaleValue), static_cast<float>(keepProb), preTockens,
                nextTockens, headNum, inputLayout, innerPrecise, sparseMode, pseType,
                seed, offset, outDtype, softmaxOutLayout));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FlashAttentionScore launch kernel failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }
    return {softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut};
}

} // namespace l0op
