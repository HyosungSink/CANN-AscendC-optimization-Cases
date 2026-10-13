/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attention_score_grad.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

#define CHECK_NULL(aclnTensor) do { if ((aclnTensor) == nullptr) { return {nullptr, nullptr, nullptr, nullptr};}} while (0)

namespace l0op {

OP_TYPE_REGISTER(FlashAttentionScoreGrad);

const std::array<const aclTensor *, MAX_FAG_OUTPUT_CNT> FlashAttentionScoreGrad(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, const aclTensor *dScaleQOptional,
    const aclTensor *dScaleKOptional, const aclTensor *dScaleVOptional, const aclTensor *dScaleDyOptional,
    const aclTensor *dScaleOOptional, const aclTensor *dsScaleOptional, const aclTensor *pScaleOptional,
    const aclTensor *queryRope, const aclTensor *keyRope, const aclTensor *sinkInOptional,
    double scaleValueOptional, double keepProbOptional, int64_t preTockensOptional, int64_t nextTockensOptional, int64_t headNum,
    char *inputLayout, int64_t innerPreciseOptional, int64_t sparseModeOptional, int64_t pseTypeOptional,
    int64_t seed, int64_t offset, int64_t outDTypeOptional,char *softmaxInLayout, aclOpExecutor *executor)
{
    L0_DFX(FlashAttentionScoreGrad, query, key, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional,
           attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional,
           prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional, kvStartIdxOptional,
           dScaleQOptional, dScaleKOptional, dScaleVOptional, dScaleDyOptional, dScaleOOptional, dsScaleOptional, pScaleOptional,
           queryRope, keyRope, scaleValueOptional, keepProbOptional, preTockensOptional, nextTockensOptional, headNum,
           inputLayout, innerPreciseOptional, sparseModeOptional, pseTypeOptional, seed, offset, outDTypeOptional,softmaxInLayout, sinkInOptional); 
    DataType outputDtype = query->GetDataType();
    if (outputDtype == DataType::DT_FLOAT8_E4M3FN || outputDtype == DataType::DT_FLOAT8_E5M2 || outputDtype == DataType::DT_HIFLOAT8) {
        if (outDTypeOptional == 0) {
            outputDtype = DataType::DT_FLOAT16;
         } else {
            outputDtype = DataType::DT_BF16;
        }
    }
    auto dqOut = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto dkOut = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto dvOut = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto dpseOut = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto dqRopeOut = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto dkRopeOut = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto dsinkOut = executor->AllocTensor(DataType::DT_FLOAT, op::Format::FORMAT_ND, op::Format::FORMAT_ND); 

    const aclTensor *prefix = nullptr;
    if (prefixOptional) {
        prefix = executor->ConvertToTensor(prefixOptional, op::DataType::DT_INT64);
        CHECK_NULL(prefix);
        const_cast<aclTensor *>(prefix)->SetStorageFormat(op::Format::FORMAT_ND);
        const_cast<aclTensor *>(prefix)->SetViewFormat(op::Format::FORMAT_ND);
        const_cast<aclTensor *>(prefix)->SetOriginalFormat(op::Format::FORMAT_ND);
    }

    const aclTensor *actualSeqQLen = nullptr;
    if (actualSeqQLenOptional) {
        actualSeqQLen = executor->ConvertToTensor(actualSeqQLenOptional, op::DataType::DT_INT64);
        CHECK_NULL(actualSeqQLen);
        const_cast<aclTensor *>(actualSeqQLen)->SetStorageFormat(op::Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqQLen)->SetViewFormat(op::Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqQLen)->SetOriginalFormat(op::Format::FORMAT_ND);
    }

    const aclTensor *actualSeqKvLen = nullptr;
    if (actualSeqKvLenOptional) {
        actualSeqKvLen = executor->ConvertToTensor(actualSeqKvLenOptional, op::DataType::DT_INT64);
        CHECK_NULL(actualSeqKvLen);
        const_cast<aclTensor *>(actualSeqKvLen)->SetStorageFormat(op::Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqKvLen)->SetViewFormat(op::Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqKvLen)->SetOriginalFormat(op::Format::FORMAT_ND);
    }

    const aclTensor *qStartIdxOptionalTensor = nullptr;
    if (qStartIdxOptional) {
        qStartIdxOptionalTensor = executor->ConvertToTensor(qStartIdxOptional, DataType::DT_INT64);
        CHECK_NULL(qStartIdxOptionalTensor);
        const_cast<aclTensor *>(qStartIdxOptionalTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(qStartIdxOptionalTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(qStartIdxOptionalTensor)->SetOriginalFormat(Format::FORMAT_ND);
    }

    const aclTensor *kvStartIdxOptionalTensor = nullptr;
    if (kvStartIdxOptional) {
        kvStartIdxOptionalTensor = executor->ConvertToTensor(kvStartIdxOptional, DataType::DT_INT64);
        CHECK_NULL(kvStartIdxOptionalTensor);
        const_cast<aclTensor *>(kvStartIdxOptionalTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(kvStartIdxOptionalTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(kvStartIdxOptionalTensor)->SetOriginalFormat(Format::FORMAT_ND);
    }

    if (dScaleQOptional == nullptr) {
        dScaleQOptional = executor->AllocTensor(DataType::DT_FLOAT, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    }

    if (dScaleKOptional == nullptr) {
        dScaleKOptional = executor->AllocTensor(DataType::DT_FLOAT, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    }

    if (dScaleVOptional == nullptr) {
        dScaleVOptional = executor->AllocTensor(DataType::DT_FLOAT, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    }

    if (dScaleDyOptional == nullptr) {
        dScaleDyOptional = executor->AllocTensor(DataType::DT_FLOAT, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    }

    if (dScaleOOptional == nullptr) {
        dScaleOOptional = executor->AllocTensor(DataType::DT_FLOAT, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    }

    if (dsScaleOptional == nullptr) {
        dsScaleOptional = executor->AllocTensor(DataType::DT_FLOAT, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    }

    if (pScaleOptional == nullptr) {
        pScaleOptional = executor->AllocTensor(DataType::DT_FLOAT, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    }
    auto ret = INFER_SHAPE(FlashAttentionScoreGrad,
                           OP_INPUT(query, key, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional,
                                    attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional,
                                    attentionInOptional, prefix, actualSeqQLen, actualSeqKvLen,
                                    qStartIdxOptionalTensor, kvStartIdxOptionalTensor,
                                    dScaleQOptional, dScaleKOptional, dScaleVOptional, dScaleDyOptional, dScaleOOptional,
                                    queryRope, keyRope, sinkInOptional, dsScaleOptional, pScaleOptional), 
                           OP_OUTPUT(dqOut, dkOut, dvOut, dpseOut, dqRopeOut, dkRopeOut, dsinkOut), 
                           OP_ATTR(static_cast<float>(scaleValueOptional), static_cast<float>(keepProbOptional),
                                   preTockensOptional, nextTockensOptional, headNum, inputLayout, innerPreciseOptional,
                                   sparseModeOptional, pseTypeOptional, seed, offset, outDTypeOptional, softmaxInLayout));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Fag InferShape failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        FlashAttentionScoreGrad,
        OP_INPUT(query, key, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
                 softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefix, actualSeqQLen,
                 actualSeqKvLen, qStartIdxOptionalTensor, kvStartIdxOptionalTensor,
                 dScaleQOptional, dScaleKOptional, dScaleVOptional, dScaleDyOptional, dScaleOOptional,
                 queryRope, keyRope, sinkInOptional, dsScaleOptional, pScaleOptional),
        OP_OUTPUT(dqOut, dkOut, dvOut, dpseOut, dqRopeOut, dkRopeOut,dsinkOut),
        OP_ATTR(static_cast<float>(scaleValueOptional), static_cast<float>(keepProbOptional), preTockensOptional,
                nextTockensOptional, headNum, inputLayout, innerPreciseOptional, sparseModeOptional, pseTypeOptional,
                seed, offset, outDTypeOptional, softmaxInLayout));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Fag launch kernel failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }

    return {dqOut, dkOut, dvOut, dpseOut, dqRopeOut, dkRopeOut, dsinkOut};
}

} // namespace l0op
