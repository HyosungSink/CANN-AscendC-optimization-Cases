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

OP_TYPE_REGISTER(NsaSelectedAttention);

const std::array<const aclTensor *, 3> NsaSelectedAttention(const aclTensor *query, 
                                                            const aclTensor *key, 
                                                            const aclTensor *value, 
                                                            const aclTensor *topkIndices, 
                                                            const aclTensor *attenMaskOptional,
                                                            const aclIntArray *actualSeqQLenOptional, 
                                                            const aclIntArray *actualSeqKvLenOptional, 
                                                            double scaleValue, 
                                                            int64_t headNum, 
                                                            const char *inputLayout,
                                                            int64_t sparseMode, 
                                                            int64_t selectedBlockSize, 
                                                            int64_t selectedBlockCount,
                                                            aclOpExecutor *executor)
{
    L0_DFX(NsaSelectedAttention, query, key, value, topkIndices, attenMaskOptional,
            actualSeqQLenOptional, actualSeqKvLenOptional, scaleValue, headNum,
            inputLayout, sparseMode, selectedBlockSize, selectedBlockCount);

    if (attenMaskOptional == nullptr) {
        attenMaskOptional = executor->AllocTensor(DataType::DT_BOOL, Format::FORMAT_ND, Format::FORMAT_ND);
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

    auto softmaxMaxOut = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxSumOut = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    auto attentionOut = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(NsaSelectedAttention,
                           OP_INPUT(query, key, value, topkIndices, attenMaskOptional, 
                                    actualSeqQLen, actualSeqKvLen),
                           OP_OUTPUT(softmaxMaxOut, softmaxSumOut, attentionOut),
                           OP_ATTR(static_cast<float>(scaleValue), headNum, sparseMode, inputLayout,
                                    selectedBlockSize, selectedBlockCount));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "NsaSelectedAttention InferShape failed.");
        return {nullptr, nullptr, nullptr};
    }

    ADD_TO_LAUNCHER_LIST_AICORE(NsaSelectedAttention,
                                OP_INPUT(query, key, value, topkIndices, attenMaskOptional,  
                                        actualSeqQLen, actualSeqKvLen),
                                OP_OUTPUT(softmaxMaxOut, softmaxSumOut, attentionOut),
                                OP_ATTR(static_cast<float>(scaleValue), headNum, sparseMode, inputLayout,
                                    selectedBlockSize, selectedBlockCount));
    return {softmaxMaxOut, softmaxSumOut, attentionOut};
}

} // namespace l0op
