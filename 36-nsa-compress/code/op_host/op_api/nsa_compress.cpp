/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nsa_compress.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(NsaCompress);

const aclTensor *NsaCompress(const aclTensor *input, const aclTensor *weight, const aclIntArray *actSeqLenOptional,
                             char *layoutOptional, int64_t compressBlockSize, int64_t compressStride,
                             int64_t actSeqLenType, aclOpExecutor *executor)
{
    // L0接口时延统计以及入参打印
    L0_DFX(NsaCompress, input, weight, actSeqLenOptional, layoutOptional, compressBlockSize, compressStride,
           actSeqLenType);

    // 处理aclIntArray，转换成aclTensor
    const aclTensor *actSeqLen = nullptr;
    if (actSeqLenOptional) {
        actSeqLen = executor->ConvertToTensor(actSeqLenOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actSeqLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actSeqLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actSeqLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actSeqLen = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    // 构造输出
    auto compress_out = executor->AllocTensor(input->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);

    // 调用inferShape
    auto ret = INFER_SHAPE(NsaCompress, OP_INPUT(input, weight, actSeqLen), OP_OUTPUT(compress_out),
                           OP_ATTR(layoutOptional, compressBlockSize, compressStride, actSeqLenType));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }

    // 发起aicore任务
    ret = ADD_TO_LAUNCHER_LIST_AICORE(NsaCompress, OP_INPUT(input, weight, actSeqLen), OP_OUTPUT(compress_out),
                                      OP_ATTR(layoutOptional, compressBlockSize, compressStride, actSeqLenType));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return compress_out;
}
} // namespace l0op
