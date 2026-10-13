/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusedmatmul.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FusedMatMul);

static const aclTensor* FusedMatMulCommon(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3,
    const op::DataType output_dtype, const op::Format output_format, const op::Format output_ori_format,
    bool transposeX1, bool transposeX2, bool enableHf32, const char* fusedOpType, aclOpExecutor* executor)
{
    L0_DFX(FusedMatMulCommon, x1, x2, bias, x3, transposeX1, transposeX2, enableHf32, fusedOpType);
    auto mm_out = executor->AllocTensor(output_dtype, output_format, output_ori_format);
    auto ret = INFER_SHAPE(
        FusedMatMul, OP_INPUT(x1, x2, bias, x3), OP_OUTPUT(mm_out),
        OP_ATTR(transposeX1, transposeX2, enableHf32, fusedOpType));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
        return nullptr;
    }
    uint32_t execMode = enableHf32 ? static_cast<uint32_t>(OpExecMode::OP_EXEC_MODE_HF32) : 0U;
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        FusedMatMul, OP_INPUT(x1, x2, bias, x3), OP_OUTPUT(mm_out),
        OP_ATTR(transposeX1, transposeX2, enableHf32, fusedOpType), OP_MODE(execMode));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr, "Add to launcher list aicore failed.");
    return mm_out;
}

const aclTensor* FusedMatMulNd(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3, bool transposeX1,
    bool transposeX2, bool enableHf32, const char* fusedOpType, aclOpExecutor* executor)
{
    L0_DFX(FusedMatMulNd);
    return FusedMatMulCommon(
        x1, x2, bias, x3, x1->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND, transposeX1, transposeX2, enableHf32,
        fusedOpType, executor);
};
} // namespace l0op