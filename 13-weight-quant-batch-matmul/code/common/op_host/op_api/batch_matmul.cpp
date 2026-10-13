/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "batch_matmul.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(BatchMatMulV2);
OP_TYPE_REGISTER(BatchMatMulV3);

const aclTensor* BatchMatMulV3Nd(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool adjX1,
    const bool adjX2, const bool offsetX, const bool enableHf32, aclOpExecutor* executor)
{
    L0_DFX(BatchMatMulV3Nd, x1, x2, bias, offsetW, adjX1, adjX2, offsetX, enableHf32);
    auto bmmOut = executor->AllocTensor(x1->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto ret = INFER_SHAPE(
        BatchMatMulV3, OP_INPUT(x1, x2, bias, nullptr), OP_OUTPUT(bmmOut), OP_ATTR(adjX1, adjX2, offsetX, enableHf32));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }
    // 0x40 : 表示使能HF32
    uint32_t execMode = enableHf32 ? static_cast<uint32_t>(OpExecMode::OP_EXEC_MODE_HF32) : 0U;
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        BatchMatMulV3, OP_INPUT(x1, x2, bias, nullptr), OP_OUTPUT(bmmOut), OP_ATTR(adjX1, adjX2, offsetX, enableHf32),
        OP_MODE(execMode));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr, "Add to launcher list aicore failed.");
    return bmmOut;
}

const aclTensor* BatchMatMulV3NdFp16Bf162Fp32(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool adjX1,
    const bool adjX2, const bool offsetX, const bool enableHf32, aclOpExecutor* executor)
{
    L0_DFX(BatchMatMulV3NdFp16Bf162Fp32, x1, x2, bias, offsetW, adjX1, adjX2, offsetX, enableHf32);
    auto bmmOut = executor->AllocTensor(op::DataType::DT_FLOAT, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto ret = INFER_SHAPE(
        BatchMatMulV3, OP_INPUT(x1, x2, bias, nullptr), OP_OUTPUT(bmmOut), OP_ATTR(adjX1, adjX2, offsetX, enableHf32));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }
    // 0x40 : 表示使能HF32
    uint32_t execMode = enableHf32 ? static_cast<uint32_t>(OpExecMode::OP_EXEC_MODE_HF32) : 0U;
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        BatchMatMulV3, OP_INPUT(x1, x2, bias, nullptr), OP_OUTPUT(bmmOut), OP_ATTR(adjX1, adjX2, offsetX, enableHf32),
        OP_MODE(execMode)); 
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr, "Add to launcher list aicore failed.");
    return bmmOut;
}

const aclTensor* BatchMatMulNd(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool adjX1,
    const bool adjX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor)
{
    L0_DFX(BatchMatMulNd, x1, x2, bias, offsetW, adjX1, adjX2, offsetX);
    auto bmmOut = executor->AllocTensor(x1->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto ret = INFER_SHAPE(
        BatchMatMulV2, OP_INPUT(x1, x2, bias, nullptr), OP_OUTPUT(bmmOut),
        OP_ATTR(adjX1, adjX2, offsetX, opImplModeEnum, 0L));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }
    // 0x40 : 表示使能HF32
    uint32_t execMode = opImplModeEnum == 0x40 ? static_cast<uint32_t>(OpExecMode::OP_EXEC_MODE_HF32) : 0U;
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        BatchMatMulV2, OP_INPUT(x1, x2, bias, nullptr), OP_OUTPUT(bmmOut),
        OP_ATTR(adjX1, adjX2, offsetX, opImplModeEnum, 0L), OP_MODE(execMode));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr, "Add to launcher list aicore failed.");
    return bmmOut;
}

const aclTensor* BatchMatMulNzFp162Fp16(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool adjX1,
    const bool adjX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor)
{
    L0_DFX(BatchMatMulNzFp162Fp16, x1, x2, bias, offsetW, adjX1, adjX2, offsetX);
    auto bmmOut = executor->AllocTensor(op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ, op::Format::FORMAT_ND);
    auto ret = INFER_SHAPE(
        BatchMatMulV2, OP_INPUT(x1, x2, bias, nullptr), OP_OUTPUT(bmmOut),
        OP_ATTR(adjX1, adjX2, offsetX, opImplModeEnum, 0L));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        BatchMatMulV2, OP_INPUT(x1, x2, bias, nullptr), OP_OUTPUT(bmmOut),
        OP_ATTR(adjX1, adjX2, offsetX, opImplModeEnum, 0L));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr, "Add to launcher list aicore failed.");
    return bmmOut;
}

const aclTensor* BatchMatMulNdFp162Fp32(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool adjX1,
    const bool adjX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor)
{
    L0_DFX(BatchMatMulNdFp162Fp32, x1, x2, bias, offsetW, adjX1, adjX2, offsetX);
    auto bmmOut = executor->AllocTensor(op::DataType::DT_FLOAT, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto ret = INFER_SHAPE(
        BatchMatMulV2, OP_INPUT(x1, x2, bias, nullptr), OP_OUTPUT(bmmOut),
        OP_ATTR(adjX1, adjX2, offsetX, opImplModeEnum, 0L));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        BatchMatMulV2, OP_INPUT(x1, x2, bias, nullptr), OP_OUTPUT(bmmOut),
        OP_ATTR(adjX1, adjX2, offsetX, opImplModeEnum, 0L));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr, "Add to launcher list aicore failed.");
    return bmmOut;
}

const aclTensor* BatchMatMulNzFp162Fp32(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool adjX1,
    const bool adjX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor)
{
    L0_DFX(BatchMatMulNzFp162Fp32, x1, x2, bias, offsetW, adjX1, adjX2, offsetX);
    auto bmmOut = executor->AllocTensor(op::DataType::DT_FLOAT, op::Format::FORMAT_FRACTAL_NZ, op::Format::FORMAT_ND);
    auto ret = INFER_SHAPE(
        BatchMatMulV2, OP_INPUT(x1, x2, bias, nullptr), OP_OUTPUT(bmmOut),
        OP_ATTR(adjX1, adjX2, offsetX, opImplModeEnum, 0L));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        BatchMatMulV2, OP_INPUT(x1, x2, bias, nullptr), OP_OUTPUT(bmmOut),
        OP_ATTR(adjX1, adjX2, offsetX, opImplModeEnum, 0L));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr, "Add to launcher list aicore failed.");
    return bmmOut;
}
} // namespace l0op
