/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MatMulV2);
OP_TYPE_REGISTER(MatMulV3);
OP_TYPE_REGISTER(GemmV3);
namespace {

static const aclTensor* MatMulV3Common(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const op::DataType output_dtype,
    const op::Format output_format, const op::Format output_ori_format, const bool transposeX1, const bool transposeX2,
    const bool offsetX, const int64_t opImplMode, aclOpExecutor* executor)
{
    const aclTensor* offsetW = nullptr;
    L0_DFX(MatMulV3Common, x1, x2, transposeX1, transposeX2, offsetX, opImplMode);
    auto mm_out = executor->AllocTensor(output_dtype, output_format, output_ori_format);
    auto ret = INFER_SHAPE(
        MatMulV3, OP_INPUT(x1, x2, bias, offsetW), OP_OUTPUT(mm_out),
        OP_ATTR(transposeX1, transposeX2, offsetX, opImplMode));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
        return nullptr;
    }
    uint32_t execMode = opImplMode == 0x40 ? static_cast<uint32_t>(OpExecMode::OP_EXEC_MODE_HF32) : 0U;
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        MatMulV3, OP_INPUT(x1, x2, bias, offsetW), OP_OUTPUT(mm_out),
        OP_ATTR(transposeX1, transposeX2, offsetX, opImplMode), OP_MODE(execMode));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr, "Add to launcher list aicore failed.");
    return mm_out;
}

static const aclTensor* MatMulCommon(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW,
    const op::DataType output_dtype, const op::Format output_format, const op::Format output_ori_format,
    const bool transposeX1, const bool transposeX2, const bool offsetX, const int64_t opImplModeEnum,
    aclOpExecutor* executor)
{
    L0_DFX(MatMulCommon, x1, x2, transposeX1, transposeX2, offsetX);
    auto mm_out = executor->AllocTensor(output_dtype, output_format, output_ori_format);
    // 当前暂时不支持input size和hiddenSize两个参数的设置
    auto ret = INFER_SHAPE(
        MatMulV2, OP_INPUT(x1, x2, bias, offsetW), OP_OUTPUT(mm_out),
        OP_ATTR(transposeX1, transposeX2, offsetX, -1L, -1L, opImplModeEnum, 1L));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
        return nullptr;
    }
    uint32_t execMode = opImplModeEnum == 0x40 ? static_cast<uint32_t>(OpExecMode::OP_EXEC_MODE_HF32) : 0U;
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        MatMulV2, OP_INPUT(x1, x2, bias, offsetW), OP_OUTPUT(mm_out),
        OP_ATTR(transposeX1, transposeX2, offsetX, -1L, -1L, opImplModeEnum, 1L), OP_MODE(execMode));

    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr, "Add to launcher list aicore failed.");
    return mm_out;
}
} // namespace
const aclTensor* MatMulNd(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool transposeX1,
    const bool transposeX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor)
{
    L0_DFX(MatMulNd);
    return MatMulCommon(
        x1, x2, bias, offsetW, x1->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND, transposeX1, transposeX2,
        offsetX, opImplModeEnum, executor);
};

const aclTensor* MatMulNdFp162Fp32(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool transposeX1,
    const bool transposeX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor)
{
    L0_DFX(MatMulNdFp162Fp32);
    return MatMulCommon(
        x1, x2, bias, offsetW, DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND, transposeX1, transposeX2,
        offsetX, opImplModeEnum, executor);
};

const aclTensor* MatMulNz(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool transposeX1,
    const bool transposeX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor)
{
    L0_DFX(MatMulNz);
    return MatMulCommon(
        x1, x2, bias, offsetW, x1->GetDataType(), Format::FORMAT_FRACTAL_NZ, Format::FORMAT_ND, transposeX1,
        transposeX2, offsetX, opImplModeEnum, executor);
};

const aclTensor* MatMulNzNzNd(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool transposeX1,
    const bool transposeX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor)
{
    L0_DFX(MatMulNzNzNd);
    return MatMulCommon(
        x1, x2, bias, offsetW, x1->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND, transposeX1, transposeX2,
        offsetX, opImplModeEnum, executor);
};

const aclTensor* MatMulNzFp162Fp32(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool transposeX1,
    const bool transposeX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor)
{
    L0_DFX(MatMulNzFp162Fp32);
    return MatMulCommon(
        x1, x2, bias, offsetW, DataType::DT_FLOAT, Format::FORMAT_FRACTAL_NZ, Format::FORMAT_ND, transposeX1,
        transposeX2, offsetX, opImplModeEnum, executor);
};

const aclTensor* MatMulNdNz(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool transposeX1,
    const bool transposeX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor)
{
    L0_DFX(MatMulNdNz);
    return MatMulCommon(
        x1, x2, bias, offsetW, x1->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND, transposeX1, transposeX2,
        offsetX, opImplModeEnum, executor);
};

const aclTensor* MatMulV3Nd(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const bool transposeX1, const bool transposeX2,
    const bool offsetX, const int64_t opImplMode, aclOpExecutor* executor)
{
    L0_DFX(MatMulV3Nd);
    return MatMulV3Common(
        x1, x2, bias, x1->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND, transposeX1, transposeX2, offsetX,
        opImplMode, executor);
};

const aclTensor* MatMulV3NdFp162Fp32(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const bool transposeX1, const bool transposeX2,
    const bool offsetX, const int64_t opImplMode, aclOpExecutor* executor)
{
    L0_DFX(MatMulV3NdFp162Fp32);
    return MatMulV3Common(
        x1, x2, bias, DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND, transposeX1, transposeX2, offsetX,
        opImplMode, executor);
};

// 输入self=NZ，输入mat2=NZ, 输出ND
const aclTensor* MatMulV3NzNzNd(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const bool transposeX1, const bool transposeX2,
    const bool offsetX, const int64_t opImplMode, aclOpExecutor* executor)
{
    L0_DFX(MatMulV3NzNzNd);
    return MatMulV3Common(
        x1, x2, bias, x1->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND, transposeX1, transposeX2, offsetX,
        opImplMode, executor);
};

// 输入self=NZ fp16，输入mat2=NZ fp16, 输出ND fp32
const aclTensor* MatMulV3NzNzNdFp162Fp32(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const bool transposeX1, const bool transposeX2,
    const bool offsetX, const int64_t opImplMode, aclOpExecutor* executor)
{
    L0_DFX(MatMulV3NzNzNdFp162Fp32);
    return MatMulV3Common(
        x1, x2, bias, DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND, transposeX1, transposeX2, offsetX,
        opImplMode, executor);
};

const aclTensor* GemmV3Nd(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* c, bool transposeX1, bool transposeX2, bool enableHf32,
    aclOpExecutor* executor)
{
    L0_DFX(GemmV3Nd, x1, x2, c, transposeX1, transposeX2, enableHf32);
    // output of gemmv3 reuse c tensor
    auto gemmOut = const_cast<aclTensor*>(c);
    uint32_t execMode = enableHf32 ? static_cast<uint32_t>(OpExecMode::OP_EXEC_MODE_HF32) : 0U;
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        GemmV3, OP_INPUT(x1, x2, c), OP_OUTPUT(gemmOut), OP_ATTR(1.0f, 1.0f, transposeX1, transposeX2, enableHf32),
        OP_MODE(execMode));

    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr, "Add to launcher list aicore failed.");
    return gemmOut;
};

const aclTensor* GemmV3NdWithAlphaBeta(const aclTensor* x1,
                                       const aclTensor* x2,
                                       const aclTensor* self,
                                       float alpha,
                                       float beta,
                                       bool transposeX1,
                                       bool transposeX2,
                                       bool enableHf32,
                                       aclOpExecutor* executor)
{
    L0_DFX(GemmV3NdWithAlphaBeta, x1, x2, self, alpha, beta, transposeX1, transposeX2, enableHf32);
    aclTensor* out = executor->AllocTensor(self->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto ret = INFER_SHAPE(
        GemmV3, OP_INPUT(x1, x2, self), OP_OUTPUT(out), OP_ATTR(alpha, beta, transposeX1, transposeX2, enableHf32));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed");
        return nullptr; 
    }
    uint32_t execMode = enableHf32 ? static_cast<uint32_t>(OpExecMode::OP_EXEC_MODE_HF32) : 0U;
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        GemmV3, OP_INPUT(x1, x2, self), OP_OUTPUT(out), OP_ATTR(alpha, beta, transposeX1, transposeX2, enableHf32));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr, "Add to launcher list aicore failed.");
    return out;
}

} // namespace l0op
