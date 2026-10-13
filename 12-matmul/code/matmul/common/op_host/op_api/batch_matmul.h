/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_BATCHMATMUL_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_BATCHMATMUL_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* BatchMatMulV3Nd(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool adjX1,
    const bool adjX2, const bool offsetX, const bool enableHf32, aclOpExecutor* executor);

const aclTensor* BatchMatMulV3NdFp16Bf162Fp32(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool adjX1,
    const bool adjX2, const bool offsetX, const bool enableHf32, aclOpExecutor* executor);

const aclTensor* BatchMatMulNd(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool adjX1,
    const bool adjX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor);

const aclTensor* BatchMatMulNzFp162Fp16(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool adjX1,
    const bool adjX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor);

const aclTensor* BatchMatMulNdFp162Fp32(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool adjX1,
    const bool adjX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor);

const aclTensor* BatchMatMulNzFp162Fp32(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW, const bool adjX1,
    const bool adjX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor);
} // namespace l0op

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_BATCHMATMUL_OP_H_
