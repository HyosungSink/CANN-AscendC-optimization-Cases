/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL0_ADD_RMS_NORM_DYNAMIC_QUANT_H_
#define OP_API_INC_LEVEL0_ADD_RMS_NORM_DYNAMIC_QUANT_H_

#include "opdev/op_executor.h"

namespace l0op {
constexpr size_t ADD_RMS_NORM_DYNAMIC_QUANT_OUT_NUM = 5;

const std::array<aclTensor*, ADD_RMS_NORM_DYNAMIC_QUANT_OUT_NUM> AddRmsNormDynamicQuant(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, const aclTensor* betaOptional, double epsilon, const aclBoolArray* outputMask,
    aclTensor* scale1Out, aclTensor* scale2Out, aclOpExecutor* executor);

const std::array<aclTensor*, ADD_RMS_NORM_DYNAMIC_QUANT_OUT_NUM> AddRmsNormDynamicQuant(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, const aclTensor* betaOptional, double epsilon, const aclBoolArray* outputMask,
    int dstType, aclTensor* scale1Out, aclTensor* scale2Out, aclOpExecutor* executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_ADD_RMS_NORM_QUANT_H_
