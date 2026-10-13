/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <tuple>
#include "qkv_rms_norm_rope_cache.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(QkvRmsNormRopeCache);

const std::tuple<aclTensor*, aclTensor*, aclTensor*> QkvRmsNormRopeCache(
    const aclTensor* qkv, const aclTensor* qGamma, const aclTensor* kGamma, const aclTensor* cos, const aclTensor* sin,
    const aclTensor* index, aclTensor* qOut, aclTensor* kCache, aclTensor* vCache, const aclTensor* kScaleOptional,
    const aclTensor* vScaleOptional, const aclTensor* kOffsetOptional, const aclTensor* vOffsetOptional,
    const aclIntArray* qkvSize, const aclIntArray* headNums, double epsilon, char* cacheModeOptional, bool isOutputQkv,
    aclOpExecutor* executor)
{
    L0_DFX(QkvRmsNormRopeCache, qkv, qGamma, kGamma, cos, sin, index, qOut,
           kCache, vCache, kScaleOptional, vScaleOptional, kOffsetOptional, vOffsetOptional, qkvSize,
           headNums, epsilon, cacheModeOptional, isOutputQkv);

    auto qOutBeforeQuant = executor->AllocTensor(qkv->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto kOutBeforeQuant = executor->AllocTensor(qkv->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto vOutBeforeQuant = executor->AllocTensor(qkv->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    
    // infershape
    auto ret = INFER_SHAPE(
        QkvRmsNormRopeCache,
        OP_INPUT(qkv, qGamma, kGamma, cos, sin, index, qOut, kCache, vCache, kScaleOptional, vScaleOptional, kOffsetOptional, vOffsetOptional),
        OP_OUTPUT(qOut, kCache, vCache, qOutBeforeQuant, kOutBeforeQuant, vOutBeforeQuant), 
        OP_ATTR(qkvSize, headNums, epsilon, cacheModeOptional, isOutputQkv));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QkvRmsNormRopeCache InferShape failed.");
        return {nullptr, nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        QkvRmsNormRopeCache,
        OP_INPUT(qkv, qGamma, kGamma, cos, sin, index, qOut, kCache, vCache, kScaleOptional, vScaleOptional, kOffsetOptional, vOffsetOptional),
        OP_OUTPUT(qOut, kCache, vCache, qOutBeforeQuant, kOutBeforeQuant, vOutBeforeQuant), 
        OP_ATTR(qkvSize, headNums, epsilon, cacheModeOptional, isOutputQkv));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QkvRmsNormRopeCache ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr};
    }
    if (isOutputQkv) {
        return std::tie(qOutBeforeQuant, kOutBeforeQuant, vOutBeforeQuant);
    }
    else {
        return {nullptr, nullptr, nullptr};
    }
}
} // namespace l0op


