/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "layer_norm_v3.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(LayerNormV3);

constexpr size_t MODE_V3_NUM = 3;

const std::array<aclTensor*, LAYER_NORM_V3_OUT_NUM> LayerNormV3(
    const LayerNormV3params& params, aclOpExecutor* executor)
{
    L0_DFX(LayerNormV3, params.input, params.weight, params.bias, params.beginAxis, params.eps);
    Shape meanOutShape = params.input->GetViewShape();
    for (size_t index = static_cast<size_t>(params.beginAxis); index < meanOutShape.GetDimNum(); index++) {
        meanOutShape[index] = 1;
    }
    auto output = executor->AllocTensor(params.input->GetViewShape(), params.input->GetDataType(), Format::FORMAT_ND);
    auto meanOut = executor->AllocTensor(meanOutShape, params.weight->GetDataType(), Format::FORMAT_ND);
    auto rstdOut = executor->AllocTensor(meanOutShape, params.bias->GetDataType(), Format::FORMAT_ND);

    ADD_TO_LAUNCHER_LIST_AICORE(
        LayerNormV3, OP_INPUT(params.input, params.weight, params.bias), OP_OUTPUT(output, meanOut, rstdOut),
        OP_ATTR(params.beginAxis, params.beginAxis, static_cast<float>(params.eps)));

    return {output, meanOut, rstdOut};
}

const std::array<aclTensor*, LAYER_NORM_V3_OUT_NUM> LayerNormV3WithImplMode(
    const LayerNormV3WithImplModeparams& params, aclOpExecutor* executor)
{
    L0_DFX(
        LayerNormV3WithImplMode, params.input, params.weight, params.bias, params.beginAxis, params.eps,
        params.implMode);
    Shape meanOutShape = params.input->GetViewShape();
    for (size_t index = static_cast<size_t>(params.beginAxis); index < meanOutShape.GetDimNum(); index++) {
        meanOutShape[index] = 1;
    }
    auto output = executor->AllocTensor(params.input->GetViewShape(), params.input->GetDataType(), Format::FORMAT_ND);
    auto meanOut = executor->AllocTensor(meanOutShape, params.weight->GetDataType(), Format::FORMAT_ND);
    auto rstdOut = executor->AllocTensor(meanOutShape, params.bias->GetDataType(), Format::FORMAT_ND);

    OpImplMode mode[MODE_V3_NUM] = {
        OpImplMode::IMPL_MODE_HIGH_PRECISION, OpImplMode::IMPL_MODE_HIGH_PERFORMANCE, OpImplMode::IMPL_MODE_KEEP_FP16};

    ADD_TO_LAUNCHER_LIST_AICORE(
        LayerNormV3, OP_INPUT(params.input, params.weight, params.bias), OP_OUTPUT(output, meanOut, rstdOut),
        OP_ATTR(params.beginAxis, params.beginAxis, static_cast<float>(params.eps)),
        OP_OPTION(mode[static_cast<size_t>(params.implMode)]));

    return {output, meanOut, rstdOut};
}

} // namespace l0op
