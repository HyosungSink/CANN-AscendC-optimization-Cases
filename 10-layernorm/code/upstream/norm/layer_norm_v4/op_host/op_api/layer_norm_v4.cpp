/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "layer_norm_v4.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(LayerNormV4);

const std::array<aclTensor*, LAYER_NORM_V4_OUT_NUM> LayerNormV4(
    const LayerNormV4Params& params, double eps, aclOpExecutor* executor)
{
    L0_DFX(LayerNormV4, params.input, params.normalizedShape, params.weight, params.bias, eps);

    // 根据array构造算子所需tensor输入
    auto normTensor = executor->ConvertToTensor(params.normalizedShape, DataType::DT_INT32);

    // 根据输入与normalizedShape关系构造输出shape
    Shape meanOutShape = params.input->GetViewShape();
    for (size_t index = meanOutShape.GetDimNum() - params.normalizedShape->Size(); index < meanOutShape.GetDimNum();
         index++) {
        meanOutShape[index] = 1;
    }
    auto output = executor->AllocTensor(params.input->GetViewShape(), params.input->GetDataType(), Format::FORMAT_ND);
    auto meanOut = executor->AllocTensor(meanOutShape, DataType::DT_FLOAT, Format::FORMAT_ND);
    auto rstdOut = executor->AllocTensor(meanOutShape, DataType::DT_FLOAT, Format::FORMAT_ND);

    ADD_TO_LAUNCHER_LIST_AICORE(
        LayerNormV4, OP_INPUT(params.input, normTensor, params.weight, params.bias),
        OP_OUTPUT(output, meanOut, rstdOut), OP_ATTR(static_cast<float>(eps)));

    return {output, meanOut, rstdOut};
}

} // namespace l0op
