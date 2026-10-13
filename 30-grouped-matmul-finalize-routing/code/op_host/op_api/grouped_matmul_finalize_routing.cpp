/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_finalize_routing.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GroupedMatmulFinalizeRouting);

const aclTensor *GroupedMatmulFinalizeRouting(const aclTensor *x1, const aclTensor *x2, const aclTensor *scale,
    const aclTensor *bias, const aclTensor *pertokenScale, const aclTensor *groupList, const aclTensor *shareInput, 
    const aclTensor *logit, const aclTensor *rowIndex, const aclTensor *offset, int64_t dtype, float sharedInputWeight,
    int64_t sharedInputOffset, bool transposeX1, bool transposeX2, int64_t outputBS, int64_t groupListType, const aclIntArray* tuningConfig,
    aclOpExecutor *executor)
{
    L0_DFX(GroupedMatmulFinalizeRouting, x1, x2, scale, bias, pertokenScale, groupList, shareInput, logit, rowIndex,
        dtype, sharedInputWeight, sharedInputOffset, transposeX1, transposeX2, outputBS, groupListType);
    DataType outType = DataType::DT_FLOAT;

    Format format = Format::FORMAT_ND;
    auto output = executor->AllocTensor(outType, format, format);

    // 适配原型接口 offset//作为grouplist
    auto ret = INFER_SHAPE(GroupedMatmulFinalizeRouting,
        OP_INPUT(x1, x2, scale, bias, pertokenScale, groupList, shareInput, logit, rowIndex, offset), OP_OUTPUT(output),
        OP_ATTR(dtype, sharedInputWeight, sharedInputOffset, transposeX1, transposeX2, outputBS, groupListType, tuningConfig));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "GroupedMatmulFinalizeRouting InferShape failed.");
    ret = ADD_TO_LAUNCHER_LIST_AICORE(GroupedMatmulFinalizeRouting,
        OP_INPUT(x1, x2, scale, bias, pertokenScale, groupList, shareInput, logit, rowIndex, offset), OP_OUTPUT(output),
        OP_ATTR(dtype, sharedInputWeight, sharedInputOffset, transposeX1, transposeX2, outputBS, groupListType, tuningConfig));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
        "GroupedMatmulFinalizeRouting ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return output;
}
}