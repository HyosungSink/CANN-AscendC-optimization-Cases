/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_OP_API_COMMON_INC_LEVEL0_OP_GROUPED_MATMUL_FINALIZE_ROUTING_H
#define OP_API_OP_API_COMMON_INC_LEVEL0_OP_GROUPED_MATMUL_FINALIZE_ROUTING_H

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
const aclTensor* GroupedMatmulFinalizeRouting(const aclTensor *x1, const aclTensor *x2, const aclTensor *scale,
    const aclTensor *bias, const aclTensor *pertokenScale, const aclTensor *groupList, const aclTensor *shareInput, 
    const aclTensor *logit, const aclTensor *rowIndex, const aclTensor *offset, int64_t dtype, float sharedInputWeight, 
    int64_t sharedInputOffset, bool transposeX1, bool transposeX2, int64_t outputBS, int64_t groupListType,
    const aclIntArray* tuningConfig, aclOpExecutor *executor);

}

#endif // OP_API_OP_API_COMMON_INC_LEVEL0_OP_GROUPED_MATMUL_FINALIZE_ROUTING_H