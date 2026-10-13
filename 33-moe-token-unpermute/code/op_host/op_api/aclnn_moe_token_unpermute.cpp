/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_moe_token_unpermute.cpp
 * \brief
 */
#include "aclnn_moe_token_unpermute.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "../../common/include/common/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "../../common/include/external/aclnn_kernels/aclnn_platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

constexpr int64_t READ_INDEX_BY_ROW = 2;

#define CHECK_LOG_RET(cond, log_func, return_expr) \
    do {                                           \
        if (cond) {                                \
            log_func;                              \
        }                                          \
        return_expr;                               \
    } while (0)

extern aclnnStatus aclnnInnerMoeFinalizeRoutingV2GetWorkspaceSize(
    const aclTensor* expandedX, const aclTensor* expandedRowIdx, const aclTensor* x1Optional,
    const aclTensor* x2Optional, const aclTensor* biasOptional, const aclTensor* scalesOptional,
    const aclTensor* expertIdxOptional, int64_t dropPadMode, const aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeFinalizeRoutingV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
extern aclnnStatus aclnnInnerMoeTokenUnpermuteGetWorkspaceSize(
    const aclTensor* permutedTokens, const aclTensor* sortedIndices, const aclTensor* probsOptional, bool paddedMode,
    const aclIntArray* restoreShapeOptional, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeTokenUnpermute(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnMoeTokenUnpermuteGetWorkspaceSize(
    const aclTensor* permutedTokens, const aclTensor* sortedIndices, const aclTensor* probsOptional, bool paddedMode,
    const aclIntArray* restoreShapeOptional, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    static bool useMoeFinalizeRoutingV2 = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeFinalizeRoutingV2) {
        return aclnnInnerMoeTokenUnpermuteGetWorkspaceSize(
            permutedTokens, sortedIndices, probsOptional, paddedMode, restoreShapeOptional, out, workspaceSize,
            executor);
    }
    CHECK_RET(paddedMode == false, ACLNN_ERR_PARAM_INVALID);
    aclnnStatus ret = aclnnInnerMoeFinalizeRoutingV2GetWorkspaceSize(
        permutedTokens, sortedIndices, nullptr, nullptr, nullptr, probsOptional, nullptr, READ_INDEX_BY_ROW, out,
        workspaceSize, executor);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(
            ACLNN_ERR_INNER,
            "aclnnMoeTokeUnpermute calls alcnnMoeFinalizeRoutingV2, please refer to the document for parameter "
            "correspondence.");
    }
    return ret;
}

aclnnStatus aclnnMoeTokenUnpermute(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    static bool useMoeFinalizeRoutingV2 = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeFinalizeRoutingV2) {
        return aclnnInnerMoeTokenUnpermute(workspace, workspaceSize, executor, stream);
    }
    aclnnStatus ret = aclnnInnerMoeFinalizeRoutingV2(workspace, workspaceSize, executor, stream);
    CHECK_LOG_RET(
        ret != ACLNN_SUCCESS,
        OP_LOGE(
            ACLNN_ERR_INNER,
            "aclnnMoeTokeUnpermute calls alcnnMoeFinalizeRoutingV2, please refer to the document for parameter "
            "correspondence."),
        return ret);
}

#ifdef __cplusplus
}
#endif