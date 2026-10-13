/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_moe_token_permute.h"
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

#define CHECK_LOG_RET(cond, log_func, return_expr) \
    do {                                           \
        if (cond) {                                \
            log_func;                              \
        }                                          \
        return_expr;                               \
    } while (0)

extern aclnnStatus aclnnInnerMoeInitRoutingV2GetWorkspaceSize(
    const aclTensor* x, const aclTensor* expertIdx, int64_t activeNumOptional, int64_t expertCapacityOptional,
    int64_t expertNumOptional, int64_t dropPadModeOptional, int64_t expertTokensCountOrCumsumFlagOptional,
    bool expertTokensBeforeCapacityFlagOptional, const aclTensor* expandedXOut, const aclTensor* expandedRowIdxOut,
    const aclTensor* expertTokensCountOrCumsumOutOptional, const aclTensor* expertTokensBeforeCapacityOutOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor);

extern aclnnStatus aclnnInnerMoeInitRoutingV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

extern aclnnStatus aclnnInnerMoeTokenPermuteGetWorkspaceSize(
    const aclTensor* tokens, const aclTensor* indices, int64_t numOutTokens, bool paddedMode,
    const aclTensor* permuteTokensOut, const aclTensor* sortedIndicesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeTokenPermute(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnMoeTokenPermuteGetWorkspaceSize(
    const aclTensor* tokens, const aclTensor* indices, int64_t numOutTokens, bool paddedMode,
    const aclTensor* permuteTokensOut, const aclTensor* sortedIndicesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    static bool useMoeInitRoutingV2 = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeInitRoutingV2) {
        return aclnnInnerMoeTokenPermuteGetWorkspaceSize(
            tokens, indices, numOutTokens, paddedMode, permuteTokensOut, sortedIndicesOut, workspaceSize, executor);
    }
    CHECK_RET(paddedMode == false, ACLNN_ERR_PARAM_INVALID);
    aclnnStatus ret = aclnnInnerMoeInitRoutingV2GetWorkspaceSize(
        tokens, indices, numOutTokens, 0, 0, 0, 0, false, permuteTokensOut, sortedIndicesOut, sortedIndicesOut,
        sortedIndicesOut, workspaceSize, executor);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(
            ACLNN_ERR_INNER,
            "aclnnMoeTokePermute calls alcnnMoeInitRoutingV2, please refer to the document for parameter "
            "correspondence.");
    }
    return ret;
}

aclnnStatus aclnnMoeTokenPermute(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    static bool useMoeInitRoutingV2 = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeInitRoutingV2) {
        return aclnnInnerMoeTokenPermute(workspace, workspaceSize, executor, stream);
    }
    aclnnStatus ret = aclnnInnerMoeInitRoutingV2(workspace, workspaceSize, executor, stream);
    CHECK_LOG_RET(
        ret != ACLNN_SUCCESS,
        OP_LOGE(
            ACLNN_ERR_INNER,
            "aclnnMoeTokePermute calls alcnnMoeInitRoutingV2, please refer to the document for parameter "
            "correspondence."),
        return ret);
}

#ifdef __cplusplus
}
#endif