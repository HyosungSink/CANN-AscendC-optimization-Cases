/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_moe_distribute_combine_add_rms_norm.h"
#include <algorithm>
#include "../common/inc/op_mc2.h"
#include "../common/inc/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "../3rd/common/op_host/op_api/matmul_util.h"
#include "moe_distribute_combine_add_rms_norm_base.h"

using namespace Ops::Transformer;
using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize(
    const aclTensor* expandX, const aclTensor* expertIds,
    const aclTensor* assistInfoForCombine, const aclTensor* epSendCounts,
    const aclTensor* expertScales, const aclTensor* residualX, const aclTensor* gamma,
    const aclTensor* tpSendCountsOptional, const aclTensor* xActiveMaskOptional,
    const aclTensor* activationScaleOptional, const aclTensor* weightScaleOptional,
    const aclTensor* groupListOptional, const aclTensor* expandScalesOptional,
    const aclTensor* sharedExpertXOptional, const char* groupEp, int64_t epWorldSize,
    int64_t epRankId, int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype,
    int64_t commQuantMode, int64_t groupListType, const char* commAlg,
    float normEps, aclTensor* yOut, aclTensor* rstdOut, aclTensor* xOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("aclnn_moe_distribute_combine_Arn WorkspaceSize start");
    aclnnStatus ret = aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSizeBase(
        expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales, residualX, gamma, tpSendCountsOptional,
        xActiveMaskOptional, activationScaleOptional, weightScaleOptional, groupListOptional, expandScalesOptional,
        sharedExpertXOptional, nullptr, nullptr, nullptr, nullptr, nullptr, groupEp, epWorldSize, epRankId,
        moeExpertNum, groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum,
        globalBs, outDtype, commQuantMode, groupListType, commAlg, normEps, 0, 0, 0, yOut, rstdOut,
        xOut, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnMoeDistributeCombineAddRmsNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  aclrtStream stream)
{
    OP_LOGD("aclnn_moe_distribute_combine_Arn start");
    aclnnStatus ret = aclnnMoeDistributeCombineAddRmsNormBase(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif