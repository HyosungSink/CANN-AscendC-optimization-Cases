/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_distribute_combine_add_rms_norm_base.cpp
 * \brief
 */

#include <algorithm>
#include "../common/inc/op_mc2.h"
#include "../common/inc/op_mc2_def.h"
#include "../3rd/ops_legacy/op_api/aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnn_moe_distribute_combine_add_rms_norm_v2.h"
#include "moe_distribute_combine_add_rms_norm_base.h"
#include "../3rd/common/op_host/op_api/matmul_util.h"

using namespace Ops::Transformer;
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
extern aclnnStatus aclnnInnerMoeDistributeCombineAddRmsNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                            aclrtStream stream);

extern aclnnStatus aclnnInnerMoeDistributeCombineAddRmsNormGetWorkspaceSize(
    const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* assistInfoForCombine, 
    const aclTensor* epSendCounts, const aclTensor* expertScales, const aclTensor* residualX, const aclTensor* gamma,
    const aclTensor* tpSendCounts, const aclTensor* xActiveMask, const aclTensor* activationScale,
    const aclTensor* weightScale, const aclTensor* groupList, const aclTensor* expandScales, 
    const aclTensor* sharedExpertX, const aclTensor* elasticInfo, const aclTensor* oriX,
    const aclTensor* constExpertAlpha1, const aclTensor* constExpertAlpha2, const aclTensor* constExpertV,
    const char* groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, const char* groupTp,
    int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t globalBs, int64_t outDtype, int64_t commQuantMode, int64_t groupListType, const char* commAlg, float normEps,
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum, aclTensor* yOut, aclTensor* rstdOut,
    aclTensor* xOut, uint64_t* workspaceSize, aclOpExecutor** executor);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// check nullptr
bool CombineArnCheckNotNull(const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* assistInfoForCombine,
    const aclTensor* epSendCounts, [[maybe_unused]] const aclTensor* tpSendCounts, const aclTensor* expertScales,
    const char* groupEp, [[maybe_unused]] const char* groupTp, aclTensor* x)
{
    OP_CHECK_NULL(expandX, return false);
    OP_CHECK_NULL(expertIds, return false);
    OP_CHECK_NULL(assistInfoForCombine, return false);
    OP_CHECK_NULL(epSendCounts, return false);
    OP_CHECK_NULL(expertScales, return false);
    OP_CHECK_NULL(x, return false);
    if ((groupEp == nullptr) || (strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required groupEp name is Empty.");
        return false;
    }
    return true;
}

// 入参校验
aclnnStatus CombineArnCheckParams(const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* expandIdx,
                            const aclTensor* epSendCounts, const aclTensor* tpSendCounts,
                            const aclTensor* expertScales, const char* groupEp, const char* groupTp,
                            aclTensor* x, bool is910B)
{
    CHECK_RET(CombineArnCheckNotNull(expandX, expertIds, expandIdx, epSendCounts, tpSendCounts, expertScales, groupEp, 
                                     groupTp, x), 
              ACLNN_ERR_PARAM_NULLPTR);

    if (is910B) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CombineArn Not Support 910B platform.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CombineArn Required groupEp Name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strnlen(groupTp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CombineArn Required groupTp Name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSizeBase(const aclTensor* expandX, const aclTensor* expertIds,
    const aclTensor* assistInfoForCombine, const aclTensor* epSendCounts,
    const aclTensor* expertScales, const aclTensor* residualX,
    const aclTensor* gamma, const aclTensor* tpSendCountsOptional,
    const aclTensor* xActiveMaskOptional, const aclTensor* activationScaleOptional,
    const aclTensor* weightScaleOptional, const aclTensor* groupListOptional,
    const aclTensor* expandScalesOptional,  const aclTensor* sharedExpertXOptional,
    const aclTensor* elasticInfoOptional, const aclTensor* oriXOptional, 
    const aclTensor* constExpertAlpha1Optional, const aclTensor* constExpertAlpha2Optional, 
    const aclTensor* constExpertVOptional, const char* groupEp, int64_t epWorldSize, 
    int64_t epRankId, int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize, 
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, 
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype,
    int64_t commQuantMode, int64_t groupListType, const char* commAlg, float normEps,
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum,
    aclTensor* yOut, aclTensor* rstdOut, aclTensor* xOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    const static bool is910B = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B;
    auto retParam = CombineArnCheckParams(expandX, expertIds, assistInfoForCombine, epSendCounts, tpSendCountsOptional,
        expertScales, groupEp, groupTp, xOut, is910B);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    aclnnStatus ret;
    if (is910B) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Not support 910B platform.");
        return ACLNN_ERR_PARAM_INVALID;
    } else {
        ret = aclnnInnerMoeDistributeCombineAddRmsNormGetWorkspaceSize(
            expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales, residualX, gamma,
            tpSendCountsOptional, xActiveMaskOptional, activationScaleOptional, weightScaleOptional,
            groupListOptional, expandScalesOptional, sharedExpertXOptional, elasticInfoOptional, oriXOptional,
            constExpertAlpha1Optional, constExpertAlpha2Optional, constExpertVOptional, groupEp, epWorldSize,
            epRankId, moeExpertNum, groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
            sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType, commAlg, normEps,
            zeroExpertNum, copyExpertNum, constExpertNum, yOut, rstdOut, xOut, workspaceSize, executor);
    }
    return ret;
}

aclnnStatus aclnnMoeDistributeCombineAddRmsNormBase(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                aclrtStream stream) {
    aclnnStatus ret = 0;
    const static bool is910B = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B;
    if (is910B) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Not support 910B platform.");
        return ACLNN_ERR_PARAM_INVALID;
    } else {
        if (NnopbaseSetHcclServerType) {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
        ret = aclnnInnerMoeDistributeCombineAddRmsNorm(workspace, workspaceSize, executor, stream);
    }
    return ret;                                            
}
#ifdef __cplusplus
}
#endif
