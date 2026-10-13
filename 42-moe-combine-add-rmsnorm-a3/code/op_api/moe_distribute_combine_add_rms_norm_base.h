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
 * \file moe_distribute_combine_add_rms_norm_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM_BASE_
#define MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM_BASE_

#include "aclnn/aclnn_base.h"
#include "../common/inc/aclnn_util.h"
#include "../3rd/common/op_host/op_api/matmul_util.h"

#ifdef __cplusplus
extern "C" {
#endif
enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

// check nullptr
ACLNN_API bool CombineArnCheckNotNull(const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* assistInfoForCombine,
    const aclTensor* epSendCounts, [[maybe_unused]] const aclTensor* tpSendCounts, const aclTensor* expertScales,
    const char* groupEp, [[maybe_unused]] const char* groupTp, aclTensor* x);

// 入参校验
ACLNN_API aclnnStatus CombineArnCheckParams(const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* expandIdx,
    const aclTensor* epSendCounts, const aclTensor* tpSendCounts,
    const aclTensor* expertScales, const char* groupEp, const char* groupTp,
    aclTensor* x, bool is910B);

ACLNN_API aclnnStatus aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSizeBase(
    const aclTensor* expandX, const aclTensor* expertIds,
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
    aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnMoeDistributeCombineAddRmsNormBase(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif //MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM_BASE_