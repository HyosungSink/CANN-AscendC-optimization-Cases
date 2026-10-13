/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_MLA_PROLOG
#define ACLNN_MLA_PROLOG
#warning "aclnn_mla_prolog.h is scheduled to be deprecated in December 2026, and will be replaced by the aclnn_mla_prolog_v3_weight_nz.h. We apologize for any inconvenience caused and appreciate your timely migration to the new interface. "
#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnMlaProlog calculates the workspace size based on the specific calculation process.
 * @domain aclnn_ops_infer
 */
__attribute__((deprecated("aclnnMlaPrologGetWorkspaceSize is scheduled to be deprecated in December 2026, and will be replaced by the aclnnMlaPrologV3WeightNzGetWorkspaceSize. "
 	                        "We apologize for any inconvenience caused and appreciate your timely migration to the new interface. ")))
__attribute__((visibility("default"))) aclnnStatus aclnnMlaPrologGetWorkspaceSize(
    const aclTensor *tokenX, const aclTensor *weightDq, const aclTensor *weightUqQr, const aclTensor *weightUk, const aclTensor *weightDkvKr,
    const aclTensor *rmsnormGammaCq, const aclTensor *rmsnormGammaCkv,
    const aclTensor *ropeSin, const aclTensor *ropeCos, const aclTensor *cacheIndex,
    aclTensor *kvCacheRef, aclTensor *krCacheRef, const aclTensor *dequantScaleXOptional,
    const aclTensor *dequantScaleWDqOptional, const aclTensor *dequantScaleWUqQrOptional,
    const aclTensor *dequantScaleWDkvKrOptional, const aclTensor *quantScaleCkvOptional,
    const aclTensor *quantScaleCkrOptional, const aclTensor *smoothScalesCqOptional,
    double rmsnormEpsilonCq, double rmsnormEpsilonCkv, char *cacheModeOptional,
    const aclTensor *queryOut, const aclTensor *queryRopeOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnMlaProlog is used to perform calculations.
 */
__attribute__((deprecated("aclnnMlaProlog is scheduled to be deprecated in December 2026, and will be replaced by the aclnnMlaPrologV3WeightNz. "
                          "We apologize for any inconvenience caused and appreciate your timely migration to the new interface. ")))
__attribute__((visibility("default"))) aclnnStatus aclnnMlaProlog(void *workspace,
                                                                            uint64_t workspaceSize,
                                                                            aclOpExecutor *executor,
                                                                            aclrtStream stream);


#ifdef __cplusplus
}
#endif

#endif // ACLNN_MLA_PROLOG