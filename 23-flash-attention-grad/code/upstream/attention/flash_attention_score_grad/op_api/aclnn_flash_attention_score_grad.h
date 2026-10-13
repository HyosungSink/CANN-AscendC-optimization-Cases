/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_FLASH_ATTENTION_SCORE_GRAD_H_
#define OP_API_INC_FLASH_ATTENTION_SCORE_GRAD_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFlashAttentionScoreGrad的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnFlashAttentionScoreGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionScoreGrad的第二段接口，用于执行计算。
 */
aclnnStatus aclnnFlashAttentionScoreGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         const aclrtStream stream);

/**
 * @brief aclnnFlashAttentionUnpaddingScoreGrad的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum,
    char *inputLayout, int64_t innerPrecise, int64_t sparseMode, const aclTensor *dqOut,
    const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionUnpaddingScoreGrad的第二段接口，用于执行计算。
 */
aclnnStatus aclnnFlashAttentionUnpaddingScoreGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  const aclrtStream stream);


/**
 * @brief aclnnFlashAttentionScoreGradV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnFlashAttentionScoreGradV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode,
    int64_t pseType, const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dpseOut, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionScoreGradV2的第二段接口，用于执行计算。
*/
aclnnStatus aclnnFlashAttentionScoreGradV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           const aclrtStream stream);

                                           /**
 * @brief aclnnFlashAttentionScoreGradV3的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnFlashAttentionScoreGradV3GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, 
    const aclTensor *sinkInOptional,
    const aclIntArray *prefixOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode,
    int64_t pseType, const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dpseOut, 
    const aclTensor *dsinkOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionScoreGradV3的第二段接口，用于执行计算。
*/
aclnnStatus aclnnFlashAttentionScoreGradV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           const aclrtStream stream);
/**
 * @brief aclnnFlashAttentionScoreGradV4的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnFlashAttentionScoreGradV4GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional,
    const aclTensor *sinkInOptional, const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional, const aclTensor *dScaleQOptional,
    const aclTensor *dScaleKOptional, const aclTensor *dScaleVOptional, const aclTensor *dScaleDyOptional,
    const aclTensor *dScaleOOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum,
    char *inputLayout, char *softmaxInLayout, int64_t innerPrecise, int64_t sparseMode, int64_t pseType,
    int64_t seed, int64_t offset, int64_t outDtype,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dqRopeOut, const aclTensor *dkRopeOut, const aclTensor *dpseOut, const aclTensor *dsinkOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);
 
/**
 * @brief aclnnFlashAttentionScoreGradV4的第二段接口，用于执行计算。
*/
aclnnStatus aclnnFlashAttentionScoreGradV4(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           aclrtStream stream);
/**
 * @brief aclnnFlashAttentionUnpaddingScoreGradV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum,
    char *inputLayout, int64_t innerPrecise, int64_t sparseMode, int64_t pseType,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionUnpaddingScoreGradV2的第二段接口，用于执行计算。
*/
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    const aclrtStream stream);
  
 /**
  * @brief aclnnFlashAttentionUnpaddingScoreGradV3的第一段接口，根据具体的计算流程，计算workspace大小。
  * @domain aclnn_ops_train
 */
 aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV3GetWorkspaceSize(
     const aclTensor *query, const aclTensor *queryRope, const aclTensor *keyIn, const aclTensor *keyInRope, 
     const aclTensor *value, const aclTensor *dy,
     const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
     const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
     const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
     const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional,
     const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValue,
     double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum,
     char *inputLayout, int64_t innerPrecise, int64_t sparseMode, int64_t pseType,
     const aclTensor *dqOut, const aclTensor *dqRopeOut, const aclTensor *dkOut, const aclTensor *dkRopeOut, 
     const aclTensor *dvOut, const aclTensor *dpseOut,
     uint64_t *workspaceSize, aclOpExecutor **executor);
 
 /**
  * @brief aclnnFlashAttentionUnpaddingScoreGradV3的第二段接口，用于执行计算。
 */
 aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                     const aclrtStream stream);
                                                     
 /**
 * @brief aclnnFlashAttentionUnpaddingScoreGradV4的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV4GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum,
    char *inputLayout, int64_t innerPrecise, int64_t sparseMode, const aclTensor *dqOut,
    const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut, char *softmaxInLayout,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionUnpaddingScoreGradV4的第二段接口，用于执行计算。
 */
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV4(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  const aclrtStream stream);

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize(
    const aclTensor *query, 
    const aclTensor *queryRope, 
    const aclTensor *keyIn, 
    const aclTensor *keyInRope, 
    const aclTensor *value, 
    const aclTensor *dy,
    const aclTensor *pseShiftOptional, 
    const aclTensor *dropMaskOptional, 
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, 
    const aclTensor *softmaxMaxOptional, 
    const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, 
    const aclTensor *attentionInOptional, 
    const aclTensor *sinkInOptional, 
    const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, 
    const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, 
    const aclIntArray *kvStartIdxOptional, 
    double scaleValue,
    double keepProb, 
    int64_t preTokens, 
    int64_t nextTokens, 
    int64_t headNum,
    char *inputLayout, 
    int64_t innerPrecise, 
    int64_t sparseMode, 
    int64_t pseType,
    char *softmaxInLayout, 
    const aclTensor *dqOut, 
    const aclTensor *dqRopeOut, 
    const aclTensor *dkOut, 
    const aclTensor *dkRopeOut, 
    const aclTensor *dvOut, 
    const aclTensor *dpseOut,
    const aclTensor *dsinkOut, 
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV5(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    const aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_FLASH_ATTENTION_SCORE_GRAD_H_

