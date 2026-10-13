/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_FLASH_ATTENTION_SCORE_H_
#define OP_API_INC_LEVEL2_ACLNN_FLASH_ATTENTION_SCORE_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFlashAttentionScore的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnFlashAttentionScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode, const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut, const aclTensor *attentionOutOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionScore的第二段接口，用于执行计算。
 */
aclnnStatus aclnnFlashAttentionScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream);

/**
 * @brief aclnnFlashAttentionVarLenScore的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnFlashAttentionVarLenScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, double scaleValue, double keepProb, int64_t preTokens,
    int64_t nextTokens, int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode,
    const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionVarLenScore的第二段接口，用于执行计算。
 */
aclnnStatus aclnnFlashAttentionVarLenScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           const aclrtStream stream);


/**
 * @brief aclnnFlashAttentionScoreV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnFlashAttentionScoreV2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional,
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
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionScoreV2的第二段接口，用于执行计算。
*/
aclnnStatus aclnnFlashAttentionScoreV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

/**
 * @brief aclnnFlashAttentionScoreV3的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnFlashAttentionScoreV3GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *sinkOptional,
    const aclIntArray *prefixOptional,
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
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionScoreV3的第二段接口，用于执行计算。
*/
aclnnStatus aclnnFlashAttentionScoreV3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);


/**
 * @brief aclnnFlashAttentionScoreV4的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnFlashAttentionScoreV4GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,
    const aclTensor *dScaleQOptional,
    const aclTensor *dScaleKOptional,
    const aclTensor *dScaleVOptional,
    const aclTensor *sinkOptional,
    const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, /*varlen only*/
    const aclIntArray *actualSeqKvLenOptional, /*varlen only*/
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
    int64_t outDtype,
    int64_t pseType,
    char *softmaxOutLayout,
    int64_t seed, /*dropout ADD*/
    int64_t offset, /*dropout ADD*/
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionScoreV4的第二段接口，用于执行计算。
 */
aclnnStatus aclnnFlashAttentionScoreV4(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);


/**
 * @brief aclnnQuantFlashAttentionScore的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnQuantFlashAttentionScoreGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *attenMaskOptional,
    const aclTensor *dScaleQ,
    const aclTensor *dScaleK,
    const aclTensor *dScaleV,
    const aclTensor *pScale,
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    int64_t headNum,
    char *inputLayout,
    int64_t sparseMode,
    aclTensor *softmaxMaxOut,
    aclTensor *softmaxSumOut,
    aclTensor *softmaxOutout,
    aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnQuantFlashAttentionScore的第二段接口，用于执行计算。
 */
aclnnStatus aclnnQuantFlashAttentionScore(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);


/**
 * @brief aclnnFlashAttentionVarLenScoreV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnFlashAttentionVarLenScoreV2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
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
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionVarLenScoreV2的第二段接口，用于执行计算。
*/
aclnnStatus aclnnFlashAttentionVarLenScoreV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

/**
 * @brief aclnnFlashAttentionVarLenScoreV3的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnFlashAttentionVarLenScoreV3GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *queryRope,
    const aclTensor *key,
    const aclTensor *keyRope,
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
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
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionVarLenScoreV3的第二段接口，用于执行计算。
*/
aclnnStatus aclnnFlashAttentionVarLenScoreV3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

/**
 * @brief aclnnFlashAttentionVarLenScoreV4的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnFlashAttentionVarLenScoreV4GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, double scaleValue, double keepProb, int64_t preTokens,
    int64_t nextTokens, int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode, char *softmaxOutLayout,
    const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionVarLenScoreV4的第二段接口，用于执行计算。
 */
aclnnStatus aclnnFlashAttentionVarLenScoreV4(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           const aclrtStream stream);

/**
 * @brief aclnnFlashAttentionVarLenScoreV5的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *queryRope,
    const aclTensor *key,
    const aclTensor *keyRope,
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *sinkOptional,
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
    char *softmaxOutLayout,
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionVarLenScoreV5的第二段接口，用于执行计算。
 */
aclnnStatus aclnnFlashAttentionVarLenScoreV5(
    void *workspace, 
    uint64_t workspaceSize, 
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_FLASH_ATTENTION_SCORE_H_
