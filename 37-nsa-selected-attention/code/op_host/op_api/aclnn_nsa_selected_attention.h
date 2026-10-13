/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_NSA_SELECTED_ATTENTION_H_
#define OP_API_INC_LEVEL2_ACLNN_NSA_SELECTED_ATTENTION_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnNsaSelectedAttention的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
aclnnStatus aclnnNsaSelectedAttentionGetWorkspaceSize(
    const aclTensor *query, 
    const aclTensor *key, 
    const aclTensor *value, 
    const aclTensor *topkIndices, 
    const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, 
    double scaleValue, 
    int64_t headNum, 
    char *inputLayout,
    int64_t sparseMode, 
    int64_t selectedBlockSize,
    int64_t selectedBlockCount, 
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut, 
    const aclTensor *attentionOut,
    uint64_t *workspaceSize, 
    aclOpExecutor **executor);

/**
 * @brief aclnnNsaSelectedAttention的第二段接口，用于执行计算。
 */
aclnnStatus aclnnNsaSelectedAttention(
    void *workspace, 
    uint64_t workspaceSize, 
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_NSA_SELECTED_ATTENTION_H_
