/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ROTARY_POSITION_EMBEDDING_H_
#define OP_API_INC_ROTARY_POSITION_EMBEDDING_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRotaryPositionEmbedding的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,支持非连续的Tensor,数据格式支持ND。
 * @param [in] sin: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,数据类型与x一致,shape与x满足broadcast关系
 * 支持非连续的Tensor,数据格式支持ND。
 * @param [in] cos: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,数据类型与x一致,shape与sin一致
 * 支持非连续的Tensor,数据格式支持ND。
 * @param [in] mode: 支持int64_t类型,取值范围[0, 3],分别表示half,quarter,interleave,half-interleave四种模式.
 * @param [in] out: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,且数据类型与x一致,
 * shape与x相同, 数据格式支持ND, 且数据格式需要与x一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRotaryPositionEmbeddingGetWorkspaceSize(const aclTensor* x, const aclTensor* cos,
                                                                   const aclTensor* sin, int64_t mode, aclTensor* out,
                                                                   uint64_t* workspaceSize, aclOpExecutor** executor);
/* @brief aclnnRotaryPositionEmbedding的第二段接口，用于执行计算.
 * @param [in] workspace: 在npu device侧申请的workspace内存起址
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnBitwiseNotGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRotaryPositionEmbedding(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
