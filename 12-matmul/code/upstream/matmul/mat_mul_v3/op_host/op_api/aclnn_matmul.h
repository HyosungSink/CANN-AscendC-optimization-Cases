
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MATMUL_H_
#define OP_API_INC_MATMUL_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMatmul的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnMatmulGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnMatmul的第二段接口，用于执行计算。
 */
// for any shape mat multiply
ACLNN_API aclnnStatus aclnnMatmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnMatmulWeightNz的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：相对于aclnnMatmul, mat2为NZ格式。
 * @param [in] self: matmul左矩阵，数据类型支持：float16, bfloat16, format只支持ND。
 * @param [in] mat2: matmul右矩阵，数据类型支持：float16, bfloat16, 支持NZ格式。
 * @param [in] cubeMathType: 用于指定Cube单元的计算逻辑，类型为Host侧的整型int8_t。
 * @param [out] out: 计算结果，数据类型：float16, bfloat16。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 */
ACLNN_API aclnnStatus aclnnMatmulWeightNzGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnMatmulWeightNz的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnMatmulWeightNzGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
// for any shape mat multiply
ACLNN_API aclnnStatus
aclnnMatmulWeightNz(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MATMUL_H_
