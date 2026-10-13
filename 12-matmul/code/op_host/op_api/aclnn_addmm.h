/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ADDMM_H_
#define OP_API_INC_ADDMM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAddmm的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnAddmmGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat1, const aclTensor* mat2, const aclScalar* beta, const aclScalar* alpha,
    aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceAddmm的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnInplaceAddmmGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* mat1, const aclTensor* mat2, const aclScalar* beta,
    const aclScalar* alpha, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAddmm的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus
aclnnAddmm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceAddmm的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus
aclnnInplaceAddmm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnAddmmWeightNz的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：相对于aclnnAddmm, mat2为NZ格式。
 * @param [in] self: bias矩阵，数据类型支持：float16, bfloat16，float32，format只支持ND。
 * @param [in] mat1: matmul左矩阵，数据类型支持：float16, bfloat16, format只支持ND。
 * @param [in] mat2: matmul右矩阵，数据类型支持：float16, bfloat16, 支持NZ格式。
 * @param [in] beta: self矩阵系数，数据类型支持：FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8。
 * @param [in] alpha: (mat1 @ mat2)矩阵计算结果系数，数据类型支持：FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8。
 * @param [in] cubeMathType: 用于指定Cube单元的计算逻辑，类型为Host侧的整型int8_t。
 * @param [out] out: 计算结果，数据类型：float16, bfloat16, float32。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 */
ACLNN_API aclnnStatus aclnnAddmmWeightNzGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat1, const aclTensor* mat2, const aclScalar* beta, const aclScalar* alpha,
    aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAddmmWeightNz的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnAddmmWeightNzGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus
aclnnAddmmWeightNz(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ADD_H_
