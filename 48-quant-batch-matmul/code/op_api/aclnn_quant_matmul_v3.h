/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_QUANT_MATMUL_V3_
#define OP_API_INC_QUANT_MATMUL_V3_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnQuantMatmulV3的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：实现QuantBatchMatmulV3计算
 * @param [in] x1: matmul左矩阵，数据类型支持：int8, int4, int32。
 * @param [in] x2: matmul右矩阵，数据类型支持：int8, int4, int32。
 * @param [in] scale: 量化参数，数据类型支持：uint64, float32, int64, bfloat16。
 * @param [in] offset: 量化参数，数据类型支持：float32。
 * @param [in] bias: 偏置，数据类型支持：int32, bfloat16, float32。
 * @param [in] transposeX1: x1矩阵是否转置。
 * @param [in] transposeX2: x2矩阵是否转置。
 * @param [out] out: 计算结果，数据类型：float16, int8, bfloat16, int32。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
__attribute__((deprecated("aclnnQuantMatmulV3GetWorkspaceSize is scheduled to be deprecated in a post-December 2026 version update, "
                        "and will be replaced by the aclnnQuantMatmulV5GetWorkspaceSize. "
                        "We apologize for any inconvenience caused and appreciate your timely migration to the new interface.")))
ACLNN_API aclnnStatus aclnnQuantMatmulV3GetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
                                                         const aclTensor* scale, const aclTensor* offset,
                                                         const aclTensor* bias, bool transposeX1, bool transposeX2,
                                                         const aclTensor* out, uint64_t* workspaceSize,
                                                         aclOpExecutor** executor);

/**
 * @brief aclnnQuantMatmulV3的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnQuantMatmulV3GetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
__attribute__((deprecated("aclnnQuantMatmulV3 is scheduled to be deprecated in a post-December 2026 version update, "
                        "and will be replaced by the aclnnQuantMatmulV5. "
                        "We apologize for any inconvenience caused and appreciate your timely migration to the new interface.")))
ACLNN_API aclnnStatus aclnnQuantMatmulV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QUANT_MATMUL_V3_