/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_QUANT_MATMUL_NZ
#define OP_API_INC_QUANT_MATMUL_NZ

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnQuantMatmulWeightNz的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：相对于aclnnQuantBatchMatmulV4, 新增了支持x2为nz格式。
 * @param [in] x1: matmul左矩阵，数据类型支持：int8, float8_e4m3fn。
 * @param [in] x2: matmul右矩阵，数据类型支持：int8, float8_e4m3fn。
 * @param [in] x1Scale: 量化参数，数据类型支持：float32, float8_e8m0。
 * @param [in] x2Scale: 量化参数，数据类型支持：uint64, float32, bfloat16, int64, float8_e8m0。
 * @param [in] yScale: 预留参数，当前接口不支持该参数。
 * @param [in] x1Offset: 预留参数，当前接口不支持该参数。
 * @param [in] x2Offset: 量化参数，数据类型支持：float32。
 * @param [in] yOffset: 预留参数，当前接口不支持该参数。
 * @param [in] bias: 偏置，数据类型支持：int32, bfloat16, float16, float32。
 * @param [in] transposeX1: a矩阵是否转置，默认值：false。
 * @param [in] transposeX2: b矩阵是否转置，默认值：false。
 * @param [in] groupSize: 量化参数，数据类型支持：int64。
 * @param [out] out: 计算结果，数据类型：float16, int8, bfloat16, int32, float32。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnQuantMatmulWeightNzGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                               const aclTensor *x1Scale, const aclTensor *x2Scale,
                                                               const aclTensor *yScale, const aclTensor *x1Offset,
                                                               const aclTensor *x2Offset, const aclTensor *yOffset,
                                                               const aclTensor *bias, bool transposeX1,
                                                               bool transposeX2, int64_t groupSize, aclTensor *out,
                                                               uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnQuantMatmulWeightNz的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnQuantMatmulWeightNzGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnQuantMatmulWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QUANT_MATMUL_NZ