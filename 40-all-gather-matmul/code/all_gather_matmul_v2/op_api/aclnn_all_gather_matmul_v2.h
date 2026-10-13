/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ALL_GATHER_MATMUL_V2_
#define OP_API_INC_ALL_GATHER_MATMUL_V2_

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * 算子功能：实现allGather + mm 融合计算
 * @brief aclnnAllGatherMatmulV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x1: matmul左矩阵，数据类型支持：float16, bfloat16, float8_e4m3fn, float8_e5m2, hifloat8, int8, int4。
 * @param [in] x2: matmul右矩阵，数据类型支持：float16, bfloat16, float8_e4m3fn, float8_e5m2, hifloat8, int8, int4。
 * @param [in] bias: 偏置，数据类型支持：float16, bfloat16, float32。
 * @param [in] x1Scale: matmul左矩阵量化参数，数据类型支持：float16, bfloat16, float32。
 * @param [in] x2Scale: matmul右矩阵量化参数，数据类型支持：float16, bfloat16, float32。
 * @param [in] quantScale: 量化参数，数据类型支持：float32。
 * @param [in] blockSize: 一个量化系数在output中不同轴对应的值的数量, 默认值: 0。
 * @param [in] group: 标识列组的字符串。
 * @param [in] gatherIndex: 标识gather目标，0：左矩阵，1：右矩阵。
 * @param [in] commTurn: 通信数据切分数，即总数据量/单次通信量。
 * @param [in] streamMode: acl流模式的枚举，类型支持：0/1。
 * @param [in] groupSize: 一个反量化系数在x1/x2中不同轴对应的值的数量, 默认值：0。
 * @param [in] commMode: 通信模式。当前支持两种模式: ccu/aiv。
 * @param [out] output: 计算+通信的结果，数据类型支持：float16, bfloat16, float32。
 * @param [out] gatherOut: 仅gather通信操作的结果，数据类型：同输入。
 * @param [out] amaxOut: matmul输出的最大值，数据类型：float32。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnAllGatherMatmulV2GetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                                   const aclTensor* x1Scale, const aclTensor* x2Scale,
                                                   const aclTensor* quantScale, int64_t blockSize, const char* group,
                                                   int64_t gatherIndex, int64_t commTurn, int64_t streamMode,
                                                   int64_t groupSize, const char* commMode, aclTensor* output, aclTensor* gatherOut,
                                                   aclTensor* amaxOut, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor);

/**
 * @brief aclnnAllGatherMatmulV2的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnAllGatherMatmulV2GetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnAllGatherMatmulV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ALL_GATHER_MATMUL_V2_