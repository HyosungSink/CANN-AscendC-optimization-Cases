/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLNN_DYNAMIC_QUANT_V3_H_
#define ACLNN_DYNAMIC_QUANT_V3_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnDynamicQuantV3GetWorkspaceSize的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x: npu
 * device侧的aclTensor，算子输入的Tensor，shape维度要大于1，Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] smoothScalesOptional: npu
 * 算子输入的smoothScales，
 * 当没有groupIndexOptional时，shape维度为1维；
 *  - 如果quantMode为"perchannel"，shape的第一维大小是x的倒数第二维。
 *  - 如果quantMode为"pertoken"或者"pertensor"，shape的第一维大小是x的最后一维。
 * 当有groupIndexOptional时，shape是两维
 *  - 第一维大小是专家数，不超过1024，第二维大小是x的最后一维。Device侧的aclTensor
 * 数据类型支持FLOAT16、BFLOAT16，并且数据类型要和x保持一致，支持非连续的Tensor，数据格式支持ND。
 * @param [in] groupIndexOptional: npu
 * 算子输入的groupIndex，Device侧的aclTensor，数据类型支持INT32，支持非连续的Tensor，数据格式支持ND。
 * shape只支持一维，且维度大小等于smoothScalesOptional的第一维。groupIndexOptional非nullptr时，smoothScalesOptional必须非nullptr。
 * @param [in]
 * dstType：输出yOut的数据类型对应的枚举值，Host侧的int。支持DT_INT8，DT_INT32，DT_INT4，DT_FLOAT8_E5M2，DT_FLOAT8_E4M3FN，DT_HIFLOAT8。
 * @param [in]
 * isSymmetrical：指定是否对称量化，Host侧的bool。
 * @param [in]
 * quantMode：用于指定量化的模式，Host侧的char*。当前支持"pertoken"，"pertensor"和"perchannel"。当quantMode取"pertensor"、"perchannel"时，groupIndexOptional必须是nullptr。
 * @param [in] yOut: npu
 * 量化后的输出Tensor，shape和x保持一致，类型由dstType指定。支持非连续的Tensor，数据格式支持ND。
 * @param [in] scaleOut: npu
 * 量化使用的scale，Device侧的aclTensor，数据类型支持FLOAT，支持非连续的Tensor，数据格式支持ND。
 * @param [in] offsetOut: npu
 * 量化使用的offset，Device侧的aclTensor，数据类型支持FLOAT，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDynamicQuantV3GetWorkspaceSize(
    const aclTensor* x, const aclTensor* smoothScalesOptional, const aclTensor* groupIndexOptional,
    int64_t dstType, bool isSymmetrical, const char* quantMode,
    const aclTensor* yOut, const aclTensor* scaleOut, const aclTensor* offsetOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnDynamicQuantV3的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnDynamicQuantV3GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDynamicQuantV3(void* workspace, uint64_t workspaceSize,
                                                                       aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_DYNAMIC_QUANT_V3_H_