/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ADD_RMS_NORM_DYNAMIC_QUANT_H_
#define OP_API_INC_LEVEL2_ADD_RMS_NORM_DYNAMIC_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "../../common/inc/op_api/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAddRmsNormDynamicQuant的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：Add + RmsNorm + 量化计算的融合算子，将加法的计算结果做层归一化计算后进行量化，
 * 并将加法的计算结果，被量化的归一化计算结果和量化参数返回。
 * 计算公式：
 *     x = x1 + x2
 *     y = RmsNorm(x, gamma)
 *     y1 = dynamicQuant(y, smoothScale1Optional, smoothScale2Optional)
 *     y2 = dynamicQuant(y, smoothScale1Optional, smoothScale2Optional)
 *
 * @param [in] x1:
 * 公式中的输入`x1`，数据类型支持BFLOAT16、FLOAT16，shape维度支持2-8维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] x2:
 * 公式中的输入`x2`，数据类型支持BFLOAT16、FLOAT16，shape维度支持2-8维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] gamma:
 * 公式中的输入`gamma`，数据类型支持BFLOAT16、FLOAT16
 * ，shape维度需要小于或等于8维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] smoothScale1Optional:
 * 公式中的输入`smoothScale1`，数据类型支持BFLOAT16、FLOAT16，shape维度需要等于x1的最后一维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] smoothScale2Optional:
 * 公式中的输入`smoothScale2Optional`，数据类型支持BFLOAT16、FLOAT16，shape维度需要等于x1的最后一维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] epsilon: double 类型，层归一化中用到的防止除0的参数。
 * @param [in] y1Out:
 * 公式中的输出`y1`，数据类型支持INT8、HiFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN，shape需要与x1一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] y2Out:
 * 公式中的输出`y2`，数据类型支持INT8、HiFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN，shape需要与x1一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] xOut:
 * 公式中的输出`x`，数据类型支持BFLOAT、FLOAT16且需要与x1一致，shape需要与x1一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] scale1Out:
 * 公式中的输出`scale1`，数据类型支持FLOAT，shape需要与x1除最后一维一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] scale2Out:
 * 公式中的输出`scale2`，数据类型支持FLOAT，shape需要与x1除最后一维一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAddRmsNormDynamicQuantGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, double epsilon, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut,
    aclTensor* scale1Out, aclTensor* scale2Out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAddRmsNormQuant的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnAddRmsNormDynamicQuantGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnAddRmsNormDynamicQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ADD_RMS_NORM_QUANT_H_
