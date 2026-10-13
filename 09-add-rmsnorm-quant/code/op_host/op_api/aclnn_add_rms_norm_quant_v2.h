/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ADD_RMS_NORM_QUANT_V2_H_
#define OP_API_INC_LEVEL2_ADD_RMS_NORM_QUANT_V2_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAddRmsNormQuantV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：Add + RmsNorm + 量化计算的融合算子，将加法的计算结果做层归一化计算后进行量化，
 * 并将加法的计算结果，被量化的归一化计算结果和量化参数返回。
 * 计算公式：
 *     x = x1 + x2
 *     y = RmsNorm(x, gamma)
 *     y1 = quant(y, scales1, zero_points1)
 *     y2 = quant(y, scales2, zero_points2)
 *
 * @param [in] x1:
 * 公式中的输入`x1`，数据类型支持BFLOAT16、FLOAT、FLOAT16，shape维度需要小于或等于8维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] x2:
 * 公式中的输入`x2`，数据类型支持BFLOAT16、FLOAT、FLOAT16，shape维度需要小于或等于8维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] gamma:
 * 公式中的输入`gamma`，数据类型支持BFLOAT16、FLOAT、FLOAT16，shape维度需要小于或等于8维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] scales1:
 * 公式中的输入`scales1`，数据类型支持BFLOAT16、FLOAT、FLOAT16，shape维度需要小于或等于8维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] scales2Optional:
 * 公式中的输入`scales2`，数据类型支持BFLOAT16、FLOAT、FLOAT16，shape维度需要小于或等于8维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] zeroPoints1Optional:
 * 公式中的输入`zero_points1`，可选参数，数据类型支持INT32、BFLOAT16、FLOAT、FLOAT16，shape维度需要小于或等于8维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] zeroPoints2Optional:
 * 公式中的输入`zero_points2`，可选参数，数据类型支持INT32、BFLOAT16、FLOAT、FLOAT16，shape维度需要小于或等于8维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] betaOptional:
 * 公式中的输入`beta`，可选参数，数据类型支持BFLOAT16、FLOAT、FLOAT16，shape维度需要小于或等于8维。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] axis: int64 类型，指定要应用的量化轴，当前仅支持-1。
 * @param [in] epsilon: double 类型，层归一化中用到的防止除0的参数。
 * @param [in] divMode: bool 类型，用于指定静态量化计算的算法。当为False时，静态量化计算计算使用乘法，反之亦然。
 * @param [in] y1Out:
 * 公式中的输出`y1`，数据类型支持INT8，shape需要与x1一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] y2Out:
 * 公式中的输出`y2`，数据类型支持INT8，shape需要与x1一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] xOut:
 * 公式中的输出`x`，数据类型支持BFLOAT16、FLOAT、FLOAT16且需要与x1一致，shape需要与x1一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] rmsNormOut:
 * 公式中的输出`rmsNormout`，数据类型支持BFLOAT16、FLOAT、FLOAT16且需要与x1一致，shape需要与x1一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAddRmsNormQuantV2GetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *gamma,
    const aclTensor *scales1, const aclTensor *scales2Optional,
    const aclTensor *zeroPoints1Optional, const aclTensor *zeroPoints2Optional,
    const aclTensor *betaOptional,
    int64_t axis, double epsilon, bool divMode,
    aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut, aclTensor* rmsNormOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAddRmsNormQuantV2的第二段接口，用于执行计算。
 *
 * 算子功能：将输入tensor转换为指定的dtype类型。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnAddRmsNormQuantV2GetWorkspaceSize。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnAddRmsNormQuantV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ADD_RMS_NORM_QUANT_V2_H_
