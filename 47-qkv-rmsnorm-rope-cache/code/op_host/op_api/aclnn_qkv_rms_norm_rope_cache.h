/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_qkv_rms_norm_rope_cache.h
 * \brief
 */

#ifndef OP_API_ACLNN_QKV_RMS_NORM_ROPE_CACHE_H_
#define OP_API_ACLNN_QKV_RMS_NORM_ROPE_CACHE_H_

#include "aclnn/aclnn_base.h"
#include "../../common/include/external/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnQkvRmsNormRopeCache的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] qkv: 计算输入张量，npu
 * device侧的aclTensor，数据类型支持FLOAT16，BFLOAT16，数据格式支持ND。
 * @param [in] qGamma: 计算输入张量，npu
 * device侧的aclTensor， 数据类型支持FLOAT16，BFLOAT16，数据格式支持ND。
 * @param [in] kGamma: 计算输入张量，npu
 * device侧的aclTensor， 数据类型支持FLOAT16，BFLOAT16，数据格式支持ND。
 * @param [in] cos: 计算输入张量，npu
 * device侧的aclTensor， 数据类型支持FLOAT16，BFLOAT16，数据格式支持ND。
 * @param [in] sin: 计算输入张量，npu
 * device侧的aclTensor， 数据类型支持FLOAT16，BFLOAT16，数据格式支持ND。
 * @param [in] index: 计算输入张量，npu
 * device侧的aclTensor， 数据类型支持INT64，数据格式支持ND。
 * @param [in] qOut: 计算输出张量，原地更新，npu
 * device侧的aclTensor， 数据类型支持FLOAT16，BFLOAT16，数据格式支持ND。
 * @param [in] kCache: 计算输出张量，原地更新，npu
 * device侧的aclTensor， 数据类型支持FLOAT16，BFLOAT16，INT8，数据格式支持ND。
 * @param [in] vCache: 计算输出张量，原地更新，npu
 * device侧的aclTensor， 数据类型支持FLOAT16，BFLOAT16，INT8，数据格式支持ND。
 * @param [in] kScale: 可选输入张量，npu
 * device侧的aclTensor， 数据类型支持FLOAT32，数据格式支持ND。
 * @param [in] vScale: 可选输入张量，npu
 * device侧的aclTensor，数据类型支持FLOAT32，数据格式支持ND。
 * @param [in] kOffset: 可选输入张量，npu
 * device侧的aclTensor，数据类型支持FLOAT32，数据格式支持ND。
 * @param [in] vOffset: 可选输入张量，npu
 * device侧的aclTensor，数据类型支持FLOAT32，数据格式支持ND。
 * @param [in] qkvSize: 计算属性，host侧的aclIntArray。
 * @param [in] headNums: 计算属性，host侧的aclIntArray。
 * @param [in] epsilon: 计算属性，数据类型支持double，防止计算除0。
 * @param [in] cacheModeOptional: 计算属性，host侧的char*，cacheModeOptional=nullptr是，使用建议值"PA_NZ"。
 * @param [in] qOutBeforeQuant: 可选输出张量，npu
 * device侧的aclTensor，数据类型支持FLOAT16，BFLOAT16，数据格式支持ND。
 * @param [in] kOutBeforeQuant: 可选输出张量，npu
 * device侧的aclTensor，数据类型支持FLOAT16，BFLOAT16，数据格式支持ND。
 * @param [in] vOutBeforeQuant: 可选输出张量，npu
 * device侧的aclTensor，数据类型支持FLOAT16，BFLOAT16，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小.
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnQkvRmsNormRopeCacheGetWorkspaceSize(
    const aclTensor* qkv, const aclTensor* qGamma, const aclTensor* kGamma, const aclTensor* cos, const aclTensor* sin, 
    const aclTensor* index, aclTensor* qOut, aclTensor* kCache, aclTensor* vCache, const aclTensor* kScaleOptional, const aclTensor* vScaleOptional, 
    const aclTensor* kOffsetOptional, const aclTensor* vOffsetOptional, const aclIntArray* qkvSize, const aclIntArray* headNums, double epsilon, char* cacheModeOptional,
    const aclTensor* qOutBeforeQuant, const aclTensor* kOutBeforeQuant, const aclTensor* vOutBeforeQuant, uint64_t* workspaceSize, aclOpExecutor** executor);

/* @brief aclnnQkvRmsNormRopeCache的第二段接口，用于执行计算.
 * @param [in] workspace: 在npu device侧申请的workspace内存起址
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnQkvRmsNormRopeCacheGetWorkspaceSize获取.
 * @param [in] executor: op执行器，包含了算子计算流程.
 * @param [in] stream: acl stream流.
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnQkvRmsNormRopeCache(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_QKV_RMS_NORM_ROPE_CACHE_H_
