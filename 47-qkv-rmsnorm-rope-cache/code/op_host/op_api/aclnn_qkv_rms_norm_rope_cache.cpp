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
 * \file aclnn_qkv_rms_norm_rope_cache.cpp
 * \brief
 */

#include "aclnn_qkv_rms_norm_rope_cache.h"
#include "qkv_rms_norm_rope_cache.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace qkv_rms_norm_rope_cache {
    static inline bool CheckNotNull(const aclTensor *qkv, 
                                    const aclTensor *qGamma,
                                    const aclTensor *kGamma,
                                    const aclTensor* cos, 
                                    const aclTensor* sin, 
                                    const aclTensor* index,
                                    aclTensor* qOut, 
                                    aclTensor* kCache, 
                                    aclTensor* vCache,
                                    const aclIntArray* qkvSize, 
                                    const aclIntArray* headNums) {
        OP_CHECK_NULL(qkv, return false);
        OP_CHECK_NULL(qGamma, return false);
        OP_CHECK_NULL(kGamma,  return false);
        OP_CHECK_NULL(cos,  return false);
        OP_CHECK_NULL(sin, return false);
        OP_CHECK_NULL(index, return false);
        OP_CHECK_NULL(qOut, return false);
        OP_CHECK_NULL(kCache,  return false);
        OP_CHECK_NULL(vCache,  return false);
        OP_CHECK_NULL(qkvSize, return false);
        OP_CHECK_NULL(headNums, return false);
        return true;
    }
} // namespace qkv_rms_norm_rope_cache

aclnnStatus aclnnQkvRmsNormRopeCacheGetWorkspaceSize(
    const aclTensor* qkv, const aclTensor* qGamma, const aclTensor* kGamma, const aclTensor* cos, const aclTensor* sin, 
    const aclTensor* index, aclTensor* qOut, aclTensor* kCache, aclTensor* vCache, const aclTensor* kScaleOptional, const aclTensor* vScaleOptional, 
    const aclTensor* kOffsetOptional, const aclTensor* vOffsetOptional, const aclIntArray* qkvSize, const aclIntArray* headNums, double epsilon, char* cacheModeOptional,
    const aclTensor* qOutBeforeQuant, const aclTensor* kOutBeforeQuant, const aclTensor* vOutBeforeQuant, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnQkvRmsNormRopeCache,
                   DFX_IN(qkv, qGamma, kGamma, cos, sin, index, qOut, kCache, vCache, kScaleOptional, vScaleOptional, kOffsetOptional, vOffsetOptional, qkvSize, headNums, epsilon, cacheModeOptional),
                   DFX_OUT(qOut, kCache, vCache, qOutBeforeQuant, kOutBeforeQuant, vOutBeforeQuant));
    
    // 固定写法，参数检查
    auto ret = qkv_rms_norm_rope_cache::CheckNotNull(qkv, qGamma, kGamma, cos, sin, index, qOut, kCache, vCache, 
                            qkvSize, headNums);

    CHECK_RET(ret, ACLNN_ERR_PARAM_NULLPTR);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，将输入self转换成连续的tensor
    auto qkvContiguous = l0op::Contiguous(qkv, uniqueExecutor.get()); 
    CHECK_RET(qkvContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto qGammaContiguous = l0op::Contiguous(qGamma, uniqueExecutor.get()); 
    CHECK_RET(qGammaContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto kGammaContiguous = l0op::Contiguous(kGamma, uniqueExecutor.get()); 
    CHECK_RET(kGammaContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto cosContiguous = l0op::Contiguous(cos, uniqueExecutor.get()); 
    CHECK_RET(cosContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto sinContiguous = l0op::Contiguous(sin, uniqueExecutor.get()); 
    CHECK_RET(sinContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto indexContiguous = l0op::Contiguous(index, uniqueExecutor.get()); 
    CHECK_RET(indexContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto qOutContiguous = l0op::Contiguous(qOut, uniqueExecutor.get());
    qOut = const_cast<aclTensor*>(qOutContiguous);
    auto kCacheContiguous = l0op::Contiguous(kCache, uniqueExecutor.get());
    kCache = const_cast<aclTensor*>(kCacheContiguous);
    auto vCacheContiguous = l0op::Contiguous(vCache, uniqueExecutor.get());
    vCache = const_cast<aclTensor*>(vCacheContiguous);

    // 固定写法，可选输入转成连续的tensor
    const aclTensor* kScaleContiguous = nullptr;
    const aclTensor* kOffsetContiguous = nullptr;
    const aclTensor* vScaleContiguous = nullptr;
    const aclTensor* vOffsetContiguous = nullptr;
    if (kScaleOptional != nullptr) {
        kScaleContiguous = l0op::Contiguous(kScaleOptional, uniqueExecutor.get()); 
        CHECK_RET(kScaleContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    }
    if (kOffsetOptional != nullptr) {
        kOffsetContiguous = l0op::Contiguous(kOffsetOptional, uniqueExecutor.get()); 
        CHECK_RET(kOffsetContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    }
    if (vScaleOptional != nullptr) {
        vScaleContiguous = l0op::Contiguous(vScaleOptional, uniqueExecutor.get()); 
        CHECK_RET(vScaleContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    }
    if (vOffsetOptional != nullptr) {
        vOffsetContiguous = l0op::Contiguous(vOffsetOptional, uniqueExecutor.get()); 
        CHECK_RET(vOffsetContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    }
    
    // 设置isOutputQkv
    bool isOutputQkv = qOutBeforeQuant == nullptr ? false : true;
    
    auto qkvRmsNormRopeCacheResult = std::tuple<aclTensor*, aclTensor*, aclTensor*>(nullptr, nullptr, nullptr);
    
    // 调用v3版本的l0接口进行计算
    qkvRmsNormRopeCacheResult = l0op::QkvRmsNormRopeCache(qkvContiguous, qGammaContiguous, kGammaContiguous, cosContiguous, 
                                        sinContiguous, indexContiguous, qOut, kCache, vCache, kScaleContiguous, vScaleContiguous,
                                        kOffsetContiguous, vOffsetContiguous, qkvSize, headNums, epsilon, cacheModeOptional, isOutputQkv, uniqueExecutor.get());
    auto [qOutBeforeQuant_, kOutBeforeQuant_, vOutBeforeQuant_] = qkvRmsNormRopeCacheResult;
    bool checkBeforeQuantOut = ((qOutBeforeQuant_ != nullptr) && (kOutBeforeQuant_ != nullptr) && (vOutBeforeQuant_ != nullptr)) || 
                      ((qOutBeforeQuant_ == nullptr) && (kOutBeforeQuant_ == nullptr) && (vOutBeforeQuant_ == nullptr));
    CHECK_RET(checkBeforeQuantOut == true, ACLNN_ERR_INNER_NULLPTR);

    bool hasBeforeQuantOut = qOutBeforeQuant_ != nullptr ? true : false;

    if (hasBeforeQuantOut) {
        auto viewCopyQOutBeforeQuantResult = l0op::ViewCopy(qOutBeforeQuant_, qOutBeforeQuant, uniqueExecutor.get());
        CHECK_RET(viewCopyQOutBeforeQuantResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyKOutBeforeQuantResult = l0op::ViewCopy(kOutBeforeQuant_, kOutBeforeQuant, uniqueExecutor.get());
        CHECK_RET(viewCopyKOutBeforeQuantResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyVOutBeforeQuantOutResult = l0op::ViewCopy(vOutBeforeQuant_, vOutBeforeQuant, uniqueExecutor.get());
        CHECK_RET(viewCopyVOutBeforeQuantOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQkvRmsNormRopeCache(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQkvRmsNormRopeCache);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif 