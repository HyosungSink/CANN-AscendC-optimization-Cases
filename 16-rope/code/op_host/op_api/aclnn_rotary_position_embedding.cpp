/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_rotary_position_embedding.h"
#include "aclnn_rotary_position_embedding_v2.h"
#include "rotary_position_embedding.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static inline bool TensorContiguousProcess(const aclTensor *&contiguousTensor, aclOpExecutor *executor)
{
    if (contiguousTensor == nullptr) {
        OP_LOGD("RotaryPositionEmbedding no need to do contiguous process.");
        return true;
    }

    contiguousTensor = l0op::Contiguous(contiguousTensor, executor);
    CHECK_RET(contiguousTensor != nullptr, false);
    return true;
}

static aclnnStatus PreProcess(const aclTensor *&x, const aclTensor *&cos, const aclTensor *&sin,
                              const aclTensor *&rotate, aclOpExecutor *executor)
{
    CHECK_RET(TensorContiguousProcess(x, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(TensorContiguousProcess(cos, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(TensorContiguousProcess(sin, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(TensorContiguousProcess(rotate, executor), ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus RotaryPositionEmbeddingCommonProcess(const aclTensor *x, const aclTensor *cos, const aclTensor *sin,
                                                         const aclTensor *rotate, int64_t mode, aclTensor *out,
                                                         aclOpExecutor *executor)
{
    const aclTensor *xProcessed = x;
    const aclTensor *cosProcessed = cos;
    const aclTensor *sinProcessed = sin;
    const aclTensor *rotateProcessed = rotate;

    auto ret = PreProcess(xProcessed, cosProcessed, sinProcessed, rotateProcessed, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto result = l0op::RotaryPositionEmbedding(xProcessed, cosProcessed, sinProcessed, rotateProcessed, mode, executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outContiguous = l0op::Contiguous(out, executor);
    CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(result, outContiguous, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

}

/* v1 interface - rotate parameter is implicitly nullptr */
aclnnStatus aclnnRotaryPositionEmbeddingGetWorkspaceSize(const aclTensor* x, const aclTensor* cos, const aclTensor* sin,
                                                         int64_t mode, aclTensor* out, uint64_t* workspaceSize,
                                                         aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnRotaryPositionEmbedding,
        DFX_IN(x, cos, sin, mode),
        DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor *rotate = nullptr;  // v1接口不支持rotate参数
    auto ret = RotaryPositionEmbeddingCommonProcess(x, cos, sin, rotate, mode, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRotaryPositionEmbedding(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRotaryPositionEmbedding);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

/* v2 interface - supports rotate parameter */
aclnnStatus aclnnRotaryPositionEmbeddingV2GetWorkspaceSize(const aclTensor* x, const aclTensor* cos, const aclTensor* sin,
                                                           int64_t mode, const aclTensor* rotate, aclTensor* out,
                                                           uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnRotaryPositionEmbeddingV2,
        DFX_IN(x, cos, sin, mode, rotate),
        DFX_OUT(out));

    if (rotate != nullptr) {
        CHECK_COND(op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201, 
                   ACLNN_ERR_PARAM_INVALID, "the soc verison is not support");
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    
    auto ret = RotaryPositionEmbeddingCommonProcess(x, cos, sin, rotate, mode, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRotaryPositionEmbeddingV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRotaryPositionEmbeddingV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
