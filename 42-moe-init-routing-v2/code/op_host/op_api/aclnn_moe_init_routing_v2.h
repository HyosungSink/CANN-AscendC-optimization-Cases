/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_moe_init_routing_v2.h
 * \brief
 */
#ifndef ACLNN_MOE_INIT_ROUTING_V2_H_
#define ACLNN_MOE_INIT_ROUTING_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMoeInitRoutingV2GetWorkspaceSize
 * parameters :
 * x : required
 * expertIdx : required
 * activeNumOptional : optional
 * expertCapacityOptional : optional
 * expertNumOptional : optional
 * dropPadModeOptional : optional
 * expertTokensCountOrCumsumFlagOptional : optional
 * expertTokensBeforeCapacityFlagOptional : optional
 * expandedXOut : required
 * expandedRowIdxOut : required
 * expertTokensCountOrCumsumOutOptional : optional
 * expertTokensBeforeCapacityOutOptional : optional
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMoeInitRoutingV2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIdx, int64_t activeNumOptional, int64_t expertCapacityOptional,
    int64_t expertNumOptional, int64_t dropPadModeOptional, int64_t expertTokensCountOrCumsumFlagOptional,
    bool expertTokensBeforeCapacityFlagOptional, const aclTensor *expandedXOut, const aclTensor *expandedRowIdxOut,
    const aclTensor *expertTokensCountOrCumsumOutOptional, const aclTensor *expertTokensBeforeCapacityOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/* funtion: aclnnMoeInitRoutingV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMoeInitRoutingV2(void *workspace, uint64_t workspaceSize,
                                                                         aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
