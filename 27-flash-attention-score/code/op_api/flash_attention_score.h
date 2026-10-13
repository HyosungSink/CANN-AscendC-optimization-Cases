/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_OP_FLASH_ATTENTION_SCORE_OP_H_
#define OP_API_INC_LEVEL0_OP_FLASH_ATTENTION_SCORE_OP_H_

#include "opdev/op_executor.h"

namespace l0op {

const std::array<const aclTensor *, 4> FlashAttentionScore(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclTensor *sinkOptional, const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional, const aclTensor *dScaleQOptional, const aclTensor *dScaleKOptional,
    const aclTensor *dScaleVOptional, const aclTensor *pScaleOptional, const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional, double scaleValue, double keepProb, int64_t preTockens,
    int64_t nextTockens, int64_t headNum, const char *inputLayout, int64_t innerPrecise,
    int64_t sparseMode, int64_t pseType, int64_t seed, int64_t offset, int64_t outDtype, const char *softmaxOutLayout,
    aclOpExecutor *executor);
}

#endif // OP_API_INC_LEVEL0_OP_FLASH_ATTENTION_SCORE_OP_H_
