/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_FLASH_ATTENTION_SCORE_GRAD_H_
#define OP_API_INC_LEVEL0_FLASH_ATTENTION_SCORE_GRAD_H_

#include "opdev/op_executor.h"

namespace l0op {

const std::size_t MAX_OPTIONAL_CNT = 9;
const std::size_t MAX_FAG_OUTPUT_CNT = 7;

const std::array<const aclTensor *, MAX_FAG_OUTPUT_CNT> FlashAttentionScoreGrad(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, const aclTensor *dScaleQOptional,
    const aclTensor *dScaleKOptional, const aclTensor *dScaleVOptional, const aclTensor *dScaleDyOptional,
    const aclTensor *dScaleOOptional, const aclTensor *dsScaleOptional, const aclTensor *pScaleOptional,
    const aclTensor *queryRope, const aclTensor *keyRope,
    const aclTensor *sinkInOptional, double scaleValueOptional, double keepProbOptional, 
    int64_t preTockensOptional, int64_t nextTockensOptional, int64_t headNum,
    char *inputLayout, int64_t innerPreciseOptional, int64_t sparseModeOptional, int64_t pseTypeOptional,
    int64_t seed, int64_t offset, int64_t outDTypeOptional,char *softmaxInLayout, aclOpExecutor *executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_FLASH_ATTENTION_SCORE_GRAD_H_
