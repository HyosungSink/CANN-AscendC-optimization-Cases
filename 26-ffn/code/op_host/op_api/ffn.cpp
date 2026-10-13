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
 * \file ffn.cpp
 * \brief
 */

#include "ffn.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(FFN);

const aclTensor *FFN(const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2,
                     const aclTensor *expertTokensOptional, const aclTensor *bias1Optional,
                     const aclTensor *bias2Optional, const aclTensor *scaleOptional, const aclTensor *offsetOptional,
                     const aclTensor *deqScale1Optional, const aclTensor *deqScale2Optional,
                     const aclTensor *antiquantScale1Optional, const aclTensor *antiquantScale2Optional,
                     const aclTensor *antiquantOffset1Optional, const aclTensor *antiquantOffset2Optional,
                     const char *activation, int64_t innerPrecise, const op::DataType yDtype, bool tokensIndexFlag,
                     aclOpExecutor *executor)
{
    L0_DFX(FFN, x, weight1, weight2, bias1Optional, bias2Optional, scaleOptional, offsetOptional, deqScale1Optional,
           deqScale2Optional, antiquantScale1Optional, antiquantScale2Optional, antiquantOffset1Optional,
           antiquantOffset2Optional, activation, innerPrecise, yDtype, tokensIndexFlag);
    auto ffnOut = executor->AllocTensor(x->GetStorageShape(), x->GetViewShape(), yDtype, x->GetStorageFormat(),
                                        x->GetOriginalFormat());
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(FFN,
                                           OP_INPUT(x, weight1, weight2, expertTokensOptional, bias1Optional,
                                                    bias2Optional, scaleOptional, offsetOptional, deqScale1Optional,
                                                    deqScale2Optional, antiquantScale1Optional, antiquantScale2Optional,
                                                    antiquantOffset1Optional, antiquantOffset2Optional),
                                           OP_OUTPUT(ffnOut), OP_ATTR(activation, innerPrecise, -1, tokensIndexFlag));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FFN launch kernel failed.");
        return nullptr;
    }
    return ffnOut;
}

} // namespace l0op