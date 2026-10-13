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
 * \file moe_token_unpermute.cpp
 * \brief
 */
#include "moe_token_unpermute.h"
#include "kernel_operator.h"

#define MOE_TOKEN_UNPERMUTE_IMPL(T1, T2, T3, haveProbs)                                  \
    do {                                                                                 \
        GET_TILING_DATA(tiling_data_in, tiling);                                         \
        const MoeTokenUnpermuteTilingData* __restrict tiling_data = &tiling_data_in;     \
        KernelMoeTokenUnpermute<T1, T2, T3, haveProbs> op;                               \
        op.Init(permuted_tokens, sorted_indices, probs, unpermuted_tokens, tiling_data, &pipe); \
        op.Process();                                                                    \
    } while (0)

extern "C" __global__ __aicore__ void moe_token_unpermute(
    GM_ADDR permuted_tokens, GM_ADDR sorted_indices, GM_ADDR probs, GM_ADDR unpermuted_tokens, GM_ADDR workspace,
    GM_ADDR tiling)
{
    TPipe pipe;
#if __CCE_AICORE__ == 200
    if (TILING_KEY_IS(2)) {
        MOE_TOKEN_UNPERMUTE_IMPL(half, int32_t, half, false);
    } else if (TILING_KEY_IS(3)) {
        MOE_TOKEN_UNPERMUTE_IMPL(half, int32_t, half, true);
    } else if (TILING_KEY_IS(27)) {
        // mix mode
        MOE_TOKEN_UNPERMUTE_IMPL(half, int32_t, float, true);
    } else if (TILING_KEY_IS(4)) {
        MOE_TOKEN_UNPERMUTE_IMPL(float, int32_t, float, false);
    } else if (TILING_KEY_IS(5)) {
        MOE_TOKEN_UNPERMUTE_IMPL(float, int32_t, float, true);
    } else if (TILING_KEY_IS(21)) {
        // mix mode
        MOE_TOKEN_UNPERMUTE_IMPL(float, int32_t, half, true);
    }
#else
    //==================================BF16==================================
    if (TILING_KEY_IS(0)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        MOE_TOKEN_UNPERMUTE_IMPL(bfloat16_t, int32_t, bfloat16_t, false);
    } else if (TILING_KEY_IS(1)) {
        MOE_TOKEN_UNPERMUTE_IMPL(bfloat16_t, int32_t, bfloat16_t, true);
    } else if (TILING_KEY_IS(17)) {
        // mix mode
        MOE_TOKEN_UNPERMUTE_IMPL(bfloat16_t, int32_t, half, true);
    } else if (TILING_KEY_IS(25)) {
        // mix mode
        MOE_TOKEN_UNPERMUTE_IMPL(bfloat16_t, int32_t, float, true);
#endif
    }
    //==================================FP16==================================
    else if (TILING_KEY_IS(2)) {
        MOE_TOKEN_UNPERMUTE_IMPL(half, int32_t, half, false);
    } else if (TILING_KEY_IS(3)) {
        MOE_TOKEN_UNPERMUTE_IMPL(half, int32_t, half, true);
    } else if (TILING_KEY_IS(11)) {
        // mix mode
        MOE_TOKEN_UNPERMUTE_IMPL(half, int32_t, bfloat16_t, true);
    } else if (TILING_KEY_IS(27)) {
        // mix mode
        MOE_TOKEN_UNPERMUTE_IMPL(half, int32_t, float, true);
    }
    //==================================FP32==================================
    else if (TILING_KEY_IS(4)) {
        MOE_TOKEN_UNPERMUTE_IMPL(float, int32_t, float, false);
    } else if (TILING_KEY_IS(5)) {
        MOE_TOKEN_UNPERMUTE_IMPL(float, int32_t, float, true);
    } else if (TILING_KEY_IS(13)) {
        // mix mode
        MOE_TOKEN_UNPERMUTE_IMPL(float, int32_t, bfloat16_t, true);
    } else if (TILING_KEY_IS(21)) {
        // mix mode
        MOE_TOKEN_UNPERMUTE_IMPL(float, int32_t, half, true);
    }
#endif
}
