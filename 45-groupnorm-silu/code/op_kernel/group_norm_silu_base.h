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
 * \file group_norm_silu_base.h
 * \brief
 */

#ifndef GROUP_NORM_SILU_BASE_H_
#define GROUP_NORM_SILU_BASE_H_

#include "kernel_operator.h"

namespace platform {
__aicore__ inline constexpr bool IsDataCopyPadSupport()
{
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    return true;
#else
    return false;
#endif
}
}

namespace GroupNormSilu {
using namespace AscendC;

constexpr int32_t BLOCK_SIZE = 32;

template <typename T>
class GroupNormSiluBase
{
public:
    __aicore__ inline GroupNormSiluBase(){};

protected:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    };

    __aicore__ inline RoundMode GetRoundMode()
    {
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        return RoundMode::CAST_ROUND;
#else
        return RoundMode::CAST_NONE;
#endif
    };

    template <typename T1, bool isAlign = true>
    __aicore__ inline void CopyInData(
        const LocalTensor<T1>& dstUB, const GlobalTensor<T1>& srcGM, const int64_t dataCount)
    {
        if constexpr (isAlign) {
            DataCopy(dstUB, srcGM, dataCount);
        } else {
            if constexpr (platform::IsDataCopyPadSupport()) {
                DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
                copyParams.blockLen = dataCount * sizeof(T1);
                DataCopyPadExtParams<T1> padParams = {false, 0, 0, 0};
                DataCopyPad(dstUB, srcGM, copyParams, padParams);
            } else {
                int64_t elementsPerBlock = BLOCK_SIZE / sizeof(T1);
                int64_t dataCountAlign = CeilDiv(dataCount, elementsPerBlock) * elementsPerBlock;
                DataCopy(dstUB, srcGM, dataCountAlign);
            }
        }
    }

    template <typename T1, bool isAlign = true>
    __aicore__ inline void CopyOutData(
        const GlobalTensor<T1>& dstGM, const LocalTensor<T1>& srcUB, const int64_t dataCount)
    {
        if constexpr (isAlign) {
            DataCopy(dstGM, srcUB, dataCount);
        } else {
            if constexpr (platform::IsDataCopyPadSupport()) {
                DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
                copyParams.blockLen = dataCount * sizeof(T1);
                DataCopyPad(dstGM, srcUB, copyParams);
            } else {
                int64_t elementsPerBlock = BLOCK_SIZE / sizeof(T1);
                int64_t dataCountAlign = CeilDiv(dataCount, elementsPerBlock) * elementsPerBlock;
                DataCopy(dstGM, srcUB, dataCountAlign);
            }
        }
    }
};

} // namespace GroupNormSilu

#endif // GROUP_NORM_SILU_BASE_H_
