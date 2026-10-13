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
 * \file moe_token_unpermute_common.h
 * \brief
 */
#ifndef MOE_TOKEN_UNPERMUTE_COMMON_H
#define MOE_TOKEN_UNPERMUTE_COMMON_H

#include "kernel_operator.h"

namespace MoeTokenUnPermute {
using namespace AscendC;

constexpr int64_t BLOCK_BYTES = 32;

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

template <typename T>
__aicore__ inline void DataCopyPadCustom(
    LocalTensor<T> inLocal, GlobalTensor<T> srcGm, DataCopyExtParams tokenCopyParams, DataCopyPadExtParams<T> padParams)
{
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    DataCopyPad(inLocal, srcGm, tokenCopyParams, padParams);
#else
    int64_t elem = tokenCopyParams.blockLen / sizeof(T);
    int64_t numPerBlock = BLOCK_BYTES / sizeof(T);
    int64_t alignElem = AlignUp(elem, numPerBlock);

    if (likely(alignElem == elem)) {
        DataCopyParams copyParams = {tokenCopyParams.blockCount, static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
        DataCopy(inLocal, srcGm, copyParams);
    } else {
        DataCopyParams copyParams = {1, static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
        for (uint32_t i = 0; i < tokenCopyParams.blockCount; i++) {
            DataCopy(inLocal[i * alignElem], srcGm[i * elem], copyParams);
        }
    }
#endif
}

template <typename T>
__aicore__ inline void DataCopyCustom(
    GlobalTensor<T> dstGm, LocalTensor<T> inLocal, int64_t blockCount, int64_t blockLen)
{
    int64_t elem = blockLen / sizeof(T);
    int64_t numPerBlock = sizeof(T) == 0 ? 1 : BLOCK_BYTES / sizeof(T);
    int64_t alignElem = AlignUp(elem, numPerBlock);

    if (likely(alignElem == elem)) {
        DataCopyParams copyParams = {
            static_cast<uint16_t>(blockCount), static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
        DataCopy(dstGm, inLocal, copyParams);
    } else {
        if (blockCount == 1) {
            DataCopyParams copyParams = {
                static_cast<uint16_t>(blockCount), static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
            DataCopy(dstGm, inLocal, copyParams);
        }
        else {
            DataCopyParams copyParams = {1, static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
            for (uint32_t i = 0; i < blockCount; i++) {
                DataCopy(dstGm[i * elem], inLocal[i * alignElem], copyParams);
                PipeBarrier<PIPE_MTE3>();
            }
        }
    }
}

} // namespace MoeTokenUnPermute
#endif // MOE_TOKEN_UNPERMUTE_COMMON_H
