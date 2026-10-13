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
 * \file tile_copy_out.h
 * \brief
 */

#ifndef TILE_COPY_OUT_H
#define TILE_COPY_OUT_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace NormTile {
using namespace AscendC;

class TileCopyOut {
public:
    __aicore__ inline TileCopyOut() {}

    template <typename Tfm>
    __aicore__ inline void operator()(const GlobalTensor<Tfm> yGlobal, const LocalTensor<float> yLocal, uint32_t dataSize) {
        uint32_t dataBytesSize = dataSize * sizeof(Tfm);
        TEventID eventIdVToMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        if constexpr (IsSameType<Tfm, float>::value) {
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            DataCopyPad(yGlobal, yLocal, {1, dataBytesSize, 0, 0, 0});
        } else {
            LocalTensor<Tfm> yLocalTfm = yLocal.template ReinterpretCast<Tfm>();
            PipeBarrier<PIPE_V>();
            if constexpr (IsSameType<Tfm, bfloat16_t>::value) {
                Cast(yLocalTfm, yLocal, RoundMode::CAST_ROUND, dataSize);
            } else {
                Cast(yLocalTfm, yLocal, RoundMode::CAST_NONE, dataSize);
            }
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            DataCopyPad(yGlobal, yLocalTfm, {1, dataBytesSize, 0, 0, 0});
        }
    }
};
} // namespace NormTile

#endif // Tile_COPY_OUT_H