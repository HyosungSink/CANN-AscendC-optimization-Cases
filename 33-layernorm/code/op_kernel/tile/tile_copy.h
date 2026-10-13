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
 * \file tile_copy.h
 * \brief
 */

#ifndef TILE_COPY_H
#define TILE_COPY_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace NormTile {
using namespace AscendC;

class TileCopy {
public:
    static constexpr int32_t FP16_BLOCK_SIZE = 16;

    __aicore__ inline TileCopy() {}

    template <typename T>
    __aicore__ inline void operator()(const LocalTensor<float> xLocal, const GlobalTensor<T> xGlobal, uint32_t dataSize) {
        uint32_t dataBytesSize = dataSize * sizeof(T);
        TEventID eventIdMte2ToV = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(xLocal, xGlobal, {1, dataBytesSize, 0, 0, 0}, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        } else {
            uint32_t dataSizeAligned = (dataSize + FP16_BLOCK_SIZE - 1) / FP16_BLOCK_SIZE * FP16_BLOCK_SIZE;
            LocalTensor<T> xLocalTfm = xLocal.template ReinterpretCast<T>();
            DataCopyPad(xLocalTfm[dataSizeAligned], xGlobal, {1, dataBytesSize, 0, 0, 0}, {false, 0, 0, 0});

            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Cast(xLocal, xLocalTfm[dataSizeAligned], RoundMode::CAST_NONE, dataSize);
            PipeBarrier<PIPE_V>();
        }
    }
};
} // namespace NormTile

#endif // Tile_COPY_H