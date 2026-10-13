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
 * \file block_layer_norm_copy_in.h
 * \brief
 */

#ifndef BLOCK_LAYER_NORM_COPY_IN_X_H
#define BLOCK_LAYER_NORM_COPY_IN_X_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../subblock/sub_block_empty.h"

namespace NormBlock {
using namespace AscendC;

template <
    class SubBlockLayerNormCopyIn_,
    class BlockEpilogue_
>
class BlockLayerNormCopyInX {
public:
    using SubBlockLayerNormCopyIn = SubBlockLayerNormCopyIn_;
    using BlockEpilogue = BlockEpilogue_;

    __aicore__ inline BlockLayerNormCopyInX() {}

    template <typename T>
    __aicore__ inline void operator()(const LocalTensor<float> xLocal, const GlobalTensor<T> xGlobal, uint32_t dataSize) {
        if constexpr(!std::is_same_v<SubBlockLayerNormCopyIn, NormSubBlock::SubBlockEmpty>) {
            SubBlockLayerNormCopyIn subBlockLayerNormCopyIn;
            subBlockLayerNormCopyIn(xLocal, xGlobal, dataSize);
        }
        BlockEpilogue blockEpilogue;
        blockEpilogue();
    }
};
} // namespace NormBlock

#endif // BLOCK_LAYER_NORM_COPY_IN_X_H