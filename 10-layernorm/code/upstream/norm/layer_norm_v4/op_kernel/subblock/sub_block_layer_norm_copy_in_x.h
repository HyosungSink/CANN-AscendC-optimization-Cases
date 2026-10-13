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
 * \file sub_block_layer_norm_copy_in_x.h
 * \brief
 */

#ifndef SUB_BLOCK_LAYER_NORM_COPY_IN_X_H
#define SUB_BLOCK_LAYER_NORM_COPY_IN_X_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace NormSubBlock {
using namespace AscendC;

template <
    class TileCopy_
>
class SubBlockLayerNormCopyInX {
public:
    using TileCopy = TileCopy_;

    __aicore__ inline SubBlockLayerNormCopyInX() {}

    template <typename T>
    __aicore__ inline void operator()(const LocalTensor<float> xLocal, const GlobalTensor<T> xGlobal, uint32_t dataSize) {
        tileCopy(xLocal, xGlobal, dataSize);
    }

protected:
    TileCopy tileCopy;
};
} // namespace NormSubBlock

#endif // LAYER_NORM_COPY_IN_BLOCK_X_H