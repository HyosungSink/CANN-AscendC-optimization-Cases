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
 * \file sub_block_empty.h
 * \brief
 */

#ifndef SUB_BLOCK_EMPTY_H
#define SUB_BLOCK_EMPTY_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace NormSubBlock {
using namespace AscendC;

class SubBlockEmpty {
public:

    __aicore__ inline SubBlockEmpty() {}

    __aicore__ inline void operator()() {
    }
};
} // namespace NormSubBlock

#endif // SUB_BLOCK_EMPTY_H