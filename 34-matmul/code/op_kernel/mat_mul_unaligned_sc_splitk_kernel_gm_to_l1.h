/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mat_mul_unaligned_sc_splitk_kernel_gm_to_l1.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_UNALIGNED_SC_SPLITK_KERNEL_GM_TO_L1_H__
#define __OP_KERNEL_MATMUL_V3_UNALIGNED_SC_SPLITK_KERNEL_GM_TO_L1_H__

#define GM_TO_L1

#include "mat_mul_unaligned_sc_splitk_kernel_base.h"

using namespace AscendC;
using namespace matmul;

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = MatmulSingleCoreSplitKBaseBlock,
    const MatmulConfig& MM_CFG = MM_CFG_NO_PRELOAD>
class MatMulUnAlignedSingleCoreSplitKKernelGmToL1
    : public MatMulUnAlignedSingleCoreSplitKKernelBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> {
public:
    __aicore__ inline MatMulUnAlignedSingleCoreSplitKKernelGmToL1() = default;
};
#endif // MMV3_MATMUL_KERNEL_H
