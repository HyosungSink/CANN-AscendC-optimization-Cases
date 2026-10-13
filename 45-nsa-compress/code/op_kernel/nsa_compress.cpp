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
 * \file nsa_compress.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "nsa_compress_seq_manager.h"
#include "nsa_compress_tiling.h"
#include "nsa_compress.h"


using namespace NASCompress;

namespace NASCompress {

extern "C" __global__ __aicore__ void nsa_compress(__gm__ uint8_t *input, __gm__ uint8_t *weight,
                                                   __gm__ uint8_t *actSeqLenOptional, __gm__ uint8_t *output,
                                                   __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    AscendC::TPipe pipe;

    GET_TILING_DATA(tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 强制指定算子类型为AIV_ONLY,修复算子类型推导bug

    int32_t blockId = AscendC::GetBlockIdx();
    if (tiling_data.PerCoreOutputNum[blockId] == 0 || tiling_data.PerCoreHeadNum[blockId] == 0) {
        return;
    }
#if (ORIG_DTYPE_INPUT == DT_BF16)
    if (TILING_KEY_IS(0)) {
        KernelNASCompress<bfloat16_t> op;
        op.Init(input, weight, actSeqLenOptional, output, tiling_data, &pipe);
        op.Process();
    }
#endif
#if (ORIG_DTYPE_INPUT == DT_FLOAT16)
    if (TILING_KEY_IS(0)) {
        KernelNASCompress<half> op;
        op.Init(input, weight, actSeqLenOptional, output, tiling_data, &pipe);
        op.Process();
    }
#endif
}

} // namespace NASCompress