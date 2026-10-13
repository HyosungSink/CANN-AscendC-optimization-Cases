/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qkv_rms_norm_rope_cache.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "qkv_rms_norm_rope_cache_b16_pa_nz_quant.h"
using namespace AscendC;
using namespace QkvRmsNormRopeCache;

#define QKV_RMS_NORM_ROPE_CACHE_B16_PA_NZ_QUANT 3

extern "C" __global__ __aicore__ void qkv_rms_norm_rope_cache(
    GM_ADDR qkv, GM_ADDR q_gamma, GM_ADDR k_gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR q_out, GM_ADDR k_cache, GM_ADDR v_cache,
    GM_ADDR k_scale, GM_ADDR v_scale, GM_ADDR k_offset, GM_ADDR v_offset, 
    GM_ADDR q_out_out, GM_ADDR k_cache_out, GM_ADDR v_cache_out, GM_ADDR q_out_proto, GM_ADDR k_cache_proto, GM_ADDR v_cache_proto, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if (TILING_KEY_IS(QKV_RMS_NORM_ROPE_CACHE_B16_PA_NZ_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(QkvRmsNormRopeCacheTilingData, tiling_data_in, tiling);
        const QkvRmsNormRopeCacheTilingData* __restrict tilingData = &tiling_data_in;
        KernelQkvRmsNormRopeCacheQuantB16PANZ<DTYPE_QKV, DTYPE_K_CACHE, DTYPE_V_CACHE> op(&pipe, tilingData);
        op.Init(qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache, k_scale, v_scale, k_offset, v_offset, q_out_proto, k_cache_proto, v_cache_proto);
        op.Process();
    } 
}