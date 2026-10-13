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
 * \file dequant_rope_quant_kvcache.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_DEQUANT_ROPE_QUANT_KVCACHE_H
#define OPS_BUILD_IN_OP_TILING_DEQUANT_ROPE_QUANT_KVCACHE_H

#include "register/op_impl_registry.h"
#include "../common/include/tiling_base/tiling_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
struct DequantRopeQuantKvcacheCompileInfo {
};

BEGIN_TILING_DATA_DEF(DequantRopeQuantKvcacheTilingData)
TILING_DATA_FIELD_DEF(int64_t, qHeadNum);
TILING_DATA_FIELD_DEF(int64_t, kvHeadNum);
TILING_DATA_FIELD_DEF(int64_t, hiddenSize);
TILING_DATA_FIELD_DEF(int64_t, hiddenSizeFp32Align);
TILING_DATA_FIELD_DEF(int64_t, hiddenSizeFp16Align);
TILING_DATA_FIELD_DEF(int64_t, hiddenSizeInt8Align);
TILING_DATA_FIELD_DEF(int64_t, OnceUBMaxS);
TILING_DATA_FIELD_DEF(int64_t, cacheSeqlen);
TILING_DATA_FIELD_DEF(int64_t, seqlen);
TILING_DATA_FIELD_DEF(int64_t, qHiddenSize);
TILING_DATA_FIELD_DEF(int64_t, kHiddenSize);
TILING_DATA_FIELD_DEF(int64_t, vHiddenSize);
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, frontCoreNum);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailCoreBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, hasQuantOffset);
TILING_DATA_FIELD_DEF(int64_t, ifKVout);
TILING_DATA_FIELD_DEF(int64_t, isPA);
TILING_DATA_FIELD_DEF(int64_t, hasBias);
TILING_DATA_FIELD_DEF(int64_t, hasAS);
TILING_DATA_FIELD_DEF(int64_t, batch);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DequantRopeQuantKvcache, DequantRopeQuantKvcacheTilingData)
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_DEQUANT_ROPE_QUANT_KVCACHE_H