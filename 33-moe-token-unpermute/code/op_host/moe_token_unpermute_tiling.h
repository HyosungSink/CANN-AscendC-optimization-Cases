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
 * \file moe_token_unpermute_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_UNPERMUTE_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_UNPERMUTE_H

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "log/log.h"

namespace optiling {

constexpr int64_t FLOAT_DATA_SIZE = 4;
constexpr int64_t MIN_BUFFER_NUM = 2;
constexpr int64_t ALIGN_512 = 512;
constexpr int64_t ALIGN_256 = 256;
constexpr int64_t QUE_NUM = 2;
constexpr int64_t CAST_NUM = 2;
constexpr int64_t TILINGKEY_PROBS = 1;
constexpr int64_t TILINGKEY_FLOAT16 = 2;
constexpr int64_t TILINGKEY_FLOAT = 4;
constexpr int64_t TILINGKEY_MIX_BF16 = 8;
constexpr int64_t TILINGKEY_MIX_FP16 = 16;
constexpr int64_t TILINGKEY_MIX_FP32 = 24;
constexpr int64_t TILINGKEY_DROPPAD = 1000;
constexpr int64_t MAX_HIDDEN_SIZE_310P = 10240;
constexpr int64_t ALIGN_128 = 128;
struct CoreParam {
    int64_t maxCoreMemery = 0;
    int64_t maxCoreNum = 0;
    int64_t usedCoreNum = 0;
    int64_t remainMemerySpace = 0;
    int64_t bufferNum = 0;
    int64_t tilingKey = 0;
};

struct InputParam {
    int64_t tokensNum = 0;
    int64_t topK = 0;
    int64_t hiddenSize = 0;
    int64_t totalLength = 0;
    int64_t numOutTokens = 0;
    int64_t tokensDtypeSize = 0;
    int64_t indicesDtypeSize = 0;
    int64_t probsDtypeSize = 0;
    bool haveProbs = false;
};

struct TilingParam {
    int64_t length = 0;
    int64_t num = 0;
    int64_t remain = 0;
};

struct MoeTokenUnpermuteParam {
    InputParam input;
    TilingParam hiddenTiling;
    TilingParam tokenTiling;
    TilingParam tokenPerCore;
    CoreParam core;
};
ge::graphStatus TilingCompute(gert::TilingContext* context, const int64_t topK);
BEGIN_TILING_DATA_DEF(MoeTokenUnpermuteTilingData)
TILING_DATA_FIELD_DEF(int64_t, hidden_size);
TILING_DATA_FIELD_DEF(int64_t, top_k);
TILING_DATA_FIELD_DEF(int64_t, num_out_tokens);

TILING_DATA_FIELD_DEF(int64_t, hidden_splited_length);
TILING_DATA_FIELD_DEF(int64_t, hidden_splited_num);
TILING_DATA_FIELD_DEF(int64_t, hidden_splited_remain);

TILING_DATA_FIELD_DEF(int64_t, tokens_core_length);
TILING_DATA_FIELD_DEF(int64_t, tokens_core_remain);
TILING_DATA_FIELD_DEF(int64_t, tokens_splited_length);
TILING_DATA_FIELD_DEF(int64_t, tokens_splited_num);
TILING_DATA_FIELD_DEF(int64_t, tokens_splited_remain);

TILING_DATA_FIELD_DEF(int64_t, buffer_num);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeTokenUnpermute, MoeTokenUnpermuteTilingData)
struct MoeTokenUnpermuteCompileInfo {
};

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_UNPERMUTE_H
