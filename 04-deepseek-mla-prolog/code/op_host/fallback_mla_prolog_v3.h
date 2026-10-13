/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FALLBACK_MLA_PROLOG_V3_H
#define FALLBACK_MLA_PROLOG_V3_H

#include <vector>
#include "log/log.h"
#include "fallback/fallback_comm.h"
#include "fallback/fallback.h"
#include "../../mla_prolog/op_host/fallback_mla_prolog.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

constexpr size_t DEQUANT_SCALE_Q_NOPE_INDEX = 4;
constexpr size_t QUERY_NORM_INDEX = 5;
constexpr size_t DEQUANT_SCALE_Q_NORM_INDEX = 6;

// INPUT
constexpr size_t TOKEN_X_INDEX_V3 = 0;
constexpr size_t WEIGHT_DQ_INDEX_V3 = 1;
constexpr size_t WEIGHT_UQ_QR_INDEX_V3 = 2;
constexpr size_t WEIGHT_UK_INDEX_V3 = 3;
constexpr size_t WEIGHT_DKV_KR_INDEX_V3 = 4;
constexpr size_t RMSNORM_GAMMA_CQ_INDEX_V3 = 5;
constexpr size_t RMSNORM_GAMMA_CKV_INDEX_V3 = 6;
constexpr size_t ROPE_SIN_INDEX_V3 = 7;
constexpr size_t ROPE_COS_INDEX_V3 = 8;
constexpr size_t KV_CACHE_INDEX_V3 = 9;
constexpr size_t KR_CACHE_INDEX_V3 = 10;
constexpr size_t CACHE_INDEX_V3 = 11;
constexpr size_t DEQUANT_SCALE_X_INDEX_V3 = 12;
constexpr size_t DEQUANT_SCALE_W_DQ_INDEX_V3 = 13;
constexpr size_t DEQUANT_SCALE_W_UQ_QR_INDEX_V3 = 14;
constexpr size_t DEQUANT_SCALE_W_DKV_KR_INDEX_V3 = 15;
constexpr size_t QUANT_SCALE_CKV_INDEX_V3 = 16;
constexpr size_t QUANT_SCALE_CKR_INDEX_V3 = 17;
constexpr size_t SMOOTH_SCALES_CQ_INDEX_V3 = 18;
constexpr uint32_t ACTUAL_SEQ_LEN_INDEX_V3 = 19;
constexpr uint32_t K_NOPE_CLIP_ALPHA_INDEX_V3 = 20;

constexpr size_t ATTR_QUERY_NORM_FLAG_INDEX = 3;
constexpr size_t ATTR_WEIGHT_QUANT_MODE_INDEX = 4;
constexpr size_t ATTR_KV_CACHE_QUANT_MODE_INDEX = 5;
constexpr size_t ATTR_QUERY_QUANT_MODE_INDEX = 6;
constexpr size_t ATTR_CKVKR_REPO_MODE_INDEX = 7;
constexpr size_t ATTR_QUANT_SCALE_REPO_MODE_INDEX = 8;
constexpr size_t ATTR_TILE_SIZE_INDEX = 9;
constexpr size_t ATTR_QC_QR_SCALE_INDEX = 10;
constexpr size_t ATTR_KC_SCALE_INDEX = 11;

struct MlaPrologV3FallBackParam : MlaPrologFallBackParam {
    const gert::Tensor *actualAeqLen = nullptr;
    const gert::Tensor *kNopeClipAlpha = nullptr;
    const gert::Tensor *dequantScaleQNope = nullptr;
    const gert::Tensor *queryNorm = nullptr;
    const gert::Tensor *dequantScaleQNorm = nullptr;
    bool queryNormFlag = false;
    int weightQuantMode = 0;
    int kvCacheQuantMode = 0;
    int queryQuantMode = 0;
    int ckvkrRepoMode = 0;
    int quantScaleRepoMode = 0;
    int tileSize = 128;
    double qcQrScale = 1.0f;
    double kcScale = 1.0f;
};

graphStatus GetMlaPrologV3OutputTensor(const OpExecuteContext *ctx, MlaPrologV3FallBackParam &param);
graphStatus MlaV3HostExecuteFunc(OpExecuteContext *host_api_ctx);

} // namespace fallback

#ifdef __cplusplus
}
#endif

#endif // FALLBACK_MLA_PROLOG_V3_H