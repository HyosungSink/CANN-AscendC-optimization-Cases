/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FALLBACK_MLA_PROLOG_H
#define FALLBACK_MLA_PROLOG_H

#include <vector>
#include "log/log.h"
#include "fallback/fallback_comm.h"
#include "fallback/fallback.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

// INPUT
constexpr size_t TOKEN_X_INDEX = 0;
constexpr size_t WEIGHT_DQ_INDEX = 1;
constexpr size_t WEIGHT_UQ_QR_INDEX = 2;
constexpr size_t WEIGHT_UK_INDEX = 3;
constexpr size_t WEIGHT_DKV_KR_INDEX = 4;
constexpr size_t RMSNORM_GAMMA_CQ_INDEX = 5;
constexpr size_t RMSNORM_GAMMA_CKV_INDEX = 6;
constexpr size_t ROPE_SIN_INDEX = 7;
constexpr size_t ROPE_COS_INDEX = 8;
constexpr size_t CACHE_INDEX_INDEX = 9;
constexpr size_t KV_CACHE_INDEX = 10;
constexpr size_t KR_CACHE_INDEX = 11;
constexpr size_t DEQUANT_SCALE_X_INDEX = 12;
constexpr size_t DEQUANT_SCALE_W_DQ_INDEX = 13;
constexpr size_t DEQUANT_SCALE_W_UQ_QR_INDEX = 14;
constexpr size_t DEQUANT_SCALE_W_DKV_KR_INDEX = 15;
constexpr size_t QUANT_SCALE_CKV_INDEX = 16;
constexpr size_t QUANT_SCALE_CKR_INDEX = 17;
constexpr size_t SMOOTH_SCALES_CQ_INDEX = 18;
// ATTR
constexpr size_t ATTR_RMSNORM_EPSILON_CQ_INDEX = 0;
constexpr size_t ATTR_RMSNORM_EPSILON_CKV_INDEX = 1;
constexpr size_t ATTR_CACHE_MODE_INDEX = 2;
// OUTPUT
constexpr size_t QUERY_INDEX = 0;
constexpr size_t QUERY_ROPE_INDEX = 1;
constexpr size_t KV_CACHE_OUT_INDEX = 2;
constexpr size_t KR_CACHE_OUT_INDEX = 3;

struct MlaPrologFallBackParam {
    const gert::Tensor *tokenX = nullptr;
    const gert::Tensor *weightDq = nullptr;
    const gert::Tensor *weightUqQr = nullptr;
    const gert::Tensor *weightUk = nullptr;
    const gert::Tensor *weightDkvKr = nullptr;
    const gert::Tensor *rmsnormGammaCq = nullptr;
    const gert::Tensor *rmsnormGammaCkv = nullptr;
    const gert::Tensor *ropeSin = nullptr;
    const gert::Tensor *ropeCos = nullptr;
    const gert::Tensor *cacheIndex = nullptr;
    const gert::Tensor *kvCache = nullptr;
    const gert::Tensor *krCache = nullptr;
    const gert::Tensor *query = nullptr;
    const gert::Tensor *queryRope = nullptr;
    const gert::Tensor *kvCacheOut = nullptr;
    const gert::Tensor *krCacheOut = nullptr;
    const gert::Tensor *dequantScaleX = nullptr;
    const gert::Tensor *dequantScaleWDq = nullptr;
    const gert::Tensor *dequantScaleWUqQr = nullptr;
    const gert::Tensor *dequantScaleWDkvKr = nullptr;
    const gert::Tensor *quantScaleCkv = nullptr;
    const gert::Tensor *quantScaleCkr = nullptr;
    const gert::Tensor *smoothScalesCq = nullptr;
    double rmsnormEpsilonCq = 1e-05f;
    double rmsnormEpsilonCkv = 1e-05f;
    const char *cacheMode = nullptr;
};

graphStatus GetMlaPrologInputTensor(const OpExecuteContext *ctx, MlaPrologFallBackParam &param);
graphStatus GetMlaPrologOutputTensor(const OpExecuteContext *ctx, MlaPrologFallBackParam &param);
graphStatus GetMlaPrologAttr(const OpExecuteContext *ctx, MlaPrologFallBackParam &param);
graphStatus MlaHostExecuteFunc(OpExecuteContext *host_api_ctx);
} // namespace fallback

#ifdef __cplusplus
}
#endif

#endif // FALLBACK_MLA_PROLOG_H