/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fallback_mla_prolog.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

using namespace ge;
using namespace gert;

graphStatus GetMlaPrologInputTensor(const OpExecuteContext *ctx, MlaPrologFallBackParam &param)
{
    param.tokenX = ctx->GetRequiredInputTensor(TOKEN_X_INDEX);
    OP_CHECK_IF(param.tokenX == nullptr, OP_LOGE("aclnnfallback", "tokenX is null"), return GRAPH_FAILED);

    param.weightDq = ctx->GetRequiredInputTensor(WEIGHT_DQ_INDEX);
    OP_CHECK_IF(param.weightDq == nullptr, OP_LOGE("aclnnfallback", "weightDq is null"), return GRAPH_FAILED);

    param.weightUqQr = ctx->GetRequiredInputTensor(WEIGHT_UQ_QR_INDEX);
    OP_CHECK_IF(param.weightUqQr == nullptr, OP_LOGE("aclnnfallback", "weightUqQr is null"), return GRAPH_FAILED);

    param.weightUk = ctx->GetRequiredInputTensor(WEIGHT_UK_INDEX);
    OP_CHECK_IF(param.weightUk == nullptr, OP_LOGE("aclnnfallback", "weightUk is null"), return GRAPH_FAILED);

    param.weightDkvKr = ctx->GetRequiredInputTensor(WEIGHT_DKV_KR_INDEX);
    OP_CHECK_IF(param.weightDkvKr == nullptr, OP_LOGE("aclnnfallback", "weightDkvKr is null"), return GRAPH_FAILED);
    
    param.rmsnormGammaCq = ctx->GetRequiredInputTensor(RMSNORM_GAMMA_CQ_INDEX);
    OP_CHECK_IF(param.rmsnormGammaCq == nullptr, OP_LOGE("aclnnfallback", "rmsnormGammaCq is null"),
        return GRAPH_FAILED);

    param.rmsnormGammaCkv = ctx->GetRequiredInputTensor(RMSNORM_GAMMA_CKV_INDEX);
    OP_CHECK_IF(param.rmsnormGammaCkv == nullptr, OP_LOGE("aclnnfallback", "rmsnormGammaCkv is null"), 
        return GRAPH_FAILED);

    param.ropeSin = ctx->GetRequiredInputTensor(ROPE_SIN_INDEX);
    OP_CHECK_IF(param.ropeSin == nullptr, OP_LOGE("aclnnfallback", "ropeSin is null"), return GRAPH_FAILED);

    param.ropeCos = ctx->GetRequiredInputTensor(ROPE_COS_INDEX);
    OP_CHECK_IF(param.ropeCos == nullptr, OP_LOGE("aclnnfallback", "ropeCos is null"), return GRAPH_FAILED);

    param.cacheIndex = ctx->GetRequiredInputTensor(CACHE_INDEX_INDEX);
    OP_CHECK_IF(param.cacheIndex == nullptr, OP_LOGE("aclnnfallback", "cacheIndex is null"), return GRAPH_FAILED);

    param.kvCache = ctx->GetRequiredInputTensor(KV_CACHE_INDEX);
    OP_CHECK_IF(param.kvCache == nullptr, OP_LOGE("aclnnfallback", "kvCache is null"), return GRAPH_FAILED);

    param.krCache = ctx->GetRequiredInputTensor(KR_CACHE_INDEX);
    OP_CHECK_IF(param.krCache == nullptr, OP_LOGE("aclnnfallback", "krCache is null"), return GRAPH_FAILED);

    param.dequantScaleX = ctx->GetOptionalInputTensor(DEQUANT_SCALE_X_INDEX);
    param.dequantScaleWDq = ctx->GetOptionalInputTensor(DEQUANT_SCALE_W_DQ_INDEX);
    param.dequantScaleWUqQr = ctx->GetOptionalInputTensor(DEQUANT_SCALE_W_UQ_QR_INDEX);
    param.dequantScaleWDkvKr = ctx->GetOptionalInputTensor(DEQUANT_SCALE_W_DKV_KR_INDEX);
    param.quantScaleCkv = ctx->GetOptionalInputTensor(QUANT_SCALE_CKV_INDEX);
    param.quantScaleCkr = ctx->GetOptionalInputTensor(QUANT_SCALE_CKR_INDEX);
    param.smoothScalesCq = ctx->GetOptionalInputTensor(SMOOTH_SCALES_CQ_INDEX);

    return GRAPH_SUCCESS;
}

graphStatus GetMlaPrologOutputTensor(const OpExecuteContext *ctx, MlaPrologFallBackParam &param)
{
    param.query = ctx->GetOutputTensor(QUERY_INDEX);
    OP_CHECK_IF(param.query == nullptr, OP_LOGE("aclnnfallback", "query is null"), return GRAPH_FAILED);

    param.queryRope = ctx->GetOutputTensor(QUERY_ROPE_INDEX);
    OP_CHECK_IF(param.queryRope == nullptr, OP_LOGE("aclnnfallback", "queryRope is null"), return GRAPH_FAILED);

    param.kvCacheOut = ctx->GetOutputTensor(KV_CACHE_OUT_INDEX);
    OP_CHECK_IF(param.kvCacheOut == nullptr, OP_LOGE("aclnnfallback", "kvCacheOut is null"), return GRAPH_FAILED);

    param.krCacheOut = ctx->GetOutputTensor(KR_CACHE_OUT_INDEX);
    OP_CHECK_IF(param.krCacheOut == nullptr, OP_LOGE("aclnnfallback", "krCacheOut is null"), return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

graphStatus GetMlaPrologAttr(const OpExecuteContext *ctx, MlaPrologFallBackParam &param)
{
    auto attrs = ctx->GetAttrs();
    const double *getRmsnormEpsilonCq = attrs->GetAttrPointer<double>(ATTR_RMSNORM_EPSILON_CQ_INDEX);
    const double *getRmsnormEpsilonCkv = attrs->GetAttrPointer<double>(ATTR_RMSNORM_EPSILON_CKV_INDEX);
    const char *getCacheMode = attrs->GetAttrPointer<char>(ATTR_CACHE_MODE_INDEX);

    param.rmsnormEpsilonCq = *getRmsnormEpsilonCq;
    param.rmsnormEpsilonCkv = *getRmsnormEpsilonCkv;
    param.cacheMode = getCacheMode;

    OP_LOGD("aclnnFallback", "MlaProlog fallback begin, RmsnormEpsilonCq = %f, RmsnormEpsilonCkv = %f",
              param.rmsnormEpsilonCq, param.rmsnormEpsilonCkv);

    return GRAPH_SUCCESS;
}

graphStatus MlaHostExecuteFunc(OpExecuteContext *host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

    MlaPrologFallBackParam param {};
    auto apiRet = GetMlaPrologInputTensor(host_api_ctx, param);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "Context get input tesnor failed"),
        return GRAPH_FAILED);

    apiRet = GetMlaPrologOutputTensor(host_api_ctx, param);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "Context get output tesnor failed"),
        return GRAPH_FAILED);

    apiRet = GetMlaPrologAttr(host_api_ctx, param);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "Context get attr failed"),
        return GRAPH_FAILED);

    apiRet = EXEC_OPAPI_CMD(
        aclnnMlaProlog, param.tokenX, param.weightDq, param.weightUqQr, param.weightUk, param.weightDkvKr,
        param.rmsnormGammaCq, param.rmsnormGammaCkv, param.ropeSin, param.ropeCos, param.cacheIndex,
        param.kvCache, param.krCache, param.dequantScaleX, param.dequantScaleWDq, param.dequantScaleWUqQr,
        param.dequantScaleWDkvKr, param.quantScaleCkv, param.quantScaleCkr, param.smoothScalesCq,
        param.rmsnormEpsilonCq, param.rmsnormEpsilonCkv, param.cacheMode,
        param.query, param.queryRope, param.kvCacheOut, param.krCacheOut);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "ret failed:%u", apiRet),
        return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(MlaProlog).OpExecuteFunc(MlaHostExecuteFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif