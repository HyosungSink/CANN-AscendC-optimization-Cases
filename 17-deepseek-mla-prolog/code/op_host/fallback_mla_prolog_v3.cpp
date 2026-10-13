/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fallback_mla_prolog_v3.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

using namespace ge;
using namespace gert;

graphStatus GetMlaPrologV3InputTensor(const OpExecuteContext *ctx, MlaPrologV3FallBackParam &param)
{
    param.tokenX = ctx->GetRequiredInputTensor(TOKEN_X_INDEX_V3);
    OPS_ERR_IF(param.tokenX == nullptr, OPS_LOG_E("aclnnfallback", "tokenX is null"), return GRAPH_FAILED);

    param.weightDq = ctx->GetRequiredInputTensor(WEIGHT_DQ_INDEX_V3);
    OPS_ERR_IF(param.weightDq == nullptr, OPS_LOG_E("aclnnfallback", "weightDq is null"), return GRAPH_FAILED);

    param.weightUqQr = ctx->GetRequiredInputTensor(WEIGHT_UQ_QR_INDEX_V3);
    OPS_ERR_IF(param.weightUqQr == nullptr, OPS_LOG_E("aclnnfallback", "weightUqQr is null"), return GRAPH_FAILED);

    param.weightUk = ctx->GetRequiredInputTensor(WEIGHT_UK_INDEX_V3);
    OPS_ERR_IF(param.weightUk == nullptr, OPS_LOG_E("aclnnfallback", "weightUk is null"), return GRAPH_FAILED);

    param.weightDkvKr = ctx->GetRequiredInputTensor(WEIGHT_DKV_KR_INDEX_V3);
    OPS_ERR_IF(param.weightDkvKr == nullptr, OPS_LOG_E("aclnnfallback", "weightDkvKr is null"), return GRAPH_FAILED);
    
    param.rmsnormGammaCq = ctx->GetRequiredInputTensor(RMSNORM_GAMMA_CQ_INDEX_V3);
    OPS_ERR_IF(param.rmsnormGammaCq == nullptr, OPS_LOG_E("aclnnfallback", "rmsnormGammaCq is null"),
        return GRAPH_FAILED);

    param.rmsnormGammaCkv = ctx->GetRequiredInputTensor(RMSNORM_GAMMA_CKV_INDEX_V3);
    OPS_ERR_IF(param.rmsnormGammaCkv == nullptr, OPS_LOG_E("aclnnfallback", "rmsnormGammaCkv is null"), 
        return GRAPH_FAILED);

    param.ropeSin = ctx->GetRequiredInputTensor(ROPE_SIN_INDEX_V3);
    OPS_ERR_IF(param.ropeSin == nullptr, OPS_LOG_E("aclnnfallback", "ropeSin is null"), return GRAPH_FAILED);

    param.ropeCos = ctx->GetRequiredInputTensor(ROPE_COS_INDEX_V3);
    OPS_ERR_IF(param.ropeCos == nullptr, OPS_LOG_E("aclnnfallback", "ropeCos is null"), return GRAPH_FAILED);

    param.kvCache = ctx->GetRequiredInputTensor(KV_CACHE_INDEX_V3);
    OPS_ERR_IF(param.kvCache == nullptr, OPS_LOG_E("aclnnfallback", "kvCache is null"), return GRAPH_FAILED);

    param.krCache = ctx->GetRequiredInputTensor(KR_CACHE_INDEX_V3);
    OPS_ERR_IF(param.krCache == nullptr, OPS_LOG_E("aclnnfallback", "krCache is null"), return GRAPH_FAILED);

    param.cacheIndex = ctx->GetOptionalInputTensor(CACHE_INDEX_V3);
    param.dequantScaleX = ctx->GetOptionalInputTensor(DEQUANT_SCALE_X_INDEX_V3);
    param.dequantScaleWDq = ctx->GetOptionalInputTensor(DEQUANT_SCALE_W_DQ_INDEX_V3);
    param.dequantScaleWUqQr = ctx->GetOptionalInputTensor(DEQUANT_SCALE_W_UQ_QR_INDEX_V3);
    param.dequantScaleWDkvKr = ctx->GetOptionalInputTensor(DEQUANT_SCALE_W_DKV_KR_INDEX_V3);
    param.quantScaleCkv = ctx->GetOptionalInputTensor(QUANT_SCALE_CKV_INDEX_V3);
    param.quantScaleCkr = ctx->GetOptionalInputTensor(QUANT_SCALE_CKR_INDEX_V3);
    param.smoothScalesCq = ctx->GetOptionalInputTensor(SMOOTH_SCALES_CQ_INDEX_V3);
    param.actualAeqLen = ctx->GetOptionalInputTensor(ACTUAL_SEQ_LEN_INDEX_V3);
    param.kNopeClipAlpha = ctx->GetOptionalInputTensor(K_NOPE_CLIP_ALPHA_INDEX_V3);

    return GRAPH_SUCCESS;
}

graphStatus GetMlaPrologV3OutputTensor(const OpExecuteContext *ctx, MlaPrologV3FallBackParam &param)
{
    auto ret = GetMlaPrologOutputTensor(ctx, param);
    OP_CHECK_IF(ret != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "GetOutputTensor failed"), return GRAPH_FAILED);

    param.dequantScaleQNope = ctx->GetOutputTensor(DEQUANT_SCALE_Q_NOPE_INDEX);
    OP_CHECK_IF(param.dequantScaleQNope == nullptr, OP_LOGE("aclnnfallback", "dequantScaleQNope is null"), 
        return GRAPH_FAILED);

    param.queryNorm = ctx->GetOutputTensor(QUERY_NORM_INDEX);
    OP_CHECK_IF(param.queryNorm == nullptr, OP_LOGE("aclnnfallback", "queryNorm is null"), 
        return GRAPH_FAILED);

    param.dequantScaleQNorm = ctx->GetOutputTensor(DEQUANT_SCALE_Q_NORM_INDEX);
    OP_CHECK_IF(param.dequantScaleQNorm == nullptr, OP_LOGE("aclnnfallback", "dequantScaleQNorm is null"), 
        return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

graphStatus GetMlaPrologV3Attr(const OpExecuteContext *ctx, MlaPrologV3FallBackParam &param)
{
    auto ret = GetMlaPrologAttr(ctx, param);
    OP_CHECK_IF(ret != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "GetAttr failed"), return GRAPH_FAILED);
    const bool *getQueryNormFlag = ctx->GetAttrs()->GetAttrPointer<bool>(ATTR_QUERY_NORM_FLAG_INDEX);
    const int *getWeightQuantMode = ctx->GetAttrs()->GetAttrPointer<int>(ATTR_WEIGHT_QUANT_MODE_INDEX);
    const int *getKvCacheQuantMode = ctx->GetAttrs()->GetAttrPointer<int>(ATTR_KV_CACHE_QUANT_MODE_INDEX);
    const int *getQueryQuantMode = ctx->GetAttrs()->GetAttrPointer<int>(ATTR_QUERY_QUANT_MODE_INDEX);
    const int *getCkvkrRepoMode = ctx->GetAttrs()->GetAttrPointer<int>(ATTR_CKVKR_REPO_MODE_INDEX);
    const int *getQuantScaleRepoMode = ctx->GetAttrs()->GetAttrPointer<int>(ATTR_QUANT_SCALE_REPO_MODE_INDEX);
    const int *getTileSize = ctx->GetAttrs()->GetAttrPointer<int>(ATTR_TILE_SIZE_INDEX);
    const double *getQcQrScale = ctx->GetAttrs()->GetAttrPointer<double>(ATTR_QC_QR_SCALE_INDEX);
    const double *getKcScale = ctx->GetAttrs()->GetAttrPointer<double>(ATTR_KC_SCALE_INDEX);
    param.queryNormFlag = *getQueryNormFlag;
    param.weightQuantMode = *getWeightQuantMode;
    param.kvCacheQuantMode = *getKvCacheQuantMode;
    param.queryQuantMode = *getQueryQuantMode;
    param.ckvkrRepoMode = *getCkvkrRepoMode;
    param.quantScaleRepoMode = *getQuantScaleRepoMode;
    param.tileSize = *getTileSize;
    param.qcQrScale = getQcQrScale ? *getQcQrScale : 1.0;
    param.kcScale = getKcScale ? *getKcScale : 1.0;
    return GRAPH_SUCCESS;
}

graphStatus MlaV3HostExecuteFunc(OpExecuteContext *host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

    MlaPrologV3FallBackParam param {};
    auto apiRet = GetMlaPrologV3InputTensor(host_api_ctx, param);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "Context get input tesnor failed"),
        return GRAPH_FAILED);

    apiRet = GetMlaPrologV3OutputTensor(host_api_ctx, param);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "Context get output tesnor failed"),
        return GRAPH_FAILED);

    apiRet = GetMlaPrologV3Attr(host_api_ctx, param);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "Context get attr failed"),
        return GRAPH_FAILED);

    apiRet = EXEC_OPAPI_CMD(
        aclnnMlaPrologV3WeightNz, param.tokenX, param.weightDq, param.weightUqQr, param.weightUk, param.weightDkvKr,
        param.rmsnormGammaCq, param.rmsnormGammaCkv, param.ropeSin, param.ropeCos,
        param.kvCache, param.krCache, param.cacheIndex, // cacheIndex当前为可选参数
        param.dequantScaleX, param.dequantScaleWDq, param.dequantScaleWUqQr,
        param.dequantScaleWDkvKr, param.quantScaleCkv, param.quantScaleCkr, param.smoothScalesCq, param.actualAeqLen, param.kNopeClipAlpha, 
        param.rmsnormEpsilonCq, param.rmsnormEpsilonCkv, param.cacheMode,
        param.weightQuantMode, param.kvCacheQuantMode, param.queryQuantMode, param.ckvkrRepoMode, param.quantScaleRepoMode,
        param.tileSize, param.qcQrScale, param.kcScale,
        param.query, param.queryRope, param.kvCacheOut, param.krCacheOut,
        param.dequantScaleQNope, param.queryNorm, param.dequantScaleQNorm);

    OP_CHECK_IF(apiRet != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "ret failed:%u", apiRet),
        return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(MlaPrologV3).OpExecuteFunc(MlaV3HostExecuteFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif