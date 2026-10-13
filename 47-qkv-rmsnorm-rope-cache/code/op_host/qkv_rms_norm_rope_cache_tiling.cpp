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
 * \file qkv_rms_norm_rope_cache_ds_tiling.cpp
 * \brief
 */
#include "qkv_rms_norm_rope_cache_tiling.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"

namespace optiling {

ge::graphStatus QkvRmsNormRopeCacheTilingDs::CheckQkvValid()
{
    auto qkvShapeTuple = GetShapeTupleOfTH(context_, QKV_INDEX); // 返回[B*Sqkv, N*D]
    bool isValid = true;
    isValid = isValid && (std::get<SHAPE_IDX_BS>(qkvShapeTuple) == batchSize_ * seqLength_);
    isValid = isValid && (std::get<SHAPE_IDX_ND>(qkvShapeTuple) == numHead_ * qkvDim_);
    OP_CHECK_IF(
        !isValid, OP_LOGE(context_->GetNodeName(), "the shape of qkv is incorrect."),
        return ge::GRAPH_FAILED);
    auto qkvDesc = context_->GetInputDesc(QKV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qkvDesc);
    qkvDtype_ = qkvDesc->GetDataType();
    OP_CHECK_IF(
        qkvDtype_ != ge::DataType::DT_FLOAT16 && qkvDtype_ != ge::DataType::DT_BF16, OP_LOGE(context_->GetNodeName(), "qkv only support float16 and bfloat16 now"),
        return ge::GRAPH_FAILED);
    qkvDtypeSize_ = FLOAT16_BYTES;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingDs::CheckGammaValid(int64_t gammaIdx)
{
    const gert::StorageShape* gammaShapePtr = context_->GetInputShape(gammaIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaShapePtr);
    bool isValid = true;
    isValid = isValid && (gammaShapePtr->GetStorageShape().GetDimNum() == 1);
    isValid = isValid && (gammaShapePtr->GetStorageShape().GetDim(0) == qkvDim_);
    OP_CHECK_IF(
        !isValid, OP_LOGE(context_->GetNodeName(), "the shape of gamma is invalid."),
        return ge::GRAPH_FAILED);

    auto gammaDesc = context_->GetInputDesc(gammaIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaDesc);
    auto gammaDtype = gammaDesc->GetDataType();
    OP_CHECK_IF(
        gammaDtype != qkvDtype_, OP_LOGE(context_->GetNodeName(), "gamma dtype must be the same with qkv."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingDs::CheckCosSinValid()
{
    auto cosShapeTuple = GetShapeTupleOfTH(context_, COS_INDEX);
    auto sinShapeTuple = GetShapeTupleOfTH(context_, SIN_INDEX);
    bool isValid = true;
    isValid = isValid && ((std::get<SHAPE_IDX_BS>(cosShapeTuple) == batchSize_ * seqLength_) 
                || (std::get<SHAPE_IDX_BS>(cosShapeTuple) == batchSize_ * NUM_ONE));
    isValid = isValid && (std::get<SHAPE_IDX_ND>(cosShapeTuple) == NUM_ONE * ropeRange_);
    isValid = isValid && ((std::get<SHAPE_IDX_BS>(sinShapeTuple) == batchSize_ * seqLength_)
                || (std::get<SHAPE_IDX_BS>(sinShapeTuple) == batchSize_ * NUM_ONE));
    isValid = isValid && (std::get<SHAPE_IDX_ND>(sinShapeTuple) == NUM_ONE * ropeRange_);
    auto cosSeq = std::get<SHAPE_IDX_BS>(cosShapeTuple);
    auto sinSeq = std::get<SHAPE_IDX_BS>(sinShapeTuple);
    isValid = isValid && (cosSeq == sinSeq);
    OP_CHECK_IF(
        !isValid, OP_LOGE(context_->GetNodeName(), "the shape of cos or sin is invalid."),
        return ge::GRAPH_FAILED);
    rope_seq_ = std::get<SHAPE_IDX_BS>(cosShapeTuple) / batchSize_;

    auto cosDesc = context_->GetInputDesc(COS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cosDesc);
    auto cosDtype = cosDesc->GetDataType();
    OP_CHECK_IF(
        cosDtype != qkvDtype_, OP_LOGE(context_->GetNodeName(), "cos dtype must be the same with qkv."),
        return ge::GRAPH_FAILED);

    auto sinDesc = context_->GetInputDesc(SIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sinDesc);
    auto sinDtype = sinDesc->GetDataType();
    OP_CHECK_IF(
        sinDtype != qkvDtype_, OP_LOGE(context_->GetNodeName(), "sin dtype must be the same with qkv."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingDs::CheckIndexValid()
{
    bool isValid = true;
    const gert::StorageShape* indexShapePtr = context_->GetInputShape(INDEX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indexShapePtr);
    isValid = isValid && (indexShapePtr->GetStorageShape().GetDimNum() == NUM_ONE);
    isValid = isValid && (indexShapePtr->GetStorageShape().GetShapeSize() == batchSize_ * seqLength_);
    OP_CHECK_IF(!isValid, OP_LOGE(context_->GetNodeName(), "Index's shape size must be equal to B*S."), return ge::GRAPH_FAILED);

    auto indexDesc = context_->GetInputDesc(INDEX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indexDesc);
    auto indexDtype = indexDesc->GetDataType();
    OP_CHECK_IF(indexDtype != ge::DataType::DT_INT64, OP_LOGE(context_->GetNodeName(), "index dtype must be int64."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingDs::CheckQOutValid()
{
    bool isValid = true;
    const gert::StorageShape* qOutShapePtr = context_->GetInputShape(Q_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qOutShapePtr);
    isValid = isValid && (qOutShapePtr->GetStorageShape().GetDimNum() == NUM_TWO);
    OP_CHECK_IF(!isValid, OP_LOGE(context_->GetNodeName(), "q_out must be 2D tensor."), return ge::GRAPH_FAILED);
    isValid = isValid && (qOutShapePtr->GetStorageShape().GetDim(0) == batchSize_ * seqLength_);
    isValid = isValid && (qOutShapePtr->GetStorageShape().GetDim(1) == numHeadQ_ * qkvDim_);
    OP_CHECK_IF(!isValid, OP_LOGE(context_->GetNodeName(), "the shape of q_out is invalid."), return ge::GRAPH_FAILED);

    auto qOutDesc = context_->GetInputDesc(Q_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qOutDesc);
    auto qOutDtype = qOutDesc->GetDataType();
    OP_CHECK_IF(qOutDtype != qkvDtype_, OP_LOGE(context_->GetNodeName(), "q_out dtype must be the same with qkv."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingDs::CheckKCacheValid()
{
    auto kCacheDesc = context_->GetInputDesc(K_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kCacheDesc);
    auto kcacheDtype = kCacheDesc->GetDataType();
    OP_CHECK_IF(kcacheDtype != qkvDtype_ && kcacheDtype != ge::DataType::DT_INT8 , OP_LOGE(context_->GetNodeName(), "k_cache dtype must either be the same with qkv, or be int8."),
        return ge::GRAPH_FAILED);
    isKQuant_ = kcacheDtype == ge::DataType::DT_INT8 ? 1 : 0;

    bool isValid = true;
    const gert::StorageShape* kCacheShapePtr = context_->GetInputShape(K_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kCacheShapePtr);
    isValid = isValid && (kCacheShapePtr->GetStorageShape().GetDimNum() == NUM_FOUR);
    OP_CHECK_IF(!isValid, OP_LOGE(context_->GetNodeName(), "k_cache must be 4D tensor."), return ge::GRAPH_FAILED);
    blockNum_ = kCacheShapePtr->GetStorageShape().GetDim(0);
    blockSize_ = kCacheShapePtr->GetStorageShape().GetDim(2);
    if (isKQuant_) {
        isValid = isValid && (kCacheShapePtr->GetStorageShape().GetDim(1) == numHeadK_ * qkvDim_ / INT8_BLOCK_ALIGN_NUM);
        isValid = isValid && (kCacheShapePtr->GetStorageShape().GetDim(3) == INT8_BLOCK_ALIGN_NUM);
    }
    else {
        isValid = isValid && (kCacheShapePtr->GetStorageShape().GetDim(1) == numHeadK_ * qkvDim_ / FP16_BLOCK_ALIGN_NUM);
        isValid = isValid && (kCacheShapePtr->GetStorageShape().GetDim(3) == FP16_BLOCK_ALIGN_NUM);
    }
    OP_CHECK_IF(blockNum_ < ((seqLength_ + blockSize_ - 1) / blockSize_ * batchSize_), OP_LOGE(context_->GetNodeName(), 
                "blockNum must be greater than or equal to Ceil(Sqkv / blockSize) * Bqkv."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!isValid, OP_LOGE(context_->GetNodeName(), "the shape of k_cache is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockSize_ % BASE_BLOCK_SIZE != 0, OP_LOGE(context_->GetNodeName(), "blockSize must be divisible by 32."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingDs::CheckVCacheValid()
{
    auto vCacheDesc = context_->GetInputDesc(V_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vCacheDesc);
    auto vcacheDtype = vCacheDesc->GetDataType();
    OP_CHECK_IF(vcacheDtype != qkvDtype_ && vcacheDtype != ge::DataType::DT_INT8 , OP_LOGE(context_->GetNodeName(), "v_cache dtype must either be the same with qkv, or be int8."),
        return ge::GRAPH_FAILED);
    isVQuant_ = vcacheDtype == ge::DataType::DT_INT8 ? 1 : 0;
    
    bool isValid = true;
    const gert::StorageShape* vCacheShapePtr = context_->GetInputShape(V_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vCacheShapePtr);
    isValid = isValid && (vCacheShapePtr->GetStorageShape().GetDimNum() == NUM_FOUR);
    OP_CHECK_IF(!isValid, OP_LOGE(context_->GetNodeName(), "v_cache must be 4D tensor."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(vCacheShapePtr->GetStorageShape().GetDim(0) != blockNum_, OP_LOGE(context_->GetNodeName(), "the blockNums of v_cache must be equal to that of k_cache."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(vCacheShapePtr->GetStorageShape().GetDim(2) != blockSize_, OP_LOGE(context_->GetNodeName(), "the blockSize of v_cache must be equal to that of k_cache."), return ge::GRAPH_FAILED);
    if (isVQuant_) {
        isValid = isValid && (vCacheShapePtr->GetStorageShape().GetDim(1) == numHeadV_ * qkvDim_ / INT8_BLOCK_ALIGN_NUM);
        isValid = isValid && (vCacheShapePtr->GetStorageShape().GetDim(3) == INT8_BLOCK_ALIGN_NUM);
    }
    else {
        isValid = isValid && (vCacheShapePtr->GetStorageShape().GetDim(1) == numHeadV_ * qkvDim_ / FP16_BLOCK_ALIGN_NUM);
        isValid = isValid && (vCacheShapePtr->GetStorageShape().GetDim(3) == FP16_BLOCK_ALIGN_NUM);
    }
    OP_CHECK_IF(!isValid, OP_LOGE(context_->GetNodeName(), "the shape of v_cache is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingDs::CheckKScaleValid()
{
    const gert::StorageShape* kScaleShapePtr = context_->GetOptionalInputShape(K_SCALE_IDX);
    OP_CHECK_IF(kScaleShapePtr == nullptr && isKQuant_ == 1, OP_LOGE(context_->GetNodeName(), "when k_Cache dtype is int8, k_scale cannot be None."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(kScaleShapePtr != nullptr && isKQuant_ == 0, OP_LOGE(context_->GetNodeName(), "when k_Cache dtype is not int8, k_scale must be None."), return ge::GRAPH_FAILED);
    if (kScaleShapePtr != nullptr) {
        OP_CHECK_IF(kScaleShapePtr->GetStorageShape().GetDimNum() != NUM_TWO, OP_LOGE(context_->GetNodeName(), "k_scale must be 2D tensor."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(kScaleShapePtr->GetStorageShape().GetDim(0) != numHeadK_, OP_LOGE(context_->GetNodeName(), "the first dimension of k_scale must be %ld.", numHeadK_), return ge::GRAPH_FAILED);
        OP_CHECK_IF(kScaleShapePtr->GetStorageShape().GetDim(1) != qkvDim_, OP_LOGE(context_->GetNodeName(), "the second dimension of k_scale must be %ld.", qkvDim_), return ge::GRAPH_FAILED);
        auto kScaleDesc = context_->GetOptionalInputDesc(K_SCALE_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, kScaleDesc);
        auto kscaleDtype = kScaleDesc->GetDataType();
        OP_CHECK_IF(kscaleDtype != ge::DataType::DT_FLOAT, OP_LOGE(context_->GetNodeName(), "k_scale dtype must be float32."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingDs::CheckVScaleValid()
{
    const gert::StorageShape* vScaleShapePtr = context_->GetOptionalInputShape(V_SCALE_IDX);
    OP_CHECK_IF(vScaleShapePtr == nullptr && isVQuant_ == 1, OP_LOGE(context_->GetNodeName(), "when v_Cache dtype is int8, v_scale cannot be None."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(vScaleShapePtr != nullptr && isVQuant_ == 0, OP_LOGE(context_->GetNodeName(), "when v_Cache dtype is not int8, v_scale must be None."), return ge::GRAPH_FAILED);
    if (vScaleShapePtr != nullptr) {
        OP_CHECK_IF(vScaleShapePtr->GetStorageShape().GetDimNum() != NUM_TWO, OP_LOGE(context_->GetNodeName(), "v_scale must be 2D tensor."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(vScaleShapePtr->GetStorageShape().GetDim(0) != numHeadV_, OP_LOGE(context_->GetNodeName(), "the first dimension of v_scale must be %ld.", numHeadV_), return ge::GRAPH_FAILED);
        OP_CHECK_IF(vScaleShapePtr->GetStorageShape().GetDim(1) != qkvDim_, OP_LOGE(context_->GetNodeName(), "the second dimension of v_scale must be %ld.", qkvDim_), return ge::GRAPH_FAILED);
        auto vScaleDesc = context_->GetOptionalInputDesc(V_SCALE_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, vScaleDesc);
        auto vscaleDtype = vScaleDesc->GetDataType();
        OP_CHECK_IF(vscaleDtype != ge::DataType::DT_FLOAT, OP_LOGE(context_->GetNodeName(), "v_scale dtype must be float32."),
            return ge::GRAPH_FAILED);    
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingDs::CheckKOffsetValid()
{
    const gert::StorageShape* kOffsetShapePtr = context_->GetOptionalInputShape(K_OFFSET_IDX);
    OP_CHECK_IF(kOffsetShapePtr != nullptr && isKQuant_ == 0, OP_LOGE(context_->GetNodeName(), "when k_Cache dtype is not int8, k_offset must be None."), return ge::GRAPH_FAILED);
    if (kOffsetShapePtr != nullptr) {
        auto kOffsetDesc = context_->GetOptionalInputDesc(K_OFFSET_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, kOffsetDesc);
        auto kOffsetDtype = kOffsetDesc->GetDataType();
        OP_CHECK_IF(kOffsetDtype != ge::DataType::DT_FLOAT, OP_LOGE(context_->GetNodeName(), "k_offset dtype must be float32."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingDs::CheckVOffsetValid()
{
    const gert::StorageShape* vOffsetShapePtr = context_->GetOptionalInputShape(V_OFFSET_IDX);
    OP_CHECK_IF(vOffsetShapePtr != nullptr && isVQuant_ == 0, OP_LOGE(context_->GetNodeName(), "when v_Cache dtype is not int8, v_offset must be None."), return ge::GRAPH_FAILED);
    if (vOffsetShapePtr != nullptr) {
        auto vOffsetDesc = context_->GetOptionalInputDesc(V_OFFSET_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, vOffsetDesc);
        auto vOffsetDtype = vOffsetDesc->GetDataType();
        OP_CHECK_IF(vOffsetDtype != ge::DataType::DT_FLOAT, OP_LOGE(context_->GetNodeName(), "v_offset dtype must be float32."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool QkvRmsNormRopeCacheTilingDs::IsCapable()
{
    return !isRegbase_;
}

ge::graphStatus QkvRmsNormRopeCacheTilingDs::GetShapeAttrsInfoInner()
{
    OP_CHECK_IF(
        context_ == nullptr, OP_LOGE(context_->GetNodeName(), "context_ can not be nullptr."), return ge::GRAPH_FAILED);
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto qkvSizePtr = attrs->GetListInt(QKV_SIZE_IDX);// 获取qkv_size
    OP_CHECK_NULL_WITH_CONTEXT(context_, qkvSizePtr);
    OP_CHECK_IF(
        qkvSizePtr->GetSize() != NUM_FOUR,
        OP_LOGE(context_->GetNodeName(), "Check qkv_size failed, the size of qkv_size must be 4."),
        return ge::GRAPH_FAILED);
    batchSize_ = qkvSizePtr->GetData()[SHAPE_IDX_B];
    seqLength_ = qkvSizePtr->GetData()[SHAPE_IDX_S];
    numHead_ = qkvSizePtr->GetData()[SHAPE_IDX_N];
    qkvDim_ = qkvSizePtr->GetData()[SHAPE_IDX_D];
    OP_CHECK_IF(
        batchSize_ <= 0 || seqLength_ <= 0 || numHead_ <= 0 || qkvDim_ <= 0,
        OP_LOGE(context_->GetNodeName(), "Check qkv_size failed, the value of qkv_size must > 0."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(qkvDim_ != 128, OP_LOGE(context_->GetNodeName(), "Dqkv must be 128."),
        return ge::GRAPH_FAILED);
    reciprocal_ = 1.0 / static_cast<float>(qkvDim_);
    auto headNumsPtr = attrs->GetListInt(HEAD_NUMS_IDX); // 获取Nq、Nk、Nv
    OP_CHECK_NULL_WITH_CONTEXT(context_, headNumsPtr);
    OP_CHECK_IF(
        headNumsPtr->GetSize() != 3,
        OP_LOGE(context_->GetNodeName(), "Check headNums failed, the size of headNums must be 3."),
        return ge::GRAPH_FAILED);
    numHeadQ_ = headNumsPtr->GetData()[DIM_ZERO];
    numHeadK_ = headNumsPtr->GetData()[DIM_ONE];
    numHeadV_ = headNumsPtr->GetData()[DIM_TWO];
    OP_CHECK_IF(
        numHeadQ_ <= 0 || numHeadK_ <= 0 || numHeadV_ <= 0,
        OP_LOGE(context_->GetNodeName(), "Check headNums failed, the value of headNums must > 0."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        numHead_ != numHeadQ_ + numHeadK_ + numHeadV_,
        OP_LOGE(context_->GetNodeName(), "Check headNums failed, N is not equal to Nq + Nk + Nv."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        numHeadK_ != numHeadV_,
        OP_LOGE(context_->GetNodeName(), "Check headNums failed, Nk is not equal to Nv."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        numHeadQ_ % numHeadK_ != 0,
        OP_LOGE(context_->GetNodeName(), "Check headNums failed, Nq is not mutiple of Nv."),
        return ge::GRAPH_FAILED);
    ropeRange_ = qkvDim_; // 获取ropeRange
    const float* epsilon = attrs->GetFloat(EPSILON_IDX); 
    epsilon_ = epsilon == nullptr ? 1e-6 : *epsilon; // 获取epsilon
    const char* tmpmode = attrs->GetStr(CACHE_MODE_IDX); // 获取cacheMode
    std::unordered_map<std::string, CacheMode> cacheModeMap = {
        {"PA_NZ", CacheMode::PA_NZ}
    };
    if(tmpmode == nullptr) {
        currentCacheMode_ = CacheMode::PA_NZ;
    }
    else {
        std::string cacheMode = tmpmode;
        if(cacheModeMap.find(cacheMode) != cacheModeMap.end()) {
            currentCacheMode_ = cacheModeMap[cacheMode];
        }
        else {
            OP_LOGE(context_->GetNodeName(), "Check CacheMode failed, CacheMode only support PA_NZ now.");
            return ge::GRAPH_FAILED;
        }
    }
    // 获取is_output_qkv
    const bool* isOutputQkv = attrs->GetBool(IS_OUTPUT_QKV_IDX);
    bool tmp = isOutputQkv == nullptr ? false : *isOutputQkv;
    isOutputQkv_ = tmp ? 1 : 0;
    // 校验并获取输入tensor
    OP_CHECK_IF(CheckQkvValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "qkv is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckGammaValid(GAMMA_Q_INDEX) == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "gamma_q is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckGammaValid(GAMMA_K_INDEX) == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "gamma_k is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckCosSinValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "cos or sin is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckIndexValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "index is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckQOutValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "qout is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckKCacheValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "k_cache is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckVCacheValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "v_cache is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckKScaleValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "k_scale is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckVScaleValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "v_scale is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckKOffsetValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "k_offset is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckVOffsetValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "v_offset is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheTilingDs::CalUbTiling()
{
    int64_t tokenSumK = batchSize_ * seqLength_ * numHeadK_;
    int64_t tokenSumV = batchSize_ * seqLength_ * numHeadV_;
    int64_t tokenSumQ = batchSize_ * seqLength_ * numHeadQ_;
    int64_t coreNumKv_ = coreNum_ * (numHeadK_ + numHeadV_) >= numHead_ ? coreNum_ * (numHeadK_ + numHeadV_) / numHead_ : NUM_TWO;
    int64_t coreNumK_ = coreNumKv_ == NUM_TWO ? NUM_ONE : static_cast<int64_t>(coreNumKv_ * FAC_K + NEAR_ONE);
    coreNumK_ = coreNumKv_ == coreNumK_ ? coreNumK_ - NUM_ONE : coreNumK_;
    int64_t coreNumV_ = coreNumKv_ == NUM_TWO ? NUM_ONE : coreNumKv_ - coreNumK_;
    blockFactorK_ = (tokenSumK + coreNumK_ - 1) / coreNumK_; // 头核处理K token数量
    blockFactorV_ = (tokenSumV + coreNumV_ - 1) / coreNumV_; // 头核处理V token数量
    blockDimK_ = (tokenSumK + blockFactorK_ - 1) / blockFactorK_; // K使用的核数
    blockDimV_ = (tokenSumV + blockFactorV_ - 1) / blockFactorV_; // V使用的核数

    int64_t coreNumQ_ = coreNum_ - blockDimK_ - blockDimV_;
    blockFactorQ_ = (tokenSumQ + coreNumQ_ - 1) / coreNumQ_; // 头核处理Q token数量
    blockDimQ_ = (tokenSumQ + blockFactorQ_ - 1) / blockFactorQ_; // Q使用的核数
    blockDim_ = blockDimQ_ + blockDimK_ + blockDimV_;

    int64_t gammaBuf = qkvDim_ * FLOAT32_BYTES;
    int64_t inQueueX = DOUBLE_BUFFER * qkvDim_ * qkvDtypeSize_;
    int64_t inQueueY = DOUBLE_BUFFER * qkvDim_ * qkvDtypeSize_;
    int64_t locBuf0 = qkvDim_ * FLOAT32_BYTES;
    int64_t locBuf1 = qkvDim_ * FLOAT32_BYTES;
    int64_t locBuf2 = qkvDim_ * FLOAT32_BYTES;
    int64_t locBuf3 = qkvDim_ * FLOAT32_BYTES;
    int64_t outQueue = DOUBLE_BUFFER * qkvDim_ * qkvDtypeSize_;

    int64_t spaceWithUbfactorQ = inQueueX + inQueueY + NUM_TWO * locBuf0 + locBuf1 + NUM_TWO * locBuf2 + locBuf3 + outQueue;
    int64_t numeratorQ = ubSize_ - UB_RESERVED_BYTES - gammaBuf;
    ubFactorQ_ = numeratorQ / spaceWithUbfactorQ;
    ubFactorK_ = ubFactorQ_;

    int64_t spaceWithUbfactorV = inQueueX + locBuf0 + locBuf1 + locBuf2 + locBuf3 + outQueue;
    int64_t numeratorV = ubSize_ - UB_RESERVED_BYTES;
    ubFactorV_ = numeratorV / spaceWithUbfactorV;
    ubFactor_ = ubFactorQ_;
}

ge::graphStatus QkvRmsNormRopeCacheTilingDs::DoOpTiling()
{
    // 参数校验与获取
    OP_CHECK_IF(
        GetShapeAttrsInfoInner() == ge::GRAPH_FAILED,
        OP_LOGE(context_->GetNodeName(), "GetShapeAttrsInfoInner failed."), return ge::GRAPH_FAILED);

    // 计算切分参数：两种切分策略，MTP一种，非MTP一种，先写非MTP：cos/sin[B*S, 1*D]
    CalUbTiling();
    OP_CHECK_IF(ubFactor_ < 1, OP_LOGE(context_->GetNodeName(), "token shape is too big."), return ge::GRAPH_FAILED);
    // 计算tilingkey
    tilingKey_ =  (static_cast<int64_t>(currentCacheMode_) + NUM_ONE);

    // 设置结构体参数
    tilingData_.set_batchSize(batchSize_);
    tilingData_.set_seqLength(seqLength_);
    tilingData_.set_numHead(numHead_);
    tilingData_.set_qkvDim(qkvDim_);
    tilingData_.set_ropeRange(ropeRange_);
    tilingData_.set_numHeadQ(numHeadQ_);
    tilingData_.set_numHeadK(numHeadK_);
    tilingData_.set_numHeadV(numHeadV_);
    tilingData_.set_blockNum(blockNum_);
    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_epsilon(epsilon_);
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_blockFactorQ(blockFactorQ_);
    tilingData_.set_blockFactorK(blockFactorK_);
    tilingData_.set_blockFactorV(blockFactorV_);
    tilingData_.set_blockDim(blockDim_);
    tilingData_.set_blockDimQ(blockDimQ_);
    tilingData_.set_blockDimK(blockDimK_);
    tilingData_.set_blockDimV(blockDimV_);
    tilingData_.set_ubFactor(ubFactor_);
    tilingData_.set_ubFactorQ(ubFactorQ_);
    tilingData_.set_ubFactorK(ubFactorK_);
    tilingData_.set_ubFactorV(ubFactorV_);
    tilingData_.set_reciprocal(reciprocal_);
    tilingData_.set_isOutputQkv(isOutputQkv_);
    tilingData_.set_isKQuant(isKQuant_);
    tilingData_.set_isVQuant(isVQuant_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingDs::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(tilingData_.get_blockDim());
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t sysWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(0UL + sysWorkSpaceSize);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheTilingDs::DumpTilingInfo()
{
    OP_LOGD(context_->GetNodeName(), "tilingKey_:    %ld", tilingKey_);
    OP_LOGD(context_->GetNodeName(), "coreNum_:  %ld", coreNum_);
    OP_LOGD(context_->GetNodeName(), "ubSize_:  %ld", ubSize_);
    OP_LOGD(context_->GetNodeName(), "isRegbase_:  %ld", static_cast<int64_t>(isRegbase_));
    OP_LOGD(context_->GetNodeName(), "rope_seq_:  %ld", rope_seq_);
    OP_LOGD(context_->GetNodeName(), "currentCacheMode_:  %ld", static_cast<int64_t>(currentCacheMode_));
    OP_LOGD(context_->GetNodeName(), "quantMode_:  %ld", quantMode_);
    OP_LOGD(context_->GetNodeName(), "batchSize_:  %ld", batchSize_);
    OP_LOGD(context_->GetNodeName(), "seqLength_:  %ld", seqLength_);
    OP_LOGD(context_->GetNodeName(), "numHead_:  %ld", numHead_);
    OP_LOGD(context_->GetNodeName(), "qkvDim_:  %ld", qkvDim_);
    OP_LOGD(context_->GetNodeName(), "ropeRange_:  %ld", ropeRange_);
    OP_LOGD(context_->GetNodeName(), "numHeadQ_:  %ld", numHeadQ_);
    OP_LOGD(context_->GetNodeName(), "numHeadK_:  %ld", numHeadK_);
    OP_LOGD(context_->GetNodeName(), "numHeadV_:  %ld", numHeadV_);
    OP_LOGD(context_->GetNodeName(), "blockNum_:  %ld", blockNum_);
    OP_LOGD(context_->GetNodeName(), "blockSize_:  %ld", blockSize_);
    OP_LOGD(context_->GetNodeName(), "epsilon_:  %f", epsilon_);
    OP_LOGD(context_->GetNodeName(), "blockFactor_:  %ld", blockFactor_);
    OP_LOGD(context_->GetNodeName(), "blockFactorQ_:  %ld", blockFactorQ_);
    OP_LOGD(context_->GetNodeName(), "blockFactorK_:  %ld", blockFactorK_);
    OP_LOGD(context_->GetNodeName(), "blockFactorV_:  %ld", blockFactorV_);
    OP_LOGD(context_->GetNodeName(), "blockDim_:  %ld", blockDim_);
    OP_LOGD(context_->GetNodeName(), "blockDimQ_:  %ld", blockDimQ_);
    OP_LOGD(context_->GetNodeName(), "blockDimK_:  %ld", blockDimK_);
    OP_LOGD(context_->GetNodeName(), "blockDimV_:  %ld", blockDimV_);
    OP_LOGD(context_->GetNodeName(), "ubFactor_:  %ld", ubFactor_);
    OP_LOGD(context_->GetNodeName(), "ubFactorQ_:  %ld", ubFactorQ_);
    OP_LOGD(context_->GetNodeName(), "ubFactorK_:  %ld", ubFactorK_);
    OP_LOGD(context_->GetNodeName(), "ubFactorV_:  %ld", ubFactorV_);
    OP_LOGD(context_->GetNodeName(), "reciprocal_:  %f", reciprocal_);
    OP_LOGD(context_->GetNodeName(), "isOutputQkv_:  %ld", isOutputQkv_);
    OP_LOGD(context_->GetNodeName(), "isKQuant_:  %ld",    isKQuant_);
    OP_LOGD(context_->GetNodeName(), "isVQuant_:  %ld",    isVQuant_);
}
REGISTER_OPS_TILING_TEMPLATE(QkvRmsNormRopeCache, QkvRmsNormRopeCacheTilingDs, TEMPLATE_DS_PRIORITY);
} // namespace optiling
