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
 * \file layer_norm_custom_align_tiling.cpp
 * \brief
 */

#include "layer_norm_v4_tiling.h"


namespace optiling {
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t SIZE_OF_FLOAT = 4;
constexpr uint64_t SIZE_OF_FLOAT16 = 2;
constexpr uint64_t BLOCK_SIZE_OF_FLOAT16 = 16;
constexpr uint64_t BASE_WSP_SIZE = 32;
constexpr uint64_t NUM_TWO = 2;
constexpr uint64_t NUM_TEN = 10;
constexpr uint64_t BASE_KEY = 700U;

constexpr uint32_t SYS_WORKSPACESIZE = 16 * 1024 * 1024;
constexpr uint32_t NORMAL_UB_SIZE = 128 * 1024;
constexpr uint32_t NEED_UB_SIZE = 188 * 1024;
constexpr uint32_t OTHERS_UB_SIZE = 3 * 1024;
constexpr uint32_t TMP_UB_SIZE = 1024;
constexpr uint32_t MEAN_UB_SIZE = 512;

bool LayerNormV4CommonTiling::IsCapable()
{
    if (commonParams.isRegBase || commonParams.isAscend310P || commonParams.isV3) {
        return false;
    }
    return true;
}

uint64_t LayerNormV4CommonTiling::GetTilingKey() const
{
    uint64_t selfTilingKy_ = BASE_KEY;
    selfTilingKy_ += commonParams.dtypeKey % NUM_TEN;
    OP_LOGD(context_->GetNodeName(), "tilingKy:            %lu", selfTilingKy_);
    return selfTilingKy_;
}

ge::graphStatus LayerNormV4CommonTiling::DoOpTiling()
{
    int64_t selfBlockDim_;
    float coefficient_;
    int32_t tileLength_;
    uint32_t tileCount_;
    uint32_t blockLength_;
    uint32_t maxtileLength_;

    coefficient_ = static_cast<float>(1.0) / static_cast<float>(commonParams.rowSize);
    
    if (commonParams.colSize <= commonParams.coreNum) {
        selfBlockDim_ = commonParams.colSize;
    } else {
        selfBlockDim_ = commonParams.coreNum;
    }

    blockLength_ = commonParams.rowSize;
    uint64_t usedUbSize = commonParams.ubSizePlatForm - OTHERS_UB_SIZE;
    maxtileLength_ = usedUbSize / SIZE_OF_FLOAT;
    if (maxtileLength_ / NUM_TWO > blockLength_) {
        tileLength_ = blockLength_;
    } else {
        tileLength_ = maxtileLength_ / NUM_TWO;
    }
    tileCount_ = (blockLength_ + tileLength_ - 1) / tileLength_;
    uint32_t tileLengthFp16Aligned = (tileLength_ + BLOCK_SIZE_OF_FLOAT16 - 1) / 
                                         BLOCK_SIZE_OF_FLOAT16 * BLOCK_SIZE_OF_FLOAT16;
    uint32_t maxtileCount = maxtileLength_ - tileLengthFp16Aligned;
    if (maxtileCount < tileCount_) {
        OP_LOGE(context_->GetNodeName(), "The size of normalized shape is too big.");
        return ge::GRAPH_FAILED;
    }
    td_.set_numBlocks(selfBlockDim_);
    td_.set_colSize(commonParams.rowSize);
    td_.set_rowSize(commonParams.colSize);
    td_.set_eps(commonParams.eps);
    td_.set_space(commonParams.ubSizePlatForm);
    td_.set_coefficient(coefficient_);
    td_.set_nullptrGamma(commonParams.gammaNullPtr);   // 
    td_.set_nullptrBeta(commonParams.betaNullPtr);   // 
    TilingDataPrint();

    return ge::GRAPH_SUCCESS;
}

void LayerNormV4CommonTiling::TilingDataPrint()
{
    OP_LOGD(context_->GetNodeName(), "numBlocks:             %ld", td_.get_numBlocks());
    OP_LOGD(context_->GetNodeName(), "colSize:              %u", td_.get_colSize());
    OP_LOGD(context_->GetNodeName(), "rowSize:              %u", td_.get_rowSize());
    OP_LOGD(context_->GetNodeName(), "eps:                  %f", td_.get_eps());
    OP_LOGD(context_->GetNodeName(), "space:          %f", td_.get_space());
}

ge::graphStatus LayerNormV4CommonTiling::PostTiling()
{
    context_->SetBlockDim(td_.get_numBlocks());
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("LayerNormV4", LayerNormV4CommonTiling, 2000);
} // namespace optiling
