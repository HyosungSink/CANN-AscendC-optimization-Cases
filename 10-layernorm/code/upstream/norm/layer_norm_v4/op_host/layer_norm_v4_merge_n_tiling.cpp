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
 * \file layer_norm_v4_merge_n_tiling.cpp
 * \brief
 */

#include "layer_norm_v4_tiling.h"

namespace optiling {
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t NEED_ALIGN_BLOCK_SIZE = 24;
constexpr uint64_t BLOCK_NUM = 8;
constexpr uint64_t FLOAT_SIZE = 4;
constexpr uint64_t HALF_SIZE = 2;
constexpr uint64_t NUM_EIGHT = 8;
constexpr uint64_t NUM_TWO = 2;
constexpr uint64_t NUM_TEN = 10;
constexpr uint64_t N_ROW_LIMIT = 32;
constexpr uint64_t N_ROW_ALIGN = 64;
constexpr uint64_t COL_LIMIT = 1280;
constexpr uint64_t UB_BUF = 1024 * 2;
constexpr uint64_t MAX_ROW_LIMIT = 2040;
constexpr uint64_t TILINGKEY_FLOAT16_TO_FLOAT32 = 10;
constexpr uint64_t TILINGKEY_FLOAT16_TO_FLOAT16 = 11;
constexpr uint64_t TILINGKEY_BFLOAT16_TO_FLOAT32 = 20;
constexpr uint64_t TILINGKEY_BFLOAT16_TO_FLOAT16 = 22;
constexpr uint64_t TILINGKEY_ALIGN_LIMIT = 1000;

bool LayerNormV4MergeNTiling::IsCapable()
{
    if (commonParams.isRegBase) {
        return false;
    }
    if (commonParams.isAscend310P) {
        return false;
    }
    if (commonParams.isV3) {
        return false;
    }
    uint32_t nRow = (commonParams.colSize + commonParams.coreNum - 1 ) / commonParams.coreNum;
    uint32_t rowMax = (commonParams.ubSizePlatForm - UB_BUF) / FLOAT_SIZE / NUM_TWO;
    if ((commonParams.rowSize < rowMax && nRow > NUM_TWO) || ((commonParams.rowSize <= COL_LIMIT) && nRow > NUM_TWO)){
        return true;
    }
    return false;
}

uint64_t LayerNormV4MergeNTiling::GetTilingKey() const
{
    uint64_t tilingKy = 0;
    tilingKy = static_cast<uint64_t>(LayerNormV4TilingKey::LAYER_NORM_MERGE_N_FLOAT32_FLOAT32);
    tilingKy += commonParams.dtypeKey % NUM_TEN;

    if (commonParams.rowSize <= COL_LIMIT){
        tilingKy += TILINGKEY_ALIGN_LIMIT;
        if (commonParams.rowSize == 1) {
            tilingKy += TILINGKEY_ALIGN_LIMIT;
        }
    }
    OP_LOGD(context_, "TilingKy For MergeN is %d", tilingKy);
    return tilingKy;
}

void LayerNormV4MergeNTiling::PrepareTiling(uint64_t ubSize, uint32_t& nRow, uint32_t tailNRow, uint32_t tailNum)
{
    uint32_t loopCount = 0;
    uint32_t tailLoop = 0;
    uint32_t alignment = 0;
    uint32_t tailRemainNum = 0;
    uint32_t formerRemainNum = 0;
    if (commonParams.tensorDtype == ge::DT_FLOAT) {
        alignment = (BLOCK_SIZE / FLOAT_SIZE);
    } else {
        alignment = (BLOCK_SIZE / HALF_SIZE);
    }

    // rowsize是一行的大小，需要向上取整，对其ub的block
    uint64_t rowAlign = (commonParams.rowSize + alignment - 1) / alignment * alignment;

    // 一次可以计算行数
    uint32_t ableNRow = ubSize / FLOAT_SIZE / (1 + rowAlign) / NUM_TWO;

    if (rowAlign <= N_ROW_LIMIT) {
        ableNRow = ableNRow / N_ROW_ALIGN * N_ROW_ALIGN; 
    }

    if(nRow > ableNRow){
        loopCount = nRow / ableNRow;
        formerRemainNum = nRow % ableNRow;
        nRow = ableNRow;
    }else {
        loopCount = 1;
    }

    if (tailNum > 0){
        if(tailNRow > ableNRow){
            tailLoop = tailNRow / ableNRow;
            tailRemainNum = tailNRow % ableNRow;
            tailNRow = ableNRow;
        } else {
            tailLoop = 1;
        }
    }
    td_.set_loopCount(loopCount);   // 单核循环次数(可能有问题)
    td_.set_tailLoop(tailLoop);   // 同loopcount 此处小核
    td_.set_tailNRow(tailNRow);   // 小核处理行数
    td_.set_tailRemainNum(tailRemainNum);   // 小核尾行
    td_.set_formerRemainNum(formerRemainNum);   // 大核尾行
    td_.set_eps(commonParams.eps);   // eps
    td_.set_ableNRow(ableNRow);   // 单核一次可处理行数
    td_.set_tileLength(nRow * rowAlign);   // 大核单核总数据量
    td_.set_blockLength(ableNRow * rowAlign);   // 大核单核总大小
    td_.set_rowAlign(rowAlign);   // 列对齐
}

ge::graphStatus LayerNormV4MergeNTiling::DoOpTiling()
{
    OP_LOGD(context_, "Do MergeN OpTiling");
    uint32_t numBlocks = 0;
    uint32_t nRow = 0;
    uint32_t tailNRow = 0;
    uint64_t rowSize = commonParams.rowSize;
    uint64_t colSize = commonParams.colSize;
    float coefficient = static_cast<float>(1.0) / static_cast<float>(rowSize);
    uint32_t formerNum = 0;
    
    uint32_t tailNum = 0;
    uint64_t coreNum = commonParams.coreNum;
    uint64_t ubSize = commonParams.ubSizePlatForm - UB_BUF;
    if (colSize <= coreNum) {
        nRow = 1;
        numBlocks = static_cast<uint32_t>(colSize);
        formerNum = colSize;
    } else {
        numBlocks = static_cast<uint32_t>(coreNum);
        // 余数为前n个处理多余的行数
        formerNum = colSize % numBlocks;
        if (formerNum == 0){
            formerNum = numBlocks;
            nRow = colSize / numBlocks;
        }else {
            // 总核数减去前面的就是正常函数的个数40 - n
            tailNum = numBlocks - formerNum;
            // 向上取整
            nRow = (colSize + numBlocks - 1 ) / numBlocks;
            tailNRow = colSize / numBlocks;
        }
    }
    PrepareTiling(ubSize, nRow, tailNRow, tailNum);
    td_.set_numBlocks(numBlocks);   // 核数
    td_.set_colSize(colSize);   // 行数
    td_.set_rowSize(rowSize);   // 列数
    td_.set_coefficient(coefficient);   // 系数
    td_.set_nRow(nRow);   // 单核处理行数
    td_.set_formerNum(formerNum);   // 大核数量
    td_.set_tailNum(tailNum);   // 小核数量
    td_.set_nullptrGamma(commonParams.gammaNullPtr);   // 
    td_.set_nullptrBeta(commonParams.betaNullPtr);   // 
    OP_LOGD(context_, "Exit MergeN OpTiling");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4MergeNTiling::PostTiling()
{
    context_->SetBlockDim(td_.get_numBlocks());
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("LayerNormV4", LayerNormV4MergeNTiling, 1220);
} // namespace optiling
