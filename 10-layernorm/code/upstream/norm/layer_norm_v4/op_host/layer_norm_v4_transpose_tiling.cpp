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
 * \file layer_norm_v4_transpose_tiling.cpp
 * \brief
 */

#include "layer_norm_v4_tiling.h"

namespace optiling {
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t FLOAT_SIZE = 4;
constexpr uint64_t UB_SIZE_RESERVED = 1024;
constexpr uint64_t B32_BLOCK_ALIGN_NUM = 8;
constexpr uint64_t B16_BLOCK_ALIGN_NUM = 16;
constexpr uint64_t TRANSPOSE_C0_SIZE = 16;
constexpr uint64_t TRANSPOSE_ROW_LIMIT = 64;
constexpr uint64_t TWO = 2;
constexpr uint64_t FOUR_BUF_NODE = 4;
constexpr uint32_t TWO_POWER_ONE = 2;
constexpr uint32_t TWO_POWER_TWO = 4;
constexpr uint32_t TWO_POWER_THREE = 8;
constexpr uint32_t TWO_POWER_FOUR = 16;

uint64_t LayerNormV4TransposeTiling::CalcBorrowFactor(uint64_t oriFactor)
{
    return (oriFactor + TRANSPOSE_C0_SIZE - 1) / TRANSPOSE_C0_SIZE;
}

uint32_t LayerNormV4TransposeTiling::FindDichotomizeAddDiffSize()
{
    // 找到row与小于row的最近二次幂的差值 eg：rowSize = 15，结果为15 - 8 = 7
    if ((commonParams.rowSize & (commonParams.rowSize - 1)) != 0) {
        uint32_t temp = commonParams.rowSize - 1;
        temp |= temp >> 1;
        temp |= temp >> TWO_POWER_ONE;
        temp |= temp >> TWO_POWER_TWO;
        temp |= temp >> TWO_POWER_THREE;
        temp |= temp >> TWO_POWER_FOUR;
        return (commonParams.rowSize - ((temp + 1) / TWO));
    } else {
        return 0;
    }
}

bool LayerNormV4TransposeTiling::IsCapable()
{
    if (commonParams.isRegBase) {
        return false;
    }
    if ((commonParams.rowSize > TRANSPOSE_ROW_LIMIT) || (commonParams.rowSize == commonParams.rowAlign)) {
        OP_LOGI(
            context_->GetNodeName(),
            "LayerNormV4Transpose Template only support rowSize <= 64 and not align 16, rowSize: %u",
            static_cast<uint32_t>(commonParams.rowSize));
        return false;
    }
    if (LayerNormV4TransposeTiling::GetTilingKey() == 0) {
        OP_LOGI(
            context_->GetNodeName(), "LayerNormV4Transpose Template Unsupported dtype, tensorDtype: %d, paramDtype: %d",
            commonParams.tensorDtype, commonParams.paramDtype);
        return false;
    }
    return true;
}

uint64_t LayerNormV4TransposeTiling::GetTilingKey() const
{
    uint64_t tilingKey = 0;
    if (commonParams.tensorDtype == ge::DT_FLOAT16 && commonParams.paramDtype == ge::DT_FLOAT16) {
        tilingKey = static_cast<uint64_t>(LayerNormV4TilingKey::LAYER_NORM_TRANSPOSE_FLOAT16_FLOAT16);
    }
    if (commonParams.tensorDtype == ge::DT_FLOAT16 && commonParams.paramDtype == ge::DT_FLOAT) {
        tilingKey = static_cast<uint64_t>(LayerNormV4TilingKey::LAYER_NORM_TRANSPOSE_FLOAT16_FLOAT32);
    }
    if (!commonParams.isAscend310P) {
        if (commonParams.tensorDtype == ge::DT_FLOAT && commonParams.paramDtype == ge::DT_FLOAT) {
            tilingKey = static_cast<uint64_t>(LayerNormV4TilingKey::LAYER_NORM_TRANSPOSE_FLOAT32_FLOAT32);
        }
        if (commonParams.tensorDtype == ge::DT_BF16 && commonParams.paramDtype == ge::DT_FLOAT) {
            tilingKey = static_cast<uint64_t>(LayerNormV4TilingKey::LAYER_NORM_TRANSPOSE_BFLOAT16_FLOAT32);
        }
        if (commonParams.tensorDtype == ge::DT_BF16 && commonParams.paramDtype == ge::DT_BF16) {
            tilingKey = static_cast<uint64_t>(LayerNormV4TilingKey::LAYER_NORM_TRANSPOSE_BFLOAT16_BFLOAT16);
        }
    }
    return tilingKey;
}

void LayerNormV4TransposeTiling::DoBlockTiling(BlockTilingData& blockTilingParams)
{
    bool meanOutMoreThanBLOCK = false;
    bool yOutMoreThanBLOCK = false;
    uint64_t blockAlign = (commonParams.tensorDtype == ge::DT_FLOAT ? B32_BLOCK_ALIGN_NUM : B16_BLOCK_ALIGN_NUM);
    for (int64_t curBlockNum = commonParams.coreNum; curBlockNum > 0; curBlockNum--) {
        blockTilingParams.blockFormer = (commonParams.colSize + curBlockNum - 1) / curBlockNum;
        // 910B直接获取切分，无需后续判断
        if (!commonParams.isAscend310P) {
            break;
        }
        meanOutMoreThanBLOCK = (blockTilingParams.blockFormer >= B32_BLOCK_ALIGN_NUM);
        yOutMoreThanBLOCK = ((blockTilingParams.blockFormer * commonParams.rowSize) >= blockAlign);
        // mean,rstd输出
        if ((commonParams.meanAndRstdNullPtr == 0) && meanOutMoreThanBLOCK && yOutMoreThanBLOCK) {
            break;
        }
        // mean,rstd空输出
        if ((commonParams.meanAndRstdNullPtr == 1) && yOutMoreThanBLOCK) {
            break;
        }
        // 置0，表示未找到合适的切分
        blockTilingParams.blockFormer = 0;
    }
    if (blockTilingParams.blockFormer == 0) {
        blockTilingParams.blockFormer = commonParams.colSize;
        blockTilingParams.blockTail = commonParams.colSize;
    }
    blockTilingParams.numBlocks =
        (commonParams.colSize + blockTilingParams.blockFormer - 1) / blockTilingParams.blockFormer;
    blockTilingParams.blockTail =
        commonParams.colSize - (blockTilingParams.numBlocks - 1) * blockTilingParams.blockFormer;
}

void LayerNormV4TransposeTiling::DoUbTiling(const BlockTilingData& blockTilingParams, UbTilingData& ubTilingParams)
{
    bool meanOutLessThanBLOCK = false;
    bool yOutLessThanBLOCK = false;
    uint64_t blockAlign = (commonParams.tensorDtype == ge::DT_FLOAT ? B32_BLOCK_ALIGN_NUM : B16_BLOCK_ALIGN_NUM);
    uint64_t blockAlignGamma = (commonParams.paramDtype == ge::DT_FLOAT ? B32_BLOCK_ALIGN_NUM : B16_BLOCK_ALIGN_NUM);
    uint64_t gammaBufferSize =
        (commonParams.rowSize + blockAlignGamma - 1) / blockAlignGamma * blockAlignGamma * sizeof(float);
    uint64_t curUbSize = commonParams.ubSizePlatForm - gammaBufferSize * TWO - UB_SIZE_RESERVED;
    // 依次为inputX * 2 + outputY + tmpBuf + reduceBuf, 因为 (16 * bFormer) >= ubFormer,
    uint64_t maxUbFormer = curUbSize / (commonParams.rowSize * FLOAT_SIZE * TWO + commonParams.rowSize * FLOAT_SIZE +
                                        commonParams.rowSize * FLOAT_SIZE + FLOAT_SIZE);

    for (int64_t curUbFormer = std::min(maxUbFormer, blockTilingParams.blockFormer); curUbFormer > 0; curUbFormer--) {
        ubTilingParams.ubFormer = curUbFormer;
        ubTilingParams.bFormer = CalcBorrowFactor(ubTilingParams.ubFormer);
        uint64_t alignBAndRow = (ubTilingParams.bFormer * commonParams.rowSize + TRANSPOSE_C0_SIZE - 1) /
                                TRANSPOSE_C0_SIZE * TRANSPOSE_C0_SIZE;
        uint64_t alignB = (ubTilingParams.bFormer + TRANSPOSE_C0_SIZE - 1) / TRANSPOSE_C0_SIZE * TRANSPOSE_C0_SIZE;
        // 借轴后超出总空间，跳过该切分
        // 依次为reduceBuf + inputX * 4(4表示inputX DB + outputY + tmpBuf) + outputMean * 2
        if (ubTilingParams.bFormer * TRANSPOSE_C0_SIZE * FLOAT_SIZE +
                TRANSPOSE_C0_SIZE * alignBAndRow * FLOAT_SIZE * FOUR_BUF_NODE +
                alignB * TRANSPOSE_C0_SIZE * FLOAT_SIZE * TWO >
            curUbSize) {
            continue;
        }
        ubTilingParams.ubLoopOfFormerBlock =
            (blockTilingParams.blockFormer + ubTilingParams.ubFormer - 1) / ubTilingParams.ubFormer;
        ubTilingParams.ubLoopOfTailBlock =
            (blockTilingParams.blockTail + ubTilingParams.ubFormer - 1) / ubTilingParams.ubFormer;
        ubTilingParams.ubTailOfFormerBlock =
            blockTilingParams.blockFormer - (ubTilingParams.ubLoopOfFormerBlock - 1) * ubTilingParams.ubFormer;
        ubTilingParams.ubTailOfTailBlock =
            blockTilingParams.blockTail - (ubTilingParams.ubLoopOfTailBlock - 1) * ubTilingParams.ubFormer;
        // 910B直接获取切分，无需后续判断
        if (!commonParams.isAscend310P) {
            return;
        }
        // 单核无需判断踩踏情况
        if (blockTilingParams.numBlocks == 1) {
            return;
        }
        meanOutLessThanBLOCK = (ubTilingParams.ubTailOfFormerBlock < B32_BLOCK_ALIGN_NUM);
        yOutLessThanBLOCK = ((ubTilingParams.ubTailOfFormerBlock * commonParams.rowSize) < blockAlign);
        // 310P: 如果mean、rstd输出
        if ((commonParams.meanAndRstdNullPtr == 0) && (meanOutLessThanBLOCK || yOutLessThanBLOCK)) {
            continue;
        }
        // mean,rstd空输出
        if ((commonParams.meanAndRstdNullPtr == 1) && yOutLessThanBLOCK) {
            continue;
        }
        return;
    }
    // 未提前return，表示未找到可用切分
    ubTilingParams.ubFormer = 0;
    return;
}

ge::graphStatus LayerNormV4TransposeTiling::DoOpTiling()
{
    BlockTilingData blockTilingParams;
    DoBlockTiling(blockTilingParams);

    UbTilingData ubTilingParams;
    DoUbTiling(blockTilingParams, ubTilingParams);
    if (ubTilingParams.ubFormer == 0) {
        // ub切分不满足，走单核策略
        blockTilingParams.numBlocks = 1;
        blockTilingParams.blockFormer = commonParams.colSize;
        blockTilingParams.blockTail = commonParams.colSize;
        DoUbTiling(blockTilingParams, ubTilingParams);
    }

    uint32_t dichotomizeAddDiffSize = 0;
    dichotomizeAddDiffSize = FindDichotomizeAddDiffSize();

    // set TilingData
    td_.set_col(commonParams.colSize);
    td_.set_row(commonParams.rowSize);
    td_.set_numBlocks(blockTilingParams.numBlocks);
    td_.set_blockFormer(blockTilingParams.blockFormer);
    td_.set_blockTail(blockTilingParams.blockTail);
    td_.set_ubFormer(ubTilingParams.ubFormer);
    td_.set_ubLoopOfFormerBlock(ubTilingParams.ubLoopOfFormerBlock);
    td_.set_ubLoopOfTailBlock(ubTilingParams.ubLoopOfTailBlock);
    td_.set_ubTailOfFormerBlock(ubTilingParams.ubTailOfFormerBlock);
    td_.set_ubTailOfTailBlock(ubTilingParams.ubTailOfTailBlock);
    td_.set_bFormer(ubTilingParams.bFormer);
    td_.set_dichotomizeAddDiffSize(dichotomizeAddDiffSize);
    td_.set_eps(commonParams.eps);
    td_.set_coefficient(commonParams.coefficient);
    td_.set_nullptrGamma(commonParams.gammaNullPtr);
    td_.set_nullptrBeta(commonParams.betaNullPtr);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV4TransposeTiling::PostTiling()
{
    context_->SetBlockDim(td_.get_numBlocks());
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("LayerNormV4", LayerNormV4TransposeTiling, 100);
} // namespace optiling
