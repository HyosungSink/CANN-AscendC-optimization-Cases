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
 * \file moe_gating_top_k_softmax_tiling_e_k_full_load.cpp
 * \brief
 */

#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "../common/include/tiling_base/tiling_templates_registry.h"
#include "../common/include/tiling_base/tiling_type.h"
#include "moe_gating_top_k_softmax_tiling_base.h"
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;
namespace optiling {
static const int32_t DIM_0 = 0;
static const int32_t DIM_1 = 1;
static const int32_t DIM_2 = 2;
static const int32_t SIZE_2 = 2;
static const int32_t SIZE_3 = 3;
static const int32_t FP32_SIZE = 4;
static const int32_t FP16_SIZE = 2;
static const int32_t BF16_SIZE = 2;
static const int32_t INT32_SIZE = 4;
static const int32_t BOOL_SIZE = 1;
static const int64_t BLOCK_SIZE = 32;
static const int64_t ALIGN_NUM = 32;
static const bool IS_SOFTMAX_REUSE_SOURCE = true;
static const bool IS_TOP_K_REUSE_SOURCE = true;
static const int32_t MAX_COL_IN_UB = 98304; // ubSize/minTypeSize

static inline uint32_t CeilDiv(uint32_t value, uint32_t factor)
{
    if (factor == 0U) {
        return value;
    }
    return (value + factor - 1U) / factor;
}

static inline int64_t calcUbAlignBufferSize(const uint32_t curRowInUb, const uint32_t col, const int typeSize)
{
    return static_cast<int64_t>(CeilDiv(col * static_cast<uint32_t>(typeSize), static_cast<uint32_t>(BLOCK_SIZE))) *
           BLOCK_SIZE * static_cast<int64_t>(curRowInUb);
}

static inline uint32_t calcGatingAlignCol(const uint32_t col, const ge::DataType dtypeLocal)
{
    (void)dtypeLocal;
    // 对齐成32个数处理
    return CeilDiv(col, static_cast<uint32_t>(ALIGN_NUM)) * static_cast<uint32_t>(ALIGN_NUM);
}

class MoeGatingTopKSoftmaxEKFullLoadTiling : public MoeGatingTopKSoftmaxBaseTiling
{
public:
    explicit MoeGatingTopKSoftmaxEKFullLoadTiling(gert::TilingContext* context)
        : MoeGatingTopKSoftmaxBaseTiling(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    uint32_t maxRow;
    uint32_t gatingAlignCol;
    bool doubleBufferFlag;
    MoeGatingTopKSoftmaxEKFullLoadTilingData tilingData;
    uint32_t calcMaxRowInUb(
        const bool doubleBufferFlag, const uint64_t ubSize, const ge::DataType dtype, const uint32_t k,
        const uint32_t blockRow);
    bool isBufferSizeEnough(
        const uint32_t curRowInUb, const uint32_t gatingAlignCol, const uint64_t tmpUbSize, const ge::DataType dtype,
        const uint32_t k);
    bool getDoubleBufferFlag(
        const uint32_t gatingAlignCol, const uint64_t ubSize, const ge::DataType dtype, const uint32_t k);
};

bool MoeGatingTopKSoftmaxEKFullLoadTiling::IsCapable()
{
    if (col > static_cast<uint32_t>(MAX_COL_IN_UB)) {
        return false;
    }
    gatingAlignCol = calcGatingAlignCol(col, dtype);
    doubleBufferFlag = getDoubleBufferFlag(gatingAlignCol, ubSize, dtype, k);
    maxRow = calcMaxRowInUb(doubleBufferFlag, ubSize, dtype, k, CeilDiv(row, coreNum));
    if (maxRow == 0U) {
        return false;
    }
    return true;
}

ge::graphStatus MoeGatingTopKSoftmaxEKFullLoadTiling::DoOpTiling()
{
    tilingData.set_row(row);
    tilingData.set_col(col);

    tilingData.set_k(k);
    tilingData.set_kAlignB16(CeilDiv(k * FP16_SIZE, BLOCK_SIZE) * BLOCK_SIZE);
    tilingData.set_kAlignB32(CeilDiv(k * FP32_SIZE, BLOCK_SIZE) * BLOCK_SIZE);

    tilingData.set_blockFormer(CeilDiv(row, coreNum));
    tilingData.set_blockNum(CeilDiv(row, tilingData.get_blockFormer()));
    tilingData.set_blockTail(row - (tilingData.get_blockNum() - 1) * tilingData.get_blockFormer());

    tilingData.set_colAlign(gatingAlignCol);
    tilingData.set_ubLoopOfFormerBlock(CeilDiv(tilingData.get_blockFormer(), maxRow));
    tilingData.set_ubFormer(maxRow);
    tilingData.set_ubTailOfFormerBlock(
        tilingData.get_blockFormer() - (tilingData.get_ubLoopOfFormerBlock() - 1) * tilingData.get_ubFormer());
    tilingData.set_ubLoopOfTailBlock(CeilDiv(tilingData.get_blockTail(), maxRow));
    tilingData.set_ubTailOfTailBlock(
        tilingData.get_blockTail() - (tilingData.get_ubLoopOfTailBlock() - 1) * tilingData.get_ubFormer());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxEKFullLoadTiling::DoLibApiTiling()
{
    int dataTypeSize;
    if (dtype == ge::DataType::DT_BF16) {
        dataTypeSize = FP32_SIZE;
    } else {
        dataTypeSize = ge::GetSizeByDataType(dtype);
    }
    auto softmaxShape = ge::Shape({tilingData.get_ubFormer(), gatingAlignCol});
    SoftMaxTilingFunc(
        softmaxShape, dataTypeSize, GetSoftMaxMaxTmpSize(softmaxShape, dataTypeSize, IS_SOFTMAX_REUSE_SOURCE),
        tilingData.formerSoftmaxTilingData);

    softmaxShape = ge::Shape({tilingData.get_ubTailOfFormerBlock(), gatingAlignCol});
    SoftMaxTilingFunc(
        softmaxShape, dataTypeSize, GetSoftMaxMaxTmpSize(softmaxShape, dataTypeSize, IS_SOFTMAX_REUSE_SOURCE),
        tilingData.formerBlockTailSoftmaxTilingData);

    softmaxShape = ge::Shape({tilingData.get_ubTailOfTailBlock(), gatingAlignCol});
    SoftMaxTilingFunc(
        softmaxShape, dataTypeSize, GetSoftMaxMaxTmpSize(softmaxShape, dataTypeSize, IS_SOFTMAX_REUSE_SOURCE),
        tilingData.tailBlockTailSoftmaxTilingData);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    TopKTilingFunc(
        ascendcPlatform, gatingAlignCol, tilingData.get_ubFormer(), k, dataTypeSize, true, TopKMode::TOPK_NORMAL, true,
        tilingData.formerTopkTilingData);

    TopKTilingFunc(
        ascendcPlatform, gatingAlignCol, tilingData.get_ubTailOfFormerBlock(), k, dataTypeSize, true,
        TopKMode::TOPK_NORMAL, true, tilingData.formerBlockTailTopkTilingData);

    TopKTilingFunc(
        ascendcPlatform, gatingAlignCol, tilingData.get_ubTailOfTailBlock(), k, dataTypeSize, true,
        TopKMode::TOPK_NORMAL, true, tilingData.tailBlockTailTopkTilingData);
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeGatingTopKSoftmaxEKFullLoadTiling::GetTilingKey() const
{
    switch (dtype) {
        case ge::DataType::DT_FLOAT16:
            return doubleBufferFlag ? MOE_GATING_SOFTMAX_FLOAT16_DOUBLE_BUFFER : MOE_GATING_SOFTMAX_FLOAT16;
        case ge::DataType::DT_FLOAT:
            return doubleBufferFlag ? MOE_GATING_SOFTMAX_FLOAT_DOUBLE_BUFFER : MOE_GATING_SOFTMAX_FLOAT;
        case ge::DataType::DT_BF16:
            return doubleBufferFlag ? MOE_GATING_SOFTMAX_BF16_DOUBLE_BUFFER : MOE_GATING_SOFTMAX_BF16;
        default:
            break;
    }
    return tilingKey_;
}

ge::graphStatus MoeGatingTopKSoftmaxEKFullLoadTiling::GetWorkspaceSize()
{
    workspaceSize_ = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxEKFullLoadTiling::PostTiling()
{
    tilingData.set_tilingKey(GetTilingKey());
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(tilingData.get_blockNum());
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

bool MoeGatingTopKSoftmaxEKFullLoadTiling::getDoubleBufferFlag(
    const uint32_t gatingAlignColLocal, const uint64_t ubSizeLocal, const ge::DataType dtypeLocal, const uint32_t kLocal)
{
    // 判断一行的数据是否能搬进一半大小的ub空间
    return isBufferSizeEnough(1, gatingAlignColLocal, ubSizeLocal / SIZE_2, dtypeLocal, kLocal);
}

bool MoeGatingTopKSoftmaxEKFullLoadTiling::isBufferSizeEnough(
    const uint32_t curRowInUb, const uint32_t gatingAlignColLocal, const uint64_t tmpUbSize, const ge::DataType dtypeLocal,
    const uint32_t kLocal)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    // bf16类型cast成fp32处理
    if (dtypeLocal == ge::DataType::DT_BF16) {
        // 1.bf16场景gating复用缓存softmax输出结果，按fp32分配内存
        uint64_t gatingBufferSize = static_cast<uint64_t>(curRowInUb * gatingAlignColLocal);
        gatingBufferSize = gatingBufferSize * static_cast<uint64_t>(FP32_SIZE);
        if (gatingBufferSize > tmpUbSize) {
            return false;
        }
        // 2.计算softmax
        auto shape = ge::Shape({curRowInUb, gatingAlignColLocal});
        uint64_t softMaxMaxTmpSize = GetSoftMaxMaxTmpSize(shape, FP32_SIZE, IS_SOFTMAX_REUSE_SOURCE);

        // 3.计算topk
        uint64_t topKUbOutBufferSize = static_cast<uint64_t>(calcUbAlignBufferSize(curRowInUb, kLocal, FP32_SIZE));
        uint64_t topKIndicesBufferSize = static_cast<uint64_t>(calcUbAlignBufferSize(curRowInUb, kLocal, INT32_SIZE));

        uint32_t maxValue;
        uint32_t minValue;
        GetTopKMaxMinTmpSize(
            ascendcPlatform, gatingAlignColLocal, curRowInUb, IS_TOP_K_REUSE_SOURCE, true, TopKMode::TOPK_NORMAL, true,
            FP32_SIZE, maxValue, minValue);

        maxValue = maxValue > softMaxMaxTmpSize ? maxValue : softMaxMaxTmpSize;

        uint64_t finishedUbBufferSize = static_cast<uint64_t>(
            CeilDiv(static_cast<uint32_t>(BOOL_SIZE) * curRowInUb, static_cast<uint32_t>(BOOL_SIZE)) * BLOCK_SIZE);

        // sourceRowOut用来复用缓存softmax输出，按r*E分配大小
        uint64_t sourceRowOutAlignBufferSize =
            static_cast<uint64_t>(curRowInUb * gatingAlignColLocal * static_cast<uint32_t>(INT32_SIZE));

        if (static_cast<uint64_t>(gatingAlignColLocal) * static_cast<uint64_t>(INT32_SIZE) + gatingBufferSize +
                topKUbOutBufferSize + topKIndicesBufferSize + static_cast<uint64_t>(maxValue) + finishedUbBufferSize +
                sourceRowOutAlignBufferSize > tmpUbSize) {
            return false;
        }
        return true;
    }

    // 1.搬入gating
    int typeSize = ge::GetSizeByDataType(dtypeLocal);
    uint64_t gatingAlignBufferSize = static_cast<uint64_t>(curRowInUb * gatingAlignColLocal);
    gatingAlignBufferSize = gatingAlignBufferSize * static_cast<uint64_t>(typeSize);
    if (gatingAlignBufferSize > tmpUbSize) {
        return false;
    }

    // 2.执行softmax
    auto shape = ge::Shape({curRowInUb, gatingAlignColLocal});
    uint64_t softMaxMaxTmpSize = GetSoftMaxMaxTmpSize(shape, typeSize, IS_SOFTMAX_REUSE_SOURCE);

    // 3.计算topk
    uint32_t maxValue;
    uint32_t minValue;
    GetTopKMaxMinTmpSize(
        ascendcPlatform, gatingAlignColLocal, curRowInUb, IS_TOP_K_REUSE_SOURCE, true, TopKMode::TOPK_NORMAL, true, typeSize,
        maxValue, minValue);

    maxValue = maxValue > softMaxMaxTmpSize ? maxValue : softMaxMaxTmpSize;

    uint64_t topKOutAlignBufferSize = static_cast<uint64_t>(calcUbAlignBufferSize(curRowInUb, kLocal, typeSize));
    uint64_t topKIndicesAlignBufferSize = static_cast<uint64_t>(calcUbAlignBufferSize(curRowInUb, kLocal, INT32_SIZE));

    uint64_t finishedUbBufferSize = static_cast<uint64_t>(
        CeilDiv(static_cast<uint32_t>(BOOL_SIZE) * curRowInUb, static_cast<uint32_t>(BOOL_SIZE)) * BLOCK_SIZE);

    // sourceRowOut用来复用缓存softmax输出，按r*E分配大小
    uint64_t sourceRowOutAlignBufferSize =
        static_cast<uint64_t>(curRowInUb * gatingAlignColLocal * static_cast<uint32_t>(INT32_SIZE));

    if (static_cast<uint64_t>(gatingAlignColLocal) * static_cast<uint64_t>(INT32_SIZE) + gatingAlignBufferSize +
            topKOutAlignBufferSize + topKIndicesAlignBufferSize + sourceRowOutAlignBufferSize +
            static_cast<uint64_t>(maxValue) + finishedUbBufferSize > tmpUbSize) {
        return false;
    }
    return true;
}

uint32_t MoeGatingTopKSoftmaxEKFullLoadTiling::calcMaxRowInUb(
    const bool doubleBufferFlagLocal, const uint64_t ubSizeLocal, const ge::DataType dtypeLocal, const uint32_t kLocal,
    const uint32_t blockRow)
{
    uint32_t ubOuter = 1;
    uint64_t tmpUbSize = doubleBufferFlagLocal ? ubSizeLocal / static_cast<uint64_t>(2) : ubSizeLocal;
    uint64_t curRowInUb;
    while (true) {
        curRowInUb = CeilDiv(blockRow, ubOuter);
        if (isBufferSizeEnough(curRowInUb, gatingAlignCol, tmpUbSize, dtypeLocal, kLocal)) {
            break;
        }
        ubOuter++;
        if (ubOuter > blockRow) {
            return 0;
        }
    }
    return curRowInUb;
}
REGISTER_OPS_TILING_TEMPLATE(MoeGatingTopKSoftmax, MoeGatingTopKSoftmaxEKFullLoadTiling, 10000);
} // namespace optiling