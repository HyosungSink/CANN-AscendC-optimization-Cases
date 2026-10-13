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
 * \file moe_gating_top_k_softmax_tiling_k_full_load.cpp
 * \brief
 */
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "../common/include/tiling_base/tiling_templates_registry.h"
#include "../common/include/tiling_base/tiling_type.h"
#include "tiling/tiling_api.h"
#include "moe_gating_top_k_softmax_tiling_base.h"
#include "log/log.h"
using namespace Ops::Transformer::OpTiling;
using namespace AscendC;
using namespace ge;

namespace optiling {
static const int32_t FP32_SIZE = 4;
static const int64_t ALIGN_NUM = 32;
static const int64_t DOUBLE_BUFFER = 2;
static const int32_t MAX_COL_IN_UB = 98304; // ubSize/minTypeSize

static inline uint32_t CeilDiv(uint32_t value, uint32_t factor)
{
    if (factor == 0U) {
        return value;
    }
    return (value + factor - 1U) / factor;
}

class MoeGatingTopKSoftmaxKFullLoadTiling : public MoeGatingTopKSoftmaxBaseTiling
{
public:
    explicit MoeGatingTopKSoftmaxKFullLoadTiling(gert::TilingContext* context) : MoeGatingTopKSoftmaxBaseTiling(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    uint32_t ubOuter;
    uint32_t ubLoop;
    uint32_t ubFormer;
    uint32_t ubTail;
    uint32_t ubFormerAlign;
    uint32_t ubTailAlign;
    MoeGatingTopKSoftmaxKFullLoadTilingData tilingData;
    bool CalculateParamInUb();
    uint64_t GetTotalTmpSizeInUb(uint32_t kAlign);
};

bool MoeGatingTopKSoftmaxKFullLoadTiling::IsCapable()
{
    return true;
}

uint64_t MoeGatingTopKSoftmaxKFullLoadTiling::GetTotalTmpSizeInUb(uint32_t kAlign)
{
    int32_t dataTypeSize;
    // if dtype is bf16,will cast fp32
    if (dtype == ge::DataType::DT_BF16) {
        dataTypeSize = FP32_SIZE;
    } else {
        dataTypeSize = ge::GetSizeByDataType(dtype);
    }
    uint64_t gatingUbTmpSize = static_cast<uint64_t>(ubFormerAlign) * static_cast<uint64_t>(dataTypeSize);
    uint64_t topkOutUbTmpSize = static_cast<uint64_t>((ubFormerAlign + kAlign)) * static_cast<uint64_t>(dataTypeSize);
    uint64_t topkIndicesOutUbTmpSize =
        static_cast<uint64_t>((ubFormerAlign + kAlign)) * static_cast<uint64_t>(FP32_SIZE);
    uint64_t rowsOutUbTmpSize = 0;
    // if dtype is bf16,will cast fp32
    if (dtype == ge::DataType::DT_BF16) {
        rowsOutUbTmpSize = static_cast<uint64_t>(ubFormerAlign) * static_cast<uint64_t>(FP32_SIZE);
    } else {
        rowsOutUbTmpSize = static_cast<uint64_t>(kAlign) * static_cast<uint64_t>(FP32_SIZE);
    }

    uint64_t finishUbTmpSize = ALIGN_NUM;
    auto shape = ge::Shape({ubFormerAlign});
    uint64_t softmaxUbTmpSize = GetSoftMaxFlashV2MaxTmpSize(shape, dataTypeSize, dataTypeSize, true);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());

    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    (void)GetTopKMaxMinTmpSize(
        ascendcPlatform, kAlign + ubFormerAlign, 1, false, true, TopKMode::TOPK_NORMAL, true, dataTypeSize, maxValue,
        minValue);
    maxValue = maxValue > softmaxUbTmpSize ? maxValue : softmaxUbTmpSize;

    uint64_t additionalUbtmpSize = static_cast<uint64_t>(ALIGN_NUM + ALIGN_NUM + ALIGN_NUM);

    uint64_t totalSize = gatingUbTmpSize + topkOutUbTmpSize + topkIndicesOutUbTmpSize + rowsOutUbTmpSize +
                         finishUbTmpSize + maxValue + additionalUbtmpSize;
    return totalSize;
}

bool MoeGatingTopKSoftmaxKFullLoadTiling::CalculateParamInUb()
{
    ubOuter = 1U;
    if (static_cast<int32_t>(col) > MAX_COL_IN_UB) {
        ubOuter = col / static_cast<uint32_t>(MAX_COL_IN_UB);
    }
    uint32_t kAlign =
        CeilDiv(static_cast<uint32_t>(k), static_cast<uint32_t>(ALIGN_NUM)) * static_cast<uint32_t>(ALIGN_NUM);
    while (true) {
        ubFormer = col / ubOuter;
        ubFormerAlign = CeilDiv(ubFormer, static_cast<uint32_t>(ALIGN_NUM)) * static_cast<uint32_t>(ALIGN_NUM);
        uint64_t totalSize = GetTotalTmpSizeInUb(kAlign);
        if (totalSize <= (ubSize / static_cast<uint64_t>(DOUBLE_BUFFER))) {
            break;
        }
        if (ubFormerAlign < kAlign) {
            ubOuter = 0U;
            break;
        }
        ubOuter++;
    }
    if (ubOuter == 0U) {
        return false;
    }
    auto ubFormerDownAlign = (ubFormer / static_cast<uint32_t>(ALIGN_NUM)) * static_cast<uint32_t>(ALIGN_NUM);
    auto ubOuterDownAlign = (col + ubFormerDownAlign - 1U) / ubFormerDownAlign;
    ubLoop = static_cast<uint32_t>(ubOuterDownAlign);
    ubFormer = static_cast<uint32_t>(ubFormerDownAlign);
    ubFormerAlign = ubFormer;
    ubTail = col - (ubLoop - 1U) * ubFormerAlign;
    ubTailAlign = CeilDiv(ubTail, static_cast<uint32_t>(ALIGN_NUM)) * static_cast<uint32_t>(ALIGN_NUM);
    return true;
}

ge::graphStatus MoeGatingTopKSoftmaxKFullLoadTiling::DoOpTiling()
{
    tilingData.set_row(row);
    tilingData.set_col(col);
    tilingData.set_k(k);
    tilingData.set_kAlign(CeilDiv(k, ALIGN_NUM) * ALIGN_NUM);
    tilingData.set_blockFormer(CeilDiv(row, coreNum));
    tilingData.set_blockNum(CeilDiv(row, tilingData.get_blockFormer()));
    tilingData.set_blockTail(row - (tilingData.get_blockNum() - 1) * tilingData.get_blockFormer());

    if (!CalculateParamInUb()) {
        OP_LOGE("[MoeGatingTopKSoftmaxK]", "AutoTiling failed, the K is too large.");
        return ge::GRAPH_FAILED;
    }
    tilingData.set_ubLoop(ubLoop);
    tilingData.set_ubFormer(ubFormer);
    tilingData.set_ubFormerAlign(ubFormerAlign);
    tilingData.set_ubTail(ubTail);
    tilingData.set_ubTailAlign(ubTailAlign);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxKFullLoadTiling::DoLibApiTiling()
{
    int dataTypeSize;
    if (dtype == ge::DataType::DT_BF16) {
        dataTypeSize = FP32_SIZE;
    } else {
        dataTypeSize = ge::GetSizeByDataType(dtype);
    }
    uint32_t ubFormerAlignLocal = CeilDiv(ubFormer, static_cast<uint32_t>(ALIGN_NUM)) * static_cast<uint32_t>(ALIGN_NUM);
    uint32_t kAlign =
        CeilDiv(static_cast<uint32_t>(k), static_cast<uint32_t>(ALIGN_NUM)) * static_cast<uint32_t>(ALIGN_NUM);
    auto softmaxShape = ge::Shape({tilingData.get_ubFormer()});
    SoftMaxFlashV2TilingFunc(
        softmaxShape, dataTypeSize, dataTypeSize,
        GetSoftMaxFlashV2MaxTmpSize(softmaxShape, dataTypeSize, dataTypeSize, true),
        tilingData.ubFormerSoftmaxTilingData, true);

    softmaxShape = ge::Shape({tilingData.get_ubTail()});
    SoftMaxFlashV2TilingFunc(
        softmaxShape, dataTypeSize, dataTypeSize,
        GetSoftMaxFlashV2MaxTmpSize(softmaxShape, dataTypeSize, dataTypeSize, true), tilingData.ubTailSoftmaxTilingData,
        true);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());

    TopKTilingFunc(
        ascendcPlatform, kAlign + ubFormerAlignLocal, 1, kAlign, dataTypeSize, true, TopKMode::TOPK_NORMAL, true,
        tilingData.topkFormerTilingData);

    TopKTilingFunc(
        ascendcPlatform, kAlign + ubTailAlign, 1, kAlign, dataTypeSize, true, TopKMode::TOPK_NORMAL, true,
        tilingData.topkTailTilingData);
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeGatingTopKSoftmaxKFullLoadTiling::GetTilingKey() const
{
    switch (dtype) {
        case ge::DataType::DT_FLOAT16:
            return MOE_GATING_SOFTMAX_K_FULL_LOAD_FLOAT16;
        case ge::DataType::DT_FLOAT:
            return MOE_GATING_SOFTMAX_K_FULL_LOAD_FLOAT;
        case ge::DataType::DT_BF16:
            return MOE_GATING_SOFTMAX_K_FULL_LOAD_BF16;
        default:
            break;
    }
    return tilingKey_;
}

ge::graphStatus MoeGatingTopKSoftmaxKFullLoadTiling::GetWorkspaceSize()
{
    workspaceSize_ = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxKFullLoadTiling::PostTiling()
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

REGISTER_OPS_TILING_TEMPLATE(MoeGatingTopKSoftmax, MoeGatingTopKSoftmaxKFullLoadTiling, 20000);
} // namespace optiling