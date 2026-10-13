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
 * \file fused_floyd_attention_tiling.cpp
 * \brief
 */

#include "fused_floyd_attention_tiling.h"
#include <queue>
#include <cmath>
#include <cfloat>
#include "log/log.h"
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"
#include "fused_floyd_attention_tiling_common.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {
namespace FFA {
static uint32_t Ceil(uint32_t num1, uint32_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

class FusedFloydAttentionEmptyInputTiling {
public:
    FusedFloydAttentionTilingData tilingData;

    void FusedFloydAttentionSetEmptyInputTilingData(gert::TilingContext *context,
                                                    FusedFloydAttentionTilingData &faTilingData);
    void GetTilingKeyAttentionScore4EmptyInput(uint32_t &tilingKey, const gert::TilingContext *context);
};

void FusedFloydAttentionEmptyInputTiling::GetTilingKeyAttentionScore4EmptyInput(uint32_t &tilingKey,
                                                                                const gert::TilingContext *context)
{
    OP_CHECK_IF(context->GetInputDesc(KEY1_INPUT_INDEX) == nullptr,
        OP_LOGE(context, "GetTilingKeyAttentionScore4EmptyInput occurs nullptr!"),
        return);
    auto kernelType = context->GetInputDesc(KEY1_INPUT_INDEX)->GetDataType();
    if (kernelType == ge::DT_FLOAT16) {
        tilingKey = TILING_KEY_FP16;
    } else if (kernelType == ge::DT_FLOAT) {
        tilingKey = TILING_KEY_FP32;
    } else {
        tilingKey = TILING_KEY_BF16;
    }
}

void FusedFloydAttentionEmptyInputTiling::FusedFloydAttentionSetEmptyInputTilingData(
    gert::TilingContext *context, FusedFloydAttentionTilingData &faTilingData)
{
    OP_CHECK_IF(context->GetRawTilingData() == nullptr,
        OP_LOGE(context, "FusedFloydAttentionSetEmptyInputTilingData occurs nullptr!"),
        return);
    faTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(faTilingData.GetDataSize());
}

static ge::graphStatus CheckParams(gert::TilingContext *context)
{
    if (context->GetInputShape(QUERY_INPUT_INDEX) != nullptr && context->GetInputShape(KEY1_INPUT_INDEX) != nullptr &&
        context->GetInputShape(VALUE1_INPUT_INDEX) != nullptr && context->GetInputShape(KEY2_INPUT_INDEX) != nullptr &&
        context->GetInputShape(VALUE2_INPUT_INDEX) && context->GetAttrs() != nullptr ) {
        if (CheckBaseInput(context) == ge::GRAPH_FAILED) {
            OP_LOGW(context, "fail to get shape or attr from context");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::SUCCESS;
}

static bool IsEmptyInput(gert::TilingContext *context)
{
    auto attenOutShape = context->GetOutputShape(ATTENTIONOUT_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attenOutShape);

    auto queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    auto key1Shape = context->GetInputShape(KEY1_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, key1Shape);

    auto softmaxSumShape = context->GetOutputShape(SOFTMAXSUM_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxSumShape);

    int64_t attentionOutShapeSize = attenOutShape->GetStorageShape().GetShapeSize();
    int64_t queryShapeSize = queryShape->GetStorageShape().GetShapeSize();
    int64_t key1ShapeSize = key1Shape->GetStorageShape().GetShapeSize();
    int64_t softmaxSumShapeSize = softmaxSumShape->GetStorageShape().GetShapeSize();
    if ((queryShapeSize == 0 || key1ShapeSize == 0) && (attentionOutShapeSize != 0 || softmaxSumShapeSize != 0)) {
        /* 以 MIN_COPY_UINT_SIZE 为 32Byte说明, blocks为数据的块数, blocks与coreNum存在三种关系:
          (1) blocks % coreNum == 0
             主核数量为coreNum,主核处理块数为blocks / coreNum, 最后一个核处理非32Byte对齐的数据, 尾核数量为0
          (2) blocks % coreNum != 0
              (2.1) blocks < coreNum
                    主核数量为blocks,主核处理块数为1, 最后一个核处理非32Byte对齐的数据, 尾核数量为0
              (2.2) blocks > coreNum
                    主核数量为blocks % coreNum, 尾核数量为coreNum - (blocks % coreNum), 尾核处理块数为blocks / coreNum
                    主核处理块数为blocks / coreNum + 1,最后一个尾核处理非对齐场景
        (2.2)情况如下:
        |-------------主核块-----------------|------------尾核块-----------|非对齐块|
        |                                   |                             |       |
        |                                   |                             |       |
        |--------n*(blocks/coreNum+1)-------|-----m*(blocks/coreNum)------|<32Byte|
        */
        auto kernelType = context->GetInputDesc(KEY1_INPUT_INDEX)->GetDataType();
        FusedFloydAttentionEmptyInputTiling emptyInputTiling;
        auto compileInfoPtr = reinterpret_cast<const FlashAttentionScoreGradCompileInfo *>(context->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context, "compileInfoPtr is null"),
                   return false);
        uint32_t coreNum = compileInfoPtr->aivNum;
        OP_CHECK_IF((coreNum <= 0),
                   OP_LOGE(context, "platform info is invalid, coreNum=%u.", coreNum), return false);
        OP_CHECK_IF((kernelType != ge::DT_FLOAT16 && kernelType != ge::DT_FLOAT && kernelType != ge::DT_BF16),
                   OP_LOGE(context, "kernelType is invalid, kernelType is %d", kernelType),
                   return false);
        uint32_t attentionOutFormerNum;          // attentionOut的主核
        uint32_t attentionOutTailNum;            // attentionOut的尾核
        uint32_t softmaxMaxFormerNum;            // softmaxMax 和 softmaxSum的主核
        uint32_t softmaxMaxTailNum;              // softmaxMax 和 softmaxSum的尾核
        uint64_t attentionOutSingleCoreDataSize; // attentionOut的每个主核处理的数据个数
        uint64_t attentionOutTailCoreDataSize;   // attentionOut的每个尾核处理的数据个数
        uint64_t softmaxMaxSingleCoreDataSize;
        uint64_t softmaxMaxTailCoreDataSize;
        uint64_t attentionOutLastCoreDataSize = 0; // 最后一个核应该处理的数据量
        uint64_t attentionOutLastCoreIndex = 0;    // 最后一个核起始地址
        uint32_t tilingKey = 0;
        uint64_t attentionOutBlockSize = 0;
        uint64_t softmaxSumBlockSize = 0;

        // 计算 MIN_COPY_UINT_SIZE 块数
        attentionOutBlockSize = Ceil(attentionOutShapeSize * ge::GetSizeByDataType(kernelType), MIN_COPY_UINT_SIZE);
        // softmaxSum 和 softmaxMax 输出为 fp32
        softmaxSumBlockSize = Ceil(softmaxSumShapeSize * ge::GetSizeByDataType(ge::DT_FLOAT), MIN_COPY_UINT_SIZE);

        if (attentionOutShapeSize != 0) {
            if (attentionOutBlockSize % coreNum == 0) {
                attentionOutTailCoreDataSize = 0;
                attentionOutFormerNum = coreNum;
                attentionOutTailNum = 0;
                attentionOutSingleCoreDataSize =
                    attentionOutBlockSize / coreNum * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(kernelType);
                attentionOutLastCoreDataSize =
                    attentionOutSingleCoreDataSize -
                    (attentionOutBlockSize * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(kernelType) -
                     attentionOutShapeSize);
                attentionOutLastCoreIndex = (attentionOutFormerNum - 1) * attentionOutSingleCoreDataSize;
            } else {
                attentionOutTailCoreDataSize =
                    attentionOutBlockSize / coreNum * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(kernelType);
                attentionOutSingleCoreDataSize =
                    attentionOutTailCoreDataSize + MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(kernelType);
                if (attentionOutBlockSize > coreNum) {
                    attentionOutFormerNum = attentionOutBlockSize % coreNum;
                    attentionOutTailNum = coreNum - attentionOutFormerNum;
                    attentionOutLastCoreIndex = attentionOutFormerNum * attentionOutSingleCoreDataSize +
                                                (attentionOutTailNum - 1) * attentionOutTailCoreDataSize;
                    attentionOutLastCoreDataSize =
                        attentionOutTailCoreDataSize -
                        (attentionOutSingleCoreDataSize * attentionOutFormerNum +
                         attentionOutTailCoreDataSize * attentionOutTailNum - attentionOutShapeSize);
                } else {
                    attentionOutFormerNum = attentionOutBlockSize;
                    attentionOutTailNum = 0;
                    attentionOutLastCoreIndex = (attentionOutFormerNum - 1) * attentionOutSingleCoreDataSize;
                    attentionOutLastCoreDataSize =
                        attentionOutSingleCoreDataSize -
                        (attentionOutFormerNum * attentionOutSingleCoreDataSize - attentionOutShapeSize);
                }
            }
        } else {
            attentionOutFormerNum = 0;
            attentionOutTailNum = 0;
            attentionOutSingleCoreDataSize = 0;
            attentionOutTailCoreDataSize = 0;
            attentionOutLastCoreDataSize = 0;
            attentionOutLastCoreIndex = 0;
        }

        if (softmaxSumBlockSize % coreNum == 0) {
            softmaxMaxSingleCoreDataSize =
                softmaxSumBlockSize / coreNum * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(ge::DT_FLOAT);
            softmaxMaxTailCoreDataSize = 0;
            softmaxMaxFormerNum = coreNum;
            softmaxMaxTailNum = 0;
        } else {
            if (softmaxSumBlockSize > coreNum) {
                softmaxMaxFormerNum = softmaxSumBlockSize % coreNum;
                softmaxMaxTailNum = coreNum - softmaxMaxFormerNum;
            } else {
                softmaxMaxFormerNum = softmaxSumBlockSize;
                softmaxMaxTailNum = 0;
            }
            softmaxMaxTailCoreDataSize =
                softmaxSumBlockSize / coreNum * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(ge::DT_FLOAT);
            softmaxMaxSingleCoreDataSize =
                softmaxMaxTailCoreDataSize + MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(ge::DT_FLOAT);
        }

        emptyInputTiling.tilingData.emptyInputTilingData.set_coreNum(coreNum);
        emptyInputTiling.tilingData.emptyInputTilingData.set_attentionOutFormerNum(attentionOutFormerNum);
        emptyInputTiling.tilingData.emptyInputTilingData.set_attentionOutTailNum(attentionOutTailNum);
        emptyInputTiling.tilingData.emptyInputTilingData.set_softmaxMaxFormerNum(softmaxMaxFormerNum);
        emptyInputTiling.tilingData.emptyInputTilingData.set_softmaxMaxTailNum(softmaxMaxTailNum);
        emptyInputTiling.tilingData.emptyInputTilingData.set_attentionOutSingleCoreDataSize(
            attentionOutSingleCoreDataSize);
        emptyInputTiling.tilingData.emptyInputTilingData.set_attentionOutTailCoreDataSize(attentionOutTailCoreDataSize);
        emptyInputTiling.tilingData.emptyInputTilingData.set_softmaxMaxSingleCoreDataSize(softmaxMaxSingleCoreDataSize);
        emptyInputTiling.tilingData.emptyInputTilingData.set_softmaxMaxTailCoreDataSize(softmaxMaxTailCoreDataSize);
        emptyInputTiling.tilingData.emptyInputTilingData.set_attentionOutLastCoreDataSize(attentionOutLastCoreDataSize);
        emptyInputTiling.tilingData.emptyInputTilingData.set_attentionOutLastCoreIndex(attentionOutLastCoreIndex);
        emptyInputTiling.FusedFloydAttentionSetEmptyInputTilingData(context, emptyInputTiling.tilingData);
        emptyInputTiling.GetTilingKeyAttentionScore4EmptyInput(tilingKey, context);
        context->SetTilingKey(tilingKey);
        uint32_t aivActualNum =
            std::max((attentionOutFormerNum + attentionOutTailNum), (softmaxMaxFormerNum + softmaxMaxTailNum));
        context->SetBlockDim(optiling::FFA::FloydCalcTschBlockDim(aivActualNum, 0, compileInfoPtr->aivNum));
        size_t *workspaces = context->GetWorkspaceSizes(1);
        // workspace上预留100M
        workspaces[0] = 100 * 1024 * 1024;

        return true;
    }
    return false;
}

ASCENDC_EXTERN_C ge::graphStatus TilingFusedFloydAttention(gert::TilingContext *context)
{
    if (CheckParams(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (IsEmptyInput(context)) {
        return ge::GRAPH_SUCCESS;
    } else {
        auto resultCode = TilingRegistry::GetInstance().DoTilingImpl(context);
        return resultCode;
    }
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForFusedFloydAttention(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OP_LOGE(context, "platformInfoPtr is null"),
        return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<FlashAttentionScoreGradCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
        OP_LOGE(context, "compileInfoPtr is null"),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP(FusedFloydAttention)
    .Tiling(TilingFusedFloydAttention)
    .TilingParse<FlashAttentionScoreGradCompileInfo>(TilingPrepareForFusedFloydAttention);  // 向框架注册入口函数

} // namespace FFA
} // namespace optiling
