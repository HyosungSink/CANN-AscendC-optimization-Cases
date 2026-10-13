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
 * \file flash_attention_score_tiling.cpp
 * \brief
 */

#include <queue>
#include <cmath>
#include <cfloat>
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "../common/include/tiling_base/data_copy_transpose_tiling.h"
#include "../common/include/tiling_base/tiling_templates_registry.h"
#include "flash_attention_score_tiling_common.h"
#include "../op_kernel/arch32/flash_attention_score_tiling.h"


using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {

constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr size_t KEY_INPUT_INDEX = 1;
constexpr size_t VALUE_INPUT_INDEX = 2;
constexpr size_t SOFTMAXSUM_OUTPUT_INDEX = 1;
constexpr size_t ATTENTIONOUT_OUTPUT_INDEX = 3;
constexpr size_t INPUTLAYOUT_ATTRS_INDEX = 5;
constexpr size_t MIN_COPY_UINT_SIZE = 32;
constexpr size_t VALUE_100 = 100;
constexpr size_t VALUE_1024 = 1024;
constexpr uint32_t TILING_KEY_1 = 1U;
constexpr uint32_t FA_EMPTY_TILING_KEY = 1;

struct EmptyArgs
{
    uint32_t coreNum;
    uint32_t attentionOutFormerNum;          // attentionOut的主核
    uint32_t attentionOutTailNum;            // attentionOut的尾核
    uint32_t softmaxMaxFormerNum;            // softmaxMax 和 softmaxSum的主核
    uint32_t softmaxMaxTailNum;              // softmaxMax 和 softmaxSum的尾核
    uint64_t attentionOutSingleCoreDataSize; // attentionOut的每个主核处理的数据个数
    uint64_t attentionOutTailCoreDataSize;   // attentionOut的每个尾核处理的数据个数
    uint64_t softmaxMaxSingleCoreDataSize;
    uint64_t softmaxMaxTailCoreDataSize;
    uint64_t attentionOutLastCoreDataSize = 0ULL; // 最后一个核应该处理的数据量
    uint64_t attentionOutLastCoreIndex = 0ULL;    // 最后一个核起始地址
    uint64_t attentionOutBlockSize = 0ULL;
    uint64_t softmaxSumBlockSize = 0ULL;
    uint32_t aivActualNum = 0U;
    uint64_t tilingKey = 0ULL;
};

static uint32_t Ceil(uint32_t num1, uint32_t num2)
{
    if (num2 == 0U) {
        return 0;
    }
    return (num1 + num2 - 1U) / num2;
}

static ge::graphStatus CheckParams(const gert::TilingContext *context)
{
    if (context->GetAttrs() != nullptr && context->GetInputShape(QUERY_INPUT_INDEX) != nullptr &&
        context->GetInputShape(KEY_INPUT_INDEX) != nullptr && context->GetInputShape(VALUE_INPUT_INDEX) != nullptr) {
        auto &valueShape = context->GetInputShape(VALUE_INPUT_INDEX)->GetStorageShape();
        auto &keyShape = context->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
        auto &queryShape = context->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
        const char *inputLayout = context->GetAttrs()->GetAttrPointer<char>(INPUTLAYOUT_ATTRS_INDEX);
        if (strlen(inputLayout) == 3) { // 3: BSH or SBH
            if (inputLayout[0] == 'B') {
                // layout is BSH
                OP_CHECK_IF((queryShape.GetDim(0) != keyShape.GetDim(0)),
                           OP_LOGE(context, "query or key shape is invalid"),
                           return ge::GRAPH_FAILED);
            } else {
                if (inputLayout[0] == 'T') { // TND  N1 != N2
                    // q_D != k_D
                    OP_CHECK_IF((queryShape.GetDim(2) != keyShape.GetDim(2)),
                               OP_LOGE(context, "query or key shape is invalid"),
                               return ge::GRAPH_FAILED);
                    return ge::SUCCESS;
                }
                // layout is SBH
                OP_CHECK_IF((queryShape.GetDim(1) != keyShape.GetDim(1)),
                           OP_LOGE(context, "query or key shape is invalid"),
                           return ge::GRAPH_FAILED);
            }
            // kD < vD
            OP_CHECK_IF((keyShape.GetDim(2) < valueShape.GetDim(2)),
                OP_LOGE(context, "key or value shape is invalid"), return ge::GRAPH_FAILED);
        } else if (strlen(inputLayout) == 4) { // 4: layout is BNSD or BSND
            OP_CHECK_IF((queryShape.GetDim(0) != keyShape.GetDim(0)),
                       OP_LOGE(context, "query or key shape is invalid"), return ge::GRAPH_FAILED);
            OP_CHECK_IF((queryShape.GetDim(3) != keyShape.GetDim(3)),
                       OP_LOGE(context, "query or key shape is invalid"), return ge::GRAPH_FAILED);
            OP_CHECK_IF((keyShape.GetDim(3) < valueShape.GetDim(3)),
                       OP_LOGE(context, "key or value shape is invalid"), return ge::GRAPH_FAILED);
        } else {
            OP_LOGW(context, "invalid input_layout[%s].", inputLayout);
            return ge::GRAPH_FAILED;
        }
        return ge::SUCCESS;
    }
    OP_LOGW(context, "fail to get shape or attr from context");
    return ge::GRAPH_FAILED;
}

static bool GetEmptyArgs(EmptyArgs &emptyArgs, gert::TilingContext *context, const uint32_t &coreNum,
                  const uint64_t &attentionOutShapeSize, const int64_t &softmaxSumShapeSize)
{
    emptyArgs.coreNum = coreNum;
    OP_CHECK_IF((coreNum <= 0),
                OP_LOGE(context, "platform info is invalid, coreNum=%u.", coreNum), return false);
    auto kernelType = context->GetInputDesc(KEY_INPUT_INDEX)->GetDataType();
    OP_CHECK_IF((kernelType != ge::DT_FLOAT16 && kernelType != ge::DT_FLOAT && kernelType != ge::DT_BF16),
               OP_LOGE(context, "kernelType is invalid, kernelType is %d.", kernelType),
               return false);
    uint32_t kernelTypeSize = ge::GetSizeByDataType(kernelType);
    OP_CHECK_IF((kernelTypeSize <= 0),
               OP_LOGE(context, "kernelType size is invalid, kernelType size is %u.",
               kernelTypeSize), return false);
    // 计算 MIN_COPY_UINT_SIZE 块数
    emptyArgs.attentionOutBlockSize = Ceil(static_cast<uint32_t>(attentionOutShapeSize) * kernelTypeSize,
                                           MIN_COPY_UINT_SIZE);
    if (attentionOutShapeSize != 0ULL) {
        if (static_cast<uint64_t>(emptyArgs.attentionOutBlockSize % coreNum) == 0ULL) {
            emptyArgs.attentionOutTailCoreDataSize = 0ULL;
            emptyArgs.attentionOutFormerNum = coreNum;
            emptyArgs.attentionOutTailNum = 0U;
            emptyArgs.attentionOutSingleCoreDataSize = emptyArgs.attentionOutBlockSize / coreNum * MIN_COPY_UINT_SIZE /
                                                       kernelTypeSize;
            emptyArgs.attentionOutLastCoreDataSize = emptyArgs.attentionOutSingleCoreDataSize -
                (emptyArgs.attentionOutBlockSize * MIN_COPY_UINT_SIZE / kernelTypeSize - attentionOutShapeSize);
            emptyArgs.attentionOutLastCoreIndex = static_cast<uint64_t>(emptyArgs.attentionOutFormerNum - 1U) *
                                                  emptyArgs.attentionOutSingleCoreDataSize;
        } else {
            emptyArgs.attentionOutTailCoreDataSize = emptyArgs.attentionOutBlockSize / coreNum * MIN_COPY_UINT_SIZE /
                                                     kernelTypeSize;
            emptyArgs.attentionOutSingleCoreDataSize = emptyArgs.attentionOutTailCoreDataSize + MIN_COPY_UINT_SIZE /
                                                       kernelTypeSize;
            if (emptyArgs.attentionOutBlockSize > coreNum) {
                emptyArgs.attentionOutFormerNum = static_cast<uint32_t>(emptyArgs.attentionOutBlockSize % coreNum);
                emptyArgs.attentionOutTailNum = coreNum - emptyArgs.attentionOutFormerNum;
                emptyArgs.attentionOutLastCoreIndex = 
                    static_cast<uint64_t>(emptyArgs.attentionOutFormerNum) * emptyArgs.attentionOutSingleCoreDataSize +
                    static_cast<uint64_t>(emptyArgs.attentionOutTailNum - 1U) * emptyArgs.attentionOutTailCoreDataSize;
                emptyArgs.attentionOutLastCoreDataSize =
                    emptyArgs.attentionOutTailCoreDataSize - (emptyArgs.attentionOutSingleCoreDataSize *
                    emptyArgs.attentionOutFormerNum + emptyArgs.attentionOutTailCoreDataSize *
                    emptyArgs.attentionOutTailNum - attentionOutShapeSize);
            } else {
                emptyArgs.attentionOutFormerNum = emptyArgs.attentionOutBlockSize;
                emptyArgs.attentionOutTailNum = 0U;
                emptyArgs.attentionOutLastCoreIndex = static_cast<uint64_t>(emptyArgs.attentionOutFormerNum - 1U) *
                    emptyArgs.attentionOutSingleCoreDataSize;
                emptyArgs.attentionOutLastCoreDataSize = emptyArgs.attentionOutSingleCoreDataSize -
                    (emptyArgs.attentionOutFormerNum * emptyArgs.attentionOutSingleCoreDataSize - attentionOutShapeSize);
            }
        }
    } else {
        emptyArgs.attentionOutFormerNum = 0U;
        emptyArgs.attentionOutTailNum = 0U;
        emptyArgs.attentionOutSingleCoreDataSize = 0ULL;
        emptyArgs.attentionOutTailCoreDataSize = 0ULL;
        emptyArgs.attentionOutLastCoreDataSize = 0ULL;
        emptyArgs.attentionOutLastCoreIndex = 0ULL;
    }

    uint32_t floatDataSize =  ge::GetSizeByDataType(ge::DT_FLOAT);
    OP_CHECK_IF((floatDataSize <= 0),
               OP_LOGE(context, "float data size is invalid, size is %u.", floatDataSize),
               return false);
    // softmaxSum 和 softmaxMax 输出为 fp32
    emptyArgs.softmaxSumBlockSize = Ceil(static_cast<uint32_t>(softmaxSumShapeSize) * floatDataSize, MIN_COPY_UINT_SIZE);
    if (static_cast<uint64_t>(emptyArgs.softmaxSumBlockSize % coreNum) == 0ULL) {
        emptyArgs.softmaxMaxSingleCoreDataSize = emptyArgs.softmaxSumBlockSize / coreNum * MIN_COPY_UINT_SIZE /
                                                 floatDataSize;
        emptyArgs.softmaxMaxTailCoreDataSize = 0U;
        emptyArgs.softmaxMaxFormerNum = coreNum;
        emptyArgs.softmaxMaxTailNum = 0U;
    } else {
        if (emptyArgs.softmaxSumBlockSize > coreNum) {
            emptyArgs.softmaxMaxFormerNum = static_cast<uint32_t>(emptyArgs.softmaxSumBlockSize % coreNum);
            emptyArgs.softmaxMaxTailNum = coreNum - emptyArgs.softmaxMaxFormerNum;
        } else {
            emptyArgs.softmaxMaxFormerNum = emptyArgs.softmaxSumBlockSize;
            emptyArgs.softmaxMaxTailNum = 0U;
        }
        emptyArgs.softmaxMaxTailCoreDataSize =
            emptyArgs.softmaxSumBlockSize / coreNum * MIN_COPY_UINT_SIZE / floatDataSize;
        emptyArgs.softmaxMaxSingleCoreDataSize =
            emptyArgs.softmaxMaxTailCoreDataSize + MIN_COPY_UINT_SIZE / floatDataSize;
    }
    emptyArgs.aivActualNum = std::max((emptyArgs.attentionOutFormerNum + emptyArgs.attentionOutTailNum),
                                      (emptyArgs.softmaxMaxFormerNum + emptyArgs.softmaxMaxTailNum));
    return true;
}

static bool IsEmptyInput(gert::TilingContext *context)
{
    auto attentionOutShape = context->GetOutputShape(ATTENTIONOUT_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);

    auto queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    auto keyShape = context->GetInputShape(KEY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);

    auto softmaxSumShape = context->GetOutputShape(SOFTMAXSUM_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxSumShape);

    int64_t attentionOutShapeSize = attentionOutShape->GetStorageShape().GetShapeSize();
    int64_t queryShapeSize = queryShape->GetStorageShape().GetShapeSize();
    int64_t keyShapeSize = keyShape->GetStorageShape().GetShapeSize();
    int64_t softmaxSumShapeSize = softmaxSumShape->GetStorageShape().GetShapeSize();
    if ((queryShapeSize == 0 || keyShapeSize == 0) && (attentionOutShapeSize != 0 || softmaxSumShapeSize != 0)) {
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
        FlashAttentionScoreTilingData* emptyInputTiling = context->GetTilingData<FlashAttentionScoreTilingData>();
        emptyInputTiling->reset();
        auto compileInfoPtr = reinterpret_cast<const FlashAttentionScoreCompileInfo *>(context->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context, "compileInfoPtr is null"),
                   return false);
        EmptyArgs emptyArgs;
        if (!GetEmptyArgs(emptyArgs, context, compileInfoPtr->aivNum, attentionOutShapeSize, softmaxSumShapeSize)){
            return false;
        }
        emptyInputTiling->emptyInputTilingData.set_coreNum(emptyArgs.coreNum);
        emptyInputTiling->emptyInputTilingData.set_attentionOutFormerNum(emptyArgs.attentionOutFormerNum);
        emptyInputTiling->emptyInputTilingData.set_attentionOutTailNum(emptyArgs.attentionOutTailNum);
        emptyInputTiling->emptyInputTilingData.set_softmaxMaxFormerNum(emptyArgs.softmaxMaxFormerNum);
        emptyInputTiling->emptyInputTilingData.set_softmaxMaxTailNum(emptyArgs.softmaxMaxTailNum);
        emptyInputTiling->emptyInputTilingData.set_attentionOutSingleCoreDataSize(emptyArgs.attentionOutSingleCoreDataSize);
        emptyInputTiling->emptyInputTilingData.set_attentionOutTailCoreDataSize(emptyArgs.attentionOutTailCoreDataSize);
        emptyInputTiling->emptyInputTilingData.set_softmaxMaxSingleCoreDataSize(emptyArgs.softmaxMaxSingleCoreDataSize);
        emptyInputTiling->emptyInputTilingData.set_softmaxMaxTailCoreDataSize(emptyArgs.softmaxMaxTailCoreDataSize);
        emptyInputTiling->emptyInputTilingData.set_attentionOutLastCoreDataSize(emptyArgs.attentionOutLastCoreDataSize);
        emptyInputTiling->emptyInputTilingData.set_attentionOutLastCoreIndex(emptyArgs.attentionOutLastCoreIndex);
        context->SetTilingKey(FA_EMPTY_TILING_KEY);
        auto platformInfoPtr = context->GetPlatformInfo();
        OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "platformInfoPtr is null"), return false);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(emptyArgs.aivActualNum, 0, compileInfoPtr->aivNum));
        size_t *workspaces = context->GetWorkspaceSizes(1);
        // workspace上预留100M
        workspaces[0] = VALUE_100 * VALUE_1024 * VALUE_1024;
        return true;
    }
    return false;
}

ASCENDC_EXTERN_C ge::graphStatus TilingFlashAttentionScore(gert::TilingContext *context)
{
    OP_LOGW(context, "Start registering tiling.");
    if (CheckParams(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
 
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OP_LOGE(context, "platformInfoPtr is null"),
        return ge::GRAPH_FAILED);
 
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    {
        OP_LOGW(context, "Current npu arch is not dav-3510.");
        if (IsEmptyInput(context)) {
            return ge::GRAPH_SUCCESS;
        }
    }

    auto resultCode = TilingRegistryArch::GetInstance().DoTilingImpl(context);
    return resultCode;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForFlashAttentionScore(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OP_LOGE(context, "platformInfoPtr is null"),
        return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<FlashAttentionScoreCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
        OP_LOGE(context, "compileInfoPtr is null"),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2CacheSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FlashAttentionScore)
    .Tiling(TilingFlashAttentionScore)
    .TilingInputsDataDependency({7, 8, 9, 10, 11})
    .TilingParse<FlashAttentionScoreCompileInfo>(TilingPrepareForFlashAttentionScore);  // 向框架注册入口函数

} // namespace optiling
