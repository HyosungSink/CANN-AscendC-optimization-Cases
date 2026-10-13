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
 * \file nsa_selected_attention_tiling.cpp
 * \brief
 */

#include "nsa_selected_attention_tiling.h"
#include <queue>
#include <cmath>
#include <cfloat>
#include "log/log.h"
#include "../common/include/err/ops_err.h"
#include <register/op_impl_registry.h>
#include "../common/include/tiling_base/data_copy_transpose_tiling.h"
#include "../common/include/tiling_base/tiling_templates_registry.h"
using namespace Ops::Transformer::OpTiling;
using namespace ge;
using namespace AscendC;

namespace optiling {

constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr size_t KEY_INPUT_INDEX = 1;
constexpr size_t VALUE_INPUT_INDEX = 2;
constexpr size_t SOFTMAXSUM_OUPUT_INDEX = 1;
constexpr size_t ATTENTIONOUT_OUPUT_INDEX = 2;
constexpr size_t INPUTLAYOUT_ATTRS_INDEX = 3;
constexpr size_t MIN_COPY_UINT_SIZE = 32;

static uint32_t Ceil(uint32_t num1, uint32_t num2)
{
    if (num2 == 0U) {
        return 0;
    }
    return (num1 + num2 - 1U) / num2;
}

class NsaSelectedAttentionEmptyInputTiling {
public:
    NsaSelectedAttentionTilingData tilingData;

    void NsaSelectedAttentionSetEmptyInputTilingData(gert::TilingContext *context,
                                                    NsaSelectedAttentionTilingData &nsaTilingData);
    void GetTilingKeyAttentionScore4EmptyInput(uint32_t &tilingKey, const gert::TilingContext *context);
};

void NsaSelectedAttentionEmptyInputTiling::GetTilingKeyAttentionScore4EmptyInput(uint32_t &tilingKey,
                                                                                const gert::TilingContext *context)
{
    OP_CHECK_IF(context->GetInputDesc(KEY_INPUT_INDEX) == nullptr,
                OP_LOGE(unlikely(((context) == nullptr) || (context)->GetNodeName() == nullptr) ? "nil" :(context)->GetNodeName(),
                "context->GetInputDesc(KEY_INPUT_INDEX) is nullptr!"),
                return);
    auto kernelType = context->GetInputDesc(KEY_INPUT_INDEX)->GetDataType();
    if (kernelType == ge::DT_FLOAT16) {
        tilingKey = 90U;
    } else if (kernelType == ge::DT_FLOAT) {
        tilingKey = 92U;
    } else {
        tilingKey = 94U;
    }
}

void NsaSelectedAttentionEmptyInputTiling::NsaSelectedAttentionSetEmptyInputTilingData(
    gert::TilingContext *context, NsaSelectedAttentionTilingData &nsaTilingData)
{
    OP_CHECK_IF(context->GetRawTilingData() == nullptr,
                OP_LOGE(unlikely(((context) == nullptr) || (context)->GetNodeName() == nullptr) ? "nil" :(context)->GetNodeName(),
                "context->GetRawTilingData() is nullptr!"),
                return);
    nsaTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(nsaTilingData.GetDataSize());
}

static ge::graphStatus CheckParams(const gert::TilingContext *context)
{
    if (context->GetInputShape(QUERY_INPUT_INDEX) != nullptr && context->GetInputShape(KEY_INPUT_INDEX) != nullptr &&
        context->GetInputShape(VALUE_INPUT_INDEX) != nullptr && context->GetAttrs() != nullptr) {
        auto &queryShape = context->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
        auto &keyShape = context->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
        auto &valueShape = context->GetInputShape(VALUE_INPUT_INDEX)->GetStorageShape();
        const char *inputLayout = context->GetAttrs()->GetAttrPointer<char>(INPUTLAYOUT_ATTRS_INDEX);
        if (strlen(inputLayout) == 3) { // 3: BSH or SBH
            if (inputLayout[0] == 'B') {
                // layout is BSH
                OP_CHECK_IF((queryShape.GetDim(0) != keyShape.GetDim(0)),
                        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "query or key shape is invalid"),
                        return ge::GRAPH_FAILED);
            } else {
                if (inputLayout[0] == 'T') { // TND  N1 != N2
                    // q_D != k_D
                    OP_CHECK_IF((queryShape.GetDim(2) != keyShape.GetDim(2)),
                            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "query or key shape is invalid"),
                            return ge::GRAPH_FAILED);
                    return ge::SUCCESS;
                }
                // layout is SBH
                OP_CHECK_IF((queryShape.GetDim(1) != keyShape.GetDim(1)),
                        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "query or key shape is invalid"),
                        return ge::GRAPH_FAILED);
            }
            // kD < vD
            OP_CHECK_IF((keyShape.GetDim(2) < valueShape.GetDim(2)),
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "key or value shape is invalid"),
                return ge::GRAPH_FAILED);
        } else if (strlen(inputLayout) == 4) { // 4: layout is BNSD or BSND
            OP_CHECK_IF((queryShape.GetDim(0) != keyShape.GetDim(0)),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "query or key shape is invalid"), return ge::GRAPH_FAILED);
            OP_CHECK_IF((queryShape.GetDim(3) != keyShape.GetDim(3)),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "query or key shape is invalid"), return ge::GRAPH_FAILED);
            OP_CHECK_IF((keyShape.GetDim(3) < valueShape.GetDim(3)),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "key or value shape is invalid"), return ge::GRAPH_FAILED);
        } else {
            OP_LOGW(context, "invalid input_layout[%s].", inputLayout);
            return ge::GRAPH_FAILED;
        }
        return ge::SUCCESS;
    }
    OP_LOGW(context, "fail to get shape or attr from context");
    return ge::GRAPH_FAILED;
}
static uint32_t CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum)
    {
        uint32_t ration;
        if (aicCoreNum == 0U || aivCoreNum == 0U || aicCoreNum > aivCoreNum) {
            return sliceNum;
        }
        ration = aivCoreNum / aicCoreNum;
        return (sliceNum + (ration - 1U)) / ration;
    }

static bool IsEmptyInput(gert::TilingContext *context)
{
    auto attenOutShape = context->GetOutputShape(ATTENTIONOUT_OUPUT_INDEX);
    OP_CHECK_IF(attenOutShape == nullptr,
                OP_LOGE(unlikely(((context) == nullptr) || (context)->GetNodeName() == nullptr) ? "nil" :(context)->GetNodeName(),
                "attenOutShape is nullptr!"),
                return false);
    auto queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OP_CHECK_IF(queryShape == nullptr,
                OP_LOGE(unlikely(((context) == nullptr) || (context)->GetNodeName() == nullptr) ? "nil" :(context)->GetNodeName(),
                "queryShape is nullptr!"),
                return false);
    auto keyShape = context->GetInputShape(KEY_INPUT_INDEX);
    OP_CHECK_IF(keyShape == nullptr,
                OP_LOGE(unlikely(((context) == nullptr) || (context)->GetNodeName() == nullptr) ? "nil" :(context)->GetNodeName(),
                "keyShape is nullptr!"),
                return false);
    auto softmaxSumShape = context->GetOutputShape(SOFTMAXSUM_OUPUT_INDEX);
    OP_CHECK_IF(softmaxSumShape == nullptr,
                OP_LOGE(unlikely(((context) == nullptr) || (context)->GetNodeName() == nullptr) ? "nil" :(context)->GetNodeName(),
                "softmaxSumShape is nullptr!"),
                return false);
    int64_t attentionOutShapeSize = attenOutShape->GetStorageShape().GetShapeSize();
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
        auto kernelType = context->GetInputDesc(KEY_INPUT_INDEX)->GetDataType();
        NsaSelectedAttentionEmptyInputTiling emptyInputTiling;
        auto compileInfoPtr = context->GetCompileInfo<NsaSelectedAttentionCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
                return false);
        uint32_t coreNum = compileInfoPtr->aivNum;
        OP_CHECK_IF((coreNum <= 0),
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "platform info is invalid, coreNum=%u.", coreNum), return false);
        OP_CHECK_IF((kernelType != ge::DT_FLOAT16 && kernelType != ge::DT_FLOAT && kernelType != ge::DT_BF16),
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "kernelType is invalid, kernelType is %d", kernelType),
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
            if (attentionOutBlockSize % coreNum == 0U) {
                attentionOutTailCoreDataSize = 0ULL;
                attentionOutFormerNum = coreNum;
                attentionOutTailNum = 0U;
                attentionOutSingleCoreDataSize =
                    attentionOutBlockSize / coreNum * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(kernelType);
                attentionOutLastCoreDataSize =
                    attentionOutSingleCoreDataSize -
                    (attentionOutBlockSize * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(kernelType) -
                    attentionOutShapeSize);
                attentionOutLastCoreIndex = (attentionOutFormerNum - 1U) * attentionOutSingleCoreDataSize;
            } else {
                attentionOutTailCoreDataSize =
                    attentionOutBlockSize / coreNum * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(kernelType);
                attentionOutSingleCoreDataSize =
                    attentionOutTailCoreDataSize + MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(kernelType);
                if (attentionOutBlockSize > coreNum) {
                    attentionOutFormerNum =
                        static_cast<uint32_t>(attentionOutBlockSize % static_cast<uint64_t>(coreNum));
                    attentionOutTailNum = coreNum - attentionOutFormerNum;
                    attentionOutLastCoreIndex = attentionOutFormerNum * attentionOutSingleCoreDataSize +
                                                (attentionOutTailNum - 1U) * attentionOutTailCoreDataSize;
                    attentionOutLastCoreDataSize =
                        attentionOutTailCoreDataSize -
                        (attentionOutSingleCoreDataSize * attentionOutFormerNum +
                         attentionOutTailCoreDataSize * attentionOutTailNum - attentionOutShapeSize);
                } else {
                    attentionOutFormerNum = attentionOutBlockSize;
                    attentionOutTailNum = 0U;
                    attentionOutLastCoreIndex = (attentionOutFormerNum - 1U) * attentionOutSingleCoreDataSize;
                    attentionOutLastCoreDataSize =
                        attentionOutSingleCoreDataSize -
                        (attentionOutFormerNum * attentionOutSingleCoreDataSize - attentionOutShapeSize);
                }
            }
        } else {
            attentionOutFormerNum = 0U;
            attentionOutTailNum = 0U;
            attentionOutSingleCoreDataSize = 0ULL;
            attentionOutTailCoreDataSize = 0ULL;
            attentionOutLastCoreDataSize = 0ULL;
            attentionOutLastCoreIndex = 0ULL;
        }

        if (softmaxSumBlockSize % coreNum == 0U) {
            softmaxMaxSingleCoreDataSize =
                softmaxSumBlockSize / coreNum * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(ge::DT_FLOAT);
            softmaxMaxTailCoreDataSize = 0ULL;
            softmaxMaxFormerNum = coreNum;
            softmaxMaxTailNum = 0U;
        } else {
            if (softmaxSumBlockSize > coreNum) {
                softmaxMaxFormerNum = static_cast<uint32_t>(softmaxSumBlockSize % static_cast<uint64_t>(coreNum));
                softmaxMaxTailNum = coreNum - softmaxMaxFormerNum;
            } else {
                softmaxMaxFormerNum = softmaxSumBlockSize;
                softmaxMaxTailNum = 0U;
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
        emptyInputTiling.NsaSelectedAttentionSetEmptyInputTilingData(context, emptyInputTiling.tilingData);
        emptyInputTiling.GetTilingKeyAttentionScore4EmptyInput(tilingKey, context);
        context->SetTilingKey(tilingKey);
        uint32_t aivActualNum =
            std::max((attentionOutFormerNum + attentionOutTailNum), (softmaxMaxFormerNum + softmaxMaxTailNum));
        context->SetBlockDim(CalcTschBlockDim(aivActualNum, 0, compileInfoPtr->aivNum));
        size_t *workspaces = context->GetWorkspaceSizes(1);
        // workspace上预留100M
        workspaces[0] = static_cast<size_t>(100 * 1024 * 1024);

        return true;
    }
    return false;
}

ASCENDC_EXTERN_C ge::graphStatus TilingNsaSelectedAttention(gert::TilingContext *context)
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

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForNsaSelectedAttention(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                OP_LOGE(context->GetNodeName(),
                "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<NsaSelectedAttentionCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                OP_LOGE(context->GetNodeName(),
                "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2CacheSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NsaSelectedAttention)
    .Tiling(TilingNsaSelectedAttention)
    .TilingInputsDataDependency({5, 6})
    .TilingParse<NsaSelectedAttentionCompileInfo>(TilingPrepareForNsaSelectedAttention);  // 向框架注册入口函数

} // namespace optiling
