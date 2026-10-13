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
 * \file flash_attention_score_grad_tiling_s1s2_bn2gs1s2_basic.cpp
 * \brief
 */
#include "flash_attention_score_grad_tiling_s1s2_bn2gs1s2_basic.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_type.h"
#include "../../op_kernel/arch32/flash_attention_score_grad_tiling.h"
#include "../../op_kernel/arch32/flash_attention_score_grad_template_tiling_key.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

constexpr int64_t ATTEN_MASK_DIM_NUM = 2;
constexpr int64_t ATTEN_MASK_NUM = 2048;
constexpr int64_t S_MAX = 32768;
constexpr int64_t DIM_NUM_2 = 2;
constexpr int64_t SPECIAL_HEADDIM_128 = 128;
constexpr int64_t C0_SIZE = 16;

ge::graphStatus FlashAttentionScoreGraTilingMla::GetShapeAttrsInfo()
{
    auto ret = SetAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = SetBaseInfo();
    return ret;
}


ge::graphStatus FlashAttentionScoreGraTilingMla::SetAttrsInfo()
{
    // option input
    auto pseShape = context_->GetOptionalInputShape(PSE_SHIFT);
    auto dropMask = context_->GetOptionalInputShape(DROP_MASK);
    auto attenMaskShape = context_->GetOptionalInputShape(ATTEN_MASK);

    fBaseParams.attenMskEnable = attenMaskShape != nullptr;
    if (fBaseParams.attenMskEnable) {
        auto storageShape = attenMaskShape->GetStorageShape();
        auto maskShapeDims = storageShape.GetDimNum();
        if (maskShapeDims != ATTEN_MASK_DIM_NUM) {
            OP_LOGI(context_, "FlashAttentionScoreGradMla not support attenMaskShape dim num: %lu", maskShapeDims);
            return ge::GRAPH_PARAM_INVALID;
        }

        auto dim0 = storageShape.GetDim(0);
        auto dim1 = storageShape.GetDim(1);
        if ((dim0 != ATTEN_MASK_NUM) && (dim1 != ATTEN_MASK_NUM)) {
            OP_LOGI(context_, "FlashAttentionScoreGradMla attenMaskShape should be 2048 * 2048.");
            return ge ::GRAPH_PARAM_INVALID;
        }
    }

    // attr
    fBaseParams.scaleValue = *context_->GetAttrs()->GetAttrPointer<float>(SCALE_VALUE);
    fBaseParams.keepProb = *context_->GetAttrs()->GetAttrPointer<float>(KEEP_PROB);
    fBaseParams.inputLayout = context_->GetAttrs()->GetAttrPointer<char>(INPUT_LAYOUT);
    fBaseParams.sparseMode = *context_->GetAttrs()->GetAttrPointer<int>(SPARSE_MODE);

    OP_CHECK_IF((fBaseParams.keepProb <= 0 || fBaseParams.keepProb > 1),
               OP_LOGE(context_, "keepProb is illegal."), return ge::GRAPH_FAILED);

    fBaseParams.dropMskEnable = (dropMask != nullptr || fBaseParams.keepProb < 1.0f);
    if (context_->GetAttrs()->GetAttrNum() > static_cast<size_t>(PSETYPE)) {
        auto pseType = *context_->GetAttrs()->GetAttrPointer<int>(PSETYPE);
        fBaseParams.pseEnable = (pseShape != nullptr || pseType != 1);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGraTilingMla::SetBaseInfo()
{
    if (strcmp(fBaseParams.inputLayout, TND_STR) != 0) {
        OP_LOGI(context_, "FlashAttentionScoreGradMla only support TND inputLayout, now is %s.",
                  fBaseParams.inputLayout);
        return ge ::GRAPH_PARAM_INVALID;
    }
    const gert::StorageShape *queryShape = context_->GetInputShape(0);
    const gert::StorageShape *keyShape = context_->GetInputShape(1);
    const gert::StorageShape *valueShape = context_->GetInputShape(2);

    auto actualSeqQLenTensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_LEN);
    auto actualSeqKvLenTensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_KV_LEN);
    auto qStartTensor = context_->GetOptionalInputTensor(Q_START_IDX);
    auto kvStartTensor = context_->GetOptionalInputTensor(KV_START_IDX);

    if (actualSeqQLenTensor == nullptr || actualSeqKvLenTensor == nullptr) {
        OP_LOGE(context_, "actualSeqQLenTensor or actualSeqKvLenTensor is nullptr");
        return ge::GRAPH_FAILED;
    }
    if (qStartTensor != nullptr || kvStartTensor != nullptr) {
        OP_LOGI(context_, "FlashAttentionScoreGradMla not support q_start_idx or kv_start_idx.");
        return ge::GRAPH_PARAM_INVALID;
    }

    const int64_t *qValue = actualSeqQLenTensor->GetData<int64_t>();
    const int64_t *kvValue = actualSeqKvLenTensor->GetData<int64_t>();
    const size_t seqQShapeSize = static_cast<size_t>(actualSeqQLenTensor->GetShapeSize());
    const size_t kvSeqShapeSize = static_cast<size_t>(actualSeqKvLenTensor->GetShapeSize());

    OP_CHECK_IF((qValue == nullptr || kvValue == nullptr),
               OP_LOGE(context_, "qValue or kvValue is nullptr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((seqQShapeSize != kvSeqShapeSize),
               OP_LOGE(context_, "actualSeqQLenTensor shapeSize is not equal actualSeqKvLenTensor"),
               return ge::GRAPH_FAILED);

    for (size_t i = 0; i < seqQShapeSize; i++) {
        int64_t qSeqLen = (i == 0 ? qValue[i] : (qValue[i] - qValue[i - 1]));
        int64_t kvSeqLen = (i == 0 ? kvValue[i] : (kvValue[i] - kvValue[i - 1]));
        fBaseParams.actualSeqQlen.push_back(qSeqLen);
        fBaseParams.actualSeqKvlen.push_back(kvSeqLen);
        if (qSeqLen != kvSeqLen) {
            fBaseParams.eaqualActSeqLen = false;
        }
    }

    uint64_t tailZeroCount = 0;
    for (auto i = seqQShapeSize - 1; i >= 1; --i) {
        if (fBaseParams.actualSeqQlen[i] > 0 && fBaseParams.actualSeqKvlen[i] > 0) {
            break;   
        }
        ++tailZeroCount;
    }

    // base shape
    fBaseParams.b = seqQShapeSize - tailZeroCount;
    fBaseParams.queryType = static_cast<uint32_t>(context_->GetInputDesc(QUERY)->GetDataType());
    fBaseParams.t1 = queryShape->GetStorageShape().GetDim(0);
    fBaseParams.t2 = keyShape->GetStorageShape().GetDim(0);
    fBaseParams.n1 = queryShape->GetStorageShape().GetDim(1);
    fBaseParams.n2 = keyShape->GetStorageShape().GetDim(1);
    fBaseParams.d = queryShape->GetStorageShape().GetDim(DIM_NUM_2);
    fBaseParams.dv = valueShape->GetStorageShape().GetDim(DIM_NUM_2);
    OP_CHECK_IF(fBaseParams.n2 == 0, OP_LOGE(context_, "dim N2 is 0."), return ge::GRAPH_FAILED);
    fBaseParams.g = fBaseParams.n1 / fBaseParams.n2;
    fBaseParams.s1 = *std::max_element(fBaseParams.actualSeqQlen.begin(), fBaseParams.actualSeqQlen.end());
    fBaseParams.s1 = *std::max_element(fBaseParams.actualSeqKvlen.begin(), fBaseParams.actualSeqKvlen.end());

    return CheckTndShapeValid(context_, fBaseParams.t1, fBaseParams.n1, fBaseParams.d);
}

bool FlashAttentionScoreGraTilingMla::IsCapable()
{
    /**
     * The basic template does not support the following features:
     * 1. The dtype of float32
     * 2. Layouts other than TND
     * 3. Deterministic
     * 4. Sparse mode not equal to 0 or 2 or 3
     * 5. PSE
     * 6. Drop mask
     */
    auto sinkShape = context_->GetOptionalInputShape(SINK_IN);
    if (sinkShape != nullptr && sinkShape->GetStorageShape().GetDimNum() == 1 ) {
        return false;
    }
    if (fBaseParams.queryType == ge::DT_FLOAT) {
        OP_LOGI(context_, "FlashAttentionScoreGradMla does not support float32.");
        return false;
    } else if (strcmp(fBaseParams.inputLayout, TND_STR) != 0) {
        OP_LOGI(context_, "FlashAttentionScoreGradMla does not support Layouts other than TND, now is %s.",
                  fBaseParams.inputLayout);
        return false;
    } else if (context_->GetDeterministic() == 1) {
        OP_LOGI(context_, "FlashAttentionScoreGradMla does not support deterministic.");
        return false;
    } else if (fBaseParams.pseEnable) {
        OP_LOGI(context_, "FlashAttentionScoreGradMla does not support PSE feature.");
        return false;
    } else if (fBaseParams.dropMskEnable) {
        OP_LOGI(context_, "FlashAttentionScoreGradMla does not support dropMask feature.");
        return false;
    } else if (!IsAttenMskCapable()) {
        return false;
    }

    if (!IsShapeCapable()) {
        OP_LOGI(context_, "FlashAttentionScoreGradMla does not support current input shape.");
        return false;
    }

    OP_LOGI(context_, "FlashAttentionScoreGradMla is capable.");
    return true;
}

bool FlashAttentionScoreGraTilingMla::IsAttenMskCapable()
{
    if (fBaseParams.sparseMode != NO_MASK && fBaseParams.sparseMode != RIGHT_DOWN_CAUSAL &&
        fBaseParams.sparseMode != LEFT_UP_CAUSAL) {
        OP_LOGI(context_, "FlashAttentionScoreGradMla only support sparseMode=%d or %d or %d, now is %u.", NO_MASK,
                LEFT_UP_CAUSAL, RIGHT_DOWN_CAUSAL, fBaseParams.sparseMode);
        return false;
    } else if (fBaseParams.sparseMode == NO_MASK && fBaseParams.attenMskEnable) {
        OP_LOGI(context_, "FlashAttentionScoreGradMla sparseMode=%u, but attenMask is exsit.",
                  fBaseParams.sparseMode);
        return false;
    } else if (fBaseParams.sparseMode == LEFT_UP_CAUSAL && !fBaseParams.attenMskEnable) {
        OP_LOGI(context_, "FlashAttentionScoreGradMla sparseMode=%u, but attenMask is nullptr.",
                  fBaseParams.sparseMode);
        return false;
    } else if (fBaseParams.sparseMode == RIGHT_DOWN_CAUSAL && !fBaseParams.attenMskEnable) {
        OP_LOGI(context_, "FlashAttentionScoreGradMla sparseMode=%u, but attenMask is nullptr.",
                  fBaseParams.sparseMode);
        return false;
    }

    return true;
}

bool FlashAttentionScoreGraTilingMla::IsShapeCapable()
{
    if (fBaseParams.d != fBaseParams.dv || fBaseParams.d > SPECIAL_HEADDIM_128 || fBaseParams.d % C0_SIZE != 0) {
        return false;
    }

    if (fBaseParams.t1 != fBaseParams.t2 || !fBaseParams.eaqualActSeqLen || fBaseParams.s1 > S_MAX) {
        return false;
    }

    return true;
}

ge::graphStatus FlashAttentionScoreGraTilingMla::DoOpTiling()
{
    auto compileInfoPtr = context_->GetCompileInfo<MultiHeadLatentAttentionGradCompileInfo>();

    int64_t qSize = fBaseParams.t1 * fBaseParams.n1 * fBaseParams.d;
    int64_t kvSize = fBaseParams.t2 * fBaseParams.n2 * fBaseParams.d;
    int64_t sfmgSize = fBaseParams.t1 * fBaseParams.n1 * 8;
    uint32_t vectorCoreNum = compileInfoPtr->aivNum;

    tilingData->mlaTensorTilingData.set_coreNum(vectorCoreNum);
    tilingData->mlaTensorTilingData.set_scaleValue(fBaseParams.scaleValue);
    tilingData->mlaTensorTilingData.set_b(fBaseParams.b);
    tilingData->mlaTensorTilingData.set_t1(fBaseParams.t1);
    tilingData->mlaTensorTilingData.set_t2(fBaseParams.t2);
    tilingData->mlaTensorTilingData.set_n2(fBaseParams.n2);
    tilingData->mlaTensorTilingData.set_g(fBaseParams.g);
    tilingData->mlaTensorTilingData.set_d(fBaseParams.d);
    tilingData->mlaTensorTilingData.set_qSize(qSize);
    tilingData->mlaTensorTilingData.set_kvSize(kvSize);
    tilingData->mlaTensorTilingData.set_sfmgSize(sfmgSize);
    tilingData->mlaTensorTilingData.set_sparseMode(fBaseParams.sparseMode);
    bool tndSoftmaxIn = context_->GetAttrs()->GetAttrNum() > static_cast<size_t>(TND_SOFTMAX_IN) ?
                            *(context_->GetAttrs()->GetAttrPointer<bool>(TND_SOFTMAX_IN)) :
                            false;
    if (tndSoftmaxIn) {
        tilingData->mlaTensorTilingData.set_tndSoftmaxIn(1);
    } else {
        tilingData->mlaTensorTilingData.set_tndSoftmaxIn(0);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGraTilingMla::DoLibApiTiling()
{
    constexpr uint32_t tmpBufferSize = 33 * 1024;
    constexpr uint32_t s1VecSize = 64;
    constexpr uint32_t s2VecSize = 128;
    int64_t d = tilingData->mlaTensorTilingData.get_d();
    auto softmaxShape = ge::Shape({s1VecSize, s2VecSize});
    AscendC::SoftMaxTilingFunc(softmaxShape, sizeof(float), tmpBufferSize,
                               tilingData->mlaTensorTilingData.softmaxTilingData);

    // softmaxGrad tiling
    constexpr uint32_t inputBufferLen = 24 * 1024;
    constexpr uint32_t castBufferLen = 48 * 1024; // castBuffer 48K*2=96K
    // uint32_t outputBufferLen = CeilDiv(castBufferLen, d) * 8; // 输出(s1,8)
    uint32_t outputBufferLen = (castBufferLen + d - 1) / d * 8; // 输出(s1,8)
    uint32_t tempBufferLen = 40 * 1024 - outputBufferLen;

    int64_t singleLoopNBurstNum = inputBufferLen / sizeof(float) / d;
    auto softmaxGradShape = ge::Shape({singleLoopNBurstNum, d});
    AscendC::SoftMaxGradTilingFunc(softmaxGradShape, sizeof(float), tempBufferLen,
                                   tilingData->mlaTensorTilingData.softmaxGradTilingData, true);

    return ge::GRAPH_SUCCESS;
}

uint64_t FlashAttentionScoreGraTilingMla::GetTilingKey() const
{
    uint64_t tilingKey = 0;
    auto queryType = fBaseParams.queryType;
    if (queryType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(9,9,9,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
    } else if (queryType == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(9,9,9,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
    }
    OP_LOGI(context_, "FAGTiling MLA DoTiling success, tilingkey is %lu.", tilingKey);
    return tilingKey;
}

ge::graphStatus FlashAttentionScoreGraTilingMla::GetWorkspaceSize()
{
    auto compileInfoPtr = context_->GetCompileInfo<MultiHeadLatentAttentionGradCompileInfo>();
    uint32_t cubeCoreNum = compileInfoPtr->aicNum;
    constexpr size_t WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
    constexpr size_t GM_ALIGN = 512;
    constexpr size_t DB_NUM = 2;
    constexpr size_t matmulSize = 16 * 128 * 128;

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    size_t workspaceOffset = WORKSPACE_RSV_BYTE;

    // matmal3 q
    tilingData->mlaTensorTilingData.set_dqWorkSpaceOffset(workspaceOffset);
    workspaceOffset =
        (workspaceOffset + tilingData->mlaTensorTilingData.get_qSize() * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // matmal3 k
    tilingData->mlaTensorTilingData.set_dkWorkSpaceOffset(workspaceOffset);
    workspaceOffset = (workspaceOffset + tilingData->mlaTensorTilingData.get_kvSize() * sizeof(float) + GM_ALIGN) /
                      GM_ALIGN * GM_ALIGN;
    // matmal3 v
    tilingData->mlaTensorTilingData.set_dvWorkSpaceOffset(workspaceOffset);
    workspaceOffset = (workspaceOffset + tilingData->mlaTensorTilingData.get_kvSize() * sizeof(float) + GM_ALIGN) /
                      GM_ALIGN * GM_ALIGN;
    // sfmg workspace
    tilingData->mlaTensorTilingData.set_sfmgWorkspaceOffset(workspaceOffset);
    workspaceOffset = (workspaceOffset + tilingData->mlaTensorTilingData.get_sfmgSize() * sizeof(float) + GM_ALIGN) /
                      GM_ALIGN * GM_ALIGN;

    // matmal1/matmal2 workspace size
    tilingData->mlaTensorTilingData.set_mm1WorkspaceOffset(workspaceOffset);
    workspaceOffset =
        (workspaceOffset + cubeCoreNum * matmulSize * sizeof(float) * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    tilingData->mlaTensorTilingData.set_mm2WorkspaceOffset(workspaceOffset);
    workspaceOffset =
        (workspaceOffset + cubeCoreNum * matmulSize * sizeof(float) * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    workspaceOffset += WORKSPACE_RSV_BYTE;
    // set workspace
    workspaces[0] = (workspaceOffset - 0);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGraTilingMla::PostTiling()
{
    auto compileInfoPtr = context_->GetCompileInfo<MultiHeadLatentAttentionGradCompileInfo>();

    // setBlockDim
    uint32_t cubeCoreNum = compileInfoPtr->aicNum;
    context_->SetBlockDim(cubeCoreNum);
    context_->SetScheduleMode(1);

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus FlashAttentionScoreGraTilingMla::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo *>(
            context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile_info is null"),
                   return ge::GRAPH_FAILED);

        aicoreParams_.numBlocks = compileInfoPtr->aivNum;
        aicoreParams_.aicNum = compileInfoPtr->aicNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0aSize = compileInfoPtr->l0aSize;
        aicoreParams_.l0bSize = compileInfoPtr->l0bSize;
        aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        aicoreParams_.aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, aicoreParams_.l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, aicoreParams_.l0bSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    }

    OP_CHECK_IF((aicoreParams_.numBlocks == 0) || (aicoreParams_.aicNum == 0),
               OP_LOGE(context_, "num of coreNum(aivNum) is %lu, num of aicNum is %lu.",
                                           aicoreParams_.numBlocks, aicoreParams_.aicNum),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForMultiHeadLatentAttentionGrad(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("TilingPrepare", "context is null."), return ge::GRAPH_FAILED);
    
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                OP_LOGE(context, "platformInfoPtr is null."), return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<MultiHeadLatentAttentionGradCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                OP_LOGE(context, "compileInfoPtr is null."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2CacheSize);

    OP_LOGI(context,
              "parse TilingParseContext succ. aivNum:%u, aicNum:%u, "
              "ubSize:%lu, l1Size:%lu, l0aSize:%lu, l0bSize:%lu, l0cSize:%lu, l2CacheSize:%lu",
              compileInfoPtr->aivNum, compileInfoPtr->aicNum, compileInfoPtr->ubSize, compileInfoPtr->l1Size,
              compileInfoPtr->l0aSize, compileInfoPtr->l0bSize, compileInfoPtr->l0cSize, compileInfoPtr->l2CacheSize);

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(
    FlashAttentionScoreGrad, FlashAttentionScoreGraTilingMla,
    std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}),
    1001);
} // namespace optiling