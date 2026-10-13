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
 * \file flash_attention_score_grad_tiling_s1s2_bn2gs1s2_basic_det.cpp
 * \brief
 */
#include "flash_attention_score_grad_tiling_common.h"
#include "flash_attention_score_grad_tiling_s1s2_bn2gs1s2_basic_det.h"
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
constexpr int64_t AIV_NUM_910B2 = 48;

ge::graphStatus FlashAttentionScoreGraTilingBasicDet::GetShapeAttrsInfo()
{
    auto ret = SetAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = SetBaseInfo();

    if(fBaseParams.sparseMode == RIGHT_DOWN_CAUSAL){
        fBaseParams.preTockens = fBaseParams.s2;
        fBaseParams.nextTockens = 0;
    }

    return ret;
}


ge::graphStatus FlashAttentionScoreGraTilingBasicDet::SetAttrsInfo()
{
    // option input
    auto pseShape = context_->GetOptionalInputShape(PSE_SHIFT);

    // attr
    fBaseParams.scaleValue = *context_->GetAttrs()->GetAttrPointer<float>(SCALE_VALUE);
    fBaseParams.keepProb = *context_->GetAttrs()->GetAttrPointer<float>(KEEP_PROB);
    fBaseParams.inputLayout = context_->GetAttrs()->GetAttrPointer<char>(INPUT_LAYOUT);
    fBaseParams.preTockens = *context_->GetAttrs()->GetAttrPointer<int64_t>(PRE_TOKENS);
    fBaseParams.nextTockens = *context_->GetAttrs()->GetAttrPointer<int64_t>(NEXT_TOKENS);
    fBaseParams.sparseMode = *context_->GetAttrs()->GetAttrPointer<int>(SPARSE_MODE);
    fBaseParams.dqPostAbsorb = 0;

    OP_CHECK_IF((fBaseParams.keepProb <= 0.0 || fBaseParams.keepProb > 1.0),
               OP_LOGE(context_, "keepProb is illegal."), return ge::GRAPH_FAILED);

    if (context_->GetAttrs()->GetAttrNum() > static_cast<size_t>(PSETYPE)) {
        auto pseType = *context_->GetAttrs()->GetAttrPointer<int>(PSETYPE);
        fBaseParams.pseEnable = (pseShape != nullptr || pseType != 1);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGraTilingBasicDet::SetBaseInfo()
{
    int64_t headNum = *context_->GetAttrs()->GetAttrPointer<int>(HEAD_NUM);
    const gert::StorageShape *queryShape = context_->GetInputShape(0);
    const gert::StorageShape *keyShape = context_->GetInputShape(1);
    const gert::StorageShape *valueShape = context_->GetInputShape(2);
    if (strcmp(fBaseParams.inputLayout, BSH_STR) == 0) {
        fBaseParams.queryType = static_cast<uint32_t>(context_->GetInputDesc(QUERY)->GetDataType());
        fBaseParams.b = queryShape->GetStorageShape().GetDim(DIM_0);
        OP_CHECK_IF(keyShape->GetStorageShape().GetDim(DIM_2) == 0,
                   OP_LOGE(context_, "dim N2 is 0."), return ge::GRAPH_FAILED);
        fBaseParams.g = queryShape->GetStorageShape().GetDim(DIM_2) / keyShape->GetStorageShape().GetDim(DIM_2);
        OP_CHECK_IF(fBaseParams.g == 0, OP_LOGE(context_, "g is 0"), return ge::GRAPH_FAILED);
        fBaseParams.n1 = headNum;
        fBaseParams.n2 = headNum / fBaseParams.g;
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(DIM_1);
        fBaseParams.d = queryShape->GetStorageShape().GetDim(DIM_2) / headNum; // H=N*D
        fBaseParams.dv = valueShape->GetStorageShape().GetDim(DIM_NUM_2) / fBaseParams.n2;
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(DIM_1);
        fBaseParams.t1 = fBaseParams.b * fBaseParams.s1;
        return ge::GRAPH_SUCCESS;
    } else if (strcmp(fBaseParams.inputLayout, BNSD_STR) == 0) {
        fBaseParams.queryType = static_cast<uint32_t>(context_->GetInputDesc(QUERY)->GetDataType());
        fBaseParams.b = queryShape->GetStorageShape().GetDim(DIM_0);
        OP_CHECK_IF(keyShape->GetStorageShape().GetDim(DIM_1) == 0, OP_LOGE(context_, "dim N2 is 0."), return ge::GRAPH_FAILED);
        fBaseParams.g = queryShape->GetStorageShape().GetDim(DIM_1) / keyShape->GetStorageShape().GetDim(DIM_1);
        OP_CHECK_IF(fBaseParams.g == 0, OP_LOGE(context_, "g is 0"), return ge::GRAPH_FAILED);
        fBaseParams.n1 = queryShape->GetStorageShape().GetDim(DIM_1);
        fBaseParams.n2 = keyShape->GetStorageShape().GetDim(DIM_1);
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(DIM_2);
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(DIM_2);
        fBaseParams.d = queryShape->GetStorageShape().GetDim(DIM_3);
        fBaseParams.dv = valueShape->GetStorageShape().GetDim(DIM_3);
        fBaseParams.t1 = fBaseParams.b * fBaseParams.s1;
        return ge::GRAPH_SUCCESS;
    } else if (strcmp(fBaseParams.inputLayout, SBH_STR) == 0) {
        fBaseParams.queryType = static_cast<uint32_t>(context_->GetInputDesc(QUERY)->GetDataType());
        fBaseParams.b = queryShape->GetStorageShape().GetDim(DIM_1);
        OP_CHECK_IF(keyShape->GetStorageShape().GetDim(DIM_2) == 0, OP_LOGE(context_, "dim N2 is 0."), return ge::GRAPH_FAILED);
        fBaseParams.g = queryShape->GetStorageShape().GetDim(DIM_2) / keyShape->GetStorageShape().GetDim(DIM_2);
        OP_CHECK_IF(fBaseParams.g == 0, OP_LOGE(context_, "g is 0"), return ge::GRAPH_FAILED);
        fBaseParams.n1 = headNum;
        fBaseParams.n2 = headNum / fBaseParams.g;
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(DIM_0);
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(DIM_0);
        fBaseParams.d = queryShape->GetStorageShape().GetDim(DIM_2) / headNum; // H=N*D
        fBaseParams.dv = valueShape->GetStorageShape().GetDim(DIM_NUM_2) / fBaseParams.n2;
        fBaseParams.t1 = fBaseParams.b * fBaseParams.s1;
        return ge::GRAPH_SUCCESS;
    } else if(strcmp(fBaseParams.inputLayout, TND_STR) == 0) {
        auto actualSeqQLenTensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_LEN);
        auto actualSeqKvLenTensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_KV_LEN);
        auto qStartTensor = context_->GetOptionalInputTensor(Q_START_IDX);
        auto kvStartTensor = context_->GetOptionalInputTensor(KV_START_IDX);

        if (actualSeqQLenTensor == nullptr || actualSeqKvLenTensor == nullptr) {
            OP_LOGE(context_, "actualSeqQLenTensor or actualSeqKvLenTensor is nullptr");
            return ge::GRAPH_FAILED;
        }
        if (qStartTensor != nullptr || kvStartTensor != nullptr) {
            OP_LOGI(context_, "FlashAttentionScoreGraTilingBasicDet not support q_start_idx or kv_start_idx.");
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
            fBaseParams.sumS1S2Product += qSeqLen * kvSeqLen;
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
        fBaseParams.s2 = *std::max_element(fBaseParams.actualSeqKvlen.begin(), fBaseParams.actualSeqKvlen.end());
        return CheckTndShapeValid(context_, fBaseParams.t1, fBaseParams.n1, fBaseParams.d);
    } else {
        OP_LOGI(context_, "FlashAttentionScoreGraTilingBasicDet only support TND, BSH, BNSD, SBH inputLayout, now is %s.",
                  fBaseParams.inputLayout);
        return ge ::GRAPH_PARAM_INVALID;
    }
    
}

bool FlashAttentionScoreGraTilingBasicDet::IsCapable()
{
    /**
     * The basic template does not support the following features:
     * 1. The dtype of float32
     * 2. Layouts other than TND
     * 3. Not Deterministic
     * 4. Sparse mode not equal to 0 or 3
     * 5. PSE
     * 6. headDim <= 128
     */
    if (context_->GetDeterministic() == 0) {
        OP_LOGI(context_, "FlashAttentionScoreGraTilingBasicDet only support Deterministic.");
        return false;
    } 
    else if (fBaseParams.queryType == ge::DT_FLOAT) {
        OP_LOGI(context_, "FlashAttentionScoreGraTilingBasicDet does not support float32.");
        return false;
    } else if (fBaseParams.pseEnable) {
        OP_LOGI(context_, "FlashAttentionScoreGraTilingBasicDet does not support PSE feature.");
        return false;
    } else if (!IsAttenMskCapable()) {
        return false;
    } else if (!IsDropMskCapable()) {
        return false;
    } else if (!IsShapeCapable()) {
        OP_LOGI(context_, "FlashAttentionScoreGraTilingBasicDet does not support current input shape.");
        return false;
    }
    OP_LOGI(context_, "FlashAttentionScoreGraTilingBasicDet is capable.");


    return true;
}

bool FlashAttentionScoreGraTilingBasicDet::IsAttenMskCapable()
{
    auto attenMaskShape = context_->GetOptionalInputShape(ATTEN_MASK);
    fBaseParams.attenMskEnable = attenMaskShape != nullptr;

    if(fBaseParams.sparseMode == NO_MASK){
        if(fBaseParams.attenMskEnable){
            OP_LOGI(context_, "FlashAttentionScoreGraTilingBasicDet sparseMode=%u, but attenMask is exsit.",
                fBaseParams.sparseMode);
            return false;
        }
        return true;
    }
    if (attenMaskShape == nullptr){
        return false;
    }
    auto storageShape = attenMaskShape->GetStorageShape();
    auto attenMaskDim0 = storageShape.GetDim(0);
    auto attenMaskDim1 = storageShape.GetDim(1);

    if(fBaseParams.sparseMode != NO_MASK && !fBaseParams.attenMskEnable){
        OP_LOGI(context_, "FlashAttentionScoreGraTilingBasicDet sparseMode=%u, but attenMask not exsit.",
            fBaseParams.sparseMode);
        return false;
    }

    if (fBaseParams.sparseMode == ALL_MASK &&
                (attenMaskDim0 != fBaseParams.s1 || attenMaskDim1 != fBaseParams.s2)){
        OP_LOGI(context_, "FlashAttentionScoreGraTilingBasicDet not support current shape of attenMask.");
        return false;
    } else if (fBaseParams.sparseMode == RIGHT_DOWN_CAUSAL &&
                (attenMaskDim0 != ATTEN_MASK_NUM || attenMaskDim1 != ATTEN_MASK_NUM)){
        OP_LOGI(context_, "FlashAttentionScoreGraTilingBasicDet not support current shape of attenMask.");
        return false;
    } else if (fBaseParams.sparseMode == BAND){
        OP_LOGI(context_, "FlashAttentionScoreGraTilingBasicDet not support current shape of attenMask.");
        return false;
    }

    if (fBaseParams.sparseMode != NO_MASK &&
        fBaseParams.sparseMode != ALL_MASK &&
        fBaseParams.sparseMode != RIGHT_DOWN_CAUSAL &&
        fBaseParams.sparseMode != BAND){
        OP_LOGI(context_, "FlashAttentionScoreGraTilingBasicDet not support sparseMode=%u.",
                fBaseParams.sparseMode);
        return false;
    }

    return true;
}


bool FlashAttentionScoreGraTilingBasicDet::IsDropMskCapable()
{
    auto dropMask = context_->GetOptionalInputDesc(DROP_MASK);
    auto dropMaskShape = context_->GetOptionalInputShape(DROP_MASK);
    fBaseParams.dropMskEnable = dropMask != nullptr;

    if(!fBaseParams.dropMskEnable){
        if(fBaseParams.keepProb < 1.0f){
            OP_LOGI(context_, "FlashAttentionScoreGraTilingBasicDet keepProb=%f, but dropMask is not exsit.",
                fBaseParams.keepProb);
            return false;
        }
        return true;
    }

    if(fBaseParams.keepProb == 1.0f){
        OP_LOGI(context_, "FlashAttentionScoreGraTilingBasicDet keepProb=%f, but dropMask is exsit.",
            fBaseParams.keepProb);
        return false;
    }

     auto dropMaskType = dropMask->GetDataType();
    if(dropMaskType != ge::DT_UINT8){
        OP_LOGI(context_, "FlashAttentionScoreGraTilingBasicDet invalid dropMask dtype[%s], only support uint8.",
                    ge::TypeUtils::DataTypeToSerialString(dropMaskType).c_str());
        return false;
    }

    if(dropMaskShape == nullptr){
        OP_LOGI(context_, "FlashAttentionScoreGraTilingBasicDet keepProb [%f], dropMaskShape can not be nullptr",
                    fBaseParams.keepProb);
        return false;
    }

    int64_t ExceptDropMaskSize = AlignUp<int64_t>(fBaseParams.n1 * fBaseParams.sumS1S2Product, 8) / 8;
    int64_t dropMaskDim = dropMaskShape->GetStorageShape().GetDimNum();
    int64_t dropMaskShapeSize = 1;
    for (int64_t i = 0; i < dropMaskDim; ++i) {
        int64_t dimValue = dropMaskShape->GetStorageShape().GetDim(i);
        dropMaskShapeSize *= dimValue;
    }
    if (dropMaskShapeSize <= ExceptDropMaskSize) {
        OP_LOGI(context_, "FlashAttentionScoreGraTilingBasicDet Input dropMask shapeSize is invalid, it should be less than %ld, but got %ld",
                    ExceptDropMaskSize, dropMaskShapeSize);
        return false;
    }

    return false;
}

bool FlashAttentionScoreGraTilingBasicDet::IsShapeCapable()
{
    if (fBaseParams.d != fBaseParams.dv || fBaseParams.d > SPECIAL_HEADDIM_128 || fBaseParams.d % C0_SIZE != 0) {
        return false;
    }
    if (strcmp(fBaseParams.inputLayout, TND_STR) != 0 && fBaseParams.s1 != fBaseParams.s2) {
        return false;
    }
    
    return true;
}

ge::graphStatus FlashAttentionScoreGraTilingBasicDet::DoOpTiling()
{
    auto compileInfoPtr = context_->GetCompileInfo<FlashAttentionGradBasicDetCompileInfo>();
    int64_t qSize = 0;
    int64_t kvSize = 0;
    int64_t sfmgSize = 0;
    if (strcmp(fBaseParams.inputLayout, TND_STR) == 0){
        qSize = fBaseParams.t1 * fBaseParams.n1 * fBaseParams.d;
        kvSize = fBaseParams.t2 * fBaseParams.n2 * fBaseParams.d;
        sfmgSize = fBaseParams.t1 * fBaseParams.n1 * 8;
    } else {
        qSize = fBaseParams.b * fBaseParams.s1 * fBaseParams.n1 * fBaseParams.d;
        kvSize = fBaseParams.b * fBaseParams.s2 * fBaseParams.n2 * fBaseParams.d;
        sfmgSize = fBaseParams.b * fBaseParams.s1 * fBaseParams.n1 * 8;
    }

    uint32_t vectorCoreNum = compileInfoPtr->aivNum;
    tilingData->basicDetTensorTilingData.set_coreNum(vectorCoreNum);
    tilingData->basicDetTensorTilingData.set_scaleValue(fBaseParams.scaleValue);
    tilingData->basicDetTensorTilingData.set_b(fBaseParams.b);
    tilingData->basicDetTensorTilingData.set_t1(fBaseParams.t1);
    tilingData->basicDetTensorTilingData.set_t2(fBaseParams.t2);
    tilingData->basicDetTensorTilingData.set_n2(fBaseParams.n2);
    tilingData->basicDetTensorTilingData.set_g(fBaseParams.g);
    tilingData->basicDetTensorTilingData.set_d(fBaseParams.d);
    tilingData->basicDetTensorTilingData.set_qSize(qSize);
    tilingData->basicDetTensorTilingData.set_kvSize(kvSize);
    tilingData->basicDetTensorTilingData.set_sfmgSize(sfmgSize);
    tilingData->basicDetTensorTilingData.set_keepProb(fBaseParams.keepProb);
    tilingData->basicDetTensorTilingData.set_preTockens(fBaseParams.preTockens);
    tilingData->basicDetTensorTilingData.set_nextTockens(fBaseParams.nextTockens);
    tilingData->basicDetTensorTilingData.set_dqPostAbsorb(fBaseParams.dqPostAbsorb);
    tilingData->basicDetTensorTilingData.set_sparseMode(fBaseParams.sparseMode);
    const char *inputLayout = fBaseParams.inputLayout;
    if (strcmp(inputLayout, BSH_STR) == 0) {
        tilingData->basicDetTensorTilingData.set_layout(static_cast<uint32_t>(KernelInputLayout::BSNGD));
    } else if (strcmp(inputLayout, TND_STR) == 0) {
        tilingData->basicDetTensorTilingData.set_layout(static_cast<uint32_t>(KernelInputLayout::TND));
    } else if (strcmp(inputLayout, BNSD_STR) == 0) {
        tilingData->basicDetTensorTilingData.set_layout(static_cast<uint32_t>(KernelInputLayout::BNGSD));
    } else if (strcmp(inputLayout, SBH_STR) == 0) {
        tilingData->basicDetTensorTilingData.set_layout(static_cast<uint32_t>(KernelInputLayout::SBNGD));
    }else {
        OP_LOGW(context_, "FlashAttentionBasicDet unsupported layout");
        return ge::GRAPH_PARAM_INVALID;
    }
    bool tndSoftmaxIn = context_->GetAttrs()->GetAttrNum() > static_cast<size_t>(TND_SOFTMAX_IN) ?
                            *(context_->GetAttrs()->GetAttrPointer<bool>(TND_SOFTMAX_IN)) :
                            false;
    if (tndSoftmaxIn) {
        tilingData->basicDetTensorTilingData.set_tndSoftmaxIn(1);
    } else {
        tilingData->basicDetTensorTilingData.set_tndSoftmaxIn(0);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGraTilingBasicDet::DoLibApiTiling()
{
    constexpr uint32_t tmpBufferSize = 33 * 1024;
    constexpr uint32_t s1VecSize = 64;
    constexpr uint32_t s2VecSize = 128;
    int64_t d = tilingData->basicDetTensorTilingData.get_d();
    auto softmaxShape = ge::Shape({s1VecSize, s2VecSize});
    AscendC::SoftMaxTilingFunc(softmaxShape, sizeof(float), tmpBufferSize,
                               tilingData->basicDetTensorTilingData.softmaxTilingData);

    // softmaxGrad tiling
    constexpr uint32_t inputBufferLen = 24 * 1024;
    constexpr uint32_t castBufferLen = 48 * 1024; // castBuffer 48K*2=96K
    // uint32_t outputBufferLen = CeilDiv(castBufferLen, d) * 8; // 输出(s1,8)
    uint32_t outputBufferLen = (castBufferLen + d - 1) / d * 8; // 输出(s1,8)
    uint32_t tempBufferLen = 40 * 1024 - outputBufferLen;

    int64_t singleLoopNBurstNum = inputBufferLen / sizeof(float) / d;
    auto softmaxGradShape = ge::Shape({singleLoopNBurstNum, d});
    AscendC::SoftMaxGradTilingFunc(softmaxGradShape, sizeof(float), tempBufferLen,
                                   tilingData->basicDetTensorTilingData.softmaxGradTilingData, true);

    return ge::GRAPH_SUCCESS;
}

uint64_t FlashAttentionScoreGraTilingBasicDet::GetTilingKey() const
{
    uint64_t tilingKey = 0;
    auto queryType = fBaseParams.queryType;
    if (queryType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(9,9,9,0,3,0,0,0,0,0,static_cast<uint8_t>(fBaseParams.dropMskEnable),0,0,0,0,0,0,0,1,0);
    } else if (queryType == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(9,9,9,0,2,0,0,0,0,0,static_cast<uint8_t>(fBaseParams.dropMskEnable),0,0,0,0,0,0,0,1,0);
    }

    OP_LOGI(context_, "FAGTiling Basic Det DoTiling success, tilingkey is %lu.", tilingKey);
    return tilingKey;
}

ge::graphStatus FlashAttentionScoreGraTilingBasicDet::GetWorkspaceSize()
{
    auto compileInfoPtr = context_->GetCompileInfo<FlashAttentionGradBasicDetCompileInfo>();
    uint32_t cubeCoreNum = compileInfoPtr->aicNum;
    constexpr size_t WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
    constexpr size_t GM_ALIGN = 512;
    constexpr size_t DB_NUM = 2;
    constexpr size_t matmulSize = 16 * 128 * 128;

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    size_t workspaceOffset = 0;

    // matmal3 q
    tilingData->basicDetTensorTilingData.set_dqWorkSpaceOffset(workspaceOffset);
    workspaceOffset = tilingData->basicDetTensorTilingData.get_dqPostAbsorb() 
                      ? workspaceOffset + GM_ALIGN * tilingData->basicDetTensorTilingData.get_d() * sizeof(float)
                      : (workspaceOffset + tilingData->basicDetTensorTilingData.get_qSize() * sizeof(float) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // matmal3 k
    tilingData->basicDetTensorTilingData.set_dkWorkSpaceOffset(workspaceOffset);
    workspaceOffset = (workspaceOffset + tilingData->basicDetTensorTilingData.get_kvSize() * sizeof(float) + GM_ALIGN) /
                      GM_ALIGN * GM_ALIGN;
    // matmal3 v
    tilingData->basicDetTensorTilingData.set_dvWorkSpaceOffset(workspaceOffset);
    workspaceOffset = (workspaceOffset + tilingData->basicDetTensorTilingData.get_kvSize() * sizeof(float) + GM_ALIGN) /
                      GM_ALIGN * GM_ALIGN;
    // sfmg workspace
    tilingData->basicDetTensorTilingData.set_sfmgWorkspaceOffset(workspaceOffset);
    workspaceOffset = (workspaceOffset + tilingData->basicDetTensorTilingData.get_sfmgSize() * sizeof(float) + GM_ALIGN) /
                      GM_ALIGN * GM_ALIGN;

    // matmal1/matmal2 workspace size
    tilingData->basicDetTensorTilingData.set_mm1WorkspaceOffset(workspaceOffset);
    workspaceOffset =
        (workspaceOffset + cubeCoreNum * matmulSize * sizeof(float) * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    tilingData->basicDetTensorTilingData.set_mm2WorkspaceOffset(workspaceOffset);
    workspaceOffset =
        (workspaceOffset + cubeCoreNum * matmulSize * sizeof(float) * DB_NUM + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    if (context_->GetDeterministic() == 1) {
        tilingData->basicDetTensorTilingData.set_dqDetWorkspaceOffset(workspaceOffset);
        workspaceOffset = workspaceOffset + cubeCoreNum * GM_ALIGN * tilingData->basicDetTensorTilingData.get_d() * sizeof(float);
        tilingData->basicDetTensorTilingData.set_dkDetWorkspaceOffset(workspaceOffset);
        workspaceOffset = workspaceOffset + cubeCoreNum * GM_ALIGN * tilingData->basicDetTensorTilingData.get_d() * sizeof(float);
        tilingData->basicDetTensorTilingData.set_dvDetWorkspaceOffset(workspaceOffset);
        workspaceOffset = workspaceOffset + cubeCoreNum * GM_ALIGN * tilingData->basicDetTensorTilingData.get_d() * sizeof(float);
  }

    workspaceOffset += WORKSPACE_RSV_BYTE;
    // set workspace
    workspaces[0] = (workspaceOffset - 0);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGraTilingBasicDet::PostTiling()
{
    auto compileInfoPtr = context_->GetCompileInfo<FlashAttentionGradBasicDetCompileInfo>();

    // setBlockDim
    uint32_t cubeCoreNum = compileInfoPtr->aicNum;
    context_->SetBlockDim(cubeCoreNum);
    context_->SetScheduleMode(1);

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus FlashAttentionScoreGraTilingBasicDet::GetPlatformInfo()
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


REGISTER_TILING_TEMPLATE_WITH_ARCH(
    FlashAttentionScoreGrad, FlashAttentionScoreGraTilingBasicDet,
    std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}),
    1002);
} // namespace optiling