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
 * \file flash_attention_score_grad_tiling_s1s2_bn2.cc
 * \brief
 */

#include "flash_attention_score_grad_tiling_s1s2_bn2.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../../op_kernel/arch32/flash_attention_score_grad_tiling.h"
#include "../../op_kernel/arch32/flash_attention_score_grad_template_tiling_key.h"

namespace optiling {

constexpr uint32_t DIMS_FOUR = 4;
constexpr uint32_t DIMS_THREE = 3;
constexpr uint32_t HALF_PRECISION_SIZE = 2;
constexpr uint32_t FLOAT_PRECISION_SIZE = 4;
constexpr uint32_t MASK_FLOAT_VALUE = 64;
constexpr uint32_t MASK_HALF_VALUE = 128;
constexpr uint32_t WORKSPACE_BASE_CAL = 32 * 1024 * 1024; // 100MB系统预留
constexpr uint32_t BLOCK = 32;                            // 32B
constexpr uint32_t B32 = 4;                               // 4B
constexpr uint32_t B16 = 2;
constexpr uint32_t HIGH_PRECISION_0 = 0;
constexpr uint32_t HIGH_PERFORMANCE_1 = 1;
constexpr uint32_t BASE_LEN_128 = 128;
constexpr uint32_t BASE_LEN_256 = 256;
constexpr uint32_t BASE_LEN_512 = 512;
constexpr uint32_t BASE_LEN_64 = 64;
constexpr uint32_t BASE_LEN_32 = 32;
constexpr uint32_t NUMBER_2 = 2;
constexpr uint32_t BIT_NUM = 8;
constexpr int64_t GM_ALIGN = 512;
constexpr uint32_t MAX_STRIDE_LIMIT = 65535;
constexpr uint32_t MM_RATIO_4 = 4;
constexpr uint32_t MM_RATIO_8 = 8;
constexpr uint32_t PSE_ALIBI_S1_SIZE = 1024;
constexpr uint32_t PSE_ALIBI_S2_LIMIT_SIZE = 1024;
constexpr uint32_t PSE_DIM_NUM_1 = 1;
constexpr uint32_t PSE_DIM_NUM_2 = 2;
constexpr uint32_t MATMUL_MINI_N = 16;
constexpr uint32_t VEC_REPEAT_LIMIT = 255;
constexpr uint32_t BAND_MODE = 3;
constexpr uint32_t POST_COEX_NODE = 3;
constexpr uint32_t BUFFER_NUM = 1;
constexpr int64_t S_LEN_512 = 512;
constexpr int64_t S_LEN_1024 = 1024;
constexpr int64_t FP16_BLOCK_NUMS = 16;
constexpr uint64_t B4_L2_CACHESIZE = 96 * 1024 * 1024;
constexpr int64_t SAMEAB_MIN_S_THRESHOLD = 768;
constexpr int64_t L1_MAX_SIZE = 512 * 1024;
constexpr int64_t L1_CUSTOM_DS_SIZE = 128 * 128 * 2 * 2;
constexpr int64_t L1_CUSTOM_D_72 = 72;
constexpr int64_t L1_CUSTOM_D_88 = 88;
constexpr int64_t L1_CUSTOM_D_128 = 128;
constexpr int64_t L1_CUSTOM_S_256 = 256;
constexpr int64_t L1_CUSTOM_S_500 = 500;
constexpr int64_t L1_CUSTOM_D128_S2MAX = 640;
constexpr int64_t L1_CUSTOM_NUM_4 = 4;
constexpr int64_t L1_CUSTOM_NUM_8 = 8;
constexpr int64_t B_1 = 1;
constexpr int64_t B_6 = 6;
constexpr int64_t G_4 = 4;
constexpr int64_t SPARSE_MODE_3 = 3;

const char *templateNameBn2 = "FlashAttentionScoreGradTilingS1s2Bn2";
const std::vector<gert::Shape> MODULE1_TEMPLATE_SUPPORTED_SHAPE = {
    // 5d shape [B, N,S1,S2,D]
    {24, 5, 9216, 9216, 64},
    {24, 5, 9216, 80, 64},
    {24, 10, 2304, 2304, 64},
};

enum KernelBranch {
    BRANCH_FP16_HIGH_PRECISION = 10001,
    BRANCH_FP16_HIGH_PERFORMANCE,
    BRANCH_BF16,
    BRANCH_FP32
};

bool FlashAttentionScoreGradTilingS1s2Bn2::IsCapable()
{
    auto sinkShape = context_->GetOptionalInputShape(SINK_IN);
    if (sinkShape != nullptr && sinkShape->GetStorageShape().GetDimNum() == 1 ) {
        return false;
    }
    const char *tndSoftmaxIn = context_->GetAttrs()->GetAttrNum() > static_cast<size_t>(TND_SOFTMAX_IN) ? context_->GetAttrs()->GetAttrPointer<char>(TND_SOFTMAX_IN) : "";
    if (strcmp(tndSoftmaxIn, "") != 0) return false;

    if (context_->GetDeterministic() == 1) {
        return false;
    }

    if (tmpData_.layout == static_cast<uint32_t>(InputLayout::TND)) {
        OP_LOGI(context_, "bn2 support TND layout when is deterministic.");
        return false;
    }

    if (td_->opInfo.get_S1() >= 1024 || td_->opInfo.get_S2() >= 1024) {
        OP_LOGI(context_, "maxS1 or maxS2 is less than 1024, FlashAttentionScoreGradTilingS1s2Bn2 not support.");
        return false;
    }

    if (context_->GetOptionalInputShape(QUERY_ROPE) != nullptr) {
        return false;
    }

    // Key Value D不等长情况不能处理
    OP_CHECK_IF(((context_->GetInputShape(VALUE) == nullptr) || (context_->GetInputShape(KEY) == nullptr)),
               OP_LOGE(context_, "InputShape of value or key is nullptr."),
               return ge::GRAPH_FAILED);
    const gert::StorageShape *valueShape = context_->GetInputShape(VALUE);
    const gert::StorageShape *keyShape = context_->GetInputShape(KEY);
    if (!IsSameShape(keyShape, valueShape) && IsSameShapeButValueDLeEqD(keyShape, valueShape)) {
        return false;
    }

    // G>1且BN2不能开满50%核的时候，需要借G轴开核
    if (td_->opInfo.get_G() > 1 &&
        td_->opInfo.get_B() * td_->opInfo.get_N2() * 2 <= static_cast<int64_t>(aicoreParams_.numBlocks)) {
        OP_LOGI(context_, "G is larger than 1 and BN2 is less than coreNum, S1s2Bn2 cannot enable full core.");
        return false;
    }

    // S1或S2大于768且BN2不能开满核的时候，需要借S1或S2轴开核
    if (td_->opInfo.get_B() * td_->opInfo.get_N2() < static_cast<int64_t>(aicoreParams_.numBlocks) &&
        (td_->opInfo.get_S1() > SAMEAB_MIN_S_THRESHOLD || td_->opInfo.get_S2() > SAMEAB_MIN_S_THRESHOLD)) {
        OP_LOGI(context_, "BN2 is less than coreNum, S1s2Bn2 cannot enable full core.");
        return false;
    }

    float keep_prob = td_->opInfo.get_keepProb();
    float epsilon = 1e-10f;
    auto pseShape = context_->GetOptionalInputShape(PSE_SHIFT);
    if ((td_->opInfo.get_D() == L1_CUSTOM_D_128) && (td_->opInfo.get_B() >= B_1) && (td_->opInfo.get_B() <= B_6) &&
        (td_->opInfo.get_G() >= G_4) && (tmpData_.t1 == tmpData_.t2) &&
        (td_->opInfo.get_sparseMode() == SPARSE_MODE_3 || td_->opInfo.get_sparseMode() == 0) &&
        (1.0f - keep_prob < epsilon) && (pseShape == nullptr)) {
        return false;
    }

    return true;
}

bool FlashAttentionScoreGradTilingDeterministic::IsCapable()
{
    auto sinkShape = context_->GetOptionalInputShape(SINK_IN);
    if (sinkShape != nullptr && sinkShape->GetStorageShape().GetDimNum() == 1 ) {
        return false;
    }
    const char *tndSoftmaxIn = context_->GetAttrs()->GetAttrNum() > static_cast<size_t>(TND_SOFTMAX_IN) ? context_->GetAttrs()->GetAttrPointer<char>(TND_SOFTMAX_IN) : "";
    if (strcmp(tndSoftmaxIn, "") != 0) return false;

    OP_LOGD(context_, "Get deterministic flag is %d", context_->GetDeterministic());
    if (context_->GetDeterministic() != 1) {
        return false;
    }

    if (context_->GetOptionalInputShape(QUERY_ROPE) != nullptr) {
        return false;
    }

    OP_CHECK_IF(((context_->GetInputShape(VALUE) == nullptr) || (context_->GetInputShape(KEY) == nullptr)),
               OP_LOGE(context_, "InputShape of value or key is nullptr."),
               return ge::GRAPH_FAILED);
    const gert::StorageShape *valueShape = context_->GetInputShape(VALUE);
    const gert::StorageShape *keyShape = context_->GetInputShape(KEY);
    if (!IsSameShape(keyShape, valueShape) && IsSameShapeButValueDLeEqD(keyShape, valueShape)) {
        return false;
    }

    // 支持fp32 确定性计算，走该模板
    OP_CHECK_IF(context_->GetInputDesc(QUERY) == nullptr,
               OP_LOGE(context_, "InputDesc of query is nullptr."), return false);
    if (context_->GetInputDesc(QUERY)->GetDataType() == ge::DT_FLOAT) {
        return true;
    }

    if (td_->opInfo.get_S1() >= 1024 || td_->opInfo.get_S2() >= 1024) {
        OP_LOGI(context_, "maxS1 or maxS2 is less than 1024, FlashAttentionScoreGradTilingS1s2Bn2 not support.");
        return false;
    }

    return true;
}

uint64_t FlashAttentionScoreGradTilingS1s2Bn2::GetTilingKey() const
{
    bool isBsh = tmpData_.layout == static_cast<uint32_t>(InputLayout::BSH) ||
                 tmpData_.layout == static_cast<uint32_t>(InputLayout::BSND);
    bool isSbh = tmpData_.layout == static_cast<uint32_t>(InputLayout::SBH);
    bool isBnsd = tmpData_.layout == static_cast<uint32_t>(InputLayout::BNSD);
    bool isTnd = tmpData_.layout == static_cast<uint32_t>(InputLayout::TND);

    uint64_t tilingKey = 0;
    CubeFormatEnum mmOutFormat = CubeFormatEnum::ND;
    MatmulConfig mmConfig = MatmulConfig::NULL_CONFIG;
    DtypeEnum inDtype = DtypeEnum::FLOAT16;
    LayoutEnum layout = LayoutEnum::BNSD;
    // current only BNSD
    if ((tmpData_.queryType == ge::DT_BF16)) {
        inDtype = DtypeEnum::BFLOAT16;
    }
    if ((tmpData_.queryType == ge::DT_FLOAT)) {
        inDtype = DtypeEnum::FLOAT32;
    }
    if (tmpData_.queryType != ge::DT_FLOAT) {
        // 非cacheline对齐场景使能NZ优化
        mmOutFormat = CubeFormatEnum::NZ;
    }
    if (isBsh) {
        layout = LayoutEnum::BSND;
    } else if (isSbh) {
        if (tmpData_.b * tmpData_.n2 * tmpData_.g * tmpData_.d > MAX_STRIDE_LIMIT) {
            // SBH: B*H > 65535
            mmConfig = MatmulConfig::NORMAL_CONFIG;
        }
        layout = LayoutEnum::SBND;
    } else if (isBnsd) {
        layout = LayoutEnum::BNSD;
    } else if (isTnd) {
        layout = LayoutEnum::TND;
        mmOutFormat = CubeFormatEnum::ND;
    }

    auto mm345NZOut = tmpData_.isMM345NZOut ? OptionEnum::ENABLE : OptionEnum::DISABLE;
    auto l1Custom = tmpData_.isL1CustomEnable ? OptionEnum::ENABLE : OptionEnum::DISABLE;
    tilingKey =
        GET_TPL_TILING_KEY(static_cast<uint8_t>(AxisEnum::S2), static_cast<uint8_t>(AxisEnum::S1), static_cast<uint8_t>(AxisEnum::N2), 0, static_cast<uint8_t>(inDtype),
            static_cast<uint8_t>(layout), static_cast<uint8_t>(SparseEnum::ALL), static_cast<uint8_t>(mmConfig), static_cast<uint8_t>(mmOutFormat),
            static_cast<uint8_t>(mm345NZOut), static_cast<uint8_t>(tmpData_.drop_out_cfg), static_cast<uint8_t>(tmpData_.pse_cfg),
            static_cast<uint8_t>(tmpData_.atten_mask_cfg), static_cast<uint8_t>(l1Custom), 0, 0, 0, 0, 0, 0);
    OP_LOGI(context_, "FAGTiling S1s2Bn2 DoTiling success, tilingkey is %lu.", tilingKey);
    return tilingKey;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::GetPlatformInfo()
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
        l2CacheSize =
            compileInfoPtr->l2CacheSize; // AiCoreParams使用的是cann仓的结构体，l2CacheSize暂时定义成类成员变量
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        aicoreParams_.aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, aicoreParams_.l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, aicoreParams_.l0bSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    }

    OP_CHECK_IF((aicoreParams_.numBlocks == 0) || (aicoreParams_.aicNum == 0),
               OP_LOGE(context_, "num of coreNum(aivNum) is %lu, num of aicNum is %lu.",
                                           aicoreParams_.numBlocks, aicoreParams_.aicNum),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(aicoreParams_.ubSize <= 0 || l2CacheSize <= 0,
               OP_LOGE(context_, "ubSize or l2CacheSize is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingS1s2Bn2::DecideBranch()
{
    // according calcMode decide which params and branch
    tmpData_.branch = BRANCH_FP32;
    tmpData_.dataTypeSize = FLOAT_PRECISION_SIZE;
    tmpData_.mask = MASK_FLOAT_VALUE;
    tmpData_.queryType = static_cast<uint32_t>(context_->GetInputDesc(QUERY)->GetDataType());

    tmpData_.calcMode = HIGH_PRECISION_0;
    OP_LOGD(context_, "calcMode is %u", tmpData_.calcMode);

    // assgin params
    if (tmpData_.queryType == ge::DT_FLOAT16) {
        if (tmpData_.calcMode == HIGH_PERFORMANCE_1) {
            tmpData_.branch = BRANCH_FP16_HIGH_PERFORMANCE;
            tmpData_.dataTypeSize = HALF_PRECISION_SIZE;
            tmpData_.mask = MASK_HALF_VALUE;
        } else {
            tmpData_.branch = BRANCH_FP16_HIGH_PRECISION;
        }
    } else if (tmpData_.queryType == ge::DT_BF16) {
        tmpData_.branch = BRANCH_BF16;
    }
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::SetMaskShapeType(const gert::Shape &storageShape,
                                                                       const uint32_t maskShapeDims)
{
    if (maskShapeDims == DIM_2) {
        td_->opInfo.set_maskShapeType(ATTEN_MASK_SHAPE_TYPE_SS);
    } else if (maskShapeDims == DIM_4) {
        auto dim0 = storageShape.GetDim(DIM_0);
        auto dim1 = storageShape.GetDim(DIM_1);
        if ((dim0 == td_->opInfo.get_B()) && (dim1 == td_->opInfo.get_N2() * td_->opInfo.get_G())) {
            td_->opInfo.set_maskShapeType(ATTEN_MASK_SHAPE_TYPE_BNSS);
        } else if ((dim0 == td_->opInfo.get_B()) && (dim1 == 1)) {
            td_->opInfo.set_maskShapeType(ATTEN_MASK_SHAPE_TYPE_B1SS);
        } else if ((dim0 == 1) && (dim1 == 1)) {
            td_->opInfo.set_maskShapeType(ATTEN_MASK_SHAPE_TYPE_SS);
        } else {
            OP_LOGE(context_, "FlashAttentionScoreGradTilingS1s2Bn2 not support attenMaskShape dim0: %ld, dim1: %ld",
                      dim0, dim1);
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE(context_, "FlashAttentionScoreGradTilingS1s2Bn2 not support attenMaskShape dim num: %u",
                  maskShapeDims);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::GetAttenMaskInfo()
{
    /*
    Get attenmask input info
    */
    td_->opInfo.set_existAttenMask(0);
    attenMaskCompressMode = NO_COMPRESS_MODE;
    auto attenMaskShape = context_->GetOptionalInputShape(ATTEN_MASK);
    if (attenMaskShape != nullptr && attenMaskShape->GetStorageShape().GetDimNum() != 0) {
        // attenMask support [S1,S2](0) + [B,1,S1,S2](1) + [B,N1,S1,S2](2)
        td_->opInfo.set_existAttenMask(1);
        tmpData_.atten_mask_cfg = AttenMaskConfig::EXIST_ATTEN_MASK;
        auto storageShape = attenMaskShape->GetStorageShape();
        auto maskShapeDims = storageShape.GetDimNum();
        if (SetMaskShapeType(storageShape, maskShapeDims) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        attenMaskS2Size = storageShape.GetDim(maskShapeDims - LAST_AXIS_IDX);
        attenMaskS1Size = storageShape.GetDim(maskShapeDims - SEC_LAST_AXIS_IDX);

        if (sparseMode == LEFT_UP_CAUSAL) {
            attenMaskCompressMode = LEFT_UP_CAUSAL_MODE;
        } else if (sparseMode == RIGHT_DOWN_CAUSAL) {
            attenMaskCompressMode = RIGHT_DOWN_CAUSAL_MODE;
        } else if (sparseMode == BAND) {
            attenMaskCompressMode = BAND_MODE;
        } else if (sparseMode == PREFIX_COMPRESS) {
            attenMaskCompressMode = PREFIX_COMPRESS_MODE;
        } else if (sparseMode == RIGHT_DOWN_CASUAL_BAND) {
            attenMaskCompressMode = RIGHT_DOWN_CASUAL_BAND_MODE;
        } else if (sparseMode == BAND_LEFT_UP_CASUAL) {
            attenMaskCompressMode = BAND_LEFT_UP_CASUAL_MODE;
        }
        auto attenMaskDesc = context_->GetOptionalInputDesc(ATTEN_MASK);
        if (attenMaskDesc != nullptr) {
            auto attenMaskDtype = attenMaskDesc->GetDataType();
            if (attenMaskDtype == tmpData_.queryType) {
                td_->opInfo.set_maskDataType(ATTEN_MASK_TYPE_SAME);
            } else if (attenMaskDtype == ge::DT_BOOL || attenMaskDtype == ge::DT_UINT8) {
                td_->opInfo.set_maskDataType(ATTEN_MASK_TYPE_U8_BOOL);
            } else {
                OP_LOGE(context_, "FlashAttentionScoreGradTilingS1s2Bn2 not support attenMaskDtype: %d",
                          attenMaskDtype);
                return ge::GRAPH_FAILED;
            }
        }
    }
    td_->opInfo.set_attenMaskS2Size(attenMaskS2Size);
    td_->opInfo.set_attenMaskCompressMode(attenMaskCompressMode);
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingS1s2Bn2::SetQKVStartIdx()
{
    td_->opInfo.set_qStartIdx(0);
    td_->opInfo.set_kvStartIdx(0);
    auto qStartIdxTensor = context_->GetOptionalInputTensor(Q_START_IDX);
    if (qStartIdxTensor == nullptr) {
        OP_LOGW(context_, "[%s]qStartIdxTensor is null pointer", templateNameBn2);
        return;
    }
    auto &qStartIdxShape = qStartIdxTensor->GetShape().GetStorageShape();
    if (qStartIdxShape.GetDimNum() != 1 || qStartIdxShape.GetDim(0) <= 0) {
        OP_LOGW(context_, "[%s]qStartIdxShape is invalid %lu %ld", templateNameBn2, qStartIdxShape.GetDimNum(),
                  qStartIdxShape.GetDim(0));
        return;
    }
    /* Get Data from tensor. */
    const int64_t *value = qStartIdxTensor->GetData<int64_t>();
    if (value == nullptr) {
        OP_LOGW(context_, "[%s]qStartIdxShape data is null pointer", templateNameBn2);
        return;
    }
    tmpData_.qStartIdx = value[0];

    auto kvStartIdxTensor = context_->GetOptionalInputTensor(KV_START_IDX);
    if (kvStartIdxTensor == nullptr) {
        OP_LOGW(context_, "[%s]kvStartIdxTensor is null pointer", templateNameBn2);
        return;
    }
    auto &kvStartIdxShape = kvStartIdxTensor->GetShape().GetStorageShape();
    if (kvStartIdxShape.GetDimNum() != 1 || kvStartIdxShape.GetDim(0) <= 0) {
        OP_LOGW(context_, "[%s]kvStartIdxShape is invalid %lu %ld", templateNameBn2, kvStartIdxShape.GetDimNum(),
                  kvStartIdxShape.GetDim(0));
        return;
    }
    /* Get Data from tensor. */
    const int64_t *kvValue = kvStartIdxTensor->GetData<int64_t>();
    if (kvValue == nullptr) {
        OP_LOGW(context_, "[%s]qStartIdxShape data is null pointer", templateNameBn2);
        return;
    }
    tmpData_.kvStartIdx = kvValue[0];

    td_->opInfo.set_qStartIdx(tmpData_.qStartIdx);
    td_->opInfo.set_kvStartIdx(tmpData_.kvStartIdx);
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::ProcessNormalPse()
{
    auto pseShape = context_->GetOptionalInputShape(PSE_SHIFT);
    auto storageShape = pseShape->GetStorageShape();
    auto pseShapeDims = storageShape.GetDimNum();
    if (pseShapeDims != DIM_4) {
        OP_LOGE(context_, "FlashAttentionScoreGradTilingS1s2Bn2 not support pseShape dim num: %zu", pseShapeDims);
        return ge::GRAPH_FAILED;
    }
    auto dim0 = storageShape.GetDim(DIM_0);
    auto dim1 = storageShape.GetDim(DIM_1);
    auto dim2 = storageShape.GetDim(DIM_2);
    auto dim3 = storageShape.GetDim(DIM_3);
    // [B N S1 S2](0)  [B N 1 S2](1)  [1 N S1 S2](2)shape判断
    int64_t shapeN1 = td_->opInfo.get_N2() * td_->opInfo.get_G();
    bool isBNS = (dim0 == td_->opInfo.get_B()) && (dim1 == shapeN1) && (dim3 == td_->opInfo.get_S2());
    bool isBNSS = isBNS && (dim2 == td_->opInfo.get_S1());
    bool isBN1S = isBNS && (dim2 == 1);
    bool is1NSS = (dim0 == 1) && (dim1 == shapeN1) && (dim2 == td_->opInfo.get_S1()) && (dim3 == td_->opInfo.get_S2());
    // [B N H S2](4)  [1 N H S2](5) shape判断
    bool isNHS = (dim1 == shapeN1) && (dim2 == PSE_ALIBI_S1_SIZE) && (dim3 == td_->opInfo.get_S2());
    // alibi条件1: S1 = S2 且 S1 >= 1024
    // alibi条件2: 下三角
    bool alibiCondi1 = (td_->opInfo.get_S1() == td_->opInfo.get_S2()) && (td_->opInfo.get_S1() >= PSE_ALIBI_S1_SIZE);
    bool alibiCondi2 =
        (td_->opInfo.get_preTokens() >= int32_t(td_->opInfo.get_S1())) && (td_->opInfo.get_nextTokens() == 0);
    bool isAlibiBNHS = isNHS && (dim0 == td_->opInfo.get_B()) && alibiCondi1 && alibiCondi2;
    bool isAlibi1NHS = isNHS && (dim0 == 1) && alibiCondi1 && alibiCondi2;
    bool isUnpad = td_->opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND);
    // 设置shape类型
    if (isUnpad && isNHS && (dim0 == 1) && alibiCondi2) {
        td_->opInfo.set_pseShapeType(PSE_SHAPE_TYPE_1NHS);
    } else if (isUnpad && isNHS && (dim0 == td_->opInfo.get_B()) && alibiCondi2) {
        td_->opInfo.set_pseShapeType(PSE_SHAPE_TYPE_BNHS);
    } else if (is1NSS && !isUnpad) {
        td_->opInfo.set_pseShapeType(PSE_SHAPE_TYPE_1NSS);
    } else if (isBN1S && !isUnpad) {
        td_->opInfo.set_pseShapeType(PSE_SHAPE_TYPE_BN1S);
    } else if (isBNSS && !isUnpad) {
        td_->opInfo.set_pseShapeType(PSE_SHAPE_TYPE_BNSS);
    } else if (isAlibi1NHS) {
        td_->opInfo.set_pseShapeType(PSE_SHAPE_TYPE_1NHS);
    } else if (isAlibiBNHS) {
        td_->opInfo.set_pseShapeType(PSE_SHAPE_TYPE_BNHS);
    } else {
        OP_LOGE(context_, "The shape of pse[%ld,%ld,%ld,%ld] is invalid or tocken[%ld,%ld] not casual", dim0, dim1,
                  dim2, dim3, td_->opInfo.get_preTokens(), td_->opInfo.get_nextTokens());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::GetPseInfo()
{
    /*
    Get pse input info
    */
    auto pseShape = context_->GetOptionalInputShape(PSE_SHIFT);
    auto pse = context_->GetOptionalInputDesc(PSE_SHIFT);

    if (pseShape == nullptr || pseShape->GetStorageShape().GetDimNum() == 0 ) {
        if(tmpData_.pseType != PSE_OUTER_ADD_MUL_TYPE){
            /*
            * 1. pseType非默认值
            * 2. 未传入PSE
            * FA正向对这种情况进行了兼容，能够得到正确的计算结果。
            * FA反向未兼容，因此统一拦截异常输入。
            */
            OP_LOGE(context_, "Get PseInput is nullptr, but pseType is not default=%u, now pseType=%ld.", PSE_OUTER_ADD_MUL_TYPE, tmpData_.pseType);
            return ge::GRAPH_FAILED;
        }
    }

    if (pse != nullptr && pseShape != nullptr && pseShape->GetStorageShape().GetDimNum() != 0) {
        // pse support [B N S1 S2](0) + [B N 1 S2](1) + [1 N S1 S2](2)
        // pse support alibi: [B N H S2](3) + [1 N H S2](4)
        tmpData_.pse_cfg = PseConfig::EXIST_PSE;
        auto storageShape = pseShape->GetStorageShape();
        auto pseShapeDims = storageShape.GetDimNum();
        bool isUnpad = td_->opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND);
        if (tmpData_.pseType == PSE_OUTER_MUL_ADD_TYPE || tmpData_.pseType == PSE_OUTER_ADD_MUL_TYPE) {
            OP_CHECK_IF(pse->GetDataType() != context_->GetInputDesc(QUERY)->GetDataType(),
                       OP_LOGE(context_,
                                                   "FAG invalid pse dtype[%s], should be same with query's dtype",
                                                   ge::TypeUtils::DataTypeToSerialString(pse->GetDataType()).c_str()),
                       return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(pse->GetDataType() != ge::DT_FLOAT,
                       OP_LOGE(context_, "FAG invalid pse dtype[%s], should be ge::DT_FLOAT",
                                                   ge::TypeUtils::DataTypeToSerialString(pse->GetDataType()).c_str()),
                       return ge::GRAPH_FAILED);
        }
        if (tmpData_.pseType == PSE_INNER_MUL_ADD_TYPE || tmpData_.pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
            if (pseShapeDims == PSE_DIM_NUM_1) {
                auto dim0 = pseShape->GetStorageShape().GetDim(DIM_0);
                OP_CHECK_IF(dim0 != tmpData_.n2 * tmpData_.g,
                           OP_LOGE(context_,
                                                       "FAG invalid pse shape %ld, should be same with n1 %ld", dim0,
                                                       tmpData_.n2 * tmpData_.g),
                           return ge::GRAPH_FAILED);
                td_->opInfo.set_pseShapeType(PSE_1_N2_G_SLOPE);
            } else if (pseShapeDims == PSE_DIM_NUM_2) {
                auto dim0 = pseShape->GetStorageShape().GetDim(DIM_0);
                auto dim1 = pseShape->GetStorageShape().GetDim(DIM_1);
                OP_CHECK_IF(dim0 != tmpData_.b || dim1 != tmpData_.n2 * tmpData_.g,
                           OP_LOGE(
                               context_, "FAG invalid pse shape {%ld, %ld} should be same with b n1 {%ld, %ld}", dim0,
                               dim1, tmpData_.b, tmpData_.n2 * tmpData_.g),
                           return ge::GRAPH_FAILED);
                td_->opInfo.set_pseShapeType(PSE_B_N2_G_SLOPE);
            } else {
                OP_LOGE(context_, "pse inner mode, unsupported pse shape");
                return ge::GRAPH_FAILED;
            }
        } else if (pseShapeDims == PSE_DIM_NUM_1 && isUnpad) {
            auto dim0 = storageShape.GetDim(DIM_0);
            int64_t shapeN1 = td_->opInfo.get_N2() * td_->opInfo.get_G();
            bool isUnpadBN1S = isUnpad && (dim0 == tmpData_.t2 * shapeN1);
            bool isUnpadBNSS = isUnpad && (dim0 == tmpData_.sumS1S2Product * shapeN1);
            if (isUnpadBN1S) {
                td_->opInfo.set_pseShapeType(PSE_SHAPE_TYPE_BN1S);
            } else if (isUnpadBNSS) {
                td_->opInfo.set_pseShapeType(PSE_SHAPE_TYPE_BNSS);
            } else {
                OP_LOGE(context_, "pse outer mode, tnd, unsupported pse shape");
                return ge::GRAPH_FAILED;
            }
        } else {
            return ProcessNormalPse();
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreGradTilingS1s2Bn2::SparseTokenProcess()
{
    int32_t preTokens = td_->opInfo.get_preTokens();
    int32_t nextTokens = td_->opInfo.get_nextTokens();
    OP_LOGD(context_, "preTokens: %d, nextTokens: %d, sparseMode: %d.", preTokens, nextTokens, sparseMode);

    if (sparseMode == NO_MASK) {
        if (int32_t(td_->opInfo.get_S1()) <= preTokens && nextTokens == 0) { // 原causal 场景
            td_->opInfo.set_isSparse(1);
        } else if (int32_t(td_->opInfo.get_S1()) > preTokens ||
                   int32_t(td_->opInfo.get_S2()) > nextTokens) { // 原band场景
            td_->opInfo.set_isSparse(1);
        } else {
            OP_LOGI(context_,
                      "FAG S1s2Bn2 sparseMode[%d] with s1[%ld] s2[%ld] preToken[%d] nextToken[%d] is not support!",
                      sparseMode, td_->opInfo.get_S1(), td_->opInfo.get_S2(), preTokens, nextTokens);
        }
        return false;
    }

    if (sparseMode != LEFT_UP_CAUSAL && sparseMode != RIGHT_DOWN_CAUSAL && sparseMode != BAND && sparseMode != PREFIX &&
        sparseMode != PREFIX_COMPRESS && sparseMode != RIGHT_DOWN_CASUAL_BAND && sparseMode != BAND_LEFT_UP_CASUAL) {
        OP_LOGW(context_, "FAG S1s2Bn2 sparseMode[%d] is not support!", sparseMode);
        return false;
    }

    if (sparseMode == BAND) {
        // BAND模式tokens以右下角为基准
        if ((int32_t(td_->opInfo.get_S2()) <= preTokens && int32_t(td_->opInfo.get_S1()) <= nextTokens) ||
            (preTokens + nextTokens) < 0) {
            OP_LOGW(context_,
                      "FAG S1s2Bn2 sparseMode is band, but preToken[%d] or nextToken[%d] is invalid "
                      "with s1[%ld] s2[%ld]!",
                      preTokens, nextTokens, td_->opInfo.get_S1(), td_->opInfo.get_S2());
            return false;
        }
    }
    return true;
}

bool FlashAttentionScoreGradTilingS1s2Bn2::ProcessPrefix()
{
    if (sparseMode != PREFIX && sparseMode != PREFIX_COMPRESS) {
        return true;
    }

    auto prefixNTensor = context_->GetOptionalInputTensor(PREFIX_N);
    if (prefixNTensor == nullptr) {
        OP_LOGW(context_, "FAG S1s2Bn2 sparseMode is prefix, but prefixN tensor is null!");
        return false;
    }

    auto &prefixShape = prefixNTensor->GetShape().GetStorageShape();
    if (prefixShape.GetDimNum() != 1 || prefixShape.GetDim(0) != td_->opInfo.get_B()) {
        OP_LOGW(context_, "FAG S1s2Bn2 sparseMode is prefix, but prefixshape size[%zu] or value is invalid!",
                  prefixShape.GetDimNum());
        return false;
    }

    const int64_t *value = prefixNTensor->GetData<int64_t>();
    if (value == nullptr) {
        OP_LOGW(context_, "FAG S1s2Bn2 sparseMode is prefix, but prefixN data is null pointer!");
        return false;
    }
    auto shapeSize = prefixNTensor->GetShapeSize();
    for (int64_t i = 0; i < shapeSize; i++) {
        tmpData_.prefixN.push_back(value[i]);
    }

    if (tmpData_.prefixN.size() != static_cast<uint64_t>(td_->opInfo.get_B())) {
        OP_LOGW(context_, "FAG S1s2Bn2 sparseMode is prefix, but prefixN value is invalid!");
        return false;
    }

    return true;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::SetSparseParams()
{
    td_->opInfo.set_isSparse(0);
    td_->opInfo.set_sparseMode(sparseMode);
    auto attenMaskShape = context_->GetOptionalInputShape(ATTEN_MASK);
    auto attenMaskDesc = context_->GetOptionalInputDesc(ATTEN_MASK);
    if (sparseMode == ALL_MASK || attenMaskShape == nullptr || attenMaskDesc == nullptr) {
        OP_LOGI(context_,
                  "FAG S1s2Bn2 sparseMode[%d] is all_mask or attenmask is nullptr,"
                  "not support sparse mode!",
                  sparseMode);
        return ge::GRAPH_SUCCESS;
    }

    if (!SparseTokenProcess()) {
        return ge::GRAPH_SUCCESS;
    }

    auto status = ProcessPrefix();
    if (status == false) {
        if (sparseMode != PREFIX_COMPRESS) {
            return ge::GRAPH_SUCCESS;
        }

        OP_LOGE(context_, "Sparse capability must be supported under prefix compress mode, pls check input params");
        return ge::GRAPH_FAILED;
    }

    td_->opInfo.set_isSparse(1);

    if (sparseMode == LEFT_UP_CAUSAL || sparseMode == RIGHT_DOWN_CAUSAL) {
        td_->opInfo.set_preTokens(INT32_MAX);
        td_->opInfo.set_nextTokens(0);
    }
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingS1s2Bn2::SetBandIdx()
{
    if (sparseMode == RIGHT_DOWN_CASUAL_BAND) {
        for (int i = tmpData_.b - 1; i >= 0; i--) {
            if (tmpData_.actualSeqQlen[i] != 0) {
                td_->opInfo.set_bandIdx(i);
                return;
            }
        }
    } else if (sparseMode == BAND_LEFT_UP_CASUAL) {
        for (int64_t i = 0; i < tmpData_.b; i++) {
            if (tmpData_.actualSeqQlen[i] != 0) {
                td_->opInfo.set_bandIdx(i);
                return;
            }
        }
    }
    // 添加保护
    td_->opInfo.set_bandIdx(0);
}

bool FlashAttentionScoreGradTilingS1s2Bn2::IsModuleOneShape()
{
    gert::Shape inputShape5d = {td_->opInfo.get_B(), td_->opInfo.get_N2() * td_->opInfo.get_G(), td_->opInfo.get_S1(),
                                td_->opInfo.get_S2(), td_->opInfo.get_D()};
    return !(std::find(MODULE1_TEMPLATE_SUPPORTED_SHAPE.begin(), MODULE1_TEMPLATE_SUPPORTED_SHAPE.end(),
                       inputShape5d) == MODULE1_TEMPLATE_SUPPORTED_SHAPE.end());
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::CheckOutOfTokens(const int64_t s1, const int64_t s2)
{
    int32_t negPretMax = sparseMode == NO_MASK ? s2 : s1;
    int32_t negNexttMax = sparseMode == NO_MASK ? s1 : s2;
    if (-td_->opInfo.get_preTokens() > negPretMax || -td_->opInfo.get_nextTokens() > negNexttMax ||
        (int64_t(td_->opInfo.get_preTokens()) + int64_t(td_->opInfo.get_nextTokens())) < 0) {
        OP_LOGE(context_, "FAG S1s2Bn2 with s1[%ld] s2[%ld] preToken[%ld] nextToken[%ld] is not support!", s1, s2,
                  td_->opInfo.get_preTokens(), td_->opInfo.get_nextTokens());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::CheckTokens()
{
    bool enableTokens = sparseMode == NO_MASK || sparseMode == BAND || sparseMode == RIGHT_DOWN_CASUAL_BAND ||
                        sparseMode == BAND_LEFT_UP_CASUAL;
    if (!enableTokens) {
        return ge::GRAPH_SUCCESS;
    }

    int64_t realS1 = tmpData_.s1;
    int64_t realS2 = tmpData_.s2;
    if (td_->opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        if (sparseMode == NO_MASK || sparseMode == BAND) {
            for (int64_t i = 0; i < tmpData_.b; i++) {
                realS1 = tmpData_.actualSeqQlen[i];
                realS2 = tmpData_.actualSeqKvlen[i];
                auto status = CheckOutOfTokens(realS1, realS2);
                if (status != ge::GRAPH_SUCCESS) {
                    return status;
                }
            }
            return ge::GRAPH_SUCCESS;
        }

        realS1 = tmpData_.actualSeqQlen[td_->opInfo.get_bandIdx()];
        realS2 = tmpData_.actualSeqKvlen[td_->opInfo.get_bandIdx()];
    }
    return CheckOutOfTokens(realS1, realS2);
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::GetShapeAttrsInfo()
{
    /*
    Get all shape info and attr
    */
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(context_, "context is nullptr."),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetAttrs() == nullptr, OP_LOGE(context_, "GetAttrs is nullptr."),
               return ge::GRAPH_FAILED);

    auto status = GetLayoutInfo();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = GetBaseShapeInfo();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    DecideBranch();

    // setTilingData
    td_->opInfo.set_scaleValue(*context_->GetAttrs()->GetAttrPointer<float>(SCALE_VALUE));
    td_->opInfo.set_keepProb(*context_->GetAttrs()->GetAttrPointer<float>(KEEP_PROB));
    OP_CHECK_IF((td_->opInfo.get_keepProb() <= 0 || td_->opInfo.get_keepProb() > 1),
               OP_LOGE(context_, "keepProb is illegal."), return ge::GRAPH_FAILED);

    if (td_->opInfo.get_keepProb() < 1) {
        tmpData_.drop_out_cfg = DropOutConfig::EXIST_DROP_OUT;
        status = ProcessDropInfo();
        if (status != ge::GRAPH_SUCCESS) {
            PrintShapeInfo();
            return status;
        }
    }

    // 新增SPARSE_MODE属性，上库兼容处理
    auto attrs = context_->GetAttrs();
    if (attrs->GetAttrNum() > static_cast<size_t>(SPARSE_MODE)) {
        sparseMode = SparseMode(*(attrs->GetAttrPointer<int>(SPARSE_MODE))); // 7
        uint32_t sparse = *(attrs->GetAttrPointer<int>(SPARSE_MODE));
        if (sparse > BAND_LEFT_UP_CASUAL) {
            OP_LOGE(context_, "FAG sparseMode %u is invalid", sparse);
            return ge::GRAPH_FAILED;
        }
    }

    if ((td_->opInfo.get_layout() != static_cast<uint32_t>(InputLayout::TND)) &&
        (sparseMode == RIGHT_DOWN_CASUAL_BAND || sparseMode == BAND_LEFT_UP_CASUAL)) {
        OP_LOGE(context_, " layout %u not support sparsemode %d", td_->opInfo.get_layout(), sparseMode);
        return ge::GRAPH_FAILED;
    }

    auto pse = context_->GetOptionalInputDesc(PSE_SHIFT);
    if (td_->opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        tnd2bsh = (td_->opInfo.get_B() * td_->opInfo.get_S1() == tmpData_.t1) &&
                  (td_->opInfo.get_B() * td_->opInfo.get_S2() == tmpData_.t2) &&
                  (td_->opInfo.get_S1() == td_->opInfo.get_S2());
        if (context_->GetDeterministic() != 1 && pse == nullptr && tnd2bsh &&
            !(sparseMode == RIGHT_DOWN_CASUAL_BAND || sparseMode == BAND_LEFT_UP_CASUAL)) {
            tmpData_.layout = static_cast<uint32_t>(InputLayout::BSH);
            td_->opInfo.set_layout(static_cast<uint32_t>(InputLayout::BSH));
        }
        OP_LOGI(context_, "bn2 template enable tnd to bsh.");
    }

    td_->opInfo.set_preTokens(*context_->GetAttrs()->GetAttrPointer<int>(PRE_TOKENS));
    td_->opInfo.set_nextTokens(*context_->GetAttrs()->GetAttrPointer<int>(NEXT_TOKENS));
    if (attrs->GetAttrNum() > static_cast<size_t>(PSETYPE)) {
        tmpData_.pseType = *(context_->GetAttrs()->GetAttrPointer<int64_t>(PSETYPE)); // 8
        OP_CHECK_IF((tmpData_.pseType < 0 || tmpData_.pseType >= PSE_INVALID_TYPE),
                   OP_LOGE(context_, "pseType is invalid."), return ge::GRAPH_FAILED);
    }
    td_->opInfo.set_pseType(tmpData_.pseType);

    SetBandIdx();
    status = CheckTokens();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = GetAttenMaskInfo();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = SetSparseParams();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    if ((td_->opInfo.get_isSparse() == 0) && IsModuleOneShape() && context_->GetDeterministic() != 1) {
        OP_LOGD(context_, "Sparse disable. FlashAttentionScoreGradTilingS1s2Bn2 not support this case");
        return ge::GRAPH_PARAM_INVALID;
    }

    OP_LOGD(context_, "FAG S1s2Bn2 sparse %s.", td_->opInfo.get_isSparse() == 0 ? "disable" : "enable");

    status = GetPseInfo();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    tmpData_.isL1CustomEnable =
        (tmpData_.queryType != ge::DT_FLOAT) && (tmpData_.queryType != ge::DT_FLOAT16) &&
        (tmpData_.layout != static_cast<uint32_t>(InputLayout::TND)) && (tmpData_.pse_cfg != PseConfig::EXIST_PSE) &&
        (tmpData_.atten_mask_cfg != AttenMaskConfig::EXIST_ATTEN_MASK) &&
        (tmpData_.drop_out_cfg != DropOutConfig::EXIST_DROP_OUT) && (td_->opInfo.get_D() <= L1_CUSTOM_D_128) &&
        (td_->opInfo.get_D() >= L1_CUSTOM_D_72) && (td_->opInfo.get_isSparse() == 0) &&
        (td_->opInfo.get_sparseMode() == NO_MASK) && (td_->opInfo.get_S1() > L1_CUSTOM_S_256) &&
        (td_->opInfo.get_S2() > L1_CUSTOM_S_256); // sparseMode=0

    int64_t dAlign = AlignData(td_->opInfo.get_D(), FP16_BLOCK_NUMS);
    if (tmpData_.isL1CustomEnable) {
        if (td_->opInfo.get_D() == L1_CUSTOM_D_72) {
            tmpData_.l1CustomSingleN = AlignData(td_->opInfo.get_S2(), BASE_LEN_128);
            tmpData_.l1CustomSingleM =
                (L1_MAX_SIZE - L1_CUSTOM_DS_SIZE - tmpData_.l1CustomSingleN * dAlign * L1_CUSTOM_NUM_4) /
                (L1_CUSTOM_NUM_8 * dAlign);
            tmpData_.l1CustomSingleM = tmpData_.l1CustomSingleM / BASE_LEN_64 * BASE_LEN_64;
            int64_t l1CustomSingleMTmp = AlignData(td_->opInfo.get_S1(), BASE_LEN_64);
            tmpData_.l1CustomSingleM =
                (tmpData_.l1CustomSingleM > l1CustomSingleMTmp) ? l1CustomSingleMTmp : tmpData_.l1CustomSingleM;
        } else if (td_->opInfo.get_D() == L1_CUSTOM_D_88) {
            tmpData_.l1CustomSingleN = AlignData(td_->opInfo.get_S2(), BASE_LEN_128);
            tmpData_.l1CustomSingleM =
                (L1_MAX_SIZE - L1_CUSTOM_DS_SIZE - tmpData_.l1CustomSingleN * dAlign * L1_CUSTOM_NUM_4) /
                (L1_CUSTOM_NUM_8 * dAlign);
            tmpData_.l1CustomSingleM = tmpData_.l1CustomSingleM / BASE_LEN_64 * BASE_LEN_64;
            int64_t l1CustomSingleMTmp = AlignData(td_->opInfo.get_S1(), BASE_LEN_64);
            tmpData_.l1CustomSingleM =
                (tmpData_.l1CustomSingleM > l1CustomSingleMTmp) ? l1CustomSingleMTmp : tmpData_.l1CustomSingleM;
        } else if (td_->opInfo.get_D() == L1_CUSTOM_D_128 && td_->opInfo.get_S1() > L1_CUSTOM_S_500 &&
                   td_->opInfo.get_S2() > L1_CUSTOM_S_500) {
            if (td_->opInfo.get_S2() >= L1_CUSTOM_D128_S2MAX) {
                tmpData_.l1CustomSingleN = L1_CUSTOM_D128_S2MAX;
                tmpData_.l1CustomSingleM =
                    (L1_MAX_SIZE - L1_CUSTOM_DS_SIZE - tmpData_.l1CustomSingleN * dAlign * L1_CUSTOM_NUM_4) /
                    (L1_CUSTOM_NUM_8 * dAlign);
                tmpData_.l1CustomSingleM = tmpData_.l1CustomSingleM / BASE_LEN_128 * BASE_LEN_128;
                int64_t l1CustomSingleMTmp = AlignData(td_->opInfo.get_S1(), BASE_LEN_128);
                tmpData_.l1CustomSingleM =
                    (tmpData_.l1CustomSingleM > l1CustomSingleMTmp) ? l1CustomSingleMTmp : tmpData_.l1CustomSingleM;
            } else {
                tmpData_.l1CustomSingleN = AlignData(td_->opInfo.get_S2(), BASE_LEN_128);
                tmpData_.l1CustomSingleM =
                    (L1_MAX_SIZE - L1_CUSTOM_DS_SIZE - tmpData_.l1CustomSingleN * dAlign * L1_CUSTOM_NUM_4) /
                    (L1_CUSTOM_NUM_8 * dAlign);
                tmpData_.l1CustomSingleM = tmpData_.l1CustomSingleM / BASE_LEN_128 * BASE_LEN_128;
                int64_t l1CustomSingleMTmp = AlignData(td_->opInfo.get_S1(), BASE_LEN_128);
                tmpData_.l1CustomSingleM =
                    (tmpData_.l1CustomSingleM > l1CustomSingleMTmp) ? l1CustomSingleMTmp : tmpData_.l1CustomSingleM;
            }
        } else {
            tmpData_.isL1CustomEnable = false;
            OP_LOGD(context_, "FlashAttentionScoreGradTilingS1s2Bn2 l1CustomManage disable.");
        }
    } else {
        OP_LOGD(context_, "FlashAttentionScoreGradTilingS1s2Bn2 l1CustomManage disable.");
    }

    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingS1s2Bn2::NMDStrategy()
{
    // SBH: B*H > 65535, 本身搬运效率太差，不调整S1/S2方向配比，使用默认参数
    if (tmpData_.layout == static_cast<uint32_t>(InputLayout::SBH) &&
        tmpData_.b * tmpData_.n2 * tmpData_.d > MAX_STRIDE_LIMIT) {
        return;
    }

    // D>64时，调整S2满配比及mm基本块
    if (td_->opInfo.get_D() > BASE_LEN_64) {
        if (td_->opInfo.get_S2() < S_LEN_512) {
            // 减少S1方向循环次数，调大mm基本块
            baseMmm = std::min(BASE_LEN_256, AlignData(static_cast<uint32_t>(td_->opInfo.get_S1()), FRACTAL_NUM));
            baseNmm = std::min(BASE_LEN_128, AlignData(static_cast<uint32_t>(td_->opInfo.get_S2()), FRACTAL_NUM));
            needSetFixSplit = true;
        } else if (td_->opInfo.get_S2() > S_LEN_512 && td_->opInfo.get_S2() < S_LEN_1024 &&
                   l2CacheSize > B4_L2_CACHESIZE) { // 特殊场景，受L2限制，不使能这个优化
            mmRatio = MM_RATIO_8;                   // S2方向满配比，减少S2方向数据重复搬运
        }
    }

    // 优先S2方向配比，S2方向不够时，S1方向配比，但保证S1和S2方向的配比不超过MM_RATIO
    s1Ratio = 1;
    s2Ratio = CeilCommon(td_->opInfo.get_S2(), baseNmm);
    s2Ratio = s2Ratio > mmRatio ? mmRatio : s2Ratio;
    s1Ratio = mmRatio / s2Ratio;
    // 根据实际S1长度调整配比
    s1Ratio = s1Ratio * baseMmm > td_->opInfo.get_S1() ? CeilCommon(td_->opInfo.get_S1(), baseMmm) : s1Ratio;

    // 强行保护s1Ratio、s2Ratio
    s1Ratio = s1Ratio > 0 ? s1Ratio : 1;
    s2Ratio = s2Ratio > 0 ? s2Ratio : 1;
    OP_LOGI(context_, "FlashAttentionScoreGradTilingS1s2Bn2 s1Ratio: %u, s2Ratio: %u", s1Ratio, s2Ratio);
}

void FlashAttentionScoreGradTilingS1s2Bn2::VectorBaseMNSplit()
{
    // Return Vector baseMN
    // dimS2(baseN) is lastAxis in ub need upper align.
    if (dimS2 < baseN && dimS1 > baseM) {
        baseN = AlignData(dimS2, MATMUL_MINI_N);
        // 16对齐，减少冗余数据；如果有dropOut和atten的时候会修正为32对其；
        baseM = tensorSize / baseN / (MATMUL_MINI_N) * (MATMUL_MINI_N);
        baseM = baseM > VEC_REPEAT_LIMIT ? (BASE_LEN_256 - (MATMUL_MINI_N * NUMBER_2)) : baseM;
    } else if (dimS2 > baseN && dimS1 < baseM) {
        baseM = dimS1;
        baseN = tensorSize / baseM / MATMUL_MINI_N * MATMUL_MINI_N;
    }

    baseM = baseM > BASE_LEN_128 ? BASE_LEN_128 : baseM; // simple-softmax only had 4KB UB
    baseN = baseN > BASE_LEN_512 ? BASE_LEN_512 : baseN; // nz-format only support 512

    bool existDropout = tmpData_.drop_out_cfg == DropOutConfig::EXIST_DROP_OUT;
    bool existAttenMask = tmpData_.atten_mask_cfg == AttenMaskConfig::EXIST_ATTEN_MASK;
    if (existDropout || existAttenMask) {
        // dropout and atten-mask only had 8K, baseN need align to 32
        baseM = tensorSize / (CeilCommon(baseN, BLOCK) * BLOCK);
        baseN = CeilCommon(baseN, BLOCK) * BLOCK;
        baseM = baseM > BASE_LEN_128 ? BASE_LEN_128 : baseM; // simple-softmax only had 4KB UB
        baseN = baseN > BASE_LEN_512 ? BASE_LEN_512 : baseN; // nz-format only support 512
    }
    return;
}

void FlashAttentionScoreGradTilingS1s2Bn2::SFTBaseMDSplit()
{
    // Return SFT-BaseM SFT-BaseD
    uint32_t typeSize = tmpData_.queryType == ge::DT_FLOAT ? B32 : B16;

    uint32_t blockNum = BLOCK / typeSize;
    uint32_t dimDAlign = CeilCommon(dimD, blockNum) * blockNum;

    // step0: 确定存储大小
    sftSingleM = BASE_LEN_256;
    if (tmpData_.isL1CustomEnable) {
        if (singleM / 2 < sftSingleM) {
            sftSingleM = singleM / 2;
        }
    } else {
        if (singleM < sftSingleM) {
            sftSingleM = singleM;
        }
    }

    // step1: 确定D是否需要切分
    if (dimDAlign > tensorSize) {
        // template
        sftSingleM = BASE_LEN_128;
        sftBaseM = BASE_LEN_128;
        baseD = BASE_LEN_64;
        needSplitD = true;
    } else {
        baseD = dimDAlign;
        needSplitD = false;
    }

    // step2: 确定sftBaseM
    sftBaseM = tensorSize / baseD;
    if (sftBaseM > sftSingleM) {
        sftBaseM = sftSingleM;
    }
}

void FlashAttentionScoreGradTilingS1s2Bn2::DecideBaseMND()
{
    /*
    Func:
    0. 确定SFT-BaseM
    1. 确定VECTOR-BaseMN
    2. 确定MATMUL-BaseMN
    3. 确定SingleMN(BMM与VECTOR共有参数)
    4. 确定D通道切分(受限于新算法, D通道切分依赖于Matmul-BaseM)
       eg: baseMmm = 128, baseD <= 64
    */
    // Init
    baseM = BASE_LEN_64;
    baseN = BASE_LEN_128;
    baseMmm = BASE_LEN_128;
    baseNmm = BASE_LEN_128;
    tensorSize = baseM * baseN;
    mmRatio = MM_RATIO_4;
    dimD = td_->opInfo.get_D();
    dimS2 = td_->opInfo.get_S2();
    dimS1 = td_->opInfo.get_S1();

    VectorBaseMNSplit();

    // Matmul特殊处理（cacheline友好）
    if (dimD <= BASE_LEN_64 && dimS2 >= BASE_LEN_256) {
        baseMmm = BASE_LEN_128;
        baseNmm = BASE_LEN_256;
        baseM = BASE_LEN_32;
        baseN = BASE_LEN_256;
    } else {
        baseMmm = BASE_LEN_128;
        baseNmm = BASE_LEN_128;
    }

    // 部分场景需要调整配比
    s1Ratio = 1;
    s2Ratio = mmRatio;
    NMDStrategy();

    if (tmpData_.isL1CustomEnable) {
        singleM = baseMmm * s1Ratio * 2; // sameAB调大cube侧基本块
    } else {
        singleM = baseMmm * s1Ratio;
    }
    singleN = baseNmm * s2Ratio;

    if (tmpData_.isL1CustomEnable) {
        singleM = tmpData_.l1CustomSingleM;
        singleN = tmpData_.l1CustomSingleN;
        OP_LOGD(context_, "FlashAttentionScoreGradTilingS1s2Bn2 l1CustomManage enable, singleMN [%u %u]", singleM,
                  singleN);
    }

    SFTBaseMDSplit();
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::CheckAttenMaskShape()
{
    // check atten_mask shape when enable atten_mask_compress
    if (attenMaskCompressMode == NO_COMPRESS_MODE) {
        bool invalid =
            td_->opInfo.get_existAttenMask() != 0 && (attenMaskS1Size != tmpData_.s1 || attenMaskS2Size != tmpData_.s2);
        if (invalid) {
            OP_LOGE(context_, "atten mask shape [%ld,%ld] is invalid.", attenMaskS1Size, attenMaskS2Size);
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (attenMaskCompressMode == PREFIX_COMPRESS_MODE) {
        if (attenMaskS1Size != PREFIX_COMPRESS_S1_SIZE || attenMaskS2Size != ATTEN_MASK_COMPRESS_LIMIT) {
            OP_LOGE(context_,
                      "atten mask shape for prefix compress mode is invalid, try setting it to [3072, 2048].");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (attenMaskS1Size != attenMaskS2Size) {
        OP_LOGE(context_, "atten mask shape is not square.");
        return ge::GRAPH_FAILED;
    }
    if (attenMaskS2Size != ATTEN_MASK_COMPRESS_LIMIT) {
        OP_LOGE(context_, "atten mask shape is invalid, try setting it to [2048, 2048].");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreGradTilingS1s2Bn2::CheckForDichotomy(std::vector<int64_t> &nums, uint32_t x, uint32_t m)
{
    int64_t sum = 0;
    uint32_t cnt = 1;
    for (long unsigned int i = 0; i < nums.size(); i++) {
        if (sum + nums[i] > x) {
            cnt++;
            sum = nums[i];
        } else {
            sum += nums[i];
        }
    }
    return (cnt <= m);
}

int64_t FlashAttentionScoreGradTilingS1s2Bn2::GetSplitArrayMinMaxSum(std::vector<int64_t> &s1s2WeightNums,
                                                                     uint32_t coreNum)
{
    int64_t left = 0;
    int64_t right = 0;
    for (long unsigned int i = 0; i < s1s2WeightNums.size(); i++) {
        right += s1s2WeightNums[i];
        if (left < s1s2WeightNums[i]) {
            left = s1s2WeightNums[i];
        }
    }
    while (left < right) {
        int64_t mid = (left + right) >> 1;
        if (CheckForDichotomy(s1s2WeightNums, mid, coreNum)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::DoBlockTiling()
{
    int64_t nNums = td_->opInfo.get_B() * td_->opInfo.get_N2();
    int64_t formerCoreProcessNNums = CeilCommon(nNums, aicoreParams_.numBlocks);
    int64_t remainCoreNum = formerCoreProcessNNums * aicoreParams_.numBlocks - nNums;
    td_->opInfo.set_usedCoreNum(aicoreParams_.numBlocks);
    int64_t sameAbBlockDim = 0;
    if (tmpData_.isL1CustomEnable) {
        sameAbBlockDim = aicoreParams_.aicNum;
        formerCoreProcessNNums = CeilCommon(nNums, sameAbBlockDim);
        remainCoreNum = formerCoreProcessNNums * sameAbBlockDim - nNums;
        td_->opInfo.set_usedCoreNum(sameAbBlockDim);
    }

    td_->opInfo.set_castUsedCoreNum(aicoreParams_.numBlocks); // for pre & post process
    if (nNums < static_cast<int64_t>(aicoreParams_.numBlocks)) {
        if (td_->opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
            // 需要给出真实使用的core数，用于bN2idxStarts，bN2idxEnds的检索
            if (tmpData_.isL1CustomEnable) {
                if (nNums < sameAbBlockDim) {
                    td_->opInfo.set_usedCoreNum(nNums);
                }
            } else {
                td_->opInfo.set_usedCoreNum(nNums);
            }
            td_->opInfo.set_castUsedCoreNum(nNums);
        } else {
            // 非TND
            int64_t minS = std::min(td_->opInfo.get_S1(), td_->opInfo.get_S2());
            if (nNums * minS * td_->opInfo.get_D() / aicoreParams_.numBlocks * tmpData_.dataTypeSize < 8 * 1024) {
                // 后处理ub利用率严重不足时，减少核启动
                if (tmpData_.isL1CustomEnable) {
                    if (nNums < sameAbBlockDim) {
                        td_->opInfo.set_usedCoreNum(nNums);
                    }
                } else {
                    td_->opInfo.set_usedCoreNum(nNums);
                }
                td_->opInfo.set_castUsedCoreNum(nNums);
                needAdjustBlockDim = true;
            }
        }
    }

    td_->opInfo.set_formerCoreNum((tmpData_.isL1CustomEnable ? sameAbBlockDim : aicoreParams_.numBlocks) - remainCoreNum);
    td_->opInfo.set_formerCoreProcessNNum(formerCoreProcessNNums);
    td_->opInfo.set_remainCoreProcessNNum(static_cast<uint32_t>(formerCoreProcessNNums - 1));

    // 确定性计算下的TND对应的分核策略
    if (td_->opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        if (nNums <= static_cast<int64_t>(tmpData_.isL1CustomEnable ? sameAbBlockDim : aicoreParams_.numBlocks)) {
            for (int64_t i = 0; i < nNums; i++) {
                tmpData_.bN2idxStarts[i] = i;
                tmpData_.bN2idxEnds[i] = i;
            }
        } else {
            int64_t minMaxSum = GetSplitArrayMinMaxSum(
                tmpData_.s1s2Weight, tmpData_.isL1CustomEnable ? sameAbBlockDim : aicoreParams_.numBlocks);
            OP_LOGD(context_->GetNodeName(), "FlashAttentionScoreGradTilingS1s2Bn2 splite B*N2 minMaxSum: %ld.",
                      minMaxSum);
            int64_t tempSum = 0;
            uint32_t coreIdxBegin = 0;
            uint32_t coreIdxEnd = 0;
            uint32_t tempSumIdx = 0;
            tmpData_.bN2idxStarts[coreIdxBegin++] = 0;
            for (uint32_t i = 0; i < tmpData_.s1s2Weight.size(); i++) {
                tempSum += tmpData_.s1s2Weight[i];
                if ((tempSum > minMaxSum) ||
                    ((tmpData_.isL1CustomEnable ? sameAbBlockDim : aicoreParams_.numBlocks) - (tempSumIdx + 1) >=
                     (tmpData_.s1s2Weight.size() - i))) {
                    tempSumIdx++;
                    tempSum = tmpData_.s1s2Weight[i];
                    tmpData_.bN2idxEnds[coreIdxEnd++] = i - 1;
                    tmpData_.bN2idxStarts[coreIdxBegin++] = i;
                    if ((tmpData_.isL1CustomEnable ? sameAbBlockDim : aicoreParams_.numBlocks) - tempSumIdx >=
                        (tmpData_.s1s2Weight.size() - i)) {
                        coreIdxBegin--;
                        for (uint32_t k = 0; k < tmpData_.s1s2Weight.size() - i; k++) {
                            tmpData_.bN2idxStarts[coreIdxBegin++] = i + k;
                            tmpData_.bN2idxEnds[coreIdxEnd++] = i + k;
                        }
                        coreIdxEnd--;
                        break;
                    }
                }
            }
            tmpData_.bN2idxEnds[coreIdxEnd] = tmpData_.s1s2Weight.size() - 1;
        }
        int64_t maxCoreNum = std::min(nNums, static_cast<int64_t>(aicoreParams_.numBlocks));
        if (tndEmptyTensorFlag) {
            for (int64_t coreIdx = 0; coreIdx < maxCoreNum; ++coreIdx) {
                int64_t idx = tmpData_.bN2idxEnds[coreIdx];
                while (idx >= tmpData_.bN2idxStarts[coreIdx]) {
                    if (tmpData_.s1s2Weight[idx] != 0) {
                        tmpData_.bN2idxEnds[coreIdx] = idx;
                        if (coreIdx < maxCoreNum - 1) {
                            tmpData_.bN2idxStarts[coreIdx + 1] = idx + 1;
                        }
                        break;
                    }
                    --idx;
                }
            }
        }
        td_->tndSplitCoreParams.set_bN2idxStarts(tmpData_.bN2idxStarts);
        td_->tndSplitCoreParams.set_bN2idxEnds(tmpData_.bN2idxEnds);
    }

    // debugInfo
    OP_LOGD(context_->GetNodeName(), "FlashAttentionScoreGradTilingS1s2Bn2 splite B*N2 usedCoreNum: %u.",
              td_->opInfo.get_usedCoreNum());
    for (uint32_t i = 0; i < td_->opInfo.get_usedCoreNum(); i++) {
        OP_LOGD(context_->GetNodeName(),
                  "FlashAttentionScoreGradTilingS1s2Bn2 splite B*N2 CoreIdx: %u begin: %ld end: %ld.", i,
                  td_->tndSplitCoreParams.get_bN2idxStarts()[i], td_->tndSplitCoreParams.get_bN2idxEnds()[i]);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::DoOpTiling()
{
    OP_LOGI(context_, "FlashAttentionScoreGradTilingS1s2Bn2 DoTiling start");

    // 确定基本块大小
    DecideBaseMND();

    // 确定主kernel-UBLength
    auto inputBufferLen = tensorSize * B32;
    td_->opInfo.set_inputBufferLen(inputBufferLen);
    td_->opInfo.set_helpBufferLen(inputBufferLen);

    // setTilingData
    td_->splitCoreParams.set_SFTBaseM(sftBaseM);
    td_->splitCoreParams.set_SFTSingleM(sftSingleM);
    td_->splitCoreParams.set_dInner(baseD);
    td_->splitCoreParams.set_baseM(baseM);
    td_->splitCoreParams.set_baseN(baseN);
    td_->splitCoreParams.set_singleM(singleM);
    td_->splitCoreParams.set_singleN(singleN);
    td_->splitCoreParams.set_s1OuterOuter(CeilCommon(td_->opInfo.get_S1(), singleM));
    td_->splitCoreParams.set_s2OuterOuter(CeilCommon(td_->opInfo.get_S2(), singleN));

    auto status = DoBlockTiling();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = DoCastTiling();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    DoPreTiling();
    SetQKVStartIdx();

    if (CheckAttenMaskShape() != ge::GRAPH_SUCCESS) {
        PrintShapeInfo();
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingS1s2Bn2::PrintShapeInfo()
{
    OP_LOGI(context_,
              "FAG S1s2Bn2 with shape b[%ld] n2[%ld] g[%ld] s1[%ld] s2[%ld] d[%ld] preToken[%ld] nextToken[%ld]!",
              td_->opInfo.get_B(), td_->opInfo.get_N2(), td_->opInfo.get_G(), td_->opInfo.get_S1(), td_->opInfo.get_S2(),
              td_->opInfo.get_D(), td_->opInfo.get_preTokens(), td_->opInfo.get_nextTokens());
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::DoLibApiTiling()
{
    // calc for simpleSoftMax which dstShape is as same as srcShape
    auto simpleSoftMaxShape = ge::Shape({baseM, baseN});
    auto helpLenA = tensorSize * tmpData_.dataTypeSize; // UB内数据类型
    AscendC::SoftMaxTilingFunc(simpleSoftMaxShape, sizeof(float), helpLenA, td_->softmaxTilingData);

    // calc for softmaxGrad
    // calcByFp32 is [sftBaseM, 8], calcByFP16 is [sftBaseM, 16]
    auto softmaxGradShape = ge::Shape({sftBaseM, BLOCK / tmpData_.dataTypeSize});
    auto helpLenB = 2 * tensorSize * tmpData_.dataTypeSize; // UB内数据类型 64KB
    AscendC::SoftMaxGradTilingFunc(softmaxGradShape, tmpData_.dataTypeSize, helpLenB, td_->softmaxGradTilingData, true);

    // calc BMM
    auto status = SetBmm1TilingData(baseMmm, baseNmm, aicoreParams_.l1Size);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = SetBmm31TilingData(aicoreParams_.l1Size);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = SetBmm4TilingData(aicoreParams_.l1Size);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::DoCastTiling()
{
    // query
    int64_t dAlign = (td_->opInfo.get_D() + 15) / 16 * 16;
    int64_t allNumQuery = td_->opInfo.get_B() * td_->opInfo.get_N2() * td_->opInfo.get_G() * td_->opInfo.get_S1() * dAlign;
    // TND时候要按照真实的query的num数计算
    if (td_->opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumQuery = tmpData_.t1 * td_->opInfo.get_N2() * td_->opInfo.get_G() * td_->opInfo.get_D();
    }

    // K V
    int64_t allNumKv = td_->opInfo.get_B() * td_->opInfo.get_N2() * td_->opInfo.get_S2() * dAlign;
    // TND时候要按照真实的k，v的num数计算
    if (td_->opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        allNumKv = tmpData_.t2 * td_->opInfo.get_N2() * 1 * td_->opInfo.get_D();
    }
    uint32_t typeSize = tmpData_.queryType == ge::DT_FLOAT ? B32 : B16;
    uint32_t usedCoreNum = td_->opInfo.get_castUsedCoreNum();
    constexpr uint32_t postNzCoexNode = 10;
    constexpr uint32_t blockSize = 32;
    constexpr uint32_t postNzReservedN = 1;
    tmpData_.isMM345NZOut =
        tmpData_.queryType != ge::DT_FLOAT && td_->opInfo.get_layout() != static_cast<uint32_t>(InputLayout::TND) &&
        (tmpData_.d == 72 || tmpData_.d == 80 || tmpData_.d == 88 || tmpData_.d == 96); // mm345NZ出 72, 80, 88, 96
    uint32_t postUbBaseSize = 0;
    uint32_t qPostBaseNum = 0;
    int64_t nzReservedSize = 0;
    if (!tmpData_.isMM345NZOut) {
        postUbBaseSize = (aicoreParams_.ubSize) / POST_COEX_NODE / BUFFER_NUM / BASE_LEN_256 * BASE_LEN_256;
        qPostBaseNum = postUbBaseSize / typeSize;
    } else {
        int64_t curPostCoexNode = postNzCoexNode;
        nzReservedSize = dAlign / 16 * blockSize * postNzReservedN; // 16为一个单元长度
        postUbBaseSize = (aicoreParams_.ubSize - 2 * nzReservedSize) / curPostCoexNode /
                         BUFFER_NUM / // 开DB预留2份nzReservedSize
                         BASE_LEN_256 * BASE_LEN_256;
        qPostBaseNum = postUbBaseSize / typeSize / dAlign * td_->opInfo.get_D();
    }

    OP_CHECK_IF(qPostBaseNum == 0, OP_LOGE(context_, "qPostBaseNum is 0."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(usedCoreNum == 0, OP_LOGE(context_, "castUsedCoreNum is 0."),
               return ge::GRAPH_FAILED);
    int64_t qPostBlockTotal = allNumQuery / dAlign * td_->opInfo.get_D();
    int64_t qSizeAlign = (qPostBlockTotal + BASE_LEN_256 - 1) / GM_ALIGN * GM_ALIGN * typeSize;
    int64_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
    int64_t qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
    int64_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;
    int64_t qPostBlockFactor = (qPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    int64_t kvPostBaseNum = qPostBaseNum;
    OP_CHECK_IF(kvPostBaseNum == 0, OP_LOGE(context_, "kvPostBaseNum is 0."),
               return ge::GRAPH_FAILED);
    int64_t kvPostBlockTotal = allNumKv / dAlign * td_->opInfo.get_D();
    int64_t kvSizeAlign = (kvPostBlockTotal + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN * typeSize;
    int64_t kvPostTailNumTmp = kvPostBlockTotal % kvPostBaseNum;
    int64_t kvPostTailNum = kvPostTailNumTmp == 0 ? kvPostBaseNum : kvPostTailNumTmp;
    int64_t kvPostBlockOuterTotal = (kvPostBlockTotal + kvPostBaseNum - 1) / kvPostBaseNum;
    int64_t kvPostBlockFactor = (kvPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

    int64_t vecQueUbSize = 183 * 1024; // 与kernel vecQue共用ub空间，最大只有183K
    int64_t totalDtypeSize = 0;
    uint32_t postMaxDataSize = 0;

    if (!tmpData_.isMM345NZOut) {
        totalDtypeSize = tmpData_.queryType == ge::DT_FLOAT ? B32 + B32 : B16 + B32;
    } else {
        totalDtypeSize = tmpData_.queryType == ge::DT_FLOAT ? B32 + B32 : B16 + B16 + B32;
    }
    postMaxDataSize = vecQueUbSize / totalDtypeSize;

    td_->postTilingData.set_coreNum(usedCoreNum);
    td_->postTilingData.set_scaleValue(td_->opInfo.get_scaleValue());
    td_->postTilingData.set_postUbBaseSize(postUbBaseSize);
    td_->postTilingData.set_postMaxDataSize(postMaxDataSize);
    td_->postTilingData.set_qPostBlockFactor(qPostBlockFactor);
    td_->postTilingData.set_qPostBlockTotal(qPostBlockTotal);
    td_->postTilingData.set_qPostBaseNum(qPostBaseNum);
    td_->postTilingData.set_qPostTailNum(qPostTailNum);
    td_->postTilingData.set_qSizeAlign(qSizeAlign);

    td_->postTilingData.set_kvPostBlockFactor(kvPostBlockFactor);
    td_->postTilingData.set_kvPostBlockTotal(kvPostBlockTotal);
    td_->postTilingData.set_kvPostBaseNum(kvPostBaseNum);
    td_->postTilingData.set_kvPostTailNum(kvPostTailNum);
    td_->postTilingData.set_kvSizeAlign(kvSizeAlign);
    td_->postTilingData.set_nzReservedSize(nzReservedSize);

    td_->opInfo.set_dqWorkspaceLen((allNumQuery * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN);
    td_->opInfo.set_dkWorkspaceLen((allNumKv * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN);
    td_->opInfo.set_dvWorkspaceLen((allNumKv * B32 + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN);

    td_->postTilingData.set_b(td_->opInfo.get_B());
    td_->postTilingData.set_n2(td_->opInfo.get_N2());
    td_->postTilingData.set_g(td_->opInfo.get_G());
    td_->postTilingData.set_s1(td_->opInfo.get_S1());
    td_->postTilingData.set_s2(td_->opInfo.get_S2());
    td_->postTilingData.set_d(td_->opInfo.get_D());

    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingS1s2Bn2::DoPreTiling()
{
    // 120KB FP16Tensor, 60KB U8Tensor, 8KB MaskTensor, 512B HelpTensor which less than UB(192KB).
    // singleUBProcessNum: UB最大处理FP16数据大小，需保证能被128整除.
    uint32_t castBufferLen = 60 * 1024;
    uint32_t outputBufferLen = 30 * 1024;
    uint32_t inputBufferLen = 4 * 1024;
    int64_t singleUBProcessNum = castBufferLen / NUMBER_2;

    tmpData_.dropMaskSize = static_cast<int64_t>(td_->opInfo.get_B()) * td_->opInfo.get_N2() * td_->opInfo.get_G() *
                            td_->opInfo.get_S1() * td_->opInfo.get_S2();
    if (td_->opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        tmpData_.dropMaskSize = tmpData_.sumS1S2Product * td_->opInfo.get_N2() * td_->opInfo.get_G();
    }
    uint32_t usedCoreNum = td_->opInfo.get_castUsedCoreNum();

    int64_t maskSize = AlignTo(tmpData_.dropMaskSize, static_cast<int64_t>(BOOL_BLOCK_NUMS));
    int64_t singleCoreNum =
        AlignTo(CeilCommon(maskSize, static_cast<int64_t>(usedCoreNum)), static_cast<int64_t>(BOOL_BLOCK_NUMS));
    uint32_t maskUsedCoreNum = static_cast<uint32_t>(CeilCommon(maskSize, singleCoreNum));

    int64_t tailCoreNum = maskSize - (maskUsedCoreNum - 1) * singleCoreNum;
    tailCoreNum = AlignTo(tailCoreNum, static_cast<int64_t>(BOOL_BLOCK_NUMS));

    uint32_t singleCoreUBLoop = static_cast<uint32_t>(CeilCommon(singleCoreNum, singleUBProcessNum));
    uint32_t tailCoreUBLoop = static_cast<uint32_t>(CeilCommon(tailCoreNum, singleUBProcessNum));

    uint32_t singleCoreUBLastLoopNum =
        static_cast<uint32_t>(singleCoreNum - (singleCoreUBLoop - 1) * singleUBProcessNum);
    uint32_t tailCoreUBLastLoopNum = static_cast<uint32_t>(tailCoreNum - (tailCoreUBLoop - 1) * singleUBProcessNum);

    td_->preTilingData.set_maskCoreNum(maskUsedCoreNum);
    td_->preTilingData.set_castBufferLen(castBufferLen);
    td_->preTilingData.set_outputBufferLen(outputBufferLen);
    td_->preTilingData.set_inputBufferLen(inputBufferLen);
    td_->preTilingData.set_singleUBProcessNum(static_cast<uint32_t>(singleUBProcessNum));
    td_->preTilingData.set_maskSingleCoreNum(singleCoreNum); // size == num
    td_->preTilingData.set_maskSingleCoreLoop(singleCoreUBLoop);
    td_->preTilingData.set_maskLastLoopNum(singleCoreUBLastLoopNum);
    td_->preTilingData.set_maskTailCoreLoop(tailCoreUBLoop);
    td_->preTilingData.set_maskTailCoreLastLoopNum(tailCoreUBLastLoopNum);

    td_->preTilingData.set_qPreBlockFactor(0);
    td_->preTilingData.set_qPreBlockTotal(0);
    td_->preTilingData.set_qPreBlockTail(0);
    td_->preTilingData.set_kvPreBlockFactor(0);
    td_->preTilingData.set_kvPreBlockTotal(0);
    td_->preTilingData.set_kvPreBlockTail(0);
    td_->preTilingData.set_maskPreBlockTotal(0);

    int64_t dropoutWorkspaceLen = 0;
    uint32_t dropoutIsDivisibleBy8 = 1;
    // TND layout对应的每个B下面的S2分别对baseN进行求余，判断是否可以整除
    if (td_->opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        if (td_->opInfo.get_keepProb() < 1) {
            for (int64_t i = 0; i < tmpData_.b; i++) {
                if (tmpData_.actualSeqKvlen[i] % baseN != 0) {
                    dropoutIsDivisibleBy8 = 0;
                    break;
                }
            }
            if (dropoutIsDivisibleBy8 == 0) {
                dropoutWorkspaceLen = static_cast<int64_t>(tmpData_.sumS1S2Product);
                dropoutWorkspaceLen *= static_cast<int64_t>(td_->opInfo.get_N2());
                dropoutWorkspaceLen *= static_cast<int64_t>(td_->opInfo.get_G());
                dropoutWorkspaceLen = (dropoutWorkspaceLen + BLOCK - 1) / BLOCK * BLOCK;
                dropoutWorkspaceLen = (dropoutWorkspaceLen + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;
            }
        }
    } else if (td_->opInfo.get_keepProb() < 1 && td_->opInfo.get_S2() % baseN != 0) {
        dropoutIsDivisibleBy8 = 0;
        dropoutWorkspaceLen = td_->opInfo.get_B();
        dropoutWorkspaceLen *= td_->opInfo.get_N2();
        dropoutWorkspaceLen *= td_->opInfo.get_G();
        dropoutWorkspaceLen *= td_->opInfo.get_S1();
        dropoutWorkspaceLen *= td_->opInfo.get_S2();
        dropoutWorkspaceLen = (dropoutWorkspaceLen + BLOCK - 1) / BLOCK * BLOCK;
        dropoutWorkspaceLen = (dropoutWorkspaceLen + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;
    }
    td_->opInfo.set_dropoutWorkspaceLen(dropoutWorkspaceLen);
    td_->preTilingData.set_dropoutIsDivisibleBy8(dropoutIsDivisibleBy8);
    td_->preTilingData.set_dropBeginAddr(0);
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::GetWorkspaceSize()
{
    int64_t currentUseCoreNum =
        tmpData_.isL1CustomEnable ? (td_->opInfo.get_usedCoreNum() * 2) : td_->opInfo.get_usedCoreNum();
    int64_t launchBlockDims = needAdjustBlockDim ? currentUseCoreNum : aicoreParams_.numBlocks;
    // Tiling传递的内存大小、起始地址，统一为字节数，单位为B
    auto blockdim = CalcTschBlockDim(launchBlockDims, aicoreParams_.aicNum, aicoreParams_.numBlocks);
    OP_CHECK_IF(blockdim == 0,
               OP_LOGE(context_, "blockdim is 0, aicNum is %lu, aivNum is %lu.",
                                           aicoreParams_.aicNum, aicoreParams_.numBlocks),
               return ge::GRAPH_FAILED);
    context_->SetBlockDim(blockdim);
    context_->SetScheduleMode(1);

    // 系统预留
    int64_t sysLen = WORKSPACE_BASE_CAL;
    // dropout output
    int64_t dropoutWorkspaceLen = td_->opInfo.get_dropoutWorkspaceLen();

    // bmm1/bmm2 output
    int64_t mm1WorkspaceLen = singleM * singleN * tmpData_.dataTypeSize;
    mm1WorkspaceLen = AlignData(mm1WorkspaceLen, GM_ALIGN);
    int64_t mm2WorkspaceLen = singleM * singleN * tmpData_.dataTypeSize;
    mm2WorkspaceLen = AlignData(mm2WorkspaceLen, GM_ALIGN);

    // bmm4/bmm3.1/bmm3.2 input
    int64_t typeSize = tmpData_.queryType == ge::DT_FLOAT ? B32 : B16;
    int64_t space = singleM * singleN * typeSize;
    space = AlignData(space, GM_ALIGN);
    // 输出matmul异步需要做PingPang防止被踩
    int64_t mm4InputWorkspaceLen = space + space;
    int64_t mm3InputWorkspaceLen = space + space;

    // dqCast
    int64_t dqWorkspaceLen = td_->opInfo.get_dqWorkspaceLen();
    // dkCast
    int64_t dkWorkspaceLen = td_->opInfo.get_dkWorkspaceLen();
    // dvCast
    int64_t dvWorkspaceLen = td_->opInfo.get_dvWorkspaceLen();

    // set global workspace
    // 内存顺序排布
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = sysLen;
    workspaces[0] += dropoutWorkspaceLen;
    workspaces[0] += (mm1WorkspaceLen + mm2WorkspaceLen) * currentUseCoreNum;
    workspaces[0] += (mm4InputWorkspaceLen + mm3InputWorkspaceLen) * td_->opInfo.get_usedCoreNum();
    workspaces[0] += dqWorkspaceLen + dkWorkspaceLen + dvWorkspaceLen;
    if (tmpData_.pseType == PSE_INNER_MUL_ADD_TYPE || tmpData_.pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
        tmpData_.pseAlibiBaseS2 = PSE_ALIBI_S2_LIMIT_SIZE;
        int64_t s2Tail = td_->opInfo.get_S2() % PSE_ALIBI_S2_LIMIT_SIZE;
        if (s2Tail != 0) {
            tmpData_.pseAlibiBaseS1 =
                std::min(static_cast<int64_t>(BASE_LEN_128), UB_BASIC_LIMIT_SIZE / AlignUp(s2Tail, FRACTAL_NUM));
        } else {
            tmpData_.pseAlibiBaseS1 =
                std::min(static_cast<int64_t>(BASE_LEN_128), UB_BASIC_LIMIT_SIZE / tmpData_.pseAlibiBaseS2);
        }
        tmpData_.pseAlibiBaseS1 = std::max(tmpData_.pseAlibiBaseS1, UB_BASIC_LIMIT_SIZE / BASE_LEN_128);
        int64_t pseAlibiBytes =
            AlignUp(tmpData_.pseAlibiBaseS2 * tmpData_.pseAlibiBaseS1 * 2, GM_ALIGN) * td_->opInfo.get_usedCoreNum();
        workspaces[0] += pseAlibiBytes;

        td_->opInfo.set_pseAlibiBaseS1(tmpData_.pseAlibiBaseS1);
        td_->opInfo.set_pseAlibiBaseS2(tmpData_.pseAlibiBaseS2);
    }

    // setTilingData
    td_->opInfo.set_mm1WorkspaceLen(mm1WorkspaceLen);
    td_->opInfo.set_mm2WorkspaceLen(mm2WorkspaceLen);
    td_->opInfo.set_mm4InputWorkspaceLen(mm4InputWorkspaceLen);
    td_->opInfo.set_mm3InputWorkspaceLen(mm3InputWorkspaceLen);

    int64_t workspaceOffsets =
        td_->opInfo.get_dropoutWorkspaceLen() +
        (td_->opInfo.get_mm1WorkspaceLen() + td_->opInfo.get_mm2WorkspaceLen()) * currentUseCoreNum;
    workspaceOffsets +=
        (td_->opInfo.get_mm4InputWorkspaceLen() + td_->opInfo.get_mm3InputWorkspaceLen()) * td_->opInfo.get_usedCoreNum();

    td_->postTilingData.set_dqWorkSpaceOffset(workspaceOffsets);

    workspaceOffsets = workspaceOffsets + td_->opInfo.get_dqWorkspaceLen();
    td_->postTilingData.set_dkWorkSpaceOffset(workspaceOffsets);

    workspaceOffsets = workspaceOffsets + td_->opInfo.get_dkWorkspaceLen();
    td_->postTilingData.set_dvWorkSpaceOffset(workspaceOffsets);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::GetLayoutInfo()
{
    const char *inputLayout = context_->GetAttrs()->GetAttrPointer<char>(INPUT_LAYOUT);
    if (strcmp(inputLayout, BSH_STR) == 0) {
        td_->opInfo.set_layout(static_cast<uint32_t>(InputLayout::BSH));
    } else if (strcmp(inputLayout, SBH_STR) == 0) {
        td_->opInfo.set_layout(static_cast<uint32_t>(InputLayout::SBH));
    } else if (strcmp(inputLayout, BNSD_STR) == 0) {
        td_->opInfo.set_layout(static_cast<uint32_t>(InputLayout::BNSD));
    } else if (strcmp(inputLayout, BSND_STR) == 0) {
        td_->opInfo.set_layout(static_cast<uint32_t>(InputLayout::BSND));
    } else if (strcmp(inputLayout, TND_STR) == 0) {
        td_->opInfo.set_layout(static_cast<uint32_t>(InputLayout::TND));
    } else {
        OP_LOGW(context_, "FlashAttentionScoreGradTilingS1s2Bn2 unsupported layout");
        return ge::GRAPH_PARAM_INVALID;
    }
    tmpData_.layout = td_->opInfo.get_layout();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::SetBaseInfo(const gert::Shape &queryShape,
                                                                  const gert::Shape &keyShape, int64_t dimN1)
{
    OP_CHECK_IF(dimN1 == 0, OP_LOGW(context_, "headNum is 0."), return ge::GRAPH_PARAM_INVALID);

    uint32_t dimSize = queryShape.GetDimNum();
    const char *inputLayout = context_->GetAttrs()->GetAttrPointer<char>(INPUT_LAYOUT);
    OP_CHECK_IF(static_cast<size_t>(dimSize) != strlen(inputLayout),
               OP_LOGW(context_, "layout dims is not inputLayout's length."), return ge::GRAPH_PARAM_INVALID);
    if (td_->opInfo.get_layout() == static_cast<uint32_t>(InputLayout::BNSD)) {
        td_->opInfo.set_B(queryShape.GetDim(DIM_0));
        OP_CHECK_IF(keyShape.GetDim(DIM_1) == 0, OP_LOGW(context_, "dim 1 of key is 0."),
                   return ge::GRAPH_PARAM_INVALID);
        td_->opInfo.set_G(queryShape.GetDim(DIM_1) / keyShape.GetDim(DIM_1));
        OP_CHECK_IF(td_->opInfo.get_G() == 0, OP_LOGW(context_, "g is 0"), return ge::GRAPH_PARAM_INVALID);
        td_->opInfo.set_N2(keyShape.GetDim(DIM_1));
        td_->opInfo.set_S1(queryShape.GetDim(DIM_2));
        td_->opInfo.set_S2(keyShape.GetDim(DIM_2));
        td_->opInfo.set_D(queryShape.GetDim(DIM_3));
    } else if (td_->opInfo.get_layout() == static_cast<uint32_t>(InputLayout::BSND)) {
        OP_CHECK_IF((queryShape.GetDimNum() != DIMS_FOUR || keyShape.GetDimNum() != DIMS_FOUR),
                   OP_LOGW(context_, "the dim of query or key is not 4."), return ge::GRAPH_PARAM_INVALID);
        td_->opInfo.set_B(queryShape.GetDim(DIM_0));
        OP_CHECK_IF(keyShape.GetDim(DIM_2) == 0, OP_LOGW(context_, "dim 2 of key is 0."),
                   return ge::GRAPH_PARAM_INVALID);
        td_->opInfo.set_G(queryShape.GetDim(DIM_2) / keyShape.GetDim(DIM_2));
        OP_CHECK_IF(td_->opInfo.get_G() == 0, OP_LOGW(context_, "g is 0"), return ge::GRAPH_PARAM_INVALID);
        td_->opInfo.set_N2(keyShape.GetDim(DIM_2));
        td_->opInfo.set_S1(queryShape.GetDim(DIM_1));
        td_->opInfo.set_S2(keyShape.GetDim(DIM_1));
        td_->opInfo.set_D(queryShape.GetDim(DIM_3));
    } else if (td_->opInfo.get_layout() == static_cast<uint32_t>(InputLayout::BSH)) {
        OP_CHECK_IF((queryShape.GetDimNum() != DIMS_THREE || keyShape.GetDimNum() != DIMS_THREE),
                   OP_LOGW(context_, "the dim of query or key is not 3."), return ge::GRAPH_PARAM_INVALID);
        int64_t tempDimD = queryShape.GetDim(DIM_2) / dimN1;
        OP_CHECK_IF(tempDimD == 0, OP_LOGW(context_, "tempDimD is 0."), return ge::GRAPH_PARAM_INVALID);
        int64_t dimN2 = keyShape.GetDim(DIM_2) / tempDimD;
        OP_CHECK_IF(dimN2 == 0, OP_LOGW(context_, "dimN2 is 0."), return ge::GRAPH_PARAM_INVALID);
        td_->opInfo.set_B(queryShape.GetDim(DIM_0));
        td_->opInfo.set_G(dimN1 / dimN2);
        OP_CHECK_IF(td_->opInfo.get_G() == 0, OP_LOGW(context_, "g is 0"), return ge::GRAPH_PARAM_INVALID);
        td_->opInfo.set_N2(dimN2);
        td_->opInfo.set_S1(queryShape.GetDim(DIM_1));
        td_->opInfo.set_S2(keyShape.GetDim(DIM_1));
        td_->opInfo.set_D(tempDimD);
    } else if (td_->opInfo.get_layout() == static_cast<uint32_t>(InputLayout::SBH)) {
        int64_t tempDimD = queryShape.GetDim(DIM_2) / dimN1;
        OP_CHECK_IF(tempDimD == 0, OP_LOGW(context_, "tempDimD is 0."), return ge::GRAPH_PARAM_INVALID);
        int64_t dimN2 = keyShape.GetDim(DIM_2) / tempDimD;
        OP_CHECK_IF(dimN2 == 0, OP_LOGW(context_, "dimN2 is 0."), return ge::GRAPH_PARAM_INVALID);
        td_->opInfo.set_B(queryShape.GetDim(DIM_1));
        td_->opInfo.set_G(dimN1 / dimN2);
        OP_CHECK_IF(td_->opInfo.get_G() == 0, OP_LOGW(context_, "g is 0"), return ge::GRAPH_PARAM_INVALID);
        td_->opInfo.set_N2(dimN2);
        td_->opInfo.set_S1(queryShape.GetDim(DIM_0));
        td_->opInfo.set_S2(keyShape.GetDim(DIM_0));
        td_->opInfo.set_D(tempDimD);
    } else if (td_->opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND)) {
        auto actualSeqQlenTensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_LEN);
        auto actualSeqKvlenTensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_KV_LEN);
        OP_CHECK_IF((actualSeqQlenTensor == nullptr || actualSeqKvlenTensor == nullptr),
                   OP_LOGW(context_, "actualSeqQlenTensor or actualSeqKvlenTensor is nullptr."),
                   return ge::GRAPH_PARAM_INVALID);

        const size_t seqQShapeSize = static_cast<size_t>(actualSeqQlenTensor->GetShapeSize());
        const size_t kvSeqShapeSize = static_cast<size_t>(actualSeqKvlenTensor->GetShapeSize());
        OP_CHECK_IF((seqQShapeSize != kvSeqShapeSize),
                   OP_LOGW(context_, "actualSeqQlenTensor shapeSize is not equal actualSeqKvlenTensor."),
                   return ge::GRAPH_PARAM_INVALID);

        std::vector<int64_t> actualSeqQlen;
        std::vector<int64_t> actualSeqKvlen;
        const int64_t *qValue = actualSeqQlenTensor->GetData<int64_t>();
        const int64_t *kvValue = actualSeqKvlenTensor->GetData<int64_t>();
        OP_CHECK_IF((qValue == nullptr || kvValue == nullptr),
                   OP_LOGE(context_, "qValue or kvValue is nullptr."), return ge::GRAPH_FAILED);
        int64_t tempN2 = keyShape.GetDim(DIM_1);
        for (size_t i = 0; i < seqQShapeSize; i++) {
            int64_t qSeqLen = (i == 0 ? qValue[i] : (qValue[i] - qValue[i - 1]));
            int64_t kvSeqLen = (i == 0 ? kvValue[i] : (kvValue[i] - kvValue[i - 1]));
            tmpData_.actualSeqQlen.push_back(qSeqLen);
            tmpData_.actualSeqKvlen.push_back(kvSeqLen);
            int64_t s1s2Product = tmpData_.actualSeqQlen[i] * tmpData_.actualSeqKvlen[i];
            tmpData_.sumS1S2Product += s1s2Product;
            int64_t s1s2ProductAlign = s1s2Product;
            s1s2ProductAlign = CeilCommon(tmpData_.actualSeqQlen[i], BASE_LEN_64) *
                               CeilCommon(tmpData_.actualSeqKvlen[i], BASE_LEN_64);
            // 将s1*s2的权重，按照B*N2的大小进行摊开存储，摊开顺序是先N2再B
            for (int64_t n2 = 0; n2 < tempN2; n2++) {
                tmpData_.s1s2Weight.push_back(s1s2ProductAlign);
            }
            if (qSeqLen == 0 || kvSeqLen == 0) {
                tndEmptyTensorFlag = true;
                OP_LOGI(context_, "TND EmptyInput detected, tndEmptyTensorFlag is set to True");
            }
        }

        tmpData_.t1 = queryShape.GetDim(DIM_0);
        tmpData_.t2 = keyShape.GetDim(DIM_0);
        uint64_t tailZeroCount = 0;
        for (auto i = seqQShapeSize - 1; i >= 1; --i) {
            if (tmpData_.actualSeqQlen[i] <= 0 && tmpData_.actualSeqKvlen[i] <= 0) {
                ++tailZeroCount;
            } else {
                break;
            }
        }
        auto realBSize = seqQShapeSize - tailZeroCount;
        td_->opInfo.set_B(realBSize);

        // query [t1, n1, d]   kv [t2, n2, d]   dy [t1, n1, d]
        OP_CHECK_IF(keyShape.GetDim(DIM_1) == 0, OP_LOGW(context_, "dim 1 of key is 0."),
                   return ge::GRAPH_PARAM_INVALID);
        td_->opInfo.set_G(queryShape.GetDim(DIM_1) / keyShape.GetDim(DIM_1));
        OP_CHECK_IF(td_->opInfo.get_G() == 0, OP_LOGW(context_, "g is 0"), return ge::GRAPH_PARAM_INVALID);
        td_->opInfo.set_N2(keyShape.GetDim(DIM_1));
        td_->opInfo.set_S1(*std::max_element(tmpData_.actualSeqQlen.begin(), tmpData_.actualSeqQlen.end()));
        td_->opInfo.set_S2(*std::max_element(tmpData_.actualSeqKvlen.begin(), tmpData_.actualSeqKvlen.end()));
        td_->opInfo.set_D(queryShape.GetDim(DIM_2));
    } else {
        OP_LOGW(context_, "inputLayout is invalid");
        return ge::GRAPH_PARAM_INVALID;
    }
    td_->opInfo.set_unpadEmptyInput(static_cast<uint32_t>(tndEmptyTensorFlag));
    tmpData_.b = td_->opInfo.get_B();
    tmpData_.n2 = td_->opInfo.get_N2();
    tmpData_.g = td_->opInfo.get_G();
    tmpData_.d = td_->opInfo.get_D();
    tmpData_.s1 = td_->opInfo.get_S1();
    tmpData_.s2 = td_->opInfo.get_S2();

    auto ret = CheckDtypeValid(context_);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGW(context_, "dtype is invalid."), return ret);

    return td_->opInfo.get_layout() == static_cast<uint32_t>(InputLayout::TND) ?
               CheckTndShapeValid(context_, tmpData_.t1, dimN1, tmpData_.d) :
               CheckShapeValid(context_, tmpData_.b, dimN1, tmpData_.s1, tmpData_.d);
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::GetBaseShapeInfo()
{
    OP_CHECK_IF(((context_->GetInputShape(QUERY) == nullptr) || (context_->GetInputShape(KEY) == nullptr)),
               OP_LOGE(context_, "InputShape of query or key is nullptr."),
               return ge::GRAPH_FAILED);
    const gert::Shape &queryShape = context_->GetInputShape(QUERY)->GetStorageShape();
    const gert::Shape &keyShape = context_->GetInputShape(KEY)->GetStorageShape();
    int64_t dimN1 = *context_->GetAttrs()->GetAttrPointer<int>(HEAD_NUM); // headNum is as same as N1
    auto shapeStatus = SetBaseInfo(queryShape, keyShape, dimN1);
    if (shapeStatus != ge::GRAPH_SUCCESS) {
        return shapeStatus;
    }

    const gert::StorageShape *qShape = context_->GetInputShape(QUERY);
    const gert::StorageShape *kShape = context_->GetInputShape(KEY);
    const gert::StorageShape *vShape = context_->GetInputShape(VALUE);
    const gert::StorageShape *dyShape = context_->GetInputShape(DY);

    auto ret = IsSameShapeButValueDLeEqD(qShape, dyShape);
    OP_CHECK_IF(!ret, OP_LOGE(context_, "FAG different shapequeryShape and dyShape"),
               return ge::GRAPH_FAILED);
    ret = IsSameShapeButValueDLeEqD(kShape, vShape);
    OP_CHECK_IF(!ret, OP_LOGE(context_, "FAG different shape keyShape and valueShape"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::ProcessDropInfo()
{
    auto dropMask = context_->GetOptionalInputDesc(DROP_MASK);
    OP_CHECK_IF(dropMask == nullptr,
               OP_LOGE(context_, "FAG keepProb [%f], dropMask can not be nullptr",
                                           td_->opInfo.get_keepProb()),
               return ge::GRAPH_FAILED);
    auto dropMaskType = dropMask->GetDataType();
    OP_CHECK_IF(dropMaskType != ge::DT_UINT8,
               OP_LOGE(context_, "FAG invalid dropMask dtype[%s], only support uint8.",
                                           ge::TypeUtils::DataTypeToSerialString(dropMaskType).c_str()),
               return ge::GRAPH_FAILED);

    auto dropMaskShape = context_->GetOptionalInputShape(DROP_MASK);
    OP_CHECK_IF(dropMaskShape == nullptr,
               OP_LOGE(context_, "FAG keepProb [%f], dropMaskShape can not be nullptr",
                                           td_->opInfo.get_keepProb()),
               return ge::GRAPH_FAILED);
    int64_t dropMaskDim = dropMaskShape->GetStorageShape().GetDimNum();
    int64_t dropMaskShapeSize = 1;
    for (int64_t i = 0; i < dropMaskDim; ++i) {
        int64_t dimValue = dropMaskShape->GetStorageShape().GetDim(i);
        dropMaskShapeSize *= dimValue;
    }

    auto shapeSize = AlignUp(tmpData_.dropMaskSize, static_cast<int64_t>(BIT_NUM)) / BIT_NUM;
    if (dropMaskShapeSize < shapeSize) {
        OP_LOGE(context_, "FAG Input dropMask shapeSize is invalid, it should not be less than %ld, but got %ld",
                  shapeSize, dropMaskShapeSize);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::SetBmm1TilingData(uint32_t sOut, uint32_t sFla,
                                                                        uint32_t l1SizeRemain)
{
    /*
    BSH/BSND:
    A：[B, S1, N1, D] + B: [B, S2, N2, D] = C[S1, S2]
    A\B In Gm, C In UB
    For A: m = S1, k = N1 * D
    For B: n = S2, K = N2 * D
    For C：don't care

    SBH:
    A: [S1, B, N1, D] + B: [S2, B, N2, D] = C[S1, S2]
    A\B In Gm, C In UB
    For A: m = S1, k = B * N1 * D
    For B: n = S2, k = B * N2 * D
    For C: don't case

    BNSD:
    m = S1, k = D, n = S2
    */
    uint32_t layout = td_->opInfo.get_layout();
    bool isBsh = layout == static_cast<uint32_t>(InputLayout::BSH);
    bool isSbh = layout == static_cast<uint32_t>(InputLayout::SBH);
    bool isBsnd = layout == static_cast<uint32_t>(InputLayout::BSND);
    bool isTnd = layout == static_cast<uint32_t>(InputLayout::TND);

    int64_t ka = td_->opInfo.get_D();
    int64_t kb = td_->opInfo.get_D();
    int64_t n = td_->opInfo.get_S2();
    if (isBsh || isBsnd || isTnd) {
        ka = td_->opInfo.get_G() * td_->opInfo.get_N2() * td_->opInfo.get_D();
        kb = td_->opInfo.get_N2() * td_->opInfo.get_D();
    } else if (isSbh) {
        ka = td_->opInfo.get_B() * td_->opInfo.get_G() * td_->opInfo.get_N2() * td_->opInfo.get_D();
        kb = td_->opInfo.get_B() * td_->opInfo.get_N2() * td_->opInfo.get_D();
    }

    auto inputType = matmul_tiling::DataType::DT_FLOAT16;
    auto outputType = matmul_tiling::DataType::DT_FLOAT16;
    auto valueType = static_cast<uint32_t>(context_->GetInputDesc(VALUE)->GetDataType());
    if (valueType == ge::DT_BF16) {
        inputType = matmul_tiling::DataType::DT_BF16;
        outputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (valueType == ge::DT_FLOAT) {
        inputType = matmul_tiling::DataType::DT_FLOAT;
        outputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (valueType == ge::DT_FLOAT16 && tmpData_.branch == BRANCH_FP16_HIGH_PRECISION) {
        outputType = matmul_tiling::DataType::DT_FLOAT;
    }

    matmul_tiling::MatmulApiTiling mm1;
    mm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputType, false);
    mm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputType, true);
    mm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, outputType);
    mm1.SetShape(singleM, singleN, td_->opInfo.get_D());
    mm1.SetOrgShape(singleM, n, ka, kb);
    mm1.SetBias(false);
    mm1.SetBufferSpace(l1SizeRemain, aicoreParams_.l0cSize);
    mm1.SetFixSplit(sOut, sFla, -1);
    if (tmpData_.isL1CustomEnable) {
        mm1.SetFixSplit(128, 128, -1);
    }

    OP_CHECK_IF((mm1.GetTiling(td_->mm1TilingData) != 0),
               OP_LOGW(context_, "FlashAttentionScoreGradTilingS1s2Bn2 mm1 GetTiling Failed."),
               return ge::GRAPH_PARAM_INVALID);
    td_->mm1TilingData.shareMode = 0;
    td_->mm1TilingData.shareL1Size = l1SizeRemain;
    td_->mm1TilingData.shareL0CSize = aicoreParams_.l0cSize;
    td_->mm1TilingData.iterateOrder = 1;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::SetBmm31TilingData(uint32_t l1SizeRemain)
{
    /*
    BSH/BSND:
    A: [S1, S2] + B: [B, S2, N2, D] = C[B, S1, N1, D]
    A In UB, B\C In Gm
    For A: don't care (m = S1, k = S2)
    For B: n = N2 * D, k = S2
    For C: m = S1, n = N1 * D

    SBH:
    A: [S1, S2] + B: [S2, B, N2, D] = C[S1, B, N1, D]
    A In UB, B\C In Gm
    For A: don't care(m = S1, k = S2)
    For B: n = B * N2 * D, k = S2
    For C: m = S1, n = B * N1 * D

    BNSD
    m = S1, k = S2, n = D
    */
    uint32_t layout = td_->opInfo.get_layout();
    bool isBsh = layout == static_cast<uint32_t>(InputLayout::BSH);
    bool isSbh = layout == static_cast<uint32_t>(InputLayout::SBH);
    bool isBsnd = layout == static_cast<uint32_t>(InputLayout::BSND);
    bool isTnd = layout == static_cast<uint32_t>(InputLayout::TND);

    int64_t ka = td_->opInfo.get_S2();
    int64_t kb = td_->opInfo.get_S2();
    int64_t m = td_->opInfo.get_S1();
    int64_t n = td_->opInfo.get_D();
    if (isBsh || isBsnd || isTnd) {
        // not support different N
        n = td_->opInfo.get_D() * td_->opInfo.get_N2();
    } else if (isSbh) {
        // not support differnet N
        n = td_->opInfo.get_D() * td_->opInfo.get_N2() * td_->opInfo.get_B();
    }

    auto inputAType = matmul_tiling::DataType::DT_FLOAT16;
    auto inputBType = matmul_tiling::DataType::DT_FLOAT16;
    auto outputType = matmul_tiling::DataType::DT_FLOAT16;
    auto keyType = static_cast<uint32_t>(context_->GetInputDesc(KEY)->GetDataType());
    if (keyType == ge::DT_BF16) {
        inputAType = matmul_tiling::DataType::DT_BF16;
        inputBType = matmul_tiling::DataType::DT_BF16;
        outputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (keyType == ge::DT_FLOAT) {
        inputAType = matmul_tiling::DataType::DT_FLOAT;
        inputBType = matmul_tiling::DataType::DT_FLOAT;
        outputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (keyType == ge::DT_FLOAT16 && tmpData_.branch == BRANCH_FP16_HIGH_PRECISION) {
        inputAType = matmul_tiling::DataType::DT_FLOAT16;
        inputBType = matmul_tiling::DataType::DT_FLOAT16;
        outputType = matmul_tiling::DataType::DT_FLOAT;
    }

    matmul_tiling::MatmulApiTiling mm31;
    mm31.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, false);
    mm31.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputBType, false);
    mm31.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, outputType);
    mm31.SetShape(singleM, td_->opInfo.get_D(), singleN);
    mm31.SetOrgShape(m, n, ka, kb);
    mm31.SetBias(false);
    if (tmpData_.isL1CustomEnable) {
        mm31.SetFixSplit(128, -1, -1);
    }
    mm31.SetBufferSpace(l1SizeRemain, aicoreParams_.l0cSize);
    OP_CHECK_IF((mm31.GetTiling(td_->mm31TilingData) != 0),
               OP_LOGW(context_, "FlashAttentionScoreGradTilingS1s2Bn2 mm31 GetTiling Failed."),
               return ge::GRAPH_PARAM_INVALID);
    td_->mm31TilingData.shareMode = 0;
    td_->mm31TilingData.shareL1Size = l1SizeRemain;
    td_->mm31TilingData.shareL0CSize = aicoreParams_.l0cSize;
    td_->mm31TilingData.iterateOrder = 0;

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus FlashAttentionScoreGradTilingS1s2Bn2::SetBmm4TilingData(uint32_t l1SizeRemain)
{
    /*
    BSH/BSND
    A:[S1, S2] + B: [B,S1,N1,D] = C[B,S2,N2,D]
    A In UB， B\C In GM
    A: don't care (m = S2, k = S1)
    B: n = N1 * D, k = S1
    C: m = S2, n = N2 * D

    SBH
    A:[S1, S2] + B: [S1, B, N1, D] = C: [S2, B, N2, D]
    A In UB， B\C In GM
    A: don't care (m= S2, k= S1)
    B: n = B * N1 * D, k = S1
    C: m = S2, n = B * N2 * D

    BNSD:
    m = S2, n = D, k = S1
    */

    uint32_t layout = td_->opInfo.get_layout();
    bool isBsh = layout == static_cast<uint32_t>(InputLayout::BSH);
    bool isSbh = layout == static_cast<uint32_t>(InputLayout::SBH);
    bool isBsnd = layout == static_cast<uint32_t>(InputLayout::BSND);
    bool isTnd = layout == static_cast<uint32_t>(InputLayout::TND);

    int64_t ka = td_->opInfo.get_S1();
    int64_t kb = td_->opInfo.get_S1();
    int64_t m = td_->opInfo.get_S2();
    int64_t n = td_->opInfo.get_D();
    if (isBsh || isBsnd || isTnd) {
        // not support different N
        n = td_->opInfo.get_D() * td_->opInfo.get_N2() * td_->opInfo.get_G();
    } else if (isSbh) {
        // not support differnet N
        n = td_->opInfo.get_D() * td_->opInfo.get_N2() * td_->opInfo.get_G() * td_->opInfo.get_B();
    }

    auto inputAType = matmul_tiling::DataType::DT_FLOAT16;
    auto inputBType = matmul_tiling::DataType::DT_FLOAT16;
    auto outputType = matmul_tiling::DataType::DT_FLOAT16;
    auto keyType = static_cast<uint32_t>(context_->GetInputDesc(KEY)->GetDataType());
    if (keyType == ge::DT_BF16) {
        inputAType = matmul_tiling::DataType::DT_BF16;
        inputBType = matmul_tiling::DataType::DT_BF16;
        outputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (keyType == ge::DT_FLOAT) {
        inputAType = matmul_tiling::DataType::DT_FLOAT;
        inputBType = matmul_tiling::DataType::DT_FLOAT;
        outputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (keyType == ge::DT_FLOAT16 && tmpData_.branch == BRANCH_FP16_HIGH_PRECISION) {
        inputAType = matmul_tiling::DataType::DT_FLOAT16;
        inputBType = matmul_tiling::DataType::DT_FLOAT16;
        outputType = matmul_tiling::DataType::DT_FLOAT;
    }

    matmul_tiling::MatmulApiTiling mm4;
    mm4.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputAType, true);
    mm4.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputBType, false);
    mm4.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, outputType);
    mm4.SetShape(singleN, td_->opInfo.get_D(), singleM);
    mm4.SetOrgShape(m, n, ka, kb);
    mm4.SetBias(false);

    int64_t dAlign = (td_->opInfo.get_D() + FP16_BLOCK_NUMS - 1) / FP16_BLOCK_NUMS * FP16_BLOCK_NUMS;
    int64_t minBaseN = std::min(BASE_LEN_128, static_cast<uint32_t>(dAlign));
    if (tmpData_.isMM345NZOut) {
        minBaseN = std::min(BASE_LEN_256, static_cast<uint32_t>(dAlign));
        mm4.SetFixSplit(-1, minBaseN, -1);
    }

    if (needSetFixSplit) {
        int64_t s2Align = (td_->opInfo.get_S2() + FRACTAL_NUM - 1) / FRACTAL_NUM * FRACTAL_NUM;
        int64_t minBaseM = std::min(BASE_LEN_256, static_cast<uint32_t>(s2Align));
        mm4.SetFixSplit(minBaseM, minBaseN, -1);
    }
    if (tmpData_.isL1CustomEnable) {
        mm4.SetFixSplit(128, -1, -1);
    }

    mm4.SetBufferSpace(l1SizeRemain, aicoreParams_.l0cSize);
    OP_CHECK_IF((mm4.GetTiling(td_->mm4TilingData) != 0),
               OP_LOGW(context_, "FlashAttentionScoreGradTilingS1s2Bn2 mm4 GetTiling Failed."),
               return ge::GRAPH_PARAM_INVALID);
    td_->mm4TilingData.shareMode = 0;
    td_->mm4TilingData.shareL1Size = l1SizeRemain;
    td_->mm4TilingData.shareL0CSize = aicoreParams_.l0cSize;
    td_->mm4TilingData.iterateOrder = 0;

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(
    FlashAttentionScoreGrad, FlashAttentionScoreGradTilingS1s2Bn2,
    std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}),
    15000);
REGISTER_TILING_TEMPLATE_WITH_ARCH(
    FlashAttentionScoreGrad, FlashAttentionScoreGradTilingDeterministic,
    std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}),
    1000);

} // namespace optiling
