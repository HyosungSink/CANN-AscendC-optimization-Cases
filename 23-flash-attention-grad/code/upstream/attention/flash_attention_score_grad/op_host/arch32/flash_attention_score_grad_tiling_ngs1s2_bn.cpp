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
 * \file flash_attention_score_grad_tiling_ngs1s2_bn.cc
 * \brief
 */

#include "flash_attention_score_grad_tiling_common.h"
#include "tiling_base/tiling_type.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../../op_kernel/arch32/flash_attention_score_grad_tiling.h"
#include "../../op_kernel/arch32/flash_attention_score_grad_template_tiling_key.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {

constexpr int64_t BSH_SBH_DIM_NUM = 3;
constexpr int64_t BNSD_DIM_NUM = 4;
constexpr uint32_t DIM_COUNT_NUM_4 = 4;
constexpr int64_t ATTEN_MASK_TYPE_11SS_DIM_NUM = 2;
constexpr int64_t WORK_SPACE_BASE_CAL = 16 * 1024 * 1024;
constexpr int64_t FP32_BYTES_NUM = 4;
constexpr int64_t FP16_BYTES_NUM = 2;
constexpr int64_t PER_SUB_RANGE = 8;
constexpr int64_t C0_SIZE = 16;
constexpr int64_t VEC_REPEAT = 8;
constexpr int64_t NZ_S_MIN = 4;
constexpr uint32_t POST_NZ_COEX_NODE = 10;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t POST_NZ_RESERVED_N = 1;
constexpr uint32_t MM_MAX_STRIDE_LIMIT = 65535;
constexpr int64_t CV_RATIO = 4;
constexpr uint32_t WORKSPACE_ALIGN_SIZE = 512;
constexpr uint32_t POST_COEX_NODE = 3;
constexpr uint32_t BASE_LEN_256 = 256;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t MAX_KV_SEQLEN = 1536;
constexpr uint32_t D_SIZE_72 = 72;
constexpr uint32_t S_SIZE_60 = 60;
constexpr uint32_t S_SIZE_48 = 48;
constexpr uint32_t S_SIZE_64 = 64;
constexpr uint32_t S_SIZE_30 = 30;
constexpr uint32_t S_SIZE_32 = 32;

enum class DTemplateType {
    NonAligned = 0,
    NonAligned72 = 3,
    Aligned80 = 5,
    Aligned96 = 6,
    DTemplateBottom
};

enum class STemplateType {
    NonAligned = 0,
    Aligned48 = 1,
    Aligned64 = 2,
    NonAligned60 = 3,
    Aligned32 = 4,
    NonAligned30 = 5,
    DTemplateBottom
};

#define CHECK_ZERO(num)                                                                                                \
    do {                                                                                                               \
        if ((num) == 0) {                                                                                              \
            OP_LOGW("template 3.1 number[%s] is zero.", #num);                                                       \
            return ge::GRAPH_PARAM_INVALID;                                                                            \
        }                                                                                                              \
    } while (0)

/* 这里的基本块和矩阵乘法的基本块定义有所区别（这里表示单次搬运和vector计算一次使用的数据量,
 * 原来表示一次matmul计算用到的数据量），
 * 在当前切N轴的模板下，内层matmul单次计算量仍然是sQ * sKVAlign,
 * 不过我们搬运的数据量和vector的数据量会变大。 参考之前的经验，128 * 128 * 2 bytes的数据是比较合适的。 */
const int64_t BEST_BASIC_BLOCK_SIZE = 32768; // 64 * 128 * 4 == s1 * s2 * 4 B

struct TempParamsUngs1s2Bbn {
    uint32_t dataType = 0;
    uint32_t precisionMode = 0;
    uint32_t attenMaskCompressMode = 0;
    int64_t attenMaskS1Size = 0;
    int64_t attenMaskS2Size = 0;
    uint32_t layout = 0;
    int64_t ubSizeRemain = 0;
    bool mmPreIsNZOut = false;
    bool mmNextIsNZOut = false;
    int64_t b = 0;
    int64_t n = 0;
    int64_t sQ = 0;
    int64_t d = 0;
};

class FlashAttentionScoreGradUngs1s2BbnTiling : public TilingBaseClass {
public:
    FlashAttentionScoreGradTilingDataUngs1s2Bbn *td_ = context_->GetTilingData<FlashAttentionScoreGradTilingDataUngs1s2Bbn>();
    TempParamsUngs1s2Bbn basicParams;

    explicit FlashAttentionScoreGradUngs1s2BbnTiling(gert::TilingContext *context) : TilingBaseClass(context)
    {
        td_->reset();
    };

    ~FlashAttentionScoreGradUngs1s2BbnTiling() override = default;

    inline uint64_t AlignSize(const uint64_t n, const uint64_t alignSize) const
    {
        if (alignSize == 0ULL) {
            return 0;
        }
        return (n + alignSize - 1ULL) & (~(alignSize - 1ULL));
    }

    uint64_t GetTilingKey() const override
    {
        uint64_t tilingKey = 0;
        STemplateType s1TemplateType = STemplateType::NonAligned;
        STemplateType s2TemplateType = STemplateType::NonAligned;
        DTemplateType dTemplateType = DTemplateType::NonAligned;
        optiling::LayoutEnum layout = optiling::LayoutEnum::BSND;
        if (basicParams.layout == static_cast<uint32_t>(InputLayout::BNSD)) {
            layout = LayoutEnum::BNSD;
        } else if (basicParams.layout == static_cast<uint32_t>(InputLayout::SBH)) {
            layout = LayoutEnum::SBND;
        }

        auto dtype = DtypeEnum::FLOAT16;
        if (basicParams.dataType == ge::DT_FLOAT16) {
            if (basicParams.precisionMode == static_cast<uint32_t>(HIGH_PRECISION)) {
                dtype = DtypeEnum::FLOAT16_PRECISION;
            }
        } else if (basicParams.dataType == ge::DT_BF16) {
            dtype = DtypeEnum::BFLOAT16;
        } else {
            dtype = DtypeEnum::FLOAT32;
        }

        auto mmPreIsNZOut = basicParams.mmPreIsNZOut ? OptionEnum::ENABLE : OptionEnum::DISABLE;
        auto mmNextIsNZOut = basicParams.mmNextIsNZOut ? OptionEnum::ENABLE : OptionEnum::DISABLE;
        auto unique = OptionEnum::DISABLE;

        if (basicParams.layout == static_cast<uint32_t>(InputLayout::SBH) &&
            basicParams.b * basicParams.n * basicParams.d > static_cast<int64_t>(MM_MAX_STRIDE_LIMIT)) {
            tilingKey = GET_TPL_TILING_KEY(static_cast<uint8_t>(AxisEnum::NONE), static_cast<uint8_t>(AxisEnum::NONE), static_cast<uint8_t>(AxisEnum::N2), 0, static_cast<uint8_t>(dtype),
                static_cast<uint8_t>(layout), static_cast<uint8_t>(SparseEnum::ALL), static_cast<uint8_t>(MatmulConfig::NORMAL_CONFIG), static_cast<uint8_t>(mmPreIsNZOut),
                static_cast<uint8_t>(mmNextIsNZOut), static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                0, 0, 0, static_cast<uint8_t>(s1TemplateType), static_cast<uint8_t>(s2TemplateType), static_cast<uint8_t>(dTemplateType), 0, 0);
            OP_LOGD(context_, "Ungs1s2Bbn tilingKey is %lu.", tilingKey);
            return tilingKey;
        } else {
            // SBH: B*H <= 65535
            if (unique == OptionEnum::DISABLE && mmPreIsNZOut == OptionEnum::ENABLE && mmNextIsNZOut == OptionEnum::DISABLE
            && dtype == DtypeEnum::BFLOAT16 && layout == optiling::LayoutEnum::BSND) {
                if (basicParams.d == static_cast<int64_t>(D_SIZE_72) &&
                    basicParams.sQ == static_cast<int64_t>(S_SIZE_60)) {
                    dTemplateType = DTemplateType::NonAligned72;
                    s1TemplateType = STemplateType::NonAligned60;
                    s2TemplateType = STemplateType::NonAligned60;
                } else if (basicParams.d == static_cast<int64_t>(D_SIZE_72) &&
                           basicParams.sQ == static_cast<int64_t>(S_SIZE_48)) {
                    dTemplateType = DTemplateType::NonAligned72;
                    s1TemplateType = STemplateType::Aligned48;
                    s2TemplateType = STemplateType::Aligned48;
                } else if (basicParams.d == static_cast<int64_t>(D_SIZE_72) &&
                           basicParams.sQ == static_cast<int64_t>(S_SIZE_64)) {
                    dTemplateType = DTemplateType::NonAligned72;
                    s1TemplateType = STemplateType::Aligned64;
                    s2TemplateType = STemplateType::Aligned64;
                } else if (basicParams.d == static_cast<int64_t>(D_SIZE_72) &&
                           basicParams.sQ == static_cast<int64_t>(S_SIZE_30)) {
                    dTemplateType = DTemplateType::NonAligned72;
                    s1TemplateType = STemplateType::NonAligned30;
                    s2TemplateType = STemplateType::NonAligned30;
                } else if (basicParams.d == static_cast<int64_t>(D_SIZE_72) &&
                           basicParams.sQ == static_cast<int64_t>(S_SIZE_32)) {
                    dTemplateType = DTemplateType::NonAligned72;
                    s1TemplateType = STemplateType::Aligned32;
                    s2TemplateType = STemplateType::Aligned32;
                }
            }
            tilingKey = GET_TPL_TILING_KEY(static_cast<uint8_t>(AxisEnum::NONE), static_cast<uint8_t>(AxisEnum::NONE), static_cast<uint8_t>(AxisEnum::N2), 0, static_cast<uint8_t>(dtype),
                static_cast<uint8_t>(layout), static_cast<uint8_t>(SparseEnum::ALL), static_cast<uint8_t>(unique), static_cast<uint8_t>(mmPreIsNZOut),
                static_cast<uint8_t>(mmNextIsNZOut), static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                0, 0, 0, static_cast<uint8_t>(s1TemplateType), static_cast<uint8_t>(s2TemplateType), static_cast<uint8_t>(dTemplateType), 0, 0);
            OP_LOGD(context_, "Ungs1s2Bbn tilingKey is %lu.", tilingKey);
            return tilingKey;
        }
    }

    bool IsCapable() override
    {   
        auto sinkShape = context_->GetOptionalInputShape(SINK_IN);
        if (sinkShape != nullptr && sinkShape->GetStorageShape().GetDimNum() == 1 ) {
            return false;
        }
        const char *tndSoftmaxIn = context_->GetAttrs()->GetAttrNum() > static_cast<size_t>(TND_SOFTMAX_IN) ? context_->GetAttrs()->GetAttrPointer<char>(TND_SOFTMAX_IN) : "";
        if (strcmp(tndSoftmaxIn, "") != 0) {
            return false;
        }

        if (basicParams.dataType == ge::DT_FLOAT){
            OP_LOGD(context_, "Ungs1s2Bbn is not support fp32");
            return false;
        }

        OP_CHECK_IF(context_->GetAttrs() == nullptr,
                   OP_LOGW(context_, "GetAttrs is nullptr."),
                   return false);

        if (context_->GetAttrs()->GetAttrNum() > static_cast<size_t>(PSETYPE)) {
            auto psetype = *context_->GetAttrs()->GetAttrPointer<int>(PSETYPE); // 8
            if (psetype != 1) { // 不支持非默认的psetype
                return false;
            }
        }

        OP_LOGD(context_, "Ungs1s2Bbn check is capable.");
        int64_t gSqSkvSize = td_->opInfo.get_g() * td_->opInfo.get_sQ() * td_->opInfo.get_sKVAlignSizeVec();
        /* 计算g * sQ * sKVAlign的大小是否小于当前 */
        if (gSqSkvSize == 0 || td_->opInfo.get_g() != 1 || td_->opInfo.get_sKVAlignSizeVec() > MAX_KV_SEQLEN) {
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

        if (gSqSkvSize <= BEST_BASIC_BLOCK_SIZE) {
            OP_LOGD(context_, "Ungs1s2Bbn isCapable check ok.");
            return true;
        }

        OP_LOGD(context_, "Ungs1s2Bbn isCapable check false.");
        return false;
    }

    ge::graphStatus GetPlatformInfo() override
    {
        OP_LOGD(context_, "Get platform informations.");
        auto platformInfoPtr = context_->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            auto compileInfoPtr =
                reinterpret_cast<const FlashAttentionScoreGradCompileInfo *>(context_->GetCompileInfo());
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

        OP_CHECK_IF(aicoreParams_.ubSize <= 0,
                   OP_LOGE(context_, "ubSize is invalid."),
                   return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus MakeAttenMaskShapeDims()
    {
        auto attenMaskShape = context_->GetOptionalInputShape(ATTEN_MASK);
        if (attenMaskShape != nullptr && attenMaskShape->GetStorageShape().GetShapeSize() != 0) {
            td_->opInfo.set_hasAttenMask(1);
            auto storageShape = attenMaskShape->GetStorageShape();
            auto maskShapeDims = storageShape.GetDimNum();
            if (maskShapeDims == ATTEN_MASK_TYPE_11SS_DIM_NUM) {
                td_->opInfo.set_attenMaskShapeType(ATTEN_MASK_SHAPE_TYPE_SS);
            } else if (maskShapeDims == DIM_COUNT_NUM_4) {
                auto dim0 = storageShape.GetDim(DIM_0);
                auto dim1 = storageShape.GetDim(DIM_1);
                if (dim0 == 1 && dim1 == 1) {
                    td_->opInfo.set_attenMaskShapeType(ATTEN_MASK_SHAPE_TYPE_SS);
                } else if (dim0 == td_->opInfo.get_b() && dim1 == 1) {
                    td_->opInfo.set_attenMaskShapeType(ATTEN_MASK_SHAPE_TYPE_B1SS);
                } else if (dim0 == td_->opInfo.get_b() && dim1 == td_->opInfo.get_n() * td_->opInfo.get_g()) {
                    td_->opInfo.set_attenMaskShapeType(ATTEN_MASK_SHAPE_TYPE_BNSS);
                } else {
                    return ge::GRAPH_PARAM_INVALID;
                }
                OP_LOGD(context_, "Ungs1s2Bbn get attenmask shape dims success.");
            } else {
                return ge::GRAPH_PARAM_INVALID;
            }
            basicParams.attenMaskS2Size = storageShape.GetDim(maskShapeDims - LAST_AXIS_IDX);
            basicParams.attenMaskS1Size = storageShape.GetDim(maskShapeDims - SEC_LAST_AXIS_IDX);
            int sparseMode = NO_MASK;
            if (context_->GetAttrs()->GetAttrNum() > static_cast<size_t>(SPARSE_MODE)) {
                sparseMode = *context_->GetAttrs()->GetAttrPointer<int>(SPARSE_MODE); // 7
            }
            if (sparseMode == LEFT_UP_CAUSAL) {
                basicParams.attenMaskCompressMode = LEFT_UP_CAUSAL_MODE;
            } else if (sparseMode == RIGHT_DOWN_CAUSAL) {
                basicParams.attenMaskCompressMode = RIGHT_DOWN_CAUSAL_MODE;
            } else if (sparseMode > RIGHT_DOWN_CAUSAL) {
                return ge::GRAPH_PARAM_INVALID;
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetAttenMaskTilingInfo()
    {
        OP_LOGD(context_, "Ungs1s2Bbn set attenmask tiling info.");
        basicParams.attenMaskCompressMode = NO_COMPRESS_MODE;
        auto attenMaskDesc = context_->GetOptionalInputDesc(ATTEN_MASK);
        if (attenMaskDesc != nullptr) {
            auto attenMaskDtype = attenMaskDesc->GetDataType();
            OP_CHECK_IF(
                (attenMaskDtype != ge::DT_BOOL && attenMaskDtype != ge::DT_UINT8), OP_LOGW(context_, "AttenMaskDtype(%d) is not bool or uint8.", attenMaskDtype),
                        return ge::GRAPH_PARAM_INVALID);
        }
        auto ret = MakeAttenMaskShapeDims();
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                    OP_LOGW(context_, "MakeAttenMaskShapeDims fail."),
                    return ge::GRAPH_PARAM_INVALID);
        td_->opInfo.set_attenMaskS2Size(basicParams.attenMaskS2Size);
        td_->opInfo.set_attenMaskCompressMode(basicParams.attenMaskCompressMode);
        OP_LOGD(context_, "Ungs1s2Bbn set attenmask tiling info success.");
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetAttrsInfo()
    {
        td_->opInfo.set_scaleValue(*context_->GetAttrs()->GetAttrPointer<float>(SCALE_VALUE));
        td_->opInfo.set_keepProb(*context_->GetAttrs()->GetAttrPointer<float>(KEEP_PROB));
        OP_CHECK_IF((td_->opInfo.get_keepProb() <= 0 || td_->opInfo.get_keepProb() > 1),
                    OP_LOGW(context_, "keepProb is illegal."),
                    return ge::GRAPH_PARAM_INVALID);
        td_->opInfo.set_preTokens(*context_->GetAttrs()->GetAttrPointer<uint32_t>(PRE_TOKENS));
        td_->opInfo.set_nextTokens(*context_->GetAttrs()->GetAttrPointer<uint32_t>(NEXT_TOKENS));
        td_->opInfo.set_headNum(*context_->GetAttrs()->GetAttrPointer<uint32_t>(HEAD_NUM));
        OP_CHECK_IF(td_->opInfo.get_headNum() == 0, OP_LOGW(context_, "headNum is 0."),
                   return ge::GRAPH_PARAM_INVALID);
        const char *inputLayout = context_->GetAttrs()->GetAttrPointer<char>(INPUT_LAYOUT);
        if (strcmp(inputLayout, BSH_STR) == 0) {
            td_->opInfo.set_inputLayout(static_cast<uint32_t>(InputLayout::BSH));
        } else if (strcmp(inputLayout, SBH_STR) == 0) {
            td_->opInfo.set_inputLayout(static_cast<uint32_t>(InputLayout::SBH));
        } else if (strcmp(inputLayout, BNSD_STR) == 0) {
            td_->opInfo.set_inputLayout(static_cast<uint32_t>(InputLayout::BNSD));
        } else if (strcmp(inputLayout, BSND_STR) == 0) {
            td_->opInfo.set_inputLayout(static_cast<uint32_t>(InputLayout::BSND));
        } else {
            return ge::GRAPH_PARAM_INVALID;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetShapeInfo()
    {
        OP_CHECK_IF(((context_->GetInputShape(QUERY) == nullptr) || (context_->GetInputShape(KEY) == nullptr)),
                    OP_LOGE(context_, "InputShape of query or key is nullptr."),
                    return ge::GRAPH_FAILED);
        const gert::Shape &queryShape = context_->GetInputShape(QUERY)->GetStorageShape();
        const gert::Shape &keyShape = context_->GetInputShape(KEY)->GetStorageShape();
        if (td_->opInfo.get_inputLayout() == static_cast<uint32_t>(InputLayout::BNSD) ||
            td_->opInfo.get_inputLayout() == static_cast<uint32_t>(InputLayout::BSND)) {
            OP_CHECK_IF((queryShape.GetDimNum() != BNSD_DIM_NUM || keyShape.GetDimNum() != BNSD_DIM_NUM),
                      OP_LOGW(context_, "the dim of query or key is not 4."),
                      return ge::GRAPH_PARAM_INVALID);
            OP_LOGD(context_, "Ungs1s2Bbn get input dim success.");
            size_t layoutIdx = static_cast<size_t>(td_->opInfo.get_inputLayout());
            td_->opInfo.set_b(queryShape.GetDim(LAYOUT_TO_AXIS[layoutIdx][B]));
            td_->opInfo.set_sQ(queryShape.GetDim(LAYOUT_TO_AXIS[layoutIdx][S]));
            td_->opInfo.set_sKV(keyShape.GetDim(LAYOUT_TO_AXIS[layoutIdx][S]));
            td_->opInfo.set_d(queryShape.GetDim(LAYOUT_TO_AXIS[layoutIdx][AXIS4_D]));
            td_->opInfo.set_hQ(queryShape.GetDim(LAYOUT_TO_AXIS[layoutIdx][AXIS4_N]) * td_->opInfo.get_d());
            td_->opInfo.set_hKV(keyShape.GetDim(LAYOUT_TO_AXIS[layoutIdx][AXIS4_N]) * td_->opInfo.get_d());
            CHECK_ZERO(td_->opInfo.get_hKV());
            td_->opInfo.set_g(td_->opInfo.get_hQ() / td_->opInfo.get_hKV());
            CHECK_ZERO(td_->opInfo.get_g());
            td_->opInfo.set_n(td_->opInfo.get_headNum() / td_->opInfo.get_g());
        } else {
            OP_CHECK_IF((queryShape.GetDimNum() != BSH_SBH_DIM_NUM || keyShape.GetDimNum() != BSH_SBH_DIM_NUM),
                      OP_LOGW(context_, "the dim of query or key is not 3."),
                      return ge::GRAPH_PARAM_INVALID);
            OP_LOGD(context_, "Ungs1s2Bbn get input dim success.");
            size_t layoutIdx = static_cast<size_t>(td_->opInfo.get_inputLayout());
            td_->opInfo.set_b(queryShape.GetDim(LAYOUT_TO_AXIS[layoutIdx][B]));
            td_->opInfo.set_sQ(queryShape.GetDim(LAYOUT_TO_AXIS[layoutIdx][S]));
            td_->opInfo.set_sKV(keyShape.GetDim(LAYOUT_TO_AXIS[layoutIdx][S]));
            td_->opInfo.set_hQ(queryShape.GetDim(LAYOUT_TO_AXIS[layoutIdx][H]));
            td_->opInfo.set_hKV(keyShape.GetDim(LAYOUT_TO_AXIS[layoutIdx][H]));

            CHECK_ZERO(td_->opInfo.get_hKV());
            td_->opInfo.set_g(td_->opInfo.get_hQ() / td_->opInfo.get_hKV());
            CHECK_ZERO(td_->opInfo.get_g());
            td_->opInfo.set_n(td_->opInfo.get_headNum() / td_->opInfo.get_g());
            CHECK_ZERO(td_->opInfo.get_n());
            td_->opInfo.set_d(td_->opInfo.get_hKV() / td_->opInfo.get_n());
        }

        basicParams.b = td_->opInfo.get_b();
        basicParams.n = td_->opInfo.get_n() * td_->opInfo.get_g();
        basicParams.sQ = td_->opInfo.get_sQ();
        basicParams.d = td_->opInfo.get_d();
        return CheckShapeValid(context_, basicParams.b, basicParams.n, td_->opInfo.get_sQ(), basicParams.d);
    }

    ge::graphStatus GetDataTypeInfo()
    {
        auto ret = CheckDtypeValid(context_);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                   OP_LOGW(context_, "dtype is invalid."),
                   return ret);
        td_->opInfo.set_inputDType(static_cast<uint32_t>(context_->GetInputDesc(QUERY)->GetDataType()));
        basicParams.dataType = td_->opInfo.get_inputDType();

        int64_t inputDTypeSize =
            static_cast<int64_t>(GetSizeByDataType(static_cast<ge::DataType>(td_->opInfo.get_inputDType())));
        OP_CHECK_IF(inputDTypeSize >= ge::kDataTypeSizeBitOffset,
                   OP_LOGW(context_, "input data dtype size is invalid."),
                   return ge::GRAPH_PARAM_INVALID);

        OP_LOGD(context_, "Ungs1s2Bbn get inputDTypeSize success.");
        // bf16 场景下使用fp32(4 bytes)来进行vector的数据计算。
        uint32_t vecCalcDTypeSize = 0;
        if (td_->opInfo.get_inputDType() == ge::DT_BF16 || basicParams.precisionMode == HIGH_PRECISION) {
            vecCalcDTypeSize = (static_cast<uint64_t>(inputDTypeSize)) << 1;
        } else {
            vecCalcDTypeSize = static_cast<uint32_t>(inputDTypeSize);
        }
        CHECK_ZERO(static_cast<int32_t>(vecCalcDTypeSize));
        CHECK_ZERO(inputDTypeSize);
        td_->opInfo.set_vecCalcDTypeSize(vecCalcDTypeSize);
        td_->opInfo.set_inputDTypeSize(inputDTypeSize);
        int64_t sKVAlignSize = Align(td_->opInfo.get_sKV() * inputDTypeSize);
        td_->opInfo.set_sKVAlignSize(sKVAlignSize);
        td_->opInfo.set_sKVAlign(sKVAlignSize / inputDTypeSize);
        td_->opInfo.set_sKVAlignSizeVec(Align(td_->opInfo.get_sKVAlign() * td_->opInfo.get_vecCalcDTypeSize()));
        td_->opInfo.set_sKVAlignVec(td_->opInfo.get_sKVAlignSizeVec() / vecCalcDTypeSize);
        td_->opInfo.set_originalDAlign(Align(td_->opInfo.get_d() * inputDTypeSize) / inputDTypeSize);
        td_->opInfo.set_sKVAlignByte(Align(td_->opInfo.get_sKV()));

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetPseInfo()
    {
        auto pseShape = context_->GetOptionalInputShape(PSE_SHIFT);
        if (pseShape != nullptr && pseShape->GetStorageShape().GetShapeSize() != 0) {
            // pse support: 0 -- BNS1S2; 1 -- BN1S2; 2 -- 1NS1S2
            auto storageShape = pseShape->GetStorageShape();
            auto pseShapeDims = storageShape.GetDimNum();
            if (pseShapeDims == DIM_4) {
                auto dim0 = storageShape.GetDim(DIM_0);
                auto dim1 = storageShape.GetDim(DIM_1);
                auto dim2 = storageShape.GetDim(DIM_2);
                auto dim3 = storageShape.GetDim(DIM_3);
                int64_t shapeN1 = td_->opInfo.get_n() * td_->opInfo.get_g();
                bool isBNS = (dim0 == td_->opInfo.get_b()) && (dim1 == shapeN1) && (dim3 == td_->opInfo.get_sKV());
                bool isBNSS = isBNS && (dim2 == td_->opInfo.get_sQ());
                bool isBN1S = isBNS && (dim2 == 1);
                bool is1NSS =
                    (dim0 == 1) && (dim1 == shapeN1) && (dim2 == td_->opInfo.get_sQ()) && (dim3 == td_->opInfo.get_sKV());
                td_->opInfo.set_pseSq(dim2);
                // 设置shape类型
                if (is1NSS) {
                    td_->opInfo.set_pseShapeType(PSE_SHAPE_TYPE_1NSS);
                } else if (isBN1S) {
                    td_->opInfo.set_pseShapeType(PSE_SHAPE_TYPE_BN1S);
                } else if (isBNSS) {
                    td_->opInfo.set_pseShapeType(PSE_SHAPE_TYPE_BNSS);
                } else {
                    return ge::GRAPH_PARAM_INVALID;
                }
            } else {
                return ge::GRAPH_PARAM_INVALID;
            }
        } else {
            td_->opInfo.set_pseSq(0);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        OP_CHECK_IF(context_ == nullptr,
                OP_LOGE(context_, "context is nullptr."),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->GetAttrs() == nullptr,
                OP_LOGE(context_, "GetAttrs is nullptr."),
                return ge::GRAPH_FAILED);
        OP_LOGD(context_, "Ungs1s2Bbn get shape attrsInfo.");

        // 1. 获取属性信息
        auto ret = GetAttrsInfo();
        OP_CHECK_IF((ret != ge::GRAPH_SUCCESS),
                OP_LOGW(context_, "get AttrsInfo fail."),
                return ret);
        OP_LOGD(context_, "Ungs1s2Bbn get input layout success.");

        basicParams.layout = td_->opInfo.get_inputLayout();

        td_->opInfo.set_precisionMode(HIGH_PRECISION);
        basicParams.precisionMode = td_->opInfo.get_precisionMode();

        // 2. 获取shape和轴信息
        ret = GetShapeInfo();
        OP_CHECK_IF((ret != ge::GRAPH_SUCCESS),
                OP_LOGW(context_, "get ShapeInfo fail."),
                return ret);

        // 3. 获取data type信息
        ret = GetDataTypeInfo();
        OP_CHECK_IF((ret != ge::GRAPH_SUCCESS),
                OP_LOGW(context_, "get DataTypeInfo fail."),
                return ret);
        uint32_t vecCalcDTypeSize = td_->opInfo.get_vecCalcDTypeSize();
        int64_t inputDTypeSize = td_->opInfo.get_inputDTypeSize();

        int64_t dMax = td_->opInfo.get_sKVAlign();
        if (dMax > td_->opInfo.get_d()) {
            dMax = td_->opInfo.get_d();
        }

        CHECK_ZERO(dMax);
        CHECK_ZERO(static_cast<int32_t>(vecCalcDTypeSize));
        int64_t sKvAlign = AlignSize(td_->opInfo.get_sKV(), 16);
        int64_t nIn = BEST_BASIC_BLOCK_SIZE / (td_->opInfo.get_g() * td_->opInfo.get_sQ() * sKvAlign * vecCalcDTypeSize);
        if (nIn > td_->opInfo.get_n()) {
            nIn = td_->opInfo.get_n();
        }
        CHECK_ZERO(nIn);
        CHECK_ZERO(inputDTypeSize);
        td_->singleCoreParams.set_nIn(nIn);
        td_->singleCoreParams.set_nInTail(td_->opInfo.get_n() % nIn);
        int64_t splitedDAlign = Align(dMax * inputDTypeSize) / inputDTypeSize;
        td_->singleCoreParams.set_splitedDAlign(splitedDAlign);
        int64_t dRange = CeilCommon(td_->opInfo.get_originalDAlign(), splitedDAlign);
        td_->singleCoreParams.set_dRange(dRange);

        /* 4. 获取其他输入shape信息 */
        ret = GetPseInfo();
        OP_CHECK_IF((ret != ge::GRAPH_SUCCESS),
                OP_LOGW(context_, "get PseInfo fail."),
                return ret);
        ret = SetAttenMaskTilingInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        return ge::GRAPH_SUCCESS;
    }

    uint32_t GetApiTmpBufferSize(int64_t nIn, int64_t sQ, int64_t sKVAlign)
    {
        // softmax和dropout对应的vector计算的shape是一样的
        auto shape = ge::Shape({nIn * td_->opInfo.get_g() * sQ, sKVAlign});

        uint32_t softmaxTmpSize = GetSoftMaxMinTmpSize(shape, td_->opInfo.get_vecCalcDTypeSize(), true);
        uint32_t dropoutTmpSize = GetDropOutMinTmpSize(shape, td_->opInfo.get_vecCalcDTypeSize(), true);

        return std::max(softmaxTmpSize, dropoutTmpSize);
    }

    bool CheckArgsLegal(int64_t nIn)
    {
        uint32_t inputDTypeSize = td_->opInfo.get_inputDTypeSize();
        int64_t sQ = td_->opInfo.get_sQ();
        int64_t sKVAlign = td_->opInfo.get_sKVAlign();
        int64_t dAlign = td_->singleCoreParams.get_splitedDAlign();
        int64_t nInTimesG = nIn * td_->opInfo.get_g();
        // 计算vecIn1和vecIn2的d轴需要限制，如果d轴大于128，这里仍然使用128计算，并做切分，否则
        // 在fp32场景下只能使用 32 * 32 的基本块；
        // 在fp16场景下只能用 64 * 64 的基本块
        // 在bf16场景下仍然可以用 64 * 128 的基本块
        // 由于d轴在softmaxgrad计算之后可以reduce掉，我们默认会对超过128的d轴做核内切分
        // 左移一位表示有两组数据用vectorInQueue1，分别是attention_in和dx
        int64_t vecInQue1SizeWithPse = nInTimesG * sQ;
        // vecInQue3Size是dropout mask、pse、attention mask的输入，dropout mask和attention mak一定是bool类型，
        // pse可能有三种类型，按最大shape和dtype来规划大小。
        // vecInque2Size 的size为pse的shape size * pse的数据类型size 对齐到pse的数据类型，pse和q\k\v 输入相同类型
        int64_t vecInQue2Size = nInTimesG * sQ * td_->opInfo.get_sKVAlignSize();

        /* Bf16下使用fp32进行vector计算，这一步已经在vecCalcDTypeSize的计算中区分。
         * 当Bf16和Fp16高精度模式下, 这两个临时buffer会被dx和attentionIn的输入的Cast复用,
         * 要取dAlign和sKVAlign的较大值作为size 注意: 在前面的计算中已经保证了这里最大的vecClc Size不会超过36K。 */
        int64_t vecClc1Size = nInTimesG * sQ * td_->opInfo.get_vecCalcDTypeSize();
        int64_t dAlignSize = dAlign * td_->opInfo.get_inputDTypeSize();
        // s 大于等于4时，mmNz才有收益
        basicParams.mmPreIsNZOut = td_->opInfo.get_sQ() >= 4 ? true : false;
        if (td_->opInfo.get_inputDType() == static_cast<uint32_t>(ge::DT_BF16) ||
            basicParams.precisionMode == HIGH_PRECISION) {
            /* 注意!: 如果bf16或者float16高精度模式下，左边计算图dropOut的输出和mul的输出需要借助vecInQue1去做
               Cast成bf16或者fp16, 所以需要用sKVAlign和dAlign取较大值再乘以sizeof(fp16)。 */
            /* 同时，在存在pse的情况下，pse会借用vecInQue1其中的一个去完成fp32的转换，所以vecInQue1SizeWithPse又需要用
               sKVAlignSizeVec和dAlign * sizeof(fp16)取较大值 */
            if (td_->opInfo.get_pseSq() == 1 || basicParams.mmPreIsNZOut) {
                vecInQue1SizeWithPse *= std::max(td_->opInfo.get_sKVAlignSizeVec(), dAlignSize);
            } else {
                vecInQue1SizeWithPse *= std::max(td_->opInfo.get_sKVAlignSize(), dAlignSize);
            }
            if (basicParams.mmPreIsNZOut) {
                vecInQue1SizeWithPse += nInTimesG * sKVAlign / C0_SIZE * VEC_REPEAT * sizeof(float);
            }
            vecClc1Size *= std::max(td_->opInfo.get_sKVAlignVec(), dAlign);
        } else {
            /* 不需要Cast，直接提供给attentionIn和Dx使用。 */
            vecInQue1SizeWithPse *= dAlignSize;
            vecClc1Size *= sKVAlign;
        }

        // 如果是nz，对于s2每一个分形需要多8个数据
        if(basicParams.mmPreIsNZOut) {
            vecClc1Size += nInTimesG * sKVAlign / 16 * 32;
        }
        int64_t vecClc2Size = vecClc1Size;
        td_->singleCoreParams.set_innerTmpBufSize(vecClc1Size);
        td_->singleCoreParams.set_vecQueIn1Size(vecInQue1SizeWithPse);

        // D如果切分，需要多一块S1*DAlign的buff用于cast
        int64_t vecCastSize = 0;
        if (dAlign < td_->opInfo.get_originalDAlign()) {
            vecCastSize = nInTimesG * sQ * dAlign * td_->opInfo.get_vecCalcDTypeSize();
        }
        td_->singleCoreParams.set_vecCastSize(vecCastSize);
        int64_t queBufferSizeUb = vecInQue1SizeWithPse * 2 + vecInQue2Size + vecClc1Size + vecClc2Size + vecCastSize;
        // softmaxGrad，softmax，dropout计算所需要的tmpSize
        uint32_t maxTmpBufferSize = GetApiTmpBufferSize(nIn, sQ, sKVAlign); // 0.5K
        uint32_t usedBufferSize = static_cast<uint32_t>(queBufferSizeUb) + maxTmpBufferSize + SOFTMAX_REMAIN_SIZE;

        int64_t bufferSizeL0c = std::max(sQ * dAlign * FP32_BYTES_NUM, sKVAlign * dAlign * FP32_BYTES_NUM);
        bufferSizeL0c = std::max(bufferSizeL0c, sQ * sKVAlign * FP32_BYTES_NUM);

        int64_t bufferSizeL0a = std::max(sQ, sKVAlign) * dAlign * inputDTypeSize;
        bufferSizeL0a = std::max(bufferSizeL0a, sQ * sKVAlign * static_cast<int64_t>(inputDTypeSize));

        basicParams.ubSizeRemain = static_cast<int64_t>(aicoreParams_.ubSize) - usedBufferSize;

        if (basicParams.ubSizeRemain >= 0 && bufferSizeL0c <= static_cast<int64_t>(aicoreParams_.l0cSize) &&
            bufferSizeL0a <= static_cast<int64_t>(aicoreParams_.l0aSize)) {
            td_->splitCoreParams.set_apiClcQueueSize(basicParams.ubSizeRemain + API_RSDV_BUFFER_SIZE);

            // 因为bmm的L1size限制问题，check nIn之后是否能放的下，放不下就走后面的模板
            uint64_t inputSize = td_->opInfo.get_inputDTypeSize();
            int64_t d = td_->opInfo.get_d();
            int64_t sKV = td_->opInfo.get_sKV();
            uint64_t dAlign16 = AlignSize(static_cast<uint64_t>(d), 16ULL);
            uint64_t sqAlign16 = AlignSize(static_cast<uint64_t>(sQ), 16ULL);
            uint64_t skvAlign16 = AlignSize(static_cast<uint64_t>(sKV), 16ULL);
            if ((nInTimesG * sqAlign16 + nIn * skvAlign16) * dAlign16 * inputSize <= aicoreParams_.l1Size &&
                (dAlign16 + skvAlign16) * nInTimesG * sqAlign16 * inputSize <= aicoreParams_.l1Size &&
                (sqAlign16 + dAlign16) * nInTimesG * skvAlign16 * inputSize <= aicoreParams_.l1Size) {
                return true;
            }
        }
        return false;
    }

    void DoPreTiling()
    {
        uint32_t dropoutIsDivisibleBy8 = 1;
        if (td_->opInfo.get_keepProb() < 1.0 && context_->GetOptionalInputShape(DROP_MASK) != nullptr &&
            context_->GetOptionalInputShape(DROP_MASK)->GetStorageShape().GetDimNum() != 0) {
            // 120KB FP16Tensor, 60KB U8Tensor, 8KB MaskTensor, 512B HelpTensor which less than UB(192KB).
            // singleUBProcessNum: UB最大处理FP16数据大小，需保证能被128整除.
            uint32_t castBufferLen = 60 * 1024;
            uint32_t outputBufferLen = 30 * 1024;
            uint32_t inputBufferLen = 4 * 1024;
            int64_t singleUBProcessNum = static_cast<int64_t>(castBufferLen) / 2LL;

            int64_t dropMaskSize = td_->opInfo.get_b() * td_->opInfo.get_n() * td_->opInfo.get_g() * td_->opInfo.get_sQ() *
                                   td_->opInfo.get_sKV();

            int64_t maskSize = AlignTo(dropMaskSize, static_cast<int64_t>(BOOL_BLOCK_NUMS));
            int64_t singleCoreNum =
                AlignTo(CeilCommon(maskSize, static_cast<int64_t>(td_->splitCoreParams.get_usedCoreNum())),
                        static_cast<int64_t>(BOOL_BLOCK_NUMS));
            uint32_t maskUsedCoreNum = static_cast<uint32_t>(CeilCommon(maskSize, singleCoreNum));

            int64_t tailCoreNum = maskSize - (static_cast<int64_t>(maskUsedCoreNum) - 1LL) * singleCoreNum;
            tailCoreNum = AlignTo(tailCoreNum, static_cast<int64_t>(BOOL_BLOCK_NUMS));

            uint32_t singleCoreUBLoop = static_cast<uint32_t>(CeilCommon(singleCoreNum, singleUBProcessNum));
            uint32_t tailCoreUBLoop = static_cast<uint32_t>(CeilCommon(tailCoreNum, singleUBProcessNum));

            uint32_t singleCoreUBLastLoopNum = static_cast<uint32_t>(singleCoreNum) -
                                               (singleCoreUBLoop - 1U) * static_cast<uint32_t>(singleUBProcessNum);
            uint32_t tailCoreUBLastLoopNum =
                static_cast<uint32_t>(tailCoreNum) - (tailCoreUBLoop - 1U) * static_cast<uint32_t>(singleUBProcessNum);

            if (td_->opInfo.get_sKV() % DROPOUT4BIT_LEN != 0) {
                dropoutIsDivisibleBy8 = 0U;
            }

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
            td_->preTilingData.set_dropoutIsDivisibleBy8(dropoutIsDivisibleBy8);
            td_->preTilingData.set_dropBeginAddr(0);

            int64_t dropoutWorkspaceLen = CeilCommon(dropMaskSize, static_cast<int64_t>(WORKSPACE_ALIGN_SIZE)) *
                                          static_cast<int64_t>(WORKSPACE_ALIGN_SIZE);
            td_->opInfo.set_dropoutWorkspaceLen(dropoutWorkspaceLen);

            return;
        }
        td_->opInfo.set_dropoutWorkspaceLen(0);
        td_->preTilingData.set_dropoutIsDivisibleBy8(dropoutIsDivisibleBy8);
    }

    void PrintShapeInfo()
    {
        OP_LOGI(context_,
                  "FAG ngs1s2_bn with shape b[%ld] n2[%ld] g[%ld] s1[%ld] s2[%ld] d[%ld] preToken[%ld] nextToken[%ld]!",
                  td_->opInfo.get_b(), td_->opInfo.get_n(), td_->opInfo.get_g(), td_->opInfo.get_sQ(), td_->opInfo.get_sKV(),
                  td_->opInfo.get_d(), td_->opInfo.get_preTokens(), td_->opInfo.get_nextTokens());
    }

    ge::graphStatus DoOpTiling() override
    {
        auto ret = DoCoresSplitTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            PrintShapeInfo();
            return ret;
        }
        ret = DoInCoreTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            PrintShapeInfo();
            return ret;
        }

        DoPreTiling();

        ret = DoMulsTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            PrintShapeInfo();
            return ret;
        }
        ret = CheckAttenMaskShape();
        if (ret != ge::GRAPH_SUCCESS) {
            PrintShapeInfo();
            return ret;
        }
        return ge::GRAPH_SUCCESS;
    }

    bool IsBiggerThanL1Size(int64_t nCvInner)
    {
        uint64_t inputSize = td_->opInfo.get_inputDTypeSize();
        int64_t d = td_->opInfo.get_d();
        int64_t sKV = td_->opInfo.get_sKV();
        uint64_t sQ = td_->opInfo.get_sQ();
        uint64_t dAlign16 = AlignSize(static_cast<uint64_t>(d), 16ULL);
        uint64_t sqAlign16 = AlignSize(static_cast<uint64_t>(sQ), 16ULL);
        uint64_t skvAlign16 = AlignSize(static_cast<uint64_t>(sKV), 16ULL);
        uint64_t nInTimesG = nCvInner * td_->opInfo.get_g();
        if ((sqAlign16 + skvAlign16) * nInTimesG * dAlign16 * inputSize <= aicoreParams_.l1Size &&
            (dAlign16 + skvAlign16) * nInTimesG * sqAlign16 * inputSize <= aicoreParams_.l1Size &&
            (sqAlign16 + dAlign16) * nInTimesG * skvAlign16 * inputSize <= aicoreParams_.l1Size) {
            return false;
        }
        return true;
    }

    void GetMinCvRatio(int64_t &nCvRatio, int64_t &nCvInner, int64_t nIn)
    {
        do {
            if (IsBiggerThanL1Size(nCvInner)) {
                nCvRatio--;
                nCvInner = nIn * nCvRatio;
            } else {
                break;
            }
        } while (nCvRatio > 1);
    }

    /* 切N轴分核的基础是，当前g * sQ * sKV比较小，且N比较大的时候，需要将N拆成nOut和nIn，其中
     * nIn参与到内层g * sQ * sKV的数据搬运和计算中，保证单次数据搬运和vector算力可以发挥到极致。
     * nOut参与到外层的分核，b和nOut一起分核。
     * 当前计算nOut和nIn的方式是让 nIn * g * sQ * sKV 尽可能的接近128 *
     * 128。然后将ubsize累加之后计算是否总的ub大小够放， 如果够的话直接使用当前nIn，否则使用64 * 128，64 *
     * 64的拼凑方式。 */
    ge::graphStatus DoCoresSplitTiling()
    {
        OP_LOGD(context_, "Do op core split tiling.");
        int64_t gSqSkvAlign = td_->opInfo.get_g() * td_->opInfo.get_sQ() * td_->opInfo.get_sKVAlign();
        OP_CHECK_IF(gSqSkvAlign == 0,
                   OP_LOGW(context_, "gSqSkvAlign is 0."),
                   return ge::GRAPH_PARAM_INVALID);

        int64_t nIn = td_->singleCoreParams.get_nIn();
        bool ret = CheckArgsLegal(nIn);
        OP_CHECK_IF(!ret,
                   OP_LOGW(context_, "check args fail."),
                   return ge::GRAPH_PARAM_INVALID);

        /* CV 配比*/
        int64_t nCvRatio = CV_RATIO;
        int64_t nCvInner = nIn * nCvRatio;
        if (nCvInner > td_->opInfo.get_n()) {
            nCvInner = td_->opInfo.get_n();
            nCvRatio = CeilCommon(td_->opInfo.get_n(), nIn);
        }

        // 因为bmm的L1size限制问题，check nIn之后是否能放的下，放不下就走后面的模板
        GetMinCvRatio(nCvRatio, nCvInner, nIn);

        td_->singleCoreParams.set_nCvInner(nCvInner);

        /* 用于分核的参数 */
        int64_t nOut = CeilCommon(td_->opInfo.get_n(), nCvInner);
        td_->splitCoreParams.set_nOut(nOut);
        td_->splitCoreParams.set_totalBatch(td_->opInfo.get_b() * nOut);

        /* 单个核计算的参数 */
        td_->singleCoreParams.set_singleCoreBatchRange(
            CeilCommon(td_->splitCoreParams.get_totalBatch(), aicoreParams_.numBlocks));
        CHECK_ZERO(td_->singleCoreParams.get_singleCoreBatchRange());
        td_->singleCoreParams.set_singleCoreBatchRangeTail(td_->splitCoreParams.get_totalBatch() %
                                                          td_->singleCoreParams.get_singleCoreBatchRange());

        td_->splitCoreParams.set_usedCoreNum(
            CeilCommon(td_->splitCoreParams.get_totalBatch(), td_->singleCoreParams.get_singleCoreBatchRange()));
        td_->splitCoreParams.set_mm1ResSize(td_->opInfo.get_sQ() * td_->opInfo.get_sKVAlign() *
                                           td_->opInfo.get_vecCalcDTypeSize());
        return ge::GRAPH_SUCCESS;
    }

    /* 计算每个核内ub切分的情况：当前S1、S2是不切分的，但是S1或者S2都是允许大于128的，可能会超过单次vector计算的block上限。
     * 1. vector计算时，S1, S2需要拆分时记录需要的切分数据，用于减少一些scalar计算。
     * 2. D轴切分相关的参数。 */
    ge::graphStatus DoInCoreTiling()
    {
        OP_LOGD(context_, "Do op in core split tiling.");
        OP_CHECK_IF(td_->opInfo.get_vecCalcDTypeSize() == 0,
                   OP_LOGW(context_, "vecCalcDTypeSize is 0."),
                   return ge::GRAPH_PARAM_INVALID);
        td_->singleCoreParams.set_subRange(CeilCommon(td_->opInfo.get_sQ(), PER_SUB_RANGE));
        td_->singleCoreParams.set_subMask(SINGLE_VEC_INST_DATASIZE / td_->opInfo.get_vecCalcDTypeSize());
        td_->singleCoreParams.set_subMaskTail((td_->opInfo.get_sQ() % PER_SUB_RANGE) *
                                             (BYTE_PER_BLOCK / td_->opInfo.get_vecCalcDTypeSize()));
        td_->singleCoreParams.set_sKVAlignBlockNumVec(td_->opInfo.get_sKVAlignSizeVec() / BYTE_PER_BLOCK);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoMulsTiling()
    {
        OP_LOGD(context_, "Do muls tiling.");
        int64_t dAlign = (td_->opInfo.get_d() + 15) / 16 * 16;
        uint64_t allNumQuery =
            td_->opInfo.get_b() * td_->opInfo.get_n() * td_->opInfo.get_g() * td_->opInfo.get_sQ() * dAlign;
        uint64_t allNumKv = td_->opInfo.get_b() * td_->opInfo.get_n() * td_->opInfo.get_sKV() * dAlign;
        int64_t usedCoreNum = td_->splitCoreParams.get_usedCoreNum();
        OP_CHECK_IF(usedCoreNum == 0,
                   OP_LOGW(context_, "usedCoreNum is 0."),
                   return ge::GRAPH_PARAM_INVALID);

        basicParams.mmNextIsNZOut = (td_->opInfo.get_sQ() >= NZ_S_MIN
                              && td_->opInfo.get_inputLayout() == static_cast<uint32_t>(InputLayout::BNSD))
                              ? true : false;
        uint32_t postUbBaseSize = 0;
        uint32_t qPostBaseNum = 0;
        uint32_t nzReservedSize = 0;
        if (!basicParams.mmNextIsNZOut) {
            postUbBaseSize = (aicoreParams_.ubSize) / POST_COEX_NODE / BUFFER_NUM / BASE_LEN_256 * BASE_LEN_256;
            qPostBaseNum = postUbBaseSize / FP16_BYTES_NUM;
        } else {
            int64_t curPostCoexNode = POST_NZ_COEX_NODE;
            nzReservedSize =
                static_cast<uint32_t>(dAlign / C0_SIZE) * BLOCK_SIZE * POST_NZ_RESERVED_N; // 16为一个单元长度
            postUbBaseSize = (aicoreParams_.ubSize - 2U * nzReservedSize) / static_cast<uint32_t>(curPostCoexNode) /
                             BUFFER_NUM / // 开DB预留2份nzReservedSize
                             BASE_LEN_256 * BASE_LEN_256;
            qPostBaseNum = postUbBaseSize / FP16_BYTES_NUM / dAlign * td_->opInfo.get_d();
        }

        OP_CHECK_IF(qPostBaseNum == 0,
                   OP_LOGW(context_, "qPostBaseNum is 0."),
                   return ge::GRAPH_PARAM_INVALID);
        uint64_t qPostBlockTotal = allNumQuery / dAlign * td_->opInfo.get_d();
        uint64_t qSizeAlign =
            (qPostBlockTotal + BASE_LEN_256 - 1) / WORKSPACE_ALIGN_SIZE * WORKSPACE_ALIGN_SIZE * FP16_BYTES_NUM;
        int64_t qPostTailNumTmp = static_cast<int64_t>(qPostBlockTotal % qPostBaseNum);
        int64_t qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
        int64_t qPostBlockOuterTotal = static_cast<int64_t>((qPostBlockTotal + qPostBaseNum - 1U) / qPostBaseNum);
        int64_t qPostBlockFactor = (qPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

        int64_t kvPostBaseNum = static_cast<int64_t>(qPostBaseNum);
        OP_CHECK_IF(kvPostBaseNum == 0,
                   OP_LOGW(context_, "kvPostBaseNum is 0."),
                   return ge::GRAPH_PARAM_INVALID);
        uint64_t kvPostBlockTotal = allNumKv / dAlign * td_->opInfo.get_d();
        uint64_t kvSizeAlign = (kvPostBlockTotal + WORKSPACE_ALIGN_SIZE - 1U) / WORKSPACE_ALIGN_SIZE *
                               WORKSPACE_ALIGN_SIZE * static_cast<uint64_t>(FP16_BYTES_NUM);
        int64_t kvPostTailNumTmp = static_cast<int64_t>(kvPostBlockTotal) % kvPostBaseNum;
        int64_t kvPostTailNum = kvPostTailNumTmp == 0 ? kvPostBaseNum : kvPostTailNumTmp;
        int64_t kvPostBlockOuterTotal = (static_cast<int64_t>(kvPostBlockTotal) + kvPostBaseNum - 1) / kvPostBaseNum;
        int64_t kvPostBlockFactor = (kvPostBlockOuterTotal + usedCoreNum - 1) / usedCoreNum;

        td_->postTilingData.set_coreNum(usedCoreNum);
        td_->postTilingData.set_scaleValue(td_->opInfo.get_scaleValue());
        td_->postTilingData.set_postUbBaseSize(postUbBaseSize);
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

        td_->postTilingData.set_b(td_->opInfo.get_b());
        td_->postTilingData.set_n2(td_->opInfo.get_n());
        td_->postTilingData.set_g(td_->opInfo.get_g());
        td_->postTilingData.set_s1(td_->opInfo.get_sQ());
        td_->postTilingData.set_s2(td_->opInfo.get_sKV());
        td_->postTilingData.set_d(td_->opInfo.get_d());

        td_->opInfo.set_dqWorkspaceLen(CeilCommon(allNumQuery * FP32_BYTES_NUM, WORKSPACE_ALIGN_SIZE) *
                                      WORKSPACE_ALIGN_SIZE);
        td_->opInfo.set_dkWorkspaceLen(CeilCommon(allNumKv * FP32_BYTES_NUM, WORKSPACE_ALIGN_SIZE) *
                                      WORKSPACE_ALIGN_SIZE);

        uint64_t allNumDropGm = td_->opInfo.get_b() * td_->opInfo.get_n() * td_->opInfo.get_g() * td_->opInfo.get_sQ() *
                                td_->opInfo.get_sKVAlign();
        uint64_t allNumMulGm = td_->opInfo.get_b() * td_->opInfo.get_n() * td_->opInfo.get_g() * td_->opInfo.get_sQ() *
                               td_->opInfo.get_sKVAlign();

        // CV并行实现，需要申请双倍的bmm345的输入空间
        td_->opInfo.set_dropGmWorkspaceLen(2 * CeilCommon(allNumDropGm * FP16_BYTES_NUM, WORKSPACE_ALIGN_SIZE) *
                                          WORKSPACE_ALIGN_SIZE);
        td_->opInfo.set_mulGmWorkspaceLen(2 * CeilCommon(allNumMulGm * FP16_BYTES_NUM, WORKSPACE_ALIGN_SIZE) *
                                         WORKSPACE_ALIGN_SIZE);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckAttenMaskShape()
    {
        if (basicParams.attenMaskCompressMode == NO_COMPRESS_MODE) {
            bool invalid =
                td_->opInfo.get_hasAttenMask() != 0 && (basicParams.attenMaskS1Size * basicParams.attenMaskS2Size <
                                                       td_->opInfo.get_sQ() * td_->opInfo.get_sKV());
            OP_CHECK_IF(invalid,
                    OP_LOGW(context_, "atten mask shape [%ld,%ld] is invalid.", basicParams.attenMaskS1Size,
                              basicParams.attenMaskS2Size),
                    return ge::GRAPH_PARAM_INVALID);
        } else {
            OP_CHECK_IF((basicParams.attenMaskS1Size != basicParams.attenMaskS2Size),
                    OP_LOGW(context_, "atten mask shape is not square."),
                    return ge::GRAPH_PARAM_INVALID);
            OP_CHECK_IF((basicParams.attenMaskS2Size < std::max(td_->opInfo.get_sQ(), td_->opInfo.get_sKV()) * MULT_BASE),
                       OP_LOGW(context_, "atten mask shape is small, try setting it to [2048, 2048]."),
                       return ge::GRAPH_PARAM_INVALID);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetMm1AndMm2Tiling(matmul_tiling::MatmulApiTiling &mm1AndMm2, int64_t nCvInner,
                                       AscendC::tiling::TCubeTiling* mm1AndMm2Tiling)
    {
        mm1AndMm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_FLOAT16, false);
        mm1AndMm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_FLOAT16, true);
        mm1AndMm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_FLOAT16);
        mm1AndMm2.SetShape(td_->opInfo.get_sQ(), td_->opInfo.get_sKV(), td_->opInfo.get_d());

        mm1AndMm2.SetALayout(td_->opInfo.get_b(), td_->opInfo.get_sQ(), td_->opInfo.get_n(), td_->opInfo.get_g(),
                             td_->opInfo.get_d());
        mm1AndMm2.SetBLayout(td_->opInfo.get_b(), td_->opInfo.get_sKV(), td_->opInfo.get_n(), 1, td_->opInfo.get_d());
        mm1AndMm2.SetCLayout(td_->opInfo.get_b(), td_->opInfo.get_sQ(), td_->opInfo.get_n(), td_->opInfo.get_g(),
                             td_->opInfo.get_sKV());
        mm1AndMm2.SetBatchNum(nCvInner * td_->opInfo.get_g());
        uint32_t layout = td_->opInfo.get_inputLayout();
        if (layout == static_cast<uint32_t>(InputLayout::BSH) || layout == static_cast<uint32_t>(InputLayout::BSND)) {
            mm1AndMm2.SetOrgShape(td_->opInfo.get_sQ(), td_->opInfo.get_sKV(), td_->opInfo.get_hQ(), td_->opInfo.get_hKV());
        } else if (layout == static_cast<uint32_t>(InputLayout::SBH)) {
            mm1AndMm2.SetOrgShape(td_->opInfo.get_sQ(), td_->opInfo.get_sKV(), td_->opInfo.get_b() * td_->opInfo.get_hQ(),
                                  td_->opInfo.get_b() * td_->opInfo.get_hKV());
        } else if (layout == static_cast<uint32_t>(InputLayout::BNSD)) {
            mm1AndMm2.SetOrgShape(td_->opInfo.get_sQ(), td_->opInfo.get_sKV(), td_->opInfo.get_d());
        } else {
            return ge::GRAPH_PARAM_INVALID;
        }

        OP_CHECK_IF((mm1AndMm2.GetTiling(*mm1AndMm2Tiling) != 0),
                   OP_LOGW(context_, "Failed to do mm1 and mm2 tiling."),
                   return ge::GRAPH_PARAM_INVALID);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetMm31Tiling(matmul_tiling::MatmulApiTiling &mm31, int64_t nCvInner, AscendC::tiling::TCubeTiling* mm31Tiling)
    {
        mm31.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                      false);
        mm31.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                      false);
        mm31.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
        mm31.SetShape(td_->opInfo.get_sQ(), td_->opInfo.get_d(), td_->opInfo.get_sKV());

        mm31.SetALayout(td_->opInfo.get_b(), td_->opInfo.get_sQ(), td_->opInfo.get_n(), td_->opInfo.get_g(),
                        td_->opInfo.get_sKV());
        mm31.SetBLayout(td_->opInfo.get_b(), td_->opInfo.get_sKV(), td_->opInfo.get_n(), 1, td_->opInfo.get_d());
        mm31.SetCLayout(td_->opInfo.get_b(), td_->opInfo.get_sQ(), td_->opInfo.get_n(), td_->opInfo.get_g(),
                        td_->opInfo.get_d());
        mm31.SetOrgShape(td_->opInfo.get_sQ(), td_->opInfo.get_d(), td_->opInfo.get_sKV());
        mm31.SetBatchNum(nCvInner * td_->opInfo.get_g());

        OP_CHECK_IF((mm31.GetTiling(*mm31Tiling) != 0),
                   OP_LOGW(context_, "Failed to do mm31 tiling."),
                   return ge::GRAPH_PARAM_INVALID);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetMm32AndMm4Tiling(matmul_tiling::MatmulApiTiling &mm32AndMm4, int64_t nCvInner,
                                        AscendC::tiling::TCubeTiling* mm32AndMm4Tiling)
    {
        mm32AndMm4.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_FLOAT16, true);
        mm32AndMm4.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_FLOAT16, false);
        mm32AndMm4.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_FLOAT16);
        mm32AndMm4.SetShape(td_->opInfo.get_sKV(), td_->opInfo.get_d(), td_->opInfo.get_sQ());

        mm32AndMm4.SetALayout(td_->opInfo.get_b(), td_->opInfo.get_sKV(), td_->opInfo.get_n(), td_->opInfo.get_g(),
                              td_->opInfo.get_sQ());
        mm32AndMm4.SetBLayout(td_->opInfo.get_b(), td_->opInfo.get_sQ(), td_->opInfo.get_n(), td_->opInfo.get_g(),
                              td_->opInfo.get_d());
        mm32AndMm4.SetCLayout(td_->opInfo.get_b(), td_->opInfo.get_sKV(), td_->opInfo.get_n(), 1, td_->opInfo.get_d());
        mm32AndMm4.SetOrgShape(td_->opInfo.get_sKV(), td_->opInfo.get_d(), td_->opInfo.get_sQ());
        mm32AndMm4.SetBatchNum(nCvInner * td_->opInfo.get_g());

        OP_CHECK_IF((mm32AndMm4.GetTiling(*mm32AndMm4Tiling) != 0),
                   OP_LOGW(context_, "Failed to do mm32AndMm4Tiling tiling."),
                   return ge::GRAPH_PARAM_INVALID);
        return ge::GRAPH_SUCCESS;
    }

    // 4、计算高阶API的tiling
    ge::graphStatus DoLibApiTiling() override
    {
        // mm tiling
        ge::graphStatus ret;
        matmul_tiling::MatmulApiTiling mm1AndMm2;
        ret = SetMm1AndMm2Tiling(mm1AndMm2, td_->singleCoreParams.get_nCvInner(), &td_->mm1AndMm2TilingData);
        if (ret != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_PARAM_INVALID;
        }

        matmul_tiling::MatmulApiTiling mm31;
        ret = SetMm31Tiling(mm31, td_->singleCoreParams.get_nCvInner(), &td_->mm31TilingData);
        if (ret != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_PARAM_INVALID;
        }

        matmul_tiling::MatmulApiTiling mm32AndMm4;
        ret = SetMm32AndMm4Tiling(mm32AndMm4, td_->singleCoreParams.get_nCvInner(), &td_->mm32AndMm4TilingData);
        if (ret != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_PARAM_INVALID;
        }

        // vector tiling
        auto softmaxShape = Shape(
            {td_->singleCoreParams.get_nIn() * td_->opInfo.get_g() * td_->opInfo.get_sQ(), td_->opInfo.get_sKVAlign()});

        int64_t softmaxTmpSize = GetSoftMaxMinTmpSize(softmaxShape, sizeof(float), true);

        auto softmaxGradShape = Shape({td_->singleCoreParams.get_nIn() * td_->opInfo.get_g() * td_->opInfo.get_sQ(),
                                       td_->singleCoreParams.get_splitedDAlign()});

        int64_t softmaxGradTmpSize =
            GetSoftMaxGradMinTmpSize(softmaxGradShape, td_->opInfo.get_vecCalcDTypeSize(), true, true);
        if (basicParams.ubSizeRemain < softmaxGradTmpSize || basicParams.ubSizeRemain < softmaxTmpSize) {
            return ge::GRAPH_PARAM_INVALID;
        }

        SoftMaxTilingFunc(softmaxShape, sizeof(float), basicParams.ubSizeRemain, td_->softmaxTilingData);
        SoftMaxGradTilingFunc(softmaxGradShape, td_->opInfo.get_vecCalcDTypeSize(), basicParams.ubSizeRemain,
                              td_->softmaxGradTilingData, true);
        return ge::GRAPH_SUCCESS;
    }

    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override
    {
        uint32_t sysLen = WORK_SPACE_BASE_CAL;
        uint64_t mm1WorkspaceLen = td_->singleCoreParams.get_nCvInner() * td_->opInfo.get_g() * td_->opInfo.get_sQ() *
                                   td_->opInfo.get_sKVAlign() * td_->opInfo.get_vecCalcDTypeSize();
        mm1WorkspaceLen = CeilCommon(mm1WorkspaceLen, WORKSPACE_ALIGN_SIZE) * WORKSPACE_ALIGN_SIZE *
                          td_->splitCoreParams.get_usedCoreNum();
        uint64_t mm2WorkspaceLen = mm1WorkspaceLen;
        uint64_t dqWorkspaceLen = td_->opInfo.get_dqWorkspaceLen();
        uint64_t dkWorkspaceLen = td_->opInfo.get_dkWorkspaceLen();
        uint64_t dropOutWorkspaceLen = td_->opInfo.get_dropoutWorkspaceLen();

        uint64_t mulGmWorkspaceLen = td_->opInfo.get_mulGmWorkspaceLen();
        uint64_t dropGmWorkspaceLen = td_->opInfo.get_dropGmWorkspaceLen();

        uint64_t workspaceOffsets = dropOutWorkspaceLen + mm1WorkspaceLen + mm2WorkspaceLen;
        td_->postTilingData.set_dqWorkSpaceOffset(workspaceOffsets);

        workspaceOffsets = workspaceOffsets + td_->opInfo.get_dqWorkspaceLen();
        td_->postTilingData.set_dkWorkSpaceOffset(workspaceOffsets);

        workspaceOffsets = workspaceOffsets + td_->opInfo.get_dkWorkspaceLen();
        td_->postTilingData.set_dvWorkSpaceOffset(workspaceOffsets);

        workspaceSize_ = sysLen + dropOutWorkspaceLen + mm1WorkspaceLen + mm2WorkspaceLen + dqWorkspaceLen +
                         dkWorkspaceLen + dropGmWorkspaceLen + mulGmWorkspaceLen;

        OP_LOGD(context_, "Calc workspace size: workspaceSize is %lu, mm1WorkspaceLen is %lu.", workspaceSize_,
                  mm1WorkspaceLen);
        td_->opInfo.set_mm1WorkspaceLen(mm1WorkspaceLen);
        td_->opInfo.set_mm2WorkspaceLen(mm2WorkspaceLen);
        return ge::GRAPH_SUCCESS;
    }

    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override
    {
        OP_LOGD(context_, "Ungs1s2Bbn post tiling.");
        auto blockdim =
            CalcTschBlockDim(td_->splitCoreParams.get_usedCoreNum(), aicoreParams_.aicNum, aicoreParams_.numBlocks);
        OP_CHECK_IF(blockdim == 0,
                   OP_LOGW(context_,
                             "blockdim is 0, aicNum is %lu, aivNum is %lu.", aicoreParams_.aicNum,
                             aicoreParams_.numBlocks),
                   return ge::GRAPH_PARAM_INVALID);
        context_->SetBlockDim(blockdim);
        context_->SetScheduleMode(1);

        size_t *workspaces = context_->GetWorkspaceSizes(1);
        workspaces[0] = workspaceSize_;
        return ge::GRAPH_SUCCESS;
    }
};

REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttentionScoreGrad, FlashAttentionScoreGradUngs1s2BbnTiling,
                                         std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}),
                                         11000);

} // namespace optiling
