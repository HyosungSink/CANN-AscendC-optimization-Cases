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
 * \file flash_attention_score_grad_tiling_bngs1s2_b.cc
 * \brief
 */

#include "flash_attention_score_grad_tiling_common.h"
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
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
constexpr int64_t FP16_BLOCK_ELES = 16;
constexpr int64_t FP32_BLOCK_ELES = 8;
constexpr int64_t BLOCK_BYTE = 32;
constexpr int64_t C0_SIZE = 16;
constexpr int64_t VEC_REPEAT = 8;
constexpr int64_t NZ_S_MIN = 4;
constexpr uint32_t POST_NZ_COEX_NODE = 10;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t POST_NZ_RESERVED_N = 1;
constexpr int64_t PER_SUB_RANGE = 8;
constexpr int64_t CV_RATIO = 2;
constexpr int64_t BEST_BASIC_BLOCK_SIZE = 64 * 128 * 4;
constexpr int64_t BEST_BASIC_BLOCK_NUM = 64 * 128;
constexpr int64_t L1_BYTE_SIZE = 512 * 1024;
constexpr int64_t CHECKNUM_FOR_L1_SIZE = 3;
constexpr uint32_t MAX_STRIDE_LIMIT = 65535;
constexpr uint32_t POST_COEX_NODE = 3;
constexpr uint32_t BASE_LEN_256 = 256;
constexpr uint32_t WORKSPACE_ALIGN_SIZE = 512;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t SYNC_LEN = 3200;
constexpr uint32_t D_SIZE_72 = 72;
constexpr uint32_t D_SIZE_88 = 88;
constexpr uint32_t S_SIZE_15 = 15;
constexpr uint32_t S_SIZE_16 = 16;

enum class DTemplateType {
    NonAligned = 0,
    NonAligned72 = 3,
    NonAligned88 = 4,
    Aligned80 = 5,
    Aligned96 = 6,
    DTemplateBottom
};

enum class STemplateType {
    NonAligned = 0,
    Aligned16 = 1,
    Aligned32 = 2,
    NonAligned15 = 3,
    NonAligned30 = 4,
    DTemplateBottom
};

#define CHECK_ZERO(num)                                                                                                \
    do {                                                                                                               \
        if ((num) == 0) {                                                                                              \
            OP_LOGW("template 4.1 number[%s] is zero.", #num);                                                       \
            return ge::GRAPH_PARAM_INVALID;                                                                            \
        }                                                                                                              \
    } while (0)

#define CHECK_ZERO_FALSE(num)                                                                                          \
    do {                                                                                                               \
        if ((num) == 0) {                                                                                              \
            OP_LOGW("template 4.1 number[%s] is zero.", #num);                                                       \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

template <typename... Args> constexpr uint64_t GET_B_TILINGKEY(Args... templateIds)
{
    return TILINGKEYOFFSET + RecursiveSum(templateIds...);
}

struct TempParamsUngs1s2Bb {
    uint32_t dataType;
    uint32_t precisionMode;
    uint32_t layout;
    uint32_t attenMaskCompressMode;
    int64_t attenMaskS1Size = 0;
    int64_t attenMaskS2Size = 0;
    bool mmPreIsNZOut;
    bool mmNextIsNZOut;
    int64_t b;
    int64_t n;
    int64_t sQ;
    int64_t d;
};

class FlashAttentionScoreGradUbngs1s2BbTiling : public TilingBaseClass {
public:
    FlashAttentionScoreGradUbngs1s2BbTilingData *td_ = context_->GetTilingData<FlashAttentionScoreGradUbngs1s2BbTilingData>();
    TempParamsUngs1s2Bb basicParams;

    explicit FlashAttentionScoreGradUbngs1s2BbTiling(gert::TilingContext *context) : TilingBaseClass(context)
    {
        td_->reset();
    };

    ~FlashAttentionScoreGradUbngs1s2BbTiling() override = default;

    inline int64_t Align(const int64_t n, const int64_t alignSize)
    {
        if (alignSize == 0) {
            return 0;
        }
        return (n + alignSize - 1) & (~(alignSize - 1));
    }

    uint64_t GetTilingKey() const override
    {
        uint64_t tilingKey = 0;
        auto dtype = DtypeEnum::FLOAT16;
        STemplateType s1TemplateType = STemplateType::NonAligned;
        STemplateType s2TemplateType = STemplateType::NonAligned;
        DTemplateType dTemplateType = DTemplateType::NonAligned;
        
        if (basicParams.dataType == static_cast<uint32_t>(ge::DT_FLOAT16)) {
            dtype = DtypeEnum::FLOAT16_PRECISION;
        } else if (basicParams.dataType == static_cast<uint32_t>(ge::DT_BF16)) {
            dtype = DtypeEnum::BFLOAT16;
        } else {
            dtype = DtypeEnum::FLOAT32;
        }

        auto mmPreIsNZOut = basicParams.mmPreIsNZOut ? OptionEnum::ENABLE : OptionEnum::DISABLE;
        auto mmNextIsNZOut = basicParams.mmNextIsNZOut ? OptionEnum::ENABLE : OptionEnum::DISABLE;
        auto unique = OptionEnum::DISABLE;

        if (basicParams.layout == static_cast<uint32_t>(InputLayout::SBH) &&
            basicParams.b * basicParams.n * basicParams.d > static_cast<int64_t>(MAX_STRIDE_LIMIT)) {
            tilingKey = GET_TPL_TILING_KEY(static_cast<uint8_t>(AxisEnum::NONE), static_cast<uint8_t>(AxisEnum::NONE), static_cast<uint8_t>(AxisEnum::B), 0, static_cast<uint8_t>(dtype),
                static_cast<uint8_t>(basicParams.layout), static_cast<uint8_t>(SparseEnum::NONE), static_cast<uint8_t>(MatmulConfig::NORMAL_CONFIG), static_cast<uint8_t>(mmPreIsNZOut),
                static_cast<uint8_t>(mmNextIsNZOut), static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                0, 0, 0, static_cast<uint8_t>(s1TemplateType), static_cast<uint8_t>(s2TemplateType), static_cast<uint8_t>(dTemplateType), 0, 0);
            return tilingKey;
        } else {
            // SBH: BND <= 65535
            if (unique == OptionEnum::DISABLE && mmPreIsNZOut == OptionEnum::ENABLE && mmNextIsNZOut == OptionEnum::DISABLE
            && dtype == DtypeEnum::BFLOAT16 && basicParams.layout == static_cast<uint32_t>(InputLayout::BSH)) {
                if (basicParams.d == static_cast<int64_t>(D_SIZE_72) &&
                    basicParams.sQ == static_cast<int64_t>(S_SIZE_15)) {
                    dTemplateType = DTemplateType::NonAligned72;
                    s1TemplateType = STemplateType::NonAligned15;
                    s2TemplateType = STemplateType::NonAligned15;
                } else if (basicParams.d == static_cast<int64_t>(D_SIZE_72) &&
                           basicParams.sQ == static_cast<int64_t>(S_SIZE_16)) {
                    dTemplateType = DTemplateType::NonAligned72;
                    s1TemplateType = STemplateType::Aligned16;
                    s2TemplateType = STemplateType::Aligned16;
                } else if (basicParams.d == static_cast<int64_t>(D_SIZE_88) &&
                           basicParams.sQ == static_cast<int64_t>(S_SIZE_16)) {
                    dTemplateType = DTemplateType::NonAligned88;
                    s1TemplateType = STemplateType::Aligned16;
                    s2TemplateType = STemplateType::Aligned16;
                }
            } else if (unique == OptionEnum::DISABLE && mmPreIsNZOut == OptionEnum::ENABLE && mmNextIsNZOut == OptionEnum::DISABLE
            && dtype == DtypeEnum::BFLOAT16 && basicParams.layout == static_cast<uint32_t>(InputLayout::BSND)) {
                if (basicParams.d == static_cast<int64_t>(D_SIZE_72) &&
                    basicParams.sQ == static_cast<int64_t>(S_SIZE_15)) {
                    dTemplateType = DTemplateType::NonAligned72;
                    s1TemplateType = STemplateType::NonAligned15;
                    s2TemplateType = STemplateType::NonAligned15;
                } else if (basicParams.d == static_cast<int64_t>(D_SIZE_72) &&
                           basicParams.sQ == static_cast<int64_t>(S_SIZE_16)) {
                    dTemplateType = DTemplateType::NonAligned72;
                    s1TemplateType = STemplateType::Aligned16;
                    s2TemplateType = STemplateType::Aligned16;
                } else if (basicParams.d == D_SIZE_88 && basicParams.sQ == S_SIZE_16) {
                    dTemplateType = DTemplateType::NonAligned88;
                    s1TemplateType = STemplateType::Aligned16;
                    s2TemplateType = STemplateType::Aligned16;
                }
            }
            tilingKey = GET_TPL_TILING_KEY(static_cast<uint8_t>(AxisEnum::NONE), static_cast<uint8_t>(AxisEnum::NONE), static_cast<uint8_t>(AxisEnum::B), 0, static_cast<uint8_t>(dtype),
                static_cast<uint8_t>(basicParams.layout), static_cast<uint8_t>(SparseEnum::NONE), static_cast<uint8_t>(unique), static_cast<uint8_t>(mmPreIsNZOut),
                static_cast<uint8_t>(mmNextIsNZOut), static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                0, 0, 0, static_cast<uint8_t>(s1TemplateType), static_cast<uint8_t>(s2TemplateType), static_cast<uint8_t>(dTemplateType), 0, 0);
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

        if (basicParams.dataType == ge::DT_FLOAT) {
            OP_LOGD(context_, "Ubngs1s2Bb is not support fp32");
            return false;
        }

        OP_CHECK_IF(context_->GetAttrs() == nullptr,
                   OP_LOGW(context_, "GetAttrs is nullptr."),
                   return false);

        if (context_->GetAttrs()->GetAttrNum() > static_cast<size_t>(PSETYPE)) {
            auto pseType = *context_->GetAttrs()->GetAttrPointer<int>(PSETYPE); // 8
            if (pseType != 1) { // 不支持非默认的psetype
                return false;
            }
        }

        auto bestBasicBlockNum = BEST_BASIC_BLOCK_NUM;
        if (td_->opInfo.get_sQ() >= 4 ? true : false) {
            bestBasicBlockNum = BEST_BASIC_BLOCK_NUM / 4 * 3;
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

        if ((td_->opInfo.get_n() * td_->opInfo.get_g() * Align(td_->opInfo.get_sQ(), FP16_BLOCK_ELES) *
             Align(td_->opInfo.get_sKV(), FP16_BLOCK_ELES)) <= bestBasicBlockNum) {
            OP_LOGD(context_, "Ubngs1s2Bb isCapable check true.");
            return true;
        }

        OP_LOGD(context_, "Ubngs1s2Bb isCapable check false.");
        return false;
    }

    ge::graphStatus GetPlatformInfo() override
    {
        OP_LOGD(context_, "Get platform info begin.");

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

    ge::graphStatus SetAttenMaskTilingInfo()
    {
        OP_LOGD(context_, "Ubngs1s2Bb set attenmask tiling info begin.");
        basicParams.attenMaskCompressMode = NO_COMPRESS_MODE;
        auto attenMaskDesc = context_->GetOptionalInputDesc(ATTEN_MASK);
        if (attenMaskDesc != nullptr) {
            auto attenMaskDtype = attenMaskDesc->GetDataType();
            OP_CHECK_IF(
                (attenMaskDtype != ge::DT_BOOL && attenMaskDtype != ge::DT_UINT8), OP_LOGW(context_, "AttenMaskDtype(%d) is not bool or uint8.", attenMaskDtype),
                      return ge::GRAPH_PARAM_INVALID);
        }
        auto attenMaskShape = context_->GetOptionalInputShape(ATTEN_MASK);
        if (attenMaskShape != nullptr && attenMaskShape->GetStorageShape().GetDimNum() != 0) {
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
                    OP_LOGW(context_, "unsupport attenmask shape type.");
                    return ge::GRAPH_PARAM_INVALID;
                }
                OP_LOGD(context_, "Ubngs1s2Bb get attenmask shape dims success.");
            } else {
                OP_LOGW(context_, "Ubngs1s2Bb set attenmask shape dims fail.");
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
        td_->opInfo.set_attenMaskS2Size(basicParams.attenMaskS2Size);
        td_->opInfo.set_attenMaskCompressMode(basicParams.attenMaskCompressMode);
        OP_LOGD(context_, "Ungs1s2Bb set attenmask tiling info success.");
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetDataTypeInfo()
    {
        /* 3. 获取data type信息 */
        td_->opInfo.set_inputDType(static_cast<uint32_t>(context_->GetInputDesc(QUERY)->GetDataType()));
        basicParams.dataType = td_->opInfo.get_inputDType();
        int64_t inputDTypeSize =
            static_cast<int64_t>(GetSizeByDataType(static_cast<ge::DataType>(td_->opInfo.get_inputDType())));
        if (inputDTypeSize >= ge::kDataTypeSizeBitOffset) {
            return ge::GRAPH_PARAM_INVALID;
        }
        OP_LOGD(context_, "Ungs1s2Bb get inputDTypeSize success.");

        // bf16 场景下使用fp32(4 bytes)来进行vector的数据计算。
        uint32_t vecCalcDTypeSize = FP32_BYTES_NUM;

        td_->opInfo.set_vecCalcDTypeSize(vecCalcDTypeSize);
        td_->opInfo.set_inputDTypeSize(inputDTypeSize);
        int64_t sKVAlignSize = Align(td_->opInfo.get_sKV() * inputDTypeSize, BLOCK_BYTE);
        td_->opInfo.set_sKVAlignSize(sKVAlignSize);
        CHECK_ZERO(inputDTypeSize);
        td_->opInfo.set_sKVAlign(sKVAlignSize / inputDTypeSize);
        td_->opInfo.set_sKVAlignByte(Align(td_->opInfo.get_sKV(), BLOCK_BYTE));
        td_->opInfo.set_originalDAlign(Align(td_->opInfo.get_d() * inputDTypeSize, BLOCK_BYTE) / inputDTypeSize);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetBaseAttrsInfo()
    {
        /* 1. 获取属性信息 */
        td_->opInfo.set_scaleValue(*context_->GetAttrs()->GetAttrPointer<float>(SCALE_VALUE));
        td_->opInfo.set_keepProb(*context_->GetAttrs()->GetAttrPointer<float>(KEEP_PROB));
        OP_CHECK_IF((td_->opInfo.get_keepProb() <= 0 || td_->opInfo.get_keepProb() > 1),
                    OP_LOGW(context_, "keepProb is illegal."),
                    return ge::GRAPH_PARAM_INVALID);
        td_->opInfo.set_preTokens(*context_->GetAttrs()->GetAttrPointer<int64_t>(PRE_TOKENS));
        td_->opInfo.set_nextTokens(*context_->GetAttrs()->GetAttrPointer<int64_t>(NEXT_TOKENS));
        td_->opInfo.set_headNum(*context_->GetAttrs()->GetAttrPointer<uint32_t>(HEAD_NUM));
        OP_CHECK_IF(td_->opInfo.get_headNum() == 0,
                   OP_LOGW(context_, "headNum is 0."),
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
            OP_LOGW(context_, "INPUT_LAYOUT %s is not valid.",
                                        inputLayout);
            return ge::GRAPH_PARAM_INVALID;
        }
        td_->opInfo.set_precisionMode(HIGH_PRECISION);

        basicParams.precisionMode = td_->opInfo.get_precisionMode();
        basicParams.layout = td_->opInfo.get_inputLayout();
        OP_LOGD(context_, "Ungs1s2Bb get input layout success.");
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetDimAttrsInfo()
    {
        /* 2. 获取shape和轴信息 */
        OP_CHECK_IF(((context_->GetInputShape(QUERY) == nullptr) || (context_->GetInputShape(KEY) == nullptr)),
                    OP_LOGE(context_, "InputShape of query or key is nullptr."),
                    return ge::GRAPH_FAILED);
        const gert::Shape &queryShape = context_->GetInputShape(QUERY)->GetStorageShape();
        const gert::Shape &keyShape = context_->GetInputShape(KEY)->GetStorageShape();
        if (td_->opInfo.get_inputLayout() == static_cast<uint32_t>(InputLayout::BNSD) ||
            td_->opInfo.get_inputLayout() == static_cast<uint32_t>(InputLayout::BSND)) {
            OP_CHECK_IF((queryShape.GetDimNum() != BNSD_DIM_NUM || keyShape.GetDimNum() != BNSD_DIM_NUM),
                      OP_LOGW(context_, "Shape size is not = 4[%zu, %zu].",
                      queryShape.GetDimNum(), keyShape.GetDimNum()),
                      return ge::GRAPH_PARAM_INVALID);
            OP_LOGD(context_, "Ungs1s2Bb get input dim success.");
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
                      OP_LOGW(context_, "Shape size is not = 3[%zu, %zu].",
                      queryShape.GetDimNum(), keyShape.GetDimNum()),
                      return ge::GRAPH_PARAM_INVALID);
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

        auto ret = CheckDtypeValid(context_);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                   OP_LOGW(context_, "dtype is invalid."),
                   return ret);

        return CheckShapeValid(context_, basicParams.b, basicParams.n, td_->opInfo.get_sQ(), basicParams.d);
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        OP_CHECK_IF(context_ == nullptr,
                OP_LOGE(context_, "context is nullptr."),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->GetAttrs() == nullptr,
                OP_LOGE(context_, "GetAttrs is nullptr."),
                return ge::GRAPH_FAILED);
        /* 1. 获取属性信息 */
        auto ret = SetBaseAttrsInfo();
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGW(context_, "Ungs1s2Bb set base attrs info fail."),
                return ret);

        /* 2. 获取shape和轴信息 */
        ret = GetDimAttrsInfo();
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGW(context_, "Ungs1s2Bb get dim attrs info fail."),
                return ret);

        /* 3. 获取data type信息 */
        ret = SetDataTypeInfo();
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGW(context_, "Ungs1s2Bb set data type fail."),
                return ret);

        /* 4. 获取pse shape信息 */
        ret = GetPseInfo();
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGW(context_, "Ungs1s2Bb get pse info fail."),
                return ret);

        ret = SetAttenMaskTilingInfo();
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGW(context_, "Ungs1s2Bb set attenmask tiling fail."),
                return ret);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetPseInfo()
    {
        /*
        Get pse input info
        */
        td_->opInfo.set_existPse(0);
        auto pseShape = context_->GetOptionalInputShape(PSE_SHIFT);
        if (pseShape != nullptr && pseShape->GetStorageShape().GetDimNum() != 0) {
            // pse support [B N S1 S2](0) + [B N 1 S2](1) + [1 N S1 S2](2)
            // 4.1 模板不涉及alibi，收益不大
            td_->opInfo.set_existPse(1);
            auto storageShape = pseShape->GetStorageShape();
            auto pseShapeDims = storageShape.GetDimNum();
            if (pseShapeDims == DIM_4) {
                auto dim0 = storageShape.GetDim(DIM_0);
                auto dim1 = storageShape.GetDim(DIM_1);
                auto dim2 = storageShape.GetDim(DIM_2);
                auto dim3 = storageShape.GetDim(DIM_3);
                td_->opInfo.set_pseSq(dim2);
                // [B N S1 S2](0)  [B N 1 S2](1)  [1 N S1 S2](2)shape判断
                uint32_t shapeN1 = td_->opInfo.get_n() * td_->opInfo.get_g();
                bool isBNS = (dim0 == td_->opInfo.get_b()) && (dim1 == shapeN1) && (dim3 == td_->opInfo.get_sKV());
                bool isBNSS = isBNS && (dim2 == td_->opInfo.get_sQ());
                bool isBN1S = isBNS && (dim2 == 1);
                bool is1NSS =
                    (dim0 == 1) && (dim1 == shapeN1) && (dim2 == td_->opInfo.get_sQ()) && (dim3 == td_->opInfo.get_sKV());
                // 设置shape类型
                if (is1NSS) {
                    td_->opInfo.set_pseShapeType(PSE_SHAPE_TYPE_1NSS);
                } else if (isBN1S) {
                    td_->opInfo.set_pseShapeType(PSE_SHAPE_TYPE_BN1S);
                } else if (isBNSS) {
                    td_->opInfo.set_pseShapeType(PSE_SHAPE_TYPE_BNSS);
                } else {
                    OP_LOGW(context_,
                              "Ungs1s2Bb not support pseShape [%ld, %ld, %ld, %ld]", dim0, dim1, dim2, dim3);
                    return ge::GRAPH_PARAM_INVALID;
                }
            } else {
                OP_LOGW(context_, "Ungs1s2Bb not support pseShape dim num: %zu",
                          pseShapeDims);
                return ge::GRAPH_PARAM_INVALID;
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    uint32_t GetApiTmpBufferSize(int64_t bIn, int64_t sQ, int64_t sKVAlign)
    {
        // softmax和dropout对应的vector计算的shape是一样的
        auto shape = ge::Shape({bIn * td_->opInfo.get_n() * td_->opInfo.get_g() * sQ, sKVAlign});

        uint32_t softmaxTmpSize = GetSoftMaxMinTmpSize(shape, td_->opInfo.get_vecCalcDTypeSize(), true);
        uint32_t dropoutTmpSize = GetDropOutMinTmpSize(shape, td_->opInfo.get_vecCalcDTypeSize(), true);

        return std::max(softmaxTmpSize, dropoutTmpSize);
    }

    bool CheckArgsLegal(int64_t bIn)
    {
        int64_t sQ = td_->opInfo.get_sQ();
        int64_t sKVAlign = td_->opInfo.get_sKVAlign();
        // s大于等于4性能才有收益，此处先判断mmPreIsNZOut是否为true
        // mmPreIsNZOut若为ture，需要扩大部分UB，缩小部分UB
        auto bestBasicBlockSize = BEST_BASIC_BLOCK_SIZE;
        if (td_->opInfo.get_sQ() >= 4 ? true : false) {
            bestBasicBlockSize = BEST_BASIC_BLOCK_SIZE / 4 * 3;
        }
        // 计算vecIn1和vecIn2的d轴需要限制，如果d轴大于128，这里仍然使用128计算，并做切分，否则
        // 在fp32场景下只能使用 32 * 32 的基本块；
        // 在fp16场景下只能用 64 * 64 的基本块
        // 在bf16场景下仍然可以用 64 * 128 的基本块
        // 由于d轴在softmaxgrad计算之后可以reduce掉，我们默认会对超过128的d轴做核内切分
        int64_t vecInQue1Size = bestBasicBlockSize / 2;

        /* Bf16下使用fp32进行vector计算，这一步已经在vecCalcDTypeSize的计算中区分。 */
        int64_t vecClc1Size = bestBasicBlockSize;
        int64_t softmaxGradQueSize = bestBasicBlockSize / 2;
        int64_t dropoutQueSize = bestBasicBlockSize / 4;
        int64_t maxSumQueSize = bestBasicBlockSize;
        // bmm的TensorA + TensorB < 512KB (L1_Size)
        int64_t inputDTypeSize = td_->opInfo.get_inputDTypeSize();
        CHECK_ZERO_FALSE(inputDTypeSize);
        int64_t sQAlign4Input = Align(td_->opInfo.get_sQ() * inputDTypeSize, BLOCK_BYTE) / inputDTypeSize;
        int64_t sKVAlign4Input = td_->opInfo.get_sKVAlign();
        int64_t origDAlign4Input = td_->opInfo.get_originalDAlign();
        int64_t baseByteSize = td_->singleCoreParams.get_bCvInner() * td_->opInfo.get_n() * inputDTypeSize;

        // qDx(B, N2, G, S1_Align, D_Align)   kV(B, N2, 1, S2_Align, D_Align)   s1s2(B, N2, G, S2_Align, S2_Align)
        int64_t qDxByteSize = baseByteSize * td_->opInfo.get_g() * sQAlign4Input * origDAlign4Input;
        int64_t kVByteSize = baseByteSize * td_->opInfo.get_g() * sKVAlign4Input * origDAlign4Input;
        int64_t s1s2ByteSize = baseByteSize * td_->opInfo.get_g() * sQAlign4Input * sKVAlign4Input;
        std::array<int64_t, CHECKNUM_FOR_L1_SIZE> numbers = {qDxByteSize, kVByteSize, s1s2ByteSize};
        std::sort(numbers.begin(), numbers.end());
        int64_t largest = numbers[2];
        int64_t secondLargest = numbers[1];

        if ((largest + secondLargest) >= L1_BYTE_SIZE) {
            OP_LOGD(context_,
                      "Ungs1s2Bb bmm TensorA+B size:%ld out of range L1 size:%ld!", largest + secondLargest,
                      L1_BYTE_SIZE);
            return false;
        }

        // s大于等于4性能才有收益
        basicParams.mmPreIsNZOut = td_->opInfo.get_sQ() >= 4 ? true : false;
        // 如果是nz，对于s2每一个分形需要多8个数据
        if (basicParams.mmPreIsNZOut) {
            vecClc1Size += bIn * td_->opInfo.get_n() * td_->opInfo.get_g()
                           * sKVAlign / C0_SIZE * VEC_REPEAT * sizeof(float);
        }
        td_->singleCoreParams.set_innerTmpBufSize(vecClc1Size);
        td_->singleCoreParams.set_vecQueIn1Size(vecInQue1Size);
        int64_t queBufferSizeUb =
            vecInQue1Size * 2 + vecClc1Size * 2 + softmaxGradQueSize + maxSumQueSize + dropoutQueSize;
        // softmaxGrad，softmax，dropout计算所需要的tmpSize
        uint32_t maxTmpBufferSize = GetApiTmpBufferSize(bIn, sQ, sKVAlign);
        OP_LOGD(context_, "softmaxGradComputeSize is:%u.", maxTmpBufferSize);
        int64_t usedBufferSize = queBufferSizeUb + static_cast<int64_t>(maxTmpBufferSize) +
                                static_cast<int64_t>(SOFTMAX_REMAIN_SIZE);

        int64_t ubSizeRemain = static_cast<int64_t>(aicoreParams_.ubSize) - usedBufferSize;
        if (ubSizeRemain >= 0) {
            td_->splitCoreParams.set_apiClcQueueSize(ubSizeRemain + API_RSDV_BUFFER_SIZE);
            return true;
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

            int64_t dropMaskSize = static_cast<int64_t>(td_->opInfo.get_b()) * td_->opInfo.get_n() * td_->opInfo.get_g() *
                                   td_->opInfo.get_sQ() * td_->opInfo.get_sKV();

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
            int64_t dropBeginAddr = SYNC_LEN / sizeof(uint8_t);
            td_->preTilingData.set_dropBeginAddr(dropBeginAddr);

            int64_t dropoutWorkspaceLen = CeilCommon(dropMaskSize, BLOCK_BYTE) * BLOCK_BYTE;
            td_->opInfo.set_dropoutWorkspaceLen(dropoutWorkspaceLen);
            return;
        }
        td_->opInfo.set_dropoutWorkspaceLen(0);
        td_->preTilingData.set_dropoutIsDivisibleBy8(dropoutIsDivisibleBy8);
    }

    ge::graphStatus DoOpTiling() override
    {
        auto ret = DoCoresSplitTiling();
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGW(context_, "Ungs1s2Bb get core split tiling fail."),
                return ret);

        ret = DoInCoreTiling();
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGW(context_, "Ungs1s2Bb get single core tiling fail."),
                return ret);

        DoPreTiling();

        ret = DoMulsTiling();
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGW(context_, "Ungs1s2Bb get muls tiling fail."),
                return ret);

        ret = CheckAttenMaskShape();
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGW(context_, "Ungs1s2Bb check atten mask shape fail."),
                return ret);

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

    /* 切N轴分核的基础是，当前g * sQ * sKV比较小，且N比较大的时候，需要将N拆成nOut和nIn，其中
     * nIn参与到内层g * sQ * sKV的数据搬运和计算中，保证单次数据搬运和vector算力可以发挥到极致。
     * nOut参与到外层的分核，b和nOut一起分核。
     * 当前计算nOut和nIn的方式是让 nIn * g * sQ * sKV 尽可能的接近128 *
     * 128。然后将ubsize累加之后计算是否总的ub大小够放， 如果够的话直接使用当前nIn，否则使用64 * 128，64 *
     * 64的拼凑方式。 */
    ge::graphStatus DoCoresSplitTiling()
    {
        auto bestBasicBlockSize = BEST_BASIC_BLOCK_SIZE;
        auto bestBasicBlockNum = BEST_BASIC_BLOCK_NUM;
        if (td_->opInfo.get_sQ() >= 4 ? true : false) {
            bestBasicBlockSize = BEST_BASIC_BLOCK_SIZE / 4 * 3;
            bestBasicBlockNum = BEST_BASIC_BLOCK_NUM / 4 * 3;
        }
        int64_t nGSqSkvAlign =
            td_->opInfo.get_n() * td_->opInfo.get_g() * td_->opInfo.get_sQ() * td_->opInfo.get_sKVAlign();
        CHECK_ZERO(nGSqSkvAlign);
        CHECK_ZERO(td_->opInfo.get_vecCalcDTypeSize());
        int64_t bIn = bestBasicBlockSize / (nGSqSkvAlign * td_->opInfo.get_vecCalcDTypeSize());
        int64_t b = td_->opInfo.get_b();
        if (bIn > b) {
            bIn = b;
        }
        td_->singleCoreParams.set_bIn(bIn);
        int64_t cvRatio = CV_RATIO;
        int64_t bCvRatioInner = bIn * cvRatio;
        if (bCvRatioInner > b) {
            bCvRatioInner = bIn;
            cvRatio = 1;
        }
        td_->singleCoreParams.set_bCvInner(bCvRatioInner);
        td_->singleCoreParams.set_bCvRatio(cvRatio);

        bool ret = CheckArgsLegal(bIn);
        OP_CHECK_IF(!ret,
                   OP_LOGW(context_, "check args fail."),
                   return ge::GRAPH_PARAM_INVALID);
        int64_t inputDTypeSize = td_->opInfo.get_inputDTypeSize();
        int64_t bInNGSq = bIn * td_->opInfo.get_n() * td_->opInfo.get_g() * td_->opInfo.get_sQ();
        CHECK_ZERO(bInNGSq);
        int64_t dMaxAlign = Align(bestBasicBlockNum / bInNGSq, FP16_BLOCK_ELES);

        uint32_t clcDInner = FP16_BLOCK_ELES;
        // 最大不能超过32k，所以需要减16B
        if (dMaxAlign > FP16_BLOCK_ELES) {
            clcDInner = static_cast<uint32_t>(dMaxAlign - FP16_BLOCK_ELES);
        }
        int64_t dSize = (td_->opInfo.get_d() / clcDInner) + ((td_->opInfo.get_d() % clcDInner == 0) ? 0 : 1);
        int64_t dInnerTail = td_->opInfo.get_d() - (dSize - 1) * clcDInner;
        CHECK_ZERO(inputDTypeSize);
        int64_t dInnerTailAlign = Align(dInnerTail * inputDTypeSize, BLOCK_BYTE) / inputDTypeSize;

        td_->singleCoreParams.set_clcDInner(clcDInner);
        td_->singleCoreParams.set_dSize(dSize);
        td_->singleCoreParams.set_dInnerTail(dInnerTail);
        td_->singleCoreParams.set_dInnerTailAlign(dInnerTailAlign);

        int64_t bOut = CeilCommon(td_->opInfo.get_b(), bCvRatioInner);
        td_->splitCoreParams.set_bOut(bOut);

        td_->singleCoreParams.set_singleCoreBatchRange(
            CeilCommon(td_->splitCoreParams.get_bOut(), aicoreParams_.numBlocks));
        td_->splitCoreParams.set_usedCoreNum(
            CeilCommon(td_->splitCoreParams.get_bOut(), td_->singleCoreParams.get_singleCoreBatchRange()));

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoInCoreTiling()
    {
        td_->singleCoreParams.set_subRange(CeilCommon(td_->opInfo.get_sQ(), PER_SUB_RANGE));
        td_->singleCoreParams.set_subMask(SINGLE_VEC_INST_DATASIZE / td_->opInfo.get_vecCalcDTypeSize());
        td_->singleCoreParams.set_subMaskTail((td_->opInfo.get_sQ() % PER_SUB_RANGE) *
                                             (BLOCK_BYTE / td_->opInfo.get_vecCalcDTypeSize()));
        td_->singleCoreParams.set_sKVAlignBlockNum(td_->opInfo.get_sKVAlign() * FP32_BYTES_NUM / BLOCK_BYTE);

        int64_t rightPadding = td_->opInfo.get_sKVAlign() - td_->opInfo.get_sKV();
        int64_t dstStride = 0;

        // kernel是按f32计算，如果rightpadding大于8，padding超过32B，会有问题，因此需要额外处理
        if (td_->opInfo.get_sKVAlign() - td_->opInfo.get_sKV() >= FP32_BLOCK_ELES) {
            dstStride = 1;
            rightPadding = td_->opInfo.get_sKVAlign() - td_->opInfo.get_sKV() - FP32_BLOCK_ELES;
        }
        td_->singleCoreParams.set_dstStride(dstStride);
        td_->singleCoreParams.set_rightPadding(rightPadding);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoMulsTiling()
    {
        OP_LOGD(context_, "Do muls tiling begin.");
        int64_t dAlign = (td_->opInfo.get_d() + 15) / 16 * 16;
        int64_t allNumQuery =
            td_->opInfo.get_b() * td_->opInfo.get_n() * td_->opInfo.get_g() * td_->opInfo.get_sQ() * dAlign;
        int64_t allNumKv = td_->opInfo.get_b() * td_->opInfo.get_n() * td_->opInfo.get_sKV() * dAlign;

        uint32_t usedCoreNum = td_->splitCoreParams.get_usedCoreNum();
        if (usedCoreNum == 0U) {
            return ge::GRAPH_PARAM_INVALID;
        }

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
            nzReservedSize = static_cast<uint32_t>(dAlign / C0_SIZE) * BLOCK_SIZE * POST_NZ_RESERVED_N;
            postUbBaseSize = (aicoreParams_.ubSize - 2 * nzReservedSize) / curPostCoexNode / BUFFER_NUM /  // 开DB预留2份nzReservedSize
                                 BASE_LEN_256 * BASE_LEN_256;
            qPostBaseNum = postUbBaseSize / FP16_BYTES_NUM / dAlign * td_->opInfo.get_d();
        }
        OP_CHECK_IF(qPostBaseNum == 0,
                   OP_LOGW(context_, "qPostBaseNum is 0."),
                   return ge::GRAPH_PARAM_INVALID);
        int64_t qPostBlockTotal = allNumQuery / dAlign * td_->opInfo.get_d();
        int64_t qSizeAlign = (qPostBlockTotal + static_cast<int64_t>(BASE_LEN_256) - 1LL) /
                             static_cast<int64_t>(WORKSPACE_ALIGN_SIZE) * static_cast<int64_t>(WORKSPACE_ALIGN_SIZE) *
                             FP16_BYTES_NUM;
        int64_t qPostTailNumTmp = qPostBlockTotal % static_cast<int64_t>(qPostBaseNum);
        int64_t qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
        int64_t qPostBlockOuterTotal =
            (qPostBlockTotal + static_cast<int64_t>(qPostBaseNum) - 1) / static_cast<int64_t>(qPostBaseNum);
        int64_t qPostBlockFactor =
            (qPostBlockOuterTotal + static_cast<int64_t>(usedCoreNum) - 1) / static_cast<int64_t>(usedCoreNum);

        int64_t kvPostBaseNum = static_cast<int64_t>(qPostBaseNum);
        OP_CHECK_IF(kvPostBaseNum == 0,
                   OP_LOGW(context_, "kvPostBaseNum is 0."),
                   return ge::GRAPH_PARAM_INVALID);
        int64_t kvPostBlockTotal = allNumKv / dAlign * td_->opInfo.get_d();;
        int64_t kvSizeAlign = (kvPostBlockTotal + static_cast<int64_t>(WORKSPACE_ALIGN_SIZE) - 1) /
                              static_cast<int64_t>(WORKSPACE_ALIGN_SIZE) * static_cast<int64_t>(WORKSPACE_ALIGN_SIZE) *
                              FP16_BYTES_NUM;
        int64_t kvPostTailNumTmp = kvPostBlockTotal % kvPostBaseNum;
        int64_t kvPostTailNum = kvPostTailNumTmp == 0 ? kvPostBaseNum : kvPostTailNumTmp;
        int64_t kvPostBlockOuterTotal = (kvPostBlockTotal + kvPostBaseNum - 1) / kvPostBaseNum;
        int64_t kvPostBlockFactor =
            (kvPostBlockOuterTotal + static_cast<int64_t>(usedCoreNum) - 1) / static_cast<int64_t>(usedCoreNum);

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

        int64_t allNumDropGm = td_->opInfo.get_b() * td_->opInfo.get_n() * td_->opInfo.get_g() * td_->opInfo.get_sQ() *
                               td_->opInfo.get_sKVAlign();
        int64_t allNumMulGm = td_->opInfo.get_b() * td_->opInfo.get_n() * td_->opInfo.get_g() * td_->opInfo.get_sQ() *
                              td_->opInfo.get_sKVAlign();

        // CV并行实现，需要申请双倍的bmm345的输入空间
        td_->opInfo.set_dropGmWorkspaceLen(2 * CeilCommon(allNumDropGm * FP16_BYTES_NUM, WORKSPACE_ALIGN_SIZE) *
                                          WORKSPACE_ALIGN_SIZE);
        td_->opInfo.set_mulGmWorkspaceLen(2 * CeilCommon(allNumMulGm * FP16_BYTES_NUM, WORKSPACE_ALIGN_SIZE) *
                                         WORKSPACE_ALIGN_SIZE);

        td_->opInfo.set_dqWorkspaceLen(CeilCommon(allNumQuery * FP32_BYTES_NUM, WORKSPACE_ALIGN_SIZE) *
                                      WORKSPACE_ALIGN_SIZE);
        td_->opInfo.set_dkWorkspaceLen(CeilCommon(allNumKv * FP32_BYTES_NUM, WORKSPACE_ALIGN_SIZE) *
                                      WORKSPACE_ALIGN_SIZE);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetMm1AndMm2Tiling(matmul_tiling::MatmulApiTiling &mm1AndMm2, int64_t bIn,
                                       AscendC::tiling::TCubeTiling* mm1AndMm2Tiling)
    {
        mm1AndMm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_FLOAT16, false);
        mm1AndMm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_FLOAT16, true);
        mm1AndMm2.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_FLOAT16);
        mm1AndMm2.SetShape(td_->opInfo.get_sQ(), td_->opInfo.get_sKV(), td_->opInfo.get_d());
        mm1AndMm2.SetOrgShape(td_->opInfo.get_sQ(), td_->opInfo.get_sKV(), td_->opInfo.get_d());
        mm1AndMm2.SetALayout(td_->opInfo.get_b(), td_->opInfo.get_sQ(), td_->opInfo.get_n(), td_->opInfo.get_g(),
                             td_->opInfo.get_d());
        mm1AndMm2.SetBLayout(td_->opInfo.get_b(), td_->opInfo.get_sKV(), td_->opInfo.get_n(), 1, td_->opInfo.get_d());
        mm1AndMm2.SetCLayout(td_->opInfo.get_b(), td_->opInfo.get_sQ(), td_->opInfo.get_n(), td_->opInfo.get_g(),
                             td_->opInfo.get_sKV());

        uint32_t layout = td_->opInfo.get_inputLayout();
        if (layout == static_cast<uint32_t>(InputLayout::BSH) || layout == static_cast<uint32_t>(InputLayout::BSND)) {
            mm1AndMm2.SetBatchNum(td_->opInfo.get_n() * td_->opInfo.get_g());
        } else {
            // SBH BNSD
            mm1AndMm2.SetBatchNum(bIn * td_->singleCoreParams.get_bCvRatio() * td_->opInfo.get_n() * td_->opInfo.get_g());
        }
        if (mm1AndMm2.GetTiling(*mm1AndMm2Tiling) != 0) {
            return ge::GRAPH_PARAM_INVALID;
        }

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetMm31Tiling(matmul_tiling::MatmulApiTiling &mm31, int64_t bIn,
                                  AscendC::tiling::TCubeTiling* mm31Tiling)
    {
        mm31.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                      false);
        mm31.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                      false);
        mm31.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
        mm31.SetShape(td_->opInfo.get_sQ(), td_->opInfo.get_d(), td_->opInfo.get_sKV());
        mm31.SetOrgShape(td_->opInfo.get_sQ(), td_->opInfo.get_d(), td_->opInfo.get_sKV());
        mm31.SetALayout(td_->opInfo.get_b(), td_->opInfo.get_sQ(), td_->opInfo.get_n(), td_->opInfo.get_g(),
                        td_->opInfo.get_sKV());
        mm31.SetBLayout(td_->opInfo.get_b(), td_->opInfo.get_sKV(), td_->opInfo.get_n(), 1, td_->opInfo.get_d());
        mm31.SetCLayout(td_->opInfo.get_b(), td_->opInfo.get_sQ(), td_->opInfo.get_n(), td_->opInfo.get_g(),
                        td_->opInfo.get_d());

        uint32_t layout = td_->opInfo.get_inputLayout();
        if (layout == static_cast<uint32_t>(InputLayout::BSH) || layout == static_cast<uint32_t>(InputLayout::BSND)) {
            mm31.SetBatchNum(td_->opInfo.get_n() * td_->opInfo.get_g());
        } else {
            // SBH BNSD
            mm31.SetBatchNum(bIn * td_->singleCoreParams.get_bCvRatio() * td_->opInfo.get_n() * td_->opInfo.get_g());
        }
        if (mm31.GetTiling(*mm31Tiling) != 0) {
            return ge::GRAPH_PARAM_INVALID;
        }

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetMm32AndMm4Tiling(matmul_tiling::MatmulApiTiling &mm32AndMm4, int64_t bIn,
                                        AscendC::tiling::TCubeTiling* mm32AndMm4Tiling)
    {
        mm32AndMm4.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_FLOAT16, true);
        mm32AndMm4.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_FLOAT16, false);
        mm32AndMm4.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_FLOAT16);
        mm32AndMm4.SetShape(td_->opInfo.get_sKV(), td_->opInfo.get_d(), td_->opInfo.get_sQ());
        mm32AndMm4.SetOrgShape(td_->opInfo.get_sKV(), td_->opInfo.get_d(), td_->opInfo.get_sQ());
        mm32AndMm4.SetALayout(td_->opInfo.get_b(), td_->opInfo.get_sKV(), td_->opInfo.get_n(), td_->opInfo.get_g(),
                              td_->opInfo.get_sQ());
        mm32AndMm4.SetBLayout(td_->opInfo.get_b(), td_->opInfo.get_sQ(), td_->opInfo.get_n(), td_->opInfo.get_g(),
                              td_->opInfo.get_d());
        mm32AndMm4.SetCLayout(td_->opInfo.get_b(), td_->opInfo.get_sKV(), td_->opInfo.get_n(), 1, td_->opInfo.get_d());

        uint32_t layout = td_->opInfo.get_inputLayout();
        if (layout == static_cast<uint32_t>(InputLayout::BSH) || layout == static_cast<uint32_t>(InputLayout::BSND)) {
            mm32AndMm4.SetBatchNum(td_->opInfo.get_n() * td_->opInfo.get_g());
        } else {
            // SBH BNSD
            mm32AndMm4.SetBatchNum(bIn * td_->singleCoreParams.get_bCvRatio() * td_->opInfo.get_n() * td_->opInfo.get_g());
        }
        if (mm32AndMm4.GetTiling(*mm32AndMm4Tiling) != 0) {
            return ge::GRAPH_PARAM_INVALID;
        }
        return ge::GRAPH_SUCCESS;
    }

    // 4、计算高阶API的tiling
    ge::graphStatus DoLibApiTiling() override
    {
        // mm tiling
        OP_LOGD("FAG_SPLIT_B", "DoLibApiTiling begin.");
        matmul_tiling::MatmulApiTiling mm1AndMm2;
        ge::graphStatus ret = SetMm1AndMm2Tiling(mm1AndMm2, td_->singleCoreParams.get_bIn(), &td_->mm1AndMm2TilingData);
        // 如果mm参数设置失败，则流入到下一个模板
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                  OP_LOGW(context_,
                            "Failed to set tiling parameters of mm1 and mm2."),
                  return ret);

        matmul_tiling::MatmulApiTiling mm31;
        ret = SetMm31Tiling(mm31, td_->singleCoreParams.get_bIn(), &td_->mm31TilingData);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                  OP_LOGW(context_,
                            "Failed to set tiling parameters of mm31."),
                  return ret);

        matmul_tiling::MatmulApiTiling mm32AndMm4;
        ret = SetMm32AndMm4Tiling(mm32AndMm4, td_->singleCoreParams.get_bIn(), &td_->mm32AndMm4TilingData);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                  OP_LOGW(context_,
                            "Failed to set tiling parameters of mm32 and mm41."),
                  return ret);

        // vector tiling
        auto softmaxShape =
            Shape({td_->singleCoreParams.get_bIn() * td_->opInfo.get_g() * td_->opInfo.get_n() * td_->opInfo.get_sQ(),
                   td_->opInfo.get_sKVAlign()});
        uint32_t softmaxComputeSize = td_->splitCoreParams.get_apiClcQueueSize();
        SoftMaxTilingFunc(softmaxShape, sizeof(float), softmaxComputeSize, td_->softmaxTilingData);
        auto softmaxGradShape =
            Shape({td_->singleCoreParams.get_bIn() * td_->opInfo.get_n() * td_->opInfo.get_g() * td_->opInfo.get_sQ(),
                   td_->singleCoreParams.get_clcDInner()});
        int64_t softmaxGradComputeSize = td_->splitCoreParams.get_apiClcQueueSize();
        SoftMaxGradTilingFunc(softmaxGradShape, td_->opInfo.get_vecCalcDTypeSize(), softmaxGradComputeSize,
                              td_->softmaxGradTilingData, true);
        return ge::GRAPH_SUCCESS;
    }

    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override
    {
        /* atttention mask可能需要额外的workspace */
        uint32_t sysLen = WORK_SPACE_BASE_CAL;
        uint32_t syncLen = SYNC_LEN;
        int64_t mm1WorkspaceLen = td_->singleCoreParams.get_bIn() * td_->singleCoreParams.get_bCvRatio() *
                                  td_->opInfo.get_n() * td_->opInfo.get_g() * td_->opInfo.get_sQ() *
                                  td_->opInfo.get_sKVAlign() * td_->opInfo.get_vecCalcDTypeSize() *
                                  td_->splitCoreParams.get_usedCoreNum();
        int64_t mm2WorkspaceLen = mm1WorkspaceLen;
        int64_t dqWorkspaceLen = td_->opInfo.get_dqWorkspaceLen();
        int64_t dkWorkspaceLen = td_->opInfo.get_dkWorkspaceLen();
        int64_t dropOutWorkspaceLen = td_->opInfo.get_dropoutWorkspaceLen();
        int64_t dropGmWorkspaceLen = td_->opInfo.get_dropGmWorkspaceLen();
        int64_t mulGmWorkspaceLen = td_->opInfo.get_mulGmWorkspaceLen();

        int64_t workspaceOffsets = static_cast<int64_t>(syncLen) + dropOutWorkspaceLen + mm1WorkspaceLen + mm2WorkspaceLen;
        td_->postTilingData.set_dqWorkSpaceOffset(workspaceOffsets);

        workspaceOffsets = workspaceOffsets + td_->opInfo.get_dqWorkspaceLen();
        td_->postTilingData.set_dkWorkSpaceOffset(workspaceOffsets);

        workspaceOffsets = workspaceOffsets + td_->opInfo.get_dkWorkspaceLen();
        td_->postTilingData.set_dvWorkSpaceOffset(workspaceOffsets);

        workspaceSize_ = sysLen + syncLen + dropOutWorkspaceLen + mm1WorkspaceLen + mm2WorkspaceLen + dqWorkspaceLen +
                         dkWorkspaceLen + dropGmWorkspaceLen + mulGmWorkspaceLen;
        td_->opInfo.set_syncLen(syncLen);
        td_->opInfo.set_mm1WorkspaceLen(mm1WorkspaceLen);
        td_->opInfo.set_mm2WorkspaceLen(mm2WorkspaceLen);
        return ge::GRAPH_SUCCESS;
    }

    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override
    {
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

REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttentionScoreGrad, FlashAttentionScoreGradUbngs1s2BbTiling,
                                         std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}),
                                         10000);

} // namespace optiling
