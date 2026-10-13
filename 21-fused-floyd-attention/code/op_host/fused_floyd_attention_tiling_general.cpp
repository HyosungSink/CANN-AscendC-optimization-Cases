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
 * \file fused_floyd_attention_tiling_general.cpp
 * \brief
 */

#include <numeric>
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_type.h"
#include "tiling_base/data_copy_transpose_tiling.h"
#include "fused_floyd_attention_tiling_common.h"
#include "fused_floyd_attention_tiling.h"
#include "err/ops_err.h"
using namespace Ops::Transformer::OpTiling;

namespace optiling {
namespace FFA {
const uint32_t BYTE_BLOCK = 32;
const int64_t FRACTAL_NUM = 16L;
const int64_t S1_VEC2_BASE_SIZE_MAX = 512L;

const size_t ATTEN_OUT_INDEX = 2UL;
const int64_t BMM_SOFTMAX_RATIO = 4L;
const int64_t HIGH_PERF_BUFFER_NUM = 6L;
const int64_t HIGH_PERF_API_BUFFER_MULTIPLE = 2L;

constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;
const int64_t MAX_VAR_LEN_SEQ_LEN = 4096L;
const int64_t D_SPECIFIC_SIZE = 64L;
const int64_t S2_NZTOND_SIZE_64 = 64L;
const int64_t SPACE_NUM_2 = 2L;
const int64_t SPACE_NUM_3 = 3L;
const int64_t SPACE_NUM_4 = 4L;
const int64_t L1REUSE_D_Limit = 128L;
const int64_t L1REUSE_BNG_Limit = 10L;
const int64_t L1REUSE_S2_Limit_2048 = 2048L;
const int64_t L1REUSE_S2_LIMIT_256 = 256;
const int64_t L1REUSE_S2_LIMIT_4032 = 4032;
const int64_t AICAIV_RATIO_2 = 2L;
const int64_t DATA_TYPE_FP32 = 4L;
enum LayoutType : uint8_t {
    None = 0,
    LAYOUT_BSH = 1,
    LAYOUT_BSND = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
};

enum AttenMaskShapeType : uint8_t {
    ATTEN_B_N2_G_S1_S2 = 0,
    ATTEN_B_1_1_S1_S2 = 1,
    ATTEN_1_1_1_S1_S2 = 2,
    ATTEN_B_N2_1_1_S2 = 3,
    ATTEN_1_1_1_T_T = 99,
};

enum PseShapeType : uint8_t {
    PSE_B_N2_G_S1_S2 = 0,
    PSE_B_N2_G_1_S2 = 1,
    PSE_B_N2_G_SLOPE,
    PSE_1_N2_G_SLOPE
};

enum SparseMode : uint8_t {
    NO_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    PREFIX,
    PREFIX_COMPRESS,
    RIGHT_DOWN_CAUSAL_BAND,
    BAND_LEFT_UP_CAUSAL
};

enum AttenMaskCompressMode : uint8_t {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE,
    RIGHT_DOWN_CAUSAL_MODE,
    BAND_MODE,
    PREFIX_MODE,
    RIGHT_DOWN_CAUSAL_BAND_MODE = 5,
    BAND_LEFT_UP_CAUSAL_MODE
};

enum ImplMode : uint8_t {
    AA_HIGH_PRECISION = 0,
    AA_HIGH_PERFORMANCE = 1,
    AA_INVALID_LINE_HIGH_PRECISION = 2
};

enum PseType : uint8_t {
    PSE_OUTER_MUL_ADD_TYPE = 0, // v2 default
    PSE_OUTER_ADD_MUL_TYPE, // v1 current usage
    PSE_INNER_MUL_ADD_TYPE,
    PSE_INNER_MUL_ADD_SQRT_TYPE,
    PSE_INVALID_TYPE
};


enum PseEncodeType : uint8_t {
    PES_ENCODE_NONE = 0,
    PSE_ENCODE_ALIBI_S2_FULL = 0x11, // shape: (1024, S2)
};

template <typename T> static T AlignUp(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T> static T AlignDown(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return num1 / num2 * num2;
}

template <typename T> static T CeilDivision(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T> static T CeilDiv(const T n1, const T n2)
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? (((n1 - 1) / n2) + 1) : n1;
}

template <typename T> static T CalcTailSize(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

class TilingKey {
public:
    TilingKey() : splitS1(0), splitS2(0), splitD(0), dtype(0), layoutType(0), sparseType(0), reserved(0)
    {
    }

    void Reset()
    {
        splitS1 = 0;
        splitS2 = 0;
        splitD = 0;
        dtype = 0;
        layoutType = 0;
        sparseType = 0;
        reserved = 0;
    }

    uint32_t GetRawTilingKey() const
    {
        return *(reinterpret_cast<const uint32_t *>(this));
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << " splitS1: " << splitS1 << " splitS2: " << splitS2 << " splitD: " << splitD;
        return ss.str();
    }

    uint32_t splitS1    : 1;
    uint32_t splitS2    : 1;
    uint32_t splitD     : 1;
    uint32_t dtype      : 2;
    uint32_t layoutType : 2;
    uint32_t sparseType : 2;
    uint32_t reserved   : 23; // to fullfil 32 bit, if add new template bit then decrease this number
};

inline bool operator==(const TilingKey &left, const TilingKey &right)
{
    return left.GetRawTilingKey() == right.GetRawTilingKey();
}

using TemplateType = TilingKey;

class BufferNum {
public:
    // sum and max always use fp32, shape is (S1, 1), inner axis align 32B.
    size_t bufferS1S2Num; // unit: input dtype
    size_t bufferS1DNum;
    size_t bufferExpNum; // unit: input dtype, shape: [S1, 1], inner axis align 32B.
};

class FusedFloydAttentionTilingBase : public TilingBaseClass {
public:
    explicit FusedFloydAttentionTilingBase(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~FusedFloydAttentionTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    [[nodiscard]] gert::TilingContext *GetContext()
    {
        return context_;
    }
    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    void SetSparseTilingInfo(SparseEnum &sparseType);
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override = 0;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    virtual void GetBufferNum(BufferNum &bufferNum) const = 0;

    void Reset();

    virtual int64_t GetNRatio();

    virtual int64_t GetMinS1BasicBlock() const
    {
        return std::min(64L, alignedS1);
    }

    virtual bool IsTemplateMatched() const
    {
        return expectTemplate == actualTemplate;
    }

    ge::graphStatus CheckContext();
    virtual bool AnalyzeDtype();
    bool AnalyzeAttrs();
    bool AnalyzeLayout();
    bool Analyze4DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &attenShape);
    bool AnalyzeOptionalInput();
    bool MatchTemplate();
    virtual void CalcS1S2BasicBlock() = 0;
    virtual void CalcDBasicBlock() = 0;
    int64_t CalcMaxS1BasicBlockSize(int64_t actualD, const BufferNum &bufferNum) const;
    int64_t CalcMaxS2BasicBlockSize(const BufferNum &bufferNum, int64_t tmpS1BasicBlock) const;
    virtual bool CalcUBSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock) = 0;
    bool IsBasicBlockInSoftMax(const ge::Shape &shape) const;
    virtual bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                    matmul_tiling::MatmulApiTiling &bmm1);
    virtual bool SetBmm1k1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, 
                            matmul_tiling::MatmulApiTiling &bmm1) = 0;
    virtual bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                    matmul_tiling::MatmulApiTiling &bmm2) = 0;

    virtual bool SetBmm2v2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                    matmul_tiling::MatmulApiTiling &bmm2v2) = 0;

    bool SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                         matmul_tiling::MatmulApiTiling &bmm1, matmul_tiling::MatmulApiTiling &bmm1k1,
                         matmul_tiling::MatmulApiTiling &bmm2, matmul_tiling::MatmulApiTiling &bmm2v2);
    bool SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock);
    bool CaclBmmBatch(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock);
    virtual void SetCoreParams();
    virtual void SetMultiBatchCoreParams();
    virtual void SetMultiCoreParams();
    virtual void SetSoftMaxTiling();
    void SetDataCopyTransposeTiling();
    virtual void SetTensorSizeParams();
    uint32_t aivNum;
    uint32_t aicNum;
    int64_t apiMaxUBSize = 0;

    matmul_tiling::DataType bmmDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

    ge::DataType inputDtype;
    int64_t inputDtypeBytes;
    int64_t calcTypeSize;

    bool isHighPercision; // fp16 high percision mode

    DtypeEnum tilingKeyDType;
    LayoutType tilingKeyLayout;
    ImplMode implMode;
    CubeFormatEnum tilingKeyBmm1Format = CubeFormatEnum::ND;
    CubeInputSourceEnum tilingKeyBmm1Source = CubeInputSourceEnum::GM;
    CubeInputSourceEnum tilingKeyBmm2Source = CubeInputSourceEnum::GM;

    int64_t bSize;
    int64_t gSize;
    int64_t dSize;
    int64_t n1Size;
    int64_t n2Size;
    int64_t s1Size;
    int64_t s2Size;
    int64_t atten_bSize;
    int64_t s1StrideSize; // query Shape S inner axes, for bmm1
    int64_t s2StrideSize; // key Shape S inner axes, for bmm1
    int64_t preTokens;
    int64_t nextTokens;
    int64_t s1SparseValidSize;
    int64_t s2SparseValidSize;
    int64_t sparseMode;
    int64_t pseType;
    int64_t pseAlibiBaseS1;
    int64_t pseAlibiBaseS2;
    int64_t qStartIdx;
    int64_t kvStartIdx;
    int64_t maxS1Val;
    int64_t minS1Val;
    int64_t accumS1;
    int64_t accumS1BlockNum;
    int64_t dropTotalSize;
    int64_t maxS2Val;
    int64_t minS2Val;
    int64_t accumS2;
    int64_t bandIndex;
    std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> actualSeqLenData;
    std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> actualSeqLenKvData;
    float keepProb;
    float scaleValue;
    uint8_t attenMaskCompressMode;
    uint8_t pseExistFlag;
    uint8_t attenMaskExistFlag;
    uint8_t dropMaskExistFlag;

    int64_t alignedS1;
    int64_t alignedS2;
    int64_t alignedD;

    int64_t s1BasicBlock;
    int64_t s2BasicBlock;
    // FIXED
    int64_t n2BasicBlock;
    int64_t dBasicBlock;
    int64_t batchBasic;
    int64_t nRatio;

    int64_t minUsedUBSize;
    int64_t maxValidS2Len;

    const char *templateName = "base";
    const char *opName;
    const char *inputLayout;
    const int64_t *prefixNData;
    TemplateType expectTemplate;
    TemplateType actualTemplate;

    bool isSparseValidSizeAligned = false;
    bool hasPse = false;
    bool hasAttenMask = false;
    bool hasDropOut = false;
    FusedFloydAttentionGeneralTilingData tilingData;
};

int64_t FusedFloydAttentionTilingBase::GetNRatio()
{
    return BMM_SOFTMAX_RATIO;
}

ge::graphStatus FusedFloydAttentionTilingBase::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const FlashAttentionScoreGradCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "compileInfoPtr is null."),
                   return ge::GRAPH_FAILED);
        aivNum = compileInfoPtr->aivNum;
        aicNum = compileInfoPtr->aicNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aivNum = ascendcPlatform.GetCoreNumAiv();
        aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    }
    OP_LOGI(context_, "get platform from compileInfo. aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu).",
              aivNum, aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0cSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedFloydAttentionTilingBase::CheckContext()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    size_t *workspaces = context_->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleValuePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);

    auto queryShape = context_->GetInputShape(0);
    auto queryDesc = context_->GetInputDesc(0);
    auto keyShape = context_->GetInputShape(1);
    auto attenOutShape = context_->GetOutputShape(ATTEN_OUT_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context_, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attenOutShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    return ge::GRAPH_SUCCESS;
}

void FusedFloydAttentionTilingBase::SetSparseTilingInfo(SparseEnum &sparseType)
{
    auto &inputParams = tilingData.inputParams;
    inputParams.set_attenMaskCompressMode(attenMaskCompressMode);
    inputParams.set_sparseType(static_cast<uint8_t>(sparseType));
}


ge::graphStatus FusedFloydAttentionTilingBase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOGD(opName, "TilingContext: %s.", GetTilingContextDebugStr().c_str());
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid context."),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout() || !AnalyzeOptionalInput(),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to analyze context info."), return ge::GRAPH_FAILED);

    alignedS1 = AlignUp(s1Size, FRACTAL_NUM);
    alignedS2 = AlignUp(s2Size, FRACTAL_NUM);
    alignedD = AlignUp(dSize, FRACTAL_NUM);

    OP_CHECK_IF(alignedS1 <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid alignedS1 %ld.", alignedS1),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(alignedS2 <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid alignedS2 %ld.", alignedS2),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(alignedD <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid alignedD %ld.", alignedD),
        return ge::GRAPH_FAILED);

    auto &inputParams = tilingData.inputParams;
    inputParams.set_bSize(bSize);
    inputParams.set_n2Size(n2Size);
    inputParams.set_gSize(gSize);
    inputParams.set_s1Size(s1Size);
    inputParams.set_s2Size(s2Size);
    inputParams.set_dSize(dSize);
    inputParams.set_attenbSize(atten_bSize);
    inputParams.set_keepProb(keepProb);
    inputParams.set_scaleValue(scaleValue);
    inputParams.set_alignedS2(alignedS2);
    inputParams.set_pseType(static_cast<uint32_t>(pseType));
    OP_LOGD(context_, "input params: bn2gs1s2d[%ld, %ld, %ld, %ld, %ld, %ld], keepProb[%f], scaleValue[%f],"
              "pseType:%ld.", bSize, n2Size, gSize, s1Size, s2Size, dSize, keepProb, scaleValue, pseType);

    return ge::GRAPH_SUCCESS;
}

void FusedFloydAttentionTilingBase::Reset()
{
    tilingData.SetDataPtr(context_->GetRawTilingData()->GetData());
    apiMaxUBSize = 0;

    bmmDtype = matmul_tiling::DataType::DT_FLOAT;
    bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
    bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

    inputDtype = ge::DT_FLOAT16;
    inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
    calcTypeSize = inputDtypeBytes;

    tilingKeyDType = DtypeEnum::FLOAT16;
    tilingKeyLayout = LayoutType::LAYOUT_BNSD;
    tilingKeyBmm1Format = CubeFormatEnum::ND;
    tilingKeyBmm1Source = CubeInputSourceEnum::GM;

    bSize = 0LL;
    gSize = 0LL;
    dSize = 0LL;
    n1Size = 0LL;
    n2Size = 0LL;
    s1Size = 0LL;
    s2Size = 0LL;
    atten_bSize = 0LL;
    maxS1Val = 0LL;
    minS1Val = 0LL;
    accumS1 = 0LL;
    accumS2 = 0LL;
    bandIndex = 0LL;
    dropTotalSize = 0LL;
    maxS2Val = 0LL;
    minS2Val = 0LL;

    s1StrideSize = 0LL;
    s2StrideSize = 0LL;
    preTokens = std::numeric_limits<int32_t>::max();
    nextTokens = std::numeric_limits<int32_t>::max();
    sparseMode = static_cast<int64_t>(NO_MASK);
    pseType = PSE_OUTER_ADD_MUL_TYPE;
    pseAlibiBaseS1 = 0;
    pseAlibiBaseS2 = 0;
    qStartIdx = 0;
    kvStartIdx = 0;
    keepProb = 1.0f;
    scaleValue = 1.0f;
    pseExistFlag = 0;
    attenMaskCompressMode = NO_COMPRESS_MODE;
    attenMaskExistFlag = 0;
    dropMaskExistFlag = 0;
    isHighPercision = true;

    alignedS1 = 0LL;
    alignedS2 = 0LL;
    alignedD = 0LL;

    s1BasicBlock = std::numeric_limits<int64_t>::max();
    s2BasicBlock = std::numeric_limits<int64_t>::max();
    // FIXED
    n2BasicBlock = std::numeric_limits<int64_t>::max();
    dBasicBlock = std::numeric_limits<int64_t>::max();
    nRatio = GetNRatio();

    minUsedUBSize = 0LL;
    maxValidS2Len = 0LL;
    batchBasic = 1LL;

    opName = nullptr;
    inputLayout = nullptr;

    actualTemplate.Reset();
}

bool FusedFloydAttentionTilingBase::AnalyzeDtype()
{
    inputDtype = context_->GetInputDesc(0)->GetDataType();
    inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
    switch (inputDtype) {
        case ge::DT_FLOAT16:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT16;
            bmm1OutDtype = isHighPercision ? matmul_tiling::DataType::DT_FLOAT : matmul_tiling::DataType::DT_FLOAT16;
            tilingKeyDType = isHighPercision ? DtypeEnum::FLOAT16_PRECISION : DtypeEnum::FLOAT16;
            calcTypeSize = isHighPercision ? ge::GetSizeByDataType(ge::DT_FLOAT) : ge::GetSizeByDataType(inputDtype);
            break;
        case ge::DT_FLOAT:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            tilingKeyDType = DtypeEnum::FLOAT32;
            isHighPercision = false;
            calcTypeSize = ge::GetSizeByDataType(inputDtype);
            break;
        case ge::DT_BF16:
            bmmDtype = matmul_tiling::DataType::DT_BF16;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            tilingKeyDType = DtypeEnum::BFLOAT16;
            calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
            isHighPercision = false;
            break;
        default:
            OPS_REPORT_VECTOR_INNER_ERR(opName, "not support input dtype: %s for now",
                                        ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str());
            return false;
    }

    bmm2OutDtype = bmm1OutDtype;
    OP_LOGD(context_, "Get high precision flag: %d.", isHighPercision);
    return true;
}

bool FusedFloydAttentionTilingBase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);

    auto &queryShape = context_->GetInputShape(0)->GetStorageShape();
    auto n1SizePtr = queryShape.GetDim(2);
    inputLayout = "BNSD";

    scaleValue = *scaleValuePtr;
    n1Size = n1SizePtr;

    implMode = ImplMode::AA_HIGH_PRECISION;

    isHighPercision = true; // use default value

    return true;
}

bool FusedFloydAttentionTilingBase::AnalyzeLayout()
{
    auto &queryShape = context_->GetInputShape(0)->GetStorageShape();
    auto &keyShape = context_->GetInputShape(1)->GetStorageShape();
    auto &attenShape = context_->GetInputShape(ATTENTION_MASK_INPUT_INDEX)->GetStorageShape();

    OP_CHECK_IF(!Analyze4DimLayout(queryShape, keyShape, attenShape),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "Get unsupported layout: %s", inputLayout), return false);
    return true;
}


bool FusedFloydAttentionTilingBase::Analyze4DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &attenShape)
{
    // Q:   B H N M D
    // key: B H N K D
    bSize = queryShape.GetDim(0) * queryShape.GetDim(1);
    n2Size = keyShape.GetDim(2); // 2:N
    gSize = 1;
    s1Size = queryShape.GetDim(3); // 3:M
    s2Size = keyShape.GetDim(3); // 3:M
    dSize = queryShape.GetDim(4); // 4:D
    atten_bSize = attenShape.GetDim(0) * attenShape.GetDim(1);
    s1StrideSize = dSize;
    s2StrideSize = dSize;
    tilingData.inputParams.set_layoutType(LAYOUT_BNSD);
    tilingKeyLayout = LayoutType::LAYOUT_BNSD;

    return true;
}


bool FusedFloydAttentionTilingBase::AnalyzeOptionalInput()
{
    auto attenMaskShape = context_->GetOptionalInputShape(ATTENTION_MASK_INPUT_INDEX);
    attenMaskExistFlag = 1;
    tilingData.inputParams.set_attenMaskDataType(1);
    AttenMaskShapeType attenMaskShapeType = ATTEN_B_N2_G_S1_S2;
    auto &attenMaskStorageShape = attenMaskShape->GetStorageShape();
    auto attenMaskDimNum = attenMaskStorageShape.GetDimNum();
    if (attenMaskDimNum == 3) { // 3:shape is B_N2_1_1_S2
        attenMaskShapeType = ATTEN_B_N2_1_1_S2;
    } else if (attenMaskDimNum == 5) { // 5:shape is B_N2_G_S1_S2
        int64_t attenMaskDim1Size = attenMaskStorageShape.GetDim(1);
        int64_t attenMaskDim3Size = attenMaskStorageShape.GetDim(3);

        if (attenMaskDim1Size == 1 && attenMaskDim3Size == 1) {
            attenMaskShapeType = ATTEN_B_N2_1_1_S2;
        } else {
            attenMaskShapeType = ATTEN_B_N2_G_S1_S2;
        }
    } else {
        OP_LOGE(context_, "get unsupported atten_mask shape, dims is %lu!", attenMaskDimNum);
        return false;
    }

    tilingData.inputParams.set_attenMaskShapeType(attenMaskShapeType);
    tilingData.inputParams.set_attenMaskS2Size(attenMaskStorageShape.GetDim(attenMaskDimNum - 1));
    return true;
}

ge::graphStatus FusedFloydAttentionTilingBase::DoOpTiling()
{
    auto &inputParams = tilingData.inputParams;
    OP_LOGD(context_, "[%s]try template[%s]", templateName, expectTemplate.ToString().c_str());
    if (!MatchTemplate()) {
        OP_LOGI(context_,
                  "[%s]not match template[%s], input params: bn2gs1s2d[%ld, %ld, %ld, %ld, %ld, %ld], "
                  "keepProb[%f]",
                  templateName, expectTemplate.ToString().c_str(), inputParams.get_bSize(), inputParams.get_n2Size(),
                  inputParams.get_gSize(), inputParams.get_s1Size(), inputParams.get_s2Size(), inputParams.get_dSize(),
                  inputParams.get_keepProb());
        return ge::GRAPH_PARAM_INVALID;
    }

    SparseEnum sparseType = SparseEnum::ALL;

    SetSparseTilingInfo(sparseType);
    inputParams.set_implMode(implMode);
    if (!isSparseValidSizeAligned) {
        s1SparseValidSize = preTokens;
        s2SparseValidSize = nextTokens;
    }
    SetCoreParams();
    SetMultiCoreParams();
    SetTensorSizeParams();

    return ge::GRAPH_SUCCESS;
}

bool FusedFloydAttentionTilingBase::MatchTemplate()
{
    // UB Size calc logic: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
    BufferNum bufferNum;
    GetBufferNum(bufferNum);

    s1BasicBlock = std::numeric_limits<int64_t>::max();
    s2BasicBlock = std::numeric_limits<int64_t>::max();
    CalcS1S2BasicBlock();

    if (s2BasicBlock == std::numeric_limits<int64_t>::max()) {
        OP_LOGD(context_,
                  "[%s]can't find proper S1S2 basic block for shape: S1[%ld] S2[%ld], D[%ld], minS1BasicBlock[%ld], "
                  "dtype[%s], high precision[%d]",
                  templateName, s1Size, s2Size, dSize, GetMinS1BasicBlock(),
                  ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str(), isHighPercision);
        return false;
    }

    CalcDBasicBlock();
    actualTemplate.splitS1 = s1BasicBlock < alignedS1 ? 1 : 0;
    actualTemplate.splitS2 = s2BasicBlock < alignedS2 ? 1 : 0;
    actualTemplate.splitD = dBasicBlock < alignedD ? 1 : 0;

    if (IsTemplateMatched()) {
        (void)CalcUBSize(s1BasicBlock, s2BasicBlock);
        OP_LOGD(context_, "[%s]final basic block: [%ld, %ld, %ld], match template[%s].", templateName, s1BasicBlock,
                  s2BasicBlock, dBasicBlock, actualTemplate.ToString().c_str());
        return true;
    }

    return false;
}

void FusedFloydAttentionTilingBase::CalcDBasicBlock()
{
    return;
}

int64_t FusedFloydAttentionTilingBase::CalcMaxS1BasicBlockSize(int64_t actualD, const BufferNum &bufferNum) const
{
    // if S2 basic block is min value 16, s1 basic block can reach max value, then we get:
    // s1 * 16 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp =>
    // s1 * (16 * X + D * Y + (expNum + 2) * (32 / sizeof(T))) * sizeof(T) + apiTmp
    // just ignore apiTmp now, consider it at last
    int64_t alignUnit = BYTE_BLOCK / inputDtypeBytes;
    int64_t maxS1BasicBlock =
        aicoreParams_.ubSize / inputDtypeBytes /
        (FRACTAL_NUM * bufferNum.bufferS1S2Num + actualD * bufferNum.bufferS1DNum +
         (bufferNum.bufferExpNum + 2) * alignUnit); // here 2 means FlashSoftMax sum and max output
    return AlignDown(maxS1BasicBlock, FRACTAL_NUM);
}

int64_t FusedFloydAttentionTilingBase::CalcMaxS2BasicBlockSize(const BufferNum &bufferNum,
                                                               int64_t tmpS1BasicBlock) const
{
    // used UB: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
    // if D full load, use alignedD in above formula
    // if D not full load, use S2 basic block var in above formula
    // just ignore apiTmp now, consider it at last
    int64_t tmpS2BasicBlock;
    if (expectTemplate.splitD == 0) {
        // here 2 means FlashSoftMax sum and max output
        tmpS2BasicBlock = (aicoreParams_.ubSize - tmpS1BasicBlock * (bufferNum.bufferExpNum + 2) * BYTE_BLOCK -
                           tmpS1BasicBlock * alignedD * bufferNum.bufferS1DNum * inputDtypeBytes) /
                          (tmpS1BasicBlock * bufferNum.bufferS1S2Num * inputDtypeBytes);
    } else {
        // here 2 means FlashSoftMax sum and max output
        tmpS2BasicBlock = (aicoreParams_.ubSize - tmpS1BasicBlock * (bufferNum.bufferExpNum + 2) * BYTE_BLOCK) /
                          (tmpS1BasicBlock * (bufferNum.bufferS1DNum + bufferNum.bufferS1S2Num) * inputDtypeBytes);
    }
    return std::min(AlignDown(tmpS2BasicBlock, FRACTAL_NUM), alignedS2);
}

bool FusedFloydAttentionTilingBase::IsBasicBlockInSoftMax(const ge::Shape &shape) const
{
    // 2 axes at least
    if (shape.GetDimNum() < 2) {
        return false;
    }

    int64_t lastAxis = shape.GetDim(shape.GetDimNum() - 1);
    // last axis should be less than 2048 and fullfil 64 times
    int64_t basicLastAxis = 64;
    int64_t lastAxisNum = 2048;
    if (lastAxis > lastAxisNum || lastAxis % basicLastAxis != 0) {
        return false;
    }

    int64_t preAxes = 1;
    for (size_t idx = 0; idx < shape.GetDimNum() - 1; ++idx) {
        preAxes *= shape.GetDim(idx);
    }

    // all axes except last one should be 8 times
    return preAxes % 8 == 0;
}

void FusedFloydAttentionTilingBase::SetCoreParams()
{
    auto &coreParams = tilingData.coreParams;
    coreParams.set_s1BaseSize(s1BasicBlock);
    coreParams.set_s1BaseTailSize(CalcTailSize(s1Size, s1BasicBlock));
    coreParams.set_s1OuterSize(CeilDivision(s1Size, s1BasicBlock));
    coreParams.set_s2BaseSize(s2BasicBlock);
    coreParams.set_s2BaseTailSize(CalcTailSize(s2Size, s2BasicBlock));
    coreParams.set_s2OuterSize(CeilDivision(s2Size, s2BasicBlock));
    if (expectTemplate.splitS2 == 1) {
        nRatio = std::min(GetNRatio(), coreParams.get_s2OuterSize());
        coreParams.set_s2OuterSize(CeilDivision(coreParams.get_s2OuterSize(), nRatio));
    } else if (expectTemplate.splitS1 == 1) {
        nRatio = std::min(GetNRatio(), coreParams.get_s1OuterSize());
    } else {
        nRatio = 1;
    }
    coreParams.set_nRatio(nRatio);

    coreParams.set_dBaseSize(dBasicBlock);
    coreParams.set_dBaseTailSize(CalcTailSize(dSize, dBasicBlock));
    coreParams.set_dOuterSize(CeilDivision(dSize, dBasicBlock));
    // 向下取整保证数据量不超32K
    int64_t s1Vec2BaseSize = 8 * 1024 * 2 / (alignedD * inputDtypeBytes);
    coreParams.set_s1Vec2BaseSize(std::min(s1Vec2BaseSize, S1_VEC2_BASE_SIZE_MAX));
    coreParams.set_s1Vec2BaseTailSize(s1Size % coreParams.get_s1Vec2BaseSize());
    SetMultiBatchCoreParams();
}

void FusedFloydAttentionTilingBase::SetMultiBatchCoreParams()
{
    auto &coreParams = tilingData.coreParams;
    coreParams.set_bBaseSize(1);
    coreParams.set_bBaseTailSize(1);
    coreParams.set_bOuterSize(bSize);

    // FIXED
    coreParams.set_n2BaseSize(n2BasicBlock);
    coreParams.set_n2BaseTailSize(CalcTailSize(n2Size, n2BasicBlock));
    coreParams.set_n2OuterSize(CeilDivision(n2Size, n2BasicBlock));

    coreParams.set_gBaseSize(1);
    coreParams.set_gBaseTailSize(1);
    coreParams.set_gOuterSize(gSize);
}

void FusedFloydAttentionTilingBase::SetMultiCoreParams()
{
    auto &multiCoreParams = tilingData.multiCoreParams;
    auto &coreParams = tilingData.coreParams;
    int64_t totalSize = coreParams.get_bOuterSize() * coreParams.get_n2OuterSize() * coreParams.get_gOuterSize() *
                        coreParams.get_s1OuterSize();
    int64_t actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));

    multiCoreParams.set_coreNum(static_cast<int32_t>(actualUsedAivNum));
    multiCoreParams.set_totalSize(totalSize);
    multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualUsedAivNum));
    multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
}

ge::graphStatus FusedFloydAttentionTilingBase::DoLibApiTiling()
{
    if (!SetMatMulTiling(s1BasicBlock, s2BasicBlock)) {
        return ge::GRAPH_FAILED;
    }
    SetSoftMaxTiling();
    SetDataCopyTransposeTiling();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedFloydAttentionTilingBase::PostTiling()
{
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize()); // already check capcity in CheckContext
    auto blockDim = optiling::FFA::FloydCalcTschBlockDim(tilingData.multiCoreParams.get_coreNum(), aicNum, aivNum);
    context_->SetBlockDim(blockDim);
    size_t *workspaces = context_->GetWorkspaceSizes(1);

    OP_LOGD(context_, "[%s] final workspace size:%zu, pseAlibiBaseS1:%ld, pseAlibiBaseS2:%ld.",
              templateName, workspaces[0], pseAlibiBaseS1, pseAlibiBaseS2);
    OP_LOGD(opName, "[%s] tiling data:%s", templateName, GetTilingDataDebugStr().c_str());

    return ge::GRAPH_SUCCESS;
}


bool FusedFloydAttentionTilingBase::SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                       matmul_tiling::MatmulApiTiling &bmm1)
{
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
    bmm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
    bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpS2BasicBlock, s2Size), dSize);
    bmm1.SetOrgShape(s1Size, s2Size, s1StrideSize, s2StrideSize);
    bmm1.SetBias(false);
    if (bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
        return false;
    }
    if (bmm1.SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock) != 0) {
        return false;
    }

    return true;
}

bool FusedFloydAttentionTilingBase::CaclBmmBatch(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock)
{
    int64_t s2BaseSize = tmpS2BasicBlock * tilingData.coreParams.get_nRatio();
    int64_t batchData = n2BasicBlock * dSize * (tmpS1BasicBlock + s2BaseSize) * 2;

    int64_t loops = CeilDiv(batchData, static_cast<int64_t>(aicoreParams_.l1Size));
    if (loops <= 0) {
        OP_LOGE(context_, "Function CaclBmmBatch failed.");
        return false;
    }
    int32_t bmm1Num = n2BasicBlock / loops;
    tilingData.coreParams.set_bmm1Num(bmm1Num);

    int64_t bmmv2Num = aicoreParams_.l1Size / 2 / s2Size / (dSize + n2BasicBlock);
    bmmv2Num = std::min(bmmv2Num, static_cast<int64_t>(4));
    if (bmmv2Num == 3)
    {
        bmmv2Num = 2;
    }
    tilingData.coreParams.set_bmmv2Num(bmmv2Num);

    return true;
}


bool FusedFloydAttentionTilingBase::SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                    matmul_tiling::MatmulApiTiling &bmm1,
                                                    matmul_tiling::MatmulApiTiling &bmm1k1,
                                                    matmul_tiling::MatmulApiTiling &bmm2,
                                                    matmul_tiling::MatmulApiTiling &bmm2v2)
{
    if (!CaclBmmBatch(tmpS1BasicBlock, tmpS2BasicBlock))
    {
        return false;
    }

    if (!SetBmm1TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, bmm1) ||
        !SetBmm2TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, bmm2)) {
        return false;
    }
    
    if (!SetBmm1k1TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, bmm1k1) || 
        !SetBmm2v2TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, bmm2v2)) {
        return false;
    }

    if (bmm1.GetTiling(tilingData.bmm1TilingData) == -1) {
        OP_LOGE(context_, "BMM1 tiling failed.");
        return false;
    }
    tilingData.bmm1TilingData.set_shareMode(0);
    tilingData.bmm1TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.bmm1TilingData.set_shareL0CSize(aicoreParams_.l0cSize);

    if (bmm2.GetTiling(tilingData.bmm2TilingData) == -1) {
        OP_LOGE(context_, "BMM2 tiling failed.");
        return false;
    }
    tilingData.bmm2TilingData.set_shareMode(0);
    tilingData.bmm2TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.bmm2TilingData.set_shareL0CSize(aicoreParams_.l0cSize);

    // k1
    if (bmm1k1.GetTiling(tilingData.bmm1k1TilingData) == -1) {
        OP_LOGE(context_, "BMM1k1 tiling failed.");
        return false;
    }
    tilingData.bmm1k1TilingData.set_shareMode(0);
    tilingData.bmm1k1TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.bmm1k1TilingData.set_shareL0CSize(aicoreParams_.l0cSize);

    // v2
    if (bmm2v2.GetTiling(tilingData.bmm2v2TilingData) == -1) {
        OP_LOGE(context_, "BMM2v2 tiling failed.");
        return false;
    }
    tilingData.bmm2v2TilingData.set_shareMode(0);
    tilingData.bmm2v2TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.bmm2v2TilingData.set_shareL0CSize(aicoreParams_.l0cSize);

    return true;
}

bool FusedFloydAttentionTilingBase::SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock)
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        matmul_tiling::MatmulApiTiling bmm1(ascendcPlatform);
        matmul_tiling::MatmulApiTiling bmm1k1(ascendcPlatform);
        matmul_tiling::MatmulApiTiling bmm2(ascendcPlatform);
        matmul_tiling::MatmulApiTiling bmm2v2(ascendcPlatform);
        return SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, bmm1, bmm1k1, bmm2, bmm2v2);
    } else {
        OP_LOGD(context_, "platform info is null, use default info to generate matmul tiling.");
        matmul_tiling::MatmulApiTiling bmm1;
        matmul_tiling::MatmulApiTiling bmm1k1;
        matmul_tiling::MatmulApiTiling bmm2;
        matmul_tiling::MatmulApiTiling bmm2v2;
        return SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, bmm1, bmm1k1, bmm2, bmm2v2);
    }
}

void FusedFloydAttentionTilingBase::SetSoftMaxTiling()
{
    auto softmaxShape = ge::Shape({batchBasic, std::min(s1BasicBlock, alignedS1), std::min(s2BasicBlock, alignedS2)});

    AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                      tilingData.softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
}

void FusedFloydAttentionTilingBase::SetDataCopyTransposeTiling()
{
    auto &coreParams = tilingData.coreParams;
    auto transposeSrcShape = ge::Shape({coreParams.get_bBaseSize(), 1, std::min(s1BasicBlock, alignedS1),
                                        coreParams.get_gBaseSize() * std::min(dBasicBlock, alignedD)});
    auto transposeDstShape = ge::Shape({bSize, n1Size, s1Size, n1Size * dSize});
    GetDataCopyTransposeTiling(transposeDstShape, transposeSrcShape, inputDtypeBytes, tilingData.transposeTilingData);
}

void FusedFloydAttentionTilingBase::SetTensorSizeParams()
{
    auto &tensorSizeParams = tilingData.tensorSizeParams;
    auto &coreParams = tilingData.coreParams;
    int64_t batchInnerSize = coreParams.get_bBaseSize() * coreParams.get_n2BaseSize() * coreParams.get_gBaseSize();
    tensorSizeParams.set_bmm1ResUbSize(batchInnerSize * s1BasicBlock * s2BasicBlock);
    tensorSizeParams.set_attenMaskUbSize(attenMaskExistFlag * batchInnerSize * s1BasicBlock * s2BasicBlock);
    if (tilingData.inputParams.get_pseShapeType() == PSE_B_N2_G_S1_S2) {
        tensorSizeParams.set_pseUbSize(pseExistFlag * batchInnerSize * s1BasicBlock * s2BasicBlock);
    } else {
        tensorSizeParams.set_pseUbSize(pseExistFlag * batchInnerSize * s2BasicBlock); // PSE_B_N2_G_1_S2
    }

    if (tensorSizeParams.get_attenMaskUbSize() > 0) {
        hasAttenMask = true;
    }
    if (inputDtype == ge::DT_BF16 || isHighPercision) {
        if (expectTemplate.splitS2 == 1) {
            tensorSizeParams.set_castUbSize(batchInnerSize * s1BasicBlock * std::max(s2BasicBlock, dBasicBlock));
        } else {
            tensorSizeParams.set_castUbSize(batchInnerSize * s1BasicBlock * s2BasicBlock);
        }
    }
    tensorSizeParams.set_softmaxMaxUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
    tensorSizeParams.set_softmaxSumUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
    tensorSizeParams.set_softmaxExpUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / calcTypeSize));
    tensorSizeParams.set_apiTmpBufferBytes(apiMaxUBSize);

    tensorSizeParams.set_bmm2ResUbSize(batchInnerSize * s1BasicBlock * dBasicBlock);
}


ge::graphStatus FusedFloydAttentionTilingBase::GetWorkspaceSize()
{
    auto &tensorSizeParams = tilingData.tensorSizeParams;
    auto &coreParams = tilingData.coreParams;

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    int64_t bmm1Byetes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;
    int64_t bmm2Byetes = tensorSizeParams.get_bmm2ResUbSize() * calcTypeSize;
    workspaces[0] = static_cast<size_t>((bmm1Byetes + bmm2Byetes) * aivNum) + WORK_SPACE_RESERVE_SIZE;
    return ge::GRAPH_SUCCESS;
}


class FusedFloydAttentionTilingS1s2Bn2gs1 : public FusedFloydAttentionTilingBase {
public:
    explicit FusedFloydAttentionTilingS1s2Bn2gs1(gert::TilingContext *context) : FusedFloydAttentionTilingBase(context)
    {
        expectTemplate.splitS1 = 1;
        expectTemplate.splitS2 = 1;
        templateName = "FusedFloydAttentionS1s2Bn2gs1";
    }
    ~FusedFloydAttentionTilingS1s2Bn2gs1() override = default;

protected:
    int64_t s2sizeLimitMin = 128;
    int64_t dAlignSize = 16;
    bool enableL1Reuse = false;

    int64_t GetNRatio() override
    {
        return 64L;
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = HIGH_PERF_BUFFER_NUM;
    }

    void CalcS1S2BasicBlock() override
    {
        s1BasicBlock = std::min(64L, alignedS1);
        // d轴为64
        if (bSize * n1Size * gSize * CeilDiv(s1Size, s1BasicBlock) > aivNum) {
            s1BasicBlock = std::min(32L, alignedS1);
        }
        s2BasicBlock = std::min(32L, alignedS2);
        if (s2Size % S2_NZTOND_SIZE_64 != 0) {
            tilingKeyBmm1Format = CubeFormatEnum::NZ;
        }

        n2BasicBlock = 32LL;
    }

    void CalcDBasicBlock() override
    {
        dBasicBlock = std::min(128L, alignedD);
    }

    bool IsSpecialShape()
    {
        return bSize == 8 && n1Size == 32 && n2Size == 32 && s1Size == 2048 && s2Size == 2048 && dSize == 128 &&
               preTokens == 2048 && nextTokens == 0 && inputLayout[0] == 'S' && inputLayout[1] == 'B' &&
               inputLayout[2] == 'H' && pseExistFlag == 0 && attenMaskExistFlag == 1 &&
               tilingData.inputParams.get_attenMaskShapeType() == ATTEN_1_1_1_S1_S2;
    }

    void SetEnableL1Reuse()
    {
        // FP32场景，不开启L1reuse
        if (inputDtypeBytes == DATA_TYPE_FP32) {
            enableL1Reuse = false;
            return;
        }
        if (dSize > L1REUSE_D_Limit) {
            OP_LOGD(context_, "Current condition [dSize(%ld) > L1REUSE_D_Limit(%ld)] does not enable L1Reuse", dSize,
                      L1REUSE_D_Limit);
            return;
        }
        if (dSize == D_SPECIFIC_SIZE && tilingData.inputParams.get_layoutType() == LAYOUT_BNSD &&
            !(s2Size % L1REUSE_S2_LIMIT_256 == 0 || s2Size == L1REUSE_S2_LIMIT_4032)) {
            OP_LOGD(context_, "Current condition [dSize(%ld) && layout(BNSD)] does not enable L1Reuse", dSize);
            return;
        }
        if (tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::ALL)) {
            enableL1Reuse = true;
            return;
        }

        if ((tilingData.inputParams.get_layoutType() == LAYOUT_BSND || tilingData.inputParams.get_layoutType() ==
            LAYOUT_BSH) && s2Size <= L1REUSE_S2_Limit_2048 && dSize <= D_SPECIFIC_SIZE &&
            bSize * n1Size <= L1REUSE_BNG_Limit) {
            OP_LOGD(context_, "Current condition [dSize(%ld) && layout(BSH/BSND) && BN(%ld)] does not enable L1Reuse",
                      dSize, bSize * n1Size);
            return;
        }

    }


    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        int64_t s2BaseSize = tmpS2BasicBlock * tilingData.coreParams.get_nRatio();
        int32_t batchNum = tilingData.coreParams.get_bmm1Num();
        
        bmm1.SetShape(tmpS1BasicBlock, s2BaseSize, dSize); 
        bmm1.SetOrgShape(tmpS1BasicBlock, s2BaseSize, dSize);
        bmm1.SetBias(false);
        bmm1.SetBufferSpace(-1, -1, -1);
        bmm1.SetBatchNum(batchNum);
        bmm1.SetBatchInfoForNormal(batchNum, batchNum, tmpS1BasicBlock, s2BaseSize, dSize);

        return true;
    }


    bool SetBmm1k1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, 
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        int64_t s2BaseSize = tmpS2BasicBlock * tilingData.coreParams.get_nRatio();

        bmm1.SetShape(n2BasicBlock, s2BaseSize, dSize); 
        bmm1.SetOrgShape(n2BasicBlock, s2BaseSize, dSize);
        bmm1.SetBias(false);
        bmm1.SetBufferSpace(-1, -1, -1);
        bmm1.SetALayout(1, n2BasicBlock, 1, s1Size, dSize); 
        bmm1.SetBLayout(1, s2BaseSize, 1, s1Size, dSize); 
        bmm1.SetCLayout(1, n2BasicBlock, 1, tmpS1BasicBlock, s2BaseSize); 
        int32_t batchNum = tilingData.coreParams.get_bmmv2Num();
        bmm1.SetBatchNum(batchNum);

        return true;
    }

    bool SetBmm2v2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, 
                            matmul_tiling::MatmulApiTiling &bmm2v2) override
    {
        bmm2v2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2v2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2v2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        int64_t s2BaseSize = tmpS2BasicBlock * tilingData.coreParams.get_nRatio();

        bmm2v2.SetShape(n2BasicBlock, dSize, s2BaseSize); 
        bmm2v2.SetOrgShape(n2BasicBlock, dSize, s2BaseSize);
        bmm2v2.SetBias(false);
        bmm2v2.SetBufferSpace(-1, -1, -1);

        bmm2v2.SetALayout(1, n2BasicBlock, 1, tmpS1BasicBlock, s2BaseSize);
        bmm2v2.SetBLayout(1, s2BaseSize, 1, s1Size, dSize);
        bmm2v2.SetCLayout(1, n2BasicBlock, 1, tmpS1BasicBlock, dSize);
        int32_t batchNum = tilingData.coreParams.get_bmmv2Num();
        bmm2v2.SetBatchNum(batchNum);

        return true;
    }


    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                            matmul_tiling::MatmulApiTiling &bmm2v2) override
    {
        bmm2v2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2v2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm2v2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        int64_t s2BaseSize = tmpS2BasicBlock * tilingData.coreParams.get_nRatio();
        // NMD, NKD
        bmm2v2.SetShape(tmpS1BasicBlock, dSize, s2BaseSize); // N, K, D
        bmm2v2.SetOrgShape(tmpS1BasicBlock, dSize, s2BaseSize);
        bmm2v2.SetBias(false);
        bmm2v2.SetBufferSpace(-1, -1, -1);

        bmm2v2.SetBatchInfoForNormal(n2BasicBlock, n2BasicBlock, tmpS1BasicBlock, dSize, s2BaseSize);

        return true;
    }


    uint64_t GetTilingKey() const override
    {
        // not care about layout in tiling key, pass BSND(enum value is 0)
        return GET_TILINGKEY(AxisEnum::S1, AxisEnum::S2, AxisEnum::NONE, implMode, tilingKeyLayout,
                             tilingKeyBmm1Format, SparseEnum::ANY, PerformanceOrientedEnum::BIG_DOUBLE_BUFFER,
                             false, hasAttenMask, false, false);
    }

    bool IsCapable() override
    {
        if (s2Size >= s2sizeLimitMin) {
            return true;
        }
        return false;
    }

    bool IsTemplateMatched() const override
    {
        return true;
    }

    bool CalcUBSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock) override
    {
        apiMaxUBSize = HIGH_PERF_API_BUFFER_MULTIPLE * tmpS1BasicBlock * tmpS2BasicBlock * sizeof(float);
        return true;
    }

    void RefreshSplitFactor()
    {
        SetEnableL1Reuse();
        if (enableL1Reuse) {
            auto &multiCoreParams = tilingData.multiCoreParams;
            int64_t totalSize = multiCoreParams.get_totalSize();
            multiCoreParams.set_splitFactorSize(
                CeilDivision(totalSize, static_cast<int64_t>(multiCoreParams.get_coreNum())) * AICAIV_RATIO_2);
            multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
        }
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        RefreshSplitFactor();

        auto &tensorSizeParams = tilingData.tensorSizeParams;
        auto &coreParams = tilingData.coreParams;

        size_t *workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Bytes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;

        // dSize小于64的场景，无需切D， workspace占用较小
        if (dSize <= D_SPECIFIC_SIZE) {
            // stage1占用2倍的空间，stage2占用2倍空间
            workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_2 + SPACE_NUM_2 * coreParams.get_n2BaseSize() *
                                                                               coreParams.get_s1BaseSize() * alignedD *
                                                                               calcTypeSize) *
                                                aivNum) +
                            WORK_SPACE_RESERVE_SIZE;
            // NZND场景，stage1占用3倍的空间，stage2占用2倍空间
            if (s2Size % S2_NZTOND_SIZE_64 != 0) {
                workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_3 +
                                                     SPACE_NUM_2 * coreParams.get_n2BaseSize() *
                                                         coreParams.get_s1BaseSize() * alignedD * calcTypeSize) *
                                                    aivNum) +
                                WORK_SPACE_RESERVE_SIZE;
            }
            // FP32场景，stage1占用4倍的空间，stage2占用2倍空间
            if (inputDtypeBytes == DATA_TYPE_FP32) {
                workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_4 +
                                                     SPACE_NUM_2 * coreParams.get_n2BaseSize() *
                                                         coreParams.get_s1BaseSize() * alignedD * calcTypeSize) *
                                                    aivNum) +
                                WORK_SPACE_RESERVE_SIZE;
            }
        } else {
            // 切D场景，stage1占用2倍的空间，stage2占用4倍空间
            workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_2 + SPACE_NUM_4 * coreParams.get_n2BaseSize() *
                                                                               coreParams.get_s1BaseSize() * alignedD *
                                                                               calcTypeSize) *
                                                aivNum) +
                            WORK_SPACE_RESERVE_SIZE;
            // NZND场景，stage1占用3倍的空间，stage2占用4倍空间
            if (s2Size % S2_NZTOND_SIZE_64 != 0) {
                workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_3 +
                                                     SPACE_NUM_4 * coreParams.get_n2BaseSize() *
                                                         coreParams.get_s1BaseSize() * alignedD * calcTypeSize) *
                                                    aivNum) +
                                WORK_SPACE_RESERVE_SIZE;
            }
            // FP32场景，stage1占用4倍的空间，stage2占用4倍空间
            if (inputDtypeBytes == DATA_TYPE_FP32) {
                workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_4 +
                                                     SPACE_NUM_4 * coreParams.get_n2BaseSize() *
                                                         coreParams.get_s1BaseSize() * alignedD * calcTypeSize) *
                                                    aivNum) +
                                WORK_SPACE_RESERVE_SIZE;
            }
        }

        return ge::GRAPH_SUCCESS;
    }

    void SetSoftMaxTiling() override
    {
        auto softmaxShape = ge::Shape({s1BasicBlock / GetNRatio(), s2BasicBlock * GetNRatio()});

        AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                          tilingData.softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
    }
};


// NOTE manually initialize tiling data in hostapi scenario in highest priority template
REGISTER_OPS_TILING_TEMPLATE(FusedFloydAttention, FusedFloydAttentionTilingS1s2Bn2gs1, 96);
} // namespace FFA
} // namespace optiling
