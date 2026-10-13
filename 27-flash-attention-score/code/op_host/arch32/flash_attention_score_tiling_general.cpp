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
 * \file flash_attention_score_tiling_general.cpp
 * \brief
 */

#include <numeric>
#include <alog_pub.h>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include "log/log.h"
#include "../../common/include/tiling_base/tiling_base.h"
#include "../../common/include/tiling_base/tiling_templates_registry.h"
#include "../../common/include/tiling_base/tiling_type.h"
#include "../flash_attention_score_tiling_common.h"
#include "../../op_kernel/arch32/flash_attention_score_tiling.h"
#include "../../op_kernel/arch32/flash_attention_score_template_tiling_key.h"

using namespace Ops::Transformer::OpTiling;
namespace optiling {
namespace FA {
static const uint32_t BYTE_BLOCK = 32;
static const int64_t GM_ALIGN = 512;
static const int64_t FRACTAL_NUM = 16L;
static const int64_t PSE_DIM_NUM = 4L;
static const int64_t S1_VEC2_BASE_SIZE_MAX = 512L;

static const int64_t BYTE_BIT_NUM = 8UL;
static const size_t PSE_INPUT_INDEX = 3UL;
static const size_t DROP_MASK_INPUT_INDEX = 4UL;
static const size_t ATTENTION_MASK_INPUT_INDEX = 6UL;
static const size_t PREFIX_INPUT_INDEX = 7UL;
static const size_t ACTUAL_SEQ_LENGTH_INPUT_INDEX = 8UL;
static const size_t ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX = 9UL;
static const size_t Q_START_IDX_INPUT_INDEX = 10UL;
static const size_t KV_START_IDX_INPUT_INDEX = 11UL;
static const size_t QUERY_ROPE_INPUT_INDEX = 15UL;
static const size_t ATTEN_OUT_INDEX = 3UL;
static const size_t ATTENTION_MASK_DIM_NUM_4 = 4UL;
static const size_t ATTENTION_MASK_DIM_NUM_2 = 2UL;
static const int64_t BMM_SOFTMAX_RATIO = 4L;
static const int64_t MAX_AIV_NUM = 48L;
static const int64_t MAX_AIC_NUM = 24L;
static const int64_t AIV_AIC_NUM_RATIO = 2L;
static const int64_t DROP_MASK_ALIGN_UNIT = 256L; // input bits, and align to 32B in UB
static const int64_t HIGH_PERF_BUFFER_NUM = 6L;
static const int64_t HIGH_PERF_SUPPORT_S1_BASIC = 128L;
static const int64_t HIGH_PERF_SUPPORT_S2_BASIC = 128L;
static const int64_t HIGH_PERF_API_BUFFER_MULTIPLE = 2L;
static const int64_t HIGH_PERF_BLOCK_SIZE = 128L;
static const uint32_t PSE_ALIBI_S_SIZE = 1024;
static constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;
static const int64_t ATTEN_MASK_S1_REV_INDEX = 2L;
static const int64_t ATTEN_MASK_COMPRESS_LIMIT = 2048L;
static const int64_t ATTEN_MASK_COMPRESS_PREFIX_LIMIT = 3072L;
static const int64_t MAX_VAR_LEN_SEQ_LEN = 20000L;
static const int64_t S2_REUSE_SIZE_512 = 512L;
static const int64_t S2_REUSE_SIZE_1024 = 1024L;
static const int64_t S1_REUSE_SIZE_3840 = 3840L;
static const int64_t S2_REUSE_SIZE_5120 = 5120L;
static const int64_t D_SPECIFIC_SIZE = 64L;
static const int64_t D_SPECIFIC_SIZE_96 = 96L;
static const int64_t BALANCE_LOAD_LIST_SIZE = 8L;
static constexpr int64_t SAMEAB_TND_BASIC_SIZE = 8 * 1024L;
static constexpr int64_t COF[BALANCE_LOAD_LIST_SIZE] = {256, 384, 512, 640, 768, 896, 960, 1024};
static constexpr int64_t SORA_TND_CASE_T1 = 10000;
static constexpr int64_t NUM_2 = 2L;
static constexpr int64_t NUM_5 = 5L;
static constexpr int64_t NUM_1024 = 1024L;
static constexpr int64_t NUM_512 = 512L;
static const int64_t BMM1_DEPTH_A1_2 = 2L;
static const int64_t BMM1_DEPTH_A1_3 = 3L;
static const int64_t BMM1_BASICBLOCK_M_128 = 128L;
static const int64_t BMM1_BASICBLOCK_N_256 = 256L;
static const int64_t BMM1_BASICBLOCK_N_128 = 128L;
static const int64_t BMM1_BASICBLOCK_K_64 = 64L;
static const int64_t BMM1_BASICBLOCK_K_128 = 128L;
static const int64_t S2_NZTOND_SIZE_64 = 64L;
static const int64_t SPACE_NUM_2 = 2L;
static const int64_t SPACE_NUM_3 = 3L;
static const int64_t SPACE_NUM_4 = 4L;
static const int64_t BMM2_BASICBLOCK_M_128 = 128L;
static const int64_t BMM2_BASICBLOCK_N_128 = 128L;
static const int64_t BMM2_BASICBLOCK_M_64 = 64L;
static const int64_t BMM2_BASICBLOCK_N_64 = 64L;
static const int64_t BMM2_BASICBLOCK_K_256 = 256L;
static const int64_t S2_SPECIFIC_SIZE_928 = 928L;
static const int64_t S2_NZTOND_SIZE_128 = 128L;
static const int64_t UB_BASIC_LIMIT_SIZE = static_cast<int64_t>(8) * 1024;
static const int64_t SLOPE_BN_DIM_NUM = 2L;
static const int64_t SLOPE_N_DIM_NUM = 1L;
static const int64_t L1REUSE_D_LIMIT = 128L;
static const int64_t L1REUSE_BNG_LIMIT = 10L;
static const int64_t L1REUSE_S2_LIMIT_1024 = 1024;
static const int64_t L1REUSE_S2_LIMIT_2048 = 2048L;
static const int64_t L1REUSE_S2_LIMIT_256 = 256;
static const int64_t L1REUSE_S2_LIMIT_4032 = 4032;
static const int64_t L1REUSE_S2_LIMIT_42192 = 42192;
static const int64_t AICAIV_RATIO_2 = 2L;
static const int64_t L1REUSE_S2_LIMIT_512 = 512;
static const int64_t L1REUSE_BNG_LIMIT_64 = 64;
static const int64_t L1REUSE_BNG_LIMIT_4800 = 4800L;
static const int64_t L1REUSE_D_LIMIT_144 = 144L;
static const int64_t INVALID_ROW_SPARSE_RATIO = 6L;
static const int64_t HEAD_DIM_MAX_VALUE = 768L;
static const int64_t DATA_TYPE_FP16 = 2L;
static const int64_t DATA_TYPE_FP32 = 4L;
static const int64_t SAMEAB_D_LIMIT_128 = 128L;
static const int64_t SAMEAB_D_LIMIT_196 = 196L;
static const int64_t SAMEAB_D_LIMIT_192 = 192L;
static const int64_t TND_S1_BASICBLOCK_128 = 128L;
static const int64_t TND_S1_BASICBLOCK_256 = 256L;
static const int64_t TND_S1_BASICBLOCK_512 = 512L;
static const int64_t SAMEAB_BASIC_BLOCK_OPTIMIZE = 2048L;
static constexpr uint64_t B4_L2_CACHESIZE = static_cast<uint64_t>(96) * 1024 * 1024;
static const int64_t S2_SPECIFIC_SIZE_18432 = 18432L;
static const int64_t S1_BASIC_BLOCK_L1CARRY_MAX = 128L;
static const int64_t D_SIZE_L1CARRY_MAX = 256L;
static const int64_t D2_SIZE_L1CARRY_MAX = 256L;
static const int64_t SOFTMAX_OUT_LAYOUT_INDEX = 12L;
static const int64_t B4_SEQ_LIMIT = 48000L;
static const int64_t SINK_INPUT_INDEX = 17L;

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
    PSE_OUTER_MUL_ADD_TYPE = 0, 
    PSE_OUTER_ADD_MUL_TYPE = 1, // default
    PSE_INNER_MUL_ADD_TYPE,
    PSE_INNER_MUL_ADD_SQRT_TYPE,
    PSE_INVALID_TYPE
};


enum PseEncodeType : uint8_t {
    PSE_ENCODE_NONE = 0,
    PSE_ENCODE_ALIBI_S2_FULL = 0x11, // shape: (1024, S2)
};

enum class STemplateType : uint8_t {
    S_TEMPLATE_UNKNOW = 0,
    ALIGNED_16 = 1,
    ALIGNED_32 = 2,
    ALIGNED_48 = 3,
    ALIGNED_64 = 4,
    ALIGNED_80 = 5,
    ALIGNED_96 = 6,
    ALIGNED_112 = 7,
    ALIGNED_128 = 8,
};

enum class DTemplateType : uint8_t {
    D_TEMPLATE_UNKNOW = 0,
    ALIGNED_80 = 5,
    ALIGNED_96 = 6,
    ALIGNED_128 = 8,
};

struct MatmulConstParams {
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
    STemplateType s1Type;
    STemplateType s2Type;
    DTemplateType dType;
};

enum MatmulPolicyType : uint8_t{
    MATMUL_POLICY_NORMAL = 0,
    MATMUL_POLICY_UNSPLITK = 1
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
        splitS1 = 0U;
        splitS2 = 0U;
        splitD = 0U;
        dtype = 0U;
        layoutType = 0U;
        sparseType = 0U;
        reserved = 0U;
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

class FlashAttentionScoreTilingBase : public TilingBaseClass {
public:
    explicit FlashAttentionScoreTilingBase(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~FlashAttentionScoreTilingBase() override = default;

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
    virtual bool GetSparseInfo(SparseEnum &sparseType);
    void EnableBandInvalidLineImplMode();
    bool SparseModeProcess(SparseEnum &sparseType);
    void SetSparseTilingInfo(SparseEnum &sparseType);
    bool SparseBandModeCheck(int64_t maxS1Value, int64_t maxS2Value, int64_t minS1Value, int64_t minS2Value,
                             SparseEnum &sparseType);
    bool PretokenAndNexttokenAdjustment(SparseEnum &sparseType);

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

    void GetActualSeqLenData(int64_t inputIdx, std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &res, int64_t &actualLen, int64_t &actualBatch, int64_t &endLen) const;

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
    bool CouldConvertTND2BSH(std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &resQ, std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &resKV,
                                            const uint32_t &firstValidIndex, const uint32_t &lastValidIndex, const int64_t &actualQBatch, const int64_t &actualKVBatch, 
                                            int64_t &s1Max, int64_t &s2Max, int64_t &t1Size, int64_t &t2Size) const;
    
    bool Analyze3DimLayout(const gert::Shape &queryShape, const gert::Shape *queryRopeShape,
                           const gert::Shape &keyShape, const gert::Shape &valueShape, size_t layoutLen);
    bool Analyze4DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &valueShape, size_t layoutLen);
    bool AnalyzeOptionalInput();
    bool MatchTemplate();
    virtual void CalcS1S2BasicBlock(const BufferNum &bufferNum);
    virtual void CalcNRatio();
    virtual void CalcDBasicBlock();
    int64_t CalcMaxS1BasicBlockSize(int64_t actualD, const BufferNum &bufferNum) const;
    int64_t CalcMaxS2BasicBlockSize(const BufferNum &bufferNum, int64_t tmpS1BasicBlock) const;
    virtual bool CalcUBSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock) = 0;
    bool IsBasicBlockInSoftMax(const ge::Shape &shape) const;
    virtual bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                    int64_t batch, matmul_tiling::MatmulApiTiling &bmm1);
    virtual bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock,
                                    int64_t batch, matmul_tiling::MatmulApiTiling &bmm2) = 0;
    bool SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                         matmul_tiling::MatmulApiTiling &bmm1, matmul_tiling::MatmulApiTiling &bmm2);
    bool SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch = 1);
    virtual void SetCoreParams();
    virtual void SetQKVStartIdx();
    virtual void SetMultiBatchCoreParams();
    virtual void SetMultiCoreParams();
    virtual void SetSoftMaxTiling();
    void SetDataCopyTransposeTiling();
    virtual void SetTensorSizeParams();
    virtual void SetSparseParams();
    virtual bool SetPseAlibiParams();
    void SetScalarConst();
    virtual void SetMatmulConstArr();
    virtual bool CheckScalarConstCondation();
    virtual bool MatchMatmulConst(MatmulConstParams &matmulConst);
    virtual bool InitSparseValidArray(std::vector<int64_t> &sparseValidArray, int64_t bIdx);
    virtual bool SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, MultiCoreParams &multiCoreParams);
    void SetPrefixSparseStartIdx(const std::vector<std::vector<int64_t>> &sparseValidArray,
                                 MultiCoreParams &multiCoreParams);
    void PrintSparseMaxMinLoadPerCore(const std::vector<int64_t> &sparseValidArray, int64_t *sparseStartIdx,
                                      int32_t validAivNum, int64_t avgLoadSize);
    bool PartitionSparseData(const std::vector<int64_t> &sparseRollingArray, int64_t sparseRollingArraySum,
                             int64_t sparseArraySize, int64_t loadMaxEachCore, std::vector<int64_t> &partitionResult);
    SparseEnum GetPrefixNList(std::ostringstream &failReason);

    uint32_t aivNum;
    uint32_t aicNum;
    int64_t apiMaxUBSize = 0;
    uint64_t l2CacheSize = 0;

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
    int64_t dRopeSize = 0;
    int64_t d2Size;
    int64_t n1Size;
    int64_t n2Size;
    int64_t s1Size;
    int64_t s2Size;
    int64_t s1StrideSize; // query Shape S inner axes, for bmm1
    int64_t s2StrideSize; // key Shape S inner axes, for bmm1
    int64_t vs2StrideSize;
    int64_t preTokens;
    int64_t nextTokens;
    int64_t s1SparseValidSize;
    int64_t s2SparseValidSize;
    int64_t sparseMode;
    uint8_t tndSoftmaxOut = 0;
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
    int64_t realT1Size;
    std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> actualSeqLenData;
    std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> actualSeqLenKvData;
    float keepProb;
    float scaleValue;
    uint8_t attenMaskCompressMode;
    uint8_t pseExistFlag;
    uint8_t attenMaskExistFlag;
    uint8_t sinkExistFlag;
    uint8_t dropMaskExistFlag;
    uint8_t matmulPolicyType = MATMUL_POLICY_NORMAL;

    int64_t alignedS1;
    int64_t alignedS2;
    int64_t alignedD;
    int64_t alignedD2;

    int64_t s1BasicBlock;
    int64_t s2BasicBlock;
    int64_t s1BasicBlockBest;
    int64_t s1VecBasicBlock;
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

    int32_t mm1BaseM;
    int32_t mm1BaseN;
    int32_t mm1BaseK;
    int32_t mm2BaseM;
    int32_t mm2BaseN;
    int32_t mm2BaseK;
    STemplateType s1TemplateType = STemplateType::S_TEMPLATE_UNKNOW;
    STemplateType s2TemplateType = STemplateType::S_TEMPLATE_UNKNOW;
    DTemplateType dTemplateType = DTemplateType::D_TEMPLATE_UNKNOW;
    std::vector<MatmulConstParams> matmulConstArr = {};

    bool isSparseValidSizeAligned = false;
    bool hasPse = false;
    bool hasAttenMask = false;
    bool hasDropOut = false;
    bool hasRope = false;
    bool isSameAB = false;
    bool enableBestBlock = false;
    bool needL1Carry = false;
    bool enableL1Reuse = false;
    FlashAttentionScoreGeneralTilingData *tilingData = context_->GetTilingData<FlashAttentionScoreGeneralTilingData>();
};

int64_t FlashAttentionScoreTilingBase::GetNRatio()
{
    return BMM_SOFTMAX_RATIO;
}

ge::graphStatus FlashAttentionScoreTilingBase::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const FlashAttentionScoreCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(opName, "compileInfoPtr is null."),
                   return ge::GRAPH_FAILED);
        aivNum = compileInfoPtr->aivNum;
        aicNum = compileInfoPtr->aicNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
        l2CacheSize = compileInfoPtr->l2CacheSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aivNum = ascendcPlatform.GetCoreNumAiv();
        aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize);
    }
    OP_LOGI(context_, "get platform from compileInfo.aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu).",
              aivNum, aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0cSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreTilingBase::CheckContext()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto keepProbPtr = attrs->GetAttrPointer<float>(idx++);
    auto preTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto nextTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto n1SizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto inputLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    size_t *workspaces = context_->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleValuePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keepProbPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, preTokensPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, nextTokensPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, n1SizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputLayoutPtr);
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

void FlashAttentionScoreTilingBase::SetSparseTilingInfo(SparseEnum &sparseType)
{
    auto &inputParams = tilingData->inputParams;
    inputParams.set_attenMaskCompressMode(attenMaskCompressMode);
    inputParams.set_sparseType(static_cast<uint8_t>(sparseType));
    inputParams.set_preTokens(preTokens);
    inputParams.set_nextTokens(nextTokens);
}

void FlashAttentionScoreTilingBase::EnableBandInvalidLineImplMode()
{
    if (implMode == AA_INVALID_LINE_HIGH_PRECISION) {
        return;
    }
    // pretoken and nexttoken are already valid values (leftup vertex) after adjusted
    if (preTokens < (s1Size - s2Size) || nextTokens < 0) {
        implMode = AA_INVALID_LINE_HIGH_PRECISION;
        OP_LOGI(context_, "Enable invalid line impl mode.");
        return;
    }
}

bool FlashAttentionScoreTilingBase::PretokenAndNexttokenAdjustment(SparseEnum &sparseType)
{
    if (sparseMode == static_cast<int64_t>(ALL_MASK) || sparseMode == static_cast<int64_t>(PREFIX) ||
        sparseMode == static_cast<int64_t>(PREFIX_COMPRESS)) {
        if (preTokens < s1Size - 1 || nextTokens < s2Size - 1) {
            OP_LOGW(context_,
                      "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                      "preTokens and nextTokens will be reset max int value.",
                      preTokens, nextTokens, sparseMode);
            preTokens = std::numeric_limits<int32_t>::max();
            nextTokens = std::numeric_limits<int32_t>::max();
        }
        sparseType = (sparseMode == PREFIX_COMPRESS) ? static_cast<SparseEnum>(static_cast<uint8_t>(PREFIX)) :
                                                       static_cast<SparseEnum>(static_cast<uint8_t>(sparseMode));
    } else if (sparseMode == static_cast<int64_t>(LEFT_UP_CAUSAL)) {
        if (preTokens != s1Size || nextTokens != 0) {
            OP_LOGW(context_,
                      "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                      "preTokens will be reset max int value and nextTokens will be reset 0.",
                      preTokens, nextTokens, sparseMode);
            preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
            nextTokens = 0;
        }
        sparseType = SparseEnum::CAUSAL;
    } else if (sparseMode == static_cast<int64_t>(RIGHT_DOWN_CAUSAL)) {
        if (s1Size == s2Size) {
            if (preTokens != s1Size || nextTokens != 0) {
                OP_LOGW(context_,
                          "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                          "preTokens will be reset max int value and nextTokens will be reset 0.",
                          preTokens, nextTokens, sparseMode);
                preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
                nextTokens = 0;
            }
            sparseType = SparseEnum::CAUSAL;
        } else {
            // unequal S, change to band
            preTokens = s1Size;
            nextTokens = s2Size - s1Size;
            OP_LOGD(context_,
                      "Unequal s, sparseType rightDownCasual reset to band, and reset preTokens[%ld] "
                      "and nextTokens[%ld].",
                      preTokens, nextTokens);
            sparseType = SparseEnum::BAND;
            // check need to enable AA_INVALID_LINE_HIGH_PRECISION
            EnableBandInvalidLineImplMode();
            s1SparseValidSize = preTokens;
            s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
            isSparseValidSizeAligned = true;
        }
    } else if (sparseMode == static_cast<int64_t>(BAND)) {
        // unequal s, pretoken and nexttoken count from rigthDown vertex, need to change to leftUp vertex
        if (s1Size != s2Size) {
            preTokens = s1Size - s2Size + preTokens;
            nextTokens = s2Size - s1Size + nextTokens;
        }
        if (!SparseBandModeCheck(s1Size, s2Size, s1Size, s2Size, sparseType)) {
            return false;
        }
    }
    return true;
}

bool FlashAttentionScoreTilingBase::SparseBandModeCheck(int64_t maxS1Value, int64_t maxS2Value, int64_t minS1Value,
                                                        int64_t minS2Value, SparseEnum &sparseType)
{
    int64_t oriPreTokens = (sparseMode == static_cast<int64_t>(BAND)) ? (preTokens + s2Size - s1Size) : preTokens;
    int64_t oriNextTokens = (sparseMode == static_cast<int64_t>(BAND)) ? (nextTokens + s1Size - s2Size) : nextTokens;
    if (preTokens >= 0 && nextTokens >= 0) {
        if (preTokens >= maxS1Value && nextTokens >= maxS2Value) {
            OP_LOGW(context_,
                      "PreTokens[%ld] and nextTokens[%ld] config error, should not both greater than maxS1Val[%ld] "
                      "maxS2Val[%ld].",
                      oriPreTokens, oriNextTokens, maxS1Value, maxS2Value);
            return true;
        }
        s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), s1Size);
        s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
        isSparseValidSizeAligned = true;
        sparseType = SparseEnum::BAND;
        // check need to enable AA_INVALID_LINE_HIGH_PRECISION
        EnableBandInvalidLineImplMode();
        return true;
    }

    if (preTokens < 0 && nextTokens < 0) {
        OP_LOGE(context_, "PreTokens[%ld] and nextTokens[%ld] config error, there is no valid data block.",
                  oriPreTokens, oriNextTokens);
        return false;
    }

    if (preTokens < 0 && nextTokens >= 0) {
        int64_t absPreTokens = std::abs(preTokens);
        if (nextTokens >= absPreTokens && absPreTokens < minS2Value) {
            // check need to enable AA_INVALID_LINE_HIGH_PRECISION
            EnableBandInvalidLineImplMode();
            s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), 0L);
            s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
            isSparseValidSizeAligned = true;
            sparseType = SparseEnum::BAND;
            return true;
        } else {
            OP_LOGE(context_,
                      "PreTokens[%ld] and nextTokens[%ld] config error with S1[%ld], there is no valid data block.",
                      oriPreTokens, oriNextTokens, minS1Value);
            return false;
        }
    }

    if (preTokens >= 0 && nextTokens < 0) {
        int64_t absNextTokens = std::abs(nextTokens);
        if (absNextTokens <= preTokens && absNextTokens < minS1Value) {
            // check need to enable AA_INVALID_LINE_HIGH_PRECISION
            EnableBandInvalidLineImplMode();
            s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), s1Size);
            s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), 0L);
            isSparseValidSizeAligned = true;
            sparseType = SparseEnum::BAND;
            return true;
        } else {
            OP_LOGE(context_,
                      "PreTokens[%ld] and nextTokens[%ld] config error with S2[%ld], there is no valid data block.",
                      oriPreTokens, oriNextTokens, minS2Value);
            return false;
        }
    }
    return true;
}

bool FlashAttentionScoreTilingBase::SparseModeProcess(SparseEnum &sparseType)
{
    if (sparseMode > static_cast<int64_t>(PREFIX_COMPRESS)) {
        OP_LOGE(context_, "Not support sparse mode of %ld.", sparseMode);
        return false;
    }

    if (!PretokenAndNexttokenAdjustment(sparseType)) {
        return false;
    }

    if (sparseMode == static_cast<int64_t>(SparseEnum::PREFIX) || sparseMode == static_cast<int64_t>(PREFIX_COMPRESS)) {
        std::ostringstream failReason;
        sparseType = GetPrefixNList(failReason);
        if (sparseType != SparseEnum::PREFIX && sparseMode == static_cast<int64_t>(PREFIX_COMPRESS)) {
            OP_LOGE(context_, "[%s] %s.", templateName, failReason.str().c_str());
            return false;
        }

        if (sparseType == SparseEnum::PREFIX && sparseMode == static_cast<int64_t>(PREFIX) &&
            tilingData->inputParams.get_attenMaskShapeType() != ATTEN_B_N2_G_S1_S2 &&
            tilingData->inputParams.get_attenMaskShapeType() != ATTEN_B_1_1_S1_S2 && bSize != 1) {
            OP_LOGE(context_, "Prefix mode get invalid atten_mask shape, should be [BNSS] or [B1SS].");
            return false;
        }
    }
    return true;
}

bool FlashAttentionScoreTilingBase::GetSparseInfo(SparseEnum &sparseType)
{
    OP_LOGD(context_, "check sparse info: preTokens[%ld], nextTokens[%ld], s1[%ld], s2[%ld], attenMaskExistFlag[%d].",
              preTokens, nextTokens, s1Size, s2Size, attenMaskExistFlag);
    if (attenMaskExistFlag != 1) {
        return true;
    }

    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        return true;
    }

    if (sparseMode == static_cast<int64_t>(NO_MASK)) {
        if (preTokens >= s1Size && nextTokens == 0) {
            sparseType = SparseEnum::CAUSAL;
            preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
        } else {
            if (preTokens >= s1Size && nextTokens >= s2Size) {
                return true;
            }
            if (!SparseBandModeCheck(s1Size, s2Size, s1Size, s2Size, sparseType)) {
                return false;
            }
        }
    } else {
        if (!SparseModeProcess(sparseType)) {
            return false;
        }
    }
    return true;
}

bool FlashAttentionScoreTilingBase::SetPseAlibiParams()
{
    auto &inputParams = tilingData->inputParams;
    inputParams.set_pseEncodeType(PSE_ENCODE_NONE);
    auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
    if (pseShape == nullptr || pseShape->GetStorageShape().GetDimNum() == 0) {
        return true;
    }
    if (pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_TYPE) || pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_SQRT_TYPE)) {
        if (s1Size != s2Size) {
            OP_LOGE(context_, "INNER Pse alibi is supported only when s1Size and s2Size are equal.");
            return false;
        }
        return true;
    }
    if (pseShape->GetStorageShape().GetDimNum() < 2) {  // 2 is min dim num of legal pse
        OP_LOGE(context_, "Invalid Pse DimNum(%zu), PseType(%ld).",
                  pseShape->GetStorageShape().GetDimNum(), pseType);
        return false;
    }
    auto pseS1Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 2);
    auto pseS2Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 1);
    if (pseS1Size == PSE_ALIBI_S_SIZE && s1Size > PSE_ALIBI_S_SIZE && pseS2Size == s2Size) {
        if (s1Size != s2Size) {
            OP_LOGE(opName, "Pse alibi only support same S1 S2 when S1 lager than 1024");
            return false;
        }
    }
    return true;
}

ge::graphStatus FlashAttentionScoreTilingBase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOGD(opName, "TilingContext: %s.", GetTilingContextDebugStr().c_str());
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(opName, "invalid context."),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout() || !AnalyzeOptionalInput(),
               OP_LOGE(opName, "fail to analyze context info."), return ge::GRAPH_FAILED);

    alignedS1 = AlignUp(s1Size, FRACTAL_NUM);
    alignedS2 = AlignUp(s2Size, FRACTAL_NUM);
    alignedD = AlignUp((dSize + dRopeSize), FRACTAL_NUM);
    alignedD2 = AlignUp(d2Size, FRACTAL_NUM);

    OP_CHECK_IF(alignedS1 <= 0, OP_LOGE(opName, "invalid alignedS1 %ld.", alignedS1),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(alignedS2 <= 0, OP_LOGE(opName, "invalid alignedS2 %ld.", alignedS2),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(alignedD <= 0, OP_LOGE(opName, "invalid alignedD %ld.", alignedD),
        return ge::GRAPH_FAILED);

    auto &inputParams = tilingData->inputParams;
    inputParams.set_bSize(bSize);
    inputParams.set_n2Size(n2Size);
    inputParams.set_gSize(gSize);
    inputParams.set_s1Size(s1Size);
    inputParams.set_s2Size(s2Size);
    inputParams.set_dSize(dSize);
    inputParams.set_keepProb(keepProb);
    inputParams.set_scaleValue(scaleValue);
    inputParams.set_alignedS2(alignedS2);
    inputParams.set_pseType(static_cast<uint32_t>(pseType));
    inputParams.set_tndSoftmaxOut(tndSoftmaxOut);
    OP_LOGD(context_, "input params: bn2gs1s2d[%ld, %ld, %ld, %ld, %ld, %ld], keepProb[%f], scaleValue[%f],"
              "pseType:%ld, s1BasicBlockBest:%ld.", bSize, n2Size, gSize, s1Size, s2Size,
              dSize, keepProb, scaleValue, pseType, s1BasicBlockBest);

    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreTilingBase::Reset()
{
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
    dRopeSize = 0LL;
    d2Size = 0LL;
    n1Size = 0LL;
    n2Size = 0LL;
    s1Size = 0LL;
    s2Size = 0LL;
    maxS1Val = 0LL;
    minS1Val = 0LL;
    accumS1 = 0LL;
    accumS2 = 0LL;
    bandIndex = 0LL;
    dropTotalSize = 0LL;
    maxS2Val = 0LL;
    minS2Val = 0LL;
    realT1Size = 0LL;

    s1StrideSize = 0LL;
    s2StrideSize = 0LL;
    vs2StrideSize = 0LL;
    preTokens = std::numeric_limits<int32_t>::max();
    nextTokens = std::numeric_limits<int32_t>::max();
    sparseMode = static_cast<int64_t>(NO_MASK);
    pseType = static_cast<int64_t>(PSE_OUTER_ADD_MUL_TYPE);
    pseAlibiBaseS1 = 0;
    pseAlibiBaseS2 = 0;
    qStartIdx = 0;
    kvStartIdx = 0;
    keepProb = 1.0f;
    scaleValue = 1.0f;
    pseExistFlag = static_cast<uint8_t>(0);
    attenMaskCompressMode = NO_COMPRESS_MODE;
    attenMaskExistFlag = static_cast<uint8_t>(0);
    dropMaskExistFlag = static_cast<uint8_t>(0);
    sinkExistFlag = static_cast<uint8_t>(0);
    isHighPercision = true;

    alignedS1 = 0LL;
    alignedS2 = 0LL;
    alignedD = 0LL;
    alignedD2 = 0LL;

    s1BasicBlock = std::numeric_limits<int64_t>::max();
    s2BasicBlock = std::numeric_limits<int64_t>::max();
    dBasicBlock = std::numeric_limits<int64_t>::max();
    s1BasicBlockBest = TND_S1_BASICBLOCK_128;
    nRatio = GetNRatio();

    minUsedUBSize = 0LL;
    maxValidS2Len = 0LL;
    batchBasic = 1LL;

    opName = nullptr;
    inputLayout = nullptr;

    actualTemplate.Reset();

    tilingData->reset();
}

bool FlashAttentionScoreTilingBase::AnalyzeDtype()
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
            OP_LOGE(opName, "not support input dtype: %s for now",
                                        ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str());
            return false;
    }

    bmm2OutDtype = bmm1OutDtype;
    OP_LOGD(context_, "Get high precision flag: %d.", isHighPercision);
    return true;
}

bool FlashAttentionScoreTilingBase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto keepProbPtr = attrs->GetAttrPointer<float>(idx++);
    auto preTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto nextTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto n1SizePtr = attrs->GetAttrPointer<uint32_t>(idx++);
    inputLayout = attrs->GetAttrPointer<char>(idx++);

    preTokens = *preTokensPtr;
    nextTokens = *nextTokensPtr;
    keepProb = *keepProbPtr;
    scaleValue = *scaleValuePtr;
    n1Size = *n1SizePtr;
    OP_CHECK_IF(n1Size == 0, OP_LOGE(opName, "Head num is zero."), return false);
    OP_CHECK_IF(keepProb <= 0.0 || keepProb > 1.0,
               OP_LOGE(opName, "keepProb value must be in range of (0, 1]."), return false);

    implMode = ImplMode::AA_HIGH_PRECISION;
    if (attrs->GetAttrNum() > idx) {
        auto implModePtr = attrs->GetAttrPointer<uint8_t>(idx++);
        if (static_cast<ImplMode>(*implModePtr) == ImplMode::AA_INVALID_LINE_HIGH_PRECISION) {
            implMode = ImplMode::AA_INVALID_LINE_HIGH_PRECISION;
        }
        isHighPercision = true; // use default value
    }

    if (attrs->GetAttrNum() > idx) {
        auto sparseModePtr = attrs->GetAttrPointer<int64_t>(idx++);
        sparseMode = *sparseModePtr;
        if (sparseMode == static_cast<int64_t>(LEFT_UP_CAUSAL)) {
            attenMaskCompressMode = LEFT_UP_CAUSAL_MODE;
        } else if (sparseMode == static_cast<int64_t>(RIGHT_DOWN_CAUSAL)) {
            attenMaskCompressMode = RIGHT_DOWN_CAUSAL_MODE;
        } else if (sparseMode == static_cast<int64_t>(BAND)) {
            attenMaskCompressMode = BAND_MODE;
        } else if (sparseMode == static_cast<int64_t>(RIGHT_DOWN_CAUSAL_BAND)) {
            attenMaskCompressMode = RIGHT_DOWN_CAUSAL_BAND_MODE;
        } else if (sparseMode == static_cast<int64_t>(BAND_LEFT_UP_CAUSAL)) {
            attenMaskCompressMode = BAND_LEFT_UP_CAUSAL_MODE;
        } else if (sparseMode == static_cast<int64_t>(PREFIX_COMPRESS)) {
            attenMaskCompressMode = PREFIX_MODE;
        }
        OP_LOGD(context_, "The current value of attenMaskCompressMode is %u.", attenMaskCompressMode);
    }
    if (attrs->GetAttrNum() > idx) {
        auto pseTypePtr = attrs->GetAttrPointer<int64_t>(idx++);
        pseType = *pseTypePtr;
        OP_CHECK_IF(pseType < 0 || pseType >= PSE_INVALID_TYPE,
                   OP_LOGE(opName, "pseType value is out of range"), return false);
    }
    if (attrs->GetAttrNum() > static_cast<size_t>(SOFTMAX_OUT_LAYOUT_INDEX)) {
        // read 13th attr softmax_out_layout
        const char *softmaxOutLayout = attrs->GetAttrPointer<char>(SOFTMAX_OUT_LAYOUT_INDEX);
        if (strcmp(inputLayout, "TND") == 0 && strcmp(softmaxOutLayout, "same_as_input") == 0) {
            // check whether softmax_out_layout is TND
            tndSoftmaxOut = static_cast<uint8_t>(1);
        }
    }
    OP_LOGD(context_, "attrs: scale_value[%f] keep_prob[%f] pre_tockens[%ld] next_tockens[%ld] head_num[%ld]"
              "input_layout[%s] inner_precise[%d] sparse_mode[%ld] pseType[%ld].",
              scaleValue, keepProb, preTokens, nextTokens, n1Size, inputLayout, implMode, sparseMode, pseType);
    return true;
}

bool FlashAttentionScoreTilingBase::AnalyzeLayout()
{
    auto &queryShape = context_->GetInputShape(0)->GetStorageShape();
    auto *queryRopeShape = context_->GetOptionalInputShape(15) ? &context_->GetOptionalInputShape(15)->GetStorageShape() : nullptr;
    auto &keyShape = context_->GetInputShape(1)->GetStorageShape();
    auto &valueShape = context_->GetInputShape(2)->GetStorageShape();

    size_t layoutLen = strlen(inputLayout);
    OP_LOGD(context_, "Get input_layout [%s].", inputLayout);
    OP_CHECK_IF(queryShape.GetDimNum() != layoutLen || keyShape.GetDimNum() != layoutLen,
               OP_LOGE(opName, "Invalid layout[%s].", inputLayout), return false);
    OP_CHECK_IF(!Analyze3DimLayout(queryShape, queryRopeShape, keyShape, valueShape, layoutLen) ||
                !Analyze4DimLayout(queryShape, keyShape, valueShape, layoutLen),
               OP_LOGE(opName, "Get unsupported layout: %s", inputLayout), return false);
    OP_CHECK_IF(gSize == 0, OP_LOGE(opName, "gSize is zero."), return false);
    OP_CHECK_IF(n2Size == 0, OP_LOGE(opName, "n2Size is zero."), return false);
    OP_CHECK_IF(dSize > HEAD_DIM_MAX_VALUE || dSize <= 0L,
               OP_LOGE(opName, "dSize is not in range:(0, 512]."), return false);
    OP_CHECK_IF(n1Size % n2Size != 0,
               OP_LOGE(opName, "n1Size [%ld] should be a multiple of n2Size [%ld].", n1Size, n2Size),
               return false);
    return true;
}

bool FlashAttentionScoreTilingBase::CouldConvertTND2BSH(std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &resQ,
                                                        std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &resKV,
                                                        const uint32_t &firstValidIndex, const uint32_t &lastValidIndex,
                                                        const int64_t &actualQBatch, const int64_t &actualKVBatch, int64_t &s1Max,
                                                        int64_t &s2Max, int64_t &t1Size, int64_t &t2Size) const
{
    auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
    auto queryRopeShape = context_->GetOptionalInputShape(QUERY_ROPE_INPUT_INDEX);
    if (queryRopeShape != nullptr) {
        return false;
    }
    if (tndSoftmaxOut == 1) {
        return false; // could not switch to BSH layout with TND softmax 
    }
    if (!(pseShape == nullptr || pseShape->GetStorageShape().GetDimNum() == 0)) {
        return false; 
    }
    if (sparseMode == RIGHT_DOWN_CAUSAL_BAND || sparseMode == BAND_LEFT_UP_CAUSAL) {
        return false;
    }
    if (actualQBatch != actualKVBatch) {
        return false;
    }
    if (s1Max * actualQBatch != t1Size || s2Max * actualKVBatch != t2Size) {
        return false;
    }
    for (uint32_t i = firstValidIndex; i <= lastValidIndex; i++) {
        if (resQ[i] == 0 && resKV[i] == 0) {
            continue;
        }
        if ((resKV[i] == 0) || (resQ[i] == 0)) {
            return false;
        }
    }
    return true;
}

void FlashAttentionScoreTilingBase::GetActualSeqLenData(int64_t inputIdx, std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &res,
                                                        int64_t &actualLen, int64_t &actualBatch, int64_t &endLen) const
{
    auto actualSeqLenTensor = context_->GetOptionalInputTensor(inputIdx);
    if (actualSeqLenTensor == nullptr) {
        OP_LOGW(context_, "[%s]actualSeqLenTensor is null pointer", templateName);
        return;
    }
    auto &actualSeqLenShape = actualSeqLenTensor->GetShape().GetStorageShape();
    if (actualSeqLenShape.GetDimNum() != 1) {
        OP_LOGW(context_, "[%s]actualSeqLenShape is invalid %lu %ld", templateName, actualSeqLenShape.GetDimNum(),
                  actualSeqLenShape.GetDim(0));
        return;
    }
    /* Get Data from tensor. */
    const int64_t *value = actualSeqLenTensor->GetData<int64_t>();
    if (value == nullptr) {
        OP_LOGW(context_, "[%s]actualSeqLenTensor data is null pointer", templateName);
        return;
    }
    res[0] = value[0];
    if (value[0] != 0) {
        actualBatch++;
    }
    actualLen++;
    for (int64_t i = 1; i < actualSeqLenShape.GetDim(0); ++i) {
        auto qLen = value[i] - value[i - 1];
        if (qLen <= 0) {
            res[i] = 0;
        } else {
            res[i] = qLen;
            actualBatch++;
        }
        actualLen++;
    }
    endLen = value[actualSeqLenShape.GetDim(0) - 1];
}

bool FlashAttentionScoreTilingBase::Analyze3DimLayout(const gert::Shape &queryShape, const gert::Shape *queryRopeShape,
                                                      const gert::Shape &keyShape, const gert::Shape &valueShape, size_t layoutLen)
{
    int64_t h1 = 0;
    int64_t h2 = 0;
    int64_t h3 = 0;
    dRopeSize = 0;
    if (layoutLen == 3UL) {
        if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'H') { // 2: H idx
            s1Size = queryShape.GetDim(1);
            bSize = queryShape.GetDim(0);
            s2Size = keyShape.GetDim(1);
            h1 = queryShape.GetDim(2); // 2: H idx
            h2 = keyShape.GetDim(2);   // 2: H idx
            h3 = valueShape.GetDim(2); // 2: H idx
            s1StrideSize = h1;
            s2StrideSize = h2;
            vs2StrideSize = h3;
            tilingData->inputParams.set_layoutType(LAYOUT_BSH);
            tilingKeyLayout = LayoutType::LAYOUT_BSH;
        } else if (inputLayout[0] == 'S' && inputLayout[1] == 'B' && inputLayout[2] == 'H') { // 2: H idx
            s1Size = queryShape.GetDim(0);
            s2Size = keyShape.GetDim(0);
            bSize = queryShape.GetDim(1);
            h1 = queryShape.GetDim(2); // 2: H idx
            h2 = keyShape.GetDim(2);   // 2: H idx
            h3 = valueShape.GetDim(2); // 2: H idx
            s1StrideSize = h1 * bSize;
            s2StrideSize = h2 * bSize;
            vs2StrideSize = h3 * bSize;
            tilingData->inputParams.set_layoutType(LAYOUT_SBH);
            tilingKeyLayout = LayoutType::LAYOUT_SBH;
        } else if (inputLayout[0] == 'T' && inputLayout[1] == 'N' && inputLayout[2] == 'D') {
            int64_t actualSeqQLen = 0;
            int64_t actualSeqKVLen = 0;
            int64_t t1Size = queryShape.GetDim(0);
            int64_t t2Size = keyShape.GetDim(0);
            int64_t actualQBatch = 0;
            int64_t actualKVBatch = 0;
            int64_t endQLen = -1;
            int64_t endKvLen = -1;
            realT1Size = t1Size;
            std::fill(actualSeqLenData.begin(), actualSeqLenData.end(), 0);
            std::fill(actualSeqLenKvData.begin(), actualSeqLenKvData.end(), 0);
            GetActualSeqLenData(ACTUAL_SEQ_LENGTH_INPUT_INDEX, actualSeqLenData, actualSeqQLen,actualQBatch, endQLen);
            GetActualSeqLenData(ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX, actualSeqLenKvData, actualSeqKVLen,actualKVBatch, endKvLen);
            OP_CHECK_IF(actualSeqQLen != actualSeqKVLen,
                       OP_LOGE(opName, "VarLen scene, q is not equal kv."), return false);
            bSize = actualSeqQLen;
            accumS1 = std::accumulate(actualSeqLenData.begin(), actualSeqLenData.end(), 0LL);
            accumS2 = std::accumulate(actualSeqLenKvData.begin(), actualSeqLenKvData.end(), 0LL);
            OP_CHECK_IF(
                t1Size < accumS1 || t2Size < accumS2,
                OP_LOGE(
                    opName,
                    "Query T(%ld) and key T(%ld) need larger than respectively sum of seqLen(%ld) and sekvLen(%ld).",
                    t1Size, t2Size, accumS1, accumS2),
                return false);
            // 校验EOD场景尾部是否补0
            if (t1Size > accumS1 && t2Size > accumS2) {
                if (endQLen != 0 || endKvLen != 0) {
                    OP_LOGE(opName, "The end of actualSeqQLen & actualSeqKvLen should be 0 in EOD scenario, but got (%d) and (%d).", 
                                            endQLen, endKvLen);
                    return false;
                }
            }
            uint32_t firstValidIndex = 0;
            uint32_t lastValidIndex = static_cast<uint32_t>(bSize - 1LL);
            for (int64_t i = 0; i < bSize; ++i) {
                if (actualSeqLenData[i] != 0) {
                    firstValidIndex = static_cast<uint32_t>(i);
                    break;
                }
            }
            for (auto i = bSize - 1; i >= 0; --i) {
                if (actualSeqLenData[i] != 0) {
                    lastValidIndex = static_cast<uint32_t>(i);
                    break;
                }
            }
            maxS1Val = *std::max_element(actualSeqLenData.begin(), actualSeqLenData.end());
            maxS2Val = *std::max_element(actualSeqLenKvData.begin(), actualSeqLenKvData.end());
            bool couldConvert = CouldConvertTND2BSH(actualSeqLenData, actualSeqLenKvData,firstValidIndex,lastValidIndex,actualQBatch,actualKVBatch,maxS1Val,maxS2Val,t1Size,t2Size);
            if (couldConvert && queryShape.GetDim(2) == 128 && keyShape.GetDim(2) == 128 && valueShape.GetDim(2) == 128) {
                bSize = actualQBatch;
                s1Size = maxS1Val;
                s2Size = maxS2Val;
                s1StrideSize = queryShape.GetDim(1) * queryShape.GetDim(2);
                s2StrideSize = keyShape.GetDim(1) * keyShape.GetDim(2);
                vs2StrideSize = valueShape.GetDim(1) * valueShape.GetDim(2);
                h1 = s1StrideSize;
                h2 = s2StrideSize;
                h3 = vs2StrideSize;
                tilingData->inputParams.set_layoutType(LAYOUT_BSH);
                tilingKeyLayout = LayoutType::LAYOUT_BSH;
            } else {
                s1Size = maxS1Val;
                s2Size = maxS2Val;

                if (sparseMode == static_cast<int64_t>(RIGHT_DOWN_CAUSAL_BAND)) {
                    bandIndex = static_cast<int64_t>(lastValidIndex);
                    tilingData->inputParams.set_bandIndex(lastValidIndex);
                }
                if (sparseMode == BAND_LEFT_UP_CAUSAL) {
                    bandIndex = firstValidIndex;
                    tilingData->inputParams.set_bandIndex(firstValidIndex);
                }
                maxS1Val = *std::max_element(actualSeqLenData.begin(), actualSeqLenData.end());
                maxS2Val = *std::max_element(actualSeqLenKvData.begin(), actualSeqLenKvData.end());
                s1Size = maxS1Val;
                s2Size = maxS2Val;
                OP_CHECK_IF(n1Size != queryShape.GetDim(1),
                        OP_LOGE(opName, "head_num is [%ld], but got query dim1 [%ld].", n1Size,
                                                    queryShape.GetDim(1)),
                        return false);
                n2Size = keyShape.GetDim(1);
                OP_CHECK_IF(n2Size == 0, OP_LOGE(opName, "N2 is zero."), return false);
                gSize = queryShape.GetDim(1) / n2Size;
                dSize = queryShape.GetDim(2);
                dRopeSize = queryRopeShape ? queryRopeShape->GetDim(2) : 0;
                d2Size = valueShape.GetDim(2);
                h1 = n1Size * dSize;
                h2 = n2Size * dSize;
                h3 = n2Size * d2Size;
                s1StrideSize = gSize * n2Size * dSize;
                s2StrideSize = n2Size * dSize;
                vs2StrideSize = n2Size * d2Size;
                tilingData->inputParams.set_layoutType(LAYOUT_TND);
                tilingKeyLayout = LayoutType::LAYOUT_TND;
                int32_t count512to1024 = 0;
                int64_t seqQTotal = 0;
                for (int64_t i  = 0; i < bSize; i++) {
                    if (actualSeqLenKvData[i] >= NUM_512 && actualSeqLenKvData[i] < NUM_1024) {
                        count512to1024++;
                    }
                    seqQTotal += actualSeqLenData[i];
                }
                if (seqQTotal > SORA_TND_CASE_T1 && s2Size < NUM_1024 && sparseMode != PREFIX_COMPRESS) {
                    if (count512to1024 * 2 >= bSize) { // 512-1024 超过一半使用256
                        s1BasicBlockBest = TND_S1_BASICBLOCK_256;
                    } else {
                        s1BasicBlockBest = TND_S1_BASICBLOCK_512;
                    }
                    if (bSize >= NUM_5) {
                        s1BasicBlockBest = TND_S1_BASICBLOCK_256;
                    }
                }
            }
        } else {
            return false;
        }
        OP_CHECK_IF(h1 == 0 || h2 == 0, OP_LOGE(opName, "H is zero."), return false);
        OP_CHECK_IF(h1 % n1Size != 0,
                   OP_LOGE(opName, "h1 [%ld] should be a multiple of n1Size [%ld].", h1, n1Size),
                   return false);
        dSize = h1 / n1Size;
        gSize = h1 / h2;
        n2Size = h2 / dSize;
        d2Size = h3 / n2Size;
    }

    return true;
}

bool FlashAttentionScoreTilingBase::Analyze4DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &valueShape,
                                                      size_t layoutLen)
{
    if (layoutLen == 4UL) {
        // 2: N idx, 3: D idx
        if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'N' && inputLayout[3] == 'D') {
            bSize = queryShape.GetDim(0);
            s1Size = queryShape.GetDim(1);
            s2Size = keyShape.GetDim(1);
            n2Size = keyShape.GetDim(2); // 2: N idx
            OP_CHECK_IF(n2Size == 0, OP_LOGE(opName, "N2 is zero."), return false);
            OP_CHECK_IF(n1Size != queryShape.GetDim(2),
                       OP_LOGE(opName, "head_num is [%ld], but got query dim2 [%ld].", n1Size,
                                                   queryShape.GetDim(2)),
                       return false);
            gSize = queryShape.GetDim(2) / n2Size; // 2: N idx
            dSize = queryShape.GetDim(3);          // 3: D1 idx
            d2Size = valueShape.GetDim(3);         // 3: D2 idx
            s1StrideSize = gSize * n2Size * dSize;
            s2StrideSize = n2Size * dSize;
            vs2StrideSize = n2Size * d2Size;
            tilingData->inputParams.set_layoutType(LAYOUT_BSND);
            tilingKeyLayout = LayoutType::LAYOUT_BSND;
        } else if (inputLayout[0] == 'B' && inputLayout[1] == 'N' &&
                   // 2: S idx, 3: N idx
                   inputLayout[2] == 'S' && inputLayout[3] == 'D') {
            bSize = queryShape.GetDim(0);
            n2Size = keyShape.GetDim(1); // 1: N idx
            OP_CHECK_IF(n2Size == 0, OP_LOGE(opName, "N2 is zero."), return false);
            OP_CHECK_IF(n1Size != queryShape.GetDim(1),
                       OP_LOGE(opName, "head_num is [%ld], but got query dim1 [%ld].", n1Size,
                                                   queryShape.GetDim(1)),
                       return false);
            gSize = queryShape.GetDim(1) / n2Size;
            s1Size = queryShape.GetDim(2); // 2: S idx
            s2Size = keyShape.GetDim(2);   // 2: S idx
            dSize = queryShape.GetDim(3);  // 3: D1 idx
            d2Size = valueShape.GetDim(3); // 3: D2 idx
            s1StrideSize = dSize;
            s2StrideSize = dSize;
            vs2StrideSize = d2Size;
            tilingData->inputParams.set_layoutType(LAYOUT_BNSD);
            tilingKeyLayout = LayoutType::LAYOUT_BNSD;
        } else {
            return false;
        }
    }

    return true;
}

SparseEnum FlashAttentionScoreTilingBase::GetPrefixNList(std::ostringstream &failReason)
{
    auto prefixN = context_->GetOptionalInputTensor(PREFIX_INPUT_INDEX);
    if (prefixN == nullptr) {
        OP_LOGW(context_, "[%s]prefixN is null pointer while sparse mode is prefix", templateName);
        failReason << "prefixN is null pointer while sparse mode is prefix";
        return SparseEnum::ALL;
    }

    auto &prefixShape = prefixN->GetShape().GetStorageShape();
    if (prefixShape.GetDimNum() != 1) {
        OP_LOGW(context_, "[%s] prefixN shape is invalid, DimNum should be 1, but it is %lu.", templateName,
                  prefixShape.GetDimNum());
        failReason << "prefixN shape is invalid, DimNum should be 1, but it is " << prefixShape.GetDimNum();
        return SparseEnum::ALL;
    }
    if (prefixShape.GetDim(0) != bSize) {
        OP_LOGW(context_, "[%s] prefixN is invalid, it should be the same size as bSize[%ld], but it is %ld.",
                  templateName, bSize, prefixShape.GetDim(0));
        failReason << "prefixN is invalid, it should be the same size as bSize[" << bSize
                   << "], but it is " << prefixShape.GetDim(0);
        return SparseEnum::ALL;
    }
    /* Get Data from tensor. */
    prefixNData = prefixN->GetData<int64_t>();
    if (prefixNData == nullptr) {
        OP_LOGW(context_, "[%s]prefixN data is null pointer", templateName);
        failReason << "prefixN data is null pointer";
        return SparseEnum::ALL;
    }

    int64_t nMin = ((s2Size - s1Size) > 0) ? (s2Size - s1Size) : 0;
    for (int64_t i = 0; i < bSize; ++i) {
        if (prefixNData[i] < nMin || prefixNData[i] > s2Size) {
            OP_LOGW(context_, "[%s] batch[%ld] prefixN=%ld is invalid, should be in range of [%ld, %ld]",
                      templateName, i, prefixNData[i], nMin, s2Size);
            failReason << "batch[" << i << "] prefixN=" << prefixNData[i] << " is invalid, should be in range of ["
                       << nMin << ", " << s2Size << "]";
            return SparseEnum::ALL;
        }

        if (s1Size > s2Size && prefixNData[i] == 0) {
            implMode = AA_INVALID_LINE_HIGH_PRECISION;
            OP_LOGD(context_, "Enable invalid line impl mode.");
        }
    }

    return SparseEnum::PREFIX;
}

void FlashAttentionScoreTilingBase::SetQKVStartIdx() {
    tilingData->inputParams.set_qStartIdx(0);
    tilingData->inputParams.set_kvStartIdx(0);
    auto qStartIdxTensor = context_->GetOptionalInputTensor(Q_START_IDX_INPUT_INDEX);
    if (qStartIdxTensor == nullptr) {
        OP_LOGW(context_, "[%s]qStartIdxTensor is null pointer", templateName);
        return;
    }
    auto &qStartIdxShape = qStartIdxTensor->GetShape().GetStorageShape();
    if (qStartIdxShape.GetDimNum() != 1) {
        return;
    }
    if (qStartIdxShape.GetDim(0) <= 0) {
        return;
    }
    /* Get Data from tensor. */
    const int64_t *value = qStartIdxTensor->GetData<int64_t>();
    if (value == nullptr) {
        return;
    }
    qStartIdx = value[0];

    auto kvStartIdxTensor = context_->GetOptionalInputTensor(KV_START_IDX_INPUT_INDEX);
    if (kvStartIdxTensor == nullptr) {
        OP_LOGW(context_, "[%s]kvStartIdxTensor is null pointer", templateName);
        return;
    }
    auto &kvStartIdxShape = kvStartIdxTensor->GetShape().GetStorageShape();
    if (kvStartIdxShape.GetDimNum() != 1) {
        return;
    }
    if (kvStartIdxShape.GetDim(0) <= 0) {
        return;
    }
    /* Get Data from tensor. */
    const int64_t *kvValue = kvStartIdxTensor->GetData<int64_t>();
    if (kvValue == nullptr) {
        return;
    }
    kvStartIdx = kvValue[0];

    tilingData->inputParams.set_qStartIdx(qStartIdx);
    tilingData->inputParams.set_kvStartIdx(kvStartIdx);
    OP_LOGD(context_, "[%s] SetQKVStartIdx qStartIdx:%ld, kvStartIdx:%ld", templateName, qStartIdx, kvStartIdx);
}

bool FlashAttentionScoreTilingBase::AnalyzeOptionalInput()
{
    // 0: (B,N2,G,S1,S2), 1: (B,N2,G,1,S2)
    PseShapeType pseShapeType = PSE_B_N2_G_1_S2;
    auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
    if (pseShape == nullptr || pseShape->GetStorageShape().GetDimNum() == 0) {
        if(pseType != PSE_OUTER_ADD_MUL_TYPE){
            /*
            * 1. pseType非默认值
            * 2. 未传入PSE
            * FA正向对这种情况进行了兼容，能够得到正确的计算结果。
            * FA反向未兼容，因此统一拦截异常输入。
            */
            OP_LOGE(context_, "Get PseInput is nullptr, but pseType is not default=%u, now pseType=%ld.", PSE_OUTER_ADD_MUL_TYPE, pseType);
            return false;
        }
    }
    if (pseShape != nullptr && pseShape->GetStorageShape().GetDimNum() != 0) {
        pseExistFlag = static_cast<uint8_t>(1);
        auto &pseShapeDims = pseShape->GetStorageShape();
        size_t pseDimNum = pseShapeDims.GetDimNum();
        int64_t pseBSize = pseShapeDims.GetDim(0);
        if (pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_TYPE) || pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            if (pseDimNum != SLOPE_BN_DIM_NUM && pseDimNum != SLOPE_N_DIM_NUM) {
                OP_LOGE(context_, "pse inner mode, unsupported pse shape");
                return false;
            }
            pseShapeType = PSE_B_N2_G_SLOPE;
            if (pseDimNum == 1) {
                pseShapeType = PSE_1_N2_G_SLOPE;
                pseBSize = 1;
            }
        } else if (tilingData->inputParams.get_layoutType() == LAYOUT_TND) {
            int64_t accumS1S2 = 0;
            for (int64_t i = 0; i < bSize; i++) {
                accumS1S2 += (actualSeqLenData[i] * actualSeqLenKvData[i]);
            }
            if (pseBSize == accumS2 * n1Size) {
                pseShapeType = PSE_B_N2_G_1_S2;
            } else if (pseBSize == accumS1S2 * n1Size) {
                pseShapeType = PSE_B_N2_G_S1_S2;
            } else if (pseDimNum == PSE_DIM_NUM && (pseShapeDims.GetDim(0) == 1 || pseShapeDims.GetDim(0) == bSize) &&
                       pseShapeDims.GetDim(1) == n1Size && pseShapeDims.GetDim(2) == PSE_ALIBI_S_SIZE &&
                       pseShapeDims.GetDim(3) == s2Size) {
                pseShapeType = PSE_B_N2_G_S1_S2;
            } else {
                OP_LOGE(context_, "get unsupported pse shape");
                return false;
            }
        } else {
            if (pseDimNum != PSE_DIM_NUM) {
                OP_LOGE(context_, "pse dim should be 4, but got %zu", pseDimNum);
                return false;
            }
            if (pseBSize != bSize && pseBSize != 1) {
                OP_LOGE(context_, "pse batchsize should be 1 or %ld, but got %ld", bSize, pseBSize);
                return false;
            }

            int64_t pseDim1Size = pseShapeDims.GetDim(1);
            int64_t pseDim2Size = pseShapeDims.GetDim(2);
            int64_t pseDim3Size = pseShapeDims.GetDim(3);
            if (pseDim1Size == n1Size && pseDim2Size == s1Size && pseDim3Size == s2Size) { // 2: pre last axiss
                pseShapeType = PSE_B_N2_G_S1_S2;
            } else if (pseDim1Size == n1Size && pseDim2Size == 1 && pseDim3Size == s2Size) {
                pseShapeType = PSE_B_N2_G_1_S2;
            } else if (pseDim1Size == n1Size && pseDim2Size == static_cast<int64_t>(PSE_ALIBI_S_SIZE) && pseDim3Size == s2Size) {
                if (s1Size < pseDim2Size) {
                    OP_LOGE(opName, "get unsupported pse shape, the shape is [%ld, %ld, %ld, %ld]", pseBSize, pseDim1Size,
                              pseDim2Size, pseDim3Size);
                    return false;
                }
                pseShapeType = PSE_B_N2_G_S1_S2;
            } else {
                OP_LOGE(opName, "get unsupported pse shape, the shape is [%ld, %ld, %ld, %ld]", pseBSize, pseDim1Size,
                          pseDim2Size, pseDim3Size);
                return false;
            }
        }
        tilingData->inputParams.set_pseBSize(static_cast<uint32_t>(pseBSize));
    }

    tilingData->inputParams.set_pseShapeType(pseShapeType);

    auto attenMaskInput = context_->GetOptionalInputDesc(ATTENTION_MASK_INPUT_INDEX);
    auto attenMaskShape = context_->GetOptionalInputShape(ATTENTION_MASK_INPUT_INDEX);
    if (attenMaskInput != nullptr && attenMaskShape != nullptr && attenMaskShape->GetStorageShape().GetDimNum() != 0) {
        attenMaskExistFlag = static_cast<uint8_t>(1);
        auto attenMaskType = attenMaskInput->GetDataType();
        OP_CHECK_IF(attenMaskType != ge::DT_BOOL && attenMaskType != ge::DT_UINT8,
                   OP_LOGE(opName, "invalid attenMask dtype[%s], only support bool or uint8.",
                                               ge::TypeUtils::DataTypeToSerialString(attenMaskType).c_str()),
                   return false);

        tilingData->inputParams.set_attenMaskDataType(1);
        // 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
        AttenMaskShapeType attenMaskShapeType = ATTEN_B_N2_G_S1_S2;
        auto &attenMaskStorageShape = attenMaskShape->GetStorageShape();
        size_t attenMaskDimNum = attenMaskStorageShape.GetDimNum();
        if (attenMaskDimNum == ATTENTION_MASK_DIM_NUM_4) {
            int64_t attenMaskDim0Size = attenMaskStorageShape.GetDim(0);
            int64_t attenMaskDim1Size = attenMaskStorageShape.GetDim(1);
            int64_t attenMaskDim2Size = attenMaskStorageShape.GetDim(2);
            int64_t attenMaskDim3Size = attenMaskStorageShape.GetDim(3);
            if (attenMaskDim0Size == 1 && attenMaskDim1Size == 1 && attenMaskDim2Size == s1Size &&
                attenMaskDim3Size == s2Size) {
                attenMaskShapeType = ATTEN_1_1_1_S1_S2;
            } else if (attenMaskDim0Size == bSize && attenMaskDim1Size == 1 && attenMaskDim2Size == s1Size &&
                       attenMaskDim3Size == s2Size) {
                attenMaskShapeType = ATTEN_B_1_1_S1_S2;
            } else if (attenMaskDim0Size == bSize && attenMaskDim1Size == n1Size && attenMaskDim2Size == s1Size &&
                       attenMaskDim3Size == s2Size) {
                attenMaskShapeType = ATTEN_B_N2_G_S1_S2;
            } else {
                OP_LOGE(context_,
                          "get unsupported atten_mask shape, the shape is [%ld, %ld, %ld, %ld]. B=[%ld], N=[%ld], "
                          "Sq=[%ld], Skv=[%ld], supported atten_mask shape can be [B, N, Sq, Skv], [B, 1, Sq, Skv], "
                          "[1, 1, Sq, Skv] and [Sq, Skv].",
                          attenMaskDim0Size, attenMaskDim1Size, attenMaskDim2Size, attenMaskDim3Size, bSize, n1Size,
                          s1Size, s2Size);
                return false;
            }
        } else if (attenMaskDimNum == ATTENTION_MASK_DIM_NUM_2) {
            int64_t attenMaskDim0Size = attenMaskStorageShape.GetDim(0);
            int64_t attenMaskDim1Size = attenMaskStorageShape.GetDim(1);
            if ((attenMaskDim0Size == s1Size && attenMaskDim1Size == s2Size) ||
                (attenMaskCompressMode != NO_COMPRESS_MODE)) {
                attenMaskShapeType = ATTEN_1_1_1_S1_S2; // maybe [S1, S2]
            } else if (attenMaskDim0Size == accumS1 && attenMaskDim1Size == accumS2) {
                attenMaskShapeType = ATTEN_1_1_1_T_T;
            } else {
                if (tilingData->inputParams.get_layoutType() == LAYOUT_TND) {
                    OP_LOGE(context_,
                              "get unsupported atten_mask shape, the shape is [%ld, %ld]. MaxSq=[%ld],  MaxSkv=[%ld], "
                              "when input_layout is TND, the supported atten_mask shape is [MaxSq, MaxSkv].",
                              attenMaskDim0Size, attenMaskDim1Size, s1Size, s2Size);
                } else {
                    OP_LOGE(
                        context_,
                        "get unsupported atten_mask shape, the shape is [%ld, %ld]. B=[%ld], N=[%ld], Sq=[%ld], "
                        "Skv=[%ld], supported atten_mask shape can be [B, N, Sq, Skv], [B, 1, Sq, Skv], [1, 1, Sq, "
                        "Skv] and [Sq, Skv].",
                        attenMaskDim0Size, attenMaskDim1Size, bSize, n1Size, s1Size, s2Size);
                }
                return false;
            }
        } else {
            OP_LOGE(context_, "atten mask dim should be 2 or 4, but got %zu", attenMaskDimNum);
            return false;
        }

        tilingData->inputParams.set_attenMaskShapeType(attenMaskShapeType);

        if ((attenMaskCompressMode != NO_COMPRESS_MODE && attenMaskCompressMode != PREFIX_MODE) &&
            ((attenMaskStorageShape.GetDim(attenMaskDimNum - ATTEN_MASK_S1_REV_INDEX) != ATTEN_MASK_COMPRESS_LIMIT) ||
             (attenMaskStorageShape.GetDim(attenMaskDimNum - 1) != ATTEN_MASK_COMPRESS_LIMIT))) {
            OP_LOGE(context_, "In the attenmask compression, please set the atten_mask_shape to [2048,2048].");
            return false;
        }
        if (attenMaskCompressMode == PREFIX_MODE &&
            ((attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - ATTEN_MASK_S1_REV_INDEX) !=
              ATTEN_MASK_COMPRESS_PREFIX_LIMIT) ||
             (attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - 1) != ATTEN_MASK_COMPRESS_LIMIT))) {
            OP_LOGE(context_, "In the prefix attenmask compression, please set the atten_mask_shape to [3072,2048].");
            return false;
        }
        tilingData->inputParams.set_attenMaskS2Size(attenMaskStorageShape.GetDim(attenMaskDimNum - 1));
    }

    auto dropMaskShape = context_->GetOptionalInputShape(DROP_MASK_INPUT_INDEX);
    auto dropMaskInput = context_->GetOptionalInputDesc(DROP_MASK_INPUT_INDEX);
    if (dropMaskInput != nullptr && dropMaskShape != nullptr && dropMaskShape->GetStorageShape().GetDimNum() != 0) {
        auto dropMaskDtype = dropMaskInput->GetDataType();
        OP_CHECK_IF(dropMaskDtype != ge::DT_UINT8,
                   OP_LOGE(opName, "invalid dropMask dtype[%s], only support uint8.",
                                               ge::TypeUtils::DataTypeToSerialString(dropMaskDtype).c_str()),
                   return false);
        int64_t dimNum = dropMaskShape->GetStorageShape().GetDimNum();
        int64_t dropMaskShapeSize = 1;
        int64_t shapeSize = 0;
        for (int64_t i = 0; i < dimNum; ++i) {
            int64_t dimValue = dropMaskShape->GetStorageShape().GetDim(i);
            dropMaskShapeSize *= dimValue;
        }
        if (tilingData->inputParams.get_layoutType() == LAYOUT_TND) {
            int64_t accumS1S2 = 0;
            for (int64_t i = 0; i < bSize; i++) {
                accumS1S2 += (actualSeqLenData[i] * actualSeqLenKvData[i]);
            }
            shapeSize = accumS1S2 * n1Size;
        } else {
            shapeSize = bSize * n1Size * s1Size * s2Size;
        }
        shapeSize = AlignUp(shapeSize, BYTE_BIT_NUM) / BYTE_BIT_NUM;
        if (dropMaskShapeSize < shapeSize) {
            OP_LOGE(context_, "Input dropMask shapeSize is invalid, it should not be less than %ld, but got %ld",
                      shapeSize, dropMaskShapeSize);
            return false;
        }
        dropMaskExistFlag = static_cast<uint8_t>(1);
    }

    // if s2Size algined to 8, then no need dropMaskOp to transfer dropMask from bit to byte format
    tilingData->inputParams.set_needDropMaskOp(static_cast<uint8_t>(dropMaskExistFlag == 1 && s2Size % 8 != 0));
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        auto needDropMaskOp = (dropMaskExistFlag == 1) && (s2Size % 8 != 0 || bSize > 1);
        tilingData->inputParams.set_needDropMaskOp(static_cast<uint8_t>(needDropMaskOp));
    }

    auto sinkShapePtr = context_->GetOptionalInputShape(SINK_INPUT_INDEX);
    auto sinkInputPtr = context_->GetOptionalInputDesc(SINK_INPUT_INDEX);
    if (sinkInputPtr != nullptr && sinkShapePtr != nullptr && sinkShapePtr->GetStorageShape().GetDimNum() != 0) {
        auto shape = sinkShapePtr->GetStorageShape();
        int64_t dimNum = shape.GetDimNum();
        auto sinkDtype = sinkInputPtr->GetDataType();
        OP_CHECK_IF(sinkDtype != ge::DT_FLOAT,
                OP_LOGE(opName, "invalid sink dtype[%s], only support float.",
                        ge::TypeUtils::DataTypeToSerialString(sinkDtype).c_str()),
                return false);

        std::string sinkShape = "";
        for (int i = 0; i < dimNum; ++i) {
            sinkShape += std::to_string(shape.GetDim(i));
            if (i < dimNum - 1) {
                sinkShape += ", ";
            }
        }
        OP_CHECK_IF(dimNum != 1, OP_LOGE(opName, "invalid sink shape [%s], sink shape only support [n,].",sinkShape.c_str()),return false);

        int64_t expectShapeSize = n1Size;
        auto actualSinkShapeSize = shape.GetShapeSize();

        if (actualSinkShapeSize != expectShapeSize) {
            OP_LOGE(context_, "Input sink shapeSize is invalid, it should be %ld, but got %ld",
                    expectShapeSize, actualSinkShapeSize);
            return false;
        }
        sinkExistFlag = static_cast<uint8_t>(1);
    }

    tilingData->inputParams.set_needSinkOp(sinkExistFlag);

    OP_LOGD(context_, "pseExistFlag: %d, attenMaskExistFlag: %d, dropMaskExistFlag: %d, sinkExistFlag: %d.", pseExistFlag,
            attenMaskExistFlag, dropMaskExistFlag, sinkExistFlag);
    return true;
}

ge::graphStatus FlashAttentionScoreTilingBase::DoOpTiling()
{
    auto &inputParams = tilingData->inputParams;
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
    OP_CHECK_IF(!GetSparseInfo(sparseType), OP_LOGE(opName, "fail to get sparse info."),
               return ge::GRAPH_FAILED);
    SetSparseTilingInfo(sparseType);
    inputParams.set_implMode(implMode);
    if (!isSparseValidSizeAligned) {
        s1SparseValidSize = preTokens;
        s2SparseValidSize = nextTokens;
    }
    SetQKVStartIdx();
    SetCoreParams();
    SetMultiCoreParams();
    SetTensorSizeParams();
    SetSparseParams();
    OP_CHECK_IF(!SetPseAlibiParams(), OP_LOGE(opName, "fail to set pse alibi info."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreTilingBase::MatchTemplate()
{
    // UB Size calc logic: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
    BufferNum bufferNum;
    GetBufferNum(bufferNum);

    s1BasicBlock = std::numeric_limits<int64_t>::max();
    s2BasicBlock = std::numeric_limits<int64_t>::max();
    CalcS1S2BasicBlock(bufferNum);
    CalcNRatio();

    if (s2BasicBlock == std::numeric_limits<int64_t>::max()) {
        OP_LOGD(context_,
                  "[%s]can't find proper S1S2 basic block for shape: S1[%ld] S2[%ld], D[%ld], minS1BasicBlock[%ld], "
                  "dtype[%s], high precision[%d]",
                  templateName, s1Size, s2Size, dSize, GetMinS1BasicBlock(),
                  ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str(), isHighPercision);
        return false;
    }

    CalcDBasicBlock();
    actualTemplate.splitS1 = s1BasicBlock < alignedS1 ? 1U : 0U;
    actualTemplate.splitS2 = s2BasicBlock < alignedS2 ? 1U : 0U;
    actualTemplate.splitD = dBasicBlock < alignedD ? 1U : 0U;

    if (IsTemplateMatched()) {
        (void)CalcUBSize(s1BasicBlock, s2BasicBlock);
        OP_LOGD(context_, "[%s]final basic block: [%ld, %ld, %ld], match template[%s].", templateName, s1BasicBlock,
                  s2BasicBlock, dBasicBlock, actualTemplate.ToString().c_str());
        return true;
    }

    return false;
}

void FlashAttentionScoreTilingBase::CalcS1S2BasicBlock(const BufferNum &bufferNum)
{
    // calc s1 s2 first, we set d basic block as s2 now
    const int64_t actualD = expectTemplate.splitD == 0 ? alignedD : FRACTAL_NUM; // if split d we use min s2 16
    int64_t maxS1BasicBlock = CalcMaxS1BasicBlockSize(actualD, bufferNum);
    maxS1BasicBlock = std::min(maxS1BasicBlock, alignedS1);
    if (maxS1BasicBlock == 0) {
        return;
    }

    for (int64_t tmpS1BasicBlock = std::min(GetMinS1BasicBlock(), maxS1BasicBlock); tmpS1BasicBlock <= maxS1BasicBlock;
         tmpS1BasicBlock += FRACTAL_NUM) {
        int64_t tmpS2BasicBlock = CalcMaxS2BasicBlockSize(bufferNum, tmpS1BasicBlock);
        tmpS2BasicBlock = std::min(tmpS2BasicBlock, alignedS2);
        for (; tmpS2BasicBlock >= FRACTAL_NUM; tmpS2BasicBlock -= FRACTAL_NUM) {
            // drop mask bug workaround
            if (dropMaskExistFlag == 1 &&
                (tmpS2BasicBlock <= static_cast<int64_t>(BYTE_BLOCK) ||
                 CalcTailSize(alignedS2, tmpS2BasicBlock) <= static_cast<int64_t>(BYTE_BLOCK))) {
                continue;
            }

            int64_t tmpDBasicBlock = expectTemplate.splitD == 1 ? std::min(tmpS2BasicBlock, alignedD) : alignedD;
            OP_LOGD(context_, "[%s]try basic block: [%ld, %ld]", templateName, tmpS1BasicBlock, tmpS2BasicBlock);
            if (CalcUBSize(tmpS1BasicBlock, tmpS2BasicBlock) &&
                SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock)) {
                break;
            }
        }

        // check whether is valid, if tmpS1BasicBlock is too big, then there is no proper tmpS2BasicBlock
        if (tmpS2BasicBlock < FRACTAL_NUM) {
            break;
        }

        OP_LOGD(context_, "[%s]get candidate basic block: [%ld, %ld]", templateName, tmpS1BasicBlock,
                  tmpS2BasicBlock);
        if (s2BasicBlock == std::numeric_limits<int64_t>::max()) {
            s1BasicBlock = tmpS1BasicBlock;
            s2BasicBlock = tmpS2BasicBlock;
        } else if (s2BasicBlock == tmpS2BasicBlock && s1BasicBlock < tmpS1BasicBlock) {
            s1BasicBlock = tmpS1BasicBlock;
        } else {
            break;
        }
    }
}

void FlashAttentionScoreTilingBase::CalcNRatio()
{
    return;
}

void FlashAttentionScoreTilingBase::CalcDBasicBlock()
{
    return;
}

int64_t FlashAttentionScoreTilingBase::CalcMaxS1BasicBlockSize(int64_t actualD, const BufferNum &bufferNum) const
{
    // if S2 basic block is min value 16, s1 basic block can reach max value, then we get:
    // s1 * 16 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp =>
    // s1 * (16 * X + D * Y + (expNum + 2) * (32 / sizeof(T))) * sizeof(T) + apiTmp
    // just ignore apiTmp now, consider it at last
    int64_t alignUnit = static_cast<int64_t>(BYTE_BLOCK) / inputDtypeBytes;
    int64_t maxS1BasicBlock =
        aicoreParams_.ubSize / inputDtypeBytes /
        (FRACTAL_NUM * bufferNum.bufferS1S2Num + actualD * bufferNum.bufferS1DNum +
         (bufferNum.bufferExpNum + 2) * alignUnit); // here 2 means FlashSoftMax sum and max output
    return AlignDown(maxS1BasicBlock, FRACTAL_NUM);
}

int64_t FlashAttentionScoreTilingBase::CalcMaxS2BasicBlockSize(const BufferNum &bufferNum,
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

bool FlashAttentionScoreTilingBase::IsBasicBlockInSoftMax(const ge::Shape &shape) const
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

void FlashAttentionScoreTilingBase::SetCoreParams()
{
    auto &coreParams = tilingData->coreParams;
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
    coreParams.set_dRopeSize(dRopeSize);
    coreParams.set_d2Size(d2Size);
    SetMultiBatchCoreParams();
}

void FlashAttentionScoreTilingBase::SetMultiBatchCoreParams()
{
    auto &coreParams = tilingData->coreParams;
    coreParams.set_bBaseSize(1);
    coreParams.set_bBaseTailSize(1);
    coreParams.set_bOuterSize(bSize);

    coreParams.set_n2BaseSize(1);
    coreParams.set_n2BaseTailSize(1);
    coreParams.set_n2OuterSize(n2Size);

    coreParams.set_gBaseSize(1);
    coreParams.set_gBaseTailSize(1);
    coreParams.set_gOuterSize(gSize);
}

void FlashAttentionScoreTilingBase::SetMultiCoreParams()
{
    auto &multiCoreParams = tilingData->multiCoreParams;
    auto &coreParams = tilingData->coreParams;
    int64_t totalSize = coreParams.get_bOuterSize() * coreParams.get_n2OuterSize() * coreParams.get_gOuterSize() *
                        coreParams.get_s1OuterSize();
    int64_t actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
    multiCoreParams.set_coreNum(static_cast<int32_t>(actualUsedAivNum));
    multiCoreParams.set_totalSize(totalSize);
    multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualUsedAivNum));
    multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
}

ge::graphStatus FlashAttentionScoreTilingBase::DoLibApiTiling()
{
    if (!SetMatMulTiling(s1BasicBlock, s2BasicBlock, dBasicBlock, batchBasic)) {
        return ge::GRAPH_FAILED;
    }
    SetSoftMaxTiling();
    SetDataCopyTransposeTiling();
    SetScalarConst();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreTilingBase::PostTiling()
{
    auto blockDim = CalcTschBlockDim(tilingData->multiCoreParams.get_coreNum(), aicNum, aivNum);
    context_->SetBlockDim(blockDim);
    auto &inputParams = tilingData->inputParams;
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    if (inputParams.get_needDropMaskOp() == 1) {
        blockDim = CalcTschBlockDim(aivNum, aicNum, aivNum);
        context_->SetBlockDim(blockDim);
        context_->SetScheduleMode(1);

        int64_t shapeTotalSize = inputParams.get_bSize() * inputParams.get_n2Size() * inputParams.get_gSize() *
                                 inputParams.get_s1Size() * inputParams.get_s2Size();
        auto layoutType = tilingData->inputParams.get_layoutType();
        if (layoutType == LAYOUT_TND) {
            for (int64_t i = 0; i < bSize; i++) {
                dropTotalSize += (actualSeqLenData[i] * actualSeqLenKvData[i]);
            }
            shapeTotalSize = inputParams.get_n2Size() * inputParams.get_gSize() * dropTotalSize;
        }
        shapeTotalSize = AlignUp(shapeTotalSize, GM_ALIGN);
        workspaces[0] += static_cast<size_t>(shapeTotalSize);
    }

    if (pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_TYPE) ||
        pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_SQRT_TYPE)) {
        tilingData->coreParams.set_pseAlibiBaseS1(pseAlibiBaseS1);
        tilingData->coreParams.set_pseAlibiBaseS2(pseAlibiBaseS2);
        int64_t pseAlibiBytes =
            AlignUp(pseAlibiBaseS2 * pseAlibiBaseS1 * 2, GM_ALIGN) * tilingData->multiCoreParams.get_coreNum();
        workspaces[0] += pseAlibiBytes;
    }
    OP_LOGD(context_, "[%s] final workspace size:%zu, pseAlibiBaseS1:%ld, pseAlibiBaseS2:%ld.",
              templateName, workspaces[0], pseAlibiBaseS1, pseAlibiBaseS2);
    OP_LOGD(opName, "[%s] tiling data:%s", templateName, GetTilingDataDebugStr().c_str());

    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreTilingBase::SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                       [[maybe_unused]] int64_t batch,
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

bool FlashAttentionScoreTilingBase::SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                    int64_t tmpDBasicBlock, int64_t batch,
                                                    matmul_tiling::MatmulApiTiling &bmm1,
                                                    matmul_tiling::MatmulApiTiling &bmm2)
{
    if (!SetBmm1TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, batch, bmm1) ||
        !SetBmm2TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, batch, bmm2)) {
        return false;
    }

    if (bmm1.GetTiling(tilingData->bmm1TilingData) == -1) {
        OP_LOGE(context_, "BMM1 tiling failed.");
        return false;
    }

    if (isSameAB && tilingData->bmm1TilingData.baseK >= tilingData->bmm1TilingData.singleCoreK &&
        tilingData->bmm1TilingData.depthA1 == BMM1_DEPTH_A1_2) {
        tilingData->bmm1TilingData.depthA1 = BMM1_DEPTH_A1_3;
    }

    // 当D > 128，由于L0 DB开启的限制，在设置tiling时把baseK切小，在MatmulPolicy中仍然按照不切K搬运
    if (matmulPolicyType == MATMUL_POLICY_UNSPLITK) {
        tilingData->bmm1TilingData.baseK = AlignUp(dSize / NUM_2, FRACTAL_NUM);
        // TSCM 自主管理场景, mm1 开启dbL0C
        tilingData->bmm1TilingData.dbL0C = 2;
    }

    tilingData->bmm1TilingData.shareMode = 0;
    tilingData->bmm1TilingData.shareL1Size = aicoreParams_.l1Size;
    tilingData->bmm1TilingData.shareL0CSize = aicoreParams_.l0cSize;

    if (bmm2.GetTiling(tilingData->bmm2TilingData) == -1) {
        OP_LOGE(context_, "BMM2 tiling failed.");
        return false;
    }

    tilingData->bmm2TilingData.shareMode = 0;
    tilingData->bmm2TilingData.shareL1Size = aicoreParams_.l1Size;
    tilingData->bmm2TilingData.shareL0CSize = aicoreParams_.l0cSize;

    mm1BaseM = bmm1.GetBaseM();
    mm1BaseN = bmm1.GetBaseN();
    mm1BaseK = bmm1.GetBaseK();
    mm2BaseM = bmm2.GetBaseM();
    mm2BaseN = bmm2.GetBaseN();
    mm2BaseK = bmm2.GetBaseK();

    return true;
}

bool FlashAttentionScoreTilingBase::SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                    int64_t tmpDBasicBlock, int64_t batch)
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        matmul_tiling::MatmulApiTiling bmm1(ascendcPlatform);
        matmul_tiling::MatmulApiTiling bmm2(ascendcPlatform);
        return SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, batch, bmm1, bmm2);
    } else {
        OP_LOGD(context_, "platform info is null, use default info to generate matmul tiling.");
        matmul_tiling::MatmulApiTiling bmm1;
        matmul_tiling::MatmulApiTiling bmm2;
        return SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, batch, bmm1, bmm2);
    }
}

void FlashAttentionScoreTilingBase::SetSoftMaxTiling()
{
    auto softmaxShape = ge::Shape({batchBasic, std::min(s1BasicBlock, alignedS1), std::min(s2BasicBlock, alignedS2)});

    AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                      tilingData->softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
}

void FlashAttentionScoreTilingBase::SetDataCopyTransposeTiling()
{
    auto &coreParams = tilingData->coreParams;
    auto transposeSrcShape = ge::Shape({coreParams.get_bBaseSize(), 1, std::min(s1BasicBlock, alignedS1),
                                        coreParams.get_gBaseSize() * std::min(dBasicBlock, alignedD)});
    auto transposeDstShape = ge::Shape({bSize, n1Size, s1Size, n1Size * dSize});
    tilingData->transposeTilingData.GetDataCopyTransposeTiling(coreParams.get_bBaseSize(), 1, std::min(s1BasicBlock, alignedS1),
                                                               coreParams.get_gBaseSize() * std::min(dBasicBlock, alignedD),
                                                               bSize, n1Size, s1Size, n1Size * dSize, inputDtypeBytes);
}

void FlashAttentionScoreTilingBase::SetTensorSizeParams()
{
    auto &tensorSizeParams = tilingData->tensorSizeParams;
    auto &coreParams = tilingData->coreParams;
    int64_t batchInnerSize = coreParams.get_bBaseSize() * coreParams.get_n2BaseSize() * coreParams.get_gBaseSize();
    tensorSizeParams.set_bmm1ResUbSize(batchInnerSize * s1BasicBlock * s2BasicBlock);
    tensorSizeParams.set_attenMaskUbSize(attenMaskExistFlag * batchInnerSize * s1BasicBlock * s2BasicBlock);
    if (tilingData->inputParams.get_pseShapeType() == PSE_B_N2_G_S1_S2) {
        tensorSizeParams.set_pseUbSize(pseExistFlag * batchInnerSize * s1BasicBlock * s2BasicBlock);
    } else {
        tensorSizeParams.set_pseUbSize(pseExistFlag * batchInnerSize * s2BasicBlock); // PSE_B_N2_G_1_S2
    }

    tensorSizeParams.set_dropMaskUbSize(dropMaskExistFlag * batchInnerSize * s1BasicBlock *
                                        AlignUp(s2BasicBlock, DROP_MASK_ALIGN_UNIT) / BYTE_BIT_NUM / inputDtypeBytes);

    if (tensorSizeParams.get_pseUbSize() > 0) {
        hasPse = true;
    }
    if (tensorSizeParams.get_dropMaskUbSize() > 0) {
        hasDropOut = true;
    }
    if (tensorSizeParams.get_attenMaskUbSize() > 0) {
        hasAttenMask = true;
    }
    if (coreParams.get_dRopeSize() != 0) {
        hasRope = true;
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

bool FlashAttentionScoreTilingBase::InitSparseValidArray(std::vector<int64_t> &sparseValidArray, int64_t bIdx)
{
    OP_CHECK_IF(sparseValidArray.size() == 0,
               OP_LOGE(opName, "Sparse valid array size should be larger than 0."), return false);
    uint8_t sparseType = tilingData->inputParams.get_sparseType();
    if (sparseType == static_cast<uint8_t>(SparseEnum::PREFIX)) {
        for (int64_t i = 0; i < static_cast<int64_t>(sparseValidArray.size()); i++) {
            int64_t s2IgnoredEndLen =
                tilingData->inputParams.get_s1Size() - tilingData->coreParams.get_s1BaseSize() * (i + 1);
            int64_t s2EndLen = 0;
            s2IgnoredEndLen = std::max(static_cast<int64_t>(0), s2IgnoredEndLen);
            if (tilingData->inputParams.get_s2Size() > s2IgnoredEndLen) {
                s2EndLen = tilingData->inputParams.get_s2Size() - s2IgnoredEndLen;
                s2EndLen = std::max(s2EndLen, prefixNData[bIdx]);
            } else {
                s2EndLen = tilingData->inputParams.get_s2Size();
                s2EndLen = std::min(s2EndLen, prefixNData[bIdx]);
            }

            s2EndLen = std::min(s2EndLen, tilingData->inputParams.get_s2Size());
            sparseValidArray[i] = CeilDivision(s2EndLen, s2BasicBlock);
        }
    } else {
        int64_t s2BlkNum = CeilDivision(s2Size, s2BasicBlock);
        int64_t validS1Size = CeilDivision(s1SparseValidSize, s1BasicBlock);
        int64_t validS2Size = CeilDivision(s2SparseValidSize, s2BasicBlock);
        int64_t invalidRowSparseRatio = INVALID_ROW_SPARSE_RATIO;
        if (s2Size <= s2BasicBlock) {
            invalidRowSparseRatio = 1;
        }
        for (int64_t i = 0; i < static_cast<int64_t>(sparseValidArray.size()); i++) {
            int64_t reduceBlk =
                (i < validS1Size) ? 0 : (CeilDivision((i + 1) * s1BasicBlock - s1SparseValidSize, s2BasicBlock) - 1);
            int64_t addBlk =
                std::min(s2BlkNum - validS2Size,
                         CeilDivision((i + 1) * s1BasicBlock + s2SparseValidSize, s2BasicBlock) - validS2Size);
            int64_t validBlockNum = validS2Size - reduceBlk + addBlk;
            sparseValidArray[i] = validBlockNum > 0 ? validBlockNum : invalidRowSparseRatio;
            maxValidS2Len = std::max(sparseValidArray[i] * s1BasicBlock, maxValidS2Len);
        }
    }
    return true;
}

bool FlashAttentionScoreTilingBase::PartitionSparseData(const std::vector<int64_t> &sparseRollingArray,
                                                        int64_t sparseRollingArraySum, int64_t sparseArraySize,
                                                        int64_t loadMaxEachCore, std::vector<int64_t> &partitionResult)
{
    OP_CHECK_IF(partitionResult.size() == 0,
               OP_LOGE(opName, "partitionResult size should be larger than 0."), return false);

    OP_CHECK_IF(sparseRollingArraySum <= 0,
               OP_LOGE(opName, "sparseRollingArraySum should be larger than 0."), return false);
    int64_t s1OuterCutEachCore = loadMaxEachCore / sparseRollingArraySum;
    int64_t s1OuterLoadEachCore = s1OuterCutEachCore * sparseRollingArraySum;
    int64_t s1OuterNumEachCore = s1OuterCutEachCore * sparseRollingArray.size();

    int64_t targetCoreNum = partitionResult.size();
    int64_t coreIdx = 0;
    int64_t rollingIdx = 0;
    int64_t loadSize = s1OuterLoadEachCore;
    partitionResult[0] = 0;
    for (int64_t i = s1OuterNumEachCore; i < sparseArraySize; i++, rollingIdx++) {
        rollingIdx = (static_cast<uint64_t>(rollingIdx) >= sparseRollingArray.size()) ? 0 : rollingIdx;
        int64_t loadNext = sparseRollingArray[rollingIdx];
        bool needOneMoreCore = (loadSize + loadNext) > loadMaxEachCore;
        if (needOneMoreCore && coreIdx >= (targetCoreNum - 1)) {
            return false;
        }

        if (needOneMoreCore) {
            partitionResult[++coreIdx] = i;
            i += s1OuterNumEachCore;
            i--;
            rollingIdx--;
            loadSize = s1OuterLoadEachCore;
            continue;
        }

        loadSize += loadNext;
    }

    std::fill(partitionResult.begin() + coreIdx + 1, partitionResult.end(), sparseArraySize);
    return true;
}

void FlashAttentionScoreTilingBase::SetPrefixSparseStartIdx(const std::vector<std::vector<int64_t>> &sparseValidArray,
                                                            MultiCoreParams &multiCoreParams)
{
    int64_t validCoreNum;
    if (isSameAB) {
        validCoreNum = std::min(static_cast<int64_t>(multiCoreParams.get_coreNum() / AIV_AIC_NUM_RATIO), MAX_AIC_NUM);
    } else {
        validCoreNum = std::min(static_cast<int64_t>(multiCoreParams.get_coreNum()), MAX_AIV_NUM);
    }
    int64_t totalSize = multiCoreParams.get_totalSize(); // BN2GS1.o
    int64_t *sparseStartIdx = multiCoreParams.get_sparseStartIdx();
    for (int64_t idx = 0; idx < MAX_AIV_NUM; ++idx) {
        sparseStartIdx[idx] = totalSize;
    }
    if (totalSize <= validCoreNum) {
        int64_t idx = 0;
        for (; idx < totalSize; ++idx) {
            sparseStartIdx[idx] = idx;
        }
        for (; idx < validCoreNum; ++idx) {
            sparseStartIdx[idx] = totalSize;
        }
        return;
    }

    int64_t loadTotal = 0;
    /* Need to adapt when we split b. */
    for (int64_t i = 0; i < bSize; i++) {
        loadTotal += std::accumulate(sparseValidArray[i].begin(), sparseValidArray[i].end(), 0LL);
    }
    int64_t n2G = tilingData->coreParams.get_n2OuterSize() * tilingData->coreParams.get_gOuterSize();
    loadTotal *= n2G;

    auto loadEachCoreExpect = CeilDivision(loadTotal, validCoreNum);
    int64_t s1OuterSize = tilingData->coreParams.get_s1OuterSize();
    int64_t tempBlock = 0;
    int64_t coreIdx = 0;
    int64_t loadStartIdx = 0;

    for (int64_t bNGS1Index = 0; bNGS1Index < bSize * n2G * s1OuterSize; ++bNGS1Index) {
        int64_t bIdx = bNGS1Index / (n2G * s1OuterSize);
        if (s1OuterSize == 0) {
            continue;
        }
        int64_t s1Idx = bNGS1Index % s1OuterSize;
        auto currBlockNum = sparseValidArray[bIdx][s1Idx];
        if (tempBlock >= loadEachCoreExpect) {
            if ((tempBlock + currBlockNum - loadEachCoreExpect) >= (loadEachCoreExpect - (tempBlock))) {
                /* 不累加当前block */
                sparseStartIdx[coreIdx++] = loadStartIdx;
                loadStartIdx = bNGS1Index;
                // 下一个核使用当前这个S1
                tempBlock = currBlockNum;
            } else {
                sparseStartIdx[coreIdx++] = loadStartIdx;
                loadStartIdx = (bNGS1Index + 1);
                tempBlock = 0;
            }
        } else {
            tempBlock += currBlockNum;
        }
    }

    if (tempBlock != 0) {
        sparseStartIdx[coreIdx++] = loadStartIdx;
    }

    /* 将没用到的核的start index置为最大值 */
    for (; coreIdx < MAX_AIV_NUM; ++coreIdx) {
        sparseStartIdx[coreIdx] = totalSize;
    }
}

bool FlashAttentionScoreTilingBase::SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray,
                                                      MultiCoreParams &multiCoreParams)
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t validCoreNum;
    if (isSameAB) {
        validCoreNum = std::min(static_cast<int64_t>(multiCoreParams.get_coreNum() / AIV_AIC_NUM_RATIO), MAX_AIC_NUM);
    } else {
        validCoreNum = std::min(static_cast<int64_t>(multiCoreParams.get_coreNum()), MAX_AIV_NUM);
    }
    int64_t totalSize = multiCoreParams.get_totalSize(); // BN2GS1.o
    int64_t *sparseStartIdx = multiCoreParams.get_sparseStartIdx();
    for (int64_t idx = 0; idx < MAX_AIV_NUM; ++idx) {
        sparseStartIdx[idx] = totalSize;
    }

    if (totalSize <= validCoreNum) {
        for (int64_t idx = 0; idx < totalSize; ++idx) {
            sparseStartIdx[idx] = idx;
        }

        return true;
    }

    // Minimize the max load each core to find a load balance result.
    // The range of max load each core is (loadEachCoreLowerBound, loadEachCoreUpperBound].
    std::vector<int64_t> partitionResult(validCoreNum, totalSize);
    std::vector<int64_t> lastValidPartitionResult(validCoreNum, totalSize);
    int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL);
    int64_t loadTotal = sparseArraySum * (totalSize / sparseValidArray.size());
    OP_CHECK_IF(validCoreNum <= 0, OP_LOGE(opName, "validCoreNum should be larger than 0."),
               return false);
    int64_t loadEachCoreLowerBound = loadTotal / validCoreNum - 1;
    int64_t loadEachCoreUpperBound =
        CeilDivision(loadTotal, validCoreNum) + (*std::max_element(sparseValidArray.begin(), sparseValidArray.end()));
    while (loadEachCoreLowerBound + 1 < loadEachCoreUpperBound) {
        int64_t loadMax = loadEachCoreLowerBound + (loadEachCoreUpperBound - loadEachCoreLowerBound) / 2;
        if ((loadMax * validCoreNum >= loadTotal) &&
            PartitionSparseData(sparseValidArray, sparseArraySum, totalSize, loadMax, partitionResult)) {
            loadEachCoreUpperBound = loadMax;
            lastValidPartitionResult.swap(partitionResult);
            continue;
        }
        loadEachCoreLowerBound = loadMax;
    }

    for (int64_t idx = 0; idx < validCoreNum; ++idx) {
        sparseStartIdx[idx] = lastValidPartitionResult[idx];
    }

    if (CheckLogLevel(OP, DLOG_DEBUG) == 1) {
        PrintSparseMaxMinLoadPerCore(sparseValidArray, sparseStartIdx, validCoreNum,
                                     CeilDivision(loadTotal, validCoreNum));
    }
    return true;
}

void FlashAttentionScoreTilingBase::PrintSparseMaxMinLoadPerCore(const std::vector<int64_t> &sparseValidArray,
                                                                 int64_t *sparseStartIdx, int32_t validAivNum,
                                                                 int64_t avgLoadSize)
{
    int64_t maxLoadSize = 0;
    int64_t minLoadSize = std::numeric_limits<int64_t>::max();
    int64_t totalSize = tilingData->multiCoreParams.get_totalSize();
    int64_t s1OuterSize = tilingData->coreParams.get_s1OuterSize();
    if (s1OuterSize == 0) {
        return;
    }
    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        int64_t startIdx = sparseStartIdx[idx];
        int64_t endIdx = totalSize;
        if (idx + 1 < validAivNum) {
            endIdx = sparseStartIdx[idx + 1];
        }

        if (startIdx >= endIdx) {
            minLoadSize = 0;
            break;
        }

        int64_t s1OuterStartIdx = startIdx % s1OuterSize;
        int64_t s1OuterEndIdx = endIdx % s1OuterSize;
        int64_t loadSize = 0;
        if (s1OuterEndIdx > s1OuterStartIdx) {
            loadSize = std::accumulate(sparseValidArray.begin() + s1OuterStartIdx,
                                       sparseValidArray.begin() + s1OuterEndIdx, 0);
        } else {
            loadSize = std::accumulate(sparseValidArray.begin() + s1OuterStartIdx, sparseValidArray.end(), 0);
            loadSize = std::accumulate(sparseValidArray.begin(), sparseValidArray.begin() + s1OuterEndIdx, loadSize);
        }

        int64_t s1OuterLoop = (endIdx / s1OuterSize) - (startIdx / s1OuterSize);
        if (s1OuterLoop > 1) {
            if (s1OuterEndIdx > s1OuterStartIdx) {
                loadSize += s1OuterLoop * std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0);
            } else {
                loadSize += (s1OuterLoop - 1) * std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0);
            }
        }

        maxLoadSize = std::max(maxLoadSize, loadSize);
        minLoadSize = std::min(minLoadSize, loadSize);
    }

    OP_LOGD(context_, "[%s]each core load: max[%ld], min[%ld], avg[%ld]", templateName, maxLoadSize, minLoadSize,
              avgLoadSize);
}

void FlashAttentionScoreTilingBase::SetSparseParams()
{
    if (tilingData->inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::ALL)) {
        return;
    }

    if (expectTemplate.splitS2 == 0) {
        OP_LOGI(context_, "[%s]match not split S2 template, close sparse feature", templateName);
        tilingData->inputParams.set_sparseType(static_cast<uint8_t>(SparseEnum::ALL));
        return;
    }

    auto &coreParams = tilingData->coreParams;
    coreParams.set_s1SparseValidSize(s1SparseValidSize);
    coreParams.set_s2SparseValidSize(s2SparseValidSize);

    auto &multiCoreParams = tilingData->multiCoreParams;
    if (tilingData->inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::PREFIX)) {
        std::vector<std::vector<int64_t>> sparseValidArray;
        for (int64_t bIdx = 0; bIdx < bSize; bIdx++) {
            sparseValidArray.emplace_back(std::vector<int64_t>(coreParams.get_s1OuterSize(), 0));
            InitSparseValidArray(sparseValidArray.back(), bIdx);
        }
        SetPrefixSparseStartIdx(sparseValidArray, multiCoreParams);
    } else {
        std::vector<int64_t> sparseValidArray(coreParams.get_s1OuterSize(), 0);
        InitSparseValidArray(sparseValidArray, 0);
        SetSparseStartIdx(sparseValidArray, multiCoreParams);
    }
}

/* 在子类中设置matmulConstArr列表 */
void FlashAttentionScoreTilingBase::SetMatmulConstArr()
{
    return;
}

/* 检查是否满足进行matmul常量化的基本条件 */
bool FlashAttentionScoreTilingBase::CheckScalarConstCondation()
{
    return false;
}

/* 检查当前tiling中的baseMNK是否与子类枚举的常量化参数匹配 */
bool FlashAttentionScoreTilingBase::MatchMatmulConst(MatmulConstParams &matmulConst)
{
    return mm1BaseM == matmulConst.baseM && mm1BaseN == matmulConst.baseN && mm1BaseK == matmulConst.baseK;
}

/* 从子类常量化枚举列表中查找匹配的参数组合 */
void FlashAttentionScoreTilingBase::SetScalarConst()
{
    SetMatmulConstArr();
    if (!(CheckScalarConstCondation())) {
        return;
    }
    for (auto &matmulConst : matmulConstArr) {
        if (MatchMatmulConst(matmulConst)) {
            s1TemplateType = matmulConst.s1Type;
            s2TemplateType = matmulConst.s2Type;
            dTemplateType = matmulConst.dType;
            break;
        }
    }
    OP_LOGI(context_, "SetScalarConst_[s1TemplateType: %d, s2TemplateType: %d, dTemplateType: %d]",
              static_cast<int>(s1TemplateType), static_cast<int>(s2TemplateType), static_cast<int>(dTemplateType));
}

ge::graphStatus FlashAttentionScoreTilingBase::GetWorkspaceSize()
{
    auto &tensorSizeParams = tilingData->tensorSizeParams;
    auto &coreParams = tilingData->coreParams;

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    int64_t bmm1Byetes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;
    int64_t bmm2Byetes = tensorSizeParams.get_bmm2ResUbSize() * calcTypeSize;
    workspaces[0] = static_cast<size_t>((bmm1Byetes + bmm2Byetes) * aivNum) + WORK_SPACE_RESERVE_SIZE;
    return ge::GRAPH_SUCCESS;
}

class FlashAttentionScoreTilingS1Bn2gs1 : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionScoreTilingS1Bn2gs1(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context)
    {
        expectTemplate.splitS1 = 1;
        expectTemplate.splitD = 1;
        templateName = "FlashAttentionScoreS1Bn2gs1";
    }
    ~FlashAttentionScoreTilingS1Bn2gs1() override = default;

protected:
    int64_t s1Ratio = 1;
    int64_t workspaceLimit = 131072; // 8*128*128
    int64_t softmaxExtraSize = 512;
    int64_t s1dHighPerfBufferNum = 4;
    int64_t s2SizeLimitMax = 1024;
    int64_t s2SizeLimitMin = 128;
    int64_t isBasicBlockNum = 64;
    int64_t minSizeLimit = 65536; // 64 * 1024
    int64_t nRatioMax = 4;
    int64_t highPerfBlock = 128;
    int64_t l1SizeRemain = 0;
    int64_t elementSize = 4;
    int64_t nzndDataLimit = 20480; // 20 * 1024
    int64_t s2SizeNzndMinLimit = 704;
    int64_t dSizeLimit = 256;
    int64_t aicRatio = 1;
    int64_t aicRatioL1reuse = 2;
    bool enableL1Reuse = false;

    bool AnalyzeDtype() override
    {
        OP_CHECK_IF(!FlashAttentionScoreTilingBase::AnalyzeDtype(),
                   OP_LOGE(opName, "fail to analyze base dtype."), return false);
        bmm2OutDtype = bmmDtype;
        return true;
    }

    void SetEnableL1Reuse()
    {
        // FP32场景，不开启L1reuse
        if (inputDtypeBytes == DATA_TYPE_FP32) {
            enableL1Reuse = false;
            return;
        }
        // 使能增量L1reuse条件：s2>=512且BNG>64且D<=128
        // 增量L1reuse与原始L1reuse互斥
        if ((s2Size >= L1REUSE_S2_LIMIT_512 && bSize * n2Size * gSize > L1REUSE_BNG_LIMIT_64 &&
             dSize <= L1REUSE_D_LIMIT) ||
            (bSize * n2Size * gSize >= L1REUSE_BNG_LIMIT_4800 && s2Size == BMM2_BASICBLOCK_K_256 &&
             dSize == L1REUSE_D_LIMIT_144)) {
            enableL1Reuse = true;
            aicRatio = aicRatioL1reuse;
        }
        // 原始L1reuse说明
        // 因为一个Cube对应两个Vector, 一共需要两份L1空间存放Bmm2的右矩阵S2 * D
        // Nz的shape需要将s2Size对齐到16来计算剩余空间
        // 如果s2SizeALign16 * dSizeAlign16大于64K，则不使能该优化
        // L1reuse入口条件逻辑为：
        // 512<=S2<=1024且S1<=3840且D=64，并在非稀疏时开启
        if ((alignedS2 >= S2_REUSE_SIZE_512 && alignedS2 <= S2_REUSE_SIZE_1024) && s1Size <= S1_REUSE_SIZE_3840 &&
            alignedD == D_SPECIFIC_SIZE && (tilingData->inputParams.get_sparseType() == 0)) {
            tilingKeyBmm2Source = CubeInputSourceEnum::L1;
            enableL1Reuse = false;
            aicRatio = 1;
            if (alignedS2 == S2_REUSE_SIZE_512) {
                nRatioMax = 1;
            }
        }
    }

    void SetMultiCoreParams() override
    {
        auto &multiCoreParams = tilingData->multiCoreParams;
        auto &coreParams = tilingData->coreParams;
        int64_t totalSize = coreParams.get_bOuterSize() * coreParams.get_n2OuterSize() * coreParams.get_gOuterSize() *
                            coreParams.get_s1OuterSize();
        int64_t actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
        multiCoreParams.set_coreNum(static_cast<int32_t>(actualUsedAivNum));
        multiCoreParams.set_totalSize(totalSize);
        multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualUsedAivNum) * aicRatio);
        multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
    }

    void SetCoreParams() override
    {
        SetEnableL1Reuse();
        // 稀疏场景不开启S1轴N:1配比
        FlashAttentionScoreTilingBase::SetCoreParams();
        if (tilingData->inputParams.get_sparseType() != static_cast<uint8_t>(SparseEnum::ALL)) {
            return;
        }
        SetMultiCoreParams();
        auto &coreParams = tilingData->coreParams;
        auto &multiCoreParams = tilingData->multiCoreParams;
        // 对于S2 < 128且S1 16对齐的场景，bmm1输出改成NZ格式，配比设为4，提升fix pipe效率
        if (alignedS2 < s2SizeLimitMin) {
            nRatioMax = 1;
        }
        // NZND入口条件逻辑为
        // 1、S2=64时开启；
        // 2、S2非64对齐时开启；
        // 3、S2大于s2SizeNzndMinLimit，且64对齐但非128对齐，BNGS1数据量大于nzndDataLimit时开启；
        // 4、满足以上条件且D>256时，开启NZND但不改变N配比；
        if ((s2Size % S2_NZTOND_SIZE_64 != 0 || s2Size == S2_NZTOND_SIZE_64) ||
            (s2Size >= s2SizeNzndMinLimit && s2Size % S2_NZTOND_SIZE_64 == 0 && s2Size % S2_NZTOND_SIZE_128 != 0 &&
             bSize * n2Size * gSize * s1Size > nzndDataLimit)) {
            if (dSize <= dSizeLimit) {
                nRatioMax = 4;
            }
            tilingKeyBmm1Format = CubeFormatEnum::NZ;
        }
        // 当前能分满核，考虑增大N
        while (s1Ratio < nRatioMax && multiCoreParams.get_totalSize() > ((GetNRatio() - 1) * aivNum / GetNRatio()) &&
               s1BasicBlock * GetNRatio() < alignedS1 && GetNRatio() * s1BasicBlock * alignedS2 <= workspaceLimit) {
            s1Ratio++;
            FlashAttentionScoreTilingBase::SetCoreParams();
            // S1轴N:1使能
            coreParams.set_s1OuterSize(CeilDivision(coreParams.get_s1OuterSize(), GetNRatio()));
            coreParams.set_s1BaseSize(s1BasicBlock * GetNRatio());
            coreParams.set_s1BaseTailSize(CalcTailSize(s1Size, s1BasicBlock * GetNRatio()));
            SetMultiCoreParams();
        }
        // 分不满核，减小N
        while (multiCoreParams.get_totalSize() <= ((GetNRatio() - 1) * aivNum / GetNRatio()) ||
               s1BasicBlock * GetNRatio() > alignedS1 || GetNRatio() * s1BasicBlock * alignedS2 > workspaceLimit) {
            s1Ratio--;
            FlashAttentionScoreTilingBase::SetCoreParams();
            coreParams.set_s1OuterSize(CeilDivision(coreParams.get_s1OuterSize(), GetNRatio()));
            coreParams.set_s1BaseSize(s1BasicBlock * GetNRatio());
            coreParams.set_s1BaseTailSize(CalcTailSize(s1Size, s1BasicBlock * GetNRatio()));
        }
    }

    void SetSparseParams() override
    {
        if (tilingData->inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::ALL)) {
            return;
        }
        auto &coreParams = tilingData->coreParams;
        coreParams.set_s1SparseValidSize(s1SparseValidSize);
        coreParams.set_s2SparseValidSize(s2SparseValidSize);
        auto &multiCoreParams = tilingData->multiCoreParams;
        if (tilingData->inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::PREFIX)) {
            std::vector<std::vector<int64_t>> sparseValidArray;
            for (int64_t bIdx = 0; bIdx < bSize; bIdx++) {
                sparseValidArray.emplace_back(std::vector<int64_t>(coreParams.get_s1OuterSize(), 0));
                InitSparseValidArray(sparseValidArray.back(), bIdx);
            }
            SetPrefixSparseStartIdx(sparseValidArray, multiCoreParams);
        } else {
            std::vector<int64_t> sparseValidArray(coreParams.get_s1OuterSize(), 0);
            InitSparseValidArray(sparseValidArray, 0);
            SetSparseStartIdx(sparseValidArray, multiCoreParams);
        }
    }

    int64_t GetNRatio() override
    {
        return s1Ratio;
    }

    bool CalcUBSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock) override
    {
        apiMaxUBSize = tmpS1BasicBlock * tmpS2BasicBlock * sizeof(float) + softmaxExtraSize;
        return true;
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = s1dHighPerfBufferNum;
    }

    void CalcS1S2BasicBlock([[maybe_unused]] const BufferNum &bufferNum) override
    {
        s1BasicBlock = std::min(128L, alignedS1); // PERFORMANCE OPT
        s2BasicBlock = std::min(128L, alignedS2);
    }

    void CalcDBasicBlock() override
    {
        dBasicBlock = std::min(128L, alignedD);
    }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, [[maybe_unused]] int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        bmm1.SetShape(std::min(static_cast<int64_t>(tilingData->coreParams.get_s1BaseSize()), s1Size), s2Size, dSize);
        bmm1.SetOrgShape(s1Size, s2Size, s1StrideSize, s2StrideSize);
        bmm1.SetBias(false);
        if (bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        if (bmm1.SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock) != 0) {
            return false;
        }
        if (tilingKeyBmm2Source == CubeInputSourceEnum::L1) {
            l1SizeRemain = aicoreParams_.l1Size - alignedS2 * alignedD * elementSize;
        } else {
            l1SizeRemain = aicoreParams_.l1Size;
        }
        if (bmm1.SetBufferSpace(l1SizeRemain, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, [[maybe_unused]] int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock,
                            [[maybe_unused]] int64_t batch, matmul_tiling::MatmulApiTiling &bmm2) override
    {
        bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(std::min(static_cast<int64_t>(tilingData->coreParams.get_s1BaseSize()), s1Size), dSize, s2Size);
        bmm2.SetOrgShape(s1Size, s2StrideSize, s2Size, s2StrideSize);
        bmm2.SetBias(false);
        if (bmm2.SetBufferSpace(l1SizeRemain, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        // 在S1=S2，S2大于S2_SPECIFIC_SIZE_928且D=64时，使用性能更亲和的BMM2基本块
        if (s1Size == s2Size && s2Size >= S2_SPECIFIC_SIZE_928 && dSize == D_SPECIFIC_SIZE &&
            tilingData->inputParams.get_sparseType() == 0) {
            if (bmm2.SetFixSplit(BMM2_BASICBLOCK_M_64, BMM2_BASICBLOCK_N_64, BMM2_BASICBLOCK_K_256) != 0) {
                return false;
            }
        } else {
            if (bmm2.SetFixSplit(tmpS1BasicBlock, tmpDBasicBlock) != 0) {
                return false;
            }
        }
        return true;
    }

    bool IsTemplateMatched() const override
    {
        if (s2Size > s2SizeLimitMax) {
            return false;
        }
        if (s2Size > s2SizeLimitMin) {
            return true;
        }
        if (static_cast<uint64_t>(n2Size * gSize * ((alignedS1 + alignedS2) * alignedD + alignedS2)
            * inputDtypeBytes) >= aicoreParams_.l1Size ||
            static_cast<uint64_t>(n2Size * gSize * (alignedS1 + alignedD) * alignedS2
            * inputDtypeBytes) >= aicoreParams_.l1Size) {
            return true;
        }
        if (n2Size * gSize * alignedS1 * alignedS2 * inputDtypeBytes <= minSizeLimit * DATA_TYPE_FP16) {
            return false;
        }
        return true;
    }

     void SetMatmulConstArr() override
    {
        matmulConstArr = {
            {128, 128, 80, STemplateType::ALIGNED_128, STemplateType::ALIGNED_128, DTemplateType::ALIGNED_80},
            {128, 128, 96, STemplateType::ALIGNED_128, STemplateType::ALIGNED_128, DTemplateType::ALIGNED_96},
            {128, 128, 128, STemplateType::ALIGNED_128, STemplateType::ALIGNED_128, DTemplateType::ALIGNED_128}};
    }

    bool CheckScalarConstCondation() override
    {
        if (mm1BaseM == mm2BaseM && mm1BaseN == mm2BaseK && mm1BaseK == mm2BaseN && hasDropOut == false &&
            hasAttenMask == false && hasPse == false && tilingKeyDType == DtypeEnum::BFLOAT16) {
            if (tilingKeyLayout == LayoutType::LAYOUT_BSH ||
                (tilingKeyLayout == LayoutType::LAYOUT_SBH &&
                 mm1BaseK == static_cast<int64_t>(DTemplateType::ALIGNED_96) * FRACTAL_NUM)) {
                return true;
            }
        }
        return false;
    }

    bool MatchMatmulConst(MatmulConstParams &matmulConst) override
    {
        if (enableL1Reuse == true) {
            return false;
        }
        return mm1BaseM == matmulConst.baseM && mm1BaseN == matmulConst.baseN && mm1BaseK == matmulConst.baseK;
    }

    uint64_t GetTilingKey() const override
    {
        return GET_TPL_TILING_KEY(0, static_cast<uint8_t>(AxisEnum::S1), static_cast<uint8_t>(AxisEnum::D),
            static_cast<uint8_t>(AxisEnum::NONE), static_cast<uint8_t>(implMode), static_cast<uint8_t>(tilingKeyDType),
            static_cast<uint8_t>(tilingKeyLayout), static_cast<uint8_t>(tilingKeyBmm1Format),
            static_cast<uint8_t>(tilingKeyBmm2Source), static_cast<uint8_t>(SparseEnum::ANY),
            static_cast<uint8_t>(PerformanceOrientedEnum::BIG_DOUBLE_BUFFER), static_cast<uint8_t>(hasDropOut),
            static_cast<uint8_t>(hasAttenMask), static_cast<uint8_t>(hasPse), static_cast<uint8_t>(enableL1Reuse),
            static_cast<uint8_t>(hasRope), static_cast<uint8_t>(matmulPolicyType), 
            static_cast<uint8_t>(s1TemplateType), static_cast<uint8_t>(s2TemplateType), static_cast<uint8_t>(dTemplateType));
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        if (tilingData->inputParams.get_sparseType() == 0) {
            int32_t actualUsedAivNum = CeilDivision(tilingData->multiCoreParams.get_totalSize(),
                                                    (tilingData->multiCoreParams.get_splitFactorSize() / aicRatio));
            int32_t actualUsedAivNumMod2 = actualUsedAivNum % 2;
            if (enableL1Reuse && actualUsedAivNumMod2) {
                actualUsedAivNum++;
            }
            tilingData->multiCoreParams.set_coreNum(std::min(int32_t(aivNum), actualUsedAivNum));
        }
        tilingData->bmm1TilingData.shareL1Size = l1SizeRemain;
        tilingData->bmm2TilingData.shareL1Size = l1SizeRemain;
        auto &coreParams = tilingData->coreParams;
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Size = 0;
        int64_t bmm1AlignBytes = 0;
        int64_t s2SizeAlign16 = CeilDivision(s2Size, 16L) * 16L;
        bmm1Size = CeilDivision(coreParams.get_s1BaseSize() * s2SizeAlign16, 256L) * 256L;
        bmm1AlignBytes = bmm1Size * calcTypeSize * 2;
        // bmm1和stage1的workspace不能复用
        int64_t stage1AlignBytes = bmm1AlignBytes * inputDtypeBytes / DATA_TYPE_FP32;

        /* 计算bmm2需要用的workspace, 大小为CoreNum * s1BaseSize * alignedD (16对齐）,
         * bmm2计算完成后将数据输出在这块workspace上。
         * 这块workspace主要的作用是存放bmm2的后继输出，用来做div softmax sum和cast。 */
        int64_t bmm2AlignBytes = CeilDivision(coreParams.get_s1BaseSize() * alignedD, 256L) * 256L * calcTypeSize * 2;
        workspaces[0] = static_cast<size_t>((bmm1AlignBytes + stage1AlignBytes + bmm2AlignBytes) *
                                            tilingData->multiCoreParams.get_coreNum()) +
                        WORK_SPACE_RESERVE_SIZE;
        if (pseType == PSE_INNER_MUL_ADD_TYPE || pseType == PSE_INNER_MUL_ADD_SQRT_TYPE) {
            pseAlibiBaseS2 = alignedS2;
            pseAlibiBaseS1 = std::min(static_cast<int64_t>(coreParams.get_s1BaseSize()),
                                      UB_BASIC_LIMIT_SIZE / pseAlibiBaseS2);
            pseAlibiBaseS1 = std::max(pseAlibiBaseS1, UB_BASIC_LIMIT_SIZE / coreParams.get_s1BaseSize());
        }
        return ge::GRAPH_SUCCESS;
    }
};

class FlashAttentionScoreTilingB : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionScoreTilingB(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context)
    {
        templateName = "FlashAttentionScoreB";
    }
    ~FlashAttentionScoreTilingB() override = default;

protected:
    int64_t blockBSizeLimit_ = 64 * 1024;
    int64_t blockBL2SizeLimit_ = 128 * 1024;
    int64_t blockBUBSizeLimit_ = 8 * 1024;
    int64_t maxS1BaseSize_ = 256;
    int64_t dVec2BasicBlock_ = 1;
    int64_t s1Vec2BasicBlock_ = 1;

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = HIGH_PERF_BUFFER_NUM;
    }

    void CalcS1S2BasicBlock([[maybe_unused]] const BufferNum &bufferNum) override
    {
        s2BasicBlock = alignedS2;
        s1BasicBlock = blockBUBSizeLimit_ / s2BasicBlock / FRACTAL_NUM * FRACTAL_NUM;
        s1BasicBlock = std::min(s1BasicBlock, alignedS1);
        s1BasicBlock = std::min(maxS1BaseSize_, s1BasicBlock);
        dVec2BasicBlock_ = alignedD;
        if (dVec2BasicBlock_ <= S1_VEC2_BASE_SIZE_MAX) {
            s1Vec2BasicBlock_ = blockBUBSizeLimit_ / dVec2BasicBlock_ / FRACTAL_NUM * FRACTAL_NUM *
                                DATA_TYPE_FP16 / inputDtypeBytes;
        } else {
            s1Vec2BasicBlock_ = blockBUBSizeLimit_ / dVec2BasicBlock_ *
                                DATA_TYPE_FP16 / inputDtypeBytes;
        }
        s1Vec2BasicBlock_ = std::min(s1Vec2BasicBlock_, alignedS1);
    }

    void SetCoreParams() override
    {
        auto &coreParams = tilingData->coreParams;
        auto &inputParams = tilingData->inputParams;
        int64_t n2 = inputParams.get_n2Size();
        int64_t g = inputParams.get_gSize();
        int64_t b = inputParams.get_bSize();
        int64_t s1 = inputParams.get_s1Size();
        int64_t bIn = 1;
        coreParams.set_bBaseSize(bIn);
        coreParams.set_bBaseTailSize(CalcTailSize(b, bIn));
        coreParams.set_bOuterSize(CeilDivision(b, bIn));
        coreParams.set_s1BaseSize(s1BasicBlock);
        coreParams.set_s1BaseTailSize(CalcTailSize(s1, s1BasicBlock));
        coreParams.set_s1OuterSize(CeilDivision(s1, s1BasicBlock));
        coreParams.set_s2BaseSize(s2BasicBlock);
        coreParams.set_s2BaseTailSize(CalcTailSize(s2Size, s2BasicBlock));
        coreParams.set_s2OuterSize(CeilDivision(s2Size, s2BasicBlock));
        coreParams.set_s1Vec2BaseSize(s1Vec2BasicBlock_);
        coreParams.set_s1Vec2BaseTailSize(CalcTailSize(s1, s1Vec2BasicBlock_));
        coreParams.set_s1Vec2OuterSize(CeilDivision(s1, s1Vec2BasicBlock_));
        coreParams.set_dBaseSize(dBasicBlock);
        coreParams.set_dBaseTailSize(CalcTailSize(dSize, dBasicBlock));
        coreParams.set_dOuterSize(CeilDivision(dSize, dBasicBlock));
        coreParams.set_s1SparseValidSize(s1SparseValidSize);
        coreParams.set_s2SparseValidSize(s2SparseValidSize);
        batchBasic = coreParams.get_bBaseSize() * n2 * g;
        OP_LOGD(context_, "[b:%ld, n2:%ld, g:%ld, s1:%ld, s2:%ld, batchBasic:%ld].", b, n2, g, s1,
                  inputParams.get_s2Size(), batchBasic);
        OP_LOGD(context_, "[bBaseSize: %d, bBaseTailSize:%d, bOuterSize:%ld].", coreParams.get_bBaseSize(),
                  coreParams.get_bBaseTailSize(), coreParams.get_bOuterSize());
        OP_LOGD(context_, "[s1BaseSize: %d, s1BaseTailSize:%d, s1OuterSize:%ld].", coreParams.get_s1BaseSize(),
                  coreParams.get_s1BaseTailSize(), coreParams.get_s1OuterSize());
        OP_LOGD(context_, "[s1Vec2BaseSize: %d, s1Vec2BaseTailSize:%d, s1Vec2OuterSize: %ld].",
                  coreParams.get_s1Vec2BaseSize(), coreParams.get_s1Vec2BaseTailSize(),
                  coreParams.get_s1Vec2OuterSize());
        OP_LOGD(context_, "[s2BaseSize: %d, s2BaseTailSize:%d, s2OuterSize:%ld].", coreParams.get_s2BaseSize(),
                  coreParams.get_s2BaseTailSize(), coreParams.get_s2OuterSize());
    }

    void SetMultiCoreParams() override
    {
        auto &multiCoreParams = tilingData->multiCoreParams;
        auto &coreParams = tilingData->coreParams;
        int64_t totalSize = coreParams.get_bOuterSize(); // 核间一共处理的Bo大小
        int64_t tempUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
        multiCoreParams.set_totalSize(totalSize);
        multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, tempUsedAivNum)); // 每个核处理的Bo大小
        multiCoreParams.set_splitFactorTailSize(
            CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize())); // 最后一个核处理的Bo大小
        multiCoreParams.set_coreNum(
            static_cast<int32_t>(CeilDivision(totalSize, multiCoreParams.get_splitFactorSize())));
        OP_LOGD(context_,
                  "[totalSize:%ld, tempUsedAivNum:%ld, splitFactorSize:%ld, splitFactorTailSize:%ld, coreNum: %d].",
                  totalSize, tempUsedAivNum, multiCoreParams.get_splitFactorSize(),
                  multiCoreParams.get_splitFactorTailSize(), multiCoreParams.get_coreNum());
    }

    void SetTensorSizeParams() override
    {
        auto &tensorSizeParams = tilingData->tensorSizeParams;
        auto &coreParams = tilingData->coreParams;
        tensorSizeParams.set_bmm1ResUbSize(s1BasicBlock * s2BasicBlock);
        tensorSizeParams.set_attenMaskUbSize(attenMaskExistFlag * s1BasicBlock * s2BasicBlock);
        tensorSizeParams.set_pseUbSize(pseExistFlag * s1BasicBlock * s2BasicBlock);
        tensorSizeParams.set_dropMaskUbSize(dropMaskExistFlag * s1BasicBlock *
                                            AlignUp(s2BasicBlock, DROP_MASK_ALIGN_UNIT) / BYTE_BIT_NUM /
                                            inputDtypeBytes);
        if (tensorSizeParams.get_pseUbSize() > 0) {
            hasPse = true;
        }
        if (tensorSizeParams.get_dropMaskUbSize() > 0) {
            hasDropOut = true;
        }
        if (tensorSizeParams.get_attenMaskUbSize() > 0) {
            hasAttenMask = true;
        }
        if (inputDtype == ge::DT_BF16 || isHighPercision) {
            if (expectTemplate.splitS2 == 1) {
                tensorSizeParams.set_castUbSize(s1BasicBlock * std::max(s2BasicBlock, dBasicBlock));
            } else {
                tensorSizeParams.set_castUbSize(s1BasicBlock * s2BasicBlock);
            }
        }
        tensorSizeParams.set_softmaxMaxUbSize(s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
        tensorSizeParams.set_softmaxSumUbSize(s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
        tensorSizeParams.set_softmaxExpUbSize(s1BasicBlock * (BYTE_BLOCK / calcTypeSize));
        tensorSizeParams.set_apiTmpBufferBytes(apiMaxUBSize);
        tensorSizeParams.set_bmm2ResUbSize(coreParams.get_s1Vec2BaseSize() * dBasicBlock);
    }

    void CalcDBasicBlock() override
    {
        dBasicBlock = alignedD;
    }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        bmm1.SetShape(s1Size, s2Size, dSize);
        bmm1.SetOrgShape(s1Size, s2Size, s1StrideSize, s2StrideSize);
        bmm1.SetALayout(bSize, s1Size, n2Size, gSize, dSize);
        bmm1.SetBLayout(bSize, s2Size, n2Size, 1, dSize);
        bmm1.SetCLayout(bSize, s1Size, n2Size, gSize, s2Size);
        bmm1.SetBatchNum(batch);
        bmm1.SetBias(false);
        if (bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        if (bmm1.SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock) != 0) {
            return false;
        }
        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, [[maybe_unused]] int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock,
                            int64_t batch, matmul_tiling::MatmulApiTiling &bmm2) override
    {
        bmm2.SetAType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::NZ, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(s1Size, dSize, s2Size);
        bmm2.SetOrgShape(s1Size, dSize, s2Size); // consider broadcst, N same as A tensor
        bmm2.SetALayout(bSize, s1Size, n2Size, gSize, s2Size);
        bmm2.SetBLayout(bSize, s2Size, n2Size, 1, dSize);
        bmm2.SetCLayout(bSize, s1Size, n2Size, gSize, dSize);
        bmm2.SetBatchNum(batch);
        bmm2.SetBias(false);
        if (bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        int64_t maxDBasicBlock = AlignDown(aicoreParams_.l0cSize / (tmpS1BasicBlock * calcTypeSize), 16UL);
        if (bmm2.SetFixSplit(tmpS1BasicBlock, std::min(maxDBasicBlock, tmpDBasicBlock)) != 0) {
            return false;
        }
        return true;
    }

    void SetMatmulConstArr() override
    {
        matmulConstArr = {
            {16, 16, 80, STemplateType::ALIGNED_16, STemplateType::ALIGNED_16, DTemplateType::ALIGNED_80},
            {32, 32, 80, STemplateType::ALIGNED_32, STemplateType::ALIGNED_32, DTemplateType::ALIGNED_80},
            {48, 48, 80, STemplateType::ALIGNED_48, STemplateType::ALIGNED_48, DTemplateType::ALIGNED_80},
            {64, 64, 80, STemplateType::ALIGNED_64, STemplateType::ALIGNED_64, DTemplateType::ALIGNED_80},
            {16, 16, 96, STemplateType::ALIGNED_16, STemplateType::ALIGNED_16, DTemplateType::ALIGNED_96},
            {32, 32, 96, STemplateType::ALIGNED_32, STemplateType::ALIGNED_32, DTemplateType::ALIGNED_96}};
    }

    bool CheckScalarConstCondation() override
    {
        if (mm1BaseM == mm2BaseM && mm1BaseN == mm2BaseK && mm1BaseK == mm2BaseN &&
            tilingKeyDType == DtypeEnum::BFLOAT16 && tilingKeyLayout == LayoutType::LAYOUT_BSH &&
            hasDropOut == false && hasAttenMask == false && hasPse == false) {
            return true;
        }
        return false;
    }

    uint64_t GetTilingKey() const override
    {
        return GET_TPL_TILING_KEY(0, static_cast<uint8_t>(AxisEnum::NONE), static_cast<uint8_t>(AxisEnum::NONE),
            static_cast<uint8_t>(AxisEnum::B), static_cast<uint8_t>(implMode), static_cast<uint8_t>(tilingKeyDType),
            static_cast<uint8_t>(tilingKeyLayout), static_cast<uint8_t>(tilingKeyBmm1Format),
            static_cast<uint8_t>(tilingKeyBmm2Source), static_cast<uint8_t>(SparseEnum::NONE),
            static_cast<uint8_t>(PerformanceOrientedEnum::BIG_DOUBLE_BUFFER), static_cast<uint8_t>(hasDropOut),
            static_cast<uint8_t>(hasAttenMask), static_cast<uint8_t>(hasPse), static_cast<uint8_t>(enableL1Reuse),
            static_cast<uint8_t>(hasRope), static_cast<uint8_t>(matmulPolicyType), 
            static_cast<uint8_t>(s1TemplateType), static_cast<uint8_t>(s2TemplateType), static_cast<uint8_t>(dTemplateType));
    }

    bool IsCapable() override
    {
        auto &inputParams = tilingData->inputParams;
        int64_t n2 = inputParams.get_n2Size();
        int64_t g = inputParams.get_gSize();
        bool notMatched = false;
        if (alignedS2 > HIGH_PERF_SUPPORT_S2_BASIC) {
            notMatched = true;
        }
        if (n2 * g * alignedS1 * alignedS2 * inputDtypeBytes > blockBSizeLimit_ * DATA_TYPE_FP16) {
            notMatched = true;
        }
        if (notMatched) {
            OP_LOGE(context_,
                      "[%s]not match template[%s], input params: bn2gs1s2d[%ld, %ld, %ld, %ld, %ld, %ld], "
                      "keepProb[%f]",
                      templateName, expectTemplate.ToString().c_str(), inputParams.get_bSize(),
                      inputParams.get_n2Size(), inputParams.get_gSize(), inputParams.get_s1Size(),
                      inputParams.get_s2Size(), inputParams.get_dSize(), inputParams.get_keepProb());
            return false;
        }
        return true;
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

    ge::graphStatus GetWorkspaceSize() override
    {
        auto &inputParams = tilingData->inputParams;
        auto &coreParams = tilingData->coreParams;
        auto &multiCoreParams = tilingData->multiCoreParams;
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Byetes = coreParams.get_bBaseSize() * inputParams.get_n2Size() * inputParams.get_gSize() *
                             inputParams.get_s1Size() * alignedS2 * calcTypeSize * inputDtypeBytes / DATA_TYPE_FP16;
        int64_t bmm2Byetes = coreParams.get_bBaseSize() * inputParams.get_n2Size() * inputParams.get_gSize() *
                             inputParams.get_s1Size() * alignedD * calcTypeSize;
        bmm1Byetes = AlignUp(bmm1Byetes, GM_ALIGN);
        bmm2Byetes = AlignUp(bmm2Byetes, GM_ALIGN);
        size_t pingPongNum = 2;
        workspaces[0] = WORK_SPACE_RESERVE_SIZE +
                        static_cast<size_t>((bmm1Byetes + bmm2Byetes) * pingPongNum * multiCoreParams.get_coreNum());

        if (pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_TYPE) ||
            pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            pseAlibiBaseS2 = alignedS2;
            pseAlibiBaseS1 = std::min(s1BasicBlock, UB_BASIC_LIMIT_SIZE / pseAlibiBaseS2);
            pseAlibiBaseS1 = std::max(pseAlibiBaseS1, UB_BASIC_LIMIT_SIZE / s1BasicBlock);
        }
        return ge::GRAPH_SUCCESS;
    }

    void SetSoftMaxTiling() override
    {
        auto softmaxShape = ge::Shape({s1BasicBlock, s2BasicBlock});
        AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                          tilingData->softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
    }
};

class FlashAttentionScoreTilingS1s2Bn2gs1 : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionScoreTilingS1s2Bn2gs1(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context)
    {
        expectTemplate.splitS1 = 1U;
        expectTemplate.splitS2 = 1U;
        templateName = "FlashAttentionScoreS1s2Bn2gs1";
    }
    ~FlashAttentionScoreTilingS1s2Bn2gs1() override = default;

protected:
    int64_t s2sizeLimitMin = 1024;
    int64_t dAlignSize = 16;
    bool enableL1Reuse = false;

    int64_t GetNRatio() override
    {
        return nRatio;
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = HIGH_PERF_BUFFER_NUM;
    }

    void CalcS1S2BasicBlock([[maybe_unused]] const BufferNum &bufferNum) override
    {
        s1BasicBlock = std::min(64L, alignedS1);
        // d轴为64
        if (bSize * n1Size * gSize * CeilDiv(s1Size, s1BasicBlock) > static_cast<int64_t>(aivNum)) {
            s1BasicBlock = std::min(128L, alignedS1);
        }
        s2BasicBlock = std::min(128L, alignedS2);
        if (s2Size % S2_NZTOND_SIZE_64 != 0 && dSize != D_SPECIFIC_SIZE) {
            tilingKeyBmm1Format = CubeFormatEnum::NZ;
        }
    }

    void CalcNRatio() override
    {
        nRatio = 8L;
        // s2Size (1025, 1040]
        if (s2Size / s2BasicBlock == 8L && s2Size % s2BasicBlock > 0 && s2Size % s2BasicBlock <= 16L) {
            nRatio = 5L;
        }
        if (s2Size <= 2304 && s2Size >= 2049 && dSize == 64 && tilingData->inputParams.get_layoutType() == LAYOUT_BNSD) {
            nRatio = 6L;
        }
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
               tilingData->inputParams.get_attenMaskShapeType() == ATTEN_1_1_1_S1_S2;
    }

    void SetEnableL1Reuse()
    {
        // FP32场景，不开启L1reuse
        if (inputDtypeBytes == DATA_TYPE_FP32) {
            enableL1Reuse = false;
            return;
        }
        if (dSize > L1REUSE_D_LIMIT) {
            OP_LOGD(context_, "Current condition [dSize(%ld) > L1REUSE_D_LIMIT(%ld)] does not enable L1Reuse", dSize,
                      L1REUSE_D_LIMIT);
            return;
        }
        if (dSize == D_SPECIFIC_SIZE && tilingData->inputParams.get_layoutType() == LAYOUT_BNSD &&
            !(s2Size % L1REUSE_S2_LIMIT_256 == 0 || s2Size == L1REUSE_S2_LIMIT_4032) &&
            s2Size != L1REUSE_S2_LIMIT_42192) {
            OP_LOGD(context_, "Current condition [dSize(%ld) && layout(BNSD)] does not enable L1Reuse", dSize);
            return;
        }
        if (tilingData->inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::ALL)) {
            enableL1Reuse = true;
            return;
        }

        if ((tilingData->inputParams.get_layoutType() == LAYOUT_BSND || tilingData->inputParams.get_layoutType() ==
            LAYOUT_BSH) && s2Size <= L1REUSE_S2_LIMIT_2048 && dSize <= D_SPECIFIC_SIZE &&
            bSize * n1Size <= L1REUSE_BNG_LIMIT) {
            OP_LOGD(context_, "Current condition [dSize(%ld) && layout(BSH/BSND) && BN(%ld)] does not enable L1Reuse",
                      dSize, bSize * n1Size);
            return;
        }

        if (tilingData->inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::CAUSAL) ||
            tilingData->inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::PREFIX)) {
            if (bSize * n1Size * gSize <= L1REUSE_BNG_LIMIT && s2Size <= L1REUSE_S2_LIMIT_2048) {
                OP_LOGD(context_, "Current condition [BNG(%ld) && s2Size(%ld)] does not enable L1Reuse.",
                          bSize * n1Size * gSize, s2Size);
                return;
            }
            enableL1Reuse = true;
            return;
        }

        if (tilingData->inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::BAND)) {
            if ((bSize * n1Size * gSize <= L1REUSE_BNG_LIMIT && maxValidS2Len <= L1REUSE_S2_LIMIT_2048) ||
                (maxValidS2Len <= L1REUSE_S2_LIMIT_1024)) {
                OP_LOGD(context_, "Current condition [BNG(%ld) && maxValidS2Len(%ld)] does not enable L1Reuse.",
                          bSize * n1Size * gSize, maxValidS2Len);
                return;
            }
            enableL1Reuse = true;
        }
    }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, [[maybe_unused]] int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        // 分不满核，且稀疏场景，shape设置的较小能产生更好的tiling
        bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size),
                      std::min(tmpS2BasicBlock * tilingData->coreParams.get_nRatio(), s2Size), dSize);
        bmm1.SetOrgShape(s1Size, tmpS2BasicBlock * tilingData->coreParams.get_nRatio(), s1StrideSize, s2StrideSize);
        bmm1.SetBias(false);
        if (bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        if (dSize > BMM1_BASICBLOCK_K_64 && dSize <= BMM1_BASICBLOCK_K_128 && inputDtypeBytes != DATA_TYPE_FP32) {
            int64_t baseM = std::min(tmpS1BasicBlock, AlignUp(s1Size, FRACTAL_NUM));
            bmm1.SetFixSplit(baseM, BMM1_BASICBLOCK_N_128, dSize);
        }

        if (IsSpecialShape()) {
            if (bmm1.SetFixSplit(BMM1_BASICBLOCK_M_128, BMM1_BASICBLOCK_N_256, BMM1_BASICBLOCK_K_64) != 0) {
                return false;
            }
        }
        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, [[maybe_unused]] int64_t tmpDBasicBlock,
                            [[maybe_unused]] int64_t batch, matmul_tiling::MatmulApiTiling &bmm2) override
    {
        int64_t singleM = std::min(tmpS1BasicBlock, s1Size);
        bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(singleM, dSize,
                      std::min(tmpS2BasicBlock * tilingData->coreParams.get_nRatio(), s2Size));
        bmm2.SetOrgShape(s1Size, s2StrideSize, std::min(tmpS2BasicBlock * tilingData->coreParams.get_nRatio(), s2Size),
                         s2StrideSize);
        bmm2.SetBias(false);
        if (bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        if (dSize == D_SPECIFIC_SIZE && tilingData->inputParams.get_layoutType() == LAYOUT_BNSD &&
            tilingData->inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::ALL) &&
            singleM >= BMM2_BASICBLOCK_M_64) {
            if (bmm2.SetFixSplit(BMM2_BASICBLOCK_M_64, BMM2_BASICBLOCK_M_64, BMM2_BASICBLOCK_K_256) != 0) {
                return false;
            }
        }
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        // not care about layout in tiling key, pass BSND(enum value is 0)
        return GET_TPL_TILING_KEY(0, static_cast<uint8_t>(AxisEnum::S1), static_cast<uint8_t>(AxisEnum::S2),
            static_cast<uint8_t>(AxisEnum::NONE), static_cast<uint8_t>(implMode), static_cast<uint8_t>(tilingKeyDType),
            static_cast<uint8_t>(tilingKeyLayout), static_cast<uint8_t>(tilingKeyBmm1Format),
            static_cast<uint8_t>(tilingKeyBmm2Source), static_cast<uint8_t>(SparseEnum::ANY),
            static_cast<uint8_t>(PerformanceOrientedEnum::BIG_DOUBLE_BUFFER), static_cast<uint8_t>(hasDropOut),
            static_cast<uint8_t>(hasAttenMask), static_cast<uint8_t>(hasPse), static_cast<uint8_t>(enableL1Reuse),
            static_cast<uint8_t>(hasRope), static_cast<uint8_t>(matmulPolicyType), 
            static_cast<uint8_t>(s1TemplateType), static_cast<uint8_t>(s2TemplateType), static_cast<uint8_t>(dTemplateType));
    }

    bool IsCapable() override
    {
        if (s2Size > s2sizeLimitMin) {
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
            auto &multiCoreParams = tilingData->multiCoreParams;
            int64_t totalSize = multiCoreParams.get_totalSize();
            multiCoreParams.set_splitFactorSize(
                CeilDivision(totalSize, static_cast<int64_t>(multiCoreParams.get_coreNum())) * AICAIV_RATIO_2);
            multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
        }
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        RefreshSplitFactor();

        auto &tensorSizeParams = tilingData->tensorSizeParams;
        auto &coreParams = tilingData->coreParams;

        size_t *workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Bytes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;

        // 切D场景，stage1占用2倍的空间，stage2占用4倍空间
        workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_2 +
                        SPACE_NUM_4 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                        WORK_SPACE_RESERVE_SIZE;
        // NZND场景，stage1占用3倍的空间，stage2占用4倍空间
        if (s2Size % S2_NZTOND_SIZE_64 != 0) {
            workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_3 +
                            SPACE_NUM_4 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                            WORK_SPACE_RESERVE_SIZE;
        }
        // FP32场景，stage1占用4倍的空间，stage2占用4倍空间
        if (inputDtypeBytes == DATA_TYPE_FP32) {
            workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_4 +
                            SPACE_NUM_4 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize) * aivNum) +
                            WORK_SPACE_RESERVE_SIZE;
        }
        if (pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_TYPE) ||
            pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            pseAlibiBaseS2 = s2sizeLimitMin;
            if (tilingData->inputParams.get_sparseType() != static_cast<uint8_t>(SparseEnum::ALL)) {
                pseAlibiBaseS1 = s1BasicBlock;
            } else {
                int64_t s2Tail = s2Size % s2sizeLimitMin;
                if (s2Tail != 0) {
                    pseAlibiBaseS1 = std::min(s1BasicBlock, UB_BASIC_LIMIT_SIZE / AlignUp(s2Tail, FRACTAL_NUM));
                } else {
                    pseAlibiBaseS1 = std::min(s1BasicBlock, UB_BASIC_LIMIT_SIZE / pseAlibiBaseS2);
                }
            }
            pseAlibiBaseS1 = std::max(pseAlibiBaseS1, UB_BASIC_LIMIT_SIZE / s1BasicBlock);
        }

        return ge::GRAPH_SUCCESS;
    }

    void SetSoftMaxTiling() override
    {
        auto softmaxShape = ge::Shape({s1BasicBlock / GetNRatio(), s2BasicBlock * GetNRatio()});

        AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                          tilingData->softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
    }

    bool SetPseAlibiParams() override
    {
        auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
        if (pseShape == nullptr || pseShape->GetStorageShape().GetDimNum() == 0) {
            return true;
        }
        if (pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_TYPE) ||
            pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            if (s1Size != s2Size) {
                OP_LOGE(context_, "INNER Pse alibi is supported only when s1Size and s2Size are equal.");
                return false;
            }
            return true;
        }
        // 2: pre last axiss
        if (pseShape->GetStorageShape().GetDimNum() < 2) {  // 2 is min dim num of legal pse
            OP_LOGE(context_, "Invalid Pse DimNum(%zu), PseType(%ld).",
                      pseShape->GetStorageShape().GetDimNum(), pseType);
            return false;
        }
        auto pseS1Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 2);
        auto pseS2Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 1);

        PseEncodeType pseEncodeType = PSE_ENCODE_NONE;
        if (pseS1Size == PSE_ALIBI_S_SIZE && s1Size > PSE_ALIBI_S_SIZE) {
            if (s1Size == s2Size) {
                OP_CHECK_IF(tilingData->inputParams.get_sparseType() != static_cast<uint8_t>(SparseEnum::CAUSAL),
                           OP_LOGE(opName, "Pse alibi only support causal sparse type."), return false);
                pseEncodeType = PSE_ENCODE_ALIBI_S2_FULL;
            } else {
                OP_LOGE(opName, "Pse alibi only support same S1 S2 when S1 lager than 1024");
                return false;
            }
        }
        tilingData->inputParams.set_pseEncodeType(pseEncodeType);
        tilingData->inputParams.set_pseS1Size(pseS1Size);
        tilingData->inputParams.set_pseS2Size(pseS2Size);
        return true;
    }
};

class FlashAttentionScoreTilingS1s2Bn2gs1SameAB : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionScoreTilingS1s2Bn2gs1SameAB(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context)
    {
        expectTemplate.splitS1 = 1U;
        expectTemplate.splitS2 = 1U;
        templateName = "FlashAttentionScoreTilingS1s2Bn2gs1SameAB";
    }
    ~FlashAttentionScoreTilingS1s2Bn2gs1SameAB() override = default;

protected:
    int64_t s2sizeLimitMin = 1024;
    int64_t dAlignSize = 16;
    int64_t sAlignSize = 64;

    int64_t GetNRatio() override
    {
        return nRatio;
    }

    void CalcNRatio() override
    {
        nRatio = 8L;
        // s2Size (1025, 1040]
        if (enableBestBlock &&
            (s2Size / s2BasicBlock == 8L && s2Size % s2BasicBlock > 0 && s2Size % s2BasicBlock <= 16L)) {
            nRatio = 5L;
        }
        if (s2Size <= 2304 && s2Size >= 2049 && dSize == 64 && tilingData->inputParams.get_layoutType() == LAYOUT_BNSD) {
            nRatio = 6L;
        }
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = HIGH_PERF_BUFFER_NUM;
    }

    void CalcS1S2BasicBlock([[maybe_unused]] const BufferNum &bufferNum) override
    {
        s1BasicBlockBest = 256L;
        if (enableBestBlock) {
            s1BasicBlockBest = CeilDivision(alignedS1, 384L) < CeilDivision(alignedS1, 256L) ? 384L : 256L;
        }
        s1BasicBlock = std::min(s1BasicBlockBest, alignedS1);
        s2BasicBlock = std::min(128L, alignedS2);
        s1VecBasicBlock = s1BasicBlock / AIV_AIC_NUM_RATIO;
        if (s2Size % S2_NZTOND_SIZE_64 != 0) {
            tilingKeyBmm1Format = CubeFormatEnum::NZ;
        }
    }

    void CalcDBasicBlock() override
    {
        dBasicBlock = std::min(128L, alignedD);
    }

    void SetMultiCoreParams() override
    {
        auto &multiCoreParams = tilingData->multiCoreParams;
        auto &coreParams = tilingData->coreParams;
        int64_t totalSize = coreParams.get_bOuterSize() * coreParams.get_n2OuterSize() * coreParams.get_gOuterSize() *
                            coreParams.get_s1OuterSize();
        int64_t actualUsedAicNum = std::min(totalSize, static_cast<int64_t>(aicNum));
        multiCoreParams.set_totalSize(totalSize);
        multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualUsedAicNum));
        multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
        actualUsedAicNum = CeilDivision(totalSize, multiCoreParams.get_splitFactorSize());
        if (totalSize > aicNum){
            multiCoreParams.set_coreNum(static_cast<int32_t>(aicNum * AIV_AIC_NUM_RATIO));
        }else{
            multiCoreParams.set_coreNum(static_cast<int32_t>(actualUsedAicNum * AIV_AIC_NUM_RATIO));
        }
    }

    bool IsSpecialShape()
    {
        return bSize == 8 && n1Size == 32 && n2Size == 32 && s1Size == 2048 && s2Size == 2048 && dSize == 128 &&
               preTokens == 2048 && nextTokens == 0 && inputLayout[0] == 'S' && inputLayout[1] == 'B' &&
               inputLayout[2] == 'H' && pseExistFlag == 0 && attenMaskExistFlag == 1 &&
               tilingData->inputParams.get_attenMaskShapeType() == ATTEN_1_1_1_S1_S2;
    }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, [[maybe_unused]] int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        // 分不满核，且稀疏场景，shape设置的较小能产生更好的tiling
        bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size),
                      std::min(tmpS2BasicBlock * tilingData->coreParams.get_nRatio(), s2Size), dSize);
        bmm1.SetOrgShape(s1Size, tmpS2BasicBlock * tilingData->coreParams.get_nRatio(), s1StrideSize, s2StrideSize);
        bmm1.SetBias(false);
        if (bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        if ((dSize > BMM1_BASICBLOCK_K_64 && dSize <= BMM1_BASICBLOCK_K_128) ||
             matmulPolicyType == MATMUL_POLICY_UNSPLITK) {
            int64_t baseM = std::min(BMM1_BASICBLOCK_M_128, AlignUp(s1Size, FRACTAL_NUM));
            int64_t baseN = std::min(BMM1_BASICBLOCK_N_128, AlignUp(s2Size, FRACTAL_NUM));
            bmm1.SetFixSplit(baseM, baseN, alignedD);
        }

        if (IsSpecialShape()) {
            if (bmm1.SetFixSplit(BMM1_BASICBLOCK_M_128, BMM1_BASICBLOCK_N_256, BMM1_BASICBLOCK_K_64) != 0) {
                return false;
            }
        }
        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, [[maybe_unused]] int64_t tmpDBasicBlock,
                            [[maybe_unused]] int64_t batch, matmul_tiling::MatmulApiTiling &bmm2) override
    {
        int64_t singleM = std::min(tmpS1BasicBlock, s1Size);
        bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(singleM, d2Size,
                      std::min(tmpS2BasicBlock * tilingData->coreParams.get_nRatio(), s2Size));
        bmm2.SetOrgShape(s1Size, vs2StrideSize, std::min(tmpS2BasicBlock * tilingData->coreParams.get_nRatio(), s2Size),
                         vs2StrideSize);
        bmm2.SetBias(false);
        if (bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        if (d2Size > BMM1_BASICBLOCK_K_64 && d2Size <= BMM1_BASICBLOCK_K_128) {
            int64_t baseM = std::min(BMM2_BASICBLOCK_M_128, AlignUp(s1Size, FRACTAL_NUM));
            int64_t baseN = std::min(BMM2_BASICBLOCK_N_128, AlignUp(d2Size, FRACTAL_NUM));
            bmm2.SetFixSplit(baseM, baseN);
        }
        if (matmulPolicyType == MATMUL_POLICY_UNSPLITK) {
            int64_t baseM = std::min(BMM2_BASICBLOCK_M_128, AlignUp(s1Size, FRACTAL_NUM));
            int64_t baseN = AlignUp(d2Size, FRACTAL_NUM);
            bmm2.SetFixSplit(baseM, baseN);
        }
        if (d2Size == D_SPECIFIC_SIZE && tilingData->inputParams.get_layoutType() == LAYOUT_BNSD &&
            tilingData->inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::ALL) &&
            singleM >= BMM2_BASICBLOCK_M_64) {
            if (bmm2.SetFixSplit(BMM2_BASICBLOCK_M_64, BMM2_BASICBLOCK_M_64, BMM2_BASICBLOCK_K_256) != 0) {
                return false;
            }
        }
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        return GET_TPL_TILING_KEY(0, static_cast<uint8_t>(AxisEnum::S1), static_cast<uint8_t>(AxisEnum::NONE),
            static_cast<uint8_t>(AxisEnum::NONE), static_cast<uint8_t>(implMode), static_cast<uint8_t>(tilingKeyDType),
            static_cast<uint8_t>(tilingKeyLayout), static_cast<uint8_t>(tilingKeyBmm1Format),
            static_cast<uint8_t>(tilingKeyBmm2Source), static_cast<uint8_t>(SparseEnum::ANY),
            static_cast<uint8_t>(PerformanceOrientedEnum::BIG_DOUBLE_BUFFER), static_cast<uint8_t>(hasDropOut),
            static_cast<uint8_t>(hasAttenMask), static_cast<uint8_t>(hasPse), static_cast<uint8_t>(enableL1Reuse),
            static_cast<uint8_t>(hasRope), static_cast<uint8_t>(matmulPolicyType), 
            static_cast<uint8_t>(s1TemplateType), static_cast<uint8_t>(s2TemplateType), static_cast<uint8_t>(dTemplateType));
    }

    bool IsCapable() override
    {
        if (inputDtypeBytes == DATA_TYPE_FP32) {
            return false;
        }
        if ((dSize % FRACTAL_NUM != 0 || dSize == D_SPECIFIC_SIZE_96) && (s2Size >= S2_REUSE_SIZE_512)) {
            isSameAB = true;
            if (s1Size > SAMEAB_BASIC_BLOCK_OPTIMIZE && tilingKeyLayout != LayoutType::LAYOUT_BSH
                && l2CacheSize > B4_L2_CACHESIZE) {
                enableBestBlock = true;
            }
            return true;
        }
        if (s2Size > s2sizeLimitMin && dSize > SAMEAB_D_LIMIT_128 && dSize < SAMEAB_D_LIMIT_196) {
            isSameAB = true;
            if (dSize == SAMEAB_D_LIMIT_192 && s1Size % HIGH_PERF_SUPPORT_S1_BASIC == 0 &&
                s2Size % HIGH_PERF_SUPPORT_S2_BASIC == 0 && inputLayout[0] == 'B' &&
                inputLayout[1] == 'N' && inputLayout[2] == 'S' && inputLayout[3] == 'D') {
                matmulPolicyType = MATMUL_POLICY_UNSPLITK;
            }
            return true;
        }
        // D=64切sameAB
        if (dSize == D_SPECIFIC_SIZE && s2Size % sAlignSize != 0 &&
            (s2Size > s2sizeLimitMin * 2 && s2Size < S2_SPECIFIC_SIZE_18432)) { // s2Size 大于最小限制的两倍且小于特定大小18432
            isSameAB = true;
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

    ge::graphStatus GetWorkspaceSize() override
    {
        auto &tensorSizeParams = tilingData->tensorSizeParams;
        auto &coreParams = tilingData->coreParams;

        size_t *workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Bytes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;

        // UB不常驻，stage1占用3倍的空间，stage2占用4倍空间
        workspaces[0] = static_cast<size_t>((bmm1Bytes * SPACE_NUM_3 +
                                            SPACE_NUM_4 * coreParams.get_s1BaseSize() * alignedD2 * calcTypeSize) *
                                            aicNum) + WORK_SPACE_RESERVE_SIZE;
        if (pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_TYPE) ||
            pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            pseAlibiBaseS2 = s2sizeLimitMin;
            if (tilingData->inputParams.get_sparseType() != static_cast<uint8_t>(SparseEnum::ALL)) {
                pseAlibiBaseS1 = s1VecBasicBlock;
            } else {
                int64_t s2Tail = s2Size % s2sizeLimitMin;
                if (s2Tail != 0) {
                    pseAlibiBaseS1 = std::min(s1VecBasicBlock, UB_BASIC_LIMIT_SIZE / AlignUp(s2Tail, FRACTAL_NUM));
                } else {
                    pseAlibiBaseS1 = std::min(s1VecBasicBlock, UB_BASIC_LIMIT_SIZE / pseAlibiBaseS2);
                }
                pseAlibiBaseS1 = std::max(pseAlibiBaseS1, UB_BASIC_LIMIT_SIZE / s1VecBasicBlock);
            }
        }

        return ge::GRAPH_SUCCESS;
    }

    void SetSoftMaxTiling() override
    {
        auto softmaxShape = ge::Shape({s1VecBasicBlock / GetNRatio(), s2BasicBlock * GetNRatio()});

        AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                          tilingData->softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
    }

    bool SetPseAlibiParams() override
    {
        auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
        if (pseShape == nullptr || pseShape->GetStorageShape().GetDimNum() == 0) {
            return true;
        }
        if (pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_TYPE) ||
            pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            if (s1Size != s2Size) {
                OP_LOGE(context_, "INNER Pse alibi is supported only when s1Size and s2Size are equal.");
                return false;
            }
            return true;
        }
        // 2: pre last axiss
        if (pseShape->GetStorageShape().GetDimNum() < 2) {  // 2 is min dim num of legal pse
            OP_LOGE(context_, "Invalid Pse DimNum(%zu), PseType(%ld).",
                      pseShape->GetStorageShape().GetDimNum(), pseType);
            return false;
        }
        auto pseS1Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 2);
        auto pseS2Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 1);

        PseEncodeType pseEncodeType = PSE_ENCODE_NONE;
        if (pseS1Size == PSE_ALIBI_S_SIZE && s1Size > PSE_ALIBI_S_SIZE) {
            if (s1Size == s2Size) {
                OP_CHECK_IF(tilingData->inputParams.get_sparseType() != static_cast<uint8_t>(SparseEnum::CAUSAL),
                           OP_LOGE(opName, "Pse alibi only support causal sparse type."), return false);
                pseEncodeType = PSE_ENCODE_ALIBI_S2_FULL;
            } else {
                OP_LOGE(opName, "Pse alibi only support same S1 S2 when S1 lager than 1024");
                return false;
            }
        }
        tilingData->inputParams.set_pseEncodeType(pseEncodeType);
        tilingData->inputParams.set_pseS1Size(pseS1Size);
        tilingData->inputParams.set_pseS2Size(pseS2Size);
        return true;
    }
};

class FlashAttentionVarLenScoreTiling : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionVarLenScoreTiling(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context)
    {
        expectTemplate.splitS1 = 1U;
        expectTemplate.splitS2 = 1U;
        templateName = "FlashAttentionVarLenScore";
    }
    ~FlashAttentionVarLenScoreTiling() override = default;
    using FlashAttentionScoreTilingBase::InitSparseValidArray;

protected:
    int64_t s2sizeLimitMax = 1024;

    int64_t GetNRatio() override
    {
        if (l2CacheSize <= B4_L2_CACHESIZE && isSameAB && alignedD == SAMEAB_D_LIMIT_128 && realT1Size >= B4_SEQ_LIMIT) {
            return 4L;
        } else {
            return 8L;
        }
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = HIGH_PERF_BUFFER_NUM;
    }

    void CalcS1S2BasicBlock([[maybe_unused]] const BufferNum &bufferNum) override
    {
        if (isSameAB) {
            s1BasicBlock = std::min(TND_S1_BASICBLOCK_256, alignedS1);
            s1VecBasicBlock = s1BasicBlock / 2;
        } else {
            s1BasicBlock = s1BasicBlockBest;
            s1BasicBlock = std::min(s1BasicBlock, alignedS1);
            if (attenMaskExistFlag) {
                s1BasicBlock = std::min(TND_S1_BASICBLOCK_256, s1BasicBlock);
            }
        }
        s2BasicBlock = std::min(TND_S1_BASICBLOCK_128, alignedS2);
    }

    void CalcDBasicBlock() override
    {
        dBasicBlock = std::min(128L, alignedD);
    }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, [[maybe_unused]] int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        // 分不满核，且稀疏场景，shape设置的较小能产生更好的tiling
        if (bSize * n1Size * gSize * CeilDiv(s1Size, s1BasicBlock) <= static_cast<int64_t>(aivNum) &&
            attenMaskExistFlag == 1 && !isSameAB) {
            bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpS2BasicBlock, s2Size), dSize);
        } else {
            bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size),
                          std::min(tmpS2BasicBlock * tilingData->coreParams.get_nRatio(), s2Size), dSize);
        }
        bmm1.SetOrgShape(s1Size, tmpS2BasicBlock * tilingData->coreParams.get_nRatio(), s1StrideSize, s2StrideSize);
        bmm1.SetBias(false);
        if (bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }

        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, [[maybe_unused]] int64_t tmpDBasicBlock,
                            [[maybe_unused]] int64_t batch, matmul_tiling::MatmulApiTiling &bmm2) override
    {
        bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(std::min(tmpS1BasicBlock, s1Size), dSize,
                      std::min(tmpS2BasicBlock * tilingData->coreParams.get_nRatio(), s2Size));
        bmm2.SetOrgShape(s1Size, s2StrideSize, std::min(tmpS2BasicBlock * tilingData->coreParams.get_nRatio(), s2Size),
                         s2StrideSize);
        bmm2.SetBias(false);
        if (!isSameAB && inputDtypeBytes != DATA_TYPE_FP32 && !hasRope && 
            s1BasicBlock <= S1_BASIC_BLOCK_L1CARRY_MAX && 
            dSize <= D_SIZE_L1CARRY_MAX && d2Size <= D2_SIZE_L1CARRY_MAX) {
            // bmm2L1Carry走MDL，不切M/N
            bmm2.SetFixSplit(AlignUp(std::min(tmpS1BasicBlock, s1Size), FRACTAL_NUM), AlignUp(d2Size, FRACTAL_NUM), -1);
        }
        if (isSameAB && (dSize == 192 || dSize == 128 || dSize == 88 || dSize == 80)) {
            tmpS1BasicBlock = 256; // sameAB Bmm2Carry MDL 模板需要baseM >= singleM
            bmm2.SetFixSplit(AlignUp(std::min(tmpS1BasicBlock, s1Size), FRACTAL_NUM), -1, -1);
        }
        if (bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
            return false;
        }
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        // not care about layout in tiling key, pass BSND(enum value is 0)
        if (isSameAB) {
            return GET_TPL_TILING_KEY(0, static_cast<uint8_t>(AxisEnum::S1), static_cast<uint8_t>(AxisEnum::NONE),
            static_cast<uint8_t>(AxisEnum::NONE), static_cast<uint8_t>(implMode), static_cast<uint8_t>(tilingKeyDType),
            static_cast<uint8_t>(tilingKeyLayout), static_cast<uint8_t>(tilingKeyBmm1Format),
            static_cast<uint8_t>(tilingKeyBmm2Source), static_cast<uint8_t>(SparseEnum::ANY),
            static_cast<uint8_t>(PerformanceOrientedEnum::BIG_DOUBLE_BUFFER), static_cast<uint8_t>(hasDropOut),
            static_cast<uint8_t>(hasAttenMask), static_cast<uint8_t>(hasPse), static_cast<uint8_t>(enableL1Reuse),
            static_cast<uint8_t>(hasRope), static_cast<uint8_t>(matmulPolicyType), 
            static_cast<uint8_t>(s1TemplateType), static_cast<uint8_t>(s2TemplateType), static_cast<uint8_t>(dTemplateType));
        }
        return GET_TPL_TILING_KEY(0, static_cast<uint8_t>(AxisEnum::S1), static_cast<uint8_t>(AxisEnum::S2),
            static_cast<uint8_t>(AxisEnum::NONE), static_cast<uint8_t>(implMode), static_cast<uint8_t>(tilingKeyDType),
            static_cast<uint8_t>(tilingKeyLayout), static_cast<uint8_t>(tilingKeyBmm1Format),
            static_cast<uint8_t>(tilingKeyBmm2Source), static_cast<uint8_t>(SparseEnum::ANY),
            static_cast<uint8_t>(PerformanceOrientedEnum::BIG_DOUBLE_BUFFER), static_cast<uint8_t>(hasDropOut),
            static_cast<uint8_t>(hasAttenMask), static_cast<uint8_t>(hasPse), static_cast<uint8_t>(enableL1Reuse),
            static_cast<uint8_t>(hasRope), static_cast<uint8_t>(matmulPolicyType), 
            static_cast<uint8_t>(s1TemplateType), static_cast<uint8_t>(s2TemplateType), static_cast<uint8_t>(dTemplateType));
    }

    bool SetPseAlibiParams() override
    {
        OP_LOGD(context_, "Set pseAlibiParams begin.");
        auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
        if (pseShape == nullptr || pseShape->GetStorageShape().GetDimNum() == 0) {
            OP_LOGD(context_, "SetPseAlibiParams pseShape == nullptr.");
            return true;
        }
        if (pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_TYPE) ||
            pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            OP_CHECK_IF(tilingData->inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::RIGHT_DOWN_CAUSAL_BAND),
                       OP_LOGE(opName, "INNER Pse does not support sparse mode 7."), return false);
            if (tilingData->inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::BAND_LEFT_UP_CAUSAL)) {
                for (int64_t i = 0L; i < bSize; ++i) {
                    if (i == 0) {
                        if (actualSeqLenData[0] - actualSeqLenKvData[0] + qStartIdx - kvStartIdx == 0) {
                            continue;
                        } else {
                            OP_LOGE(context_, "INNER Pse sparse mode 8 is only supported when actualSeqLenData[0] %ld + qStartIdx %ld - actualSeqLenKvData[0] %ld - kvStartIdx %ld == 0.",
                            actualSeqLenData[0], qStartIdx, actualSeqLenKvData[0], kvStartIdx);
                            return false;
                        }
                    }
                    if (actualSeqLenData[i] != actualSeqLenKvData[i]) {
                        OP_LOGE(context_, "INNER Pse sparse mode 8 is only supported when actualSeqQLen[%ld] %ld and actualSeqKvLen[%ld] %ld are equal.", i, actualSeqLenData[i], i, actualSeqLenKvData[i]);
                        return false;
                    }
                }
            }
            return true;
        }
        // 2: pre last axiss
        if (pseShape->GetStorageShape().GetDimNum() < 2) {  // 2 is min dim num of legal pse
            OP_LOGE(context_, "Invalid Pse DimNum(%zu), PseType(%ld).",
                      pseShape->GetStorageShape().GetDimNum(), pseType);
            return false;
        }
        auto pseS1Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 2);
        auto pseS2Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 1);

        PseEncodeType pseEncodeType = PSE_ENCODE_NONE;
        OP_LOGD(context_, "[%s] pseS1Size:%ld, pseS2Size:%ld.", templateName, pseS1Size, pseS2Size);
        if (pseS1Size == PSE_ALIBI_S_SIZE) {
            for (int64_t i = 0L; i < bSize; ++i) {
                if (actualSeqLenData[i] != actualSeqLenKvData[i]) {
                    OP_LOGE(context_, "Pse alibi only support when actualSeqQLen and actualSeqKvLen are equal.");
                    return false;
                }
            }
            OP_CHECK_IF(tilingData->inputParams.get_sparseType() != static_cast<uint8_t>(SparseEnum::CAUSAL) &&
                           tilingData->inputParams.get_sparseType() !=
                               static_cast<uint8_t>(SparseEnum::RIGHT_DOWN_CAUSAL),
                       OP_LOGE(opName, "Pse alibi only support causal sparse type."), return false);
            pseEncodeType = PSE_ENCODE_ALIBI_S2_FULL;
            OP_LOGD(context_, "[%s] PSE_ENCODE_ALIBI_S2_FULL.", templateName);
        }
        tilingData->inputParams.set_pseEncodeType(pseEncodeType);
        tilingData->inputParams.set_pseS1Size(pseS1Size);
        tilingData->inputParams.set_pseS2Size(pseS2Size);
        return true;
    }

    bool IsCapable() override
    {
        if (tilingKeyLayout != LayoutType::LAYOUT_TND) {
            return false;
        }
        // 此处走SameAB模板的准入条件均为经验值
        if ((bSize >= SPACE_NUM_4 && accumS1 >= SAMEAB_TND_BASIC_SIZE && accumS2 >= SAMEAB_TND_BASIC_SIZE &&
            s1Size >= S2_REUSE_SIZE_512 && s2Size >= S2_REUSE_SIZE_512) ||
            (s2Size > S2_REUSE_SIZE_5120 && s1Size > S2_REUSE_SIZE_5120)) {
            isSameAB = true;
        }

        // FP32是因为sameAB暂不支持FP32，n1Size == 1 && n2Size == 1是因为测到的这种情况的case性能劣化较大
        if (inputDtypeBytes == DATA_TYPE_FP32 || (n1Size == 1 && n2Size == 1) ||
            implMode == AA_INVALID_LINE_HIGH_PRECISION) {
            isSameAB = false;
        }
        return true;
    }
    void SetMultiCoreParams() override
    {
        auto &multiCoreParams = tilingData->multiCoreParams;
        auto &coreParams = tilingData->coreParams;
        accumS1BlockNum = 0;
        for (int64_t i = 0; i < bSize; i++) {
            OP_LOGD(context_, "[%s]actualSeqLenData data %ld is %ld.", templateName, i, actualSeqLenData[i]);
            OP_LOGD(context_, "[%s]actualSeqLenKvData data %ld is %ld.", templateName, i, actualSeqLenKvData[i]);
            accumS1BlockNum += CeilDivision(actualSeqLenData[i], s1BasicBlock);
        }
        int64_t totalSize = accumS1BlockNum * coreParams.get_n2OuterSize() * coreParams.get_gOuterSize();
        int64_t actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
        int64_t actualUsedAicNum = std::min(totalSize, static_cast<int64_t>(aicNum));
        int64_t actualUsedAiCoreNum = isSameAB ? actualUsedAicNum * 2 : actualUsedAivNum;
        int64_t actualSplitAiCoreNum = isSameAB ? actualUsedAicNum : actualUsedAivNum;
        multiCoreParams.set_coreNum(static_cast<int32_t>(actualUsedAiCoreNum));
        multiCoreParams.set_totalSize(totalSize);
        multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualSplitAiCoreNum));
        multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
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

    ge::graphStatus GetWorkspaceSize() override
    {
        auto &tensorSizeParams = tilingData->tensorSizeParams;
        auto &coreParams = tilingData->coreParams;
        // 走SameAB模板并满足条件S1BaseSize <= 256 && dSize == 192时使能L1自主管理，shareL1Size需设为0
	    if (isSameAB && coreParams.get_s1BaseSize() <= 256 && (dSize == 192 || dSize == 128 || dSize == 88 || dSize == 80)) {
            tilingData->bmm1TilingData.shareL1Size = 0;
            tilingData->bmm2TilingData.shareL1Size = 0; 
        }
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Byetes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;
        int64_t stage1Bytes = 0;
        // FP32场景，stage1需要再额外申请2倍的空间
        if (inputDtypeBytes == DATA_TYPE_FP32) {
            stage1Bytes = bmm1Byetes;
        }

        // 切D场景，默认使能NZND，stage1占用3倍的空间，stage2占用4倍空间
        int64_t totalSize = accumS1BlockNum * coreParams.get_n2OuterSize() * coreParams.get_gOuterSize();
        int64_t actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
        int64_t actualUsedAicNum = std::min(totalSize, static_cast<int64_t>(aicNum));
        int64_t actualSplitAiCoreNum = isSameAB ? actualUsedAicNum : actualUsedAivNum;
        int64_t pseAlignSize = 0;
        if (pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_TYPE) ||
            pseType == static_cast<int64_t>(PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            if (s2Size > s2sizeLimitMax) {
                pseAlibiBaseS2 = s2sizeLimitMax;
            } else {
                pseAlibiBaseS2 = alignedS2;
            }
            pseAlibiBaseS1 = s1BasicBlock;
            int64_t pseSize = pseAlibiBaseS1 * pseAlibiBaseS2 * 2;
            if (isSameAB) {
                pseAlignSize = (pseSize + NUM_512 - 1) / NUM_512 * NUM_512;
            }
        }
        workspaces[0] = static_cast<size_t>(
                        (bmm1Byetes * SPACE_NUM_3 + stage1Bytes * SPACE_NUM_2 +
                        SPACE_NUM_4 * coreParams.get_s1BaseSize() * alignedD * calcTypeSize + pseAlignSize) * actualSplitAiCoreNum) +
                        WORK_SPACE_RESERVE_SIZE;
        return ge::GRAPH_SUCCESS;
    }

    void SetSoftMaxTiling() override
    {
        auto softmaxShape = ge::Shape({s1BasicBlock / GetNRatio(), s2BasicBlock * GetNRatio()});
        AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                          tilingData->softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
    }

    bool CheckPretokenAndNexttoken(SparseEnum &sparseType)
    {
        if (sparseMode == static_cast<int64_t>(ALL_MASK)) {
            if (preTokens < s1Size - 1 || nextTokens < s2Size - 1) {
                OP_LOGW(context_,
                          "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                          "preTokens and nextTokens will be reset max int value.",
                          preTokens, nextTokens, sparseMode);
                preTokens = std::numeric_limits<int32_t>::max();
                nextTokens = std::numeric_limits<int32_t>::max();
            }
            sparseType = static_cast<SparseEnum>(static_cast<uint8_t>(sparseMode));
        } else if (sparseMode == static_cast<int64_t>(LEFT_UP_CAUSAL)) {
            preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
            nextTokens = 0;
            sparseType = SparseEnum::CAUSAL;
        } else if (sparseMode == static_cast<int64_t>(RIGHT_DOWN_CAUSAL)) {
            for (int64_t i = 0L; i < bSize; ++i) {
                if (actualSeqLenData[i] > actualSeqLenKvData[i]) {
                    OP_LOGE(context_, "Batch[%ld] s1[%ld] is larger than s2[%ld], exist invalid row.", i,
                              actualSeqLenData[i], actualSeqLenKvData[i]);
                    return false;
                }
            }
            preTokens = s2Size; // if sparse type is causal, template always need preTokens equal to s1Size
            nextTokens = 0;
            sparseType = SparseEnum::RIGHT_DOWN_CAUSAL;
        } else if (sparseMode == static_cast<int64_t>(BAND)) {
            if (preTokens < 0) {
                OP_LOGE(context_, "pre_tokens[%ld] config error, has invalid data block.", preTokens);
                return false;
            }
            if (nextTokens < 0 && preTokens + nextTokens < 0) {
                OP_LOGE(context_, "pre_tokens[%ld], next_tokens[%ld], invalid config.", preTokens, nextTokens);
                return false;
            }
            for (int64_t i = 0L; i < bSize; ++i) {
                if (actualSeqLenData[i] == 0 || actualSeqLenKvData[i] == 0) {
                    continue;
                }
                if (actualSeqLenData[i] - nextTokens > actualSeqLenKvData[i]) {
                    OP_LOGE(context_, "Batch[%ld], s1[%ld], s2[%ld], next_tokens[%ld], has invalid row.", i,
                              actualSeqLenData[i], actualSeqLenKvData[i], nextTokens);
                    return false;
                }
            }
            if (preTokens >= s2Size && nextTokens == 0) {
                preTokens = s2Size;
                nextTokens = 0;
                sparseType = SparseEnum::RIGHT_DOWN_CAUSAL;
                return true;
            }
            sparseType = SparseEnum::BAND_COMPRESS;
        } else if (sparseMode == static_cast<int64_t>(RIGHT_DOWN_CAUSAL_BAND)) {
            int64_t lastS2 = actualSeqLenKvData[bandIndex];
            if (preTokens < lastS2 || nextTokens > 0) {
                OP_LOGE(context_,
                          "RightDownCausal_Band mode: pre_tokens[%ld] is smaller than last valid s2[%ld]"
                          "or next_tokens[%ld] is larger than 0, wrong config.",
                          preTokens, lastS2, nextTokens);
                return false;
            }
            for (int64_t i = 0L; i < bSize; ++i) {
                if (actualSeqLenData[i] == 0 || actualSeqLenKvData[i] == 0) {
                    continue;
                }
                if (actualSeqLenData[i] > actualSeqLenKvData[i]) {
                    OP_LOGE(context_, "Batch[%ld] s1[%ld] is larger than s2[%ld].", i, actualSeqLenData[i],
                              actualSeqLenKvData[i]);
                    return false;
                }
                if ((i == bandIndex) && (actualSeqLenData[i] - nextTokens > actualSeqLenKvData[i])) {
                    OP_LOGE(context_, "Batch[%ld], s1[%ld], s2[%ld], next_tokens[%ld], has invalid row.", i,
                              actualSeqLenData[i], actualSeqLenKvData[i], nextTokens);
                    return false;
                }
            }
            sparseType = SparseEnum::RIGHT_DOWN_CAUSAL_BAND;
        } else if (sparseMode == static_cast<int64_t>(BAND_LEFT_UP_CAUSAL)) {
            if (actualSeqLenData[bandIndex] - nextTokens > actualSeqLenKvData[bandIndex]) {
                OP_LOGE(context_, "Batch[%ld], s1[%ld], s2[%ld], next_tokens[%ld], has invalid row.", bandIndex,
                          actualSeqLenData[0], actualSeqLenKvData[0], nextTokens);
                return false;
            }
            int64_t firstS2 = actualSeqLenKvData[bandIndex];
            if (preTokens < firstS2) {
                OP_LOGE(context_, "Band_LeftUpCausal mode: pre_tokens[%ld] is smaller than first valid s2[%ld].",
                          preTokens, firstS2);
                return false;
            }
            sparseType = SparseEnum::BAND_LEFT_UP_CAUSAL;
        }
        return true;
    }

    bool SparseNoMaskModeCheck(int64_t maxS1Value, int64_t maxS2Value, int64_t minS2Value,
                               SparseEnum &sparseType)
    {
        if (nextTokens < 0) {
            OP_LOGE(context_, "nextTokens[%ld] config error, there is no valid data block.", nextTokens);
            return false;
        }
        if (preTokens >= maxS1Value && nextTokens >= maxS2Value) {
            return true;
        }
        for (int64_t i = 0L; i < bSize; ++i) {
            if (actualSeqLenData[i] == 0 || actualSeqLenKvData[i] == 0) {
                continue;
            }
            if (actualSeqLenKvData[i] + preTokens < actualSeqLenData[i]) {
                OP_LOGE(context_, "Batch[%ld] s1[%ld] s2[%ld] has invalid row, check pre_tokens and next_tokens.", i,
                          actualSeqLenData[i], actualSeqLenKvData[i]);
                return false;
            }
        }
        if (preTokens >= 0) {
            s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), s1Size);
            s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
            sparseType = SparseEnum::BAND;
            isSparseValidSizeAligned = true;
            return true;
        }

        if (preTokens < 0) {
            int64_t absPreTokens = std::abs(preTokens);
            if (nextTokens >= absPreTokens) {
                s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), 0L);
                s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
                sparseType = SparseEnum::BAND;
                isSparseValidSizeAligned = true;
                return true;
            } else {
                OP_LOGE(context_,
                          "preTokens[%ld] and nextTokens[%ld] config error with S[%ld], has invalid data block.",
                          preTokens, nextTokens, minS2Value);
                return false;
            }
        }
        return true;
    }

    bool VarLenGetPrefixNList(SparseEnum &sparseType)
    {
        auto prefixN = context_->GetOptionalInputTensor(PREFIX_INPUT_INDEX);
        if (prefixN == nullptr) {
            OP_LOGE(context_, "[%s] prefixN is null pointer while sparse mode is prefix compress", templateName);
            return false;
        }

        auto &prefixShape = prefixN->GetShape().GetStorageShape();
        if (prefixShape.GetDimNum() != 1) {
            OP_LOGE(context_, "[%s] prefixN shape is invalid, DimNum should be 1, but it is %zu.", templateName,
                      prefixShape.GetDimNum());
            return false;
        }
        if (prefixShape.GetDim(0) != bSize) {
            OP_LOGE(context_,
                      "[%s] prefixN is invalid, it should be the same size as bSize[%ld], but it "
                      "is %ld.",
                      templateName, bSize, prefixShape.GetDim(0));
            return false;
        }

        /* Get Data from tensor. */
        prefixNData = prefixN->GetData<int64_t>();
        if (prefixNData == nullptr) {
            OP_LOGE(context_, "[%s] prefixN data is null pointer", templateName);
            return false;
        }

        for (int64_t i = 0; i < bSize; ++i) {
            if (actualSeqLenData[i] > actualSeqLenKvData[i]) {
                if (prefixNData[i] < 0 || prefixNData[i] > actualSeqLenKvData[i]) {
                    OP_LOGE(context_, "[%s] batch[%ld] prefixN=%ld is invalid, should be in range of [0, %ld]",
                              templateName, i, prefixNData[i], actualSeqLenKvData[i]);
                    return false;
                }
                if (prefixNData[i] == 0) {
                    implMode = AA_INVALID_LINE_HIGH_PRECISION;
                    OP_LOGD(context_, "Enable invalid line impl mode.");
                }
            } else {
                if (prefixNData[i] < actualSeqLenKvData[i] - actualSeqLenData[i] ||
                    prefixNData[i] > actualSeqLenKvData[i]) {
                    OP_LOGE(context_, "[%s] batch[%ld] prefixN=%ld is invalid, should be in range of [%ld, %ld]",
                              templateName, i, prefixNData[i], actualSeqLenKvData[i] - actualSeqLenData[i],
                              actualSeqLenKvData[i]);
                    return false;
                }
            }
        }

        sparseType = SparseEnum::PREFIX;
        return true;
    }

    bool VarLenSparseModeProcess(SparseEnum &sparseType)
    {
        if (sparseMode == static_cast<int64_t>(PREFIX) || sparseMode > static_cast<int64_t>(BAND_LEFT_UP_CAUSAL)) {
            OP_LOGE(context_, "Var len not support sparse mode %ld.", sparseMode);
            return false;
        }

        if (!CheckPretokenAndNexttoken(sparseType)) {
            OP_LOGE(context_, "Check pre_tokens and next_tokens failed.");
            return false;
        }

        if (sparseMode == static_cast<int64_t>(PREFIX_COMPRESS)) {
            if (!VarLenGetPrefixNList(sparseType)) {
                return false;
            }
        }
        return true;
    }

    bool GetSparseInfo(SparseEnum &sparseType) override
    {
        OP_LOGD(context_,
                  "check sparse feature: preTokens[%ld], nextTokens[%ld], s1[%ld], s2[%ld], attenMaskExist[%d].",
                  preTokens, nextTokens, s1Size, s2Size, attenMaskExistFlag);
        if (attenMaskExistFlag != 1 || tilingKeyLayout != LayoutType::LAYOUT_TND) {
            return true;
        }

        // if sparseMode is NoMask, preTokens and nextTokens start from top left vertex;
        // if sparseMode is Band, preTokens and nextTokens start from bottom right vertex.
        if (sparseMode == static_cast<int64_t>(NO_MASK)) {
            if (preTokens >= s1Size && nextTokens == 0) {
                sparseType = SparseEnum::CAUSAL;
                preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
            } else {
                if (preTokens >= s1Size && nextTokens >= s2Size) {
                    return true;
                }
                int64_t minS2Value = *std::min_element(actualSeqLenKvData.begin(), actualSeqLenKvData.begin() + bSize);
                if (!SparseNoMaskModeCheck(s1Size, s2Size, minS2Value, sparseType)) {
                    return false;
                }
            }
        } else {
            if (!VarLenSparseModeProcess(sparseType)) {
                return false;
            }
        }
        return true;
    }

    int64_t GetS2RealSize(uint8_t sparseType, int32_t bOutIdx, int64_t s1OutIdx)
    {
        int64_t s2RealSize = s2Size;
        if (sparseType == static_cast<uint8_t>(SparseEnum::CAUSAL)) {
            s2RealSize = s1BasicBlock * (s1OutIdx + 1);
        } else if (sparseType == static_cast<uint8_t>(SparseEnum::PREFIX)) {
            s2RealSize = std::max(s1BasicBlock * (s1OutIdx + 1) - actualSeqLenData[bOutIdx] +
                actualSeqLenKvData[bOutIdx], prefixNData[bOutIdx]);
        } else if (sparseType == static_cast<uint8_t>(SparseEnum::BAND_COMPRESS)) {
            int64_t s2StartIdx = std::max(s1OutIdx * s1BasicBlock - actualSeqLenData[bOutIdx] +
                std::max(actualSeqLenKvData[bOutIdx] - preTokens, 0L), 0L);
            int64_t s2EndIdx = std::min((s1OutIdx + 1) * s1BasicBlock + actualSeqLenKvData[bOutIdx] -
                std::max(actualSeqLenData[bOutIdx] - nextTokens, 0L), actualSeqLenKvData[bOutIdx]);
            if (s2EndIdx - s2StartIdx <= 0) {
                s2StartIdx = 0;
                s2EndIdx = actualSeqLenKvData[bOutIdx];
            }
            s2RealSize = s2EndIdx - s2StartIdx;
        }

        return std::min(s2RealSize, actualSeqLenKvData[bOutIdx]);
    }

    bool InitSparseValidArray(std::vector<int64_t> &sparseValidArray)
    {
        uint8_t sparseType = tilingData->inputParams.get_sparseType();
        auto &coreParams = tilingData->coreParams;
        for (int32_t i = 0; i < bSize; i++) {
            int64_t n2G = coreParams.get_n2OuterSize() * coreParams.get_gOuterSize();
            int64_t s1BlockNum = CeilDivision(actualSeqLenData[i], s1BasicBlock);
            // 每个s1方向上切分块的计算量
            for (int64_t k = 0; k < n2G; ++k) {
                for (int64_t j = 0; j < s1BlockNum; ++j) {
                    // 此处暂时设置为1, 由于实测尾块1和128性能差距不大，理论上应该如下所示
                    // 理论值: s1RealSize为std::min(s1BasicBlock, (actualSeqLenData[i] - s1BasicBlock * j))
                    int64_t s1RealSize = 1;
                    int64_t s2RealSize = GetS2RealSize(sparseType, i, j);
                    // 新增一个系数, 解决理论和实际的差异
                    int64_t s2RemainSize = s2RealSize % s2sizeLimitMax;
                    s2RealSize = (s2RealSize / s2sizeLimitMax) * s2sizeLimitMax;
                    s2RealSize += ((s2RemainSize > 0) ? COF[CeilDivision(s2RemainSize, 128L) - 1] : 0);
                    sparseValidArray.emplace_back(s1RealSize * s2RealSize);
                }
            }
        }

        return true;
    }

    bool BalanceLoad(const std::vector<int64_t> &sparseValidArray, MultiCoreParams &multiCoreParams,
                     std::vector<int64_t> &localValue, std::vector<int64_t> &sparseStartIdx)
    {
        // to avoid buffer overflow, or maybe sometimes we want to only verify single core
        int64_t validAiCoreNum = isSameAB ? std::min(static_cast<int64_t>(multiCoreParams.get_coreNum() / 2), MAX_AIC_NUM)
                                          : std::min(static_cast<int64_t>(multiCoreParams.get_coreNum()), MAX_AIV_NUM);
        int64_t totalSize = multiCoreParams.get_totalSize();
        int64_t maxVal = *std::max_element(localValue.begin(), localValue.end());
        int64_t tmpMaxVal = maxVal;

        // 从前往后遍历
        for (int64_t idx = 1; idx < validAiCoreNum; ++idx) {
            int64_t start = sparseStartIdx[idx];
            if (start < totalSize && start > 0 && ((localValue[idx - 1] + sparseValidArray[start]) < maxVal)) {
                localValue[idx - 1] += sparseValidArray[start];
                localValue[idx] -= sparseValidArray[start];
                sparseStartIdx[idx] += 1;
            } else if (start == totalSize) {
                break;
            }
        }
        tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

        // 从后往前遍历
        for (int64_t idx = validAiCoreNum - 1; idx > 0; --idx) {
            int64_t start = sparseStartIdx[idx];
            if (start == totalSize) {
                if (sparseStartIdx[idx - 1] == totalSize) {
                    continue;
                }
                localValue[idx - 1] -= sparseValidArray[start - 1];
                localValue[idx] = sparseValidArray[start - 1];
                sparseStartIdx[idx] -= 1;
            } else if (start > 0) {
                if ((localValue[idx] + sparseValidArray[start - 1]) >= tmpMaxVal) {
                    continue;
                }
                localValue[idx - 1] -= sparseValidArray[start - 1];
                localValue[idx] += sparseValidArray[start - 1];
                sparseStartIdx[idx] -= 1;
            } else {
                break;
            }
        }
        tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

        return (tmpMaxVal >= maxVal) ? false : true;
    }

    inline bool InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t validAivNum, int64_t totalSize,
                              const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue)
    {
        for (int64_t idx = 0; idx < validAivNum; ++idx) {
            int64_t start = sparseStartIdx[idx];
            int64_t end = ((idx + 1) < validAivNum) ? sparseStartIdx[idx + 1] : totalSize;
            if (start < totalSize) {
                localValue[idx] =
                    std::accumulate(sparseValidArray.begin() + start, sparseValidArray.begin() + end, 0LL);
            } else {
                break;
            }
        }
        return true;
    }

    bool SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, MultiCoreParams &multiCoreParams) override
    {
        // to avoid buffer overflow, or maybe sometimes we want to only verify single core
        int64_t validAiCoreNum = isSameAB ? std::min(static_cast<int64_t>(multiCoreParams.get_coreNum() / 2), MAX_AIC_NUM)
                                          : std::min(static_cast<int64_t>(multiCoreParams.get_coreNum()), MAX_AIV_NUM);
        int64_t totalSize = multiCoreParams.get_totalSize(); // BN2GS1.o
        int64_t *sparseStartIdx = multiCoreParams.get_sparseStartIdx();
        int64_t maxAiCoreNum = isSameAB ? MAX_AIC_NUM : MAX_AIV_NUM;
        OP_CHECK_IF(totalSize <= 0, OP_LOGE(opName, "totalSize should be larger than 0."), return false);

        // initLoad: 使用均分策略, 保证后续不会比均分差
        int64_t splitFactorSize = multiCoreParams.get_splitFactorSize();
        std::vector<int64_t> localSparseStartIdx(maxAiCoreNum, totalSize);
        for (int64_t idx = 0; idx < maxAiCoreNum; ++idx) {
            localSparseStartIdx[idx] = std::min((idx * splitFactorSize), totalSize);
        }
        std::vector<int64_t> localValue(validAiCoreNum, 0);
        InitLoadValue(sparseValidArray, validAiCoreNum, totalSize, localSparseStartIdx, localValue);

        // 负载均衡粗调
        std::vector<int64_t> tmpLocalValue(validAiCoreNum, 0);
        std::vector<int64_t> tmpsparseStartIdx(maxAiCoreNum, totalSize);
        int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL);
        int64_t avgVal = CeilDivision(sparseArraySum, validAiCoreNum);

        tmpsparseStartIdx[0] = 0;
        for (int64_t idx = 1; idx < maxAiCoreNum; ++idx) {
            int64_t start = tmpsparseStartIdx[idx - 1];
            int64_t singleLoadValue = 0;
            tmpsparseStartIdx[idx] = start;
            while (singleLoadValue < avgVal && tmpsparseStartIdx[idx] < totalSize) {
                singleLoadValue += sparseValidArray[tmpsparseStartIdx[idx]];
                tmpsparseStartIdx[idx] += 1;
            }

            if ((start + 1) < tmpsparseStartIdx[idx]) {
                int64_t redoSingleLoadValue = singleLoadValue - sparseValidArray[tmpsparseStartIdx[idx] - 1];
                bool loadValueFlag = (singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue);
                tmpsparseStartIdx[idx] = loadValueFlag ? (tmpsparseStartIdx[idx] - 1) : (tmpsparseStartIdx[idx]);
                singleLoadValue = loadValueFlag ? redoSingleLoadValue : singleLoadValue;
                sparseArraySum -= singleLoadValue;
                avgVal = CeilDivision(sparseArraySum, (validAiCoreNum - idx));
            }
        }

        InitLoadValue(sparseValidArray, validAiCoreNum, totalSize, tmpsparseStartIdx, tmpLocalValue);

        // 负载均衡精调
        while (BalanceLoad(sparseValidArray, multiCoreParams, tmpLocalValue, tmpsparseStartIdx)) {
            // 根据负载均衡是否能得到更好预测结果决定是否结束循环
        }

        // exchange initLoad and 负载均衡
        if ((*std::max_element(localValue.begin(), localValue.end())) >
            (*std::max_element(tmpLocalValue.begin(), tmpLocalValue.end()))) {
            localSparseStartIdx.swap(tmpsparseStartIdx);
            localValue.swap(tmpLocalValue);
        }
        for (int64_t idx = 0; idx < maxAiCoreNum; ++idx) {
            sparseStartIdx[idx] = localSparseStartIdx[idx];
        }

        return true;
    }

    void SetSparseParams() override
    {
        auto &coreParams = tilingData->coreParams;
        auto &multiCoreParams = tilingData->multiCoreParams;
        std::vector<int64_t> sparseValidArray;
        sparseValidArray.reserve(multiCoreParams.get_totalSize());
        InitSparseValidArray(sparseValidArray);
        SetSparseStartIdx(sparseValidArray, multiCoreParams);

        coreParams.set_s1SparseValidSize(s1SparseValidSize);
        coreParams.set_s2SparseValidSize(s2SparseValidSize);
    }

    ge::graphStatus PostTiling() override
    {
        auto ret = FlashAttentionScoreTilingBase::PostTiling();

        auto &inputParams = tilingData->inputParams;
        // TND EOD+Padding 场景下 kernel侧使用t1Size复用s1Size
        inputParams.set_s1Size(realT1Size);
        if (!isSameAB && inputDtypeBytes != DATA_TYPE_FP32 && !hasRope && 
            s1BasicBlock <= S1_BASIC_BLOCK_L1CARRY_MAX && 
            dSize <= D_SIZE_L1CARRY_MAX && d2Size <= D2_SIZE_L1CARRY_MAX) {
            this->needL1Carry = true;
            tilingData->bmm1TilingData.shareL1Size = 0;
            tilingData->bmm2TilingData.shareL1Size = 0;
        }
        inputParams.set_needL1Carry(this->needL1Carry);
        return ret;
    }
};

class FlashAttentionScoreTilingDropMask : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionScoreTilingDropMask(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context) {}
    ~FlashAttentionScoreTilingDropMask() override = default;

protected:
    ge::graphStatus DoOpTiling() override
    {
        auto &inputParams = tilingData->inputParams;
        auto &dropmaskParams = tilingData->dropmaskParams;
        if (inputParams.get_needDropMaskOp() == 0) {
            dropmaskParams.reset();
            return ge::GRAPH_PARAM_INVALID;
        }

        int64_t shapeTotalSize = inputParams.get_bSize() * inputParams.get_n2Size() * inputParams.get_gSize() *
                                 inputParams.get_s1Size() * inputParams.get_s2Size();
        auto layoutType = tilingData->inputParams.get_layoutType();
        if (layoutType == LAYOUT_TND) {
            for (int64_t i = 0; i < bSize; i++) {
                dropTotalSize += (actualSeqLenData[i] * actualSeqLenKvData[i]);
            }
            shapeTotalSize = inputParams.get_n2Size() * inputParams.get_gSize() * dropTotalSize;
            OP_LOGD(context_, "shapeTotalSize %ld dropTotalSize %ld.", shapeTotalSize, dropTotalSize);
        }
        // 保证每核计算数据量256倍数，2048表示bit位，256 * 8
        const int64_t ubCalFactor = 2048;
        int64_t shapeSplitCoreSize = CeilDivision(shapeTotalSize, ubCalFactor);
        int64_t shapeSingleCoreSize = CeilDivision(shapeSplitCoreSize, static_cast<int64_t>(aivNum));

        // ub能计算的最大元素数, 向下对齐
        // 单次ub计算量为x个元素,空间占用：x/8 * 1 [1个 uint8]+ 2x * 2[2个fp16,select的src和res] + x * 1 [1个uint8]共6份
        int64_t baseUbCalSize = AlignDown(CeilDivision(static_cast<int64_t>(aicoreParams_.ubSize), 6L), ubCalFactor);
        baseUbCalSize = std::min(baseUbCalSize, shapeSingleCoreSize * ubCalFactor);
        // ub 的外层循环次数
        int64_t multiCoreFactorSize = CeilDivision(shapeSingleCoreSize * ubCalFactor, baseUbCalSize);

        dropmaskParams.set_shapeTotalSize(shapeTotalSize);
        dropmaskParams.set_multiCoreFactorSize(static_cast<int32_t>(multiCoreFactorSize));
        dropmaskParams.set_multiCoreTotalSize(CeilDivision(shapeSplitCoreSize * ubCalFactor, baseUbCalSize));
        dropmaskParams.set_baseUbCalSize(static_cast<int32_t>(baseUbCalSize));
        return ge::GRAPH_PARAM_INVALID;
    }

    bool CalcUBSize([[maybe_unused]] int64_t tmpS1BasicBlock, [[maybe_unused]] int64_t tmpS2BasicBlock) override
    {
        return true;
    }

    void GetBufferNum([[maybe_unused]] BufferNum &bufferNum) const override
    {
    }

    bool SetBmm2TilingInput([[maybe_unused]] int64_t tmpS1BasicBlock, [[maybe_unused]] int64_t tmpS2BasicBlock,
                            [[maybe_unused]] int64_t tmpDBasicBlock, [[maybe_unused]] int64_t batch,
                            [[maybe_unused]] matmul_tiling::MatmulApiTiling &bmm2) override
    {
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        return GET_TPL_TILING_KEY(0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0);
    }
};

// NOTE manually initialize tiling data in hostapi scenario in highest priority template
REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttentionScore, FlashAttentionScoreTilingDropMask, std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}), 90);
REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttentionScore, FlashAttentionVarLenScoreTiling, std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}), 94);
REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttentionScore, FlashAttentionScoreTilingS1s2Bn2gs1SameAB, std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}), 95);
REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttentionScore, FlashAttentionScoreTilingS1s2Bn2gs1, std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}), 96);
REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttentionScore, FlashAttentionScoreTilingS1Bn2gs1, std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}), 97);
REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttentionScore, FlashAttentionScoreTilingB, std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}), 98);
} // namespace FA
} // namespace optiling
