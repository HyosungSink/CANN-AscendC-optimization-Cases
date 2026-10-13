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
 * \file nsa_selected_attention_tiling_general.cpp
 * \brief
 */

#include "log/log.h"
#include "../common/include/err/ops_err.h"
#include <numeric>
#include "../common/include/tiling_base/tiling_type.h"
#include "../common/include/tiling_base/tiling_templates_registry.h"
#include "nsa_selected_attention_tiling.h"
#include "../common/include/tiling_base/tiling_base.h"
using namespace Ops::Transformer::OpTiling;

namespace optiling {
namespace NSA {
const int64_t FRACTAL_NUM = 16L;
const int64_t ONE = 1;
const size_t ATTENTION_MASK_INPUT_INDEX = 4UL;
const size_t ACTUAL_SEQ_LENGTH_INPUT_INDEX = 5UL;
const size_t ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX = 6UL;
const size_t ATTEN_OUT_INDEX = 2UL;
const size_t ATTENTION_MASK_DIM_NUM_2 = 2UL;
const size_t SOFTMAX_MAX_BUF_SIZE = 128 * 1024;
constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t SELECTED_BLOCK_COUNT_MAX = 128;
const int64_t MAX_VAR_LEN_SEQ_LEN = 4096L;
const int64_t HEAD_DIM_MAX_VALUE = 768L;

enum class LayoutType : uint8_t {
    None = 0,
    LAYOUT_BSH = 1,
    LAYOUT_BSND = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,  //NSA需要用到
};

enum class SparseMode : uint8_t {
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

enum class ImplMode : uint8_t {
    AA_HIGH_PRECISION = 0,
    AA_HIGH_PERFORMANCE = 1,
    AA_INVALID_LINE_HIGH_PRECISION = 2
};


template <typename T> 
static T AlignUp(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T> 
static T CeilDivision(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

class NsaSelectedAttentionTiling : public TilingBaseClass {
public:
    explicit NsaSelectedAttentionTiling(gert::TilingContext *context) : TilingBaseClass(context)
    {
        templateName = "NsaSelectedAttentionTiling";
    }
    ~NsaSelectedAttentionTiling() override = default;

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    void Reset(gert::TilingContext *context) override;

    void GetActualSeqLenData(int64_t inputIdx, std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &res, int64_t &actualLen) const;
    ge::graphStatus CheckContext();
    bool AnalyzeDtype();
    bool AnalyzeAttrs();
    bool AnalyzeLayout();
    bool Analyze3DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &valueShape);
    bool AnalyzeOptionalInput();
    bool SetBmm1TilingInput(matmul_tiling::MatmulApiTiling &bmm1);
    bool SetBmm2TilingInput(matmul_tiling::MatmulApiTiling &bmm2);
    bool SetMatMulTiling();
    void SetMultiCoreParams();
    void SetSoftMaxTiling();

    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t l2CacheSize = 0;

    matmul_tiling::DataType bmmDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

    ge::DataType inputDtype;
    int64_t inputDtypeBytes;
    int64_t calcTypeSize;

    bool isHighPercision; // fp16 high percision mode
    ImplMode implMode;

    int64_t bSize;
    int64_t gSize;
    int64_t t1Size;
    int64_t dSize;
    int64_t d2Size;
    int64_t n1Size;
    int64_t n2Size;
    int64_t s1Size;
    int64_t s2Size;
    int64_t s1StrideSize; // query Shape S inner axes, for bmm1
    int64_t s2StrideSize; // key Shape S inner axes, for bmm1
    int64_t vs2StrideSize;
    int64_t selectedBlockCount;
    int64_t selectedBlockSize;
    int64_t selectedLength;
    int64_t sparseMode;
    int64_t maxS1Val;
    int64_t minS1Val;
    int64_t accumS1;
    int64_t maxS2Val;
    int64_t minS2Val;
    int64_t accumS2;
    std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> actualSeqLenData;
    std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> actualSeqLenKvData;
    float scaleValue;

    int64_t alignedS1;
    int64_t alignedS2;
    int64_t alignedD;
    int64_t alignedD2;

    const char *templateName = "base";
    const char *opName;
    const char *inputLayout;

    int32_t mm1BaseM = 16L;
    int32_t mm1BaseN = 64L;
    int32_t mm1BaseK = 192L;
    int32_t mm2BaseM = 16L;
    int32_t mm2BaseN = 128L;
    int32_t mm2BaseK = 128L;
    NsaSelectedAttentionTilingData tilingData;
};

ge::graphStatus NsaSelectedAttentionTiling::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<NsaSelectedAttentionCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "compileInfoPtr is null."),
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
    OP_LOGI(context_,
        "get platform from compileInfo.aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu).",
            aivNum, aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0cSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionTiling::CheckContext()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto n1SizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto sparseModePtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto inputLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    auto selectedBlockSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto selectedBlockCountPtr = attrs->GetAttrPointer<int64_t>(idx++);
    size_t *workspaces = context_->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleValuePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, n1SizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sparseModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputLayoutPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, selectedBlockSizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, selectedBlockCountPtr);
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
    OP_CHECK_IF(
        context_->GetRawTilingData()->GetCapacity() < tilingData.GetDataSize(), OPS_REPORT_VECTOR_INNER_ERR(opName, "context tiling data capacity %zu < actual tiling data size %zu.",
                                        context_->GetRawTilingData()->GetCapacity(), tilingData.GetDataSize()),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionTiling::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOGD(opName, "TilingContext: " "%s.", GetTilingContextDebugStr().c_str());
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid context."),
            return ge::GRAPH_FAILED);

    OP_CHECK_IF(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout() || !AnalyzeOptionalInput(),
            OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to analyze context info."), return ge::GRAPH_FAILED);

    alignedS1 = AlignUp(s1Size, FRACTAL_NUM);  // 64*1024
    alignedS2 = AlignUp(s2Size, FRACTAL_NUM);   // 64*1024
    alignedD = AlignUp(dSize, FRACTAL_NUM); //192
    alignedD2 = AlignUp(d2Size, FRACTAL_NUM); // 128

    OP_CHECK_IF(alignedD <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid alignedD" " %ld.", alignedD),
        return ge::GRAPH_FAILED);

    auto &inputParams = tilingData.inputParams;
    inputParams.set_bSize(bSize);
    inputParams.set_n2Size(n2Size);
    inputParams.set_gSize(gSize);
    inputParams.set_s1Size(s1Size);
    inputParams.set_s2Size(s2Size);
    inputParams.set_dSize(dSize);
    inputParams.set_d2Size(d2Size);
    inputParams.set_scaleValue(scaleValue);
    inputParams.set_alignedS2(alignedS2);
    inputParams.set_selectedBlockSize(selectedBlockSize);
    inputParams.set_selectedBlockCount(selectedBlockCount);
    OP_LOGD(context_, "input params: bn2gs1s2d[%ld, %ld, %ld, %ld, %ld, %ld], scaleValue[%f]."
            , bSize, n2Size, gSize, s1Size, s2Size,
            dSize, scaleValue);

    return ge::GRAPH_SUCCESS;
}

void NsaSelectedAttentionTiling::Reset([[maybe_unused]] gert::TilingContext *context)
{
    tilingData.SetDataPtr(context_->GetRawTilingData()->GetData());
}

bool NsaSelectedAttentionTiling::AnalyzeDtype()
{
    inputDtype = context_->GetInputDesc(0)->GetDataType();
    inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
    switch (inputDtype) {
        case ge::DT_FLOAT16:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT16;
            bmm1OutDtype = isHighPercision ? matmul_tiling::DataType::DT_FLOAT : matmul_tiling::DataType::DT_FLOAT16;
            calcTypeSize = isHighPercision ? ge::GetSizeByDataType(ge::DT_FLOAT) : ge::GetSizeByDataType(inputDtype);
            break;
        case ge::DT_BF16:
            bmmDtype = matmul_tiling::DataType::DT_BF16;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
            break;
        default:
            OPS_REPORT_VECTOR_INNER_ERR(opName, "not support input dtype: %s for now",
                                        ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str());
            return false;
    }

    bmm2OutDtype = bmm1OutDtype;
    OP_LOGD(context_, "Get high precision flag: " "%d.", isHighPercision);
    return true;
}

bool NsaSelectedAttentionTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto n1SizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto sparseModePtr = attrs->GetAttrPointer<int64_t>(idx++);
    inputLayout = attrs->GetAttrPointer<char>(idx++);
    auto selectedBlockSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto selectedBlockCountPtr = attrs->GetAttrPointer<int64_t>(idx++);

    n1Size = *n1SizePtr;
    scaleValue = *scaleValuePtr;
    sparseMode = *sparseModePtr;
    selectedBlockSize = *selectedBlockSizePtr;
    OP_CHECK_IF(selectedBlockSize > 128,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "selectedBlockSize [%ld] should <= 128!", selectedBlockSize),
               return false);
    OP_CHECK_IF(selectedBlockSize % FRACTAL_NUM != 0,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "selectedBlockSize [%ld] should be multiple of 16!",
                                           selectedBlockSize),
               return false);
    selectedBlockCount = *selectedBlockCountPtr;
    OP_CHECK_IF(selectedBlockCount > SELECTED_BLOCK_COUNT_MAX,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "selectedBlockCount [%ld] should be <= 128!", selectedBlockCount),
               return false);
    selectedLength = selectedBlockSize * selectedBlockCount;
    OP_CHECK_IF(n1Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "Head num is zero."), return false);
    implMode = ImplMode::AA_HIGH_PRECISION;
    isHighPercision = true; // use default value

    // 当前场景只支持下三角 nexttokens和pretokens忽略
    bool supportedSparseMode = (sparseMode == static_cast<int64_t>(SparseMode::LEFT_UP_CAUSAL)) ||
                                 (sparseMode == static_cast<int64_t>(SparseMode::NO_MASK));
    OP_CHECK_IF(
        !supportedSparseMode, OPS_REPORT_VECTOR_INNER_ERR(opName, "Sparse Mode is just support LEFT UP CAUSAL and NO MASK."),
                return false);
    
    OP_LOGD(context_, 
        "attrs: scale_value[%f] input_layout[%s] inner_precise[%d] sparse_mode[%ld].",
            scaleValue, inputLayout, static_cast<int32_t>(implMode), sparseMode);
    return true;
}

bool NsaSelectedAttentionTiling::AnalyzeLayout()
{
    auto &queryShape = context_->GetInputShape(0)->GetStorageShape();
    auto &keyShape = context_->GetInputShape(1)->GetStorageShape();
    auto &valueShape = context_->GetInputShape(2)->GetStorageShape();
    auto &topkIndicesShape = context_->GetInputShape(3)->GetStorageShape();

    size_t layoutLen = strlen(inputLayout);

    OP_LOGD(context_, "Get input_layout " "[%s].", inputLayout);
    OP_CHECK_IF(topkIndicesShape.GetDim(2) != selectedBlockCount,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "topkIndices tail dim[%s] is not equal selected_block_count.", inputLayout), return false);
    OP_CHECK_IF(queryShape.GetDimNum() != layoutLen || keyShape.GetDimNum() != layoutLen || topkIndicesShape.GetDimNum() != layoutLen,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "Invalid layout[%s].", inputLayout), return false);
    OP_CHECK_IF(!Analyze3DimLayout(queryShape, keyShape, valueShape),
            OPS_REPORT_VECTOR_INNER_ERR(opName, "Get unsupported layout: %s", inputLayout), return false);
    OP_CHECK_IF(gSize == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "gSize is zero."), return false);
    OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "n2Size is zero."), return false);
    OP_CHECK_IF(dSize > HEAD_DIM_MAX_VALUE || dSize <= 0L,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "dSize is not in range:(0, 512]."), return false);
    OP_CHECK_IF(n1Size % n2Size != 0,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "n1Size [%ld] should be a multiple of n2Size [%ld].", n1Size, n2Size),
            return false);
    return true;
}

void NsaSelectedAttentionTiling::GetActualSeqLenData(int64_t inputIdx, std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &res,
                                                        int64_t &actualLen) const
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
    actualLen++;
    for (int64_t i = 1; i < actualSeqLenShape.GetDim(0); ++i) {
        auto qLen = value[i] - value[i - 1];
        res[i] = qLen < 0 ? 0 : qLen;
        actualLen++;
    }
}

bool NsaSelectedAttentionTiling::Analyze3DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &valueShape)
{
    int64_t h1 = 0;
    int64_t h2 = 0;
    int64_t actualSeqQLen = 0;
    int64_t actualSeqKVLen = 0;
    t1Size = queryShape.GetDim(0);
    int64_t t2Size = keyShape.GetDim(0);
    std::fill(actualSeqLenData.begin(), actualSeqLenData.end(), 0);
    std::fill(actualSeqLenKvData.begin(), actualSeqLenKvData.end(), 0);
    GetActualSeqLenData(ACTUAL_SEQ_LENGTH_INPUT_INDEX, actualSeqLenData, actualSeqQLen);
    GetActualSeqLenData(ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX, actualSeqLenKvData, actualSeqKVLen);
    OP_CHECK_IF(actualSeqQLen != actualSeqKVLen,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "VarLen scene, q is not equal kv."), return false);
    bSize = actualSeqQLen;
    OP_CHECK_IF(bSize > 1024,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "batch size [%ld] should <= 1024", bSize),
               return false);
    accumS1 = std::accumulate(actualSeqLenData.begin(), actualSeqLenData.end(), 0LL);
    accumS2 = std::accumulate(actualSeqLenKvData.begin(), actualSeqLenKvData.end(), 0LL);
    OP_CHECK_IF(
        t1Size < accumS1 || t2Size < accumS2,
        OPS_REPORT_VECTOR_INNER_ERR(
            opName,
            "Query T(%ld) and key T(%ld) need larger than respectively sum of seqLen(%ld) and sekvLen(%ld).",
            t1Size, t2Size, accumS1, accumS2),
        return false);
    maxS1Val = *std::max_element(actualSeqLenData.begin(), actualSeqLenData.end());
    maxS2Val = *std::max_element(actualSeqLenKvData.begin(), actualSeqLenKvData.end());
    s1Size = maxS1Val;
    s2Size = maxS2Val;
    
    n2Size = keyShape.GetDim(1);
    OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
    gSize =  queryShape.GetDim(1) / n2Size;
    dSize = queryShape.GetDim(2);
    d2Size = valueShape.GetDim(2);
    h1 = n1Size * dSize;
    h2 = n2Size * dSize;
    s1StrideSize = gSize * n2Size * dSize;
    s2StrideSize = n2Size * dSize;
    vs2StrideSize = n2Size * d2Size;
    tilingData.inputParams.set_layoutType(static_cast<uint8_t>(LayoutType::LAYOUT_TND));
    OP_CHECK_IF(h1 == 0 || h2 == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "H is zero."), return false);
    OP_CHECK_IF(h1 % n1Size != 0,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "h1 [%ld] should be a multiple of n1Size [%ld].", h1, n1Size),
            return false);
    return true;
}

bool NsaSelectedAttentionTiling::AnalyzeOptionalInput()
{
    auto attenMaskInput = context_->GetOptionalInputDesc(ATTENTION_MASK_INPUT_INDEX);
    auto attenMaskShape = context_->GetOptionalInputShape(ATTENTION_MASK_INPUT_INDEX);
    if (attenMaskInput != nullptr && attenMaskShape != nullptr && attenMaskShape->GetStorageShape().GetDimNum() != 0) {
        auto attenMaskType = attenMaskInput->GetDataType();
        OP_CHECK_IF(attenMaskType != ge::DT_BOOL && attenMaskType != ge::DT_UINT8,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid attenMask dtype[%s], only support bool or uint8.",
                                            ge::TypeUtils::DataTypeToSerialString(attenMaskType).c_str()),
                return false);

        tilingData.inputParams.set_attenMaskDataType(1);
        auto &attenMaskStorageShape = attenMaskShape->GetStorageShape();
        size_t attenMaskDimNum = attenMaskStorageShape.GetDimNum();
        if (attenMaskDimNum != ATTENTION_MASK_DIM_NUM_2) {
            OP_LOGE(context_, "atten mask dim should be 2 but got %zu", attenMaskDimNum);
            return false;
        }

        tilingData.inputParams.set_attenMaskS2Size(attenMaskStorageShape.GetDim(attenMaskDimNum - 1));
    } else {
        sparseMode = static_cast<int64_t>(SparseMode::NO_MASK);
        OP_LOGD(context_, "AttenMask is nullptr, set sparseMode: %ld.", sparseMode);
    }
    return true;
}

ge::graphStatus NsaSelectedAttentionTiling::DoOpTiling()
{
    auto &inputParams = tilingData.inputParams;
    inputParams.set_implMode(static_cast<uint8_t>(implMode));
    SetMultiCoreParams();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionTiling::DoLibApiTiling()
{
    if (!SetMatMulTiling()) {
        return ge::GRAPH_FAILED;
    }
    SetSoftMaxTiling();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectedAttentionTiling::PostTiling()
{
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize()); // already check capcity in CheckContext
    auto blockDim = tilingData.multiCoreParams.get_coreNum();
    context_->SetBlockDim(blockDim);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_LOGD(context_, "[%s] final workspace size:%zu.",
            templateName, workspaces[0]);
    OP_LOGD(opName, "[%s] tiling data:%s.", templateName, GetTilingDataDebugStr().c_str());
    OP_LOGD(context_, "[%s] tiling data size: %zu.", templateName, tilingData.GetDataSize());
    context_->SetTilingKey(GetTilingKey());
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

bool NsaSelectedAttentionTiling::SetBmm1TilingInput(matmul_tiling::MatmulApiTiling &bmm1)
{
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
    bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
    bmm1.SetShape(gSize, selectedLength, dSize);
    bmm1.SetOrgShape(gSize, selectedLength, dSize);
    bmm1.SetBias(false);
    mm1BaseN = std::min(mm1BaseN, int32_t((selectedLength + FRACTAL_NUM - 1) / FRACTAL_NUM * FRACTAL_NUM));
    bmm1.SetFixSplit(mm1BaseM, mm1BaseN, mm1BaseK);
    if (bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
        return false;
    }
    return true;
}

bool NsaSelectedAttentionTiling::SetBmm2TilingInput(matmul_tiling::MatmulApiTiling &bmm2)
{
    bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
    bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
    bmm2.SetShape(gSize, d2Size, selectedLength);
    bmm2.SetOrgShape(gSize, d2Size, selectedLength);
    bmm2.SetBias(false);
    bmm2.SetFixSplit(mm2BaseM, mm2BaseN, mm2BaseK);
    if (bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0) {
        return false;
    }
    
    return true;
}

uint64_t NsaSelectedAttentionTiling::GetTilingKey() const
{
    uint64_t tilingKey = 0;
    if (sparseMode != static_cast<int64_t>(SparseMode::NO_MASK)) {
        tilingKey += 1U;
    }
    return tilingKey;
}

bool NsaSelectedAttentionTiling::IsCapable()
{
    return true;
}

void NsaSelectedAttentionTiling::SetMultiCoreParams()
{
    auto &multiCoreParams = tilingData.multiCoreParams;
    // 头核
    int64_t coreNumUsed = std::max(std::min(static_cast<int64_t>(aivNum), t1Size), ONE);
    int64_t s1NumPerHeadCore = CeilDivision(t1Size, coreNumUsed);
    // 尾核
    int64_t tailCoreNum = s1NumPerHeadCore* coreNumUsed - t1Size;
    int64_t headCoreNum = coreNumUsed - tailCoreNum;
    int64_t s1NumPerTailCore = static_cast<int32_t>(s1NumPerHeadCore - 1);
    multiCoreParams.set_coreNum(static_cast<int32_t>(coreNumUsed));
    multiCoreParams.set_headCoreNum(headCoreNum);
    multiCoreParams.set_tailCoreNum(tailCoreNum);
    multiCoreParams.set_s1NumPerHeadCore(s1NumPerHeadCore);
    multiCoreParams.set_s1NumPerTailCore(s1NumPerTailCore);
}

bool NsaSelectedAttentionTiling::SetMatMulTiling()
{   
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    matmul_tiling::MatmulApiTiling bmm1(ascendcPlatform);
    matmul_tiling::MatmulApiTiling bmm2(ascendcPlatform);

    if (!SetBmm1TilingInput(bmm1) || !SetBmm2TilingInput(bmm2)) {
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
    return true;
}

void NsaSelectedAttentionTiling::SetSoftMaxTiling()
{
    int64_t row = SOFTMAX_MAX_BUF_SIZE / selectedLength / sizeof(float);
    if (gSize < row) {
        row = gSize;
    }
    auto softmaxShape = ge::Shape({row, selectedLength});
    const uint32_t localWorkSpaceSize = AscendC::GetSoftMaxMinTmpSize(softmaxShape, sizeof(float), false);
    AscendC::SoftMaxTilingFunc(softmaxShape, sizeof(float), localWorkSpaceSize, tilingData.softmaxTilingData);
    tilingData.multiCoreParams.set_localWorkSpaceSize(localWorkSpaceSize);
}

ge::graphStatus NsaSelectedAttentionTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    int64_t singleCoreSelectKworkspaces = dSize * selectedBlockCount * selectedBlockSize * calcTypeSize;
    int64_t singleCoreSelectVworkspaces = d2Size * selectedBlockCount * selectedBlockSize * calcTypeSize;
    int64_t singleCorePworkspaces =
        gSize * selectedBlockCount * selectedBlockSize * static_cast<int64_t>(sizeof(float));
    int64_t singleCoreSoftmaxResworkspaces = gSize * selectedBlockCount * selectedBlockSize * calcTypeSize;

    workspaces[0] = static_cast<size_t>(
                        (singleCoreSelectKworkspaces + singleCoreSelectVworkspaces + singleCorePworkspaces + singleCoreSoftmaxResworkspaces) * aivNum) +
                        WORK_SPACE_RESERVE_SIZE;
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(NsaSelectedAttention, NsaSelectedAttentionTiling, 0);
} // namespace nsa
} // namespace optiling
