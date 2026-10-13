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
 * \file moe_token_permute_tiling.cpp
 * \brief
 */
#include "moe_token_permute_tiling.h"

namespace {
const static int64_t SPLIT_N = 0;
const static int64_t SPLIT_K = 1;
const static int64_t SPLIT_ACTIVATE_ROW = 2;
const static int64_t NUM_TWO = 2;
const static int64_t NUM_THREE = 3;
const static int64_t NUM_FOUR = 4;
const static int64_t MRG_LIST_NUM = 4;
const static int64_t SORT32_ALIGN_ELEMENT = 32;
const static int64_t ONE_BLOCK_BYTE = 32;
const static size_t DIM_ONE = 1;
const static size_t DIM_TWO = 2;
const static int32_t SIZE_16 = 16;
const static int32_t SIZE_31 = 31;
const static int32_t LENGTH_1024 = 1024;
const static int64_t MAX_COLS_ONE_LOOP = 16376;
const static int64_t ASSIST_NUM = 256;
const static int64_t SPLIT_K_THRESHOLD = 512;
const static int64_t MAX_INDICES_NUM = 512;
const static int64_t INT32_DTYPE_SIZE = 4;
const static int64_t DATA_MOVE_ALIGN = 512;
const static int64_t BUFFER_NUM = 2;
const static int64_t MAX_BLOCK_COUNT = 4095;
const static uint64_t SORT_ONE_CORE_MODE = 1UL;
const static uint64_t SORT_MULTI_CORE_MODE = 2UL;
const static uint64_t ENABLE_NUMOUTTOKENS = 4L;
const static uint64_t SPILT_D_MODE = 2L;
const static int64_t SORT_LIMIT_LENGTH = 16777215;
const static int64_t SORT_WORK_SPACE_NUM = 2;

template <typename T>
static auto GetCeilInt(const T& value1, const T& value2) -> T
{
    if (value2 == 0) {
        return value2;
    }
    return (value1 + value2 - 1) / value2;
}

template <typename T>
static auto GetDiv(const T& value1, const T& value2) -> T
{
    if (value2 == 0) {
        return value2;
    }
    return (value1) / value2;
}

template <typename T>
static auto GetRem(const T& value1, const T& value2) -> T
{
    if (value2 == 0) {
        return value2;
    }
    return value1 % value2;
}

template <typename T1, typename T2>
inline auto FloorAlign(const T1& a, const T2& b) -> T1
{
    if (b != 0) {
        return (a) / b * b;
    }
    return a;
}

template <typename T1, typename T2>
inline auto UpAlign(const T1& a, const T2& b) -> T1
{
    if (b != 0) {
        return (a + b - 1) / b * b;
    }
    return a;
}

inline bool GetLengthByType(int32_t dtype, uint32_t& dsize)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
        case ge::DT_BF16:
            dsize = sizeof(int16_t);
            return true;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            dsize = sizeof(int32_t);
            return true;
        case ge::DT_DOUBLE:
        case ge::DT_INT64:
        case ge::DT_UINT64:
            dsize = sizeof(int64_t);
            return true;
        default:
            return false;
    }
}

inline static int64_t CeilLog4(int64_t x)
{
    return (int64_t)std::ceil(std::log(x) / std::log(NUM_FOUR));
}

inline static int64_t VmsLoops(int64_t x)
{
    int64_t srcWsIndex = 0;
    for (int64_t i = 0; x >= 1; i++) {
        x = (x + NUM_FOUR - 1) / NUM_FOUR;
        srcWsIndex = (srcWsIndex + 1) % SORT_WORK_SPACE_NUM;
        if (x == 1) {
            break;
        }
    }
    return srcWsIndex;
}
} // namespace

namespace optiling {
class MoeTokenPermuteTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass
{
public:
    explicit MoeTokenPermuteTilingBase(gert::TilingContext* context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~MoeTokenPermuteTilingBase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
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
    void Reset();

private:
    ge::graphStatus CheckOutShape();
    void Tiling4IndexCopyCompute();
    void Tiling4SortOutCompute();
    void Tiling4VMSMiddleCompute();
    void Tiling4VBSCompute();
    void ShowTilingData();
    void Tinlig4VBSMultiCoreCompute(PermuteVBSComputeTilingData* tilingData);
    void Tinlig4VBSOneCoreCompute(PermuteVBSComputeTilingData* tilingData);

    int64_t aivNum = 0;
    int64_t realCoreNumAiv = 0;
    int64_t inputDimNum = 0;
    int64_t numOutTokens = 0;
    int64_t totalLength = 0;
    int64_t activateNum = 0;
    int64_t tokenBtypeSize = 0;
    int64_t indicesBtypeSize = 0;
    int64_t sortLoopMaxElement = 0;
    int64_t mrgSortListMaxElement = 1024;
    bool paddedMode = false;
    const char* opName = nullptr;
    MoeTokenPermuteTilingData moeTokenPermuteTilingData;
};

void MoeTokenPermuteTilingBase::Reset()
{
    opName = nullptr;
    return;
}

ge::graphStatus MoeTokenPermuteTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    auto indicesPtr = context_->GetInputTensor(1);
    OP_CHECK_IF(
        indicesPtr == nullptr, OP_LOGE(opName, "fail to get input [indices]"),
        return ge::GRAPH_FAILED);
    if (indicesPtr->GetShapeSize() <= SORT32_ALIGN_ELEMENT) {
        aivNum = 1;
    } else {
        aivNum = ascendcPlatform.GetCoreNumAiv();
    }
    realCoreNumAiv = ascendcPlatform.GetCoreNumAiv();
    aicoreParams_.numBlocks = aivNum;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = FloorAlign(ubSizePlatForm, ONE_BLOCK_BYTE);
    moeTokenPermuteTilingData.set_coreNum(aivNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteTilingBase::CheckOutShape()
{
    // 获取输入shape
    const auto tokenOutput = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tokenOutput);
    const gert::Shape tokensShape = tokenOutput->GetStorageShape();
    const auto indicesOutput = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesOutput);
    const gert::Shape IndicesShape = indicesOutput->GetStorageShape();

    size_t tokensDimNnum = tokensShape.GetDimNum();
    if (tokensDimNnum < DIM_TWO) {
        OP_LOGE(
            context_->GetNodeName(), "The dim number of Output permute_tokens should be greater than 1 but got [%lu].",
            tokensDimNnum);
        return ge::GRAPH_FAILED;
    }

    int64_t cols = 1;
    for (size_t i = 1; i < tokensDimNnum; i++) {
        cols *= tokensShape.GetDim(i);
    }

    size_t IndicesDimNnum = IndicesShape.GetDimNum();
    if (IndicesDimNnum != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "The dim number of Output sort_indices should be 1.");
        return ge::GRAPH_FAILED;
    }

    if (IndicesShape.GetDim(0) != totalLength) {
        OP_LOGE(context_->GetNodeName(), "The size of Output sort_indices should be [%ld].", totalLength);
        return ge::GRAPH_FAILED;
    }

    if (cols != moeTokenPermuteTilingData.get_cols()) {
        OP_LOGE(
            context_->GetNodeName(), "The hidden_size of output permuteTokens should be %ld but got %ld.",
            moeTokenPermuteTilingData.get_cols(), cols);
        return ge::GRAPH_FAILED;
    }

    if (tokensShape.GetDim(0) != numOutTokens) {
        OP_LOGE(
            context_->GetNodeName(), "The dim 0 of output permuteTokens should be %ld but got %ld.", numOutTokens,
            tokensShape.GetDim(0));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteTilingBase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOGD(opName, "MoeTokenPermute Tiling initing.");

    // 获取输入shape
    const gert::Shape tokensShape = context_->GetInputShape(0)->GetStorageShape();
    const gert::Shape IndicesShape = context_->GetInputShape(1)->GetStorageShape();
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    int64_t id = 0;
    const int64_t* numOutTokensPtr = attrs->GetAttrPointer<int64_t>(id++);
    const bool* paddedModePtr = attrs->GetAttrPointer<bool>(id);
    numOutTokens = *numOutTokensPtr;
    paddedMode = *paddedModePtr;

    if (paddedMode == true) {
        OP_LOGE(context_->GetNodeName(), "Currently only support padded_mode is false");
        return ge::GRAPH_FAILED;
    }

    size_t TokensDimNnum = tokensShape.GetDimNum();

    int64_t cols = 1;
    for (size_t i = 1; i < TokensDimNnum; i++) {
        cols *= tokensShape.GetDim(i);
    }

    size_t indicesDimNum = IndicesShape.GetDimNum();
    if (indicesDimNum != DIM_TWO && indicesDimNum != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "The dim number of indices should be 2 or 1 but got [%lu].", indicesDimNum);
        return ge::GRAPH_FAILED;
    }

    if (tokensShape.GetDim(0) != IndicesShape.GetDim(0)) {
        OP_LOGE(
            context_->GetNodeName(), "Input token's dim 0 [%ld] should be same with indices' dim 0 [%ld].",
            tokensShape.GetDim(0), IndicesShape.GetDim(0));
        return ge::GRAPH_FAILED;
    }

    tokenBtypeSize = ge::GetSizeByDataType(context_->GetInputDesc(0)->GetDataType());
    indicesBtypeSize = ge::GetSizeByDataType(context_->GetInputDesc(1)->GetDataType());

    moeTokenPermuteTilingData.set_cols(cols);
    auto tokenOneBlockNum = GetDiv(ONE_BLOCK_BYTE, tokenBtypeSize);
    auto colsAlign = GetCeilInt(cols, tokenOneBlockNum) * tokenOneBlockNum;
    moeTokenPermuteTilingData.set_n(IndicesShape.GetDim(0));
    moeTokenPermuteTilingData.set_colsAlign(colsAlign);

    int64_t topK = (indicesDimNum == 1) ? 1 : IndicesShape.GetDim(1);
    if (topK > MAX_INDICES_NUM) {
        OP_LOGE(
            context_->GetNodeName(), "Input indices's dim 1 [%ld] should not large than max topK[%ld].", topK,
            MAX_INDICES_NUM);
        return ge::GRAPH_FAILED;
    }

    moeTokenPermuteTilingData.set_topK(topK);
    totalLength = moeTokenPermuteTilingData.get_n() * moeTokenPermuteTilingData.get_topK();
    if (totalLength >= SORT_LIMIT_LENGTH) {
        OP_LOGE(
            context_->GetNodeName(), "The elements num of indices [%ld] should be less than [%ld].", totalLength,
            SORT_LIMIT_LENGTH);
        return ge::GRAPH_FAILED;
    }

    numOutTokens = (numOutTokens <= 0) ? numOutTokens + totalLength : numOutTokens;
    numOutTokens = std::min(numOutTokens, totalLength);
    numOutTokens = std::max(numOutTokens, (int64_t)0);
    auto ret = CheckOutShape();
    return ret;
}

void MoeTokenPermuteTilingBase::ShowTilingData()
{
    OP_LOGD(
        opName,
        "indexCopyCTilingData is needCoreNum:%ld, frontCoreNum:%ld, "
        "tailCoreNum:%ld, coreCalcNum:%ld, coreCalcTail:%ld, oneTokenBtypeSize:%ld, "
        "onceIndicesTokenMoveTimes:%ld, onceUbTokenNums:%ld, onceIndicesTokenNums:%ld, "
        "onceIndices:%ld, oneTokenlastMove:%ld, oneTokenOnceMove:%ld, oneTokenMoveTimes:%ld, "
        "frontCoreLoop:%ld, frontCoreLastTokenNums:%ld, tailCoreLoop:%ld, tailCoreLastTokenNums:%ld, "
        "tailLastonceIndicesTokenMoveTimes:%ld, tailLastIndicesLastTokenNums:%ld, "
        "frontLastonceIndicesTokenMoveTimes:%ld, frontLastIndicesLastTokenNums:%ld, "
        "numOutTokens:%ld, tokenUB:%ld, indicesUB:%ld",
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_needCoreNum(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_frontCoreNum(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_tailCoreNum(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_coreCalcNum(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_coreCalcTail(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_oneTokenBtypeSize(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_onceIndicesTokenMoveTimes(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_onceUbTokenNums(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_onceIndicesTokenNums(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_onceIndices(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_oneTokenlastMove(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_oneTokenOnceMove(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_oneTokenMoveTimes(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_frontCoreLoop(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_frontCoreLastTokenNums(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_tailCoreLoop(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_tailCoreLastTokenNums(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_tailLastonceIndicesTokenMoveTimes(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_tailLastIndicesLastTokenNums(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_frontLastonceIndicesTokenMoveTimes(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_frontLastIndicesLastTokenNums(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_numOutTokens(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_tokenUB(),
        moeTokenPermuteTilingData.indexCopyComputeParamsOp.get_indicesUB());
    OP_LOGD(
        opName,
        "PermuteVBSComputeTilingData is needCoreNum:%ld, perCoreElements:%ld, perCoreLoops:%ld, "
        "perCorePerLoopElements:%ld, "
        "perCoreLastLoopElements:%ld, lastCoreElements:%ld, lastCoreLoops:%ld, lastCorePerLoopElements:%ld, "
        "lastCoreLastLoopElements:%ld, oneLoopMaxElements:%ld, lastCoreWSindex:%ld",
        moeTokenPermuteTilingData.vbsComputeParamsOp.get_needCoreNum(),
        moeTokenPermuteTilingData.vbsComputeParamsOp.get_perCoreElements(),
        moeTokenPermuteTilingData.vbsComputeParamsOp.get_perCoreLoops(),
        moeTokenPermuteTilingData.vbsComputeParamsOp.get_perCorePerLoopElements(),
        moeTokenPermuteTilingData.vbsComputeParamsOp.get_perCoreLastLoopElements(),
        moeTokenPermuteTilingData.vbsComputeParamsOp.get_lastCoreElements(),
        moeTokenPermuteTilingData.vbsComputeParamsOp.get_lastCoreLoops(),
        moeTokenPermuteTilingData.vbsComputeParamsOp.get_lastCorePerLoopElements(),
        moeTokenPermuteTilingData.vbsComputeParamsOp.get_lastCoreLastLoopElements(),
        moeTokenPermuteTilingData.vbsComputeParamsOp.get_oneLoopMaxElements(),
        moeTokenPermuteTilingData.vbsComputeParamsOp.get_lastCoreWSindex());
    OP_LOGD(
        opName, "PermuteVMSMiddleComputeTilingData is needCoreNum:%ld",
        moeTokenPermuteTilingData.vmsMiddleComputeParamsOp.get_needCoreNum());
    OP_LOGD(
        opName, "moeTokenPermuteTilingData is coreNum:%ld, n:%ld, cols:%ld, colsAlign:%ld, k:%ld",
        moeTokenPermuteTilingData.get_coreNum(), moeTokenPermuteTilingData.get_n(),
        moeTokenPermuteTilingData.get_cols(), moeTokenPermuteTilingData.get_colsAlign(),
        moeTokenPermuteTilingData.get_topK());
    OP_LOGD(
        opName, "PermuteSortOutComputeTilingData is oneLoopMaxElements:%ld",
        moeTokenPermuteTilingData.sortOutComputeParamsOp.get_oneLoopMaxElements());
}
ge::graphStatus MoeTokenPermuteTilingBase::DoOpTiling()
{
    sortLoopMaxElement = (aicoreParams_.ubSize - aivNum * ONE_BLOCK_BYTE) / (NUM_FOUR * NUM_TWO * NUM_FOUR) /
                         SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
    Tiling4VBSCompute();
    Tiling4VMSMiddleCompute();
    Tiling4SortOutCompute();
    Tiling4IndexCopyCompute();
    ShowTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeTokenPermuteTilingBase::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus MoeTokenPermuteTilingBase::GetWorkspaceSize()
{
    // 计算workspace大小
    size_t sortWorkspaceSize = sizeof(float) * static_cast<size_t>(totalLength * NUM_TWO * NUM_THREE); // 排序需要的空间
    size_t coreSyncWorkspaceSize =
        moeTokenPermuteTilingData.get_coreNum() * SORT32_ALIGN_ELEMENT * NUM_TWO; // 多核同步需要的空间
    workspaceSize_ = sortWorkspaceSize + coreSyncWorkspaceSize + SIZE_16 * LENGTH_1024 * LENGTH_1024;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteTilingBase::PostTiling()
{
    context_->SetBlockDim(aivNum);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    // 涉及核间同步的算子必须设置schedule_mode为1，独占全核
    context_->SetScheduleMode(1);
    moeTokenPermuteTilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(moeTokenPermuteTilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MoeTokenPermuteTilingBase::Tinlig4VBSOneCoreCompute(PermuteVBSComputeTilingData* tilingData)
{
    tilingData->set_needCoreNum(1);
    tilingData->set_perCoreElements(totalLength);
    tilingData->set_perCoreLoops(1);
    tilingData->set_perCorePerLoopElements(tilingData->get_perCoreElements());
    tilingData->set_perCoreLastLoopElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreLoops(1);
    tilingData->set_lastCorePerLoopElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreLastLoopElements(tilingData->get_perCoreElements());
}

void MoeTokenPermuteTilingBase::Tinlig4VBSMultiCoreCompute(PermuteVBSComputeTilingData* tilingData)
{
    int64_t needCoreNum = GetCeilInt(totalLength, sortLoopMaxElement);      // 向上取整
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum))); // 用到多核时，核数最多是4^x
    needCoreNum = std::min(needCoreNum, aivNum);                            // 不能超过物理核数

    int64_t perCoreElements = GetDiv(totalLength, needCoreNum); // 每个核处理的元素数
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = GetCeilInt(totalLength, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    tilingData->set_needCoreNum(needCoreNum);
    tilingData->set_perCoreElements(perCoreElements);
    tilingData->set_perCoreLoops(
        GetCeilInt(tilingData->get_perCoreElements(), sortLoopMaxElement)); // 每个核处理的loop数
    tilingData->set_perCorePerLoopElements(std::min(tilingData->get_perCoreElements(), sortLoopMaxElement));

    tilingData->set_perCoreLastLoopElements(
        tilingData->get_perCoreElements() -
        (tilingData->get_perCoreLoops() - 1) * tilingData->get_perCorePerLoopElements());

    tilingData->set_lastCoreElements(
        totalLength - (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreElements());
    tilingData->set_lastCoreLoops(GetCeilInt(tilingData->get_lastCoreElements(), sortLoopMaxElement));
    tilingData->set_lastCorePerLoopElements(std::min(tilingData->get_lastCoreElements(), sortLoopMaxElement));
    tilingData->set_lastCoreLastLoopElements(
        tilingData->get_lastCoreElements() -
        (tilingData->get_lastCoreLoops() - 1) * tilingData->get_lastCorePerLoopElements());
    tilingData->set_lastCoreWSindex(
        std::abs(VmsLoops(tilingData->get_lastCoreLoops()) - VmsLoops(tilingData->get_perCoreLoops())));
}

void MoeTokenPermuteTilingBase::Tiling4VBSCompute()
{
    if (totalLength <= sortLoopMaxElement) { // 排序只用到一个核排序
        tilingKey_ = SORT_ONE_CORE_MODE;
    } else {
        tilingKey_ = SORT_MULTI_CORE_MODE;
    }

    auto tilingData = &moeTokenPermuteTilingData.vbsComputeParamsOp;
    tilingData->set_oneLoopMaxElements(sortLoopMaxElement);
    if (GetTilingKey() == 1UL) { // 只用到一个核
        Tinlig4VBSOneCoreCompute(tilingData);
        return;
    }
    Tinlig4VBSMultiCoreCompute(tilingData);
}

void MoeTokenPermuteTilingBase::Tiling4VMSMiddleCompute()
{
    auto vbsComputeTilingData = &moeTokenPermuteTilingData.vbsComputeParamsOp;
    auto tilingData = &moeTokenPermuteTilingData.vmsMiddleComputeParamsOp;
    if (vbsComputeTilingData->get_needCoreNum() <= MRG_LIST_NUM) { // 队列数小于一次vms则没有中间归并
        tilingData->set_needCoreNum(0);                            // 需要的核数
        return;
    }
    int64_t needCoreNum = GetCeilInt(vbsComputeTilingData->get_needCoreNum(), MRG_LIST_NUM);
    tilingData->set_needCoreNum(needCoreNum); // 需要的核数
}

void MoeTokenPermuteTilingBase::Tiling4SortOutCompute()
{
    auto tilingData = &moeTokenPermuteTilingData.sortOutComputeParamsOp;
    tilingData->set_oneLoopMaxElements(mrgSortListMaxElement);
}
void MoeTokenPermuteTilingBase::Tiling4IndexCopyCompute()
{
    auto tilingData = &moeTokenPermuteTilingData.indexCopyComputeParamsOp;
    int64_t tokenNums = moeTokenPermuteTilingData.get_n();
    int64_t topK = moeTokenPermuteTilingData.get_topK();
    int64_t cols = moeTokenPermuteTilingData.get_cols();

    if (numOutTokens != tokenNums * topK) {
        tilingKey_ = tilingKey_ + ENABLE_NUMOUTTOKENS;
    }
    tilingData->set_numOutTokens(numOutTokens);

    int64_t frontCoreNum = GetRem(tokenNums, realCoreNumAiv) != 0 ? GetRem(tokenNums, realCoreNumAiv) : realCoreNumAiv;
    int64_t tailCoreNum = tokenNums <= realCoreNumAiv ? 0 : realCoreNumAiv - frontCoreNum;
    int64_t numBlocks = frontCoreNum + tailCoreNum;
    int64_t coreCalcNum = GetCeilInt(tokenNums, realCoreNumAiv);
    int64_t coreCalcTail = GetDiv(tokenNums, realCoreNumAiv);

    int64_t ubLeft = aicoreParams_.ubSize - MAX_INDICES_NUM * INT32_DTYPE_SIZE;
    int64_t oneTokenBtypeSize = cols * tokenBtypeSize;

    int64_t oneTokenBtypeSizeAlign32 = UpAlign(oneTokenBtypeSize, ONE_BLOCK_BYTE);

    int64_t oneTokenlastMove = 1;
    int64_t oneTokenOnceMove = 1;
    int64_t oneTokenMoveTimes = 1;
    int64_t onceIndicesTokenMoveTimes = 1;
    int64_t onceUbTokenNums = 1;
    int64_t onceIndicesTokenNums = 1;
    int64_t onceIndices = 1;
    int64_t tokenUB = 1;
    int64_t indicesUB = 1;
    if (ubLeft >= BUFFER_NUM * oneTokenBtypeSizeAlign32) {
        onceUbTokenNums = GetDiv(
            static_cast<int64_t>(aicoreParams_.ubSize),
            oneTokenBtypeSizeAlign32 * BUFFER_NUM + topK * BUFFER_NUM * INT32_DTYPE_SIZE);
        onceUbTokenNums = std::min(onceUbTokenNums, MAX_BLOCK_COUNT);
        int64_t TopKUbLeft = aicoreParams_.ubSize - onceUbTokenNums * oneTokenBtypeSizeAlign32 * BUFFER_NUM;
        onceIndicesTokenMoveTimes = GetDiv(TopKUbLeft, onceUbTokenNums * topK * INT32_DTYPE_SIZE);
        onceIndicesTokenNums = onceIndicesTokenMoveTimes * onceUbTokenNums;
        onceIndices = onceIndicesTokenNums * topK;
        tokenUB = onceUbTokenNums * oneTokenBtypeSizeAlign32;
        indicesUB = UpAlign(onceIndices * INT32_DTYPE_SIZE, ONE_BLOCK_BYTE);
    } else {
        onceIndicesTokenNums = GetDiv(MAX_INDICES_NUM, topK);
        onceIndices = onceIndicesTokenNums * topK;
        oneTokenOnceMove = GetDiv(FloorAlign(GetDiv(ubLeft, BUFFER_NUM), DATA_MOVE_ALIGN), tokenBtypeSize);
        oneTokenMoveTimes = GetCeilInt(cols, oneTokenOnceMove);
        oneTokenlastMove = cols - (oneTokenMoveTimes - 1) * oneTokenOnceMove;
        tilingKey_ = tilingKey_ + SPILT_D_MODE;
        tokenUB = oneTokenOnceMove * tokenBtypeSize;
        indicesUB = MAX_INDICES_NUM * INT32_DTYPE_SIZE;
    }

    int64_t frontCoreLoop = GetCeilInt(coreCalcNum, onceIndicesTokenNums);
    int64_t frontCoreLastTokenNums = coreCalcNum - (frontCoreLoop - 1) * onceIndicesTokenNums;
    int64_t tailCoreLoop = GetCeilInt(coreCalcTail, onceIndicesTokenNums);
    int64_t tailCoreLastTokenNums = coreCalcTail - (tailCoreLoop - 1) * onceIndicesTokenNums;
    int64_t tailLastonceIndicesTokenMoveTimes = GetCeilInt(tailCoreLastTokenNums, onceUbTokenNums);
    int64_t tailLastIndicesLastTokenNums =
        tailCoreLastTokenNums - (tailLastonceIndicesTokenMoveTimes - 1) * onceUbTokenNums;
    int64_t frontLastonceIndicesTokenMoveTimes = GetCeilInt(frontCoreLastTokenNums, onceUbTokenNums);

    int64_t frontLastIndicesLastTokenNums =
        frontCoreLastTokenNums - (frontLastonceIndicesTokenMoveTimes - 1) * onceUbTokenNums;
    tilingData->set_tokenUB(tokenUB);
    tilingData->set_indicesUB(indicesUB);
    tilingData->set_needCoreNum(numBlocks);
    tilingData->set_frontCoreNum(frontCoreNum);
    tilingData->set_tailCoreNum(tailCoreNum);
    tilingData->set_coreCalcNum(coreCalcNum);
    tilingData->set_coreCalcTail(coreCalcTail);
    tilingData->set_oneTokenBtypeSize(oneTokenBtypeSize);
    tilingData->set_onceIndicesTokenMoveTimes(onceIndicesTokenMoveTimes);
    tilingData->set_onceUbTokenNums(onceUbTokenNums);
    tilingData->set_onceIndicesTokenNums(onceIndicesTokenNums);
    tilingData->set_onceIndices(onceIndices);
    tilingData->set_oneTokenlastMove(oneTokenlastMove);
    tilingData->set_oneTokenOnceMove(oneTokenOnceMove);
    tilingData->set_oneTokenMoveTimes(oneTokenMoveTimes);
    tilingData->set_frontCoreLoop(frontCoreLoop);
    tilingData->set_frontCoreLastTokenNums(frontCoreLastTokenNums);
    tilingData->set_tailCoreLoop(tailCoreLoop);
    tilingData->set_tailCoreLastTokenNums(tailCoreLastTokenNums);
    tilingData->set_tailLastonceIndicesTokenMoveTimes(tailLastonceIndicesTokenMoveTimes);
    tilingData->set_tailLastIndicesLastTokenNums(tailLastIndicesLastTokenNums);
    tilingData->set_frontLastonceIndicesTokenMoveTimes(frontLastonceIndicesTokenMoveTimes);
    tilingData->set_frontLastIndicesLastTokenNums(frontLastIndicesLastTokenNums);
    aivNum = std::max(aivNum, numBlocks);
}

static ge::graphStatus TilingForMoeTokenPermute(gert::TilingContext* context)
{
    MoeTokenPermuteTilingBase tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForMoeTokenPermute(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeTokenPermute)
    .Tiling(TilingForMoeTokenPermute)
    .TilingParse<MoeTokenPermuteCompileInfo>(TilingPrepareForMoeTokenPermute);
} // namespace optiling
