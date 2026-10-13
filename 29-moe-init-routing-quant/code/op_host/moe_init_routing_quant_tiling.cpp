/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_init_routing_quant_tiling.cpp
 * \brief
 */
#include "moe_init_routing_quant_tiling.h"

namespace optiling {
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

inline static int64_t CeilLog4(int64_t x)
{
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
}

class MoeInitRoutingQuantTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass
{
public:
    explicit MoeInitRoutingQuantTilingBase(gert::TilingContext* context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~MoeInitRoutingQuantTilingBase() override = default;

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
    void Tiling4GatherOutComputeSplitK();
    void Tiling4GatherOutComputeSplitN();
    void Tiling4GatherOutComputeRow();
    void Tiling4GatherOutCompute();
    void Tiling4SrcToDstCompute();
    void Tiling4SortOutCompute();
    void Tiling4VMSMiddleCompute();
    void Tiling4VBSCompute();
    void ShowTilingData();
    void Tinlig4VBSMultiCoreCompute(QuantVBSComputeTilingData* tilingData);
    void Tinlig4VBSOneCoreCompute(QuantVBSComputeTilingData* tilingData);
    int64_t CalPerLoopMaxRows(int64_t maxColsOneLoop, int64_t kFactor);

    int64_t aivNum;
    int64_t sortLoopMaxElement = 0;
    int64_t mrgSortListMaxElement = 1024;
    int64_t totalLength = 0;
    int64_t activateNum = 0;
    int64_t inuptXDtypeSize_;

    const char* opName = "";
    MoeInitRoutingQuantTilingData moeInitRoutingTilingData;
};

void MoeInitRoutingQuantTilingBase::Reset()
{
    opName = nullptr;
    return;
}

ge::graphStatus MoeInitRoutingQuantTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (context_->GetInputTensor(1)->GetShapeSize() <= SORT32_ALIGN_ELEMENT) {
        aivNum = 1;
    } else {
        aivNum = ascendcPlatform.GetCoreNumAiv();
    }
    aicoreParams_.numBlocks = aivNum;
    uint64_t ubSizePlatForm;
    uint64_t l1SizePlatForm;
    uint64_t l0CSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1SizePlatForm);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;
    aicoreParams_.l1Size = l1SizePlatForm;
    aicoreParams_.l0cSize = l0CSizePlatForm;
    moeInitRoutingTilingData.set_coreNum(aivNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingQuantTilingBase::CheckOutShape()
{
    // 获取输入shape
    const gert::Shape expandedXShape = context_->GetOutputShape(0)->GetStorageShape();
    const gert::Shape expandedRowIdxShape = context_->GetOutputShape(1)->GetStorageShape();
    const gert::Shape expandedExpertIdxShape = context_->GetOutputShape(2)->GetStorageShape();

    size_t expandedXDimNnum = expandedXShape.GetDimNum();
    if (expandedXDimNnum != DIM_TWO) {
        OP_LOGE(context_->GetNodeName(), "The dim number of expanded_x should be 2.");
        return ge::GRAPH_FAILED;
    }

    size_t expandedRowIdxDimNnum = expandedRowIdxShape.GetDimNum();
    if (expandedRowIdxDimNnum != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "The dim number of expanded_row_idx should be 1.");
        return ge::GRAPH_FAILED;
    }

    if (expandedRowIdxShape != expandedExpertIdxShape) {
        OP_LOGE(context_->GetNodeName(), "The shape of expanded_row_idx and expanded_expert_idx should be same.");
        return ge::GRAPH_FAILED;
    }

    if (expandedXShape.GetDim(0) !=
        std::min(moeInitRoutingTilingData.get_n(), activateNum) * moeInitRoutingTilingData.get_k()) {
        OP_LOGE(
            context_->GetNodeName(), "The first dim of expanded_x should be %ld.",
            std::min(moeInitRoutingTilingData.get_n(), activateNum) * moeInitRoutingTilingData.get_k());
        return ge::GRAPH_FAILED;
    }

    if (expandedXShape.GetDim(1) != moeInitRoutingTilingData.get_cols()) {
        OP_LOGE(
            context_->GetNodeName(), "The second dim of expanded_x should be %ld.",
            moeInitRoutingTilingData.get_cols());
        return ge::GRAPH_FAILED;
    }

    if (expandedRowIdxShape.GetDim(0) != totalLength) {
        OP_LOGE(
            context_->GetNodeName(), "The first dim of expanded_row_idx and expanded_expert_idx should be %ld.",
            totalLength);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingQuantTilingBase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();

    // 获取输入shape
    const gert::Shape xShape = context_->GetInputShape(0)->GetStorageShape();
    const gert::Shape rowIdxShape = context_->GetInputShape(1)->GetStorageShape();
    const gert::Shape expertIdxShape = context_->GetInputShape(2)->GetStorageShape();

    auto attrs = context_->GetAttrs();
    const int64_t* activateNumPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, activateNumPtr);
    activateNum = *activateNumPtr;
    OP_LOGI(context_->GetNodeName(), "activateNum is: %ld", activateNum);

    // 参数校验
    size_t xDimNnum = xShape.GetDimNum();
    if (xDimNnum != DIM_TWO) {
        OP_LOGE(context_->GetNodeName(), "The dim number of x should be 2.");
        return ge::GRAPH_FAILED;
    }

    size_t rowIdxDimNum = rowIdxShape.GetDimNum();
    if (rowIdxDimNum != DIM_TWO) {
        OP_LOGE(context_->GetNodeName(), "The dim number of row_idx should be 2.");
        return ge::GRAPH_FAILED;
    }

    if (rowIdxShape != expertIdxShape) {
        OP_LOGE(context_->GetNodeName(), "The shape of row_idx and expert_idx should be same.");
        return ge::GRAPH_FAILED;
    }

    if (xShape.GetDim(0) != expertIdxShape.GetDim(0)) {
        OP_LOGE(context_->GetNodeName(), "Input rows should be same.");
        return ge::GRAPH_FAILED;
    }

    if (activateNum < 0) {
        OP_LOGE(context_->GetNodeName(), "active_num must be a non-negative number.");
        return ge::GRAPH_FAILED;
    }

    const float* scalePtr = attrs->GetAttrPointer<float>(1);
    float scale = *scalePtr;

    const float* offsetPtr = attrs->GetAttrPointer<float>(2);
    float offset = *offsetPtr;

    moeInitRoutingTilingData.set_scale(scale);
    moeInitRoutingTilingData.set_offset(offset);

    moeInitRoutingTilingData.set_cols(xShape.GetDim(1));
    moeInitRoutingTilingData.set_n(expertIdxShape.GetDim(0));
    moeInitRoutingTilingData.set_k(expertIdxShape.GetDim(1));
    totalLength = moeInitRoutingTilingData.get_n() * moeInitRoutingTilingData.get_k();
    inuptXDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(context_->GetInputDesc(0)->GetDataType()));
    auto ret = CheckOutShape();
    return ret;
}

void MoeInitRoutingQuantTilingBase::ShowTilingData()
{
    OP_LOGI(
        opName, "moeInitRoutingTilingData is coreNum:%ld, n:%ld, cols:%ld, k:%ld, scale is: %f, offset is: %f",
        moeInitRoutingTilingData.get_coreNum(), moeInitRoutingTilingData.get_n(), moeInitRoutingTilingData.get_cols(),
        moeInitRoutingTilingData.get_k(), moeInitRoutingTilingData.get_scale(), moeInitRoutingTilingData.get_offset());
    OP_LOGI(
        opName,
        "VBSComputeTilingData is needCoreNum:%ld, perCoreElements:%ld, perCoreLoops:%ld, perCorePerLoopElements:%ld, "
        "perCoreLastLoopElements:%ld, lastCoreElements:%ld, lastCoreLoops:%ld, lastCorePerLoopElements:%ld, "
        "lastCoreLastLoopElements:%ld, oneLoopMaxElements:%ld",
        moeInitRoutingTilingData.vbsComputeParamsOp.get_needCoreNum(),
        moeInitRoutingTilingData.vbsComputeParamsOp.get_perCoreElements(),
        moeInitRoutingTilingData.vbsComputeParamsOp.get_perCoreLoops(),
        moeInitRoutingTilingData.vbsComputeParamsOp.get_perCorePerLoopElements(),
        moeInitRoutingTilingData.vbsComputeParamsOp.get_perCoreLastLoopElements(),
        moeInitRoutingTilingData.vbsComputeParamsOp.get_lastCoreElements(),
        moeInitRoutingTilingData.vbsComputeParamsOp.get_lastCoreLoops(),
        moeInitRoutingTilingData.vbsComputeParamsOp.get_lastCorePerLoopElements(),
        moeInitRoutingTilingData.vbsComputeParamsOp.get_lastCoreLastLoopElements(),
        moeInitRoutingTilingData.vbsComputeParamsOp.get_oneLoopMaxElements());
    OP_LOGI(
        opName, "VMSMiddleComputeTilingData is needCoreNum:%ld",
        moeInitRoutingTilingData.vmsMiddleComputeParamsOp.get_needCoreNum());
    OP_LOGI(
        opName, "SortOutComputeTilingData is oneLoopMaxElements:%ld",
        moeInitRoutingTilingData.sortOutComputeParamsOp.get_oneLoopMaxElements());
    OP_LOGI(
        opName,
        "SrcToDstComputeTilingData is needCoreNum:%ld, activateRows:%ld, perCoreRows:%ld, perCorePerLoopRows:%ld, "
        "perCoreLastLoopRows:%ld, lastCoreRows:%ld, lastCorePerLoopRows:%ld, lastCoreLastLoopRows:%ld, "
        "maxColsOneLoop:%ld",
        moeInitRoutingTilingData.srcToDstComputeParamsOp.get_needCoreNum(),
        moeInitRoutingTilingData.srcToDstComputeParamsOp.get_activateRows(),
        moeInitRoutingTilingData.srcToDstComputeParamsOp.get_perCoreRows(),
        moeInitRoutingTilingData.srcToDstComputeParamsOp.get_perCorePerLoopRows(),
        moeInitRoutingTilingData.srcToDstComputeParamsOp.get_perCoreLastLoopRows(),
        moeInitRoutingTilingData.srcToDstComputeParamsOp.get_lastCoreRows(),
        moeInitRoutingTilingData.srcToDstComputeParamsOp.get_lastCorePerLoopRows(),
        moeInitRoutingTilingData.srcToDstComputeParamsOp.get_lastCoreLastLoopRows(),
        moeInitRoutingTilingData.srcToDstComputeParamsOp.get_maxColsOneLoop());
    OP_LOGI(
        opName,
        "GatherOutComputeTilingData is needCoreNum:%ld, activateRows:%ld, perCoreRows:%ld, perCoreK:%ld, "
        "perCorePerLoopK:%ld, perCoreLastLoopK:%ld, perCorePerLoopRows:%ld, "
        "perCoreLastLoopRows:%ld, lastCoreRows:%ld, lastCoreK:%ld, lastCorePerLoopK:%ld, lastCoreLastLoopK:%ld, "
        "lastCorePerLoopRows:%ld, lastCoreLastLoopRows:%ld, "
        "maxColsOneLoop:%ld, splitFlag:%ld",
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_needCoreNum(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_activateRows(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCoreRows(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCoreK(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCorePerLoopK(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCoreLastLoopK(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCorePerLoopRows(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCoreLastLoopRows(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCoreRows(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCoreK(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCorePerLoopK(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCoreLastLoopK(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCorePerLoopRows(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCoreLastLoopRows(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_maxColsOneLoop(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_splitFlag());
}

ge::graphStatus MoeInitRoutingQuantTilingBase::DoOpTiling()
{
    sortLoopMaxElement = (aicoreParams_.ubSize - aivNum * ONE_BLOCK_BYTE) / (NUM_FOUR * NUM_TWO * NUM_FOUR) /
                         SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
    Tiling4VBSCompute();
    Tiling4VMSMiddleCompute();
    Tiling4SortOutCompute();
    Tiling4SrcToDstCompute();
    Tiling4GatherOutCompute();
    ShowTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingQuantTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeInitRoutingQuantTilingBase::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus MoeInitRoutingQuantTilingBase::GetWorkspaceSize()
{
    // 计算workspace大小
    size_t sortWorkspaceSize = sizeof(float) * static_cast<size_t>(totalLength * NUM_TWO * NUM_THREE); // 排序需要的空间
    size_t coreSyncWorkspaceSize =
        moeInitRoutingTilingData.get_coreNum() * SORT32_ALIGN_ELEMENT * NUM_TWO; // 多核同步需要的空间
    size_t scatterWorkspaceSize = totalLength * sizeof(int32_t);
    workspaceSize_ =
        sortWorkspaceSize + coreSyncWorkspaceSize + scatterWorkspaceSize + SIZE_16 * LENGTH_1024 * LENGTH_1024;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingQuantTilingBase::PostTiling()
{
    context_->SetBlockDim(aivNum);
    // 涉及核间同步的算子必须设置schedule_mode为1，独占全核
    context_->SetScheduleMode(1);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    moeInitRoutingTilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(moeInitRoutingTilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
void MoeInitRoutingQuantTilingBase::Tinlig4VBSOneCoreCompute(QuantVBSComputeTilingData* tilingData)
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

void MoeInitRoutingQuantTilingBase::Tinlig4VBSMultiCoreCompute(QuantVBSComputeTilingData* tilingData)
{
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength, sortLoopMaxElement); // 向上取整
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum)));                    // 用到多核时，核数最多是4^x
    needCoreNum = std::min(needCoreNum, aivNum);                         // 不能超过物理核数

    int64_t perCoreElements = totalLength / needCoreNum; // 每个核处理的元素数
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = Ops::Base::CeilDiv(totalLength, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    tilingData->set_needCoreNum(needCoreNum);
    do {
        tilingData->set_perCoreElements(perCoreElements);
        tilingData->set_perCoreLoops(
            Ops::Base::CeilDiv(tilingData->get_perCoreElements(), sortLoopMaxElement)); // 每个核处理的loop数
        tilingData->set_perCorePerLoopElements(std::min(tilingData->get_perCoreElements(), sortLoopMaxElement));

        tilingData->set_perCoreLastLoopElements(
            tilingData->get_perCoreElements() -
            (tilingData->get_perCoreLoops() - 1) * tilingData->get_perCorePerLoopElements());

        tilingData->set_lastCoreElements(
            totalLength - (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreElements());
        tilingData->set_lastCoreLoops(tilingData->get_perCoreLoops());
        int64_t lastCorePerLoopElements =
            Ops::Base::CeilDiv(
                Ops::Base::CeilDiv(tilingData->get_lastCoreElements(), tilingData->get_lastCoreLoops()),
                SORT32_ALIGN_ELEMENT) *
            SORT32_ALIGN_ELEMENT;
        tilingData->set_lastCorePerLoopElements(lastCorePerLoopElements);
        tilingData->set_lastCoreLastLoopElements(
            tilingData->get_lastCoreElements() -
            (tilingData->get_lastCoreLoops() - 1) * tilingData->get_lastCorePerLoopElements());
        perCoreElements -= SORT32_ALIGN_ELEMENT;
    } while (tilingData->get_lastCoreLastLoopElements() <= 0 && perCoreElements > 0);
    OP_CHECK_IF(tilingData->get_lastCoreLastLoopElements() <= 0,
                    OP_LOGE(opName, "vbs tiling failed"),
                    ;);
}

void MoeInitRoutingQuantTilingBase::Tiling4VBSCompute()
{
    if (totalLength <= sortLoopMaxElement) { // 排序只用到一个核排序
        tilingKey_ = 1UL;
    } else {
        tilingKey_ = 2UL;
    }

    auto tilingData = &moeInitRoutingTilingData.vbsComputeParamsOp;
    tilingData->set_oneLoopMaxElements(sortLoopMaxElement);
    if (GetTilingKey() == 1UL) { // 只用到一个核
        Tinlig4VBSOneCoreCompute(tilingData);
        return;
    }
    Tinlig4VBSMultiCoreCompute(tilingData);
}

void MoeInitRoutingQuantTilingBase::Tiling4VMSMiddleCompute()
{
    auto vbsComputeTilingData = &moeInitRoutingTilingData.vbsComputeParamsOp;
    auto tilingData = &moeInitRoutingTilingData.vmsMiddleComputeParamsOp;
    if (vbsComputeTilingData->get_needCoreNum() <= MRG_LIST_NUM) { // 队列数小于一次vms则没有中间归并
        tilingData->set_needCoreNum(0);                            // 需要的核数
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(vbsComputeTilingData->get_needCoreNum(), MRG_LIST_NUM);
    tilingData->set_needCoreNum(needCoreNum); // 需要的核数
}

void MoeInitRoutingQuantTilingBase::Tiling4SortOutCompute()
{
    auto tilingData = &moeInitRoutingTilingData.sortOutComputeParamsOp;
    tilingData->set_oneLoopMaxElements(mrgSortListMaxElement);
}

void MoeInitRoutingQuantTilingBase::Tiling4SrcToDstCompute()
{
    auto tilingData = &moeInitRoutingTilingData.srcToDstComputeParamsOp;

    int64_t perLoopMaxRows = (aicoreParams_.ubSize - ASSIST_NUM * sizeof(float) - aivNum * SORT32_ALIGN_ELEMENT) /
                             (SORT32_ALIGN_ELEMENT * NUM_TWO) / NUM_TWO;
    int64_t perCoreRows = Ops::Base::CeilDiv(totalLength, aivNum);
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength, perCoreRows);
    tilingData->set_needCoreNum(needCoreNum);
    int64_t lastCoreNum = totalLength - perCoreRows * (tilingData->get_needCoreNum() - 1);

    tilingData->set_perCoreRows(perCoreRows);

    if (perLoopMaxRows >= tilingData->get_perCoreRows()) { // 一个loop结束
        tilingData->set_perCorePerLoopRows(tilingData->get_perCoreRows());
        tilingData->set_perCoreLastLoopRows(tilingData->get_perCoreRows());
    } else {
        tilingData->set_perCorePerLoopRows(perLoopMaxRows);
        tilingData->set_perCoreLastLoopRows(
            tilingData->get_perCoreRows() -
            (Ops::Base::CeilDiv(tilingData->get_perCoreRows(), perLoopMaxRows) - 1) * perLoopMaxRows);
    }

    tilingData->set_lastCoreRows(lastCoreNum);
    if (perLoopMaxRows >= tilingData->get_lastCoreRows()) {
        tilingData->set_lastCorePerLoopRows(tilingData->get_lastCoreRows());
        tilingData->set_lastCoreLastLoopRows(tilingData->get_lastCoreRows());
    } else {
        tilingData->set_lastCorePerLoopRows(perLoopMaxRows);
        tilingData->set_lastCoreLastLoopRows(
            tilingData->get_lastCoreRows() -
            (Ops::Base::CeilDiv(tilingData->get_lastCoreRows(), perLoopMaxRows) - 1) * perLoopMaxRows);
    }
}

void MoeInitRoutingQuantTilingBase::Tiling4GatherOutComputeSplitK()
{
    auto tilingData = &moeInitRoutingTilingData.gatherOutComputeParamsOp;
    tilingData->set_splitFlag(SPLIT_K);
    int64_t realRows = moeInitRoutingTilingData.get_n();
    activateNum = std::min(activateNum, realRows);
    if (activateNum <= 0) {
        OP_LOGW(opName, "activateNum or numRows is %ld, less than or equal to0", activateNum);
        tilingData->set_needCoreNum(0);
        return;
    }
    tilingData->set_perCoreRows(moeInitRoutingTilingData.get_n());
    tilingData->set_lastCoreRows(moeInitRoutingTilingData.get_n());
    tilingData->set_activateRows(activateNum * moeInitRoutingTilingData.get_k());
    int64_t perCoreK = Ops::Base::CeilDiv(moeInitRoutingTilingData.get_k(), aivNum);
    tilingData->set_perCoreK(perCoreK);
    int needCoreNum = Ops::Base::CeilDiv(moeInitRoutingTilingData.get_k(), perCoreK);
    tilingData->set_needCoreNum(needCoreNum);
    tilingData->set_lastCoreK(
        moeInitRoutingTilingData.get_k() - (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreK());

    int64_t maxColsOneLoop = std::min(MAX_COLS_ONE_LOOP / NUM_FOUR, moeInitRoutingTilingData.get_cols());
    int64_t kFactor = moeInitRoutingTilingData.get_k();
    int64_t perLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - kFactor * ONE_BLOCK_BYTE) /
                             (static_cast<int64_t>(sizeof(int32_t)) * kFactor +
                              (maxColsOneLoop + ONE_BLOCK_BYTE) *
                                  (inuptXDtypeSize_ + static_cast<int64_t>(sizeof(float) + sizeof(int16_t))) +
                              maxColsOneLoop * inuptXDtypeSize_ + ONE_BLOCK_BYTE);
    OP_LOGI(opName, "perLoopMaxRows is %ld", perLoopMaxRows);
    while (perLoopMaxRows <= 0) {
        OP_LOGI(opName, "perLoopMaxRows is %ld, less than 0", perLoopMaxRows);
        kFactor = kFactor / NUM_TWO;
        perLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - kFactor * ONE_BLOCK_BYTE) /
                         (static_cast<int64_t>(sizeof(int32_t)) * kFactor +
                          (maxColsOneLoop + ONE_BLOCK_BYTE) *
                              (inuptXDtypeSize_ + static_cast<int64_t>(sizeof(float) + sizeof(int16_t))) +
                          maxColsOneLoop * inuptXDtypeSize_ + ONE_BLOCK_BYTE);
    }

    tilingData->set_maxColsOneLoop(maxColsOneLoop);
    tilingData->set_perCorePerLoopK(kFactor);
    tilingData->set_perCoreLastLoopK(
        tilingData->get_perCoreK() - (Ops::Base::CeilDiv(tilingData->get_perCoreK(), kFactor) - 1) * kFactor);
    tilingData->set_lastCorePerLoopK(kFactor);
    tilingData->set_lastCoreLastLoopK(
        tilingData->get_perCoreK() - (Ops::Base::CeilDiv(tilingData->get_perCoreK(), kFactor) - 1) * kFactor);

    tilingData->set_perCorePerLoopRows(std::min(tilingData->get_perCoreRows(), perLoopMaxRows));
    tilingData->set_perCoreLastLoopRows(
        tilingData->get_perCoreRows() -
        (Ops::Base::CeilDiv(tilingData->get_perCoreRows(), tilingData->get_perCorePerLoopRows()) - 1) *
            tilingData->get_perCorePerLoopRows());
    tilingData->set_lastCorePerLoopRows(std::min(tilingData->get_lastCoreRows(), perLoopMaxRows));
    tilingData->set_lastCoreLastLoopRows(
        tilingData->get_lastCoreRows() -
        (Ops::Base::CeilDiv(tilingData->get_lastCoreRows(), tilingData->get_lastCorePerLoopRows()) - 1) *
            tilingData->get_lastCorePerLoopRows());
}

int64_t MoeInitRoutingQuantTilingBase::CalPerLoopMaxRows(int64_t maxColsOneLoop, int64_t kFactor)
{
    return (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - kFactor * ONE_BLOCK_BYTE) / 
           (static_cast<int64_t>(sizeof(int32_t)) * kFactor + (maxColsOneLoop + ONE_BLOCK_BYTE) *
           (inuptXDtypeSize_ + static_cast<int64_t>(sizeof(float) + sizeof(int16_t))) + 
           maxColsOneLoop * inuptXDtypeSize_ + ONE_BLOCK_BYTE);
}

void MoeInitRoutingQuantTilingBase::Tiling4GatherOutComputeSplitN()
{
    auto tilingData = &moeInitRoutingTilingData.gatherOutComputeParamsOp;
    tilingData->set_splitFlag(SPLIT_N);
    int64_t realRows = moeInitRoutingTilingData.get_n();
    activateNum = std::min(activateNum, realRows);
    int perCoreRows = Ops::Base::CeilDiv(realRows, aivNum);
    if (perCoreRows <= 0) {
        tilingData->set_perCoreRows(0);
        return;
    }

    int64_t maxColsOneLoop = std::min(MAX_COLS_ONE_LOOP / NUM_FOUR, moeInitRoutingTilingData.get_cols());
    int64_t kFactor = moeInitRoutingTilingData.get_k();
    int64_t perLoopMaxRows = CalPerLoopMaxRows(maxColsOneLoop, kFactor);
    OP_LOGD(opName, "perLoopMaxRows is %ld", perLoopMaxRows);
    while (perLoopMaxRows <= 0) {
        OP_LOGW(opName, "perLoopMaxRows is %ld, less than 0", perLoopMaxRows);
        kFactor = kFactor / NUM_TWO;
        perLoopMaxRows = CalPerLoopMaxRows(maxColsOneLoop, kFactor);
    }

    tilingData->set_maxColsOneLoop(maxColsOneLoop);
    tilingData->set_perCoreK(moeInitRoutingTilingData.get_k());
    tilingData->set_perCorePerLoopK(kFactor);
    tilingData->set_perCoreLastLoopK(
        moeInitRoutingTilingData.get_k() - (Ops::Base::CeilDiv(moeInitRoutingTilingData.get_k(), kFactor) - 1) * kFactor);
    tilingData->set_lastCoreK(moeInitRoutingTilingData.get_k());
    tilingData->set_lastCorePerLoopK(kFactor);
    tilingData->set_lastCoreLastLoopK(
        moeInitRoutingTilingData.get_k() - (Ops::Base::CeilDiv(moeInitRoutingTilingData.get_k(), kFactor) - 1) * kFactor);

    tilingData->set_activateRows(activateNum * moeInitRoutingTilingData.get_k());
    tilingData->set_perCoreRows(perCoreRows);
    tilingData->set_needCoreNum(Ops::Base::CeilDiv(realRows, tilingData->get_perCoreRows()));
    tilingData->set_perCorePerLoopRows(std::min(tilingData->get_perCoreRows(), perLoopMaxRows));

    tilingData->set_perCoreLastLoopRows(tilingData->get_perCoreRows() -
        (Ops::Base::CeilDiv(tilingData->get_perCoreRows(), tilingData->get_perCorePerLoopRows()) - 1) *
            tilingData->get_perCorePerLoopRows());
    tilingData->set_needCoreNum(Ops::Base::CeilDiv(realRows, tilingData->get_perCoreRows()));
    tilingData->set_lastCoreRows(realRows - (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreRows());
    tilingData->set_lastCorePerLoopRows(std::min(tilingData->get_lastCoreRows(), perLoopMaxRows));
    tilingData->set_lastCoreLastLoopRows(tilingData->get_lastCoreRows() -
        (Ops::Base::CeilDiv(tilingData->get_lastCoreRows(), tilingData->get_lastCorePerLoopRows()) - 1) *
            tilingData->get_lastCorePerLoopRows());
}

void MoeInitRoutingQuantTilingBase::Tiling4GatherOutComputeRow()
{
    auto tilingData = &moeInitRoutingTilingData.gatherOutComputeParamsOp;
    tilingData->set_splitFlag(SPLIT_ACTIVATE_ROW);
    int64_t realRows = moeInitRoutingTilingData.get_n();
    activateNum = std::min(activateNum, realRows) * moeInitRoutingTilingData.get_k();
    int perCoreRows = Ops::Base::CeilDiv(activateNum, aivNum);
    if (perCoreRows <= 0) {
        tilingData->set_perCoreRows(0);
        return;
    }

    int64_t maxColsOneLoop = std::min(MAX_COLS_ONE_LOOP / NUM_TWO, moeInitRoutingTilingData.get_cols());
    int64_t xUbSize = (maxColsOneLoop * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE +
                      (maxColsOneLoop * sizeof(float) + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE +
                      (maxColsOneLoop * sizeof(int16_t) + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE +
                      (maxColsOneLoop * sizeof(int8_t) + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    int64_t perLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - xUbSize) /
                             static_cast<int64_t>(sizeof(int32_t)) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    OP_LOGD(opName, "perLoopMaxRows is %ld", perLoopMaxRows);
    while (perLoopMaxRows <= 0) {
        OP_LOGW(opName, "perLoopMaxRows is %ld, less than 0", perLoopMaxRows);
        maxColsOneLoop = maxColsOneLoop / NUM_TWO;
        xUbSize = (maxColsOneLoop * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE +
                  (maxColsOneLoop * sizeof(float) + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE +
                  (maxColsOneLoop * sizeof(int16_t) + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE +
                  (maxColsOneLoop * sizeof(int8_t) + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        perLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - xUbSize) /
                         static_cast<int64_t>(sizeof(int32_t)) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    }

    tilingData->set_maxColsOneLoop(maxColsOneLoop);
    tilingData->set_activateRows(activateNum);
    tilingData->set_perCoreRows(perCoreRows);
    tilingData->set_needCoreNum(Ops::Base::CeilDiv(activateNum, tilingData->get_perCoreRows()));
    tilingData->set_perCorePerLoopRows(std::min(tilingData->get_perCoreRows(), perLoopMaxRows));

    tilingData->set_perCoreLastLoopRows(
        tilingData->get_perCoreRows() -
        (Ops::Base::CeilDiv(tilingData->get_perCoreRows(), tilingData->get_perCorePerLoopRows()) - 1) *
            tilingData->get_perCorePerLoopRows());
    tilingData->set_needCoreNum(Ops::Base::CeilDiv(activateNum, tilingData->get_perCoreRows()));
    tilingData->set_lastCoreRows(activateNum - (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreRows());
    tilingData->set_lastCorePerLoopRows(std::min(tilingData->get_lastCoreRows(), perLoopMaxRows));
    tilingData->set_lastCoreLastLoopRows(
        tilingData->get_lastCoreRows() -
        (Ops::Base::CeilDiv(tilingData->get_lastCoreRows(), tilingData->get_lastCorePerLoopRows()) - 1) *
            tilingData->get_lastCorePerLoopRows());
}

void MoeInitRoutingQuantTilingBase::Tiling4GatherOutCompute()
{
    if (moeInitRoutingTilingData.get_n() / NUM_TWO > activateNum) {
        tilingKey_ = tilingKey_ + 2L;
        return Tiling4GatherOutComputeRow();
    }

    if (moeInitRoutingTilingData.get_n() < aivNum &&
        moeInitRoutingTilingData.get_n() < moeInitRoutingTilingData.get_k()) {
        return Tiling4GatherOutComputeSplitK();
    }
    return Tiling4GatherOutComputeSplitN();
}

static ge::graphStatus TilingForMoeInitRoutingQuant(gert::TilingContext* context)
{
    MoeInitRoutingQuantTilingBase tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForMoeInitRoutingQuant(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeInitRoutingQuant)
    .Tiling(TilingForMoeInitRoutingQuant)
    .TilingParse<MoeInitRoutingQuantCompileInfo>(TilingPrepareForMoeInitRoutingQuant);
} // namespace optiling
