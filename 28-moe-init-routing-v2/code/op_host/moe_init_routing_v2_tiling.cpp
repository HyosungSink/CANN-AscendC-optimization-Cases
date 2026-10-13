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
 * \file moe_init_routing_v2_tiling.cpp
 * \brief
 */
#include "moe_init_routing_v2_tiling.h"
#include "../common/include/tiling_base/tiling_util.h"

namespace optiling {
const static int64_t TILING_KEY_DROPLESS_SORT_ONE_CORE = 10001;
const static int64_t TILING_KEY_DROPLESS_SORT_MULTI_CORE = 10002;
const static int64_t TILING_KEY_DROPLESS_HIST_REGBASE_SORT_ONE_CORE = 11001;
const static int64_t TILING_KEY_DROPLESS_HIST_REGBASE_SORT_MULTI_CORE = 11002;
const static int64_t TILING_KEY_DROP_PAD_MODE_SORT_ONE_CORE = 10011;
const static int64_t TILING_KEY_DROP_PAD_MODE_SORT_MULTI_CORE = 10012;
const static int64_t TILING_KEY_DROP_PAD_MODE_HIST_REGBASE_SORT_ONE_CORE = 11011;
const static int64_t TILING_KEY_DROP_PAD_MODE_HIST_REGBASE_SORT_MULTI_CORE = 11012;
const static int64_t TILING_KEY_HIGH_PERFORMANCE = 20000;
const static int64_t HIST_REGBASE_MAX_EXPERT_NUM = 256;
const static int64_t NUM_TWO = 2;
const static int64_t NUM_THREE = 3;
const static int64_t NUM_FOUR = 4;
const static int64_t MRG_LIST_NUM = 4;
const static int64_t SORT32_ALIGN_ELEMENT = 32;
const static int64_t ONE_BLOCK_BYTE = 32;
const static size_t DIM_ONE = 1;
const static size_t DIM_TWO = 2;
const static size_t DIM_THREE = 3;
const static int32_t SIZE_16 = 16;
const static int32_t LENGTH_1024 = 1024;
const static int64_t MAX_COLS_ONE_LOOP = 16376;
const static int64_t ASSIST_NUM = 256;
const static int64_t INDEX_INPUT_X = 0;
const static int64_t INDEX_INPUT_EXPERT_IDX = 1;
const static int64_t ATTR_ACTIVE_ROWS = 0;
const static int64_t ATTR_EXPERT_CAPACITY = 1;
const static int64_t ATTR_EXPERT_NUM = 2;
const static int64_t ATTR_DROP_PAD_MODE = 3;
const static int64_t ATTR_EXPERT_TOKENS_COUNT_OR_CUMSUM_FLAG = 4;
const static int64_t ATTR_EXPERT_TOKENS_BEFORE_CAPACITY_FLAG = 5;
const static int64_t OUTOUT_EXPANDED_X = 0;
const static int64_t OUTOUT_EXPANDED_ROW_IDX = 1;
const static int64_t OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM = 2;
const static int64_t OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY = 3;
const static int64_t ONE_CORE_SORT_BUFFER = 6;
const static int64_t REGIONP_ROPOSAL_SIZE = 8;
const static int64_t SYNC_WORKSPACE = 8192;
const static int64_t EXPERT_TOKENS_COUNT = 2;
const static int64_t SIMT_UB_SIZE_BYTE = 40960;

#define CHECK_FAIL(context, cond, ...)                                                                                 \
    do {                                                                                                               \
        if (cond) {                                                                                                    \
            OP_LOGE(context->GetNodeName(), ##__VA_ARGS__);                                                          \
            return ge::GRAPH_FAILED;                                                                                   \
        }                                                                                                              \
    } while (0)

inline static int64_t CeilLog4(int64_t x)
{
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
}

inline static int64_t GetPerOrLastValue(int64_t x, int64_t y)
{
    if (y == 0) {
        return 0;
    }
    return x <= y ? x : x % y;
}

void MoeInitRoutingV2TilingBase::Reset()
{
    opName = nullptr;
    return;
}

ge::graphStatus MoeInitRoutingV2TilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_->GetNodeName(), "fail to get platform info."),
              return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aivNum = ascendcPlatform.GetCoreNumAiv();
    aicoreParams_.numBlocks = aivNum;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm - SIMT_UB_SIZE_BYTE;
    mrgSortListMaxElement = 2048; // DAVID下，单词搬运最大元素个数2048
    regBase = false;
    if (!regBase) {
        aicoreParams_.ubSize = ubSizePlatForm;
        mrgSortListMaxElement = 2040; // 单词搬运最大元素个数2040
    }
    moeInitRoutingTilingData.set_coreNum(aivNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2TilingBase::CheckTokenCount(int64_t num, const char *tag)
{
    auto expertTokensShapePtr = context_->GetOutputShape(num);
    OP_CHECK_IF(expertTokensShapePtr == nullptr, OP_LOGE(context_->GetNodeName(), "%s is nullptr!", tag), return ge::GRAPH_FAILED);

    auto expertTokensDesc = context_->GetOutputDesc(num);
    OP_CHECK_IF(expertTokensDesc == nullptr, OP_LOGE(context_->GetNodeName(), "%s is nullptr!", tag), return ge::GRAPH_FAILED);
    auto dt = expertTokensDesc->GetDataType();
    CHECK_FAIL(context_, dt != ge::DT_INT32, "The data type of %s should be int32.", tag);

    const gert::Shape expertTokensShape = expertTokensShapePtr->GetStorageShape();
    size_t expertTokensDimNum = expertTokensShape.GetDimNum();
    CHECK_FAIL(context_, expertTokensDimNum != DIM_ONE, "The dim number of %s should be 1.", tag);
    CHECK_FAIL(context_, expertTokensShape.GetDim(0) != expertNum, "The first dim of %s should be %ld.", tag,
               expertNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2TilingBase::CheckOutShape(bool isRegbase)
{
    auto expandedXShapePtr = context_->GetOutputShape(OUTOUT_EXPANDED_X); // 获取输出shape
    OP_CHECK_IF(expandedXShapePtr == nullptr, OP_LOGE(context_->GetNodeName(), "expandedX is nullptr!"), return ge::GRAPH_FAILED);

    const gert::Shape expandedXShape = expandedXShapePtr->GetStorageShape();

    auto expandedRowIdxShapePtr = context_->GetOutputShape(OUTOUT_EXPANDED_ROW_IDX);
    OP_CHECK_IF(expandedRowIdxShapePtr == nullptr, OP_LOGE(context_->GetNodeName(), "expandedRowIdx is nullptr!"),
               return ge::GRAPH_FAILED);
    const gert::Shape expandedRowIdxShape = expandedRowIdxShapePtr->GetStorageShape();

    size_t expandedXDimNum = expandedXShape.GetDimNum();
    if (dropPadMode > 0) {
        CHECK_FAIL(context_, expandedXDimNum != DIM_THREE, "The dim number of expandedX should be 3.");
        CHECK_FAIL(context_, expandedXShape.GetDim(0) != expertNum, "The first dim of expandedX should be %ld.",
                   expertNum);
        CHECK_FAIL(context_, expandedXShape.GetDim(1) != expertCapacity, "The second dim of expandedX should be %ld.",
                   expertCapacity);
        CHECK_FAIL(context_, expandedXShape.GetDim(NUM_TWO) != moeInitRoutingTilingData.get_cols(),
                   "The third dim of expandedX should be %ld.", moeInitRoutingTilingData.get_cols());
    } else {
        CHECK_FAIL(context_, expandedXDimNum != DIM_TWO, "The dim number of expandedX should be 2.");
        int64_t firstDim = moeInitRoutingTilingData.get_n() * moeInitRoutingTilingData.get_k();
        if (!isRegbase) {
            activateNum = activateNum == 0 ? firstDim : std::min(firstDim, activateNum);
        } else {
            activateNum = (activateNum <= 0) ? activateNum + totalLength : activateNum;
            activateNum = std::min(activateNum, firstDim);
            activateNum = std::max(activateNum, static_cast<int64_t>(0));
        }
        CHECK_FAIL(context_, expandedXShape.GetDim(0) != activateNum, "The first dim of expandedX should be %ld.",
                   activateNum);
        CHECK_FAIL(context_, expandedXShape.GetDim(1) != moeInitRoutingTilingData.get_cols(),
                   "The second dim of expandedX should be %ld.", moeInitRoutingTilingData.get_cols());
    }

    size_t expandedRowIdxDimNum = expandedRowIdxShape.GetDimNum();
    CHECK_FAIL(context_, expandedRowIdxDimNum != DIM_ONE, "The dim number of expandedRowIdx should be 1.");
    CHECK_FAIL(context_, expandedRowIdxShape.GetDim(0) != totalLength, "The first dim of expandedRowIdx should be %ld.",
               totalLength);

    if (dropPadMode == 0 && expertTokensCountOrCumsumFlag != 0) {
        if (CheckTokenCount(OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM, "expertTokensCountOrCumsum") == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    }

    if (dropPadMode == 1 && expertTokensBeforeCapacityFlag) {
        if (CheckTokenCount(OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY, "expertTokensBeforeCapacity") == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2TilingBase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();

    // 获取输入shape
    auto xShapePtr = context_->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_IF(xShapePtr == nullptr, OP_LOGE(context_->GetNodeName(), "x is nullptr!"), return ge::GRAPH_FAILED);
    const gert::Shape xShape = xShapePtr->GetStorageShape();
    auto expertIdxShapePtr = context_->GetInputShape(INDEX_INPUT_EXPERT_IDX);
    OP_CHECK_IF(expertIdxShapePtr == nullptr, OP_LOGE(context_->GetNodeName(), "expertIdx is nullptr!"), return ge::GRAPH_FAILED);
    const gert::Shape expertIdxShape = expertIdxShapePtr->GetStorageShape();

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t *activateNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVE_ROWS);
    if (activateNumPtr != nullptr) {
        activateNum = *activateNumPtr;
    }
    const int64_t *expertCapacityPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_CAPACITY);
    if (expertCapacityPtr != nullptr) {
        expertCapacity = *expertCapacityPtr;
    }
    const int64_t *expertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_NUM);
    if (expertNumPtr != nullptr) {
        expertNum = *expertNumPtr;
    }
    const int64_t *dropPadModePtr = attrs->GetAttrPointer<int64_t>(ATTR_DROP_PAD_MODE);
    if (dropPadModePtr != nullptr) {
        dropPadMode = *dropPadModePtr;
    }
    const int64_t *expertTokensCountOrCumsumFlagPtr =
        attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_TOKENS_COUNT_OR_CUMSUM_FLAG);
    if (expertTokensCountOrCumsumFlagPtr != nullptr) {
        expertTokensCountOrCumsumFlag = *expertTokensCountOrCumsumFlagPtr;
    }
    const bool *expertTokensBeforeCapacityFlagPtr =
        attrs->GetAttrPointer<bool>(ATTR_EXPERT_TOKENS_BEFORE_CAPACITY_FLAG);
    if (expertTokensBeforeCapacityFlagPtr != nullptr) {
        expertTokensBeforeCapacityFlag = *expertTokensBeforeCapacityFlagPtr;
    }

    // 参数校验
    size_t xDimNnum = xShape.GetDimNum();
    CHECK_FAIL(context_, xDimNnum != DIM_TWO, "The dim number of x should be 2.");
    CHECK_FAIL(context_, xShape.GetDim(0) != expertIdxShape.GetDim(0),
               "The first dim of x and expertIdx should be equal.");
    CHECK_FAIL(context_, xShape.GetDim(0) < 0, "The first dim of x cannot be less than 0.");
    CHECK_FAIL(context_, xShape.GetDim(1) < 0, "The second dim of x cannot be less than 0.");
    CHECK_FAIL(context_, expertIdxShape.GetDim(0) < 0, "The first dim of expertIdx cannot be less than 0.");
    CHECK_FAIL(context_, expertCapacity < 0, "The expertCapacity cannot be less than 0.");
    CHECK_FAIL(context_, expertNum < 0, "The expertNum cannot be less than 0.");
    CHECK_FAIL(context_, dropPadMode < 0 || dropPadMode > 1, "The dropPadMode should be 0 or 1.");
    CHECK_FAIL(context_, dropPadMode > 0 && (expertCapacity < 1 || expertNum < 1),
               "The expertCapacity and expertNum should be greater than 0 when dropPadMode is 1");
    CHECK_FAIL(context_, expertTokensCountOrCumsumFlag < 0 || expertTokensCountOrCumsumFlag > EXPERT_TOKENS_COUNT,
               "The expertTokensCountOrCumsumFlag should be 0, 1 or 2.");
    CHECK_FAIL(context_, expertTokensCountOrCumsumFlag > 0 && expertNum <= 0,
               "The expertNum should be greater than 0 when expertTokensCountOrCumsumFlag is greater than 0");
    CHECK_FAIL(context_, dropPadMode > 0 && expertCapacity > xShape.GetDim(0),
               "The first dim of x cannot be less than expertCapacity");
    if (dropPadMode == 1) {
        // droppad场景下不输出expertTokensCountOrCumsum
        expertTokensCountOrCumsumFlag = 0;
    } else {
        // dropless场景下不输出expertTokensBeforeCapacity
        expertTokensBeforeCapacityFlag = false;
    }
    size_t expertIdxDimNum = expertIdxShape.GetDimNum();
    bool isRegbase = false;
    if (isRegbase) {
        CHECK_FAIL(context_, expertIdxDimNum != DIM_ONE && expertIdxDimNum != DIM_TWO,
                   "The dim number of expertIdx should be 1 or 2.");
        if (expertIdxDimNum == DIM_TWO) {
            CHECK_FAIL(context_, expertIdxShape.GetDim(1) < 0, "The second dim of expertIdx cannot be less than 0.");
        }
        moeInitRoutingTilingData.set_k(expertIdxDimNum == DIM_TWO ? expertIdxShape.GetDim(1) : 1);
    } else {
        CHECK_FAIL(context_, activateNum < 0, "The activeNum cannot be less than 0.");
        CHECK_FAIL(context_, expertIdxDimNum != DIM_TWO, "The dim number of expertIdx should be 2.");
        CHECK_FAIL(context_, expertIdxShape.GetDim(1) < 0, "The second dim of expertIdx cannot be less than 0.");
        moeInitRoutingTilingData.set_k(expertIdxShape.GetDim(1));
    }
    moeInitRoutingTilingData.set_cols(xShape.GetDim(1));
    moeInitRoutingTilingData.set_n(expertIdxShape.GetDim(0));
    moeInitRoutingTilingData.set_expertCapacity(expertCapacity);
    moeInitRoutingTilingData.set_expertNum(expertNum);
    moeInitRoutingTilingData.set_dropPadMode(dropPadMode);
    moeInitRoutingTilingData.set_expertTokensCountOrCumsumFlag(expertTokensCountOrCumsumFlag);
    moeInitRoutingTilingData.set_expertTokensBeforeCapacityFlag(expertTokensBeforeCapacityFlag);
    totalLength = moeInitRoutingTilingData.get_n() * moeInitRoutingTilingData.get_k();

    auto ret = CheckOutShape(isRegbase);
    inuptXDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(context_->GetInputDesc(0)->GetDataType()));
    return ret;
}

void MoeInitRoutingV2TilingBase::ShowTilingData()
{
    OP_LOGI(opName,
              "moeInitRoutingTilingData is coreNum:%ld, n:%ld, cols:%ld, k:%ld, expertCapacity:%ld, expertNum:%ld, "
              "dropPadMode:%ld, expertTokensCountOrCumsumFlag:%ld, expertTokensBeforeCapacityFlag:%ld",
              moeInitRoutingTilingData.get_coreNum(), moeInitRoutingTilingData.get_n(),
              moeInitRoutingTilingData.get_cols(), moeInitRoutingTilingData.get_k(),
              moeInitRoutingTilingData.get_expertCapacity(), moeInitRoutingTilingData.get_expertNum(),
              moeInitRoutingTilingData.get_dropPadMode(), moeInitRoutingTilingData.get_expertTokensCountOrCumsumFlag(),
              moeInitRoutingTilingData.get_expertTokensBeforeCapacityFlag());
    OP_LOGI(opName,
              "MoeV2VBSComputeTilingData is needCoreNum:%ld, perCoreElements:%ld, perCoreLoops:%ld, "
              "perCorePerLoopElements:%ld, "
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
    OP_LOGI(opName, "VMSMiddleComputeTilingData is needCoreNum:%ld",
              moeInitRoutingTilingData.vmsMiddleComputeParamsOp.get_needCoreNum());
    OP_LOGI(opName, "SortOutComputeTilingData is oneLoopMaxElements:%ld",
              moeInitRoutingTilingData.sortOutComputeParamsOp.get_oneLoopMaxElements());
    OP_LOGI(
        opName,
        "SrcToDstComputeTilingData is needCoreNum:%ld, activateRows:%ld, perCoreRows:%ld, perCorePerLoopRows:%ld, "
        "perCoreLastLoopRows:%ld, lastCoreRows:%ld, lastCorePerLoopRows:%ld, lastCoreLastLoopRows:%ld,",
        moeInitRoutingTilingData.srcToDstComputeParamsOp.get_needCoreNum(),
        moeInitRoutingTilingData.srcToDstComputeParamsOp.get_activateRows(),
        moeInitRoutingTilingData.srcToDstComputeParamsOp.get_perCoreRows(),
        moeInitRoutingTilingData.srcToDstComputeParamsOp.get_perCorePerLoopRows(),
        moeInitRoutingTilingData.srcToDstComputeParamsOp.get_perCoreLastLoopRows(),
        moeInitRoutingTilingData.srcToDstComputeParamsOp.get_lastCoreRows(),
        moeInitRoutingTilingData.srcToDstComputeParamsOp.get_lastCorePerLoopRows(),
        moeInitRoutingTilingData.srcToDstComputeParamsOp.get_lastCoreLastLoopRows());
    OP_LOGI(opName,
              "SrcToDstComputeCapacityTilingData is needCoreNum:%ld, perCoreRows:%ld, perCorePerLoopRows:%ld, "
              "perCoreLastLoopRows:%ld, lastCoreRows:%ld, lastCorePerLoopRows:%ld, lastCoreLastLoopRows:%ld,",
              moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_needCoreNum(),
              moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_perCoreRows(),
              moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_perCorePerLoopRows(),
              moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_perCoreLastLoopRows(),
              moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_lastCoreRows(),
              moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_lastCorePerLoopRows(),
              moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_lastCoreLastLoopRows());
    OP_LOGI(
        opName,
        "GatherOutComputeTilingData is needCoreNum:%ld, activateRows:%ld, perCoreRows:%ld, perCorePerLoopRows:%ld, "
        "perCoreLastLoopRows:%ld, lastCoreRows:%ld, lastCorePerLoopRows:%ld, lastCoreLastLoopRows:%ld,",
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_needCoreNum(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_activateRows(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCoreRows(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCorePerLoopRows(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCoreLastLoopRows(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCoreRows(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCorePerLoopRows(),
        moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCoreLastLoopRows());
}

ge::graphStatus MoeInitRoutingV2TilingBase::DoOpTiling()
{

    if (!regBase) {
        // NUM_TWO sort value and indices
        // NUM_FOUR sort need space
        // SORT32_ALIGN_ELEMENT 32Bytes aligned
        auto oneEleSizeFactor =
            sizeof(int32_t) * NUM_TWO * NUM_FOUR;
        sortLoopMaxElement = (aicoreParams_.ubSize) / oneEleSizeFactor / SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
        mrgSortListMaxElement = mrgSortListMaxElement;
        isFullLoad = IsFullLoad();
    } else {
        auto expertIdx = context_->GetInputDesc(INDEX_INPUT_EXPERT_IDX);
        OP_CHECK_IF(expertIdx == nullptr, OP_LOGE(context_->GetNodeName(), "expertIdx is nullptr!"), return ge::GRAPH_FAILED);
        bool isIdxLong = expertIdx->GetDataType() == ge::DataType::DT_INT64;
        sortLoopMaxElement = (aicoreParams_.ubSize) / (sizeof(int32_t) * NUM_TWO * NUM_FOUR + isIdxLong) /
                             SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
    }
    Tiling4VBSCompute();
    Tiling4VMSMiddleCompute();
    Tiling4SortOutCompute();
    Tiling4SrcToDstCompute();
    Tiling4SrcToDstCapacityCompute();
    Tiling4GatherOutCompute();
    ShowTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2TilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeInitRoutingV2TilingBase::GetTilingKey() const
{
    if (isFullLoad) {
        return TILING_KEY_HIGH_PERFORMANCE;
    }
    context_->SetScheduleMode(1);

    bool histWithRegBase = regBase && expertNum <= HIST_REGBASE_MAX_EXPERT_NUM;
    if (dropPadMode == 0) {
        if (totalLength <= sortLoopMaxElement) { // 排序只用到一个核排序
            return histWithRegBase ? TILING_KEY_DROPLESS_HIST_REGBASE_SORT_ONE_CORE : TILING_KEY_DROPLESS_SORT_ONE_CORE;
        } else {
            return histWithRegBase ? TILING_KEY_DROPLESS_HIST_REGBASE_SORT_MULTI_CORE :
                                     TILING_KEY_DROPLESS_SORT_MULTI_CORE;
        }
    } else {
        if (totalLength <= sortLoopMaxElement) {
            return histWithRegBase ? TILING_KEY_DROP_PAD_MODE_HIST_REGBASE_SORT_ONE_CORE :
                                     TILING_KEY_DROP_PAD_MODE_SORT_ONE_CORE;
        } else {
            return histWithRegBase ? TILING_KEY_DROP_PAD_MODE_HIST_REGBASE_SORT_MULTI_CORE :
                                     TILING_KEY_DROP_PAD_MODE_SORT_MULTI_CORE;
        }
    }
    return tilingKey_;
}

ge::graphStatus MoeInitRoutingV2TilingBase::GetWorkspaceSize()
{
    // 计算workspace大小
    size_t sortWorkspaceSize;
    size_t expertTokenFlagSize = static_cast<size_t>(aivNum) * 2U * sizeof(int32_t);
    size_t scatterWorkspaceSize = static_cast<size_t>(totalLength * sizeof(int32_t) * NUM_TWO);
    sortWorkspaceSize =
        static_cast<size_t>(totalLength * sizeof(float) * NUM_TWO * NUM_THREE);
    if (regBase) {
        size_t expertNumSize = expertNum * sizeof(int32_t);
        workspaceSize_ = sortWorkspaceSize + scatterWorkspaceSize + expertTokenFlagSize + expertNumSize +
                         SIZE_16 * LENGTH_1024 * LENGTH_1024;
    } else {
        workspaceSize_ =
            sortWorkspaceSize + scatterWorkspaceSize + expertTokenFlagSize + SIZE_16 * LENGTH_1024 * LENGTH_1024;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV2TilingBase::PostTiling()
{
    context_->SetBlockDim(aivNum);
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    if (regBase) {
        context_->SetLocalMemorySize(aicoreParams_.ubSize);
    }
    moeInitRoutingTilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                          context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(moeInitRoutingTilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MoeInitRoutingV2TilingBase::Tiling4VBSOneCoreCompute(MoeV2VBSComputeTilingData *tilingData)
{
    uint64_t needCoreNum = totalLength <= 0 ? 0 : 1;
    tilingData->set_needCoreNum(needCoreNum);
    tilingData->set_perCoreElements(totalLength);
    tilingData->set_perCoreLoops(1);
    tilingData->set_perCorePerLoopElements(tilingData->get_perCoreElements());
    tilingData->set_perCoreLastLoopElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreLoops(1);
    tilingData->set_lastCorePerLoopElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreLastLoopElements(tilingData->get_perCoreElements());
}

void MoeInitRoutingV2TilingBase::Tiling4VBSMultiCoreCompute(MoeV2VBSComputeTilingData *tilingData)
{
    int64_t needCoreNum = CeilDiv(totalLength, sortLoopMaxElement);         // 向上取整
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum))); // 用到多核时，核数最多是4^x
    needCoreNum = std::min(needCoreNum, aivNum);                            // 不能超过物理核数
    if (needCoreNum == 0) {
        return;
    }
    int64_t perCoreElements = totalLength / needCoreNum; // 每个核处理的元素数
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = CeilDiv(totalLength, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    tilingData->set_needCoreNum(needCoreNum);
    do {
        tilingData->set_perCoreElements(perCoreElements);
        tilingData->set_perCoreLoops(
            CeilDiv(tilingData->get_perCoreElements(), sortLoopMaxElement)); // 每个核处理的loop数
        tilingData->set_perCorePerLoopElements(std::min(tilingData->get_perCoreElements(), sortLoopMaxElement));

        tilingData->set_perCoreLastLoopElements(tilingData->get_perCoreElements() -
                                                (tilingData->get_perCoreLoops() - 1) *
                                                    tilingData->get_perCorePerLoopElements());

        tilingData->set_lastCoreElements(totalLength -
                                         (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreElements());
        tilingData->set_lastCoreLoops(tilingData->get_perCoreLoops());
        int64_t lastCorePerLoopElements =
            CeilDiv(CeilDiv(tilingData->get_lastCoreElements(), tilingData->get_lastCoreLoops()),
                    SORT32_ALIGN_ELEMENT) *
            SORT32_ALIGN_ELEMENT;
        tilingData->set_lastCorePerLoopElements(lastCorePerLoopElements);
        tilingData->set_lastCoreLastLoopElements(tilingData->get_lastCoreElements() -
                                                 (tilingData->get_lastCoreLoops() - 1) *
                                                     tilingData->get_lastCorePerLoopElements());
        perCoreElements -= SORT32_ALIGN_ELEMENT;
    } while (tilingData->get_lastCoreLastLoopElements() <= 0 && perCoreElements > 0);
    OP_CHECK_IF(
        tilingData->get_lastCoreLastLoopElements() <= 0, OP_LOGE(context_->GetNodeName(), "vbs tiling failed."), ;);
}

void MoeInitRoutingV2TilingBase::Tiling4VBSCompute()
{
    auto tilingData = &moeInitRoutingTilingData.vbsComputeParamsOp;
    tilingData->set_oneLoopMaxElements(sortLoopMaxElement);
    if (totalLength <= sortLoopMaxElement) { // 只用到一个核
        Tiling4VBSOneCoreCompute(tilingData);
        return;
    }
    Tiling4VBSMultiCoreCompute(tilingData);
}

void MoeInitRoutingV2TilingBase::Tiling4VMSMiddleCompute()
{
    auto vbsComputeTilingData = &moeInitRoutingTilingData.vbsComputeParamsOp;
    auto tilingData = &moeInitRoutingTilingData.vmsMiddleComputeParamsOp;
    if (vbsComputeTilingData->get_needCoreNum() <= MRG_LIST_NUM) { // 队列数小于一次vms则没有中间归并
        tilingData->set_needCoreNum(0);                            // 需要的核数
        return;
    }
    int64_t needCoreNum = CeilDiv(vbsComputeTilingData->get_needCoreNum(), MRG_LIST_NUM);
    tilingData->set_needCoreNum(needCoreNum); // 需要的核数
}

void MoeInitRoutingV2TilingBase::Tiling4SortOutCompute()
{
    auto tilingData = &moeInitRoutingTilingData.sortOutComputeParamsOp;
    tilingData->set_oneLoopMaxElements(mrgSortListMaxElement);
}

void MoeInitRoutingV2TilingBase::Tiling4SrcToDstCompute()
{
    auto tilingData = &moeInitRoutingTilingData.srcToDstComputeParamsOp;

    int64_t useCore = aivNum;
    int64_t perLoopMaxRows = (aicoreParams_.ubSize -
                              ASSIST_NUM * sizeof(float) - useCore * SORT32_ALIGN_ELEMENT) /
                             (SORT32_ALIGN_ELEMENT * NUM_TWO) / NUM_TWO;
    int64_t perCoreRows = CeilDiv(totalLength, useCore);
    if (perCoreRows <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    int64_t needCoreNum = CeilDiv(totalLength, perCoreRows);
    tilingData->set_needCoreNum(needCoreNum);
    int64_t lastCoreNum = totalLength - perCoreRows * (tilingData->get_needCoreNum() - 1);

    tilingData->set_perCoreRows(perCoreRows);

    if (perLoopMaxRows >= tilingData->get_perCoreRows()) { // 一个loop结束
        tilingData->set_perCorePerLoopRows(tilingData->get_perCoreRows());
        tilingData->set_perCoreLastLoopRows(tilingData->get_perCoreRows());
    } else {
        tilingData->set_perCorePerLoopRows(perLoopMaxRows);
        tilingData->set_perCoreLastLoopRows(tilingData->get_perCoreRows() -
                                            (CeilDiv(tilingData->get_perCoreRows(), perLoopMaxRows) - 1) *
                                                perLoopMaxRows);
    }

    tilingData->set_lastCoreRows(lastCoreNum);
    if (perLoopMaxRows >= tilingData->get_lastCoreRows()) {
        tilingData->set_lastCorePerLoopRows(tilingData->get_lastCoreRows());
        tilingData->set_lastCoreLastLoopRows(tilingData->get_lastCoreRows());
    } else {
        tilingData->set_lastCorePerLoopRows(perLoopMaxRows);
        tilingData->set_lastCoreLastLoopRows(tilingData->get_lastCoreRows() -
                                             (CeilDiv(tilingData->get_lastCoreRows(), perLoopMaxRows) - 1) *
                                                 perLoopMaxRows);
    }
}

void MoeInitRoutingV2TilingBase::Tiling4SrcToDstCapacityCompute()
{
    auto tilingData = &moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp;

    int64_t perCoreRows = CeilDiv(totalLength, aivNum);
    if (perCoreRows <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    int64_t needCoreNum = CeilDiv(totalLength, perCoreRows);
    tilingData->set_needCoreNum(needCoreNum);
    int64_t cols = moeInitRoutingTilingData.get_cols();
    tilingData->set_perCoreRows(perCoreRows);
    int64_t lastCoreRows = totalLength - perCoreRows * (needCoreNum - 1);
    tilingData->set_lastCoreRows(lastCoreRows);

    int64_t rowSize =
        (perCoreRows * sizeof(int32_t) * 2 + ONE_BLOCK_BYTE + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    int64_t colSize = (cols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;

    if (rowSize + colSize < static_cast<int64_t>(aicoreParams_.ubSize)) {
        tilingData->set_perCorePerLoopRows(perCoreRows);
        tilingData->set_perCoreLastLoopRows(perCoreRows);
        tilingData->set_lastCorePerLoopRows(lastCoreRows);
        tilingData->set_lastCoreLastLoopRows(lastCoreRows);
        tilingData->set_perCoreLoops(1);
        tilingData->set_lastCoreLoops(1);
        tilingData->set_perLoopCols(cols);
        tilingData->set_lastLoopCols(cols);
        tilingData->set_colLoops(1);
    } else {
        int64_t baseMaxCols = MAX_COLS_ONE_LOOP;
        int64_t baseMaxColsSize =
            (baseMaxCols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        int64_t basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) - baseMaxColsSize - ONE_BLOCK_BYTE) /
                                     static_cast<int64_t>(sizeof(int32_t)) / NUM_TWO / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        if (cols < MAX_COLS_ONE_LOOP) {
            basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) - colSize - ONE_BLOCK_BYTE) /
                                 static_cast<int64_t>(sizeof(int32_t)) / NUM_TWO / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        } else if (perCoreRows < basePerLoopMaxRows) {
            baseMaxCols = (static_cast<int64_t>(aicoreParams_.ubSize) - rowSize) / inuptXDtypeSize_ / ONE_BLOCK_BYTE *
                          ONE_BLOCK_BYTE;
        }
        tilingData->set_perLoopCols(std::min(baseMaxCols, cols));
        tilingData->set_lastLoopCols(GetPerOrLastValue(cols, baseMaxCols));
        tilingData->set_colLoops((cols + baseMaxCols - 1) / baseMaxCols);

        tilingData->set_perCorePerLoopRows(std::min(perCoreRows, basePerLoopMaxRows));
        tilingData->set_perCoreLastLoopRows(GetPerOrLastValue(perCoreRows, basePerLoopMaxRows));
        tilingData->set_perCoreLoops((perCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);

        tilingData->set_lastCorePerLoopRows(std::min(lastCoreRows, basePerLoopMaxRows));
        tilingData->set_lastCoreLastLoopRows(GetPerOrLastValue(lastCoreRows, basePerLoopMaxRows));
        tilingData->set_lastCoreLoops((lastCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
    }
}

void MoeInitRoutingV2TilingBase::Tiling4GatherOutCompute()
{
    auto tilingData = &moeInitRoutingTilingData.gatherOutComputeParamsOp;
    tilingData->set_activateRows(totalLength);
    if (dropPadMode == 0) {
        tilingData->set_activateRows(activateNum);
    }
    int64_t perCoreRows = CeilDiv(totalLength, aivNum);
    if (perCoreRows <= 0 || moeInitRoutingTilingData.get_cols() <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    tilingData->set_needCoreNum(CeilDiv(totalLength, perCoreRows));
    int64_t cols = moeInitRoutingTilingData.get_cols();
    tilingData->set_perCoreRows(perCoreRows);
    int64_t lastCoreRows = totalLength - perCoreRows * (tilingData->get_needCoreNum() - 1);
    tilingData->set_lastCoreRows(lastCoreRows);

    int64_t rowSize = (perCoreRows * sizeof(int32_t) + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    int64_t colSize = (cols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;

    int64_t ubSize = static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO;
    if (rowSize + colSize < ubSize) {
        tilingData->set_perCorePerLoopRows(perCoreRows);
        tilingData->set_perCoreLastLoopRows(perCoreRows);
        tilingData->set_lastCorePerLoopRows(lastCoreRows);
        tilingData->set_lastCoreLastLoopRows(lastCoreRows);
        tilingData->set_perCoreLoops(1);
        tilingData->set_lastCoreLoops(1);
        int64_t loopCols = regBase && (rowSize + colSize * NUM_TWO < ubSize) ? cols * NUM_TWO : cols;
        tilingData->set_perLoopCols(loopCols);
        tilingData->set_lastLoopCols(loopCols);
        tilingData->set_colLoops(1);
    } else {
        int64_t baseMaxCols = MAX_COLS_ONE_LOOP;
        int64_t baseMaxColsSize =
            (baseMaxCols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        int64_t basePerLoopMaxRows = (ubSize - baseMaxColsSize) / sizeof(int32_t) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        if (cols < MAX_COLS_ONE_LOOP) {
            basePerLoopMaxRows = (ubSize - colSize) / sizeof(int32_t) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        } else if (perCoreRows < basePerLoopMaxRows) {
            baseMaxCols = (ubSize - rowSize) / inuptXDtypeSize_ / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        }
        tilingData->set_perLoopCols(std::min(baseMaxCols, cols));
        tilingData->set_lastLoopCols(GetPerOrLastValue(cols, baseMaxCols));
        tilingData->set_colLoops((cols + baseMaxCols - 1) / baseMaxCols);

        tilingData->set_perCorePerLoopRows(std::min(perCoreRows, basePerLoopMaxRows));
        tilingData->set_perCoreLastLoopRows(GetPerOrLastValue(perCoreRows, basePerLoopMaxRows));
        tilingData->set_perCoreLoops((perCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);

        tilingData->set_lastCorePerLoopRows(std::min(lastCoreRows, basePerLoopMaxRows));
        tilingData->set_lastCoreLastLoopRows(GetPerOrLastValue(lastCoreRows, basePerLoopMaxRows));
        tilingData->set_lastCoreLoops((lastCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
    }
}

bool MoeInitRoutingV2TilingBase::IsFullLoad()
{
    if (totalLength > sortLoopMaxElement || moeInitRoutingTilingData.get_cols() > MAX_COLS_ONE_LOOP ||
        this->dropPadMode == 1) {
        return false;
    }
    int64_t sortBufferNum = ONE_CORE_SORT_BUFFER;

    int64_t sortSpace =
        CeilDiv(this->totalLength, SORT32_ALIGN_ELEMENT) * SORT32_ALIGN_ELEMENT * sizeof(int32_t) * sortBufferNum;
    int64_t otherSpace =
        CeilDiv(this->totalLength, SORT32_ALIGN_ELEMENT) * SORT32_ALIGN_ELEMENT * sizeof(int32_t) * NUM_THREE;
    int64_t expertSpace = CeilDiv(this->expertNum * int64_t(sizeof(int32_t)), ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE;
    int64_t perCoreXRows = moeInitRoutingTilingData.get_n() / aivNum;
    int64_t remainder = moeInitRoutingTilingData.get_n() % aivNum;
    // NUM_TWO is Max xRows need add 2 becauseof the left and right row may be another row.
    perCoreXRows = remainder <= 1 ? perCoreXRows + 1 : perCoreXRows + NUM_TWO;
    int64_t gatherSpace =
        CeilDiv(moeInitRoutingTilingData.get_cols() * inuptXDtypeSize_, ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE * perCoreXRows;
    int64_t remainUbAfterSort = aicoreParams_.ubSize - sortSpace - otherSpace - expertSpace - gatherSpace;
    return remainUbAfterSort > 0;
}

ASCENDC_EXTERN_C ge::graphStatus TilingForMoeInitRoutingV2(gert::TilingContext *context)
{
    MoeInitRoutingV2TilingBase tiling(context);
    return tiling.DoTiling();
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForMoeInitRoutingV2(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForMoeInitRoutingV2.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeInitRoutingV2)
    .Tiling(TilingForMoeInitRoutingV2)
    .TilingParse<MoeInitRoutingV2CompileInfo>(TilingPrepareForMoeInitRoutingV2);

} // namespace optiling
