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
 * \file group_norm_silu_tiling.cpp
 * \brief
 */
#include "group_norm_silu_tiling.h"
#include "../common/inc/op_host/tiling_util.h"

namespace optiling {
using namespace Ops::NN::OpTiling;
static const uint64_t INPUT_IDX_X = 0;
static const uint64_t INPUT_IDX_GAMMA = 1;
static const uint64_t INPUT_IDX_BETA = 2;
static const uint64_t X_SHAPE_MIN_LEN = 2;
static const uint64_t INDEX_NUM_GROUPS = 0;
static const uint64_t INDEX_EPSILON = 1;
static const uint64_t INDEX_ACTIVATE_SILU = 2;
static const uint64_t DIM_0 = 0;
static const uint64_t DIM_1 = 1;
static const uint64_t DEFAULT_PROCESSSIZE = 8192;
static const uint64_t DEFAULT_PROCESSSIZE_KIRINX90 = 5120;
static const uint64_t DEFAULT_NUMGROUPS = 32;
static const uint64_t RESERVED_WORKSPACE_SIZE_910B = static_cast<uint64_t>(16 * 1024 * 1024);
static const uint64_t FLOAT32_BYTES = 4;
static const uint64_t FLOAT16_BYTES = 2;
static const uint64_t BLOCK_SIZE = 32;
static const int64_t HW_CAP = 4096;
static const int64_t UPPER_LIMIT_TWO = 4000;
static const int64_t UPPER_LIMIT_ONE = 2700;
static const uint64_t FLOAT_EIGHT = 8;
static const uint64_t FLOAT_DOUBLE_EIGHT = 16;
static const uint64_t GAMMA_BETA_UB_NUM = 6;
static const uint64_t RESERVED_BLOCK_NUM = 2;
static const uint64_t INPUT_OUTPUT_UB_NUM = 20;

inline static ge::graphStatus GroupNormSiluSetTilingData(
    gert::TilingContext* context, GroupNormSiluTilingData& tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

inline static int64_t CeilDiv(int64_t value, int64_t factor)
{
    if (factor == 0) {
        return value;
    } else if (value % factor == 0) {
        return value / factor;
    } else {
        return value / factor + 1;
    }
}

inline static int64_t Gcd(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return Gcd(b, a % b);
}

inline static int64_t Lcm(int64_t a, int64_t b)
{
    return a * b / Gcd(a, b);
}

static ge::graphStatus CheckInputParams(const gert::TilingContext* context)
{
    // check x
    auto inputX = context->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto xDtype = inputX->GetDataType();
    uint64_t xDtypeSize = ge::GetSizeByDataType(xDtype);
    OP_CHECK_IF(
        (xDtypeSize <= 0),
        OP_LOGE(context->GetNodeType(), "xDtypeSize is invalid %lu, please check.", xDtypeSize),
        return ge::GRAPH_FAILED);
    auto xShapePtr = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    uint64_t xDims = xShape.GetDimNum();
    OP_CHECK_IF(
        (xDims < X_SHAPE_MIN_LEN),
        OP_LOGE(context->GetNodeType(), "inputDims can't be smaller than 2."),
        return ge::GRAPH_FAILED);
    uint64_t channel = xShape.GetDim(DIM_1);
    // check gamma and beta
    auto gammaShapePtr = context->GetInputShape(INPUT_IDX_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShapePtr);
    auto gammaShape = gammaShapePtr->GetStorageShape();
    uint64_t gammaSizes = gammaShape.GetDim(DIM_0);
    OP_CHECK_IF(
        (gammaShape.GetDimNum() != 1 || gammaSizes != channel),
        OP_LOGE(
            context->GetNodeType(),
            "The shape of gamma must be"
            " the same as channe, currently is %lu.",
            gammaSizes),
        return ge::GRAPH_FAILED);
    auto betaShapePtr = context->GetInputShape(INPUT_IDX_BETA);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaShapePtr);
    auto betaShape = betaShapePtr->GetStorageShape();
    uint64_t betaSizes = betaShape.GetDim(DIM_0);
    OP_CHECK_IF(
        (betaShape.GetDimNum() != 1 || betaSizes != channel),
        OP_LOGE(
            context->GetNodeType(),
            "The shape of beta must be"
            " the same as channel, currently is %lu.",
            betaSizes),
        return ge::GRAPH_FAILED);
    auto gammaDtypePtr = context->GetInputDesc(INPUT_IDX_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaDtypePtr);
    auto gammaDtype = gammaDtypePtr->GetDataType();
    uint64_t gammaDtypeSize = ge::GetSizeByDataType(gammaDtype);
    auto betaDtypePtr = context->GetInputDesc(INPUT_IDX_BETA);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDtypePtr);
    auto betaDtype = betaDtypePtr->GetDataType();
    uint64_t betaDtypeSize = ge::GetSizeByDataType(betaDtype);
    OP_CHECK_IF(
        (gammaDtypeSize != betaDtypeSize),
        OP_LOGE(
            context->GetNodeType(),
            "The dtype of gamma and beta must"
            " be consistent."),
        return ge::GRAPH_FAILED);
    if (xDtypeSize == FLOAT32_BYTES) {
        OP_CHECK_IF(
            (gammaDtypeSize != xDtypeSize || betaDtypeSize != xDtypeSize),
            OP_LOGE(
                context->GetNodeType(),
                "The dtype of x is float32, gamma and beta must"
                " be consistent."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext* tilingContext)
{
    auto xShape = tilingContext->GetInputShape(INPUT_IDX_X)->GetStorageShape();
    uint64_t channel = xShape.GetDim(DIM_1);
    // check num_groups
    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    const int64_t* numGroups = attrs->GetAttrPointer<int64_t>(INDEX_NUM_GROUPS);
    OP_CHECK_IF(
        (*numGroups <= 0), OP_LOGE(tilingContext->GetNodeType(), "numGroups must be bigger than 0."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (channel % *numGroups != 0),
        OP_LOGE(tilingContext->GetNodeType(), "channel must be integer multiples of numGroups."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GroupNormSilu(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Start running TilingPrepare4GroupNormSilu.");
    auto compileInfo = context->GetCompiledInfo<GroupNormSiluCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(context, "Get core num for ai_core:%d", compileInfo->totalCoreNum);


    OP_LOGD(context, "Get total core num:%d", compileInfo->totalCoreNum);
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0),
        OP_LOGE(context->GetNodeType(), "Failed to get core num."), return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_LOGD(context, "Get total ub size:%lu", compileInfo->ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0),
        OP_LOGE(context->GetNodeType(), "Failed to get ub size."), return ge::GRAPH_FAILED);

    OP_LOGD(context, "TilingPrepare4GroupNormSilu ends.");
    return ge::GRAPH_SUCCESS;
}

static void SetAttrParams(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData)
{
    auto attrs = context->GetAttrs();
    const int64_t* numGroups = attrs->GetAttrPointer<int64_t>(INDEX_NUM_GROUPS);
    const float* epsilon = attrs->GetAttrPointer<float>(INDEX_EPSILON);
    const bool* activateSilu = attrs->GetAttrPointer<bool>(INDEX_ACTIVATE_SILU);
    tilingData.set_numGroups(*numGroups);
    tilingData.set_epsilon(*epsilon);
    tilingData.set_activateSilu(*activateSilu);
}

static void SetTilingParams(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData)
{
    auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
    uint64_t hwNum = 1;
    uint64_t xDims = xShape.GetDimNum();
    for (uint64_t i = 2; i < xDims; i++) {
        hwNum = hwNum * xShape.GetDim(i);
    }
    tilingData.set_shapeC(xShape.GetDim(DIM_1));
    tilingData.set_shapeD(tilingData.get_shapeC() / tilingData.get_numGroups());
    tilingData.set_hwNum(hwNum);
    tilingData.set_elemNum(tilingData.get_shapeD() * hwNum);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t processSize = (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::KIRINX90 ||
                            ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::KIRIN9030) ?
                               DEFAULT_PROCESSSIZE_KIRINX90 :
                               DEFAULT_PROCESSSIZE;
    tilingData.set_processSize(static_cast<int64_t>(processSize));
}

static void SetBlockTiling(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData)
{
    auto xDtype = context->GetInputDesc(INPUT_IDX_X)->GetDataType();
    uint64_t xDtypeSize = ge::GetSizeByDataType(xDtype);
    if (xDtypeSize == 0){
        OP_LOGE(context, "Division by zero!");
        return;
    }
    auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
    auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
    uint64_t shapeN = xShape.GetDim(DIM_0);
    uint64_t totalGroups = static_cast<uint64_t>(shapeN) * static_cast<uint64_t>(tilingData.get_numGroups());
    tilingData.set_numPerCore(CeilDiv(totalGroups, compileInfo->totalCoreNum));
    tilingData.set_realCoreNum(CeilDiv(totalGroups, tilingData.get_numPerCore()));
    if (tilingData.get_hwNum() < static_cast<int64_t>(BLOCK_SIZE / xDtypeSize) &&
        (tilingData.get_hwNum() != 1 || tilingData.get_shapeD() < static_cast<int64_t>(BLOCK_SIZE / xDtypeSize))) {
        tilingData.set_realCoreNum(1);
    }

    tilingData.set_numLastCore(
        totalGroups - tilingData.get_numPerCore() * (tilingData.get_realCoreNum() - 1));
    // split coreNum according to N
    if (tilingData.get_hwNum() == 1 && tilingData.get_numGroups() == tilingData.get_shapeC()) {
        if (tilingData.get_shapeC() % (BLOCK_SIZE / xDtypeSize) != 0) {
            tilingData.set_realCoreNum(1);
            tilingData.set_numLastCore(shapeN);
        } else {
            tilingData.set_numPerCore(CeilDiv(shapeN, compileInfo->totalCoreNum));
            tilingData.set_realCoreNum(CeilDiv(shapeN, tilingData.get_numPerCore()));
        }
        tilingData.set_numLastCore(shapeN - tilingData.get_numPerCore() * (tilingData.get_realCoreNum() - 1));
    }

    uint64_t xShapeSize = xShape.GetShapeSize();
    if (xShapeSize == 0) {
        tilingData.set_realCoreNum(-1);
    }
}

static void SetTilingWithSmallHW(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData)
{
    auto xDtype = context->GetInputDesc(INPUT_IDX_X)->GetDataType();
    uint64_t xDtypeSize = ge::GetSizeByDataType(xDtype);
    auto gammaDtype = context->GetInputDesc(INPUT_IDX_GAMMA)->GetDataType();
    uint64_t gammaDtypeSize = ge::GetSizeByDataType(gammaDtype);
    if (tilingData.get_hwNum() == 1) {
        if (xDtypeSize == FLOAT32_BYTES) {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_HW_ONE_B32));
        } else if (gammaDtypeSize == FLOAT32_BYTES) {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_HW_ONE_MIXTYPE));
        } else {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_HW_ONE_B16));
        }
    } else {
        if (xDtypeSize == FLOAT32_BYTES) {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_BASIC_TEM_B32));
        } else if (gammaDtypeSize == FLOAT32_BYTES) {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_BASIC_TEM_MIXTYPE));
        } else {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_BASIC_TEM_B16));
        }
    }
}

static void SetUbTiling(GroupNormSiluTilingData& tilingData)
{
    tilingData.set_loopNum(CeilDiv(tilingData.get_elemNum(), tilingData.get_processSize()));
    tilingData.set_loopTail(tilingData.get_elemNum() - tilingData.get_processSize() * (tilingData.get_loopNum() - 1));
    tilingData.set_innerLoopNum(CeilDiv(tilingData.get_hwNum(), tilingData.get_processSize()));
    tilingData.set_innerLoopTail(
        tilingData.get_hwNum() - tilingData.get_processSize() * (tilingData.get_innerLoopNum() - 1));
}

static void SetTiling(const gert::TilingContext* context, GroupNormSiluTilingData& tilingData)
{
    auto xDtype = context->GetInputDesc(INPUT_IDX_X)->GetDataType();
    uint64_t xDtypeSize = ge::GetSizeByDataType(xDtype);
        if (xDtypeSize == 0){
        OP_LOGE(context, "Division by zero!");
        return;
    }
    auto gammaDtype = context->GetInputDesc(INPUT_IDX_GAMMA)->GetDataType();
    uint64_t gammaDtypeSize = ge::GetSizeByDataType(gammaDtype);
    if (tilingData.get_hwNum() < static_cast<int64_t>(BLOCK_SIZE / xDtypeSize)) {
        SetTilingWithSmallHW(context, tilingData);
    } else if (
        xDtypeSize == FLOAT32_BYTES && tilingData.get_hwNum() % FLOAT_EIGHT == 0 && tilingData.get_hwNum() <= HW_CAP &&
        tilingData.get_shapeC() + tilingData.get_numPerCore() <= UPPER_LIMIT_TWO) {
        tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_SMALL_SHAPE_B32));
    } else if (
        xDtypeSize == FLOAT32_BYTES && tilingData.get_shapeC() + tilingData.get_numPerCore() <= UPPER_LIMIT_TWO) {
        tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_HIGH_PERF_B32));
    } else if (
        xDtypeSize == FLOAT16_BYTES && tilingData.get_hwNum() % FLOAT_DOUBLE_EIGHT == 0 &&
        tilingData.get_hwNum() <= HW_CAP && tilingData.get_shapeC() + tilingData.get_numPerCore() <= UPPER_LIMIT_ONE) {
        if (gammaDtypeSize == FLOAT32_BYTES) {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_SMALL_SHAPE_MIXTYPE));
        } else {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_SMALL_SHAPE_B16));
        }
    } else if (
        xDtypeSize == FLOAT16_BYTES && tilingData.get_shapeC() + tilingData.get_numPerCore() <= UPPER_LIMIT_ONE) {
        if (gammaDtypeSize == FLOAT32_BYTES) {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_HIGH_PERF_MIXTYPE));
        } else {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_HIGH_PERF_B16));
        }
    } else if (xDtypeSize == FLOAT32_BYTES) {
        tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_BASIC_TEM_B32));
    } else {
        if (gammaDtypeSize == FLOAT32_BYTES) {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_BASIC_TEM_MIXTYPE));
        } else {
            tilingData.set_tilingKey(static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_BASIC_TEM_B16));
        }
    }
}





static ge::graphStatus Tiling4GroupNormSilu(gert::TilingContext* context)
{
    OP_LOGD(context, "Start running Tiling4GroupNormSilu.");
    auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
    if (compileInfo->isRegbase == 1) {
        return Tiling4GroupNormSiluRegBase(context);
    }
    // check input && attrs params
    OP_CHECK_IF(
        (CheckInputParams(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeType(), "InputParams is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (CheckAttrParams(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeType(), "AttrParams is invalid."), return ge::GRAPH_FAILED);
    GroupNormSiluTilingData tilingData;
    SetAttrParams(context, tilingData);
    SetTilingParams(context, tilingData);
    // block tiling
    SetBlockTiling(context, tilingData);
    // tiling key
    SetTiling(context, tilingData);
    // ub tiling
    SetUbTiling(tilingData);
    size_t sysWorkspaceSize = RESERVED_WORKSPACE_SIZE_910B;

    OP_CHECK_IF(
        GroupNormSiluSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeType(), "GroupNormSiluSetTilingData set tiling data fail."),
        return ge::GRAPH_FAILED);
    OP_LOGD(
        context,
        "tilingData is numGroups:%ld, hwNum:%ld, elemNum:%ld, shapeC:%ld, shapeD:%ld, realCoreNum:%ld, \
                numPerCore:%ld, numLastCore:%ld, processSize:%ld, loopNum:%ld, loopTail:%ld, innerLoopNum:%ld, \
                innerLoopTail:%ld, tilingKey:%ld, epsilon:%f, activateSilu:%ld, Tiling4GroupNormSilu ends. ",
        tilingData.get_numGroups(), tilingData.get_hwNum(), tilingData.get_elemNum(), tilingData.get_shapeC(),
        tilingData.get_shapeD(), tilingData.get_realCoreNum(), tilingData.get_numPerCore(),
        tilingData.get_numLastCore(), tilingData.get_processSize(), tilingData.get_loopNum(), tilingData.get_loopTail(),
        tilingData.get_innerLoopNum(), tilingData.get_innerLoopTail(), tilingData.get_tilingKey(),
        tilingData.get_epsilon(), tilingData.get_activateSilu());

    // block dim, tilingKey
    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupNormSilu)
    .Tiling(Tiling4GroupNormSilu)
    .TilingParse<GroupNormSiluCompileInfo>(TilingPrepare4GroupNormSilu);
} // namespace optiling
