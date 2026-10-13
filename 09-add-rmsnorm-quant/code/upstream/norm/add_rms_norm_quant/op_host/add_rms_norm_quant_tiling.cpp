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
 * \file add_rms_norm_quant_tiling.cpp
 * \brief
 */
#include <iostream>
#include "op_host/tiling_util.h"
#include "add_rms_norm_quant_tiling.h"

namespace optiling {
using namespace Ops::NN::OpTiling;

constexpr uint32_t DTYPE_KEY_FP16 = 1;
constexpr uint32_t DTYPE_KEY_FP32 = 2;
constexpr uint32_t DTYPE_KEY_BF16 = 3;
constexpr uint32_t UB_FACTOR_B16 = 8192;
constexpr uint32_t UB_FACTOR_B16_CUTD = 7680;
constexpr uint32_t UB_FACTOR_B32_CUTD = 4096;
constexpr uint32_t UB_FACTOR_SINGLE_N_B16 = 12224;
constexpr uint32_t BLOCK_ALIGN_NUM = 16;
constexpr uint32_t BLOCK_ALIGN_NUM_32 = 32;
constexpr size_t INPUT_IDX_GAMMA = 2;
constexpr size_t INPUT_IDX_SCALES2 = 4;
constexpr size_t INPUT_IDX_ZERO_POINTS1 = 5;
constexpr size_t INPUT_IDX_ZERO_POINTS2 = 6;
constexpr size_t INPUT_IDX_BETA = 7;
constexpr uint32_t MODE_NORMAL = 0;
constexpr uint32_t MODE_SPLIT_D = 1;
constexpr uint32_t MODE_SINGLE_N = 3;
constexpr uint64_t CUT_UB_FOR_FREE = 1024;
constexpr float FLOAT_ZERO = 0.0f;
constexpr float FLOAT_ONE = 1.0f;
constexpr uint32_t INT_ZERO = 0;
constexpr uint32_t INT_ONE = 1;
constexpr uint32_t SIZE_FLOAT = 4;
constexpr uint32_t SIZE_HALF = 2;
constexpr uint32_t SIZE_OF_UB = 196608;
constexpr size_t DEST_MAX = 100;
constexpr size_t MAX_LEN_SIMPLIFIED_KEY = 256;

uint32_t optionalUbNum_0 = 0;
uint32_t optionalUbNum_1 = 0;
constexpr uint32_t MODEKEY_WEIGHT = 10000;
constexpr uint32_t BIAS_WEIGHT = 1000;
constexpr uint32_t OUT_SECOND_WEIGHT = 100;
constexpr uint32_t OUT_THIRD_WEIGHT = 10;
constexpr uint32_t BIAS_INDEX = 7;
constexpr uint32_t RESOUT_INDEX = 3;

AddRMSNormQuantTilingData addRMSNormQuantTilingData;

static void InitPlatformParams(
    gert::TilingContext* context, const AddRmsNormQuantCompileInfo* ptrCompileInfo, uint32_t& numCore)
{
    if (nullptr == ptrCompileInfo) {
        auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        numCore = ascendc_platform.GetCoreNumAiv();
    } else {
        numCore = ptrCompileInfo->totalCoreNum;
    }
}

bool CheckOptionalParamsExisting(const gert::StorageShape* checkShape)
{
    OP_CHECK_IF(nullptr == checkShape, OP_LOGD("CheckOptionalParamsExisting", "Get nullptr checkShape"), return false);
    int64_t checkShapeSize = checkShape->GetOriginShape().GetShapeSize();
    OP_CHECK_IF((checkShapeSize <= 0), OP_LOGD("CheckOptionalParamsExisting", "Get empty checkShapeSize"), return false);
    return true;
}

static uint32_t CalcTilingKeyWithOptionalInput(gert::TilingContext* context)
{
    uint32_t modeKy = context->GetTilingKey();
    uint32_t biasKey = 0;
    uint32_t out2Key = 0;
    uint32_t out3Key = 0;
    uint32_t quantTypeKey = 0;
    int64_t mode = static_cast<int64_t>(-1);
    if (context->GetOptionalInputDesc(BIAS_INDEX) != nullptr) {
        biasKey = 1U;
    }
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    if (attrs != nullptr) {
        mode = *attrs->GetInt(0);
    }
    if (mode == 0) {
        out2Key = 1U;
    } else if (mode == 1) {
        out3Key = 1U;
    }
    if (context->GetInputDesc(SCALES1_INDEX) != nullptr &&
        context->GetInputShape(SCALES1_INDEX)->GetStorageShape().GetDimNum() == 1 &&
        context->GetInputShape(SCALES1_INDEX)->GetStorageShape().GetShapeSize() == 1) {
        quantTypeKey = 1U;
    }
    modeKy = modeKy * MODEKEY_WEIGHT + biasKey * BIAS_WEIGHT + out2Key * OUT_SECOND_WEIGHT +
             out3Key * OUT_THIRD_WEIGHT + quantTypeKey;
    OP_LOGD("Tiling4AddRmsNormQuantV2", "tiling_ky value is: %u", modeKy);
    context->SetTilingKey(modeKy);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetOpDescInfo(gert::TilingContext* context, uint32_t& numCol, uint32_t& numRow)
{
    const gert::StorageShape* scales2Shape = context->GetOptionalInputShape(INPUT_IDX_SCALES2);
    const gert::StorageShape* zeroPoints1Shape = context->GetOptionalInputShape(INPUT_IDX_ZERO_POINTS1);
    const gert::StorageShape* betaShape = context->GetOptionalInputShape(INPUT_IDX_BETA);
    const gert::StorageShape* zeroPoints2Shape = context->GetOptionalInputShape(INPUT_IDX_ZERO_POINTS2);
    bool scales2Exist = CheckOptionalParamsExisting(scales2Shape);
    bool zeroPoints2Exist = CheckOptionalParamsExisting(zeroPoints2Shape);
    bool zeroPoints1Exist = CheckOptionalParamsExisting(zeroPoints1Shape);
    bool betaExist = CheckOptionalParamsExisting(betaShape);
    uint32_t hasZeroPoints1 = zeroPoints1Exist ? INT_ONE : INT_ZERO;
    uint32_t hasBeta = betaExist ? INT_ONE : INT_ZERO;
    uint32_t hasScales2 = scales2Exist ? INT_ONE : INT_ZERO;
    uint32_t hasZeroPoints2 = zeroPoints2Exist && scales2Exist ? INT_ONE : INT_ZERO;

    const gert::Shape xShape = context->GetInputShape(0)->GetStorageShape();
    std::string opType(context->GetNodeType());
    const gert::Shape gammaShape = context->GetInputShape(INPUT_IDX_GAMMA)->GetStorageShape();
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    float epsilon = *attrs->GetFloat(1);
    bool divMode = *attrs->GetBool(2);
    numCol = gammaShape.GetShapeSize();
    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();
    numRow = 1U;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        numRow *= xShape.GetDim(i);
    }
    float avgFactor = (numCol == INT_ZERO) ? 0.0f : 1.0f / static_cast<float>(numCol);
    // 可选参数占用ubFactor的大小, 有beta和s2时，才是新增场景，否则不做改变
    if (hasScales2 > INT_ZERO || hasBeta > INT_ZERO) {
        optionalUbNum_0 = hasBeta * SIZE_HALF + hasZeroPoints1 * SIZE_FLOAT + SIZE_FLOAT;
        optionalUbNum_1 = SIZE_FLOAT;
        if (hasScales2 > INT_ZERO) {
            optionalUbNum_0 += SIZE_FLOAT + SIZE_HALF;
            optionalUbNum_1 += SIZE_FLOAT;
            if (hasZeroPoints2 > INT_ZERO) {
                optionalUbNum_0 += SIZE_FLOAT;
                optionalUbNum_1 += SIZE_FLOAT;
            }
        }
    }

    addRMSNormQuantTilingData.set_divMode(divMode ? INT_ONE : INT_ZERO);
    addRMSNormQuantTilingData.set_epsilon(epsilon);
    addRMSNormQuantTilingData.set_avgFactor(avgFactor);
    addRMSNormQuantTilingData.set_hasScales2(hasScales2);
    addRMSNormQuantTilingData.set_hasZeroPoints1(hasZeroPoints1);
    addRMSNormQuantTilingData.set_hasZeroPoints2(hasZeroPoints2);
    addRMSNormQuantTilingData.set_hasBeta(hasBeta);

    return ge::GRAPH_SUCCESS;
}

static void CalcModeAndUbFactor(
    gert::TilingContext* context, const AddRmsNormQuantCompileInfo* ptrCompileInfo, uint32_t blockFactor,
    uint32_t numCol)
{
    platform_ascendc::SocVersion socVersion;
    uint64_t ubSize = 0;
    if (nullptr == ptrCompileInfo) {
        auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        socVersion = ascendc_platform.GetSocVersion();
        ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    } else {
        ubSize = ptrCompileInfo->maxUbSize;
        socVersion = ptrCompileInfo->curSocVersion;
    }

    ubSize = ubSize > INT_ZERO ? ubSize : SIZE_OF_UB;
    uint32_t modeKey = MODE_NORMAL; // 0: Normal, 1: SplitD, 2: MultiN 3: SingleN
    auto xDataType = context->GetInputDesc(0)->GetDataType();
    // 默认 8192
    uint32_t ubFactor = UB_FACTOR_B16;
    uint64_t actualMaxUbSize = ubSize - CUT_UB_FOR_FREE;
    // 原始的ubFactor值
    uint64_t singleUbFactor = UB_FACTOR_SINGLE_N_B16;
    uint64_t normalUbFactor = UB_FACTOR_B16;

    uint32_t colTileNum =
        Ops::Base::CeilDiv(numCol, ((xDataType == ge::DT_FLOAT) ? UB_FACTOR_B32_CUTD : UB_FACTOR_B16_CUTD));
    uint64_t splitDUbFactor = Ops::Base::CeilDiv(numCol, colTileNum * BLOCK_ALIGN_NUM) * BLOCK_ALIGN_NUM;

    if (optionalUbNum_0 > INT_ZERO) {
        uint64_t staticBlockNum4Normal = 20;
        uint64_t staticUsedUb4Normal = 256;
        uint64_t tmp =
            (actualMaxUbSize - staticUsedUb4Normal) / (staticBlockNum4Normal + static_cast<uint64_t>(optionalUbNum_0));
        normalUbFactor = (tmp / BLOCK_ALIGN_NUM_32) * BLOCK_ALIGN_NUM_32;
        uint64_t staticBlockNum4SplitD = 22;
        uint64_t staticUsedUb4SplitD = 2560;
        tmp =
            (actualMaxUbSize - staticUsedUb4SplitD) / (staticBlockNum4SplitD + static_cast<uint64_t>(optionalUbNum_0));
        splitDUbFactor = (tmp / BLOCK_ALIGN_NUM_32) * BLOCK_ALIGN_NUM_32;
    }

    if (optionalUbNum_1 > INT_ZERO) {
        uint64_t tmp = actualMaxUbSize / (static_cast<uint64_t>(16) + static_cast<uint64_t>(optionalUbNum_1));
        singleUbFactor = (tmp / BLOCK_ALIGN_NUM_32) * BLOCK_ALIGN_NUM_32;
    }

    if (blockFactor == 1 && socVersion != platform_ascendc::SocVersion::ASCEND310P && numCol <= singleUbFactor) {
        modeKey = MODE_SINGLE_N;
        ubFactor = singleUbFactor;
        OP_LOGD("Tiling4AddRmsNormQuantNotRegbase", "kernel template: singleN, ubFactor: %d", ubFactor);
    } else if (numCol > normalUbFactor) {
        modeKey = MODE_SPLIT_D;
        ubFactor = splitDUbFactor;
        OP_LOGD("Tiling4AddRmsNormQuantNotRegbase", "kernel template: splitD, ubFactor: %d", ubFactor);
    } else {
        modeKey = MODE_NORMAL;
        ubFactor = normalUbFactor;
        OP_LOGD("Tiling4AddRmsNormQuantNotRegbase", "kernel template: normal, ubFactor: %d", ubFactor);
    }
    context->SetTilingKey(modeKey);
    addRMSNormQuantTilingData.set_ubFactor(ubFactor);
}

static ge::graphStatus Tiling4AddRmsNormQuantNotRegbase(gert::TilingContext* context)
{
    OP_LOGD("Tiling4AddRmsNormQuantNotRegbase", "Enter Tiling4AddRmsNormQuantNotRegbase");

    auto ptrCompileInfo = reinterpret_cast<const AddRmsNormQuantCompileInfo*>(context->GetCompileInfo());
    uint32_t numCore;
    InitPlatformParams(context, ptrCompileInfo, numCore);

    uint32_t numCol;
    uint32_t numRow;
    OP_CHECK_IF(
        GetOpDescInfo(context, numCol, numRow) != ge::GRAPH_SUCCESS, OP_LOGE(context, "Get Opdesc Info failed."),
        return ge::GRAPH_FAILED);

    uint32_t blockFactor = 1;
    uint32_t tileNum = Ops::Base::CeilDiv(numRow, numCore * blockFactor);
    OP_LOGD("Tiling4AddRmsNormQuantNotRegbase", "Core Num: %u, tile num: %d", numCore, tileNum);
    blockFactor *= tileNum;
    uint32_t useCoreNum = Ops::Base::CeilDiv(numRow, blockFactor);

    context->SetBlockDim(useCoreNum);

    uint32_t rowFactor = 64;
    CalcModeAndUbFactor(context, ptrCompileInfo, blockFactor, numCol);
    addRMSNormQuantTilingData.set_numRow(numRow);
    addRMSNormQuantTilingData.set_numCol(numCol);
    addRMSNormQuantTilingData.set_blockFactor(blockFactor);
    addRMSNormQuantTilingData.set_rowFactor(rowFactor);

    addRMSNormQuantTilingData.SaveToBuffer(
        context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(addRMSNormQuantTilingData.GetDataSize());

    size_t sysWorkspaceSize = 16UL * 1024UL * 1024UL;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t usrSize = 256;
    currentWorkspace[0] = usrSize + sysWorkspaceSize;

    OP_LOGD("Tiling4AddRmsNormQuantNotRegbase", "Block Dim: %u", useCoreNum);
    OP_LOGD("Tiling4AddRmsNormQuantNotRegbase", "usr Workspace: %zu", usrSize);
    OP_LOGD(
        "Tiling4AddRmsNormQuantNotRegbase", "numRow: %d, numCol: %d, blockFactor: %d, rowFactor: %d", numRow, numCol,
        blockFactor, rowFactor);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4AddRmsNormQuant(gert::TilingContext* context)
{
    if (IsRegbaseSocVersion(context)) {
        AddRmsNormQuantRegbaseTiling regbaseTiling(context);
        return regbaseTiling.DoTiling();
    }
    return Tiling4AddRmsNormQuantNotRegbase(context);
}

ge::graphStatus Tiling4AddRmsNormQuantV2(gert::TilingContext* context)
{
    OP_CHECK_IF(
        Tiling4AddRmsNormQuantNotRegbase(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "Tiling4AddRmsNormQuantNotRegbase failed."),
        return ge::GRAPH_FAILED);
    CalcTilingKeyWithOptionalInput(context);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4AddRmsNormQuant(gert::TilingParseContext* context)
{
    OP_CHECK_IF(nullptr == context, OP_LOGE("AddRmsNormQuant", "Context is null"), return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter TilingPrepare4AddRmsNormQuant.");
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfoPtr == nullptr, OP_LOGE("AddRmsNormQuant", "PlatformInfoPtr is null"),
        return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<AddRmsNormQuantCompileInfo>();
    OP_CHECK_IF(
        compileInfoPtr == nullptr, OP_LOGE("AddRmsNormQuant", "CompileInfoPtr is null"),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->curSocVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->maxUbSize);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus GenSimplifiedKey4AddRmsNormQuant(gert::TilingContext* context, ge::char_t* simplifiedKey)
{
    OP_CHECK_IF(nullptr == context, OP_LOGE("AddRmsNormQuant", "Context is null"), return ge::GRAPH_FAILED);
    OP_LOGW(context, "Enter AddRmsNormQuant genSimplifiedKey.");

    OP_CHECK_IF(
        nullptr == simplifiedKey, OP_LOGE(context, "SimplifiedKey is null"), return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(X1_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(X2_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(GAMMA_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(SCALES1_INDEX));

    int32_t x1Dtype = static_cast<int32_t>(context->GetInputDesc(X1_INDEX)->GetDataType());
    int32_t x2Dtype = static_cast<int32_t>(context->GetInputDesc(X2_INDEX)->GetDataType());
    int32_t gammaDtype = static_cast<int32_t>(context->GetInputDesc(GAMMA_INDEX)->GetDataType());
    int32_t scales1Dtype = static_cast<int32_t>(context->GetInputDesc(SCALES1_INDEX)->GetDataType());

    int32_t scales2Dtype = -1;
    int32_t zeroPoints1Dtype = -1;
    int32_t zeroPoints2Dtype = -1;

    int32_t y1Dtype = static_cast<int32_t>(context->GetOutputDesc(Y1_INDEX)->GetDataType());

    OP_CHECK_IF(
        context->GetOptionalInputDesc(SCALES2_INDEX) != nullptr,
        OP_LOGW(context, "Optional input scale2 exist"),
        scales2Dtype = static_cast<int32_t>(context->GetOptionalInputDesc(SCALES2_INDEX)->GetDataType()));
    OP_CHECK_IF(
        context->GetOptionalInputDesc(ZERO_POINTS1_INDEX) != nullptr,
        OP_LOGW(context, "Optional input zeroPoints1 exist"),
        zeroPoints1Dtype = static_cast<int32_t>(context->GetOptionalInputDesc(ZERO_POINTS1_INDEX)->GetDataType()));
    OP_CHECK_IF(
        context->GetOptionalInputDesc(ZERO_POINTS2_INDEX) != nullptr,
        OP_LOGW(context, "Optional input zeroPoints2 exist"),
        zeroPoints2Dtype = static_cast<int32_t>(context->GetOptionalInputDesc(ZERO_POINTS2_INDEX)->GetDataType()));

    std::string simpleKeyTemp = "";
    strcat_s(simplifiedKey, DEST_MAX, "diy,");
    simpleKeyTemp
        .append(std::to_string(x1Dtype)) // x1
        .append("/")
        .append(std::to_string(x2Dtype)) // x2
        .append("/")
        .append(std::to_string(gammaDtype)) // gamma
        .append("/")
        .append(std::to_string(scales1Dtype)) // scales1
        .append("/")
        .append(std::to_string(scales2Dtype)) // scales2
        .append("/")
        .append(std::to_string(zeroPoints1Dtype)) // zeroPoints1
        .append("/")
        .append(std::to_string(zeroPoints2Dtype)) // zeroPoints2
        .append("/")
        .append(std::to_string(y1Dtype)) // y1Dtype
        .append("/");
    OP_LOGW(context, "SimpleKeyTemp: %s", simpleKeyTemp.c_str());
    errno_t err = strcat_s(simplifiedKey, DEST_MAX, simpleKeyTemp.c_str());
    OP_CHECK_IF(
        (err != 0), OP_LOGE(context, "Error: strcat_s failed with error code %d.", err),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        strlen(simplifiedKey) > MAX_LEN_SIMPLIFIED_KEY,
        OP_LOGE(context, "Len of simplifiedKey exceeds max length."), return ge::GRAPH_FAILED);
    OP_LOGW(context, "Finish AddRmsNormQuant genSimplifiedKey.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddRmsNormQuant)
    .Tiling(Tiling4AddRmsNormQuant)
    .TilingParse<AddRmsNormQuantCompileInfo>(TilingPrepare4AddRmsNormQuant)
    .GenSimplifiedKey(GenSimplifiedKey4AddRmsNormQuant);

IMPL_OP_OPTILING(AddRmsNormQuantV2)
    .Tiling(Tiling4AddRmsNormQuantV2)
    .TilingParse<AddRmsNormQuantCompileInfo>(TilingPrepare4AddRmsNormQuant);
} // namespace optiling