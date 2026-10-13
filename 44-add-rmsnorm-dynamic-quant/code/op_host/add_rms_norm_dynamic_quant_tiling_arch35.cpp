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
 * \file add_rms_norm_dynamic_quant_tiling_arch35.cpp
 * \brief
 */

#include "add_rms_norm_dynamic_quant_tiling.h"
#include "../norm_common/op_host/norm_tiling_check_common.h"

namespace optiling {
using namespace NormCheck;

constexpr uint64_t X1_INDEX = 0;
constexpr uint64_t X2_INDEX = 1;
constexpr uint64_t GAMMA_INDEX = 2;
constexpr uint64_t SMOOTH_SCALE1_INDEX = 3;
constexpr uint64_t SMOOTH_SCALE2_INDEX = 4;
constexpr uint64_t Y1_INDEX = 0;
constexpr uint64_t Y2_INDEX = 1;
constexpr uint64_t X_INDEX = 2;
constexpr uint64_t SCALE1_INDEX = 3;
constexpr uint64_t SCALE2_INDEX = 4;
constexpr uint64_t EPS_ATTR_INDEX = 0;
constexpr uint64_t DST_TYPE__ATTR_INDEX = 2;

constexpr uint32_t LOG_2 = 2;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t MAX_DIM_CNT = 8;
constexpr uint32_t WORKSPACE_COUNT = 3;
constexpr uint32_t B32_BLOCK_NUM = 8;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t ALING_FACTOR_512 = 512;
constexpr uint32_t ONCE_VECTOR_SIZE = 256;
constexpr uint32_t DEFAULT_SYS_WORKSPACE = 16 * 1024 * 1024;

constexpr uint32_t LEVEL_BUFFER_CNT = 3;
constexpr uint32_t MULTI_FACTOR_2 = 2;
constexpr uint32_t SMOOTH_SCALE1_BIN_OFFSET = 1;
constexpr uint32_t FULL_LOAD_R_MAX = 16384;
constexpr uint32_t ALIGN_SPACE = 1 * 1024;
constexpr uint32_t DOUBLE_BUFFER = 2;

    ge::graphStatus
    AddRmsNormDynamicQuantRegbaseTiling::CheckDtypeVaild(
        ge::DataType & srcDtype, std::vector<ge::DataType>& supportDtypeList, string srcName)
{
    for (const auto& supportedDtype : supportDtypeList) {
        if (supportedDtype == srcDtype) {
            return ge::GRAPH_SUCCESS;
        }
    }
    OP_LOGE(
        nodeName.c_str(), "Dtype check invalid, %s dtype is %s, not in supportDtypeList.", srcName.c_str(),
        Ops::Base::ToString(srcDtype).c_str());
    return ge::GRAPH_FAILED;
}

bool AddRmsNormDynamicQuantRegbaseTiling::CheckShapeNull()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormDynamicQuantRegbaseTiling CheckShapeNull.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    const gert::StorageShape* xShape = context_->GetOutputShape(X_INDEX);

    OP_CHECK_IF(
        (nullptr == x1Shape) || (nullptr == x2Shape) || (nullptr == gammaShape) || (nullptr == xShape), , return false);
    return true;
}

bool AddRmsNormDynamicQuantRegbaseTiling::CheckOptionalInput()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormDynamicQuantRegbaseTiling CheckOptionalInput.");
    const gert::StorageShape* smoothScale1Shape = context_->GetOptionalInputShape(SMOOTH_SCALE1_INDEX);
    const gert::StorageShape* smoothScale2Shape = context_->GetOptionalInputShape(SMOOTH_SCALE2_INDEX);

    tilingParams.quantBufCnt = 0;
    if (smoothScale1Shape != nullptr) {
        tilingParams.hasSmoothScale1 = true;
        tilingParams.quantBufCnt++;
    }
    if (smoothScale2Shape != nullptr) {
        tilingParams.hasSmoothScale2 = true;
        tilingParams.quantBufCnt++;
    }
    OP_CHECK_IF(
        !tilingParams.hasSmoothScale1 && tilingParams.hasSmoothScale2,
        OP_LOGE(nodeName.c_str(), "When input smoothScale2, must input smoothScale1."), return false);
    tilingParams.hasY2Scale2 = tilingParams.hasSmoothScale2;
    return true;
}

bool AddRmsNormDynamicQuantRegbaseTiling::CheckInputShapeDim()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormDynamicQuantRegbaseTiling CheckInputShapeDim.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);

    // Not support zero shape.
    size_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
    size_t x2DimNum = x2Shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        (x1DimNum > MAX_DIM_CNT) || (x2DimNum > MAX_DIM_CNT),
        OP_LOGE(nodeName.c_str(), "Input x1/x2 dim should not bigger than %u.", MAX_DIM_CNT), return false);
    OP_CHECK_IF(!CheckDimBiggerZero(x1Shape, x1DimNum, nodeName, "x1"), , return false);
    OP_CHECK_IF(!CheckDimBiggerZero(x2Shape, x2DimNum, nodeName, "x2"), , return false);
    return true;
}

bool AddRmsNormDynamicQuantRegbaseTiling::CheckInputShapeValue()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormDynamicQuantRegbaseTiling CheckInputShapeValue.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    const gert::StorageShape* smoothScale1Shape = context_->GetOptionalInputShape(SMOOTH_SCALE1_INDEX);
    const gert::StorageShape* smoothScale2Shape = context_->GetOptionalInputShape(SMOOTH_SCALE2_INDEX);
    const gert::StorageShape* xShape = context_->GetOutputShape(X_INDEX);

    // Check x1&x2&y1&y2&x's shape should be equal
    if (!NormCheck::CheckShapeSame(x1Shape, x2Shape, nodeName, "x1", "x2")) {
        return false;
    };
    if (!NormCheck::CheckShapeSame(x1Shape, xShape, nodeName, "x1", "x")) {
        return false;
    };

    // Check smoothScale&scale's shape should be equal
    if (tilingParams.hasSmoothScale1 &&
        !NormCheck::CheckShapeSame(gammaShape, smoothScale1Shape, nodeName, "gamma", "smoothScale1")) {
        return false;
    };
    if (tilingParams.hasSmoothScale2 &&
        !NormCheck::CheckShapeSame(gammaShape, smoothScale2Shape, nodeName, "gamma", "smoothScale2")) {
        return false;
    };

    // Check gamma should be last dim of x
    // Check scale should be not last dim of x
    if ((1 == gammaShape->GetStorageShape().GetDimNum()) &&
        !NormCheck::CheckShapeBC(x1Shape, gammaShape, nodeName, "x1", "gamma", true)) {
        return false;
    };

    return true;
}

bool AddRmsNormDynamicQuantRegbaseTiling::CheckOutputDtype()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormDynamicQuantRegbaseTiling CheckOutputDtype.");
    std::vector<ge::DataType> supportedYDtypes = {
        ge::DataType::DT_INT8, ge::DataType::DT_HIFLOAT8, ge::DataType::DT_FLOAT8_E4M3FN, ge::DataType::DT_FLOAT8_E5M2};
    auto y1DataType = context_->GetOutputDesc(Y1_INDEX)->GetDataType();
    auto y2DataType = context_->GetOutputDesc(Y2_INDEX)->GetDataType();
    if ((ge::GRAPH_SUCCESS != CheckDtypeVaild(y1DataType, supportedYDtypes, "AddRmsNormDynamicQuant")) ||
        (ge::GRAPH_SUCCESS != CheckDtypeVaild(y2DataType, supportedYDtypes, "AddRmsNormDynamicQuant")) ||
        (y1DataType != y2DataType)) {
        OP_LOGE(
            nodeName.c_str(), "Output dtype should be int8 fp8e4m3 fp8e5m2 hifp8 and y1DataType y2DataType need same.");
        return false;
    }
    return true;
}

bool AddRmsNormDynamicQuantRegbaseTiling::CheckInputDtype()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormDynamicQuantRegbaseTiling CheckInputDtype.");
    std::vector<ge::DataType> supportedXGammaDtypes = {ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16};
    std::vector<ge::DataType> supportedSmoothScaleDtypes = {ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16};

    ge::DataType x1Dtype = context_->GetInputTensor(X1_INDEX)->GetDataType();
    ge::DataType x2Dtype = context_->GetInputTensor(X2_INDEX)->GetDataType();
    ge::DataType gammaDtype = context_->GetInputTensor(GAMMA_INDEX)->GetDataType();
    ge::DataType smoothScale1Dtype = ge::DT_FLOAT;
    ge::DataType smoothScale2Dtype = ge::DT_FLOAT;
    if (tilingParams.hasSmoothScale1) {
        smoothScale1Dtype = context_->GetOptionalInputTensor(SMOOTH_SCALE1_INDEX)->GetDataType();
    }
    if (tilingParams.hasSmoothScale2) {
        smoothScale2Dtype = context_->GetOptionalInputTensor(SMOOTH_SCALE2_INDEX)->GetDataType();
    }
    if ((x1Dtype != x2Dtype) || (x1Dtype != gammaDtype)) {
        OP_LOGE(nodeName.c_str(), "Input x1/gamma/xout dtype should be equal.");
        return false;
    }
    if (tilingParams.hasSmoothScale1 && (x1Dtype != smoothScale1Dtype)) {
        OP_LOGE(nodeName.c_str(), "Input smoothScale1/x1 dtype should be equal.");
        return false;
    }
    if (tilingParams.hasSmoothScale2 && (x1Dtype != smoothScale2Dtype)) {
        OP_LOGE(nodeName.c_str(), "Input smoothScale2/x1 dtype should be equal.");
        return false;
    }

    return true;
}

ge::graphStatus AddRmsNormDynamicQuantRegbaseTiling::SetInputParams()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormDynamicQuantRegbaseTiling SetInputParams.");
    // Set input dim
    const gert::Shape x1Shape = context_->GetInputShape(X1_INDEX)->GetStorageShape();
    const gert::Shape gammaShape = context_->GetInputShape(GAMMA_INDEX)->GetStorageShape();
    size_t x1DimNum = x1Shape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();
    uint64_t numM = 1;
    uint64_t numN = 1;
    for (size_t i = 0; i < x1DimNum - gammaDimNum; i++) {
        numM *= x1Shape.GetDim(i);
    }
    for (size_t i = 0; i < gammaDimNum; i++) {
        numN *= gammaShape.GetDim(i);
    }
    tilingParams.numM = numM;
    tilingParams.numN = numN;

    // Set input dtype
    auto xDataType = context_->GetInputTensor(X_INDEX)->GetDataType();
    tilingParams.xDtypeSize = GetSizeByDataType(xDataType);
    tilingParams.xDtypeAlignNum = BLOCK_SIZE / tilingParams.xDtypeSize;
    tilingParams.xReduceAlignNum = ALING_FACTOR_512 / tilingParams.xDtypeSize;
    tilingParams.vecLength = Ops::Base::GetVRegSize(context_) / sizeof(float);

    // Set input attr
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilon = attrs->GetFloat(EPS_ATTR_INDEX);
    tilingParams.epsilon = *epsilon;
    tilingParams.needRun = true;
    if ((0 == numN) || (0 == numM)) {
        tilingParams.needRun = false;
    }
    tilingParams.avgFactor = (0 == numN) ? 0.0f : 1.0f / static_cast<float>(numN);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicQuantRegbaseTiling::GetShapeAttrsInfo()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormDynamicQuantRegbaseTiling GetShapeAttrsInfo.");
    OP_CHECK_IF(
        !CheckShapeNull(), OP_LOGE(nodeName.c_str(), "The not optional input is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckOptionalInput(), OP_LOGE(nodeName.c_str(), "The optional input is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckInputShapeDim(), OP_LOGE(nodeName.c_str(), "The input shape dim is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckInputShapeValue(), OP_LOGE(nodeName.c_str(), "The input shape relationship is invalid."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputDtype(), OP_LOGE(nodeName.c_str(), "The input dtype is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckOutputDtype(), OP_LOGE(nodeName.c_str(), "The output dtype is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ge::GRAPH_SUCCESS != SetInputParams(), OP_LOGE(nodeName.c_str(), "Set input shape failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicQuantRegbaseTiling::GetPlatformInfo()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormDynamicQuantRegbaseTiling GetPlatformInfo.");
    if (context_->GetCompileInfo() == nullptr) {
        OP_LOGD(nodeName.c_str(), "GetPlatformInfo return nullptr, need re get later.");
        tilingParams.needGetCompileInfo = true;
    } else {
        auto compileInfo = reinterpret_cast<const AddRmsNormDynamicQuantCompileInfo*>(context_->GetCompileInfo());
        tilingParams.totalCoreNum = compileInfo->totalCoreNum;
        tilingParams.maxUbSize = compileInfo->maxUbSize;
        tilingParams.needGetCompileInfo = false;
    }
    return ge::GRAPH_SUCCESS;
}

bool AddRmsNormDynamicQuantRegbaseTiling::IsCapable()
{
    return true;
}

/**
 * @brief: Cal base UB total size
 * @param M dim Num
 * @param N dim Num
 * @return Bytes of UBSize
 */
uint64_t AddRmsNormDynamicQuantRegbaseTiling::CalUBTotalSize(
    uint64_t baseM, uint64_t baseN, const uint32_t tilingType = TILING_TYPE_NORMAL)
{
    uint64_t baseMB32Align = Ops::Base::CeilAlign(baseM, static_cast<uint64_t>(B32_BLOCK_NUM));
    uint64_t baseNB8Align = Ops::Base::CeilAlign(baseN, static_cast<uint64_t>(BLOCK_SIZE));
    uint64_t baseNReduceAlign = Ops::Base::CeilAlign(baseN, tilingParams.xReduceAlignNum);
    uint64_t baseNDtypeAlign = Ops::Base::CeilAlign(baseN, tilingParams.xDtypeAlignNum);
    uint64_t baseNB32Align = Ops::Base::CeilAlign(baseN, static_cast<uint64_t>(B32_BLOCK_NUM));
    uint64_t reduceSumBufLen = baseNReduceAlign / (2 * tilingParams.vecLength);
    uint64_t reduceSumBufLenAlign = Ops::Base::CeilAlign(reduceSumBufLen, static_cast<uint64_t>(B32_BLOCK_NUM));

    uint64_t totalSize =
        3 * baseNReduceAlign * tilingParams.xDtypeSize +                       // x1/x2/xout
        1 * baseNReduceAlign * sizeof(float) +                                 // xoutTmp(alloc bigger than use)
        1 * reduceSumBufLenAlign * sizeof(float) +                             // reduceBuf
        1 * baseNDtypeAlign * tilingParams.xDtypeSize +                        // gamma
        tilingParams.quantBufCnt * baseNDtypeAlign * tilingParams.xDtypeSize + // smoothScale1/smoothScale2
        1 * baseNB8Align * sizeof(int8_t) +                                    // y1
        1 * baseNB32Align * sizeof(float);                                     // y1Tmp

    if (tilingParams.hasY2Scale2) {
        totalSize += 1 * baseNB8Align * sizeof(int8_t); // y2
        totalSize += 1 * baseNB32Align * sizeof(float); // y2Tmp
    }

    if (TILING_TYPE_NORMAL == tilingType) {
        totalSize += NUM_TWO * baseMB32Align * sizeof(float); // rstd/scale1
        if (tilingParams.hasY2Scale2) {
            totalSize += 1 * baseMB32Align * sizeof(float); // scale2
        }
    } else {
        totalSize += NUM_TWO * B32_BLOCK_NUM * sizeof(float); // rstd/scale1
        if (tilingParams.hasY2Scale2) {
            totalSize += 1 * B32_BLOCK_NUM * sizeof(float); // scale2
        }
        totalSize += LEVEL_BUFFER_CNT * ONCE_VECTOR_SIZE * sizeof(float); // levelbuf
        totalSize += 1 * tilingParams.vecLength * sizeof(float);          // tempBuf
    }

    return totalSize;
}

int64_t AddRmsNormDynamicQuantRegbaseTiling::CalFullLoadBaseM(uint64_t baseN, int64_t& tmpPower)
{
    uint64_t baseNB8Align = Ops::Base::CeilAlign(baseN, static_cast<uint64_t>(BLOCK_SIZE));
    uint64_t baseNB32Align = Ops::Base::CeilAlign(baseN, static_cast<uint64_t>(B32_BLOCK_NUM));
    uint64_t baseNDtypeAlign = Ops::Base::CeilAlign(baseN, tilingParams.xDtypeAlignNum);
    tmpPower = std::floor(std::log(baseNDtypeAlign - 1) / std::log(LOG_2));
    tmpPower = std::pow(LOG_2, tmpPower);
    int64_t blockSize = Ops::Base::GetUbBlockSize(context_);
    int64_t vectorLength = Ops::Base::GetVRegSize(context_) / sizeof(float);
    int64_t firstVcaddLength = Ops::Base::CeilDiv(Ops::Base::CeilDiv(tmpPower, vectorLength), blockSize) * blockSize;
    int64_t LastUbSize =
        tilingParams.maxUbSize - baseNDtypeAlign * tilingParams.xDtypeSize -   // gamma
        tilingParams.quantBufCnt * baseNDtypeAlign * tilingParams.xDtypeSize - // smoothScale1/smoothScale2
        ALIGN_SPACE;                                                           // Scale1/rstd/xReduceTmp align space

    int64_t mutilBaseM = 3 * baseNDtypeAlign * tilingParams.xDtypeSize + // x1/x2/xout
                         baseNDtypeAlign * sizeof(float) +               // xoutTmp
                         DOUBLE_BUFFER * baseNB8Align * sizeof(int8_t) + // y1 * double
                         baseNB32Align * sizeof(float) +                 // y1Tmp
                         2 * sizeof(float) +                             // rstd/xReduceTmp
                         DOUBLE_BUFFER * sizeof(float) +                 // Scale1 * double
                         firstVcaddLength * sizeof(float);               // xTmp

    if (tilingParams.hasY2Scale2) {
        mutilBaseM += DOUBLE_BUFFER * baseNB8Align * sizeof(int8_t) + // y2 * double
                      baseNB32Align * sizeof(float) +                 // y2Tmp
                      DOUBLE_BUFFER * sizeof(float);                  // Scale2 * double
    }
    int64_t fullLoadBaseM = LastUbSize / mutilBaseM;
    uint64_t usedUbSize =
        CalUsedSize(fullLoadBaseM, baseNB8Align, baseNB32Align, baseNDtypeAlign, firstVcaddLength);
    while (usedUbSize > tilingParams.maxUbSize && fullLoadBaseM > 0) {
        fullLoadBaseM--;
        usedUbSize =
            CalUsedSize(fullLoadBaseM, baseNB8Align, baseNB32Align, baseNDtypeAlign, firstVcaddLength);
    }
    return fullLoadBaseM;
}

uint64_t AddRmsNormDynamicQuantRegbaseTiling::CalUsedSize(
    uint64_t baseM, uint64_t baseNB8Align, uint64_t baseNB32Align, uint64_t baseNDtypeAlign, 
    int64_t firstVcaddLength)
{
    uint64_t ubFactorRstd = Ops::Base::CeilAlign(baseM, static_cast<uint64_t>(B32_BLOCK_NUM));
    uint64_t totalSize = 0;
    totalSize += baseNDtypeAlign * tilingParams.xDtypeSize +
                 tilingParams.quantBufCnt * baseNDtypeAlign * tilingParams.xDtypeSize + ALIGN_SPACE;
    totalSize +=
        baseM * (3 * baseNDtypeAlign * tilingParams.xDtypeSize + baseNDtypeAlign * sizeof(float)); // x1/x2/xout + xTmp
    totalSize += baseM * (baseNB8Align * sizeof(int8_t) * DOUBLE_BUFFER +                          // y1 * double
                          baseNB32Align * sizeof(float));                                          // y1Tmp
    totalSize += 2 * ubFactorRstd * sizeof(float);                                                 // rstd/xReduceTmp
    totalSize += DOUBLE_BUFFER * ubFactorRstd * sizeof(float);                                     // Scale1 * double
    totalSize += baseM * firstVcaddLength * sizeof(float);                                         // xTmp
    if (tilingParams.hasY2Scale2) {
        totalSize += baseM * (baseNB8Align * sizeof(int8_t) * DOUBLE_BUFFER + // y2 * double
                              baseNB32Align * sizeof(float));                 // y2Tmp
        totalSize += DOUBLE_BUFFER * ubFactorRstd * sizeof(float);            // Scale2 * double
    }
    return totalSize;
}

ge::graphStatus AddRmsNormDynamicQuantRegbaseTiling::SetTilingParams()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormDynamicQuantRegbaseTiling SetTilingParams.");
    uint64_t tmpUBSize;
    tilingParams.powerLoop = 1;

    // 1. 全载模版切分修改 全载选择条件修改为新UB buffer分配方式 perf
    int64_t tmpPower = 0;
    int64_t fullLoadBaseM = CalFullLoadBaseM(tilingParams.numN, tmpPower);
    if (fullLoadBaseM >= 1 && tilingParams.numN <= FULL_LOAD_R_MAX) {
        tilingParams.baseN = tilingParams.numN;
        tilingParams.baseM = std::min(fullLoadBaseM, static_cast<int64_t>(tilingParams.mPerCore));
        tilingParams.powerSplit = tmpPower;
        tilingParams.tilingType = TILING_TYPE_PERF;
        return ge::GRAPH_SUCCESS;
    }

    // 2.全载性能模版未覆盖到的部分走原来的模版
    tmpUBSize = CalUBTotalSize(1, tilingParams.numN);
    if (tmpUBSize <= tilingParams.maxUbSize) {
        tilingParams.baseN = tilingParams.numN;
        uint64_t justNUBSize = CalUBTotalSize(0, tilingParams.baseN);
        uint64_t rstdCount = tilingParams.hasY2Scale2 ? 3 : 2; // rstd/scale1/scale2
        uint64_t rstdRemainUBSize = rstdCount * BLOCK_SIZE;
        // Note: CalUBTotalSize(M, N) can be see as:
        //       CalUBTotalSize(M, N) = rstdCount*AlignB32(M) + b*Align1(N) + c*M*Align2(N)
        //       CalUBTotalSize(1, N) = rstdCount*BLOCK_SIZE + b*Align1(N) + c*Align2(N)
        //       CalUBTotalSize(0, N) = b*Align1(N)
        // Note:
        //       rstdCount*M*sizeof(float) + b*Align1(N) + c*M*Align2(N) ~= UBSize - rstdCount*BLOCK_SIZE
        //       baseM ~= (UBSize - rstdCount*BLOCK_SIZE - b*Align1(N)) / (c*Align2(N) + rstdCount*sizeof(float))
        //       baseM ~= (UBSize - rstdCount*BLOCK_SIZE - CalUBTotalSize(1, N)) /
        //                (CalUBTotalSize(1, N) - rstdCount*BLOCK_SIZE - CalUBTotalSize(0, N) + rstdCount*sizeof(float))
        tilingParams.baseM = 1;
        if (rstdRemainUBSize + justNUBSize <= tilingParams.maxUbSize) {
            tilingParams.baseM = (tilingParams.maxUbSize - rstdRemainUBSize - justNUBSize) /
                                 (tmpUBSize - rstdRemainUBSize - justNUBSize + rstdCount * sizeof(float));
        }
        tilingParams.tilingType = TILING_TYPE_NORMAL;
        return ge::GRAPH_SUCCESS;
    }

    // 3. Cut n
    tmpUBSize = CalUBTotalSize(1, tilingParams.xReduceAlignNum, TILING_TYPE_SPILT);
    if (tmpUBSize <= tilingParams.maxUbSize) {
        uint64_t tmpPowerCutN = tilingParams.xReduceAlignNum;
        while (CalUBTotalSize(1, tmpPowerCutN * MULTI_FACTOR_2, TILING_TYPE_SPILT) <= tilingParams.maxUbSize) {
            tmpPowerCutN *= MULTI_FACTOR_2;
        }
        tilingParams.powerSplit = tmpPowerCutN;
        uint64_t tmpLoop = 1;
        while (tmpLoop * MULTI_FACTOR_2 * tilingParams.powerSplit <= tilingParams.numN) {
            tmpLoop *= MULTI_FACTOR_2;
        }
        tilingParams.powerLoop = tmpLoop;
        tilingParams.baseM = 1;
        tilingParams.baseN = tilingParams.powerSplit;
        tilingParams.tilingType = TILING_TYPE_SPILT;
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGE(nodeName.c_str(), "Can not find one tiling.");
    return ge::GRAPH_FAILED;
}

ge::graphStatus AddRmsNormDynamicQuantRegbaseTiling::DoOpTiling()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormDynamicQuantRegbaseTiling DoOpTiling.");
    if (tilingParams.needGetCompileInfo) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, tilingParams.maxUbSize);
        tilingParams.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    }

    tilingParams.mPerCore = Ops::Base::CeilDiv(tilingParams.numM, tilingParams.totalCoreNum);
    tilingParams.usedCoreNum = Ops::Base::CeilDiv(tilingParams.numM, tilingParams.mPerCore);
    tilingParams.mLastCore = tilingParams.numM - (tilingParams.usedCoreNum - 1) * tilingParams.mPerCore;

    ge::graphStatus res = SetTilingParams();
    OP_CHECK_IF(ge::GRAPH_SUCCESS != res, , return res);

    // Set align params
    tilingParams.baseNDtypeAlign = Ops::Base::CeilAlign(tilingParams.baseN, tilingParams.xDtypeAlignNum);
    tilingParams.baseNB8Align = Ops::Base::CeilAlign(tilingParams.baseN, static_cast<uint64_t>(BLOCK_SIZE));
    tilingParams.baseNReduceAlign = Ops::Base::CeilAlign(tilingParams.baseN, tilingParams.xReduceAlignNum);
    uint64_t reduceBufLen = tilingParams.baseNReduceAlign / (2 * tilingParams.vecLength);
    tilingParams.reduceBufLenAlign = Ops::Base::CeilAlign(reduceBufLen, static_cast<uint64_t>(B32_BLOCK_NUM));

    if (TILING_TYPE_NORMAL == tilingParams.tilingType) {
        uint64_t tmpPower = std::floor(std::log(tilingParams.baseNReduceAlign) / std::log(LOG_2));
        tilingParams.powerSplit = std::pow(LOG_2, tmpPower);
    }

    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void AddRmsNormDynamicQuantRegbaseTiling::SetTilingData()
{
    tilingData.set_numM(tilingParams.numM);
    tilingData.set_numN(tilingParams.numN);
    tilingData.set_baseM(tilingParams.baseM);
    tilingData.set_baseN(tilingParams.baseN);
    tilingData.set_baseNDtypeAlign(tilingParams.baseNDtypeAlign);
    tilingData.set_baseNReduceAlign(tilingParams.baseNReduceAlign);
    tilingData.set_powerSplit(tilingParams.powerSplit);
    tilingData.set_powerLoop(tilingParams.powerLoop);
    tilingData.set_mPerCore(tilingParams.mPerCore);
    tilingData.set_mLastCore(tilingParams.mLastCore);
    tilingData.set_epsilon(tilingParams.epsilon);
    tilingData.set_avgFactor(tilingParams.avgFactor);
}

void AddRmsNormDynamicQuantRegbaseTiling::PrintTilingData()
{
    OP_LOGI(
        nodeName.c_str(),
        "TilingData numM: %lu, numN: %lu, baseM: %lu, baseN: %lu, "
        "baseNDtypeAlign: %lu, baseNReduceAlign: %lu, powerSplit: %lu, powerLoop: %lu, "
        "mPerCore: %lu, mLastCore: %lu, "
        "epsilon: %f, avgFactor: %f.",
        tilingData.get_numM(), tilingData.get_numN(), tilingData.get_baseM(), tilingData.get_baseN(),
        tilingData.get_baseNDtypeAlign(), tilingData.get_baseNReduceAlign(), tilingData.get_powerSplit(),
        tilingData.get_powerLoop(), tilingData.get_mPerCore(), tilingData.get_mLastCore(), tilingData.get_epsilon(),
        tilingData.get_avgFactor());
}

ge::graphStatus AddRmsNormDynamicQuantRegbaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicQuantRegbaseTiling::GetWorkspaceSize()
{
    tilingParams.workspaceSize = 0;
    if (TILING_TYPE_SPILT == tilingParams.tilingType) {
        tilingParams.workspaceSize = WORKSPACE_COUNT * tilingParams.usedCoreNum * tilingParams.numN * sizeof(float);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicQuantRegbaseTiling::PostTiling()
{
    OP_LOGD(nodeName.c_str(), "Tiling usedCoreNum is %lu.", tilingParams.usedCoreNum);
    context_->SetBlockDim(tilingParams.usedCoreNum);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t usrWorkspaceSize = tilingParams.workspaceSize;
    size_t sysWorkSpaceSize = DEFAULT_SYS_WORKSPACE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrWorkspaceSize + sysWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

uint64_t AddRmsNormDynamicQuantRegbaseTiling::GetTilingKey() const
{
    uint64_t tilingKey = TILING_OFFSET_REGBASE;
    tilingKey += TILING_OFFSET_HAS_QUANT *
                 ((tilingParams.hasSmoothScale1 << SMOOTH_SCALE1_BIN_OFFSET) | tilingParams.hasSmoothScale2);
    tilingKey += tilingParams.tilingType;

    if (!tilingParams.needRun) {
        tilingKey = TILING_KEY_UNRUN;
    }
    OP_LOGD(nodeName.c_str(), "TilingKey is %lu.", tilingKey);
    return tilingKey;
}

} // namespace optiling
