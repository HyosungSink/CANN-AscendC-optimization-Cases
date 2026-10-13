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
 * \file group_norm_silu_tiling_arch35.cpp
 * \brief
 */
#include "group_norm_silu_tiling.h"
namespace optiling {
static const uint64_t X_SHAPE_MIN_LEN = 2;
static const uint64_t INPUT_IDX_X = 0;
static const uint64_t INPUT_IDX_GAMMA = 1;
static const uint64_t INPUT_IDX_BETA = 2;
static const uint64_t INDEX_NUM_GROUPS = 0;
static const uint64_t INDEX_EPSILON = 1;
static const uint64_t INDEX_ACTIVATE_SILU = 2;
static const uint64_t OUTPUT_IDX_Y = 0;
static const uint64_t OUTPUT_IDX_MEAN = 1;
static const uint64_t OUTPUT_IDX_RSTD = 2;
static const uint64_t EMPTY_SHAPE_SIZE = 0;
static const uint64_t DIM_0 = 0;
static const uint64_t DIM_1 = 1;
static const uint64_t DEFAULT_NUMGROUPS = 32;
static const uint64_t RESERVED_WORKSPACE_SIZE = 16 * 1024 * 1024;
static const uint64_t FLOAT32_BYTES = 4;
static const uint64_t FLOAT16_BYTES = 2;
static const uint64_t BLOCK_SIZE = 32;
static const uint64_t BUFFER_NUM = 2;
static const uint64_t DOUBLE_BUFFER = 2;
static const uint64_t MAX_CHANNEL_SIZE = 4096;
static const uint64_t MAX_NUM_PER_CORE = 2048;
static const uint64_t DICHOTOMY_ADD_COEFF = 2;
static const uint64_t ULONG_BIT_LEN = 64;
static const float DEFAULT_EPS = 1e-5;
static const bool DEFAULT_ACTIVATESILU = true;
static const uint64_t NDDMA_MAX_SIZE = 32; // 可根据实际测试结果更改此值

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

inline static int64_t DownAlign(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a / b) * b;
}

inline static int64_t RoundUp(int64_t a, int64_t b)
{
    return CeilDiv(a, b) * b;
}

static ge::graphStatus CheckMixRegBase(const gert::TilingContext* context)
{	
    auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();	
    if (compileInfo->isRegbase) {	
        auto out1Ptr = context->GetOutputDesc(OUTPUT_IDX_MEAN);	
        OP_CHECK_NULL_WITH_CONTEXT(context, out1Ptr);	
        auto out1Dtype = out1Ptr->GetDataType();	
        auto out2Ptr = context->GetOutputDesc(OUTPUT_IDX_RSTD);	
        OP_CHECK_NULL_WITH_CONTEXT(context, out2Ptr);	
        auto out2Dtype = out2Ptr->GetDataType();	
        if ((out1Dtype == out2Dtype) && (out1Dtype == ge::DT_FLOAT)) {	
            return ge::GRAPH_SUCCESS;	
        }
        return ge::GRAPH_FAILED;
    } else {
        return ge::GRAPH_SUCCESS;
    }
}

// GNS支持两种混合精度(bfloat16, float32, float32) (float16, float32, float32)
static ge::graphStatus CheckMixType(
    const gert::TilingContext* context, const ge::DataType& xDtype, const ge::DataType& gammaDtype)
{
    if (xDtype == gammaDtype) {
        return ge::GRAPH_SUCCESS;
    }

    if (xDtype == ge::DT_FLOAT16 && gammaDtype == ge::DT_FLOAT) {
        return CheckMixRegBase(context);
    }
    if (xDtype == ge::DT_BF16 && gammaDtype == ge::DT_FLOAT) {
        return CheckMixRegBase(context);
    }
    return ge::GRAPH_FAILED;
}

static ge::graphStatus CheckGammaAndBetaParams(
    const gert::TilingContext* context, const ge::DataType& xDtype, uint64_t channel)
{
    auto gammaShapePtr = context->GetOptionalInputShape(INPUT_IDX_GAMMA);
    auto gammaDtypePtr = context->GetOptionalInputDesc(INPUT_IDX_GAMMA);
    bool hasGamma = gammaShapePtr != nullptr;
    auto betaShapePtr = context->GetOptionalInputShape(INPUT_IDX_BETA);
    auto betaDtypePtr = context->GetOptionalInputDesc(INPUT_IDX_BETA);
    bool hasBeta = betaShapePtr != nullptr;
    if (!hasGamma && !hasBeta) {
        return ge::GRAPH_SUCCESS;
    }

    if (hasGamma && hasBeta) {
        auto gammaDtype = gammaDtypePtr->GetDataType();
        uint64_t gammaDtypeSize = ge::GetSizeByDataType(gammaDtype);
        auto betaDtype = betaDtypePtr->GetDataType();
        uint64_t betaDtypeSize = ge::GetSizeByDataType(betaDtype);
        OP_CHECK_IF(
            (gammaDtypeSize != betaDtypeSize),
            OP_LOGE(
                context->GetNodeType(),
                "The dtype of gamma and beta must be consistent, currently gamma is %d, beta is %d.", gammaDtype,
                betaDtype),
            return ge::GRAPH_FAILED);
    }

    if (hasGamma) {
        auto gammaShape = gammaShapePtr->GetStorageShape();
        uint64_t gammaSizes = gammaShape.GetDim(DIM_0);
        OP_CHECK_IF(
            (gammaShape.GetDimNum() != 1 || gammaSizes != channel),
            OP_LOGE(
                context->GetNodeType(),
                "Gamma dimension should be one, and the shape of gamma must be "
                "the same as channel size of input, currently is "
                "%lu.",
                gammaSizes),
            return ge::GRAPH_FAILED);
        auto gammaDtype = gammaDtypePtr->GetDataType();
        OP_CHECK_IF(
            (CheckMixType(context, xDtype, gammaDtype) == ge::GRAPH_FAILED),
            OP_LOGE(
                context->GetNodeType(), "The dtype combination of gamma, beta and inputs is invalid."),
            return ge::GRAPH_FAILED);
    }

    if (hasBeta) {
        auto betaShape = betaShapePtr->GetStorageShape();
        uint64_t betaSizes = betaShape.GetDim(DIM_0);
        OP_CHECK_IF(
            (betaShape.GetDimNum() != 1 || betaSizes != channel),
            OP_LOGE(
                context->GetNodeType(),
                "Beta dimension should be one, and the shape of beta must be "
                "the same as channel size of input, currently is "
                "%lu.",
                betaSizes),
            return ge::GRAPH_FAILED);
        auto betaDtype = betaDtypePtr->GetDataType();
        OP_CHECK_IF(
            (CheckMixType(context, xDtype, betaDtype) == ge::GRAPH_FAILED),
            OP_LOGE(
                context->GetNodeType(), "The dtype combination of gamma, beta and inputs is invalid."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputXShape(const gert::TilingContext* context, const gert::Shape& xShape)
{
    uint64_t xDims = xShape.GetDimNum();
    OP_CHECK_IF(
        (xDims < X_SHAPE_MIN_LEN),
        OP_LOGE(context->GetNodeType(), "inputDims can't be smaller than 2."),
        return ge::GRAPH_FAILED);
    for (uint64_t i = 0; i < xDims; i++) {
        int64_t curDim = xShape.GetDim(i);
        OP_CHECK_IF(
            (curDim < 0),
            OP_LOGE(
                context->GetNodeType(), "The input %lu dimension must be greater than 0, currently is %ld.", i, curDim),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputParams(const gert::TilingContext* context)
{
    // check x
    auto inputX = context->GetInputTensor(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto xDtype = context->GetInputDesc(INPUT_IDX_X)->GetDataType();
    uint64_t xDtypeSize = ge::GetSizeByDataType(xDtype);
    OP_CHECK_IF(
        (xDtypeSize <= 0),
        OP_LOGE(context->GetNodeType(), "xDtypeSize is invalid %lu, please check.", xDtypeSize),
        return ge::GRAPH_FAILED);
    auto xShape = inputX->GetStorageShape();
    uint64_t channel = xShape.GetDim(DIM_1);
    if (CheckInputXShape(context, xShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // check gamma and beta
    if (CheckGammaAndBetaParams(context, xDtype, channel) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext* context)
{
    auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
    uint64_t channel = xShape.GetDim(DIM_1);
    // check num_groups
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numGroups = attrs->GetAttrPointer<int64_t>(INDEX_NUM_GROUPS);
    OP_CHECK_IF(
        (*numGroups <= 0), OP_LOGE(context->GetNodeType(), "numGroups must be bigger than 0."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (channel % *numGroups != 0),
        OP_LOGE(context->GetNodeType(), "channel must be integer multiples of numGroups."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static bool isMixType(const gert::TilingContext* context)
{
    auto xDtype = context->GetInputDesc(INPUT_IDX_X)->GetDataType();
    uint64_t xDtypeSize = ge::GetSizeByDataType(xDtype);
    auto gammaDesc = context->GetOptionalInputDesc(INPUT_IDX_GAMMA);
    bool hasGamma = gammaDesc != nullptr;
    auto betaDesc = context->GetOptionalInputDesc(INPUT_IDX_BETA);
    bool hasBeta = betaDesc != nullptr;
    if (!hasGamma && !hasBeta) {
        return false;
    }
    if (hasGamma) {
        uint64_t gammaDtypeSize = ge::GetSizeByDataType(gammaDesc->GetDataType());
        if (gammaDtypeSize == xDtypeSize) {
            return false;
        }
    }
    if (hasBeta) {
        uint64_t betaDtypeSize = ge::GetSizeByDataType(betaDesc->GetDataType());
        if (betaDtypeSize == xDtypeSize) {
            return false;
        }
    }
    return true;
}

static void GetDichotomyAddParams(
    const gert::TilingContext* context, uint64_t r, uint64_t& power, uint64_t& dichotomyK, uint64_t& extraSize,
    uint64_t& lastNum)
{
    auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
    uint32_t vl = compileInfo->vectorLength / FLOAT32_BYTES;
    uint32_t blockSize = compileInfo->blockSizePlatform;
    uint64_t basePower = (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(r)));
    power = basePower == r ? basePower / DICHOTOMY_ADD_COEFF : basePower;
    uint64_t extraOriSize = power / vl;
    extraSize = RoundUp(extraOriSize * FLOAT32_BYTES, blockSize);
    dichotomyK = 0;
    if (extraOriSize < vl) {
        lastNum = extraOriSize;
        return;
    }
    uint64_t totalNum = extraOriSize / vl;
    uint64_t base = 1;
    lastNum = vl;
    while (base < totalNum) {
        dichotomyK++;
        base *= DICHOTOMY_ADD_COEFF;
    }
}

static uint64_t GetOptionalInputTensorSize(
    const gert::TilingContext* context, GroupNormSiluRegbaseTilingData& tilingData, uint64_t index,
    uint64_t specifiedValue = 0, bool useNddma = false)
{
    auto tensorDesc = context->GetOptionalInputDesc(index);
    if (tensorDesc == nullptr) {
        return 0;
    }
    auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
    uint32_t blockSize = compileInfo->blockSizePlatform;
    auto dtypeSize = ge::GetSizeByDataType(tensorDesc->GetDataType());
    if (specifiedValue != 0) {
        return RoundUp(specifiedValue * dtypeSize, blockSize);
    }

    auto storageShape = context->GetOptionalInputShape(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, storageShape);
    auto shape = storageShape->GetStorageShape();
    uint64_t num = 1;
    for (uint64_t i = 0; i < shape.GetDimNum(); i++) {
        num = num * shape.GetDim(i);
    }
    auto numUbSize = RoundUp(num * dtypeSize, blockSize);
    uint64_t hwNum = tilingData.get_hwNum();
    if (hwNum <= NDDMA_MAX_SIZE && useNddma) {
        int32_t xDtype = ge::GetSizeByDataType(context->GetInputDesc(INPUT_IDX_X)->GetDataType());
        if (xDtype == 0){
            OP_LOGE(context, "Division by zero!");
            return 0;
        }
        int64_t eleNumAlign = RoundUp(tilingData.get_elemNum(), blockSize / xDtype);
        return RoundUp(eleNumAlign * dtypeSize, blockSize) * tilingData.get_numGroups();
    }
    return numUbSize;
}

/*
regbase tilingKey划分逻辑如下:
1. UB内默认存放2048根A轴，gamma和beta在UB内全载，在开启DB条件下，计算出能够全载的最大R轴
    1.1 R轴小于可全载的最大R轴,则走TILINGKEY_TWOPASS_PERF性能模板
    1.2 尝试切分gamma和beta，每次拷入gamma和beta时，拷贝一个完整的D大小的数据
计算出新的可全载的最大R轴，如果大于R轴实际大小，则走TILINGKEY_TWOPASS_GENERALIZED泛化模板
    1.3
当R轴不能全载时，如果gamma和beta小于4096，则走TILINGKEY_WELFORD_PERF模板，否则走TILINGKEY_WELFORD_GENERALIZED模板
二分累加所需要的UB大小，计算规则如下:
1. 优先按照R轴可以全载计算所需要的额外空间
2. 如果R轴无法全载，则需要根据可用的UB空间，结合二分累加的额外空间，重新计算出最大的并行度
在非全载模板下，二分累加的UB额外空间不会影响normalize+swish阶段一次可载入的R轴大小
当Gamma或者Beta非空，并且和输入数据类型不一致时，认为是mix type场景
*/
static void SetTilingKey4Regbase(
    const gert::TilingContext* context, uint64_t& maxReduceCount, uint64_t& ubRemain, bool& isReduceFullLoad,
    GroupNormSiluRegbaseTilingData& tilingData)
{
    auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
    uint64_t ubSize = compileInfo->ubSizePlatForm;
    uint32_t blockSize = compileInfo->blockSizePlatform;
    uint64_t reduceCount = tilingData.get_shapeD() * tilingData.get_hwNum();
    uint64_t gammaUbSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_GAMMA, 0, true);
    uint64_t betaUbSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_BETA, 0, true);
    uint64_t realNumPerCore = std::min(
        MAX_NUM_PER_CORE, static_cast<uint64_t>(std::max(tilingData.get_numPerCore(), tilingData.get_numLastCore())));
    uint64_t meanUbSize = RoundUp(realNumPerCore * FLOAT32_BYTES, blockSize);
    uint64_t rstdUbSize = RoundUp(realNumPerCore * FLOAT32_BYTES, blockSize);

    uint64_t otherUbSize = gammaUbSize + betaUbSize + meanUbSize + rstdUbSize;
    uint64_t xDtypeSize = ge::GetSizeByDataType(context->GetInputDesc(INPUT_IDX_X)->GetDataType());
    uint64_t meanUbExtraSize = 0;
    uint64_t rstdUbExtraSize = 0;
    uint64_t outDtypeSize = xDtypeSize;
    if (context->GetInputDesc(INPUT_IDX_GAMMA) != nullptr) {
        outDtypeSize = ge::GetSizeByDataType(context->GetInputDesc(INPUT_IDX_GAMMA)->GetDataType());
    } else if (context->GetInputDesc(INPUT_IDX_BETA) != nullptr) {
        outDtypeSize = ge::GetSizeByDataType(context->GetInputDesc(INPUT_IDX_BETA)->GetDataType());
    }
    if (outDtypeSize != FLOAT32_BYTES) {
        meanUbExtraSize = RoundUp(realNumPerCore * xDtypeSize, blockSize);
        rstdUbExtraSize = RoundUp(realNumPerCore * xDtypeSize, blockSize);
        otherUbSize = otherUbSize + meanUbExtraSize + rstdUbExtraSize;
    }
    bool mixType = isMixType(context);
    uint64_t xShapeSize = context->GetInputShape(INPUT_IDX_X)->GetStorageShape().GetShapeSize();
    if (xShapeSize == EMPTY_SHAPE_SIZE) {
        int64_t tilingKey = mixType ? static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_EMPTY_TENSOR_MIX_TYPE) :
                                      static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_EMPTY_TENSOR);
        tilingData.set_tilingKey(tilingKey);
        return;
    }

    bool hasOptional = context->GetOptionalInputDesc(INPUT_IDX_GAMMA) != nullptr ||
                       context->GetOptionalInputDesc(INPUT_IDX_BETA) != nullptr;
    uint64_t dichotomyAddPower = 0;
    uint64_t dichotomyAddK = 0;
    uint64_t dichotomyAddExtraSize = 0;
    uint64_t dichotomyAddLastNum = 0;
    GetDichotomyAddParams(
        context, reduceCount, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize, dichotomyAddLastNum);
    otherUbSize += dichotomyAddExtraSize;

    ubRemain = ubSize <= otherUbSize ? 0 : ubSize - otherUbSize;
    if (xDtypeSize == 0){
        OP_LOGE(context, "Division by zero!");
        return;
    }
    maxReduceCount = (ubRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;

    if (maxReduceCount > reduceCount) {
        isReduceFullLoad = true;
        int64_t tilingKey = mixType ? static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_TWOPASS_PERF_MIX_TYPE) :
                                      static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_TWOPASS_PERF);
        tilingData.set_tilingKey(tilingKey);
        return;
    }
    // 如果使用了nddma之后, 无法走全载模板，则尝试使用move_align搬入gamma或者beta，然后再次尝试是否能采用R轴全载性能模板
    bool isUseNddma = hasOptional && (tilingData.get_hwNum() <= static_cast<int64_t>(NDDMA_MAX_SIZE));
    uint64_t gammaNewUbSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_GAMMA, 0, false);
    uint64_t betaNewUbSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_BETA, 0, false);
    if (isUseNddma) {
        otherUbSize = otherUbSize + (gammaUbSize - gammaNewUbSize) + (betaUbSize - betaNewUbSize);
        maxReduceCount = (ubRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;
        if (maxReduceCount > reduceCount) {
            isReduceFullLoad = true;
            int64_t tilingKey = mixType ?
                                    static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_TWOPASS_PERF_MIX_TYPE) :
                                    static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_TWOPASS_PERF);
            tilingData.set_tilingKey(tilingKey);
            return;
        }
    }
    gammaUbSize = gammaNewUbSize;
    betaUbSize = betaNewUbSize;
    otherUbSize = gammaUbSize + betaUbSize + meanUbSize + rstdUbSize;
    if (outDtypeSize != FLOAT32_BYTES) {
        otherUbSize = otherUbSize + meanUbExtraSize + rstdUbExtraSize;
    }
    otherUbSize += dichotomyAddExtraSize;
    ubRemain = ubSize <= otherUbSize ? 0 : ubSize - otherUbSize;
    bool isLargeChannel = static_cast<uint64_t>(tilingData.get_shapeC()) > MAX_CHANNEL_SIZE;
    uint64_t newUbRemain = ubRemain;
    // 对于Channel轴过大的场景，尝试将gamma大小限制为ShapeD，beta大小限制为shapeD，重新计算最大可全载的R轴
    // 如果此时仍然无法全载，则走channel过大的非全载模板
    if (isLargeChannel || reduceCount <= compileInfo->vectorLength / FLOAT32_BYTES) {
        uint64_t gammaSplitUbSize =
            GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_GAMMA, tilingData.get_shapeD());
        uint64_t betaSplitUbSize =
            GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_BETA, tilingData.get_shapeD());
        otherUbSize = otherUbSize - gammaUbSize - betaUbSize + gammaSplitUbSize + betaSplitUbSize;
        newUbRemain = ubSize <= otherUbSize ? 0 : ubSize - otherUbSize;
        uint64_t newMaxReduceCount = (newUbRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;
        if (newMaxReduceCount > reduceCount) {
            isReduceFullLoad = true;
            maxReduceCount = newMaxReduceCount;
            ubRemain = newUbRemain;
            int64_t tilingKey =
                mixType ? static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_TWOPASS_GENERALIZED_MIX_TYPE) :
                          static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_TWOPASS_GENERALIZED);
            tilingData.set_tilingKey(tilingKey);
            return;
        }
    }
    // R轴过大，无法走全载模版，此时需要将二分累加所需要的ub空间释放，重新更新ubRemain和newMaxReduceCount
    isReduceFullLoad = false;
    int64_t meanAndRstdSize = meanUbSize + rstdUbSize + meanUbExtraSize + rstdUbExtraSize;
    if (isLargeChannel && hasOptional) {
        ubRemain = ubSize - meanAndRstdSize;
        int64_t tilingKey = mixType ?
                                static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_WELFORD_GENERALIZED_MIX_TYPE) :
                                static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_WELFORD_GENERALIZED);
        tilingData.set_tilingKey(tilingKey);
    } else {
        gammaUbSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_GAMMA, 0, false);
        betaUbSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_BETA, 0, false);
        ubRemain = ubSize - meanAndRstdSize - gammaUbSize - betaUbSize;
        int64_t tilingKey = mixType ? static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_WELFORD_PERF_MIX_TYPE) :
                                      static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_WELFORD_PERF);
        tilingData.set_tilingKey(tilingKey);
    }
    maxReduceCount = (ubRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;
}

static void SetDichotomyAddParams(const gert::TilingContext* context, GroupNormSiluRegbaseTilingData& tilingData)
{
    uint64_t reduceCount = tilingData.get_shapeD() * tilingData.get_hwNum();
    uint64_t dichotomyAddPower = 0;
    uint64_t dichotomyAddK = 0;
    uint64_t dichotomyAddExtraSize = 0;
    uint64_t dichotomyAddLastNum = 0;
    GetDichotomyAddParams(
        context, reduceCount, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize, dichotomyAddLastNum);
    uint64_t powerOfTwoForReduce = (1L << (ULONG_BIT_LEN - __builtin_clzl(reduceCount)));
    tilingData.set_powerOfTwoForReduce(powerOfTwoForReduce);
    tilingData.set_dichotomyAddPower(dichotomyAddPower);
    tilingData.set_dichotomyAddK(dichotomyAddK);
    tilingData.set_dichotomyAddLastNum(dichotomyAddLastNum);
}

static void SetWelfordParallelN(
    const gert::TilingContext* context, uint64_t xDtypeSize, uint64_t ubRemain,
    GroupNormSiluRegbaseTilingData& tilingData)
{
    if (xDtypeSize == 0){
        OP_LOGE(context, "Division by zero!");
        return;
    }
    auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
    uint32_t blockSize = compileInfo->blockSizePlatform;
    uint32_t coeff = FLOAT32_BYTES / xDtypeSize;
    uint32_t totalNum = BUFFER_NUM * (coeff + 1);
    uint32_t welfordBase = blockSize / xDtypeSize;
    uint32_t maxParallelN = DownAlign((ubRemain / xDtypeSize) / totalNum, welfordBase);

    uint64_t dichotomyAddPower = 0;
    uint64_t dichotomyAddK = 0;
    uint64_t dichotomyAddExtraSize = 0;
    uint64_t dichotomyAddLastNum = 0;
    GetDichotomyAddParams(
        context, maxParallelN, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize, dichotomyAddLastNum);
    uint32_t ubCurUse =
        maxParallelN * BUFFER_NUM * xDtypeSize + dichotomyAddExtraSize + maxParallelN * BUFFER_NUM * FLOAT32_BYTES;
    while (ubCurUse > ubRemain) {
        maxParallelN -= welfordBase;
        GetDichotomyAddParams(
            context, maxParallelN, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize, dichotomyAddLastNum);
        ubCurUse =
            maxParallelN * BUFFER_NUM * xDtypeSize + dichotomyAddExtraSize + maxParallelN * BUFFER_NUM * FLOAT32_BYTES;
    }

    if (maxParallelN > tilingData.get_elemNum()) {
        maxParallelN = tilingData.get_elemNum();
        GetDichotomyAddParams(
            context, maxParallelN, dichotomyAddPower, dichotomyAddK, dichotomyAddExtraSize, dichotomyAddLastNum);
    }
    tilingData.set_dichotomyAddPower(dichotomyAddPower);
    tilingData.set_dichotomyAddK(dichotomyAddK);
    tilingData.set_dichotomyAddLastNum(dichotomyAddLastNum);
    tilingData.set_parallelN(maxParallelN);
}

static void SetUbTiling4TwoPass(
    const gert::TilingContext* context, GroupNormSiluRegbaseTilingData& tilingData, uint64_t maxReduceCount,
    uint32_t xDtypeSize)
{
    auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
    uint32_t blockSize = compileInfo->blockSizePlatform;
    uint64_t elemNum = tilingData.get_elemNum();
    if (xDtypeSize == 0){
        OP_LOGE(context, "Division by zero!");
        return;
    }
    uint64_t elemNumAlign = RoundUp(elemNum, blockSize / xDtypeSize);
    SetDichotomyAddParams(context, tilingData);
    if (elemNumAlign == 0){
        OP_LOGE(context, "Division by zero!");
        return;
    }
    uint64_t count = maxReduceCount / elemNumAlign;
    uint64_t processSize = count * elemNumAlign;
    tilingData.set_processSize(processSize);
}

static void SetUbTiling4WelfordPerf(
    const gert::TilingContext* context, GroupNormSiluRegbaseTilingData& tilingData, uint64_t maxReduceCount,
    uint32_t ubRemain, uint32_t xDtypeSize)
{
    auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
    uint32_t blockSize = compileInfo->blockSizePlatform;
    SetWelfordParallelN(context, xDtypeSize, ubRemain, tilingData);
    uint64_t loopNum = 0;
    uint64_t loopTail = 0;
    uint64_t processSize = 0;
    uint64_t innerLoopNum = 0;
    uint64_t innerLoopTail = 0;
    uint64_t hwNum = tilingData.get_hwNum();
    if (xDtypeSize == 0){
        OP_LOGE(context, "Division by zero!");
        return;
    }
    uint64_t hwNumAlign = RoundUp(hwNum, blockSize / xDtypeSize);
    if (hwNumAlign == 0){
        OP_LOGE(context, "Division by zero!");
        return;
    }
    uint64_t count = maxReduceCount / hwNumAlign;
    if (count >= 1) {
        loopNum = CeilDiv(tilingData.get_shapeD(), count);
        loopTail = (tilingData.get_shapeD() - (loopNum - 1) * count) * hwNumAlign;
        processSize = count * hwNumAlign;
        innerLoopNum = 1;
    } else {
        auto maxReduceCountDownAlign = DownAlign(maxReduceCount, blockSize / xDtypeSize);
        innerLoopNum = CeilDiv(hwNum, maxReduceCountDownAlign);
        innerLoopTail = hwNum - maxReduceCountDownAlign * (innerLoopNum - 1);
        processSize = maxReduceCountDownAlign;
        loopNum = tilingData.get_shapeD();
        loopTail = 1;
    }
    tilingData.set_loopNum(loopNum);
    tilingData.set_loopTail(loopTail);
    tilingData.set_processSize(processSize);
    tilingData.set_innerLoopNum(innerLoopNum);
    tilingData.set_innerLoopTail(innerLoopTail);
}
static void SetUbTiling4WelfordGeneralized(
    const gert::TilingContext* context, GroupNormSiluRegbaseTilingData& tilingData, uint32_t ubRemain,
    uint32_t xDtypeSize)
{
    auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
    uint32_t blockSize = compileInfo->blockSizePlatform;
    uint64_t loopNum = 0;
    uint64_t loopTail = 0;
    uint64_t processSize = 0;
    uint64_t innerLoopNum = 0;
    uint64_t innerLoopTail = 0;
    uint64_t hwNum = tilingData.get_hwNum();
    if (xDtypeSize == 0){
        OP_LOGE(context, "Division by zero!");
        return;
    }
    uint64_t hwNumAlign = RoundUp(hwNum, blockSize / xDtypeSize);
    uint64_t maxReduceCount = (ubRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;
    if (hwNumAlign == 0){
        OP_LOGE(context, "Division by zero!");
        return;
    }
    uint64_t count = maxReduceCount / hwNumAlign;
    uint64_t gammaRealSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_GAMMA, count);
    uint64_t betaRealSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_BETA, count);
    uint64_t curUbSize = gammaRealSize + betaRealSize + count * hwNumAlign * xDtypeSize * BUFFER_NUM * DOUBLE_BUFFER;
    while (curUbSize > ubRemain && count >= 1) {
        count--;
        uint64_t tmpGammaRealSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_GAMMA, count);
        uint64_t tmpBetaRealSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_BETA, count);
        curUbSize = tmpGammaRealSize + tmpBetaRealSize + count * hwNumAlign * xDtypeSize * BUFFER_NUM * DOUBLE_BUFFER;
    }
    if (count >= 1) {
        loopNum = CeilDiv(tilingData.get_shapeD(), count);
        loopTail = (tilingData.get_shapeD() - (loopNum - 1) * count) * hwNumAlign;
        processSize = count * hwNumAlign;
        innerLoopNum = 1;
        ubRemain = ubRemain - gammaRealSize - betaRealSize;
    } else {
        gammaRealSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_GAMMA, 1);
        betaRealSize = GetOptionalInputTensorSize(context, tilingData, INPUT_IDX_BETA, 1);
        ubRemain = ubRemain - gammaRealSize - betaRealSize;
        maxReduceCount = (ubRemain / (DOUBLE_BUFFER * BUFFER_NUM)) / xDtypeSize;
        uint64_t maxReduceCountDownAlign = DownAlign(maxReduceCount, blockSize / xDtypeSize);
        innerLoopNum = CeilDiv(hwNum, maxReduceCountDownAlign);
        innerLoopTail = hwNum - maxReduceCountDownAlign * (innerLoopNum - 1);
        processSize = maxReduceCountDownAlign;
        loopNum = tilingData.get_shapeD();
        loopTail = 1;
    }
    SetWelfordParallelN(context, xDtypeSize, ubRemain, tilingData);
    tilingData.set_loopNum(loopNum);
    tilingData.set_loopTail(loopTail);
    tilingData.set_processSize(processSize);
    tilingData.set_innerLoopNum(innerLoopNum);
    tilingData.set_innerLoopTail(innerLoopTail);
}

static void SetUbTiling4Welford(
    const gert::TilingContext* context, GroupNormSiluRegbaseTilingData& tilingData, uint64_t maxReduceCount,
    uint64_t ubRemain, uint32_t xDtypeSize)
{
    if (tilingData.get_tilingKey() == static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_WELFORD_PERF) ||
        tilingData.get_tilingKey() == static_cast<int64_t>(GroupNormSiluTilingKey::TILINGKEY_WELFORD_PERF_MIX_TYPE)) {
        return SetUbTiling4WelfordPerf(context, tilingData, maxReduceCount, ubRemain, xDtypeSize);
    }
    return SetUbTiling4WelfordGeneralized(context, tilingData, ubRemain, xDtypeSize);
}

static void SetUbTiling4Regbase(
    const gert::TilingContext* context, uint64_t maxReduceCount, uint64_t ubRemain, bool isReduceFullLoad,
    GroupNormSiluRegbaseTilingData& tilingData)
{
    auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
    int32_t ubSize = compileInfo->ubSizePlatForm;
    uint64_t xDtypeSize = ge::GetSizeByDataType(context->GetInputDesc(INPUT_IDX_X)->GetDataType());
    tilingData.set_ubSize(ubSize);

    uint64_t xShapeSize = context->GetInputShape(INPUT_IDX_X)->GetStorageShape().GetShapeSize();
    if (xShapeSize == EMPTY_SHAPE_SIZE) {
        return;
    }

    if (!isReduceFullLoad) {
        SetUbTiling4Welford(context, tilingData, maxReduceCount, ubRemain, xDtypeSize);
    } else {
        SetUbTiling4TwoPass(context, tilingData, maxReduceCount, xDtypeSize);
    }
}

static void SetTilingForRegbase(const gert::TilingContext* context, GroupNormSiluRegbaseTilingData& tilingData)
{
    uint64_t maxReduceCount = 0;
    uint64_t ubRemain = 0;
    bool reduceFullLoad = false;
    SetTilingKey4Regbase(context, maxReduceCount, ubRemain, reduceFullLoad, tilingData);
    SetUbTiling4Regbase(context, maxReduceCount, ubRemain, reduceFullLoad, tilingData);
}

static void SetAttrParams(const gert::TilingContext* context, GroupNormSiluRegbaseTilingData& tilingData)
{
    auto attrs = context->GetAttrs();
    const int64_t* numGroupsPtr = attrs->GetAttrPointer<int64_t>(INDEX_NUM_GROUPS);
    const float* epsilonPtr = attrs->GetAttrPointer<float>(INDEX_EPSILON);
    const bool* activateSiluPtr = attrs->GetAttrPointer<bool>(INDEX_ACTIVATE_SILU);
    float eps = epsilonPtr == nullptr ? DEFAULT_EPS : *epsilonPtr;
    bool activateSilu = activateSiluPtr == nullptr ? DEFAULT_ACTIVATESILU : *activateSiluPtr;
    tilingData.set_numGroups(*numGroupsPtr);
    tilingData.set_epsilon(eps);
    tilingData.set_activateSilu(activateSilu);
}

static void SetTilingParams(const gert::TilingContext* context, GroupNormSiluRegbaseTilingData& tilingData)
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
}

static void SetBlockTiling(const gert::TilingContext* context, GroupNormSiluRegbaseTilingData& tilingData)
{
    auto compileInfo = context->GetCompileInfo<GroupNormSiluCompileInfo>();
    auto xShape = context->GetInputShape(INPUT_IDX_X)->GetStorageShape();
    uint64_t shapeN = xShape.GetDim(DIM_0);
    tilingData.set_numPerCore(CeilDiv(shapeN * tilingData.get_numGroups(), compileInfo->totalCoreNum));
    tilingData.set_realCoreNum(CeilDiv(shapeN * tilingData.get_numGroups(), tilingData.get_numPerCore()));
    tilingData.set_numLastCore(
        shapeN * tilingData.get_numGroups() - tilingData.get_numPerCore() * (tilingData.get_realCoreNum() - 1));
    tilingData.set_shapeN(shapeN);
    uint64_t xShapeSize = xShape.GetShapeSize();
    if (xShapeSize == 0) {
        int64_t realCoreNum = tilingData.get_realCoreNum();
        tilingData.set_realCoreNum((realCoreNum == 0) ? compileInfo->totalCoreNum : realCoreNum);
    }
}

inline static ge::graphStatus GroupNormSiluSetTilingData(
    gert::TilingContext* context, GroupNormSiluRegbaseTilingData& tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckEmptyTensorParams(const gert::TilingContext* context)
{
    auto yOutputShape = context->GetInputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yOutputShape);
    auto yShapeSize = yOutputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(
        yShapeSize != 0,
        OP_LOGE(
            context->GetNodeName(), "y must be an empty tensor when x is empty, but total num of y is %ld.",
            yShapeSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4GroupNormSiluRegBase(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeType(), "Start running Tiling4GroupNormSilu.");
    // check input && attrs params
    OP_CHECK_IF(
        (CheckInputParams(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeType(), "InputParams is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (CheckAttrParams(context) != ge::GRAPH_SUCCESS),
        OP_LOGE(context->GetNodeType(), "AttrParams is invalid."), return ge::GRAPH_FAILED);
    auto xInputShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInputShape);
    uint64_t xShapeSize = xInputShape->GetStorageShape().GetShapeSize();
    if (xShapeSize == 0 && CheckEmptyTensorParams(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    GroupNormSiluRegbaseTilingData tilingData;
    SetAttrParams(context, tilingData);
    SetTilingParams(context, tilingData);
    // block tiling
    SetBlockTiling(context, tilingData);
    // ub tiling
    size_t sysWorkspaceSize = RESERVED_WORKSPACE_SIZE;
    SetTilingForRegbase(context, tilingData);
    OP_CHECK_IF(
        GroupNormSiluSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeType(), "GroupNormSiluSetTilingData set tiling data fail."),
        return ge::GRAPH_FAILED);
    OP_LOGI(
        context->GetNodeType(),
        "tilingData is numGroups:%ld, hwNum:%ld, elemNum:%ld, shapeN:%ld, shapeC:%ld, shapeD:%ld, realCoreNum:%ld, \
            numPerCore:%ld, numLastCore:%ld, processSize:%ld, loopNum:%ld, loopTail:%ld, innerLoopNum:%ld, \
            innerLoopTail:%ld, tilingKey:%ld, epsilon:%f, activateSilu:%ld, parallelN:%ld, ubSize:%ld, dichotomyAddPower:%ld, \
            dichotomyAddK:%ld, dichotomyAddLastNum:%ld, powerOfTwoForReduce:%ld",
        tilingData.get_numGroups(), tilingData.get_hwNum(), tilingData.get_elemNum(), tilingData.get_shapeN(),
        tilingData.get_shapeC(), tilingData.get_shapeD(), tilingData.get_realCoreNum(), tilingData.get_numPerCore(),
        tilingData.get_numLastCore(), tilingData.get_processSize(), tilingData.get_loopNum(), tilingData.get_loopTail(),
        tilingData.get_innerLoopNum(), tilingData.get_innerLoopTail(), tilingData.get_tilingKey(),
        tilingData.get_epsilon(), tilingData.get_activateSilu(), tilingData.get_parallelN(), tilingData.get_ubSize(),
        tilingData.get_dichotomyAddPower(), tilingData.get_dichotomyAddK(), tilingData.get_dichotomyAddLastNum(),
        tilingData.get_powerOfTwoForReduce());

    // block dim, tilingKey
    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling