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
 * \file dequant_rope_quant_kvcache_tiling.cpp
 * \brief
 */

#include "dequant_rope_quant_kvcache_tiling.h"

namespace {
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t FP32_BYTE = 4;
constexpr int64_t FP16_BYTE = 2;
constexpr int64_t INT8_BYTE = 1;
constexpr int64_t FP16_ONE_BLOCK_NUM = 16;
constexpr int64_t DOUBLE_BUFFER = 1;
constexpr int64_t QKV_TENSOR_NUM_FP16 = 1;
constexpr int64_t ROPE_TENSOR_NUM_FP16 = 2;
constexpr int64_t QUANT_TENSOR_NUM_FP32 = 4;
constexpr int64_t Q_OUT_TENSOR_NUM_FP16 = 1;
constexpr int64_t KV_OUT_TENSOR_NUM_FP32 = 2;
constexpr int32_t KV_NUM = 2;
constexpr int64_t QKV_NUM = 3;
constexpr int32_t CAST_KV_NUM = 2;
constexpr int32_t ONCE_INDICE_BYTE_SIZE = 512;
constexpr int32_t SIN_COS_NUM = 2;
constexpr int32_t QUANT_TENSOR_NUM = 4;
constexpr int32_t SIZE_SPLIT_Q = 0;
constexpr int32_t SIZE_SPLIT_K = 1;
constexpr int32_t SIZE_SPLIT_V = 2;
constexpr int32_t X_INPUT_INDEX = 0;
constexpr int32_t COS_INDEX = 1;
constexpr int32_t SIN_INDEX = 2;
constexpr int32_t KVCACHE_INPUT_INDEX = 3;
constexpr int32_t VCACHE_INPUT_INDEX = 4;
constexpr int32_t INDICES_INPUT_INDEX = 5;
constexpr int32_t QUANT_K_SCALE_INPUT_INDEX = 6;
constexpr int32_t QUANT_V_SCALE_INPUT_INDEX = 7;
constexpr int32_t QUANT_K_OFFSET_INPUT_INDEX = 8;
constexpr int32_t QUANT_V_OFFSET_INPUT_INDEX = 9;
constexpr int32_t WEIGHT_SCALES_INPUT_INDEX = 10;
constexpr int32_t ACTIVATION_INPUT_INDEX = 11;
constexpr int32_t BIAS_INPUT_INDEX = 12;
constexpr int32_t IFKV_INDEX = 3;
constexpr int32_t CACHE_MODE_IDX = 4;
constexpr int32_t KVCACHE_HEAD_DIM_INDEX = 2;
constexpr int32_t KVCACHE_HIDDEN_DIM_INDEX = 3;
constexpr uint32_t MINIMAL_WORKSPACE = static_cast<uint32_t>(16 * 1024 * 1024);
constexpr uint32_t KEY_WEIGHT_IF_KV_OUT = 1;
constexpr uint32_t KEY_WEIGHT_IF_QUANT_OFFSET = 10;
constexpr uint32_t KEY_WEIGHT_IF_RESHAPE_CACHE = 100;
constexpr uint32_t KEY_WEIGHT_IF_BIAS = 1000;
constexpr uint32_t KEY_WEIGHT_IF_AS = 10000;
constexpr uint32_t KEY_WEIGHT_IF_BIAS_DATATYPE = 100000;
constexpr int64_t HIDDEN_SIZE_LIMIT_LENGTH = 12800;
constexpr int64_t INPUT_X_DIM = 3;
constexpr int64_t INPUT_CACHE_DIM = 4;
constexpr int64_t BASE_2 = 2;
constexpr int64_t DATA_TYPE_FLOAT = 0;
constexpr int64_t DATA_TYPE_FLOAT16 = 1;
constexpr int64_t DATA_TYPE_INT32 = 2;
constexpr int64_t DATA_TYPE_BF16 = 3;

template <typename T1, typename T2>
inline auto UpAlign(const T1& a, const T2& b) -> T1
{
    if (b != 0) {
        return (a + b - 1) / b * b;
    }
    return a;
}

template <typename T>
static auto GetRem(const T& value1, const T& value2) -> T
{
    if (value2 == 0) {
        return value2;
    }
    return value1 % value2;
}

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
std::string Shape2String(const T& shape) {
  std::ostringstream oss;
  oss << "[";
  if (shape.GetDimNum() > 0) {
    for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
      oss << shape.GetDim(i) << ", ";
    }
    oss << shape.GetDim(shape.GetDimNum() - 1);
  }
  oss << "]";
  return oss.str();
}

struct TilingInfoDRQK {
    int64_t batch = 0;
    int64_t cacheSeqlen = 0;
    int64_t seqlen = 0;
    int64_t qHeadNum = 0;
    int64_t kvHeadNum = 0;
    int64_t hiddenSize = 0;
    int64_t hiddenBtyeSize = 0;
    int64_t hiddenSizeUpCastBtyeAlign = 0;
    int64_t hiddenSizeUpBtyeAlign = 0;
    int64_t hiddenSizeUpOutBtyeAlign = 0;
    int64_t inputBtyeSize = 0;
    int64_t indiceBtyeSize = 0;
    int64_t scalesBtyeSize = 0;
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    int64_t ifKVout = 0;
    int64_t onceS = 0;
    int64_t hasQuantOffset = 0;
    int64_t isPA = 0;
    int64_t hasBias = 0;
    int64_t hasAS = 0;
    int64_t biasDataType = 0;
    int64_t isDequant = 0;
    const int64_t* attrData;
};
} // namespace
using namespace std;

namespace optiling {

static void DQRKPrintParam(gert::TilingContext* context, DequantRopeQuantKvcacheTilingData& tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print DequantRopeQuantKvcache tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, ">>> qHeadNum:                %ld", tiling.get_qHeadNum());
    OP_LOGD(nodeName, ">>> kvHeadNum:                     %ld", tiling.get_kvHeadNum());
    OP_LOGD(nodeName, ">>> hiddenSize:               %ld", tiling.get_hiddenSize());
    OP_LOGD(nodeName, ">>> hiddenSizeFp32Align:             %ld", tiling.get_hiddenSizeFp32Align());
    OP_LOGD(nodeName, ">>> hiddenSizeFp16Align:            %ld", tiling.get_hiddenSizeFp16Align());
    OP_LOGD(nodeName, ">>> hiddenSizeInt8Align:              %ld", tiling.get_hiddenSizeInt8Align());
    OP_LOGD(nodeName, ">>> OnceUBMaxS:         %ld", tiling.get_OnceUBMaxS());
    OP_LOGD(nodeName, ">>> cacheSeqlen:         %ld", tiling.get_cacheSeqlen());
    OP_LOGD(nodeName, ">>> seqlen:         %ld", tiling.get_seqlen());
    OP_LOGD(nodeName, ">>> qHiddenSize:         %ld", tiling.get_qHiddenSize());
    OP_LOGD(nodeName, ">>> kHiddenSize:          %ld", tiling.get_kHiddenSize());
    OP_LOGD(nodeName, ">>> vHiddenSize:      %ld", tiling.get_vHiddenSize());
    OP_LOGD(nodeName, ">>> realCoreNum:           %ld", tiling.get_realCoreNum());
    OP_LOGD(nodeName, ">>> frontCoreNum:          %ld", tiling.get_frontCoreNum());
    OP_LOGD(nodeName, ">>> blockFactor:        %ld", tiling.get_blockFactor());
    OP_LOGD(nodeName, ">>> tailCoreBlockFactor:   %ld", tiling.get_tailCoreBlockFactor());
    OP_LOGD(nodeName, ">>> hasQuantOffset:              %ld", tiling.get_hasQuantOffset());
    OP_LOGD(nodeName, ">>> ifKVout:              %ld", tiling.get_ifKVout());
    OP_LOGD(nodeName, ">>> isPA:              %ld", tiling.get_isPA());
    OP_LOGD(nodeName, ">>> hasBias:              %ld", tiling.get_hasBias());
    OP_LOGD(nodeName, ">>> hasAS:              %ld", tiling.get_hasAS());
    OP_LOGD(nodeName, ">>> batch:              %ld", tiling.get_batch());
}
ge::graphStatus checkOptDtype(const uint64_t index, gert::TilingContext* context, const ge::char_t* nodeName_)
{
    auto optDesc = context->GetOptionalInputDesc(index);
    if (optDesc != nullptr) {
        auto optDtype = optDesc->GetDataType();
        OP_CHECK_IF(
            optDtype != ge::DT_FLOAT,
            OP_LOGE(nodeName_, "input %lu 's dtype should be float.", index),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus checkDRQKDim(
    const gert::StorageShape* inShape, const std::vector<int64_t>& realShape, const ge::char_t* nodeName_,
    std::string logName)
{
    OP_CHECK_IF(
        inShape->GetStorageShape().GetDimNum() != realShape.size(),
        OP_LOGE(
            nodeName_, "%s's dim size should be %zu but got %zu", logName.c_str(), realShape.size(),
            inShape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    for (uint64_t dimI = 0; dimI < realShape.size(); dimI++) {
        OP_CHECK_IF(
            inShape->GetStorageShape().GetDim(dimI) != realShape[dimI],
            OP_LOGE(
                nodeName_, "%s's dim[%lu] should be [%ld] but got %ld.", logName.c_str(), dimI, realShape[dimI],
                inShape->GetStorageShape().GetDim(dimI)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool CheckPaCacheMode(const gert::RuntimeAttrs* attrs)
{
    const char* tmpmode = attrs->GetStr(CACHE_MODE_IDX);
    if (tmpmode == nullptr) {
        return false;
    }
    std::string cacheMode = tmpmode;
    std::transform(cacheMode.begin(), cacheMode.end(), cacheMode.begin(), ::tolower);
    return (cacheMode == "page");
}

ge::graphStatus getDRQKShapeInfo(gert::TilingContext* context, TilingInfoDRQK& tilingInfoDRQK)
{
    auto platformInfo = context->GetPlatformInfo();
    auto nodeName = context->GetNodeName();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    tilingInfoDRQK.coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingInfoDRQK.ubSize = ubSize;

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto attr = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    OP_CHECK_IF(
        attr->GetSize() != QKV_NUM,
        OP_LOGE(
            context->GetNodeName(), "the attr size_splits's size %zu is invalid, it should be 3", attr->GetSize()),
        return ge::GRAPH_FAILED);
    tilingInfoDRQK.attrData = reinterpret_cast<const int64_t*>(attr->GetData());
    int64_t qHiddensize = tilingInfoDRQK.attrData[0];
    int64_t kHiddensize = tilingInfoDRQK.attrData[1];
    int64_t vHiddensize = tilingInfoDRQK.attrData[2];
    int64_t qkvHiddensize = qHiddensize + kHiddensize + vHiddensize;
    OP_CHECK_IF(
        kHiddensize != vHiddensize,
        OP_LOGE(
            context->GetNodeName(), "the attr size_splits's kHiddensize %ld is not same with vHiddensize %ld",
            kHiddensize, vHiddensize),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (qHiddensize < 0) || (kHiddensize < 0),
        OP_LOGE(
            context->GetNodeName(), "size_splits[0] %ld size_splits[1] %ld should not less than 0", qHiddensize,
            kHiddensize),
        return ge::GRAPH_FAILED);
    tilingInfoDRQK.ifKVout = *attrs->GetAttrPointer<bool>(IFKV_INDEX) == true ? 1 : 0;

    auto quantKOffsetDesc = context->GetOptionalInputDesc(QUANT_K_OFFSET_INPUT_INDEX);
    tilingInfoDRQK.hasQuantOffset = (quantKOffsetDesc == nullptr) ? 0 : 1;
    auto quantVOffsetDesc = context->GetOptionalInputDesc(QUANT_V_OFFSET_INPUT_INDEX);
    tilingInfoDRQK.hasQuantOffset = (quantVOffsetDesc == nullptr) ? 0 : tilingInfoDRQK.hasQuantOffset;
    for (uint64_t indexOpt = QUANT_K_OFFSET_INPUT_INDEX; indexOpt < BIAS_INPUT_INDEX; indexOpt++) {
        if (checkOptDtype(indexOpt, context, nodeName) == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    }
    auto optDesc = context->GetOptionalInputDesc(BIAS_INPUT_INDEX);
    if (optDesc != nullptr) {
        auto optDtype = optDesc->GetDataType();
        OP_CHECK_IF(
            (optDtype != ge::DT_FLOAT && optDtype != ge::DT_FLOAT16 && optDtype != ge::DT_BF16 &&
             optDtype != ge::DT_INT32),
            OP_LOGE(nodeName, "input bias 's dtype should be in{float float16 bfloat16 int32}"),
            return ge::GRAPH_FAILED);
        if (optDtype == ge::DT_FLOAT) {
            tilingInfoDRQK.biasDataType = DATA_TYPE_FLOAT;
        } else if (optDtype == ge::DT_FLOAT16) {
            tilingInfoDRQK.biasDataType = DATA_TYPE_FLOAT16;
        } else if (optDtype == ge::DT_INT32) {
            tilingInfoDRQK.biasDataType = DATA_TYPE_INT32;
        } else if (optDtype == ge::DT_BF16) {
            tilingInfoDRQK.biasDataType = DATA_TYPE_BF16;
        } else {
            tilingInfoDRQK.biasDataType = DATA_TYPE_FLOAT;
        }
    }
    auto bias = context->GetOptionalInputDesc(BIAS_INPUT_INDEX);
    tilingInfoDRQK.hasBias = (bias == nullptr) ? 0 : 1;

    auto as = context->GetOptionalInputDesc(ACTIVATION_INPUT_INDEX);
    tilingInfoDRQK.hasAS = (as == nullptr) ? 0 : 1;

    const gert::StorageShape* xShape = context->GetInputShape(X_INPUT_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    int64_t xShapeDimNum = xShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        (xShapeDimNum != INPUT_X_DIM && xShapeDimNum != BASE_2),
        OP_LOGE(context->GetNodeName(), "x's dim [%ld] is not 3D or 2D.", xShapeDimNum),
        return ge::GRAPH_FAILED);
    tilingInfoDRQK.batch = xShape->GetStorageShape().GetDim(0);
    tilingInfoDRQK.seqlen = (xShapeDimNum == BASE_2) ? 1 : xShape->GetStorageShape().GetDim(1);
    tilingInfoDRQK.isPA = CheckPaCacheMode(attrs);

    OP_CHECK_IF(
        xShape->GetStorageShape().GetDim(xShapeDimNum - 1) != qkvHiddensize,
        OP_LOGE(
            context->GetNodeName(), "x's dim[-1] [%ld] should be [%ld].",
            xShape->GetStorageShape().GetDim(xShapeDimNum - 1), qkvHiddensize),
        return ge::GRAPH_FAILED);
    const gert::StorageShape* cacheShape = context->GetInputShape(KVCACHE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheShape);

    OP_CHECK_IF(
        cacheShape->GetStorageShape().GetDimNum() != INPUT_CACHE_DIM,
        OP_LOGE(
            context->GetNodeName(), "cacheK's dim [%zu] is not 4D.", cacheShape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);

    int64_t xHiddenSize = xShape->GetStorageShape().GetDim(xShapeDimNum - 1);

    tilingInfoDRQK.cacheSeqlen = cacheShape->GetStorageShape().GetDim(1);
    tilingInfoDRQK.kvHeadNum = cacheShape->GetStorageShape().GetDim(KVCACHE_HEAD_DIM_INDEX);
    tilingInfoDRQK.hiddenSize = cacheShape->GetStorageShape().GetDim(KVCACHE_HIDDEN_DIM_INDEX);
    int64_t quantShapeSize = tilingInfoDRQK.kvHeadNum * tilingInfoDRQK.hiddenSize;
    OP_CHECK_IF(
        cacheShape->GetStorageShape().GetDim(KVCACHE_HEAD_DIM_INDEX) * tilingInfoDRQK.hiddenSize != kHiddensize,
        OP_LOGE(
            context->GetNodeName(), "cacheK's headdim * hiddenSize should same with [%ld] but got [%ld].", kHiddensize,
            cacheShape->GetStorageShape().GetDim(KVCACHE_HEAD_DIM_INDEX) * tilingInfoDRQK.hiddenSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        tilingInfoDRQK.hiddenSize % FP16_ONE_BLOCK_NUM != 0,
        OP_LOGE(
            context->GetNodeName(), "hiddenSize [%ld] should be a multiple of 16", tilingInfoDRQK.hiddenSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        qHiddensize % tilingInfoDRQK.hiddenSize != 0,
        OP_LOGE(
            context->GetNodeName(), "qhiddenSize [%ld] should be a multiple of hiddenSize", qHiddensize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        vHiddensize % tilingInfoDRQK.hiddenSize != 0,
        OP_LOGE(
            context->GetNodeName(), "vhiddenSize [%ld] should be a multiple of hiddenSize", vHiddensize),
        return ge::GRAPH_FAILED);
    tilingInfoDRQK.qHeadNum = GetDiv(tilingInfoDRQK.attrData[0], tilingInfoDRQK.hiddenSize);
    const gert::StorageShape* cosShape = context->GetInputShape(COS_INDEX);
    const gert::StorageShape* sinShape = context->GetInputShape(SIN_INDEX);
    const gert::StorageShape* vCacheShape = context->GetInputShape(VCACHE_INPUT_INDEX);
    const gert::StorageShape* indicesShape = context->GetInputShape(INDICES_INPUT_INDEX);
    const gert::StorageShape* kScaleShape = context->GetInputShape(QUANT_K_SCALE_INPUT_INDEX);
    const gert::StorageShape* vScaleShape = context->GetInputShape(QUANT_V_SCALE_INPUT_INDEX);

    std::vector<int64_t> expSinCos = {tilingInfoDRQK.batch, tilingInfoDRQK.seqlen, 1, tilingInfoDRQK.hiddenSize};
    std::vector<int64_t> expX = {tilingInfoDRQK.batch, tilingInfoDRQK.seqlen, qkvHiddensize};

    if (xShapeDimNum == BASE_2) {
        expSinCos = {tilingInfoDRQK.batch, tilingInfoDRQK.hiddenSize};
        expX = {tilingInfoDRQK.batch, qkvHiddensize};
    }

    if (tilingInfoDRQK.isPA) {
        tilingInfoDRQK.batch = tilingInfoDRQK.batch * tilingInfoDRQK.seqlen;
        tilingInfoDRQK.seqlen = 1;
    }
    if ((checkDRQKDim(xShape, expX, nodeName, "input x") == ge::GRAPH_FAILED) ||
        (checkDRQKDim(cosShape, expSinCos, nodeName, "input cos") == ge::GRAPH_FAILED) ||
        (checkDRQKDim(sinShape, expSinCos, nodeName, "input sin") == ge::GRAPH_FAILED) ||
        (checkDRQKDim(indicesShape, {tilingInfoDRQK.batch}, nodeName, "input indices") == ge::GRAPH_FAILED)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        cacheShape->GetStorageShape() != vCacheShape->GetStorageShape(),
        OP_LOGE(
            context->GetNodeName(), "kcache shape:%s should be equal with vcache shape:%s",
            Shape2String(cacheShape->GetStorageShape()).c_str(),
            Shape2String(vCacheShape->GetStorageShape()).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (kScaleShape->GetStorageShape().GetShapeSize() != quantShapeSize ||
         vScaleShape->GetStorageShape().GetShapeSize() != quantShapeSize),
        OP_LOGE(
            context->GetNodeName(),
            "kScaleShape shape size:%ld or vScaleShape shape size:%ld should be equal with :%ld",
            kScaleShape->GetStorageShape().GetShapeSize(), vScaleShape->GetStorageShape().GetShapeSize(),
            quantShapeSize),
        return ge::GRAPH_FAILED);
    if (!tilingInfoDRQK.isPA) {
        OP_CHECK_IF(
            vCacheShape->GetStorageShape().GetDim(0) < xShape->GetStorageShape().GetDim(0),
            OP_LOGE(
                context->GetNodeName(), "vcache's dim[0]:%ld should not be less than input x's dim[0]:%ld",
                vCacheShape->GetStorageShape().GetDim(0), xShape->GetStorageShape().GetDim(0)),
            return ge::GRAPH_FAILED);
    }

    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    tilingInfoDRQK.isDequant = xDesc->GetDataType() == ge::DT_INT32;

    auto indiceDesc = context->GetInputDesc(INDICES_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indiceDesc);
    auto cosDesc = context->GetInputDesc(COS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cosDesc);
    bool ifBfloat16 = cosDesc->GetDataType() == ge::DT_BF16;

    tilingInfoDRQK.inputBtyeSize = ifBfloat16 ? FP32_BYTE : ge::GetSizeByDataType(cosDesc->GetDataType());
    tilingInfoDRQK.inputBtyeSize = tilingInfoDRQK.isDequant ? FP32_BYTE : tilingInfoDRQK.inputBtyeSize;
    tilingInfoDRQK.indiceBtyeSize = ge::GetSizeByDataType(indiceDesc->GetDataType());
    tilingInfoDRQK.hiddenBtyeSize = tilingInfoDRQK.hiddenSize * tilingInfoDRQK.inputBtyeSize;
    auto scaleDesc = context->GetInputDesc(QUANT_K_SCALE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleDesc);
    if (tilingInfoDRQK.hasQuantOffset == 1) {
        const gert::StorageShape* kOffsetShape = context->GetOptionalInputShape(QUANT_K_OFFSET_INPUT_INDEX);
        const gert::StorageShape* vOffsetShape = context->GetOptionalInputShape(QUANT_V_OFFSET_INPUT_INDEX);
        OP_CHECK_IF(
            (kOffsetShape->GetStorageShape().GetShapeSize() != quantShapeSize ||
             vOffsetShape->GetStorageShape().GetShapeSize() != quantShapeSize),
            OP_LOGE(
                context->GetNodeName(),
                "kOffsetShape shape size:%ld or vOffsetShape shape size:%ld should be equal with :%ld",
                kOffsetShape->GetStorageShape().GetShapeSize(), vOffsetShape->GetStorageShape().GetShapeSize(),
                quantShapeSize),
            return ge::GRAPH_FAILED);
        tilingInfoDRQK.scalesBtyeSize = ge::GetSizeByDataType(scaleDesc->GetDataType());
    }
    auto wsDesc = context->GetOptionalInputDesc(WEIGHT_SCALES_INPUT_INDEX);
    if (tilingInfoDRQK.isDequant) {
        OP_CHECK_NULL_WITH_CONTEXT(context, wsDesc);
    }
    tilingInfoDRQK.hiddenSizeUpBtyeAlign = UpAlign(tilingInfoDRQK.hiddenBtyeSize, BLOCK_SIZE);
    tilingInfoDRQK.hiddenSizeUpOutBtyeAlign = UpAlign(tilingInfoDRQK.hiddenSize * FP32_BYTE, BLOCK_SIZE);

    tilingInfoDRQK.hiddenSizeUpCastBtyeAlign = UpAlign(tilingInfoDRQK.hiddenSize * INT8_BYTE, BLOCK_SIZE);
    int64_t onceSeqInDtypeHNum = tilingInfoDRQK.kvHeadNum * KV_NUM + tilingInfoDRQK.qHeadNum + SIN_COS_NUM;

    // outque: kvq inque: kvq cos sin indice tmpbuffer
    int64_t onceSSize = tilingInfoDRQK.kvHeadNum * tilingInfoDRQK.hiddenSizeUpCastBtyeAlign * KV_NUM +
                        tilingInfoDRQK.qHeadNum * tilingInfoDRQK.hiddenSizeUpBtyeAlign +
                        onceSeqInDtypeHNum * DOUBLE_BUFFER * tilingInfoDRQK.hiddenSizeUpBtyeAlign +
                        tilingInfoDRQK.indiceBtyeSize * DOUBLE_BUFFER +
                        tilingInfoDRQK.kvHeadNum * tilingInfoDRQK.hiddenSizeUpOutBtyeAlign +
                        tilingInfoDRQK.hasAS * FP32_BYTE;

    int64_t remainSize =
        BLOCK_SIZE * DOUBLE_BUFFER + ONCE_INDICE_BYTE_SIZE +
        KV_NUM * tilingInfoDRQK.kvHeadNum * tilingInfoDRQK.hiddenSizeUpOutBtyeAlign +
        KV_NUM * tilingInfoDRQK.kvHeadNum * tilingInfoDRQK.hiddenSizeUpOutBtyeAlign * tilingInfoDRQK.hasQuantOffset +
        tilingInfoDRQK.hasBias * qkvHiddensize * FP32_BYTE + tilingInfoDRQK.isDequant * qkvHiddensize * FP32_BYTE;

    int64_t unUsedSize = tilingInfoDRQK.ubSize - remainSize;

    OP_CHECK_IF(
        (unUsedSize <= 0),
        OP_LOGE(
            context->GetNodeName(), "The elements num of x dim[-1] [%ld] should be less than [%ld].", xHiddenSize,
            HIDDEN_SIZE_LIMIT_LENGTH),
        return ge::GRAPH_FAILED);

    tilingInfoDRQK.onceS = GetDiv(unUsedSize, onceSSize);

    OP_CHECK_IF(
        (tilingInfoDRQK.onceS == 0),
        OP_LOGE(
            context->GetNodeName(), "The elements num of x dim[-1] [%ld] should be less than [%ld].", xHiddenSize,
            HIDDEN_SIZE_LIMIT_LENGTH),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingDequantRopeQuantKvcache(gert::TilingContext* context)
{
    TilingInfoDRQK tilingInfoDRQK;
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Tiling start.");

    if (getDRQKShapeInfo(context, tilingInfoDRQK) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    int64_t taskNum = tilingInfoDRQK.batch * tilingInfoDRQK.seqlen;
    int64_t taskNumRem = GetRem(taskNum, tilingInfoDRQK.coreNum);
    int64_t frontCoreNum = taskNumRem != 0 ? taskNumRem : tilingInfoDRQK.coreNum;
    int64_t tailCoreNum = taskNum <= tilingInfoDRQK.coreNum ? 0 : tilingInfoDRQK.coreNum - frontCoreNum;
    int64_t numBlocks = frontCoreNum + tailCoreNum;

    int64_t blockFactor = GetCeilInt(taskNum, numBlocks);
    int64_t lastCoreNum = GetDiv(taskNum, numBlocks);

    DequantRopeQuantKvcacheTilingData tiling;
    tiling.set_cacheSeqlen(tilingInfoDRQK.cacheSeqlen);
    tiling.set_seqlen(tilingInfoDRQK.seqlen);

    tiling.set_qHeadNum(tilingInfoDRQK.qHeadNum);
    tiling.set_kvHeadNum(tilingInfoDRQK.kvHeadNum);
    tiling.set_hiddenSize(tilingInfoDRQK.hiddenSize);
    tiling.set_hiddenSizeFp16Align(tilingInfoDRQK.hiddenSizeUpBtyeAlign);
    tiling.set_hiddenSizeFp32Align(tilingInfoDRQK.hiddenSizeUpCastBtyeAlign);
    tiling.set_hiddenSizeInt8Align(tilingInfoDRQK.hiddenSizeUpOutBtyeAlign);
    tiling.set_OnceUBMaxS(tilingInfoDRQK.onceS);

    tiling.set_qHiddenSize(tilingInfoDRQK.attrData[SIZE_SPLIT_Q]);
    tiling.set_kHiddenSize(tilingInfoDRQK.attrData[SIZE_SPLIT_K]);
    tiling.set_vHiddenSize(tilingInfoDRQK.attrData[SIZE_SPLIT_V]);
    tiling.set_frontCoreNum(frontCoreNum);

    tiling.set_realCoreNum(numBlocks);
    tiling.set_blockFactor(blockFactor);
    tiling.set_tailCoreBlockFactor(lastCoreNum);
    tiling.set_hasQuantOffset(tilingInfoDRQK.hasQuantOffset);
    tiling.set_ifKVout(tilingInfoDRQK.ifKVout);
    tiling.set_isPA(tilingInfoDRQK.isPA);
    tiling.set_hasBias(tilingInfoDRQK.hasBias);
    tiling.set_hasAS(tilingInfoDRQK.hasAS);
    tiling.set_batch(tilingInfoDRQK.batch);
    uint32_t tilingKey = tilingInfoDRQK.biasDataType;

    context->SetBlockDim(numBlocks);
    context->SetTilingKey(tilingKey);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);

    currentWorkspace[0] = MINIMAL_WORKSPACE;

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    DQRKPrintParam(context, tiling);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForDequantRopeQuantKvcache(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DequantRopeQuantKvcache)
    .Tiling(TilingDequantRopeQuantKvcache)
    .TilingParse<DequantRopeQuantKvcacheCompileInfo>(TilingPrepareForDequantRopeQuantKvcache);
} // namespace optiling
