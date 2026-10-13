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
 * \file weight_quant_batch_matmul_v2_weight_nz_tiling.cpp
 * \brief
 */
#include "weight_quant_batch_matmul_v2_weight_nz_tiling.h"

#include "op_cache_tiling.h"
#include "matmul/common/op_host/math_util.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "op_host/tiling_key.h"
#include "platform/platform_infos_def.h"
#include "../../op_kernel/weight_quant_batch_matmul_v2_kernel_tiling_key.h"

using Ops::NN::Optiling::RecursiveSum;
using Ops::NN::GenTiling;
using Ops::NN::TilingPrepareForOpCache;

namespace {
constexpr uint64_t MAX_SHAPE_DIM = 65535UL;
constexpr size_t VALID_INPUT_DIM_NUM = 2UL;
constexpr uint64_t BLOCK_REDUCE_INT8 = 32UL;
constexpr size_t MM_SHAPE_LEN_ND = 2UL;
constexpr size_t MM_SHAPE_LEN_NZ = 4UL;
static const int32_t ANTI_QUANT_TENSOR = 2;
static const int32_t DB_BUFFER = 2;
constexpr size_t QUANT_SCALE_INDEX = 4UL;
constexpr size_t QUANT_OFFSET_INDEX = 5UL;
constexpr size_t BIAS_INDEX = 6UL;
static const uint64_t MAX_BLOCK_STRIDE = 256;
static const uint64_t MAX_NBUB_SIZE = 128;
static const uint64_t ANTIQUANT_BUF_NUM = 74;
static const uint64_t BUB_SIZE_ALIGN = 32;
static const uint64_t BUB_BUF_NUM = 6;
static const uint64_t AUB_SIZE_ALIGN = 128 * 128;
static const uint64_t AUB_BUF_NUM = 8;

constexpr uint64_t MM_SHAPE_LEN_ND_MIN = 2;
constexpr uint64_t MM_SHAPE_LEN_ND_MAX = 6;
constexpr uint64_t MM_SHAPE_LEN_NZ_MIN = 4;
constexpr uint64_t MM_SHAPE_LEN_NZ_MAX = 8;

constexpr uint64_t NO_BATCH_SHAPE_DIM = 2;
constexpr uint64_t ONE_BATCH_SHAPE_DIM = 3;
constexpr uint64_t TWO_BATCH_SHAPE_DIM = 4;
constexpr uint64_t THREE_BATCH_SHAPE_DIM = 5;
constexpr uint64_t FOUR_BATCH_SHAPE_DIM = 6;
constexpr uint64_t ND_NZ_DIM_DIFF = 2;

class WhiteListShape
{
public:
    bool operator<(const WhiteListShape& right) const
    {
        return memcmp(this, &right, sizeof(WhiteListShape)) < 0;
    }

    uint64_t mSize_;
    uint64_t nSize_;
    uint64_t kSize_;
};

class WeightNzTilingCache
{
public:
    uint8_t cubeBlockDimN_;
    uint8_t cubeBlockDimM_;
    uint16_t aL1Pingpong_;
    uint16_t bL1Pingpong_;
    uint64_t kAlign_;
    uint64_t mSize_;
    uint64_t kSize_;
    uint64_t nSize_;
    uint64_t mAubSize_;
    uint64_t kAubSize_;
    uint64_t nBubSize_;
    uint64_t kBubSize_;
    uint64_t mCubSize_;
    uint64_t nCubSize_;
    uint64_t mAL1Size_;
    uint64_t kAL1Size_;
    uint64_t nBL1Size_;
    uint64_t kBL1Size_;
    uint64_t groupSize_;

    // matmul字段
    int32_t m_;
    int32_t n_;
    int32_t ka_;
    int32_t kb_;
    int32_t singleCoreM_;
    int32_t singleCoreN_;
    int32_t singleCoreK_;
    int32_t baseM_;
    int32_t baseN_;
    int32_t baseK_;
    int32_t depthA1_;
    int32_t depthB1_;
    int32_t stepM_;
    int32_t stepN_;
    int32_t transLength_;
    int32_t iterateOrder_;
    int32_t shareMode_;
    int32_t shareL1Size_;
    int32_t shareL0CSize_;
    int32_t stepKa_;
    int32_t stepKb_;
    int32_t dbL0A_;
    int32_t dbL0B_;
    int32_t dbL0C_;
};

const std::map<WhiteListShape, WeightNzTilingCache> WEIGHT_NZ_TILING_CACHE = {
    // 125.6
    {{4, 2048, 4096},
     {8,   1,  2,   1,   4096, 4,   4096, 2048, 64, 256, 128, 128,   64, 256, 64, 256, 256, 256, 0, 64, 256, 256,
      256, 64, 256, 256, 64,   256, 64,   64,   64, 1,   1,   32768, 0,  0,   0,  0,   64,  64,  2, 2,  2}},

    // 337.9
    {{4, 5504, 4096},
     {8,   1,  2,   1,   4096, 4,   4096, 5504, 16, 256, 128, 128,   16, 176, 16, 256, 704, 256, 0, 16, 704, 256,
      256, 16, 704, 256, 16,   704, 16,   32,   32, 1,   1,   45056, 0,  0,   0,  0,   32,  32,  2, 2,  2}},

    // 123.3
    {{4, 4096, 2048},
     {8,   1,  2,   1,   2048, 4,   2048, 4096, 16, 256, 128, 128,   16, 256, 16, 256, 512, 256, 0, 16, 512, 256,
      256, 16, 512, 256, 16,   512, 32,   16,   16, 1,   1,   32768, 0,  0,   0,  0,   16,  16,  2, 2,  2}},

    // 3530
    {{4, 65000, 4096},
     {8,   1,  2,   1,   4096, 4,   4096, 65000, 16, 128, 128, 128,   16, 256, 16, 256, 512, 256, 0, 16, 512, 256,
      256, 16, 512, 256, 16,   512, 32,   16,    16, 1,   1,   32768, 0,  0,   0,  0,   16,  16,  2, 2,  2}},

    // 298.3
    {{4, 4096, 5504},
     {8,   1,  2,   2,   5504, 4,   5504, 4096, 16, 256, 128, 128,   16, 256, 16, 256, 512, 256, 0, 16, 512, 256,
      256, 16, 512, 256, 16,   512, 32,   8,    8,  1,   1,   32768, 0,  0,   0,  0,   8,   8,   2, 2,  2}},

    // 2590.5
    {{4096, 2048, 4096},
     {4,   2,   2,   2,   4096, 4096, 4096, 2048, 128, 128, 128, 128,    512, 64, 512, 128, 128, 128, 0, 512, 128, 128,
      128, 512, 128, 128, 512,  128,  32,   4,    4,   1,   1,   262144, 1,   0,  0,   0,   4,   4,   2, 2,   1}},

    // 7930
    {{4096, 5504, 4096},
     {4,   2,   2,   2,   4096, 4096, 4096, 5504, 128, 128, 128, 128,    512, 64, 512, 128, 128, 128, 0, 512, 128, 128,
      128, 512, 128, 128, 512,  128,  32,   4,    4,   1,   1,   262144, 1,   0,  0,   0,   4,   4,   2, 2,   1}},

    // 81465
    {{4096, 65000, 4096},
     {8,   1,   2,   2,   4096, 4096, 4096, 65000, 128, 128, 128, 128,    512, 64, 512, 128, 128, 128, 0, 512, 128, 128,
      128, 512, 128, 128, 512,  128,  32,   4,     4,   1,   1,   262144, 1,   0,  0,   0,   4,   4,   2, 2,   1}},

    // 425
    {{512, 4096, 2048},
     {8,   1,   2,   2,   2048, 512, 2048, 4096, 128, 128, 128, 128,    256, 128, 256, 256, 256, 256, 0, 256, 256, 256,
      256, 256, 256, 256, 256,  256, 64,   4,    4,   1,   1,   262144, 0,   0,   0,   0,   4,   4,   2, 2,   1}},

    // 791
    {{896, 4096, 2048},
     {8,   1,   2,   2,   2048, 896, 2048, 4096, 112, 128, 128, 128,    224, 256, 224, 256, 256, 256, 0, 224, 256, 256,
      256, 224, 256, 256, 224,  256, 64,   4,    4,   1,   1,   229376, 0,   0,   0,   0,   4,   4,   2, 2,   1}},

    // 1021
    {{512, 4096, 5504},
     {8,   1,   2,   2,   5504, 512, 5504, 4096, 128, 128, 128, 128,    256, 128, 256, 256, 256, 256, 0, 256, 256, 256,
      256, 256, 256, 256, 256,  256, 64,   4,    4,   1,   1,   262144, 0,   0,   0,   0,   4,   4,   2, 2,   1}},

    // 1931
    {{896, 4096, 5504},
     {8,   1,   2,   2,   5504, 896, 5504, 4096, 112, 128, 128, 128,    224, 256, 224, 256, 256, 256, 0, 224, 256, 256,
      256, 224, 256, 256, 224,  256, 64,   4,    4,   1,   1,   229376, 0,   0,   0,   0,   4,   4,   2, 2,   1}},

    {{4, 11008, 4096},
     {8,   1,  2,   1,   4096, 4,   4096, 11008, 16, 256, 128, 128,   16, 16, 16, 512, 688, 512, 0, 16, 688, 512,
      512, 16, 688, 512, 16,   688, 16,   32,    32, 1,   1,   44032, 0,  0,  0,  0,   32,  32,  2, 2,  1}},

    {{4812, 9612, 2462},
     {1,   1,   2,   2,   2464, 4812, 2462, 9612, 128, 128, 128, 32,     128, 256, 128, 128, 256, 128, 0, 128, 256, 128,
      128, 128, 256, 128, 128,  256,  64,   2,    2,   1,   1,   131072, 0,   0,   0,   0,   2,   2,   2, 2,   1}},

    {{1811, 7925, 29},
     {1,  1,   2,   2,  32,  1811, 29, 7925, 128, 64, 128, 32, 128, 256, 128, 64, 256, 64, 0, 128, 256, 64,
      64, 128, 256, 64, 128, 256,  32, 2,    2,   1,  1,   0,  1,   0,   0,   0,  2,   2,  2, 2,   1}},

    {{14512, 3376, 48},
     {1,   1,   2,   2,   64,  14512, 48, 3376, 128, 128, 128, 32, 128, 256, 128, 576, 256, 576, 0, 128, 256, 576,
      576, 128, 256, 576, 128, 256,   48, 12,   12,  1,   1,   0,  0,   0,   0,   0,   12,  12,  2, 2,   1}},

    {{5232, 3872, 6736},
     {1,   1,   2,   2,   6752, 5232, 6736, 3872, 128, 128, 128, 32, 128, 256, 128, 256, 256, 256, 0, 128, 256, 256,
      256, 128, 256, 256, 128,  256,  64,   4,    4,   1,   1,   0,  0,   0,   0,   0,   4,   4,   2, 2,   1}},

    {{5236, 13521, 7310},
     {1,   1,   2,   2,   7328, 5236, 7310, 13521, 128, 128, 128, 32, 128, 256, 128, 128, 256, 128, 0, 128, 256, 128,
      128, 128, 256, 128, 128,  256,  64,   2,     2,   1,   1,   0,  0,   0,   0,   0,   2,   2,   2, 2,   1}}};

} // namespace
namespace optiling {

bool WeightQuantBatchMatmulV2WeightNz::IsCapable()
{
    // 判断weight矩阵是否是4维
    auto weightShape = context_->GetInputShape(1);
    auto weightDimNum = weightShape->GetStorageShape().GetDimNum();
    OP_LOGI_IF_RETURN(
        weightDimNum < MM_SHAPE_LEN_NZ_MIN && weightDimNum > MM_SHAPE_LEN_NZ_MAX, false, inputParams_.opName,
        "weightNz template only support right matrix weight >= 4 or <= 8 dims Tensor,now is %ld", weightDimNum);

    // weightNz模板只支持weight矩阵是转置的场景
    OP_LOGI_IF_RETURN(
        !inputParams_.transB, false, inputParams_.opName, "weightNz template only support right matrix is transposed.");

    // weightNz模板只支持weight矩阵的format是FRACTAL_NZ的
    auto weightFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(1)->GetStorageFormat()));
    OP_LOGI_IF_RETURN(
        weightFormat != ge::FORMAT_FRACTAL_NZ, false, inputParams_.opName,
        "weightNz template only support right matrix format is NZ.");

    return true;
}

void WeightQuantBatchMatmulV2WeightNz::SetMatmulTiling()
{
    auto& mmtiling = tilingData_->matmulTiling;
    uint32_t minKL1 = std::min(tilingData_->kBL1Size, tilingData_->kAL1Size);
    uint32_t depthK = ops::CeilDiv(minKL1, static_cast<uint32_t>(mmtiling.baseK));
    mmtiling.M = mmtiling.baseM;
    mmtiling.Ka = minKL1;
    mmtiling.N = mmtiling.baseN;
    mmtiling.Kb = minKL1;
    mmtiling.singleCoreM = mmtiling.baseM;
    mmtiling.singleCoreK = minKL1;
    mmtiling.singleCoreN = mmtiling.baseN;
    mmtiling.depthA1 = depthK;
    mmtiling.depthB1 = depthK;
    mmtiling.stepKa = depthK;
    mmtiling.stepKb = depthK;
    mmtiling.stepM = 1;
    mmtiling.stepN = 1;
    mmtiling.shareL0CSize = 0;
    mmtiling.shareL1Size = 0;
    uint64_t singleMLoop = ops::CeilDiv(
        ops::CeilDiv(tilingData_->mSize, tilingData_->mAL1Size),
        static_cast<uint64_t>(tilingData_->cubeBlockDimM));
    auto mDim = ops::CeilDiv(tilingData_->mSize, singleMLoop * tilingData_->mAL1Size);
    uint64_t singleNLoop = ops::CeilDiv(
        ops::CeilDiv(tilingData_->nSize, tilingData_->nBL1Size),
        static_cast<uint64_t>(tilingData_->cubeBlockDimN));
    auto nDim = ops::CeilDiv(tilingData_->nSize, singleNLoop * tilingData_->nBL1Size);
    tilingData_->cubeBlockDimN = nDim;
    tilingData_->cubeBlockDimM = mDim;
}

void WeightQuantBatchMatmulV2WeightNz::GetubFactorTiling()
{
    auto& mmtiling = tilingData_->matmulTiling;
    uint64_t mmUsedUbSize = mmtiling.baseM * mmtiling.baseN * GetSizeByDataType(ge::DT_FLOAT16);
    uint64_t ubLength =
        inputParams_.hasBias ? aicoreParams_.ubSize - mmtiling.baseN * sizeof(float) : aicoreParams_.ubSize;
    ubLength -= ANTIQUANT_BUF_NUM * tilingData_->matmulTiling.baseN;
    uint64_t bubUsedSize = CalBubFactorTiling(ubLength);
    uint64_t aubUsedSize = CalAubFactorTiling(ubLength - bubUsedSize);
    uint64_t cubNz2NdCanUseSize = CalCubFactorTiling(ubLength - mmUsedUbSize);
    OP_LOGD(
        "WeightQuantBatchmatmul", "cubUseSize %lu, bubusedsize %lu, aubusedsize %lu, cubnd2nzusedsize %lu",
        mmUsedUbSize, bubUsedSize, aubUsedSize, cubNz2NdCanUseSize);
}

ge::graphStatus WeightQuantBatchMatmulV2WeightNz::GetPlatformInfo()
{
    auto compileInfoPtr = compileInfoPtr_ ?
                              compileInfoPtr_.get() :
                              reinterpret_cast<const WeightQuantBatchMatmulV2CompileInfo*>(context_->GetCompileInfo());
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context_->GetNodeName(), "compileInfoPtr is null");
    if (compileInfoPtr_ == nullptr) {
        compileInfoPtr_ = std::unique_ptr<WeightQuantBatchMatmulV2CompileInfo>(
            new (std::nothrow) WeightQuantBatchMatmulV2CompileInfo());
        OP_LOGE_IF(compileInfoPtr_ == nullptr, ge::GRAPH_FAILED, context_->GetNodeName(), "compileInfoPtr_ is null");
        compileInfoPtr_->workspaceNum = compileInfoPtr->workspaceNum;
    }

    aicoreParams_.blockDim = 0;
    aicoreParams_.ubSize = compileInfoPtr->ubSize;
    aicoreParams_.l1Size = compileInfoPtr->l1Size;
    aicoreParams_.l0cSize = compileInfoPtr->l0cSize;

    OP_LOGI(
        inputParams_.opName, "get platform: aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu)",
        compileInfoPtr->aivNum, compileInfoPtr->aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size,
        aicoreParams_.l0cSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2WeightNz::CheckContext() const
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    auto xShape = context_->GetInputShape(idx);
    auto xDesc = context_->GetInputDesc(idx++);
    auto weightShape = context_->GetInputShape(idx);
    auto weightDesc = context_->GetInputDesc(idx++);
    auto antiQuantScaleShape = context_->GetInputShape(idx);
    auto antiQuantScaleDesc = context_->GetInputDesc(idx++);
    auto quantScaleShape = context_->GetOptionalInputShape(QUANT_SCALE_INDEX);
    auto quantOffsetShape = context_->GetOptionalInputShape(QUANT_OFFSET_INDEX);
    auto outputShape = context_->GetOutputShape(0);
    auto outputDesc = context_->GetOutputDesc(0);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, weightShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, antiQuantScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, antiQuantScaleDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    // 不支持C8场景
    OP_LOGE_IF(quantScaleShape != nullptr, ge::GRAPH_FAILED, context_->GetNodeName(), "quant scale is not supported.");
    OP_LOGE_IF(
        quantOffsetShape != nullptr, ge::GRAPH_FAILED, context_->GetNodeName(), "quant offset is not supported.");

    return ge::GRAPH_SUCCESS;
}

bool WeightQuantBatchMatmulV2WeightNz::AnalyzeBiasDtype(const gert::CompileTimeTensorDesc* biasDesc)
{
    if (biasDesc != nullptr) {
        inputParams_.biasDtype = biasDesc->GetDataType();

        OP_TILING_CHECK(
            (inputParams_.biasDtype != inputParams_.aDtype),
            VECTOR_INNER_ERR_REPORT_TILIING(
                inputParams_.opName, "bias[%s] dtype should same as x[%s]",
                ge::TypeUtils::DataTypeToAscendString(inputParams_.biasDtype).GetString(),
                ge::TypeUtils::DataTypeToAscendString(inputParams_.aDtype).GetString()),
            return false);
    }

    return true;
}

bool WeightQuantBatchMatmulV2WeightNz::AnalyzeDtype()
{
    size_t idx = 0;
    inputParams_.aDtype = context_->GetInputDesc(idx++)->GetDataType();
    inputParams_.bDtype = context_->GetInputDesc(idx++)->GetDataType();
    auto antiQuantScaleDtype = context_->GetInputDesc(idx++)->GetDataType();
    auto antiQuantOffsetDesc = context_->GetOptionalInputDesc(idx++);
    auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
    OP_TILING_CHECK(
        (inputParams_.aDtype != ge::DT_FLOAT16),
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "x only support DT_FLOAT16 dtype, get actual dtype[%s]",
            ge::TypeUtils::DataTypeToAscendString(inputParams_.aDtype).GetString()),
        return false);

    inputParams_.cDtype = context_->GetOutputDesc(0)->GetDataType();
    OP_TILING_CHECK(
        (inputParams_.bDtype != ge::DT_INT8),
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "weight only support DT_INT8 dtype, get actual dtype[%s]",
            ge::TypeUtils::DataTypeToAscendString(inputParams_.bDtype).GetString()),
        return false);

    OP_TILING_CHECK(
        (inputParams_.cDtype != inputParams_.aDtype),
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName,
            "y dtype should same as x or be DT_INT8 if quant param is provided, get actual dtype[%s]",
            ge::TypeUtils::DataTypeToAscendString(inputParams_.cDtype).GetString()),
        return false);
    return AnalyzeBiasDtype(biasDesc) && AnalyzeAntiQuantDtype(antiQuantScaleDtype, antiQuantOffsetDesc);
}

bool WeightQuantBatchMatmulV2WeightNz::AnalyzeAntiQuantDtype(
    ge::DataType antiQuantScaleDtype, const gert::CompileTimeTensorDesc* antiQuantOffsetDesc) const
{
    OP_TILING_CHECK(
        antiQuantScaleDtype != inputParams_.aDtype,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "antiquant_scale[%s] dtype should same as x[%s]",
            ge::TypeUtils::DataTypeToAscendString(antiQuantScaleDtype).GetString(),
            ge::TypeUtils::DataTypeToAscendString(inputParams_.aDtype).GetString()),
        return false);
    if (antiQuantOffsetDesc != nullptr) {
        auto antiQuantOffsetDtype = antiQuantOffsetDesc->GetDataType();
        OP_TILING_CHECK(
            antiQuantOffsetDtype != inputParams_.aDtype,
            VECTOR_INNER_ERR_REPORT_TILIING(
                inputParams_.opName, "antiquant_offset[%s] dtype should same as x[%s]",
                ge::TypeUtils::DataTypeToAscendString(antiQuantOffsetDtype).GetString(),
                ge::TypeUtils::DataTypeToAscendString(inputParams_.aDtype).GetString()),
            return false);
    }

    return true;
}

bool WeightQuantBatchMatmulV2WeightNz::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    auto transposeX = attrs->GetAttrPointer<bool>(idx++);
    auto transposeWeight = attrs->GetAttrPointer<bool>(idx++);
    const int64_t* groupSizePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        groupSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }
    if (groupSizePtr != nullptr && *groupSizePtr > 0) {
        inputParams_.groupSize = *groupSizePtr;
    }

    // OP_LOG_FULL
    OPS_LOG_W(inputParams_.opName, "current attr param num is [%lu]", attrs->GetAttrNum());
    inputParams_.transA = transposeX != nullptr && *transposeX;
    inputParams_.transB = transposeWeight != nullptr && *transposeWeight;

    return true;
}

bool WeightQuantBatchMatmulV2WeightNz::SetAntiQuantType(const gert::StorageShape* antiQuantScaleShape)
{
    // antiQuantScaleShape nullptr is impossible
    size_t antiQuantScaleShapeSize = static_cast<size_t>(antiQuantScaleShape->GetStorageShape().GetShapeSize());
    OP_TILING_CHECK(
        antiQuantScaleShapeSize == 0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "antiquant_scale %s cannot be empty tensor",
            Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str()),
        return false);

    if (antiQuantScaleShapeSize == 1) {
        inputParams_.antiQuantType = QuantType::PER_TENSOR;
    } else if (inputParams_.groupSize > 0) {
        inputParams_.antiQuantType = QuantType::PER_GROUP;
    } else {
        OP_TILING_CHECK(
            antiQuantScaleShapeSize != inputParams_.nSize,
            VECTOR_INNER_ERR_REPORT_TILIING(
                inputParams_.opName, "antiquant_scale %s shape size should same as N[%lu]",
                Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str(), inputParams_.nSize),
            return false);
        inputParams_.antiQuantType = QuantType::PER_CHANNEL;
    }
    return true;
}

bool WeightQuantBatchMatmulV2WeightNz::AnalyzeBiasShape(const gert::StorageShape* outShape)
{
    auto biasShape = context_->GetOptionalInputShape(BIAS_INDEX);
    if (biasShape != nullptr) {
        auto biasStorageShape = biasShape->GetStorageShape();
        auto biasDims = biasStorageShape.GetDimNum();
        auto biasShapeSize = static_cast<uint64_t>(biasShape->GetStorageShape().GetShapeSize());
        if (biasShapeSize != 0) {
            inputParams_.hasBias = true;
        }
        if (biasDims <= NO_BATCH_SHAPE_DIM) {
            inputParams_.biasWithBatch = false;
            OP_TILING_CHECK(
                biasShapeSize != inputParams_.nSize && biasShapeSize != 0,
                VECTOR_INNER_ERR_REPORT_TILIING(
                    inputParams_.opName,
                    "when bias without batch, bias shape size[%ld] is not equal nSize[%lu] or zero",
                    biasShape->GetStorageShape().GetShapeSize(), inputParams_.nSize),
                return false);
            OP_TILING_CHECK(
                biasShape->GetStorageShape().GetDimNum() > 1 && biasShape->GetStorageShape().GetDim(0) != 1,
                VECTOR_INNER_ERR_REPORT_TILIING(
                    inputParams_.opName, "when bias without batch, bias shape only support [1, n] or [n,], input is %s",
                    Ops::Base::ToString(biasShape->GetStorageShape()).c_str()),
                return false);
        } else {
            inputParams_.biasWithBatch = true;
            auto outDims = outShape->GetStorageShape().GetDimNum();
            OP_TILING_CHECK(
                biasDims != outDims,
                VECTOR_INNER_ERR_REPORT_TILIING(
                    inputParams_.opName,
                    "when bias with batch, bias shape dim number [%lu] should equal out shape dim numder [%lu]",
                    biasDims, outDims),
                return false);
            OP_TILING_CHECK(
                static_cast<uint64_t>(biasShape->GetStorageShape().GetDim(biasDims - 1)) != inputParams_.nSize ||
                    biasShape->GetStorageShape().GetDim(biasDims - NO_BATCH_SHAPE_DIM) != 1,
                VECTOR_INNER_ERR_REPORT_TILIING(
                    inputParams_.opName, "when bias with batch, bias latest two dims only support [1, n], input is %s",
                    Ops::Base::ToString(biasShape->GetStorageShape()).c_str()),
                return false);

            uint64_t biasWithBatch3 =
                biasDims > NO_BATCH_SHAPE_DIM ?
                    static_cast<uint64_t>(biasStorageShape.GetDim(biasDims - ONE_BATCH_SHAPE_DIM)) :
                    1;
            uint64_t biasWithBatch2 =
                biasDims > ONE_BATCH_SHAPE_DIM ?
                    static_cast<uint64_t>(biasStorageShape.GetDim(biasDims - TWO_BATCH_SHAPE_DIM)) :
                    1;
            uint64_t biasWithBatch1 =
                biasDims > TWO_BATCH_SHAPE_DIM ?
                    static_cast<uint64_t>(biasStorageShape.GetDim(biasDims - THREE_BATCH_SHAPE_DIM)) :
                    1;
            uint64_t biasWithBatch0 =
                biasDims > THREE_BATCH_SHAPE_DIM ?
                    static_cast<uint64_t>(biasStorageShape.GetDim(biasDims - FOUR_BATCH_SHAPE_DIM)) :
                    1;
            bool biasBatchValid = biasWithBatch3 == inputParams_.batchY3 && biasWithBatch2 == inputParams_.batchY2 &&
                                  biasWithBatch1 == inputParams_.batchY1 && biasWithBatch0 == inputParams_.batchY0;
            if (!biasBatchValid) {
                OP_LOGE("[WeightQuantBatchMatmulV2]", "Batch of Bias  must be equal to Out !");
                return false;
            }
        }
    }
    return true;
}

bool WeightQuantBatchMatmulV2WeightNz::AnalyzeInputs()
{
    size_t idx = 0;
    auto xShape = context_->GetInputShape(idx++);
    auto weightShape = context_->GetInputShape(idx++);
    auto antiQuantScaleShape = context_->GetInputShape(idx++);
    auto antiQuantOffsetShape = context_->GetOptionalInputShape(idx++);
    auto outShape = context_->GetOutputShape(0);

    return AnalyzeInputShape(xShape, weightShape, outShape) &&
           AnalyzeAntiQuantShape(antiQuantScaleShape, antiQuantOffsetShape) && SetAntiQuantType(antiQuantScaleShape) &&
           AnalyzeBiasShape(outShape);
}

bool WeightQuantBatchMatmulV2WeightNz::AnalyzeInputShape(
    const gert::StorageShape* xShape, const gert::StorageShape* weightShape, const gert::StorageShape* outShape)
{
    auto xStorageShape = xShape->GetStorageShape();
    auto weightStorageShape = weightShape->GetStorageShape();
    auto outStorageShape = outShape->GetStorageShape();
    OP_TILING_CHECK(
        xShape->GetStorageShape().GetShapeSize() == 0 || weightShape->GetStorageShape().GetShapeSize() == 0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "not support empty tensor, x %s, weight %s",
            Ops::Base::ToString(xShape->GetStorageShape()).c_str(),
            Ops::Base::ToString(weightShape->GetStorageShape()).c_str()),
        return false);
    auto xDims = xStorageShape.GetDimNum();
    auto weightDims = weightStorageShape.GetDimNum();
    auto outDims = outStorageShape.GetDimNum();

    OP_TILING_CHECK(
        (xDims < MM_SHAPE_LEN_ND_MIN || xDims > MM_SHAPE_LEN_ND_MAX),
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "left matrix X dim support >=2 and <=6, now is %s",
            Ops::Base::ToString(xShape->GetStorageShape()).c_str()),
        return false);
    inputParams_.mSize = static_cast<uint64_t>(
        inputParams_.transA ? xShape->GetStorageShape().GetDim(xDims - 1) :
                              xShape->GetStorageShape().GetDim(xDims - MM_SHAPE_LEN_ND));
    inputParams_.kSize = static_cast<uint64_t>(
        inputParams_.transA ? xShape->GetStorageShape().GetDim(xDims - MM_SHAPE_LEN_ND) :
                              xShape->GetStorageShape().GetDim(xDims - 1));
    inputParams_.nSize = static_cast<uint64_t>(outStorageShape.GetDim(outDims - 1));
    uint64_t kBSize = static_cast<uint64_t>(
        weightShape->GetStorageShape().GetDim(weightDims - MM_SHAPE_LEN_NZ) *
        weightShape->GetStorageShape().GetDim(weightDims - 1));
    // OP_LOG_FULL
    OPS_LOG_W(inputParams_.opName, "kSize_[%lu] ,KBSize[%lu]", inputParams_.kSize, kBSize);

    weightDims = weightDims - ND_NZ_DIM_DIFF;
    inputParams_.batchX3 = xDims > NO_BATCH_SHAPE_DIM ? xStorageShape.GetDim(xDims - ONE_BATCH_SHAPE_DIM) : 1;
    inputParams_.batchX2 = xDims > ONE_BATCH_SHAPE_DIM ? xStorageShape.GetDim(xDims - TWO_BATCH_SHAPE_DIM) : 1;
    inputParams_.batchX1 = xDims > TWO_BATCH_SHAPE_DIM ? xStorageShape.GetDim(xDims - THREE_BATCH_SHAPE_DIM) : 1;
    inputParams_.batchX0 = xDims > THREE_BATCH_SHAPE_DIM ? xStorageShape.GetDim(xDims - FOUR_BATCH_SHAPE_DIM) : 1;
    inputParams_.batchWeight3 =
        weightDims > NO_BATCH_SHAPE_DIM ? weightStorageShape.GetDim(weightDims - ONE_BATCH_SHAPE_DIM) : 1;
    inputParams_.batchWeight2 =
        weightDims > ONE_BATCH_SHAPE_DIM ? weightStorageShape.GetDim(weightDims - TWO_BATCH_SHAPE_DIM) : 1;
    inputParams_.batchWeight1 =
        weightDims > TWO_BATCH_SHAPE_DIM ? weightStorageShape.GetDim(weightDims - THREE_BATCH_SHAPE_DIM) : 1;
    inputParams_.batchWeight0 =
        weightDims > THREE_BATCH_SHAPE_DIM ? weightStorageShape.GetDim(weightDims - FOUR_BATCH_SHAPE_DIM) : 1;
    inputParams_.batchY3 = outDims > NO_BATCH_SHAPE_DIM ? outStorageShape.GetDim(outDims - ONE_BATCH_SHAPE_DIM) : 1;
    inputParams_.batchY2 = outDims > ONE_BATCH_SHAPE_DIM ? outStorageShape.GetDim(outDims - TWO_BATCH_SHAPE_DIM) : 1;
    inputParams_.batchY1 = outDims > TWO_BATCH_SHAPE_DIM ? outStorageShape.GetDim(outDims - THREE_BATCH_SHAPE_DIM) : 1;
    inputParams_.batchY0 = outDims > THREE_BATCH_SHAPE_DIM ? outStorageShape.GetDim(outDims - FOUR_BATCH_SHAPE_DIM) : 1;
    bool batch3Invalid = inputParams_.batchX3 != inputParams_.batchWeight3 && inputParams_.batchX3 != 1 &&
                         inputParams_.batchWeight3 != 1;
    bool batch2Invalid = inputParams_.batchX2 != inputParams_.batchWeight2 && inputParams_.batchX2 != 1 &&
                         inputParams_.batchWeight2 != 1;
    bool batch1Invalid = inputParams_.batchX1 != inputParams_.batchWeight1 && inputParams_.batchX1 != 1 &&
                         inputParams_.batchWeight1 != 1;
    bool batch0Invalid = inputParams_.batchX0 != inputParams_.batchWeight0 && inputParams_.batchX0 != 1 &&
                         inputParams_.batchWeight0 != 1;
    if (batch3Invalid || batch2Invalid || batch1Invalid || batch0Invalid) {
        OP_LOGE("[WeightQuantBatchMatmulV2]", "x and weight batch broadcast failed!");
        return false;
    }
    return true;
}

bool WeightQuantBatchMatmulV2WeightNz::AnalyzeAntiQuantShape(
    const gert::StorageShape* antiQuantScaleShape, const gert::StorageShape* antiQuantOffsetShape)
{
    if (inputParams_.groupSize > 0) {
        gert::Shape expectShape;
        uint64_t groupNum = ops::CeilDiv(inputParams_.kSize, inputParams_.groupSize);
        expectShape.AppendDim(static_cast<int64_t>(groupNum));
        expectShape.AppendDim(static_cast<int64_t>(inputParams_.nSize));
        OP_TILING_CHECK(
            expectShape != antiQuantScaleShape->GetStorageShape(),
            VECTOR_INNER_ERR_REPORT_TILIING(
                inputParams_.opName, "Antiquant shape expect %s, but is %s", Ops::Base::ToString(expectShape).c_str(),
                Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str()),
            return false);
    } else {
        OP_TILING_CHECK(
            antiQuantScaleShape->GetStorageShape().GetDimNum() > VALID_INPUT_DIM_NUM ||
                (antiQuantScaleShape->GetStorageShape().GetDimNum() == VALID_INPUT_DIM_NUM &&
                 antiQuantScaleShape->GetStorageShape().GetDim(1) != 1),
            VECTOR_INNER_ERR_REPORT_TILIING(
                inputParams_.opName,
                "per-channel antiquant_scale shape only support [n, 1] or [n,], actual input shape is %s",
                Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str()),
            return false);
    }

    if (CheckOptionalInputByShape(antiQuantOffsetShape)) {
        OP_TILING_CHECK(
            antiQuantScaleShape->GetStorageShape() != antiQuantOffsetShape->GetStorageShape(),
            VECTOR_INNER_ERR_REPORT_TILIING(
                inputParams_.opName, "antiquant_scale %s and antiquant_offset %s should have same shape",
                Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str(),
                Ops::Base::ToString(antiQuantOffsetShape->GetStorageShape()).c_str()),
            return false);
        inputParams_.hasAntiQuantOffset = true;
    }
    return true;
}

ge::graphStatus WeightQuantBatchMatmulV2WeightNz::GetShapeAttrsInfo()
{
    inputParams_.opName = context_->GetNodeName();
    // OP_LOG_FULL
    OPS_LOG_I(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    auto compileInfoPtr = compileInfoPtr_ ?
                              compileInfoPtr_.get() :
                              context_->GetCompileInfo<WeightQuantBatchMatmulV2CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context_->GetNodeName(), "compileInfoPtr is null");
    // OP_LOG_FULL
    OPS_LOG_D(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK(
        CheckContext() != ge::GRAPH_SUCCESS, VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "invalid context"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        !AnalyzeDtype() || !AnalyzeAttrs() || !AnalyzeInputs(),
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "fail to analyze context info"), return ge::GRAPH_FAILED);

    bool maxDimCheck = inputParams_.kSize > MAX_SHAPE_DIM || inputParams_.nSize > MAX_SHAPE_DIM;
    // A矩阵转置场景，m需要小于65535
    if (inputParams_.transA) {
        maxDimCheck |= inputParams_.mSize > MAX_SHAPE_DIM;
    }
    OP_TILING_CHECK(
        maxDimCheck,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "only support MKN in range [1, %lu], get actual value[%lu, %lu, %lu]", MAX_SHAPE_DIM,
            inputParams_.mSize, inputParams_.kSize, inputParams_.nSize),
        return ge::GRAPH_FAILED);

    OP_LOGD(
        inputParams_.opName, "input params: MKN[%lu, %lu, %lu], transA[%s], transB[%s], bias[%s]", inputParams_.mSize,
        inputParams_.kSize, inputParams_.nSize, inputParams_.transA ? "true" : "false",
        inputParams_.transB ? "true" : "false", inputParams_.hasBias ? "true" : "false");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2WeightNz::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        tilingDataManager_ = std::unique_ptr<WeightQuantBatchMatmulV2NzTilingData>(
            new (std::nothrow) WeightQuantBatchMatmulV2NzTilingData());
        tilingData_ = tilingDataManager_.get();
    }
    OP_TILING_CHECK(
        tilingData_ == nullptr,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "failed to instantiate tilingData"),
        return ge::GRAPH_FAILED);
    size_t tilingDataSize = sizeof(WeightQuantBatchMatmulV2NzTilingData);
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingDataSize,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "tiling data capacity %zu < actual tiling data size %zu",
            context_->GetRawTilingData()->GetCapacity(), tilingDataSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2WeightNz::DoOpTiling()
{
    OP_TILING_CHECK(
        InstantiateTilingData() == ge::GRAPH_FAILED,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "unable to get pointer of tiling data"),
        return ge::GRAPH_FAILED);

    // 优先从cache中获取Tiling
    if (inputParams_.groupSize > 0) {
        OP_LOGI(inputParams_.opName, "per-group has no weightNz tiling cache");
    } else if (GetTilingFromCache()) {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t weightBlockAlignSize = GetBlockAlignSizeByDataType(inputParams_.bDtype);
    if (inputParams_.transB) {
        tilingData_->kAlign = ops::CeilAlign(inputParams_.kSize, weightBlockAlignSize);
        tilingData_->nAlign = inputParams_.nSize;
    } else {
        tilingData_->kAlign = inputParams_.kSize;
        tilingData_->nAlign = ops::CeilAlign(inputParams_.nSize, weightBlockAlignSize);
    }
    tilingData_->kSize = inputParams_.kSize;
    tilingData_->nSize = inputParams_.nSize;
    tilingData_->mSize = inputParams_.mSize;
    tilingData_->groupSize = inputParams_.groupSize;

    tilingData_->batchY0 = inputParams_.batchY0;
    tilingData_->batchY1 = inputParams_.batchY1;
    tilingData_->batchY2 = inputParams_.batchY2;
    tilingData_->batchY3 = inputParams_.batchY3;
    tilingData_->batchX0 = inputParams_.batchX0;
    tilingData_->batchX1 = inputParams_.batchX1;
    tilingData_->batchX2 = inputParams_.batchX2;
    tilingData_->batchX3 = inputParams_.batchX3;
    tilingData_->batchWeight0 = inputParams_.batchWeight0;
    tilingData_->batchWeight1 = inputParams_.batchWeight1;
    tilingData_->batchWeight2 = inputParams_.batchWeight2;
    tilingData_->batchWeight3 = inputParams_.batchWeight3;

    tilingData_->biasWithBatch = static_cast<uint32_t>(inputParams_.biasWithBatch);
    OP_CHECK_IF(
        !GetMatMulTiling(),
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "failed to get mm tiling for mnk[%lu, %lu, %lu]", inputParams_.mSize,
            inputParams_.nSize, inputParams_.kSize),
        return ge::GRAPH_FAILED);
    GetL1tiling();
    GetL1Pingpong(); // 判断A/B在L1上是否开DB_BUFFER
    GetubFactorTiling();
    SetMatmulTiling();
    return ge::GRAPH_SUCCESS;
}

uint64_t WeightQuantBatchMatmulV2WeightNz::GetTilingKey() const
{
    uint64_t batch = 1;
    if (inputParams_.batchY3 == 1 && inputParams_.batchY2 == 1 && inputParams_.batchY1 == 1 &&
        inputParams_.batchY0 == 1) {
        batch = 0;
    }
    uint64_t socVersionType = 1UL; // 1 means SUPPORT_L0C_TO_OUT
    uint64_t subSocVersionType = 0UL;
    uint64_t antiquantScenario = 0UL;
    uint64_t algorithm = 3UL; // 3 means CUSTOM tilingkey algorithm
    uint64_t subAlgorithm = 0UL;
    uint64_t subAlgorithmCustom = static_cast<uint64_t>(KernelTemplateType::WEIGHT_NZ);
    uint64_t innerPrecise = 0UL;
    uint64_t templateCustom = 0UL;
    uint64_t apiConstexpr = 0UL;
    bool transA = inputParams_.transA;
    bool transB = false;
    uint64_t antiquantType = static_cast<uint64_t>(inputParams_.antiQuantType);
    uint64_t quantType = static_cast<uint64_t>(QuantType::NONE);
    bool hasAntiquantOffset = inputParams_.hasAntiQuantOffset;
    bool hasBias = false;
    bool isBiasFp32 = false;
    bool isWeightNz = false;
    uint64_t templateExtra = 3UL; // 3 means TEMPLATE_EXTRA_NOT_USED
    uint64_t fullLoadMode = static_cast<uint64_t>(L1FullloadMode_);
    return GET_TPL_TILING_KEY(
        socVersionType, subSocVersionType, antiquantScenario, algorithm, subAlgorithm, subAlgorithmCustom,
        innerPrecise, templateCustom, apiConstexpr, transA, transB, antiquantType, quantType, hasAntiquantOffset,
        hasBias, isBiasFp32, isWeightNz, templateExtra, fullLoadMode, batch);
}

void WeightQuantBatchMatmulV2WeightNz::Reset()
{
    inputParams_.transA = false;
    inputParams_.transB = false;
    inputParams_.hasBias = false;
    inputParams_.biasWithBatch = false;
    inputParams_.hasAntiQuantOffset = false;
    inputParams_.groupSize = 0UL;
    inputParams_.mSize = 0L;
    inputParams_.kSize = 0L;
    inputParams_.nSize = 0L;

    inputParams_.batchX0 = 1L;
    inputParams_.batchX1 = 1L;
    inputParams_.batchX2 = 1L;
    inputParams_.batchX3 = 1L;
    inputParams_.batchWeight0 = 1L;
    inputParams_.batchWeight1 = 1L;
    inputParams_.batchWeight2 = 1L;
    inputParams_.batchWeight3 = 1L;
    inputParams_.batchY0 = 1L;
    inputParams_.batchY1 = 1L;
    inputParams_.batchY2 = 1L;
    inputParams_.batchY3 = 1L;

    inputParams_.aDtype = ge::DT_FLOAT16;
    inputParams_.bDtype = ge::DT_INT8;
    inputParams_.cDtype = ge::DT_FLOAT16;
    inputParams_.biasDtype = ge::DT_FLOAT16;
    aFormat = ge::FORMAT_ND;
    bFormat = ge::FORMAT_ND;
    inputParams_.antiQuantType = QuantType::NONE;
    inputParams_.quantType = QuantType::PER_TENSOR; // set default value as per tensor, kernel can detect ydtype firstly
    inputParams_.opName = nullptr;
}

void WeightQuantBatchMatmulV2WeightNz::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr_ =
        std::unique_ptr<WeightQuantBatchMatmulV2CompileInfo>(new (std::nothrow) WeightQuantBatchMatmulV2CompileInfo());
    OP_TILING_CHECK(
        compileInfoPtr_ == nullptr,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "failed to instantiate compile info"), return);

    compileInfoPtr_->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr_->aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr_->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr_->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr_->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr_->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr_->l0bSize);
    compileInfoPtr_->workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfoPtr_->socVersion = ascendcPlatform.GetSocVersion();
    std::string mmad;
    bool res = platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_mmad", mmad);
    compileInfoPtr_->supportMmadS8S4 = res && mmad.find("s8s4") != std::string::npos;

    TilingPrepareForOpCache(context_);
    OP_LOGD(context_->GetNodeName(), "MatmulAllReduce Init Quant Tiling Compile Info Success");
}

static void GetMatmulTilingFromCache(AscendC::tiling::TCubeTiling& matmulTiling, const WeightNzTilingCache& weightNzTilingCache)
{
    matmulTiling.M = weightNzTilingCache.m_;
    matmulTiling.N = weightNzTilingCache.n_;
    matmulTiling.Ka = weightNzTilingCache.ka_;
    matmulTiling.Kb = weightNzTilingCache.kb_;
    matmulTiling.singleCoreM = weightNzTilingCache.singleCoreM_;
    matmulTiling.singleCoreN = weightNzTilingCache.singleCoreN_;
    matmulTiling.singleCoreK = weightNzTilingCache.singleCoreK_;
    matmulTiling.baseM = weightNzTilingCache.baseM_;
    matmulTiling.baseN = weightNzTilingCache.baseN_;
    matmulTiling.baseK = weightNzTilingCache.baseK_;
    matmulTiling.depthA1 = weightNzTilingCache.depthA1_;
    matmulTiling.depthB1 = weightNzTilingCache.depthB1_;
    matmulTiling.stepM = weightNzTilingCache.stepM_;
    matmulTiling.stepN = weightNzTilingCache.stepN_;
    matmulTiling.stepKa = weightNzTilingCache.stepKa_;
    matmulTiling.stepKb = weightNzTilingCache.stepKb_;
    matmulTiling.transLength = weightNzTilingCache.transLength_;
    matmulTiling.iterateOrder = weightNzTilingCache.iterateOrder_;
    matmulTiling.shareL1Size = weightNzTilingCache.shareL1Size_;
    matmulTiling.shareL0CSize = weightNzTilingCache.shareL0CSize_;
    matmulTiling.dbL0A = weightNzTilingCache.dbL0A_;
    matmulTiling.dbL0B = weightNzTilingCache.dbL0B_;
    matmulTiling.dbL0C = weightNzTilingCache.dbL0C_;
    matmulTiling.usedCoreNum = 1;
    matmulTiling.batchM = 1;
    matmulTiling.batchN = 1;
    matmulTiling.singleBatchM = 1;
    matmulTiling.singleBatchN = 1;
}

bool WeightQuantBatchMatmulV2WeightNz::GetTilingFromCache()
{
    WhiteListShape shape({inputParams_.mSize, inputParams_.nSize, inputParams_.kSize});
    auto it = WEIGHT_NZ_TILING_CACHE.find(shape);
    if (it == WEIGHT_NZ_TILING_CACHE.end()) {
        OP_LOGI(inputParams_.opName, "not find weightNz tiling from cache");
        return false;
    }

    OP_LOGD(inputParams_.opName, "get weightNz tiling from cache");

    auto& weightNzTilingCache = it->second;
    tilingData_->cubeBlockDimN = weightNzTilingCache.cubeBlockDimN_;
    tilingData_->cubeBlockDimM = weightNzTilingCache.cubeBlockDimM_;
    tilingData_->AL1Pingpong = weightNzTilingCache.aL1Pingpong_;
    tilingData_->BL1Pingpong = weightNzTilingCache.bL1Pingpong_;
    tilingData_->kAlign = weightNzTilingCache.kAlign_;
    tilingData_->mSize = weightNzTilingCache.mSize_;
    tilingData_->kSize = weightNzTilingCache.kSize_;
    tilingData_->nSize = weightNzTilingCache.nSize_;
    tilingData_->nAlign = weightNzTilingCache.nSize_;
    tilingData_->mAubSize = weightNzTilingCache.mAubSize_;
    tilingData_->kAubSize = weightNzTilingCache.kAubSize_;
    tilingData_->nBubSize = weightNzTilingCache.nBubSize_;
    tilingData_->kBubSize = weightNzTilingCache.kBubSize_;
    tilingData_->mCubSize = weightNzTilingCache.mCubSize_;
    tilingData_->nCubSize = weightNzTilingCache.nCubSize_;
    tilingData_->mAL1Size = weightNzTilingCache.mAL1Size_;
    tilingData_->kAL1Size = weightNzTilingCache.kAL1Size_;
    tilingData_->nBL1Size = weightNzTilingCache.nBL1Size_;
    tilingData_->kBL1Size = weightNzTilingCache.kBL1Size_;
    tilingData_->groupSize = weightNzTilingCache.groupSize_;
    tilingData_->batchX0 = inputParams_.batchX0;
    tilingData_->batchX1 = inputParams_.batchX1;
    tilingData_->batchX2 = inputParams_.batchX2;
    tilingData_->batchX3 = inputParams_.batchX3;
    tilingData_->batchWeight0 = inputParams_.batchWeight0;
    tilingData_->batchWeight1 = inputParams_.batchWeight1;
    tilingData_->batchWeight2 = inputParams_.batchWeight2;
    tilingData_->batchWeight3 = inputParams_.batchWeight3;
    tilingData_->batchY0 = inputParams_.batchY0;
    tilingData_->batchY1 = inputParams_.batchY1;
    tilingData_->batchY2 = inputParams_.batchY2;
    tilingData_->batchY3 = inputParams_.batchY3;
    tilingData_->biasWithBatch = static_cast<uint32_t>(inputParams_.biasWithBatch);

    auto& matmulTiling = tilingData_->matmulTiling;
    matmulTiling.isBias = static_cast<int32_t>(inputParams_.hasBias);
    GetMatmulTilingFromCache(matmulTiling, weightNzTilingCache);

    return true;
}

ge::graphStatus WeightQuantBatchMatmulV2WeightNz::GetWorkspaceSize()
{
    workspaceSize_ = compileInfoPtr_->workspaceNum;
    return ge::GRAPH_SUCCESS;
}

bool WeightQuantBatchMatmulV2WeightNz::GetMmTilingInput(MmTilingInput& mmTilingInput)
{
    mmTilingInput.aDtype = GetMatmulTilingDtype(inputParams_.aDtype);
    mmTilingInput.bDtype = GetMatmulTilingDtype(inputParams_.aDtype); // same to a
    if (aFormat == ge::FORMAT_ND || aFormat == ge::FORMAT_FRACTAL_NZ) {
        mmTilingInput.aPosition = matmul_tiling::TPosition::GM;
    } else {
        OP_LOGE(inputParams_.opName, "The format of input a is unsupported.");
        return false;
    }
    if (bFormat == ge::FORMAT_ND || bFormat == ge::FORMAT_FRACTAL_NZ) {
        mmTilingInput.aPosition = matmul_tiling::TPosition::GM;
    } else {
        OP_LOGE(inputParams_.opName, "The format of input b is unsupported.");
        return false;
    }
    mmTilingInput.bPosition = matmul_tiling::TPosition::GM;
    mmTilingInput.biasPosition =
        inputParams_.biasDtype == ge::DT_FLOAT ? matmul_tiling::TPosition::GM : matmul_tiling::TPosition::VECCALC;
    return true;
}

bool WeightQuantBatchMatmulV2WeightNz::CheckUBSize()
{
    auto& mmtiling = tilingData_->matmulTiling;
    uint64_t mmUsedUbSize = mmtiling.baseM * mmtiling.baseN * GetSizeByDataType(ge::DT_FLOAT16);
    if (inputParams_.hasBias) {
        if (inputParams_.biasDtype == ge::DT_FLOAT16) {
            mmUsedUbSize +=
                mmtiling.baseN * (GetSizeByDataType(ge::DT_FLOAT16) + GetSizeByDataType(ge::DT_FLOAT));
        } else {
            mmUsedUbSize += mmtiling.baseN * GetSizeByDataType(ge::DT_FLOAT);
        }
    }
    // cub 至少占用的空间
    uint64_t cubNz2NdCanUseSize =
        std::min(mmtiling.baseM, mmtiling.baseN) * GetSizeByDataType(ge::DT_FLOAT16) * BLOCK_CUBE;
    return mmUsedUbSize + cubNz2NdCanUseSize < aicoreParams_.ubSize;
}

MatrixTraverse WeightQuantBatchMatmulV2WeightNz::GetIteratorOrder(
    const CacheTilingData& tbeTiling, int32_t singleCoreM, int32_t singleCoreN, int32_t singleCoreK) const
{
    int32_t reduceSize = static_cast<int32_t>((inputParams_.aDtype));
    bool fullkAL1Load = singleCoreK <= (tbeTiling.kal1_16 * reduceSize);
    bool fullkBL1Load = singleCoreK <= (tbeTiling.kbl1_16 * reduceSize);

    // if KAL1 and KBL1 both can not be full loaded, then select m or n which is no matter
    if (!fullkAL1Load && !fullkBL1Load) {
        return MatrixTraverse::FIRSTM;
    } else if (fullkAL1Load && !fullkBL1Load) { // if KAL1 is full loaded, then select the order N first
        return MatrixTraverse::FIRSTN;
    } else if (!fullkAL1Load && fullkBL1Load) { // if KBL1 is full loaded, then select the order M first
        return MatrixTraverse::FIRSTM;
    } else {
        // if AL1LoadSize less than BL1LoadSize, then select order N first, vice versa.
        int32_t mLoop = ops::CeilDiv(singleCoreM, static_cast<int32_t>(tbeTiling.m_al1 * tbeTiling.m_l0 * BLOCK_CUBE));
        int32_t nLoop = ops::CeilDiv(singleCoreN, static_cast<int32_t>(tbeTiling.n_bl1 * tbeTiling.n_l0 * BLOCK_CUBE));
        int32_t aL1LoadSize = singleCoreM + singleCoreN * mLoop;
        int32_t bL1LoadSize = singleCoreN + singleCoreM * nLoop;
        return aL1LoadSize < bL1LoadSize ? MatrixTraverse::FIRSTN : MatrixTraverse::FIRSTM;
    }
}

void WeightQuantBatchMatmulV2WeightNz::Convert2AscendCTiling(
    const CacheTilingData& tbeTiling, AscendC::tiling::TCubeTiling& matmulTiling)
{
    auto mDim =
        ops::CeilDiv(inputParams_.mSize, ops::CeilDiv(inputParams_.mSize, static_cast<uint64_t>(tbeTiling.m_dim)));
    auto nDim =
        ops::CeilDiv(inputParams_.nSize, ops::CeilDiv(inputParams_.nSize, static_cast<uint64_t>(tbeTiling.n_dim)));
    tilingData_->cubeBlockDimM = static_cast<uint64_t>(mDim);
    tilingData_->cubeBlockDimN = static_cast<uint64_t>(nDim);

    // 内轴需256对齐以提高nd2nz效率
    matmulTiling.Kb = ops::CeilAlign(tilingData_->kSize, static_cast<uint64_t>(256));

    matmulTiling.singleCoreM = static_cast<int32_t>(ops::CeilDiv(inputParams_.mSize, mDim));
    matmulTiling.singleCoreN = static_cast<int32_t>(ops::CeilDiv(inputParams_.nSize, nDim));
    matmulTiling.baseN = tbeTiling.n_l0 * BLOCK_CUBE;
    matmulTiling.baseM = tbeTiling.m_l0 * BLOCK_CUBE;
    int32_t reduceSize = static_cast<int32_t>(GetBlockAlignSizeByDataType(inputParams_.aDtype));
    matmulTiling.baseK = tbeTiling.k_l0 * reduceSize;
    auto depthA1 = std::max(ops::CeilDiv(tbeTiling.kal1_16, tbeTiling.k_l0) * tbeTiling.m_al1 * tbeTiling.db_al1, 2L);
    auto depthB1 = std::max(ops::CeilDiv(tbeTiling.kbl1_16, tbeTiling.k_l0) * tbeTiling.n_bl1 * tbeTiling.db_bl1, 2L);
    matmulTiling.depthA1 = depthA1;
    matmulTiling.depthB1 = depthB1;
    matmulTiling.stepM = tbeTiling.m_al1;
    matmulTiling.stepN = tbeTiling.n_bl1;
    matmulTiling.stepKa = ops::CeilDiv(tbeTiling.kal1_16, tbeTiling.k_l0);
    matmulTiling.stepKb = ops::CeilDiv(tbeTiling.kbl1_16, tbeTiling.k_l0);
    int32_t a1Length = static_cast<int32_t>(
        GetShapeSizeWithDataType(matmulTiling.baseM * matmulTiling.baseK, inputParams_.aDtype));
    int32_t b1Length = static_cast<int32_t>(
        GetShapeSizeWithDataType(matmulTiling.baseN * matmulTiling.baseK, inputParams_.aDtype));
    int32_t c1Length = matmulTiling.baseN * matmulTiling.baseM * sizeof(float); // L0C

    matmulTiling.transLength = std::max(std::max(a1Length, b1Length), c1Length);
    // MatrixTraverse枚举值和matmul api使用的枚举值相差1
    matmulTiling.iterateOrder =
        static_cast<int32_t>(GetIteratorOrder(
            tbeTiling, matmulTiling.singleCoreM, matmulTiling.singleCoreN,
            matmulTiling.singleCoreK)) - 1;
    matmulTiling.dbL0C = tbeTiling.db_l0c;
    int32_t aL1Size = a1Length * matmulTiling.depthA1;
    int32_t biasL1Size =
        inputParams_.hasBias ?
            GetShapeSizeWithDataType(matmulTiling.baseN, inputParams_.biasDtype) * tbeTiling.n_bl1 : 0;
    int32_t bL1Size = b1Length * matmulTiling.depthB1;
    matmulTiling.shareL1Size = aL1Size + bL1Size + biasL1Size;
    matmulTiling.shareL0CSize = c1Length;

    SetAscendCTiling(matmulTiling);
}

void WeightQuantBatchMatmulV2WeightNz::SetAscendCTiling(AscendC::tiling::TCubeTiling& matmulTiling)
{
    matmulTiling.singleCoreK = inputParams_.kSize;
    matmulTiling.M = inputParams_.mSize;
    matmulTiling.N = inputParams_.nSize;
    matmulTiling.Ka = inputParams_.kSize;

    matmulTiling.shareMode = 0;
    matmulTiling.dbL0A = 2; // db switch, 1: off, 2: on
    matmulTiling.dbL0B = 2; // db switch, 1: off, 2: on

    matmulTiling.shareUbSize = 0;
    matmulTiling.batchM = 1;
    matmulTiling.batchN = 1;
    matmulTiling.singleBatchM = 1;
    matmulTiling.singleBatchN = 1;
    matmulTiling.isBias = inputParams_.hasBias ? 1 : 0;

    matmulTiling.usedCoreNum = 1;
    matmulTiling.Kb = tilingData_->kAlign;
}

bool WeightQuantBatchMatmulV2WeightNz::InvokeCacheTiling()
{
    BatchmatmulCompileParas compileParams;
    compileParams.binary_mode_flag = true;
    compileParams.bias_flag = inputParams_.hasBias;
    compileParams.pattern_flag = true;

    bool alignedMKN = inputParams_.mSize % BLOCK_CUBE == 0 && inputParams_.kSize % BLOCK_CUBE == 0 &&
                      inputParams_.nSize % BLOCK_CUBE == 0;
    BatchmatmulRunParas runParams;
    runParams.trans_a_flag = inputParams_.transA;
    runParams.trans_b_flag = inputParams_.transB;
    runParams.format_a_nd = true;
    runParams.format_out_nd = true;
    runParams.nd_flag = runParams.format_a_nd && runParams.format_b_nd;
    runParams.used_aligned_pattern = alignedMKN && runParams.nd_flag;
    runParams.bias_flag = inputParams_.hasBias;
    runParams.pattern_flag = !inputParams_.hasBias;
    runParams.unaligned_flag = !alignedMKN;
    runParams.zero_flag = compileParams.zero_flag;
    runParams.weight_nz_flag = true;
    runParams.hf32_flag = 0;
    runParams.dtype_a = static_cast<int32_t>(inputParams_.aDtype);
    runParams.dtype_b = runParams.dtype_a;
    runParams.dtype_out = runParams.dtype_a;
    runParams.dtype_bias = ge::GetSizeByDataType(inputParams_.biasDtype);
    runParams.m = ops::CeilDiv(inputParams_.mSize, static_cast<uint64_t>(BLOCK_CUBE));
    runParams.k = ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(BLOCK_CUBE));
    runParams.n = ops::CeilDiv(inputParams_.nSize, static_cast<uint64_t>(BLOCK_CUBE));
    runParams.batch = 1;
    runParams.ori_shape_m = inputParams_.mSize;
    runParams.ori_shape_k = inputParams_.kSize;
    runParams.ori_shape_n = inputParams_.nSize;
    runParams.m_quant_check = inputParams_.transA && inputParams_.cDtype == ge::DT_INT8;
    runParams.n_quant_check = !inputParams_.transB;
    runParams.bias_dtype = inputParams_.biasDtype;
    runParams.vector_pre_conv_mode = inputParams_.cDtype == ge::DT_INT8;
    CacheTilingData tiling;
    tiling.tiling_id = std::numeric_limits<uint64_t>::max();
    GenTiling("WeightQuantBatchMatmulV2", compileParams, runParams, tiling, context_);
    OP_LOGI_IF_RETURN(
        tiling.tiling_id == std::numeric_limits<uint64_t>::max(), false, inputParams_.opName,
        "cannot get tiling from cachetiling, mnk[%lu, %lu, %lu]", inputParams_.mSize, inputParams_.nSize,
        inputParams_.kSize);
    Convert2AscendCTiling(tiling, tilingData_->matmulTiling);

    return true;
}

bool WeightQuantBatchMatmulV2WeightNz::GetMatMulTiling()
{
    if (InvokeCacheTiling() && CheckUBSize()) {
        OP_LOGD(inputParams_.opName, "invoke cache tiling success");
        return true;
    }
    return false;
}

void WeightQuantBatchMatmulV2WeightNz::GetBaseMKNByTrans(matmul_tiling::MatmulApiTiling& mmTiling) const
{
    if (aFormat != ge::FORMAT_ND && bFormat != ge::FORMAT_ND) {
        return;
    }
    mmTiling.SetFixSplit(cubeBaseM_, cubeBaseN_);
    mmTiling.SetSplitRange(defaultValue_, defaultValue_, defaultValue_, defaultValue_, defaultValue_, cubeSingleMinK_);
}

bool WeightQuantBatchMatmulV2WeightNz::GetLoopOrder()
{
    uint64_t mteWeight = 1;
    uint64_t vecWeight = 1;
    uint64_t singleCoreM = ops::CeilDiv(inputParams_.mSize, static_cast<uint64_t>(tilingData_->cubeBlockDimM));

    uint64_t mLoop = ops::CeilDiv(singleCoreM, static_cast<uint64_t>(tilingData_->mAL1Size));
    uint64_t singleCoreN = ops::CeilDiv(inputParams_.nSize, static_cast<uint64_t>(tilingData_->cubeBlockDimN));
    uint64_t nLoop = ops::CeilDiv(singleCoreN, static_cast<uint64_t>(tilingData_->nBL1Size));
    // 额外重复载入及重复计算代价，搬运加权暂定为1，计算加权按照指令条数。
    uint64_t extraAub =
        (nLoop - 1) * (mteWeight * singleCoreN * inputParams_.kSize * GetSizeByDataType(inputParams_.aDtype) +
                       vecWeight * singleCoreM * inputParams_.kSize);

    // cast (add) mul nd2nz 4条指令， add可选
    uint64_t extraBub =
        (mLoop - 1) * (mteWeight * singleCoreN * inputParams_.kSize * GetSizeByDataType(inputParams_.bDtype) +
                       vecWeight * singleCoreN * inputParams_.kSize * ANTI_QUANT_TENSOR);

    return extraAub >= extraBub;
}

// 判断A/B在L1是否开DB_BUFFER
void WeightQuantBatchMatmulV2WeightNz::GetL1Pingpong()
{
    auto mmtiling = tilingData_->matmulTiling;
    uint64_t loopM = static_cast<uint64_t>(ops::CeilDiv(mmtiling.singleCoreM, mmtiling.baseM));
    uint64_t loopN = static_cast<uint64_t>(ops::CeilDiv(mmtiling.singleCoreN, mmtiling.baseN));
    uint64_t depthAk = static_cast<uint64_t>(ops::CeilDiv(tilingData_->kAlign, tilingData_->kAL1Size));
    uint64_t depthBk = static_cast<uint64_t>(ops::CeilDiv(tilingData_->kAlign, tilingData_->kBL1Size));
    // if L1fullload both loopM, depthAk is 1
    uint64_t pingPongA = std::min(static_cast<uint64_t>(DB_BUFFER), loopM * depthAk);
    // if L1fullload both loopN, depthBk is 1
    uint64_t pingPongB = std::min(static_cast<uint64_t>(DB_BUFFER), loopN * depthBk);
    uint64_t aL1Size =
        tilingData_->mAL1Size * tilingData_->kAL1Size * GetSizeByDataType(inputParams_.aDtype);
    uint64_t bL1Size =
        tilingData_->kBL1Size * tilingData_->nBL1Size * GetSizeByDataType(inputParams_.aDtype);
    if (pingPongA * aL1Size + pingPongB * bL1Size <= aicoreParams_.l1Size) {
        tilingData_->AL1Pingpong = pingPongA;
        tilingData_->BL1Pingpong = pingPongB;
    } else if (
        pingPongA * aL1Size + bL1Size <= aicoreParams_.l1Size && aL1Size + pingPongB * bL1Size > aicoreParams_.l1Size) {
        tilingData_->AL1Pingpong = pingPongA;
        tilingData_->BL1Pingpong = 1;
    } else if (
        aL1Size + pingPongB * bL1Size <= aicoreParams_.l1Size && pingPongA * aL1Size + bL1Size > aicoreParams_.l1Size) {
        tilingData_->AL1Pingpong = 1;
        tilingData_->BL1Pingpong = pingPongB;
    } else if (
        pingPongA * aL1Size + bL1Size <= aicoreParams_.l1Size &&
        aL1Size + pingPongB * bL1Size <= aicoreParams_.l1Size) {
        // according M/N choose pingPong
        tilingData_->AL1Pingpong = mmtiling.singleCoreN >= mmtiling.singleCoreM ? 1 : pingPongA;
        tilingData_->BL1Pingpong = mmtiling.singleCoreN >= mmtiling.singleCoreM ? pingPongB : 1;
    } else {
        tilingData_->AL1Pingpong = 1;
        tilingData_->BL1Pingpong = 1;
    }
}

void WeightQuantBatchMatmulV2WeightNz::GetL1tiling()
{
    auto mmtiling = tilingData_->matmulTiling;
    tilingData_->mAL1Size = tilingData_->matmulTiling.baseM;
    tilingData_->nBL1Size = tilingData_->matmulTiling.baseN;
    auto minKL1 = std::min(mmtiling.depthA1, mmtiling.depthB1);
    tilingData_->kBL1Size = minKL1 * mmtiling.baseK;
    tilingData_->kAL1Size = minKL1 * mmtiling.baseK;
    L1FullloadMode_ = FullLoadMode::NONE_AB_K;
}

uint64_t WeightQuantBatchMatmulV2WeightNz::GetBubSize(uint64_t bubN, uint64_t bubD) const
{
    return BUB_BUF_NUM * bubN * ops::CeilAlign(bubD, BUB_SIZE_ALIGN);
}

uint64_t WeightQuantBatchMatmulV2WeightNz::GetAubSize(uint64_t aubN, uint64_t aubD) const
{
    return AUB_BUF_NUM * ops::CeilAlign(aubN * aubD, AUB_SIZE_ALIGN);
}

uint64_t WeightQuantBatchMatmulV2WeightNz::CalBubFactorTiling(uint64_t bubCanUseUbSize)
{
    auto mmtiling = tilingData_->matmulTiling;
    auto minKL1 = std::min(tilingData_->kBL1Size, tilingData_->kAL1Size);
    uint64_t bL1Ddim = minKL1;
    uint64_t bL1Ndim = inputParams_.transB ? mmtiling.baseN : minKL1;
    uint64_t bShapeDdim = inputParams_.transB ? inputParams_.kSize : inputParams_.nSize;

    uint64_t bubDdim = bL1Ddim;
    uint64_t bubNdim = bL1Ndim;
    uint64_t bubUsedSize = 0;
    uint64_t bubTempDdim = std::min(bL1Ddim, static_cast<uint64_t>(MAX_BLOCK_STRIDE));
    for (; bubTempDdim >= BLOCK_CUBE; bubTempDdim -= BLOCK_CUBE) {
        if (bubTempDdim < bShapeDdim && bubTempDdim % BLOCK_REDUCE_INT8 != 0) {
            continue;
        }
        for (uint64_t bubTempNdim = MAX_NBUB_SIZE; bubTempNdim >= BLOCK_CUBE; bubTempNdim -= BLOCK_CUBE) {
            if (bL1Ndim % bubTempNdim != 0) {
                continue;
            }
            bubUsedSize = GetBubSize(bubTempNdim, bubTempDdim);
            if (bubUsedSize < bubCanUseUbSize) {
                bubDdim = bubTempDdim;
                bubNdim = bubTempNdim;
                break;
            }
        }
    }
    tilingData_->kBubSize = bubDdim;
    tilingData_->nBubSize = bubNdim;
    return bubUsedSize;
}

uint64_t WeightQuantBatchMatmulV2WeightNz::CalAubFactorTiling(uint64_t aubCanUseUbSize)
{
    auto mmtiling = tilingData_->matmulTiling;
    auto minKL1 = std::min(tilingData_->kBL1Size, tilingData_->kAL1Size);
    uint64_t aL1Ddim = inputParams_.transA ? mmtiling.baseM : minKL1;
    uint64_t aL1Ndim = inputParams_.transA ? minKL1 : mmtiling.baseM;
    uint64_t aubDdim = aL1Ddim;
    uint64_t aubNdim = aL1Ndim;
    uint64_t aubUsedSize = 0;
    uint64_t aubTempDdim = std::min(aL1Ddim, static_cast<uint64_t>(MAX_BLOCK_STRIDE));
    for (; aubTempDdim >= BLOCK_CUBE; aubTempDdim -= BLOCK_CUBE) {
        if (aubTempDdim > 0 && aL1Ddim % aubTempDdim != 0) {
            continue;
        }
        for (uint64_t aubTempNdim = aubNdim; aubTempNdim >= BLOCK_CUBE; aubTempNdim -= BLOCK_CUBE) {
            if (aL1Ndim % aubTempNdim != 0) {
                continue;
            }
            aubUsedSize = GetAubSize(aubTempNdim, aubTempDdim);
            if (aubUsedSize <= aubCanUseUbSize) {
                aubDdim = aubTempDdim;
                aubNdim = aubTempNdim;
                break;
            }
        }
        if (aubUsedSize <= aubCanUseUbSize) {
            break;
        }
    }
    tilingData_->kAubSize = inputParams_.transA ? aubNdim : aubDdim;
    tilingData_->mAubSize = inputParams_.transA ? aubDdim : aubNdim;
    return aubUsedSize;
}

uint64_t WeightQuantBatchMatmulV2WeightNz::CalCubFactorTiling(uint64_t cubNz2NdCanUseSize)
{
    uint64_t cubM = tilingData_->matmulTiling.baseM;
    uint64_t cubN = tilingData_->matmulTiling.baseN;
    uint64_t cubUsedSize = 0;
    // cubM is factor of baseM, cubN is factor of baseN
    uint64_t cubTempN = std::min(cubN, static_cast<uint64_t>(MAX_BLOCK_STRIDE));
    for (uint64_t tempM = cubM; tempM >= BLOCK_CUBE; tempM = tempM - BLOCK_CUBE) {
        if (cubM % tempM != 0) {
            continue;
        }
        for (uint64_t tempN = cubTempN; tempN >= BLOCK_CUBE; tempN = tempN - BLOCK_CUBE) {
            if (cubN % tempN != 0) {
                continue;
            }
            cubUsedSize = tempM * tempN * GetSizeByDataType(ge::DT_FLOAT16);
            if (cubUsedSize < cubNz2NdCanUseSize) {
                tilingData_->nCubSize = tempN;
                break;
            }
        }
        if (cubUsedSize < cubNz2NdCanUseSize) {
            tilingData_->mCubSize = tempM;
            break;
        }
    }
    return cubUsedSize;
}

void WeightQuantBatchMatmulV2WeightNz::PrintCVTilingData(bool debugLevel)
{
    if (debugLevel && CheckLogLevel(OP, DLOG_DEBUG) != 1) {
        return;
    }

    std::stringstream ss;
    ss << "kAlign: " << tilingData_->kAlign << " kSize: " << tilingData_->kSize
       << " nSize: " << tilingData_->nSize << " mSize: " << tilingData_->mSize
       << " cubeBlockDimN: " << static_cast<uint32_t>(tilingData_->cubeBlockDimN)
       << " cubeBlockDimM: " << static_cast<uint32_t>(tilingData_->cubeBlockDimM)
       << " mAubSize: " << tilingData_->mAubSize << " kAubSize: " << tilingData_->kAubSize
       << " nBubSize: " << tilingData_->nBubSize << " kBubSize: " << tilingData_->kBubSize
       << " mCubSize: " << tilingData_->mCubSize << " nCubSize: " << tilingData_->nCubSize
       << " mAL1Size: " << tilingData_->mAL1Size << " kAL1Size: " << tilingData_->kAL1Size
       << " nBL1Size: " << tilingData_->nBL1Size << " kBL1Size: " << tilingData_->kBL1Size
       << " AL1Pingpong: " << tilingData_->AL1Pingpong << " BL1Pingpong: " << tilingData_->BL1Pingpong
       << " batchX0: " << tilingData_->batchX0 << " batchX1: " << tilingData_->batchX1
       << " batchX2: " << tilingData_->batchX2 << " batchX3: " << tilingData_->batchX3
       << " batchWeight0: " << tilingData_->batchWeight0 << " batchWeight1: " << tilingData_->batchWeight1
       << " batchWeight2: " << tilingData_->batchWeight2 << " batchWeight3: " << tilingData_->batchWeight3
       << " batchY0: " << tilingData_->batchY0 << " batchY1: " << tilingData_->batchY1
       << " batchY2: " << tilingData_->batchY2 << " batchY3: " << tilingData_->batchY3 << " biasWithBatch "
       << tilingData_->biasWithBatch;
    // OP_LOG_FULL
    if (debugLevel) {
        OPS_LOG_D(inputParams_.opName, "tiling data: %s", ss.str().c_str());
    } else {
        OPS_LOG_E(inputParams_.opName, "tiling data: %s", ss.str().c_str());
    }
}

ge::graphStatus WeightQuantBatchMatmulV2WeightNz::PostTiling()
{
    size_t tilingDataSize = sizeof(WeightQuantBatchMatmulV2NzTilingData);
    OP_LOGD(inputParams_.opName, "final tiling data size: %zu", tilingDataSize);
    OP_CHECK_IF(
        tilingDataSize % sizeof(uint64_t) != 0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "tiling data size[%zu] not aligned to 8", tilingDataSize),
        return ge::GRAPH_FAILED);

    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    uint32_t usedAicNum = tilingData_->cubeBlockDimM * tilingData_->cubeBlockDimN;
    context_->SetBlockDim(usedAicNum);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), tilingData_, tilingDataSize);
    if (ret != EOK){
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    PrintCVTilingData(true);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling