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
 * \file weight_quant_batch_matmul_v2_checker_for_mmads8s4.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_checker_for_mmads8s4.h"
#include "matmul/common/op_host/math_util.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"

namespace {
constexpr size_t MM_SHAPE_LEN_ND = 2UL;
constexpr size_t MIN_SHAPE_LEN_ND = 2UL;
constexpr size_t MAX_SHAPE_LEN_ND = 6UL;
constexpr uint64_t MAX_SHAPE_DIM = 65535UL;
constexpr uint64_t MIN_GROUP_SIZE = 32UL;
constexpr uint64_t MAX_INT32 = 2147483647UL;
constexpr uint32_t BIAS_INDEX = 6;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t LAST_BATCH_DIM_INDEX = 3;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
}

namespace optiling {
ge::graphStatus WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckContext()
{
    // check Raw TilingData
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    // check the Required input and output desc
    size_t idx = 0;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx++));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx++));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx++));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    inputParams_.opName = context_->GetNodeName();
    // OP_LOG_FULL
    OPS_LOG_I(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    return ge::GRAPH_SUCCESS;
}

/*
The function is check the dtype limit:
    1. Input x dtype should be fp16, weight should be int8
    2. Output dtype should be same with x dtype without quant
    3. antiquant scale dtype should be uint64
    4. Bias dtype should be int32/fp16
    5. other params should be null
*/
bool WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckDtype()
{
    size_t idx = 0;
    inputParams_.aDtype = context_->GetInputDesc(idx++)->GetDataType();
    inputParams_.bDtype = context_->GetInputDesc(idx++)->GetDataType();
    inputParams_.antiQuantScaleDtype = context_->GetInputDesc(idx++)->GetDataType();
    inputParams_.cDtype = context_->GetOutputDesc(0)->GetDataType();
    OP_TILING_CHECK(inputParams_.aDtype != ge::DT_FLOAT16,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input x dtype must be FLOAT16, but is [%s]",
                                                    ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str()),
                    return false);

    OP_TILING_CHECK(inputParams_.bDtype != ge::DT_INT8,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input weight dtype must be INT8, but is [%s]",
                                                    ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
                    return false);

    OP_TILING_CHECK(
        inputParams_.antiQuantScaleDtype != ge::DT_UINT64 && inputParams_.antiQuantScaleDtype != ge::DT_INT64,
        VECTOR_INNER_ERR_REPORT_TILIING(
            inputParams_.opName, "Input antiquant scale dtype must be UINT64/INT64, but is [%s]",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.antiQuantScaleDtype).c_str()),
        return false);

    OP_TILING_CHECK(inputParams_.cDtype != ge::DT_FLOAT16 && inputParams_.cDtype != ge::DT_INT8,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Output y dtype must be FLOAT16/INT8, but is [%s]",
                                                    ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str()),
                    return false);

    auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
    if (biasDesc != nullptr) {
        inputParams_.biasDtype = biasDesc->GetDataType();
        OP_TILING_CHECK(inputParams_.biasDtype != ge::DT_FLOAT16 && inputParams_.biasDtype != ge::DT_INT32,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams_.opName,"Input bias dtype must be FLOAT16/INT32, but is [%s]",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str()),
                        return false);
    }
    return true;
}

/*
The function is check the attr limit:
    1. Group size should be 0
    2. innerPrecise only support 0 or 1
*/
bool WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckAttr()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    auto transposeX = attrs->GetAttrPointer<bool>(idx++);
    auto transposeWeight = attrs->GetAttrPointer<bool>(idx++);
    const int64_t *groupSizePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        groupSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }
    idx++;  // 跳过dtype属性
    const int64_t *innerPrecisePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        innerPrecisePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }
    inputParams_.transA = transposeX != nullptr && *transposeX;
    inputParams_.transB = transposeWeight != nullptr && *transposeWeight;

    if (groupSizePtr != nullptr) {
        OP_TILING_CHECK(
            *groupSizePtr != 0,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Group size should be 0, but is [%ld]", *groupSizePtr),
            return false);
        inputParams_.groupSize = static_cast<uint64_t>(*groupSizePtr);
    }
    if (innerPrecisePtr != nullptr) {
        OP_TILING_CHECK((*innerPrecisePtr != 0) && (*innerPrecisePtr != 1),
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams_.opName, "innerPrecise only support 0 or 1, but is [%ld]", *innerPrecisePtr),
                        return false);
        inputParams_.innerPrecise = static_cast<uint64_t>(*innerPrecisePtr);
    }
    return true;
}

/*
The function is check the shape limit:
    1. the input x shape dims should be [2, 6](ND), weight shape dims should be [2, 6](ND)
    2. the k of x and weight must be same
    3. antiquant shape must be:
        per_channel: not trans_n: (1, n) or (n); trans_b: (n, 1) or (n)
        per_tensor: (1,)
    4. quant shape must be empty tensor
*/
bool WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckShape()
{
    size_t idx = 0;
    auto xShape = context_->GetInputShape(idx++);
    auto weightShape = context_->GetInputShape(idx++);
    auto antiQuantScaleShape = context_->GetInputShape(idx++);
    auto antiQuantOffsetShape = context_->GetOptionalInputShape(idx++);
    auto quantScaleShape = context_->GetOptionalInputShape(idx++);
    auto quantOffsetShape = context_->GetOptionalInputShape(idx++);
    auto biasShape = context_->GetOptionalInputShape(idx++);
    auto outputShape = context_->GetOutputShape(0);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, weightShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, antiQuantScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    // not yet support empty tensor for input
    OP_TILING_CHECK(xShape->GetStorageShape().GetShapeSize() == 0 ||
                        weightShape->GetStorageShape().GetShapeSize() == 0 ||
                        antiQuantScaleShape->GetStorageShape().GetShapeSize() == 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Not yet support empty tensor"), return false);
    // check x, weight
    inputParams_.aFormat = GetInputStorageFormat(context_, 0);
    inputParams_.bFormat = GetInputStorageFormat(context_, 1);
    OP_TILING_CHECK(!CheckInputShape(xShape, weightShape),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check input x and weight shape failed"),
                    return false);
    // check batch
    OP_TILING_CHECK(!CheckBatchInfo(xShape->GetOriginShape(), weightShape->GetOriginShape()),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                                                    "Batch dimension should be equal when their value not 1."),
                    return false);
    // check antiquant scale, antiquant offset
    OP_TILING_CHECK(!CheckAntiQuantShape(antiQuantScaleShape, antiQuantOffsetShape),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check antiquant shape failed"), return false);
    // check quant scale, quant offset
    OP_TILING_CHECK(quantScaleShape != nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Quant scale should be nullptr"),
                    return false);
    OP_TILING_CHECK(quantOffsetShape != nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Quant offset should be nullptr"),
                    return false);
    // check bias
    if (biasShape != nullptr) {
        OP_TILING_CHECK(biasShape->GetStorageShape().GetDimNum() != 1,
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Bias dimension should equal to 1"),
                                                        return false);
        OP_TILING_CHECK(static_cast<uint64_t>(biasShape->GetStorageShape().GetDim(0)) != inputParams_.nSize,
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Bias size should be equal to nsize"),
                                                        return false);
    } 
    // check dim value
    OP_TILING_CHECK(inputParams_.groupSize >= inputParams_.kSize || inputParams_.groupSize % MIN_GROUP_SIZE != 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        inputParams_.opName, "Group sizes should not more than [%lu] and align to 32, but is [%lu]",
                        inputParams_.kSize, inputParams_.groupSize), return false);
    return true;
}

bool WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckInputShape(const gert::StorageShape *xShape,
                                                               const gert::StorageShape *weightShape)
{
    OP_TILING_CHECK(inputParams_.aFormat != ge::Format::FORMAT_ND && inputParams_.aFormat != ge::Format::FORMAT_NCHW,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Input x format should be ND , actual format is %s",
                                          ge::TypeUtils::FormatToSerialString(inputParams_.aFormat).c_str()),
                    return false);

    OP_TILING_CHECK(inputParams_.bFormat != ge::Format::FORMAT_ND && inputParams_.bFormat != ge::Format::FORMAT_NCHW,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Input weight format should be ND , actual format is %s",
                                          ge::TypeUtils::FormatToSerialString(inputParams_.bFormat).c_str()),
                    return false);

    size_t xOriDimNum = xShape->GetOriginShape().GetDimNum();
    size_t weightOriDimNum = weightShape->GetOriginShape().GetDimNum();
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t weightDimNum = weightShape->GetStorageShape().GetDimNum();

    OP_TILING_CHECK(xOriDimNum < MIN_SHAPE_LEN_ND || xOriDimNum > MAX_SHAPE_LEN_ND,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "OriginalShape of x can not be greater than %zu and \
                                        less than %zu, but is [%zu]", MAX_SHAPE_LEN_ND, MIN_SHAPE_LEN_ND, xOriDimNum),
        return false);

    OP_TILING_CHECK(weightOriDimNum < MIN_SHAPE_LEN_ND || weightDimNum > MAX_SHAPE_LEN_ND,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "OriginalShape of weight can not be greater than %zu and \
                                        less than %zu, but is [%zu]", MAX_SHAPE_LEN_ND, MIN_SHAPE_LEN_ND,
                                        weightOriDimNum),
        return false);

    OP_TILING_CHECK(xDimNum < MIN_SHAPE_LEN_ND || xDimNum > MAX_SHAPE_LEN_ND,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "StorageShape of x can not be greater than %zu and \
                                        less than %zu, but is [%zu]", MAX_SHAPE_LEN_ND, MIN_SHAPE_LEN_ND, xDimNum),
        return false);

    OP_TILING_CHECK(weightDimNum < MIN_SHAPE_LEN_ND || weightDimNum > MAX_SHAPE_LEN_ND,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "StorageShape of weight can not be greater \
                                                    than %zu and less than %zu, but is [%zu]", MAX_SHAPE_LEN_ND,
                                                    MIN_SHAPE_LEN_ND, weightDimNum),
                    return false);

    auto xShapeLen = xShape->GetOriginShape().GetDimNum();
    auto weightShapeLen = weightShape->GetOriginShape().GetDimNum();
    uint64_t weightLastDim = weightShape->GetOriginShape().GetDim(weightShapeLen - LAST_FIRST_DIM_INDEX);
    uint64_t weightFirstDim = weightShape->GetOriginShape().GetDim(weightShapeLen - LAST_SECOND_DIM_INDEX);
    uint64_t xFirstDim = xShape->GetOriginShape().GetDim(xShapeLen- LAST_SECOND_DIM_INDEX);
    uint64_t xLastDim = xShape->GetOriginShape().GetDim(xShapeLen- LAST_FIRST_DIM_INDEX);
    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ? xLastDim : xFirstDim);
    inputParams_.kSize = static_cast<uint64_t>(inputParams_.transA ? xFirstDim : xLastDim);
    inputParams_.nSize = static_cast<uint64_t>(inputParams_.transB ? weightFirstDim : weightLastDim);

    auto kBSize = static_cast<uint64_t>(inputParams_.transB ? weightLastDim : weightFirstDim);
    OP_TILING_CHECK(inputParams_.kSize != kBSize,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                                                    "K dim of x and weight must equal, but they are [%lu] and [%lu]",
                                                    inputParams_.kSize, kBSize),
                    return false);
    return true;
}

bool WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckBatchInfo(const gert::Shape &xShape,
                                                              const gert::Shape &weightShape) const
{
    auto x1DimNum = xShape.GetDimNum();
    auto x2DimNum = weightShape.GetDimNum();
    auto outDimNum = std::max(x1DimNum, x2DimNum);
    const gert::Shape &shapeLong = x1DimNum > x2DimNum ? xShape : weightShape;
    const gert::Shape &shapeShort = x1DimNum > x2DimNum ? weightShape : xShape;
    size_t validOffset = outDimNum - std::min(x1DimNum, x2DimNum);
    for (size_t i = 0; i < outDimNum - LAST_SECOND_DIM_INDEX; i++) {
        auto shortDim = i < validOffset ? 1 : shapeShort.GetDim(i - validOffset);
        auto longDim = shapeLong.GetDim(i);
        if (shortDim > 1 && longDim > 1 && shortDim != longDim) {
            return false;
        }
    }
    return true;
}

bool WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckAntiQuantShape(const gert::StorageShape *antiQuantScaleShape,
                                                                   const gert::StorageShape *antiQuantOffsetShape)
{
    size_t antiQuantScaleDimNum = antiQuantScaleShape->GetStorageShape().GetDimNum();
    size_t antiQuantScaleShapeSize = static_cast<size_t>(antiQuantScaleShape->GetStorageShape().GetShapeSize());
    OP_TILING_CHECK(antiQuantScaleDimNum > MM_SHAPE_LEN_ND,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        inputParams_.opName, "antiquant scale shape size should not be more than 2, but is [%zu]",
                        antiQuantScaleDimNum),
                    return false);
    if (antiQuantScaleShapeSize != 1) {
        if (antiQuantScaleDimNum == MM_SHAPE_LEN_ND) {
            gert::Shape expectShape;
            uint64_t kNum = inputParams_.groupSize > 0 ? ops::CeilDiv(inputParams_.kSize, inputParams_.groupSize) : 1;
            if (inputParams_.transB) {
                expectShape.AppendDim(static_cast<int64_t>(inputParams_.nSize));
                expectShape.AppendDim(static_cast<int64_t>(kNum));
            } else {
                expectShape.AppendDim(static_cast<int64_t>(kNum));
                expectShape.AppendDim(static_cast<int64_t>(inputParams_.nSize));
            }
            OP_TILING_CHECK(
                expectShape != antiQuantScaleShape->GetStorageShape(),
                VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Antiquant shape expect %s, but is %s",
                                                Ops::Base::ToString(expectShape).c_str(),
                                                Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str()),
                return false);
            inputParams_.antiQuantType = inputParams_.groupSize > 0 ? QuantType::PER_GROUP : QuantType::PER_CHANNEL;
        } else {
            OP_TILING_CHECK(antiQuantScaleShapeSize != inputParams_.nSize,
                            VECTOR_INNER_ERR_REPORT_TILIING(
                                inputParams_.opName, "Antiquant size should be n size when perchannel, but is %s",
                                Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str()),
                            return false);
            inputParams_.antiQuantType = QuantType::PER_CHANNEL;
        }
    } else {
        inputParams_.antiQuantType = QuantType::PER_TENSOR;
    }

    OP_TILING_CHECK(antiQuantOffsetShape != nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Antiquant offset should be nullptr"),
                    return false);
    return true;
}

ge::graphStatus WeightQuantBatchMatmulV2Checker4MmadS8S4::Check()
{
    auto ret = CheckContext();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    // check the input and output dtype: x, weight, antiquant_scale, y.
    // In RESERVED, antiquant_offset, quant_scale, quant_offset don't need to check, cause they are nullptr
    OP_TILING_CHECK(!CheckDtype(),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check input dtype failed"),
                    return ge::GRAPH_FAILED);
    // check attrs: transA, transB, group_size, dtype, innerPrecise
    OP_TILING_CHECK(!CheckAttr(),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check attr failed"), return ge::GRAPH_FAILED);
    // check the input and output shape: all
    OP_TILING_CHECK(!CheckShape(),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check shape failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling