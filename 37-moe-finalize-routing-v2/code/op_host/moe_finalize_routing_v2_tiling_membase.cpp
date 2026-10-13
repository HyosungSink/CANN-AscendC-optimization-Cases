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
 * \file moe_finalize_routing_v2_tiling_membase.cpp
 * \brief
 */
#include "moe_finalize_routing_v2_tiling.h"
#include "../common/include/tiling_base/tiling_util.h"
using namespace std;
using namespace ge;

namespace optiling {

static const int64_t WORKSPACE_SIZE_V2 = 16 * 1024 * 1024;
static const int64_t ONE_BLK_SIZE_V2 = 32;
static const size_t TIMES_V2 = 2;
static const int64_t TRIPLE_BUFFER_V2 = 3; // 当数据类型为BF16时转为float类型计算，部分输入所需内存为原来的3倍
static const int64_t DOUBLE_BUFFER_V2 = 2;      // 启动double buffer
static const int64_t H_SIZE_PER_SLICE_V2 = 256; // 切H后，每一块的H大小
static const int64_t INT_NUM_OF_BYTES_V2 = 4;
static const int64_t UNROLL_TIMES_WITH_K2_V2 = 2;
static const int64_t UNROLL_TIMES_WITH_K4_V2 = 4;
static const int64_t BOUND_K_V2 = 256;
static const int64_t ALIGNED_H_V2 = 512;
static const int64_t NETWORKSIZE2_V2 = 5120;
static const int64_t RESERVED_BUFFER = 1024;

static const size_t INDEX_IN_EXPAND_PERMUTED_ROWS_V2 = 0;
static const size_t INDEX_IN_EXPANDED_SRC_TO_DST_ROW_V2 = 1;
static const size_t INDEX_IN_SKIP1_V2 = 2;
static const size_t INDEX_IN_SKIP2_V2 = 3;
static const size_t INDEX_IN_BIAS_V2 = 4;
static const size_t INDEX_IN_SCALES_V2 = 5;
static const size_t INDEX_IN_EXPERT_FOR_SOURCE_ROW_V2 = 6;
static const size_t INDEX_OUT = 0;
static const size_t INPUT_NUM_V2 = 7;
static const size_t SHAPE_SIZE_V2 = 2;
static const size_t SHAPE_SIZE_V3 = 3;
static const int64_t STRIDE_BLOCK_LIMIT_V2 = numeric_limits<uint32_t>::max() / sizeof(int32_t);
static const int64_t DROP_MODE_VALUE_0 = 0;
static const int64_t DROP_MODE_VALUE_1 = 1;
static const int64_t DROP_MODE_VALUE_2 = 2;
static const int64_t DROP_MODE_VALUE_3 = 3;
static const size_t DIM_INDEX_0 = 0;
static const size_t DIM_INDEX_1 = 1;
static const size_t DIM_INDEX_2 = 2;

class MoeFinalizeRoutingV2Membase : public MoeFinalizeRoutingTilingV2
{
public:
    explicit MoeFinalizeRoutingV2Membase(gert::TilingContext* context) : MoeFinalizeRoutingTilingV2(context)
    {}
    ~MoeFinalizeRoutingV2Membase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        MoeFinalizeRoutingTilingV2::Reset(context);
    }

protected:
    ge::graphStatus DoGetPlatformInfo() override;
    ge::graphStatus DoGetShapeAttrsInfo() override;
    ge::graphStatus CalcOpTiling() override;
    ge::graphStatus CalcTilingKey() override;
    void DoPostTiling() override;
    void PrintTilingData() override;

    bool IsCapable() override
    {

        return true;
    }

    ShapeParamsV2 params;
    MoeFinalizeRoutingV2TilingData tilingData;

private:
    void CalcTilingKeyWithScales();
    ge::graphStatus SetParamInfo();
    void GetTilingData() const;
    ge::graphStatus LoadHKAndCalcTiling();
    ge::graphStatus LoadBiasAndCalcTiling();
    ge::graphStatus OptimizedCutH();
    void SetTilingData();
    void CutH();
    ge::graphStatus CalcTilingData();
    int64_t GetAllBiasTilingKey() const;
    int64_t GetTilingKeyForBigK() const;
    int64_t GetTilingKeyForLoadH() const;
    int64_t GetTilingKeyForK2() const;
    int64_t GetTilingKeyForK4() const;
    int64_t GetTilingKeyForDefault() const;
};

inline static int64_t AlignParamV2(const int64_t param, const int32_t typeSize)
{
    return ((param * typeSize + ONE_BLK_SIZE_V2 - 1) / ONE_BLK_SIZE_V2 * ONE_BLK_SIZE_V2) / typeSize;
}

ge::graphStatus MoeFinalizeRoutingV2Membase::DoGetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    totalCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize_ = static_cast<int64_t>(ubSize) - RESERVED_BUFFER;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Membase::SetParamInfo()
{
    // 获取bias的Shape
    biasRowNum_ = 0;
    if (params.biasShape != nullptr && (params.biasShape)->GetStorageShape().GetShapeSize() != 0) {
        biasRowNum_ = params.biasShape->GetStorageShape().GetDim(DIM_INDEX_0);
    }

    // 获取skip1的输入数据类型
    auto expandedXInputDesc = context_->GetInputDesc(INDEX_IN_EXPAND_PERMUTED_ROWS_V2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedXInputDesc);
    dataType_ = expandedXInputDesc->GetDataType();
    inputDataTypeSize_ = ge::GetSizeByDataType(dataType_);

    // 获取 scale 的输入数据类型
    auto scaleInputDesc = context_->GetOptionalInputDesc(INDEX_IN_SCALES_V2);
    scaleDataType_ = dataType_;
    scaleInputDataTypeSize_ = inputDataTypeSize_;
    if (nullptr != scaleInputDesc) {
        scaleDataType_ = scaleInputDesc->GetDataType();
        scaleInputDataTypeSize_ = ge::GetSizeByDataType(scaleDataType_);
    }

    // skip2是否为空，为空时不需要buffer分配
    auto skip2Input = params.x2Shape;
    skip2IsNull_ = skip2Input == nullptr || skip2Input->GetStorageShape().GetShapeSize() == 0 ? 1 : 0;
    skip1IsNull_ = params.x1Shape == nullptr || (params.x1Shape)->GetStorageShape().GetShapeSize() == 0 ? 1 : 0;
    biasIsNull_ = params.biasShape == nullptr || (params.biasShape)->GetStorageShape().GetShapeSize() == 0 ? 1 : 0;
    scalesIsNull_ =
        params.scalesShape == nullptr || (params.scalesShape)->GetStorageShape().GetShapeSize() == 0 ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Membase::DoGetShapeAttrsInfo()
{
    offset_ = 0;
    auto expandedXShapePtr = context_->GetInputShape(INDEX_IN_EXPAND_PERMUTED_ROWS_V2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedXShapePtr);
    auto expandedXShape = expandedXShapePtr->GetStorageShape();

    auto expandedRowIdxShapePtr = context_->GetInputShape(INDEX_IN_EXPANDED_SRC_TO_DST_ROW_V2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedRowIdxShapePtr);
    auto expandedRowIdxShape = expandedRowIdxShapePtr->GetStorageShape();

    auto x1Shape = context_->GetOptionalInputShape(INDEX_IN_SKIP1_V2);

    auto x2Shape = context_->GetOptionalInputShape(INDEX_IN_SKIP2_V2);

    auto biasShape = context_->GetOptionalInputShape(INDEX_IN_BIAS_V2);

    auto scalesShape = context_->GetOptionalInputShape(INDEX_IN_SCALES_V2);

    auto expandedExpertIdxShape = context_->GetOptionalInputShape(INDEX_IN_EXPERT_FOR_SOURCE_ROW_V2);

    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    auto attrsPtr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrsPtr);
    dropPadMode_ = *(attrsPtr->GetAttrPointer<int64_t>(0));

    params.expandedXShape = expandedXShapePtr;
    params.x1Shape = x1Shape;
    params.x2Shape = x2Shape;
    params.biasShape = biasShape;
    params.scalesShape = scalesShape;
    params.expandedRowIdxShape = expandedRowIdxShapePtr;

    OP_CHECK_IF(
        dropPadMode_ < DROP_MODE_VALUE_0 || dropPadMode_ > DROP_MODE_VALUE_3,
        OP_LOGE(context_->GetNodeName(), "the value of drop_pad_mode should be [0,3]."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        expandedRowIdxShape.GetDimNum() != 1,
        OP_LOGE(context_->GetNodeName(), "the expanded_row_idx of input should be 1D tensor."),
        return ge::GRAPH_FAILED);
    int64_t nk = expandedRowIdxShape.GetDim(DIM_INDEX_0);
    k_ = 1;
    if (scalesShape != nullptr) {
        OP_CHECK_IF(
            scalesShape->GetStorageShape().GetDimNum() != SHAPE_SIZE_V2,
            OP_LOGE(context_->GetNodeName(), "the scales of input should be 2D tensor."),
            return ge::GRAPH_FAILED);
        k_ = (params.scalesShape)->GetStorageShape().GetDim(DIM_INDEX_1);

        OP_CHECK_IF(
            k_ == 0, OP_LOGE(context_->GetNodeName(), "K can not be 0."), return ge::GRAPH_FAILED);
    }

    int64_t n = nk / k_;
    totalRowNum_ = n;
    int64_t e = 0;

    if (dropPadMode_ == DROP_MODE_VALUE_0 || dropPadMode_ == DROP_MODE_VALUE_2) {
        OP_CHECK_IF(
            expandedXShape.GetDimNum() != SHAPE_SIZE_V2,
            OP_LOGE(
                context_->GetNodeName(), "the expanded_x of input should be 2D tensor when drop_pad_mode is 0 or 2."),
            return ge::GRAPH_FAILED);
        h_ = expandedXShape.GetDim(DIM_INDEX_1);
    } else {
        OP_CHECK_IF(
            expandedXShape.GetDimNum() != SHAPE_SIZE_V3,
            OP_LOGE(
                context_->GetNodeName(), "the expanded_x of input should be 3D tensor when drop_pad_mode is 1 or 3."),
            return ge::GRAPH_FAILED);
        e = expandedXShape.GetDim(DIM_INDEX_0);
        h_ = expandedXShape.GetDim(DIM_INDEX_2);
    }

    OP_LOGD(context_->GetNodeName(), "keys debug... n:%ld, k:%ld, e:%ld,h:%ld", n, k_, e, h_);
    // check x1
    if (x1Shape != nullptr) {
        OP_CHECK_IF(
            x1Shape->GetStorageShape().GetDimNum() != SHAPE_SIZE_V2,
            OP_LOGE(context_->GetNodeName(), "the x1 of input should be 2D tensor."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            n != x1Shape->GetStorageShape().GetDim(DIM_INDEX_0),
            OP_LOGE(context_->GetNodeName(), "the dim 0 of x1 should be %ld.", n),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            h_ != x1Shape->GetStorageShape().GetDim(DIM_INDEX_1),
            OP_LOGE(context_->GetNodeName(), "the dim 1 of x1 should be %ld.", h_),
            return ge::GRAPH_FAILED);
    }

    // check x2
    if (x2Shape != nullptr) {
        OP_CHECK_IF(
            x1Shape == nullptr,
            OP_LOGE(
                context_->GetNodeName(),
                "In the case of x1 parameter is not input, x2 parameter can't be inputted either."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            x2Shape->GetStorageShape().GetDimNum() != SHAPE_SIZE_V2,
            OP_LOGE(context_->GetNodeName(), "the x2 of input should be 2D tensor."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            n != x2Shape->GetStorageShape().GetDim(DIM_INDEX_0),
            OP_LOGE(context_->GetNodeName(), "the dim 0 of x2 should be %ld.", n),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            h_ != x2Shape->GetStorageShape().GetDim(DIM_INDEX_1),
            OP_LOGE(context_->GetNodeName(), "the dim 1 of x2 should be %ld.", h_),
            return ge::GRAPH_FAILED);
    }

    // check bias
    if (biasShape != nullptr) {
        OP_CHECK_IF(
            expandedExpertIdxShape == nullptr,
            OP_LOGE(context_->GetNodeName(), "the expert_idx must exist when bias exist."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            biasShape->GetStorageShape().GetDimNum() != SHAPE_SIZE_V2,
            OP_LOGE(context_->GetNodeName(), "the bias of input should be 2D tensor."),
            return ge::GRAPH_FAILED);
        if (dropPadMode_ == DROP_MODE_VALUE_1 || dropPadMode_ == DROP_MODE_VALUE_3) {
            OP_CHECK_IF(
                e != biasShape->GetStorageShape().GetDim(DIM_INDEX_0),
                OP_LOGE(
                    context_->GetNodeName(), "the dim 0 of bias should be %ld when drop_pad_mode is 1 or 3.", e),
                return ge::GRAPH_FAILED);
        } else {
            e = biasShape->GetStorageShape().GetDim(DIM_INDEX_0);
        }
        OP_CHECK_IF(
            h_ != biasShape->GetStorageShape().GetDim(DIM_INDEX_1),
            OP_LOGE(context_->GetNodeName(), "the dim 1 of bias should be %ld.", h_),
            return ge::GRAPH_FAILED);
    }

    // check expert_idx
    if (expandedExpertIdxShape != nullptr) {
        OP_CHECK_IF(
            expandedExpertIdxShape->GetStorageShape().GetDimNum() != SHAPE_SIZE_V2,
            OP_LOGE(context_->GetNodeName(), "the expert_idx of input should be 2D tensor."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            n != expandedExpertIdxShape->GetStorageShape().GetDim(DIM_INDEX_0),
            OP_LOGE(context_->GetNodeName(), "the dim 0 of expert_idx should be %ld.", n),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            k_ != expandedExpertIdxShape->GetStorageShape().GetDim(DIM_INDEX_1),
            OP_LOGE(context_->GetNodeName(), "the dim 1 of expert_idx should be %ld.", k_),
            return ge::GRAPH_FAILED);
    }

    if (biasShape != nullptr && expandedExpertIdxShape != nullptr) {
        OP_CHECK_IF(
            biasShape->GetStorageShape().GetDim(DIM_INDEX_0) <
                expandedExpertIdxShape->GetStorageShape().GetDim(DIM_INDEX_1),
            OP_LOGE(context_->GetNodeName(), "E should be larger than or equal to K."),
            return ge::GRAPH_FAILED);
    }

    if (biasShape != nullptr) {
        OP_CHECK_IF(
            (biasShape->GetStorageShape().GetDim(DIM_INDEX_0) == 0),
            OP_LOGE(context_->GetNodeName(), "E can not be 0."), return ge::GRAPH_FAILED);
    }

    // check output
    OP_CHECK_IF(
        yShape.GetDimNum() != SHAPE_SIZE_V2,
        OP_LOGE(context_->GetNodeName(), "the y should be 2D tensor."), return ge::GRAPH_FAILED);
    auto outputDesc = context_->GetOutputDesc(INDEX_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    auto expandedXInputDesc = context_->GetInputDesc(INDEX_IN_EXPAND_PERMUTED_ROWS_V2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedXInputDesc);
    OP_CHECK_IF(
        expandedXInputDesc->GetDataType() != outputDesc->GetDataType(),
        OP_LOGE(context_->GetNodeName(), "the dtype of y should be same with expanded_x."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        n != yShape.GetDim(DIM_INDEX_0),
        OP_LOGE(context_->GetNodeName(), "the dim 0 of output should be %ld.", n),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        h_ != yShape.GetDim(DIM_INDEX_1),
        OP_LOGE(context_->GetNodeName(), "the dim 1 of output should be %ld.", h_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void MoeFinalizeRoutingV2Membase::SetTilingData()
{
    tilingData.set_totalCoreNum(totalCoreNum_);
    tilingData.set_usedCoreNum(usedCoreNum_);
    tilingData.set_skip2IsNull(skip2IsNull_);
    tilingData.set_biasRowNum(biasRowNum_);
    tilingData.set_totalRowNum(totalRowNum_);
    tilingData.set_H(h_);
    tilingData.set_normalH(normalH_);
    tilingData.set_unnormalH(unnormalH_);
    tilingData.set_hSliceNum(hSliceNum_);
    tilingData.set_normalK(normalK_);
    tilingData.set_unnormalK(unnormalK_);
    tilingData.set_kSliceNum(kSliceNum_);
    tilingData.set_K(k_);
    tilingData.set_normalCoreHandleNum(normalCoreHandleNum_);
    tilingData.set_normalCoreLoopNum(normalCoreLoopNum_);
    tilingData.set_normalCoreHandleNumPerLoop(normalCoreHandleNumPerLoop_);
    tilingData.set_normalCoreHandleNumTailLoop(normalCoreHandleNumTailLoop_);
    tilingData.set_tailCoreHandleNum(tailCoreHandleNum_);
    tilingData.set_tailCoreLoopNum(tailCoreLoopNum_);
    tilingData.set_tailCoreHandleNumPerLoop(tailCoreHandleNumPerLoop_);
    tilingData.set_tailCoreHandleNumTailLoop(tailCoreHandleNumTailLoop_);
    tilingData.set_skip1IsNull(skip1IsNull_);
    tilingData.set_scalesIsNull(scalesIsNull_);
    tilingData.set_dropPadMode(dropPadMode_);
}

void MoeFinalizeRoutingV2Membase::CutH()
{
    // int32类型，k按照32字节对齐后的列数
    int64_t alignIntK = AlignParamV2(k_, INT_NUM_OF_BYTES_V2);
    // scale数据类型，k按照32字节对齐后的列数
    int64_t alignK = AlignParamV2(k_, scaleInputDataTypeSize_);
    // 如果K大于256，直接从gm中取值，不需要占用buffer
    if (k_ > BOUND_K_V2) {
        alignIntK = BOUND_K_V2;
        alignK = BOUND_K_V2;
        normalK_ = BOUND_K_V2;
        kSliceNum_ = (k_ + normalK_ - 1) / normalK_;
        unnormalK_ = k_ - k_ / normalK_ * normalK_;
        unnormalK_ = unnormalK_ == 0 ? normalK_ : unnormalK_;
    }
    int64_t skip2IsNeedBuffer = (skip2IsNull_ == 1 ? 0 : 1);
    int64_t biasIsNeedBuffer = (biasIsNull_ == 1 ? 0 : 1);
    int64_t unrollTimes = UNROLL_TIMES_WITH_K2_V2;
    if (k_ == UNROLL_TIMES_WITH_K4_V2) {
        unrollTimes = UNROLL_TIMES_WITH_K4_V2;
    }
    if (dataType_ != ge::DT_BF16) {
        normalH_ = (ubSize_ - alignIntK * INT_NUM_OF_BYTES_V2 - alignK * scaleInputDataTypeSize_) /
                   (1 + skip2IsNeedBuffer + 1 + unrollTimes * biasIsNeedBuffer + unrollTimes + 1) / ONE_BLK_SIZE_V2 *
                   ONE_BLK_SIZE_V2 / inputDataTypeSize_;
    } else {
        normalH_ = (ubSize_ - alignIntK * INT_NUM_OF_BYTES_V2 - alignK * scaleInputDataTypeSize_) /
                   (TRIPLE_BUFFER_V2 + skip2IsNeedBuffer * TRIPLE_BUFFER_V2 + 1 +
                    (unrollTimes * biasIsNeedBuffer + unrollTimes) * TRIPLE_BUFFER_V2) /
                   ONE_BLK_SIZE_V2 * ONE_BLK_SIZE_V2 / inputDataTypeSize_;
    }
    // H切分后，切了多少次
    hSliceNum_ = (h_ + normalH_ - 1) / normalH_ - 1;
    // H切分后，最后一个块的列数
    unnormalH_ = h_ - h_ / normalH_ * normalH_;
    unnormalH_ = unnormalH_ == 0 ? normalH_ : unnormalH_;
    normalCoreLoopNum_ = normalCoreHandleNum_ * (hSliceNum_ + 1);

    // 非尾核，每个核，非尾Loop，每次loop需要处理的skip1行数 (每次loop处理的行数，尽量用满inputUbSize)
    normalCoreHandleNumPerLoop_ = 1;
    // 非尾核，每个核，尾Loop需要处理的skip1行数
    normalCoreHandleNumTailLoop_ = 1;
    // 尾核，非尾Loop需要处理的skip1行数
    tailCoreHandleNumPerLoop_ = min(normalCoreHandleNumPerLoop_, tailCoreHandleNum_);
    // 尾核，非尾loop需要的循环次数
    tailCoreLoopNum_ = (tailCoreHandleNumPerLoop_ == 0) ? 0 : tailCoreHandleNum_ * (hSliceNum_ + 1);
    // 尾核，尾Loop需要处理的skip1行数
    tailCoreHandleNumTailLoop_ = tailCoreHandleNumPerLoop_;
}

ge::graphStatus MoeFinalizeRoutingV2Membase::LoadHKAndCalcTiling()
{
    // int32类型，k按照32字节对齐后的列数
    int64_t alignIntK = AlignParamV2(k_, INT_NUM_OF_BYTES_V2);
    // 输入数据类型，k按照32字节对齐后的列数
    int64_t alignK = AlignParamV2(k_, scaleInputDataTypeSize_);

    // 输入数据类型，h按照32字节对齐后的列数
    int64_t alignH = AlignParamV2(h_, inputDataTypeSize_);
    int64_t skip2IsNeedBuffer = (skip2IsNull_ == 1) ? 0 : 1;
    int64_t biasIsNeedBuffer = (biasIsNull_ == 1) ? 0 : 1;
    // bias和expandedPermutedRows所占的内存大小总和
    int64_t biasAndPermutedRowsBytesSum = (alignH * biasIsNeedBuffer + alignH) * inputDataTypeSize_ * DOUBLE_BUFFER_V2;
    // skip1,skip2,out在单行的情况下所占的内存大小总和
    int64_t skip1Skip2OutBytesSumPerRow = (alignH + alignH * skip2IsNeedBuffer + alignH + alignH) * inputDataTypeSize_;
    if (dataType_ == ge::DT_BF16) {
        biasAndPermutedRowsBytesSum *= TRIPLE_BUFFER_V2;
        skip1Skip2OutBytesSumPerRow =
            (alignH * TRIPLE_BUFFER_V2 + alignH * skip2IsNeedBuffer * TRIPLE_BUFFER_V2 + alignH) * inputDataTypeSize_;
    }

    // 每个loop可以处理的行数，尽可能将内存占满
    int64_t normalCoreHandleNumPerLoop = -1;
    // 如果K大于256，将对K进行切分，最大使用256个数的buffer
    if (k_ > BOUND_K_V2) {
        alignIntK = BOUND_K_V2;
        alignK = BOUND_K_V2;
        normalK_ = BOUND_K_V2;
        kSliceNum_ = (k_ + normalK_ - 1) / normalK_;
        unnormalK_ = k_ - k_ / normalK_ * normalK_;
        unnormalK_ = unnormalK_ == 0 ? normalK_ : unnormalK_;
        normalCoreHandleNumPerLoop = (ubSize_ - biasAndPermutedRowsBytesSum - alignIntK * INT_NUM_OF_BYTES_V2 -
                                      alignK * scaleInputDataTypeSize_) /
                                     skip1Skip2OutBytesSumPerRow;
    } else {
        normalCoreHandleNumPerLoop =
            (ubSize_ - biasAndPermutedRowsBytesSum) /
            (alignIntK * INT_NUM_OF_BYTES_V2 + alignK * scaleInputDataTypeSize_ + skip1Skip2OutBytesSumPerRow);
    }
    if (normalCoreHandleNumPerLoop <= 0) {
        return ge::GRAPH_FAILED; // 计算失败，需要切分H
    }
    // 非尾核，每个核，非尾Loop，每次loop需要处理的skip1行数 (每次loop处理的行数，尽量用满inputUbSize)
    normalCoreHandleNumPerLoop_ = min(normalCoreHandleNumPerLoop, normalCoreHandleNum_);
    // 非尾核，每个核的loop次数
    normalCoreLoopNum_ = (normalCoreHandleNum_ + normalCoreHandleNumPerLoop_ - 1) / normalCoreHandleNumPerLoop_;
    // 非尾核，每个核，尾Loop需要处理的skip1行数
    normalCoreHandleNumTailLoop_ = normalCoreHandleNum_ - (normalCoreLoopNum_ - 1) * normalCoreHandleNumPerLoop_;
    // 尾核，非尾Loop需要处理的skip1行数
    tailCoreHandleNumPerLoop_ = min(normalCoreHandleNumPerLoop_, tailCoreHandleNum_);
    // 尾核，非尾loop需要的循环次数
    tailCoreLoopNum_ = (tailCoreHandleNumPerLoop_ == 0) ?
                           0 :
                           (tailCoreHandleNum_ + tailCoreHandleNumPerLoop_ - 1) / tailCoreHandleNumPerLoop_;
    // 尾核，尾Loop需要处理的skip1行数
    tailCoreHandleNumTailLoop_ = tailCoreHandleNum_ - (tailCoreLoopNum_ - 1) * tailCoreHandleNumPerLoop_;
    normalH_ = h_;
    unnormalH_ = h_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Membase::LoadBiasAndCalcTiling()
{
    if (k_ > BOUND_K_V2) {
        return ge::GRAPH_FAILED;
    }
    // int32类型，k按照32字节对齐后的列数
    int64_t alignIntK = AlignParamV2(k_, INT_NUM_OF_BYTES_V2);
    // 输入数据类型，k按照32字节对齐后的列数
    int64_t alignK = AlignParamV2(k_, scaleInputDataTypeSize_);

    // 输入数据类型，h按照32字节对齐后的列数
    int64_t alignH = AlignParamV2(h_, inputDataTypeSize_);
    int64_t skip2IsNeedBuffer = (skip2IsNull_ == 1) ? 0 : 1;
    int64_t biasIsNeedBuffer = (biasIsNull_ == 1) ? 0 : 1;
    // bias和expandedPermutedRows,tmpbuff所占的内存大小总和
    int64_t biasAndPermutedRowsBytesSum =
        (alignH * biasIsNeedBuffer + alignH) * inputDataTypeSize_ * DOUBLE_BUFFER_V2 + alignH * inputDataTypeSize_;
    // skip1,skip2,out在单行的情况下所占的内存大小总和
    int64_t skip1Skip2OutBytesSumPerRow = (alignH + alignH * skip2IsNeedBuffer + alignH) * inputDataTypeSize_;

    int64_t expandedSrcToDstRowSum = AlignParamV2(normalCoreHandleNum_, INT_NUM_OF_BYTES_V2) * k_ * INT_NUM_OF_BYTES_V2;

    if (dataType_ == ge::DT_BF16) {
        biasAndPermutedRowsBytesSum =
            (alignH * biasIsNeedBuffer + alignH) * inputDataTypeSize_ * DOUBLE_BUFFER_V2 * TRIPLE_BUFFER_V2;
        skip1Skip2OutBytesSumPerRow =
            (alignH * TRIPLE_BUFFER_V2 + alignH * skip2IsNeedBuffer * TRIPLE_BUFFER_V2 + alignH) * inputDataTypeSize_;
    }

    int64_t normalCoreHandleNumPerLoop =
        (ubSize_ - biasAndPermutedRowsBytesSum - expandedSrcToDstRowSum) /
        (skip1Skip2OutBytesSumPerRow + alignIntK * INT_NUM_OF_BYTES_V2 + alignK * scaleInputDataTypeSize_);

    if (normalCoreHandleNumPerLoop <= 0) {
        isCanLoadAllBias_ = false;
        return ge::GRAPH_FAILED;
    }
    // 非尾核，每个核，非尾Loop，每次loop需要处理的skip1行数 (每次loop处理的行数，尽量用满inputUbSize)
    normalCoreHandleNumPerLoop_ = min(normalCoreHandleNumPerLoop, normalCoreHandleNum_);
    // 非尾核，每个核的loop次数
    normalCoreLoopNum_ = (normalCoreHandleNum_ + normalCoreHandleNumPerLoop_ - 1) / normalCoreHandleNumPerLoop_;
    // 非尾核，每个核，尾Loop需要处理的skip1行数
    normalCoreHandleNumTailLoop_ = normalCoreHandleNum_ - (normalCoreLoopNum_ - 1) * normalCoreHandleNumPerLoop_;
    // 尾核，非尾Loop需要处理的skip1行数
    tailCoreHandleNumPerLoop_ = min(normalCoreHandleNumPerLoop_, tailCoreHandleNum_);
    // 尾核，非尾loop需要的循环次数
    tailCoreLoopNum_ = (tailCoreHandleNumPerLoop_ == 0) ?
                           0 :
                           (tailCoreHandleNum_ + tailCoreHandleNumPerLoop_ - 1) / tailCoreHandleNumPerLoop_;
    // 尾核，尾Loop需要处理的skip1行数
    tailCoreHandleNumTailLoop_ = tailCoreHandleNum_ - (tailCoreLoopNum_ - 1) * tailCoreHandleNumPerLoop_;
    normalH_ = h_;
    unnormalH_ = h_;
    isCanLoadAllBias_ = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Membase::OptimizedCutH()
{
    // 对H切分一半并且按照512字节对齐
    normalH_ =
        (h_ * inputDataTypeSize_ / TIMES_V2 + ALIGNED_H_V2 - 1) / ALIGNED_H_V2 * ALIGNED_H_V2 / inputDataTypeSize_;
    hSliceNum_ = (h_ + normalH_ - 1) / normalH_ - 1;

    unnormalH_ = h_ - h_ / normalH_ * normalH_;
    unnormalH_ = unnormalH_ == 0 ? normalH_ : unnormalH_;

    int64_t alignIntK = AlignParamV2(k_, INT_NUM_OF_BYTES_V2);
    int64_t alignK = AlignParamV2(k_, scaleInputDataTypeSize_);

    int64_t alignH = normalH_;
    int64_t skip2IsNeedBuffer = (skip2IsNull_ == 1) ? 0 : 1;
    int64_t biasIsNeedBuffer = (biasIsNull_ == 1) ? 0 : 1;

    int64_t biasAndPermutedRowsBytesSum =
        (alignH * biasIsNeedBuffer + alignH) * inputDataTypeSize_ * UNROLL_TIMES_WITH_K2_V2 +
        alignH * inputDataTypeSize_;

    int64_t skip1Skip2OutBytesSumPerRow = (alignH + alignH * skip2IsNeedBuffer + alignH) * inputDataTypeSize_;

    int64_t expandedSrcToDstRowSum = AlignParamV2(normalCoreHandleNum_, INT_NUM_OF_BYTES_V2) * k_ * INT_NUM_OF_BYTES_V2;

    int64_t normalCoreHandleNumPerLoop =
        (ubSize_ - biasAndPermutedRowsBytesSum - expandedSrcToDstRowSum) /
        (skip1Skip2OutBytesSumPerRow * DOUBLE_BUFFER_V2 + alignIntK * INT_NUM_OF_BYTES_V2 * DOUBLE_BUFFER_V2 +
         alignK * scaleInputDataTypeSize_ * DOUBLE_BUFFER_V2);

    if (normalCoreHandleNumPerLoop <= 0) {
        isOptimizedCutH_ = false;
        return ge::GRAPH_FAILED;
    }

    normalCoreHandleNumPerLoop_ = min(normalCoreHandleNumPerLoop, normalCoreHandleNum_);
    normalCoreLoopNum_ = (normalCoreHandleNum_ + normalCoreHandleNumPerLoop_ - 1) / normalCoreHandleNumPerLoop_;

    normalCoreHandleNumTailLoop_ = normalCoreHandleNum_ - (normalCoreLoopNum_ - 1) * normalCoreHandleNumPerLoop_;

    tailCoreHandleNumPerLoop_ = min(normalCoreHandleNumPerLoop_, tailCoreHandleNum_);
    tailCoreLoopNum_ = (tailCoreHandleNumPerLoop_ == 0) ?
                           0 :
                           (tailCoreHandleNum_ + tailCoreHandleNumPerLoop_ - 1) / tailCoreHandleNumPerLoop_;
    tailCoreHandleNumTailLoop_ = tailCoreHandleNum_ - (tailCoreLoopNum_ - 1) * tailCoreHandleNumPerLoop_;

    isOptimizedCutH_ = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Membase::CalcTilingData()
{
    // 非尾核，每个核处理的skip1行数
    normalCoreHandleNum_ = (totalRowNum_ + totalCoreNum_ - 1) / totalCoreNum_;
    // 使用的核数
    usedCoreNum_ = min((totalRowNum_ + normalCoreHandleNum_ - 1) / normalCoreHandleNum_, totalCoreNum_);
    // 尾核, 需要处理的skip1行数
    tailCoreHandleNum_ = totalRowNum_ - (usedCoreNum_ - 1) * normalCoreHandleNum_;
    if ((h_ == NETWORKSIZE2_V2) && (dataType_ == ge::DT_FLOAT) && (skip1IsNull_ == 0) && (biasIsNull_ == 0) &&
        (k_ % TIMES_V2 == 0) && (dropPadMode_ == 0)) {
        if (OptimizedCutH() == ge::GRAPH_SUCCESS) {
            return ge::GRAPH_SUCCESS;
        }
    }

    if (LoadBiasAndCalcTiling() == ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }
    // 开启double buffer并全载H和K，计算tilingData
    if (LoadHKAndCalcTiling() == ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGI("Tiling4MoeFinalizeRouting", "CalcTilingData load all h_ failed, will cut h_.");
    // 切分H并计算tilingData
    isCanLoadH_ = false;
    CutH();
    return ge::GRAPH_SUCCESS;
}

int64_t MoeFinalizeRoutingV2Membase::GetAllBiasTilingKey() const
{
    if (dataType_ == ge::DT_FLOAT) {
        if (biasIsNull_ == 1) {
            return DTYPE_FLOAT_DB_ALL_BIAS_V2_WITHOUT_BIAS;
        }
        return DTYPE_FLOAT_DB_ALL_BIAS_V2;
    }
    if (dataType_ == ge::DT_FLOAT16) {
        if (biasIsNull_ == 1) {
            return DTYPE_FLOAT16_DB_ALL_BIAS_V2_WITHOUT_BIAS;
        }
        return DTYPE_FLOAT16_DB_ALL_BIAS_V2;
    }
    if (scaleDataType_ == ge::DT_FLOAT) {
        if (biasIsNull_ == 1) {
            return DTYPE_BF16_ALL_BIAS_V2_WITHOUT_BIAS_MIX_PRECISION;
        }
        return DTYPE_BF16_ALL_BIAS_V2_MIX_PRECISION;
    }
    if (biasIsNull_ == 1) {
        return DTYPE_BF16_ALL_BIAS_V2_WITHOUT_BIAS;
    }
    return DTYPE_BF16_ALL_BIAS_V2;
}

int64_t MoeFinalizeRoutingV2Membase::GetTilingKeyForBigK() const
{
    if (dataType_ == ge::DT_FLOAT) {
        if (biasIsNull_ == 1) {
            return DTYPE_FLOAT_BIG_K_V2_WITHOUT_BIAS;
        }
        return DTYPE_FLOAT_BIG_K_V2;
    }
    if (dataType_ == ge::DT_FLOAT16) {
        if (biasIsNull_ == 1) {
            return DTYPE_FLOAT16_BIG_K_V2_WITHOUT_BIAS;
        }
        return DTYPE_FLOAT16_BIG_K_V2;
    }
    if (scaleDataType_ == ge::DT_FLOAT) {
        if (biasIsNull_ == 1) {
            return DTYPE_BF16_BIG_K_V2_WITHOUT_BIAS_MIX_PRECISION;
        }
        return DTYPE_BF16_BIG_K_V2_MIX_PRECISION;
    }
    if (biasIsNull_ == 1) {
        return DTYPE_BF16_BIG_K_V2_WITHOUT_BIAS;
    }
    return DTYPE_BF16_BIG_K_V2;
}

int64_t MoeFinalizeRoutingV2Membase::GetTilingKeyForLoadH() const
{
    if (dataType_ == ge::DT_FLOAT) {
        if (biasIsNull_ == 1) {
            return DTYPE_FLOAT_DB_V2_WITHOUT_BIAS;
        }
        return DTYPE_FLOAT_DB_V2;
    }
    if (dataType_ == ge::DT_FLOAT16) {
        if (biasIsNull_ == 1) {
            return DTYPE_FLOAT16_DB_V2_WITHOUT_BIAS;
        }
        return DTYPE_FLOAT16_DB_V2;
    }
    if (scaleDataType_ == ge::DT_FLOAT) {
        if (biasIsNull_ == 1) {
            return DTYPE_BF16_V2_WITHOUT_BIAS_MIX_PRECISION;
        }
        return DTYPE_BF16_V2_MIX_PRECISION;
    }
    if (biasIsNull_ == 1) {
        return DTYPE_BF16_V2_WITHOUT_BIAS;
    }
    return DTYPE_BF16_V2;
}

int64_t MoeFinalizeRoutingV2Membase::GetTilingKeyForK2() const
{
    if (dataType_ == ge::DT_FLOAT) {
        if (biasIsNull_ == 1) {
            return DTYPE_FLOAT_CUTH_K2_V2_WITHOUT_BIAS;
        }
        return DTYPE_FLOAT_CUTH_K2_V2;
    }
    if (dataType_ == ge::DT_FLOAT16) {
        if (biasIsNull_ == 1) {
            return DTYPE_FLOAT16_CUTH_K2_V2_WITHOUT_BIAS;
        }
        return DTYPE_FLOAT16_CUTH_K2_V2;
    }
    if (scaleDataType_ == ge::DT_FLOAT) {
        if (biasIsNull_ == 1) {
            return DTYPE_BF16_CUTH_K2_V2_WITHOUT_BIAS_MIX_PRECISION;
        }
        return DTYPE_BF16_CUTH_K2_V2_MIX_PRECISION;
    }
    if (biasIsNull_ == 1) {
        return DTYPE_BF16_CUTH_K2_V2_WITHOUT_BIAS;
    }
    return DTYPE_BF16_CUTH_K2_V2;
}

int64_t MoeFinalizeRoutingV2Membase::GetTilingKeyForK4() const
{
    if (dataType_ == ge::DT_FLOAT) {
        if (biasIsNull_ == 1) {
            return DTYPE_FLOAT_CUTH_K4_V2_WITHOUT_BIAS;
        }
        return DTYPE_FLOAT_CUTH_K4_V2;
    }
    if (dataType_ == ge::DT_FLOAT16) {
        if (biasIsNull_ == 1) {
            return DTYPE_FLOAT16_CUTH_K4_V2_WITHOUT_BIAS;
        }
        return DTYPE_FLOAT16_CUTH_K4_V2;
    }
    if (scaleDataType_ == ge::DT_FLOAT) {
        if (biasIsNull_ == 1) {
            return DTYPE_BF16_CUTH_K4_V2_WITHOUT_BIAS_MIX_PRECISION;
        }
        return DTYPE_BF16_CUTH_K4_V2_MIX_PRECISION;
    }
    if (biasIsNull_ == 1) {
        return DTYPE_BF16_CUTH_K4_V2_WITHOUT_BIAS;
    }
    return DTYPE_BF16_CUTH_K4_V2;
}

int64_t MoeFinalizeRoutingV2Membase::GetTilingKeyForDefault() const
{
    if (dataType_ == ge::DT_FLOAT) {
        if (biasIsNull_ == 1) {
            return DTYPE_FLOAT_CUTH_V2_WITHOUT_BIAS;
        }
        return DTYPE_FLOAT_CUTH_V2;
    }
    if (dataType_ == ge::DT_FLOAT16) {
        if (biasIsNull_ == 1) {
            return DTYPE_FLOAT16_CUTH_V2_WITHOUT_BIAS;
        }
        return DTYPE_FLOAT16_CUTH_V2;
    }
    if (scaleDataType_ == ge::DT_FLOAT) {
        if (biasIsNull_ == 1) {
            return DTYPE_BF16_CUTH_V2_WITHOUT_BIAS_MIX_PRECISION;
        }
        return DTYPE_BF16_CUTH_V2_MIX_PRECISION;
    }
    if (biasIsNull_ == 1) {
        return DTYPE_BF16_CUTH_V2_WITHOUT_BIAS;
    }
    return DTYPE_BF16_CUTH_V2;
}

void MoeFinalizeRoutingV2Membase::CalcTilingKeyWithScales()
{
    if (isOptimizedCutH_) {
        tilingKey_ = (biasIsNull_ == 1) ? DTYPE_FLOAT_CUTH_NETWORK_V2_WITHOUT_BIAS : DTYPE_FLOAT_CUTH_NETWORK_V2;
    } else if (isCanLoadAllBias_) {
        tilingKey_ = GetAllBiasTilingKey();
    } else if (k_ > BOUND_K_V2) {
        tilingKey_ = GetTilingKeyForBigK();
    } else if (isCanLoadH_) {
        tilingKey_ = GetTilingKeyForLoadH();
    } else if (k_ == UNROLL_TIMES_WITH_K2_V2) {
        tilingKey_ = GetTilingKeyForK2();
    } else if (k_ == UNROLL_TIMES_WITH_K4_V2) {
        tilingKey_ = GetTilingKeyForK4();
    } else {
        tilingKey_ = GetTilingKeyForDefault();
    }
}

ge::graphStatus MoeFinalizeRoutingV2Membase::CalcTilingKey()
{
    CalcTilingKeyWithScales();

    return ge::GRAPH_SUCCESS;
}

void MoeFinalizeRoutingV2Membase::DoPostTiling()
{
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

void MoeFinalizeRoutingV2Membase::PrintTilingData()
{
    OP_LOGI("MoeFinalizeRoutingV2", "totalCoreNum: %ld", tilingData.get_totalCoreNum());
    OP_LOGI("MoeFinalizeRoutingV2", "usedCoreNum: %ld", tilingData.get_usedCoreNum());
    OP_LOGI("MoeFinalizeRoutingV2", "skip2IsNull: %ld", tilingData.get_skip2IsNull());
    OP_LOGI("MoeFinalizeRoutingV2", "biasRowNum: %ld", tilingData.get_biasRowNum());
    OP_LOGI("MoeFinalizeRoutingV2", "totalRowNum: %ld", tilingData.get_totalRowNum());
    OP_LOGI("MoeFinalizeRoutingV2", "H: %ld", tilingData.get_H());
    OP_LOGI("MoeFinalizeRoutingV2", "normalH: %ld", tilingData.get_normalH());
    OP_LOGI("MoeFinalizeRoutingV2", "unnormalH: %ld", tilingData.get_unnormalH());
    OP_LOGI("MoeFinalizeRoutingV2", "hSliceNum: %ld", tilingData.get_hSliceNum());
    OP_LOGI("MoeFinalizeRoutingV2", "normalK: %ld", tilingData.get_normalK());
    OP_LOGI("MoeFinalizeRoutingV2", "unnormalK: %ld", tilingData.get_unnormalK());
    OP_LOGI("MoeFinalizeRoutingV2", "kSliceNum: %ld", tilingData.get_kSliceNum());
    OP_LOGI("MoeFinalizeRoutingV2", "K: %ld", tilingData.get_K());
    OP_LOGI("MoeFinalizeRoutingV2", "normalCoreHandleNum: %ld", tilingData.get_normalCoreHandleNum());
    OP_LOGI("MoeFinalizeRoutingV2", "normalCoreLoopNum: %ld", tilingData.get_normalCoreLoopNum());
    OP_LOGI("MoeFinalizeRoutingV2", "normalCoreHandleNumPerLoop: %ld", tilingData.get_normalCoreHandleNumPerLoop());
    OP_LOGI("MoeFinalizeRoutingV2", "normalCoreHandleNumTailLoop: %ld", tilingData.get_normalCoreHandleNumTailLoop());
    OP_LOGI("MoeFinalizeRoutingV2", "tailCoreHandleNum: %ld", tilingData.get_tailCoreHandleNum());
    OP_LOGI("MoeFinalizeRoutingV2", "tailCoreLoopNum: %ld", tilingData.get_tailCoreLoopNum());
    OP_LOGI("MoeFinalizeRoutingV2", "tailCoreHandleNumPerLoop: %ld", tilingData.get_tailCoreHandleNumPerLoop());
    OP_LOGI("MoeFinalizeRoutingV2", "tailCoreHandleNumTailLoop: %ld", tilingData.get_tailCoreHandleNumTailLoop());
    OP_LOGI("MoeFinalizeRoutingV2", "skip1IsNull: %ld", tilingData.get_skip1IsNull());
    OP_LOGI("MoeFinalizeRoutingV2", "dropPadMode: %ld", tilingData.get_dropPadMode());
}

ge::graphStatus MoeFinalizeRoutingV2Membase::CalcOpTiling()
{

    // for nk can not equal dim0 of expandedX in regbase
    if (dropPadMode_ == DROP_MODE_VALUE_0 || dropPadMode_ == DROP_MODE_VALUE_2) {
        auto expandedXShape = params.expandedXShape->GetStorageShape();
        auto expandedRowIdxShape = params.expandedRowIdxShape->GetStorageShape();
        int64_t nk = expandedRowIdxShape.GetDim(DIM_INDEX_0);

        OP_CHECK_IF(
            nk != expandedXShape.GetDim(DIM_INDEX_0),
            OP_LOGE(
                context_->GetNodeName(), "the dim 0 of expanded_x should be %ld when drop_pad_mod is 0 or 2", nk),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        SetParamInfo() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "MoeFinalizeRoutingV2Membase get input param info fail."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcTilingData() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "MoeFinalizeRoutingTilingV2 calc tilingData fail."),
        return ge::GRAPH_FAILED);
    SetTilingData();

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MoeFinalizeRoutingV2, MoeFinalizeRoutingV2Membase, 10000);

} // namespace optiling
