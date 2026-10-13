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
 * \file deformable_conv2d_tiling.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "log/log.h"
#include "error_util.h"
#include "deformable_conv2d_tiling.h"

namespace optiling {
constexpr int32_t INPUT_TENSOR_NUM = 3;
constexpr int32_t INDEX_ZERO = 0;
constexpr int32_t INDEX_ONE = 1;
constexpr int32_t INDEX_TWO = 2;
constexpr int32_t INDEX_THREE = 3;

constexpr int32_t DIM_FOUR = 4;
constexpr int32_t DIM_ONE = 1;

constexpr uint8_t FLOAT_TYPE = 1;
constexpr uint8_t HALF_TYPE = 2;
constexpr uint8_t BFLOAT_TYPE = 3;
constexpr uint8_t BYTE_LEN_4 = 4;
constexpr uint8_t BYTE_LEN_2 = 2;
constexpr int32_t HALF_BLOCK_NUM = 16;
constexpr int32_t FLOAT_BLOCK_NUM = 8;

constexpr int32_t PADDING_SIZE_CNT = 4;
constexpr int32_t KERNEL_SIZE_CNT = 2;
constexpr int32_t MAX_KERNEL_SIZE = 2048;
constexpr int32_t MAX_MATMUL_K = 65535;
constexpr int32_t INTMAX = 2147483647;

constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;

class DeformableConv2dTiling {
public:
    explicit DeformableConv2dTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus RunBigKernelTiling();

private:
    uint8_t GetDataTypeSize() const;
    ge::graphStatus GetShapes();
    ge::graphStatus GetAttrs();
    ge::graphStatus GetTCubeTiling();
    void SplitCore(int32_t coreNumPlatform);
    void FillTilingData();

    template <typename T1, typename T2>
    inline auto CeilA2B(T1 a, T2 b) const -> T1;

private:
    DeformableConv2dTilingData tilingData;

    ge::DataType dataType = ge::DT_UNDEFINED;
    gert::TilingContext *tilingContext = nullptr;
    gert::Shape xShape;
    gert::Shape weightShape;
    gert::Shape offsetShape;
    gert::Shape biasShape;

    const gert::RuntimeAttrs *attrs = nullptr;
    const gert::ContinuousVector *kernelSizePtr = nullptr;
    const gert::ContinuousVector *stridePtr = nullptr;
    const gert::ContinuousVector *paddingPtr = nullptr;
    const gert::ContinuousVector *dilationPtr = nullptr;
    const int64_t *groups = nullptr;
    const int64_t *deformableGroups = nullptr;

    bool hasBias = false;
    uint8_t dataTypeSize = BYTE_LEN_4;

    int64_t n = 0;
    int64_t inC = 0;
    int64_t inH = 0;
    int64_t inW = 0;
    int64_t outC = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t kH = 0;
    int64_t kW = 0;

    int64_t padTop = 0;
    int64_t padBottom = 0;
    int64_t padLeft = 0;
    int64_t padRight = 0;
    int64_t strideH = 0;
    int64_t strideW = 0;
    int64_t dilationH = 0;
    int64_t dilationW = 0;

    int64_t slideSizeW = 0;
    int64_t groupLen = 0;
};

ge::graphStatus DeformableConv2dTiling::RunBigKernelTiling()
{
    // 获取输入矩阵
    auto xTensor = tilingContext->GetInputTensor(INDEX_ZERO);
    auto weightTensor = tilingContext->GetInputTensor(INDEX_ONE);
    auto offsetTensor = tilingContext->GetInputTensor(INDEX_TWO);
    auto biasTensor = tilingContext->GetOptionalInputTensor(INDEX_THREE);
    if (xTensor == nullptr || weightTensor == nullptr || offsetTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    hasBias = (biasTensor != nullptr);

    // 获取输入的数据类型，输入Tensor的数据类型保持一致
    int32_t inputTensorNum = hasBias ? INPUT_TENSOR_NUM + 1 : INPUT_TENSOR_NUM;
    for (int i = 0; i < inputTensorNum; i++) {
        auto temp = tilingContext->GetInputDesc(i);
        if (dataType == ge::DT_UNDEFINED) {
            dataType = temp->GetDataType();
        } else if (dataType != temp->GetDataType()) {
            return ge::GRAPH_FAILED;
        }
    }

    attrs = tilingContext->GetAttrs();
    if (GetAttrs() != ge::GRAPH_SUCCESS || GetShapes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 预留workspace
    dataTypeSize = GetDataTypeSize();
    size_t *workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = (n * outH * outW * kH * kW * inC) * dataTypeSize + WORK_SPACE_SIZE;

    auto compileInfo = static_cast<const DeformableConv2dCompileInfo *>(tilingContext->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
    int32_t coreNumPlatform = compileInfo->coreNum;
    SplitCore(coreNumPlatform);
    if (GetTCubeTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tilingContext->SetBlockDim(coreNumPlatform);
    tilingContext->SetTilingKey(1);

    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

void DeformableConv2dTiling::SplitCore(int32_t coreNumPlatform)
{
    int32_t vecNum = coreNumPlatform + coreNumPlatform;
    if (*deformableGroups * kH * kW > MAX_KERNEL_SIZE) {
        slideSizeW = 1;
        groupLen = MAX_KERNEL_SIZE / (kH * kW);
    } else {
        int32_t blockNum = (dataType == ge::DT_FLOAT) ? FLOAT_BLOCK_NUM : HALF_BLOCK_NUM;
        int64_t ceilSize = CeilA2B(*deformableGroups * kH * kW, blockNum) * blockNum;
        slideSizeW = MAX_KERNEL_SIZE / ceilSize;
        int64_t slideNumW = CeilA2B(outW, slideSizeW);
        slideSizeW = CeilA2B(outW, slideNumW);
        groupLen = *deformableGroups;
    }
    int64_t singleVecNum = vecNum != 0 ? (n * outH) / vecNum : 0;
    int64_t tailVecNum = vecNum != 0 ? (n * outH) % vecNum : 0;

    tilingData.set_slideSizeW(slideSizeW);
    tilingData.set_groupLen(groupLen);
    tilingData.set_singleVecNum(singleVecNum);
    tilingData.set_tailVecNum(tailVecNum);
}

ge::graphStatus DeformableConv2dTiling::GetTCubeTiling()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTiling;
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, true);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTiling.SetOrgShape(outC / *groups, outW, kH * kW * inC / *groups);
    mmTiling.SetShape(outC / *groups, outW, kH * kW * inC / *groups);
    if (mmTiling.GetTiling(tilingData.mmTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void DeformableConv2dTiling::FillTilingData()
{
    tilingData.set_n(n);
    tilingData.set_inC(inC);
    tilingData.set_inH(inH);
    tilingData.set_inW(inW);
    tilingData.set_outC(outC);
    tilingData.set_outH(outH);
    tilingData.set_outW(outW);
    tilingData.set_kH(kH);
    tilingData.set_kW(kW);

    tilingData.set_padTop(padTop);
    tilingData.set_padLeft(padLeft);
    tilingData.set_strideH(strideH);
    tilingData.set_strideW(strideW);
    tilingData.set_dilationH(dilationH);
    tilingData.set_dilationW(dilationW);
    tilingData.set_deformableGroups(*deformableGroups);
    tilingData.set_groups(*groups);
    tilingData.set_hasBias(hasBias);

    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

ge::graphStatus DeformableConv2dTiling::GetAttrs()
{
    size_t idx = 0;
    kernelSizePtr = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    stridePtr = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    paddingPtr = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    dilationPtr = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    groups = attrs->GetAttrPointer<int64_t>(idx++);
    deformableGroups = attrs->GetAttrPointer<int64_t>(idx++);

    if (kernelSizePtr->GetSize() != KERNEL_SIZE_CNT || stridePtr->GetSize() != DIM_FOUR ||
        paddingPtr->GetSize() != PADDING_SIZE_CNT || dilationPtr->GetSize() != DIM_FOUR) {
        return ge::GRAPH_FAILED;
    }

    const int64_t *kernelSize = static_cast<const int64_t *>(kernelSizePtr->GetData());
    kH = kernelSize[0];
    kW = kernelSize[1];
    if (kH <= 0 || kW <= 0 || kH * kW > MAX_KERNEL_SIZE) {
        return ge::GRAPH_FAILED;
    }

    const int64_t *stride = static_cast<const int64_t *>(stridePtr->GetData());
    const int64_t *dilation = static_cast<const int64_t *>(dilationPtr->GetData());
    // stride和dilation前两维固定为1
    for (int i = 0; i <= 1; i++) {
        if (stride[i] != 1 || dilation[i] != 1) {
            return ge::GRAPH_FAILED;
        }
    }
    strideH = stride[INDEX_TWO];
    strideW = stride[INDEX_THREE];
    dilationH = dilation[INDEX_TWO];
    dilationW = dilation[INDEX_THREE];
    if (strideH <= 0 || strideW <= 0 || dilationH <= 0 || dilationW <= 0) {
        return ge::GRAPH_FAILED;
    }

    const int64_t *padding = static_cast<const int64_t *>(paddingPtr->GetData());
    padTop = padding[INDEX_ZERO];
    padBottom = padding[INDEX_ONE];
    padLeft = padding[INDEX_TWO];
    padRight = padding[INDEX_THREE];

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DeformableConv2dTiling::GetShapes()
{
    xShape = tilingContext->GetInputShape(INDEX_ZERO)->GetOriginShape();
    weightShape = tilingContext->GetInputShape(INDEX_ONE)->GetOriginShape();
    offsetShape = tilingContext->GetInputShape(INDEX_TWO)->GetOriginShape();
    if (xShape.GetDimNum() != DIM_FOUR || weightShape.GetDimNum() != DIM_FOUR || offsetShape.GetDimNum() != DIM_FOUR) {
        return ge::GRAPH_FAILED;
    }

    // x(n, inH, inW, inC)
    n = xShape.GetDim(INDEX_ZERO);
    inH = xShape.GetDim(INDEX_ONE);
    inW = xShape.GetDim(INDEX_TWO);
    inC = xShape.GetDim(INDEX_THREE);
    if (inH * inW > INTMAX) {
        return ge::GRAPH_FAILED;
    }

    // weight(outC, kH, kW, inC/groups)
    outC = weightShape.GetDim(INDEX_ZERO);
    if (weightShape.GetDim(INDEX_ONE) != kH || weightShape.GetDim(INDEX_TWO) != kW || weightShape.GetDim(INDEX_THREE) != inC / *groups) {
        return ge::GRAPH_FAILED;
    }
    if (kH * kW * inC / *groups > MAX_MATMUL_K) {
        return ge::GRAPH_FAILED;
    }

    // offset(n, outH, outW, 3 * deformableGroups * kH * kW)
    int64_t offsetC = 3 * (*deformableGroups) * kH * kW;
    int64_t dilH = (kH - 1) * dilationH + 1;
    int64_t dilW = (kW - 1) * dilationW + 1;
    outH = (inH + padTop + padBottom - dilH) / strideH + 1;
    outW = (inW + padLeft + padRight - dilW) / strideW + 1;
    if (outH <= 0 || outW <= 0) {
        return ge::GRAPH_FAILED;
    }
    if (offsetShape.GetDim(INDEX_ZERO) != n || offsetShape.GetDim(INDEX_ONE) != outH ||
        offsetShape.GetDim(INDEX_TWO) != outW || offsetShape.GetDim(INDEX_THREE) != offsetC) {
        return ge::GRAPH_FAILED;
    }

    // bias(outC)
    if (hasBias) {
        biasShape = tilingContext->GetInputShape(INDEX_THREE)->GetOriginShape();
        if (biasShape.GetDimNum() != DIM_ONE || biasShape.GetDim(INDEX_ZERO) != outC) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

uint8_t DeformableConv2dTiling::GetDataTypeSize() const
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return BYTE_LEN_4;
        case ge::DT_FLOAT16:
            return BYTE_LEN_2;
        case ge::DT_BF16:
            return BYTE_LEN_2;
        default:
            return BYTE_LEN_4;
    }
}

template <typename T1, typename T2>
inline auto DeformableConv2dTiling::CeilA2B(T1 a, T2 b) const -> T1
{
    return (b != 0) ? (a + b - 1) / b : a;
}

static ge::graphStatus Tiling4DeformableConv2dTiling(gert::TilingContext *context)
{
    DeformableConv2dTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepareTiling(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<DeformableConv2dCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(compileInfo->coreNum <= 0,
        OP_LOGE(context->GetNodeName(), "DeformableConv2dTiling GetHardwareInfo Failed"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DeformableConv2d)
    .Tiling(Tiling4DeformableConv2dTiling)
    .TilingParse<DeformableConv2dCompileInfo>(TilingPrepareTiling);
}  // namespace optiling
