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
 * \file quant_batch_matmul_v3_tiling_arch20.cpp
 * \brief
 */

#include <vector>
#include <cmath>
#include "log/log.h"
#include "matmul/common/op_host/math_util.h"
#include "error_util.h"
#include "quant_batch_matmul_v3_tiling_arch20.h"
#include "quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_tiling_key.h"
#include "../../../transpose_batch_mat_mul/op_host/op_tiling/pp_matmul_default.h"
using Ops::NN::MathUtil;

namespace optiling {

constexpr uint32_t INDEX_X1 = 0;
constexpr uint32_t INDEX_X2 = 1;
constexpr uint32_t INDEX_SCALE = 2;
constexpr uint32_t INDEX_OFFSET = 3;
constexpr uint32_t INDEX_BIAS = 4;
constexpr uint32_t INDEX_PERTOKEN = 5;
constexpr uint32_t INDEX_ATTR_TRANS_A = 1;
constexpr uint32_t INDEX_ATTR_TRANS_B = 2;
constexpr uint32_t CONST_ZERO = 0;
constexpr uint32_t CONST_ONE = 1;
constexpr uint32_t CONST_TWO = 2;
constexpr uint32_t CONST_THREE = 3;

bool IsSocVersionArch20Pertoken(const gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE("Arch20Pertoken: ", "context is nullptr"), return false);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2002 && context->GetOptionalInputDesc(INDEX_PERTOKEN) != nullptr) {
        return true;
    }
    return false;
}

ge::graphStatus QuantBatchMatmulPertokenArch20::GetShapeAttrsInfo()
{
    OP_TILING_CHECK(context_ == nullptr, OP_LOGE("Arch20Pertoken: ", "context is nullptr"), return ge::GRAPH_FAILED;);
    params_.opName = context_->GetNodeName();
    params_.formatA = context_->GetInputDesc(INDEX_X1)->GetStorageFormat();
    params_.formatB = context_->GetInputDesc(INDEX_X2)->GetStorageFormat();
    params_.transA = *(context_->GetAttrs()->GetAttrPointer<bool>(INDEX_ATTR_TRANS_A));
    params_.transB = *(context_->GetAttrs()->GetAttrPointer<bool>(INDEX_ATTR_TRANS_B));
    params_.dtypeA = context_->GetInputDesc(INDEX_X1)->GetDataType();
    params_.dtypeB = context_->GetInputDesc(INDEX_X2)->GetDataType();
    params_.sizeInDtype = static_cast<float>(ge::GetSizeByDataType(params_.dtypeA));

    auto aShape = context_->GetInputShape(INDEX_X1)->GetOriginShape();
    auto bShape = context_->GetInputShape(INDEX_X2)->GetOriginShape();
    auto cShape = context_->GetOutputShape(INDEX_X1)->GetOriginShape();
    size_t aDims = aShape.GetDimNum();
    size_t bDims = bShape.GetDimNum();

    OP_TILING_CHECK((aDims < CONST_TWO || bDims < CONST_TWO),
                    OP_LOGE(params_.opName, "x1 and x2 must have at least 2 dimensions."), return ge::GRAPH_FAILED;);
    size_t batch_dims_count_x1 = aDims - CONST_TWO;
    size_t batch_dims_count_x2 = bDims - CONST_TWO;
    OP_TILING_CHECK((batch_dims_count_x1 != batch_dims_count_x2),
                    OP_LOGE(params_.opName, "x1 and x2 must have same batch."), return ge::GRAPH_FAILED;);

    uint32_t total_batch = 1;
    for (size_t i = 0; i < batch_dims_count_x1; ++i) {
        OP_TILING_CHECK(((aShape[i] != bShape[i]) || (aShape[i] == 0)),
                        OP_LOGE(params_.opName, "x1 and x2 must have same batch and batch can't be 0."),
                        return ge::GRAPH_FAILED;);
        total_batch *= aShape[i];
    }

    params_.batchSize = total_batch;
    params_.m = aShape[aDims - CONST_TWO];
    params_.k = aShape[aDims - CONST_ONE];
    params_.n = bShape[bDims - CONST_TWO];

    // Process Bias
    if (context_->GetOptionalInputDesc(INDEX_BIAS) != nullptr &&
        context_->GetOptionalInputDesc(INDEX_OFFSET) == nullptr) {
        qbmmTilingDataArch20_.withBias = true;
        auto biasShape = context_->GetOptionalInputShape(INDEX_BIAS)->GetOriginShape();
        if (biasShape.GetDimNum() == 1) {
            qbmmTilingDataArch20_.biasWithBatch = false;
        } else if (biasShape.GetDimNum() == CONST_THREE) {
            qbmmTilingDataArch20_.biasWithBatch = true;
        } else {
            OP_LOGW(context_->GetNodeName(), "Arch20 Pertoken mode bias only support [n] or [b, 1, n]");
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulPertokenArch20::PostTiling()
{
    // TilingData
    qbmmTilingDataArch20_.batchSize = tilingData_.opShape.batchSize;
    qbmmTilingDataArch20_.m = tilingData_.opShape.m;
    qbmmTilingDataArch20_.k = tilingData_.opShape.k;
    qbmmTilingDataArch20_.n = tilingData_.opShape.n;
    qbmmTilingDataArch20_.m0 = tilingData_.opShape.m0;
    qbmmTilingDataArch20_.k0 = tilingData_.opShape.k0;
    qbmmTilingDataArch20_.n0 = tilingData_.opShape.n0;
    qbmmTilingDataArch20_.mLoop = tilingData_.mLoop;
    qbmmTilingDataArch20_.nLoop = tilingData_.nLoop;
    qbmmTilingDataArch20_.kLoop = tilingData_.kLoop;
    qbmmTilingDataArch20_.coreLoop = tilingData_.coreLoop;
    qbmmTilingDataArch20_.blockDim = tilingData_.blockDim;
    qbmmTilingDataArch20_.swizzleDirect = tilingData_.swizzleDirect;
    qbmmTilingDataArch20_.swizzleCount = tilingData_.swizzleCount;
    // TilingData Memory Copy
    size_t tilingDataSize = sizeof(QuantMatmulPertokenTilingDataArch20);
    errno_t ret = memcpy_s(
        context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
        reinterpret_cast<void*>(&qbmmTilingDataArch20_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(qbmmTilingDataArch20_.blockDim);
    // Workspace
    auto platformInfo = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    size_t sysWorkspaceSize = platformInfo.GetLibApiWorkSpaceSize();
    size_t* currentWorkSpace = context_->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBatchMatmulPertokenArch20::DoTiling()
{
    OP_TILING_CHECK(context_ == nullptr, OP_LOGE("Arch20Pertoken: ", "context is nullptr"), return false);
    params_.isPertokenArch20 = true;
    params_.isInt8 = true;
    tiling_.GetHardwareInfo();
    GetShapeAttrsInfo();
    if (!tiling_.GetMatMulTilingData()) {
        return ge::GRAPH_FAILED;
    }
    uint64_t TRANS = 1;                // B trans
    uint64_t KERNEL_TEMPLATE_TYPE = 1; // basic
    uint64_t IS_PERTOKEN = 1;          // pertoken
    uint64_t OPTION_ATTRS = 0;         // option_none
    tilingKey_ = GET_TPL_TILING_KEY(TRANS, KERNEL_TEMPLATE_TYPE, IS_PERTOKEN, OPTION_ATTRS);
    ge::graphStatus ret = PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
