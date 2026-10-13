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
 * \file deformable_conv2d_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_DEFORMABLE_CONV2D_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_DEFORMABLE_CONV2D_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct DeformableConv2dCompileInfo {
    int32_t coreNum;
};

BEGIN_TILING_DATA_DEF(DeformableConv2dTilingData)
// shape信息
TILING_DATA_FIELD_DEF(int64_t, n)
TILING_DATA_FIELD_DEF(int64_t, inC)
TILING_DATA_FIELD_DEF(int64_t, inH)
TILING_DATA_FIELD_DEF(int64_t, inW)
TILING_DATA_FIELD_DEF(int64_t, outC)
TILING_DATA_FIELD_DEF(int64_t, outH)
TILING_DATA_FIELD_DEF(int64_t, outW)
TILING_DATA_FIELD_DEF(int64_t, kH)
TILING_DATA_FIELD_DEF(int64_t, kW)
// 属性参数
TILING_DATA_FIELD_DEF(int64_t, padTop)
TILING_DATA_FIELD_DEF(int64_t, padLeft)
TILING_DATA_FIELD_DEF(int64_t, strideH)
TILING_DATA_FIELD_DEF(int64_t, strideW)
TILING_DATA_FIELD_DEF(int64_t, dilationH)
TILING_DATA_FIELD_DEF(int64_t, dilationW)
TILING_DATA_FIELD_DEF(int64_t, deformableGroups)
TILING_DATA_FIELD_DEF(int64_t, groups)
TILING_DATA_FIELD_DEF(bool, hasBias)
// vec分核
TILING_DATA_FIELD_DEF(int64_t, slideSizeW)
TILING_DATA_FIELD_DEF(int64_t, groupLen)
TILING_DATA_FIELD_DEF(int64_t, singleVecNum)
TILING_DATA_FIELD_DEF(int64_t, tailVecNum)
// 矩阵乘
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DeformableConv2d, DeformableConv2dTilingData)
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_DEFORMABLE_CONV2D_TILING_H
