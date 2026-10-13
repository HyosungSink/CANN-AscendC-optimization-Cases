/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ADD_RMS_NORM_QUANT_TILING_H_
#define ADD_RMS_NORM_QUANT_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct AddRMSNormQuantTilingData {
    uint32_t numRow;
    uint32_t numCol;
    uint32_t blockFactor;
    uint32_t rowFactor;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    uint32_t hasZeroPoints1;
    uint32_t hasBeta;
    uint32_t divMode;
    uint32_t hasScales2;
    uint32_t hasZeroPoints2;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                 \
    AddRMSNormQuantTilingData tilingData;                                          \
    INIT_TILING_DATA(AddRMSNormQuantTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).numRow = tilingDataPointer->numRow;                               \
    (tilingData).numCol = tilingDataPointer->numCol;                               \
    (tilingData).blockFactor = tilingDataPointer->blockFactor;                     \
    (tilingData).rowFactor = tilingDataPointer->rowFactor;                         \
    (tilingData).ubFactor = tilingDataPointer->ubFactor;                           \
    (tilingData).epsilon = tilingDataPointer->epsilon;                             \
    (tilingData).avgFactor = tilingDataPointer->avgFactor;                         \
    (tilingData).hasZeroPoints1 = tilingDataPointer->hasZeroPoints1;               \
    (tilingData).hasBeta = tilingDataPointer->hasBeta;                             \
    (tilingData).divMode = tilingDataPointer->divMode;                             \
    (tilingData).hasScales2 = tilingDataPointer->hasScales2;                       \
    (tilingData).hasZeroPoints2 = tilingDataPointer->hasZeroPoints2

#define DTYPE_X1 half
#define DTYPE_SCALES1 float
#define DTYPE_ZERO_POINTS1 int32_t

#endif