/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADD_LAYER_NORM_NORM_TILING_H_
#define ADD_LAYER_NORM_NORM_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct AddLayerNormTilingData {
    uint32_t numCore = 0;
    uint32_t numLastDim = 0;
    uint32_t numFirstDim = 0;
    uint32_t firstDimPerCore = 0;
    uint32_t firstDimPerCoreTail = 0;
    uint32_t firstDimPerTime = 0;
    uint32_t lastDimPerTime = 0;
    float eps = 0;
    float aveFactor = 0;
    uint32_t colMoveCnt = 0;
    uint32_t colTail = 0;
    uint32_t workspaceSize = 0;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                              \
    AddLayerNormTilingData tilingData;                                          \
    INIT_TILING_DATA(AddLayerNormTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).numCore = tilingDataPointer->numCore;                          \
    (tilingData).numLastDim = tilingDataPointer->numLastDim;                    \
    (tilingData).numFirstDim = tilingDataPointer->numFirstDim;                  \
    (tilingData).firstDimPerCore = tilingDataPointer->firstDimPerCore;          \
    (tilingData).firstDimPerCoreTail = tilingDataPointer->firstDimPerCoreTail;  \
    (tilingData).firstDimPerTime = tilingDataPointer->firstDimPerTime;          \
    (tilingData).lastDimPerTime = tilingDataPointer->lastDimPerTime;            \
    (tilingData).eps = tilingDataPointer->eps;                                  \
    (tilingData).aveFactor = tilingDataPointer->aveFactor;                      \
    (tilingData).colMoveCnt = tilingDataPointer->colMoveCnt;                    \
    (tilingData).colTail = tilingDataPointer->colTail;                          \
    (tilingData).workspaceSize = tilingDataPointer->workspaceSize;

#define DTYPE_X1 half
#define DTYPE_X2 half
#define DTYPE_GAMMA half
#define DTYPE_Y half
#define DTYPE_X half
#endif