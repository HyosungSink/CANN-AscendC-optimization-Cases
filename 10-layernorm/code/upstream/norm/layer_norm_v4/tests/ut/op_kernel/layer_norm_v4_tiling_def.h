/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * \file test_layer_norm_v4.h
 * \brief
 */

#ifndef _LAYER_NORM_V4_TILING_H_
#define _LAYER_NORM_V4_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__
#define DTYPE_X float

#pragma pack(1)

struct LayerNormV4TilingDataSingleRead {
    uint32_t numBlocks = 0;
    uint32_t colSize = 0;
    uint32_t rowSize = 0;
    float eps = 0;
    float coefficient = 0;
    uint32_t rowAlign = 0;
    uint32_t nRow = 0;
    uint32_t tailNRow = 0;
    uint32_t loopCount = 0;
    uint32_t tailLoop = 0;
    uint32_t tileLength = 0;
    uint32_t blockLength = 0;
    uint32_t nullptrGamma = 0;
    uint32_t nullptrBeta = 0;
};

#pragma pack()

#pragma pack(1)

struct LayerNormV4TilingDataTranspose {
    uint64_t col = 0;                 // 输入tensor的行
    uint64_t row = 0;                 // 输入tensor的列，即reduce的轴
    uint64_t numBlocks = 0;            // 实际使用的core数量
    uint64_t blockFormer = 0;         // 整核处理的row大小
    uint64_t blockTail = 0;           // 尾核处理的row大小
    uint64_t ubFormer = 0;            // ub整循环处理的row大小
    uint64_t ubLoopOfFormerBlock = 0; // 整核处理的ub循环次数
    uint64_t ubLoopOfTailBlock = 0;   // 尾核处理的ub循环次数
    uint64_t ubTailOfFormerBlock = 0; // 整核ub尾循环处理的row大小
    uint64_t ubTailOfTailBlock = 0;   // 尾核ub尾循环处理的row大小
    uint64_t bFormer = 0;             // ubFormer借轴大小，ubFormer->16*bFormer
    uint64_t dichotomizeAddDiffSize = 0;
    float eps = 0;
    float coefficient = 0;
    uint32_t nullptrGamma = 0;
    uint32_t nullptrBeta = 0;
};

#pragma pack()

#pragma pack(1)

struct LayerNormV4MergeNTilingData {
    uint32_t numBlocks = 0;
    uint32_t colSize = 0;
    uint32_t rowSize = 0;
    float eps = 0;
    float coefficient = 0;
    uint32_t rowAlign = 0;
    uint32_t nRow = 0;
    uint32_t ableNRow = 0;
    uint32_t loopCount = 0;
    uint32_t formerNum = 0;
    uint32_t formerRemainNum = 0;
    uint32_t tailNRow = 0;
    uint32_t tailLoop = 0;
    uint32_t tailNum = 0;
    uint32_t tailRemainNum = 0;
    uint32_t tileLength = 0;
    uint32_t blockLength = 0;

    uint32_t nullptrGamma = 0;
    uint32_t nullptrBeta = 0;
};

#pragma pack()

#pragma pack(1)

struct LayerNormV4CommonTilingData {
    int64_t numBlocks = 0;
    uint32_t colSize = 0;
    uint32_t rowSize = 0;
    float eps = 0;
    uint32_t space = 0;
    float coefficient = 0;
    uint32_t nullptrGamma = 0;
    uint32_t nullptrBeta = 0;
};

#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, LayerNormV4TilingDataSingleRead* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(LayerNormV4TilingDataSingleRead) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, LayerNormV4TilingDataSingleRead* const_data)
{
    memcpy(const_data, tiling, sizeof(LayerNormV4TilingDataSingleRead));
}
#endif

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, LayerNormV4TilingDataTranspose* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(LayerNormV4TilingDataTranspose) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, LayerNormV4TilingDataTranspose* const_data)
{
    memcpy(const_data, tiling, sizeof(LayerNormV4TilingDataTranspose));
}
#endif

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, LayerNormV4MergeNTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(LayerNormV4MergeNTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, LayerNormV4MergeNTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(LayerNormV4MergeNTilingData));
}
#endif

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, LayerNormV4CommonTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(LayerNormV4CommonTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, LayerNormV4CommonTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(LayerNormV4CommonTilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#endif