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
 * \file scaled_masked_softmax_v2_tiling.h
 * \brief
 */

#ifndef __ASCENDC_SCALED_MASKED_SOFTMAX_V2_H__
#define __ASCENDC_SCALED_MASKED_SOFTMAX_V2_H__

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScaledMaskedSoftmaxV2TilingData)
TILING_DATA_FIELD_DEF(uint64_t, coreNum);

TILING_DATA_FIELD_DEF(uint64_t, batch);   // 输入x的第1维大小
TILING_DATA_FIELD_DEF(uint64_t, channel); // 输入x的第2维大小
TILING_DATA_FIELD_DEF(uint64_t, height);  // 输入x的第3维大小
TILING_DATA_FIELD_DEF(uint64_t, width);   // 输入x的第4维大小

TILING_DATA_FIELD_DEF(uint64_t, maskBatch);   // 输入mask的第1维大小
TILING_DATA_FIELD_DEF(uint64_t, maskChannel); // 输入mask的第2维大小
TILING_DATA_FIELD_DEF(uint64_t, maskHeight);  // 输入mask的第3维大小
TILING_DATA_FIELD_DEF(uint64_t, maskWidth);   // 输入mask的第4维大小

TILING_DATA_FIELD_DEF(float, scale);                 // scale值
TILING_DATA_FIELD_DEF(uint64_t, maskMode);           // mask和x的shape对齐模式
TILING_DATA_FIELD_DEF(uint64_t, paddingNum);         // 对齐需要补充的数据个数
TILING_DATA_FIELD_DEF(uint64_t, padLineNum);         // 对齐后的一行长度
TILING_DATA_FIELD_DEF(uint64_t, alignedMaskPadding); // mask对齐需要补充的数据个数
TILING_DATA_FIELD_DEF(uint64_t, alignedMaskWidth);   // mask对齐后的一行长度

TILING_DATA_FIELD_DEF(uint64_t, nStep); // x映射mask的batch比例
TILING_DATA_FIELD_DEF(uint64_t, cStep); // x映射mask的channel比例

TILING_DATA_FIELD_DEF(uint64_t, headCoreNum); // 大核数量

TILING_DATA_FIELD_DEF(uint64_t, lineHeadCore);     // 大核处理的行数
TILING_DATA_FIELD_DEF(uint64_t, iterHeadCore);     // 大核的循环数
TILING_DATA_FIELD_DEF(uint64_t, lineHeadIter);     // 大核每次循环的行数
TILING_DATA_FIELD_DEF(uint64_t, lineLastHeadIter); // 大核尾核循环的行数

TILING_DATA_FIELD_DEF(uint64_t, lineTailCore);     // 小核处理的行数
TILING_DATA_FIELD_DEF(uint64_t, iterTailCore);     // 小核的循环数
TILING_DATA_FIELD_DEF(uint64_t, lineTailIter);     // 小核每次循环的行数
TILING_DATA_FIELD_DEF(uint64_t, lineLastTailIter); // 小核尾核循环的行数

TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData); // softmax高阶api所需tiling

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScaledMaskedSoftmaxV2, ScaledMaskedSoftmaxV2TilingData)
} // namespace optiling

#endif // __ASCENDC_SCALED_MASKED_SOFTMAX_V2_H__