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
 * \file nsa_compre_tiling.h
 * \brief
 */

#pragma once

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../common/include/tiling_base/tiling_templates_registry.h"

namespace optiling {
const size_t MAX_CORE_NUM = 48; // 使用的CORE数

BEGIN_TILING_DATA_DEF(NsaCompressTilingData)
TILING_DATA_FIELD_DEF(uint32_t, InputDtype);        // 输入kv的总大小，初始化时指定kvGM的大小
TILING_DATA_FIELD_DEF(uint32_t, TotalKvSize);       // 输入kv的总大小，初始化时指定kvGM的大小
TILING_DATA_FIELD_DEF(uint32_t, TotalCompressSize); // 输出compress_kv的总大小，初始化时指定compressKvGM的大小
TILING_DATA_FIELD_DEF(uint32_t, WeightSize);        // 输入weight的总大小，初始化时指定weightGM的大小
TILING_DATA_FIELD_DEF(uint32_t, BatchSize);         // 输入Kv的batch数，用于指定actseqlenGM的大小
TILING_DATA_FIELD_DEF(uint32_t, CompressBlockSize); // 属性CompressBlockSize值
TILING_DATA_FIELD_DEF(uint32_t, CompressStride);    // 属性CompressStride值
TILING_DATA_FIELD_DEF(uint32_t, HeadNum);           // 数据的HeadNum
TILING_DATA_FIELD_DEF(uint32_t, HeadDim);           // 数据的HeadDim
TILING_DATA_FIELD_DEF(uint32_t, BlocksNums);        // 单次搬移kv的seq数量
TILING_DATA_FIELD_DEF(uint32_t, DivHeadNums);
TILING_DATA_FIELD_DEF(uint32_t, DivSeqNums);
TILING_DATA_FIELD_DEF(uint32_t, MaxOverlap);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 48, PerCoreHeadIdx);           // 各core处理的head起始索引
TILING_DATA_FIELD_DEF_ARR(uint32_t, 48, PerCoreHeadNum);           // 各core处理的head数
TILING_DATA_FIELD_DEF_ARR(uint32_t, 48, PerCoreOutputNum);         // 各core的输出token数
TILING_DATA_FIELD_DEF_ARR(uint32_t, 48, PerCoreStartOutputOffset); // 各core输出token占总输出token的偏移量
TILING_DATA_FIELD_DEF_ARR(uint32_t, 48, kvStartBatchIdx);          // 各core处理的首个kv所在batch
TILING_DATA_FIELD_DEF_ARR(uint32_t, 48, kvStartTokenIdx);          // 各core处理的首个kv所在的seq_len索引
TILING_DATA_FIELD_DEF_ARR(uint32_t, 48, kvStartOffset);            // 各core处理的首个kv相对所有kv的偏移量
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NsaCompress, NsaCompressTilingData)
} // namespace optiling
