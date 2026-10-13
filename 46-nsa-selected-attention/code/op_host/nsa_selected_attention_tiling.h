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
 * \file nsa_selected_attention_tiling.h
 * \brief
 */
#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include "../common/include/tiling_base/data_copy_transpose_tiling_def.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(NsaSelectedAttentionInputParams)
TILING_DATA_FIELD_DEF(int64_t, bSize);  // batch数量
TILING_DATA_FIELD_DEF(int64_t, n2Size); //N2数量
TILING_DATA_FIELD_DEF(int64_t, gSize);  // group数量
TILING_DATA_FIELD_DEF(int64_t, s1Size); //最大q的数据量
TILING_DATA_FIELD_DEF(int64_t, s2Size); //最大kv的数据量
TILING_DATA_FIELD_DEF(int64_t, alignedS2);  //最大q对齐16的数据量
TILING_DATA_FIELD_DEF(int64_t, dSize);  //q和k的D数据量192 
TILING_DATA_FIELD_DEF(int64_t, d2Size);  //v的D数据量128
TILING_DATA_FIELD_DEF(float, scaleValue);   //入参scale_value
TILING_DATA_FIELD_DEF(int64_t, selectedBlockSize);  //入参selected_block_size
TILING_DATA_FIELD_DEF(int64_t, selectedBlockCount); //入参selected_block_count
TILING_DATA_FIELD_DEF(uint8_t, layoutType); // LAYOUT_TND
// 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
TILING_DATA_FIELD_DEF(uint8_t, attenMaskShapeType); //attenMaskshape类型
// 0: fp16, 1: bool(uint8)
TILING_DATA_FIELD_DEF(uint8_t, attenMaskDataType);
// ALL: 0, NONE: 1, ANY: 2, CAUSAL: 3, BAND: 4 };
TILING_DATA_FIELD_DEF(uint8_t, attenMaskCompressMode);
// 0: high precise, 1: high performance, 2: invalid line high precise
TILING_DATA_FIELD_DEF(uint8_t, implMode); // 接口模式 高精度
TILING_DATA_FIELD_DEF(uint8_t, sparseType);  //sparseMode=2是SparseEnum::CAUSAL
TILING_DATA_FIELD_DEF(uint32_t, attenMaskS2Size); //attenMask尾轴大小
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(NsaSelectedAttentionInputParamsOp, NsaSelectedAttentionInputParams)

BEGIN_TILING_DATA_DEF(NsaSelectedAttentionMultiCoreParams)
TILING_DATA_FIELD_DEF(int32_t, coreNum); //实际用到的AIC核数
TILING_DATA_FIELD_DEF(int64_t, headCoreNum); //aic头核
TILING_DATA_FIELD_DEF(int64_t, tailCoreNum); //aic尾核
TILING_DATA_FIELD_DEF(int64_t, s1NumPerHeadCore); //aic头核行数
TILING_DATA_FIELD_DEF(int64_t, s1NumPerTailCore); //aic尾核行数
TILING_DATA_FIELD_DEF(int64_t, localWorkSpaceSize); // softmax计算用localWorkSpaceSize
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(NsaSelectedAttentionMultiCoreParamsOp, NsaSelectedAttentionMultiCoreParams)


BEGIN_TILING_DATA_DEF(NsaSelectedAttentionEmptyInputTilingData)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);      // AIC NUM
TILING_DATA_FIELD_DEF(uint32_t, attentionOutFormerNum); // 输出的头核
TILING_DATA_FIELD_DEF(uint32_t, attentionOutTailNum);   // 输出的尾核
TILING_DATA_FIELD_DEF(uint32_t, softmaxMaxFormerNum);   // softmaxMax头核和softmaxSum一致
TILING_DATA_FIELD_DEF(uint32_t, softmaxMaxTailNum);     // softmaxMax尾核和softmaxSum一致
TILING_DATA_FIELD_DEF(uint64_t, attentionOutSingleCoreDataSize);    // attentionOut的每个头核处理的数据个数
TILING_DATA_FIELD_DEF(uint64_t, attentionOutTailCoreDataSize);  // attentionOut的每个尾核处理的数据个数
TILING_DATA_FIELD_DEF(uint64_t, softmaxMaxSingleCoreDataSize);  // attentionOut的每个头核处理的数据个数
TILING_DATA_FIELD_DEF(uint64_t, softmaxMaxTailCoreDataSize);    // attentionOut的每个尾核处理的数据个数
TILING_DATA_FIELD_DEF(uint64_t, attentionOutLastCoreDataSize);  // 最后一个核应该处理的数据量
TILING_DATA_FIELD_DEF(uint64_t, attentionOutLastCoreIndex); // 最后一个核起始地址
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(NsaSelectedAttentionEmptyInputTilingDataOp, NsaSelectedAttentionEmptyInputTilingData)

BEGIN_TILING_DATA_DEF(NsaSelectedAttentionTilingData)
TILING_DATA_FIELD_DEF_STRUCT(NsaSelectedAttentionInputParams, inputParams);
TILING_DATA_FIELD_DEF_STRUCT(NsaSelectedAttentionMultiCoreParams, multiCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingData);  // 第一个矩阵乘的参数对象
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingData);  // 第二个矩阵乘的参数对象
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);    // softmax高级api需要的参数对象
TILING_DATA_FIELD_DEF_STRUCT(NsaSelectedAttentionEmptyInputTilingData, emptyInputTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(NsaSelectedAttention, NsaSelectedAttentionTilingData)

struct NsaSelectedAttentionCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
};
} //namesapce optiling
