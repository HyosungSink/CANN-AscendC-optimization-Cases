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
 * \file fused_floyd_attention_tiling_common.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <register/op_impl_registry.h>
#include <tiling/tiling_api.h>

#include "log/log.h"

namespace optiling {
namespace FFA {
constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY1_INPUT_INDEX = 1;
constexpr uint32_t VALUE1_INPUT_INDEX = 2;
constexpr uint32_t KEY2_INPUT_INDEX = 3;
constexpr uint32_t VALUE2_INPUT_INDEX = 4;
constexpr uint32_t ATTENTION_MASK_INPUT_INDEX = 5;
constexpr uint32_t SOFTMAXSUM_OUTPUT_INDEX = 1;
// FIXED
constexpr uint32_t ATTENTIONOUT_OUTPUT_INDEX = 2;
constexpr uint32_t MIN_COPY_UINT_SIZE = 32;
constexpr uint32_t TILING_KEY_FP16 = 90;
constexpr uint32_t TILING_KEY_FP32 = 92;
constexpr uint32_t TILING_KEY_BF16 = 94;
constexpr uint32_t SUPPORT_DIM_NUM = 5;
constexpr uint32_t ALIGNED_NUM_16 = 16;
constexpr uint32_t ALIGNED_NUM_128 = 128;

struct FusedFloydAttentionCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0cSize;
};

uint32_t FloydCalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum);
ge::graphStatus CheckAttentionMaskShape(gert::TilingContext *context, int64_t b, int64_t s1, int64_t s3);
ge::graphStatus CheckSupportShape(gert::TilingContext *context);
ge::graphStatus CheckInputShapeValid(gert::TilingContext *context, int64_t b, int64_t n, int64_t s1, int64_t s2,
                                     int64_t s3, int64_t d);
bool CheckSameShape(const gert::StorageShape *aShape, const gert::StorageShape *bShape);
ge::graphStatus CheckAttrs(gert::TilingContext *context);
ge::graphStatus CheckBaseInput(gert::TilingContext *context);

} // namespace FFA
} // namespace optiling
