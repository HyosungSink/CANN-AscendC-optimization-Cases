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
 * \file detect_flash_attention_score_tiling_common.h
 * \brief
 */

#pragma once

#include <cstdint>

namespace optiling {

constexpr size_t INPUT_INPUT_INDEX = 0;
constexpr size_t WEIGHT_INPUT_INDEX = 1;
constexpr size_t ACT_SEQ_LEN_INPUT_INDEX = 2;

constexpr size_t INPUTLAYOUT_ATTRS_INDEX = 0;
constexpr size_t COMPRESS_BLOCK_SIZE_ATTRS_INDEX = 1;
constexpr size_t COMPRESS_STRIDE_ATTRS_INDEX = 2;
constexpr size_t ACT_SEQ_LEN_TYPE_ATTRS_INDEX = 3;

constexpr uint32_t MIN_COPY_UINT_SIZE = 32;

const size_t Zero = 0;
const size_t One = 1;
const size_t Two = 2;

struct NsaCompressCompileInfo {
    uint32_t aivNum;
    uint64_t ubSize;
};

} // namespace optiling
