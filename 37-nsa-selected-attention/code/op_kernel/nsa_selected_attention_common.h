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
* \file nsa_selected_attention_common.h
* \brief
*/

#ifndef NSA_SELECTED_ATTENTION_COMMON_H
#define NSA_SELECTED_ATTENTION_COMMON_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "stdarg.h"

using AscendC::AdjustSoftMaxRes;
using AscendC::AIC;
using AscendC::BinaryRepeatParams;
using AscendC::Cast;
using AscendC::CopyRepeatParams;
using AscendC::Div;
using AscendC::DropOut;
using AscendC::DROPOUT_MODE_BIT_MISALIGN;
using AscendC::DROPOUT_MODE_BYTE_MISALIGN;
using AscendC::DropOutShapeInfo;
using AscendC::Duplicate;
using AscendC::GetBlockIdx;
using AscendC::GetSubBlockIdx;
using AscendC::GetUserWorkspace;
using AscendC::Nd2NzParams;
using AscendC::RoundMode;
using AscendC::SelectWithBytesMask;
using AscendC::SelectWithBytesMaskShapeInfo;
using AscendC::SoftmaxFlashV2;
using AscendC::SoftMaxShapeInfo;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TSCM;

constexpr MatmulConfig CFG_EXCEED = GetNormalConfig(true);
constexpr static uint64_t BLOCK_BYTE = 32;
constexpr static uint64_t BF16_BLOCK_NUM = 16;
constexpr static uint64_t FP32_BLOCK_NUM = 8;
constexpr static int32_t SOFTMAX_M_ALIGNED_SIZE = 8;
constexpr static int32_t SOFTMAX_K_ALIGNED_SIZE = 64;
constexpr static uint32_t SOFTMAX_REDUCE_SIZE = 8;
constexpr static uint64_t DATACOPYPAD_PADDING_VALUE_ZERO = 0;
constexpr static uint32_t NEGATIVE_MIN_VAULE_FP32 = 0xFF7FFFFF;
constexpr static uint32_t NEGATIVE_MIN_VAULE_FP16 = 0xFBFF;
constexpr static uint32_t POSITIVE_MAX_VALUE_FP32 = 0x7F7FFFFF;
constexpr static uint32_t POSITIVE_MAX_VALUE_FP16 = 0x7BFF;
constexpr static uint16_t SOFTMAX_CHECK_RES_DEFAULT_VALUE = 0xFFFF;
constexpr static int64_t attenMaskBN2GS1S2 = 0;
constexpr static int64_t attenMaskBS1S2 = 1;
constexpr static int64_t attenMaskS1S2 = 2;
constexpr static int64_t attenMaskTT = 99;

constexpr static uint32_t SELECTED_BLOCK_COUNT = 16;
constexpr static uint32_t SELECTED_BLOCK_SIZE = 64;
constexpr static uint32_t SELECT_BUFFE_LENGTH = 64 * 16;
constexpr static uint32_t SELECT_KEY_MAXBLOCK = 3;
constexpr static uint32_t DIM_6 = 6;
constexpr static uint32_t DIM_10 = 10;
constexpr static uint32_t SELECT_VALUE_MAXBLOCK = 5;
constexpr static uint32_t G = 4;
constexpr static uint32_t DATA_BLOCK = 16;
constexpr static uint32_t FP32_DATA_BLOCK = 8;
constexpr static uint32_t S64_DATA_BLOCK = 4;

constexpr AscendC::SoftmaxConfig SOFTMAX_DEFAULT_CFG = {true, 0, 0};

// 0级接口的block间隔范围需要满足32B对齐
constexpr static int32_t fp32BaseSize = 8;

enum class SparseModeEnum : uint32_t {
    ALL = 0,
    NONE = 1,
    ANY = 2,
    CAUSAL = 3,
    BAND = 4,
    PREFIX = 5,
    BAND_COMPRESS = 6,
    RIGHT_DOWN_CAUSAL = 7,
    RIGHT_DOWN_CAUSAL_BAND = 8,
    BAND_LEFT_UP_CAUSAL = 9
};

template <typename T> 
__aicore__ inline void GetExtremeValue(T &negativeScalar, T &positiveScalar)
{
    if constexpr (AscendC::IsSameType<T, float>::value) {
        uint32_t tmp1 = NEGATIVE_MIN_VAULE_FP32;
        uint32_t tmp2 = POSITIVE_MAX_VALUE_FP32;
        negativeScalar = *((float *)&tmp1);
        positiveScalar = *((float *)&tmp2);
    } else {
        uint16_t tmp1 = NEGATIVE_MIN_VAULE_FP16;
        uint16_t tmp2 = POSITIVE_MAX_VALUE_FP16;
        negativeScalar = *((half *)&tmp1);
        positiveScalar = *((half *)&tmp2);
    }
}

#endif // NSA_SELECTED_ATTENTION_COMMON_H
