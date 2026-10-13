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
 * \file nsa_compress_common.h
 * \brief
 */

#ifndef NSA_COMPRESS_SEQ_COMMON_H
#define NSA_COMPRESS_SEQ_COMMON_H

#include "kernel_operator.h"


namespace NASCompress {

const int32_t NSASuccess = 0;
const int32_t NSADone = 1;
const int32_t NSAFAILED = -1;

const int32_t MAX_OVERLAP_NUM = 20;
const int32_t HAS_OVERLAP = 1;
const int32_t NO_OVERLAP = 0;

const size_t Zero = 0;
const size_t One = 1;
const size_t Two = 2;
const size_t Four = 4;
const size_t Eight = 8;
const size_t SixTeen = 16;


enum class CompressState : uint16_t {
    COMPRESS_TOKEN_INITIATED = 1,
    COMPRESS_TOKEN_PROCESSING = 2,
    COMPRESS_TOKEN_COMPLETED = 3,
    COMPRESS_TOKEN_UNDEF = 99,
};

enum class SampleHeadState : uint16_t {
    SAMPLE_FIRST_COMPRESS_TOKEN = 0,
    SAMPLE_MID_COMPRESS_TOKEN = 1,
    SAMPLE_LAST_COMPRESS_TOKEN = 2,
    SAMPLE_UNDEF_COMPRESS_TOKEN = 99,
};

enum class SampleHead : int32_t {
    SAMPLE_HEAD_VALID = 100,
    SAMPLE_HEAD_INVALID = 999,
};

enum class WeightOffsetType : uint8_t {
    TOKEN_OFFSET = 0,
    BUFFSET_OFFSET = 1,
    UNDEF_OFFSET = 255,
};

} // namespace NASCompress

#endif // NSA_COMPRESS_SEQ_COMMON_H