/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RMS_NORM_TILING_H_
#define RMS_NORM_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct RMSNormTilingData {
    uint64_t num_row;
    uint64_t num_col;
    uint64_t num_col_align;
    uint64_t block_factor;
    uint32_t row_factor;
    uint32_t ub_factor;
    uint32_t reduce_mask;
    uint32_t left_num;
    uint32_t last_reduce_mask;
    uint32_t last_left_num;
    uint32_t rstd_size;
    uint32_t ub_loop;
    uint32_t col_buffer_length;
    uint32_t multi_n_num;
    uint32_t is_nddma;
    float epsilon;
    float avg_factor;
    uint8_t is_gemma;
    uint64_t last_block_factor;
    uint64_t row_loop;
    uint64_t last_block_row_loop;
    uint64_t row_tail;
    uint64_t last_block_row_tail;
    uint32_t mul_loop;
    uint32_t mul_tail;
    uint8_t dst_rep_stride;
    uint8_t is_performance;
    uint8_t normal_flag;
};
#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                         \
    RMSNormTilingData tilingData;                                          \
    INIT_TILING_DATA(RMSNormTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).num_row = tilingDataPointer->num_row;                     \
    (tilingData).num_col = tilingDataPointer->num_col;                     \
    (tilingData).num_col_align = tilingDataPointer->num_col_align;         \
    (tilingData).block_factor = tilingDataPointer->block_factor;           \
    (tilingData).row_factor = tilingDataPointer->row_factor;               \
    (tilingData).ub_factor = tilingDataPointer->ub_factor;                 \
    (tilingData).reduce_mask = tilingDataPointer->reduce_mask;             \
    (tilingData).left_num = tilingDataPointer->left_num;                   \
    (tilingData).last_reduce_mask = tilingDataPointer->last_reduce_mask;   \
    (tilingData).last_left_num = tilingDataPointer->last_left_num;         \
    (tilingData).rstd_size = tilingDataPointer->rstd_size;                 \
    (tilingData).ub_loop = tilingDataPointer->ub_loop;                     \
    (tilingData).col_buffer_length = tilingDataPointer->col_buffer_length; \
    (tilingData).multi_n_num = tilingDataPointer->multi_n_num;             \
    (tilingData).is_nddma = tilingDataPointer->is_nddma;                   \
    (tilingData).epsilon = tilingDataPointer->epsilon;                     \
    (tilingData).avg_factor = tilingDataPointer->avg_factor;               \
    (tilingData).is_gemma = tilingDataPointer->is_gemma;                   \
    (tilingData).last_block_factor = tilingDataPointer->last_block_factor;                 \
    (tilingData).row_loop = tilingDataPointer->row_loop;                   \
    (tilingData).last_block_row_loop = tilingDataPointer->last_block_row_loop;             \
    (tilingData).last_block_row_tail = tilingDataPointer->last_block_row_tail;             \
    (tilingData).mul_loop = tilingDataPointer->mul_loop;                   \
    (tilingData).mul_tail = tilingDataPointer->mul_tail;                   \
    (tilingData).dst_rep_stride = tilingDataPointer->dst_rep_stride;       \
    (tilingData).is_performance = tilingDataPointer->is_performance;       \
    (tilingData).normal_flag = tilingDataPointer->normal_flag;

#define DTYPE_X half
#define DTYPE_GAMMA half

#endif