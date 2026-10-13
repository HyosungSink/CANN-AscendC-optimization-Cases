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
 * \file rms_norm_apt.cpp
 * \brief rmsnorm kernel file
 */
#include "rms_norm.h"
#include "rms_norm_split_d.h"
#include "rms_norm_whole_reduce_sum.h"
#include "rms_norm_merge_n.h"
#include "rms_norm_single_row.h"

using namespace AscendC;
using namespace RmsNorm;

#define RMSNORM_TILING_NORMAL 0
#define RMSNORM_TILING_SPILT_D 1
#define RMSNORM_TILING_MERGE_N 2
#define RMSNORM_TILING_SINGLE_ROW 3

#define GENERAL_OP_IMPL(templateClass, ...)      \
    do {                                         \
        templateClass<__VA_ARGS__> op;           \
        op.Init(x, gamma, y, rstd, &tilingData); \
        op.Process();                            \
    } while (0)

extern "C" __global__ __aicore__ void rms_norm(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(RMSNORM_TILING_NORMAL)) {
        GENERAL_OP_IMPL(KernelRmsNorm, DTYPE_X, DTYPE_GAMMA);
    } else if (TILING_KEY_IS(RMSNORM_TILING_SPILT_D)) {
        GENERAL_OP_IMPL(KernelRmsNormSplitD, DTYPE_X, DTYPE_GAMMA);
    } else if (TILING_KEY_IS(RMSNORM_TILING_MERGE_N)) {
        GENERAL_OP_IMPL(KernelRmsNormMergeN, DTYPE_X, DTYPE_GAMMA);
    } else if (TILING_KEY_IS(RMSNORM_TILING_SINGLE_ROW)) {
        GENERAL_OP_IMPL(KernelRmsNormSingleRow, DTYPE_X, DTYPE_GAMMA);
    } else if (TILING_KEY_IS(1110)) {
        GENERAL_OP_IMPL(KernelRmsNormWholeReduceSum, half, true);
    } else if (TILING_KEY_IS(1120)) {
        GENERAL_OP_IMPL(KernelRmsNormWholeReduceSum, float, true);
    } else if (TILING_KEY_IS(1130)) {
        GENERAL_OP_IMPL(KernelRmsNormWholeReduceSum, bfloat16_t, true);
    } else if (TILING_KEY_IS(1010)) {
        GENERAL_OP_IMPL(KernelRmsNormWholeReduceSum, half, false);
    } else if (TILING_KEY_IS(1020)) {
        GENERAL_OP_IMPL(KernelRmsNormWholeReduceSum, float, false);
    } else if (TILING_KEY_IS(1030)) {
        GENERAL_OP_IMPL(KernelRmsNormWholeReduceSum, bfloat16_t, false);
    }
}
