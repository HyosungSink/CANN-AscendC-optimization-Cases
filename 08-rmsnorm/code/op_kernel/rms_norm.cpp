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
#if ((defined(__CCE_AICORE__) && __CCE_AICORE__ != 310) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113)) \
         && (defined(__NPU_ARCH__) && __NPU_ARCH__ != 5102)
#include "rms_norm_split_d.h"
#endif
#if ((defined(__CCE_AICORE__) && __CCE_AICORE__ == 310) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 5102)) \
 	     && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113)
#include "arch35/rms_norm_regbase.h"
#include "arch35/rms_norm_regbase_split_d.h"
#endif
#include "rms_norm_whole_reduce_sum.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 100
#include "rms_norm_merge_n.h"
#include "rms_norm_single_row.h"
#endif

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
#if ((defined(__CCE_AICORE__) && __CCE_AICORE__ != 310) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113)) \
 	     && (defined(__NPU_ARCH__) && __NPU_ARCH__ != 5102)
    if (TILING_KEY_IS(RMSNORM_TILING_NORMAL)) {
        GENERAL_OP_IMPL(KernelRmsNorm, DTYPE_X, DTYPE_GAMMA);
    } else if (TILING_KEY_IS(RMSNORM_TILING_SPILT_D)) {
        GENERAL_OP_IMPL(KernelRmsNormSplitD, DTYPE_X, DTYPE_GAMMA);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 100
    } else if (TILING_KEY_IS(RMSNORM_TILING_MERGE_N)) {
        GENERAL_OP_IMPL(KernelRmsNormMergeN, DTYPE_X, DTYPE_GAMMA);
    } else if (TILING_KEY_IS(RMSNORM_TILING_SINGLE_ROW)) {
        GENERAL_OP_IMPL(KernelRmsNormSingleRow, DTYPE_X, DTYPE_GAMMA);
#endif
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
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
#endif
    }
#endif
#if ((defined(__CCE_AICORE__) && __CCE_AICORE__ == 310) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 5102)) \
 	     && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(5000)) {
        GENERAL_OP_IMPL(RmsNorm::KernelRmsNormRegBase, DTYPE_X, DTYPE_GAMMA);
    } else if (TILING_KEY_IS(2001)) {
        GENERAL_OP_IMPL(RmsNorm::KernelRmsNormRegBaseSplitD, DTYPE_X, DTYPE_GAMMA);
    }
#endif
}
