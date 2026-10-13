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
 * \file add_rms_norm_quant_apt.cpp
 * \brief
 */
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/add_rms_norm_quant_regbase.h"
#include "arch35/add_rms_norm_quant_regbase_perf.h"
#include "arch35/add_rms_norm_quant_regbase_split_reduce.h"
#else
#include "add_rms_norm_quant.h"
#include "add_rms_norm_quant_split_d.h"
#include "add_rms_norm_quant_single_n.h"
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using namespace AddRmsNormQuant;
#endif
using namespace AscendC;

#define INIT_AND_PROCESS                                                                             \
    do {                                                                                             \
        op.Init(x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, y1, y2, x, tilingData); \
        op.Process();                                                                                \
    } while (0)

#ifndef DTYPE_ZERO_POINTS2
#define DTYPE_ZERO_POINTS2 float
#endif

#ifndef DTYPE_ZERO_POINTS1
#define DTYPE_ZERO_POINTS1 DTYPE_ZERO_POINTS2
#endif

extern "C" __global__ __aicore__ void add_rms_norm_quant(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zero_points1, GM_ADDR zero_points2,
    GM_ADDR beta, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    GET_TILING_DATA_WITH_STRUCT(AddRmsNormQuantRegbaseTilingData, tilingDataIn, tiling);
    const AddRmsNormQuantRegbaseTilingData* __restrict tilingData = &tilingDataIn;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    #if !defined(ORIG_DTYPE_SCALES2) && !defined(ORIG_DTYPE_ZERO_POINTS1) && !defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1000)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 0> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1100)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 100> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1001)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 1> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1101)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 101> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1002)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 2> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1102)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 102> op(&pipe);
            INIT_AND_PROCESS;
        }
    #endif
    #if !defined(ORIG_DTYPE_SCALES2) && !defined(ORIG_DTYPE_ZERO_POINTS1) && defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1010)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 10> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1110)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 110> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1011)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 11> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1111)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 111> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1012)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 12> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1112)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 112> op(&pipe);
            INIT_AND_PROCESS;
        }
    #endif
    #if !defined(ORIG_DTYPE_SCALES2) && defined(ORIG_DTYPE_ZERO_POINTS1) && !defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1040)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 40> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1140)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 140> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1041)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 41> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1141)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 141> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1042)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 42> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1142)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 142> op(&pipe);
            INIT_AND_PROCESS;
        }
    #endif
    #if !defined(ORIG_DTYPE_SCALES2) && defined(ORIG_DTYPE_ZERO_POINTS1) && defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1050)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 50> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1150)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 150> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1051)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 51> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1151)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 151> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1052)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 52> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1152)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 152> op(&pipe);
            INIT_AND_PROCESS;
        }
    #endif
    #if defined(ORIG_DTYPE_SCALES2) && !defined(ORIG_DTYPE_ZERO_POINTS1) && !defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1020)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 20> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1120)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 120> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1021)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 21> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1121)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 121> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1022)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 22> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1122)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 122> op(&pipe);
            INIT_AND_PROCESS;
        }
    #endif
    #if defined(ORIG_DTYPE_SCALES2) && !defined(ORIG_DTYPE_ZERO_POINTS1) && defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1030)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 30> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1130)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 130> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1031)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 31> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1131)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 131> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1032)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 32> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1132)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 132> op(&pipe);
            INIT_AND_PROCESS;
        }
    #endif
    #if defined(ORIG_DTYPE_SCALES2) && defined(ORIG_DTYPE_ZERO_POINTS1) && !defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1060)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 60> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1160)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 160> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1061)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 61> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1161)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 161> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1062)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 62> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1162)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 162> op(&pipe);
            INIT_AND_PROCESS;
        }
    #endif
    #if defined(ORIG_DTYPE_SCALES2) && defined(ORIG_DTYPE_ZERO_POINTS1) && defined(ORIG_DTYPE_ZERO_POINTS2)
        if (TILING_KEY_IS(1070)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 70> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1170)) {
            KernelAddRmsNormQuantRegbase<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 170> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1071)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 71> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1171)) {
            KernelAddRmsNormQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 171> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1072)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 72> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(1172)) {
            KernelAddRmsNormQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1, 172> op(&pipe);
            INIT_AND_PROCESS;
        }
    #endif
#else
    GET_TILING_DATA_WITH_STRUCT(AddRMSNormQuantTilingData, tilingDataIn, tiling);
    const AddRMSNormQuantTilingData* __restrict tilingData = &tilingDataIn;
    if (TILING_KEY_IS(0)) {
        KernelAddRmsNormQuant<DTYPE_X1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1)) {
        KernelAddRmsNormQuantSplitD<DTYPE_X1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3)) {
        KernelAddRmsNormQuantSingleN<DTYPE_X1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1> op(&pipe);
        INIT_AND_PROCESS;
    }
#endif
}
