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
 * \file grouped_matmul_finalize_routing.cpp
 * \brief
 */
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#include "grouped_matmul_finalize_routing.h"
#include "grouped_matmul_finalize_routing_antiquant_a8w4_msd_pre.h"
#include "grouped_matmul_finalize_routing_antiquant_a8w4_msd.h"
#include "grouped_matmul_finalize_routing_antiquant_a8w4_msd_l1_opt.h"
#endif

namespace GroupedMatmulFinalizeRouting {
constexpr MatmulConfig A8W4_GMM_CFG_MDL = GetNormalConfig();

#if defined(FORMAT_W) && defined(FORMAT_FRACTAL_NZ) && (FORMAT_W == FORMAT_FRACTAL_NZ)
constexpr CubeFormat wFormat = CubeFormat::NZ;
#else
constexpr CubeFormat wFormat = CubeFormat::ND;
#endif
} // namespace GroupedMatmulFinalizeRouting

using namespace AscendC;
using namespace matmul;
using namespace GroupedMatmulFinalizeRouting;

#define GMMFR_A8W8_IMPL(RowIndexDtype, scaleType, groupListType, sharedInputIsNone)                                    \
    do {                                                                                                               \
        TPipe pipe;                                                                                                    \
        MT mm;                                                                                                         \
        if ASCEND_IS_AIC {                                                                                             \
            mm.SetSubBlockIdx(0);                                                                                      \
            mm.Init(&tilingData.matmulTiling, &pipe);                                                                  \
        }                                                                                                              \
        using param =                                                                                                  \
            Param<true, RowIndexDtype, GroupMatmulFRTilingData, scaleType, groupListType, sharedInputIsNone>;          \
        QuantGroupMatmul<param> op(mm);                                                                                \
        op.Init(initParams, &tilingData, &pipe);                                                                       \
        op.Process();                                                                                                  \
    } while (0)

#define GMMFR_A8W4_IMPL(groupListType, sharedInputIsNone)                                                              \
    do {                                                                                                               \
        TPipe pipe;                                                                                                    \
        if ASCEND_IS_AIV {                                                                                             \
            using param = ParamW4A8Pre<groupListType>;                                                                 \
            GMMA8W4PreProcess<param> op1;                                                                              \
            MMPreInitParams preInitParams{x, x, group_list, workspaceGM};                                              \
            op1.Init(preInitParams, tilingData, &pipe);                                                                \
            op1.Process();                                                                                             \
            pipe.Reset();                                                                                              \
            pipe.Destroy();                                                                                            \
            pipe.Init();                                                                                               \
        }                                                                                                              \
        using aT = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;                                          \
        using bT = MatmulType<TPosition::GM, wFormat, int4b_t, false>;                                                 \
        using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;                                       \
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;                                             \
        using matmulType = MMImplType<aT, bT, cT, biasT, GroupedMatmulFinalizeRouting::A8W4_GMM_CFG_MDL,               \
                                      groupListType, sharedInputIsNone>;                                               \
        matmulType::MT mm;                                                                                             \
        if ASCEND_IS_AIC {                                                                                             \
            mm.SetSubBlockIdx(0);                                                                                      \
            mm.Init(&tilingData.matmulTiling, &pipe);                                                                  \
        }                                                                                                              \
        GMMA8W4MSDCompute<matmulType> op(mm);                                                                          \
        op.Init(initParams, &tilingData, &pipe);                                                                       \
        op.Process();                                                                                                  \
    } while (0)

#define GMMFR_A8W4_L1_OPT_IMPL(groupListType, sharedInputIsNone)                                                       \
    do {                                                                                                               \
        TPipe pipe;                                                                                                    \
        if ASCEND_IS_AIV {                                                                                             \
            using param = ParamW4A8Pre<groupListType>;                                                                 \
            GMMA8W4PreProcess<param> op1;                                                                              \
            MMPreInitParams preInitParams{x, x, group_list, user};                                                     \
            op1.Init(preInitParams, tilingData, &pipe);                                                                \
            op1.Process();                                                                                             \
            pipe.Reset();                                                                                              \
            pipe.Destroy();                                                                                            \
            pipe.Init();                                                                                               \
        }                                                                                                              \
        using bT = MatmulType<TPosition::GM, wFormat, int4b_t, false>;                                                 \
        using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;                                       \
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;                                             \
        using aT = MatmulType<TPosition::TSCM, CubeFormat::NZ, int4b_t, false>;                                        \
        using matmulType = MMImplType<aT, bT, cT, biasT, GroupedMatmulFinalizeRouting::A8W4_GMM_CFG_MDL,               \
                                      groupListType, sharedInputIsNone>;                                               \
        matmulType::MT mm;                                                                                             \
        if ASCEND_IS_AIC {                                                                                             \
            mm.SetSubBlockIdx(0);                                                                                      \
            mm.Init(&tilingData.matmulTiling, &pipe);                                                                  \
        }                                                                                                              \
        GMMA8W4MSDL1OptCompute<matmulType> op(mm);                                                                     \
        op.Init(initParams, &tilingData, &pipe);                                                                       \
        op.Process();                                                                                                  \
    } while (0)

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
extern "C" __global__ __aicore__ void grouped_matmul_finalize_routing(GM_ADDR x, GM_ADDR w, GM_ADDR scale, GM_ADDR bias,
                                                                      GM_ADDR pertoken_scale, GM_ADDR group_list,
                                                                      GM_ADDR share_input, GM_ADDR logit,
                                                                      GM_ADDR row_index, GM_ADDR offset, GM_ADDR y,
                                                                      GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);
    MMInitParams initParams{x,      w,     bias,      group_list,  scale, pertoken_scale,
                            offset, logit, row_index, share_input, y,     workspaceGM};
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(10000000000000000001UL)) {
        GMMFR_A8W8_IMPL(int64_t, float, false, false);
    } else if (TILING_KEY_IS(10000000000000000011UL)) {
        GMMFR_A8W8_IMPL(int32_t, float, false, false);
    } else if (TILING_KEY_IS(10000000000000000101UL)) {
        GMMFR_A8W8_IMPL(int64_t, bfloat16_t, false, false);
    } else if (TILING_KEY_IS(10000000000000000111UL)) {
        GMMFR_A8W8_IMPL(int32_t, bfloat16_t, false, false);
    } else if (TILING_KEY_IS(11000000000000000011UL)) {
        GMMFR_A8W4_IMPL(false, false);
    } else if (TILING_KEY_IS(11000000000000000111UL)) {
        GMMFR_A8W4_L1_OPT_IMPL(false, false);
    } else if (TILING_KEY_IS(10000000000000001001UL)) {
        GMMFR_A8W8_IMPL(int64_t, float, true, false);
    } else if (TILING_KEY_IS(10000000000000001011UL)) {
        GMMFR_A8W8_IMPL(int32_t, float, true, false);
    } else if (TILING_KEY_IS(10000000000000001101UL)) {
        GMMFR_A8W8_IMPL(int64_t, bfloat16_t, true, false);
    } else if (TILING_KEY_IS(10000000000000001111UL)) {
        GMMFR_A8W8_IMPL(int32_t, bfloat16_t, true, false);
    } else if (TILING_KEY_IS(11000000000000001011UL)) {
        GMMFR_A8W4_IMPL(true, false);
    } else if (TILING_KEY_IS(11000000000000001111UL)) {
        GMMFR_A8W4_L1_OPT_IMPL(true, false);
    } else if (TILING_KEY_IS(10000000000000010001UL)) {
        GMMFR_A8W8_IMPL(int64_t, float, false, true);
    } else if (TILING_KEY_IS(10000000000000010011UL)) {
        GMMFR_A8W8_IMPL(int32_t, float, false, true);
    } else if (TILING_KEY_IS(10000000000000010101UL)) {
        GMMFR_A8W8_IMPL(int64_t, bfloat16_t, false, true);
    } else if (TILING_KEY_IS(10000000000000010111UL)) {
        GMMFR_A8W8_IMPL(int32_t, bfloat16_t, false, true);
    } else if (TILING_KEY_IS(11000000000000010011UL)) {
        GMMFR_A8W4_IMPL(false, true);
    } else if (TILING_KEY_IS(11000000000000010111UL)) {
        GMMFR_A8W4_L1_OPT_IMPL(false, true);
    } else if (TILING_KEY_IS(10000000000000011001UL)) {
        GMMFR_A8W8_IMPL(int64_t, float, true, true);
    } else if (TILING_KEY_IS(10000000000000011011UL)) {
        GMMFR_A8W8_IMPL(int32_t, float, true, true);
    } else if (TILING_KEY_IS(10000000000000011101UL)) {
        GMMFR_A8W8_IMPL(int64_t, bfloat16_t, true, true);
    } else if (TILING_KEY_IS(10000000000000011111UL)) {
        GMMFR_A8W8_IMPL(int32_t, bfloat16_t, true, true);
    } else if (TILING_KEY_IS(11000000000000011011UL)) {
        GMMFR_A8W4_IMPL(true, true);
    } else if (TILING_KEY_IS(11000000000000011111UL)) {
        GMMFR_A8W4_L1_OPT_IMPL(true, true);
    }
}
#endif