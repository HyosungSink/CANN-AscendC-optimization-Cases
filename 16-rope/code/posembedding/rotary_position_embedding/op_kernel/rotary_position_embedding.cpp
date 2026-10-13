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
 * \file rotary_position_embedding.cpp
 * \brief
 */
#include "rotate_half.h"
#include "rotate_half_bf16.h"

#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 200)
#include "rotate_interleaved_split_s.h"
#include "rotate_interleaved_split_bs.h"
#include "rotate_interleaved_split_bsn.h"
#include "rotate_interleaved_split_s_pad.h"
#include "rotate_interleaved_split_bs_pad.h"
#include "rotate_interleaved_split_bsn_pad.h"
using namespace RotateInterleavedN;
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#include "rotate_matrix.h"
using namespace RotateMatrix;
using namespace matmul;
#endif

using namespace AscendC;
using namespace RotateHalfN;

extern "C" __global__ __aicore__ void rotary_position_embedding(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR rotate,
                                                                GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    // mode: rotate_half
    if (TILING_KEY_IS(1011)) {
        RotateHalf<float> rotateHalfOp;
        rotateHalfOp.Init(x, cos, sin, y, tilingData);
        rotateHalfOp.Process();
    } else if (TILING_KEY_IS(1021)) {
        RotateHalf<float> rotateHalfOp;
        rotateHalfOp.Init(x, cos, sin, y, tilingData);
        rotateHalfOp.Process();
    } else if (TILING_KEY_IS(1031)) {
        RotateHalf<float> rotateHalfOp;
        rotateHalfOp.Init(x, cos, sin, y, tilingData);
        rotateHalfOp.Process();
    } else if (TILING_KEY_IS(1041)) {
        RotateHalf<float> rotateHalfOp;
        rotateHalfOp.Init(x, cos, sin, y, tilingData);
        rotateHalfOp.Process();
    } else if (TILING_KEY_IS(1051)) {
        RotateHalf<float> rotateHalfOp;
        rotateHalfOp.Init(x, cos, sin, y, tilingData);
        rotateHalfOp.Process();
    } else if (TILING_KEY_IS(1061)) {
        RotateHalf<float> rotateHalfOp;
        rotateHalfOp.Init(x, cos, sin, y, tilingData);
        rotateHalfOp.Process();
    } else if (TILING_KEY_IS(1012)) {
        RotateHalfBf16<half, float> rotateHalfOp;
        rotateHalfOp.Init(x, cos, sin, y, tilingData);
        rotateHalfOp.Process();
    } else if (TILING_KEY_IS(1022)) {
        RotateHalfBf16<half, float> rotateHalfOp;
        rotateHalfOp.Init(x, cos, sin, y, tilingData);
        rotateHalfOp.Process();
    } else if (TILING_KEY_IS(1032)) {
        RotateHalfBf16<half, float> rotateHalfOp;
        rotateHalfOp.Init(x, cos, sin, y, tilingData);
        rotateHalfOp.Process();
    } else if (TILING_KEY_IS(1042)) {
        RotateHalfBf16<half, float> rotateHalfOp;
        rotateHalfOp.Init(x, cos, sin, y, tilingData);
        rotateHalfOp.Process();
    } else if (TILING_KEY_IS(1052)) {
        RotateHalfBf16<half, float> rotateHalfOp;
        rotateHalfOp.Init(x, cos, sin, y, tilingData);
        rotateHalfOp.Process();
    } else if (TILING_KEY_IS(1062)) {
        RotateHalfBf16<half, float> rotateHalfOp;
        rotateHalfOp.Init(x, cos, sin, y, tilingData);
        rotateHalfOp.Process();
#if !(((defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)) || (defined(__CCE_AICORE__) && __CCE_AICORE__ == 200)))
    } else if (TILING_KEY_IS(1013)) {
        RotateHalfBf16<bfloat16_t, float> rotateHalfOp;
        rotateHalfOp.Init(x, cos, sin, y, tilingData);
        rotateHalfOp.Process();
    } else if (TILING_KEY_IS(1023)) {
        RotateHalfBf16<bfloat16_t, float> rotateHalfOp;
        rotateHalfOp.Init(x, cos, sin, y, tilingData);
        rotateHalfOp.Process();
    } else if (TILING_KEY_IS(1033)) {
        RotateHalfBf16<bfloat16_t, float> rotateHalfOp;
        rotateHalfOp.Init(x, cos, sin, y, tilingData);
        rotateHalfOp.Process();
    } else if (TILING_KEY_IS(1043)) {
        RotateHalfBf16<bfloat16_t, float> rotateHalfOp;
        rotateHalfOp.Init(x, cos, sin, y, tilingData);
        rotateHalfOp.Process();
    } else if (TILING_KEY_IS(1053)) {
        RotateHalfBf16<bfloat16_t, float> rotateHalfOp;
        rotateHalfOp.Init(x, cos, sin, y, tilingData);
        rotateHalfOp.Process();
    } else if (TILING_KEY_IS(1063)) {
        RotateHalfBf16<bfloat16_t, float> rotateHalfOp;
        rotateHalfOp.Init(x, cos, sin, y, tilingData);
        rotateHalfOp.Process();
#endif
    }

    // mode: rotate_interleaved
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 200)
    if (TILING_KEY_IS(2000)) {
        TPipe pipe;
        InterleavedSplitS<half> interleavedSplitS;
        interleavedSplitS.Init(x, cos, sin, y, tilingData, &pipe);
        interleavedSplitS.Process();
    } else if (TILING_KEY_IS(2010)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        TPipe pipe;
        InterleavedSplitS<bfloat16_t> interleavedSplitS;
        interleavedSplitS.Init(x, cos, sin, y, tilingData, &pipe);
        interleavedSplitS.Process();
#endif
    } else if (TILING_KEY_IS(2020)) {
        TPipe pipe;
        InterleavedSplitS<float> interleavedSplitS;
        interleavedSplitS.Init(x, cos, sin, y, tilingData, &pipe);
        interleavedSplitS.Process();
    } else if (TILING_KEY_IS(2100)) {
        TPipe pipe;
        InterleavedSplitBS<half> interleavedSplitBS;
        interleavedSplitBS.Init(x, cos, sin, y, tilingData, &pipe);
        interleavedSplitBS.Process();
    } else if (TILING_KEY_IS(2110)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        TPipe pipe;
        InterleavedSplitBS<bfloat16_t> interleavedSplitBS;
        interleavedSplitBS.Init(x, cos, sin, y, tilingData, &pipe);
        interleavedSplitBS.Process();
#endif
    } else if (TILING_KEY_IS(2120)) {
        TPipe pipe;
        InterleavedSplitBS<float> interleavedSplitBS;
        interleavedSplitBS.Init(x, cos, sin, y, tilingData, &pipe);
        interleavedSplitBS.Process();
    } else if (TILING_KEY_IS(2200)) {
        TPipe pipe;
        InterleavedSplitBSN<half> interleavedSplitBSN;
        interleavedSplitBSN.Init(x, cos, sin, y, tilingData, &pipe);
        interleavedSplitBSN.Process();
    } else if (TILING_KEY_IS(2210)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        TPipe pipe;
        InterleavedSplitBSN<bfloat16_t> interleavedSplitBSN;
        interleavedSplitBSN.Init(x, cos, sin, y, tilingData, &pipe);
        interleavedSplitBSN.Process();
#endif
    } else if (TILING_KEY_IS(2220)) {
        TPipe pipe;
        InterleavedSplitBSN<float> interleavedSplitBSN;
        interleavedSplitBSN.Init(x, cos, sin, y, tilingData, &pipe);
        interleavedSplitBSN.Process();
    } else if (TILING_KEY_IS(2001)) {
        TPipe pipe;
        InterleavedSplitSPad<half> interleavedSplitSPad;
        interleavedSplitSPad.Init(x, cos, sin, y, tilingData, &pipe);
        interleavedSplitSPad.Process();
    } else if (TILING_KEY_IS(2011)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        TPipe pipe;
        InterleavedSplitSPad<bfloat16_t> interleavedSplitSPad;
        interleavedSplitSPad.Init(x, cos, sin, y, tilingData, &pipe);
        interleavedSplitSPad.Process();
#endif
    } else if (TILING_KEY_IS(2021)) {
        TPipe pipe;
        InterleavedSplitSPad<float> interleavedSplitSPad;
        interleavedSplitSPad.Init(x, cos, sin, y, tilingData, &pipe);
        interleavedSplitSPad.Process();
    } else if (TILING_KEY_IS(2101)) {
        TPipe pipe;
        InterleavedSplitBSPad<half> interleavedSplitBSPad;
        interleavedSplitBSPad.Init(x, cos, sin, y, tilingData, &pipe);
        interleavedSplitBSPad.Process();
    } else if (TILING_KEY_IS(2111)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        TPipe pipe;
        InterleavedSplitBSPad<bfloat16_t> interleavedSplitBSPad;
        interleavedSplitBSPad.Init(x, cos, sin, y, tilingData, &pipe);
        interleavedSplitBSPad.Process();
#endif
    } else if (TILING_KEY_IS(2121)) {
        TPipe pipe;
        InterleavedSplitBSPad<float> interleavedSplitBSPad;
        interleavedSplitBSPad.Init(x, cos, sin, y, tilingData, &pipe);
        interleavedSplitBSPad.Process();
    } else if (TILING_KEY_IS(2201)) {
        TPipe pipe;
        InterleavedSplitBSNPad<half> interleavedSplitBSNPad;
        interleavedSplitBSNPad.Init(x, cos, sin, y, tilingData, &pipe);
        interleavedSplitBSNPad.Process();
    } else if (TILING_KEY_IS(2211)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        TPipe pipe;
        InterleavedSplitBSNPad<bfloat16_t> interleavedSplitBSNPad;
        interleavedSplitBSNPad.Init(x, cos, sin, y, tilingData, &pipe);
        interleavedSplitBSNPad.Process();
#endif
    } else if (TILING_KEY_IS(2221)) {
        TPipe pipe;
        InterleavedSplitBSNPad<float> interleavedSplitBSNPad;
        interleavedSplitBSNPad.Init(x, cos, sin, y, tilingData, &pipe);
        interleavedSplitBSNPad.Process();
    }
#endif
    
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (TILING_KEY_IS(3011)) {
        using aT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
        using bT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
        using MT = matmul::MatmulImpl<aT, bT, cT>;
        MT mm;

        TPipe pipe;
        KERNEL_TASK_TYPE(3011, KERNEL_TYPE_MIX_AIC_1_2);
        RotateMatrixAll<float, float, MT> op(mm);
        op.Init(x, cos, sin, rotate, y, usrWorkspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3012)) {
        using aT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
        using bT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
        using MT = matmul::MatmulImpl<aT, bT, cT>;
        MT mm;

        TPipe pipe;
        KERNEL_TASK_TYPE(3012, KERNEL_TYPE_MIX_AIC_1_2);
        RotateMatrixAll<half, half, MT> op(mm);
        op.Init(x, cos, sin, rotate, y, usrWorkspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3013)) {
        using aT = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
        using bT = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
        using MT = matmul::MatmulImpl<aT, bT, cT>;
        MT mm;

        TPipe pipe;
        KERNEL_TASK_TYPE(3013, KERNEL_TYPE_MIX_AIC_1_2);
        RotateMatrixAll<bfloat16_t, bfloat16_t, MT> op(mm);
        op.Init(x, cos, sin, rotate, y, usrWorkspace, tilingData, &pipe);
        op.Process();
    }
#endif
}
