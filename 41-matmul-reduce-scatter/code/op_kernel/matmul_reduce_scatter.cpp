/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_reduce_scatter.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

#include "matmul_reduce_scatter_tiling.h"
#include "matmul_reduce_scatter_tiling_key.h"
#if __CCE_AICORE__ == 310
#else
    #include "matmul_reduce_scatter_full_mesh.h"
#endif

#define INVOKE_MATMUL_REDUCE_SCATTER_OP_IMPL(templateClass, ...)                         \
    do {                                                                                 \
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_DTYPE, true>; \
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;       \
        templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                    \
        op.Init(aGM, bGM, biasGM, cGM, workspaceGM, contextGM, &tilingData, &pipe, mc2InitTiling, mc2CcTiling);  \
        op.Process();                                                                    \
    } while (0)
using namespace AscendC;

namespace AscendC {
template <class T> struct BiasType {
    using type = float;
};
template <> struct BiasType<half> {
    using type = half;
};

template<bool IsFullMesh, bool IsNd2Nz, bool IsBias>
__global__ __aicore__ void matmul_reduce_scatter(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MatmulReduceScatterTilingData);
    // matmulreducescatter算子kernel侧代码不支持A5
    #if __CCE_AICORE__ != 310
        auto tiling = (__gm__ MatmulReduceScatterTilingData*)tilingGM;
        __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));
        __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));
        GET_TILING_DATA(tilingData, tilingGM);

        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        TPipe pipe;
        GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();

        if constexpr (IsFullMesh && IsNd2Nz && !IsBias) {
            // full mesh + nd2nz + no bias cast
            using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, B_DTYPE, true>;
            using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, typename BiasType<BIAS_DTYPE>::type>;
            INVOKE_MATMUL_REDUCE_SCATTER_OP_IMPL(MatmulReduceScatterFullMesh, IsNd2Nz, IsBias);
        } else if constexpr (IsFullMesh && !IsNd2Nz && !IsBias) {
            // full mesh + no nd2nz + no bias cast
            using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_DTYPE, true>;
            using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, typename BiasType<BIAS_DTYPE>::type>;
            INVOKE_MATMUL_REDUCE_SCATTER_OP_IMPL(MatmulReduceScatterFullMesh, IsNd2Nz, IsBias);
        } else if constexpr (IsFullMesh && IsNd2Nz && IsBias) {
            // full mesh + nd2nz + bias cast
            using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, B_DTYPE, true>;
            using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
            INVOKE_MATMUL_REDUCE_SCATTER_OP_IMPL(MatmulReduceScatterFullMesh, IsNd2Nz, IsBias);
        } else if constexpr (IsFullMesh && !IsNd2Nz && IsBias) {
            // full mesh + no nd2nz + bias cast
            using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_DTYPE, true>;
            using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
            INVOKE_MATMUL_REDUCE_SCATTER_OP_IMPL(MatmulReduceScatterFullMesh, IsNd2Nz, IsBias);
        }
    #endif
}
}