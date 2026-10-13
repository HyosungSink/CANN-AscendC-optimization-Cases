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
 * \file matmul_reduce_scatter_base.h
 * \brief
 */
#ifndef MATMUL_REDUCE_SCATTER_BASE_H
#define MATMUL_REDUCE_SCATTER_BASE_H

#include "lib/matmul_intf.h"
#include "reduce_scatter_nd_to_nz.h"

#if __has_include("../common/inc/kernel/mc2_nd_to_nz.h")
#include "../common/inc/kernel/mc2_nd_to_nz.h"
#include "../common/inc/kernel/mc2_matmul_compute.h"
#else
#include "../common/inc/kernel/mc2_nd_to_nz.h"
#include "../common/inc/kernel/mc2_matmul_compute.h"
#endif

namespace AscendC {
constexpr uint8_t MC2_DEBUG_ONLY_CUBE = 1;  // 只计算不通信
constexpr uint8_t MC2_DEBUG_ONLY_AICPU = 4; // 只通信不计算

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
class MatmulReduceScatterBase {
public:
    __aicore__ inline MatmulReduceScatterBase() {}
    __aicore__ inline void InitBase(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
        GM_ADDR contextGM, MatmulReduceScatterTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Nd2NzBiasCast();

protected:
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR biasGM_;
    GM_ADDR cGM_;
    GM_ADDR gmToFloat_;
    GM_ADDR workspaceGM_;
    MatmulReduceScatterTilingData *tilingData_;
    TPipe *tPipe_;
    uint32_t rankId_{ 0 };
    uint32_t rankDim_{ 8 };
    bool debugOnlyCalc_{ false };
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void MatmulReduceScatterBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::InitBase(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR contextGM,
    MatmulReduceScatterTilingData *tilingData, TPipe *tPipe)
{
    aGM_ = aGM;
    bGM_ = bGM;
    biasGM_ = biasGM;
    cGM_ = cGM;
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;
    tPipe_ = tPipe;

    gmToFloat_ = workspaceGM_; // step1, C矩阵float
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void MatmulReduceScatterBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::Nd2NzBiasCast()
{
    TBuf<TPosition::VECCALC> totalUbBuf;
    tPipe_->InitBuffer(totalUbBuf, TOTAL_UB_SIZE);

    // workspace 空间划分:
    GM_ADDR gmNd2NzAddr = workspaceGM_ + (uint64_t)tilingData_->param.cToFloatLen; // step2, ND2NZ 后的地址
    GM_ADDR biasToFloat = gmNd2NzAddr + (uint64_t)tilingData_->param.nd2NzWorkLen; // step3, bias 后的地址
    if constexpr (Bias2Float) {
        CastBFtoFloat(biasToFloat, biasGM_, tilingData_->param.rankN, totalUbBuf);
        biasGM_ = biasToFloat;
        SyncAll<true>(); 
    }
    if constexpr (BNd2Nz) {
        MatrixBtoNZMc2<typename B_TYPE::T>(gmNd2NzAddr, bGM_, tilingData_, tilingData_->param.isTransposeB, totalUbBuf);
        bGM_ = gmNd2NzAddr;
    }
}
}
#endif