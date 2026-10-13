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
 * \file matmul_reduce_scatter_full_mesh.h
 * \brief
 */
#ifndef MATMUL_REDUCE_SCATTER_FULL_MESH_H
#define MATMUL_REDUCE_SCATTER_FULL_MESH_H

#include "matmul_reduce_scatter_base.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"

namespace AscendC {
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
class MatmulReduceScatterFullMesh
    : public MatmulReduceScatterBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float> {
public:
    __aicore__ inline MatmulReduceScatterFullMesh() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
        GM_ADDR contextGM, MatmulReduceScatterTilingData *tilingData, TPipe *tPipe, __gm__ void* mc2InitTiling, __gm__ void* mc2CcTiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void HcclPrepare();
    __aicore__ inline void InnerProcess();
    __aicore__ inline void MatmulKernelCompute(GM_ADDR aGM, GM_ADDR gmToFloat, TCubeTiling &tiling,
        Mc2Tiling::TileL2Tiling &l2Tiling, HcclHandle &handleId, uint32_t tileCnt);
    __aicore__ inline void MatmulKernelComputeL2Cache(GM_ADDR aGM, GM_ADDR gmToFloat, TCubeTiling &tiling,
        Mc2Tiling::TileL2Tiling &l2Tiling, HcclHandle &handleId, uint32_t tileCnt);
    __aicore__ inline void MatmulKernelReduceScatter(GM_ADDR aGM, GM_ADDR gmToFloat, TCubeTiling &tiling,
        Mc2Tiling::TileL2Tiling &l2Tiling, HcclHandle &handleId, uint32_t tileCnt);
    __aicore__ inline void HcclFinalize();

private:
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    HcclHandle handleId_{ INVALID_HANDLE_ID };
    HcclHandle tailHandleId_{ INVALID_HANDLE_ID };
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void MatmulReduceScatterFullMesh<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR contextGM,
    MatmulReduceScatterTilingData *tilingData, TPipe *tPipe, __gm__ void* mc2InitTiling, __gm__ void* mc2CcTiling)
{
    this->InitBase(aGM, bGM, biasGM, cGM, workspaceGM, contextGM, tilingData, tPipe);
    hccl_.Init(contextGM, mc2InitTiling);
    hccl_.SetCcTiling(mc2CcTiling);
    this->rankId_ = hccl_.GetRankId();
    this->rankDim_ = hccl_.GetRankDim();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void MatmulReduceScatterFullMesh<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::Process()
{
    this->Nd2NzBiasCast();
    HcclPrepare();
    InnerProcess();
    HcclFinalize();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void MatmulReduceScatterFullMesh<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::HcclPrepare()
{
    if ASCEND_IS_AIC {
        if (this->debugOnlyCalc_) {
            return;
        }
        auto &&cfg = this->tilingData_->param;
        auto &&tileTiling = this->tilingData_->tileTiling;
        auto &&tailTiling = this->tilingData_->tailTiling;

        // 下发任务消息
        const uint64_t tileDataCnt = (uint64_t)tileTiling.M * (uint64_t)tileTiling.N;
        const uint64_t tileDataOff = tileDataCnt * (uint64_t)cfg.tileCnt * sizeof(typename C_TYPE::T);
        const uint64_t rankDataCnt = (uint64_t)cfg.rankM * (uint64_t)cfg.rankN / (uint64_t)this->rankDim_;

        handleId_ = hccl_.ReduceScatter<false>(this->gmToFloat_, this->cGM_, tileDataCnt, HcclDataType(cfg.dataType),
            HcclReduceOp(cfg.subtype), rankDataCnt, cfg.tileCnt);
        if (cfg.tailCnt > 0) {
            const uint64_t tailDataCnt = (uint64_t)tailTiling.M * (uint64_t)tailTiling.N;
            tailHandleId_ = hccl_.ReduceScatter<false>(this->gmToFloat_ + tileDataOff, this->cGM_ + tileDataOff,
                tailDataCnt, HcclDataType(cfg.dataType), HcclReduceOp(cfg.subtype), rankDataCnt, cfg.tailCnt);
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void
MatmulReduceScatterFullMesh<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::InnerProcess()
{
    if ASCEND_IS_AIC {
        auto &&cfg = this->tilingData_->param;
        // mm计算
        MatmulKernelReduceScatter(this->aGM_, this->gmToFloat_, this->tilingData_->tileTiling,
        this->tilingData_->tileL2Tiling, handleId_, cfg.tileCnt);
        if (cfg.tailCnt > 0) { // 存在尾块
        auto &&tiling = this->tilingData_->tileTiling;
        auto aGMTail = this->aGM_ +
            (uint64_t)tiling.M * (uint64_t)tiling.Ka * (uint64_t)cfg.tileCnt * sizeof(typename A_TYPE::T);
        auto gmToFloatTail = this->gmToFloat_ +
            (uint64_t)tiling.M * (uint64_t)tiling.N * (uint64_t)cfg.tileCnt * sizeof(typename C_TYPE::T);
        // 尾块
        MatmulKernelReduceScatter(aGMTail, gmToFloatTail, this->tilingData_->tailTiling,
            this->tilingData_->tailL2Tiling, tailHandleId_, cfg.tailCnt);           
    }

        if (!this->debugOnlyCalc_) {
            // 等待所有通信任务结束
            for (uint32_t i = 0; i < cfg.tileCnt; i++) {
                hccl_.Wait(handleId_);
            }
            for (uint32_t i = 0; i < cfg.tailCnt; i++) {
                hccl_.Wait(tailHandleId_);
            }
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void
MatmulReduceScatterFullMesh<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::MatmulKernelCompute(GM_ADDR aGM,
    GM_ADDR gmToFloat, TCubeTiling &tiling, Mc2Tiling::TileL2Tiling &l2Tiling, HcclHandle &handleId, uint32_t tileCnt)
{
    if (GetBlockIdx() >= tiling.usedCoreNum) {
        for (uint32_t i = 0; i < tileCnt; i++) {
            CrossCoreSetFlag<0, PIPE_FIX>(EVENT_ID_6);
            CrossCoreWaitFlag(EVENT_ID_6);
        }
        return;
    }
    using A_T = typename A_TYPE::T;
    using C_T = typename C_TYPE::T;
    auto &&cfg = this->tilingData_->param;
    const uint64_t aOffset = (uint64_t)tiling.M * (uint64_t)tiling.Ka * sizeof(A_T);
    const uint64_t cOffset = (uint64_t)tiling.M * (uint64_t)tiling.N * sizeof(C_T);
    const uint64_t raOffset = (uint64_t)cfg.rankM * (uint64_t)cfg.rankK / (uint64_t)this->rankDim_ * sizeof(A_T);
    const uint64_t rcOffset = (uint64_t)cfg.rankM * (uint64_t)cfg.rankN / (uint64_t)this->rankDim_ * sizeof(C_T);

    // 归一化Matmul计算类，负责MC2的Matmul计算
    MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, SplitType::DEFAULT> mm;
    mm.InitGlobalTensor(this->bGM_, this->biasGM_);
    mm.Init(cfg, tiling, l2Tiling, this->rankId_);

    auto aWork = aGM;
    auto cWork = gmToFloat;
    for (uint32_t i = 0; i < tileCnt; i++) {
        for (uint32_t j = 0; j < this->rankDim_; j++) {
            uint32_t rank = (j + GetBlockIdx()) % this->rankDim_;
            // 计算A和C矩阵的首地址, 并初始化Buffer
            auto aWorkAddr = aWork + (uint64_t)rank * raOffset;
            auto cWorkAddr = cWork + (uint64_t)rank * rcOffset;
            mm.InitGlobalTensor(aWorkAddr, aOffset, cWorkAddr, cOffset);
            // Matmul计算主流程
            uint32_t index = rank + i * this->rankDim_;
            mm.Compute(index);
        }
        // 纯Cube核同步后再进行commit
        CrossCoreSetFlag<0, PIPE_FIX>(EVENT_ID_6);
        CrossCoreWaitFlag(EVENT_ID_6);
        if (!this->debugOnlyCalc_) {
            hccl_.Commit(handleId);
        }
        aWork += aOffset;
        cWork += cOffset;
    }
    mm.End();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void
MatmulReduceScatterFullMesh<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::MatmulKernelComputeL2Cache(
    GM_ADDR aGM, GM_ADDR gmToFloat, TCubeTiling &tiling, Mc2Tiling::TileL2Tiling &l2Tiling, HcclHandle &handleId, uint32_t tileCnt)
{
    if (GetBlockIdx() >= tiling.usedCoreNum) {
        for (uint32_t i = 0; i < tileCnt; i++) {
            CrossCoreSetFlag<0, PIPE_FIX>(EVENT_ID_6);
            CrossCoreWaitFlag(EVENT_ID_6);
        }
        return;
    }
    using C_T = typename C_TYPE::T;
    using A_T = typename A_TYPE::T;
    auto &&cfg = this->tilingData_->param;
    const uint64_t cOffset = (uint64_t)tiling.M * (uint64_t)tiling.N * sizeof(C_T);
    const uint64_t aOffset = (uint64_t)tiling.M * (uint64_t)tiling.Ka * sizeof(A_T);
    const uint64_t raOffset = (uint64_t)cfg.rankM * (uint64_t)cfg.rankK / (uint64_t)this->rankDim_ * sizeof(A_T);
    const uint64_t rcOffset = (uint64_t)cfg.rankM * (uint64_t)cfg.rankN / (uint64_t)this->rankDim_ * sizeof(C_T);

    // 归一化Matmul计算类，负责MC2的Matmul计算
    MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, SplitType::L2CACHE> mm;
    mm.InitGlobalTensor(this->bGM_, this->biasGM_);
    mm.Init(cfg, tiling, l2Tiling, this->rankId_);

    auto aWork = aGM;
    auto cWork = gmToFloat;
    for (uint32_t i = 0; i < tileCnt; i++) {
        // Matmul计算主流程
        mm.InitGlobalTensor(aWork, raOffset * (uint64_t)this->rankDim_, cWork, rcOffset * (uint64_t)this->rankDim_);
        mm.ComputeWithL2Cache(i);

        // 纯Cube核同步后再进行commit
        CrossCoreSetFlag<0, PIPE_FIX>(EVENT_ID_6);
        CrossCoreWaitFlag(EVENT_ID_6);
        if (!this->debugOnlyCalc_) {
            hccl_.Commit(handleId);
        }

        cWork += cOffset;
        aWork += aOffset;
    }
    mm.End();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void
MatmulReduceScatterFullMesh<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::MatmulKernelReduceScatter(
    GM_ADDR aGM, GM_ADDR gmToFloat, TCubeTiling &tiling, Mc2Tiling::TileL2Tiling &l2Tiling, HcclHandle &handleId, uint32_t tileCnt)
{
    // Matmul的一次计算流程
    if (l2Tiling.enableL2Tile > 0) {
        MatmulKernelComputeL2Cache(aGM, gmToFloat, tiling, l2Tiling, handleId, tileCnt);
    } else {
        MatmulKernelCompute(aGM, gmToFloat, tiling, l2Tiling, handleId, tileCnt);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
__aicore__ inline void
MatmulReduceScatterFullMesh<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>::HcclFinalize()
{
     if ASCEND_IS_AIC {
        if (!this->debugOnlyCalc_) {
            // 保证所有核计算结束再Finalize
            CrossCoreSetFlag<0, PIPE_FIX>(EVENT_ID_6);
            CrossCoreWaitFlag(EVENT_ID_6);
            hccl_.Finalize();
        }
    }
}
}
#endif