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
 * \file grouped_matmul_finalize_routing_antiquant_a8w4_msd_l1_opt.h
 * \brief
 */

#ifndef ASCENDC_GROUPED_MATMUL_FINALIZE_ROUTING_ANTIQUANT_A8W4_MSD_L1_OPT_H
#define ASCENDC_GROUPED_MATMUL_FINALIZE_ROUTING_ANTIQUANT_A8W4_MSD_L1_OPT_H

#include <cstdint>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "grouped_matmul_finalize_routing_utils.h"

namespace GroupedMatmulFinalizeRouting {
using namespace matmul;
using namespace AscendC;

constexpr uint32_t BUFFER_NUM_L1OPT = 1;

template <class mmType>
class GMMA8W4MSDL1OptCompute {
public:
    using cT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    using DTYPE_OUT = float;

public:
    __aicore__ inline GMMA8W4MSDL1OptCompute(typename mmType::MT &matmul) : mm(matmul) {}
    __aicore__ inline void Init(const MMInitParams& initParams, const GroupMatmulFRTilingData *tilingData, TPipe *tPipeIn);
    __aicore__ inline void Process();
private:
    __aicore__ inline void InitMMBuffer();
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void PreProcess();
    __aicore__ inline void PreProcessInit();
    __aicore__ inline void InitOutputWithZeros(uint64_t offset, uint64_t size);
    __aicore__ inline void CopyInA1(uint64_t xOffset, uint32_t row, uint32_t col, LocalTensor<int4b_t> a1Local);
    __aicore__ inline void MMComputeSingleN(uint32_t groupIdx, MNConfig& mnConfig,
                                            uint32_t singleNOffset, uint32_t curSingleN, uint32_t curSingleM);
    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig& mnConfig);
    __aicore__ inline void VectorCompute(uint32_t groupIdx, MNConfig& mnConfig);
    __aicore__ inline void ComputeDequantAndActivate(MNConfig& mnConfig, uint32_t curVecBaseM, uint32_t alignBaseN,
                                                     uint32_t curVecBaseN, uint32_t offsetM);
    __aicore__ inline void CastMulsAdds(uint32_t computeSize, uint32_t addStartAddr);
    __aicore__ inline void AfterMuls(uint32_t addStartAddr);
    __aicore__ inline void DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset);
    __aicore__ inline void DataCopyOffset(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset);
    __aicore__ inline void DataCopyPerTokenScaleAndBrcb(MNConfig& mnConfig, uint32_t curBaseM, uint32_t alignBaseN,
                                                        uint32_t offsetM);
    __aicore__ inline void VectorAtomicProcess(uint32_t curVecBaseM, uint32_t curVecBaseN, uint32_t alignBaseN,
                                               uint32_t rowIndexOffset, uint64_t yGmOffset);
    __aicore__ inline void  VectorProcess(VectorAtomicParams& vecAParams, MNConfig& mnConfig, uint64_t mmOutOffset,
                                          LocalTensor<cT::T>& mmOutLocal);
    __aicore__ inline void CopyMMOutLocal(LocalTensor<cT::T> &mmOutTensor, uint32_t dstOffset, uint64_t srcAddr,
                            uint32_t curVecBaseM, uint32_t curVecBaseN);
    __aicore__ inline void DataCopyAndBrcbOfRowSum(MNConfig& mnConfig, uint32_t curBaseM, uint32_t alignBaseN,
                                                        uint32_t offsetM);
private:
    typename mmType::MT& mm;
    const uint32_t HALF_ALIGN = 16;
    GlobalTensor<int4b_t> xGm;
    GlobalTensor<int4b_t> weightGm;
    GlobalTensor<float> biasGm;  // for 8 * weight
    GlobalTensor<cT::T> mmOutGm;
    GlobalTensor<uint64_t> scaleGm;
    GlobalTensor<float> perTokenScaleGm;
    GlobalTensor<float> offsetGm;
    GlobalTensor<int64_t> groupTokensGm;
    GlobalTensor<float> logitsGm;
    GlobalTensor<bfloat16_t> residualGm;
    GlobalTensor<int64_t> tokenRanksGm;
    GlobalTensor<DTYPE_OUT> yGm;
    GlobalTensor<float> xRowSumGm;
    // define the que
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;
    TQue<QuePosition::VECIN, 1> scaleInQueue;
    TQue<QuePosition::VECIN, 1> offsetInQueue;
    TQue<QuePosition::VECIN, 1> perTokenScaleInQueue;
    TQue<QuePosition::VECIN, 1> xRowSumInQueue;
    TSCM<TPosition::GM, 1> xL1Queue;
    TBuf<TPosition::VECCALC> tmpBuff;
    LocalTensor<float> offsetInUb;
    LocalTensor<float> scaleInUb;
    LocalTensor<float> afterProcessTempBuffer;
    LocalTensor<float> floatCalBuffer;
    LocalTensor<float> afterProcessOutBuffer;
    LocalTensor<float> scaleCalBuffer;
    LocalTensor<float> offsetCalBuffer;
    LocalTensor<uint8_t> tempBuffer;
    LocalTensor<uint8_t> tempRowSumBuffer;
    uint32_t subBlockIdx;
    uint32_t coreIdx;
    uint32_t quantGroupSize;
    uint32_t baseCubeCount = 0;
    uint32_t vecCount = 0;
    uint32_t xRowSumCount = 0;
    uint32_t withOffset;
    TPipe *pipe;
    const GroupMatmulFRTilingData *tiling;
};

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::Init(const MMInitParams& initParams,
    const GroupMatmulFRTilingData *tilingData, TPipe *tPipeIn)
{
    tiling = tilingData;
    xRowSumCount = tiling->totalInGroup;
    withOffset = tiling->withOffset;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t *>(initParams.x));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t *>(initParams.weight));
    biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.bias));
    scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(initParams.scale));
    perTokenScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.pertoken_scale));
    groupTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(initParams.group_tokens));
    logitsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.logits));
    tokenRanksGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(initParams.token_ranks));
    residualGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(initParams.residual));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_OUT *>(initParams.y));
    if (withOffset == uint32_t(1)) {
        xRowSumGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.workspace));
        mmOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ cT::T *>(initParams.workspace +
                                xRowSumCount * sizeof(float))); // add xRowSumGm
        offsetGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.offset));
    } else {
        mmOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ cT::T *>(initParams.workspace));
    }
    quantGroupSize = tiling->k / tiling->quantGroupNum;  // 约束为整除关系
    subBlockIdx = GetSubBlockIdx();
    coreIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        if (GetTaskRation() != 0) {
            coreIdx /= GetTaskRation();
        }
    }
    pipe = tPipeIn;
    InitMMBuffer();
    InitUbBuffer();
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::InitMMBuffer()
{
    if ASCEND_IS_AIV {
        return;
    }
    pipe->InitBuffer(xL1Queue, 1, (tiling->matmulTiling.baseM * tiling->k) / 2); // divide size by 2 to get sizeof(int4b_t)
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::InitUbBuffer()
{
    if ASCEND_IS_AIC {
        return;
    }
    pipe->InitBuffer(scaleInQueue, BUFFER_NUM_L1OPT, tiling->matmulTiling.baseN * sizeof(float)); // bias queue
    // 2: pertoken scale和logits般到一块buffer上
    pipe->InitBuffer(perTokenScaleInQueue, BUFFER_NUM_L1OPT, Ceil(tiling->vBaseM * sizeof(float) * 2,
        UB_ALIGN_LEN) * UB_ALIGN_LEN);
    pipe->InitBuffer(vecInQueue, BUFFER_NUM_L1OPT, tiling->ubCalSize * 2 * sizeof(cT::T));
    pipe->InitBuffer(vecOutQueue, BUFFER_NUM_L1OPT, tiling->ubCalSize * sizeof(DTYPE_OUT));
    pipe->InitBuffer(tmpBuff, tiling->ubRestBytes);
    uint32_t ubCalSizeFloat = tiling->ubCalSize * sizeof(float);
    // ub分配，依次划分中间结果，划分方式参考设计文档
    afterProcessTempBuffer = tmpBuff.GetWithOffset<float>(tiling->ubCalSize * 2, 0);
    uint32_t offset = ubCalSizeFloat * uint32_t(2);
    floatCalBuffer = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, offset);
    if (withOffset == uint32_t(1)) {
        pipe->InitBuffer(offsetInQueue, BUFFER_NUM_L1OPT, tiling->matmulTiling.baseN * sizeof(float));
        pipe->InitBuffer(xRowSumInQueue, BUFFER_NUM_L1OPT, Ceil(tiling->vBaseM * sizeof(float), 32) * 32);
        offsetCalBuffer = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, ubCalSizeFloat);
        tempBuffer = tmpBuff.GetWithOffset<uint8_t>(ubCalSizeFloat, 0);
        tempRowSumBuffer = tmpBuff.GetWithOffset<uint8_t>(ubCalSizeFloat, ubCalSizeFloat);
    } else {
        tempBuffer = tmpBuff.GetWithOffset<uint8_t>(2 * ubCalSizeFloat, 0);
    }
    offset += ubCalSizeFloat;
    afterProcessOutBuffer = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, offset);
    scaleCalBuffer = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, 0);
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::InitOutputWithZeros(uint64_t offset, uint64_t size) {
    uint64_t singelCount = Ceil(size, uint32_t(GetBlockNum() * GetTaskRation()));
    singelCount = Ceil(singelCount, 512) * 512;
    uint64_t baseOffsetL1Opt = GetBlockIdx() * singelCount;
    if (baseOffsetL1Opt >= size) {
        return;
    }
    if (baseOffsetL1Opt + singelCount > size) {
        singelCount = size - baseOffsetL1Opt;
    }
    baseOffsetL1Opt += offset;

    uint64_t times = (singelCount + UINT32_MAX - uint32_t(1)) / UINT32_MAX;
    for (uint32_t i = 0; i < times; i++) {
        InitOutput<DTYPE_OUT>(yGm[baseOffsetL1Opt + (i * UINT32_MAX)], singelCount - (i * UINT32_MAX), 0);
    }
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::PreProcessInit() {
    if (tiling->sharedInputOffset > 0) {
        InitOutputWithZeros(0, (static_cast<uint64_t>(tiling->n)) * tiling->sharedInputOffset);
    }
    uint64_t tail = tiling->sharedInputOffset + tiling->sharedInputLen;
    if (tail < tiling->batch) {
        InitOutputWithZeros(tail * (static_cast<uint64_t>(tiling->n)), tiling->n * (tiling->batch - tail));
    }
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::PreProcess() {
    if constexpr (mmType::sharedInputIsNone) {
        InitOutputWithZeros(0, tiling->n * tiling->batch);
        return;
    }
    PreProcessInit();
    uint64_t totalOutput = (static_cast<uint64_t>(tiling->n)) * tiling->sharedInputLen;
    uint64_t singeCount = Ceil(totalOutput, uint32_t(GetBlockNum() * GetTaskRation()));
    uint64_t baseOffset;
    uint64_t outOffset;
    uint64_t curCount = tiling->ubCalSize;

    singeCount = Ceil(singeCount, tiling->ubCalSize) * tiling->ubCalSize;
    baseOffset = GetBlockIdx() * singeCount;
    if (baseOffset >= totalOutput) {
        return;
    }

    if (baseOffset + singeCount > totalOutput) {
        singeCount = totalOutput - baseOffset;
    }
    outOffset = baseOffset + tiling->n * tiling->sharedInputOffset;

    DataCopyPadExtParams<bfloat16_t> padParams;
    for (uint32_t offset = 0; offset < singeCount; offset += curCount) {
        if (unlikely(offset + curCount > singeCount)) {
            curCount = singeCount - offset;
        }
        auto residualLocalTensor = vecInQueue.AllocTensor<bfloat16_t>();
        // 32B对齐搬运可以简化参数，这里按不对齐处理
        DataCopyExtParams params{1, static_cast<uint32_t>(curCount * sizeof(bfloat16_t)), 1, 1, 0};
        DataCopyPad(residualLocalTensor, residualGm[baseOffset + offset], params, padParams);
        vecInQueue.EnQue(residualLocalTensor);
        residualLocalTensor = vecInQueue.DeQue<bfloat16_t>();

        Cast(floatCalBuffer, residualLocalTensor, AscendC::RoundMode::CAST_NONE, curCount);
        vecInQueue.FreeTensor(residualLocalTensor);
        PipeBarrier<PIPE_V>();
        LocalTensor<DTYPE_OUT> yLocal = vecOutQueue.AllocTensor<DTYPE_OUT>();
        Muls(yLocal, floatCalBuffer, tiling->residualScale, curCount);
        vecOutQueue.EnQue(yLocal);

        DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curCount * sizeof(float)), 1, 1, 0};
        yLocal = vecOutQueue.DeQue<DTYPE_OUT>();
        DataCopyPad(yGm[outOffset + offset], yLocal, paramsOut);
        vecOutQueue.FreeTensor(yLocal);
    }
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::Process()
{
    if ASCEND_IS_AIV {
        PreProcess();
        SyncAll();
    }
    MNConfig mnConfig;
    mnConfig.baseM = tiling->matmulTiling.baseM;
    mnConfig.baseN = tiling->matmulTiling.baseN;
    mnConfig.singleM = mnConfig.baseM;
    mnConfig.singleN = tiling->matmulTiling.singleCoreN;
    mnConfig.blockDimN = Ceil(tiling->n, mnConfig.singleN);
    mnConfig.baseNEachSingleN = Ceil(mnConfig.singleN, mnConfig.baseN);
    if ASCEND_IS_AIC {
        SyncAll<false>();
    }
    for (uint32_t groupIdx = 0, preCount = 0; groupIdx < tiling->groupNum; ++groupIdx) {
        int32_t m = static_cast<int32_t>(groupTokensGm.GetValue(groupIdx));
        if (m <= 0) {
            continue;
        }
        if constexpr (mmType::groupListType) {
            m -= mnConfig.offsetM / 2;
        }
        mnConfig.m = static_cast<uint32_t>(m) * 2;      // 2: int8 has been split in 2 int4
        mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
        mm.SetOrgShape(mnConfig.m, tiling->n, tiling->k);
        uint32_t curCount = preCount + mnConfig.blockDimN * mnConfig.blockDimM;
        uint32_t curBlock = coreIdx >= preCount ? coreIdx : coreIdx + tiling->coreNum;

        while (curBlock < curCount) {
            mnConfig.mIdx = (curBlock - preCount) / mnConfig.blockDimN;
            mnConfig.nIdx = (curBlock - preCount) % mnConfig.blockDimN;

            if ASCEND_IS_AIC {
                MMCompute(groupIdx, mnConfig);
            }

            if ASCEND_IS_AIV {
                VectorCompute(groupIdx, mnConfig);
            }
            curBlock += tiling->coreNum;
        }
        preCount = curCount % tiling->coreNum;
        mnConfig.offsetM += mnConfig.m;
    }
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::CopyInA1(uint64_t xOffset, uint32_t row, uint32_t col, LocalTensor<int4b_t> a1Local)
{
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = row;
    nd2nzParams.dValue = col / 2;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.srcDValue = col / 2;
    nd2nzParams.dstNzC0Stride = Ceil(row, 16) * 16;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;

    DataCopy(a1Local, xGm[xOffset], nd2nzParams);
    xL1Queue.EnQue(a1Local);
    a1Local = xL1Queue.DeQue<int4b_t>();    
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::MMComputeSingleN(uint32_t groupIdx, MNConfig& mnConfig,
                                                                        uint32_t singleNOffset, uint32_t curSingleN, uint32_t curSingleM)
{
    uint64_t xOffset = (static_cast<uint64_t>(mnConfig.offsetM) + mnConfig.mIdx * mnConfig.singleM) * tiling->k;
    LocalTensor<int4b_t> xL1LocalInt4 = xL1Queue.AllocTensor<int4b_t>();
    CopyInA1(xOffset, curSingleM, tiling->k, xL1LocalInt4);

    uint32_t baseNCurSingleN = Ceil(curSingleN, mnConfig.baseN);

    for (uint32_t baseNIdx = 0; baseNIdx < baseNCurSingleN; baseNIdx++) {
        if (baseCubeCount % mnConfig.baseNEachSingleN == 0) { // 进入新一轮 更新workspace从起点开始
            mnConfig.workSpaceOffset = mnConfig.baseM * mnConfig.singleN * coreIdx;
        }
        if (baseCubeCount >= mnConfig.baseNEachSingleN) {
            CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
        }
        uint64_t weightOffset;
        if constexpr (mmType::BT::format == CubeFormat::NZ) {
            weightOffset = (static_cast<uint64_t>(groupIdx)) * tiling->n * tiling->k + singleNOffset * tiling->k + baseNIdx * mnConfig.baseN * tiling->k;
        } else {
            weightOffset = (static_cast<uint64_t>(groupIdx)) * tiling->n * tiling->k + singleNOffset + baseNIdx * mnConfig.baseN;
        }

        uint32_t curBaseN = mnConfig.baseN;
        if (unlikely(baseNIdx == baseNCurSingleN - 1)) {
            curBaseN = curSingleN - baseNIdx * mnConfig.baseN;
        }
        mm.SetSingleShape(curSingleM, curBaseN, quantGroupSize);
        GlobalTensor<int4b_t> weightSlice;
        for (uint32_t loopK = 0; loopK < tiling->quantGroupNum; loopK++) {
            mm.SetTensorA(xL1LocalInt4[loopK * quantGroupSize]); // Copy x from L1
            if constexpr (mmType::BT::format == CubeFormat::NZ) {
                weightSlice = weightGm[weightOffset + loopK * quantGroupSize * 64];
            } else {
                weightSlice = weightGm[weightOffset + loopK * quantGroupSize * tiling->n];
            }
            if (mnConfig.blockDimM == 1) {
                weightSlice.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            }
            mm.SetTensorB(weightSlice);
            mm.SetQuantVector(scaleGm[groupIdx * tiling->n * tiling->quantGroupNum + loopK * tiling->n + singleNOffset + baseNIdx * mnConfig.baseN]);
            mm.Iterate();
            mm.GetTensorC(mmOutGm[mnConfig.workSpaceOffset], loopK == 0 ? 0 : 1, true);
        }
        mm.End();
        mnConfig.workSpaceOffset += mnConfig.baseM * mnConfig.baseN;
        CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);  // 2: mode为2, group内同步
        baseCubeCount++;
    }
    xL1Queue.FreeTensor(xL1LocalInt4);
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::MMCompute(uint32_t groupIdx, MNConfig& mnConfig)
{
    uint32_t singleNOffset = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.singleN;
    if (unlikely(mnConfig.nIdx == mnConfig.blockDimN - 1)) {
        curSingleN = tiling->n - singleNOffset;
    }
    uint32_t curSingleM = mnConfig.singleM;
    if (unlikely(mnConfig.mIdx == mnConfig.blockDimM - 1)) {
        curSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    }
    MMComputeSingleN(groupIdx, mnConfig, singleNOffset, curSingleN, curSingleM);
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::CopyMMOutLocal(LocalTensor<cT::T> &mmOutTensor, uint32_t dstOffset, uint64_t srcAddr,
                            uint32_t curVecBaseM, uint32_t curVecBaseN)
{
    DataCopy2DDimParams copyDimParams{static_cast<uint32_t>(curVecBaseM),
                                      static_cast<uint32_t>(curVecBaseN),
                                      static_cast<uint32_t>(curVecBaseN * static_cast<uint32_t>(2))};
    if constexpr (mmType::BT::format == CubeFormat::ND) {
        DataCopyPad2DA8W4ND(mmOutTensor[dstOffset],
            mmOutGm[srcAddr], copyDimParams);    // 2: 2 lines int4 to 1 line int8
    } else {
        DataCopyPad2DA8W4(mmOutTensor[dstOffset],
            mmOutGm[srcAddr], copyDimParams);    // 2: 2 lines int4 to 1 line int8
    }
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::VectorAtomicProcess(uint32_t curVecBaseM, uint32_t curVecBaseN, uint32_t alignBaseN,
                                        uint32_t rowIndexOffset, uint64_t yGmOffset)
{
    LocalTensor<DTYPE_OUT> yLocal = vecOutQueue.DeQue<DTYPE_OUT>();

    SetAtomicAdd<float>();
    DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curVecBaseN * sizeof(float)), 1, 1, 0};
    for (uint32_t i = 0; i < curVecBaseM; i++) {
        auto outRow = static_cast<uint64_t>(tokenRanksGm.GetValue(rowIndexOffset + i));
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopyPad(yGm[outRow * tiling->n + yGmOffset], yLocal[i * alignBaseN], paramsOut);
    }
    SetAtomicNone();

    vecOutQueue.FreeTensor(yLocal);
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::VectorProcess(VectorAtomicParams& vecAParams, MNConfig& mnConfig,
    uint64_t mmOutOffset, LocalTensor<cT::T>& mmOutLocal)
{
    uint32_t targetAddr = vecAParams.curVecBaseM * vecAParams.alignBaseN;
    uint64_t lowBitAddr = mmOutOffset + (vecAParams.offsetM * 2UL + 1UL) * vecAParams.curVecBaseN;
    CopyMMOutLocal(mmOutLocal, targetAddr, lowBitAddr, vecAParams.curVecBaseM, vecAParams.curVecBaseN);
    PipeBarrier<PIPE_MTE2>();
    vecInQueue.EnQue(mmOutLocal);
    ComputeDequantAndActivate(mnConfig, vecAParams.curVecBaseM, vecAParams.alignBaseN, vecAParams.curVecBaseN,
                              vecAParams.offsetM);
    VectorAtomicProcess(vecAParams.curVecBaseM, vecAParams.curVecBaseN, vecAParams.alignBaseN,
                        vecAParams.yGmOffset0, vecAParams.yGmOffset1);
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::VectorCompute(uint32_t groupIdx, MNConfig& mnConfig)
{
    uint32_t curCubeSingleN = (mnConfig.nIdx == mnConfig.blockDimN - 1) ? tiling->n - mnConfig.nIdx * mnConfig.singleN : mnConfig.singleN;
    uint32_t curCubeSingleM = mnConfig.singleM / 2;
    uint32_t mGlobalOffset = mnConfig.offsetM / 2 + mnConfig.mIdx * curCubeSingleM; // 2: 2 lines int4 to 1 line int8
     // 2: 2 lines int4 to 1 line int8
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {
        curCubeSingleM = mnConfig.m / 2 - mnConfig.mIdx * curCubeSingleM;
    }
    uint32_t vecBaseM = tiling->ubCalSize / (Ceil(mnConfig.baseN, uint32_t(8)) * 8);  //  8: num int32_t in 32B ub block  32*256/256
    vecBaseM = vecBaseM < curCubeSingleM ? vecBaseM : curCubeSingleM;
    uint32_t curVecBaseN = mnConfig.baseN;
    uint64_t scaleOffset = (static_cast<uint64_t>(groupIdx)) * tiling->n + mnConfig.nIdx * mnConfig.singleN;
    uint64_t offsetOffset = scaleOffset;
    uint32_t taskRation = GetTaskRation();
    for (uint32_t offsetN = 0; offsetN < curCubeSingleN; offsetN += mnConfig.baseN) {
        if (baseCubeCount % mnConfig.baseNEachSingleN == 0) { // 进入新一轮 更新workspace从起点开始
            mnConfig.workSpaceOffset = mnConfig.baseM * mnConfig.singleN * coreIdx;
        }
        CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
        curVecBaseN = unlikely(offsetN + mnConfig.baseN >= curCubeSingleN) ? curCubeSingleN - offsetN : curVecBaseN;
        uint32_t alignBaseN = Ceil(curVecBaseN, uint32_t(8)) * 8;  //  8: num int32_t in 32B ub block
        DataCopyScale(curVecBaseN, alignBaseN, scaleOffset + offsetN);
        if (withOffset == uint32_t(1)) {
            DataCopyOffset(curVecBaseN, alignBaseN, offsetOffset + offsetN);
        }
        uint32_t curVecBaseM = vecBaseM;
        for (uint32_t offsetM = 0; offsetM < curCubeSingleM; offsetM += vecBaseM, vecCount++) {
            if (taskRation != 0UL && vecCount % taskRation != subBlockIdx) {
                continue;
            }
            curVecBaseM = unlikely(offsetM + vecBaseM >= curCubeSingleM) ? curCubeSingleM - offsetM : curVecBaseM;

            LocalTensor<cT::T> mmOutLocal = vecInQueue.AllocTensor<cT::T>();
            uint64_t highBitAddr = mnConfig.workSpaceOffset + offsetM * 2UL * curVecBaseN;
            CopyMMOutLocal(mmOutLocal, 0, highBitAddr, curVecBaseM, curVecBaseN);

            if constexpr (mmType::BT::format == CubeFormat::ND) {
                alignBaseN = (alignBaseN + HALF_ALIGN - uint32_t(1)) / HALF_ALIGN * HALF_ALIGN;
            }
            VectorAtomicParams vecAParams{curVecBaseM, curVecBaseN, alignBaseN, offsetM, mGlobalOffset,
                mGlobalOffset + offsetM, mnConfig.nIdx * mnConfig.singleN + offsetN};
            VectorProcess(vecAParams, mnConfig, mnConfig.workSpaceOffset, mmOutLocal);
        }
        scaleInQueue.FreeTensor(scaleInUb);
        if (withOffset == uint32_t(1)) {
            offsetInQueue.FreeTensor(offsetInUb);
        }
        mnConfig.workSpaceOffset += mnConfig.baseM * mnConfig.baseN;
        CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC);  // 2: mode为2, group内同步
        baseCubeCount++;
    }
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::CastMulsAdds( uint32_t computeSize, uint32_t addStartAddr)
{
    LocalTensor<cT::T> mmOutInUb = vecInQueue.DeQue<cT::T>();
    uint32_t castSize;
    if constexpr (mmType::BT::format == CubeFormat::ND) {
        castSize = computeSize + (computeSize + HALF_ALIGN - uint32_t(1)) / HALF_ALIGN * HALF_ALIGN;
    } else {
        castSize = computeSize * uint32_t(2);
    }
    Cast(afterProcessTempBuffer, mmOutInUb, RoundMode::CAST_NONE, castSize);
    PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(mmOutInUb);
    const float RIGHT_MOVE = 16.0f;         // right move int4 to int8
    Muls(floatCalBuffer, afterProcessTempBuffer, RIGHT_MOVE, computeSize);
    PipeBarrier<PIPE_V>();
    Add(afterProcessOutBuffer, afterProcessTempBuffer[addStartAddr], floatCalBuffer, computeSize);
    PipeBarrier<PIPE_V>();
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::AfterMuls(uint32_t addStartAddr)
{
    PipeBarrier<PIPE_V>();
    LocalTensor<DTYPE_OUT> yLocalInUb = vecOutQueue.AllocTensor<DTYPE_OUT>();
    if (withOffset == uint32_t(1)) {
        Mul(yLocalInUb, offsetCalBuffer, afterProcessOutBuffer, addStartAddr);
    } else {
        Mul(yLocalInUb, floatCalBuffer, afterProcessOutBuffer, addStartAddr);
    }
    PipeBarrier<PIPE_V>();
    vecOutQueue.EnQue(yLocalInUb);
}


template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::ComputeDequantAndActivate(MNConfig& mnConfig, 
    uint32_t curVecBaseM, uint32_t alignBaseN, uint32_t curVecBaseN, uint32_t offsetM)
{
    uint32_t computeSize = curVecBaseM * alignBaseN;
    uint32_t addStartAddr;
    if constexpr (mmType::BT::format == CubeFormat::ND) {
        addStartAddr = (computeSize + HALF_ALIGN - uint32_t(1)) / HALF_ALIGN * HALF_ALIGN;
    } else {
        addStartAddr = computeSize;
    }
    CastMulsAdds(computeSize, addStartAddr);
    uint32_t loop = alignBaseN / uint32_t(64); // 256B为64个float，alignBaseN需约束为64倍数
    uint8_t blkStride = static_cast<uint8_t>(alignBaseN * sizeof(float) / uint32_t(32));  //32: 单位32B
    BinaryRepeatParams param(1, 1, 1, blkStride, blkStride, 0);
    uint64_t mask = 64;
    uint64_t last = alignBaseN % uint64_t(64);
    for (uint32_t i = 0; i < loop; i++) {
        uint32_t offset = i * uint32_t(64); // 每次64个元素
        Add(floatCalBuffer[offset], afterProcessOutBuffer[offset], scaleInUb[offset], mask, curVecBaseM, param);
    }
    PipeBarrier<PIPE_V>();
    if (unlikely(last > uint64_t(0))) {
        uint32_t offset = loop * uint32_t(64);
        Add(floatCalBuffer[offset], afterProcessOutBuffer[offset], scaleInUb[offset], last, curVecBaseM, param);
    }
    PipeBarrier<PIPE_V>();
    if (withOffset == uint32_t(1)) {
        DataCopyAndBrcbOfRowSum(mnConfig, curVecBaseM, alignBaseN, offsetM);
        PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < loop; i++) {
            uint32_t offset = i * uint32_t(64); // 每次64个元素
            Mul(scaleCalBuffer[offset], afterProcessOutBuffer[offset], offsetInUb[offset], mask, curVecBaseM, param);
        }
        PipeBarrier<PIPE_V>();
        if (unlikely(last > uint64_t(0))) {
            uint32_t offset = loop * uint32_t(64);
            Mul(scaleCalBuffer[offset], afterProcessOutBuffer[offset], offsetInUb[offset], last, curVecBaseM, param);
        }
        PipeBarrier<PIPE_V>();

        Add(offsetCalBuffer, scaleCalBuffer, floatCalBuffer, computeSize);
        PipeBarrier<PIPE_V>();
    }
    DataCopyPerTokenScaleAndBrcb(mnConfig, curVecBaseM, alignBaseN, offsetM);
    AfterMuls(addStartAddr);
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset)
{
    // GM拷贝scale
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams scaleParams{1, static_cast<uint32_t>(curBaseN * sizeof(float)), 1, 1, 0};
    LocalTensor<float> scaleLocal = scaleInQueue.AllocTensor<float>();
    DataCopyPad(scaleLocal, biasGm[scaleOffset], scaleParams, padParams);
    scaleInQueue.EnQue(scaleLocal);
    scaleInUb = scaleInQueue.DeQue<float>();
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::DataCopyOffset(uint32_t curBaseN, uint32_t alignBaseN, uint64_t offsetOffset)
{
    DataCopyExtParams offsetParams{1, static_cast<uint32_t>(curBaseN * sizeof(float)), 1, 1, 0};
    DataCopyPadExtParams<float> offsetPadParams;
    LocalTensor<float> offsetLocal = offsetInQueue.AllocTensor<float>();
    DataCopyPad(offsetLocal, offsetGm[offsetOffset], offsetParams, offsetPadParams);
    offsetInQueue.EnQue(offsetLocal);
    offsetInUb = offsetInQueue.DeQue<float>();
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::DataCopyPerTokenScaleAndBrcb(MNConfig& mnConfig,
        uint32_t curBaseM, uint32_t alignBaseN, uint32_t offsetM)
{
    uint64_t perTokenScaleOffset = mnConfig.offsetM / 2 + mnConfig.mIdx * mnConfig.singleM / 2 + offsetM; //2: m方向两行合并为1行
    uint32_t alignBaseM = Ceil(curBaseM, uint32_t(8)) * 8;  //  8: num int32_t in 32B ub block
    // GM拷贝per token scale
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams perTokenScaleParams{1, static_cast<uint32_t>(curBaseM * sizeof(float)), 0, 0, 0};
    LocalTensor<float> perTokenScaleLocal = perTokenScaleInQueue.AllocTensor<float>();
    DataCopyPad(perTokenScaleLocal, perTokenScaleGm[perTokenScaleOffset], perTokenScaleParams, padParams);

    DataCopyPad(perTokenScaleLocal[alignBaseM], logitsGm[perTokenScaleOffset], perTokenScaleParams, padParams);

    perTokenScaleInQueue.EnQue(perTokenScaleLocal);

    perTokenScaleLocal = perTokenScaleInQueue.DeQue<float>();
    auto scaleTmp = perTokenScaleLocal;

    Mul(scaleCalBuffer, perTokenScaleLocal, perTokenScaleLocal[alignBaseM], curBaseM);
    PipeBarrier<PIPE_V>();
    scaleTmp = scaleCalBuffer;

    const uint32_t broadCastDst[2] = {curBaseM, alignBaseN};
    const uint32_t broadCastSrc[2] = {curBaseM, 1};
    BroadCast<float, 2, 1>(afterProcessOutBuffer, scaleTmp, broadCastDst, broadCastSrc, tempBuffer);
    perTokenScaleInQueue.FreeTensor(perTokenScaleLocal);
}

template <typename mmType>
__aicore__ inline void GMMA8W4MSDL1OptCompute<mmType>::DataCopyAndBrcbOfRowSum(MNConfig& mnConfig,
        uint32_t curBaseM, uint32_t alignBaseN, uint32_t offsetM)
{
    const uint64_t xRowSumOffset = mnConfig.offsetM / 2 + mnConfig.mIdx * mnConfig.singleM / 2 + offsetM; //2: m方向两行合并为1行

    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams xRowSumParams{1, static_cast<uint32_t>(curBaseM * sizeof(float)), 0, 0, 0};
    LocalTensor<float> xRowSumLocal = xRowSumInQueue.AllocTensor<float>();
    DataCopyPad(xRowSumLocal, xRowSumGm[xRowSumOffset], xRowSumParams, padParams);

    xRowSumInQueue.EnQue(xRowSumLocal);
    xRowSumLocal = xRowSumInQueue.DeQue<float>();

    const uint32_t xRowSumBroadCastDst[2] = {curBaseM, alignBaseN};
    const uint32_t xRowSumBroadCastSrc[2] = {curBaseM, 1};
    BroadCast<float, 2, 1>(afterProcessOutBuffer, xRowSumLocal, xRowSumBroadCastDst, xRowSumBroadCastSrc, tempRowSumBuffer);
    xRowSumInQueue.FreeTensor(xRowSumLocal);
}
} // namespace GROUPED_MATMUL
#endif