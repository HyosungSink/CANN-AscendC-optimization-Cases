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
 * \file rotate_matrix.h
 * \brief
 */
#ifndef ROTATE_MATRIX_H
#define ROTATE_MATRIX_H

#include "lib/matmul_intf.h"

struct MMConfig {
    uint32_t m_ = 0;
    uint32_t n_ = 0;
    uint32_t k_ = 0;
    uint32_t baseM_ = 0;
    uint32_t baseN_ = 0;
    uint32_t curSingleM_ = 0;
    uint32_t curSingleN_ = 0;
    uint32_t blockDimM_ = 0;
    uint32_t blockDimN_ = 0;
    uint32_t mIdx_ = 0;
    uint32_t nIdx_ = 0;
    uint32_t singleM_ = 0;
    uint32_t singleN_ = 0;
    uint64_t baseOffsetM_ = 0;
    uint64_t workspaceOffset_ = 0;
};

struct ROPEInitParams {
    GM_ADDR x;
    GM_ADDR cos;
    GM_ADDR sin;
    GM_ADDR rotationMatrix;
    GM_ADDR y;
    GM_ADDR workspace;
};

struct VectorOffsetParams {
    uint32_t singleCoreM;
    uint32_t offsetMStart;
    uint32_t offsetMEnd;
};

struct SinCosCopyParams {
    uint32_t curVecBaseM = 0;
    uint32_t broadShape = 0;
    uint64_t cosSinGMOffset = 0;
    uint32_t globalOffsetM = 0;
    uint32_t copyUbOffsetM = 0;
};

struct ShapeParams {
    uint64_t X1 = 0;
    uint64_t X2 = 0;
    uint64_t X3 = 0;
    uint64_t R1 = 0;
    uint64_t R2 = 0;
    uint64_t R3 = 0;
    uint64_t D = 0;
    uint64_t x2X3Size = 0;
    uint64_t x3DSize = 0;
    uint64_t r2R3DSize = 0;
    uint64_t r3DSize = 0;
    uint32_t broadcastFirstDim = 0;
    uint32_t broadcastSecondDim = 0;
    uint32_t broadcastThirdDim = 0;
};

namespace RotateMatrix {
using namespace AscendC;
using namespace matmul;

constexpr uint64_t BUFFER_NUM = 1;
constexpr uint64_t SYNC_AIV_TO_AIC = 1;
constexpr uint64_t SYNC_AIC_TO_AIV = 2;
const uint64_t TILING_MODE_BNSD_BROADCAST_TWODIM = 2;
const uint64_t TILING_MODE_BSND_BROADCAST_TWODIM = 3;
const uint64_t TILING_MODE_BNSD_BROADCAST_ONEDIM = 5;

template <typename inType, typename outType, typename MT>
class RotateMatrixAll {
public:
    __aicore__ inline RotateMatrixAll(MT &mm_) : mm(mm_){};

    __aicore__ inline void SetGlobalTensors(const ROPEInitParams &initParams);
    __aicore__ inline void InitLocalBuffers();
    __aicore__ inline void InitShapeParams(const RotaryPositionEmbeddingTilingData &tilingData);
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR rotate, GM_ADDR y, GM_ADDR workSpace,
                                const RotaryPositionEmbeddingTilingData &tilingData, TPipe *pipe);

    __aicore__ inline void Process();
    __aicore__ inline void MMCompute(uint32_t curBlock);
    __aicore__ inline void VectorCompute(uint32_t curBlock);

    __aicore__ inline void VectorComputeOffset(uint32_t curBlock);
    __aicore__ inline void VectorComputePre(uint32_t curBlock, uint32_t curVecBaseM, uint32_t offsetM);
    __aicore__ inline void VectorComputeProcess(uint32_t curBlock, uint32_t curVecBaseM, uint32_t offsetM);

    __aicore__ inline void DataCopyX(GlobalTensor<inType> &xGM, LocalTensor<inType> &xLocal, uint32_t curVecBaseM,
                                     uint32_t offsetM);
    __aicore__ inline void DataCopyxRotated(GlobalTensor<float> &xRotatedGM, LocalTensor<float> &xRotatedLocal,
                                            uint32_t curVecBaseM, uint32_t offsetM);
    __aicore__ inline void DataCopyOut(uint32_t curVecBaseM, uint32_t offsetM);

    __aicore__ inline void DataCopySinCos(GlobalTensor<inType> &cosSinGM, LocalTensor<inType> &cosSinLocal,
                                          LocalTensor<inType> &tmpUb, uint32_t curVecBaseM, uint32_t offsetM);

    __aicore__ inline void DataCopySinCosBXXD(GlobalTensor<inType> &cosSinGM, LocalTensor<inType> &cosSinLocal,
                                              LocalTensor<inType> &tmpUb, SinCosCopyParams &sinCosCopyParams);
    __aicore__ inline void DataCopySinCosXXSD(GlobalTensor<inType> &cosSinGM, LocalTensor<inType> &cosSinLocal,
                                              SinCosCopyParams &sinCosCopyParams);
    __aicore__ inline void DataCopySinCos1S1D(GlobalTensor<inType> &cosSinGM, LocalTensor<inType> &cosSinLocal,
                                              LocalTensor<inType> &tmpUb, SinCosCopyParams &sinCosCopyParams);
    __aicore__ inline void CopyAndBroadcastCosSin(GlobalTensor<inType> &cosSinGM, LocalTensor<inType> &cosSinLocal,
                                                  LocalTensor<inType> &ubLocal, uint32_t curResM,
                                                  SinCosCopyParams &sinCosCopyParams);

protected:
    MT &mm;
    TPipe *pipe_;

    MMConfig mmConfig_;
    VectorOffsetParams vectorOffsetM;
    ROPEInitParams initParams;
    TCubeTiling cubeTiling_; // Matmul Tiling数据
    ShapeParams shape;
    SinCosCopyParams sinCosCopyParams;

    GlobalTensor<inType> xGM_;
    GlobalTensor<inType> cosGM_;
    GlobalTensor<inType> sinGM_;
    GlobalTensor<inType> roMatGM_;
    GlobalTensor<outType> yGM_;
    GlobalTensor<float> workspaceGM_;

    TQue<QuePosition::VECIN, 1> xRotatedInQueue_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECIN, 1> cosSinInQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuff;

    LocalTensor<float> xUbFloat;
    LocalTensor<float> sinCosUbFloat;
    LocalTensor<float> xRotatedUbFloat;
    LocalTensor<inType> sinLocal;
    LocalTensor<inType> cosLocal;
    LocalTensor<outType> yLocal;
    LocalTensor<inType> xLocal;
    LocalTensor<inType> cosSinTmpUb;

    uint32_t coreNum = 0;
    uint32_t subBlockIdx;
    uint32_t coreIdx;
    uint32_t parallNum = 0;
    uint32_t cubeCount = 0;
    uint32_t vectorCount = 0;
    uint32_t vBaseM = 64;
    uint32_t align32Byte = 0;
    uint64_t tilingMode = 0;
};

template <typename inType, typename outType, typename MT>
__aicore__ inline void RotateMatrixAll<inType, outType, MT>::SetGlobalTensors(const ROPEInitParams &initParams)
{
    xGM_.SetGlobalBuffer((__gm__ inType *)initParams.x);
    cosGM_.SetGlobalBuffer((__gm__ inType *)initParams.cos);
    sinGM_.SetGlobalBuffer((__gm__ inType *)initParams.sin);
    roMatGM_.SetGlobalBuffer((__gm__ inType *)initParams.rotationMatrix);
    yGM_.SetGlobalBuffer((__gm__ outType *)initParams.y);
    workspaceGM_.SetGlobalBuffer((__gm__ float *)initParams.workspace);
}

template <typename inType, typename outType, typename MT>
__aicore__ inline void RotateMatrixAll<inType, outType, MT>::InitLocalBuffers()
{
    if ASCEND_IS_AIC {
        return;
    }
    pipe_->InitBuffer(xRotatedInQueue_, BUFFER_NUM, vBaseM * mmConfig_.baseN_ * sizeof(float));
    pipe_->InitBuffer(cosSinInQueue_, BUFFER_NUM, vBaseM * mmConfig_.baseN_ * sizeof(inType));
    pipe_->InitBuffer(xInQueue_, BUFFER_NUM, vBaseM * mmConfig_.baseN_ * sizeof(inType));
    pipe_->InitBuffer(outQueue_, BUFFER_NUM, vBaseM * mmConfig_.baseN_ * sizeof(outType));

    uint64_t buffOffset = 0;
    if (!std::is_same_v<inType, float>) {
        pipe_->InitBuffer(tmpBuff, (2 * vBaseM * mmConfig_.baseN_) * sizeof(float) + vBaseM * mmConfig_.baseN_ * sizeof(inType));

        xUbFloat = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(vBaseM * mmConfig_.baseN_), buffOffset);
        buffOffset += vBaseM * mmConfig_.baseN_ * sizeof(float);
        sinCosUbFloat = tmpBuff.GetWithOffset<float>(static_cast<uint32_t>(vBaseM * mmConfig_.baseN_), buffOffset);
        buffOffset += vBaseM * mmConfig_.baseN_ * sizeof(float);
    } else {
        pipe_->InitBuffer(tmpBuff, vBaseM * mmConfig_.baseN_ * sizeof(inType));
    }

    cosSinTmpUb = tmpBuff.GetWithOffset<inType>(static_cast<uint32_t>(vBaseM * mmConfig_.baseN_), buffOffset);
}

template <typename inType, typename outType, typename MT>
__aicore__ inline void
RotateMatrixAll<inType, outType, MT>::InitShapeParams(const RotaryPositionEmbeddingTilingData &tilingData)
{
    const RotateMatrixParams &rotateTiling = tilingData.rotateMatrixParams;
    shape.X1 = rotateTiling.xFirstDim;
    shape.X2 = rotateTiling.xSecondDim;
    shape.X3 = rotateTiling.xThirdDim;

    shape.R1 = rotateTiling.cosSinFirstDim;
    shape.R2 = rotateTiling.cosSinSecondDim;
    shape.R3 = rotateTiling.cosSinThirdDim;

    shape.D = rotateTiling.dLength;

    shape.x2X3Size = rotateTiling.xSecondDim * rotateTiling.xThirdDim;
    shape.x3DSize = rotateTiling.xThirdDim * rotateTiling.dLength;
    shape.r2R3DSize = rotateTiling.cosSinSecondDim * rotateTiling.cosSinThirdDim * rotateTiling.dLength;
    shape.r3DSize = rotateTiling.cosSinThirdDim * rotateTiling.dLength;

    shape.broadcastFirstDim = rotateTiling.broadcastFirstDim;
    shape.broadcastSecondDim = rotateTiling.broadcastSecondDim;
    shape.broadcastThirdDim = rotateTiling.broadcastThirdDim;
}

template <typename inType, typename outType, typename MT>
__aicore__ inline void RotateMatrixAll<inType, outType, MT>::Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR rotate,
                                                                  GM_ADDR y, GM_ADDR workSpace,
                                                                  const RotaryPositionEmbeddingTilingData &tilingData,
                                                                  TPipe *pipe)
{
    const RotateMatrixParams &rotateTiling = tilingData.rotateMatrixParams;
    mmConfig_.baseM_ = rotateTiling.baseM;
    mmConfig_.baseN_ = rotateTiling.baseN;
    mmConfig_.m_ = rotateTiling.m;
    mmConfig_.blockDimM_ = rotateTiling.blockNumM;
    mmConfig_.blockDimN_ = rotateTiling.blockNumN;
    coreNum = rotateTiling.coreNum;
    parallNum = rotateTiling.cvParallNum;
    tilingMode = rotateTiling.tilingMode;

    pipe_ = pipe;
    initParams = {x, cos, sin, rotate, y, workSpace};
    InitShapeParams(tilingData);
    SetGlobalTensors(initParams);
    InitLocalBuffers();

    coreIdx = GetBlockIdx();
    subBlockIdx = GetSubBlockIdx();
    mmConfig_.n_ = shape.D;
    mmConfig_.k_ = shape.D;
    
    if ASCEND_IS_AIV {
        coreIdx /= GetTaskRation();
    }
    if ASCEND_IS_AIC {
        // 初始化Matmul Tiling
        cubeTiling_ = tilingData.rotateMatrixParams.matmulTiling;
        mm.Init(&cubeTiling_, pipe_);
    }
    if constexpr (std::is_same_v<inType, float>) {
        align32Byte = 8;  // 8: inType为fp32时需8对齐
    } else {
        align32Byte = 16; // 16: inType为bf16/fp16时需16对齐
    }
}

template <typename inType, typename outType, typename MT>
__aicore__ inline void RotateMatrixAll<inType, outType, MT>::Process()
{
    uint64_t loop = 0;
    uint64_t globalOffsetM = 0;
    // 11SD、B1SD
    if (tilingMode == TILING_MODE_BNSD_BROADCAST_TWODIM || tilingMode == TILING_MODE_BNSD_BROADCAST_ONEDIM) {
        loop = shape.X1 * shape.X2;
        globalOffsetM = shape.X3;
    } else { // BNSD、BSND、SBND、BS1D、1S1D、SB1D、S11D
        loop = 1;
        globalOffsetM = 0;
    }
    mmConfig_.baseOffsetM_ = 0;
    uint32_t totalBlock = mmConfig_.blockDimM_ * mmConfig_.blockDimN_;
    mmConfig_.singleM_ = mmConfig_.baseM_;
    mmConfig_.singleN_ = mmConfig_.baseN_;

    for (uint32_t i = 0, preCount = 0; i < loop; i++) {
        uint32_t curBlock = coreIdx >= preCount ? coreIdx : coreIdx + coreNum;
        uint32_t curCount = preCount + totalBlock;
        while (curBlock < curCount) {
            mmConfig_.mIdx_ = (curBlock - preCount) / mmConfig_.blockDimN_;
            mmConfig_.nIdx_ = (curBlock - preCount) % mmConfig_.blockDimN_;
            mmConfig_.curSingleM_ = mmConfig_.baseM_;
            mmConfig_.curSingleN_ = mmConfig_.baseN_;

            if (mmConfig_.mIdx_ == mmConfig_.blockDimM_ - 1) { // m方向尾块
                mmConfig_.curSingleM_ = mmConfig_.m_ - mmConfig_.mIdx_ * mmConfig_.singleM_;
            }
            if (mmConfig_.nIdx_ == mmConfig_.blockDimN_ - 1) { // n方向尾块
                mmConfig_.curSingleN_ = mmConfig_.n_ - mmConfig_.nIdx_ * mmConfig_.singleN_;
            }
            
            if ASCEND_IS_AIC {
                MMCompute(curBlock);
            }
            if ASCEND_IS_AIV {
                VectorCompute(curBlock);
            }
            curBlock += coreNum;
        }
        preCount = curCount % coreNum;
        mmConfig_.baseOffsetM_ += globalOffsetM;
    }
}

template <typename inType, typename outType, typename MT>
__aicore__ inline void RotateMatrixAll<inType, outType, MT>::MMCompute(uint32_t curBlock)
{
    uint64_t xOffset = (mmConfig_.baseOffsetM_ + mmConfig_.mIdx_ * mmConfig_.singleM_) * mmConfig_.k_;
    uint64_t roMatOffset = mmConfig_.nIdx_ * mmConfig_.singleN_;
    mmConfig_.workspaceOffset_ =
        mmConfig_.singleM_ * mmConfig_.singleN_ * (coreIdx + (cubeCount % parallNum) * coreNum);

    mm.SetOrgShape(mmConfig_.m_, mmConfig_.n_, mmConfig_.k_);
    mm.SetSingleShape(mmConfig_.curSingleM_, mmConfig_.curSingleN_, mmConfig_.k_);
    mm.SetTensorA(xGM_[xOffset]);
    mm.SetTensorB(roMatGM_[roMatOffset]);
    while (mm.Iterate()) {
        mm.GetTensorC(workspaceGM_[mmConfig_.workspaceOffset_], 0, true);
    }
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_TO_AIV);
    cubeCount++;
    if (cubeCount >= parallNum) {
        AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
    }
}

template <typename inType, typename outType, typename MT>
__aicore__ inline void RotateMatrixAll<inType, outType, MT>::VectorCompute(uint32_t curBlock)
{
    VectorComputeOffset(curBlock);

    if (vectorOffsetM.singleCoreM <= 0) {
        AscendC::CrossCoreWaitFlag(SYNC_AIC_TO_AIV); // 等待cube
        vectorCount++;
        CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_TO_AIC);
        return;
    }

    uint32_t curVecBaseM = vBaseM;
    for (uint32_t offsetM = vectorOffsetM.offsetMStart, count = 0; offsetM < vectorOffsetM.offsetMEnd;
         offsetM += vBaseM, count++) {
        if (unlikely((offsetM + vBaseM) >= vectorOffsetM.offsetMEnd)) {
            curVecBaseM = vectorOffsetM.offsetMEnd - offsetM;
        }
        VectorComputePre(curBlock, curVecBaseM, offsetM);
        if (count == 0) {
            AscendC::CrossCoreWaitFlag(SYNC_AIC_TO_AIV); // 等待cube
        }
        VectorComputeProcess(curBlock, curVecBaseM, offsetM);
    }
    vectorCount++;
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_TO_AIC);
}

template <typename inType, typename outType, typename MT>
__aicore__ inline void RotateMatrixAll<inType, outType, MT>::VectorComputeOffset(uint32_t curBlock)
{
    vectorOffsetM.singleCoreM = mmConfig_.curSingleM_ / 2; // 分核处理数据
    if (subBlockIdx == 0) {
        vectorOffsetM.offsetMStart = 0;
        vectorOffsetM.offsetMEnd = vectorOffsetM.singleCoreM;
    } else {
        vectorOffsetM.offsetMStart = vectorOffsetM.singleCoreM;
        vectorOffsetM.singleCoreM = mmConfig_.curSingleM_ - vectorOffsetM.singleCoreM;
        vectorOffsetM.offsetMEnd = mmConfig_.curSingleM_;
    }
}

template <typename inType, typename outType, typename MT>
__aicore__ inline void RotateMatrixAll<inType, outType, MT>::VectorComputePre(uint32_t curBlock, uint32_t curVecBaseM,
                                                                              uint32_t offsetM)
{
    // 拷贝cos、x
    cosLocal = cosSinInQueue_.AllocTensor<inType>();
    DataCopySinCos(cosGM_, cosLocal, cosSinTmpUb, curVecBaseM, offsetM);
    cosSinInQueue_.EnQue<inType>(cosLocal);
    cosLocal = cosSinInQueue_.DeQue<inType>();

    xLocal = xInQueue_.AllocTensor<inType>();
    DataCopyX(xGM_, xLocal, curVecBaseM, offsetM);
    xInQueue_.EnQue<inType>(xLocal);
    xLocal = xInQueue_.DeQue<inType>();

    uint32_t computeLen = curVecBaseM * Ceil(mmConfig_.curSingleN_, align32Byte) * align32Byte;
    if constexpr (std::is_same_v<inType, float>) {
        // cos *x
        Mul(xLocal, xLocal, cosLocal, computeLen);
    } else {
        // cos、x数据类型转换为float
        Cast(xUbFloat, xLocal, AscendC::RoundMode::CAST_NONE, computeLen);
        Cast(sinCosUbFloat, cosLocal, AscendC::RoundMode::CAST_NONE, computeLen);
        // cos * x
        Mul(xUbFloat, xUbFloat, sinCosUbFloat, computeLen);
    }
    PipeBarrier<PIPE_V>();
    cosSinInQueue_.FreeTensor(cosLocal);
    // 拷贝sin
    sinLocal = cosSinInQueue_.AllocTensor<inType>();
    DataCopySinCos(sinGM_, sinLocal, cosSinTmpUb, curVecBaseM, offsetM);
    cosSinInQueue_.EnQue<inType>(sinLocal);
    sinLocal = cosSinInQueue_.DeQue<inType>();

    if constexpr (!std::is_same_v<inType, float>) {
        // sin数据类型转换为float
        Cast(sinCosUbFloat, sinLocal, AscendC::RoundMode::CAST_NONE, computeLen);
        PipeBarrier<PIPE_V>();
    }
}

template <typename inType, typename outType, typename MT>
__aicore__ inline void
RotateMatrixAll<inType, outType, MT>::VectorComputeProcess(uint32_t curBlock, uint32_t curVecBaseM, uint32_t offsetM)
{
    uint32_t computeLen = curVecBaseM * Ceil(mmConfig_.curSingleN_, align32Byte) * align32Byte;
    // copy 旋转后的矩阵
    xRotatedUbFloat = xRotatedInQueue_.AllocTensor<float>();
    DataCopyxRotated(workspaceGM_, xRotatedUbFloat, curVecBaseM, offsetM);
    xRotatedInQueue_.EnQue<float>(xRotatedUbFloat);
    xRotatedUbFloat = xRotatedInQueue_.DeQue<float>();

    yLocal = outQueue_.AllocTensor<outType>();
    if constexpr (std::is_same_v<inType, float>) {
        // sin * xRotate
        Mul(xRotatedUbFloat, xRotatedUbFloat, sinLocal, computeLen);
        PipeBarrier<PIPE_V>();
        // cos * x + sin * xRotate
        Add(yLocal, xRotatedUbFloat, xLocal, computeLen);
        outQueue_.EnQue<outType>(yLocal);
    } else {
        // sin * xRotate
        Mul(xRotatedUbFloat, xRotatedUbFloat, sinCosUbFloat, computeLen);
        PipeBarrier<PIPE_V>();
        // cos * x + sin * xRotate
        Add(xRotatedUbFloat, xRotatedUbFloat, xUbFloat, computeLen);
        PipeBarrier<PIPE_V>();
        Cast(yLocal, xRotatedUbFloat, AscendC::RoundMode::CAST_RINT, computeLen);
        outQueue_.EnQue<outType>(yLocal);
    }
    yLocal = outQueue_.DeQue<outType>();

    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);

    DataCopyOut(curVecBaseM, offsetM);

    cosSinInQueue_.FreeTensor(sinLocal);
    xInQueue_.FreeTensor(xLocal);
    xRotatedInQueue_.FreeTensor(xRotatedUbFloat);
    outQueue_.FreeTensor(yLocal);
}

template <typename inType, typename outType, typename MT>
__aicore__ inline void RotateMatrixAll<inType, outType, MT>::DataCopyX(GlobalTensor<inType> &xGM,
                                                                       LocalTensor<inType> &xLocal,
                                                                       uint32_t curVecBaseM, uint32_t offsetM)
{
    DataCopyPadExtParams<inType> padParams;
    DataCopyExtParams Params{static_cast<uint16_t>(curVecBaseM),
                             static_cast<uint32_t>(mmConfig_.curSingleN_ * sizeof(inType)),
                             static_cast<uint32_t>((mmConfig_.n_ - mmConfig_.curSingleN_) * sizeof(inType)), 0, 0};
    uint64_t offset =
        (mmConfig_.baseOffsetM_ + mmConfig_.mIdx_ * mmConfig_.singleM_) * mmConfig_.n_ + mmConfig_.nIdx_ * mmConfig_.singleN_ + offsetM * mmConfig_.n_;

    DataCopyPad(xLocal, xGM[offset], Params, padParams);
}

template <typename inType, typename outType, typename MT>
__aicore__ inline void RotateMatrixAll<inType, outType, MT>::DataCopyxRotated(GlobalTensor<float> &xRotatedGM,
                                                                              LocalTensor<float> &xRotatedLocal,
                                                                              uint32_t curVecBaseM, uint32_t offsetM)
{
    uint32_t alignN = 0;
    if constexpr (!std::is_same_v<inType, float>) {
        alignN = mmConfig_.curSingleN_ % align32Byte <= 8 && mmConfig_.curSingleN_ % align32Byte != 0 ? 1 : 0;
    }

    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams Params{static_cast<uint16_t>(curVecBaseM),
                             static_cast<uint32_t>(mmConfig_.curSingleN_ * sizeof(float)), static_cast<uint32_t>(0),
                             alignN, 0};
    mmConfig_.workspaceOffset_ = mmConfig_.singleM_ * mmConfig_.singleN_ * (coreIdx + (vectorCount % parallNum) * coreNum) +
                               offsetM * mmConfig_.curSingleN_;

    DataCopyPad(xRotatedLocal, xRotatedGM[mmConfig_.workspaceOffset_], Params, padParams);
}

template <typename inType, typename outType, typename MT>
__aicore__ inline void RotateMatrixAll<inType, outType, MT>::DataCopyOut(uint32_t curVecBaseM, uint32_t offsetM)
{
    DataCopyParams yParams{static_cast<uint16_t>(curVecBaseM),
                           static_cast<uint16_t>(mmConfig_.curSingleN_ * sizeof(inType)), 0,
                           static_cast<uint16_t>((mmConfig_.n_ - mmConfig_.curSingleN_) * sizeof(inType))};

    uint64_t offset = (mmConfig_.baseOffsetM_ + mmConfig_.mIdx_ * mmConfig_.singleM_) * mmConfig_.n_ +
                      mmConfig_.nIdx_ * mmConfig_.singleN_ + offsetM * mmConfig_.n_;
    DataCopyPad(yGM_[offset], yLocal, yParams);
}

template <typename inType, typename outType, typename MT>
__aicore__ inline void
RotateMatrixAll<inType, outType, MT>::DataCopySinCos(GlobalTensor<inType> &cosSinGM, LocalTensor<inType> &cosSinLocal,
                                                     LocalTensor<inType> &tmpUb, uint32_t curVecBaseM, uint32_t offsetM)
{
    sinCosCopyParams.globalOffsetM  = mmConfig_.baseOffsetM_ + mmConfig_.mIdx_ * mmConfig_.singleM_ + offsetM;
    uint64_t r1 = sinCosCopyParams.globalOffsetM  / shape.x2X3Size;
    uint64_t r2 = sinCosCopyParams.globalOffsetM  % shape.x2X3Size / shape.X3;
    uint64_t r3 = sinCosCopyParams.globalOffsetM  % shape.x2X3Size % shape.X3;
    sinCosCopyParams.cosSinGMOffset = r1 * shape.r2R3DSize * shape.broadcastFirstDim +
                              r2 * shape.r3DSize * shape.broadcastSecondDim + r3 * shape.D * shape.broadcastThirdDim +
                              mmConfig_.nIdx_ * mmConfig_.singleN_;

    sinCosCopyParams.curVecBaseM = curVecBaseM;

    if (shape.broadcastThirdDim != 0) {
        // 11SD、B1SD、BNSD、BSND、SBND
        DataCopySinCosXXSD(cosSinGM, cosLocal, sinCosCopyParams);
    } else if ( tilingMode == TILING_MODE_BSND_BROADCAST_TWODIM) {
        // 1S1D
        DataCopySinCos1S1D(cosSinGM, cosLocal, tmpUb, sinCosCopyParams);
    } else {
        // BS1D、SB1D、S11D
        DataCopySinCosBXXD(cosSinGM, cosLocal, tmpUb, sinCosCopyParams);
    }
}

// BS1D、SB1D、S11D
template <typename inType, typename outType, typename MT>
__aicore__ inline void RotateMatrixAll<inType, outType, MT>::DataCopySinCosBXXD(
    GlobalTensor<inType> &cosSinGM, LocalTensor<inType> &cosSinLocal, LocalTensor<inType> &ubLocal, SinCosCopyParams &sinCosCopyParams)
{
    if (shape.broadcastSecondDim == 0 && shape.broadcastThirdDim == 0) { // B11SD
        sinCosCopyParams.broadShape = shape.X2 * shape.X3;
    } else if (shape.broadcastThirdDim == 0) {
        sinCosCopyParams.broadShape = shape.X3;
    }
    uint32_t broadUbOffsetM = 0;
    CopyAndBroadcastCosSin(cosSinGM, cosSinLocal, ubLocal, sinCosCopyParams.curVecBaseM, sinCosCopyParams);
}

// 11SD、B1SD、BNSD、BSND、SBND
template <typename inType, typename outType, typename MT>
__aicore__ inline void RotateMatrixAll<inType, outType, MT>::DataCopySinCosXXSD(GlobalTensor<inType> &cosSinGM,
                                                                                LocalTensor<inType> &cosSinLocal,
                                                                                SinCosCopyParams &sinCosCopyParams)
{
    DataCopyPadExtParams<inType> padParams;
    DataCopyExtParams Params{static_cast<uint16_t>(sinCosCopyParams.curVecBaseM),
                             static_cast<uint32_t>(mmConfig_.curSingleN_ * sizeof(inType)),
                             static_cast<uint32_t>((mmConfig_.n_ - mmConfig_.curSingleN_) * sizeof(inType)), 0, 0};
    DataCopyPad(cosSinLocal, cosSinGM[sinCosCopyParams.cosSinGMOffset], Params, padParams);
}

// 1S1D
template <typename inType, typename outType, typename MT>
__aicore__ inline void
RotateMatrixAll<inType, outType, MT>::DataCopySinCos1S1D(GlobalTensor<inType> &cosSinGM,
                                                         LocalTensor<inType> &cosSinLocal, LocalTensor<inType> &ubLocal,
                                                         SinCosCopyParams &sinCosCopyParams)
{
    uint32_t alignBaseN = Ceil(mmConfig_.curSingleN_, align32Byte) * align32Byte;
    uint32_t resM = sinCosCopyParams.curVecBaseM;
    uint32_t cosSinUbOffsetM = 0;

    sinCosCopyParams.copyUbOffsetM = 0;
    sinCosCopyParams.broadShape = shape.X3;

    while (resM) {
        auto cosSinBuf = cosSinLocal[cosSinUbOffsetM * alignBaseN];
        auto copyUbBuf = ubLocal[sinCosCopyParams.copyUbOffsetM * alignBaseN];

        uint64_t x2AndX3Shape = shape.x2X3Size;
        uint64_t x2AndX3Offset = sinCosCopyParams.globalOffsetM % x2AndX3Shape;
        uint32_t resMInX2AndX3 = x2AndX3Shape - x2AndX3Offset;
        uint32_t curResM = 0;

        if (resMInX2AndX3 < resM) {
            curResM = resMInX2AndX3;
            resM = resM - curResM;
        } else {
            curResM = resM;
            resM = 0;
        }

        CopyAndBroadcastCosSin(cosSinGM, cosSinBuf, copyUbBuf, curResM, sinCosCopyParams);

        cosSinUbOffsetM = sinCosCopyParams.curVecBaseM - resM;
        sinCosCopyParams.globalOffsetM = sinCosCopyParams.globalOffsetM + curResM;
        uint64_t r1 = sinCosCopyParams.globalOffsetM / shape.x2X3Size;
        uint64_t r2 = sinCosCopyParams.globalOffsetM % shape.x2X3Size / shape.X3;
        uint64_t r3 = sinCosCopyParams.globalOffsetM % shape.x2X3Size % shape.X3;
        sinCosCopyParams.cosSinGMOffset = r1 * shape.r2R3DSize * shape.broadcastFirstDim +
                                  r2 * shape.r3DSize * shape.broadcastSecondDim +
                                  r3 * shape.D * shape.broadcastThirdDim + mmConfig_.nIdx_ * mmConfig_.singleN_;
    }
}

template <typename inType, typename outType, typename MT>
__aicore__ inline void RotateMatrixAll<inType, outType, MT>::CopyAndBroadcastCosSin(
    GlobalTensor<inType> &cosSinGM, LocalTensor<inType> &cosSinLocal, LocalTensor<inType> &ubLocal, uint32_t curResM,
    SinCosCopyParams &sinCosCopyParams)
{
    uint32_t curCopyNum = 0;
    uint32_t middleCopyNum = 0;

    uint32_t firstBlockNum = 0;
    uint32_t middleBlockNum = 0;
    uint32_t lastBlockNum = 0;

    uint32_t alignBaseN = Ceil(mmConfig_.curSingleN_, align32Byte) * align32Byte;
    uint32_t resMInX3 = sinCosCopyParams.broadShape - (sinCosCopyParams.globalOffsetM % sinCosCopyParams.broadShape);

    if (resMInX3 >= curResM) {
        curCopyNum = 1;
        firstBlockNum = curResM;
    } else {
        curCopyNum += 1;
        firstBlockNum = resMInX3;

        middleCopyNum = (curResM - firstBlockNum) / sinCosCopyParams.broadShape;
        curCopyNum += middleCopyNum;
        middleBlockNum = sinCosCopyParams.broadShape * middleCopyNum;

        if (curResM - firstBlockNum - middleBlockNum > 0) {
            curCopyNum += 1;
            lastBlockNum = curResM - firstBlockNum - middleBlockNum;
        }
    }
    sinCosCopyParams.copyUbOffsetM += curCopyNum;

    DataCopyPadExtParams<inType> padParams;
    DataCopyExtParams Params{static_cast<uint16_t>(curCopyNum),
                             static_cast<uint32_t>(mmConfig_.curSingleN_ * sizeof(inType)),
                             static_cast<uint32_t>((mmConfig_.n_ - mmConfig_.curSingleN_) * sizeof(inType)), 0, 0};

    DataCopyPad(ubLocal, cosSinGM[sinCosCopyParams.cosSinGMOffset], Params, padParams);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    if (firstBlockNum > 0) {
        uint32_t firstXShape[2] = {firstBlockNum, alignBaseN};
        uint32_t firstCosSinShape[2] = {1, alignBaseN};
        Broadcast<inType, 2, 0>(cosSinLocal, ubLocal, firstXShape, firstCosSinShape);
    }
    if (middleBlockNum > 0) {
        PipeBarrier<PIPE_V>();
        uint32_t middleXShape[2] = {middleBlockNum / middleCopyNum, alignBaseN};
        uint32_t middleCosSinShape[2] = {1, alignBaseN};
        for (int i = 0; i < middleCopyNum; i++) {
            Broadcast<inType, 2, 0>(cosSinLocal[(firstBlockNum + i * (middleBlockNum / middleCopyNum)) * alignBaseN],
                                    ubLocal[(1 + i) * alignBaseN], middleXShape, middleCosSinShape);
        }
    }
    if (lastBlockNum > 0) {
        PipeBarrier<PIPE_V>();
        uint32_t lastXShape[2] = {lastBlockNum, alignBaseN};
        uint32_t lastCosSinShape[2] = {1, alignBaseN};
        Broadcast<inType, 2, 0>(cosSinLocal[(firstBlockNum + middleBlockNum) * alignBaseN], ubLocal[(1 + middleCopyNum) * alignBaseN],
                                lastXShape, lastCosSinShape);
    }
}

} // namespace RotateMatrix
#endif // ROTATE_MATRIX_H