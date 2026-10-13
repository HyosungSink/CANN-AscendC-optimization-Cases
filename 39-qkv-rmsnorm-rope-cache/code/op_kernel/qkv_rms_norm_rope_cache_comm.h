/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qkv_rms_norm_rope_cache_comm.h
 * \brief
 */
#ifndef _QKV_RMS_NORM_ROPE_CACHE_COMM_H_
#define _QKV_RMS_NORM_ROPE_CACHE_COMM_H_
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h" 

namespace QkvRmsNormRopeCache {
using namespace AscendC;

constexpr int64_t NUM_ZERO = 0;
constexpr int64_t NUM_ONE = 1;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_THREE = 3;
constexpr int64_t NUM_FOUR = 4;
constexpr int64_t NUM_FIVE = 5;
constexpr int64_t NUM_SIX = 6;
constexpr int64_t NUM_EIGHT = 8;
constexpr int64_t NUM_TWELVE = 12;
constexpr int64_t NUM_SIXTEEN = 16;
constexpr int64_t NUM_TWENTY_FOUR = 24;
constexpr int64_t NUM_FOURTY_EIGHT = 48;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t NUM_SIXTY_FOUR = 64;
constexpr int64_t GAMMA_BUFFER_COUNT = 2;
constexpr int64_t X_BUFFER_COUNT = 2;
constexpr int64_t OUT_BUFFER_COUNT = 2;
constexpr int64_t WORKSPACE_BUFFER_MULTIPLIER = 3;
constexpr int64_t D1_SIZE2_PA_NZ_QUANT = 2;
constexpr int64_t D1_SIZE4_PA_NZ_QUANT = 4;
constexpr int64_t D1_SIZE6_PA_NZ_QUANT = 6;
constexpr int64_t D1_SIZE16_PA_NZ_QUANT = 16;
constexpr int64_t D0_SIZE32_PA_NZ_QUANT = 32;
constexpr int64_t D1_SIZE4_PA_NZ = 4;
constexpr int64_t D1_SIZE8_PA_NZ = 8;
constexpr int64_t D1_SIZE12_PA_NZ = 12;
constexpr int64_t D1_SIZE32_PA_NZ = 32;
constexpr int64_t D0_SIZE16_PA_NZ = 16;

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

__aicore__ inline void SToMTE3Sync()
{
    event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
}

// Template Branch Levels:
//  1) K/V TYPE Consistency
//  2) Quant Mode Availability: Differentiated Quant Implementation according to Tensor Fractal and Quant Inputs.  

template <typename QKV_DTYPE, typename K_DTYPE, typename V_DTYPE>
class KernelQkvRmsNormRopeCacheBase{
public:
    __aicore__ inline KernelQkvRmsNormRopeCacheBase(TPipe* pipe, const QkvRmsNormRopeCacheTilingData* tiling)
        : pipe_(pipe), tilingData_(tiling)
    {}

    __aicore__ inline void InitSharedData(){
        this->rmsNormLength = this->tilingData_->qkvDim;
        this->ropeLength = this->tilingData_->ropeRange;
        this->numHead = this->tilingData_->numHead;
        this->numHeadQ = this->tilingData_->numHeadQ;
        this->numHeadK = this->tilingData_->numHeadK;
        this->numHeadV = this->tilingData_->numHeadV;
    }

protected:
    int64_t rmsNormLength{128};
    int64_t ropeLength{128};
    int64_t numHead {18};
    int64_t numHeadQ {16};
    int64_t numHeadK{1};
    int64_t numHeadV{1};

    // Sahred members
    TPipe* pipe_ = nullptr;
    const QkvRmsNormRopeCacheTilingData* tilingData_;

    GlobalTensor<QKV_DTYPE> qkvGm, gammaQGm, gammaKGm, cosGm, sinGm, qOutGm;
    GlobalTensor<K_DTYPE> kCacheGm;
    GlobalTensor<V_DTYPE> vCacheGm;
    GlobalTensor<int64_t> indexGm;

    TQue<QuePosition::VECIN, 1> inQueueX;       // (ubFator * rms_len) or max(ubFactorQ, ubFactorK, ubFactorV) * rms_len
    TQue<QuePosition::VECIN, 1> inQueueY;       // (ubFator * rms_len) or max(ubFactorQ, ubFactorK, ubFactorV) * rms_len 
};

template <typename QKV_DTYPE, typename K_DTYPE, typename V_DTYPE>
class KernelQkvRmsNormRopeCacheCutBSN : public KernelQkvRmsNormRopeCacheBase<QKV_DTYPE, K_DTYPE, V_DTYPE> {
public:
    __aicore__ inline KernelQkvRmsNormRopeCacheCutBSN(TPipe* pipe, const QkvRmsNormRopeCacheTilingData* tiling)
        : KernelQkvRmsNormRopeCacheBase<QKV_DTYPE, K_DTYPE, V_DTYPE>(pipe, tiling)
    {}
    
protected:
    TBuf<TPosition::VECCALC> locBuf0;  // (ubFactor * rms_len) 
    TBuf<TPosition::VECCALC> locBuf1;  // (ubFactor * rms_len) 
    TBuf<TPosition::VECCALC> locBuf2;  // (ubFactor * rms_len) 
    TBuf<TPosition::VECCALC> locBuf3;  // (ubFactor * rms_len) 
    // Specialized buffer 
    TBuf<TPosition::VECCALC> gammaBuf; // rms_len -> gmmma
    TQue<QuePosition::VECOUT, 1> outQueue; // (ubFactor * rms_len) * (half)

    int64_t ubLoopQ{1};
    int64_t ubFactorQ{8};
    int64_t ubTailQ{0};

    int64_t ubLoopK{1};
    int64_t ubFactorK{8};
    int64_t ubTailK{0};
    int64_t isK{0};

    int64_t ubLoopV{1};
    int64_t ubFactorV{8};
    int64_t ubTailV{0};
    int64_t isV{0};

    GlobalTensor<float> kScaleGm, vScaleGm;       // SYMMETRIC QUANT
    GlobalTensor<float> kOffsetGm, vOffsetGm;     // ASYMMETRIC QUANT: Missing implementation
    GlobalTensor<QKV_DTYPE> qCacheGmNd, kCacheGmNd, vCacheGmNd;
};

} // namespace QkvRmsNormRopeCache
#endif // _QKV_RMS_NORM_ROPE_CACHE_COMM_H_