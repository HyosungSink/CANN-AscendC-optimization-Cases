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
 * \file qkv_rms_norm_rope_cache_b16_pa_nz_quant.h
 * \brief
 */
#ifndef _QKV_RMS_NORM_ROPE_CACHE_B16_PA_NZ_QUANT_H_
#define _QKV_RMS_NORM_ROPE_CACHE_B16_PA_NZ_QUANT_H_
#include "qkv_rms_norm_rope_cache_comm.h"

namespace QkvRmsNormRopeCache {
using namespace AscendC;

template <typename QKV_DTYPE, typename K_DTYPE, typename V_DTYPE>
class KernelQkvRmsNormRopeCacheQuantB16PANZ
    : public KernelQkvRmsNormRopeCacheCutBSN<QKV_DTYPE, K_DTYPE, V_DTYPE> {
public:
    __aicore__ inline KernelQkvRmsNormRopeCacheQuantB16PANZ(TPipe* pipe, const QkvRmsNormRopeCacheTilingData* tiling)
        : KernelQkvRmsNormRopeCacheCutBSN<QKV_DTYPE, K_DTYPE, V_DTYPE>(pipe, tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR qkv, GM_ADDR q_gamma, GM_ADDR k_gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR q_out, GM_ADDR k_cache, GM_ADDR v_cache, 
        GM_ADDR k_scale, GM_ADDR v_scale, GM_ADDR k_offset, GM_ADDR v_offset, GM_ADDR q_out_proto, GM_ADDR k_cache_proto, GM_ADDR v_cache_proto)
    {
        this->InitSharedData();

        currentBlockFactorQ = this->tilingData_->blockFactorQ; 
        currentBlockFactorK = this->tilingData_->blockFactorK; // tiling传过来处理K的核心需要处理的K的128的数量 
        currentBlockFactorV = this->tilingData_->blockFactorV; // tiling传过来处理V的核心需要处理的V的128的数量
        // 前面的核心计算K 后面的核心计算V 最后的核心Q
        this->isK = GetBlockIdx() < this->tilingData_->blockDimK ? 1 : 0;
        this->isV = (GetBlockIdx() < (this->tilingData_->blockDimK + this->tilingData_->blockDimV) && GetBlockIdx() >= this->tilingData_->blockDimK) ? 1 : 0; 
        
        if (GetBlockIdx() == ((this->tilingData_->blockDimK + this->tilingData_->blockDimV + this->tilingData_->blockDimQ - 1))) {
            currentBlockFactorQ = this->tilingData_->batchSize * this->tilingData_->seqLength * this->tilingData_->numHeadQ -
                                  (this->tilingData_->blockDimQ - 1) * this->tilingData_->blockFactorQ;
        }

        if (GetBlockIdx() == (this->tilingData_->blockDimK - 1)) {
            currentBlockFactorK = this->tilingData_->batchSize * this->tilingData_->seqLength  * this->tilingData_->numHeadK -
                                 (this->tilingData_->blockDimK - 1) * this->tilingData_->blockFactorK; 
        }

        if (GetBlockIdx() == (this->tilingData_->blockDimK + this->tilingData_->blockDimV - 1)) {
            currentBlockFactorV = this->tilingData_->batchSize * this->tilingData_->seqLength  * this->tilingData_->numHeadV -
                                 (this->tilingData_->blockDimV - 1) * this->tilingData_->blockFactorV; 
        }

        this->ubFactorQ = this->tilingData_->ubFactor;
        this->ubLoopQ = currentBlockFactorQ / this->ubFactorQ;
        this->ubTailQ = currentBlockFactorQ - this->ubLoopQ * this->ubFactorQ;
        
        this->ubFactorK = this->tilingData_->ubFactor;
        this->ubLoopK = currentBlockFactorK / this->ubFactorK;
        this->ubTailK = currentBlockFactorK - this->ubLoopK * this->ubFactorK;

        this->ubFactorV = this->tilingData_->ubFactorV;
        this->ubLoopV = currentBlockFactorV / this->ubFactorV;
        this->ubTailV = currentBlockFactorV - this->ubLoopV * this->ubFactorV;

        // init global memory
        this->qkvGm.SetGlobalBuffer(
            (__gm__ QKV_DTYPE*)qkv); // [B*S, N*D]
        this->gammaQGm.SetGlobalBuffer((__gm__ QKV_DTYPE*)q_gamma); // [D]
        this->gammaKGm.SetGlobalBuffer((__gm__ QKV_DTYPE*)k_gamma); // [D]

        this->cosGm.SetGlobalBuffer((__gm__ QKV_DTYPE*)cos); // [B*S, 1*D]
        this->sinGm.SetGlobalBuffer((__gm__ QKV_DTYPE*)sin); // [B*S, 1*D]
        this->indexGm.SetGlobalBuffer((__gm__ int64_t*)index); // [B * ceil_div(S, BlkSize)]
        this->qOutGm.SetGlobalBuffer(((__gm__ QKV_DTYPE*)q_out));  // [B*S, Nq*D]
        this->kCacheGm.SetGlobalBuffer((__gm__ K_DTYPE*)k_cache); // [BlkNum, BlkSize, Nk, D]
        this->vCacheGm.SetGlobalBuffer((__gm__ V_DTYPE*)v_cache); // [BlkNum, BlkSize, Nv, D]
 
        if (this->tilingData_->isOutputQkv) {
            this->qCacheGmNd.SetGlobalBuffer((__gm__ QKV_DTYPE*)q_out_proto);   // [B*S, Nq*D]
            this->kCacheGmNd.SetGlobalBuffer((__gm__ QKV_DTYPE*)k_cache_proto); // [B*S, Nk*D]
            this->vCacheGmNd.SetGlobalBuffer((__gm__ QKV_DTYPE*)v_cache_proto); // [B*S, Nq*D]
        }
        this->kScaleGm.SetGlobalBuffer((__gm__ float*)k_scale);   // [Nk, D]
        this->vScaleGm.SetGlobalBuffer((__gm__ float*)v_scale);   // [Nv, D]

        if (k_offset != nullptr && this->tilingData_->isKQuant) {
            this->kOffsetGm.SetGlobalBuffer((__gm__ float*)k_offset);
            isAsymmetricKQuant = true;
        }
        if (v_offset != nullptr && this->tilingData_->isVQuant) {
            this->vOffsetGm.SetGlobalBuffer((__gm__ float*)v_offset);
            isAsymmetricVQuant = true;
        }

        // qkv [ubFactor, N*D]
        // init pipe
        // ubFactorQ = 64 ->192K ubFactor = 42 -> 189.5k
        if (this->isV) {
            this->pipe_->InitBuffer(this->inQueueX, NUM_TWO, this->ubFactorV * this->rmsNormLength * sizeof(QKV_DTYPE));  // 2 * ubFactorV * 128 * 2 / 1024 = 32K
            this->pipe_->InitBuffer(this->locBuf0, this->ubFactorV * this->rmsNormLength * sizeof(float));                // ubFactorV * 128 * 4 / 1024 = 32K
            this->pipe_->InitBuffer(this->locBuf1, this->ubFactorV * this->rmsNormLength * sizeof(float));                // ubFactorV * 128 * 4 / 1024 = 32K
            this->pipe_->InitBuffer(this->locBuf2, this->ubFactorV * this->rmsNormLength * sizeof(float));                // ubFactorV * 128 * 4 / 1024 = 32K
            this->pipe_->InitBuffer(this->locBuf3, this->ubFactorV * this->rmsNormLength * sizeof(float));                // ubFactorV * 128 * 4 / 1024 = 32K
            this->pipe_->InitBuffer(this->outQueue, NUM_TWO, this->ubFactorV * this->rmsNormLength * sizeof(QKV_DTYPE));  // 2 * ubFactorV * 128 * 2 / 1024 = 32K
        } else {
            this->pipe_->InitBuffer(this->inQueueX, NUM_TWO, this->tilingData_->ubFactor * this->rmsNormLength * sizeof(QKV_DTYPE));  // 2 * ubFactor * 128 * 2 / 1024 = 21K
            this->pipe_->InitBuffer(this->inQueueY, NUM_TWO, this->tilingData_->ubFactor * this->rmsNormLength * sizeof(QKV_DTYPE));  // 2 * ubFactor * 128 * 2 / 1024 = 21K
            this->pipe_->InitBuffer(this->locBuf0, NUM_TWO * this->tilingData_->ubFactor * this->rmsNormLength * sizeof(float));                // ubFactor * 128 * 4 * 2 / 1024 = 42K
            this->pipe_->InitBuffer(this->locBuf1, this->tilingData_->ubFactor * this->rmsNormLength * sizeof(float));                // ubFactor * 128 * 4 / 1024 = 21K
            this->pipe_->InitBuffer(this->locBuf2, NUM_TWO * this->tilingData_->ubFactor * this->rmsNormLength * sizeof(float));                // ubFactor * 128 * 4 * 2 / 1024 = 42K
            this->pipe_->InitBuffer(this->locBuf3, this->tilingData_->ubFactor * this->rmsNormLength * sizeof(float));                // ubFactor * 128 * 4 / 1024 = 21K
            this->pipe_->InitBuffer(this->gammaBuf, this->rmsNormLength * sizeof(float)); // q_gamma or k_gamma                       // 128 * 4 / 1024 = 0.5K
            this->pipe_->InitBuffer(this->outQueue, NUM_TWO, this->tilingData_->ubFactor * this->rmsNormLength * sizeof(QKV_DTYPE));  // 2 * ubFactor * 128 * 2 / 1024 = 21K
        } 
    }

    __aicore__ inline void Process()
    {
        if (this->isK) {
            ProcessK();
        } else if(this->isV) {
            ProcessV();
        } else {
            ProcessQ();
        }
    }

    __aicore__ inline void ProcessK()
    {
        DataCopyPadExtParams<QKV_DTYPE> padParams{false, 0, 0, 0};
        DataCopyExtParams copyParamsContinguous;
        copyParamsContinguous.blockCount = 1;
        // CopyIn gamma
        LocalTensor<float> gammaLocalFp32K = this->gammaBuf.template Get<float>();
        LocalTensor<QKV_DTYPE> gammaLocalK =
            gammaLocalFp32K.template ReinterpretCast<QKV_DTYPE>()[this->rmsNormLength];    
        copyParamsContinguous.blockLen = this->rmsNormLength * sizeof(QKV_DTYPE); 
        DataCopyPad(gammaLocalK, this->gammaKGm, copyParamsContinguous, padParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        Cast(gammaLocalFp32K, gammaLocalK, RoundMode::CAST_NONE, this->rmsNormLength);

        for(int64_t loopIdx = 0; loopIdx < this->ubLoopK; ++loopIdx) {
            int64_t startIdx = GetBlockIdx() * this->tilingData_->blockFactorK + loopIdx * this->ubFactorK; // startIdx表示：Cut K BSN时，当次循环所计算的K的128的起始偏移
            // Calc K
            // SplitK
            LocalTensor<QKV_DTYPE> xLocal = this->inQueueX.template AllocTensor<QKV_DTYPE>();
            SplitQkv<QKV_DTYPE>(xLocal, this->ubFactorK, startIdx, this->numHeadQ, this->numHeadK);
            this->inQueueX.EnQue(xLocal);
            xLocal = this->inQueueX.template DeQue<QKV_DTYPE>();
            // Cast to Fp32
            LocalTensor<float> xLocalFp32 = this->locBuf0.template Get<float>();
            LocalTensor<float> dstLocalFp32 = this->locBuf3.template Get<float>();
            Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, this->ubFactorK * this->rmsNormLength);
            Cast(dstLocalFp32, xLocal, RoundMode::CAST_NONE, this->ubFactorK * this->rmsNormLength);
            this->inQueueX.FreeTensor(xLocal);
            // Calc RmsNorm
            LocalTensor<float> xSquareLocal = this->locBuf1.template Get<float>();
            LocalTensor<float> xSumLocal = this->locBuf2.template Get<float>();
            RmsNorm<QKV_DTYPE>(xLocalFp32, xSquareLocal, xSumLocal, dstLocalFp32, gammaLocalFp32K, this->ubFactorK, this->rmsNormLength);
            // CopyIn Cos Sin
            CopyInRopeFac<QKV_DTYPE>(xLocalFp32, this->ubFactorK, this->ropeLength, this->numHeadK, startIdx);
            // Calc Rope
            Rope<QKV_DTYPE>(dstLocalFp32, xLocalFp32, this->ubFactorK, this->rmsNormLength);
            // Cast to outLocal and CopyOut
            if (this->tilingData_->isOutputQkv) {
                CopyoutProto<QKV_DTYPE>(xLocalFp32, this->kCacheGmNd, startIdx, this->ubFactorK, this->rmsNormLength);
            }
            // Calc : do kQuant
            LocalTensor<QKV_DTYPE> outLocal = this->outQueue.template AllocTensor<QKV_DTYPE>();
            if (this->tilingData_->isKQuant) {
                LocalTensor<float> scaleLocal = this->locBuf1.template Get<float>();
                LocalTensor<float> offsetLocal = this->locBuf2.template Get<float>();
                CopyInQuantFac(scaleLocal, offsetLocal, this->kScaleGm, this->kOffsetGm, this->ubFactorK, startIdx, this->numHeadK, isAsymmetricKQuant);
                LocalTensor<int8_t> quantOutLocal = outLocal.template ReinterpretCast<int8_t>();
                Quant<QKV_DTYPE>(quantOutLocal, dstLocalFp32, scaleLocal, offsetLocal, isAsymmetricKQuant, this->rmsNormLength, this->ubFactorK);
                this->outQueue.EnQue(outLocal);
                outLocal = this->outQueue.template DeQue<QKV_DTYPE>();
                quantOutLocal = outLocal.template ReinterpretCast<int8_t>();
                // Scatter Update kCache
                ScatterUpdateNZWithQuant<int8_t, D1_SIZE4_PA_NZ_QUANT, D0_SIZE32_PA_NZ_QUANT>
                ((GlobalTensor<int8_t>&)this->kCacheGm, quantOutLocal, startIdx, this->ubFactorK, this->rmsNormLength, this->numHeadK);
            } else {
                // Cast and Scatter Update kCache
                LocalTensor<K_DTYPE> outputLocal = outLocal.template ReinterpretCast<K_DTYPE>();
                if constexpr (std::is_same<K_DTYPE, bfloat16_t>::value) {
                    Cast(outputLocal, dstLocalFp32, RoundMode::CAST_RINT, this->ubFactorK * this->rmsNormLength);
                } else if constexpr (std::is_same<K_DTYPE, half>::value) {
                    Cast(outputLocal, dstLocalFp32, RoundMode::CAST_NONE, this->ubFactorK * this->rmsNormLength);
                }
                this->outQueue.EnQue(outLocal);
                outLocal = this->outQueue.template DeQue<QKV_DTYPE>();
                outputLocal = outLocal.template ReinterpretCast<K_DTYPE>();
                ScatterUpdateNZWithQuant<K_DTYPE, D1_SIZE8_PA_NZ, D0_SIZE16_PA_NZ>
                (this->kCacheGm, outputLocal, startIdx, this->ubFactorK, this->rmsNormLength, this->numHeadK);
            }
            this->outQueue.FreeTensor(outLocal);
        } 
        if (this->ubTailK > 0) {
            int64_t startIdx = GetBlockIdx() * this->tilingData_->blockFactorK + this->ubLoopK * this->ubFactorK; // startIdx表示：Cut K BSN时，当次循环所计算的K的128的起始偏移
            // Calc K
            // Split K
            LocalTensor<QKV_DTYPE> xLocal = this->inQueueX.template AllocTensor<QKV_DTYPE>();
            SplitQkv<QKV_DTYPE>(xLocal, this->ubTailK, startIdx, this->numHeadQ, this->numHeadK);
            this->inQueueX.EnQue(xLocal);
            xLocal = this->inQueueX.template DeQue<QKV_DTYPE>();
            // Cast to Fp32
            LocalTensor<float> xLocalFp32 = this->locBuf0.template Get<float>();
            LocalTensor<float> dstLocalFp32 = this->locBuf3.template Get<float>();
            Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, this->ubTailK * this->rmsNormLength);
            Cast(dstLocalFp32, xLocal, RoundMode::CAST_NONE, this->ubTailK * this->rmsNormLength);
            this->inQueueX.FreeTensor(xLocal);
            // Calc RmsNorm
            LocalTensor<float> xSquareLocal = this->locBuf1.template Get<float>();
            LocalTensor<float> xSumLocal = this->locBuf2.template Get<float>();
            RmsNorm<QKV_DTYPE>(xLocalFp32, xSquareLocal, xSumLocal, dstLocalFp32, gammaLocalFp32K, this->ubTailK, this->rmsNormLength);
            // CopyIn Cos Sin
            CopyInRopeFac<QKV_DTYPE>(xLocalFp32, this->ubTailK, this->ropeLength, this->numHeadK, startIdx);
            // Calc Rope
            Rope<QKV_DTYPE>(dstLocalFp32, xLocalFp32, this->ubTailK, this->rmsNormLength);
            // Cast to outLocal and CopyOut
            if (this->tilingData_->isOutputQkv) {
                CopyoutProto<QKV_DTYPE>(xLocalFp32, this->kCacheGmNd, startIdx, this->ubTailK, this->rmsNormLength);
            }
            // Calc : do kQuant
            LocalTensor<QKV_DTYPE> outLocal = this->outQueue.template AllocTensor<QKV_DTYPE>();
            if (this->tilingData_->isKQuant) {
                LocalTensor<float> scaleLocal = this->locBuf1.template Get<float>();
                LocalTensor<float> offsetLocal = this->locBuf2.template Get<float>();
                CopyInQuantFac(scaleLocal, offsetLocal, this->kScaleGm, this->kOffsetGm, this->ubTailK, startIdx, this->numHeadK, isAsymmetricKQuant);
                LocalTensor<int8_t> quantOutLocal = outLocal.template ReinterpretCast<int8_t>();
                Quant<QKV_DTYPE>(quantOutLocal, dstLocalFp32, scaleLocal, offsetLocal, isAsymmetricKQuant, this->rmsNormLength, this->ubTailK);
                this->outQueue.EnQue(outLocal);
                outLocal = this->outQueue.template DeQue<QKV_DTYPE>();
                quantOutLocal = outLocal.template ReinterpretCast<int8_t>();
                // Scatter Update kCache
                ScatterUpdateNZWithQuant<int8_t, D1_SIZE4_PA_NZ_QUANT, D0_SIZE32_PA_NZ_QUANT>
                ((GlobalTensor<int8_t>&)this->kCacheGm, quantOutLocal, startIdx, this->ubTailK, this->rmsNormLength, this->numHeadK);
            } else {
                // Cast and Scatter Update kCache
                LocalTensor<K_DTYPE> outputLocal = outLocal.template ReinterpretCast<K_DTYPE>();
                if constexpr (std::is_same<K_DTYPE, bfloat16_t>::value) {
                    Cast(outputLocal, dstLocalFp32, RoundMode::CAST_RINT, this->ubTailK * this->rmsNormLength);
                } else if constexpr (std::is_same<K_DTYPE, half>::value) {
                    Cast(outputLocal, dstLocalFp32, RoundMode::CAST_NONE, this->ubTailK * this->rmsNormLength);
                }
                this->outQueue.EnQue(outLocal);
                outLocal = this->outQueue.template DeQue<QKV_DTYPE>();
                outputLocal = outLocal.template ReinterpretCast<K_DTYPE>();
                ScatterUpdateNZWithQuant<K_DTYPE, D1_SIZE8_PA_NZ, D0_SIZE16_PA_NZ>
                (this->kCacheGm, outputLocal, startIdx, this->ubTailK, this->rmsNormLength, this->numHeadK);
            }
            this->outQueue.FreeTensor(outLocal);
        }
    }

    __aicore__ inline void ProcessV()
    {
        for(int64_t loopIdx = 0; loopIdx < this->ubLoopV; ++loopIdx) {
            int64_t startIdx = (GetBlockIdx() - this->tilingData_->blockDimK) * this->tilingData_->blockFactorV + loopIdx * this->ubFactorV; 
            // Calc: V
            // SplitV
            LocalTensor<QKV_DTYPE> xLocal = this->inQueueX.template AllocTensor<QKV_DTYPE>();
            SplitQkv<QKV_DTYPE>(xLocal, this->ubFactorV, startIdx, (this->numHeadK + this->numHeadQ), this->numHeadV);
            this->inQueueX.EnQue(xLocal);
            xLocal = this->inQueueX.template DeQue<QKV_DTYPE>();
            // Cast and Copyout vProto
            LocalTensor<float> xLocalFp32 = this->locBuf3.template Get<float>();
            Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, this->ubFactorV * this->rmsNormLength);
            this->inQueueX.FreeTensor(xLocal);
            if (this->tilingData_->isOutputQkv) {
                CopyoutProto<QKV_DTYPE>(xLocalFp32, this->vCacheGmNd, startIdx, this->ubFactorV, this->rmsNormLength);
            }
            // Do VQuant
            LocalTensor<QKV_DTYPE> outLocal = this->outQueue.template AllocTensor<QKV_DTYPE>();
            if (this->tilingData_->isVQuant) {
                LocalTensor<float> scaleLocal = this->locBuf1.template Get<float>();
                LocalTensor<float> offsetLocal = this->locBuf2.template Get<float>();
                CopyInQuantFac(scaleLocal, offsetLocal, this->vScaleGm, this->vOffsetGm, this->ubFactorV, startIdx, this->numHeadV, isAsymmetricVQuant);
                LocalTensor<int8_t> quantOutLocal = outLocal.template ReinterpretCast<int8_t>();
                Quant<QKV_DTYPE>(quantOutLocal, xLocalFp32, scaleLocal, offsetLocal, isAsymmetricVQuant, this->rmsNormLength, this->ubFactorV);
                this->outQueue.EnQue(outLocal);
                outLocal = this->outQueue.template DeQue<QKV_DTYPE>();
                quantOutLocal = outLocal.template ReinterpretCast<int8_t>();
                // Scatter Update vCache
                ScatterUpdateNZWithQuant<int8_t, D1_SIZE4_PA_NZ_QUANT, D0_SIZE32_PA_NZ_QUANT>
                ((GlobalTensor<int8_t>&)this->vCacheGm, quantOutLocal, startIdx, this->ubFactorV, this->rmsNormLength, this->numHeadV);
            } else {
                // Cast
                LocalTensor<V_DTYPE> outputLocal = outLocal.template ReinterpretCast<V_DTYPE>();
                if constexpr (std::is_same<V_DTYPE, bfloat16_t>::value) {
                    Cast(outputLocal, xLocalFp32, RoundMode::CAST_RINT, this->ubFactorV * this->rmsNormLength);
                } else if constexpr (std::is_same<V_DTYPE, half>::value) {
                    Cast(outputLocal, xLocalFp32, RoundMode::CAST_NONE, this->ubFactorV * this->rmsNormLength);
                }
                this->outQueue.EnQue(outLocal);
                outLocal = this->outQueue.template DeQue<QKV_DTYPE>();
                outputLocal = outLocal.template ReinterpretCast<V_DTYPE>();
                // Scatter Update vCache
                ScatterUpdateNZWithQuant<V_DTYPE, D1_SIZE8_PA_NZ, D0_SIZE16_PA_NZ>
                (this->vCacheGm, outputLocal, startIdx, this->ubFactorV, this->rmsNormLength, this->numHeadV);
            }
            this->outQueue.FreeTensor(outLocal);
        } 
        if (this->ubTailV > 0) {
            int64_t startIdx = (GetBlockIdx() - this->tilingData_->blockDimK) * this->tilingData_->blockFactorV + this->ubLoopV * this->ubFactorV; 
            // Calc: V
            // SplitV
            LocalTensor<QKV_DTYPE> xLocal = this->inQueueX.template AllocTensor<QKV_DTYPE>();
            SplitQkv<QKV_DTYPE>(xLocal, this->ubTailV, startIdx, (this->numHeadK + this->numHeadQ), this->numHeadV);
            this->inQueueX.EnQue(xLocal);
            xLocal = this->inQueueX.template DeQue<QKV_DTYPE>();
            // Cast and Copyout vProto
            LocalTensor<float> xLocalFp32 = this->locBuf3.template Get<float>();
            Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, this->ubTailV * this->rmsNormLength);
            this->inQueueX.FreeTensor(xLocal);
            if (this->tilingData_->isOutputQkv) {
                CopyoutProto<QKV_DTYPE>(xLocalFp32, this->vCacheGmNd, startIdx, this->ubTailV, this->rmsNormLength);
            }
            // Do VQuant
            LocalTensor<QKV_DTYPE> outLocal = this->outQueue.template AllocTensor<QKV_DTYPE>();
            if (this->tilingData_->isVQuant) {
                LocalTensor<float> scaleLocal = this->locBuf1.template Get<float>();
                LocalTensor<float> offsetLocal = this->locBuf2.template Get<float>();
                CopyInQuantFac(scaleLocal, offsetLocal, this->vScaleGm, this->vOffsetGm, this->ubTailV, startIdx, this->numHeadV, isAsymmetricVQuant);
                LocalTensor<int8_t> quantOutLocal = outLocal.template ReinterpretCast<int8_t>();
                Quant<QKV_DTYPE>(quantOutLocal, xLocalFp32, scaleLocal, offsetLocal, isAsymmetricVQuant, this->rmsNormLength, this->ubTailV);
                this->outQueue.EnQue(outLocal);
                outLocal = this->outQueue.template DeQue<QKV_DTYPE>();
                quantOutLocal = outLocal.template ReinterpretCast<int8_t>();
                // Scatter Update vCache
                ScatterUpdateNZWithQuant<int8_t, D1_SIZE4_PA_NZ_QUANT, D0_SIZE32_PA_NZ_QUANT>
                ((GlobalTensor<int8_t>&)this->vCacheGm, quantOutLocal, startIdx, this->ubTailV, this->rmsNormLength, this->numHeadV);
            } else {
                // Cast
                LocalTensor<V_DTYPE> outputLocal = outLocal.template ReinterpretCast<V_DTYPE>();
                if constexpr (std::is_same<V_DTYPE, bfloat16_t>::value) {
                    Cast(outputLocal, xLocalFp32, RoundMode::CAST_RINT, this->ubTailV * this->rmsNormLength);
                } else if constexpr (std::is_same<V_DTYPE, half>::value) {
                    Cast(outputLocal, xLocalFp32, RoundMode::CAST_NONE, this->ubTailV * this->rmsNormLength);
                }
                this->outQueue.EnQue(outLocal);
                outLocal = this->outQueue.template DeQue<QKV_DTYPE>();
                outputLocal = outLocal.template ReinterpretCast<V_DTYPE>();
                // Scatter Update vCache
                ScatterUpdateNZWithQuant<V_DTYPE, D1_SIZE8_PA_NZ, D0_SIZE16_PA_NZ>
                (this->vCacheGm, outputLocal, startIdx, this->ubTailV, this->rmsNormLength, this->numHeadV);
            }
            this->outQueue.FreeTensor(outLocal);
        }
    }

    __aicore__ inline void ProcessQ()
    {
        DataCopyPadExtParams<QKV_DTYPE> padParams{false, 0, 0, 0};
        DataCopyExtParams copyParamsContinguous;
        copyParamsContinguous.blockCount = 1;
        // CopyIn gamma
        LocalTensor<float> gammaLocalFp32Q = this->gammaBuf.template Get<float>();
        LocalTensor<QKV_DTYPE> gammaLocalQ =
            gammaLocalFp32Q.template ReinterpretCast<QKV_DTYPE>()[this->rmsNormLength];    
        copyParamsContinguous.blockLen = this->rmsNormLength * sizeof(QKV_DTYPE); 
        DataCopyPad(gammaLocalQ, this->gammaQGm, copyParamsContinguous, padParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        Cast(gammaLocalFp32Q, gammaLocalQ, RoundMode::CAST_NONE, this->rmsNormLength);
        for (int64_t loopIdx = 0; loopIdx < this->ubLoopQ; ++loopIdx) {
            int64_t startIdx = (GetBlockIdx() - this->tilingData_->blockDimK - this->tilingData_->blockDimV) * this->tilingData_->blockFactorQ + loopIdx * this->ubFactorQ;
            // Calc: Q
            // SplitQ
            LocalTensor<QKV_DTYPE> xLocal = this->inQueueX.template AllocTensor<QKV_DTYPE>();
            SplitQkv<QKV_DTYPE>(xLocal, this->ubFactorQ, startIdx, 0, this->numHeadQ);
            this->inQueueX.EnQue(xLocal);
            xLocal = this->inQueueX.template DeQue<QKV_DTYPE>();
            // Cast to Fp32
            LocalTensor<float> xLocalFp32 = this->locBuf0.template Get<float>();
            LocalTensor<float> dstLocalFp32 = this->locBuf3.template Get<float>();
            Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, this->ubFactorQ * this->rmsNormLength);
            Cast(dstLocalFp32, xLocal, RoundMode::CAST_NONE, this->ubFactorQ * this->rmsNormLength);
            this->inQueueX.FreeTensor(xLocal);
            // Calc RmsNorm
            LocalTensor<float> xSquareLocal = this->locBuf1.template Get<float>();
            LocalTensor<float> xSumLocal = this->locBuf2.template Get<float>();
            RmsNorm<QKV_DTYPE>(xLocalFp32, xSquareLocal, xSumLocal, dstLocalFp32, gammaLocalFp32Q, this->ubFactorQ, this->rmsNormLength);
            // CopyIn Cos Sin
            CopyInRopeFac<QKV_DTYPE>(xLocalFp32, this->ubFactorQ, this->ropeLength, this->numHeadQ, startIdx);
            // Calc Rope
            Rope<QKV_DTYPE>(dstLocalFp32, xLocalFp32, this->ubFactorQ, this->rmsNormLength);
            // Cast to outLocal and CopyOut
            if (this->tilingData_->isOutputQkv) {
                CopyoutProto<QKV_DTYPE>(xLocalFp32, this->qCacheGmNd, startIdx, this->ubFactorQ, this->rmsNormLength);
            }
            // do Copyout qOut
            CopyoutProto<QKV_DTYPE>(dstLocalFp32, this->qOutGm, startIdx, this->ubFactorQ, this->rmsNormLength);
        }
        if (this->ubTailQ > 0) {
            int64_t startIdx = (GetBlockIdx() - this->tilingData_->blockDimK - this->tilingData_->blockDimV) * this->tilingData_->blockFactorQ + this->ubLoopQ * this->ubFactorQ;
            // Calc: Q
            // SplitQ
            LocalTensor<QKV_DTYPE> xLocal = this->inQueueX.template AllocTensor<QKV_DTYPE>();
            SplitQkv<QKV_DTYPE>(xLocal, this->ubTailQ, startIdx, 0, this->numHeadQ);
            this->inQueueX.EnQue(xLocal);
            xLocal = this->inQueueX.template DeQue<QKV_DTYPE>();
            // Cast to Fp32
            LocalTensor<float> xLocalFp32 = this->locBuf0.template Get<float>();
            LocalTensor<float> dstLocalFp32 = this->locBuf3.template Get<float>();
            Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, this->ubTailQ * this->rmsNormLength);
            Cast(dstLocalFp32, xLocal, RoundMode::CAST_NONE, this->ubTailQ * this->rmsNormLength);
            this->inQueueX.FreeTensor(xLocal);
            // Calc RmsNorm
            LocalTensor<float> xSquareLocal = this->locBuf1.template Get<float>();
            LocalTensor<float> xSumLocal = this->locBuf2.template Get<float>();
            RmsNorm<QKV_DTYPE>(xLocalFp32, xSquareLocal, xSumLocal, dstLocalFp32, gammaLocalFp32Q, this->ubTailQ, this->rmsNormLength);
            // CopyIn Cos Sin
            CopyInRopeFac<QKV_DTYPE>(xLocalFp32, this->ubTailQ, this->ropeLength, this->numHeadQ, startIdx);
            // Calc Rope
            Rope<QKV_DTYPE>(dstLocalFp32, xLocalFp32, this->ubTailQ, this->rmsNormLength);
            // Cast to outLocal and CopyOut
            if (this->tilingData_->isOutputQkv) {
                CopyoutProto<QKV_DTYPE>(xLocalFp32, this->qCacheGmNd, startIdx, this->ubTailQ, this->rmsNormLength);
            }
            // do Copyout qOut
            CopyoutProto<QKV_DTYPE>(dstLocalFp32, this->qOutGm, startIdx, this->ubTailQ, this->rmsNormLength);
        }
    }

    template <typename T>
    __aicore__ inline void SplitQkv(const LocalTensor<T>& xLocal, int64_t rows, int64_t startIdx, int64_t numHeadBefore, int64_t numHeads)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyParams{
            /* blockCount */ 1,
            /* blockLen (Byte) */ static_cast<uint32_t>(this->rmsNormLength * sizeof(T)),
            /* srcStride */ 0,
            /* dstStride */ 0,
            /* rsv */ 0};
        int64_t tokenIdx = startIdx / numHeads;
        int64_t batchIdx = tokenIdx / this->tilingData_->seqLength;
        int64_t seqIdx = tokenIdx % this->tilingData_->seqLength;
        int64_t nIdx = startIdx % numHeads;
        for(int64_t i = 0; i < rows; ++i) {
            int64_t offset  =  batchIdx * this->tilingData_->seqLength * this->numHead * this->rmsNormLength + 
                               seqIdx * this->numHead * this->rmsNormLength + 
                               numHeadBefore * this->rmsNormLength + nIdx * this->rmsNormLength;
            int64_t ubOffset = i * this->rmsNormLength;
            ++nIdx;
            seqIdx = nIdx < numHeads ? seqIdx : ++seqIdx;
            batchIdx = seqIdx < this->tilingData_->seqLength ? batchIdx : ++batchIdx;
            nIdx = nIdx < numHeads ? nIdx : 0;
            seqIdx = seqIdx < this->tilingData_->seqLength ? seqIdx : 0;
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            DataCopyPad(xLocal[ubOffset], this->qkvGm[offset], copyParams, padParams);   
        }
    }

    __aicore__ inline void CopyInQuantFac(const LocalTensor<float>& scaleLocal, const LocalTensor<float>& offsetLocal, const GlobalTensor<float>& scaleGm, const GlobalTensor<float>& offsetGm,
                                          int64_t rows, int64_t startIdx, int64_t numHead, bool ifAsymmetricQuant)
    {
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyExtParams copyParamsContinguous;
        copyParamsContinguous.blockLen = this->rmsNormLength * sizeof(float);
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        for (int64_t i = 0; i < rows; ++i) {
            int64_t nIdx = (startIdx + i) % numHead;
            int64_t ubOffset = i * this->rmsNormLength; 
            int64_t gmOffset = nIdx * this->rmsNormLength;
            DataCopyPad(scaleLocal[ubOffset], scaleGm[gmOffset], copyParamsContinguous, padParams);
            if (ifAsymmetricQuant) {
                DataCopyPad(offsetLocal[ubOffset], offsetGm[gmOffset], copyParamsContinguous, padParams);
            }
        }
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    }

    template <typename T>
    __aicore__ inline void CopyInRopeFac(const LocalTensor<float>& cosSinLocalFp32, int64_t rows, int64_t ropeLen, int64_t numHead, int64_t startIdx)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyParamsContinguous;
        copyParamsContinguous.blockLen = ropeLen * sizeof(T);
        LocalTensor<T> cosLocal = this->inQueueX.template AllocTensor<T>();
        LocalTensor<T> sinLocal = this->inQueueY.template AllocTensor<T>();
        for(int64_t i = 0; i < rows; ++i) {
            int64_t tokenIdx = (startIdx + i) / numHead;
            int64_t ropeFacOffset = tokenIdx * ropeLen;
            int64_t ubOffset = i * ropeLen;
            DataCopyPad(cosLocal[ubOffset], this->cosGm[ropeFacOffset], copyParamsContinguous, padParams);
            DataCopyPad(sinLocal[ubOffset], this->sinGm[ropeFacOffset], copyParamsContinguous, padParams);
        }
        this->inQueueX.EnQue(cosLocal);
        cosLocal = this->inQueueX.template DeQue<T>();
        this->inQueueY.EnQue(sinLocal);
        sinLocal = this->inQueueY.template DeQue<T>();
        Cast(cosSinLocalFp32, cosLocal, RoundMode::CAST_NONE, rows * ropeLen);
        Cast(cosSinLocalFp32[rows * ropeLen], sinLocal, RoundMode::CAST_NONE, rows * ropeLen);
        PipeBarrier<PIPE_V>();
        this->inQueueX.FreeTensor(cosLocal);
        this->inQueueY.FreeTensor(sinLocal);
    }

    template <typename T>
    __aicore__ inline void Quant(
        const LocalTensor<int8_t>& quantOutLocal, const LocalTensor<float>& srcTensor, const LocalTensor<float>& scaleLocal, const LocalTensor<float>& offsetLocal, 
        bool ifAsymmetricQuant, int64_t headSize, int64_t rows)
    {       
        LocalTensor<float> buf0 = this->locBuf0.template Get<float>();
        int64_t totalMask = headSize;
        while (totalMask > 0) {
            // issue a instruction
            int64_t offset = headSize - totalMask;
            Div(buf0[offset], srcTensor[offset], scaleLocal[offset],
                /*mask*/ totalMask >= NUM_SIXTY_FOUR ? NUM_SIXTY_FOUR : totalMask,
                /*repeat*/ rows, {1, 1, 1, static_cast<uint8_t>(headSize / NUM_EIGHT), static_cast<uint8_t>(headSize / NUM_EIGHT), static_cast<uint8_t>(headSize / NUM_EIGHT)});
            if (ifAsymmetricQuant) {
                PipeBarrier<PIPE_V>();
                Add(buf0[offset], buf0[offset], offsetLocal[offset],
                    totalMask >= NUM_SIXTY_FOUR ? NUM_SIXTY_FOUR : totalMask, rows,
                    {1, 1, 1, static_cast<uint8_t>(headSize / NUM_EIGHT), static_cast<uint8_t>(headSize / NUM_EIGHT), static_cast<uint8_t>(headSize / NUM_EIGHT)});
            }
            totalMask -= NUM_SIXTY_FOUR;
        }
        PipeBarrier<PIPE_V>();
        RoundFloat2Int8(quantOutLocal, buf0, rows * headSize);
    }

    __aicore__ inline void RoundFloat2Int8(
        const LocalTensor<int8_t>& dstTensor, const LocalTensor<float>& srcTensor, int32_t size)
    {
        Cast(srcTensor.ReinterpretCast<int32_t>(), srcTensor, RoundMode::CAST_RINT, size);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(srcTensor.ReinterpretCast<half>(), srcTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        Cast(dstTensor, srcTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, size);
    }

    template <typename T1, int64_t D1, int64_t D0>
    __aicore__ inline void ScatterUpdateNZWithQuant(
         const GlobalTensor<T1>& dst,  const LocalTensor<T1>& outLocal, int64_t startIdx, int64_t rows, int64_t headSize, int64_t numHead)
    {
        DataCopyExtParams copyParamsNz{/*burstNum*/ static_cast<uint16_t>(D1), // 4 
                                       /*burstLength*/ static_cast<uint32_t>(D0 * sizeof(T1)), // 32 
                                       /*srcGap*/ 0,
                                       /*dstGap*/ static_cast<uint32_t>(this->tilingData_->blockSize * D0 * sizeof(T1)) -
                                           static_cast<uint32_t>(D0 * sizeof(T1)), 
                                       0};
        for (int64_t i = 0; i < rows; i++) {
            int64_t tokenId = (startIdx + i) / numHead;
            int64_t nIdx = (startIdx + i) % numHead;
            int64_t ubOffset = headSize * i;                             
            int64_t pageOffset = this->indexGm(tokenId);   // index.shape[Bqkv*Sqkv]->index_value=BlkNum*BlkSize
            // pageOffset = pageId * pageSize + tokenOffsetInCurrentPage
            int64_t pageId = pageOffset / this->tilingData_->blockSize;
            int64_t tokenOffsetInCurrentPage = pageOffset % this->tilingData_->blockSize;
            if (pageOffset >= 0) {
                int64_t gmOffsetNz = pageId * numHead * D1 * this->tilingData_->blockSize * D0 +  //begin
                                     nIdx * D1 * this->tilingData_->blockSize * D0 +    //head
                                     tokenOffsetInCurrentPage * D0;                        //token
                SToMTE3Sync();
                DataCopyPad(dst[gmOffsetNz], outLocal[ubOffset], copyParamsNz); // outLocal.shape=[ubFactor, headSize] -> [BSNk, 128] 
                // dst.shape = [BlkNum, Nk * D1, BlkSize, D0]
            }
        }
    }

    template <typename T>
    __aicore__ inline void RmsNorm(const LocalTensor<float>& xLocalFp32, const LocalTensor<float>& xSquareLocal, const LocalTensor<float>& xSumLocal, const LocalTensor<float>& dstLocalFp32,
                                   const LocalTensor<float>& gammaLocalFp32, int64_t rows, int64_t headSize)
    {
        Mul(xSquareLocal, xLocalFp32, xLocalFp32, rows * headSize);
        PipeBarrier<PIPE_V>();
        int64_t repeatTimes = rows * (headSize >> 1) / NUM_SIXTY_FOUR;
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_SIXTY_FOUR], NUM_SIXTY_FOUR, repeatTimes,
            {1, 1, 1, NUM_EIGHT, NUM_SIXTEEN, NUM_SIXTEEN}); // 64
        PipeBarrier<PIPE_V>();
        WholeReduceSum(xSumLocal, xSquareLocal, NUM_SIXTY_FOUR, rows, 1, 1, NUM_EIGHT);
        PipeBarrier<PIPE_V>();
        // Calc: xSum = xSum * reciprocal
        Muls<float>(xSumLocal, xSumLocal, this->tilingData_->reciprocal, rows);
        PipeBarrier<PIPE_V>();
        // Calc: xSum = xSum + epsilon
        Adds<float>(xSumLocal, xSumLocal, this->tilingData_->epsilon, rows);
        PipeBarrier<PIPE_V>();
        // Calc: xSum = sqrt(xSum)
        Sqrt(xSumLocal, xSumLocal, rows);
        PipeBarrier<PIPE_V>();

        // Brcb
        int64_t block_count = (rows + NUM_EIGHT - 1) / NUM_EIGHT;
        Brcb(xSquareLocal, xSumLocal, block_count, {1, NUM_EIGHT});
        /*Brcb每次只能取8个数去填充，每个数复制成8个相同值（8个相同值是因为一个数复制到一个block里，一个block针对float32是8个数），
        所以迭代次数为rows/8，以遍历所有xSumLocal数
        {1, NUM_EIGHT}是固定写法，1表示单次迭代内，目的操作数不同block间是连续的，8表示不同迭代间间隔8个block*/
        PipeBarrier<PIPE_V>();

        // Calc: xLocalFp32 = xLocalFp32 / xSquareLocal
        for (int64_t rowId = 0; rowId < rows; rowId++) {
            Div(dstLocalFp32[rowId * headSize], dstLocalFp32[rowId * headSize], xSquareLocal[rowId * NUM_EIGHT],
                NUM_SIXTY_FOUR, (headSize / NUM_SIXTY_FOUR), {1, 1, 0, NUM_EIGHT, NUM_EIGHT, 0});
        }
        PipeBarrier<PIPE_V>();

        // Calc: xLocalFp32 = xLocalFp32 * xSquareLocal [rows, RMS_NORM_LENGTH] * [RMS_NORM_LENGTH]
        for (int64_t rowId = 0; rowId < rows; rowId++) {
            Mul(dstLocalFp32[rowId * headSize], dstLocalFp32[rowId * headSize], gammaLocalFp32, headSize);
        }
    }

    template<typename T>
    __aicore__ inline void Rope(
        const LocalTensor<float>& srcLocalFp32, const LocalTensor<float>& cosSinLocalFp32, int64_t rows, int64_t ropeLen)
    {
        // step #1: Get odd and even, cos and sin space
        LocalTensor<float> realLocalFp32 = this->locBuf1.template Get<float>();
        LocalTensor<float> imagLocalFp32 = realLocalFp32[rows * (ropeLen>>1)];
        LocalTensor<float> cosLocalFp32 = cosSinLocalFp32;
        LocalTensor<float> sinLocalFp32 = cosSinLocalFp32[rows * ropeLen];
        LocalTensor<float> y0 = this->locBuf2.template Get<float>();
        LocalTensor<float> y1 = y0[rows * ropeLen];
        // step #2: Half out real and imag
        Copy<float, true>(realLocalFp32, srcLocalFp32, (ropeLen >> 1), rows, {1, 1, static_cast<uint8_t>((ropeLen >> 1) / NUM_EIGHT), static_cast<uint8_t>(ropeLen / NUM_EIGHT)});
        PipeBarrier<PIPE_V>();
        Copy<float, true>(imagLocalFp32, srcLocalFp32[(ropeLen >> 1)], (ropeLen >> 1), rows, {1, 1, static_cast<uint8_t>((ropeLen >> 1) / NUM_EIGHT), static_cast<uint8_t>(ropeLen / NUM_EIGHT)});
        PipeBarrier<PIPE_V>();
        // step #3: y0 = (realLocalFp32, imagLocalFp32) * cosLocalFp32
        Mul(y0, realLocalFp32, cosLocalFp32, /*mask*/ (ropeLen >> 1), /*repeat*/ rows,
            {1, 1, 1, static_cast<uint8_t>(ropeLen / NUM_EIGHT), static_cast<uint8_t>((ropeLen >> 1) / NUM_EIGHT), static_cast<uint8_t>(ropeLen / NUM_EIGHT)});
        Mul(y0[(ropeLen >> 1)], imagLocalFp32, cosLocalFp32[(ropeLen >> 1)], /*mask*/ (ropeLen >> 1),
            /*repeat*/ rows, {1, 1, 1, static_cast<uint8_t>(ropeLen / NUM_EIGHT), static_cast<uint8_t>((ropeLen >> 1) / NUM_EIGHT), static_cast<uint8_t>(ropeLen / NUM_EIGHT)});
        PipeBarrier<PIPE_V>();
        // step #4: y1 = (-imagLocalFp32, realLocalFp32) * sinLocalFp32
        Muls<float>(imagLocalFp32, imagLocalFp32, -1.0f, rows * (ropeLen >> 1));
        PipeBarrier<PIPE_V>();
        Mul(y1, imagLocalFp32, sinLocalFp32, /*mask*/ (ropeLen >> 1), /*repeat*/ rows,
            {1, 1, 1, static_cast<uint8_t>(ropeLen / NUM_EIGHT), static_cast<uint8_t>((ropeLen >> 1) / NUM_EIGHT), static_cast<uint8_t>(ropeLen / NUM_EIGHT)});
        Mul(y1[(ropeLen >> 1)], realLocalFp32, sinLocalFp32[(ropeLen >> 1)], /*mask*/ (ropeLen >> 1),
            /*repeat*/ rows, {1, 1, 1, static_cast<uint8_t>(ropeLen / NUM_EIGHT), static_cast<uint8_t>((ropeLen >> 1) / NUM_EIGHT), static_cast<uint8_t>(ropeLen / NUM_EIGHT)});
        PipeBarrier<PIPE_V>();
        // step #5: y0 = y0 + y1
        Add(srcLocalFp32, y0, y1, rows * ropeLen);
        PipeBarrier<PIPE_V>();
        Copy<float, true>(cosSinLocalFp32, srcLocalFp32, (ropeLen >> 1), rows * NUM_TWO, {1, 1, static_cast<uint8_t>((ropeLen >> 1) / NUM_EIGHT), static_cast<uint8_t>((ropeLen >> 1) / NUM_EIGHT)});
        PipeBarrier<PIPE_V>();
    }

    template <typename T>
    __aicore__ inline void CopyoutProto(const LocalTensor<float>& srcLocalFp32, const GlobalTensor<T>& dstNd, int64_t startIdx, int64_t rows, int64_t headSize)
    {
        LocalTensor<T> outLocal = this->outQueue.template AllocTensor<T>();
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            Cast(outLocal, srcLocalFp32, RoundMode::CAST_RINT, rows * headSize);
        } else if constexpr (std::is_same<T, half>::value) {
            Cast(outLocal, srcLocalFp32, RoundMode::CAST_NONE, rows * headSize);
        }
        this->outQueue.EnQue(outLocal);
        outLocal = this->outQueue.template DeQue<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyParamsNd{
            /* blockCount */ static_cast<uint16_t>(rows),
            /* blockLen (Byte) */ static_cast<uint32_t>(headSize * sizeof(T)),
            /* srcStride */ 0,
            /* dstStride */ 0,
            /* rsv */ 0};
        int64_t gmOffsetNd = startIdx * headSize;
        DataCopyPad(dstNd[gmOffsetNd], outLocal, copyParamsNd); //dstNd-> [B*S, Nq/Nk/Nv * D]
        this->outQueue.FreeTensor(outLocal);
    }

private:
    // indexGm used as indexGmNz in this template.
    bool isAsymmetricKQuant = false;
    bool isAsymmetricVQuant = false;
    int64_t currentBlockFactorK = 0;
    int64_t currentBlockFactorV = 0;
    int64_t currentBlockFactorQ = 0;
};
} // namespace QkvRmsNormRopeCache

#endif // _QKV_RMS_NORM_ROPE_CACHE_B16_PA_NZ_QUANT_H_