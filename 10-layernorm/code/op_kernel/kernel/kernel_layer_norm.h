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
 * \file kernel_layer_norm.h
 * \brief
 */

#ifndef KERNEL_LAYER_NORM_H
#define KERNEL_LAYER_NORM_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace NormKernel {
using namespace AscendC;

template <
    class BlockLayerNormCopyInX_,
    class BlockLayerNormCopyInGamma_,
    class BlockLayerNormCopyInBeta_,
    class BlockLayerNormCopyOutX_,
    class BlockLayerNormCopyOutMean_,
    class BlockLayerNormCopyOutRstd_,
    typename Tfm,
    typename Tweight
>
class KernelLayerNormBasic {
static constexpr int32_t BLOCK_BYTES_SIZE = 32;
static constexpr int32_t FP16_BLOCK_SIZE = 16;
static constexpr int32_t FP32_BYTES_SIZE = 4;

static constexpr uint64_t BLOCK_SIZE = 32;
static constexpr uint64_t SIZE_OF_FLOAT = 4;
static constexpr uint64_t SIZE_OF_FLOAT16 = 2;
static constexpr uint64_t BLOCK_SIZE_OF_FLOAT16 = 16;
static constexpr uint64_t BASE_WSP_SIZE = 32;

static constexpr uint32_t SYS_WORKSPACESIZE = 16 * 1024 * 1024;
static constexpr uint32_t NORMAL_UB_SIZE = 128 * 1024;
static constexpr uint32_t NEED_UB_SIZE = 188 * 1024;
static constexpr uint32_t OTHERS_UB_SIZE = 3 * 1024;
static constexpr uint32_t TMP_UB_SIZE = 1024;
static constexpr uint32_t MEAN_UB_SIZE = 512;
static constexpr uint32_t NUM_TWO = 2;
public:
    using BlockLayerNormCopyInX = BlockLayerNormCopyInX_;
    using BlockLayerNormCopyInGamma = BlockLayerNormCopyInGamma_;
    using BlockLayerNormCopyInBeta = BlockLayerNormCopyInBeta_;
    using BlockLayerNormCopyOutX = BlockLayerNormCopyOutX_;
    using BlockLayerNormCopyOutMean = BlockLayerNormCopyOutMean_;
    using BlockLayerNormCopyOutRstd = BlockLayerNormCopyOutRstd_;

    __aicore__ inline KernelLayerNormBasic() {}

    struct Arguments {
        GM_ADDR x;
        GM_ADDR normalized_shape;
        GM_ADDR gamma;
        GM_ADDR beta;
        GM_ADDR y;
        GM_ADDR mean;
        GM_ADDR rstd;
        GM_ADDR workspace;
        GM_ADDR tiling;
        TPipe *pipe;
    };

    __aicore__ inline void ToUnderlyingArguments(const Arguments &args)
    {
        InitInput(args);
        InitBuffer();
    }

    __aicore__ inline void operator()() {
        if (GetBlockIdx() >= this->blockDim_) {
            return;
        }
        TEventID eventIdSToMte3 = GetTPipePtr()->FetchEventID(HardEvent::S_MTE3);
        TEventID eventIdMte3ToV = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        uint32_t meanCopyOutIndex = 0;
        for (this->curLoop_ = 0;this->curLoop_ < this->loopCount_;this->curLoop_++) {
            this->curMeanIndex_ = this->curLoop_ % meanLocalSize_;
            this->ProcessNormCoefficient();
            this->ProcessNormX();

            if (unlikely(this->curMeanIndex_ == (meanLocalSize_ - 1))) {
                SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
                WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
                blockLayerNormCopyOutMean(this->meanGm_[meanCopyOutIndex * meanLocalSize_], this->meanLocal_, meanLocalSize_);
                blockLayerNormCopyOutRstd(this->rstdGm_[meanCopyOutIndex * meanLocalSize_], this->rstdLocal_, meanLocalSize_);
                meanCopyOutIndex += 1;
                SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
                WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            }
        }
        if (this->curMeanIndex_ != (meanLocalSize_ - 1)) {
            uint32_t tailMeanLocalSize = this->loopCount_ % meanLocalSize_;
            SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
            WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
            blockLayerNormCopyOutMean(this->meanGm_[meanCopyOutIndex * meanLocalSize_], this->meanLocal_, tailMeanLocalSize);
            blockLayerNormCopyOutRstd(this->rstdGm_[meanCopyOutIndex * meanLocalSize_], this->rstdLocal_, tailMeanLocalSize);
        }
    }

    __aicore__ inline void InitInput(const Arguments &args)
    {
        this->pipe_ = args.pipe;
        this->coreNum_ = (int64_t)(*((__gm__ uint64_t *)args.tiling));
        this->colSize_ = (uint32_t)(*((__gm__ uint32_t *)args.tiling + 2));
        this->rowSize_ = (uint32_t)(*((__gm__ uint32_t *)args.tiling + 3));
        this->eps_ = (float)(*((__gm__ float *)args.tiling + 4));
        this->space_ = (uint32_t)(*((__gm__ uint32_t *)args.tiling + 5));
        this->space_ = this->space_ / BLOCK_BYTES_SIZE * BLOCK_BYTES_SIZE; 
        this->coefficient_ = (float)(*((__gm__ float *)args.tiling + 6));
        this->nullptrGamma_ = (uint32_t)(*((__gm__ uint32_t *)args.tiling + 7));
        this->nullptrBeta_ = (uint32_t)(*((__gm__ uint32_t *)args.tiling + 8));

        uint32_t maxtileLength_;

        if (rowSize_ <= coreNum_) {
            this->blockDim_ = rowSize_;
            this->tailBlock_ = 0U;
            this->loopCount_ = 1U;
        } else {
            this->loopCount_ = rowSize_ / coreNum_;
            this->tailBlock_ = rowSize_ % coreNum_;
            this->blockDim_ = coreNum_;
        }

        this->blockLength_ = this->colSize_;
        uint64_t usedUbSize = this->space_ - OTHERS_UB_SIZE;
        maxtileLength_ = usedUbSize / SIZE_OF_FLOAT;
        if (maxtileLength_ / NUM_TWO > this->blockLength_) {
            this->tileLength_ = this->blockLength_;
        } else {
            this->tileLength_ = maxtileLength_ / NUM_TWO;
        }
        this->tileCount_ = (this->blockLength_ + this->tileLength_ - 1) / this->tileLength_;
        this->tailTileLength_ = this->blockLength_ % this->tileLength_;

        tmpBufSize_ = TMP_UB_SIZE;
        uint32_t tileLengthFp16Aligned = (this->tileLength_ + BLOCK_SIZE_OF_FLOAT16 - 1) / 
                                            BLOCK_SIZE_OF_FLOAT16 * BLOCK_SIZE_OF_FLOAT16;
        uint32_t maxtileCount = maxtileLength_ - tileLengthFp16Aligned;
        inBufSize_ = this->space_ - OTHERS_UB_SIZE;
        meanBufSize_ = MEAN_UB_SIZE;
        this->meanLocalSize_ = meanBufSize_ / FP32_BYTES_SIZE;
        rstdBufSize_ = MEAN_UB_SIZE;

        // calculate xGm and paramGm offset and size
        InitGmBuffer(args);
    }

    __aicore__ inline void InitGmBuffer(const Arguments &args) {
        uint32_t xGmOffset = 0;
        uint32_t xGmSize = 0;
        uint32_t paramGmOffset = 0;
        uint32_t paramGmSize = 0;
        if (GetBlockIdx() < this->tailBlock_) {
            this->loopCount_ += 1;
            xGmOffset = this->loopCount_ * this->blockLength_ * GetBlockIdx();
            paramGmOffset = this->loopCount_ * GetBlockIdx();
        } else {
            xGmOffset = this->loopCount_ * this->blockLength_ * GetBlockIdx() +
                    this->blockLength_ * this->tailBlock_;
            paramGmOffset = this->loopCount_ * GetBlockIdx() + this->tailBlock_;
        }
        xGmSize = this->loopCount_ * this->blockLength_;
        paramGmSize = this->loopCount_;

        // init global buffer
        this->xGm_.SetGlobalBuffer((__gm__ Tfm *)args.x + xGmOffset, xGmSize);
        this->yGm_.SetGlobalBuffer((__gm__ Tfm *)args.y + xGmOffset, xGmSize);
        this->meanGm_.SetGlobalBuffer((__gm__ float*)args.mean + paramGmOffset, paramGmSize);
        this->rstdGm_.SetGlobalBuffer((__gm__ float*)args.rstd + paramGmOffset, paramGmSize);
        this->gammaGm_.SetGlobalBuffer((__gm__ Tweight*)args.gamma, this->colSize_);
        this->betaGm_.SetGlobalBuffer((__gm__ Tweight*)args.beta, this->colSize_);
    }

    // template <typename Tfm, typename Tweight>
    __aicore__ inline void InitBuffer()
    {
        this->pipe_->InitBuffer(this->totalBuf_, this->space_);
        // xLocal_使用整个inXBuf大小
        this->xLocal_ = this->totalBuf_.template Get<float>();

        // calcLocal_和yLocal_各使用inXBuf部分空间
        int32_t tileLengthAligned = (this->tileLength_ + BLOCK_BYTES_SIZE - 1) / BLOCK_BYTES_SIZE * BLOCK_BYTES_SIZE;
        uint32_t ubOffset = 0;
        this->calcLocal_ = this->totalBuf_.template GetWithOffset<float>(tileLengthAligned, ubOffset);
        ubOffset += tileLengthAligned * sizeof(float);
        this->yLocal_ = this->totalBuf_.template GetWithOffset<float>(tileLengthAligned, ubOffset);

        tileLengthAligned = TMP_UB_SIZE / SIZE_OF_FLOAT;
        ubOffset = this->inBufSize_;
        // tmpTensor
        this->tmpLocal_ = this->totalBuf_.template GetWithOffset<float>(tileLengthAligned,ubOffset);
        ubOffset += MEAN_UB_SIZE;
        this->meanLocal_ = this->totalBuf_.template GetWithOffset<float>(tileLengthAligned,ubOffset);
        ubOffset += MEAN_UB_SIZE;
        this->rstdLocal_ = this->totalBuf_.template GetWithOffset<float>(tileLengthAligned,ubOffset);
    }

    __aicore__ inline void ProcessNormCoefficient()
    {
        int32_t xTileLength = this->tileLength_;
        int32_t xLoopCount = (this->colSize_ + xTileLength - 1) / xTileLength;
        int32_t xTailLength = this->tailTileLength_;
        xTailLength = (xTailLength == 0) ? xTileLength : xTailLength;
        GlobalTensor<Tfm> curXGm = xGm_[this->curLoop_ * this->colSize_];
        uint32_t curXLength = 0;
        TEventID eventIdVToMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
        TEventID eventIdVToS = GetTPipePtr()->FetchEventID(HardEvent::V_S);

        for (uint32_t i = 0;i < xLoopCount; i++) {
            curXLength = (i == (xLoopCount - 1)) ? xTailLength : xTileLength;
            blockLayerNormCopyInX(this->calcLocal_, curXGm[i * xTileLength], curXLength);

            Muls(this->calcLocal_, this->calcLocal_, this->coefficient_, curXLength);
            PipeBarrier<PIPE_V>();

            ReduceSum<float>(this->yLocal_[i], this->calcLocal_, this->tmpLocal_, curXLength);
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        }
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(this->meanLocal_[this->curMeanIndex_], this->yLocal_, this->tmpLocal_, xLoopCount);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        this->meanX_ = this->meanLocal_[this->curMeanIndex_].GetValue(0);

        for (uint32_t i = 0;i < xLoopCount; i++) {
            curXLength = (i == (xLoopCount - 1)) ? xTailLength : xTileLength;
            blockLayerNormCopyInX(this->calcLocal_, curXGm[i * xTileLength], curXLength);

            Adds(this->calcLocal_, this->calcLocal_, -this->meanX_, curXLength);
            PipeBarrier<PIPE_V>();

            Mul(this->calcLocal_, this->calcLocal_, this->calcLocal_, curXLength);
            PipeBarrier<PIPE_V>();

            ReduceSum<float>(this->yLocal_[i], this->calcLocal_, this->tmpLocal_, curXLength);
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        }
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(this->yLocal_, this->yLocal_, this->tmpLocal_, xLoopCount);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        this->varX_ = this->yLocal_.GetValue(0) * this->coefficient_;
        this->rstd_ = 1.0f / sqrt(this->varX_ + this->eps_);

        this->rstdLocal_.SetValue(this->curMeanIndex_, this->rstd_);
    }

    __aicore__ inline void ProcessNormX()
    {
        int32_t xTileLength = this->tileLength_;
        int32_t xLoopCount = (this->colSize_ + xTileLength - 1) / xTileLength;
        int32_t xTailLength = this->tailTileLength_;
        xTailLength = (xTailLength == 0) ? xTileLength : xTailLength;
        GlobalTensor<Tfm> curXGm = xGm_[this->curLoop_ * this->colSize_];
        GlobalTensor<Tfm> curYGm = yGm_[this->curLoop_ * this->colSize_];
        uint32_t curXLength = 0;
        TEventID eventIdVToMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
        TEventID eventIdMte3ToMte2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);

        for (uint32_t i = 0;i < xLoopCount; i++) {
            curXLength = (i == (xLoopCount - 1)) ? xTailLength : xTileLength;
            blockLayerNormCopyInX(this->calcLocal_, curXGm[i * xTileLength], curXLength);
            
            Adds(this->calcLocal_, this->calcLocal_, -this->meanX_, curXLength);
            PipeBarrier<PIPE_V>();

            Muls(this->calcLocal_, this->calcLocal_, this->rstd_, curXLength);
            PipeBarrier<PIPE_V>();

            if (!nullptrGamma_) {
                blockLayerNormCopyInGamma(this->yLocal_, this->gammaGm_[i * xTileLength], curXLength);

                Mul(this->calcLocal_, this->yLocal_, this->calcLocal_, curXLength);
            }

            if (!nullptrBeta_) {
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                blockLayerNormCopyInBeta(this->yLocal_, this->betaGm_[i * xTileLength], curXLength);

                Add(this->calcLocal_, this->calcLocal_, this->yLocal_, curXLength);
            }

            blockLayerNormCopyOutX(curYGm[i * xTileLength], this->calcLocal_, curXLength);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
    }

protected:
    BlockLayerNormCopyInX blockLayerNormCopyInX;
    BlockLayerNormCopyInGamma blockLayerNormCopyInGamma;
    BlockLayerNormCopyInBeta blockLayerNormCopyInBeta;
    BlockLayerNormCopyOutX blockLayerNormCopyOutX;
    BlockLayerNormCopyOutMean blockLayerNormCopyOutMean;
    BlockLayerNormCopyOutRstd blockLayerNormCopyOutRstd;
    TPipe *pipe_;
    TBuf<> totalBuf_;

    GlobalTensor<Tfm> xGm_;
    GlobalTensor<Tfm> yGm_;
    GlobalTensor<Tweight> gammaGm_;
    GlobalTensor<Tweight> betaGm_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> rstdGm_;

    LocalTensor<float> xLocal_;
    LocalTensor<float> calcLocal_;
    LocalTensor<float> yLocal_;
    LocalTensor<float> tmpLocal_;
    LocalTensor<float> meanLocal_;
    LocalTensor<float> rstdLocal_;

    float sumX_ = 0.0;
    float meanX_ = 0.0;
    float varX_ = 0.0;
    float rstd_ = 0.0;

    int64_t blockDim_ = 0;
    uint32_t colSize_ = 0;
    uint32_t rowSize_ = 0;
    float eps_ = 0.0;
    float coefficient_ = 0.0;
    uint32_t loopCount_ = 0;
    uint32_t tailBlock_ = 0;
    int32_t tileLength_ = 0;
    int32_t tailTileLength_ = 0;
    uint32_t tileCount_ = 0;
    uint32_t blockLength_ = 0;
    uint32_t nullptrGamma_ = 0;
    uint32_t nullptrBeta_ = 0;

    uint32_t meanLocalSize_ = 0;

    int64_t coreNum_ = 0;
    uint32_t curTileLength_ = 0;
    uint32_t curLoop_ = 0;
    uint32_t curMeanIndex_ = 0;
    uint32_t inBufSize_ = 0;
    uint32_t tmpBufSize_ = 0;
    uint32_t meanBufSize_ = 0;
    uint32_t rstdBufSize_ = 0;
    uint32_t space_ = 0;
};
} // namespace NormKernel

#endif // LAYER_NORM_COPY_IN_BLOCK_H