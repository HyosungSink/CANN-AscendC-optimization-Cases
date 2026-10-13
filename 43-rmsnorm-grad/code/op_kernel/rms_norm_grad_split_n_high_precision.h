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
 * \file rms_norm_grad_split_n_high_precision.h
 * \brief
 */
#ifndef _RMS_NORM_GRAD_SPLIT_N_HIGH_PRECISION_H_
#define _RMS_NORM_GRAD_SPLIT_N_HIGH_PRECISION_H_
#include "rms_norm_grad_common.h"

using namespace RmsNormGrad;

template <typename T_DY, typename T_GAMMA>
class RmsNormGradSplitNHighPrecision {
public:
    __aicore__ inline RmsNormGradSplitNHighPrecision()
    {}

    __aicore__ inline void Init(
        GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma,
        const RmsNormGradTilingData* tiling, GM_ADDR usrWorkspace)
    {
        InitVar(tiling);
        InitInputGmBuffer(dy, x, rstd, gamma, blockDim_, coreCalcTail_);
        InitOutputGmBuffer(dx, dgamma);
        InitInputQue();
        InitOutputQue();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        InitTmpBuffer();
        if (isDeterministic_ == 1) {
            InitWorkspace(usrWorkspace);
            workspaceGmOri_.SetGlobalBuffer((__gm__ float*)usrWorkspace);
        } else {
            SyncAll();
        }
#else
        syncTmpSpaceGm_.SetGlobalBuffer((__gm__ int32_t*)usrWorkspace, ALIGN_32 * GetBlockNum());
        uint32_t syncLen = ALIGN_32 * GetBlockNum();
        pipe.InitBuffer(outZeroTmpBuf_, colValAlign_ * sizeof(float));
        pipe.InitBuffer(syncTmpBuf_, syncLen * sizeof(int32_t));

        InitGmZero<int32_t>(syncTmpSpaceGm_, outZeroTmpBuf_, syncLen, (uint32_t)0);
        if (isDeterministic_ != 1) {
            if (GetBlockIdx() == 0) {
                InitGmZero<float>(dgammaGm_, outZeroTmpBuf_, colValAlign_, (uint32_t)0);
            }
            LocalTensor<int32_t> workLocal = syncTmpBuf_.Get<int32_t>();
            SyncAll(syncTmpSpaceGm_, workLocal);
        } else {
            workspaceGm_.SetGlobalBuffer(
                (__gm__ float*)usrWorkspace + ALIGN_32 * GetBlockNum() + GetBlockIdx() * colVal_);
            if (GetBlockIdx() == 0) {
                InitGmZero<float>(workspaceGm_, outZeroTmpBuf_, colValAlign_, (uint32_t)0);
            }
        }
#endif
    }

    __aicore__ inline void InitWorkspace(GM_ADDR usrWorkspace)
    {
        workspaceGm_.SetGlobalBuffer((__gm__ float*)usrWorkspace + GetBlockIdx() * colValAlign_);
    }

    __aicore__ inline void InitVar(const RmsNormGradTilingData* tiling)
    {
        blockDim_ = tiling->block_dim;
        rowVal_ = tiling->row;
        colVal_ = tiling->col;
        avgFactor_ = tiling->avg_factor;
        dataType_ = tiling->data_type;
        coreCalcNum_ = tiling->core_calc_num;
        coreCalcTail_ = tiling->core_calc_tail;
        blockFactor_ = tiling->block_factor;
        ubFactor_ = tiling->ub_factor;
        ubCalcNum = tiling->ub_calc_num;
        ubCalcTail_ = tiling->ub_calc_tail;
        ubCalcLoop_ = tiling->ub_calc_loop;
        ubCalcTailNum_ = tiling->ub_calc_tail_num;
        ubCalcTailTail_ = tiling->ub_calc_tail_tail;
        ubCalcTailLoop_ = tiling->ub_calc_tail_loop;
        alignLen_ = dataType_ == FLOAT_DTYPE ? ALIGN_32 : ALIGN_16;
        colValAlign_ = (colVal_ + alignLen_ - 1) / alignLen_ * alignLen_;
        isDeterministic_ = tiling->fixed_output;
    }

    __aicore__ inline void InitInputGmBuffer(
        GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, uint32_t numBlocks, uint32_t coreCalcTail)
    {
        if (GetBlockIdx() < numBlocks - 1) {
            coreOffset_ = blockFactor_;
        } else {
            coreOffset_ = coreCalcTail > 0 ? coreCalcTail : blockFactor_;
        }
        dyGm_.SetGlobalBuffer((__gm__ T_DY*)dy + GetBlockIdx() * blockFactor_ * colVal_, coreOffset_ * colVal_);
        xGm_.SetGlobalBuffer((__gm__ T_DY*)x + GetBlockIdx() * blockFactor_ * colVal_, coreOffset_ * colVal_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * blockFactor_, coreOffset_);
        gammaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)gamma, colVal_);
    }

    __aicore__ inline void InitOutputGmBuffer(GM_ADDR dx, GM_ADDR dgamma)
    {
        dxGm_.SetGlobalBuffer((__gm__ T_DY*)dx + GetBlockIdx() * blockFactor_ * colVal_, coreOffset_ * colVal_);
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
        dgammaGm_.SetGlobalBuffer((__gm__ float*)dgamma, colValAlign_);
#else
        dgammaGm_.SetGlobalBuffer((__gm__ float*)dgamma, colVal_);
        if (isDeterministic_ == 1) {
            return;
        } else {
            if (GetBlockIdx() == 0) {
                InitOutput<float>(dgammaGm_, colVal_, 0);
            }
        }
#endif
    }

    __aicore__ inline void InitInputQue()
    {
        ubFactorAlign_ = ubFactor_ * colValAlign_;
        rstdLen_ = (ubFactor_ + alignLen_ - 1) / alignLen_ * alignLen_;
        bufferLenSize_ = ubFactorAlign_ * sizeof(float);
        bufferNum_ = BUFFER_NUM_DB;
        pipe.InitBuffer(inQueDY_, bufferNum_, bufferLenSize_);
        pipe.InitBuffer(inQueX_, bufferNum_, bufferLenSize_);
        pipe.InitBuffer(inQueRstd_, bufferNum_, rstdLen_ * sizeof(float));
        pipe.InitBuffer(inQueGamma_, 1, colValAlign_ * sizeof(float));
    }

    __aicore__ inline void InitOutputQue()
    {
        pipe.InitBuffer(outQueDX_, bufferNum_, bufferLenSize_);
        pipe.InitBuffer(outQueDgamma_, 1, colValAlign_ * sizeof(float));
    }

    __aicore__ inline void InitTmpBuffer()
    {
        if (colValAlign_ <= SMALLD_THRESHOLD) {
            pipe.InitBuffer(tmpBuf_, ubFactor_ * ELEM_PER_REP_FP32 * sizeof(float));
            pipe.InitBuffer(tmpMeanBuf_, ubFactor_ * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        CopyGammaIn();
        LocalTensor<float> gammaLocal = inQueGamma_.DeQue<float>();
        Cast2FloatIf<T_GAMMA>(gammaLocal, colValAlign_, colValAlign_);
        LocalTensor<float> dgammaLocal = outQueDgamma_.AllocTensor<float>();
        Duplicate(dgammaLocal, 0.0f, colValAlign_);
        PipeBarrier<PIPE_V>();
        if (coreCalcTail_ == 0) {
            for (uint32_t i = 0; i < (ubCalcTail_ == 0 ? ubCalcLoop_ : ubCalcLoop_ - 1); i++) {
                SubProcess(i, ubCalcNum, gammaLocal, dgammaLocal);
            }
            if (ubCalcTail_ != 0) {
                SubProcess(ubCalcLoop_ - 1, ubCalcTail_, gammaLocal, dgammaLocal);
            }
        } else {
            if (GetBlockIdx() < blockDim_ - 1) {
                for (uint32_t i = 0; i < (ubCalcTail_ == 0 ? ubCalcLoop_ : ubCalcLoop_ - 1); i++) {
                    SubProcess(i, ubCalcNum, gammaLocal, dgammaLocal);
                }
                if (ubCalcTail_ != 0) {
                    SubProcess(ubCalcLoop_ - 1, ubCalcTail_, gammaLocal, dgammaLocal);
                }
            } else {
                for (uint32_t i = 0; i < (ubCalcTailTail_ == 0 ? ubCalcTailLoop_ : ubCalcTailLoop_ - 1); i++) {
                    SubProcess(i, ubCalcTailNum_, gammaLocal, dgammaLocal);
                }
                if (ubCalcTailTail_ != 0) {
                    SubProcess(ubCalcTailLoop_ - 1, ubCalcTailTail_, gammaLocal, dgammaLocal);
                }
            }
        }
        inQueGamma_.FreeTensor(gammaLocal);
        outQueDgamma_.EnQue(dgammaLocal);
        if (isDeterministic_ == 1) {
            CopyDgammaOutWorkspace();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
            LocalTensor<int32_t> workLocal = syncTmpBuf_.Get<int32_t>();
            SyncAll(syncTmpSpaceGm_, workLocal);
            AddDgamma();
#else
            SyncAll();
            doDeter();
#endif
        } else {
            CopyDgammaOut();
        }
    }

    __aicore__ inline void doDeter() {
        LocalTensor<float> buffer1_ = inQueX_.AllocTensor<float>();
        LocalTensor<float> buffer2_ = inQueDY_.AllocTensor<float>();
        deterministic_struct deterministicStruct = {buffer1_, buffer2_, workspaceGmOri_, dgammaGm_};
        FinalProcessDeterministic(colValAlign_, blockDim_, colVal_, deterministicStruct);
        inQueX_.FreeTensor(buffer1_);
        inQueDY_.FreeTensor(buffer2_);
    }

    __aicore__ inline void CopyGammaIn()
    {
        LocalTensor<T_GAMMA> gammaLocal = inQueGamma_.AllocTensor<T_GAMMA>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParams{1, (uint32_t)(colVal_ * sizeof(T_GAMMA)), 0, 0, 0};
        DataCopyPadExtParams<T_GAMMA> padParams{true, 0, 0, 0};
        if constexpr (!is_same<T_GAMMA, float>::value) {
            DataCopyPad(gammaLocal[colValAlign_], gammaGm_, dataCopyParams, padParams);
        } else {
            DataCopyPad(gammaLocal, gammaGm_, dataCopyParams, padParams);
        }
#else
        if constexpr (!is_same<T_GAMMA, float>::value) {
            DataCopy(gammaLocal[colValAlign_], gammaGm_, colValAlign_);
        } else {
            DataCopy(gammaLocal, gammaGm_, colValAlign_);
        }
#endif
        inQueGamma_.EnQue(gammaLocal);
    }

    __aicore__ inline void SubProcess(
        uint32_t loopIdx, uint32_t calcLen, LocalTensor<float>& gammaLocal, LocalTensor<float>& dgammaLocal)
    {
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        if (colValAlign_ > SMALLD_THRESHOLD) {
#endif
            for (uint32_t iIner = 0; iIner < calcLen; iIner++) {
                CopyIn(loopIdx * ubFactor_ + iIner, 1);
                Compute(gammaLocal, dgammaLocal);
                CopyOut(loopIdx * ubFactor_ + iIner, 1);
            }
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        } else {
            CopyIn(loopIdx * ubFactor_, calcLen);
            ComputeSmallD(calcLen, gammaLocal, dgammaLocal);
            CopyOut(loopIdx * ubFactor_, calcLen);
        }
#endif
    }

    __aicore__ inline void CopyIn(uint32_t rowIdx, uint32_t calcLen)
    {
        LocalTensor<float> rstd = inQueRstd_.AllocTensor<float>();
        LocalTensor<T_DY> xLocal = inQueX_.AllocTensor<T_DY>();
        LocalTensor<T_DY> dy = inQueDY_.AllocTensor<T_DY>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParamsRstd{1, (uint32_t)(calcLen * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParamsRstd{true, 0, 0, 0};

        DataCopyExtParams dataCopyParams{(uint16_t)calcLen, (uint32_t)(colVal_ * sizeof(T_DY)), 0, 0, 0};
        DataCopyPadExtParams<T_DY> padParams{true, 0, 0, 0};
        DataCopyPad(rstd, rstdGm_[rowIdx], dataCopyParamsRstd, padParamsRstd);
        if constexpr (!is_same<T_DY, float>::value) {
            DataCopyPad(xLocal[calcLen * colValAlign_], xGm_[rowIdx * colVal_], dataCopyParams, padParams);
        } else {
            DataCopyPad(xLocal, xGm_[rowIdx * colVal_], dataCopyParams, padParams);
        }
        if constexpr (!is_same<T_DY, float>::value) {
            DataCopyPad(dy[calcLen * colValAlign_], dyGm_[rowIdx * colVal_], dataCopyParams, padParams);
        } else {
            DataCopyPad(dy, dyGm_[rowIdx * colVal_], dataCopyParams, padParams);
        }
#else
        uint32_t calcLenAlign = ROUND_UP(calcLen, alignLen_);
        DataCopy(rstd, rstdGm_[rowIdx], calcLenAlign);
        if constexpr (!is_same<T_DY, float>::value) {
            DataCopy(xLocal[calcLen * colValAlign_], xGm_[rowIdx * colVal_], colValAlign_);
        } else {
            DataCopy(xLocal, xGm_[rowIdx * colVal_], colValAlign_);
        }
        if constexpr (!is_same<T_DY, float>::value) {
            DataCopy(dy[calcLen * colValAlign_], dyGm_[rowIdx * colVal_], colValAlign_);
        } else {
            DataCopy(dy, dyGm_[rowIdx * colVal_], colValAlign_);
        }
#endif

        inQueRstd_.EnQue(rstd);
        inQueX_.EnQue(xLocal);
        inQueDY_.EnQue(dy);
    }

    __aicore__ inline void CopyOut(uint32_t rowIdx, uint32_t calcLen)
    {
        LocalTensor<T_DY> dx = outQueDX_.DeQue<T_DY>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParams{(uint16_t)calcLen, (uint32_t)(colVal_ * sizeof(T_DY)), 0, 0, 0};
        DataCopyPad(dxGm_[rowIdx * colVal_], dx, dataCopyParams);
#else
        DataCopyCustom<T_DY>(dxGm_, dx, rowIdx * colVal_, 0, colVal_);
#endif
        outQueDX_.FreeTensor(dx);
    }

    __aicore__ inline void CopyDgammaOut()
    {
        LocalTensor<float> dgammaOut = outQueDgamma_.DeQue<float>();
        SetAtomicAdd<float>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParams{1, (uint32_t)(colVal_ * sizeof(float)), 0, 0, 0};
        DataCopyPad(dgammaGm_, dgammaOut, dataCopyParams);
#else
        DataCopy(dgammaGm_, dgammaOut, ROUND_UP(colVal_, ALIGN_32));
#endif
        SetAtomicNone();
        outQueDgamma_.FreeTensor(dgammaOut);
    }

    __aicore__ inline void CopyDgammaOutWorkspace()
    {
        LocalTensor<float> dgammaOut = outQueDgamma_.DeQue<float>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParams{(uint16_t)1, (uint32_t)(colVal_ * sizeof(float)), 0, 0, 0};
        DataCopyPad(workspaceGm_, dgammaOut, dataCopyParams);
#else
        uint32_t colValAlign = (colVal_ / ALIGN_32) * ALIGN_32;
        uint32_t colValTail = colVal_ % ALIGN_32;
        DataCopy(workspaceGm_, dgammaOut, colValAlign);
        if (colValTail != 0) {
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            for (uint32_t i = 0; i < ALIGN_32; i++) {
                float tensorValue = dgammaOut.GetValue(colVal_ - ALIGN_32 + i);
                dgammaOut.SetValue(i, tensorValue);
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(workspaceGm_[colVal_ - ALIGN_32], dgammaOut, ALIGN_32);
        }
#endif
        outQueDgamma_.FreeTensor(dgammaOut);
    }

    __aicore__ inline void AddDgamma()
    {
        if (GetBlockIdx() != 0) {
            return;
        }
        LocalTensor<float> dgammaLocal = outQueDgamma_.AllocTensor<float>();
        Duplicate(dgammaLocal, 0.0f, colVal_);
        PipeBarrier<PIPE_V>();
        DataCopyExtParams dataCopyParams{(uint16_t)1, (uint32_t)(colVal_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, 0, 0};
        for (uint32_t blockidx = 0; blockidx < blockDim_; blockidx++) {
            LocalTensor<float> dgammaPart = inQueGamma_.AllocTensor<float>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
            DataCopyPad(dgammaPart, workspaceGm_[blockidx * colVal_], dataCopyParams, padParams);
#else
            DataCopy(dgammaPart, workspaceGm_[blockidx * colVal_], colValAlign_);
#endif
            inQueGamma_.EnQue(dgammaPart);
            LocalTensor<float> dgammaPartLocal = inQueGamma_.DeQue<float>();
            Add(dgammaLocal, dgammaLocal, dgammaPartLocal, colValAlign_);
            PipeBarrier<PIPE_V>();
            inQueGamma_.FreeTensor(dgammaPartLocal);
        }
        outQueDgamma_.EnQue(dgammaLocal);
        LocalTensor<float> dgammaOut = outQueDgamma_.DeQue<float>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParamsOut{1, (uint32_t)(colVal_ * sizeof(float)), 0, 0, 0};
        DataCopyPad(dgammaGm_, dgammaOut, dataCopyParamsOut);
#else
        DataCopy(dgammaGm_, dgammaOut, colValAlign_);
#endif
        outQueDgamma_.FreeTensor(dgammaOut);
    }

    __aicore__ inline void Compute(LocalTensor<float>& gammaLocal, LocalTensor<float>& dgammaLocal)
    {
        LocalTensor<float> rstdLocal = inQueRstd_.DeQue<float>();
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float rstd_value = rstdLocal.GetValue(0);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        inQueRstd_.FreeTensor(rstdLocal);
        LocalTensor<float> xLocal = inQueX_.DeQue<float>();
        Cast2FloatIf<T_DY>(xLocal, colValAlign_, colValAlign_);
        WaitFlag<HardEvent::S_V>(eventSV);
        Muls(xLocal, xLocal, rstd_value, colValAlign_);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> dyLocal = inQueDY_.DeQue<float>();
        Cast2FloatIf<T_DY>(dyLocal, colValAlign_, colValAlign_);
        LocalTensor<float> dxLocal = outQueDX_.AllocTensor<float>();
        Mul(dxLocal, dyLocal, xLocal, colValAlign_);
        PipeBarrier<PIPE_V>();
        Add(dgammaLocal, dgammaLocal, dxLocal, colValAlign_);
        Mul(dyLocal, dyLocal, gammaLocal, colValAlign_);
        PipeBarrier<PIPE_V>();
        Mul(dxLocal, dyLocal, xLocal, colValAlign_);
        PipeBarrier<PIPE_V>();
        float sumValue = ReduceSumHalfInterval(dxLocal, colVal_);
        float meanValue = sumValue * avgFactor_;
        event_t eventSV2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV2);
        WaitFlag<HardEvent::S_V>(eventSV2);
        Muls(dxLocal, xLocal, meanValue, colValAlign_);
        PipeBarrier<PIPE_V>();
        Sub(dxLocal, dyLocal, dxLocal, colValAlign_);
        PipeBarrier<PIPE_V>();
        Muls(dxLocal, dxLocal, rstd_value, colValAlign_);
        PipeBarrier<PIPE_V>();
        if constexpr (is_same<T_DY, half>::value) {
            LocalTensor<T_DY> dxLocalB16 = dxLocal.ReinterpretCast<T_DY>();
            Cast(dxLocalB16, dxLocal, RoundMode::CAST_NONE, colValAlign_);
            PipeBarrier<PIPE_V>();
        }
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        else if constexpr (is_same<T_DY, bfloat16_t>::value) {
            LocalTensor<T_DY> dxLocalB16 = dxLocal.ReinterpretCast<T_DY>();
            Cast(dxLocalB16, dxLocal, RoundMode::CAST_RINT, colValAlign_);
            PipeBarrier<PIPE_V>();
        }
#endif
        inQueX_.FreeTensor(xLocal);
        inQueDY_.FreeTensor(dyLocal);
        outQueDX_.EnQue(dxLocal);
    }

    __aicore__ inline void ComputeSmallD(
        uint32_t calcLen, LocalTensor<float>& gammaLocal, LocalTensor<float>& dgammaLocal)
    {
        uint32_t elementNum = colValAlign_ * calcLen;

        LocalTensor<float> tmp_reduce_buf = tmpBuf_.Get<float>();
        LocalTensor<float> rstdLocal = inQueRstd_.DeQue<float>();
        LocalTensor<float> dxLocal = outQueDX_.AllocTensor<float>();
        // y = x * rstd
        const uint32_t srcN1Shape[2] = {calcLen, 1};
        const uint32_t dstNDShape[2] = {calcLen, colValAlign_};
        auto sharedTmp = tmp_reduce_buf.ReinterpretCast<uint8_t>();
        BroadCast<float, DIM_NUM, DIM_D>(dxLocal, rstdLocal, dstNDShape, srcN1Shape, sharedTmp);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> xLocal = inQueX_.DeQue<float>();
        Cast2FloatIf<T_DY>(xLocal, elementNum, elementNum);

        Mul(xLocal, xLocal, dxLocal, elementNum); // x save x*rstd
        PipeBarrier<PIPE_V>();

        LocalTensor<float> dyLocal = inQueDY_.DeQue<float>();
        Cast2FloatIf<T_DY>(dyLocal, elementNum, elementNum);
        // dg=sum(dy * (x * rstd), dim=0)
        Mul(dxLocal, dyLocal, xLocal, elementNum);
        PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < calcLen; i++) {
            Add(dgammaLocal, dxLocal[i * colValAlign_], dgammaLocal, colValAlign_);
            PipeBarrier<PIPE_V>();
        }

        // broadcast gamma
        const uint32_t src1DShape[2] = {1, colValAlign_};
        BroadCast<float, DIM_NUM, DIM_N>(dxLocal, gammaLocal, dstNDShape, src1DShape, sharedTmp); // x reuse gamma_nd
        PipeBarrier<PIPE_V>();
        // dy * gamma
        Mul(dyLocal, dyLocal, dxLocal, elementNum); // dy save dy*gamma
        PipeBarrier<PIPE_V>();
        Mul(dxLocal, dyLocal, xLocal, elementNum);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> tmpMeanLocal = tmpMeanBuf_.Get<float>();
        ReduceSumMultiN(tmpMeanLocal, dxLocal, tmp_reduce_buf, calcLen, colVal_, colValAlign_);
        PipeBarrier<PIPE_V>();
        Muls(tmpMeanLocal, tmpMeanLocal, avgFactor_, calcLen);
        PipeBarrier<PIPE_V>();
        BroadCast<float, DIM_NUM, DIM_D>(dxLocal, tmpMeanLocal, dstNDShape, srcN1Shape, sharedTmp);
        PipeBarrier<PIPE_V>();
        Mul(dxLocal, xLocal, dxLocal, elementNum);
        PipeBarrier<PIPE_V>();
        Sub(dxLocal, dyLocal, dxLocal, elementNum);
        PipeBarrier<PIPE_V>();
        BroadCast<float, DIM_NUM, DIM_D>(dyLocal, rstdLocal, dstNDShape, srcN1Shape, sharedTmp);
        PipeBarrier<PIPE_V>();
        inQueRstd_.FreeTensor(rstdLocal);
        Mul(dxLocal, dxLocal, dyLocal, elementNum);
        PipeBarrier<PIPE_V>();
        if constexpr (is_same<T_DY, half>::value) {
            LocalTensor<T_DY> dxLocalB16 = dxLocal.ReinterpretCast<T_DY>();
            Cast(dxLocalB16, dxLocal, RoundMode::CAST_NONE, elementNum);
            PipeBarrier<PIPE_V>();
        }
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        else if constexpr (is_same<T_DY, bfloat16_t>::value) {
            LocalTensor<T_DY> dxLocalB16 = dxLocal.ReinterpretCast<T_DY>();
            Cast(dxLocalB16, dxLocal, RoundMode::CAST_RINT, elementNum);
            PipeBarrier<PIPE_V>();
        }
#endif
        inQueX_.FreeTensor(xLocal);
        inQueDY_.FreeTensor(dyLocal);
        outQueDX_.EnQue(dxLocal);
    }

public:
    uint32_t rowVal_;
    uint32_t colVal_;
    uint32_t colValAlign_;
    float avgFactor_{1.0f};
    uint32_t coreCalcNum_;
    uint32_t coreCalcTail_;
    uint32_t blockFactor_;
    uint32_t blockDim_;
    uint32_t ubFactor_;
    uint32_t ubCalcNum;
    uint32_t ubCalcTail_;
    uint32_t ubCalcLoop_;
    uint32_t ubCalcTailNum_;
    uint32_t ubCalcTailTail_;
    uint32_t ubCalcTailLoop_;
    uint32_t dataType_;
    uint32_t alignLen_;
    uint32_t coreOffset_;
    uint32_t ubFactorAlign_;
    uint32_t rstdLen_;
    uint32_t bufferLenSize_;
    int32_t bufferNum_;
    uint32_t isDeterministic_{0};

    TPipe pipe;
    GlobalTensor<T_DY> dyGm_, dxGm_, xGm_;
    GlobalTensor<T_GAMMA> gammaGm_;
    GlobalTensor<float> dgammaGm_, rstdGm_, workspaceGm_;
    GlobalTensor<float> workspaceGmOri_;
    GlobalTensor<int32_t> syncTmpSpaceGm_;
    TQue<QuePosition::VECIN, 1> inQueDY_, inQueX_, inQueRstd_, inQueGamma_;
    TQue<QuePosition::VECOUT, 1> outQueDX_, outQueDgamma_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> tmpMeanBuf_;
    TBuf<TPosition::VECCALC> outZeroTmpBuf_;
    TBuf<TPosition::VECCALC> syncTmpBuf_;
};
#endif // _RMS_NORM_GRAD_SPLIT_N_HIGH_PRECISION_H_