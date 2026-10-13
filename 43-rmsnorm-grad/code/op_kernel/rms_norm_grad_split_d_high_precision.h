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
 * \file rms_norm_grad_split_d_high_precision.h
 * \brief
 */
#ifndef _RMS_NORM_GRAD_SPLIT_D_HIGH_PRECISION_H_
#define _RMS_NORM_GRAD_SPLIT_D_HIGH_PRECISION_H_
#include "rms_norm_grad_common.h"

using namespace RmsNormGrad;

template <typename T_DY, typename T_GAMMA>
class RmsNormGradSplitDHighPrecision {
public:
    __aicore__ inline RmsNormGradSplitDHighPrecision()
    {}

    __aicore__ inline void Init(
        GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma,
        const RmsNormGradTilingData* tiling, GM_ADDR usrWorkspace)
    {
        InitVar(tiling);
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ != 200)
        dgammaGm_.SetGlobalBuffer((__gm__ float*)dgamma, colVal_);
        if (isDeterministic_ == 0 && GetBlockIdx() == 0) {
            InitOutput<float>(dgammaGm_, colVal_, 0);
        }
        if (isDeterministic_) {
            workspaceGm_.SetGlobalBuffer((__gm__ float*)usrWorkspace + GetBlockIdx() * colValAlign_);
            InitOutput<float>(workspaceGm_, colVal_, 0);
            workspaceGmOri_.SetGlobalBuffer((__gm__ float*)usrWorkspace);
        }
        SyncAll();
#endif
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
        InitGammaFor310(gamma, dgamma, usrWorkspace);
#endif
        InitGmBuffer(dy, x, rstd, gamma, dx);
        InitUB();
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

        loopMainCol_ = colVal_ / ubFactor_;
        tailCol_ = colVal_ % ubFactor_;

        rowFactor_ = ROW_FACTOR_SPLIT_D;
        alignLen_ = dataType_ == FLOAT_DTYPE ? ALIGN_32 : ALIGN_16;

        colValAlign_ = (colVal_ + alignLen_ - 1) / alignLen_ * alignLen_;
        isDeterministic_ = tiling->fixed_output;
    }

    __aicore__ inline void InitGmBuffer(
        GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx)
    {
        if (GetBlockIdx() < blockDim_ - 1) {
            coreOffset_ = blockFactor_;
        } else {
            coreOffset_ = coreCalcTail_ > 0 ? coreCalcTail_ : blockFactor_;
        }
        coreOffsetStart_ = blockFactor_ * colVal_;
        coreOffsetLen_ = coreOffset_ * colVal_;
        dyGm_.SetGlobalBuffer((__gm__ T_DY*)dy + GetBlockIdx() * coreOffsetStart_, coreOffsetLen_);
        xGm_.SetGlobalBuffer((__gm__ T_DY*)x + GetBlockIdx() * coreOffsetStart_, coreOffsetLen_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * blockFactor_, coreOffset_);
        dxGm_.SetGlobalBuffer((__gm__ T_DY*)dx + GetBlockIdx() * coreOffsetStart_, coreOffsetLen_);
        gammaGm_.SetGlobalBuffer((__gm__ T_GAMMA*)gamma, colVal_);
    }

    __aicore__ inline void InitGammaFor310(GM_ADDR gamma, GM_ADDR dgamma, GM_ADDR usrWorkspace)
    {
        uint32_t syncLen = ALIGN_32 * GetBlockNum();
        dgammaGm_.SetGlobalBuffer((__gm__ float*)dgamma, colValAlign_);
        syncTmpSpaceGm_.SetGlobalBuffer((__gm__ int32_t*)usrWorkspace, syncLen);

        pipe.InitBuffer(outZeroTmpBuf_, ubFactor_ * sizeof(float));
        pipe.InitBuffer(syncTmpBuf_, syncLen * sizeof(int32_t));

        InitGmZero<int32_t>(syncTmpSpaceGm_, outZeroTmpBuf_, syncLen, (uint32_t)0);
        if (isDeterministic_ == 0) {
            if (GetBlockIdx() == 0) {
                InitDgammaOut(dgammaGm_);
            }
            LocalTensor<int32_t> workLocal = syncTmpBuf_.Get<int32_t>();
            SyncAll(syncTmpSpaceGm_, workLocal);
        } else if (isDeterministic_) {
            workspaceGm_.SetGlobalBuffer((__gm__ float*)usrWorkspace + syncLen + GetBlockIdx() * colVal_);
            if (GetBlockIdx() == 0) {
                InitDgammaOut(workspaceGm_);
            }
        }
    }

    __aicore__ inline void InitDgammaOut(GlobalTensor<float>& outGm)
    {
        for (uint32_t iOuter = 0; iOuter < loopMainCol_; iOuter++) {
            InitGmZero<float>(outGm, outZeroTmpBuf_, ubFactor_, iOuter * ubFactor_);
        }

        if (tailCol_ > 0) {
            InitGmZero<float>(outGm, outZeroTmpBuf_, tailCol_, loopMainCol_ * ubFactor_);
        }
    }

    __aicore__ inline void InitUB()
    {
        bufferLenSize_ = ubFactor_ * sizeof(float);
        pipe.InitBuffer(inQueDY_, BUFFER_NUM_DB, bufferLenSize_);
        pipe.InitBuffer(inQueX_, BUFFER_NUM_DB, bufferLenSize_);
        pipe.InitBuffer(inQueRstd_, 1, rowFactor_ * sizeof(float));
        pipe.InitBuffer(inQueGamma_, 1, bufferLenSize_);
        pipe.InitBuffer(outQueDX_, BUFFER_NUM_DB, bufferLenSize_);
        pipe.InitBuffer(outQueDgamma_, 1, bufferLenSize_);
        pipe.InitBuffer(meanBuf_, rowFactor_ * sizeof(float));
        pipe.InitBuffer(meanTmpBuf_, rowFactor_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (coreCalcTail_ == 0) {
            ProcessMain(blockFactor_);
        } else {
            if (GetBlockIdx() < blockDim_ - 1) {
                ProcessMain(blockFactor_);
            } else {
                ProcessMain(coreCalcTail_);
            }
        }
    }

    __aicore__ inline void ProcessMain(uint32_t loop_len)
    {
        uint32_t loopMainOuter = loop_len / rowFactor_;
        uint32_t tailOuter = loop_len % rowFactor_;
        for (uint32_t iOuter = 0; iOuter < loopMainOuter; iOuter++) {
            SubProcess(iOuter, rowFactor_);
        }
        if (tailOuter > 0) {
            SubProcess(loopMainOuter, tailOuter);
        }
        if (isDeterministic_) {
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
            LocalTensor<int32_t> workLocal = syncTmpBuf_.Get<int32_t>();
            SyncAll(syncTmpSpaceGm_, workLocal);
            if (GetBlockIdx() != 0) {
                return;
            }
            for (uint32_t j = 0; j < loopMainCol_; j++) {
                AddDgamma(j, ubFactor_);
            }
            if (tailCol_ > 0) {
                AddDgamma(loopMainCol_, tailCol_);
            }
#else
            SyncAll();
            LocalTensor<float> buffer1_ = inQueX_.AllocTensor<float>();
            LocalTensor<float> buffer2_ = inQueDY_.AllocTensor<float>();
            deterministic_struct deterministicStruct = {buffer1_, buffer2_, workspaceGmOri_, dgammaGm_};
            FinalProcessDeterministic(colValAlign_, blockDim_, colVal_, deterministicStruct);
            inQueX_.FreeTensor(buffer1_);
            inQueDY_.FreeTensor(buffer2_);
#endif
        }
    }

    __aicore__ inline void SubProcess(uint32_t iOuter, uint32_t calcRow)
    {
        // CopyRstd
        CopyRstdIn(iOuter, calcRow);
        LocalTensor<float> rstdLocal = inQueRstd_.DeQue<float>();
        LocalTensor<float> meanLocal = meanBuf_.Get<float>();
        Duplicate(meanLocal, 0.0f, calcRow);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < loopMainCol_; j++) {
            loopColProcessBeforeReduce(iOuter, j, calcRow, ubFactor_, rstdLocal);
        }
        if (tailCol_ > 0) {
            loopColProcessBeforeReduce(iOuter, loopMainCol_, calcRow, tailCol_, rstdLocal);
        }
        Muls(meanLocal, meanLocal, avgFactor_, calcRow);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < loopMainCol_; j++) {
            loopColProcessAfterReduce(iOuter, j, calcRow, ubFactor_, rstdLocal);
        }
        if (tailCol_ > 0) {
            loopColProcessAfterReduce(iOuter, loopMainCol_, calcRow, tailCol_, rstdLocal);
        }
        inQueRstd_.FreeTensor(rstdLocal);
    }

    __aicore__ inline void loopColProcessBeforeReduce(
        uint32_t iOuter, uint32_t j, uint32_t calcRow, uint32_t calcCol, LocalTensor<float>& rstdLocal)
    {
        CopyGammaIn(j, calcCol);
        LocalTensor<float> gammaLocal = inQueGamma_.DeQue<float>();
        Cast2FloatIf<T_GAMMA>(gammaLocal, ubFactor_, calcCol);
        LocalTensor<float> dgamma = outQueDgamma_.AllocTensor<float>();
        Duplicate(dgamma, 0.0f, calcCol);
        PipeBarrier<PIPE_V>();
        for (uint32_t iInner = 0; iInner < calcRow; iInner++) {
            CopyIn(iOuter * rowFactor_ + iInner, j, calcCol);
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            float rstdValue = rstdLocal.GetValue(iInner);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            ComputeFormer(iInner, rstdValue, calcCol, gammaLocal, dgamma);
        }
        inQueGamma_.FreeTensor(gammaLocal);
        outQueDgamma_.EnQue(dgamma);
        if (isDeterministic_) {
            CopyDgammaOut(j, calcCol, workspaceGm_);
        } else {
            CopyDgammaOut(j, calcCol, dgammaGm_);
        }
        LocalTensor<float> meanTmpLocal = meanTmpBuf_.Get<float>();
        LocalTensor<float> meanLocal = meanBuf_.Get<float>();
        Add(meanLocal, meanLocal, meanTmpLocal, calcRow);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void loopColProcessAfterReduce(
        uint32_t iOuter, uint32_t j, uint32_t calcRow, uint32_t calcCol, LocalTensor<float>& rstdLocal)
    {
        CopyGammaIn(j, calcCol);
        LocalTensor<float> gammaLocal = inQueGamma_.DeQue<float>();
        Cast2FloatIf<T_GAMMA>(gammaLocal, ubFactor_, calcCol);
        LocalTensor<float> meanLocal = meanBuf_.Get<float>();
        for (uint32_t iInner = 0; iInner < calcRow; iInner++) {
            CopyIn(iOuter * rowFactor_ + iInner, j, calcCol);
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            float rstdValue = rstdLocal.GetValue(iInner);
            float meanValue = meanLocal.GetValue(iInner);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            ComputeLatter(rstdValue, meanValue, calcCol, gammaLocal);
            CopyDxOut(iOuter * rowFactor_ + iInner, j, calcCol);
        }
        inQueGamma_.FreeTensor(gammaLocal);
    }

    __aicore__ inline void CopyRstdIn(uint32_t iOuter, uint32_t calcRow)
    {
        LocalTensor<float> rstdLocal = inQueRstd_.AllocTensor<float>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParamsRstd{1, (uint32_t)(calcRow * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, 0, 0};
        DataCopyPad(rstdLocal, rstdGm_[iOuter * rowFactor_], dataCopyParamsRstd, padParams);
#else
        uint32_t calcRow_align = ROUND_UP(calcRow, alignLen_);
        DataCopy(rstdLocal, rstdGm_[iOuter * rowFactor_], calcRow_align);
#endif
        inQueRstd_.EnQue(rstdLocal);
    }

    __aicore__ inline void CopyGammaIn(uint32_t dIdx, uint32_t calcLen)
    {
        LocalTensor<T_GAMMA> gammaLocal = inQueGamma_.AllocTensor<T_GAMMA>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParams{1, (uint32_t)(calcLen * sizeof(T_GAMMA)), 0, 0, 0};
        DataCopyPadExtParams<T_GAMMA> padParams{false, 0, 0, 0};
        if constexpr (!is_same<T_GAMMA, float>::value) {
            DataCopyPad(gammaLocal[ubFactor_], gammaGm_[dIdx * ubFactor_], dataCopyParams, padParams);
        } else {
            DataCopyPad(gammaLocal, gammaGm_[dIdx * ubFactor_], dataCopyParams, padParams);
        }
#else
        uint32_t calcLen_align = ROUND_UP(calcLen, alignLen_);
        if constexpr (!is_same<T_GAMMA, float>::value) {
            DataCopy(gammaLocal[ubFactor_], gammaGm_[dIdx * ubFactor_], calcLen_align);
        } else {
            DataCopy(gammaLocal, gammaGm_[dIdx * ubFactor_], calcLen_align);
        }
#endif

        inQueGamma_.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyDgammaOut(uint32_t dIdx, uint32_t calcLen, GlobalTensor<float>& outGm)
    {
        LocalTensor<float> dgamma_out = outQueDgamma_.DeQue<float>();
        SetAtomicAdd<float>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParams{1, (uint32_t)(calcLen * sizeof(float)), 0, 0, 0};
        DataCopyPad(outGm[dIdx * ubFactor_], dgamma_out, dataCopyParams);
#else
        if (calcLen < ALIGN_32) {
            for (uint32_t i = 0; i < ALIGN_32; i++) {
                if (i >= calcLen) {
                    dgamma_out.SetValue(i, 0.0f);
                }
            }
            DataCopy(outGm[dIdx * ubFactor_], dgamma_out, ALIGN_32);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        } else {
            uint32_t calcLenAlign = (calcLen / ALIGN_32) * ALIGN_32;
            uint32_t calcLenTail = calcLen - calcLenAlign;
            DataCopy(outGm[dIdx * ubFactor_], dgamma_out, calcLenAlign);
            if (calcLenTail > 0) {
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                for (uint32_t i = 0; i < ALIGN_32; i++) {
                    if (i < (ALIGN_32 - calcLenTail)) {
                        dgamma_out.SetValue(i, 0.0f);
                    } else {
                        uint32_t tailOffset = calcLenAlign + i - (ALIGN_32 - calcLenTail);
                        dgamma_out.SetValue(i, dgamma_out.GetValue(tailOffset));
                    }
                }
                DataCopy(outGm[dIdx * ubFactor_ + calcLen - ALIGN_32], dgamma_out, ALIGN_32);
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            }
        }
#endif
        SetAtomicNone();
        outQueDgamma_.FreeTensor(dgamma_out);
    }

    __aicore__ inline void CopyIn(uint32_t nIdx, uint32_t dIdx, uint32_t calcLen)
    {
        // x dy, rstd
        LocalTensor<T_DY> xLocal = inQueX_.AllocTensor<T_DY>();
        LocalTensor<T_DY> dyLocal = inQueDY_.AllocTensor<T_DY>();
        uint32_t gmOffset = nIdx * colVal_ + dIdx * ubFactor_;
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParams{1, (uint32_t)(calcLen * sizeof(T_DY)), 0, 0, 0};
        DataCopyPadExtParams<T_DY> padParams{true, 0, 0, 0};
        if constexpr (!is_same<T_DY, float>::value) {
            DataCopyPad(xLocal[ubFactor_], xGm_[gmOffset], dataCopyParams, padParams);
        } else {
            DataCopyPad(xLocal, xGm_[gmOffset], dataCopyParams, padParams);
        }
        if constexpr (!is_same<T_DY, float>::value) {
            DataCopyPad(dyLocal[ubFactor_], dyGm_[gmOffset], dataCopyParams, padParams);
        } else {
            DataCopyPad(dyLocal, dyGm_[gmOffset], dataCopyParams, padParams);
        }
#else
        uint32_t calcLen_align = ROUND_UP(calcLen, alignLen_);
        if constexpr (!is_same<T_DY, float>::value) {
            DataCopy(xLocal[ubFactor_], xGm_[gmOffset], calcLen_align);
        } else {
            DataCopy(xLocal, xGm_[gmOffset], calcLen_align);
        }
        if constexpr (!is_same<T_DY, float>::value) {
            DataCopy(dyLocal[ubFactor_], dyGm_[gmOffset], calcLen_align);
        } else {
            DataCopy(dyLocal, dyGm_[gmOffset], calcLen_align);
        }
#endif
        inQueX_.EnQue(xLocal);
        inQueDY_.EnQue(dyLocal);
    }

    __aicore__ inline void ComputeFormer(
        uint32_t iInner, float rstdValue, uint32_t calcLen, LocalTensor<float>& gammaLocal, LocalTensor<float>& dgamma)
    {
        LocalTensor<float> xLocal = inQueX_.DeQue<float>();
        Cast2FloatIf<T_DY>(xLocal, ubFactor_, calcLen);
        Muls(xLocal, xLocal, rstdValue, calcLen); // x*rstd
        PipeBarrier<PIPE_V>();

        LocalTensor<float> dyLocal = inQueDY_.DeQue<float>();
        Cast2FloatIf<T_DY>(dyLocal, ubFactor_, calcLen);
        Mul(xLocal, dyLocal, xLocal, calcLen); // dy*x*rstd
        PipeBarrier<PIPE_V>();
        Add(dgamma, dgamma, xLocal, calcLen);
        PipeBarrier<PIPE_V>();
        Mul(xLocal, xLocal, gammaLocal, calcLen); // dy*gamma*x*rstd
        PipeBarrier<PIPE_V>();
        float sumValue = ReduceSumHalfInterval(xLocal, calcLen);
        LocalTensor<float> meanTmpLocal = meanTmpBuf_.Get<float>();
        meanTmpLocal.SetValue(iInner, sumValue);
        event_t eventSV2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV2);
        WaitFlag<HardEvent::S_V>(eventSV2);
        inQueX_.FreeTensor(xLocal);
        inQueDY_.FreeTensor(dyLocal);
    }

    __aicore__ inline void ComputeLatter(
        float rstdValue, float meanValue, uint32_t calcLen, LocalTensor<float>& gammaLocal)
    {
        LocalTensor<float> xLocal = inQueX_.DeQue<float>();
        Cast2FloatIf<T_DY>(xLocal, ubFactor_, calcLen);
        Muls(xLocal, xLocal, rstdValue, calcLen); // x*rstd
        PipeBarrier<PIPE_V>();
        Muls(xLocal, xLocal, meanValue, calcLen); // x*rstd*mean
        LocalTensor<float> dyLocal = inQueDY_.DeQue<float>();
        Cast2FloatIf<T_DY>(dyLocal, ubFactor_, calcLen);
        Mul(dyLocal, dyLocal, gammaLocal, calcLen); // dy*gamma
        PipeBarrier<PIPE_V>();
        LocalTensor<float> dxLocal = outQueDX_.AllocTensor<float>();
        Sub(dxLocal, dyLocal, xLocal, calcLen);
        PipeBarrier<PIPE_V>();
        Muls(dxLocal, dxLocal, rstdValue, calcLen);
        PipeBarrier<PIPE_V>();
        if constexpr (is_same<T_DY, half>::value) {
            LocalTensor<T_DY> dxLocalB16 = dxLocal.ReinterpretCast<T_DY>();
            Cast(dxLocalB16, dxLocal, RoundMode::CAST_NONE, calcLen);
            PipeBarrier<PIPE_V>();
        }
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        else if constexpr (is_same<T_DY, bfloat16_t>::value) {
            LocalTensor<T_DY> dxLocalB16 = dxLocal.ReinterpretCast<T_DY>();
            Cast(dxLocalB16, dxLocal, RoundMode::CAST_RINT, calcLen);
            PipeBarrier<PIPE_V>();
        }
#endif
        inQueX_.FreeTensor(xLocal);
        inQueDY_.FreeTensor(dyLocal);
        outQueDX_.EnQue(dxLocal);
    }

    __aicore__ inline void CopyDxOut(uint32_t nIdx, uint32_t dIdx, uint32_t calcLen)
    {
        LocalTensor<T_DY> dx = outQueDX_.DeQue<T_DY>();
        uint32_t gmOffset = nIdx * colVal_ + dIdx * ubFactor_;
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParams{1, (uint32_t)(calcLen * sizeof(T_DY)), 0, 0, 0};
        DataCopyPad(dxGm_[gmOffset], dx, dataCopyParams);
#else
        uint32_t calcLenAlign32 = (calcLen / alignLen_) * alignLen_;
        if (calcLenAlign32 > 0) {
            DataCopy(dxGm_[gmOffset], dx, calcLenAlign32);
        }
        uint32_t calcLenTail32 = calcLen % alignLen_;

        if (calcLenTail32 > 0) {
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            for (uint32_t i = 0; i < calcLenTail32; i++) {
                dxGm_.SetValue(gmOffset + calcLenAlign32 + i, dx.GetValue(calcLenAlign32 + i));
            }
            DataCacheCleanAndInvalid<T_DY, CacheLine::ENTIRE_DATA_CACHE>(dxGm_);
            PipeBarrier<PIPE_ALL>();
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        }
#endif
        outQueDX_.FreeTensor(dx);
    }

    __aicore__ inline void AddDgamma(uint32_t j, uint32_t calcCol)
    {
        uint32_t calcCol_align = ROUND_UP(calcCol, ALIGN_32);
        LocalTensor<float> dgamma = outQueDgamma_.AllocTensor<float>();
        Duplicate(dgamma, 0.0f, calcCol_align);
        PipeBarrier<PIPE_V>();
        DataCopyExtParams dataCopyParams{1, (uint32_t)(calcCol * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, 0, 0};
        for (uint32_t blockidx = 0; blockidx < blockDim_; blockidx++) {
            LocalTensor<float> dgammaPart = inQueGamma_.AllocTensor<float>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
            DataCopyPad(dgammaPart, workspaceGm_[blockidx * colVal_ + j * ubFactor_], dataCopyParams, padParams);
#else
            DataCopy(dgammaPart, workspaceGm_[blockidx * colVal_ + j * ubFactor_], calcCol_align);
#endif
            inQueGamma_.EnQue(dgammaPart);
            LocalTensor<float> dgammaPartLocal = inQueGamma_.DeQue<float>();
            Add(dgamma, dgamma, dgammaPartLocal, calcCol);
            PipeBarrier<PIPE_V>();
            inQueGamma_.FreeTensor(dgammaPartLocal);
        }
        outQueDgamma_.EnQue(dgamma);
        LocalTensor<float> dgammaOut = outQueDgamma_.DeQue<float>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParamsOut{1, (uint32_t)(calcCol * sizeof(float)), 0, 0, 0};
        DataCopyPad(dgammaGm_[j * ubFactor_], dgammaOut, dataCopyParamsOut);
#else
        DataCopy(dgammaGm_[j * ubFactor_], dgammaOut, calcCol_align);
#endif
        outQueDgamma_.FreeTensor(dgammaOut);
    }

public:
    uint32_t rowVal_{0};
    uint32_t colVal_{0};
    uint32_t colValAlign_{0};
    float avgFactor_{1.0f};
    uint32_t coreCalcNum_{0};
    uint32_t coreCalcTail_{0};
    uint32_t blockFactor_{0};
    uint32_t blockDim_{0};
    uint32_t ubFactor_{0};

    uint32_t loopMainCol_{0};
    uint32_t tailCol_{0};

    uint32_t dataType_{0};
    uint32_t alignLen_{0};
    uint32_t coreOffset_{0};

    uint32_t rowFactor_{0};
    uint32_t bufferLenSize_{0};
    uint32_t coreOffsetStart_{0};
    uint32_t coreOffsetLen_{0};
    uint32_t isDeterministic_{0};

    TPipe pipe;
    GlobalTensor<T_DY> dyGm_;
    GlobalTensor<T_GAMMA> gammaGm_;
    GlobalTensor<T_DY> dxGm_;
    GlobalTensor<T_DY> xGm_;
    GlobalTensor<float> workspaceGm_;
    GlobalTensor<float> workspaceGmOri_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<float> dgammaGm_;
    GlobalTensor<int32_t> syncTmpSpaceGm_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueDY_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueX_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueRstd_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueGamma_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueDX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueDgamma_;
    TBuf<TPosition::VECCALC> meanBuf_;
    TBuf<TPosition::VECCALC> meanTmpBuf_;
    TBuf<TPosition::VECCALC> outZeroTmpBuf_;
    TBuf<TPosition::VECCALC> syncTmpBuf_;
};
#endif // _RMS_NORM_GRAD_SPLIT_D_HIGH_PRECISION_H_