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
 * \file rms_norm.h
 * \brief rms norm h file
 */
#ifndef RMS_NORM_H_
#define RMS_NORM_H_
#include "rms_norm_base.h"

namespace RmsNorm {
using namespace AscendC;

template <typename T, typename T_GAMMA>
class KernelRmsNorm : KernelRmsNormBase<T, T_GAMMA> {
public:
    __aicore__ inline KernelRmsNorm()
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, const RMSNormTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "block dim2 can not be zero!");
        InitVar(tiling);

        blockIdx_ = GetBlockIdx();
        if (blockIdx_ < GetBlockNum() - 1) {
            this->row_work = block_factor;
        } else if (blockIdx_ == GetBlockNum() - 1) {
            this->row_work = num_row - (GetBlockNum() - 1) * block_factor;
        }
        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx_ * block_factor * num_col, row_work * num_col);
        gammaGm.SetGlobalBuffer((__gm__ T_GAMMA*)gamma, num_col);
        yGm.SetGlobalBuffer((__gm__ T*)y + blockIdx_ * block_factor * num_col, row_work * num_col);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockIdx_ * block_factor, block_factor);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
        InitRstdData();
#endif
        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, ub_factor * sizeof(T));
        pipe.InitBuffer(inQueueGamma, BUFFER_NUM, ub_factor * sizeof(T_GAMMA));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, ub_factor * sizeof(T));
        pipe.InitBuffer(outQueueRstd, BUFFER_NUM, row_factor * sizeof(float));

        if (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value || is_gemma == 1) {
            pipe.InitBuffer(x_fp32_buf, ub_factor * sizeof(float));
        }
        pipe.InitBuffer(sqx_buf, ub_factor * sizeof(float));
        pipe.InitBuffer(reduce_fp32_buf, NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void InitRstdData()
    {
        uint32_t row_factor_align = ROUND_UP(row_factor, NUM_PER_BLK_FP32);
        pipe.InitBuffer(outTmpZeroBuf, row_factor_align * sizeof(float));
        LocalTensor<float> temp_zero_tensor = outTmpZeroBuf.Get<float>();
        Duplicate(temp_zero_tensor, (float)0.0, row_factor_align);

        PipeBarrier<PIPE_ALL>();
        uint32_t i_o_max = CeilDiv(row_work, row_factor);
        uint32_t row_tail = row_work - (i_o_max - 1) * row_factor;
        for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
            DataCopy(rstdGm[i_o * row_factor], temp_zero_tensor, row_factor_align);
        }
        DataCopy(rstdGm[(i_o_max - 1) * row_factor], temp_zero_tensor, ROUND_UP(row_tail, NUM_PER_BLK_FP32));
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void InitVar(const RMSNormTilingData* tiling)
    {
        is_gemma = tiling->is_gemma;
        num_row = tiling->num_row;
        num_col = tiling->num_col;
        block_factor = tiling->block_factor;
        row_factor = tiling->row_factor;
        ub_factor = tiling->ub_factor;
        epsilon = tiling->epsilon;
        if (num_col != 0) {
            avg_factor = (float)1.0 / num_col;
        } else {
            avg_factor = 0;
        }

        num_row_align = ROUND_UP(num_row, NUM_PER_BLK_FP32);
    }
    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<T_GAMMA> gammaLocal = inQueueGamma.DeQue<T_GAMMA>();

        uint32_t i_o_max = CeilDiv(row_work, row_factor);
        uint32_t row_tail = row_work - (i_o_max - 1) * row_factor;

        for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
            SubProcess(i_o, row_factor, gammaLocal);
        }
        SubProcess(i_o_max - 1, row_tail, gammaLocal);
        inQueueGamma.FreeTensor(gammaLocal);
    }

    __aicore__ inline void SubProcess(uint32_t i_o, uint32_t calc_row_num, LocalTensor<T_GAMMA>& gammaLocal)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
        Duplicate(rstdLocal, (float)0.0, ROUND_UP(calc_row_num, NUM_PER_BLK_FP32));
#endif

        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            uint32_t gm_bias = (i_o * row_factor + i_i) * num_col;
            CopyIn(gm_bias);
            Compute(i_i, gammaLocal, rstdLocal);
            CopyOutY(gm_bias);
        }
        outQueueRstd.EnQue<float>(rstdLocal);
        CopyOutRstd(i_o, calc_row_num);
    }

private:
    __aicore__ inline void CopyIn(uint32_t gm_bias)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopyCustom<T>(xLocal, xGm[gm_bias], num_col);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T_GAMMA> gammaLocal = inQueueGamma.AllocTensor<T_GAMMA>();
        DataCopyCustom<T_GAMMA>(gammaLocal, gammaGm, num_col);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void ComputeMulGammaCast(LocalTensor<T_GAMMA> gammaLocal, uint32_t elementNum)
    {
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<float> sqx = sqx_buf.Get<float>();

        if constexpr (is_same<T, float>::value) {
            if (is_gemma == 1) {
                LocalTensor<float> gammaFp32 = x_fp32_buf.Get<float>();
                Adds(gammaFp32, gammaLocal, static_cast<float>(1.0), elementNum);
                PipeBarrier<PIPE_V>();
                Mul(yLocal, sqx, gammaFp32, elementNum);
            } else {
                Mul(yLocal, sqx, gammaLocal, elementNum);
            }
        } else {
            if constexpr (IS_MIX_DTYPE) {
                Mul(sqx, sqx, gammaLocal, elementNum);
            } else {
                LocalTensor<float> gammaFp32 = x_fp32_buf.Get<float>();
                Cast(gammaFp32, gammaLocal, RoundMode::CAST_NONE, elementNum);
                PipeBarrier<PIPE_V>();
                if (is_gemma == 1) {
                    Adds(gammaFp32, gammaFp32, static_cast<float>(1.0), elementNum);
                    PipeBarrier<PIPE_V>();
                }
                Mul(sqx, sqx, gammaFp32, elementNum);
            }
            PipeBarrier<PIPE_V>();
            if constexpr (is_same<T, half>::value) {
                Cast(yLocal, sqx, RoundMode::CAST_NONE, elementNum);
            } else {
                Cast(yLocal, sqx, RoundMode::CAST_RINT, elementNum);
            }
        }
        outQueueY.EnQue<T>(yLocal);
    }

    __aicore__ inline void Compute(
        uint32_t innerProgress, LocalTensor<T_GAMMA> gammaLocal, LocalTensor<float> rstdLocal)
    {
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<float> sqx = sqx_buf.Get<float>();
        LocalTensor<float> reduceBufLocal = reduce_fp32_buf.Get<float>();
        LocalTensor<float> xBufFp32;

        // 1. Cast x and Cal x^2
        if constexpr (is_same<T, float>::value) {
            Mul(sqx, xLocal, xLocal, num_col);
        } else {
            xBufFp32 = x_fp32_buf.Get<float>();
            Cast(xBufFp32, xLocal, RoundMode::CAST_NONE, num_col);
            PipeBarrier<PIPE_V>();
            inQueueX.FreeTensor(xLocal);
            Mul(sqx, xBufFp32, xBufFp32, num_col);
        }
        PipeBarrier<PIPE_V>();

        // 2. Rstd = 1 / sqrt(1 / reduceDim * reducesum(x^2) + eps)
        float reduceOut = ReduceSumHalfInterval(sqx, num_col);
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float rstdValue = 1 / sqrt(reduceOut * avg_factor + epsilon);
        rstdLocal.SetValue(innerProgress, rstdValue);
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);

        // 3. Y = x * rstd * gamma
        if constexpr (is_same<T, float>::value) { // fp32 use inQueueX store x
            Muls(sqx, xLocal, rstdValue, num_col);
            inQueueX.FreeTensor(xLocal);
        } else { // fp16/bf16 use xFp32Buf store x
            Muls(sqx, xBufFp32, rstdValue, num_col);
        }
        PipeBarrier<PIPE_V>();
        ComputeMulGammaCast(gammaLocal, num_col);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOutRstd(uint32_t outer_progress, uint32_t num)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
        uint32_t copyRstdNumAlgin32 = ROUND_UP(num, NUM_PER_BLK_FP32);
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        DataCopyCustom<float>(rstdGm[outer_progress * row_factor], rstdLocal, num);
#else
        SetAtomicAdd<float>();
        DataCopy(rstdGm[outer_progress * row_factor], rstdLocal, copyRstdNumAlgin32);
        SetAtomicNone();
#endif
        outQueueRstd.FreeTensor(rstdLocal);
    }

    __aicore__ inline void CopyOutY(uint32_t progress)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyCustom<T>(yGm[progress], yLocal, num_col);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGamma;
    // create queues for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueRstd;

    TBuf<TPosition::VECCALC> x_fp32_buf;
    TBuf<TPosition::VECCALC> sqx_buf;
    TBuf<TPosition::VECCALC> reduce_fp32_buf;
    TBuf<TPosition::VECCALC> outTmpZeroBuf;
    GlobalTensor<T> xGm;
    GlobalTensor<T_GAMMA> gammaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> rstdGm;

    uint32_t num_row;
    uint32_t num_col;
    uint32_t block_factor; // number of calculations rows on each core
    uint32_t row_factor;
    uint32_t ub_factor;
    uint32_t num_row_align;
    float epsilon;
    float avg_factor;
    int32_t blockIdx_;
    uint32_t row_work = 1;
    uint8_t is_gemma = 0;
};
} // namespace RmsNorm
#endif // RMS_NORM_H_