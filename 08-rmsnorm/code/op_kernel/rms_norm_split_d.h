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
 * \file rms_norm_split_d.h
 * \brief rms norm split d file
 */
#ifndef RMS_NORM_SPLIT_D_H_
#define RMS_NORM_SPLIT_D_H_
#include "rms_norm_base.h"

namespace RmsNorm {
using namespace AscendC;
constexpr uint32_t MULTI_8 = 8;

template <typename T, typename T_GAMMA>
class KernelRmsNormSplitD {
#define IS_X_FP32 (is_same<T, float>::value)
#define IS_GAMMA_FP32 (is_same<T_GAMMA, float>::value)
#define IS_MIX_DTYPE ((!IS_X_FP32) && IS_GAMMA_FP32)
public:
    __aicore__ inline KernelRmsNormSplitD()
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, const RMSNormTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        InitVar(tiling);

        blockIdx_ = GetBlockIdx();
        if (blockIdx_ < GetBlockNum() - 1) {
            this->row_work = block_factor;
        } else if (blockIdx_ == GetBlockNum() - 1) {
            this->row_work = num_row - (GetBlockNum() - 1) * block_factor;
        }
        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx_ * block_factor * num_col, row_work * num_col);
        yGm.SetGlobalBuffer((__gm__ T*)y + blockIdx_ * block_factor * num_col, row_work * num_col);
        gammaGm.SetGlobalBuffer((__gm__ T_GAMMA*)gamma, num_col);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockIdx_ * block_factor, block_factor);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
        InitRstdData();
#endif

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueGamma, BUFFER_NUM, ub_factor * sizeof(T_GAMMA));
        pipe.InitBuffer(inQueueX, BUFFER_NUM, ub_factor * sizeof(T));
        pipe.InitBuffer(outQueueRstd, BUFFER_NUM, row_factor * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, ub_factor * sizeof(T));

        if (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value || is_gemma == 1) {
            pipe.InitBuffer(x_fp32_buf, ub_factor * sizeof(float));
        }
        pipe.InitBuffer(sum_buf, row_factor * NUM_PER_BLK_FP32 * sizeof(float));
        pipe.InitBuffer(sqx_buf, ub_factor * sizeof(float));
        pipe.InitBuffer(reduce_fp32_buf, NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void InitRstdData()
    {
        uint32_t row_factor_align_v1 = ROUND_UP(row_factor, NUM_PER_BLK_FP32);
        pipe.InitBuffer(outTmpZeroBuf, row_factor_align_v1 * sizeof(float));
        LocalTensor<float> temp_zero_tensor = outTmpZeroBuf.Get<float>();
        Duplicate(temp_zero_tensor, (float)0.0, row_factor_align_v1);

        PipeBarrier<PIPE_ALL>();
        uint32_t i_o_max = CeilDiv(row_work, row_factor);
        uint32_t row_tail = row_work - (i_o_max - 1) * row_factor;
        for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
            DataCopy(rstdGm[i_o * row_factor], temp_zero_tensor, row_factor_align_v1);
        }
        DataCopy(rstdGm[(i_o_max - 1) * row_factor], temp_zero_tensor, ROUND_UP(row_tail, NUM_PER_BLK_FP32));
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void InitVar(const RMSNormTilingData* tiling)
    {
        is_gemma = tiling->is_gemma;
        num_row = tiling->num_row;
        num_col = tiling->num_col;
        num_col_align = tiling->num_col_align;
        block_factor = tiling->block_factor;
        ub_factor = tiling->ub_factor;
        row_factor = NUM_PER_REP_FP32;
        epsilon = tiling->epsilon;
        avg_factor = tiling->avg_factor;
        data_per_block = (BLOCK_SIZE / sizeof(T));
        reduce_mask = tiling->reduce_mask;
        left_num = tiling->left_num;
        last_reduce_mask = tiling->last_reduce_mask;
        last_left_num = tiling->last_left_num;
        num_row_align = ROUND_UP(num_row, NUM_PER_BLK_FP32);
    }

    __aicore__ inline void Process()
    {
        uint32_t iOuterMax = CeilDiv(row_work, row_factor);
        uint32_t rowTail = row_work - (iOuterMax - 1) * row_factor;
        uint32_t jMax = CeilDiv(num_col, (uint64_t)ub_factor);
        uint32_t jTail = num_col - (jMax - 1) * ub_factor;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 100
        for (uint32_t iOuter = 0; iOuter < iOuterMax - 1; iOuter++) {
            SubProcess(iOuter, row_factor, jMax, jTail);
        }
        SubProcess(iOuterMax - 1, rowTail, jMax, jTail);
#else
        uint32_t jFactor = ub_factor * row_factor;
        uint32_t jOuterMax = CeilDiv(num_col, (uint64_t)jFactor);
        uint32_t jOuterTail = num_col - (jOuterMax - 1) * jFactor;
        uint32_t jInnerMax = CeilDiv(jOuterTail, ub_factor);
        for (uint32_t iOuter = 0; iOuter < iOuterMax - 1; iOuter++) {
            SubProcess(iOuter, row_factor, jMax, jTail, jOuterMax, jInnerMax);
        }
        SubProcess(iOuterMax - 1, rowTail, jMax, jTail, jOuterMax, jInnerMax);
#endif
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 100
    __aicore__ inline void SubProcess(uint32_t i_o, uint32_t calc_row_num, uint32_t j_max, uint32_t col_tail)
    {
        LocalTensor<float> sumLocal = sum_buf.Get<float>();
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        Duplicate(rstdLocal, (float)0.0, ROUND_UP(calc_row_num, NUM_PER_BLK_FP32));
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < j_max - 1; j++) {
            ComputeFormer(i_o, calc_row_num, j, rstdLocal, sumLocal, ub_factor, left_num, reduce_mask);
        }
        // do tail
        ComputeFormer(i_o, calc_row_num, j_max - 1, rstdLocal, sumLocal, col_tail, last_left_num, last_reduce_mask);
        ComputeRstd(rstdLocal, calc_row_num);
        ComputeLatter(i_o, calc_row_num, j_max - 1, rstdLocal, col_tail);
        for (uint32_t j = 0; j < j_max - 1; j++) {
            ComputeLatter(i_o, calc_row_num, j, rstdLocal, ub_factor);
        }
        outQueueRstd.EnQue<float>(rstdLocal);
        CopyOutRstd(i_o, calc_row_num);
    }

#else
    __aicore__ inline void SubProcess(
        uint32_t iOuter, uint32_t calcRowNum, uint32_t jMax, uint32_t jTail, uint32_t jOuterMax, uint32_t jInnerMax)
    {
        LocalTensor<float> sumLocal = sum_buf.Get<float>();
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        Duplicate(rstdLocal, (float)0.0, ROUND_UP(calcRowNum, NUM_PER_BLK_FP32));
        PipeBarrier<PIPE_V>();
        ComputeFormerPrecision(iOuter, calcRowNum, jTail, jOuterMax, jInnerMax, rstdLocal);
        ComputeRstd(rstdLocal, calcRowNum);
        ComputeLatter(iOuter, calcRowNum, jMax - 1, rstdLocal, jTail);
        for (uint32_t j = 0; j < jMax - 1; j++) {
            ComputeLatter(iOuter, calcRowNum, j, rstdLocal, ub_factor);
        }
        outQueueRstd.EnQue<float>(rstdLocal);
        CopyOutRstd(iOuter, calcRowNum);
    }
#endif

private:
    __aicore__ inline void CopyIn(uint32_t i_idx, uint32_t j_idx, uint32_t num)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopyCustom<T>(xLocal, xGm[i_idx * num_col + j_idx * ub_factor], num);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void ComputeFormerPrecision(
        uint32_t iOuter, uint32_t calcRowNum, uint32_t jTail, uint32_t jOuterMax, uint32_t jInnerTail,
        LocalTensor<float>& rstdLocal)
    {
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        LocalTensor<float> sumLocal = sum_buf.Get<float>();
        const uint32_t jInnerMax = row_factor;

        for (uint32_t iInner = 0; iInner < calcRowNum; iInner++) {
            float reduceOut = 0;
            uint32_t rowIndex = iOuter * row_factor + iInner;

            // 1. jOuter main loop
            for (uint32_t jOuterIdx = 0; jOuterIdx < jOuterMax - 1; jOuterIdx++) {
                for (uint32_t jInnerIdx = 0; jInnerIdx < jInnerMax; jInnerIdx++) {
                    CopyIn(rowIndex, jOuterIdx * jInnerMax + jInnerIdx, ub_factor);
                    ComputeSumPrecision(sumLocal[jInnerIdx * NUM_PER_BLK_FP32], ub_factor);
                }
                SetFlag<HardEvent::V_S>(eventVS);
                WaitFlag<HardEvent::V_S>(eventVS);
                for (uint32_t jInnerIdx = 0; jInnerIdx < jInnerMax; jInnerIdx++) {
                    reduceOut += sumLocal.GetValue(jInnerIdx * NUM_PER_BLK_FP32);
                }
                SetFlag<HardEvent::S_V>(eventSV);
                WaitFlag<HardEvent::S_V>(eventSV);
            }
            // 2. jInner main loop
            uint32_t outerOffsetIdx = (jOuterMax - 1) * jInnerMax;
            for (uint32_t jInnerIdx = 0; jInnerIdx < jInnerTail - 1; jInnerIdx++) {
                CopyIn(rowIndex, outerOffsetIdx + jInnerIdx, ub_factor);
                ComputeSumPrecision(sumLocal[jInnerIdx * NUM_PER_BLK_FP32], ub_factor);
            }
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            for (uint32_t jInnerIdx = 0; jInnerIdx < jInnerTail - 1; jInnerIdx++) {
                reduceOut += sumLocal.GetValue(jInnerIdx * NUM_PER_BLK_FP32);
            }
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            // 3. jInner tail loop
            CopyIn(rowIndex, outerOffsetIdx + jInnerTail - 1, jTail);
            ComputeSumPrecision(sumLocal[0], jTail);
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            reduceOut += sumLocal.GetValue(0);
            rstdLocal.SetValue(iInner, reduceOut);
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
        }
    }

    __aicore__ inline void ComputeSumPrecision(const LocalTensor<float>& sumLocal, uint32_t num)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<float> sqx = sqx_buf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduce_fp32_buf.Get<float>();
        if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
            LocalTensor<float> xBufFp32 = x_fp32_buf.Get<float>();
            Cast(xBufFp32, xLocal, RoundMode::CAST_NONE, num);
            inQueueX.FreeTensor(xLocal);
            PipeBarrier<PIPE_V>();
            Mul(sqx, xBufFp32, xBufFp32, num);
        } else {
            Mul(sqx, xLocal, xLocal, num);
            inQueueX.FreeTensor(xLocal);
        }
        PipeBarrier<PIPE_V>();
        ReduceSumHalfInterval(sumLocal, sqx, num);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeFormer(
        uint32_t i_o_idx, uint32_t calc_row_num, uint32_t j_idx, LocalTensor<float>& rstdLocal,
        LocalTensor<float>& sumLocal, uint32_t num, uint32_t left_num = 0, uint32_t reduce_mask = 0)
    {
        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            CopyIn(i_o_idx * row_factor + i_i, j_idx, num);
            ComputeSum(i_i, sumLocal, num, left_num, reduce_mask);
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 100
        WholeReduceSum<float>(sumLocal, sumLocal, NUM_PER_BLK_FP32, calc_row_num, 1, 1, 1);
#else
        BlockReduceSumFP32(sumLocal, sumLocal, calc_row_num * NUM_PER_BLK_FP32);
#endif
        Add(rstdLocal, rstdLocal, sumLocal, calc_row_num);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeSum(
        uint32_t i_i_idx, LocalTensor<float>& sumLocal, uint32_t num, uint32_t left_num = 0, uint32_t reduce_mask = 0)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<float> sqx = sqx_buf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduce_fp32_buf.Get<float>();
        if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
            LocalTensor<float> x_fp32 = x_fp32_buf.Get<float>();
            Cast(x_fp32, xLocal, RoundMode::CAST_NONE, num);
            inQueueX.FreeTensor(xLocal);
            PipeBarrier<PIPE_V>();
            Mul(sqx, x_fp32, x_fp32, num);
        } else {
            Mul(sqx, xLocal, xLocal, num);
            inQueueX.FreeTensor(xLocal);
        }
        PipeBarrier<PIPE_V>();
        Muls(sqx, sqx, avg_factor, num);
        PipeBarrier<PIPE_V>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 100
        ReduceSumHalfIntervalToRepeat(sumLocal[i_i_idx * NUM_PER_BLK_FP32], sqx, num, left_num);
#else
        ReduceSumFP32ToBlock(sumLocal[i_i_idx * MULTI_8], sqx, reduce_buf_local, num);
#endif
    }

    __aicore__ inline void ComputeRstd(LocalTensor<float> rstdLocal, uint32_t num)
    {
        LocalTensor<float> reduce_buf_local = reduce_fp32_buf.Get<float>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 100
        Muls(rstdLocal, rstdLocal, avg_factor, num);
        PipeBarrier<PIPE_V>();
#endif
        Adds(rstdLocal, rstdLocal, epsilon, num);
        PipeBarrier<PIPE_V>();
        Sqrt(rstdLocal, rstdLocal, num);
        Duplicate(reduce_buf_local, ONE, num);
        PipeBarrier<PIPE_V>();
        Div(rstdLocal, reduce_buf_local, rstdLocal, num);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeLatter(
        uint32_t i_o_idx, uint32_t calc_row_num, uint32_t j_idx, LocalTensor<float>& rstdLocal, uint32_t num)
    {
        CopyInGamma(j_idx, num);
        LocalTensor<T_GAMMA> gammaLocal = inQueueGamma.DeQue<T_GAMMA>();
        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            CopyIn(i_o_idx * row_factor + i_i, j_idx, num);
            ComputeY(i_i, gammaLocal, rstdLocal, num);
            CopyOutY(i_o_idx * row_factor + i_i, j_idx, num);
        }
        inQueueGamma.FreeTensor(gammaLocal);
    }

    __aicore__ inline void CopyInGamma(uint32_t j_idx, uint32_t num)
    {
        LocalTensor<T_GAMMA> gammaLocal = inQueueGamma.AllocTensor<T_GAMMA>();
        DataCopyCustom<T_GAMMA>(gammaLocal, gammaGm[j_idx * ub_factor], num);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void ComputeY(
        uint32_t rstdIdx, LocalTensor<T_GAMMA>& gammaLocal, LocalTensor<float>& rstdLocal, uint32_t num)
    {
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<float> sqx = sqx_buf.Get<float>();
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float rstdValue = rstdLocal.GetValue(rstdIdx);
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        // 1. Cast x and Mul rstd
        if constexpr (is_same<T, float>::value) {
            Muls(sqx, xLocal, rstdValue, num);
            PipeBarrier<PIPE_V>();
            if (is_gemma == 1) {
                LocalTensor<float> gammaFp32 = x_fp32_buf.Get<float>();
                Adds(gammaFp32, gammaLocal, static_cast<float>(1.0), num);
                PipeBarrier<PIPE_V>();
                Mul(yLocal, sqx, gammaFp32, num);
            } else {
                Mul(yLocal, sqx, gammaLocal, num);
            }
        } else {
            Cast(sqx, xLocal, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
            Muls(sqx, sqx, rstdValue, num);
            PipeBarrier<PIPE_V>();
            if constexpr (IS_MIX_DTYPE) {
                Mul(sqx, sqx, gammaLocal, num);
            } else {
                LocalTensor<float> gammaFp32 = x_fp32_buf.Get<float>();
                Cast(gammaFp32, gammaLocal, RoundMode::CAST_NONE, num);
                PipeBarrier<PIPE_V>();
                if (is_gemma == 1) {
                    Adds(gammaFp32, gammaFp32, static_cast<float>(1.0), num);
                    PipeBarrier<PIPE_V>();
                }
                Mul(sqx, sqx, gammaFp32, num);
            }
            PipeBarrier<PIPE_V>();
            if constexpr (is_same<T, half>::value) {
                Cast(yLocal, sqx, RoundMode::CAST_NONE, num);
            } else {
                Cast(yLocal, sqx, RoundMode::CAST_RINT, num);
            }
        }
        inQueueX.FreeTensor(xLocal);
        outQueueY.EnQue<T>(yLocal);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOutY(uint32_t i_idx, uint32_t j_idx, uint32_t num)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyCustom<T>(yGm[i_idx * num_col + j_idx * ub_factor], yLocal, num);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(uint32_t i_o_idx, uint32_t num)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
#if (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 100) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        DataCopyCustom<float>(rstdGm[i_o_idx * row_factor], rstdLocal, num);
#else
        uint32_t copyRstd_num = ROUND_UP(num, NUM_PER_BLK_FP32);
        SetAtomicAdd<float>();
        DataCopy(rstdGm[i_o_idx * row_factor], rstdLocal, copyRstd_num);
        SetAtomicNone();
#endif
        outQueueRstd.FreeTensor(rstdLocal);
    }

private:
    TPipe pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGamma;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    // create queues for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueRstd;
    TBuf<TPosition::VECCALC> x_fp32_buf;
    TBuf<TPosition::VECCALC> outTmpZeroBuf;
    TBuf<TPosition::VECCALC> sqx_buf;
    TBuf<TPosition::VECCALC> reduce_fp32_buf;
    TBuf<TPosition::VECCALC> sum_buf;

    GlobalTensor<T> xGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<T_GAMMA> gammaGm;
    GlobalTensor<T> yGm;

    uint64_t num_row;
    uint64_t num_col;
    uint64_t num_col_align;
    uint32_t reduce_mask; // number of calculations rows on each core
    uint32_t left_num;
    uint32_t last_reduce_mask; // number of calculations rows on each core
    uint32_t last_left_num;
    uint32_t block_factor; // number of calculations rows on each core
    uint32_t row_factor;
    uint32_t ub_factor;
    float epsilon;
    float avg_factor;
    int32_t blockIdx_;
    uint32_t data_per_block;
    uint32_t row_work = 1;
    uint32_t num_row_align;
    uint8_t is_gemma = 0;
    int tempbuf_num;
};
} // namespace RmsNorm
#endif // _RMS_NORM_SPLIT_D_H_
