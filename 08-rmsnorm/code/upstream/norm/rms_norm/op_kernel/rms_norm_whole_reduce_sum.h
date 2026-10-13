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
 * \file rms_norm_whole_reduce_sum.h
 * \brief rms norm whole reduce sum file
 */
#ifndef ASCENDC_RMS_NORM_WHOLE_REDUCE_SUM_H_
#define ASCENDC_RMS_NORM_WHOLE_REDUCE_SUM_H_
#include "rms_norm_base.h"

namespace RmsNorm {
using namespace AscendC;
template <typename T, bool IF_ALIGN>
class KernelRmsNormWholeReduceSum {
public:
    __aicore__ inline KernelRmsNormWholeReduceSum()
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, const RMSNormTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "block dim3 can not be zero!");
        InitVar(tiling);

        blockIdx_ = GetBlockIdx();
        if (blockIdx_ < GetBlockNum() - 1) {
            this->row_work = block_factor;
        } else if (blockIdx_ == GetBlockNum() - 1) {
            this->row_work = num_row - (GetBlockNum() - 1) * block_factor;
        } else {
        }
        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx_ * block_factor * num_col, row_work * num_col);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, num_col);
        yGm.SetGlobalBuffer((__gm__ T*)y + blockIdx_ * block_factor * num_col, row_work * num_col);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockIdx_ * block_factor, block_factor);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, row_factor * num_col_align * FLOAT_BTYPE_SIZE);
        pipe.InitBuffer(inQueueGamma, BUFFER_NUM, num_col_align * FLOAT_BTYPE_SIZE);
        pipe.InitBuffer(outQueueY, BUFFER_NUM, row_factor * num_col_align * FLOAT_BTYPE_SIZE);
        pipe.InitBuffer(outQueueRstd, 1, tiling->rstd_size);
        pipe.InitBuffer(reduce_fp32_buf, NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void InitVar(const RMSNormTilingData* tiling)
    {
        num_row = tiling->num_row;
        num_col = tiling->num_col;
        num_col_align = tiling->num_col_align;
        reduce_mask = tiling->reduce_mask;
        left_num = tiling->left_num;
        block_factor = tiling->block_factor;
        ub_factor = tiling->ub_factor;
        row_factor = tiling->row_factor;
        epsilon = tiling->epsilon;
        avg_factor = tiling->avg_factor;
        rstd_once_row = NUM_PER_REP_FP32;
        data_per_block = (BLOCK_SIZE / sizeof(T));
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
        LocalTensor<float> reduce_buf_local = reduce_fp32_buf.Get<float>();
        PipeBarrier<PIPE_V>();
        Duplicate(reduce_buf_local, ONE, NUM_PER_BLK_FP32);
        uint64_t i_o_max = CeilDiv(row_work, row_factor);
        uint64_t row_tail = row_work - (i_o_max - 1) * row_factor;
        uint64_t once_num = row_factor * num_col_align;
        mul_repeat_time = row_factor * mul_repeat_factor;
        uint64_t rstd_index;
        uint64_t rstd_offset;
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();

        for (uint64_t i_o = 0; i_o < i_o_max - 1; i_o++) {
            rstd_offset = i_o / rstd_once_row * rstd_once_row;
            SubProcess910(i_o, row_factor, rstdLocal, gammaLocal, once_num);
            if ((i_o - rstd_offset) == (rstd_once_row - 1)) {
                outQueueRstd.EnQue(rstdLocal); // TEST
                CopyOutRstd(rstd_offset * row_factor, rstd_once_row * row_factor);
                rstdLocal = outQueueRstd.AllocTensor<float>();
            }
            CopyOutY(i_o);
        }

        rstd_offset = (i_o_max - 1) / rstd_once_row * rstd_once_row * row_factor;
        uint32_t last_rstd_num = AlignUp(row_work - rstd_offset, NUM_PER_BLK_FP32);
        uint32_t last_block_row = data_per_block / num_col;
        SubProcess910(i_o_max - 1, row_tail, rstdLocal, gammaLocal, once_num);
        outQueueRstd.EnQue(rstdLocal); // TEST
        CopyOutRstd(rstd_offset, last_rstd_num);
        CopyOutTailY(i_o_max, row_tail, last_block_row);
        inQueueGamma.FreeTensor(gammaLocal);
    }

    __aicore__ inline void SubProcess910(
        uint64_t i_o, uint32_t calc_row_num, LocalTensor<float> rstdLocal, LocalTensor<T>& gammaLocal,
        uint64_t once_num)
    {
        uint64_t rstd_index = i_o % rstd_once_row * row_factor;
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        if constexpr (IF_ALIGN) {
            DataCopy(xLocal, xGm[(i_o * row_factor) * num_col], num_col * calc_row_num);
        } else {
            for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
                DataCopy(xLocal[i_i * num_col_align], xGm[(i_o * row_factor + i_i) * num_col], num_col_align);
            }
        }
        inQueueX.EnQue(xLocal);
        Compute(rstd_index, gammaLocal, rstdLocal, calc_row_num, once_num);
    }

private:
    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
        if constexpr (is_same<T, bfloat16_t>::value) {
            LocalTensor<T> gamma_tmp = inQueueX.AllocTensor<T>();
            DataCopy(gamma_tmp, gammaGm, num_col_align);
            LocalTensor<float> gamma_fp32 = gammaLocal.template ReinterpretCast<float>();
            inQueueX.EnQue(gamma_tmp);
            inQueueX.DeQue();
            Cast(gamma_fp32, gamma_tmp, RoundMode::CAST_NONE, num_col);
            inQueueX.FreeTensor(gamma_tmp);
        } else {
            DataCopy(gammaLocal, gammaGm, num_col_align);
        }
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void ComputeRstd(
        uint64_t inner_progress, LocalTensor<float> rstdLocal, LocalTensor<float> x_fp32, LocalTensor<float> sqx,
        LocalTensor<float> reduce_buf_local, LocalTensor<float> x, uint32_t calc_row_num, uint64_t once_num)
    {
        uint32_t repeat_time = calc_row_num / MAX_REAPEAT;
        uint32_t last_repeat_num = calc_row_num % MAX_REAPEAT;
        uint32_t start_index;
        Mul(sqx, x_fp32, x_fp32, once_num);
        PipeBarrier<PIPE_V>();

        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            ReduceSumHalfInterval(sqx[i_i * NUM_PER_BLK_FP32], sqx[i_i * num_col_align], num_col);
        }

        for (uint32_t i_i = 0; i_i < repeat_time; i_i++) {
            start_index = i_i * MAX_REAPEAT * NUM_PER_BLK_FP32;
            Muls(sqx[start_index], sqx[start_index], avg_factor, NUM_PER_BLK_FP32, MAX_REAPEAT, MulsParams);
            PipeBarrier<PIPE_V>();
            Adds(sqx[start_index], sqx[start_index], epsilon, NUM_PER_BLK_FP32, MAX_REAPEAT, MulsParams);
            PipeBarrier<PIPE_V>();
            Sqrt(sqx[start_index], sqx[start_index], NUM_PER_BLK_FP32, MAX_REAPEAT, MulsParams);
            PipeBarrier<PIPE_V>();
            Div(sqx[start_index], reduce_buf_local, sqx[start_index], NUM_PER_BLK_FP32, MAX_REAPEAT, DivParams);
        }
        start_index = repeat_time * MAX_REAPEAT * NUM_PER_BLK_FP32;
        Muls(sqx[start_index], sqx[start_index], avg_factor, NUM_PER_BLK_FP32, last_repeat_num, MulsParams);
        PipeBarrier<PIPE_V>();
        Adds(sqx[start_index], sqx[start_index], epsilon, NUM_PER_BLK_FP32, last_repeat_num, MulsParams);
        PipeBarrier<PIPE_V>();
        Sqrt(sqx[start_index], sqx[start_index], NUM_PER_BLK_FP32, last_repeat_num, MulsParams);
        PipeBarrier<PIPE_V>();
        Div(sqx[start_index], reduce_buf_local, sqx[start_index], NUM_PER_BLK_FP32, last_repeat_num, DivParams);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        event_t eventId_S_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            float rstd_value = sqx.GetValue(i_i * NUM_PER_BLK_FP32);
            SetFlag<HardEvent::S_V>(eventId_S_V);
            WaitFlag<HardEvent::S_V>(eventId_S_V);
            Muls(x[i_i * num_col_align], x[i_i * num_col_align], rstd_value, num_col_align);
            rstdLocal.SetValue(inner_progress + i_i, rstd_value);
        }
    }

    __aicore__ inline void Compute(
        uint64_t inner_progress, LocalTensor<bfloat16_t> gammaLocal, LocalTensor<float> rstdLocal,
        uint32_t calc_row_num, uint64_t once_num)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<float> gamma_fp32 = gammaLocal.template ReinterpretCast<float>();

        LocalTensor<float> x_fp32 = yLocal.template ReinterpretCast<float>();
        LocalTensor<float> sqx = xLocal.template ReinterpretCast<float>();
        LocalTensor<float> reduce_buf_local = reduce_fp32_buf.Get<float>();

        Cast(x_fp32, xLocal, RoundMode::CAST_NONE, once_num);
        PipeBarrier<PIPE_V>();
        ComputeRstd(inner_progress, rstdLocal, x_fp32, sqx, reduce_buf_local, x_fp32, calc_row_num, once_num);
        inQueueX.FreeTensor(xLocal);
        PipeBarrier<PIPE_V>();
        for (int32_t i_i = 0; i_i < calc_row_num; i_i++) {
            Mul(x_fp32[i_i * num_col_align], gamma_fp32, x_fp32[i_i * num_col_align], num_col_align);
        }
        Cast(yLocal, x_fp32, RoundMode::CAST_NONE, once_num);
        PipeBarrier<PIPE_V>();
        outQueueY.EnQue<T>(yLocal);
    }

    __aicore__ inline void Compute(
        uint64_t inner_progress, LocalTensor<half> gammaLocal, LocalTensor<float> rstdLocal, uint32_t calc_row_num,
        uint64_t once_num)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();

        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        PipeBarrier<PIPE_V>();

        LocalTensor<float> x_fp32 = yLocal.template ReinterpretCast<float>();
        LocalTensor<float> sqx = xLocal.template ReinterpretCast<float>();
        LocalTensor<float> reduce_buf_local = reduce_fp32_buf.Get<float>();
        Cast(x_fp32, xLocal, RoundMode::CAST_NONE, once_num);
        PipeBarrier<PIPE_V>();
        ComputeRstd(inner_progress, rstdLocal, x_fp32, sqx, reduce_buf_local, x_fp32, calc_row_num, once_num);
        PipeBarrier<PIPE_V>();
        inQueueX.FreeTensor(xLocal);

        PipeBarrier<PIPE_V>();
        Cast(yLocal, x_fp32, RoundMode::CAST_NONE, once_num);
        PipeBarrier<PIPE_V>();

        for (int32_t i_i = 0; i_i < calc_row_num; i_i++) {
            Mul(yLocal[i_i * num_col_align], gammaLocal, yLocal[i_i * num_col_align], num_col_align);
        }
        outQueueY.EnQue<T>(yLocal);
    }

    __aicore__ inline void Compute(
        uint64_t inner_progress, LocalTensor<float> gammaLocal, LocalTensor<float> rstdLocal, uint32_t calc_row_num,
        uint64_t once_num)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<float> reduce_buf_local = reduce_fp32_buf.Get<float>();
        DataCopy(yLocal, xLocal, once_num);

        PipeBarrier<PIPE_V>();
        ComputeRstd(inner_progress, rstdLocal, xLocal, xLocal, reduce_buf_local, yLocal, calc_row_num, once_num);
        inQueueX.FreeTensor(xLocal);
        for (int32_t i_i = 0; i_i < calc_row_num; i_i++) {
            Mul(yLocal[i_i * num_col_align], gammaLocal, yLocal[i_i * num_col_align], num_col_align);
        }
        outQueueY.EnQue<T>(yLocal);
    }

    __aicore__ inline void CopyOutTailY(uint32_t i_o_max, uint64_t row_tail, uint32_t last_block_row)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        if constexpr (IF_ALIGN) {
            DataCopy(yGm[((i_o_max - 1) * row_factor) * num_col], yLocal, row_tail * num_col);
            outQueueY.FreeTensor(yLocal);
        } else {
            if (row_tail * num_col >= data_per_block) {
                for (uint32_t i_i = 0; i_i < row_tail - last_block_row; i_i++) {
                    PipeBarrier<PIPE_MTE3>();
                    DataCopy(
                        yGm[((i_o_max - 1) * row_factor + i_i) * num_col], yLocal[i_i * num_col_align], num_col_align);
                }
                if (num_col > data_per_block) {
                    PipeBarrier<PIPE_MTE3>();
                    DataCopy(
                        yGm[(row_work - 1) * num_col], yLocal[(row_tail - 1) * num_col_align],
                        num_col_align - data_per_block);
                }
                event_t eventIdMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
                SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
                WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
                for (uint32_t last_data = 0; last_data < data_per_block; last_data++) {
                    uint32_t row_index = last_data / num_col;
                    uint32_t col_index = last_data % num_col;
                    yLocal.SetValue(
                        row_tail * num_col_align - 1 - last_data,
                        yLocal.GetValue((row_tail - 1 - row_index) * num_col_align + num_col - 1 - col_index));
                }
                PipeBarrier<PIPE_MTE3>();
                event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
                WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
                DataCopy(
                    yGm[row_work * num_col - data_per_block], yLocal[row_tail * num_col_align - data_per_block],
                    data_per_block);
            } else {
                uint32_t start_index = (i_o_max - 1) * row_factor * num_col;
                for (uint32_t last_data = 0; last_data < row_tail * num_col; last_data++) {
                    uint32_t row_index = last_data / num_col;
                    uint32_t col_index = last_data % num_col;
                    yLocal.SetValue(start_index + last_data, yLocal.GetValue((row_index)*num_col_align + col_index));
                }
                DataCopy(yGm[start_index], yLocal, data_per_block);
            }
            outQueueY.FreeTensor(yLocal);
        }
    }

    __aicore__ inline void CopyOutY(uint32_t i_o)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        if constexpr (IF_ALIGN) {
            DataCopy(yGm[(i_o * row_factor) * num_col], yLocal, row_factor * num_col);
        } else {
            for (uint32_t i_i = 0; i_i < row_factor; i_i++) {
                PipeBarrier<PIPE_MTE3>();
                DataCopy(yGm[(i_o * row_factor + i_i) * num_col], yLocal[i_i * num_col_align], num_col_align);
            }
        }
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(uint32_t outer_progress, uint32_t num)
    {
        event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(event);
        WaitFlag<HardEvent::S_MTE3>(event);

        LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
        DataCopy(rstdGm[outer_progress], rstdLocal, num);

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

    TBuf<TPosition::VECCALC> sqx_buf;
    TBuf<TPosition::VECCALC> reduce_fp32_buf;
    TBuf<TPosition::VECCALC> x_fp32_buf;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<float> rstdGm;
    uint64_t num_row;
    uint64_t num_col;
    uint64_t num_col_align;
    uint8_t col_align_blk;
    uint32_t once_reduce_max_cols; // number of calculations rows on each core
    uint32_t reduce_repeat_time;
    uint32_t reduce_mask; // number of calculations rows on each core
    uint32_t left_num;
    uint32_t block_factor; // number of calculations rows on each core
    uint32_t row_factor;
    uint32_t ub_factor;
    uint32_t mul_mask;
    uint32_t mul_repeat_factor;
    uint32_t mul_repeat_time;
    uint8_t mul_blk;
    uint32_t rstd_once_row;
    uint32_t data_per_block;
    UnaryRepeatParams MulsParams = {1, 1, 1, 1};
    BinaryRepeatParams DivParams = {1, 1, 1, 1, 0, 1};
    float epsilon;
    float avg_factor;
    float rstd_value;
    int32_t blockIdx_;
    uint32_t row_work = 1;
};
} // namespace RmsNorm
#endif // _ASCENDC_RMS_NORM_WHOLE_REDUCE_SUM_H_