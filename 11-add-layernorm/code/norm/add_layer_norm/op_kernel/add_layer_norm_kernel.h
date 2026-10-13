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
 * \file add_layer_norm_kernel.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_KERNEL_H_
#define ADD_LAYER_NORM_KERNEL_H_

#include "add_layer_norm_base.h"

template <typename T_X1, typename T_X2, typename T_GAMMA, typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddLayerNorm {
#define IS_ADDITIONAL_OUTPUT_ENABLE ((TILING_KEY % 1000) / 100 == 1)
#define IS_NORMAL_CASE ((TILING_KEY % 100) / 10 == 0)
#define IS_SLICE_CASE ((TILING_KEY % 100) / 10 == 1)
#define IS_SINGLE_ROW_CASE ((TILING_KEY % 100) / 10 == 2)
#define IS_SINGLE_ROW_EXT_CASE ((TILING_KEY % 100) / 10 == 3)
#define IS_NORMAL_BIG_N_CASE ((TILING_KEY % 100) / 10 == 4)
#define IS_SLICE_EXT_CASE ((TILING_KEY % 100) / 10 == 5)
#define IS_BIAS_PRESENT ((TILING_KEY % 10) == 1)
#define IS_BIAS_BROADCAST ((TILING_KEY % 10) == 2)
#define IS_CAST_BEFORE_ADD (!is_same<T_X1, T_X2>::value)
#define IS_X1_NEEDCAST ((!is_same<T_X1, float>::value) && IS_CAST_BEFORE_ADD)
#define IS_X2_NEEDCAST ((!is_same<T_X2, float>::value) && IS_CAST_BEFORE_ADD)
#define IS_BETAGAMMA_NEEDCAST (!is_same<T_GAMMA, float>::value)

public:
    __aicore__ inline KernelAddLayerNorm(TPipe* pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline uint32_t CEIL_DIV(uint32_t x2, uint32_t y)
    {
        if (y > 0) {
            return (x2 + y - 1) / y;
        }
        return 0;
    }

    __aicore__ inline uint32_t ROUND_UP32(uint32_t x)
    {
        return (x + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }

    __aicore__ inline uint32_t MIN_KERNEL(uint32_t x, uint32_t y)
    {
        return x < y ? x : y;
    }

    __aicore__ inline uint32_t MAX_KERNEL(uint32_t x, uint32_t y)
    {
        return x > y ? x : y;
    }

    __aicore__ inline void Init(
        __gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* gamma, __gm__ uint8_t* beta, __gm__ uint8_t* bias,
        __gm__ uint8_t* y, __gm__ uint8_t* mean, __gm__ uint8_t* rstd, __gm__ uint8_t* x, __gm__ uint8_t* workspace,
        uint32_t num_core_, uint32_t num_Last_dim_, uint32_t num_first_dim_, uint32_t nl_first_dim_per_core_,
        uint32_t l_first_dim_per_core_, uint32_t first_dim_per_time_, uint32_t last_dim_per_time_, float eps_,
        float aveNum_, uint32_t col_move_cnt_, uint32_t col_tail_, uint32_t workspace_size)
    {
        num_core = num_core_;
        num_last_dim = num_Last_dim_;
        num_first_dim = num_first_dim_;
        nl_first_dim_per_core = nl_first_dim_per_core_;
        l_first_dim_per_core = l_first_dim_per_core_;
        first_dim_per_time = first_dim_per_time_;
        last_dim_per_time = last_dim_per_time_;
        aveNum = aveNum_;
        eps = eps_;
        col_move_cnt = col_move_cnt_;
        col_tail = col_tail_;
        if (block_idx != num_core - 1) {
            row_work = nl_first_dim_per_core;
            row_step = first_dim_per_time;
        } else {
            row_work = l_first_dim_per_core;
            row_step = MIN_KERNEL(first_dim_per_time, row_work);
        }
        row_tail_ = (row_work % row_step == 0) ? row_step : (row_work % row_step);
        gm_offset_ = nl_first_dim_per_core * num_last_dim;
        x1_gm.SetGlobalBuffer((__gm__ T_X1*)(x1) + block_idx * gm_offset_);
        x2_gm.SetGlobalBuffer((__gm__ T_X2*)(x2) + block_idx * gm_offset_);
        if constexpr (IS_BIAS_PRESENT) {
            bias_gm.SetGlobalBuffer((__gm__ T*)(bias) + block_idx * gm_offset_);
        } else if constexpr (IS_BIAS_BROADCAST) {
            bias_gm.SetGlobalBuffer((__gm__ T*)bias);
        }
        gamma_gm.SetGlobalBuffer((__gm__ T_GAMMA*)gamma);
        beta_gm.SetGlobalBuffer((__gm__ T_GAMMA*)beta);
        y_gm.SetGlobalBuffer((__gm__ T*)(y) + block_idx * gm_offset_);
        // mean/rstd always output fp32
        mean_gm.SetGlobalBuffer((__gm__ float*)mean + block_idx * nl_first_dim_per_core);
        rstd_gm.SetGlobalBuffer((__gm__ float*)rstd + block_idx * nl_first_dim_per_core);
        x_gm.SetGlobalBuffer((__gm__ T*)(x) + block_idx * gm_offset_);
        workspace_gm.SetGlobalBuffer((__gm__ float*)workspace + workspace_size);
        num_last_dim_aligned = num_last_dim;
        if (ROUND_UP32(num_last_dim * sizeof(T)) != num_last_dim * sizeof(T)) {
            last_dim_pad = true;
            num_last_dim_aligned = ROUND_UP32(num_last_dim * sizeof(T)) / sizeof(T);
        }
        if constexpr (IS_X1_NEEDCAST || IS_X2_NEEDCAST) {
            numLastDimAlignedMixDtype = num_last_dim;
            if (ROUND_UP32(num_last_dim * sizeof(half)) != num_last_dim * sizeof(half)) {
                lastDimPadMixDtype = true;
                numLastDimAlignedMixDtype = ROUND_UP32(num_last_dim * sizeof(half)) / sizeof(half);
            }
        }

        if constexpr (IS_NORMAL_CASE || IS_NORMAL_BIG_N_CASE) { // normal case
            Ppipe->InitBuffer(x1_que, BUFFER_NUM, ROUND_UP32(row_step * num_last_dim_aligned * sizeof(T)));
            Ppipe->InitBuffer(x2_que, BUFFER_NUM, ROUND_UP32(row_step * num_last_dim_aligned * sizeof(T)));
            Ppipe->InitBuffer(y_que, BUFFER_NUM, ROUND_UP32(row_step * num_last_dim_aligned * sizeof(T)));
            Ppipe->InitBuffer(beta_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T_GAMMA)));
            Ppipe->InitBuffer(gamma_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T_GAMMA)));
            if (IS_NORMAL_BIG_N_CASE && num_last_dim_aligned < ONE_BLK_FLOAT_NUM * 2) {
                Ppipe->InitBuffer(x_buf_fp32, ROUND_UP32(row_step * num_last_dim_aligned * 2 * sizeof(float)));
            } else {
                Ppipe->InitBuffer(x_buf_fp32, ROUND_UP32(row_step * num_last_dim_aligned * sizeof(float)));
            }
            Ppipe->InitBuffer(y_buf_fp32, ROUND_UP32(row_step * num_last_dim_aligned * sizeof(float)));
            Ppipe->InitBuffer(z_buf_fp32, ROUND_UP32(row_step * num_last_dim_aligned * sizeof(float)));
            if constexpr (IS_BIAS_BROADCAST) {
                Ppipe->InitBuffer(bias_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T)));
            }
        } else if constexpr (IS_SLICE_CASE) { // slice case
            Ppipe->InitBuffer(x1_que, BUFFER_NUM, ROUND_UP32(last_dim_per_time * sizeof(T)));
            Ppipe->InitBuffer(x2_que, BUFFER_NUM, ROUND_UP32(last_dim_per_time * sizeof(T)));
            Ppipe->InitBuffer(y_que, BUFFER_NUM, ROUND_UP32(last_dim_per_time * sizeof(T)));
            Ppipe->InitBuffer(beta_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T_GAMMA))); // full load beta/gamma
            Ppipe->InitBuffer(gamma_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T_GAMMA)));
            Ppipe->InitBuffer(x_buf_fp32, ROUND_UP32(num_last_dim * sizeof(float))); // store x
            Ppipe->InitBuffer(y_buf_fp32, ROUND_UP32(last_dim_per_time * sizeof(float)));
            Ppipe->InitBuffer(z_buf_fp32, ROUND_UP32(last_dim_per_time * sizeof(float)));
            if constexpr (IS_BIAS_BROADCAST) {
                Ppipe->InitBuffer(bias_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T))); // full load bias
            }
        } else if constexpr (IS_SLICE_EXT_CASE) { // slice ext case
            Ppipe->InitBuffer(x1_que, BUFFER_NUM, ROUND_UP32(last_dim_per_time * sizeof(T)));
            Ppipe->InitBuffer(x2_que, BUFFER_NUM, ROUND_UP32(last_dim_per_time * sizeof(T)));
            Ppipe->InitBuffer(y_que, BUFFER_NUM, ROUND_UP32(last_dim_per_time * sizeof(T)));
            Ppipe->InitBuffer(beta_que, BUFFER_NUM, ROUND_UP32(last_dim_per_time * sizeof(T_GAMMA)));
            Ppipe->InitBuffer(gamma_que, BUFFER_NUM, ROUND_UP32(last_dim_per_time * sizeof(T_GAMMA)));
            Ppipe->InitBuffer(x_buf_fp32, ROUND_UP32(last_dim_per_time * sizeof(float)));
            Ppipe->InitBuffer(y_buf_fp32, ROUND_UP32(last_dim_per_time * sizeof(float)));
            Ppipe->InitBuffer(z_buf_fp32, ROUND_UP32(last_dim_per_time * sizeof(float)));
            if constexpr (IS_BIAS_BROADCAST) {
                Ppipe->InitBuffer(bias_que, BUFFER_NUM, ROUND_UP32(last_dim_per_time * sizeof(T)));
            }
        } else if constexpr (IS_SINGLE_ROW_CASE) { // single row
            Ppipe->InitBuffer(x1_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T)));
            Ppipe->InitBuffer(x2_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T)));
            Ppipe->InitBuffer(y_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T)));
            Ppipe->InitBuffer(beta_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T_GAMMA)));
            Ppipe->InitBuffer(gamma_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T_GAMMA)));
            Ppipe->InitBuffer(x_buf_fp32, ROUND_UP32(num_last_dim * sizeof(float)));
            Ppipe->InitBuffer(y_buf_fp32, ROUND_UP32(num_last_dim * sizeof(float)));
        } else if constexpr (IS_SINGLE_ROW_EXT_CASE) { // single row ext case
            Ppipe->InitBuffer(x1_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T)));
            Ppipe->InitBuffer(y_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T)));
            Ppipe->InitBuffer(beta_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T_GAMMA)));
            Ppipe->InitBuffer(gamma_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T_GAMMA)));
            Ppipe->InitBuffer(x_buf_fp32, ROUND_UP32(num_last_dim * sizeof(float)));
            Ppipe->InitBuffer(y_buf_fp32, ROUND_UP32(num_last_dim * sizeof(float)));
        }
#if OUTPUT_MEAN_RSTD == 1
        Ppipe->InitBuffer(mean_que, BUFFER_NUM, ROUND_UP32(row_step * sizeof(float)));
        Ppipe->InitBuffer(rstd_que, BUFFER_NUM, ROUND_UP32(row_step * sizeof(float)));
#endif
        if constexpr (IS_ADDITIONAL_OUTPUT_ENABLE) {
            if constexpr (IS_NORMAL_CASE || IS_NORMAL_BIG_N_CASE) {
                Ppipe->InitBuffer(x_que, BUFFER_NUM, ROUND_UP32(row_step * num_last_dim_aligned * sizeof(T)));
            } else if constexpr (!IS_SINGLE_ROW_EXT_CASE) { // SINGLE_ROW_EXT_CASE x share que with y
                Ppipe->InitBuffer(x_que, BUFFER_NUM, ROUND_UP32(row_step * last_dim_per_time * sizeof(T)));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (IS_NORMAL_CASE || IS_NORMAL_BIG_N_CASE) {
            ProcessNormal();
        } else if constexpr (IS_SLICE_CASE) {
            ProcessSlice();
        } else if constexpr (IS_SINGLE_ROW_CASE || IS_SINGLE_ROW_EXT_CASE) {
            ProcessSingleRow();
        } else if constexpr (IS_SLICE_EXT_CASE) {
            ProcessSliceExt();
        }
    }

    __aicore__ inline void ProcessNormal()
    {
        int32_t row_move_cnt = CEIL_DIV(row_work, row_step);
        CopyInPhase0();
        LocalTensor<T_GAMMA> gamma_local = gamma_que.template DeQue<T_GAMMA>();
        LocalTensor<T_GAMMA> beta_local = beta_que.template DeQue<T_GAMMA>();
        LocalTensor<T> bias_local;
        if constexpr (IS_BIAS_BROADCAST) {
            bias_local = bias_que.template DeQue<T>();
        }
        for (int32_t row_idx = 0; row_idx < row_move_cnt; ++row_idx) {
            if (row_idx < row_move_cnt - 1) {
                CopyIn(row_idx, row_step);
                if constexpr (IS_BIAS_PRESENT) {
                    CopyInAddBiasNormal(row_idx, row_step);
                } else if constexpr (IS_BIAS_BROADCAST) {
                    AddBiasBroadCast(row_step, bias_local);
                }
                CopyOutAdditionalOutput(row_idx, row_step);
                if constexpr (IS_NORMAL_BIG_N_CASE) {
                    precision_compute_big_n(row_step, gamma_local, beta_local);
                } else {
                    precision_compute(row_step, gamma_local, beta_local);
                }
                CopyOut(row_idx, row_step);
            } else {
                CopyIn(row_idx, row_tail_);
                if constexpr (IS_BIAS_PRESENT) {
                    CopyInAddBiasNormal(row_idx, row_tail_);
                } else if constexpr (IS_BIAS_BROADCAST) {
                    AddBiasBroadCast(row_tail_, bias_local);
                }
                CopyOutAdditionalOutput(row_idx, row_tail_);
                if constexpr (IS_NORMAL_BIG_N_CASE) {
                    precision_compute_big_n(row_tail_, gamma_local, beta_local);
                } else {
                    precision_compute(row_tail_, gamma_local, beta_local);
                }
                CopyOut(row_idx, row_tail_);
            }
        }
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
        if constexpr (IS_BIAS_BROADCAST) {
            bias_que.FreeTensor(bias_local);
        }
    }

    __aicore__ inline void ProcessSingleRow()
    {
        int32_t row_move_cnt = CEIL_DIV(row_work, row_step);
        CopyInPhase0();
        LocalTensor<T_GAMMA> gamma_local = gamma_que.template DeQue<T_GAMMA>();
        LocalTensor<T_GAMMA> beta_local = beta_que.template DeQue<T_GAMMA>();
        for (int32_t row_idx = 0; row_idx < row_move_cnt; ++row_idx) {
            CopyInAddSingleRow(row_idx, num_last_dim);
            precision_compute_single_row(gamma_local, beta_local);
            CopyOut(row_idx, 1);
        }
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
    }

    __aicore__ inline void ProcessSlice()
    {
        int32_t row_move_cnt = CEIL_DIV(row_work, row_step);
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();

        CopyInSlicePhase0(num_last_dim);
        LocalTensor<T_GAMMA> beta_local = beta_que.template DeQue<T_GAMMA>();
        LocalTensor<T_GAMMA> gamma_local = gamma_que.template DeQue<T_GAMMA>();
        LocalTensor<T> bias_local;
        if constexpr (IS_BIAS_BROADCAST) {
            bias_local = bias_que.template DeQue<T>();
        }
        for (int32_t row_idx = 0; row_idx < row_move_cnt; ++row_idx) {
#if OUTPUT_MEAN_RSTD == 1
            LocalTensor<float> mean_local = mean_que.template AllocTensor<float>();
            LocalTensor<float> rstd_local = rstd_que.template AllocTensor<float>();
#endif
            // Reduce Mean
            float ave_tmp = 0;
            Duplicate(z_local_fp32, aveNum, last_dim_per_time);
            for (int32_t col_idx = 0; col_idx < col_move_cnt; ++col_idx) {
                auto col_offset = col_idx * last_dim_per_time;
                int process_count = col_idx < col_move_cnt - 1 ? last_dim_per_time : col_tail;
                CopyInSlicePhase1(row_idx, process_count, col_offset);
                LocalTensor<T> x1_local = x1_que.template DeQue<T>();
                LocalTensor<T> x2_local = x2_que.template DeQue<T>();
                if constexpr (is_same<T, float>::value) {
                    if constexpr (IS_X1_NEEDCAST) {
                        auto y_local_buffer = y_local_fp32.template ReinterpretCast<T_X1>();
                        Cast(x1_local, y_local_buffer, RoundMode::CAST_NONE, process_count);
                        PipeBarrier<PIPE_V>();
                    }
                    if constexpr (IS_X2_NEEDCAST) {
                        auto y_local_buffer = y_local_fp32.template ReinterpretCast<T_X2>();
                        Cast(x2_local, y_local_buffer, RoundMode::CAST_NONE, process_count);
                        PipeBarrier<PIPE_V>();
                    }
                    Add(x_local_fp32[col_offset], x1_local, x2_local, process_count);
                    PipeBarrier<PIPE_V>();
                } else {
                    Cast(x_local_fp32[col_offset], x1_local, RoundMode::CAST_NONE, process_count);
                    Cast(y_local_fp32, x2_local, RoundMode::CAST_NONE, process_count);
                    PipeBarrier<PIPE_V>();
                    Add(x_local_fp32[col_offset], x_local_fp32[col_offset], y_local_fp32, process_count);
                    PipeBarrier<PIPE_V>();
                }
                x1_que.FreeTensor(x1_local);
                x2_que.FreeTensor(x2_local);

                if constexpr (IS_BIAS_PRESENT) {
                    LocalTensor<T> x3_in = x1_que.template AllocTensor<T>();
                    uint32_t gm_offset = row_idx * row_step * num_last_dim + col_offset;
                    DataCopyEx(x3_in, bias_gm[gm_offset], process_count);
                    x1_que.EnQue(x3_in);
                    auto x3_local = x1_que.template DeQue<T>();
                    if constexpr (is_same<T, float>::value) {
                        Add(x_local_fp32[col_offset], x3_local, x_local_fp32[col_offset], process_count);
                        PipeBarrier<PIPE_V>();
                    } else {
                        Cast(y_local_fp32, x3_local, RoundMode::CAST_NONE, process_count);
                        PipeBarrier<PIPE_V>();
                        Add(x_local_fp32[col_offset], y_local_fp32, x_local_fp32[col_offset], process_count);
                        PipeBarrier<PIPE_V>();
                    }
                    x1_que.FreeTensor(x3_local);
                } else if constexpr (IS_BIAS_BROADCAST) {
                    if constexpr (is_same<T, float>::value) {
                        Add(x_local_fp32[col_offset], bias_local[col_offset], x_local_fp32[col_offset], process_count);
                        PipeBarrier<PIPE_V>();
                    } else {
                        Cast(y_local_fp32, bias_local[col_offset], RoundMode::CAST_NONE, process_count);
                        PipeBarrier<PIPE_V>();
                        Add(x_local_fp32[col_offset], y_local_fp32, x_local_fp32[col_offset], process_count);
                        PipeBarrier<PIPE_V>();
                    }
                }

                if constexpr (IS_ADDITIONAL_OUTPUT_ENABLE) {
                    LocalTensor<T> x_local = x_que.template AllocTensor<T>();
                    if constexpr (is_same<T, float>::value) {
                        Adds(x_local, x_local_fp32[col_offset], ZERO, process_count);
                    } else if constexpr (is_same<T, half>::value) {
                        Cast(x_local, x_local_fp32[col_offset], RoundMode::CAST_NONE, process_count);
                    } else {
                        Cast(x_local, x_local_fp32[col_offset], RoundMode::CAST_RINT, process_count);
                    }
                    x_que.EnQue(x_local);
                    CopyOutSlicePhase0(row_idx, process_count, last_dim_per_time * col_idx);
                }
                Mul(y_local_fp32, x_local_fp32[col_offset], z_local_fp32, process_count);
                PipeBarrier<PIPE_V>();
                ReduceSum(y_local_fp32, y_local_fp32, y_local_fp32, process_count);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                ave_tmp += y_local_fp32.GetValue(0);
            }
            // 2. Reduce Var
            float var_tmp_2 = 0;
            for (int32_t col_idx_2 = 0; col_idx_2 < col_move_cnt; ++col_idx_2) {
                auto col_offset_2 = col_idx_2 * last_dim_per_time;
                int process_count_2 = col_idx_2 < col_move_cnt - 1 ? last_dim_per_time : col_tail;
                Adds(x_local_fp32[col_offset_2], x_local_fp32[col_offset_2], -1 * ave_tmp, process_count_2);
                PipeBarrier<PIPE_V>();
                Mul(y_local_fp32, x_local_fp32[col_offset_2], x_local_fp32[col_offset_2], process_count_2);
                PipeBarrier<PIPE_V>();
                Muls(y_local_fp32, y_local_fp32, aveNum, process_count_2);
                PipeBarrier<PIPE_V>();
                ReduceSum(y_local_fp32, y_local_fp32, y_local_fp32, process_count_2);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                var_tmp_2 += y_local_fp32.GetValue(0);
            }
            float rstd_tmp_2 = 1 / sqrt(var_tmp_2 + eps);
#if OUTPUT_MEAN_RSTD == 1
            mean_local.SetValue(0, ave_tmp);
            rstd_local.SetValue(0, rstd_tmp_2);
#endif
            // 3. Compute result
            for (int32_t col_idx = 0; col_idx < col_move_cnt; ++col_idx) {
                auto col_offset = col_idx * last_dim_per_time;
                int process_count = col_idx < col_move_cnt - 1 ? last_dim_per_time : col_tail;
                LocalTensor<T> y_local = y_que.template AllocTensor<T>();
                // x_local_fp32[col_offset] = (x - ave)
                Muls(y_local_fp32, x_local_fp32[col_offset], rstd_tmp_2, process_count);
                PipeBarrier<PIPE_V>();
                if constexpr (IS_BETAGAMMA_NEEDCAST) {
                    if constexpr (is_same<T, float>::value) {
                        Cast(x_local_fp32[col_offset], gamma_local[col_offset], RoundMode::CAST_NONE, process_count);
                        PipeBarrier<PIPE_V>();
                        Mul(y_local_fp32, x_local_fp32[col_offset], y_local_fp32, process_count);
                        Cast(x_local_fp32[col_offset], beta_local[col_offset], RoundMode::CAST_NONE, process_count);
                        PipeBarrier<PIPE_V>();
                        Add(y_local, y_local_fp32, x_local_fp32[col_offset], process_count);
                        PipeBarrier<PIPE_V>();
                    } else {
                        Cast(x_local_fp32[col_offset], gamma_local[col_offset], RoundMode::CAST_NONE, process_count);
                        PipeBarrier<PIPE_V>();
                        Mul(y_local_fp32, x_local_fp32[col_offset], y_local_fp32, process_count);
                        Cast(x_local_fp32[col_offset], beta_local[col_offset], RoundMode::CAST_NONE, process_count);
                        PipeBarrier<PIPE_V>();
                        Add(y_local_fp32, y_local_fp32, x_local_fp32[col_offset], process_count);
                        PipeBarrier<PIPE_V>();
                        if constexpr (is_same<T, half>::value) {
                            Cast(y_local, y_local_fp32, RoundMode::CAST_NONE, process_count);
                        } else {
                            Cast(y_local, y_local_fp32, RoundMode::CAST_RINT, process_count);
                        }
                    }
                } else {
                    if constexpr (is_same<T, float>::value) {
                        Mul(y_local_fp32, gamma_local[col_offset], y_local_fp32, process_count);
                        PipeBarrier<PIPE_V>();
                        Add(y_local, y_local_fp32, beta_local[col_offset], process_count);
                    } else {
                        Mul(y_local_fp32, gamma_local[col_offset], y_local_fp32, process_count);
                        PipeBarrier<PIPE_V>();
                        Add(y_local_fp32, y_local_fp32, beta_local[col_offset], process_count);
                        PipeBarrier<PIPE_V>();
                        if constexpr (is_same<T, half>::value) {
                            Cast(y_local, y_local_fp32, RoundMode::CAST_NONE, process_count);
                        } else {
                            Cast(y_local, y_local_fp32, RoundMode::CAST_RINT, process_count);
                        }
                    }
                }

                y_que.EnQue(y_local);
                CopyOutSlicePhase1(row_idx, process_count, last_dim_per_time * col_idx);
            }
#if OUTPUT_MEAN_RSTD == 1
            mean_que.EnQue(mean_local);
            rstd_que.EnQue(rstd_local);
            CopyOutSlicePhase2(row_idx);
#endif
        }
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
    }

    __aicore__ inline void CopyInAddSlice(int32_t row_idx, int32_t process_count, int32_t col_offset)
    {
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32_2 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32_2 = z_buf_fp32.Get<float>();
        CopyInSlicePhase1(row_idx, process_count, col_offset);
        LocalTensor<T> x1_local_2 = x1_que.template DeQue<T>();
        LocalTensor<T> x2_local_2 = x2_que.template DeQue<T>();
        if constexpr (is_same<T, float>::value) {
            if constexpr (IS_X1_NEEDCAST) {
                auto y_local_buffer_2 = y_local_fp32_2.template ReinterpretCast<T_X1>();
                Cast(x1_local_2, y_local_buffer_2, RoundMode::CAST_NONE, process_count);
                PipeBarrier<PIPE_V>();
            }
            if constexpr (IS_X2_NEEDCAST) {
                auto y_local_buffer_2 = y_local_fp32_2.template ReinterpretCast<T_X2>();
                Cast(x2_local_2, y_local_buffer_2, RoundMode::CAST_NONE, process_count);
                PipeBarrier<PIPE_V>();
            }
            Add(x_local_fp32, x1_local_2, x2_local_2, process_count);
            PipeBarrier<PIPE_V>();
        } else {
            Cast(x_local_fp32, x1_local_2, RoundMode::CAST_NONE, process_count);
            Cast(y_local_fp32_2, x2_local_2, RoundMode::CAST_NONE, process_count);
            PipeBarrier<PIPE_V>();
            Add(x_local_fp32, x_local_fp32, y_local_fp32_2, process_count);
            PipeBarrier<PIPE_V>();
        }
        x1_que.FreeTensor(x1_local_2);
        x2_que.FreeTensor(x2_local_2);
        if constexpr (IS_BIAS_PRESENT) {
            LocalTensor<T> x3_in_2 = x1_que.template AllocTensor<T>();
            uint32_t gm_offset_2 = row_idx * row_step * num_last_dim + col_offset;
            DataCopyEx(x3_in_2, bias_gm[gm_offset_2], process_count);
            x1_que.EnQue(x3_in_2);
            auto x3_local_2 = x1_que.template DeQue<T>();
            if constexpr (is_same<T, float>::value) {
                Add(x_local_fp32, x3_local_2, x_local_fp32, process_count);
                PipeBarrier<PIPE_V>();
            } else {
                Cast(y_local_fp32_2, x3_local_2, RoundMode::CAST_NONE, process_count);
                PipeBarrier<PIPE_V>();
                Add(x_local_fp32, y_local_fp32_2, x_local_fp32, process_count);
                PipeBarrier<PIPE_V>();
            }
            x1_que.FreeTensor(x3_local_2);
        } else if constexpr (IS_BIAS_BROADCAST) {
            LocalTensor<T> bias_local_in = bias_que.template AllocTensor<T>();
            DataCopyEx(bias_local_in, bias_gm[col_offset], process_count);
            bias_que.EnQue(bias_local_in);
            auto bias_local = bias_que.template DeQue<T>();
            if constexpr (is_same<T, float>::value) {
                Add(x_local_fp32, bias_local, x_local_fp32, process_count);
                PipeBarrier<PIPE_V>();
            } else {
                Cast(y_local_fp32_2, bias_local, RoundMode::CAST_NONE, process_count);
                PipeBarrier<PIPE_V>();
                Add(x_local_fp32, y_local_fp32_2, x_local_fp32, process_count);
                PipeBarrier<PIPE_V>();
            }
            bias_que.FreeTensor(bias_local);
        }
    }

    __aicore__ inline void CopyOutAdditionalOutputSlice(int32_t row_idx, int32_t process_count, int32_t col_offset)
    {
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<T> x_local = x_que.template AllocTensor<T>();
        if constexpr (is_same<T, float>::value) {
            Adds(x_local, x_local_fp32, ZERO, process_count);
        } else if constexpr (is_same<T, half>::value) {
            Cast(x_local, x_local_fp32, RoundMode::CAST_NONE, process_count);
        } else {
            Cast(x_local, x_local_fp32, RoundMode::CAST_RINT, process_count);
        }
        x_que.EnQue(x_local);
        CopyOutSlicePhase0(row_idx, process_count, col_offset);
    }

    __aicore__ inline void ProcessSliceExt()
    {
        int32_t row_move_cnt = CEIL_DIV(row_work, row_step);
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        for (int32_t row_idx = 0; row_idx < row_move_cnt; ++row_idx) {
#if OUTPUT_MEAN_RSTD == 1
            LocalTensor<float> mean_local = mean_que.template AllocTensor<float>();
            LocalTensor<float> rstd_local = rstd_que.template AllocTensor<float>();
#endif
            // Reduce Mean
            float ave_tmp = 0;
            for (int32_t col_idx = 0; col_idx < col_move_cnt; ++col_idx) {
                auto col_offset = col_idx * last_dim_per_time;
                int process_count = col_idx < col_move_cnt - 1 ? last_dim_per_time : col_tail;
                CopyInAddSlice(row_idx, process_count, col_offset);
                Muls(y_local_fp32, x_local_fp32, aveNum, process_count);
                PipeBarrier<PIPE_V>();
                ReduceSum(y_local_fp32, y_local_fp32, y_local_fp32, process_count);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                ave_tmp += y_local_fp32.GetValue(0);
            }
            // 2. Reduce Var
            float var_tmp = 0;
            for (int32_t col_idx = 0; col_idx < col_move_cnt; ++col_idx) {
                auto col_offset = col_idx * last_dim_per_time;
                int process_count = col_idx < col_move_cnt - 1 ? last_dim_per_time : col_tail;
                CopyInAddSlice(row_idx, process_count, col_offset);
                Adds(x_local_fp32, x_local_fp32, -1 * ave_tmp, process_count);
                PipeBarrier<PIPE_V>();
                Mul(y_local_fp32, x_local_fp32, x_local_fp32, process_count);
                PipeBarrier<PIPE_V>();
                Muls(y_local_fp32, y_local_fp32, aveNum, process_count);
                PipeBarrier<PIPE_V>();
                ReduceSum(y_local_fp32, y_local_fp32, y_local_fp32, process_count);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                var_tmp += y_local_fp32.GetValue(0);
            }
            float rstd_tmp = 1 / sqrt(var_tmp + eps);
#if OUTPUT_MEAN_RSTD == 1
            mean_local.SetValue(0, ave_tmp);
            rstd_local.SetValue(0, rstd_tmp);
            mean_que.EnQue(mean_local);
            rstd_que.EnQue(rstd_local);
            CopyOutSlicePhase2(row_idx);
#endif
            // 3. Compute result
            for (int32_t col_idx = 0; col_idx < col_move_cnt; ++col_idx) {
                auto col_offset = col_idx * last_dim_per_time;
                int process_count = col_idx < col_move_cnt - 1 ? last_dim_per_time : col_tail;
                CopyInAddSlice(row_idx, process_count, col_offset);
                if constexpr (IS_ADDITIONAL_OUTPUT_ENABLE) {
                    CopyOutAdditionalOutputSlice(row_idx, process_count, col_offset);
                }
                CopyInSlicePhase2(process_count, col_offset);
                LocalTensor<T_GAMMA> beta_local = beta_que.template DeQue<T_GAMMA>();
                LocalTensor<T_GAMMA> gamma_local = gamma_que.template DeQue<T_GAMMA>();
                LocalTensor<T> y_local = y_que.template AllocTensor<T>();
                Adds(x_local_fp32, x_local_fp32, -1 * ave_tmp, process_count);
                PipeBarrier<PIPE_V>();
                Muls(y_local_fp32, x_local_fp32, rstd_tmp, process_count);
                PipeBarrier<PIPE_V>();

                if constexpr (IS_BETAGAMMA_NEEDCAST) {
                    if constexpr (is_same<T, float>::value) {
                        Cast(x_local_fp32, gamma_local, RoundMode::CAST_NONE, process_count);
                        PipeBarrier<PIPE_V>();
                        Mul(y_local_fp32, x_local_fp32, y_local_fp32, process_count);
                        Cast(x_local_fp32, beta_local, RoundMode::CAST_NONE, process_count);
                        PipeBarrier<PIPE_V>();
                        Add(y_local, y_local_fp32, x_local_fp32, process_count);
                    } else {
                        Cast(x_local_fp32, gamma_local, RoundMode::CAST_NONE, process_count);
                        PipeBarrier<PIPE_V>();
                        Mul(y_local_fp32, x_local_fp32, y_local_fp32, process_count);
                        Cast(x_local_fp32, beta_local, RoundMode::CAST_NONE, process_count);
                        PipeBarrier<PIPE_V>();
                        Add(y_local_fp32, y_local_fp32, x_local_fp32, process_count);
                        PipeBarrier<PIPE_V>();
                        if constexpr (is_same<T, half>::value) {
                            Cast(y_local, y_local_fp32, RoundMode::CAST_NONE, process_count);
                        } else {
                            Cast(y_local, y_local_fp32, RoundMode::CAST_RINT, process_count);
                        }
                    }
                } else {
                    if constexpr (is_same<T, float>::value) {
                        Mul(y_local_fp32, gamma_local, y_local_fp32, process_count);
                        PipeBarrier<PIPE_V>();
                        Add(y_local, y_local_fp32, beta_local, process_count);
                    } else {
                        Mul(y_local_fp32, gamma_local, y_local_fp32, process_count);
                        PipeBarrier<PIPE_V>();
                        Add(y_local_fp32, y_local_fp32, beta_local, process_count);
                        PipeBarrier<PIPE_V>();
                        if constexpr (is_same<T, half>::value) {
                            Cast(y_local, y_local_fp32, RoundMode::CAST_NONE, process_count);
                        } else {
                            Cast(y_local, y_local_fp32, RoundMode::CAST_RINT, process_count);
                        }
                    }
                }

                beta_que.FreeTensor(beta_local);
                gamma_que.FreeTensor(gamma_local);
                y_que.EnQue(y_local);
                CopyOutSlicePhase1(row_idx, process_count, last_dim_per_time * col_idx);
            }
        }
    }

private:
    __aicore__ inline void CopyInSlicePhase1(int32_t row_idx, int32_t size, int32_t offset = 0)
    {
        event_t event_mte3_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));

        LocalTensor<T> x1_local = x1_que.template AllocTensor<T>();
        LocalTensor<T> x2_local = x2_que.template AllocTensor<T>();
        uint32_t gm_offset = row_idx * row_step * num_last_dim + offset;
        if constexpr (IS_X1_NEEDCAST) {
            SetFlag<HardEvent::MTE3_S>(event_mte3_s);
            WaitFlag<HardEvent::MTE3_S>(event_mte3_s);
            LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
            auto y_local_buffer = y_local_fp32.template ReinterpretCast<T_X1>();
            DataCopyEx(y_local_buffer, x1_gm[gm_offset], size);
        } else {
            DataCopyEx(x1_local, x1_gm[gm_offset], size);
        }
        if constexpr (IS_X2_NEEDCAST) {
            SetFlag<HardEvent::MTE3_S>(event_mte3_s);
            WaitFlag<HardEvent::MTE3_S>(event_mte3_s);
            LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
            auto y_local_buffer = y_local_fp32.template ReinterpretCast<T_X2>();
            DataCopyEx(y_local_buffer, x2_gm[gm_offset], size);
        } else {
            DataCopyEx(x2_local, x2_gm[gm_offset], size);
        }
        x1_que.EnQue(x1_local);
        x2_que.EnQue(x2_local);
    }

    __aicore__ inline void CopyInSlicePhase2(int32_t size, int32_t offset = 0)
    {
        LocalTensor<T_GAMMA> beta_local = beta_que.template AllocTensor<T_GAMMA>();
        LocalTensor<T_GAMMA> gamma_local = gamma_que.template AllocTensor<T_GAMMA>();
        DataCopyEx(beta_local, beta_gm[offset], size);
        DataCopyEx(gamma_local, gamma_gm[offset], size);
        beta_que.EnQue(beta_local);
        gamma_que.EnQue(gamma_local);
    }

    __aicore__ inline void CopyInSlicePhase0(int32_t size, int32_t offset = 0)
    {
        LocalTensor<T_GAMMA> beta_local = beta_que.template AllocTensor<T_GAMMA>();
        LocalTensor<T_GAMMA> gamma_local = gamma_que.template AllocTensor<T_GAMMA>();
        DataCopyEx(beta_local, beta_gm[offset], size);
        DataCopyEx(gamma_local, gamma_gm[offset], size);
        beta_que.EnQue(beta_local);
        gamma_que.EnQue(gamma_local);
        if constexpr (IS_BIAS_BROADCAST) {
            LocalTensor<T> bias_local = bias_que.template AllocTensor<T>();
            DataCopyEx(bias_local, bias_gm[offset], size);
            bias_que.EnQue(bias_local);
        }
    }

    __aicore__ inline void CopyInPhase0()
    {
        LocalTensor<T_GAMMA> beta_local = beta_que.template AllocTensor<T_GAMMA>();
        LocalTensor<T_GAMMA> gamma_local = gamma_que.template AllocTensor<T_GAMMA>();
        DataCopyEx(beta_local, beta_gm, num_last_dim);
        DataCopyEx(gamma_local, gamma_gm, num_last_dim);
        beta_que.EnQue(beta_local);
        gamma_que.EnQue(gamma_local);
        if constexpr (IS_BIAS_BROADCAST && (!(IS_SINGLE_ROW_CASE || IS_SINGLE_ROW_EXT_CASE))) {
            LocalTensor<T> bias_local = bias_que.template AllocTensor<T>();
            DataCopyEx(bias_local, bias_gm, num_last_dim);
            bias_que.EnQue(bias_local);
        }
    }

    __aicore__ inline void CopyIn(int32_t proc_id, int32_t row_count)
    {
        event_t event_mte3_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_buf_local = y_buf_fp32.Get<float>();
        LocalTensor<float> add_buf_local = z_buf_fp32.Get<float>();
        uint32_t gm_offset = proc_id * row_step * num_last_dim;
        auto elementCount = num_last_dim_aligned * row_count;
        DataCopyPadParams padParams;
        if (last_dim_pad) {
            padParams.isPad = true;
            padParams.paddingValue = 0;
            padParams.rightPadding = num_last_dim_aligned - num_last_dim;
        }
        LocalTensor<T> x1_local_in = x1_que.template AllocTensor<T>();
        if constexpr (IS_X1_NEEDCAST) {
            DataCopyPadParams padParamsFp16;
            if (lastDimPadMixDtype) {
                padParamsFp16.isPad = true;
                padParamsFp16.paddingValue = 0;
                padParamsFp16.rightPadding = numLastDimAlignedMixDtype - num_last_dim;
            }
            SetFlag<HardEvent::MTE3_S>(event_mte3_s);
            WaitFlag<HardEvent::MTE3_S>(event_mte3_s);
            auto y_buf = y_buf_local.template ReinterpretCast<T_X1>();
            DataCopyEx(y_buf, x1_gm[gm_offset], num_last_dim, row_count, padParamsFp16);
        } else {
            DataCopyEx(x1_local_in, x1_gm[gm_offset], num_last_dim, row_count, padParams);
        }
        x1_que.EnQue(x1_local_in);
        auto x1_local = x1_que.template DeQue<T>();
        if constexpr (is_same<float, T>::value) {
            if constexpr (IS_X1_NEEDCAST) {
                auto y_buf = y_buf_local.template ReinterpretCast<T_X1>();
                for (uint32_t i = 0; i < row_count; i++) {
                    Cast(
                        add_buf_local[i * num_last_dim_aligned], y_buf[i * numLastDimAlignedMixDtype],
                        RoundMode::CAST_NONE, num_last_dim);
                }
            } else {
                Adds(add_buf_local, x1_local, ZERO, elementCount);
            }
            PipeBarrier<PIPE_V>();
        } else {
            Cast(add_buf_local, x1_local, RoundMode::CAST_NONE, elementCount);
        }
        x1_que.FreeTensor(x1_local);

        LocalTensor<T> x2_local_in = x2_que.template AllocTensor<T>();
        if constexpr (IS_X2_NEEDCAST) {
            DataCopyPadParams padParamsFp16;
            if (lastDimPadMixDtype) {
                padParamsFp16.isPad = true;
                padParamsFp16.paddingValue = 0;
                padParamsFp16.rightPadding = numLastDimAlignedMixDtype - num_last_dim;
            }
            SetFlag<HardEvent::MTE3_S>(event_mte3_s);
            WaitFlag<HardEvent::MTE3_S>(event_mte3_s);
            auto y_buf = x_local_fp32.template ReinterpretCast<T_X2>();
            DataCopyEx(y_buf, x2_gm[gm_offset], num_last_dim, row_count, padParamsFp16);
        } else {
            DataCopyEx(x2_local_in, x2_gm[gm_offset], num_last_dim, row_count, padParams);
        }
        x2_que.EnQue(x2_local_in);
        auto x2_local = x2_que.template DeQue<T>();
        if constexpr (is_same<float, T>::value) {
            if constexpr (IS_X2_NEEDCAST) {
                auto y_buf = x_local_fp32.template ReinterpretCast<T_X2>();
                for (uint32_t i = 0; i < row_count; i++) {
                    Cast(
                        x2_local[i * num_last_dim_aligned], y_buf[i * numLastDimAlignedMixDtype], RoundMode::CAST_NONE,
                        num_last_dim);
                }
                PipeBarrier<PIPE_V>();
                Add(add_buf_local, x2_local, add_buf_local, elementCount);
                PipeBarrier<PIPE_V>();
            } else {
                Add(add_buf_local, x2_local, add_buf_local, elementCount);
                PipeBarrier<PIPE_V>();
            }
        } else {
            Cast(y_buf_local, x2_local, RoundMode::CAST_NONE, elementCount);
            PipeBarrier<PIPE_V>();
            Add(add_buf_local, y_buf_local, add_buf_local, elementCount);
            PipeBarrier<PIPE_V>();
        }
        x2_que.FreeTensor(x2_local);
    }

    __aicore__ inline void CopyOutAdditionalOutput(int32_t proc_id, int32_t row_count)
    {
        if constexpr (IS_ADDITIONAL_OUTPUT_ENABLE) {
            LocalTensor<float> add_buf_local = z_buf_fp32.Get<float>();
            uint32_t gm_offset = proc_id * row_step * num_last_dim;
            auto elementCount = num_last_dim_aligned * row_count;
            auto x_local = x_que.template AllocTensor<T>();
            if constexpr (is_same<T, float>::value) {
                Adds(x_local, add_buf_local, ZERO, elementCount);
            } else if constexpr (is_same<T, half>::value) {
                Cast(x_local, add_buf_local, RoundMode::CAST_NONE, elementCount);
            } else {
                Cast(x_local, add_buf_local, RoundMode::CAST_RINT, elementCount);
            }
            PipeBarrier<PIPE_V>();
            x_que.template EnQue<T>(x_local);

            auto x = x_que.template DeQue<T>();
            DataCopyEx(x_gm[gm_offset], x, num_last_dim, row_count);
            x_que.FreeTensor(x);
        }
    }

    __aicore__ inline void AddBiasBroadCast(int32_t row_count, const LocalTensor<T>& bias_local)
    {
        auto y_buf_local = y_buf_fp32.Get<float>();
        auto add_buf_local = z_buf_fp32.Get<float>();
        if constexpr (is_same<float, T>::value) {
            for (int i = 0; i < row_count; i++) {
                Add(add_buf_local[i * num_last_dim_aligned], bias_local, add_buf_local[i * num_last_dim_aligned],
                    num_last_dim);
                PipeBarrier<PIPE_V>();
            }
        } else {
            Cast(y_buf_local, bias_local, RoundMode::CAST_NONE, num_last_dim);
            PipeBarrier<PIPE_V>();
            for (int i = 0; i < row_count; i++) {
                Add(add_buf_local[i * num_last_dim_aligned], y_buf_local, add_buf_local[i * num_last_dim_aligned],
                    num_last_dim);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void CopyInAddBiasNormal(int32_t proc_id, int32_t row_count)
    {
        auto y_buf_local = y_buf_fp32.Get<float>();
        auto add_buf_local = z_buf_fp32.Get<float>();
        uint32_t gm_offset = proc_id * row_step * num_last_dim;
        auto elementCount = num_last_dim_aligned * row_count;
        DataCopyPadParams padParams;
        if (last_dim_pad) {
            padParams.isPad = true;
            padParams.paddingValue = 0;
            padParams.rightPadding = num_last_dim_aligned - num_last_dim;
        }
        LocalTensor<T> x3_local_in = x1_que.template AllocTensor<T>();
        DataCopyEx(x3_local_in, bias_gm[gm_offset], num_last_dim, row_count, padParams);
        x1_que.EnQue(x3_local_in);
        auto x3_local = x1_que.template DeQue<T>();
        if constexpr (is_same<float, T>::value) {
            Add(add_buf_local, x3_local, add_buf_local, elementCount);
            PipeBarrier<PIPE_V>();
        } else {
            Cast(y_buf_local, x3_local, RoundMode::CAST_NONE, elementCount);
            PipeBarrier<PIPE_V>();
            Add(add_buf_local, y_buf_local, add_buf_local, elementCount);
            PipeBarrier<PIPE_V>();
        }
        x1_que.FreeTensor(x3_local);
    }

    __aicore__ inline void CopyInAddSingleRow(int32_t row_idx, int32_t size)
    {
        event_t event_mte3_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        auto add_buf_local = x_buf_fp32.Get<float>();
        auto y_buf_local = y_buf_fp32.Get<float>();
        uint32_t gm_offset = row_idx * row_step * num_last_dim;
        LocalTensor<T> x1_local_in = x1_que.template AllocTensor<T>();
        if constexpr (IS_X1_NEEDCAST) {
            SetFlag<HardEvent::MTE3_S>(event_mte3_s);
            WaitFlag<HardEvent::MTE3_S>(event_mte3_s);
            auto y_local_buffer = y_buf_local.template ReinterpretCast<T_X1>();
            DataCopyEx(y_local_buffer, x1_gm[gm_offset], size);
        } else {
            DataCopyEx(x1_local_in, x1_gm[gm_offset], size);
        }
        x1_que.EnQue(x1_local_in);
        auto x1_local = x1_que.template DeQue<T>();
        if constexpr (is_same<float, T>::value) {
            if constexpr (IS_X1_NEEDCAST) {
                auto y_local_buffer = y_buf_local.template ReinterpretCast<T_X1>();
                Cast(add_buf_local, y_local_buffer, RoundMode::CAST_NONE, size);
            } else {
                Adds(add_buf_local, x1_local, ZERO, size);
            }
            PipeBarrier<PIPE_V>();
        } else {
            Cast(add_buf_local, x1_local, RoundMode::CAST_NONE, size);
        }
        x1_que.FreeTensor(x1_local);

        if constexpr (IS_SINGLE_ROW_EXT_CASE) {
            LocalTensor<T> x2_local_in = x1_que.template AllocTensor<T>();
            if constexpr (IS_X2_NEEDCAST) {
                SetFlag<HardEvent::MTE3_S>(event_mte3_s);
                WaitFlag<HardEvent::MTE3_S>(event_mte3_s);
                auto y_local_buffer = y_buf_local.template ReinterpretCast<T_X2>();
                DataCopyEx(y_local_buffer, x2_gm[gm_offset], size);
            } else {
                DataCopyEx(x2_local_in, x2_gm[gm_offset], size);
            }
            x1_que.EnQue(x2_local_in);
            auto x2_local = x1_que.template DeQue<T>();
            if constexpr (is_same<float, T>::value) {
                if constexpr (IS_X2_NEEDCAST) {
                    auto y_local_buffer = y_buf_local.template ReinterpretCast<T_X2>();
                    Cast(x2_local, y_local_buffer, RoundMode::CAST_NONE, size);
                    PipeBarrier<PIPE_V>();
                }
                Add(add_buf_local, x2_local, add_buf_local, size);
                PipeBarrier<PIPE_V>();
            } else {
                Cast(y_buf_local, x2_local, RoundMode::CAST_NONE, size);
                PipeBarrier<PIPE_V>();
                Add(add_buf_local, y_buf_local, add_buf_local, size);
                PipeBarrier<PIPE_V>();
            }
            x1_que.FreeTensor(x2_local);
        } else {
            LocalTensor<T> x2_local_in = x2_que.template AllocTensor<T>();
            if constexpr (IS_X2_NEEDCAST) {
                SetFlag<HardEvent::MTE3_S>(event_mte3_s);
                WaitFlag<HardEvent::MTE3_S>(event_mte3_s);
                auto y_local_buffer = y_buf_local.template ReinterpretCast<T_X2>();
                DataCopyEx(y_local_buffer, x2_gm[gm_offset], size);
            } else {
                DataCopyEx(x2_local_in, x2_gm[gm_offset], size);
            }
            x2_que.EnQue(x2_local_in);
            auto x2_local_2 = x2_que.template DeQue<T>();
            if constexpr (is_same<float, T>::value) {
                if constexpr (IS_X2_NEEDCAST) {
                    auto y_local_buffer_2 = y_buf_local.template ReinterpretCast<T_X2>();
                    Cast(x2_local_2, y_local_buffer_2, RoundMode::CAST_NONE, size);
                    PipeBarrier<PIPE_V>();
                }
                Add(add_buf_local, x2_local_2, add_buf_local, size);
                PipeBarrier<PIPE_V>();
            } else {
                Cast(y_buf_local, x2_local_2, RoundMode::CAST_NONE, size);
                PipeBarrier<PIPE_V>();
                Add(add_buf_local, y_buf_local, add_buf_local, size);
                PipeBarrier<PIPE_V>();
            }
            x2_que.FreeTensor(x2_local_2);
        }

        if constexpr (IS_BIAS_PRESENT || IS_BIAS_BROADCAST) {
            LocalTensor<T> x3_local_in = x1_que.template AllocTensor<T>();
            if constexpr (IS_BIAS_PRESENT) {
                DataCopyEx(x3_local_in, bias_gm[gm_offset], size);
            } else if constexpr (IS_BIAS_BROADCAST) {
                DataCopyEx(x3_local_in, bias_gm, size);
            }
            x1_que.EnQue(x3_local_in);
            auto x3_local = x1_que.template DeQue<T>();
            if constexpr (is_same<float, T>::value) {
                Add(add_buf_local, x3_local, add_buf_local, size);
                PipeBarrier<PIPE_V>();
            } else {
                Cast(y_buf_local, x3_local, RoundMode::CAST_NONE, size);
                PipeBarrier<PIPE_V>();
                Add(add_buf_local, y_buf_local, add_buf_local, size);
                PipeBarrier<PIPE_V>();
            }
            x1_que.FreeTensor(x3_local);
        }

        if constexpr (IS_ADDITIONAL_OUTPUT_ENABLE) {
            if constexpr (IS_SINGLE_ROW_EXT_CASE) {
                auto x_local = y_que.template AllocTensor<T>();
                if constexpr (is_same<T, float>::value) {
                    Adds(x_local, add_buf_local, ZERO, size);
                } else if constexpr (is_same<T, half>::value) {
                    Cast(x_local, add_buf_local, RoundMode::CAST_NONE, size);
                } else {
                    Cast(x_local, add_buf_local, RoundMode::CAST_RINT, size);
                }
                PipeBarrier<PIPE_V>();
                y_que.template EnQue<T>(x_local);

                auto x = y_que.template DeQue<T>();
                DataCopyEx(x_gm[gm_offset], x, size);
                y_que.FreeTensor(x);
            } else {
                auto x_local = x_que.template AllocTensor<T>();
                if constexpr (is_same<T, float>::value) {
                    Adds(x_local, add_buf_local, ZERO, size);
                } else if constexpr (is_same<T, half>::value) {
                    Cast(x_local, add_buf_local, RoundMode::CAST_NONE, size);
                } else {
                    Cast(x_local, add_buf_local, RoundMode::CAST_RINT, size);
                }
                PipeBarrier<PIPE_V>();
                x_que.template EnQue<T>(x_local);

                auto x = x_que.template DeQue<T>();
                DataCopyEx(x_gm[gm_offset], x, size);
                x_que.FreeTensor(x);
            }
        }
    }

    __aicore__ inline void precision_compute(
        int32_t nums, LocalTensor<T_GAMMA>& gamma_local, LocalTensor<T_GAMMA>& beta_local)
    {
        LocalTensor<T> y_local = y_que.template AllocTensor<T>();
#if OUTPUT_MEAN_RSTD == 1
        LocalTensor<float> mean_local = mean_que.template AllocTensor<float>();
        LocalTensor<float> rstd_local = rstd_que.template AllocTensor<float>();
#endif
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        for (int32_t rid = 0; rid < nums; ++rid) {
            auto roundOffset = rid * num_last_dim_aligned;
            Muls(y_local_fp32, z_local_fp32[roundOffset], aveNum, num_last_dim);
            PipeBarrier<PIPE_V>();
            auto ave_local_temp = ReduceSumFP32(y_local_fp32, num_last_dim);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Adds(z_local_fp32[roundOffset], z_local_fp32[roundOffset], ave_local_temp * -1, num_last_dim);
            PipeBarrier<PIPE_V>();
            Mul(x_local_fp32, z_local_fp32[roundOffset], z_local_fp32[roundOffset], num_last_dim);
            PipeBarrier<PIPE_V>();
            Muls(y_local_fp32, x_local_fp32, aveNum, num_last_dim);
            PipeBarrier<PIPE_V>();
            float var_local_temp = ReduceSumFP32(y_local_fp32, num_last_dim);
            float rstd_local_temp = 1 / sqrt(var_local_temp + eps);
#if OUTPUT_MEAN_RSTD == 1
            mean_local.SetValue(rid, ave_local_temp);
            rstd_local.SetValue(rid, rstd_local_temp);
#endif
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Muls(x_local_fp32, z_local_fp32[roundOffset], rstd_local_temp, num_last_dim);
            PipeBarrier<PIPE_V>();
            if constexpr (IS_BETAGAMMA_NEEDCAST) {
                if constexpr (!is_same<T, float>::value) {
                    Cast(z_local_fp32[roundOffset], gamma_local, RoundMode::CAST_NONE, num_last_dim);
                    PipeBarrier<PIPE_V>();
                    Mul(y_local_fp32, x_local_fp32, z_local_fp32[roundOffset], num_last_dim);
                    Cast(x_local_fp32, beta_local, RoundMode::CAST_NONE, num_last_dim);
                    PipeBarrier<PIPE_V>();
                    Add(z_local_fp32[roundOffset], y_local_fp32, x_local_fp32, num_last_dim);
                    PipeBarrier<PIPE_V>();
                    if constexpr (is_same<T, half>::value) {
                        Cast(y_local[roundOffset], z_local_fp32[roundOffset], RoundMode::CAST_NONE, num_last_dim);
                    } else {
                        Cast(y_local[roundOffset], z_local_fp32[roundOffset], RoundMode::CAST_RINT, num_last_dim);
                    }
                    PipeBarrier<PIPE_V>();
                } else {
                    Cast(z_local_fp32[roundOffset], gamma_local, RoundMode::CAST_NONE, num_last_dim);
                    PipeBarrier<PIPE_V>();
                    Mul(y_local_fp32, x_local_fp32, z_local_fp32[roundOffset], num_last_dim);
                    Cast(x_local_fp32, beta_local, RoundMode::CAST_NONE, num_last_dim);
                    PipeBarrier<PIPE_V>();
                    Add(y_local[roundOffset], y_local_fp32, x_local_fp32, num_last_dim);
                    PipeBarrier<PIPE_V>();
                }
            } else {
                if constexpr (!is_same<T, float>::value) {
                    Mul(y_local_fp32, x_local_fp32, gamma_local, num_last_dim);
                    PipeBarrier<PIPE_V>();
                    Add(z_local_fp32[roundOffset], y_local_fp32, beta_local, num_last_dim);
                    PipeBarrier<PIPE_V>();
                    if constexpr (is_same<T, half>::value) {
                        Cast(y_local[roundOffset], z_local_fp32[roundOffset], RoundMode::CAST_NONE, num_last_dim);
                    } else {
                        Cast(y_local[roundOffset], z_local_fp32[roundOffset], RoundMode::CAST_RINT, num_last_dim);
                    }
                    PipeBarrier<PIPE_V>();
                } else {
                    Mul(y_local_fp32, x_local_fp32, gamma_local, num_last_dim);
                    PipeBarrier<PIPE_V>();
                    Add(y_local[roundOffset], y_local_fp32, beta_local, num_last_dim);
                    PipeBarrier<PIPE_V>();
                }
            }
        }
#if OUTPUT_MEAN_RSTD == 1
        mean_que.EnQue(mean_local);
        rstd_que.EnQue(rstd_local);
#endif
        y_que.EnQue(y_local);
    }

    __aicore__ inline void precision_compute_big_n(
        int32_t nums, LocalTensor<T_GAMMA>& gamma_local, LocalTensor<T_GAMMA>& beta_local)
    {
        LocalTensor<T> y_local = y_que.template AllocTensor<T>();
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
#if OUTPUT_MEAN_RSTD == 1
        auto mean_local = mean_que.template AllocTensor<float>();
        auto rstd_local = rstd_que.template AllocTensor<float>();
#else
        auto mean_local = x_local_fp32[nums * ONE_BLK_FLOAT_NUM];
        auto rstd_local = x_local_fp32[nums * ONE_BLK_FLOAT_NUM];
#endif
        int32_t elementNum = num_last_dim_aligned * nums;
        Muls(y_local_fp32, z_local_fp32, aveNum, elementNum);
        PipeBarrier<PIPE_V>();
        ReduceSumShort(mean_local, y_local_fp32, x_local_fp32, num_last_dim_aligned, num_last_dim, nums);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (int32_t idx = nums - 1; idx >= 0; idx--) {
            uint32_t offset = idx * num_last_dim_aligned;
            float meanTmp = mean_local.GetValue(idx);
            Adds(z_local_fp32[offset], z_local_fp32[offset], meanTmp * (-1), num_last_dim);
        }
        PipeBarrier<PIPE_V>();
        Mul(y_local_fp32, z_local_fp32, z_local_fp32, elementNum);
        PipeBarrier<PIPE_V>();
        Muls(y_local_fp32, y_local_fp32, aveNum, elementNum);
        PipeBarrier<PIPE_V>();
        ReduceSumShort(rstd_local, y_local_fp32, x_local_fp32, num_last_dim_aligned, num_last_dim, nums);
        PipeBarrier<PIPE_V>();
        Adds(rstd_local, rstd_local, eps, nums);
        PipeBarrier<PIPE_V>();
        Sqrt(rstd_local, rstd_local, nums);
        Duplicate(y_local_fp32, (float)1, nums);
        PipeBarrier<PIPE_V>();
        Div(rstd_local, y_local_fp32, rstd_local, nums);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (int32_t idx = nums - 1; idx >= 0; idx--) {
            uint32_t offset = idx * num_last_dim_aligned;
            float rstdTmp = rstd_local.GetValue(idx);
            Muls(z_local_fp32[offset], z_local_fp32[offset], rstdTmp, num_last_dim);
        }
        PipeBarrier<PIPE_V>();

        if constexpr (IS_BETAGAMMA_NEEDCAST) {
            if constexpr (!is_same<T, float>::value) {
                Cast(x_local_fp32, gamma_local, RoundMode::CAST_NONE, num_last_dim);
                Cast(y_local_fp32, beta_local, RoundMode::CAST_NONE, num_last_dim);
                PipeBarrier<PIPE_V>();
                for (auto i = 0; i < nums; i++) {
                    Mul(z_local_fp32[i * num_last_dim_aligned], z_local_fp32[i * num_last_dim_aligned], x_local_fp32,
                        num_last_dim);
                }
                PipeBarrier<PIPE_V>();
                for (auto i = 0; i < nums; i++) {
                    Add(z_local_fp32[i * num_last_dim_aligned], z_local_fp32[i * num_last_dim_aligned], y_local_fp32,
                        num_last_dim);
                }
                PipeBarrier<PIPE_V>();
                if constexpr (is_same<T, half>::value) {
                    Cast(y_local, z_local_fp32, RoundMode::CAST_NONE, elementNum);
                } else {
                    Cast(y_local, z_local_fp32, RoundMode::CAST_RINT, elementNum);
                }
            } else {
                Cast(x_local_fp32, gamma_local, RoundMode::CAST_NONE, num_last_dim);
                Cast(y_local_fp32, beta_local, RoundMode::CAST_NONE, num_last_dim);
                PipeBarrier<PIPE_V>();
                for (auto i = 0; i < nums; i++) {
                    Mul(z_local_fp32[i * num_last_dim_aligned], z_local_fp32[i * num_last_dim_aligned], x_local_fp32,
                        num_last_dim);
                }
                PipeBarrier<PIPE_V>();
                for (auto i = 0; i < nums; i++) {
                    Add(y_local[i * num_last_dim_aligned], z_local_fp32[i * num_last_dim_aligned], y_local_fp32,
                        num_last_dim);
                }
                PipeBarrier<PIPE_V>();
            }
        } else {
            if constexpr (!is_same<T, float>::value) {
                for (auto i = 0; i < nums; i++) {
                    Mul(z_local_fp32[i * num_last_dim_aligned], z_local_fp32[i * num_last_dim_aligned], gamma_local,
                        num_last_dim);
                }
                PipeBarrier<PIPE_V>();
                for (auto i = 0; i < nums; i++) {
                    Add(z_local_fp32[i * num_last_dim_aligned], z_local_fp32[i * num_last_dim_aligned], beta_local,
                        num_last_dim);
                }
                PipeBarrier<PIPE_V>();
                if constexpr (is_same<T, half>::value) {
                    Cast(y_local, z_local_fp32, RoundMode::CAST_NONE, elementNum);
                } else {
                    Cast(y_local, z_local_fp32, RoundMode::CAST_RINT, elementNum);
                }
            } else {
                for (auto i = 0; i < nums; i++) {
                    Mul(y_local[i * num_last_dim_aligned], z_local_fp32[i * num_last_dim_aligned], gamma_local,
                        num_last_dim);
                }
                PipeBarrier<PIPE_V>();
                for (auto i = 0; i < nums; i++) {
                    Add(y_local[i * num_last_dim_aligned], y_local[i * num_last_dim_aligned], beta_local, num_last_dim);
                }
                PipeBarrier<PIPE_V>();
            }
        }

#if OUTPUT_MEAN_RSTD == 1
        mean_que.EnQue(mean_local);
        rstd_que.EnQue(rstd_local);
#endif
        y_que.EnQue(y_local);
    }

    __aicore__ inline void precision_compute_single_row(
        LocalTensor<T_GAMMA>& gamma_local, LocalTensor<T_GAMMA>& beta_local)
    {
#if OUTPUT_MEAN_RSTD == 1
        LocalTensor<float> mean_local = mean_que.template AllocTensor<float>();
        LocalTensor<float> rstd_local = rstd_que.template AllocTensor<float>();
#endif
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        Muls(y_local_fp32, x_local_fp32, aveNum, num_last_dim);
        PipeBarrier<PIPE_V>();
        float ave_local_temp = ReduceSumFP32(y_local_fp32, num_last_dim);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Adds(x_local_fp32, x_local_fp32, ave_local_temp * -1, num_last_dim);
        PipeBarrier<PIPE_V>();
        Mul(y_local_fp32, x_local_fp32, x_local_fp32, num_last_dim);
        PipeBarrier<PIPE_V>();
        Muls(y_local_fp32, y_local_fp32, aveNum, num_last_dim);
        PipeBarrier<PIPE_V>();
        float var_local_temp = ReduceSumFP32(y_local_fp32, num_last_dim);
        float rstd_local_temp = 1 / sqrt(var_local_temp + eps);
#if OUTPUT_MEAN_RSTD == 1
        mean_local.SetValue(0, ave_local_temp);
        rstd_local.SetValue(0, rstd_local_temp);
#endif
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Muls(x_local_fp32, x_local_fp32, rstd_local_temp, num_last_dim);
        PipeBarrier<PIPE_V>();
        LocalTensor<T> y_local = y_que.template AllocTensor<T>();

        if constexpr (IS_BETAGAMMA_NEEDCAST) {
            if constexpr (!is_same<T, float>::value) {
                Cast(y_local_fp32, gamma_local, RoundMode::CAST_NONE, num_last_dim);
                PipeBarrier<PIPE_V>();
                Mul(x_local_fp32, x_local_fp32, y_local_fp32, num_last_dim);
                Cast(y_local_fp32, beta_local, RoundMode::CAST_NONE, num_last_dim);
                PipeBarrier<PIPE_V>();
                Add(x_local_fp32, x_local_fp32, y_local_fp32, num_last_dim);
                PipeBarrier<PIPE_V>();
                if constexpr (is_same<T, half>::value) {
                    Cast(y_local, x_local_fp32, RoundMode::CAST_NONE, num_last_dim);
                } else {
                    Cast(y_local, x_local_fp32, RoundMode::CAST_RINT, num_last_dim);
                }
            } else {
                Cast(y_local_fp32, gamma_local, RoundMode::CAST_NONE, num_last_dim);
                PipeBarrier<PIPE_V>();
                Mul(x_local_fp32, x_local_fp32, y_local_fp32, num_last_dim);
                Cast(y_local_fp32, beta_local, RoundMode::CAST_NONE, num_last_dim);
                PipeBarrier<PIPE_V>();
                Add(y_local, x_local_fp32, y_local_fp32, num_last_dim);
                PipeBarrier<PIPE_V>();
            }
        } else {
            if constexpr (!is_same<T, float>::value) {
                Mul(x_local_fp32, x_local_fp32, gamma_local, num_last_dim);
                PipeBarrier<PIPE_V>();
                Add(x_local_fp32, x_local_fp32, beta_local, num_last_dim);
                PipeBarrier<PIPE_V>();
                if constexpr (is_same<T, half>::value) {
                    Cast(y_local, x_local_fp32, RoundMode::CAST_NONE, num_last_dim);
                } else {
                    Cast(y_local, x_local_fp32, RoundMode::CAST_RINT, num_last_dim);
                }
            } else {
                Mul(y_local_fp32, x_local_fp32, gamma_local, num_last_dim);
                PipeBarrier<PIPE_V>();
                Add(y_local, y_local_fp32, beta_local, num_last_dim);
            }
        }

        y_que.EnQue(y_local);
#if OUTPUT_MEAN_RSTD == 1
        mean_que.EnQue(mean_local);
        rstd_que.EnQue(rstd_local);
#endif
    }

    __aicore__ inline void CopyOut(int32_t row_idx, int32_t row_count)
    {
        LocalTensor<T> res = y_que.template DeQue<T>();
        uint32_t gm_offset = row_idx * row_step * num_last_dim;
        DataCopyEx(y_gm[gm_offset], res, num_last_dim, row_count);
        y_que.FreeTensor(res);

#if OUTPUT_MEAN_RSTD == 1
        uint32_t gm_offset_mean = row_idx * row_step;
        LocalTensor<float> mean = mean_que.template DeQue<float>();
        LocalTensor<float> rstd = rstd_que.template DeQue<float>();
        DataCopyEx(mean_gm[gm_offset_mean], mean, row_count);
        DataCopyEx(rstd_gm[gm_offset_mean], rstd, row_count);
        mean_que.FreeTensor(mean);
        rstd_que.FreeTensor(rstd);
#endif
    }

    __aicore__ inline void CopyOutSlicePhase0(int32_t row_idx, int32_t size, int32_t offset = 0)
    {
        LocalTensor<T> x = x_que.template DeQue<T>();
        uint32_t gm_offset = row_idx * row_step * num_last_dim + offset;
        DataCopyEx(x_gm[gm_offset], x, size);
        x_que.FreeTensor(x);
    }

    __aicore__ inline void CopyOutSlicePhase1(int32_t row_idx, int32_t size, int32_t offset = 0)
    {
        LocalTensor<T> res = y_que.template DeQue<T>();
        uint32_t gm_offset = row_idx * row_step * num_last_dim + offset;
        DataCopyEx(y_gm[gm_offset], res, size);
        y_que.FreeTensor(res);
    }

    __aicore__ inline void CopyOutSlicePhase2(int32_t row_idx)
    {
#if OUTPUT_MEAN_RSTD == 1
        LocalTensor<float> mean = mean_que.template DeQue<float>();
        LocalTensor<float> rstd = rstd_que.template DeQue<float>();
        DataCopyEx(mean_gm[row_idx * row_step], mean, 1);
        DataCopyEx(rstd_gm[row_idx * row_step], rstd, 1);
        mean_que.FreeTensor(mean);
        rstd_que.FreeTensor(rstd);
#endif
    }

private:
    TPipe* Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> gamma_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> beta_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> bias_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> x_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> y_que;
#if OUTPUT_MEAN_RSTD == 1
    TQue<QuePosition::VECOUT, BUFFER_NUM> mean_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rstd_que;
#endif
    TBuf<TPosition::VECCALC> x_buf_fp32;
    TBuf<TPosition::VECCALC> y_buf_fp32;
    TBuf<TPosition::VECCALC> z_buf_fp32;
    GlobalTensor<T_X1> x1_gm;
    GlobalTensor<T_X2> x2_gm;
    GlobalTensor<T_GAMMA> gamma_gm;
    GlobalTensor<T_GAMMA> beta_gm;
    GlobalTensor<T> bias_gm;
    GlobalTensor<T> y_gm;
    GlobalTensor<T> x_gm;
    GlobalTensor<float> mean_gm;
    GlobalTensor<float> rstd_gm;
    GlobalTensor<float> workspace_gm;
    uint32_t num_core;
    uint32_t num_first_dim;
    uint32_t num_last_dim;
    uint32_t row_step;
    uint32_t row_work;
    uint32_t gm_offset_;
    uint32_t row_tail_;
    uint32_t col_tail;
    uint32_t col_move_cnt;
    uint32_t first_dim_per_time;
    uint32_t last_dim_per_time;
    uint32_t nl_first_dim_per_core;
    uint32_t l_first_dim_per_core;
    float eps;
    float aveNum;
    bool last_dim_pad = false;
    size_t num_last_dim_aligned;
    bool lastDimPadMixDtype = false;
    size_t numLastDimAlignedMixDtype;
};

#endif // __ADD_LAYER_NORM_KERNEL_H_