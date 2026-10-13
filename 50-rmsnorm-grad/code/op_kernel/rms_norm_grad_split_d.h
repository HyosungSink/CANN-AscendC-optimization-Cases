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
 * \file rms_norm_grad_split_d.h
 * \brief
 */

#ifndef RMS_NORM_GRAD_SPLIT_D_H
#define RMS_NORM_GRAD_SPLIT_D_H
#include "rms_norm_grad_common.h"
template <typename T1>
class RmsNormGradSplitD {
public:
    __aicore__ inline RmsNormGradSplitD()
    {}

    __aicore__ inline void Init(
        GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma,
        const RmsNormGradTilingData* tiling, GM_ADDR usrWorkspace)
    {
        InitVar(tiling);
        InitInputGmBuffer(dy, x, rstd, gamma, block_dim, core_calc_num, core_calc_tail);
        InitOutputGmBuffer(dx, dgamma, block_dim, core_calc_num, core_calc_tail);
        if (fixed_output == 1) {
            InitWorkspace(usrWorkspace);
        }
        InitInputQue();
        InitOutputQue();
        InitTmpBuffer();
    }

    __aicore__ inline void InitVar(const RmsNormGradTilingData* tiling)
    {
        block_dim = tiling->block_dim;
        row_val = tiling->row;
        col_val = tiling->col;
        avg_factor = tiling->avg_factor;
        data_type = tiling->data_type;
        core_calc_num = tiling->core_calc_num;
        core_calc_tail = tiling->core_calc_tail;
        block_factor = tiling->block_factor;
        ub_factor = tiling->ub_factor;
        ub_calc_num = tiling->ub_calc_num;
        ub_calc_tail = tiling->ub_calc_tail;
        ub_calc_loop = tiling->ub_calc_loop;
        ub_calc_tail_num = tiling->ub_calc_tail_num;
        ub_calc_tail_tail = tiling->ub_calc_tail_tail;
        ub_calc_tail_loop = tiling->ub_calc_tail_loop;
        align_len = data_type == FLOAT_DTYPE ? ALIGN_32 : ALIGN_16;
        col_val_align = (col_val + align_len - 1) / align_len * align_len;
        fixed_output = tiling->fixed_output;
    }

    __aicore__ inline void InitInputGmBuffer(
        GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, uint32_t block_dim, uint32_t core_calc_num,
        uint32_t core_calc_tail)
    {
        if (GetBlockIdx() < block_dim - 1) {
            core_offset = block_factor;
        } else {
            core_offset = core_calc_tail > 0 ? core_calc_tail : block_factor;
        }
        core_offset_start = block_factor * col_val;
        core_offset_len = core_offset * col_val;
        dyGm.SetGlobalBuffer((__gm__ T1*)dy + GetBlockIdx() * core_offset_start, core_offset_len);
        xGm.SetGlobalBuffer((__gm__ T1*)x + GetBlockIdx() * core_offset_start, core_offset_len);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * block_factor, core_offset);
        gammaGm.SetGlobalBuffer((__gm__ T1*)gamma, col_val);
    }

    __aicore__ inline void InitOutputGmBuffer(
        GM_ADDR dx, GM_ADDR dgamma, uint32_t block_dim, uint32_t core_calc_num, uint32_t core_calc_tail)
    {
        dxGm.SetGlobalBuffer((__gm__ T1*)dx + GetBlockIdx() * core_offset_start, core_offset_len);
        dgammaGm.SetGlobalBuffer((__gm__ float*)dgamma, col_val);
        if (fixed_output != 1 && GetBlockIdx() == 0) {
            InitOutput<float>(dgammaGm, col_val, 0);
        }
    }

    __aicore__ inline void InitInputQue()
    {
        ub_factor_align = (ub_factor + align_len - 1) / align_len * align_len;
        rstd_len = align_len;
        buffer_len_size = ub_factor_align * sizeof(T1);
        buffer_num = data_type != FLOAT_DTYPE ? BUFFER_NUM_DB : BUFFER_NUM;
        pipe.InitBuffer(inQueDY, buffer_num, buffer_len_size);
        pipe.InitBuffer(inQueX, buffer_num, buffer_len_size);
        pipe.InitBuffer(inQueRstd, buffer_num, rstd_len * sizeof(float));
        pipe.InitBuffer(inQueGamma, buffer_num, buffer_len_size);
    }

    __aicore__ inline void InitOutputQue()
    {
        pipe.InitBuffer(outQueDX, buffer_num, buffer_len_size);
        pipe.InitBuffer(outQueDgamma, buffer_num, ub_factor_align * sizeof(float));
    }

    __aicore__ inline void InitTmpBuffer()
    {
        uint32_t tmp_buffer_size = ub_factor_align * sizeof(float);
        pipe.InitBuffer(ndBufFp32Buf1, tmp_buffer_size);
        pipe.InitBuffer(nFp32Buf, align_len * sizeof(float));
        if (data_type != FLOAT_DTYPE) {
            pipe.InitBuffer(ndBufFp32Buf2, tmp_buffer_size);
            pipe.InitBuffer(ndBufFp32Buf3, tmp_buffer_size);
            pipe.InitBuffer(dFp32Buf, tmp_buffer_size);
        }
        if (fixed_output != 1) {
            SyncAll();
        }
    }

    __aicore__ inline void InitWorkspace(GM_ADDR usrWorkspace)
    {
        workspace_gm.SetGlobalBuffer((__gm__ float*)usrWorkspace + GetBlockIdx() * col_val);
    }

    __aicore__ inline void Process()
    {
        if (core_calc_tail == 0) {
            ProcessMain(block_factor);
        } else {
            if (GetBlockIdx() < block_dim - 1) {
                ProcessMain(block_factor);
            } else {
                ProcessMain(core_calc_tail);
            }
        }
    }

    __aicore__ inline void ProcessMain(uint32_t loop_len)
    {
        for (uint32_t i = 0; i < loop_len; i++) {
            // Calc mean firstly
            LocalTensor<float> dy_sum = nFp32Buf.Get<float>();
            Duplicate(dy_sum, 0.0f, align_len);
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < (ub_calc_tail == 0 ? ub_calc_loop : ub_calc_loop - 1); j++) {
                ComputeDySum(i, j, ub_factor, ub_factor, dy_sum);
            }
            if (ub_calc_tail != 0) {
                ub_tail_align = (ub_calc_tail + align_len - 1) / align_len * align_len;
                ComputeDySum(i, ub_calc_loop - 1, ub_calc_tail, ub_tail_align, dy_sum);
            }
            Muls(dy_sum, dy_sum, avg_factor, align_len);
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < (ub_calc_tail == 0 ? ub_calc_loop : ub_calc_loop - 1); j++) {
                CopyGammaIn(j, ub_factor);
                LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
                LocalTensor<float> dgamma;
                CopyIn(i, j, ub_factor, ub_factor);
                ComputeMain(i, j, ub_factor, gamma_ub, dgamma, dy_sum);
                CopyOut(i, j, ub_factor, ub_factor);
            }
            if (ub_calc_tail != 0) {
                CopyGammaIn(ub_calc_loop - 1, ub_calc_tail);
                LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
                LocalTensor<float> dgamma;
                CopyIn(i, ub_calc_loop - 1, ub_calc_tail, ub_tail_align);
                ComputeMain(i, ub_calc_loop - 1, ub_calc_tail, gamma_ub, dgamma, dy_sum);
                CopyOut(i, ub_calc_loop - 1, ub_calc_tail, ub_tail_align);
            }
        }
        ComputeDgammaMain(loop_len);
    }

    __aicore__ inline void ComputeDgammaMain(uint32_t loop_len)
    {
        for (uint32_t j = 0; j < (ub_calc_tail == 0 ? ub_calc_loop : ub_calc_loop - 1); j++) {
            CopyGammaIn(j, ub_factor);
            LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
            LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
            Duplicate(dgamma, 0.0f, ub_factor);
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < loop_len; i++) {
                CopyIn(i, j, ub_factor, ub_factor);
                if constexpr (is_same<T1, float>::value) {
                    ComputeDgammaFp32(i, j, ub_factor, gamma_ub, dgamma);
                } else if constexpr (is_same<T1, half>::value) {
                    ComputeDgammaFp16(i, j, ub_factor, gamma_ub, dgamma);
                } else {
                    ComputeDgammaBf16(i, j, ub_factor, gamma_ub, dgamma);
                }
            }
            inQueGamma.FreeTensor(gamma_ub);
            outQueDgamma.EnQue(dgamma);
            if (fixed_output == 1) {
                CopyDgammaOutWorkspace(j, ub_factor);
            } else {
                CopyDgammaOut(j, ub_factor);
            }
        }
        if (ub_calc_tail != 0) {
            CopyGammaIn(ub_calc_loop - 1, ub_calc_tail);
            LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
            LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
            Duplicate(dgamma, 0.0f, ub_calc_tail);
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < loop_len; i++) {
                CopyIn(i, ub_calc_loop - 1, ub_calc_tail, ub_tail_align);
                if constexpr (is_same<T1, float>::value) {
                    ComputeDgammaFp32(i, ub_calc_loop - 1, ub_calc_tail, gamma_ub, dgamma);
                } else if constexpr (is_same<T1, half>::value) {
                    ComputeDgammaFp16(i, ub_calc_loop - 1, ub_calc_tail, gamma_ub, dgamma);
                } else {
                    ComputeDgammaBf16(i, ub_calc_loop - 1, ub_calc_tail, gamma_ub, dgamma);
                }
            }
            inQueGamma.FreeTensor(gamma_ub);
            outQueDgamma.EnQue(dgamma);
            if (fixed_output == 1) {
                CopyDgammaOutWorkspace(ub_calc_loop - 1, ub_calc_tail);
            } else {
                CopyDgammaOut(ub_calc_loop - 1, ub_calc_tail);
            }
        }
        if (fixed_output == 1) {
            SyncAll();
            AddDgamma(ub_factor);
        }
    }

    __aicore__ inline void CopyDgammaOutWorkspace(uint32_t d_idx, uint32_t calc_len)
    {
        LocalTensor<float> dgamma_out = outQueDgamma.DeQue<float>();
        DataCopyParams data_copy_params{(uint16_t)1, (uint16_t)(calc_len * sizeof(float)), 0, 0};
        DataCopyPad(workspace_gm[d_idx * ub_factor], dgamma_out, data_copy_params);
        outQueDgamma.FreeTensor(dgamma_out);
    }

    __aicore__ inline void AddDgamma(uint32_t calc_len)
    {
        if (GetBlockIdx() != 0) {
            return;
        }
        LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
        LocalTensor<float> tmpBuf = inQueRstd.AllocTensor<float>();
        PipeBarrier<PIPE_ALL>();
        auto calcLenAlign = (calc_len + ALIGN_32 - 1) / ALIGN_32 * ALIGN_32;
        DataCopyParams dataCopyParams{(uint16_t)1, (uint16_t)(calc_len * sizeof(float)), 0, 0};
        DataCopyPadParams padParams{true, 0, (uint8_t)(calcLenAlign - calc_len), 0};
        for (uint32_t j = 0; j < (ub_calc_tail == 0 ? ub_calc_loop : ub_calc_loop - 1); j++) {
            for (uint32_t blockidx = 0; blockidx < block_dim; blockidx++) {
                if (blockidx == 0) {
                    DataCopyPad(dgamma, workspace_gm[calc_len * j], dataCopyParams, padParams);
                } else {
                    DataCopyPad(tmpBuf, workspace_gm[blockidx * col_val + calc_len * j], dataCopyParams, padParams);
                    PipeBarrier<PIPE_ALL>();
                    Add(dgamma, dgamma, tmpBuf, calc_len);
                    PipeBarrier<PIPE_ALL>();
                }
            }
            PipeBarrier<PIPE_ALL>();
            DataCopyParams dataCopyParams{(uint16_t)1, (uint16_t)(calc_len * sizeof(float)), 0, 0};
            DataCopyPad(dgammaGm[j * calc_len], dgamma, dataCopyParams);
            PipeBarrier<PIPE_ALL>();
        }
        if (ub_calc_tail != 0) {
            DataCopyParams tailCopyParams{(uint16_t)1, (uint16_t)(ub_calc_tail * sizeof(float)), 0, 0};
            auto tailLenAlign = (ub_calc_tail + ALIGN_32 - 1) / ALIGN_32 * ALIGN_32;
            DataCopyPadParams tailPadParams{true, 0, (uint8_t)(tailLenAlign - ub_calc_tail), 0};
            for (uint32_t blockidx = 0; blockidx < block_dim; blockidx++) {
                if (blockidx == 0) {
                    DataCopyPad(dgamma, workspace_gm[(ub_calc_loop - 1) * calc_len], tailCopyParams, tailPadParams);
                } else {
                    DataCopyPad(
                        tmpBuf, workspace_gm[blockidx * col_val + (ub_calc_loop - 1) * calc_len], tailCopyParams,
                        tailPadParams);
                    PipeBarrier<PIPE_ALL>();
                    Add(dgamma, dgamma, tmpBuf, ub_calc_tail);
                    PipeBarrier<PIPE_ALL>();
                }
            }
            PipeBarrier<PIPE_ALL>();
            DataCopyParams data_copy_params{(uint16_t)1, (uint16_t)(ub_calc_tail * sizeof(float)), 0, 0};
            DataCopyPad(dgammaGm[(ub_calc_loop - 1) * calc_len], dgamma, data_copy_params);
        }
        outQueDgamma.FreeTensor(dgamma);
        inQueRstd.FreeTensor(tmpBuf);
    }

    __aicore__ inline void ComputeDgammaFp32(
        uint32_t i, uint32_t j, uint32_t calc_len, LocalTensor<T1>& gamma_ub, LocalTensor<float>& dgamma)
    {
        LocalTensor<T1> x_ub = inQueX.DeQue<T1>();
        LocalTensor<T1> dy_ub = inQueDY.DeQue<T1>();
        LocalTensor<float> rstd_ub = inQueRstd.DeQue<float>();
        // dy * (x * rstd)  -> sum
        event_t event_mte_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(event_mte_s);
        WaitFlag<HardEvent::MTE2_S>(event_mte_s);
        float rstd_value = rstd_ub.GetValue(0);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        Muls(x_ub, x_ub, rstd_value, calc_len);
        PipeBarrier<PIPE_V>();
        inQueRstd.FreeTensor(rstd_ub);
        Mul(dy_ub, dy_ub, x_ub, calc_len);
        PipeBarrier<PIPE_V>();
        inQueX.FreeTensor(x_ub);
        Add(dgamma, dgamma, dy_ub, calc_len);
        PipeBarrier<PIPE_V>();
        inQueDY.FreeTensor(dy_ub);
    }

    __aicore__ inline void ComputeDySum(
        uint32_t i, uint32_t j, uint32_t calc_len, uint32_t calc_len_align, LocalTensor<float>& dy_sum)
    {
        CopyGammaIn(j, calc_len);
        CopyIn(i, j, calc_len, calc_len_align);
        LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
        LocalTensor<T1> x_ub = inQueX.DeQue<T1>();
        LocalTensor<T1> dy_ub = inQueDY.DeQue<T1>();
        LocalTensor<float> rstd_ub = inQueRstd.DeQue<float>();
        event_t event_mte_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(event_mte_s);
        WaitFlag<HardEvent::MTE2_S>(event_mte_s);
        float rstd_value = rstd_ub.GetValue(0);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        LocalTensor<float> dy_sum_part = ndBufFp32Buf1.Get<float>();
        Duplicate(dy_sum_part, 0.0f, ub_tail_align);
        PipeBarrier<PIPE_V>();
        // grad_y = dy * gamma
        Mul(dy_ub, dy_ub, gamma_ub, calc_len);
        Muls(x_ub, x_ub, rstd_value, calc_len);
        PipeBarrier<PIPE_V>();
        inQueGamma.FreeTensor(gamma_ub);
        inQueRstd.FreeTensor(rstd_ub);
        // sum(x * rstd * grad_y)
        Mul(dy_sum_part, dy_ub, x_ub, calc_len);
        PipeBarrier<PIPE_V>();
        inQueX.FreeTensor(x_ub);
        ReduceSumFP32(0, dy_sum_part, dy_sum_part, dy_ub, calc_len, col_val_align);
        inQueDY.FreeTensor(dy_ub);
        Add(dy_sum, dy_sum, dy_sum_part, align_len);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyGammaIn(uint32_t d_idx, uint32_t calc_len)
    {
        LocalTensor<T1> gamma_ub = inQueGamma.AllocTensor<T1>();
        DataCopyParams data_copy_params{(uint16_t)1, (uint16_t)(calc_len * sizeof(T1)), 0, 0};
        DataCopyPadParams pad_params{false, 0, 0, 0};
        DataCopyPad(gamma_ub, gammaGm[d_idx * ub_factor], data_copy_params, pad_params);
        inQueGamma.EnQue(gamma_ub);
    }

    __aicore__ inline void CopyDgammaOut(uint32_t d_idx, uint32_t calc_len)
    {
        LocalTensor<float> dgamma_out = outQueDgamma.DeQue<float>();
        SetAtomicAdd<float>();
        DataCopyParams data_copy_params{(uint16_t)1, (uint16_t)(calc_len * sizeof(float)), 0, 0};
        DataCopyPad(dgammaGm[d_idx * ub_factor], dgamma_out, data_copy_params);
        SetAtomicNone();
        outQueDgamma.FreeTensor(dgamma_out);
    }

    __aicore__ inline void CopyIn(uint32_t n_idx, uint32_t d_idx, uint32_t calc_unit, uint32_t calc_unit_align)
    {
        // x dy, rstd
        DataCopyParams data_copy_params{(uint16_t)1, (uint16_t)(calc_unit * sizeof(T1)), 0, 0};
        DataCopyPadParams pad_params{true, 0, (uint8_t)(calc_unit_align - calc_unit), 0};
        uint32_t gm_offset = n_idx * col_val + d_idx * ub_factor;
        LocalTensor<float> rstd_ub = inQueRstd.AllocTensor<float>();
        DataCopyParams data_copy_params_rstd{(uint16_t)1, (uint16_t)(1 * sizeof(float)), 0, 0};
        DataCopyPadParams pad_params_rstd{true, 0, (uint8_t)(0), 0};
        DataCopyPad(rstd_ub, rstdGm[n_idx], data_copy_params_rstd, pad_params_rstd);
        inQueRstd.EnQue(rstd_ub);
        LocalTensor<T1> x_ub = inQueX.AllocTensor<T1>();
        DataCopyPad(x_ub, xGm[gm_offset], data_copy_params, pad_params);
        inQueX.EnQue(x_ub);
        LocalTensor<T1> dy_ub = inQueDY.AllocTensor<T1>();
        DataCopyPad(dy_ub, dyGm[gm_offset], data_copy_params, pad_params);
        inQueDY.EnQue(dy_ub);
    }

    __aicore__ inline void CopyOut(uint32_t n_idx, uint32_t d_idx, uint32_t calc_unit, uint32_t calc_unit_align)
    {
        LocalTensor<T1> dx_ub = outQueDX.DeQue<T1>();
        DataCopyParams data_copy_params{(uint16_t)1, (uint16_t)(calc_unit * sizeof(T1)), 0, 0};
        DataCopyPad(dxGm[n_idx * col_val + d_idx * ub_factor], dx_ub, data_copy_params);
        outQueDX.FreeTensor(dx_ub);
    }

    __aicore__ inline void ComputeMain(
        uint32_t n_idx, uint32_t d_idx, uint32_t calc_len, LocalTensor<T1>& gamma_ub, LocalTensor<float>& dgamma,
        LocalTensor<float>& dy_sum)
    {
        LocalTensor<T1> x_ub = inQueX.DeQue<T1>();
        LocalTensor<T1> dy_ub = inQueDY.DeQue<T1>();
        LocalTensor<float> rstd_ub = inQueRstd.DeQue<float>();
        event_t event_mte_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(event_mte_s);
        WaitFlag<HardEvent::MTE2_S>(event_mte_s);
        float rstd_value = rstd_ub.GetValue(0);
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float dy_sum_val = dy_sum.GetValue(0);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);

        Muls(x_ub, x_ub, rstd_value, calc_len);
        PipeBarrier<PIPE_V>();
        inQueRstd.FreeTensor(rstd_ub);
        // grad_y = grad* gamma
        Mul(dy_ub, dy_ub, gamma_ub, calc_len);
        Muls(x_ub, x_ub, dy_sum_val, calc_len);
        PipeBarrier<PIPE_V>();
        inQueGamma.FreeTensor(gamma_ub);
        Sub(dy_ub, dy_ub, x_ub, calc_len);
        PipeBarrier<PIPE_V>();
        inQueX.FreeTensor(x_ub);
        LocalTensor<T1> dx_ub = outQueDX.AllocTensor<T1>();
        Muls(dx_ub, dy_ub, rstd_value, calc_len);
        PipeBarrier<PIPE_V>();
        inQueDY.FreeTensor(dy_ub);
        outQueDX.EnQue(dx_ub);
    }

    __aicore__ inline void ProcessFp16()
    {
        if (core_calc_tail == 0) {
            ProcessFp16Main(block_factor);
        } else {
            if (GetBlockIdx() < block_dim - 1) {
                ProcessFp16Main(block_factor);
            } else {
                ProcessFp16Main(core_calc_tail);
            }
        }
    }

    __aicore__ inline void ProcessFp16Main(uint32_t loop_len)
    {
        for (uint32_t i = 0; i < loop_len; i++) {
            // Calc mean firstly
            LocalTensor<float> dy_sum = nFp32Buf.Get<float>();
            Duplicate(dy_sum, 0.0f, align_len);
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < (ub_calc_tail == 0 ? ub_calc_loop : ub_calc_loop - 1); j++) {
                ComputeDySumFp16(i, j, ub_factor, ub_factor, dy_sum);
            }
            if (ub_calc_tail != 0) {
                ub_tail_align = (ub_calc_tail + align_len - 1) / align_len * align_len;
                ComputeDySumFp16(i, ub_calc_loop - 1, ub_calc_tail, ub_tail_align, dy_sum);
            }
            Muls(dy_sum, dy_sum, avg_factor, align_len);
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < (ub_calc_tail == 0 ? ub_calc_loop : ub_calc_loop - 1); j++) {
                CopyIn(i, j, ub_factor, ub_factor);
                CopyGammaIn(j, ub_factor);
                LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
                ComputeMainFp16(i, j, ub_factor, gamma_ub, dy_sum);
                if (fixed_output == 0) {
                    CopyDgammaOut(j, ub_factor);
                }
                CopyOut(i, j, ub_factor, ub_factor);
            }
            if (ub_calc_tail != 0) {
                CopyIn(i, ub_calc_loop - 1, ub_calc_tail, ub_tail_align);
                CopyGammaIn(ub_calc_loop - 1, ub_calc_tail);
                LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
                ComputeMainFp16(i, ub_calc_loop - 1, ub_calc_tail, gamma_ub, dy_sum);
                if (fixed_output == 0) {
                    CopyDgammaOut(ub_calc_loop - 1, ub_calc_tail);
                }
                CopyOut(i, ub_calc_loop - 1, ub_calc_tail, ub_tail_align);
            }
        }
        if (fixed_output == 1) {
            ComputeDgammaMain(loop_len);
        }
    }

    __aicore__ inline void ComputeDgammaFp16(
        uint32_t i, uint32_t j, uint32_t calc_len, LocalTensor<T1>& gamma_ub, LocalTensor<float>& dgamma)
    {
        LocalTensor<T1> x_ub = inQueX.DeQue<T1>();
        LocalTensor<T1> dy_ub = inQueDY.DeQue<T1>();
        LocalTensor<float> rstd_ub = inQueRstd.DeQue<float>();
        LocalTensor<float> tmp_32_buf_2 = ndBufFp32Buf3.Get<float>();
        // dy * (x * rstd)  -> sum
        event_t event_mte_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(event_mte_s);
        WaitFlag<HardEvent::MTE2_S>(event_mte_s);
        float rstd_value = rstd_ub.GetValue(0);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        Cast(tmp_32_buf_2, x_ub, RoundMode::CAST_NONE, calc_len);
        PipeBarrier<PIPE_V>();
        Muls(tmp_32_buf_2, tmp_32_buf_2, rstd_value, calc_len);
        PipeBarrier<PIPE_V>();
        inQueRstd.FreeTensor(rstd_ub);
        Cast(x_ub, tmp_32_buf_2, RoundMode::CAST_NONE, calc_len);
        PipeBarrier<PIPE_V>();
        Mul(dy_ub, dy_ub, x_ub, calc_len);
        PipeBarrier<PIPE_V>();
        inQueX.FreeTensor(x_ub);
        Cast(tmp_32_buf_2, dy_ub, RoundMode::CAST_NONE, calc_len);
        PipeBarrier<PIPE_V>();
        inQueDY.FreeTensor(dy_ub);
        Add(dgamma, dgamma, tmp_32_buf_2, calc_len);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeDySumFp16(
        uint32_t i, uint32_t j, uint32_t calc_len, uint32_t calc_len_align, LocalTensor<float>& dy_sum)
    {
        CopyGammaIn(j, calc_len);
        CopyIn(i, j, calc_len, calc_len_align);
        LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
        LocalTensor<T1> x_ub = inQueX.DeQue<T1>();
        LocalTensor<T1> dy_ub = inQueDY.DeQue<T1>();
        LocalTensor<float> rstd_ub = inQueRstd.DeQue<float>();
        event_t event_mte_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(event_mte_s);
        WaitFlag<HardEvent::MTE2_S>(event_mte_s);
        float rstd_value = rstd_ub.GetValue(0);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);

        LocalTensor<float> dy_sum_part = ndBufFp32Buf1.Get<float>();
        LocalTensor<float> tmp_32_buf = ndBufFp32Buf2.Get<float>();
        LocalTensor<float> tmp_32_buf_2 = ndBufFp32Buf3.Get<float>();

        Duplicate(dy_sum_part, 0.0f, ub_tail_align);
        // grad_y = dy * gamma
        Mul(dy_ub, dy_ub, gamma_ub, calc_len);
        Cast(tmp_32_buf_2, x_ub, RoundMode::CAST_NONE, calc_len);
        PipeBarrier<PIPE_V>();
        inQueGamma.FreeTensor(gamma_ub);
        inQueRstd.FreeTensor(rstd_ub);
        inQueX.FreeTensor(x_ub);
        Cast(tmp_32_buf, dy_ub, RoundMode::CAST_NONE, calc_len);
        PipeBarrier<PIPE_V>();
        inQueDY.FreeTensor(dy_ub);
        // sum(x * rstd * grad_y)
        Muls(tmp_32_buf_2, tmp_32_buf_2, rstd_value, calc_len);
        PipeBarrier<PIPE_V>();
        Mul(dy_sum_part, tmp_32_buf, tmp_32_buf_2, calc_len);
        PipeBarrier<PIPE_V>();
        ReduceSumFP32(0, dy_sum_part, dy_sum_part, tmp_32_buf_2, calc_len, col_val_align);
        Add(dy_sum, dy_sum, dy_sum_part, align_len);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeMainFp16(
        uint32_t n_idx, uint32_t d_idx, uint32_t calc_len, LocalTensor<T1>& gamma_ub, LocalTensor<float>& dy_sum)
    {
        LocalTensor<T1> x_ub = inQueX.DeQue<T1>();
        LocalTensor<T1> dy_ub = inQueDY.DeQue<T1>();
        LocalTensor<float> rstd_ub = inQueRstd.DeQue<float>();
        LocalTensor<float> tmp_32_buf_1 = ndBufFp32Buf1.Get<float>();
        LocalTensor<float> tmp_32_buf_2 = ndBufFp32Buf2.Get<float>();
        LocalTensor<float> tmp_32_buf_3 = ndBufFp32Buf3.Get<float>();
        event_t event_mte_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(event_mte_s);
        WaitFlag<HardEvent::MTE2_S>(event_mte_s);
        float rstd_value = rstd_ub.GetValue(0);
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float dy_sum_val = dy_sum.GetValue(0);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        // dg = sum((dy * (x * rstd)), dim=0)
        Cast(tmp_32_buf_2, x_ub, RoundMode::CAST_NONE, calc_len);
        PipeBarrier<PIPE_V>();
        inQueRstd.FreeTensor(rstd_ub);
        Muls(tmp_32_buf_2, tmp_32_buf_2, rstd_value, calc_len);
        PipeBarrier<PIPE_V>();
        if (fixed_output == 1) {
            Mul(dy_ub, dy_ub, gamma_ub, calc_len);
            PipeBarrier<PIPE_V>();
        } else {
            Cast(x_ub, tmp_32_buf_2, RoundMode::CAST_NONE, calc_len);
            PipeBarrier<PIPE_V>();
            Mul(x_ub, dy_ub, x_ub, calc_len);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
            Mul(dy_ub, dy_ub, gamma_ub, calc_len);
            Cast(dgamma, x_ub, RoundMode::CAST_NONE, calc_len);
            PipeBarrier<PIPE_V>();
            outQueDgamma.EnQue(dgamma);
        }
        inQueGamma.FreeTensor(gamma_ub);
        inQueX.FreeTensor(x_ub);
        // grad_y = grad* gamma
        Cast(tmp_32_buf_1, dy_ub, RoundMode::CAST_NONE, calc_len);
        // dx = (grad_y - y * mean) * rstd
        Muls(tmp_32_buf_2, tmp_32_buf_2, dy_sum_val, calc_len);
        PipeBarrier<PIPE_V>();
        inQueDY.FreeTensor(dy_ub);
        Sub(tmp_32_buf_1, tmp_32_buf_1, tmp_32_buf_2, calc_len);
        PipeBarrier<PIPE_V>();
        Muls(tmp_32_buf_1, tmp_32_buf_1, rstd_value, calc_len);
        PipeBarrier<PIPE_V>();
        LocalTensor<T1> dx_ub = outQueDX.AllocTensor<T1>();
        Cast(dx_ub, tmp_32_buf_1, RoundMode::CAST_NONE, calc_len);
        PipeBarrier<PIPE_V>();
        outQueDX.EnQue(dx_ub);
    }

    __aicore__ inline void ProcessBf16()
    {
        if (core_calc_tail == 0) {
            ProcessBf16Main(block_factor);
        } else {
            if (GetBlockIdx() < block_dim - 1) {
                ProcessBf16Main(block_factor);
            } else {
                ProcessBf16Main(core_calc_tail);
            }
        }
    }

    __aicore__ inline void ProcessBf16Main(uint32_t loop_len)
    {
        for (uint32_t i = 0; i < loop_len; i++) {
            // Calc mean firstly
            LocalTensor<float> dy_sum = nFp32Buf.Get<float>();
            Duplicate(dy_sum, 0.0f, align_len);
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < (ub_calc_tail == 0 ? ub_calc_loop : ub_calc_loop - 1); j++) {
                ComputeDySumBf16(i, j, ub_factor, ub_factor, dy_sum);
            }
            if (ub_calc_tail != 0) {
                ub_tail_align = (ub_calc_tail + align_len - 1) / align_len * align_len;
                ComputeDySumBf16(i, ub_calc_loop - 1, ub_calc_tail, ub_tail_align, dy_sum);
            }
            Muls(dy_sum, dy_sum, avg_factor, align_len);
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < (ub_calc_tail == 0 ? ub_calc_loop : ub_calc_loop - 1); j++) {
                CopyGammaIn(j, ub_factor);
                LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();

                CopyIn(i, j, ub_factor, ub_factor);
                ComputeMainBf16(i, j, ub_factor, gamma_ub, dy_sum);
                if (fixed_output == 0) {
                    CopyDgammaOut(j, ub_factor);
                }
                CopyOut(i, j, ub_factor, ub_factor);
            }
            if (ub_calc_tail != 0) {
                CopyGammaIn(ub_calc_loop - 1, ub_calc_tail);
                LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
                CopyIn(i, ub_calc_loop - 1, ub_calc_tail, ub_tail_align);
                ComputeMainBf16(i, ub_calc_loop - 1, ub_calc_tail, gamma_ub, dy_sum);
                if (fixed_output == 0) {
                    CopyDgammaOut(ub_calc_loop - 1, ub_calc_tail);
                }
                CopyOut(i, ub_calc_loop - 1, ub_calc_tail, ub_tail_align);
            }
        }
        if (fixed_output == 1) {
            ComputeDgammaMain(loop_len);
        }
    }

    __aicore__ inline void ComputeDgammaBf16(
        uint32_t i, uint32_t j, uint32_t calc_len, LocalTensor<T1>& gamma_ub, LocalTensor<float>& dgamma)
    {
        LocalTensor<T1> x_ub = inQueX.DeQue<T1>();
        LocalTensor<T1> dy_ub = inQueDY.DeQue<T1>();
        LocalTensor<float> rstd_ub = inQueRstd.DeQue<float>();
        LocalTensor<float> tmp_32_buf_2 = ndBufFp32Buf3.Get<float>();
        LocalTensor<float> tmp_32_buf_1 = ndBufFp32Buf1.Get<float>();
        // dy * (x * rstd)  -> sum
        event_t event_mte_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(event_mte_s);
        WaitFlag<HardEvent::MTE2_S>(event_mte_s);
        float rstd_value = rstd_ub.GetValue(0);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        Cast(tmp_32_buf_2, x_ub, RoundMode::CAST_NONE, calc_len);
        Cast(tmp_32_buf_1, dy_ub, RoundMode::CAST_NONE, calc_len);
        PipeBarrier<PIPE_V>();
        Muls(tmp_32_buf_2, tmp_32_buf_2, rstd_value, calc_len);
        PipeBarrier<PIPE_V>();
        inQueRstd.FreeTensor(rstd_ub);
        Cast(x_ub, tmp_32_buf_2, RoundMode::CAST_RINT, calc_len);
        PipeBarrier<PIPE_V>();
        Cast(tmp_32_buf_2, x_ub, RoundMode::CAST_NONE, calc_len);
        PipeBarrier<PIPE_V>();
        inQueX.FreeTensor(x_ub);
        Mul(tmp_32_buf_2, tmp_32_buf_1, tmp_32_buf_2, calc_len);
        PipeBarrier<PIPE_V>();
        Cast(dy_ub, tmp_32_buf_2, RoundMode::CAST_RINT, calc_len);
        PipeBarrier<PIPE_V>();
        Cast(tmp_32_buf_2, dy_ub, RoundMode::CAST_NONE, calc_len);
        PipeBarrier<PIPE_V>();
        inQueDY.FreeTensor(dy_ub);
        Add(dgamma, dgamma, tmp_32_buf_2, calc_len);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeDySumBf16(
        uint32_t i, uint32_t j, uint32_t calc_len, uint32_t calc_len_align, LocalTensor<float>& dy_sum)
    {
        CopyGammaIn(j, calc_len);
        CopyIn(i, j, calc_len, calc_len_align);
        LocalTensor<T1> gamma_ub = inQueGamma.DeQue<T1>();
        LocalTensor<T1> x_ub = inQueX.DeQue<T1>();
        LocalTensor<T1> dy_ub = inQueDY.DeQue<T1>();
        LocalTensor<float> rstd_ub = inQueRstd.DeQue<float>();
        event_t event_mte_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(event_mte_s);
        WaitFlag<HardEvent::MTE2_S>(event_mte_s);
        float rstd_value = rstd_ub.GetValue(0);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);

        LocalTensor<float> dy_sum_part = ndBufFp32Buf1.Get<float>();
        LocalTensor<float> tmp_32_buf = ndBufFp32Buf2.Get<float>();
        LocalTensor<float> tmp_32_buf_2 = ndBufFp32Buf3.Get<float>();

        // grad_y = dy * gamma
        Cast(tmp_32_buf, dy_ub, RoundMode::CAST_NONE, calc_len);
        Cast(tmp_32_buf_2, gamma_ub, RoundMode::CAST_NONE, calc_len);
        PipeBarrier<PIPE_V>();
        inQueGamma.FreeTensor(gamma_ub);
        Mul(tmp_32_buf, tmp_32_buf, tmp_32_buf_2, calc_len);
        PipeBarrier<PIPE_V>();
        Cast(dy_ub, tmp_32_buf, RoundMode::CAST_RINT, calc_len);
        PipeBarrier<PIPE_V>();
        Cast(tmp_32_buf, dy_ub, RoundMode::CAST_NONE, calc_len);
        // sum(x * rstd * grad_y)
        Cast(tmp_32_buf_2, x_ub, RoundMode::CAST_NONE, calc_len);
        PipeBarrier<PIPE_V>();
        inQueX.FreeTensor(x_ub);
        inQueDY.FreeTensor(dy_ub);
        inQueRstd.FreeTensor(rstd_ub);
        Muls(tmp_32_buf_2, tmp_32_buf_2, rstd_value, calc_len);
        PipeBarrier<PIPE_V>();
        Mul(dy_sum_part, tmp_32_buf, tmp_32_buf_2, calc_len);
        PipeBarrier<PIPE_V>();
        ReduceSumFP32(0, dy_sum_part, dy_sum_part, tmp_32_buf_2, calc_len, col_val_align);
        Add(dy_sum, dy_sum, dy_sum_part, align_len);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeMainBf16(
        uint32_t n_idx, uint32_t d_idx, uint32_t calc_len, LocalTensor<T1>& gamma_ub, LocalTensor<float>& dy_sum)
    {
        LocalTensor<T1> x_ub = inQueX.DeQue<T1>();
        LocalTensor<T1> dy_ub = inQueDY.DeQue<T1>();
        LocalTensor<float> rstd_ub = inQueRstd.DeQue<float>();
        LocalTensor<float> tmp_32_buf_1 = ndBufFp32Buf1.Get<float>();
        LocalTensor<float> tmp_32_buf_2 = ndBufFp32Buf2.Get<float>();
        LocalTensor<float> tmp_32_buf_3 = ndBufFp32Buf3.Get<float>();
        event_t event_mte_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(event_mte_s);
        WaitFlag<HardEvent::MTE2_S>(event_mte_s);
        float rstd_value = rstd_ub.GetValue(0);
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float dy_sum_val = dy_sum.GetValue(0);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        // dg = sum((dy * (x * rstd)), dim=0)
        Cast(tmp_32_buf_2, x_ub, RoundMode::CAST_NONE, calc_len);
        Cast(tmp_32_buf_1, dy_ub, RoundMode::CAST_NONE, calc_len);
        PipeBarrier<PIPE_V>();
        inQueRstd.FreeTensor(rstd_ub);
        Muls(tmp_32_buf_2, tmp_32_buf_2, rstd_value, calc_len);
        PipeBarrier<PIPE_V>();
        if (fixed_output == 1) {
            Cast(tmp_32_buf_3, gamma_ub, RoundMode::CAST_NONE, calc_len);
            Muls(tmp_32_buf_2, tmp_32_buf_2, dy_sum_val, calc_len);
            PipeBarrier<PIPE_V>();
        } else {
            Cast(x_ub, tmp_32_buf_2, RoundMode::CAST_RINT, calc_len);
            PipeBarrier<PIPE_V>();
            Cast(tmp_32_buf_3, x_ub, RoundMode::CAST_NONE, calc_len);
            PipeBarrier<PIPE_V>();
            Mul(tmp_32_buf_3, tmp_32_buf_1, tmp_32_buf_3, calc_len);
            PipeBarrier<PIPE_V>();
            Cast(dy_ub, tmp_32_buf_3, RoundMode::CAST_RINT, calc_len);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
            Cast(dgamma, dy_ub, RoundMode::CAST_NONE, calc_len);
            // grad_y = grad* gamma
            Cast(tmp_32_buf_3, gamma_ub, RoundMode::CAST_NONE, calc_len);
            Muls(tmp_32_buf_2, tmp_32_buf_2, dy_sum_val, calc_len);
            PipeBarrier<PIPE_V>();
            outQueDgamma.EnQue(dgamma);
        }
        inQueGamma.FreeTensor(gamma_ub);
        Mul(tmp_32_buf_1, tmp_32_buf_1, tmp_32_buf_3, calc_len);
        PipeBarrier<PIPE_V>();
        Cast(dy_ub, tmp_32_buf_1, RoundMode::CAST_RINT, calc_len);
        PipeBarrier<PIPE_V>();
        Cast(tmp_32_buf_1, dy_ub, RoundMode::CAST_NONE, calc_len);
        PipeBarrier<PIPE_V>();
        inQueX.FreeTensor(x_ub);
        inQueDY.FreeTensor(dy_ub);
        // dx = (grad_y - y * mean) * rstd
        Sub(tmp_32_buf_1, tmp_32_buf_1, tmp_32_buf_2, calc_len);
        PipeBarrier<PIPE_V>();
        Muls(tmp_32_buf_1, tmp_32_buf_1, rstd_value, calc_len);
        PipeBarrier<PIPE_V>();
        LocalTensor<T1> dx_ub = outQueDX.AllocTensor<T1>();
        Cast(dx_ub, tmp_32_buf_1, RoundMode::CAST_RINT, calc_len);
        PipeBarrier<PIPE_V>();
        outQueDX.EnQue(dx_ub);
    }

public:
    uint32_t ub_tail_align{0};
    uint32_t row_val{0};
    uint32_t col_val{0};
    uint32_t col_val_align{0};
    float avg_factor{1.0f};
    uint32_t core_calc_num{0};
    uint32_t core_calc_tail{0};
    uint32_t block_factor{0};
    uint32_t block_dim{0};
    uint32_t ub_factor{0};
    uint32_t ub_calc_num{0};
    uint32_t ub_calc_tail{0};
    uint32_t ub_calc_loop{0};
    uint32_t ub_calc_tail_num{0};
    uint32_t ub_calc_tail_tail{0};
    uint32_t ub_calc_tail_loop{0};
    uint32_t data_type{0};
    uint32_t align_len{0};
    uint32_t core_offset{0};
    uint32_t ub_factor_align{0};
    uint32_t rstd_len{0};
    uint32_t buffer_len_size{0};
    uint32_t core_offset_start{0};
    uint32_t core_offset_len{0};
    int32_t buffer_num{1};
    uint32_t fixed_output{0};

    TPipe pipe;
    GlobalTensor<T1> dyGm;
    GlobalTensor<T1> gammaGm;
    GlobalTensor<T1> dxGm;
    GlobalTensor<T1> xGm;
    GlobalTensor<float> workspace_gm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<float> dgammaGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueDY;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueRstd;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueGamma;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueDX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueDgamma;
    TBuf<TPosition::VECCALC> ndBufFp32Buf1;
    TBuf<TPosition::VECCALC> ndBufFp32Buf2;
    TBuf<TPosition::VECCALC> ndBufFp32Buf3;
    TBuf<TPosition::VECCALC> dFp32Buf;
    TBuf<TPosition::VECCALC> nFp32Buf;
};
#endif // RMS_NORM_GRAD_SPLIT_D_H