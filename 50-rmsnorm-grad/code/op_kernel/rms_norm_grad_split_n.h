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
 * \file rms_norm_grad_split_n.h
 * \brief
 */

#ifndef RMS_NORM_GRAD_SPLIT_N_H
#define RMS_NORM_GRAD_SPLIT_N_H
#include "rms_norm_grad_common.h"
template <typename T>
class RmsNormGradSplitN {
public:
    __aicore__ inline RmsNormGradSplitN()
    {}

    __aicore__ inline void Init(
        GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma,
        const RmsNormGradTilingData* tiling, GM_ADDR usrWorkspace)
    {
        InitVar(tiling);
        InitInputGmBuffer(dy, x, rstd, gamma, block_dim, core_calc_num, core_calc_tail);
        InitOutputGmBuffer(dx, dgamma, block_dim, core_calc_num, core_calc_tail);
        InitInputQue();
        InitOutputQue();
        InitTmpBuffer();
        if (fixed_output == 1) {
            InitWorkspace(usrWorkspace);
        } else {
            SyncAll();
        }
    }

    __aicore__ inline void InitWorkspace(GM_ADDR usrWorkspace)
    {
        workspace_gm.SetGlobalBuffer((__gm__ float*)usrWorkspace + GetBlockIdx() * col_val);
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
        dyGm.SetGlobalBuffer((__gm__ T*)dy + GetBlockIdx() * block_factor * col_val, core_offset * col_val);
        xGm.SetGlobalBuffer((__gm__ T*)x + GetBlockIdx() * block_factor * col_val, core_offset * col_val);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * block_factor, core_offset);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, col_val);
    }

    __aicore__ inline void InitOutputGmBuffer(
        GM_ADDR dx, GM_ADDR dgamma, uint32_t block_dim, uint32_t core_calc_num, uint32_t core_calc_tail)
    {
        dxGm.SetGlobalBuffer((__gm__ T*)dx + GetBlockIdx() * block_factor * col_val, core_offset * col_val);
        dgammaGm.SetGlobalBuffer((__gm__ float*)dgamma, col_val);
        if (fixed_output == 1) {
            return;
        } else {
            if (GetBlockIdx() == 0) {
                InitOutput<float>(dgammaGm, col_val, 0);
            }
        }
    }

    __aicore__ inline void InitInputQue()
    {
        ub_factor_align = ub_factor * col_val_align;
        rstd_len = (ub_factor + align_len - 1) / align_len * align_len;
        buffer_len_size = ub_factor_align * sizeof(T);
        buffer_num = data_type == BFLOAT16_DTYPE ? BUFFER_NUM_DB : BUFFER_NUM;
        pipe.InitBuffer(inQueDY, buffer_num, buffer_len_size);
        pipe.InitBuffer(inQueX, buffer_num, buffer_len_size);
        pipe.InitBuffer(inQueRstd, buffer_num, rstd_len * sizeof(float));
        pipe.InitBuffer(inQueGamma, BUFFER_NUM, col_val_align * sizeof(T));
    }

    __aicore__ inline void InitOutputQue()
    {
        pipe.InitBuffer(outQueDX, buffer_num, buffer_len_size);
        pipe.InitBuffer(outQueDgamma, BUFFER_NUM, col_val_align * sizeof(float));
    }

    __aicore__ inline void InitTmpBuffer()
    {
        uint32_t ub_factor_len = (ub_factor + align_len - 1) / align_len * align_len;
        uint32_t ub_factor_align_len = ub_factor_align * sizeof(float);
        pipe.InitBuffer(ndBufFp32Buf1, ub_factor_align_len);
        pipe.InitBuffer(nFp32Buf, ub_factor_len * sizeof(float));
        pipe.InitBuffer(dFp32Buf, col_val_align * sizeof(float));
        if (data_type != FLOAT_DTYPE) {
            pipe.InitBuffer(ndBufFp32Buf2, ub_factor_align_len);
            pipe.InitBuffer(ndBufFp32Buf3, ub_factor_align_len);
        }
        if (col_val_align < SMALLD_THRESHOLD) {
            pipe.InitBuffer(tmpBuf, ub_factor * ELEM_PER_REP_FP32 * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        CopyGammaIn();
        LocalTensor<T> gamma_ub = inQueGamma.DeQue<T>();
        LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
        Duplicate(dgamma, 0.0f, col_val_align);
        PipeBarrier<PIPE_V>();
        if (core_calc_tail == 0) {
            for (uint32_t i = 0; i < (ub_calc_tail == 0 ? ub_calc_loop : ub_calc_loop - 1); i++) {
                CopyIn(i, ub_calc_num);
                Compute(i, ub_calc_num, gamma_ub, dgamma);
                CopyOut(i, ub_calc_num);
            }
            if (ub_calc_tail != 0) {
                CopyIn(ub_calc_loop - 1, ub_calc_tail);
                Compute(ub_calc_loop - 1, ub_calc_tail, gamma_ub, dgamma);
                CopyOut(ub_calc_loop - 1, ub_calc_tail);
            }
        } else {
            if (GetBlockIdx() < block_dim - 1) {
                for (uint32_t i = 0; i < (ub_calc_tail == 0 ? ub_calc_loop : ub_calc_loop - 1); i++) {
                    CopyIn(i, ub_calc_num);
                    Compute(i, ub_calc_num, gamma_ub, dgamma);
                    CopyOut(i, ub_calc_num);
                }
                if (ub_calc_tail != 0) {
                    CopyIn(ub_calc_loop - 1, ub_calc_tail);
                    Compute(ub_calc_loop - 1, ub_calc_tail, gamma_ub, dgamma);
                    CopyOut(ub_calc_loop - 1, ub_calc_tail);
                }
            } else {
                for (uint32_t i = 0; i < (ub_calc_tail_tail == 0 ? ub_calc_tail_loop : ub_calc_tail_loop - 1); i++) {
                    CopyIn(i, ub_calc_tail_num);
                    Compute(i, ub_calc_tail_num, gamma_ub, dgamma);
                    CopyOut(i, ub_calc_tail_num);
                }
                if (ub_calc_tail_tail != 0) {
                    CopyIn(ub_calc_tail_loop - 1, ub_calc_tail_tail);
                    Compute(ub_calc_tail_loop - 1, ub_calc_tail_tail, gamma_ub, dgamma);
                    CopyOut(ub_calc_tail_loop - 1, ub_calc_tail_tail);
                }
            }
        }
        inQueGamma.FreeTensor(gamma_ub);
        outQueDgamma.EnQue(dgamma);
        if (fixed_output == 1) {
            CopyDgammaOutWorkspace();
            SyncAll();
            AddDgamma();
        } else {
            CopyDgammaOut();
        }
    }

    __aicore__ inline void CopyGammaIn()
    {
        LocalTensor<T> gamma_ub = inQueGamma.AllocTensor<T>();
        DataCopyParams data_copy_params{(uint16_t)1, (uint16_t)(col_val * sizeof(T)), 0, 0};
        DataCopyPadParams pad_params{true, 0, (uint8_t)(col_val_align - col_val), 0};
        DataCopyPad(gamma_ub, gammaGm, data_copy_params, pad_params);
        inQueGamma.EnQue(gamma_ub);
    }

    __aicore__ inline void CopyIn(uint32_t loop_idx, uint32_t calc_len)
    {
        DataCopyParams data_copy_params_rstd{(uint16_t)1, (uint16_t)(calc_len * sizeof(float)), 0, 0};
        DataCopyPadParams pad_params_rstd{true, 0, 0, 0};
        if (core_calc_tail == 0) {
            DataCopyParams data_copy_params{(uint16_t)calc_len, (uint16_t)(col_val * sizeof(T)), 0, 0};
            DataCopyPadParams pad_params{true, 0, (uint8_t)(col_val_align - col_val), 0};
            LocalTensor<float> rstd = inQueRstd.AllocTensor<float>();
            DataCopyPad(rstd, rstdGm[loop_idx * ub_factor], data_copy_params_rstd, pad_params_rstd);
            inQueRstd.EnQue(rstd);
            LocalTensor<T> x = inQueX.AllocTensor<T>();
            DataCopyPad(x, xGm[loop_idx * ub_factor * col_val], data_copy_params, pad_params);
            inQueX.EnQue(x);
            LocalTensor<T> dy = inQueDY.AllocTensor<T>();
            DataCopyPad(dy, dyGm[loop_idx * ub_factor * col_val], data_copy_params, pad_params);
            inQueDY.EnQue(dy);
        } else {
            if (GetBlockIdx() < block_dim - 1) {
                DataCopyParams data_copy_params{(uint16_t)calc_len, (uint16_t)(col_val * sizeof(T)), 0, 0};
                DataCopyPadParams pad_params{true, 0, (uint8_t)(col_val_align - col_val), 0};
                LocalTensor<float> rstd = inQueRstd.AllocTensor<float>();
                DataCopyPad(rstd, rstdGm[loop_idx * ub_factor], data_copy_params_rstd, pad_params_rstd);
                inQueRstd.EnQue(rstd);
                LocalTensor<T> x = inQueX.AllocTensor<T>();
                DataCopyPad(x, xGm[loop_idx * ub_factor * col_val], data_copy_params, pad_params);
                inQueX.EnQue(x);
                LocalTensor<T> dy = inQueDY.AllocTensor<T>();
                DataCopyPad(dy, dyGm[loop_idx * ub_factor * col_val], data_copy_params, pad_params);
                inQueDY.EnQue(dy);
            } else {
                DataCopyParams data_copy_params{(uint16_t)calc_len, (uint16_t)(col_val * sizeof(T)), 0, 0};
                DataCopyPadParams pad_params{true, 0, (uint8_t)(col_val_align - col_val), 0};
                LocalTensor<float> rstd = inQueRstd.AllocTensor<float>();
                DataCopyPad(rstd, rstdGm[loop_idx * ub_calc_tail_num], data_copy_params_rstd, pad_params_rstd);
                inQueRstd.EnQue(rstd);
                LocalTensor<T> x = inQueX.AllocTensor<T>();
                DataCopyPad(x, xGm[loop_idx * ub_calc_tail_num * col_val], data_copy_params, pad_params);
                inQueX.EnQue(x);
                LocalTensor<T> dy = inQueDY.AllocTensor<T>();
                DataCopyPad(dy, dyGm[loop_idx * ub_calc_tail_num * col_val], data_copy_params, pad_params);
                inQueDY.EnQue(dy);
            }
        }
    }

    __aicore__ inline void CopyOut(uint32_t loop_idx, uint32_t calc_len)
    {
        if (core_calc_tail == 0) {
            DataCopyParams data_copy_params{(uint16_t)calc_len, (uint16_t)(col_val * sizeof(T)), 0, 0};
            LocalTensor<T> dx = outQueDX.DeQue<T>();
            DataCopyPad(dxGm[loop_idx * ub_factor * col_val], dx, data_copy_params);
            outQueDX.FreeTensor(dx);
        } else {
            if (GetBlockIdx() < block_dim - 1) {
                DataCopyParams data_copy_params{(uint16_t)calc_len, (uint16_t)(col_val * sizeof(T)), 0, 0};
                LocalTensor<T> dx = outQueDX.DeQue<T>();
                DataCopyPad(dxGm[loop_idx * ub_factor * col_val], dx, data_copy_params);
                outQueDX.FreeTensor(dx);
            } else {
                DataCopyParams data_copy_params{(uint16_t)calc_len, (uint16_t)(col_val * sizeof(T)), 0, 0};
                LocalTensor<T> dx = outQueDX.DeQue<T>();
                DataCopyPad(dxGm[loop_idx * ub_calc_tail_num * col_val], dx, data_copy_params);
                outQueDX.FreeTensor(dx);
            }
        }
    }

    __aicore__ inline void CopyDgammaOut()
    {
        LocalTensor<float> dgamma_out = outQueDgamma.DeQue<float>();
        SetAtomicAdd<float>();
        DataCopyParams data_copy_params{(uint16_t)1, (uint16_t)(col_val * sizeof(float)), 0, 0};
        DataCopyPad(dgammaGm, dgamma_out, data_copy_params);
        SetAtomicNone();
        outQueDgamma.FreeTensor(dgamma_out);
    }

    __aicore__ inline void CopyDgammaOutWorkspace()
    {
        LocalTensor<float> dgamma_out = outQueDgamma.DeQue<float>();
        DataCopyParams data_copy_params{(uint16_t)1, (uint16_t)(col_val * sizeof(float)), 0, 0};
        DataCopyPad(workspace_gm, dgamma_out, data_copy_params);
        outQueDgamma.FreeTensor(dgamma_out);
    }

    __aicore__ inline void AddDgamma()
    {
        if (GetBlockIdx() != 0) {
            return;
        }
        LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
        LocalTensor<float> tmp_buf1 = ndBufFp32Buf1.Get<float>(col_val_align);
        PipeBarrier<PIPE_ALL>();
        DataCopyParams data_copy_params{(uint16_t)1, (uint16_t)(col_val * sizeof(float)), 0, 0};
        uint8_t rightPadding = (uint8_t)(col_val_align - col_val);
        if (rightPadding > ALIGN_32) {
            rightPadding = 0;
        }
        DataCopyPadParams pad_params{true, 0, rightPadding, 0};
        for (uint32_t blockidx = 0; blockidx < block_dim; blockidx++) {
            if (blockidx == 0) {
                DataCopyPad(dgamma, workspace_gm[blockidx * col_val], data_copy_params, pad_params);
            } else {
                DataCopyPad(tmp_buf1, workspace_gm[blockidx * col_val], data_copy_params, pad_params);
                PipeBarrier<PIPE_ALL>();
                Add(dgamma, dgamma, tmp_buf1, col_val_align);
                PipeBarrier<PIPE_ALL>();
            }
        }
        outQueDgamma.EnQue(dgamma);
        LocalTensor<float> dgamma_out = outQueDgamma.DeQue<float>();
        DataCopyParams data_copy_params_out{(uint16_t)1, (uint16_t)(col_val * sizeof(float)), 0, 0};
        DataCopyPad(dgammaGm, dgamma_out, data_copy_params_out);
        outQueDgamma.FreeTensor(dgamma_out);
    }

    __aicore__ inline void Compute(
        uint32_t loop_idx, uint32_t calc_len, LocalTensor<float>& gamma_ub, LocalTensor<float>& dgamma)
    {
        LocalTensor<T> x_ub = inQueX.DeQue<T>();
        LocalTensor<T> dy_ub = inQueDY.DeQue<T>();
        LocalTensor<float> rstd_ub = inQueRstd.DeQue<float>();

        LocalTensor<T> dx_ub = outQueDX.AllocTensor<T>();
        LocalTensor<float> tmp_buf = ndBufFp32Buf1.Get<float>(calc_len * col_val_align);

        if (col_val_align < SMALLD_THRESHOLD) {
            ComputeMainSmallD(x_ub, tmp_buf, dx_ub, dy_ub, rstd_ub, gamma_ub, dgamma, calc_len);
        } else {
            ComputeMain(x_ub, tmp_buf, dx_ub, dy_ub, rstd_ub, gamma_ub, dgamma, calc_len);
        }
        outQueDX.EnQue(dx_ub);
    }

    __aicore__ inline void ComputeMainSmallD(
        LocalTensor<float>& x, LocalTensor<float>& tmp_32_buf, LocalTensor<float>& dx, LocalTensor<float>& dy,
        LocalTensor<float>& rstd, LocalTensor<float>& gamma, LocalTensor<float>& dgamma, uint32_t calc_len)
    {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<float> tmp_rstd_buf = nFp32Buf.Get<float>();
        LocalTensor<float> tmp_reduce_buf = tmpBuf.Get<float>();

        uint32_t element_num = col_val_align * calc_len;

        // y = x * rstd
        const uint32_t src_n1_shape[2] = {calc_len, 1};
        const uint32_t dst_nd_shape[2] = {calc_len, col_val_align};
        auto shared_tmp = tmp_reduce_buf.ReinterpretCast<uint8_t>();
        BroadCast<float, DIM_NUM, DIM_D>(dx, rstd, dst_nd_shape, src_n1_shape, shared_tmp);
        inQueRstd.FreeTensor(rstd);
        PipeBarrier<PIPE_V>();
        Mul(x, x, dx, element_num); // x save x*rstd
        PipeBarrier<PIPE_V>();
        // dg=sum(dy * (x * rstd), dim=0)
        Mul(tmp_32_buf, dy, x, element_num);
        PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < calc_len; i++) {
            Add(dgamma, tmp_32_buf[i * col_val_align], dgamma, col_val_align);
            PipeBarrier<PIPE_V>();
        }

        // gamma
        const uint32_t src_1d_shape[2] = {1, col_val_align};
        BroadCast<float, DIM_NUM, DIM_N>(tmp_32_buf, gamma, dst_nd_shape, src_1d_shape, shared_tmp); // x reuse gamma_nd

        // dy * gamma
        PipeBarrier<PIPE_V>();
        Mul(dy, dy, tmp_32_buf, element_num); // dy save dy*gamma

        PipeBarrier<PIPE_V>();
        Mul(tmp_32_buf, dy, x, element_num);
        PipeBarrier<PIPE_V>();
        ReduceSumMultiN(tmp_rstd_buf, tmp_32_buf, tmp_reduce_buf, calc_len, col_val, col_val_align);
        PipeBarrier<PIPE_V>();
        Muls(tmp_rstd_buf, tmp_rstd_buf, avg_factor, calc_len);
        PipeBarrier<PIPE_V>();
        BroadCast<float, DIM_NUM, DIM_D>(tmp_32_buf, tmp_rstd_buf, dst_nd_shape, src_n1_shape, shared_tmp);
        PipeBarrier<PIPE_V>();
        Mul(tmp_32_buf, tmp_32_buf, x, element_num);
        inQueX.FreeTensor(x);
        PipeBarrier<PIPE_V>();
        Sub(dy, dy, tmp_32_buf, element_num);
        PipeBarrier<PIPE_V>();
        Mul(dx, dy, dx, element_num);
        inQueDY.FreeTensor(dy);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeMain(
        LocalTensor<float>& x, LocalTensor<float>& tmp_buf, LocalTensor<float>& dx, LocalTensor<float>& dy,
        LocalTensor<float>& rstd, LocalTensor<float>& gamma, LocalTensor<float>& dgamma, uint32_t calc_len)
    {
        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0; i < calc_len; i++) {
            float rstd_value = rstd.GetValue(i);
            event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(event_s_v);
            WaitFlag<HardEvent::S_V>(event_s_v);
            // grad_y = dy * g
            Mul(tmp_buf[i * col_val_align], dy[i * col_val_align], gamma, col_val_align);
            // y = x * rstd
            Muls(x[i * col_val_align], x[i * col_val_align], rstd_value, col_val_align);
            PipeBarrier<PIPE_V>();
            if (i == calc_len - 1) {
                inQueRstd.FreeTensor(rstd);
            }
            // dg = sum(dy * (x * rstd), dim=0)
            Mul(dy[i * col_val_align], dy[i * col_val_align], x[i * col_val_align], col_val_align);
            PipeBarrier<PIPE_V>();
            Add(dgamma, dgamma, dy[i * col_val_align], col_val_align);
            PipeBarrier<PIPE_V>();
            // mean = sum(grad_y * y) * avg_factor
            Duplicate(dy[i * col_val_align], 0.0f, col_val_align);
            PipeBarrier<PIPE_V>();
            Mul(dy[i * col_val_align], tmp_buf[i * col_val_align], x[i * col_val_align], col_val);
            PipeBarrier<PIPE_V>();
            float reduce_val = ReduceSumFP32_V2(dy[i * col_val_align], col_val_align);
            float dy_sum_val = reduce_val * avg_factor;
            Muls(x[i * col_val_align], x[i * col_val_align], dy_sum_val, col_val_align);
            PipeBarrier<PIPE_V>();
            Sub(tmp_buf[i * col_val_align], tmp_buf[i * col_val_align], x[i * col_val_align], col_val_align);
            PipeBarrier<PIPE_V>();
            if (i == calc_len - 1) {
                inQueX.FreeTensor(x);
                inQueDY.FreeTensor(dy);
            }
            Muls(dx[i * col_val_align], tmp_buf[i * col_val_align], rstd_value, col_val_align);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessFp16()
    {
        CopyGammaIn();
        LocalTensor<T> gamma_ub = inQueGamma.DeQue<T>();
        LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
        Duplicate(dgamma, 0.0f, col_val_align);
        PipeBarrier<PIPE_V>();
        if (core_calc_tail == 0) {
            for (uint32_t i = 0; i < (ub_calc_tail == 0 ? ub_calc_loop : ub_calc_loop - 1); i++) {
                CopyIn(i, ub_calc_num);
                ComputeFp16(i, ub_calc_num, gamma_ub, dgamma);
                CopyOut(i, ub_calc_num);
            }
            if (ub_calc_tail != 0) {
                CopyIn(ub_calc_loop - 1, ub_calc_tail);
                ComputeFp16(ub_calc_loop - 1, ub_calc_tail, gamma_ub, dgamma);
                CopyOut(ub_calc_loop - 1, ub_calc_tail);
            }
        } else {
            if (GetBlockIdx() < block_dim - 1) {
                for (uint32_t i = 0; i < (ub_calc_tail == 0 ? ub_calc_loop : ub_calc_loop - 1); i++) {
                    CopyIn(i, ub_calc_num);
                    ComputeFp16(i, ub_calc_num, gamma_ub, dgamma);
                    CopyOut(i, ub_calc_num);
                }
                if (ub_calc_tail != 0) {
                    CopyIn(ub_calc_loop - 1, ub_calc_tail);
                    ComputeFp16(ub_calc_loop - 1, ub_calc_tail, gamma_ub, dgamma);
                    CopyOut(ub_calc_loop - 1, ub_calc_tail);
                }
            } else {
                for (uint32_t i = 0; i < (ub_calc_tail_tail == 0 ? ub_calc_tail_loop : ub_calc_tail_loop - 1); i++) {
                    CopyIn(i, ub_calc_tail_num);
                    ComputeFp16(i, ub_calc_tail_num, gamma_ub, dgamma);
                    CopyOut(i, ub_calc_tail_num);
                }
                if (ub_calc_tail_tail != 0) {
                    CopyIn(ub_calc_tail_loop - 1, ub_calc_tail_tail);
                    ComputeFp16(ub_calc_tail_loop - 1, ub_calc_tail_tail, gamma_ub, dgamma);
                    CopyOut(ub_calc_tail_loop - 1, ub_calc_tail_tail);
                }
            }
        }
        inQueGamma.FreeTensor(gamma_ub);
        outQueDgamma.EnQue(dgamma);
        if (fixed_output == 1) {
            CopyDgammaOutWorkspace();
            SyncAll();
            AddDgamma();
        } else {
            CopyDgammaOut();
        }
    }

    __aicore__ inline void ComputeFp16(
        uint32_t loop_idx, uint32_t calc_len, LocalTensor<T>& gamma_ub, LocalTensor<float>& dgamma)
    {
        LocalTensor<T> x_ub = inQueX.DeQue<T>();
        LocalTensor<T> dy_ub = inQueDY.DeQue<T>();
        LocalTensor<float> rstd_ub = inQueRstd.DeQue<float>();
        LocalTensor<T> dx_ub = outQueDX.AllocTensor<T>();
        if (col_val_align < SMALLD_THRESHOLD) {
            ComputeMainFp16SmallD(x_ub, dx_ub, dy_ub, rstd_ub, gamma_ub, dgamma, calc_len);
        } else {
            ComputeMainFp16(x_ub, dx_ub, dy_ub, rstd_ub, gamma_ub, dgamma, calc_len);
        }
        outQueDX.EnQue(dx_ub);
    }

    __aicore__ inline void ComputeMainFp16SmallD(
        LocalTensor<T>& x, LocalTensor<T>& dx, LocalTensor<T>& dy, LocalTensor<float>& rstd, LocalTensor<T>& gamma,
        LocalTensor<float>& dgamma, uint32_t calc_len)
    {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<float> dy_sum = ndBufFp32Buf1.Get<float>();
        LocalTensor<float> tmp_32_buf = ndBufFp32Buf3.Get<float>();
        LocalTensor<float> tmp_32_buf_2 = ndBufFp32Buf2.Get<float>();
        LocalTensor<float> tmp_rstd_buf = nFp32Buf.Get<float>();
        LocalTensor<float> tmp_reduce_buf = tmpBuf.Get<float>();

        uint32_t element_num = col_val_align * calc_len;
        Cast(tmp_32_buf_2, x, RoundMode::CAST_NONE, element_num);
        PipeBarrier<PIPE_V>();

        // y = x * rstd
        const uint32_t src_n1_shape[2] = {calc_len, 1};
        const uint32_t dst_nd_shape[2] = {calc_len, col_val_align};
        auto shared_tmp = tmp_reduce_buf.ReinterpretCast<uint8_t>();
        BroadCast<float, DIM_NUM, DIM_D>(tmp_32_buf, rstd, dst_nd_shape, src_n1_shape, shared_tmp);
        PipeBarrier<PIPE_V>();
        Mul(tmp_32_buf_2, tmp_32_buf_2, tmp_32_buf, element_num); // tmp_32_buf_2 save x*rstd
        PipeBarrier<PIPE_V>();
        // dg=sum(dy * (x * rstd), dim=0)
        Cast(x, tmp_32_buf_2, RoundMode::CAST_NONE, element_num);
        PipeBarrier<PIPE_V>();
        Mul(x, dy, x, element_num); // mul_1
        PipeBarrier<PIPE_V>();
        Cast(dy_sum, x, RoundMode::CAST_NONE, element_num);
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < calc_len; i++) {
            Add(dgamma, dy_sum[i * col_val_align], dgamma, col_val_align);
            PipeBarrier<PIPE_V>();
        }

        // gamma
        const uint32_t src_1d_shape[2] = {1, col_val_align};
        BroadCast<half, DIM_NUM, DIM_N>(x, gamma, dst_nd_shape, src_1d_shape, shared_tmp); // x reuse gamma_nd

        // dy * gamma
        PipeBarrier<PIPE_V>();
        Mul(dy, dy, x, element_num);
        inQueX.FreeTensor(x);
        PipeBarrier<PIPE_V>();

        Cast(dy_sum, dy, RoundMode::CAST_NONE, element_num); // dy_sum save dy * gamma
        PipeBarrier<PIPE_V>();

        inQueDY.FreeTensor(dy);
        Mul(tmp_32_buf, dy_sum, tmp_32_buf_2, element_num);
        PipeBarrier<PIPE_V>();
        ReduceSumMultiN(tmp_rstd_buf, tmp_32_buf, tmp_reduce_buf, calc_len, col_val, col_val_align);
        PipeBarrier<PIPE_V>();
        Muls(tmp_rstd_buf, tmp_rstd_buf, avg_factor, calc_len); // muls_0
        PipeBarrier<PIPE_V>();
        BroadCast<float, DIM_NUM, DIM_D>(tmp_32_buf, tmp_rstd_buf, dst_nd_shape, src_n1_shape, shared_tmp);
        PipeBarrier<PIPE_V>();
        Mul(tmp_32_buf_2, tmp_32_buf, tmp_32_buf_2, element_num);
        PipeBarrier<PIPE_V>();
        Sub(dy_sum, dy_sum, tmp_32_buf_2, element_num);
        PipeBarrier<PIPE_V>();
        BroadCast<float, DIM_NUM, DIM_D>(tmp_32_buf, rstd, dst_nd_shape, src_n1_shape, shared_tmp);
        inQueRstd.FreeTensor(rstd);
        PipeBarrier<PIPE_V>();
        Mul(dy_sum, dy_sum, tmp_32_buf, element_num);
        PipeBarrier<PIPE_V>();
        Cast(dx, dy_sum, RoundMode::CAST_NONE, element_num);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeMainFp16(
        LocalTensor<T>& x, LocalTensor<T>& dx, LocalTensor<T>& dy, LocalTensor<float>& rstd, LocalTensor<T>& gamma,
        LocalTensor<float>& dgamma, uint32_t calc_len)
    {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<float> dy_sum = ndBufFp32Buf1.Get<float>();
        LocalTensor<float> tmp_32_buf = ndBufFp32Buf3.Get<float>();
        LocalTensor<float> tmp_32_buf_2 = ndBufFp32Buf2.Get<float>();
        LocalTensor<T> d_tmp_buf_fp16 = dFp32Buf.Get<T>();
        for (uint32_t i = 0; i < calc_len; i++) {
            float rstd_value = rstd.GetValue(i);
            event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(event_s_v);
            WaitFlag<HardEvent::S_V>(event_s_v);
            // y = x * rstd
            Cast(tmp_32_buf_2[i * col_val_align], x[i * col_val_align], RoundMode::CAST_NONE, col_val_align);
            PipeBarrier<PIPE_V>();
            if (i == calc_len - 1) {
                inQueRstd.FreeTensor(rstd);
            }
            Muls(tmp_32_buf_2[i * col_val_align], tmp_32_buf_2[i * col_val_align], rstd_value, col_val_align);
            Mul(d_tmp_buf_fp16, dy[i * col_val_align], gamma, col_val_align);
            PipeBarrier<PIPE_V>();
            Cast(tmp_32_buf[i * col_val_align], d_tmp_buf_fp16, RoundMode::CAST_NONE, col_val_align);
            PipeBarrier<PIPE_V>();
            // mean = sum (grad_y * y) * avg_factor
            Duplicate(dy_sum[i * col_val_align], 0.0f, col_val_align);
            PipeBarrier<PIPE_V>();
            Mul(dy_sum[i * col_val_align], tmp_32_buf[i * col_val_align], tmp_32_buf_2[i * col_val_align], col_val);
            PipeBarrier<PIPE_V>();
            float reduce_val = ReduceSumFP32_V2(dy_sum[i * col_val_align], col_val_align);
            float dy_sum_val = reduce_val * avg_factor;
            // dx = (grad_y - y * mean) * rstd
            Muls(dy_sum[i * col_val_align], tmp_32_buf_2[i * col_val_align], dy_sum_val, col_val_align);
            PipeBarrier<PIPE_V>();
            Sub(dy_sum[i * col_val_align], tmp_32_buf[i * col_val_align], dy_sum[i * col_val_align], col_val_align);
            Cast(x[i * col_val_align], tmp_32_buf_2[i * col_val_align], RoundMode::CAST_NONE, col_val_align);
            PipeBarrier<PIPE_V>();
            Muls(dy_sum[i * col_val_align], dy_sum[i * col_val_align], rstd_value, col_val_align);
            Mul(dy[i * col_val_align], x[i * col_val_align], dy[i * col_val_align], col_val_align);
            PipeBarrier<PIPE_V>();
            if (i == calc_len - 1) {
                inQueX.FreeTensor(x);
            }
            Cast(dx[i * col_val_align], dy_sum[i * col_val_align], RoundMode::CAST_NONE, col_val_align);
            Cast(tmp_32_buf[i * col_val_align], dy[i * col_val_align], RoundMode::CAST_NONE, col_val_align);
            PipeBarrier<PIPE_V>();
            if (i == calc_len - 1) {
                inQueDY.FreeTensor(dy);
            }
            Add(dgamma, tmp_32_buf[i * col_val_align], dgamma, col_val_align);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessBf16()
    {
        CopyGammaIn();
        LocalTensor<T> gamma_ub = inQueGamma.DeQue<T>();
        LocalTensor<float> dgamma = outQueDgamma.AllocTensor<float>();
        Duplicate(dgamma, 0.0f, col_val_align);
        PipeBarrier<PIPE_V>();
        if (core_calc_tail == 0) {
            for (uint32_t i = 0; i < (ub_calc_tail == 0 ? ub_calc_loop : ub_calc_loop - 1); i++) {
                CopyIn(i, ub_calc_num);
                ComputeBf16(i, ub_calc_num, gamma_ub, dgamma);
                CopyOut(i, ub_calc_num);
            }
            if (ub_calc_tail != 0) {
                CopyIn(ub_calc_loop - 1, ub_calc_tail);
                ComputeBf16(ub_calc_loop - 1, ub_calc_tail, gamma_ub, dgamma);
                CopyOut(ub_calc_loop - 1, ub_calc_tail);
            }
        } else {
            if (GetBlockIdx() < block_dim - 1) {
                for (uint32_t i = 0; i < (ub_calc_tail == 0 ? ub_calc_loop : ub_calc_loop - 1); i++) {
                    CopyIn(i, ub_calc_num);
                    ComputeBf16(i, ub_calc_num, gamma_ub, dgamma);
                    CopyOut(i, ub_calc_num);
                }
                if (ub_calc_tail != 0) {
                    CopyIn(ub_calc_loop - 1, ub_calc_tail);
                    ComputeBf16(ub_calc_loop - 1, ub_calc_tail, gamma_ub, dgamma);
                    CopyOut(ub_calc_loop - 1, ub_calc_tail);
                }
            } else {
                for (uint32_t i = 0; i < (ub_calc_tail_tail == 0 ? ub_calc_tail_loop : ub_calc_tail_loop - 1); i++) {
                    CopyIn(i, ub_calc_tail_num);
                    ComputeBf16(i, ub_calc_tail_num, gamma_ub, dgamma);
                    CopyOut(i, ub_calc_tail_num);
                }
                if (ub_calc_tail_tail != 0) {
                    CopyIn(ub_calc_tail_loop - 1, ub_calc_tail_tail);
                    ComputeBf16(ub_calc_tail_loop - 1, ub_calc_tail_tail, gamma_ub, dgamma);
                    CopyOut(ub_calc_tail_loop - 1, ub_calc_tail_tail);
                }
            }
        }
        inQueGamma.FreeTensor(gamma_ub);
        outQueDgamma.EnQue(dgamma);
        if (fixed_output == 1) {
            CopyDgammaOutWorkspace();
            SyncAll();
            AddDgamma();
        } else {
            CopyDgammaOut();
        }
    }

    __aicore__ inline void ComputeBf16(
        uint32_t loop_idx, uint32_t calc_len, LocalTensor<T>& gamma_ub, LocalTensor<float>& dgamma)
    {
        LocalTensor<T> x_ub = inQueX.DeQue<T>();
        LocalTensor<T> dy_ub = inQueDY.DeQue<T>();
        LocalTensor<float> rstd_ub = inQueRstd.DeQue<float>();
        LocalTensor<T> dx_ub = outQueDX.AllocTensor<T>();
        if (col_val_align < SMALLD_THRESHOLD) {
            ComputeMainBf16SmallD(x_ub, dx_ub, dy_ub, rstd_ub, gamma_ub, dgamma, calc_len);
        } else {
            ComputeMainBf16(x_ub, dx_ub, dy_ub, rstd_ub, gamma_ub, dgamma, calc_len);
        }
        outQueDX.EnQue(dx_ub);
    }

    __aicore__ inline void ComputeMainBf16SmallD(
        LocalTensor<T>& x, LocalTensor<T>& dx, LocalTensor<T>& dy, LocalTensor<float>& rstd, LocalTensor<T>& gamma,
        LocalTensor<float>& dgamma, uint32_t calc_len)
    {
        LocalTensor<float> dy_sum = ndBufFp32Buf1.Get<float>();
        LocalTensor<float> tmp_32_buf = ndBufFp32Buf3.Get<float>();
        LocalTensor<float> tmp_32_buf_2 = ndBufFp32Buf2.Get<float>();
        LocalTensor<float> gamma_fp32 = dFp32Buf.Get<float>();
        LocalTensor<float> tmp_rstd_buf = nFp32Buf.Get<float>();
        LocalTensor<float> tmp_reduce_buf = tmpBuf.Get<float>();
        uint32_t element_num = col_val_align * calc_len;
        Cast(tmp_32_buf_2, x, RoundMode::CAST_NONE, element_num);
        PipeBarrier<PIPE_V>();

        // y = x * rstd
        const uint32_t src_n1_shape[2] = {calc_len, 1};
        const uint32_t dst_nd_shape[2] = {calc_len, col_val_align};
        auto shared_tmp = tmp_reduce_buf.ReinterpretCast<uint8_t>();
        BroadCast<float, DIM_NUM, DIM_D>(tmp_32_buf, rstd, dst_nd_shape, src_n1_shape, shared_tmp);
        PipeBarrier<PIPE_V>();
        Mul(tmp_32_buf_2, tmp_32_buf_2, tmp_32_buf, element_num); // tmp_32_buf_2 save x*rstd
        PipeBarrier<PIPE_V>();
        // dg=sum(dy * (x * rstd), dim=0)
        Cast(x, tmp_32_buf_2, RoundMode::CAST_RINT, element_num);
        PipeBarrier<PIPE_V>();
        Cast(tmp_32_buf, x, RoundMode::CAST_NONE, element_num);
        Cast(dy_sum, dy, RoundMode::CAST_NONE, element_num);
        PipeBarrier<PIPE_V>();
        Mul(dy_sum, tmp_32_buf, dy_sum, element_num);
        Cast(x, dy_sum, RoundMode::CAST_RINT, element_num);
        PipeBarrier<PIPE_V>();
        Cast(dy_sum, x, RoundMode::CAST_NONE, element_num);
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < calc_len; i++) {
            Add(dgamma, dy_sum[i * col_val_align], dgamma, col_val_align);
            PipeBarrier<PIPE_V>();
        }

        // gamma
        Cast(gamma_fp32, gamma, RoundMode::CAST_NONE, col_val_align);
        PipeBarrier<PIPE_V>();
        const uint32_t src_1d_shape[2] = {1, col_val_align};
        BroadCast<float, DIM_NUM, DIM_N>(tmp_32_buf, gamma_fp32, dst_nd_shape, src_1d_shape, shared_tmp);

        // dy * gamma
        Cast(dy_sum, dy, RoundMode::CAST_NONE, element_num);
        inQueDY.FreeTensor(dy);
        PipeBarrier<PIPE_V>();
        Mul(dy_sum, dy_sum, tmp_32_buf, element_num);
        PipeBarrier<PIPE_V>();

        Cast(x, dy_sum, RoundMode::CAST_RINT, element_num);
        PipeBarrier<PIPE_V>();
        Cast(dy_sum, x, RoundMode::CAST_NONE, element_num); // dy_sum save dy * gamma
        PipeBarrier<PIPE_V>();
        inQueX.FreeTensor(x);

        Mul(tmp_32_buf, dy_sum, tmp_32_buf_2, element_num);
        PipeBarrier<PIPE_V>();
        ReduceSumMultiN(tmp_rstd_buf, tmp_32_buf, tmp_reduce_buf, calc_len, col_val, col_val_align);
        PipeBarrier<PIPE_V>();
        Muls(tmp_rstd_buf, tmp_rstd_buf, avg_factor, calc_len);
        PipeBarrier<PIPE_V>();
        BroadCast<float, DIM_NUM, DIM_D>(tmp_32_buf, tmp_rstd_buf, dst_nd_shape, src_n1_shape, shared_tmp);
        PipeBarrier<PIPE_V>();
        Mul(tmp_32_buf_2, tmp_32_buf, tmp_32_buf_2, element_num);
        PipeBarrier<PIPE_V>();
        Sub(dy_sum, dy_sum, tmp_32_buf_2, element_num);
        PipeBarrier<PIPE_V>();
        BroadCast<float, DIM_NUM, DIM_D>(tmp_32_buf, rstd, dst_nd_shape, src_n1_shape, shared_tmp);
        inQueRstd.FreeTensor(rstd);
        PipeBarrier<PIPE_V>();
        Mul(dy_sum, dy_sum, tmp_32_buf, element_num);
        PipeBarrier<PIPE_V>();
        Cast(dx, dy_sum, RoundMode::CAST_RINT, element_num);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeMainBf16(
        LocalTensor<T>& x, LocalTensor<T>& dx, LocalTensor<T>& dy, LocalTensor<float>& rstd, LocalTensor<T>& gamma,
        LocalTensor<float>& dgamma, uint32_t calc_len)
    {
        LocalTensor<float> dy_sum = ndBufFp32Buf1.Get<float>();
        LocalTensor<float> tmp_32_buf = ndBufFp32Buf3.Get<float>();
        LocalTensor<float> tmp_32_buf_2 = ndBufFp32Buf2.Get<float>();
        LocalTensor<float> gamma_fp32 = dFp32Buf.Get<float>();
        for (uint32_t i = 0; i < calc_len; i++) {
            Cast(gamma_fp32, gamma, RoundMode::CAST_NONE, col_val_align);
            Cast(tmp_32_buf_2[i * col_val_align], x[i * col_val_align], RoundMode::CAST_NONE, col_val_align);
            Cast(tmp_32_buf[i * col_val_align], dy[i * col_val_align], RoundMode::CAST_NONE, col_val_align);
            PipeBarrier<PIPE_V>();
            event_t event_mte_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(event_mte_s);
            WaitFlag<HardEvent::MTE2_S>(event_mte_s);
            float rstd_value = rstd.GetValue(i);
            event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(event_s_v);
            WaitFlag<HardEvent::S_V>(event_s_v);
            // y = x * rstd
            Muls(tmp_32_buf_2[i * col_val_align], tmp_32_buf_2[i * col_val_align], rstd_value, col_val_align);
            PipeBarrier<PIPE_V>();
            Mul(tmp_32_buf[i * col_val_align], tmp_32_buf[i * col_val_align], gamma_fp32, col_val_align);
            PipeBarrier<PIPE_V>();
            // mean = sum (grad_y * y) * avg_factor
            Duplicate(dy_sum[i * col_val_align], 0.0f, col_val_align);
            Cast(x[i * col_val_align], tmp_32_buf[i * col_val_align], RoundMode::CAST_RINT, col_val_align);
            PipeBarrier<PIPE_V>();
            Cast(tmp_32_buf[i * col_val_align], x[i * col_val_align], RoundMode::CAST_NONE, col_val_align);
            PipeBarrier<PIPE_V>();
            if (i == calc_len - 1) {
                inQueRstd.FreeTensor(rstd);
                inQueX.FreeTensor(x);
            }
            Mul(dy_sum[i * col_val_align], tmp_32_buf[i * col_val_align], tmp_32_buf_2[i * col_val_align], col_val);
            PipeBarrier<PIPE_V>();
            float reduce_val = ReduceSumFP32_V2(dy_sum[i * col_val_align], col_val_align);
            float dy_sum_val = reduce_val * avg_factor;
            // dx = (grad_y - y * mean) * rstd
            Muls(dy_sum[i * col_val_align], tmp_32_buf_2[i * col_val_align], dy_sum_val, col_val_align);
            PipeBarrier<PIPE_V>();
            Sub(dy_sum[i * col_val_align], tmp_32_buf[i * col_val_align], dy_sum[i * col_val_align], col_val_align);
            PipeBarrier<PIPE_V>();
            Muls(dy_sum[i * col_val_align], dy_sum[i * col_val_align], rstd_value, col_val_align);
            PipeBarrier<PIPE_V>();
            Cast(dx[i * col_val_align], dy_sum[i * col_val_align], RoundMode::CAST_RINT, col_val_align);
            Cast(tmp_32_buf[i * col_val_align], dy[i * col_val_align], RoundMode::CAST_NONE, col_val_align);
            PipeBarrier<PIPE_V>();
            // dg=sum(dy * (x * rstd), dim=0)
            Cast(dy[i * col_val_align], tmp_32_buf_2[i * col_val_align], RoundMode::CAST_RINT, col_val_align);
            PipeBarrier<PIPE_V>();
            Cast(dy_sum[i * col_val_align], dy[i * col_val_align], RoundMode::CAST_NONE, col_val_align);
            PipeBarrier<PIPE_V>();
            Mul(dy_sum[i * col_val_align], tmp_32_buf[i * col_val_align], dy_sum[i * col_val_align], col_val_align);
            PipeBarrier<PIPE_V>();
            Cast(dy[i * col_val_align], dy_sum[i * col_val_align], RoundMode::CAST_RINT, col_val_align);
            PipeBarrier<PIPE_V>();
            Cast(dy_sum[i * col_val_align], dy[i * col_val_align], RoundMode::CAST_NONE, col_val_align);
            PipeBarrier<PIPE_V>();
            if (i == calc_len - 1) {
                inQueDY.FreeTensor(dy);
            }
            Add(dgamma, dy_sum[i * col_val_align], dgamma, col_val_align);
            PipeBarrier<PIPE_V>();
        }
    }

public:
    uint32_t ub_tail_align;
    uint32_t row_val;
    uint32_t col_val;
    uint32_t col_val_align;
    float avg_factor{1.0f};
    uint32_t core_calc_num;
    uint32_t core_calc_tail;
    uint32_t block_factor;
    uint32_t block_dim;
    uint32_t ub_factor;
    uint32_t ub_calc_num;
    uint32_t ub_calc_tail;
    uint32_t ub_calc_loop;
    uint32_t ub_calc_tail_num;
    uint32_t ub_calc_tail_tail;
    uint32_t ub_calc_tail_loop;
    uint32_t data_type;
    uint32_t align_len;
    uint32_t core_offset;
    uint32_t ub_factor_align;
    uint32_t rstd_len;
    uint32_t buffer_len_size;
    int32_t buffer_num;
    uint32_t fixed_output{0};

    TPipe pipe;
    GlobalTensor<T> dyGm, gammaGm, dxGm, xGm;
    GlobalTensor<float> dgammaGm, rstdGm, workspace_gm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueDY, inQueX, inQueRstd, inQueGamma;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueDX, outQueDgamma;
    TBuf<TPosition::VECCALC> ndBufFp32Buf1, ndBufFp32Buf2, ndBufFp32Buf3;
    TBuf<TPosition::VECCALC> dFp32Buf;
    TBuf<TPosition::VECCALC> nFp32Buf;
    TBuf<TPosition::VECCALC> tmpBuf;
};
#endif // RMS_NORM_GRAD_SPLIT_N_H