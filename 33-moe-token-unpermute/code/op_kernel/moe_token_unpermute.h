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
 * \file moe_token_unpermute.h
 * \brief
 */
#ifndef MOE_TOKEN_UNPERMUTE
#define MOE_TOKEN_UNPERMUTE

#include "kernel_operator.h"
#if __CCE_AICORE__ == 200
#include "moe_token_unpermute_common.h"
#endif

using namespace AscendC;

template <typename T1, typename T2, typename T3, bool PROBS>
class KernelMoeTokenUnpermute
{
public:
    __aicore__ inline KernelMoeTokenUnpermute()
    {}

    __aicore__ inline void Init(
        GM_ADDR permuted_tokens, GM_ADDR sorted_indices, GM_ADDR probs, GM_ADDR unpermuted_tokens,
        const MoeTokenUnpermuteTilingData* __restrict tiling_data, TPipe* pipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CalMultiOutToken(const int64_t out_offset, const int64_t out_tokens_number);
    __aicore__ inline void CalSingleOutToken(const int64_t start_token, const int64_t out_token_idx);
    __aicore__ inline void CalPartOutToken(
        const int64_t start_token, const int64_t h_index, const int64_t h_length, const int64_t out_token_index);
    __aicore__ inline void CopyTokenIn(const T2 in_token_index, const int64_t h_index, const int64_t h_length);
    __aicore__ inline void CalFirstToken(const float prob_value, const int64_t h_length);
    __aicore__ inline void CalToken(const float prob_value, const int64_t h_length);
    __aicore__ inline void CopyOut(const int64_t out_token_index, const int64_t h_index, const int64_t h_length);

    TQue<QuePosition::VECIN, 1> tokens_inque, indices_inque, probs_inque;
    TBuf<TPosition::VECCALC> temp_buffer0, temp_buffer1, temp_buffer2;
    TQue<QuePosition::VECOUT, 1> outque;
    GlobalTensor<T1> tokensGM, outGM;
    GlobalTensor<T2> indicesGM;
    GlobalTensor<T3> probsGM;
    LocalTensor<T2> indicesLocal;
    LocalTensor<float> token_tensor0, token_tensor1, probs_tensor;
    DataCopyPadExtParams<T1> extParams1{false, 0, 0, 0};
    DataCopyPadExtParams<T2> extParams2{false, 0, 0, 0};
    DataCopyPadExtParams<T3> extParams3{false, 0, 0, 0};
    DataCopyExtParams copyParams{1, 0, 0, 0, 0};

    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static uint32_t ALIGN_512 = 512;

    int64_t hidden_size;
    int64_t top_k;
    int64_t num_out_tokens;
    int64_t hidden_splited_length;
    int64_t hidden_splited_num;
    int64_t hidden_splited_remain;
    int64_t tokens_core_length;
    int64_t tokens_core_remain;
    int64_t tokens_splited_length;
    int64_t tokens_splited_num;
    int64_t tokens_splited_remain;
};

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::Init(
    GM_ADDR permuted_tokens, GM_ADDR sorted_indices, GM_ADDR probs, GM_ADDR unpermuted_tokens,
    const MoeTokenUnpermuteTilingData* __restrict tiling_data, TPipe* pipe)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    // row_input
    this->hidden_size = tiling_data->hidden_size;
    this->top_k = tiling_data->top_k;
    this->num_out_tokens = tiling_data->num_out_tokens;
    // hidden_tiling
    this->hidden_splited_length = tiling_data->hidden_splited_length;
    this->hidden_splited_num = tiling_data->hidden_splited_num;
    this->hidden_splited_remain = tiling_data->hidden_splited_remain;
    // token_tiling
    this->tokens_core_length = tiling_data->tokens_core_length;
    this->tokens_core_remain = tiling_data->tokens_core_remain;
    this->tokens_splited_length = tiling_data->tokens_splited_length;
    this->tokens_splited_num = tiling_data->tokens_splited_num;
    this->tokens_splited_remain = tiling_data->tokens_splited_remain;

    // 处理token_by_core尾块
    if (this->tokens_core_remain > 0 && GetBlockIdx() < this->tokens_core_remain) {
        this->tokens_core_length += 1;
        this->tokens_splited_remain += 1;
    }

    int64_t hidden_splited_length_align512 = (this->hidden_splited_length + ALIGN_512 - 1) & ~(ALIGN_512 - 1);

    int64_t block_length = this->tokens_core_length * this->top_k;
    int64_t block_splited_length = this->tokens_splited_length * this->top_k;

    int64_t block_offset;
    if (this->tokens_core_remain > 0) {
        if (GetBlockIdx() < this->tokens_core_remain) {
            block_offset = block_length * GetBlockIdx();
        } else {
            block_offset = (block_length + this->top_k) * this->tokens_core_remain +
                           block_length * (GetBlockIdx() - this->tokens_core_remain);
        }
    } else {
        block_offset = block_length * GetBlockIdx();
    }

    this->tokensGM.SetGlobalBuffer((__gm__ T1*)permuted_tokens);
    this->indicesGM.SetGlobalBuffer((__gm__ T2*)sorted_indices + block_offset, block_length);

    int64_t out_block_offset;
    if (this->tokens_core_remain > 0) {
        if (GetBlockIdx() < this->tokens_core_remain) {
            out_block_offset = this->tokens_core_length * GetBlockIdx() * hidden_size;
        } else {
            out_block_offset = (this->tokens_core_length + 1) * this->tokens_core_remain +
                               this->tokens_core_length * (GetBlockIdx() - this->tokens_core_remain);
            out_block_offset *= this->hidden_size;
        }
    } else {
        out_block_offset = this->tokens_core_length * GetBlockIdx() * hidden_size;
    }

    this->outGM.SetGlobalBuffer(
        (__gm__ T1*)unpermuted_tokens + out_block_offset, this->tokens_core_length * this->hidden_size);

    pipe->InitBuffer(tokens_inque, tiling_data->buffer_num, hidden_splited_length_align512 * sizeof(T1));
    pipe->InitBuffer(indices_inque, 1, block_splited_length * (sizeof(T2)));
    pipe->InitBuffer(outque, 1, hidden_splited_length_align512 * sizeof(T1));

    if constexpr (!IsSameType<T1, float>::value) {
        pipe->InitBuffer(temp_buffer0, hidden_splited_length_align512 * sizeof(float) + 256);
        pipe->InitBuffer(temp_buffer1, hidden_splited_length_align512 * sizeof(float));
        this->token_tensor0 = this->temp_buffer0.template Get<float>();
        this->token_tensor1 = this->temp_buffer1.template Get<float>();
    }

    if constexpr (PROBS) {
        this->probsGM.SetGlobalBuffer((__gm__ T3*)probs + block_offset, block_length);
        pipe->InitBuffer(probs_inque, 1, block_splited_length * (sizeof(T3)));
        if constexpr (!IsSameType<T3, float>::value) {
            pipe->InitBuffer(temp_buffer2, block_splited_length * sizeof(float));
            this->probs_tensor = this->temp_buffer2.template Get<float>();
        }
    }
};

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::Process()
{
    for (int64_t i = 0; i < this->tokens_splited_num; ++i) {
        CalMultiOutToken(i * this->tokens_splited_length, this->tokens_splited_length);
    }
    // 处理tokens_num不能均匀分核数的尾块
    if (this->tokens_splited_remain > 0) {
        CalMultiOutToken(this->tokens_splited_num * this->tokens_splited_length, this->tokens_splited_remain);
    }
}

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::CalMultiOutToken(
    const int64_t out_offset, const int64_t out_tokens_number)
{
    this->indicesLocal = this->indices_inque.template AllocTensor<T2>();
    int64_t in_offset = out_offset * this->top_k;
    this->copyParams.blockLen = out_tokens_number * this->top_k * sizeof(T2);
#if __CCE_AICORE__ == 200
    MoeTokenUnPermute::DataCopyPadCustom(this->indicesLocal, this->indicesGM[in_offset], this->copyParams, this->extParams2);
#else
    DataCopyPad(this->indicesLocal, this->indicesGM[in_offset], this->copyParams, this->extParams2);
#endif
    this->indices_inque.template EnQue(this->indicesLocal);

    if constexpr (PROBS) {
        LocalTensor<T3> temp_probs_tensor = this->probs_inque.template AllocTensor<T3>();
        this->copyParams.blockLen = out_tokens_number * this->top_k * sizeof(T3);
#if __CCE_AICORE__ == 200
        MoeTokenUnPermute::DataCopyPadCustom(temp_probs_tensor, this->probsGM[in_offset], this->copyParams, this->extParams3);
#else
        DataCopyPad(temp_probs_tensor, this->probsGM[in_offset], this->copyParams, this->extParams3);
#endif
        this->probs_inque.template EnQue(temp_probs_tensor);
        temp_probs_tensor = this->probs_inque.template DeQue<T3>();
        if constexpr (!IsSameType<T3, float>::value) {
            Cast(this->probs_tensor, temp_probs_tensor, RoundMode::CAST_NONE, out_tokens_number * this->top_k);
            this->probs_inque.FreeTensor(temp_probs_tensor);
            PipeBarrier<PIPE_V>();
        } else {
            this->probs_tensor = temp_probs_tensor;
        }
    }
    this->indicesLocal = this->indices_inque.template DeQue<T2>();

#pragma bisheng auto_sync parallel
    for (int64_t out_token_idx = 0; out_token_idx < out_tokens_number; ++out_token_idx) {
        CalSingleOutToken(out_token_idx * this->top_k, out_offset + out_token_idx);
    }
    // Free Tensor
    this->indices_inque.FreeTensor(this->indicesLocal);
    if constexpr (PROBS && IsSameType<T3, float>::value) {
        this->probs_inque.FreeTensor(this->probs_tensor);
    }
}

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::CalSingleOutToken(
    const int64_t start_token, const int64_t out_token_idx)
{
    for (int64_t h_index = 0; h_index < this->hidden_splited_num; ++h_index) {
        CalPartOutToken(start_token, h_index, this->hidden_splited_length, out_token_idx);
    }
    // 一次不能完整容纳完整的hidden_size, 处理尾块
    if (this->hidden_splited_remain > 0) {
        CalPartOutToken(start_token, this->hidden_splited_num, this->hidden_splited_remain, out_token_idx);
    }
}

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::CalPartOutToken(
    const int64_t start_token, const int64_t h_index, const int64_t h_length, const int64_t out_token_index)
{
    if constexpr (IsSameType<T1, float>::value) {
        this->token_tensor0 = this->outque.template AllocTensor<T1>();
    }
    int64_t end_token = start_token + this->top_k;
    bool needCopyIn = false;
    float prob_value = 0;
    T2 acl_token_idx = this->indicesLocal.GetValue(start_token);
    if constexpr (PROBS) {
        prob_value = this->probs_tensor.GetValue(start_token);
        needCopyIn = acl_token_idx < this->num_out_tokens && prob_value != 0;
    } else {
        needCopyIn = acl_token_idx < this->num_out_tokens;
    }
    // 处理第一个Token数据
    if (needCopyIn) {
        CopyTokenIn(acl_token_idx, h_index, h_length);
        PipeBarrier<PIPE_V>();
        CalFirstToken(prob_value, h_length);
    } else {
        PipeBarrier<PIPE_V>();
        Duplicate(this->token_tensor0, static_cast<float>(0), h_length);
    }
    // 处理剩余的Token数据
    for (int64_t token_index = start_token + 1; token_index < end_token; ++token_index) {
        acl_token_idx = this->indicesLocal.GetValue(token_index);
        if constexpr (PROBS) {
            prob_value = this->probs_tensor.GetValue(token_index);
            needCopyIn = acl_token_idx < this->num_out_tokens && prob_value != 0;
        } else {
            needCopyIn = acl_token_idx < this->num_out_tokens;
        }
        if (needCopyIn) {
            CopyTokenIn(acl_token_idx, h_index, h_length);
            PipeBarrier<PIPE_V>();
            CalToken(prob_value, h_length);
        }
    }
    // 输出计算结果
    CopyOut(out_token_index, h_index, h_length);
}

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::CopyTokenIn(
    const T2 in_token_index, const int64_t h_index, const int64_t h_length)
{
    LocalTensor<T1> tokensLocal = this->tokens_inque.template AllocTensor<T1>();
    int64_t offset = in_token_index * this->hidden_size + h_index * this->hidden_splited_length;

    if (likely((h_length * sizeof(T1)) % BLOCK_SIZE == 0)) {
        DataCopy(tokensLocal, this->tokensGM[offset], h_length);
    } else {
        this->copyParams.blockLen = h_length * sizeof(T1);
#if __CCE_AICORE__ == 200
        MoeTokenUnPermute::DataCopyPadCustom(tokensLocal, this->tokensGM[offset], this->copyParams, this->extParams1);
#else
        DataCopyPad(tokensLocal, this->tokensGM[offset], this->copyParams, this->extParams1);
#endif
    }

    this->tokens_inque.template EnQue(tokensLocal);
}

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::CalFirstToken(
    const float prob_value, const int64_t h_length)
{
    LocalTensor<T1> tokensLocal = this->tokens_inque.template DeQue<T1>();

    if constexpr (!IsSameType<T1, float>::value) {
        Cast(this->token_tensor0, tokensLocal, RoundMode::CAST_NONE, h_length);
    } else {
        uint64_t byteAlign32 = (h_length * sizeof(float) + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
        DataCopy(this->token_tensor0, tokensLocal, byteAlign32 / sizeof(float));
    }

    this->tokens_inque.FreeTensor(tokensLocal);

    if constexpr (PROBS) {
        PipeBarrier<PIPE_V>();
        Muls(this->token_tensor0, this->token_tensor0, prob_value, h_length);
    }
}

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::CalToken(
    const float prob_value, const int64_t h_length)
{
    LocalTensor<T1> tokensLocal = this->tokens_inque.template DeQue<T1>();

    if constexpr (!IsSameType<T1, float>::value) {
        Cast(this->token_tensor1, tokensLocal, RoundMode::CAST_NONE, h_length);
        this->tokens_inque.FreeTensor(tokensLocal);
        if constexpr (PROBS) {
            PipeBarrier<PIPE_V>();
            Muls(this->token_tensor1, this->token_tensor1, prob_value, h_length);
        }
        PipeBarrier<PIPE_V>();
        Add(this->token_tensor0, this->token_tensor0, this->token_tensor1, h_length);
    } else {
        if constexpr (PROBS) {
            Muls(tokensLocal, tokensLocal, prob_value, h_length);
            PipeBarrier<PIPE_V>();
        }
        Add(this->token_tensor0, this->token_tensor0, tokensLocal, h_length);
        this->tokens_inque.FreeTensor(tokensLocal);
    }
}

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::CopyOut(
    const int64_t out_token_index, const int64_t h_index, const int64_t h_length)
{
    LocalTensor<T1> temp_out_tensors;
    if constexpr (!IsSameType<T1, float>::value) {
        temp_out_tensors = this->outque.template AllocTensor<T1>();
        PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
        Cast(temp_out_tensors, this->token_tensor0, RoundMode::CAST_NONE, h_length);
#else
        Cast(temp_out_tensors, this->token_tensor0, RoundMode::CAST_RINT, h_length);
#endif
    } else {
        temp_out_tensors = this->token_tensor0;
    }

    this->outque.template EnQue<T1>(temp_out_tensors);
    temp_out_tensors = this->outque.template DeQue<T1>();

    int64_t offset = out_token_index * this->hidden_size + h_index * this->hidden_splited_length;
    if (likely((h_length * sizeof(T1)) % BLOCK_SIZE == 0)) {
        DataCopy(this->outGM[offset], temp_out_tensors, h_length);
    } else {
        this->copyParams.blockLen = h_length * sizeof(T1);
#if __CCE_AICORE__ == 200
        MoeTokenUnPermute::DataCopyCustom(this->outGM[offset], temp_out_tensors, this->copyParams.blockCount, this->copyParams.blockLen);
#else
        DataCopyPad(this->outGM[offset], temp_out_tensors, this->copyParams);
#endif
    }

    this->outque.FreeTensor(temp_out_tensors);
}
#endif // MOE_TOKEN_UNPERMUTE
