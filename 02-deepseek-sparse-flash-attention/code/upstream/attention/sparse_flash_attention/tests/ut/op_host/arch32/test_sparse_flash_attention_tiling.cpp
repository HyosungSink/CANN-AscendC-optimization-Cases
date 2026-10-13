/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
using namespace std;

class SparseFlashAttentionTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseFlashAttentionTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseFlashAttentionTiling TearDown" << std::endl;
    }
};

// Shape of tensor softmaxMax is nullptr
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_0)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {1, 1, 1, 1};
    int64_t actual_seq_kvlist[] = {65536, 65536, 65536, 65536};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // query            input0
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // key              input1
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // value            input2
            {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},              // sparse_indices   input3
            {{{4, 1024}, {4, 1024}}, ge::DT_INT32, ge::FORMAT_ND},                      // block_table      input4
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},          // actual_seq_lengths_query
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},         // actual_seq_lengths_kv 
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query_rope       input5
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}      // key_rope         input6
        },
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // attention_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // softmax_max
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                 // softmax_sum
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Shape of queryRope is nullptr
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_1)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {1, 2};
    int64_t actual_seq_kvlist[] = {1, 2};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{2, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // query            input0
            {{{2, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // key              input1
            {{{2, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // value            input2
            {{{2, 1, 206}, {4, 1, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},            // sparse_indices   input3
            {{{4, 1024}, {4, 1024}}, ge::DT_INT32, ge::FORMAT_ND},                  // block_table      input4
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_query
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv 
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                 // query_rope       input5
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}                                  // key_rope         input6
        },
        {
            {{{2, 128, 512}, {2, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},       // attention_out
            {{{1, 2, 128}, {1, 2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},          // softmax_max
            {{{1, 2, 128}, {1, 2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}           // softmax_sum
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Shape of tensor softmaxMax is nullptr
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_2)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {1, 1, 1, 1};
    int64_t actual_seq_kvlist[] = {65536, 65536, 65536, 65536};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // query            input0
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // key              input1
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // value            input2
            {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},              // sparse_indices   input3
            {{{4, 1024}, {4, 1024}}, ge::DT_INT32, ge::FORMAT_ND},                      // block_table      input4
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},          // actual_seq_lengths_query
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},         // actual_seq_lengths_kv 
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query_rope       input5
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}      // key_rope         input6
        },
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // attention_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // softmax_max
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                 // softmax_sum
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// The dimensions of query and query_rope should be equal, but query has dimension 3 while query_rope has dimension 4.
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_3)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {18432, 36864, 55296, 73728, 92160, 110592, 129024, 147456, 165888, 184320};
    int64_t actual_seq_kvlist[] = {37748, 40836, 49869, 64087, 19343, 52909, 4619, 22910, 7926, 65488};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{184320, 4, 512}, {184320, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // query            input0
            {{{10, 65488, 1, 512}, {10, 65488, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // key              input1
            {{{10, 65488, 1, 512}, {10, 65488, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // value            input2
            {{{184320, 1, 16}, {184320, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},          // sparse_indices   input3
            {{{10, 128}, {10, 128}}, ge::DT_INT32, ge::FORMAT_ND},                      // block_table      input4
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},        // actual_seq_lengths_query
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},       // actual_seq_lengths_kv 
            {{{753, 512, 1, 512}, {753, 512, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query_rope       input5
            {{{753, 512, 1, 512}, {753, 512, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}      // key_rope         input6
        },
        {
            {{{184320, 4, 512}, {184320, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // attention_out
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // softmax_max
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                                   // softmax_sum
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Shape of tensor softmaxMax is nullptr
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_4)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {1, 1, 1, 1};
    int64_t actual_seq_kvlist[] = {65536, 65536, 65536, 65536};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // query            input0
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // key              input1
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // value            input2
            {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},              // sparse_indices   input3
            {{{4, 1024}, {4, 1024}}, ge::DT_INT32, ge::FORMAT_ND},                      // block_table      input4
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},          // actual_seq_lengths_query
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},         // actual_seq_lengths_kv 
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query_rope       input5
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}      // key_rope         input6
        },
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // attention_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // softmax_max
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                 // softmax_sum
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Shape of tensor softmaxMax is nullptr
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_5)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {1, 1, 1, 1};
    int64_t actual_seq_kvlist[] = {65536, 65536, 65536, 65536};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // query            input0
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // key              input1
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // value            input2
            {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},              // sparse_indices   input3
            {{{4, 1024}, {4, 1024}}, ge::DT_INT32, ge::FORMAT_ND},                      // block_table      input4
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},          // actual_seq_lengths_query
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},         // actual_seq_lengths_kv 
            {{{4096, 64, 1, 64}, {4096, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope       input5
            {{{4096, 64, 1, 64}, {4096, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}        // key_rope         input6
        },
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // attention_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // softmax_max
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                 // softmax_sum
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// query_rope layout is BSND, shape is [1, 128, 1, 64], expected shape is [1, 128, 128, 64].
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_6)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {113};
    int64_t actual_seq_kvlist[] = {113};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // query            input0
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},             // key              input1
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},             // value            input2
            {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},          // sparse_indices   input3
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},                                // block_table      input4
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},              // actual_seq_lengths_query
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},             // actual_seq_lengths_kv 
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},               // query_rope       input5
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}                // key_rope         input6
        },
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // attention_out
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                    // softmax_max
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                                     // softmax_sum
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 0 < blockSize_ < 1024 success
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_7)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {113};
    int64_t actual_seq_kvlist[] = {113};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // query            input0
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},             // key              input1
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},             // value            input2
            {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},          // sparse_indices   input3
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},                                // block_table      input4
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},              // actual_seq_lengths_query
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},             // actual_seq_lengths_kv 
            {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},           // query_rope       input5
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}                // key_rope         input6
        },
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // attention_out
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // softmax_max
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                                   // softmax_sum
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 576;
    std::string expectTilingData =
        "549755813889 128 128 4407522836751581185 4294967424 1 3 "
        "9223372036854775807 9223372036854775807 2 1 2048 36028797018963968 "
        "262144 64 281474976727040 549755814016 ";
    std::vector<size_t> expectWorkspaces = {214302720};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// the layout_kv is PA_BSND, blockTable must be provided.
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_8)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {18432, 36864, 55296, 73728, 92160, 110592, 129024, 147456, 165888, 184320};
    int64_t actual_seq_kvlist[] = {37748, 40836, 49869, 64087, 19343, 52909, 4619, 22910, 7926, 65488};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{184320, 4, 512}, {184320, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // query            input0
            {{{10, 65488, 1, 512}, {10, 65488, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // key              input1
            {{{10, 65488, 1, 512}, {10, 65488, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // value            input2
            {{{184320, 1, 16}, {184320, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},          // sparse_indices   input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                    // block_table      input4
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},        // actual_seq_lengths_query
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},       // actual_seq_lengths_kv 
            {{{753, 512, 1, 512}, {753, 512, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query_rope       input5
            {{{753, 512, 1, 512}, {753, 512, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}      // key_rope         input6
        },
        {
            {{{184320, 4, 512}, {184320, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // attention_out
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // softmax_max
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                                   // softmax_sum
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// the dim num of block_table is 3, it should be 2.
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_9)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {18432, 36864, 55296, 73728, 92160, 110592, 129024, 147456, 165888, 184320};
    int64_t actual_seq_kvlist[] = {37748, 40836, 49869, 64087, 19343, 52909, 4619, 22910, 7926, 65488};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{184320, 4, 512}, {184320, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // query            input0
            {{{10, 65488, 1, 512}, {10, 65488, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // key              input1
            {{{10, 65488, 1, 512}, {10, 65488, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // value            input2
            {{{184320, 1, 16}, {184320, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},          // sparse_indices   input3
            {{{1, 2, 1}, {1, 2, 1}}, ge::DT_INT32, ge::FORMAT_ND},                      // block_table      input4
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},        // actual_seq_lengths_query
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},       // actual_seq_lengths_kv 
            {{{753, 512, 1, 512}, {753, 512, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query_rope       input5
            {{{753, 512, 1, 512}, {753, 512, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}      // key_rope         input6
        },
        {
            {{{184320, 4, 512}, {184320, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // attention_out
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // softmax_max
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                                   // softmax_sum
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// block_table's second dimension(-1) should be greater than 0
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_10)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {18432, 36864, 55296, 73728, 92160, 110592, 129024, 147456, 165888, 184320};
    int64_t actual_seq_kvlist[] = {37748, 40836, 49869, 64087, 19343, 52909, 4619, 22910, 7926, 65488};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{184320, 4, 512}, {184320, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // query            input0
            {{{10, 65488, 1, 512}, {10, 65488, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // key              input1
            {{{10, 65488, 1, 512}, {10, 65488, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // value            input2
            {{{184320, 1, 16}, {184320, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},          // sparse_indices   input3
            {{{1, -1}, {1, -1}}, ge::DT_INT32, ge::FORMAT_ND},                          // block_table      input4
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},        // actual_seq_lengths_query
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},       // actual_seq_lengths_kv 
            {{{753, 512, 1, 512}, {753, 512, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query_rope       input5
            {{{753, 512, 1, 512}, {753, 512, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}      // key_rope         input6
        },
        {
            {{{184320, 4, 512}, {184320, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // attention_out
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // softmax_max
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                                   // softmax_sum
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Shape of tensor softmaxMax is nullptr
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_11)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {1, 1, 1, 1};
    int64_t actual_seq_kvlist[] = {65536, 65536, 65536, 65536};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // query            input0
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // key              input1
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // value            input2
            {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},              // sparse_indices   input3
            {{{4, 1024}, {4, 1024}}, ge::DT_INT32, ge::FORMAT_ND},                      // block_table      input4
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},          // actual_seq_lengths_query
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},         // actual_seq_lengths_kv 
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query_rope       input5
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}      // key_rope         input6
        },
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // attention_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // softmax_max
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                                 // softmax_sum
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// when the layout_kv is BSND, block_table should be null
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_12)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {113};
    int64_t actual_seq_kvlist[] = {113};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // query            input0
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},             // key              input1
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},             // value            input2
            {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},          // sparse_indices   input3
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},                                // block_table      input4
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},              // actual_seq_lengths_query
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},             // actual_seq_lengths_kv 
            {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},           // query_rope       input5
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}                // key_rope         input6
        },
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // attention_out
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                                  // softmax_max
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                                   // softmax_sum
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// when return_softmax_lse is true, key layout do not support PA_BSND.
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_13)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {1, 1, 1, 1};
    int64_t actual_seq_kvlist[] = {65536, 65536, 65536, 65536};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1, 128, 128}, {4, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 1, 128, 128}, {4, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Tensor query only supports dtype DT_FLOAT16, DT_BFLOAT16, but got DT_FLOAT
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_14)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {113};
    int64_t actual_seq_kvlist[] = {113};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// The dimensions of query and query_rope should be equal, but query has dimension 2 while query_rope has dimension 4.
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_15)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {113};
    int64_t actual_seq_kvlist[] = {113};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{128, 512}, {128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// oftmax_max dimension is 3, expected dimension is 4.
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_16)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {113};
    int64_t actual_seq_kvlist[] = {113};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 128, 8}, {1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 128, 8}, {1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// query_rope layout is TND, shape is [2, 1, 64], expected shape is [2, 128, 64].
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_17)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {1, 2};
    int64_t actual_seq_kvlist[] = {1, 2};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{2, 128, 512}, {2, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 512}, {2, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 512}, {2, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 206}, {2, 1, 206}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{2, 1, 64}, {2, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 64}, {2, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {{{2, 128, 512}, {2, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 8, 1}, {2, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8, 1}, {2, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// split kv try
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_18)
{
    struct SparseFlashAttentionCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {1};
    int64_t actual_seq_kvlist[] = {8192};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{1, 1, 128, 512}, {1, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 8192, 1, 512}, {1, 8192, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 8192, 1, 512}, {1, 8192, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 1, 8192}, {1, 1, 1, 8192}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 8192, 1, 64}, {1, 8192, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}
        },
        {
            {{{1, 1, 128, 512}, {1, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 512;
    std::string expectTilingData =
        "35184372088833 1 0 4407522836751581184 4294967424 1 3 "
        "9223372036854775807 9223372036854775807 2 1 8192 36028797018963968 "
        "262144 64 281474976776192 2199023255680 ";
    std::vector<size_t> expectWorkspaces = {252051456};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}
