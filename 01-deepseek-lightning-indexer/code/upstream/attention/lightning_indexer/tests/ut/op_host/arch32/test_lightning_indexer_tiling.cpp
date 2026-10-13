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

class LightningIndexerTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LightningIndexerTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LightningIndexerTiling TearDown" << std::endl;
    }
};

// when key layout is not PA_BSND, input block_table must be null
TEST_F(LightningIndexerTiling, LightningIndexer_910b_tiling_0)
{
    struct LightningIndexerCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {509};
    int64_t actual_seq_kvlist[] = {1111};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexer",
        {
            {{{1, 1856, 8, 128}, {1, 1856, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},       // query        input0
            {{{1, 131072, 1, 128}, {1, 131072, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // key          input1
            {{{1, 1856, 8}, {1, 1856, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},                // weights      input2
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},          // actual_seq_lengths_query
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},         // actual_seq_lengths_key 
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}                                   // block_table  input3
        },
        {
            {{{1, 1856, 1, 1024}, {1, 1856, 1, 1024}}, ge::DT_INT32, ge::FORMAT_ND},    // sparse_indices
            {{{1, 1856, 1, 1024}, {1, 1856, 1, 1024}}, ge::DT_BF16, ge::FORMAT_ND}      // sparse_values
        },
        {
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"return_values", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// key shape[2] is numhead, only support 1.
TEST_F(LightningIndexerTiling, LightningIndexer_910b_tiling_1)
{
    struct LightningIndexerCompileInfo {} compileInfo;
    int64_t* actual_seq_qlist = nullptr;
    int64_t actual_seq_kvlist[] = {16484, 16484, 16484, 16484};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexer",
        {
            {{{4, 16484, 64, 128}, {4, 16484, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // query        input0
            {{{4, 1, 16484, 128}, {4, 1, 16484, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // key          input1
            {{{4, 16484, 64}, {4, 16484, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // weights      input2
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},                  // actual_seq_lengths_query
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},                 // actual_seq_lengths_key 
            {{{4, 129}, {4, 129}}, ge::DT_INT32, ge::FORMAT_ND}                                 // block_table  input3
        },
        {
            {{{4, 16484, 1, 3072}, {4, 16484, 1, 3072}}, ge::DT_INT32, ge::FORMAT_ND},      // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND}                                     // sparse_values
        },
        {
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3072)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"return_values", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// when key layout is not PA_BSND, input block_table must be null
TEST_F(LightningIndexerTiling, LightningIndexer_910b_tiling_2)
{
    struct LightningIndexerCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {64, 128, 192, 256, 320, 384, 448, 512};
    int64_t actual_seq_kvlist[] = {64, 128, 192, 256, 320, 384, 448, 512};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexer",
        {
            {{{512, 8, 128}, {420, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},         // query        input0
            {{{512, 1, 128}, {1344, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},         // key          input1
            {{{512, 8}, {420, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // weights      input2
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},     // actual_seq_lengths_query
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},    // actual_seq_lengths_key 
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}                              // block_table  input3
        },
        {
            {{{512, 1, 3072}, {512, 1, 3072}}, ge::DT_INT32, ge::FORMAT_ND},        // sparse_indices
            {{{512, 1, 3072}, {512, 1, 3072}}, ge::DT_BF16, ge::FORMAT_ND}          // sparse_values
        },
        {
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3072)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"return_values", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// key shape[2] is numhead, only support 1.
TEST_F(LightningIndexerTiling, LightningIndexer_910b_tiling_3)
{
    struct LightningIndexerCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192};
    int64_t actual_seq_kvlist[] = {8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexer",
        {
            {{{24, 8192, 64, 128}, {24, 8192, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},           // query        input0
            {{{24, 1, 8192, 128}, {24, 1, 8192, 128}}, ge::DT_BF16, ge::FORMAT_ND},             // key          input1
            {{{24, 8192, 64}, {24, 8192, 64}}, ge::DT_BF16, ge::FORMAT_ND},                     // weights      input2
            {{{24}, {24}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},                // actual_seq_lengths_query
            {{{24}, {24}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},               // actual_seq_lengths_key 
            {{{24, 32}, {24, 32}}, ge::DT_INT32, ge::FORMAT_ND}                                 // block_table  input3
        },
        {
            {{{24, 8192, 1, 2048}, {24, 8192, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},          // sparse_indices
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND}                                            // sparse_values
        },
        {
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"return_values", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// when key layout is not PA_BSND, input block_table must be null
TEST_F(LightningIndexerTiling, LightningIndexer_910b_tiling_4)
{
    struct LightningIndexerCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {33, 33};
    int64_t actual_seq_kvlist[] = {137, 2304};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexer",
        {
            {{{2, 3072, 64, 128}, {2, 3072, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},         // query        input0
            {{{2, 2304, 1, 128}, {2, 2304, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},           // key          input1
            {{{2, 3072, 64}, {2, 3072, 64}}, ge::DT_BF16, ge::FORMAT_ND},                   // weights      input2
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},              // actual_seq_lengths_query
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},             // actual_seq_lengths_key 
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}                                       // block_table  input3
        },
        {
            {{{2, 3072, 1, 2048}, {2, 3072, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},        // sparse_indices
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND}                                        // sparse_values
        },
        {
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"return_values", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// input key's block_size must be a multiple of 16 and belong to (0, 1024].
TEST_F(LightningIndexerTiling, LightningIndexer_910b_tiling_5)
{
    struct LightningIndexerCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {39, 39};
    int64_t actual_seq_kvlist[] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexer",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},         // query        input0
            {{{2, 1, 1, 128}, {2, 1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},             // key          input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_BF16, ge::FORMAT_ND},                   // weights      input2
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},          // actual_seq_lengths_query
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},         // actual_seq_lengths_key 
            {{{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND}                             // block_table  input3
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},        // sparse_indices
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND}                                    // sparse_values
        },
        {
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"return_values", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// key Dim(2)=1 Dim(1)=16 success
TEST_F(LightningIndexerTiling, LightningIndexer_910b_tiling_6)
{
    struct LightningIndexerCompileInfo {} compileInfo;
    int64_t actual_seq_qlist[] = {39, 39};
    int64_t actual_seq_kvlist[] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexer",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},         // query        input0
            {{{2, 16, 1, 128}, {2, 16, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},           // key          input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_BF16, ge::FORMAT_ND},                   // weights      input2
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},          // actual_seq_lengths_query
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},         // actual_seq_lengths_key 
            {{{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND}                             // block_table  input3
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},        // sparse_indices
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND}                                    // sparse_values
        },
        {
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"return_values", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        },
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 1090722587;
    std::string expectTilingData =
        "2 167503724608 8796093022224 68719476800 12884901889 "
        "9223372036854775807 9223372036854775807 0 ";
    std::vector<size_t> expectWorkspaces = {167903232};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}
