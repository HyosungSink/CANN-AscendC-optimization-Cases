/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/rotary_position_embedding_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class RotaryPositionEmbeddingTiling : public testing::Test {
public:
    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 24}
                                    })";

protected:
    static void SetUpTestCase()
    {
        std::cout << "RotaryPositionEmbeddingTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RotaryPositionEmbeddingTiling TearDown" << std::endl;
    }
};

TEST_F(RotaryPositionEmbeddingTiling, RotaryPositionEmbedding_fp16_001)
{
    optiling::RotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbedding",
                                              {
                                                  // input info
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1032;
    string expectTilingData = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "3 8192 1 2 64 32 32 64 1 64 84 5376 5376 64 1 0 1 128 64 8192 4096 64 64 0 1 0 1 128 64 "
                              "64 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RotaryPositionEmbeddingTiling, RotaryPositionEmbedding_bf16_001)
{
    optiling::RotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbedding",
                                              {
                                                  // input info
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1033;
    string expectTilingData = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "3 8192 1 2 64 32 32 64 1 64 84 5376 5376 64 1 0 1 128 64 8192 4096 64 64 0 1 0 1 128 64 "
                              "64 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RotaryPositionEmbeddingTiling, RotaryPositionEmbedding_fp32_001)
{
    optiling::RotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbedding",
                                              {
                                                  // input info
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1031;
    string expectTilingData = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "3 8192 1 2 64 32 32 64 1 64 126 8064 8064 64 1 0 1 128 64 8192 4096 64 64 0 1 0 1 128 "
                              "64 64 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RotaryPositionEmbeddingTiling, RotaryPositionEmbedding_fp32_001_tnd)
{
    optiling::RotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbedding",
                                              {
                                                  // input info
                                                  {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 1, 64}, {64, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 1, 64}, {64, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1031;
    string expectTilingData = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "3 8192 1 2 64 32 32 64 1 64 126 8064 8064 64 1 0 1 128 64 8192 4096 64 64 0 1 0 1 128 "
                              "64 64 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test rotate matrix mode with BNSD layout
// Input: x=(1, 24, 32, 128) bf16, cos/sin=(1, 1, 32, 128) bf16, rotate=(128, 128) bf16
TEST_F(RotaryPositionEmbeddingTiling, RotaryPositionEmbedding_rotate_matrix_bf16_001)
{
    optiling::RotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbedding",
                                              {
                                                  // input info
                                                  {{{1, 24, 32, 128}, {1, 24, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 1, 32, 128}, {1, 1, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 1, 32, 128}, {1, 1, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 128}, {128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1, 24, 32, 128}, {1, 24, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 3013;

    string expectTilingData = "137438953473 549755814016 137438953600 549755814016 549755813920 8589934720 4294967298 "
                              "1 0 175921860444160 16384 4294967297 4294967297 17179869188 0 8589934594 2 0 0 0 0 0 0 0 "
                              "0 1 1 1 8 2 98304 0 0 1 128 32 128 128 64 32 1 24 32 1 1 32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {50331648};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test rotate matrix mode with variable S dimension
// Verify S dimension variability (S=64 vs S=32 in test_001)
TEST_F(RotaryPositionEmbeddingTiling, RotaryPositionEmbedding_rotate_matrix_bf16_002_variable_s)
{
    optiling::RotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbedding",
                                              {
                                                  // input info
                                                  {{{1, 24, 64, 128}, {1, 24, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 1, 64, 128}, {1, 1, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 1, 64, 128}, {1, 1, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 128}, {128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1, 24, 64, 128}, {1, 24, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 3013;
    // blockNumM changes (from ceil(32/32)=1 to ceil(64/32)=2), blockNum changes to 8
    string expectTilingData = "274877906945 549755814016 274877907072 549755814016 549755813952 8589934720 4294967298 "
                              "1 0 211106232532992 32768 4294967297 4294967297 17179869188 0 8589934594 2 0 0 0 0 0 0 0 "
                              "0 1 1 1 8 2 196608 0 0 1 128 64 128 128 64 64 1 24 64 1 1 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {50331648};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test rotate matrix mode with multiple batches
// Verify batch dimension variability (B=2 vs B=1 in test_001)
TEST_F(RotaryPositionEmbeddingTiling, RotaryPositionEmbedding_rotate_matrix_bf16_003_multi_batch)
{
    optiling::RotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbedding",
                                              {
                                                  // input info
                                                  {{{2, 24, 32, 128}, {2, 24, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 1, 32, 128}, {1, 1, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 1, 32, 128}, {1, 1, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 128}, {128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{2, 24, 32, 128}, {2, 24, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 3013;
    string expectTilingData = "137438953473 549755814016 137438953600 549755814016 549755813920 8589934720 4294967298 "
                              "1 0 175921860444160 16384 4294967297 4294967297 17179869188 0 8589934594 2 0 0 0 0 0 0 0 "
                              "0 1 1 1 8 2 196608 0 0 1 128 32 128 128 64 32 2 24 32 1 1 32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {50331648};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RotaryPositionEmbeddingTiling, RotaryPositionEmbedding_rotate_matrix_bf16_002)
{
    optiling::RotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbedding",
                                              {
                                                  // input info
                                                  {{{1, 24, 32, 128}, {1, 24, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 1, 32, 128}, {1, 1, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 1, 32, 128}, {1, 1, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 128}, {128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1, 24, 32, 128}, {1, 24, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 3013;

    string expectTilingData = "137438953473 549755814016 137438953600 549755814016 549755813920 8589934720 4294967298 "
                              "1 0 175921860444160 16384 4294967297 4294967297 17179869188 0 8589934594 2 0 0 0 0 0 0 0 "
                              "0 1 1 1 8 2 98304 0 0 1 128 32 128 128 64 32 1 24 32 1 1 32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {50331648};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RotaryPositionEmbeddingTiling, RotaryPositionEmbedding_rotate_matrix_fp16_002)
{
    optiling::RotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbedding",
                                              {
                                                  // input info
                                                  {{{1, 24, 32, 128}, {1, 24, 32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 32, 128}, {1, 1, 32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1, 32, 128}, {1, 1, 32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{128, 128}, {128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1, 24, 32, 128}, {1, 24, 32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 3012;

    string expectTilingData = "137438953473 549755814016 137438953600 549755814016 549755813920 8589934720 4294967298 "
                              "1 0 175921860444160 16384 4294967297 4294967297 17179869188 0 8589934594 2 0 0 0 0 0 0 0 "
                              "0 1 1 1 8 2 98304 0 0 1 128 32 128 128 64 32 1 24 32 1 1 32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {50331648};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RotaryPositionEmbeddingTiling, RotaryPositionEmbedding_rotate_matrix_fp32_002)
{
    optiling::RotaryPositionEmbeddingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("RotaryPositionEmbedding",
                                              {
                                                  // input info
                                                  {{{1, 24, 32, 128}, {1, 24, 32, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1, 32, 128}, {1, 1, 32, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1, 32, 128}, {1, 1, 32, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // output info
                                                  {{{1, 24, 32, 128}, {1, 24, 32, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // attr
                                                  {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 3011;

    string expectTilingData = "137438953473 549755814016 137438953600 549755814016 549755813920 8589934656 4294967298 "
                              "1 0 351843720888320 16384 4294967297 4294967297 17179869188 0 8589934594 2 0 0 0 0 0 0 0 "
                              "0 1 1 1 8 2 98304 0 0 1 128 32 128 128 64 32 1 24 32 1 1 32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {50331648};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}