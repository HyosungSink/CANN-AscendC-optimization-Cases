/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class RotaryPositionEmbeddingInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RotaryPositionEmbeddingInfershape Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RotaryPositionEmbeddingInfershape Proto Test TearDown" << std::endl;
    }
};

TEST_F(RotaryPositionEmbeddingInfershape, RotaryPositionEmbedding_infer_shape_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotaryPositionEmbedding",
        {
            // input info
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 64, 2, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RotaryPositionEmbeddingInfershape, RotaryPositionEmbedding_infer_shape_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotaryPositionEmbedding",
        {
            // input info
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // attr
            {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 64, 2, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RotaryPositionEmbeddingInfershape, RotaryPositionEmbedding_infer_shape_fp32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotaryPositionEmbedding",
        {
            // input info
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // attr
            {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 64, 2, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RotaryPositionEmbeddingInfershape, RotaryPositionEmbedding_rotate_matrix_bf16_001)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotaryPositionEmbedding",
        {
            // input info
            {{{1, 24, 32, 128}, {1, 24, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 32, 128}, {1, 1, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
             {{{1, 1, 32, 128}, {1, 1, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 128}, {128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // attr
            {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 24, 32, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RotaryPositionEmbeddingInfershape, RotaryPositionEmbedding_rotate_matrix_bf16_002_variable_s)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotaryPositionEmbedding",
        {
            // input info
            {{{1, 24, 64, 128}, {1, 24, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 64, 128}, {1, 1, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 64, 128}, {1, 1, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 128}, {128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // attr
            {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 24, 64, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RotaryPositionEmbeddingInfershape, RotaryPositionEmbedding_rotate_matrix_bf16_003_multi_batch)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RotaryPositionEmbedding",
        {
            // input info
            {{{2, 24, 32, 128}, {2, 24, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 32, 128}, {1, 1, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 32, 128}, {1, 1, 32, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 128}, {128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // attr
            {"mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 24, 32, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}