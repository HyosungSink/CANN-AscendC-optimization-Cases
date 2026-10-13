/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class LightningIndexerProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LightningIndexerProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LightningIndexerProto TearDown" << std::endl;
    }
};

// BSND_BSND
TEST_F(LightningIndexerProto, LightningIndexer_infershape_0)
{
    int64_t actual_seq_qlist[] = {509};
    int64_t actual_seq_kvlist[] = {1111};
    gert::InfershapeContextPara infershapeContextPara(
        "LightningIndexer",
        // 输入Tensor
        {
            {{{1, 1856, 8, 128}, {1, 1856, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},       // query        input0
            {{{1, 131072, 1, 128}, {1, 131072, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // key          input1
            {{{1, 1856, 8}, {1, 1856, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},                // weights      input2
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},          // actual_seq_lengths_query
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},         // actual_seq_lengths_key 
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}                                   // block_table  input3
        },
        {// 输出Tensor
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},    // sparse_indices
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}      // sparse_values
        },
        {// 属性
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"return_values", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape = { // 预期输出
        {1, 1856, 1, 1024},
        {1, 1856, 1, 1024}
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);   // 比对成功返回SUCCESS
}

// PA_TND
TEST_F(LightningIndexerProto, LightningIndexer_infershape_1)
{
    int64_t actual_seq_qlist[] = {1};
    int64_t actual_seq_kvlist[] = {8191};
    gert::InfershapeContextPara infershapeContextPara(
        "LightningIndexer",
        // 输入Tensor
        {
            {{{1, 64, 128}, {1, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},                 // query        input0
            {{{1, 1, 8191, 128}, {1, 1, 8191, 128}}, ge::DT_BF16, ge::FORMAT_ND},       // key          input1
            {{{1, 64}, {1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                           // weights      input2
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},          // actual_seq_lengths_query
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},         // actual_seq_lengths_key 
            {{{1, 32}, {1, 32}}, ge::DT_INT32, ge::FORMAT_ND}                           // block_table  input3
        },
        {// 输出Tensor
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},    // sparse_indices
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}      // sparse_values
        },
        {// 属性
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"return_values", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape = { // 预期输出
        {1, 8191, 2048},
        {0}
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);   // 比对成功返回SUCCESS
}

// TND_TND
TEST_F(LightningIndexerProto, LightningIndexer_infershape_2)
{
    int64_t actual_seq_qlist[] = {73, 111, 209, 331, 403, 420};
    int64_t actual_seq_kvlist[] = {224, 448, 672, 896, 1120, 1344};
    gert::InfershapeContextPara infershapeContextPara(
        "LightningIndexer",
        // 输入Tensor
        {
            {{{420, 64, 128}, {420, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},         // query        input0
            {{{1344, 1, 128}, {1344, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},         // key          input1
            {{{420, 64}, {420, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // weights      input2
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_query
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_key 
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}                               // block_table  input3
        },
        {// 输出Tensor
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},    // sparse_indices
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}      // sparse_values
        },
        {// 属性
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5120)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"return_values", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape = { // 预期输出
        {420, 1, 5120},
        {420, 1, 5120}
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);   // 比对成功返回SUCCESS
}

// infer dataType
TEST_F(LightningIndexerProto, LightningIndexer_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("LightningIndexer")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_ref0 = ge::DT_FLOAT16;
        ge::DataType input_ref1 = ge::DT_FLOAT;
        ge::DataType input_ref2 = ge::DT_INT32;
        ge::DataType output_ref0 = ge::DT_INT32;
        ge::DataType output_ref1 = ge::DT_FLOAT16;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(6, 2)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref0, &input_ref0, &input_ref1, &input_ref2, &input_ref2, &input_ref2})
                .NodeAttrs({   
                    {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                    {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                    {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
                    {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                    {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                    {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                    {"return_values", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
                })
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref0);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref1);

    }
}