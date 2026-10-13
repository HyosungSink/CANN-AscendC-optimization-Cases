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

class SparseFlashAttentionProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseFlashAttentionProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseFlashAttentionProto TearDown" << std::endl;
    }
};

// BNSD
TEST_F(SparseFlashAttentionProto, SparseFlashAttention_infershape_0)
{
    int64_t actual_seq_qlist[] = {1, 1, 1, 1};
    int64_t actual_seq_kvlist[] = {65536, 65536, 65536, 65536};
    gert::InfershapeContextPara infershapeContextPara(
        "SparseFlashAttention",
        // 输入Tensor
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
        {// 输出Tensor
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},     // attention_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // softmax_max
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}     // softmax_sum
        },
        {// 属性
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
        });

    std::vector<std::vector<int64_t>> expectOutputShape = { // 预期输出
        {4, 1, 128, 512},
        {0},
        {0}
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);   // 比对成功返回SUCCESS
}

// TND_TND
TEST_F(SparseFlashAttentionProto, SparseFlashAttention_infershape_1)
{
    int64_t actual_seq_qlist[] = {1, 2};
    int64_t actual_seq_kvlist[] = {1, 2};
    gert::InfershapeContextPara infershapeContextPara(
        "SparseFlashAttention",
        // 输入Tensor
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
        {// 输出Tensor
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},     // attention_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // softmax_max
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}     // softmax_sum
        },
        {// 属性
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        });

    std::vector<std::vector<int64_t>> expectOutputShape = { // 预期输出
        {2, 128, 512},
        {1, 2, 128},
        {1, 2, 128}
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);   // 比对成功返回SUCCESS
}

// BNSD_BNSD
TEST_F(SparseFlashAttentionProto, SparseFlashAttention_infershape_2)
{
    int64_t actual_seq_qlist[] = {2944, 5760, 9728};
    int64_t actual_seq_kvlist[] = {2944, 5760, 9728};
    gert::InfershapeContextPara infershapeContextPara(
        "SparseFlashAttention",
        // 输入Tensor
        {
            {{{3, 13754, 8, 512}, {3, 13754, 8, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // query            input0
            {{{3, 13754, 1, 512}, {3, 13754, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // key              input1
            {{{3, 13754, 1, 512}, {3, 13754, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},        // value            input2
            {{{3, 13754, 1, 32}, {3, 13754, 1, 32}}, ge::DT_INT32, ge::FORMAT_ND},            // sparse_indices   input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                  // block_table      input4
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_query
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv 
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                 // query_rope       input5
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}                                  // key_rope         input6
        },
        {// 输出Tensor
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // attention_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},    // softmax_max
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}     // softmax_sum
        },
        {// 属性
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666)},
            {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}
        });

    std::vector<std::vector<int64_t>> expectOutputShape = { // 预期输出
        {3, 13754, 8, 512},
        {3, 1, 13754, 8},
        {3, 1, 13754, 8}
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);   // 比对成功返回SUCCESS
}

// infer dataType
TEST_F(SparseFlashAttentionProto, SparseFlashAttention_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("SparseFlashAttention")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_ref0 = ge::DT_BF16;
        ge::DataType input_ref1 = ge::DT_FLOAT16;
        ge::DataType input_ref2 = ge::DT_INT32;
        ge::DataType output_ref0 = ge::DT_BF16;
        ge::DataType output_ref1 = ge::DT_FLOAT;
        ge::DataType output_ref2 = ge::DT_FLOAT16;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(9, 3)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref1, &input_ref1, &input_ref1, &input_ref2, &input_ref2, &input_ref2, &input_ref2, &input_ref1, &input_ref1})
                .NodeAttrs({   
                    {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666)},
                    {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                    {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                    {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
                    {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                    {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                    {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                    {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                    {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}
                })
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref2);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref1);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref1);

    }
}
