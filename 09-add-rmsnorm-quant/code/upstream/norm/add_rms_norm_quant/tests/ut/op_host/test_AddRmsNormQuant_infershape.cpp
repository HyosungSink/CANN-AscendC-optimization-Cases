/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 *
 * @file test_AddRmsNormQuant_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "../../../op_graph/add_rms_norm_quant_proto.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

class AddRmsNormQuant : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddRmsNormQuant Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddRmsNormQuant Proto Test TearDown" << std::endl;
    }
};

TEST_F(AddRmsNormQuant, AddRmsNormQuant_infershape_case_0)
{
    ge::op::AddRmsNormQuant op;
    op.UpdateInputDesc("x1", create_desc({4, 1, 8}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x2", create_desc({4, 1, 8}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({8}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scales1", create_desc({8}, ge::DT_FLOAT));
    op.UpdateInputDesc("scales2", create_desc({8}, ge::DT_FLOAT));
    op.UpdateInputDesc("zero_points1", create_desc({8}, ge::DT_INT32));
    op.UpdateInputDesc("zero_points2", create_desc({8}, ge::DT_INT32));
    //   op.SetAttr("epsilon", 0.1);
    //   Runtime2TestParam param{{"epsilon"},{},{}};
    //   EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_x_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {4, 1, 8};
    std::vector<int64_t> expected_x_shape = {4, 1, 8};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_x_desc.GetShape().GetDims(), expected_x_shape);
}

TEST_F(AddRmsNormQuant, AddRmsNormQuant_InferDtype_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType scale_ref = ge::DT_FLOAT;
        ge::DataType zero_point_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_INT8;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(7)
                .NodeIoNum(7, 3)
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes(
                    {&input_ref, &input_ref, &input_ref, &scale_ref, &scale_ref, &zero_point_ref, &zero_point_ref})
                .OutputDataTypes({&output_ref, &output_ref, &input_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
        EXPECT_EQ(context->GetOutputDataType(2), input_ref);
    }
}

TEST_F(AddRmsNormQuant, AddRmsNormQuant_infershape_case_1)
{
    ge::op::AddRmsNormQuant op;
    op.UpdateInputDesc("x1", create_desc({4, 1, 16}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x2", create_desc({4, 1, 16}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({16}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scales1", create_desc({16}, ge::DT_FLOAT));
    op.UpdateInputDesc("scales2", create_desc({16}, ge::DT_FLOAT));
    op.UpdateInputDesc("zero_points1", create_desc({16}, ge::DT_INT32));
    op.UpdateInputDesc("zero_points2", create_desc({16}, ge::DT_INT32));
    //   op.SetAttr("epsilon", 0.1);
    //   Runtime2TestParam param{{"epsilon"},{},{}};
    //   EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_x_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {4, 1, 16};
    std::vector<int64_t> expected_x_shape = {4, 1, 16};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_x_desc.GetShape().GetDims(), expected_x_shape);
}

TEST_F(AddRmsNormQuant, AddRmsNormQuant_InferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType scale_ref = ge::DT_FLOAT;
        ge::DataType zero_point_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_INT8;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(7)
                .NodeIoNum(7, 3)
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                .InputDataTypes(
                    {&input_ref, &input_ref, &input_ref, &scale_ref, &scale_ref, &zero_point_ref, &zero_point_ref})
                .OutputDataTypes({&output_ref, &output_ref, &input_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
        EXPECT_EQ(context->GetOutputDataType(2), input_ref);
    }
}

TEST_F(AddRmsNormQuant, AddRmsNormQuant_infershape_case_2)
{
    ge::op::AddRmsNormQuant op;
    op.UpdateInputDesc("x1", create_desc({16, 1, 16}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x2", create_desc({16, 1, 16}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({16}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scales1", create_desc({16}, ge::DT_FLOAT));
    op.UpdateInputDesc("scales2", create_desc({16}, ge::DT_FLOAT));
    op.UpdateInputDesc("zero_points1", create_desc({16}, ge::DT_INT32));
    op.UpdateInputDesc("zero_points2", create_desc({16}, ge::DT_INT32));
    //   op.SetAttr("epsilon", 0.1);
    //   Runtime2TestParam param{{"epsilon"},{},{}};
    //   EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_x_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {16, 1, 16};
    std::vector<int64_t> expected_x_shape = {16, 1, 16};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_x_desc.GetShape().GetDims(), expected_x_shape);
}

TEST_F(AddRmsNormQuant, AddRmsNormQuant_InferDtype_case_2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType scale_ref = ge::DT_FLOAT;
        ge::DataType zero_point_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_HIFLOAT8;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(7)
                .NodeIoNum(7, 3)
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)},
                            {"div_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(34)}})
                .InputDataTypes(
                    {&input_ref, &input_ref, &input_ref, &scale_ref, &scale_ref, &zero_point_ref, &zero_point_ref})
                .OutputDataTypes({&output_ref, &output_ref, &input_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
        EXPECT_EQ(context->GetOutputDataType(2), input_ref);
    }
}

TEST_F(AddRmsNormQuant, AddRmsNormQuant_infershape_case_3)
{
    ge::op::AddRmsNormQuant op;
    op.UpdateInputDesc("x1", create_desc({4, 32, 1, 16}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x2", create_desc({4, 32, 1, 16}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({16}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scales1", create_desc({16}, ge::DT_FLOAT));
    op.UpdateInputDesc("scales2", create_desc({16}, ge::DT_FLOAT));
    op.UpdateInputDesc("zero_points1", create_desc({16}, ge::DT_INT32));
    op.UpdateInputDesc("zero_points2", create_desc({16}, ge::DT_INT32));
    //   op.SetAttr("epsilon", 0.1);
    //   Runtime2TestParam param{{"epsilon"},{},{}};
    //   EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_x_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {4, 32, 1, 16};
    std::vector<int64_t> expected_x_shape = {4, 32, 1, 16};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_x_desc.GetShape().GetDims(), expected_x_shape);
}

TEST_F(AddRmsNormQuant, AddRmsNormQuant_InferDtype_case_3)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType scale_ref = ge::DT_FLOAT;
        ge::DataType zero_point_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_FLOAT8_E5M2;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(7)
                .NodeIoNum(7, 3)
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)},
                            {"div_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}})
                .InputDataTypes(
                    {&input_ref, &input_ref, &input_ref, &scale_ref, &scale_ref, &zero_point_ref, &zero_point_ref})
                .OutputDataTypes({&output_ref, &output_ref, &input_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
        EXPECT_EQ(context->GetOutputDataType(2), input_ref);
    }
}

TEST_F(AddRmsNormQuant, AddRmsNormQuant_infershape_case_4)
{
    ge::op::AddRmsNormQuant op;
    op.UpdateInputDesc("x1", create_desc({4, 64, 16}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x2", create_desc({4, 64, 16}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({16}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scales1", create_desc({16}, ge::DT_FLOAT));
    op.UpdateInputDesc("scales2", create_desc({16}, ge::DT_FLOAT));
    op.UpdateInputDesc("zero_points1", create_desc({16}, ge::DT_INT32));
    op.UpdateInputDesc("zero_points2", create_desc({16}, ge::DT_INT32));
    //   op.SetAttr("epsilon", 0.1);
    //   Runtime2TestParam param{{"epsilon"},{},{}};
    //   EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_x_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {4, 64, 16};
    std::vector<int64_t> expected_x_shape = {4, 64, 16};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_x_desc.GetShape().GetDims(), expected_x_shape);
}

TEST_F(AddRmsNormQuant, AddRmsNormQuant_InferDtype_case_4)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType scale_ref = ge::DT_FLOAT;
        ge::DataType zero_point_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_FLOAT8_E4M3FN;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(7)
                .NodeIoNum(7, 3)
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                            {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)},
                            {"div_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(36)}})
                .InputDataTypes(
                    {&input_ref, &input_ref, &input_ref, &scale_ref, &scale_ref, &zero_point_ref, &zero_point_ref})
                .OutputDataTypes({&output_ref, &output_ref, &input_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
        EXPECT_EQ(context->GetOutputDataType(2), input_ref);
    }
}