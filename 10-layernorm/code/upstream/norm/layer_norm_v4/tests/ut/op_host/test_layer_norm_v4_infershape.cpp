/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "array_ops.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "../../../op_graph/layer_norm_v4_proto.h"

class LayerNormV4Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LayerNormV4Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LayerNormV4Test TearDown" << std::endl;
    }
};

TEST_F(LayerNormV4Test, layer_norm_v4_infershape_test_0)
{
    using namespace ge;
    // input x info
    auto input_x_shape = vector<int64_t>({100, -1, -1, -1});
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{100, 100}, {100, 200}, {1, -1}, {1, -1}};
    auto input_x_dtype = DT_FLOAT;
    // input axes info
    auto normalized_shape_shape = vector<int64_t>({2});
    auto normalized_shape_dtype = DT_INT32;
    std::vector<std::pair<int64_t, int64_t>> normalized_shape_range = {{2, 2}};
    // expect result info
    std::vector<int64_t> expected_output_shape = input_x_shape;
    std::vector<int64_t> expected_output1_shape = vector<int64_t>({100, -1, 1, 1});

    // gen ReduceSum op
    auto test_op = op::LayerNormV4("LayerNormV4");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_ND, shape_range);
    TENSOR_INPUT_WITH_SHAPE(
        test_op, normalized_shape, normalized_shape_shape, normalized_shape_dtype, FORMAT_ND, normalized_shape_range);

    Runtime2TestParam param;
    vector<bool> input_const = {false, false};
    param.input_const = input_const;
    EXPECT_EQ(InferShapeTest(test_op, param), ge::GRAPH_SUCCESS);
    auto output0_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
    auto output1_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_output1_shape);
    auto output2_desc = test_op.GetOutputDesc(2);
    EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_output1_shape);
}

TEST_F(LayerNormV4Test, layer_norm_v4_infershape_test_1)
{
    using namespace ge;
    // input x info
    auto input_x_shape = vector<int64_t>({100, -1, -1, -1});
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{100, 100}, {100, 200}, {1, -1}, {1, -1}};
    auto input_x_dtype = DT_FLOAT;
    // input axes info
    auto normalized_shape_shape = vector<int64_t>({-1});
    auto normalized_shape_dtype = DT_INT32;
    std::vector<std::pair<int64_t, int64_t>> normalized_shape_range = {{2, 3}};
    // expect result info
    std::vector<int64_t> expected_output_shape = input_x_shape;
    std::vector<int64_t> expected_output1_shape = vector<int64_t>({-1, -1, -1, -1});

    // gen ReduceSum op
    auto test_op = op::LayerNormV4("LayerNormV4");
    TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, FORMAT_ND, shape_range);
    TENSOR_INPUT_WITH_SHAPE(
        test_op, normalized_shape, normalized_shape_shape, normalized_shape_dtype, FORMAT_ND, normalized_shape_range);

    Runtime2TestParam param;
    vector<bool> input_const = {false, false};
    param.input_const = input_const;
    EXPECT_EQ(InferShapeTest(test_op, param), ge::GRAPH_SUCCESS);
    auto output0_desc = test_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
    auto output1_desc = test_op.GetOutputDesc(1);
    EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_output1_shape);
    auto output2_desc = test_op.GetOutputDesc(2);
    EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_output1_shape);
}

TEST_F(LayerNormV4Test, LayerNormV4InferDataType)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormV4"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormV4")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_x = ge::DT_FLOAT;
    ge::DataType shape_dtype = ge::DT_INT32;
    ge::DataType out_datatype = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(2)
                              .NodeIoNum(2, 3)
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_x, &shape_dtype})
                              .OutputDataTypes({&out_datatype, &out_datatype, &out_datatype})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetInputDataType(0), input_x);
    EXPECT_EQ(context->GetInputDataType(1), shape_dtype);
    EXPECT_EQ(context->GetOutputDataType(0), input_x);
}

TEST_F(LayerNormV4Test, layer_norm_v4_infershape_range_test0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormV4"), nullptr);
    auto infer_shape_range_func = gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormV4")->infer_shape_range;

    gert::Shape input_x_range_min{2, 2, 3, 8};
    gert::Shape input_x_range_max{2, -1, 3, 8};
    gert::Shape input_norm_shape_range_min{2};
    gert::Shape input_norm_shape_range_max{3};
    gert::Shape null1{};
    gert::Shape null2{};
    gert::Shape null3{};
    gert::Shape null4{};
    gert::Shape null5{};
    gert::Shape null6{};

    gert::Range<gert::Shape> input_x_range(&input_x_range_min, &input_x_range_max);
    gert::Range<gert::Shape> input_norm_shape_range(&input_norm_shape_range_min, &input_norm_shape_range_max);
    gert::Range<gert::Shape> output_0_shape_range(&null1, &null2);
    gert::Range<gert::Shape> output_1_shape_range(&null3, &null4);
    gert::Range<gert::Shape> output_2_shape_range(&null5, &null6);

    gert::Range<gert::Shape> expect_output_0_shape_range(&input_x_range_min, &input_x_range_max);

    gert::Shape output_1_range_min{2, 0, 1, 1};
    gert::Shape output_1_range_max{2, -1, 1, 1};
    gert::Range<gert::Shape> expect_output_1_shape_range(&output_1_range_min, &output_1_range_max);

    auto context_holder = gert::InferShapeRangeContextFaker()
                              .IrInputNum(2)
                              .NodeIoNum(2, 3)
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputShapeRanges({&input_x_range, &input_norm_shape_range})
                              .OutputShapeRanges({&output_0_shape_range, &output_1_shape_range, &output_2_shape_range})
                              .Build();

    auto context = context_holder.GetContext<gert::InferShapeRangeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_shape_range_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShapeRange(0)->GetMax()), Ops::Base::ToString(input_x_range_max));
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShapeRange(0)->GetMin()), Ops::Base::ToString(input_x_range_min));

    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShapeRange(1)->GetMax()), Ops::Base::ToString(output_1_range_max));
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShapeRange(1)->GetMin()), Ops::Base::ToString(output_1_range_min));
}
