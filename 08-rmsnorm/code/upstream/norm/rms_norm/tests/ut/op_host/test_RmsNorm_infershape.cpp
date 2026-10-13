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
#include "ut_op_common.h"
#include "../../../op_graph/rms_norm_proto.h"

class RmsNorm : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "RmsNorm Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RmsNorm Proto Test TearDown" << std::endl;
    }
};

TEST_F(RmsNorm, RmsNorm_infershape_case_0)
{
    ge::op::RmsNorm op;
    op.UpdateInputDesc("x", create_desc({4, 1, 8}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({8}, ge::DT_FLOAT16));
    //   op.SetAttr("epsilon", 0.1);
    //   Runtime2TestParam param{{"epsilon"},{},{}};
    //   EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_rstd_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {4, 1, 8};
    std::vector<int64_t> expected_rstd_shape = {4, 1, 1};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(RmsNorm, RmsNorm_infershape_case_1)
{
    ge::op::RmsNorm op;
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({8}, ge::DT_FLOAT16));
    //   op.SetAttr("epsilon", 0.1);
    //   Runtime2TestParam param{{"epsilon"},{},{}};
    //   EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_rstd_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {-2};
    std::vector<int64_t> expected_rstd_shape = {-2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(RmsNorm, RmsNorm_InferDtype_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNorm"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNorm")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType output_ref = ge::DT_FLOAT16;
        ge::DataType rstd_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref, &rstd_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), rstd_ref);
    }
}