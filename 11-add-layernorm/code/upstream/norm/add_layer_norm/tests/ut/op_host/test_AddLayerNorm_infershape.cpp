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
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"
#include "../../../op_graph/add_layer_norm_proto.h"

class AddLayerNorm : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddLayerNorm Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddLayerNorm Proto Test TearDown" << std::endl;
    }
};

TEST_F(AddLayerNorm, AddLayerNorm_infershape_case_0)
{
    ge::op::AddLayerNorm op;
    op.UpdateInputDesc("x1", create_desc({4, 1, 8}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x2", create_desc({4, 1, 8}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({8}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({8}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mean_desc = op.GetOutputDesc(1);
    auto output_rstd_desc = op.GetOutputDesc(2);
    auto output_x_desc = op.GetOutputDesc(3);
    std::vector<int64_t> expected_y_shape = {4, 1, 8};
    std::vector<int64_t> expected_x_shape = {4, 1, 8};
    std::vector<int64_t> expected_mean_shape = {4, 1, 1};
    std::vector<int64_t> expected_rstd_shape = {4, 1, 1};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_x_desc.GetShape().GetDims(), expected_x_shape);
    EXPECT_EQ(output_mean_desc.GetShape().GetDims(), expected_mean_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(AddLayerNorm, AddLayerNorm_InferDtype_mix_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNorm"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNorm")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType x1_ref = ge::DT_FLOAT16;
        ge::DataType x2_ref = ge::DT_FLOAT;
        ge::DataType gamma_ref = ge::DT_FLOAT;
        ge::DataType beta_ref = ge::DT_FLOAT;
        ge::DataType y_ref = ge::DT_FLOAT;
        ge::DataType mean_ref = ge::DT_FLOAT;
        ge::DataType rstd_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&x1_ref, &x2_ref, &gamma_ref, &beta_ref})
                                  .OutputDataTypes({&y_ref, &mean_ref, &rstd_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), mean_ref);
        EXPECT_EQ(context->GetOutputDataType(2), rstd_ref);
    }
}
