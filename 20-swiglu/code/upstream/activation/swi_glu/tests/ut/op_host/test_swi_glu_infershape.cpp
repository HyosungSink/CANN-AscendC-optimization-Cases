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
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "../../../op_graph/swi_glu_proto.h"

class SwiGluTest : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "SwiGluTest Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "SwiGluTest Proto Test TearDown" << std::endl;
    }
};

TEST_F(SwiGluTest, SwiGlu_infershape_diff_test_legal_input) {
    ge::op::SwiGlu op;
    op.UpdateInputDesc("x", create_desc({4, 4}, ge::DT_FLOAT16));
    op.SetAttr("dim", -1);
    Runtime2TestParam param{{"dim"},{},{}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    std::vector<int64_t> expected_output_shape = {4, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(SwiGluTest, SwiGlu_infershape_diff_test_dynamic_input) {
  ge::op::SwiGlu op;
  op.UpdateInputDesc("x", create_desc({-1, -1}, ge::DT_FLOAT16));
  op.SetAttr("dim", -1);
  Runtime2TestParam param{{"dim"},{},{}};
  EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

  auto output_y_desc = op.GetOutputDesc(0);
  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(SwiGluTest, SwiGlu_infershape_diff_test_illegal_input) {
  ge::op::SwiGlu op;
  op.UpdateInputDesc("x", create_desc({1, 3}, ge::DT_FLOAT16));
  op.SetAttr("dim", -1);
  Runtime2TestParam param{{"dim"},{},{}};
  EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(SwiGluTest, SwiGlu_infer_dtype_test) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SwiGlu"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SwiGlu")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_ref = ge::DT_FLOAT;
    ge::DataType output_ref = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
        .IrInputNum(1)
        .NodeIoNum(1,1)
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputDataTypes({&input_ref})
        .OutputDataTypes({&output_ref})
        .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetInputDataType(0), input_ref);
    EXPECT_EQ(context->GetOutputDataType(0), output_ref);
  }
}

TEST_F(SwiGluTest, SwiGlu_infershape_diff_test_dynamic_input_negative_two) {
    ge::op::SwiGlu op;
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_FLOAT16));
    op.SetAttr("dim", -1);
    Runtime2TestParam param{{"dim"},{},{}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    std::vector<int64_t> expected_output_shape = {-2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
}
