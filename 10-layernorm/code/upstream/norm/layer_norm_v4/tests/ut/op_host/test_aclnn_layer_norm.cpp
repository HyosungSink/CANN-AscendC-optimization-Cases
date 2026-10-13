/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_layer_norm.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_layer_norm_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "layer_norm_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "layer_norm_test TearDown" << endl;
    }
};

// float32
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_float32_nd_float32_nd)
{
    auto inputDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{6, 5, 6});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float64: kernel not support now
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_double_nd_double_nd)
{
    auto inputDesc = TensorDesc({2, 6, 5, 6}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({6, 5, 6}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({6, 5, 6}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{6, 5, 6});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({2, 6, 5, 6}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 1, 1, 1}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({2, 1, 1, 1}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error input dtype
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_error_input_dtype)
{
    auto inputDesc = TensorDesc({2, 6, 5, 6}, ACL_UINT64, ACL_FORMAT_NCHW).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({6, 5, 6}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({6, 5, 6}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{6, 5, 6});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({2, 6, 5, 6}, ACL_UINT64, ACL_FORMAT_NCHW).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 1, 1, 1}, ACL_UINT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({2, 1, 1, 1}, ACL_UINT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error weight dtype
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_error_weight_dtype)
{
    auto inputDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({6, 5, 6}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({6, 5, 6}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{6, 5, 6});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error bias dtype
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_error_bias_dtype)
{
    auto inputDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({6, 5, 6}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{6, 5, 6});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error weight and bias dtype
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_weight_bias_diff_dtype)
{
    auto inputDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({6, 5, 6}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{6, 5, 6});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error out dtype
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_error_out_dtype)
{
    auto inputDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{6, 5, 6});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({2, 6, 5, 6}, ACL_UINT64, ACL_FORMAT_NCHW).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 1, 1, 1}, ACL_UINT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({2, 1, 1, 1}, ACL_UINT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// hwcn, float32, all reduce
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_float32_hwcn_float32_hwcn_all_reduce)
{
    auto inputDesc = TensorDesc({3, 2, 1, 16}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(2, 10);
    auto weightDesc = TensorDesc({3, 2, 1, 16}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(0, 1);
    auto biasDesc = TensorDesc({3, 2, 1, 16}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(1, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{3, 2, 1, 16});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({3, 2, 1, 16}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// ndhwc, float32
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_float32_ndhwc_float32_ndhwc)
{
    auto inputDesc = TensorDesc({2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{3, 4, 5, 6});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_NDHWC).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({2, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// ncdhw, float32
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_float32_ncdhw_float32_ncdhw)
{
    auto inputDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{2, 2});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 2, 2, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({2, 2, 2, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// M = 0
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_0_2_7_18_float_nd_7_18_float_nd)
{
    auto inputDesc = TensorDesc({0, 2, 7, 18}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({7, 18}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({7, 18}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{7, 18});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({0, 2, 7, 18}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({0, 2, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({0, 2, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// N = 0
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_2_0_4_float_nd_0_4_float_nd)
{
    auto inputDesc = TensorDesc({2, 0, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{0, 4});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({2, 0, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({2, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// error input len
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_error_input_len)
{
    auto inputDesc = TensorDesc({1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({2, 1, 2, 1, 2, 1, 2, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({2, 1, 2, 1, 2, 1, 2, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{2, 1, 2, 1, 2, 1, 2, 1, 2, 2});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error normalizedShape len
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_error_normalizedShape_len)
{
    auto inputDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error weight, bias and normalizedShape
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_diff_weight_and_normalizedShape)
{
    auto inputDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{5, 6});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error input and normalizedShape
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_error_input_and_normalizedShape_shape)
{
    auto inputDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{7, 8});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 6, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({2, 6, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error weight and bias shape
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_error_weight_and_bias_shape)
{
    auto inputDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{2});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input nullptr
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_input_nullptr)
{
    auto weightDesc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{8, 9});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({7, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({7, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT((aclTensor*)nullptr, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// nomalizedShape nullptr
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_normalizedShape_nullptr)
{
    auto inputDesc = TensorDesc({7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    double eps = 1e-12;
    auto outputDesc = TensorDesc({7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({7, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({7, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, (aclIntArray*)nullptr, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out nullptr
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_out_nullptr)
{
    auto inputDesc = TensorDesc({7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{8, 9});
    double eps = 1e-12;
    auto meanDesc = TensorDesc({7, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({7, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    uint64_t workspaceSize = 0;
    // out nullptr
    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT((aclTensor*)nullptr, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// meanOut nullptr
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_meanOout_nullptr)
{
    auto inputDesc = TensorDesc({7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{8, 9});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({7, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    uint64_t workspaceSize = 0;
    // meanOut nullptr
    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, (aclTensor*)nullptr, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// rstdOut nullptr
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_rstdOut_nullptr)
{
    auto inputDesc = TensorDesc({7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{8, 9});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({7, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, (aclTensor*)nullptr));
    // SAMPLE: only test GetWorkspaceSize
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float16+float32+float32
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_float16_mix_dtype)
{
    auto inputDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{6, 5, 6});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// bfloat16+float32+float32
TEST_F(l2_layer_norm_test, ascend910B2_aclnnLayerNorm_bfloat16_mix_dtype)
{
    auto inputDesc = TensorDesc({2, 6, 5, 6}, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{6, 5, 6});
    double eps = 1e-12;
    auto outputDesc = TensorDesc({2, 6, 5, 6}, ACL_BF16, ACL_FORMAT_NCHW).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto rstdDesc = TensorDesc({2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNorm, INPUT(inputDesc, normalizedShape, weightDesc, biasDesc, eps),
        OUTPUT(outputDesc, meanDesc, rstdDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
