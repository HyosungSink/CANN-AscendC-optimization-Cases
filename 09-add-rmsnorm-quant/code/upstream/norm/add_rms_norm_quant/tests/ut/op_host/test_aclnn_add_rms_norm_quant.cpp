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

#include "../../../op_host/op_api/aclnn_add_rms_norm_quant.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_add_rms_norm_quant_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "add_rms_norm_quant_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "add_rms_norm_quant_test TearDown" << endl;
    }
};

TEST_F(l2_add_rms_norm_quant_test, ascend950PR_9589_case_dyn_001)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t axis = -1L;
    double eps = 1e-5;
    bool divMode = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2, (aclTensor*)nullptr,
            (aclTensor*)nullptr, axis, eps, divMode),
        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_quant_test, ascend950PR_9589_case_stc_001)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_z1 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_z2 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t axis = -1;
    double eps = 1e-5;
    bool divMode = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2, tensor_desc_z1,
            tensor_desc_z2, axis, eps, divMode),
        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

//**********************************************************************************************************************
TEST_F(l2_add_rms_norm_quant_test, ascend950PR_9589_case_dyn_002)
{
    auto tensor_desc_x1 = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            16,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc(
        {
            16,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc(
        {
            16,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_z1 = TensorDesc(
        {
            16,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_z2 = TensorDesc(
        {
            16,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 16}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 16}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t axis = -1;
    double eps = 1e-5;
    bool divMode = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2, tensor_desc_z1,
            tensor_desc_z2, axis, eps, divMode),
        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_quant_test, ascend950PR_9589_case_stc_002)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_z1 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_z2 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t axis = -1;
    double eps = 1e-5;
    bool divMode = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2, tensor_desc_z1,
            tensor_desc_z2, axis, eps, divMode),
        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_quant_test, ascend950PR_9589_case_dyn_003)
{
    auto tensor_desc_x1 = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            16,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc(
        {
            16,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc(
        {
            16,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_z1 = TensorDesc(
        {
            16,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_z2 = TensorDesc(
        {
            16,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t axis = -1;
    double eps = 1e-5;
    bool divMode = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2, tensor_desc_z1,
            tensor_desc_z2, axis, eps, divMode),
        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_quant_test, ascend950PR_9589_case_stc_003)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_z1 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_z2 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t axis = -1;
    double eps = 1e-5;
    bool divMode = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2, tensor_desc_z1,
            tensor_desc_z2, axis, eps, divMode),
        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_quant_test, ascend950PR_9589_case_dyn_004)
{
    auto tensor_desc_x1 = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            16,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc(
        {
            16,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc(
        {
            16,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_z1 = TensorDesc(
        {
            16,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_z2 = TensorDesc(
        {
            16,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 16}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 16}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t axis = -1;
    double eps = 1e-5;
    bool divMode = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2, tensor_desc_z1,
            tensor_desc_z2, axis, eps, divMode),
        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_quant_test, ascend950PR_9589_case_stc_004)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_z1 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_z2 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t axis = -1;
    double eps = 1e-5;
    bool divMode = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2, tensor_desc_z1,
            tensor_desc_z2, axis, eps, divMode),
        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
