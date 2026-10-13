/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <memory>
#include <vector>
#include <array>
#include <cstdint>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_sparse_flash_attention.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
using namespace std;
using namespace op;

namespace {
void DestroyAclTensor(aclTensor *tensor)
{
    Release(tensor);
}
} // namespace

class sparse_flash_attention_opapi_ut : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "sparse_flash_attention_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "sparse_flash_attention_opapi_ut TearDown" << endl;
    }
};

// Missing lse outputs
TEST_F(sparse_flash_attention_opapi_ut, sparse_flash_attention_aclnn_0)
{
    char layoutQ[] = "BSND";
    char layoutKv[] = "BSND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto key = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 128, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto value = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 128, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseIndices = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 1, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto attentionOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnSparseFlashAttentionGetWorkspaceSize(
        query.get(),
        key.get(),
        value.get(),
        sparseIndices.get(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        0.0416666666666667,
        64,
        layoutQ,
        layoutKv,
        3,
        INT64_MAX,
        INT64_MAX,
        0,
        true,
        attentionOut.get(),
        nullptr,
        nullptr,
        &workspaceSize,
        &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// Holder fallback
TEST_F(sparse_flash_attention_opapi_ut, sparse_flash_attention_aclnn_1)
{
    char layoutQ[] = "BSND";
    char layoutKv[] = "BSND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto value = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 128, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseIndices = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 1, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto attentionOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({2, 1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnSparseFlashAttentionGetWorkspaceSize(
        query.get(),
        nullptr,
        value.get(),
        sparseIndices.get(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        0.0416666666666667,
        64,
        layoutQ,
        layoutKv,
        3,
        INT64_MAX,
        INT64_MAX,
        0,
        false,
        attentionOut.get(),
        nullptr,
        nullptr,
        &workspaceSize,
        &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}
