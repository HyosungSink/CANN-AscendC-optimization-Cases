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
#include <cstdint>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_lightning_indexer.h"
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

class lightning_indexer_opapi_ut : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "lightning_indexer_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "lightning_indexer_opapi_ut TearDown" << endl;
    }
};

// Null query
TEST_F(lightning_indexer_opapi_ut, lightning_indexer_aclnn_0)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerGetWorkspaceSize(
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        layoutQuery,
        layoutKey,
        4,
        3,
        INT64_MAX,
        INT64_MAX,
        true,
        nullptr,
        nullptr,
        &workspaceSize,
        &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// Missing sparse values
TEST_F(lightning_indexer_opapi_ut, lightning_indexer_aclnn_1)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "BSND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto key = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 64, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weights = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseIndicesOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerGetWorkspaceSize(
        query.get(),
        key.get(),
        weights.get(),
        nullptr,
        nullptr,
        nullptr,
        layoutQuery,
        layoutKey,
        4,
        3,
        INT64_MAX,
        INT64_MAX,
        true,
        sparseIndicesOut.get(),
        nullptr,
        &workspaceSize,
        &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// Holder fallback
TEST_F(lightning_indexer_opapi_ut, lightning_indexer_aclnn_2)
{
    char layoutQuery[] = "BSND";
    char layoutKey[] = "BSND";
    auto query = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto weights = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    auto sparseIndicesOut = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>(
        TensorDesc({1, 8, 1, 4}, ACL_INT32, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnLightningIndexerGetWorkspaceSize(
        query.get(),
        nullptr,
        weights.get(),
        nullptr,
        nullptr,
        nullptr,
        layoutQuery,
        layoutKey,
        4,
        3,
        INT64_MAX,
        INT64_MAX,
        false,
        sparseIndicesOut.get(),
        nullptr,
        &workspaceSize,
        &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}
