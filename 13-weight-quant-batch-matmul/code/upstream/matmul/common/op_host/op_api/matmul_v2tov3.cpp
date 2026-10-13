/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 兼容opp整包、静态库和子包场景，向算子业务侧代码屏蔽差异：
// 子包场景：通过适配层dlopen方式找到libophost_comm_legacy.so里的C接口实现，向本仓算子侧提供Ops::NN namepsace的接口调用
// 整包和静态库场景：只做l0op namespace下的接口声明，向本仓算子侧提供Ops::NN namepsace的接口调用
#ifdef NN_ENABLE_DLOPEN_LEGACY
#include "matmul/common/op_host/op_api/matmul_v2tov3.h"
#include "opdev/platform.h"
#include "legacy_common_manager.h"

namespace Ops {
namespace NN {
static void ConvertTensor(const aclTensor*src, gert::Tensor&dst)
{
    if (src != nullptr) {
        dst.MutableOriginShape() = src->GetViewShape();
        dst.MutableStorageShape() = src->GetStorageShape();
        dst.SetOriginFormat(src->GetViewFormat());
        dst.SetStorageFormat(src->GetStorageFormat());
        dst.SetDataType(src->GetDataType());
    }
}

// 包间接口的适配，做透传，参数封装的价值不大
bool MmCheckHitV3Shape(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const bool transposeX1, const bool transposeX2,
    op::Format mat2_format, bool supportSplitK)
{
    using FuncType = bool (*)(
        const gert::Tensor*, const gert::Tensor*, const gert::Tensor*, const bool, const bool, op::Format, bool,
        uint32_t, const std::string&);
    const char* symbolName = "LegacyMmCheckHitV3Shape";
    static FuncType func = Ops::NN::LegacyCommonMgr::GetInstance().GetFunc<FuncType>(symbolName);
    if (func == nullptr) {
        OP_LOGW("dest func %s pointer is null.", symbolName);
        return false;
    } else {
        gert::Tensor x1Tensor;
        gert::Tensor x2Tensor;
        gert::Tensor biasTensor;
        ConvertTensor(x1, x1Tensor);
        ConvertTensor(x2, x2Tensor);
        ConvertTensor(bias, biasTensor);
        gert::Tensor *biasPtr = bias == nullptr ? nullptr: &biasTensor;
        return func(
            &x1Tensor, &x2Tensor, biasPtr, transposeX1, transposeX2, mat2_format, supportSplitK,
            op::GetCurrentPlatformInfo().GetCubeCoreNum(), op::GetCurrentPlatformInfo().GetSocLongVersion());
    }
}

bool BmmCheckHitV3Shape(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const bool adjX1, const bool adjX2,
    op::Format self_format, op::Format mat2_format, const bool enableFp16Bf16InFp32Out)
{
    using FuncType = bool (*)(
        const gert::Tensor*, const gert::Tensor*, const gert::Tensor*, const bool, const bool, op::Format, op::Format,
        const bool, uint32_t, const std::string&, op::SocVersion);
    const char* symbolName = "LegacyBmmCheckHitV3Shape";
    static FuncType func = Ops::NN::LegacyCommonMgr::GetInstance().GetFunc<FuncType>(symbolName);
    if (func == nullptr) {
        OP_LOGW("dest func %s pointer is null.", symbolName);
        return false;
    } else {
        gert::Tensor x1Tensor;
        gert::Tensor x2Tensor;
        gert::Tensor biasTensor;
        ConvertTensor(x1, x1Tensor);
        ConvertTensor(x2, x2Tensor);
        ConvertTensor(bias, biasTensor);
        gert::Tensor* biasPtr = bias == nullptr ? nullptr : &biasTensor;
        return func(
            &x1Tensor, &x2Tensor, biasPtr, adjX1, adjX2, self_format, mat2_format, enableFp16Bf16InFp32Out,
            op::GetCurrentPlatformInfo().GetCubeCoreNum(), op::GetCurrentPlatformInfo().GetSocLongVersion(),
            op::GetCurrentPlatformInfo().GetSocVersion());
    }
}
} // namespace NN
} // namespace Ops
#endif
