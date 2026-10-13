/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_MATMUL_V2TOV3_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_MATMUL_V2TOV3_H_

#include "opdev/common_types.h"

// 兼容opp整包、静态库和子包场景，向算子业务侧代码屏蔽差异：
// 子包场景：通过适配层dlopen方式找到libophost_comm_legacy.so里的C接口实现，向本仓算子侧提供Ops::NN namepsace的接口调用
// 整包和静态库场景：只做l0op namespace下的接口声明，向本仓算子侧提供Ops::NN namepsace的接口调用
#ifdef NN_ENABLE_DLOPEN_LEGACY
namespace Ops {
namespace NN {
// 包间接口的适配，做透传，参数封装的价值不大
bool MmCheckHitV3Shape(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const bool transposeX1, const bool transposeX2,
    op::Format mat2_format, bool supportSplitK);

bool BmmCheckHitV3Shape(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const bool adjX1, const bool adjX2,
    op::Format self_format, op::Format mat2_format, const bool enableFp16Bf16InFp32Out);
} // namespace NN
} // namespace Ops
#else
namespace l0op {
bool MmCheckHitV3Shape(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const bool transposeX1, const bool transposeX2,
    op::Format mat2_format, bool supportSplitK);
bool BmmCheckHitV3Shape(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const bool adjX1, const bool adjX2,
    op::Format self_format, op::Format mat2_format, const bool enableFp16Bf16InFp32Out);
} // namespace l0op

namespace Ops {
namespace NN {
using l0op::BmmCheckHitV3Shape;
using l0op::MmCheckHitV3Shape;
} // namespace NN
} // namespace Ops
#endif
#endif  // PTA_NPU_OP_API_INC_LEVEL0_OP_MATMUL_V2TOV3_H_