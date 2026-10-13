/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef MATH_TOPK_OP_HOST_OP_API_TOPK_H_
#define MATH_TOPK_OP_HOST_OP_API_TOPK_H_

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
std::tuple<aclTensor*, aclTensor*> Topk(
    const aclTensor* self, int64_t k, int64_t dim, bool largest, bool sorted, op::DataType indicesDType,
    aclOpExecutor* executor);
}

#endif // MATH_TOPK_OP_HOST_OP_API_TOPK_H_
