/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_var.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL0_REDUCE_VAR_H
#define OP_API_INC_LEVEL0_REDUCE_VAR_H
#include "opdev/op_executor.h"

namespace l0op {

const std::tuple<const aclTensor *, const aclTensor *> ReduceVar(const aclTensor* self, const aclIntArray* dim,
    int64_t correction, bool keepdim, bool isMeanOut, aclOpExecutor* executor);
}

#endif