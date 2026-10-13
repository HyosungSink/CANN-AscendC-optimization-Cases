/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL0_OP_NSA_COMPRESS_OP_H
#define OP_API_INC_LEVEL0_OP_NSA_COMPRESS_OP_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *NsaCompress(const aclTensor *input, const aclTensor *weight, const aclIntArray *actSeqLenOptional,
                             char *layoutOptional, int64_t compressBlockSize, int64_t compressStride,
                             int64_t actSeqLenType, aclOpExecutor *executor);
} // namespace l0op

#endif