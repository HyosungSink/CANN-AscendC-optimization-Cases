/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_MAX_POOL3D_WITH_ARGMAX_V2_H_
#define OP_API_INC_LEVEL0_MAX_POOL3D_WITH_ARGMAX_V2_H_

#include "opdev/op_executor.h"

namespace l0op {
const std::tuple<const aclTensor*, const aclTensor*> MaxPool3DWithArgmaxV2Ncdhw(const aclTensor* self, const aclIntArray* kernelSize,
                                                                                const aclIntArray* stride, const aclIntArray* padding,
                                                                                const aclIntArray* dilation, bool ceilMode,
                                                                                std::string dataFormat, aclOpExecutor* executor,
                                                                                op::DataType indicesDtype = op::DataType::DT_INT32);
}  // namespace l0op

#endif  // OP_API_INC_LEVEL0_MAX_POOL3D_WITH_ARGMAX_V2_H_