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
 * \file aclnn_util.h
 * \brief
 */
#ifndef COMMON_NN_ACLNN_UTIL_H
#define COMMON_NN_ACLNN_UTIL_H

#define ACLNN_API __attribute__((visibility("default")))

#include "opdev/platform.h"
#include <set>

namespace Ops {
namespace NN {
namespace AclnnUtil {

using namespace op;

/**
 * 检查当前芯片架构是否为RegBase
 */
inline static bool IsRegbase()
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    const static std::set<NpuArch> regbaseNpuArchs = {
        NpuArch::DAV_3510};
    return regbaseNpuArchs.find(npuArch) != regbaseNpuArchs.end();
}

inline static bool IsRegbase(NpuArch npuArch)
{
    const static std::set<NpuArch> regbaseNpuArchs = {
        NpuArch::DAV_3510};
    return regbaseNpuArchs.find(npuArch) != regbaseNpuArchs.end();
}

} // namespace OpTiling
} // namespace NN
} // namespace AclnnUtil

#endif  // COMMON_NN_ACLNN_UTIL_H