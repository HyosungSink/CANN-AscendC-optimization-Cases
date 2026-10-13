/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_rms_norm_tiling.h
 * \brief test RmsNorm Op Tiling
 */
#ifndef OPS_BUILT_IN_OP_TEST_TILING_RUNTIME_RMS_NORM_H_
#define OPS_BUILT_IN_OP_TEST_TILING_RUNTIME_RMS_NORM_H_

#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

namespace optiling {

struct Tiling4RmsNormCompileInfo {
    uint32_t totalCoreNum = 0;
    uint64_t totalUbSize = 0;
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TEST_TILING_RUNTIME_RMS_NORM_H_
