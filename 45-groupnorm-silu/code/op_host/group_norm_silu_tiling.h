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
 * \file group_norm_silu_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GROUP_NORM_SILU_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GROUP_NORM_SILU_TILING_H_

#include <cstdint>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../common/inc/op_host/tiling_base.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "../common/inc/op_host/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "../common/inc/op_host/tiling_templates_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GroupNormSiluTilingData)
TILING_DATA_FIELD_DEF(int64_t, numGroups);
TILING_DATA_FIELD_DEF(int64_t, hwNum);
TILING_DATA_FIELD_DEF(int64_t, elemNum);
TILING_DATA_FIELD_DEF(int64_t, shapeC);
TILING_DATA_FIELD_DEF(int64_t, shapeD);
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, numPerCore);
TILING_DATA_FIELD_DEF(int64_t, numLastCore);
TILING_DATA_FIELD_DEF(int64_t, processSize);
TILING_DATA_FIELD_DEF(int64_t, loopNum);
TILING_DATA_FIELD_DEF(int64_t, loopTail);
TILING_DATA_FIELD_DEF(int64_t, innerLoopNum);
TILING_DATA_FIELD_DEF(int64_t, innerLoopTail);
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(int64_t, activateSilu);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupNormSilu, GroupNormSiluTilingData)

BEGIN_TILING_DATA_DEF(GroupNormSiluRegbaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, numGroups);
TILING_DATA_FIELD_DEF(int64_t, hwNum);
TILING_DATA_FIELD_DEF(int64_t, elemNum);
TILING_DATA_FIELD_DEF(int64_t, shapeN);
TILING_DATA_FIELD_DEF(int64_t, shapeC);
TILING_DATA_FIELD_DEF(int64_t, shapeD);
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, numPerCore);
TILING_DATA_FIELD_DEF(int64_t, numLastCore);
TILING_DATA_FIELD_DEF(int64_t, processSize);
TILING_DATA_FIELD_DEF(int64_t, loopNum);
TILING_DATA_FIELD_DEF(int64_t, loopTail);
TILING_DATA_FIELD_DEF(int64_t, innerLoopNum);
TILING_DATA_FIELD_DEF(int64_t, innerLoopTail);
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(int64_t, activateSilu);
TILING_DATA_FIELD_DEF(int64_t, parallelN);
TILING_DATA_FIELD_DEF(int64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, dichotomyAddPower);
TILING_DATA_FIELD_DEF(int64_t, dichotomyAddK);
TILING_DATA_FIELD_DEF(int64_t, dichotomyAddLastNum);
TILING_DATA_FIELD_DEF(int64_t, powerOfTwoForReduce);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupNormSilu_1000, GroupNormSiluRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormSilu_1001, GroupNormSiluRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormSilu_1100, GroupNormSiluRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormSilu_1101, GroupNormSiluRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormSilu_1110, GroupNormSiluRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormSilu_1111, GroupNormSiluRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormSilu_1120, GroupNormSiluRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormSilu_1121, GroupNormSiluRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormSilu_1130, GroupNormSiluRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(GroupNormSilu_1131, GroupNormSiluRegbaseTilingData)

struct GroupNormSiluCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    int32_t is310P = 0;
    int32_t isRegbase = 0;
    uint32_t vectorLength = 0;
    uint32_t blockSizePlatform = 0;
};

enum class GroupNormSiluTilingKey : int64_t
{
    TILINGKEY_SMALL_SHAPE_B16 = 1011,       // small shape and dtype is b16(float16/bfloat16)
    TILINGKEY_SMALL_SHAPE_MIXTYPE = 1012,   // small shape and mixed dtype
    TILINGKEY_SMALL_SHAPE_B32 = 102,        // small shape and dtype is b32(float32)
    TILINGKEY_HIGH_PERF_B16 = 1031,         // high performance and dtype is b16
    TILINGKEY_HIGH_PERF_MIXTYPE = 1032,     // high performance and mixed dtype
    TILINGKEY_HIGH_PERF_B32 = 104,          // high performance and dtype is b32
    TILINGKEY_BASIC_TEM_B16 = 1051,         // basic template and dtype is b16
    TILINGKEY_BASIC_TEM_MIXTYPE = 1052,     // basic template and mixed dtype
    TILINGKEY_BASIC_TEM_B32 = 106,          // basic template and dtype is b32
    TILINGKEY_HW_ONE_B16 = 1071,            // H*W is 1 and dtype is b16
    TILINGKEY_HW_ONE_MIXTYPE = 1072,        // H*W is 1 and mixed dtype
    TILINGKEY_HW_ONE_B32 = 108,             // H*W is 1 and dtype is b32
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_GROUP_NORM_SILU_TILING_H_
