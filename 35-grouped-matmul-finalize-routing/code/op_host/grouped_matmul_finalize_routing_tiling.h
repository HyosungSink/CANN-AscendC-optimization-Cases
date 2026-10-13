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
 * \file grouped_matmul_finalize_routing_tiling.h
 * \brief
 */
#ifndef __OP_HOST_GROUPED_MATMUL_FINALIZE_ROUTING_TILING_H__
#define __OP_HOST_GROUPED_MATMUL_FINALIZE_ROUTING_TILING_H__
#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "../common/include/err/ops_err.h"

namespace optiling {

struct GroupedMatmulFinalizeRoutingCompileInfo {
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
    uint64_t l1Size{0UL};
    uint64_t l2Size{0UL};
    uint64_t l0CSize{0UL};
    uint64_t l0ASize{0UL};
    uint64_t l0BSize{0UL};
    uint64_t btSize{0UL};
    float cubeFreq{0};
    platform_ascendc::SocVersion socVersion;
    bool supportL0c2out = false;
    bool supportL12BtBf16 = false;
    NpuArch npuArch;
};

BEGIN_TILING_DATA_DEF(GroupMatmulFRTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);
  TILING_DATA_FIELD_DEF(uint32_t, groupNum);
  TILING_DATA_FIELD_DEF(uint32_t, totalInGroup);
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, k);
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, vBaseM);
  TILING_DATA_FIELD_DEF(uint32_t, ubCalSize);
  TILING_DATA_FIELD_DEF(uint32_t, ubRestBytes);
  TILING_DATA_FIELD_DEF(uint32_t, parallNum);
  TILING_DATA_FIELD_DEF(uint32_t, sharedInputOffset);
  TILING_DATA_FIELD_DEF(uint32_t, sharedInputLen);
  TILING_DATA_FIELD_DEF(float, residualScale);
  TILING_DATA_FIELD_DEF(uint32_t, quantGroupNum);
  TILING_DATA_FIELD_DEF(uint32_t, withOffset);
  TILING_DATA_FIELD_DEF(uint32_t, hasPertokenScale);
  TILING_DATA_FIELD_DEF(uint32_t, hasBias);
  TILING_DATA_FIELD_DEF(uint32_t, deterministicFlag);
  TILING_DATA_FIELD_DEF(uint32_t, deterWorkspaceSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupedMatmulFinalizeRouting, GroupMatmulFRTilingData)
REGISTER_TILING_DATA_CLASS(GroupMatmulFRTilingDataOp, GroupMatmulFRTilingData)
}
#endif // __OP_HOST_GROUPED_MATMUL_FINALIZE_ROUTING_TILING_H__