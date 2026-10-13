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
 * \file moe_init_routing_quant_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INIT_ROUTING_QUANT_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INIT_ROUTING_QUANT_H
#include <cmath>
#include <string>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "../common/include/tiling_base/tiling_base.h"
#include "../common/include/tiling_base/tiling_templates_registry.h"
#include "util/math_util.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(QuantVBSComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, perCoreElements);
TILING_DATA_FIELD_DEF(int64_t, perCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopElements);
TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopElements);
TILING_DATA_FIELD_DEF(int64_t, oneLoopMaxElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(QuantVBSComputeTilingDataOp, QuantVBSComputeTilingData)

BEGIN_TILING_DATA_DEF(QuantVMSMiddleComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(QuantVMSMiddleComputeTilingDataOp, QuantVMSMiddleComputeTilingData)

BEGIN_TILING_DATA_DEF(QuantSortOutComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, oneLoopMaxElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(QuantSortOutComputeTilingDataOp, QuantSortOutComputeTilingData)

BEGIN_TILING_DATA_DEF(QuantGatherOutComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, activateRows);
TILING_DATA_FIELD_DEF(int64_t, perCoreRows);
TILING_DATA_FIELD_DEF(int64_t, perCoreK);
TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopK);
TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopK);
TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopRows);
TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopRows);
TILING_DATA_FIELD_DEF(int64_t, lastCoreRows);
TILING_DATA_FIELD_DEF(int64_t, lastCoreK);
TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopK);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopK);
TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopRows);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopRows);
TILING_DATA_FIELD_DEF(int64_t, maxColsOneLoop);
TILING_DATA_FIELD_DEF(int64_t, splitFlag);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(QuantGatherOutComputeTilingDataOp, QuantGatherOutComputeTilingData)

BEGIN_TILING_DATA_DEF(MoeInitRoutingQuantTilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, cols);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(float, offset);
TILING_DATA_FIELD_DEF_STRUCT(QuantVBSComputeTilingData, vbsComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(QuantVMSMiddleComputeTilingData, vmsMiddleComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(QuantSortOutComputeTilingData, sortOutComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(QuantGatherOutComputeTilingData, srcToDstComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(QuantGatherOutComputeTilingData, gatherOutComputeParamsOp);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeInitRoutingQuant, MoeInitRoutingQuantTilingData)
struct MoeInitRoutingQuantCompileInfo {
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INIT_ROUTING_QUANT_H