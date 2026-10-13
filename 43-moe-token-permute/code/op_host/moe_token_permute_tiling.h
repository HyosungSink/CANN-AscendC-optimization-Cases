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
 * \file moe_token_permute_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_PERMUTE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_PERMUTE_H_
#include <cmath>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "../common/include/tiling_base/tiling_base.h"
#include "../common/include/tiling_base/tiling_templates_registry.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(PermuteVBSComputeTilingData)
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
TILING_DATA_FIELD_DEF(int64_t, lastCoreWSindex);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PermuteVBSComputeTilingDataOp, PermuteVBSComputeTilingData)

BEGIN_TILING_DATA_DEF(PermuteVMSMiddleComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PermuteVMSMiddleComputeTilingDataOp, PermuteVMSMiddleComputeTilingData)

BEGIN_TILING_DATA_DEF(PermuteSortOutComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, oneLoopMaxElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PermuteSortOutComputeTilingDataOp, PermuteSortOutComputeTilingData)

BEGIN_TILING_DATA_DEF(IndexCopyComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, frontCoreNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreNum);
TILING_DATA_FIELD_DEF(int64_t, coreCalcNum);
TILING_DATA_FIELD_DEF(int64_t, coreCalcTail);
TILING_DATA_FIELD_DEF(int64_t, oneTokenBtypeSize);
TILING_DATA_FIELD_DEF(int64_t, onceIndicesTokenMoveTimes);
TILING_DATA_FIELD_DEF(int64_t, onceUbTokenNums);
TILING_DATA_FIELD_DEF(int64_t, onceIndicesTokenNums);
TILING_DATA_FIELD_DEF(int64_t, onceIndices);
TILING_DATA_FIELD_DEF(int64_t, oneTokenlastMove);
TILING_DATA_FIELD_DEF(int64_t, oneTokenOnceMove);
TILING_DATA_FIELD_DEF(int64_t, oneTokenMoveTimes);
TILING_DATA_FIELD_DEF(int64_t, frontCoreLoop);
TILING_DATA_FIELD_DEF(int64_t, frontCoreLastTokenNums);
TILING_DATA_FIELD_DEF(int64_t, tailCoreLoop);
TILING_DATA_FIELD_DEF(int64_t, tailCoreLastTokenNums);
TILING_DATA_FIELD_DEF(int64_t, tailLastonceIndicesTokenMoveTimes);
TILING_DATA_FIELD_DEF(int64_t, tailLastIndicesLastTokenNums);
TILING_DATA_FIELD_DEF(int64_t, frontLastonceIndicesTokenMoveTimes);
TILING_DATA_FIELD_DEF(int64_t, frontLastIndicesLastTokenNums);
TILING_DATA_FIELD_DEF(int64_t, numOutTokens);
TILING_DATA_FIELD_DEF(int64_t, tokenUB);
TILING_DATA_FIELD_DEF(int64_t, indicesUB);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(IndexCopyComputeTilingDataOp, IndexCopyComputeTilingData)

BEGIN_TILING_DATA_DEF(MoeTokenPermuteTilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, cols);
TILING_DATA_FIELD_DEF(int64_t, colsAlign);

TILING_DATA_FIELD_DEF(int64_t, topK);
TILING_DATA_FIELD_DEF_STRUCT(PermuteVBSComputeTilingData, vbsComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(PermuteVMSMiddleComputeTilingData, vmsMiddleComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(PermuteSortOutComputeTilingData, sortOutComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(IndexCopyComputeTilingData, indexCopyComputeParamsOp);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeTokenPermute, MoeTokenPermuteTilingData)
struct MoeTokenPermuteCompileInfo {
};
} // namespace optiling
#endif // MOE_TOKEN_PERMUTE