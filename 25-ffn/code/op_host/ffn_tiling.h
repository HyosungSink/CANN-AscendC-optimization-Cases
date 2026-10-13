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
 * \file ffn_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FFN_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FFN_H_
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FFNBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, totalTokens);
TILING_DATA_FIELD_DEF(uint32_t, k1);
TILING_DATA_FIELD_DEF(uint32_t, n1);
TILING_DATA_FIELD_DEF(uint32_t, n2);
TILING_DATA_FIELD_DEF(uint32_t, expertNum);
TILING_DATA_FIELD_DEF(uint32_t, maxTokens);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, activeType);
TILING_DATA_FIELD_DEF(uint64_t, workspace1Size);
TILING_DATA_FIELD_DEF(uint64_t, workspace2Size);
TILING_DATA_FIELD_DEF(uint32_t, syncWorkspaceSize);
TILING_DATA_FIELD_DEF(uint32_t, dataTypeSize);
TILING_DATA_FIELD_DEF(uint32_t, scale1GroupNum);
TILING_DATA_FIELD_DEF(uint32_t, scale2GroupNum);
TILING_DATA_FIELD_DEF(uint32_t, tokensIndexFlag);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FFNBaseParamsOp, FFNBaseParams)

BEGIN_TILING_DATA_DEF(FFNSingleCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, baseM1);
TILING_DATA_FIELD_DEF(uint32_t, baseN1);
TILING_DATA_FIELD_DEF(uint32_t, baseN2);
TILING_DATA_FIELD_DEF(uint32_t, ubCalSize);
TILING_DATA_FIELD_DEF(uint32_t, ubRestBytes);
TILING_DATA_FIELD_DEF(uint32_t, mm1ResUbSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FFNSingleCoreParamsOp, FFNSingleCoreParams)

BEGIN_TILING_DATA_DEF(FFNTilingData)
TILING_DATA_FIELD_DEF_STRUCT(FFNSingleCoreParams, ffnSingleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(FFNBaseParams, ffnBaseParams);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FFN, FFNTilingData)
} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_FFN_H_