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
 * \file moe_gating_top_k_softmax_tiling.cpp
 * \brief
 */
#include "moe_gating_top_k_softmax_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "../common/include/tiling_base/tiling_templates_registry.h"
using namespace Ops::Transformer::OpTiling;
using namespace AscendC;
namespace optiling {

static ge::graphStatus TilingMoeGatingTopKSoftmax(gert::TilingContext* context)
{
    // 初始化算子Tiling类
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4MoeGatingTopKSoftmax(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeGatingTopKSoftmax)
    .Tiling(TilingMoeGatingTopKSoftmax)
    .TilingParse<MoeGatingTopKSoftmaxCompileInfo>(TilingPrepare4MoeGatingTopKSoftmax);
} // namespace optiling