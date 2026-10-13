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
 * \file moe_distribute_combine_add_rms_norm_tiling.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>

#include "../../common/inc/tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../common/inc/mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2_tiling.h"
#include "../../common/inc/mc2_hcom_topo_info.h"
#include "../../moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_tiling_helper.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace optiling {
static ge::graphStatus MoeDistributeCombineAddRmsNormTilingFunc(gert::TilingContext* context)
{
    ge::graphStatus ret = optiling::MoeDistributeCombineV2TilingFunc(context);
    return ret;
}

struct MoeDistributeCombineAddRmsNormCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeCombineAddRmsNorm(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeCombineAddRmsNorm)
    .Tiling(MoeDistributeCombineAddRmsNormTilingFunc)
    .TilingParse<MoeDistributeCombineAddRmsNormCompileInfo>(TilingParseForMoeDistributeCombineAddRmsNorm);
} // namespace optiling