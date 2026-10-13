/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_reduce_scatter_gen_task.cpp
 * \brief
 */
#include <vector>

#include "../common/inc/op_mc2.h"
#include "platform/platform_info.h"

#ifdef BUILD_OPEN_PROJECT
#include "../common/inc/mc2_gen_task_ops_utils.h"
#include "../common/inc/mc2_log.h"
#include "graph/kernel_launch_info.h"
#include "graph/arg_desc_info.h"
#include "register/op_impl_registry.h"
#else
#include "mc2_gen_task_utils.h"
#include "register/op_ct_impl_registry.h"
#endif

namespace ops {
#ifdef BUILD_OPEN_PROJECT
static ge::Status MatmulReduceScatterCalcOpParam(gert::ExeResGenerationContext *context)
{
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

static ge::Status MatmulReduceScatterGenTask(const gert::ExeResGenerationContext *context,
                                             std::vector<std::vector<uint8_t>> &tasks)
{
    return Mc2GenTaskOpsUtils::CommonKFCMc2GenTask(context, tasks);
}

IMPL_OP(MatmulReduceScatter).CalcOpParam(MatmulReduceScatterCalcOpParam).GenerateTask(MatmulReduceScatterGenTask);
#else // mc2 gen task utils
static ge::Status MatmulReduceScatterGenTaskCallback(const gert::ExeResGenerationContext *context,
                                                     std::vector<domi::TaskDef> &tasks)
{
    return Mc2GenTaskUtils::Mc2GenTaskCallBack910A2(context, tasks);
}

static ge::Status MatmulReduceScatterCalcOpParam(gert::ExeResGenerationContext *context)
{
    return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

static ge::Status MatmulReduceScatterGenTask(const gert::ExeResGenerationContext *context,
                                             std::vector<std::vector<uint8_t>> &tasks)
{
    return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, MatmulReduceScatterGenTaskCallback);
}

IMPL_OP_CT(MatmulReduceScatter).CalcOpParam(MatmulReduceScatterCalcOpParam).GenerateTask(MatmulReduceScatterGenTask);
#endif
} // namespace ops