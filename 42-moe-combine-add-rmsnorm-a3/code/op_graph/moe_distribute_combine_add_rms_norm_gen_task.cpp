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
 * \file moe_distribute_combine_add_rms_norm_gen_task.cpp
 * \brief
 */
#include <vector>
#include <set>
#include <string>

#include "../common/inc/op_mc2.h"
#include "platform/platform_info.h"

#ifdef BUILD_OPEN_PROJECT
#include "../common/inc/mc2_gen_task_ops_utils.h"
#include "../common/inc/mc2_moe_gen_task_ops_utils.h"
#include "graph/arg_desc_info.h"
#include "graph/kernel_launch_info.h"
#include "register/op_impl_registry.h"
#include "../common/inc/mc2_log.h"
#else
#include "ops_error.h"
#include "mc2_gen_task_moe.h"
#include "mc2_gen_task_utils.h"
#include "register/op_ct_impl_registry.h"
#include "register/op_ext_gentask_registry.h"
#endif

namespace ops {

static bool IsPlatform910B(const char *nodeName)
{
    fe::PlatFormInfos platform_info;
    fe::OptionalInfos optional_info;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
        ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "Cannot get platform info!");
        return false;
    }
    static std::set<std::string> supported_soc = {"Ascend910B"};
    std::string short_soc_version;
    if (!platform_info.GetPlatformRes("version", "Short_SoC_version", short_soc_version) || short_soc_version.empty()) {
        OPS_LOG_E(nodeName, "Cannot get short soc version!");
        return false;
    }
    OPS_LOG_D(nodeName, "Get soc version: %s", short_soc_version.c_str());
    return supported_soc.count(short_soc_version) > 0;
}

#ifdef BUILD_OPEN_PROJECT
ge::Status MoeDistributeCombineAddRmsNormCalcParamFunc(gert::ExeResGenerationContext *context)
{
    const ge::AscendString name = "aicpu kfc server";
    const ge::AscendString reuseKey = "kfc_stream";
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, name, reuseKey);
}

ge::Status MoeDistributeCombineAddRmsNormGenTaskFunc(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks)
{
    const char *nodeName = context->GetNodeName();
    OPS_LOG_I(nodeName, "MC2 Generate task start.");
    if (IsPlatform910B(nodeName)) {
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
    return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallbackV2(context, tasks);
}

// new ver
IMPL_OP(MoeDistributeCombineAddRmsNorm)
    .CalcOpParam(MoeDistributeCombineAddRmsNormCalcParamFunc)
    .GenerateTask(MoeDistributeCombineAddRmsNormGenTaskFunc);
#else // mc2 gen task utils
ge::Status MoeDistributeCombineAddRmsNormCalcParamFunc(gert::ExeResGenerationContext *context)
{
    const ge::AscendString name = "aicpu kfc server";
    const ge::AscendString reuseKey = "kfc_stream";
    return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, name, reuseKey);
}

ge::Status MoeDistributeCombineAddRmsNormGenTaskFunc(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks)
{
    const char *nodeName = context->GetNodeName();
    if (IsPlatform910B(nodeName)) {
        return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2GenTaskMoe::Mc2MoeGenTaskCallback);
    }
    return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2GenTaskMoe::Mc2MoeGenTaskCallbackV2);
}

IMPL_OP_CT(MoeDistributeCombineAddRmsNorm)
    .CalcOpParam(MoeDistributeCombineAddRmsNormCalcParamFunc)
    .GenerateTask(MoeDistributeCombineAddRmsNormGenTaskFunc);

REGISTER_EXT_TASK_TYPE(MoeDistributeCombineAddRmsNorm, fe::ExtTaskType::kAicoreTask);
#endif
} // namespace ops
