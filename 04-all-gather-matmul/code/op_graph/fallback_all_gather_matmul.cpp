/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../common/include/fallback/fallback.h"
#include "../common/inc/op_mc2.h"
#include "../common/inc/mc2_log.h"

namespace fallback {
const char *allGatherInfo = "AllGatherMatmulFallback";

static ge::graphStatus AllGatherMatmulExecuteFunc(gert::OpExecuteContext *host_api_ctx) {
  OPS_LOG_D(allGatherInfo, "Start to fallback for allgather matmul.");
  OPS_ERR_IF(host_api_ctx == nullptr, OPS_LOG_E(allGatherInfo, "host_api_ctx is null"), return ge::GRAPH_FAILED);
  
  const auto x1 = host_api_ctx->GetInputTensor(static_cast<size_t>(ops::MC2InputIdx::K_X1));
  OPS_ERR_IF(x1 == nullptr, OPS_LOG_E(allGatherInfo, "x1 is null"), return ge::GRAPH_FAILED);

  const auto x2 = host_api_ctx->GetInputTensor(static_cast<size_t>(ops::MC2InputIdx::K_X2));
  OPS_ERR_IF(x2 == nullptr, OPS_LOG_E(allGatherInfo, "x2 is null"), return ge::GRAPH_FAILED);

  const auto bias = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(ops::MC2InputIdx::K_BIAS));

  const auto y = host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::MC2OutputIdx::K_Y));
  OPS_ERR_IF(y == nullptr, OPS_LOG_E(allGatherInfo, "y is null"), return ge::GRAPH_FAILED);

  const auto gatherOut = host_api_ctx->GetOutputTensor(static_cast<size_t>(ops::MC2OutputIdx::K_GATHER_OUT));
  OPS_ERR_IF(gatherOut == nullptr, OPS_LOG_E(allGatherInfo, "gatherOut is null"), return ge::GRAPH_FAILED);

  const auto attrs = host_api_ctx->GetAttrs();
  OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(allGatherInfo, "attrs is null"), return ge::GRAPH_FAILED);

  const char *group = attrs->GetStr(static_cast<size_t>(ops::AllGatherMMAttrIdx::K_GROUP));
  OPS_ERR_IF(group == nullptr, OPS_LOG_E(allGatherInfo, "group is null"), return ge::GRAPH_FAILED);

  const bool *trans_x1_ptr = attrs->GetBool(static_cast<size_t>(ops::AllGatherMMAttrIdx::K_TRANS_X1));
  const bool trans_x1 = (trans_x1_ptr != nullptr ? *trans_x1_ptr : false);
  auto x1_acl = ConvertMmType(x1, trans_x1);
  OPS_ERR_IF(x1_acl == nullptr, OPS_LOG_E(allGatherInfo, "x1_acl is null"), return ge::GRAPH_FAILED);

  const bool *trans_x2_ptr = attrs->GetBool(static_cast<size_t>(ops::AllGatherMMAttrIdx::K_TRANS_X2));
  const bool trans_x2 = (trans_x2_ptr != nullptr ? *trans_x2_ptr : false);
  auto x2_acl = ConvertMmType(x2, trans_x2);
  OPS_ERR_IF(x2_acl == nullptr, OPS_LOG_E(allGatherInfo, "x2_acl is null"), return ge::GRAPH_FAILED);

  const int64_t *gather_index_ptr = attrs->GetInt(static_cast<size_t>(ops::AllGatherMMAttrIdx::K_GATHER_IDX));
  const int64_t gather_index = (gather_index_ptr != nullptr ? *gather_index_ptr : 0);

  const int64_t *comm_turn_ptr = attrs->GetInt(static_cast<size_t>(ops::AllGatherMMAttrIdx::K_COMM_TURN));
  const int64_t comm_turn = (comm_turn_ptr != nullptr ? *comm_turn_ptr : 0);
  const int64_t stream_mode = 1; // STOP_ON_FAILURE
  const auto api_ret = EXEC_OPAPI_CMD(aclnnAllGatherMatmul, x1_acl, x2_acl, bias, group, gather_index, comm_turn,
                                      stream_mode, y, gatherOut);
  OPS_ERR_IF(api_ret != ge::GRAPH_SUCCESS, OPS_LOG_E(allGatherInfo, "Aclnn api error code %u", api_ret),
           return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(AllGatherMatmul).OpExecuteFunc(AllGatherMatmulExecuteFunc);
}  // namespace fallback
