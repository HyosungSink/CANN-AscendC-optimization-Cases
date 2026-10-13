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
 * \file aclnn_matmul_reduce_scatter.cpp
 * \brief
 */
#include "aclnn_matmul_reduce_scatter.h"
#include "../matmul_reduce_scatter_v2/op_api/aclnn_matmul_reduce_scatter_v2.h"
#include "securec.h"
#include "acl/acl.h"
#include "../common/inc/op_mc2.h"
#include "../common/inc/op_mc2_def.h"
#include "../3rd/ops_legacy/op_api/aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "../3rd/common/op_host/op_api/matmul_util.h"
#include "../common/inc/hccl_util.h"
#include "../common/inc/mc2_aclnn_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif
static constexpr int64_t KVALUE_MIN = 256;
static constexpr int64_t KVALUE_MAX = 65535;
static constexpr int64_t NUM_ACL_STOP_ON_FAILURE = 1;
static constexpr size_t TWO_DIMS = 2;

typedef struct {
  uint32_t id;
  const char *funcName;
  bool hasReg;
} NnopbaseDfxId;

extern aclnnStatus aclnnInnerMatmulReduceScatterGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                             const aclTensor *bias,
                                                             const char *group, const char *reduce_op, bool transposeX1, bool transposeX2,
                                                             int64_t commTurn, int64_t rankSize,
                                                             const aclTensor *output, uint64_t *workspaceSize,
                                                             aclOpExecutor **executor);
extern aclnnStatus aclnnInnerMatmulReduceScatter(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                 aclrtStream stream);               
extern "C" void* NnopbaseGetUserHandle(void *executor);
extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);
extern "C" void NnopbaseReportLaunchInfo(const uint64_t beginTime, const char *const opType);
extern "C" aclnnStatus NnopbaseReportAicpuAdditionInfo(const uint64_t timeStamp, const char *const opType);
extern "C" aclnnStatus __attribute__((weak)) NnopbaseDisableOptionalInput(void *executor, const size_t irIndex);
extern "C" aclnnStatus NnopbaseGetAttrAddr(void *executor, const size_t index, void **attrAddr, size_t *attrLen);
extern "C" void NnopbaseGetOutputTensorAddr(void *executor, const size_t index, void **addr);
extern "C" void NnopbaseGetInputTensorAddr(void *executor, const size_t index, void **addr);
extern "C" void NnopbaseSetInputTensorAddr(void *executor, const size_t index, const void *const addr);
extern "C" void NnopbaseGetTilingData(void *executor, void **tilingData, uint64_t *dataLen);
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);


static inline bool IsAscend910A5(void)
{
    return false;
}

static uint8_t GetDebugMode() {
  auto debugModeEnv = getenv("ASCEND_MC2_DEBUG_MODE");
  uint8_t debugMode = 0;
  if (debugModeEnv != nullptr) {
    debugMode = std::atoi(debugModeEnv);
  }
  OP_LOGD("Debug mode is %u.", debugMode);
  return debugMode;
}

// 检查入参是否为nullptr
static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* output) {
  OP_CHECK_NULL(x1, return false);
  OP_CHECK_NULL(x2, return false);
  OP_CHECK_NULL(output, return false);
  return true;
}
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT16, op::DataType::DT_BF16
};
static bool CheckDtypeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* output) {
  // 检查x1、x2、bias、output的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(output, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(x2, DTYPE_SUPPORT_LIST, return false);
  // 检查bias的数据类型是否在算子的支持列表内
  if (bias != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST, return false);
  }

  // 检查x1和output的数据类型是否相同
  OP_CHECK_DTYPE_NOT_SAME(x1, output, return false);
  // 检查x1和x2的数据类型是否相同
  OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
  // 检查output和bias的数据类型是否相同
  if (bias != nullptr) {
    OP_CHECK_DTYPE_NOT_SAME(bias, x1, return false);
  }
  return true;
}

static bool CheckNotEmptyTensor(const aclTensor* x1) {
  auto kValue = x1->GetViewShape().GetDim(1);
  OP_API_CHECK((kValue == 0), {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "X1 is empty tensor with zero dimK, which is unsupported.");
    return false;
  });
  return true;
}

// 检查传入的reduction数值是否在可选范围内
static bool CheckAttr(int64_t streamMode) {
  if (streamMode != NUM_ACL_STOP_ON_FAILURE) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected streamMode to be 1, but got %ld.", streamMode);
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                               int64_t streamMode, const aclTensor *output) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(x1, x2, output), ACLNN_ERR_PARAM_NULLPTR);

  if (bias != nullptr) {
   OP_LOGW("MatmulReducescatter, The current version does not support the scenario where bias is not 0. "
    "Features and accuracy are not guaranteed if inputting bias with values other than 0s.");
  }

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(x1, x2, bias, output), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查attr是否符合规则
  CHECK_RET(CheckAttr(streamMode), ACLNN_ERR_PARAM_INVALID);

  CHECK_RET(CheckNotEmptyTensor(x1), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static bool CheckShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *output, bool isTransA) {
  OP_CHECK_WRONG_DIMENSION(x1, TWO_DIMS, return false);
  OP_CHECK_WRONG_DIMENSION(x2, TWO_DIMS, return false);
  OP_API_CHECK(isTransA, {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The x1 should not be transposed, but it is transposed.");
    return false;
  });
  auto kVal1 = x1->GetViewShape().GetDim(1);
  auto kVal2 = x2->GetViewShape().GetDim(0);
  if (GetDebugMode() != 4) { // 4 ONLY_AICPU
    OP_API_CHECK((kVal1 != kVal2), {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
      "The k-axis of x1 and x2 should be same, but x1's k-axis is: %ld and x2's k-axis is: %ld.", kVal1, kVal2);
      return false;
    });
    OP_API_CHECK((kVal1 >= KVALUE_MAX || kVal1 < KVALUE_MIN), {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The k-axis should be in range[256, 65535), but it is: %ld.", kVal1);
      return false;
    });

    auto nVal2 = output->GetViewShape().GetDim(1);
    auto nVal1 = x2->GetViewShape().GetDim(1);
    OP_API_CHECK((nVal1 != nVal2), {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
      "The n-axis of x2 and output should be same, but x2's n-axis is: %ld and output's n-axis is: %ld.", nVal1, nVal2);
      return false;
    });
  }
  return true;
}

aclnnStatus aclnnMatmulReduceScatterGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                                 const char *group, const char *reduce_op, int64_t commTurn,
                                                 int64_t streamMode, const aclTensor *output,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor) {
  uint64_t timeStamp = NnopbaseMsprofSysTime();
  // 固定写法，参数检查
  auto retParam = CheckParams(x1, x2, bias, streamMode, output);
  CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
  // 处理空tensor
  if (x1->IsEmpty() || x2->IsEmpty()) {
    OP_LOGD("MatmulReduceScatter, dealing with empty tensor.");
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  OP_LOGD("X1 is %s.", x1->ToString().GetString());
  OP_LOGD("X2 is %s.", x2->ToString().GetString());
  OP_LOGD("output is %s.", output->ToString().GetString());

  uint32_t rankSize = 0;
  bool transposeX1 = Ops::Transformer::IsTransposeLastTwoDims(x1);
  bool transposeX2 = Ops::Transformer::IsTransposeLastTwoDims(x2);
  CHECK_RET(CheckShape(x1, x2, output, transposeX1), ACLNN_ERR_PARAM_INVALID);
  // 【A2】检查x2矩阵非连续合法性
  if (op::GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
    if (!Ops::Transformer::IsTransposeLastTwoDims(x2) && !MC2Aclnn::IsTensorContiguous(x2)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The x2 without transpose in MatmulReduceScatter must be contiguous,"
              "but it is non-contiguous.");
      return ACLNN_ERR_PARAM_INVALID;
    }
  }
  if (IsAscend910A5()) {
    const char *commMode = "ccu";
    return aclnnMatmulReduceScatterV2GetWorkspaceSize(x1, x2, bias, nullptr, nullptr, nullptr, 0, group, reduce_op,
                                                      commTurn, streamMode, 0, commMode, const_cast<aclTensor *>(output),
                                                      nullptr, workspaceSize, executor);
  }
  aclnnStatus ret = aclnnInnerMatmulReduceScatterGetWorkspaceSize(x1, x2, bias, group, reduce_op, transposeX1,
                                                              transposeX2, commTurn, rankSize, output,
                                                              workspaceSize, executor);
  OP_LOGD("MatmulReduceScatter, aclnnnGetWorkspaceSize ret %d.", ret);
  static NnopbaseDfxId dfxId = {0x60000, __func__, false};
  NnopbaseReportApiInfo(timeStamp, dfxId);
  return ret;
}

aclnnStatus aclnnMatmulReduceScatter(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     aclrtStream stream) {
  if (IsAscend910A5()) {
    return aclnnMatmulReduceScatterV2(workspace, workspaceSize, executor, stream);
  }
  if (workspace == nullptr || workspaceSize == 0UL) {
    OP_LOGD("Skip the api for empty tensor, workspace size %lu.", workspaceSize);
    return ACLNN_SUCCESS;
  }
  uint64_t timeStamp = NnopbaseMsprofSysTime();
  auto ret = aclnnInnerMatmulReduceScatter(workspace, workspaceSize, executor, stream);
  if (ret != 0) {
    OP_LOGE(ACLNN_ERR_INNER, "MatmulReduceScatter launch task failed.");
    return ACLNN_ERR_INNER;
  }
  static NnopbaseDfxId dfxId = {0x60000, __func__, false};
  NnopbaseReportApiInfo(timeStamp, dfxId);
  return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
