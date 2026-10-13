/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_all_gather_matmul_v2.h"
#include "securec.h"
#include "acl/acl.h"
#include "../../common/inc/op_mc2.h"
#include "../../common/inc/op_mc2_def.h"
#include "../../3rd/ops_legacy/op_api/aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "../../3rd/common/op_host/op_api/matmul_util.h"
#include "../../common/inc/hccl_util.h"
#include "../../common/inc/mc2_aclnn_util.h"

using namespace op;
using namespace Ops::Transformer;

#ifdef __cplusplus
extern "C" {
#endif
static constexpr int64_t NUM_ACL_STOP_ON_FAILURE = 1;
static constexpr int64_t ONE_DIMS = 1;
static constexpr int64_t SCALAR = 1;
static constexpr int64_t TWO_DIMS = 2;
static constexpr int64_t KVALUE_MIN = 256;
static constexpr int64_t KVALUE_MAX = 65535;
typedef struct {
  uint32_t id;
  const char *funcName;
  bool hasReg;
} NnopbaseDfxId;

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerAllGatherMatmulV2GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                               const aclTensor *bias, const aclTensor *x1Scale,
                                                               const aclTensor *x2Scale, const aclTensor *quantScale,
                                                               const char *group, bool transposeX1, bool transposeX2,
                                                               int64_t gatherIndex, int64_t commTurn, int64_t rankSize,
                                                               int64_t blockSize, int64_t groupSize,
                                                               bool isGatherOut, bool isAMaxOut, int64_t yDtype, const char *commMode,
                                                               aclTensor *output, aclTensor *gatherOut,
                                                               aclTensor *amaxOut, uint64_t *workspaceSize,
                                                               aclOpExecutor **executor);
extern aclnnStatus aclnnInnerAllGatherMatmulV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             aclrtStream stream);
extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// check nullptr
static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* output)
{
  OP_CHECK_NULL(x1, return false);
  OP_CHECK_NULL(x2, return false);
  OP_CHECK_NULL(output, return false);
  return true;
}
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2,
  op::DataType::DT_HIFLOAT8
};

static const std::initializer_list<op::DataType> BIAS_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT
};

static const std::initializer_list<op::DataType> FP8_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_HIFLOAT8
};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = BIAS_DTYPE_SUPPORT_LIST;

static const std::initializer_list<op::DataType> AIV_MODE_INPUT_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_INT8, op::DataType::DT_INT4
};

static const std::initializer_list<op::DataType> AIV_MODE_OUTPUT_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT16, op::DataType::DT_BF16
};

static bool CheckSupportDtype(const aclTensor* x1, const std::initializer_list<op::DataType>& supportTypes)
{
  return std::find(supportTypes.begin(), supportTypes.end(), x1->GetDataType()) != supportTypes.end();
}

static bool CheckOutDtypeValid(int64_t outDtype, const std::initializer_list<op::DataType>& validTypes)
{
  OP_LOGD("outDtype value is: %ld", outDtype);
  return std::find(validTypes.begin(), validTypes.end(), static_cast<op::DataType>(outDtype)) != validTypes.end();
}

static bool CheckDataTypeFp16Valid(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                const aclTensor* output)
{
  OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
  OP_CHECK_DTYPE_NOT_SAME(x1, output, return false);
  if (bias != nullptr) {
    // 当x1类型为fp16/bf16时, bias类型为与x1保持一致
    OP_CHECK_DTYPE_NOT_SAME(bias, x1, return false);
  }
  return true;
}

static bool CheckDataTypeFp8Valid(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias)
{
  if (x1->GetDataType() == op::DataType::DT_HIFLOAT8) {
    OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
  }
  if (bias != nullptr) {
    // 当x1类型为fp16/bf16时, bias类型为与x1保持一致
    if (bias->GetDataType() != op::DataType::DT_FLOAT) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias is just support float type when x1 is fp8 or hif8. but bias Dtype is %u",
              static_cast<uint32_t>(bias->GetDataType()));
      return false;
    }
  }
  return true;
}

static bool IsFp16orBf16Input(const aclTensor* x1)
{
  return (x1->GetDataType() == op::DataType::DT_FLOAT16) || (x1->GetDataType() == op::DataType::DT_BF16);
}

static bool CheckDtypeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* output)
{
  OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(x2, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(output, OUT_DTYPE_SUPPORT_LIST, return false);
  if (bias != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(bias, BIAS_DTYPE_SUPPORT_LIST, return false);
  }

  if (IsFp16orBf16Input(x1)) {
    return CheckDataTypeFp16Valid(x1, x2, bias, output);
  } else if (CheckSupportDtype(x1, FP8_DTYPE_SUPPORT_LIST)) {
    return CheckDataTypeFp8Valid(x1, x2, bias);
  }
  return true;
}

static bool CheckAIVModeDtypeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* output)
{
  OP_CHECK_DTYPE_NOT_SUPPORT(x1, AIV_MODE_INPUT_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(x2, AIV_MODE_INPUT_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(output, AIV_MODE_OUTPUT_DTYPE_SUPPORT_LIST, return false);
  return true;
}


static bool CheckAttr(int64_t streamMode)
{
  if (streamMode != NUM_ACL_STOP_ON_FAILURE) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected streamMode to be 1, but got %ld.", streamMode);
    return false;
  }
  return true;
}

static bool IsGatherOut(const aclTensor *gatherOut)
{
  OP_CHECK_NULL(gatherOut, return false);
  if (gatherOut->IsEmpty()) {
    OP_LOGD("AllGahterMatmulV2, get gather out is false.");
    return false;
  }
  return true;
}

static bool CheckShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *output, const aclTensor *gatherOut,
                       bool isTransA)
{
  OP_CHECK_WRONG_DIMENSION(x1, TWO_DIMS, return false);
  OP_CHECK_WRONG_DIMENSION(x2, TWO_DIMS, return false);
  // A矩阵不能转置
  OP_API_CHECK(isTransA, {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The x1 should not be transposed, but it is transposed.");
    return false;
  });

  const auto kValX1 = x1->GetViewShape().GetDim(1);
  const auto kValX2 = x2->GetViewShape().GetDim(0);
  OP_API_CHECK((kValX1 != kValX2), {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
    "The k-axis of x1 and x2 should be same, but x1's k-axis is: %ld and x2's k-axis is: %ld.", kValX1, kValX2);
    return false;
  });

  OP_API_CHECK((kValX1 < KVALUE_MIN) || (kValX1 >= KVALUE_MAX), {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The k-axis should be in range[256, 65535), but it is: %ld.", kValX1);
    return false;
  });

  const auto nVal1 = x2->GetViewShape().GetDim(1);
  const auto nVal2 = output->GetViewShape().GetDim(1);
  OP_API_CHECK((nVal1 != nVal2), {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
    "The n-axis of x2 and output should be same, but x2's n-axis is: %ld and output's n-axis is: %ld.", nVal1, nVal2);
    return false;
  });
  return true;
}

// 分别对输入类型为fp8/hif8和fp16/bf16数据类型进行分别校验
static aclnnStatus CheckParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                               int64_t streamMode, const aclTensor *output)
{
  CHECK_RET(CheckNotNull(x1, x2, output), ACLNN_ERR_PARAM_NULLPTR);

  CHECK_RET(CheckDtypeValid(x1, x2, bias, output), ACLNN_ERR_PARAM_INVALID);

  CHECK_RET(CheckAttr(streamMode), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus CheckParamsAndShapeForAIVMode(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *output,
                                                const aclTensor *gatherOut, bool isTransA, bool isViewTransB, int64_t streamMode)
{
  CHECK_RET(CheckNotNull(x1, x2, output), ACLNN_ERR_PARAM_NULLPTR);

  CHECK_RET(CheckAIVModeDtypeValid(x1, x2, output), ACLNN_ERR_PARAM_INVALID);

  CHECK_RET(CheckAttr(streamMode), ACLNN_ERR_PARAM_INVALID);

  OP_CHECK_WRONG_DIMENSION(x1, TWO_DIMS, return ACLNN_ERR_PARAM_INVALID);
  OP_CHECK_WRONG_DIMENSION(x2, TWO_DIMS, return ACLNN_ERR_PARAM_INVALID);
  // A矩阵不能转置
  OP_API_CHECK(isTransA, {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The x1 should not be transposed, but it is transposed.");
    return ACLNN_ERR_PARAM_INVALID;
  });

  const auto kValX1 = x1->GetViewShape().GetDim(1);
  const auto kValX2 = x2->GetViewShape().GetDim(isViewTransB ? 1 : 0);
  OP_API_CHECK((kValX1 != kValX2), {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
    "The k-axis of x1 and x2 should be same, but x1's k-axis is: %ld and x2's k-axis is: %ld.", kValX1, kValX2);
    return ACLNN_ERR_PARAM_INVALID;
  });

  if (IsGatherOut(gatherOut)) {
    const auto kVal = gatherOut->GetViewShape().GetDim(1);
    OP_API_CHECK((kValX1 != kVal), {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "The k-axis of x1 and gatherOut should be same, but x1's k-axis is: %ld and gatherOut's k-axis is: %ld.",
      kValX1, kVal);
      return ACLNN_ERR_PARAM_INVALID;
    });
  }

  const auto nVal1 = x2->GetViewShape().GetDim(isViewTransB ? 0 : 1);
  const auto nVal2 = output->GetViewShape().GetDim(1);
  OP_API_CHECK((nVal1 != nVal2), {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
    "The n-axis of x2 and output should be same, but x2's n-axis is: %ld and output's n-axis is: %ld.", nVal1, nVal2);
    return ACLNN_ERR_PARAM_INVALID;
  });
  return ACLNN_SUCCESS;
}

static inline bool CheckParamDtypeFP8Vaild(const aclTensor* tensor)
{
  return tensor->GetDataType() == op::DataType::DT_FLOAT;
}

static aclnnStatus CheckScale(const aclTensor* x1Scale, const aclTensor* x2Scale, const aclTensor* quantScale)
{
  if (x1Scale == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input is fp8 or hif8, x1Scale should not be nullptr.");
    return ACLNN_ERR_PARAM_INVALID;
  }
  if (x2Scale == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input is fp8 or hif8, x2Scale should not be nullptr.");
    return ACLNN_ERR_PARAM_INVALID;
  }
  // 如果scaleInV1 和 scaleInV2都不为空指针则为scalar类型数据
  auto x1ScaleLen = x1Scale->GetViewShape().GetDim(0);
  auto x2ScaleLen = x2Scale->GetViewShape().GetDim(0);
  OP_LOGD("AllGahterMatmulV2, x1ScaleLen is %ld.", x1ScaleLen);
  OP_LOGD("AllGahterMatmulV2, x2ScaleLen is %ld.", x2ScaleLen);

  // scale不为空指针为则scalar类型
  if (quantScale != nullptr) {
    OP_CHECK_WRONG_DIMENSION(quantScale, ONE_DIMS, return ACLNN_ERR_PARAM_INVALID);
    auto scaleLen = quantScale->GetViewShape().GetDim(0);
    OP_LOGD("AllGahterMatmulV2, scaleLen is %ld.", scaleLen);
    if (scaleLen != SCALAR) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantScale len should be 1, but actual is %ld.", scaleLen);
      return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(CheckParamDtypeFP8Vaild(quantScale), ACLNN_ERR_PARAM_INVALID);
  }
  return ACLNN_SUCCESS;
}

static bool IsAMaxOut(const aclTensor *amaxOut)
{
  if (amaxOut == nullptr) {
    return false;
  }
  if (amaxOut->IsEmpty()) {
    OP_LOGD("AllGahterMatmulV2, get amax out is false.");
    return false;
  }
  return true;
}

// fp16/bf16场景不支持amaxOut输入，fp8/hif8场景amaxOut为scalar且只支持float类型
static bool CheckAMaxOutVaild(const aclTensor* x1, const aclTensor* amaxOut)
{
  if (IsFp16orBf16Input(x1)) {
    if (amaxOut != nullptr) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "not support amaxOut when input datatype is fp16/bf16.");
      return false;
    }
  }
  if (IsAMaxOut(amaxOut)) {
    OP_CHECK_WRONG_DIMENSION(amaxOut, ONE_DIMS, return false);
    auto amaxOutLen = amaxOut->GetViewShape().GetDim(0);
    if (amaxOutLen != SCALAR) {
      return false;
    }
    CHECK_RET(CheckParamDtypeFP8Vaild(amaxOut), false);
  }
  return true;
}

// fp8/hif8场景不支持空tensor输入
static bool DealEmptyTensor(const aclTensor* x1, const aclTensor* x2)
{
  if (x1->IsEmpty()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "AllGahterMatmulV2 not support empty tensor x1 when input datatype is fp8/hif8.");
    return false;
  }

  if (x2->IsEmpty()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "AllGahterMatmulV2 not support empty tensor x2 when input datatype is fp8/hif8.");
    return false;
  }
  return true;
}

static aclnnStatus DealWithX1Empty(uint64_t* workspaceSize, aclOpExecutor** executor)
{
  OP_LOGD("AllGatherMatmulV2 dealing with empty tensor.");
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  *workspaceSize = 0U;
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

static const aclTensor *TransX2Tensor(const aclTensor *x2)
{
  uint64_t storageDimsNum = x2->GetStorageShape().GetDimNum();
  std::vector<int64_t> storageDims(storageDimsNum);
  for (uint64_t i = 0; i < storageDimsNum; i++) {
    storageDims[i] = x2->GetStorageShape().GetDim(i);
  }

  uint64_t viewDimsNum = x2->GetViewShape().GetDimNum();
  std::vector<int64_t> viewDims;
  viewDims.resize(viewDimsNum);
  for (uint64_t i = 0; i < viewDimsNum; i++) {
    viewDims[i] = x2->GetViewShape().GetDim(i);
  }
  // transpose the viewshape last two dimensions
  viewDims[0] = x2->GetViewShape().GetDim(1);
  viewDims[1] = x2->GetViewShape().GetDim(0);

  aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
  aclGetDataType(x2, &dataType);
  std::vector<int64_t> stride(viewDimsNum);
  auto transStride = x2->GetViewStrides();
  stride = std::vector<int64_t>(transStride.begin(), transStride.end());
  // transpose the two dimensions
  stride[0] = transStride[1];
  stride[1] = transStride[0];
  auto offset = x2->GetViewOffset();
  aclFormat format = aclFormat::ACL_FORMAT_ND;

  return aclCreateTensor(viewDims.data(), viewDimsNum, dataType, stride.data(), offset, format, storageDims.data(),
                          storageDimsNum, x2->GetTensor()->GetAddr());
}

static bool checkInputEmptyTensor(const aclTensor *x1, const aclTensor *x2) {
  if (CheckSupportDtype(x1, FP8_DTYPE_SUPPORT_LIST)) {
    CHECK_RET(DealEmptyTensor(x1, x2), ACLNN_ERR_PARAM_INVALID);
  } else {
    const auto kValX1 = x1->GetViewShape().GetDim(1);
    if (kValX1 != 0) {
      if (x1->IsEmpty()) {
        return true;
      }
    } else {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Does not support the case where x1 and x2 are empty tensors when k is 0.");
    }
  }
  return false;
}

aclnnStatus allGatherMatmulV2GetWorkspaceSizeCCUMode(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                                   const aclTensor* x1Scale, const aclTensor* x2Scale,
                                                   const aclTensor* quantScale, int64_t blockSize, const char* group,
                                                   int64_t gatherIndex, int64_t commTurn, int64_t streamMode,
                                                   int64_t groupSize, const char* commMode, aclTensor* output, aclTensor* gatherOut,
                                                   aclTensor* amaxOut, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor)
{
  uint64_t timeStamp = NnopbaseMsprofSysTime();
  auto retParam = CheckParams(x1, x2, bias, streamMode, output);
  CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
  // bf16/fp16 处理空tensor 如果x1不为空 x2为空 需要进行gatherOut
  if (checkInputEmptyTensor(x1, x2))
    return DealWithX1Empty(workspaceSize, executor);

  OP_LOGD("X1 is %s.", x1->ToString().GetString());
  OP_LOGD("X2 is %s.", x2->ToString().GetString());

  if (CheckSupportDtype(x1, FP8_DTYPE_SUPPORT_LIST)) {
      auto retScaleChk = CheckScale(x1Scale, x2Scale, quantScale);
      CHECK_RET(retScaleChk == ACLNN_SUCCESS, retScaleChk);
  }

  uint32_t rankSize = 0;
  bool transposeX1 = IsTransposeLastTwoDims(x1);
  bool transposeX2 = IsTransposeLastTwoDims(x2);

  CHECK_RET(CheckShape(x1, x2, output, gatherOut, transposeX1), ACLNN_ERR_PARAM_INVALID);
  bool isGatherOut = IsGatherOut(gatherOut);
  bool isAMaxOut = IsAMaxOut(amaxOut);
  // 如果为bf16/fp16的,不能输入amaxout, 如果为低精度，amaxout 数据类型只能为float类型且维度为1维
  CHECK_RET(CheckAMaxOutVaild(x1, amaxOut), ACLNN_ERR_PARAM_INVALID);
  // outDtype 通过output类型判断
  uint64_t outDtype = static_cast<uint64_t>(output->GetDataType());
  CHECK_RET(CheckOutDtypeValid(outDtype, OUT_DTYPE_SUPPORT_LIST), ACLNN_ERR_PARAM_INVALID);
  auto transX2 = x2;
  auto transX2Scale = x2Scale;
  if (transposeX2) {
    // x2转置时将两轴shape调换
    if(x2->GetTensor() == nullptr){
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Tensor of x2 is null.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    transX2 = TransX2Tensor(x2);
  }
  if ((x2Scale != nullptr) && MC2Aclnn::IsNeedScaleTrans(x2Scale)) {
    transX2Scale = TransX2Tensor(x2Scale);
  }
  aclnnStatus ret = aclnnInnerAllGatherMatmulV2GetWorkspaceSize(x1, transX2, bias, x1Scale, transX2Scale, quantScale, group,
                                                                transposeX1, transposeX2, gatherIndex, commTurn,
                                                                rankSize, blockSize, groupSize, isGatherOut, isAMaxOut,
                                                                outDtype, commMode, output, gatherOut, amaxOut, workspaceSize,
                                                                executor);
  OP_LOGD("AllGatherMatmulV2, aclnnInnerGetWorkspaceSize ret = %d.", ret);
  static NnopbaseDfxId dfxId = {0x60000, __func__, false};
  NnopbaseReportApiInfo(timeStamp, dfxId);
  return ret;
}

bool IsViewTransposeX2(const aclTensor* x1, const aclTensor* x2, bool isTransX1) 
{
  const auto kValX1 = x1->GetViewShape().GetDim(isTransX1 ? 0 : 1);
  const auto x2Dim0 = x2->GetViewShape().GetDim(0);
  const auto x2Dim1 = x2->GetViewShape().GetDim(1);
  return (kValX1 != x2Dim0) && (kValX1 == x2Dim1);
}

aclnnStatus allGatherMatmulV2GetWorkspaceSizeAIVMode(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                                   const aclTensor* x1Scale, const aclTensor* x2Scale,
                                                   const aclTensor* quantScale, int64_t blockSize, const char* group,
                                                   int64_t gatherIndex, int64_t commTurn, int64_t streamMode,
                                                   int64_t groupSize, const char* commMode, aclTensor* output, aclTensor* gatherOut,
                                                   aclTensor* amaxOut, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor)
{
    OP_LOGD("allGatherMatmulV2GetWorkspaceSizeAIVMode start");
    bool transposeX1 = IsTransposeLastTwoDims(x1);
    bool viewTransposeX2 = IsViewTransposeX2(x1, x2, transposeX1);
    bool transposeX2 = viewTransposeX2 || IsTransposeLastTwoDims(x2);
    uint32_t rankSize = 0;
    bool isAmaxOut = false;
    bool isGatherOut = IsGatherOut(gatherOut);
    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    auto retParam = CheckParamsAndShapeForAIVMode(x1, x2, bias, output, gatherOut, transposeX1, viewTransposeX2, streamMode);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    // 【A2、A3】校验非连续入参合法性
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
      OP_API_CHECK(!transposeX2 && !MC2Aclnn::IsTensorContiguous(x2), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The x2 without transpose in aclnnAllGatherMatmul must be contiguous, but it is non-contiguous.");
        return ACLNN_ERR_PARAM_INVALID;});
    }
    aclnnStatus ret = aclnnInnerAllGatherMatmulV2GetWorkspaceSize(x1, x2, bias, x1Scale, x2Scale, quantScale, group,
                                                                transposeX1, transposeX2, gatherIndex, commTurn,
                                                                rankSize, blockSize, groupSize, isGatherOut, isAmaxOut,
                                                                yDtype, commMode, output, gatherOut, amaxOut, workspaceSize,
                                                                executor);
    OP_LOGD("allGatherMatmulV2AIVMode, aclnnInnerGetWorkspaceSize ret = %d.", ret);
    return ret;
}

aclnnStatus aclnnAllGatherMatmulV2GetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                                   const aclTensor* x1Scale, const aclTensor* x2Scale,
                                                   const aclTensor* quantScale, int64_t blockSize, const char* group,
                                                   int64_t gatherIndex, int64_t commTurn, int64_t streamMode,
                                                   int64_t groupSize, const char* commMode, aclTensor* output, aclTensor* gatherOut,
                                                   aclTensor* amaxOut, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor)
{
    aclnnStatus ret = ACLNN_SUCCESS;
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        ret = allGatherMatmulV2GetWorkspaceSizeAIVMode(x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, group, gatherIndex, commTurn,
                                                       streamMode, groupSize, commMode, output, gatherOut, amaxOut, workspaceSize, executor);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported npuArch");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ret;
}

aclnnStatus aclnnAllGatherMatmulV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream)
{
  if ((workspace == nullptr) || (workspaceSize == 0UL)) {
    OP_LOGD("Skip the api for empty tensor, workspace size %lu.", workspaceSize);
    return ACLNN_SUCCESS;
  }

  if (NnopbaseSetHcclServerType) {
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
      NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
  }

  return aclnnInnerAllGatherMatmulV2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif