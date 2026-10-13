/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_nsa_compress.h"

#include <dlfcn.h>
#include <new>

#include "aclnn_kernels/transdata.h"
#include "nsa_compress.h"
#include "aclnn_kernels/contiguous.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static const uint64_t DIM_NUM_2 = 2;
static const uint64_t DIM_NUM_1 = 1;
static const uint64_t DIM_NUM_0 = 0;

enum class InputLayout : uint32_t {
    BSH = 0,
    SBH = 1,
    BNSD = 2,
    BSND = 3,
    TND
};

struct CompressShapeInfo {
    InputLayout inputLayout;
    string l0InputLayoutStr;

    int64_t headDim = 0;
    int64_t headNum = 0;
    int64_t compressBlockSize = 0;
    int64_t compressStride = 0;
    int64_t actSeqLenType = 0;
};

static aclnnStatus CheckNsaCompressParam(const aclTensor *input, const aclTensor *weight, const aclTensor *output,
                                         const uint64_t *workspaceSize, aclOpExecutor ** const executor)
{
    // 必须的参数指针判空
    CHECK_RET(input != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(weight != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(output != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus InputDtypeCheck(const aclTensor *input, const aclTensor *weight, const aclTensor *output)
{
    auto inputDtype = input->GetDataType();
    auto weightDtype = weight->GetDataType();
    auto outputDtype = output->GetDataType();
    if (inputDtype != weightDtype || inputDtype != outputDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of input[%s], weight[%s], output[%s] are not equal.",
                op::ToString(DataType(inputDtype)).GetString(), op::ToString(DataType(weightDtype)).GetString(),
                op::ToString(DataType(outputDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus Contiguous(const aclTensor *&input, const aclTensor *&weight, aclOpExecutor *executor)
{
    input = l0op::Contiguous(input, executor);
    CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);
    weight = l0op::Contiguous(weight, executor);
    CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus AnalysisAxis(const aclTensor *input, const aclTensor *weight, char *layoutOptional,
                                int64_t compressBlockSize, int64_t compressStride, CompressShapeInfo &shapeInfo)
{
    Shape inputShape = input->GetViewShape();
    Shape weightShape = weight->GetViewShape();

    shapeInfo.compressBlockSize = compressBlockSize;
    shapeInfo.compressStride = compressStride;

    if (inputShape[DIM_NUM_1] <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Head Num must > 0, but got %ld", inputShape[DIM_NUM_1]);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (inputShape[DIM_NUM_2] <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Head Dim must > 0, but got %ld", inputShape[DIM_NUM_2]);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 检查weight的shape(compressBlockSize, N), weight与input的shape满足broadcast关系
    if (weightShape[DIM_NUM_0] != compressBlockSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "weight.shape[0] and compressBlockSize should be same, but got weight.shape[0]=%ld "
                "and compressBlockSize=%ld",
                weightShape[DIM_NUM_0], compressBlockSize);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (weightShape[DIM_NUM_1] != inputShape[DIM_NUM_1]) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "weight.shape[1] and headNum should be same, but got weight.shape[1]=%ld "
                "and headNum=%ld",
                weightShape[DIM_NUM_1], inputShape[DIM_NUM_1]);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // layoutOptional值检查
    std::string layoutStr = op::ToString(layoutOptional).GetString();
    if (layoutStr == "TND") {
        shapeInfo.inputLayout = InputLayout::TND;
        shapeInfo.l0InputLayoutStr = "TND";
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "not support layout %s", layoutOptional);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus AnalysisInput(const aclTensor *input, const aclTensor *weight, char *layoutOptional,
                                 int64_t compressBlockSize, int64_t compressStride, int64_t actSeqLenType,
                                 CompressShapeInfo &shapeInfo, const aclIntArray *actSeqLenOptional = nullptr)
{
    // 从shape中取需要的值到shapeInfo
    CHECK_RET(AnalysisAxis(input, weight, layoutOptional, compressBlockSize, compressStride, shapeInfo) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    if (actSeqLenOptional == nullptr || actSeqLenOptional->Size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "actSeqLenOptional is not currently support null");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (actSeqLenType != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "actSeqLenType currently only support 0");
        return ACLNN_ERR_PARAM_INVALID;
    }

    OP_LOGD("Analysis input success.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNsaCompressGetWorkspaceSize(const aclTensor *input, const aclTensor *weight,
                                             const aclIntArray *actSeqLenOptional, char *layoutOptional,
                                             int64_t compressBlockSize, int64_t compressStride, int64_t actSeqLenType,
                                             aclTensor *output, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 检查入参
    CHECK_RET(CheckNsaCompressParam(input, weight, output, workspaceSize, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    // L2接口阶段1
    L2_DFX_PHASE_1(
        aclnnNsaCompress,
        DFX_IN(input, weight, actSeqLenOptional, layoutOptional, compressBlockSize, compressStride, actSeqLenType),
        DFX_OUT(output));

    CHECK_RET(InputDtypeCheck(input, weight, output) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CompressShapeInfo shapeInfo;
    CHECK_RET(AnalysisInput(input, weight, layoutOptional, compressBlockSize, compressStride, actSeqLenType, shapeInfo,
                            actSeqLenOptional) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    // 获取executor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    CHECK_RET(Contiguous(input, weight, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // 调用L0接口
    auto l0NsaCompressOuts = l0op::NsaCompress(input, weight, actSeqLenOptional, layoutOptional, compressBlockSize,
                                               compressStride, actSeqLenType, l0Executor);

    // 检查输出
    CHECK_RET(l0NsaCompressOuts != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(l0NsaCompressOuts, output, l0Executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNsaCompress(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    // 固定写法
    L2_DFX_PHASE_2(aclnnNsaCompress);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in NsaCompress launch aicore");
    return ACLNN_SUCCESS;
}
} // namespace

#ifdef __cplusplus
}
#endif
