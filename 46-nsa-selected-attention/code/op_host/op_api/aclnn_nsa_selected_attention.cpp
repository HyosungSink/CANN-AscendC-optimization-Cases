/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "nsa_selected_attention.h"
#include "aclnn_nsa_selected_attention.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const uint64_t DIM_NUM_4 = 4;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_2 = 2;
static const int64_t HEAD_DIM_MAX = 768;

struct AxesInfo {
    int64_t b;
    int64_t n1;
    int64_t n2;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t dk;
    int64_t dv;
};

enum class InputLayout : uint32_t {
    BSND,
    SBH,
    BNSD,
    BSH,
    TND
};

struct NSAShapeInfo {
    AxesInfo axes;

    InputLayout inputLayout;
    string l0InputLayoutStr;

    uint64_t dimNum = 0;
    uint64_t padNum = 0;

    FVector<int64_t, DIM_NUM_4> perm_in;
    FVector<int64_t, DIM_NUM_4> perm_out;
    FVector<int64_t, DIM_NUM_4> reshapedQueryShape;
    FVector<int64_t, DIM_NUM_4> reshapedKeyValueShape;
    FVector<int64_t, DIM_NUM_4> reshapedValueBefore;
    FVector<int64_t, DIM_NUM_3> reshapedValueAfter;
    FVector<int64_t, DIM_NUM_4> reshapedTopKIndicesShape;
    FVector<int64_t, DIM_NUM_4> reshapedOutBefore;
    FVector<int64_t, DIM_NUM_3> reshapedOutAfter;

    bool needPad = false;
    bool needTranspose = false;
    bool needReshape = false;
    bool needPadValue = false;
};

void AnalysisAxisForTnd(const Shape &qShape, const Shape &kShape, const Shape &vShape, NSAShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::TND;
    shapeInfo.l0InputLayoutStr = "TND";
    shapeInfo.axes.n1 = qShape[1];
    shapeInfo.axes.n2 = kShape[1];
    shapeInfo.axes.d = qShape[DIM_NUM_2];
    shapeInfo.axes.dk = kShape[DIM_NUM_2];
    shapeInfo.axes.dv = vShape[DIM_NUM_2];
}

aclnnStatus AnalysisAxis(const aclTensor *query, const aclTensor *key, const aclTensor *value, 
                         const char *inputLayout, NSAShapeInfo &shapeInfo)
{
    Shape qShape = query->GetViewShape();
    Shape kShape = key->GetViewShape();
    Shape vShape = value->GetViewShape();
    shapeInfo.dimNum = qShape.GetDimNum();

    // 记录轴的长度 b, n2, g, s1, s2, d
    // H1等于N1*D, H2等于N2*D
    // N1等于g*N2
    std::string inputLayoutStr = op::ToString(inputLayout).GetString();
    if (shapeInfo.dimNum == DIM_NUM_3 && inputLayoutStr == "TND") {
        // query: (T,N1,D)
        // key/value: (T,N2,D)
        AnalysisAxisForTnd(qShape, kShape, vShape, shapeInfo);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "not support input_layout %s with dim_num %lu", inputLayout, shapeInfo.dimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (shapeInfo.axes.d != shapeInfo.axes.dk) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qD and kD should be same, but got qD=%ld kD=%ld", shapeInfo.axes.d, 
            shapeInfo.axes.dk);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (shapeInfo.axes.d < shapeInfo.axes.dv) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support kD >= vD, but got kD=%ld vD=%ld", shapeInfo.axes.d, 
            shapeInfo.axes.dv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto qDim = qShape.GetDimNum();
    auto kDim = kShape.GetDimNum();
    auto vDim = vShape.GetDimNum();
    if (qDim != DIM_NUM_3 || kDim != DIM_NUM_3 || vDim != DIM_NUM_3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "input shape error: expected all shapes to be %dD, but got qDim=%ld, kDim=%ld, vDim=%ld",
                DIM_NUM_3, qDim, kDim, vDim);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qShape[0] == 0 || kShape[0] == 0 || shapeInfo.axes.n1 == 0 || shapeInfo.axes.n2 == 0 ||shapeInfo.axes.d == 0 || shapeInfo.axes.dv == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,  "input shape error, got 0 in Tq Tkv Nq Nkv Dqk Dv (%ld, %ld, %ld, %ld, %ld, %ld)", 
            qShape[0], kShape[0], shapeInfo.axes.n1, shapeInfo.axes.n2, shapeInfo.axes.d, shapeInfo.axes.dv);
            return ACLNN_ERR_PARAM_INVALID;
        }
    return ACLNN_SUCCESS;
}

aclnnStatus InputDtypeCheck(const aclTensor *query, const aclTensor *key, const aclTensor *value)
{
    auto vDtype = value->GetDataType();
    auto kDtype = key->GetDataType();
    auto qDtype = query->GetDataType();
    if (qDtype != kDtype || kDtype != vDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of query[%s], key[%s], value[%s] are not equal.",
                op::ToString(DataType(qDtype)).GetString(), op::ToString(DataType(kDtype)).GetString(),
                op::ToString(DataType(vDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AnalysisInput(const aclTensor *query, const aclTensor *key, const aclTensor *value, 
                          [[maybe_unused]] const aclTensor *topkIndices, char *inputLayout, NSAShapeInfo &shapeInfo, 
                          const aclIntArray *actualSeqQLenOptional = nullptr,
                          const aclIntArray *actualSeqKvLenOptional = nullptr)
{
    CHECK_RET(
        AnalysisAxis(query, key, value, inputLayout, shapeInfo) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    if (shapeInfo.axes.d > HEAD_DIM_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Head dim must <= 768, but got %ld", shapeInfo.axes.d);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if ((actualSeqQLenOptional != nullptr && actualSeqKvLenOptional != nullptr) && 
                    (actualSeqQLenOptional->Size() > 3 * 1024 || actualSeqKvLenOptional->Size() > 3 * 1024)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
                "actualSeqQLen size and actualSeqKvLen size must <= 3072, but got %lu ", actualSeqQLenOptional->Size());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (shapeInfo.axes.n2 == 0 || shapeInfo.axes.d == 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (shapeInfo.inputLayout != InputLayout::TND &&
        (shapeInfo.axes.b == 0 || shapeInfo.axes.s1 == 0 || shapeInfo.axes.s2 == 0)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    OP_LOGD("Analysis input success. Analysis result: " 
            "[needReshape]: %d, [needPad]: %d, [padNum]: %lu,"
            "[needTranspose]: %d.",
            shapeInfo.needReshape, shapeInfo.needPad, shapeInfo.padNum, shapeInfo.needTranspose);
    return ACLNN_SUCCESS;
}

static inline const aclTensor *GeneratePaddings(int32_t dimNum, int32_t padNum, aclOpExecutor *executor)
{
    // 2代表每根轴的前后都可以补0
    FVector<int64_t> padVec(dimNum * 2, 0);
    padVec[padVec.size() - 1] = padNum;

    auto padArray = executor->AllocIntArray(padVec.data(), padVec.size());
    if (padArray == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc padVec failed");
        return nullptr;
    }

    auto padTensor = executor->ConvertToTensor(padArray, DataType::DT_INT64);
    return padTensor;
}

aclnnStatus Contiguous(const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
                       const aclTensor *&topkIndices, const aclTensor *&attenMaskOptional,
                       aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    topkIndices = l0op::Contiguous(topkIndices, executor);
    CHECK_RET(topkIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (attenMaskOptional) {
        attenMaskOptional = l0op::Contiguous(attenMaskOptional, executor);
        CHECK_RET(attenMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus PreprocessQKV(const aclTensor *&value,
                          const struct NSAShapeInfo &shapeInfo, aclOpExecutor *executor)
{
    if (shapeInfo.needPadValue) {
        int32_t dimNum = (shapeInfo.inputLayout == InputLayout::TND) ? DIM_NUM_3 : DIM_NUM_4;
        auto paddings = GeneratePaddings(dimNum, shapeInfo.axes.d - shapeInfo.axes.dv, executor);
        value = l0op::Pad(value, paddings, executor);
        CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);

    }
    return ACLNN_SUCCESS;
}


aclnnStatus CheckNsaParam(const aclTensor *query, const aclTensor *key, const aclTensor *value, 
    const aclTensor *topkIndices, const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional, 
    const char *inputLayout, const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, 
    const aclTensor *attentionOut, const uint64_t *workspaceSize, aclOpExecutor ** const executor)
{
    // 必须的参数指针判空
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(topkIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(actualSeqQLenOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(actualSeqKvLenOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(inputLayout != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(softmaxMaxOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(softmaxSumOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}


aclnnStatus aclnnNsaSelectedAttentionGetWorkspaceSize(
    const aclTensor *query, 
    const aclTensor *key, 
    const aclTensor *value, 
    const aclTensor *topkIndices, 
    const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, 
    double scaleValue, 
    int64_t headNum,
    char *inputLayout, 
    int64_t sparseMode, 
    int64_t selectedBlockSize,
    int64_t selectedBlockCount, 
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut, 
    const aclTensor *attentionOut,
    uint64_t *workspaceSize, 
    aclOpExecutor **executor)
{
    CHECK_RET(CheckNsaParam(query, key, value, topkIndices, actualSeqQLenOptional, actualSeqKvLenOptional, 
    inputLayout, softmaxMaxOut, softmaxSumOut, attentionOut, 
    workspaceSize, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    L2_DFX_PHASE_1(aclnnNsaSelectedAttention,
                   DFX_IN(query, key, value, topkIndices, attenMaskOptional, actualSeqQLenOptional, 
                          actualSeqKvLenOptional, scaleValue, headNum, inputLayout,
                          sparseMode, selectedBlockSize, selectedBlockCount),
                   DFX_OUT(softmaxMaxOut, softmaxSumOut, attentionOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // b, n1, s1 为0时，不进行任何处理
    // n2, s2, d 为0时，直接调用l0接口处理
    if (softmaxMaxOut->IsEmpty() && softmaxSumOut->IsEmpty() && attentionOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Layout %s is not TND, invalid shape, please check", inputLayout);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_INVALID;
    }
    NSAShapeInfo shapeInfo;
    CHECK_RET(AnalysisInput(query, key, value, topkIndices, inputLayout, 
                            shapeInfo, actualSeqQLenOptional,
                            actualSeqKvLenOptional) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    CHECK_RET(Contiguous(query, key, value, topkIndices, attenMaskOptional, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(PreprocessQKV(value, shapeInfo, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

        auto l0NsaSelectedAttentionOuts = l0op::NsaSelectedAttention(
        query, key, value, topkIndices, attenMaskOptional, actualSeqQLenOptional, 
                          actualSeqKvLenOptional, scaleValue, headNum, shapeInfo.l0InputLayoutStr.c_str(),
                          sparseMode, selectedBlockSize, selectedBlockCount, l0Executor);

    CHECK_RET(l0NsaSelectedAttentionOuts[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0NsaSelectedAttentionOuts[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0NsaSelectedAttentionOuts[2] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto l0SoftmaxMaxOut = l0NsaSelectedAttentionOuts[0];
    auto l0SoftmaxSumOut = l0NsaSelectedAttentionOuts[1];
    // l0SoftmaxOutOut not used now
    auto l0attentionOut = l0NsaSelectedAttentionOuts[2];

    if (l0SoftmaxMaxOut == nullptr || l0SoftmaxSumOut == nullptr || l0attentionOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "l0SoftmaxMaxOut or l0SoftmaxSumOut or l0attentionOut is null");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto viewCopyResult0 = l0op::ViewCopy(l0SoftmaxMaxOut, softmaxMaxOut, l0Executor);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(l0SoftmaxSumOut, softmaxSumOut, l0Executor);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // l0SoftmaxOutOut not used now
    auto viewCopyResult2 = l0op::ViewCopy(l0attentionOut, attentionOut, l0Executor);
    CHECK_RET(viewCopyResult2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNsaSelectedAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNsaSelectedAttention);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
}  // namespace

#ifdef __cplusplus
}
#endif
