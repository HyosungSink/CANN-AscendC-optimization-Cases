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
 * \file fused_floyd_attention_case.cpp
 * \brief FusedFloydAttention 测试用例.
 */
#include "fused_floyd_attention_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

#define FUSED_FLOYD_ATTENTION_KERNEL_PARAM                                                        \
    (uint8_t *query, uint8_t *key_1, uint8_t *value_1, uint8_t *key_2, uint8_t * value_2, \
     uint8_t *attenMask, uint8_t *softmaxMax, uint8_t *softmaxSum, uint8_t *attentionOut,     \
     uint8_t *workspace, uint8_t *tiling)

using FusedFloydAttentionKernelFunc = void(*) FUSED_FLOYD_ATTENTION_KERNEL_PARAM;

extern "C" __global__ __aicore__ void fused_floyd_attention_fp16 FUSED_FLOYD_ATTENTION_KERNEL_PARAM;

using namespace ops::adv::tests::FusedFloydAttention;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;
using FusedFloydAttentionCase = ops::adv::tests::FusedFloydAttention::FusedFloydAttentionCase;

namespace {
const uint32_t MAX_TILING_DATA_SIZE = 29540;  // 2854
}

bool RunFusedFloydAttention(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                         std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (FusedFloydAttentionKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim,
                inputs[0]->GetDevData(),      // query
                inputs[1]->GetDevData(),      // key1
                inputs[2]->GetDevData(),      // value1
                inputs[3]->GetDevData(),      // key2
                inputs[4]->GetDevData(),      // value2
                inputs[5]->GetDevData(),      // attenMask
                outputs[0]->GetDevData(),     // softmaxMax
                outputs[1]->GetDevData(),     // softmaxSum
                outputs[2]->GetDevData(),     // attentionOut
                workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingForFusedFloydAttentionStub(gert::TilingContext *context)
{
    auto *fusedFloydAttentionCase = static_cast<FusedFloydAttentionCase *>(Case::GetCurrentCase());
    if (fusedFloydAttentionCase != nullptr) {
        if(context == nullptr){
            LOG_ERR("context is null");
            return ge::GRAPH_FAILED;
        }
        return fusedFloydAttentionCase->fusedFloydAttentionTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}


bool FusedFloydAttentionCase::InitParam()
{
    std::vector<int64_t> layoutQ;
    std::vector<int64_t> layoutK;
    std::vector<int64_t> layoutK1;
    std::vector<int64_t> layoutV;
    std::vector<int64_t> layoutV1;
    std::vector<int64_t> layoutAttenOut;
    std::vector<int64_t> layoutAttenMask;
    std::vector<int64_t> layoutSoftmax;
    std::string qkvStr = "BHNMD";
    std::string attenMaskStr = "B_1_N_1_K";
    std::string layoutSoftmaxStr = "BNNM8";  // T_N1_8

    layoutQ = {mParam.B, mParam.H, mParam.N, mParam.M, mParam.D};
    layoutK = {mParam.B, mParam.H, mParam.N, mParam.K, mParam.D};
    layoutK1 = {mParam.B, mParam.H, mParam.K, mParam.M, mParam.D};
    layoutV = {mParam.B, mParam.H, mParam.N, mParam.K, mParam.D};
    layoutV1 = {mParam.B, mParam.H, mParam.K, mParam.M, mParam.D};
    layoutAttenMask = {mParam.B, 1, mParam.N, 1, mParam.K};
    layoutAttenOut = {mParam.B, mParam.H, mParam.N, mParam.M, mParam.D};
    layoutSoftmax = {mParam.B, mParam.H, mParam.N, mParam.M, 8};

    query = Tensor("query", layoutQ, qkvStr.c_str(), mParam.dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    key1 = Tensor("key1", layoutK, qkvStr.c_str(), mParam.dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    key2 = Tensor("key2", layoutK1, qkvStr.c_str(), mParam.dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    value1 = Tensor("value1", layoutV, qkvStr.c_str(), mParam.dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    value2 = Tensor("value2", layoutV1, qkvStr.c_str(), mParam.dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);

    attenMask = Tensor("attenMask", layoutAttenMask, attenMaskStr.c_str(), ge::DataType::DT_UINT8, ge::FORMAT_ND);
    attenOut = Tensor("attenOut", layoutAttenOut, qkvStr.c_str(), ge::DataType::DT_FLOAT, ge::FORMAT_ND, 
                        Tensor::TensorType::REQUIRED_OUTPUT);
    softmaxMax = Tensor("softmaxMax", layoutSoftmax, layoutSoftmaxStr.c_str(), ge::DataType::DT_FLOAT, ge::FORMAT_ND,
                        Tensor::TensorType::REQUIRED_OUTPUT);
    softmaxSum = Tensor("softmaxSum", layoutSoftmax, layoutSoftmaxStr.c_str(), ge::DataType::DT_FLOAT, ge::FORMAT_ND,
                        Tensor::TensorType::REQUIRED_OUTPUT);
    return true;
}

bool FusedFloydAttentionCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("FusedFloydAttention");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&query, &key, &value, &key1, &value1, &attenMask});
    rst = rst && mCtx.SetOutputs({&softmaxMax, &softmaxSum, &attenOut});
    rst = rst && mCtx.SetAttrs({{"scale_value", mParam.scale}});
    rst = rst && mCtx.SetTilingDataMaxSize(MAX_TILING_DATA_SIZE);
    rst = rst && mCtx.SetKernelRunCbf(RunFusedFloydAttention);
    rst = rst && mCtx.SetKernelMainFunc((void *)fused_floyd_attention_fp16);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    fusedFloydAttentionTilingFunc = (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingFusedFloydAttention");
    if (fusedFloydAttentionTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, fusedFloydAttentionTilingFunc(%p)", fusedFloydAttentionTilingFunc);
        return false;
    }
    IMPL_OP(FusedFloydAttention).Tiling(TilingForFusedFloydAttentionStub);
    return rst;
}

bool FusedFloydAttentionCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool FusedFloydAttentionCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mOpInfo.ProcessTiling(mName)) {
        return false;
    }
    if (!mOpInfo.ProcessKernel(mName)) {
        return false;
    }
    return true;
}


FusedFloydAttentionCase::FusedFloydAttentionCase()
{
    this->mOpInfo.mName = "FusedFloydAttention";
}


FusedFloydAttentionCase::Param::Param(int64_t pB, int64_t pH, int64_t pN, int64_t pM, int64_t pK, int64_t pD, ge::DataType pDtype, float pScale)
    : B(pB), H(pH), N(pN), M(pM), K(pK), D(pD), dtype(pDtype), scale(pScale)
{
}
