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
 * \file fused_floyd_attention_case.h
 * \brief FusedFloydAttention 测试用例.
 */

#pragma once
#include <vector>
#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>
#include "graph/types.h"
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "tests/utils/tensor.h"
#include "tests/utils/tensor_list.h"

namespace ops::adv::tests::FusedFloydAttention {
class FusedFloydAttentionCase : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using Tensor = ops::adv::tests::utils::Tensor;

public:
    class Param {
    public:
        /* 设置参数 */
        int64_t B = 0;
        int64_t H = 0;
        int64_t N = 0;
        int64_t M = 0;
        int64_t K = 0;
        int64_t D = 0;
        ge::DataType dtype = ge::DataType::DT_UNDEFINED;
        float scale = 1.0f;

        Param() = default;
        Param(int64_t pB, int64_t pH, int64_t pN, int64_t pM, int64_t pK, int64_t pD, 
                ge::DataType pDtype, float pScale);
    };

    Tensor query;
    Tensor key;
    Tensor key1;
    Tensor value1;
    Tensor value2;
    Tensor attenMask;
    Tensor softmaxMax;
    Tensor softmaxSum;
    Tensor attenOut;

    OpInfo mOpInfo;
    Context mCtx;
    Param mParam;
    gert::OpImplRegisterV2::TilingKernelFunc fusedFloydAttentionTilingFunc = nullptr;

    FusedFloydAttentionCase();
    bool Run() override;
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
};

} // namespace ops::adv::tests::FusedFloydAttention