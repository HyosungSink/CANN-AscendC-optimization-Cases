/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GROUPED_MATMUL_FINALIZE_ROUTING_UTIL_H
#define OP_API_INC_GROUPED_MATMUL_FINALIZE_ROUTING_UTIL_H
#include "opdev/common_types.h"

namespace GmmFinalizeRouting {
using namespace op;
struct GroupedMatmulFinalizeRoutingParams {
    const aclTensor *x1 = nullptr;
    const aclTensor *x2 = nullptr;
    const aclTensor *scale1Optional = nullptr;
    const aclTensor *scale2 = nullptr;
    const aclTensor *groupList = nullptr;
    aclTensor *yRef = nullptr;
    int64_t groupListType = 0;
    int64_t groupSize = 0;
};

struct GroupedMatmulParams {
    // mandatory
    const aclTensor *x1{nullptr};
    const aclTensor *x2{nullptr};
    const aclTensor *out{nullptr};
    // optional
    const aclTensor *scale{nullptr};
    const aclTensor *bias{nullptr};
    const aclTensor *pertokenScaleOptional{nullptr};
    const aclTensor *groupList{nullptr};
    const aclTensor *shareInput{nullptr};
    const aclTensor *logit{nullptr};
    const aclTensor *rowIndex{nullptr};
    const aclTensor *offset{nullptr};
    const aclIntArray *tuningConfig{nullptr};
    // numbers
    float shareInputWeight{0.0f};
    int64_t shareInputOffset{0};
    int64_t groupListType{0};
    // attrs
    bool transposeX1{false};
    bool transposeX2{false};
};

class GroupedMatmulParamsBuilder {
public:
    static GroupedMatmulParamsBuilder Create(const aclTensor *x1, const aclTensor *x2, const aclTensor *out)
    {
        GroupedMatmulParamsBuilder b;
        b.p_.x1 = x1;
        b.p_.x2 = x2;
        b.p_.out = out;
        return b;
    }

    GroupedMatmulParamsBuilder &SetScale(const aclTensor *scale)
    {
        p_.scale = scale;
        return *this;
    }

    GroupedMatmulParamsBuilder &SetBias(const aclTensor *bias)
    {
        p_.bias = bias;
        return *this;
    }

    GroupedMatmulParamsBuilder &SetPertokenScale(const aclTensor *pertoken)
    {
        p_.pertokenScaleOptional = pertoken;
        return *this;
    }

    GroupedMatmulParamsBuilder &SetGroupList(const aclTensor *groupList)
    {
        p_.groupList = groupList;
        return *this;
    }

    GroupedMatmulParamsBuilder &SetShareInput(const aclTensor *shareInput)
    {
        p_.shareInput = shareInput;
        return *this;
    }

    GroupedMatmulParamsBuilder &SetLogit(const aclTensor *logit)
    {
        p_.logit = logit;
        return *this;
    }

    GroupedMatmulParamsBuilder &SetRowIndex(const aclTensor *rowIndex)
    {
        p_.rowIndex = rowIndex;
        return *this;
    }

    GroupedMatmulParamsBuilder &SetOffset(const aclTensor *offset)
    {
        p_.offset = offset;
        return *this;
    }

    GroupedMatmulParamsBuilder &SetTuningConfig(const aclIntArray *tuningConfig)
    {
        p_.tuningConfig = tuningConfig;
        return *this;
    }

    GroupedMatmulParamsBuilder &SetNumbers(float shareInputWeight, int64_t shareInputOffset, int64_t groupListType)
    {
        p_.shareInputWeight = shareInputWeight;
        p_.shareInputOffset = shareInputOffset;
        p_.groupListType = groupListType;
        return *this;
    }

    GroupedMatmulParamsBuilder &SetTranspose(bool transposeX1, bool transposeX2)
    {
        p_.transposeX1 = transposeX1;
        p_.transposeX2 = transposeX2;
        return *this;
    }

    GroupedMatmulParams Build() const
    {
        return p_;
    }

private:
    GroupedMatmulParams p_;
};

} // namespace GmmFinalizeRouting
#endif