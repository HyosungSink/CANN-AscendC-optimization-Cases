/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_SRC_QUANT_MATMUL_CHECKER_H_
#define OP_API_SRC_QUANT_MATMUL_CHECKER_H_

#include <map>
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using TupleInput = std::tuple<const aclTensor *, const aclTensor *>;
using TupleQuant = std::tuple<const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
                              const aclTensor *, const aclTensor *, const aclTensor *, const int64_t &, const int64_t &>;
using TupleAttr = std::tuple<bool, bool>;

namespace op {
class QuantMatmulChecker {
public:
    QuantMatmulChecker(const TupleInput &inputTensors, const TupleQuant &quantTensors,
                       const TupleAttr &boolsTrans, const aclTensor *out, const bool isWeightNz = false)
     : inputTensors_(inputTensors), quantTensors_(quantTensors), boolsTrans_(boolsTrans), out_(out), isWeightNz_(isWeightNz) {}

    ~QuantMatmulChecker() = default;
    void Init();
    aclnnStatus CheckParams() const;
    bool IsA4W4PergroupNonSymmetric(const uint64_t groupSizeK) const;
    bool InferGroupSize(int64_t &groupSize);

private:
    void GetX1X2DimValue();
    bool CheckDtype4WeightNz() const;
    aclnnStatus CheckDtype() const;
    aclnnStatus CheckDtypeL0c2outOrL0c2ub() const;
    bool CheckDoubleScaleAndFp8Hif8PertokenPerblock() const;
    bool CheckL0c2outOrL0c2ubPertoken() const;
    bool CheckMicroScaling() const;
    bool CheckL0c2outOrL0c2ubPertensorPerchannel() const;
    bool CheckL0c2outOrL0c2ubPertensorPerchannel4Int8Input() const;
    aclnnStatus CheckDtypeOnlyL0c2out() const;
    aclnnStatus CheckDtypeOnlyL0c2ub() const;
    bool CheckOnlyL0c2outPertoken() const;
    bool CheckL0c2outAndL0c2ubPertensorPerchannel() const;
    bool CheckL0c2outPertensorPerchannel() const;
    bool CheckOnlyL0c2ubPertoken() const;
    bool CheckFormat() const;
    bool CheckShape() const;
    bool CheckScaleDimRange() const;
    bool CheckGroupSize() const;
    bool CheckOutShape(bool twoDimMatmulCaseFlag, const std::vector<int64_t> &batchRecord) const;
    bool CheckBiasShape(const std::vector<int64_t> &batchRecord, int64_t inferedOutbatchValue) const;
    int64_t InferOutputShape(std::vector<int64_t> &batchRecord) const;
    bool CheckDimValue() const;
    bool CheckDimValuePertokenDoubleScale() const;
    bool CheckDimValuePerblock() const;
    bool CheckGroupSizeShape(uint64_t groupSizeM) const;
    bool CheckKDimValueFp4Fp8WeightNZMicroScaling() const;
    bool CheckDimValueMicroScaling() const;
    bool CheckShapeForWeightNz() const;
    bool CheckShapeInt4() const;
    bool CheckDtypeValidOnOnlyL0c2outForA4W4() const;
    bool CheckFormatInt4() const;
    bool CheckEmptyTensor() const;
    std::string GetX1ScaleName() const;
    std::string GetX2ScaleName() const;
    std::string GetX2OffsetName() const;
    bool ReCalcGroupSize(int64_t inputSize, int64_t scaleSize, int64_t &groupSize, const char *dimensionName) const;

public:
    const TupleInput inputTensors_;
    const TupleQuant quantTensors_;
    const TupleAttr boolsTrans_;
    const aclTensor *out_ = nullptr;
    const bool isWeightNz_;

private:
    const aclTensor *x1_ = nullptr;
    const aclTensor *x2_ = nullptr;
    const aclTensor *x1Scale_ = nullptr;
    const aclTensor *x2Scale_ = nullptr;
    const aclTensor *yScale_ = nullptr;
    const aclTensor *x1Offset_ = nullptr;
    const aclTensor *x2Offset_ = nullptr;
    const aclTensor *yOffset_ = nullptr;
    const aclTensor *bias_ = nullptr;
    bool transposeX1_ = false;
    bool transposeX2_ = false;
    bool isA4W4_ = false;
    mutable uint64_t groupSize_ = 0;
    int64_t interfaceType_ = 5;
    DataType outType_;
    DataType x2ScaleType_;
    int64_t x1KDim_ = 0;
    int64_t x1MDim_ = 0;
    int64_t x2KDim_ = 0;
    int64_t x2NDim_ = 0;
    SocVersion socVersion_;
    NpuArch npuArch_;
};
}

#endif  // OP_API_SRC_QUANT_MATMUL_CHECKER_H_