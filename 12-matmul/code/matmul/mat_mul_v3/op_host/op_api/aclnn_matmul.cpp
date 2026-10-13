/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_matmul.cpp
 * \brief
 */

#include "aclnn_matmul.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h" 
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/dot.h"
#include "level0/fill.h"
#include "matmul/common/op_host/op_api/matmul.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "level0/unsqueeze.h"

#include "util/math_util.h"
#include "matmul/common/op_host/op_api/cube_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "matmul/common/op_host/op_api/batch_matmul_util.h"
#include "op_api/op_api_def.h"

using Ops::Base::CeilDiv;
using namespace Ops::NN;
using namespace op;

namespace {
static constexpr size_t DIMS_TWO = 2;
static constexpr size_t DIMS_FOUR = 4;
static constexpr size_t LAST_SECOND_DIM_INDEX = 2;
static constexpr size_t LAST_FIRST_DIM_INDEX = 1;
static const int NZ_K0_VALUE_16 = 16;
static const int NZ_K0_VALUE_32 = 8;
static const int NZ_STORAGE_PENULTIMATE_DIM = 16;
static const size_t MAX_SUPPORT_MATMUL_DIMS_NUMS = 6;
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_WITHOUT_BF16 = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16};
inline static bool CheckNotNull(const aclTensor* self, const aclTensor* mat2, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(mat2, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckNpuArchIsSupportBf16(void)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    return (npuArch == NpuArch::DAV_2201) || (npuArch == NpuArch::DAV_3510);
}

static inline bool CheckMathType(const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType)
{
    bool selfFloat = self->GetDataType() == DataType::DT_FLOAT;
    bool mat2Float = mat2->GetDataType() == DataType::DT_FLOAT;
    auto promoteType = selfFloat || mat2Float ? DataType::DT_FLOAT : self->GetDataType();
    return CheckCubeMathTypeForMm(promoteType, cubeMathType);
}

static bool CheckWeightNzDtype(const aclTensor* self, const aclTensor* mat2)
{
    if (mat2->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
        // weightnz暂不支持self,mat2为bfloat16和float16的数据类型推导
        if ((self->GetDataType() == op::DataType::DT_FLOAT16 && mat2->GetDataType() == op::DataType::DT_BF16) ||
            (self->GetDataType() == op::DataType::DT_BF16 && mat2->GetDataType() == op::DataType::DT_FLOAT16)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "The dtypes of self and mat2 must be both DT_BF16 or both DT_FLOAT16, now self's dtype is [%s], mat2's dtype is [%s]",
                op::ToString(self->GetDataType()).GetString(), op::ToString(mat2->GetDataType()).GetString());
            return false;
        }
    }
    return true;
}

inline static bool CheckWeightNzDtypeValid(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* out, int8_t cubeMathType)
{
    bool bf16flag = CheckNpuArchIsSupportBf16();
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    auto dtypeList = bf16flag ? DTYPE_SUPPORT_LIST : DTYPE_SUPPORT_LIST_WITHOUT_BF16;
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(mat2, dtypeList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeList, return false);
    if (!bf16flag && (self->GetDataType() == op::DataType::DT_BF16 || mat2->GetDataType() == op::DataType::DT_BF16 ||
                      out->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Bfloat16 is unsupported by the current SOC version [%s], now self is %s, mat2 is %s, out is %s",
            op::ToString(socVersion).GetString(), op::ToString(self->GetDataType()).GetString(),
            op::ToString(mat2->GetDataType()).GetString(), op::ToString(out->GetDataType()).GetString());
        return false;
    }

    // keeptype模式支持类型检查
    if (cubeMathType == KEEP_DTYPE && !IsInputSupportFp32() &&
        (self->GetDataType() == DataType::DT_FLOAT || mat2->GetDataType() == DataType::DT_FLOAT)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self dtype %s or mat2 dtype %s not support under keep type mode.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(mat2->GetDataType()).GetString());
        return false;
    }
    if (cubeMathType == KEEP_DTYPE && out->GetDataType() == op::DataType::DT_FLOAT16 &&
        self->GetDataType() == op::DataType::DT_FLOAT) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Input tensor's dtype[DT_FLOAT] should be same with output's dtype[DT_FLOAT16].");
        return false;
    }
    return CheckWeightNzDtype(self, mat2);
}

// 获取broadcast shape
inline static op::Shape GetBroadcastShape(const aclTensor* tensor)
{
    op::Shape shape;
    size_t dimNum = tensor->GetViewShape().GetDimNum();
    size_t loopDims = dimNum - 2; // the dims except the last two
    for (size_t idx = 0; idx < loopDims; idx++) {
        int64_t tmpVal = tensor->GetViewShape().GetDim(idx);
        shape.AppendDim(tmpVal);
    }
    if (shape.GetDimNum() == 0) {
        shape.AppendDim(1);
    }
    return shape;
}

static bool CheckShapeValid(const aclTensor* self, const aclTensor* mat2)
{
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    auto dimTensor1 = selfShape.GetDimNum();
    auto dimTensor2 = mat2Shape.GetDimNum();
    int64_t selfKDim = 0;
    int64_t mat2KDim = 0;

    // 超出最大支持维度返回
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_MATMUL_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(mat2, MAX_SUPPORT_MATMUL_DIMS_NUMS, return false);

    // Tensor1 dims number is 0 OR error dims number is 0
    if (dimTensor1 == 0 || dimTensor2 == 0) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Matmul not support %s, %s", op::ToString(selfShape).GetString(),
            op::ToString(mat2Shape).GetString());
        return false;
    } else if (dimTensor2 == 1 || dimTensor2 == 2) { // tensor1 dims number is 1 OR tensor2 dims number is 2
        selfKDim = selfShape.GetDim(dimTensor1 - 1); // the rear dim 1
        mat2KDim = mat2Shape.GetDim(0);              // the front 0
        if (selfKDim != mat2KDim) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different %s, %s",
                op::ToString(selfShape).GetString(), op::ToString(mat2Shape).GetString());
            return false;
        }
    } else if (dimTensor2 >= 3) {                    // tensor2 dims number >= 3
        selfKDim = selfShape.GetDim(dimTensor1 - 1); // the rear dim 1
        mat2KDim = mat2Shape.GetDim(dimTensor2 - 2); // the rear dim 2
        if (selfKDim != mat2KDim) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different %s, %s",
                op::ToString(selfShape).GetString(), op::ToString(mat2Shape).GetString());
            return false;
        }
    }
    // 检查是否满足broadcast规则
    if (dimTensor1 >= 2 && dimTensor2 >= 2) { // the dims larger than 2
        op::Shape broadcastShape;
        auto selfBroadcastShape = GetBroadcastShape(self);
        auto mat2BroadcastShape = GetBroadcastShape(mat2);
        if (!BroadcastInferShape(selfBroadcastShape, mat2BroadcastShape, broadcastShape)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
                op::ToString(mat2->GetViewShape()).GetString());
            return false;
        }
    }

    return true;
}

inline static aclnnStatus CheckParam(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* out, int8_t cubeMathType)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, mat2, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckWeightNzDtypeValid(self, mat2, out, cubeMathType), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查Shape是否支持
    CHECK_RET(CheckShapeValid(self, mat2), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查cubeMathType
    CHECK_RET(CheckMathType(self, mat2, cubeMathType), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool CheckFormat(
    const aclTensor* selfTensor, [[maybe_unused]] const aclTensor* mat2Tensor, const aclTensor* outTensor)
{
    auto selfFormat = selfTensor->GetStorageFormat();
    auto outTensorFormat = outTensor->GetStorageFormat();
    bool noSupportFormat =
        ((selfFormat == Format::FORMAT_FRACTAL_NZ) || (outTensorFormat == Format::FORMAT_FRACTAL_NZ));
    if (noSupportFormat) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            " The 'self' or 'out' tensor currently not supports NZ format"
            "format");
        return false;
    }
    return true;
}

inline static aclnnStatus CheckInputParams(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* out, int8_t cubeMathType)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, mat2, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    auto archRule = BuildRule();
    CHECK_RET(archRule != nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(
        archRule -> CheckInput(self, mat2, nullptr, out, cubeMathType),
        ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckWeightNzDtype(self, mat2), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查Shape是否支持
    CHECK_RET(CheckShapeValid(self, mat2), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查self和mat2的format是否符合要求
    CHECK_RET(CheckFormat(self, mat2, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* ProcessEmptyTensor(const aclTensor* self, const aclTensor* out, aclOpExecutor* executor)
{
    // 获取shape信息
    op::Shape outShape = out->GetViewShape();
    auto output = executor->AllocTensor(outShape, self->GetDataType());
    CHECK_RET(output != nullptr, nullptr);
    if (output->IsEmpty()) {
        OP_LOGI("Returning an empty tensor without actually doing calculation.");
        return output;
    }
    FVector<int64_t> fillShape = GetShape(output);
    const aclTensor* dims = executor->ConvertToTensor(fillShape.data(), fillShape.size(), op::DataType::DT_INT64);
    aclIntArray* shapeArray = executor->AllocIntArray(fillShape.data(), fillShape.size());
    const aclScalar* valueScalar = executor->AllocScalar(0);
    const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, out->GetDataType());
    auto fillTensor = l0op::Fill(dims, valueTensor, shapeArray, executor);
    return fillTensor;
}

static inline const aclTensor* ContiguousUnsqueezeNd(
    const aclTensor* input, FVector<int64_t>& dimData, aclOpExecutor* executor)
{
    auto inputContiguous = l0op::Contiguous(input, executor);
    CHECK_RET(inputContiguous != nullptr, nullptr);

    auto dims = executor->AllocIntArray(dimData.data(), dimData.size());
    auto output = l0op::UnsqueezeNd(inputContiguous, dims, executor);
    CHECK_RET(output != nullptr, nullptr);

    return output;
}

const aclTensor* SetTensorToNZFormat(const aclTensor* input, op::Shape& shape, aclOpExecutor* executor)
{
    auto formatTensor = executor->CreateView(input, shape, input->GetViewOffset());
    CHECK_RET(formatTensor != nullptr, nullptr);
    formatTensor->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    formatTensor->SetOriginalFormat(op::Format::FORMAT_ND);
    formatTensor->SetViewShape(input->GetViewShape());
    return formatTensor;
}

bool CheckWeightNzShapeValid(const aclTensor* self, const aclTensor* mat2)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    bool isSupportNpuArch = ((npuArch == NpuArch::DAV_2201) || (npuArch == NpuArch::DAV_3510));
    if (!isSupportNpuArch) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Weight NZ is unsupported by the current SOC version [%s].",
            op::ToString(socVersion).GetString());
        return false;
    }

    // mat2 format must be NZ
    if (ge::GetPrimaryFormat(mat2->GetStorageFormat()) != Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of mat2 must be FRACTAL_NZ, actual is %s.",
            op::ToString(mat2->GetStorageFormat()).GetString());
        return false;
    }

    // view shape为2
    OP_CHECK_WRONG_DIMENSION(self, DIMS_TWO, return false);
    OP_CHECK_WRONG_DIMENSION(mat2, DIMS_TWO, return false);

    auto storageShape = mat2->GetStorageShape();
    auto storageShapeDim = storageShape.GetDimNum();
    OP_CHECK(
        storageShapeDim == 4,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Only support mat2 storageShapeDim is 4, which are [%zu].", storageShapeDim),
        return false);

    // check viewShape
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    if (mat2Shape.GetDim(0) == 1 || mat2Shape.GetDim(1) == 1) { // 不支持N=1 or K = 1
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Not support mat2 n = 1 or k = 1 when foramt is FRACTAL_NZ, mat2: %s",
            op::ToString(mat2Shape).GetString());
        return false;
    }
    auto dimTensor1 = selfShape.GetDimNum();
    auto selfKDim = selfShape.GetDim(dimTensor1 - 1);
    auto mat2KDim = mat2Shape.GetDim(0);
    if (selfKDim != mat2KDim) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different %s, %s",
            op::ToString(selfShape).GetString(), op::ToString(mat2Shape).GetString());
        return false;
    }
    return true;
}

aclnnStatus CheckWeightNzParam(const aclTensor* self, const aclTensor* mat2, const aclTensor* out, int8_t cubeMathType)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, mat2, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckWeightNzDtypeValid(self, mat2, out, cubeMathType), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查Shape是否支持
    CHECK_RET(CheckWeightNzShapeValid(self, mat2), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查cubeMathType
    CHECK_RET(CheckMathType(self, mat2, cubeMathType), ACLNN_ERR_PARAM_INVALID);
    auto archRule = BuildRule();
    CHECK_RET(archRule != nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(
        archRule -> CheckInput(self, mat2, nullptr, out, cubeMathType),
        ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("MatmulWeightNz check params success.");
    return ACLNN_SUCCESS;
}

aclnnStatus CheckWeightNzInputParams(const aclTensor* self, const aclTensor* mat2, const aclTensor* out, int8_t cubeMathType)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, mat2, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    auto archRule = BuildRule();
    CHECK_RET(archRule != nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(
        archRule -> CheckInput(self, mat2, nullptr, out, cubeMathType),
        ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckWeightNzDtype(self, mat2), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查Shape是否支持
    CHECK_RET(CheckWeightNzShapeValid(self, mat2), ACLNN_ERR_PARAM_INVALID);

    OP_LOGD("MatmulWeightNz check params success.");

    return ACLNN_SUCCESS;
}

op::Shape GetWeightNzShape(const aclTensor* input, bool transpose)
{
    size_t viewDimNum = input->GetViewShape().GetDimNum();
    uint64_t k = transpose ? input->GetViewShape().GetDim(viewDimNum - LAST_FIRST_DIM_INDEX)
                           : input->GetViewShape().GetDim(viewDimNum - LAST_SECOND_DIM_INDEX);
    uint64_t n = transpose ? input->GetViewShape().GetDim(viewDimNum - LAST_SECOND_DIM_INDEX)
                           : input->GetViewShape().GetDim(viewDimNum - LAST_FIRST_DIM_INDEX);
    // cal C0
    int c0 = NZ_K0_VALUE_16;
    if (input->GetDataType() == DataType::DT_FLOAT) {
        c0 = NZ_K0_VALUE_32;
    }
    uint64_t k0 = transpose ? c0 : NZ_STORAGE_PENULTIMATE_DIM;
    uint64_t n0 = transpose ? NZ_STORAGE_PENULTIMATE_DIM : c0;
    uint64_t k1 = static_cast<uint64_t>(CeilDiv(static_cast<uint64_t>(k), k0));
    uint64_t n1 = static_cast<uint64_t>(CeilDiv(static_cast<uint64_t>(n), n0));

    op::Shape weightNzShape;
    // for batch dims
    for (size_t i = 0; i < viewDimNum - LAST_SECOND_DIM_INDEX; i++) {
        weightNzShape.AppendDim(input->GetViewShape().GetDim(i));
    }

    if (transpose) {
        weightNzShape.AppendDim(k1);
        weightNzShape.AppendDim(n1);
    } else {
        weightNzShape.AppendDim(n1);
        weightNzShape.AppendDim(k1);
    }
    // 16 c0 外轴固定16, 内轴c0, 不转置时 [n1, k1, 16, c0]， 转置时[k1, n1, 16, c0]
    weightNzShape.AppendDim(NZ_STORAGE_PENULTIMATE_DIM);
    weightNzShape.AppendDim(c0);

    return weightNzShape;
}

bool CheckWeightNzStorageShape(const op::Shape& nzShape, const op::Shape& storageShape)
{
    uint64_t nzDimMultiply = 1;
    uint64_t nzDimNum = nzShape.GetDimNum();
    for (uint64_t i = 0; i < nzDimNum; i++) {
        nzDimMultiply *= nzShape[i];
    }

    uint64_t storageDimMultiply = 1;
    uint64_t storageDimNum = storageShape.GetDimNum();
    for (uint64_t i = 0; i < storageDimNum; i++) {
        storageDimMultiply *= storageShape[i];
    }

    return nzDimMultiply == storageDimMultiply;
}

static bool GetTransposeAttrValue(const aclTensor *tensor) {
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - LAST_FIRST_DIM_INDEX;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - LAST_SECOND_DIM_INDEX;
    // check if tensor is contiguous layout
    // viewStride [1, K] viewShape [K, N] -> transpose=True
    // viewStride [K, 1] viewShape [K, N] -> transpose=False
    // K or N = 1 -> transpose=undeterminated
    if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        OP_LOGI("Matmul GetTransposeAttrValue, find tensor not contiguous.");
        return true;
    }
    return false;
}

static const aclTensor* BuildMatMulWeightNzGraph(
    const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType, aclOpExecutor* executor)
{
    // 空tensor 处理
    if (self->IsEmpty() || mat2->IsEmpty()) {
        auto emptyOut = ProcessEmptyTensor(self, out, executor);
        CHECK_RET(emptyOut != nullptr, nullptr);
        return emptyOut;
    }

    const aclTensor* matmulOut = nullptr;

    // adpat for weightNz transpose scene
    bool transposeX2 = GetTransposeAttrValue(mat2);
    // swap last two dims value
    if (transposeX2) {
        const_cast<aclTensor *>(mat2)->SetViewShape(SwapLastTwoDimValue(mat2->GetViewShape()));
    }
    // Check storage shape Nz shape
    op::Shape weightNzShape = GetWeightNzShape(mat2, transposeX2);
    if (!CheckWeightNzStorageShape(weightNzShape, mat2->GetStorageShape())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "mat2'format only support NZ, but now mat2's format is not NZ, please convert the input format to NZ.");
        return nullptr;
    }

    // Set Nz format
    mat2 = SetTensorToNZFormat(mat2, weightNzShape, executor);
    CHECK_RET(mat2 != nullptr, nullptr);

    // 固定selt二维 mat2四维
    matmulOut = ExecMmOpWithBias(self, mat2, nullptr, cubeMathType, executor, transposeX2);
    CHECK_RET(matmulOut != nullptr, nullptr);

    // Reshape to out shape
    auto matReshape = l0op::Reshape(matmulOut, out->GetViewShape(), executor);
    CHECK_RET(matReshape != nullptr, nullptr);

    return matReshape;
}

static inline const aclTensor* BuildBatchMatmulGraph(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* out, int8_t cubeMathType, aclOpExecutor* executor)
{
    auto matmulOut = ExecBmmOp(self, mat2, out, cubeMathType, executor);
    CHECK_RET(matmulOut != nullptr, nullptr);
    return matmulOut;
}

// ===========================================================================================

static const aclTensor* MatmulProcess(const aclTensor* mat1, const aclTensor* mat2, const aclTensor* out, int8_t cubeMathType, MmOpInfo& mmOpInfo, aclOpExecutor* executor)
{
    return MatmulCommonProcess(mat1, mat2, nullptr, out, cubeMathType, mmOpInfo, executor, false);
}

static inline const aclTensor* BatchMatmulProcess(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* out, int8_t cubeMathType, aclOpExecutor* executor)
{
    auto matmulOut = ExecBmmOpWithBiasV2(self, mat2, nullptr, out, cubeMathType, executor);
    CHECK_RET(matmulOut != nullptr, nullptr);
    return matmulOut;
}

class MatMulEmptyTensorGraph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus Impl() override{
        // 获取shape信息
        op::Shape outShape = output->GetViewShape();
        auto outputNew = executor->AllocTensor(outShape, matA->GetDataType());
        CHECK_RET(outputNew != nullptr, ACLNN_ERR_INNER_NULLPTR);
        FVector<int64_t> fillShape = GetShape(outputNew);
        const aclTensor* dims = executor->ConvertToTensor(fillShape.data(), fillShape.size(), op::DataType::DT_INT64);
        CHECK_RET(dims != nullptr, ACLNN_ERR_INNER_NULLPTR);
        aclIntArray* shapeArray = executor->AllocIntArray(fillShape.data(), fillShape.size());
        CHECK_RET(shapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const aclScalar* valueScalar = executor->AllocScalar(0);
        CHECK_RET(valueScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, output->GetDataType());
        CHECK_RET(valueTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto fillTensor = l0op::Fill(dims, valueTensor, shapeArray, executor);
        CHECK_RET(fillTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        convOut = fillTensor;
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override{
        // 执行ViewCopy
        auto result = l0op::ViewCopy(convOut, output, executor);
        CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    };

    ~MatMulEmptyTensorGraph() override = default;
};

class MatMulDotGraph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus PreProcess() override{
        auto self = matA;
        auto mat2 = matB;

        // 检查输入size是否相等
        auto dimSize1 = self->GetViewShape().GetDim(0);
        auto dimSize2 = mat2->GetViewShape().GetDim(0);
        if (dimSize1 != dimSize2) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Self dimSize [%ld] should be same as mat2 dimSize [%ld].", dimSize1, dimSize2);
            return ACLNN_ERR_PARAM_INVALID;
        }

        // 连续性转换
        self = l0op::Contiguous(self, executor);
        CHECK_RET(self != nullptr, ACLNN_ERR_INNER_NULLPTR);
        mat2 = l0op::Contiguous(mat2, executor);
        CHECK_RET(mat2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 全部转成ND
        self = l0op::ReFormat(self, op::Format::FORMAT_ND);
        CHECK_RET(self != nullptr, ACLNN_ERR_INNER_NULLPTR);
        mat2 = l0op::ReFormat(mat2, op::Format::FORMAT_ND);
        CHECK_RET(mat2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 提升精度
        auto promoteType = op::PromoteType(self->GetDataType(), mat2->GetDataType());
        self = l0op::Cast(self, promoteType, executor);
        CHECK_RET(self != nullptr, ACLNN_ERR_INNER_NULLPTR);
        mat2 = l0op::Cast(mat2, promoteType, executor);
        CHECK_RET(mat2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

        matA = self;
        matB = mat2;
        return ACLNN_SUCCESS;
    };

    aclnnStatus Impl() override{
        // 执行点乘
        const aclTensor* out  = l0op::Dot(matA, matB, executor);
        CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);
        convOut = out;
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override{
        return CommonPostProcessWithReshape();
    };

    ~MatMulDotGraph() override = default;
};

class MatMulDimNum1And2Graph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus PreProcess() override{
        // matA的Unsqueeze
        FVector<int64_t> dimData{0};
        auto matAUnsqueeze = ContiguousUnsqueezeNd(matA, dimData, executor);
        CHECK_RET(matAUnsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);
        matA = matAUnsqueeze;
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override{
        return CommonPostProcessWithReshape();
    };

    aclnnStatus Impl() override{
        // 执行 Matmul: out = matA @ matB
        const aclTensor* out = MatmulProcess(matA, matB, output, cubeMathType, opInfo, executor);
        CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);

        convOut = out;
        return ACLNN_SUCCESS;
    };

    ~MatMulDimNum1And2Graph() override = default;
};

class MatMulDimNum2And1Graph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus PostProcess() override{
        return CommonPostProcessWithReshape();
    };

    aclnnStatus PreProcess() override{
        // matB的Unsqueeze
        FVector<int64_t> dimData{-1};
        auto matBUnsqueeze = ContiguousUnsqueezeNd(matB, dimData, executor);
        CHECK_RET(matBUnsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);
        matB = matBUnsqueeze;
        return ACLNN_SUCCESS;
    };

    aclnnStatus Impl() override{
        // 执行 Matmul: out = matA @ matB
        const aclTensor* out = MatmulProcess(matA, matB, output, cubeMathType, opInfo, executor);
        CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);

        convOut = out;
        return ACLNN_SUCCESS;
    };

    ~MatMulDimNum2And1Graph() override = default;
};

class MatMulDimNum2And2Graph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus Impl() override{
        // 执行 Matmul: out = matA @ matB
        const aclTensor* out = MatmulProcess(matA, matB, output, cubeMathType, opInfo, executor);
        CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);

        convOut = out;
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override{
        return CommonPostProcessWithReshape();
    };

    ~MatMulDimNum2And2Graph() override = default;
};

class MatMulDimNumMatAGe3Graph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus PreProcess() override{
        auto mat2Unsqueeze = matB;
        auto dimTensor1 = matA->GetViewShape().GetDimNum();
        auto dimTensor2 = matB->GetViewShape().GetDimNum();
        if (dimTensor2 == 1) {
            FVector<int64_t> dimData{-1};
            mat2Unsqueeze = ContiguousUnsqueezeNd(matB, dimData, executor);
            CHECK_RET(mat2Unsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        auto selfReshape = matA;
        // 非连续3D * 2D直接走MM, 无需转连续
        if (!Ops::NN::IsSliceNonContiguous(matA, matB)) {
            // Fold the batch into the first dimension
            auto selfContiguous = l0op::Contiguous(matA, executor);
            CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
            op::Shape shape{-1, selfContiguous->GetViewShape().GetDim(dimTensor1 - 1)};
            selfReshape = l0op::Reshape(selfContiguous, shape, executor);
            CHECK_RET(selfReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        // exec matA matB
        matA = selfReshape;
        matB = mat2Unsqueeze;
        return ACLNN_SUCCESS;
    };

    aclnnStatus Impl() override{
        // 执行 Matmul: out = matA @ matB
        const aclTensor* out = MatmulProcess(matA, matB, output, cubeMathType, opInfo, executor);
        CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);

        convOut = out;
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override{
        return CommonPostProcessWithReshape();
    };

    ~MatMulDimNumMatAGe3Graph() override = default;
};

class MatMulDimNumMatBGe3Graph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus PreProcess() override{
        auto dimTensor1 = matA->GetViewShape().GetDimNum();
        FVector<int64_t> dimData;
        if (dimTensor1 == 1) {
            dimData = FVector<int64_t>{0}; // unsquee dim 0
        } else {
            dimData = FVector<int64_t>{0, 1}; //  unsquee dim 0,1
        }
        auto selfUnsqueeze = ContiguousUnsqueezeNd(matA, dimData, executor);
        CHECK_RET(selfUnsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);

        matA = selfUnsqueeze;
        return ACLNN_SUCCESS;
    };

    aclnnStatus Impl() override{
        // 执行 BatchMatmul: out = matA @ matB
        auto out = BatchMatmulProcess(matA, matB, output, cubeMathType, executor);

        CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);

        convOut = out;
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override{
        return CommonPostProcessWithReshape();
    };

    ~MatMulDimNumMatBGe3Graph() override = default;
};

class MatMulDimNumBothGe3Graph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus Impl() override{
        // 执行 BatchMatmul: out = matA @ matB
        auto out = BatchMatmulProcess(matA, matB, output, cubeMathType, executor);

        CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);

        convOut = out;
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override{
        return CommonPostProcessWithReshape();
    };

    ~MatMulDimNumBothGe3Graph() override = default;
};

class MatMulWeightNzGraph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus Impl() override{
        // adpat for weightNz transpose scene
        bool transposeX2 = GetTransposeAttrValue(matB);
        // swap last two dims value
        if (transposeX2) {
            const_cast<aclTensor *>(matB)->SetViewShape(SwapLastTwoDimValue(matB->GetViewShape()));
        }
        // Check storage shape Nz shape
        op::Shape weightNzShape = GetWeightNzShape(matB, transposeX2);
        if (!CheckWeightNzStorageShape(weightNzShape, matB->GetStorageShape())) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "mat2'format only support NZ, but now mat2's format is not NZ, please convert the input format to NZ.");
            return ACLNN_ERR_PARAM_INVALID;
        }

        // Set Nz format
        matB = SetTensorToNZFormat(matB, weightNzShape, executor);
        CHECK_RET(matB != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 执行 Matmul: out = matA @ matB
        const aclTensor* out = MatmulCommonProcess(matA, matB, nullptr, output, cubeMathType, opInfo, executor, transposeX2);
        CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);

        convOut = out;
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override{
        return CommonPostProcessWithReshape();
    };

    ~MatMulWeightNzGraph() override = default;
};

// 创建Matmul计算图
std::shared_ptr<MatmulGraphImpl> CreateMatmulGraphImpl(
    const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType, aclOpExecutor* executor) {
    std::shared_ptr<MatmulGraphImpl> matmulGraph = nullptr;

    // 空tensor处理, 当输出的shape不为空时
    if ((self->IsEmpty() || mat2->IsEmpty()) && (!out -> IsEmpty())) {
        matmulGraph = std::make_shared<MatMulEmptyTensorGraph>(self, mat2, out, cubeMathType, executor);
        return matmulGraph;
    }

    auto dimTensor1 = self->GetViewShape().GetDimNum();
    auto dimTensor2 = mat2->GetViewShape().GetDimNum();
    if (dimTensor1 == 1 && dimTensor2 == 1) {
        // dot_out
        matmulGraph = std::make_shared<MatMulDotGraph>(self, mat2, out, cubeMathType, executor);
    } else if (dimTensor1 == 2 && dimTensor2 == 1) { // Tensor1 dims number 2  && Tensor2 dims number 1
        matmulGraph = std::make_shared<MatMulDimNum2And1Graph>(self, mat2, out, cubeMathType, executor);
    } else if (dimTensor1 == 1 && dimTensor2 == 2) { // Tensor1 dims number 1 && Tensor2 dims number 2
        matmulGraph = std::make_shared<MatMulDimNum1And2Graph>(self, mat2, out, cubeMathType, executor);
    } else if (dimTensor1 == 2 && dimTensor2 == 2) { // Tensor1 dims number 2 && Tensor2 dims number 2
        matmulGraph = std::make_shared<MatMulDimNum2And2Graph>(self, mat2, out, cubeMathType, executor);
    } else if (dimTensor1 >= 3 && (dimTensor2 == 1 || dimTensor2 == 2)) { // dimTensor1 is 1 or 2 && dimTensor2  >= 3
        // t1:(N, n, m) * t2:(m, p)
        matmulGraph = std::make_shared<MatMulDimNumMatAGe3Graph>(self, mat2, out, cubeMathType, executor);
    } else if ((dimTensor1 == 1 || dimTensor1 == 2) &&  dimTensor2 >= 3) { // dimTensor2 >= 3  && dimTensor1 is 1 or 2
        // t1:(n, m) * t2:(N, m, p)
        matmulGraph = std::make_shared<MatMulDimNumMatBGe3Graph>(self, mat2, out, cubeMathType, executor);
    } else if (dimTensor1 >= 3 && dimTensor2 >= 3) { // Tensor1 dims number >= 3 && Tensor2 dim number >= 3
        matmulGraph = std::make_shared<MatMulDimNumBothGe3Graph>(self, mat2, out, cubeMathType, executor);
    } else { // Impossible cases.
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Internal error self: %s, mat2: %s",
            op::ToString(self->GetViewShape()).GetString(), op::ToString(mat2->GetViewShape()).GetString());
        return nullptr;
    }
    return matmulGraph;
}

// 创建WeightNZ计算图
std::shared_ptr<MatmulGraphImpl> CreateMatmulWeightNZGraphImpl(
    const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType, aclOpExecutor* executor) {
    std::shared_ptr<MatmulGraphImpl> matmulGraph = nullptr;

    // 空tensor处理, 当输出的shape不为空时
    if ((self->IsEmpty() || mat2->IsEmpty()) && (!out -> IsEmpty())) {
        matmulGraph = std::make_shared<MatMulEmptyTensorGraph>(self, mat2, out, cubeMathType, executor);
        return matmulGraph;
    }

    matmulGraph = std::make_shared<MatMulWeightNzGraph>(self, mat2, out, cubeMathType, executor);

    return matmulGraph;
}

} // namespace

aclnnStatus aclnnMatmulGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType, size_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMatmul, DFX_IN(self, mat2, cubeMathType), DFX_OUT(out));

    // 入参检查
    auto ret = CheckInputParams(self, mat2, out, cubeMathType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空tensor处理
    if (out -> IsEmpty() && (self->IsEmpty() || mat2->IsEmpty())) {
        OP_LOGI("Returning an empty tensor without actually doing calculation.");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 构建matmul计算图
    std::shared_ptr<MatmulGraphImpl> matmulGraph = CreateMatmulGraphImpl(self, mat2, out, cubeMathType, uniqueExecutor.get());
    CHECK_RET(matmulGraph != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 执行计算图
    auto executeStatus = matmulGraph -> Execute();
    CHECK_RET(executeStatus == ACLNN_SUCCESS, executeStatus);

    // 获取workspace
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMatmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMatmul);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnMatmulWeightNzGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType, size_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMatmulWeightNz, DFX_IN(self, mat2, cubeMathType), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 入参检查
    auto ret = CheckWeightNzParam(self, mat2, out, cubeMathType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 构建matmul计算图
    auto matmulOut = BuildMatMulWeightNzGraph(self, mat2, out, cubeMathType, uniqueExecutor.get());
    CHECK_RET(matmulOut != nullptr, ACLNN_ERR_PARAM_INVALID);
    if (matmulOut->IsEmpty()) {
        // 当输出为空tensor的场景，空tensor处理
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto viewCopyResult = l0op::ViewCopy(matmulOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取workspace
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMatmulWeightNz(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMatmulWeightNz);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}