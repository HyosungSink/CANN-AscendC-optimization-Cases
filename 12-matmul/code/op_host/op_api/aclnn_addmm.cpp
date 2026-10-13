/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_addmm.h"
#include "level0/add.h"
#include "level0/axpy.h"
#include "level0/broadcast_to.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "matmul/common/op_host/op_api/matmul.h"
#include "matmul/common/op_host/op_api/cube_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "level0/muls.h"

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "util/math_util.h"

using Ops::Base::CeilDiv;
using namespace Ops::NN;
using namespace op;
namespace {
struct AclnnAddmmTensor {
    const aclTensor* self;
    const aclTensor* mat1;
    const aclTensor* mat2;
    const aclScalar* beta;
    const aclScalar* alpha;
    aclTensor* out;
};

static const size_t LAST_SECOND_DIM_INDEX = 2;
static const size_t LAST_FIRST_DIM_INDEX = 1;
static const int NZ_STORAGE_PENULTIMATE_DIM = 16;
static const int NZ_K0_VALUE_16 = 16;

static inline bool CheckNotNull(AclnnAddmmTensor& addmmTensor)
{
    OP_CHECK_NULL(addmmTensor.self, return false);
    OP_CHECK_NULL(addmmTensor.mat1, return false);
    OP_CHECK_NULL(addmmTensor.mat2, return false);
    OP_CHECK_NULL(addmmTensor.beta, return false);
    OP_CHECK_NULL(addmmTensor.alpha, return false);
    OP_CHECK_NULL(addmmTensor.out, return false);
    return true;
}

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> dtypeSupportList = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> dtypeSupportListWithoutBf16 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> dtypeSupportListMat1AndMat2 = {
 	op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static inline bool CheckNpuArchIsSupportBf16(void)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    return (npuArch == NpuArch::DAV_2201) || (npuArch == NpuArch::DAV_3510);
}

static inline bool CheckWeightNzDtypeValid(
    const aclTensor* self, const aclTensor* mat1, const aclTensor* mat2, const aclTensor* out)
{
    bool bf16flag = CheckNpuArchIsSupportBf16();
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    auto dtypeList = bf16flag ? dtypeSupportList : dtypeSupportListWithoutBf16;
    auto dtypeListMat1AndMat2 = bf16flag ? dtypeSupportListMat1AndMat2 : dtypeSupportListWithoutBf16;
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(mat1, dtypeListMat1AndMat2, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(mat2, dtypeListMat1AndMat2, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeList, return false);
    if (!bf16flag && (self->GetDataType() == op::DataType::DT_BF16 || mat1->GetDataType() == op::DataType::DT_BF16 ||
                      mat2->GetDataType() == op::DataType::DT_BF16 || out->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Bfloat16 is unsupported by the current SOC version [%s], self is %s, mat1 is %s, mat2 is %s, out is %s",
            op::ToString(socVersion).GetString(), op::ToString(self->GetDataType()).GetString(),
            op::ToString(mat1->GetDataType()).GetString(), op::ToString(mat2->GetDataType()).GetString(),
            op::ToString(out->GetDataType()).GetString());
        return false;
    }

    return true;
}

static inline bool CheckMatmul(const aclTensor* self, const aclTensor* mat1, const aclTensor* mat2)
{
    // check mat1 dims number is 2
    OP_CHECK_WRONG_DIMENSION(mat1, 2, return false);

    // check mat2 dims number is 2
    OP_CHECK_WRONG_DIMENSION(mat2, 2, return false);

    if ((mat1->GetViewShape())[1] != (mat2->GetViewShape())[0]) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different.");
        return false;
    }

    if (self->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The formats of self should be ND, but the actual formats are [%s]",
                op::ToString(self->GetStorageFormat()).GetString());
        return false;
    }

    return true;
}

static inline bool CheckBroadcast(const aclTensor* self, const aclTensor* mat1, const aclTensor* mat2)
{
    // self的维度不能大于2
    if (self->GetViewShape().GetDimNum() > 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim of self should be less than 3.");
        return false;
    }
    if (self->GetStorageShape().GetDimNum() < 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self can not be empty.");
        return false;
    }
    op::Shape matmulShape = {(mat1->GetViewShape())[0], (mat2->GetViewShape())[1]};
    OP_CHECK_BROADCAST_WITH_SHAPE(self, matmulShape, return false);

    return true;
}

// 假设mat1是 n x m，mat2是 m x p，out必须是 n x p    如果n / p为0，那么out为empty即可
static inline bool CheckOutShape(const aclTensor* mat1, const aclTensor* mat2, const aclTensor* out)
{
    int64_t n = mat1->GetViewShape().GetDim(0);
    int64_t p = mat2->GetViewShape().GetDim(1);
    int64_t out_n = out->GetViewShape().GetDim(0);
    int64_t out_p = out->GetViewShape().GetDim(1);
    if (n == 0 || p == 0) {
        return out->IsEmpty();
    }
    if (n != out_n || p != out_p) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of out should be [%ld,%ld], but the current is %s.", n, p,
            op::ToString(out->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckMathType(const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType)
{
    bool selfFloat = self->GetDataType() == DataType::DT_FLOAT;
    bool mat2Float = mat2->GetDataType() == DataType::DT_FLOAT;
    auto promoteType = selfFloat || mat2Float ? DataType::DT_FLOAT : self->GetDataType();
    return CheckCubeMathTypeForMm(promoteType, cubeMathType);
}

static aclnnStatus CheckInputParams(AclnnAddmmTensor& addmmTensor, int8_t cubeMathType)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(addmmTensor), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    auto archRule = BuildRule();
    CHECK_RET(archRule != nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(
        archRule -> CheckInput(addmmTensor.mat1, addmmTensor.mat2, addmmTensor.self, addmmTensor.out, cubeMathType),
        ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self, mat1和mat2是否满足matmul条件
    CHECK_RET(CheckMatmul(addmmTensor.self, addmmTensor.mat1, addmmTensor.mat2), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查self和mat1@mat2是否能broadcast
    CHECK_RET(CheckBroadcast(addmmTensor.self, addmmTensor.mat1, addmmTensor.mat2), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查out必须和mat1@mat2的shape一致
    CHECK_RET(CheckOutShape(addmmTensor.mat1, addmmTensor.mat2, addmmTensor.out), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查是否满足GemmV3BaseKernel条件。
    CHECK_RET(
        CheckCubeMathTypeForAddMm(addmmTensor.mat1, addmmTensor.mat2, addmmTensor.self, addmmTensor.out, cubeMathType),
        ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

// ================================================================================================

// mat1: a x b, mat2: b x c  ->   a x c 是否为空tensor，为空tensor返回true
static inline bool CheckMulResIsEmpty(const aclTensor* mat1, const aclTensor* mat2)
{
    return mat1->GetViewShape().GetDim(0) == 0 || mat2->GetViewShape().GetDim(1) == 0;
}

// mat1 * mat2结果为空tensor，返回 beta * self 并且broadcast成out的shape
static aclnnStatus AddmmMulEmptyProcess(
    const aclTensor* self, const aclScalar* beta, aclTensor* out, aclOpExecutor* executor)
{
    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto mulOut = l0op::Muls(selfContiguous, beta->ToFloat(), executor);
    CHECK_RET(mulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // broadcast成和out一个shape
    if (mulOut->GetViewShape() != out->GetViewShape()) {
        int64_t tensorSize = static_cast<int64_t>(out->GetViewShape().GetDimNum());
        std::vector<int64_t> tensorShape(tensorSize);
        for (int64_t i = 0; i < tensorSize; i++) {
            tensorShape[i] = (out->GetViewShape())[i];
        }
        auto outShape = executor->AllocIntArray(tensorShape.data(), tensorSize);
        CHECK_RET(outShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        mulOut = l0op::BroadcastTo(mulOut, outShape, executor);
        CHECK_RET(mulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto viewCopyResult = l0op::ViewCopy(mulOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static const aclTensor* AddProcess(
    const aclTensor* mulOut, const aclTensor* matmulOut, const aclScalar* alpha, bool isInplace, aclOpExecutor* executor)
{
    // Add算子需要对self和mat1两个输入做隐式数据类型转换，根据具体算子语义按需调用
    auto promoteTypeAdd = op::PromoteType(mulOut->GetDataType(), matmulOut->GetDataType());

    auto mulOutCasted = l0op::Cast(mulOut, promoteTypeAdd, executor);
    CHECK_RET(mulOutCasted != nullptr, nullptr);

    auto matmulOutCasted = l0op::Cast(matmulOut, promoteTypeAdd, executor);
    CHECK_RET(matmulOutCasted != nullptr, nullptr);

    // 进行Add计算
    const aclTensor* addOut = nullptr;
    if (std::abs(alpha->ToFloat() - 1.0f) <= std::numeric_limits<float>::epsilon()) {
        addOut = reinterpret_cast<void*>(l0op::AddInplace) != nullptr && !isInplace ?
            l0op::AddInplace(mulOutCasted, matmulOutCasted, executor) :
            l0op::Add(mulOutCasted, matmulOutCasted, executor);
    } else {
        addOut = reinterpret_cast<void*>(l0op::AxpyInplace) != nullptr && !isInplace ?
            l0op::AxpyInplace(mulOutCasted, matmulOutCasted, alpha->ToFloat(), executor) :
            l0op::Axpy(mulOutCasted, matmulOutCasted, alpha->ToFloat(), executor);
    }
    return addOut;
}

static const aclTensor* MulsProcess(const aclTensor* mat, const aclScalar* scalar, bool isMulsInplace, aclOpExecutor* executor){
    const aclTensor* mulsOut = nullptr;
    auto matContiguous = l0op::Contiguous(mat, executor);
    if (fabs(scalar->ToFloat() - 1.0f) <= numeric_limits<float>::epsilon()) {
        mulsOut = matContiguous;
    } else {
        mulsOut = (reinterpret_cast<void*>(l0op::MulsInplace) != nullptr && isMulsInplace) ?
            l0op::MulsInplace(matContiguous, scalar->ToFloat(), executor) :
            l0op::Muls(matContiguous, scalar->ToFloat(), executor);
        CHECK_RET(mulsOut != nullptr, nullptr);
    }
    return mulsOut;
}

static const aclTensor* MatmulProcess(const aclTensor* mat1, const aclTensor* mat2, const aclTensor* out, int8_t cubeMathType, MmOpInfo& mmOpInfo, aclOpExecutor* executor)
{
    return MatmulCommonProcess (mat1, mat2, nullptr, out, cubeMathType, mmOpInfo, executor, false);
}

static const aclTensor* MatmulWithBiasProcess(const aclTensor* mat1, const aclTensor* mat2, const aclTensor* self, const aclTensor* out, int8_t cubeMathType, MmOpInfo& mmOpInfo, aclOpExecutor* executor)
{
    return MatmulCommonProcess (mat1, mat2, self, out, cubeMathType, mmOpInfo, executor, false);
}

// ============================================================================

static const aclTensor* MatmulMulProcess(AclnnAddmmTensor& addmmTensor, int8_t cubeMathType, aclOpExecutor* executor)
{
    auto matmulOut = ExecMmOp(addmmTensor.mat1, addmmTensor.mat2, cubeMathType, executor);
    CHECK_RET(matmulOut != nullptr, nullptr);

    const aclTensor* mulOut = nullptr;
    if (fabs(addmmTensor.alpha->ToFloat() - 1.0f) <= numeric_limits<float>::epsilon()) {
        mulOut = matmulOut;
    } else {
        mulOut = reinterpret_cast<void*>(l0op::MulsInplace) != nullptr ?
            l0op::MulsInplace(matmulOut, addmmTensor.alpha->ToFloat(), executor) :
            l0op::Muls(matmulOut, addmmTensor.alpha->ToFloat(), executor);
        CHECK_RET(mulOut != nullptr, nullptr);
    }
    auto castOut = l0op::Cast(mulOut, addmmTensor.out->GetDataType(), executor);
    return castOut;
}

static const aclTensor* AddMatmulProcess(
    AclnnAddmmTensor& addmmTensor, int8_t cubeMathType, aclOpExecutor* uniqueExecutor)
{
    auto selfContiguous = l0op::Contiguous(addmmTensor.self, uniqueExecutor);
    if (addmmTensor.self != nullptr && addmmTensor.self->GetDataType() == op::DataType::DT_BF16) {
        selfContiguous = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, uniqueExecutor);
    }
    CHECK_RET(selfContiguous != nullptr, nullptr);

    const aclTensor* mulOut = nullptr;
    if (fabs(addmmTensor.beta->ToFloat() - 1.0f) <= numeric_limits<float>::epsilon()) {
        mulOut = selfContiguous;
    } else {
        mulOut = l0op::Muls(selfContiguous, addmmTensor.beta->ToFloat(), uniqueExecutor);
        CHECK_RET(mulOut != nullptr, nullptr);
    }

    // matmul
    auto matmulOut = ExecMmOp(addmmTensor.mat1, addmmTensor.mat2, cubeMathType, uniqueExecutor);
    CHECK_RET(matmulOut != nullptr, nullptr);

    bool isInplace = false;
    auto addOut = AddProcess(mulOut, matmulOut, addmmTensor.alpha, isInplace, uniqueExecutor);
    CHECK_RET(addOut != nullptr, nullptr);

    auto castOut = l0op::Cast(addOut, addmmTensor.out->GetDataType(), uniqueExecutor);
    return castOut;
}

static op::Shape GetWeightNzShape(const aclTensor* input, bool transpose)
{
    size_t viewDimNum = input->GetViewShape().GetDimNum();
    uint64_t k = transpose ? input->GetViewShape().GetDim(viewDimNum - LAST_FIRST_DIM_INDEX) :
                             input->GetViewShape().GetDim(viewDimNum - LAST_SECOND_DIM_INDEX);
    uint64_t n = transpose ? input->GetViewShape().GetDim(viewDimNum - LAST_SECOND_DIM_INDEX) :
                             input->GetViewShape().GetDim(viewDimNum - LAST_FIRST_DIM_INDEX);
    // cal C0
    int c0 = NZ_K0_VALUE_16;
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

static bool CheckWeightNzStorageShape(const op::Shape& nzShape, const op::Shape& storageShape)
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

static inline bool CheckMatmulWeightNz(const aclTensor* mat1, const aclTensor* mat2)
{
    if (mat1->GetDataType() == op::DataType::DT_FLOAT || mat2->GetDataType() == op::DataType::DT_FLOAT ||
        mat1->GetDataType() != mat2->GetDataType()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "invalid mat1 dtype [%s] or mat2 dtype [%s] ",
            op::ToString(mat1->GetDataType()).GetString(), op::ToString(mat2->GetDataType()).GetString());
        return false;
    }
    auto storageShape = mat2->GetStorageShape();
    auto storageShapeDim = storageShape.GetDimNum();
    OP_CHECK(
        storageShapeDim == 4,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Only support mat2 storageShapeDim is 4, which are [%zu].", storageShapeDim),
        return false);

    // Check storage shape Nz shape
    op::Shape weightNzShape = GetWeightNzShape(mat2, false);
    if (!CheckWeightNzStorageShape(weightNzShape, mat2->GetStorageShape())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "mat2'format only support NZ, but now mat2's format is not NZ, please convert the input format to NZ.");
        return false;
    }

    if (mat1->GetViewShape().GetDimNum() != 2 || mat2->GetViewShape().GetDimNum() != 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim of mat1 and mat2 should be 2.");
        return false;
    }
    if ((mat1->GetViewShape())[1] != (mat2->GetViewShape())[0]) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different.");
        return false;
    }

    if ((mat2->GetViewShape())[0] == 1 || (mat2->GetViewShape())[1] == 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The k-axis or n-axis can not be 1.");
        return false;
    }

    return true;
}

static aclnnStatus AddmmCheckWeightNzParam(AclnnAddmmTensor& addmmTensor, int8_t cubeMathType)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    bool isSupportNpuArch = ((npuArch == NpuArch::DAV_2201) || (npuArch == NpuArch::DAV_3510));
    if (!isSupportNpuArch) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Weight NZ is unsupported by the current SOC version [%s].",
            op::ToString(socVersion).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(addmmTensor), ACLNN_ERR_PARAM_NULLPTR);
    
    // 仅支持 self ND， mat1 Nd，mat2 Nz排布
    if (addmmTensor.self->GetStorageFormat() != Format::FORMAT_ND ||
        addmmTensor.mat1->GetStorageFormat() != Format::FORMAT_ND ||
        addmmTensor.mat2->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Invalid format, Format of self is [%s], mat1 is [%s], mat2 is [%s].",
            op::ToString(addmmTensor.self->GetStorageFormat()).GetString(),
            op::ToString(addmmTensor.mat1->GetStorageFormat()).GetString(),
            op::ToString(addmmTensor.mat2->GetStorageFormat()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckWeightNzDtypeValid(addmmTensor.self, addmmTensor.mat1, addmmTensor.mat2,
        addmmTensor.out), ACLNN_ERR_PARAM_INVALID);

    auto archRule = BuildRule();
    CHECK_RET(archRule != nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(
        archRule->CheckInput(addmmTensor.mat1, addmmTensor.mat2, addmmTensor.self, addmmTensor.out, cubeMathType),
        ACLNN_ERR_PARAM_INVALID);

    // 3. 检查mat1和mat2是否满足matmulweightNz条件
    CHECK_RET(CheckMatmulWeightNz(addmmTensor.mat1, addmmTensor.mat2), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查self和mat1@mat2是否能broadcast
    CHECK_RET(CheckBroadcast(addmmTensor.self, addmmTensor.mat1, addmmTensor.mat2), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查out必须和mat1@mat2的shape一致
    CHECK_RET(CheckOutShape(addmmTensor.mat1, addmmTensor.mat2, addmmTensor.out), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查cubeMathType
    CHECK_RET(CheckMathType(addmmTensor.mat1, addmmTensor.mat2, cubeMathType), ACLNN_ERR_PARAM_INVALID);

    return ACL_SUCCESS;
}

static bool ProcessEmptyTensor(
   AclnnAddmmTensor& addmmTensor, uint64_t* workspaceSize, aclOpExecutor* executor)
{
    // 如果self是空tensor，返回空tensor。如果mat1 a*b 和mat2 b*c是空tensor，a*c也是空tensor，返回空tensor
    if (addmmTensor.self->IsEmpty() || CheckMulResIsEmpty(addmmTensor.mat1, addmmTensor.mat2)) {
        *workspaceSize = 0;
        return true;
    }

    // 如果mat1 a*b 和mat2 b*c是空tensor，但是a*c不是空tensor, 返回Beta self
    if (!CheckMulResIsEmpty(addmmTensor.mat1, addmmTensor.mat2) &&
        (addmmTensor.mat1->IsEmpty() || addmmTensor.mat2->IsEmpty())) {
        auto addmmRes = AddmmMulEmptyProcess(addmmTensor.self, addmmTensor.beta, addmmTensor.out, executor);
        CHECK_RET(addmmRes == ACLNN_SUCCESS, addmmRes);
        *workspaceSize = executor->GetWorkspaceSize();
        return true;
    }
    return false;
}

// =============================================================================================================================

class AddmmMatmulGraph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus Impl() override{
        if (CheckGemmV3WithAlphaBeta(bias, matA, matB, cubeMathType)) {
            auto outGemmV3 = ExecGemmV3WithAlphaBetaOp(bias, matA, matB, alpha, beta, executor);
            CHECK_RET(outGemmV3 != nullptr, ACLNN_ERR_INNER_NULLPTR);
            convOut = outGemmV3;
            return ACLNN_SUCCESS;
        }
        // 执行 Muls: out1 = beta * bias
        // 非inplace接口不能改变输入tensor，isMulsInplace=false
        const aclTensor* out1 = MulsProcess(bias, beta, false, executor);
        CHECK_RET(out1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 执行 Matmul: out2 = mat1 @ mat2
        // 为了提升addmm的精度，如果输入是fp16或者bf16时，输出需要是fp32类型
        auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
        bool isSupportNpuArch = (npuArch == NpuArch::DAV_2201);
        if (((matA->GetDataType() == DataType::DT_FLOAT16 && matB->GetDataType() == DataType::DT_FLOAT16) || 
            (matA->GetDataType() == DataType::DT_BF16 && matB->GetDataType() == DataType::DT_BF16)) &&
            (cubeMathType == KEEP_DTYPE || cubeMathType == USE_HF32) && isSupportNpuArch) {
            cubeMathType = USE_HIGH_PREC_MODE;
        }
        const aclTensor* out2 = MatmulProcess(matA, matB, output, cubeMathType, opInfo, executor);
        CHECK_RET(out2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        bool isInplace = bias->GetData() == output->GetData();
        // 执行 Add: out = out1 + out2 或Axpy: out = alpha * out2 + out1
        const aclTensor* out = AddProcess(out1, out2, alpha, isInplace, executor);
        CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);
        convOut = out;
        return ACLNN_SUCCESS;
    };

    ~AddmmMatmulGraph() override = default;
};

class AddmmEmptyTensorGraph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus Impl() override{
        // 空Tensor 不做处理
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override{
        return ACLNN_SUCCESS;
    };

    ~AddmmEmptyTensorGraph() override = default;
};

class AddmmAlpha0Graph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus Impl() override{
        // 执行 Muls: out1 = beta * bias
        // 非inplace接口不能改变输入tensor，isMulsInplace=false
        const aclTensor* out1 = MulsProcess(bias, beta, false, executor);
        CHECK_RET(out1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        convOut = out1;
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override{
        // 执行broadCast操作, broadcast成和out一个shape
        if (convOut->GetViewShape() != output->GetViewShape()) {
            int64_t tensorSize = static_cast<int64_t>(output->GetViewShape().GetDimNum());
            std::vector<int64_t> tensorShape(tensorSize);
            for (int64_t i = 0; i < tensorSize; i++) {
                tensorShape[i] = (output->GetViewShape())[i];
            }
            auto outShape = executor->AllocIntArray(tensorShape.data(), tensorSize);
            CHECK_RET(outShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
            convOut = l0op::BroadcastTo(convOut, outShape, executor);
            CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        // 执行cast等操作
        convOut = l0op::Cast(convOut, output->GetDataType(), executor);
        CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 执行ViewCopy
        auto result = l0op::ViewCopy(convOut, output, executor);
        CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    };

    ~AddmmAlpha0Graph() override = default;
};

class AddmmBeta0Graph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus Impl() override{
        // 执行 Matmul: out = mat1 @ mat2
        const aclTensor* out = MatmulProcess(matA, matB, output, cubeMathType, opInfo, executor);
        CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 执行 Muls: out1 = alpha * out
        // 对mmout多muls，为中间内存可以进行再复用，isMulsInplace=true
        const aclTensor* out1 = MulsProcess(out, alpha, true, executor);
        CHECK_RET(out1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        convOut = out1;
        return ACLNN_SUCCESS;
    };

    ~AddmmBeta0Graph() override = default;
};

class AddmmMmOpWithBiasGraph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus Impl() override{
        // 执行 MatmulWithBias: out = mat1 @ mat2 + self
        const aclTensor* out = MatmulWithBiasProcess(matA, matB, bias, output, cubeMathType, opInfo, executor);
        CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);
        convOut = out;
        return ACLNN_SUCCESS;
    };

    ~AddmmMmOpWithBiasGraph() override = default;
};

class AddmmGemmV3Graph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus Impl() override{
        // 执行 Gemmv3 流程, mmOpInfo需要提前获取
        const aclTensor* out = ExecGemmV3Op(matA, matB, bias, opInfo, executor);
        CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);
        convOut = out;
        return ACLNN_SUCCESS;
    };

    ~AddmmGemmV3Graph() override = default;
};

// 创建计算图
std::shared_ptr<MatmulGraphImpl> CreateAddmmGraphImpl(
    const aclTensor* self, const aclTensor* mat1, const aclTensor* mat2, const aclScalar* beta, const aclScalar* alpha,
    aclTensor* out, int8_t cubeMathType, aclOpExecutor* executor, bool isAclnnWeightNz = false)
{
    std::shared_ptr<MatmulGraphImpl> matmulGraph = nullptr;

    // 空tensor处理: 如果self是空tensor，返回空tensor。如果mat1 a*b 和mat2 b*c是空tensor，a*c也是空tensor，返回空tensor
    if (self->IsEmpty() || CheckMulResIsEmpty(mat1, mat2)) {
        matmulGraph = std::make_shared<AddmmEmptyTensorGraph>(mat1, mat2, self, out, alpha, beta, cubeMathType, executor);
        return matmulGraph;
    }

    // 空tensor处理与判断: 如果mat1 a*b 和mat2 b*c是空tensor，但是a*c不是空tensor, 返回Beta self
    if (!CheckMulResIsEmpty(mat1, mat2) && (mat1->IsEmpty() || mat2->IsEmpty())){
        matmulGraph = std::make_shared<AddmmAlpha0Graph>(mat1, mat2, self, out, alpha, beta, cubeMathType, executor);
        return matmulGraph;
    }

    // alpha == 0, 返回Beta self
    if (fabs(alpha->ToFloat() - 0.0f) <= numeric_limits<float>::epsilon()) {
        matmulGraph = std::make_shared<AddmmAlpha0Graph>(mat1, mat2, self, out, alpha, beta, cubeMathType, executor);
        return matmulGraph;
    }

    // alpha != 0 && beta == 0
    if (fabs(beta->ToFloat() - 0.0f) <= numeric_limits<float>::epsilon()){
        matmulGraph = std::make_shared<AddmmBeta0Graph>(mat1, mat2, self, out, alpha, beta, cubeMathType, executor);
        return matmulGraph;
    }
    // 以下全部是 alpha != 0 && beta != 0

    if (NeedToConvertBias(self, mat1, mat2, beta, alpha)) {
        OP_LOGI("run in NeedToConvertBias branch");
        matmulGraph = std::make_shared<AddmmMmOpWithBiasGraph>(mat1, mat2, self, out, alpha, beta, cubeMathType, executor);
        return matmulGraph;
    }

    if(!isAclnnWeightNz) {
        MmOpInfo mmOpInfo;
        if (fabs(alpha->ToFloat() - 1.0f) <= numeric_limits<float>::epsilon() &&
                    fabs(beta->ToFloat() - 1.0f) <= numeric_limits<float>::epsilon() && out->GetData() == self->GetData() &&
                    out->GetDataType() == op::DataType::DT_FLOAT && CheckGemmV3Support(mat1, mat2, mmOpInfo, cubeMathType)) {
            // 切换GemmV3算子条件：判断alpha和beta等于1 && self和out共地址 && shape属于支持范围 && 输出类型dtype=fp32
            matmulGraph = std::make_shared<AddmmGemmV3Graph>(mat1, mat2, self, out, alpha, beta, cubeMathType, executor);
            matmulGraph -> SetOpInfo (mmOpInfo); // 将CheckGemmV3Support计算出的mmOpInfo 直接赋值;
            return matmulGraph;
        }
    }

    // 多数场景
    matmulGraph = std::make_shared<AddmmMatmulGraph>(mat1, mat2, self, out, alpha, beta, cubeMathType, executor);
    return matmulGraph;
}

} // namespace

aclnnStatus aclnnAddmmGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat1, const aclTensor* mat2, const aclScalar* beta, const aclScalar* alpha,
    aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAddmm, DFX_IN(self, mat1, mat2, beta, alpha, cubeMathType), DFX_OUT(out));

    // 校验参数
    AclnnAddmmTensor addmmTensor = {self, mat1, mat2, beta, alpha, out};
    auto ret = CheckInputParams(addmmTensor, cubeMathType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空tensor处理
    if (self->IsEmpty() || CheckMulResIsEmpty(mat1, mat2)) {
        // 如果self是空tensor，返回空tensor。如果mat1 a*b 和mat2 b*c是空tensor，a*c也是空tensor，返回空tensor
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 根据不同的输入选择不同的计算图
    std::shared_ptr<MatmulGraphImpl> matmulGraph = CreateAddmmGraphImpl(self, mat1, mat2, beta, alpha, out, cubeMathType, uniqueExecutor.get(), false);
    CHECK_RET(matmulGraph != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 执行计算图
    auto executeStatus = matmulGraph -> Execute();
    CHECK_RET(executeStatus == ACLNN_SUCCESS, executeStatus);

    // return
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddmm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddmm);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceAddmmGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* mat1, const aclTensor* mat2, const aclScalar* beta,
    const aclScalar* alpha, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto out = const_cast<aclTensor*>(selfRef);
    return aclnnAddmmGetWorkspaceSize(selfRef, mat1, mat2, beta, alpha, out, cubeMathType, workspaceSize, executor);
}

aclnnStatus aclnnInplaceAddmm(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnInplaceAddmm);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

ACLNN_API aclnnStatus aclnnAddmmWeightNzGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat1, const aclTensor* mat2, const aclScalar* beta, const aclScalar* alpha,
    aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAddmmWeightNz, DFX_IN(self, mat1, mat2, beta, alpha, cubeMathType), DFX_OUT(out));

    AclnnAddmmTensor addmmTensor = {self, mat1, mat2, beta, alpha, out};

    auto ret = AddmmCheckWeightNzParam(addmmTensor, cubeMathType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (ProcessEmptyTensor(addmmTensor, workspaceSize, uniqueExecutor.get())) {
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* castOut = nullptr;
    if (fabs(beta->ToFloat() - 0.0f) <= numeric_limits<float>::epsilon()) {
        castOut = MatmulMulProcess(addmmTensor, cubeMathType, uniqueExecutor.get());
    } else if (NeedToConvertBias(self, mat1, mat2, beta, alpha)) {
        OP_LOGD("aclnnAddmmWeightNz run in NeedToConvertBias branch");
        auto biasMmOut = ExecMmOpWithBias(mat1, mat2, self, cubeMathType, uniqueExecutor.get());
        CHECK_RET(biasMmOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        castOut = l0op::Cast(biasMmOut, out->GetDataType(), uniqueExecutor.get());
    } else {
        castOut = AddMatmulProcess(addmmTensor, cubeMathType, uniqueExecutor.get());
    }
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

ACLNN_API aclnnStatus
aclnnAddmmWeightNz(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddmmWeightNz);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}