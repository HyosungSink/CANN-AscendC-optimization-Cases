/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "matmul_util.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "level0/fill.h"
#include "level0/padv3.h"
#include "level0/mul.h"
#include "matmul/common/op_host/op_api/matmul_v2tov3.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/platform.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "op_api/op_api_def.h"
#include "matmul/common/op_host/op_api/cube_util.h"
#include "matmul/common/op_host/math_util.h"
#include "matmul/common/op_host/op_api/matmul.h"

using namespace std;
using namespace op;
using namespace Ops::NN;

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;

namespace {
static const int64_t SPLIT_K_MULTI = 8;
static const int64_t MKN_MAX = 8000000000;
static const int64_t MN_MULTI = 50;
static const int64_t N_KEQAL1_LIMIT = 4000;
static const int64_t MM_KEQAL1_LIMIT = 10000;
static const size_t MM_DIM = 2;
static const int32_t INNER_AXIS = 1;
static const int32_t OUTER_AXIS = 2;
static const int64_t DIM_EQUAL_ONE = 1;
static const uint64_t SMALL_SHAPE_LIMIT = 524288UL;
static const uint32_t kDeqScaleMul = 0xFFFFE000;
static const uint64_t BASIC_BLOCK_SIZE_256 = 256;
static const uint64_t NUM_HALF = 2;
static const uint64_t FP32_HF32_DTYPE_SIZE = 4;
static const uint64_t BASIC_BLOCK_K_256_BYTE = 256;
static const uint64_t BASIC_BLOCK_SIZE_32 = 32;
static const uint64_t BASIC_BLOCK_SIZE_128 = 128;
static const double THRESHOLD = 0.7;
static const uint64_t CACHELINE = 512;
static const uint64_t NUM_TWO = 2;
static const int64_t DIMS_TWO = 2;
static const int64_t DIMS_THREE = 3;
static const int64_t HALF_ALIGN_UNIT = 256;
static const int64_t ALIGN_UNIT = 512;
static const int64_t M_DIM_SELF_IDX = 0;
static const int64_t K_DIM_SELF_IDX = 1;
static const int64_t N_DIM_SELF_IDX = 1;
static const int64_t ALIGN_UNIT_16 = 16;
static const int64_t ALIGN_UNIT_128 = 128;
static const int64_t MIN_V3_SHAPE_310 = 2048;
static const int64_t MAX_V3_SHAPE_310 = 5504;
static const int64_t SINGLE_CORE_SPLIT_K = 27392;
static const int64_t BLOCK_CUBE = 16;
static const int64_t BLOCK_BYTE_SIZE = 32;
static const uint64_t MB = 1024UL * 1024UL;
static const int64_t MAX_DIM_NUM = 4;
static const int64_t FP32_SPLIT_K_THRESHOLD = 8192;
using StrideIndexPairs = op::FVector<std::pair<int64_t, std::pair<int64_t, int64_t>>, MAX_DIM_NUM>;
static const std::set<std::vector<int>> transposeNeed = {{2, 0, 1}, {0, 3, 1, 2}};
static const std::set<std::vector<int>> transposeNoNeed = {{1, 0, 2}, {0, 2, 1, 3}};

static const std::initializer_list<op::DataType> V100_DTYPE_SUPPORT = {DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_WITHOUT_BF16 = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16};

static const std::map<op::DataType, uint64_t> ALIGN_UNIT_MAP = {
    {DataType::DT_FLOAT16, BASIC_BLOCK_SIZE_256},
    {DataType::DT_BF16, BASIC_BLOCK_SIZE_256},
    {DataType::DT_FLOAT, BASIC_BLOCK_SIZE_128}};

static inline bool CheckMathType(const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType)
{
    bool mat2Float = mat2->GetDataType() == DataType::DT_FLOAT;
    bool selfFloat = self->GetDataType() == DataType::DT_FLOAT;
    auto promoteType = (mat2Float || selfFloat) ? DataType::DT_FLOAT : self->GetDataType();
    return CheckCubeMathTypeForMm(promoteType, cubeMathType);
}

static inline bool CheckKEqual1Support(void)
{
    return (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201);
}

static inline bool CheckNpuArchIsSupportBf16(void)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    return (npuArch == NpuArch::DAV_2201) || (npuArch == NpuArch::DAV_3510);
}

static inline bool CheckMMV3NzNzNdSupport(MmOpInfo& mmOpInfo)
{
    // 当前切换场景不支持输入self、mat2混精度场景，且输入类型只支持fp16、bf16、fp32
    if (mmOpInfo.support_info.self_dtype != mmOpInfo.support_info.mat2_dtype ||
        ALIGN_UNIT_MAP.find(mmOpInfo.support_info.self_dtype) == ALIGN_UNIT_MAP.end()) {
        return false;
    }
    auto it = ALIGN_UNIT_MAP.find(mmOpInfo.support_info.self_dtype);
    uint64_t alignUnit = it->second;
    return (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) &&
           (mmOpInfo.shapeInfo.nDim % alignUnit == 0);
}

static bool CheckDtypeValid(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* bias, const aclTensor* out, int8_t cubeMathType)
{
    bool bf16flag = CheckNpuArchIsSupportBf16();
    if (bf16flag) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mat2, DTYPE_SUPPORT_LIST, return false);
        if (bias != nullptr) {
            OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST, return false);
        }
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_WITHOUT_BF16, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mat2, DTYPE_SUPPORT_LIST_WITHOUT_BF16, return false);
        if (bias != nullptr) {
            OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST_WITHOUT_BF16, return false);
        }
    }

    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (!bf16flag && (self->GetDataType() == op::DataType::DT_BF16 || mat2->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Bfloat16 is unsupported by the current SOC version [%s], now self is %s, mat2 is %s",
            op::ToString(socVersion).GetString(), op::ToString(self->GetDataType()).GetString(),
            op::ToString(mat2->GetDataType()).GetString());
        return false;
    }
    if (out != nullptr && cubeMathType == KEEP_DTYPE && out->GetDataType() == op::DataType::DT_FLOAT16 &&
        self->GetDataType() == op::DataType::DT_FLOAT) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Input tensor's dtype[DT_FLOAT] should be same with output's dtype[DT_FLOAT16].");
        return false;
    }
    // self和mat2的dtype不相等时，会做promote处理。
    bool dtype_match = self->GetDataType() == mat2->GetDataType();
    if (!dtype_match) {
        OP_LOGW(
            "Self's dtype [%s] and mat2's dtype [%s] are not equal. Promotion of Data Type will be applied",
            op::ToString(self->GetDataType()).GetString(), op::ToString(mat2->GetDataType()).GetString());
    }
    return true;
}

static bool CheckShapeValid(
    const aclTensor* self, const aclTensor* mat2, bool transposeX2 = false, bool isSlice = false)
{
    if (isSlice) {
        OP_CHECK_WRONG_DIMENSION(self, DIMS_THREE, return false);
    } else {
        OP_CHECK_WRONG_DIMENSION(self, DIMS_TWO, return false);
    }
    OP_CHECK_WRONG_DIMENSION(mat2, DIMS_TWO, return false);
    op::Shape mat2Shape = mat2->GetViewShape();
    op::Shape selfShape = self->GetViewShape();
    int64_t selfKDim = selfShape.GetDim(K_DIM_SELF_IDX); // self固定不转置
    int64_t mat2KDim = transposeX2 ? mat2Shape.GetDim(K_DIM_SELF_IDX) : mat2Shape.GetDim(M_DIM_SELF_IDX);
    if (isSlice) {
        selfKDim = selfShape.GetDim(OUTER_AXIS);
    }
    if (mat2KDim != selfKDim) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different, self Kdim[%ld], mat2 Kdim[%ld].",
            selfKDim, mat2KDim);
        return false;
    }
    return true;
}

static bool CheckShapeValidWithTrans(const aclTensor* self, const aclTensor* mat2, int64_t transSelf, int64_t transMat2)
{
    OP_CHECK_WRONG_DIMENSION(self, DIMS_TWO, return false);
    OP_CHECK_WRONG_DIMENSION(mat2, DIMS_TWO, return false);
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    auto selfKDim = (transSelf != 0LL) ? selfShape.GetDim(0) : selfShape.GetDim(1);
    auto mat2KDim = (transMat2 != 0LL) ? mat2Shape.GetDim(1) : mat2Shape.GetDim(0);
    if (selfKDim != mat2KDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different.");
        return false;
    }
    return true;
}

static const aclTensor* ProcessEmptyTensor(const aclTensor* self, const aclTensor* mat2, aclOpExecutor* executor)
{
    // 获取shape信息
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    op::Shape outShape = {selfShape.GetDim(0), mat2Shape.GetDim(1)};
    auto out = executor->AllocTensor(outShape, self->GetDataType());
    if (out->IsEmpty()) {
        OP_LOGI("Returning an empty tensor without actually doing calculation");
        return out;
    }
    op::FVector<int64_t> fillShape = GetShape(out);
    const aclTensor* dims = executor->ConvertToTensor(fillShape.data(), fillShape.size(), op::DataType::DT_INT64);
    aclIntArray* shapeArray = executor->AllocIntArray(fillShape.data(), fillShape.size());
    const aclScalar* valueScalar = executor->AllocScalar(0);
    const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, out->GetDataType());
    auto fillTensor = l0op::Fill(dims, valueTensor, shapeArray, executor);
    return fillTensor;
}

static const aclTensor* ProcessEmptyTensorWithTrans(
    const aclTensor* self, const aclTensor* mat2, int64_t transSelf, int64_t transMat2, aclOpExecutor* executor)
{
    // 获取shape信息
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    auto mDim = (transSelf != 0LL) ? selfShape.GetDim(1) : selfShape.GetDim(0);
    auto nDim = (transMat2 != 0LL) ? mat2Shape.GetDim(0) : mat2Shape.GetDim(1);
    op::Shape outShape = {mDim, nDim};
    auto out = executor->AllocTensor(outShape, self->GetDataType());
    if (out->IsEmpty()) {
        OP_LOGI("Returning an empty tensor without actually doing calculation");
        return out;
    }
    op::FVector<int64_t> fillShape = GetShape(out);
    const aclTensor* dims = executor->ConvertToTensor(fillShape.data(), fillShape.size(), op::DataType::DT_INT64);
    aclIntArray* shapeArray = executor->AllocIntArray(fillShape.data(), fillShape.size());
    const aclScalar* valueScalar = executor->AllocScalar(0);
    const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, out->GetDataType());
    auto fillTensor = l0op::Fill(dims, valueTensor, shapeArray, executor);
    return fillTensor;
}

static bool CheckSupportSingleSplitKFp16Bf16(
    const aclTensor* self, const aclTensor* mat2, const DataType selfDtype, const DataType mat2Dtype)
{
    // 判决门限
    // 1. 输入数据类型为fp16/bf16
    // 2. 在K轴非256字节对齐场景下，输入数据大小不超过INT32最大值
    // 3. K轴大于27392
    // 4. M、N中最大不超过K轴的一半
    bool supportCurrentArch = (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201);
    if (!supportCurrentArch) {
        return false;
    }
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();

    int64_t splitKMultiThres = 2;
    int64_t kDim = selfShape.GetDim(1);
    bool dtypeCorrect = ((selfDtype == DataType::DT_BF16) && (mat2Dtype == DataType::DT_BF16)) ||
                        ((selfDtype == DataType::DT_FLOAT16) && (mat2Dtype == DataType::DT_FLOAT16));
    int64_t checkSize = 0;

    int dtypeASize = ge::GetSizeByDataType(selfDtype);
    int dtypeBSize = ge::GetSizeByDataType(mat2Dtype);
    if ((kDim * dtypeASize) % 256 != 0) { // check if inner-dim is aligned to 256 byte
        checkSize += selfShape[selfShape.GetDimNum() - 1] * selfShape[selfShape.GetDimNum() - DIMS_TWO] * dtypeASize;
        checkSize += mat2Shape[mat2Shape.GetDimNum() - 1] * mat2Shape[mat2Shape.GetDimNum() - DIMS_TWO] * dtypeBSize;
    }

    bool checkMemSize = checkSize <= INT32_MAX;
    return dtypeCorrect && checkMemSize && kDim >= SINGLE_CORE_SPLIT_K &&
           kDim >= splitKMultiThres * std::max(selfShape.GetDim(0), mat2Shape.GetDim(1));
}

// 910/310P 支持fp16进 fp16/fp32出，非对齐case 只能NZ进出，对齐case支持ND进出
static aclnnStatus SetMatmulOpSupportInfo(
    const aclTensor* self, const aclTensor* mat2, MmOpInfo& mmOpInfo, int8_t cubeMathType)
{
    // 判断传入L0接口，用于计算的Dtype
    SetMmSupportDType(mmOpInfo, cubeMathType);

    // 判断当前Shape是否支持使用ND输入输出
    SetMmSupportFormat(self, mat2, mmOpInfo);

    TensorInfo SpTensor_sefl = {self, mmOpInfo.support_info.self_dtype, mmOpInfo.support_info.self_format};
    TensorInfo SpTensor_mat2 = {mat2, mmOpInfo.support_info.mat2_dtype, mmOpInfo.support_info.output_format};

    if (IsSplitk(&SpTensor_sefl, &SpTensor_mat2)) {
        mmOpInfo.supporSplitK = true;
        auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
        if (npuArch == NpuArch::DAV_2002) {
            mmOpInfo.support_info.output_dtype = DataType::DT_FLOAT;
        } else if (npuArch == NpuArch::DAV_1001) {
            mmOpInfo.support_info.output_format = Format::FORMAT_FRACTAL_NZ;
            mmOpInfo.support_info.output_dtype = DataType::DT_FLOAT;
        }
    }

    if (CheckSupportSingleSplitKFp16Bf16(self, mat2, mmOpInfo.support_info.self_dtype, mmOpInfo.support_info.mat2_dtype)) {
        OP_LOGI("Hit mat_mul_v3 ND fp16/bf16 single core splitK case channel.");
        mmOpInfo.support_info.output_format = Format::FORMAT_ND;
        mmOpInfo.support_info.self_format = Format::FORMAT_ND;
        mmOpInfo.support_info.mat2_format = Format::FORMAT_ND;
        mmOpInfo.supporSplitK = true;
    }

    // self=nd, mat2=nz不支持切K
    bool isNdNzIn =
        self->GetStorageFormat() == Format::FORMAT_ND && mat2->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ;
    mmOpInfo.support_info.mat2_format = isNdNzIn ? Format::FORMAT_FRACTAL_NZ : mmOpInfo.support_info.mat2_format;
    mmOpInfo.support_info.output_dtype =
        isNdNzIn ? mmOpInfo.support_info.mat2_dtype : mmOpInfo.support_info.output_dtype;
    return ACLNN_SUCCESS;
}

static MmOpInfo GetMatmulOpInfoWithTrans(
    const aclTensor* self, const aclTensor* mat2, int64_t transSelf, int64_t transMat2, int8_t cubeMathType)
{
    // 获取m、k、n轴的大小
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    int64_t mDim = (transSelf != 0LL)  ? selfShape.GetDim(1) : selfShape.GetDim(0);
    int64_t kDim = (transSelf != 0LL)  ? selfShape.GetDim(0) : selfShape.GetDim(1);
    int64_t nDim = (transMat2 != 0LL) ? mat2Shape.GetDim(0) : mat2Shape.GetDim(1);

    // Dtype和Format初始化
    MmOpInfo mmOpInfo;
    mmOpInfo.ori_info.self_dtype = self->GetDataType();
    mmOpInfo.ori_info.self_format = op::Format::FORMAT_ND;
    mmOpInfo.ori_info.mat2_dtype = mat2->GetDataType();
    mmOpInfo.ori_info.mat2_format = op::Format::FORMAT_ND;
    mmOpInfo.ori_info.output_dtype = self->GetDataType();
    mmOpInfo.ori_info.output_format = op::Format::FORMAT_ND;

    mmOpInfo.shapeInfo.kDim = kDim;
    mmOpInfo.shapeInfo.nDim = nDim;
    mmOpInfo.shapeInfo.mDim = mDim;

    mmOpInfo.shapeInfo.transposeX1 = (transSelf != 0LL) ? true : false;
    mmOpInfo.shapeInfo.transposeX2 = (transMat2 != 0LL) ? true : false;
    mmOpInfo.support_info = mmOpInfo.ori_info;
    // 如果允许降精度处理， 则开启HF32模式（0x40），否则采用默认模式; 后续此字段配置需要按照字段表进行配置
    mmOpInfo.enableHf32 = (cubeMathType == ALLOW_FP32_DOWN_PRECISION) || (cubeMathType == USE_HF32);
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    mmOpInfo.enableForceGrpAccForFp32 =
        cubeMathType == USE_FP32_ADD && (npuArch == NpuArch::DAV_2201 || npuArch == NpuArch::DAV_3510);
    mmOpInfo.opImplModeEnum = mmOpInfo.enableHf32 ? 0x40 : (mmOpInfo.enableForceGrpAccForFp32 ? 0x4 : 0x1);
    OP_LOGD(
        "opImplModeEnum=%ld, enableHf32=%d, enableForceGrpAccForFp32=%d cubeMathType=%d", mmOpInfo.opImplModeEnum,
        mmOpInfo.enableHf32, mmOpInfo.enableForceGrpAccForFp32, cubeMathType);

    SetMatmulOpSupportInfo(self, mat2, mmOpInfo, cubeMathType);
    GetMmInfo(mmOpInfo);
    return mmOpInfo;
}

static inline bool IsSplitKThenForbiddenNd2Nz(const uint64_t mDim, const uint64_t kDim, const uint64_t nDim,
                                              const bool transposeX1, const bool transposeX2) {
  constexpr uint64_t align128 = 128;
  constexpr uint64_t numTwo = 2;
  constexpr uint64_t smallMNsplitKThres = 15000;
  bool kIsEnoughMultiCore = kDim >= smallMNsplitKThres;
  bool mnIsNotEnoughCore = (std::ceil(mDim / align128) * std::ceil(nDim / align128) <
                            static_cast<int64_t>(GetCurrentPlatformInfo().GetCubeCoreNum() / numTwo));
  // M/N轴在内轴的场景切m/n不影响MTE2搬运效率，M/N可以切小保证多核能开启，属于cube_bound场景
  return kIsEnoughMultiCore && mnIsNotEnoughCore && !(!transposeX1 && transposeX2);
}

static bool CheckAscendCScenario(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const MmOpInfo& mmOpInfo, const bool transposeX1,
    const bool transposeX2)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if ((npuArch != NpuArch::DAV_2201 && npuArch != NpuArch::DAV_3510) ||
        (mmOpInfo.support_info.self_format != ge::FORMAT_ND)) {
        OP_LOGI("Not mat_mul_v3 case for unsupported npu arch or unsupported Format.");
        return false;
    }
    bool alwaysUseV3 = (npuArch == NpuArch::DAV_3510);
    return (alwaysUseV3 || Ops::NN::MmCheckHitV3Shape(x1, x2, bias, transposeX1, transposeX2,
                                                      mmOpInfo.support_info.mat2_format, mmOpInfo.supporSplitK));
}

static bool CheckAscendCScenario2(
    const aclTensor* x1, const aclTensor* x2, const MmOpInfo& mmOpInfo, const bool transposeX1, const bool transposeX2)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch != NpuArch::DAV_2002) {
        return false;
    }
    if (x1->GetDataType() != DataType::DT_FLOAT16 || x2->GetDataType() != DataType::DT_FLOAT16) {
        return false;
    }
    if (mmOpInfo.support_info.self_format != ge::FORMAT_ND ||
        mmOpInfo.support_info.mat2_format != ge::FORMAT_FRACTAL_NZ ||
        mmOpInfo.support_info.output_format != ge::FORMAT_ND) {
        return false;
    }
    op::Shape shapeX1 = x1->GetViewShape();
    op::Shape shapeX2 = x2->GetViewShape();
    int64_t mDim = transposeX1 ? shapeX1[shapeX1.GetDimNum() - 1] : shapeX1[shapeX1.GetDimNum() - 2];
    int64_t kDim = transposeX1 ? shapeX1[shapeX1.GetDimNum() - 2] : shapeX1[shapeX1.GetDimNum() - 1];
    int64_t nDim = transposeX2 ? shapeX2[shapeX2.GetDimNum() - 2] : shapeX2[shapeX2.GetDimNum() - 1];
    if (mDim > MAX_V3_SHAPE_310 || mDim < MIN_V3_SHAPE_310 || kDim > MAX_V3_SHAPE_310 || kDim < MIN_V3_SHAPE_310 ||
        nDim > MAX_V3_SHAPE_310 || nDim < MIN_V3_SHAPE_310) {
        return false;
    }
    return true;
}

static const aclTensor* GetGemmV3Op(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* c, bool transposeX1, bool transposeX2, bool enableHf32,
    aclOpExecutor* executor)
{
    OP_LOGI("Hit gemmv3 scenario.");
    const aclTensor* mmOut = l0op::GemmV3Nd(x1, x2, c, transposeX1, transposeX2, enableHf32, executor);
    return mmOut;
}

static const bool CheckMatmulV3Support(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, MmOpInfo& mmOpInfo, const bool transposeX1,
    const bool transposeX2, const int64_t opImplModeEnum)
{
    bool enableForceGrpAccForFp32 =
        opImplModeEnum == 0x4 && mmOpInfo.shapeInfo.kDim >= 2048 && mmOpInfo.ori_info.self_dtype == DataType::DT_FLOAT;
    return CheckAscendCScenario(x1, x2, bias, mmOpInfo, transposeX1, transposeX2) ||
           CheckAscendCScenario2(x1, x2, mmOpInfo, transposeX1, transposeX2) || enableForceGrpAccForFp32 ||
           (mmOpInfo.support_info.mat2_format == ge::FORMAT_FRACTAL_NZ && mmOpInfo.support_info.mat2_dtype == DataType::DT_FLOAT);
}

static const bool CheckSupportInfoFormatNzNzNd(MmOpInfo& mmOpInfo)
{
    return mmOpInfo.support_info.self_format == ge::FORMAT_FRACTAL_NZ &&
           mmOpInfo.support_info.mat2_format == ge::FORMAT_FRACTAL_NZ &&
           mmOpInfo.support_info.output_format == ge::FORMAT_ND;
}

static const aclTensor* GetMatMulOp(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, MmOpInfo& mmOpInfo, const bool transposeX1,
    const bool transposeX2, const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor* executor)
{
    if (CheckMatmulV3Support(x1, x2, bias, mmOpInfo, transposeX1, transposeX2, opImplModeEnum) ||
        (CheckMMV3NzNzNdSupport(mmOpInfo) && CheckSupportInfoFormatNzNzNd(mmOpInfo))) {
        OP_LOGI("Hit matmul_v3 scenario.");
        if (mmOpInfo.support_info.output_dtype == DataType::DT_FLOAT &&
            ((mmOpInfo.support_info.mat2_dtype == DataType::DT_FLOAT16 &&
              mmOpInfo.support_info.self_dtype == DataType::DT_FLOAT16) ||
             (mmOpInfo.support_info.mat2_dtype == DataType::DT_BF16 &&
              mmOpInfo.support_info.self_dtype == DataType::DT_BF16)) &&
            bias == nullptr) {
            if (CheckSupportInfoFormatNzNzNd(mmOpInfo)) {
                OP_LOGD("check SocVersion, call MatMulV3NzNzNdFp162Fp32.");
                x1 = l0op::ReFormat(x1, op::Format::FORMAT_FRACTAL_NZ);
                x2 = l0op::ReFormat(x2, op::Format::FORMAT_FRACTAL_NZ);

                const aclTensor* mmOut = l0op::MatMulV3NzNzNdFp162Fp32(
                    x1, x2, bias, transposeX1, transposeX2, offsetX, opImplModeEnum, executor);
                return mmOut;
            }

            const aclTensor* mmOut =
                l0op::MatMulV3NdFp162Fp32(x1, x2, bias, transposeX1, transposeX2, offsetX, opImplModeEnum, executor);
            return mmOut;
        }

        if (CheckSupportInfoFormatNzNzNd(mmOpInfo)) {
            OP_LOGD("check SocVersion, call MatMulV3NzNzNd.");
            x1 = l0op::ReFormat(x1, op::Format::FORMAT_FRACTAL_NZ);
            x2 = l0op::ReFormat(x2, op::Format::FORMAT_FRACTAL_NZ);

            const aclTensor* mmOut =
                l0op::MatMulV3NzNzNd(x1, x2, bias, transposeX1, transposeX2, offsetX, opImplModeEnum, executor);
            return mmOut;
        }

        const aclTensor* mmOut =
            l0op::MatMulV3Nd(x1, x2, bias, transposeX1, transposeX2, offsetX, opImplModeEnum, executor);
        return mmOut;
    } else if (
        mmOpInfo.support_info.output_dtype == DataType::DT_FLOAT &&
        ((mmOpInfo.support_info.mat2_dtype == DataType::DT_FLOAT16 &&
          mmOpInfo.support_info.self_dtype == DataType::DT_FLOAT16) ||
         (mmOpInfo.support_info.mat2_dtype == DataType::DT_BF16 &&
          mmOpInfo.support_info.self_dtype == DataType::DT_BF16)) &&
        bias == nullptr) {
        // This is Split K Mode; Check if MatMul using Nd in Nd Out
        const aclTensor* mmOut =
            (mmOpInfo.support_info.self_format == ge::FORMAT_ND &&
             mmOpInfo.support_info.output_format == ge::FORMAT_ND) ?
                l0op::MatMulNdFp162Fp32(
                    x1, x2, nullptr, nullptr, transposeX1, transposeX2, offsetX, opImplModeEnum, executor) :
                l0op::MatMulNzFp162Fp32(
                    x1, x2, nullptr, nullptr, transposeX1, transposeX2, offsetX, opImplModeEnum, executor);
        return mmOut;
    } else {
        if (mmOpInfo.support_info.self_format == ge::FORMAT_ND) {
            const aclTensor* mmOut =
                (mmOpInfo.support_info.mat2_format == ge::FORMAT_ND) ?
                    l0op::MatMulNd(x1, x2, bias, nullptr, transposeX1, transposeX2, offsetX, opImplModeEnum, executor) :
                    l0op::MatMulNdNz(
                        x1, x2, bias, nullptr, transposeX1, transposeX2, offsetX, opImplModeEnum, executor);
            return mmOut;
        } else {
            OP_LOGD("self format is not ND.");
            if (mmOpInfo.support_info.output_format == ge::FORMAT_ND) {
                OP_LOGD("Output format is ND, call MatMulNzNzNd.");
                const aclTensor* mmOut = l0op::MatMulNzNzNd(
                    x1, x2, bias, nullptr, transposeX1, transposeX2, offsetX, opImplModeEnum, executor);
                return mmOut;
            }
            const aclTensor* mmOut =
                l0op::MatMulNz(x1, x2, bias, nullptr, transposeX1, transposeX2, offsetX, opImplModeEnum, executor);
            return mmOut;
        }
    }
}

static inline int64_t ComputePadNum(int64_t kDim, int64_t dataSize)
{
    return CeilAlign(kDim, CeilDiv(ALIGN_UNIT, dataSize)) - kDim;
}

static inline const aclTensor* GetPadTensor(int64_t padNum, int64_t padDim, aclOpExecutor* executor)
{
    // pad: top bottom left right
    size_t dims = 4;
    std::vector<int64_t> padVec(dims, 0);

    padVec[padDim] = padNum;

    auto padArray = executor->AllocIntArray(padVec.data(), dims);
    if (padArray == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc padVec failed");
        return nullptr;
    }

    auto padTensor = executor->ConvertToTensor(padArray, DataType::DT_INT64);
    return padTensor;
}

static bool CheckStreamKSKTiling(MmOpInfo& mmOpInfo)
{
    uint64_t kAlign =
        static_cast<uint64_t>(CeilAlign(mmOpInfo.shapeInfo.kDim, static_cast<int64_t>(BASIC_BLOCK_SIZE_256)));
    uint64_t aiCoreCnt = std::max(uint64_t{1}, static_cast<uint64_t>(mmOpInfo.aiCoreCnt));
    uint64_t dtypeASize = std::max(uint64_t{1}, static_cast<uint64_t>(mmOpInfo.shapeInfo.dtypeASize));
    constexpr uint64_t STREAM_K_MAX_K_THRESHOLD = 2000000UL;
    // 如果dtype是fp32且k轴大于200万 则走基础模板来保证fp32的精度
    if (dtypeASize == FP32_HF32_DTYPE_SIZE && !mmOpInfo.enableHf32 &&
        static_cast<uint64_t>(mmOpInfo.shapeInfo.kDim) > STREAM_K_MAX_K_THRESHOLD) {
        return false;
    }
    // 判断k轴是否大于8192 / DtypeSize_, 小于就不走stream-k-sk
    constexpr uint64_t STREAM_K_MIN_K_THRESHOLD = 8192UL;
    uint64_t kThreshold =
        std::max(STREAM_K_MIN_K_THRESHOLD, aiCoreCnt * BASIC_BLOCK_K_256_BYTE) / dtypeASize;
    if (kAlign < kThreshold) {
        return false;
    }

    uint64_t alignValue = BASIC_BLOCK_SIZE_256;
    if (mmOpInfo.shapeInfo.dtypeASize == FP32_HF32_DTYPE_SIZE && !mmOpInfo.enableHf32) {
        alignValue = BASIC_BLOCK_SIZE_32; // 如果是Fp32 基本块判断要用32
    }
    // 判断mn是否需要已经能切32份及以上
    uint64_t mCnt = static_cast<uint64_t>(CeilDiv(mmOpInfo.shapeInfo.mDim, static_cast<int64_t>(alignValue)));
    uint64_t nCnt = static_cast<uint64_t>(CeilDiv(mmOpInfo.shapeInfo.nDim, static_cast<int64_t>(alignValue)));
    return !(mCnt * nCnt > aiCoreCnt / NUM_HALF);
}

static bool CheckStreamKDPSKTiling(MmOpInfo& mmOpInfo)
{   
    constexpr uint64_t STREAM_K_MIN_K_THRESHOLD = 8192UL;
    uint64_t aiCoreCnt = std::max(uint64_t{1}, static_cast<uint64_t>(mmOpInfo.aiCoreCnt));
    uint64_t dtypeASize = std::max(uint64_t{1}, static_cast<uint64_t>(mmOpInfo.shapeInfo.dtypeASize));
    uint64_t kThreshold = std::max(STREAM_K_MIN_K_THRESHOLD, aiCoreCnt * BASIC_BLOCK_SIZE_128) / dtypeASize;
    uint64_t kDim = static_cast<uint64_t>(mmOpInfo.shapeInfo.kDim);
    // 如果k轴小于32*256/DtypeSize_ 或 mn轴不是256对齐 或 输入是fp32类型，不走stream-k-dpsk
    if (mmOpInfo.shapeInfo.mDim % BASIC_BLOCK_SIZE_256 != 0UL ||
        mmOpInfo.shapeInfo.nDim % BASIC_BLOCK_SIZE_256 != 0UL || kDim < kThreshold ||
        (dtypeASize == FP32_HF32_DTYPE_SIZE && !mmOpInfo.enableHf32)) {
        return false;
    }
    // 如果mn用256切分的份数小于核数 或者 取余核数为0或大于一半的核数，则不使用stream-k-dpsk
    uint64_t mCnt = static_cast<uint64_t>(CeilDiv(mmOpInfo.shapeInfo.mDim, static_cast<int64_t>(BASIC_BLOCK_SIZE_256)));
    uint64_t nCnt = static_cast<uint64_t>(CeilDiv(mmOpInfo.shapeInfo.nDim, static_cast<int64_t>(BASIC_BLOCK_SIZE_256)));
    uint64_t tatalMNCnt = mCnt * nCnt;
    if ((tatalMNCnt < aiCoreCnt) || (tatalMNCnt % aiCoreCnt == 0UL) || (tatalMNCnt % aiCoreCnt > aiCoreCnt / NUM_TWO)) {
        return false;
    }
    return true;
}

// 判断shape是否支持GemmV3
static bool CheckShapeSupport(MmOpInfo& mmOpInfo)
{
    // 判断是否不走stream-k
    bool notSkTiling = !CheckStreamKSKTiling(mmOpInfo) && !CheckStreamKDPSKTiling(mmOpInfo);
    // 判断m和n是否大于等于512
    bool mnValid = mmOpInfo.shapeInfo.mDim >= static_cast<int64_t>(CACHELINE) &&
                   mmOpInfo.shapeInfo.nDim >= static_cast<int64_t>(CACHELINE);
    // 判断k是否大于256
    bool kValid = mmOpInfo.shapeInfo.kDim > static_cast<int64_t>(BASIC_BLOCK_SIZE_256);
    // 满足条件shape范围
    bool shapeAswt = notSkTiling && mnValid && kValid;
    if (!shapeAswt) {
        OP_LOGI("Not support this shape in gemmV3.");
        return false;
    }
    OP_LOGD("Check shape success in gemmV3.");
    return true;
}

static const aclTensor* HandleEmptyTensor(const aclTensor* self, const aclTensor* mat2, aclOpExecutor* executor, int8_t cubeMathType)
{
    auto emptyOut = ProcessEmptyTensor(self, mat2, executor);
    CHECK_RET(emptyOut != nullptr, nullptr);
    // output cast
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if ((npuArch == NpuArch::DAV_2201) && cubeMathType == FP16FP32_KEEP_DTYPE) {
        auto castOut = l0op::Cast(emptyOut, DataType::DT_FLOAT, executor);
        CHECK_RET(castOut != nullptr, nullptr);
        return castOut;
    }
    return emptyOut;
}

static bool IsUseNonContiguous(const aclTensor* tensor)
{
    // Only support 3510
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch != NpuArch::DAV_3510) {
        return false;
    }
    return !IsContiguous(tensor);
}

// Transpose场景下原始stride和shape满足：stride[i] = stride[i+1] * shape[i+1]
// StrideIndexPairs: {stride, {index, viewShape}}
static bool IsContiguousStride(StrideIndexPairs& strideIndexPairs)
{
    int64_t expectStride = 1;
    for (auto it = strideIndexPairs.rbegin(); it != strideIndexPairs.rend(); it++) {
        if (it->first != expectStride) {
            return false;
        }
        expectStride *= it->second.second;
    }
    return true;
}

static bool ValidateSliceParams(
    const op::Shape& simpleShape, const op::Strides& simpleStrides, int64_t viewOffset, int64_t storageSize)
{
    auto dimNum = static_cast<int64_t>(simpleStrides.size());
    op::FVector<int64_t> srcShape(dimNum, 1);
    int64_t shapeSize = 1;
    for (int64_t i = dimNum - 2; i >= 0; i--) {
        if (simpleStrides[i + 1] == 0) {
            return false;
        }
        srcShape[i + 1] = simpleStrides[i] / simpleStrides[i + 1];
        shapeSize *= srcShape[i + 1];
    }
    if (storageSize % shapeSize != 0) {
        return false;
    }
    srcShape[0] = storageSize / shapeSize;

    op::FVector<int64_t> offset(dimNum, 0);
    op::FVector<int64_t> size(dimNum, 0);
    // size即切片后每个维度的值 offset表示跳过的值
    for (int64_t i = 0; i < dimNum; i++) {
        offset[i] = viewOffset / simpleStrides[i];
        size[i] = simpleShape[i];
        viewOffset = viewOffset % simpleStrides[i];
    }
    // 1.切片后各个维度大小+offset不超过原始shape各维度
    for (int64_t i = 0; i < dimNum; i++) {
        if (srcShape[i] < offset[i] + size[i]) {
            return false;
        }
    }
    // 2.限制只支持倒数第二维切片
    for (int64_t i = dimNum - 1; i >= 0; i--) {
        if (srcShape[i] != simpleShape[i] && i != (dimNum - DIMS_TWO)) {
            return false;
        }
    }
    int64_t sliceM = simpleShape[dimNum - 2];
    // 3.当3D场景倒数第二维切片后维度大小为16的倍数或者因数
    return 16 % sliceM == 0;
}

static const aclTensor* SetTensorToNDFormat(const aclTensor* input)
{
    OP_LOGD("set tensor to ND format.");
    auto formatTensor = const_cast<aclTensor*>(input);
    if (input->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ &&
        input->GetStorageFormat() != op::Format::FORMAT_ND) {
        formatTensor->SetViewFormat(op::Format::FORMAT_ND);
        formatTensor->SetOriginalFormat(op::Format::FORMAT_ND);
        formatTensor->SetStorageFormat(op::Format::FORMAT_ND);
    }
    return formatTensor;
}

static aclnnStatus SetMatmulOpSupportFormat(
    const aclTensor* self, const aclTensor* mat2, MmOpInfo& mmOpInfo)
{
    // 判断当前Shape是否支持使用ND输入输出
    SetMmSupportFormat(self, mat2, mmOpInfo);

    TensorInfo SpTensor_sefl = {self, mmOpInfo.support_info.self_dtype, mmOpInfo.support_info.self_format};
    TensorInfo SpTensor_mat2 = {mat2, mmOpInfo.support_info.mat2_dtype, mmOpInfo.support_info.output_format};

    if (IsSplitk(&SpTensor_sefl, &SpTensor_mat2)) {
        mmOpInfo.supporSplitK = true;
        auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
        if (npuArch == NpuArch::DAV_2002) {
            mmOpInfo.support_info.output_dtype = DataType::DT_FLOAT;
        } else if (npuArch == NpuArch::DAV_1001) {
            mmOpInfo.support_info.output_format = Format::FORMAT_FRACTAL_NZ;
            mmOpInfo.support_info.output_dtype = DataType::DT_FLOAT;
        }
    }

    if (CheckSupportSingleSplitKFp16Bf16(self, mat2, mmOpInfo.support_info.self_dtype, mmOpInfo.support_info.mat2_dtype)) {
        OP_LOGI("Hit mat_mul_v3 ND fp16/bf16 single core splitK case channel.");
        mmOpInfo.support_info.output_format = Format::FORMAT_ND;
        mmOpInfo.support_info.self_format = Format::FORMAT_ND;
        mmOpInfo.support_info.mat2_format = Format::FORMAT_ND;
        mmOpInfo.supporSplitK = true;
    }

    // self=nd, mat2=nz不支持切K
    bool isNdNzIn =
        self->GetStorageFormat() == Format::FORMAT_ND && mat2->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ;
    mmOpInfo.support_info.mat2_format = isNdNzIn ? Format::FORMAT_FRACTAL_NZ : mmOpInfo.support_info.mat2_format;
    mmOpInfo.support_info.output_dtype =
        isNdNzIn ? mmOpInfo.support_info.mat2_dtype : mmOpInfo.support_info.output_dtype;
    return ACLNN_SUCCESS;
}

} // namespace

namespace Ops {
namespace NN {

// 非连续条件的shape范围限制
bool CheckNonContiguousShapeSupport(MmOpInfo& mmOpInfo)
{
    // 判断是否不走stream-k
    bool isSkTiling = CheckStreamKSKTiling(mmOpInfo) || CheckStreamKDPSKTiling(mmOpInfo);
    if (isSkTiling) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Not support this shape in sk or dp-sk tiling.");
        return false;
    }
    // 判断小于一轮
    uint64_t mCore = MathUtil::CeilDivision(mmOpInfo.shapeInfo.mDim, BASIC_BLOCK_SIZE_256);
    uint64_t nCore = MathUtil::CeilDivision(mmOpInfo.shapeInfo.nDim, BASIC_BLOCK_SIZE_256);
    if (mCore * nCore > static_cast<uint64_t>(mmOpInfo.aiCoreCnt)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mCnt[%lu] and nCnt[%lu] is not in matmulv3 basic shape range", mCore, nCore);
        return false;
    }
    // 非FP32大K
    if (mmOpInfo.ori_info.mat2_dtype == DataType::DT_FLOAT && !mmOpInfo.enableHf32 &&
        mmOpInfo.shapeInfo.kDim > FP32_SPLIT_K_THRESHOLD) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "fp32 big k is not supported.");
        return false;
    }
    OP_LOGI("Check tensor shape success.");
    return true;
}

bool CheckGemmV3WithAlphaBeta(const aclTensor* bias,
                              const aclTensor* self,
                              const aclTensor* mat2,
                              int8_t cubeMathType)
{
    // cubeMathType需要为USE_FP32_ADD
    if (cubeMathType != USE_FP32_ADD) {
        return false;
    }
    // 仅支持fp16、bf16输入
    if (!(bias->GetDataType() == op::DataType::DT_FLOAT16 && self->GetDataType() == op::DataType::DT_FLOAT16 &&
          mat2->GetDataType() == op::DataType::DT_FLOAT16) &&
        !(bias->GetDataType() == op::DataType::DT_BF16 && self->GetDataType() == op::DataType::DT_BF16 &&
          mat2->GetDataType() == op::DataType::DT_BF16)) {
        return false;
    }
    OP_LOGI("Check GemmV3WithAlphaBeta success.");
    return true;
}

/*
   判断是否满足左矩阵非连续Slice
*/
bool IsSliceNonContiguous(const aclTensor* self, const aclTensor* mat2)
{
    if (!IsUseNonContiguous(self)) {
        return false;
    }
    // 不支持NZ
    if (mat2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ ||
        self->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGI("Format NZ is not supported for slice.");
        return false;
    }
    // slice场景下，增加dtype判断，仅支持左右矩阵dtype相同
    if (self->GetDataType() != mat2->GetDataType()) {
        OP_LOGI("The data type of the self does not match the type of mat2 for slice");
        return false;
    }
    int64_t dimNum = self->GetViewShape().GetDimNum();
    if (dimNum != 3) { // only support 2D or 3D
        return false;
    }
    const auto& viewStrides = self->GetViewStrides();
    const auto& viewShape = self->GetViewShape();
    auto viewOffset = self->GetViewOffset();
    auto storageSize = self->GetStorageShape().GetShapeSize();
    // Validate view params
    if (dimNum != static_cast<int64_t>(viewStrides.size()) || storageSize <= 0) {
        return false;
    }
    // Validate viewStides
    auto lastStride = viewStrides[0];
    for (int64_t i = 1; i < dimNum; i++) {
        int64_t curStride = viewStrides[i];
        if (curStride == 0 || viewShape[i] == 1) { // Only support slice
            return false;
        }
        // stride严格按照逆序排序且能依次整除
        if (lastStride < curStride || lastStride % curStride != 0) {
            return false;
        }
        if (lastStride < curStride * (viewShape[i] - 1)) { // Not support overlap
            return false;
        }
        lastStride = curStride;
    }
    // Last dim view stride must be 1
    if (viewStrides[dimNum - 1] != 1) {
        return false;
    }
    // Validate slice params
    return ValidateSliceParams(viewShape, viewStrides, viewOffset, storageSize);
}

static inline string PrintIndex(const vector<int> idx)
{
    std::ostringstream oss;
    oss << "[";

    for (size_t i = 0; i < idx.size(); i++) {
        oss << idx[i];
        if (i < idx.size() - 1) {
            oss << ",";
        }
    }
    oss << "]";
    return oss.str();
}

/*
   判断Tensor是否满足非连续Transpose
*/
bool IsTransposeNonContiguous(const aclTensor* tensor, bool& isNeedSwapInnerTwoDim)
{
    int64_t dimNum = tensor->GetViewShape().GetDimNum();
    if (!IsUseNonContiguous(tensor) || dimNum != 3) { // only support 3D && NonContiguous
        return false;
    }
    const auto& viewStrides = tensor->GetViewStrides();
    const auto& viewShape = tensor->GetViewShape();
    auto storageSize = tensor->GetStorageShape().GetShapeSize();
    // Validate view params
    if (dimNum != static_cast<int64_t>(viewStrides.size()) || storageSize <= 0) {
        return false;
    }
    StrideIndexPairs strideIndexPairs;
    strideIndexPairs.reserve(dimNum);
    auto lastStride = INT64_MAX;
    bool isTranspose = false;
    for (int64_t i = 0; i < dimNum; i++) {
        int64_t curStride = viewStrides[i];
        if (curStride == 0 || viewShape[i] == 1) {
            return false;
        }
        if (lastStride < curStride) {
            isTranspose = true;
        }
        lastStride = curStride;
        strideIndexPairs.emplace_back(std::make_pair(curStride, std::make_pair(i, viewShape[i])));
    }
    if (!isTranspose) {
        return false;
    }
    // strides逆序排序
    std::sort(strideIndexPairs.rbegin(), strideIndexPairs.rend());
    if (!IsContiguousStride(strideIndexPairs)) {
        return false;
    }
    std::vector<int> indexs;
    for (auto it = strideIndexPairs.begin(); it != strideIndexPairs.end(); it++) {
        indexs.push_back(it->second.first);
    }
    // 业务限制：只允许中间2维转置(index = 0,1)
    // [1,0,2]不需要转置, [2,0,1]需要转置
    auto isNeedSwap = find(transposeNeed.begin(), transposeNeed.end(), indexs);
    auto isNoNeedSwap = find(transposeNoNeed.begin(), transposeNoNeed.end(), indexs);
    if (isNeedSwap == transposeNeed.end() && isNoNeedSwap == transposeNoNeed.end()) {
        OP_LOGI("Current indexs: %s , which is not supported in our scenario", PrintIndex(indexs).c_str());
        return false;
    }
    if (isNeedSwap != transposeNeed.end()) {
        isNeedSwapInnerTwoDim = true; // 如果场景2则需要转置
    }
    return true;
}

/*
   交换原始shape指定的两个维度 last和secondLast, 默认1和2表示交换dim-1和dim-2
*/
op::Shape SwapLastTwoDimValue(const op::Shape tensorShape, int64_t last, int64_t secondLast)
{
    op::Shape swapedShape = tensorShape;
    // last < secondLast
    if (last >= secondLast) {
        return swapedShape;
    }
    int64_t dimNum = tensorShape.GetDimNum();
    if (dimNum >= secondLast) {
        int64_t lastDim = tensorShape.GetDim(dimNum - last);
        swapedShape.SetDim(dimNum - last, tensorShape.GetDim(dimNum - secondLast));
        swapedShape.SetDim(dimNum - secondLast, lastDim);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimNum is not supported , which is %ld.", dimNum);
    }
    return swapedShape;
}

/*
    isSlice为true时self可能为2维或者3维
*/
MmOpInfo GetMatmulOpInfo(const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType, bool isSelfSlice)
{
    // 获取m、k、n轴的大小
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    int64_t selfDimNum = selfShape.GetDimNum();
    int64_t mDim = selfShape.GetDim(M_DIM_SELF_IDX);
    int64_t kDim = selfShape.GetDim(K_DIM_SELF_IDX);
    if (isSelfSlice && selfDimNum > DIMS_TWO) {
        mDim *= selfShape.GetDim(INNER_AXIS); // m = batch * sliceM
        kDim = selfShape.GetDim(OUTER_AXIS);
    }

    int64_t nDim = mat2Shape.GetDim(N_DIM_SELF_IDX);

    // Dtype和Format初始化
    MmOpInfo mmOpInfo;
    mmOpInfo.ori_info.self_dtype = self->GetDataType();
    mmOpInfo.ori_info.self_format = op::Format::FORMAT_ND;
    mmOpInfo.ori_info.mat2_dtype = mat2->GetDataType();
    mmOpInfo.ori_info.mat2_format = op::Format::FORMAT_ND;
    mmOpInfo.ori_info.output_dtype = self->GetDataType();
    if (FP16FP32_KEEP_DTYPE == cubeMathType) {
        mmOpInfo.ori_info.output_dtype = DataType::DT_FLOAT;
    }
    mmOpInfo.ori_info.output_format = op::Format::FORMAT_ND;

    mmOpInfo.shapeInfo.kDim = kDim;
    mmOpInfo.shapeInfo.nDim = nDim;
    mmOpInfo.shapeInfo.mDim = mDim;
    mmOpInfo.shapeInfo.transposeX1 = false;
    mmOpInfo.shapeInfo.transposeX2 = false;
    mmOpInfo.shapeInfo.dtypeASize = ge::GetSizeByDataType(self->GetDataType());
    mmOpInfo.shapeInfo.dtypeBSize = ge::GetSizeByDataType(mat2->GetDataType());
    OP_LOGD(
        "mDim=%ld, kDim=%ld, nDim=%ld, dtypeASize=%ld, dtypeBSize=%ld", mDim, kDim, nDim, mmOpInfo.shapeInfo.dtypeASize,
        mmOpInfo.shapeInfo.dtypeBSize);
    mmOpInfo.support_info = mmOpInfo.ori_info;

    // 不同芯片能力不同
    // 910 310P shape是否对齐
    // fp16 fp32 选择，910 vector支持fp32
    SetMatmulOpSupportInfo(self, mat2, mmOpInfo, cubeMathType);

    // 获取aicore数目
    mmOpInfo.aiCoreCnt = GetCurrentPlatformInfo().GetCubeCoreNum();

    bool inputFp32Flag = mmOpInfo.support_info.self_dtype == DataType::DT_FLOAT &&
                         mmOpInfo.support_info.mat2_dtype == DataType::DT_FLOAT;
    // 如果允许降精度处理， 则开启HF32模式（0x40），否则采用默认模式; 后续此字段配置需要按照字段表进行配置
    mmOpInfo.enableHf32 = (cubeMathType == ALLOW_FP32_DOWN_PRECISION) || (cubeMathType == USE_HF32);
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    mmOpInfo.enableForceGrpAccForFp32 =
        cubeMathType == USE_FP32_ADD && (npuArch == NpuArch::DAV_2201 || npuArch == NpuArch::DAV_3510);
    mmOpInfo.opImplModeEnum = mmOpInfo.enableHf32 ? 0x40 : (mmOpInfo.enableForceGrpAccForFp32 ? 0x4 : 0x1);
    OP_LOGD(
        "opImplModeEnum=%ld, enableHf32=%d, enableForceGrpAccForFp32=%d cubeMathType=%d, inputFp32Flag= %d", mmOpInfo.opImplModeEnum,
        mmOpInfo.enableHf32, mmOpInfo.enableForceGrpAccForFp32, cubeMathType, inputFp32Flag);
    // Log mm info
    GetMmInfo(mmOpInfo);
    return mmOpInfo;
}

std::shared_ptr<NpuArchMatMulRuleBase> BuildRule()
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if ((npuArch == NpuArch::DAV_2201) || (npuArch == NpuArch::DAV_3510)) {
        return std::make_shared<Dav2201MatMulRule>(npuArch);
    } else {
        return std::make_shared<DefaultMatMulRule>(npuArch);
    }
}

aclnnStatus CreateMatmulOpInfo(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* bias, const aclTensor* out, int8_t cubeMathType,
    MmOpInfo& mmOpInfo, bool isSelfSlice = false)
{
    // 获取m、k、n轴的大小
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    int64_t selfDimNum = selfShape.GetDimNum();
    int64_t mDim = selfShape.GetDim(M_DIM_SELF_IDX);
    int64_t kDim = selfShape.GetDim(K_DIM_SELF_IDX);
    if (isSelfSlice && selfDimNum > DIMS_TWO) {
        mDim *= selfShape.GetDim(INNER_AXIS); // m = batch * sliceM
        kDim = selfShape.GetDim(OUTER_AXIS);
    }
    int64_t nDim = mat2Shape.GetDim(N_DIM_SELF_IDX);

    // Format初始化
    mmOpInfo.ori_info.self_format = op::Format::FORMAT_ND;
    mmOpInfo.ori_info.mat2_format = op::Format::FORMAT_ND;
    mmOpInfo.ori_info.output_format = op::Format::FORMAT_ND;

    mmOpInfo.shapeInfo.kDim = kDim;
    mmOpInfo.shapeInfo.nDim = nDim;
    mmOpInfo.shapeInfo.mDim = mDim;
    mmOpInfo.shapeInfo.transposeX1 = false;
    mmOpInfo.shapeInfo.transposeX2 = false;
    mmOpInfo.shapeInfo.dtypeASize = ge::GetSizeByDataType(self->GetDataType());
    mmOpInfo.shapeInfo.dtypeBSize = ge::GetSizeByDataType(mat2->GetDataType());
    OP_LOGD(
        "mDim=%ld, kDim=%ld, nDim=%ld, dtypeASize=%ld, dtypeBSize=%ld", mDim, kDim, nDim, mmOpInfo.shapeInfo.dtypeASize,
        mmOpInfo.shapeInfo.dtypeBSize);

    // 解析当前规格matmulop支持的dtype能力
    std::shared_ptr<NpuArchMatMulRuleBase> archRule = BuildRule();
    aclnnStatus status = archRule -> PromoteDtype(self, mat2, bias, out, cubeMathType, mmOpInfo);
    CHECK_RET(status == ACLNN_SUCCESS, status);

    // 不同芯片能力不同
    // 1980 1951 shape是否对齐
    // fp16 fp32 选择，1980 vector支持fp32
    SetMatmulOpSupportFormat(self, mat2, mmOpInfo);

    // 获取aicore数目
    mmOpInfo.aiCoreCnt = GetCurrentPlatformInfo().GetCubeCoreNum();

    bool inputFp32Flag = mmOpInfo.support_info.self_dtype == DataType::DT_FLOAT &&
                         mmOpInfo.support_info.mat2_dtype == DataType::DT_FLOAT;
    // 如果允许降精度处理， 则开启HF32模式（0x40），否则采用默认模式; 后续此字段配置需要按照字段表进行配置
    mmOpInfo.enableHf32 = (cubeMathType == ALLOW_FP32_DOWN_PRECISION) || (cubeMathType == USE_HF32);
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    mmOpInfo.enableForceGrpAccForFp32 =
        cubeMathType == USE_FP32_ADD && (npuArch == NpuArch::DAV_2201 || npuArch == NpuArch::DAV_3510);
    mmOpInfo.opImplModeEnum = mmOpInfo.enableHf32 ? 0x40 : (mmOpInfo.enableForceGrpAccForFp32 ? 0x4 : 0x1);
    OP_LOGD(
        "opImplModeEnum=%ld, enableHf32=%d, enableForceGrpAccForFp32=%d cubeMathType=%d, inputFp32Flag= %d", mmOpInfo.opImplModeEnum,
        mmOpInfo.enableHf32, mmOpInfo.enableForceGrpAccForFp32, cubeMathType, inputFp32Flag);
    // Log mm info
    GetMmInfo(mmOpInfo);
    return ACLNN_SUCCESS;
}

bool ContiguousAndCast(
    const aclTensor*& contiguousInput, const aclTensor*& castOut, bool& transposeFlag, op::DataType dtype,
    aclOpExecutor* executor)
{
    auto contiguousOut = contiguousInput;
    if (IsTransposeLastTwoDims(contiguousInput)) {
        // Swap last two dim value
        contiguousOut = executor->CreateView(
            contiguousInput, SwapLastTwoDimValue(contiguousInput->GetViewShape(), INNER_AXIS, OUTER_AXIS),
            contiguousInput->GetViewOffset());
        transposeFlag = true;
    } else {
        contiguousOut = l0op::Contiguous(contiguousInput, executor);
    }
    CHECK_RET(contiguousOut != nullptr, false);

    // cast
    castOut = l0op::Cast(contiguousOut, dtype, executor);
    CHECK_RET(castOut != nullptr, false);
    return true;
}

bool ContiguousAndCastBias(
    const aclTensor*& contiguousInput, const aclTensor*& castOut, op::DataType dtype,
    aclOpExecutor* executor)
{
    auto contiguousOut = contiguousInput;

    contiguousOut = l0op::Contiguous(contiguousInput, executor);

    CHECK_RET(contiguousOut != nullptr, false);

    // cast
    castOut = l0op::Cast(contiguousOut, dtype, executor);
    CHECK_RET(castOut != nullptr, false);
    return true;
}

const aclTensor* ExecMmOp(const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType, aclOpExecutor* executor)
{
    return ExecMmOpWithBias(self, mat2, nullptr, cubeMathType, executor, false);
}

/*
   注意：虽然申明为const指针引用，但selfReshapeOutput和mat2ReshapeOutput会重新赋值修改指针指向新的const aclTensor
*/
int64_t ProcessSpecialCases(
    const aclTensor*& selfCastOut, const aclTensor*& mat2CastOut, MmOpInfo& mmOpInfo, const aclTensor*& bias,
    const aclTensor*& selfReshapeOutput, const aclTensor*& mat2ReshapeOutput, aclOpExecutor* executor, bool& ifKEqual1)
{
    ifKEqual1 = IfKEqual1(selfCastOut, mmOpInfo, mmOpInfo.shapeInfo.transposeX1, bias);
    if (mmOpInfo.support_info.self_dtype == DataType::DT_BF16 ||
        mmOpInfo.support_info.mat2_dtype == DataType::DT_BF16) {
        ifKEqual1 = ifKEqual1 && checkBF16SizeValid(mat2CastOut, mmOpInfo.shapeInfo.transposeX2) &&
                    CheckKEqual1Support() && checkBF16MMValid(selfCastOut, mat2CastOut, mmOpInfo.shapeInfo.transposeX2);
    }
    if (ifKEqual1) {
        aclnnStatus kEqual1SelfToMKRes =
            IfKEqual1SelfToMK(selfCastOut, selfReshapeOutput, mmOpInfo.shapeInfo.transposeX1, executor);
        CHECK_RET(kEqual1SelfToMKRes == ACLNN_SUCCESS, -1);
        aclnnStatus kEqual1Mat2ToKNRes =
            IfKEqual1Mat2ToKN(mat2CastOut, mat2ReshapeOutput, mmOpInfo.shapeInfo.transposeX2, executor);
        CHECK_RET(kEqual1Mat2ToKNRes == ACLNN_SUCCESS, -1);
        OP_LOGI("Hit MatMul or BatchMatmul k=1 scenario, trans matmul to mul to calculate");
    } else {
        aclnnStatus mEqual1SelfToMKRes = IfMEqual1SelfToMK(
            selfCastOut, selfReshapeOutput, mmOpInfo.support_info.self_format, mmOpInfo.shapeInfo.transposeX1,
            executor);
        CHECK_RET(mEqual1SelfToMKRes == ACLNN_SUCCESS, -1);
        aclnnStatus nEqual1Mat2ToNKRes = IfNEqual1Mat2ToNK(
            mat2CastOut, mat2ReshapeOutput, mmOpInfo.support_info.mat2_format, mmOpInfo.shapeInfo.transposeX2,
            executor);
        CHECK_RET(nEqual1Mat2ToNKRes == ACLNN_SUCCESS, -1);
    }
    return 0;
}

/*
计算MatMul的workSize， 内涵MatMul算子的构图流程
                  self            mat2
                   |               |
              contiguous       contiguous
                   |               |
                 cast             cast
                   |               |
                  pad             pad
                   |               |
                transpose      transpose
                   |               |
                transdata      transdata
                    \              /
                        matmul_op
                            |
                        transdata
                            |
                          cast
                            |
                         output

*/
const aclTensor* ExecMmOpWithBias(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* bias, int8_t cubeMathType, aclOpExecutor* executor,
    bool transposeX2)
{
    CHECK_RET(self != nullptr, nullptr);
    CHECK_RET(mat2 != nullptr, nullptr);
    CHECK_RET(CheckDtypeValid(self, mat2, bias, nullptr, cubeMathType), nullptr);
    // 左输入矩阵非连续
    bool isSelfSlice = IsSliceNonContiguous(self, mat2);
    CHECK_RET(CheckShapeValid(self, mat2, transposeX2, isSelfSlice), nullptr);
    CHECK_RET(CheckMathType(self, mat2, cubeMathType), nullptr);
    // 空Tensor处理逻辑
    if (self->IsEmpty() || mat2->IsEmpty()) {
        return HandleEmptyTensor(self, mat2, executor, cubeMathType);
    }
    OP_LOGI(
        "Origin storage shapes: self[%s], mat2[%s].", op::ToString(self->GetStorageShape()).GetString(),
        op::ToString(mat2->GetStorageShape()).GetString());

    // 解析当前规格matmulop支持的dtype、format能力
    MmOpInfo mmOpInfo = GetMatmulOpInfo(self, mat2, cubeMathType, isSelfSlice);
    // weightNZ转置属性刷新
    mmOpInfo.shapeInfo.transposeX2 = mmOpInfo.shapeInfo.transposeX2 || transposeX2;
    bool needFoldBatch = false;
    // 校验非连续Slice场景shape
    if (isSelfSlice && !CheckNonContiguousShapeSupport(mmOpInfo)) {
        OP_LOGI("Curruent shape is not supported for slice.");
        isSelfSlice = false;
        needFoldBatch = true;
    }
    // 非ND转ND
    auto selfCastOut = self;
    if (isSelfSlice) {
        // 刷新oriShape
        selfCastOut = executor->CreateView(
            self, self->GetViewShape(), self->GetStorageShape(), self->GetViewStrides(), self->GetViewOffset());
        CHECK_RET(selfCastOut != nullptr, nullptr);
        // 非ND修改format为ND
        selfCastOut = SetTensorToNDFormat(selfCastOut);
    } else {
        // 转连续
        bool selfCastRes = ContiguousAndCast(
            self, selfCastOut, mmOpInfo.shapeInfo.transposeX1, mmOpInfo.support_info.self_dtype, executor);
        CHECK_RET(selfCastRes, nullptr);
        // 再次合并batch和M
        if (needFoldBatch) {
            // Fold the batch into the first dimension
            auto dimTensor1 = selfCastOut->GetViewShape().GetDimNum();
            op::Shape shape{-1, selfCastOut->GetViewShape().GetDim(dimTensor1 - 1)};
            selfCastOut = l0op::Reshape(selfCastOut, shape, executor);
            CHECK_RET(selfCastOut != nullptr, nullptr);
            // 更新m n k
            mmOpInfo = GetMatmulOpInfo(self, mat2, cubeMathType, isSelfSlice);
        }
        // reformat为ND
        self = l0op::ReFormat(self, op::Format::FORMAT_ND);
        CHECK_RET(self != nullptr, nullptr);
        if (mat2->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) {
            OP_LOGI("mat2 StorageFormat not FORMAT_FRACTAL_NZ.");
            mat2 = l0op::ReFormat(mat2, op::Format::FORMAT_ND);
            CHECK_RET(mat2 != nullptr, nullptr);
        }
    }
    OP_LOGI("mat2 origin storage shape is  [%s].", op::ToString(mat2->GetStorageShape()).GetString());
    // bias非连续转连续以及转换dtype
    auto contiguousBias = bias;
    if (contiguousBias != nullptr) {
        contiguousBias = ContiguousBias(self, bias, executor);
        CHECK_RET(contiguousBias != nullptr, nullptr);
    }

    auto mat2CastOut = mat2;
    auto mat2StorageShape = mat2->GetStorageShape();
    bool mat2CastRes = ContiguousAndCast(
        mat2, mat2CastOut, mmOpInfo.shapeInfo.transposeX2, mmOpInfo.support_info.mat2_dtype, executor);
    CHECK_RET(mat2CastRes, nullptr);

    if (mat2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGI("mat2 GetStorageFormat FORMAT_FRACTAL_NZ.");
        aclTensor* mat2ShapeSet = const_cast<aclTensor*>(mat2CastOut);
        mat2ShapeSet->SetStorageShape(mat2StorageShape); // 对NZ的场景用原来的stroageShape刷新
    }
    OP_LOGI("mat2 storage shape is [%s].", op::ToString(mat2StorageShape).GetString());

    // k,m,n=1特殊场景
    auto selfReshapeOutput = selfCastOut;
    auto mat2ReshapeOutput = mat2CastOut;
    bool ifKEqual1 = false;
    if (!isSelfSlice) {
        CHECK_RET(
            ProcessSpecialCases(
                selfCastOut, mat2CastOut, mmOpInfo, contiguousBias, selfReshapeOutput, mat2ReshapeOutput, executor,
                ifKEqual1) != -1,
            nullptr);
    }

    auto selfTransdataOut = selfReshapeOutput;
    if (!isSelfSlice) {
        selfTransdataOut = l0op::TransData(selfReshapeOutput, mmOpInfo.support_info.self_format, 0, executor);
        CHECK_RET(selfTransdataOut != nullptr, nullptr);
    }
    OP_LOGI("Format of self is selfTransdataOut [%s].", op::ToString(selfTransdataOut->GetStorageShape()).GetString());

    auto mat2TransdataOut = l0op::TransData(mat2ReshapeOutput, mmOpInfo.support_info.mat2_format, 0, executor);
    CHECK_RET(mat2TransdataOut != nullptr, nullptr);
    OP_LOGI("Format of mat2 is mat2TransdataOut [%s].", op::ToString(mat2TransdataOut->GetStorageShape()).GetString());

    const aclTensor* mmOut = nullptr;
    if (isSelfSlice) {
        mmOut = GetMatMulOp(
            selfTransdataOut, mat2TransdataOut, contiguousBias, mmOpInfo, mmOpInfo.shapeInfo.transposeX1,
            mmOpInfo.shapeInfo.transposeX2, 0, mmOpInfo.opImplModeEnum, executor);
    } else if (ifKEqual1) {
        mmOut = l0op::Mul(selfTransdataOut, mat2TransdataOut, executor);
    } else {
        mmOut = GetMatMulOp(
            selfTransdataOut, mat2TransdataOut, contiguousBias, mmOpInfo, mmOpInfo.shapeInfo.transposeX1,
            mmOpInfo.shapeInfo.transposeX2, 0, mmOpInfo.opImplModeEnum, executor);
    }
    CHECK_RET(mmOut != nullptr, nullptr);
    auto mmTransdataOut = l0op::TransData(mmOut, mmOpInfo.ori_info.output_format, 0, executor);
    CHECK_RET(mmTransdataOut != nullptr, nullptr);
    // output cast
    auto castOut = l0op::Cast(mmTransdataOut, mmOpInfo.ori_info.output_dtype, executor);
    CHECK_RET(castOut != nullptr, nullptr);
    return castOut;
}

const aclTensor* MatmulCommonProcess (
    const aclTensor* self, const aclTensor* mat2, const aclTensor* bias, const aclTensor* out, const int8_t cubeMathType,
    MmOpInfo& mmOpInfo, aclOpExecutor* executor, bool transposeX2)
{
    /*
                  self            mat2
                   |               |
              contiguous       contiguous
                   |               |
                 cast             cast
                   |               |
                  pad             pad
                   |               |
                transpose      transpose
                   |               |
                transdata      transdata
                    \              /
                        matmul_op
                            |
                        transdata
                            |
                         output
  */

    // 左输入矩阵非连续
    bool isSelfSlice = IsSliceNonContiguous(self, mat2);
    CHECK_RET(CheckShapeValid(self, mat2, transposeX2, isSelfSlice), nullptr);

    // 空Tensor处理逻辑
    if (self->IsEmpty() || mat2->IsEmpty()) {
        return HandleEmptyTensor(self, mat2, executor, cubeMathType);
    }

    OP_LOGI(
        "Origin storage shapes: self[%s], mat2[%s].", op::ToString(self->GetStorageShape()).GetString(),
        op::ToString(mat2->GetStorageShape()).GetString());

    // 解析当前规格matmulop支持的dtype、format能力
    aclnnStatus result = CreateMatmulOpInfo(self, mat2, bias, out, cubeMathType, mmOpInfo, isSelfSlice);
    CHECK_RET(result == ACLNN_SUCCESS, nullptr);

    // weightNZ转置属性刷新
    mmOpInfo.shapeInfo.transposeX2 = mmOpInfo.shapeInfo.transposeX2 || transposeX2;
    bool needFoldBatch = false;
    // 校验非连续Slice场景shape
    if (isSelfSlice && !CheckNonContiguousShapeSupport(mmOpInfo)) {
        OP_LOGI("Curruent shape is not supported for slice.");
        isSelfSlice = false;
        needFoldBatch = true;
    }
    // 左输入非连续转连续
    auto selfCastOut = self;
    if (isSelfSlice) {
        // 刷新oriShape
        selfCastOut = executor->CreateView(
            self, self->GetViewShape(), self->GetStorageShape(), self->GetViewStrides(), self->GetViewOffset());
        CHECK_RET(selfCastOut != nullptr, nullptr);
        // 非ND修改format为ND
        selfCastOut = SetTensorToNDFormat(selfCastOut);
    } else {
        // 转连续
        bool selfCastRes = ContiguousAndCast(
            self, selfCastOut, mmOpInfo.shapeInfo.transposeX1, mmOpInfo.support_info.self_dtype, executor);
        CHECK_RET(selfCastRes, nullptr);
        // 再次合并batch和M
        if (needFoldBatch) {
            // Fold the batch into the first dimension
            auto dimTensor1 = selfCastOut->GetViewShape().GetDimNum();
            op::Shape shape{-1, selfCastOut->GetViewShape().GetDim(dimTensor1 - 1)};
            selfCastOut = l0op::Reshape(selfCastOut, shape, executor);
            CHECK_RET(selfCastOut != nullptr, nullptr);
            // 更新m n k
            aclnnStatus result = CreateMatmulOpInfo(selfCastOut, mat2, bias, out, cubeMathType, mmOpInfo, isSelfSlice);
            CHECK_RET(result == ACLNN_SUCCESS, nullptr);
        }
        // reformat为ND
        self = l0op::ReFormat(self, op::Format::FORMAT_ND);
        CHECK_RET(self != nullptr, nullptr);
        if (mat2->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) {
            OP_LOGI("mat2 StorageFormat not FORMAT_FRACTAL_NZ.");
            mat2 = l0op::ReFormat(mat2, op::Format::FORMAT_ND);
            CHECK_RET(mat2 != nullptr, nullptr);
        }
    }
    OP_LOGI("mat2 origin storage shape is  [%s].", op::ToString(mat2->GetStorageShape()).GetString());

    auto mat2CastOut = mat2;
    auto mat2StorageShape = mat2->GetStorageShape();
    bool mat2CastRes = ContiguousAndCast(
        mat2, mat2CastOut, mmOpInfo.shapeInfo.transposeX2, mmOpInfo.support_info.mat2_dtype, executor);
    CHECK_RET(mat2CastRes, nullptr);
    if (mat2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGI("mat2 GetStorageFormat FORMAT_FRACTAL_NZ.");
        aclTensor* mat2ShapeSet = const_cast<aclTensor*>(mat2CastOut);
        mat2ShapeSet->SetStorageShape(mat2StorageShape); // 对NZ的场景用原来的stroageShape刷新
    }
    OP_LOGI("mat2 storage shape is [%s].", op::ToString(mat2StorageShape).GetString());
    // bias非连续转连续以及转换dtype
    auto contiguousBias = bias;
    if (contiguousBias != nullptr) {
        bool biasCastRes = ContiguousAndCastBias(bias, contiguousBias, mmOpInfo.support_info.bias_dtype, executor);
        CHECK_RET(biasCastRes, nullptr);
    }

    // k,m,n=1特殊场景
    auto selfReshapeOutput = selfCastOut;
    auto mat2ReshapeOutput = mat2CastOut;
    bool ifKEqual1 = false;
    if (!isSelfSlice) {
        CHECK_RET(
            ProcessSpecialCases(
                selfCastOut, mat2CastOut, mmOpInfo, contiguousBias, selfReshapeOutput, mat2ReshapeOutput, executor, ifKEqual1) !=
                -1,
            nullptr);
    }

    auto selfTransdataOut = selfReshapeOutput;
    if (!isSelfSlice) {
        selfTransdataOut = l0op::TransData(selfReshapeOutput, mmOpInfo.support_info.self_format, 0, executor);
        CHECK_RET(selfTransdataOut != nullptr, nullptr);
    }
    OP_LOGI("Format of self is selfTransdataOut [%s].", op::ToString(selfTransdataOut->GetStorageShape()).GetString());
    // TransData
    auto mat2TransdataOut = l0op::TransData(mat2ReshapeOutput, mmOpInfo.support_info.mat2_format, 0, executor);
    CHECK_RET(mat2TransdataOut != nullptr, nullptr);
    OP_LOGI("Format of mat2 is mat2TransdataOut [%s].", op::ToString(mat2TransdataOut->GetStorageShape()).GetString());

    const aclTensor* mmOut = nullptr;
    if (isSelfSlice) {
        mmOut = GetMatMulOp(
            selfTransdataOut, mat2TransdataOut, contiguousBias, mmOpInfo, mmOpInfo.shapeInfo.transposeX1,
            mmOpInfo.shapeInfo.transposeX2, 0, mmOpInfo.opImplModeEnum, executor);
    } else if (ifKEqual1) {
        mmOut = l0op::Mul(selfTransdataOut, mat2TransdataOut, executor);
    } else {
        mmOut = GetMatMulOp(
            selfTransdataOut, mat2TransdataOut, contiguousBias, mmOpInfo, mmOpInfo.shapeInfo.transposeX1,
            mmOpInfo.shapeInfo.transposeX2, 0, mmOpInfo.opImplModeEnum, executor);
    }
    CHECK_RET(mmOut != nullptr, nullptr);
    auto mmTransdataOut = l0op::TransData(mmOut, mmOpInfo.ori_info.output_format, 0, executor);
    CHECK_RET(mmTransdataOut != nullptr, nullptr);

    return mmTransdataOut;
}

/*
计算MatMul，输入根据 transSelf和transMat2决定是否转置
*/
const aclTensor* ExecMmOpWithTrans(
    const aclTensor* self, const aclTensor* mat2, int64_t transSelf, int64_t transMat2, int8_t cubeMathType,
    aclOpExecutor* executor)
{
    CHECK_RET(CheckDtypeValid(self, mat2, nullptr, nullptr, cubeMathType), nullptr);
    CHECK_RET(CheckShapeValidWithTrans(self, mat2, transSelf, transMat2), nullptr);
    CHECK_RET(CheckMathType(self, mat2, cubeMathType), nullptr);
    // 空Tensor处理逻辑
    if (self->IsEmpty() || mat2->IsEmpty()) {
        auto emptyOut = ProcessEmptyTensorWithTrans(self, mat2, transSelf, transMat2, executor);
        CHECK_RET(emptyOut != nullptr, nullptr);
        return emptyOut;
    }

    // 内部只处理ND格式
    // reformat，全部转成ND
    self = l0op::ReFormat(self, op::Format::FORMAT_ND);
    CHECK_RET(self != nullptr, nullptr);
    mat2 = l0op::ReFormat(mat2, op::Format::FORMAT_ND);
    CHECK_RET(mat2 != nullptr, nullptr);

    // 解析当前规格matmulop支持的dtype、format能力
    MmOpInfo mmOpInfo = GetMatmulOpInfoWithTrans(self, mat2, transSelf, transMat2, cubeMathType);

    // 左输入 非连续转连续 cast
    auto contiguousSelf = l0op::Contiguous(self, executor);
    CHECK_RET(contiguousSelf != nullptr, nullptr);
    auto selfCastOut = l0op::Cast(contiguousSelf, mmOpInfo.support_info.self_dtype, executor);
    CHECK_RET(selfCastOut != nullptr, nullptr);

    // 右输入 非连续转连续 cast
    auto contiguousMat2 = l0op::Contiguous(mat2, executor);
    CHECK_RET(contiguousMat2 != nullptr, nullptr);
    auto mat2CastOut = l0op::Cast(contiguousMat2, mmOpInfo.support_info.mat2_dtype, executor);
    CHECK_RET(mat2CastOut != nullptr, nullptr);

    const aclTensor* bias = nullptr;
    auto selfReshapeOutput = selfCastOut;
    auto mat2ReshapeOutput = mat2CastOut;
    bool ifKEqual1 = IfKEqual1(selfCastOut, mmOpInfo, mmOpInfo.shapeInfo.transposeX1, bias);
    if (mmOpInfo.support_info.self_dtype == DataType::DT_BF16 ||
        mmOpInfo.support_info.mat2_dtype == DataType::DT_BF16) {
        ifKEqual1 = ifKEqual1 && checkBF16SizeValid(mat2CastOut, mmOpInfo.shapeInfo.transposeX2) &&
                    CheckKEqual1Support() && checkBF16MMValid(selfCastOut, mat2CastOut, mmOpInfo.shapeInfo.transposeX2);
    }
    if (ifKEqual1) {
        aclnnStatus kEqual1SelfToMKRes =
            IfKEqual1SelfToMK(selfCastOut, selfReshapeOutput, mmOpInfo.shapeInfo.transposeX1, executor);
        CHECK_RET(kEqual1SelfToMKRes == ACLNN_SUCCESS, nullptr);
        aclnnStatus kEqual1Mat2ToKNRes =
            IfKEqual1Mat2ToKN(mat2CastOut, mat2ReshapeOutput, mmOpInfo.shapeInfo.transposeX2, executor);
        CHECK_RET(kEqual1Mat2ToKNRes == ACLNN_SUCCESS, nullptr);
        OP_LOGI("Hit MatMul or BatchMatmul k=1 scenario, trans matmul to mul to calculate");
    } else {
        aclnnStatus mEqual1SelfToMKRes = IfMEqual1SelfToMK(
            selfCastOut, selfReshapeOutput, mmOpInfo.support_info.self_format, mmOpInfo.shapeInfo.transposeX1,
            executor);
        CHECK_RET(mEqual1SelfToMKRes == ACLNN_SUCCESS, nullptr);
        aclnnStatus nEqual1Mat2ToNKRes = IfNEqual1Mat2ToNK(
            mat2CastOut, mat2ReshapeOutput, mmOpInfo.support_info.mat2_format, mmOpInfo.shapeInfo.transposeX2,
            executor);
        CHECK_RET(nEqual1Mat2ToNKRes == ACLNN_SUCCESS, nullptr);
    }

    auto selfTransdataOut = l0op::TransData(selfReshapeOutput, mmOpInfo.support_info.self_format, 0, executor);
    CHECK_RET(selfTransdataOut != nullptr, nullptr);
    auto mat2TransdataOut = l0op::TransData(mat2ReshapeOutput, mmOpInfo.support_info.mat2_format, 0, executor);
    CHECK_RET(mat2TransdataOut != nullptr, nullptr);

    const aclTensor* mmOut = nullptr;
    if (ifKEqual1) {
        mmOut = l0op::Mul(selfTransdataOut, mat2TransdataOut, executor);
    } else {
        mmOut = GetMatMulOp(
            selfTransdataOut, mat2TransdataOut, nullptr, mmOpInfo, mmOpInfo.shapeInfo.transposeX1,
            mmOpInfo.shapeInfo.transposeX2, 0, mmOpInfo.opImplModeEnum, executor);
    }
    CHECK_RET(mmOut != nullptr, nullptr);

    auto mmTransdataOut = l0op::TransData(mmOut, mmOpInfo.ori_info.output_format, 0, executor);
    CHECK_RET(mmTransdataOut != nullptr, nullptr);

    // output cast
    auto castOut = l0op::Cast(mmTransdataOut, mmOpInfo.ori_info.output_dtype, executor);
    CHECK_RET(castOut != nullptr, nullptr);

    return castOut;
}

bool CheckGemmV3Support(const aclTensor* mat1, const aclTensor* mat2, MmOpInfo& mmOpInfo, int8_t cubeMathType)
{
    CHECK_RET(mat1 != nullptr, false);
    CHECK_RET(mat2 != nullptr, false);
    CHECK_RET(CheckDtypeValid(mat1, mat2, nullptr, nullptr, cubeMathType), false);
    CHECK_RET(CheckShapeValid(mat1, mat2), false);
    CHECK_RET(CheckMathType(mat1, mat2, cubeMathType), false);
    // 空Tensor不由gemmV3处理
    if (mat1->IsEmpty() || mat2->IsEmpty()) {
        OP_LOGI("mat1 or mat2 is empty, does not support GemmV3.");
        return false;
    }

    if (mat2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ ||
        mat1->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGI("mat1 or mat2 StorageFormat is FORMAT_FRACTAL_NZ, does not support GemmV3.");
        return false;
    }
    // 当前支持平台
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if ((npuArch != NpuArch::DAV_2201) && (npuArch != NpuArch::DAV_3510)) {
        OP_LOGI("Current npu arch does not support GemmV3.");
        return false;
    }

    if (npuArch == NpuArch::DAV_2201) {
        auto dtype_mat1 = mat1->GetDataType();
        auto dtype_mat2 = mat2->GetDataType();
        if (!((dtype_mat1 == DataType::DT_FLOAT16 && dtype_mat2 == DataType::DT_FLOAT16) ||
              (dtype_mat1 == DataType::DT_BF16 && dtype_mat2 == DataType::DT_BF16))) {
                return false;
              }
    }

    // 解析当前规格matmulop支持的dtype format能力
    mmOpInfo = GetMatmulOpInfo(mat1, mat2, cubeMathType);

    // 当前支持shape范围
    return CheckShapeSupport(mmOpInfo);
}

bool CheckGemmV3Support(const aclTensor* mat1, const aclTensor* mat2, const aclTensor* bias, const aclTensor* out, MmOpInfo& mmOpInfo, int8_t cubeMathType)
{
    CHECK_RET(CheckShapeValid(mat1, mat2), false);

    // 空Tensor不由gemmV3处理
    if (mat1->IsEmpty() || mat2->IsEmpty()) {
        OP_LOGI("mat1 or mat2 is empty, does not support GemmV3.");
        return false;
    }

    if (mat2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ ||
        mat1->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGI("mat1 or mat2 StorageFormat is FORMAT_FRACTAL_NZ, does not support GemmV3.");
        return false;
    }
    // 当前支持平台
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if ((npuArch != NpuArch::DAV_2201) && (npuArch != NpuArch::DAV_3510)) {
        OP_LOGI("Current npu arch does not support GemmV3.");
        return false;
    }

    if (npuArch == NpuArch::DAV_2201) {
        auto dtype_mat1 = mat1->GetDataType();
        auto dtype_mat2 = mat2->GetDataType();
        if (!((dtype_mat1 == DataType::DT_FLOAT16 && dtype_mat2 == DataType::DT_FLOAT16) ||
              (dtype_mat1 == DataType::DT_BF16 && dtype_mat2 == DataType::DT_BF16))) {
                return false;
              }
    }

    // 解析当前规格matmulop支持的dtype、format能力
    aclnnStatus result = CreateMatmulOpInfo(mat1, mat2, bias, out, cubeMathType, mmOpInfo, false);
    CHECK_RET(result == ACLNN_SUCCESS, false);

    // 当前支持shape范围
    return CheckShapeSupport(mmOpInfo);
}

const aclTensor* ExecGemmV3Op(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* c, MmOpInfo& mmOpInfo, aclOpExecutor* executor)
{
    OP_LOGD("Format of self orign is [%s].", op::ToString(self->GetStorageShape()).GetString());
    OP_LOGD("Format of mat2 orign is [%s].", op::ToString(mat2->GetStorageShape()).GetString());
    // gemmv3只处理ND格式，reformat全部转成ND
    self = l0op::ReFormat(self, op::Format::FORMAT_ND);
    CHECK_RET(self != nullptr, nullptr);

    mat2 = l0op::ReFormat(mat2, op::Format::FORMAT_ND);
    CHECK_RET(mat2 != nullptr, nullptr);

    OP_LOGI("Format of mat2 is [%s].", op::ToString(mat2->GetStorageShape()).GetString());
    // 左输入非连续转连续
    auto selfCastOut = self;
    bool selfCastRes = ContiguousAndCast(
        self, selfCastOut, mmOpInfo.shapeInfo.transposeX1, mmOpInfo.support_info.self_dtype, executor);
    CHECK_RET(selfCastRes, nullptr);

    // 右输入非连续转连续
    auto mat2CastOut = mat2;
    bool mat2CastRes = ContiguousAndCast(
        mat2, mat2CastOut, mmOpInfo.shapeInfo.transposeX2, mmOpInfo.support_info.mat2_dtype, executor);
    CHECK_RET(mat2CastRes, nullptr);

    // 输入c非连续转连续以及转换dtype
    auto contiguousC = c;
    if (contiguousC != nullptr) {
        contiguousC = ContiguousBias(self, c, executor);
        CHECK_RET(contiguousC != nullptr, nullptr);
    }

    // GEMMV3 output fp32
    mmOpInfo.ori_info.output_dtype = DataType::DT_FLOAT;

    // Invoke GemmV3 l0 api
    const aclTensor* mmOut = GetGemmV3Op(
        selfCastOut, mat2CastOut, contiguousC, mmOpInfo.shapeInfo.transposeX1, mmOpInfo.shapeInfo.transposeX2,
        mmOpInfo.enableHf32, executor);

    CHECK_RET(mmOut != nullptr, nullptr);

    // output cast
    auto castOut = l0op::Cast(mmOut, mmOpInfo.ori_info.output_dtype, executor);
    CHECK_RET(castOut != nullptr, nullptr);

    return castOut;
}

bool IsInputSupportFp32() {
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  if ((npuArch != NpuArch::DAV_2201) && (npuArch != NpuArch::DAV_3510)) {
    return false;
  }
  return true;
}

bool CheckBatchDimBroadcast(size_t batch1DimNum, size_t batch2DimNum, const op::Shape& batch1, const op::Shape& batch2) {
    size_t batchIndex = MM_DIM;
    while (batch1DimNum > batchIndex && batch2DimNum > batchIndex) {
        if (batch1[batch1DimNum - batchIndex - 1] != 1 && batch2[batch1DimNum - batchIndex - 1] != 1 &&
            batch1[batch1DimNum - batchIndex - 1] != batch2[batch1DimNum - batchIndex - 1]) {
            return false;
        }
        batchIndex++;
    }
    return true;
}

// bmm 相对于 mm 取坐标需偏移
int64_t GetOffSet(int64_t DimNum) {
  int64_t rightMove = 0;
  // bmm DimNum 为 3, mm DimNum 为 2 ，bmm需要相对于mm向后偏移一位取行列值，默认rightMove为 0
  rightMove = DimNum == 3 ? 1 : 0;
  return rightMove;
}

// 检查单Tensor是否为支持带bias的mm的dtype
static inline bool CheckDtypeSupport(const aclTensor *tensor) {
  if (!IsInputSupportFp32()) {
    auto iter = std::find(V100_DTYPE_SUPPORT.begin(), V100_DTYPE_SUPPORT.end(), tensor->GetDataType());
    return iter != V100_DTYPE_SUPPORT.end();
  }
  auto iter = std::find(DTYPE_SUPPORT_LIST.begin(), DTYPE_SUPPORT_LIST.end(), tensor->GetDataType());
  return iter != DTYPE_SUPPORT_LIST.end();
}

// 检查是否为支持带bias的mm的dtype
static inline bool CheckDtypeSupportBias(const aclTensor *self, const aclTensor *mat1, const aclTensor *mat2) {
  bool matMulDtypeCorrect = CheckDtypeSupport(mat1) && CheckDtypeSupport(mat2);
  if (mat1->GetDataType() == DataType::DT_BF16) {
    return matMulDtypeCorrect &&
           (self->GetDataType() == DataType::DT_BF16 || self->GetDataType() == DataType::DT_FLOAT);
  }
  return CheckDtypeSupport(self) && matMulDtypeCorrect;
}

// 如果beta==1 && alpha == 1 && self.shape[0] == mat2.shape[1] && 不属于切k，直接走matmul的bias模式
bool NeedToConvertBias(const aclTensor *self, const aclTensor *mat1, const aclTensor *mat2,
                       const aclScalar *beta, const aclScalar *alpha) {
  int64_t mat1DimNum = static_cast<int64_t>(mat1->GetViewShape().GetDimNum());
  // rightMove to distinguish different shape of mm and bmm
  int64_t rightMove = 0;
  rightMove = GetOffSet(mat1DimNum);

  TensorInfo Tensor_matl = {mat1, mat1->GetDataType(), Format::FORMAT_ND};
  TensorInfo Tensor_mat2 = {mat2, mat2->GetDataType(), Format::FORMAT_ND};

  bool isSplitK = false;
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  if ((npuArch != NpuArch::DAV_2201) && (npuArch != NpuArch::DAV_3510)) {
    isSplitK = IsSplitk(&Tensor_matl, &Tensor_mat2);;
  }
  op::Shape selfShape = self->GetViewShape();
  op::Shape mat2Shape = mat2->GetViewShape();
  int64_t selfDimNum = static_cast<int64_t>(selfShape.GetDimNum());
  bool canBeBiasFlag = false;
  // bmm (DimNum==3) only apply the case of batch == 1
  bool batchIsOne = !(mat1DimNum == 3 && mat1->GetViewShape().GetDim(0) != 1);

  if (selfDimNum == 1) {
    canBeBiasFlag = (mat2->GetViewShape().GetDim(1 + rightMove) == self->GetViewShape().GetDim(0)) &&
                     CheckDtypeSupportBias(self, mat1, mat2) && batchIsOne;
    // When input tensor is a 2 dimentional tensor
  } else if (selfDimNum == 2) {
    canBeBiasFlag = (selfShape.GetDim(0) == 1) && (selfShape.GetDim(1) == mat2Shape.GetDim(1 + rightMove)) &&
                     CheckDtypeSupportBias(self, mat1, mat2) && batchIsOne;
  }
  OP_LOGI("Current Shape's canBeBiasFlag = %ld", static_cast<int64_t>(canBeBiasFlag));
  return (std::abs(alpha->ToFloat() - 1.0f) <= std::numeric_limits<float>::epsilon()) &&
         (std::abs(beta->ToFloat() - 1.0f) <= std::numeric_limits<float>::epsilon()) &&
         !isSplitK && canBeBiasFlag;
}

// Nz fp16 in fp32 out experimental rules
bool GetNzSplitKFlag(const aclTensor *self, const aclTensor *mat2, const Format selfSuppFormat, const Format outSuppFormat) {
  if ((selfSuppFormat == Format::FORMAT_ND) && (outSuppFormat == Format::FORMAT_ND)) {
    return true;
  }
  op::Shape selfShape = self->GetViewShape();
  op::Shape mat2Shape = mat2->GetViewShape();
  int64_t selfDimNum = static_cast<int64_t>(selfShape.GetDimNum());
  // rightMove to distinguish different shape of mm and bmm
  int64_t rightMove = 0;
  rightMove = GetOffSet(selfDimNum);

  int64_t m = selfShape.GetDim(rightMove);
  int64_t k = selfShape.GetDim(rightMove + 1);
  int64_t n = mat2Shape.GetDim(rightMove + 1);
  bool mn_multi = m > n ? m < (MN_MULTI * n) : n < (MN_MULTI * m);
  return (m * n * k < MKN_MAX) && mn_multi;
}

bool IsSplitk(const TensorInfo* self, const TensorInfo* mat2) {
  op::Shape selfShape = self->tensor->GetViewShape();
  op::Shape mat2Shape = mat2->tensor->GetViewShape();
  int64_t selfDimNum = static_cast<int64_t>(selfShape.GetDimNum());
  // rightMove to distinguish different shape of mm and bmm
  int64_t rightMove = 0;
  rightMove = GetOffSet(selfDimNum);
  bool NzSplitKFlag = true;
  // only apply on mm now
  if (!rightMove) {
    NzSplitKFlag = GetNzSplitKFlag(self->tensor, mat2->tensor, self->format, mat2->format);
  }

  int64_t k_dim = selfShape.GetDim(1 + rightMove);
  bool dtype_correct = (self->dataType == DataType::DT_FLOAT16) && (mat2->dataType == DataType::DT_FLOAT16);
  return dtype_correct && k_dim >= SPLIT_K_MULTI * std::max(selfShape.GetDim(rightMove), mat2Shape.GetDim(1 + rightMove)) && NzSplitKFlag;
}

bool IsFormatSupportNd(const aclTensor *self, const aclTensor *mat2) {
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  if (npuArch == NpuArch::DAV_3510) {
    return true;
  }
  if (npuArch != NpuArch::DAV_2201) {
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    int64_t dimNum = selfShape.GetDimNum();
    auto isAligin = [selfShape, mat2Shape, dimNum]() {
      return (!(static_cast<uint64_t>(selfShape.GetDim(dimNum - 2)) & 0x0000000F)) &&
             (!(static_cast<uint64_t>(selfShape.GetDim(dimNum - 1)) & 0x0000000F)) &&
             (!(static_cast<uint64_t>(mat2Shape.GetDim(dimNum - 2)) & 0x0000000F)) &&
             (!(static_cast<uint64_t>(mat2Shape.GetDim(dimNum - 1)) & 0x0000000F));
    };
    if (isAligin() && self->GetDataType() == op::DataType::DT_FLOAT16) {
      return true;
    }
    return false;
  }
  if ((self->GetDataType() == DataType::DT_FLOAT16 && mat2->GetDataType() == DataType::DT_FLOAT16) ||
      (self->GetDataType() == DataType::DT_BF16 && mat2->GetDataType() == DataType::DT_BF16)) {
    return IsNdToNzOnTheFly(self, mat2);
  }
  return true;
}

bool IsSupportNzNzNd(const aclTensor* self, const aclTensor* mat2) {
  op::Shape selfShape = self->GetViewShape();
  op::Shape mat2Shape = mat2->GetViewShape();
  int64_t dimNum = selfShape.GetDimNum();
  auto isNAligin = [mat2Shape, dimNum]() { return ((static_cast<uint64_t>(mat2Shape.GetDim(dimNum - 1)) & 0x0000000F)
                   == 0); };
  if (isNAligin() && self->GetDataType() == op::DataType::DT_FLOAT16) {
    return true;
  }
  return false;
}

bool IsNdToNzOnTheFly(const aclTensor *self, const aclTensor *mat2) {
  uint64_t kInnerAxisMinLimit = 128U;
  uint64_t kInnerAxisMaxLimit = 65535U;
  uint64_t kAxisLengthOne = 1U;
  // 如果self或mat2的维度数量小于2，则不符合判断是否16对齐的条件，返回失败
  if (self->GetViewShape().GetDimNum() < 2 || mat2->GetViewShape().GetDimNum() < 2) {
    return false;
  }
  bool isTransposeSelf = IsTransposeLastTwoDims(self);
  bool isTransposeMat2 = IsTransposeLastTwoDims(mat2);
  uint64_t selfInnerAxis = isTransposeSelf ?
                             static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 2)) :
                             static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 1));
  uint64_t mat2InnerAxis = isTransposeMat2 ?
                             static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 2)) :
                             static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 1));

  uint64_t selfOuterAxis = isTransposeSelf ?
                             static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 1)) :
                             static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 2));
  uint64_t mat2OuterAxis = isTransposeMat2 ?
                             static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 1)) :
                             static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 2));
  uint64_t mAxis = static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 2)); //倒数第2维
  uint64_t kAxis = static_cast<uint64_t>(self->GetViewShape().GetDim(self->GetViewShape().GetDimNum() - 1));
  uint64_t nAxis = static_cast<uint64_t>(mat2->GetViewShape().GetDim(mat2->GetViewShape().GetDimNum() - 1));
  if (selfInnerAxis * selfOuterAxis <= kInnerAxisMaxLimit &&
      mat2InnerAxis * mat2OuterAxis <= kInnerAxisMaxLimit) {
    // too small tensor size
    return true;
  }
  OP_LOGD("Check IsNdToNzOnTheFly, if k=1 scenerio then remains ND.");
  if (kAxis == kAxisLengthOne) {
    return true;
  }

  if (IsSplitKThenForbiddenNd2Nz(mAxis, kAxis, nAxis, isTransposeSelf, isTransposeMat2)) {
    OP_LOGD("Hit small mn multi split k.");
    return true;
  }

  return ((selfInnerAxis >= kInnerAxisMinLimit && selfInnerAxis <= kInnerAxisMaxLimit) ||
          (selfInnerAxis < kInnerAxisMinLimit && ((selfInnerAxis & 0xF) == 0))) &&
          ((mat2InnerAxis >= kInnerAxisMinLimit && mat2InnerAxis <= kInnerAxisMaxLimit) ||
          (mat2InnerAxis < kInnerAxisMinLimit && ((mat2InnerAxis & 0xF) == 0)));
}

bool IsTransposeLastTwoDims(const aclTensor *tensor) {
  // 当输入tensor的shape小于2或者大于6的时候，返回错误
  if (tensor->GetViewShape().GetDimNum() < 2 || tensor->GetViewShape().GetDimNum() > 6) {
    return false;
  }
  int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
  int64_t dim2 = tensor->GetViewShape().GetDimNum() - 2;
  // BMM 场景下，Batch维度的stride需要等于 N, D 的乘积
  if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
    int64_t tmpNxD = tensor->GetViewShape().GetDim(dim1) * tensor->GetViewShape().GetDim(dim2);
    // 多batch连续，3是batch索引
    for (int64_t batchDim = tensor->GetViewShape().GetDimNum() - 3; batchDim >= 0; batchDim--) {
    if (tensor->GetViewStrides()[batchDim] != tmpNxD) {
        return false;
      }
      tmpNxD *= tensor->GetViewShape().GetDim(batchDim);
    }
    if (tensor->GetViewShape().GetDim(dim1) == 1 && tensor->GetViewShape().GetDim(dim2) == 1) {
      return false;
    }
    return true;
  }
  return false;
}

aclnnStatus SetMmSupportDType(MmOpInfo &mmOpInfo, int8_t cubeMathType) {
  bool dtypeMismatch = mmOpInfo.ori_info.self_dtype != mmOpInfo.ori_info.mat2_dtype;
  bool tensorFloat = mmOpInfo.ori_info.self_dtype == DataType::DT_FLOAT ||
                     mmOpInfo.ori_info.mat2_dtype == DataType::DT_FLOAT;
  bool tensorBfloat16 = mmOpInfo.ori_info.self_dtype == DataType::DT_BF16 ||
                        mmOpInfo.ori_info.mat2_dtype == DataType::DT_BF16;

  bool lowPrecisionInput =
      (mmOpInfo.ori_info.self_dtype == DataType::DT_BF16 && mmOpInfo.ori_info.mat2_dtype == DataType::DT_BF16) ||
      (mmOpInfo.ori_info.self_dtype == DataType::DT_FLOAT16 && mmOpInfo.ori_info.mat2_dtype == DataType::DT_FLOAT16);
  if (!IsInputSupportFp32()) {
    mmOpInfo.support_info.self_dtype = DataType::DT_FLOAT16;
    mmOpInfo.support_info.mat2_dtype = DataType::DT_FLOAT16;
  } else if (IsInputSupportFp32() && cubeMathType == USE_FP16 && (!tensorBfloat16)) {
    // FP16
    mmOpInfo.support_info.self_dtype = DataType::DT_FLOAT16;
    mmOpInfo.support_info.mat2_dtype = DataType::DT_FLOAT16;
    mmOpInfo.support_info.output_dtype = DataType::DT_FLOAT16;
  } else if (IsInputSupportFp32() && dtypeMismatch && (tensorFloat || tensorBfloat16)) {
    // BF16或者存在FP32输入则全部dtype统一到FP32
    mmOpInfo.support_info.self_dtype = DataType::DT_FLOAT;
    mmOpInfo.support_info.mat2_dtype = DataType::DT_FLOAT;
    mmOpInfo.support_info.output_dtype = DataType::DT_FLOAT;
  } else if (IsInputSupportFp32() && cubeMathType == USE_HIGH_PREC_MODE && lowPrecisionInput) {
    mmOpInfo.support_info.output_dtype = DataType::DT_FLOAT;
  }
  return ACLNN_SUCCESS;
}

aclnnStatus SetMmSupportFormat(const aclTensor* self, const aclTensor* mat2, MmOpInfo& mmOpInfo) {
  if (IsFormatSupportNd(self, mat2)) {
    OP_LOGD("Matmul support NDNDND");
    mmOpInfo.support_info.output_format = Format::FORMAT_ND;
    mmOpInfo.support_info.self_format = Format::FORMAT_ND;
    mmOpInfo.support_info.mat2_format = Format::FORMAT_ND;
  } else {
    OP_LOGD("Matmul do not support NDNDND");
    // if Dav2002 and n%16==0
    bool isDav2002 = (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2002);
    if (IsSupportNzNzNd(self, mat2) && isDav2002) {
      mmOpInfo.support_info.output_format = Format::FORMAT_ND;
      mmOpInfo.support_info.self_format = Format::FORMAT_FRACTAL_NZ;
      mmOpInfo.support_info.mat2_format = Format::FORMAT_FRACTAL_NZ;
      return ACLNN_SUCCESS;
    }

    if (CheckMMV3NzNzNdSupport(mmOpInfo)) {
      mmOpInfo.support_info.output_format = Format::FORMAT_ND;
      mmOpInfo.support_info.self_format = Format::FORMAT_FRACTAL_NZ;
      mmOpInfo.support_info.mat2_format = Format::FORMAT_FRACTAL_NZ;
      return ACLNN_SUCCESS;
    }

    mmOpInfo.support_info.output_format = Format::FORMAT_FRACTAL_NZ;
    mmOpInfo.support_info.self_format = Format::FORMAT_FRACTAL_NZ;
    mmOpInfo.support_info.mat2_format = Format::FORMAT_FRACTAL_NZ;
  }
  return ACLNN_SUCCESS;
}

aclnnStatus GetMmInfo(MmOpInfo mmOpInfo) {
  OP_LOGI(
    "Self tensor input's ori dtype = %s and format = %s; Mat2 tensor input's ori dtype = %s and format = %s;"
    "Output tensor's ori dtype = %s and ori format = %s;"
    "Self tensor input's Npu dtype = %s and Npu format = %s; Mat2 tensor input's Npu dtype = %s and Npuformat = %s;"
    "Output tensor's Npu dtype = %s and Npu format = %s.",
    op::ToString(mmOpInfo.ori_info.self_dtype).GetString(),
    op::ToString(mmOpInfo.ori_info.self_format).GetString(),
    op::ToString(mmOpInfo.ori_info.mat2_dtype).GetString(),
    op::ToString(mmOpInfo.ori_info.mat2_format).GetString(),
    op::ToString(mmOpInfo.ori_info.output_dtype).GetString(),
    op::ToString(mmOpInfo.ori_info.output_format).GetString(),
    op::ToString(mmOpInfo.support_info.self_dtype).GetString(),
    op::ToString(mmOpInfo.support_info.self_format).GetString(),
    op::ToString(mmOpInfo.support_info.mat2_dtype).GetString(),
    op::ToString(mmOpInfo.support_info.mat2_format).GetString(),
    op::ToString(mmOpInfo.support_info.output_dtype).GetString(),
    op::ToString(mmOpInfo.support_info.output_format).GetString());
  return ACLNN_SUCCESS;
}

aclIntArray* NeedTransPerm(const aclTensor *x, aclOpExecutor *executor) {
  op::Shape shape = x->GetViewShape();
  int64_t dimSize = x->GetViewShape().GetDimNum();
  std::vector<int64_t> valuePerm(dimSize, 0);
  for (int64_t i = 0; i < dimSize; i++) {
    valuePerm[i] = shape[i];
  }
  std::swap(valuePerm[dimSize - INNER_AXIS], valuePerm[dimSize - OUTER_AXIS]);
  return executor->AllocIntArray(valuePerm.data(), dimSize);
}

bool checkBF16SizeValid(const aclTensor *&mat2, const bool &transX2Flag) {
  //校验N轴是否在优化范围内
  int64_t nDimNumWhenNoTrans = mat2->GetViewShape().GetDimNum() - INNER_AXIS;
  int64_t nDimNumWhenTrans = mat2->GetViewShape().GetDimNum() - OUTER_AXIS;
  int64_t nDim = transX2Flag ? mat2->GetViewShape().GetDim(nDimNumWhenTrans) :
                 mat2->GetViewShape().GetDim(nDimNumWhenNoTrans);
  if (nDim > N_KEQAL1_LIMIT) {
    return false;
  }
  return true;
}

bool checkBF16MMValid(const aclTensor *&self, const aclTensor *&mat2, const bool &transX2Flag) {
  //校验MN轴是否在优化范围内
  int64_t mDimNumWhenNoTrans = self->GetViewShape().GetDimNum() - OUTER_AXIS;
  int64_t mDimNumWhenTrans = self->GetViewShape().GetDimNum() - INNER_AXIS;
  int64_t mDim = transX2Flag ? self->GetViewShape().GetDim(mDimNumWhenTrans) :
                 self->GetViewShape().GetDim(mDimNumWhenNoTrans);
  int64_t nDimNumWhenNoTrans = mat2->GetViewShape().GetDimNum() - INNER_AXIS;
  int64_t nDimNumWhenTrans = mat2->GetViewShape().GetDimNum() - OUTER_AXIS;
  int64_t nDim = transX2Flag ? mat2->GetViewShape().GetDim(nDimNumWhenTrans) :
                 mat2->GetViewShape().GetDim(nDimNumWhenNoTrans);
  if(mDim * nDim < MM_KEQAL1_LIMIT){
    return false;
  }
  return true;
}

bool IfKEqual1(const aclTensor *&selfInput, const MmOpInfo& mmOpInfo, const bool &transX1Flag, const aclTensor *&bias) {
  // 不支持nz场景
  if (mmOpInfo.support_info.self_format == Format::FORMAT_FRACTAL_NZ ||
      mmOpInfo.support_info.mat2_format == Format::FORMAT_FRACTAL_NZ) {
    return false;
  }
  OP_LOGD("Check MatMul or BatchMatmul k=1 scenario, and support_info is not NZ");
  if (mmOpInfo.support_info.output_dtype != mmOpInfo.support_info.self_dtype ||
      mmOpInfo.support_info.output_dtype != mmOpInfo.support_info.mat2_dtype) {
    return false;
  }
  // 判断是否带bias
  if (bias != nullptr) {
    return false;
  }
  // 判断k轴是否满足切mul需求(等于1)
  int64_t kDimNumWhenNoTrans = selfInput->GetViewShape().GetDimNum() - INNER_AXIS;
  int64_t kDimNumWhenTrans = selfInput->GetViewShape().GetDimNum() - OUTER_AXIS;
  int64_t kDim = transX1Flag ? selfInput->GetViewShape().GetDim(kDimNumWhenTrans) :
                 selfInput->GetViewShape().GetDim(kDimNumWhenNoTrans);
  if (kDim != DIM_EQUAL_ONE) {
    return false;
  }
  return true;
}

aclnnStatus IfKEqual1SelfToMK(const aclTensor *&selfInput, const aclTensor *&selfReshapeOutput, bool &transX1Flag,
                              aclOpExecutor *executor) {
  auto x1Perm = NeedTransPerm(selfInput, executor);
  selfReshapeOutput = transX1Flag ? l0op::Reshape(selfInput, x1Perm, executor) : selfInput;
  CHECK_RET(selfReshapeOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  transX1Flag = false;
  return ACLNN_SUCCESS;
}

aclnnStatus IfKEqual1Mat2ToKN(const aclTensor *&mat2Input, const aclTensor *&mat2ReshapeOutput, bool &transX2Flag,
                              aclOpExecutor *executor) {
  auto x2Perm = NeedTransPerm(mat2Input, executor);
  mat2ReshapeOutput = transX2Flag ? l0op::Reshape(mat2Input, x2Perm, executor) : mat2Input;
  CHECK_RET(mat2ReshapeOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  transX2Flag = false;
  return ACLNN_SUCCESS;
}

aclnnStatus IfMEqual1SelfToMK(const aclTensor *&selfInput, const aclTensor *&selfReshapeOutput,
                              const Format selfInputFormat, bool &transX1Flag, aclOpExecutor *executor) {
  // 不支持nz场景
  if (selfInputFormat == Format::FORMAT_FRACTAL_NZ) {
    return ACLNN_SUCCESS;
  }
  OP_LOGD("Check MatMul or BatchMatmul m=1 scenario, and support_info is not NZ");
  // 首先判断m轴是否已经为外轴，是外轴则return
  if (!transX1Flag) {
    return ACLNN_SUCCESS;
  }
  // 判断m/n轴是否满足等于1，满足则reshape为外轴再进行mm/bmm计算
  int64_t mDimNumWhenInner = selfInput->GetViewShape().GetDimNum() - INNER_AXIS;
  int64_t mDimSize = selfInput->GetViewShape().GetDim(mDimNumWhenInner);
  if (mDimSize != DIM_EQUAL_ONE) {
    return ACLNN_SUCCESS;
  }
  auto x1Perm = NeedTransPerm(selfInput, executor);
  selfReshapeOutput = l0op::Reshape(selfInput, x1Perm, executor);
  transX1Flag = false;
  CHECK_RET(selfReshapeOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  OP_LOGI("Hit MatMul or BatchMatmul m=1 and m is inner scenario, trans m axis to outer");
  return ACLNN_SUCCESS;
}

aclnnStatus IfNEqual1Mat2ToNK(const aclTensor *&mat2Input, const aclTensor *&mat2ReshapeOutput,
                              const Format mat2InputFormat, bool &transX2Flag, aclOpExecutor *executor) {
  // 不支持nz场景。
  if (mat2InputFormat == Format::FORMAT_FRACTAL_NZ) {
    return ACLNN_SUCCESS;
  }
  OP_LOGD("Check MatMul or BatchMatmul n=1 scenario, and support_info is not NZ");
  // 首先判断n轴是否已经为外轴，是外轴则return
  if (transX2Flag) {
    return ACLNN_SUCCESS;
  }
  // 判断m/n轴是否满足等于1，满足则reshape为外轴再进行mm/bmm计算
  int64_t nDimNumWhenInner = mat2Input->GetViewShape().GetDimNum() - INNER_AXIS;
  int64_t nDimSize = mat2Input->GetViewShape().GetDim(nDimNumWhenInner);
  if (nDimSize != DIM_EQUAL_ONE) {
    return ACLNN_SUCCESS;
  }
  auto x2Perm = NeedTransPerm(mat2Input, executor);
  mat2ReshapeOutput = l0op::Reshape(mat2Input, x2Perm, executor);
  transX2Flag = true;
  CHECK_RET(mat2ReshapeOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  OP_LOGI("Hit MatMul or BatchMatmul n=1 and n is inner scenario, trans n axis to outer");
  return ACLNN_SUCCESS;
}

uint64_t TransDequantScaleToM1(const float deqScale) {
  union {
    float scaleFloat;
    uint32_t scaleInt;
  } dequantScale;
  dequantScale.scaleFloat = deqScale;
  uint64_t fixpipeDeqScale = static_cast<uint64_t>(dequantScale.scaleInt) & kDeqScaleMul;
  return fixpipeDeqScale;
}

op::FVector<int64_t> GetShape(const aclTensor *tensor) {
  op::FVector<int64_t> shape;
  if (tensor == nullptr) {
    shape.push_back(1);
    OP_LOGW("The input tensor of Func GetShape is nullptr");
    return shape;
  }
  if (tensor->GetViewShape().GetDimNum() == 0U) {
    shape.push_back(1);
  } else {
    size_t dimNum = tensor->GetViewShape().GetDimNum();
    for (size_t idx = 0U; idx < dimNum; idx++) {
      int64_t tmpVal = tensor->GetViewShape().GetDim(idx);
      shape.push_back(tmpVal);
    }
  }
  return shape;
}

const aclTensor *ContiguousBias(const aclTensor *self, const aclTensor *bias, aclOpExecutor *executor) {
    auto contiguousBias = l0op::Contiguous(bias, executor);
    CHECK_RET(contiguousBias != nullptr, nullptr);
    // bias为bf16时cast为fp32保证精度
    if ((contiguousBias->GetDataType() == DataType::DT_BF16 &&
          GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510)||
        self->GetDataType() == DataType::DT_FLOAT) {
        contiguousBias = l0op::Cast(contiguousBias, op::DataType::DT_FLOAT, executor);
        CHECK_RET(contiguousBias != nullptr, nullptr);
    }
    return contiguousBias;
}

// ==========================================================================================================

aclnnStatus MatmulGraphImpl::CommonPostProcess(){
        // 执行cast等操作
        convOut = l0op::Cast(convOut, output->GetDataType(), executor);
        CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 执行ViewCopy
        auto result = l0op::ViewCopy(convOut, output, executor);
        CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
}

aclnnStatus MatmulGraphImpl::CommonPostProcessWithReshape(){
    // 执行cast等操作
    convOut = l0op::Cast(convOut, output->GetDataType(), executor);
    CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 执行Reshape操作
    convOut = l0op::Reshape(convOut, output->GetViewShape(), executor);
    CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 执行ViewCopy
    auto result = l0op::ViewCopy(convOut, output, executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

// ==========================================================================================================
// NpuArchMatMulRuleBase

bool NpuArchMatMulRuleBase::CheckInputTensorDtypeValid(
    const aclTensor* matA, const aclTensor* matB, const aclTensor* bias, const aclTensor* out)
{
    auto dtypeList = GetSupportedDTypes();
    OP_CHECK_DTYPE_NOT_SUPPORT(matA, dtypeList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(matB, dtypeList, return false);

    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeList, return false);

    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, dtypeList, return false);
    }
    return true;
}

aclnnStatus NpuArchMatMulRuleBase::GetUpperDtype(
    const aclTensor* matA, const aclTensor* matB, int8_t cubeMathType, op::DataType& upperDtype)
{
    op::DataType typeA = matA->GetDataType();
    op::DataType typeB = matB->GetDataType();
    OP_LOGD("The input dtype is %s and %s", op::ToString(typeA).GetString(), op::ToString(typeB).GetString());
    OP_LOGD("The cubeMathType is %d", static_cast<int32_t>(cubeMathType));
    int inputCase = GetInputCase(typeA, typeB);

    PromoteResult result = GetUpperDtypeByLookUpTable(inputCase, cubeMathType);
    // process result
    if (result.logMessage != nullptr && result.isError) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s", result.logMessage);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (result.logMessage != nullptr && strlen(result.logMessage) > 0) {
        OP_LOGW("%s", result.logMessage);
    }
    // 返回推导类型
    upperDtype = result.type;
    return ACLNN_SUCCESS;
}

bool Dav2201MatMulRule::CheckInput(
    const aclTensor* matA, const aclTensor* matB, const aclTensor* bias, const aclTensor* out, int8_t cubeMathType)
{
        if (FP16FP32_KEEP_DTYPE == cubeMathType) {
            if (npuArch_ != NpuArch::DAV_2201) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported cubeMathType(FP16FP32_KEEP_DTYPE) for Cube");
                return false;
            }
        }

        bool dtypeVaild = CheckInputTensorDtypeValid(matA, matB, bias, out);
        if (!dtypeVaild) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported data types for Cube");
        }
        return dtypeVaild;
}

aclnnStatus Dav2201MatMulRule::PromoteDtype(
    const aclTensor* matA, const aclTensor* matB, const aclTensor* bias, const aclTensor* out, int8_t cubeMathType,
    struct MmOpInfo& mmOpInfo)
{
        // 输入数据类型
        mmOpInfo.ori_info.self_dtype = matA->GetDataType();
        mmOpInfo.ori_info.mat2_dtype = matB->GetDataType();
        mmOpInfo.ori_info.output_dtype = out->GetDataType();
        if (bias != nullptr){
            mmOpInfo.ori_info.bias_dtype = bias->GetDataType();
        }

        // 获取推导类型
        op::DataType upperDtype = op::DataType::DT_FLOAT;
        aclnnStatus checkResult = GetUpperDtype(matA, matB, cubeMathType, upperDtype);
        if(checkResult != ACLNN_SUCCESS) {
            return checkResult;
        }

        // 更新 inputDtype, 包含matA, matB
        mmOpInfo.support_info.self_dtype = upperDtype;
        mmOpInfo.support_info.mat2_dtype = upperDtype;

        // FP16FP32_KEEP_DTYPE
        if(FP16FP32_KEEP_DTYPE == cubeMathType) {
            mmOpInfo.support_info.output_dtype =  op::DataType::DT_FLOAT;
            if(bias != nullptr){
                mmOpInfo.support_info.bias_dtype =  op::DataType::DT_FLOAT;
            }
            return ACLNN_SUCCESS;
        }

        // 更新 kernel support outputDtype
        mmOpInfo.support_info.output_dtype = UpdateOutputDtype(upperDtype, out->GetDataType(), cubeMathType);

        // 更新 biasDtype
        if (bias != nullptr){
            mmOpInfo.support_info.bias_dtype = UpdateBiasDtype(upperDtype, bias->GetDataType());
        }

        return ACLNN_SUCCESS;
}

std::initializer_list<op::DataType> Dav2201MatMulRule::GetSupportedDTypes(){
        static constexpr std::initializer_list<op::DataType> dtypeSupportList = {
            op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
        return dtypeSupportList;
}

NpuArchMatMulRuleBase::PromoteResult Dav2201MatMulRule::GetUpperDtypeByLookUpTable(int inputCase, int8_t cubeMathType) {
        static constexpr const char* WARN0 = "The cubeMathType USE_HF32 will be ignored when the input dtype is FP16 or BF16.";

        static constexpr const char*  WARN1 = "The cubeMathType KEEP_DTYPE will be ignored when the inputs dtype are BF16 and FP16.";

        static constexpr const char*  WARN2 = "The cubeMathType USE_FP16 will be ignored when all inputs dtype are BF16.";

        static constexpr const char*  WARN3 = "The inputs are BF16 and FP16 with cubeMathType USE_FP16.";

        static constexpr const char*  WARN4 = "The inputs are BF16 and FP16 with cubeMathType USE_HF32.";

        static constexpr const char*  WARN5 = "The inputs are BF16 and FP16 with cubeMathType ALLOW_FP32_DOWN_PRECISION.";

        static constexpr const char*  WARN6 = "The inputs are BF16 and FP32 with cubeMathType USE_FP16, BF16 will be cast to FP16 for computation.";

        if (FP16FP32_KEEP_DTYPE == cubeMathType) {
            if (npuArch_ != NpuArch::DAV_2201) {
                return {FP32, true, "Unsupported cubeMathType(FP16FP32_KEEP_DTYPE) for Cube"};
            } else {
                // cubeMathType为KEEP_DTYPE时保持一致
                cubeMathType = KEEP_DTYPE;
            }
        }

        static constexpr NpuArchMatMulRuleBase::PromoteResult promoteResultTable[][6] = {
            /* cubeMathType:   KEEP_DTYPE,            ALLOW_FP32_DOWN_P,    USE_FP16,              USE_HF32,            USE_FP32_ADD,       USE_HIGH_PREC_MODE */
            /*0: FP32+FP32*/ {{FP32, false, ""},     {FP32, false, ""},    {FP16, false, ""},     {FP32, false, ""},    {FP32, false, ""},     {FP32, false, ""}},
            /*1: FP32+FP16*/ {{FP32, false, ""},     {FP32, false, ""},    {FP16, false, ""},     {FP32, false, ""},    {FP32, false, ""},     {FP32, false, ""}},
            /*2: FP16+FP16*/ {{FP16, false, ""},     {FP16, false, ""},    {FP16, false, ""},     {FP16, false, WARN0}, {FP16, false, ""},     {FP16, false, ""}},
            /*3: FP32+BF16*/ {{FP32, false, ""},     {FP32, false, ""},    {FP16, false, WARN6},  {FP32, false, ""},    {FP32, false, ""},     {FP32, false, ""}},
            /*4: FP16+BF16*/ {{FP32, false, WARN1},  {FP32, false, WARN5}, {FP16, false, WARN3},  {FP32, false, WARN4}, {FP32, false, ""},     {FP32, false, ""}},
            /*5: BF16+BF16*/ {{BF16, false, ""},     {BF16, false, ""},    {BF16, false, WARN2},  {BF16, false, WARN0}, {BF16, false, ""},     {BF16, false, ""}}
        };

        return promoteResultTable[inputCase][cubeMathType];
}

op::DataType Dav2201MatMulRule::UpdateOutputDtype(
    op::DataType upperDtype, op::DataType outOriDtype, int8_t cubeMathType) const {
    // 支持FP32的类型的out参与计算, out可以保持输入要求的类型,不需要做cast
    if ((npuArch_ == NpuArch::DAV_2201) &&
        (cubeMathType != USE_FP16) && outOriDtype == op::DataType::DT_FLOAT) {
        return op::DataType::DT_FLOAT;
    }
    return upperDtype;
}

op::DataType Dav2201MatMulRule::UpdateBiasDtype(op::DataType upperDtype, op::DataType biasOriDtype) const {
        if (biasOriDtype == op::DataType::DT_FLOAT) {
            return biasOriDtype;
        }
        if (biasOriDtype != upperDtype){
            return op::DataType::DT_FLOAT;
        }
        if ((npuArch_ == NpuArch::DAV_2201) && (upperDtype == op::DataType::DT_BF16)){
            return op::DataType::DT_FLOAT;
        }
        return upperDtype;
}


bool DefaultMatMulRule::CheckInput(const aclTensor* matA, const aclTensor* matB, const aclTensor* bias, const aclTensor* out, int8_t cubeMathType) {
        bool dtypeVaild = CheckInputTensorDtypeValid(matA, matB, bias, out);
        if (dtypeVaild == false) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported data types for Cube");
            return false;
        }
        bool isFp32TypeExist = matA->GetDataType() == op::DataType::DT_FLOAT || matB->GetDataType() == op::DataType::DT_FLOAT;
        if(isFp32TypeExist && (cubeMathType == USE_HF32 || cubeMathType == KEEP_DTYPE)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The npu arch does not support FP32 for calculations when the cubeMathType is KEEP_DTYPE or USE_HF32, "
                                        "please change the setting of cubeMathType or the Dtype of input tensor.");
            return false;
        }
        if(isFp32TypeExist && cubeMathType == FP16FP32_KEEP_DTYPE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The npu arch does not support FP32 for calculations when the cubeMathType is FP16FP32_KEEP_DTYPE, "
                                        "please change the setting of cubeMathType or the Dtype of input tensor.");
            return false;
        }
        return true;
}

aclnnStatus DefaultMatMulRule::PromoteDtype(const aclTensor* matA, const aclTensor* matB, const aclTensor* bias, const aclTensor* out, int8_t cubeMathType, struct MmOpInfo& mmOpInfo) {
        // 输入数据类型
        mmOpInfo.ori_info.self_dtype = matA->GetDataType();
        mmOpInfo.ori_info.mat2_dtype = matB->GetDataType();
        mmOpInfo.ori_info.output_dtype = out->GetDataType();
        if (bias != nullptr){
            mmOpInfo.ori_info.bias_dtype = bias->GetDataType();
        }

        // 获取推导类型
        op::DataType upperDtype = op::DataType::DT_FLOAT;
        aclnnStatus checkResult = GetUpperDtype(matA, matB, cubeMathType, upperDtype);
        if (checkResult != ACLNN_SUCCESS) {
            return checkResult;
        }

        // 更新 inputDtype, 包含matA, matB
        mmOpInfo.support_info.self_dtype = upperDtype;
        mmOpInfo.support_info.mat2_dtype = upperDtype;

        // FP16FP32_KEEP_DTYPE
        if(FP16FP32_KEEP_DTYPE == cubeMathType) {
            mmOpInfo.support_info.output_dtype =  op::DataType::DT_FLOAT;
            if(bias != nullptr){
                mmOpInfo.support_info.bias_dtype =  op::DataType::DT_FLOAT;
            }
            return ACLNN_SUCCESS;
        }

        // 更新 kernel support outputDtype
        mmOpInfo.support_info.output_dtype = UpdateOutputDtype(upperDtype, out->GetDataType());

        // 更新 biasDtype 与 outputDtype
        if (bias != nullptr){
            op::DataType upperOutputAndBiasDtype = PromoteOutputAndBiasDtype(mmOpInfo.support_info.output_dtype, bias->GetDataType());
            mmOpInfo.support_info.bias_dtype = upperOutputAndBiasDtype;
            mmOpInfo.support_info.output_dtype = upperOutputAndBiasDtype;
        }

        return ACLNN_SUCCESS;
}

std::initializer_list<op::DataType> DefaultMatMulRule::GetSupportedDTypes(){
        static constexpr std::initializer_list<op::DataType> dtypeSupportList = {
            op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
        return dtypeSupportList;
    }

NpuArchMatMulRuleBase::PromoteResult DefaultMatMulRule::GetUpperDtypeByLookUpTable(int inputCase, int8_t cubeMathType){
        static constexpr const char*  WARN0 = "The cubeMathType is USE_HF32. For input FP16, it will not be enabled.";

        static constexpr const char*  Err0 = "The npu arch does not support FP32 for calculations "
                                             "when the cubeMathType is KEEP_DTYPE or USE_HF32, "
                                             "please change the setting of cubeMathType or the Dtype of input tensor.";
        if (FP16FP32_KEEP_DTYPE == cubeMathType) {
            // 和cubeMathType为KEEP_DTYPE时保持一致
            cubeMathType = KEEP_DTYPE;
        }
        static constexpr PromoteResult promoteResultTable[][4] = {
            /* cubeMathType:   KEEP_DTYPE,             ALLOW_FP32_DOWN_P,    USE_FP16,             USE_HF32,     */
            /*0: FP32+FP32*/ {{FP16, true,  Err0},    {FP16, false, ""},    {FP16, false, ""},    {FP16, true, Err0}   },
            /*1: FP32+FP16*/ {{FP16, true,  Err0},    {FP16, false, ""},    {FP16, false, ""},    {FP16, true, Err0}   },
            /*2: FP16+FP16*/ {{FP16, false, ""},      {FP16, false, ""},    {FP16, false, ""},    {FP16, false, WARN0} }
        };

        return promoteResultTable[inputCase][cubeMathType];
}

op::DataType DefaultMatMulRule::UpdateOutputDtype(op::DataType upperDtype, op::DataType outOriDtype) const {
        // 如果输出类型是FP32，则按照16进32出的逻辑计算
        if (outOriDtype == op::DataType::DT_FLOAT) {
            return outOriDtype;
        }
        return upperDtype;
}

op::DataType DefaultMatMulRule::PromoteOutputAndBiasDtype(op::DataType outputDtype, op::DataType biasOriDtype) const {
        if (biasOriDtype == op::DataType::DT_FLOAT || outputDtype == op::DataType::DT_FLOAT) {
            return op::DataType::DT_FLOAT;
        }
        return op::DataType::DT_FLOAT16;
}

const aclTensor* ExecGemmV3WithAlphaBetaOp(const aclTensor* bias,
                                           const aclTensor* self,
                                           const aclTensor* mat2,
                                           const aclScalar* alpha,
                                           const aclScalar* beta,
                                           aclOpExecutor* executor)
{
    auto transposeSelf = Ops::NN::IsTransposeLastTwoDims(self);
    auto transposeMat2 = Ops::NN::IsTransposeLastTwoDims(mat2);
    // reformat, 转成ND
    auto reformatSelf = self;
    reformatSelf = l0op::ReFormat(self, op::Format::FORMAT_ND);
    // 左输入非连续转连续
    auto contiguousSelf = reformatSelf;
    if (transposeSelf) {
        contiguousSelf = executor->CreateView(
            reformatSelf, SwapLastTwoDimValue(reformatSelf->GetViewShape()), reformatSelf->GetViewOffset());
    } else {
        contiguousSelf = l0op::Contiguous(reformatSelf, executor);
    }
    CHECK_RET(contiguousSelf != nullptr, nullptr);
    // reformat, 转成ND
    auto reformatMat2 = mat2;
    reformatMat2 = l0op::ReFormat(mat2, op::Format::FORMAT_ND);
    // 右输入非连续转连续
    auto contiguousMat2 = reformatMat2;
    if (transposeMat2) {
        contiguousMat2 = executor->CreateView(
            reformatMat2, SwapLastTwoDimValue(reformatMat2->GetViewShape()), reformatMat2->GetViewOffset());
    } else {
        contiguousMat2 = l0op::Contiguous(reformatMat2, executor);
    }
    CHECK_RET(contiguousMat2 != nullptr, nullptr);

    // bias非连续转连续
    // reformat, 转成ND
    auto reformatBias = bias;
    reformatBias = l0op::ReFormat(bias, op::Format::FORMAT_ND);
    auto contiguousBias = reformatBias;
    contiguousBias = l0op::Contiguous(reformatBias, executor);
    CHECK_RET(contiguousBias != nullptr, nullptr);

    // 执行GemmV3NdWithAlphaBeta l0接口
    OP_LOGI("Entering l0op::GemmV3NdWithAlphaBeta.");
    const aclTensor* gemmV3OpOut = l0op::GemmV3NdWithAlphaBeta(contiguousSelf,
                                                               contiguousMat2,
                                                               contiguousBias,
                                                               alpha->ToFloat(),
                                                               beta->ToFloat(),
                                                               transposeSelf,
                                                               transposeMat2,
                                                               false,
                                                               executor);
    return gemmV3OpOut;
}
} // namespace NN
} // namespace Ops
