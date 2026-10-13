/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_SRC_LEVEL2_MATMUL_UTIL_H_
#define OP_API_SRC_LEVEL2_MATMUL_UTIL_H_

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"

#define DEPRECATED_API_WARN_ONCE(oldApiName, deprecatedDate, newApiName)                 \
    do {                                                                                 \
        static bool isFirstWarn = true;                                                  \
        if (isFirstWarn){                                                                \
            OP_LOGW("%s is scheduled to be deprecated in a post-%s version update, "                                            \
                        "and will be replaced by the %s. "                                                                      \
                        "We apologize for any inconvenience caused and appreciate your timely migration to the new interface.", \
                        (oldApiName), (deprecatedDate), (newApiName));                   \
            isFirstWarn = false;                                                         \
        }                                                                                \
    } while(0)

namespace Ops {
namespace NN {
using namespace op;
// These are used to check repo hit
const int32_t FP16_BF16_FLAG = 1;
const int32_t FP32_FLAG = 0;
const int32_t HF32_FLAG = 64;
const std::string SOC_B3 = "Ascend910B3";
const std::string SOC_C3 = "Ascend910_9372";
const std::string SOC_B4 = "Ascend910B4";
const std::string SOC_C4 = "Ascend910_9382";

struct OpBaseInfo {
  op::DataType self_dtype;
  op::Format self_format;
  op::DataType mat2_dtype;
  op::Format mat2_format;
  op::DataType output_dtype;
  op::Format output_format;
  op::DataType bias_dtype;
  op::Format bias_format;
};

struct OpShapeInfo {
  // m、k、n dim
  int64_t mDim;
  int64_t kDim;
  int64_t nDim;

  // input tranpose flag
  bool transposeX1;

  // mat2 transpose flag
  bool transposeX2;

  int64_t dtypeASize = 2; // float16 dtype 2
  int64_t dtypeBSize = 2; //float16 dtype 2
};

struct MmOpInfo {
  // mm api input info
  OpBaseInfo ori_info;
  // npu mm kernel support info
  OpBaseInfo support_info;
  // HF32 Flag
  int64_t opImplModeEnum = 0x1;
  bool enableHf32 = false;
  bool enableForceGrpAccForFp32 = false;
  bool enableFp16Bf16InFp32Out = false;
  bool supporSplitK = false;
  int64_t aiCoreCnt;
  // mm api shape info
  OpShapeInfo shapeInfo;
};

struct TensorInfo {
  const aclTensor* tensor = nullptr;
  op::DataType dataType = op::DataType::DT_FLOAT;
  op::Format format = Format::FORMAT_ND;
};

op::Shape SwapLastTwoDimValue(const op::Shape tensorShape, int64_t last = 1UL, int64_t secondLast = 2UL);

bool IsInputSupportFp32();

bool CheckBatchDimBroadcast(size_t batch1DimNum, size_t batch2DimNum, const op::Shape& batch1, const op::Shape& batch2);

bool IsFormatSupportNd(const aclTensor* self, const aclTensor* mat2);

bool IsNdToNzOnTheFly(const aclTensor* self, const aclTensor* mat2);

bool IsTransposeLastTwoDims(const aclTensor* tensor);

bool CheckGemmV3Support(const aclTensor* mat1, const aclTensor* mat2, MmOpInfo& mmOpInfo,
                                      int8_t cubeMathType);

bool IsSliceNonContiguous(const aclTensor* self, const aclTensor* mat2);

bool IsTransposeNonContiguous(const aclTensor* tensor, bool& isNeedSwapInnerTwoDim);

bool CheckNonContiguousShapeSupport(MmOpInfo& mmOpInfo);

bool CheckGemmV3WithAlphaBeta(const aclTensor* bias, const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType);

const aclTensor* ExecGemmV3Op(const aclTensor* self, const aclTensor* mat2, const aclTensor* c, MmOpInfo& mmOpInfo,
                              aclOpExecutor* executor);

const aclTensor* ExecGemmV3WithAlphaBetaOp(const aclTensor* bias,
                                           const aclTensor* self,
                                           const aclTensor* mat2,
                                           const aclScalar* alpha,
                                           const aclScalar* beta,
                                           aclOpExecutor* executor);

const aclTensor* ExecMmOp(const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType, aclOpExecutor* executor);

const aclTensor* ExecMmOpWithBias(const aclTensor* self, const aclTensor* mat2, const aclTensor* bias,
                                  int8_t cubeMathType, aclOpExecutor* executor, bool transposeX2 = false);

const aclTensor* ExecMmOpWithTrans(const aclTensor* self, const aclTensor* mat2, int64_t transSelf, int64_t transMat2,
                                   int8_t cubeMathType, aclOpExecutor* executor);

aclnnStatus SetMmSupportDType(MmOpInfo& mmOpInfo, int8_t cubeMathType);

aclnnStatus SetMmSupportFormat(const aclTensor* self, const aclTensor* mat2, MmOpInfo& mmOpInfo);

aclnnStatus CreateMatmulOpInfo(
    const aclTensor* self, const aclTensor* mat2, const aclTensor* bias, const aclTensor* out, int8_t cubeMathType,
    MmOpInfo& mmOpInfo, bool isSelfSlice);

aclnnStatus GetMmInfo(MmOpInfo mmOpInfo);
MmOpInfo GetMatmulOpInfo(const aclTensor *self, const aclTensor *mat2, int8_t cubeMathType, bool isSelfSlice = false);
bool ContiguousAndCast(const aclTensor *&contiguousInput, const aclTensor *&castOut, bool &transposeFlag,
                                     op::DataType dtype, aclOpExecutor *executor);

bool ContiguousAndCastBias(const aclTensor*& contiguousInput, const aclTensor*& castOut, op::DataType dtype,
    aclOpExecutor* executor);

bool GetNzSplitKFlag(const aclTensor *self, const aclTensor *mat2, const op::Format selfSuppFormat, const op::Format outSuppFormat);

bool IsSupportNzNzNd(const aclTensor* self, const aclTensor* mat2);

bool IsSplitk(const TensorInfo* self, const TensorInfo* mat2);

bool NeedToConvertBias(const aclTensor* self, const aclTensor* mat1, const aclTensor* mat2, const aclScalar* beta,
                       const aclScalar* alpha);

// 区别bmm 和 mm, bmm（DimNum==3）返回 1， mm（DimNum==2）返回0
int64_t GetOffSet(int64_t DimNum);

aclIntArray* NeedTransPerm(const aclTensor *x, aclOpExecutor *executor);

bool checkBF16SizeValid(const aclTensor *&mat2, const bool &transX2Flag);

bool checkBF16MMValid(const aclTensor *&self, const aclTensor *&mat2, const bool &transX2Flag);

bool IfKEqual1(const aclTensor *&selfInput, const MmOpInfo& mmOpInfo, const bool &transX1Flag, const aclTensor *&bias);

aclnnStatus IfKEqual1SelfToMK(const aclTensor *&selfInput, const aclTensor *&selfReshapeOutput, bool &transX1Flag,
                             aclOpExecutor *executor);

aclnnStatus IfKEqual1Mat2ToKN(const aclTensor *&mat2Input, const aclTensor *&mat2ReshapeOutput, bool &transX2Flag,
                             aclOpExecutor *executor);

aclnnStatus IfMEqual1SelfToMK(const aclTensor *&selfInput, const aclTensor *&selfReshapeOutput,
                              const op::Format selfInputFormat, bool &transX1Flag, aclOpExecutor *executor);

aclnnStatus IfNEqual1Mat2ToNK(const aclTensor *&mat2Input, const aclTensor *&mat2ReshapeOutput,
                              const op::Format mat2InputFormat, bool &transX2Flag, aclOpExecutor *executor);

int64_t ProcessSpecialCases(
    const aclTensor*& selfCastOut, const aclTensor*& mat2CastOut, MmOpInfo& mmOpInfo, const aclTensor*& bias,
    const aclTensor*& selfReshapeOutput, const aclTensor*& mat2ReshapeOutput, aclOpExecutor* executor, bool& ifKEqual1);

uint64_t TransDequantScaleToM1(const float deqScale);

const aclTensor *ContiguousBias(const aclTensor *self, const aclTensor *bias, aclOpExecutor *executor);

op::FVector<int64_t> GetShape(const aclTensor *tensor);

const aclTensor* MatmulCommonProcess (
    const aclTensor* self, const aclTensor* mat2, const aclTensor* bias, const aclTensor* out,
    const int8_t cubeMathType, MmOpInfo& mmOpInfo, aclOpExecutor* executor,
    bool transposeX2);

bool CheckGemmV3Support(const aclTensor* mat1, const aclTensor* mat2, const aclTensor* bias, const aclTensor* out,
MmOpInfo& mmOpInfo, int8_t cubeMathType);

// strategy =============================================================================================

class MatmulGraphImpl {
public:
    virtual aclnnStatus PreProcess(){ return ACLNN_SUCCESS; }

    virtual aclnnStatus Impl() = 0;

    virtual aclnnStatus PostProcess(){ return CommonPostProcess(); }

    MatmulGraphImpl(
        const aclTensor* matAParam, const aclTensor* matBParam, const aclTensor* biasParam, aclTensor* outputParam,
        const aclScalar* alphaNearMat1, const aclScalar* betaNearBias, int8_t cubeMathTypeParam,
        aclOpExecutor* executorParam)
        : matA(matAParam),
          matB(matBParam),
          bias(biasParam),
          output(outputParam),
          alpha(alphaNearMat1),
          beta(betaNearBias),
          cubeMathType(cubeMathTypeParam),
          executor(executorParam) {};

    MatmulGraphImpl(
        const aclTensor* matAParam, const aclTensor* matBParam, aclTensor* outputParam,
        int8_t cubeMathTypeParam, aclOpExecutor* executorParam)
        : matA(matAParam),
          matB(matBParam),
          output(outputParam),
          cubeMathType(cubeMathTypeParam),
          executor(executorParam) {};

    virtual ~MatmulGraphImpl() = default;

    aclnnStatus Execute() {
        aclnnStatus ret = PreProcess();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }

        ret = Impl();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }

        ret = PostProcess();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        return ACLNN_SUCCESS;
    }

    void SetOpInfo (MmOpInfo& mmOpInfo){
      opInfo = mmOpInfo;
    }

    aclnnStatus CommonPostProcess();

    aclnnStatus CommonPostProcessWithReshape();

protected:
    const aclTensor* matA;
    const aclTensor* matB;
    const aclTensor* bias = nullptr;
    aclTensor* output;
    const aclScalar* alpha;
    const aclScalar* beta;
    int8_t cubeMathType;
    MmOpInfo opInfo; // 用于提前计算所有前后处理相关的format、dtype等信息
    const aclTensor* convOut = nullptr;
    uint64_t* workspaceSize = nullptr;
    aclOpExecutor* executor;
};

// ======================================================================================================

// NpuArch规则基类：抽象不同NpuArch的校验和数据类型推导逻辑
class NpuArchMatMulRuleBase {
protected:
    static const op::DataType FP16 = op::DataType::DT_FLOAT16;
    static const op::DataType FP32 = op::DataType::DT_FLOAT;
    static const op::DataType BF16 = op::DataType::DT_BF16;

    // 存储NpuArch
    NpuArch npuArch_;

    // 保护构造函数：禁止直接实例化基类（只能通过派生类构造）
    NpuArchMatMulRuleBase(NpuArch npuArch) : npuArch_(npuArch) {}

public:
    // 虚析构函数：确保派生类析构正常调用
    virtual ~NpuArchMatMulRuleBase() = default;

    // 校验规则接口：检查当前输入是否符合NpuArch的约束
    // 返回：校验通过返回true，否则返回false
    virtual bool CheckInput(const aclTensor* matA, const aclTensor* matB, const aclTensor* bias, const aclTensor* out, int8_t cubeMathType) = 0;

    // 返回计算时的数据类型MmOpInfo
    virtual aclnnStatus PromoteDtype(const aclTensor* matA, const aclTensor* matB, const aclTensor* bias, const aclTensor* out, int8_t cubeMathType, struct MmOpInfo& mmOpInfo) = 0  ;

    // 获取支持的类型列表
    virtual std::initializer_list<op::DataType> GetSupportedDTypes() = 0;

    // 初步推导后的类型及兼容记录
    struct PromoteResult {
        op::DataType type = op::DataType::DT_FLOAT;
        bool isError = false;
        const char* logMessage = nullptr;

        constexpr PromoteResult(op::DataType dataType, bool err, const char* msg) : type(dataType), isError(err), logMessage(msg) {}
    };

protected:
    // CheckInputTensorDtypeValid
    bool CheckInputTensorDtypeValid(const aclTensor* matA, const aclTensor* matB, const aclTensor* bias, const aclTensor* out) ;

    inline int GetIndexByDataType(op::DataType dataType) const {
        static const std::map<op::DataType, int> typeIndexMap = {
            {op::DataType::DT_FLOAT, 0},
            {op::DataType::DT_FLOAT16, 1},
            {op::DataType::DT_BF16, 2}
        };
        return typeIndexMap.at(dataType);
    }

    // 获取两个input的组合形式
    inline int GetInputCase(const op::DataType& typeA, const op::DataType& typeB) const
    {
        int indexA = GetIndexByDataType(typeA);
        int indexB = GetIndexByDataType(typeB);

        static const int intputCaseTable[][3] = {
            /*        Fp32 FP16 BF16 */
            /*FP32*/ {0,   1,   3},
            /*FP16*/ {1,   2,   4},
            /*BF16*/ {3,   4,   5}
        };

        int inputCase = intputCaseTable[indexA][indexB];
        return inputCase;
    }

    virtual PromoteResult GetUpperDtypeByLookUpTable(int inputCase, int8_t cubeMathType) = 0;

    aclnnStatus GetUpperDtype(const aclTensor* matA, const aclTensor* matB, int8_t cubeMathType, op::DataType& upperDtype);
};

// 适用arch: DAV_2201, DAV_3510
class Dav2201MatMulRule : public NpuArchMatMulRuleBase {
public:
    Dav2201MatMulRule(NpuArch npu_arch)
        : NpuArchMatMulRuleBase(npu_arch) {}

    ~Dav2201MatMulRule() override = default;

    bool CheckInput(const aclTensor* matA, const aclTensor* matB, const aclTensor* bias, const aclTensor* out, int8_t cubeMathType) override;

    aclnnStatus PromoteDtype(const aclTensor* matA, const aclTensor* matB, const aclTensor* bias, const aclTensor* out, int8_t cubeMathType, struct MmOpInfo& mmOpInfo) override;

    std::initializer_list<op::DataType> GetSupportedDTypes() override;

private:

    PromoteResult GetUpperDtypeByLookUpTable(int inputCase, int8_t cubeMathType) override;

    op::DataType UpdateOutputDtype(op::DataType upperDtype, op::DataType outOriDtype, int8_t cubeMathType) const;

    op::DataType UpdateBiasDtype(op::DataType upperDtype, op::DataType biasOriDtype) const;
};

// 适用其他arch
class DefaultMatMulRule : public NpuArchMatMulRuleBase {
public:
    DefaultMatMulRule(NpuArch npu_arch)
        : NpuArchMatMulRuleBase(npu_arch) {}

    ~DefaultMatMulRule() override = default;

    bool CheckInput(const aclTensor* matA, const aclTensor* matB, const aclTensor* bias, const aclTensor* out, int8_t cubeMathType) override;

    aclnnStatus PromoteDtype(const aclTensor* matA, const aclTensor* matB, const aclTensor* bias, const aclTensor* out, int8_t cubeMathType, struct MmOpInfo& mmOpInfo) override;

    std::initializer_list<op::DataType> GetSupportedDTypes() override;

private:

    PromoteResult GetUpperDtypeByLookUpTable(int inputCase, int8_t cubeMathType) override;

    op::DataType UpdateOutputDtype(op::DataType upperDtype, op::DataType outOriDtype) const;

    op::DataType PromoteOutputAndBiasDtype(op::DataType outputDtype, op::DataType biasOriDtype) const;
};

std::shared_ptr<NpuArchMatMulRuleBase> BuildRule();

// =====================================================================================================

} // namespace NN
} // namespace Ops

#endif  // OP_API_SRC_LEVEL2_MATMUL_UTIL_H_