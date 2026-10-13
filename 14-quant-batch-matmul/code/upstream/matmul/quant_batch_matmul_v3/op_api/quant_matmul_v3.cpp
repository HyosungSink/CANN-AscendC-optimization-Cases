/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_matmul_v3.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(QuantBatchMatmulV3);

constexpr int64_t TYPE_FP16 = 1;
constexpr int64_t TYPE_BF16 = 27;
constexpr int64_t TYPE_INT32 = 3;
constexpr int64_t TYPE_HIF8 = 34;
constexpr int64_t TYPE_FP8_E4M3FN = 36;
constexpr int64_t TYPE_FP32 = 0;
constexpr int64_t PPMATMUL_PRIORITY_M = 1024;
constexpr int64_t NO_BATCH_DIM_SUM = 2;

const aclTensor* QuantBatchMatmulV3(const aclTensor* x1, const aclTensor* x2, const aclTensor* scale,
                                    const aclTensor* offset, const aclTensor* bias, const aclTensor* pertokenScale,
                                    int64_t dtype, bool transposeX1, bool transposeX2, int64_t groupSize,
                                    aclOpExecutor* executor) {
    L0_DFX(QuantBatchMatmulV3, x1, x2, scale, offset, bias, pertokenScale, transposeX1, transposeX2, groupSize);
    DataType outType = DataType::DT_INT8;
    if (dtype == TYPE_FP16) {
        outType = DataType::DT_FLOAT16;
    } else if (dtype == TYPE_BF16) {
        outType = DataType::DT_BF16;
    } else if (dtype == TYPE_INT32) {
        outType = DataType::DT_INT32;
    } else if (dtype == TYPE_FP32) {
        outType = DataType::DT_FLOAT;
    }
    Format format = Format::FORMAT_ND;
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P && pertokenScale != nullptr && !pertokenScale->IsEmpty()) {
        OP_LOGD("Npu_Arch = 2002 pertoken mode need transData out");
        format = Format::FORMAT_FRACTAL_NZ;
    }
    auto inputAShape = x1->GetViewShape();
    uint32_t M = inputAShape.GetDimNum() == NO_BATCH_DIM_SUM ? inputAShape[0] : inputAShape[1];
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P && M >= PPMATMUL_PRIORITY_M &&
        bias != nullptr && !transposeX1 && transposeX2 && dtype != DataType::DT_BF16) {
        OP_LOGD("Using PpMatmul mode need transData out");
        format = Format::FORMAT_FRACTAL_NZ;
    }

    auto output = executor->AllocTensor(outType, format, format);

    auto ret = INFER_SHAPE(QuantBatchMatmulV3, OP_INPUT(x1, x2, scale, offset, bias, pertokenScale), OP_OUTPUT(output),
                            OP_ATTR(dtype, transposeX1, transposeX2, groupSize));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "QuantBatchMatmulV3 InferShape failed.");
    ret = ADD_TO_LAUNCHER_LIST_AICORE(QuantBatchMatmulV3, OP_INPUT(x1, x2, scale, offset, bias, pertokenScale),
                                      OP_OUTPUT(output), OP_ATTR(dtype, transposeX1, transposeX2, groupSize));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "QuantBatchMatmulV3 ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return output;
}
}