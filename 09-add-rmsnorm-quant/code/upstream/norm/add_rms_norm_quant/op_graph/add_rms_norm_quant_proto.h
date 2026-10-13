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
 * \file add_rms_norm_quant_proto.h
 * \brief
 */
#ifndef OPS_NORM_ADD_RMS_NORM_QUANT_PROTO_H_
#define OPS_NORM_ADD_RMS_NORM_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief AddRmsNormQuant operator interface implementation.
* Calculating input: x1, x2, gamma, scales1, scales2, zero_points1, zero_points2 \n
* Calculating process: \n
*  x = x1 + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x, 2), reduce_axis, keepdims=True) + epsilon)) \n
*  rmsnorm_out = x * rstd * gamma \n
*  if div_mode is true: \n
*    y1 = round(rmsnorm_out / scales1 + zero_points1) \n
*    y2 = round(rmsnorm_out / scales2 + zero_points2) \n
*  if div_mode is false: \n
*    y1 = round(rmsnorm_out * scales1 + zero_points1) \n
*    y2 = round(rmsnorm_out * scales2 + zero_points2) \n

* @par Inputs
* @li x1: A tensor. Input x1 for the add operation.
*         Support dtype: float32/float16/bfloat16, support format: ND.
* @li x2: A tensor. Input x2 for the add operation.
*         Support dtype: float32/float16/bfloat16, support format: ND.
* @li gamma: A tensor. Describing the weight of the rmsnorm operation.
*            Support dtype: float32/float16/bfloat16, support format: ND.
* @li scales1: A tensor. Describing the weight of the first quant operation.
*              Support dtype: float32/float16/bfloat16, support format: ND.
* @li scales2: An optional input tensor. Describing the weight of the secend quant operation.
*              Support dtype: float32/float16/bfloat16, support format: ND.
* @li zero_points1: An optional input tensor. Describing the bias of the first quant operation.
*                   Support dtype: int32/float32/float16/bfloat16, support format: ND.
* @li zero_points2: An optional input tensor. Describing the bias of the secend quant operation.
*                   Support dtype: int32/float32/float16/bfloat16, support format: ND.

* @par Attributes
* @li axis: An optional attribute. Describing the axis of the quant operation, does not take effect now.
*           The type is int. Defaults to -1.
* @li epsilon: An optional attribute. Describing the epsilon of the rmsnorm operation.
*              The type is float. Defaults to 1e-6.
* @li div_mode: An optional attribute. When div_mode is true, the quant opertaion uses division, otherwise, uses
multiplication.
*               The type is bool. Defaults to true.
* @li dst_type: An optional int32. Output y data type enum value. Support DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E5M2, 
*               DT_FLOAT8_E4M3FN. Defaults to DT_INT8.
* @par Outputs
* @li y1: A tensor. Describing the output of the first quant operation.
*                   Support dtype: int8/hifloat8/float8e5m2/float8e4m3fn, support format: ND.
* @li y2: A tensor. Describing the output of the second quant operation.
*                   Support dtype: int8/hifloat8/float8e5m2/float8e4m3fn, support format: ND.
* @li x: A tensor. Describing the output of the x1+x2 add operation.
*                  Support dtype: float32/float16/bfloat16, support format: ND.
*/

REG_OP(AddRmsNormQuant)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scales1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(scales2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(zero_points1, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(zero_points2, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_INT8, DT_HIFLOAT8, DT_FP8_E5M2, DT_FP8_E4M3FN}))
    .OUTPUT(y2, TensorType({DT_INT8, DT_HIFLOAT8, DT_FP8_E5M2, DT_FP8_E4M3FN}))
    .OUTPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(axis, Int, -1)
    .ATTR(epsilon, Float, 1e-6f)
    .ATTR(div_mode, Bool, true)
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(AddRmsNormQuant)
} // namespace ge

#endif // OPS_NORM_ADD_RMS_NORM_QUANT_PROTO_H_