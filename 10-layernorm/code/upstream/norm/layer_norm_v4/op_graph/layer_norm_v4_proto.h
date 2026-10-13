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
 * \file layer_norm_v4_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_NORM1_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_NORM1_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief LayernormV4 operator interface implementation \n
* calculating: x, gamma, beta \n
* mean  = np.mean(x, reduce_axis, keepdims=True) \n
* rstd = np.rsqrt(np.mean(np.power((x - mean),2), reduce_axis, keepdims=True) + epsilon)) \n
* y = gamma*((x - mean) * rstd) + beta

* @par Inputs:
* Four inputs, including:
* @li x: A ND Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li normalized_shape: A ND Tensor. Must be one of the following types: int32, int64
* @li gamma: A ND Tensor. Must be one of the following types: float16, float32, bfloat16. Shape is normalized_shape.
* @li beta: A ND Tensor. Must be one of the following types: float16, float32, bfloat16. Shape is normalized_shape.\n

* @par Attributes:
* @li epsilon: An optional attribute, the type is float32. Defaults to 1e-5 . \n

* @par Outputs:
* Three outputs, including:
* @li y: A ND Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li mean: A ND Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li rstd: A ND Tensor. Must be one of the following types: float16, float32, bfloat16.
*/
REG_OP(LayerNormV4)
    .INPUT(x, "T1")
    .INPUT(normalized_shape, "T2")
    .OPTIONAL_INPUT(gamma, "T3")
    .OPTIONAL_INPUT(beta, "T4")
    .OUTPUT(y, "T5")
    .OUTPUT(mean, "T6")
    .OUTPUT(rstd, "T6")
    .ATTR(epsilon, Float, 0.00001f)
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_INT32, DT_INT64}))
    .DATATYPE(T3, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T4, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T5, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T6, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(LayerNormV4)

} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_NN_NORM1_OPS_H_
