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
 * \file rms_norm_grad_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_NORM1_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_NORM1_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief RmsNormGrad operator interface implementation.

* @par Inputs
* Four inputs, including:
* @li dy: The gradient returned backward.
*     A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x: The input of the forward operator.
*     A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li rstd: The intermediate computation result of the forward operator.
*     A Tensor. Support dtype: [float32], support format: [ND].
* @li gamma: The input of the forward operator.
*     A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

* @par Outputs
* @li dx: The gradient of input "x", Has the same type and shape as "x".
*     A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li dgamma: The gradient of input "gamma". Has the same type and shape as "gamma".
*     A Tensor. Support dtype: [float32], support format: [ND].
*/
REG_OP(RmsNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(dx, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(dgamma, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OP_END_FACTORY_REG(RmsNormGrad)

} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_NN_NORM1_OPS_H_
