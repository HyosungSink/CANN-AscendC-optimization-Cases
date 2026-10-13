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
 * \file swi_glu_proto.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_SWI_GLU_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SWI_GLU_PROTO_H_

#include "graph/operator_reg.h"

namespace ge{
    /**
    * @brief Compute the SwiGlu,
    * where the activations function in GLU is Swish.

    * @par Inputs:
    * One input, including:
    * @x: A Tensor. Must be one of the following types: bfloat16, float16, float32.

    * @par Outputs:
    * one output, including:
    * @y: A Tensor. Must be one of the following types: bfloat16, float16, float32.

    * @par Attributes:
    * one attribute, including:
    * @li dim: A optional int. The dimension to be split, default is -1.

    * @par Third-party framework compatibility:
    * New operator SwiGlu.
    */
    REG_OP(SwiGlu)
        .INPUT(x, "T")
        .OUTPUT(y, "T")
        .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .ATTR(dim, Int, -1)
        .OP_END_FACTORY_REG(SwiGlu)
}
#endif  // OPS_BUILT_IN_OP_PROTO_INC_SWI_GLU_PROTO_H_
