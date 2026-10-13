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
 * \file moe_init_routing_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MOE_INIT_ROUTING_OPS_H_
#define OPS_OP_PROTO_INC_MOE_INIT_ROUTING_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief compute init routing quant for moe input.
 * @par Inputs:
 * @li x: A 2D Tensor. Type is:BFloat16, Float16 or Float32. Format support ND.
 * @li row_idx: A 2D Tensor: A Tensor. Type is:Int32. Format support ND.
 * @li expert_idx: A 2D Tensor. Type is:Int32. Format support ND.
 * @par Outputs:
 * @li expanded_x: A 2D Tensor. Type is:Int8. The data type must be the same as that of x.
 *                 The first dim must be the first dim of row_idx multiply the second dim of row_idx.
 *                 The second dim must be the second dim of x. Format support ND.
 * @li expanded_row_idx: A 1D Tensor. Type is:Int32. The dim must be  the first dim of row_idx multiply the second
 *                       dim of row_idx. Format support ND.
 * @li expanded_expert_idx: A 1D Tensor. Type is:Int32. The Shape is same as expanded_row_idx. Format support ND.
 * @par Attributes:
 * @li active_num: Required parameter. Type is:Int32. The value 0 indicates a non-active
 *                 scenario, and a value greater than 0 indicates an active scenario. In the active scenario, the
 *                 size of axis 0 of expanded_x must be equal to the value of active_num.
 * @li scale: Required parameter. Type is:Float.
 * @li offset: Required parameter. Type is:Float.
 */
REG_OP(MoeInitRoutingQuant)
    .INPUT(x, "T1")
    .INPUT(row_idx, "T2")
    .INPUT(expert_idx, "T2")
    .OUTPUT(expanded_x, "T3")
    .OUTPUT(expanded_row_idx, "T2")
    .OUTPUT(expanded_expert_idx, "T2")
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_INT32}))
    .DATATYPE(T3, TensorType({DT_INT8}))
    .REQUIRED_ATTR(active_num, Int)
    .REQUIRED_ATTR(scale, Float)
    .REQUIRED_ATTR(offset, Float)
    .OP_END_FACTORY_REG(MoeInitRoutingQuant)

} // namespace ge

#endif