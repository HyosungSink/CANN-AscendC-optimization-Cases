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
 * \file all_gather_matmul_proto.h
 * \brief
 */
#ifndef ALL_GATHER_MATMUL_PROTO_H_
#define ALL_GATHER_MATMUL_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Fusion op of allgather and matmul.
* @par Inputs:
* three inputs, including:
* @li x1: A matrix Tensor. The type support float16, bfloat16. The format supports ND. The x1 only supports 2 dimensions in current version, for example (M, K). The x1 doesn't support transposed.
* @li x2: A matrix Tensor. The type support float16, bfloat16. The format supports ND. The x2 only supports 2 dimensions in current version, for example (K, N). The x2 supports transposed and non-transposed.
  The K value in x2 should be same as the K value in x1 when x2 is non-transposed, and the K value should be in range [256, 65535).
* @li bias: A matrix Tensor. The type support float16, bfloat16. The format supports ND. The current version does not support the scenario where bias is not 0.\n
*
* @par Attributes:
* @li group: A string. A required string identifying the group of ranks participating in the op.
* @li is_trans_a: A bool. If true, changes the shape of "x1" from [K, M] to [M, K] before multiplication. Default: false.
* @li is_trans_b: A bool. If true, changes the shape of "x2" from [N, K] to [K, N] before multiplication. Default: false.
* @li gather_index: An int. Represents the input index for doing gather, 0: left matrix, 1: right matrix. Default: 0. The gather_index only supports 0 in current version.
* @li comm_turn: An int. Number of communications with AICPU. Default: 0. The comm_turn only supports 0 in current version.
* @li rank_size: An int. Number of ranks in the group. Default: 0. \n
  The Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component support 2, 4, 8. \n
  The Atlas A3 Training Series Product/Atlas A3 Inference Series Product support 2, 4, 8, 16. \n
* @li is_gather_out: A bool. If true, output gather_out matrix. Default: true. \n
*
* @par Outputs:
* @li y: A matrix Tensor. The type support float16, bfloat16. The format supports ND. The y is 2 dimensions, for example (M*rank_size, N).
* @li gather_out: A matrix Tensor. The type support float16, bfloat16. The format supports ND. \n
*/
REG_OP(AllGatherMatmul)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(gather_out, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(gather_index, Int, 0)
    .ATTR(comm_turn, Int, 0)
    .ATTR(rank_size, Int, 0)
    .ATTR(is_gather_out, Bool, true)
    .OP_END_FACTORY_REG(AllGatherMatmul)
}  // namespace ge


#endif  // ALL_GATHER_MATMUL_PROTO_H_