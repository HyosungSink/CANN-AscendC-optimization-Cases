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
 * \file moe_distribute_combine_add_rms_norm_proto.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM_PROTO_H_
#define MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief MoeDistributeCombineAddRmsNorm operator interface implementation.

* @par Inputs
* Fourteen inputs, including:
* @li expand_x: A tensor. The characteristics of the extended token based on expandIds. Support dtype: bfloat16, dimension must be 2, Support Shape: (A * world_size, H), support format: ND.
* @li expert_ids: A tensor. The topK expert index for each token. Support dtype: int32, dimension must be 2, Support Shape: (BS, K), support format: ND.
* @li assist_info_for_combine: A tensor. An output of MoeDistributeDispatch. Support dtype: int32, dimension must be 1, Support Shape: (A * 128), support format: ND.
* @li ep_send_counts: A tensor. An output of MoeDistributeDispatch. Support dtype: int32, Support Shape: (expert_nums + 2 * globalBs * K * server_num, ), support format: ND.
* @li expert_scales: A tensor. The weight of the topK experts for each token. Support dtype: float32, Support Shape: (BS, K), support format: ND.
* @li residual_x: A tensor. The right matrix of the add in AddRmsNorm. Support dtype: bfloat16, Support Shape: (BS, 1, H), support format: ND.
* @li gamma: A tensor. An input of addRmsNorm. Support dtype: bfloat16, Support Shape: (H), support format: ND.
* @li tp_send_counts: A tensor. The output tpRecvCounts in MoeDistributeDispatch. Support dtype: int32, support format: ND.
* @li x_active_mask: An optional tensor. Whether the token participates in the communication. True: participate, False: do not participate. Support dtype: bool, support format: ND.
* @li activation_scale: An optional tensor. Support dtype: float32, support format: ND. It's not supported yet.
* @li weight_scale: An optional tensor. Support dtype: float32, support format: ND. It's not supported yet.
* @li group_list: An optional tensor. Support dtype: int64, support format: ND. It's not supported yet.
* @li expand_scales: A tensor. An output of MoeDistributeDispatch. Support dtype: float32, Support Shape: (A, ), support format: ND.
* @li shared_expert_x: A tensor. Share the tokens calculated by experts. Support dtype: bfloat16, support format: ND.

* @par Attributes
* @li group_ep: Input ep comm group name, ep means experts parallelism, dtype: String. Support string length [1, 128), could not bee the same with group_tp.
* @li ep_world_size: Input ep comm world size, dtype: Int64. Support value range: [1, 384].
* @li ep_rank_id: Input ep comm rank Id, dtype: Int64. Support value range: [0, ep_world_size), could not be duplicated within the same EP domain.
* @li moe_expert_num: Input moe expert num, dtype: Int64. Support value range: [0, 512).
* @li group_tp: Input tp comm group name, tp means tensor parallelism, dtype: String. Support string length [1, 128), could not bee the same with group_ep. Default value: "".
* @li tp_world_size: Input tp comm world size, dtype: Int64. Support value range: [0, 2]. Default value: 0.
* @li tp_rank_id: Input tp comm rank Id, dtype: Int64. Support value range: [0, tp_world_size), could not be duplicated within the same TP domain. Default value: 0.
* @li expert_shard_type: Input moe shard type, dtype: Int64. Only support 0 now. Default value: 0.
* @li shared_expert_num: Input shared expert num, dtype: Int64. Support value range: [0, 4].D efault value: 1.
* @li shared_expert_rank_num: Input shared expert rank num, dtype: Int64. Support value range: [0, ep_world_size - 1). Default value: 0.
* @li global_bs: Input global batch size, dtype: Int64. When bs is the same on all the ranks, globalBs = bs * ep_world_size or globalBs = 0;
      when bs is not the same on all the ranks, globalBs = maxBs * ep_world_size. Default value: 0.
* @li out_dtype: Dtype of output, 0 for bfloat16, 1 for float16, dtype: Int64. Default value: 0. It's not supported yet.
* @li comm_quant_mode: Communication quantization mode, 1 for enable, 0 for disable, dtype: Int64. Default value: 0. It's not supported yet.
* @li group_list_type: Type of input group_list, dtype: Int64. Default value: 0. It's not supported yet.
* @li comm_alg: Input comm alg type, dtype: String. Default value: "0". It's not supported yet.
* @li norm_eps: Used to prevent AddRmsNorm by 0 error, dtype:Float. Default value: 1e-6.

* @par Outputs
* Three outputs, including:
* @li y: A tensor. Result of (combine - Add) + Add + RmsNorm. Support dtype: bfloat16,  Support Shape: (BS, 1, H), support format: ND.
* @li rstd: A tensor. Result of standard deviation. Support dtype: float32,  Support Shape: (BS, 1, 1), support format: ND.
* @li x: A tensor. Result of (combine - Add) + Add. Support dtype: bfloat16,  Support Shape: (BS, 1, H), support format: ND.
*/
REG_OP(MoeDistributeCombineAddRmsNorm)
    .INPUT(expand_x, TensorType({DT_BF16}))
    .INPUT(expert_ids, TensorType({DT_INT32}))
    .INPUT(assist_info_for_combine, TensorType({DT_INT32}))
    .INPUT(ep_send_counts, TensorType({DT_INT32}))
    .INPUT(expert_scales, TensorType({DT_FLOAT}))
    .INPUT(residual_x, TensorType({DT_BF16}))
    .INPUT(gamma, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(tp_send_counts, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(x_active_mask, TensorType({DT_BOOL}))
    .OPTIONAL_INPUT(activation_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(weight_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(expand_scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(shared_expert_x, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(elastic_info, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(ori_x, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(const_expert_alpha_1, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(const_expert_alpha_2, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(const_expert_v, TensorType({DT_BF16}))
    .OUTPUT(y, TensorType({DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT}))
    .OUTPUT(x, TensorType({DT_BF16}))
    .REQUIRED_ATTR(group_ep, String)
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .ATTR(group_tp, String, "")
    .ATTR(tp_world_size, Int, 0)
    .ATTR(tp_rank_id, Int, 0)
    .ATTR(expert_shard_type, Int, 0)
    .ATTR(shared_expert_num, Int, 1)
    .ATTR(shared_expert_rank_num, Int, 0)
    .ATTR(global_bs, Int, 0)
    .ATTR(out_dtype, Int, 0)
    .ATTR(comm_quant_mode, Int, 0)
    .ATTR(group_list_type, Int, 0)
    .ATTR(comm_alg, String, "")
    .ATTR(norm_eps, Float, 1e-6f)
    .ATTR(zero_expert_num, Int, 0)
    .ATTR(copy_expert_num, Int, 0)
    .ATTR(const_expert_num, Int, 0)
    .OP_END_FACTORY_REG(MoeDistributeCombineAddRmsNorm)

}  // namespace ge


#endif  // MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM_PROTO_H_
