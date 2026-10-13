/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qkv_rms_norm_rope_cache_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_QKV_RMS_NORM_ROPE_CACHE_H_
#define OPS_OP_PROTO_INC_QKV_RMS_NORM_ROPE_CACHE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief The fusion operator of SpiltVD, RMSNorm, RotaryPositionEmbedding, Quant and Update KVCache.
 *
 * @par Inputs:
 * @li qkv: A tensor. The type support float16 and bfloat16. Its shape is [Bqkv*Sqkv,Nqkv*Dqkv]. Format: ND.
 * @li q_gamma: A tensor, used in RMS Norm. Its type is consistent with qkv. Its shape is [Dqkv,]. Format: ND.
 * @li k_gamma: A tensor, used in RMS Norm. Its type is consistent with qkv. Its shape is [Dqkv,]. Format: ND.
 * @li cos: A tensor, from position embedding. Its type is consistent with kv. Its shape is [Bqkv*Sqkv,Dqkv]. Format: ND.
 * @li sin: A tensor, from position embedding. Its type is consistent with qkv. [Bqkv*Sqkv,Dqkv]. 
 * The shape of sin should be consistent with that of cos. Format: ND.
 * @li index: A tensor. The type support int64. When the cache_mode is PA_NZ, its shape is [Bqkv*Sqkv] Format: ND.
 * @li q_out: A tensor. Its type is consistent with qkv. Its shape is [Bqkv*Sqkv, Nq*Dqkv]. Format: ND.
 * @li k_cache: A tensor. Its type is consistent with qkv or int8. When the type is int8, 
 * its shape is [BlockNum, Nk * Dqkv // 32, BlockSize, 32]. when the type consistent with qkv,
 * its shape is [BlockNum, Nk * Dqkv // 16, BlockSize, 16]. Format: ND.
 * @li v_cache: A tensor. Its type is consistent with qkv or int8. When the type is int8, 
 * its shape is [BlockNum, Nv * Dqkv // 32, BlockSize, 32]. when the type consistent with qkv,
 * its shape is [BlockNum, Nv * Dqkv // 16, BlockSize, 16]. Format: ND.
 * @li k_scale: A tensor. The type support float32. This input parameter is required when the data type of k_cache
 * is int8. Its shape can be [Nk,Dqkv]. Format: ND.
 * @li v_scale: A tensor. The type support float32. This input parameter is required when the data type of v_cache
 * is int8. Its shape can be [Nv,Dqkv]. Format: ND.
 * @li k_offset: A tensor. The type support float32. When the data type of k_cache is int8 and the corresponding
 * input exists with the quantization scenario being asymmetric quantization, this parameter input is required. Its
 * shape can be [Nk,Dqkv]. Format: ND.
 * @li v_offset: A tensor. The type support float32. When the data type of v_cache is int8 and the corresponding
 * input exists with the quantization scenario being asymmetric quantization, this parameter input is required. Its
 * shape can be [Nv,Dqkv]. Format: ND.
 *
 * @par Outputs:
 * @li q_out: A tensor. Its type is consistent with qkv. Its shape is [Bqkv*Sqkv, Nq*Dqkv]. Format: ND.
 * @li k_cache: A tensor. Its type is consistent with qkv or int8. When the type is int8, 
 * its shape is [BlockNum, Nk * Dqkv // 32, BlockSize, 32]. when the type consistent with qkv,
 * its shape is [BlockNum, Nk * Dqkv // 16, BlockSize, 16]. Format: ND.
 * @li v_cache: A tensor. Its type is consistent with qkv or int8. When the type is int8, 
 * its shape is [BlockNum, Nk * Dqkv // 32, BlockSize, 32]. when the type consistent with qkv,
 * its shape is [BlockNum, Nk * Dqkv // 16, BlockSize, 16]. Format: ND.
 * @li q_out_before_quant: A tensor. Its type is consistent with qkv. If is_output_qkv is true, this parameter output is required.
 * Its shape is [Bqkv*Sqkv, Nq*Dqkv]. Format: ND.
 * @li k_out_before_quant: A tensor. Its type is consistent with qkv. If is_output_qkv is true, this parameter output is required.
 * Its shape is [Bqkv*Sqkv, Nk*Dqkv]. Format: ND.
 * @li v_out_before_quant: A tensor. Its type is consistent with qkv. If is_output_qkv is true, this parameter output is required.
 * Its shape is [Bqkv*Sqkv, Nv*Dqkv]. Format: ND.
 *
 * @par Attributes:
 * @li qkv_size: A Listint. Provide the specific dimension sizes of the input of qkv matrices in the order of [Bqkv, Sqkv, Nqkv, Dqkv].
 * @li head_nums: A Listint. Specifying the N-dim,ensional sizes of the q, k, v components in the qkv input matrix, in the order [Nq, Nk, Nv].
 * @li epsilon: A float32. The epsilon value for RMSNorm. Default: 1e-6.
 * @li cache_mode: A string. The cache dtype. There is one types：PA_NZ. Default: PA_NZ.
 * @li is_output_qkv: A bool. Control whether k_cache_proto and v_cache_proto are output. If it is true, then k_out_before_quant and v_out_before_quant need to be
 * output. If it is false, then k_cache_proto and v_cache_proto do not need to be output. Default: false.
 *
 * @attention Constraints:
 * @li The shape format description in the parameter description:
 *     - Bqkv represents the batch size of the input qkv, and Skv represents the sequence length of the input qkv. The
 *     size is determined by the user's input scenario and has no explicit restrictions.
 *     - Nqkv is the head number of the input qkv. This operator is strongly related to the Qwen3-Moe 235b network structure.
 *     - Dqkv is the head dim of the input qkv. The data Dqkv required for rms norm calculation and the data Dqkv required
 *     for rope calculation are split from the Dqkv of the input qkv. In rope calculation, Drope=Dqkv.
 *     Therefore, Dqkv must comply with the rope rules. According to the rope rule, Dqkv is an even number.
 *     If cache_mode is an NZ scenario(cahce_mode is PA_NZ), Dqkv needs to be 32B aligned.
 *     - If cache_mode is a PA scenario, the BlockSize needs to be 32B aligned.
 *     - Regarding the above-mentioned 32B alignment situation, the alignment value is determined by the data type
 *     of the cache. Take BlockSize as an example. If the data type of the cache is int8, then BlockSize%32=0 is
 *     required. If the data type of the cache is flaot16, then BlockSize%16=0 is required.
 *     - BlockNum is the number of memory blocks written to the cache, and its size is determined by the user's input
 *     scenario without any definite limit.
 * @li The index-related constraints:
 *     When cache_mode is PA_NZ, the shape is [Bqkv*Sqkv], and the value range of the index must be
 *     [-1,BlockNum*BlockSize). The values cannot be repeated.
 */
REG_OP(QkvRmsNormRopeCache)
    .INPUT(qkv, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(q_gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(k_gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(index, TensorType({DT_INT64}))
    .INPUT(q_out, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(k_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(v_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(k_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(v_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(k_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(v_offset, TensorType({DT_FLOAT}))
    .OUTPUT(q_out, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(k_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(v_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(q_out_before_quant, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(k_out_before_quant, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(v_out_before_quant, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(qkv_size, ListInt)
    .REQUIRED_ATTR(head_nums, ListInt)
    .ATTR(epsilon, Float, 1e-6)
    .ATTR(cache_mode, String, "PA_NZ")
    .ATTR(is_output_qkv, Bool, false)
    .OP_END_FACTORY_REG(QkvRmsNormRopeCache)
} // namespace ge

#endif // OPS_OP_PROTO_INC_QKV_RMS_NORM_ROPE_CACHE_H_