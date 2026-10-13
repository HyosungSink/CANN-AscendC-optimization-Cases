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
 * \file flash_attention_score_proto.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_PROTO_H_
#define FLASH_ATTENTION_SCORE_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Fast and Memory-Efficient Exact Attention with IO-Awareness.

* @par Inputs:
* twelve inputs, including:
* @li query: A matrix Tensor. The type support float16, bf16, float32.
* @li key: A matrix Tensor. The type support float16, bf16, float32.
* @li value: A matrix Tensor. The type support float16, bf16, float32.
* @li real_shift: A matrix Tensor. An optional input parameter. The type support float16, bf16, float32.
* @li drop_mask: A matrix Tensor. An optional input parameter. The type support uint8.
* @li padding_mask: A matrix Tensor. An optional input parameter. The type support float16, bf16, float32.
* @li atten_mask: A matrix Tensor. An optional input parameter. The type support bool, uint8.
* @li prefix: A matrix Tensor. An optional input parameter. The type support int64.
* @li actual_seq_qlen: A matrix Tensor. An optional input parameter. The type support int64. If used,
* layout need to be setted TND. ex. If the q seqlen is [2,2,2,2,2], this parameter need be setted [2,4,6,8,10].
* @li actual_seq_kvlen: A matrix Tensor. An optional input parameter. The type support int64. If used,
* layout need to be setted TND. ex. If the kv seqlen is [2,2,2,2,2], this parameter need be setted [2,4,6,8,10].
* @li q_start_idx: A matrix Tensor. An optional input parameter. The type support int64.
* @li kv_start_idx: A matrix Tensor. An optional input parameter. The type support int64.
* @li d_scale_q: A matrix Tensor. An optional input parameter. The type support float32.
* @li d_scale_k: A matrix Tensor. An optional input parameter. The type support float32.
* @li d_scale_v: A matrix Tensor. An optional input parameter. The type support float32.
* @li queryRope: A matrix Tensor. An optional input parameter. The type support float16, bf16, float32.
* @li keyRope: A matrix Tensor. An optional input parameter. The type support float16, bf16, float32.

* @par Attributes:
* @li scale_value: A float. The scale value. Default: 1.0.
* @li keep_prob: A float. The keep probability of dropout. Default: 1.0.
* @li pre_tockens: An int. Previous tokens.
* @li next_tockens: An int. Next tokens.
* @li head_num: An int. A required attribute. The number of the heads.
* @li input_layout: A string. A required attribute. Specifies the layout of `query`, the value must be one of ["BSH",
* "SBH", "BNSD", "BSND", "TND"].
* @li inner_precise: An int. 0, 1, reserved value. 2, support invalid lines.
* @li sparse_mode: An int. 0, defaultMsk. 1, allMask. 2, leftUpCausal. 3, rightDownCausal. 4, band. 5, prefix.
* 6, prefixCompress. 7, rightDownCausalBand. 8, bandLeftUpCausal.
* @li pse_type: An int. Optional attribute. Users can pass in 1 if they do not specify it.
* The supported configuration values ​​are 0, 1, 2, and 3.
* @li seed: An int. Optional attribute. Default: 0.
* @li offset: An int. Optional attribute.  Default: 0.
* @li out_dtype: An int. Optional attribute.  Default: 0.
* @li softmax_out_layout: A string. Optional attribute.  Default: "", the value must be one of ["", "same_as_input"].

* @par Outputs:
* @li softmax_max: A matrix Tensor. The type support float32.
* @li softmax_sum: A matrix Tensor. The type support float32.
* @li softmax_out: A matrix Tensor. The type support float16, bf16, float32.
* @li attention_out: A matrix Tensor. The type support float16, bf16, float32.
*/
REG_OP(FlashAttentionScore)
    .INPUT(query, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(key, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .INPUT(value, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(real_shift, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_UINT8}))
    .OPTIONAL_INPUT(prefix, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_qlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_kvlen, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(q_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(d_scale_q, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_k, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(d_scale_v, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(p_scale, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_max, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OUTPUT(softmax_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(pre_tockens, Int, 2147483647)
    .ATTR(next_tockens, Int, 2147483647)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(input_layout, String)
    .ATTR(inner_precise, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(pse_type, Int, 1)
    .ATTR(seed, Int, 0)
    .ATTR(offset, Int, 0)
    .ATTR(out_dtype, Int, 0)
    .ATTR(softmax_out_layout, String, "")
    .OP_END_FACTORY_REG(FlashAttentionScore)
}  // namespace ge

#endif  // FLASH_ATTENTION_SCORE_PROTO_H_