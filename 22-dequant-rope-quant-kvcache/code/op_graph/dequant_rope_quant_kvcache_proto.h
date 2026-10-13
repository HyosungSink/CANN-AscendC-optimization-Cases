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
 * \file dequant_rope_quant_kvcache_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_DEQUANT_ROPE_OPS_H_
#define OPS_OP_PROTO_INC_DEQUANT_ROPE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief DequantRopeQuantKvcache.

* @par Inputs:
* thirteen inputs, including:
* @li x: A Tensor with shape (B, S, H) or (B, H), H is (Nq+Nkv+Nkv)*D, format support ND.
* The type support float16, bf16, int32.
* @li cos: A Tensor with shape (B, S, 1, D) or (B, D). The type support float16, bf16, format support ND.
* @li sin: A Tensor with shape (B, S, 1, D) or (B, D). The type support float16, bf16, format support ND.
* @li k_cache: A Tensor with shape (C_1, C_2, Nkv, D) indicates kcache for in-place updates.
* The type support int8, format support ND.
* @li v_cache: A Tensor with shape (C_1, C_2, Nkv, D) indicates vcache for in-place updates.
* The type support int8, format support ND.
* @li indices: A Tensor with shape (B) when cache_mode is contiguous with shape (B * S) when cache_mode is page.
* The type support int32, format support ND.
* @li scale_k: A Tensor with shape (Nkv, D). The type support float32, format support ND.
* @li scale_v: A Tensor with shape (Nkv, D). The type support float32, format support ND.
* @li offset_k: A Tensor with shape (Nkv, D). An optional input parameter. The type support float32.
* format support ND.
* @li offset_v: A Tensor with shape (Nkv, D). An optional input parameter. The type support float32.
* format support ND.
* @li weight_scale: A Tensor with shape (D) indicates the weight scale factor of the dequantization parameter.
* An optional input parameter. The type support float32.
* @li activation_scale: A Tensor with shape (B * S) or (B) indicates the activation scale factor of the dequantization
parameter.
* An optional input parameter. The type support float32.
* @li bias: A Tensor with shape (D). An optional input parameter. The type support float32, bf16, float16, int32.

* @par Attributes:
* @li size_splits: A list of int. Specifies the size of spliting qkv.
* @li quant_mode: A string. A optional attribute. Specifies the method of quant. Default: "static".
* @li layout: A string. A optional attribute. Specifies the format of input. Default: "BSND".
* @li kv_output: A bool. A optional attribute. Whether to output kv. Default: "false".
* @li cache_mode:  A string. A optional attribute. Specifies the cache mode for kcache and vcache.
*    Should be "contiguous" or "page", default is "contiguous".

* @par Outputs:
* @li q: A Tensor with shape (B, S, Nq, D) or (B, Nq, D). The type support float16, bf16.
* @li k: A Tensor with shape (B, S, Nkv, D) or (B, Nkv, D). The type support float16, bf16.
* @li v: A Tensor with shape (B, S, Nkv, D) or (B, Nkv, D). The type support float16, bf16.
* @li k_cache: A Tensor with shape (C_1, C_2, Nkv, D). The type support int8, format support ND.
* @li v_cache: A Tensor with shape (C_1, C_2, Nkv, D). The type support int8, format support ND.
*/
REG_OP(DequantRopeQuantKvcache)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(k_cache, TensorType({DT_INT8}))
    .INPUT(v_cache, TensorType({DT_INT8}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(scale_k, TensorType({DT_FLOAT32}))
    .INPUT(scale_v, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(offset_k, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(offset_v, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(weight_scale, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(activation_scale, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT32, DT_BF16, DT_FLOAT16, DT_INT32}))
    .OUTPUT(q, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(k, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(v, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(k_cache, TensorType({DT_INT8}))
    .OUTPUT(v_cache, TensorType({DT_INT8}))
    .REQUIRED_ATTR(size_splits, ListInt)
    .ATTR(quant_mode, String, "static")
    .ATTR(layout, String, "BSND")
    .ATTR(kv_output, Bool, false)
    .ATTR(cache_mode, String, "contiguous")
    .OP_END_FACTORY_REG(DequantRopeQuantKvcache)
} // namespace ge

#endif // OPS_OP_PROTO_INC_DEQUANT_ROPE_OPS_H_