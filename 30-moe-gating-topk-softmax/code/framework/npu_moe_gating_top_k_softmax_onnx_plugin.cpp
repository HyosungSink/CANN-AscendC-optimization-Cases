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
 * \file npu_moe_gating_top_k_softmax_onnx_plugin.cpp
 * \brief
 */
#include "../common/include/framework/onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status  ParseParamsMoeGatingTopKSoftmax(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("MoeGatingTopKSoftmax", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int k = -1;
  for (auto& attr : node->attribute()) {
    if (attr.name() == "k" && attr.type() == ge::onnx::AttributeProto::INT) {
      k = attr.i();
    }
  }
  if (k == -1) {
    return FAILED;
  }
  op_dest.SetAttr("k", k);
  return SUCCESS;
}

// register npu_flash_attention_score op info to GE
REGISTER_CUSTOM_OP("MoeGatingTopKSoftmax")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUMoeGatingTopKSoftmax"), 
                 ge::AscendString("ai.onnx::11::NPUMoeGatingTopKSoftmax"),
                 ge::AscendString("ai.onnx::12::NPUMoeGatingTopKSoftmax"),
                 ge::AscendString("ai.onnx::13::NPUMoeGatingTopKSoftmax"),
                 ge::AscendString("ai.onnx::14::NPUMoeGatingTopKSoftmax"),
                 ge::AscendString("ai.onnx::15::NPUMoeGatingTopKSoftmax"),
                 ge::AscendString("ai.onnx::16::NPUMoeGatingTopKSoftmax"),
                 ge::AscendString("ai.onnx::17::NPUMoeGatingTopKSoftmax"),
                 ge::AscendString("ai.onnx::18::NPUMoeGatingTopKSoftmax")})
  .ParseParamsFn(ParseParamsMoeGatingTopKSoftmax)
  .ImplyType(ImplyType::TVM);
}  // namespace domi