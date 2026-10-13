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
 * \file npu_group_norm_silu_plugin.cc
 * \brief
 */

#include "../common/inc/framework/onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsNpuGroupNormSilu(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic op_src to NodeProto failed.");
    return FAILED;
  }

  int group = 0;
  float eps = 0.00001;

  for (auto attr : node->attribute()) {
    if ((attr.name() == "group" || attr.name() == "num_groups") && attr.type() == ge::onnx::AttributeProto::INT) {
      group = attr.i();
    } else if (attr.name() == "eps" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      eps = attr.f();
    }
  }
  op_dest.SetAttr("num_groups", group);
  op_dest.SetAttr("eps", eps);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("GroupNormSilu")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUGroupNormSilu"),
                   ge::AscendString("ai.onnx::11::NPUGroupNormSilu"),
                   ge::AscendString("ai.onnx::12::NPUGroupNormSilu"),
                   ge::AscendString("ai.onnx::13::NPUGroupNormSilu"),
                   ge::AscendString("ai.onnx::14::NPUGroupNormSilu"),
                   ge::AscendString("ai.onnx::15::NPUGroupNormSilu"),
                   ge::AscendString("ai.onnx::16::NPUGroupNormSilu"),
                   ge::AscendString("ai.onnx::17::NPUGroupNormSilu"),
                   ge::AscendString("ai.onnx::18::NPUGroupNormSilu")})
    .ParseParamsFn(ParseParamsNpuGroupNormSilu)
    .ImplyType(ImplyType::TVM);
}  // domi