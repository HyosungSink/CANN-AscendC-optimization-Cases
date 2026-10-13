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
 * \file npu_rms_norm_plugin.cc
 * \brief
 */
#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsNpuRmsNorm(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  float epsilon = 1e-6;
  op_dest.SetAttr("epsilon", epsilon);
  return SUCCESS;
}

// register npu_rms_norm op info to GE
REGISTER_CUSTOM_OP("RmsNorm")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPURmsNorm"),
                 ge::AscendString("ai.onnx::11::NPURmsNorm"),
                 ge::AscendString("ai.onnx::12::NPURmsNorm"),
                 ge::AscendString("ai.onnx::13::NPURmsNorm"),
                 ge::AscendString("ai.onnx::14::NPURmsNorm"),
                 ge::AscendString("ai.onnx::15::NPURmsNorm"),
                 ge::AscendString("ai.onnx::16::NPURmsNorm"),
                 ge::AscendString("ai.onnx::17::NPURmsNorm"),
                 ge::AscendString("ai.onnx::18::NPURmsNorm")})
  .ParseParamsFn(ParseParamsNpuRmsNorm)
  .ImplyType(ImplyType::TVM);
} // namespace domi