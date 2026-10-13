/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"
#include "nlohmann/json.hpp"

using namespace ge;
using json = nlohmann::json;

namespace domi {
using NodeProto = ge::onnx::NodeProto;
static const int REQ_ATTR_NUM = 1;

static Status ParseParamsNpuRotaryPositionEmbedding(const Message* op_src, ge::Operator& op_dest) {
    const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
    if (node == nullptr) {
      OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
      return FAILED;
    }
    // The initialization of required attributes count
    int req_attr_count = 0;
    for (const auto& attr: node->attribute()) {
      if (attr.name() == "mode" && attr.type() == ge::onnx::AttributeProto::INT) {
        int mode = attr.i();
        op_dest.SetAttr("mode", mode);
        ++req_attr_count;
      }
    }
    // Node must have required attribute mode
    if (req_attr_count != REQ_ATTR_NUM) {
      OP_LOGE(GetOpName(op_dest).c_str(), "Node must have attr mode.");
      return FAILED;
    }
    return SUCCESS;
}

// register npu_rotary_position_embedding op info to GE
REGISTER_CUSTOM_OP("RotaryPositionEmbedding")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPURotaryPositionEmbedding"),
                 ge::AscendString("ai.onnx::11::NPURotaryPositionEmbedding"),
                 ge::AscendString("ai.onnx::12::NPURotaryPositionEmbedding"),
                 ge::AscendString("ai.onnx::13::NPURotaryPositionEmbedding"),
                 ge::AscendString("ai.onnx::14::NPURotaryPositionEmbedding"),
                 ge::AscendString("ai.onnx::15::NPURotaryPositionEmbedding"),
                 ge::AscendString("ai.onnx::16::NPURotaryPositionEmbedding"),
                 ge::AscendString("ai.onnx::17::NPURotaryPositionEmbedding"),
                 ge::AscendString("ai.onnx::18::NPURotaryPositionEmbedding"),
                })
  .ParseParamsFn(ParseParamsNpuRotaryPositionEmbedding)
  .ImplyType(ImplyType::TVM);
} // namespace V2domi