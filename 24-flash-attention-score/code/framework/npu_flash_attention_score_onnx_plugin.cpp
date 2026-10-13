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
 * \file npu_flash_attention_score_onnx_plugin.cpp
 * \brief
 */
#include "../common/include/framework/onnx_common.h"
#include "../op_graph/flash_attention_score_proto.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
constexpr int REQUIRED_ATTR = 2;
constexpr int ACL_UINT8 = 4;
constexpr int ONE = 1;
constexpr int ALIGN_NUM = 128;
constexpr int ONE_BYTE_BITS = 8;
constexpr int EXTRA_LENGTH = 32;
constexpr int OUTPUT_INDEX = 3;

namespace{
static void UpdateFlashAttentionByNode(ge::Operator& op_dest, const NodeProto* node) {
  int input_size = node->input_size();
  int output_size = node->output_size();
  op_dest.DynamicInputRegister("x", input_size);
  op_dest.DynamicOutputRegister("y", output_size);
  op_dest.SetAttr("name", node->name());
  op_dest.SetAttr("original_type", "npu::1::NPUFlashAttention");
}

static Status GetAttrFromPre4(
  const ge::Operator& op, int& head_num, string& input_layout, float& scale, float& keep_prob) {
  if (op.GetAttr("head_num", head_num) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get head_num from op failed");
    return FAILED;
  }
  if (op.GetAttr("input_layout", input_layout) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get input_layout from op failed");
    return FAILED;
  }
  if (op.GetAttr("scale", scale) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get scale from op failed");
    return FAILED;
  }
  if (op.GetAttr("keep_prob", keep_prob) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get keep_prob from op failed");
    return FAILED;
  }
  return SUCCESS;
}

static Status GetAttrFromLast4(
  const ge::Operator& op, int& pre_tockens, int& next_tockens, int& inner_precise, int& sparse_mode) {
  if (op.GetAttr("pre_tockens", pre_tockens) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get pre_tockens from op failed");
    return FAILED;
  }
  if (op.GetAttr("next_tockens", next_tockens) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get next_tockens from op failed");
    return FAILED;
  }
  if (op.GetAttr("inner_precise", inner_precise) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get inner_precise from op failed");
    return FAILED;
  }
  if (op.GetAttr("sparse_mode", sparse_mode) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get sparse_mode from op failed");
    return FAILED;
  }
  return SUCCESS;
}

static Status GetFinalDimsByOperator(
  const ge::Operator& op, const std::string& input_layout, int head_num, vector<int64_t>& final_dims) {
  std::vector<int64_t> dims = op.GetInputDesc(0).GetShape().GetDims();
  int64_t numels = 0;
  if (input_layout == "BSH") {
    numels = dims[0] * dims[1] * dims[1] * head_num;
  } else if (input_layout == "SBH") {
    numels = dims[1] * dims[0] * dims[0] * head_num;
  } else {
    OP_LOGE(GetOpName(op).c_str(), "input_layer is not support");
    return FAILED;
  }
  int64_t length = (numels + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM / ONE_BYTE_BITS;
  length += EXTRA_LENGTH;
  final_dims = {length};
  return SUCCESS;
}

static Status GetOriNameFromOperator(const ge::Operator& op, std::string& ori_name) {
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
    return FAILED;
  }
  return SUCCESS;
}
}

static Status ParseParamsFlashAttention(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("FlashAttention", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int head_num = 0;
  std::string input_layout = "";
  float scale = 1.0f;
  float keep_prob = 1.0f;
  int pre_tockens = 0;
  int next_tockens = 0;
  int required_attr_num = 0;
  int inner_precise = 1;
  int sparse_mode = 0;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "head_num" && attr.type() == ge::onnx::AttributeProto::INT) {
      head_num = attr.i();
      ++required_attr_num;
    } else if (attr.name() == "input_layout" && attr.type() == ge::onnx::AttributeProto::STRING) {
      input_layout = attr.s();
      ++required_attr_num;
    } else if (attr.name() == "scale" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      scale = attr.f();
    } else if (attr.name() == "keep_prob" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      keep_prob = attr.f();
    } else if (attr.name() == "pre_tockens" && attr.type() == ge::onnx::AttributeProto::INT) {
      pre_tockens = attr.i();
    } else if (attr.name() == "next_tockens" && attr.type() == ge::onnx::AttributeProto::INT) {
      next_tockens = attr.i();
    } else if (attr.name() == "inner_precise" && attr.type() == ge::onnx::AttributeProto::INT) {
      inner_precise = attr.i();
    } else if (attr.name() == "sparse_mode" && attr.type() == ge::onnx::AttributeProto::INT) {
      sparse_mode = attr.i();
    }
  }

  if (required_attr_num != REQUIRED_ATTR) {
    OP_LOGE(GetOpName(op_dest).c_str(), "attr num two is required.");
    return FAILED;
  }

  UpdateFlashAttentionByNode(op_dest, node);

  op_dest.SetAttr("head_num", head_num);
  op_dest.SetAttr("input_layout", input_layout);
  op_dest.SetAttr("scale", scale);
  op_dest.SetAttr("keep_prob", keep_prob);
  op_dest.SetAttr("pre_tockens", pre_tockens);
  op_dest.SetAttr("next_tockens", next_tockens);
  op_dest.SetAttr("inner_precise", inner_precise);
  op_dest.SetAttr("sparse_mode", sparse_mode);
  return SUCCESS;
}

static Status ParseOpToGraphNpuFlashAttentionScore(const ge::Operator& op, ge::Graph& graph) {
  std::string ori_name;
  if (GetOriNameFromOperator(op, ori_name)) {
    return FAILED;
  }

  auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
  auto data1 = ge::op::Data((ori_name + "_data1").c_str()).set_attr_index(1);
  auto data2 = ge::op::Data((ori_name + "_data2").c_str()).set_attr_index(2);
  auto data3 = ge::op::Data((ori_name + "_data3").c_str()).set_attr_index(3);
  auto data4 = ge::op::Data((ori_name + "_data4").c_str()).set_attr_index(4);
  auto data5 = ge::op::Data((ori_name + "_data5").c_str()).set_attr_index(5);
  auto data6 = ge::op::Data((ori_name + "_data6").c_str()).set_attr_index(6);
  auto data7 = ge::op::Data((ori_name + "_data7").c_str()).set_attr_index(7);
  auto data8 = ge::op::Data((ori_name + "_data8").c_str()).set_attr_index(8);

  int head_num = 0;
  string input_layout = "BSH";
  float scale = 1.0f;
  float keep_prob = 1.0f;
  int pre_tockens = 0;
  int next_tockens = 0;
  int inner_precise = 1;
  int sparse_mode = 0;
  if ((GetAttrFromPre4(op, head_num, input_layout, scale, keep_prob) != SUCCESS) ||
      (GetAttrFromLast4(op, pre_tockens, next_tockens, inner_precise, sparse_mode) != SUCCESS)) {
    return FAILED;
  }
  // create const input tensor "drop_mask" which is filled with the scalar value 1 for inferencing
  // deop_mask.size = [B, N, S, S]
  ge::Tensor saclar_one = CreateScalar(ONE, ge::DT_UINT8);
  auto const_one = ge::op::Const((ori_name + "_Const_one").c_str()).set_attr_value(saclar_one);
  vector<int64_t> final_dims;
  if (GetFinalDimsByOperator(op, input_layout, head_num, final_dims) != SUCCESS) {
    return FAILED;
  }

  auto tensor_dims = Vec2Tensor(final_dims, {1}, ge::DT_INT64);
  auto const_dims = ge::op::Const((ori_name + "_Const_dims").c_str()).set_attr_value(tensor_dims);
  auto drop_mask = ge::op::Fill((ori_name + "_Fill_ones").c_str()).set_input_dims(const_dims).set_input_value(const_one);
  auto cast_drop_mask = ge::op::Cast((ori_name + "_Cast_drop_mask").c_str()).set_input_x(drop_mask)
                                                                            .set_attr_dst_type(ACL_UINT8);

  auto AttentionScore = ge::op::FlashAttentionScore((ori_name + "_FlashAttentionScore").c_str())
                            .set_input_query(data0).set_input_key(data1).set_input_value(data2)
                            .set_input_real_shift(data3).set_input_drop_mask(cast_drop_mask)
                            .set_input_padding_mask(data4).set_input_atten_mask(data5).set_input_prefix(data6)
                            .set_input_actual_seq_qlen(data7).set_input_actual_seq_kvlen(data8)
                            .set_attr_scale_value(scale).set_attr_keep_prob(keep_prob).set_attr_pre_tockens(pre_tockens)
                            .set_attr_next_tockens(next_tockens).set_attr_head_num(head_num)
                            .set_attr_inner_precise(inner_precise).set_attr_sparse_mode(sparse_mode);
  std::vector<ge::Operator> inputs{data0, data1, data2, data3, data4, data5, data6, data7, data8};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
  outputs.emplace_back(AttentionScore, std::vector<std::size_t>{OUTPUT_INDEX});
  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

// register npu_flash_attention_score op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUFlashAttention"), 
                 ge::AscendString("ai.onnx::11::NPUFlashAttention"),
                 ge::AscendString("ai.onnx::12::NPUFlashAttention"),
                 ge::AscendString("ai.onnx::13::NPUFlashAttention"),
                 ge::AscendString("ai.onnx::14::NPUFlashAttention"),
                 ge::AscendString("ai.onnx::15::NPUFlashAttention"),
                 ge::AscendString("ai.onnx::16::NPUFlashAttention"),
                 ge::AscendString("ai.onnx::17::NPUFlashAttention"),
                 ge::AscendString("ai.onnx::18::NPUFlashAttention")})
  .ParseParamsFn(ParseParamsFlashAttention)
  .ParseOpToGraphFn(ParseOpToGraphNpuFlashAttentionScore)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
