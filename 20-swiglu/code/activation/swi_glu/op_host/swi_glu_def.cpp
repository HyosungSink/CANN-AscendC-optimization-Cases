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
 * \file swi_glu.cpp
 * \brief
 */
#include <register/op_def_registry.h>

namespace ops {

class SwiGlu : public OpDef {
public:
    explicit SwiGlu(const char* name) : OpDef(name)
    {
      this->Input("x")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .AutoContiguous();
      this->Output("y")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
      this->Attr("dim")
          .AttrType(OPTIONAL)
          .Int(-1);
      this->AICore().AddConfig("ascend910b");
      this->AICore().AddConfig("ascend910_93");

      OpAICoreConfig config_without_bf16;
      config_without_bf16.Input("x")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND})
          .AutoContiguous();
      config_without_bf16.Output("y")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND});
      config_without_bf16.DynamicCompileStaticFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true);
      this->AICore().AddConfig("ascend310p", config_without_bf16);
      this->AICore().AddConfig("kirinx90", config_without_bf16);
      this->AICore().AddConfig("kirin9030", config_without_bf16);

      OpAICoreConfig regbaseCfg;
      regbaseCfg.DynamicCompileStaticFlag(true)
              .DynamicRankSupportFlag(true)
              .DynamicShapeSupportFlag(true)
              .ExtendCfgInfo("opFile.value", "swi_glu_apt");
      this->AICore().AddConfig("ascend950", regbaseCfg);
    }
};
OP_ADD(SwiGlu);
}
