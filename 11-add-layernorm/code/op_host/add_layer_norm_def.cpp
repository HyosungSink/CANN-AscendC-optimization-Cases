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
 * \file add_layer_norm_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class AddLayerNorm : public OpDef {
#define ALL_FORMAT_ND_910                                                                                        \
    {                                                                                                            \
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, \
            ge::FORMAT_ND, ge::FORMAT_ND                                                                         \
    }
#define ALL_FORMAT_ND_310            \
    {                                \
        ge::FORMAT_ND, ge::FORMAT_ND \
    }

public:
    explicit AddLayerNorm(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910);
        this->Output("mean")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910);
        this->Output("rstd")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910);
        this->Output("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910);
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-5);
        this->Attr("additional_output").AttrType(OPTIONAL).Bool(false);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

    }
};

OP_ADD(AddLayerNorm);
} // namespace ops