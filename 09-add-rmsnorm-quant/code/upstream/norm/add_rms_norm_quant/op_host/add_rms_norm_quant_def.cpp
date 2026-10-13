/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_quant_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::DataType> xDataTypeRegbase = {
ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT16,  ge::DT_BF16,
ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT16,  ge::DT_BF16,
ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT16,  ge::DT_BF16,
ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT16,  ge::DT_BF16
};
static const std::vector<ge::DataType> scalesDataTypeRegbase = {
ge::DT_FLOAT,    ge::DT_FLOAT,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT,
ge::DT_FLOAT,    ge::DT_FLOAT,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT,
ge::DT_FLOAT,    ge::DT_FLOAT,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT,
ge::DT_FLOAT,    ge::DT_FLOAT,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT
};
static const std::vector<ge::DataType> zeroPointsDataTypeRegbase = {
ge::DT_INT32,    ge::DT_INT32,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT,
ge::DT_INT32,    ge::DT_INT32,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT,
ge::DT_INT32,    ge::DT_INT32,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT,
ge::DT_INT32,    ge::DT_INT32,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT
};
static const std::vector<ge::DataType> yDataTypeRegbase = {
ge::DT_INT8,     ge::DT_INT8,     ge::DT_INT8,     ge::DT_INT8,     ge::DT_INT8,     ge::DT_INT8,     ge::DT_INT8,
ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8
};
static const std::vector<ge::Format> formatRegbase = {
ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,
ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,
ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,
ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND
};

class AddRmsNormQuant : public OpDef {
public:
    explicit AddRmsNormQuant(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("scales1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("scales2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("zero_points1")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_BF16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("zero_points2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_BF16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("beta")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("axis").AttrType(OPTIONAL).Int(-1);
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6);
        this->Attr("div_mode").AttrType(OPTIONAL).Bool(true);
        this->Attr("dst_type").AttrType(OPTIONAL).Int(ge::DT_INT8);
        
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

        OpAICoreConfig config310P;
        config310P.Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Input("scales1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Input("scales2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Input("zero_points1")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Input("zero_points2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Input("beta")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Output("y1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Output("y2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Output("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.DynamicCompileStaticFlag(true).DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend310p", config310P);
        this->AICore().AddConfig("kirinx90", config310P);
        this->AICore().AddConfig("kirin9030", config310P);

        OpAICoreConfig configRegbase;
        configRegbase.Input("x1")
            .ParamType(REQUIRED)
            .DataType(xDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase);
        configRegbase.Input("x2")
            .ParamType(REQUIRED)
            .DataType(xDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase);
        configRegbase.Input("gamma")
            .ParamType(REQUIRED)
            .DataType(xDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase);
        configRegbase.Input("scales1")
            .ParamType(REQUIRED)
            .DataType(scalesDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase);
        configRegbase.Input("scales2")
            .ParamType(OPTIONAL)
            .DataType(scalesDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase);
        configRegbase.Input("zero_points1")
            .ParamType(OPTIONAL)
            .DataType(zeroPointsDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase);
        configRegbase.Input("zero_points2")
            .ParamType(OPTIONAL)
            .DataType(zeroPointsDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase);
        configRegbase.Input("beta")
            .ParamType(OPTIONAL)
            .DataType(xDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase);
        configRegbase.Output("y1")
            .ParamType(REQUIRED)
            .DataType(yDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase);
        configRegbase.Output("y2")
            .ParamType(REQUIRED)
            .DataType(yDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase);
        configRegbase.Output("x")
            .ParamType(REQUIRED)
            .DataType(xDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase);
        configRegbase.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "add_rms_norm_quant_apt");

        this->AICore().AddConfig("ascend950", configRegbase);
        this->AICore().AddConfig("ascend910_55", configRegbase);
    }
};
OP_ADD(AddRmsNormQuant);
} // namespace ops