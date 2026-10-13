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
 * \file add_rms_norm_dynamic_quant_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> xDataType91095 = {
ge::DT_FLOAT16,     ge::DT_BF16,        ge::DT_FLOAT16,       ge::DT_BF16,
ge::DT_FLOAT16,     ge::DT_BF16,        ge::DT_FLOAT16,       ge::DT_BF16
};
static const std::vector<ge::DataType> scalesOutDataType91095 = {
ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,         ge::DT_FLOAT,
ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,         ge::DT_FLOAT
};
static const std::vector<ge::DataType> yDataType91095 = {
ge::DT_INT8,        ge::DT_INT8,        ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8
};
static const std::vector<ge::Format> format91095 = {
ge::FORMAT_ND,      ge::FORMAT_ND,      ge::FORMAT_ND,        ge::FORMAT_ND,
ge::FORMAT_ND,      ge::FORMAT_ND,      ge::FORMAT_ND,        ge::FORMAT_ND
};
class AddRmsNormDynamicQuant : public OpDef {
public:
    explicit AddRmsNormDynamicQuant(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("smooth_scale1")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("smooth_scale2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("beta")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT4, ge::DT_INT4})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT4, ge::DT_INT4})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("scale1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("scale2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6);
        this->Attr("output_mask").AttrType(OPTIONAL).ListBool({});
        this->Attr("dst_type").AttrType(OPTIONAL).Int(ge::DT_INT8);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.Input("x1")
            .ParamType(REQUIRED)
            .DataType(xDataType91095)
            .Format(format91095)
            .UnknownShapeFormat(format91095)
            .AutoContiguous();
        aicoreConfig.Input("x2")
            .ParamType(REQUIRED)
            .DataType(xDataType91095)
            .Format(format91095)
            .UnknownShapeFormat(format91095)
            .AutoContiguous();
        aicoreConfig.Input("gamma")
            .ParamType(REQUIRED)
            .DataType(xDataType91095)
            .Format(format91095)
            .UnknownShapeFormat(format91095)
            .AutoContiguous();
        aicoreConfig.Input("smooth_scale1")
            .ParamType(OPTIONAL)
            .DataType(xDataType91095)
            .Format(format91095)
            .UnknownShapeFormat(format91095)
            .AutoContiguous();
        aicoreConfig.Input("smooth_scale2")
            .ParamType(OPTIONAL)
            .DataType(xDataType91095)
            .Format(format91095)
            .UnknownShapeFormat(format91095)
            .AutoContiguous();
        aicoreConfig.Input("beta")
            .ParamType(OPTIONAL)
            .DataType(xDataType91095)
            .Format(format91095)
            .UnknownShapeFormat(format91095)
            .AutoContiguous();
        aicoreConfig.Output("y1")
            .ParamType(REQUIRED)
            .DataType(yDataType91095)
            .Format(format91095)
            .UnknownShapeFormat(format91095)
            .AutoContiguous();
        aicoreConfig.Output("y2")
            .ParamType(REQUIRED)
            .DataType(yDataType91095)
            .Format(format91095)
            .UnknownShapeFormat(format91095)
            .AutoContiguous();
        aicoreConfig.Output("x")
            .ParamType(REQUIRED)
            .DataType(xDataType91095)
            .Format(format91095)
            .UnknownShapeFormat(format91095)
            .AutoContiguous();
        aicoreConfig.Output("scale1")
            .ParamType(REQUIRED)
            .DataType(scalesOutDataType91095)
            .Format(format91095)
            .UnknownShapeFormat(format91095)
            .AutoContiguous();
        aicoreConfig.Output("scale2")
            .ParamType(REQUIRED)
            .DataType(scalesOutDataType91095)
            .Format(format91095)
            .UnknownShapeFormat(format91095)
            .AutoContiguous();
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "add_rms_norm_dynamic_quant_apt");
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend950", aicoreConfig);

        OpAICoreConfig config_kirin = GetKirinCoreConfig();
        this->AICore().AddConfig("kirinx90", config_kirin);
        this->AICore().AddConfig("kirin9030", config_kirin);
    }

private:
    OpAICoreConfig GetKirinCoreConfig() const
    {
        OpAICoreConfig config_kirin;
        config_kirin.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        config_kirin.Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Input("smooth_scale1")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Input("smooth_scale2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Input("beta")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Output("y1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Output("y2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Output("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Output("scale1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config_kirin.Output("scale2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        return config_kirin;
    }
};
OP_ADD(AddRmsNormDynamicQuant);
} // namespace ops