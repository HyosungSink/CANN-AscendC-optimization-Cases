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
 * \file weight_quant_batch_matmul_v2_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class WeightQuantBatchMatmulV2 : public OpDef
{
public:
    WeightQuantBatchMatmulV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8,  ge::DT_INT8,  ge::DT_INT8,  ge::DT_INT8,  ge::DT_INT4,  ge::DT_INT4,  ge::DT_INT4,
                       ge::DT_INT4,  ge::DT_INT8,  ge::DT_INT8,  ge::DT_INT8,  ge::DT_INT8,  ge::DT_INT4,  ge::DT_INT4,
                       ge::DT_INT4,  ge::DT_INT4,  ge::DT_INT8,  ge::DT_INT8,  ge::DT_INT8,  ge::DT_INT8,  ge::DT_INT4,
                       ge::DT_INT4,  ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .DataTypeForBinQuery({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT4, ge::DT_INT4,
                                  ge::DT_INT4, ge::DT_INT4, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                                  ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT8, ge::DT_INT8,
                                  ge::DT_INT8, ge::DT_INT8, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4,
                                  ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4,
                                  ge::DT_INT4, ge::DT_INT4})
            .Format({ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat(
                {ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ});
        this->Input("antiquant_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_UINT64,  ge::DT_INT64,   ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("antiquant_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_INT32,   ge::DT_INT32,   ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("quant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT,  ge::DT_UINT64, ge::DT_FLOAT,  ge::DT_UINT64, ge::DT_FLOAT,  ge::DT_UINT64,
                       ge::DT_FLOAT,  ge::DT_UINT64, ge::DT_FLOAT,  ge::DT_UINT64, ge::DT_FLOAT,  ge::DT_UINT64,
                       ge::DT_FLOAT,  ge::DT_UINT64, ge::DT_FLOAT,  ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_FLOAT,  ge::DT_UINT64,
                       ge::DT_FLOAT,  ge::DT_UINT64, ge::DT_FLOAT,  ge::DT_UINT64, ge::DT_FLOAT,  ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("quant_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8, ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_INT8,    ge::DT_INT8, ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("transpose_x").AttrType(OPTIONAL).Bool(false);
        this->Attr("transpose_weight").AttrType(OPTIONAL).Bool(false);
        this->Attr("antiquant_group_size").AttrType(OPTIONAL).Int(0);
        this->Attr("dtype").AttrType(OPTIONAL).Int(-1);
        this->Attr("inner_precise").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");

        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
        aicore_config.Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config.Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ});
        aicore_config.Input("antiquant_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config.Input("antiquant_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config.Input("quant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_UINT64, ge::DT_UINT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config.Input("quant_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config.Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore().AddConfig("ascend310p", aicore_config);

        OpAICoreConfig aicore_config_950;
        aicore_config_950.Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config_950.Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT4, ge::DT_INT4,
                       ge::DT_INT4, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT4,
                       ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
                       ge::DT_HIFLOAT8, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                       ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1,
                       ge::DT_FLOAT4_E2M1, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_INT8,  ge::DT_INT8})
            .DataTypeForBinQuery({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT4, ge::DT_INT4,
                                  ge::DT_INT4, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT4,
                                  ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4,
                                  ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
                                  ge::DT_HIFLOAT8, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4,
                                  ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4,
                                  ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1,
                                  ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1,
                                  ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1,
                                  ge::DT_INT8,        ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_2, ge::FORMAT_FRACTAL_NZ_C0_2, ge::FORMAT_FRACTAL_NZ_C0_2,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ_C0_2, ge::FORMAT_FRACTAL_NZ_C0_2, ge::FORMAT_FRACTAL_NZ_C0_2, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_2, ge::FORMAT_FRACTAL_NZ_C0_2, ge::FORMAT_FRACTAL_NZ_C0_2,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_2, ge::FORMAT_FRACTAL_NZ_C0_2, ge::FORMAT_FRACTAL_NZ_C0_2,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ_C0_2, ge::FORMAT_FRACTAL_NZ_C0_2, ge::FORMAT_FRACTAL_NZ_C0_2, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_2, ge::FORMAT_FRACTAL_NZ_C0_2, ge::FORMAT_FRACTAL_NZ_C0_2,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ});
        aicore_config_950.Input("antiquant_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT16,
                       ge::DT_BF16, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config_950.Input("antiquant_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config_950.Input("quant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config_950.Input("quant_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config_950.Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config_950.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                       ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8,
                       ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config_950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("opFile.value", "weight_quant_batch_matmul_v2_apt");
        this->AICore().AddConfig("ascend950", aicore_config_950);
    }
};

OP_ADD(WeightQuantBatchMatmulV2);
} // namespace ops
