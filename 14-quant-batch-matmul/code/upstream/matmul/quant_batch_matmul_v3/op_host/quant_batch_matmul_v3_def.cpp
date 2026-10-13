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
 * \file quant_batch_matmul_v3.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class QuantBatchMatmulV3 : public OpDef {
public:
    explicit QuantBatchMatmulV3(const char *name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT4, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT4, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4})
            .Format({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT4, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT4, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4})
            .Format({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT64, ge::DT_UINT64, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_UINT64, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_UINT64, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_UINT64, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT64, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_BF16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_BF16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("pertoken_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT32, ge::DT_BF16, ge::DT_INT32, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT32, ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("dtype").AttrType(REQUIRED).Int();
        this->Attr("transpose_x1").AttrType(OPTIONAL).Bool(false);
        this->Attr("transpose_x2").AttrType(OPTIONAL).Bool(false);
        this->Attr("group_size").AttrType(OPTIONAL).Int(0);

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

        OpAICoreConfig config_310p;
        config_310p.Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ})
            .IgnoreContiguous();
        config_310p.Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .IgnoreContiguous();
        config_310p.Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT64, ge::DT_UINT64, ge::DT_INT64, ge::DT_INT64, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config_310p.Input("offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config_310p.Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config_310p.Input("pertoken_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config_310p.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ});
        config_310p.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");

        this->AICore().AddConfig("ascend310p", config_310p);

        OpAICoreConfig config950;
        config950.Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,
                       ge::DT_INT8,          ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1,
                       ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E5M2,   ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_INT8,          ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_INT8,          ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, 
                       ge::DT_FLOAT8_E5M2,   ge::DT_HIFLOAT8,
                       ge::DT_INT8,          ge::DT_HIFLOAT8,      
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_HIFLOAT8,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_INT8,          ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E5M2,   
                       ge::DT_INT8,          ge::DT_INT8,          ge::DT_HIFLOAT8,      ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
                       ge::DT_INT8,          ge::DT_INT8,          ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT4_E2M1,   ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_INT8,          ge::DT_INT8,          ge::DT_FLOAT8_E5M2,   ge::DT_INT8,          ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E5M2,
                       ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_INT8,
                       ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E5M2,   ge::DT_INT8,          ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_INT8,          ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT4_E2M1,   ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,
                       ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,        
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                     ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, 
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                                 ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, 
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();

        config950.Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,
                       ge::DT_INT8,          ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E5M2,   ge::DT_HIFLOAT8,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1,
                       ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E5M2,   ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN,
                       ge::DT_INT8,          ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, 
                       ge::DT_FLOAT8_E5M2,   ge::DT_HIFLOAT8,
                       ge::DT_INT8,          ge::DT_HIFLOAT8,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN,
                       ge::DT_INT8,          ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_INT8,
                       ge::DT_FLOAT8_E5M2,   ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN,
                       ge::DT_INT8,          ge::DT_INT8,          ge::DT_HIFLOAT8,      ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
                       ge::DT_INT8,          ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT4_E2M1,   ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_INT8,          ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_FLOAT8_E5M2,   ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E5M2,
                       ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,
                       ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E5M2,   ge::DT_INT8,          ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT4_E2M1,   ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_INT8,          ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E5M2,   ge::DT_INT8,
                       ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, 
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, 
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND, 
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, 
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, 
                     ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, 
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, 
                     ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         
                     ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND, 
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, 
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, 
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND, 
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, 
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, 
                     ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, 
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND, 
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND, 
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, 
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, 
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, 
                     ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, 
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, 
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, 
                     ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, 
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND, 
                     ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, 
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, 
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, 
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND, 
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, 
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, 
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, 
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, 
                                 ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         
                                 ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND, 
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, 
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, 
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND, 
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, 
                                 ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, 
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, 
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .IgnoreContiguous();

        config950.Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_BF16,        ge::DT_FLOAT,       ge::DT_BF16,   
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0,
                       ge::DT_INT64,       ge::DT_INT64,       ge::DT_INT64,       ge::DT_FLOAT8_E8M0, ge::DT_UINT64, 
                       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,
                       ge::DT_INT64,       ge::DT_FLOAT8_E8M0, ge::DT_BF16,        ge::DT_UINT64,
                       ge::DT_FLOAT8_E8M0, ge::DT_UINT64,      ge::DT_UINT64,
                       ge::DT_INT64,       ge::DT_FLOAT8_E8M0, 
                       ge::DT_FLOAT,       ge::DT_INT64,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, 
                       ge::DT_FLOAT,       ge::DT_BF16,        ge::DT_FLOAT,       ge::DT_INT64,       ge::DT_UINT64, 
                       ge::DT_INT64,       ge::DT_INT64,       ge::DT_FLOAT,       ge::DT_UINT64, 
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_INT64,       
                       ge::DT_FLOAT8_E8M0, ge::DT_INT64, 
                       ge::DT_UINT64,      ge::DT_FLOAT,       
                       ge::DT_FLOAT8_E8M0, ge::DT_UINT64,      ge::DT_INT64,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, 
                       ge::DT_FLOAT,       ge::DT_UINT64,      ge::DT_FLOAT8_E8M0, ge::DT_INT64,       ge::DT_INT64, 
                       ge::DT_FLOAT8_E8M0, ge::DT_INT64,       ge::DT_FLOAT8_E8M0, ge::DT_INT64,       ge::DT_UINT64, 
                       ge::DT_UINT64,      ge::DT_INT64,       ge::DT_INT64,       ge::DT_FLOAT8_E8M0, ge::DT_UINT64, 
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_INT64, 
                       ge::DT_INT64,       ge::DT_INT64,       ge::DT_INT64,       
                       ge::DT_INT64,       ge::DT_BF16,        ge::DT_UINT64,      ge::DT_FLOAT8_E8M0, ge::DT_UINT64, 
                       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_UINT64,      ge::DT_UINT64, 
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_INT64, 
                       ge::DT_UINT64,      ge::DT_FLOAT,       ge::DT_FLOAT,       
                       ge::DT_UINT64,      ge::DT_FLOAT8_E8M0, ge::DT_BF16,        ge::DT_FLOAT,       ge::DT_UINT64, 
                       ge::DT_UINT64,      ge::DT_BF16,        ge::DT_UINT64,      ge::DT_INT64,       ge::DT_FLOAT8_E8M0, 
                       ge::DT_UINT64,      ge::DT_INT64,       ge::DT_UINT64,      ge::DT_INT64,       ge::DT_INT64, 
                       ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_INT64, 
                       ge::DT_UINT64,      ge::DT_INT64,       ge::DT_FLOAT8_E8M0, ge::DT_UINT64, 
                       ge::DT_UINT64,      ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_UINT64,      ge::DT_FLOAT, 
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_UINT64, 
                       ge::DT_FLOAT8_E8M0, ge::DT_BF16,        ge::DT_INT64,       ge::DT_FLOAT,       ge::DT_UINT64, 
                       ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT, 
                       ge::DT_INT64,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, 
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_INT64,
                       ge::DT_FLOAT8_E8M0, ge::DT_INT64,       ge::DT_UINT64,      ge::DT_FLOAT,       ge::DT_INT64, 
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT, 
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_UINT64,      ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, 
                       ge::DT_FLOAT8_E8M0, ge::DT_UINT64,      ge::DT_INT64,       ge::DT_FLOAT, 
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         
                     ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         
                     ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         
                                 ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         
                                 ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                                 ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND, ge::FORMAT_ND});

        config950.Input("offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 
                       ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, 
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                     ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, 
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                                 ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, 
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        config950.Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                       ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32,
                       ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, 
                       ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_INT32, ge::DT_FLOAT, 
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_INT32, ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_BF16,  ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_BF16,  ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                     ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, 
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                                 ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, 
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        config950.Input("pertoken_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT, 
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                     ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, 
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                                 ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, 
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        config950.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32,         ge::DT_INT32,         ge::DT_INT32,         ge::DT_INT32,         ge::DT_INT32,
                       ge::DT_INT32,         ge::DT_FLOAT,         ge::DT_FLOAT,
                       ge::DT_FLOAT16,       ge::DT_FLOAT16,       ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_FLOAT16,
                       ge::DT_FLOAT16,       ge::DT_FLOAT16,       ge::DT_BF16,
                       ge::DT_FLOAT16,       ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_FLOAT,
                       ge::DT_FLOAT16,       ge::DT_FLOAT16,       ge::DT_BF16,
                       ge::DT_BF16,          ge::DT_BF16,
                       ge::DT_FLOAT16,       ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_BF16,
                       ge::DT_FLOAT,         ge::DT_BF16,          ge::DT_BF16,          ge::DT_FLOAT,         ge::DT_FLOAT16,
                       ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_BF16,          ge::DT_BF16,
                       ge::DT_FLOAT16,       ge::DT_FLOAT,         ge::DT_FLOAT16,       
                       ge::DT_BF16,          ge::DT_FLOAT16,
                       ge::DT_INT8,          ge::DT_FLOAT,         
                       ge::DT_BF16,          ge::DT_FLOAT16,       ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_FLOAT,
                       ge::DT_FLOAT16,       ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT16,
                       ge::DT_BF16,          ge::DT_FLOAT,         ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_BF16,
                       ge::DT_FLOAT16,       ge::DT_FLOAT,         ge::DT_BF16,          ge::DT_BF16,          ge::DT_BF16,
                       ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT16,
                       ge::DT_BF16,          ge::DT_FLOAT,         ge::DT_FLOAT16,
                       ge::DT_BF16,          ge::DT_BF16,          ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_FLOAT16,
                       ge::DT_FLOAT,         ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_FLOAT,
                       ge::DT_BF16,          ge::DT_BF16,          ge::DT_BF16,          ge::DT_FLOAT,
                       ge::DT_FLOAT16,       ge::DT_FLOAT,         ge::DT_FLOAT16,
                       ge::DT_BF16,          ge::DT_FLOAT,         ge::DT_BF16,          ge::DT_BF16,          ge::DT_FLOAT,
                       ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_BF16,          ge::DT_INT8,          ge::DT_FLOAT16,
                       ge::DT_FLOAT16,       ge::DT_INT8,          ge::DT_BF16,          ge::DT_BF16,          ge::DT_FLOAT,
                       ge::DT_INT8,          ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_BF16,          ge::DT_BF16,
                       ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_BF16,          ge::DT_BF16,
                       ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT16,       ge::DT_FLOAT,         ge::DT_BF16,
                       ge::DT_FLOAT16,       ge::DT_FLOAT16,       ge::DT_FLOAT16,       ge::DT_BF16,
                       ge::DT_FLOAT,         ge::DT_BF16,          ge::DT_BF16,          ge::DT_FLOAT,         ge::DT_FLOAT,
                       ge::DT_FLOAT,         ge::DT_BF16,          ge::DT_FLOAT,         ge::DT_BF16,          ge::DT_FLOAT16,
                       ge::DT_BF16,          ge::DT_BF16,          ge::DT_FLOAT,
                       ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_FLOAT,
                       ge::DT_FLOAT,         ge::DT_BF16,          ge::DT_BF16,          ge::DT_FLOAT16,       ge::DT_FLOAT,
                       ge::DT_BF16,          ge::DT_FLOAT16,       ge::DT_FLOAT16,
                       ge::DT_BF16,          ge::DT_BF16,          ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT16,
                       ge::DT_FLOAT16,       ge::DT_FLOAT16,       ge::DT_FLOAT,         ge::DT_FLOAT16,
                       ge::DT_FLOAT16,       ge::DT_FLOAT16,       ge::DT_FLOAT16,       ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                     ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, 
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                                 ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, 
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        config950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("opFile.value","quant_batch_matmul_v3_apt");
        this->AICore().AddConfig("ascend950", config950);

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
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        config_kirin.Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .IgnoreContiguous();
        config_kirin.Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT64, ge::DT_UINT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config_kirin.Input("offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config_kirin.Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config_kirin.Input("pertoken_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config_kirin.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        return config_kirin;
    }
};

OP_ADD(QuantBatchMatmulV3);
}  // namespace ops
