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
 * \file dequant_rope_quant_kvcache_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::DataType> XDtypeList = {
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
     ge::DT_INT32, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
     ge::DT_INT32}};

static const std::vector<ge::DataType> cosDtypeList = {
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
     ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16,
     ge::DT_BF16}};

static const std::vector<ge::DataType> biasDtypeList = {
    {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_FLOAT,
     ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_FLOAT}};

static const std::vector<ge::DataType> scaleDtypeList = {
    {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
     ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}};

static const std::vector<ge::DataType> cacheDtypeList = {
    {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
     ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}};

static const std::vector<ge::DataType> indicesDtypeList = {
    {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
     ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}};

static const std::vector<ge::Format> formatList = {
    {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
     ge::FORMAT_ND, ge::FORMAT_ND}};

static const std::vector<ge::DataType> XDtypeListKirin = {
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}};

static const std::vector<ge::DataType> cosDtypeListKirin = {
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}};

static const std::vector<ge::DataType> biasDtypeListKirin = {
    {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT}};

static const std::vector<ge::DataType> scaleDtypeListKirin = {
    {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}};

static const std::vector<ge::DataType> cacheDtypeListKirin = {
    {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}};

static const std::vector<ge::DataType> indicesDtypeListKirin = {
    {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}};

static const std::vector<ge::Format> formatListKirin = {
    {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}};

class DequantRopeQuantKvcache : public OpDef {
public:
    explicit DequantRopeQuantKvcache(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(XDtypeList)
            .Format(formatList)
            .UnknownShapeFormat(formatList)
            .AutoContiguous();
        this->Input("cos")
            .ParamType(REQUIRED)
            .DataType(cosDtypeList)
            .Format(formatList)
            .UnknownShapeFormat(formatList)
            .AutoContiguous();
        this->Input("sin")
            .ParamType(REQUIRED)
            .DataType(cosDtypeList)
            .Format(formatList)
            .UnknownShapeFormat(formatList)
            .AutoContiguous();
        this->Input("k_cache")
            .ParamType(REQUIRED)
            .DataType(cacheDtypeList)
            .Format(formatList)
            .UnknownShapeFormat(formatList)
            .AutoContiguous();
        this->Input("v_cache")
            .ParamType(REQUIRED)
            .DataType(cacheDtypeList)
            .Format(formatList)
            .UnknownShapeFormat(formatList)
            .AutoContiguous();
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType(indicesDtypeList)
            .Format(formatList)
            .UnknownShapeFormat(formatList)
            .AutoContiguous();
        this->Input("scale_k")
            .ParamType(REQUIRED)
            .DataType(scaleDtypeList)
            .Format(formatList)
            .UnknownShapeFormat(formatList)
            .AutoContiguous();
        this->Input("scale_v")
            .ParamType(REQUIRED)
            .DataType(scaleDtypeList)
            .Format(formatList)
            .UnknownShapeFormat(formatList)
            .AutoContiguous();
        this->Input("offset_k")
            .ParamType(OPTIONAL)
            .DataType(scaleDtypeList)
            .Format(formatList)
            .UnknownShapeFormat(formatList)
            .AutoContiguous();
        this->Input("offset_v")
            .ParamType(OPTIONAL)
            .DataType(scaleDtypeList)
            .Format(formatList)
            .UnknownShapeFormat(formatList)
            .AutoContiguous();
        this->Input("weight_scale")
            .ParamType(OPTIONAL)
            .DataType(scaleDtypeList)
            .Format(formatList)
            .UnknownShapeFormat(formatList)
            .AutoContiguous();
        this->Input("activation_scale")
            .ParamType(OPTIONAL)
            .DataType(scaleDtypeList)
            .Format(formatList)
            .UnknownShapeFormat(formatList)
            .AutoContiguous();
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType(biasDtypeList)
            .Format(formatList)
            .UnknownShapeFormat(formatList)
            .AutoContiguous();
        this->Output("q").ParamType(REQUIRED).DataType(cosDtypeList).Format(formatList).UnknownShapeFormat(formatList);
        this->Output("k").ParamType(REQUIRED).DataType(cosDtypeList).Format(formatList).UnknownShapeFormat(formatList);
        this->Output("v").ParamType(REQUIRED).DataType(cosDtypeList).Format(formatList).UnknownShapeFormat(formatList);
        this->Output("k_cache")
            .ParamType(REQUIRED)
            .DataType(cacheDtypeList)
            .Format(formatList)
            .UnknownShapeFormat(formatList);
        this->Output("v_cache")
            .ParamType(REQUIRED)
            .DataType(cacheDtypeList)
            .Format(formatList)
            .UnknownShapeFormat(formatList);
        this->Attr("size_splits").AttrType(REQUIRED).ListInt();
        this->Attr("quant_mode").AttrType(OPTIONAL).String("static");
        this->Attr("layout").AttrType(OPTIONAL).String("BSND");
        this->Attr("kv_output").AttrType(OPTIONAL).Bool(false);
        this->Attr("cache_mode").AttrType(OPTIONAL).String("contiguous");
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

    }
};

OP_ADD(DequantRopeQuantKvcache);
} // namespace ops
