/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file grouped_matmul_finalize_routing.cc
 * \brief
 */
#include <map>
#include <string>
#include <sstream>
#include <initializer_list>

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "../common/include/err/ops_err.h"
#include "util/math_util.h"


namespace {
const size_t xIndex = 0;
const size_t wIndex = 1;
const size_t scaleOptionIndex = 2;
const size_t biasOptionIndex = 3;
const size_t pertokenScaleOptionIndex = 4;
const size_t groupListOptionIndex = 5;
const size_t sharedInputOptionIndex = 6;
const size_t logitOptionIndex = 7;
const size_t rowIndexOptionIndex = 8;
const size_t offsetOptionIndex = 9;
const size_t weightTransIndex= 4;
const size_t oneDimNum = 1;
const size_t twoDimNum = 2;
const size_t threeDimNum = 3;
const size_t fourDimNum = 4;
const size_t sharedInputOffsetAttrIndex = 2;
const size_t outputBSAttrIndex = 5;
const int64_t DYNAMIC_DIM = -1;
const int64_t NZ_K0_VALUE_INT8 = 16;
const int64_t NZ_K0_VALUE_INT8_TRANS = 32;
const int64_t N_VALUE_256 = 256;
const int64_t N_VALUE_64 = 64;
const int64_t K_VALUE_128 = 128;
const int64_t DIM_ZERO = 0;
const int64_t DIM_ONE = 1;
const int64_t DIM_TWO = 2;
const int64_t DIM_THREE = 3;
const int64_t GMMFR_SPLIT_SIZE = 64;
const int64_t GMMFR_QUANT_SCALE_PARAM_COUNT = 2;
const int ND_N_VALUE_ALIGN = 8;
const int ND_K0_VALUE_INT8 = 64;
}

using namespace gert;
namespace ops {

struct ConstraintShape {
    uint32_t k;
    uint32_t n;
};

static const std::initializer_list<ConstraintShape> W4A8_K_N_SUPPORT_LIST = {{2048, 7168}};
static const std::initializer_list<ge::DataType> MX_IN_TYPE_SUPPORT_LIST = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                                                                            ge::DT_FLOAT4_E2M1};
static const std::initializer_list<ge::DataType> MXFP4_IN_TYPE_SUPPORT_LIST = {ge::DT_FLOAT4_E2M1};
static const std::initializer_list<ge::DataType> MXFP8_IN_TYPE_SUPPORT_LIST = {ge::DT_FLOAT8_E4M3FN,
                                                                               ge::DT_FLOAT8_E5M2};

template <typename T>
std::string Shape2String(const T& shape) {
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

template <typename T>
bool CheckType(const T &value, const std::initializer_list<T> list)
{
    for (const auto &item : list) {
        if (item == value) {
            return true;
        }
    }
    return false;
}

struct CheckXandWParams {
    const gert::Shape *shape_x1 = nullptr;
    const gert::Shape *shape_x2 = nullptr;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    int64_t e = 0;
    bool weightTrans = false;
};

static ge::graphStatus ValidateXAndWShapes(const char* op_name, CheckXandWParams& params)
{
    OP_CHECK_IF(params.shape_x1->GetDimNum() != twoDimNum, OPS_REPORT_CUBE_INNER_ERR(op_name, "X dim is not 2."), return ge::GRAPH_FAILED);
    params.m = params.shape_x1->GetDim(xIndex);
    params.k = params.shape_x1->GetDim(wIndex);
    OP_CHECK_IF(params.shape_x2->GetDimNum() != threeDimNum, OPS_REPORT_CUBE_INNER_ERR(op_name, "W dim is not 3."),
        return ge::GRAPH_FAILED);
    if (!params.weightTrans) {
        params.n = params.shape_x2->GetDim(twoDimNum);
    }
    else{
        params.n = params.shape_x2->GetDim(DIM_ONE);
    }
    params.e = params.shape_x2->GetDim(xIndex);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetXAndWShapesForMX(const InferShapeContext *context,const char* op_name, CheckXandWParams& params)
{
    params.m = params.shape_x1->GetDim(xIndex);
    params.k = params.shape_x1->GetDim(wIndex);

    auto shape_scale = context->GetOptionalInputShape(scaleOptionIndex);

    params.n = params.weightTrans ? shape_scale->GetDim(DIM_ONE) : shape_scale->GetDim(DIM_TWO);
    params.e = params.shape_x2->GetDim(xIndex);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateScaleAndBias(const InferShapeContext *context, const char* op_name, const CheckXandWParams& xAndWParams)
{
    auto shape_scale = context->GetOptionalInputShape(scaleOptionIndex);
    OP_CHECK_IF(shape_scale == nullptr, OPS_REPORT_CUBE_INNER_ERR(op_name, "scale is not given."), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "shape_scale: %s", Shape2String(*shape_scale).c_str());

    if (shape_scale->GetDimNum() == twoDimNum) {
        OP_CHECK_IF(shape_scale->GetDim(0) != xAndWParams.e || shape_scale->GetDim(1) != xAndWParams.n,
            OPS_REPORT_CUBE_INNER_ERR(op_name, "scale 's size is not (E,N)."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(!(((xAndWParams.k % NZ_K0_VALUE_INT8) == 0) &&
                ((xAndWParams.n % NZ_K0_VALUE_INT8_TRANS) == 0) && (xAndWParams.n >= N_VALUE_256)),
                OPS_REPORT_CUBE_INNER_ERR(op_name, "The input shape (K,N) is not supported"),
                return ge::GRAPH_FAILED);
    } else if (shape_scale->GetDimNum() == threeDimNum) {
        OP_CHECK_IF(shape_scale->GetDim(0) != xAndWParams.e || shape_scale->GetDim(2) != xAndWParams.n || shape_scale->GetDim(1) != 1,
            OPS_REPORT_CUBE_INNER_ERR(op_name, "scale 's size is not (E,1,N)."), return ge::GRAPH_FAILED);
        if (context->GetOptionalInputShape(biasOptionIndex) != nullptr) {
            OP_CHECK_IF((context->GetOptionalInputShape(biasOptionIndex)->GetDim(0) != xAndWParams.e ||
                         context->GetOptionalInputShape(biasOptionIndex)->GetDim(DIM_ONE) != xAndWParams.n),
                        OPS_REPORT_CUBE_INNER_ERR(op_name, "bias is not supported."), return ge::GRAPH_FAILED);
        }
    } else {
        OP_LOGE(op_name, "scale shape is not support");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidatePertokenAndGroupList(const InferShapeContext *context, const char *op_name,
                                                    const CheckXandWParams &xAndWParams)
{
    if (context->GetOptionalInputShape(pertokenScaleOptionIndex) != nullptr) {
        OP_CHECK_IF(context->GetOptionalInputShape(pertokenScaleOptionIndex)->GetDimNum() != oneDimNum ||
            context->GetOptionalInputShape(pertokenScaleOptionIndex)->GetDim(0) != xAndWParams.m,
            OPS_REPORT_CUBE_INNER_ERR(op_name, "pertoken_scale's size is not (M,)."), return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(context->GetOptionalInputShape(groupListOptionIndex) == nullptr,
        OPS_REPORT_CUBE_INNER_ERR(op_name, "group_list is not given."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetOptionalInputShape(groupListOptionIndex)->GetDimNum() != oneDimNum ||
        context->GetOptionalInputShape(groupListOptionIndex)->GetDim(0) != xAndWParams.e,
        OPS_REPORT_CUBE_INNER_ERR(op_name, "group_list's size is not (e,)."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateSharedInputAndLogit(const InferShapeContext *context, int& bsdp, const char* op_name, const CheckXandWParams& xAndWParams)
{
    if (context->GetOptionalInputShape(sharedInputOptionIndex) != nullptr) {
        OP_CHECK_IF(context->GetOptionalInputShape(sharedInputOptionIndex)->GetDimNum() != twoDimNum ||
            context->GetOptionalInputShape(sharedInputOptionIndex)->GetDim(1) != xAndWParams.n,
            OPS_REPORT_CUBE_INNER_ERR(op_name, "shared_input's shape is wrong."), return ge::GRAPH_FAILED);
        bsdp = context->GetOptionalInputShape(sharedInputOptionIndex)->GetDim(0);
        OP_CHECK_IF(bsdp <= 0, OPS_REPORT_CUBE_INNER_ERR(op_name, "shared_input first dim must bigger than 0 ."),
            return ge::GRAPH_FAILED);
    }
    
    if (context->GetOptionalInputShape(logitOptionIndex) != nullptr) {
        OP_CHECK_IF(context->GetOptionalInputShape(logitOptionIndex)->GetDimNum() != oneDimNum ||
            context->GetOptionalInputShape(logitOptionIndex)->GetDim(0) != xAndWParams.m,
            OPS_REPORT_CUBE_INNER_ERR(op_name, "logit's shape is wrong."), return ge::GRAPH_FAILED);
    }
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateRowIndex(const InferShapeContext *context, const char* op_name, const CheckXandWParams& xAndWParams)
{
    OP_CHECK_IF(context->GetOptionalInputShape(rowIndexOptionIndex) == nullptr,
        OPS_REPORT_CUBE_INNER_ERR(op_name, "row_index is not given."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetOptionalInputShape(rowIndexOptionIndex)->GetDimNum() != oneDimNum ||
        context->GetOptionalInputShape(rowIndexOptionIndex)->GetDim(0) != xAndWParams.m,
        OPS_REPORT_CUBE_INNER_ERR(op_name, "row_index's shape is wrong."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetupOutputAndCheckAttrs(InferShapeContext *context, const int &bsdp, const char *op_name,
                                                CheckXandWParams &xAndWParams)
{
    auto attrs = context->GetAttrs();
    auto shape_out = context->GetOutputShape(0);
    shape_out->SetDimNum(twoDimNum);
    const int *output_bs = attrs->GetAttrPointer<int>(outputBSAttrIndex);
    OP_CHECK_IF(output_bs == nullptr,
        OPS_REPORT_CUBE_INNER_ERR(op_name, "output_bs is not given."), return ge::GRAPH_FAILED);
    if (output_bs != nullptr) {
        shape_out->SetDim(0, *output_bs);
    }
    shape_out->SetDim(DIM_ONE, xAndWParams.n);
    OP_LOGI(op_name, "shape out is %ld, %ld", shape_out->GetDim(0), shape_out->GetDim(1));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetupOutputForMX(InferShapeContext *context, const int& bsdp, const char* op_name, CheckXandWParams& xAndWParams)
{
    auto attrs = context->GetAttrs();
    auto shape_out = context->GetOutputShape(0);
    
    shape_out->SetDimNum(twoDimNum);
    const int *output_bs = attrs->GetAttrPointer<int>(outputBSAttrIndex);
    
    if (output_bs != nullptr) {
        shape_out->SetDim(0, *output_bs);
    }
    auto x2_dim = xAndWParams.shape_x2->GetDimNum();
    auto shape_scale = context->GetOptionalInputShape(scaleOptionIndex);

    shape_out->SetDim(DIM_ONE, xAndWParams.weightTrans ?
                             shape_scale->GetDim(DIM_ONE) :
                             shape_scale->GetDim(DIM_TWO)); // 如果非转置，n为最后一维，如果转置，n为倒数第二维。
    OP_LOGI(op_name, "shape out is %ld, %ld", shape_out->GetDim(DIM_ZERO), shape_out->GetDim(DIM_ONE));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateOffsetShape(const InferShapeContext *context, const char* op_name, const CheckXandWParams& xAndWParams)
{
    auto shape_offset = context->GetOptionalInputShape(offsetOptionIndex);
    if (shape_offset != nullptr) {
        if (shape_offset->GetDimNum() != threeDimNum) {
            OP_LOGE(op_name, "offset shape is not support");
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(shape_offset->GetDim(0) != xAndWParams.e || shape_offset->GetDim(1) != 1 || shape_offset->GetDim(2) != xAndWParams.n,
        OPS_REPORT_CUBE_INNER_ERR(op_name, "offset 's size is not (E,1,N)."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeGroupedMatmulFinalizeRouting(InferShapeContext *context)
{
    auto op_name = context->GetNodeName();
    auto shape_x1 = context->GetInputShape(xIndex);
    auto shape_x2 = context->GetInputShape(wIndex);
    auto shape_out = context->GetOutputShape(0);
    int bsdp = -1;

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(shape_x1 == nullptr || shape_x2 == nullptr || shape_out == nullptr || attrs == nullptr,
        OPS_REPORT_CUBE_INNER_ERR(op_name, "shape or attrs is null"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "x1_shape: %s, x2_shape: %s", Shape2String(*shape_x1).c_str(), Shape2String(*shape_x2).c_str());
    
    const bool *transposeWeightPtr = attrs->GetBool(weightTransIndex);
    bool transposeWeight = (transposeWeightPtr != nullptr ? *transposeWeightPtr : false);
    
    CheckXandWParams xAndWParams{shape_x1, shape_x2, 0, 0, 0, 0, transposeWeight};
    
    // MX量化模式涉及图模式交付不走校验逻辑
    auto shape_scale = context->GetOptionalInputShape(scaleOptionIndex);
    OP_CHECK_IF(shape_scale == nullptr, OPS_REPORT_CUBE_INNER_ERR(op_name, "scale is not given."), return ge::GRAPH_FAILED);
    if (shape_scale->GetDimNum() == fourDimNum) {
        OP_CHECK_IF(SetXAndWShapesForMX(context, op_name, xAndWParams) != ge::GRAPH_SUCCESS, return ge::GRAPH_FAILED, );
        OP_CHECK_IF(SetupOutputForMX(context, bsdp, op_name, xAndWParams) != ge::GRAPH_SUCCESS,
                    return ge::GRAPH_FAILED, );
        return ge::GRAPH_SUCCESS;
    } else {
        OP_CHECK_IF(ValidateXAndWShapes(op_name, xAndWParams) != ge::GRAPH_SUCCESS, return ge::GRAPH_FAILED, );
        // 在动态图模式下，跳过校验逻辑
        if (xAndWParams.m != DYNAMIC_DIM && xAndWParams.n != DYNAMIC_DIM && xAndWParams.k != DYNAMIC_DIM &&
            xAndWParams.e != DYNAMIC_DIM) {
            OP_CHECK_IF(ValidateScaleAndBias(context, op_name, xAndWParams) != ge::GRAPH_SUCCESS,
                        return ge::GRAPH_FAILED, );
            OP_CHECK_IF(ValidatePertokenAndGroupList(context, op_name, xAndWParams) != ge::GRAPH_SUCCESS,
                        return ge::GRAPH_FAILED, );
            OP_CHECK_IF(ValidateSharedInputAndLogit(context, bsdp, op_name, xAndWParams) != ge::GRAPH_SUCCESS,
                        return ge::GRAPH_FAILED, );
            OP_CHECK_IF(ValidateRowIndex(context, op_name, xAndWParams) != ge::GRAPH_SUCCESS,
                        return ge::GRAPH_FAILED, );
            OP_CHECK_IF(ValidateOffsetShape(context, op_name, xAndWParams) != ge::GRAPH_SUCCESS,
                        return ge::GRAPH_FAILED, );
        }
        OP_CHECK_IF(SetupOutputAndCheckAttrs(context, bsdp, op_name, xAndWParams) != ge::GRAPH_SUCCESS,
                    return ge::GRAPH_FAILED, );
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateFailedDataType(const gert::InferDataTypeContext *context)
{
    // 先判断a8w4还是a8w8出问题
    if ((context->GetInputDataType(xIndex) == ge::DT_INT8 ||
         context->GetInputDataType(xIndex) == ge::DT_FLOAT8_E4M3FN ||
         context->GetInputDataType(xIndex) == ge::DT_HIFLOAT8) &&
        (context->GetInputDataType(wIndex) == ge::DT_INT8 ||
         context->GetInputDataType(wIndex) == ge::DT_FLOAT8_E4M3FN ||
         context->GetInputDataType(wIndex) == ge::DT_HIFLOAT8)) {
        OP_CHECK_IF((context->GetOptionalInputDataType(scaleOptionIndex) != ge::DT_FLOAT &&
                     context->GetOptionalInputDataType(scaleOptionIndex) != ge::DT_BF16),
                     OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "The W8A8 InputDataType of scale is wrong."),
                     return ge::GRAPH_FAILED);
        OP_CHECK_IF((context->GetOptionalInputDataType(groupListOptionIndex) != ge::DT_INT64),
                     OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "The W8A8 InputDataType of groupList is wrong."),
                     return ge::GRAPH_FAILED);
        OP_CHECK_IF((context->GetOptionalInputDataType(rowIndexOptionIndex) != ge::DT_INT64 &&
                     context->GetOptionalInputDataType(rowIndexOptionIndex) != ge::DT_INT32),
                     OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "The W8A8 InputDataType of rowIndex is wrong."),
                     return ge::GRAPH_FAILED);
    }

    if (context->GetInputDataType(xIndex) == ge::DT_INT8 && context->GetInputDataType(wIndex) == ge::DT_INT4) {
        OP_CHECK_IF((context->GetOptionalInputDataType(scaleOptionIndex) != ge::DT_INT64),
                     OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "The W4A8 InputDataType of scale is wrong."),
                     return ge::GRAPH_FAILED);
        OP_CHECK_IF((context->GetOptionalInputDataType(biasOptionIndex) != ge::DT_FLOAT),
                     OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "The W4A8 InputDataType of bias is wrong."),
                     return ge::GRAPH_FAILED);
        OP_CHECK_IF((context->GetOptionalInputDataType(groupListOptionIndex) != ge::DT_INT64),
                     OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "The W4A8 InputDataType of groupList is wrong."),
                     return ge::GRAPH_FAILED);
        OP_CHECK_IF((context->GetOptionalInputDataType(rowIndexOptionIndex) != ge::DT_INT64),
                     OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "The W4A8 InputDataType of rowIndex is wrong."),
                     return ge::GRAPH_FAILED);                     
        OP_CHECK_IF((context->GetOptionalInputDataType(logitOptionIndex) != ge::DT_FLOAT),
                     OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "The W4A8 InputDataType of logit is wrong."),
                     return ge::GRAPH_FAILED);  
    }

    if (CheckType(context->GetInputDataType(xIndex), MX_IN_TYPE_SUPPORT_LIST) &&
        CheckType(context->GetInputDataType(wIndex), MX_IN_TYPE_SUPPORT_LIST)) {
        OP_CHECK_IF(
            (context->GetOptionalInputDataType(scaleOptionIndex) != ge::DT_FLOAT8_E8M0),
            OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(),
                                      "The MXFP4/MXFP8 InputDataType of scale is wrong.Supported type:FlOAT8_E8M0"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (context->GetOptionalInputDataType(groupListOptionIndex) != ge::DT_INT64),
            OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(),
                                      "The MXFP4/MXFP8 InputDataType of groupList is wrong.Supported type:INT64"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (context->GetOptionalInputDataType(rowIndexOptionIndex) != ge::DT_INT64),
            OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(),
                                      "The MXFP4/MXFP8 InputDataType of rowIndex is wrong.Supported type:INT64"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (context->GetOptionalInputDataType(logitOptionIndex) != ge::DT_FLOAT),
            OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(),
                                      "The MXFP4/MXFP8 InputDataType of logit is wrong.Supported type:FLOAT"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((context->GetOptionalInputDataType(pertokenScaleOptionIndex) != ge::DT_FLOAT8_E8M0),
                    OPS_REPORT_CUBE_INNER_ERR(
                        context->GetNodeName(),
                        "The MXFP4/MXFP8 InputDataType of pertokenscale is wrong.Supported type:FlOAT8_E8M0"),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(
        !((context->GetInputDataType(xIndex) == ge::DT_INT8 && context->GetInputDataType(wIndex) == ge::DT_INT8) ||
          (context->GetInputDataType(xIndex) == ge::DT_INT8 && context->GetInputDataType(wIndex) == ge::DT_INT4) ||
          (CheckType(context->GetInputDataType(xIndex), MXFP4_IN_TYPE_SUPPORT_LIST) &&
           CheckType(context->GetInputDataType(wIndex), MXFP4_IN_TYPE_SUPPORT_LIST)) ||
          (CheckType(context->GetInputDataType(xIndex), MXFP8_IN_TYPE_SUPPORT_LIST) &&
           CheckType(context->GetInputDataType(wIndex), MXFP8_IN_TYPE_SUPPORT_LIST))),
        OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(),
                                  "InputDataType is wrong, only support InputDataType of "
                                  "INT4,INT8,FLOAT8_E4M3FN,FLOAT8_E5M2,FLOAT4_E2M1"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_FAILED;
}

static bool IsSupportMX(gert::InferDataTypeContext *context)
{
    if (CheckType(context->GetInputDataType(xIndex), MX_IN_TYPE_SUPPORT_LIST) &&
        CheckType(context->GetInputDataType(wIndex), MX_IN_TYPE_SUPPORT_LIST) &&
        context->GetOptionalInputDataType(scaleOptionIndex) == ge::DT_FLOAT8_E8M0 &&
        context->GetOptionalInputDataType(groupListOptionIndex) == ge::DT_INT64 &&
        context->GetOptionalInputDataType(rowIndexOptionIndex) == ge::DT_INT64 &&
        context->GetOptionalInputDataType(logitOptionIndex) == ge::DT_FLOAT &&
        context->GetOptionalInputDataType(pertokenScaleOptionIndex) == ge::DT_FLOAT8_E8M0) {
        return true;
    }
    return false;
}

static ge::graphStatus InferDataTypeGroupedMatmulFinalizeRouting(gert::InferDataTypeContext *context)
{
    bool supportDataTypeMX = IsSupportMX(context);

    bool supportDataTypeW8A8 = (context->GetInputDataType(xIndex) == ge::DT_INT8 ||
                                context->GetInputDataType(xIndex) == ge::DT_FLOAT8_E4M3FN ||
                                context->GetInputDataType(xIndex) == ge::DT_HIFLOAT8) &&
                               (context->GetInputDataType(wIndex) == ge::DT_INT8 ||
                                context->GetInputDataType(wIndex) == ge::DT_FLOAT8_E4M3FN ||
                                context->GetInputDataType(wIndex) == ge::DT_HIFLOAT8) &&
                               (context->GetOptionalInputDataType(scaleOptionIndex) == ge::DT_FLOAT ||
                                context->GetOptionalInputDataType(scaleOptionIndex) == ge::DT_BF16) &&
                               context->GetOptionalInputDataType(groupListOptionIndex) == ge::DT_INT64 &&
                               (context->GetOptionalInputDataType(rowIndexOptionIndex) == ge::DT_INT64 ||
                                context->GetOptionalInputDataType(rowIndexOptionIndex) == ge::DT_INT32);

    bool supportDataTypeW4A8 = context->GetInputDataType(xIndex) == ge::DT_INT8 && 
                               context->GetInputDataType(wIndex) == ge::DT_INT4 &&
                               context->GetOptionalInputDataType(scaleOptionIndex) == ge::DT_INT64 &&
                               context->GetOptionalInputDataType(biasOptionIndex) == ge::DT_FLOAT &&
                               context->GetOptionalInputDataType(groupListOptionIndex) == ge::DT_INT64 &&
                               context->GetOptionalInputDataType(rowIndexOptionIndex) == ge::DT_INT64 &&
                               context->GetOptionalInputDataType(logitOptionIndex) == ge::DT_FLOAT;

    if (context->GetOptionalInputDataType(logitOptionIndex) != ge::DT_UNDEFINED) {
        supportDataTypeW8A8 = supportDataTypeW8A8 && context->GetOptionalInputDataType(logitOptionIndex) == ge::DT_FLOAT;
    }

    if (context->GetOptionalInputDataType(sharedInputOptionIndex) != ge::DT_UNDEFINED) {
        supportDataTypeW8A8 = supportDataTypeW8A8 && context->GetOptionalInputDataType(sharedInputOptionIndex) == ge::DT_BF16;
        // 移除 supportDataTypeW4A8 校验sharedInput类型后, 如果sharedInput存在需要在这里复查类型
        supportDataTypeW4A8 = supportDataTypeW4A8 && context->GetOptionalInputDataType(sharedInputOptionIndex) == ge::DT_BF16;
    }

    if (context->GetOptionalInputDataType(offsetOptionIndex) != ge::DT_UNDEFINED) {
        supportDataTypeW4A8 = supportDataTypeW4A8 && context->GetOptionalInputDataType(offsetOptionIndex) == ge::DT_FLOAT;
    }
    
    if (!(supportDataTypeW4A8 || supportDataTypeW8A8 || supportDataTypeMX )) {
        OP_CHECK_IF(ValidateFailedDataType(context) != ge::GRAPH_SUCCESS, return ge::GRAPH_FAILED,);
    }
    
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedMatmulFinalizeRouting)
    .InferShape(InferShapeGroupedMatmulFinalizeRouting)
    .InferDataType(InferDataTypeGroupedMatmulFinalizeRouting);
} // namespace ops
