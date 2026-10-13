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
 * \file rms_norm_grad_tiling.cpp
 * \brief RmsNormGrad Tiling file
 */
#include "rms_norm_grad_tiling.h"
#include "register/op_def_registry.h"
#include "../common/inc/op_host/tiling_util.h"
#include "tiling/tiling_api.h"

namespace optiling {
using namespace Ops::NN::OpTiling;

static const uint32_t ALIGN_32 = 8;
static const uint32_t ALIGN_16 = 16;
static const uint64_t DTYPE_FP32 = 1;
static const uint64_t DTYPE_FP16 = 2;
static const uint64_t DTYPE_BF16 = 3;
static const uint64_t MULTI_CORE = 2;
static const uint64_t UB_SPLIT_N = 1;
static const uint64_t UB_SPLIT_D = 2;
static const uint32_t BUFFER_SIZE_SPLIT_N_HIGH_PRECISION = 6080;
static const uint32_t BUFFER_SIZE_SPLIT_D_HIGH_PRECISION = 4096;
static const uint32_t BF16_SPLIT_N_BUFFER_SIZE = 5760;
static const uint32_t BF16_SPLIT_D_BUFFER_SIZE = 4096;
static const uint32_t COL_VAL_MULTIPLE_910 = 64;
static const uint32_t FIXED_OUTPUT_WORKSPACE_SIZE = 1152000; // 5760 * 50 * 4
static const uint32_t SMALLD_THRESHOLD = 640;
static const uint32_t USED_UB = 195584; // (192 - 1) * 1024
constexpr size_t MAX_DIM_NUM = 8;
constexpr size_t MIN_DIM_X = 1;
constexpr size_t MIN_DIM_GAMMA = 1;
constexpr uint8_t SCHEDULE_MODE = 1;

static uint32_t CalcSmallDBufferSize(uint32_t colValAlign)
{
    /*
    +----------------+----------+-------------+---------------+------------+-------------+
    |                | row      | col         | sizeof()      | buffer num | 备注        |
    +----------------+----------+-------------+---------------+------------+-------------+
    | inQueueDY      | ubFactor | colValAlign | sizeof(T)     | 2(1)       | BF16开DB    |
    +----------------+----------+-------------+---------------+------------+-------------+
    | inQueueX       | ubFactor | colValAlign | sizeof(T)     | 2(1)       | BF16开DB    |
    +----------------+----------+-------------+---------------+------------+-------------+
    | inQueueRstd    | ubFactor | 1           | sizeof(float) | 2(1)       | BF16开DB    |
    +----------------+----------+-------------+---------------+------------+-------------+
    | inQueueGamma   | 1        | colValAlign | sizeof(T)     | 1          |             |
    +----------------+----------+-------------+---------------+------------+-------------+
    | outQueueDX     | ubFactor | colValAlign | sizeof(T)     | 2(1)       | BF16开DB    |
    +----------------+----------+-------------+---------------+------------+-------------+
    | outQueueDgamma | 1        | colValAlign | sizeof(float) | 1          |             |
    +----------------+----------+-------------+---------------+------------+-------------+
    | ndBufFp32Buf   | ubFactor | colValAlign | sizeof(float) | 1          |             |
    +----------------+----------+-------------+---------------+------------+-------------+
    | ndBufFp32Buf   | ubFactor | colValAlign | sizeof(float) | 1          | B16场景申请 |
    +----------------+----------+-------------+---------------+------------+-------------+
    | ndBufFp32Buf   | ubFactor | colValAlign | sizeof(float) | 1          | B16场景申请 |
    +----------------+----------+-------------+---------------+------------+-------------+
    | nFp32Buf       | ubFactor | 1           | sizeof(float) | 1          |             |
    +----------------+----------+-------------+---------------+------------+-------------+
    | dFp32Buf       | 1        | colValAlign | sizeof(float) | 1          |             |
    +----------------+----------+-------------+---------------+------------+-------------+
    | tmpBuf         | ubFactor | 64          | sizeof(float) | 1          |             |
    +----------------+----------+-------------+---------------+------------+-------------+
    */
    uint32_t ubFactor = 1;
    uint32_t rowWeight = 268;
    uint32_t colWeight = 10;
    uint32_t rowColWeight = 24;

    ubFactor = (USED_UB - colValAlign * colWeight) / (colValAlign * rowColWeight + rowWeight);
    return ubFactor * colValAlign;
}

static void LargeNSmallD(
    gert::TilingContext* context, RmsNormGradTilingData& tiling, uint32_t buffer_size, uint32_t row_val,
    uint32_t col_val, uint32_t core_num)
{
    if (core_num == 0 || col_val == 0) {
        return;
    }
    // block split
    uint32_t block_factor = (row_val + core_num - 1) / core_num;
    uint32_t block_dim = (row_val + block_factor - 1) / block_factor;
    uint32_t core_calc_num = block_factor;
    uint32_t core_calc_tail = row_val % block_factor != 0 ? row_val - (block_dim - 1) * block_factor : 0;
    tiling.set_block_factor(block_factor);
    tiling.set_core_calc_num(core_calc_num);
    tiling.set_core_calc_tail(core_calc_tail);
    context->SetBlockDim(block_dim);
    tiling.set_block_dim(block_dim);
    // ub split, assume col_val <= buffer_size
    uint32_t ub_factor = buffer_size / col_val < block_factor ? buffer_size / col_val : block_factor;
    uint32_t ub_loop = (block_factor + ub_factor - 1) / ub_factor;
    uint32_t ub_calc_num = ub_factor;
    uint32_t ub_calc_tail = block_factor % ub_factor != 0 ? block_factor - (ub_loop - 1) * ub_factor : 0;
    tiling.set_ub_split_dim(0);
    tiling.set_ub_factor(ub_factor);
    tiling.set_ub_calc_num(ub_calc_num);
    tiling.set_ub_calc_tail(ub_calc_tail);
    tiling.set_ub_calc_loop(ub_loop);
    // calc ub factor for tail core
    if (core_calc_tail == 0) {
        tiling.set_ub_calc_tail_num(0);
        tiling.set_ub_calc_tail_tail(0);
        tiling.set_ub_calc_tail_loop(0);
    } else {
        uint32_t ub_tail_factor = ub_factor;
        uint32_t ub_tail_loop = (core_calc_tail + ub_tail_factor - 1) / ub_tail_factor;
        uint32_t ub_tail_num = ub_tail_factor;
        uint32_t ub_tail_tail =
            core_calc_tail % ub_tail_factor != 0 ? core_calc_tail - (ub_tail_loop - 1) * ub_tail_factor : 0;
        tiling.set_ub_calc_tail_num(ub_tail_num);
        tiling.set_ub_calc_tail_tail(ub_tail_tail);
        tiling.set_ub_calc_tail_loop(ub_tail_loop);
    }
}

static void LargeNLargeD(
    gert::TilingContext* context, RmsNormGradTilingData& tiling, uint32_t buffer_size, uint32_t row_val,
    uint32_t col_val, uint32_t core_num)
{
    // block split
    if (core_num == 0) {
        return;
    }
    uint32_t block_factor = (row_val + core_num - 1) / core_num;
    uint32_t block_dim = (row_val + block_factor - 1) / block_factor;
    uint32_t core_calc_num = block_factor;
    uint32_t core_calc_tail = row_val % block_factor != 0 ? row_val - (block_dim - 1) * block_factor : 0;
    tiling.set_block_factor(block_factor);
    tiling.set_core_calc_num(core_calc_num);
    tiling.set_core_calc_tail(core_calc_tail);
    context->SetBlockDim(block_dim);
    tiling.set_block_dim(block_dim);
    // ub split
    uint32_t ub_factor = buffer_size;
    uint32_t ub_loop = (col_val + ub_factor - 1) / ub_factor;
    uint32_t ub_calc_num = ub_factor;
    uint32_t ub_calc_tail = col_val % ub_factor != 0 ? col_val - (ub_loop - 1) * ub_factor : 0;
    tiling.set_ub_split_dim(1);
    tiling.set_ub_factor(ub_factor);
    tiling.set_ub_calc_num(ub_calc_num);
    tiling.set_ub_calc_tail(ub_calc_tail);
    tiling.set_ub_calc_loop(ub_loop);
    if (core_calc_tail == 0) {
        tiling.set_ub_calc_tail_num(0);
        tiling.set_ub_calc_tail_tail(0);
        tiling.set_ub_calc_tail_loop(0);
    } else {
        tiling.set_ub_calc_tail_num(ub_calc_num);
        tiling.set_ub_calc_tail_tail(ub_calc_tail);
        tiling.set_ub_calc_tail_loop(ub_loop);
    }
}

static bool CheckInputDim(const gert::TilingContext* context, size_t dyDimNum, size_t xDimNum, size_t gammaDimNum)
{
    OP_CHECK_IF(
        xDimNum > MAX_DIM_NUM || xDimNum < MIN_DIM_X,
        OP_LOGE(context, "Input x's dim num should not greater than 8 or smaller than 1."), return false);
    OP_CHECK_IF(
        gammaDimNum > MAX_DIM_NUM || gammaDimNum < MIN_DIM_GAMMA,
        OP_LOGE(context, "Input gamma's dim num should not greater than 8 or smaller than 1."), return false);
    OP_CHECK_IF(
        gammaDimNum > xDimNum, OP_LOGE(context, "Input gamma's dim num should not greater than input x's."),
        return false);
    OP_CHECK_IF(
        dyDimNum != xDimNum, OP_LOGE(context, "Input dy/x shape invaild, dim num is not equal dy dim."), return false);
    return true;
}

static bool CheckOutputDim(
    const gert::TilingContext* context, size_t dyDimNum, size_t dxDimNum, size_t gammaDimNum, size_t dgammaDimNum)
{
    OP_CHECK_IF(
        dxDimNum != dyDimNum, OP_LOGE(context, "Output dx shape invaild, dim num is not equal dy dim."), return false);
    OP_CHECK_IF(
        gammaDimNum != dgammaDimNum, OP_LOGE(context, "Output dgamma shape invaild, dim num is not equal gamma dim."),
        return false);
    return true;
}

static bool CheckInputAndOutputShape(
    const gert::TilingContext* context, const gert::StorageShape* dyShape, const gert::StorageShape* xShape,
    const gert::StorageShape* dxShape)
{
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    for (uint32_t i = 0; i < xDimNum; i++) {
        OP_CHECK_IF(
            dyShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i),
            OP_LOGE(context, "Input dy/x shape invaild, shape is not equal dy first few dim."), return false);
        OP_CHECK_IF(
            dyShape->GetStorageShape().GetDim(i) != dxShape->GetStorageShape().GetDim(i),
            OP_LOGE(context, "Output dx shape invaild, shape is not equal dy first few dim."), return false);
    }
    OP_CHECK_IF(
        dyShape->GetStorageShape().GetDim(xDimNum - 1) == 0, OP_LOGE(context, "Input dy last shape can not be 0."), return false);
    return true;
}

static bool CheckRstdShape(
    const gert::TilingContext* context, const gert::StorageShape* xShape, const gert::StorageShape* rstdShape,
    size_t gammaDimNum)
{
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t rstdDimNum = rstdShape->GetStorageShape().GetDimNum();
    if (rstdDimNum < xDimNum - gammaDimNum) {
        for (uint32_t i = 0; i < rstdDimNum; i++) {
            OP_CHECK_IF(
                rstdShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i),
                OP_LOGE(context, "Input rstd shape invaild, shape is not equal x first few dim."), return false);
        }
        for (uint32_t i = rstdDimNum; i < xDimNum - gammaDimNum; i++) {
            OP_CHECK_IF(
                xShape->GetStorageShape().GetDim(i) != 1,
                OP_LOGE(context, "Input rstd shape invaild, shape is not equal x first few dim."), return false);
        }
    } else {
        for (uint32_t i = 0; i < xDimNum - gammaDimNum; i++) {
            OP_CHECK_IF(
                rstdShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i),
                OP_LOGE(context, "Input rstd shape invaild, shape is not equal x first few dim."), return false);
        }
    }
    return true;
}

static bool CheckGammaAndDgammaShape(
    const gert::TilingContext* context, const gert::StorageShape* gammaShape, const gert::StorageShape* dgammaShape,
    const gert::StorageShape* dyShape)
{
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    size_t dyDimNum = dyShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        dyDimNum < gammaDimNum, OP_LOGE(context, "dy dim num should not be smaller than gamma dim num."),
        return false);
    for (uint32_t i = 0; i < gammaDimNum; i++) {
        OP_CHECK_IF(
            gammaShape->GetStorageShape().GetDim(i) != dyShape->GetStorageShape().GetDim(dyDimNum - gammaDimNum + i),
            OP_LOGE(context, "Input gamma shape invaild, gamma shape is not equal dy last few dim."), return false);
        OP_CHECK_IF(
            dgammaShape->GetStorageShape().GetDim(i) != dyShape->GetStorageShape().GetDim(dyDimNum - gammaDimNum + i),
            OP_LOGE(context, "Input gamma shape invaild, gamma shape is not equal dy last few dim."), return false);
    }
    return true;
}

static bool CheckInputShape4RmsNormGrad(const gert::TilingContext* context)
{
    const gert::StorageShape* dyShape = context->GetInputShape(0);
    const gert::StorageShape* xShape = context->GetInputShape(1);
    const gert::StorageShape* rstdShape = context->GetInputShape(2);
    const gert::StorageShape* gammaShape = context->GetInputShape(3);
    const gert::StorageShape* dxShape = context->GetOutputShape(0);
    const gert::StorageShape* dgammaShape = context->GetOutputShape(1);

    size_t dyDimNum = dyShape->GetStorageShape().GetDimNum();
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    size_t dxDimNum = dxShape->GetStorageShape().GetDimNum();
    size_t dgammaDimNum = dgammaShape->GetStorageShape().GetDimNum();

    OP_CHECK_IF(
        !CheckInputDim(context, dyDimNum, xDimNum, gammaDimNum), OP_LOGE(context, "Input dim invalid."), return false);
    OP_CHECK_IF(
        !CheckOutputDim(context, dyDimNum, dxDimNum, gammaDimNum, dgammaDimNum),
        OP_LOGE(context, "Output dim invalid."), return false);
    OP_CHECK_IF(
        !CheckInputAndOutputShape(context, dyShape, xShape, dxShape), OP_LOGE(context, "Input/Output shape invalid."),
        return false);
    OP_CHECK_IF(
        !CheckRstdShape(context, xShape, rstdShape, gammaDimNum), OP_LOGE(context, "Rstd shape invalid."),
        return false);
    OP_CHECK_IF(
        !CheckGammaAndDgammaShape(context, gammaShape, dgammaShape, dyShape),
        OP_LOGE(context, "Gamma/dGamma shape invalid."), return false);
    return true;
}

static void SetTilingDataAndWorkspace(
    gert::TilingContext* context, RmsNormGradTilingData& tiling, uint32_t row_val, uint32_t col_val,
    uint32_t col_val_align, float avg_factor_val, uint64_t tiling_key)
{
    tiling.set_row(row_val);
    tiling.set_col(col_val);
    tiling.set_avg_factor(avg_factor_val);
    uint32_t fixed_output_flag = context->GetDeterministic() == 1 ? 1 : 0;
    OP_LOGI(context, "[RmsNormGrad] GetDeterministic state: %u", context->GetDeterministic());
    tiling.set_fixed_output(fixed_output_flag); // 0 is atomic add, 1 is fixed add output
    context->SetTilingKey(tiling_key);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t usr_workspace_size = fixed_output_flag == 1 ? (col_val_align + ALIGN_32) * tiling.get_block_dim() * 4 :
                                                         ALIGN_32 * tiling.get_block_dim() * 4;
    size_t sys_work_space_size = 16 * 1024 * 1024;
    size_t* current_workspace = context->GetWorkspaceSizes(1);
    current_workspace[0] = usr_workspace_size + sys_work_space_size;
}

static void UpdateShapeInfo(gert::TilingContext* context, uint32_t& col_val, uint32_t& row_val, uint32_t& rstd_shape)
{
    const gert::StorageShape* dy_shape = context->GetInputShape(0);
    const gert::StorageShape* rstd = context->GetInputShape(2);
    const gert::StorageShape* gamma = context->GetInputShape(3);
    auto gamma_dim_num = gamma->GetStorageShape().GetDimNum();
    for (uint32_t i = 0; i < gamma_dim_num; i++) {
        col_val *= gamma->GetStorageShape().GetDim(i);
    }
    for (uint32_t i = 0; i < dy_shape->GetStorageShape().GetDimNum() - gamma_dim_num; i++) {
        row_val *= dy_shape->GetStorageShape().GetDim(i);
    }
    for (uint32_t i = 0; i < rstd->GetStorageShape().GetDimNum(); i++) {
        rstd_shape *= rstd->GetStorageShape().GetDim(i);
    }
}

static ge::graphStatus Tiling4RmsNormGrad(gert::TilingContext* context)
{
    OP_CHECK_IF(
        !CheckInputShape4RmsNormGrad(context), OP_LOGE(context, "Input shape invalid."), return ge::GRAPH_FAILED);
    context->SetScheduleMode(SCHEDULE_MODE);
    RmsNormGradTilingData tiling;
    uint32_t col_val = 1;
    uint32_t row_val = 1;
    uint32_t rstd_shape = 1;
    UpdateShapeInfo(context, col_val, row_val, rstd_shape);
    bool isEmptyTensor = false;
    if (row_val == 0) {
        isEmptyTensor = true;
    }
    float avg_factor_val = 1.0f / col_val;
    if (rstd_shape != row_val) {
        return ge::GRAPH_FAILED;
    }
    // calc Tiling Factor
    auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t max_ub_size;
    ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, max_ub_size);
    platform_ascendc::SocVersion curSocVersion = ascendc_platform.GetSocVersion();
    OP_CHECK_IF(
        curSocVersion == platform_ascendc::SocVersion::ASCEND910 && col_val % COL_VAL_MULTIPLE_910 != 0,
        OP_LOGE(context, "The input shape is not supported on the 910 chip."), return ge::GRAPH_FAILED);
    bool isRegbase_ = IsRegbaseSocVersion(context) ? true : false;
    if (isRegbase_) {
        if (isEmptyTensor) {
            OP_LOGD(context, "RmsNormGrad Empty tiling start");
            RmsNormGradEmptyTiling rmsNormGradTiling(context);
            return rmsNormGradTiling.DoTiling();
        }
        OP_LOGD(context, "RmsNormGrad Regbase tiling start");
        return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
    }
    OP_CHECK_IF(
        isEmptyTensor, OP_LOGE(context, "Input dy shape can not be 0."), return ge::GRAPH_FAILED);
    uint32_t core_num = ascendc_platform.GetCoreNumAiv();
    auto data_type = context->GetInputDesc(0)->GetDataType();
    uint32_t buffer_size = BUFFER_SIZE_SPLIT_N_HIGH_PRECISION;
    // key include: dtype, splitN, splitD
    uint64_t dtype_key = DTYPE_FP32;
    if (data_type == ge::DT_FLOAT16) {
        dtype_key = DTYPE_FP16;
    }
    if (data_type == ge::DT_BF16) {
        dtype_key = DTYPE_BF16;
    }
    uint32_t align_factor = dtype_key == 1 ? ALIGN_32 : ALIGN_16;
    uint32_t col_val_align = (col_val + align_factor - 1) / align_factor * align_factor;
    tiling.set_data_type(dtype_key - 1);
    uint64_t ub_key = UB_SPLIT_N;
    uint64_t tiling_key;

    if (col_val <= buffer_size) {
        tiling_key = ub_key;
        if (col_val_align <= SMALLD_THRESHOLD) {
            buffer_size = CalcSmallDBufferSize(col_val_align);
        }
        LargeNSmallD(context, tiling, buffer_size, row_val, col_val_align, core_num);
    } else {
        ub_key = UB_SPLIT_D;
        tiling_key = ub_key;
        buffer_size = BUFFER_SIZE_SPLIT_D_HIGH_PRECISION;
        LargeNLargeD(context, tiling, buffer_size, row_val, col_val, core_num);
    }
    SetTilingDataAndWorkspace(context, tiling, row_val, col_val, col_val_align, avg_factor_val, tiling_key);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4RmsNormGrad(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Enter TilingPrepare4RmsNormGrad");
    return ge::GRAPH_SUCCESS;
}

struct RmsNormCompileInfo {};
IMPL_OP_OPTILING(RmsNormGrad).Tiling(Tiling4RmsNormGrad).TilingParse<RmsNormCompileInfo>(TilingPrepare4RmsNormGrad);

} // namespace optiling