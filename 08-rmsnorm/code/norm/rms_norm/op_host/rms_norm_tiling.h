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
 * \file rms_norm_tiling.h
 * \brief RmsNorm Op Tiling
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_RMS_NORM_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_RMS_NORM_H_

#include "register/tilingdata_base.h"
#include "log/log.h"
#include "error_util.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "op_host/tiling_templates_registry.h"
#include "op_common/op_host/util/platform_util.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(RMSNormTilingData)
TILING_DATA_FIELD_DEF(uint64_t, num_row);
TILING_DATA_FIELD_DEF(uint64_t, num_col);
TILING_DATA_FIELD_DEF(uint64_t, num_col_align);
TILING_DATA_FIELD_DEF(uint64_t, block_factor);
TILING_DATA_FIELD_DEF(uint32_t, row_factor);
TILING_DATA_FIELD_DEF(uint32_t, ub_factor);
TILING_DATA_FIELD_DEF(uint32_t, reduce_mask);
TILING_DATA_FIELD_DEF(uint32_t, left_num);
TILING_DATA_FIELD_DEF(uint32_t, last_reduce_mask);
TILING_DATA_FIELD_DEF(uint32_t, last_left_num);
TILING_DATA_FIELD_DEF(uint32_t, rstd_size);
TILING_DATA_FIELD_DEF(uint32_t, ub_loop);
TILING_DATA_FIELD_DEF(uint32_t, col_buffer_length);
TILING_DATA_FIELD_DEF(uint32_t, multi_n_num);
TILING_DATA_FIELD_DEF(uint32_t, is_nddma);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avg_factor);
TILING_DATA_FIELD_DEF(uint8_t, is_gemma);

TILING_DATA_FIELD_DEF(uint64_t, last_block_factor);
TILING_DATA_FIELD_DEF(uint64_t, row_loop);
TILING_DATA_FIELD_DEF(uint64_t, last_block_row_loop);
TILING_DATA_FIELD_DEF(uint64_t, row_tail);
TILING_DATA_FIELD_DEF(uint64_t, last_block_row_tail);
TILING_DATA_FIELD_DEF(uint32_t, mul_loop);
TILING_DATA_FIELD_DEF(uint32_t, mul_tail);
TILING_DATA_FIELD_DEF(uint8_t, dst_rep_stride);
TILING_DATA_FIELD_DEF(uint8_t, is_performance);
TILING_DATA_FIELD_DEF(uint8_t, normal_flag);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNorm, RMSNormTilingData)
REGISTER_TILING_DATA_CLASS(GemmaRmsNorm, RMSNormTilingData)

BEGIN_TILING_DATA_DEF(RMSNormArch35TilingData)
TILING_DATA_FIELD_DEF(uint64_t, num_row);
TILING_DATA_FIELD_DEF(uint64_t, num_col);
TILING_DATA_FIELD_DEF(uint64_t, num_col_align);
TILING_DATA_FIELD_DEF(uint64_t, block_factor);
TILING_DATA_FIELD_DEF(uint64_t, col_flod_factor);
TILING_DATA_FIELD_DEF(uint64_t, ub_factor);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avg_factor);
TILING_DATA_FIELD_DEF(uint64_t, last_block_factor);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RmsNorm_5000, RMSNormArch35TilingData)

constexpr uint32_t FLOAT_PER_REAPEAT = 64;
constexpr uint32_t ALING_FACTOR_512 = 512;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t X_REDUCE_TMP_NUM = 1;
constexpr uint32_t MODE_NORMAL = 0;
constexpr uint32_t MODE_SPLIT_D = 1;
constexpr uint32_t LOG_2 = 2;
constexpr uint32_t MULTI_FACTOR_2 = 2;
constexpr uint32_t NDDMA_BETTER_STAGE = 512;
constexpr uint32_t FLOAT_BYTE_SIZE = 4;
constexpr uint32_t RETAINED_SIZE = 256U * 5U * 4U;
constexpr uint32_t FULL_LOAD_R_MAX = 16384;
constexpr int32_t RETAINED_SIZE_MULTI_N = 256 * 3 * 4;
constexpr int32_t RETAINED_SIZE_1K = 2 * 1024;
constexpr uint32_t RMSNORM_REGBASE_NORMAL = 5000;
constexpr uint32_t RMSNORM_REGBASE_SPLIT = 2001;

struct Tiling4RmsNormCompileInfo {
    uint32_t totalCoreNum = 0;
    uint64_t totalUbSize = 0;
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
};

struct ComputeTotalBufSizeParam {
    uint32_t bufferNum = 0U;
    ge::DataType dtype;
    uint32_t dtypeSizeX = 0U;
    uint32_t dtypeSizeGamma = 0U;
    uint32_t length = 0U;
    bool split = false;
};

class RMSNormTilingInfo {
public:
    uint64_t ubSize{0};
    uint64_t numCol{0};
    uint64_t numRow{0};
    uint32_t numCore{0};
    uint64_t numColAlign{0};
    uint32_t xDtypeKey{0};

    bool isSoc910B{false};
};

const std::map<ge::DataType, uint32_t> dTypeByteMap = {
    {ge::DT_FLOAT16, 2},
    {ge::DT_FLOAT, 4},
    {ge::DT_BF16, 2},
};

template <typename T>
static auto CeilDiv(T x, T y) -> T
{
    return y == 0 ? x : (x + y - 1) / y;
}

inline uint32_t ComputeTotalBufSize(ComputeTotalBufSizeParam computeTotalBufSizeParam)
{
    uint32_t bufferNum = computeTotalBufSizeParam.bufferNum;
    ge::DataType dtype = computeTotalBufSizeParam.dtype;
    uint32_t dtypeSizeX = computeTotalBufSizeParam.dtypeSizeX;
    uint32_t dtypeSizeGamma = computeTotalBufSizeParam.dtypeSizeGamma;
    uint32_t length = computeTotalBufSizeParam.length;
    bool split = computeTotalBufSizeParam.split;

    // queBuferSize: 计算搬运需要空间大小
    uint32_t queBufSize = bufferNum * length * dtypeSizeX * MULTI_FACTOR_2 + bufferNum * length * dtypeSizeGamma +
                          FLOAT_PER_REAPEAT * bufferNum * FLOAT_BYTE_SIZE;
    uint32_t tmpBufSzie = 0; // tmpBufSzie: UB内需要临时空间大小
    if (split) {
        // 切分场景下：只需要一块
        tmpBufSzie = length * FLOAT_BYTE_SIZE;
        return queBufSize + tmpBufSzie + RETAINED_SIZE;
    } else {
        // 普通场景下：如果是float16及bfloat16数据类型，需要一块：转FP32
        tmpBufSzie = (dtype == ge::DT_FLOAT) ? 0 : length * FLOAT_BYTE_SIZE;
        return queBufSize + tmpBufSzie + static_cast<uint32_t>(RETAINED_SIZE_MULTI_N);
    }
}
ge::graphStatus TilingArch354RmsNorm(
    gert::TilingContext* context, uint64_t numRow, uint64_t numCol, uint32_t numCore, uint64_t ubSize,
    ge::DataType xDataType, ge::DataType gammaDataType, float epsilon, RMSNormTilingData& rmsNormTilingData, RMSNormArch35TilingData& rmsNormArch35TilingData);
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_RMS_NORM_H_
