/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file weight_quant_batch_matmul_v2.cpp
 * \brief
 */

#define K_MAX_SHAPE_DIM 0

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
#include "weight_quant_batch_matmul_v2_constant.h"
#include "tool.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#if (                                      \
    defined(ORIG_DTYPE_ANTIQUANT_SCALE) && \
    ((ORIG_DTYPE_ANTIQUANT_SCALE == DT_UINT64) || (ORIG_DTYPE_ANTIQUANT_SCALE == DT_INT64)))
#include "fixpipe/weight_quant_batch_matmul_v2_fixpipe.h"
#else
#include "weight_quant_batch_matmul_v2_custom.h"
#if (defined(ORIG_DTYPE_Y) && ORIG_DTYPE_Y != DT_INT8)
#include "weight_quant_batch_matmul_v2_msd_multicore.h"
#include "weight_quant_batch_matmul_v2_msd_group.h"
#include "weight_quant_batch_matmul_v2_msd_split_k.h"
#include "weight_quant_batch_matmul_v2_custom_mix_splitk.h"
#if (defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT == FORMAT_FRACTAL_NZ))
#include "weight_quant_batch_matmul_v2_custom_weight_nz.h"
#include "weight_quant_batch_matmul_v2_custom_nz_splitk.h"
#endif
using WeightQuantBatchMatmulV2Msd::WeightQuantBatchMatmulV2MsdMultiCoreKernel;
#endif
#endif
#else
#include "weight_quant_batch_matmul_v2_weight_nz_performance.h"
#endif

#include "weight_quant_batch_matmul_v2_kernel_tiling_key.h"
using namespace WeightQuantBatchMatmulV2;

// if run with ttk without bias, can't get DTYPE_BIAS macro
#ifndef DTYPE_BIAS
#if defined(ORIG_DTYPE_X) && defined(DT_FLOAT16) && ORIG_DTYPE_X == DT_FLOAT16
#define DTYPE_BIAS DTYPE_X
#else
#define DTYPE_BIAS float
#endif
#endif

#ifndef DTYPE_ANTIQUANT_OFFSET
#if defined(ORIG_DTYPE_ANTIQUANT_SCALE) && defined(DT_UINT64) && ORIG_DTYPE_ANTIQUANT_SCALE != DT_UINT64 && \
    ORIG_DTYPE_ANTIQUANT_SCALE != DT_INT64
#define DTYPE_ANTIQUANT_OFFSET DTYPE_ANTIQUANT_SCALE
#else
#define DTYPE_ANTIQUANT_OFFSET int32_t
#endif
#endif

#if defined(ORIG_DTYPE_WEIGHT) && defined(DT_INT32) && ORIG_DTYPE_WEIGHT == DT_INT32
#undef DTYPE_WEIGHT
#define DTYPE_WEIGHT AscendC::int4b_t
#endif

#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510))
#define INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(templateClass, ...)                                                      \
    do {                                                                                                         \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2TilingData, tilingDataIn, tiling);                   \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                               \
        op.Init(                                                                                                 \
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
            &tPipe);                                                                                             \
        op.Process();                                                                                            \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(templateClass, ...)                                                  \
    do {                                                                                                         \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdTilingData, tilingDataIn, tiling);                \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                               \
        op.Init(                                                                                                 \
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
            &tPipe);                                                                                             \
        op.Process();                                                                                            \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(templateClass, ...)                                              \
    do {                                                                                                         \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2FixpipeTilingData, tilingDataIn, tiling);            \
        templateClass<DTYPE_ANTIQUANT_OFFSET, DTYPE_BIAS, __VA_ARGS__> op;                                       \
        op.Init(                                                                                                 \
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
            &tPipe);                                                                                             \
        op.Process();                                                                                            \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_IMPL_DTYPE(templateClass, ...)                                                \
    do {                                                                                                         \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2TilingData, tilingDataIn, tiling);                   \
        templateClass<__VA_ARGS__> op;                                                                           \
        op.Init(                                                                                                 \
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
            &tPipe);                                                                                             \
        op.Process();                                                                                            \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(templateClass, ...)                                                   \
    do {                                                                                                         \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2NzTilingData, tilingDataIn, tiling);                 \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                               \
        op.Init(                                                                                                 \
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
            &tPipe);                                                                                             \
        op.Process();                                                                                            \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_CUSTOM_SPLITK_IMPL(templateClass, ...)                                        \
    do {                                                                                                         \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2CustomNzSplitKTilingData, tilingDataIn, tiling);     \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                               \
        op.Init(                                                                                                 \
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
            &tPipe);                                                                                             \
        op.Process();                                                                                            \
    } while (0)
#endif

template <int SOC_VERSION_TYPE, int SUB_SOC_VERSION_TYPE, int ANTIQUANT_SCENARIO, int ALGORITHM, int SUB_ALGORITHM,
          int SUB_ALGORITHM_CUSTOM, int INNER_PRECISE, int TEMPLATE_CUSTOM, int API_CONSTEXPR, bool TRANS_A, bool TRANS_B,
          int ANTIQUANT_TYPE, int QUANT_TYPE, bool HAS_ANTIQUANT_OFFSET, bool HAS_BIAS, bool IS_BIAS_FP32, bool IS_WEIGHT_NZ,
          int TEMPLATE_EXTRA, int FULL_LOAD_MODE, int BATCH>
__global__ __aicore__ void weight_quant_batch_matmul_v2(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(WeightQuantBatchMatmulV2TilingData);
    AscendC::TPipe tPipe;
#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if (                                      \
    defined(ORIG_DTYPE_ANTIQUANT_SCALE) && \
    ((ORIG_DTYPE_ANTIQUANT_SCALE == DT_UINT64) || (ORIG_DTYPE_ANTIQUANT_SCALE == DT_INT64)))
// fixp方案
#if ((ORIG_DTYPE_X == DT_FLOAT16) && (ORIG_DTYPE_Y == DT_FLOAT16) && (ORIG_DTYPE_WEIGHT == DT_INT8))
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if ASCEND_IS_AIC {
        if constexpr(SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_L0C_TO_OUT && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT &&
            ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_FIXPIPE_ANTIQUANT &&
            SUB_ALGORITHM == WQBMMV2_SUB_ALGO_VDEFAULT && INNER_PRECISE == WQBMMV2_INNER_PRECISE_ZERO && 
            API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == 0 && TRANS_B == 1 && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL &&
            QUANT_TYPE == WQBMMV2_QUANT_TYPE_NONE && IS_BIAS_FP32 == 0 && IS_WEIGHT_NZ == 0) {
            static constexpr bool A_FULL_LOAD = TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_CUSTOM_A_SINGLE_M_SINGLE_K_FULL_LOAD ?
                true : false;
            // 模板参数为Trans bTrans antiquantType quantType hasAntiquantOffset hasBias weightFormat aFullLoad
            INVOKE_WEIGHT_QUANT_BMM_OP_FIXPIPE_IMPL(
                WeightQuantBatchMatmulV2FixpipeKernel, false, true, QuantType::PER_CHANNEL, QuantType::NONE, HAS_ANTIQUANT_OFFSET,
                HAS_BIAS, CubeFormat::ND, A_FULL_LOAD);
        }
    }
#endif
#elif (ORIG_DTYPE_Y == DT_INT8)
    if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_CUSTOM_ANTIQUANT && ALGORITHM == WQBMMV2_ALGO_CUSTOM && IS_WEIGHT_NZ == 0 &&
        TEMPLATE_EXTRA == WQBMMV2_TEMPLATE_EXTRA_NOT_USED && FULL_LOAD_MODE == WQBMMV2_FULL_LOAD_MODE_NOT_USED) {
        static constexpr QuantType antiQuantType = static_cast<QuantType>(ANTIQUANT_TYPE);
        static constexpr QuantType quantType = static_cast<QuantType>(QUANT_TYPE);
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset QuantType
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(
            WeightQuantBatchMatmulV2CustomKernel, TRANS_A, TRANS_B, antiQuantType, HAS_ANTIQUANT_OFFSET, quantType);
    }
#else
#if (defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT != FORMAT_FRACTAL_NZ))
    if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_CUSTOM_ANTIQUANT && ALGORITHM == WQBMMV2_ALGO_CUSTOM && IS_WEIGHT_NZ == 0 &&
        TEMPLATE_EXTRA == WQBMMV2_TEMPLATE_EXTRA_NOT_USED && FULL_LOAD_MODE == WQBMMV2_FULL_LOAD_MODE_NOT_USED) {
        static constexpr QuantType antiQuantType = static_cast<QuantType>(ANTIQUANT_TYPE);
        static constexpr QuantType quantType = static_cast<QuantType>(QUANT_TYPE);
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset QuantType
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(
            WeightQuantBatchMatmulV2CustomKernel, TRANS_A, TRANS_B, antiQuantType, HAS_ANTIQUANT_OFFSET, quantType);
    } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_MSD_MULTI_CORE && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR &&
               ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && TRANS_A == 0 && ALGORITHM == WQBMMV2_ALGO_CUSTOM && IS_WEIGHT_NZ == 0 &&
               TEMPLATE_EXTRA == WQBMMV2_TEMPLATE_EXTRA_NOT_USED && FULL_LOAD_MODE == WQBMMV2_FULL_LOAD_MODE_NOT_USED) {
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset quantType weightFormat
        INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(
            WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, TRANS_B, QuantType::PER_CHANNEL, HAS_ANTIQUANT_OFFSET,
            QuantType::PER_TENSOR, CubeFormat::ND);
    } else if constexpr (TEMPLATE_EXTRA == WQBMMV2_TEMPLATE_EXTRA_MSD_GENERAL && IS_WEIGHT_NZ == 0 &&
               SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_MSD_MULTI_CORE && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR &&
               TRANS_B == 0 && TRANS_A == 0 && ALGORITHM == WQBMMV2_ALGO_CUSTOM &&
               FULL_LOAD_MODE == WQBMMV2_FULL_LOAD_MODE_NOT_USED) {
        static constexpr QuantType antiQuantType = static_cast<QuantType>(ANTIQUANT_TYPE);
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset quantType weightFormat
        INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(
            WeightQuantBatchMatmulV2MsdSplitKKernel, false, false, antiQuantType, HAS_ANTIQUANT_OFFSET, QuantType::PER_TENSOR,
            CubeFormat::ND);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_L0C_TO_OUT && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT &&
               ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_MULTI_SCALE_DEQUANT &&
               SUB_ALGORITHM == WQBMMV2_SUB_ALGO_SPLIT_K && INNER_PRECISE == WQBMMV2_INNER_PRECISE_ONE && 
               TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_CUSTOM_A_NORMAL_LOAD && API_CONSTEXPR == WQBMMV2_DEFAULT &&
               TRANS_A == 0 && TRANS_B == 0 && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_GROUP &&
               QUANT_TYPE == WQBMMV2_QUANT_TYPE_NONE && HAS_BIAS == 0 && 
               IS_BIAS_FP32 == 0 && IS_WEIGHT_NZ == 0) {
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset quantType weightFormat preciseType
        INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(
            WeightQuantBatchMatmulV2MsdSplitKKernel, false, false, QuantType::PER_GROUP, HAS_ANTIQUANT_OFFSET, QuantType::PER_GROUP,
            CubeFormat::ND, HighPerformanceType);
    } else if constexpr (TEMPLATE_EXTRA == WQBMMV2_TEMPLATE_EXTRA_MSD_GENERAL && IS_WEIGHT_NZ == 0 &&
               SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_MSD_MULTI_CORE && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR &&
               ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && 
               TRANS_B == 1 && TRANS_A == 0 && ALGORITHM == WQBMMV2_ALGO_CUSTOM &&
               FULL_LOAD_MODE == WQBMMV2_FULL_LOAD_MODE_NOT_USED) {
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset quantType weightFormat
        INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(
            WeightQuantBatchMatmulV2MsdSplitKKernel, false, true, QuantType::PER_CHANNEL, HAS_ANTIQUANT_OFFSET, QuantType::PER_TENSOR,
            CubeFormat::ND);
    } else if constexpr (TEMPLATE_EXTRA == WQBMMV2_TEMPLATE_EXTRA_HIGH_PRECISION && IS_WEIGHT_NZ == 0 &&
               SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_MSD_MULTI_CORE && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR &&
               ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && 
               TRANS_B == 1 && TRANS_A == 0 && ALGORITHM == WQBMMV2_ALGO_CUSTOM &&
               FULL_LOAD_MODE == WQBMMV2_FULL_LOAD_MODE_NOT_USED) {
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset quantType weightFormat precisionType
        INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(
            WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, true, QuantType::PER_CHANNEL, HAS_ANTIQUANT_OFFSET,
            QuantType::PER_TENSOR, CubeFormat::ND, PrecisionType::HIGH_PRECISION);
    } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_MSD_GROUP && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR &&
               ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_GROUP &&
               TRANS_B == 0 && TRANS_A == 0 && ALGORITHM == WQBMMV2_ALGO_CUSTOM && IS_WEIGHT_NZ == 0 &&
        TEMPLATE_EXTRA == WQBMMV2_TEMPLATE_EXTRA_NOT_USED && FULL_LOAD_MODE == WQBMMV2_FULL_LOAD_MODE_NOT_USED) {
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset quantType weightFormat precisionType
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
        WeightQuantBatchMatMulV2MsdGroupKernel<
            DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false, QuantType::PER_GROUP, HAS_ANTIQUANT_OFFSET, QuantType::PER_TENSOR,
            CubeFormat::ND, HighPreciseType>
            op;
        op.Init(
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn,
            &tPipe);
        op.Process();
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_L0C_TO_OUT && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT &&
               ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_MULTI_SCALE_DEQUANT &&
               SUB_ALGORITHM == WQBMMV2_SUB_ALGO_VDEFAULT && INNER_PRECISE == WQBMMV2_INNER_PRECISE_ONE && 
               TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_CUSTOM_A_NORMAL_LOAD && API_CONSTEXPR == WQBMMV2_DEFAULT &&
               TRANS_A == 0 && TRANS_B == 0 && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_GROUP &&
               QUANT_TYPE == WQBMMV2_QUANT_TYPE_NONE && HAS_BIAS == 0 && 
               IS_BIAS_FP32 == 0 && IS_WEIGHT_NZ == 0) {
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset quantType weightFormat precisionType
        WeightQuantBatchMatMulV2MsdGroupKernel<
            DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false, QuantType::PER_GROUP, HAS_ANTIQUANT_OFFSET, QuantType::PER_TENSOR,
            CubeFormat::ND, HighPerformanceType>
            op;
        op.Init(
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn,
            &tPipe);
        op.Process();
    } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_MIX_SPLIT_K && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR &&
               HAS_ANTIQUANT_OFFSET == 1 && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_GROUP &&
               TRANS_B == 0 && TRANS_A == 0 && ALGORITHM == WQBMMV2_ALGO_CUSTOM && IS_WEIGHT_NZ == 0 &&
        TEMPLATE_EXTRA == WQBMMV2_TEMPLATE_EXTRA_NOT_USED && FULL_LOAD_MODE == WQBMMV2_FULL_LOAD_MODE_NOT_USED) {
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2TilingData, tilingDataIn, tiling);
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset quantType
        WeightQuantBatchMatmulV2MixSplitKKernel<
            bfloat16_t, int8_t, float, bfloat16_t, false, false, QuantType::PER_GROUP, true, QuantType::PER_TENSOR>
            op;
        op.Init(
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn,
            &tPipe);
        op.Process();
    }
#else
    if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_WEIGHT_NZ && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR &&
        ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL &&
        ALGORITHM == WQBMMV2_ALGO_CUSTOM &&
        TEMPLATE_EXTRA == WQBMMV2_TEMPLATE_EXTRA_NOT_USED && FULL_LOAD_MODE == WQBMMV2_FULL_LOAD_MODE_NOT_USED) {
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset quantType
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(
            WeightQuantBatchMatmulV2CustomWeightNzKernel, TRANS_A, TRANS_B, QuantType::PER_CHANNEL, HAS_ANTIQUANT_OFFSET,
            QuantType::PER_TENSOR);
    } else if constexpr (SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_WEIGHT_NZ && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR &&
               ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_GROUP &&
               TRANS_A == 0 && ALGORITHM == WQBMMV2_ALGO_CUSTOM &&
        TEMPLATE_EXTRA == WQBMMV2_TEMPLATE_EXTRA_NOT_USED && FULL_LOAD_MODE == WQBMMV2_FULL_LOAD_MODE_NOT_USED) {
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset quantType
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(
            WeightQuantBatchMatmulV2CustomWeightNzKernel, false, TRANS_B, QuantType::PER_GROUP, HAS_ANTIQUANT_OFFSET,
            QuantType::PER_TENSOR);
    } else if constexpr (IS_WEIGHT_NZ == 1 && SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_MSD_MULTI_CORE &&
               QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR &&
               ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && 
               TRANS_A == 0 && ALGORITHM == WQBMMV2_ALGO_CUSTOM && TEMPLATE_EXTRA == WQBMMV2_TEMPLATE_EXTRA_NOT_USED &&
               FULL_LOAD_MODE == WQBMMV2_FULL_LOAD_MODE_NOT_USED) {
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset quantType weightFormat
        INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(
            WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, TRANS_B, QuantType::PER_CHANNEL, HAS_ANTIQUANT_OFFSET,
            QuantType::PER_TENSOR, CubeFormat::NZ);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_L0C_TO_OUT && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT &&
               ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_MULTI_SCALE_DEQUANT &&
               SUB_ALGORITHM == WQBMMV2_SUB_ALGO_SPLIT_K &&
               TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_CUSTOM_A_NORMAL_LOAD && API_CONSTEXPR == WQBMMV2_DEFAULT &&
               TRANS_A == 0 && TRANS_B == 0 && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_GROUP &&
               QUANT_TYPE == WQBMMV2_QUANT_TYPE_NONE && HAS_BIAS == 0 && 
               IS_BIAS_FP32 == 0 && IS_WEIGHT_NZ == 1) {
        using SelectedType = std::conditional_t<
            (INNER_PRECISE == WQBMMV2_INNER_PRECISE_ONE),
            HighPerformanceType,
            HighPreciseType
        >;
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset quantType weightFormat preciseType
        INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(
            WeightQuantBatchMatmulV2MsdSplitKKernel, false, false, QuantType::PER_GROUP, HAS_ANTIQUANT_OFFSET, QuantType::PER_GROUP,
            CubeFormat::NZ, SelectedType);
    } else if constexpr (TEMPLATE_EXTRA == WQBMMV2_TEMPLATE_EXTRA_HIGH_PRECISION && IS_WEIGHT_NZ == 1 &&
               SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_MSD_MULTI_CORE && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR &&
               ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && 
               TRANS_B == 1 && TRANS_A == 0 && ALGORITHM == WQBMMV2_ALGO_CUSTOM &&
               FULL_LOAD_MODE == WQBMMV2_FULL_LOAD_MODE_NOT_USED) {
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset quantType weightFormat precisionType
        INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(
            WeightQuantBatchMatmulV2MsdMultiCoreKernel, false, true, QuantType::PER_CHANNEL, HAS_ANTIQUANT_OFFSET,
            QuantType::PER_TENSOR, CubeFormat::NZ, PrecisionType::HIGH_PRECISION);
    } else if constexpr (TEMPLATE_EXTRA == WQBMMV2_TEMPLATE_EXTRA_MSD_GENERAL && IS_WEIGHT_NZ == 1 &&
               SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_MSD_MULTI_CORE && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR &&
               ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && 
               TRANS_A == 0 && ALGORITHM == WQBMMV2_ALGO_CUSTOM &&
               FULL_LOAD_MODE == WQBMMV2_FULL_LOAD_MODE_NOT_USED) {
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset quantType weightFormat
        INVOKE_WEIGHT_QUANT_BMM_OP_MSD_IMPL(
            WeightQuantBatchMatmulV2MsdSplitKKernel, false, TRANS_B, QuantType::PER_CHANNEL, HAS_ANTIQUANT_OFFSET, QuantType::PER_TENSOR,
            CubeFormat::NZ);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_L0C_TO_OUT && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT &&
               ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_MULTI_SCALE_DEQUANT &&
               SUB_ALGORITHM == WQBMMV2_SUB_ALGO_VDEFAULT && 
               TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_CUSTOM_A_NORMAL_LOAD && API_CONSTEXPR == WQBMMV2_DEFAULT &&
               TRANS_A == 0 && TRANS_B == 0 && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_GROUP &&
               QUANT_TYPE == WQBMMV2_QUANT_TYPE_NONE && HAS_BIAS == 0 && 
               IS_BIAS_FP32 == 0 && IS_WEIGHT_NZ == 1) {
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset quantType weightFormat precisionType
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2MsdGroupTilingData, tilingDataIn, tiling);
        using SelectedType = std::conditional_t<
            (INNER_PRECISE == WQBMMV2_INNER_PRECISE_ONE),
            HighPerformanceType,
            HighPreciseType
        >;
        WeightQuantBatchMatMulV2MsdGroupKernel<
            DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, false, false, QuantType::PER_GROUP, HAS_ANTIQUANT_OFFSET, QuantType::PER_TENSOR,
            CubeFormat::NZ, SelectedType>
            op;
        op.Init(
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn,
            &tPipe);
        op.Process();
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_L0C_TO_OUT && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT &&
               ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_VECTOR_ANTIQUANT &&
               SUB_ALGORITHM == WQBMMV2_SUB_ALGO_SPLIT_K && INNER_PRECISE == WQBMMV2_INNER_PRECISE_ZERO && 
               API_CONSTEXPR == WQBMMV2_DEFAULT &&
               TRANS_A == 0 && TRANS_B == 1 && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL &&
               QUANT_TYPE == WQBMMV2_QUANT_TYPE_NONE && HAS_BIAS == 0 && 
               IS_BIAS_FP32 == 0 && IS_WEIGHT_NZ == 1) {
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset quantType aL1FullLoad
        static constexpr bool AL1_FULL_LOAD = TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_CUSTOM_A_SINGLE_M_SINGLE_K_FULL_LOAD ?
            true : false;
        INVOKE_WEIGHT_QUANT_BMM_OP_CUSTOM_SPLITK_IMPL(
            WeightQuantBatchMatmulV2CustomNzSplitkKernel, false, true, QuantType::PER_CHANNEL, HAS_ANTIQUANT_OFFSET,
            QuantType::PER_TENSOR, AL1_FULL_LOAD);
    } 
#endif
#endif
#else
    if constexpr (BATCH == WQBMMV2_BATCH_ZERO && SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_WEIGHT_NZ &&
        FULL_LOAD_MODE == WQBMMV2_FULL_LOAD_MODE_NONE_AB_K &&
        ALGORITHM == WQBMMV2_ALGO_CUSTOM && TEMPLATE_EXTRA == WQBMMV2_TEMPLATE_EXTRA_NOT_USED) {
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset
        static constexpr QuantType antiQuantType = static_cast<QuantType>(ANTIQUANT_TYPE);
        INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(
            WeightQuantBatchMatmulV2WeightNzPerformanceKernel, TRANS_A, true, antiQuantType, HAS_ANTIQUANT_OFFSET);
    } else if constexpr (BATCH == WQBMMV2_BATCH_ONE && SUB_ALGORITHM_CUSTOM == WQBMMV2_SUB_ALGO_WEIGHT_NZ &&
               FULL_LOAD_MODE == WQBMMV2_FULL_LOAD_MODE_NONE_AB_K &&
               ALGORITHM == WQBMMV2_ALGO_CUSTOM && TEMPLATE_EXTRA == WQBMMV2_TEMPLATE_EXTRA_NOT_USED) {
        // 模板参数为aTrans bTrans antiQuantType hasAntiQuantOffset
        static constexpr QuantType antiQuantType = static_cast<QuantType>(ANTIQUANT_TYPE);
        INVOKE_WEIGHT_QUANT_BMM_OP_NZ_IMPL(
            WeightQuantBatchMatmulV2WeightNzKernel, TRANS_A, true, antiQuantType, HAS_ANTIQUANT_OFFSET);
    }
#endif
}