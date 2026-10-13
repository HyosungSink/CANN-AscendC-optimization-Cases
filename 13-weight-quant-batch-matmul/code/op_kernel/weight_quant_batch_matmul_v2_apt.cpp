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
 * \file weight_quant_batch_matmul_v2_apt.cpp
 * \brief
 */

#define ENABLE_L2_CACHE
#include "weight_quant_batch_matmul_v2_constant.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "lib/matmul_intf.h"
#include "arch35/weight_quant_batch_matmul_v2_arch35_tiling_key.h"
#include "arch35/weight_quant_batch_matmul_v2_arch35_tiling_data.h"

#if !defined(__DAV_310R6__) && !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
#if defined(A16) && (defined(WQBMMV2_S4) || defined(F4) || (defined(WQBMMV2_S8) && defined(WEIGHT_ND)))
#include "arch35/weight_quant_batch_matmul_v2_reg_base.h"
#endif
#if defined(A16) && defined(WQBMMV2_S8) && !defined(WEIGHT_ND)
#ifndef DTYPE_BIAS
#if defined(ORIG_DTYPE_X) && defined(DT_FLOAT16) && ORIG_DTYPE_X == DT_FLOAT16
#define DTYPE_BIAS DTYPE_X
#else
#define DTYPE_BIAS float
#endif
#endif
#include "weight_quant_batch_matmul_v2_custom_weight_nz.h"
#endif

#include "arch35/n_first/weight_quant_batch_matmul_v2_basic_block_controller.h"

#if defined(A16) && (defined(WQBMMV2_S4) || (defined(WQBMMV2_S8) && defined(WEIGHT_ND)) || defined(WEIGHT_F8_INPUT) || \
    defined(MICROSCALING))
#include "arch35/cmct_convertor.h"
using WeightQuantBatchMatmulV2::InvokeKernel;
#endif
#endif

#if defined(__DAV_310R6__)
#include "arch35/n_first/weight_quant_batch_matmul_v2_basic_block_controller.h"
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
#include "arch35/weight_quant_batch_matmul_v2_adaptive_sliding_window.h"
#include "arch35/weight_quant_batch_matmul_v2_iterbatch.h"
#endif

using namespace WeightQuantBatchMatmulV2;
using namespace WeightQuantBatchMatmulV2::Arch35;

constexpr MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;
constexpr MatmulBatchParams batchParams{false, BatchMode::BATCH_LESS_THAN_L1, false, BatchOutMode::MULTI_BATCH};
constexpr MatmulConfig MM_CFG_MULTI_BATCH = GetMMConfig<configMode>(batchParams);
constexpr MatmulBatchParams batchParamsNoBatchOut{false, BatchMode::BATCH_LESS_THAN_L1, false, BatchOutMode::SINGLE_BATCH};
constexpr MatmulConfig MM_CFG_MULTI_BATCH_NO_BATCH_OUT = GetMMConfig<configMode>(batchParamsNoBatchOut);

#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIGS[] = {
    {2, 512},   // idx 0
    {4, 512},   // idx 1
    {2, 1024},  // idx 2
    {4, 256},   // idx 3
    {2, 512},   // idx 4, int8 {2, 512}, int4 {4, 512}
    {2, 256},   // idx 5
    {4, 1024},  // idx 6
};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_0 = VEC_ANTIQUANT_CONFIGS[0];
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_1 = VEC_ANTIQUANT_CONFIGS[1];
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_5 = VEC_ANTIQUANT_CONFIGS[5];
#define INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(DtypeBias, templateClass, ...)                            \
    do {                                                                                                         \
        GET_TILING_DATA_WITH_STRUCT(                                                                             \
            wqbmmv2_tiling::WeightQuantBatchMatmulV2ASTilingDataParams, tilingDataIn, tiling);     \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_ANTIQUANT_SCALE, DtypeBias, DTYPE_Y, __VA_ARGS__> op;         \
        op.Init(                                                                                                 \
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
            &tPipe);                                                                                             \
        op.Process();                                                                                            \
    } while (0)
#endif

#if !defined(__DAV_310R6__) && !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
#define INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(DtypeBias, templateClass, ...)                                   \
    do {                                                                                                          \
        GET_TILING_DATA_WITH_STRUCT(                                                                              \
            wqbmmv2_tiling::WeightQuantBatchMatmulV2RegBaseTilingDataParams, tilingDataIn, tiling); \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DtypeBias, DTYPE_Y, __VA_ARGS__> op;                                 \
        op.Init(                                                                                                  \
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn,  \
            &tPipe);                                                                                              \
        op.Process();                                                                                             \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(templateClass, ...)                                                      \
    do {                                                                                                         \
        GET_TILING_DATA_WITH_STRUCT(WeightQuantBatchMatmulV2TilingData, tilingDataIn, tiling);                   \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                               \
        op.Init(                                                                                                 \
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
            &tPipe);                                                                                             \
        op.Process();                                                                                            \
    } while (0)
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
#define INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(templateClass, ...)                                                  \
    do {                                                                                                         \
        GET_TILING_DATA_WITH_STRUCT(                                                                             \
            wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingDataParams, tilingDataIn, tiling);    \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                               \
        op.Init(                                                                                                 \
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
            &tPipe);                                                                                             \
        op.Process();                                                                                            \
    } while (0)

#define INVOKE_WEIGHT_QUANT_BMM_ITERBATCH_IMPL(templateClass, ...)  \
    do {                                                                                                         \
        GET_TILING_DATA_WITH_STRUCT(                                                                             \
            wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingDataParams, tilingDataIn, tiling);    \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;                               \
        op.Init(                                                                                                 \
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, userWS, &tilingDataIn, \
            &tPipe);                                                                                             \
        op.Process();                                                                                            \
    } while (0)
#endif

template <
    int SOC_VERSION_TYPE, int SUB_SOC_VERSION_TYPE, int ANTIQUANT_SCENARIO, int ALGORITHM, int SUB_ALGORITHM,
    int TEMPLATE_CUSTOM, int API_CONSTEXPR, bool TRANS_A, bool TRANS_B, int ANTIQUANT_TYPE, int QUANT_TYPE,
    bool HAS_ANTIQUANT_OFFSET, bool HAS_BIAS, bool IS_BIAS_FP32, bool IS_WEIGHT_NZ>
__global__ __aicore__ void weight_quant_batch_matmul_v2(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    AscendC::TPipe tPipe;

#if !defined(__DAV_310R6__) && !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(wqbmmv2_tiling::WeightQuantBatchMatmulV2ASTilingDataParams);
    using DtypeBias = AscendC::Std::conditional_t<IS_BIAS_FP32, float, DTYPE_X>;
    static constexpr QuantType antiquantType = static_cast<QuantType>(ANTIQUANT_TYPE);
    static constexpr QuantType quantType = static_cast<QuantType>(QUANT_TYPE);
    if constexpr (SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ANTI_REG_SCSC) {
#if defined(A16) && (defined(WQBMMV2_S4) || defined(F4) || (defined(WQBMMV2_S8) && defined(WEIGHT_ND)))
        INVOKE_WEIGHT_QUANT_BMM_OP_REG_BASE_IMPL(
            DtypeBias, WeightQuantBatchMatmulV2RegBaseKernel, TRANS_A, TRANS_B, HAS_ANTIQUANT_OFFSET, antiquantType,
            IS_WEIGHT_NZ);
#endif
    } else if (SUB_ALGORITHM == WQBMMV2_SUB_ALGO_N_FIRST_TAIL_RESPLIT || SUB_ALGORITHM == WQBMMV2_SUB_ALGO_N_FIRST_BASIC_BLOCK) {
#if defined(A16) && (defined(WQBMMV2_S4) || (defined(WQBMMV2_S8) && defined(WEIGHT_ND)) || defined(WEIGHT_F8_INPUT) || defined(MICROSCALING))
        InvokeKernel<
            TEMPLATE_CUSTOM, TRANS_A, TRANS_B, ANTIQUANT_TYPE, HAS_ANTIQUANT_OFFSET, IS_BIAS_FP32, IS_WEIGHT_NZ>(
            KERNEL_PARAMS);
#endif
    } else if (SUB_ALGORITHM == WQBMMV2_SUB_ALGO_CUSTOM_BACKWARD_COMPATIBLE) {
#if defined(A16) && defined(WQBMMV2_S8) && !defined(WEIGHT_ND)
        INVOKE_WEIGHT_QUANT_BMM_OP_IMPL(
            WeightQuantBatchMatmulV2CustomWeightNzKernel, TRANS_A, TRANS_B, antiquantType, HAS_ANTIQUANT_OFFSET,
            quantType);
#endif
    }
#elif defined(__DAV_310R6__)
    using DtypeBias = AscendC::Std::conditional_t<IS_BIAS_FP32, float, DTYPE_X>;
    // 该分支处理bias与A类型一致的情况
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    REGISTER_TILING_DEFAULT(wqbmmv2_tiling::WeightQuantBatchMatmulV2ASTilingDataParams);
    #if (defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT != FORMAT_FRACTAL_NZ))
    if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_ASCEND910_55 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_VECTOR_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_N_FIRST_BASIC_BLOCK && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == false && TRANS_B == true && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == false && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(DtypeBias,
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_ASCEND910_55 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_VECTOR_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_N_FIRST_BASIC_BLOCK && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == false && TRANS_B == true && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == true && HAS_BIAS == false && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(DtypeBias,
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_ASCEND910_55 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_VECTOR_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_N_FIRST_BASIC_BLOCK && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_256_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == false && TRANS_B == false && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == false && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(DtypeBias,
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_5);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_ASCEND910_55 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_VECTOR_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_N_FIRST_BASIC_BLOCK && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_256_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == false && TRANS_B == false && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == true && HAS_BIAS == false && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(DtypeBias,
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_5);
    }
    #endif
    #if defined(ORIG_DTYPE_X) && defined(DT_BF16) && ORIG_DTYPE_X == DT_BF16
    #undef DTYPE_BIAS
    #define DTYPE_BIAS float
    // 该分支处理A数据类型为BF16且bias为fp32的情况
    if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_ASCEND910_55 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_VECTOR_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_N_FIRST_BASIC_BLOCK && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == false && TRANS_B == true && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == false && IS_BIAS_FP32 == true && IS_WEIGHT_NZ == false) {
        static constexpr WqmmConfig wqmmCfg = {
            false, true, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(DtypeBias,
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_ASCEND910_55 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_VECTOR_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_N_FIRST_BASIC_BLOCK && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == false && TRANS_B == true && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == true && HAS_BIAS == false && IS_BIAS_FP32 == true && IS_WEIGHT_NZ == false) {
        static constexpr WqmmConfig wqmmCfg = {false,         true, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR,
                                               CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(DtypeBias,
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_0);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_ASCEND910_55 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_VECTOR_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_N_FIRST_BASIC_BLOCK && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_256_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == false && TRANS_B == false && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == false && IS_BIAS_FP32 == true && IS_WEIGHT_NZ == false) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, QuantType::PER_CHANNEL, false, QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(DtypeBias,
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_5);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_ASCEND910_55 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_VECTOR_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_N_FIRST_BASIC_BLOCK && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_256_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == false && TRANS_B == false && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == true && HAS_BIAS == false && IS_BIAS_FP32 == true && IS_WEIGHT_NZ == false) {
        static constexpr WqmmConfig wqmmCfg = {
            false, false, QuantType::PER_CHANNEL, true, QuantType::PER_TENSOR, CubeFormat::ND};
        INVOKE_WEIGHT_QUANT_BMM_ADAPTIVE_SPLIT_OP_IMPL(DtypeBias,
            WeightQuantBatchMatmulV2BasicBlockController, wqmmCfg, VEC_ANTIQUANT_CONFIG_5);
    }
    #endif
#else
    #ifndef DTYPE_BIAS
    #define DTYPE_BIAS DTYPE_X
    #endif
    REGISTER_TILING_DEFAULT(wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingDataParams);
    #if (defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT != FORMAT_FRACTAL_NZ))
    if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_FIXPIPE_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ASW && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == false && TRANS_B == false && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == false && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(WeightQuantBatchMatmulV2ASWKernel, false, false, QuantType::PER_TENSOR,
                                            false, QuantType::PER_TENSOR);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_FIXPIPE_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ASW && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == false && TRANS_B == false && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == false && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(WeightQuantBatchMatmulV2ASWKernel, false, false, QuantType::PER_CHANNEL,
                                            false, QuantType::PER_TENSOR);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_FIXPIPE_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ASW && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == true && TRANS_B == true && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == true && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(WeightQuantBatchMatmulV2ASWKernel, true, true, QuantType::PER_TENSOR,
                                            false, QuantType::PER_TENSOR);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_FIXPIPE_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ASW && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == true && TRANS_B == true && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == false && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(WeightQuantBatchMatmulV2ASWKernel, true, true, QuantType::PER_TENSOR,
                                            false, QuantType::PER_TENSOR);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_FIXPIPE_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ASW && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == true && TRANS_B == true && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == true && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(WeightQuantBatchMatmulV2ASWKernel, true, true, QuantType::PER_CHANNEL,
                                            false, QuantType::PER_TENSOR);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_FIXPIPE_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ASW && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == true && TRANS_B == true && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == false && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(WeightQuantBatchMatmulV2ASWKernel, true, true, QuantType::PER_CHANNEL,
                                            false, QuantType::PER_TENSOR);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_FIXPIPE_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ASW && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == false && TRANS_B == false && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == true && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(WeightQuantBatchMatmulV2ASWKernel, false, false, QuantType::PER_TENSOR,
                                            false, QuantType::PER_TENSOR);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_FIXPIPE_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ASW && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == false && TRANS_B == true && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == true && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(WeightQuantBatchMatmulV2ASWKernel, false, true, QuantType::PER_TENSOR,
                                            false, QuantType::PER_TENSOR);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_FIXPIPE_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ASW && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == true && TRANS_B == false && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == true && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(WeightQuantBatchMatmulV2ASWKernel, true, false, QuantType::PER_TENSOR,
                                            false, QuantType::PER_TENSOR);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_FIXPIPE_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ASW && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == false && TRANS_B == true && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == false && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(WeightQuantBatchMatmulV2ASWKernel, false, true, QuantType::PER_TENSOR,
                                            false, QuantType::PER_TENSOR);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_FIXPIPE_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ASW && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == true && TRANS_B == false && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == false && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(WeightQuantBatchMatmulV2ASWKernel, true, false, QuantType::PER_TENSOR,
                                            false, QuantType::PER_TENSOR);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_FIXPIPE_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ASW && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == false && TRANS_B == false && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == true && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(WeightQuantBatchMatmulV2ASWKernel, false, false, QuantType::PER_CHANNEL,
                                            false, QuantType::PER_TENSOR);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_FIXPIPE_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ASW && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == false && TRANS_B == true && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == true && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(WeightQuantBatchMatmulV2ASWKernel, false, true, QuantType::PER_CHANNEL,
                                            false, QuantType::PER_TENSOR);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_FIXPIPE_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ASW && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == true && TRANS_B == false && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == true && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(WeightQuantBatchMatmulV2ASWKernel, true, false, QuantType::PER_CHANNEL,
                                            false, QuantType::PER_TENSOR);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_FIXPIPE_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ASW && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == false && TRANS_B == true && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == false && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(WeightQuantBatchMatmulV2ASWKernel, false, true, QuantType::PER_CHANNEL,
                                            false, QuantType::PER_TENSOR);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_SOC_VERSION_TYPE == WQBMMV2_DEFAULT && ANTIQUANT_SCENARIO == WQBMMV2_DEFAULT && ALGORITHM == WQBMMV2_ALGO_FIXPIPE_ANTIQUANT && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ASW && TEMPLATE_CUSTOM == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 && API_CONSTEXPR == WQBMMV2_DEFAULT && TRANS_A == true && TRANS_B == false && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL && QUANT_TYPE == WQBMMV2_QUANT_TYPE_PER_TENSOR && HAS_ANTIQUANT_OFFSET == false && HAS_BIAS == false && IS_BIAS_FP32 == false && IS_WEIGHT_NZ == false) {
        INVOKE_WEIGHT_QUANT_BMM_OP_ASW_IMPL(WeightQuantBatchMatmulV2ASWKernel, true, false, QuantType::PER_CHANNEL,
                                            false, QuantType::PER_TENSOR);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ITERATE_BATCH && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR) {
        INVOKE_WEIGHT_QUANT_BMM_ITERBATCH_IMPL(WeightQuantBatchMatmulV2IterBatchKernel, static_cast<bool>(TRANS_A), static_cast<bool>(TRANS_B), QuantType::PER_TENSOR,
                                            false, QuantType::PER_TENSOR, MM_CFG_MULTI_BATCH);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ITERATE_BATCH && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL) {
        INVOKE_WEIGHT_QUANT_BMM_ITERBATCH_IMPL(WeightQuantBatchMatmulV2IterBatchKernel, static_cast<bool>(TRANS_A), static_cast<bool>(TRANS_B), QuantType::PER_CHANNEL,
                                            false, QuantType::PER_TENSOR, MM_CFG_MULTI_BATCH);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ITERATE_BATCH_NO_BATCH_OUT && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR) {
        INVOKE_WEIGHT_QUANT_BMM_ITERBATCH_IMPL(WeightQuantBatchMatmulV2IterBatchKernel, static_cast<bool>(TRANS_A), static_cast<bool>(TRANS_B), QuantType::PER_TENSOR,
                                            false, QuantType::PER_TENSOR, MM_CFG_MULTI_BATCH_NO_BATCH_OUT);
    } else if constexpr (SOC_VERSION_TYPE == WQBMMV2_SOC_SUPPORT_MMAD_S8S4 && SUB_ALGORITHM == WQBMMV2_SUB_ALGO_ITERATE_BATCH_NO_BATCH_OUT && ANTIQUANT_TYPE == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL) {
        INVOKE_WEIGHT_QUANT_BMM_ITERBATCH_IMPL(WeightQuantBatchMatmulV2IterBatchKernel, static_cast<bool>(TRANS_A), static_cast<bool>(TRANS_B), QuantType::PER_CHANNEL,
                                            false, QuantType::PER_TENSOR, MM_CFG_MULTI_BATCH_NO_BATCH_OUT);
    }
#endif
#endif
}
