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
 * \file flash_attention_score_template_tiling_key.h
 * \brief
 */

#pragma once
#include "ascendc/host_api/tiling/template_argument.h"
#include "flash_attention_score_grad_tiling.h"

#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY (-1)
#endif

#define ASCENDC_TPL_1_BW 1
#define ASCENDC_TPL_3_BW 3
#define ASCENDC_TPL_10_BW 10
#define ASCENDC_TPL_12_BW 12

// 可表示的tilingkey范围为64bit，注意不能超过限制
ASCENDC_TPL_ARGS_DECL(
    FlashAttentionScoreGrad, // 算子唯一标识，可以opType保持一致
                             // bit: 0-3 UB0
                             //          0：B
                             //          1：N2
                             //          2：G
                             //          3：S1
                             //          4：S2
                             //          5：D
                             //          9：NONE
    ASCENDC_TPL_UINT_DECL(UB0, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 9),
    // bit: 4-7 UB1
    //           0：B
    //           1：N2
    //           2：G
    //           3：S1
    //           4：S2
    //           5：D
    //           9：NONE
    ASCENDC_TPL_UINT_DECL(UB1, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 9),
    // bit: 8-11 Block
    //           0：B
    //           1：N2
    //           2：G
    //           3：S1
    //           4：S2
    //           5：D
    //           9：NONE
    ASCENDC_TPL_UINT_DECL(Block, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 9),
    // bit: 12 IsSameAb
    //           0:False
    //           1:True
    ASCENDC_TPL_UINT_DECL(IsSameAb, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    // bit: 13-14 DataType
    //          0：FLOAT16
    //          1：FLOAT32
    //          2：BFLOAT16
    //          3：FLOAT16_PRECISION
    ASCENDC_TPL_UINT_DECL(DataType, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
    // bit: 15-16 Layout
    //          0: BSH
    //          1: SBH
    //          2: BNSD
    //          3: TND
    ASCENDC_TPL_UINT_DECL(Layout, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
    // bit: 17-20 Sparse
    //          0：ALL
    //          1：NONE
    //          2：ANY
    //          3：CAUSAL
    //          4：BAND
    //          5：PREFIX
    //          6：BAND_COMPRESS
    //          7：RIGHT_DOWN_CAUSAL
    //          8：RIGHT_DOWN_CAUSAL_BAND
    //          9：BAND_LEFT_UP_CAUSAL
    ASCENDC_TPL_UINT_DECL(Sparse, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    // bit: 21-22 MatmulConfig
    //          0：NULL_CONFIG
    //          1：NORMAL_CONFIG
    //          2：MDL_CONFIG
    ASCENDC_TPL_UINT_DECL(MatmulConfig, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2),
    // bit: 23 Mm12IsNZOut
    //          0: MM_NZ_OUT_FORMAT
    //          1: MM_ND_OUT_FORMAT
    ASCENDC_TPL_UINT_DECL(Mm12IsNZOut, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    // bit: 24 Mm12IsNZOut
    //          0: MM_NZ_OUT_FORMAT
    //          1: MM_ND_OUT_FORMAT
    ASCENDC_TPL_UINT_DECL(Mm345IsNZOut, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    // bit: 25 HasDropOut
    //          0: False
    //          1: True
    ASCENDC_TPL_UINT_DECL(HasDropOut, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    // bit: 26 HasPse
    //          0: False
    //          1: True
    ASCENDC_TPL_UINT_DECL(HasPse, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    // bit: 27 HasAttenMask
    //          0: False
    //          1: True
    ASCENDC_TPL_UINT_DECL(HasAttenMask, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    // bit: 28 EnableL1Reuse
    //          0: DISABLE
    //          1: ENABLE
    ASCENDC_TPL_UINT_DECL(EnableL1Reuse, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    // bit: 29 TNDS1Pingpong
    //          0: DISABLE
    //          1: ENABLE
    ASCENDC_TPL_UINT_DECL(TNDS1Pingpong, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    // bit: 30-33 S1TemplateType
    //          0：S_TEMPLATE_UNKNOW
    //          1：ALIGNED_16
    //          2：ALIGNED_32
    //          3：ALIGNED_48
    //          4：ALIGNED_64
    //          5：ALIGNED_80
    //          6：ALIGNED_96
    //          7：ALIGNED_112
    //          8：ALIGNED_128
    ASCENDC_TPL_UINT_DECL(S1TemplateType, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8),
    // bit: 34-37 S2TemplateType
    //          0：S_TEMPLATE_UNKNOW
    //          1：ALIGNED_16
    //          2：ALIGNED_32
    //          3：ALIGNED_48
    //          4：ALIGNED_64
    //          5：ALIGNED_80
    //          6：ALIGNED_96
    //          7：ALIGNED_112
    //          8：ALIGNED_128
    ASCENDC_TPL_UINT_DECL(S2TemplateType, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8),
    // bit: 38-41 DTemplateType
    //          0: NonAligned
    //          1: Aligned64
    //          3: NonAligned72
    //          4: NonAligned88
    //          5: Aligned128
    //          6: Aligned192
    ASCENDC_TPL_UINT_DECL(DTemplateType, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 3, 4, 5, 6, 8),
    // bit: 42 IsDeterministic
    //          0: NON_DETERMINISTIC
    //          1: True
    ASCENDC_TPL_UINT_DECL(IsDeterministic, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    // bit: 43 HasRope
    //          0: False
    //          1: DETERMINISTIC
    ASCENDC_TPL_UINT_DECL(HasRope, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1), );


ASCENDC_TPL_SEL(
    
#if (ORIG_DTYPE_QUERY == -1) || (ORIG_DTYPE_QUERY == DT_FLOAT16)
    //////////////////////////////////////////////////////////// BEGIN S1S2 ///////////////////////////////////////////////////////////////
    /////////////////////////// s1s2 INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL ///////////////////////////////
    // S1S2 NORMAL float16 2 * 2 * 2 * 2 = 16
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2)
    ),
    // S1S2 NORMAL float16 2 * 2 * 2 * 2 = 16
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2)
    ),
    // S1S2 NORMAL float16  2 * 2 * 2 = 8
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2)
    ),
    /////////////////////////// s1s2 INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL ///////////////////////////////
    //////////////////////////////////////////////////////////// END S1S2 /////////////////////////////////////////////////////////////////


    //////////////////////////////////////////////////////////// BEGIN SAMEAB ///////////////////////////////////////////////////////////////
    /////////////////////////// SAMEAB INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL ///////////////////////////////
    // SAMEAB NORMAL float16 4 * 2 * 2 * 2 * 2 * 2 * 2 = 256
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,1,2,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb)
    ),
    /////////////////////////// SAMEAB INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL ///////////////////////////////
    //////////////////////////////////////////////////////////// END SAMEAB /////////////////////////////////////////////////////////////////



    //////////////////////////////////////////////////////////// BEGIN BN2 ///////////////////////////////////////////////////////////////
    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_IMPL ///////////////////////////////
    // BN2 Float16 2 * 2 * 2 = 8
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_IMPL ///////////////////////////////

    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL ///////////////////////////////
    // BN2 NO DROP float16 2 * 2 * 2 = 8
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL ///////////////////////////////


    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_POST2KERNEL_IMPL ///////////////////////////////
    // BN2 POST2KERNEL float16 3 * 2 * 2 * 2 = 24
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,1,2),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    // BN2 POST2KERNEL float16 2 * 2 * 2 = 8
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL ///////////////////////////////

    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_POST2KERNEL_IMPL ///////////////////////////////
    // BN2 NO DROP POST2KERNEL float16 3 * 2 * 2 *2 = 24
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,1,2),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    // BN2 NO DROP POST2KERNEL float16 2 * 2 * 2 = 8
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_POST2KERNEL_IMPL ///////////////////////////////

    //////////////////////////////////////////////////////////// END BN2 /////////////////////////////////////////////////////////////////



    //////////////////////////////////////////////////////////// BEGIN BN ///////////////////////////////////////////////////////////////
    /////////////////////////// bn INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL ///////////////////////////////
    // BN NORMAL float16 3 * 2 * 2 = 12
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,1,2),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataUngs1s2Bbn)
    ),
    // BN NORMAL float16 2 * 2 = 4
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataUngs1s2Bbn)
    ),

    /////////////////////////// bn INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL ///////////////////////////////
    
    //////////////////////////////////////////////////////////// END BN /////////////////////////////////////////////////////////////////


    //////////////////////////////////////////////////////////// BEGIN B ///////////////////////////////////////////////////////////////

    /////////////////////////// b INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL ///////////////////////////////
    // B NORMAL float16 4 * 2 * 2 = 16
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,1,2,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradUbngs1s2BbTilingData)
    ),
    // B NORMAL float16 2 * 2 = 4
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradUbngs1s2BbTilingData)
    ),
    /////////////////////////// b INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL ///////////////////////////////
    //////////////////////////////////////////////////////////// END B /////////////////////////////////////////////////////////////////


    //////////////////////////////////////////////////////////// BEGIN BASIC ///////////////////////////////////////////////////////////////
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionGradMlaTilingData)
    ),
    //////////////////////////////////////////////////////////// END BASIC /////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////// BEGIN BASIC DET////////////////////////////////////////////////////////////
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionGradBasicDetTilingData)
    ),
    //////////////////////////////////////////////////////////// END BASIC DET //////////////////////////////////////////////////////////////
#endif

#if (ORIG_DTYPE_QUERY == -1) || (ORIG_DTYPE_QUERY == DT_BF16)
    //////////////////////////////////////////////////////////// BEGIN S1S2 /////////////////////////////////////////////////////////////////
    
    /////////////////////////// S1S2 INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL///////////////////////////////
    // S1S2 NORMAL bf16 2 * 2 * 2 * 2 = 16
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2)
    ),
    // S1S2 NORMAL bf16 2 * 2 * 2 * 2 = 16
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2)
    ),
    // S1S2 NORMAL bf16 2 * 2 * 2 = 8
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2)
    ),
    /////////////////////////// s1s2 INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL ///////////////////////////////
    

    /////////////////////////// s1s2 INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_ROPE_IMPL ///////////////////////////////
    // S1S2 ROPE bf16  2 * 2 = 4
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2)
    ),
    /////////////////////////// s1s2 INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_ROPE_IMPL ///////////////////////////////
    
    
    //////////////////////////////////////////////////////////// END S1S2 /////////////////////////////////////////////////////////////////
    
    
    //////////////////////////////////////////////////////////// BEGIN SAMEAB ///////////////////////////////////////////////////////////////
    /////////////////////////// SAMEAB INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL ///////////////////////////////

    // SAMEAB NORMAL BF16 4 * 2 * 2 * 2 * 2 * 2 * 2 = 256
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,1,2,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb)
    ),

    /////////////////////////// SAMEAB INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_IMPL ///////////////////////////////


    /////////////////////////// SAMEAB INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_L1_CUSTOM_IMPL ///////////////////////////////
    // SAMEAB L1 CUSTOM BF16 3 * 1 = 3
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,1,2),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb)
    ),
    // SAMEAB L1 CUSTOM BF16 3 * 2 = 6
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,1,2),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,5),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb)
    ),
    // SAMEAB L1 CUSTOM BF16 1
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,6),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb)
    ),
    /////////////////////////// SAMEAB INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_L1_CUSTOM_IMPL ///////////////////////////////


    /////////////////////////// SAMEAB INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_L1_CUSTOM_ROPE_IMPL ///////////////////////////////
    // SAMEAB L1 CUSTOM ROPE BF16 3 * 1 = 3
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,1,2),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,5),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb)
    ),
    // SAMEAB L1 CUSTOM ROPE BF16 1 * 1 = 3
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,6),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb)
    ),
    /////////////////////////// SAMEAB INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_L1_CUSTOM_ROPE_IMPL ///////////////////////////////


    /////////////////////////// SAMEAB INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_ROPE_IMPL ///////////////////////////////
    // SAMEAB ROPE BF16 4 * 2 * 2 * 2 = 32
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,1,2,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb)
    ),
    /////////////////////////// SAMEAB INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_SAMEAB_ROPE_IMPL ///////////////////////////////


    //////////////////////////////////////////////////////////// END SAMEAB ///////////////////////////////////////////////////////////////
    
    

    //////////////////////////////////////////////////////////// BEGIN BN2 ///////////////////////////////////////////////////////////////
    
    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_IMPL ///////////////////////////////
    // BN2 BF16 2 * 2 * 2 = 8
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_IMPL ///////////////////////////////
    

    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL ///////////////////////////////
    // BN2 NO DROP bf16 2 * 2 * 2 = 8
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL ///////////////////////////////


    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_POST2KERNEL_IMPL ///////////////////////////////
    // BN2 POST2KERNEL bf16 3 * 2 * 2 *2 = 24
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,1,2),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    // BN2 POST2KERNEL bf16 2 * 2 * 2 = 8
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_POST2KERNEL_IMPL ///////////////////////////////


    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_POST2KERNEL_IMPL ///////////////////////////////
    // BN2 NO DROP POST2KERNEL bf16 3 * 2 * 2 * 2 = 24
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,1,2),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    // BN2 NO DROP POST2KERNEL bf16 2 * 2 * 2 = 8
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_POST2KERNEL_IMPL ///////////////////////////////


    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROP_POSTIN_L1_IMPL ///////////////////////////////
    // BN2 NO DROP POSTIN L1 bf16 3 * 2 * 2 = 12
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,1,2),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROP_POSTIN_L1_IMPL ///////////////////////////////


    //////////////////////////////////////////////////////////// END BN2 ///////////////////////////////////////////////////////////////


    //////////////////////////////////////////////////////////// BEGIN BN ///////////////////////////////////////////////////////////////
    /////////////////////////// BN INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL ///////////////////////////////
    // BN NORMAL bf16 3 * 2 * 2 = 12
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,1,2),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataUngs1s2Bbn)
    ),
    // BN NORMAL bf16 1 = 1
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataUngs1s2Bbn)
    ),
    // BN NORMAL bf16 1 = 1
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataUngs1s2Bbn)
    ),
    // BN NORMAL bf16 1 = 1
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataUngs1s2Bbn)
    ),
    // BN NORMAL bf16 1 = 1
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataUngs1s2Bbn)
    ),
    // BN NORMAL bf16 1 = 1
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,5),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,5),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataUngs1s2Bbn)
    ),
    // BN NORMAL bf16 2 * 2 = 4
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataUngs1s2Bbn)
    ),
    /////////////////////////// BN INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL ///////////////////////////////
    //////////////////////////////////////////////////////////// END BN ///////////////////////////////////////////////////////////////


    //////////////////////////////////////////////////////////// BEGIN B ///////////////////////////////////////////////////////////////
    /////////////////////////// B INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL ///////////////////////////////
    // B NORMAL bf16 4 * 2 * 2 = 16
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,1,2,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradUbngs1s2BbTilingData)
    ),
    // B NORMAL bf16 2 * 2 = 4
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,3,4),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradUbngs1s2BbTilingData)
    ),
    // B NORMAL bf16 2 = 2
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradUbngs1s2BbTilingData)
    ),
    // B NORMAL bf16 2 * 2 = 4
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradUbngs1s2BbTilingData)
    ),
    /////////////////////////// B INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL ///////////////////////////////
    //////////////////////////////////////////////////////////// END B ///////////////////////////////////////////////////////////////


    //////////////////////////////////////////////////////////// BEGIN BASIC ///////////////////////////////////////////////////////////////
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionGradMlaTilingData)
    ),
    //////////////////////////////////////////////////////////// END BASIC /////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////// BEGIN BASIC DET ///////////////////////////////////////////////////////////////
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionGradBasicDetTilingData)
    ),
    //////////////////////////////////////////////////////////// END BASIC DET /////////////////////////////////////////////////////////////////
#endif

#if (ORIG_DTYPE_QUERY == -1) || (ORIG_DTYPE_QUERY == DT_FLOAT)

    //////////////////////////////////////////////////////////// BEGIN S1S2 ///////////////////////////////////////////////////////////////

    /////////////////////////// s1s2 INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL ///////////////////////////////
    // S1S2 NORMAL float32 4 * 2 * 2 * 2 = 32
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,1,2,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2)
    ),
    /////////////////////////// s1s2 INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL ///////////////////////////////

    //////////////////////////////////////////////////////////// END S1S2 /////////////////////////////////////////////////////////////////


    //////////////////////////////////////////////////////////// BEGIN BN2 ///////////////////////////////////////////////////////////////

    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_IMPL ///////////////////////////////
    // BN2 Float32 2 * 2 = 4
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_IMPL ///////////////////////////////

    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL ///////////////////////////////
    // BN2 NODROP float32 2 * 2 = 4
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL ///////////////////////////////

    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_POST2KERNEL_IMPL ///////////////////////////////
    // BN2 POST2KERNEL float32 3 * 2 * 2 = 12
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,1,2),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    // BN2 POST2KERNEL float32 2 * 2 = 4
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL ///////////////////////////////

    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_POST2KERNEL_IMPL ///////////////////////////////
    // BN2 NO DROP POST2KERNEL float32 3 * 2 * 2 = 12
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0,1,2),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    // BN2 NO DROP POST2KERNEL float32 2 * 2 = 4
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0,1),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_POST2KERNEL_IMPL ///////////////////////////////

    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROP_POSTIN_L1_IMPL ///////////////////////////////
    // BN2 NO DROP POSTIN L1 float32 1 = 1
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,4),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,3),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,2),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataS1s2Bn2)
    ),
    /////////////////////////// BN2 INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROP_POSTIN_L1_IMPL ///////////////////////////////

    //////////////////////////////////////////////////////////// END BN2 /////////////////////////////////////////////////////////////////



    
    //////////////////////////////////////////////////////////// BEGIN BN ///////////////////////////////////////////////////////////////
    /////////////////////////// bn INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL ///////////////////////////////
    // BN NORMAL float32 1 = 1
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingDataUngs1s2Bbn)
    ),

    /////////////////////////// bn INVOKE_FAG_GENERAL_NGS1S2_BN_IMPL ///////////////////////////////
    //////////////////////////////////////////////////////////// END BN /////////////////////////////////////////////////////////////////


    
    //////////////////////////////////////////////////////////// BEGIN B ///////////////////////////////////////////////////////////////
    /////////////////////////// b INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL ///////////////////////////////
    // B NORMAL float32  1 = 1
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,9),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,1),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradUbngs1s2BbTilingData)
    ),

    /////////////////////////// b INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL ///////////////////////////////
    //////////////////////////////////////////////////////////// END B /////////////////////////////////////////////////////////////////


#endif

    // 空tensor
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(UB0,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(UB1,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Block,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsSameAb,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DataType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Layout,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Sparse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(MatmulConfig,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm12IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(Mm345IsNZOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasDropOut,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasPse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasAttenMask,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(EnableL1Reuse,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(TNDS1Pingpong,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S1TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(S2TemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(DTemplateType,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(IsDeterministic,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_UINT_SEL(HasRope,ASCENDC_TPL_UI_LIST,0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreGradTilingData)
    ),

);