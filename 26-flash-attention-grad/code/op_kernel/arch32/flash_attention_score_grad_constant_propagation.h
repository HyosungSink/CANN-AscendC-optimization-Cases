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
 * \file flash_attention_score_grad_constant_propagation.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_CONSTANT_PROPAGATION_H_
#define FLASH_ATTENTION_SCORE_GRAD_CONSTANT_PROPAGATION_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

enum class DTemplateType {
    Aligned64 = 64,
    NotAligned72 = 72,
    Aligned80 = 80,
    NotAligned88 = 88,
    Aligned96 = 96,
    Aligned128 = 128,
    Aligned160 = 160,
    Aligned192 = 192,
    Aligned256 = 256,
    Aligned512 = 512,
    NotAligned,
};

enum class STemplateType {
    Aligned512 = 512,
    NotAligned,
};

enum class S1TemplateType {
    NotAligned15 = 15,
    Aligned16 = 16,
    NotAligned30 = 30,
    Aligned32 = 32,
    Aligned48 = 48,
    NotAligned60 = 60,
    Aligned64 = 64,
    NotAligned,
};

enum class S2TemplateType {
    NotAligned15 = 15,
    Aligned16 = 16,
    NotAligned30 = 30,
    Aligned32 = 32,
    Aligned48 = 48,
    NotAligned60 = 60,
    Aligned64 = 64,
    NotAligned,
};


constexpr MatmulConfig FAG_CP_CFG_31_41_DEFAULT = {true, false, false, 0, 0, 0, false, false, false, false,
                                                0, 0, 0, 0, 0, 0, 0, 0, true, false, false, false, false, true,
                                                BatchMode::BATCH_LESS_THAN_L1, true, true, true, false, true, false};

#endif // FLASH_ATTENTION_SCORE_GRAD_CONSTANT_PROPAGATION_H_