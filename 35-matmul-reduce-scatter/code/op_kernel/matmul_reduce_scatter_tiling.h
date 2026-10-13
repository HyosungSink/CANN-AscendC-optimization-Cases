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
 * \file matmul_reduce_scatter_tiling.h
 * \brief
 */
#ifndef MATMUL_REDUCE_SCATTER_TILING_H
#define MATMUL_REDUCE_SCATTER_TILING_H

#include "kernel_tiling/kernel_tiling.h"

#if __has_include("../../common/inc/kernel/mc2_tiling_struct.h")
#include "../common/inc/kernel/mc2_tiling_struct.h"
#include "../3rd/mat_mul_v3/op_kernel/mat_mul_v3_tiling_data.h"
#else
#include "../common/inc/kernel/mc2_tiling_struct.h"
#include "../3rd/mat_mul_v3/op_kernel/mat_mul_v3_tiling_data.h"
#endif

struct ReduceScatterSoc {
    uint32_t commAlg;
	uint32_t isA3;
    uint32_t isStep;
    uint32_t isND2NZ;
    uint32_t baseBD;
    uint32_t baseBN;
};

class MatmulReduceScatterTilingData {
public:
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    Mc2Tiling::RCSTiling param;
    TCubeTiling tileTiling;
    TCubeTiling tailTiling;
    TCubeTiling localTiling;
    Mc2Tiling::TileL2Tiling tileL2Tiling;
    Mc2Tiling::TileL2Tiling tailL2Tiling;
    Mc2Tiling::TileL2Tiling localL2Tiling;
	ReduceScatterSoc socParam;
    Mc2L2cacheUseInfo l2cacheUseInfo;
};

#endif //__MATMUL_REDUCE_SCATTER_TILING_H__