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
 * \file dynamic_quant_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/dynamic_quant_regbase_full_load.h"
#include "arch35/dynamic_quant_regbase_moe_full_load.h"
#include "arch35/dynamic_quant_regbase_large_shape_db.h"
#include "arch35/dynamic_quant_regbase_moe_large_shape.h"
#include "arch35/dynamic_quant_struct.h"
#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace AscendC;
using namespace DynamicQuantNDOpt;
using namespace DynamicQuantNDOpt2;

// 百位数选择DB，1/2分别表示不开DB/开DB
// 十位数为mode，0/1分别表示行全载模板/行切分模板
// 个位数为smooth，0/1分别表示无smooth输入/有smooth输入
template<uint64_t V>
using UIntAsBool = std::integral_constant<bool, V != 0>;

template <uint64_t useDb, uint64_t quantMode, uint64_t hasSmooth, uint64_t isSymmetrical>
__global__ __aicore__ void dynamic_quant(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_index, GM_ADDR y,
                                         GM_ADDR scale, GM_ADDR workSpace, GM_ADDR tiling) {
    #if (__NPU_ARCH__ == 3510)
        int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    #endif 
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);

    if constexpr (quantMode == TPL_COMMON_FULL_LOAD) {
        DynamicQuantRegbaseFullLoad<
            DTYPE_X, DTYPE_Y,
            UIntAsBool<hasSmooth>::value,
            static_cast<uint32_t>(useDb + 1),
            true
        > op(&pipe);
        op.Init(x, smooth_scales, y, scale, nullptr, workSpace, &tilingData);
        op.Process();
    } else if constexpr (quantMode == TPL_COMMON_LARGE_SHAPE) {
        DynamicQuantLargeShapeDb<
            DTYPE_X, DTYPE_Y,
            static_cast<int64_t>(hasSmooth),
            true
        > op(&pipe);
        op.Init(x, smooth_scales, y, scale, nullptr, workSpace, tilingData);
        op.Process();
    } else if constexpr (quantMode == TPL_MOE_FULL_LOAD) {
        DynamicQuantV2Op::DynamicQuantRegbaseFullLoadMOE<
            DTYPE_X, DTYPE_Y,
            UIntAsBool<hasSmooth>::value,
            static_cast<uint32_t>(useDb + 1),
            true
        > op(&pipe);
        op.Init(x, smooth_scales, group_index, y, scale, nullptr, workSpace, &tilingData);
        op.Process();
    } else if constexpr (quantMode == TPL_MOE_LARGE_SHAPE) {
        DynamicQuantRegBase::DynamicQuantLargeShapeMOE<
            DTYPE_X, DTYPE_Y,
            static_cast<int64_t>(hasSmooth),
            true
        > op(&pipe);
        op.Init(x, smooth_scales, group_index, y, scale, nullptr, workSpace, tilingData);
        op.Process();
    } else if constexpr (quantMode == TPL_EMPTY_TENSOR) {
        return ;
    }
    #if (__NPU_ARCH__ == 3510)
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
    #endif 
}