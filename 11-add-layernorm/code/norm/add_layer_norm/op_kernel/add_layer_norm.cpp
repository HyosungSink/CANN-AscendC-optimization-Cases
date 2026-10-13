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
 * \file add_layer_norm.cpp
 * \brief
 */

#include "add_layer_norm_special_kernel.h"
#include "add_layer_norm_normal_special_reduce.h"
#include "add_layer_norm_single_row_less_tensor.h"
#include "add_layer_norm_kernel.h"

extern "C" __global__ __aicore__ void add_layer_norm(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR bias, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x,
    GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);

#define INIT_AND_PROCESS                                                                                     \
    op.Init(                                                                                                 \
        x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace, tiling_data.numCore, tiling_data.numLastDim, \
        tiling_data.numFirstDim, tiling_data.firstDimPerCore, tiling_data.firstDimPerCoreTail,               \
        tiling_data.firstDimPerTime, tiling_data.lastDimPerTime, tiling_data.eps, tiling_data.aveFactor,     \
        tiling_data.colMoveCnt, tiling_data.colTail, tiling_data.workspaceSize);                             \
    op.Process()
    if (TILING_KEY_IS(0)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 0> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(10)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 10> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(20)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 20> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(30)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 30> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(40)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 40> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(50)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 50> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 1> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(11)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 11> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(21)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 21> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(31)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 31> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(41)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 41> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(51)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 51> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 2> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(12)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 12> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(22)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 22> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(32)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 32> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(42)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 42> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(52)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 52> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(100)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 100> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(110)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 110> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(120)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 120> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(130)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 130> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(140)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 140> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(150)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 150> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(101)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 101> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(111)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 111> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(121)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 121> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(131)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 131> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(141)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 141> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(151)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 151> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(102)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 102> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(112)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 112> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(122)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 122> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(132)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 132> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(142)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 142> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(152)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 152> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(62)) { // Better UB begin
        KernelAddLayerNormBetterUB<half, half, half, half, 62> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(162)) {
        KernelAddLayerNormBetterUB<half, half, half, half, 162> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(70)) { // Normal Special Reduce begin
        KernelAddLayerNormNormalSpecialReduce<half, half, half, half, 70> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(170)) {
        KernelAddLayerNormNormalSpecialReduce<half, half, half, half, 170> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(80)) {
        KernelAddLayerNormNormalSpecialReduce<half, half, half, half, 80> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(180)) {
        KernelAddLayerNormNormalSpecialReduce<half, half, half, half, 180> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(72)) {
        KernelAddLayerNormNormalSpecialReduce<half, half, half, half, 72> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(172)) {
        KernelAddLayerNormNormalSpecialReduce<half, half, half, half, 172> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(82)) {
        KernelAddLayerNormNormalSpecialReduce<half, half, half, half, 82> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(182)) {
        KernelAddLayerNormNormalSpecialReduce<half, half, half, half, 182> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(190)) { // Single Row Less Tensor begin
        KernelAddLayerNormSingleRowLessTensor<DTYPE_X1, DTYPE_X2, float, DTYPE_Y, 190> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(192)) {
        KernelAddLayerNormSingleRowLessTensor<DTYPE_X1, DTYPE_X2, float, DTYPE_Y, 192> op(&pipe);
        INIT_AND_PROCESS;
    }
}
