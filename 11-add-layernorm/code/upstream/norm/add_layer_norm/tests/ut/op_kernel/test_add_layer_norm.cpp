/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "add_layer_norm_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" void add_layer_norm(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR bias, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x,
    GM_ADDR workspace, GM_ADDR tiling);

class add_layer_norm_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "add_layer_norm_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "add_layer_norm_test TearDown\n" << endl;
    }
};

TEST_F(add_layer_norm_test, test_case_all_fp16)
{
    int N = 6;
    int D = 32;
    size_t inputByteSize = N * D * sizeof(int16_t);
    size_t gammaByteSize = D * sizeof(int16_t);
    size_t betaByteSize = D * sizeof(int16_t);
    size_t outputByteSize = N * D * sizeof(int16_t);
    size_t meanByteSize = N * sizeof(float);
    size_t rstdByteSize = N * sizeof(float);
    size_t tiling_data_size = sizeof(AddLayerNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* bias1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormTilingData*>(tiling);

    tilingDatafromBin->numCore = blockDim;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->firstDimPerCore = 3;
    tilingDatafromBin->firstDimPerCoreTail = 3;
    tilingDatafromBin->firstDimPerTime = 2;
    tilingDatafromBin->lastDimPerTime = D;
    tilingDatafromBin->eps = 0.01;
    tilingDatafromBin->aveFactor = 1.0 / 512;
    tilingDatafromBin->colMoveCnt = 1;
    tilingDatafromBin->colTail = D;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // normal fp16
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias1, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(101);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias1, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(102);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    // big_n fp16
    ICPU_SET_TILING_KEY(40);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(41);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias1, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(42);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(140);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(141);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias1, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(142);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    tilingDatafromBin->numCore = blockDim;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->firstDimPerCore = 3;
    tilingDatafromBin->firstDimPerCoreTail = 3;
    tilingDatafromBin->firstDimPerTime = 1;
    tilingDatafromBin->lastDimPerTime = D;
    tilingDatafromBin->eps = 0.01;
    tilingDatafromBin->aveFactor = 1.0 / 512;
    tilingDatafromBin->colMoveCnt = 1;
    tilingDatafromBin->colTail = D;

    // single fp16
    ICPU_SET_TILING_KEY(20);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(21);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias1, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(22);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(120);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(121);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias1, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(122);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    // singleext fp16
    ICPU_SET_TILING_KEY(30);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(31);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias1, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(32);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(130);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(131);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias1, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(132);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    tilingDatafromBin->numCore = blockDim;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->firstDimPerCore = 3;
    tilingDatafromBin->firstDimPerCoreTail = 3;
    tilingDatafromBin->firstDimPerTime = 1;
    tilingDatafromBin->eps = 0.01;
    tilingDatafromBin->aveFactor = 1.0 / 512;
    tilingDatafromBin->colMoveCnt = 2;
    tilingDatafromBin->colTail = D / 2;
    tilingDatafromBin->lastDimPerTime = D / 2;

    // slice fp16
    ICPU_SET_TILING_KEY(10);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(11);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias1, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(12);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(110);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(111);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias1, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(112);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    // sliceext fp16
    ICPU_SET_TILING_KEY(50);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(51);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias1, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(52);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(150);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(151);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias1, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(152);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(bias1);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_layer_norm_test, test_case_fp16_special_reduce)
{
    int N = 1024;
    int D = 640;
    size_t inputByteSize = N * D * sizeof(int16_t);
    size_t gammaByteSize = D * sizeof(int16_t);
    size_t betaByteSize = D * sizeof(int16_t);
    size_t outputByteSize = N * D * sizeof(int16_t);
    size_t meanByteSize = N * sizeof(float);
    size_t rstdByteSize = N * sizeof(float);
    size_t tiling_data_size = sizeof(AddLayerNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormTilingData*>(tiling);

    tilingDatafromBin->numCore = blockDim;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->firstDimPerCore = 26;
    tilingDatafromBin->firstDimPerCoreTail = 10;
    tilingDatafromBin->firstDimPerTime = 20;
    tilingDatafromBin->lastDimPerTime = D;
    tilingDatafromBin->eps = 0.00001;
    tilingDatafromBin->aveFactor = 1.0 / D;
    tilingDatafromBin->colMoveCnt = 1;
    tilingDatafromBin->colTail = D;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // special reduce
    ICPU_SET_TILING_KEY(70);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(72);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(170);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(172);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_layer_norm_test, test_case_fp16_special_reduce_big_N)
{
    int N = 2048;
    int D = 320;
    size_t inputByteSize = N * D * sizeof(int16_t);
    size_t gammaByteSize = D * sizeof(int16_t);
    size_t betaByteSize = D * sizeof(int16_t);
    size_t outputByteSize = N * D * sizeof(int16_t);
    size_t meanByteSize = N * sizeof(float);
    size_t rstdByteSize = N * sizeof(float);
    size_t tiling_data_size = sizeof(AddLayerNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 40;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormTilingData*>(tiling);

    tilingDatafromBin->numCore = blockDim;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->firstDimPerCore = 52;
    tilingDatafromBin->firstDimPerCoreTail = 20;
    tilingDatafromBin->firstDimPerTime = 40;
    tilingDatafromBin->lastDimPerTime = D;
    tilingDatafromBin->eps = 0.00001;
    tilingDatafromBin->aveFactor = 1.0 / D;
    tilingDatafromBin->colMoveCnt = 1;
    tilingDatafromBin->colTail = D;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // special reduce big n
    ICPU_SET_TILING_KEY(80);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(82);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(180);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(182);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// TEST_F(add_layer_norm_test, test_case_single_row_less_tensor)
// {
//     int N = 1;
//     int D = 12288;
//     size_t inputByteSize = N * D * sizeof(int16_t);
//     size_t gammaByteSize = D * sizeof(float);
//     size_t betaByteSize = D * sizeof(float);
//     size_t outputByteSize = N * D * sizeof(float);
//     size_t meanByteSize = N * sizeof(float);
//     size_t rstdByteSize = N * sizeof(float);
//     size_t tiling_data_size = sizeof(AddLayerNormTilingData);

//     uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
//     uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
//     uint8_t* bias = (uint8_t*)AscendC::GmAlloc(betaByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
//     uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 1;

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     AddLayerNormTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormTilingData*>(tiling);

//     tilingDatafromBin->numCore = blockDim;
//     tilingDatafromBin->numLastDim = D;
//     tilingDatafromBin->numFirstDim = N;
//     tilingDatafromBin->firstDimPerCore = 1;
//     tilingDatafromBin->firstDimPerCoreTail = 1;
//     tilingDatafromBin->firstDimPerTime = 1;
//     tilingDatafromBin->lastDimPerTime = D;
//     tilingDatafromBin->eps = 0.00001;
//     tilingDatafromBin->aveFactor = 1.0 / D;
//     tilingDatafromBin->colMoveCnt = 1;
//     tilingDatafromBin->colTail = D;

//     AscendC::SetKernelMode(KernelMode::AIV_MODE);

//     ICPU_SET_TILING_KEY(190);
//     ICPU_RUN_KF(
//         add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
//         (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x1);
//     AscendC::GmFree(x2);
//     AscendC::GmFree(gamma);
//     AscendC::GmFree(beta);
//     AscendC::GmFree(bias);
//     AscendC::GmFree(y);
//     AscendC::GmFree(mean);
//     AscendC::GmFree(rstd);
//     AscendC::GmFree(x);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }
