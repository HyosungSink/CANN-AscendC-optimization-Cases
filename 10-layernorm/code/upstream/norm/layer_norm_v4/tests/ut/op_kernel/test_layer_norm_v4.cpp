/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_layer_norm_v4.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "layer_norm_v4_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void layer_norm_v4(
    GM_ADDR x, GM_ADDR normalized_shape, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd,
    GM_ADDR workspace, GM_ADDR tiling);

class layer_norm_v4_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "layer_norm_v4_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "layer_norm_v4_test TearDown\n" << endl;
    }
};

TEST_F(layer_norm_v4_test, test_case_0001_100)
{
    size_t xByteSize = 1 * 960 * 8192 * sizeof(float);
    size_t gammaByteSize = 8192 * sizeof(float);
    size_t betaByteSize = 8192 * sizeof(float);
    size_t yByteSize = 1 * 960 * 8192 * sizeof(float);
    size_t meanByteSize = 1 * 960 * sizeof(float);
    size_t rstdByteSize = 1 * 960 * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormV4TilingDataSingleRead);
    uint32_t numBlocks = 48;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormV4TilingDataSingleRead* tilingDatafromBin = reinterpret_cast<LayerNormV4TilingDataSingleRead*>(tiling);

    tilingDatafromBin->numBlocks = 48;
    tilingDatafromBin->colSize = 960;
    tilingDatafromBin->rowSize = 8192;
    tilingDatafromBin->eps = 1e-05;
    tilingDatafromBin->coefficient = 0.0009765625;
    tilingDatafromBin->rowAlign = 8192;
    tilingDatafromBin->nRow = 2;
    tilingDatafromBin->tailNRow = 0;
    tilingDatafromBin->loopCount = 10;
    tilingDatafromBin->tailLoop = 0;
    tilingDatafromBin->tileLength = 16384;
    tilingDatafromBin->blockLength = 16384;
    tilingDatafromBin->nullptrGamma = 0;
    tilingDatafromBin->nullptrBeta = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(layer_norm_v4_test, test_case_0002)
{
    size_t xByteSize = 48 * 2 * sizeof(float);
    size_t gammaByteSize = 2 * sizeof(float);
    size_t betaByteSize = 2 * sizeof(float);
    size_t yByteSize = 48 * 2 * sizeof(float);
    size_t meanByteSize = 48 * sizeof(float);
    size_t rstdByteSize = 48 * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormV4TilingDataTranspose);
    uint32_t numBlocks = 48;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormV4TilingDataTranspose* tilingDatafromBin = reinterpret_cast<LayerNormV4TilingDataTranspose*>(tiling);

    tilingDatafromBin->col = 48;                // 输入tensor的行
    tilingDatafromBin->row = 2;                 // 输入tensor的列，即reduce的轴
    tilingDatafromBin->numBlocks = 48;           // 实际使用的core数量
    tilingDatafromBin->blockFormer = 1;         // 整核处理的row大小
    tilingDatafromBin->blockTail = 1;           // 尾核处理的row大小
    tilingDatafromBin->ubFormer = 1;            // ub整循环处理的row大小
    tilingDatafromBin->ubLoopOfFormerBlock = 1; // 整核处理的ub循环次数
    tilingDatafromBin->ubLoopOfTailBlock = 1;   // 尾核处理的ub循环次数
    tilingDatafromBin->ubTailOfFormerBlock = 1; // 整核ub尾循环处理的row大小
    tilingDatafromBin->ubTailOfTailBlock = 1;   // 尾核ub尾循环处理的row大小
    tilingDatafromBin->bFormer = 1;             // ubFormer借轴大小，ubFormer->16*bFormer
    tilingDatafromBin->dichotomizeAddDiffSize = 0;
    tilingDatafromBin->eps = 0.00001;
    tilingDatafromBin->coefficient = 0.5;
    tilingDatafromBin->nullptrGamma = 0;
    tilingDatafromBin->nullptrBeta = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(200);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(210);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(211);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(220);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(222);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(layer_norm_v4_test, test_case_0004)
{
    size_t xByteSize = 48 * 200 * sizeof(float);
    size_t gammaByteSize = 200 * sizeof(float);
    size_t betaByteSize = 200 * sizeof(float);
    size_t yByteSize = 48 * 200 * sizeof(float);
    size_t meanByteSize = 48 * sizeof(float);
    size_t rstdByteSize = 48 * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormV4CommonTilingData);
    uint32_t numBlocks = 48;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormV4CommonTilingData* tilingDatafromBin = reinterpret_cast<LayerNormV4CommonTilingData*>(tiling);

    tilingDatafromBin->colSize = 200;
    tilingDatafromBin->rowSize = 48;
    tilingDatafromBin->numBlocks = 48;
    tilingDatafromBin->space = 192 * 1024;
    tilingDatafromBin->eps = 0.00001;
    tilingDatafromBin->coefficient = 0.005;
    tilingDatafromBin-> nullptrGamma = false;
    tilingDatafromBin-> nullptrBeta = false;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(700);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(layer_norm_v4_test, test_case_0005)
{
    size_t xByteSize = 50 * 200 * sizeof(float);
    size_t gammaByteSize = 200 * sizeof(float);
    size_t betaByteSize = 200 * sizeof(float);
    size_t yByteSize = 50 * 200 * sizeof(float);
    size_t meanByteSize = 50 * sizeof(float);
    size_t rstdByteSize = 50 * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormV4CommonTilingData);
    uint32_t numBlocks = 50;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormV4CommonTilingData* tilingDatafromBin = reinterpret_cast<LayerNormV4CommonTilingData*>(tiling);

    tilingDatafromBin->colSize = 200;
    tilingDatafromBin->rowSize = 48;
    tilingDatafromBin->numBlocks = 48;
    tilingDatafromBin->space = 192 * 1024;
    tilingDatafromBin->eps = 0.00001;
    tilingDatafromBin->coefficient = 0.005;
    tilingDatafromBin-> nullptrGamma = false;
    tilingDatafromBin-> nullptrBeta = false;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(700);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(layer_norm_v4_test, test_case_0006)
{
    size_t xByteSize = 48 * 256 * sizeof(float);
    size_t gammaByteSize = 256 * sizeof(float);
    size_t betaByteSize = 256 * sizeof(float);
    size_t yByteSize = 48 * 256 * sizeof(float);
    size_t meanByteSize = 48 * sizeof(float);
    size_t rstdByteSize = 48 * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormV4MergeNTilingData);
    uint32_t numBlocks = 48;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormV4MergeNTilingData* tilingDatafromBin = reinterpret_cast<LayerNormV4MergeNTilingData*>(tiling);

    tilingDatafromBin-> numBlocks = 48;
    tilingDatafromBin-> colSize = 48;
    tilingDatafromBin-> rowSize = 256;
    float eps = 0.00001;
    float coefficient = 0.005;
    tilingDatafromBin-> rowAlign = 256;
    tilingDatafromBin-> nRow = 1;
    tilingDatafromBin-> ableNRow = 10;
    tilingDatafromBin-> loopCount = 1;
    tilingDatafromBin-> formerNum = 48;
    tilingDatafromBin-> formerRemainNum = 0;
    tilingDatafromBin-> tailNRow = 0;
    tilingDatafromBin-> tailLoop = 0;
    tilingDatafromBin-> tailNum = 0;
    tilingDatafromBin-> tailRemainNum = 0;
    tilingDatafromBin-> tileLength = 256;
    tilingDatafromBin-> blockLength = 2560;
    tilingDatafromBin-> nullptrGamma = false;
    tilingDatafromBin-> nullptrBeta = false;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(1600);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(layer_norm_v4_test, test_case_0007)
{
    size_t xByteSize = 193 * 256 * sizeof(float);
    size_t gammaByteSize = 256 * sizeof(float);
    size_t betaByteSize = 256 * sizeof(float);
    size_t yByteSize = 193 * 256 * sizeof(float);
    size_t meanByteSize = 193 * sizeof(float);
    size_t rstdByteSize = 193 * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormV4MergeNTilingData);
    uint32_t numBlocks = 48;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormV4MergeNTilingData* tilingDatafromBin = reinterpret_cast<LayerNormV4MergeNTilingData*>(tiling);

    tilingDatafromBin-> numBlocks = 48;
    tilingDatafromBin-> colSize = 193;
    tilingDatafromBin-> rowSize = 256;
    float eps = 0.00001;
    float coefficient = 0.005;
    tilingDatafromBin-> rowAlign = 256;
    tilingDatafromBin-> nRow = 3;
    tilingDatafromBin-> ableNRow = 3;
    tilingDatafromBin-> loopCount = 1;
    tilingDatafromBin-> formerNum = 1;
    tilingDatafromBin-> formerRemainNum = 2;
    tilingDatafromBin-> tailNRow = 3;
    tilingDatafromBin-> tailLoop = 1;
    tilingDatafromBin-> tailNum = 47;
    tilingDatafromBin-> tailRemainNum = 1;
    tilingDatafromBin-> tileLength = 784;
    tilingDatafromBin-> blockLength = 784;
    tilingDatafromBin-> nullptrGamma = false;
    tilingDatafromBin-> nullptrBeta = false;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(1600);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(layer_norm_v4_test, test_case_0008)
{
    size_t xByteSize = 48 * 2560 * sizeof(float);
    size_t gammaByteSize = 2560 * sizeof(float);
    size_t betaByteSize = 2560 * sizeof(float);
    size_t yByteSize = 48 * 2560 * sizeof(float);
    size_t meanByteSize = 48 * sizeof(float);
    size_t rstdByteSize = 48 * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormV4MergeNTilingData);
    uint32_t numBlocks = 48;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormV4MergeNTilingData* tilingDatafromBin = reinterpret_cast<LayerNormV4MergeNTilingData*>(tiling);

    tilingDatafromBin-> numBlocks = 48;
    tilingDatafromBin-> colSize = 48;
    tilingDatafromBin-> rowSize = 2560;
    float eps = 0.00001;
    float coefficient = 0.005;
    tilingDatafromBin-> rowAlign = 2560;
    tilingDatafromBin-> nRow = 1;
    tilingDatafromBin-> ableNRow = 10;
    tilingDatafromBin-> loopCount = 1;
    tilingDatafromBin-> formerNum = 48;
    tilingDatafromBin-> formerRemainNum = 0;
    tilingDatafromBin-> tailNRow = 0;
    tilingDatafromBin-> tailLoop = 0;
    tilingDatafromBin-> tailNum = 0;
    tilingDatafromBin-> tailRemainNum = 0;
    tilingDatafromBin-> tileLength = 2560;
    tilingDatafromBin-> blockLength = 5120;
    tilingDatafromBin-> nullptrGamma = false;
    tilingDatafromBin-> nullptrBeta = false;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(600);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(layer_norm_v4_test, test_case_0009)
{
    size_t xByteSize = 193 * 2560 * sizeof(float);
    size_t gammaByteSize = 2560 * sizeof(float);
    size_t betaByteSize = 2560 * sizeof(float);
    size_t yByteSize = 193 * 2560 * sizeof(float);
    size_t meanByteSize = 193 * sizeof(float);
    size_t rstdByteSize = 193 * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormV4MergeNTilingData);
    uint32_t numBlocks = 48;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormV4MergeNTilingData* tilingDatafromBin = reinterpret_cast<LayerNormV4MergeNTilingData*>(tiling);

    tilingDatafromBin-> numBlocks = 48;
    tilingDatafromBin-> colSize = 193;
    tilingDatafromBin-> rowSize = 2560;
    float eps = 0.00001;
    float coefficient = 0.005;
    tilingDatafromBin-> rowAlign = 2560;
    tilingDatafromBin-> nRow = 3;
    tilingDatafromBin-> ableNRow = 3;
    tilingDatafromBin-> loopCount = 1;
    tilingDatafromBin-> formerNum = 1;
    tilingDatafromBin-> formerRemainNum = 2;
    tilingDatafromBin-> tailNRow = 3;
    tilingDatafromBin-> tailLoop = 1;
    tilingDatafromBin-> tailNum = 47;
    tilingDatafromBin-> tailRemainNum = 1;
    tilingDatafromBin-> tileLength = 7840;
    tilingDatafromBin-> blockLength = 7840;
    tilingDatafromBin-> nullptrGamma = false;
    tilingDatafromBin-> nullptrBeta = false;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(600);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(layer_norm_v4_test, test_case_0010)
{
    size_t xByteSize = 48 * 1 * sizeof(float);
    size_t gammaByteSize = 1 * sizeof(float);
    size_t betaByteSize = 1 * sizeof(float);
    size_t yByteSize = 48 * 1 * sizeof(float);
    size_t meanByteSize = 48 * sizeof(float);
    size_t rstdByteSize = 48 * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormV4MergeNTilingData);
    uint32_t numBlocks = 48;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormV4MergeNTilingData* tilingDatafromBin = reinterpret_cast<LayerNormV4MergeNTilingData*>(tiling);

    tilingDatafromBin-> numBlocks = 48;
    tilingDatafromBin-> colSize = 48;
    tilingDatafromBin-> rowSize = 1;
    float eps = 0.00001;
    float coefficient = 0.005;
    tilingDatafromBin-> rowAlign = 8;
    tilingDatafromBin-> nRow = 1;
    tilingDatafromBin-> ableNRow = 10;
    tilingDatafromBin-> loopCount = 1;
    tilingDatafromBin-> formerNum = 48;
    tilingDatafromBin-> formerRemainNum = 0;
    tilingDatafromBin-> tailNRow = 0;
    tilingDatafromBin-> tailLoop = 0;
    tilingDatafromBin-> tailNum = 0;
    tilingDatafromBin-> tailRemainNum = 0;
    tilingDatafromBin-> tileLength = 8;
    tilingDatafromBin-> blockLength = 5120;
    tilingDatafromBin-> nullptrGamma = false;
    tilingDatafromBin-> nullptrBeta = false;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(2600);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(layer_norm_v4_test, test_case_0011)
{
    size_t xByteSize = 193 * 1 * sizeof(float);
    size_t gammaByteSize = 1 * sizeof(float);
    size_t betaByteSize = 1 * sizeof(float);
    size_t yByteSize = 193 * 1 * sizeof(float);
    size_t meanByteSize = 193 * sizeof(float);
    size_t rstdByteSize = 193 * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormV4MergeNTilingData);
    uint32_t numBlocks = 48;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormV4MergeNTilingData* tilingDatafromBin = reinterpret_cast<LayerNormV4MergeNTilingData*>(tiling);

    tilingDatafromBin-> numBlocks = 48;
    tilingDatafromBin-> colSize = 193;
    tilingDatafromBin-> rowSize = 1;
    float eps = 0.00001;
    float coefficient = 0.005;
    tilingDatafromBin-> rowAlign = 8;
    tilingDatafromBin-> nRow = 3;
    tilingDatafromBin-> ableNRow = 3;
    tilingDatafromBin-> loopCount = 1;
    tilingDatafromBin-> formerNum = 1;
    tilingDatafromBin-> formerRemainNum = 2;
    tilingDatafromBin-> tailNRow = 3;
    tilingDatafromBin-> tailLoop = 1;
    tilingDatafromBin-> tailNum = 47;
    tilingDatafromBin-> tailRemainNum = 1;
    tilingDatafromBin-> tileLength = 24;
    tilingDatafromBin-> blockLength = 24;
    tilingDatafromBin-> nullptrGamma = false;
    tilingDatafromBin-> nullptrBeta = false;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(2600);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}