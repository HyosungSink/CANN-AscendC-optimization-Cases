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
#include "add_rms_norm_quant_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void add_rms_norm_quant(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zero_points1, GM_ADDR zero_points2,
    GM_ADDR beta, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling);

class add_rms_norm_quant_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "add_rms_norm_quant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "add_rms_norm_quant_test TearDown\n" << endl;
    }
};

TEST_F(add_rms_norm_quant_test, test_case_0)
{
    size_t inputByteSize = 64 * 2560 * sizeof(int16_t);
    size_t gammaByteSize = 2560 * sizeof(int16_t);
    size_t scalesByteSize = 2560 * sizeof(float);
    size_t zeroPointsByteSize = 2560 * sizeof(int32_t);
    size_t outputYByteSize = 64 * 2560 * sizeof(int8_t);
    size_t outputXByteSize = 64 * 2560 * sizeof(int16_t);
    //   size_t rstdByteSize = 2 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormQuantTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(scalesByteSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(scalesByteSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(scalesByteSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(scalesByteSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputXByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 32;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormQuantTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormQuantTilingData*>(tiling);

    tilingDatafromBin->numRow = 64;
    tilingDatafromBin->numCol = 2560;
    tilingDatafromBin->blockFactor = 2;
    tilingDatafromBin->rowFactor = 64;
    tilingDatafromBin->ubFactor = 4864;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avgFactor = 0.01;
    tilingDatafromBin->hasZeroPoints1 = 1;
    tilingDatafromBin->hasBeta = 1U;
    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(
        add_rms_norm_quant, blockDim, x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, x,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_quant_test, test_case_1)
{
    size_t inputByteSize = 64 * 25600 * sizeof(int16_t);
    size_t gammaByteSize = 25600 * sizeof(int16_t);
    size_t scalesByteSize = 25600 * sizeof(float);
    size_t zeroPointsByteSize = 25600 * sizeof(int32_t);
    size_t outputYByteSize = 64 * 25600 * sizeof(int8_t);
    size_t outputXByteSize = 64 * 25600 * sizeof(int16_t);
    //   size_t rstdByteSize = 2 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormQuantTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(scalesByteSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(scalesByteSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(scalesByteSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(scalesByteSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputXByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 32;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormQuantTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormQuantTilingData*>(tiling);

    tilingDatafromBin->numRow = 64;
    tilingDatafromBin->numCol = 25600;
    tilingDatafromBin->blockFactor = 2;
    tilingDatafromBin->rowFactor = 64;
    tilingDatafromBin->ubFactor = 4576;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avgFactor = 0.01;
    tilingDatafromBin->hasZeroPoints1 = 1;
    tilingDatafromBin->hasBeta = 1U;

//     // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        add_rms_norm_quant, blockDim, x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, x,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_quant_test, test_case_3)
{
    size_t inputByteSize = 24 * 2560 * sizeof(int16_t);
    size_t gammaByteSize = 2560 * sizeof(int16_t);
    size_t scalesByteSize = 2560 * sizeof(float);
    size_t zeroPointsByteSize = 2560 * sizeof(int32_t);
    size_t outputYByteSize = 24 * 2560 * sizeof(int8_t);
    size_t outputXByteSize = 24 * 2560 * sizeof(int16_t);
    //   size_t rstdByteSize = 2 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormQuantTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(scalesByteSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(scalesByteSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(scalesByteSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(scalesByteSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputXByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 24;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormQuantTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormQuantTilingData*>(tiling);

    tilingDatafromBin->numRow = 24;
    tilingDatafromBin->numCol = 2560;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->rowFactor = 64;
    tilingDatafromBin->ubFactor = 6976;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avgFactor = 0.01;
    tilingDatafromBin->hasZeroPoints1 = 1;
    tilingDatafromBin->hasBeta = 1U;
    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(
        add_rms_norm_quant, blockDim, x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, x,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}