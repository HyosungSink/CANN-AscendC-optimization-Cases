/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "swi_glu_tiling_def.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void swi_glu(GM_ADDR input_gm, GM_ADDR output_gm,
                                                GM_ADDR workspace, GM_ADDR tiling);

class swi_glu_test : public testing::Test {
protected:
    static void SetUpTestCase() {
      cout << "swi_glu_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
      cout << "swi_glu_test TearDown\n" << endl;
    }
};

TEST_F(swi_glu_test, test_case_fp16) {
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int16_t);
    size_t outputByteSize = 256 * 640 * sizeof(int16_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 1;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(swi_glu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(swi_glu_test, test_case_fp32) {
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputByteSize = 256 * 320 * 2 * sizeof(int32_t);
    size_t outputByteSize = 256 * 320 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 20;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 1;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 27;
    tilingDatafromBin->baseColLen = 256;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(swi_glu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(swi_glu_test, test_case_bf16) {
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputByteSize = 256 * 320 * 2 * sizeof(int16_t);
    size_t outputByteSize = 256 * 320 * sizeof(int16_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 1;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(27);
    ICPU_RUN_KF(swi_glu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(swi_glu_test, test_case_fp16_noalign) {
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputByteSize = 256 * 300 * 2 * sizeof(int16_t);
    size_t outputByteSize = 256 * 300 * sizeof(int16_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 0;
    tilingDatafromBin->isDoubleBuffer = 1;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 300;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(swi_glu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(swi_glu_test, test_case_fp32_noalign) {
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputByteSize = 256 * 600 * 2 * sizeof(int32_t);
    size_t outputByteSize = 256 * 600 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 20;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 0;
    tilingDatafromBin->isDoubleBuffer = 1;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 300;
    tilingDatafromBin->baseRowLen = 27;
    tilingDatafromBin->baseColLen = 256;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(swi_glu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(swi_glu_test, test_case_bf16_noalign) {
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputByteSize = 256 * 600 * 2 * sizeof(int16_t);
    size_t outputByteSize = 256 * 600 * sizeof(int16_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 0;
    tilingDatafromBin->isDoubleBuffer = 1;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 600;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(27);
    ICPU_RUN_KF(swi_glu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(swi_glu_test, test_case_fp16_no_buffer) {
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int16_t);
    size_t outputByteSize = 256 * 640 * sizeof(int16_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(swi_glu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(swi_glu_test, test_case_fp32_no_buffer) {
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputByteSize = 256 * 320 * 2 * sizeof(int32_t);
    size_t outputByteSize = 256 * 320 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 20;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 27;
    tilingDatafromBin->baseColLen = 256;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(swi_glu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(swi_glu_test, test_case_bf16_no_buffer) {
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputByteSize = 256 * 320 * 2 * sizeof(int16_t);
    size_t outputByteSize = 256 * 320 * sizeof(int16_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(27);
    ICPU_RUN_KF(swi_glu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(swi_glu_test, test_case_fp16_noalign_no_buffer) {
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputByteSize = 256 * 300 * 2 * sizeof(int16_t);
    size_t outputByteSize = 256 * 300 * sizeof(int16_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 0;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 300;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(swi_glu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(swi_glu_test, test_case_fp32_noalign_no_buffer) {
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputByteSize = 256 * 600 * 2 * sizeof(int32_t);
    size_t outputByteSize = 256 * 600 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 20;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 0;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 300;
    tilingDatafromBin->baseRowLen = 27;
    tilingDatafromBin->baseColLen = 256;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(swi_glu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(swi_glu_test, test_case_bf16_noalign_no_buffer) {
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputByteSize = 256 * 600 * 2 * sizeof(int16_t);
    size_t outputByteSize = 256 * 600 * sizeof(int16_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 0;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 600;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(27);
    ICPU_RUN_KF(swi_glu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}