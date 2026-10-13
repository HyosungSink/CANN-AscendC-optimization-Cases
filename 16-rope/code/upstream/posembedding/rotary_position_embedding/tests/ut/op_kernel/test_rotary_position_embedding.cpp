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
#include "test_rotary_position_embedding.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void rotary_position_embedding(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR rotate,
                                                                GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class rotary_position_embedding_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "rotary_position_embedding_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "rotary_position_embedding_test TearDown\n" << endl;
    }
};

// [1, 64, 2, 22] "BSND" float16 pad
TEST_F(rotary_position_embedding_test, test_case_mode_1_pad_fp16_001)
{
    size_t inputXByteSize = 3 * 4 * 46 * 14 * sizeof(half);
    size_t inputCosByteSize = 1 * 1 * 46 * 14 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 40;

    system(
        "cp -r ../../../../../posembedding/rotary_position_embedding/tests/ut/op_kernel/rotary_position_embedding_data "
        "./");
    system("chmod -R 755 ./rotary_position_embedding_data/");
    system("cd ./rotary_position_embedding_data/ && rm -rf ./*bin");
    system("cd ./rotary_position_embedding_data/ && python3 gen_data.py 3 4 46 14 float16");
    system("cd ./rotary_position_embedding_data/ && python3 gen_tiling.py case0");

    char *path_ = get_current_dir_name();
    string path(path_);
    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ReadFile(path + "/rotary_position_embedding_data/x.bin", inputXByteSize, x, inputXByteSize);
    ReadFile(path + "/rotary_position_embedding_data/cos.bin", inputCosByteSize, cos, inputCosByteSize);
    ReadFile(path + "/rotary_position_embedding_data/sin.bin", inputCosByteSize, sin, inputCosByteSize);
    ReadFile(path + "/rotary_position_embedding_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2001);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, nullptr, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [3, 4, 46, 64] "BNSD" bfloat16
TEST_F(rotary_position_embedding_test, test_case_mode_1_rotate_bf16_001)
{
    size_t inputXByteSize = 3 * 4 * 46 * 32 * sizeof(half);
    size_t inputCosByteSize = 1 * 1 * 46 * 32 * sizeof(half);
    size_t inputRotateByteSize = 32 * 32 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);
    uint8_t *rotate_inter = (uint8_t *)AscendC::GmAlloc(inputRotateByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 20;

    system(
        "cp -r ../../../../../posembedding/rotary_position_embedding/tests/ut/op_kernel/rotary_position_embedding_data "
        "./");
    system("chmod -R 755 ./rotary_position_embedding_data/");
    system("cd ./rotary_position_embedding_data/ && rm -rf ./*bin");
    system("cd ./rotary_position_embedding_data/ && rm -rf ./*pt");
    system("cd ./rotary_position_embedding_data/ && python3 gen_data_torch.py 3 4 46 32 bfloat16");
    system("cd ./rotary_position_embedding_data/ && python3 gen_tiling.py case1_32");

    char *path_ = get_current_dir_name();
    string path(path_);
    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ReadFile(path + "/rotary_position_embedding_data/x.pt", inputXByteSize, x, inputXByteSize);
    ReadFile(path + "/rotary_position_embedding_data/cos.pt", inputCosByteSize, cos, inputCosByteSize);
    ReadFile(path + "/rotary_position_embedding_data/sin.pt", inputCosByteSize, sin, inputCosByteSize);
    ReadFile(path + "/rotary_position_embedding_data/rotate_inter.pt", inputRotateByteSize, rotate_inter, inputRotateByteSize);
    ReadFile(path + "/rotary_position_embedding_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_SET_TILING_KEY(3013);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, rotate_inter, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(rotate_inter);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [3, 4, 46, 32] "BNSD" float32
TEST_F(rotary_position_embedding_test, test_case_mode_1_rotate_fp32_001)
{
    size_t inputXByteSize = 3 * 4 * 46 * 32 * sizeof(float);
    size_t inputCosByteSize = 3 * 4 * 46 * 32 * sizeof(float);
    size_t inputRotateByteSize = 32 * 32 * sizeof(float);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);
    uint8_t *rotate_inter = (uint8_t *)AscendC::GmAlloc(inputRotateByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 20;

    system(
        "cp -r ../../../../../posembedding/rotary_position_embedding/tests/ut/op_kernel/rotary_position_embedding_data "
        "./");
    system("chmod -R 755 ./rotary_position_embedding_data/");
    system("cd ./rotary_position_embedding_data/ && rm -rf ./*bin");
    system("cd ./rotary_position_embedding_data/ && rm -rf ./*pt");
    system("cd ./rotary_position_embedding_data/ && python3 gen_data_torch.py 3 4 46 32 float32");
    system("cd ./rotary_position_embedding_data/ && python3 gen_tiling.py case1_32");

    char *path_ = get_current_dir_name();
    string path(path_);
    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ReadFile(path + "/rotary_position_embedding_data/x.pt", inputXByteSize, x, inputXByteSize);
    ReadFile(path + "/rotary_position_embedding_data/cos.pt", inputCosByteSize, cos, inputCosByteSize);
    ReadFile(path + "/rotary_position_embedding_data/sin.pt", inputCosByteSize, sin, inputCosByteSize);
    ReadFile(path + "/rotary_position_embedding_data/rotate_inter.pt", inputRotateByteSize, rotate_inter, inputRotateByteSize);
    ReadFile(path + "/rotary_position_embedding_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_SET_TILING_KEY(3011);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, rotate_inter, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(rotate_inter);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// [3, 4, 46, 32] "BNSD" float16
TEST_F(rotary_position_embedding_test, test_case_mode_1_rotate_fp16_001)
{
    size_t inputXByteSize = 3 * 4 * 46 * 32 * sizeof(half);
    size_t inputCosByteSize = 1 * 1 * 46 * 64 * sizeof(half);
    size_t inputRotateByteSize = 32 * 32 * sizeof(half);
    size_t inputSinByteSize = inputCosByteSize;
    size_t outputYByteSize = inputXByteSize;
    size_t tilingDataSize = sizeof(RotaryPositionEmbeddingTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(inputCosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(inputSinByteSize);
    uint8_t *rotate_inter = (uint8_t *)AscendC::GmAlloc(inputRotateByteSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputYByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 20;

    system(
        "cp -r ../../../../../posembedding/rotary_position_embedding/tests/ut/op_kernel/rotary_position_embedding_data "
        "./");
    system("chmod -R 755 ./rotary_position_embedding_data/");
    system("cd ./rotary_position_embedding_data/ && rm -rf ./*bin");
    system("cd ./rotary_position_embedding_data/ && rm -rf ./*pt");
    system("cd ./rotary_position_embedding_data/ && python3 gen_data_torch.py 3 4 46 32 float16");
    system("cd ./rotary_position_embedding_data/ && python3 gen_tiling.py case1_32");

    char *path_ = get_current_dir_name();
    string path(path_);
    RotaryPositionEmbeddingTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbeddingTilingData *>(tiling);

    ReadFile(path + "/rotary_position_embedding_data/x.pt", inputXByteSize, x, inputXByteSize);
    ReadFile(path + "/rotary_position_embedding_data/cos.pt", inputCosByteSize, cos, inputCosByteSize);
    ReadFile(path + "/rotary_position_embedding_data/sin.pt", inputCosByteSize, sin, inputCosByteSize);
    ReadFile(path + "/rotary_position_embedding_data/rotate_inter.pt", inputRotateByteSize, rotate_inter, inputRotateByteSize);
    ReadFile(path + "/rotary_position_embedding_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_SET_TILING_KEY(3012);
    ICPU_RUN_KF(rotary_position_embedding, blockDim, x, cos, sin, rotate_inter, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(rotate_inter);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}