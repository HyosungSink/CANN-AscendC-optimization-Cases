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
#include "rms_norm_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void rms_norm(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling);

class rms_norm_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "rms_norm_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "rms_norm_test TearDown\n" << endl;
    }
};

TEST_F(rms_norm_test, test_case_0)
{
    size_t inputByteSize = 2 * 256 * sizeof(int16_t);
    size_t gammaByteSize = 256 * sizeof(int16_t);
    size_t outputByteSize = 2 * 256 * sizeof(int16_t);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t tiling_data_size = sizeof(RMSNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    RMSNormTilingData* tilingDatafromBin = reinterpret_cast<RMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 256;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 64;
    tilingDatafromBin->ub_factor = 12288;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.01;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(rms_norm, blockDim, x, gamma, y, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rms_norm_test, test_case_1)
{
    size_t inputByteSize = 2 * 256 * sizeof(int16_t);
    size_t gammaByteSize = 256 * sizeof(int16_t);
    size_t outputByteSize = 2 * 256 * sizeof(int16_t);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t tiling_data_size = sizeof(RMSNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    RMSNormTilingData* tilingDatafromBin = reinterpret_cast<RMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 256;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 64;
    tilingDatafromBin->ub_factor = 12288;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.01;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(rms_norm, blockDim, x, gamma, y, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rms_norm_test, test_case_2_1)
{
    size_t inputByteSize = 48 * 256 * sizeof(int16_t);
    size_t gammaByteSize = 256 * sizeof(int16_t);
    size_t outputByteSize = 48 * 256 * sizeof(int16_t);
    size_t rstdByteSize = 48 * sizeof(float);
    size_t tiling_data_size = sizeof(RMSNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    RMSNormTilingData* tilingDatafromBin = reinterpret_cast<RMSNormTilingData*>(tiling);
    tilingDatafromBin->num_row = 48;
    tilingDatafromBin->num_col = 256;
    tilingDatafromBin->num_col_align = 256;
    tilingDatafromBin->block_factor = 2;
    tilingDatafromBin->row_factor = 44;
    tilingDatafromBin->ub_factor = 11264;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.003906;
    tilingDatafromBin->last_block_factor = 2;
    tilingDatafromBin->row_loop = 1;
    tilingDatafromBin->last_block_row_loop = 1;
    tilingDatafromBin->row_tail = 2;
    tilingDatafromBin->last_block_row_tail = 2;
    tilingDatafromBin->mul_loop = 4;
    tilingDatafromBin->mul_tail = 0;
    tilingDatafromBin->dst_rep_stride = 32;
    tilingDatafromBin->is_performance = 1;
    tilingDatafromBin->normal_flag = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(rms_norm, blockDim, x, gamma, y, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rms_norm_test, test_case_2_2)
{
    size_t inputByteSize = 48 * 2560 * sizeof(int16_t);
    size_t gammaByteSize = 2560 * sizeof(int16_t);
    size_t outputByteSize = 48 * 2560 * sizeof(int16_t);
    size_t rstdByteSize = 48 * sizeof(float);
    size_t tiling_data_size = sizeof(RMSNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    RMSNormTilingData* tilingDatafromBin = reinterpret_cast<RMSNormTilingData*>(tiling);
    tilingDatafromBin->num_row = 48;
    tilingDatafromBin->num_col = 2560;
    tilingDatafromBin->num_col_align = 2560;
    tilingDatafromBin->block_factor = 2;
    tilingDatafromBin->row_factor = 44;
    tilingDatafromBin->ub_factor = 10240;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.000391;
    tilingDatafromBin->last_block_factor = 2;
    tilingDatafromBin->row_loop = 1;
    tilingDatafromBin->last_block_row_loop = 1;
    tilingDatafromBin->row_tail = 2;
    tilingDatafromBin->last_block_row_tail = 2;
    tilingDatafromBin->mul_loop = 40;
    tilingDatafromBin->mul_tail = 0;
    tilingDatafromBin->dst_rep_stride = 64;
    tilingDatafromBin->is_performance = 1;
    tilingDatafromBin->normal_flag = 1;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(rms_norm, blockDim, x, gamma, y, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rms_norm_test, test_case_mixdtype_3)
{
    int numRow = 1;
    int numCol = 12288;
    size_t inputByteSize = numRow * numCol * sizeof(int16_t);
    size_t gammaByteSize = numCol * sizeof(float);
    size_t outputByteSize = numRow * numCol * sizeof(int16_t);
    size_t rstdByteSize = numRow * sizeof(float);
    size_t tiling_data_size = sizeof(RMSNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    RMSNormTilingData* tilingDatafromBin = reinterpret_cast<RMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 1;
    tilingDatafromBin->num_col = 12288;
    tilingDatafromBin->num_col_align = 12288;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 1;
    tilingDatafromBin->ub_factor = 12288;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.01;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(rms_norm, blockDim, x, gamma, y, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rms_norm_test, test_case_1110)
{
    size_t inputByteSize = 24 * 2560 * sizeof(int16_t);
    size_t gammaByteSize = 2560 * sizeof(int16_t);
    size_t outputByteSize = 24 * 2560 * sizeof(int16_t);
    size_t rstdByteSize = 24 * sizeof(float);
    size_t tiling_data_size = sizeof(RMSNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 256);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    RMSNormTilingData* tilingDatafromBin = reinterpret_cast<RMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 24;
    tilingDatafromBin->num_col = 2560;
    tilingDatafromBin->num_col_align = 2560;
    tilingDatafromBin->block_factor = 16;
    tilingDatafromBin->row_factor = 8;
    tilingDatafromBin->ub_factor = 1;
    tilingDatafromBin->reduce_mask = 8;
    tilingDatafromBin->left_num = 512;
    tilingDatafromBin->last_reduce_mask = 0;
    tilingDatafromBin->last_left_num = 0;
    tilingDatafromBin->rstd_size = 2048;
    tilingDatafromBin->ub_loop = 0;
    tilingDatafromBin->col_buffer_length = 0;
    tilingDatafromBin->multi_n_num = 0;
    tilingDatafromBin->is_nddma = 1;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.0004;
    tilingDatafromBin->is_gemma = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(1110);
    ICPU_RUN_KF(rms_norm, blockDim, x, gamma, y, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rms_norm_test, test_case_1120)
{
    size_t inputByteSize = 24 * 2560 * sizeof(float);
    size_t gammaByteSize = 2560 * sizeof(float);
    size_t outputByteSize = 24 * 2560 * sizeof(float);
    size_t rstdByteSize = 24 * sizeof(float);
    size_t tiling_data_size = sizeof(RMSNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    RMSNormTilingData* tilingDatafromBin = reinterpret_cast<RMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 24;
    tilingDatafromBin->num_col = 2560;
    tilingDatafromBin->num_col_align = 2560;
    tilingDatafromBin->block_factor = 16;
    tilingDatafromBin->row_factor = 8;
    tilingDatafromBin->ub_factor = 1;
    tilingDatafromBin->reduce_mask = 8;
    tilingDatafromBin->left_num = 512;
    tilingDatafromBin->last_reduce_mask = 0;
    tilingDatafromBin->last_left_num = 0;
    tilingDatafromBin->rstd_size = 2048;
    tilingDatafromBin->ub_loop = 0;
    tilingDatafromBin->col_buffer_length = 0;
    tilingDatafromBin->multi_n_num = 0;
    tilingDatafromBin->is_nddma = 1;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.0004;
    tilingDatafromBin->is_gemma = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(1120);
    ICPU_RUN_KF(rms_norm, blockDim, x, gamma, y, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// TEST_F(rms_norm_test, test_case_1130)
// {
//     size_t inputByteSize = 24 * 2560 * sizeof(int16_t);
//     size_t gammaByteSize = 2560 * sizeof(int16_t);
//     size_t outputByteSize = 24 * 2560 * sizeof(int16_t);
//     size_t rstdByteSize = 24 * sizeof(float);
//     size_t tiling_data_size = sizeof(RMSNormTilingData);

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 2;

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     RMSNormTilingData* tilingDatafromBin = reinterpret_cast<RMSNormTilingData*>(tiling);

//     tilingDatafromBin->num_row = 24;
//     tilingDatafromBin->num_col = 2560;
//     tilingDatafromBin->num_col_align = 2560;
//     tilingDatafromBin->block_factor = 16;
//     tilingDatafromBin->row_factor = 8;
//     tilingDatafromBin->ub_factor = 1;
//     tilingDatafromBin->reduce_mask = 8;
//     tilingDatafromBin->left_num = 512;
//     tilingDatafromBin->last_reduce_mask = 0;
//     tilingDatafromBin->last_left_num = 0;
//     tilingDatafromBin->rstd_size = 2048;
//     tilingDatafromBin->ub_loop = 0;
//     tilingDatafromBin->col_buffer_length = 0;
//     tilingDatafromBin->multi_n_num = 0;
//     tilingDatafromBin->is_nddma = 1;
//     tilingDatafromBin->epsilon = 0.01;
//     tilingDatafromBin->avg_factor = 0.0004;
//     tilingDatafromBin->is_gemma = 0;
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);

//     ICPU_SET_TILING_KEY(1130);
//     ICPU_RUN_KF(rms_norm, blockDim, x, gamma, y, rstd, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x);
//     AscendC::GmFree(gamma);
//     AscendC::GmFree(y);
//     AscendC::GmFree(rstd);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(rms_norm_test, test_case_1030)
// {
//     size_t inputByteSize = 24 * 2560 * sizeof(int16_t);
//     size_t gammaByteSize = 2560 * sizeof(int16_t);
//     size_t outputByteSize = 24 * 2560 * sizeof(int16_t);
//     size_t rstdByteSize = 24 * sizeof(float);
//     size_t tiling_data_size = sizeof(RMSNormTilingData);

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 2;

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     RMSNormTilingData* tilingDatafromBin = reinterpret_cast<RMSNormTilingData*>(tiling);

//     tilingDatafromBin->num_row = 24;
//     tilingDatafromBin->num_col = 2560;
//     tilingDatafromBin->num_col_align = 2560;
//     tilingDatafromBin->block_factor = 16;
//     tilingDatafromBin->row_factor = 8;
//     tilingDatafromBin->ub_factor = 1;
//     tilingDatafromBin->reduce_mask = 8;
//     tilingDatafromBin->left_num = 512;
//     tilingDatafromBin->last_reduce_mask = 0;
//     tilingDatafromBin->last_left_num = 0;
//     tilingDatafromBin->rstd_size = 2048;
//     tilingDatafromBin->ub_loop = 0;
//     tilingDatafromBin->col_buffer_length = 0;
//     tilingDatafromBin->multi_n_num = 0;
//     tilingDatafromBin->is_nddma = 1;
//     tilingDatafromBin->epsilon = 0.01;
//     tilingDatafromBin->avg_factor = 0.0004;
//     tilingDatafromBin->is_gemma = 0;
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);

//     ICPU_SET_TILING_KEY(1030);
//     ICPU_RUN_KF(rms_norm, blockDim, x, gamma, y, rstd, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x);
//     AscendC::GmFree(gamma);
//     AscendC::GmFree(y);
//     AscendC::GmFree(rstd);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

TEST_F(rms_norm_test, test_case_1020)
{
    size_t inputByteSize = 24 * 2560 * sizeof(float);
    size_t gammaByteSize = 2560 * sizeof(float);
    size_t outputByteSize = 24 * 2560 * sizeof(float);
    size_t rstdByteSize = 24 * sizeof(float);
    size_t tiling_data_size = sizeof(RMSNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    RMSNormTilingData* tilingDatafromBin = reinterpret_cast<RMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 24;
    tilingDatafromBin->num_col = 2560;
    tilingDatafromBin->num_col_align = 2560;
    tilingDatafromBin->block_factor = 16;
    tilingDatafromBin->row_factor = 8;
    tilingDatafromBin->ub_factor = 1;
    tilingDatafromBin->reduce_mask = 8;
    tilingDatafromBin->left_num = 512;
    tilingDatafromBin->last_reduce_mask = 0;
    tilingDatafromBin->last_left_num = 0;
    tilingDatafromBin->rstd_size = 2048;
    tilingDatafromBin->ub_loop = 0;
    tilingDatafromBin->col_buffer_length = 0;
    tilingDatafromBin->multi_n_num = 0;
    tilingDatafromBin->is_nddma = 1;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.0004;
    tilingDatafromBin->is_gemma = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(1020);
    ICPU_RUN_KF(rms_norm, blockDim, x, gamma, y, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(rms_norm_test, test_case_1010)
{
    size_t inputByteSize = 24 * 2560 * sizeof(int16_t);
    size_t gammaByteSize = 2560 * sizeof(int16_t);
    size_t outputByteSize = 24 * 2560 * sizeof(int16_t);
    size_t rstdByteSize = 24 * sizeof(float);
    size_t tiling_data_size = sizeof(RMSNormTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    RMSNormTilingData* tilingDatafromBin = reinterpret_cast<RMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 24;
    tilingDatafromBin->num_col = 2560;
    tilingDatafromBin->num_col_align = 2560;
    tilingDatafromBin->block_factor = 16;
    tilingDatafromBin->row_factor = 8;
    tilingDatafromBin->ub_factor = 1;
    tilingDatafromBin->reduce_mask = 8;
    tilingDatafromBin->left_num = 512;
    tilingDatafromBin->last_reduce_mask = 0;
    tilingDatafromBin->last_left_num = 0;
    tilingDatafromBin->rstd_size = 2048;
    tilingDatafromBin->ub_loop = 0;
    tilingDatafromBin->col_buffer_length = 0;
    tilingDatafromBin->multi_n_num = 0;
    tilingDatafromBin->is_nddma = 1;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.0004;
    tilingDatafromBin->is_gemma = 0;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(1010);
    ICPU_RUN_KF(rms_norm, blockDim, x, gamma, y, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}