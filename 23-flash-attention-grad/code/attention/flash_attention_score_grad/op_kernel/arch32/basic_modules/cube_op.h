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
 * \file cube_op.h
 * \brief
 */


#ifndef _CUBE_OP_H_
#define _CUBE_OP_H_

#include "kernel_operator.h"
#include "common_header.h"

using namespace AscendC;

namespace CUBE_OP {
template <typename TYPE>
class CubeOp {
public:
    /* clang-format off */
    __aicore__ inline CubeOp(){};
    __aicore__ inline void Init(TPipe *pipe, 
                                uint64_t qHeadNumIn, 
                                uint64_t kvHeadNumIn, 
                                uint64_t headDimIn, 
                                uint64_t sparseMode);
    __aicore__ inline __attribute__((always_inline)) void Cube1Process(const CubeAddrInfo &addrs,
                                                                        __gm__ TYPE *left, 
                                                                        __gm__ TYPE *right, 
                                                                        __gm__ float *out);

    __aicore__ inline __attribute__((always_inline)) void Cube2Process(const CubeAddrInfo &addrs,
                                                                       __gm__ TYPE *left, 
                                                                       __gm__ TYPE *right, 
                                                                       __gm__ float *out);

    __aicore__ inline __attribute__((always_inline)) void Cube3Process(const CubeAddrInfo &addrs,
                                                                       __gm__ TYPE *left, 
                                                                       __gm__ TYPE *right, 
                                                                       __gm__ float *out);

    __aicore__ inline __attribute__((always_inline)) void Cube23Process(const CubeAddrInfo &addrs, 
                                                                        __gm__ TYPE *left, 
                                                                        __gm__ TYPE *right1, 
                                                                        __gm__ TYPE *right2, 
                                                                        __gm__ float *out1, 
                                                                        __gm__ float *out2);

private:
    __aicore__ inline __attribute__((always_inline)) void Cube1Compute(const AddrInfo &shapeInfo, 
                                                                       __gm__ TYPE *left, 
                                                                       __gm__ TYPE *right, 
                                                                       __gm__ float *out);
    __aicore__ inline __attribute__((always_inline)) void Cube2Compute(const AddrInfo &shapeInfo, 
                                                                       __gm__ TYPE *left, 
                                                                       __gm__ TYPE *right, 
                                                                       __gm__ float *out, 
                                                                       const bool isFirst);
    __aicore__ inline __attribute__((always_inline)) void Cube3Compute(const AddrInfo &shapeInfo, 
                                                                       __gm__ TYPE *left, 
                                                                       __gm__ TYPE *right, 
                                                                       __gm__ float *out);

    __aicore__ inline __attribute__((always_inline)) void Cube23Compute(const AddrInfo &shapeInfo, 
                                                                        __gm__ TYPE *left, 
                                                                        __gm__ TYPE *right1,  
                                                                        __gm__ TYPE *right2, 
                                                                        __gm__ float *out1, 
                                                                        __gm__ float *out2);
                                                                        
    __aicore__ inline __attribute__((always_inline)) void LoadDataAToL1(LocalTensor<TYPE> dstTensor, 
                                                                        GlobalTensor<TYPE> srcTensor, 
                                                                        const int32_t mSize, 
                                                                        const int32_t kSize);
    __aicore__ inline __attribute__((always_inline)) void LoadDataBToL1(LocalTensor<TYPE> dstTensor, 
                                                                        GlobalTensor<TYPE> srcTensor, 
                                                                        const int32_t nSize, 
                                                                        const int32_t kSize);
    __aicore__ inline __attribute__((always_inline)) void LoadDataAToL0(LocalTensor<TYPE> dstTensor,
                                                                        LocalTensor<TYPE> srcTensor,
                                                                        const int32_t m0,
                                                                        const int32_t k0,
                                                                        const int32_t mSize,
                                                                        const bool skip);
    __aicore__ inline __attribute__((always_inline)) void LoadDataBToL0(LocalTensor<TYPE> dstTensor,
                                                                        LocalTensor<TYPE> srcTensor,
                                                                        const int32_t k0,
                                                                        const int32_t nSize);
    __aicore__ inline __attribute__((always_inline)) void Cube1Mmad(LocalTensor<float> dstCTensor,
                                                                    LocalTensor<TYPE> srcATensor,
                                                                    LocalTensor<TYPE> srcBTensor,
                                                                    const int32_t m_mad_,
                                                                    const int32_t n_mad_,
                                                                    const bool skip);
    __aicore__ inline __attribute__((always_inline)) void Cube1CopyOut(GlobalTensor<float> dstTensor, 
                                                                       LocalTensor<float> srcTensor, 
                                                                       const int32_t mSize, 
                                                                       const int32_t nSize);
    AscendC::Nd2NzParams commonNd2NzParamsFp32_ {
        1,
        128,
        BASE_BLOCK_LENGTH,
        0,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        1,
        0
    };
    AscendC::MmadParams cube3MadParams {
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        3,
        false,
        true
    };
    AscendC::FixpipeParamsV220 cube3FixpipeParamsV220 {
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        false
    };
    AscendC::Nd2NzParams commonNd2NzParams {
        1,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        0,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        1,
        0
    };
    AscendC::LoadData2dParams commonLoadData2dParamsTranspose {
        0,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        0,
        0,
        true,
        0
    };
    AscendC::LoadData2dParams commonLoadData2dParamsNoTranspose {
        0,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        0,
        0,
        false,
        0
    };
    AscendC::MmadParams commonMadParams {
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        3,
        false,
        true
    };
    AscendC::FixpipeParamsV220 commonFixpipeParamsV220 {
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        false
    };

    /* clang-format on */
    TBuf<AscendC::TPosition::A1> L1Buffer;
    TBuf<AscendC::TPosition::CO1> L0CBuffer;
    AsdopsBuffer<ArchType::ASCEND_V220> asdopsBuf;

    constexpr static int32_t BASE_BLOCK_LENGTH = 128;
    constexpr static int32_t BLOCK_WORKSPACE = 16 * 128 * 128;
    constexpr static int32_t FLAG_SHIFT = 3;
    constexpr static int32_t C0_SIZE = 16;
    constexpr static int32_t SIZE_16 = 16;
    constexpr static int32_t SIZE_32 = 32;
    constexpr static int32_t SIZE_64 = 64;
    constexpr static int32_t SIZE_96 = 96;
    constexpr static int32_t SIZE_128 = 128;
    constexpr static int32_t SIZE_256 = 256;
    constexpr static int32_t SIZE_320 = 320;
    constexpr static int32_t SIZE_384 = 384;
    constexpr static int32_t SIZE_448 = 448;
    constexpr static int32_t SIZE_512 = 512;
    constexpr static int32_t SIZE_ONE_K = 1024;
    constexpr static int32_t SIZE_LONG_BLOCK = 128 * 128;

    GlobalTensor<TYPE> temp_tensor_bf16;
    GlobalTensor<float> temp_tensor_fp32;
    // L1 tensor
    LocalTensor<TYPE> l1_a_ping_tensor;
    LocalTensor<TYPE> l1_a_pong_tensor;
    LocalTensor<TYPE> l1_b_ping_tensor;
    LocalTensor<TYPE> l1_b_pong_tensor;

    // extra l1_bx_buf_tensor
    LocalTensor<TYPE> l1_b1_ping_tensor;
    LocalTensor<TYPE> l1_b1_pong_tensor;
    LocalTensor<TYPE> l1_b2_ping_tensor;
    LocalTensor<TYPE> l1_b2_pong_tensor;
    
    // L0 tensor
    LocalTensor<TYPE> l0_a_ping_tensor;
    LocalTensor<TYPE> l0_a_pong_tensor;
    LocalTensor<TYPE> l0_b_ping_tensor;
    LocalTensor<TYPE> l0_b_pong_tensor;
    LocalTensor<float> l0_c_ping_tensor;
    LocalTensor<float> l0_c_pong_tensor;

    // extra l0_c_buf_tensor
    LocalTensor<float> l0_c1_ping_tensor;
    LocalTensor<float> l0_c1_pong_tensor;
    LocalTensor<float> l0_c2_ping_tensor;
    LocalTensor<float> l0_c2_pong_tensor;
    
    // ping pong flag
    uint32_t ping_pong_flag_l1_a_ = 0;
    uint32_t ping_pong_flag_l1_b_ = 0;
    uint32_t ping_pong_flag_l0_a_ = 0;
    uint32_t ping_pong_flag_l0_b_ = 0;
    uint32_t ping_pong_flag_l0_c_ = 0;
    uint32_t ping_pong_flag_l0_b_last = 0;
    uint32_t pingPongIdx = 0;
    uint64_t globalBlockOffset = 0;

    uint64_t qHeadNum;
    uint64_t kvHeadNum;
    uint64_t headDim;
    uint64_t sparseType;
};

template <typename TYPE>
__aicore__ inline void
CubeOp<TYPE>::Init(TPipe *pipe, uint64_t qHeadNumIn, uint64_t kvHeadNumIn, uint64_t headDimIn, uint64_t sparseMode)
{
    pipe->InitBuffer(L1Buffer, HardwareInfo<ArchType::ASCEND_V220>::l1Size);
    pipe->InitBuffer(L0CBuffer, HardwareInfo<ArchType::ASCEND_V220>::l0CSize);

    qHeadNum = qHeadNumIn;
    kvHeadNum = kvHeadNumIn;
    headDim = headDimIn;
    sparseType = sparseMode;
    globalBlockOffset = GetBlockIdx() * 16 * 128 * 128;

    temp_tensor_bf16.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(0));
    temp_tensor_fp32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(0));

    AscendC::SetLoadDataPaddingValue<uint64_t>(0);
    AscendC::SetNdParaImpl(0x1);

    // init L1 tensor
    l1_a_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(0);
    l1_a_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_128 * SIZE_ONE_K);
    l1_b_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_256 * SIZE_ONE_K);
    l1_b_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_384 * SIZE_ONE_K);
    // init L0A/L0B/L0C tensor
    l0_a_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, TYPE>(0);
    l0_a_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, TYPE>(SIZE_32 * SIZE_ONE_K);
    l0_b_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, TYPE>(0);
    l0_b_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, TYPE>(SIZE_32 * SIZE_ONE_K);
    l0_c_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(0);
    l0_c_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(SIZE_64 * SIZE_ONE_K);

    // init 64 extra l1_b_buf_tensor
    l1_b1_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_256 * SIZE_ONE_K);
    l1_b1_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_320 * SIZE_ONE_K);
    l1_b2_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_384 * SIZE_ONE_K);
    l1_b2_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_448 * SIZE_ONE_K);

    // init 64 L0C tensor
    l0_c1_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(0);
    l0_c1_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(SIZE_32 * SIZE_ONE_K);
    l0_c2_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(SIZE_64 * SIZE_ONE_K);
    l0_c2_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(SIZE_96 * SIZE_ONE_K);

    commonFixpipeParamsV220.quantPre = QuantMode_t::NoQuant;
    commonFixpipeParamsV220.unitFlag = 3;
}

template <typename TYPE>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<TYPE>::Cube1Process(const CubeAddrInfo &addrs, __gm__ TYPE *left, __gm__ TYPE *right, __gm__ float *out)
{
    pingPongIdx = addrs.taskId % 2;
    globalBlockOffset = GetBlockIdx() * BLOCK_WORKSPACE * 2 + pingPongIdx * BLOCK_WORKSPACE;

    for (uint32_t i = 0; i < addrs.blockLength; ++i) {
        Cube1Compute(addrs.addrInfo[i], left, right, out);
    }
}

template <typename TYPE>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<TYPE>::Cube2Process(const CubeAddrInfo &addrs, __gm__ TYPE *left, __gm__ TYPE *right, __gm__ float *out)
{
    pingPongIdx = addrs.taskId % 2;
    globalBlockOffset = GetBlockIdx() * BLOCK_WORKSPACE * 2 + pingPongIdx * BLOCK_WORKSPACE;

    for (uint32_t i = 0; i < addrs.blockLength; ++i) {
        Cube2Compute(addrs.addrInfo[i], left, right, out, i == 0);
    }
}

template <typename TYPE>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<TYPE>::Cube3Process(const CubeAddrInfo &addrs, __gm__ TYPE *left, __gm__ TYPE *right, __gm__ float *out)
{
    pingPongIdx = addrs.taskId % 2;
    globalBlockOffset = GetBlockIdx() * BLOCK_WORKSPACE * 2 + pingPongIdx * BLOCK_WORKSPACE;

    for (uint32_t i = 0; i < addrs.blockLength; ++i) {
        Cube3Compute(addrs.addrInfo[i], left, right, out);
    }
}

template <typename TYPE>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<TYPE>::Cube23Process(const CubeAddrInfo &addrs, __gm__ TYPE *left, __gm__ TYPE *right1, __gm__ TYPE *right2, __gm__ float *out1, __gm__ float *out2)
{
    pingPongIdx = addrs.taskId % 2;
    globalBlockOffset = GetBlockIdx() * BLOCK_WORKSPACE * 2 + pingPongIdx * BLOCK_WORKSPACE;

    for (uint32_t i = 0; i < addrs.blockLength; ++i) {
        Cube23Compute(addrs.addrInfo[i], left, right1, right2, out1, out2);
    }
}

#include "cube_modules/cube1_op.h"
#include "cube_modules/cube2_op.h"
#include "cube_modules/cube3_op.h"
#include "cube_modules/cube23_d64_op.h"

} // namespace CUBE_OP
#endif