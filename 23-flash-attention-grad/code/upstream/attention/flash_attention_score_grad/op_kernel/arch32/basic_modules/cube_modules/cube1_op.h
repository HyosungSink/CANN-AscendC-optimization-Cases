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
 * \file cube1_op.h
 * \brief
 */
#ifndef _CUBE1_OP_H_
#define _CUBE1_OP_H_

template <typename TYPE>
__aicore__ inline __attribute__((always_inline)) void CubeOp<TYPE>::LoadDataAToL1(LocalTensor<TYPE> dstTensor,
                                                                                  GlobalTensor<TYPE> srcTensor,
                                                                                  const int32_t mSize,
                                                                                  const int32_t kSize)
{
    int32_t mSizeAlign = RoundUp(mSize, C0_SIZE);
    commonNd2NzParams.nValue = mSize;
    commonNd2NzParams.dValue = kSize;
    commonNd2NzParams.srcDValue = kSize * qHeadNum;
    commonNd2NzParams.dstNzC0Stride = mSizeAlign;

    WAIT_FLAG(MTE1, MTE2, ping_pong_flag_l1_a_ + 2);
    AscendC::DataCopy(dstTensor, srcTensor, commonNd2NzParams);
    SET_FLAG(MTE2, MTE1, ping_pong_flag_l1_a_ + 2);
    WAIT_FLAG(MTE2, MTE1, ping_pong_flag_l1_a_ + 2);
}

template <typename TYPE>
__aicore__ inline __attribute__((always_inline)) void CubeOp<TYPE>::LoadDataBToL1(LocalTensor<TYPE> dstTensor,
                                                                                  GlobalTensor<TYPE> srcTensor,
                                                                                  const int32_t nSize,
                                                                                  const int32_t kSize)
{
    int32_t nSizeAlign = RoundUp(nSize, C0_SIZE);
    commonNd2NzParams.nValue = nSize;
    commonNd2NzParams.dValue = kSize;
    commonNd2NzParams.srcDValue = kSize * kvHeadNum;
    commonNd2NzParams.dstNzC0Stride = nSizeAlign;

    WAIT_FLAG(MTE1, MTE2, ping_pong_flag_l1_b_);
    AscendC::DataCopy(dstTensor, srcTensor, commonNd2NzParams);
    SET_FLAG(MTE2, MTE1, ping_pong_flag_l1_b_);
    WAIT_FLAG(MTE2, MTE1, ping_pong_flag_l1_b_);
}

template <typename TYPE>
__aicore__ inline __attribute__((always_inline)) void CubeOp<TYPE>::LoadDataAToL0(LocalTensor<TYPE> dstTensor,
                                                                                  LocalTensor<TYPE> srcTensor,
                                                                                  const int32_t m0,
                                                                                  const int32_t k0,
                                                                                  const int32_t mSize,
                                                                                  const bool skip)
{
    int32_t mSizeAlign = RoundUp(mSize, C0_SIZE);

    WAIT_FLAG(M, MTE1, 3 + ping_pong_flag_l0_a_);
    if (!skip) {
        commonLoadData2dParamsNoTranspose.repeatTimes = k0 / SIZE_16;
        commonLoadData2dParamsNoTranspose.srcStride = mSizeAlign / SIZE_16;
        for (int32_t i = 0; i < m0 / SIZE_16; i++) {
            AscendC::LoadData(dstTensor[i * headDim * SIZE_16], srcTensor[i * SIZE_256],
                              commonLoadData2dParamsNoTranspose);
        }
    }
    SET_FLAG(MTE1, M, ping_pong_flag_l0_a_);
    WAIT_FLAG(MTE1, M, ping_pong_flag_l0_a_);
}

template <typename TYPE>
__aicore__ inline __attribute__((always_inline)) void CubeOp<TYPE>::LoadDataBToL0(LocalTensor<TYPE> dstTensor,
                                                                                  LocalTensor<TYPE> srcTensor,
                                                                                  const int32_t k0,
                                                                                  const int32_t nSize)
{
    int32_t nSizeAlign = RoundUp(nSize, C0_SIZE);

    WAIT_FLAG(M, MTE1, 3 + ping_pong_flag_l0_b_ + 2);

    commonLoadData2dParamsNoTranspose.repeatTimes = nSizeAlign / SIZE_16;
    commonLoadData2dParamsNoTranspose.srcStride = 1;
    for (int i = 0; i < k0 / SIZE_16; i++) {
        AscendC::LoadData(dstTensor[i * nSizeAlign * SIZE_16], srcTensor[i * nSizeAlign * SIZE_16],
                          commonLoadData2dParamsNoTranspose);
    }

    SET_FLAG(MTE1, M, ping_pong_flag_l0_b_ + 2);
    WAIT_FLAG(MTE1, M, ping_pong_flag_l0_b_ + 2);
}

template <typename TYPE>
__aicore__ inline __attribute__((always_inline)) void CubeOp<TYPE>::Cube1Mmad(LocalTensor<float> dstCTensor,
                                                                              LocalTensor<TYPE> srcATensor,
                                                                              LocalTensor<TYPE> srcBTensor,
                                                                              const int32_t m_mad_,
                                                                              const int32_t n_mad_,
                                                                              const bool skip)
{
    if (skip) {
        return;
    }

    uint16_t m_modify = (m_mad_ == 1) ? 2 : m_mad_;
    commonMadParams.m = m_modify;
    commonMadParams.n = n_mad_;
    commonMadParams.k = headDim;
    commonMadParams.unitFlag = 3;
    commonMadParams.cmatrixInitVal = true;
    AscendC::Mmad(dstCTensor, srcATensor, srcBTensor, commonMadParams);
}

template <typename TYPE>
__aicore__ inline __attribute__((always_inline)) void CubeOp<TYPE>::Cube1CopyOut(GlobalTensor<float> dstTensor,
                                                                                 LocalTensor<float> srcTensor,
                                                                                 const int32_t mSize,
                                                                                 const int32_t nSize)
{
    int32_t mSizeAlign = RoundUp(mSize, C0_SIZE);
    commonFixpipeParamsV220.mSize = mSize;
    commonFixpipeParamsV220.nSize = nSize;
    commonFixpipeParamsV220.srcStride = mSizeAlign;
    commonFixpipeParamsV220.dstStride = SIZE_128;
    AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(dstTensor, srcTensor, commonFixpipeParamsV220);
}

template <typename TYPE>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<TYPE>::Cube1Compute(const AddrInfo &shapeInfo, __gm__ TYPE *left, __gm__ TYPE *right, __gm__ float *out)
{
    int32_t cube1Cnt = 0;
    int32_t km = shapeInfo.ky;
    int32_t kn = shapeInfo.kx;
    int32_t nLoop = CeilDiv(kn, SIZE_128);
    uint64_t left_offset = left + shapeInfo.left - (__gm__ TYPE *)0;
    uint64_t right_offset = right + shapeInfo.right - (__gm__ TYPE *)0;
    uint64_t gm_out_offset = out + shapeInfo.out + globalBlockOffset - (__gm__ float *)0;
    bool upperRight = !shapeInfo.upperRight;
    LocalTensor<TYPE> *l1_a_tensor = ping_pong_flag_l1_a_ ? &l1_a_pong_tensor : &l1_a_ping_tensor;
    int32_t mSizeL1 = km;
    int32_t kSizeL1 = headDim;

    LoadDataAToL1(*l1_a_tensor, temp_tensor_bf16[left_offset], mSizeL1, kSizeL1);

    for (int32_t nIdx = 0; nIdx < nLoop; nIdx++) {
        LocalTensor<TYPE> *l1_b_tensor = ping_pong_flag_l1_b_ ? &l1_b_pong_tensor : &l1_b_ping_tensor;
        LocalTensor<TYPE> *l0_b_tensor = ping_pong_flag_l0_b_ ? &l0_b_pong_tensor : &l0_b_ping_tensor;
        int32_t nSizeL1 = (nIdx == nLoop - 1) ? (kn - nIdx * SIZE_128) : SIZE_128;
        bool upper_right_flag = (sparseType != 0 && upperRight && nIdx == nLoop - 1);

        LoadDataBToL1(*l1_b_tensor, temp_tensor_bf16[right_offset + nIdx * SIZE_128 * kSizeL1 * kvHeadNum], nSizeL1,
                      kSizeL1);
        LoadDataBToL0((*l0_b_tensor), (*l1_b_tensor), headDim, nSizeL1);

        for (int32_t mOffset = 0; mOffset < mSizeL1; mOffset += SIZE_128) {
            LocalTensor<TYPE> *l0_a_tensor = ping_pong_flag_l0_a_ ? &l0_a_pong_tensor : &l0_a_ping_tensor;
            LocalTensor<float> *l0_c_tensor = ping_pong_flag_l0_c_ ? &l0_c_pong_tensor : &l0_c_ping_tensor;
            int32_t mSizeL0 = Min((mSizeL1 - mOffset), SIZE_128);
            bool l0_skip_flag = (upper_right_flag && mOffset == 0);

            LoadDataAToL0((*l0_a_tensor), (*l1_a_tensor)[mOffset * SIZE_16], SIZE_128, headDim, mSizeL1, l0_skip_flag);
            Cube1Mmad(*l0_c_tensor, *l0_a_tensor, *l0_b_tensor, mSizeL0, nSizeL1, l0_skip_flag);
            SET_FLAG(M, MTE1, 3 + ping_pong_flag_l0_a_);
            if (!l0_skip_flag) {
                Cube1CopyOut(temp_tensor_fp32[gm_out_offset + cube1Cnt * SIZE_LONG_BLOCK], *l0_c_tensor, mSizeL0,
                             nSizeL1);
                cube1Cnt++;
            }
            ping_pong_flag_l0_c_ = 1 - ping_pong_flag_l0_c_;
            ping_pong_flag_l0_a_ = 1 - ping_pong_flag_l0_a_;
        }

        SET_FLAG(M, MTE1, 3 + ping_pong_flag_l0_b_ + 2);
        SET_FLAG(MTE1, MTE2, ping_pong_flag_l1_b_);
        ping_pong_flag_l0_b_ = 1 - ping_pong_flag_l0_b_;
        ping_pong_flag_l1_b_ = 1 - ping_pong_flag_l1_b_;
    }

    SET_FLAG(MTE1, MTE2, ping_pong_flag_l1_a_ + 2);
    ping_pong_flag_l1_a_ = 1 - ping_pong_flag_l1_a_;
}
#endif