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
 * \file cube3_op.h
 * \brief
 */
#ifndef _CUBE3_OP_H_
#define _CUBE3_OP_H_

template <typename TYPE>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<TYPE>::Cube3Compute(const AddrInfo &shapeInfo, __gm__ TYPE *left, __gm__ TYPE *right, __gm__ float *out)
{
    ping_pong_flag_l0_b_ = 0;
    ping_pong_flag_l0_b_last = 0;
    int32_t s2Idx = shapeInfo.S2Idx;
    int32_t s1Idx = shapeInfo.S1Idx;
    int32_t kn = shapeInfo.kx;
    int32_t km = shapeInfo.ky;
    int32_t l1_m_size = km;
    int32_t l1_n_size = kn;
    int32_t l1_k_size = headDim;

    int32_t l1_m_size_align = RoundUp(l1_m_size, C0_SIZE);
    int32_t l1_n_size_align = RoundUp(l1_n_size, C0_SIZE);
    int32_t l1_m_block_size_tail = (l1_m_size % 128) == 0 ? 128 : (l1_m_size % 128);
    int32_t l1_n_block_size_tail = (l1_n_size % 128) == 0 ? 128 : (l1_n_size % 128);
    int32_t l1_m_block_size_align_tail = (l1_m_size_align % 128) == 0 ? 128 : (l1_m_size_align % 128);
    int32_t l1_n_block_size_align_tail = (l1_n_size_align % 128) == 0 ? 128 : (l1_n_size_align % 128);


    int32_t m_loop = CeilDiv(km, SIZE_128);
    int32_t n_loop = CeilDiv(kn, SIZE_128);


    __gm__ TYPE *gm_a = left + (shapeInfo.out + globalBlockOffset) * 2;
    __gm__ TYPE *gm_b = right + shapeInfo.left;
    __gm__ float *gm_out = out + shapeInfo.right;

    uint64_t left_offset = gm_a - (__gm__ TYPE *)0;
    uint64_t right_offset = gm_b - (__gm__ TYPE *)0;
    uint64_t out_offset = gm_out - (__gm__ float *)0;

    LocalTensor<TYPE> *l1_b_buf_tensor = ping_pong_flag_l1_b_ ? &l1_b_pong_tensor : &l1_b_ping_tensor;

    WAIT_FLAG(MTE1, MTE2, ping_pong_flag_l1_b_);

    commonNd2NzParams.nValue = l1_m_size;
    commonNd2NzParams.dValue = headDim;
    commonNd2NzParams.dstNzC0Stride = l1_m_size_align;
    commonNd2NzParams.srcDValue = qHeadNum * headDim;
    AscendC::DataCopy((*l1_b_buf_tensor)[0], temp_tensor_bf16[right_offset], commonNd2NzParams);

    SET_FLAG(MTE2, MTE1, ping_pong_flag_l1_b_);
    WAIT_FLAG(MTE2, MTE1, ping_pong_flag_l1_b_);

    uint32_t blockId[16];

    uint32_t count = 0;
    for (uint32_t m_loop_index = 0; m_loop_index < m_loop; m_loop_index += 2) {
        for (uint32_t n_loop_index = 0; n_loop_index < n_loop; n_loop_index++) {
            int32_t loop_limit = (m_loop_index + 2 > m_loop) ? 1 : 2;
            for (int j = 0; j < loop_limit; j++) {
                if (sparseType == 0 || s1Idx + m_loop_index + j >= s2Idx + n_loop_index) {
                    blockId[n_loop_index * m_loop + m_loop_index + j] = count;
                    count += 1;
                } else {
                    blockId[n_loop_index * m_loop + m_loop_index + j] = 17;
                }
            }
        }
    }


    // n: k方向
    // m: q方向
    for (uint32_t n_loop_index = 0; n_loop_index < n_loop; n_loop_index++) {
        // mmad 按m方向累加，Q方向
        for (uint32_t m_loop_index = 0; m_loop_index < m_loop; m_loop_index++) {
            bool is_skip = false;
            if (sparseType != 0 && s2Idx + n_loop_index > s1Idx + m_loop_index) {
                is_skip = true;
                continue;
            }


            LocalTensor<TYPE> *l1_a_buf_tensor = ping_pong_flag_l1_a_ ? &l1_a_pong_tensor : &l1_a_ping_tensor;
            LocalTensor<TYPE> *l0_a_buf_tensor = ping_pong_flag_l0_a_ ? &l0_a_pong_tensor : &l0_a_ping_tensor;
            LocalTensor<TYPE> *l0_b_buf_tensor = (ping_pong_flag_l0_b_last) ? &l0_b_ping_tensor : &l0_b_pong_tensor;
            LocalTensor<float> *l0_c_buf_tensor = ping_pong_flag_l0_c_ ? &l0_c_pong_tensor : &l0_c_ping_tensor;


            if (!is_skip) {
                WAIT_FLAG(M, MTE1, ping_pong_flag_l0_b_last + 2 + FLAG_SHIFT);
                commonLoadData2dParamsTranspose.repeatTimes = headDim / 16;
                commonLoadData2dParamsTranspose.srcStride = l1_m_size_align / 16;

                int32_t m_remain = (m_loop_index == m_loop - 1) ? l1_m_block_size_align_tail : 128;
                int32_t l1_b_buf_offset = 128 * 16 * m_loop_index;

                for (int32_t j = 0; j < m_remain / 16; j++) {
                    AscendC::LoadData((*l0_b_buf_tensor)[j * headDim * 16],
                                      (*l1_b_buf_tensor)[l1_b_buf_offset + j * 16 * 16],
                                      commonLoadData2dParamsTranspose);
                }
                SET_FLAG(MTE1, M, ping_pong_flag_l0_b_last + 2);
                WAIT_FLAG(MTE1, M, ping_pong_flag_l0_b_last + 2);
            }

            uint32_t real_m = (m_loop_index == m_loop - 1) ? l1_m_block_size_tail : 128;
            uint32_t real_n = (n_loop_index == n_loop - 1) ? l1_n_block_size_tail : 128;
            uint32_t real_m_align = (m_loop_index == m_loop - 1) ? l1_m_block_size_align_tail : 128;
            uint32_t real_n_align = (n_loop_index == n_loop - 1) ? l1_n_block_size_align_tail : 128;


            bool init_c = (m_loop_index == 0 || (sparseType != 0 && (s2Idx + n_loop_index == s1Idx + m_loop_index)));
            bool out_c = (m_loop_index == (m_loop - 1));


            if (!is_skip) {
                WAIT_FLAG(MTE1, MTE2, ping_pong_flag_l1_a_ + 2);
                commonNd2NzParams.dValue = real_n;
                commonNd2NzParams.dstNzC0Stride = real_m_align;
                commonNd2NzParams.srcDValue = 128;
                int32_t mUp = (real_m + 1) / 2;
                int32_t mDown = real_m - mUp;

                commonNd2NzParams.nValue = mUp;
                AscendC::DataCopy(*l1_a_buf_tensor,
                                  temp_tensor_bf16[left_offset + 2 * blockId[(n_loop_index * m_loop + m_loop_index)] *
                                                                     SIZE_128 * SIZE_128],
                                  commonNd2NzParams);

                if (mDown > 0) {
                    commonNd2NzParams.nValue = mDown;
                    AscendC::DataCopy(
                        (*l1_a_buf_tensor)[C0_SIZE * mUp],
                        temp_tensor_bf16[left_offset +
                                         2 * blockId[(n_loop_index * m_loop + m_loop_index)] * SIZE_128 * SIZE_128 +
                                         2 * mUp * 128],
                        commonNd2NzParams);
                }

                SET_FLAG(MTE2, MTE1, ping_pong_flag_l1_a_ + 2);
                WAIT_FLAG(MTE2, MTE1, ping_pong_flag_l1_a_ + 2);
            }

            if (!is_skip) {
                WAIT_FLAG(M, MTE1, ping_pong_flag_l0_a_ + FLAG_SHIFT);
                // load left matrix dS(kx, ky) from L1A to L0A
                uint32_t m_c0_loop = (real_m + 15) / 16;
                uint32_t n_c0_loop = (real_n + 15) / 16;
                commonLoadData2dParamsTranspose.repeatTimes = m_c0_loop * n_c0_loop;
                commonLoadData2dParamsTranspose.srcStride = 1;
                AscendC::LoadData((*l0_a_buf_tensor), (*l1_a_buf_tensor), commonLoadData2dParamsTranspose);

                SET_FLAG(MTE1, MTE2, ping_pong_flag_l1_a_ + 2);
            }
            if (!is_skip) {
                SET_FLAG(MTE1, M, ping_pong_flag_l0_a_);
                WAIT_FLAG(MTE1, M, ping_pong_flag_l0_a_);
                int unit_flag = 0b10;

                if (out_c) {
                    unit_flag = 0b11;
                }

                commonMadParams.m = real_n == 1 ? 2 : real_n;
                commonMadParams.n = headDim;
                commonMadParams.k = real_m;
                commonMadParams.unitFlag = unit_flag;
                commonMadParams.cmatrixInitVal = init_c;
                AscendC::Mmad(*l0_c_buf_tensor, *l0_a_buf_tensor, *l0_b_buf_tensor, commonMadParams);
                SET_FLAG(M, MTE1, ping_pong_flag_l0_b_last + 2 + FLAG_SHIFT);

                PipeBarrier<PIPE_M>();
                ping_pong_flag_l0_b_last = 1 - ping_pong_flag_l0_b_last;


                SET_FLAG(M, MTE1, ping_pong_flag_l0_a_ + FLAG_SHIFT);
            }


            if (!is_skip && out_c) {
                commonFixpipeParamsV220.mSize = real_n;
                commonFixpipeParamsV220.nSize = headDim;
                commonFixpipeParamsV220.srcStride = real_n_align;
                commonFixpipeParamsV220.dstStride = headDim * kvHeadNum;
                AscendC::SetAtomicType<float>();
                AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                    temp_tensor_fp32[out_offset + n_loop_index * 128 * kvHeadNum * headDim], *l0_c_buf_tensor,
                    commonFixpipeParamsV220);
                AscendC::SetAtomicNone();
            }

            ping_pong_flag_l1_a_ = 1 - ping_pong_flag_l1_a_;
            ping_pong_flag_l0_a_ = 1 - ping_pong_flag_l0_a_;
        }


        ping_pong_flag_l0_c_ = 1 - ping_pong_flag_l0_c_;
    }
    SET_FLAG(MTE1, MTE2, ping_pong_flag_l1_b_);
    ping_pong_flag_l1_b_ = 1 - ping_pong_flag_l1_b_;
}
#endif