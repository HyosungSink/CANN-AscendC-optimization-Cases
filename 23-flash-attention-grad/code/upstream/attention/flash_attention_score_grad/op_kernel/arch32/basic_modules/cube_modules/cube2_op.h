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
 * \file cube2_op.h
 * \brief
 */
#ifndef _CUBE2_OP_H_
#define _CUBE2_OP_H_

template <typename TYPE>
__aicore__ inline __attribute__((always_inline)) void CubeOp<TYPE>::Cube2Compute(const AddrInfo &shapeInfo,
                                                                                 __gm__ TYPE *left, __gm__ TYPE *right,
                                                                                 __gm__ float *out, const bool isFisrt)
{
    // pingpongl1a/pingpongl0a 在1 * n情况下，恒为0，在2 * n情况下，0-1跳动
    // pingpongl1b/pingpongl0b 在1 * n情况和2 * n下，随n进行n % 2
    __gm__ TYPE *left_start_addr = left + (shapeInfo.out + globalBlockOffset) * 2;
    __gm__ TYPE *right_start_addr = right + shapeInfo.right;
    __gm__ float *result_gm = out + shapeInfo.left;

    uint64_t left_offset = left_start_addr - (__gm__ TYPE *)0;
    uint64_t right_offset = right_start_addr - (__gm__ TYPE *)0;
    uint64_t out_offset = result_gm - (__gm__ float *)0;

    // left matrix is (ky, kx)
    // right matrix is (kx, headDim)
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
    bool upperRight = !shapeInfo.upperRight && sparseType != 0;
    int32_t startS1Id = shapeInfo.S1Idx;
    int32_t startS2Id = shapeInfo.S2Idx;


    int32_t skip_num = 0;
    for (uint32_t n_loop_index = 0; n_loop_index < n_loop; n_loop_index++) {
        int32_t n_remain = (n_loop_index == n_loop - 1) ? l1_n_block_size_tail : 128;
        int32_t n_remain_align = (n_loop_index == n_loop - 1) ? l1_n_block_size_align_tail : 128;
        bool l0_c_init_flag = (n_loop_index == 0);

        // load right matrix gm (kx, headDim)-> L1B
        AscendC::LocalTensor<TYPE> *l1_b_buf_tensor = ping_pong_flag_l1_b_ ? &l1_b_pong_tensor : &l1_b_ping_tensor;
        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(static_cast<int32_t>(ping_pong_flag_l1_b_));
        commonNd2NzParams.nValue = n_remain;
        commonNd2NzParams.dValue = headDim;
        commonNd2NzParams.srcDValue = kvHeadNum * headDim;
        ;
        commonNd2NzParams.dstNzC0Stride = n_remain_align;
        AscendC::DataCopy(*l1_b_buf_tensor, temp_tensor_bf16[right_offset + n_loop_index * kvHeadNum * 128 * headDim],
                          commonNd2NzParams);
        AscendC::SetFlag<HardEvent::MTE2_MTE1>(static_cast<int32_t>(ping_pong_flag_l1_b_));

        if (n_loop_index == 0 && isFisrt) {
            AscendC::WaitEvent(VEC2CUBE);
        }

        // Load Left_GM matrix A -> L1A
        ping_pong_flag_l1_a_ = 0;
        for (uint32_t m_loop_index = 0; m_loop_index < m_loop; m_loop_index++) {
            AscendC::LocalTensor<TYPE> *l1_a_buf_tensor = ping_pong_flag_l1_a_ ? &l1_a_pong_tensor : &l1_a_ping_tensor;
            int32_t m_remain = (m_loop_index == m_loop - 1) ? l1_m_block_size_tail : 128;
            int32_t m_remain_align = (m_loop_index == m_loop - 1) ? l1_m_block_size_align_tail : 128;
            bool is_skip = false;

            if (n_loop_index == n_loop - 1 && m_loop_index == 0 && upperRight) {
                skip_num++;
                is_skip = true;
            }
            AscendC::WaitFlag<HardEvent::MTE1_MTE2>(static_cast<int32_t>(ping_pong_flag_l1_a_ + 2));
            if (!is_skip) {
                commonNd2NzParams.dValue = n_remain;
                commonNd2NzParams.dstNzC0Stride = m_remain_align;
                commonNd2NzParams.srcDValue = 128;
                int32_t mUp = (m_remain + 1) / 2;
                int32_t mDown = m_remain - mUp;

                commonNd2NzParams.nValue = mUp;
                AscendC::DataCopy(
                    *l1_a_buf_tensor,
                    temp_tensor_bf16[left_offset + 2 * (m_loop * n_loop_index + m_loop_index - skip_num) * 128 * 128],
                    commonNd2NzParams);

                if (mDown > 0) {
                    commonNd2NzParams.nValue = mDown;
                    AscendC::DataCopy(
                        (*l1_a_buf_tensor)[mUp * C0_SIZE],
                        temp_tensor_bf16[left_offset +
                                         2 * (m_loop * n_loop_index + m_loop_index - skip_num) * 128 * 128 +
                                         2 * mUp * 128],
                        commonNd2NzParams);
                }
            }
            AscendC::SetFlag<HardEvent::MTE2_MTE1>(static_cast<int32_t>(ping_pong_flag_l1_a_ + 2));
            ping_pong_flag_l1_a_ = 1 - ping_pong_flag_l1_a_;
        }


        // load L1B (n, headDim) -> L0B
        AscendC::LocalTensor<TYPE> *l0_b_buf_tensor = ping_pong_flag_l0_b_ ? &l0_b_pong_tensor : &l0_b_ping_tensor;
        commonLoadData2dParamsTranspose.repeatTimes = headDim / 16;
        commonLoadData2dParamsTranspose.srcStride = n_remain_align / 16;
        AscendC::WaitFlag<HardEvent::MTE2_MTE1>(static_cast<int32_t>(ping_pong_flag_l1_b_));
        AscendC::WaitFlag<HardEvent::M_MTE1>(static_cast<int32_t>(ping_pong_flag_l0_b_ + 2 + FLAG_SHIFT));
        for (int32_t i = 0; i < n_remain_align / 16; i++) {
            AscendC::LoadData((*l0_b_buf_tensor)[i * headDim * 16], (*l1_b_buf_tensor)[i * 16 * 16],
                              commonLoadData2dParamsTranspose);
        }
        AscendC::SetFlag<HardEvent::MTE1_M>(static_cast<int32_t>(ping_pong_flag_l0_b_ + 2));
        AscendC::SetFlag<HardEvent::MTE1_MTE2>(static_cast<int32_t>(ping_pong_flag_l1_b_));

        ping_pong_flag_l1_a_ = 0;
        ping_pong_flag_l0_a_ = 0;
        AscendC::WaitFlag<HardEvent::MTE1_M>(static_cast<int32_t>(ping_pong_flag_l0_b_ + 2));
        // do m_loop times mad with l0B常驻
        for (uint32_t m_loop_index = 0; m_loop_index < m_loop; m_loop_index++) {
            AscendC::LocalTensor<TYPE> *l1_a_buf_tensor = ping_pong_flag_l1_a_ ? &l1_a_pong_tensor : &l1_a_ping_tensor;
            AscendC::LocalTensor<TYPE> *l0_a_buf_tensor = ping_pong_flag_l0_a_ ? &l0_a_pong_tensor : &l0_a_ping_tensor;
            AscendC::LocalTensor<float> *l0_c_buf_tensor = m_loop_index ? &l0_c_pong_tensor : &l0_c_ping_tensor;

            int32_t m_remain = (m_loop_index == m_loop - 1) ? l1_m_block_size_tail : 128;
            int32_t m_remain_align = (m_loop_index == m_loop - 1) ? l1_m_block_size_align_tail : 128;
            bool is_skip = false;

            if (n_loop_index == n_loop - 1 && m_loop_index == 0 && upperRight) {
                is_skip = true;
            }
            AscendC::WaitFlag<HardEvent::MTE2_MTE1>(static_cast<int32_t>(ping_pong_flag_l1_a_ + 2));
            AscendC::WaitFlag<HardEvent::M_MTE1>(static_cast<int32_t>(ping_pong_flag_l0_a_ + FLAG_SHIFT));
            if (!is_skip) {
                commonLoadData2dParamsNoTranspose.repeatTimes = n_remain_align / SIZE_16;
                commonLoadData2dParamsNoTranspose.srcStride = m_remain_align / SIZE_16;
                for (int32_t i = 0; i < m_remain_align / SIZE_16; i++) {
                    AscendC::LoadData((*l0_a_buf_tensor)[i * n_remain_align * 16], (*l1_a_buf_tensor)[i * 16 * 16],
                                      commonLoadData2dParamsNoTranspose);
                }
            }
            AscendC::SetFlag<HardEvent::MTE1_M>(static_cast<int32_t>(ping_pong_flag_l0_a_));
            AscendC::SetFlag<HardEvent::MTE1_MTE2>(static_cast<int32_t>(ping_pong_flag_l1_a_ + 2));
            bool last_k = false;
            last_k = (m_loop_index == 0 && upperRight) ? n_loop_index == n_loop - 2 : n_loop_index == n_loop - 1;

            AscendC::WaitFlag<HardEvent::MTE1_M>(static_cast<int32_t>(ping_pong_flag_l0_a_));
            if (!is_skip) {
                uint16_t m_modify = (m_remain == 1) ? 2 : m_remain;


                AscendC::Mmad(*l0_c_buf_tensor, *l0_a_buf_tensor, *l0_b_buf_tensor,
                              AscendC::MmadParams(m_modify, headDim, n_remain, last_k ? 3 : 2,
                                                  // unit_flag,
                                                  false, l0_c_init_flag));
            }
            AscendC::SetFlag<HardEvent::M_MTE1>(static_cast<int32_t>(ping_pong_flag_l0_a_ + FLAG_SHIFT));

            // fixp in n_loop tail block
            if (!is_skip && last_k) {
                commonFixpipeParamsV220.mSize = m_remain;
                commonFixpipeParamsV220.nSize = headDim;
                commonFixpipeParamsV220.srcStride = m_remain_align;
                commonFixpipeParamsV220.dstStride = qHeadNum * headDim;
                AscendC::SetAtomicType<float>();
                AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                    temp_tensor_fp32[out_offset + m_loop_index * qHeadNum * 128 * headDim], *l0_c_buf_tensor,
                    commonFixpipeParamsV220);
                AscendC::SetAtomicNone();
            }
            ping_pong_flag_l1_a_ = 1 - ping_pong_flag_l1_a_;
            ping_pong_flag_l0_a_ = 1 - ping_pong_flag_l0_a_;
        }
        AscendC::SetFlag<HardEvent::M_MTE1>(static_cast<int32_t>(ping_pong_flag_l0_b_ + 2 + FLAG_SHIFT));

        ping_pong_flag_l0_b_ = 1 - ping_pong_flag_l0_b_;
        ping_pong_flag_l1_b_ = 1 - ping_pong_flag_l1_b_;
    }
}
#endif