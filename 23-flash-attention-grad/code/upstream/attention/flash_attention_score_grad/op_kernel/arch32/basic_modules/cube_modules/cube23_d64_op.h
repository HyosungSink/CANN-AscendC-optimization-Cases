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
 * \file cube23_d64_op.h
 * \brief
 */
#ifndef _CUBE23_D64_OP_H_
#define _CUBE23_D64_OP_H_

// sync flag:
// M->MTE1 eventId flag begin from 3 for the reason of tpipe init
// MTE1->MTE2 eventId flag begin from 0
template <typename TYPE>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<TYPE>::Cube23Compute(const AddrInfo &shapeInfo, __gm__ TYPE *left, __gm__ TYPE *right1,  __gm__ TYPE *right2, __gm__ float *out1, __gm__ float *out2)
{
    __gm__ TYPE *gm_matrixA = left + (shapeInfo.out + globalBlockOffset) * 2;
    __gm__ TYPE *gm_matrixB1 = right1 + shapeInfo.right;
    __gm__ float *gm_matrixC1 = out1 + shapeInfo.left;
    __gm__ TYPE *gm_matrixB2 = right2 + shapeInfo.left;
    __gm__ float *gm_matrixC2 = out2 + shapeInfo.right;

    uint64_t matrixA_offset = gm_matrixA - (__gm__ TYPE *)0;
    uint64_t matrixB1_offset = gm_matrixB1 - (__gm__ TYPE *)0;
    uint64_t matrixC1_offset = gm_matrixC1 - (__gm__ float *)0;
    uint64_t matrixB2_offset = gm_matrixB2 - (__gm__ TYPE *)0;
    uint64_t matrixC2_offset = gm_matrixC2 - (__gm__ float *)0;

    // pingpong happen between dQ and dK

    int32_t L1A_matrix_pingpong_flag = 0;
    int32_t L1B1_matrix_pingpong_flag = 0;
    int32_t L1B2_matrix_pingpong_flag = 0;

    int32_t L1A_FLAG_SHIFT = 0;
    int32_t L1B1_FLAG_SHIFT = 2;
    int32_t L1B2_FLAG_SHIFT = 4;

    // L0A/B pingpong flag is bound
    int32_t L0A_matrix_ping_flag = 3;
    int32_t L0A_matrix_pong_flag = 4;
    int32_t L0B_matrix_ping_flag = 5;
    int32_t L0B_matrix_pong_flag = 6;
    int32_t L0C_matrixC1_pingpong_flag = 0;
    int32_t L0C_matrixC2_pingpong_flag = 0;

    // assume no mask condition
    // cube2:
    // left matrix is matrixA(ky, kx)
    // right matrix is matrixB1(kx, 64)
    
    // cube3:
    // left matrix is matrixA^T(kx, ky)
    // right matrix is matrixB2(ky, 64)

    int32_t km = shapeInfo.ky;
    int32_t kn = shapeInfo.kx;
    int32_t km_align = (km + C0_SIZE - 1) / C0_SIZE * C0_SIZE;
    int32_t kn_align = (kn + C0_SIZE - 1) / C0_SIZE * C0_SIZE;
    int32_t km_tail = (km % BASE_BLOCK_LENGTH) == 0 ? BASE_BLOCK_LENGTH : (km % BASE_BLOCK_LENGTH);
    int32_t kn_tail = (kn % BASE_BLOCK_LENGTH) == 0 ? BASE_BLOCK_LENGTH : (kn % BASE_BLOCK_LENGTH);
    int32_t km_align_tail = (km_tail + C0_SIZE - 1) / C0_SIZE * C0_SIZE;
    int32_t kn_align_tail = (kn_tail + C0_SIZE - 1) / C0_SIZE * C0_SIZE;
    
    int32_t s1_loop = CeilDiv(km, BASE_BLOCK_LENGTH);
    int32_t s2_loop = CeilDiv(kn, BASE_BLOCK_LENGTH);

    int32_t outer_loop = (s1_loop + 1) / 2;
    int32_t outer_tail = s1_loop % 2;

    // s1_loop
    for (int32_t outer_index = 0; outer_index < outer_loop; outer_index++) {
        int32_t start_s1Idx = outer_index * 2;
        int32_t inner_s1Loop = (outer_index == outer_loop - 1) && (outer_tail != 0) ? outer_tail : 2;

        // load full matrixB2 from gm -> L1 ahead
        // assume l1_b2_buf_tensor stores the whole matrixB2
        WAIT_FLAG(MTE1, MTE2, L1B2_matrix_pingpong_flag + L1B2_FLAG_SHIFT);
        LocalTensor<TYPE> *l1_b2_buf_tensor = L1B2_matrix_pingpong_flag ? &l1_b2_pong_tensor : &l1_b2_ping_tensor;
        commonNd2NzParams.nValue = km;
        commonNd2NzParams.dValue = headDim;
        commonNd2NzParams.dstNzC0Stride = km_align;
        commonNd2NzParams.srcDValue = qHeadNum * headDim;
        AscendC::DataCopy(
            (*l1_b2_buf_tensor)[0],
            temp_tensor_bf16[matrixB2_offset], 
            commonNd2NzParams
        );
        SET_FLAG(MTE2, MTE1, L1B2_matrix_pingpong_flag + L1B2_FLAG_SHIFT);

        // The calculation of dQ is isolated from the calculation of dK.
        // So the ping flag bound with dQ, the pong flag bound with dK
        for (int32_t s2_idx = 0; s2_idx < s2_loop; s2_idx++) {
            // L0C matrixC2 pingpong flag bound with s2_idx
            L0C_matrixC2_pingpong_flag = s2_idx % 2;

            for (int32_t s1_idx = 0; s1_idx < inner_s1Loop; s1_idx++) {
                // L0C matrixC1 pingpong flag bound with s1_idx
                L0C_matrixC1_pingpong_flag = s1_idx % 2;

                int32_t row_start = (start_s1Idx + s1_idx) * BASE_BLOCK_LENGTH;
                int32_t col_start = s2_idx * BASE_BLOCK_LENGTH;

                int32_t s1_block_length = (s1_idx == inner_s1Loop - 1) ? km_tail : BASE_BLOCK_LENGTH;
                int32_t s2_block_length = (s2_idx == s2_loop - 1) ? kn_tail : BASE_BLOCK_LENGTH;
                int32_t s1_block_align_length = (s1_block_length + C0_SIZE -  1) / C0_SIZE * C0_SIZE;
                int32_t s2_block_align_length = (s2_block_length + C0_SIZE -  1) / C0_SIZE * C0_SIZE;

                ///////////////////////////////// dQ /////////////////////////
                
                // load matrixA from gm -> L1
                // dS in workspace is divided by two part in s1 axis
                WAIT_FLAG(MTE1, MTE2, L1A_matrix_pingpong_flag + L1A_FLAG_SHIFT);
                AscendC::LocalTensor<TYPE> *l1_a_buf_tensor = L1A_matrix_pingpong_flag ? &l1_a_pong_tensor : &l1_a_ping_tensor;
                int32_t s1_block_up = (s1_block_length + 1) / 2;
                int32_t s1_block_down = s1_block_length - s1_block_up;
                commonNd2NzParams.dValue = s2_block_length;
                commonNd2NzParams.dstNzC0Stride = s1_block_align_length;
                commonNd2NzParams.srcDValue = BASE_BLOCK_LENGTH;
                commonNd2NzParams.nValue = s1_block_up;
                AscendC::DataCopy(
                    *l1_a_buf_tensor,
                    temp_tensor_bf16[matrixA_offset + 2 * (s2_idx * inner_s1Loop + s1_idx) * BASE_BLOCK_LENGTH * BASE_BLOCK_LENGTH],
                    commonNd2NzParams
                );
                if (s1_block_down > 0) {
                    commonNd2NzParams.nValue = s1_block_down;
                    AscendC::DataCopy(
                        (*l1_a_buf_tensor)[s1_block_up * C0_SIZE],
                        temp_tensor_bf16[matrixA_offset + 2 * (s2_idx * inner_s1Loop + s1_idx) * BASE_BLOCK_LENGTH * BASE_BLOCK_LENGTH +
                                         2 * s1_block_up * BASE_BLOCK_LENGTH],
                        commonNd2NzParams
                    );
                }
                SET_FLAG(MTE2, MTE1, L1A_matrix_pingpong_flag + L1A_FLAG_SHIFT);

                // load matrixB1 from gm -> L1
                WAIT_FLAG(MTE1, MTE2, L1B1_matrix_pingpong_flag + L1B1_FLAG_SHIFT);
                LocalTensor<TYPE> *l1_b1_buf_tensor = L1B1_matrix_pingpong_flag ? &l1_b1_pong_tensor : &l1_b1_ping_tensor;
                commonNd2NzParams.nValue = s2_block_length;
                commonNd2NzParams.dValue = headDim;
                commonNd2NzParams.srcDValue = kvHeadNum * headDim;
                commonNd2NzParams.dstNzC0Stride = s2_block_align_length;
                AscendC::DataCopy(
                    *l1_b1_buf_tensor, 
                    temp_tensor_bf16[matrixB1_offset + s2_idx * BASE_BLOCK_LENGTH * kvHeadNum * headDim],
                    commonNd2NzParams
                );
                SET_FLAG(MTE2, MTE1, L1B1_matrix_pingpong_flag + L1B1_FLAG_SHIFT);

                // load matrixA from L1 -> L0A(ping) l0_a_ping_tensor
                WAIT_FLAG(MTE2, MTE1, L1A_matrix_pingpong_flag + L1A_FLAG_SHIFT);
                WAIT_FLAG(M, MTE1, L0A_matrix_ping_flag);
                commonLoadData2dParamsNoTranspose.repeatTimes = s2_block_align_length / C0_SIZE;
                commonLoadData2dParamsNoTranspose.srcStride = s1_block_align_length / C0_SIZE;
                for (int32_t i = 0; i < s1_block_align_length / C0_SIZE; i++) {
                    AscendC::LoadData(
                        (l0_a_ping_tensor)[i * s2_block_align_length * C0_SIZE], 
                        (*l1_a_buf_tensor)[i * C0_SIZE * C0_SIZE],
                        commonLoadData2dParamsNoTranspose
                    );
                }
                SET_FLAG(MTE1, M, L0A_matrix_ping_flag);

                // load matrixB1 from L1 -> L0B(ping) l1_b1_buf_tensor(l1_b_pong_tensor)
                WAIT_FLAG(MTE2, MTE1, L1B1_matrix_pingpong_flag + L1B1_FLAG_SHIFT);
                WAIT_FLAG(M, MTE1, L0B_matrix_ping_flag);
                commonLoadData2dParamsTranspose.repeatTimes = headDim / C0_SIZE;
                commonLoadData2dParamsTranspose.srcStride = s2_block_align_length / C0_SIZE;
                for (int32_t i = 0; i < s2_block_align_length / C0_SIZE; i++) {
                    AscendC::LoadData(
                        (l0_b_ping_tensor)[i * headDim * C0_SIZE], 
                        (*l1_b1_buf_tensor)[i * C0_SIZE * C0_SIZE],
                        commonLoadData2dParamsTranspose
                    );
                }
                SET_FLAG(MTE1, MTE2, L1B1_matrix_pingpong_flag + L1B1_FLAG_SHIFT);
                SET_FLAG(MTE1, M, L0B_matrix_ping_flag);

                // mad (matrixA, matrixB1) -> l0_c1_buf_tensor
                // (s1_block_length, s2_block_length) X (s2_block_length, headDim)
                WAIT_FLAG(MTE1, M, L0A_matrix_ping_flag);
                WAIT_FLAG(MTE1, M, L0B_matrix_ping_flag);
                AscendC::LocalTensor<float> *l0_c1_buf_tensor = L0C_matrixC1_pingpong_flag ? &l0_c1_pong_tensor : &l0_c1_ping_tensor;
                uint16_t s1_modify = (s1_block_length == 1) ? 2 : s1_block_length;
                bool l0_c1_init_flag = (s2_idx == 0);
                bool l0_c1_last_flag = (s2_idx == (s2_loop - 1));
                commonMadParams.m = s1_modify;
                commonMadParams.n = headDim;
                commonMadParams.k = s2_block_length;
                commonMadParams.unitFlag = l0_c1_last_flag ? 3 : 2;
                commonMadParams.cmatrixInitVal = l0_c1_init_flag;
                AscendC::Mmad(
                    *l0_c1_buf_tensor, 
                    l0_a_ping_tensor, 
                    l0_b_ping_tensor,
                    commonMadParams
                );

                // fixpipe Mad(matrixA, matrixB1) from l0C1 -> gm_out
                if (l0_c1_last_flag) {                    
                    commonFixpipeParamsV220.mSize = s1_block_length;
                    commonFixpipeParamsV220.nSize = headDim;
                    commonFixpipeParamsV220.srcStride = s1_block_align_length;
                    commonFixpipeParamsV220.dstStride = qHeadNum * headDim;
                    AscendC::SetAtomicType<float>();
                    AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                        temp_tensor_fp32[matrixC1_offset + s1_idx * BASE_BLOCK_LENGTH * qHeadNum * headDim], 
                        *l0_c1_buf_tensor,
                        commonFixpipeParamsV220
                    );
                    AscendC::SetAtomicNone();
                }
                SET_FLAG(M, MTE1, L0A_matrix_ping_flag);
                SET_FLAG(M, MTE1, L0B_matrix_ping_flag);
                
                ///////////////////////////////// dQ /////////////////////////

                ///////////////////////////////// dK /////////////////////////

                // load matrixA^T from L1 -> L0A(pong) l0_a_pong_tensor
                WAIT_FLAG(M, MTE1, L0A_matrix_pong_flag);
                uint32_t s1_c0_loop = (s1_block_length + C0_SIZE - 1) / C0_SIZE;
                uint32_t s2_c0_loop = (s2_block_length + C0_SIZE - 1) / C0_SIZE;
                commonLoadData2dParamsTranspose.repeatTimes = s1_c0_loop * s2_c0_loop;
                commonLoadData2dParamsTranspose.srcStride = 1;
                AscendC::LoadData(
                    l0_a_pong_tensor,
                    (*l1_a_buf_tensor), 
                    commonLoadData2dParamsTranspose
                );
                SET_FLAG(MTE1, MTE2, L1A_matrix_pingpong_flag + L1A_FLAG_SHIFT);
                SET_FLAG(MTE1, M, L0A_matrix_pong_flag);
                
                // load matrixB2 from L1 -> L0B(pong) l0_b_pong_tensor
                WAIT_FLAG(M, MTE1, L0B_matrix_pong_flag);
                if (s1_idx == 0 && s2_idx == 0) {
                    WAIT_FLAG(MTE2, MTE1, L1B2_matrix_pingpong_flag + L1B2_FLAG_SHIFT);
                }
                int32_t l1_b2_buf_offset = s1_idx * BASE_BLOCK_LENGTH * C0_SIZE;
                commonLoadData2dParamsTranspose.repeatTimes = headDim / C0_SIZE;
                commonLoadData2dParamsTranspose.srcStride = km_align / C0_SIZE;
                for (int32_t i = 0; i < s1_block_align_length / C0_SIZE; i++) {
                    AscendC::LoadData(
                        (l0_b_pong_tensor)[i * C0_SIZE * headDim],
                        (*l1_b2_buf_tensor)[l1_b2_buf_offset + i * C0_SIZE * C0_SIZE],
                        commonLoadData2dParamsTranspose
                    );
                }
                if (s1_idx == inner_s1Loop - 1 && s2_idx == s2_loop - 1) {
                    SET_FLAG(MTE1, MTE2, L1B2_matrix_pingpong_flag + L1B2_FLAG_SHIFT);
                }
                SET_FLAG(MTE1, M, L0B_matrix_pong_flag);
                
                // mad (matrixA^T, matrixB2) -> l0_c2_buf_tensor 
                // (s2_block_length, s1_block_length) X (s1_block_length, headDim)
                WAIT_FLAG(MTE1, M, L0A_matrix_pong_flag);
                WAIT_FLAG(MTE1, M, L0B_matrix_pong_flag);
                AscendC::LocalTensor<float> *l0_c2_buf_tensor = L0C_matrixC2_pingpong_flag ? &l0_c2_pong_tensor : &l0_c2_ping_tensor;
                uint16_t s2_modify = (s2_block_length == 1) ? 2 : s2_block_length;
                bool l0_c2_init_flag = (s1_idx == 0);
                bool l0_c2_last_flag = (s1_idx == inner_s1Loop - 1);

                AscendC::Mmad(
                    *l0_c2_buf_tensor, 
                    l0_a_pong_tensor, 
                    l0_b_pong_tensor,
                    AscendC::MmadParams(
                        s2_modify, 
                        headDim,
                        s1_block_length, 
                        l0_c2_last_flag ? 3 : 2,
                        false,
                        l0_c2_init_flag
                    )
                );
                PipeBarrier<PIPE_M>();
                
                // fixpipe Mad(matrixA^T, matrixB2) from l0C2 -> gm_out
                if (s1_idx == inner_s1Loop - 1) {
                    commonFixpipeParamsV220.mSize = s2_block_length;
                    commonFixpipeParamsV220.nSize = headDim;
                    commonFixpipeParamsV220.srcStride = s2_block_align_length;
                    commonFixpipeParamsV220.dstStride = kvHeadNum * headDim;
                    AscendC::SetAtomicType<float>();
                    AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                        temp_tensor_fp32[matrixC2_offset + s2_idx * BASE_BLOCK_LENGTH * kvHeadNum * headDim], 
                        *l0_c2_buf_tensor,
                        commonFixpipeParamsV220
                    );
                    AscendC::SetAtomicNone();
                }

                SET_FLAG(M, MTE1, L0A_matrix_pong_flag);
                SET_FLAG(M, MTE1, L0B_matrix_pong_flag);

                ///////////////////////////////// dK /////////////////////////

                // L1B1 matrix flip in each single operation
                L1B1_matrix_pingpong_flag = 1 - L1B1_matrix_pingpong_flag;
                // L1A matrix flip in each single operation
                L1A_matrix_pingpong_flag = 1 - L1A_matrix_pingpong_flag;
            }
        }

        // L1B2 matrix flip in each round operation
        L1B2_matrix_pingpong_flag = 1 - L1B2_matrix_pingpong_flag;
    }
}
#endif