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
 * \file cube_op_bn2.h
 * \brief
 */

#ifndef __CUBE_OP_BN2_H__
#define __CUBE_OP_BN2_H__

#include "common_header.h"
#include "kernel_operator.h"

using namespace AscendC;
namespace FAG_BN2 {

template <typename INPUT_TYPE, const bool MM12_NZ, const bool MM345_NZ, const uint32_t LAYOUT>
class CubeOp {
private:
    AscendC::FixpipeParamsV220 commonFixpipeParamsV220{SIZE_128, SIZE_128, SIZE_128, SIZE_128, false};
    // l1
    LocalTensor<INPUT_TYPE> l1_query_ping_tensor;
    LocalTensor<INPUT_TYPE> l1_query_pong_tensor;
    LocalTensor<INPUT_TYPE> l1_key_ping_tensor;
    LocalTensor<INPUT_TYPE> l1_key_pong_tensor;
    LocalTensor<INPUT_TYPE> l1_dy_ping_tensor;
    LocalTensor<INPUT_TYPE> l1_dy_pong_tensor;
    LocalTensor<INPUT_TYPE> l1_tmp_tensor;
    // L0A L0B
    LocalTensor<INPUT_TYPE> l0_a_ping_tensor;
    LocalTensor<INPUT_TYPE> l0_a_pong_tensor;
    LocalTensor<INPUT_TYPE> l0_b_ping_tensor;
    LocalTensor<INPUT_TYPE> l0_b_pong_tensor;
    // L0C
    LocalTensor<float> l0_c_ping_tensor;
    LocalTensor<float> l0_c_pong_tensor;
    // GM
    GlobalTensor<INPUT_TYPE> tmp_type_gm_tensor;
    GlobalTensor<float> tmp_fp32_gm_tensor;

    uint32_t ping_pong_flag_outer = 0;
    uint32_t ping_pong_flag_inner = 0;
    uint32_t aEventIDPing = 4;
    uint32_t aEventIDPong = 5;
    uint32_t bEventIDPing = 6;
    uint32_t bEventIDPong = 7;

    // shape info
    uint32_t dimB;
    uint32_t dimN1;
    uint32_t dimN2;
    uint32_t dimD;
    uint32_t dimDAlign;
    uint32_t dimS1;
    uint32_t dimS2;

    // cal param
    uint32_t qSrcDvalue;
    uint32_t kvSrcDvalue;
    uint32_t singleM;
    uint32_t singleN;
    uint32_t singleMAlign;
    uint32_t singleNAlign;
    constexpr static uint32_t DB_NUM = 2;
    constexpr static uint32_t C0_SIZE = 16;

public:
    __aicore__ inline CubeOp(){};

    __aicore__ inline void Init(TPipe *pipe, const FlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict tiling_data)
    {
        AsdopsBuffer<ArchType::ASCEND_V220> asdopsBuf;
        uint64_t config = 0x1;
        AscendC::SetNdParaImpl(config);
        AscendC::SetLoadDataPaddingValue<uint64_t>(0);
        commonFixpipeParamsV220.quantPre = QuantMode_t::NoQuant;
        // tiling data
        dimB = tiling_data->opInfo.B;
        dimN2 = tiling_data->opInfo.N2;
        dimN1 = dimN2 * tiling_data->opInfo.G;
        dimS1 = tiling_data->opInfo.S1;
        dimS2 = tiling_data->opInfo.S2;
        dimD = tiling_data->opInfo.D;
        dimDAlign = RoundUp<uint32_t>(dimD, 16);
        singleM = tiling_data->splitCoreParams.singleM;
        singleN = tiling_data->splitCoreParams.singleN;
        singleMAlign = RoundUp<int32_t>(singleM, 16);
        singleNAlign = RoundUp<int32_t>(singleN, 16);

        if constexpr (LAYOUT == BNGSD) {
            qSrcDvalue = dimD;
            kvSrcDvalue = dimD;
        } else if constexpr (LAYOUT == BSNGD) {
            qSrcDvalue = dimN1 * dimD;
            kvSrcDvalue = dimN2 * dimD;
        } else if constexpr (LAYOUT == SBNGD) {
            qSrcDvalue = dimB * dimN1 * dimD;
            kvSrcDvalue = dimB * dimN2 * dimD;
        } else if constexpr (LAYOUT == TND) {
            qSrcDvalue = dimN1 * dimD;
            kvSrcDvalue = dimN2 * dimD;
        }

        // init L1 tensor
        uint32_t l1_offset = 0;
        // query
        l1_query_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, INPUT_TYPE>(l1_offset);
        l1_offset += singleM * dimDAlign * sizeof(INPUT_TYPE);
        l1_query_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, INPUT_TYPE>(l1_offset);
        l1_offset += singleM * dimDAlign * sizeof(INPUT_TYPE);
        // dy
        l1_dy_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, INPUT_TYPE>(l1_offset);
        l1_offset += singleM * dimDAlign * sizeof(INPUT_TYPE);
        l1_dy_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, INPUT_TYPE>(l1_offset);
        l1_offset += singleM * dimDAlign * sizeof(INPUT_TYPE);
        // key
        l1_key_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, INPUT_TYPE>(l1_offset);
        l1_offset += singleN * dimDAlign * sizeof(INPUT_TYPE);
        l1_key_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, INPUT_TYPE>(l1_offset);
        l1_offset += singleN * dimDAlign * sizeof(INPUT_TYPE);
        // tmp
        l1_tmp_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, INPUT_TYPE>(l1_offset);
        l1_offset += SIZE_128 * SIZE_128 * DB_NUM * sizeof(INPUT_TYPE);

        // init L0A/L0B/L0C tensor
        l0_a_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, INPUT_TYPE>(0);
        l0_a_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, INPUT_TYPE>(32 * 1024);
        l0_b_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, INPUT_TYPE>(0);
        l0_b_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, INPUT_TYPE>(32 * 1024);
        l0_c_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(0);
        l0_c_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(64 * 1024);
    }

    __aicore__ inline __attribute__((always_inline)) void MM12_SET_FLAG()
    {
        SET_FLAG(M, MTE1, aEventIDPing);
        SET_FLAG(M, MTE1, aEventIDPong);
        SET_FLAG(M, MTE1, bEventIDPing);
        SET_FLAG(M, MTE1, bEventIDPong);
        SET_FLAG(MTE1, MTE2, bEventIDPing);
        SET_FLAG(MTE1, MTE2, bEventIDPong);
        SET_FLAG(FIX, M, aEventIDPing);
        SET_FLAG(FIX, M, aEventIDPong);
    }

    __aicore__ inline __attribute__((always_inline)) void MM12_WAIT_FLAG()
    {
        WAIT_FLAG(M, MTE1, aEventIDPing);
        WAIT_FLAG(M, MTE1, aEventIDPong);
        WAIT_FLAG(M, MTE1, bEventIDPing);
        WAIT_FLAG(M, MTE1, bEventIDPong);
        WAIT_FLAG(MTE1, MTE2, bEventIDPing);
        WAIT_FLAG(MTE1, MTE2, bEventIDPong);
        WAIT_FLAG(FIX, M, aEventIDPing);
        WAIT_FLAG(FIX, M, aEventIDPong);
    }

    template <uint32_t CUBE_TYPE>
    __aicore__ inline __attribute__((always_inline)) void
    ComputeMM12(GlobalTensor<INPUT_TYPE> leftGm, GlobalTensor<INPUT_TYPE> rightGm, GlobalTensor<float> outGm,
                const uint32_t realM, const uint32_t realN, const uint32_t pingpong_left, const uint32_t pingpong_right,
                const bool reuse_left, const bool reuse_right)
    {
        int32_t mLoop = CeilDiv(realM, SIZE_128);
        int32_t nLoop = CeilDiv(realN, SIZE_128);
        int32_t mTail = (realM % SIZE_128) == 0 ? SIZE_128 : (realM % SIZE_128);
        int32_t nTail = (realN % SIZE_128) == 0 ? SIZE_128 : (realN % SIZE_128);

        for (int32_t nIdx = 0; nIdx < nLoop; nIdx++) {
            int32_t nProcess = (nIdx == nLoop - 1) ? nTail : SIZE_128;
            int32_t nProcessAlign = RoundUp(nProcess, SIZE_16);
            bool needCopyInAMatrix = nIdx == 0 ? true : false;
            bool needCopyInBMatrix = true;
            LocalTensor<INPUT_TYPE> l1_b_tensor;
            LocalTensor<INPUT_TYPE> l0_b_tensor = ping_pong_flag_outer ? l0_b_ping_tensor : l0_b_pong_tensor;
            uint32_t outer_event_id = ping_pong_flag_outer ? bEventIDPing : bEventIDPong;
            if (reuse_left) {
                needCopyInAMatrix = false;
            }

            for (int32_t mIdx = 0; mIdx < mLoop; mIdx++) {
                int32_t mProcess = (mIdx == mLoop - 1) ? mTail : SIZE_128;
                int32_t mProcessAlign = RoundUp(mProcess, SIZE_16);
                LocalTensor<INPUT_TYPE> l1_a_tensor;
                LocalTensor<INPUT_TYPE> l0_a_tensor = ping_pong_flag_inner ? l0_a_ping_tensor : l0_a_pong_tensor;
                LocalTensor<float> l0_c_tensor = ping_pong_flag_inner ? l0_c_ping_tensor : l0_c_pong_tensor;
                uint32_t inner_event_id = ping_pong_flag_inner ? aEventIDPing : aEventIDPong;
                uint64_t inner_query_offset, inner_key_offset, inner_out_offset;

                if constexpr (LAYOUT == BSNGD || LAYOUT == TND) {
                    inner_query_offset = mIdx * SIZE_128 * dimN1 * dimD;
                    inner_key_offset = nIdx * SIZE_128 * dimN2 * dimD;
                } else if constexpr (LAYOUT == SBNGD) {
                    inner_query_offset = mIdx * SIZE_128 * dimB * dimN1 * dimD;
                    inner_key_offset = nIdx * SIZE_128 * dimB * dimN2 * dimD;
                } else if constexpr (LAYOUT == BNGSD) {
                    inner_query_offset = mIdx * SIZE_128 * dimD;
                    inner_key_offset = nIdx * SIZE_128 * dimD;
                }

                if constexpr (CUBE_TYPE == QK) {
                    l1_a_tensor = pingpong_left ? l1_query_ping_tensor[mIdx * SIZE_128 * dimDAlign] :
                                                  l1_query_pong_tensor[mIdx * SIZE_128 * dimDAlign];
                    l1_b_tensor = pingpong_right ? l1_key_ping_tensor[nIdx * SIZE_128 * dimDAlign] :
                                                   l1_key_pong_tensor[nIdx * SIZE_128 * dimDAlign];
                } else {
                    l1_a_tensor = pingpong_left ? l1_dy_ping_tensor[mIdx * SIZE_128 * dimDAlign] :
                                                  l1_dy_pong_tensor[mIdx * SIZE_128 * dimDAlign];
                    l1_b_tensor = ping_pong_flag_outer ? l1_tmp_tensor : l1_tmp_tensor[SIZE_128 * SIZE_128];
                }

                WAIT_FLAG(M, MTE1, inner_event_id);
                load_data_gm_2_l0<true, false>(l0_a_tensor, l1_a_tensor, leftGm[inner_query_offset], mProcess, dimD,
                                               qSrcDvalue, mProcessAlign, dimDAlign, needCopyInAMatrix);

                if (needCopyInBMatrix) {
                    // l1\l0常驻
                    WAIT_FLAG(M, MTE1, outer_event_id);
                    WAIT_FLAG(MTE1, MTE2, outer_event_id);
                    load_data_gm_2_l0_trans<false, false>(l0_b_tensor, l1_b_tensor, rightGm[inner_key_offset], nProcess,
                                                          dimD, kvSrcDvalue, nProcessAlign, dimDAlign, !reuse_right);
                    SET_FLAG(MTE1, MTE2, outer_event_id);
                }

                WAIT_FLAG(FIX, M, inner_event_id);
                SET_FLAG(MTE1, M, EVENT_ID0);
                WAIT_FLAG(MTE1, M, EVENT_ID0);
                MmadParams madParams;
                madParams.m = mProcess == 1 ? 2 : mProcess;
                madParams.n = nProcess;
                madParams.k = dimD;
                madParams.cmatrixInitVal = true;
                madParams.unitFlag = 3;
                AscendC::Mmad(l0_c_tensor, l0_a_tensor, l0_b_tensor, madParams);
                AscendC::PipeBarrier<PIPE_M>();
                SET_FLAG(M, MTE1, inner_event_id);
                if (needCopyInBMatrix) {
                    SET_FLAG(M, MTE1, outer_event_id);
                    needCopyInBMatrix = false;
                }

                if constexpr (MM12_NZ) {
                    inner_out_offset = mIdx * SIZE_128 * C0_SIZE + nIdx * SIZE_128 * realM;
                    commonFixpipeParamsV220.mSize = mProcess;
                    commonFixpipeParamsV220.nSize = nProcessAlign;
                    commonFixpipeParamsV220.srcStride = mProcessAlign;
                    commonFixpipeParamsV220.dstStride = realM * 2;
                    commonFixpipeParamsV220.unitFlag = 3;
                    AscendC::Fixpipe<float, float, AscendC::CFG_NZ>(outGm[inner_out_offset], l0_c_tensor,
                                                                    commonFixpipeParamsV220);
                } else {
                    uint32_t realN_align8 = RoundUp<uint32_t>(realN, 8);
                    inner_out_offset = mIdx * SIZE_128 * realN_align8 + nIdx * SIZE_128;
                    commonFixpipeParamsV220.mSize = mProcess;
                    commonFixpipeParamsV220.nSize = nProcess;
                    commonFixpipeParamsV220.srcStride = mProcessAlign;
                    commonFixpipeParamsV220.dstStride = realN_align8;
                    commonFixpipeParamsV220.unitFlag = 3;
                    AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(outGm[inner_out_offset], l0_c_tensor,
                                                                           commonFixpipeParamsV220);
                }
                SET_FLAG(FIX, M, inner_event_id);
                ping_pong_flag_inner = 1 - ping_pong_flag_inner;
            }
            ping_pong_flag_outer = 1 - ping_pong_flag_outer;
        }
    };


    __aicore__ inline __attribute__((always_inline)) void MM345_SET_FLAG()
    {
        SET_FLAG(M, MTE1, aEventIDPing);
        SET_FLAG(M, MTE1, aEventIDPong);
        SET_FLAG(MTE1, MTE2, aEventIDPing);
        SET_FLAG(MTE1, MTE2, aEventIDPong);
        SET_FLAG(FIX, M, bEventIDPing);
        SET_FLAG(FIX, M, bEventIDPong);
    }

    __aicore__ inline __attribute__((always_inline)) void MM345_WAIT_FLAG()
    {
        WAIT_FLAG(M, MTE1, aEventIDPing);
        WAIT_FLAG(M, MTE1, aEventIDPong);
        WAIT_FLAG(MTE1, MTE2, aEventIDPing);
        WAIT_FLAG(MTE1, MTE2, aEventIDPong);
        WAIT_FLAG(FIX, M, bEventIDPing);
        WAIT_FLAG(FIX, M, bEventIDPong);
    }

    __aicore__ inline __attribute__((always_inline)) void
    ComputeMMDQ(GlobalTensor<INPUT_TYPE> leftGm, GlobalTensor<INPUT_TYPE> rightGm, GlobalTensor<float> outGm,
                const uint32_t realM, const uint32_t realN, const uint32_t taskIdPingPong, const uint8_t atomicAdd)
    {
        int32_t mLoop = CeilDiv(realM, SIZE_128);
        int32_t nLoop = CeilDiv(realN, SIZE_128);
        int32_t mTail = (realM % SIZE_128) == 0 ? SIZE_128 : (realM % SIZE_128);
        int32_t nTail = (realN % SIZE_128) == 0 ? SIZE_128 : (realN % SIZE_128);

        for (int32_t mIdx = 0; mIdx < mLoop; mIdx++) {
            int32_t mProcess = (mIdx == mLoop - 1) ? mTail : SIZE_128;
            int32_t mProcessAlign = RoundUp(mProcess, SIZE_16);
            LocalTensor<INPUT_TYPE> l1_a_tensor;
            LocalTensor<float> l0_c_tensor = ping_pong_flag_outer ? l0_c_ping_tensor : l0_c_pong_tensor;
            uint32_t outer_event_id = ping_pong_flag_outer ? bEventIDPing : bEventIDPong;
            bool needCopyInAMatrix = true;
            bool needCopyInBMatrix = false;
            bool initL0C = true;

            WAIT_FLAG(FIX, M, outer_event_id);
            for (int32_t nIdx = 0; nIdx < nLoop; nIdx++) {
                int32_t nProcess = (nIdx == nLoop - 1) ? nTail : SIZE_128;
                int32_t nProcessAlign = RoundUp(nProcess, SIZE_16);
                LocalTensor<INPUT_TYPE> l1_b_tensor;
                LocalTensor<INPUT_TYPE> l0_a_tensor = ping_pong_flag_inner ? l0_a_ping_tensor : l0_a_pong_tensor;
                LocalTensor<INPUT_TYPE> l0_b_tensor = ping_pong_flag_inner ? l0_b_ping_tensor : l0_b_pong_tensor;
                uint32_t inner_event_id = ping_pong_flag_inner ? aEventIDPing : aEventIDPong;
                uint64_t left_gm_offset, right_key_gm_offset, out_gm_offset;
                if constexpr (LAYOUT == BSNGD || LAYOUT == TND) {
                    right_key_gm_offset = nIdx * SIZE_128 * dimN2 * dimD;
                } else if constexpr (LAYOUT == SBNGD) {
                    right_key_gm_offset = nIdx * SIZE_128 * dimB * dimN2 * dimD;
                } else if constexpr (LAYOUT == BNGSD) {
                    right_key_gm_offset = nIdx * SIZE_128 * dimD;
                }
                l1_a_tensor = ping_pong_flag_inner ? l1_tmp_tensor : l1_tmp_tensor[SIZE_128 * SIZE_128];
                l1_b_tensor = taskIdPingPong ? l1_key_ping_tensor[nIdx * SIZE_128 * dimDAlign] :
                                               l1_key_pong_tensor[nIdx * SIZE_128 * dimDAlign];

                WAIT_FLAG(M, MTE1, inner_event_id);
                WAIT_FLAG(MTE1, MTE2, inner_event_id);
                if constexpr (MM12_NZ) {
                    left_gm_offset = mIdx * SIZE_128 * C0_SIZE + nIdx * SIZE_128 * singleM;
                    load_data_gm_2_l0<true, true>(l0_a_tensor, l1_a_tensor, leftGm[left_gm_offset], mProcess, nProcess,
                                                  singleM, mProcessAlign, nProcessAlign, needCopyInAMatrix);
                } else {
                    uint32_t realN_align16 = RoundUp<uint32_t>(realN, 16);
                    left_gm_offset = mIdx * SIZE_128 * realN_align16 + nIdx * SIZE_128;
                    load_data_gm_2_l0<true, false>(l0_a_tensor, l1_a_tensor, leftGm[left_gm_offset], mProcess, nProcess,
                                                   realN_align16, mProcessAlign, nProcessAlign, needCopyInAMatrix);
                }
                SET_FLAG(MTE1, MTE2, inner_event_id);

                load_data_gm_2_l0<false, false>(l0_b_tensor, l1_b_tensor, rightGm[right_key_gm_offset], nProcess, dimD,
                                                kvSrcDvalue, nProcessAlign, dimDAlign, needCopyInBMatrix);

                SET_FLAG(MTE1, M, EVENT_ID0);
                WAIT_FLAG(MTE1, M, EVENT_ID0);
                MmadParams madParams;
                madParams.m = mProcess == 1 ? 2 : mProcess;
                madParams.n = dimD;
                madParams.k = nProcess;
                madParams.cmatrixInitVal = initL0C;
                madParams.unitFlag = (nIdx == nLoop - 1) ? 3 : 2;
                AscendC::Mmad(l0_c_tensor, l0_a_tensor, l0_b_tensor, madParams);
                AscendC::PipeBarrier<PIPE_M>();
                SET_FLAG(M, MTE1, inner_event_id);
                initL0C = false;

                if (nIdx == nLoop - 1) {
                    if (atomicAdd) {
                        AscendC::SetAtomicType<float>();
                    }

                    if constexpr (MM345_NZ) {
                        out_gm_offset = mIdx * SIZE_128 * C0_SIZE;
                        commonFixpipeParamsV220.mSize = mProcess;
                        commonFixpipeParamsV220.nSize = dimDAlign;
                        commonFixpipeParamsV220.srcStride = mProcessAlign;
                        commonFixpipeParamsV220.dstStride = dimS1 * C0_SIZE * sizeof(float) / 32;
                        commonFixpipeParamsV220.unitFlag = 3;
                        AscendC::Fixpipe<float, float, AscendC::CFG_NZ>(outGm[out_gm_offset], l0_c_tensor,
                                                                        commonFixpipeParamsV220);
                    } else {
                        uint32_t out_stride;
                        if constexpr (LAYOUT == BSNGD || LAYOUT == TND) {
                            out_gm_offset = mIdx * SIZE_128 * dimN2 * dimD;
                            out_stride = dimN2 * dimD;
                        } else if constexpr (LAYOUT == SBNGD) {
                            out_gm_offset = mIdx * SIZE_128 * dimB * dimN2 * dimD;
                            out_stride = dimB * dimN2 * dimD;
                        } else if constexpr (LAYOUT == BNGSD) {
                            out_gm_offset = mIdx * SIZE_128 * dimD;
                            out_stride = dimD;
                        }
                        commonFixpipeParamsV220.mSize = mProcess;
                        commonFixpipeParamsV220.nSize = dimD;
                        commonFixpipeParamsV220.srcStride = mProcessAlign;
                        commonFixpipeParamsV220.dstStride = out_stride;
                        commonFixpipeParamsV220.unitFlag = 3;
                        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(outGm[out_gm_offset], l0_c_tensor,
                                                                               commonFixpipeParamsV220);
                    }
                    if (atomicAdd) {
                        AscendC::SetAtomicNone();
                    }
                }
                ping_pong_flag_inner = 1 - ping_pong_flag_inner;
            }
            SET_FLAG(FIX, M, outer_event_id);
            ping_pong_flag_outer = 1 - ping_pong_flag_outer;
        }
    }

    template <uint32_t CUBE_TYPE>
    __aicore__ inline __attribute__((always_inline)) void
    ComputeMMDKV(GlobalTensor<INPUT_TYPE> leftGm, GlobalTensor<INPUT_TYPE> rightGm, GlobalTensor<float> outGm,
                 const uint32_t realM, const uint32_t realN, const uint32_t taskIdPingPong, const uint8_t atomicAdd)
    {
        int32_t mLoop = CeilDiv(realM, SIZE_128);
        int32_t nLoop = CeilDiv(realN, SIZE_128);
        int32_t mTail = (realM % SIZE_128) == 0 ? SIZE_128 : (realM % SIZE_128);
        int32_t nTail = (realN % SIZE_128) == 0 ? SIZE_128 : (realN % SIZE_128);

        for (int32_t nIdx = 0; nIdx < nLoop; nIdx++) {
            int32_t nProcess = (nIdx == nLoop - 1) ? nTail : SIZE_128;
            int32_t nProcessAlign = RoundUp(nProcess, SIZE_16);
            bool initL0C = true;
            LocalTensor<INPUT_TYPE> l1_b_tensor;
            LocalTensor<float> l0_c_tensor = ping_pong_flag_outer ? l0_c_ping_tensor : l0_c_pong_tensor;
            uint32_t outer_event_id = ping_pong_flag_outer ? bEventIDPing : bEventIDPong;
            bool needCopyInAMatrix = true;
            bool needCopyInBMatrix = false;

            WAIT_FLAG(FIX, M, outer_event_id);
            for (int32_t mIdx = 0; mIdx < mLoop; mIdx++) {
                int32_t mProcess = (mIdx == mLoop - 1) ? mTail : SIZE_128;
                int32_t mProcessAlign = RoundUp(mProcess, SIZE_16);
                LocalTensor<INPUT_TYPE> l1_a_tensor =
                    ping_pong_flag_inner ? l1_tmp_tensor : l1_tmp_tensor[SIZE_128 * SIZE_128];
                LocalTensor<INPUT_TYPE> l0_a_tensor = ping_pong_flag_inner ? l0_a_ping_tensor : l0_a_pong_tensor;
                LocalTensor<INPUT_TYPE> l0_b_tensor = ping_pong_flag_inner ? l0_b_ping_tensor : l0_b_pong_tensor;
                uint32_t inner_event_id = ping_pong_flag_inner ? aEventIDPing : aEventIDPong;
                uint64_t left_gm_offset, right_query_gm_offset, out_gm_offset;
                if constexpr (LAYOUT == BSNGD || LAYOUT == TND) {
                    right_query_gm_offset = mIdx * SIZE_128 * (dimN1 * dimD);
                } else if constexpr (LAYOUT == SBNGD) {
                    right_query_gm_offset = mIdx * SIZE_128 * (dimB * dimN1 * dimD);
                } else {
                    right_query_gm_offset = mIdx * SIZE_128 * dimD;
                }

                if constexpr (CUBE_TYPE == DK) {
                    l1_b_tensor = taskIdPingPong ? l1_query_ping_tensor[mIdx * SIZE_128 * dimDAlign] :
                                                   l1_query_pong_tensor[mIdx * SIZE_128 * dimDAlign];
                } else {
                    l1_b_tensor = taskIdPingPong ? l1_dy_ping_tensor[mIdx * SIZE_128 * dimDAlign] :
                                                   l1_dy_pong_tensor[mIdx * SIZE_128 * dimDAlign];
                }

                WAIT_FLAG(M, MTE1, inner_event_id);
                WAIT_FLAG(MTE1, MTE2, inner_event_id);
                if constexpr (MM12_NZ) {
                    left_gm_offset = mIdx * SIZE_128 * C0_SIZE + nIdx * SIZE_128 * singleM;
                    load_data_gm_2_l0_trans<true, true>(l0_a_tensor, l1_a_tensor, leftGm[left_gm_offset], mProcess,
                                                        nProcess, singleM, mProcessAlign, nProcessAlign,
                                                        needCopyInAMatrix);
                } else {
                    uint32_t realN_align16 = RoundUp<uint32_t>(realN, 16);
                    left_gm_offset = mIdx * SIZE_128 * realN_align16 + nIdx * SIZE_128;
                    load_data_gm_2_l0_trans<true, false>(l0_a_tensor, l1_a_tensor, leftGm[left_gm_offset], mProcess,
                                                         nProcess, realN_align16, mProcessAlign, nProcessAlign,
                                                         needCopyInAMatrix);
                }
                SET_FLAG(MTE1, MTE2, inner_event_id);

                load_data_gm_2_l0<false, false>(l0_b_tensor, l1_b_tensor, rightGm[right_query_gm_offset], mProcess,
                                                dimD, qSrcDvalue, mProcessAlign, dimDAlign, needCopyInBMatrix);

                SET_FLAG(MTE1, M, EVENT_ID0);
                WAIT_FLAG(MTE1, M, EVENT_ID0);
                MmadParams madParams;
                madParams.m = nProcess == 1 ? 2 : nProcess;
                madParams.n = dimD;
                madParams.k = mProcess;
                madParams.cmatrixInitVal = initL0C;
                madParams.unitFlag = (mIdx == mLoop - 1) ? 3 : 2;
                AscendC::Mmad(l0_c_tensor, l0_a_tensor, l0_b_tensor, madParams);
                AscendC::PipeBarrier<PIPE_M>();
                SET_FLAG(M, MTE1, inner_event_id);
                initL0C = false;

                if (mIdx == mLoop - 1) {
                    if (atomicAdd) {
                        AscendC::SetAtomicType<float>();
                    }

                    if constexpr (MM345_NZ) {
                        out_gm_offset = nIdx * SIZE_128 * C0_SIZE;
                        commonFixpipeParamsV220.mSize = nProcess;
                        commonFixpipeParamsV220.nSize = dimDAlign;
                        commonFixpipeParamsV220.srcStride = nProcessAlign;
                        commonFixpipeParamsV220.dstStride = dimS2 * C0_SIZE * sizeof(float) / 32;
                        commonFixpipeParamsV220.unitFlag = 3;
                        AscendC::Fixpipe<float, float, AscendC::CFG_NZ>(outGm[out_gm_offset], l0_c_tensor,
                                                                        commonFixpipeParamsV220);
                    } else {
                        uint32_t out_stride;
                        if constexpr (LAYOUT == BSNGD || LAYOUT == TND) {
                            out_gm_offset = nIdx * SIZE_128 * dimN1 * dimD;
                            out_stride = dimN1 * dimD;
                        } else if constexpr (LAYOUT == SBNGD) {
                            out_gm_offset = nIdx * SIZE_128 * dimB * dimN1 * dimD;
                            out_stride = dimB * dimN1 * dimD;
                        } else if constexpr (LAYOUT == BNGSD) {
                            out_gm_offset = nIdx * SIZE_128 * dimD;
                            out_stride = dimD;
                        }
                        commonFixpipeParamsV220.mSize = nProcess;
                        commonFixpipeParamsV220.nSize = dimD;
                        commonFixpipeParamsV220.srcStride = nProcessAlign;
                        commonFixpipeParamsV220.dstStride = out_stride;
                        commonFixpipeParamsV220.unitFlag = 3;
                        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(outGm[out_gm_offset], l0_c_tensor,
                                                                               commonFixpipeParamsV220);
                    }
                    if (atomicAdd) {
                        AscendC::SetAtomicNone();
                    }
                }
                ping_pong_flag_inner = 1 - ping_pong_flag_inner;
            }

            SET_FLAG(FIX, M, outer_event_id);
            ping_pong_flag_outer = 1 - ping_pong_flag_outer;
        }
    }

private:
    template <bool NZ>
    __aicore__ inline void load_data_gm_2_l1(LocalTensor<INPUT_TYPE> dstL1Tensor, GlobalTensor<INPUT_TYPE> srcGmTensor,
                                             const int32_t mSize, const int32_t kSize, const int32_t mSizeAlign,
                                             const int32_t kSizeAlign, const int32_t srcStride)
    {
        if constexpr (NZ) {
            AscendC::DataCopyParams intriParams;
            intriParams.blockCount = kSizeAlign / C0_SIZE;
            intriParams.blockLen = mSizeAlign * C0_SIZE * sizeof(INPUT_TYPE) / 32;
            intriParams.srcStride = (srcStride - mSizeAlign) * C0_SIZE * sizeof(INPUT_TYPE) / 32;
            intriParams.dstStride = 0;
            DataCopy(dstL1Tensor, srcGmTensor, intriParams);
        } else {
            Nd2NzParams nd2nzPara;
            nd2nzPara.ndNum = 1;
            nd2nzPara.nValue = mSize;
            nd2nzPara.dValue = kSize;
            nd2nzPara.srcDValue = srcStride;
            nd2nzPara.dstNzC0Stride = mSizeAlign;
            nd2nzPara.dstNzNStride = 1;
            nd2nzPara.srcNdMatrixStride = 0;
            nd2nzPara.dstNzMatrixStride = 0;
            AscendC::DataCopy(dstL1Tensor, srcGmTensor, nd2nzPara);
        }
    }

    template <bool A_MATRIX, bool NZ>
    __aicore__ inline void load_data_gm_2_l0(LocalTensor<INPUT_TYPE> dstL0Tensor, LocalTensor<INPUT_TYPE> dstL1Tensor,
                                             GlobalTensor<INPUT_TYPE> srcGmTensor, const int32_t mSize,
                                             const int32_t kSize, const int32_t srcKSize, const int32_t mSizeAlign,
                                             const int32_t kSizeAlign, const bool mte2Copy)
    {
        if (mte2Copy) {
            load_data_gm_2_l1<NZ>(dstL1Tensor, srcGmTensor, mSize, kSize, mSizeAlign, kSizeAlign, srcKSize);
            SET_FLAG(MTE2, MTE1, EVENT_ID0);
            WAIT_FLAG(MTE2, MTE1, EVENT_ID0);
        }

        AscendC::LoadData2dParams loadData2dParams;
        loadData2dParams.repeatTimes = kSizeAlign / 16;
        loadData2dParams.srcStride = mSizeAlign / 16;
        if constexpr (A_MATRIX) {
            loadData2dParams.ifTranspose = false;
        } else {
            loadData2dParams.ifTranspose = true;
        }

        for (int32_t i = 0; i < mSizeAlign / 16; i++) {
            AscendC::LoadData(dstL0Tensor[i * kSizeAlign * 16], dstL1Tensor[i * 256], loadData2dParams);
        }
    }

    template <bool A_MATRIX, bool NZ>
    __aicore__ inline void
    load_data_gm_2_l0_trans(LocalTensor<INPUT_TYPE> dstL0Tensor, LocalTensor<INPUT_TYPE> dstL1Tensor,
                            GlobalTensor<INPUT_TYPE> srcGmTensor, const int32_t mSize, const int32_t kSize,
                            const int32_t srcKSize, const int32_t mSizeAlign, const int32_t kSizeAlign,
                            const bool mte2Copy)
    {
        if (mte2Copy) {
            load_data_gm_2_l1<NZ>(dstL1Tensor, srcGmTensor, mSize, kSize, mSizeAlign, kSizeAlign, srcKSize);
            SET_FLAG(MTE2, MTE1, EVENT_ID0);
            WAIT_FLAG(MTE2, MTE1, EVENT_ID0);
        }

        AscendC::LoadData2dParams loadData2dParams;
        loadData2dParams.repeatTimes = mSizeAlign * kSizeAlign / 256;
        loadData2dParams.srcStride = 1;
        if constexpr (A_MATRIX) {
            loadData2dParams.ifTranspose = true;
        } else {
            loadData2dParams.ifTranspose = false;
        }

        AscendC::LoadData(dstL0Tensor, dstL1Tensor, loadData2dParams);
    }
};

} // namespace FAG_BN2
#endif // __CUBE_OP_BN2_H__