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
 * \file flash_attention_score_grad_s1s2_bn2gs1s2_basic.h
 * \brief
 */

#ifndef _FLASH_ATTENTION_SCORE_GRAD_S1S2_BASIC_H_
#define _FLASH_ATTENTION_SCORE_GRAD_S1S2_BASIC_H_

#include "kernel_operator.h"
#include "./basic_modules/common_header.h"
#include "./basic_modules/cube_modules/cube_addr.h"
#include "./basic_modules/vec_modules/vec_pre.h"
#include "./basic_modules/vec_modules/vec_sfmg.h"
#include "./basic_modules/vec_modules/vec_post.h"
#include "./basic_modules/vec_modules/vec_addr.h"
#include "./basic_modules/cube_op.h"
#include "./basic_modules/vec_op.h"
using namespace AscendC;

template <typename TYPE, class TILING_CLASS>
class FlashAttentionScoreGradBasic {
public:
    __aicore__ inline FlashAttentionScoreGradBasic(){};
    __aicore__ inline void Process(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR atten_mask,
                                   GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR attention_in,
                                   GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR dq, GM_ADDR dk,
                                   GM_ADDR dv, GM_ADDR user, const TILING_CLASS *tilingData);

private:
    __aicore__ inline void CubeProcess(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR atten_mask,
                                       GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR attention_in,
                                       GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR dq, GM_ADDR dk,
                                       GM_ADDR dv, GM_ADDR user, const TILING_CLASS *tilingData);
    __aicore__ inline void VectorProcess(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR atten_mask,
                                         GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR attention_in,
                                         GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR dq, GM_ADDR dk,
                                         GM_ADDR dv, GM_ADDR user, const TILING_CLASS *tilingData);
    __aicore__ inline void VecPreProcess(GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR user,
                                         const TILING_CLASS *tilingData);
    __aicore__ inline void VecSftgProcess(GM_ADDR dy, GM_ADDR attention_in, GM_ADDR actual_seq_qlen, GM_ADDR user,
                                          const TILING_CLASS *tilingData);
    __aicore__ inline void VecMainProcess(GM_ADDR atten_mask, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                          GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR user,
                                          const TILING_CLASS *tilingData);
    __aicore__ inline void VecPostProcess(GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR user,
                                          const TILING_CLASS *tilingData);
    __aicore__ inline void SetFlag();
    __aicore__ inline void WaitFlag();
    int64_t batch;
    int64_t n1;
    int64_t n2;
    int64_t g;
    int64_t headDim;
    int64_t coreNum;
    int64_t mixCoreNum;
    uint32_t sparseMode;
    GM_ADDR mm1WorkSpaceAddr;
    GM_ADDR mm2WorkSpaceAddr;
    GM_ADDR dqWorkSpaceAddr;
    GM_ADDR dkWorkSpaceAddr;
    GM_ADDR dvWorkSpaceAddr;
};

template <typename TYPE, class TILING_CLASS>
__aicore__ inline void FlashAttentionScoreGradBasic<TYPE, TILING_CLASS>::Process(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR atten_mask, GM_ADDR softmax_max, GM_ADDR softmax_sum,
    GM_ADDR attention_in, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
    GM_ADDR user, const TILING_CLASS *tilingData)
{
    batch = tilingData->mlaTensorTilingData.b;
    n1 = tilingData->mlaTensorTilingData.n2 * tilingData->mlaTensorTilingData.g;
    n2 = tilingData->mlaTensorTilingData.n2;
    g = tilingData->mlaTensorTilingData.g;
    headDim = tilingData->mlaTensorTilingData.d;
    coreNum = tilingData->mlaTensorTilingData.coreNum;
    mixCoreNum = (coreNum + 1) / 2;
    sparseMode = tilingData->mlaTensorTilingData.sparseMode;
    mm1WorkSpaceAddr = user + tilingData->mlaTensorTilingData.mm1WorkspaceOffset;
    mm2WorkSpaceAddr = user + tilingData->mlaTensorTilingData.mm2WorkspaceOffset;
    dqWorkSpaceAddr = user + tilingData->mlaTensorTilingData.dqWorkSpaceOffset;
    dkWorkSpaceAddr = user + tilingData->mlaTensorTilingData.dkWorkSpaceOffset;
    dvWorkSpaceAddr = user + tilingData->mlaTensorTilingData.dvWorkSpaceOffset;

    if ASCEND_IS_AIC {
        CubeProcess(query, key, value, dy, atten_mask, softmax_max, softmax_sum, attention_in, actual_seq_qlen,
                    actual_seq_kvlen, dq, dk, dv, user, tilingData);
    }

    if ASCEND_IS_AIV {
        VectorProcess(query, key, value, dy, atten_mask, softmax_max, softmax_sum, attention_in, actual_seq_qlen,
                      actual_seq_kvlen, dq, dk, dv, user, tilingData);
    }
}

template <typename TYPE, class TILING_CLASS>
__aicore__ inline void FlashAttentionScoreGradBasic<TYPE, TILING_CLASS>::CubeProcess(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR atten_mask, GM_ADDR softmax_max, GM_ADDR softmax_sum,
    GM_ADDR attention_in, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
    GM_ADDR user, const TILING_CLASS *tilingData)
{
    AscendC::TPipe pipeCube;
    CUBE_OP::CubeOp<TYPE> cubeOp;
    CUBE_ADDR::CubeAddr cubeAddr;
    struct CubeAddrInfo cubeAddrInfo[2];
    struct CubeAddrInfo cube3AddrInfo[2];
    int32_t taskId = 0;
    bool running = true;
    // cube2 and cube3 mix when headDim is 64 and sparseMode is 0
    bool cube23_mix_flag = (sparseMode == 0 && headDim == 64);

    cubeOp.Init(&pipeCube, n1, n2, headDim, sparseMode);
    cubeAddr.init(batch, n1, g, headDim, GetBlockIdx(), actual_seq_qlen, actual_seq_kvlen, mixCoreNum, sparseMode);

    while (running) {
        cubeAddrInfo[taskId % 2].taskId = taskId;
        cube3AddrInfo[taskId % 2].taskId = taskId;
        cubeAddr.addr_mapping(&cubeAddrInfo[taskId % 2], &cube3AddrInfo[taskId % 2]);
        if (cubeAddrInfo[taskId % 2].blockLength > 0) {
            SetFlag();
            cubeOp.Cube1Process(cubeAddrInfo[taskId % 2], (__gm__ TYPE *)query, (__gm__ TYPE *)key,
                                (__gm__ float *)mm2WorkSpaceAddr);
            cubeOp.Cube1Process(cubeAddrInfo[taskId % 2], (__gm__ TYPE *)dy, (__gm__ TYPE *)value,
                                (__gm__ float *)mm1WorkSpaceAddr);
            WaitFlag();
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2VEC);
        }
        if (taskId > 0 && cubeAddrInfo[(taskId - 1) % 2].blockLength > 0) {
            if (cube23_mix_flag) {
                AscendC::WaitEvent(VEC2CUBE);
                SetFlag();
                cubeOp.Cube23Process(cubeAddrInfo[(taskId - 1) % 2], (__gm__ TYPE *)mm1WorkSpaceAddr, (__gm__ TYPE *)key,
                                    (__gm__ TYPE *)query, (__gm__ float *)dqWorkSpaceAddr, (__gm__ float *)dkWorkSpaceAddr);
                WaitFlag();
                SetFlag();
                cubeOp.Cube3Process(cube3AddrInfo[(taskId - 1) % 2], (__gm__ TYPE *)mm2WorkSpaceAddr, (__gm__ TYPE *)dy,
                                    (__gm__ float *)dvWorkSpaceAddr);
                WaitFlag();
            } else {
                SetFlag();
                cubeOp.Cube2Process(cubeAddrInfo[(taskId - 1) % 2], (__gm__ TYPE *)mm1WorkSpaceAddr, (__gm__ TYPE *)key,
                                (__gm__ float *)dqWorkSpaceAddr);
                WaitFlag();
                SetFlag();
                cubeOp.Cube3Process(cube3AddrInfo[(taskId - 1) % 2], (__gm__ TYPE *)mm1WorkSpaceAddr, (__gm__ TYPE *)query,
                                (__gm__ float *)dkWorkSpaceAddr);
                cubeOp.Cube3Process(cube3AddrInfo[(taskId - 1) % 2], (__gm__ TYPE *)mm2WorkSpaceAddr, (__gm__ TYPE *)dy,
                                (__gm__ float *)dvWorkSpaceAddr);
                WaitFlag();
            }
        }
        if (cubeAddrInfo[taskId % 2].blockLength == 0) {
            running = false;
        }
        taskId++;
    }
    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2POST);
}

template <typename TYPE, class TILING_CLASS>
__aicore__ inline void FlashAttentionScoreGradBasic<TYPE, TILING_CLASS>::VectorProcess(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR atten_mask, GM_ADDR softmax_max, GM_ADDR softmax_sum,
    GM_ADDR attention_in, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
    GM_ADDR user, const TILING_CLASS *tilingData)
{
    VecPreProcess(dq, dk, dv, user, tilingData);
    VecSftgProcess(dy, attention_in, actual_seq_qlen, user, tilingData);
    VecMainProcess(atten_mask, softmax_max, softmax_sum, actual_seq_qlen, actual_seq_kvlen, user, tilingData);
    VecPostProcess(dq, dk, dv, user, tilingData);
}

template <typename TYPE, class TILING_CLASS>
__aicore__ inline void FlashAttentionScoreGradBasic<TYPE, TILING_CLASS>::VecPreProcess(GM_ADDR dq, GM_ADDR dk,
                                                                                       GM_ADDR dv, GM_ADDR user,
                                                                                       const TILING_CLASS *tilingData)
{
    AscendC::TPipe pipePre;
    VectorInitOuput<TILING_CLASS> opPre;
    opPre.Init(&pipePre, dq, dk, dv, user, tilingData);
    opPre.Process();
    pipePre.Destroy();
}

template <typename TYPE, class TILING_CLASS>
__aicore__ inline void FlashAttentionScoreGradBasic<TYPE, TILING_CLASS>::VecSftgProcess(
    GM_ADDR dy, GM_ADDR attention_in, GM_ADDR actual_seq_qlen, GM_ADDR user, const TILING_CLASS *tilingData)
{
    AscendC::TPipe pipeSoftmaxGrad;
    VectorSoftmaxGrad<TYPE, TILING_CLASS> opSoftmaxGrad;
    opSoftmaxGrad.Init(&pipeSoftmaxGrad, dy, attention_in, actual_seq_qlen, user, batch, tilingData);
    opSoftmaxGrad.Process();
    pipeSoftmaxGrad.Destroy();
    AscendC::SyncAll();
}

template <typename TYPE, class TILING_CLASS>
__aicore__ inline void FlashAttentionScoreGradBasic<TYPE, TILING_CLASS>::VecMainProcess(
    GM_ADDR atten_mask, GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
    GM_ADDR user, const TILING_CLASS *tilingData)
{
    AscendC::TPipe pipeVec;
    VecOp<TYPE, TILING_CLASS> op;
    struct VecAddrInfo vecAddrInfo;
    int32_t taskId = 0;
    bool running = true;

    op.Init(softmax_max, softmax_sum, atten_mask, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipeVec);
    VEC_ADDR::VecAddr vecAddr;
    vecAddr.init(batch, n1, g, headDim, GetBlockIdx(), actual_seq_qlen, actual_seq_kvlen, mixCoreNum, sparseMode);
    while (running) {
        vecAddr.addr_mapping(&vecAddrInfo);
        if (vecAddrInfo.blockLength > 0) {
            AscendC::WaitEvent(CUBE2VEC);
            vecAddrInfo.taskId = taskId;
            op.Process(vecAddrInfo);
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(VEC2CUBE);
        }
        if (vecAddrInfo.blockLength == 0) {
            running = false;
        }
        taskId++;
    }
    pipeVec.Destroy();
    AscendC::WaitEvent(CUBE2POST);
    AscendC::SyncAll();
}

template <typename TYPE, class TILING_CLASS>
__aicore__ inline void FlashAttentionScoreGradBasic<TYPE, TILING_CLASS>::VecPostProcess(GM_ADDR dq, GM_ADDR dk,
                                                                                        GM_ADDR dv, GM_ADDR user,
                                                                                        const TILING_CLASS *tilingData)
{
    AscendC::TPipe pipePost;
    VectorPost<TYPE, TILING_CLASS> vecPost;
    vecPost.Init(&pipePost, dq, dk, dv, user, tilingData);
    vecPost.Process();
    pipePost.Destroy();
}

template <typename TYPE, class TILING_CLASS>
__aicore__ inline void FlashAttentionScoreGradBasic<TYPE, TILING_CLASS>::SetFlag()
{
    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);
    SET_FLAG(MTE1, MTE2, EVENT_ID4);
    SET_FLAG(MTE1, MTE2, EVENT_ID5);
    SET_FLAG(M, MTE1, EVENT_ID3);
    SET_FLAG(M, MTE1, EVENT_ID4);
    SET_FLAG(M, MTE1, EVENT_ID5);
    SET_FLAG(M, MTE1, EVENT_ID6);
}

template <typename TYPE, class TILING_CLASS>
__aicore__ inline void FlashAttentionScoreGradBasic<TYPE, TILING_CLASS>::WaitFlag()
{
    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID4);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID5);
    WAIT_FLAG(M, MTE1, EVENT_ID3);
    WAIT_FLAG(M, MTE1, EVENT_ID4);
    WAIT_FLAG(M, MTE1, EVENT_ID5);
    WAIT_FLAG(M, MTE1, EVENT_ID6);
}
#endif