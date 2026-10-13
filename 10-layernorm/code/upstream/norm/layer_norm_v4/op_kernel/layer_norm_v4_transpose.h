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
 * \file layer_norm_v4_transpose.h
 * \brief
 */

#ifndef LAYER_NORM_V4_TRANSPOSE_H
#define LAYER_NORM_V4_TRANSPOSE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace LayerNormV4 {
using namespace AscendC;

template <typename Tfm, typename Tweight>
class LayerNormV4Transpose {
public:
    __aicore__ inline LayerNormV4Transpose()
    {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
        const LayerNormV4TilingDataTranspose* __restrict tilingData)
    {
        // load tiling data
        row = tilingData->row;
        col = tilingData->col;
        numBlocks = tilingData->numBlocks;
        blockTail = tilingData->blockTail;
        ubFormer = tilingData->ubFormer;
        bFormer = tilingData->bFormer;
        blockFormer = tilingData->blockFormer;
        ubLoopOfTailBlock = tilingData->ubLoopOfTailBlock;
        ubTailOfFormerBlock = tilingData->ubTailOfFormerBlock;
        dichotomizeAddDiffSize = tilingData->dichotomizeAddDiffSize;
        ubTailOfTailBlock = tilingData->ubTailOfTailBlock;
        ubLoopOfFormerBlock = tilingData->ubLoopOfFormerBlock;
        eps = tilingData->eps;
        nullptrGamma = tilingData->nullptrGamma;
        nullptrBeta = tilingData->nullptrBeta;
        coefficient = tilingData->coefficient;
        // set global buffer
        xGm.SetGlobalBuffer((__gm__ Tfm*)x + blockIdx * blockFormer * row);
        betaGm.SetGlobalBuffer((__gm__ Tweight*)beta);
        gammaGm.SetGlobalBuffer((__gm__ Tweight*)gamma);

        yGm.SetGlobalBuffer((__gm__ Tfm*)y + blockIdx * blockFormer * row);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockIdx * blockFormer);
        meanGm.SetGlobalBuffer((__gm__ float*)mean + blockIdx * blockFormer);

        // pipe init buffer
        uint64_t inQueueXSize = (bFormer * row + B16_BLOCK_ALIGN_NUM - 1) / B16_BLOCK_ALIGN_NUM * B16_BLOCK_ALIGN_NUM *
                                TRANSPOSE_C0_SIZE * FLOAT_SIZE;
        uint64_t inQueueGammaSize =
            (row + GAMMA_NUM_PER_BLOCK - 1) / GAMMA_NUM_PER_BLOCK * GAMMA_NUM_PER_BLOCK * FLOAT_SIZE;
        uint64_t alignB = (bFormer + B16_BLOCK_ALIGN_NUM - 1) / B16_BLOCK_ALIGN_NUM * B16_BLOCK_ALIGN_NUM;
        pipe.InitBuffer(inQueueX, QUEUE_DEPTH, inQueueXSize);
        pipe.InitBuffer(inQueueGamma, 1, inQueueGammaSize);
        pipe.InitBuffer(inQueueBeta, 1, inQueueGammaSize);
        pipe.InitBuffer(outQueueY, 1, inQueueXSize);
        pipe.InitBuffer(outQueueMean, 1, alignB * TRANSPOSE_C0_SIZE * FLOAT_SIZE);
        pipe.InitBuffer(outQueueRstd, 1, alignB * TRANSPOSE_C0_SIZE * FLOAT_SIZE);
        pipe.InitBuffer(tmpBuf, inQueueXSize);
    }
    __aicore__ inline void Process()
    {
        uint64_t ubLoopCount;
        uint64_t ubTailLoopBlockLength;
        if (blockIdx < (numBlocks - 1)) {
            ubLoopCount = ubLoopOfFormerBlock;
            ubTailLoopBlockLength = ubTailOfFormerBlock;
        } else if (blockIdx == (numBlocks - 1)) {
            ubLoopCount = ubLoopOfTailBlock;
            ubTailLoopBlockLength = ubTailOfTailBlock;
        } else {
            return;
        }
        colLength = ubFormer;
        CalcGeneralParams();
        LocalTensor<Tweight> gammaLocal = inQueueGamma.AllocTensor<Tweight>();
        LocalTensor<Tweight> betaLocal = inQueueBeta.AllocTensor<Tweight>();
        CopyInGammaBeta(gammaLocal, betaLocal);
        // do baisc block
        for (uint64_t loopIdx = 0; loopIdx < ubLoopCount; loopIdx++) {
            if (loopIdx == (ubLoopCount - 1)) {
                colLength = ubTailLoopBlockLength;
                CalcGeneralParams();
            }
            xGmOffset = loopIdx * ubFormer * row;
            meanGmOffset = loopIdx * ubFormer;
            ProcessBasicBlock();
        }
        inQueueGamma.FreeTensor(gammaLocal);
        inQueueBeta.FreeTensor(betaLocal);
    }

private:
    __aicore__ inline void CalcGeneralParams()
    {
        bFormerFactor = (colLength + TRANSPOSE_C0_SIZE - 1) / TRANSPOSE_C0_SIZE;
        formerLoops = colLength - TRANSPOSE_C0_SIZE * (bFormerFactor - 1);
        calcXElements = row * bFormerFactor * TRANSPOSE_C0_SIZE;
        rowLineElements = bFormerFactor * TRANSPOSE_C0_SIZE;
        rFormerAxisAlign = (bFormerFactor * row + X_NUM_PER_BLOCK - 1) / X_NUM_PER_BLOCK * X_NUM_PER_BLOCK;
        bTailFactor = bFormerFactor - 1;
        rTailAxisAlign = (bTailFactor * row + X_NUM_PER_BLOCK - 1) / X_NUM_PER_BLOCK * X_NUM_PER_BLOCK;
    }

    template <typename T_COPY>
    __aicore__ inline void CopyInPad(LocalTensor<T_COPY>& dstTensor, GlobalTensor<T_COPY> inGm, uint64_t ubLoopGmOffset)
    {
        uint64_t ubOffset = 0;
        uint64_t gmOffset = 0;
        DataCopyPadExtParams<T_COPY> padParams;
        DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = row * bFormerFactor * sizeof(T_COPY);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = rFormerAxisAlign - row * bFormerFactor;
        padParams.paddingValue = 0;
        for (uint32_t i = 0; i < TRANSPOSE_C0_SIZE; i++) {
            if (i < formerLoops) {
                DataCopyPad(dstTensor[ubOffset], inGm[ubLoopGmOffset + gmOffset], intriParams, padParams);
                gmOffset += bFormerFactor * row;
                ubOffset += rFormerAxisAlign;
            } else {
                if (bTailFactor > 0) {
                    intriParams.blockLen = row * bTailFactor * sizeof(T_COPY);
                    padParams.rightPadding = rTailAxisAlign - row * bTailFactor;
                    DataCopyPad(dstTensor[ubOffset], inGm[ubLoopGmOffset + gmOffset], intriParams, padParams);
                    gmOffset += bTailFactor * row;
                    ubOffset += rFormerAxisAlign;
                }
                if ((rFormerAxisAlign - rTailAxisAlign) > 0) {
                    Duplicate<T_COPY>(
                        dstTensor[i * rFormerAxisAlign + rTailAxisAlign], 0.0, (rFormerAxisAlign - rTailAxisAlign));
                }
            }
        }
    }

    __aicore__ inline void CopyInGammaBeta(LocalTensor<Tweight>& gammaTensor, LocalTensor<Tweight>& betaTensor)
    {
        // 搬入inputGamma和inputBeta并cast
        DataCopyExtParams copyParams;
        DataCopyPadExtParams<Tweight> padParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = row * sizeof(Tweight);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.paddingValue = 0;
        uint64_t localOffset = (row + GAMMA_NUM_PER_BLOCK - 1) / GAMMA_NUM_PER_BLOCK * GAMMA_NUM_PER_BLOCK;
        padParams.rightPadding = localOffset - row;
        if (nullptrGamma == 0) {
            if constexpr (std::is_same<Tweight, float>::value) {
                localOffset = 0;
            }
            DataCopyPad(gammaTensor[localOffset], gammaGm, copyParams, padParams);
            inQueueGamma.EnQue(gammaTensor);
            inQueueGamma.DeQue<Tweight>();
            if constexpr (std::is_same<Tweight, float>::value) {
                gammaFp32 = gammaTensor;
            } else {
                gammaFp32 = gammaTensor.template ReinterpretCast<float>();
                Cast(gammaFp32, gammaTensor[localOffset], RoundMode::CAST_NONE, localOffset);
            }
        }
        if (nullptrBeta == 0) {
            if constexpr (std::is_same<Tweight, float>::value) {
                localOffset = 0;
            }
            DataCopyPad(betaTensor[localOffset], betaGm, copyParams, padParams);
            inQueueBeta.EnQue(betaTensor);
            inQueueBeta.DeQue<Tweight>();
            if constexpr (std::is_same<Tweight, float>::value) {
                betaFp32 = betaTensor;
            } else {
                betaFp32 = betaTensor.template ReinterpretCast<float>();
                Cast(betaFp32, betaTensor[localOffset], RoundMode::CAST_NONE, localOffset);
            }
        }
    }

    template <typename T_TRANS>
    __aicore__ inline void DoTranspose(LocalTensor<T_TRANS>& dstTensor, LocalTensor<T_TRANS>& srcTensor)
    {
        /*
        tiling限制repeat不大于255
        支持float16,float32
        */
        // 每行数据对齐后的size
        __ubuf__ T_TRANS* srcAddr = (__ubuf__ T_TRANS*)srcTensor.GetPhyAddr();
        __ubuf__ T_TRANS* dstAddr = (__ubuf__ T_TRANS*)dstTensor.GetPhyAddr();
        __ubuf__ T_TRANS* srcLocalList[TRANSPOSE_C0_SIZE];
        __ubuf__ T_TRANS* dstLocalList[TRANSPOSE_C0_SIZE];
        for (uint32_t i = 0; i < TRANSPOSE_C0_SIZE; i++) {
            srcLocalList[i] = srcAddr + rFormerAxisAlign * i;
            if constexpr (std::is_same<T_TRANS, float>::value) {
                dstLocalList[i] = dstAddr + B32_BLOCK_ALIGN_NUM * i;
            } else {
                dstLocalList[i] = dstAddr + B16_BLOCK_ALIGN_NUM * i;
            }
        }
        struct TransDataTo5HDParams transDataParams;
        if constexpr (std::is_same<T_TRANS, float>::value) {
            transDataParams.repeatTimes = rFormerAxisAlign / B32_BLOCK_ALIGN_NUM;
        } else {
            transDataParams.repeatTimes = rFormerAxisAlign / B16_BLOCK_ALIGN_NUM;
        }
        transDataParams.srcRepStride = 1;
        transDataParams.dstRepStride = TRANSPOSE_C0_SIZE;
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        }
        TransDataTo5HDImpl(dstLocalList, srcLocalList, transDataParams);
    }

    template <typename T_RESHAPE>
    __aicore__ inline void DoReshape(LocalTensor<T_RESHAPE>& dstTensor, LocalTensor<T_RESHAPE>& srcTensor)
    {
        /*
        支持fp32，fp16
        */
        // 一个repeat处理（128 / IN_NUM_PER_BLOCK）行数据
        uint32_t repeatTimes = row / BLOCK_NUM_PER_REP;
        uint32_t remainRepeat = row % BLOCK_NUM_PER_REP;
        uint32_t mask = 0;
        uint32_t lineBlockNum = 0;
        if constexpr (std::is_same<T_RESHAPE, float>::value) {
            mask = B32_BLOCK_ALIGN_NUM * BLOCK_NUM_PER_REP;
            lineBlockNum = TRANSPOSE_C0_SIZE / B32_BLOCK_ALIGN_NUM;
        } else {
            mask = B16_BLOCK_ALIGN_NUM * BLOCK_NUM_PER_REP;
            lineBlockNum = TRANSPOSE_C0_SIZE / B16_BLOCK_ALIGN_NUM;
        }
        if ((bFormerFactor * BLOCK_NUM_PER_REP * lineBlockNum) < MAX_REP_NUM) {
            if (repeatTimes) {
                for (uint32_t i = 0; i < bFormerFactor; i++) {
                    Copy(
                        dstTensor[i * TRANSPOSE_C0_SIZE], srcTensor[i * TRANSPOSE_C0_SIZE * row], mask, repeatTimes,
                        {(uint16_t)(bFormerFactor * lineBlockNum), (uint16_t)lineBlockNum,
                         (uint8_t)(BLOCK_NUM_PER_REP * bFormerFactor * lineBlockNum),
                         (uint8_t)(BLOCK_NUM_PER_REP * lineBlockNum)});
                    if constexpr (std::is_same<T_RESHAPE, float>::value) {
                        Copy(
                            dstTensor[i * TRANSPOSE_C0_SIZE + B32_BLOCK_ALIGN_NUM],
                            srcTensor[i * TRANSPOSE_C0_SIZE * row + B32_BLOCK_ALIGN_NUM], mask, repeatTimes,
                            {(uint16_t)(bFormerFactor * lineBlockNum), (uint16_t)lineBlockNum,
                             (uint8_t)(BLOCK_NUM_PER_REP * bFormerFactor * lineBlockNum),
                             (uint8_t)(BLOCK_NUM_PER_REP * lineBlockNum)});
                    }
                }
            }
            if (remainRepeat) {
                if constexpr (std::is_same<T_RESHAPE, float>::value) {
                    mask = remainRepeat * B32_BLOCK_ALIGN_NUM;
                } else {
                    mask = remainRepeat * B16_BLOCK_ALIGN_NUM;
                }
                for (uint32_t i = 0; i < bFormerFactor; i++) {
                    Copy(
                        dstTensor
                            [i * TRANSPOSE_C0_SIZE +
                             repeatTimes * TRANSPOSE_C0_SIZE * BLOCK_NUM_PER_REP * bFormerFactor],
                        srcTensor[i * TRANSPOSE_C0_SIZE * row + repeatTimes * TRANSPOSE_C0_SIZE * BLOCK_NUM_PER_REP],
                        mask, 1, {(uint16_t)(bFormerFactor * lineBlockNum), (uint16_t)lineBlockNum, 0, 0});
                    if constexpr (std::is_same<T_RESHAPE, float>::value) {
                        Copy(
                            dstTensor
                                [i * TRANSPOSE_C0_SIZE + B32_BLOCK_ALIGN_NUM +
                                 repeatTimes * TRANSPOSE_C0_SIZE * BLOCK_NUM_PER_REP * bFormerFactor],
                            srcTensor
                                [i * TRANSPOSE_C0_SIZE * row + B32_BLOCK_ALIGN_NUM +
                                 repeatTimes * TRANSPOSE_C0_SIZE * BLOCK_NUM_PER_REP],
                            mask, 1, {(uint16_t)(bFormerFactor * lineBlockNum), (uint16_t)lineBlockNum, 0, 0});
                    }
                }
            }
        } else {
            DataCopyParams copyParams;
            copyParams.blockCount = bFormerFactor;
            copyParams.blockLen = lineBlockNum;
            copyParams.srcStride = (row - 1) * copyParams.blockLen;
            copyParams.dstStride = 0;
            for (uint32_t i1 = 0; i1 < row; i1++) {
                DataCopy(
                    dstTensor[i1 * bFormerFactor * TRANSPOSE_C0_SIZE], srcTensor[i1 * TRANSPOSE_C0_SIZE], copyParams);
            }
        }
    }

    __aicore__ inline void DoReduce(LocalTensor<float>& dstTensor1, LocalTensor<float>& srcTensor1)
    {
        /*
        srcTensor为reduce之前的Tensor: row * rowLineElements
        dstTensor为存放reduce结果的tensor: rowLineElements
        */
        uint64_t nowRows = row;
        if (nowRows == 1) {
            Adds<float>(dstTensor1, srcTensor1, 0, rowLineElements);
            PipeBarrier<PIPE_V>();
            return;
        }
        // row为非二次幂，先将二次幂差值行加到前面
        if (dichotomizeAddDiffSize != 0) {
            Add(srcTensor1, srcTensor1, srcTensor1[(nowRows - dichotomizeAddDiffSize) * rowLineElements],
                dichotomizeAddDiffSize * rowLineElements);
            PipeBarrier<PIPE_V>();
            nowRows = nowRows - dichotomizeAddDiffSize;
        }
        while (nowRows > 1) {
            nowRows = nowRows / TWO_NUM;
            if (nowRows == 1) {
                Add(dstTensor1, srcTensor1, srcTensor1[rowLineElements], rowLineElements);
            } else {
                Add(srcTensor1, srcTensor1, srcTensor1[nowRows * rowLineElements], nowRows * rowLineElements);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void DoSub(LocalTensor<float>& dstTensor1, LocalTensor<float>& src1Tensor1)
    {
        /*
        dst复用src0，大小为row * rowLineElements
        src1大小为rowLineElements
        做inline broadcast的sub计算
        */
        if (((rowLineElements / ELEM_PER_REP_FP32) < row) && (rowLineElements < (MAX_REP_NUM * BLOCK_NUM_PER_REP))) {
            for (uint32_t i = 0; i < (rowLineElements / ELEM_PER_REP_FP32); i++) {
                Sub(dstTensor1[i * ELEM_PER_REP_FP32], dstTensor1[i * ELEM_PER_REP_FP32],
                    src1Tensor1[i * ELEM_PER_REP_FP32], ELEM_PER_REP_FP32, row,
                    {1, 1, 1, (uint8_t)(rowLineElements / BLOCK_NUM_PER_REP),
                     (uint8_t)(rowLineElements / BLOCK_NUM_PER_REP), 0});
            }
            if (rowLineElements % ELEM_PER_REP_FP32 > 0) {
                Sub(dstTensor1[rowLineElements / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32],
                    dstTensor1[rowLineElements / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32],
                    src1Tensor1[rowLineElements / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32],
                    rowLineElements % ELEM_PER_REP_FP32, row,
                    {1, 1, 1, (uint8_t)(rowLineElements / BLOCK_NUM_PER_REP),
                     (uint8_t)(rowLineElements / BLOCK_NUM_PER_REP), 0});
            }
        } else {
            for (uint64_t i = 0; i < row; i++) {
                Sub(dstTensor1[i * rowLineElements], dstTensor1[i * rowLineElements], src1Tensor1, rowLineElements);
            }
        }
    }

    __aicore__ inline void DoDiv(
        LocalTensor<float>& dstTensor, LocalTensor<float>& src0Tensor, LocalTensor<float>& src1Tensor1)
    {
        /*
        src0Tensor为置1的一个block的tensor
        src1Tensor1和dstTensor大小为rowLineElements
        */
        uint64_t repeatTimes = rowLineElements / ELEM_PER_REP_FP32;
        uint64_t repeatRemain = rowLineElements % ELEM_PER_REP_FP32;
        if (repeatTimes > 0) {
            Div(dstTensor, src0Tensor, src1Tensor1, ELEM_PER_REP_FP32, repeatTimes,
                {1, 0, 1, BLOCK_NUM_PER_REP, 0, BLOCK_NUM_PER_REP});
        }
        if (repeatRemain > 0) {
            Div(dstTensor[ELEM_PER_REP_FP32 * repeatTimes], src0Tensor, src1Tensor1[ELEM_PER_REP_FP32 * repeatTimes],
                repeatRemain, 1, {1, 0, 1, 0, 0, 0});
        }
    }

    __aicore__ inline void DoMul(LocalTensor<float>& dstTensor1, LocalTensor<float>& src1Tensor1)
    {
        /*
        dst复用src0，大小为row * rowLineElements
        src1大小为rowLineElements
        做inline broadcast的mul计算
        */
        if (((rowLineElements / ELEM_PER_REP_FP32) < row) && (rowLineElements < (MAX_REP_NUM * BLOCK_NUM_PER_REP))) {
            for (uint32_t i = 0; i < (rowLineElements / ELEM_PER_REP_FP32); i++) {
                Mul(dstTensor1[i * ELEM_PER_REP_FP32], dstTensor1[i * ELEM_PER_REP_FP32],
                    src1Tensor1[i * ELEM_PER_REP_FP32], ELEM_PER_REP_FP32, row,
                    {1, 1, 1, (uint8_t)(rowLineElements / BLOCK_NUM_PER_REP),
                     (uint8_t)(rowLineElements / BLOCK_NUM_PER_REP), 0});
            }
            if (rowLineElements % ELEM_PER_REP_FP32 > 0) {
                Mul(dstTensor1[rowLineElements / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32],
                    dstTensor1[rowLineElements / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32],
                    src1Tensor1[rowLineElements / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32],
                    rowLineElements % ELEM_PER_REP_FP32, row,
                    {1, 1, 1, (uint8_t)(rowLineElements / BLOCK_NUM_PER_REP),
                     (uint8_t)(rowLineElements / BLOCK_NUM_PER_REP), 0});
            }
        } else {
            for (uint64_t i = 0; i < row; i++) {
                Mul(dstTensor1[i * rowLineElements], dstTensor1[i * rowLineElements], src1Tensor1, rowLineElements);
            }
        }
    }

    __aicore__ inline void DoMulGamma(LocalTensor<float>& dstTensor1)
    {
        if (nullptrGamma == 1) {
            return;
        }
        for (uint64_t i = 0; i < row; i++) {
            float gammaValue = gammaFp32.GetValue(i);
            Muls(dstTensor1[i * rowLineElements], dstTensor1[i * rowLineElements], gammaValue, rowLineElements);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DoAddBeta(LocalTensor<float>& dstTensor1)
    {
        if (nullptrBeta == 1) {
            return;
        }
        for (uint64_t i = 0; i < row; i++) {
            float betaValue = betaFp32.GetValue(i);
            Adds(dstTensor1[i * rowLineElements], dstTensor1[i * rowLineElements], betaValue, rowLineElements);
        }
        PipeBarrier<PIPE_V>();
    }

    template <typename T_POST_RESHAPE>
    __aicore__ inline void DoPostReshape(LocalTensor<T_POST_RESHAPE>& dstTensor, LocalTensor<T_POST_RESHAPE>& srcTensor)
    {
        /*
        支持fp32，fp16
        */
        // 一个repeat处理（128 / IN_NUM_PER_BLOCK）行数据
        uint32_t repeatTimes = row / BLOCK_NUM_PER_REP;
        uint32_t remainRepeat = row % BLOCK_NUM_PER_REP;
        uint32_t mask = 0;
        uint32_t lineBlockNum = 0;
        if constexpr (std::is_same<T_POST_RESHAPE, float>::value) {
            mask = B32_BLOCK_ALIGN_NUM * BLOCK_NUM_PER_REP;
            lineBlockNum = TRANSPOSE_C0_SIZE / B32_BLOCK_ALIGN_NUM;
        } else {
            mask = B16_BLOCK_ALIGN_NUM * BLOCK_NUM_PER_REP;
            lineBlockNum = TRANSPOSE_C0_SIZE / B16_BLOCK_ALIGN_NUM;
        }
        if ((bFormerFactor * BLOCK_NUM_PER_REP * lineBlockNum) < MAX_REP_NUM) {
            if (repeatTimes) {
                for (uint32_t i = 0; i < bFormerFactor; i++) {
                    Copy(
                        dstTensor[i * TRANSPOSE_C0_SIZE * row], srcTensor[i * TRANSPOSE_C0_SIZE], mask, repeatTimes,
                        {(uint16_t)lineBlockNum, (uint16_t)(bFormerFactor * lineBlockNum),
                         (uint8_t)(BLOCK_NUM_PER_REP * lineBlockNum),
                         (uint8_t)(BLOCK_NUM_PER_REP * bFormerFactor * lineBlockNum)});
                    if constexpr (std::is_same<T_POST_RESHAPE, float>::value) {
                        Copy(
                            dstTensor[i * TRANSPOSE_C0_SIZE * row + B32_BLOCK_ALIGN_NUM],
                            srcTensor[i * TRANSPOSE_C0_SIZE + B32_BLOCK_ALIGN_NUM], mask, repeatTimes,
                            {(uint16_t)lineBlockNum, (uint16_t)(bFormerFactor * lineBlockNum),
                             (uint8_t)(BLOCK_NUM_PER_REP * lineBlockNum),
                             (uint8_t)(BLOCK_NUM_PER_REP * bFormerFactor * lineBlockNum)});
                    }
                }
            }
            if (remainRepeat) {
                if constexpr (std::is_same<T_POST_RESHAPE, float>::value) {
                    mask = remainRepeat * B32_BLOCK_ALIGN_NUM;
                } else {
                    mask = remainRepeat * B16_BLOCK_ALIGN_NUM;
                }
                for (uint32_t i = 0; i < bFormerFactor; i++) {
                    Copy(
                        dstTensor[i * TRANSPOSE_C0_SIZE * row + repeatTimes * TRANSPOSE_C0_SIZE * BLOCK_NUM_PER_REP],
                        srcTensor
                            [i * TRANSPOSE_C0_SIZE +
                             repeatTimes * TRANSPOSE_C0_SIZE * BLOCK_NUM_PER_REP * bFormerFactor],
                        mask, 1, {(uint16_t)lineBlockNum, (uint16_t)(bFormerFactor * lineBlockNum), 0, 0});
                    if constexpr (std::is_same<T_POST_RESHAPE, float>::value) {
                        Copy(
                            dstTensor
                                [i * TRANSPOSE_C0_SIZE * row + B32_BLOCK_ALIGN_NUM +
                                 repeatTimes * TRANSPOSE_C0_SIZE * BLOCK_NUM_PER_REP],
                            srcTensor
                                [i * TRANSPOSE_C0_SIZE + B32_BLOCK_ALIGN_NUM +
                                 repeatTimes * TRANSPOSE_C0_SIZE * BLOCK_NUM_PER_REP * bFormerFactor],
                            mask, 1, {(uint16_t)lineBlockNum, (uint16_t)(bFormerFactor * lineBlockNum), 0, 0});
                    }
                }
            }
        } else {
            DataCopyParams copyParams;
            copyParams.blockCount = bFormerFactor;
            copyParams.blockLen = lineBlockNum;
            copyParams.srcStride = 0;
            copyParams.dstStride = (row - 1) * copyParams.blockLen;
            for (uint32_t i = 0; i < row; i++) {
                DataCopy(
                    dstTensor[i * TRANSPOSE_C0_SIZE], srcTensor[i * bFormerFactor * TRANSPOSE_C0_SIZE], copyParams);
            }
        }
    }

    template <typename T_POST_TRANS>
    __aicore__ inline void DoPostTranspose(LocalTensor<T_POST_TRANS>& dstTensor, LocalTensor<T_POST_TRANS>& srcTensor)
    {
        /*
        tiling限制repeat不大于255
        支持float16,float32
        反向的转置过程，会使行对齐为16的倍数
        */
        // 每行数据对齐后的size
        uint32_t lineAlignSize = 0;
        lineAlignSize = (bFormerFactor * row + TRANSPOSE_C0_SIZE - 1) / TRANSPOSE_C0_SIZE * TRANSPOSE_C0_SIZE;
        __ubuf__ T_POST_TRANS* srcAddr = (__ubuf__ T_POST_TRANS*)srcTensor.GetPhyAddr();
        __ubuf__ T_POST_TRANS* dstAddr = (__ubuf__ T_POST_TRANS*)dstTensor.GetPhyAddr();
        __ubuf__ T_POST_TRANS* srcLocalList[TRANSPOSE_C0_SIZE];
        __ubuf__ T_POST_TRANS* dstLocalList[TRANSPOSE_C0_SIZE];
        for (uint32_t i = 0; i < TRANSPOSE_C0_SIZE; i++) {
            srcLocalList[i] = srcAddr + TRANSPOSE_C0_SIZE * i;
            if constexpr (std::is_same<T_POST_TRANS, float>::value) {
                dstLocalList[i] = dstAddr + lineAlignSize * (i / TWO_NUM) + B32_BLOCK_ALIGN_NUM * (i % TWO_NUM);
            } else {
                dstLocalList[i] = dstAddr + lineAlignSize * i;
            }
        }
        struct TransDataTo5HDParams transDataParams;
        if constexpr (std::is_same<T_POST_TRANS, float>::value) {
            transDataParams.repeatTimes = lineAlignSize / B32_BLOCK_ALIGN_NUM / TWO_NUM;
            transDataParams.srcRepStride = TRANSPOSE_C0_SIZE * TWO_NUM;
            transDataParams.dstRepStride = TWO_NUM;
        } else {
            transDataParams.repeatTimes = lineAlignSize / B16_BLOCK_ALIGN_NUM;
            transDataParams.srcRepStride = TRANSPOSE_C0_SIZE;
            transDataParams.dstRepStride = 1;
        }
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        }
        TransDataTo5HDImpl(dstLocalList, srcLocalList, transDataParams);
        if constexpr (std::is_same<T_POST_TRANS, float>::value) {
            for (uint32_t i = 0; i < TRANSPOSE_C0_SIZE; i++) {
                srcLocalList[i] = srcAddr + TRANSPOSE_C0_SIZE * i + B32_BLOCK_ALIGN_NUM;
                dstLocalList[i] = dstAddr + B32_BLOCK_ALIGN_NUM * lineAlignSize + lineAlignSize * (i / TWO_NUM) +
                                  B32_BLOCK_ALIGN_NUM * (i % TWO_NUM);
            }
            TransDataTo5HDImpl(dstLocalList, srcLocalList, transDataParams);
        }
    }

    __aicore__ inline void DoMeanOrRstdTranspose(LocalTensor<float>& dstTensor1, LocalTensor<float>& srcTensor)
    {
        /*
        tiling限制repeat不大于255
        只支持fp32
        */
        __ubuf__ float* srcAddr = (__ubuf__ float*)srcTensor.GetPhyAddr();
        __ubuf__ float* dstAddr = (__ubuf__ float*)dstTensor1.GetPhyAddr();
        __ubuf__ float* srcLocalList[TRANSPOSE_C0_SIZE];
        __ubuf__ float* dstLocalList[TRANSPOSE_C0_SIZE];
        struct TransDataTo5HDParams transDataParams;
        transDataParams.repeatTimes = (bFormerFactor + TRANSPOSE_C0_SIZE - 1) / TRANSPOSE_C0_SIZE;
        transDataParams.srcRepStride = TRANSPOSE_C0_SIZE * (FLOAT_SIZE / HALF_SIZE);
        transDataParams.dstRepStride = FLOAT_SIZE / HALF_SIZE;
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        }
        // fp32数据需要处理上下两部分
        for (uint32_t i2 = 0; i2 < (FLOAT_SIZE / HALF_SIZE); i2++) {
            for (uint32_t j = 0; j < TRANSPOSE_C0_SIZE; j++) {
                srcLocalList[j] = srcAddr + FP32_TRANSPOSE_DST_SIZE * i2 + TRANSPOSE_C0_SIZE * j;
            }
            for (uint32_t k = 0; k < FP32_TRANSPOSE_DST_SIZE; k++) {
                dstLocalList[(FLOAT_SIZE / HALF_SIZE) * k] =
                    dstAddr + transDataParams.repeatTimes * TRANSPOSE_C0_SIZE * k +
                    transDataParams.repeatTimes * TRANSPOSE_C0_SIZE * FP32_TRANSPOSE_DST_SIZE * i2;
                dstLocalList[(FLOAT_SIZE / HALF_SIZE) * k + 1] =
                    dstAddr + transDataParams.repeatTimes * TRANSPOSE_C0_SIZE * k + FP32_TRANSPOSE_DST_SIZE +
                    transDataParams.repeatTimes * TRANSPOSE_C0_SIZE * FP32_TRANSPOSE_DST_SIZE * i2;
            }
            TransDataTo5HDImpl(dstLocalList, srcLocalList, transDataParams);
        }
    }

    __aicore__ inline void CopyOutMeanOrRstd(GlobalTensor<float> outGm, LocalTensor<float>& srcTensor)
    {
        /*
        使用DataCopyPad搬运
        */
        uint32_t curFactor = bFormerFactor;
        uint64_t ubOffset = 0;
        uint64_t gmOffset = 0;
        // 每行数据对齐后的size
        uint32_t lineAlignSize = 0;
        lineAlignSize = (bFormerFactor + TRANSPOSE_C0_SIZE - 1) / TRANSPOSE_C0_SIZE * TRANSPOSE_C0_SIZE;

        DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = bFormerFactor * sizeof(float);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        for (uint32_t i = 0; i < TRANSPOSE_C0_SIZE; i++) {
            if (i < formerLoops) {
                curFactor = bFormerFactor;
            } else {
                if (bTailFactor > 0) {
                    intriParams.blockLen = bTailFactor * sizeof(float);
                    curFactor = bTailFactor;
                } else {
                    break;
                }
            }
            DataCopyPad(outGm[meanGmOffset + gmOffset], srcTensor[ubOffset], intriParams);
            // gm偏移连续，每次偏curFactor
            gmOffset += curFactor;
            // ub偏移对齐，每次固定lineAlignSize
            ubOffset += lineAlignSize;
        }
    }

    __aicore__ inline void CopyOutY(LocalTensor<Tfm>& srcTensor)
    {
        uint32_t curFactor = bFormerFactor;
        uint64_t ubOffset = 0;
        uint64_t gmOffset = 0;
        // 每行数据对齐后的size
        uint32_t lineAlignSize = 0;
        lineAlignSize = (bFormerFactor * row + TRANSPOSE_C0_SIZE - 1) / TRANSPOSE_C0_SIZE * TRANSPOSE_C0_SIZE;

        DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = row * bFormerFactor * sizeof(Tfm);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        for (uint32_t i = 0; i < TRANSPOSE_C0_SIZE; i++) {
            if (i < formerLoops) {
                curFactor = bFormerFactor;
            } else {
                if (bTailFactor > 0) {
                    intriParams.blockLen = row * bTailFactor * sizeof(Tfm);
                    curFactor = bTailFactor;
                } else {
                    break;
                }
            }
            DataCopyPad(yGm[xGmOffset + gmOffset], srcTensor[ubOffset], intriParams);
            // gm偏移连续，每次偏curFactor * row
            gmOffset += curFactor * row;
            // ub偏移对齐，每次固定lineAlignSize
            ubOffset += lineAlignSize;
        }
    }

    __aicore__ inline void ProcessBasicBlock()
    {
        LocalTensor<Tfm> xLocal = inQueueX.AllocTensor<Tfm>();
        CopyInPad<Tfm>(xLocal, xGm, xGmOffset);
        inQueueX.EnQue(xLocal);
        inQueueX.DeQue<Tfm>();

        LocalTensor<Tfm> transposeTemp = tmpBuf.Get<Tfm>();
        if constexpr (std::is_same<Tfm, bfloat16_t>::value) {
            LocalTensor<half> transposeTempHalf = transposeTemp.template ReinterpretCast<half>();
            LocalTensor<half> xLocalHalf = xLocal.template ReinterpretCast<half>();
            DoTranspose<half>(transposeTempHalf, xLocalHalf);
        } else {
            DoTranspose<Tfm>(transposeTemp, xLocal);
        }
        PipeBarrier<PIPE_V>();

        if constexpr (std::is_same<Tfm, float>::value) {
            DoReshape<Tfm>(xLocal, transposeTemp);
            PipeBarrier<PIPE_V>();
            xLocalFp32 = xLocal;
        } else {
            LocalTensor<Tfm> xLocalSecond = xLocal[calcXElements];
            if constexpr (std::is_same<Tfm, bfloat16_t>::value) {
                LocalTensor<half> xLocalSecondRHalf = xLocalSecond.template ReinterpretCast<half>();
                LocalTensor<half> transposeTempRHalf = transposeTemp.template ReinterpretCast<half>();
                DoReshape<half>(xLocalSecondRHalf, transposeTempRHalf);
            } else {
                DoReshape<Tfm>(xLocalSecond, transposeTemp);
            }
            PipeBarrier<PIPE_V>();
            xLocalFp32 = xLocal.template ReinterpretCast<float>();
            Cast(xLocalFp32, xLocalSecond, RoundMode::CAST_NONE, calcXElements);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> mulTempTensor1 = tmpBuf.Get<float>();
        // xLocalFp32需要驻留
        Muls(mulTempTensor1, xLocalFp32, coefficient, calcXElements);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> outTempTensor = outQueueRstd.AllocTensor<float>();
        DoReduce(outTempTensor, mulTempTensor1);
        PipeBarrier<PIPE_V>();

        DoSub(xLocalFp32, outTempTensor);
        LocalTensor<float> outMeanTensor1 = outQueueMean.AllocTensor<float>();
        DoMeanOrRstdTranspose(outMeanTensor1, outTempTensor);
        outQueueRstd.FreeTensor(outTempTensor);
        outQueueMean.EnQue(outMeanTensor1);
        outQueueMean.DeQue<float>();
        CopyOutMeanOrRstd(meanGm, outMeanTensor1);
        outQueueMean.FreeTensor(outMeanTensor1);

        // do mul2, xLocalFp32需要驻留
        LocalTensor<float> mul2TempTensor1 = tmpBuf.Get<float>();
        Mul(mul2TempTensor1, xLocalFp32, xLocalFp32, calcXElements);
        PipeBarrier<PIPE_V>();
        Muls(mul2TempTensor1, mul2TempTensor1, coefficient, calcXElements);
        PipeBarrier<PIPE_V>();

        // do reduce1
        LocalTensor<float> outMTensor = outQueueMean.AllocTensor<float>();
        DoReduce(outMTensor, mul2TempTensor1);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> tempTensor1 = tmpBuf.Get<float>();
        Adds(tempTensor1, outMTensor, eps, rowLineElements);
        PipeBarrier<PIPE_V>();

        Sqrt(tempTensor1, tempTensor1, rowLineElements);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> oneTensor = outQueueY.AllocTensor<float>();
        Duplicate<float>(oneTensor, 1, B32_BLOCK_ALIGN_NUM);
        PipeBarrier<PIPE_V>();

        DoDiv(outMTensor, oneTensor, tempTensor1);
        PipeBarrier<PIPE_V>();
        outQueueY.FreeTensor(oneTensor);

        DoMul(xLocalFp32, outMTensor);
        PipeBarrier<PIPE_V>();
        // output rstd
        LocalTensor<float> outRstdTensor1 = outQueueRstd.AllocTensor<float>();
        DoMeanOrRstdTranspose(outRstdTensor1, outMTensor);
        outQueueMean.FreeTensor(outMTensor);
        outQueueRstd.EnQue(outRstdTensor1);
        outQueueRstd.DeQue<float>();
        CopyOutMeanOrRstd(rstdGm, outRstdTensor1);
        outQueueRstd.FreeTensor(outRstdTensor1);

        DoMulGamma(xLocalFp32);

        DoAddBeta(xLocalFp32);

        LocalTensor<Tfm> outputYTensor = outQueueY.AllocTensor<Tfm>();
        if constexpr (!IsSameType<Tfm, float>::value) {
            RoundMode b16RoundMode = IsSameType<Tfm, bfloat16_t>::value ? RoundMode::CAST_ROUND : RoundMode::CAST_NONE;
            Cast(outputYTensor, xLocalFp32, b16RoundMode, calcXElements);
            PipeBarrier<PIPE_V>();
            inQueueX.FreeTensor(xLocalFp32);
        }

        LocalTensor<Tfm> reshapeTemp = tmpBuf.Get<Tfm>();
        if constexpr (std::is_same<Tfm, bfloat16_t>::value) {
            LocalTensor<half> reshapeTempRHalf = reshapeTemp.template ReinterpretCast<half>();
            LocalTensor<half> outputYTensorRHalf = outputYTensor.template ReinterpretCast<half>();
            DoPostReshape<half>(reshapeTempRHalf, outputYTensorRHalf);
        } else if constexpr (std::is_same<Tfm, float>::value) {
            DoPostReshape<Tfm>(reshapeTemp, xLocalFp32);
            inQueueX.FreeTensor(xLocalFp32);
        } else {
            DoPostReshape<Tfm>(reshapeTemp, outputYTensor);
        }
        PipeBarrier<PIPE_V>();

        if constexpr (std::is_same<Tfm, bfloat16_t>::value) {
            LocalTensor<half> outputYTensorHalf = outputYTensor.template ReinterpretCast<half>();
            LocalTensor<half> reshapeTempHalf = reshapeTemp.template ReinterpretCast<half>();
            DoPostTranspose<half>(outputYTensorHalf, reshapeTempHalf);
        } else {
            DoPostTranspose<Tfm>(outputYTensor, reshapeTemp);
        }
        PipeBarrier<PIPE_V>();

        outQueueY.EnQue(outputYTensor);
        outQueueY.DeQue<Tfm>();
        CopyOutY(outputYTensor);
        outQueueY.FreeTensor(outputYTensor);
    }

private:
    constexpr static uint32_t QUEUE_DEPTH = 2;
    constexpr static uint32_t FLOAT_SIZE = 4;
    constexpr static uint32_t HALF_SIZE = 2;
    constexpr static uint32_t TRANSPOSE_C0_SIZE = 16;
    constexpr static uint32_t BLOCK = 32;
    constexpr static uint32_t X_NUM_PER_BLOCK = BLOCK / sizeof(Tfm);
    constexpr static uint32_t GAMMA_NUM_PER_BLOCK = BLOCK / sizeof(Tweight);
    constexpr static uint32_t MAX_REP_NUM = 255;
    constexpr static uint32_t ELEM_PER_REP_FP32 = 64;
    constexpr static uint32_t ELEM_PER_REP_FP16 = 128;
    constexpr static uint32_t BLOCK_NUM_PER_REP = 8;
    constexpr static uint32_t FP32_TRANSPOSE_DST_SIZE = 8;
    constexpr static uint32_t B32_BLOCK_ALIGN_NUM = 8;
    constexpr static uint32_t B16_BLOCK_ALIGN_NUM = 16;
    constexpr static uint32_t TWO_NUM = 2;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueGamma, inQueueBeta;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY, outQueueMean, outQueueRstd;
    TBuf<TPosition::VECCALC> tmpBuf;

    GlobalTensor<Tfm> yGm;
    GlobalTensor<Tfm> xGm;
    GlobalTensor<Tweight> betaGm;
    GlobalTensor<Tweight> gammaGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<float> meanGm;

    uint32_t blockIdx = GetBlockIdx();

    // calculate xGm and meanGm offset for ub loop
    uint64_t meanGmOffset = 0;
    uint64_t xGmOffset = 0;
    // in ub loop col size
    uint64_t colLength = 0;

    // x搬入时一行整块的对齐长度
    uint32_t rFormerAxisAlign = 0;
    // x搬入时尾块的借轴因子
    uint32_t bTailFactor = 0;
    // 整块搬入的循环次数
    uint32_t formerLoops = 0;
    // x搬入时一行尾块的对齐长度
    uint32_t rTailAxisAlign = 0;
    // 重排后x基本块的元素个数
    uint64_t calcXElements;
    // 重排后x做row Reduce后的元素个数
    uint64_t rowLineElements;
    // x搬入时整块的借轴因子
    uint32_t bFormerFactor = 0;

    LocalTensor<float> betaFp32;
    LocalTensor<float> gammaFp32;
    LocalTensor<float> xLocalFp32;
    // tilingData
    uint64_t row;
    uint64_t numBlocks;
    uint64_t blockFormer;
    uint64_t ubFormer;
    uint64_t col;
    uint64_t blockTail;
    uint64_t ubTailOfFormerBlock;
    uint64_t dichotomizeAddDiffSize;
    uint64_t bFormer;
    uint64_t ubLoopOfTailBlock;
    uint64_t ubLoopOfFormerBlock;
    uint64_t ubTailOfTailBlock;
    float coefficient = 0.0;
    float eps = 0.0;
    uint32_t nullptrBeta;
    uint32_t nullptrGamma;
};

} // namespace LayerNormV4
#endif // LAYER_NORM_V4_TRANSPOSE_H
