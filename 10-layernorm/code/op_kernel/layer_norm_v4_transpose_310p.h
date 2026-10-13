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
 * \file layer_norm_v4_transpose_310p.h
 * \brief
 */

#ifndef LAYER_NORM_V4_TRANSPOSE_310P_H
#define LAYER_NORM_V4_TRANSPOSE_310P_H

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
        col = tilingData->col;
        row = tilingData->row;
        numBlocks = tilingData->numBlocks;
        blockFormer = tilingData->blockFormer;
        blockTail = tilingData->blockTail;
        ubFormer = tilingData->ubFormer;
        bFormer = tilingData->bFormer;
        dichotomizeAddDiffSize = tilingData->dichotomizeAddDiffSize;
        ubLoopOfFormerBlock = tilingData->ubLoopOfFormerBlock;
        ubLoopOfTailBlock = tilingData->ubLoopOfTailBlock;
        ubTailOfFormerBlock = tilingData->ubTailOfFormerBlock;
        ubTailOfTailBlock = tilingData->ubTailOfTailBlock;
        eps = tilingData->eps;
        coefficient = tilingData->coefficient;
        nullptrGamma = tilingData->nullptrGamma;
        nullptrBeta = tilingData->nullptrBeta;
        // set global buffer
        xGm.SetGlobalBuffer((__gm__ Tfm*)x + blockIdx * blockFormer * row);
        gammaGm.SetGlobalBuffer((__gm__ Tweight*)gamma);
        betaGm.SetGlobalBuffer((__gm__ Tweight*)beta);

        yGm.SetGlobalBuffer((__gm__ Tfm*)y + blockIdx * blockFormer * row);
        meanGm.SetGlobalBuffer((__gm__ float*)mean + blockIdx * blockFormer);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockIdx * blockFormer);

        // pipe init buffer
        uint64_t inQueueXSize =
            (bFormer * row + X_NUM_PER_BLOCK - 1) / X_NUM_PER_BLOCK * X_NUM_PER_BLOCK * TRANSPOSE_C0_SIZE * FLOAT_SIZE;
        uint64_t inQueueGammaSize =
            (row + GAMMA_NUM_PER_BLOCK - 1) / GAMMA_NUM_PER_BLOCK * GAMMA_NUM_PER_BLOCK * FLOAT_SIZE;
        uint64_t alignB = (bFormer + TRANSPOSE_C0_SIZE - 1) / TRANSPOSE_C0_SIZE * TRANSPOSE_C0_SIZE;
        pipe.InitBuffer(inQueueX, QUEUE_DEPTH, inQueueXSize);
        pipe.InitBuffer(inQueueGamma, 1, inQueueGammaSize);
        pipe.InitBuffer(inQueueBeta, 1, inQueueGammaSize);
        pipe.InitBuffer(outQueueY, 1, inQueueXSize);
        pipe.InitBuffer(outQueueMean, 1, alignB * TRANSPOSE_C0_SIZE * FLOAT_SIZE);
        pipe.InitBuffer(outQueueRstd, 1, alignB * TRANSPOSE_C0_SIZE * FLOAT_SIZE);
        pipe.InitBuffer(tmpBuf, inQueueXSize);
        pipe.InitBuffer(overLapBuf, BLOCK);
    }
    __aicore__ inline void Process()
    {
        uint64_t ubLoopCount;
        uint64_t ubTailLoopBlockLength;
        if (blockIdx < (numBlocks - 1)) {
            ubLoopCount = ubLoopOfFormerBlock;
            ubTailLoopBlockLength = ubTailOfFormerBlock;
            isLastCore = false;
        } else if (blockIdx == (numBlocks - 1)) {
            ubLoopCount = ubLoopOfTailBlock;
            ubTailLoopBlockLength = ubTailOfTailBlock;
            isLastCore = true;
        } else {
            return;
        }
        colLength = ubFormer;
        CalcGeneralParams();
        CopyInGammaBeta();
        // do baisc block
        for (uint64_t loopIdx = 0; loopIdx < ubLoopCount; loopIdx++) {
            if (loopIdx == (ubLoopCount - 1)) {
                colLength = ubTailLoopBlockLength;
                CalcGeneralParams();
                isLastUbLoop = true;
            } else {
                isLastUbLoop = false;
            }
            xGmOffset = loopIdx * ubFormer * row;
            meanGmOffset = loopIdx * ubFormer;
            needOverLap = ((!isLastCore) && (isLastUbLoop));
            ProcessBasicBlock();
        }
    }

private:
    __aicore__ inline void CalcGeneralParams()
    {
        bFormerFactor = (colLength + TRANSPOSE_C0_SIZE - 1) / TRANSPOSE_C0_SIZE;
        rFormerAxisAlign = (bFormerFactor * row + X_NUM_PER_BLOCK - 1) / X_NUM_PER_BLOCK * X_NUM_PER_BLOCK;
        bTailFactor = bFormerFactor - 1;
        rTailAxisAlign = (bTailFactor * row + X_NUM_PER_BLOCK - 1) / X_NUM_PER_BLOCK * X_NUM_PER_BLOCK;
        formerLoops = colLength - TRANSPOSE_C0_SIZE * (bFormerFactor - 1);
        calcXElements = row * bFormerFactor * TRANSPOSE_C0_SIZE;
        rowLineElements = bFormerFactor * TRANSPOSE_C0_SIZE;

        uint64_t blockMask = 0;
        uint64_t remainRepeat = row % BLOCK_NUM_PER_REP;
        // 组装一个block的mask构成
        if (formerLoops < TRANSPOSE_C0_SIZE) {
            for (uint32_t i = 0; i < UINT16_BIT_SIZE; i++) {
                if (i < UINT16_BIT_SIZE - formerLoops) {
                    blockMask = ((blockMask << 1) + (uint64_t)1);
                } else {
                    blockMask = (blockMask << 1);
                }
            }
            for (uint32_t i = 0; i < (UINT64_BIT_SIZE / UINT16_BIT_SIZE); i++) {
                formerMask = ((formerMask << UINT16_BIT_SIZE) + blockMask);
            }
            if (remainRepeat > (UINT64_BIT_SIZE / UINT16_BIT_SIZE)) {
                for (uint32_t i = 0; i < (remainRepeat - (UINT64_BIT_SIZE / UINT16_BIT_SIZE)); i++) {
                    remainMaskHigh = ((remainMaskHigh << UINT16_BIT_SIZE) + blockMask);
                }
                remainMaskLow = formerMask;
            } else {
                for (uint32_t i = 0; i < remainRepeat; i++) {
                    remainMaskLow = ((remainMaskLow << UINT16_BIT_SIZE) + blockMask);
                }
                remainMaskHigh = 0;
            }
        }
    }

    __aicore__ inline void CopyInPad(LocalTensor<Tfm>& dstTensor)
    {
        uint32_t curFactor = bFormerFactor;
        uint64_t ubOffset = 0;
        uint64_t gmOffset = 0;
        DataCopyParams xCopyParams;
        xCopyParams.blockCount = 1;
        xCopyParams.srcStride = 0;
        xCopyParams.dstStride = 0;
        for (uint32_t i = 0; i < TRANSPOSE_C0_SIZE; i++) {
            if (i < formerLoops) {
                xCopyParams.blockLen = rFormerAxisAlign / X_NUM_PER_BLOCK;
                curFactor = bFormerFactor;
            } else {
                if (bTailFactor > 0) {
                    xCopyParams.blockLen = rTailAxisAlign / X_NUM_PER_BLOCK;
                    curFactor = bTailFactor;
                } else {
                    break;
                }
            }
            DataCopy(dstTensor[ubOffset], xGm[xGmOffset + gmOffset], xCopyParams);
            // gm偏移连续，每次偏curFactor * row
            gmOffset += curFactor * row;
            // ub偏移对齐，每次固定rFormerAxisAlign
            ubOffset += rFormerAxisAlign;
        }
    }

    __aicore__ inline void CopyInGammaBeta()
    {
        // 搬入inputGamma和inputBeta并cast
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = (row + GAMMA_NUM_PER_BLOCK - 1) / GAMMA_NUM_PER_BLOCK;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        uint64_t localOffset = copyParams.blockLen * GAMMA_NUM_PER_BLOCK;
        if (nullptrGamma == 0) {
            LocalTensor<Tweight> gammaLocal = inQueueGamma.AllocTensor<Tweight>();
            if constexpr (std::is_same<Tweight, float>::value) {
                localOffset = 0;
            }
            DataCopy(gammaLocal[localOffset], gammaGm, copyParams);
            inQueueGamma.EnQue(gammaLocal);
            inQueueGamma.DeQue<Tweight>();
            if constexpr (std::is_same<Tweight, float>::value) {
                gammaFp32 = gammaLocal;
            } else {
                gammaFp32 = gammaLocal.template ReinterpretCast<float>();
                Cast(gammaFp32, gammaLocal[localOffset], RoundMode::CAST_NONE, localOffset);
            }
        }
        if (nullptrBeta == 0) {
            LocalTensor<Tweight> betaLocal = inQueueBeta.AllocTensor<Tweight>();
            if constexpr (std::is_same<Tweight, float>::value) {
                localOffset = 0;
            }
            DataCopy(betaLocal[localOffset], betaGm, copyParams);
            inQueueBeta.EnQue(betaLocal);
            inQueueBeta.DeQue<Tweight>();
            if constexpr (std::is_same<Tweight, float>::value) {
                betaFp32 = betaLocal;
            } else {
                betaFp32 = betaLocal.template ReinterpretCast<float>();
                Cast(betaFp32, betaLocal[localOffset], RoundMode::CAST_NONE, localOffset);
            }
        }
    }

    __aicore__ inline void DoTranspose(LocalTensor<Tfm>& dstTensor, LocalTensor<Tfm>& srcTensor)
    {
        /*
        tiling限制repeat不大于255
        暂只支持bloat16和float16
        */
        __ubuf__ Tfm* srcAddr = (__ubuf__ Tfm*)srcTensor.GetPhyAddr();
        __ubuf__ Tfm* dstAddr = (__ubuf__ Tfm*)dstTensor.GetPhyAddr();
        __ubuf__ Tfm* srcLocalList[TRANSPOSE_C0_SIZE];
        __ubuf__ Tfm* dstLocalList[TRANSPOSE_C0_SIZE];
        for (uint32_t i = 0; i < TRANSPOSE_C0_SIZE; i++) {
            srcLocalList[i] = srcAddr + rFormerAxisAlign * i;
            dstLocalList[i] = dstAddr + TRANSPOSE_C0_SIZE * i;
        }
        struct TransDataTo5HDParams transDataParams;
        transDataParams.repeatTimes = rFormerAxisAlign / TRANSPOSE_C0_SIZE;
        transDataParams.srcRepStride = 1;
        transDataParams.dstRepStride = TRANSPOSE_C0_SIZE;
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        }
        TransDataTo5HDImpl(dstLocalList, srcLocalList, transDataParams);
    }

    __aicore__ inline void DoReshape(LocalTensor<Tfm>& dstTensor, LocalTensor<Tfm>& srcTensor)
    {
        /*
        仅支持16数据类型
        */
        // 一个repeat处理（128 / 16）行数据
        uint64_t repeatTimes = row / BLOCK_NUM_PER_REP;
        uint64_t remainRepeat = row % BLOCK_NUM_PER_REP;
        uint64_t mask = ELEM_PER_REP_FP16;
        // 清0处理
        if (formerLoops < TRANSPOSE_C0_SIZE) {
            if (repeatTimes) {
                uint64_t maskDup[2] = {formerMask, formerMask};
                Duplicate(
                    srcTensor[(bFormerFactor - 1) * TRANSPOSE_C0_SIZE * row], static_cast<Tfm>(0.0), maskDup,
                    repeatTimes, 1, BLOCK_NUM_PER_REP);
            }
            if (remainRepeat) {
                uint64_t maskDup[2] = {remainMaskLow, remainMaskHigh};
                Duplicate(
                    srcTensor[(bFormerFactor - 1) * TRANSPOSE_C0_SIZE * row + repeatTimes * ELEM_PER_REP_FP16],
                    static_cast<Tfm>(0.0), maskDup, 1, 1, 0);
            }
            PipeBarrier<PIPE_V>();
        }
        if ((bFormerFactor * BLOCK_NUM_PER_REP) < MAX_REP_NUM) {
            if (repeatTimes) {
                for (uint32_t i = 0; i < bFormerFactor; i++) {
                    Adds(
                        dstTensor[i * TRANSPOSE_C0_SIZE], srcTensor[i * TRANSPOSE_C0_SIZE * row], static_cast<Tfm>(0.0),
                        mask, repeatTimes,
                        {(uint16_t)bFormerFactor, 1, (uint8_t)(bFormerFactor * BLOCK_NUM_PER_REP), BLOCK_NUM_PER_REP});
                }
            }
            if (remainRepeat) {
                mask = remainRepeat * TRANSPOSE_C0_SIZE;
                for (uint32_t i = 0; i < bFormerFactor; i++) {
                    Adds(
                        dstTensor[i * TRANSPOSE_C0_SIZE + repeatTimes * ELEM_PER_REP_FP16 * bFormerFactor],
                        srcTensor[i * TRANSPOSE_C0_SIZE * row + repeatTimes * ELEM_PER_REP_FP16], static_cast<Tfm>(0.0),
                        mask, 1, {(uint16_t)bFormerFactor, 1, 0, 0});
                }
            }
        } else {
            DataCopyParams copyParams;
            copyParams.blockCount = bFormerFactor;
            copyParams.blockLen = 1;
            copyParams.srcStride = row - 1;
            copyParams.dstStride = 0;
            for (uint32_t i = 0; i < row; i++) {
                DataCopy(
                    dstTensor[i * bFormerFactor * TRANSPOSE_C0_SIZE], srcTensor[i * TRANSPOSE_C0_SIZE], copyParams);
            }
        }
    }

    __aicore__ inline void DoReduce(LocalTensor<float>& dstTensor, LocalTensor<float>& srcTensor)
    {
        /*
        srcTensor为reduce之前的Tensor: row * rowLineElements
        dstTensor为存放reduce结果的tensor: rowLineElements
        */
        uint64_t nowRows = row;
        if (nowRows == 1) {
            Adds<float>(dstTensor, srcTensor, 0, rowLineElements);
            PipeBarrier<PIPE_V>();
            return;
        }
        // row为非二次幂，先将二次幂差值行加到前面
        if (dichotomizeAddDiffSize != 0) {
            Add(srcTensor, srcTensor, srcTensor[(nowRows - dichotomizeAddDiffSize) * rowLineElements],
                dichotomizeAddDiffSize * rowLineElements);
            PipeBarrier<PIPE_V>();
            nowRows = nowRows - dichotomizeAddDiffSize;
        }
        while (nowRows > 1) {
            nowRows = nowRows / TWO_NUM;
            if (nowRows == 1) {
                Add(dstTensor, srcTensor, srcTensor[rowLineElements], rowLineElements);
            } else {
                Add(srcTensor, srcTensor, srcTensor[nowRows * rowLineElements], nowRows * rowLineElements);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void DoSub(LocalTensor<float>& dstTensor, LocalTensor<float>& src1Tensor)
    {
        /*
        dst复用src0，大小为row * rowLineElements
        src1大小为rowLineElements
        做inline broadcast的sub计算
        */
        if (((rowLineElements / ELEM_PER_REP_FP32) < row) && (rowLineElements < (MAX_REP_NUM * BLOCK_NUM_PER_REP))) {
            for (uint32_t i = 0; i < (rowLineElements / ELEM_PER_REP_FP32); i++) {
                Sub(dstTensor[i * ELEM_PER_REP_FP32], dstTensor[i * ELEM_PER_REP_FP32],
                    src1Tensor[i * ELEM_PER_REP_FP32], ELEM_PER_REP_FP32, row,
                    {1, 1, 1, (uint8_t)(rowLineElements / BLOCK_NUM_PER_REP),
                     (uint8_t)(rowLineElements / BLOCK_NUM_PER_REP), 0});
            }
            if (rowLineElements % ELEM_PER_REP_FP32 > 0) {
                Sub(dstTensor[rowLineElements / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32],
                    dstTensor[rowLineElements / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32],
                    src1Tensor[rowLineElements / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32],
                    rowLineElements % ELEM_PER_REP_FP32, row,
                    {1, 1, 1, (uint8_t)(rowLineElements / BLOCK_NUM_PER_REP),
                     (uint8_t)(rowLineElements / BLOCK_NUM_PER_REP), 0});
            }
        } else {
            for (uint64_t i = 0; i < row; i++) {
                Sub(dstTensor[i * rowLineElements], dstTensor[i * rowLineElements], src1Tensor, rowLineElements);
            }
        }
    }

    __aicore__ inline void DoDiv(
        LocalTensor<float>& dstTensor, LocalTensor<float>& src0Tensor, LocalTensor<float>& src1Tensor)
    {
        /*
        src0Tensor为置1的一个block的tensor
        src1Tensor和dstTensor大小为rowLineElements
        */
        uint64_t repeatTimes = rowLineElements / ELEM_PER_REP_FP32;
        uint64_t repeatRemain = rowLineElements % ELEM_PER_REP_FP32;
        if (repeatTimes > 0) {
            Div(dstTensor, src0Tensor, src1Tensor, ELEM_PER_REP_FP32, repeatTimes,
                {1, 0, 1, BLOCK_NUM_PER_REP, 0, BLOCK_NUM_PER_REP});
        }
        if (repeatRemain > 0) {
            Div(dstTensor[ELEM_PER_REP_FP32 * repeatTimes], src0Tensor, src1Tensor[ELEM_PER_REP_FP32 * repeatTimes],
                repeatRemain, 1, {1, 0, 1, 0, 0, 0});
        }
    }

    __aicore__ inline void DoMul(LocalTensor<float>& dstTensor, LocalTensor<float>& src1Tensor)
    {
        /*
        dst复用src0，大小为row * rowLineElements
        src1大小为rowLineElements
        做inline broadcast的mul计算
        */
        if (((rowLineElements / ELEM_PER_REP_FP32) < row) && (rowLineElements < (MAX_REP_NUM * BLOCK_NUM_PER_REP))) {
            for (uint32_t i = 0; i < (rowLineElements / ELEM_PER_REP_FP32); i++) {
                Mul(dstTensor[i * ELEM_PER_REP_FP32], dstTensor[i * ELEM_PER_REP_FP32],
                    src1Tensor[i * ELEM_PER_REP_FP32], ELEM_PER_REP_FP32, row,
                    {1, 1, 1, (uint8_t)(rowLineElements / BLOCK_NUM_PER_REP),
                     (uint8_t)(rowLineElements / BLOCK_NUM_PER_REP), 0});
            }
            if (rowLineElements % ELEM_PER_REP_FP32 > 0) {
                Mul(dstTensor[rowLineElements / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32],
                    dstTensor[rowLineElements / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32],
                    src1Tensor[rowLineElements / ELEM_PER_REP_FP32 * ELEM_PER_REP_FP32],
                    rowLineElements % ELEM_PER_REP_FP32, row,
                    {1, 1, 1, (uint8_t)(rowLineElements / BLOCK_NUM_PER_REP),
                     (uint8_t)(rowLineElements / BLOCK_NUM_PER_REP), 0});
            }
        } else {
            for (uint64_t i = 0; i < row; i++) {
                Mul(dstTensor[i * rowLineElements], dstTensor[i * rowLineElements], src1Tensor, rowLineElements);
            }
        }
    }

    __aicore__ inline void DoMulGamma(LocalTensor<float>& dstTensor)
    {
        if (nullptrGamma == 1) {
            return;
        }
        for (uint64_t i = 0; i < row; i++) {
            float gammaValue = gammaFp32.GetValue(i);
            Muls(dstTensor[i * rowLineElements], dstTensor[i * rowLineElements], gammaValue, rowLineElements);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DoAddBeta(LocalTensor<float>& dstTensor)
    {
        if (nullptrBeta == 1) {
            return;
        }
        for (uint64_t i = 0; i < row; i++) {
            float betaValue = betaFp32.GetValue(i);
            Adds(dstTensor[i * rowLineElements], dstTensor[i * rowLineElements], betaValue, rowLineElements);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DoPostReshape(LocalTensor<Tfm>& dstTensor, LocalTensor<Tfm>& srcTensor)
    {
        uint64_t mask = ELEM_PER_REP_FP16;
        uint64_t repeatTimes = row / BLOCK_NUM_PER_REP;
        uint64_t remainRows = row % BLOCK_NUM_PER_REP;
        if ((bFormerFactor * BLOCK_NUM_PER_REP) < MAX_REP_NUM) {
            if (repeatTimes) {
                for (uint32_t i = 0; i < bFormerFactor; i++) {
                    Adds(
                        dstTensor[i * TRANSPOSE_C0_SIZE * row], srcTensor[i * TRANSPOSE_C0_SIZE], static_cast<Tfm>(0.0),
                        mask, repeatTimes,
                        {1, (uint16_t)bFormerFactor, BLOCK_NUM_PER_REP, (uint8_t)(bFormerFactor * BLOCK_NUM_PER_REP)});
                }
            }
            if (remainRows) {
                mask = remainRows * TRANSPOSE_C0_SIZE;
                for (uint32_t i = 0; i < bFormerFactor; i++) {
                    Adds(
                        dstTensor[i * TRANSPOSE_C0_SIZE * row + repeatTimes * ELEM_PER_REP_FP16],
                        srcTensor[i * TRANSPOSE_C0_SIZE + repeatTimes * ELEM_PER_REP_FP16 * bFormerFactor],
                        static_cast<Tfm>(0.0), mask, 1, {1, (uint16_t)bFormerFactor, 0, 0});
                }
            }
        } else {
            DataCopyParams copyParams;
            copyParams.blockCount = bFormerFactor;
            copyParams.blockLen = 1;
            copyParams.srcStride = 0;
            copyParams.dstStride = row - 1;
            for (uint64_t i = 0; i < row; i++) {
                DataCopy(
                    dstTensor[i * TRANSPOSE_C0_SIZE], srcTensor[i * bFormerFactor * TRANSPOSE_C0_SIZE], copyParams);
            }
        }
    }

    __aicore__ inline void DoPostTranspose(LocalTensor<Tfm>& dstTensor, LocalTensor<Tfm>& srcTensor)
    {
        /*
        tiling限制repeat不大于255
        暂只支持bloat16和float16
        */
        __ubuf__ Tfm* srcAddr = (__ubuf__ Tfm*)srcTensor.GetPhyAddr();
        __ubuf__ Tfm* dstAddr = (__ubuf__ Tfm*)dstTensor.GetPhyAddr();
        __ubuf__ Tfm* srcLocalList[TRANSPOSE_C0_SIZE];
        __ubuf__ Tfm* dstLocalList[TRANSPOSE_C0_SIZE];
        for (uint32_t i = 0; i < TRANSPOSE_C0_SIZE; i++) {
            srcLocalList[i] = srcAddr + TRANSPOSE_C0_SIZE * i;
            dstLocalList[i] = dstAddr + rFormerAxisAlign * i;
        }
        struct TransDataTo5HDParams transDataParams;
        transDataParams.repeatTimes = rFormerAxisAlign / TRANSPOSE_C0_SIZE;
        transDataParams.srcRepStride = TRANSPOSE_C0_SIZE;
        transDataParams.dstRepStride = 1;
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        }
        TransDataTo5HDImpl(dstLocalList, srcLocalList, transDataParams);
    }

    __aicore__ inline void DoMeanOrRstdTranspose(LocalTensor<float>& dstTensor, LocalTensor<float>& srcTensor)
    {
        /*
        tiling限制repeat不大于255
        只支持fp32
        */
        __ubuf__ float* srcAddr = (__ubuf__ float*)srcTensor.GetPhyAddr();
        __ubuf__ float* dstAddr = (__ubuf__ float*)dstTensor.GetPhyAddr();
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
        for (uint32_t i = 0; i < (FLOAT_SIZE / HALF_SIZE); i++) {
            for (uint32_t j = 0; j < TRANSPOSE_C0_SIZE; j++) {
                srcLocalList[j] = srcAddr + FP32_TRANSPOSE_DST_SIZE * i + TRANSPOSE_C0_SIZE * j;
            }
            for (uint32_t k = 0; k < FP32_TRANSPOSE_DST_SIZE; k++) {
                dstLocalList[(FLOAT_SIZE / HALF_SIZE) * k] =
                    dstAddr + transDataParams.repeatTimes * TRANSPOSE_C0_SIZE * k +
                    transDataParams.repeatTimes * TRANSPOSE_C0_SIZE * FP32_TRANSPOSE_DST_SIZE * i;
                dstLocalList[(FLOAT_SIZE / HALF_SIZE) * k + 1] =
                    dstAddr + transDataParams.repeatTimes * TRANSPOSE_C0_SIZE * k + FP32_TRANSPOSE_DST_SIZE +
                    transDataParams.repeatTimes * TRANSPOSE_C0_SIZE * FP32_TRANSPOSE_DST_SIZE * i;
            }
            TransDataTo5HDImpl(dstLocalList, srcLocalList, transDataParams);
        }
    }

    template <typename T>
    __aicore__ inline void GetLastBlockTensor(
        LocalTensor<T>& dstTensor, LocalTensor<T>& srcTensor, uint32_t rowNum, uint32_t lineLength)
    {
        /*
        rowNum 表示一行中一个有效元素组的元素个数
        lineLength表示ub中，一行的长度
        */
        // pipe_s等pipe_v
        PipeBarrier<PIPE_ALL>();
        uint32_t blockAlignNum = BLOCK / sizeof(T);
        uint32_t setValueNum = 0;
        uint32_t curNum = 0;
        for (int32_t i = TRANSPOSE_C0_SIZE - 1; i >= 0; i--) {
            if (i >= formerLoops) {
                curNum = bTailFactor * rowNum;
            } else {
                curNum = bFormerFactor * rowNum;
            }
            if (curNum == 0) {
                continue;
            }
            if (setValueNum >= blockAlignNum) {
                break;
            }
            for (uint32_t j = 0; j < curNum; j++) {
                float tmpValue = srcTensor.GetValue(lineLength * i + curNum - j - 1);
                dstTensor.SetValue(blockAlignNum - setValueNum - 1, tmpValue);
                setValueNum += 1;
                if (setValueNum >= blockAlignNum) {
                    break;
                }
            }
        }
        // pipe_mte3等pipe_s
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyOutMeanOrRstd(GlobalTensor<float> outGm, LocalTensor<float>& srcTensor)
    {
        /*
        16次重复按block搬运，gm按有效元素偏移，并校验是否踩多核，如果踩多核，提前退出，使用scalar拼接最后一个block数据
        */
        uint32_t curNum = bFormerFactor;
        uint32_t loopNums = TRANSPOSE_C0_SIZE;
        uint64_t ubOffset = 0;
        uint64_t gmOffset = 0;
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.blockLen = (bFormerFactor + B32_BLOCK_ALIGN_NUM - 1) / B32_BLOCK_ALIGN_NUM;
        for (uint32_t i = 0; i < loopNums; i++) {
            if (i >= formerLoops) {
                copyParams.blockLen = (bTailFactor + B32_BLOCK_ALIGN_NUM - 1) / B32_BLOCK_ALIGN_NUM;
                curNum = bTailFactor;
            }
            if (curNum == 0) {
                break;
            }
            if ((needOverLap) && ((gmOffset + copyParams.blockLen * B32_BLOCK_ALIGN_NUM) > colLength)) {
                if ((((gmOffset + copyParams.blockLen * B32_BLOCK_ALIGN_NUM - B32_BLOCK_ALIGN_NUM) <= colLength)) &&
                    (copyParams.blockLen > 1)) {
                    copyParams.blockLen = copyParams.blockLen - 1;
                    DataCopy(outGm[meanGmOffset + gmOffset], srcTensor[ubOffset], copyParams);
                    PipeBarrier<PIPE_MTE3>();
                    ;
                    ubOffset += copyParams.blockLen * B32_BLOCK_ALIGN_NUM;
                    gmOffset += copyParams.blockLen * B32_BLOCK_ALIGN_NUM;
                }
                break;
            }
            DataCopy(outGm[meanGmOffset + gmOffset], srcTensor[ubOffset], copyParams);
            PipeBarrier<PIPE_MTE3>();
            ;
            ubOffset += (bFormerFactor + TRANSPOSE_C0_SIZE - 1) / TRANSPOSE_C0_SIZE * TRANSPOSE_C0_SIZE;
            gmOffset += curNum;
        }

        if (gmOffset < colLength) {
            LocalTensor<float> lastBlockTensor = overLapBuf.Get<float>();
            GetLastBlockTensor<float>(
                lastBlockTensor, srcTensor, 1,
                (bFormerFactor + TRANSPOSE_C0_SIZE - 1) / TRANSPOSE_C0_SIZE * TRANSPOSE_C0_SIZE);
            DataCopy(outGm[meanGmOffset + colLength - B32_BLOCK_ALIGN_NUM], lastBlockTensor, B32_BLOCK_ALIGN_NUM);
        }
    }

    __aicore__ inline void CopyOutY(LocalTensor<Tfm>& srcTensor)
    {
        uint32_t curNum = bFormerFactor * row;
        uint32_t loopNums = TRANSPOSE_C0_SIZE;
        uint64_t ubOffset = 0;
        uint64_t gmOffset = 0;
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.blockLen = rFormerAxisAlign / X_NUM_PER_BLOCK;
        for (uint32_t i = 0; i < loopNums; i++) {
            if (i >= formerLoops) {
                copyParams.blockLen = rTailAxisAlign / X_NUM_PER_BLOCK;
                curNum = bTailFactor * row;
            }
            if (curNum == 0) {
                break;
            }
            if ((needOverLap) && ((gmOffset + copyParams.blockLen * X_NUM_PER_BLOCK) > colLength * row)) {
                if ((((gmOffset + copyParams.blockLen * X_NUM_PER_BLOCK - X_NUM_PER_BLOCK) <= colLength * row)) &&
                    (copyParams.blockLen > 1)) {
                    copyParams.blockLen = copyParams.blockLen - 1;
                    DataCopy(yGm[xGmOffset + gmOffset], srcTensor[ubOffset], copyParams);
                    PipeBarrier<PIPE_MTE3>();
                    ;
                    ubOffset += copyParams.blockLen * X_NUM_PER_BLOCK;
                    gmOffset += copyParams.blockLen * X_NUM_PER_BLOCK;
                }
                break;
            }
            DataCopy(yGm[xGmOffset + gmOffset], srcTensor[ubOffset], copyParams);
            PipeBarrier<PIPE_MTE3>();
            ;
            ubOffset += rFormerAxisAlign;
            gmOffset += curNum;
        }
        if (gmOffset < colLength * row) {
            LocalTensor<Tfm> lastBlockTensor = overLapBuf.Get<Tfm>();
            GetLastBlockTensor<Tfm>(lastBlockTensor, srcTensor, row, rFormerAxisAlign);
            DataCopy(yGm[xGmOffset + row * colLength - X_NUM_PER_BLOCK], lastBlockTensor, X_NUM_PER_BLOCK);
        }
    }

    __aicore__ inline void ProcessBasicBlock()
    {
        LocalTensor<Tfm> xLocal = inQueueX.AllocTensor<Tfm>();
        CopyInPad(xLocal);
        inQueueX.EnQue(xLocal);
        inQueueX.DeQue<Tfm>();

        LocalTensor<Tfm> transposeXLocal = tmpBuf.Get<Tfm>();
        DoTranspose(transposeXLocal, xLocal);
        PipeBarrier<PIPE_V>();

        LocalTensor<Tfm> xLocalSecond = xLocal[TRANSPOSE_C0_SIZE * rFormerAxisAlign];
        DoReshape(xLocalSecond, transposeXLocal);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> xLocalFp32 = xLocal.template ReinterpretCast<float>();
        Cast(xLocalFp32, xLocalSecond, RoundMode::CAST_NONE, calcXElements);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> mulTempTensor = tmpBuf.Get<float>();
        // xLocalFp32需要驻留
        Muls(mulTempTensor, xLocalFp32, coefficient, calcXElements);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> outTempTensor = outQueueRstd.AllocTensor<float>();
        DoReduce(outTempTensor, mulTempTensor);
        PipeBarrier<PIPE_V>();

        DoSub(xLocalFp32, outTempTensor);
        LocalTensor<float> outMeanTensor = outQueueMean.AllocTensor<float>();
        DoMeanOrRstdTranspose(outMeanTensor, outTempTensor);
        outQueueRstd.FreeTensor(outTempTensor);
        outQueueMean.EnQue(outMeanTensor);
        outQueueMean.DeQue<float>();
        CopyOutMeanOrRstd(meanGm, outMeanTensor);
        outQueueMean.FreeTensor(outMeanTensor);

        // do mul2, xLocalFp32需要驻留
        LocalTensor<float> mul2TempTensor = tmpBuf.Get<float>();
        Mul(mul2TempTensor, xLocalFp32, xLocalFp32, calcXElements);
        PipeBarrier<PIPE_V>();
        Muls(mul2TempTensor, mul2TempTensor, coefficient, calcXElements);
        PipeBarrier<PIPE_V>();

        // do reduce1
        LocalTensor<float> outMTensor = outQueueMean.AllocTensor<float>();
        DoReduce(outMTensor, mul2TempTensor);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> tempTensor = tmpBuf.Get<float>();
        Adds(tempTensor, outMTensor, eps, rowLineElements);
        PipeBarrier<PIPE_V>();

        Sqrt(tempTensor, tempTensor, rowLineElements);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> oneTensor = outQueueY.AllocTensor<float>();
        Duplicate<float>(oneTensor, 1, B32_BLOCK_ALIGN_NUM);
        PipeBarrier<PIPE_V>();

        DoDiv(outMTensor, oneTensor, tempTensor);
        PipeBarrier<PIPE_V>();
        outQueueY.FreeTensor(oneTensor);

        DoMul(xLocalFp32, outMTensor);
        PipeBarrier<PIPE_V>();
        // output rstd
        LocalTensor<float> outRstdTensor = outQueueRstd.AllocTensor<float>();
        DoMeanOrRstdTranspose(outRstdTensor, outMTensor);
        outQueueMean.FreeTensor(outMTensor);
        outQueueRstd.EnQue(outRstdTensor);
        outQueueRstd.DeQue<float>();
        CopyOutMeanOrRstd(rstdGm, outRstdTensor);
        outQueueRstd.FreeTensor(outRstdTensor);

        DoMulGamma(xLocalFp32);

        DoAddBeta(xLocalFp32);

        LocalTensor<Tfm> outputYTensor = outQueueY.AllocTensor<Tfm>();
        // 310P not support CAST_ROUND
        Cast(outputYTensor, xLocalFp32, RoundMode::CAST_NONE, calcXElements);
        PipeBarrier<PIPE_V>();
        inQueueX.FreeTensor(xLocalFp32);

        LocalTensor<Tfm> postReshapeTensor = tmpBuf.Get<Tfm>();
        DoPostReshape(postReshapeTensor, outputYTensor);
        PipeBarrier<PIPE_V>();

        DoPostTranspose(outputYTensor, postReshapeTensor);

        outQueueY.EnQue(outputYTensor);
        outQueueY.DeQue<Tfm>();
        CopyOutY(outputYTensor);
        outQueueY.FreeTensor(outputYTensor);
    }

private:
    constexpr static uint32_t BLOCK = 32;
    constexpr static uint32_t X_NUM_PER_BLOCK = BLOCK / sizeof(Tfm);
    constexpr static uint32_t GAMMA_NUM_PER_BLOCK = BLOCK / sizeof(Tweight);
    constexpr static uint32_t QUEUE_DEPTH = 2;
    constexpr static uint32_t FLOAT_SIZE = 4;
    constexpr static uint32_t HALF_SIZE = 2;
    constexpr static uint32_t TRANSPOSE_C0_SIZE = 16;
    constexpr static uint32_t MAX_REP_NUM = 255;
    constexpr static uint32_t ELEM_PER_REP_FP32 = 64;
    constexpr static uint32_t ELEM_PER_REP_FP16 = 128;
    constexpr static uint32_t BLOCK_NUM_PER_REP = 8;
    constexpr static uint32_t FP32_TRANSPOSE_DST_SIZE = 8;
    constexpr static uint32_t UINT16_BIT_SIZE = 16;
    constexpr static uint32_t UINT64_BIT_SIZE = 64;
    constexpr static uint32_t B32_BLOCK_ALIGN_NUM = 8;
    constexpr static uint32_t TWO_NUM = 2;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueGamma, inQueueBeta;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY, outQueueMean, outQueueRstd;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> overLapBuf;

    GlobalTensor<Tfm> xGm;
    GlobalTensor<Tfm> yGm;
    GlobalTensor<Tweight> gammaGm;
    GlobalTensor<Tweight> betaGm;
    GlobalTensor<float> meanGm;
    GlobalTensor<float> rstdGm;

    uint32_t blockIdx = GetBlockIdx();

    // calculate xGm and meanGm offset for ub loop
    uint64_t xGmOffset = 0;
    uint64_t meanGmOffset = 0;
    // in ub loop col size
    uint64_t colLength = 0;

    // x搬入时整块的借轴因子
    uint32_t bFormerFactor = 0;
    // x搬入时一行整块的对齐长度
    uint32_t rFormerAxisAlign = 0;
    // x搬入时尾块的借轴因子
    uint32_t bTailFactor = 0;
    // x搬入时一行尾块的对齐长度
    uint32_t rTailAxisAlign = 0;
    // 整块搬入的循环次数
    uint32_t formerLoops = 0;
    // 重排后x基本块的元素个数
    uint64_t calcXElements;
    // 重排后x做row Reduce后的元素个数
    uint64_t rowLineElements;
    // Duplicate清零脏数据的mask
    uint64_t formerMask = 0;
    uint64_t remainMaskLow = 0;
    uint64_t remainMaskHigh = 0;

    bool needOverLap;
    bool isLastCore;
    bool isLastUbLoop;

    LocalTensor<float> gammaFp32;
    LocalTensor<float> betaFp32;
    // tilingData
    uint64_t col;
    uint64_t row;
    uint64_t numBlocks;
    uint64_t blockFormer;
    uint64_t blockTail;
    uint64_t ubFormer;
    uint64_t bFormer;
    uint64_t dichotomizeAddDiffSize;
    uint64_t ubLoopOfFormerBlock;
    uint64_t ubLoopOfTailBlock;
    uint64_t ubTailOfFormerBlock;
    uint64_t ubTailOfTailBlock;
    float eps = 0.0;
    float coefficient = 0.0;
    uint32_t nullptrGamma;
    uint32_t nullptrBeta;
};

} // namespace LayerNormV4
#endif // LAYER_NORM_V4_TRANSPOSE_310P_H
