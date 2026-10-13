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
 * \file rms_norm_merge_n.h
 * \brief rms norm merge n file
 */
#ifndef RMS_NORM_MERGE_N_H_
#define RMS_NORM_MERGE_N_H_
#include "rms_norm_base.h"

namespace RmsNorm {
using namespace AscendC;

template <typename T, typename T_GAMMA>
class KernelRmsNormMergeN {
#define IS_X_FP32 (is_same<T, float>::value)
#define IS_GAMMA_FP32 (is_same<T_GAMMA, float>::value)
#define IS_MIX_DTYPE ((!IS_X_FP32) && IS_GAMMA_FP32)
public:
    __aicore__ inline KernelRmsNormMergeN()
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, const RMSNormTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        blockIdx_ = GetBlockIdx();
        InitVar(tiling, blockIdx_);
        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ T_GAMMA*)gamma, numCol);
        yGm.SetGlobalBuffer((__gm__ T*)y + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockIdx_ * blockFactor, blockFactor);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, ubFactor * sizeof(T));
        pipe.InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, ubFactor * sizeof(T));
        pipe.InitBuffer(outQueueRstd, BUFFER_NUM, rowFactor * sizeof(float));

        if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
            pipe.InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        }
        pipe.InitBuffer(sqxBuf, ubFactor * sizeof(float));
        pipe.InitBuffer(tmpBuf, rowFactor * NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void InitVar(const RMSNormTilingData* tiling, int32_t blockIdx_)
    {
        numRow = tiling->num_row;
        numCol = tiling->num_col;
        numColAlign = tiling->num_col_align;
        blockFactor = tiling->block_factor;
        rowFactor = tiling->row_factor;
        ubFactor = tiling->ub_factor;
        epsilon = tiling->epsilon;
        if (blockIdx_ < GetBlockNum() - 1) {
            this->rowWork = blockFactor;
            this->rowLoop = tiling->row_loop;
            this->rowTail = tiling->row_tail;
        } else if (blockIdx_ == GetBlockNum() - 1) {
            this->rowWork = tiling->last_block_factor;
            this->rowLoop = tiling->last_block_row_loop;
            this->rowTail = tiling->last_block_row_tail;
        }
        avgFactor = tiling->avg_factor;

        mulLoop = tiling->mul_loop;
        // 11 * 2 + 3 = 25
        mulTail = tiling->mul_tail;
        //按uint8解析，uint64的数据占8位，float和uint32的数据占4位，一共有11个uint64(88)，2个uint32(8)，2个float(8)，一共96位
        dstRepStride = tiling->dst_rep_stride;
        isPerformance = tiling->is_performance;
        isNorm = tiling->normal_flag;
        isNumColAlign = (numCol == numColAlign);
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<float> gammaLocal = inQueueGamma.DeQue<float>();
        if(isPerformance != 1) {
          BroadCastGamma(gammaLocal);
        }
        for (uint32_t i_o = 0; i_o < rowLoop - 1; i_o++) {
            DoMainCompute(i_o, rowFactor, gammaLocal);
        }
        DoMainCompute(rowLoop - 1, rowTail, gammaLocal);
        inQueueGamma.FreeTensor(gammaLocal);
    }

    __aicore__ inline void DoMainCompute(uint32_t i_o, uint32_t calcRowNum, LocalTensor<float>& gammaLocal)
    {
        uint32_t gmBias = i_o * rowFactor * numCol;
        uint32_t elementNum = calcRowNum * numColAlign;
        CopyIn(gmBias, calcRowNum);
        if (isNorm == 1) {
          ComputeSingleRow(i_o, calcRowNum, gammaLocal, gmBias);
        } else {
          Compute(i_o, gammaLocal, calcRowNum, elementNum);
        }
        CopyOutY(gmBias, calcRowNum);
    }

private:
    __aicore__ inline void CopyIn(uint32_t gm_bias, uint32_t calc_row_num)
    {    
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        if (isNumColAlign) {
            DataCopyCustom<T>(xLocal, xGm[gm_bias], calc_row_num * numCol);
        } else {
#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
            // only support v220
            DataCopyParams copyParams;
            copyParams.blockLen = numCol * sizeof(T);
            copyParams.blockCount = calc_row_num;
            DataCopyPadParams padParams;
            DataCopyPad(xLocal, xGm[gm_bias], copyParams, padParams);
#endif
        }

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInGamma()
    {
        event_t eventMTE2V_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        // 非混合精度，非fp32场景下， gamma转换成fp32数据类型
        if constexpr (std::is_same<T_GAMMA, half>::value || std::is_same<T_GAMMA, bfloat16_t>::value) {
            LocalTensor<float> gammaLocal = inQueueGamma.AllocTensor<float>();
            LocalTensor<T_GAMMA> gammaIn = sqxBuf.Get<T_GAMMA>();
            DataCopyCustom<T_GAMMA>(gammaIn, gammaGm, numCol);
            SetFlag<HardEvent::MTE2_S>(eventMTE2V_1);
            WaitFlag<HardEvent::MTE2_S>(eventMTE2V_1);
            Cast(gammaLocal, gammaIn, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            inQueueGamma.EnQue(gammaLocal);
        } else {
            LocalTensor<float> gammaIn = inQueueGamma.AllocTensor<float>();
            DataCopyCustom<float>(gammaIn, gammaGm, numCol);
            inQueueGamma.EnQue(gammaIn);
        }
    }

    __aicore__ inline void BroadCastGamma(LocalTensor<float>& gammaLocal)
    {
        const uint32_t srcShape[2] = {1, numColAlign};
        const uint32_t dstShape[2] = {rowFactor, numColAlign};
        LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();
        BroadCast<float, DIM_NUM, 0>(gammaLocal, gammaLocal, dstShape, srcShape, tmpLocal);
    }

    __aicore__ inline void ComputeMulGammaCast(LocalTensor<float> gammaLocal, uint32_t elementNum, uint32_t calcRowNum)
    {
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        if(isPerformance == 1) {
          LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
          repeatByRow(xFp32, xFp32, gammaLocal, calcRowNum, 0);
          if constexpr (is_same<T, half>::value) {
              Cast(yLocal, xFp32, RoundMode::CAST_NONE, elementNum);
          } else {
              Cast(yLocal, xFp32, RoundMode::CAST_RINT, elementNum);
          }
        } else {
          if constexpr (is_same<T, float>::value) {
              Mul(yLocal, sqx, gammaLocal, elementNum);
          } else {
              Mul(sqx, sqx, gammaLocal, elementNum);
              PipeBarrier<PIPE_V>();
              if constexpr (is_same<T, half>::value) {
                  Cast(yLocal, sqx, RoundMode::CAST_NONE, elementNum);
              } else {
                  Cast(yLocal, sqx, RoundMode::CAST_RINT, elementNum);
              }
          }
        }
        PipeBarrier<PIPE_V>();
        outQueueY.EnQue<T>(yLocal);
    }

    __aicore__ inline void ComputeSingleRow(uint32_t i_o, uint32_t calc_row_num, LocalTensor<float> gammaLocal, uint32_t gm_bias)
    {
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<float> xFp32;
        if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
            xFp32 = xFp32Buf.Get<float>();
            Cast(xFp32, xLocal, RoundMode::CAST_NONE, calc_row_num * numColAlign);
            inQueueX.FreeTensor(xLocal);
            PipeBarrier<PIPE_V>();
        }
        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            LocalTensor<float> sqx = sqxBuf.Get<float>();
            uint32_t rowOffset = i_i * numColAlign;
            if constexpr (is_same<T, float>::value) {
                Mul(sqx, xLocal[rowOffset], xLocal[rowOffset], numColAlign);
            } else {
                Mul(sqx, xFp32[rowOffset], xFp32[rowOffset], numColAlign);
            }
            PipeBarrier<PIPE_V>();
            float reduceOut = ReduceSumHalfInterval(sqx, numCol);
            float rstdValue = 1 / sqrt(reduceOut * avgFactor + epsilon);
            rstdLocal.SetValue(i_i, rstdValue);
            if constexpr (is_same<T, float>::value) {
                Muls(sqx, xLocal[rowOffset], rstdValue, numColAlign);
            } else {
                Muls(sqx, xFp32[rowOffset], rstdValue, numColAlign);
            }
            PipeBarrier<PIPE_V>(); 
            Mul(sqx, sqx, gammaLocal, numCol);
            if constexpr (is_same<T, half>::value) {
                Cast(yLocal[rowOffset], sqx, RoundMode::CAST_NONE, numCol);
            } else {
                Cast(yLocal[rowOffset], sqx, RoundMode::CAST_RINT, numCol);
            }
            PipeBarrier<PIPE_V>();
        }
        outQueueRstd.EnQue<float>(rstdLocal);
        CopyOutRstd(i_o, calc_row_num);
        outQueueY.EnQue<T>(yLocal);
    }

    __aicore__ inline void Compute(uint32_t i_o, LocalTensor<float> gammaLocal, uint32_t calcRowNum, uint32_t elementNum)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        LocalTensor<float> xFp32;
        
        // 1. Cast x and Cal x^2
        if constexpr (is_same<T, float>::value) {
            Mul(sqx, xLocal, xLocal, elementNum);
        } else {
            xFp32 = xFp32Buf.Get<float>();
            Cast(xFp32, xLocal, RoundMode::CAST_NONE, elementNum);
            inQueueX.FreeTensor(xLocal);
            PipeBarrier<PIPE_V>();
            Mul(sqx, xFp32, xFp32, elementNum);
        }
        PipeBarrier<PIPE_V>();

        // 2. Rstd = 1 / sqrt(1 / reduceDim * reducesum(x^2) + eps)
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        ReduceSumMultiN(rstdLocal, sqx, tmpLocal, calcRowNum, numCol, numColAlign);
        PipeBarrier<PIPE_V>();
        Muls(rstdLocal, rstdLocal, avgFactor, calcRowNum);
        PipeBarrier<PIPE_V>();
        Adds(rstdLocal, rstdLocal, epsilon, calcRowNum);
        PipeBarrier<PIPE_V>();
        Sqrt(rstdLocal, rstdLocal, calcRowNum);
        Duplicate(tmpLocal, ONE, calcRowNum);
        PipeBarrier<PIPE_V>();
        Div(rstdLocal, tmpLocal, rstdLocal, calcRowNum);
        PipeBarrier<PIPE_V>();
        outQueueRstd.EnQue<float>(rstdLocal);
        CopyOutRstd(i_o, calcRowNum);
        
        // 4. Y = x * rstd * gamma
        if (isPerformance == 1) {
            doPerformanceRstd(sqx, rstdLocal, calcRowNum);
            repeatByRow(xFp32, xFp32, sqx, calcRowNum, 1);
        } else {
            const uint32_t srcShape[2] = {calcRowNum, 1};
            const uint32_t dstShape[2] = {calcRowNum, numColAlign};
            auto sharedTmpLocal = tmpLocal.ReinterpretCast<uint8_t>();
            BroadCast<float, DIM_NUM, 1>(sqx, rstdLocal, dstShape, srcShape, sharedTmpLocal);
            PipeBarrier<PIPE_V>();

            // 4. Y = x * rstd * gamma
            if constexpr (is_same<T, float>::value) { // fp32 use inQueueX store x
                Mul(sqx, xLocal, sqx, elementNum);
                inQueueX.FreeTensor(xLocal);
            } else { // fp16/bf16 use xFp32Buf store x
                Mul(sqx, xFp32, sqx, elementNum);
            }
        }
        PipeBarrier<PIPE_V>();
        ComputeMulGammaCast(gammaLocal, elementNum, calcRowNum);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOutY(uint32_t progress, uint32_t calc_row_num)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        if (isNumColAlign) {
            DataCopyCustom<T>(yGm[progress], yLocal, calc_row_num * numCol);
        } else {
#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
            // only support v220
            DataCopyParams copyParams;
            copyParams.blockLen = numCol * sizeof(T);
            copyParams.blockCount = calc_row_num;
            DataCopyPad(yGm[progress], yLocal, copyParams);
#endif
        }
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(uint32_t outer_progress, uint32_t num)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)) || (defined(__CCE_AICORE__) && __CCE_AICORE__ == 200)
        DataCopyCustom<float>(rstdGm[outer_progress * rowFactor], rstdLocal, num);
#endif
        outQueueRstd.FreeTensor(rstdLocal);
    }

    __aicore__ inline void doPerformanceRstd(const LocalTensor<float>& dstLocal, const LocalTensor<float>& src1Local, uint32_t calcRowNum)
    {
        uint32_t splidRow = 240;
        uint32_t rowRepeatLoop = calcRowNum / splidRow;
        uint32_t rowRepeatTail = calcRowNum - rowRepeatLoop * splidRow;
        for(uint32_t r_i = 0; r_i < rowRepeatLoop; r_i ++) {
            Brcb(dstLocal[r_i * splidRow * MOV_8], src1Local[r_i * splidRow], splidRow, {1, 8});
            PipeBarrier<PIPE_V>();
        }
        
        if(rowRepeatTail > 0) {
            Brcb(dstLocal[rowRepeatLoop * splidRow * MOV_8], src1Local[rowRepeatLoop * splidRow], rowRepeatTail, {1, 8});
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void repeatByRow(const LocalTensor<float>& dstLocal, const LocalTensor<float>& src1Local, const LocalTensor<float>& src2Local, uint32_t calc_row_num, uint32_t type)
    {
        uint32_t singlT = 255;
        uint32_t rowRepeatLoop = calc_row_num / singlT;
        uint32_t rowRepeatTail = calc_row_num - rowRepeatLoop * singlT;
        for(uint32_t r_i = 0; r_i < rowRepeatLoop; r_i ++) {
            uint32_t offset2 = 0;
            if (type == 1) {
              offset2 = r_i * singlT * MOV_8;
            }
            uint32_t offset1 = r_i * singlT * numColAlign;
            mulRepeat(dstLocal[offset1], src1Local[offset1], src2Local[offset2], singlT, type);
        }
        if(rowRepeatTail > 0) {
            uint32_t offset2 = 0;
            if (type == 1) {
              offset2 = rowRepeatLoop * singlT * MOV_8;
            }
            uint32_t offset1 = rowRepeatLoop * singlT * numColAlign;
            mulRepeat(dstLocal[offset1], src1Local[offset1], src2Local[offset2], rowRepeatTail, type);
        }
    }

    __aicore__ inline void mulRepeat(LocalTensor<float> dstLocal, LocalTensor<float> src1Local, LocalTensor<float> src2Local, uint32_t calcRowNum, uint8_t isRstd)
    {
        if (isRstd == 1) {
          for (uint32_t m_i = 0; m_i < mulLoop; m_i++) {
              Mul(dstLocal[m_i * NUM_PER_REP_FP32], src1Local[m_i * NUM_PER_REP_FP32], src2Local, NUM_PER_REP_FP32, calcRowNum, {1, 1, 0, dstRepStride, dstRepStride, 1});
              PipeBarrier<PIPE_V>();
          }
          if(mulTail > 0) {
              Mul(dstLocal[mulLoop * NUM_PER_REP_FP32], src1Local[mulLoop * NUM_PER_REP_FP32], src2Local, mulTail, calcRowNum, {1, 1, 0, dstRepStride, dstRepStride, 1});
              PipeBarrier<PIPE_V>();
          }
        } else {
          for (uint32_t m_i = 0; m_i < mulLoop; m_i++) {
              Mul(dstLocal[m_i * NUM_PER_REP_FP32], src1Local[m_i * NUM_PER_REP_FP32], src2Local[m_i * NUM_PER_REP_FP32], NUM_PER_REP_FP32, calcRowNum, {1, 1, 1, dstRepStride, dstRepStride, 0});
              PipeBarrier<PIPE_V>();
          }
          if(mulTail > 0) {
              Mul(dstLocal[mulLoop * NUM_PER_REP_FP32], src1Local[mulLoop * NUM_PER_REP_FP32], src2Local[mulLoop * NUM_PER_REP_FP32], mulTail, calcRowNum, {1, 1, 1, dstRepStride, dstRepStride, 0});
              PipeBarrier<PIPE_V>();
          }
        }
    }

private:
    TPipe pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGamma;
    // create queues for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueRstd;

    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> sqxBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    GlobalTensor<T> xGm;
    GlobalTensor<T_GAMMA> gammaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> rstdGm;

    uint32_t numRow;
    uint32_t numCol;
    uint32_t numColAlign;
    uint32_t blockFactor; // number of calculations rows on each core
    uint32_t rowFactor;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    int32_t blockIdx_;
    uint32_t rowWork = 1;
    bool isNumColAlign;
    uint32_t rowLoop = 1;
    uint32_t rowTail = 0;
    uint32_t mulLoop;
    uint32_t mulTail;
    uint8_t dstRepStride;
    uint8_t isPerformance = 0;
    uint8_t isNorm = 0;
};
} // namespace RmsNorm
#endif // RMS_NORM_MERGE_N_H_
