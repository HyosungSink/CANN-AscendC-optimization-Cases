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
 * \file dynamic_quant_base.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_BASE_H
#define DYNAMIC_QUANT_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace DynamicQuantNDOpt {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t DOUBLE = 2;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t FIFTEEN = 15;
constexpr uint32_t SIXTEEN = 16;
constexpr uint32_t SEVEN = 7;
constexpr uint32_t EIGHT = 8;
constexpr uint32_t THIRTY_ONE = 31;
constexpr uint32_t THIRTY_TWO = 32;
constexpr uint32_t SIXTY_THREE = 63;
constexpr uint32_t SIXTY_FOUR = 64;
constexpr uint32_t QUANT_TEMP_LEN = 128;
constexpr uint32_t MAX_VALUE_NUM = 8;
constexpr uint32_t MAX_REPEAT = 255;
constexpr uint32_t UB_RESERVED_LENGTH = 1024;
constexpr int32_t ELEM_PER_REP_FP32 = 64;  // ONE_REPEAT_BYTE_SIZE / sizeof(float)
constexpr int32_t ELEM_PER_REP_HALF = 128; // ONE_REPEAT_BYTE_SIZE / sizeof(half)
constexpr float DYNAMIC_QUANT_INT8_SCALE = 255.0;
constexpr float DYNAMIC_QUANT_INT8_SYM_SCALE = 127.0;
constexpr float DYNAMIC_QUANT_INT8_OFFSET = 127.0;
constexpr float DYNAMIC_QUANT_INT4_SCALE = 15.0;
constexpr float DYNAMIC_QUANT_INT4_SYM_SCALE = 7.0;
constexpr float DYNAMIC_QUANT_INT4_OFFSET = 7.0;
constexpr float DYNAMIC_QUANT_EPSINON = 1e-12;
constexpr float MIN_FLOAT_VALUE = -3.4028235e+38f;
constexpr float MAX_FLOAT_VALUE = 3.402823466e+38f;

class DynamicQuantBase
{
public:
    __aicore__ inline DynamicQuantBase()
    {}

    __aicore__ inline void ParseTilingData(const DynamicQuantTilingData* tilingData)
    {
        tilingData_.rowLen = tilingData->rowLen;
        tilingData_.coreNum = tilingData->coreNum;
        tilingData_.headCoreNum = tilingData->headCoreNum;
        tilingData_.rowPerHeadCore = tilingData->rowPerHeadCore;
        tilingData_.rowPerTailCore = tilingData->rowPerTailCore;
        tilingData_.multiRowNumTailCore = tilingData->multiRowNumTailCore;
        tilingData_.multiRowNumHeadCore = tilingData->multiRowNumHeadCore;
        tilingData_.innerLoopEle = tilingData->innerLoopEle;
        tilingData_.innerLoopTail = tilingData->innerLoopTail;
        tilingData_.innerLoopTimes = tilingData->innerLoopTimes;
        tilingData_.alignGroupNum = tilingData->alignGroupNum;
        tilingData_.groupNum = tilingData->groupNum;
        tilingData_.hasSmooth = tilingData->hasSmooth;
        tilingData_.ubSize = tilingData->ubSize;
    }

    __aicore__ inline void InitBaseBuffer()
    {
        pPipe->InitBuffer(quantTemp, QUANT_TEMP_LEN * sizeof(float));
        pPipe->InitBuffer(tempCastUb, (sizeHalfLen + SIXTY_THREE) / SIXTY_FOUR * SIXTY_FOUR  * sizeof(float));
        pPipe->InitBuffer(fp32_buf_, sizeHalfLen * sizeof(float));
    }

    __aicore__ inline void InitLargeShapeBaseBuffer()
    {
        pPipe->InitBuffer(quantTemp, THIRTY_TWO);
        pPipe->InitBuffer(tempCastUb, tilingData_.innerLoopEle * sizeof(float));
        pPipe->InitBuffer(fp32_buf_, tilingData_.innerLoopEle * sizeof(float));
    }

    __aicore__ inline void DuplicateConst()
    {
        if (!isAsymmetrical) {
            LocalTensor<uint8_t> tmp = quantTemp.Get<uint8_t>();
            constScale = tmp.ReinterpretCast<float>();
            constInvScale = tmp[MAX_VALUE_NUM * sizeof(float)].ReinterpretCast<float>();
            scaleTmp = tmp[MAX_VALUE_NUM * sizeof(float) * 2].ReinterpretCast<float>();
            quantScaleTmp = tmp[MAX_VALUE_NUM * sizeof(float) * 3].ReinterpretCast<float>();
            brcbTmp = tmp[MAX_VALUE_NUM * sizeof(float) * 4].ReinterpretCast<float>();
            if constexpr (ORIG_DTYPE_Y == DT_INT4) {
                Duplicate<float>(constScale, DYNAMIC_QUANT_INT4_SYM_SCALE, MAX_VALUE_NUM);
                Duplicate<float>(constInvScale, float(1) / DYNAMIC_QUANT_INT4_SYM_SCALE, MAX_VALUE_NUM);
            } else {
                Duplicate<float>(constScale, DYNAMIC_QUANT_INT8_SYM_SCALE, MAX_VALUE_NUM);
                Duplicate<float>(constInvScale, float(1) / DYNAMIC_QUANT_INT8_SYM_SCALE, MAX_VALUE_NUM);
            }
        }
    }

    __aicore__ inline void InitCommonParams()
    {
        lenHead = rowPerHeadCore * tilingData_.rowLen;
        lenTail = rowPerTailCore * tilingData_.rowLen;
        if (ORIG_DTYPE_Y == DT_INT4) {
            outLenHead = lenHead >> 1;
            outLenTail = lenTail >> 1;
        } else {
            outLenHead = lenHead;
            outLenTail = lenTail;
        }
    }

    __aicore__ inline void InitParams(GM_ADDR offset)
    {
        blockIdx = GetBlockIdx();
        if (offset != nullptr) {
            isAsymmetrical = true;
        }
        rowPerHeadCore = tilingData_.rowPerHeadCore;
        rowPerTailCore = tilingData_.rowPerTailCore;

        if (blockIdx < tilingData_.headCoreNum) {
            multiRowNum = tilingData_.multiRowNumHeadCore;
            loopCnt = rowPerHeadCore / multiRowNum;
            remainRow = rowPerHeadCore % multiRowNum;
        } else if (blockIdx >= tilingData_.headCoreNum && blockIdx < tilingData_.coreNum) {
            multiRowNum = tilingData_.multiRowNumTailCore;
            loopCnt = rowPerTailCore / multiRowNum;
            remainRow = rowPerTailCore % multiRowNum;
        } else {
        }
        sizeHalfLen = (tilingData_.rowLen + FIFTEEN) / SIXTEEN * SIXTEEN;
        rightPadding = sizeHalfLen - tilingData_.rowLen;
        if (rightPadding > 0) {
            isPad = true;
        }
        if (ORIG_DTYPE_Y == DT_INT4) {
            outAlignLen = (tilingData_.rowLen + SIXTY_THREE) / SIXTY_FOUR * SIXTY_FOUR;
        } else {
            outAlignLen = (tilingData_.rowLen + THIRTY_ONE) / THIRTY_TWO * THIRTY_TWO;
        }
        sizeFloatLen = (multiRowNum + SEVEN) / EIGHT * EIGHT;
        lenMultiRow = multiRowNum * sizeHalfLen;
        lenGMMultiRow = multiRowNum * tilingData_.rowLen;
        outLen = multiRowNum * outAlignLen;
        InitCommonParams();
    }

    __aicore__ inline void InitLargeShapeParams(GM_ADDR offset)
    {
        blockIdx = GetBlockIdx();
        if (offset != nullptr) {
            isAsymmetrical = true;
        }
        rowPerHeadCore = tilingData_.rowPerHeadCore;
        rowPerTailCore = tilingData_.rowPerTailCore;

        // 如果是大shape，一次只算一行，所以不存在尾行
        if (blockIdx < tilingData_.headCoreNum) {
            loopCnt = rowPerHeadCore;
        } else if (blockIdx >= tilingData_.headCoreNum && blockIdx < tilingData_.coreNum) {
            loopCnt = rowPerTailCore;
        } else {
        }
        // 尾轴切分，最后剩余的部分
        uint32_t sizeTailLen = (tilingData_.innerLoopTail + FIFTEEN) / SIXTEEN * SIXTEEN;
        rightPadding = sizeTailLen - tilingData_.innerLoopTail;
        sizeFloatLen = (loopCnt + SEVEN) / EIGHT * EIGHT;
        if (ORIG_DTYPE_Y == DT_INT4) {
            outAlignLen = (tilingData_.innerLoopEle + SIXTY_THREE) / SIXTY_FOUR * SIXTY_FOUR;
        } else {
            outAlignLen = tilingData_.innerLoopEle;
        }
        InitCommonParams();
    }

    __aicore__ inline void GetScale(float max_value, float& scale)
    {
        if (ORIG_DTYPE_Y == DT_INT4) {
            scale = DYNAMIC_QUANT_INT4_SYM_SCALE / max_value;
        } else {
            scale = DYNAMIC_QUANT_INT8_SYM_SCALE / max_value;
        }
    }

    __aicore__ inline void GetScaleAndOffset(float max_value, float min_value, float& scale, float& offset)
    {
        if (ORIG_DTYPE_Y == DT_INT4) {
            scale = GetMax((max_value - min_value) / DYNAMIC_QUANT_INT4_SCALE, DYNAMIC_QUANT_EPSINON);
            offset = DYNAMIC_QUANT_INT4_OFFSET - max_value / scale;
        } else {
            scale = GetMax((max_value - min_value) / DYNAMIC_QUANT_INT8_SCALE, DYNAMIC_QUANT_EPSINON);
            offset = DYNAMIC_QUANT_INT8_OFFSET - max_value / scale;
        }
    }

    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float>& src_local, uint32_t count)
    {
        uint64_t repsFp32 = count >> 6;       // 6 is cound / ELEM_PER_REP_FP32
        uint64_t offsetsFp32 = repsFp32 << 6; // 6 is repsFp32 * ELEM_PER_REP_FP32
        uint64_t remsFp32 = count & 0x3f;     // 0x3f 63, count % ELEM_PER_REP_FP32

        if (likely(repsFp32 > 1)) {
            // 8 is rep stride
            if (repsFp32 - 1 > MAX_REPEAT) {
                Max(src_local, src_local[ELEM_PER_REP_FP32], src_local, ELEM_PER_REP_FP32, MAX_REPEAT,
                    {1, 1, 1, 0, 8, 0});
                PipeBarrier<PIPE_V>();
                Max(src_local, src_local[ELEM_PER_REP_FP32 * MAX_REPEAT], src_local, ELEM_PER_REP_FP32, repsFp32 - MAX_REPEAT - 1,
                    {1, 1, 1, 0, 8, 0});
            } else {
                Max(src_local, src_local[ELEM_PER_REP_FP32], src_local, ELEM_PER_REP_FP32, repsFp32 - 1,
                    {1, 1, 1, 0, 8, 0});
            }
            PipeBarrier<PIPE_V>();
        }
        if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
            Max(src_local, src_local[offsetsFp32], src_local, remsFp32, 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        uint32_t mask = repsFp32 > 0 ? ELEM_PER_REP_FP32 : count;
        // 8 is rep stride
        WholeReduceMax(src_local, src_local, mask, 1, 8, 1, 8);
    }

    __aicore__ inline void ReduceMaxInplace(const LocalTensor<half>& src_local, uint32_t count)
    {
        uint64_t repsHalf = count >> 7;       // 7 is count / ELEM_PER_REP_HALF
        uint64_t offsetsHalf = repsHalf << 7; // 6 is repsHalf * ELEM_PER_REP_HALF
        uint64_t remsHalf = count & 0x7f;     // 0x7f 127, count % ELEM_PER_REP_HALF

        if (likely(repsHalf > 1)) {
            // 8 is rep stride
            Max(src_local, src_local[ELEM_PER_REP_HALF], src_local, ELEM_PER_REP_HALF, repsHalf - 1,
                {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        if (unlikely(remsHalf > 0) && unlikely(offsetsHalf > 0)) {
            Max(src_local, src_local[offsetsHalf], src_local, remsHalf, 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        uint32_t mask = repsHalf > 0 ? ELEM_PER_REP_HALF : count;
        // 8 is rep stride
        WholeReduceMax(src_local, src_local, mask, 1, 8, 1, 8);
    }

    __aicore__ inline float GetMax(float a, float b)
    {
        return a > b ? a : b;
    }

    template <typename T1, typename T2>
    __aicore__ inline auto GetMin(T1 a, T2 b)
    {
        return a < b ? a : b;
    };
protected:
    TPipe* pPipe = nullptr;
    /* tiling data */
    DynamicQuantTilingData tilingData_;

    /* variable */
    uint32_t blockIdx, sizeFloatLen, sizeHalfLen, outAlignLen, multiRowNum;
    uint32_t rowPerHeadCore, rowPerTailCore;
    uint32_t lenHead, lenTail, lenMultiRow, lenGMMultiRow, outLen;
    uint32_t outInt8PaddingLen, outInt4PaddingLen;
    uint32_t outLenHead, outLenTail, originOutSize;
    uint32_t loopCnt = 0;
    uint32_t remainRow = 0;
    uint8_t rightPadding = 0;
    bool isPad = false;
    bool isAsymmetrical = false;

    AscendC::TBuf<AscendC::TPosition::VECCALC> quantTemp, fp32_buf_, tempCastUb;

    /* local memory */
    LocalTensor<float> constScale, constInvScale, scaleTmp, brcbTmp, quantScaleTmp;
};
} // namespace DynamicQuantNDOpt
#endif // DynamicQuantBase