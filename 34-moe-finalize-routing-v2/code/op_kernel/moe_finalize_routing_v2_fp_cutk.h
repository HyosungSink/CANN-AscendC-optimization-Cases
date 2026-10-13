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
 * \file moe_finalize_routing_fp_cutk.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_V2_FP_CUTK
#define MOE_FINALIZE_ROUTING_V2_FP_CUTK

#include "moe_finalize_routing_v2_common.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MoeFinalizeRoutingV2 {

using namespace AscendC;

template <typename T, const bool ISBIASEXIST>
class MoeFinalizeRoutingV2FpCutK
{
public:
    __aicore__ inline MoeFinalizeRoutingV2FpCutK(const MoeFinalizeRoutingV2TilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(
        GM_ADDR expandedPermutedRows, GM_ADDR expandedSrcToDstRow, GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias,
        GM_ADDR scales, GM_ADDR expertForSourceRow, GM_ADDR out, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CkechColAlignment();
    __aicore__ inline void CopyIn(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void Compute(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void CopyOut(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline int64_t AlignmentProcess(int64_t param);
    __aicore__ inline int64_t PadProcessT(int64_t param);

private:
    TPipe& pipe_;
    const MoeFinalizeRoutingV2TilingData& tilingData_;
    TQue<QuePosition::VECIN, 1> skip1Queue_;
    TQue<QuePosition::VECIN, 1> skip2Queue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    TBuf<QuePosition::VECCALC> expertForSourceRowBuf_;
    TBuf<QuePosition::VECCALC> scalesBuf_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsBufDb0_;
    TBuf<QuePosition::VECCALC> biasBufDb0_;

    GlobalTensor<T> gmExpandedPermutedRows_;
    GlobalTensor<T> gmSkip1_;
    GlobalTensor<T> gmSkip2_;
    GlobalTensor<T> gmBias_;
    GlobalTensor<T> gmScales_;
    GlobalTensor<int32_t> gmExpandedSrcToDstRow_;
    GlobalTensor<int32_t> gmExpertForSourceRow_;
    GlobalTensor<T> gmOut_;

    int64_t workspace_{0};

    // input number
    int64_t inputSkipIdx_{0};
    int64_t inputScalesAndExpertIdx_{0};
    int64_t outputIdx_{0};
    int64_t curCoreHandleNumPerLoop_{0};
    int64_t biasInCore_{0};

    const int64_t onceAlgnNum_{ONE_BLK_SIZE / static_cast<int64_t>(sizeof(T))};

    bool isPadNormalH_ = false;
    bool isPadUnnormalH_ = false;
    bool isPadNormalK_ = false;
    bool isPadUnnormalK_ = false;
    bool isPadNormalKInt32_ = false;
    bool isPadUnnormalKInt32_ = false;
    bool initFlag_ = true;
    bool skipsAllNone_ = false;
    uint16_t rightPaddingNormalH_ = 0;
    uint16_t rightPaddingUnnormalH_ = 0;
    uint16_t rightPaddingNormalK_ = 0;
    uint16_t rightPaddingUnnormalK_ = 0;
    uint16_t rightPaddingNormalKInt32_ = 0;
    uint16_t rightPaddingUnnormalKInt32_ = 0;
};

template <typename T, const bool ISBIASEXIST>
__aicore__ inline int64_t MoeFinalizeRoutingV2FpCutK<T, ISBIASEXIST>::AlignmentProcess(int64_t param)
{
    return (param + onceAlgnNum_ - 1) / onceAlgnNum_ * onceAlgnNum_;
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline int64_t MoeFinalizeRoutingV2FpCutK<T, ISBIASEXIST>::PadProcessT(int64_t param)
{
    return onceAlgnNum_ - param % onceAlgnNum_;
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2FpCutK<T, ISBIASEXIST>::CkechColAlignment()
{
    if (tilingData_.normalH * sizeof(T) % ONE_BLK_SIZE) {
        isPadNormalH_ = true;
        rightPaddingNormalH_ = PadProcessT(tilingData_.normalH);
    }

    if (tilingData_.unnormalH * sizeof(T) % ONE_BLK_SIZE) {
        isPadUnnormalH_ = true;
        rightPaddingUnnormalH_ = PadProcessT(tilingData_.unnormalH);
    }

    if (tilingData_.normalK * sizeof(T) % ONE_BLK_SIZE) {
        isPadNormalK_ = true;
        rightPaddingNormalK_ = PadProcessT(tilingData_.normalK);
    }

    if (tilingData_.unnormalK * sizeof(T) % ONE_BLK_SIZE) {
        isPadUnnormalK_ = true;
        rightPaddingUnnormalK_ = PadProcessT(tilingData_.unnormalK);
    }

    if (tilingData_.normalK * INT32_BYTES % ONE_BLK_SIZE) {
        isPadNormalKInt32_ = true;
        rightPaddingNormalKInt32_ = PadProcessInt32(tilingData_.normalK);
    }

    if (tilingData_.unnormalK * INT32_BYTES % ONE_BLK_SIZE) {
        isPadUnnormalKInt32_ = true;
        rightPaddingUnnormalKInt32_ = PadProcessInt32(tilingData_.unnormalK);
    }
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2FpCutK<T, ISBIASEXIST>::Init(
    GM_ADDR expandedPermutedRows, GM_ADDR expandedSrcToDstRow, GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias,
    GM_ADDR scales, GM_ADDR expertForSourceRow, GM_ADDR out, GM_ADDR workspace)
{
    if (GetBlockIdx() + 1 == tilingData_.usedCoreNum) {
        curCoreHandleNumPerLoop_ = tilingData_.tailCoreHandleNumPerLoop;
    } else {
        curCoreHandleNumPerLoop_ = tilingData_.normalCoreHandleNumPerLoop;
    }
    biasInCore_ = GetBlockIdx() * tilingData_.normalCoreHandleNum;

    // 检查要处理的列数是否对齐以及应该如何对齐
    CkechColAlignment();

    // gmInput分核 && 输入偏移量初始化
    inputSkipIdx_ = biasInCore_ * tilingData_.H;
    if (tilingData_.skip1IsNull == 0) {
        gmSkip1_.SetGlobalBuffer((__gm__ T*)skip1 + inputSkipIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);
    }
    pipe_.InitBuffer(
        skip1Queue_, 1, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(T));
    if (tilingData_.skip2IsNull == 0) {
        gmSkip2_.SetGlobalBuffer((__gm__ T*)skip2 + inputSkipIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);
        pipe_.InitBuffer(
            skip2Queue_, 1, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(T));
    }

    inputScalesAndExpertIdx_ = biasInCore_ * tilingData_.K;
    if (tilingData_.scalesIsNull == 0) {
        gmScales_.SetGlobalBuffer(
            (__gm__ T*)scales + inputScalesAndExpertIdx_, tilingData_.normalCoreHandleNum * tilingData_.K);
        pipe_.InitBuffer(scalesBuf_, AlignmentProcess(tilingData_.normalK) * sizeof(T));
    }

    gmExpandedSrcToDstRow_.SetGlobalBuffer(
        (__gm__ int32_t*)expandedSrcToDstRow, tilingData_.totalRowNum * tilingData_.K);

    outputIdx_ = biasInCore_ * tilingData_.H;
    gmOut_.SetGlobalBuffer((__gm__ T*)out + outputIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);

    gmExpandedPermutedRows_.SetGlobalBuffer(
        (__gm__ T*)expandedPermutedRows, tilingData_.totalRowNum * tilingData_.K * tilingData_.H);
    if constexpr (ISBIASEXIST) {
        gmBias_.SetGlobalBuffer((__gm__ T*)bias, tilingData_.biasRowNum * tilingData_.H);
        pipe_.InitBuffer(biasBufDb0_, AlignmentProcess(tilingData_.normalH) * sizeof(T));
        gmExpertForSourceRow_.SetGlobalBuffer(
            (__gm__ int32_t*)expertForSourceRow + inputScalesAndExpertIdx_,
            tilingData_.normalCoreHandleNum * tilingData_.K);
        pipe_.InitBuffer(expertForSourceRowBuf_, Int32AlignmentProcess(tilingData_.normalK) * sizeof(int32_t));
    }

    // 申请 buffer 空间
    pipe_.InitBuffer(
        outQueue_, BUFFER_NUM,
        tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(T));

    pipe_.InitBuffer(expandedPermutedRowsBufDb0_, AlignmentProcess(tilingData_.normalH) * sizeof(T));
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2FpCutK<T, ISBIASEXIST>::CopyIn(int64_t nLoopIdx, int64_t curRepeatTimes)
{
    bool isNormalH = (nLoopIdx + 1) % (tilingData_.hSliceNum + 1) != 0;
    int64_t bias = isNormalH ? (nLoopIdx % tilingData_.hSliceNum) * tilingData_.normalH :
                               tilingData_.hSliceNum * tilingData_.normalH;
    int64_t dataLen = isNormalH ? tilingData_.normalH : tilingData_.unnormalH;
    int64_t rightPaddingH = isNormalH ? rightPaddingNormalH_ : rightPaddingUnnormalH_;
    bool isPadH = isNormalH ? isPadNormalH_ : isPadUnnormalH_;
    LocalTensor<T> skip1Local;

    DataCopyParams copyParams{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};

    if (tilingData_.hSliceNum == 0) {
        bias = 0;
    }
    if (tilingData_.skip1IsNull == 0) {
        skip1Local = skip1Queue_.AllocTensor<T>();
        DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (
            skip1Local,
            gmSkip1_[nLoopIdx / (tilingData_.hSliceNum + 1) * curCoreHandleNumPerLoop_ * tilingData_.H + bias],
            copyParams, padParams);
        skip1Queue_.EnQue(skip1Local);
    }
    LocalTensor<T> skip2Local;
    if (tilingData_.skip2IsNull == 0) {
        skip2Local = skip2Queue_.AllocTensor<T>();
        DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (
            skip2Local,
            gmSkip2_[nLoopIdx / (tilingData_.hSliceNum + 1) * curCoreHandleNumPerLoop_ * tilingData_.H + bias],
            copyParams, padParams);
        skip2Queue_.EnQue(skip2Local);
    }
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2FpCutK<T, ISBIASEXIST>::Compute(int64_t nLoopIdx, int64_t curRepeatTimes)
{
    bool isNormalH = (nLoopIdx + 1) % (tilingData_.hSliceNum + 1) != 0;
    int64_t bias = isNormalH ? (nLoopIdx % tilingData_.hSliceNum) * tilingData_.normalH :
                               tilingData_.hSliceNum * tilingData_.normalH;
    if (tilingData_.hSliceNum == 0) {
        bias = 0;
    }
    int64_t dataLen = isNormalH ? tilingData_.normalH : tilingData_.unnormalH;
    int64_t rightPaddingH = isNormalH ? rightPaddingNormalH_ : rightPaddingUnnormalH_;
    bool isPadH = isNormalH ? isPadNormalH_ : isPadUnnormalH_;
    int64_t biasInRow = nLoopIdx / (tilingData_.hSliceNum + 1) * curCoreHandleNumPerLoop_;
    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();
    LocalTensor<T> skip1Local;
    if (tilingData_.skip1IsNull == 0) {
        skip1Local = skip1Queue_.DeQue<T>();
    } else {
        skip1Local = skip1Queue_.AllocTensor<T>();
    }

    LocalTensor<T> skip2Local;
    if (tilingData_.skip2IsNull == 0) {
        skip2Local = skip2Queue_.DeQue<T>();
    }
    if ((tilingData_.skip1IsNull == 0) && (tilingData_.skip2IsNull == 0)) {
        Add(outLocal, skip1Local, skip2Local, curRepeatTimes * AlignmentProcess(dataLen));
    } else if ((tilingData_.skip1IsNull == 0) && (tilingData_.skip2IsNull == 1)) {
        Adds(outLocal, skip1Local, (T)0, curRepeatTimes * AlignmentProcess(dataLen));
    } else if ((tilingData_.skip1IsNull == 1) && (tilingData_.skip2IsNull == 0)) {
        Adds(outLocal, skip2Local, (T)0, curRepeatTimes * AlignmentProcess(dataLen));
    } else {
        initFlag_ = false;
        skipsAllNone_ = true;
    }

    LocalTensor<T> expandedPermutedTmpUb = expandedPermutedRowsBufDb0_.Get<T>();
    LocalTensor<T> biasTmpUb;
    if constexpr (ISBIASEXIST) {
        biasTmpUb = biasBufDb0_.Get<T>();
    }

    DataCopyParams copyParams{1, static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};
    for (int64_t i = 0; i < curRepeatTimes; i++) {
        int64_t outRowIndex = i * AlignmentProcess(dataLen);
        int64_t len = tilingData_.normalK;
        bool isPadK = isPadNormalK_;
        bool isPadKInt32 = isPadNormalKInt32_;
        uint16_t rightPaddingK = rightPaddingNormalK_;
        uint16_t rightPaddingKInt32 = rightPaddingNormalKInt32_;
        if (skipsAllNone_) {
            initFlag_ = false;
        }
        for (int64_t n = 0; n < tilingData_.kSliceNum; n++) {
            if (n == tilingData_.kSliceNum - 1) {
                len = tilingData_.unnormalK;
                isPadK = isPadUnnormalK_;
                isPadKInt32 = isPadUnnormalKInt32_;
                rightPaddingK = rightPaddingUnnormalK_;
                rightPaddingKInt32 = rightPaddingUnnormalKInt32_;
            }
            int64_t biasOfK = n * tilingData_.normalK;
            // ---------------------------- [Scales] -------------------------------
            LocalTensor<T> scalesLocal;
            if (tilingData_.scalesIsNull == 0) {
                scalesLocal = scalesBuf_.Get<T>();
                DataCopyParams copyParamsScales{1, static_cast<uint16_t>(len * sizeof(T)), 0, 0};
                DataCopyPadParams padParamsScales{isPadK, 0, static_cast<uint8_t>(rightPaddingK), 0};
                DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (
                    scalesLocal, gmScales_[biasInRow * tilingData_.K + i * tilingData_.K + biasOfK], copyParamsScales,
                    padParamsScales);
            }
            // ---------------------------- [Expert] -------------------------------
            LocalTensor<int32_t> expertForSourceRowLocal;
            if constexpr (ISBIASEXIST) {
                expertForSourceRowLocal = expertForSourceRowBuf_.Get<int32_t>();
                DataCopyParams copyParamsExpert{1, static_cast<uint16_t>(len * sizeof(int32_t)), 0, 0};
                DataCopyPadParams padParamsExpert{isPadKInt32, 0, static_cast<uint8_t>(rightPaddingKInt32), 0};
                DataCopyPadCustom<int32_t, DataCopyParams, DataCopyPadParams>(
                    expertForSourceRowLocal,
                    gmExpertForSourceRow_[biasInRow * tilingData_.K + i * tilingData_.K + biasOfK], copyParamsExpert,
                    padParamsExpert);
            }
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            SetFlag<HardEvent::V_S>(EVENT_ID1);
            for (int64_t j = 0; j < len; j++) {
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                int64_t expandedSrcToDstRowIndex = 0;
                if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
                    expandedSrcToDstRowIndex = biasInRow + i + (j + biasOfK) * tilingData_.totalRowNum + biasInCore_;
                } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
                    expandedSrcToDstRowIndex = biasInRow * tilingData_.K + i * tilingData_.K + j + biasOfK +
                                               GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.K;
                }
                int64_t expandedPermutedRowsIndex = gmExpandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndex);
                SetFlag<HardEvent::S_MTE2>(EVENT_ID0);

                WaitFlag<HardEvent::V_S>(EVENT_ID1);
                int64_t biasIndexDb0 = 0;
                if constexpr (ISBIASEXIST) {
                    biasIndexDb0 = expertForSourceRowLocal.GetValue(j);
                }
                T scalesValDb0 = 1.0;
                if (tilingData_.scalesIsNull == 0) {
                    scalesValDb0 = scalesLocal.GetValue(j);
                }
                SetFlag<HardEvent::S_MTE2>(EVENT_ID1);

                WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
                if (expandedPermutedRowsIndex != INVALID_ROW_INDEX) {
                    DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (
                        expandedPermutedTmpUb,
                        gmExpandedPermutedRows_[expandedPermutedRowsIndex * tilingData_.H + bias], copyParams,
                        padParams);
                }
                if constexpr (ISBIASEXIST) {
                    DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (biasTmpUb, gmBias_[biasIndexDb0 * tilingData_.H + bias], copyParams, padParams);
                }
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                if (expandedPermutedRowsIndex == INVALID_ROW_INDEX &&
                    (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
                    Duplicate(expandedPermutedTmpUb, (T)0, dataLen);
                    PipeBarrier<PIPE_V>();
                } else {
                    if constexpr (ISBIASEXIST) {
                        Add(skip1Local, expandedPermutedTmpUb, biasTmpUb, dataLen);
                    } else {
                        Adds(skip1Local, expandedPermutedTmpUb, (T)0, dataLen);
                    }
                }
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                PipeBarrier<PIPE_V>();
                if (!initFlag_) {
                    if (expandedPermutedRowsIndex == INVALID_ROW_INDEX &&
                        (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
                        Duplicate(outLocal[outRowIndex], (T)0, dataLen);
                    } else {
                        Muls(outLocal[outRowIndex], skip1Local, scalesValDb0, dataLen);
                    }
                    initFlag_ = true;
                    SetFlag<HardEvent::V_S>(EVENT_ID1);
                } else {
                    if (expandedPermutedRowsIndex == INVALID_ROW_INDEX &&
                        (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
                        SetFlag<HardEvent::V_S>(EVENT_ID1);
                        PipeBarrier<PIPE_V>();
                    } else {
                        Muls(skip1Local, skip1Local, scalesValDb0, dataLen);
                        SetFlag<HardEvent::V_S>(EVENT_ID1);
                        PipeBarrier<PIPE_V>();
                        Add(outLocal[outRowIndex], outLocal[outRowIndex], skip1Local, dataLen);
                    }
                }
            }
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID1);
        }
    }
    outQueue_.EnQue(outLocal);

    skip1Queue_.FreeTensor(skip1Local);
    if (tilingData_.skip2IsNull == 0) {
        skip2Queue_.FreeTensor(skip2Local);
    }
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2FpCutK<T, ISBIASEXIST>::CopyOut(int64_t nLoopIdx, int64_t curRepeatTimes)
{
    bool isNormalH = (nLoopIdx + 1) % (tilingData_.hSliceNum + 1) != 0;
    int64_t bias = isNormalH ? (nLoopIdx % tilingData_.hSliceNum) * tilingData_.normalH :
                               tilingData_.hSliceNum * tilingData_.normalH;
    if (tilingData_.hSliceNum == 0) {
        bias = 0;
    }
    int64_t dataLen = isNormalH ? tilingData_.normalH : tilingData_.unnormalH;
    LocalTensor<T> outLocal = outQueue_.DeQue<T>();
    DataCopyParams copyParams{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
    DataCopyPadCustom<T>(
        gmOut_[nLoopIdx / (tilingData_.hSliceNum + 1) * curCoreHandleNumPerLoop_ * tilingData_.H + bias], outLocal,
        copyParams);
    outQueue_.FreeTensor(outLocal);
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2FpCutK<T, ISBIASEXIST>::Process()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNum) {
        return;
    }
    int64_t loopCount = tilingData_.normalCoreLoopNum;
    int64_t tailLoopBlock = tilingData_.normalCoreHandleNumTailLoop;
    if ((GetBlockIdx() + 1) == tilingData_.usedCoreNum) {
        loopCount = tilingData_.tailCoreLoopNum;
        tailLoopBlock = tilingData_.tailCoreHandleNumTailLoop;
    }

    for (int64_t n = 0; n < loopCount - 1; n++) {
        CopyIn(n, curCoreHandleNumPerLoop_);
        Compute(n, curCoreHandleNumPerLoop_);
        CopyOut(n, curCoreHandleNumPerLoop_);
    }
    // tail loop
    {
        CopyIn(loopCount - 1, tailLoopBlock);
        Compute(loopCount - 1, tailLoopBlock);
        CopyOut(loopCount - 1, tailLoopBlock);
    }
}

} // namespace MoeFinalizeRoutingV2
#endif // MOE_FINALIZE_ROUTING_V2_FP_CUTK
