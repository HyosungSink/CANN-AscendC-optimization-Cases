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
 * \file moe_finalize_routing_v2_bf16_cuth_k2.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_V2_BF16_CUTH_K_TWO
#define MOE_FINALIZE_ROUTING_V2_BF16_CUTH_K_TWO

#include "moe_finalize_routing_v2_common.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MoeFinalizeRoutingV2 {

using namespace AscendC;

template <typename T, typename TS, const bool ISBIASEXIST>
class MoeFinalizeRoutingV2Bf16CuthK2
{
public:
    __aicore__ inline MoeFinalizeRoutingV2Bf16CuthK2(const MoeFinalizeRoutingV2TilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(
        GM_ADDR expandedPermutedRows, GM_ADDR expandedSrcToDstRow, GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias,
        GM_ADDR scales, GM_ADDR expertForSourceRow, GM_ADDR out, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CheckColAlignment();
    __aicore__ inline void CopyIn(int64_t nLoopIdx, int64_t bias, int64_t dataLen, bool isPadH, int64_t rightPaddingH);
    __aicore__ inline void Compute(int64_t nLoopIdx, int64_t bias, int64_t dataLen, bool isPadH, int64_t rightPaddingH);
    __aicore__ inline void CopyOut(int64_t nLoopIdx, int64_t bias, int64_t dataLen);
    __aicore__ inline int64_t AlignmentProcess(int64_t param);
    __aicore__ inline int64_t AlignmentProcessScale(int64_t param);
    __aicore__ inline int64_t PadProcessT(int64_t param);
    __aicore__ inline int64_t PadProcessScale(int64_t param);

private:
    TPipe& pipe_;
    const MoeFinalizeRoutingV2TilingData& tilingData_;
    TQue<QuePosition::VECIN, 1> skip1Queue_;
    TQue<QuePosition::VECIN, 1> skip2Queue_;
    TQue<QuePosition::VECIN, 1> scalesQueue_;
    TQue<QuePosition::VECIN, 1> expertForSourceRowQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    TBuf<QuePosition::VECCALC> expandedPermutedRowsBufDb0_;
    TBuf<QuePosition::VECCALC> biasBufDb0_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsBufDb1_;
    TBuf<QuePosition::VECCALC> biasBufDb1_;

    TBuf<QuePosition::VECCALC> skip1CastBuf_;
    TBuf<QuePosition::VECCALC> skip2CastBuf_;
    TBuf<QuePosition::VECCALC> biasCastBuf0_;
    TBuf<QuePosition::VECCALC> biasCastBuf1_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsCastBuf0_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsCastBuf1_;

    GlobalTensor<T> gmExpandedPermutedRows_;
    GlobalTensor<T> gmSkip1_;
    GlobalTensor<T> gmSkip2_;
    GlobalTensor<T> gmBias_;
    GlobalTensor<TS> gmScales_;
    GlobalTensor<int32_t> gmExpertForSourceRow_;
    GlobalTensor<int32_t> gmExpandedSrcToDstRow_;
    GlobalTensor<T> gmOut_;

    int64_t workspace_{0};

    // input number
    int64_t inputSkipIdx_{0};
    int64_t inputScalesAndExpertIdx_{0};
    int64_t curCoreHandleNumPerLoop_{0};
    int64_t outputIdx_{0};

    const int64_t onceAlgnNum_{ONE_BLK_SIZE / static_cast<int64_t>(sizeof(T))};
    const int64_t onceAlgnNumScale_{ONE_BLK_SIZE / static_cast<int64_t>(sizeof(TS))};

    bool isPadNormalH_ = false;
    bool isPadUnnormalH_ = false;
    bool isPadK_ = false;
    bool isPadKInt32_ = false;
    bool initFlag_ = true;
    uint16_t rightPaddingNormalH_ = 0;
    uint16_t rightPaddingUnnormalH_ = 0;
    uint16_t rightPaddingK_ = 0;
    uint16_t rightPaddingKInt32_ = 0;
};

template <typename T, typename TS, const bool ISBIASEXIST>
__aicore__ inline int64_t MoeFinalizeRoutingV2Bf16CuthK2<T, TS, ISBIASEXIST>::AlignmentProcess(int64_t param)
{
    return (param + onceAlgnNum_ - 1) / onceAlgnNum_ * onceAlgnNum_;
}

template <typename T, typename TS, const bool ISBIASEXIST>
__aicore__ inline int64_t MoeFinalizeRoutingV2Bf16CuthK2<T, TS, ISBIASEXIST>::AlignmentProcessScale(int64_t param)
{
    return (param + onceAlgnNumScale_ - 1) / onceAlgnNumScale_ * onceAlgnNumScale_;
}

template <typename T, typename TS, const bool ISBIASEXIST>
__aicore__ inline int64_t MoeFinalizeRoutingV2Bf16CuthK2<T, TS, ISBIASEXIST>::PadProcessT(int64_t param)
{
    return onceAlgnNum_ - param % onceAlgnNum_;
}

template <typename T, typename TS, const bool ISBIASEXIST>
__aicore__ inline int64_t MoeFinalizeRoutingV2Bf16CuthK2<T, TS, ISBIASEXIST>::PadProcessScale(int64_t param)
{
    return onceAlgnNumScale_ - param % onceAlgnNumScale_;
}

template <typename T, typename TS, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2Bf16CuthK2<T, TS, ISBIASEXIST>::CheckColAlignment()
{
    if (tilingData_.normalH * sizeof(T) % ONE_BLK_SIZE) {
        isPadNormalH_ = true;
        rightPaddingNormalH_ = PadProcessT(tilingData_.normalH);
    }

    if (tilingData_.unnormalH * sizeof(T) % ONE_BLK_SIZE) {
        isPadUnnormalH_ = true;
        rightPaddingUnnormalH_ = PadProcessT(tilingData_.unnormalH);
    }

    if (tilingData_.K * sizeof(TS) % ONE_BLK_SIZE) {
        isPadK_ = true;
        rightPaddingK_ = PadProcessScale(tilingData_.K);
    }

    if (tilingData_.K * INT32_BYTES % ONE_BLK_SIZE) {
        isPadKInt32_ = true;
        rightPaddingKInt32_ = PadProcessInt32(tilingData_.K);
    }
}

template <typename T, typename TS, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2Bf16CuthK2<T, TS, ISBIASEXIST>::Init(
    GM_ADDR expandedPermutedRows, GM_ADDR expandedSrcToDstRow, GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias,
    GM_ADDR scales, GM_ADDR expertForSourceRow, GM_ADDR out, GM_ADDR workspace)
{
    if (GetBlockIdx() + 1 == tilingData_.usedCoreNum) {
        curCoreHandleNumPerLoop_ = tilingData_.tailCoreHandleNumPerLoop;
    } else {
        curCoreHandleNumPerLoop_ = tilingData_.normalCoreHandleNumPerLoop;
    }

    CheckColAlignment();

    // gmInput分核 && 输入偏移量初始化
    inputSkipIdx_ = GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.H;
    if (tilingData_.skip1IsNull == 0) {
        gmSkip1_.SetGlobalBuffer((__gm__ T*)skip1 + inputSkipIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);
    }
    pipe_.InitBuffer(
        skip1Queue_, 1, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(T));
    if (tilingData_.skip2IsNull == 0) {
        gmSkip2_.SetGlobalBuffer((__gm__ T*)skip2 + inputSkipIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);
        pipe_.InitBuffer(
            skip2Queue_, 1, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(T));
        pipe_.InitBuffer(
            skip2CastBuf_,
            tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(float));
    }

    inputScalesAndExpertIdx_ = GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.K;
    if (tilingData_.scalesIsNull == 0) {
        gmScales_.SetGlobalBuffer(
            (__gm__ TS*)scales + inputScalesAndExpertIdx_, tilingData_.normalCoreHandleNum * tilingData_.K);
        pipe_.InitBuffer(
            scalesQueue_, BUFFER_NUM,
            tilingData_.normalCoreHandleNumPerLoop * AlignmentProcessScale(tilingData_.K) * sizeof(TS));
    }

    gmExpandedSrcToDstRow_.SetGlobalBuffer(
        (__gm__ int32_t*)expandedSrcToDstRow, tilingData_.totalRowNum * tilingData_.K);

    outputIdx_ = GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.H;
    gmOut_.SetGlobalBuffer((__gm__ T*)out + outputIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);

    gmExpandedPermutedRows_.SetGlobalBuffer(
        (__gm__ T*)expandedPermutedRows, tilingData_.totalRowNum * tilingData_.K * tilingData_.H);

    if constexpr (ISBIASEXIST) {
        gmExpertForSourceRow_.SetGlobalBuffer(
            (__gm__ int32_t*)expertForSourceRow + inputScalesAndExpertIdx_,
            tilingData_.normalCoreHandleNum * tilingData_.K);
        pipe_.InitBuffer(
            expertForSourceRowQueue_, BUFFER_NUM,
            tilingData_.normalCoreHandleNumPerLoop * Int32AlignmentProcess(tilingData_.K) * sizeof(int32_t));
        gmBias_.SetGlobalBuffer((__gm__ T*)bias, tilingData_.biasRowNum * tilingData_.H);
        pipe_.InitBuffer(biasBufDb0_, AlignmentProcess(tilingData_.normalH) * sizeof(T));
        pipe_.InitBuffer(biasBufDb1_, AlignmentProcess(tilingData_.normalH) * sizeof(T));
        pipe_.InitBuffer(biasCastBuf0_, AlignmentProcess(tilingData_.normalH) * sizeof(float));
        pipe_.InitBuffer(biasCastBuf1_, AlignmentProcess(tilingData_.normalH) * sizeof(float));
    }

    // 申请 buffer 空间
    pipe_.InitBuffer(
        outQueue_, BUFFER_NUM,
        tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(T));

    pipe_.InitBuffer(expandedPermutedRowsBufDb0_, AlignmentProcess(tilingData_.normalH) * sizeof(T));

    pipe_.InitBuffer(expandedPermutedRowsBufDb1_, AlignmentProcess(tilingData_.normalH) * sizeof(T));

    pipe_.InitBuffer(
        skip1CastBuf_, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(float));
    pipe_.InitBuffer(expandedPermutedRowsCastBuf0_, AlignmentProcess(tilingData_.normalH) * sizeof(float));

    pipe_.InitBuffer(expandedPermutedRowsCastBuf1_, AlignmentProcess(tilingData_.normalH) * sizeof(float));
}

template <typename T, typename TS, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2Bf16CuthK2<T, TS, ISBIASEXIST>::CopyIn(
    int64_t nLoopIdx, int64_t bias, int64_t dataLen, bool isPadH, int64_t rightPaddingH)
{
    LocalTensor<TS> scalesLocal;
    LocalTensor<int32_t> expertForSourceRowLocal;
    LocalTensor<T> skip1Local, skip2Local;

    if (tilingData_.scalesIsNull == 0)
        scalesLocal = scalesQueue_.AllocTensor<TS>();
    if constexpr (ISBIASEXIST)
        expertForSourceRowLocal = expertForSourceRowQueue_.AllocTensor<int32_t>();
    if (tilingData_.skip1IsNull == 0)
        skip1Local = skip1Queue_.AllocTensor<T>();
    if (tilingData_.skip2IsNull == 0)
        skip2Local = skip2Queue_.AllocTensor<T>();

    // ---------------------------- [Skip] -------------------------------
    DataCopyParams copyParamsSkip{1, static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
    DataCopyPadParams padParamsSkip{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};
    if (tilingData_.skip1IsNull == 0) {
        DataCopyPad(
            skip1Local, gmSkip1_[nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.H + bias], copyParamsSkip,
            padParamsSkip);
    }
    if (tilingData_.skip2IsNull == 0) {
        DataCopyPad(
            skip2Local, gmSkip2_[nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.H + bias], copyParamsSkip,
            padParamsSkip);
    }

    // ---------------------------- [Scales] -------------------------------
    if (tilingData_.scalesIsNull == 0) {
        DataCopyParams copyParamsScales{1, static_cast<uint16_t>(tilingData_.K * sizeof(TS)), 0, 0};
        DataCopyPadParams padParamsScales{isPadK_, 0, static_cast<uint8_t>(rightPaddingK_), 0};
        DataCopyPad(
            scalesLocal, gmScales_[nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.K], copyParamsScales,
            padParamsScales);
    }
    // ---------------------------- [Expert] -------------------------------
    if constexpr (ISBIASEXIST) {
        DataCopyParams copyParamsExpert{1, static_cast<uint16_t>(tilingData_.K * sizeof(int32_t)), 0, 0};
        DataCopyPadParams padParamsExpert{isPadKInt32_, 0, static_cast<uint8_t>(rightPaddingKInt32_), 0};
        DataCopyPad(
            expertForSourceRowLocal, gmExpertForSourceRow_[nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.K],
            copyParamsExpert, padParamsExpert);
    }
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);

    COPY_IN_ENQUE();
}

template <typename T, typename TS, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2Bf16CuthK2<T, TS, ISBIASEXIST>::Compute(
    int64_t nLoopIdx, int64_t bias, int64_t dataLen, bool isPadH, int64_t rightPaddingH)
{
    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();

    LocalTensor<TS> scalesLocal;
    if (tilingData_.scalesIsNull == 0) {
        scalesLocal = scalesQueue_.DeQue<TS>();
    }

    LocalTensor<float> skip1CastUb = skip1CastBuf_.Get<float>();
    LocalTensor<T> skip1Local;
    if (tilingData_.skip1IsNull == 0) {
        skip1Local = skip1Queue_.DeQue<T>();
        Cast(skip1CastUb, skip1Local, RoundMode::CAST_NONE, AlignmentProcess(dataLen));
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<T> skip2Local;
    LocalTensor<float> skip2CastUb;
    if (tilingData_.skip2IsNull == 0) {
        skip2Local = skip2Queue_.DeQue<T>();
        skip2CastUb = skip2CastBuf_.Get<float>();
        Cast(skip2CastUb, skip2Local, RoundMode::CAST_NONE, AlignmentProcess(dataLen));
        PipeBarrier<PIPE_V>();
    }
    if ((tilingData_.skip1IsNull == 0) && (tilingData_.skip2IsNull == 0)) {
        Add(skip1CastUb, skip1CastUb, skip2CastUb, AlignmentProcess(dataLen));
    } else if ((tilingData_.skip1IsNull == 0) && (tilingData_.skip2IsNull == 1)) {
        Adds(skip1CastUb, skip1CastUb, float(0.0), AlignmentProcess(dataLen));
    } else if ((tilingData_.skip1IsNull == 1) && (tilingData_.skip2IsNull == 0)) {
        Adds(skip1CastUb, skip2CastUb, float(0.0), AlignmentProcess(dataLen));
    } else {
        initFlag_ = false;
    }

    LocalTensor<T> expandedPermutedTmpUbDb0 = expandedPermutedRowsBufDb0_.Get<T>();

    LocalTensor<T> expandedPermutedTmpUbDb1 = expandedPermutedRowsBufDb1_.Get<T>();

    LocalTensor<float> expandedPermutedRowsCastUb0 = expandedPermutedRowsCastBuf0_.Get<float>();

    LocalTensor<float> expandedPermutedRowsCastUb1 = expandedPermutedRowsCastBuf1_.Get<float>();

    LocalTensor<int32_t> expertForSourceRowLocal;
    LocalTensor<T> biasTmpUbDb0;
    LocalTensor<T> biasTmpUbDb1;
    LocalTensor<float> biasCastUb0;
    LocalTensor<float> biasCastUb1;
    if constexpr (ISBIASEXIST) {
        biasTmpUbDb0 = biasBufDb0_.Get<T>();
        biasTmpUbDb1 = biasBufDb1_.Get<T>();
        biasCastUb0 = biasCastBuf0_.Get<float>();
        biasCastUb1 = biasCastBuf1_.Get<float>();
        expertForSourceRowLocal = expertForSourceRowQueue_.DeQue<int32_t>();
    }
    /*******************************乒***********************************************/
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    int64_t biasIndexDb0 = 0;
    if constexpr (ISBIASEXIST) {
        biasIndexDb0 = expertForSourceRowLocal.GetValue(0);
    }
    float scalesValDb0 = 1.0;
    if (tilingData_.scalesIsNull == 0) {
        if constexpr (AscendC::IsSameType<T, TS>::value) {
            scalesValDb0 = ToFloat(scalesLocal.GetValue(0));
        } else {
            scalesValDb0 = scalesLocal.GetValue(0);
        }
    }
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    int64_t expandedSrcToDstRowIndexDb0 = 0;
    if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
        expandedSrcToDstRowIndexDb0 = nLoopIdx / (tilingData_.hSliceNum + 1) + 0 * tilingData_.totalRowNum +
                                      GetBlockIdx() * tilingData_.normalCoreHandleNum;
    } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
        expandedSrcToDstRowIndexDb0 = nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.K + 0 +
                                      GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.K;
    }
    int64_t expandedPermutedRowsIndexDb0 = gmExpandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb0);
    SetFlag<HardEvent::S_MTE2>(EVENT_ID2);

    /*******************************乓***********************************************/
    int64_t biasIndexDb1 = 0;
    if constexpr (ISBIASEXIST) {
        biasIndexDb1 = expertForSourceRowLocal.GetValue(1);
    }
    float scalesValDb1 = 1.0;
    if (tilingData_.scalesIsNull == 0) {
        if constexpr (AscendC::IsSameType<T, TS>::value) {
            scalesValDb1 = ToFloat(scalesLocal.GetValue(1));
        } else {
            scalesValDb1 = scalesLocal.GetValue(1);
        }
    }

    SetFlag<HardEvent::S_MTE2>(EVENT_ID1);
    int64_t expandedSrcToDstRowIndexDb1 = 0;
    if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
        expandedSrcToDstRowIndexDb1 = nLoopIdx / (tilingData_.hSliceNum + 1) + 1 * tilingData_.totalRowNum +
                                      GetBlockIdx() * tilingData_.normalCoreHandleNum;
    } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
        expandedSrcToDstRowIndexDb1 = nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.K + 1 +
                                      GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.K;
    }
    int64_t expandedPermutedRowsIndexDb1 = gmExpandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb1);
    SetFlag<HardEvent::S_MTE2>(EVENT_ID3);

    DataCopyParams copyParams{1, static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};

    /*******************************乒***********************************************/
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    if constexpr (ISBIASEXIST) {
        DataCopyPad(biasTmpUbDb0, gmBias_[biasIndexDb0 * tilingData_.H + bias], copyParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

    WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
    if (expandedPermutedRowsIndexDb0 != INVALID_ROW_INDEX) {
        DataCopyPad(
            expandedPermutedTmpUbDb0, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb0 * tilingData_.H + bias],
            copyParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);

    /*******************************乓***********************************************/
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
    if constexpr (ISBIASEXIST) {
        DataCopyPad(biasTmpUbDb1, gmBias_[biasIndexDb1 * tilingData_.H + bias], copyParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);

    WaitFlag<HardEvent::S_MTE2>(EVENT_ID3);
    if (expandedPermutedRowsIndexDb1 != INVALID_ROW_INDEX) {
        DataCopyPad(
            expandedPermutedTmpUbDb1, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb1 * tilingData_.H + bias],
            copyParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID3);

    /*******************************乒***********************************************/
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    if (expandedPermutedRowsIndexDb0 != INVALID_ROW_INDEX) {
        Cast(expandedPermutedRowsCastUb0, expandedPermutedTmpUbDb0, RoundMode::CAST_NONE, dataLen);
    }
    if constexpr (ISBIASEXIST) {
        Cast(biasCastUb0, biasTmpUbDb0, RoundMode::CAST_NONE, dataLen);
    }
    PipeBarrier<PIPE_V>();
    if (expandedPermutedRowsIndexDb0 != INVALID_ROW_INDEX) {
        if constexpr (ISBIASEXIST) {
            Add(expandedPermutedRowsCastUb0, expandedPermutedRowsCastUb0, biasCastUb0, dataLen);
            PipeBarrier<PIPE_V>();
        }
    }
    if (!initFlag_) {
        if (expandedPermutedRowsIndexDb0 == INVALID_ROW_INDEX &&
            (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
            Duplicate(skip1CastUb[0], (float)0.0, dataLen);
        } else {
            Muls(skip1CastUb[0], expandedPermutedRowsCastUb0, scalesValDb0, dataLen);
        }
        initFlag_ = true;
    } else {
        if (expandedPermutedRowsIndexDb0 == INVALID_ROW_INDEX &&
            (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
            PipeBarrier<PIPE_V>();
        } else {
            Muls(expandedPermutedRowsCastUb0, expandedPermutedRowsCastUb0, scalesValDb0, dataLen);
            PipeBarrier<PIPE_V>();
            Add(skip1CastUb[0], skip1CastUb[0], expandedPermutedRowsCastUb0, dataLen);
        }
    }

    /*******************************乓***********************************************/
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
    if (expandedPermutedRowsIndexDb1 != INVALID_ROW_INDEX) {
        Cast(expandedPermutedRowsCastUb1, expandedPermutedTmpUbDb1, RoundMode::CAST_NONE, dataLen);
    }
    if constexpr (ISBIASEXIST) {
        Cast(biasCastUb1, biasTmpUbDb1, RoundMode::CAST_NONE, dataLen);
    }
    PipeBarrier<PIPE_V>();
    if (expandedPermutedRowsIndexDb1 != INVALID_ROW_INDEX) {
        if constexpr (ISBIASEXIST) {
            Add(expandedPermutedRowsCastUb1, expandedPermutedRowsCastUb1, biasCastUb1, dataLen);
            PipeBarrier<PIPE_V>();
        }
    }
    if (expandedPermutedRowsIndexDb1 != INVALID_ROW_INDEX) {
        Muls(expandedPermutedRowsCastUb1, expandedPermutedRowsCastUb1, scalesValDb1, dataLen);
        PipeBarrier<PIPE_V>();
        Add(skip1CastUb[0], skip1CastUb[0], expandedPermutedRowsCastUb1, dataLen);
        PipeBarrier<PIPE_V>();
    }

    Cast(outLocal, skip1CastUb, RoundMode::CAST_RINT, dataLen);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);

    outQueue_.EnQue(outLocal);
    if constexpr (ISBIASEXIST) {
        expertForSourceRowQueue_.FreeTensor(expertForSourceRowLocal);
    }
    if (tilingData_.scalesIsNull == 0) {
        scalesQueue_.FreeTensor(scalesLocal);
    }
    if (tilingData_.skip1IsNull == 0) {
        skip1Queue_.FreeTensor(skip1Local);
    }
    if (tilingData_.skip2IsNull == 0) {
        skip2Queue_.FreeTensor(skip2Local);
    }
}

template <typename T, typename TS, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2Bf16CuthK2<T, TS, ISBIASEXIST>::CopyOut(
    int64_t nLoopIdx, int64_t bias, int64_t dataLen)
{
    LocalTensor<T> outLocal = outQueue_.DeQue<T>();
    DataCopyParams copyParams{1, static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(gmOut_[nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.H + bias], outLocal, copyParams);
    outQueue_.FreeTensor(outLocal);
}

template <typename T, typename TS, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2Bf16CuthK2<T, TS, ISBIASEXIST>::Process()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNum) {
        return;
    }
    PROCESS_IMP();
}

} // namespace MoeFinalizeRoutingV2
#endif // MOE_FINALIZE_ROUTING_BF16_CUTH_K_TWO
