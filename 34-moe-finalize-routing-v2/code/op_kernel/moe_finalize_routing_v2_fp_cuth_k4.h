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
 * \file moe_finalize_routing_v2_fp_cuth_k4.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_V2_FP_CUTH_K_FOUR
#define MOE_FINALIZE_ROUTING_V2_FP_CUTH_K_FOUR

#include "moe_finalize_routing_v2_common.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MoeFinalizeRoutingV2 {

using namespace AscendC;

template <typename T, const bool ISBIASEXIST>
class MoeFinalizeRoutingV2FpCuthK4
{
public:
    __aicore__ inline MoeFinalizeRoutingV2FpCuthK4(const MoeFinalizeRoutingV2TilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(
        GM_ADDR expandedPermutedRows, GM_ADDR expandedSrcToDstRow, GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias,
        GM_ADDR scales, GM_ADDR expertForSourceRow, GM_ADDR out, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CkechColAlignment();
    __aicore__ inline void CopyIn(int64_t nLoopIdx, int64_t bias, int64_t dataLen, bool isPadH, int64_t rightPaddingH);
    __aicore__ inline void Compute(int64_t nLoopIdx, int64_t bias, int64_t dataLen, bool isPadH, int64_t rightPaddingH);
    __aicore__ inline void CopyOut(int64_t nLoopIdx, int64_t bias, int64_t dataLen);
    __aicore__ inline int64_t AlignmentProcess(int64_t param);
    __aicore__ inline int64_t PadProcessT(int64_t param);

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
    TBuf<QuePosition::VECCALC> expandedPermutedRowsBufDb2_;
    TBuf<QuePosition::VECCALC> biasBufDb2_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsBufDb3_;
    TBuf<QuePosition::VECCALC> biasBufDb3_;

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

    const int64_t onceAlgnNum_{ONE_BLK_SIZE / static_cast<int64_t>(sizeof(T))};

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

template <typename T, const bool ISBIASEXIST>
__aicore__ inline int64_t MoeFinalizeRoutingV2FpCuthK4<T, ISBIASEXIST>::AlignmentProcess(int64_t param)
{
    return (param + onceAlgnNum_ - 1) / onceAlgnNum_ * onceAlgnNum_;
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline int64_t MoeFinalizeRoutingV2FpCuthK4<T, ISBIASEXIST>::PadProcessT(int64_t param)
{
    return onceAlgnNum_ - param % onceAlgnNum_;
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2FpCuthK4<T, ISBIASEXIST>::CkechColAlignment()
{
    if (tilingData_.normalH * sizeof(T) % ONE_BLK_SIZE) {
        isPadNormalH_ = true;
        rightPaddingNormalH_ = PadProcessT(tilingData_.normalH);
    }

    if (tilingData_.unnormalH * sizeof(T) % ONE_BLK_SIZE) {
        isPadUnnormalH_ = true;
        rightPaddingUnnormalH_ = PadProcessT(tilingData_.unnormalH);
    }

    if (tilingData_.K * sizeof(T) % ONE_BLK_SIZE) {
        isPadK_ = true;
        rightPaddingK_ = PadProcessT(tilingData_.K);
    }

    if (tilingData_.K * INT32_BYTES % ONE_BLK_SIZE) {
        isPadKInt32_ = true;
        rightPaddingKInt32_ = PadProcessInt32(tilingData_.K);
    }
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2FpCuthK4<T, ISBIASEXIST>::Init(
    GM_ADDR expandedPermutedRows, GM_ADDR expandedSrcToDstRow, GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias,
    GM_ADDR scales, GM_ADDR expertForSourceRow, GM_ADDR out, GM_ADDR workspace)
{
    if (GetBlockIdx() + 1 == tilingData_.usedCoreNum) {
        curCoreHandleNumPerLoop_ = tilingData_.tailCoreHandleNumPerLoop;
    } else {
        curCoreHandleNumPerLoop_ = tilingData_.normalCoreHandleNumPerLoop;
    }

    // 检查要处理的列数是否对齐以及应该如何对齐
    CkechColAlignment();

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
    }

    inputScalesAndExpertIdx_ = GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.K;
    if (tilingData_.scalesIsNull == 0) {
        gmScales_.SetGlobalBuffer(
            (__gm__ T*)scales + inputScalesAndExpertIdx_, tilingData_.normalCoreHandleNum * tilingData_.K);
        pipe_.InitBuffer(
            scalesQueue_, BUFFER_NUM,
            tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.K) * sizeof(T));
    }

    gmExpandedSrcToDstRow_.SetGlobalBuffer(
        (__gm__ int32_t*)expandedSrcToDstRow, tilingData_.totalRowNum * tilingData_.K);

    outputIdx_ = GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.H;
    gmOut_.SetGlobalBuffer((__gm__ T*)out + outputIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);

    gmExpandedPermutedRows_.SetGlobalBuffer(
        (__gm__ T*)expandedPermutedRows, tilingData_.totalRowNum * tilingData_.K * tilingData_.H);
    if constexpr (ISBIASEXIST) {
        gmBias_.SetGlobalBuffer((__gm__ T*)bias, tilingData_.biasRowNum * tilingData_.H);
        pipe_.InitBuffer(biasBufDb0_, AlignmentProcess(tilingData_.normalH) * sizeof(T));
        pipe_.InitBuffer(biasBufDb1_, AlignmentProcess(tilingData_.normalH) * sizeof(T));
        pipe_.InitBuffer(biasBufDb2_, AlignmentProcess(tilingData_.normalH) * sizeof(T));
        pipe_.InitBuffer(biasBufDb3_, AlignmentProcess(tilingData_.normalH) * sizeof(T));
        gmExpertForSourceRow_.SetGlobalBuffer(
            (__gm__ int32_t*)expertForSourceRow + inputScalesAndExpertIdx_,
            tilingData_.normalCoreHandleNum * tilingData_.K);
        pipe_.InitBuffer(
            expertForSourceRowQueue_, BUFFER_NUM,
            tilingData_.normalCoreHandleNumPerLoop * Int32AlignmentProcess(tilingData_.K) * sizeof(int32_t));
    }

    // 申请 buffer 空间
    pipe_.InitBuffer(
        outQueue_, BUFFER_NUM,
        tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(T));

    pipe_.InitBuffer(expandedPermutedRowsBufDb0_, AlignmentProcess(tilingData_.normalH) * sizeof(T));

    pipe_.InitBuffer(expandedPermutedRowsBufDb1_, AlignmentProcess(tilingData_.normalH) * sizeof(T));

    pipe_.InitBuffer(expandedPermutedRowsBufDb2_, AlignmentProcess(tilingData_.normalH) * sizeof(T));

    pipe_.InitBuffer(expandedPermutedRowsBufDb3_, AlignmentProcess(tilingData_.normalH) * sizeof(T));
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2FpCuthK4<T, ISBIASEXIST>::CopyIn(
    int64_t nLoopIdx, int64_t bias, int64_t dataLen, bool isPadH, int64_t rightPaddingH)
{
    LocalTensor<T> scalesLocal;
    if (tilingData_.scalesIsNull == 0) {
        scalesLocal = scalesQueue_.AllocTensor<T>();
    }
    LocalTensor<int32_t> expertForSourceRowLocal;
    if constexpr (ISBIASEXIST) {
        expertForSourceRowLocal = expertForSourceRowQueue_.AllocTensor<int32_t>();
    }
    LocalTensor<T> skip1Local;
    LocalTensor<T> skip2Local;
    if (tilingData_.skip1IsNull == 0) {
        skip1Local = skip1Queue_.AllocTensor<T>();
    }
    if (tilingData_.skip2IsNull == 0) {
        skip2Local = skip2Queue_.AllocTensor<T>();
    }

    // ---------------------------- [Skip] -------------------------------
    DataCopyParams copyParamsSkip{1, static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
    DataCopyPadParams padParamsSkip{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};
    if (tilingData_.skip1IsNull == 0) {
        DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (
            skip1Local, gmSkip1_[nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.H + bias], copyParamsSkip,
            padParamsSkip);
    }
    if (tilingData_.skip2IsNull == 0) {
        DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (
            skip2Local, gmSkip2_[nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.H + bias], copyParamsSkip,
            padParamsSkip);
    }

    // ---------------------------- [Scales] -------------------------------
    if (tilingData_.scalesIsNull == 0) {
        DataCopyParams copyParamsScales{1, static_cast<uint16_t>(tilingData_.K * sizeof(T)), 0, 0};
        DataCopyPadParams padParamsScales{isPadK_, 0, static_cast<uint8_t>(rightPaddingK_), 0};
        DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (
            scalesLocal, gmScales_[nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.K], copyParamsScales,
            padParamsScales);
    }

    // ---------------------------- [Expert] -------------------------------
    if constexpr (ISBIASEXIST) {
        DataCopyParams copyParamsExpert{1, static_cast<uint16_t>(tilingData_.K * sizeof(int32_t)), 0, 0};
        DataCopyPadParams padParamsExpert{isPadKInt32_, 0, static_cast<uint8_t>(rightPaddingKInt32_), 0};
        DataCopyPadCustom<int32_t, DataCopyParams, DataCopyPadParams>(
            expertForSourceRowLocal, gmExpertForSourceRow_[nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.K],
            copyParamsExpert, padParamsExpert);
    }
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);

    COPY_IN_ENQUE();
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2FpCuthK4<T, ISBIASEXIST>::Compute(
    int64_t nLoopIdx, int64_t bias, int64_t dataLen, bool isPadH, int64_t rightPaddingH)
{
    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();

    LocalTensor<T> scalesLocal;
    if (tilingData_.scalesIsNull == 0) {
        scalesLocal = scalesQueue_.DeQue<T>();
    }
    LocalTensor<T> skip2Local;
    LocalTensor<T> skip1Local;

    if (tilingData_.skip1IsNull == 0) {
        skip1Local = skip1Queue_.DeQue<T>();
    } else {
        skip1Local = skip1Queue_.AllocTensor<T>();
    }
    if (tilingData_.skip2IsNull == 0) {
        skip2Local = skip2Queue_.DeQue<T>();
    }
    if ((tilingData_.skip1IsNull == 0) && (tilingData_.skip2IsNull == 0)) {
        Add(outLocal, skip1Local, skip2Local, AlignmentProcess(dataLen));
    } else if ((tilingData_.skip1IsNull == 0) && (tilingData_.skip2IsNull == 1)) {
        Adds(outLocal, skip1Local, (T)0, AlignmentProcess(dataLen));
    } else if ((tilingData_.skip1IsNull == 1) && (tilingData_.skip2IsNull == 0)) {
        Adds(outLocal, skip2Local, (T)0, AlignmentProcess(dataLen));
    } else {
        initFlag_ = false;
    }

    LocalTensor<T> expandedPermutedTmpUbDb0 = expandedPermutedRowsBufDb0_.Get<T>();
    LocalTensor<T> expandedPermutedTmpUbDb1 = expandedPermutedRowsBufDb1_.Get<T>();
    LocalTensor<T> expandedPermutedTmpUbDb2 = expandedPermutedRowsBufDb2_.Get<T>();
    LocalTensor<T> expandedPermutedTmpUbDb3 = expandedPermutedRowsBufDb3_.Get<T>();
    LocalTensor<int32_t> expertForSourceRowLocal;

    LocalTensor<T> biasTmpUbDb0;
    LocalTensor<T> biasTmpUbDb1;
    LocalTensor<T> biasTmpUbDb2;
    LocalTensor<T> biasTmpUbDb3;
    if constexpr (ISBIASEXIST) {
        biasTmpUbDb0 = biasBufDb0_.Get<T>();
        biasTmpUbDb1 = biasBufDb1_.Get<T>();
        biasTmpUbDb2 = biasBufDb2_.Get<T>();
        biasTmpUbDb3 = biasBufDb3_.Get<T>();
        expertForSourceRowLocal = expertForSourceRowQueue_.DeQue<int32_t>();
    }

    /*******************************乒***********************************************/
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

    int64_t biasIndexDb0 = 0;
    int64_t biasIndexDb1 = 0;
    if constexpr (ISBIASEXIST) {
        biasIndexDb0 = expertForSourceRowLocal.GetValue(0);
        biasIndexDb1 = expertForSourceRowLocal.GetValue(1);
    }
    T scalesValDb0 = 1.0;
    T scalesValDb1 = 1.0;
    if (tilingData_.scalesIsNull == 0) {
        scalesValDb0 = scalesLocal.GetValue(0);
        scalesValDb1 = scalesLocal.GetValue(1);
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
    int64_t expandedSrcToDstRowIndexDb1 = 0;
    if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
        expandedSrcToDstRowIndexDb1 = nLoopIdx / (tilingData_.hSliceNum + 1) + 1 * tilingData_.totalRowNum +
                                      GetBlockIdx() * tilingData_.normalCoreHandleNum;
    } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
        expandedSrcToDstRowIndexDb1 = nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.K + 1 +
                                      GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.K;
    }
    int64_t expandedPermutedRowsIndexDb0 = gmExpandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb0);
    int64_t expandedPermutedRowsIndexDb1 = gmExpandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb1);
    SetFlag<HardEvent::S_MTE2>(EVENT_ID2);

    /*******************************乓***********************************************/
    int64_t biasIndexDb2 = 0;
    int64_t biasIndexDb3 = 0;
    if constexpr (ISBIASEXIST) {
        biasIndexDb2 = expertForSourceRowLocal.GetValue(2);
        biasIndexDb3 = expertForSourceRowLocal.GetValue(3);
    }
    T scalesValDb2 = 1.0;
    T scalesValDb3 = 1.0;
    if (tilingData_.scalesIsNull == 0) {
        scalesValDb2 = scalesLocal.GetValue(2);
        scalesValDb3 = scalesLocal.GetValue(3);
    }
    SetFlag<HardEvent::S_MTE2>(EVENT_ID1);

    int64_t expandedSrcToDstRowIndexDb2 = 0;
    if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
        expandedSrcToDstRowIndexDb2 = nLoopIdx / (tilingData_.hSliceNum + 1) + 2 * tilingData_.totalRowNum +
                                      GetBlockIdx() * tilingData_.normalCoreHandleNum;
    } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
        expandedSrcToDstRowIndexDb2 = nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.K + 2 +
                                      GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.K;
    }
    int64_t expandedSrcToDstRowIndexDb3 = 0;
    if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
        expandedSrcToDstRowIndexDb3 = nLoopIdx / (tilingData_.hSliceNum + 1) + 3 * tilingData_.totalRowNum +
                                      GetBlockIdx() * tilingData_.normalCoreHandleNum;
    } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
        expandedSrcToDstRowIndexDb3 = nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.K + 3 +
                                      GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.K;
    }
    int64_t expandedPermutedRowsIndexDb2 = gmExpandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb2);
    int64_t expandedPermutedRowsIndexDb3 = gmExpandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb3);
    SetFlag<HardEvent::S_MTE2>(EVENT_ID3);

    DataCopyParams copyParams{1, static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};

    /*******************************乒***********************************************/
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    if constexpr (ISBIASEXIST) {
        DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (biasTmpUbDb0, gmBias_[biasIndexDb0 * tilingData_.H + bias], copyParams, padParams);
        DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (biasTmpUbDb1, gmBias_[biasIndexDb1 * tilingData_.H + bias], copyParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

    WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
    if (expandedPermutedRowsIndexDb0 != INVALID_ROW_INDEX) {
        DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (
            expandedPermutedTmpUbDb0, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb0 * tilingData_.H + bias],
            copyParams, padParams);
    }
    if (expandedPermutedRowsIndexDb1 != INVALID_ROW_INDEX) {
        DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (
            expandedPermutedTmpUbDb1, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb1 * tilingData_.H + bias],
            copyParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);

    /*******************************乓***********************************************/
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
    if constexpr (ISBIASEXIST) {
        DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (biasTmpUbDb2, gmBias_[biasIndexDb2 * tilingData_.H + bias], copyParams, padParams);
        DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (biasTmpUbDb3, gmBias_[biasIndexDb3 * tilingData_.H + bias], copyParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);

    WaitFlag<HardEvent::S_MTE2>(EVENT_ID3);
    if (expandedPermutedRowsIndexDb2 != INVALID_ROW_INDEX) {
        DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (
            expandedPermutedTmpUbDb2, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb2 * tilingData_.H + bias],
            copyParams, padParams);
    }
    if (expandedPermutedRowsIndexDb3 != INVALID_ROW_INDEX) {
        DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (
            expandedPermutedTmpUbDb3, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb3 * tilingData_.H + bias],
            copyParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID3);

    /*******************************乒***********************************************/
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    if (expandedPermutedRowsIndexDb0 == INVALID_ROW_INDEX &&
        (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
        Duplicate(expandedPermutedTmpUbDb0, (T)0, dataLen);
        PipeBarrier<PIPE_V>();
    } else {
        if constexpr (ISBIASEXIST) {
            Add(expandedPermutedTmpUbDb0, expandedPermutedTmpUbDb0, biasTmpUbDb0, dataLen);
        }
    }
    if (expandedPermutedRowsIndexDb1 == INVALID_ROW_INDEX &&
        (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
        Duplicate(expandedPermutedTmpUbDb1, (T)0, dataLen);
        PipeBarrier<PIPE_V>();
    } else {
        if constexpr (ISBIASEXIST) {
            Add(expandedPermutedTmpUbDb1, expandedPermutedTmpUbDb1, biasTmpUbDb1, dataLen);
        }
    }
    PipeBarrier<PIPE_V>();
    if (!initFlag_) {
        if (expandedPermutedRowsIndexDb0 == INVALID_ROW_INDEX &&
            (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
            Duplicate(outLocal[0], (T)0, dataLen);
            Duplicate(expandedPermutedTmpUbDb1, (T)0, dataLen);
        } else {
            Muls(outLocal[0], expandedPermutedTmpUbDb0, scalesValDb0, dataLen);
            Muls(expandedPermutedTmpUbDb1, expandedPermutedTmpUbDb1, scalesValDb1, dataLen);
        }
        initFlag_ = true;
    } else {
        if (expandedPermutedRowsIndexDb0 == INVALID_ROW_INDEX &&
            (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
            PipeBarrier<PIPE_V>();
        } else {
            Muls(expandedPermutedTmpUbDb0, expandedPermutedTmpUbDb0, scalesValDb0, dataLen);
            Muls(expandedPermutedTmpUbDb1, expandedPermutedTmpUbDb1, scalesValDb1, dataLen);
            PipeBarrier<PIPE_V>();
            Add(outLocal[0], outLocal[0], expandedPermutedTmpUbDb0, dataLen);
        }
    }
    PipeBarrier<PIPE_V>();
    Add(outLocal[0], outLocal[0], expandedPermutedTmpUbDb1, dataLen);

    /*******************************乓***********************************************/
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
    if (expandedPermutedRowsIndexDb2 == INVALID_ROW_INDEX &&
        (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
        Duplicate(expandedPermutedTmpUbDb2, (T)0, dataLen);
        PipeBarrier<PIPE_V>();
    } else {
        if constexpr (ISBIASEXIST) {
            Add(expandedPermutedTmpUbDb2, expandedPermutedTmpUbDb2, biasTmpUbDb2, dataLen);
        }
    }
    if (expandedPermutedRowsIndexDb3 == INVALID_ROW_INDEX &&
        (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
        Duplicate(expandedPermutedTmpUbDb3, (T)0, dataLen);
        PipeBarrier<PIPE_V>();
    } else {
        if constexpr (ISBIASEXIST) {
            Add(expandedPermutedTmpUbDb3, expandedPermutedTmpUbDb3, biasTmpUbDb3, dataLen);
        }
    }
    PipeBarrier<PIPE_V>();
    Muls(expandedPermutedTmpUbDb2, expandedPermutedTmpUbDb2, scalesValDb2, dataLen);
    Muls(expandedPermutedTmpUbDb3, expandedPermutedTmpUbDb3, scalesValDb3, dataLen);
    PipeBarrier<PIPE_V>();
    Add(outLocal[0], outLocal[0], expandedPermutedTmpUbDb2, dataLen);
    PipeBarrier<PIPE_V>();
    Add(outLocal[0], outLocal[0], expandedPermutedTmpUbDb3, dataLen);
#ifndef __CCE_KT_TEST__
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
#endif
    outQueue_.EnQue(outLocal);
    if constexpr (ISBIASEXIST) {
        expertForSourceRowQueue_.FreeTensor(expertForSourceRowLocal);
    }
    if (tilingData_.scalesIsNull == 0) {
        scalesQueue_.FreeTensor(scalesLocal);
    }
    skip1Queue_.FreeTensor(skip1Local);
    if (tilingData_.skip2IsNull == 0) {
        skip2Queue_.FreeTensor(skip2Local);
    }
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2FpCuthK4<T, ISBIASEXIST>::CopyOut(
    int64_t nLoopIdx, int64_t bias, int64_t dataLen)
{
    LocalTensor<T> outLocal = outQueue_.DeQue<T>();
    DataCopyParams copyParams{1, static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
#ifndef __CCE_KT_TEST__
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
#endif
    DataCopyPadCustom<T>(gmOut_[nLoopIdx / (tilingData_.hSliceNum + 1) * tilingData_.H + bias], outLocal, copyParams);
    outQueue_.FreeTensor(outLocal);
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2FpCuthK4<T, ISBIASEXIST>::Process()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNum) {
        return;
    }
    PROCESS_IMP();
}

} // namespace MoeFinalizeRoutingV2
#endif // MOE_FINALIZE_ROUTING_V2_FP_CUTH_K_FOUR
