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
 * \file moe_finalize_routing_v2_fp_db_all_bias.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_V2_FP_DB_ALL_BIAS
#define MOE_FINALIZE_ROUTING_V2_FP_DB_ALL_BIAS

#include "moe_finalize_routing_v2_common.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MoeFinalizeRoutingV2 {

using namespace AscendC;

template <typename T, const bool ISBIASEXIST>
class MoeFinalizeRoutingV2FpDbAllBias
{
public:
    __aicore__ inline MoeFinalizeRoutingV2FpDbAllBias(const MoeFinalizeRoutingV2TilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(
        GM_ADDR expandedPermutedRows, GM_ADDR expandedSrcToDstRow, GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias,
        GM_ADDR scales, GM_ADDR expertForSourceRow, GM_ADDR out, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void Compute(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void CopyOut(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline int64_t AlignmentProcess(int64_t param);
    __aicore__ inline int64_t PadProcessT(int64_t param);
    __aicore__ inline void PrepareData();
    __aicore__ inline void CalcPaddingPara();

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
    TBuf<QuePosition::VECCALC> tmpBuf_;

    TBuf<QuePosition::VECCALC> expandedSrcToDstRowBuff_;

    GlobalTensor<T> gmExpandedPermutedRows_;
    GlobalTensor<T> gmSkip1_;
    GlobalTensor<T> gmSkip2_;
    GlobalTensor<T> gmBias_;
    GlobalTensor<T> gmScales_;
    GlobalTensor<int32_t> gmExpandedSrcToDstRow_;
    GlobalTensor<int32_t> gmExpertForSourceRow_;
    GlobalTensor<T> gmOut_;

    LocalTensor<int32_t> expandedSrcToDstRow_;

    int64_t workspace_{0};

    // input number
    int64_t inputSkipIdx_{0};
    int64_t inputScalesAndExpertIdx_{0};
    int64_t outputIdx_{0};
    int64_t curCoreHandleNumPerLoop_{0};

    const int64_t onceAlgnNum_{ONE_BLK_SIZE / static_cast<int64_t>(sizeof(T))};

    bool isPadH_ = false;
    bool isPadK_ = false;
    bool isPadKInt32_ = false;
    bool initFlag_ = true;
    bool skipsAllNone_ = false;
    uint16_t rightPaddingH_ = 0;
    uint16_t rightPaddingK_ = 0;
    uint16_t rightPaddingKInt32_ = 0;

    bool isPadSourceToDstRow_ = false;
    uint16_t rightPaddingSourceToDstRow_ = 0;
    int64_t curCoreHandleNum_ = 0;
};

template <typename T, const bool ISBIASEXIST>
__aicore__ inline int64_t MoeFinalizeRoutingV2FpDbAllBias<T, ISBIASEXIST>::AlignmentProcess(int64_t param)
{
    return (param + onceAlgnNum_ - 1) / onceAlgnNum_ * onceAlgnNum_;
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline int64_t MoeFinalizeRoutingV2FpDbAllBias<T, ISBIASEXIST>::PadProcessT(int64_t param)
{
    return onceAlgnNum_ - param % onceAlgnNum_;
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2FpDbAllBias<T, ISBIASEXIST>::CalcPaddingPara()
{
    if (tilingData_.H * sizeof(T) % ONE_BLK_SIZE) {
        isPadH_ = true;
        rightPaddingH_ = PadProcessT(tilingData_.H);
    }

    if (tilingData_.K * sizeof(T) % ONE_BLK_SIZE) {
        isPadK_ = true;
        rightPaddingK_ = PadProcessT(tilingData_.K);
    }

    if (tilingData_.K * INT32_BYTES % ONE_BLK_SIZE) {
        isPadKInt32_ = true;
        rightPaddingKInt32_ = PadProcessInt32(tilingData_.K);
    }

    if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
        if (curCoreHandleNum_ * INT32_BYTES % ONE_BLK_SIZE) {
            isPadSourceToDstRow_ = true;
            rightPaddingSourceToDstRow_ = PadProcessInt32(curCoreHandleNum_);
        }
    } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
        if (curCoreHandleNum_ * tilingData_.K * INT32_BYTES % ONE_BLK_SIZE) {
            isPadSourceToDstRow_ = true;
            rightPaddingSourceToDstRow_ = PadProcessInt32(curCoreHandleNum_ * tilingData_.K);
        }
    }
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2FpDbAllBias<T, ISBIASEXIST>::Init(
    GM_ADDR expandedPermutedRows, GM_ADDR expandedSrcToDstRow, GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias,
    GM_ADDR scales, GM_ADDR expertForSourceRow, GM_ADDR out, GM_ADDR workspace)
{
    if (GetBlockIdx() + 1 == tilingData_.usedCoreNum) {
        curCoreHandleNumPerLoop_ = tilingData_.tailCoreHandleNumPerLoop;
        curCoreHandleNum_ = tilingData_.tailCoreHandleNum;
    } else {
        curCoreHandleNumPerLoop_ = tilingData_.normalCoreHandleNumPerLoop;
        curCoreHandleNum_ = tilingData_.normalCoreHandleNum;
    }

    CalcPaddingPara();

    // gmInput分核 && 输入偏移量初始化
    inputSkipIdx_ = GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.H;
    if (tilingData_.skip1IsNull == 0) {
        gmSkip1_.SetGlobalBuffer((__gm__ T*)skip1 + inputSkipIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);
    }
    pipe_.InitBuffer(
        skip1Queue_, 1, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.H) * sizeof(T));
    if (tilingData_.skip2IsNull == 0) {
        gmSkip2_.SetGlobalBuffer((__gm__ T*)skip2 + inputSkipIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);
        pipe_.InitBuffer(
            skip2Queue_, 1, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.H) * sizeof(T));
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
        pipe_.InitBuffer(biasBufDb0_, AlignmentProcess(tilingData_.H) * sizeof(T));
        pipe_.InitBuffer(biasBufDb1_, AlignmentProcess(tilingData_.H) * sizeof(T));
        gmExpertForSourceRow_.SetGlobalBuffer(
            (__gm__ int32_t*)expertForSourceRow + inputScalesAndExpertIdx_,
            tilingData_.normalCoreHandleNum * tilingData_.K);
        pipe_.InitBuffer(
            expertForSourceRowQueue_, BUFFER_NUM,
            tilingData_.normalCoreHandleNumPerLoop * Int32AlignmentProcess(tilingData_.K) * sizeof(int32_t));
    }

    // 申请 buffer 空间
    pipe_.InitBuffer(
        outQueue_, BUFFER_NUM, tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.H) * sizeof(T));

    pipe_.InitBuffer(expandedPermutedRowsBufDb0_, AlignmentProcess(tilingData_.H) * sizeof(T));

    pipe_.InitBuffer(expandedPermutedRowsBufDb1_, AlignmentProcess(tilingData_.H) * sizeof(T));
    pipe_.InitBuffer(tmpBuf_, AlignmentProcess(tilingData_.H) * sizeof(T));

    pipe_.InitBuffer(
        expandedSrcToDstRowBuff_, Int32AlignmentProcess(curCoreHandleNum_) * tilingData_.K * sizeof(int32_t));
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2FpDbAllBias<T, ISBIASEXIST>::PrepareData()
{
    expandedSrcToDstRow_ = expandedSrcToDstRowBuff_.Get<int32_t>();
    DataCopyPadExtParams<int32_t> padParamsSrcToDstRow{
        isPadSourceToDstRow_, 0, static_cast<uint8_t>(rightPaddingSourceToDstRow_), 0};
    if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
        DataCopyExtParams copyParamsSrcToDstRow{
            static_cast<uint16_t>(tilingData_.K), static_cast<uint32_t>(curCoreHandleNum_ * sizeof(int32_t)),
            static_cast<uint32_t>((tilingData_.totalRowNum - curCoreHandleNum_) * sizeof(int32_t)), 0, 0};
        DataCopyPadCustom<int32_t, DataCopyExtParams, DataCopyPadExtParams<int32_t>>(
            expandedSrcToDstRow_, gmExpandedSrcToDstRow_[GetBlockIdx() * tilingData_.normalCoreHandleNum],
            copyParamsSrcToDstRow, padParamsSrcToDstRow);
    } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
        DataCopyExtParams copyParamsSrcToDstRow{
            static_cast<uint16_t>(1), static_cast<uint32_t>(curCoreHandleNum_ * tilingData_.K * sizeof(int32_t)), 0, 0,
            0};

        DataCopyPadCustom<int32_t, DataCopyExtParams, DataCopyPadExtParams<int32_t>>(
            expandedSrcToDstRow_,
            gmExpandedSrcToDstRow_[GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.K],
            copyParamsSrcToDstRow, padParamsSrcToDstRow);
    }
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2FpDbAllBias<T, ISBIASEXIST>::CopyIn(int64_t nLoopIdx, int64_t curRepeatTimes)
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
    DataCopyParams copyParamsSkip{
        static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(tilingData_.H * sizeof(T)), 0, 0};
    DataCopyPadParams padParamsSkip{isPadH_, 0, static_cast<uint8_t>(rightPaddingH_), 0};

    if (tilingData_.skip1IsNull == 0) {
        DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (
            skip1Local, gmSkip1_[nLoopIdx * curCoreHandleNumPerLoop_ * tilingData_.H], copyParamsSkip, padParamsSkip);
    }

    if (tilingData_.skip2IsNull == 0) {
        DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (
            skip2Local, gmSkip2_[nLoopIdx * curCoreHandleNumPerLoop_ * tilingData_.H], copyParamsSkip, padParamsSkip);
    }

    // ---------------------------- [Scales] -------------------------------
    if (tilingData_.scalesIsNull == 0) {
        DataCopyParams copyParamsScales{
            static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(tilingData_.K * sizeof(T)), 0, 0};
        DataCopyPadParams padParamsScales{isPadK_, 0, static_cast<uint8_t>(rightPaddingK_), 0};
        DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (
            scalesLocal, gmScales_[nLoopIdx * curCoreHandleNumPerLoop_ * tilingData_.K], copyParamsScales,
            padParamsScales);
    }
    // ---------------------------- [Expert] -------------------------------
    if constexpr (ISBIASEXIST) {
        DataCopyParams copyParamsExpert{
            static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(tilingData_.K * sizeof(int32_t)), 0, 0};
        DataCopyPadParams padParamsExpert{isPadKInt32_, 0, static_cast<uint8_t>(rightPaddingKInt32_), 0};
        DataCopyPadCustom<int32_t, DataCopyParams, DataCopyPadParams>(
            expertForSourceRowLocal, gmExpertForSourceRow_[nLoopIdx * curCoreHandleNumPerLoop_ * tilingData_.K],
            copyParamsExpert, padParamsExpert);
    }
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);

    COPY_IN_ENQUE();
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2FpDbAllBias<T, ISBIASEXIST>::Compute(
    int64_t nLoopIdx, int64_t curRepeatTimes)
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
        Add(outLocal, skip1Local, skip2Local, curRepeatTimes * AlignmentProcess(tilingData_.H));
    } else if ((tilingData_.skip1IsNull == 0) && (tilingData_.skip2IsNull == 1)) {
        Adds(outLocal, skip1Local, (T)0, curRepeatTimes * AlignmentProcess(tilingData_.H));
    } else if ((tilingData_.skip1IsNull == 1) && (tilingData_.skip2IsNull == 0)) {
        Adds(outLocal, skip2Local, (T)0, curRepeatTimes * AlignmentProcess(tilingData_.H));
    } else {
        initFlag_ = false;
        skipsAllNone_ = true;
    }

    LocalTensor<T> expandedPermutedTmpUbDb0 = expandedPermutedRowsBufDb0_.Get<T>();

    LocalTensor<T> expandedPermutedTmpUbDb1 = expandedPermutedRowsBufDb1_.Get<T>();
    LocalTensor<int32_t> expertForSourceRowLocal;
    LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
    LocalTensor<T> biasTmpUbDb0;
    LocalTensor<T> biasTmpUbDb1;
    if constexpr (ISBIASEXIST) {
        biasTmpUbDb0 = biasBufDb0_.Get<T>();
        biasTmpUbDb1 = biasBufDb1_.Get<T>();
        expertForSourceRowLocal = expertForSourceRowQueue_.DeQue<int32_t>();
    }

    DataCopyParams copyParams{1, static_cast<uint16_t>(tilingData_.H * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{isPadH_, 0, static_cast<uint8_t>(rightPaddingH_), 0};

    int64_t biasInLoop = nLoopIdx * curCoreHandleNumPerLoop_ * tilingData_.K;

    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

    SetFlag<HardEvent::V_S>(EVENT_ID0);
    SetFlag<HardEvent::V_S>(EVENT_ID1);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    SetFlag<HardEvent::V_S>(EVENT_ID3);
    for (int64_t i = 0; i < curRepeatTimes; i++) {
        int64_t outRowIndex = i * AlignmentProcess(tilingData_.H);
        if (skipsAllNone_) {
            initFlag_ = false;
        }
        for (int64_t j = 0; j < tilingData_.K / PARALLEL_NUM; j++) {
            /*******************************乒***********************************************/
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            int64_t expandedSrcToDstRowIndexDb0 = 0;
            if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
                expandedSrcToDstRowIndexDb0 = nLoopIdx * curCoreHandleNumPerLoop_ + i +
                                              PARALLEL_NUM * j * Int32AlignmentProcess(curCoreHandleNum_);
            } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
                expandedSrcToDstRowIndexDb0 = biasInLoop + i * tilingData_.K + PARALLEL_NUM * j;
            }
            int64_t expandedPermutedRowsIndexDb0 = expandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb0);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);

            WaitFlag<HardEvent::V_S>(EVENT_ID2);
            int64_t biasIndexDb0 = 0;
            if constexpr (ISBIASEXIST) {
                biasIndexDb0 =
                    expertForSourceRowLocal.GetValue(i * Int32AlignmentProcess(tilingData_.K) + PARALLEL_NUM * j);
            }
            T scalesValDb0 = 1.0;
            if (tilingData_.scalesIsNull == 0) {
                scalesValDb0 = scalesLocal.GetValue(i * AlignmentProcess(tilingData_.K) + PARALLEL_NUM * j);
            }
            SetFlag<HardEvent::S_MTE2>(EVENT_ID2);

            /*******************************乒***********************************************/
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
            if (expandedPermutedRowsIndexDb0 != INVALID_ROW_INDEX) {
                DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (
                    expandedPermutedTmpUbDb0, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb0 * tilingData_.H],
                    copyParams, padParams);
            }
            if constexpr (ISBIASEXIST) {
                DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (biasTmpUbDb0, gmBias_[biasIndexDb0 * tilingData_.H], copyParams, padParams);
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

            /*******************************乓***********************************************/
            WaitFlag<HardEvent::V_S>(EVENT_ID1);
            int64_t expandedSrcToDstRowIndexDb1 = 0;
            if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
                expandedSrcToDstRowIndexDb1 = nLoopIdx * curCoreHandleNumPerLoop_ + i +
                                              (PARALLEL_NUM * j + 1) * Int32AlignmentProcess(curCoreHandleNum_);
            } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
                expandedSrcToDstRowIndexDb1 = biasInLoop + i * tilingData_.K + PARALLEL_NUM * j + 1;
            }
            int64_t expandedPermutedRowsIndexDb1 = expandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb1);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID1);

            WaitFlag<HardEvent::V_S>(EVENT_ID3);

            int64_t biasIndexDb1 = 0;
            if constexpr (ISBIASEXIST) {
                biasIndexDb1 =
                    expertForSourceRowLocal.GetValue(i * Int32AlignmentProcess(tilingData_.K) + PARALLEL_NUM * j + 1);
            }
            T scalesValDb1 = 1.0;
            if (tilingData_.scalesIsNull == 0) {
                scalesValDb1 = scalesLocal.GetValue(i * AlignmentProcess(tilingData_.K) + PARALLEL_NUM * j + 1);
            }
            SetFlag<HardEvent::S_MTE2>(EVENT_ID3);

            /*******************************乓***********************************************/
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID3);
            if (expandedPermutedRowsIndexDb1 != INVALID_ROW_INDEX) {
                DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (
                    expandedPermutedTmpUbDb1, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb1 * tilingData_.H],
                    copyParams, padParams);
            }
            if constexpr (ISBIASEXIST) {
                DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (biasTmpUbDb1, gmBias_[biasIndexDb1 * tilingData_.H], copyParams, padParams);
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);

            /*******************************乒***********************************************/
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if (expandedPermutedRowsIndexDb0 == INVALID_ROW_INDEX &&
                (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
                Duplicate(expandedPermutedTmpUbDb0, (T)0, tilingData_.H);
                PipeBarrier<PIPE_V>();
            } else {
                if constexpr (ISBIASEXIST) {
                    Add(skip1Local, expandedPermutedTmpUbDb0, biasTmpUbDb0, tilingData_.H);
                } else {
                    Adds(skip1Local, expandedPermutedTmpUbDb0, (T)0, tilingData_.H);
                }
            }
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            PipeBarrier<PIPE_V>();
            if (!initFlag_) {
                if (expandedPermutedRowsIndexDb0 == INVALID_ROW_INDEX &&
                    (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
                    Duplicate(outLocal[outRowIndex], (T)0, tilingData_.H);
                } else {
                    Muls(outLocal[outRowIndex], skip1Local, scalesValDb0, tilingData_.H);
                }
                initFlag_ = true;
                SetFlag<HardEvent::V_S>(EVENT_ID2);
            } else {
                if (expandedPermutedRowsIndexDb0 == INVALID_ROW_INDEX &&
                    (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
                    SetFlag<HardEvent::V_S>(EVENT_ID2);
                    PipeBarrier<PIPE_V>();
                } else {
                    Muls(skip1Local, skip1Local, scalesValDb0, tilingData_.H);
                    SetFlag<HardEvent::V_S>(EVENT_ID2);
                    PipeBarrier<PIPE_V>();
                    Add(outLocal[outRowIndex], outLocal[outRowIndex], skip1Local, tilingData_.H);
                }
            }
            /*******************************乓***********************************************/
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            if (expandedPermutedRowsIndexDb1 == INVALID_ROW_INDEX &&
                (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
                Duplicate(expandedPermutedTmpUbDb1, (T)0, tilingData_.H);
                Duplicate(tmpLocal, (T)0, tilingData_.H);
                PipeBarrier<PIPE_V>();
            } else {
                if constexpr (ISBIASEXIST) {
                    Add(tmpLocal, expandedPermutedTmpUbDb1, biasTmpUbDb1, tilingData_.H);
                } else {
                    Adds(tmpLocal, expandedPermutedTmpUbDb1, (T)0, tilingData_.H);
                }
            }
            SetFlag<HardEvent::V_S>(EVENT_ID1);
            PipeBarrier<PIPE_V>();
            Muls(tmpLocal, tmpLocal, scalesValDb1, tilingData_.H);
            SetFlag<HardEvent::V_S>(EVENT_ID3);
            PipeBarrier<PIPE_V>();
            Add(outLocal[outRowIndex], outLocal[outRowIndex], tmpLocal, tilingData_.H);
        }
        if (tilingData_.K % PARALLEL_NUM != 0) {
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            int64_t expandedSrcToDstRowIndexDb0 = 0;
            if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
                expandedSrcToDstRowIndexDb0 = nLoopIdx * curCoreHandleNumPerLoop_ + i +
                                              (tilingData_.K - 1) * Int32AlignmentProcess(curCoreHandleNum_);
            } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
                expandedSrcToDstRowIndexDb0 = biasInLoop + i * tilingData_.K + tilingData_.K - 1;
            }
            int64_t expandedPermutedRowsIndexDb0 = expandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb0);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);

            WaitFlag<HardEvent::V_S>(EVENT_ID2);
            int64_t biasIndexDb0 = 0;
            if constexpr (ISBIASEXIST) {
                biasIndexDb0 =
                    expertForSourceRowLocal.GetValue(i * Int32AlignmentProcess(tilingData_.K) + tilingData_.K - 1);
            }
            T scalesValDb0 = 1.0;
            if (tilingData_.scalesIsNull == 0) {
                scalesValDb0 = scalesLocal.GetValue(i * AlignmentProcess(tilingData_.K) + tilingData_.K - 1);
            }
            SetFlag<HardEvent::S_MTE2>(EVENT_ID2);

            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
            if (expandedPermutedRowsIndexDb0 != INVALID_ROW_INDEX) {
                DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (
                    expandedPermutedTmpUbDb0, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb0 * tilingData_.H],
                    copyParams, padParams);
            }
            if constexpr (ISBIASEXIST) {
                DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams> (biasTmpUbDb0, gmBias_[biasIndexDb0 * tilingData_.H], copyParams, padParams);
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if (expandedPermutedRowsIndexDb0 == INVALID_ROW_INDEX &&
                (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
                Duplicate(expandedPermutedTmpUbDb0, (T)0, tilingData_.H);
                PipeBarrier<PIPE_V>();
            } else {
                if constexpr (ISBIASEXIST) {
                    Add(skip1Local, expandedPermutedTmpUbDb0, biasTmpUbDb0, tilingData_.H);
                } else {
                    Adds(skip1Local, expandedPermutedTmpUbDb0, (T)0, tilingData_.H);
                }
            }
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            PipeBarrier<PIPE_V>();
            if (!initFlag_) {
                if (expandedPermutedRowsIndexDb0 == INVALID_ROW_INDEX &&
                    (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
                    Duplicate(outLocal[outRowIndex], (T)0, tilingData_.H);
                } else {
                    Muls(outLocal[outRowIndex], skip1Local, scalesValDb0, tilingData_.H);
                }
                initFlag_ = true;
                SetFlag<HardEvent::V_S>(EVENT_ID2);
            } else {
                if (expandedPermutedRowsIndexDb0 == INVALID_ROW_INDEX &&
                    (tilingData_.dropPadMode == MODE_VALUE_1 || tilingData_.dropPadMode == MODE_VALUE_3)) {
                    SetFlag<HardEvent::V_S>(EVENT_ID2);
                    PipeBarrier<PIPE_V>();
                } else {
                    Muls(skip1Local, skip1Local, scalesValDb0, tilingData_.H);
                    SetFlag<HardEvent::V_S>(EVENT_ID2);
                    PipeBarrier<PIPE_V>();
                    Add(outLocal[outRowIndex], outLocal[outRowIndex], skip1Local, tilingData_.H);
                }
            }
        }
    }
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID1);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID3);
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
__aicore__ inline void MoeFinalizeRoutingV2FpDbAllBias<T, ISBIASEXIST>::CopyOut(
    int64_t nLoopIdx, int64_t curRepeatTimes)
{
    LocalTensor<T> outLocal = outQueue_.DeQue<T>();
    DataCopyParams copyParams{
        static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(tilingData_.H * sizeof(T)), 0, 0};
    DataCopyPadCustom<T>(gmOut_[nLoopIdx * tilingData_.H * curCoreHandleNumPerLoop_], outLocal, copyParams);
    outQueue_.FreeTensor(outLocal);
}

template <typename T, const bool ISBIASEXIST>
__aicore__ inline void MoeFinalizeRoutingV2FpDbAllBias<T, ISBIASEXIST>::Process()
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
    PrepareData();
    for (int64_t n = 0; n < loopCount - 1; n++) {
        CopyIn(n, curCoreHandleNumPerLoop_);
        Compute(n, curCoreHandleNumPerLoop_);
        CopyOut(n, curCoreHandleNumPerLoop_);
    }
    // tail loop
    CopyIn(loopCount - 1, tailLoopBlock);
    Compute(loopCount - 1, tailLoopBlock);
    CopyOut(loopCount - 1, tailLoopBlock);
}

} // namespace MoeFinalizeRoutingV2
#endif // MOE_FINALIZE_ROUTING_V2_FP_DB_ALL_BIAS
