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
 * \file moe_distribute_combine_a2.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_A2_H
#define MOE_DISTRIBUTE_COMBINE_A2_H
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/reduce/sum.h"
#include "utils/std/algorithm.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_combine_tiling.h"
#if __has_include("../common/inc/kernel/moe_distribute_base.h")
#include "../common/inc/kernel/moe_distribute_base.h"
#include "../common/inc/kernel/mc2_kernel_utils.h"
#else
#include "../../common/inc/kernel/moe_distribute_base.h"
#include "../../common/inc/kernel/mc2_kernel_utils.h"
#endif
namespace MoeDistributeCombineA2Impl {
constexpr uint8_t BUFFER_NUM = 2;                       // 多buf
constexpr uint32_t STATE_OFFSET = 512;                  // 状态空间偏移地址
constexpr uint32_t STATE_SPACE_SIZE = 1024 * 1024;      // 1M
constexpr uint32_t UB_ALIGN = 32;                       // UB按32字节对齐
constexpr uint32_t SELF_STATE_OFFSET = 512 * 1024;      // 本卡状态空间偏移地址
constexpr uint32_t BATCH_WRITE_ITEM_OFFSET = 8 * 1024;  // batchWriteInfo结构体地址相对于windowOut最后1M的偏移
constexpr uint32_t BATCH_WRITE_ITEM_SIZE = 32; // = sizeof(BatchWriteItem)
constexpr uint32_t U64_PER_ITEM = BATCH_WRITE_ITEM_SIZE / sizeof(uint64_t);
constexpr uint32_t U32_PER_ITEM = BATCH_WRITE_ITEM_SIZE / sizeof(uint32_t);
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t B32_PER_BLOCK = BLOCK_SIZE / sizeof(uint32_t);
constexpr uint32_t B64_PER_BLOCK = BLOCK_SIZE / sizeof(uint64_t);
constexpr uint32_t SKIP_OFFSET = 32;
constexpr uint32_t FLAG_VALUE = 0xFFFFFFFF;
constexpr uint32_t REPEAT_BYTES = 256;
constexpr uint64_t MB_SIZE = 1024 * 1024;
constexpr uint32_t A2_RANK_NUM_PER_SERVER = 8;

template <typename T>
inline __aicore__ T RoundUp(const T val, const T align)
{
    if (align == 0 || val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

struct TaskInfo {
    uint32_t startTaskId;
    uint32_t endTaskId;
    uint32_t taskNum;

    __aicore__ inline TaskInfo() {}
    __aicore__ inline void SplitCore(uint32_t taskNumTotal, uint32_t aivNum, uint32_t aivId) {
        if (aivNum == 0) {
            startTaskId = 0;
            endTaskId = 0;
            taskNum = 0;
            return;
        }

        uint32_t formerNum = taskNumTotal / aivNum;
        uint32_t tailNum = taskNumTotal % aivNum;
        startTaskId = formerNum * aivId;
        if (aivId < tailNum) {
            formerNum++;
            startTaskId += aivId;
        } else {
            startTaskId += tailNum;
        }
        taskNum = formerNum;
        endTaskId = startTaskId + taskNum;
    }
};

#define TemplateMC2TypeA2Class typename ExpandXType, typename ExpandIdxType
#define TemplateMC2TypeA2Func ExpandXType, ExpandIdxType
using namespace AscendC;
template <TemplateMC2TypeA2Class>
class MoeDistributeCombineA2 {
public:
    __aicore__ inline MoeDistributeCombineA2(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR sendCount,
                                GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR oriX, GM_ADDR constExpertAlpha1,
                                GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR XOut, GM_ADDR workspaceGM,
                                TPipe *pipe, const MoeDistributeCombineA2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void AlltoAllDispatch();
    __aicore__ inline void BuffInit();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline void Preload();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void CalXActiveMask();
    __aicore__ inline void ProcessMoeAndCopyExpert(uint32_t tokenIdx, uint32_t topKIdx);
    __aicore__ inline void ProcessConstantExpert(uint32_t tokenIdx, uint32_t topKIdx);
    __aicore__ inline void CopyPerformanceInfo();
    __aicore__ inline void SingleServerDispatch(LocalTensor<ExpandIdxType> sendCountInfo);
    __aicore__ inline void MultiServerDispatch(LocalTensor<ExpandIdxType> sendCountInfo);
    __aicore__ inline uint32_t GetRankTokenNumAndDataCopy2WindowOut(LocalTensor<ExpandIdxType> sendCountInfo,
        GlobalTensor<ExpandXType> rankWindowOut, uint32_t rankId);
    __aicore__ inline void ConstructBatchWriteInfo(LocalTensor<ExpandIdxType> sendCountInfo);

    TPipe *tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGlobal_;
    GlobalTensor<ExpandIdxType> expertIdsGlobal_;
    GlobalTensor<ExpandIdxType> expandIdxGlobal_;
    GlobalTensor<ExpandIdxType> sendCountGlobal_;
    GlobalTensor<float> expandScalesGlobal_;
    GlobalTensor<ExpandXType> expandOutGlobal_;
    GlobalTensor<ExpandXType> rankWindow_;  // 用于存对端window的变量
    GlobalTensor<ExpandXType> localOutWindow_;
    GlobalTensor<ExpandXType> localInWindow_;
    GlobalTensor<uint32_t> windowInstatusTensor_;
    GlobalTensor<uint32_t> bufferIdGlobal_;     // win区状态位置拷入相关参数
    GlobalTensor<uint64_t> workspaceGlobal_;    // 存储batchWriteInfo结构体信息
    GlobalTensor<uint32_t> workspaceGlobal32_;  // 存储batchWriteInfo结构体信息
    GlobalTensor<uint32_t> flagGlobal_;
    GlobalTensor<bool> xActiveMaskGlobal_;  // xActiveMask
    GlobalTensor<ExpandXType> oriXGlobal_;  // 表示未经过FFN的token数据，在使能copyExpert或使能constExpert的场景下需要本输入数据
    GlobalTensor<ExpandXType> constExpertAlpha1Global_; // 在使能constExpert的场景下需要输入的计算系数alpha1
    GlobalTensor<ExpandXType> constExpertAlpha2Global_; // 在使能constExpert的场景下需要输入的计算系数alpha2
    GlobalTensor<ExpandXType> constExpertVGlobal_;      // 在使能constExpert的场景下需要输入的计算系数v
    GlobalTensor<int32_t> performanceInfoI32GMTensor_;

    LocalTensor<uint64_t> batchWriteItemLocalB64;
    LocalTensor<uint32_t> batchWriteItemLocalB32;
    LocalTensor<uint32_t> recvCountLocal_;
    LocalTensor<uint32_t> expertWindowOffsetLocal_;
    LocalTensor<float> sumFloatLocal_;
    LocalTensor<ExpandIdxType> expertIdsSegLocal_;
    LocalTensor<float> expandScalesSegLocal_;
    LocalTensor<ExpandIdxType> indexCountsSegLocal_;
    LocalTensor<bool> expertMaskTensor_;
    LocalTensor<ExpandXType> tmpUb_;
    LocalTensor<uint32_t> statusTensor_;
    LocalTensor<int32_t> performanceInfoI32Tensor_;

    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR expandXGM_;
    GM_ADDR expertIdsGM_;
    GM_ADDR expandIdxGM_;
    GM_ADDR sendCountGM_;
    GM_ADDR scalesGM_;
    GM_ADDR XOutGM_;
    GM_ADDR oriXGM_;
    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};  // topK
    uint32_t aivNum_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t coreIdx_{0};              // aiv id
    uint32_t sharedExpertRankNum_{0};  // 共享专家卡数
    uint32_t moeExpertNum_{0};         // moe专家数, 等于worldSize_ - 共享专家卡数
    uint32_t localMoeExpertNum_{0};    // 每张卡的专家数
    uint32_t zeroExpertNum_{0};
    uint32_t copyExpertNum_{0};
    uint32_t constExpertNum_{0};
    uint64_t rankSizeOnWin_{0};
    uint64_t dataOffsetOnWin_{0};
    uint64_t stateOffsetOnWin_{0};
    uint32_t axisHFloatSize_{0};
    uint32_t axisHExpandXTypeSize_{0};
    uint32_t bsKAlign_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t sendRankNum_{0};
    uint32_t halfWinSize_{0};
    uint32_t dataSpaceSize_{0};
    uint32_t bufferId_{0};
    uint32_t tokenNumPerCore_{0};
    // 分核片上相对偏移
    uint32_t tokenBeginIndex_{0};
    uint32_t expertIdsSegBaseOffset_ {0};
    uint32_t expandScalesSegBaseOffset_ {0};
    uint32_t indexCountsSegBaseOffset_ {0};

    bool isInputTokenMaskFlag_{false};
    bool isInputExpertMaskFlag_{false};
    uint32_t performanceInfoSize_{0};
    bool needPerformanceInfo_{false};
    bool isSingleServer_{false};

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> moeQueue_;
    TBuf<> expertIdsBuf_;
    TBuf<> expandScalesBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> indexCountsBuf_;
    TBuf<> tokenBuf_;
    TBuf<> batchWriteItemBuf_;
    TBuf<> performanceInfoBuf_;
    // 二维expertMask
    TBuf<> expertMaskBuf_;

    TaskInfo taskInfo_;

    GlobalTensor<uint32_t> expertRecvCountGlobal_;
    GlobalTensor<uint32_t> expertWindowOffsetGlobal_;

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    __gm__ HcclOpResParam *winContext_{nullptr};
};
template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::Init(GM_ADDR expandX, GM_ADDR expertIds,
    GM_ADDR expandIdx, GM_ADDR sendCount, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR XOut, GM_ADDR workspaceGM,
    TPipe *pipe, const MoeDistributeCombineA2TilingData *tilingData)
{
    tpipe_ = pipe;
    expandXGM_ = expandX;
    expertIdsGM_ = expertIds;
    expandIdxGM_ = expandIdx;
    sendCountGM_ = sendCount;
    scalesGM_ = scales;
    oriXGM_ = oriX;
    XOutGM_ = XOut;
    rankId_ = tilingData->moeDistributeCombineInfo.epRankId;
    axisBS_ = tilingData->moeDistributeCombineInfo.bs;
    axisH_ = tilingData->moeDistributeCombineInfo.h;
    axisK_ = tilingData->moeDistributeCombineInfo.k;
    aivNum_ = tilingData->moeDistributeCombineInfo.aivNum;
    moeExpertNum_ = tilingData->moeDistributeCombineInfo.moeExpertNum;
    zeroExpertNum_ = tilingData->moeDistributeCombineInfo.zeroExpertNum;
    copyExpertNum_ = tilingData->moeDistributeCombineInfo.copyExpertNum;
    constExpertNum_ = tilingData->moeDistributeCombineInfo.constExpertNum;
    worldSize_ = tilingData->moeDistributeCombineInfo.epWorldSize;
    isInputTokenMaskFlag_ = tilingData->moeDistributeCombineInfo.isTokenMask;
    isInputExpertMaskFlag_ = tilingData->moeDistributeCombineInfo.isExpertMask;
    isSingleServer_ = worldSize_ <= A2_RANK_NUM_PER_SERVER;
    auto contextGM = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_ = (__gm__ HcclOpResParam *)contextGM;
    hccl_.InitV2(contextGM, tilingData);
    hccl_.SetCcTilingV2(offsetof(MoeDistributeCombineA2TilingData, mc2CcTiling));
    halfWinSize_ = winContext_->winSize / 2;
    dataSpaceSize_ = halfWinSize_ - STATE_SPACE_SIZE;
    windowInGM_ = hccl_.GetWindowsInAddr(rankId_);
    bufferIdGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_ + dataSpaceSize_));
    bufferId_ = bufferIdGlobal_.GetValue(0);
    windowInGM_ = windowInGM_ + halfWinSize_ * bufferId_;
    windowOutGM_ = hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_;
    coreIdx_ = GetBlockIdx();
    windowInstatusTensor_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_));
    expandXGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)expandX);
    expertIdsGlobal_.SetGlobalBuffer((__gm__ ExpandIdxType *)expertIds);
    expandIdxGlobal_.SetGlobalBuffer((__gm__ ExpandIdxType *)expandIdx);
    sendCountGlobal_.SetGlobalBuffer((__gm__ int32_t *)sendCount);
    expandScalesGlobal_.SetGlobalBuffer((__gm__ float *)scales);
    expandOutGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)XOut);
    workspaceGlobal_.SetGlobalBuffer((__gm__ uint64_t *)(windowOutGM_ + dataSpaceSize_ + BATCH_WRITE_ITEM_OFFSET));
    workspaceGlobal32_.SetGlobalBuffer((__gm__ uint32_t *)(windowOutGM_ + dataSpaceSize_ + BATCH_WRITE_ITEM_OFFSET));

    expertRecvCountGlobal_.SetGlobalBuffer((__gm__ uint32_t *)workspaceGM);
    expertWindowOffsetGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(workspaceGM + moeExpertNum_ * sizeof(uint32_t)));
    xActiveMaskGlobal_.SetGlobalBuffer((__gm__ bool*)xActiveMask);
    oriXGlobal_.SetGlobalBuffer((__gm__ ExpandXType*)oriX);
    constExpertAlpha1Global_.SetGlobalBuffer((__gm__ ExpandXType*)constExpertAlpha1);
    constExpertAlpha2Global_.SetGlobalBuffer((__gm__ ExpandXType*)constExpertAlpha2);
    constExpertVGlobal_.SetGlobalBuffer((__gm__ ExpandXType*)constExpertV);
    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    rankSizeOnWin_ = dataSpaceSize_ / worldSize_ / BLOCK_SIZE * BLOCK_SIZE;
    dataOffsetOnWin_ = rankId_ * rankSizeOnWin_;
    stateOffsetOnWin_ = dataSpaceSize_ + rankId_ * STATE_OFFSET;
    axisHFloatSize_ = axisH_ * sizeof(float);
    axisHExpandXTypeSize_ = axisH_ * sizeof(ExpandXType);
    bsKAlign_ = RoundUp(axisBS_ * axisK_, B32_PER_BLOCK);

    uint64_t stateSizeMaxSize = 2 * STATE_SPACE_SIZE; // 2: 实际上是(DATA_OFFSET+SKIP_OFFSET+sizeof(uint32)) + STATE_SPACE_SIZE，近似计算使用2 * STATE_SPACE_SIZE
    uint64_t winSizeMin = (axisBS_ * worldSize_ * (localMoeExpertNum_ > axisK_ ? axisK_ : localMoeExpertNum_) *
        axisH_ * sizeof(uint16_t) + stateSizeMaxSize) * BUFFER_NUM; // 考虑负载极其不均衡时，HCCL BUFFSIZE需要开的大小

    BuffInit();
    needPerformanceInfo_ = performanceInfo != nullptr;
    if (unlikely(needPerformanceInfo_)) {
        performanceInfoSize_ = worldSize_;
        performanceInfoI32GMTensor_.SetGlobalBuffer((__gm__ int32_t *)performanceInfo);
        tpipe_->InitBuffer(performanceInfoBuf_, RoundUp(performanceInfoSize_, B64_PER_BLOCK) * sizeof(int64_t));
        performanceInfoI32Tensor_ = performanceInfoBuf_.Get<int32_t>();
        Duplicate<int32_t>(performanceInfoI32Tensor_, 0, RoundUp(performanceInfoSize_, B64_PER_BLOCK) * sizeof(int64_t) / sizeof(int32_t));
    }
    SplitCoreCal();
}
template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::BuffInit()
{
    uint32_t expertIdsNumPerCore = Ceil(axisBS_, aivNum_) * axisK_; // 每个核分配到的task处理个数
    uint32_t expertIdsBufSizePerCore = RoundUp(expertIdsNumPerCore * static_cast<uint32_t>(sizeof(int32_t)), UB_ALIGN);
    uint32_t moeExpertNumInt32Size = RoundUp(moeExpertNum_ * static_cast<uint32_t>(sizeof(int32_t)), UB_ALIGN) ;
    tpipe_->InitBuffer(moeQueue_, BUFFER_NUM, axisHExpandXTypeSize_);
    // REPEAT_BYTES = 256 原因：在TokenActiveMaskCal函数中，复用expertIdsBuf_；
    // 若bs和k较小时，复用的大小不够，同时对齐指令需要至少256B。
    tpipe_->InitBuffer(expertIdsBuf_, Std::max(expertIdsBufSizePerCore, REPEAT_BYTES));

    tpipe_->InitBuffer(expandScalesBuf_, expertIdsBufSizePerCore);
    tpipe_->InitBuffer(tokenBuf_, Std::max(axisHFloatSize_, moeExpertNumInt32Size));
    tpipe_->InitBuffer(rowTmpFloatBuf_, Std::max(axisHFloatSize_, moeExpertNumInt32Size));
    tpipe_->InitBuffer(sumFloatBuf_, Std::max(axisHFloatSize_, moeExpertNumInt32Size));
    tpipe_->InitBuffer(indexCountsBuf_, Std::max(expertIdsBufSizePerCore, REPEAT_BYTES));
    tpipe_->InitBuffer(batchWriteItemBuf_, Std::max(BATCH_WRITE_ITEM_SIZE * worldSize_, moeExpertNumInt32Size));
    batchWriteItemLocalB64 = batchWriteItemBuf_.Get<uint64_t>();
    batchWriteItemLocalB32 = batchWriteItemLocalB64.template ReinterpretCast<uint32_t>();
    // xActivateMask是二维表
    if (isInputExpertMaskFlag_) {
        tpipe_->InitBuffer(expertMaskBuf_, RoundUp(static_cast<uint32_t>(axisBS_ * axisK_ * sizeof(bool)), UB_ALIGN));
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::TokenActiveMaskCal()
{
    LocalTensor<bool> xActiveMaskTensor;
    LocalTensor<int8_t> xActiveMaskInt8Tensor;
    LocalTensor<half> xActiveMaskHalfTensor;
    LocalTensor<half> sumOutTensor;
    LocalTensor<uint8_t> tempTensor;
    uint32_t axisBsAlignSize = RoundUp(axisBS_, UB_ALIGN);
    xActiveMaskTensor = expertIdsBuf_.Get<bool>(axisBsAlignSize);
    xActiveMaskHalfTensor = expertIdsBuf_.GetWithOffset<half>(axisBsAlignSize, axisBsAlignSize);
    sumOutTensor = expertIdsBuf_.Get<half>(UB_ALIGN);
    tempTensor = indexCountsBuf_.Get<uint8_t>();
    DataCopyExtParams xActiveMaskParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xActiveMaskTensor, xActiveMaskGlobal_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    xActiveMaskInt8Tensor = xActiveMaskTensor.ReinterpretCast<int8_t>();
    Cast(xActiveMaskHalfTensor, xActiveMaskInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize, axisBS_};
    Sum(sumOutTensor, xActiveMaskHalfTensor, tempTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    axisBS_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
    bsKAlign_ = RoundUp(axisBS_ * axisK_, B32_PER_BLOCK);
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::SplitCoreCal()
{
    // 对worldSize按卡分核，得到每个核上处理的卡的数量
    sendRankNum_ = worldSize_ / aivNum_;
    uint32_t remainderRankNum = worldSize_ % aivNum_;
    startRankId_ = sendRankNum_ * coreIdx_;
    if (coreIdx_ < remainderRankNum) {
        sendRankNum_++;
        startRankId_ += coreIdx_;
    } else {
        startRankId_ += remainderRankNum;
    }
    endRankId_ = startRankId_ + sendRankNum_;
}

template <TemplateMC2TypeA2Class>
__aicore__ inline uint32_t MoeDistributeCombineA2<TemplateMC2TypeA2Func>::GetRankTokenNumAndDataCopy2WindowOut(
    LocalTensor<ExpandIdxType> sendCountInfo, GlobalTensor<ExpandXType> rankWindowOut, uint32_t rankId)
{
    uint32_t rankTokenNum = 0;
    for (uint32_t expertId = 0; expertId < localMoeExpertNum_; ++expertId) {
        uint32_t preCount = 0;
        if (expertId != 0 || rankId != 0) {
            preCount = static_cast<uint32_t>(sendCountInfo.GetValue(expertId * worldSize_ + rankId - 1));
        }
        uint32_t startTokenAddr = preCount * axisH_;
        uint32_t tokenNum = sendCountInfo(expertId * worldSize_ + rankId) - preCount;
        for (uint32_t tokenId = 0; tokenId < tokenNum; ++tokenId) {
            LocalTensor<ExpandXType> InUb = moeQueue_.AllocTensor<ExpandXType>();
            DataCopy(InUb, expandXGlobal_[startTokenAddr], axisH_);
            moeQueue_.EnQue(InUb);
            LocalTensor<ExpandXType> OutUb = moeQueue_.DeQue<ExpandXType>();
            DataCopy(rankWindowOut[rankTokenNum * axisH_], OutUb, axisH_);
            moeQueue_.FreeTensor<ExpandXType>(OutUb);
            startTokenAddr += axisH_;
            rankTokenNum++;
        }
    }
    return rankTokenNum;
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::SingleServerDispatch(
    LocalTensor<ExpandIdxType> sendCountInfo)
{
    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; ++dstRankId) {
        localOutWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(windowOutGM_ + dstRankId * rankSizeOnWin_));

        uint32_t rankTokenNum = GetRankTokenNumAndDataCopy2WindowOut(sendCountInfo, localOutWindow_, dstRankId);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();

        GlobalTensor<ExpandXType> dstGlobal;
        dstGlobal.SetGlobalBuffer((__gm__ ExpandXType *)(hccl_.GetWindowsInAddr(dstRankId) +
            halfWinSize_ * bufferId_ + dataOffsetOnWin_));
        for (uint32_t tokenId = 0; tokenId < rankTokenNum; ++tokenId) {
            LocalTensor<ExpandXType> InUb = moeQueue_.AllocTensor<ExpandXType>();
            DataCopy(InUb, localOutWindow_[tokenId * axisH_], axisH_);
            moeQueue_.EnQue(InUb);
            LocalTensor<ExpandXType> OutUb = moeQueue_.DeQue<ExpandXType>();
            DataCopy(dstGlobal[tokenId * axisH_], OutUb, axisH_);
            moeQueue_.FreeTensor<ExpandXType>(OutUb);
        }
        PipeBarrier<PIPE_MTE3>();
        batchWriteItemLocalB32(0) = FLAG_VALUE;
        flagGlobal_.SetGlobalBuffer(
            (__gm__ uint32_t *)dstGlobal.GetPhyAddr(rankTokenNum * axisH_ + SKIP_OFFSET / sizeof(ExpandXType)));
        DataCopyExtParams copySkipOffsetFlagParams{1, static_cast<uint32_t>(sizeof(uint32_t)), 0, 0, 0};
        DataCopyPad(flagGlobal_, batchWriteItemLocalB32, copySkipOffsetFlagParams);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::ConstructBatchWriteInfo(
    LocalTensor<ExpandIdxType> sendCountInfo)
{
    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; ++dstRankId) {
        localOutWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(windowOutGM_ + dstRankId * rankSizeOnWin_));
        uint32_t rankTokenNum = GetRankTokenNumAndDataCopy2WindowOut(sendCountInfo, localOutWindow_, dstRankId);

        flagGlobal_.SetGlobalBuffer(
            (__gm__ uint32_t *)(localOutWindow_.GetPhyAddr(rankTokenNum * axisH_) + SKIP_OFFSET / sizeof(ExpandXType)));
        flagGlobal_(0) = FLAG_VALUE;
        uint32_t rankIdOffset = dstRankId - startRankId_;
        batchWriteItemLocalB64(rankIdOffset * U64_PER_ITEM) = (uint64_t)(localOutWindow_.GetPhyAddr());
        batchWriteItemLocalB64(rankIdOffset * U64_PER_ITEM + 1) =
            (uint64_t)(hccl_.GetWindowsInAddr(dstRankId) + halfWinSize_ * bufferId_ + dataOffsetOnWin_);
        batchWriteItemLocalB64(rankIdOffset * U64_PER_ITEM + 2) = rankTokenNum * axisH_ + SKIP_OFFSET / sizeof(ExpandXType) + 2;
        batchWriteItemLocalB32(rankIdOffset * U32_PER_ITEM + 6) = HcclDataType::HCCL_DATA_TYPE_FP16;
        batchWriteItemLocalB32(rankIdOffset * U32_PER_ITEM + 7) = dstRankId;
        DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            flagGlobal_);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(workspaceGlobal_[startRankId_ * U64_PER_ITEM], batchWriteItemLocalB64, sendRankNum_ * U64_PER_ITEM);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::MultiServerDispatch(
    LocalTensor<ExpandIdxType> sendCountInfo)
{
    if ASCEND_IS_AIV {
        if (coreIdx_ == 0) {
            HcclHandle handleId = hccl_.BatchWrite<true>((GM_ADDR)(workspaceGlobal_.GetPhyAddr()), worldSize_);
            bufferIdGlobal_(0) = bufferId_ ^ 1;
        }
        if (rankId_ >= startRankId_ && rankId_ < endRankId_) {
            localOutWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(windowOutGM_ + dataOffsetOnWin_));
            localInWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(windowInGM_ + dataOffsetOnWin_));
            uint32_t rankIdOffset = rankId_ - startRankId_;
            uint64_t rankTokenNum = (batchWriteItemLocalB64(rankIdOffset * 4 + 2) - SKIP_OFFSET / sizeof(ExpandXType) - 2) / axisH_;
            for (uint32_t tokenId = 0; tokenId < rankTokenNum; ++tokenId) {
                LocalTensor<ExpandXType> InUb = moeQueue_.AllocTensor<ExpandXType>();
                DataCopy(InUb, localOutWindow_[tokenId * axisH_], axisH_);
                moeQueue_.EnQue(InUb);
                LocalTensor<ExpandXType> OutUb = moeQueue_.DeQue<ExpandXType>();
                DataCopy(localInWindow_[tokenId * axisH_], OutUb, axisH_);
                moeQueue_.FreeTensor<ExpandXType>(OutUb);
            }
            flagGlobal_.SetGlobalBuffer(
                (__gm__ uint32_t *)localInWindow_.GetPhyAddr(rankTokenNum * axisH_ + SKIP_OFFSET / sizeof(ExpandXType)));
            flagGlobal_(0) = FLAG_VALUE;
            DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
                flagGlobal_);
        }
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::AlltoAllDispatch()
{
    if (sendRankNum_ == 0) {
        SyncAll<true>();
        return;
    }
    LocalTensor<ExpandIdxType> sendCountLocal = tokenBuf_.Get<int32_t>(); // 复用tokenBuf_
    DataCopy(sendCountLocal, sendCountGlobal_, RoundUp(moeExpertNum_, B32_PER_BLOCK));
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    if (isSingleServer_) {
        SingleServerDispatch(sendCountLocal);
        SyncAll<true>();
        if (coreIdx_ == 0) {
            bufferIdGlobal_(0) = bufferId_ ^ 1;
        }
    } else {
        ConstructBatchWriteInfo(sendCountLocal);
        SyncAll<true>();
        MultiServerDispatch(sendCountLocal);
    }
}
template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::Preload()
{
    recvCountLocal_ = rowTmpFloatBuf_.Get<uint32_t>();          // 复用rowTmpFloatBuf_
    expertWindowOffsetLocal_ = batchWriteItemBuf_.Get<uint32_t>();    // 复用batchWriteItemBuf_
    // 缩减UB占用，只读取1/AivNum的专家序号片段，其他核处理部分不读取
    taskInfo_.SplitCore(axisBS_ * axisK_, aivNum_, coreIdx_);
    expertIdsSegLocal_ = expertIdsBuf_.Get<ExpandIdxType>();
    DataCopyPad(expertIdsSegLocal_, expertIdsGlobal_[taskInfo_.startTaskId],
        {1, static_cast<uint32_t>(taskInfo_.taskNum * sizeof(uint32_t)), 0, 0, 0}, {false, 0, 0, 0});
    expertIdsSegBaseOffset_ = taskInfo_.startTaskId;

    Duplicate(recvCountLocal_, (uint32_t)0, moeExpertNum_);
    Duplicate(expertWindowOffsetLocal_, (uint32_t)0, moeExpertNum_);

    SyncFunc<AscendC::HardEvent::V_MTE3>();

    if (coreIdx_ == aivNum_ - 1) {
        DataCopyPad(expertRecvCountGlobal_, recvCountLocal_,
            {1, static_cast<uint32_t>(moeExpertNum_ * sizeof(uint32_t)), 0, 0, 0});
    }

    SyncAll<true>();

    if (isInputExpertMaskFlag_) {
        // 需要额外校验Mask，Mask表为全量表，专家表为片段表
        for (uint32_t i = taskInfo_.startTaskId; i < taskInfo_.endTaskId; ++i) {
            if (expertMaskTensor_(i) == false){ // 全量表，用[0-bs*k]做索引
                continue;
            }

            uint32_t expId = expertIdsSegLocal_.GetValue(i - taskInfo_.startTaskId); // 片段表，用[0-taskNum]做索引
            if (expId < moeExpertNum_) {
                recvCountLocal_(expId) += 1;
            }
        }
    } else {
        // 无需校验Mask，直接用片段表
        for (uint32_t i = 0; i < taskInfo_.taskNum; ++i) {
            uint32_t expId = expertIdsSegLocal_.GetValue(i);
            if (expId < moeExpertNum_)
                recvCountLocal_(expId) += 1;
        }
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();

    SetAtomicAdd<int32_t>();
    DataCopyPad(expertRecvCountGlobal_, recvCountLocal_,
        {1, static_cast<uint32_t>(moeExpertNum_ * sizeof(uint32_t)), 0, 0, 0});
    SetAtomicNone();

    SyncAll<true>();

    DataCopyPad(recvCountLocal_, expertRecvCountGlobal_,
        {1, static_cast<uint32_t>(moeExpertNum_ * sizeof(uint32_t)), 0, 0, 0}, {false, 0, 0, 0});

    SyncFunc<AscendC::HardEvent::MTE2_S>();

    taskInfo_.SplitCore(moeExpertNum_ / localMoeExpertNum_, aivNum_, coreIdx_);
    for (uint32_t groupIdx = taskInfo_.startTaskId; groupIdx < taskInfo_.endTaskId; ++groupIdx) {
        uint32_t start = groupIdx * localMoeExpertNum_;
        uint32_t end = start + localMoeExpertNum_;
        uint32_t prefixSum = 0;
        for (uint32_t i = start; i < end; ++i) {
            expertWindowOffsetLocal_(i - start) = prefixSum;
            prefixSum += recvCountLocal_.GetValue(i);
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(expertWindowOffsetGlobal_[start], expertWindowOffsetLocal_,
            {1, static_cast<uint32_t>(localMoeExpertNum_ * sizeof(uint32_t)), 0, 0, 0});
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
    SyncAll<true>();

    DataCopyPad(expertWindowOffsetLocal_, expertWindowOffsetGlobal_,
        {1, static_cast<uint32_t>(moeExpertNum_ * sizeof(uint32_t)), 0, 0, 0}, {false, 0, 0, 0});

    tokenNumPerCore_ = axisBS_ / aivNum_;
    uint32_t undoTokenNum = axisBS_ % aivNum_;
    tokenBeginIndex_ = 0U;
    if (coreIdx_ < undoTokenNum) {
        tokenNumPerCore_ = tokenNumPerCore_ + 1;
        tokenBeginIndex_ = coreIdx_ * tokenNumPerCore_;
    } else {
        tokenBeginIndex_ = (undoTokenNum + coreIdx_ * tokenNumPerCore_);
    }
    if (tokenNumPerCore_ == 0) {
        return;
    }

     // 缩减UB占用，只读取1/AivNum的Scale\IndexCounts片段，其他核处理部分不读取
    expandScalesSegLocal_ = expandScalesBuf_.Get<float>();
    indexCountsSegLocal_ = indexCountsBuf_.Get<ExpandIdxType>();
    DataCopyPad(expandScalesSegLocal_, expandScalesGlobal_[tokenBeginIndex_ * axisK_],
        {1, static_cast<uint32_t>(tokenNumPerCore_ * axisK_ * sizeof(uint32_t)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(indexCountsSegLocal_, expandIdxGlobal_[tokenBeginIndex_ * axisK_],
        {1, static_cast<uint32_t>(tokenNumPerCore_ * axisK_ * sizeof(ExpandIdxType)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(expertIdsSegLocal_, expertIdsGlobal_[tokenBeginIndex_ * axisK_],
        {1, static_cast<uint32_t>(tokenNumPerCore_ * axisK_ * sizeof(uint32_t)), 0, 0, 0}, {false, 0, 0, 0});

    expandScalesSegBaseOffset_ = tokenBeginIndex_ * axisK_;
    indexCountsSegBaseOffset_ = tokenBeginIndex_ * axisK_;
    expertIdsSegBaseOffset_ = tokenBeginIndex_ * axisK_;
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::WaitDispatch()
{
    if (startRankId_ >= worldSize_) {
        SyncAll<true>();
        return;
    }
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t waitFlagNum = 0;
    int64_t startTime = GetCurrentTimestampUs();
    while (waitFlagNum < sendRankNum_) {
        for (uint32_t rankId = startRankId_; rankId < endRankId_; ++rankId) {
            uint32_t tokenIdx = (rankId + 1) * localMoeExpertNum_ - 1;
            GM_ADDR wAddr = windowInGM_ + rankSizeOnWin_ * rankId + SKIP_OFFSET +
                            (recvCountLocal_(tokenIdx) + expertWindowOffsetLocal_(tokenIdx)) * axisHExpandXTypeSize_;
            flagGlobal_.SetGlobalBuffer((__gm__ uint32_t *)wAddr);
            DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(flagGlobal_);

            uint32_t flag = flagGlobal_(0);
            if (flag != FLAG_VALUE) {
                continue;
            }
            waitFlagNum++;
            flagGlobal_(0) = 0;
            // 重要：要下DCCI保证清零写进去，避免下一次判断时又判断生效，重复累计waitFlagNum
            DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(flagGlobal_);
            if (unlikely(needPerformanceInfo_)) {
                auto srcRankId = rankId;
                RecordRankCommDuration(performanceInfoI32Tensor_, srcRankId, startTime);
            }
        }
    }
    SyncAll<true>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::CalXActiveMask()
{
    if (isInputTokenMaskFlag_) {
        TokenActiveMaskCal(); // 计算一维mask
    }
    if (isInputExpertMaskFlag_) {
        expertMaskTensor_ = expertMaskBuf_.Get<bool>();
        DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
        DataCopyExtParams maskParams{1U, static_cast<uint32_t>(axisBS_ * axisK_ * sizeof(bool)), 0U, 0U, 0U};
        DataCopyPad(expertMaskTensor_, xActiveMaskGlobal_, maskParams, maskCopyPadParams);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::CopyPerformanceInfo()
{
    if (unlikely(needPerformanceInfo_)) {
        AscendC::SetAtomicAdd<int32_t>();
        AscendC::DataCopyPad(performanceInfoI32GMTensor_, performanceInfoI32Tensor_,
            {1, static_cast<uint32_t>(performanceInfoSize_ * sizeof(int64_t)), 0, 0, 0});
        AscendC::SetAtomicNone();
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::Process()
{
    if ASCEND_IS_AIV {
        CalXActiveMask();
        AlltoAllDispatch();
        Preload();
        WaitDispatch();
        LocalWindowCopy();
        CopyPerformanceInfo();
        hccl_.Finalize();
    }
}
template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::LocalWindowCopy()
{
    sumFloatLocal_ = sumFloatBuf_.Get<float>();
    if (tokenNumPerCore_ == 0) {
        return;
    }
    // step 4 & step 5
    uint32_t expId = 0U;
    float scaleVal = 0.0;
    for (uint32_t i = 0; i < tokenNumPerCore_; i++) {
        uint32_t tokenIdx = tokenBeginIndex_ + i;
        Duplicate(sumFloatLocal_, 0.0f, axisH_);
        for (uint32_t topKIdx = 0; topKIdx < axisK_; topKIdx++) {
            uint32_t tokentopKIdx = tokenIdx * axisK_ + topKIdx;
            if (isInputExpertMaskFlag_) {
                bool maskExpertFlag = expertMaskTensor_.GetValue(tokentopKIdx);
                if (!maskExpertFlag) {
                    continue;
                }
            }
            expId = expertIdsSegLocal_.GetValue(tokentopKIdx - expertIdsSegBaseOffset_);
            if (expId < moeExpertNum_) {
                ProcessMoeAndCopyExpert(tokenIdx, topKIdx);
            } else if (expId < moeExpertNum_ + zeroExpertNum_) {
                continue; // 零专家不需要任何操作
            } else if (expId < moeExpertNum_ + zeroExpertNum_ + copyExpertNum_) {
                ProcessMoeAndCopyExpert(tokenIdx, topKIdx);
            } else if (expId < moeExpertNum_ + zeroExpertNum_ + copyExpertNum_ + constExpertNum_) {
                ProcessConstantExpert(tokenIdx, topKIdx);
            }
        }
        // 结果搬出
        PipeBarrier<PIPE_V>();
        LocalTensor<ExpandXType> sumBufLocal_ = tokenBuf_.Get<ExpandXType>();
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Cast(sumBufLocal_, sumFloatLocal_, AscendC::RoundMode::CAST_RINT, axisH_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(expandOutGlobal_[tokenIdx * axisH_], sumBufLocal_, axisH_);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::ProcessMoeAndCopyExpert(
    uint32_t tokenIdx, uint32_t topKIdx)
{
    GM_ADDR wAddr;
    LocalTensor<float> rowTmpFloatLocal = rowTmpFloatBuf_.Get<float>();
    uint32_t tokentopKIdx = tokenIdx * axisK_ + topKIdx;
    float scaleVal = expandScalesSegLocal_.GetValue(tokentopKIdx - expandScalesSegBaseOffset_);
    uint32_t expId = expertIdsSegLocal_.GetValue(tokentopKIdx - expertIdsSegBaseOffset_);
    if (expId < moeExpertNum_) {
        uint32_t rank = expId / localMoeExpertNum_;
        wAddr = (__gm__ uint8_t *)(windowInGM_) + rankSizeOnWin_ * rank +
            expertWindowOffsetLocal_.GetValue(expId) * axisHExpandXTypeSize_ +
            indexCountsSegLocal_.GetValue(tokentopKIdx - indexCountsSegBaseOffset_) * axisHExpandXTypeSize_;
    } else {
        wAddr = (__gm__ uint8_t *)(oriXGM_) + tokenIdx * axisHExpandXTypeSize_;
    }
    // copy experts from window
    rankWindow_.SetGlobalBuffer((__gm__ ExpandXType *)wAddr);
    tmpUb_ = moeQueue_.AllocTensor<ExpandXType>();
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    DataCopy(tmpUb_, rankWindow_, axisH_);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    // cast before muls
    Cast(rowTmpFloatLocal, tmpUb_, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    // muls expert and scaleVal, use inplace scalar muls, do not need extra buf
    AscendC::Muls(rowTmpFloatLocal, rowTmpFloatLocal, scaleVal, axisH_);//tokenXscale
    PipeBarrier<PIPE_V>();
    // add rowTmpFloatLocal to sumFloatBufLocal
    AscendC::Add(sumFloatLocal_, sumFloatLocal_, rowTmpFloatLocal, axisH_);
    moeQueue_.FreeTensor<ExpandXType>(tmpUb_);
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::ProcessConstantExpert(
    uint32_t tokenIdx, uint32_t topKIdx)
{
    PipeBarrier<PIPE_ALL>();
    uint32_t tokentopKIdx = tokenIdx * axisK_ + topKIdx;
    float scaleVal = expandScalesSegLocal_.GetValue(tokentopKIdx - expandScalesSegBaseOffset_);
    uint32_t expId = expertIdsSegLocal_.GetValue(tokentopKIdx - expertIdsSegBaseOffset_);
    uint32_t constExpertIdx = expId - (moeExpertNum_ + zeroExpertNum_ + copyExpertNum_);

    LocalTensor<float> constVFloatLocal = tokenBuf_.Get<float>();
    LocalTensor<float> constXFloatLocal = rowTmpFloatBuf_.Get<float>();
    LocalTensor<ExpandXType> constVInUB = moeQueue_.AllocTensor<ExpandXType>();
    LocalTensor<ExpandXType> constXInUB = moeQueue_.AllocTensor<ExpandXType>();

    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(axisHExpandXTypeSize_), 0U, 0U, 0U};

    // 直接从GM读取当前常量专家的alpha1和alpha2参数
    ExpandXType alpha1 = constExpertAlpha1Global_.GetValue(constExpertIdx);
    ExpandXType alpha2 = constExpertAlpha2Global_.GetValue(constExpertIdx);

    float alpha1Float;
    float alpha2Float;
    if constexpr (std::is_same_v<ExpandXType, bfloat16_t>) {
        alpha1Float = ToFloat(alpha1);
        alpha2Float = ToFloat(alpha2);
    } else {
        alpha1Float = static_cast<float>(alpha1);
        alpha2Float = static_cast<float>(alpha2);
    }

    // 读取输入token并转float
    DataCopyPad(constVInUB, constExpertVGlobal_[constExpertIdx * axisH_], expandXCopyParams, copyPadExtParams);
    DataCopyPad(constXInUB, oriXGlobal_[tokenIdx * axisH_], expandXCopyParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    Cast(constXFloatLocal, constXInUB, AscendC::RoundMode::CAST_NONE, axisH_);
    Cast(constVFloatLocal, constVInUB, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    moeQueue_.FreeTensor<ExpandXType>(constVInUB);
    moeQueue_.FreeTensor<ExpandXType>(constXInUB);

    // 计算 alpha1 * x + alpha2 * v ,结果存放到x
    AscendC::Muls(constXFloatLocal, constXFloatLocal, alpha1Float, axisH_);
    AscendC::Muls(constVFloatLocal, constVFloatLocal, alpha2Float, axisH_);
    PipeBarrier<PIPE_V>();
    AscendC::Add(constXFloatLocal, constXFloatLocal, constVFloatLocal, axisH_);
    PipeBarrier<PIPE_V>();

    // 乘以专家权重
    AscendC::Muls(constXFloatLocal, constXFloatLocal, scaleVal, axisH_);
    PipeBarrier<PIPE_V>();
    AscendC::Add(sumFloatLocal_, sumFloatLocal_, constXFloatLocal, axisH_);
    PipeBarrier<PIPE_V>();
}

}  // namespace MoeDistributeCombineA2Impl
#endif  // MOE_DISTRIBUTE_COMBINE_A2_H
