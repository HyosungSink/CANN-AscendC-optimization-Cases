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
 * \file nsa_compress.h
 * \brief
 */

#ifndef ASCENDC_NSA_COMPRESS_H
#define ASCENDC_NSA_COMPRESS_H

#include "kernel_operator.h"
#include "nsa_compress_seq_manager.h"
#include "nsa_compress_tiling.h"
#include "nsa_compress_common.h"
#include "../common/include/kernel/util.h"


namespace NASCompress {

template <class T> 
class KernelNASCompress {
public:
    __aicore__ inline KernelNASCompress(){};
    __aicore__ inline void Init(__gm__ uint8_t *input, __gm__ uint8_t *weight, __gm__ uint8_t *actSeqLens,
                                __gm__ uint8_t *output, NsaCompressTilingData tilingData, AscendC::TPipe *tPipe);

    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut();

    __aicore__ inline void InitWeight();

    __aicore__ inline void Process()
    {
        // 循环直到输出的token个数符合
        while (coreInfo.coreCompressIdx < coreInfo.coreCompressNum) {
            // 当前sample还能出copyin多少seq_lenth
            if (overlapMgt.seqContext[0].sampleContext.GetSampleRemainCopyLenth() != 0) {
                CopyIn();
                Compute();
                CopyOut();
                coreInfo.coreSeqIdx += subTiling.subSeqLen;
            } else {
                coreInfo.coreSeqIdx = actseqlensGm.GetValue(coreInfo.coreBatchIdx);
                coreInfo.coreBatchIdx++;
                subTiling.subSeqLen = tiling.maxSeqLen;
                ResetOverlapContext();
            }
        }
    }

    __aicore__ inline void setTiling(NsaCompressTilingData tilingData)
    {
        { // 全局信息
            tiling.batchSize = tilingData.BatchSize;
            tiling.totalKvSize = tilingData.TotalKvSize;
            tiling.totalCompressSize = tilingData.TotalCompressSize;
            tiling.headNum = tilingData.HeadNum;
            tiling.headDim = tilingData.HeadDim;
            tiling.weightSize = tilingData.WeightSize;
            tiling.compressBlockSize = tilingData.CompressBlockSize;
            tiling.compressStride = tilingData.CompressStride;
            tiling.compressKvSize = tilingData.HeadNum * tilingData.HeadDim;
            tiling.maxOverlapNum = tilingData.MaxOverlap;
            tiling.maxSeqLen = tilingData.BlocksNums;
        }

        int32_t blockId = AscendC::GetBlockIdx();
        { // core间的相对偏移量
            coreInfo.coreBatchIdx = tilingData.kvStartBatchIdx[blockId];
            coreInfo.coreSeqIdx = tilingData.kvStartTokenIdx[blockId];
            coreInfo.coreHeadNums = tilingData.PerCoreHeadNum[blockId];
            coreInfo.coreHeadIdx = tilingData.PerCoreHeadIdx[blockId];
            coreInfo.coreCompressNum = tilingData.PerCoreOutputNum[blockId];
            coreInfo.coreCompressOffset = tilingData.PerCoreStartOutputOffset[blockId];
            coreInfo.coreCompressSize = coreInfo.coreHeadNums * tiling.headDim;
            coreInfo.coreCompressIdx = 0;
        }

        { // core内的相对偏移量
            subTiling.subSeqLen = tilingData.BlocksNums;
            subTiling.subHeadDim = tiling.headDim;
            subTiling.subHeadNum = coreInfo.coreHeadNums;
            subTiling.subKvSize = subTiling.subSeqLen * coreInfo.coreHeadNums * tiling.headDim;
            subTiling.subCompressKvSize = coreInfo.coreHeadNums * tiling.headDim;
        }
    }

    __aicore__ inline int32_t GetSampleLenById(int32_t sampleIdx)
    {
        // sample lenth 前缀和模式
        int32_t sampleLenth = actseqlensGm.GetValue(sampleIdx);
        if (sampleIdx > 0) {
            sampleLenth -= actseqlensGm.GetValue(sampleIdx - 1);
        }
        return sampleLenth;
    }

    __aicore__ inline int32_t GetCoreSampleOffsetById(int32_t sampleIdx)
    {
        // sample lenth 前缀和模式
        int32_t coreInSampleOffset = this->coreInfo.coreSeqIdx;
        if (sampleIdx > 0) {
            coreInSampleOffset -= actseqlensGm.GetValue(sampleIdx - 1);
        }

        return coreInSampleOffset;
    }

    __aicore__ inline void ResetOverlapContext()
    {
        // sample lenth 前缀和模式
        int32_t sampleLenth = actseqlensGm.GetValue(this->coreInfo.coreBatchIdx);
        int32_t coreInSampleOffset = this->coreInfo.coreSeqIdx;
        if (this->coreInfo.coreBatchIdx > 0) {
            sampleLenth -= actseqlensGm.GetValue(this->coreInfo.coreBatchIdx - 1);
            coreInSampleOffset -= actseqlensGm.GetValue(this->coreInfo.coreBatchIdx - 1);
        }
        this->subTiling.subKvSampleOffset = coreInSampleOffset;
        overlapMgt.overlapIdx = 0;
        // 每个context指向当前core压缩的前overlap个compress_token sample offset
        for (int idx = 0; idx < overlapMgt.overlapNum; idx++) {
            // 设置每个overlap 的sample offset
            overlapMgt.seqContext[idx].sampleContext.InitSampleHead(
                this->coreInfo.coreBatchIdx, coreInSampleOffset + idx * this->tiling.compressStride, sampleLenth,
                &(this->tiling));
            overlapMgt.seqContext[idx].compressMeta.InitCompressTokenMetaData(&(this->tiling));

            // overlap处理完整compress_token指向的 sample offset
            overlapMgt.seqContext[idx].compressMeta._SetPreserveSamplePosDuringOverlap(
                this->tiling.compressStride * idx + coreInSampleOffset);
        }
    }

    __aicore__ inline void InitSubResult()
    {
        pipe->InitBuffer(inBufOutputTmp, subTiling.subKvSize * sizeof(float));
        this->subResultLocal = inBufOutputTmp.Get<float>();
    }

    __aicore__ inline uint32_t getCeilPower2(uint32_t num)
    {
        num--;
        num |= num >> One;
        num |= num >> Two;
        num |= num >> Four;
        num |= num >> Eight;
        num |= num >> SixTeen;
        return ++num;
    }

    __aicore__ inline void ReduceBlock(AscendC::LocalTensor<float> tensor, uint32_t reduceSize)
    {
        // align step: align reduce size to power of two, such as 15->8, 16->8
        uint32_t alignReduceSize = getCeilPower2(reduceSize) / 2;
        uint32_t addVectorNum = reduceSize - alignReduceSize;
        uint32_t dim = this->subTiling.subHeadNum * this->subTiling.subHeadDim;

        AscendC::Add(tensor, tensor, tensor[alignReduceSize * dim], addVectorNum * dim);
        AscendC::PipeBarrier<PIPE_V>();

        // reduce step: reduce compressBlockSize to 1
        while (alignReduceSize > 1) {
            alignReduceSize = alignReduceSize >> 1;
            AscendC::Add(tensor[0], tensor[0], tensor[alignReduceSize * dim], alignReduceSize * dim);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b)
    {
        return a < b ? a : b;
    }

    __aicore__ inline void UpdateResult(uint32_t dstOverlapIdx, uint32_t remainingLenth, CompressState beforeState,
                                        uint32_t remainingSubKvLen)
    {
        CompressState cur_state = overlapMgt.seqContext[dstOverlapIdx].CheckSeqCompressTokenFinished();
        if (CompressState::COMPRESS_TOKEN_COMPLETED != cur_state) {
            // compress metadata对应的首个 sub_kv
            if (beforeState != CompressState::COMPRESS_TOKEN_INITIATED) {
                AscendC::Add(overlapMgt.overlapLocal[dstOverlapIdx * subTiling.GetSubTilingKvDim()],
                             overlapMgt.overlapLocal[dstOverlapIdx * subTiling.GetSubTilingKvDim()], subResultLocal,
                             subTiling.GetSubTilingKvDim());
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::DataCopy(overlapMgt.overlapLocal[dstOverlapIdx * subTiling.GetSubTilingKvDim()],
                                  subResultLocal, subTiling.GetSubTilingKvDim());
                AscendC::PipeBarrier<PIPE_V>();
            }
        } else {
            if (beforeState == CompressState::COMPRESS_TOKEN_INITIATED &&
                cur_state == CompressState::COMPRESS_TOKEN_COMPLETED) {
                AscendC::DataCopy(overlapMgt.overlapLocal[dstOverlapIdx * subTiling.GetSubTilingKvDim()],
                                  this->subResultLocal, this->subTiling.GetSubTilingKvDim());
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                // 压缩成功，需要把当前sub_kv & overlap 交集部分(mul & reduce规约)累加到前一次数据上
                AscendC::Add(overlapMgt.overlapLocal[dstOverlapIdx * subTiling.GetSubTilingKvDim()],
                             overlapMgt.overlapLocal[dstOverlapIdx * subTiling.GetSubTilingKvDim()],
                             this->subResultLocal, this->subTiling.GetSubTilingKvDim());
                AscendC::PipeBarrier<PIPE_V>();
            }

            if (outQueCompressKv.HasTensorInQue()) {
                CopyOut();
            }
            AscendC::LocalTensor<T> compressKvCacheLocal = outQueCompressKv.AllocTensor<T>();
            AscendC::Cast(compressKvCacheLocal, overlapMgt.overlapLocal[dstOverlapIdx * subTiling.GetSubTilingKvDim()],
                          AscendC::RoundMode::CAST_ROUND, this->subTiling.subHeadNum * this->subTiling.subHeadDim);
            AscendC::PipeBarrier<PIPE_V>();

            outQueCompressKv.EnQue<T>(compressKvCacheLocal);

            // 压缩完整Token时，更新状态
            // sub_kv & overlap 相交无尾块情况。算子仅会走这种情况
            SingleTokenMetadata curCompressMeta = overlapMgt.seqContext[dstOverlapIdx].GetCompressMeta();
            // sample 状态在 Mul &&
            // Reduce后已更新，此处不需要再更新；需要更新（token_processing_lenth|compress_token_in_sample_pos|token_state)
            overlapMgt.seqContext[dstOverlapIdx].UpdateCompletedCompressTokenMetadata(
                &curCompressMeta, remainingSubKvLen - remainingLenth, overlapMgt.overlapNum, &(this->tiling));

            overlapMgt.UpdateOverlapIdx(1);
        }
    }

    __aicore__ inline void VecMulBlkMats(uint32_t overlapIdx, uint32_t subKvOffset, uint32_t reduceToken)
    {
        // 当前overlap(压缩compress_token)对应UB常驻weight_offset
        uint32_t overlapWeightOffset =
            overlapMgt.seqContext[overlapIdx].GetWeightOffset(WeightOffsetType::BUFFSET_OFFSET, &(this->subTiling));

        int rowStart = subKvOffset / this->subTiling.GetSubTilingKvDim();
        for (int rowIdx = rowStart; rowIdx < rowStart + reduceToken; rowIdx++) {
            AscendC::LocalTensor<float> dstUb =
                this->subResultLocal[(rowIdx - rowStart) * this->subTiling.GetSubTilingKvDim()];
            AscendC::LocalTensor<float> src0Ub =
                castKvCacheLocal[subKvOffset + (rowIdx - rowStart) * this->subTiling.GetSubTilingKvDim()];
            AscendC::LocalTensor<float> src1Ub =
                this->broadcastWeightLocal[overlapWeightOffset + (rowIdx - rowStart) * 8 * this->subTiling.subHeadNum];

            VecMulBlkMat(dstUb, src0Ub, src1Ub);
        }
    }
    __aicore__ inline void VecMulBlkMat(AscendC::LocalTensor<float> dstUb, AscendC::LocalTensor<float> src0Ub,
                                        AscendC::LocalTensor<float> src1Ub)
    {
        // vec mul
        // src0Ub:[1, headNum*headDim] src1Ub:[1, 8*headNum] dstUb:[1, headNum*headDim]
        constexpr uint32_t REPEAT_BLOCK_BYTE = 256;
        constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

        AscendC::BinaryRepeatParams repeatParams;
        uint32_t mask = FP32_REPEAT_ELEMENT_NUM; // 一次处理元素个数
        uint32_t loopCount = subTiling.subHeadDim / mask;
        uint32_t remainCount = subTiling.subHeadDim % mask;
        uint32_t headNum = subTiling.subHeadNum;
        uint32_t headDim = subTiling.subHeadDim;

        // src0 [1, headNum*headDim]
        repeatParams.src0BlkStride = 1;
        repeatParams.src0RepStride = Eight;

        // src1 [1, 8]
        repeatParams.src1BlkStride = 0;
        repeatParams.src1RepStride = 0;

        repeatParams.dstBlkStride = 1;
        repeatParams.dstRepStride = Eight;
        for (uint32_t headIdx = 0; headIdx < headNum; headIdx++) {
            uint32_t src0UbOffset = headDim * headIdx;
            uint32_t src1UbOffset = Eight * headIdx;
            uint32_t dstUbOffset = headDim * headIdx;
            Mul(dstUb[dstUbOffset], src0Ub[src0UbOffset], src1Ub[src1UbOffset], mask, loopCount, repeatParams);
            AscendC::PipeBarrier<PIPE_V>();

            if (remainCount > 0) {
                Mul(dstUb[dstUbOffset + loopCount * mask], src0Ub[src0UbOffset + loopCount * mask],
                    src1Ub[src1UbOffset], remainCount, 1, repeatParams);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

private:
    AscendC::TPipe *pipe;
    const int32_t QUE_DEPTH = 1;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueKvFp16;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueKvFp32;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueWeightFp32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inBufOutputTmp;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueCompressKv;

    AscendC::GlobalTensor<T> kvGm;
    AscendC::GlobalTensor<T> weightGm;
    AscendC::GlobalTensor<T> compressKvGm;
    AscendC::GlobalTensor<int64_t> actseqlensGm;

    AscendC::LocalTensor<float> broadcastWeightLocal;
    AscendC::LocalTensor<float> subResultLocal;
    AscendC::LocalTensor<float> castKvCacheLocal;

    NSACompressTiling tiling;
    struct CompSingleCoreInfo coreInfo;
    SubTilingInfo subTiling;

    using OverlapMgt = struct OverlapBufferManager_st;
    OverlapMgt overlapMgt;
};


template <typename T>
__aicore__ inline void KernelNASCompress<T>::Init(__gm__ uint8_t *input, __gm__ uint8_t *weight,
                                                  __gm__ uint8_t *actSeqLens, __gm__ uint8_t *output,
                                                  NsaCompressTilingData tilingData, AscendC::TPipe *tPipe)
{
    pipe = tPipe;

    setTiling(tilingData);

    kvGm.SetGlobalBuffer((__gm__ T *)input, this->tiling.totalKvSize * sizeof(T));
    weightGm.SetGlobalBuffer((__gm__ T *)weight, this->tiling.weightSize * sizeof(T));
    actseqlensGm.SetGlobalBuffer((__gm__ int64_t *)actSeqLens, this->tiling.batchSize * sizeof(int64_t));
    compressKvGm.SetGlobalBuffer((__gm__ T *)output, this->tiling.totalCompressSize * sizeof(T));

    pipe->InitBuffer(inQueueKvFp16, 1, this->subTiling.subKvSize * sizeof(T));
    pipe->InitBuffer(inQueueKvFp32, 1, this->subTiling.subKvSize * sizeof(float));
    pipe->InitBuffer(outQueCompressKv, 1, this->subTiling.subCompressKvSize * sizeof(T));

    InitWeight();
    InitSubResult();

    castKvCacheLocal = inQueueKvFp32.AllocTensor<float>();

    overlapMgt.InitOverlapMgt(pipe, &(this->tiling), &(this->subTiling));
    overlapMgt.overlapLocal = overlapMgt.OverlapBuffer.Get<float>();
    // sample lenth 前缀和模式
    ResetOverlapContext();
}


template <typename T> 
__aicore__ inline void KernelNASCompress<T>::CopyIn()
{
    uint32_t block_start =
        coreInfo.coreSeqIdx * tiling.headNum * tiling.headDim + coreInfo.coreHeadIdx * tiling.headDim;

    AscendC::LocalTensor<T> kvCacheLocal = inQueueKvFp16.AllocTensor<T>();

    int32_t copySeqLenth;
    int32_t sampleRemainSeqLenth = overlapMgt.seqContext[0].sampleContext.GetSampleRemainCopyLenth();
    if (sampleRemainSeqLenth < subTiling.subSeqLen) {
        subTiling.subSeqLen = sampleRemainSeqLenth;
        overlapMgt.seqContext[0].sampleContext.UpdateSampleRemainCopyLenth(0);
    } else {
        overlapMgt.seqContext[0].sampleContext.UpdateSampleRemainCopyLenth(sampleRemainSeqLenth - subTiling.subSeqLen);
    }

    AscendC::DataCopy(
        kvCacheLocal, kvGm[block_start],
        AscendC::DataCopyParams(subTiling.subSeqLen, coreInfo.coreHeadNums * tiling.headDim * sizeof(T) / 32,
                                (tiling.headNum - coreInfo.coreHeadNums) * tiling.headDim * sizeof(T) / 32, 0));

    inQueueKvFp16.EnQue(kvCacheLocal);
}


template <typename T> 
__aicore__ inline void KernelNASCompress<T>::InitWeight()
{
    int32_t ubBlockNum = 32 / sizeof(T);
    int32_t ubBlockFloatNum = 32 / sizeof(float);
    // // coreHeadNum向上最接近的一个32B倍数
    uint8_t coreHeadNumAlign = CeilDiv(coreInfo.coreHeadNums, ubBlockNum) * ubBlockNum;
    // // 对齐32B所需要的脏数据个数
    uint8_t dummySize = coreHeadNumAlign - coreInfo.coreHeadNums;
    // uint8_t dummySize = ubBlockNum - (coreInfo.coreHeadNums % ubBlockNum);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueWeightFp16;
    pipe->InitBuffer(inQueueWeightFp16, 1, tiling.compressBlockSize * coreHeadNumAlign * sizeof(T));

    AscendC::LocalTensor<T> weightLocal = inQueueWeightFp16.AllocTensor<T>();
    AscendC::DataCopyPadParams padParams = {true, 0, static_cast<uint8_t>(dummySize), 0};
    AscendC::DataCopyPad(weightLocal, weightGm[coreInfo.coreHeadIdx],
                         AscendC::DataCopyParams(tiling.compressBlockSize, coreInfo.coreHeadNums * sizeof(T),
                                                 (tiling.headNum - coreInfo.coreHeadNums) * sizeof(T), 0),
                         padParams);
    inQueueWeightFp16.EnQue<T>(weightLocal);
    weightLocal = inQueueWeightFp16.DeQue<T>();

    AscendC::TBuf<AscendC::TPosition::VECIN> inQueueCastWeightFp32;
    pipe->InitBuffer(inQueueCastWeightFp32, tiling.compressBlockSize * coreHeadNumAlign * sizeof(float));

    AscendC::LocalTensor<float> WeightLocalFP32 = inQueueCastWeightFp32.Get<float>();
    AscendC::Cast(WeightLocalFP32, weightLocal, AscendC::RoundMode::CAST_NONE,
                  tiling.compressBlockSize * coreHeadNumAlign);
    AscendC::PipeBarrier<PIPE_V>();

    uint32_t dstShape[2];
    uint32_t srcShape[2];
    srcShape[0] = tiling.compressBlockSize * coreHeadNumAlign;
    srcShape[1] = 1;
    dstShape[0] = tiling.compressBlockSize * coreHeadNumAlign;
    dstShape[1] = ubBlockFloatNum;
    AscendC::TBuf<AscendC::TPosition::VECIN> inBroadCastWeightFp32;
    pipe->InitBuffer(inBroadCastWeightFp32,
                     tiling.compressBlockSize * coreHeadNumAlign * ubBlockFloatNum * sizeof(float));

    AscendC::LocalTensor<float> BroadCastWeightLocal = inBroadCastWeightFp32.Get<float>();
    AscendC::BroadCast<float, 2, 1>(BroadCastWeightLocal, WeightLocalFP32, dstShape, srcShape);
    AscendC::PipeBarrier<PIPE_V>();

    pipe->InitBuffer(inQueueWeightFp32, 1,
                     tiling.compressBlockSize * coreInfo.coreHeadNums * ubBlockFloatNum * sizeof(float));

    broadcastWeightLocal = inQueueWeightFp32.AllocTensor<float>();
    AscendC::DataCopy(broadcastWeightLocal, BroadCastWeightLocal,
                      AscendC::DataCopyParams(tiling.compressBlockSize, coreInfo.coreHeadNums, dummySize, 0));
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T> 
__aicore__ inline void KernelNASCompress<T>::Compute()
{
    // acquire sub_compress_kv
    AscendC::LocalTensor<T> kvCacheLocal = inQueueKvFp16.DeQue<T>();
    AscendC::Cast(castKvCacheLocal, kvCacheLocal, AscendC::RoundMode::CAST_NONE, this->subTiling.subKvSize);
    AscendC::PipeBarrier<PIPE_V>();
    inQueueKvFp16.FreeTensor(kvCacheLocal);

    // 1. 依次遍历每个overlap区域
    uint32_t curOverlapIdx = overlapMgt.overlapIdx;
    for (int i = 0; i < overlapMgt.overlapNum; i++) {
        uint32_t dstOverlapIdx = (curOverlapIdx + i) % overlapMgt.overlapNum;
        int32_t result = overlapMgt.CheckSubTilingOverlap(dstOverlapIdx, this->coreInfo.coreBatchIdx,
                                                          this->subTiling.subKvSampleOffset, this->subTiling.subSeqLen,
                                                          &(this->subTiling), &(this->coreInfo));
        if (HAS_OVERLAP == result) {
            uint32_t subKvOffset = 0;
            // 确定overlap 和 sub_kv 交集, subKvOffset
            uint32_t remainingSubKvLen = this->subTiling.subSeqLen;
            uint32_t compressTokenRemainingOffset =
                overlapMgt.seqContext[dstOverlapIdx].GetPreserveSamplePosDuringOverlap() +
                overlapMgt.seqContext[dstOverlapIdx].GetProcessingLenth();
            if (this->subTiling.subKvSampleOffset >= compressTokenRemainingOffset) {
                subKvOffset = 0;
            } else {
                subKvOffset = (compressTokenRemainingOffset - this->subTiling.subKvSampleOffset) *
                              this->subTiling.GetSubTilingKvDim();
                remainingSubKvLen =
                    this->subTiling.subSeqLen - (compressTokenRemainingOffset - this->subTiling.subKvSampleOffset);
            }

            // 确定overlap 和 sub_kv 交集, 用于mul && reduce.
            uint32_t remainingLenth = overlapMgt.seqContext[dstOverlapIdx].GetRemainingLenth();
            uint32_t reduceToken = Min(remainingSubKvLen, remainingLenth);

            // 依次处理每一行
            VecMulBlkMats(dstOverlapIdx, subKvOffset, reduceToken);

            if (reduceToken > 1) {
                ReduceBlock(this->subResultLocal, reduceToken);
                AscendC::PipeBarrier<PIPE_V>();
            }

            CompressState beforeState = overlapMgt.seqContext[dstOverlapIdx].GetCompressState();
            // 确定overlap最新compress token metadata
            int32_t ret = overlapMgt.seqContext[dstOverlapIdx].UpdateCompressTokenMetadata(
                &(this->tiling), &(this->subTiling), reduceToken);
            if (ret == NSAFAILED) {
                return;
            }
            overlapMgt.seqContext[dstOverlapIdx].UpdateSampleContext(&(this->subTiling), &(this->tiling), reduceToken);

            // 根据状态决定是缓存中间结果还是更新最终结果
            UpdateResult(dstOverlapIdx, remainingLenth, beforeState, remainingSubKvLen);
        }
    }
    this->subTiling.subKvSampleOffset += this->subTiling.subSeqLen;
    return;
}

template <typename T> 
__aicore__ inline void KernelNASCompress<T>::CopyOut()
{
    if (!outQueCompressKv.HasTensorInQue()) {
        // 未压缩成功，直接返回
        return;
    }

    // 成功压缩一个compress token. 数据UB->GM
    AscendC::LocalTensor<T> compressKvCacheLocal = outQueCompressKv.DeQue<T>();
    if (coreInfo.coreCompressIdx < coreInfo.coreCompressNum) {
        uint32_t compressOffset =
            coreInfo.coreCompressOffset + coreInfo.coreCompressIdx * tiling.headNum * tiling.headDim;
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(compressKvGm[compressOffset], compressKvCacheLocal, coreInfo.coreCompressSize);
        AscendC::PipeBarrier<PIPE_ALL>();
        coreInfo.coreCompressIdx += 1;
    }
    outQueCompressKv.FreeTensor(compressKvCacheLocal);
}

} // namespace NASCompress

#endif // ASCENDC_NSA_COMPRESS_H
