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
 * \file nsa_compress_seq_manager.h
 * \brief
 */

#ifndef NSA_COMPRESS_SEQ_MANAGER_H
#define NSA_COMPRESS_SEQ_MANAGER_H

#include "kernel_operator.h"
#include "nsa_compress_common.h"
#include "nsa_compress_tiling.h"


namespace NASCompress {

struct SampleManager_st {
    // 当前compress_token 对应的sampleId. 范围[0, actual_seqs_len-1]
    int32_t sampleIdx = static_cast<int32_t>(SampleHead::SAMPLE_HEAD_INVALID);
    // 当前compress_token 对应的整个sample长度。sample 中间分核时，值不变; value = actual_seqs_len[sampleIdx] -
    // actual_seqs_len[sampleIdx-1]
    int32_t sampleLen = 0;
    // 当前compress_token正在处理的 sub_kv 对应的sample偏移。相对sample起始相对偏移; 范围[0, sample].
    // 每个overlap区域之间间隔stride倍数
    int32_t sampleOffset = 0;

    // 当前compress_token 对应的sample偏移。相对sample起始相对偏移; 范围[0, sample].
    // 一个Sample压缩期间固定不变。跨Sample时更新
    int32_t sampleOriginOffset = 0;

    // 当前正在处理的sample head指向的sample信息
    SampleHeadState sample_state = SampleHeadState::SAMPLE_UNDEF_COMPRESS_TOKEN;

    // 当前core指向sample 已经压缩出多少compress_token
    int32_t finished_compress_num = 0;
    // 当前core指向sample 需要处理compress_token数量
    int32_t remainingCompressCount = 0;
    // 当前core指向sample还可以copyin的seq_len长度
    int32_t remainingCopySeq = 0;

    // 当前core需要处理的sample区间对应的压缩tokens数量
    int32_t core_compress_tokens = 0;

    __aicore__ inline int32_t _GetSampleId()
    {
        return sampleIdx;
    }

    __aicore__ inline int32_t _GetSampleLen()
    {
        return sampleLen;
    }

    __aicore__ inline int32_t _GetSampleOffset()
    {
        return sampleOffset;
    }

    /// @brief 更新当前core正在处理的Sample head信息
    /// @param id
    /// @param lenth
    /// @return
    __aicore__ inline void UpdateSampleHead(int32_t lenth, NSACompressTiling *tiling)
    {
        sampleOffset += lenth;
    }

    /// @brief 初始化当前core正在处理的Sample head信息
    /// @param sid 当前compress_token 对应的sampleId
    /// @param offset 当前compress_token正在处理的 sub_kv 对应的sample偏移。相对sample起始相对偏移; 范围[0, sample].
    /// 每个overlap区域之间间隔stride倍数
    /// @param lenth 当前compress_token 对应的整个sample长度。sample 中间分核时，值不变;
    /// @param tiling tiling参数
    /// @return
    __aicore__ inline void InitSampleHead(int32_t sid, int32_t offset, int32_t lenth, NSACompressTiling *tiling)
    {
        sampleIdx = sid;
        sampleLen = lenth;
        sampleOffset = offset;
        sampleOriginOffset = offset;
        sample_state = SampleHeadState::SAMPLE_FIRST_COMPRESS_TOKEN;
        int32_t rear = 0;
        if (sampleLen - sampleOriginOffset >= tiling->compressBlockSize) {
            remainingCompressCount =
                (sampleLen - sampleOriginOffset - tiling->compressBlockSize) / tiling->compressStride + 1;
            remainingCopySeq = tiling->compressBlockSize + (remainingCompressCount - 1) * tiling->compressStride;
        } else {
            remainingCompressCount = 0;
            remainingCopySeq = 0;
        }
        finished_compress_num = 0;
    }

    __aicore__ inline int32_t GetSampleRemainCopyLenth()
    {
        return remainingCopySeq;
    }

    __aicore__ inline void UpdateSampleRemainCopyLenth(int32_t value)
    {
        remainingCopySeq = value;
    }

    __aicore__ inline int32_t GetSampleHeadOffset()
    {
        return sampleOffset;
    }
};

struct SingleCompressionTokenMetadata_st {
    uint32_t weightOffset = 0;
    enum WeightOffsetType offsetType = WeightOffsetType::TOKEN_OFFSET;
    const uint32_t WEIGHT_OFFSET_INVALID = 0xFFFF;

    // sub_kv对应的compress token 已处理长度. 当tokenProcessingLenth = compressBlockSize，已完整压缩一个token
    uint32_t tokenProcessingLenth = 0;
    uint32_t compressBlockSize = 0;

    // sub_kv所属CompressToken对应的Sample Offset; 相对sample的偏移，非core见分核的起始点偏移
    uint32_t compressTokenInSampleOffset = 0;
    enum CompressState tokenState = CompressState::COMPRESS_TOKEN_UNDEF;

    __aicore__ inline uint32_t _GetProcessingLenth()
    {
        return tokenProcessingLenth;
    }

    __aicore__ inline void _SetPreserveSamplePosDuringOverlap(uint32_t offset)
    {
        compressTokenInSampleOffset = offset;
    }

    __aicore__ inline uint32_t _GetPreserveSamplePosDuringOverlap()
    {
        return compressTokenInSampleOffset;
    }

    __aicore__ inline int32_t _GetRemainingLenth()
    {
        return compressBlockSize - tokenProcessingLenth;
    }

    __aicore__ inline void RefreshSingleCompressionTokenMetadata(struct SingleCompressionTokenMetadata_st *meta,
                                                                 uint32_t processingLenth, uint32_t newSampleOffset,
                                                                 CompressState state)
    {
        weightOffset = meta->weightOffset;
        offsetType = meta->offsetType;
        tokenProcessingLenth = meta->tokenProcessingLenth;
        compressBlockSize = meta->compressBlockSize;

        tokenProcessingLenth = processingLenth;
        compressTokenInSampleOffset = newSampleOffset;
        tokenState = state;
    }

    __aicore__ inline void InitCompressTokenMetaData(NSACompressTiling *tiling)
    {
        uint32_t initWeightOffset = 0;
        SetCompressBlockSize(tiling->compressBlockSize);
        UpdateWeightOffset(initWeightOffset);
        tokenProcessingLenth = 0;
        SetCompressState(CompressState::COMPRESS_TOKEN_INITIATED);
    }

    __aicore__ inline uint32_t GetCompressBlockSize()
    {
        return compressBlockSize;
    }

    __aicore__ inline uint32_t GetCurTokenProccessing()
    {
        return tokenProcessingLenth;
    }

    __aicore__ inline void UpdateCurTokenProccessing(uint32_t lenth)
    {
        tokenProcessingLenth += lenth;
        if (CompressState::COMPRESS_TOKEN_COMPLETED == _CheckCompressTokenFinished()) {
            SetCompressState(CompressState::COMPRESS_TOKEN_COMPLETED);
        } else {
            SetCompressState(CompressState::COMPRESS_TOKEN_PROCESSING);
        }
        return;
    }

    __aicore__ inline uint32_t _GetWeightOffset(WeightOffsetType type, SubTilingInfo *subtiling = nullptr)
    {
        if (type == WeightOffsetType::BUFFSET_OFFSET && subtiling == nullptr) {
            return WEIGHT_OFFSET_INVALID;
        }
        if (type == WeightOffsetType::BUFFSET_OFFSET && subtiling) {
            uint32_t weight_dim = 8;
            return weightOffset * subtiling->subHeadNum * weight_dim;
        }
        if (type == WeightOffsetType::TOKEN_OFFSET) {
            return weightOffset;
        }
        return WEIGHT_OFFSET_INVALID;
    }

    __aicore__ inline void UpdateWeightOffset(uint32_t offset)
    {
        weightOffset = offset;
    }

    /// @brief 检查当前sub_kv 对应的compress token状态。不修改状态
    /// @return
    __aicore__ inline CompressState _CheckCompressTokenFinished()
    {
        if (compressBlockSize == 0) {
            return CompressState::COMPRESS_TOKEN_UNDEF;
        }
        if (tokenProcessingLenth == 0)
            return CompressState::COMPRESS_TOKEN_INITIATED;
        return tokenProcessingLenth == compressBlockSize ? CompressState::COMPRESS_TOKEN_COMPLETED :
                                                           CompressState::COMPRESS_TOKEN_PROCESSING;
    }

    __aicore__ inline void SetCompressBlockSize(uint32_t size)
    {
        compressBlockSize = size;
    }

    /// @brief 获取本次compute时对应的Compress Token状态。
    /// @return
    __aicore__ inline CompressState _GetCompressState()
    {
        return tokenState;
    }

    __aicore__ inline void SetCompressState(CompressState state)
    {
        tokenState = state;
    }
};
using SingleTokenMetadata = struct SingleCompressionTokenMetadata_st;

struct SequenceManager {
    using SampleMgt = struct SampleManager_st;

    SampleMgt sampleContext;
    SingleTokenMetadata compressMeta;

    __aicore__ inline SingleTokenMetadata GetCompressMeta()
    {
        return compressMeta;
    }

    /// @brief overlap(压缩compress_token) 对应Sample offset起始位置
    /// @return
    __aicore__ inline uint32_t GetPreserveSamplePosDuringOverlap()
    {
        return compressMeta._GetPreserveSamplePosDuringOverlap();
    }

    __aicore__ inline uint32_t CalculateCompressTokenOffsetInSample()
    {
        return sampleContext.GetSampleHeadOffset();
    }

    /// @brief NSACompress Op调用Wrapper. 判断本次UB 上数据Compute计算完能否压缩出一个完整Compress Token
    /// @return
    __aicore__ inline CompressState CheckSeqCompressTokenFinished()
    {
        return compressMeta._CheckCompressTokenFinished();
    }

    /// @brief UB上执行一次subTiling Token压缩，更新输出CompressToken元数据信息
    ///        具体包括 tokenProcessingLenth & weightOffset & CompressState; 不更新compressTokenInSampleOffset
    /// @param subtiling UB Tiling
    /// @param tiling NSACompress Tiling相关信息
    /// @return
    __aicore__ inline int32_t UpdateCompressTokenMetadata(NSACompressTiling *tiling, SubTilingInfo *subtiling,
                                                          uint32_t curCompressLen)
    {
        uint32_t tokenLen = curCompressLen;
        if (compressMeta.GetCurTokenProccessing() + tokenLen <= compressMeta.GetCompressBlockSize()) {
            compressMeta.UpdateCurTokenProccessing(tokenLen);

            uint32_t nextOffset =
                (compressMeta._GetWeightOffset(WeightOffsetType::TOKEN_OFFSET) + tokenLen) % tiling->compressBlockSize;
            compressMeta.UpdateWeightOffset(nextOffset);

            if (compressMeta.GetCurTokenProccessing() < compressMeta.GetCompressBlockSize()) {
                compressMeta.SetCompressState(CompressState::COMPRESS_TOKEN_PROCESSING);
            }
            return NSASuccess;
        } else {
            return NSAFAILED;
        }
    }

    __aicore__ inline void UpdateCompletedCompressTokenMetadata(SingleTokenMetadata *meta, uint32_t tailLenth,
                                                                uint32_t overlapNum, NSACompressTiling *tiling)
    {
        // （tokenProcessingLenth|compressTokenInSampleOffset|tokenState)
        uint32_t newSampleOffset =
            compressMeta._GetPreserveSamplePosDuringOverlap() + overlapNum * tiling->compressStride;
        CompressState state =
            tailLenth == 0 ? CompressState::COMPRESS_TOKEN_INITIATED : CompressState::COMPRESS_TOKEN_PROCESSING;
        if (tailLenth > 0 && tailLenth <= overlapNum * tiling->compressStride) {
            tailLenth = 0;
            state = CompressState::COMPRESS_TOKEN_INITIATED;
        }
        compressMeta.RefreshSingleCompressionTokenMetadata(meta, tailLenth, newSampleOffset, state);
    }

    /// @brief UB上执行一次subTiling Token压缩，更新当前Core处理的Sample offset相关信息
    /// @param subtiling
    /// @param tiling
    /// @return
    __aicore__ inline void UpdateSampleContext(SubTilingInfo *subtiling, NSACompressTiling *tiling,
                                               uint32_t curCompressLen)
    {
        uint32_t len = curCompressLen;
        sampleContext.UpdateSampleHead(len, tiling);
    }

    /// @brief NSACompress Op调用Wrapper. 获取本次compute时对应的Compress Token状态
    /// @return
    __aicore__ inline CompressState GetCompressState()
    {
        return compressMeta._GetCompressState();
    }

    /// @brief
    /// @return
    __aicore__ inline uint32_t GetWeightOffset(WeightOffsetType offsetType, SubTilingInfo *subtiling = nullptr)
    {
        return compressMeta._GetWeightOffset(offsetType, subtiling);
    }

    __aicore__ inline uint32_t GetProcessingLenth()
    {
        return compressMeta._GetProcessingLenth();
    }

    __aicore__ inline int32_t GetRemainingLenth()
    {
        return compressMeta._GetRemainingLenth();
    }

    __aicore__ inline int32_t GetSampleLen()
    {
        return sampleContext._GetSampleLen();
    }

    __aicore__ inline int32_t GetSampleId()
    {
        return sampleContext._GetSampleId();
    }

    __aicore__ inline int32_t GetSampleOffset()
    {
        return sampleContext._GetSampleOffset();
    }

    __aicore__ inline int32_t ContextCheckSubTilingOverlap(uint32_t offset, uint32_t len)
    {
        // overlap_mata start: overlapMeta[i].compressTokenInSampleOffset. len = compressBlockSize
        // cur_sub_kv: start: offset, len = sub_kv_len
        if (offset + len <= compressMeta.compressTokenInSampleOffset ||
            compressMeta.compressTokenInSampleOffset + compressMeta.compressBlockSize <= offset) {
            return NO_OVERLAP;
        }
        return HAS_OVERLAP;
    }
};

struct OverlapBufferManager_st {
    uint32_t overlapNum = 0;
    uint32_t overlapBufferSize = 0;
    uint32_t overlapIdx = 0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> OverlapBuffer;
    AscendC::LocalTensor<float> overlapLocal;
    SingleTokenMetadata overlapMeta[MAX_OVERLAP_NUM];
    struct SequenceManager seqContext[MAX_OVERLAP_NUM];

    __aicore__ inline int32_t CheckSubTilingOverlap(int overlapIdx, int32_t sampleId, int32_t offset, uint32_t len,
                                                    SubTilingInfo *subptr, struct CompSingleCoreInfo *pcore)
    {
        int result = HAS_OVERLAP;
        if (sampleId != seqContext[overlapIdx].GetSampleId()) {
            result = NO_OVERLAP;
        }

        // sub_kv offset + len < 当前overlap 起始位置 + tokenProcessingLenth || sub_kv offset >= 当前overlap 起始位置
        // + compressBlockSize sub_kv offset + len < 当前overlap 起始位置 + tokenProcessingLenth
        uint32_t overlapPreserveOffset = seqContext[overlapIdx].GetPreserveSamplePosDuringOverlap();
        uint32_t compressBlockSize = seqContext[overlapIdx].GetCompressMeta().compressBlockSize;
        uint32_t compressBlockProcessingLenth = seqContext[overlapIdx].GetProcessingLenth();
        if (overlapPreserveOffset + compressBlockSize <= offset ||
            offset + len <= overlapPreserveOffset + compressBlockProcessingLenth) {
            result = NO_OVERLAP;
        }
        return result;
    }

    __aicore__ inline void UpdateOverlapIdx(uint32_t delta)
    {
        overlapIdx = (overlapIdx + delta) % overlapNum;
    }

    __aicore__ inline int32_t InitOverlapMgt(AscendC::TPipe *pipe, NSACompressTiling *tiling, SubTilingInfo *subtiling)
    {
        {
            overlapIdx = 0;
            overlapNum = tiling->maxOverlapNum;
            overlapBufferSize = overlapNum * subtiling->GetSubTilingKvDim() * sizeof(float);
            pipe->InitBuffer(OverlapBuffer, overlapBufferSize);
        }
        return NSASuccess;
    }
};

} // namespace NASCompress

#endif // NSA_COMPRESS_SEQ_MANAGER_H