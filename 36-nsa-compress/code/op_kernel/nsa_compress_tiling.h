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
 * \file nsa_compress_tiling.h
 * \brief
 */
#ifndef NSA_COMPRESS_TILING_H
#define NSA_COMPRESS_TILING_H

namespace NASCompress {

#pragma pack(push, 8)

struct NSACompressTiling_st {
    uint32_t totalKvSize = 0;
    uint32_t batchSize = 0;

    uint32_t headNum = 0;
    uint32_t headDim = 0;

    uint32_t weightSize = 0;
    uint32_t compressBlockSize = 0;
    uint32_t compressStride = 16;
    uint32_t compressKvSize = 0;

    uint32_t totalCompressSize = 0;
    uint32_t maxOverlapNum;
    uint32_t maxSeqLen = 0; // tiling计算的最大可搬运seqLen
};

struct CompSingleCoreInfo {
    int32_t coreBatchIdx;       // core的首个kv所在batch，在所有batch的索引
    int32_t coreSeqIdx;         // core的首个seq，相对所有seqs的索引
    int32_t coreCompressOffset; // core的首个compress_kv，相对所有compress_kv的起始索引
    int32_t coreCompressNum;    // core所需输出的token个数
    int32_t coreCompressIdx;    // core已经输出的token个数
    int32_t coreCompressSize;   // core输出的token大小
    int32_t coreHeadNums;       // core所需处理的head个数
    int32_t coreHeadIdx;        // core所处理head的起始索引
};

struct SubTilingInfo_st {
    uint32_t subKvSize = 0;
    uint32_t subSeqLen = 0;
    uint32_t subHeadNum = 0;
    uint32_t subHeadDim = 0;
    uint32_t subCompressKvSize = 0;
    uint32_t subKvSampleOffset = 0;

    __aicore__ inline uint32_t GetSubTokenLen()
    {
        return subSeqLen;
    }

    __aicore__ inline uint32_t GetSubTilingKvDim()
    {
        return subHeadNum * subHeadDim;
    }
};

#pragma pack(pop)

using NSACompressTiling = struct NSACompressTiling_st;
using SubTilingInfo = struct SubTilingInfo_st;

} // namespace NASCompress

#endif // NSA_COMPRESS_TILING_H