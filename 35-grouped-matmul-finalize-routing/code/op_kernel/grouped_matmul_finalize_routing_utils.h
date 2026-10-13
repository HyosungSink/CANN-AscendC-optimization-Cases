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
 * \file grouped_matmul_finalize_routing_utils.h
 * \brief
 */

#ifndef __GROUPED_MATMUL_FINALIZE_ROUTING_KERNEL_UTILS_H_
#define __GROUPED_MATMUL_FINALIZE_ROUTING_KERNEL_UTILS_H_

constexpr uint32_t thresholdBlockNum = 8;   // 8 is obtained by tests, indicating the threshold of basic block numbers
                                            // in both directions when assigning data blocks to cube cores when using
                                            // diagnal strategy
constexpr uint32_t thresholdDimM = 5;       // 5 is obtained by tests, indicating the threshold for distinguishing
                                            // strategies for large/small shapes

namespace GroupedMatmulFinalizeRouting {
using namespace AscendC;

constexpr uint64_t SYNC_AIV_TO_AIC = 3;
constexpr uint64_t SYNC_AIC_TO_AIV = 5;
constexpr uint32_t DETER_UB_SIZE = 12 * 1024;
constexpr uint32_t UB_ALIGN_LEN = 32;

struct MNConfig {
    uint32_t m = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t mIdx = 0;
    uint32_t nIdx = 0;
    uint32_t blockDimM = 0;
    uint32_t blockDimN = 0;
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint32_t offsetM = 0;
    uint32_t baseNEachSingleN = 0;    
    uint64_t workSpaceOffset = 0;
    uint64_t curBlockM = 0;
};

struct SyncConfig {
    uint64_t curM = 0;      // M of valid data in the sliding window
    uint64_t curGroup = 0;  // Current group in sliding window
    uint64_t curGroupM = 0; // The last line of curGroup
    uint64_t lowBoundM = 0; // Lower bound of sliding window
    uint64_t windowSize = 0;      // Maximum num of rows in deterministic workspace
    uint64_t baseN = 0;     // Finalize Routing BaseN
};

template<class AT_, class BT_, class CT_, class BiasT_, const auto& MM_CFG = CFG_MDL,
         bool groupListType_ = false, bool sharedInputIsNone_ = false>
struct MMImplType {
  using AT = AT_;
  using BT = BT_;
  using CT = CT_;
  using BiasT = BiasT_;
  using MT = matmul::MatmulImpl<AT, BT, CT, BiasT, MM_CFG>;
  static const bool groupListType = groupListType_;
  static const bool sharedInputIsNone = sharedInputIsNone_;
};


struct DataCopy2DDimParams {
    uint32_t dim1;
    uint32_t dim0;
    uint32_t srcDim0;
};

struct MMInitParams {
  GM_ADDR x;
  GM_ADDR weight;
  GM_ADDR bias;
  GM_ADDR group_tokens;
  GM_ADDR scale;          // notice
  GM_ADDR pertoken_scale;
  GM_ADDR offset;
  GM_ADDR logits;
  GM_ADDR token_ranks;
  GM_ADDR residual;
  GM_ADDR y;
  GM_ADDR workspace;
};

struct VectorAtomicParams {
    uint32_t curVecBaseM;
    uint32_t curVecBaseN;
    uint32_t alignBaseN;
    uint32_t offsetM;
    uint64_t mGlobalOffset;
    uint64_t yGmOffset0;
    uint64_t yGmOffset1;
};

struct VectorOffsetParams{
    uint32_t singleCoreM;
    uint32_t perTokenOffsetM;
    uint32_t offsetMStart;
    uint32_t offsetMEnd;
};

template <typename T>
__aicore__ inline T GreatestCommonDivisor(T a, T b)
{
    T c = a;
    if (a < b) {
        a = b;
        b = c;
    }
    while (b != 0) {
        c = a;
        a = b;
        b = c % b;
    }
    return a;
}

template <typename T>
__aicore__ inline T LeastCommonMultiple(T a, T b)
{
    return a * b / GreatestCommonDivisor(a, b);
}

template <typename T>
__aicore__ inline T AlignDown(T a, T base)
{
    if (unlikely(base == 0)) {
        return a;
    }
    return a / base * base;
}

__aicore__ inline void MNBlockIdxCompute(MNConfig& mnConfig, const uint32_t curBlock, const uint32_t count,
                                         const uint32_t thresholdMDimN, const uint32_t deterministicFlag)
{
    if (mnConfig.blockDimM <= thresholdDimM || thresholdDimM == 1 || deterministicFlag == 1) {
        mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;
        mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;
    } else {
        uint32_t relativeBlock = curBlock - count;
        uint32_t curThresholdM = relativeBlock >= AlignDown(mnConfig.blockDimM * mnConfig.blockDimN, thresholdMDimN) ?
                                     mnConfig.blockDimM % thresholdBlockNum : thresholdBlockNum;
        uint32_t curThresholdMThresholdN = curThresholdM * thresholdBlockNum;
        uint32_t curThresholdN =
            relativeBlock % thresholdMDimN >= AlignDown(curThresholdM * mnConfig.blockDimN, curThresholdMThresholdN) ?
                mnConfig.blockDimN % thresholdBlockNum : thresholdBlockNum;

        uint32_t localRelativeBlock = relativeBlock % thresholdMDimN % curThresholdMThresholdN;
        mnConfig.mIdx = localRelativeBlock % curThresholdM + relativeBlock / thresholdMDimN * thresholdBlockNum;
        mnConfig.nIdx = (localRelativeBlock + localRelativeBlock / LeastCommonMultiple(curThresholdM, curThresholdN)) %
                            curThresholdN +
                        relativeBlock % thresholdMDimN / curThresholdMThresholdN * thresholdBlockNum;
    }
}

template <typename T>
__aicore__ inline void DataCopyPad2DA8W4(const LocalTensor<T> &dst, const GlobalTensor<T> &src,
                                         const DataCopy2DDimParams& copyDimParams) {
    DataCopyExtParams datacopyParams;
    datacopyParams.blockCount = copyDimParams.dim1;
    datacopyParams.blockLen = copyDimParams.dim0 * sizeof(T);
    datacopyParams.srcStride = (copyDimParams.srcDim0 - copyDimParams.dim0) * sizeof(T);
    // 32: int32 -> float16, 为防止跨行数据进入同一32B block，提前每行按偶数block对齐
    datacopyParams.dstStride = Ceil(copyDimParams.dim0 * sizeof(T), 32) % 2;

    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, datacopyParams, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2DA8W4ND(const LocalTensor<T> &dst, const GlobalTensor<T> &src,
                                           const DataCopy2DDimParams& copyDimParams) {
    DataCopyExtParams params;
    params.blockCount = copyDimParams.dim1;
    params.blockLen = copyDimParams.dim0 * sizeof(T);
    params.srcStride = (copyDimParams.srcDim0 - copyDimParams.dim0) * sizeof(T);
    params.dstStride = 0;

    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, params, padParams);
    return;
}

template <typename T>
__aicore__ inline void DataCopyPad2DA8W4(const GlobalTensor<T> &dst, const LocalTensor<T> &src,
                                         const DataCopy2DDimParams& copyDimParams, uint32_t dstDim0) {
    DataCopyExtParams params;
    params.blockCount = copyDimParams.dim1;
    params.blockLen = copyDimParams.dim0 * sizeof(T);
    // 32: ub访问粒度为32B
    params.srcStride = (copyDimParams.srcDim0 - copyDimParams.dim0) * sizeof(T) / 32;
    params.dstStride = (dstDim0 - copyDimParams.dim0) * sizeof(T);
    DataCopyPad(dst, src, params);
}

} // namespace GroupedMatmulFinalizeRouting
#endif