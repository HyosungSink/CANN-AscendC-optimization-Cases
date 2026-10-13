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
 * \file flash_attention_score_empty_tensor.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_EMPTY_TENSOR_H
#define FLASH_ATTENTION_SCORE_EMPTY_TENSOR_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "flash_attention_score_tiling.h"

template <typename INPUT_T> class FlashAttentionScoreEmptyTensor {
public:
    __aicore__ inline FlashAttentionScoreEmptyTensor(){};
    __aicore__ inline void Init(__gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *attentionOut,
                                const FlashAttentionScoreTilingData *__restrict tiling);
    __aicore__ inline void Process();

protected:
    AscendC::GlobalTensor<float> softmaxMaxGm;
    AscendC::GlobalTensor<float> softmaxSumGm;
    AscendC::GlobalTensor<INPUT_T> attentionOutGm;
    uint32_t tmpBlockIdx;
    const FlashAttentionScoreTilingData *__restrict tilingData;
    __aicore__ inline void ComputeEachCore();
};

template <typename INPUT_T>
__aicore__ inline void
FlashAttentionScoreEmptyTensor<INPUT_T>::Init(__gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                                              __gm__ uint8_t *attentionOut,
                                              const FlashAttentionScoreTilingData *__restrict tiling)
{
    tmpBlockIdx = AscendC::GetBlockIdx();
    tilingData = tiling;
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    attentionOutGm.SetGlobalBuffer((__gm__ INPUT_T *)attentionOut);
}

template <typename INPUT_T> __aicore__ inline void FlashAttentionScoreEmptyTensor<INPUT_T>::ComputeEachCore()
{
    uint32_t coreNum = tilingData->emptyInputTilingData.coreNum;
    uint32_t attentionOutFormerNum = tilingData->emptyInputTilingData.attentionOutFormerNum;
    uint32_t attentionOutTailNum = tilingData->emptyInputTilingData.attentionOutTailNum;
    uint32_t softmaxMaxFormerNum = tilingData->emptyInputTilingData.softmaxMaxFormerNum;
    uint32_t softmaxMaxTailNum = tilingData->emptyInputTilingData.softmaxMaxTailNum;
    uint64_t attentionOutSingleCoreDataSize = tilingData->emptyInputTilingData.attentionOutSingleCoreDataSize;
    uint64_t attentionOutTailCoreDataSize = tilingData->emptyInputTilingData.attentionOutTailCoreDataSize;
    uint64_t softmaxMaxSingleCoreDataSize = tilingData->emptyInputTilingData.softmaxMaxSingleCoreDataSize;
    uint64_t softmaxMaxTailCoreDataSize = tilingData->emptyInputTilingData.softmaxMaxTailCoreDataSize;
    uint64_t attentionOutLastCoreDataSize = tilingData->emptyInputTilingData.attentionOutLastCoreDataSize;
    uint64_t attentionOutLastCoreIndex = tilingData->emptyInputTilingData.attentionOutLastCoreIndex;

    // 初始化 attentionOut
    if (attentionOutFormerNum == coreNum || (attentionOutFormerNum + attentionOutTailNum) < coreNum) {
        if (tmpBlockIdx < attentionOutFormerNum - 1) {
            AscendC::InitOutput<INPUT_T>(attentionOutGm[tmpBlockIdx * attentionOutSingleCoreDataSize],
                                         attentionOutSingleCoreDataSize, static_cast<INPUT_T>(0.0));
        } else if (tmpBlockIdx == attentionOutFormerNum - 1) {
            AscendC::InitOutput<INPUT_T>(attentionOutGm[attentionOutLastCoreIndex], attentionOutLastCoreDataSize,
                                         static_cast<INPUT_T>(0.0));
        }
    } else {
        if (tmpBlockIdx < attentionOutFormerNum) {
            AscendC::InitOutput<INPUT_T>(attentionOutGm[tmpBlockIdx * attentionOutSingleCoreDataSize],
                                         attentionOutSingleCoreDataSize, static_cast<INPUT_T>(0.0));
        } else if (tmpBlockIdx >= attentionOutFormerNum && tmpBlockIdx < coreNum - 1) {
            AscendC::InitOutput<INPUT_T>(
                attentionOutGm[attentionOutFormerNum * attentionOutSingleCoreDataSize +
                               (tmpBlockIdx - attentionOutFormerNum) * attentionOutTailCoreDataSize],
                attentionOutTailCoreDataSize, static_cast<INPUT_T>(0.0));
        } else if (tmpBlockIdx == coreNum - 1) {
            AscendC::InitOutput<INPUT_T>(attentionOutGm[attentionOutLastCoreIndex], attentionOutLastCoreDataSize,
                                         static_cast<INPUT_T>(0.0));
        }
    }

    // 初始化 softmaxMax 和 softmaxSum
    if (tmpBlockIdx >= (softmaxMaxFormerNum + softmaxMaxTailNum)) {
        return;
    } else if (tmpBlockIdx < softmaxMaxFormerNum) {
        AscendC::InitOutput<float>(softmaxMaxGm[tmpBlockIdx * softmaxMaxSingleCoreDataSize],
                                   tilingData->emptyInputTilingData.softmaxMaxSingleCoreDataSize,
                                   static_cast<float>(0.0));
        AscendC::InitOutput<float>(softmaxSumGm[tmpBlockIdx * softmaxMaxSingleCoreDataSize],
                                   tilingData->emptyInputTilingData.softmaxMaxSingleCoreDataSize,
                                   static_cast<float>(0.0));
    } else {
        AscendC::InitOutput<float>(softmaxMaxGm[softmaxMaxFormerNum * softmaxMaxSingleCoreDataSize +
                                                (tmpBlockIdx - softmaxMaxFormerNum) * softmaxMaxTailCoreDataSize],
                                   tilingData->emptyInputTilingData.softmaxMaxTailCoreDataSize,
                                   static_cast<float>(0.0));
        AscendC::InitOutput<float>(softmaxSumGm[softmaxMaxFormerNum * softmaxMaxSingleCoreDataSize +
                                                (tmpBlockIdx - softmaxMaxFormerNum) * softmaxMaxTailCoreDataSize],
                                   tilingData->emptyInputTilingData.softmaxMaxTailCoreDataSize,
                                   static_cast<float>(0.0));
    }
}

template <typename INPUT_T> __aicore__ inline void FlashAttentionScoreEmptyTensor<INPUT_T>::Process()
{
    ComputeEachCore();
}

#endif // FLASH_ATTENTION_SCORE_EMPTY_TENSOR_H
