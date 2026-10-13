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
 * \file layer_norm_v4_common.h
 * \brief
 */

#ifndef LAYER_NORM_V4_COMMON
#define LAYER_NORM_V4_COMMON

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace LayerNormV4 {
using namespace AscendC;
static constexpr int32_t FLOAT16_SIZE = 2;
static constexpr int32_t NUM_TWO = 2;

struct DataCopyStruct {
uint32_t blockCount = 0;
uint32_t blockLen = 0;
uint32_t srcStride = 0;
uint32_t dstStride = 0;
bool isPad = false;
uint8_t leftPad = 0;
uint8_t rightPad = 0;
};

template <typename T>
__aicore__ inline void DataCopyInContiguous(LocalTensor<T> dstTensor, GlobalTensor<T> srcTensor, DataCopyStruct dataCopyStruct, uint32_t tileLength) {
    DataCopyExtParams dataCopyParams;
    DataCopyPadExtParams<T> padParams{dataCopyStruct.isPad, dataCopyStruct.leftPad, dataCopyStruct.rightPad, 0};
    dataCopyParams.blockLen = dataCopyStruct.blockLen;
    dataCopyParams.blockCount = dataCopyStruct.blockCount;
    dataCopyParams.srcStride = dataCopyStruct.srcStride;
    dataCopyParams.dstStride = dataCopyStruct.dstStride;
    DataCopyPad(dstTensor[(sizeof(T) == FLOAT16_SIZE) * tileLength],
        srcTensor,
        dataCopyParams,
        padParams);
}


template <typename T>
__aicore__ inline void DataCopyOutContiguous(GlobalTensor<T> dstTensor, LocalTensor<T> srcTensor, DataCopyStruct dataCopyStruct) {
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dataCopyStruct.blockCount;
    dataCopyParams.blockLen = dataCopyStruct.blockLen;
    dataCopyParams.srcStride = dataCopyStruct.srcStride;
    dataCopyParams.dstStride = dataCopyStruct.dstStride;
    DataCopyPad(dstTensor, srcTensor, dataCopyParams);
}

template <HardEvent evt>
__aicore__ inline void SetEvtFlag() {
    event_t eventFlag = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<evt>(eventFlag);
    WaitFlag<evt>(eventFlag);
}

} // namespace LayerNormGradV3
#endif
