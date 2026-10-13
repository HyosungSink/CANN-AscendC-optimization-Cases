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
 * \file dequant_rope_quant_kvcache.h
 * \brief
 */
#ifndef DEQUANT_ROPE_QUANT_KVCACHE_DEQUANT_ROPE_QUANT_KVCACHE_H
#define DEQUANT_ROPE_QUANT_KVCACHE_DEQUANT_ROPE_QUANT_KVCACHE_H

namespace DequantRopeQuantKvcache {

using namespace AscendC;

constexpr int64_t BUFFER_NUM = 1;
constexpr int64_t DOUBLE_BUFFER_NUM = 1;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t MAX_UINT8 = 255;
constexpr int64_t ROPE_LAST_DIM_SPLIT = 2;
constexpr int64_t FP16_ONE_BLOCK_NUM = 16;
constexpr int64_t FP32_ONE_BLOCK_NUM = 8;
constexpr int64_t ONE_REPEAT_SIZE = 256;
constexpr int64_t FP16_ONE_REPEAT_NUM = 128;
constexpr int64_t FP32_ONE_REPEAT_NUM = 64;
constexpr int64_t INDICES_MAX__NUM = 512;
constexpr int64_t OFFSET_OVERRIDE = 2;

template <typename T>
__aicore__ inline T GetDiv(const T& value1, const T& value2)
{
    if (value2 == 0) {
        return value2;
    }
    return (value1) / value2;
}

template <typename T>
__aicore__ inline T GetCeilInt(const T& value1, const T& value2)
{
    if (value2 == 0) {
        return value2;
    }
    return (value1 + value2 - 1) / value2;
}

template <typename T>
__aicore__ inline T GetRem(const T& value1, const T& value2)
{
    if (value2 == 0) {
        return value2;
    }
    return value1 % value2;
}
template <class T>
struct queType {
    using type = float;
};
template <>
struct queType<half> {
    using type = half;
};

template <typename XTYPE, typename BIASTYPE, typename COSTYPE>
class RopeQuantKvcacheV2 {
public:
    __aicore__ inline RopeQuantKvcacheV2(const DequantRopeQuantKvcacheTilingData* tilingData)
    {
        this->cacheSeqlen = tilingData->cacheSeqlen;
        this->seqlen = tilingData->seqlen;
        this->qHeadNum = tilingData->qHeadNum;
        this->kvHeadNum = tilingData->kvHeadNum;
        this->hiddenSize = tilingData->hiddenSize;
        this->qHiddenSize = tilingData->qHiddenSize;
        this->kHiddenSize = tilingData->kHiddenSize;
        this->vHiddenSize = tilingData->vHiddenSize;
        this->realCoreNum = tilingData->realCoreNum;
        this->frontCoreNum = tilingData->frontCoreNum;
        this->blockFactor = tilingData->blockFactor;
        this->tailCoreBlockFactor = tilingData->tailCoreBlockFactor;
        this->hasQuantOffset = tilingData->hasQuantOffset;
        this->hiddenSizeFp32Align = tilingData->hiddenSizeFp32Align;
        this->hiddenSizeFp16Align = tilingData->hiddenSizeFp16Align;
        this->hiddenSizeInt8Align = tilingData->hiddenSizeInt8Align;
        this->OnceUBMaxS = tilingData->OnceUBMaxS;
        this->inputHiddenSize = this->qHiddenSize + this->kHiddenSize + this->vHiddenSize;
        this->isPA = tilingData->isPA;
        this->ifKVout = tilingData->ifKVout;
        this->hasBias = tilingData->hasBias;
        this->hasAS = tilingData->hasAS;
        this->batch = tilingData->batch;
    }

    __aicore__ inline void Init(
        GM_ADDR qkv, GM_ADDR cos, GM_ADDR sin, GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR indice, GM_ADDR weight_scale,
        GM_ADDR activation_scale, GM_ADDR bias, GM_ADDR quant_scale_k, GM_ADDR quant_scale_v, GM_ADDR quant_offset_k,
        GM_ADDR quant_offset_v, GM_ADDR q_out, GM_ADDR k_out, GM_ADDR v_out, GM_ADDR k_cache_out, GM_ADDR v_cache_out)
    {
        auto blockIdx = GetBlockIdx();
        if (blockIdx > this->frontCoreNum - 1) {
            sCoreOffset =
                this->blockFactor * this->frontCoreNum + (blockIdx - this->frontCoreNum) * this->tailCoreBlockFactor;
            coreCalcSNum = this->tailCoreBlockFactor;
        } else {
            sCoreOffset = this->blockFactor * blockIdx;
            coreCalcSNum = this->blockFactor;
        }
        bOffset = GetDiv(sCoreOffset, this->seqlen);
        coreCalcSLoop = GetCeilInt(coreCalcSNum, OnceUBMaxS);
        coreCalcSLastNum = coreCalcSNum - (coreCalcSLoop - 1) * OnceUBMaxS;
        uint64_t onceB = GetCeilInt(OnceUBMaxS, seqlen);
        uint64_t kvDataNum = OnceUBMaxS * this->kvHeadNum * this->hiddenSize; // 8*128
        uint64_t qkvBlockOffset = sCoreOffset * (this->qHiddenSize + this->kHiddenSize + this->vHiddenSize);
        uint64_t cossinBlockOffset = sCoreOffset * this->hiddenSize;
        uint64_t kvCacheBlockOffset = bOffset * this->cacheSeqlen * this->kvHeadNum * this->hiddenSize;
        uint64_t indiceOffset = bOffset;
        this->indiceUbNum = ((onceB + FP32_ONE_BLOCK_NUM - 1) / FP32_ONE_BLOCK_NUM) * FP32_ONE_BLOCK_NUM;
        uint64_t qDataNum = OnceUBMaxS * this->qHeadNum * this->hiddenSize;
        uint64_t qBlockOffset = blockIdx * qDataNum;
        inputGm.SetGlobalBuffer((__gm__ XTYPE*)qkv + qkvBlockOffset);
        cosGm.SetGlobalBuffer((__gm__ COSTYPE*)cos + cossinBlockOffset, this->blockFactor * this->hiddenSize);
        cosGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        sinGm.SetGlobalBuffer((__gm__ COSTYPE*)sin + cossinBlockOffset, this->blockFactor * this->hiddenSize);
        sinGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        quantScaleKGm.SetGlobalBuffer((__gm__ float*)quant_scale_k, this->kHiddenSize);
        quantScaleKGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        quantScaleVGm.SetGlobalBuffer((__gm__ float*)quant_scale_v, this->kHiddenSize);
        quantScaleVGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        if (this->hasQuantOffset == true) {
            quantOffsetKGm.SetGlobalBuffer((__gm__ float*)quant_offset_k, this->kHiddenSize);
            quantOffsetKGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            quantOffsetVGm.SetGlobalBuffer((__gm__ float*)quant_offset_v, this->kHiddenSize);
            quantOffsetVGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            pipe.InitBuffer(offsetKQueue, this->kHiddenSize * sizeof(float));
            pipe.InitBuffer(offsetVQueue, this->kHiddenSize * sizeof(float));
        }

        indiceGm.SetGlobalBuffer((__gm__ int32_t*)indice);
        indiceGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        outputGm.SetGlobalBuffer((__gm__ COSTYPE*)q_out + sCoreOffset * this->qHiddenSize);
        outputKGm.SetGlobalBuffer((__gm__ COSTYPE*)k_out + sCoreOffset * this->kHiddenSize);
        outputVGm.SetGlobalBuffer((__gm__ COSTYPE*)v_out + sCoreOffset * this->vHiddenSize);
        vCacheGm.SetGlobalBuffer((__gm__ int8_t*)v_cache_out);
        kCacheGm.SetGlobalBuffer((__gm__ int8_t*)k_cache_out);

        if constexpr (IsSameType<XTYPE, int32_t>::value) {
            if (this->hasBias == true) {
                biasGM.SetGlobalBuffer((__gm__ BIASTYPE*)bias);
            }
            if (this->hasAS == true) {
                asGM.SetGlobalBuffer((__gm__ float*)activation_scale + sCoreOffset);
            }
            wsGM.SetGlobalBuffer((__gm__ float*)weight_scale);
        }

        if constexpr (IsSameType<COSTYPE, half>::value && !IsSameType<XTYPE, int32_t>::value) {
            pipe.InitBuffer(qQueue, BUFFER_NUM, qDataNum * sizeof(half));
            pipe.InitBuffer(kQueue, BUFFER_NUM, kvDataNum * sizeof(half));
            pipe.InitBuffer(vQueue, BUFFER_NUM, kvDataNum * sizeof(half));
        } else {
            pipe.InitBuffer(qQueue, BUFFER_NUM, qDataNum * sizeof(float));
            pipe.InitBuffer(kQueue, BUFFER_NUM, kvDataNum * sizeof(float));
            pipe.InitBuffer(vQueue, BUFFER_NUM, kvDataNum * sizeof(float));
        }
        if constexpr (IsSameType<COSTYPE, bfloat16_t>::value) {
            cosUbOffset = OnceUBMaxS * this->hiddenSize;
        } else {
            cosUbOffset = 0;
        }
        if constexpr (IsSameType<XTYPE, bfloat16_t>::value) {
            qUbOffset = qDataNum;
            kUbOffset = kvDataNum;
        } else {
            qUbOffset = 0;
            kUbOffset = 0;
        }

        if constexpr (IsSameType<XTYPE, int32_t>::value) {
            if (this->hasBias == true) {
                pipe.InitBuffer(biasQueue, this->inputHiddenSize * sizeof(int32_t));
            }

            if (this->hasAS == true) {
                // as需要32字节对齐，ws和bias已对齐
                int64_t asUbNum = (OnceUBMaxS + FP32_ONE_BLOCK_NUM - 1) / FP32_ONE_BLOCK_NUM * FP32_ONE_BLOCK_NUM;
                pipe.InitBuffer(asQueue, BUFFER_NUM, asUbNum * sizeof(float));
                this->asOffset = 0;
            }
            pipe.InitBuffer(wsQueue, this->inputHiddenSize * sizeof(float));
        }

        pipe.InitBuffer(sinQueue, BUFFER_NUM, OnceUBMaxS * this->hiddenSize * sizeof(typename queType<COSTYPE>::type));
        pipe.InitBuffer(cosQueue, BUFFER_NUM, OnceUBMaxS * this->hiddenSize * sizeof(typename queType<COSTYPE>::type));
        pipe.InitBuffer(qOutQueue, BUFFER_NUM, qDataNum * sizeof(typename queType<COSTYPE>::type));

        pipe.InitBuffer(scaleKQueue, this->kHiddenSize * sizeof(float));
        pipe.InitBuffer(scaleVQueue, this->kHiddenSize * sizeof(float));

        pipe.InitBuffer(indiceQueue, BUFFER_NUM, this->indiceUbNum * sizeof(int32_t));

        pipe.InitBuffer(tmpBuf0, kvDataNum * sizeof(float));

        pipe.InitBuffer(kCacheOutQueue, BUFFER_NUM, kvDataNum * sizeof(int8_t));
        pipe.InitBuffer(vCacheOutQueue, BUFFER_NUM, kvDataNum * sizeof(int8_t));

        eventIdMTE2ToS = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
        eventIdVToMTE3V = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        eventIdVToMTE3K = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        eventIdMTE3ToMTE2K = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        eventIdMTE2ToMTE3V = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
        eventIdMTE3ToMTE2V = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        halfSize = hiddenSize / ROPE_LAST_DIM_SPLIT;
        oneRepeatNum = ONE_REPEAT_SIZE / sizeof(typename queType<COSTYPE>::type);
        repeatStride = hiddenSize * sizeof(typename queType<COSTYPE>::type) / BLOCK_SIZE;
        hRepeatTime = this->kvHeadNum * hiddenSize / FP32_ONE_REPEAT_NUM;
        repeatStrideF = this->kvHeadNum * hiddenSize * sizeof(float) / BLOCK_SIZE;
    }

    __aicore__ inline void Process()
    {
        quantScaleKUb = scaleKQueue.Get<float>();
        quantScaleVUb = scaleVQueue.Get<float>();

        DataCopy(quantScaleKUb, quantScaleKGm, this->kHiddenSize);
        DataCopy(quantScaleVUb, quantScaleVGm, this->kHiddenSize);

        if (this->hasQuantOffset == true) {
            offsetKUb = offsetKQueue.Get<float>();
            offsetVUb = offsetVQueue.Get<float>();
            DataCopy(offsetKUb, quantOffsetKGm, this->kHiddenSize);
            DataCopy(offsetVUb, quantOffsetVGm, this->kHiddenSize);
        }

        if constexpr (IsSameType<XTYPE, int32_t>::value) {
            wsUb = wsQueue.Get<float>();
            DataCopy(wsUb, wsGM, inputHiddenSize);
            qWsUb = wsUb;
            kWsUb = wsUb[qHiddenSize];
            vWsUb = wsUb[qHiddenSize + kHiddenSize];
            if (this->hasBias == true) {
                biasUb = biasQueue.Get<BIASTYPE>();
                if constexpr (IsSameType<BIASTYPE, bfloat16_t>::value || IsSameType<BIASTYPE, half>::value) {
                    biasUbOffset = inputHiddenSize;
                } else {
                    biasUbOffset = 0;
                }

                DataCopy(biasUb[biasUbOffset], biasGM, inputHiddenSize);
            }
        }

        PipeBarrier<PIPE_ALL>();

        if (IsSameType<XTYPE, int32_t>::value && IsSameType<BIASTYPE, float>::value == false &&
            IsSameType<BIASTYPE, int32_t>::value == false && this->hasBias == true) {
            LocalTensor<float> biasInUb = biasUb.template ReinterpretCast<float>();
            Cast(biasInUb, biasUb[biasUbOffset], RoundMode::CAST_NONE, inputHiddenSize);
        }

        if (IsSameType<XTYPE, int32_t>::value && this->hasBias == true) {
            qBiasUb = biasUb;
            if constexpr (IsSameType<BIASTYPE, bfloat16_t>::value || IsSameType<BIASTYPE, half>::value) {
                kBiasUb = biasUb[qHiddenSize * OFFSET_OVERRIDE];
                vBiasUb = biasUb[(qHiddenSize + kHiddenSize) * OFFSET_OVERRIDE];
            } else {
                kBiasUb = biasUb[qHiddenSize];
                vBiasUb = biasUb[qHiddenSize + kHiddenSize];
            }
        }

        if (this->ifKVout == true) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2K);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2V);
        }

        dataCopyParamsK_ = {
            (uint16_t)OnceUBMaxS, static_cast<uint32_t>(this->kHiddenSize * sizeof(XTYPE)),
            static_cast<uint32_t>((this->vHiddenSize + this->qHiddenSize) * sizeof(XTYPE)), 0, 0};
        dataCopyParamsQ_ = {
            (uint16_t)OnceUBMaxS, static_cast<uint32_t>(this->qHiddenSize * sizeof(XTYPE)),
            static_cast<uint32_t>((this->kHiddenSize + this->vHiddenSize) * sizeof(XTYPE)), 0, 0};
        dataCopyParamsV_ = {
            (uint16_t)OnceUBMaxS, static_cast<uint32_t>(this->vHiddenSize * sizeof(XTYPE)),
            static_cast<uint32_t>((this->qHiddenSize + this->kHiddenSize) * sizeof(XTYPE)), 0, 0};
        if (!this->isPA) {
            dataCopyParamsIndice_ = {
                1, static_cast<uint32_t>(
                    AscendC::Std::min(OnceUBMaxS, this->batch - this->bOffset) * sizeof(int32_t)
                ),
                0, 0, 0};
        } else {
            dataCopyParamsIndice_ = {1, static_cast<uint32_t>(OnceUBMaxS * sizeof(int32_t)), 0, 0, 0};
        }

        for (uint32_t ubLoopIndex = 0; ubLoopIndex < coreCalcSLoop - 1; ubLoopIndex++) {
            computeQKV(OnceUBMaxS);
        }

        dataCopyParamsK_ = {
            (uint16_t)coreCalcSLastNum, static_cast<uint32_t>(this->kHiddenSize * sizeof(XTYPE)),
            static_cast<uint32_t>((this->vHiddenSize + this->qHiddenSize) * sizeof(XTYPE)), 0, 0};
        dataCopyParamsQ_ = {
            (uint16_t)coreCalcSLastNum, static_cast<uint32_t>(this->qHiddenSize * sizeof(XTYPE)),
            static_cast<uint32_t>((this->kHiddenSize + this->vHiddenSize) * sizeof(XTYPE)), 0, 0};
        dataCopyParamsV_ = {
            (uint16_t)coreCalcSLastNum, static_cast<uint32_t>(this->vHiddenSize * sizeof(XTYPE)),
            static_cast<uint32_t>((this->qHiddenSize + this->kHiddenSize) * sizeof(XTYPE)), 0, 0};
        if (!this->isPA) {
            dataCopyParamsIndice_ = {
                1, static_cast<uint32_t>(
                    AscendC::Std::min(coreCalcSLastNum, this->batch - this->bOffset) * sizeof(int32_t)
                ),
                0, 0, 0};
        } else {
            dataCopyParamsIndice_ = {1, static_cast<uint32_t>(coreCalcSLastNum * sizeof(int32_t)), 0, 0, 0};
        }
        computeQKV(coreCalcSLastNum);
        if (this->ifKVout == true) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2K);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2V);
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(eventIdMTE2ToS);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMTE3V);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMTE3K);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2K);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3V);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2V);
    }

private:
    __aicore__ inline void copyOutcache(
        int64_t bIndex, int64_t sNum, int64_t sIndex, int64_t sIndexUb, LocalTensor<int8_t> vOut,
        LocalTensor<int8_t> kOut)
    {
        int64_t index = indiceUb.GetValue(bIndex);
        if (this->isPA == true) {
            if (index >= 0) {
                DataCopy(
                    vCacheGm[index * this->kvHeadNum * this->hiddenSize],
                    vOut[sIndexUb * this->kvHeadNum * this->hiddenSize], sNum * this->kvHeadNum * this->hiddenSize);
                DataCopy(
                    kCacheGm[index * this->kvHeadNum * this->hiddenSize],
                    kOut[sIndexUb * this->kvHeadNum * this->hiddenSize], sNum * this->kvHeadNum * this->hiddenSize);
            }
        } else {
            DataCopy(
                vCacheGm
                    [((bOffset + bIndex) * this->cacheSeqlen + index + sIndex) * this->kvHeadNum * this->hiddenSize],
                vOut[sIndexUb * this->kvHeadNum * this->hiddenSize], sNum * this->kvHeadNum * this->hiddenSize);
            DataCopy(
                kCacheGm
                    [((bOffset + bIndex) * this->cacheSeqlen + index + sIndex) * this->kvHeadNum * this->hiddenSize],
                kOut[sIndexUb * this->kvHeadNum * this->hiddenSize], sNum * this->kvHeadNum * this->hiddenSize);
        }
    }

    __aicore__ inline void computeQKV(int64_t onceS)
    {
        int64_t cosOffset = sOffset * this->hiddenSize;

        LocalTensor<COSTYPE> cosInUb = cosQueue.AllocTensor<COSTYPE>();
        LocalTensor<COSTYPE> sinInUb = sinQueue.AllocTensor<COSTYPE>();

        DataCopy(cosInUb[cosUbOffset], cosGm[cosOffset], onceS * this->hiddenSize);
        DataCopy(sinInUb[cosUbOffset], sinGm[cosOffset], onceS * this->hiddenSize);

        sinQueue.EnQue(sinInUb);
        cosQueue.EnQue(cosInUb);
        sinInUb = sinQueue.DeQue<COSTYPE>();
        cosInUb = cosQueue.DeQue<COSTYPE>();

        indiceUb = indiceQueue.AllocTensor<int32_t>();
        bOffset = GetDiv(sCoreOffset, seqlen);

        DataCopyPad(indiceUb, indiceGm[bOffset], dataCopyParamsIndice_, padParamsIndice_);
        SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);

        if (IsSameType<XTYPE, int32_t>::value && this->hasAS == true) {
            DataCopyExtParams dataCopyParamsAs;
            DataCopyPadExtParams<float> padParamsAs;
            dataCopyParamsAs.blockCount = 1;
            dataCopyParamsAs.blockLen = onceS * sizeof(float);
            dataCopyParamsAs.srcStride = 0;
            dataCopyParamsAs.dstStride = 0;
            padParamsAs.isPad = false;
            padParamsAs.paddingValue = static_cast<float>(0);
            padParamsAs.leftPadding = 0;
            padParamsAs.rightPadding = 0;
            asUb = asQueue.AllocTensor<float>();
            DataCopyPad(asUb, asGM[asOffset], dataCopyParamsAs, padParamsAs);
        }

        LocalTensor<XTYPE> qCopyInUb = qQueue.AllocTensor<XTYPE>();
        DataCopyPad(qCopyInUb[qUbOffset], inputGm[sOffset * inputHiddenSize], dataCopyParamsQ_, padParamsQ_);

        if (this->ifKVout == true) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2K);
        }

        LocalTensor<XTYPE> kCopyInUb = kQueue.AllocTensor<XTYPE>();
        DataCopyPad(
            kCopyInUb[kUbOffset], inputGm[sOffset * inputHiddenSize + qHiddenSize], dataCopyParamsK_, padParamsK_);

        kQueue.EnQue(kCopyInUb);
        qQueue.EnQue(qCopyInUb);

        LocalTensor<typename queType<COSTYPE>::type> cosUb =
            cosInUb.template ReinterpretCast<typename queType<COSTYPE>::type>();
        LocalTensor<typename queType<COSTYPE>::type> sinUb =
            sinInUb.template ReinterpretCast<typename queType<COSTYPE>::type>();
        if constexpr (IsSameType<COSTYPE, bfloat16_t>::value) {
            Cast(sinUb, sinInUb[cosUbOffset], RoundMode::CAST_NONE, onceS * this->hiddenSize);
            Cast(cosUb, cosInUb[cosUbOffset], RoundMode::CAST_NONE, onceS * this->hiddenSize);
            PipeBarrier<PIPE_V>();
        }

        if (this->hiddenSize > ROPE_LAST_DIM_SPLIT * oneRepeatNum || repeatStride > MAX_UINT8 || onceS > MAX_UINT8) {
            for (uint64_t r = 0; r < onceS; r++) {
                Muls(sinUb[r * hiddenSize], sinUb[r * hiddenSize], (typename queType<COSTYPE>::type) - 1.0, halfSize);
            }
        } else {
            Muls(
                sinUb, sinUb, (typename queType<COSTYPE>::type) - 1.0, halfSize, onceS,
                {1, 1, repeatStride, repeatStride});
        }
        int64_t firstS = GetRem(sCoreOffset, seqlen);
        int64_t firstStepSeq = seqlen - firstS < onceS ? seqlen - firstS : onceS;
        int64_t seqLoop = GetDiv((onceS - firstStepSeq), seqlen);
        int64_t lastStepSeq = GetRem((onceS - firstStepSeq), seqlen);
        int64_t sIndexUb = 0;
        PipeBarrier<PIPE_V>();

        kCopyInUb = kQueue.DeQue<XTYPE>();
        qCopyInUb = qQueue.DeQue<XTYPE>();

        if constexpr (IsSameType<XTYPE, int32_t>::value) {
            dequantUb(kCopyInUb[kUbOffset], kBiasUb, kWsUb, onceS, this->kHiddenSize);
            dequantUb(qCopyInUb[qUbOffset], qBiasUb, qWsUb, onceS, this->qHiddenSize);

            if constexpr (IsSameType<COSTYPE, half>::value) {
                LocalTensor<COSTYPE> kInUb = kCopyInUb.template ReinterpretCast<COSTYPE>();
                LocalTensor<COSTYPE> qInUb = qCopyInUb.template ReinterpretCast<COSTYPE>();
                LocalTensor<float> kInF32Ub = kCopyInUb.template ReinterpretCast<float>();
                LocalTensor<float> qInF32Ub = qCopyInUb.template ReinterpretCast<float>();
                PipeBarrier<PIPE_V>();
                Cast(kInUb, kInF32Ub, RoundMode::CAST_RINT, onceS * this->kHiddenSize);
                Cast(qInUb, qInF32Ub, RoundMode::CAST_RINT, onceS * this->qHiddenSize);
                PipeBarrier<PIPE_V>();
            }
        }
        LocalTensor<typename queType<COSTYPE>::type> kInUb =
            kCopyInUb.template ReinterpretCast<typename queType<COSTYPE>::type>();
        LocalTensor<typename queType<COSTYPE>::type> qInUb =
            qCopyInUb.template ReinterpretCast<typename queType<COSTYPE>::type>();
        if constexpr (IsSameType<XTYPE, bfloat16_t>::value) {
            Cast(kInUb, kCopyInUb[kUbOffset], RoundMode::CAST_NONE, onceS * this->kHiddenSize);
            Cast(qInUb, qCopyInUb[qUbOffset], RoundMode::CAST_NONE, onceS * this->qHiddenSize);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> tmpBufFP32 = tmpBuf0.Get<float>();
        LocalTensor<typename queType<COSTYPE>::type> qOutUb = qOutQueue.AllocTensor<typename queType<COSTYPE>::type>();

        // k rope
        LocalTensor<typename queType<COSTYPE>::type> kOutUbF16 =
            tmpBufFP32.template ReinterpretCast<typename queType<COSTYPE>::type>();

        if (halfSize <= oneRepeatNum && kvHeadNum == 1 && repeatStride < MAX_UINT8 && onceS < MAX_UINT8) {
            AscendC::SetMaskNorm();
            SetVectorMask<typename queType<COSTYPE>::type>(halfSize);

            Mul<typename queType<COSTYPE>::type, false>(
                kOutUbF16, kInUb[halfSize], sinUb, halfSize, onceS,
                {1, 1, 1, repeatStride, repeatStride, repeatStride});
            Mul<typename queType<COSTYPE>::type, false>(
                kOutUbF16[halfSize], kInUb, sinUb[halfSize], halfSize, onceS,
                {1, 1, 1, repeatStride, repeatStride, repeatStride});
            PipeBarrier<PIPE_V>();
            Mul<typename queType<COSTYPE>::type, false>(
                kInUb, kInUb, cosUb, halfSize, onceS, {1, 1, 1, repeatStride, repeatStride, repeatStride});
            Mul<typename queType<COSTYPE>::type, false>(
                kInUb[halfSize], kInUb[halfSize], cosUb[halfSize], halfSize, onceS,
                {1, 1, 1, repeatStride, repeatStride, repeatStride});
        } else {
            for (uint64_t loopS = 0; loopS < onceS; loopS++) {
                for (uint64_t loopH = 0; loopH < this->kvHeadNum; loopH++) {
                    Mul(kOutUbF16[(loopS * this->kvHeadNum + loopH) * this->hiddenSize],
                        kInUb[(loopS * this->kvHeadNum + loopH) * this->hiddenSize + halfSize],
                        sinUb[loopS * this->hiddenSize], halfSize);
                    Mul(kOutUbF16[(loopS * this->kvHeadNum + loopH) * this->hiddenSize + halfSize],
                        kInUb[(loopS * this->kvHeadNum + loopH) * this->hiddenSize],
                        sinUb[loopS * this->hiddenSize + halfSize], halfSize);
                }
            }
            PipeBarrier<PIPE_V>();
            for (uint64_t loopS = 0; loopS < onceS; loopS++) {
                for (uint64_t loopH = 0; loopH < this->kvHeadNum; loopH++) {
                    Mul(kInUb[(loopS * this->kvHeadNum + loopH) * this->hiddenSize],
                        kInUb[(loopS * this->kvHeadNum + loopH) * this->hiddenSize], cosUb[loopS * this->hiddenSize],
                        hiddenSize);
                }
            }
        }
        PipeBarrier<PIPE_V>();

        if constexpr (IsSameType<typename queType<COSTYPE>::type, float>::value) {
            Add(tmpBufFP32, kOutUbF16, kInUb, onceS * this->kHiddenSize);
        } else {
            Add(kInUb, kOutUbF16, kInUb, onceS * this->kHiddenSize);
            PipeBarrier<PIPE_V>();
            Cast(tmpBufFP32, kInUb, RoundMode::CAST_NONE, onceS * this->kHiddenSize);
        }

        if (this->ifKVout == true) {
            auto kRealOut = kInUb.template ReinterpretCast<COSTYPE>();
            if constexpr (IsSameType<COSTYPE, bfloat16_t>::value) {
                PipeBarrier<PIPE_V>();
                Cast(kRealOut, tmpBufFP32, RoundMode::CAST_RINT, onceS * this->kHiddenSize);
            }
            SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3K);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3K);
            DataCopy(outputKGm[sOffset * this->kHiddenSize], kRealOut, onceS * this->kHiddenSize);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2K);
        }
        // k quant

        kQueue.FreeTensor(kInUb);

        PipeBarrier<PIPE_V>();

        if (onceS * this->kvHeadNum < MAX_UINT8 && repeatStrideF < MAX_UINT8) {
            AscendC::SetMaskNorm();
            SetVectorMask<float>(FP32_ONE_REPEAT_NUM);
            for (uint32_t r = 0; r < hRepeatTime; r++) {
                Div<float, false>(
                    tmpBufFP32[r * FP32_ONE_REPEAT_NUM], tmpBufFP32[r * FP32_ONE_REPEAT_NUM],
                    quantScaleKUb[r * FP32_ONE_REPEAT_NUM], FP32_ONE_REPEAT_NUM, onceS,
                    {1, 1, 1, repeatStrideF, repeatStrideF, 0});
            }
            PipeBarrier<PIPE_V>();
            if (this->hasQuantOffset == true) {
                for (uint32_t r = 0; r < hRepeatTime; r++) {
                    Add<float, false>(
                        tmpBufFP32[r * FP32_ONE_REPEAT_NUM], tmpBufFP32[r * FP32_ONE_REPEAT_NUM],
                        offsetKUb[r * FP32_ONE_REPEAT_NUM], FP32_ONE_REPEAT_NUM, onceS,
                        {1, 1, 1, repeatStrideF, repeatStrideF, 0});
                }
            }
        } else {
            for (uint32_t r = 0; r < onceS; r++) {
                Div(tmpBufFP32[r * this->hiddenSize * this->kvHeadNum],
                    tmpBufFP32[r * this->hiddenSize * this->kvHeadNum], quantScaleKUb,
                    this->hiddenSize * this->kvHeadNum);
            }
            PipeBarrier<PIPE_V>();
            if (this->hasQuantOffset == true) {
                for (uint32_t r = 0; r < onceS; r++) {
                    Add(tmpBufFP32[r * this->hiddenSize * this->kvHeadNum],
                        tmpBufFP32[r * this->hiddenSize * this->kvHeadNum], offsetKUb,
                        this->hiddenSize * this->kvHeadNum);
                }
            }
        }

        PipeBarrier<PIPE_V>();
        LocalTensor<half> kOutF16 = tmpBufFP32.template ReinterpretCast<half>();
        LocalTensor<int8_t> kOutUbS8 = kCacheOutQueue.AllocTensor<int8_t>();

        LocalTensor<int16_t> kOutUbS16 = tmpBufFP32.template ReinterpretCast<int16_t>();
        Cast(kOutUbS16, tmpBufFP32, RoundMode::CAST_RINT, onceS * this->kHiddenSize);
        PipeBarrier<PIPE_V>();
        Cast(kOutF16, kOutUbS16, RoundMode::CAST_NONE, onceS * this->kHiddenSize);
        PipeBarrier<PIPE_V>();
        Cast(kOutUbS8, kOutF16, RoundMode::CAST_NONE, onceS * this->kHiddenSize);
        PipeBarrier<PIPE_V>();
        kCacheOutQueue.EnQue(kOutUbS8);
        LocalTensor<int8_t> kOut = kCacheOutQueue.DeQue<int8_t>();

        if (this->ifKVout == true) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2V);
        }
        LocalTensor<XTYPE> vCopyInUb = vQueue.AllocTensor<XTYPE>();
        DataCopyPad(
            vCopyInUb, inputGm[sOffset * inputHiddenSize + this->qHiddenSize + this->kHiddenSize], dataCopyParamsV_,
            padParamsV_);
        if (this->ifKVout == true) {
            SetFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3V);
        }
        vQueue.EnQue(vCopyInUb);

        // q rope

        if (halfSize <= oneRepeatNum && repeatStride < MAX_UINT8 && this->qHeadNum < MAX_UINT8) {
            AscendC::SetMaskNorm();
            SetVectorMask<typename queType<COSTYPE>::type>(halfSize);
            for (uint64_t r = 0; r < onceS; r++) {
                Mul<typename queType<COSTYPE>::type, false>(
                    qOutUb[r * this->qHiddenSize], qInUb[r * this->qHiddenSize + halfSize], sinUb[r * this->hiddenSize],
                    halfSize, this->qHeadNum, {1, 1, 1, repeatStride, repeatStride, 0});
                Mul<typename queType<COSTYPE>::type, false>(
                    qOutUb[r * this->qHiddenSize + halfSize], qInUb[r * this->qHiddenSize],
                    sinUb[r * this->hiddenSize + halfSize], halfSize, this->qHeadNum,
                    {1, 1, 1, repeatStride, repeatStride, 0});
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < onceS; r++) {
                Mul<typename queType<COSTYPE>::type, false>(
                    qInUb[r * this->qHiddenSize], qInUb[r * this->qHiddenSize], cosUb[r * this->hiddenSize], halfSize,
                    this->qHeadNum, {1, 1, 1, repeatStride, repeatStride, 0});
                Mul<typename queType<COSTYPE>::type, false>(
                    qInUb[r * this->qHiddenSize + halfSize], qInUb[r * this->qHiddenSize + halfSize],
                    cosUb[r * this->hiddenSize + halfSize], halfSize, this->qHeadNum,
                    {1, 1, 1, repeatStride, repeatStride, 0});
            }
        } else {
            for (uint64_t loopS = 0; loopS < onceS; loopS++) {
                for (uint64_t loopH = 0; loopH < this->qHeadNum; loopH++) {
                    Mul(qOutUb[(loopS * this->qHeadNum + loopH) * this->hiddenSize],
                        qInUb[(loopS * this->qHeadNum + loopH) * this->hiddenSize + halfSize],
                        sinUb[loopS * this->hiddenSize], halfSize);
                    Mul(qOutUb[(loopS * this->qHeadNum + loopH) * this->hiddenSize + halfSize],
                        qInUb[(loopS * this->qHeadNum + loopH) * this->hiddenSize],
                        sinUb[loopS * this->hiddenSize + halfSize], halfSize);
                }
            }
            PipeBarrier<PIPE_V>();
            for (uint64_t loopS = 0; loopS < onceS; loopS++) {
                for (uint64_t loopH = 0; loopH < this->qHeadNum; loopH++) {
                    Mul(qInUb[(loopS * this->qHeadNum + loopH) * this->hiddenSize],
                        qInUb[(loopS * this->qHeadNum + loopH) * this->hiddenSize], cosUb[loopS * this->hiddenSize],
                        hiddenSize);
                }
            }
        }
        sinQueue.FreeTensor(sinUb);
        cosQueue.FreeTensor(cosUb);
        PipeBarrier<PIPE_V>();
        Add(qOutUb, qOutUb, qInUb, onceS * this->qHiddenSize);
        LocalTensor<COSTYPE> qRealOut = qOutUb.template ReinterpretCast<COSTYPE>();
        if constexpr (IsSameType<COSTYPE, bfloat16_t>::value) {
            PipeBarrier<PIPE_V>();
            Cast(qRealOut, qOutUb, RoundMode::CAST_RINT, onceS * this->qHiddenSize);
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();
        qQueue.FreeTensor(qInUb);
        qOutQueue.EnQue(qOutUb);
        auto qOut = qOutQueue.DeQue<typename queType<COSTYPE>::type>();
        qRealOut = qOut.template ReinterpretCast<COSTYPE>();

        DataCopy(outputGm[sOffset * this->qHiddenSize], qRealOut, onceS * this->qHiddenSize);
        qOutQueue.FreeTensor(qOut);

        // v quant
        LocalTensor<int8_t> vOutUbS8 = vCacheOutQueue.AllocTensor<int8_t>();
        vCopyInUb = vQueue.DeQue<XTYPE>();
        if constexpr (IsSameType<XTYPE, int32_t>::value) {
            dequantUb(vCopyInUb, vBiasUb, vWsUb, onceS, this->vHiddenSize);
        }
        if (IsSameType<XTYPE, int32_t>::value && this->hasAS == true) {
            asQueue.FreeTensor(asUb);
            this->asOffset += onceS;
        }
        if constexpr (IsSameType<XTYPE, int32_t>::value) {
            PipeBarrier<PIPE_V>();
            LocalTensor<float> vFloatUb = vCopyInUb.template ReinterpretCast<float>();
            DataCopy(tmpBufFP32, vFloatUb, onceS * this->vHiddenSize);
            PipeBarrier<PIPE_V>();
        }
        if (this->ifKVout == true) {
            WaitFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3V);
            LocalTensor<COSTYPE> vCopyOutUb = vCopyInUb.template ReinterpretCast<COSTYPE>();

            if constexpr (IsSameType<XTYPE, int32_t>::value) {
                LocalTensor<float> vFloatUb = vCopyInUb.template ReinterpretCast<float>();
                Cast(vCopyOutUb, vFloatUb, RoundMode::CAST_RINT, onceS * this->vHiddenSize);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3V);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3V);
            }
            DataCopy(outputVGm[sOffset * this->vHiddenSize], vCopyOutUb, onceS * this->vHiddenSize);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2V);
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<COSTYPE> vCopyOutUb = vCopyInUb.template ReinterpretCast<COSTYPE>();
        if constexpr (!IsSameType<XTYPE, int32_t>::value) {
            Cast(tmpBufFP32, vCopyOutUb, RoundMode::CAST_NONE, onceS * this->vHiddenSize);
        }

        vQueue.FreeTensor(vCopyInUb);
        PipeBarrier<PIPE_V>();

        if (onceS * this->kvHeadNum < MAX_UINT8 && repeatStrideF < MAX_UINT8) {
            AscendC::SetMaskNorm();
            SetVectorMask<float>(FP32_ONE_REPEAT_NUM);
            for (uint32_t r = 0; r < hRepeatTime; r++) {
                Div<float, false>(
                    tmpBufFP32[r * FP32_ONE_REPEAT_NUM], tmpBufFP32[r * FP32_ONE_REPEAT_NUM],
                    quantScaleVUb[r * FP32_ONE_REPEAT_NUM], FP32_ONE_REPEAT_NUM, onceS,
                    {1, 1, 1, repeatStrideF, repeatStrideF, 0});
            }
            PipeBarrier<PIPE_V>();
            if (this->hasQuantOffset == true) {
                for (uint32_t r = 0; r < hRepeatTime; r++) {
                    Add<float, false>(
                        tmpBufFP32[r * FP32_ONE_REPEAT_NUM], tmpBufFP32[r * FP32_ONE_REPEAT_NUM],
                        offsetVUb[r * FP32_ONE_REPEAT_NUM], FP32_ONE_REPEAT_NUM, onceS,
                        {1, 1, 1, repeatStrideF, repeatStrideF, 0});
                }
            }
        } else {
            for (uint32_t r = 0; r < onceS; r++) {
                Div(tmpBufFP32[r * this->hiddenSize * this->kvHeadNum],
                    tmpBufFP32[r * this->hiddenSize * this->kvHeadNum], quantScaleVUb,
                    this->hiddenSize * this->kvHeadNum);
            }
            PipeBarrier<PIPE_V>();
            if (this->hasQuantOffset == true) {
                for (uint32_t r = 0; r < onceS; r++) {
                    Add(tmpBufFP32[r * this->hiddenSize * this->kvHeadNum],
                        tmpBufFP32[r * this->hiddenSize * this->kvHeadNum], offsetVUb,
                        this->hiddenSize * this->kvHeadNum);
                }
            }
        }

        PipeBarrier<PIPE_V>();

        LocalTensor<int16_t> vOutUbS16 = tmpBufFP32.template ReinterpretCast<int16_t>();
        Cast(vOutUbS16, tmpBufFP32, RoundMode::CAST_RINT, onceS * this->kvHeadNum * this->hiddenSize);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> vOutUbF16 = tmpBufFP32.template ReinterpretCast<half>();
        Cast(vOutUbF16, vOutUbS16, RoundMode::CAST_NONE, onceS * this->kvHeadNum * this->hiddenSize);
        PipeBarrier<PIPE_V>();
        Cast(vOutUbS8, vOutUbF16, RoundMode::CAST_NONE, onceS * this->kvHeadNum * this->hiddenSize);
        PipeBarrier<PIPE_V>();
        vCacheOutQueue.EnQue(vOutUbS8);
        LocalTensor<int8_t> vOut = vCacheOutQueue.DeQue<int8_t>();

        PipeBarrier<PIPE_V>();

        WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);

        copyOutcache(0, firstStepSeq, firstS, sIndexUb, vOut, kOut);
        sIndexUb += firstStepSeq;

        for (uint32_t seqLoopIndex = 1; seqLoopIndex < seqLoop + 1; seqLoopIndex++) {
            copyOutcache(seqLoopIndex, seqlen, 0, sIndexUb, vOut, kOut);
            sIndexUb += seqlen;
        }
        if (lastStepSeq != 0) {
            copyOutcache(seqLoop + 1, lastStepSeq, 0, sIndexUb, vOut, kOut);
        }

        indiceQueue.FreeTensor(indiceUb);
        kCacheOutQueue.FreeTensor(kOut);
        vCacheOutQueue.FreeTensor(vOut);

        sCoreOffset += onceS;
        sOffset += onceS;
    }

    __aicore__ inline void dequantUb(
        LocalTensor<XTYPE> copyInUb, LocalTensor<BIASTYPE> curBiasUb, LocalTensor<float> curWsUb, int64_t onceS,
        int64_t calcSize)
    {
        int64_t offset = 0;
        PipeBarrier<PIPE_ALL>();
        if (this->hasBias == true) {
            if constexpr (IsSameType<BIASTYPE, int32_t>::value) {
                for (uint32_t r = 0; r < onceS; r++) {
                    offset = calcSize * r;
                    Add(copyInUb[offset], copyInUb[offset], curBiasUb, calcSize);
                }

                PipeBarrier<PIPE_V>();

                LocalTensor<float> copyInFloatUb = copyInUb.template ReinterpretCast<float>();
                Cast(copyInFloatUb, copyInUb, RoundMode::CAST_RINT, onceS * calcSize);
                PipeBarrier<PIPE_V>();

                for (uint32_t r = 0; r < onceS; r++) {
                    offset = calcSize * r;
                    Mul(copyInFloatUb[offset], copyInFloatUb[offset], curWsUb, calcSize);
                }
                PipeBarrier<PIPE_V>();

                if (this->hasAS == true) {
                    for (uint32_t r = 0; r < onceS; r++) {
                        offset = calcSize * r;
                        Muls(copyInFloatUb[offset], copyInFloatUb[offset], asUb.GetValue(r), calcSize);
                    }
                }
                PipeBarrier<PIPE_V>();
            } else {
                LocalTensor<float> copyInFloatUb = copyInUb.template ReinterpretCast<float>();
                Cast(copyInFloatUb, copyInUb, RoundMode::CAST_RINT, onceS * calcSize);
                PipeBarrier<PIPE_V>();

                for (uint32_t r = 0; r < onceS; r++) {
                    offset = calcSize * r;
                    Mul(copyInFloatUb[offset], copyInFloatUb[offset], curWsUb, calcSize);
                }
                PipeBarrier<PIPE_V>();

                if (this->hasAS == true) {
                    for (uint32_t r = 0; r < onceS; r++) {
                        offset = calcSize * r;
                        Muls(copyInFloatUb[offset], copyInFloatUb[offset], asUb.GetValue(r), calcSize);
                    }
                }
                PipeBarrier<PIPE_V>();
                LocalTensor<float> curBiasFloatUb = curBiasUb.template ReinterpretCast<float>();
                if (this->hasBias == true) {
                    for (uint32_t r = 0; r < onceS; r++) {
                        offset = calcSize * r;
                        Add(copyInFloatUb[offset], copyInFloatUb[offset], curBiasFloatUb, calcSize);
                    }
                }
                PipeBarrier<PIPE_V>();
            }
        } else {
            LocalTensor<float> copyInFloatUb = copyInUb.template ReinterpretCast<float>();

            Cast(copyInFloatUb, copyInUb, RoundMode::CAST_RINT, onceS * calcSize);

            PipeBarrier<PIPE_V>();

            for (uint32_t r = 0; r < onceS; r++) {
                offset = calcSize * r;
                Mul(copyInFloatUb[offset], copyInFloatUb[offset], curWsUb, calcSize);
            }
            PipeBarrier<PIPE_V>();

            if (this->hasAS == true) {
                for (uint32_t r = 0; r < onceS; r++) {
                    offset = calcSize * r;
                    Muls(copyInFloatUb[offset], copyInFloatUb[offset], asUb.GetValue(r), calcSize);
                }
            }
            PipeBarrier<PIPE_V>();
        }
    }

    /* global memory address */
    GlobalTensor<XTYPE> inputGm;

    GlobalTensor<COSTYPE> cosGm;
    GlobalTensor<COSTYPE> sinGm;

    GlobalTensor<int32_t> indiceGm;

    GlobalTensor<BIASTYPE> biasGM;
    GlobalTensor<float> asGM;
    GlobalTensor<float> wsGM;

    GlobalTensor<float> quantScaleKGm;
    GlobalTensor<float> quantScaleVGm;
    GlobalTensor<float> quantOffsetKGm;
    GlobalTensor<float> quantOffsetVGm;

    GlobalTensor<COSTYPE> outputGm;
    GlobalTensor<COSTYPE> outputKGm;
    GlobalTensor<COSTYPE> outputVGm;

    GlobalTensor<int8_t> kCacheGm;
    GlobalTensor<int8_t> vCacheGm;
    LocalTensor<float> quantScaleKUb;
    LocalTensor<float> quantScaleVUb;

    LocalTensor<float> offsetKUb;
    LocalTensor<float> offsetVUb;
    LocalTensor<int32_t> indiceUb;
    LocalTensor<BIASTYPE> biasUb;
    LocalTensor<BIASTYPE> qBiasUb;
    LocalTensor<BIASTYPE> kBiasUb;
    LocalTensor<BIASTYPE> vBiasUb;
    LocalTensor<float> asUb;
    LocalTensor<float> wsUb;
    LocalTensor<float> qWsUb;
    LocalTensor<float> kWsUb;
    LocalTensor<float> vWsUb;

    /* variable */
    uint64_t halfSize;
    uint8_t repeatStride;
    int64_t cacheSeqlen;
    int64_t seqlen;
    int64_t frontCoreNum;
    int64_t qHeadNum;
    int64_t kvHeadNum;
    int64_t inputHiddenSize;
    int64_t hiddenSize;
    int64_t qHiddenSize;
    int64_t kHiddenSize;
    int64_t vHiddenSize;
    int64_t realCoreNum;
    int64_t blockFactor;
    int64_t tailCoreBlockFactor;
    int64_t hasQuantOffset;
    int64_t indiceUbNum;
    int64_t hiddenSizeFp32Align;
    int64_t hiddenSizeFp16Align;
    int64_t hiddenSizeInt8Align;
    int64_t OnceUBMaxS;
    int64_t sOffset = 0;
    int64_t bOffset;
    int64_t coreCalcSNum;
    int64_t coreCalcSLastNum;
    int64_t coreCalcSLoop;
    int64_t LastLoopS;
    int64_t sCoreOffset;
    int64_t oneRepeatNum;
    int64_t qUbOffset;
    int64_t kUbOffset;
    int64_t cosUbOffset;
    int64_t asOffset;
    int64_t biasUbOffset;
    int64_t hRepeatTime;
    uint8_t repeatStrideF;
    int64_t ifKVout;
    int64_t isPA;
    int64_t hasBias;
    int64_t hasAS;
    int64_t batch;

    int64_t indicesArray[INDICES_MAX__NUM];

    event_t eventIdMTE2ToS;
    event_t eventIdVToMTE3K;
    event_t eventIdMTE3ToMTE2K;
    event_t eventIdMTE2ToMTE3V;
    event_t eventIdMTE3ToMTE2V;
    event_t eventIdVToMTE3V;
    event_t eventIdMTE3ToVV;
    /* ascendc variable */
    TPipe pipe;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> qQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> kQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> vQueue;

    TQue<QuePosition::VECIN, 1> sinQueue;
    TQue<QuePosition::VECIN, 1> cosQueue;
    TBuf<TPosition::VECCALC> scaleKQueue;
    TBuf<TPosition::VECCALC> scaleVQueue;
    TBuf<TPosition::VECCALC> offsetKQueue;
    TBuf<TPosition::VECCALC> offsetVQueue;
    TBuf<TPosition::VECCALC> biasQueue;
    TBuf<TPosition::VECCALC> wsQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> asQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> indiceQueue;

    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> qOutQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> kCacheOutQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> vCacheOutQueue;

    TBuf<TPosition::VECCALC> tmpBuf0;

    DataCopyExtParams dataCopyParamsIndice_;
    DataCopyPadExtParams<int32_t> padParamsIndice_{false, static_cast<int32_t>(0), 0, 0};

    DataCopyExtParams dataCopyParamsQ_;
    DataCopyPadExtParams<XTYPE> padParamsQ_;

    DataCopyExtParams dataCopyParamsK_;
    DataCopyPadExtParams<XTYPE> padParamsK_;

    DataCopyExtParams dataCopyParamsV_;
    DataCopyPadExtParams<XTYPE> padParamsV_;
};
} // namespace DequantRopeQuantKvcache
#endif // DEQUANT_ROPE_QUANT_KVCACHE_DEQUANT_ROPE_QUANT_KVCACHE_H