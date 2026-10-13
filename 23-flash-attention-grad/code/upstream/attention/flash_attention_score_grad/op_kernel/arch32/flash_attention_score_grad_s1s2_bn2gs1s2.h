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
 * \file flash_attention_score_grad_s1s2_bn2gs1s2.h
 * \brief
 */

#ifndef _FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2GS1S2_H_
#define _FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2GS1S2_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "pse.h"
#include "dropmask.h"

using namespace matmul;

struct EvenvIdList
{
    event_t structVWaitMte2Ping;
    event_t structVWaitMte2Pong;
    event_t structMte2WaitMte3Ping;
    event_t structMte2WaitMte3Pong;
    event_t structMte3WaitVPing;
    event_t structMte3WaitVPong;
    event_t structMte2WaitVPing;
    event_t structMte2WaitVPong;
};

constexpr inline MatmulConfig NORM_DISABLE_INIT = {true,  false, false, 0,     0,     0,     false, false,
                                                   false, false, 0,     0,     0,     0,     0,     0,
                                                   0,     0,     true,  false, false, false, false, false};

__aicore__ inline void DataCopyOut(const __gm__ void *gm, const LocalTensor<int8_t> &co1Local,
                                   const void *dataCopyOutParams, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    const DataCopyOutParams *param = reinterpret_cast<const DataCopyOutParams *>(dataCopyOutParams);
    uint64_t dstStride = dataPtr * 16 / 8 - param->burstLen;
    FixpipeParams<float> fixpipeParams(param->cBurstNum, param->burstLen, param->srcStride,
                                       static_cast<uint32_t>(dstStride));

    if (param->enUnitFlag) {
        fixpipeParams.unitFlag = 3;
    }
    LocalTensor<float> tmpLocal = co1Local.template ReinterpretCast<float>();
    GlobalTensor<float> tmpGm;
    tmpGm.SetGlobalBuffer((__gm__ float *)(gm));
    Fixpipe(tmpGm, tmpLocal, fixpipeParams);
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK = 0, const uint32_t IS_PSE = 1,
          const uint32_t IS_DROP = 1, const CubeFormat MM_OUT_FORMAT = CubeFormat::ND, const uint32_t INPUT_LAYOUT = 0,
          const CubeFormat MM2_OUT_FORMAT = CubeFormat::NZ, const uint32_t TND_S1_PP = 0, const uint32_t HAS_ROPE = 0>
class FlashAttentionScoreGradS1s2Bn2gs1s2 {
public:
    __aicore__ inline FlashAttentionScoreGradS1s2Bn2gs1s2(){};

    __aicore__ inline void Init(__gm__ uint8_t *key, __gm__ uint8_t *keyRope, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query, __gm__ uint8_t *queryRope,
                                __gm__ uint8_t *pse_shift, __gm__ uint8_t *drop_mask, __gm__ uint8_t *atten_mask,
                                __gm__ uint8_t *forward_res, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,
                                __gm__ uint8_t *prefixN, __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen,
                                __gm__ uint8_t *dq, __gm__ uint8_t *dqRope, __gm__ uint8_t *dk, __gm__ uint8_t *dkRope, __gm__ uint8_t *dv, __gm__ uint8_t *dpse,
                                __gm__ uint8_t *workspace,
                                const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *__restrict ordTilingData,
                                TPipe *pipe_in);
    __aicore__ inline void CopyInSoftMaxGrad(LocalTensor<T1> &dstTensor, LocalTensor<T1> &dstTensor2,
                                             int64_t softmaxGradOffset, uint32_t s1Extend, uint32_t dExtend,
                                             uint32_t dExtendAlign);
    __aicore__ inline void CalcSoftMaxGrad(LocalTensor<float> &sfmgClc3, int64_t aTensorOffset, uint32_t s1Extend);
    __aicore__ inline void CalcSoftMaxGradPingPong(int64_t outerPingPong, int64_t aTensorOffset, uint32_t s1Extend,
                                                   EvenvIdList &eventIdList);
    __aicore__ inline void CopyInSoftMax(LocalTensor<float> &dstTensor, uint32_t s1Extend, uint32_t softMaxOffset);
    __aicore__ inline void CalcSoftMax(LocalTensor<T2> &dstTensor, LocalTensor<float> &srcTensor, uint32_t s1Extend,
                                       uint32_t s2Extend, uint32_t s2ExtendAlign, const SoftMaxTiling &tiling);
    __aicore__ inline void CopyInAttenMaskBool(LocalTensor<uint8_t> &dstTensor, int64_t attenMaskOffset,
                                               uint32_t s1Extend, uint32_t s2Extend);
    __aicore__ inline void CalcAttenMaskBool(LocalTensor<T2> &dstTensor, LocalTensor<uint8_t> srcTensor,
                                             uint32_t s1Extend, uint32_t s2Extend, uint8_t maskType = 0);
    __aicore__ inline void CalcAttenMaskOffset(int64_t &attenMaskOffset, const int64_t delta, uint32_t s1VSize,
                                               uint32_t s2VSize);
    __aicore__ inline void CalcAttenBandMode(int64_t compressMode, int64_t causal_delta);
    __aicore__ inline void CalcAttenMaskOffsetForPrefixCompressMode(int64_t &attenMaskOffset, int64_t &attenMaskOffse2,
                                                                    const int64_t delta, uint32_t s1VSize,
                                                                    uint32_t s2VSize, uint32_t s2VBegin,
                                                                    bool &canSimplify);
    __aicore__ inline void CalcAttenMaskOffsetWithSparseMode(int64_t &attenMaskOffset, int64_t &attenMaskOffset2,
                                                             uint32_t s1VSize, uint32_t s2VSize, int64_t curS1Idx,
                                                             uint32_t s2VBegin, bool &canSimplify);
    __aicore__ inline void CalcAttenMaskOffsetWithSparseModeForUnpad(int64_t &attenMaskOffset,
                                                                     int64_t &attenMaskOffset2, uint32_t s1VSize,
                                                                     uint32_t s2VSize, int64_t curS1Idx,
                                                                     uint32_t s2VBegin, bool unpadUseBand,
                                                                     bool &canSimplify);
    __aicore__ inline void DropOutCopy(LocalTensor<uint8_t> &vecInDropBuffer, int64_t curS1Idx, int64_t s2VBegin);

    __aicore__ inline void Process();
    __aicore__ inline void UpdateToken(int64_t bIdx);
    __aicore__ inline bool IsCubeBlockNeedCompute(int64_t baseIndex);
    __aicore__ inline void InitIndex(int64_t index);
    __aicore__ inline void SubGrapA(int64_t curIdx, int64_t curS1Idx, int64_t curS2Idx, EvenvIdList &eventIdList);
    __aicore__ inline void SubGrapB(int64_t curIdx, int64_t curS1Idx, int64_t curS2Idx, EvenvIdList &eventIdList);
    __aicore__ inline void Compute(int64_t preIndex, int64_t nextIndex);
    __aicore__ inline void SyncALLCores();
    __aicore__ inline bool CheckIsValidBlock(int64_t baseIdx, int64_t s1oDimIdx, int64_t s2oCvDimIdx, int64_t curBIdx);
    __aicore__ inline void GetSeqQlenKvlenByBidx(int64_t bIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen);
    __aicore__ inline void LocalAllocEventID(EvenvIdList &eventIdList);
    __aicore__ inline void LocalReleaseEventID(EvenvIdList &eventIdList);

    using aType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
    using bType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
    using cType1 = MatmulType<TPosition::GM, MM_OUT_FORMAT, T2>;
    using biasType1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using aType2 = MatmulType<TPosition::GM, MM_OUT_FORMAT, T1, true>;
    using bType2 = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
    using cType2 = MatmulType<TPosition::GM, MM2_OUT_FORMAT, float>;
    using biasType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    Matmul<aType1, bType1, cType1, biasType1, NORM_DISABLE_INIT> mm1;
    using modeTypeMm = typename AscendC::Conditional<
        (MM2_OUT_FORMAT == CubeFormat::NZ),
        Matmul<aType2, bType2, cType2, biasType2, NORM_DISABLE_INIT, MatmulCallBackFunc<DataCopyOut>>,
        Matmul<aType2, bType2, cType2, biasType2, NORM_DISABLE_INIT>>::type;
    modeTypeMm mm3;

    using modeTypeMm4 = typename AscendC::Conditional<
        (MM2_OUT_FORMAT == CubeFormat::NZ),
        Matmul<aType2, bType2, cType2, biasType2, NORM_DISABLE_INIT, MatmulCallBackFunc<DataCopyOut>>,
        Matmul<aType2, bType2, cType2, biasType2, NORM_DISABLE_INIT>>::type;

    modeTypeMm4 mm4;

    __aicore__ inline void NZCopyIn(int64_t mmAddr, GlobalTensor<T2> &mmWspGm, LocalTensor<T2> &mmTensorCurr,
                                    uint32_t s1VecSize, uint32_t s2VecSize);
    __aicore__ inline void NZ2ND(LocalTensor<T2> &mmTensorCurr, LocalTensor<T2> &tmpTensor, uint32_t s1VecSize,
                                 uint32_t s2VecSize);
    __aicore__ inline void ND2NZ(LocalTensor<T1> &mmTensorCurr, LocalTensor<T1> &tmpTensor, uint32_t s1VecSize,
                                 uint32_t s2VecSize);

protected:
    TPipe *pipe;
    TBuf<> ubBuffer;
    TBuf<> tmpBuffer;
    TBuf<> vecClc3;

    uint32_t coreNum;
    int64_t cBlockIdx;
    const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *__restrict TilingData;

    // input
    GlobalTensor<T1> keyGm, valueGm, dxGm, queryGm, forwardResGm, pseGm;
    GlobalTensor<T1> keyRopeGm;
    GlobalTensor<T1> queryRopeGm;
    GlobalTensor<uint8_t> maskWorkSpaceGm, attenMaskU8Gm, dropMaskGm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm;

    // output
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    GlobalTensor<float> dqRopeWorkSpaceGm;
    GlobalTensor<float> dkRopeWorkSpaceGm;
    GlobalTensor<T1> dropWorkSpaceGm, mulWorkSpaceGm;

    // workspace
    GlobalTensor<T2> mm1WorkspaceGm;
    GlobalTensor<T2> mm2WorkspaceGm;

    __gm__ uint8_t *prefixN_addr;
    __gm__ uint8_t *actual_seq_qlen_addr;
    __gm__ uint8_t *actual_seq_kvlen_addr;

    // AscendC
    GlobalTensor<int32_t> syncGlobal;

    // matmal1/matmal2 result buffer
    GlobalTensor<float> matmalResultBuffer1;
    GlobalTensor<float> matmalResultBuffer2;

    GlobalTensor<half> pseAlibiGm;
    GlobalTensor<float> dvGm;
    __gm__ uint8_t *pseSlope;

    PseInfo pseInfo = {0};

    constexpr static uint32_t BNGSD = 0;
    constexpr static uint32_t SBNGD = 1;
    constexpr static uint32_t BSNGD = 2;
    constexpr static uint32_t TND = 3;
    constexpr static uint32_t ENABLE = 1;

    T2 mulsValue = -10000.0;

    // optional control
    float keepProb;
    int64_t s1Token;
    int64_t s2Token;
    int64_t actualCalcS1Token;
    int64_t actualCalcS2Token;
    uint32_t sparseMode;
    bool isSparse;

    // org shape info
    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t rope_d = 0;
    int64_t value_d;
    int64_t attenMaskDimS2;

    uint32_t baseMN;
    uint32_t cubeBaseMN;

    // split info
    int64_t s1Outer;
    uint32_t s1CvRatio;
    uint32_t s1CvInner;
    uint32_t s1CvTail;
    uint32_t s1CvExtend{0};

    int64_t s2Outer;
    uint32_t s2CvRatio;
    uint32_t s2Inner;
    uint32_t s2CvInner;
    uint32_t sfmgdOuter;
    uint32_t sfmgdInner;
    uint32_t sfmgdTail;
    uint32_t sfmgdTailAlign;
    bool dropBitMode;

    int64_t blockOuter;

    // sparse block info
    const int64_t *blockStarts;
    const int64_t *blockEnds;

    // buferinfo
    uint32_t matmalWorkspaceSize;

    // base info
    int64_t n2gs1os2o;
    int64_t gs1os2o;
    int64_t s1os2o;

    int64_t baseIdx{0};
    int64_t bDimIdx{0};
    int64_t n2DimIdx{0};
    int64_t gDimIdx{0};
    int64_t s1oDimIdx{0};
    int64_t s2oCvDimIdx{0};
    int64_t totalBaseIdx{0};
    int64_t preTotal{0};
    int64_t bDimTail{0};
    int64_t actualS1Len;
    int64_t actualS2Len;
    int64_t s1OuterTmp{0};
    int64_t s2OuterTmp{0};
    bool isInitIndex{false};

    uint32_t preS2CvBegin{0};
    uint32_t preS2CvEnd{0};
    uint32_t nextS2CvBegin{0};
    uint32_t nextS2CvEnd{0};

    int32_t isStart = 1;
    uint32_t pingpongIdx = 1;

    // db
    int64_t s2CvExtend = 0;
    int64_t s2CvExtendAlign = 0;
    int64_t s1CvExtendAlign = 0;
    uint32_t s1VecLoop = 0;
    uint32_t s1VecSize = 0;
    uint32_t s1ExtendSubGraph = 0;
    uint32_t s2Extend = 0;
    uint32_t s2ExtendAlign = 0;
    uint32_t s2VecLoop = 0;
    uint32_t s2VecSize = 0;
    uint32_t s2VecSizeAlign = 0;

    // unpack
    int64_t bDimIdxTmp = 0;
    int64_t n2DimIdxTmp = 0;
    int64_t gDimIdxTmp = 0;
    int64_t s1oDimIdxTmp = 0;
    int64_t bandIdx = 0;

    DropMaskInfo dropMaskInfo = {0};
    // db buffer
    constexpr static uint32_t T2Begin = 0;
    constexpr static uint32_t T1Begin = 33 * 1024;
    constexpr static uint32_t BoolBegin = 51 * 1024;
    constexpr static uint32_t T2BlockBegin = 59 * 1024;
    constexpr static uint32_t U8Begin = 67 * 1024;
    constexpr static uint32_t DbBegin = 75 * 1024;
    constexpr static uint32_t hufTmpBuffBegin = 16 * 1024;

    // calDtype
    constexpr static uint32_t calDtypeBytes = 4;

    // other const
    constexpr static uint32_t cal_block_num = 32 / sizeof(T2);
    constexpr static uint32_t cal_repeat_num = 256 / sizeof(T2);
    constexpr static uint32_t input_block_num = 32 / sizeof(T1);
    constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint32_t INPUT_NUMS = 2;
    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static int64_t C0_SIZE = 16;
    constexpr static int64_t VEC_REPEAT = 8;
    constexpr static uint32_t MAX_BASIC_BLOCK_SIZE = 1024;
    constexpr static uint32_t PSE_PERFORMANCE_MODE = 0x12;
    constexpr static uint32_t PREFIX_COMPRESS_CAUSAL_S_SIZE = 2048;
    constexpr static uint32_t PREFIX_COMPRESS_ALL_MASK_S1_SIZE = 1024;
    constexpr static uint32_t SFMG_RESULT_DB_SIZE = 4 * 1024;
    // SFMG_DB_SIZE = 18 + 18 + 9+ 9 + 18 = 72K（GraphAB的DB是74K）, CLC1占用18K
    constexpr static uint32_t SFMG_DB_INPUT_UBSIZE = 18 * 1024;
    constexpr static uint32_t SFMG_S1_ALIGN_NUM = 8;

    constexpr static uint32_t VEC_S2_LEN = 256;
    bool tndSoftmaxIn;
    enum class AttenMaskCompress {
        Empty = 0,
        PreOnly = 1,
        NextOnly = 2,
        All = 3
    };
    AttenMaskCompress AttenBandMode = AttenMaskCompress::All;

    __aicore__ 
    inline void AddMM1Offsets1(int64_t& _mm1aTensorOffsetCv1, int64_t& _bTensorOffsetCv1, int64_t _d) const 
    {
        if constexpr (INPUT_LAYOUT == TND) {
            if (this->bDimIdx > 0) {
                _mm1aTensorOffsetCv1 += ((__gm__ int64_t *)this->actual_seq_qlen_addr)[this->bDimIdx - 1] * this->n2 * this->g * _d;
                _bTensorOffsetCv1 += ((__gm__ int64_t *)this->actual_seq_kvlen_addr)[this->bDimIdx - 1] * this->n2 * _d;
            }
            _mm1aTensorOffsetCv1 +=
                ((this->s1oDimIdx * this->s1CvInner * this->n2 + this->n2DimIdx) * this->g + this->gDimIdx) * _d;
            _bTensorOffsetCv1 += (this->nextS2CvBegin * this->n2 + this->n2DimIdx) * _d;
        } else if constexpr (INPUT_LAYOUT == BNGSD) {
            _mm1aTensorOffsetCv1 +=
                (((this->bDimIdx * this->n2 + this->n2DimIdx) * this->g + this->gDimIdx) * this->s1 + this->s1oDimIdx * this->s1CvInner) * _d;
            _bTensorOffsetCv1 += ((this->bDimIdx * this->n2 + this->n2DimIdx) * this->s2 + this->nextS2CvBegin) * _d;
        } else if constexpr (INPUT_LAYOUT == SBNGD) {
            _mm1aTensorOffsetCv1 +=
                ((((this->s1oDimIdx * this->s1CvInner) * this->b + this->bDimIdx) * this->n2 + this->n2DimIdx) * this->g + this->gDimIdx) * _d;
            _bTensorOffsetCv1 += ((this->nextS2CvBegin * this->b + this->bDimIdx) * this->n2 + this->n2DimIdx) * _d;
        } else if constexpr (INPUT_LAYOUT == BSNGD) {
            _mm1aTensorOffsetCv1 +=
                (((this->bDimIdx * this->s1 + this->s1oDimIdx * this->s1CvInner) * this->n2 + this->n2DimIdx) * this->g + this->gDimIdx) * _d;
            _bTensorOffsetCv1 += ((this->bDimIdx * this->s2 + this->nextS2CvBegin) * this->n2 + this->n2DimIdx) * _d;
        }  
    }
};

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2<
    T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::Init(__gm__ uint8_t *key, __gm__ uint8_t *keyRope, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query, __gm__ uint8_t *queryRope,
                          __gm__ uint8_t *pse_shift, __gm__ uint8_t *drop_mask, __gm__ uint8_t *atten_mask,
                          __gm__ uint8_t *forward_res, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,
                          __gm__ uint8_t *prefixN, __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen,
                    __gm__ uint8_t *dq, __gm__ uint8_t *dqRope, __gm__ uint8_t *dk, __gm__ uint8_t *dkRope, __gm__ uint8_t *dv, __gm__ uint8_t *dpse,
                          __gm__ uint8_t *workspace,
                          const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *__restrict ordTilingData,
                          TPipe *pipe_in)
{
    keyGm.SetGlobalBuffer((__gm__ T1 *)key);
    valueGm.SetGlobalBuffer((__gm__ T1 *)value);
    dxGm.SetGlobalBuffer((__gm__ T1 *)dx);
    queryGm.SetGlobalBuffer((__gm__ T1 *)query);
    forwardResGm.SetGlobalBuffer((__gm__ T1 *)forward_res);
    pseGm.SetGlobalBuffer((__gm__ T1 *)pse_shift);
    pseSlope = pse_shift;
    if constexpr (HAS_ROPE == ENABLE) {
        keyRopeGm.SetGlobalBuffer((__gm__ T1 *)keyRope);
        queryRopeGm.SetGlobalBuffer((__gm__ T1 *)queryRope);
    }
    dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)drop_mask);
    attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t *)atten_mask);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmax_max);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmax_sum);
    dvGm.SetGlobalBuffer((__gm__ float *)dv);

    // init current core tilingInfo
    cBlockIdx = GetBlockIdx();
    TilingData = ordTilingData;
    pipe = pipe_in;
    coreNum = TilingData->s1s2BNGS1S2BaseParams.coreNum;

    // shape info
    b = TilingData->s1s2BNGS1S2BaseParams.b;
    n2 = TilingData->s1s2BNGS1S2BaseParams.n2;
    g = TilingData->s1s2BNGS1S2BaseParams.g;
    s1 = TilingData->s1s2BNGS1S2BaseParams.s1;
    s2 = TilingData->s1s2BNGS1S2BaseParams.s2;
    d = TilingData->s1s2BNGS1S2BaseParams.d;
    tndSoftmaxIn = TilingData->s1s2BNGS1S2BaseParams.tndSoftmaxIn == 1 ? true : false;
    value_d = TilingData->s1s2BNGS1S2BaseParams.value_d;
    if constexpr (HAS_ROPE == ENABLE) {
        rope_d = TilingData->s1s2BNGS1S2BaseParams.rope_d;
    }
    attenMaskDimS2 = TilingData->s1s2BNGS1S2BaseParams.attenMaskS2Size;

    s1Token = TilingData->s1s2BNGS1S2BaseParams.s1Token;
    s2Token = TilingData->s1s2BNGS1S2BaseParams.s2Token;
    actualCalcS1Token = s1Token;
    actualCalcS2Token = s2Token;
    sparseMode = TilingData->s1s2BNGS1S2BaseParams.sparseMode;
    isSparse = false;
    if (TilingData->s1s2BNGS1S2BaseParams.isSparse == 1) {
        isSparse = true;
    }
    bandIdx = TilingData->s1s2BNGS1S2SplitCoreParams.bandIdx;

    // split info
    s1Outer = TilingData->s1s2BNGS1S2SplitCoreParams.s1Outer;
    s1CvRatio = TilingData->s1s2BNGS1S2SplitCoreParams.s1CvRatio;
    s1CvInner = TilingData->s1s2BNGS1S2SplitCoreParams.s1CvInner;
    s1CvTail = TilingData->s1s2BNGS1S2SplitCoreParams.s1CvTail;
    s2Outer = TilingData->s1s2BNGS1S2SplitCoreParams.s2Outer;
    s2CvRatio = TilingData->s1s2BNGS1S2SplitCoreParams.s2CvRatio;
    s2Inner = TilingData->s1s2BNGS1S2SplitCoreParams.s2Inner;
    s2CvInner = s2Inner * s2CvRatio;

    sfmgdOuter = TilingData->s1s2BNGS1S2SplitCoreParams.sfmgdOuter;
    sfmgdInner = TilingData->s1s2BNGS1S2SplitCoreParams.sfmgdFactor;
    sfmgdTail = TilingData->s1s2BNGS1S2SplitCoreParams.sfmgdTail;
    sfmgdTailAlign = (sfmgdTail + input_block_num - 1) / input_block_num * input_block_num;

    // no sparse blockouter ceil to even
    blockOuter = (TilingData->s1s2BNGS1S2SplitCoreParams.blockOuter + 1) / 2 * 2;
    baseMN = TilingData->s1s2BNGS1S2SplitCoreParams.baseMN;
    cubeBaseMN = s1CvRatio * s2CvRatio * baseMN;

    blockStarts = TilingData->s1s2BNGS1S2BlockNumList.blockStarts;
    blockEnds = TilingData->s1s2BNGS1S2BlockNumList.blockEnds;

    prefixN_addr = prefixN;
    actual_seq_qlen_addr = actual_seq_qlen;
    actual_seq_kvlen_addr = actual_seq_kvlen;

    if constexpr (IS_PSE == ENABLE) {
        uint32_t pseShapeType = TilingData->s1s2BNGS1S2BaseParams.pseShapeType;
        pseInfo.s2Size = s2;
        pseInfo.s1Size = s1;
        pseInfo.gSize = g;
        pseInfo.n2G = n2 * g;
        pseInfo.pseType = TilingData->s1s2BNGS1S2BaseParams.pseType;
        pseInfo.pseShapeType = pseShapeType;
        if (pseShapeType == 2 || pseShapeType == 3 || pseShapeType == 4) {
            pseInfo.pseShapeType = 0;
        } else if (pseShapeType == 5) {
            pseInfo.pseShapeType = 2;
        } else if (pseShapeType == 6) {
            pseInfo.pseShapeType = 3;
        }

        pseInfo.pseAlibiBaseS1 = TilingData->s1s2BNGS1S2BaseParams.pseAlibiBaseS1;
        pseInfo.pseAlibiBaseS2 = TilingData->s1s2BNGS1S2BaseParams.pseAlibiBaseS2;
        pseInfo.qStartIdx = TilingData->s1s2BNGS1S2BaseParams.qStartIdx;
        pseInfo.kvStartIdx = TilingData->s1s2BNGS1S2BaseParams.kvStartIdx;
        pseInfo.pseEncodeType = (pseShapeType == 3 || pseShapeType == 4) ? 0x11 : 0;
        pseInfo.pseBSize = (pseShapeType == 2 || pseShapeType == 4) ? 1 : b;
        pseInfo.pseS1Size = 1024;
        pseInfo.pseS2Size = s2;
        pseInfo.needCast = false;
    }

    dropBitMode = s2 % 8 == 0;
    keepProb = TilingData->s1s2BNGS1S2BaseParams.keepProb;
    if constexpr (INPUT_LAYOUT == TND) {
        int64_t seqS2Len = 0;
        seqS2Len = ((__gm__ int64_t *)actual_seq_kvlen)[0];
        dropBitMode = (seqS2Len % 8 == 0);
        for (int64_t i = 0; i + 1 < b; i++) {
            seqS2Len = ((__gm__ int64_t *)actual_seq_kvlen)[i + 1] - ((__gm__ int64_t *)actual_seq_kvlen)[i];
            dropBitMode = (dropBitMode && (seqS2Len % 8 == 0));
        }
    }

    // idx info
    n2gs1os2o = n2 * g * s1Outer * s2Outer;
    gs1os2o = g * s1Outer * s2Outer;
    s1os2o = s1Outer * s2Outer;

    int64_t maskPreBlockTotal = TilingData->preTilingData.maskPreBlockTotal;
    int64_t qPostBlockTotal = TilingData->postTilingData.qSizeAlign;
    int64_t kvPostBlockTotal = TilingData->postTilingData.kvSizeAlign;
    int64_t vPostBlockTotal = TilingData->postTilingData.vSizeAlign;
    int64_t qRopePostBlockTotal = 0;
    int64_t kRopePostBlockTotal = 0;
    if constexpr (HAS_ROPE == ENABLE) {
        qRopePostBlockTotal = TilingData->postTilingData.qRopeSizeAlign;
        kRopePostBlockTotal = TilingData->postTilingData.kRopeSizeAlign;
    }

    // init workspace address
    syncGlobal.SetGlobalBuffer((__gm__ int32_t *)workspace);
    int64_t workspaceOffsets = SYNC_GLOBAL_WORKSPACE_SIZE;
    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + workspaceOffsets / sizeof(T2));
    workspaceOffsets =
        (workspaceOffsets + qPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    if constexpr (HAS_ROPE == ENABLE) {
        dqRopeWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + workspaceOffsets / sizeof(T2));
        workspaceOffsets =
            (workspaceOffsets + qRopePostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    }
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + workspaceOffsets / sizeof(T2));
    workspaceOffsets =
        (workspaceOffsets + kvPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    if constexpr (HAS_ROPE == ENABLE) {
        dkRopeWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + workspaceOffsets / sizeof(T2));
        workspaceOffsets =
            (workspaceOffsets + kRopePostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    }
    dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + workspaceOffsets / sizeof(T2));
    workspaceOffsets =
        (workspaceOffsets + vPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    

    if constexpr (IS_DROP == ENABLE) {
        if (!dropBitMode) {
            maskWorkSpaceGm.SetGlobalBuffer((__gm__ uint8_t *)workspace + workspaceOffsets);
            workspaceOffsets =
                (workspaceOffsets + maskPreBlockTotal + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
        }
    }

    int64_t pseInnerAlibiSize = TilingData->s1s2BNGS1S2BaseParams.pseAlibiBaseS1 *
                                this->TilingData->s1s2BNGS1S2BaseParams.pseAlibiBaseS2 * sizeof(half);
    int64_t pseAlibiOffset =  CeilDiv(pseInnerAlibiSize, 512) * 512;

    // matmal1 and matmal2 workspace size
    matmalWorkspaceSize = cubeBaseMN * sizeof(float);
    mm1WorkspaceGm.SetGlobalBuffer((__gm__ T2 *)(workspace + workspaceOffsets + cBlockIdx * matmalWorkspaceSize));
    mm2WorkspaceGm.SetGlobalBuffer(
        (__gm__ T2 *)(workspace + workspaceOffsets + coreNum * matmalWorkspaceSize + cBlockIdx * matmalWorkspaceSize));

    // drop workspace offset
    workspaceOffsets = (workspaceOffsets + coreNum * cubeBaseMN * sizeof(float) * INPUT_NUMS + ADDR_ALIGN_SIZE) /
                       ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    dropWorkSpaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + workspaceOffsets / sizeof(T1));

    // mul workspace offset
    workspaceOffsets = (workspaceOffsets + coreNum * cubeBaseMN * sizeof(T1) * 2 + ADDR_ALIGN_SIZE) /
                       ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    mulWorkSpaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + workspaceOffsets / sizeof(T1));

    uint64_t pseAlibiAddr = (workspaceOffsets + coreNum * cubeBaseMN * sizeof(T1) * 2 + ADDR_ALIGN_SIZE) /
                       ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    this->pseAlibiGm.SetGlobalBuffer((__gm__ half*)(workspace + pseAlibiAddr + cBlockIdx * pseAlibiOffset));

    InitOutput<int32_t>(syncGlobal[GetBlockIdx() * 8], 8, 0);

    pipe->InitBuffer(ubBuffer, 150 * 1024);
    pipe->InitBuffer(tmpBuffer, 33 * 1024);
    pipe->InitBuffer(vecClc3, 8 * 1024);

    if constexpr (IS_DROP == ENABLE) {
        if constexpr (INPUT_LAYOUT != TND) {
            // for compute dropout mask offset
            dropMaskInfo.s1Size = s1;
            dropMaskInfo.s2Size = s2;
        }

        // for compute dropout mask offset
        dropMaskInfo.n2G = n2 * g;
        dropMaskInfo.gSize = g;
        dropMaskInfo.s2Idx = 1;
        dropMaskInfo.s1BaseSize = s1CvInner;

        // for copy and compute in dropout mask
        dropMaskInfo.boolMode = dropBitMode ? false : true;
        dropMaskInfo.keepProb = keepProb;
    }

    if constexpr (IS_PSE == ENABLE) {
        if (cBlockIdx < coreNum &&
            (TilingData->s1s2BNGS1S2BaseParams.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            TilingData->s1s2BNGS1S2BaseParams.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            LocalTensor<half> pseHelpBuffer = ubBuffer.GetWithOffset<half>(16 * 1024 / sizeof(half), T1Begin);
            PseInnerAlibiCreate<true>(this->pseAlibiGm, pseHelpBuffer, pseInfo);
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::GetSeqQlenKvlenByBidx(int64_t bIdx,
                                                                            int64_t &actualSeqQlen,
                                                                            int64_t &actualSeqKvlen)
{
    if (unlikely(bIdx == 0)) {
        actualSeqQlen = ((__gm__ int64_t *)actual_seq_qlen_addr)[0];
        actualSeqKvlen = ((__gm__ int64_t *)actual_seq_kvlen_addr)[0];
    } else {
        actualSeqQlen =
            ((__gm__ int64_t *)actual_seq_qlen_addr)[bIdx] - ((__gm__ int64_t *)actual_seq_qlen_addr)[bIdx - 1];
        actualSeqKvlen =
            ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIdx] - ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIdx - 1];
    }
    return;
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::CopyInSoftMaxGrad(LocalTensor<T1> &dstTensor,
                                                                       LocalTensor<T1> &dstTensor2,
                                                                       int64_t softmaxGradFrontOffset,
                                                                       uint32_t s1Extend, uint32_t dExtend,
                                                                       uint32_t dExtendAlign)
{
    int64_t transpse_stride = 0;
    if constexpr (INPUT_LAYOUT == BNGSD) {
        transpse_stride = (value_d - dExtend) * sizeof(T1);
    } else if constexpr (INPUT_LAYOUT == SBNGD) {
        transpse_stride = (static_cast<int64_t>(b) * n2 * g * value_d - dExtend) * sizeof(T1);
    } else if constexpr (INPUT_LAYOUT == BSNGD) {
        transpse_stride = (n2 * g * value_d - dExtend) * sizeof(T1);
    } else if constexpr (INPUT_LAYOUT == TND) {
        transpse_stride = (n2 * g * value_d - dExtend) * sizeof(T1);
    }

    // 如果n2*g*d = dExtend,但是不对齐，理应走DataCopyPad给后面补0
    if (transpse_stride == 0 && (dExtend % 16 == 0)) {
        DataCopy(dstTensor, forwardResGm[softmaxGradFrontOffset], s1Extend * dExtend);
        DataCopy(dstTensor2, dxGm[softmaxGradFrontOffset], s1Extend * dExtend);
    } else {
        DataCopyPad(dstTensor, forwardResGm[softmaxGradFrontOffset],
                    {static_cast<uint16_t>(s1Extend), static_cast<uint32_t>(dExtend * sizeof(T1)),
                     static_cast<uint32_t>(transpse_stride), 0, 0},
                    {true, 0, static_cast<uint8_t>((dExtendAlign - dExtend)), 0});
        DataCopyPad(dstTensor2, dxGm[softmaxGradFrontOffset],
                    {static_cast<uint16_t>(s1Extend), static_cast<uint32_t>(dExtend * sizeof(T1)),
                     static_cast<uint32_t>(transpse_stride), 0, 0},
                    {true, 0, static_cast<uint8_t>((dExtendAlign - dExtend)), 0});
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::CalcSoftMaxGradPingPong(int64_t outerPingPong, int64_t aTensorOffset,
        uint32_t s1Extend, EvenvIdList &eventIdList)
{
    bool isOuterPing = ((outerPingPong % 2) == 0);
    int32_t curEventIdMte2WaitMte3 = static_cast<int32_t>(isOuterPing ?
                                     eventIdList.structMte2WaitMte3Ping : eventIdList.structMte2WaitMte3Pong);
    // SFMG加入到PingPong循环中，这里需要等待GraphB中的set的MTE3、2；
    if (outerPingPong > 1) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(curEventIdMte2WaitMte3);
    }

    int64_t sfmgResultOffset = isOuterPing ? 0 : SFMG_RESULT_DB_SIZE;
    int64_t pongOffset = isOuterPing ? 0 : DbBegin;
    LocalTensor<float> sfmgClc3 = vecClc3.GetWithOffset<float>(SFMG_RESULT_DB_SIZE / sizeof(float), sfmgResultOffset);

    // sfmg的两个直接输入各占用18K
    LocalTensor<float> sfmgClc1 = ubBuffer.GetWithOffset<float>(18 * 1024 / sizeof(T2), pongOffset + 0);
    LocalTensor<float> sfmgClc2 = ubBuffer.GetWithOffset<float>(18 * 1024 / sizeof(T2), pongOffset + 18 * 1024);
    Duplicate<float>(sfmgClc3, 0.0, s1Extend * 32 / sizeof(float));

    LocalTensor<T1> vecInBuffer;
    LocalTensor<T1> vecInBuffer2;
    // 传入的sfmg的两个输入（非FP32时）各占用9K
    if constexpr (!IsSameType<T1, float>::value) {
        vecInBuffer = ubBuffer.GetWithOffset<T1>(9 * 1024 / sizeof(T1), pongOffset + 36 * 1024);
        vecInBuffer2 = ubBuffer.GetWithOffset<T1>(9 * 1024 / sizeof(T1), pongOffset + 45 * 1024);
    }
    // sfmg的所需的临时空间18K
    LocalTensor<uint8_t> vecOutBuffer = ubBuffer.GetWithOffset<uint8_t>(18 * 1024 / sizeof(uint8_t), pongOffset + 54 * 1024);
    int64_t softmaxGradFrontOffset = aTensorOffset;
    uint32_t dExtend = sfmgdTail;
    uint32_t dExtendAlign = sfmgdTailAlign;
    bool isBasicBlock = (s1Extend % 8 == 0) && (dExtend % 64 == 0);

    if constexpr (!IsSameType<T1, float>::value) {
        CopyInSoftMaxGrad(vecInBuffer, vecInBuffer2, softmaxGradFrontOffset, s1Extend, dExtend, dExtendAlign);
    } else {
        CopyInSoftMaxGrad(sfmgClc1, sfmgClc2, softmaxGradFrontOffset, s1Extend, dExtend, dExtendAlign);
    }
    int32_t vWaitMte2Pingpong = static_cast<int32_t>(isOuterPing ?
                        eventIdList.structVWaitMte2Ping : eventIdList.structVWaitMte2Pong);
    AscendC::SetFlag<HardEvent::MTE2_V>(vWaitMte2Pingpong);
    AscendC::WaitFlag<HardEvent::MTE2_V>(vWaitMte2Pingpong);
    if constexpr (!IsSameType<T1, float>::value) {
        Cast(sfmgClc1, vecInBuffer, RoundMode::CAST_NONE, s1Extend * dExtendAlign);
        Cast(sfmgClc2, vecInBuffer2, RoundMode::CAST_NONE, s1Extend * dExtendAlign);
        AscendC::PipeBarrier<PIPE_V>();
    }
    uint32_t shapeArray[2];
    shapeArray[0] = s1Extend;
    shapeArray[1] = dExtendAlign;
    sfmgClc1.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    sfmgClc2.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    uint32_t shapeArrayResult[2];
    shapeArrayResult[0] = s1Extend;
    shapeArrayResult[1] = 32 / sizeof(float);

    sfmgClc3.SetShapeInfo(ShapeInfo(2, shapeArrayResult, DataFormat::ND));

    if (isBasicBlock) {
        SoftmaxGradFront<float, true>(sfmgClc3, sfmgClc1, sfmgClc2, vecOutBuffer, TilingData->softmaxGradTilingData);
    } else {
        SoftmaxGradFront<float, false>(sfmgClc3, sfmgClc1, sfmgClc2, vecOutBuffer, TilingData->softmaxGradTilingData);
    }
    int32_t mte2WaitVSfmg2A = static_cast<int32_t>(isOuterPing ?
                              eventIdList.structMte2WaitVPing : eventIdList.structMte2WaitVPong);
    AscendC::SetFlag<HardEvent::V_MTE2>(mte2WaitVSfmg2A);
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::CalcSoftMaxGrad(LocalTensor<float> &sfmgClc3,
                                                                     int64_t aTensorOffset, uint32_t s1Extend)
{
    LocalTensor<float> sfmgClc1 = ubBuffer.GetWithOffset<float>(32 * 1024 / sizeof(T2), 0);
    LocalTensor<float> sfmgClc2 = ubBuffer.GetWithOffset<float>(32 * 1024 / sizeof(T2), 32 * 1024);
    Duplicate<float>(sfmgClc3, 0.0, s1Extend * 32 / sizeof(float));

    int32_t mte2WaitV = static_cast<int32_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    for (uint32_t sfmgdIdx = 0; sfmgdIdx < sfmgdOuter; sfmgdIdx++) {
        LocalTensor<T1> vecInBuffer;
        LocalTensor<T1> vecInBuffer2;
        if constexpr (!IsSameType<T1, float>::value) {
            vecInBuffer = ubBuffer.GetWithOffset<T1>(16 * 1024 / sizeof(T1), 64 * 1024);
            vecInBuffer2 = ubBuffer.GetWithOffset<T1>(16 * 1024 / sizeof(T1), 80 * 1024);
        }
        LocalTensor<uint8_t> vecOutBuffer = ubBuffer.GetWithOffset<uint8_t>(32 * 1024 / sizeof(uint8_t), 96 * 1024);
        LocalTensor<T2> softmaxGradTmp = ubBuffer.GetWithOffset<T2>(8 * 1024 / sizeof(T2), 128 * 1024);
        int64_t softmaxGradFrontOffset = aTensorOffset + sfmgdIdx * sfmgdInner;
        uint32_t dExtend = (sfmgdIdx == sfmgdOuter - 1) ? sfmgdTail : sfmgdInner;
        uint32_t dExtendAlign = (sfmgdIdx == sfmgdOuter - 1) ? sfmgdTailAlign : sfmgdInner;
        bool isBasicBlock = (s1Extend % 8 == 0) && (dExtend % 64 == 0);

        if (sfmgdIdx > 0) {
            AscendC::WaitFlag<HardEvent::V_MTE2>(mte2WaitV);
        }
        if constexpr (!IsSameType<T1, float>::value) {
            CopyInSoftMaxGrad(vecInBuffer, vecInBuffer2, softmaxGradFrontOffset, s1Extend, dExtend, dExtendAlign);
        } else {
            CopyInSoftMaxGrad(sfmgClc1, sfmgClc2, softmaxGradFrontOffset, s1Extend, dExtend, dExtendAlign);
        }
        int32_t vWaitMte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        AscendC::SetFlag<HardEvent::MTE2_V>(vWaitMte2);
        AscendC::WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
        if constexpr (!IsSameType<T1, float>::value) {
            Cast(sfmgClc1, vecInBuffer, RoundMode::CAST_NONE, s1Extend * dExtendAlign);
            Cast(sfmgClc2, vecInBuffer2, RoundMode::CAST_NONE, s1Extend * dExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
        }
        uint32_t shapeArray[2];
        shapeArray[0] = s1Extend;
        shapeArray[1] = dExtendAlign;
        sfmgClc1.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        sfmgClc2.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        uint32_t shapeArray1[2];
        shapeArray1[0] = s1Extend;
        shapeArray1[1] = 32 / sizeof(float);
        softmaxGradTmp.SetShapeInfo(ShapeInfo(2, shapeArray1, DataFormat::ND));

        if (isBasicBlock) {
            SoftmaxGradFront<float, true>(softmaxGradTmp, sfmgClc1, sfmgClc2, vecOutBuffer,
                                          TilingData->softmaxGradTilingData);
        } else {
            SoftmaxGradFront<float, false>(softmaxGradTmp, sfmgClc1, sfmgClc2, vecOutBuffer,
                                           TilingData->softmaxGradTilingData);
        }
        AscendC::PipeBarrier<PIPE_V>();
        Add(sfmgClc3, softmaxGradTmp, sfmgClc3, s1Extend * 32 / sizeof(float));
        if (sfmgdIdx < (sfmgdOuter - 1)) {
            AscendC::SetFlag<HardEvent::V_MTE2>(mte2WaitV);
        }
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(mte2WaitV);
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::CopyInSoftMax(LocalTensor<float> &dstTensor, uint32_t s1Extend,
                                                                   uint32_t softMaxOffset)
{
    DataCopyPad(dstTensor, softmaxSumGm[softMaxOffset], {1, static_cast<uint16_t>(s1Extend * 32), 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(dstTensor[s1Extend * 32 / sizeof(float)], softmaxMaxGm[softMaxOffset],
                {1, static_cast<uint16_t>(s1Extend * 32), 0, 0}, {false, 0, 0, 0});
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::CalcSoftMax(LocalTensor<T2> &dstTensor,
                                                                 LocalTensor<float> &srcTensor, uint32_t s1Extend,
                                                                 uint32_t s2Extend, uint32_t s2ExtendAlign,
                                                                 const SoftMaxTiling &tiling)
{
    bool isBasicBlock = (s1Extend % 8 == 0) && (s2Extend % 64 == 0);

    if (isBasicBlock) {
        LocalTensor<uint8_t> vecOutBuffer = tmpBuffer.Get<uint8_t>();
        uint32_t shapeArray1[2];
        shapeArray1[0] = s1Extend;
        shapeArray1[1] = s2Extend;
        dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray1, DataFormat::ND));
        SimpleSoftMax<T2, true, true>(dstTensor, srcTensor, srcTensor[s1Extend * 32 / sizeof(float)], dstTensor,
                                      vecOutBuffer, tiling);
    } else {
        LocalTensor<T2> vecOutBuffer = tmpBuffer.Get<T2>();
        uint32_t sub_block_count = (s2Extend + cal_repeat_num - 1) / cal_repeat_num;

        for (uint32_t subIdx = 0; subIdx < sub_block_count; subIdx++) {
            uint32_t subMaskCount =
                (subIdx == sub_block_count - 1) ? (s2Extend - subIdx * cal_repeat_num) : cal_repeat_num;
            Sub(dstTensor[subIdx * cal_repeat_num], dstTensor[subIdx * cal_repeat_num], srcTensor[s1Extend * 8],
                subMaskCount, s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8), 1});
            AscendC::PipeBarrier<PIPE_V>();
            Exp(vecOutBuffer[subIdx * cal_repeat_num], dstTensor[subIdx * cal_repeat_num], subMaskCount, s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8)});
            AscendC::PipeBarrier<PIPE_V>();
            Div(dstTensor[subIdx * cal_repeat_num], vecOutBuffer[subIdx * cal_repeat_num], srcTensor, subMaskCount,
                s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8), 1});
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::CopyInAttenMaskBool(LocalTensor<uint8_t> &dstTensor,
                                                                         int64_t attenMaskOffset, uint32_t s1Extend,
                                                                         uint32_t s2Extend)
{
    AscendC::DataCopyExtParams intriParams;
    intriParams.blockCount = s1Extend;
    intriParams.blockLen = s2Extend * sizeof(uint8_t);
    intriParams.srcStride = (attenMaskDimS2 - s2Extend) * sizeof(uint8_t);
    intriParams.dstStride = 0;
    intriParams.rsv = 0;
    DataCopyPad(dstTensor, attenMaskU8Gm[attenMaskOffset], intriParams, {false, 0, 0, 0});
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::CalcAttenMaskBool(LocalTensor<T2> &dstTensor,
                                                                       LocalTensor<uint8_t> srcTensor,
                                                                       uint32_t s1Extend, uint32_t s2Extend,
                                                                       uint8_t maskType)
{
    LocalTensor<uint8_t> tmpUbBuffer = tmpBuffer.Get<uint8_t>();

    T2 scalar;
    if constexpr (IsSameType<T2, float>::value) {
        uint32_t tmp = 0xFF7FFFFF;
        scalar = *((float *)&tmp);
    } else {
        uint16_t tmp = 0xFBFF;
        scalar = *((half *)&tmp);
    }

    SelectWithBytesMaskShapeInfo info;
    info.firstAxis = s1Extend;
    info.srcLastAxis = s2Extend;
    info.maskLastAxis = (s2Extend * sizeof(uint8_t) + 31) / 32 * 32 / sizeof(uint8_t);
    dstTensor.SetSize(info.firstAxis * info.srcLastAxis);
    srcTensor.SetSize(info.firstAxis * info.maskLastAxis);
    if (maskType == 0) {
        SelectWithBytesMask(dstTensor, dstTensor, scalar, srcTensor, tmpUbBuffer, info);
    } else {
        SelectWithBytesMask(dstTensor, scalar, dstTensor, srcTensor, tmpUbBuffer, info);
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,
                                                           INPUT_LAYOUT, MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::Process()
{
    if (isSparse) {
        int64_t preIndex = blockStarts[cBlockIdx];
        if (blockEnds[cBlockIdx] == 0) {
            return;
        }

        InitIndex(preIndex);
        int64_t blockEndsTemp = blockEnds[cBlockIdx];
        for (int64_t blockInnerIdx = blockStarts[cBlockIdx] + 1; blockInnerIdx <= blockEndsTemp; blockInnerIdx++) {
            if (isStart == 1) {
                if (!IsCubeBlockNeedCompute(preIndex)) {
                    preIndex = blockInnerIdx;
                    continue;
                }
            }
            preS2CvBegin = nextS2CvBegin;
            preS2CvEnd = nextS2CvEnd;
            if (IsCubeBlockNeedCompute(blockInnerIdx) || (blockInnerIdx >= blockEnds[cBlockIdx])) {
                int64_t nextIndex = blockInnerIdx;
                if (blockInnerIdx >= blockEnds[cBlockIdx]) {
                    nextIndex = 0u;
                }
                Compute(preIndex, nextIndex);
                preIndex = nextIndex;
            }
        }
    } else {
        int64_t preIndex = cBlockIdx;
        int64_t total = static_cast<int64_t>(b) * n2 * g * s1Outer * s2Outer;
        if (cBlockIdx >= total) {
            return;
        }

        IsCubeBlockNeedCompute(preIndex);
        preS2CvBegin = nextS2CvBegin;
        preS2CvEnd = nextS2CvEnd;

        int64_t totalTemp = total + blockOuter;
        for (int64_t blockInnerIdx = cBlockIdx + blockOuter; blockInnerIdx < totalTemp; blockInnerIdx += blockOuter) {
            if (IsCubeBlockNeedCompute(blockInnerIdx) || (blockInnerIdx >= total)) {
                int64_t nextIndex = blockInnerIdx;
                if (blockInnerIdx >= total) {
                    nextIndex = 0u;
                }
                Compute(preIndex, nextIndex);
                preIndex = nextIndex;
                preS2CvBegin = nextS2CvBegin;
                preS2CvEnd = nextS2CvEnd;
            }
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline bool
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::CheckIsValidBlock(int64_t baseIdx, int64_t s1oDimIdx,
                                                                       int64_t s2oCvDimIdx, int64_t curBIdx)
{
    int64_t S1 = static_cast<int64_t>(s1);
    int64_t S2 = static_cast<int64_t>(s2);

    if constexpr (INPUT_LAYOUT == TND) {
        GetSeqQlenKvlenByBidx(curBIdx, S1, S2);
    }
    int64_t cvS2Inner = static_cast<int64_t>(s2CvRatio) * s2Inner;
    int64_t s2IgnoredEndLen = S1 - static_cast<int64_t>(s1CvInner * (s1oDimIdx + 1));
    int64_t s2EndLen = 0;
    if (S2 > s2IgnoredEndLen) {
        s2EndLen = S2 - s2IgnoredEndLen;
    } else {
        s2EndLen = 0;
    }

    if (sparseMode == 5 || sparseMode == 6) {
        s2EndLen = s2EndLen > ((__gm__ int64_t *)prefixN_addr)[curBIdx]
                            ? s2EndLen
                            : ((__gm__ int64_t *)prefixN_addr)[curBIdx];
        s2EndLen = s2EndLen < S2 ? s2EndLen : S2;
    }

    uint32_t s2IdxLeft = s2oCvDimIdx * s2CvInner;
    uint32_t s2IdxRight = (s2oCvDimIdx + 1) * cvS2Inner;
    bool doSparse = s2IdxLeft < s2EndLen;
    if (doSparse) {
        nextS2CvBegin = s2IdxLeft;
        nextS2CvEnd = s2IdxRight > s2EndLen ? s2EndLen : s2IdxRight;
    }
    return doSparse;
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,
                                                           INPUT_LAYOUT, MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::UpdateToken(int64_t bIdx)
{
    // sparse_mode =4 (band)时 或者sparse_mode ==3 (RIGHT_DOWN_CAUSAL) 时，token以右下角为基准，需要校正
    if constexpr (IS_ATTEN_MASK != ENABLE) {
        return;
    }

    int64_t actualS1Len;
    int64_t actualS2Len;
    if (sparseMode == 7 && bIdx != bandIdx) {
        GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
        actualCalcS1Token = static_cast<int64_t>(INT32_MAX) + actualS1Len - actualS2Len;
        actualCalcS2Token = static_cast<int64_t>(0) - actualS1Len + actualS2Len;
    } else if (sparseMode == 8 && bIdx != bandIdx) {
        actualCalcS1Token = INT32_MAX;
        actualCalcS2Token = 0;
    } else if (sparseMode == 3 || sparseMode == 4 || (sparseMode == 7 && bIdx == bandIdx) ||
               (sparseMode == 8 && bIdx == bandIdx)) {
        GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
        actualCalcS1Token = s1Token + actualS1Len - actualS2Len;
        actualCalcS2Token = s2Token - actualS1Len + actualS2Len;
    }
}


template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline bool
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::IsCubeBlockNeedCompute(int64_t baseIdx)
{
    if constexpr (INPUT_LAYOUT == TND) {
        // 安全防护，baseIdx不可小于0，防止gDimTail、s2oCvDimIdx等出现除0
        int64_t resbaseIdx = baseIdx < 0 ? 0 : baseIdx;
        int64_t actualS1LenCur = actualS1Len;
        int64_t actualS2LenCur = actualS2Len;
        int64_t s1OuterTmpCur = s1OuterTmp;
        int64_t s2OuterTmpCur = s2OuterTmp;
        int64_t s1s2OuterTmp = s1OuterTmpCur * s2OuterTmpCur;
        int64_t preTotalCur = preTotal;
        int64_t totalBaseIdxTmp = totalBaseIdx;
        int64_t bIdx = bDimIdx;
        if (resbaseIdx >= totalBaseIdx) {
            for (bIdx = bDimIdx + 1; bIdx < b; bIdx++) {
                GetSeqQlenKvlenByBidx(bIdx, actualS1LenCur, actualS2LenCur);
                s1OuterTmpCur = (actualS1LenCur + s1CvInner - 1) / s1CvInner;
                s2OuterTmpCur = (actualS2LenCur + s2CvInner - 1) / s2CvInner;
                s1s2OuterTmp = s1OuterTmpCur * s2OuterTmpCur;
                preTotalCur = totalBaseIdx;
                totalBaseIdxTmp += static_cast<int64_t>(n2) * g * s1s2OuterTmp;
                if (resbaseIdx < totalBaseIdxTmp) {
                    break;
                }
            }
        }
        int64_t gDimTail = (resbaseIdx - preTotalCur) % s1s2OuterTmp;
        int64_t s2oCvDimIdx = gDimTail / s1OuterTmpCur;
        int64_t s1oDimIdx = gDimTail % s1OuterTmpCur;
        int64_t s2IdxLeft = s2oCvDimIdx * s2CvInner;
        int64_t s2IdxRight = (s2oCvDimIdx + 1) * s2CvInner < actualS2LenCur
                            ? (s2oCvDimIdx + 1) * s2CvInner
                            : actualS2LenCur;

        // 6: prefix压缩，unpad只支持prefix压缩，不支持prefix
        if (sparseMode == 6) {
            return CheckIsValidBlock(baseIdx, s1oDimIdx, s2oCvDimIdx, bIdx);
        }
        UpdateToken(bIdx);
        int64_t s2SparseLeftPre = int64_t(s1CvInner * s1oDimIdx) - actualCalcS1Token;
        int64_t s2SparseLeft = s2SparseLeftPre < 0 ?
                                0 : s2SparseLeftPre;
        s2SparseLeft = s2SparseLeft / 64 * 64;
        int64_t s2SparseRight = (s1CvInner * (s1oDimIdx + 1) + actualCalcS2Token + 63) / 64 * 64 < 0 ?
                                0 :
                                (s1CvInner * (s1oDimIdx + 1) + actualCalcS2Token + 63) / 64 * 64;
        s2SparseRight = static_cast<uint32_t>(s2SparseRight)
                            < actualS2LenCur ? s2SparseRight : actualS2LenCur;
        if (s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft) {
            nextS2CvBegin = s2IdxLeft < s2SparseLeft ? s2SparseLeft : s2IdxLeft;
            nextS2CvEnd = s2IdxRight > s2SparseRight ? s2SparseRight : s2IdxRight;
            return true;
        } else {
            return false;
        }
    } else {
        uint32_t gDimTail = baseIdx % s1os2o;
        uint32_t s2oCvDimIdx = gDimTail / s1Outer;
        uint32_t s1oDimIdx = gDimTail % s1Outer;
        uint32_t s2IdxLeft = s2oCvDimIdx * s2CvInner;
        uint32_t s2IdxRight =
            (s2oCvDimIdx + 1) * s2CvInner < s2 ? (s2oCvDimIdx + 1) * s2CvInner : s2;
        if (!isSparse) {
            nextS2CvBegin = s2IdxLeft;
            nextS2CvEnd = s2IdxRight;
            return true;
        }

        if (sparseMode == 5 || sparseMode == 6) {
            uint32_t curBIdx = baseIdx / n2gs1os2o;
            return CheckIsValidBlock(baseIdx, s1oDimIdx, s2oCvDimIdx, curBIdx);
        } else {
            uint32_t s2SparseLeft =
                int64_t(s1CvInner * s1oDimIdx) - actualCalcS1Token < 0 ? 0 : s1CvInner * s1oDimIdx - actualCalcS1Token;
            s2SparseLeft = s2SparseLeft / 64 * 64;
            uint32_t s2SparseRight = (s1CvInner * (s1oDimIdx + 1) + actualCalcS2Token + 63) / 64 * 64 < 0 ?
                                         0 :
                                         (s1CvInner * (s1oDimIdx + 1) + actualCalcS2Token + 63) / 64 * 64;
            s2SparseRight = s2SparseRight < s2 ? s2SparseRight : s2;
            if (s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft) {
                nextS2CvBegin = s2IdxLeft < s2SparseLeft ? s2SparseLeft : s2IdxLeft;
                nextS2CvEnd = s2IdxRight > s2SparseRight ? s2SparseRight : s2IdxRight;
                return true;
            } else {
                return false;
            }
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::CalcAttenMaskOffset(int64_t &attenMaskOffset, const int64_t delta,
                                                                         uint32_t s1VSize, uint32_t s2VSize)
{
    if (delta == 0) {
        attenMaskOffset = 0;
    } else if (delta < 0) {
        if (-delta > s1VSize) {
            attenMaskOffset = s1VSize;
        } else {
            attenMaskOffset = -delta;
        }
    } else {
        if (delta > s2VSize) {
            attenMaskOffset = s2VSize * attenMaskDimS2;
        } else {
            attenMaskOffset = delta * attenMaskDimS2;
        }
    }
}


template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::CalcAttenMaskOffsetForPrefixCompressMode(int64_t &attenMaskOffset,
                                                                                              int64_t &attenMaskOffset2,
                                                                                              const int64_t delta,
                                                                                              uint32_t s1VSize,
                                                                                              uint32_t s2VSize,
                                                                                              uint32_t s2VBegin,
                                                                                              bool &canSimplify)
{
    /*
      prefix压缩attenmask形状:
      ||
      ||||
      ||||||            Causal
      ||||||||
      ||||              All Mask
      ||||

      s1 + N <= S2，等效于RightDownCausal
      S1 + N > S2 场景
      先推出映射在压缩Prefix下三角部分的Mask(Mask1)的偏移
      再推出映射在压缩Prefix矩形部分的Mask(Mask2)的偏移
      如果整个vector基本块在N范围内，则直接使用Mask2
    */

    canSimplify = false;

    int64_t S1 = static_cast<int64_t>(s1);
    int64_t S2 = static_cast<int64_t>(s2);
    uint32_t curBatchDimIdx = bDimIdx;
    if constexpr (INPUT_LAYOUT == TND) {
        curBatchDimIdx = bDimIdxTmp;
        GetSeqQlenKvlenByBidx(curBatchDimIdx, S1, S2);
    }

    int64_t N = ((__gm__ int64_t *)prefixN_addr)[curBatchDimIdx];

    // s1 + N <= s2, equivalent to RightDownCausal
    if (S1 + N <= S2) {
        canSimplify = true;
        int64_t causal_delta = delta - S1 + S2;
        CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1VSize, s2VSize);
        return;
    }

    int64_t delta1 = delta - S1 + S2;
    int64_t delta2 = N + 1 - static_cast<int64_t>(s2VBegin);

    // Y + n <= N, return mask2 offset directly
    if (delta2 > static_cast<int64_t>(s2VSize)) {
        canSimplify = true;
        attenMaskOffset = PREFIX_COMPRESS_CAUSAL_S_SIZE * attenMaskDimS2;
        return;
    }

    // other, mask = mask1 & mask2, need calculate two mask offsets
    // mask1 part
    if (delta1 >= 0) {
        attenMaskOffset = (delta1 <= s2VSize) ? delta1 * static_cast<int64_t>(attenMaskDimS2) :
                                                s2VSize * static_cast<int64_t>(attenMaskDimS2);
    } else {
        attenMaskOffset = (-delta1 <= s1VSize) ? -delta1 : s1VSize;
    }

    // mask2 part
    int64_t offsetStartPos =
        (int64_t)PREFIX_COMPRESS_CAUSAL_S_SIZE * (int64_t)attenMaskDimS2 + (int64_t)PREFIX_COMPRESS_ALL_MASK_S1_SIZE;
    attenMaskOffset2 = (delta2 > 0) ? (offsetStartPos - delta2 + 1) : offsetStartPos;
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2<
    T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::CalcAttenMaskOffsetWithSparseModeForUnpad(int64_t &attenMaskOffset, int64_t &attenMaskOffset2,
                                                               uint32_t s1VSize, uint32_t s2VSize, int64_t curS1Idx,
                                                               uint32_t s2VBegin, bool unpadUseBand, bool &canSimplify)
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    uint64_t compressMode = TilingData->s1s2BNGS1S2BaseParams.attenMaskCompressMode;
    int64_t causal_delta =
        static_cast<int64_t>(s1oDimIdxTmp * s1CvInner + curS1Idx * s1VecSize) - static_cast<int64_t>(s2VBegin);
    CalcAttenBandMode(compressMode, causal_delta);
    if (compressMode == 1 || (sparseMode == 8 && bDimIdxTmp != bandIdx)) { // causal s1==s2
        CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1VSize, s2VSize);
        return;
    }

    if (compressMode == 2 || (sparseMode == 7 && bDimIdxTmp != bandIdx)) { // causal s1!=s2
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        causal_delta = causal_delta - actualS1Len + actualS2Len;
        CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1VSize, s2VSize);
        return;
    }

    if (compressMode == 3 || unpadUseBand) { // band
        int64_t next_delta = causal_delta + actualCalcS2Token;
        CalcAttenMaskOffset(attenMaskOffset, next_delta, s1VSize, s2VSize);
        int64_t pre_delta = causal_delta - actualCalcS1Token - 1;
        CalcAttenMaskOffset(attenMaskOffset2, pre_delta, s1VSize, s2VSize);
        return;
    }

    if (compressMode == 4) { // 4: prefix compress
        CalcAttenMaskOffsetForPrefixCompressMode(attenMaskOffset, attenMaskOffset2, causal_delta, s1VSize, s2VSize,
                                                 s2VBegin, canSimplify);
        return;
    }

    if (TilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType == 0) {
        attenMaskDimS2 = (uint32_t)s2;
        attenMaskOffset += (static_cast<int64_t>(s1oDimIdxTmp) * s1CvInner + curS1Idx * s1VecSize) * s2 + s2VBegin;
    } else if (TilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType == 1) {
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        attenMaskDimS2 = (uint32_t)actualS2Len;
        for (uint32_t bidx = 0; bidx < bDimIdxTmp; bidx++) {
            GetSeqQlenKvlenByBidx(bidx, actualS1Len, actualS2Len);
            attenMaskOffset += actualS1Len * actualS2Len;
        }
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        attenMaskOffset += (static_cast<int64_t>(s1oDimIdxTmp) * s1CvInner + curS1Idx * s1VecSize) *
                           actualS2Len + s2VBegin;
    } else {
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        attenMaskDimS2 = (uint32_t)actualS2Len;
        for (uint32_t bidx = 0; bidx < bDimIdxTmp; bidx++) {
            GetSeqQlenKvlenByBidx(bidx, actualS1Len, actualS2Len);
            attenMaskOffset += static_cast<int64_t>(n2) * g * actualS1Len * actualS2Len;
        }
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        attenMaskOffset += ((static_cast<int64_t>(n2DimIdxTmp) * g + gDimIdxTmp) * actualS1Len +
                           s1oDimIdxTmp * s1CvInner + curS1Idx * s1VecSize) * actualS2Len + s2VBegin;
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::CalcAttenMaskOffsetWithSparseMode(int64_t &attenMaskOffset,
                                                                                       int64_t &attenMaskOffset2,
                                                                                       uint32_t s1VSize,
                                                                                       uint32_t s2VSize,
                                                                                       int64_t curS1Idx,
                                                                                       uint32_t s2VBegin,
                                                                                       bool &canSimplify)
{
    uint64_t compressMode = TilingData->s1s2BNGS1S2BaseParams.attenMaskCompressMode;
    int64_t causal_delta =
        static_cast<int64_t>(s1oDimIdx * s1CvInner + curS1Idx * s1VecSize) - static_cast<int64_t>(s2VBegin);
    CalcAttenBandMode(compressMode, causal_delta);
    if (compressMode == 1) { // 1: LeftUpCausal
        // causal s1==s2
        CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1VSize, s2VSize);
        return;
    }

    if (compressMode == 2) { // 2: RightDownCausal
        // causal s1!=s2
        causal_delta = causal_delta - s1 + s2;
        CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1VSize, s2VSize);
        return;
    }

    if (compressMode == 3) { // 3: band
        int64_t pre_delta = causal_delta - actualCalcS1Token - 1;
        CalcAttenMaskOffset(attenMaskOffset2, pre_delta, s1VSize, s2VSize);
        int64_t next_delta = causal_delta + actualCalcS2Token;
        CalcAttenMaskOffset(attenMaskOffset, next_delta, s1VSize, s2VSize);
        return;
    }

    if (compressMode == 4) { // 4: prefix compress
        CalcAttenMaskOffsetForPrefixCompressMode(attenMaskOffset, attenMaskOffset2, causal_delta, s1VSize, s2VSize,
                                                 s2VBegin, canSimplify);
        return;
    }

    if (TilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType == 0) {
        attenMaskOffset = (static_cast<int64_t>(s1oDimIdx) * s1CvInner + curS1Idx * s1VecSize) * s2 + s2VBegin;
    } else if (TilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType == 1) {
        attenMaskOffset =
            (static_cast<int64_t>(bDimIdx) * s1 + s1oDimIdx * s1CvInner + curS1Idx * s1VecSize) * s2 + s2VBegin;
    } else {
        attenMaskOffset = (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * s1 +
                           s1oDimIdx * s1CvInner + curS1Idx * s1VecSize) *
                              s2 +
                          s2VBegin;
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::NZCopyIn(int64_t mmAddr, GlobalTensor<T2> &mmWspGm,
                                                              LocalTensor<T2> &mmTensorCurr, uint32_t s1VecSize,
                                                              uint32_t s2VecSize)
{
    /*
    Func:
    MM输出NZ数据，数据搬运进UB
    */
    DataCopyParams intriParams;
    intriParams.blockCount = s2VecSize / C0_SIZE;
    intriParams.blockLen = s1VecSize * C0_SIZE / cal_block_num;
    intriParams.srcStride = s1CvExtend * C0_SIZE / cal_block_num - intriParams.blockLen;
    intriParams.dstStride = 1;
    DataCopy(mmTensorCurr, mmWspGm[mmAddr], intriParams);
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::NZ2ND(LocalTensor<T2> &mmTensorCurr, LocalTensor<T2> &tmpTensor,
                                                           uint32_t s1VecSize, uint32_t s2VecSize)
{
    /*
    Func:
    将NZ转为ND
    */
    CopyRepeatParams nz2ndParams;
    nz2ndParams.srcStride = s1VecSize * C0_SIZE / cal_block_num + 1;
    nz2ndParams.dstStride = C0_SIZE / cal_block_num;
    nz2ndParams.srcRepeatSize = C0_SIZE / cal_block_num;
    nz2ndParams.dstRepeatSize = s2VecSize / cal_block_num;

    uint16_t c0_repeat = C0_SIZE / cal_block_num;
    uint16_t c1_repeat = s2VecSize / C0_SIZE / VEC_REPEAT;
    uint16_t c1_remain = s2VecSize / C0_SIZE % VEC_REPEAT;
    uint16_t n_repeat = s1VecSize;
    for (uint16_t i = 0; i < c0_repeat; ++i) {
        for (uint16_t j = 0; j < c1_repeat; ++j) {
            Copy(mmTensorCurr[i * cal_block_num + j * VEC_REPEAT * C0_SIZE],
                 tmpTensor[i * cal_block_num + j * VEC_REPEAT * (s1VecSize * C0_SIZE + cal_block_num)],
                 VEC_REPEAT * cal_block_num, n_repeat, nz2ndParams);
        }
        if (c1_remain > 0) {
            Copy(mmTensorCurr[i * cal_block_num + c1_repeat * VEC_REPEAT * C0_SIZE],
                 tmpTensor[i * cal_block_num + c1_repeat * VEC_REPEAT * (s1VecSize * C0_SIZE + cal_block_num)],
                 VEC_REPEAT * c1_remain, n_repeat, nz2ndParams);
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::ND2NZ(LocalTensor<T1> &mmTensorCurr, LocalTensor<T1> &tmpTensor,
                                                           uint32_t s1VecSize, uint32_t s2VecSize)
{
    /*
    Func:
    将ND转为NZ
    */

    CopyRepeatParams nd2nzParams;
    nd2nzParams.dstStride = s1VecSize * C0_SIZE / input_block_num + 1;
    nd2nzParams.srcStride = C0_SIZE / input_block_num;
    nd2nzParams.dstRepeatSize = C0_SIZE / input_block_num;
    nd2nzParams.srcRepeatSize = s2VecSize / input_block_num;

    uint16_t c1_repeat = s2VecSize / C0_SIZE / VEC_REPEAT;
    uint16_t c1_remain = s2VecSize / C0_SIZE % VEC_REPEAT;

    auto mmTensorCurrTmp = mmTensorCurr.template ReinterpretCast<half>();
    auto tmpTensorTmp = tmpTensor.template ReinterpretCast<half>();

    for (uint16_t j = 0; j < c1_repeat; ++j) {
        Copy(mmTensorCurrTmp[j * 8 * (s1VecSize + 1) * C0_SIZE], tmpTensorTmp[j * 128], VEC_REPEAT * input_block_num,
             s1VecSize, nd2nzParams);
    }

    if (c1_remain > 0) {
        Copy(mmTensorCurrTmp[c1_repeat * 8 * (s1VecSize + 1) * C0_SIZE], tmpTensorTmp[c1_repeat * 128],
             input_block_num * c1_remain, s1VecSize, nd2nzParams);
    }
}


template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,
                                                           INPUT_LAYOUT, MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::InitIndex(int64_t index)
{
    if constexpr (INPUT_LAYOUT == TND) {
        // 安全防护，index不可小于0，防止n2DimIdx、gDimIdx等出现除0
        int64_t resbaseIdx = index < 0 ? 0 : index;
        if (resbaseIdx >= totalBaseIdx) {
            bDimIdx = isInitIndex ? (bDimIdx + 1) : 0;
            for (uint32_t bIdx = bDimIdx; bIdx < b; bIdx++) {
                GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
                s1OuterTmp = (actualS1Len + s1CvInner - 1) / s1CvInner;
                s2OuterTmp = (actualS2Len + s2CvInner - 1) / s2CvInner;
                preTotal = totalBaseIdx;
                totalBaseIdx += static_cast<int64_t>(n2) * g * s1OuterTmp * s2OuterTmp;
                if (resbaseIdx < totalBaseIdx) {
                    bDimIdx = bIdx;
                    break;
                }
            }
            isInitIndex = true;
        }
        uint32_t s1CvTailTmp = actualS1Len - (s1OuterTmp - 1) * s1CvInner;
        bDimTail = resbaseIdx - preTotal;
        n2DimIdx = bDimTail / (g * s1OuterTmp * s2OuterTmp);
        uint32_t n2DimTail = bDimTail % (g * s1OuterTmp * s2OuterTmp);
        gDimIdx = n2DimTail / (s1OuterTmp * s2OuterTmp);
        uint32_t gDimTail = n2DimTail % (s1OuterTmp * s2OuterTmp);
        s2oCvDimIdx = gDimTail / s1OuterTmp;
        s1oDimIdx = gDimTail % s1OuterTmp;
        s1CvExtend = (s1oDimIdx == s1OuterTmp - 1) ? s1CvTailTmp : s1CvInner;
    } else {
        baseIdx = index;
        bDimIdx = baseIdx / n2gs1os2o;
        uint32_t bDimTail = baseIdx % n2gs1os2o;
        n2DimIdx = bDimTail / gs1os2o;
        uint32_t n2DimTail = bDimTail % gs1os2o;
        gDimIdx = n2DimTail / s1os2o;
        uint32_t gDimTail = n2DimTail % s1os2o;
        s2oCvDimIdx = gDimTail / s1Outer;
        s1oDimIdx = gDimTail % s1Outer;
        s1CvExtend = (s1oDimIdx == s1Outer - 1) ? s1CvTail : s1CvInner;
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::CalcAttenBandMode(int64_t compressMode, int64_t causal_delta)
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    if (compressMode == 1 || compressMode == 2 || compressMode == 3 || sparseMode == 7 || sparseMode == 8) { // compress
        int64_t next_delta = causal_delta;
        int64_t pre_delta = causal_delta - INT32_MAX - 1;
        if (compressMode == 1 || (sparseMode == 8 && bDimIdxTmp != bandIdx)) {
        } else if (compressMode == 2) {
            if constexpr (INPUT_LAYOUT == TND) {
                GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
                next_delta = causal_delta - actualS1Len + actualS2Len;
            } else {
                next_delta = causal_delta - s1 + s2;
            }
        } else if (sparseMode == 7 && bDimIdxTmp != bandIdx) {
            GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
            next_delta = causal_delta - actualS1Len + actualS2Len;
        } else {
            next_delta = causal_delta + actualCalcS2Token;
            pre_delta = causal_delta - actualCalcS1Token - 1;
        }

        bool NoNext = (next_delta - s2Extend >= 0);
        bool NoPre = (pre_delta + 1 + s1ExtendSubGraph <= 0);

        if (NoNext && NoPre) {
            AttenBandMode = AttenMaskCompress::Empty;
        } else if (NoNext && !NoPre) {
            AttenBandMode = AttenMaskCompress::PreOnly;
        } else if (!NoNext && NoPre) {
            AttenBandMode = AttenMaskCompress::NextOnly;
        } else {
            AttenBandMode = AttenMaskCompress::All;
        }
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::DropOutCopy(LocalTensor<uint8_t> &vecInDropBuffer,
                                                                 int64_t curS1Idx, int64_t s2VBegin)
{
    // for compute dropout mask offset
    dropMaskInfo.s2StartIdx = s2VBegin;
    // for copy in dropout mask
    dropMaskInfo.s2CopySize = s2Extend;

    CopyInDropMask<true>(vecInDropBuffer, maskWorkSpaceGm, dropMaskGm, this->dropMaskInfo);
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::SubGrapA(int64_t curIdx, int64_t curS1Idx, int64_t curS2Idx,
                                    EvenvIdList &eventIdList)
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    bool isPing = ((curIdx % 2) == 0);
    s2Extend = (curS2Idx == s2VecLoop - 1) ? (s2CvExtend - (s2VecLoop - 1) * s2VecSize) : s2VecSize;
    s2ExtendAlign = (s2Extend + 15) / 16 * 16;
    uint32_t s2VBegin = preS2CvBegin + curS2Idx * s2VecSize;

    int32_t curEventIdMte2WaitMte3 = static_cast<int32_t>(isPing ?
                                     eventIdList.structMte2WaitMte3Ping : eventIdList.structMte2WaitMte3Pong);
    uint32_t ubBufferOffset = isPing ? 0 : DbBegin;

    if constexpr (TND_S1_PP == ENABLE) {
        int32_t mte2WaitVSfmg2A = static_cast<int32_t>(isPing ?
                                  eventIdList.structMte2WaitVPing : eventIdList.structMte2WaitVPong);
        AscendC::WaitFlag<HardEvent::V_MTE2>(mte2WaitVSfmg2A);
    } else {
        if (curIdx > 1) {
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(curEventIdMte2WaitMte3);
        }
    }

    LocalTensor<float> vecInBuffer3 =
        ubBuffer.GetWithOffset<float>(8 * 1024 / sizeof(float), ubBufferOffset + T2BlockBegin);
    int64_t softMaxOffset = 0;
    if constexpr (INPUT_LAYOUT == TND) {
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        if (tndSoftmaxIn) {
            int64_t innerRowOffsetLeft = unlikely(bDimIdxTmp == 0) ? 0 : ((__gm__ int64_t *)actual_seq_qlen_addr)[bDimIdxTmp - 1] * 32 / sizeof(float);
            int64_t originInnerBatchOffset = ((n2DimIdxTmp * g + gDimIdxTmp) * actualS1Len +
                            s1oDimIdxTmp * s1CvInner + curS1Idx * s1VecSize) * 32 / sizeof(float);
            softMaxOffset = ((((__gm__ int64_t *)actual_seq_qlen_addr)[b - 1] * 32 / sizeof(float)) * (n2DimIdxTmp * g + gDimIdxTmp) + innerRowOffsetLeft + originInnerBatchOffset % (actualS1Len * 32 / sizeof(float)));
        } else {
            if (bDimIdxTmp > 0) {
                softMaxOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[bDimIdxTmp - 1] * n2 * g * 32 / sizeof(float);
            }
            softMaxOffset += ((n2DimIdxTmp * g + gDimIdxTmp) * actualS1Len +
                            s1oDimIdxTmp * s1CvInner + curS1Idx * s1VecSize) * 32 / sizeof(float);
        }

    } else {
        softMaxOffset = (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * s1 + s1oDimIdx * s1CvInner +
                         curS1Idx * s1VecSize) * 32 / sizeof(float);
    }
    CopyInSoftMax(vecInBuffer3, s1ExtendSubGraph, softMaxOffset);

    LocalTensor<T1> pseUbT1 = ubBuffer.GetWithOffset<T1>(16 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
    LocalTensor<half> pseUb = pseUbT1.template ReinterpretCast<half>();
    if constexpr (IS_PSE == ENABLE) {
        pseInfo.bSSOffset = bDimIdx * s1 * s2;
        pseInfo.s2SizeAcc = bDimIdx * s2;
        pseInfo.boIdx = bDimIdx;
        pseInfo.n2oIdx = n2DimIdx;
        pseInfo.goIdx = gDimIdx;
        pseInfo.s1oIdx = s1oDimIdx;
        pseInfo.loopIdx = curS1Idx;
        pseInfo.vec1S1BaseSize = s1VecSize;
        pseInfo.vec1S1RealSize = s1ExtendSubGraph;
        pseInfo.s1BaseSize = s1CvInner;
        pseInfo.s2RealSize = s2Extend;
        pseInfo.s2AlignedSize = s2ExtendAlign;
        pseInfo.s2StartIdx = s2VBegin;
        LocalTensor<T2> noCastedPseUb = ubBuffer.GetWithOffset<T2>(0 / sizeof(T2), 0);
        bool innerAlibiFlag = false; // alibi核内生成相关配置，仅在LAYOUT=TND，SparseMode=8时生效
        if constexpr (INPUT_LAYOUT == TND) {
            if (sparseMode == 8 && pseInfo.boIdx != 0) {
                innerAlibiFlag = true;
            }
        }

        if (pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            if (innerAlibiFlag) {
                pseInfo.kvStartIdx = 0;
                pseInfo.qStartIdx = 0;
            }
            PseSlopeCopyIn<T2, true>(noCastedPseUb, pseUb, pseSlope, this->pseAlibiGm, pseInfo);
        } else {
            if constexpr (!IsSameType<T1, float>::value) {
                if constexpr (INPUT_LAYOUT == TND) {
                    PseCopyIn<T1, T2, LayOutTypeEnum::LAYOUT_TND, true>(noCastedPseUb, pseUbT1, this->pseGm, pseInfo);
                } else {
                    PseCopyIn<T1, T2, LayOutTypeEnum::LAYOUT_BNSD, true>(noCastedPseUb, pseUbT1, this->pseGm, pseInfo);
                }
            }
        }
    }

    LocalTensor<uint8_t> attenMaskUbuint8 =
        ubBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + BoolBegin);
    bool unpadUseBand = (sparseMode == 7 && bDimIdxTmp == bandIdx) || (sparseMode == 8 && bDimIdxTmp == bandIdx);
    int64_t attenMaskOffsetPre = 0;
    bool prefixCompressCanSimplify = false;
    if constexpr (IS_ATTEN_MASK == ENABLE) {
        int64_t attenMaskOffset = 0;
        if constexpr (INPUT_LAYOUT == TND) {
            CalcAttenMaskOffsetWithSparseModeForUnpad(attenMaskOffset, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend,
                                                      curS1Idx, s2VBegin, unpadUseBand, prefixCompressCanSimplify);
        } else {
            CalcAttenMaskOffsetWithSparseMode(attenMaskOffset, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend, curS1Idx,
                                              s2VBegin, prefixCompressCanSimplify);
        }
        // uint8_t
        if (AttenBandMode == AttenMaskCompress::All || AttenBandMode == AttenMaskCompress::NextOnly) {
            CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffset, s1ExtendSubGraph, s2Extend);
        } else if (AttenBandMode == AttenMaskCompress::PreOnly) {
            CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend);
        }
    }

    LocalTensor<uint8_t> vecInDropBuffer =
        ubBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + U8Begin);
    if constexpr (IS_DROP == ENABLE) {
        if constexpr (IsSameType<T1, float>::value) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        DropOutCopy(vecInDropBuffer, curS1Idx, s2VBegin);
    }

    LocalTensor<float> vecClc2Buffer =
        ubBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), ubBufferOffset + T2Begin);
    if constexpr (MM_OUT_FORMAT == CubeFormat::ND) {
        if (s2VecLoop == 1) {
            DataCopy(vecClc2Buffer, mm2WorkspaceGm[curS1Idx * s1VecSize * s2ExtendAlign],
                     s1ExtendSubGraph * s2ExtendAlign);
        } else {
            DataCopyPad(vecClc2Buffer, mm2WorkspaceGm[curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                        {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                         static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(float)), 0},
                        {false, 0, 0, 0});
        }
        int32_t vWaitMte2 = static_cast<int32_t>(isPing ?
                                     eventIdList.structVWaitMte2Ping : eventIdList.structVWaitMte2Pong);
        AscendC::SetFlag<HardEvent::MTE2_V>(vWaitMte2);
        AscendC::WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
    } else {
        int64_t mmAddr = curS1Idx * s1VecSize * C0_SIZE + curS2Idx * s1CvExtend * s2VecSizeAlign;
        NZCopyIn(mmAddr, mm2WorkspaceGm, vecClc2Buffer, s1VecSize, s2ExtendAlign);
        int32_t vWaitMte2 = static_cast<int32_t>(isPing ?
                                     eventIdList.structVWaitMte2Ping : eventIdList.structVWaitMte2Pong);
        AscendC::SetFlag<HardEvent::MTE2_V>(vWaitMte2);
        AscendC::WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
        auto tmpTensor = tmpBuffer.Get<T2>();
        DataCopy(tmpTensor, vecClc2Buffer, s1VecSize * s2ExtendAlign + s2ExtendAlign / C0_SIZE * VEC_REPEAT);
        AscendC::PipeBarrier<PIPE_V>();
        NZ2ND(vecClc2Buffer, tmpTensor, s1VecSize, s2ExtendAlign);
    }

    ///////////////////////////////////////////////////////////////
    // pse + muls
    ///////////////////////////////////////////////////////////////
    // pse shape  0--BN2G1S2    1--BN2GS1S2
    if constexpr (IS_PSE == ENABLE) {
        if (TilingData->s1s2BNGS1S2BaseParams.pseType != (uint32_t)PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
        AscendC::PipeBarrier<PIPE_V>();
        Muls(vecClc2Buffer, vecClc2Buffer, (T2)(TilingData->s1s2BNGS1S2BaseParams.scaleValue),
            s1ExtendSubGraph * s2ExtendAlign);
        }
        uint16_t repeatTimes = static_cast<uint16_t>(s1ExtendSubGraph);
        if (TilingData->s1s2BNGS1S2BaseParams.pseShapeType == 1) {
            repeatTimes = 1;
        }
        LocalTensor<T2> castTensor = tmpBuffer.Get<T2>();
        if (!(pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE)) {

            if constexpr (!IsSameType<T1, float>::value) {
                uint32_t calculateRowsAlign = (s2Extend + input_block_num - 1) / input_block_num * input_block_num;
                Cast(castTensor, pseUbT1, RoundMode::CAST_NONE, repeatTimes * calculateRowsAlign);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                int32_t mte2WaitV = static_cast<int32_t>(isPing ?
                                     eventIdList.structMte2WaitVPing : eventIdList.structMte2WaitVPong);
                AscendC::SetFlag<HardEvent::V_MTE2>(mte2WaitV);
                AscendC::WaitFlag<HardEvent::V_MTE2>(mte2WaitV);
                if constexpr (INPUT_LAYOUT == TND) {
                    PseCopyIn<T1, T2, LayOutTypeEnum::LAYOUT_TND, true>(castTensor, castTensor, this->pseGm, pseInfo);
                } else {
                    PseCopyIn<T1, T2, LayOutTypeEnum::LAYOUT_BNSD, true>(castTensor, castTensor, this->pseGm, pseInfo);
                }
                int32_t vWaitMte2 = static_cast<int32_t>(isPing ?
                                     eventIdList.structVWaitMte2Ping : eventIdList.structVWaitMte2Pong);
                AscendC::SetFlag<HardEvent::MTE2_V>(vWaitMte2);
                AscendC::WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
            }
        } else {
            PseSlopeCast<T2, true>(castTensor, pseUb, pseSlope, pseInfo);
        }
        AscendC::PipeBarrier<PIPE_V>();
        PseCompute<T2, true>(vecClc2Buffer, castTensor, pseInfo);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (TilingData->s1s2BNGS1S2BaseParams.pseType == (uint32_t)PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
        AscendC::PipeBarrier<PIPE_V>();
        Muls(vecClc2Buffer, vecClc2Buffer, (T2)(TilingData->s1s2BNGS1S2BaseParams.scaleValue),
            s1ExtendSubGraph * s2ExtendAlign);
    }
    ///////////////////////////////////////////////////////////////
    // attenMask
    ///////////////////////////////////////////////////////////////
    // attenMaskOffset     attenMaskShapeType  0--111S1S2        1--B11S1S2         2--BN2GS1S2
    if constexpr (IS_ATTEN_MASK == ENABLE) {
        int64_t compressMode = TilingData->s1s2BNGS1S2BaseParams.attenMaskCompressMode;
        AscendC::PipeBarrier<PIPE_V>();

        if (compressMode == 4) { // 4: prefix compress
            if (prefixCompressCanSimplify == false) {
                uint32_t ubTmpBufferOffset = isPing ? 0 : hufTmpBuffBegin;
                LocalTensor<uint8_t> attenMaskUbPreuint8 =
                    tmpBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubTmpBufferOffset);
                uint32_t s2ExtendPadAlign = (s2Extend + 31) / 32 * 32; // attenmask做pad时会32对齐，故加31/32做ceil
                int32_t maskNum = s1ExtendSubGraph * s2ExtendPadAlign / 2; // 除2数据量按照uint16类型折半

                int32_t mte2WaitV = static_cast<int32_t>(isPing ?
                                     eventIdList.structMte2WaitVPing : eventIdList.structMte2WaitVPong);
                AscendC::SetFlag<HardEvent::V_MTE2>(mte2WaitV);
                AscendC::WaitFlag<HardEvent::V_MTE2>(mte2WaitV);
                CopyInAttenMaskBool(attenMaskUbPreuint8, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend);

                int32_t vWaitMte2 = static_cast<int32_t>(isPing ?
                                     eventIdList.structVWaitMte2Ping : eventIdList.structVWaitMte2Pong);
                AscendC::SetFlag<HardEvent::MTE2_V>(vWaitMte2);
                AscendC::WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
                auto attenMaskUbuint8Tmp = attenMaskUbuint8.ReinterpretCast<uint16_t>();
                auto attenMaskUbPreuint8Tmp = attenMaskUbPreuint8.ReinterpretCast<uint16_t>();
                And(attenMaskUbuint8Tmp, attenMaskUbPreuint8Tmp, attenMaskUbuint8Tmp, maskNum);
                AscendC::PipeBarrier<PIPE_V>();
                attenMaskUbuint8 = attenMaskUbuint8Tmp.ReinterpretCast<uint8_t>();
            }
        }

        // uint8_t
        if (AttenBandMode == AttenMaskCompress::All || AttenBandMode == AttenMaskCompress::NextOnly) {
            CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8, s1ExtendSubGraph, s2ExtendAlign);
        } else if (AttenBandMode == AttenMaskCompress::PreOnly) {
            CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8, s1ExtendSubGraph, s2ExtendAlign, 1);
        }

        if ((compressMode == 3 || unpadUseBand) && AttenBandMode == AttenMaskCompress::All) { // 3: band
            int32_t mte2WaitV = static_cast<int32_t>(isPing ?
                                     eventIdList.structMte2WaitVPing : eventIdList.structMte2WaitVPong);
            AscendC::SetFlag<HardEvent::V_MTE2>(mte2WaitV);
            AscendC::WaitFlag<HardEvent::V_MTE2>(mte2WaitV);
            CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend);
            int32_t vWaitMte2 = static_cast<int32_t>(isPing ?
                                     eventIdList.structVWaitMte2Ping : eventIdList.structVWaitMte2Pong);
            AscendC::SetFlag<HardEvent::MTE2_V>(vWaitMte2);
            AscendC::WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
            CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8, s1ExtendSubGraph, s2ExtendAlign, 1);
        }
    }

    ///////////////////////////////////////////////////////////////
    // simpleSoftMax
    ///////////////////////////////////////////////////////////////
    AscendC::PipeBarrier<PIPE_V>();
    CalcSoftMax(vecClc2Buffer, vecInBuffer3, s1ExtendSubGraph, s2Extend, s2ExtendAlign, TilingData->softmaxTilingData);

    ///////////////////////////////////////////////////////////////
    // dropout
    ///////////////////////////////////////////////////////////////
    LocalTensor<T2> vecDropBuffer = vecClc2Buffer;
    if constexpr (IS_DROP == ENABLE) {
        vecDropBuffer = tmpBuffer.GetWithOffset<T2>(32 * 1024 / sizeof(T2), 0);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<uint8_t> tmpDropBuffer =
            ubBuffer.GetWithOffset<uint8_t>(32 * 1024 / sizeof(uint8_t), ubBufferOffset + T1Begin);

        // for compute dropout mask
        dropMaskInfo.lstAxis = s2ExtendAlign;
        dropMaskInfo.maskLstAxis = s2ExtendAlign;
        ComputeDropMask<T2, true>(vecDropBuffer, vecClc2Buffer, vecInDropBuffer, tmpDropBuffer, this->dropMaskInfo);
        if constexpr (IsSameType<T1, float>::value) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    ///////////////////////////////////////////////////////////////
    // cast fp322bf16
    ///////////////////////////////////////////////////////////////
    LocalTensor<T1> vecOut1Buffer1;
    if constexpr (!IsSameType<T1, float>::value) {
        vecOut1Buffer1 = ubBuffer.GetWithOffset<T1>(17 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(vecOut1Buffer1, vecDropBuffer, RoundMode::CAST_ROUND, s1ExtendSubGraph * s2ExtendAlign);
    }
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<T1> tmpTensor = tmpBuffer.Get<T1>();
        if constexpr (!IsSameType<T1, float>::value) {
            DataCopy(tmpTensor, vecOut1Buffer1, s1ExtendSubGraph * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
            ND2NZ(vecOut1Buffer1, tmpTensor, s1ExtendSubGraph, s2ExtendAlign);

            int32_t mte3WaitV = static_cast<int32_t>(isPing ?
                                     eventIdList.structMte3WaitVPing : eventIdList.structMte3WaitVPong);
            AscendC::SetFlag<HardEvent::V_MTE3>(mte3WaitV);
            AscendC::WaitFlag<HardEvent::V_MTE3>(mte3WaitV);
            DataCopyPad(dropWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                        curS1Idx * s1VecSize * C0_SIZE + curS2Idx * s1CvExtendAlign * s2VecSize],
                        vecOut1Buffer1,
                        {static_cast<uint16_t>(s2ExtendAlign / C0_SIZE),
                        static_cast<uint16_t>(s1ExtendSubGraph * C0_SIZE * sizeof(T1)), 1,
                        static_cast<uint16_t>((s1CvExtendAlign - s1ExtendSubGraph) * C0_SIZE * sizeof(T1))});
        } else {
            DataCopy(tmpTensor, vecDropBuffer, s1ExtendSubGraph * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
            ND2NZ(vecDropBuffer, tmpTensor, s1ExtendSubGraph, s2ExtendAlign);

            int32_t mte3WaitV = static_cast<int32_t>(isPing ?
                                     eventIdList.structMte3WaitVPing : eventIdList.structMte3WaitVPong);
            AscendC::SetFlag<HardEvent::V_MTE3>(mte3WaitV);
            AscendC::WaitFlag<HardEvent::V_MTE3>(mte3WaitV);
            DataCopyPad(dropWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                        curS1Idx * s1VecSize * C0_SIZE + curS2Idx * s1CvExtendAlign * s2VecSize],
                        vecDropBuffer,
                        {static_cast<uint16_t>(s2ExtendAlign / C0_SIZE),
                        static_cast<uint16_t>(s1ExtendSubGraph * C0_SIZE * sizeof(T1)), 1,
                        static_cast<uint16_t>((s1CvExtendAlign - s1ExtendSubGraph) * C0_SIZE * sizeof(T1))});
        }
    } else {
        int32_t mte3WaitV = static_cast<int32_t>(isPing ?
                                     eventIdList.structMte3WaitVPing : eventIdList.structMte3WaitVPong);
        AscendC::SetFlag<HardEvent::V_MTE3>(mte3WaitV);
        AscendC::WaitFlag<HardEvent::V_MTE3>(mte3WaitV);
        if constexpr (!IsSameType<T1, float>::value) {
                DataCopyPad(dropWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                    curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                    vecOut1Buffer1,
                    {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)), 0,
                     static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(T1))});
        } else {
                DataCopyPad(dropWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                    curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                    vecDropBuffer,
                    {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)), 0,
                     static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(T1))});
        }
    }
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(curEventIdMte2WaitMte3);
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT,
                                    MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::SubGrapB(int64_t curIdx, int64_t curS1Idx, int64_t curS2Idx,
                                    EvenvIdList &eventIdList)
{
    bool isPing = ((curIdx % 2) == 0);
    int32_t curEventId = static_cast<int32_t>(isPing ?
                                     eventIdList.structMte2WaitMte3Ping : eventIdList.structMte2WaitMte3Pong);
    uint32_t ubBufferOffset = isPing ? 0 : DbBegin;
    s2Extend = (curS2Idx == s2VecLoop - 1) ? (s2CvExtend - (s2VecLoop - 1) * s2VecSize) : s2VecSize;
    s2ExtendAlign = (s2Extend + 15) / 16 * 16;

    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(curEventId);

    LocalTensor<uint8_t> vecInDropBuffer =
        ubBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + U8Begin);
    if constexpr (IS_DROP == ENABLE) {
        int64_t s2VBegin = preS2CvBegin + curS2Idx * s2VecSize;
        DropOutCopy(vecInDropBuffer, curS1Idx, s2VBegin);
        if constexpr (IsSameType<T1, float>::value) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    LocalTensor<T2> vecClc1Buffer = ubBuffer.GetWithOffset<T2>(32 * 1024 / sizeof(T2), ubBufferOffset + T1Begin);
    if constexpr (MM_OUT_FORMAT == CubeFormat::ND) {
        if (s2VecLoop == 1) {
            DataCopy(vecClc1Buffer, mm1WorkspaceGm[curS1Idx * s1VecSize * s2ExtendAlign],
                     s1ExtendSubGraph * s2ExtendAlign);
        } else {
            DataCopyPad(vecClc1Buffer, mm1WorkspaceGm[curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                        {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                         static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(float)), 0},
                        {false, 0, 0, 0});
        }
        int32_t vWaitMte2 = static_cast<int32_t>(isPing ?
                                     eventIdList.structVWaitMte2Ping : eventIdList.structVWaitMte2Pong);
        AscendC::SetFlag<HardEvent::MTE2_V>(vWaitMte2);
        AscendC::WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
    } else {
        int64_t mmAddr = curS1Idx * s1VecSize * C0_SIZE + curS2Idx * s1CvExtend * s2VecSizeAlign;
        NZCopyIn(mmAddr, mm1WorkspaceGm, vecClc1Buffer, s1VecSize, s2ExtendAlign);
        int32_t vWaitMte2 = static_cast<int32_t>(isPing ?
                                     eventIdList.structVWaitMte2Ping : eventIdList.structVWaitMte2Pong);
        AscendC::SetFlag<HardEvent::MTE2_V>(vWaitMte2);
        AscendC::WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
        auto tmpTensor = tmpBuffer.Get<T2>();
        DataCopy(tmpTensor, vecClc1Buffer, s1VecSize * s2ExtendAlign + s2ExtendAlign / C0_SIZE * VEC_REPEAT);
        AscendC::PipeBarrier<PIPE_V>();
        NZ2ND(vecClc1Buffer, tmpTensor, s1VecSize, s2ExtendAlign);
    }

    ///////////////////////////////////////////////////////////////
    // ss
    ///////////////////////////////////////////////////////////////
    if constexpr (IS_DROP == ENABLE) {
        LocalTensor<uint8_t> tmpDropBuffer = tmpBuffer.GetWithOffset<uint8_t>(32 * 1024 / sizeof(uint8_t), 0);

        // for compute dropout mask
        dropMaskInfo.lstAxis = s2ExtendAlign;
        dropMaskInfo.maskLstAxis = s2ExtendAlign;
        ComputeDropMask<T2, true>(vecClc1Buffer, vecClc1Buffer, vecInDropBuffer, tmpDropBuffer, this->dropMaskInfo);
    }

    ///////////////////////////////////////////////////////////////
    // sub to improve
    ///////////////////////////////////////////////////////////////
    uint32_t sub_block_cout = (s2ExtendAlign + cal_repeat_num - 1) / cal_repeat_num;
    uint32_t sfmgStartIndex = s1CvRatio > 1 ? 0 : curS1Idx * s1VecSize * 8;

    LocalTensor<float> sfmgClc3;
    if constexpr (TND_S1_PP == ENABLE) {
        sfmgClc3 = vecClc3.GetWithOffset<float>(SFMG_RESULT_DB_SIZE / sizeof(float),
            isPing ? 0 : SFMG_RESULT_DB_SIZE);
    } else {
        sfmgClc3 = vecClc3.Get<float>();
    }

    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t subIdx = 0; subIdx < sub_block_cout; subIdx++) {
        uint32_t subMaskCout =
            (subIdx == sub_block_cout - 1) ? (s2ExtendAlign - subIdx * cal_repeat_num) : cal_repeat_num;
        Sub(vecClc1Buffer[subIdx * cal_repeat_num], vecClc1Buffer[subIdx * cal_repeat_num], sfmgClc3[sfmgStartIndex],
            subMaskCout, s1ExtendSubGraph,
            {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
             static_cast<uint8_t>(s2ExtendAlign / 8), 1});
    }

    ///////////////////////////////////////////////////////////////
    // mul
    ///////////////////////////////////////////////////////////////
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<float> vecClc2Buffer =
        ubBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), ubBufferOffset + T2Begin);
    Mul(vecClc1Buffer, vecClc1Buffer, vecClc2Buffer, s1ExtendSubGraph * s2ExtendAlign);
    LocalTensor<T1> vecOutBuffer;
    if constexpr (!IsSameType<T1, float>::value) {
        vecOutBuffer = ubBuffer.GetWithOffset<T1>(17 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(vecOutBuffer, vecClc1Buffer, RoundMode::CAST_ROUND, s1ExtendSubGraph * s2ExtendAlign);
    }
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        AscendC::PipeBarrier<PIPE_V>();
        auto tmpTensor1 = tmpBuffer.Get<T1>();
        if constexpr (IsSameType<T1, float>::value) {
            DataCopy(tmpTensor1, vecClc1Buffer, s1ExtendSubGraph * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
            ND2NZ(vecClc1Buffer, tmpTensor1, s1ExtendSubGraph, s2ExtendAlign);
        } else {
            DataCopy(tmpTensor1, vecOutBuffer, s1ExtendSubGraph * s2ExtendAlign);
            AscendC::PipeBarrier<PIPE_V>();
            ND2NZ(vecOutBuffer, tmpTensor1, s1ExtendSubGraph, s2ExtendAlign);
        }

        int32_t mte3WaitV = static_cast<int32_t>(isPing ?
                                     eventIdList.structMte3WaitVPing : eventIdList.structMte3WaitVPong);
        AscendC::SetFlag<HardEvent::V_MTE3>(mte3WaitV);
        AscendC::WaitFlag<HardEvent::V_MTE3>(mte3WaitV);

        if constexpr(IsSameType<T1, float>::value){
            DataCopyPad(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                   curS1Idx * s1VecSize * C0_SIZE + curS2Idx * s1CvExtendAlign * s2VecSize],
            vecClc1Buffer,
            {static_cast<uint16_t>(s2ExtendAlign / C0_SIZE),
                static_cast<uint16_t>(s1ExtendSubGraph * C0_SIZE * sizeof(T1)), 1,
            static_cast<uint16_t>((s1CvExtendAlign - s1ExtendSubGraph) * C0_SIZE * sizeof(T1))});
        } else {
            DataCopyPad(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                   curS1Idx * s1VecSize * C0_SIZE + curS2Idx * s1CvExtendAlign * s2VecSize],
            vecOutBuffer,
            {static_cast<uint16_t>(s2ExtendAlign / C0_SIZE),
                static_cast<uint16_t>(s1ExtendSubGraph * C0_SIZE * sizeof(T1)), 1,
            static_cast<uint16_t>((s1CvExtendAlign - s1ExtendSubGraph) * C0_SIZE * sizeof(T1))});
        }
    } else {
        int32_t mte3WaitV = static_cast<int32_t>(isPing ?
                                     eventIdList.structMte3WaitVPing : eventIdList.structMte3WaitVPong);
        AscendC::SetFlag<HardEvent::V_MTE3>(mte3WaitV);
        AscendC::WaitFlag<HardEvent::V_MTE3>(mte3WaitV);

        if constexpr(IsSameType<T1, float>::value) {
               DataCopyPad(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                   curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                    vecClc1Buffer,
                    {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)), 0,
                     static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(T1))});
        } else {
                  DataCopyPad(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                                   curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                    vecOutBuffer,
                    {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)), 0,
                     static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(T1))});
        }
    }

    if ((s1VecLoop * s2VecLoop > 2) && (curIdx < (s1VecLoop * s2VecLoop - 2))) {
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(curEventId);
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,
                                                           INPUT_LAYOUT, MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::Compute(int64_t preIndex,
                                                                                                  int64_t nextIndex)
{
    pingpongIdx = 1 - pingpongIdx;
    if (isStart == 1) {
        InitIndex(preIndex);
    }
    bDimIdxTmp = bDimIdx;
    n2DimIdxTmp = n2DimIdx;
    gDimIdxTmp = gDimIdx;
    s1oDimIdxTmp = s1oDimIdx;
    int64_t mm1aTensorOffsetCv = 0;
    int64_t mm2aTensorOffsetCv = 0;
    int64_t bTensorOffsetCv = 0;
    int64_t mm1aTensorOffsetCv_rope = 0; //TODOGY use has_Rope to exclude
    int64_t mm2aTensorOffsetCv_rope = 0;
    int64_t bTensorOffsetCv_rope = 0;
    s2CvExtend = preS2CvEnd - preS2CvBegin;
    s2CvExtendAlign = (s2CvExtend + 15) / 16 * 16;
    s1CvExtendAlign = (s1CvExtend + 15) / 16 * 16;
    int64_t dqOffset = 0;
    int64_t dkvOffset = 0;
    int64_t dqRopeOffset = 0;
    int64_t dkRopeOffset = 0;
    int64_t dvOffset = 0;
    int64_t s1StrideSize = 0;
    int64_t s2StrideSize = 0;
    int64_t s1StrideSize_rope = 0;
    int64_t s2StrideSize_rope = 0;
    int64_t dAlign = (d + 15) / 16 * 16;
    int64_t rope_dAlign = 0;
    int64_t dAlign2 = (value_d + 15) / 16 * 16;
    int64_t actualS1Len = 0;
    int64_t actualS2Len;
    if constexpr (HAS_ROPE == ENABLE) {
        rope_dAlign = (rope_d + 15) / 16 * 16;
    }
    if constexpr (INPUT_LAYOUT == TND) {
        UpdateToken(bDimIdxTmp);
        if (bDimIdxTmp > 0) {
            mm1aTensorOffsetCv = ((__gm__ int64_t *)actual_seq_qlen_addr)[bDimIdxTmp - 1] * n2 * g * d;
            bTensorOffsetCv = ((__gm__ int64_t *)actual_seq_kvlen_addr)[bDimIdxTmp - 1] * n2 * d;
            if constexpr (HAS_ROPE == ENABLE) {
                mm1aTensorOffsetCv_rope = ((__gm__ int64_t *)actual_seq_qlen_addr)[bDimIdxTmp - 1] * n2 * g * rope_d;
                bTensorOffsetCv_rope = ((__gm__ int64_t *)actual_seq_kvlen_addr)[bDimIdxTmp - 1] * n2 * rope_d;
            }
        }
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
            dqOffset = mm1aTensorOffsetCv / d * dAlign;
            dkvOffset = bTensorOffsetCv / d * dAlign;
            dvOffset = bTensorOffsetCv / d * dAlign2;
            dqOffset += ((n2DimIdxTmp * g + gDimIdxTmp) * actualS1Len) * dAlign + s1oDimIdxTmp * s1CvInner * C0_SIZE;
            dkvOffset += (n2DimIdxTmp * actualS2Len) * dAlign + preS2CvBegin * C0_SIZE;
            
            if constexpr (HAS_ROPE == ENABLE) {
                dqRopeOffset = mm1aTensorOffsetCv_rope / rope_d * rope_dAlign;
                dkRopeOffset = bTensorOffsetCv_rope / rope_d * rope_dAlign;
                dqRopeOffset += ((n2DimIdxTmp * g + gDimIdxTmp) * actualS1Len) * rope_dAlign + s1oDimIdxTmp * s1CvInner * C0_SIZE;
                dkRopeOffset += (n2DimIdxTmp * actualS2Len) * rope_dAlign + preS2CvBegin * C0_SIZE;
            }
            dvOffset += (n2DimIdxTmp * actualS2Len) * dAlign2 + preS2CvBegin * C0_SIZE;
        }
        mm1aTensorOffsetCv +=
            ((static_cast<int64_t>(s1oDimIdxTmp) * s1CvInner * n2 + n2DimIdxTmp) * g + gDimIdxTmp) * d;
        bTensorOffsetCv += (preS2CvBegin * n2 + n2DimIdxTmp) * d;
        mm2aTensorOffsetCv = mm1aTensorOffsetCv;
        s1StrideSize = n2 * g * d;
        s2StrideSize = n2 * d;
        if constexpr (HAS_ROPE == ENABLE) {
            mm1aTensorOffsetCv_rope +=
                ((static_cast<int64_t>(s1oDimIdxTmp) * s1CvInner * n2 + n2DimIdxTmp) * g + gDimIdxTmp) * rope_d;
            bTensorOffsetCv_rope += (preS2CvBegin * n2 + n2DimIdxTmp) * rope_d;
            mm2aTensorOffsetCv_rope = mm1aTensorOffsetCv_rope;
            s1StrideSize_rope = n2 * g * rope_d;
            s2StrideSize_rope = n2 * rope_d;
        }
    } else {
        if constexpr (INPUT_LAYOUT == BNGSD) {
            mm1aTensorOffsetCv =
                (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * s1 + s1oDimIdx * s1CvInner) * d;
            mm2aTensorOffsetCv = mm1aTensorOffsetCv;
            bTensorOffsetCv = ((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * s2 + preS2CvBegin) * d;
            s1StrideSize = d;
            s2StrideSize = d;
            if constexpr (HAS_ROPE == ENABLE) {
                mm1aTensorOffsetCv_rope =
                    (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * s1 + s1oDimIdx * s1CvInner) * rope_d;
                mm2aTensorOffsetCv_rope = mm1aTensorOffsetCv_rope;
                bTensorOffsetCv_rope = ((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * s2 + preS2CvBegin) * rope_d;
                s1StrideSize_rope = rope_d;
                s2StrideSize_rope = rope_d;
            }
        } else if constexpr (INPUT_LAYOUT == SBNGD) {
            mm1aTensorOffsetCv =
                ((((static_cast<int64_t>(s1oDimIdx) * s1CvInner) * b + bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * d;
            mm2aTensorOffsetCv = mm1aTensorOffsetCv;
            bTensorOffsetCv = ((static_cast<int64_t>(preS2CvBegin) * b + bDimIdx) * n2 + n2DimIdx) * d;
            s1StrideSize = static_cast<int64_t>(b) * n2 * g * d;
            s2StrideSize = static_cast<int64_t>(b) * n2 * d;
            if constexpr (HAS_ROPE == ENABLE) {
                mm1aTensorOffsetCv_rope =
                    ((((static_cast<int64_t>(s1oDimIdx) * s1CvInner) * b + bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * rope_d;
                mm2aTensorOffsetCv_rope = mm1aTensorOffsetCv_rope;
                bTensorOffsetCv_rope = ((static_cast<int64_t>(preS2CvBegin) * b + bDimIdx) * n2 + n2DimIdx) * rope_d;
                s1StrideSize_rope = static_cast<int64_t>(b) * n2 * g * rope_d;
                s2StrideSize_rope = static_cast<int64_t>(b) * n2 * rope_d;
            }
        } else if constexpr (INPUT_LAYOUT == BSNGD) {
            mm1aTensorOffsetCv =
                (((static_cast<int64_t>(bDimIdx) * s1 + s1oDimIdx * s1CvInner) * n2 + n2DimIdx) * g + gDimIdx) * d;
            mm2aTensorOffsetCv = mm1aTensorOffsetCv;
            bTensorOffsetCv = ((static_cast<int64_t>(bDimIdx) * s2 + preS2CvBegin) * n2 + n2DimIdx) * d;
            s1StrideSize = n2 * g * d;
            s2StrideSize = n2 * d;
            if constexpr (HAS_ROPE == ENABLE) {
                mm1aTensorOffsetCv_rope =
                    (((static_cast<int64_t>(bDimIdx) * s1 + s1oDimIdx * s1CvInner) * n2 + n2DimIdx) * g + gDimIdx) * rope_d;
                mm2aTensorOffsetCv_rope = mm1aTensorOffsetCv_rope;
                bTensorOffsetCv_rope = ((static_cast<int64_t>(bDimIdx) * s2 + preS2CvBegin) * n2 + n2DimIdx) * rope_d;
                s1StrideSize_rope = n2 * g * rope_d;
                s2StrideSize_rope = n2 * rope_d;
            }
        }
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            dqOffset = (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * s1) * dAlign + s1oDimIdx * s1CvInner * C0_SIZE;
            dkvOffset = ((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * s2) * dAlign + preS2CvBegin * C0_SIZE;
            dvOffset = ((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * s2) * dAlign2 + preS2CvBegin * C0_SIZE;
            if constexpr (HAS_ROPE == ENABLE) {
                dqRopeOffset = (((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * s1) * rope_dAlign + s1oDimIdx * s1CvInner * C0_SIZE;
                dkRopeOffset = ((static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * s2) * rope_dAlign + preS2CvBegin * C0_SIZE;
            }
        }
    }
    if constexpr (MM2_OUT_FORMAT == CubeFormat::ND) {
        dqOffset = mm1aTensorOffsetCv;
        dkvOffset = bTensorOffsetCv;
        dvOffset = bTensorOffsetCv / d * value_d;
        if constexpr (HAS_ROPE == ENABLE) {
            dqRopeOffset = mm1aTensorOffsetCv_rope;
            dkRopeOffset = bTensorOffsetCv_rope;
        }
    }

    if (isStart == 1) {
        if constexpr (INPUT_LAYOUT == TND) {
            GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
            if (MM_OUT_FORMAT == CubeFormat::NZ) {
                mm1.SetOrgShape(s1CvExtend, s2CvExtend, s1StrideSize / d * value_d, s2StrideSize / d * value_d, s2CvExtendAlign);
            } else {
                mm1.SetOrgShape(actualS1Len, actualS2Len, s1StrideSize / d * value_d, s2StrideSize / d * value_d, s2CvExtendAlign);
            }
        } else {
            if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
                mm1.SetOrgShape(s1CvExtend, s2, s1StrideSize / d * value_d, s2StrideSize / d * value_d, s2CvExtendAlign);
            } else {
                mm1.SetOrgShape(s1, s2, s1StrideSize / d * value_d, s2StrideSize / d * value_d, s2CvExtendAlign);
            }
        }
        mm1.SetTail(s1CvExtend, s2CvExtend, value_d); // M N K
        mm1.SetTensorA(dxGm[mm1aTensorOffsetCv / d * value_d]);
        mm1.SetTensorB(valueGm[bTensorOffsetCv / d * value_d], true);
        mm1.template IterateAll<false>(mm1WorkspaceGm, false, false, true);

        if constexpr (INPUT_LAYOUT == TND) {
            GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
            if (MM_OUT_FORMAT == CubeFormat::NZ) {
                mm1.SetOrgShape(s1CvExtend, s2CvExtend, s1StrideSize, s2StrideSize, s2CvExtendAlign);
            } else {
                mm1.SetOrgShape(actualS1Len, actualS2Len, s1StrideSize, s2StrideSize, s2CvExtendAlign);
            }
        } else {
            if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
                mm1.SetOrgShape(s1CvExtend, s2, s1StrideSize, s2StrideSize, s2CvExtendAlign);
            } else {
                mm1.SetOrgShape(s1, s2, s1StrideSize, s2StrideSize, s2CvExtendAlign);
            }
        }
        mm1.SetTail(s1CvExtend, s2CvExtend, d); // M N K
        mm1.SetTensorA(queryGm[mm2aTensorOffsetCv]);
        mm1.SetTensorB(keyGm[bTensorOffsetCv], true);
        mm1.template IterateAll<false>(mm2WorkspaceGm, false, false, true);

        if constexpr (HAS_ROPE == ENABLE) {
            if constexpr (INPUT_LAYOUT == TND) {
                GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
                if (MM_OUT_FORMAT == CubeFormat::NZ) {
                    mm1.SetOrgShape(s1CvExtend, s2CvExtend, s1StrideSize_rope, s2StrideSize_rope, s2CvExtendAlign);
                } else {
                    mm1.SetOrgShape(actualS1Len, actualS2Len, s1StrideSize_rope, s2StrideSize_rope, s2CvExtendAlign);
                }
            } else {
                if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
                    mm1.SetOrgShape(s1CvExtend, s2, s1StrideSize_rope, s2StrideSize_rope, s2CvExtendAlign);
                } else {
                    mm1.SetOrgShape(s1, s2, s1StrideSize_rope, s2StrideSize_rope, s2CvExtendAlign);
                }
            }
            mm1.SetTail(s1CvExtend, s2CvExtend, rope_d);
            mm1.SetTensorA(queryRopeGm[mm2aTensorOffsetCv_rope]);
            mm1.SetTensorB(keyRopeGm[bTensorOffsetCv_rope], true);
            mm1.template IterateAll<false>(mm2WorkspaceGm, true, false, true);
        }

        isStart = 0;
    }

    s2VecSize = s2CvExtend > VEC_S2_LEN ? VEC_S2_LEN : s2CvExtend;
    s2VecSize = s2VecSize == 0 ? cal_block_num : s2VecSize;
    s2VecLoop = (s2CvExtend + s2VecSize - 1) / s2VecSize;

    if constexpr (IS_DROP == ENABLE) {
        // dropout last dim 32B align
        s2VecSizeAlign = (s2VecSize + 31) / 32 * 32;
    } else if constexpr (IS_ATTEN_MASK == ENABLE) {
        // attenmask last dim 32B align
        s2VecSizeAlign = (s2VecSize + 31) / 32 * 32;
    } else {
        s2VecSizeAlign = (s2VecSize + 15) / 16 * 16;
    }

    if constexpr (TND_S1_PP == ENABLE) {
        // sfmg不切D,针对于Vector计算，重新计算s1VecSize（sfmgdInner不能超过4608）
        s1VecSize = SFMG_DB_INPUT_UBSIZE / calDtypeBytes / sfmgdInner;
        // SFMG计算时输入的数据行数，尽量是8的倍数，否则SoftmaxGradFront接口中会存在 scalar等
        // vector的wait等待，导致SFMG pingpong不起来，存在串行时序；
        s1VecSize = s1VecSize / SFMG_S1_ALIGN_NUM * SFMG_S1_ALIGN_NUM;
        s1VecSize = s1VecSize < s1CvExtend ? s1VecSize : s1CvExtend;
        s1VecSize = s1VecSize > SFMG_S1_ALIGN_NUM ? s1VecSize / SFMG_S1_ALIGN_NUM * SFMG_S1_ALIGN_NUM : s1VecSize;

        // 计算GraphAB过程中可以支持的最大s1VecSize，相较于sfmg计算出来的s1VecSize，取其较小值，并尽可能保证8对齐；
        uint32_t s1VecSizeGraphAB = baseMN / s2VecSizeAlign;
        s1VecSizeGraphAB = s1VecSizeGraphAB < s1CvExtend ? s1VecSizeGraphAB : s1CvExtend;
        s1VecSizeGraphAB = s1VecSizeGraphAB > 128 ? 128 : s1VecSizeGraphAB;
        s1VecSizeGraphAB = s1VecSizeGraphAB > SFMG_S1_ALIGN_NUM ? s1VecSizeGraphAB /
            SFMG_S1_ALIGN_NUM * SFMG_S1_ALIGN_NUM : s1VecSizeGraphAB;

        s1VecSize = s1VecSizeGraphAB < s1VecSize ? s1VecSizeGraphAB : s1VecSize;
        // 除零保护
        s1VecSize = s1VecSize == 0 ? 1 : s1VecSize;
        s1VecLoop = (s1CvExtend + s1VecSize - 1) / s1VecSize;
    } else {
        s1VecSize = baseMN / s2VecSizeAlign;
        s1VecSize = s1VecSize < s1CvExtend ? s1VecSize : s1CvExtend;
        s1VecSize = s1VecSize > 128 ? 128 : s1VecSize;
        s1VecLoop = (s1CvExtend + s1VecSize - 1) / s1VecSize;
    }

    dropMaskInfo.splitS1BaseSize = s1VecSize;
    if constexpr (INPUT_LAYOUT == TND) {
        GetSeqQlenKvlenByBidx(bDimIdx, actualS1Len, actualS2Len);
        dropMaskInfo.s2TotalSize = actualS2Len;
        int64_t bSSOffset = 0;
        int64_t s2Accu = 0;
        for (int64_t bidx = 0; bidx < bDimIdxTmp; bidx++) {
            GetSeqQlenKvlenByBidx(bidx, actualS1Len, actualS2Len);
            bSSOffset += actualS1Len * actualS2Len;
            s2Accu += actualS2Len;
        }
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        dropMaskInfo.bSSOffset = bSSOffset;
        dropMaskInfo.s1Size = actualS1Len;
        dropMaskInfo.s2Size = actualS2Len;
        pseInfo.bSSOffset = bSSOffset;
        pseInfo.s2SizeAcc = s2Accu;
        pseInfo.s1Size = dropMaskInfo.s1Size;
        pseInfo.s2Size = dropMaskInfo.s2Size;
    } else {
        dropMaskInfo.s2TotalSize = s2;
        dropMaskInfo.bSSOffset = bDimIdx * s1 * s2;
        pseInfo.s2SizeAcc = bDimIdx * s2;
        pseInfo.bSSOffset = dropMaskInfo.bSSOffset;
    }
    // for compute dropout mask offset
    dropMaskInfo.gOutIdx = gDimIdx;
    dropMaskInfo.n2OutIdx = n2DimIdx;
    dropMaskInfo.s1OutIdx = s1oDimIdx;

    ///////////////////////////////////////////////////////////////
    // SoftmaxGradFront
    ///////////////////////////////////////////////////////////////
    LocalTensor<float> sfmgClc3 = vecClc3.Get<float>();
    if (s1CvRatio <= 1) {
        CalcSoftMaxGrad(sfmgClc3, mm1aTensorOffsetCv / d * value_d, s1CvExtend);
    }

    mm1.WaitIterateAll();
    mm1.WaitIterateAll();
    if constexpr (HAS_ROPE == ENABLE) {
        mm1.WaitIterateAll();
    }

    if constexpr (TND_S1_PP == ENABLE) {
        uint32_t curIdxPing = 0;
        uint32_t curIdxPong = 0;
        uint32_t curS1IdxPing = 0;
        uint32_t curS1IdxPong = 0;

        for (uint32_t curS2Idx = 0; curS2Idx < s2VecLoop; curS2Idx++) {
            EvenvIdList eventIdList;
            LocalAllocEventID(eventIdList);

            for (uint32_t curS1Idx = 0; curS1Idx < s1VecLoop; curS1Idx = curS1Idx + 2) {
                curS1IdxPing = curS1Idx;
                curS1IdxPong = curS1Idx + 1;
                curIdxPing = curS2Idx * s1VecLoop + curS1IdxPing;
                curIdxPong = curS2Idx * s1VecLoop + curS1IdxPong;

                s1ExtendSubGraph = (curS1IdxPing == s1VecLoop - 1) ? (s1CvExtend - (s1VecLoop - 1) * s1VecSize) :
                    s1VecSize;
                int64_t sfmgOffset = 0;
                if (bDimIdxTmp > 0) {
                    sfmgOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[bDimIdxTmp - 1] * n2 * g * value_d;
                }
                sfmgOffset += ( ( ( static_cast<int64_t>(s1oDimIdxTmp) * s1CvInner + curS1IdxPing * s1VecSize ) 
                                  * n2 + n2DimIdxTmp ) * g + gDimIdxTmp ) * value_d;
                CalcSoftMaxGradPingPong(curIdxPing, sfmgOffset, s1ExtendSubGraph, eventIdList);

                if (curS1IdxPong < s1VecLoop) {
                    s1ExtendSubGraph = (curS1IdxPong == s1VecLoop - 1) ? (s1CvExtend - (s1VecLoop - 1) * s1VecSize) :
                        s1VecSize;
                    int64_t sfmgOffset = 0;
                    if (bDimIdxTmp > 0) {
                        sfmgOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[bDimIdxTmp - 1] * n2 * g * value_d;
                    }
                    sfmgOffset += ( ( ( static_cast<int64_t>(s1oDimIdxTmp) * s1CvInner + curS1IdxPong * s1VecSize ) 
                                      * n2 + n2DimIdxTmp ) * g + gDimIdxTmp ) * value_d;
                    CalcSoftMaxGradPingPong(curIdxPong, sfmgOffset, s1ExtendSubGraph, eventIdList);
                }

                s1ExtendSubGraph = (curS1IdxPing == s1VecLoop - 1) ? (s1CvExtend - (s1VecLoop - 1) * s1VecSize) :
                    s1VecSize;
                dropMaskInfo.s1CopySize = s1ExtendSubGraph;
                dropMaskInfo.s1InnerIdx = curS1IdxPing;
                dropMaskInfo.firstAxis = s1ExtendSubGraph;
                SubGrapA(curIdxPing, curS1IdxPing, curS2Idx, eventIdList);
                if (curS1IdxPong < s1VecLoop) {
                    s1ExtendSubGraph = (curS1IdxPong == s1VecLoop - 1) ? (s1CvExtend - (s1VecLoop - 1) * s1VecSize) :
                        s1VecSize;
                    dropMaskInfo.s1CopySize = s1ExtendSubGraph;
                    dropMaskInfo.s1InnerIdx = curS1IdxPong;
                    dropMaskInfo.firstAxis = s1ExtendSubGraph;
                    SubGrapA(curIdxPong, curS1IdxPong, curS2Idx, eventIdList);
                }

                s1ExtendSubGraph = (curS1IdxPing == s1VecLoop - 1) ? (s1CvExtend - (s1VecLoop - 1) * s1VecSize) :
                    s1VecSize;
                dropMaskInfo.s1CopySize = s1ExtendSubGraph;
                dropMaskInfo.s1InnerIdx = curS1IdxPing;
                dropMaskInfo.firstAxis = s1ExtendSubGraph;

                SubGrapB(curIdxPing, curS1IdxPing, curS2Idx, eventIdList);
                if (curS1IdxPong < s1VecLoop) {
                    s1ExtendSubGraph = (curS1IdxPong == s1VecLoop - 1) ? (s1CvExtend - (s1VecLoop - 1) * s1VecSize) :
                        s1VecSize;
                    dropMaskInfo.s1CopySize = s1ExtendSubGraph;
                    dropMaskInfo.s1InnerIdx = curS1IdxPong;
                    dropMaskInfo.firstAxis = s1ExtendSubGraph;
                    SubGrapB(curIdxPong, curS1IdxPong, curS2Idx, eventIdList);
                }
            }
            LocalReleaseEventID(eventIdList);
        }
    } else {
        uint32_t curIdxPing = 0;
        uint32_t curIdxPong = 0;
        uint32_t curS2IdxPing = 0;
        uint32_t curS2IdxPong = 0;
        for (uint32_t curS1Idx = 0; curS1Idx < s1VecLoop; curS1Idx++) {
            s1ExtendSubGraph = (curS1Idx == s1VecLoop - 1) ? (s1CvExtend - (s1VecLoop - 1) * s1VecSize) : s1VecSize;
            dropMaskInfo.s1CopySize = s1ExtendSubGraph;
            if (s1CvRatio > 1) {
                AscendC::PipeBarrier<PIPE_ALL>();
                int64_t sfmgOffset = 0;
                if constexpr (INPUT_LAYOUT == TND) {
                    if (bDimIdxTmp > 0) {
                        sfmgOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[bDimIdxTmp - 1] * n2 * g * value_d;
                    }
                    sfmgOffset += ( ( ( static_cast<int64_t>(s1oDimIdxTmp) * s1CvInner + curS1Idx * s1VecSize ) 
                                      * n2 + n2DimIdxTmp ) * g + gDimIdxTmp) * value_d;
                } else {
                    if constexpr (INPUT_LAYOUT == BNGSD) {
                        sfmgOffset = ( ( ( static_cast<int64_t>(bDimIdx) * n2 + n2DimIdx) * g + gDimIdx ) 
                                       * s1 + s1oDimIdx * s1CvInner + curS1Idx * s1VecSize ) * value_d;
                    } else if constexpr (INPUT_LAYOUT == SBNGD) {
                        sfmgOffset = ( ( ( ( static_cast<int64_t>(s1oDimIdx) * s1CvInner + curS1Idx * s1VecSize ) * b + bDimIdx ) 
                                         * n2 + n2DimIdx ) * g + gDimIdx ) * value_d;
                    } else if constexpr (INPUT_LAYOUT == BSNGD) {
                        sfmgOffset = ( ( ( static_cast<int64_t>(bDimIdx) * s1 + s1oDimIdx * s1CvInner + curS1Idx * s1VecSize ) 
                                         * n2 + n2DimIdx ) * g + gDimIdx ) * value_d;
                    }
                }
                LocalTensor<float> sfmgClc3 = vecClc3.Get<float>();
                CalcSoftMaxGrad(sfmgClc3, sfmgOffset, s1ExtendSubGraph);
            }

            // for compute dropout mask offset
            dropMaskInfo.s1InnerIdx = curS1Idx;
            // for compute dropout mask
            dropMaskInfo.firstAxis = s1ExtendSubGraph;

            EvenvIdList eventIdList;
            LocalAllocEventID(eventIdList);

            for (uint32_t curS2Idx = 0; curS2Idx < s2VecLoop; curS2Idx = curS2Idx + 2) {
                curS2IdxPing = curS2Idx;
                curS2IdxPong = curS2Idx + 1;
                curIdxPing = curS1Idx * s2VecLoop + curS2IdxPing;
                curIdxPong = curS1Idx * s2VecLoop + curS2IdxPong;
                SubGrapA(curIdxPing, curS1Idx, curS2IdxPing, eventIdList);
                if (curS2IdxPong < s2VecLoop) {
                    SubGrapA(curIdxPong, curS1Idx, curS2IdxPong, eventIdList);
                }

                SubGrapB(curIdxPing, curS1Idx, curS2IdxPing, eventIdList);
                if (curS2IdxPong < s2VecLoop) {
                    SubGrapB(curIdxPong, curS1Idx, curS2IdxPong, eventIdList);
                }
            }
            LocalReleaseEventID(eventIdList);
        }
    }

    uint32_t preS1Extend = s1CvExtend;
    if (nextIndex != 0) {
        InitIndex(nextIndex);
        int64_t nextS2CvExtend = nextS2CvEnd - nextS2CvBegin;
        int64_t nextS2CvExtendAlign = (nextS2CvExtend + 15) / 16 * 16;

        int64_t mm1aTensorOffsetCv1 = 0;
        int64_t bTensorOffsetCv1 = 0;
        AddMM1Offsets1(mm1aTensorOffsetCv1, bTensorOffsetCv1, this->value_d);

        if constexpr (INPUT_LAYOUT == TND) {
            GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
            if (MM_OUT_FORMAT == CubeFormat::NZ) {
                mm1.SetOrgShape(s1CvExtend, s2CvExtend, s1StrideSize / d * value_d, s2StrideSize / d * value_d, nextS2CvExtendAlign);
            } else {
                mm1.SetOrgShape(actualS1Len, actualS2Len, s1StrideSize / d * value_d, s2StrideSize / d * value_d, nextS2CvExtendAlign);
            }
        } else {
            if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
                mm1.SetOrgShape(s1CvExtend, s2, s1StrideSize / d * value_d, s2StrideSize / d * value_d, nextS2CvExtendAlign);
            } else {
                mm1.SetOrgShape(s1, s2, s1StrideSize / d * value_d, s2StrideSize / d * value_d, nextS2CvExtendAlign);
            }
        }
        mm1.SetTail(s1CvExtend, nextS2CvExtend, value_d);
        mm1.SetTensorA(dxGm[mm1aTensorOffsetCv1]);
        mm1.SetTensorB(valueGm[bTensorOffsetCv1], true);
        mm1.template IterateAll<false>(mm1WorkspaceGm, false, false, true);

        mm1aTensorOffsetCv1 = 0;
        bTensorOffsetCv1 = 0;
        AddMM1Offsets1(mm1aTensorOffsetCv1, bTensorOffsetCv1, this->d);

        if constexpr (INPUT_LAYOUT == TND) {
            GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
            if (MM_OUT_FORMAT == CubeFormat::NZ) {
                mm1.SetOrgShape(s1CvExtend, s2CvExtend, s1StrideSize, s2StrideSize, nextS2CvExtendAlign);
            } else {
                mm1.SetOrgShape(actualS1Len, actualS2Len, s1StrideSize, s2StrideSize, nextS2CvExtendAlign);
            }
        } else {
            if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
                mm1.SetOrgShape(s1CvExtend, s2, s1StrideSize, s2StrideSize, nextS2CvExtendAlign);
            } else {
                mm1.SetOrgShape(s1, s2, s1StrideSize, s2StrideSize, nextS2CvExtendAlign);
            }
        }
        mm1.SetTail(s1CvExtend, nextS2CvExtend, d);
        mm1.SetTensorA(queryGm[mm1aTensorOffsetCv1]);
        mm1.SetTensorB(keyGm[bTensorOffsetCv1], true);
        mm1.template IterateAll<false>(mm2WorkspaceGm, false, false, true);
        if constexpr (HAS_ROPE == ENABLE) {
            if constexpr (INPUT_LAYOUT == TND) {
                GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
                if (MM_OUT_FORMAT == CubeFormat::NZ) {
                    mm1.SetOrgShape(s1CvExtend, s2CvExtend, s1StrideSize_rope, s2StrideSize_rope, nextS2CvExtendAlign);
                } else {
                    mm1.SetOrgShape(actualS1Len, actualS2Len, s1StrideSize_rope, s2StrideSize_rope, nextS2CvExtendAlign);
                }
            } else {
                if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
                    mm1.SetOrgShape(s1CvExtend, s2, s1StrideSize_rope, s2StrideSize_rope, nextS2CvExtendAlign);
                } else {
                    mm1.SetOrgShape(s1, s2, s1StrideSize_rope, s2StrideSize_rope, nextS2CvExtendAlign);
                }
            }
            mm1.SetTail(s1CvExtend, nextS2CvExtend, rope_d);
            mm1.SetTensorA(queryRopeGm[mm1aTensorOffsetCv1 / d * rope_d]);
            mm1.SetTensorB(keyRopeGm[bTensorOffsetCv1 / d * rope_d], true);
            mm1.template IterateAll<false>(mm2WorkspaceGm, true, false, true);
        }
        
    }

    int64_t s1_size = s1;
    int64_t s1TndSize = actualS1Len;
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        s1_size = s1CvExtendAlign;
        s1TndSize = s1CvExtendAlign;
    }

    ///////////////////////////////////////////////////////////////
    // Matmal4 dq
    ///////////////////////////////////////////////////////////////
    // left [B, N2, G, S1, s2] right [B, N2, 1, S2, D] output [B, N2, G, S1, D]
    if constexpr (INPUT_LAYOUT == BNGSD) {
        if (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm4.SetOrgShape(s1_size, d, s2CvExtendAlign);
            mm4.SetSelfDefineData(s1);
        } else {
            mm4.SetOrgShape(s1_size, d, s2CvExtendAlign);
        }
    } else if constexpr (INPUT_LAYOUT == SBNGD) {
        if (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm4.SetOrgShape(s1_size, static_cast<int64_t>(b) * n2 * d, s2CvExtendAlign, s2, d);
            mm4.SetSelfDefineData(s1);
        } else {
            mm4.SetOrgShape(s1_size, static_cast<int64_t>(b) * n2 * d, s2CvExtendAlign, s2,
                        static_cast<int64_t>(b) * n2 * g * d);
        }
    } else if constexpr (INPUT_LAYOUT == BSNGD) {
        if (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm4.SetOrgShape(s1_size, n2 * d, s2CvExtendAlign, s2, d);
            mm4.SetSelfDefineData(s1);
        } else {
            mm4.SetOrgShape(s1_size, n2 * d, s2CvExtendAlign, s2, n2 * g * d);
        }
    } else if constexpr (INPUT_LAYOUT == TND) {
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm4.SetOrgShape(s1TndSize, n2 * d, s2CvExtendAlign, actualS2Len, d);
            mm4.SetSelfDefineData(actualS1Len);
        } else {
            mm4.SetOrgShape(s1TndSize, n2 * d, s2CvExtendAlign, actualS2Len, n2 * g * d);
        }
    }
    mm4.SetTail(preS1Extend, d, s2CvExtend); // M N K
    mm4.SetTensorA(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN]);
    mm4.SetTensorB(keyGm[bTensorOffsetCv]);
    mm4.template IterateAll<false>(dqWorkSpaceGm[dqOffset], true);
    mm4.End();

    ///////////////////////////////////////////////////////////////
    // Matmal4 dqRope
    ///////////////////////////////////////////////////////////////
    // left [B, N2, G, S1, s2] right [B, N2, 1, S2, D] output [B, N2, G, S1, D]
    if constexpr (HAS_ROPE == ENABLE) {
        if constexpr (INPUT_LAYOUT == BNGSD) {
            if (MM2_OUT_FORMAT == CubeFormat::NZ) {
                mm4.SetOrgShape(s1_size, rope_d, s2CvExtendAlign);
                mm4.SetSelfDefineData(s1);
            } else {
                mm4.SetOrgShape(s1_size, rope_d, s2CvExtendAlign);
            }
        } else if constexpr (INPUT_LAYOUT == SBNGD) {
            if (MM2_OUT_FORMAT == CubeFormat::NZ) {
                mm4.SetOrgShape(s1_size, static_cast<int64_t>(b) * n2 * rope_d, s2CvExtendAlign, s2, rope_d);
                mm4.SetSelfDefineData(s1);
            } else {
                mm4.SetOrgShape(s1_size, static_cast<int64_t>(b) * n2 * rope_d, s2CvExtendAlign, s2,
                            static_cast<int64_t>(b) * n2 * g * rope_d);
            }
        } else if constexpr (INPUT_LAYOUT == BSNGD) {
            if (MM2_OUT_FORMAT == CubeFormat::NZ) {
                mm4.SetOrgShape(s1_size, n2 * rope_d, s2CvExtendAlign, s2, rope_d);
                mm4.SetSelfDefineData(s1);
            } else {
                mm4.SetOrgShape(s1_size, n2 * rope_d, s2CvExtendAlign, s2, n2 * g * rope_d);
            }
        } else if constexpr (INPUT_LAYOUT == TND) {
            GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                mm4.SetOrgShape(s1TndSize, n2 * rope_d, s2CvExtendAlign, actualS2Len, rope_d);
                mm4.SetSelfDefineData(actualS1Len);
            } else {
                mm4.SetOrgShape(s1TndSize, n2 * rope_d, s2CvExtendAlign, actualS2Len, n2 * g * rope_d);
            }
        }
        mm4.SetTail(preS1Extend, rope_d, s2CvExtend); // M N K
        mm4.SetTensorA(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN]);
        mm4.SetTensorB(keyRopeGm[bTensorOffsetCv_rope]);
        mm4.template IterateAll<false>(dqRopeWorkSpaceGm[dqRopeOffset], true);
        mm4.End();
    }

    ///////////////////////////////////////////////////////////////
    // Matmal4 dk
    ///////////////////////////////////////////////////////////////
    // left [B, N2, G, S1, S2] right [B, N2, 1, S1, D] output [B, N2, G, S2, D]
    if constexpr (INPUT_LAYOUT == BNGSD) {
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm3.SetOrgShape(s2CvExtendAlign, d, s1_size);
            mm3.SetSelfDefineData(s2);
        } else {
            mm3.SetOrgShape(s2CvExtendAlign, d, s1_size);
        }
    } else if constexpr (INPUT_LAYOUT == SBNGD) {
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm3.SetOrgShape(s2CvExtendAlign, static_cast<int64_t>(b) * n2 * g * d, s1_size, s1, d);
            mm3.SetSelfDefineData(s2);
        } else {
            mm3.SetOrgShape(s2CvExtendAlign, static_cast<int64_t>(b) * n2 * g * d, s1_size, s1,
                        static_cast<int64_t>(b) * n2 * d);
        }
    } else if constexpr (INPUT_LAYOUT == BSNGD) {
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm3.SetOrgShape(s2CvExtendAlign, n2 * g * d, s1_size, s1, d);
            mm3.SetSelfDefineData(s2);
        } else {
            mm3.SetOrgShape(s2CvExtendAlign, n2 * g * d, s1_size, s1, n2 * d);
        }
    } else if constexpr (INPUT_LAYOUT == TND) {
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm3.SetOrgShape(s2CvExtendAlign, n2 * g * d, s1TndSize, actualS1Len, d);
            mm3.SetSelfDefineData(actualS2Len);
        } else {
            mm3.SetOrgShape(s2CvExtendAlign, n2 * g * d, s1TndSize, actualS1Len, n2 * d);
        }
    }
    mm3.SetTail(s2CvExtend, d, preS1Extend);
    mm3.SetTensorA(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN], true);
    mm3.SetTensorB(queryGm[mm2aTensorOffsetCv]);
    mm3.template IterateAll<false>(dkWorkSpaceGm[dkvOffset], true);
    mm3.End();

    ///////////////////////////////////////////////////////////////
    // Matmal5 dv
    ///////////////////////////////////////////////////////////////
    // left [B, N2, G, S1, S2] right [B, N2, G, S1, D2] output [B, N2, 1, S2, D2]
    if constexpr (INPUT_LAYOUT == BNGSD) {
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm3.SetOrgShape(s2CvExtendAlign, value_d, s1_size);
            mm3.SetSelfDefineData(s2);
        } else {
            mm3.SetOrgShape(s2CvExtendAlign, value_d, s1_size);
        }
    } else if constexpr (INPUT_LAYOUT == SBNGD) {
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm3.SetOrgShape(s2CvExtendAlign, static_cast<int64_t>(b) * n2 * g * value_d, s1_size, s1, value_d);
            mm3.SetSelfDefineData(s2);
        } else {
            mm3.SetOrgShape(s2CvExtendAlign, static_cast<int64_t>(b) * n2 * g * value_d, s1_size, s1,
                        static_cast<int64_t>(b) * n2 * value_d);
        }
    } else if constexpr (INPUT_LAYOUT == BSNGD) {
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm3.SetOrgShape(s2CvExtendAlign, n2 * g * value_d, s1_size, s1, value_d);
            mm3.SetSelfDefineData(s2);
        } else {
            mm3.SetOrgShape(s2CvExtendAlign, n2 * g * value_d, s1_size, s1, n2 * value_d);
        }
    } else if constexpr (INPUT_LAYOUT == TND) {
        GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm3.SetOrgShape(s2CvExtendAlign, n2 * g * value_d, s1TndSize, actualS1Len, value_d);
            mm3.SetSelfDefineData(actualS2Len);
        } else {
            mm3.SetOrgShape(s2CvExtendAlign, n2 * g * value_d, s1TndSize, actualS1Len, n2 * value_d);
        }
    }

    mm3.SetTail(s2CvExtend, value_d, preS1Extend);
    mm3.SetTensorA(dropWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN], true);
    mm3.SetTensorB(dxGm[mm1aTensorOffsetCv / d * value_d]);
    if constexpr (IsSameType<T1, float>::value) {
        if (nextIndex == 0) {
            mm3.template IterateAll<true>(dvGm[dvOffset], true);
        } else {
            mm3.template IterateAll<false>(dvGm[dvOffset], true);
        }
    } else {
        if (nextIndex == 0) {
            mm3.template IterateAll<true>(dvWorkSpaceGm[dvOffset], true);
        } else {
            mm3.template IterateAll<false>(dvWorkSpaceGm[dvOffset], true);
        }
    }

    mm3.End();

    ///////////////////////////////////////////////////////////////
    // Matmal4 dkRope
    ///////////////////////////////////////////////////////////////
    // left [B, N2, G, S1, S2] right [B, N2, 1, S1, D] output [B, N2, G, S2, D]
    if constexpr (HAS_ROPE == ENABLE) {
        if constexpr (INPUT_LAYOUT == BNGSD) {
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                mm3.SetOrgShape(s2CvExtendAlign, rope_d, s1_size);
                mm3.SetSelfDefineData(s2);
            } else {
                mm3.SetOrgShape(s2CvExtendAlign, rope_d, s1_size);
            }
        } else if constexpr (INPUT_LAYOUT == SBNGD) {
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                mm3.SetOrgShape(s2CvExtendAlign, static_cast<int64_t>(b) * n2 * g * rope_d, s1_size, s1, rope_d);
                mm3.SetSelfDefineData(s2);
            } else {
                mm3.SetOrgShape(s2CvExtendAlign, static_cast<int64_t>(b) * n2 * g * rope_d, s1_size, s1,
                            static_cast<int64_t>(b) * n2 * rope_d);
            }
        } else if constexpr (INPUT_LAYOUT == BSNGD) {
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                mm3.SetOrgShape(s2CvExtendAlign, n2 * g * rope_d, s1_size, s1, rope_d);
                mm3.SetSelfDefineData(s2);
            } else {
                mm3.SetOrgShape(s2CvExtendAlign, n2 * g * rope_d, s1_size, s1, n2 * rope_d);
            }
        } else if constexpr (INPUT_LAYOUT == TND) {
            GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Len, actualS2Len);
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                mm3.SetOrgShape(s2CvExtendAlign, n2 * g * rope_d, s1TndSize, actualS1Len, rope_d);
                mm3.SetSelfDefineData(actualS2Len);
            } else {
                mm3.SetOrgShape(s2CvExtendAlign, n2 * g * rope_d, s1TndSize, actualS1Len, n2 * rope_d);
            }
        }
        mm3.SetTail(s2CvExtend, rope_d, preS1Extend);
        mm3.SetTensorA(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN], true);
        mm3.SetTensorB(queryRopeGm[mm2aTensorOffsetCv_rope]);
        mm3.template IterateAll<false>(dkRopeWorkSpaceGm[dkRopeOffset], true);
        mm3.End();
    }
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,
                                                           INPUT_LAYOUT, MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::LocalAllocEventID(EvenvIdList &eventIdList)
{
    eventIdList.structVWaitMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    eventIdList.structVWaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    eventIdList.structMte2WaitMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    eventIdList.structMte2WaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    eventIdList.structMte3WaitVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    eventIdList.structMte3WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    eventIdList.structMte2WaitVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    eventIdList.structMte2WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,
                                                           INPUT_LAYOUT, MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::LocalReleaseEventID(EvenvIdList &eventIdList)
{
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdList.structVWaitMte2Ping);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdList.structVWaitMte2Pong);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdList.structMte2WaitMte3Ping);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdList.structMte2WaitMte3Pong);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdList.structMte3WaitVPing);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdList.structMte3WaitVPong);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdList.structMte2WaitVPing);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdList.structMte2WaitVPong);
}

template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK, const uint32_t IS_PSE, const uint32_t IS_DROP,
          const CubeFormat MM_OUT_FORMAT, const uint32_t INPUT_LAYOUT, const CubeFormat MM2_OUT_FORMAT, const uint32_t TND_S1_PP, const uint32_t HAS_ROPE>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2<T1, T2, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,
                                                           INPUT_LAYOUT, MM2_OUT_FORMAT, TND_S1_PP, HAS_ROPE>::SyncALLCores()
{
    SyncAll();
}

#endif // _FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2GS1S2_H_
