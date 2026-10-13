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
 * \file flash_attention_score_grad_s1s2_bn2gs1s2_sab.h
 * \brief
 */

#ifndef _FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2GS1S2_SAMEAB_H_
#define _FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2GS1S2_SAMEAB_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "pse.h"
#include "dropmask.h"
#include "./basic_modules/common_header.h"


using namespace matmul;

constexpr inline MatmulConfig SAB_NORM_DISABLE_INIT = {true,  false, false, 0,     0,     0,     false, false,
                                                   false, false, 0,     0,     0,     0,     0,     0,
                                                   0,     0,     true,  false, false, false, false, false};
constexpr MatmulConfig CFG_DIS_UNIT_FLAG_EXCEED = GetNormalConfig(true, false, false, BatchMode::BATCH_LESS_THAN_L1, true, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT, false);

struct DBParams {
  int64_t blockId;
  int64_t taskId;
  int64_t bIdx;
  int64_t n2Idx;
  int64_t s2oIdx;
  int64_t gIdx;
  int64_t s1oIdx;
  int32_t s1CvExtend;
  int32_t s2CvExtend;
  int32_t s1CvExtendAlign;
  int32_t s2CvExtendAlign;
  int64_t aTensorOffsetCv{0};
  int64_t aTensorOffsetCv_rope{0};
  int64_t bTensorOffsetCv{0};
  int64_t bTensorOffsetCv_rope{0};
  int64_t actualS1Len{0};
  int64_t actualS2Len{0};
  int64_t s1Stride;
  int64_t s2Stride;
  int64_t blockIdArr[24];  //确定性计算预留
  int32_t s1CvExtendArr[24];
  int32_t s2CvExtendArr[24];
  int8_t dqGroupId[24];
  int8_t kvGroupId[24];
};

struct IndexParams {
    int64_t bIdx;
    int64_t n2Idx;
    int64_t s2oIdx;
    int64_t gIdx;
    int64_t s1oIdx;
};

__aicore__ inline void DataCopyOutForNz(const __gm__ void *gm, const LocalTensor<int8_t> &co1Local,
                                   const void *dataCopyOutParams, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    const DataCopyOutParams *param = reinterpret_cast<const DataCopyOutParams *>(dataCopyOutParams);
    uint64_t dstStride = tilingPtr * 16 / 8 - param->burstLen;
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

// FAGType define
template <typename T1, typename T2, const uint32_t IS_ATTEN_MASK = 0, const uint32_t IS_PSE = 1,
          const uint32_t IS_DROP = 1, const CubeFormat MM_OUT_FORMAT = CubeFormat::ND, const uint32_t INPUT_LAYOUT = 0,
          const CubeFormat MM2_OUT_FORMAT = CubeFormat::NZ, const uint32_t IS_DTM = 0,
          const STemplateType S1TEMPLATETYPE = STemplateType::NotAligned,
          const STemplateType S2TEMPLATETYPE = STemplateType::NotAligned,
          const DTemplateType DTEMPLATETYPE = DTemplateType::NotAligned,
          const uint32_t HAS_ROPE = 0,
          typename... Args>
struct FAGType {
    using t1 = T1;
    using t2 = T2;
    static constexpr uint32_t isAttenMask = IS_ATTEN_MASK;
    static constexpr uint32_t isPse = IS_PSE;
    static constexpr uint32_t isDrop = IS_DROP;
    static constexpr CubeFormat mmOutFormat = MM_OUT_FORMAT;
    static constexpr uint32_t inputLayout = INPUT_LAYOUT;
    static constexpr CubeFormat mm2OutFormat = MM2_OUT_FORMAT;
    static constexpr uint32_t isDtm = IS_DTM;
    static constexpr STemplateType s1TemplateType = S1TEMPLATETYPE;
    static constexpr STemplateType s2TemplateType = S2TEMPLATETYPE;
    static constexpr DTemplateType dTemplateType = DTEMPLATETYPE;
    static constexpr uint32_t hasRope = HAS_ROPE;
};

template <typename FAGT>
class FlashAttentionScoreGradS1s2Bn2gs1s2SameAB {
public:
    __aicore__ inline FlashAttentionScoreGradS1s2Bn2gs1s2SameAB(){};
 
     __aicore__ inline void Init(__gm__ uint8_t *key, __gm__ uint8_t *keyRope, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query, __gm__ uint8_t *queryRope,
                                __gm__ uint8_t *pse_shift, __gm__ uint8_t *drop_mask, __gm__ uint8_t *atten_mask,
                                __gm__ uint8_t *forward_res, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,__gm__ uint8_t *sink,
                                __gm__ uint8_t *prefixN, __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen,
                                __gm__ uint8_t *dq, __gm__ uint8_t *dqRope, __gm__ uint8_t *dk, __gm__ uint8_t *dkRope, __gm__ uint8_t *dv, __gm__ uint8_t *dpse,__gm__ uint8_t *dsink,
                                __gm__ uint8_t *workspace,
                                const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb *__restrict ordTilingData);

    using T1 = typename FAGT::t1;
    using T2 = typename FAGT::t2;
    static constexpr uint32_t IS_ATTEN_MASK = FAGT::isAttenMask;
    static constexpr uint32_t IS_PSE = FAGT::isPse;
    static constexpr uint32_t IS_DROP = FAGT::isDrop;
    static constexpr CubeFormat MM_OUT_FORMAT = FAGT::mmOutFormat;
    static constexpr uint32_t INPUT_LAYOUT = FAGT::inputLayout;
    static constexpr CubeFormat MM2_OUT_FORMAT = FAGT::mm2OutFormat;
    static constexpr uint32_t IS_DTM = FAGT::isDtm;
    static constexpr STemplateType S1TEMPLATETYPE = FAGT::s1TemplateType;
    static constexpr STemplateType S2TEMPLATETYPE = FAGT::s2TemplateType;
    static constexpr DTemplateType DTEMPLATETYPE = FAGT::dTemplateType;
    static constexpr uint32_t HAS_ROPE = FAGT::hasRope;

    __aicore__ inline void InitBuffer(TPipe *pipe_in);
    __aicore__ inline void InitTscmBuffer(TPipe *pipe_in);
    __aicore__ inline void CopyInSoftMax(LocalTensor<float> &dstTensor, uint32_t s1Extend, uint32_t softMaxOffset);
    __aicore__ inline void CalcSoftMax(LocalTensor<T2> &dstTensor, LocalTensor<float>& src0Tensor, LocalTensor<float>& src1Tensor, uint32_t s1Extend,
                                       uint32_t s2Extend, uint32_t s2ExtendAlign, const SoftMaxTiling &tiling);
    __aicore__ inline void CopyInAttenMaskBool(LocalTensor<uint8_t> &dstTensor, int64_t attenMaskOffset,
                                               uint32_t s1Extend, uint32_t s2Extend);
    __aicore__ inline void CalcAttenMaskBool(LocalTensor<T2> &dstTensor, LocalTensor<uint8_t> srcTensor,
                                             uint32_t s1Extend, uint32_t s2Extend, uint8_t maskType = 0);
    __aicore__ inline void CalcAttenMaskOffset(int64_t &attenMaskOffset, const int64_t delta, uint32_t s1VSize,
                                               uint32_t s2VSize);
    __aicore__ inline void CalcAttenBandMode(int64_t compressMode, int64_t causal_delta, DBParams &dbParam);
    __aicore__ inline void CalcAttenMaskOffsetForPrefixCompressMode(int64_t &attenMaskOffset, int64_t &attenMaskOffse2,
                                                                    const int64_t delta, uint32_t s1VSize,
                                                                    uint32_t s2VSize, uint32_t s2VBegin,
                                                                    bool &canSimplify, DBParams &dbParam);
    __aicore__ inline void CalcAttenMaskOffsetWithSparseMode(int64_t &attenMaskOffset, int64_t &attenMaskOffset2,
                                                             uint32_t s1VSize, uint32_t s2VSize, int64_t curS1Idx,
                                                             uint32_t s2VBegin, bool &canSimplify, DBParams &dbParam);
    __aicore__ inline void CalcAttenMaskOffsetWithSparseModeForUnpad(int64_t &attenMaskOffset,
                                                                     int64_t &attenMaskOffset2, uint32_t s1VSize,
                                                                     uint32_t s2VSize, int64_t curS1Idx,
                                                                     uint32_t s2VBegin, bool unpadUseBand,
                                                                     bool &canSimplify, DBParams &dbParam);
    __aicore__ inline void DropOutCopy(LocalTensor<uint8_t> &vecInDropBuffer, int64_t curS1Idx, int64_t s2VBegin);

    __aicore__ inline void Process();
    __aicore__ inline void ProcessFirstMM();
    __aicore__ inline void UpdateToken(int64_t bIdx);
    __aicore__ inline void SubGrapA(int64_t curIdx, int64_t curS1Idx, int64_t curS2Idx, DBParams& dbParam,
                                    event_t mte2WaitMte3A);
    __aicore__ inline void SubGrapB(int64_t curIdx, int64_t s1VecLoop, int64_t s2VecLoop, int64_t curS1Idx, int64_t curS2Idx, DBParams& dbParam,
                                    event_t mte2WaitMte3B, float* dsinkSumLocal);
    __aicore__ inline void ComputeVec(DBParams& dbParam);
    __aicore__ inline void SyncALLCores();
    __aicore__ inline void GetSeqQlenKvlenByBidx(int64_t bIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen);

    using aType1 = typename AscendC::Conditional<
        S1TEMPLATETYPE == STemplateType::Aligned512 && S2TEMPLATETYPE == STemplateType::Aligned512
            && (DTEMPLATETYPE == DTemplateType::Aligned128 || DTEMPLATETYPE == DTemplateType::Aligned192 || DTEMPLATETYPE == DTemplateType::Aligned64),
        MatmulType<TPosition::TSCM, CubeFormat::ND, T1, false, LayoutMode::NONE, true>,
        MatmulType<TPosition::GM, CubeFormat::ND, T1, false, LayoutMode::NONE, true>>::type;

    using bType1 = typename AscendC::Conditional<
        S1TEMPLATETYPE == STemplateType::Aligned512 && S2TEMPLATETYPE == STemplateType::Aligned512
            && (DTEMPLATETYPE == DTemplateType::Aligned128 || DTEMPLATETYPE == DTemplateType::Aligned192 || DTEMPLATETYPE == DTemplateType::Aligned64),
        MatmulType<TPosition::TSCM, CubeFormat::ND, T1, true, LayoutMode::NONE, true>,
        MatmulType<TPosition::GM, CubeFormat::ND, T1, true, LayoutMode::NONE, true>>::type;
    using cType1 = MatmulType<TPosition::GM, MM_OUT_FORMAT, T2>;
    using biasType1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    using aType2 = typename AscendC::Conditional<
        S1TEMPLATETYPE == STemplateType::Aligned512 && S2TEMPLATETYPE == STemplateType::Aligned512
            && (DTEMPLATETYPE == DTemplateType::Aligned128 || DTEMPLATETYPE == DTemplateType::Aligned192 || DTEMPLATETYPE == DTemplateType::Aligned64),
        MatmulType<TPosition::TSCM, MM_OUT_FORMAT, T1, true, LayoutMode::NONE, true>,
        MatmulType<TPosition::GM, MM_OUT_FORMAT, T1, true, LayoutMode::NONE, true>>::type;

    using bType2 = typename AscendC::Conditional<
        S1TEMPLATETYPE == STemplateType::Aligned512 && S2TEMPLATETYPE == STemplateType::Aligned512
            && (DTEMPLATETYPE == DTemplateType::Aligned128 || DTEMPLATETYPE == DTemplateType::Aligned192 || DTEMPLATETYPE == DTemplateType::Aligned64),
        MatmulType<TPosition::TSCM, CubeFormat::ND, T1, false, LayoutMode::NONE, true>,
        MatmulType<TPosition::GM, CubeFormat::ND, T1, false, LayoutMode::NONE, true>>::type;
    using cType2 = MatmulType<TPosition::GM, MM2_OUT_FORMAT, float>;
    using biasType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    constexpr static MatmulConfig mm12TemplateParam = 
        (S1TEMPLATETYPE == STemplateType::Aligned512 && 
        S2TEMPLATETYPE == STemplateType::Aligned512 && 
        (DTEMPLATETYPE == DTemplateType::Aligned192))
        ? CFG_DIS_UNIT_FLAG_EXCEED
        : SAB_NORM_DISABLE_INIT;
    using mm1Type = matmul::MatmulImpl<aType1, bType1, cType1, biasType1, mm12TemplateParam>;
    mm1Type mm1;
    constexpr static MatmulConfig mm34TemplateParam = 
        (S1TEMPLATETYPE == STemplateType::Aligned512 && 
        S2TEMPLATETYPE == STemplateType::Aligned512 && 
        (DTEMPLATETYPE == DTemplateType::Aligned128 || 
        DTEMPLATETYPE == DTemplateType::Aligned192 || 
        DTEMPLATETYPE == DTemplateType::Aligned64))
        ? CFG_DIS_UNIT_FLAG_EXCEED
        : SAB_NORM_DISABLE_INIT;
    
    using modeTypeMm = typename AscendC::Conditional<
        (MM2_OUT_FORMAT == CubeFormat::NZ &&
        !(DTEMPLATETYPE == DTemplateType::Aligned128 || 
        DTEMPLATETYPE == DTemplateType::Aligned192 || 
        DTEMPLATETYPE == DTemplateType::Aligned64)),
        matmul::MatmulImpl<aType2, bType2, cType2, biasType2, mm34TemplateParam, MatmulCallBackFunc<DataCopyOutForNz>>,
        matmul::MatmulImpl<aType2, bType2, cType2, biasType2, mm34TemplateParam>>::type;
    modeTypeMm mm3;
    modeTypeMm mm4;

    __aicore__ inline void NZCopyIn(int64_t mmAddr, GlobalTensor<T2> &mmWspGm, LocalTensor<T2> &mmTensorCurr,
                                    uint32_t s1VecSize, uint32_t s2VecSize, uint32_t s1CvInner);
    __aicore__ inline void NZ2ND(LocalTensor<T2> &mmTensorCurr, LocalTensor<T2> &tmpTensor, uint32_t s1VecSize,
                                 uint32_t s2VecSize);
    __aicore__ inline void ND2NZ(LocalTensor<T1> &mmTensorCurr, LocalTensor<T1> &tmpTensor, uint32_t s1VecSize,
                                 uint32_t s2VecSize);
    __aicore__ inline bool CalcValidBlock(int64_t& baseIdx, int64_t& startCoreId, DBParams& dbParam);
    __aicore__ inline void UpdateIndex();
    __aicore__ inline void ComputeMM1(DBParams& dbParam);
    __aicore__ inline void ComputeMMDqkv(DBParams& dbParam, int64_t nextBlockId);
    __aicore__ inline void DTMComputeMMDqkv(DBParams& dbParam, int64_t nextBlockId);
    __aicore__ inline void ComputeL1CustomMMDqkv(DBParams& dbParam, int64_t nextBlockId);
    __aicore__ inline void CalckvReduce(DBParams& dbParam, GlobalTensor<float> &srcTensor,
                                        GlobalTensor<float> &dstTensor);
    __aicore__ inline void GetIndex(int64_t baseIdx, IndexParams& idx);
    __aicore__ inline void CalcDqReduce(DBParams& dbParam, GlobalTensor<float> &srcTensor, GlobalTensor<float> &dstTensor,
        int64_t d, int64_t dAlign, uint32_t vecCalBlockNum);
    __aicore__ inline void CalcDkvReduce(DBParams& dbParam, GlobalTensor<float> &srcTensor, GlobalTensor<float> &dstTensor,
        int64_t d, int64_t dAlign, uint32_t vecCalBlockNum);
    __aicore__ inline void ComputeVecAdd(DBParams& dbParam);
    __aicore__ inline void CopyGmToL1(const LocalTensor<T1> &l1Tensor, const GlobalTensor<T1> &gmSrcTensor, uint32_t srcN, uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void LoadDataAToL1(LocalTensor<T1> dstTensor, 
                                        GlobalTensor<T1> srcTensor, 
                                        const int32_t mSize, 
                                        const int32_t kSize,
                                        uint32_t srcKstride);
    __aicore__ inline void LoadDataBToL1(LocalTensor<T1> dstTensor, 
                                        GlobalTensor<T1> srcTensor, 
                                        const int32_t nSize, 
                                        const int32_t kSize,
                                        uint32_t srcKstride);
    __aicore__ inline void LoadDataAToL0(LocalTensor<T1> dstTensor,
                                        LocalTensor<T1> srcTensor,
                                        const int32_t k0,
                                        const int32_t mSize);
    __aicore__ inline void LoadDataBToL0(LocalTensor<T1> dstTensor,
                                        LocalTensor<T1> srcTensor,
                                        const int32_t k0,
                                        const int32_t nSize);
    __aicore__ inline void Cube1Mmad(LocalTensor<float> dstCTensor,
                                        LocalTensor<T1> srcATensor,
                                        LocalTensor<T1> srcBTensor,
                                        const int32_t m_mad_,
                                        const int32_t n_mad_,
                                        const int32_t k0);
    __aicore__ inline void Cube1CopyOut(GlobalTensor<float> dstTensor, 
                                        LocalTensor<float> srcTensor, 
                                        const int32_t mSize, 
                                        const int32_t nSize,
                                        DBParams& dbParam);
    __aicore__ inline void Cube1Compute(const GlobalTensor<T1> &left,
                                        const GlobalTensor<T1> &right,
                                        const GlobalTensor<float> &out,
                                        const int32_t headDim,
                                        DBParams& dbParam);
    __aicore__ inline void ReleaseTSCMAllocEventID();
    AscendC::Nd2NzParams commonNd2NzParamsFp32_ {
        1,
        128,
        MMAD_BASE_SIZE,
        0,
        MMAD_BASE_SIZE,
        MMAD_BASE_SIZE,
        1,
        0
    };
    AscendC::MmadParams cube3MadParams {
        MMAD_BASE_SIZE,
        MMAD_BASE_SIZE,
        MMAD_BASE_SIZE,
        3,
        false,
        true
    };
    AscendC::FixpipeParamsV220 cube3FixpipeParamsV220 {
        MMAD_BASE_SIZE,
        MMAD_BASE_SIZE,
        MMAD_BASE_SIZE,
        MMAD_BASE_SIZE,
        false
    };
    AscendC::Nd2NzParams commonNd2NzParams {
        1,
        MMAD_BASE_SIZE,
        MMAD_BASE_SIZE,
        0,
        MMAD_BASE_SIZE,
        MMAD_BASE_SIZE,
        1,
        0
    };
    AscendC::LoadData2dParams commonLoadData2dParamsTranspose {
        0,
        MMAD_BASE_SIZE,
        MMAD_BASE_SIZE,
        0,
        0,
        true,
        0
    };
    AscendC::LoadData2dParams commonLoadData2dParamsNoTranspose {
        0,
        MMAD_BASE_SIZE,
        MMAD_BASE_SIZE,
        0,
        0,
        false,
        0
    };
    AscendC::MmadParams commonMadParams {
        MMAD_BASE_SIZE,
        MMAD_BASE_SIZE,
        MMAD_BASE_SIZE,
        3,
        false,
        true
    };
    AscendC::FixpipeParamsV220 commonFixpipeParamsV220 {
        MMAD_BASE_SIZE,
        MMAD_BASE_SIZE,
        MMAD_BASE_SIZE,
        MMAD_BASE_SIZE,
        false
    };

protected:
    TPipe *pipe;
    TBuf<> unifiedBuffer;

    uint32_t coreNum;
    uint32_t cubeCoreNum;
    uint32_t cBlockIdx;
    uint32_t cCubeBlockIdx;
    uint32_t cSubIdx;

    uint32_t vecBlockNum;

    const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb *__restrict TilingData;

    // input
    GlobalTensor<T1> keyGm, valueGm, dxGm, queryGm, forwardResGm, pseGm;
    GlobalTensor<T1> keyRopeGm;
    GlobalTensor<T1> queryRopeGm;
    GlobalTensor<uint8_t> maskWorkSpaceGm, attenMaskU8Gm, dropMaskGm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm, sinkGm;

    // output
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, sfmgWorkspaceGm, dqDtmWsGm, dkDtmWsGm, dvDtmWsGm;
    GlobalTensor<float> dqRopeWorkSpaceGm;
    GlobalTensor<float> dkRopeWorkSpaceGm;
    GlobalTensor<float> dqRopeDtmWsGm;
    GlobalTensor<float> dkRopeDtmWsGm;

    GlobalTensor<T1> dropWorkSpaceGm, mulWorkSpaceGm;

    // workspace
    GlobalTensor<T2> mm1WorkspaceGm;
    GlobalTensor<T2> mm2WorkspaceGm;
    GlobalTensor<float> dsinksumWorkSpaceGm;
    GlobalTensor<uint32_t> dsinksumDataSizeGm;
    // L1 buffer
    TBuf<TPosition::A1> queryBufL1;
    LocalTensor<T1> qL1Tensor;
    TBuf<TPosition::A1> keyBufL1;
    LocalTensor<T1> vL1Tensor;
    LocalTensor<T1> kL1Tensor;
    TBuf<TPosition::A1> dsBufL1;
    LocalTensor<T1> dsL1Tensor;
    LocalTensor<T1> dxL1Tensor;
    // L0 tensor
    LocalTensor<T1> l0_a_ping_tensor;
    LocalTensor<T1> l0_a_pong_tensor;
    LocalTensor<T1> l0_b_ping_tensor;
    LocalTensor<T1> l0_b_pong_tensor;
    LocalTensor<float> l0_c_ping_tensor;
    LocalTensor<float> l0_c_pong_tensor;

    event_t eventIdMte1AToM;
    event_t eventIdMte1BToM;
    event_t eventIdMToMte1_ID3;
    event_t eventIdMToMte1_ID4;
    event_t eventIdMToMte1_ID5;
    event_t eventIdMToMte1_ID6;
    event_t eventIdFixpipeToM_ID0;
    event_t eventIdFixpipeToM_ID1;
    event_t eventIdMToFixpipe_ID0;
    event_t eventIdMToFixpipe_ID1;
    event_t eventIdMte2ToMte1_ID0;
    event_t eventIdMte2ToMte1_ID1;
    event_t eventIdMte2ToMte1_ID2;
    event_t eventIdMte2ToMte1_ID3;
    event_t eventIdMte1ToMte2_ID0;
    event_t eventIdMte1ToMte2_ID1;

    AsdopsBuffer<ArchType::ASCEND_V220> asdopsBuf;

    __gm__ uint8_t *prefixN_addr;
    __gm__ uint8_t *actual_seq_qlen_addr;
    __gm__ uint8_t *actual_seq_kvlen_addr;

    GM_ADDR workspaceAddr;

    // AscendC
    GlobalTensor<int32_t> syncGlobal;

    GlobalTensor<half> pseAlibiGm;
    GlobalTensor<float> dvGm;
    __gm__ uint8_t *pseSlope;

    PseInfo pseInfo = {0};

    constexpr static uint32_t BNGSD = 0;
    constexpr static uint32_t SBNGD = 1;
    constexpr static uint32_t BSNGD = 2;
    constexpr static uint32_t TND = 3;
    constexpr static uint32_t ENABLE = 1;

    constexpr static uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_V1_C2_FLAG[3] = {4, 5, 6};
    static constexpr uint64_t SYNC_C1_V1_FLAG[3] = {1, 2, 3};
    static constexpr uint64_t SYNC_C2_V1_FLAG[3] = {7, 8, 9};

    // optional control
    float keepProb;
    int64_t s1Token;
    int64_t s2Token;
    int64_t actualCalcS1Token;
    int64_t actualCalcS2Token;
    uint32_t sparseMode;
    bool dropBitMode;

    // org shape info
    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t rope_d = 0; // q rope / k rope
    int64_t value_d;
    int64_t dAlign;
    int64_t rope_dAlign = 0;
    int64_t value_dAlign;
    int64_t attenMaskDimS2;

    uint32_t baseMN;
    uint32_t cubeBaseMN;

    // split info
    int64_t s1Outer;
    uint32_t s1CvInner;
    uint32_t s1CvTail;
    int64_t s2Outer;
    uint32_t s2CvInner;
    uint32_t s2CvTail;

    int64_t sfmgOffset = 0;
    uint32_t preS1Idx = -1;

    // base info
    int64_t baseIdx{0};
    int64_t bDimIdx{0};
    int64_t n2DimIdx{0};
    int64_t gDimIdx{0};
    int64_t s1oDimIdx{0};
    int64_t s2oCvDimIdx{0};

    int32_t isStart = 1;
    uint32_t pingpongIdx = 1;
    int32_t vecLoopStart;
    int32_t vecLoopEnd;

    // db
    uint32_t s1VecLoop = 0;
    uint32_t s1VecSize = 0;
    uint32_t s1ExtendSubGraph = 0;
    uint32_t s2Extend = 0;
    uint32_t s2ExtendAlign = 0;
    uint32_t s2VecLoop = 0;
    uint32_t s2VecSize = 0;

    int64_t dqOutBase{0};
    int64_t kvOutBase{0};
    int64_t dqOutIdx{0};   // bn2gs1o
    int64_t kvOutIdx{0};   // bn2s2o
    int64_t dqOutArr[24];
    int64_t kvOutArr[24];

    DBParams dbParams[3];
    int64_t blockStartIdx = 0;

    // unpack
    int64_t bandIdx = 0;

    DropMaskInfo dropMaskInfo = {0};
    // ping pong flag
    uint32_t ping_pong_flag_l1_a_ = 0;
    uint32_t ping_pong_flag_l0_a_ = 0;
    uint32_t ping_pong_flag_l0_b_ = 0;
    uint32_t ping_pong_flag_l0_c_ = 0;
    // db buffer
    constexpr static uint32_t T2Begin = 0;
    constexpr static uint32_t T1Begin = 33 * 1024;
    constexpr static uint32_t BoolBegin = 50 * 1024;
    constexpr static uint32_t T2BlockBegin = 58 * 1024;
    constexpr static uint32_t U8Begin = 66 * 1024;
    constexpr static uint32_t DbBegin = 74 * 1024;

    // other const
    constexpr static uint32_t DTYPE_FACTOR = sizeof(T2) / sizeof(T1);
    constexpr static uint32_t cal_block_num = 32 / sizeof(T2);
    constexpr static uint32_t cal_repeat_num = 256 / sizeof(T2);
    constexpr static uint32_t input_block_num = 32 / sizeof(T1);
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint32_t INPUT_NUMS = 2;
    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static int64_t C0_SIZE = 16;
    constexpr static int64_t VEC_REPEAT = 8;
    constexpr static uint32_t PREFIX_COMPRESS_CAUSAL_S_SIZE = 2048;
    constexpr static uint32_t PREFIX_COMPRESS_ALL_MASK_S1_SIZE = 1024;
    constexpr static int64_t GM_DOUBLE_BUFFER = 2;
    constexpr static int64_t TMP_UB_OFFSET = 148 * 1024;
    constexpr static int64_t SFMG_UB_OFFSET = (148 + 33) * 1024;
    constexpr static int64_t TMP_UB_SIZE = 33 * 1024;
    constexpr static int64_t SFMG_UB_SIZE = 8 * 1024;
    constexpr static int64_t TOTAL_SIZE = 189 * 1024;

    constexpr static uint32_t MMAD_BASE_SIZE = 128;
    constexpr static uint32_t S_BASE_SIZE = 512;
    constexpr static uint32_t S_SPLITT_SIZE = 256;
    constexpr static uint32_t L1_CACHE_CAPACITY_LIMIT = 14;
    constexpr static uint32_t DIM_64 = 64;
    constexpr static uint32_t VEC_S2_LEN = 256;
    constexpr static uint32_t SIZE_256 = 256;
    constexpr static int8_t OUTIDX= -1;
    bool tndSoftmaxIn;
    enum class AttenMaskCompress {
        Empty = 0,
        PreOnly = 1,
        NextOnly = 2,
        All = 3
    };
    AttenMaskCompress AttenBandMode = AttenMaskCompress::All;
};

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::Init(
                          __gm__ uint8_t *key, __gm__ uint8_t *keyRope, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query, __gm__ uint8_t *queryRope,
                          __gm__ uint8_t *pse_shift, __gm__ uint8_t *drop_mask, __gm__ uint8_t *atten_mask,
                          __gm__ uint8_t *forward_res, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum, __gm__ uint8_t *sink,
                          __gm__ uint8_t *prefixN, __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen,
                          __gm__ uint8_t *dq, __gm__ uint8_t *dqRope, __gm__ uint8_t *dk, __gm__ uint8_t *dkRope, __gm__ uint8_t *dv, __gm__ uint8_t *dpse, __gm__ uint8_t *dsink,
                          __gm__ uint8_t *workspace,
                          const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb *__restrict ordTilingData)
{
    keyGm.SetGlobalBuffer((__gm__ T1 *)key);
    valueGm.SetGlobalBuffer((__gm__ T1 *)value);
    dxGm.SetGlobalBuffer((__gm__ T1 *)dx);
    queryGm.SetGlobalBuffer((__gm__ T1 *)query);
    forwardResGm.SetGlobalBuffer((__gm__ T1 *)forward_res);
    pseGm.SetGlobalBuffer((__gm__ T1 *)pse_shift);
    sinkGm.SetGlobalBuffer((__gm__ float *)sink);
    if constexpr (HAS_ROPE == ENABLE) {
        keyRopeGm.SetGlobalBuffer((__gm__ T1 *)keyRope);
        queryRopeGm.SetGlobalBuffer((__gm__ T1 *)queryRope);
    }

    pseSlope = pse_shift;

    dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)drop_mask);
    attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t *)atten_mask);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmax_max);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmax_sum);
    dvGm.SetGlobalBuffer((__gm__ float *)dv);

    // init current core tilingInfo
    if ASCEND_IS_AIV {
        cBlockIdx = GetBlockIdx();
        cCubeBlockIdx = cBlockIdx / 2;
        cSubIdx = cBlockIdx % 2;
    } else {
        cCubeBlockIdx = GetBlockIdx();
        cBlockIdx = cCubeBlockIdx * 2;
    }
    TilingData = ordTilingData;

    if ASCEND_IS_AIC {
        mm1.Init(&TilingData->mm1TilingData, pipe);
        mm3.Init(&TilingData->mm2TilingData, pipe);
        mm4.Init(&TilingData->mm3TilingData, pipe);
    }

    coreNum = TilingData->s1s2BNGS1S2BaseParams.coreNum;
    cubeCoreNum = coreNum / 2;

    vecBlockNum = coreNum / 3;

    // shape info
    b = TilingData->s1s2BNGS1S2BaseParams.b;
    n2 = TilingData->s1s2BNGS1S2BaseParams.n2;
    g = TilingData->s1s2BNGS1S2BaseParams.g;
    s1 = TilingData->s1s2BNGS1S2BaseParams.s1;
    s2 = TilingData->s1s2BNGS1S2BaseParams.s2;
    d = TilingData->s1s2BNGS1S2BaseParams.d;
    value_d = TilingData->s1s2BNGS1S2BaseParams.value_d;
    tndSoftmaxIn = TilingData->s1s2BNGS1S2BaseParams.tndSoftmaxIn == 1 ? true : false;
    dAlign = (d + 15) / 16 * 16;
    value_dAlign = (value_d + 15) / 16 * 16;
    if constexpr (HAS_ROPE == ENABLE) {
        rope_d = TilingData->s1s2BNGS1S2BaseParams.rope_d;
        rope_dAlign = (rope_d + 15) / 16 * 16;
    }
    attenMaskDimS2 = TilingData->s1s2BNGS1S2BaseParams.attenMaskS2Size;

    s1Token = TilingData->s1s2BNGS1S2BaseParams.s1Token;
    s2Token = TilingData->s1s2BNGS1S2BaseParams.s2Token;
    actualCalcS1Token = s1Token;
    actualCalcS2Token = s2Token;
    sparseMode = TilingData->s1s2BNGS1S2BaseParams.sparseMode;
    bandIdx = TilingData->s1s2BNGS1S2SplitCoreParams.bandIdx;

    // split info
    s1Outer = TilingData->s1s2BNGS1S2SplitCoreParams.s1Outer;
    s1CvInner = TilingData->s1s2BNGS1S2SplitCoreParams.s1CvInner;
    s1CvTail = TilingData->s1s2BNGS1S2SplitCoreParams.s1CvTail;
    s2Outer = TilingData->s1s2BNGS1S2SplitCoreParams.s2Outer;
    s2CvInner = TilingData->s1s2BNGS1S2SplitCoreParams.s2CvInner;
    s2CvTail = s2 - (s2Outer - 1) * s2CvInner;

    baseMN = TilingData->s1s2BNGS1S2SplitCoreParams.baseMN;
    cubeBaseMN = s1CvInner * s2CvInner;

    prefixN_addr = prefixN;
    actual_seq_qlen_addr = actual_seq_qlen;
    actual_seq_kvlen_addr = actual_seq_kvlen;

    dropBitMode = s2 % 8 == 0;
    keepProb = TilingData->s1s2BNGS1S2BaseParams.keepProb;
    int64_t sfmgOutputSize = b * n2 * g * s1 * 8;
    if constexpr (INPUT_LAYOUT == TND) {
        int64_t seqS2Len = 0;
        seqS2Len = ((__gm__ int64_t *)actual_seq_kvlen)[0];
        dropBitMode = (seqS2Len % 8 == 0);
        for (int64_t i = 0; i + 1 < b; i++) {
            seqS2Len = ((__gm__ int64_t *)actual_seq_kvlen)[i + 1] - ((__gm__ int64_t *)actual_seq_kvlen)[i];
            dropBitMode = (dropBitMode && (seqS2Len % 8 == 0));
        }
        sfmgOutputSize = ((__gm__ int64_t*)actual_seq_qlen)[b - 1] * n2 * g * 8;
    }

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
    

    workspaceAddr = workspace;

    // init workspace address
    syncGlobal.SetGlobalBuffer((__gm__ int32_t *)workspace);
    InitOutput<int32_t>(syncGlobal[GetBlockIdx() * 256], 256, 0);  // 前64K留给同步使用，每个

    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                  TilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                TilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
    dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                TilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));
    if constexpr (HAS_ROPE == ENABLE) {
        dqRopeWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + 
            TilingData->postTilingData.dqRopeWorkSpaceOffset / sizeof(float));
        dkRopeWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + 
            TilingData->postTilingData.dkRopeWorkSpaceOffset / sizeof(float));
    }

    if constexpr (IS_DROP == ENABLE) {
        if (!dropBitMode) {
            maskWorkSpaceGm.SetGlobalBuffer((__gm__ uint8_t *)workspace + TilingData->preTilingData.dropBeginAddr);
        }
    }
    // sfmg, softmax grad
    sfmgWorkspaceGm.SetGlobalBuffer((__gm__ T2 *)workspace + TilingData->preSfmgTilingData.sfmgPreBeginAddr / sizeof(T2));
    int64_t workspaceOffsets =
        (TilingData->preSfmgTilingData.sfmgPreBeginAddr + sfmgOutputSize * sizeof(float) + ADDR_ALIGN_SIZE) /
        ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;

    int64_t pseInnerAlibiSize = TilingData->s1s2BNGS1S2BaseParams.pseAlibiBaseS1 *
                                this->TilingData->s1s2BNGS1S2BaseParams.pseAlibiBaseS2 * sizeof(half);
    int64_t pseAlibiOffset =  CeilDiv(pseInnerAlibiSize, 512) * 512;

    // matmul1 and matmul2 workspace size
    uint32_t matmulWorkspaceSize = cubeBaseMN * sizeof(float);
    mm1WorkspaceGm.SetGlobalBuffer((__gm__ T2 *)(workspace + workspaceOffsets +
                                                 cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));
    mm2WorkspaceGm.SetGlobalBuffer(
        (__gm__ T2 *)(workspace + workspaceOffsets + cubeCoreNum * matmulWorkspaceSize * GM_DOUBLE_BUFFER +
                      cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));

    // drop workspace offset 和 mm2WorkspaceGm 地址相同
    dropWorkSpaceGm.SetGlobalBuffer(
        (__gm__ T1 *)(workspace + workspaceOffsets + cubeCoreNum * matmulWorkspaceSize * GM_DOUBLE_BUFFER +
                      cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));

    // mul workspace offset 和 mm1WorkspaceGm 地址相同
    mulWorkSpaceGm.SetGlobalBuffer((__gm__ T1 *)(workspace + workspaceOffsets +
                                                 cCubeBlockIdx * matmulWorkspaceSize * GM_DOUBLE_BUFFER));

    dsinksumWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                            TilingData->postTilingData.dsinksumWorkSpaceOffset / sizeof(float));

    dsinksumDataSizeGm.SetGlobalBuffer((__gm__ uint32_t *)workspace +
                            TilingData->postTilingData.dsinksumDataSizeOffset / sizeof(uint32_t));
                            
    uint64_t pseAlibiAddr = (workspaceOffsets + cubeCoreNum * matmulWorkspaceSize * INPUT_NUMS *
                                                    GM_DOUBLE_BUFFER + ADDR_ALIGN_SIZE + TilingData->postTilingData.sinkDataSize) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    this->pseAlibiGm.SetGlobalBuffer((__gm__ half*)(workspace + pseAlibiAddr + cBlockIdx * pseAlibiOffset));

    if constexpr (IS_DTM == ENABLE) {
        workspaceOffsets = (pseAlibiAddr + coreNum * pseAlibiOffset + ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE *
                       ADDR_ALIGN_SIZE;

        dqDtmWsGm.SetGlobalBuffer((__gm__ float *)(workspace + workspaceOffsets));
        workspaceOffsets = (workspaceOffsets + s1CvInner * dAlign * sizeof(float) * cubeCoreNum * GM_DOUBLE_BUFFER +
                            ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
        if constexpr (HAS_ROPE == ENABLE) {
            dqRopeDtmWsGm.SetGlobalBuffer((__gm__ float *)(workspace + workspaceOffsets));
            workspaceOffsets = (workspaceOffsets + s1CvInner * rope_dAlign * sizeof(float) * cubeCoreNum * GM_DOUBLE_BUFFER +
                                ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
        }
 
        dkDtmWsGm.SetGlobalBuffer((__gm__ float *)(workspace + workspaceOffsets));
        workspaceOffsets = (workspaceOffsets + s2CvInner * dAlign * sizeof(float) * cubeCoreNum * GM_DOUBLE_BUFFER +
                            ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
        if constexpr (HAS_ROPE == ENABLE) {
            dkRopeDtmWsGm.SetGlobalBuffer((__gm__ float *)(workspace + workspaceOffsets));
            workspaceOffsets = (workspaceOffsets + s2CvInner * rope_dAlign * sizeof(float) * cubeCoreNum * GM_DOUBLE_BUFFER +
                                ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
        }
 
        dvDtmWsGm.SetGlobalBuffer((__gm__ float *)(workspace + workspaceOffsets));
    }

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
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::InitBuffer(TPipe *pipe_in)
{
    pipe = pipe_in;
    if ASCEND_IS_AIV {
        pipe->InitBuffer(unifiedBuffer, TOTAL_SIZE);
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
            if (cBlockIdx < coreNum &&
                (TilingData->s1s2BNGS1S2BaseParams.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                TilingData->s1s2BNGS1S2BaseParams.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
                LocalTensor<half> pseHelpBuffer = unifiedBuffer.GetWithOffset<half>(16 * 1024 / sizeof(half), T1Begin);
                PseInnerAlibiCreate<true>(this->pseAlibiGm, pseHelpBuffer, pseInfo);
            }
        }
    }
    SyncAll();  //保证清零完成
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::InitTscmBuffer(TPipe *pipe_in)
{
    pipe = pipe_in;
    if ASCEND_IS_AIC {
        if constexpr (DTEMPLATETYPE == DTemplateType::Aligned128) {
            pipe->InitBuffer(queryBufL1, 512 * 128 * 2);
            qL1Tensor = queryBufL1.Get<T1>();

            pipe->InitBuffer(keyBufL1, 512 * 128 * 2);
            kL1Tensor = keyBufL1.Get<T1>();

            pipe->InitBuffer(dsBufL1, 512 * 128 * 2 * 2);
            dsL1Tensor = dsBufL1.Get<T1>();

            dxL1Tensor = queryBufL1.Get<T1>();
            vL1Tensor = keyBufL1.Get<T1>();
            // init L0A/L0B/L0C tensor
            l0_a_ping_tensor = asdopsBuf.template GetBuffer<BufferType::ASCEND_L0A, T1>(0);
            l0_a_pong_tensor = asdopsBuf.template GetBuffer<BufferType::ASCEND_L0A, T1>(SIZE_32 * SIZE_ONE_K);
            l0_b_ping_tensor = asdopsBuf.template GetBuffer<BufferType::ASCEND_L0B, T1>(0);
            l0_b_pong_tensor = asdopsBuf.template GetBuffer<BufferType::ASCEND_L0B, T1>(SIZE_32 * SIZE_ONE_K);
            l0_c_ping_tensor = asdopsBuf.template GetBuffer<BufferType::ASCEND_L0C, float>(0);
            l0_c_pong_tensor = asdopsBuf.template GetBuffer<BufferType::ASCEND_L0C, float>(SIZE_64 * SIZE_ONE_K);
            eventIdMte1AToM = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>());
            eventIdMte1BToM = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>());
            eventIdMToMte1_ID3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
            eventIdMToMte1_ID4 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
            eventIdMToMte1_ID5 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
            eventIdMToMte1_ID6 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
            eventIdFixpipeToM_ID0 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::FIX_M>());
            eventIdFixpipeToM_ID1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::FIX_M>());
            eventIdMToFixpipe_ID0 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_FIX>());
            eventIdMToFixpipe_ID1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_FIX>());
            eventIdMte2ToMte1_ID0 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
            eventIdMte2ToMte1_ID1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
            eventIdMte2ToMte1_ID2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
            eventIdMte2ToMte1_ID3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
            eventIdMte1ToMte2_ID0 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
            eventIdMte1ToMte2_ID1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
        } else if constexpr (DTEMPLATETYPE == DTemplateType::Aligned64) {
            pipe->InitBuffer(keyBufL1, 512 * 64 * 2);
            pipe->InitBuffer(dsBufL1, 512 * (512 - 64) * 2);
            qL1Tensor = dsBufL1.Get<T1>();
            dsL1Tensor = dsBufL1.Get<T1>();
            dxL1Tensor = dsBufL1.Get<T1>();
            kL1Tensor = keyBufL1.Get<T1>();
            vL1Tensor = keyBufL1.Get<T1>();
        } else if constexpr (DTEMPLATETYPE == DTemplateType::Aligned192) {
            pipe->InitBuffer(keyBufL1, 512 * 256 * 2);
            pipe->InitBuffer(dsBufL1, 512 * 256* 2);
            qL1Tensor = dsBufL1.Get<T1>();
            dsL1Tensor = dsBufL1.Get<T1>();
            dxL1Tensor = dsBufL1.Get<T1>();
            kL1Tensor = keyBufL1.Get<T1>();
            vL1Tensor = keyBufL1.Get<T1>();
        }
    }
}

template <typename FAGT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::GetSeqQlenKvlenByBidx(int64_t bIdx,
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

template <typename FAGT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::CopyInSoftMax(LocalTensor<float> &dstTensor, uint32_t s1Extend,
                                                               uint32_t softMaxOffset)
{
    DataCopyPad(dstTensor, softmaxSumGm[softMaxOffset], {1, static_cast<uint16_t>(s1Extend * 32), 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(dstTensor[s1Extend * 32 / sizeof(float)], softmaxMaxGm[softMaxOffset],
                {1, static_cast<uint16_t>(s1Extend * 32), 0, 0}, {false, 0, 0, 0});
}

template <typename FAGT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::CalcSoftMax(
    LocalTensor<T2>& dstTensor, LocalTensor<float>& src0Tensor, LocalTensor<float>& src1Tensor, uint32_t s1Extend, uint32_t s2Extend,
    uint32_t s2ExtendAlign, const SoftMaxTiling& tiling) {
  bool isBasicBlock = (s1Extend % 8 == 0) && (s2Extend % 64 == 0);

  if (isBasicBlock) {
    LocalTensor<uint8_t> vecOutBuffer = unifiedBuffer.GetWithOffset<uint8_t>(TMP_UB_SIZE / sizeof(uint8_t), TMP_UB_OFFSET);
    uint32_t shapeArray1[2];
    shapeArray1[0] = s1Extend;
    shapeArray1[1] = s2Extend;
    dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray1, DataFormat::ND));
    src0Tensor.SetShapeInfo(ShapeInfo(2, shapeArray1, DataFormat::ND));
    SimpleSoftMax<T2, false, true>(dstTensor, src1Tensor, src1Tensor[s1Extend * 32 / sizeof(float)], src0Tensor,
                                  vecOutBuffer, tiling);
  } else {
    LocalTensor<T2> vecOutBuffer = unifiedBuffer.GetWithOffset<T2>(TMP_UB_SIZE / sizeof(T2), TMP_UB_OFFSET);
    uint32_t sub_block_count = (s2Extend + cal_repeat_num - 1) / cal_repeat_num;

    for(uint32_t subIdx = 0; subIdx < sub_block_count; subIdx++) {
      uint32_t subMaskCount = (subIdx == sub_block_count - 1) ? (s2Extend - subIdx * cal_repeat_num) : cal_repeat_num;
      Sub(dstTensor[subIdx * cal_repeat_num], src0Tensor[subIdx * cal_repeat_num], src1Tensor[s1Extend * 8],
              subMaskCount, s1Extend,
              {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0,
              static_cast<uint8_t>(s2ExtendAlign / 8), static_cast<uint8_t>(s2ExtendAlign / 8), 1});
      AscendC::PipeBarrier<PIPE_V>();
      Exp(vecOutBuffer[subIdx * cal_repeat_num], dstTensor[subIdx * cal_repeat_num],
          subMaskCount, s1Extend,
              {static_cast<uint8_t>(1), static_cast<uint8_t>(1),
              static_cast<uint8_t>(s2ExtendAlign / 8), static_cast<uint8_t>(s2ExtendAlign / 8)});
      AscendC::PipeBarrier<PIPE_V>();
      Div(dstTensor[subIdx * cal_repeat_num], vecOutBuffer[subIdx * cal_repeat_num], src1Tensor,
              subMaskCount, s1Extend,
              {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0,
              static_cast<uint8_t>(s2ExtendAlign / 8), static_cast<uint8_t>(s2ExtendAlign / 8), 1});
      AscendC::PipeBarrier<PIPE_V>();
    }
  }
}

template <typename FAGT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::CopyInAttenMaskBool(LocalTensor<uint8_t> &dstTensor,
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

template <typename FAGT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::CalcAttenMaskBool(LocalTensor<T2> &dstTensor,
                                                                   LocalTensor<uint8_t> srcTensor,
                                                                   uint32_t s1Extend, uint32_t s2Extend,
                                                                   uint8_t maskType)
{
    LocalTensor<uint8_t> tmpUbBuffer = unifiedBuffer.GetWithOffset<uint8_t>(TMP_UB_SIZE / sizeof(uint8_t), TMP_UB_OFFSET);

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

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::UpdateIndex()
{
    s1oDimIdx = 0;
    if constexpr (IS_DTM == ENABLE) {
        if (gDimIdx < g - 1) {
            gDimIdx++;
        } else if (s2oCvDimIdx < s2Outer - 1) {
            gDimIdx = 0;
            s2oCvDimIdx++;
        } else if (n2DimIdx < n2 - 1) {
            gDimIdx = 0;
            s2oCvDimIdx = 0;
            n2DimIdx++;
        } else {
            gDimIdx = 0;
            s2oCvDimIdx = 0;
            n2DimIdx = 0;
            bDimIdx++;
            dqOutBase += n2 * g * s1Outer;
            kvOutBase += n2 * s2Outer;
        }
    } else {
        if (s2oCvDimIdx < s2Outer - 1) {
            s2oCvDimIdx++;
        } else if (gDimIdx < g - 1) {
            s2oCvDimIdx = 0;
            gDimIdx++;
        } else if (n2DimIdx < n2 - 1) {
            gDimIdx = 0;
            s2oCvDimIdx = 0;
            n2DimIdx++;
        } else {
            gDimIdx = 0;
            s2oCvDimIdx = 0;
            n2DimIdx = 0;
            bDimIdx++;
            dqOutBase += n2 * g * s1Outer;
            kvOutBase += n2 * s2Outer;
        }
    }
}

template <typename FAGT>
__aicore__ inline bool FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::CalcValidBlock(
    int64_t& baseIdx, int64_t& startCoreId, DBParams& dbParam)
{
    if (bDimIdx >= b) { // 越界情形
        if (cCubeBlockIdx >= startCoreId) {
            dbParam.blockId = -1; // 没有数据需要处理
        }
        dbParam.blockIdArr[startCoreId] = -1;
        baseIdx = -1;
        return true;
    }
    int64_t actualSeqQlen = s1;
    int64_t actualSeqKvlen = s2;
    if constexpr(INPUT_LAYOUT == TND) {
        UpdateToken(bDimIdx);
        GetSeqQlenKvlenByBidx(bDimIdx, actualSeqQlen, actualSeqKvlen);
        s1Outer = (actualSeqQlen + s1CvInner - 1) / s1CvInner;
        s2Outer = (actualSeqKvlen + s2CvInner - 1) / s2CvInner;
        s1CvTail = actualSeqQlen - (s1Outer - 1) * s1CvInner;
        s2CvTail = actualSeqKvlen - (s2Outer - 1) * s2CvInner;
    }
    dqOutIdx = dqOutBase + (n2DimIdx * g + gDimIdx) * s1Outer + s1oDimIdx;
    kvOutIdx = kvOutBase + n2DimIdx * s2Outer + s2oCvDimIdx;

    int64_t curPrefixN = 0;
    int64_t s1IdxUp = s2oCvDimIdx * s2CvInner - actualCalcS2Token;
    if (sparseMode == 5 || sparseMode == 6) {  // prefix场景
      curPrefixN = ((__gm__ int64_t*)prefixN_addr)[bDimIdx];
      s1IdxUp = s2oCvDimIdx * s2CvInner < curPrefixN ? 0 : s1IdxUp;
    }

    // s2token 保护: 1、sparse场景，基本块无效。2、s1==0 or s2==0 场景，无基本块。跳过
    if (s1IdxUp >= actualSeqQlen || actualSeqKvlen == 0) {
        // 当出现s2==0的空Tensor场景时，不进行基本块编号
        baseIdx += actualSeqKvlen ? (s1Outer - s1oDimIdx) : 0;
        UpdateIndex();
        return false;
    }

    int64_t s1oIdxUp = s1IdxUp / s1CvInner;
    s1oIdxUp = s1oIdxUp > 0 ? s1oIdxUp : 0;

    int64_t s1IdxDown = (s2oCvDimIdx + 1) * s2CvInner + actualCalcS1Token;
    s1IdxDown = s1IdxDown > actualSeqQlen ? actualSeqQlen : s1IdxDown;
    int64_t s1oIdxDown = (s1IdxDown + s1CvInner - 1) / s1CvInner - 1;

    // 当前s1方向没有有效基本块或者起始位置在preToken下方，跳过当前s1
    if (s1oIdxDown < s1oIdxUp || s1oDimIdx > s1oIdxDown) {
        baseIdx += (s1Outer - s1oDimIdx);
        UpdateIndex();
        return false;
    }
    // 起始在nextToken上方，跳到第一个有效块
    if (s1oDimIdx < s1oIdxUp) {
        baseIdx += s1oIdxUp - s1oDimIdx;
        dqOutIdx += s1oIdxUp - s1oDimIdx;
        s1oDimIdx = s1oIdxUp;
    }
    int64_t validNum = s1oIdxDown - s1oDimIdx + 1;
    if (cCubeBlockIdx >= startCoreId && cCubeBlockIdx - startCoreId < validNum) {
        dbParam.blockId = baseIdx + cCubeBlockIdx - startCoreId;
        dbParam.bIdx = bDimIdx;
        dbParam.n2Idx = n2DimIdx;
        dbParam.s2oIdx = s2oCvDimIdx;
        dbParam.gIdx = gDimIdx;
        dbParam.s1oIdx = s1oDimIdx + cCubeBlockIdx - startCoreId;
        dbParam.s1CvExtend = (dbParam.s1oIdx == s1Outer - 1) ? s1CvTail : s1CvInner;
        dbParam.s2CvExtend = (dbParam.s2oIdx == s2Outer - 1) ? s2CvTail : s2CvInner;
        int64_t s2RightIdx = dbParam.s1oIdx * s1CvInner + dbParam.s1CvExtend + actualCalcS2Token;
        s2RightIdx = s2RightIdx > 0 ? s2RightIdx : 0;
        if (sparseMode == 5 || sparseMode == 6) {
            s2RightIdx = s2RightIdx < curPrefixN ? curPrefixN : s2RightIdx;
        }
        s2RightIdx = (s2RightIdx + 7) / 8 * 8;
        dbParam.s2CvExtend = s2RightIdx > (dbParam.s2oIdx * s2CvInner + dbParam.s2CvExtend) ? dbParam.s2CvExtend :
                             s2RightIdx - dbParam.s2oIdx * s2CvInner;
        dbParam.s2CvExtend = dbParam.s2CvExtend > 0 ? dbParam.s2CvExtend : 0;
        dbParam.s1CvExtendAlign = (dbParam.s1CvExtend + 15) / 16 * 16;
        dbParam.s2CvExtendAlign = (dbParam.s2CvExtend + 15) / 16 * 16;
    }

    // -----确定性计算，计算kvGroupId和dqGroupId
    if constexpr (IS_DTM == ENABLE) {
        int8_t kvGroupId = startCoreId;
        if (startCoreId > 0 && kvOutIdx == kvOutArr[startCoreId - 1] && dbParam.kvGroupId[startCoreId - 1] != OUTIDX) {
            kvGroupId = dbParam.kvGroupId[startCoreId - 1];
        }
        uint32_t s2CvExtend = (s2oCvDimIdx == s2Outer - 1) ? s2CvTail : s2CvInner;

        for (int32_t i = startCoreId; i < cubeCoreNum; i++) {
            if (i - startCoreId >= validNum) {
                break;
            }
            dqOutArr[i] = dqOutIdx + i - startCoreId;
            kvOutArr[i] = kvOutIdx;
            dbParam.blockIdArr[i] = baseIdx + i - startCoreId;
            dbParam.s1CvExtendArr[i] = (s1oDimIdx + i - startCoreId) == (s1Outer - 1) ? s1CvTail : s1CvInner;
            int64_t s2RightIdx = (s1oDimIdx + i - startCoreId) * s1CvInner + dbParam.s1CvExtendArr[i] + actualCalcS2Token;
            s2RightIdx = s2RightIdx > 0 ? s2RightIdx : 0;
            if (sparseMode == 5 || sparseMode == 6) {  // prefix场景
                s2RightIdx = s2RightIdx < curPrefixN ? curPrefixN : s2RightIdx;
            }
            s2RightIdx = (s2RightIdx + 7) / 8 * 8;
            dbParam.s2CvExtendArr[i] = s2RightIdx > (s2oCvDimIdx * s2CvInner + s2CvExtend) ?
                                    s2CvExtend : (s2RightIdx - s2oCvDimIdx * s2CvInner);
            dbParam.kvGroupId[i] = kvGroupId;
            dbParam.dqGroupId[i] = i;
            if (s2oCvDimIdx == 0) {
                dbParam.dqGroupId[i] = OUTIDX;
                continue;
            }

            for (int32_t j = 0; j < startCoreId; j++) {
                if (dqOutArr[i] == dqOutArr[j]) {
                    dbParam.dqGroupId[i] = j;
                    break;
                }
            }
        }
        if (s1oDimIdx == s1oIdxUp && gDimIdx == 0) {
            dbParam.kvGroupId[startCoreId] = OUTIDX;
        }
    }
    //-----------

    if (cubeCoreNum - startCoreId <= validNum) {
        baseIdx = baseIdx + cubeCoreNum - startCoreId;
        if ((s1oDimIdx + cubeCoreNum - startCoreId) < s1Outer) {
            s1oDimIdx = s1oDimIdx + cubeCoreNum - startCoreId;
        } else {
            UpdateIndex();
        }
        return true;
    } else {
        baseIdx += (s1Outer - s1oDimIdx);
        startCoreId += validNum;
        UpdateIndex();
        return false;
    }
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::CopyGmToL1(
    const LocalTensor<T1> &l1Tensor, const GlobalTensor<T1> &gmSrcTensor, uint32_t srcN, uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // 行数
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = CeilDiv(srcN, 16) * 16; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    AscendC::DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::LoadDataAToL0(LocalTensor<T1> dstTensor,
                                                                                  LocalTensor<T1> srcTensor,
                                                                                  const int32_t k0,
                                                                                  const int32_t mSize)
{
    AscendC::WaitFlag<HardEvent::M_MTE1>(3 + ping_pong_flag_l0_a_); // 信号量3，4用于控制L0A搬运
    int32_t mSizeAlign = RoundUp(mSize, (int32_t)C0_SIZE);
    commonLoadData2dParamsNoTranspose.repeatTimes = k0 / C0_SIZE;
    commonLoadData2dParamsNoTranspose.srcStride = mSizeAlign / C0_SIZE;
    for (int32_t i = 0; i < mSizeAlign / C0_SIZE; i++) {
        AscendC::LoadData(dstTensor[i * d * C0_SIZE],
                            srcTensor[i * SIZE_256],
                            commonLoadData2dParamsNoTranspose);
    }
    AscendC::SetFlag<HardEvent::MTE1_M>(eventIdMte1AToM);
    AscendC::WaitFlag<HardEvent::MTE1_M>(eventIdMte1AToM);
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::LoadDataBToL0(LocalTensor<T1> dstTensor,
                                                                                  LocalTensor<T1> srcTensor,
                                                                                  const int32_t k0,
                                                                                  const int32_t nSize)
{
    AscendC::WaitFlag<HardEvent::M_MTE1>(3 + ping_pong_flag_l0_b_ + 2); // 信号量5、6用于控制L0B搬运
    int32_t nSizeAlign = RoundUp(nSize, (int32_t)C0_SIZE);
    commonLoadData2dParamsNoTranspose.repeatTimes = nSizeAlign / C0_SIZE;
    commonLoadData2dParamsNoTranspose.srcStride = 1;
    for (int32_t i = 0; i < k0 / C0_SIZE; i++) {
        AscendC::LoadData(dstTensor[i * nSizeAlign * C0_SIZE],
                            srcTensor[i * nSizeAlign * C0_SIZE],
                            commonLoadData2dParamsNoTranspose);
    }
    AscendC::SetFlag<HardEvent::MTE1_M>(eventIdMte1BToM);
    AscendC::WaitFlag<HardEvent::MTE1_M>(eventIdMte1BToM);
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::Cube1Mmad(LocalTensor<float> dstCTensor,
                                                                              LocalTensor<T1> srcATensor,
                                                                              LocalTensor<T1> srcBTensor,
                                                                              const int32_t m_mad_,
                                                                              const int32_t n_mad_,
                                                                              const int32_t k0)
{
    uint16_t m_modify = (m_mad_ == 1) ? 2 : m_mad_;
    commonMadParams.m = m_modify;
    commonMadParams.n = n_mad_;
    commonMadParams.k = k0;
    commonMadParams.unitFlag = 3;
    commonMadParams.cmatrixInitVal = true;
    AscendC::Mmad(dstCTensor, srcATensor, srcBTensor, commonMadParams);
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::Cube1CopyOut(GlobalTensor<float> dstTensor,
                                                                                 LocalTensor<float> srcTensor,
                                                                                 const int32_t mSize,
                                                                                 const int32_t nSize,
                                                                                 DBParams& dbParam)
{
    int32_t mSizeAlign = RoundUp(mSize, (int32_t)C0_SIZE);
    int32_t nSizeAlign = RoundUp(nSize, (int32_t)C0_SIZE);
    commonFixpipeParamsV220.mSize = mSize;
    commonFixpipeParamsV220.nSize = nSizeAlign;
    commonFixpipeParamsV220.srcStride = mSizeAlign;
    commonFixpipeParamsV220.dstStride = dbParam.s1CvExtendAlign * 2;
    commonFixpipeParamsV220.quantPre = QuantMode_t::NoQuant;
    commonFixpipeParamsV220.unitFlag = 3;
    AscendC::Fixpipe<float, float, AscendC::CFG_NZ>(dstTensor, srcTensor, commonFixpipeParamsV220);
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::Cube1Compute(const GlobalTensor<T1> &left,
                                                                                  const GlobalTensor<T1> &right,
                                                                                  const GlobalTensor<float> &out,
                                                                                  const int32_t headDim,
                                                                                  DBParams& dbParam)
{
    event_t eventIdMte2ToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
    event_t eventIdMte1ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
    int64_t specify_for_v_aTensorOffsetCv = dbParam.aTensorOffsetCv / d * value_d;
    int64_t specify_for_v_bTensorOffsetCv = dbParam.bTensorOffsetCv / d * value_d;
    uint32_t mSplitSize = MMAD_BASE_SIZE;
    uint32_t mLoops = (dbParam.s1CvExtend + mSplitSize - 1) / mSplitSize;
    uint32_t mTail = dbParam.s1CvExtend - (mLoops - 1) * mSplitSize;
    uint32_t subMSizeAct = mSplitSize;
    uint32_t subMSizeActAlign = CeilDiv(subMSizeAct, C0_SIZE) * C0_SIZE;

    // n为s2方向切片
    uint32_t nSplitSize = MMAD_BASE_SIZE;
    uint32_t nLoops = (dbParam.s2CvExtend + nSplitSize - 1) / nSplitSize;
    uint32_t nTail = dbParam.s2CvExtend - (nLoops - 1) * nSplitSize;
    uint32_t subNSizeAct = nSplitSize;
    uint32_t subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;
    
    uint32_t headDimAlign = (headDim + 15) / 16 * 16;
    for (uint32_t m = 0; m < mLoops; m++) {
        subMSizeAct = m == (mLoops - 1) ? mTail : mSplitSize;
        subMSizeActAlign = CeilDiv(subMSizeAct, C0_SIZE) * C0_SIZE;
        // A矩阵
        uint64_t aL1Offset = m * mSplitSize * headDimAlign;
        LocalTensor<T1> *l0_a_tensor = ping_pong_flag_l0_a_ ? &l0_a_pong_tensor : &l0_a_ping_tensor;
        CopyGmToL1(qL1Tensor[aL1Offset],
            left[specify_for_v_aTensorOffsetCv + m * mSplitSize * dbParam.s1Stride / d * headDim], subMSizeAct,
            headDim, dbParam.s1Stride / d * headDim);
        AscendC::SetFlag<HardEvent::MTE2_MTE1>(ping_pong_flag_l0_a_);
        AscendC::WaitFlag<HardEvent::MTE2_MTE1>(ping_pong_flag_l0_a_);
        
        LoadDataAToL0((*l0_a_tensor), qL1Tensor[aL1Offset], headDim, subMSizeAct);
        for (uint32_t n = 0; n < nLoops; n++) {
            subNSizeAct = n == (nLoops - 1) ? nTail : nSplitSize;
            subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;
            uint64_t bL1Offset = n * nSplitSize * headDimAlign;
            LocalTensor<T1> *l0_b_tensor = ping_pong_flag_l0_b_ ? &l0_b_pong_tensor : &l0_b_ping_tensor;
            LocalTensor<float> *l0_c_tensor = ping_pong_flag_l0_c_ ? &l0_c_pong_tensor : &l0_c_ping_tensor;
            // B矩阵
            if (m == 0) {
                CopyGmToL1(kL1Tensor[bL1Offset],
                    right[specify_for_v_bTensorOffsetCv + n * nSplitSize * dbParam.s2Stride / d * headDim],
                    subNSizeAct, headDim, dbParam.s2Stride / d * headDim);
                AscendC::SetFlag<HardEvent::MTE2_MTE1>(ping_pong_flag_l0_b_ + 2);
                AscendC::WaitFlag<HardEvent::MTE2_MTE1>(ping_pong_flag_l0_b_ + 2);
            }
            LoadDataBToL0((*l0_b_tensor), kL1Tensor[bL1Offset], headDim, subNSizeAct);
            AscendC::WaitFlag<HardEvent::FIX_M>(ping_pong_flag_l0_c_);
            Cube1Mmad(*l0_c_tensor, *l0_a_tensor, *l0_b_tensor, subMSizeAct, subNSizeAct, headDim);
            AscendC::SetFlag<HardEvent::M_MTE1>(ping_pong_flag_l0_b_ + 3 + 2);
            if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
                Cube1CopyOut(out[pingpongIdx * cubeBaseMN + m * mSplitSize * C0_SIZE +
                                              n * nSplitSize * dbParam.s1CvExtendAlign], *l0_c_tensor, subMSizeAct,
                subNSizeAct, dbParam);
            } else {
                Cube1CopyOut(out[pingpongIdx * cubeBaseMN + m * mSplitSize * dbParam.s2CvExtend +
                                              n * nSplitSize], *l0_c_tensor, subMSizeAct, subNSizeAct, dbParam);
            }
            AscendC::SetFlag<HardEvent::FIX_M>(ping_pong_flag_l0_c_);
            ping_pong_flag_l0_b_ = 1 - ping_pong_flag_l0_b_;
            ping_pong_flag_l0_c_ = 1 - ping_pong_flag_l0_c_;
        }
        AscendC::SetFlag<HardEvent::M_MTE1>(ping_pong_flag_l0_a_ + 3);
        ping_pong_flag_l0_a_ = 1 - ping_pong_flag_l0_a_;
    }
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::ComputeMM1(DBParams& dbParam)
{
    pingpongIdx = dbParam.taskId % 2;
    dbParam.actualS1Len = s1;
    dbParam.actualS2Len = s2;
    dbParam.s1Stride = 0;
    dbParam.s2Stride = 0;
    if constexpr (INPUT_LAYOUT == TND) {
        UpdateToken(dbParam.bIdx);
        GetSeqQlenKvlenByBidx(dbParam.bIdx, dbParam.actualS1Len, dbParam.actualS2Len);
        dbParam.aTensorOffsetCv = 0;
        dbParam.aTensorOffsetCv_rope = 0;
        dbParam.bTensorOffsetCv = 0;
        dbParam.bTensorOffsetCv_rope = 0;
        if (dbParam.bIdx > 0) {
            dbParam.aTensorOffsetCv = ((__gm__ int64_t *)actual_seq_qlen_addr)[dbParam.bIdx - 1] * n2 * g * d;
            dbParam.aTensorOffsetCv_rope = ((__gm__ int64_t *)actual_seq_qlen_addr)[dbParam.bIdx - 1] * n2 * g * rope_d;
            dbParam.bTensorOffsetCv  = ((__gm__ int64_t *)actual_seq_kvlen_addr)[dbParam.bIdx - 1] * n2 * d;
            dbParam.bTensorOffsetCv_rope  = ((__gm__ int64_t *)actual_seq_kvlen_addr)[dbParam.bIdx - 1] * n2 * rope_d;
        }
        dbParam.aTensorOffsetCv += ((dbParam.s1oIdx * s1CvInner * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * d;
        dbParam.aTensorOffsetCv_rope += ((dbParam.s1oIdx * s1CvInner * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * rope_d;
        dbParam.bTensorOffsetCv += (dbParam.s2oIdx * s2CvInner * n2 + dbParam.n2Idx) * d;
        dbParam.bTensorOffsetCv_rope += (dbParam.s2oIdx * s2CvInner * n2 + dbParam.n2Idx) * rope_d;
        dbParam.s1Stride = n2 * g * d;
        dbParam.s2Stride = n2 * d;
    } else if constexpr (INPUT_LAYOUT == BNGSD) {
        dbParam.aTensorOffsetCv = (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1 + dbParam.s1oIdx * s1CvInner) * d;
        dbParam.aTensorOffsetCv_rope = (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1 + dbParam.s1oIdx * s1CvInner) * rope_d;
        dbParam.bTensorOffsetCv = ((dbParam.bIdx * n2 + dbParam.n2Idx) * s2 + dbParam.s2oIdx * s2CvInner) * d;
        dbParam.bTensorOffsetCv_rope = ((dbParam.bIdx * n2 + dbParam.n2Idx) * s2 + dbParam.s2oIdx * s2CvInner) * rope_d;
        dbParam.s1Stride = d;
        dbParam.s2Stride = d;
    } else if constexpr (INPUT_LAYOUT == SBNGD) {
        dbParam.aTensorOffsetCv = ((((dbParam.s1oIdx * s1CvInner) * b + dbParam.bIdx) * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * d;
        dbParam.aTensorOffsetCv_rope = ((((dbParam.s1oIdx * s1CvInner) * b + dbParam.bIdx) * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * rope_d;
        dbParam.bTensorOffsetCv = ((dbParam.s2oIdx * s2CvInner * b + dbParam.bIdx) * n2 + dbParam.n2Idx) * d;
        dbParam.bTensorOffsetCv_rope = ((dbParam.s2oIdx * s2CvInner * b + dbParam.bIdx) * n2 + dbParam.n2Idx) * rope_d;
        dbParam.s1Stride = b * n2 * g * d;
        dbParam.s2Stride = b * n2 * d;
    } else if constexpr (INPUT_LAYOUT == BSNGD) {
        dbParam.aTensorOffsetCv = (((dbParam.bIdx * s1 + dbParam.s1oIdx * s1CvInner) * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * d;
        dbParam.aTensorOffsetCv_rope = (((dbParam.bIdx * s1 + dbParam.s1oIdx * s1CvInner) * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * rope_d;
        dbParam.bTensorOffsetCv = ((dbParam.bIdx * s2 + dbParam.s2oIdx * s2CvInner) * n2 + dbParam.n2Idx) * d;
        dbParam.bTensorOffsetCv_rope = ((dbParam.bIdx * s2 + dbParam.s2oIdx * s2CvInner) * n2 + dbParam.n2Idx) * rope_d;
        dbParam.s1Stride = n2 * g * d;
        dbParam.s2Stride = n2 * d;
    }

    int64_t s1_size = dbParam.actualS1Len;
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        s1_size = dbParam.s1CvExtendAlign;
    }

    // dy * v need align the d to value_d.
    int64_t specify_for_v_aTensorOffsetCv = dbParam.aTensorOffsetCv / d * value_d;
    int64_t specify_for_v_bTensorOffsetCv = dbParam.bTensorOffsetCv / d * value_d;
    int64_t specify_for_v_s1Stride = dbParam.s1Stride / d * value_d;
    int64_t specify_for_v_s2Stride = dbParam.s2Stride / d * value_d;

    // m为s1方向切片
    uint32_t mSplitSize = MMAD_BASE_SIZE;
    uint32_t mLoops = (dbParam.s1CvExtend + mSplitSize - 1) / mSplitSize;
    uint32_t mTail = dbParam.s1CvExtend - (mLoops - 1) * mSplitSize;
    uint32_t subMSizeAct = mSplitSize;
    uint32_t subMSizeActAlign = CeilDiv(subMSizeAct, C0_SIZE) * C0_SIZE;

    // n为s2方向切片
    uint32_t nSplitSize = MMAD_BASE_SIZE;
    uint32_t nLoops = (dbParam.s2CvExtend + nSplitSize - 1) / nSplitSize;
    uint32_t nTail = dbParam.s2CvExtend - (nLoops - 1) * nSplitSize;
    uint32_t subNSizeAct = nSplitSize;
    uint32_t subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;

    event_t eventIdMte1ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
    event_t eventIdMte2ToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
    event_t eventIdMToFixpipe = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_FIX));
    event_t eventIdFixpipeToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::FIX_M));
    if (DTEMPLATETYPE == DTemplateType::Aligned64){
        // mm-dyv
        mm1.SetOrgShape(s1_size, dbParam.actualS2Len, specify_for_v_s1Stride, specify_for_v_s2Stride, dbParam.s2CvExtendAlign);
        for (uint32_t m = 0; m < mLoops; m++) {
            subMSizeAct = m == (mLoops - 1) ? mTail : mSplitSize;
            subMSizeActAlign = CeilDiv(subMSizeAct, C0_SIZE) * C0_SIZE;
            // A矩阵
            uint64_t aL1Offset = m * mSplitSize * value_dAlign;
            CopyGmToL1(dxL1Tensor[aL1Offset],
                dxGm[specify_for_v_aTensorOffsetCv + m * mSplitSize * dbParam.s1Stride / d * value_d], subMSizeAct,
                value_d, dbParam.s1Stride / d * value_d);
            for (uint32_t n = 0; n < nLoops; n++) {
                subNSizeAct = n == (nLoops - 1) ? nTail : nSplitSize;
                subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;
                uint64_t bL1Offset = value_dAlign * n * nSplitSize;
                // B矩阵
                if (m == 0) {
                    CopyGmToL1(vL1Tensor[bL1Offset],
                        valueGm[specify_for_v_bTensorOffsetCv + n * nSplitSize * dbParam.s2Stride / d * value_d],
                        subNSizeAct, value_d, dbParam.s2Stride / d * value_d);
                }
                AscendC::SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                AscendC::WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                mm1.SetTensorA(dxL1Tensor[aL1Offset]);
                mm1.SetTensorB(vL1Tensor[bL1Offset], true);
                mm1.SetTail(subMSizeAct, subNSizeAct, value_d);
                mm1.template Iterate<false>();
                if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
                    mm1.GetTensorC(mm1WorkspaceGm[pingpongIdx * cubeBaseMN + m * mSplitSize * C0_SIZE +
                                                  n * nSplitSize * dbParam.s1CvExtendAlign]);
                } else {
                    mm1.GetTensorC(mm1WorkspaceGm[pingpongIdx * cubeBaseMN + m * mSplitSize * dbParam.s2CvExtend +
                                                  n * nSplitSize]);
                }
            }
        }
        AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        mm1.End();

        // mm-qk
        mm1.SetOrgShape(s1_size, dbParam.actualS2Len, dbParam.s1Stride, dbParam.s2Stride, dbParam.s2CvExtendAlign);
        for (uint32_t m = 0; m < mLoops; m++) {
            subMSizeAct = m == (mLoops - 1) ? mTail : mSplitSize;
            subMSizeActAlign = CeilDiv(subMSizeAct, C0_SIZE) * C0_SIZE;
            uint64_t aL1Offset = m * mSplitSize * dAlign;
            // A矩阵
            CopyGmToL1(qL1Tensor[aL1Offset],
                queryGm[dbParam.aTensorOffsetCv + m * mSplitSize * dbParam.s1Stride], subMSizeAct, d,
                dbParam.s1Stride / d * d);
            for (uint32_t n = 0; n < nLoops; n++) {
                subNSizeAct = n == (nLoops - 1) ? nTail : nSplitSize;
                subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;
                uint64_t bL1Offset = dAlign * n * nSplitSize;
                // B矩阵
                if (m == 0) {
                    CopyGmToL1(kL1Tensor[bL1Offset],
                        keyGm[dbParam.bTensorOffsetCv + n * nSplitSize * dbParam.s2Stride], subNSizeAct, d,
                        dbParam.s2Stride / d * d);
                }
                AscendC::SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                AscendC::WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                mm1.SetTail(subMSizeAct, subNSizeAct, d);
                mm1.SetTensorA(qL1Tensor[aL1Offset]);
                mm1.SetTensorB(kL1Tensor[bL1Offset], true);
                mm1.template Iterate<false>();
                if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
                    mm1.GetTensorC(mm2WorkspaceGm[pingpongIdx * cubeBaseMN +
                        m * mSplitSize * C0_SIZE + n * nSplitSize * dbParam.s1CvExtendAlign]);
                } else {
                    mm1.GetTensorC(mm2WorkspaceGm[pingpongIdx * cubeBaseMN +
                        m * mSplitSize * dbParam.s2CvExtend + n * nSplitSize]);
                }
            }
        }
        AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        mm1.End();
    } else if (DTEMPLATETYPE == DTemplateType::Aligned128){
        AscendC::SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_ID3);
        AscendC::SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_ID4);
        AscendC::SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_ID5);
        AscendC::SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_ID6);
        AscendC::SetFlag<HardEvent::FIX_M>(eventIdFixpipeToM_ID0);
        AscendC::SetFlag<HardEvent::FIX_M>(eventIdFixpipeToM_ID1);
        Cube1Compute(queryGm, keyGm, mm2WorkspaceGm, d, dbParam);
        AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        Cube1Compute(dxGm, valueGm, mm1WorkspaceGm, value_d, dbParam);
        AscendC::WaitFlag<HardEvent::FIX_M>(eventIdFixpipeToM_ID0);
        AscendC::WaitFlag<HardEvent::FIX_M>(eventIdFixpipeToM_ID1);
        AscendC::WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_ID3);
        AscendC::WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_ID4);
        AscendC::WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_ID5);
        AscendC::WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_ID6);
        AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
    } else if (DTEMPLATETYPE == DTemplateType::Aligned192) {
        uint32_t dSplitSize = (d / 2 + C0_SIZE - 1) / C0_SIZE * C0_SIZE;
        uint32_t dLoops = (d + dSplitSize - 1) / dSplitSize;
        uint32_t dTail = d - (dLoops - 1) * dSplitSize;
        uint32_t subDSizeAct = dSplitSize;
        uint32_t subDSizeActAlign = CeilDiv(subDSizeAct, C0_SIZE) * C0_SIZE;

        uint32_t valueDSplitSize = value_d > MMAD_BASE_SIZE ?
                                   ((value_d + 1) / 2 + C0_SIZE - 1) / C0_SIZE * C0_SIZE : value_d;
        uint32_t valueDLoops = (value_d + dSplitSize - 1) / valueDSplitSize;
        uint32_t valueDTail = value_d - (valueDLoops - 1) * valueDSplitSize;
        uint32_t subValueDSizeAct = valueDSplitSize;
        uint32_t subValueDSizeActAligned = CeilDiv(subValueDSizeAct, C0_SIZE) * C0_SIZE;

        constexpr int32_t s1SliceSize = S_SPLITT_SIZE;
        int32_t s1SliceNum = (dbParam.s1CvExtend + s1SliceSize - 1) / s1SliceSize;
        int32_t s1SliceTail = dbParam.s1CvExtend - (s1SliceNum - 1) * s1SliceSize;

        // mm-dyv
        mm1.SetOrgShape(s1_size, dbParam.actualS2Len, specify_for_v_s1Stride, specify_for_v_s2Stride,
                        dbParam.s2CvExtendAlign);
        for (int32_t sliceIdx = 0; sliceIdx < s1SliceNum; ++sliceIdx) {
            int32_t s1SliceExtend = sliceIdx == (s1SliceNum - 1) ? s1SliceTail : s1SliceSize;
            int32_t s1SliceExtendAlign = CeilDiv(s1SliceExtend, C0_SIZE) * C0_SIZE;
            uint32_t mInnerLoops = (s1SliceExtend + mSplitSize - 1) / mSplitSize;
            uint32_t mInnerTail = s1SliceExtend - (mInnerLoops - 1) * mSplitSize;
            for (uint32_t n = 0; n < nLoops; n++) {
                subNSizeAct = n == (nLoops - 1) ? nTail : nSplitSize;
                subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;
                for (uint32_t m = 0; m < mInnerLoops; m++) {
                    subMSizeAct = m == (mInnerLoops - 1) ? mInnerTail : mSplitSize;
                    subMSizeActAlign = CeilDiv(subMSizeAct, C0_SIZE) * C0_SIZE;
                    uint64_t aL1Offset = (m * mSplitSize + sliceIdx * s1SliceSize) * value_dAlign;
                    uint64_t bL1Offset = value_dAlign * n * nSplitSize;
                    // A矩阵
                    if (n == 0) {
                        CopyGmToL1(dxL1Tensor[aL1Offset],
                            dxGm[specify_for_v_aTensorOffsetCv +
                                (m * mSplitSize + sliceIdx * s1SliceSize) * dbParam.s1Stride / d * value_d],
                            subMSizeAct, value_d,  dbParam.s1Stride / d * value_d);
                    }
                    // B矩阵
                    if (m == 0 && sliceIdx == 0) {
                        CopyGmToL1(vL1Tensor[bL1Offset],
                            valueGm[specify_for_v_bTensorOffsetCv + n * nSplitSize * dbParam.s2Stride / d * value_d],
                            subNSizeAct, value_d, dbParam.s2Stride / d * value_d);
                    }
                    AscendC::SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                    AscendC::WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                    for (uint32_t vdLoopIdx = 0; vdLoopIdx < valueDLoops; vdLoopIdx++) {
                        bool atomicAddFlag = vdLoopIdx != 0;
                        subValueDSizeAct = vdLoopIdx == (valueDLoops - 1) ? valueDTail : valueDSplitSize;
                        subValueDSizeActAligned = CeilDiv(subValueDSizeAct, C0_SIZE) * C0_SIZE;
                        mm1.SetTail(subMSizeAct, subNSizeAct, subValueDSizeAct);
                        mm1.SetTensorA(dxL1Tensor[aL1Offset + vdLoopIdx * valueDSplitSize * subMSizeActAlign]);
                        mm1.SetTensorB(vL1Tensor[bL1Offset + vdLoopIdx * valueDSplitSize * subNSizeActAlign], true);
                        mm1.template Iterate<false>(atomicAddFlag);
                    }
                    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
                        mm1.GetTensorC(mm1WorkspaceGm[pingpongIdx * cubeBaseMN +
                            (m * mSplitSize + sliceIdx * s1SliceSize) * C0_SIZE + 
                            n * nSplitSize * dbParam.s1CvExtendAlign], 0);
                    } else {
                        mm1.GetTensorC(mm1WorkspaceGm[pingpongIdx * cubeBaseMN +
                            (m * mSplitSize + sliceIdx * s1SliceSize) * dbParam.s2CvExtend + n * nSplitSize], 0);
                    }
                }
            }
        }
        AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        mm1.End();

        // mm-qk
        mm1.SetOrgShape(s1_size, dbParam.actualS2Len, dbParam.s1Stride, dbParam.s2Stride, dbParam.s2CvExtendAlign);
        for (int32_t sliceIdx = 0; sliceIdx < s1SliceNum; ++sliceIdx) {
            int32_t s1SliceExtend = sliceIdx == (s1SliceNum - 1) ? s1SliceTail : s1SliceSize;
            int32_t s1SliceExtendAlign = CeilDiv(s1SliceExtend, C0_SIZE) * C0_SIZE;
            uint32_t mInnerLoops = (s1SliceExtend + mSplitSize - 1) / mSplitSize;
            uint32_t mInnerTail = s1SliceExtend - (mInnerLoops - 1) * mSplitSize;
            for (uint32_t n = 0; n < nLoops; n++) {
                subNSizeAct = n == (nLoops - 1) ? nTail : nSplitSize;
                subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;
                uint64_t bL1Offset = dAlign * n * nSplitSize;
                for (uint32_t m = 0; m < mInnerLoops; m++) {
                    subMSizeAct = m == (mInnerLoops - 1) ? mInnerTail : mSplitSize;
                    subMSizeActAlign = CeilDiv(subMSizeAct, C0_SIZE) * C0_SIZE;
                    uint64_t aL1Offset = (m * mSplitSize + sliceIdx * s1SliceSize) * dAlign;
                    // A矩阵
                    if (n == 0) {
                        CopyGmToL1(
                            qL1Tensor[aL1Offset],
                            queryGm[dbParam.aTensorOffsetCv +
                                (m * mSplitSize + sliceIdx * s1SliceSize) * dbParam.s1Stride],
                            subMSizeAct, d,  dbParam.s1Stride);
                    }
                    // B矩阵
                    if (m == 0 && sliceIdx == 0) {
                        CopyGmToL1(kL1Tensor[bL1Offset],
                            keyGm[dbParam.bTensorOffsetCv + n * nSplitSize * dbParam.s2Stride],
                                subNSizeAct, d,  dbParam.s2Stride);
                    }
                    AscendC::SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                    AscendC::WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                    for (uint32_t dLoopIdx = 0; dLoopIdx < dLoops; dLoopIdx++) {
                        bool atomicAddFlag = dLoopIdx != 0;
                        subDSizeAct = dLoopIdx == (dLoops - 1) ? dTail : dSplitSize;
                        subDSizeActAlign = CeilDiv(subDSizeAct, C0_SIZE) * C0_SIZE;
                        mm1.SetTail(subMSizeAct, subNSizeAct, subDSizeAct);
                        mm1.SetTensorA(qL1Tensor[subMSizeActAlign * dLoopIdx * dSplitSize + aL1Offset]);
                        mm1.SetTensorB(kL1Tensor[bL1Offset + dLoopIdx * dSplitSize * subNSizeActAlign], true);
                        mm1.template Iterate<false>(atomicAddFlag);
                    }
                    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
                        mm1.GetTensorC(mm2WorkspaceGm[pingpongIdx * cubeBaseMN +
                            (m * mSplitSize + sliceIdx * s1SliceSize) * C0_SIZE +
                            n * nSplitSize * dbParam.s1CvExtendAlign], 0);
                    } else {
                        mm1.GetTensorC(mm2WorkspaceGm[pingpongIdx * cubeBaseMN +
                            (m * mSplitSize + sliceIdx * s1SliceSize) * dbParam.s2CvExtend + n * nSplitSize], 0);
                    }
                }
            }
        }
        AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        mm1.End();
    } else {
        // mm-dyv
        mm1.SetOrgShape(s1_size, dbParam.actualS2Len, specify_for_v_s1Stride, specify_for_v_s2Stride, dbParam.s2CvExtendAlign);
        mm1.SetTail(dbParam.s1CvExtend, dbParam.s2CvExtend, value_d); // M N K
        mm1.SetTensorA(dxGm[specify_for_v_aTensorOffsetCv]);
        mm1.SetTensorB(valueGm[specify_for_v_bTensorOffsetCv], true);
        mm1.template IterateAll<false>(mm1WorkspaceGm[pingpongIdx * cubeBaseMN], false);
        // mm-qk
        // mm m, n, ka, kb, kc
        mm1.SetOrgShape(s1_size, dbParam.actualS2Len, dbParam.s1Stride, dbParam.s2Stride, dbParam.s2CvExtendAlign);
        mm1.SetTail(dbParam.s1CvExtend, dbParam.s2CvExtend, d); // M N K
        mm1.SetTensorA(queryGm[dbParam.aTensorOffsetCv]);
        mm1.SetTensorB(keyGm[dbParam.bTensorOffsetCv], true);
        mm1.template IterateAll<false>(mm2WorkspaceGm[pingpongIdx * cubeBaseMN], false);
        // mm-qk 2
        if constexpr (HAS_ROPE == ENABLE) {
            // mm m, n, ka, kb, kc
            mm1.SetOrgShape(s1_size, dbParam.actualS2Len, dbParam.s1Stride / d * rope_d, dbParam.s2Stride / d * rope_d, dbParam.s2CvExtendAlign);
            mm1.SetTail(dbParam.s1CvExtend, dbParam.s2CvExtend, rope_d); // M N K
            mm1.SetTensorA(queryRopeGm[dbParam.aTensorOffsetCv_rope]);
            mm1.SetTensorB(keyRopeGm[dbParam.bTensorOffsetCv_rope], true);
            mm1.template IterateAll<false>(mm2WorkspaceGm[pingpongIdx * cubeBaseMN], true);
        }
    }
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::DTMComputeMMDqkv(
    DBParams& dbParam, int64_t nextBlockId)
{
    pingpongIdx = dbParam.taskId % 2;
    int64_t dqOffset = 0;
    int64_t dqRopeOffset = 0;
    int64_t dkvOffset = 0;
    int64_t dkRopeOffset = 0;
    int64_t dvOffset = 0;
    int64_t dqOrgN = 0;
    if constexpr (MM2_OUT_FORMAT == CubeFormat::ND) {
        dqOffset = dbParam.aTensorOffsetCv;
        dqRopeOffset = dbParam.aTensorOffsetCv_rope;
        dkvOffset = dbParam.bTensorOffsetCv;
        dkRopeOffset = dbParam.bTensorOffsetCv_rope;
        dvOffset = dbParam.bTensorOffsetCv / d * value_d;
    } else if constexpr (INPUT_LAYOUT == TND) {  // TND Nz
        UpdateToken(dbParam.bIdx);
        if (dbParam.bIdx > 0) {
            dqOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[dbParam.bIdx - 1] * n2 * g * dAlign;
            dqRopeOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[dbParam.bIdx - 1] * n2 * g * rope_dAlign;
            dkvOffset = ((__gm__ int64_t *)actual_seq_kvlen_addr)[dbParam.bIdx - 1] * n2 * dAlign;
            dkRopeOffset = ((__gm__ int64_t *)actual_seq_kvlen_addr)[dbParam.bIdx - 1] * n2 * rope_dAlign;
            dvOffset = ((__gm__ int64_t *)actual_seq_kvlen_addr)[dbParam.bIdx - 1] * n2 * value_dAlign;
        }
        dqOffset += ((dbParam.n2Idx * g + dbParam.gIdx) * dbParam.actualS1Len) * dAlign + dbParam.s1oIdx * s1CvInner * C0_SIZE;
        dqRopeOffset += ((dbParam.n2Idx * g + dbParam.gIdx) * dbParam.actualS1Len) * rope_dAlign + dbParam.s1oIdx * s1CvInner * C0_SIZE;
        dkvOffset += (dbParam.n2Idx * dbParam.actualS2Len) * dAlign + dbParam.s2oIdx * s2CvInner * C0_SIZE;
        dkRopeOffset += (dbParam.n2Idx * dbParam.actualS2Len) * rope_dAlign + dbParam.s2oIdx * s2CvInner * C0_SIZE;
        dvOffset += (dbParam.n2Idx * dbParam.actualS2Len) * value_dAlign + dbParam.s2oIdx * s2CvInner * C0_SIZE;
    } else {  // Other Nz
        dqOffset = (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1) * dAlign + dbParam.s1oIdx * s1CvInner * C0_SIZE;
        dqRopeOffset = (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1) * rope_dAlign + dbParam.s1oIdx * s1CvInner * C0_SIZE;
        dkvOffset = ((dbParam.bIdx * n2 + dbParam.n2Idx) * s2) * dAlign + dbParam.s2oIdx * s2CvInner * C0_SIZE;
        dkRopeOffset = ((dbParam.bIdx * n2 + dbParam.n2Idx) * s2) * rope_dAlign + dbParam.s2oIdx * s2CvInner * C0_SIZE;
        dvOffset = ((dbParam.bIdx * n2 + dbParam.n2Idx) * s2) * value_dAlign + dbParam.s2oIdx * s2CvInner * C0_SIZE;
    }

    uint64_t dqKc = dbParam.s1Stride;
    uint64_t dkvKc = dbParam.s2Stride;
    uint64_t dvKc = dbParam.s2Stride / d * value_d;
    uint64_t dqOutMStride = dbParam.actualS1Len;
    uint64_t dkvOutMStride = dbParam.actualS2Len;
    uint64_t dvOutMStride = dbParam.actualS2Len;
    if (dbParam.dqGroupId[cCubeBlockIdx] != OUTIDX) { //mm写出到dqDtmWsGm上
        dqOffset = pingpongIdx * cubeCoreNum * s1CvInner * dAlign + cCubeBlockIdx * s1CvInner * dAlign;
        dqRopeOffset = pingpongIdx * cubeCoreNum * s1CvInner * rope_dAlign + cCubeBlockIdx * s1CvInner * rope_dAlign;
        dqKc = d;
        dqOutMStride = dbParam.s1CvExtend;
    }
    if (dbParam.kvGroupId[cCubeBlockIdx] != OUTIDX) {
        dkvOffset = pingpongIdx * cubeCoreNum * s2CvInner * dAlign + cCubeBlockIdx * s2CvInner * dAlign;
        dkRopeOffset = pingpongIdx * cubeCoreNum * s2CvInner * rope_dAlign + cCubeBlockIdx * s2CvInner * rope_dAlign;
        dvOffset = pingpongIdx * cubeCoreNum * s2CvInner * value_dAlign + cCubeBlockIdx * s2CvInner * value_dAlign;
        dkvKc = d;
        dvKc = value_d;
        dkvOutMStride = dbParam.s2CvExtend;
        dvOutMStride = dbParam.s2CvExtend;
    }

    uint64_t s1_size = dbParam.actualS1Len;
    uint64_t s2_size = dbParam.s2CvExtendAlign * DTYPE_FACTOR;
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        s1_size = dbParam.s1CvExtendAlign * DTYPE_FACTOR;
        s2_size = dbParam.s2CvExtendAlign;
    }
    if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
        dqKc = d;
        dkvKc = d;
        dvKc = value_d;
    }

    ///////////////////////////////////////////////////////////////
    // Matmal4 dq
    ///////////////////////////////////////////////////////////////
    // left [B, N2, G, S1, s2] right [B, N2, 1, S2, D] output [B, N2, G, S1, D]
    if (MM2_OUT_FORMAT == CubeFormat::NZ) {
        mm4.SetUserDefInfo(dqOutMStride);
    }

    mm4.SetOrgShape(s1_size, dbParam.s2Stride, s2_size, dbParam.actualS2Len, dqKc);
    mm4.SetTail(dbParam.s1CvExtend, d, dbParam.s2CvExtend); // M N K
    mm4.SetTensorA(mulWorkSpaceGm[pingpongIdx * cubeBaseMN * 2]);
    mm4.SetTensorB(keyGm[dbParam.bTensorOffsetCv]);
    if (dbParam.dqGroupId[cCubeBlockIdx] == OUTIDX) {
        mm4.template IterateAll<false>(dqWorkSpaceGm[dqOffset], 0);
    } else {
        mm4.template IterateAll<false>(dqDtmWsGm[dqOffset], 0);
    }
    mm4.End();
    if constexpr (HAS_ROPE == ENABLE) {
        ///////////////////////////////////////////////////////////////
        // Matmal4 dq rope
        ///////////////////////////////////////////////////////////////
        // left [B, N2, G, S1, s2] right [B, N2, 1, S2, D] output [B, N2, G, S1, D]
        if (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm4.SetUserDefInfo(dqOutMStride);
        }

        mm4.SetOrgShape(s1_size, dbParam.s2Stride / d * rope_d, s2_size, dbParam.actualS2Len, dqKc / d * rope_d);
        mm4.SetTail(dbParam.s1CvExtend, rope_d, dbParam.s2CvExtend); // M N K
        mm4.SetTensorA(mulWorkSpaceGm[pingpongIdx * cubeBaseMN * 2]);
        mm4.SetTensorB(keyRopeGm[dbParam.bTensorOffsetCv_rope]);
        if (dbParam.dqGroupId[cCubeBlockIdx] == OUTIDX) {
            mm4.template IterateAll<false>(dqRopeWorkSpaceGm[dqRopeOffset], 0);
        } else {
            mm4.template IterateAll<false>(dqRopeDtmWsGm[dqRopeOffset], 0);
        }
        mm4.End();
    }

    ///////////////////////////////////////////////////////////////
    // Matmal4 dk
    ///////////////////////////////////////////////////////////////
    // left [B, N2, G, S1, S2] right [B, N2, 1, S1, D] output [B, N2, G, S2, D]
    if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
        mm3.SetUserDefInfo(dkvOutMStride);
    }

    mm3.SetOrgShape(s2_size, dbParam.s1Stride, s1_size, dbParam.actualS1Len, dkvKc);
    mm3.SetTail(dbParam.s2CvExtend, d, dbParam.s1CvExtend);
    mm3.SetTensorA(mulWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], true);
    mm3.SetTensorB(queryGm[dbParam.aTensorOffsetCv]);
    if (dbParam.kvGroupId[cCubeBlockIdx] == OUTIDX) {
        mm3.template IterateAll<false>(dkWorkSpaceGm[dkvOffset], 0);
    } else {
        mm3.template IterateAll<false>(dkDtmWsGm[dkvOffset], 0);
    }
    mm3.End();

    if constexpr (HAS_ROPE == ENABLE) {
        ///////////////////////////////////////////////////////////////
        // Matmal4 dk rope
        ///////////////////////////////////////////////////////////////
        // left [B, N2, G, S1, S2] right [B, N2, 1, S1, D] output [B, N2, G, S2, D]
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm3.SetUserDefInfo(dkvOutMStride);
        }

        mm3.SetOrgShape(s2_size, dbParam.s1Stride / d * rope_d, s1_size, dbParam.actualS1Len, dkvKc /  d * rope_d);
        mm3.SetTail(dbParam.s2CvExtend, rope_d, dbParam.s1CvExtend);
        mm3.SetTensorA(mulWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], true);
        mm3.SetTensorB(queryRopeGm[dbParam.aTensorOffsetCv_rope]);
        if (dbParam.kvGroupId[cCubeBlockIdx] == OUTIDX) {
            mm3.template IterateAll<false>(dkRopeWorkSpaceGm[dkRopeOffset], 0);
        } else {
            mm3.template IterateAll<false>(dkRopeDtmWsGm[dkRopeOffset], 0);
        }
        mm3.End();
    }

    ///////////////////////////////////////////////////////////////
    // Matmal5 dv
    ///////////////////////////////////////////////////////////////
    // left [B, N2, G, S1, S2] right [B, N2, G, S1, D2] output [B, N2, 1, S2, D2]
    if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
        mm3.SetUserDefInfo(dvOutMStride);
    }
    mm3.SetOrgShape(s2_size, dbParam.s1Stride / d * value_d, s1_size, dbParam.actualS1Len, dvKc);
    mm3.SetTail(dbParam.s2CvExtend, value_d, dbParam.s1CvExtend);
    mm3.SetTensorA(dropWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], true);
    mm3.SetTensorB(dxGm[dbParam.aTensorOffsetCv / d * value_d]);
    if (dbParam.kvGroupId[cCubeBlockIdx] == OUTIDX) {
        if constexpr (IsSameType<T1, float>::value) {
            mm3.template IterateAll<false>(dvGm[dvOffset], 0);
        } else {
            mm3.template IterateAll<false>(dvWorkSpaceGm[dvOffset], 0);
        }
     } else {
        mm3.template IterateAll<false>(dvDtmWsGm[dvOffset], 0);
     }
    mm3.End();
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::ComputeMMDqkv(
    DBParams& dbParam, int64_t nextBlockId)
{
    pingpongIdx = dbParam.taskId % 2;
    int64_t dqOffset = 0;
    int64_t dqRopeOffset = 0;
    int64_t dkvOffset = 0;
    int64_t dkRopeOffset = 0;
    int64_t dvOffset = 0;
    if constexpr (MM2_OUT_FORMAT == CubeFormat::ND) {
        dqOffset = dbParam.aTensorOffsetCv;
        dkvOffset = dbParam.bTensorOffsetCv;
        if constexpr (HAS_ROPE == ENABLE) {
            dqRopeOffset = dbParam.aTensorOffsetCv_rope;
            dkRopeOffset = dbParam.bTensorOffsetCv_rope;
        }
        dvOffset = dbParam.bTensorOffsetCv / d * value_d;
    } else if constexpr (INPUT_LAYOUT == TND) {  // TND Nz
        UpdateToken(dbParam.bIdx);
        if (dbParam.bIdx > 0) {
            dqOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[dbParam.bIdx - 1] * n2 * g * dAlign;
            dkvOffset = ((__gm__ int64_t *)actual_seq_kvlen_addr)[dbParam.bIdx - 1] * n2 * dAlign;
            dvOffset = ((__gm__ int64_t *)actual_seq_kvlen_addr)[dbParam.bIdx - 1] * n2 * value_dAlign;
            if constexpr (HAS_ROPE == ENABLE) {
                dqRopeOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[dbParam.bIdx - 1] * n2 * g * rope_dAlign;
                dkRopeOffset = ((__gm__ int64_t *)actual_seq_kvlen_addr)[dbParam.bIdx - 1] * n2 * rope_dAlign;
            }
        }
        dqOffset += ((dbParam.n2Idx * g + dbParam.gIdx) * dbParam.actualS1Len) * dAlign + dbParam.s1oIdx * s1CvInner * C0_SIZE;
        dkvOffset += (dbParam.n2Idx * dbParam.actualS2Len) * dAlign + dbParam.s2oIdx * s2CvInner * C0_SIZE;
        dvOffset += (dbParam.n2Idx * dbParam.actualS2Len) * value_dAlign + dbParam.s2oIdx * s2CvInner * C0_SIZE;
        if constexpr (HAS_ROPE == ENABLE) {
            dqRopeOffset += ((dbParam.n2Idx * g + dbParam.gIdx) * dbParam.actualS1Len) * rope_dAlign + dbParam.s1oIdx * s1CvInner * C0_SIZE;
            dkRopeOffset += (dbParam.n2Idx * dbParam.actualS2Len) * rope_dAlign + dbParam.s2oIdx * s2CvInner * C0_SIZE;
        }
    } else {  // Other Nz
        dqOffset = (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1) * dAlign + dbParam.s1oIdx * s1CvInner * C0_SIZE;
        dkvOffset = ((dbParam.bIdx * n2 + dbParam.n2Idx) * s2) * dAlign + dbParam.s2oIdx * s2CvInner * C0_SIZE;
        dvOffset = ((dbParam.bIdx * n2 + dbParam.n2Idx) * s2) * value_dAlign + dbParam.s2oIdx * s2CvInner * C0_SIZE;
        if constexpr (HAS_ROPE == ENABLE) {
            dqRopeOffset = (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1) * rope_dAlign + dbParam.s1oIdx * s1CvInner * C0_SIZE;
            dkRopeOffset = ((dbParam.bIdx * n2 + dbParam.n2Idx) * s2) * rope_dAlign + dbParam.s2oIdx * s2CvInner * C0_SIZE;
        }
    }

    uint64_t dqKc = dbParam.s1Stride;
    uint64_t dkvKc = dbParam.s2Stride;
    uint64_t dvKc = dbParam.s2Stride / d * value_d;
    uint64_t s1_size = dbParam.actualS1Len;

    uint64_t s2_size = dbParam.s2CvExtendAlign * DTYPE_FACTOR;
    uint64_t dvOutMStride = dbParam.actualS2Len;
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        s1_size = dbParam.s1CvExtendAlign * DTYPE_FACTOR;
        s2_size = dbParam.s2CvExtendAlign;
    }
    if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
        dqKc = d;
        dkvKc = d;
        dvKc = value_d;
    }

    event_t eventIdMte1ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
    event_t eventIdMte2ToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
    // mm4 切分s1
    constexpr int32_t s1SliceSize = S_SPLITT_SIZE;
    int32_t s1SliceNum = (dbParam.s1CvExtend + s1SliceSize - 1) / s1SliceSize;
    int32_t s1SliceTail = dbParam.s1CvExtend - (s1SliceNum - 1) * s1SliceSize;

    // mm3 切分s2
    constexpr int32_t s2SliceSize = S_SPLITT_SIZE;
    int32_t s2SliceNum = (dbParam.s2CvExtend + s2SliceSize - 1) / s2SliceSize;
    int32_t s2SliceTail = dbParam.s2CvExtend - (s2SliceNum - 1) * s2SliceSize;

    // m是s1方向的循环
    uint32_t mSplitSize = MMAD_BASE_SIZE;
    uint32_t mLoops = (dbParam.s1CvExtend + mSplitSize - 1) / mSplitSize;
    uint32_t mTail = dbParam.s1CvExtend - (mLoops - 1) * mSplitSize;
    uint32_t subMSizeAct = mSplitSize;
    uint32_t subMSizeActAlign = CeilDiv(subMSizeAct, C0_SIZE) * C0_SIZE;

    // n 是s2方向的循环
    uint32_t nSplitSize = MMAD_BASE_SIZE;
    uint32_t nLoops = (dbParam.s2CvExtend + nSplitSize - 1) / nSplitSize;
    uint32_t nTail = dbParam.s2CvExtend - (nLoops - 1) * nSplitSize;
    uint32_t subNSizeAct = nSplitSize;
    uint32_t subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;

    // d方向循环
    uint32_t dSplitSize = MMAD_BASE_SIZE;
    uint32_t dLoops = (d + dSplitSize - 1) / dSplitSize;
    uint32_t dTail = d - (dLoops - 1) * dSplitSize;
    uint32_t subDSizeAct = dSplitSize;
    uint32_t subDSizeActAlign = CeilDiv(subDSizeAct, C0_SIZE) * C0_SIZE;

    // value_d方向循环
    uint32_t valueDLoops = (value_d + dSplitSize - 1) / dSplitSize;
    uint32_t valueDTail = value_d - (valueDLoops - 1) * dSplitSize;
    uint32_t gmBaseBlockOffset = 0;
    
    if (DTEMPLATETYPE == DTemplateType::Aligned192) {
        ///////////////////////////////////////////////////////////////
        // Matmal4 dq
        ///////////////////////////////////////////////////////////////
        // left [B, N2, G, S1, s2] right [B, N2, 1, S2, D] output [B, N2, G, S1, D]
        mm4.SetOrgShape(s1, dbParam.s2Stride, s2_size, dbParam.actualS2Len, dqKc);
        for (int32_t sliceIdx = 0; sliceIdx < s1SliceNum; ++sliceIdx) {
            // slice s1
            int32_t s1SliceExtend = sliceIdx == (s1SliceNum - 1) ? s1SliceTail : s1SliceSize;
            uint32_t mInnerSplitSize = MMAD_BASE_SIZE;
            uint32_t mInnerLoops = (s1SliceExtend + mSplitSize - 1) / mSplitSize;
            uint32_t mInnerTail = s1SliceExtend - (mInnerLoops - 1) * mSplitSize;
            for (uint32_t m = 0; m < mInnerLoops; m++) {
                subMSizeAct = m == (mInnerLoops - 1) ? mInnerTail : mSplitSize;
                subMSizeActAlign = CeilDiv(subMSizeAct, C0_SIZE) * C0_SIZE;
                for (uint32_t dLoopIdx = 0; dLoopIdx < dLoops; dLoopIdx++) {
                    subDSizeAct = dLoopIdx == (dLoops - 1) ? dTail : dSplitSize;
                    subDSizeActAlign = CeilDiv(subDSizeAct, C0_SIZE) * C0_SIZE;
                    for (uint32_t n = 0; n < nLoops; n++) {
                        bool atomicAddFlag = n != 0;
                        subNSizeAct = n == (nLoops - 1) ? nTail : nSplitSize;
                        subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;
                        uint64_t aL1Offset = m * MMAD_BASE_SIZE * S_BASE_SIZE + n * MMAD_BASE_SIZE * MMAD_BASE_SIZE;
                        uint64_t bL1Offset = dLoopIdx * MMAD_BASE_SIZE * S_BASE_SIZE + n * MMAD_BASE_SIZE * MMAD_BASE_SIZE;
                        // B矩阵
                        if (sliceIdx == 0 && m == 0) {
                            CopyGmToL1(kL1Tensor[bL1Offset],
                                keyGm[dbParam.bTensorOffsetCv + n * nSplitSize * dbParam.s2Stride +
                                    dLoopIdx * dSplitSize], subNSizeAct, subDSizeAct, dbParam.s2Stride);
                        }
                        // A矩阵
                        if (dLoopIdx == 0) {
                            uint16_t blockCount = subNSizeActAlign / C0_SIZE;
                            uint16_t blockLen = subMSizeActAlign * C0_SIZE * sizeof(T1) / BLOCK_SIZE;
                            uint16_t srcStride = (s1_size - subMSizeActAlign) * C0_SIZE * sizeof(T1) / BLOCK_SIZE;
                            uint16_t dstStride = 0;
                            AscendC::DataCopyParams copyParams = {blockCount, blockLen, srcStride, dstStride};
                            DataCopy(dsL1Tensor[aL1Offset],
                                mulWorkSpaceGm[pingpongIdx * cubeBaseMN * 2 +
                                    (m * mSplitSize + s1SliceSize * sliceIdx) * C0_SIZE +
                                    n * nSplitSize * dbParam.s1CvExtendAlign * DTYPE_FACTOR],
                                copyParams);
                        }
                        AscendC::SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                        AscendC::WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                        mm4.SetTail(subMSizeAct, subDSizeAct, subNSizeAct);
                        mm4.SetTensorB(kL1Tensor[bL1Offset]);
                        mm4.SetTensorA(dsL1Tensor[aL1Offset]);
                        mm4.template Iterate<false>(atomicAddFlag);
                    }
                    if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                        gmBaseBlockOffset = sliceIdx * s1SliceSize * C0_SIZE +
                            m * mSplitSize * C0_SIZE + dLoopIdx * dSplitSize * s1;
                    } else {
                        gmBaseBlockOffset = (sliceIdx * s1SliceSize+ m * mSplitSize) * dbParam.s1Stride +
                            dLoopIdx * dSplitSize;
                    }
                    mm4.GetTensorC(dqWorkSpaceGm[dqOffset + gmBaseBlockOffset], 1);
                }
            }
            AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
            AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        }
        mm4.End();
        ///////////////////////////////////////////////////////////////
        // Matmal4 dk
        ///////////////////////////////////////////////////////////////
        // left [B, N2, G, S1, S2] right [B, N2, 1, S1, D] output [B, N2, G, S2, D]
        mm3.SetOrgShape(s2, dbParam.s1Stride, s1_size, dbParam.actualS1Len, dkvKc);
        for (int32_t sliceIdx = 0; sliceIdx < s2SliceNum; ++sliceIdx) {
            int32_t s2SliceExtend = sliceIdx == (s2SliceNum - 1) ? s2SliceTail : s2SliceSize;
            // m是s1方向的循环
            uint32_t nInnerLoops = (s2SliceExtend + nSplitSize - 1) / nSplitSize;
            uint32_t nInnerTail = s2SliceExtend - (nInnerLoops - 1) * nSplitSize;
            for (uint32_t n = 0; n < nInnerLoops; n++) {
                subNSizeAct = n == (nInnerLoops - 1) ? nInnerTail : nSplitSize;
                subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;
                for (uint32_t dLoopIdx = 0; dLoopIdx < dLoops; dLoopIdx++) {
                    subDSizeAct = dLoopIdx == (dLoops - 1) ? dTail : dSplitSize;
                    subDSizeActAlign = CeilDiv(subDSizeAct, C0_SIZE) * C0_SIZE;
                    for (uint32_t m = 0; m < mLoops; m++) {
                        subMSizeAct = m == (mLoops - 1) ? mTail : mSplitSize;
                        subMSizeActAlign = CeilDiv(subMSizeAct, C0_SIZE) * C0_SIZE;
                        uint64_t aL1Offset = S_BASE_SIZE * n * MMAD_BASE_SIZE + m * MMAD_BASE_SIZE * MMAD_BASE_SIZE;
                        uint64_t bL1Offset = dLoopIdx * MMAD_BASE_SIZE * S_BASE_SIZE + m * MMAD_BASE_SIZE * MMAD_BASE_SIZE;
                        bool atomicAddFlag = !(m == 0);
                        // A矩阵
                        if (dLoopIdx == 0) {
                            uint16_t blockCount = subNSizeActAlign / C0_SIZE;
                            uint16_t blockLen = subMSizeActAlign * C0_SIZE * sizeof(T1) / BLOCK_SIZE;
                            uint16_t srcStride = (s1_size - subMSizeActAlign) * C0_SIZE * sizeof(T1) / BLOCK_SIZE;
                            uint16_t dstStride = 0;
                            AscendC::DataCopyParams copyParams = {blockCount, blockLen, srcStride, dstStride};
                            DataCopy(dsL1Tensor[aL1Offset],
                                mulWorkSpaceGm[pingpongIdx * cubeBaseMN * 2 +
                                    (sliceIdx * s2SliceSize + n * nSplitSize) * dbParam.s1CvExtendAlign * DTYPE_FACTOR +
                                    m * mSplitSize * C0_SIZE],
                                copyParams);
                        }
                        // B矩阵
                        if (sliceIdx == 0 && n == 0) {
                            CopyGmToL1(kL1Tensor[bL1Offset],
                                queryGm[dbParam.aTensorOffsetCv + m * mSplitSize * dbParam.s1Stride +
                                    dLoopIdx * dSplitSize], subMSizeAct, subDSizeAct, dbParam.s1Stride);
                        }
                        AscendC::SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                        AscendC::WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                        mm3.SetTail(subNSizeAct, subDSizeAct, subMSizeAct);
                        mm3.SetTensorA(dsL1Tensor[aL1Offset], true);
                        mm3.SetTensorB(kL1Tensor[bL1Offset]);
                        mm3.template Iterate<false>(atomicAddFlag);
                    }
                    if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                        gmBaseBlockOffset = sliceIdx * s2SliceSize * C0_SIZE + n * nSplitSize * C0_SIZE +
                            dLoopIdx * dSplitSize * s2;
                    } else {
                        gmBaseBlockOffset = sliceIdx * s2SliceSize * dbParam.s2Stride +
                            n * nSplitSize * dbParam.s2Stride + dLoopIdx * dSplitSize;
                    }
                    mm3.GetTensorC(dkWorkSpaceGm[dkvOffset + gmBaseBlockOffset], 1);
                }
            }
            AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
            AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        }
        mm3.End();

        ///////////////////////////////////////////////////////////////
        // Matmal5 dv
        ///////////////////////////////////////////////////////////////
        // left [B, N2, G, S1, S2] right [B, N2, G, S1, D2] output [B, N2, 1, S2, D2]
        mm3.SetOrgShape(s2, dbParam.s1Stride / d * value_d, s1_size, dbParam.actualS1Len, dvKc);
        for (int32_t sliceIdx = 0; sliceIdx < s2SliceNum; ++sliceIdx) {
            int32_t s2SliceExtend = sliceIdx == (s2SliceNum - 1) ? s2SliceTail : s2SliceSize;
            // m是s1方向的循环
            uint32_t nInnerSplitSize = MMAD_BASE_SIZE;
            uint32_t nInnerLoops = (s2SliceExtend + nSplitSize - 1) / nSplitSize;
            uint32_t nInnerTail = s2SliceExtend - (nInnerLoops - 1) * nSplitSize;
            for (uint32_t n = 0; n < nInnerLoops; n++) {
                subNSizeAct = n == (nInnerLoops - 1) ? nInnerTail : nSplitSize;
                subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;
                for (uint32_t dLoopIdx = 0; dLoopIdx < valueDLoops; dLoopIdx++) {
                    subDSizeAct = dLoopIdx == (valueDLoops - 1) ? valueDTail : dSplitSize;
                    subDSizeActAlign = CeilDiv(subDSizeAct, C0_SIZE) * C0_SIZE;
                    for (uint32_t m = 0; m < mLoops; m++) {
                        bool atomicAddFlag = !(m == 0);
                        subMSizeAct = m == (mLoops - 1) ? mTail : mSplitSize;
                        subMSizeActAlign = CeilDiv(subMSizeAct, C0_SIZE) * C0_SIZE;
                        uint64_t aL1Offset = S_BASE_SIZE * n * MMAD_BASE_SIZE + m * MMAD_BASE_SIZE * MMAD_BASE_SIZE;
                        uint64_t bL1Offset = dLoopIdx * MMAD_BASE_SIZE * S_BASE_SIZE + m * MMAD_BASE_SIZE * MMAD_BASE_SIZE;
                        // A矩阵
                        if (dLoopIdx == 0) {
                            uint16_t blockCount = subNSizeActAlign / C0_SIZE;
                            uint16_t blockLen = subMSizeActAlign * C0_SIZE * sizeof(T1) / BLOCK_SIZE;
                            uint16_t srcStride = (s1_size - subMSizeActAlign) * C0_SIZE * sizeof(T1) / BLOCK_SIZE;
                            uint16_t dstStride = 0;
                            AscendC::DataCopyParams copyParams = {blockCount, blockLen, srcStride, dstStride};
                            DataCopy(dsL1Tensor[aL1Offset],
                                dropWorkSpaceGm[pingpongIdx * cubeBaseMN * 2 +
                                    (sliceIdx * s2SliceSize + n * nInnerSplitSize) * dbParam.s1CvExtendAlign * 2 +
                                    m * mSplitSize * C0_SIZE],
                                copyParams);
                        }
                        // B矩阵
                        if (sliceIdx == 0 && n == 0) {
                            CopyGmToL1(kL1Tensor[bL1Offset],
                                dxGm[dbParam.aTensorOffsetCv / d * value_d + m * mSplitSize * dbParam.s1Stride / d * value_d +
                                    dLoopIdx * dSplitSize], subMSizeAct, subDSizeAct, dbParam.s1Stride / d * value_d );
                        }
                        AscendC::SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                        AscendC::WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                        mm3.SetTail(subNSizeAct, subDSizeAct, subMSizeAct);
                        mm3.SetTensorA(dsL1Tensor[aL1Offset], true);
                        mm3.SetTensorB(kL1Tensor[bL1Offset]);
                        mm3.template Iterate<false>(atomicAddFlag);
                    }
                    if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                        gmBaseBlockOffset = sliceIdx * s2SliceSize * C0_SIZE + n * nSplitSize * C0_SIZE +
                            dLoopIdx * dSplitSize * s2;
                    } else {
                        gmBaseBlockOffset = (sliceIdx * s2SliceSize + n * nSplitSize) * dbParam.s2Stride / d * value_d +
                            dLoopIdx * dSplitSize;
                    }
                    mm3.GetTensorC(dvWorkSpaceGm[dvOffset + gmBaseBlockOffset], 1);
                }
            }
            AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
            AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        }
        mm3.End();
    } else if (DTEMPLATETYPE == DTemplateType::Aligned64) {
        ///////////////////////////////////////////////////////////////
        // Matmal4 dq
        ///////////////////////////////////////////////////////////////
        // left [B, N2, G, S1, S2] right [B, N2, 1, S1, D] output [B, N2, G, S2, D]
        mm4.SetOrgShape(s1, dbParam.s2Stride, s2_size, dbParam.actualS2Len, dqKc);
        for (uint32_t m = 0; m < mLoops; m++) {
            subMSizeAct = m == (mLoops - 1) ? mTail : mSplitSize;
            subMSizeActAlign = CeilDiv(subMSizeAct, C0_SIZE) * C0_SIZE;
            for (uint32_t n = 0; n < nLoops; n++) {
                bool atomicAddFlag = !(n == 0);
                subNSizeAct = n == (nLoops - 1) ? nTail : nSplitSize;
                subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;
                uint64_t bL1Offset = n * MMAD_BASE_SIZE * DIM_64;
                uint64_t aL1Offset = 0;
                // B矩阵
                if (m == 0) {
                    CopyGmToL1(kL1Tensor[bL1Offset], keyGm[dbParam.bTensorOffsetCv +
                        n * nSplitSize * dbParam.s2Stride], subNSizeAct, d, dbParam.s2Stride);
                }
                // A矩阵
                uint16_t blockCount = subNSizeActAlign / C0_SIZE;
                uint16_t blockLen = subMSizeActAlign * C0_SIZE * sizeof(T1) / BLOCK_SIZE;
                uint16_t srcStride = (s1_size - subMSizeActAlign) * C0_SIZE * sizeof(T1) / BLOCK_SIZE;
                uint16_t dstStride = 0;
                AscendC::DataCopyParams copyParams = {blockCount, blockLen, srcStride, dstStride};
                if (m * nLoops + n + 1 <= L1_CACHE_CAPACITY_LIMIT) {
                    aL1Offset = m * MMAD_BASE_SIZE * MMAD_BASE_SIZE * nLoops + MMAD_BASE_SIZE * n * MMAD_BASE_SIZE;
                    DataCopy(dsL1Tensor[aL1Offset],
                            mulWorkSpaceGm[pingpongIdx * cubeBaseMN * DTYPE_FACTOR + m * mSplitSize * C0_SIZE +
                                n * nSplitSize * dbParam.s1CvExtendAlign * DTYPE_FACTOR], copyParams);
                    mm4.SetTensorA(dsL1Tensor[aL1Offset]);
                } else if (m * nLoops + n + 1 > L1_CACHE_CAPACITY_LIMIT) {
                    aL1Offset = (m - 3) * MMAD_BASE_SIZE * MMAD_BASE_SIZE * nLoops +
                        (n - 2) * MMAD_BASE_SIZE * MMAD_BASE_SIZE;
                    DataCopy(dsL1Tensor[aL1Offset],
                        mulWorkSpaceGm[pingpongIdx * cubeBaseMN * DTYPE_FACTOR + m * mSplitSize * C0_SIZE +
                            n * nSplitSize * dbParam.s1CvExtendAlign * DTYPE_FACTOR], copyParams);
                    mm4.SetTensorA(dsL1Tensor[aL1Offset]);
                }
                AscendC::SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                AscendC::WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                mm4.SetTail(subMSizeAct, d, subNSizeAct);
                mm4.SetTensorB(kL1Tensor[bL1Offset]);
                mm4.template Iterate<false>(atomicAddFlag);
                if (m * nLoops + n + 1 == L1_CACHE_CAPACITY_LIMIT) {
                    AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
                    AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
                }
            }
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                gmBaseBlockOffset = m * mSplitSize * C0_SIZE;
            } else {
                gmBaseBlockOffset = m * mSplitSize * dbParam.s1Stride;
            }
            mm4.template GetTensorC<false>(dqWorkSpaceGm[dqOffset + gmBaseBlockOffset], 1); //加上累加
        }
        AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        mm4.End();
        ///////////////////////////////////////////////////////////////
        // Matmal4 dk
        ///////////////////////////////////////////////////////////////
        // left [B, N2, G, S1, S2] right [B, N2, 1, S1, D] output [B, N2, G, S2, D]
        mm3.SetOrgShape(s2, dbParam.s1Stride, s1_size, dbParam.actualS1Len, dkvKc);
        for (uint32_t n = 0; n < nLoops; n++) { // s2方向
            subNSizeAct = n == (nLoops - 1) ? nTail : nSplitSize;
            subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;
            for (uint32_t m = 0; m < mLoops; m++) { // s1方向
                bool atomicAddFlag = (m != 0);
                subMSizeAct = m == (mLoops - 1) ? mTail : mSplitSize;
                subMSizeActAlign = CeilDiv(subMSizeAct, C0_SIZE) * C0_SIZE;
                uint64_t bL1Offset = m * MMAD_BASE_SIZE * DIM_64;
                uint64_t aL1Offset = m * MMAD_BASE_SIZE * MMAD_BASE_SIZE * nLoops + MMAD_BASE_SIZE * n * MMAD_BASE_SIZE;
                // B矩阵
                if (n == 0) {
                    CopyGmToL1(kL1Tensor[bL1Offset], queryGm[dbParam.aTensorOffsetCv +
                        m * mSplitSize * dbParam.s1Stride], subMSizeAct, d, dbParam.s1Stride);
                }
                // A矩阵
                uint16_t blockCount = subNSizeActAlign / C0_SIZE;
                uint16_t blockLen = subMSizeActAlign * C0_SIZE * sizeof(T1) / BLOCK_SIZE;
                uint16_t srcStride = (s1_size - subMSizeActAlign) * C0_SIZE * sizeof(T1) / BLOCK_SIZE;
                uint16_t dstStride = 0;
                AscendC::DataCopyParams copyParams = {blockCount, blockLen, srcStride, dstStride};
                if ((m * nLoops + n + 1) <= 2) {
                    DataCopy(dsL1Tensor[aL1Offset],
                        mulWorkSpaceGm[pingpongIdx * cubeBaseMN * DTYPE_FACTOR + m * mSplitSize * C0_SIZE +
                            n * nSplitSize * dbParam.s1CvExtendAlign * DTYPE_FACTOR],
                        copyParams);
                    mm3.SetTensorA(dsL1Tensor[aL1Offset], true);
                } else if ((m * nLoops + n + 1) > L1_CACHE_CAPACITY_LIMIT) {
                    aL1Offset = (m - 3) * MMAD_BASE_SIZE * MMAD_BASE_SIZE * nLoops +
                        MMAD_BASE_SIZE * (n - 2) * MMAD_BASE_SIZE;
                    DataCopy(dsL1Tensor[aL1Offset],
                        mulWorkSpaceGm[pingpongIdx * cubeBaseMN * DTYPE_FACTOR + m * mSplitSize * C0_SIZE +
                            n * nSplitSize * dbParam.s1CvExtendAlign * DTYPE_FACTOR],
                        copyParams);
                    mm3.SetTensorA(dsL1Tensor[aL1Offset], true);
                } else if ((m * nLoops + n + 1 <= L1_CACHE_CAPACITY_LIMIT)) {
                    mm3.SetTensorA(dsL1Tensor[aL1Offset], true);
                }
                AscendC::SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                AscendC::WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                mm3.SetTail(subNSizeAct, d, subMSizeAct);
                mm3.SetTensorB(kL1Tensor[bL1Offset]);
                mm3.template Iterate<false>(atomicAddFlag);
                if (m * nLoops + n + 1 == L1_CACHE_CAPACITY_LIMIT) {
                    AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
                    AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
                }
            }
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                gmBaseBlockOffset = n * nSplitSize * C0_SIZE;
            } else {
                gmBaseBlockOffset = n * nSplitSize * dbParam.s2Stride;
            }
            mm3.GetTensorC(dkWorkSpaceGm[dkvOffset + gmBaseBlockOffset], 1);
        }
        AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        mm3.End();
        ///////////////////////////////////////////////////////////////
        // Matmal5 dv
        ///////////////////////////////////////////////////////////////
        // left [B, N2, G, S1, S2] right [B, N2, G, S1, D2] output [B, N2, 1, S2, D2]
        mm3.SetOrgShape(s2, dbParam.s1Stride / d * value_d, s1_size, dbParam.actualS1Len, dvKc);
        for (uint32_t n = 0; n < nLoops; n++) {
            subNSizeAct = n == (nLoops - 1) ? nTail : nSplitSize;
            subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;
            for (uint32_t m = 0; m < mLoops; m++) {
                bool atomicAddFlag = !(m == 0);
                subMSizeAct = m == (mLoops - 1) ? mTail : mSplitSize;
                subMSizeActAlign = CeilDiv(subMSizeAct, C0_SIZE) * C0_SIZE;
                uint64_t bL1Offset = m * MMAD_BASE_SIZE * DIM_64;
                uint64_t aL1Offset = m * MMAD_BASE_SIZE * MMAD_BASE_SIZE * nLoops + MMAD_BASE_SIZE * n * MMAD_BASE_SIZE;
                // B矩阵
                if (n == 0) {
                    // 实际意义为dx
                    CopyGmToL1(vL1Tensor[bL1Offset],
                        dxGm[dbParam.aTensorOffsetCv / d * value_d + m * mSplitSize * dbParam.s1Stride / d * value_d],
                        subMSizeAct, value_d, dbParam.s1Stride / d * value_d);
                }
                // A矩阵
                uint16_t blockCount = subNSizeActAlign / C0_SIZE;
                uint16_t blockLen = subMSizeActAlign * C0_SIZE * sizeof(T1) / BLOCK_SIZE;
                uint16_t srcStride = (s1_size - subMSizeActAlign) * C0_SIZE * sizeof(T1) / BLOCK_SIZE;
                uint16_t dstStride = 0;
                AscendC::DataCopyParams copyParams = {blockCount, blockLen, srcStride, dstStride};
                if ((m * nLoops + n + 1) <= L1_CACHE_CAPACITY_LIMIT) {
                    DataCopy(dsL1Tensor[aL1Offset],
                        dropWorkSpaceGm[pingpongIdx * cubeBaseMN * DTYPE_FACTOR + m * mSplitSize * C0_SIZE +
                            n * nSplitSize * dbParam.s1CvExtendAlign * DTYPE_FACTOR],
                        copyParams);
                    mm3.SetTensorA(dsL1Tensor[aL1Offset], true);
                } else if ((m * nLoops + n + 1) > L1_CACHE_CAPACITY_LIMIT) {
                    aL1Offset = (m - 3) * MMAD_BASE_SIZE * MMAD_BASE_SIZE * nLoops + MMAD_BASE_SIZE * (n - 2) * MMAD_BASE_SIZE;
                    DataCopy(dsL1Tensor[aL1Offset],
                        dropWorkSpaceGm[pingpongIdx * cubeBaseMN * DTYPE_FACTOR + m * mSplitSize * C0_SIZE +
                            n * nSplitSize * dbParam.s1CvExtendAlign * DTYPE_FACTOR],
                        copyParams);
                    mm3.SetTensorA(dsL1Tensor[aL1Offset], true);
                }
                AscendC::SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                AscendC::WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                mm3.SetTail(subNSizeAct, value_d, subMSizeAct);
                mm3.SetTensorB(vL1Tensor[bL1Offset]);
                mm3.template Iterate<false>(atomicAddFlag);
                if (m * nLoops + n + 1 == L1_CACHE_CAPACITY_LIMIT) {
                    AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
                    AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
                }
            }
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                gmBaseBlockOffset = n * nSplitSize * C0_SIZE;
            } else {
                gmBaseBlockOffset = n * nSplitSize * dbParam.s2Stride / d * value_d;
            }
            mm3.GetTensorC(dvWorkSpaceGm[dvOffset + gmBaseBlockOffset], 1);
        }
        AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        mm3.End();
    } else {
        ///////////////////////////////////////////////////////////////
        // Matmal4 dq
        ///////////////////////////////////////////////////////////////
        // left [B, N2, G, S1, s2] right [B, N2, 1, S2, D] output [B, N2, G, S1, D]
        if (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm4.SetUserDefInfo(dbParam.actualS1Len);
        }
        mm4.SetOrgShape(s1_size, dbParam.s2Stride, s2_size, dbParam.actualS2Len, dqKc);
        mm4.SetTail(dbParam.s1CvExtend, d, dbParam.s2CvExtend); // M N K
        mm4.SetTensorA(mulWorkSpaceGm[pingpongIdx * cubeBaseMN * 2]);
        mm4.SetTensorB(keyGm[dbParam.bTensorOffsetCv]);
        mm4.template IterateAll<false>(dqWorkSpaceGm[dqOffset], 1);
        mm4.End();

        if constexpr (HAS_ROPE == ENABLE) {
            ///////////////////////////////////////////////////////////////
            // Matmal4 dq rope
            ///////////////////////////////////////////////////////////////
            // left [B, N2, G, S1, s2] right [B, N2, 1, S2, D] output [B, N2, G, S1, D]
            if (MM2_OUT_FORMAT == CubeFormat::NZ) {
                mm4.SetUserDefInfo(dbParam.actualS1Len);
            }
            mm4.SetOrgShape(s1_size, dbParam.s2Stride / d * rope_d, s2_size, dbParam.actualS2Len, dqKc / d * rope_d);
            mm4.SetTail(dbParam.s1CvExtend, rope_d, dbParam.s2CvExtend); // M N K
            mm4.SetTensorA(mulWorkSpaceGm[pingpongIdx * cubeBaseMN * 2]);
            mm4.SetTensorB(keyRopeGm[dbParam.bTensorOffsetCv_rope]);
            mm4.template IterateAll<false>(dqRopeWorkSpaceGm[dqRopeOffset], 1);
            mm4.End();
        }

        ///////////////////////////////////////////////////////////////
        // Matmal4 dk
        ///////////////////////////////////////////////////////////////
        // left [B, N2, G, S1, S2] right [B, N2, 1, S1, D] output [B, N2, G, S2, D]

        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm3.SetUserDefInfo(dbParam.actualS2Len);
        }
        mm3.SetOrgShape(s2_size, dbParam.s1Stride, s1_size, dbParam.actualS1Len, dkvKc);
        mm3.SetTail(dbParam.s2CvExtend, d, dbParam.s1CvExtend);
        mm3.SetTensorA(mulWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], true);
        mm3.SetTensorB(queryGm[dbParam.aTensorOffsetCv]);
        mm3.template IterateAll<false>(dkWorkSpaceGm[dkvOffset], 1);
        mm3.End();
    
        if constexpr (HAS_ROPE == ENABLE) {
            ///////////////////////////////////////////////////////////////
            // Matmal4 dk rope
            ///////////////////////////////////////////////////////////////
            // left [B, N2, G, S1, S2] right [B, N2, 1, S1, D] output [B, N2, G, S2, D]

            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                mm3.SetUserDefInfo(dbParam.actualS2Len);
            }
            mm3.SetOrgShape(s2_size, dbParam.s1Stride / d * rope_d, s1_size, dbParam.actualS1Len, dkvKc / d * rope_d);
            mm3.SetTail(dbParam.s2CvExtend, rope_d, dbParam.s1CvExtend);
            mm3.SetTensorA(mulWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], true);
            mm3.SetTensorB(queryRopeGm[dbParam.aTensorOffsetCv_rope]);
            mm3.template IterateAll<false>(dkRopeWorkSpaceGm[dkRopeOffset], 1);
            mm3.End();
        }

        ///////////////////////////////////////////////////////////////
        // Matmal5 dv
        ///////////////////////////////////////////////////////////////
        // left [B, N2, G, S1, S2] right [B, N2, G, S1, D2] output [B, N2, 1, S2, D2]

        mm3.SetOrgShape(s2_size, dbParam.s1Stride / d * value_d, s1_size, dbParam.actualS1Len, dvKc);
        mm3.SetTail(dbParam.s2CvExtend, value_d, dbParam.s1CvExtend);
        mm3.SetTensorA(dropWorkSpaceGm[pingpongIdx * cubeBaseMN * 2], true);
        mm3.SetTensorB(dxGm[dbParam.aTensorOffsetCv / d * value_d]);
        if constexpr (IsSameType<T1, float>::value) {
            if (nextBlockId == -1) {
                mm3.template IterateAll<true>(dvGm[dvOffset], 1);
            } else {
                mm3.template IterateAll<false>(dvGm[dvOffset], 1);
            }
        } else {
            if (nextBlockId == -1) {
                mm3.template IterateAll<true>(dvWorkSpaceGm[dvOffset], 1);
            } else {
                mm3.template IterateAll<false>(dvWorkSpaceGm[dvOffset], 1);
            }
        }
        mm3.End();
    }
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::ReleaseTSCMAllocEventID()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_M>(eventIdMte1AToM);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_M>(eventIdMte1BToM);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventIdMToMte1_ID3);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventIdMToMte1_ID4);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventIdMToMte1_ID5);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventIdMToMte1_ID6);
    GetTPipePtr()->ReleaseEventID<HardEvent::FIX_M>(eventIdFixpipeToM_ID0);
    GetTPipePtr()->ReleaseEventID<HardEvent::FIX_M>(eventIdFixpipeToM_ID1);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_FIX>(eventIdMToFixpipe_ID0);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_FIX>(eventIdMToFixpipe_ID1);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1_ID0);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1_ID1);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1_ID2);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1_ID3);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2_ID0);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2_ID1);
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::ComputeL1CustomMMDqkv(
    DBParams& dbParam, int64_t nextBlockId)
{
    pingpongIdx = dbParam.taskId % 2;
    int64_t dqOffset = 0;
    int64_t dqRopeOffset = 0;
    int64_t dkvOffset = 0;
    int64_t dkRopeOffset = 0;
    int64_t dvOffset = 0;
    if constexpr (MM2_OUT_FORMAT == CubeFormat::ND) {
        dqOffset = dbParam.aTensorOffsetCv;
        dkvOffset = dbParam.bTensorOffsetCv;
        // 在mla场景，d指qk的d，value_d指v的d，其余场景value_d一般和d相等
        dvOffset = dbParam.bTensorOffsetCv / d * value_d;
        dqRopeOffset = dbParam.aTensorOffsetCv / d * rope_d;
        dkRopeOffset = dbParam.bTensorOffsetCv / d * rope_d;
    } else if constexpr (INPUT_LAYOUT == TND) {  // TND Nz
        UpdateToken(dbParam.bIdx);
        if (dbParam.bIdx > 0) {
            dqOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[dbParam.bIdx - 1] * n2 * g * dAlign;
            dkvOffset = ((__gm__ int64_t *)actual_seq_kvlen_addr)[dbParam.bIdx - 1] * n2 * dAlign;
            dqRopeOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[dbParam.bIdx - 1] * n2 * g * rope_dAlign;
            dkRopeOffset = ((__gm__ int64_t *)actual_seq_kvlen_addr)[dbParam.bIdx - 1] * n2 * rope_dAlign;
            dvOffset = ((__gm__ int64_t *)actual_seq_kvlen_addr)[dbParam.bIdx - 1] * n2 * value_dAlign;
        }
        dqOffset += ((dbParam.n2Idx * g + dbParam.gIdx) * dbParam.actualS1Len) * dAlign + dbParam.s1oIdx * s1CvInner * C0_SIZE;
        dkvOffset += (dbParam.n2Idx * dbParam.actualS2Len) * dAlign + dbParam.s2oIdx * s2CvInner * C0_SIZE;
        dqRopeOffset += ((dbParam.n2Idx * g + dbParam.gIdx) * dbParam.actualS1Len) * rope_dAlign + dbParam.s1oIdx * s1CvInner * C0_SIZE;
        dkRopeOffset += (dbParam.n2Idx * dbParam.actualS2Len) * rope_dAlign + dbParam.s2oIdx * s2CvInner * C0_SIZE;
        dvOffset += (dbParam.n2Idx * dbParam.actualS2Len) * value_dAlign + dbParam.s2oIdx * s2CvInner * C0_SIZE;
    } else {  // Other Nz
        dqOffset = (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1) * dAlign + dbParam.s1oIdx * s1CvInner * C0_SIZE;
        dkvOffset = ((dbParam.bIdx * n2 + dbParam.n2Idx) * s2) * dAlign + dbParam.s2oIdx * s2CvInner * C0_SIZE;
        dqRopeOffset = (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1) * rope_dAlign + dbParam.s1oIdx * s1CvInner * C0_SIZE;
        dkRopeOffset = ((dbParam.bIdx * n2 + dbParam.n2Idx) * s2) * rope_dAlign + dbParam.s2oIdx * s2CvInner * C0_SIZE;
        dvOffset = ((dbParam.bIdx * n2 + dbParam.n2Idx) * s2) * value_dAlign + dbParam.s2oIdx * s2CvInner * C0_SIZE; //注意最后bngsd和sbngd的偏移计算
    }

    uint64_t dqKc = dbParam.s1Stride;
    uint64_t dkvKc = dbParam.s2Stride;
    uint64_t dvKc = dbParam.s2Stride / d * value_d;
    uint64_t s1_size = dbParam.actualS1Len;
    uint64_t s2_size = dbParam.s2CvExtendAlign * DTYPE_FACTOR;
    uint64_t dvOutMStride = dbParam.actualS2Len;
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        s1_size = dbParam.s1CvExtendAlign * DTYPE_FACTOR;
        s2_size = dbParam.s2CvExtendAlign;
    }
    if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
        dqKc = d;
        dkvKc = d;
        dvKc = value_d;
    }

    constexpr int32_t s2SliceSize = S_SPLITT_SIZE;
    int32_t s2SliceNum = (dbParam.s2CvExtend + s2SliceSize - 1) / s2SliceSize;
    int32_t s2SliceTail = dbParam.s2CvExtend - (s2SliceNum - 1) * s2SliceSize;
    event_t eventIdMte1ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
    event_t eventIdMte2ToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
    uint32_t gmBaseBlockOffset = 0;
    // m是s1方向的循环
    uint32_t mSplitSize = MMAD_BASE_SIZE;
    uint32_t mLoops = (dbParam.s1CvExtend + mSplitSize - 1) / mSplitSize;
    uint32_t mTail = dbParam.s1CvExtend - (mLoops - 1) * mSplitSize;
    uint32_t subMSizeAct = mSplitSize;
    uint32_t subMSizeActAlign = CeilDiv(subMSizeAct, MMAD_BASE_SIZE) * MMAD_BASE_SIZE;

    // n 是s2方向的循环
    uint32_t nSplitSize = MMAD_BASE_SIZE;
    uint32_t nLoops = (dbParam.s2CvExtend + nSplitSize - 1) / nSplitSize;
    uint32_t nTail = dbParam.s2CvExtend - (nLoops - 1) * nSplitSize;
    uint32_t subNSizeAct = nSplitSize;
    uint32_t subNSizeActAlign = CeilDiv(subNSizeAct, MMAD_BASE_SIZE) * MMAD_BASE_SIZE;
    event_t eventIdMToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_MTE1));
    AscendC::SetFlag<HardEvent::FIX_M>(eventIdFixpipeToM_ID0);
    AscendC::SetFlag<HardEvent::FIX_M>(eventIdFixpipeToM_ID1);
    AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2_ID0);
    AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2_ID1);
    AscendC::SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_ID3);
    AscendC::SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_ID4);
    AscendC::SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_ID5);
    AscendC::SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_ID6);

    for (int32_t sliceIdx = 0; sliceIdx < s2SliceNum; ++sliceIdx) {
        // slice s2
        int32_t s2SliceExtend = sliceIdx == (s2SliceNum - 1) ? s2SliceTail : s2SliceSize;
        int32_t s2SliceExtendAlign = CeilDiv(s2SliceExtend, MMAD_BASE_SIZE) * MMAD_BASE_SIZE;
        int64_t dSOffsetBias = sliceIdx * s2SliceSize;
        // n 是s2方向的循环
        uint32_t nInnerLoops = (s2SliceExtend + nSplitSize - 1) / nSplitSize;
        uint32_t nInnerTail = s2SliceExtend - (nInnerLoops - 1) * nSplitSize;
        if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
            dSOffsetBias *= dbParam.s1CvExtendAlign * DTYPE_FACTOR;
        }
        int64_t bTensorOffsetBias = sliceIdx * s2SliceSize * dbParam.s2Stride;
        int64_t bTensorOffsetBias_d3 = sliceIdx * s2SliceSize * dbParam.s2Stride / d * rope_d;
        int64_t dkOffsetBias = bTensorOffsetBias;
        int64_t dkOffsetBias_d3 = bTensorOffsetBias_d3;
        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            dkOffsetBias = sliceIdx * s2SliceSize * C0_SIZE;
        }
        ///////////////////////////////////////////////////////////////
        // Matmal4 dq
        ///////////////////////////////////////////////////////////////
        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(ping_pong_flag_l1_a_);
        for (uint32_t m = 0; m < mLoops; m++) {
            subMSizeAct = m == (mLoops - 1) ? mTail : mSplitSize;
            subMSizeActAlign = CeilDiv(subMSizeAct, C0_SIZE) * C0_SIZE;
            LocalTensor<float> *l0_c_tensor = ping_pong_flag_l0_c_ ? &l0_c_pong_tensor : &l0_c_ping_tensor;
            for (uint32_t n = 0; n < nInnerLoops; n++) {
                subNSizeAct = n == (nInnerLoops - 1) ? nInnerTail : nSplitSize;
                subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;
                uint64_t bL1Offset = n * MMAD_BASE_SIZE * MMAD_BASE_SIZE + sliceIdx * S_SPLITT_SIZE * MMAD_BASE_SIZE;
                uint64_t aL1Offset = m * MMAD_BASE_SIZE * S_SPLITT_SIZE + MMAD_BASE_SIZE * n * MMAD_BASE_SIZE;
                LocalTensor<T1> *l0_a_tensor = ping_pong_flag_l0_a_ ? &l0_a_pong_tensor : &l0_a_ping_tensor;
                LocalTensor<T1> *l0_b_tensor = ping_pong_flag_l0_b_ ? &l0_b_pong_tensor : &l0_b_ping_tensor;
                // load gm to L1A
                uint16_t blockCount = subNSizeActAlign / C0_SIZE; 
                uint16_t blockLen = subMSizeActAlign * C0_SIZE * sizeof(T1) / 32;
                uint16_t srcStride = (s1_size - subMSizeActAlign) * C0_SIZE * sizeof(T1) / 32;
                uint16_t dstStride = 0;
                AscendC::DataCopyParams copyParams = {blockCount, blockLen, srcStride, dstStride};
                DataCopy(dsL1Tensor[aL1Offset],
                    mulWorkSpaceGm[pingpongIdx * cubeBaseMN * DTYPE_FACTOR + dSOffsetBias + m * mSplitSize * C0_SIZE +
                        n * nSplitSize * dbParam.s1CvExtendAlign * DTYPE_FACTOR],
                    copyParams);
                AscendC::SetFlag<HardEvent::MTE2_MTE1>(ping_pong_flag_l0_a_);
                AscendC::WaitFlag<HardEvent::MTE2_MTE1>(ping_pong_flag_l0_a_);
                AscendC::WaitFlag<HardEvent::M_MTE1>(ping_pong_flag_l0_a_ + 3);
                // load L1A To L0A
                commonLoadData2dParamsNoTranspose.repeatTimes = subNSizeActAlign / C0_SIZE;
                commonLoadData2dParamsNoTranspose.srcStride = subMSizeActAlign / C0_SIZE;
                for (int32_t i = 0; i < subMSizeActAlign / C0_SIZE; i++) {
                    AscendC::LoadData((*l0_a_tensor)[i * subNSizeActAlign * C0_SIZE],
                                        dsL1Tensor[aL1Offset + i * SIZE_256],
                                        commonLoadData2dParamsNoTranspose);
                }
                AscendC::SetFlag<HardEvent::MTE1_M>(eventIdMte1AToM);
                AscendC::WaitFlag<HardEvent::MTE1_M>(eventIdMte1AToM);

                // load gm to L1B
                if (m == 0) {
                    CopyGmToL1(kL1Tensor[bL1Offset],
                        keyGm[dbParam.bTensorOffsetCv + bTensorOffsetBias + n * nSplitSize * dbParam.s2Stride],
                        subNSizeAct, d, dbParam.s2Stride);
                    AscendC::SetFlag<HardEvent::MTE2_MTE1>(ping_pong_flag_l0_b_ + 2);
                    AscendC::WaitFlag<HardEvent::MTE2_MTE1>(ping_pong_flag_l0_b_ + 2);
                }
                // // load L1B To L0b
                AscendC::WaitFlag<HardEvent::M_MTE1>(ping_pong_flag_l0_b_ + 3 + 2);
                commonLoadData2dParamsTranspose.repeatTimes = d / C0_SIZE;
                commonLoadData2dParamsTranspose.srcStride = subNSizeActAlign / C0_SIZE;
                for (int32_t i = 0; i < subNSizeActAlign / C0_SIZE; i++) {
                    AscendC::LoadData((*l0_b_tensor)[i * d * C0_SIZE],
                                        kL1Tensor[bL1Offset + i * SIZE_256],
                                        commonLoadData2dParamsTranspose);
                }
                AscendC::SetFlag<HardEvent::MTE1_M>(eventIdMte1BToM);
                AscendC::WaitFlag<HardEvent::MTE1_M>(eventIdMte1BToM);
                // matmul
                uint16_t m_modify = (subMSizeAct == 1) ? 2 : subMSizeAct;
                commonMadParams.m = m_modify;
                commonMadParams.n = d;
                commonMadParams.k = subNSizeAct;
                commonMadParams.unitFlag = n == nInnerLoops - 1 ? 3 : 2;
                commonMadParams.cmatrixInitVal = n == 0 ? true : false;

                if ( n==0 ) {
                    AscendC::WaitFlag<HardEvent::FIX_M>(ping_pong_flag_l0_c_);
                }
                if (n == nInnerLoops - 1) {
                    AscendC::PipeBarrier<PIPE_M>();
                }

                AscendC::Mmad(*l0_c_tensor, *l0_a_tensor, *l0_b_tensor, commonMadParams);
                AscendC::SetFlag<HardEvent::M_MTE1>(ping_pong_flag_l0_b_ + 3 + 2);
                AscendC::SetFlag<HardEvent::M_MTE1>(ping_pong_flag_l0_a_ + 3);
                ping_pong_flag_l0_a_ = 1 - ping_pong_flag_l0_a_;
                ping_pong_flag_l0_b_ = 1 - ping_pong_flag_l0_b_;
            }
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                gmBaseBlockOffset = m * mSplitSize * C0_SIZE;
                commonFixpipeParamsV220.mSize = subMSizeAct;
                commonFixpipeParamsV220.nSize = d;
                commonFixpipeParamsV220.srcStride = subMSizeActAlign;
                commonFixpipeParamsV220.dstStride = s1 * 2;
                commonFixpipeParamsV220.quantPre = QuantMode_t::NoQuant;
                commonFixpipeParamsV220.unitFlag = 3;
                AscendC::SetAtomicType<float>();
                AscendC::Fixpipe<float, float, AscendC::CFG_NZ>(dqWorkSpaceGm[dqOffset + gmBaseBlockOffset], *l0_c_tensor, commonFixpipeParamsV220);
                AscendC::SetAtomicNone();
            } else {
                gmBaseBlockOffset = m * mSplitSize * dbParam.s1Stride;
            }
            AscendC::SetFlag<HardEvent::FIX_M>(ping_pong_flag_l0_c_);
            ping_pong_flag_l0_c_ = 1 - ping_pong_flag_l0_c_;
        }
        AscendC::SetFlag<HardEvent::MTE1_MTE2>(ping_pong_flag_l1_a_);
        ping_pong_flag_l1_a_ = 1 - ping_pong_flag_l1_a_;

        AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        ///////////////////////////////////////////////////////////////
        // Matmal4 dk
        ///////////////////////////////////////////////////////////////
        for (uint32_t n_loop = 0; n_loop < nInnerLoops; n_loop++) {
            subNSizeAct = n_loop == (nInnerLoops - 1) ? nInnerTail : nSplitSize;
            subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;
            LocalTensor<float> *l0_c_tensor = ping_pong_flag_l0_c_ ? &l0_c_pong_tensor : &l0_c_ping_tensor;
            for (uint32_t m_loop = 0; m_loop < mLoops; m_loop++) {
                subMSizeAct = m_loop == (mLoops - 1) ? mTail : mSplitSize;
                subMSizeActAlign = CeilDiv(subMSizeAct, C0_SIZE) * C0_SIZE;
                uint64_t bL1Offset = m_loop * MMAD_BASE_SIZE * MMAD_BASE_SIZE;
                uint64_t aL1Offset = m_loop * MMAD_BASE_SIZE * S_SPLITT_SIZE + n_loop * MMAD_BASE_SIZE * MMAD_BASE_SIZE;
                LocalTensor<T1> *l0_a_tensor = ping_pong_flag_l0_a_ ? &l0_a_pong_tensor : &l0_a_ping_tensor;
                LocalTensor<T1> *l0_b_tensor = ping_pong_flag_l0_b_ ? &l0_b_pong_tensor : &l0_b_ping_tensor;

                // load gm to L1B
                if (n_loop == 0 && sliceIdx == 0) {
                    CopyGmToL1(qL1Tensor[bL1Offset],
                        queryGm[dbParam.aTensorOffsetCv + m_loop * mSplitSize * dbParam.s1Stride],
                        subMSizeAct, d, dbParam.s1Stride);
                    AscendC::SetFlag<HardEvent::MTE2_MTE1>(ping_pong_flag_l0_b_ + 2);
                    AscendC::WaitFlag<HardEvent::MTE2_MTE1>(ping_pong_flag_l0_b_ + 2);
                }
                // load L1A To L0A
                AscendC::WaitFlag<HardEvent::M_MTE1>(ping_pong_flag_l0_a_ + 3);
                uint32_t s1_L10_count = (subMSizeActAlign + C0_SIZE - 1) / C0_SIZE;
                uint32_t s2_L10_count = (subNSizeActAlign + C0_SIZE - 1) / C0_SIZE;
                commonLoadData2dParamsTranspose.repeatTimes = s1_L10_count * s2_L10_count;
                commonLoadData2dParamsTranspose.srcStride = 1;
                AscendC::LoadData(
                    *l0_a_tensor,
                    dsL1Tensor[m_loop * MMAD_BASE_SIZE * S_SPLITT_SIZE + MMAD_BASE_SIZE * n_loop * MMAD_BASE_SIZE], 
                    commonLoadData2dParamsTranspose
                );
                // load L1B To L0b
                AscendC::WaitFlag<HardEvent::M_MTE1>(ping_pong_flag_l0_b_ + 3 + 2);
                commonLoadData2dParamsTranspose.repeatTimes = d / C0_SIZE;
                commonLoadData2dParamsTranspose.srcStride = subMSizeActAlign / C0_SIZE;
                for (int32_t i = 0; i < subMSizeActAlign / C0_SIZE; i++) {
                    AscendC::LoadData((*l0_b_tensor)[i * d * C0_SIZE],
                                        qL1Tensor[bL1Offset + i * SIZE_256],
                                        commonLoadData2dParamsTranspose);
                }
                AscendC::SetFlag<HardEvent::MTE1_M>(eventIdMte1BToM);
                AscendC::WaitFlag<HardEvent::MTE1_M>(eventIdMte1BToM);
                // matmul
                uint16_t s2_modify = (subNSizeAct == 1) ? 2 : subNSizeAct;
                commonMadParams.m = s2_modify;
                commonMadParams.n = d;
                commonMadParams.k = subMSizeAct;
                commonMadParams.unitFlag = m_loop == mLoops - 1 ? 3 : 2;
                commonMadParams.cmatrixInitVal = m_loop == 0 ? true : false;

                if ( m_loop==0 ) {
                    AscendC::WaitFlag<HardEvent::FIX_M>(ping_pong_flag_l0_c_);
                }

                if (m_loop == mLoops - 1) {
                    AscendC::PipeBarrier<PIPE_M>();
                }
                AscendC::Mmad(*l0_c_tensor, *l0_a_tensor, *l0_b_tensor, commonMadParams);
                AscendC::SetFlag<HardEvent::M_MTE1>(ping_pong_flag_l0_b_ + 3 + 2);
                AscendC::SetFlag<HardEvent::M_MTE1>(ping_pong_flag_l0_a_ + 3);
                ping_pong_flag_l0_a_ = 1 - ping_pong_flag_l0_a_;
                ping_pong_flag_l0_b_ = 1 - ping_pong_flag_l0_b_;
            }
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                gmBaseBlockOffset = n_loop * nSplitSize * C0_SIZE;
                commonFixpipeParamsV220.mSize = subNSizeAct;
                commonFixpipeParamsV220.nSize = d;
                commonFixpipeParamsV220.srcStride = subNSizeActAlign;
                commonFixpipeParamsV220.dstStride = s2 * 2;
                commonFixpipeParamsV220.quantPre = QuantMode_t::NoQuant;
                commonFixpipeParamsV220.unitFlag = 3;
                AscendC::SetAtomicType<float>();
                AscendC::Fixpipe<float, float, AscendC::CFG_NZ>(dkWorkSpaceGm[dkvOffset + dkOffsetBias + gmBaseBlockOffset], *l0_c_tensor, commonFixpipeParamsV220);
                AscendC::SetAtomicNone();
            } else {
                gmBaseBlockOffset = n_loop * nSplitSize * dbParam.s2Stride;
            }
            AscendC::SetFlag<HardEvent::FIX_M>(ping_pong_flag_l0_c_);
            ping_pong_flag_l0_c_ = 1 - ping_pong_flag_l0_c_;
            AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
            AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        }   
    }
    AscendC::WaitFlag<HardEvent::FIX_M>(eventIdFixpipeToM_ID0);
    AscendC::WaitFlag<HardEvent::FIX_M>(eventIdFixpipeToM_ID1);
    AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2_ID0);
    AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2_ID1);
    AscendC::WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_ID3);
    AscendC::WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_ID4);
    AscendC::WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_ID5);
    AscendC::WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_ID6);
    AscendC::SetFlag<HardEvent::M_MTE1>(eventIdMToMte1);
    AscendC::WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1);
    ///////////////////////////////////////////////////////////////
    // Matmal5 dv
    ///////////////////////////////////////////////////////////////
    mm3.SetOrgShape(dbParam.actualS2Len, dbParam.s1Stride / d * value_d, s1_size, dbParam.actualS1Len, dvKc);
    for (int32_t sliceIdx = 0; sliceIdx < s2SliceNum; ++sliceIdx) {
        int32_t s2SliceExtend = sliceIdx == (s2SliceNum - 1) ? s2SliceTail : s2SliceSize;
        int32_t s2SliceExtendAlign = CeilDiv(s2SliceExtend, C0_SIZE) * C0_SIZE;
        uint32_t nInnerLoops = (s2SliceExtend + nSplitSize - 1) / nSplitSize;
        uint32_t nInnerTail = s2SliceExtend - (nInnerLoops - 1) * nSplitSize;
        uint32_t subNSizeAct = nSplitSize;
        uint32_t subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;
        for (uint32_t n = 0; n < nInnerLoops; n++) { // s2
            subNSizeAct = n == (nInnerLoops - 1) ? nInnerTail : nSplitSize;
            subNSizeActAlign = CeilDiv(subNSizeAct, C0_SIZE) * C0_SIZE;
            for (uint32_t m = 0; m < mLoops; m++) { // s1
                bool atomicAddFlag = m != 0;
                subMSizeAct = m == (mLoops - 1) ? mTail : mSplitSize;
                subMSizeActAlign = CeilDiv(subMSizeAct, C0_SIZE) * C0_SIZE;
                uint64_t aL1Offset = MMAD_BASE_SIZE * MMAD_BASE_SIZE * (m + 4 * n);
                uint64_t bL1Offset = m * MMAD_BASE_SIZE * MMAD_BASE_SIZE;
                // B矩阵
                if (n == 0 && sliceIdx == 0) {
                    CopyGmToL1(dxL1Tensor[bL1Offset],
                        dxGm[dbParam.aTensorOffsetCv / d * value_d + m * mSplitSize * dbParam.s1Stride / d * value_d],
                        subMSizeAct, value_d, dbParam.s1Stride / d * value_d);
                }
                // A矩阵
                uint16_t blockCount = subNSizeActAlign / C0_SIZE;
                uint16_t blockLen = subMSizeActAlign * C0_SIZE * sizeof(T1) / BLOCK_SIZE;
                uint16_t srcStride = (s1_size - subMSizeActAlign) * C0_SIZE * sizeof(T1) / BLOCK_SIZE;
                uint16_t dstStride = 0;
                AscendC::DataCopyParams copyParams = {blockCount, blockLen, srcStride, dstStride};
                DataCopy(dsL1Tensor[aL1Offset],
                    dropWorkSpaceGm[pingpongIdx * cubeBaseMN * DTYPE_FACTOR + m * mSplitSize * C0_SIZE +
                        (n * nSplitSize + sliceIdx * s2SliceSize) * dbParam.s1CvExtendAlign * DTYPE_FACTOR],
                    copyParams);
                AscendC::SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                AscendC::WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
                mm3.SetTensorA(dsL1Tensor[aL1Offset], true);
                mm3.SetTensorB(dxL1Tensor[bL1Offset]);
                mm3.SetTail(subNSizeAct, value_d, subMSizeAct);
                mm3.template Iterate<false>(atomicAddFlag);
            }
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                gmBaseBlockOffset = (sliceIdx * s2SliceSize + n * nSplitSize) * C0_SIZE;
            } else {
                gmBaseBlockOffset = (sliceIdx * s2SliceSize + n * nSplitSize) * dbParam.s2Stride / d * value_d;
            }
            mm3.GetTensorC(dvWorkSpaceGm[dvOffset + gmBaseBlockOffset], 1);
        }
        AscendC::SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
    }
    mm3.End();
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::GetIndex(int64_t baseIdx, IndexParams& idx)
{
    if constexpr(INPUT_LAYOUT == TND) {
        int64_t actualSeqQlen = 0;
        int64_t actualSeqKvlen = 0;
        int64_t resbaseIdx = baseIdx;
        for (int64_t bIdx = 0; bIdx < b; bIdx++) {
            GetSeqQlenKvlenByBidx(bIdx, actualSeqQlen, actualSeqKvlen);
            s1Outer = (actualSeqQlen + s1CvInner - 1) / s1CvInner;
            s2Outer = (actualSeqKvlen + s2CvInner - 1) / s2CvInner;
            int64_t totalBaseIdx = n2 * g * s1Outer * s2Outer;
            if (resbaseIdx < totalBaseIdx) {
                idx.bIdx = bIdx;
                idx.n2Idx = resbaseIdx / (s2Outer * g * s1Outer);
                int64_t n2DimTail = resbaseIdx % (s2Outer * g * s1Outer);
                idx.s2oIdx = n2DimTail / (g * s1Outer);
                int64_t s2oDimTail = n2DimTail % (g * s1Outer);
                idx.gIdx = s2oDimTail / s1Outer;
                idx.s1oIdx = n2DimTail % s1Outer;
                break;
            } else {
                resbaseIdx -= totalBaseIdx;
            }
        }
    } else {
        idx.bIdx = baseIdx / (n2 * s2Outer * g * s1Outer);
        int64_t bDimTail = baseIdx % (n2 * s2Outer * g * s1Outer);
        idx.n2Idx  = bDimTail / (s2Outer * g * s1Outer);
        int64_t n2DimTail = baseIdx % (s2Outer * g * s1Outer);
        idx.s2oIdx = n2DimTail / (g * s1Outer);
        int64_t s2oDimTail = n2DimTail % (g * s1Outer);
        idx.gIdx = s2oDimTail / s1Outer;
        idx.s1oIdx = s2oDimTail % s1Outer;
    }
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::CalcDkvReduce(DBParams& dbParam,
    GlobalTensor<float> &srcTensor, GlobalTensor<float> &dstTensor, int64_t d, int64_t dAlign, uint32_t vecCalBlockNum)
{
    pingpongIdx = dbParam.taskId % 2;
    // 整块切分
    uint32_t s2CalcInner = (s2CvInner + vecCalBlockNum - 1) / vecCalBlockNum;
    int64_t singleCoreDataNum = s2CalcInner * dAlign;

    LocalTensor<float> resBuf = unifiedBuffer.GetWithOffset<float>(singleCoreDataNum, 0);
    LocalTensor<float> inBuf = unifiedBuffer.GetWithOffset<float>(singleCoreDataNum, singleCoreDataNum * sizeof(float));

    // 按gs1方向连续分核，dk、dv需要累加的数据是连续的
    for (int8_t groupId = 0; groupId < cubeCoreNum; groupId++) {
        if (dbParam.blockIdArr[groupId] == -1) {  // 后面的核没有数据
            break;
        }
        if (dbParam.kvGroupId[groupId] < groupId && dbParam.kvGroupId[groupId] != OUTIDX) {
            continue;
        }
        event_t vWaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        AscendC::SetFlag<HardEvent::MTE3_V>(vWaitMte3);
        AscendC::WaitFlag<HardEvent::MTE3_V>(vWaitMte3);
        Duplicate<float>(resBuf, 0.0, singleCoreDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        int64_t blockId = -1;
        int64_t maxS2Extend = 0;
        for (int8_t coreId = groupId; coreId < cubeCoreNum; coreId++) {
            if (dbParam.blockIdArr[coreId] == -1) { // 后面的核没有数据
                break;
            }
            if (groupId == dbParam.kvGroupId[coreId]) {
                blockId = dbParam.blockIdArr[coreId];
                uint32_t usedCoreNum = (dbParam.s2CvExtendArr[coreId] + s2CalcInner - 1) / s2CalcInner;
                if (cBlockIdx % vecCalBlockNum >= usedCoreNum) {
                    continue;
                }
                uint32_t s2CalcTail = dbParam.s2CvExtendArr[coreId] - s2CalcInner * (usedCoreNum - 1);
                uint32_t s2CalcExtend = (cBlockIdx % vecCalBlockNum == usedCoreNum - 1) ? s2CalcTail : s2CalcInner;
                maxS2Extend = maxS2Extend > s2CalcExtend ? maxS2Extend : s2CalcExtend;

                if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                    if (s2CalcExtend != s2CalcInner) {
                        AscendC::PipeBarrier<PIPE_V>();
                        Duplicate<float>(inBuf, 0.0, singleCoreDataNum);
                        event_t mte2WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                        AscendC::SetFlag<HardEvent::V_MTE2>(mte2WaitV);
                        AscendC::WaitFlag<HardEvent::V_MTE2>(mte2WaitV);
                    }
                    uint64_t srcOffset = pingpongIdx * cubeCoreNum * s2CvInner * dAlign + coreId * s2CvInner * dAlign +
                                     cBlockIdx % vecCalBlockNum * s2CalcInner * C0_SIZE;
                    AscendC::DataCopyExtParams intriParams;
                    intriParams.blockCount = dAlign / C0_SIZE;
                    intriParams.blockLen = s2CalcExtend * C0_SIZE * sizeof(float);
                    intriParams.srcStride = dbParam.s2CvExtendArr[coreId] * C0_SIZE * sizeof(float) - intriParams.blockLen;
                    intriParams.dstStride = (s2CalcInner - s2CalcExtend) * C0_SIZE / 8;
                    intriParams.rsv = 0;

                    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                    AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
                    AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
                    DataCopyPad(inBuf, srcTensor[srcOffset], intriParams, {false, 0, 0, 0});
                    event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    AscendC::SetFlag<HardEvent::MTE2_V>(vWaitMte2);
                    AscendC::WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
                    Add(resBuf, resBuf, inBuf, s2CalcInner * dAlign);
                    event_t mte2WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                    AscendC::SetFlag<HardEvent::V_MTE2>(mte2WaitV);  // 循环间的反向同步
                    AscendC::WaitFlag<HardEvent::V_MTE2>(mte2WaitV);

                } else {
                    uint64_t srcOffset = pingpongIdx * cubeCoreNum * s2CvInner * dAlign + coreId * s2CvInner * dAlign +
                                     cBlockIdx % vecCalBlockNum * s2CalcInner * d;

                    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                    AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
                    AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
                    DataCopy(inBuf, srcTensor[srcOffset], s2CalcExtend * d);
                    event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    AscendC::SetFlag<HardEvent::MTE2_V>(vWaitMte2);
                    AscendC::WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
                    Add(resBuf, resBuf, inBuf, s2CalcExtend * d);
                    event_t mte2WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                    AscendC::SetFlag<HardEvent::V_MTE2>(mte2WaitV);  // 循环间的反向同步
                    AscendC::WaitFlag<HardEvent::V_MTE2>(mte2WaitV);
                }
            }
        }
        //copyOut
        if (blockId != -1 && maxS2Extend != 0) {
            IndexParams idx;
            GetIndex(blockId, idx);
            uint64_t dstOffset = 0;
            uint32_t copyOutDstStride = 0;
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                if constexpr (INPUT_LAYOUT == TND) {  // TND Nz
                    int64_t actualS1Len = 0;
                    int64_t actualS2Len = 0;
                    GetSeqQlenKvlenByBidx(idx.bIdx, actualS1Len, actualS2Len);
                    if (idx.bIdx > 0) {
                        dstOffset = ((__gm__ int64_t *)actual_seq_kvlen_addr)[idx.bIdx - 1] * n2 * dAlign;
                    }
                    dstOffset += (idx.n2Idx * actualS2Len) * dAlign + (idx.s2oIdx * s2CvInner + cBlockIdx % vecCalBlockNum * s2CalcInner) * C0_SIZE;
                    copyOutDstStride = (actualS2Len - maxS2Extend) * C0_SIZE;
                } else {  // Other Nz
                    dstOffset = ((idx.bIdx * n2 + idx.n2Idx) * s2) * dAlign + (idx.s2oIdx * s2CvInner  + cBlockIdx % vecCalBlockNum * s2CalcInner) * C0_SIZE;
                    copyOutDstStride = (s2 - maxS2Extend) * C0_SIZE;
                }
            } else {
                if constexpr (INPUT_LAYOUT == TND) {
                if (idx.bIdx > 0) {
                    dstOffset = ((__gm__ int64_t *)actual_seq_kvlen_addr)[idx.bIdx - 1] * n2 * d;
                }
                    dstOffset += ((idx.s2oIdx * s2CvInner + cBlockIdx % vecCalBlockNum * s2CalcInner) * n2 + idx.n2Idx) * d;
                    copyOutDstStride = n2 * d - d;
                } else if constexpr (INPUT_LAYOUT == BNGSD) {
                    dstOffset = ((idx.bIdx * n2 + idx.n2Idx) * s2 + idx.s2oIdx * s2CvInner + cBlockIdx % vecCalBlockNum * s2CalcInner) * d;
                } else if constexpr (INPUT_LAYOUT == SBNGD) {
                    dstOffset = (((idx.s2oIdx * s2CvInner + cBlockIdx % vecCalBlockNum * s2CalcInner) * b + idx.bIdx) * n2 + idx.n2Idx) * d;
                    copyOutDstStride = b * n2 * d - d;
                } else if constexpr (INPUT_LAYOUT == BSNGD) {
                    dstOffset = ((idx.bIdx * s2 + idx.s2oIdx * s2CvInner + cBlockIdx % vecCalBlockNum * s2CalcInner) * n2 + idx.n2Idx) * d;
                    copyOutDstStride = n2 * d - d;
                }
            }
            event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            AscendC::SetFlag<HardEvent::V_MTE3>(mte3WaitV);
            AscendC::WaitFlag<HardEvent::V_MTE3>(mte3WaitV);
            SetAtomicAdd<float>();
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                DataCopyPad(dstTensor[dstOffset], resBuf, 
                            {static_cast<uint16_t>(dAlign / C0_SIZE), 
                            static_cast<uint32_t>(maxS2Extend * C0_SIZE * sizeof(float)),
                            static_cast<uint32_t>((s2CalcInner - maxS2Extend) * C0_SIZE / 8),
                            static_cast<uint32_t>(copyOutDstStride * sizeof(float)), 0});
            } else {
                DataCopyPad(dstTensor[dstOffset], resBuf, 
                    {static_cast<uint16_t>(maxS2Extend), static_cast<uint32_t>(d * sizeof(float)), 0,
                    static_cast<uint32_t>(copyOutDstStride * sizeof(float)), 0});
            }
            SetAtomicNone();
        }
    }
}

template <typename FAGT>
 __aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::CalcDqReduce(DBParams& dbParam,
    GlobalTensor<float> &srcTensor, GlobalTensor<float> &dstTensor, int64_t d, int64_t dAlign, uint32_t vecCalBlockNum)
{
    pingpongIdx = dbParam.taskId % 2;
    // 整块切分
    uint32_t s1CalcInner = (s1CvInner + vecCalBlockNum - 1) / vecCalBlockNum;
    int64_t singleCoreDataNum = s1CalcInner * dAlign;

    LocalTensor<float> dqRes = unifiedBuffer.GetWithOffset<float>(singleCoreDataNum, 0);
    LocalTensor<float> inBuf = unifiedBuffer.GetWithOffset<float>(singleCoreDataNum, singleCoreDataNum * sizeof(float));

    for (int8_t groupId = 0; groupId < cubeCoreNum; groupId++) {
        if (dbParam.blockIdArr[groupId] == -1) {  //后面的核没有数据
            break;
        }
        if (dbParam.dqGroupId[groupId] < groupId && dbParam.dqGroupId[groupId] != OUTIDX) {
            continue;
        }
        event_t vWaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        AscendC::SetFlag<HardEvent::MTE3_V>(vWaitMte3);
        AscendC::WaitFlag<HardEvent::MTE3_V>(vWaitMte3);
        Duplicate<float>(dqRes, 0.0, singleCoreDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        int64_t blockId = -1;
        int64_t maxS1Extend = 0;
        for (int8_t coreId = groupId; coreId < cubeCoreNum; coreId++) {
            if (dbParam.blockIdArr[coreId] == -1) { //后面的核没有数据
                break;
            }
            if (groupId == dbParam.dqGroupId[coreId]) {
                blockId = dbParam.blockIdArr[coreId];
                uint32_t usedCoreNum = (dbParam.s1CvExtendArr[coreId] + s1CalcInner - 1) / s1CalcInner;
                if (cBlockIdx % vecCalBlockNum >= usedCoreNum) {
                    continue;
                }
                uint32_t s1CalcTail = dbParam.s1CvExtendArr[coreId] - s1CalcInner * (usedCoreNum - 1);
                uint32_t s1CalcExtend = (cBlockIdx % vecCalBlockNum == usedCoreNum - 1) ? s1CalcTail : s1CalcInner;
                maxS1Extend = maxS1Extend > s1CalcExtend ? maxS1Extend : s1CalcExtend;

                //copyOut & add
                if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                    if (s1CalcExtend != s1CalcInner) {
                        AscendC::PipeBarrier<PIPE_V>();  // 循环间的反向同步
                        Duplicate<float>(inBuf, 0.0, singleCoreDataNum);
                        event_t mte2WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                        AscendC::SetFlag<HardEvent::V_MTE2>(mte2WaitV);
                        AscendC::WaitFlag<HardEvent::V_MTE2>(mte2WaitV);
                    }
                    uint64_t srcOffset = pingpongIdx * cubeCoreNum * s1CvInner * dAlign + coreId * s1CvInner * dAlign +
                                     cBlockIdx % vecCalBlockNum * s1CalcInner * C0_SIZE;
                    AscendC::DataCopyExtParams intriParams;
                    intriParams.blockCount = dAlign / C0_SIZE;
                    intriParams.blockLen = s1CalcExtend * C0_SIZE * sizeof(float);
                    intriParams.srcStride = dbParam.s1CvExtendArr[coreId] * C0_SIZE * sizeof(float) - intriParams.blockLen;
                    intriParams.dstStride = (s1CalcInner - s1CalcExtend) * C0_SIZE / 8;  // ub内按整块大小放置
                    intriParams.rsv = 0;

                    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                    AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
                    AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
                    DataCopyPad(inBuf, srcTensor[srcOffset], intriParams, {false, 0, 0, 0});
                    event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    AscendC::SetFlag<HardEvent::MTE2_V>(vWaitMte2);
                    AscendC::WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
                    Add(dqRes, dqRes, inBuf, s1CalcInner * dAlign);
                    event_t mte2WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                    AscendC::SetFlag<HardEvent::V_MTE2>(mte2WaitV);  // 循环间的反向同步
                    AscendC::WaitFlag<HardEvent::V_MTE2>(mte2WaitV);

                } else {
                    uint64_t srcOffset = pingpongIdx * cubeCoreNum * s1CvInner * dAlign + coreId * s1CvInner * dAlign +
                                     cBlockIdx % vecCalBlockNum * s1CalcInner * d;

                    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                    AscendC::SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
                    AscendC::WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
                    DataCopy(inBuf, srcTensor[srcOffset], s1CalcExtend * d);
                    event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    AscendC::SetFlag<HardEvent::MTE2_V>(vWaitMte2);
                    AscendC::WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
                    Add(dqRes, dqRes, inBuf, s1CalcExtend * d);
                    event_t mte2WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                    AscendC::SetFlag<HardEvent::V_MTE2>(mte2WaitV);  // 循环间的反向同步
                    AscendC::WaitFlag<HardEvent::V_MTE2>(mte2WaitV);
                }
            }
        }
        //copyOut
        if (blockId != -1 && maxS1Extend != 0) {
            IndexParams idx;
            GetIndex(blockId, idx);
            uint64_t dstOffset = 0;
            uint32_t copyOutDstStride = 0;
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                if constexpr (INPUT_LAYOUT == TND) {  // TND Nz
                    int64_t actualS1Len = 0;
                    int64_t actualS2Len = 0;
                    GetSeqQlenKvlenByBidx(idx.bIdx, actualS1Len, actualS2Len);
                    if (idx.bIdx > 0) {
                        dstOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[idx.bIdx - 1] * n2 * g * dAlign;
                    }
                    dstOffset += ((idx.n2Idx * g + idx.gIdx) * actualS1Len) * dAlign + 
                                 (idx.s1oIdx * s1CvInner + cBlockIdx % vecCalBlockNum * s1CalcInner) * C0_SIZE;
                    copyOutDstStride = (actualS1Len - maxS1Extend) * C0_SIZE;
                } else {  // Other Nz
                    dstOffset = (((idx.bIdx * n2 + idx.n2Idx) * g + idx.gIdx) * s1) * dAlign + 
                                (idx.s1oIdx * s1CvInner + cBlockIdx % vecCalBlockNum * s1CalcInner) * C0_SIZE;
                    copyOutDstStride = (s1 - maxS1Extend) * C0_SIZE;
                }
            } else {
                if constexpr (INPUT_LAYOUT == TND) {
                    if (idx.bIdx > 0) {
                        dstOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[idx.bIdx - 1] * n2 * g * d;
                    }
                    dstOffset += (((idx.s1oIdx * s1CvInner + cBlockIdx % vecCalBlockNum * s1CalcInner) * n2 + idx.n2Idx) * g + idx.gIdx) * d;
                    copyOutDstStride = n2 * g * d - d;
                } else if constexpr (INPUT_LAYOUT == BNGSD) {
                    dstOffset = (((idx.bIdx * n2 + idx.n2Idx) * g + idx.gIdx) * s1 + idx.s1oIdx * s1CvInner + cBlockIdx % vecCalBlockNum * s1CalcInner) * d;
                } else if constexpr (INPUT_LAYOUT == SBNGD) {
                    dstOffset = ((((idx.s1oIdx * s1CvInner + cBlockIdx % vecCalBlockNum * s1CalcInner) * b + idx.bIdx) * n2 + idx.n2Idx) * g + idx.gIdx) * d;
                    copyOutDstStride = b * n2 * g * d - d;
                } else if constexpr (INPUT_LAYOUT == BSNGD) {
                    dstOffset = (((idx.bIdx * s1 + idx.s1oIdx * s1CvInner + cBlockIdx % vecCalBlockNum * s1CalcInner) * n2 + idx.n2Idx) * g + idx.gIdx) * d;
                    copyOutDstStride = n2 * g * d - d;
                }
            }

            event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            AscendC::SetFlag<HardEvent::V_MTE3>(mte3WaitV);
            AscendC::WaitFlag<HardEvent::V_MTE3>(mte3WaitV);
            SetAtomicAdd<float>();
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                DataCopyPad(dstTensor[dstOffset], dqRes, 
                            {static_cast<uint16_t>(dAlign / C0_SIZE), 
                            static_cast<uint32_t>(maxS1Extend * C0_SIZE * sizeof(float)),
                            static_cast<uint32_t>((s1CalcInner - maxS1Extend) * C0_SIZE / 8),
                            static_cast<uint32_t>(copyOutDstStride * sizeof(float)), 0});
            } else {
                DataCopyPad(dstTensor[dstOffset], dqRes,
                            {static_cast<uint16_t>(maxS1Extend), static_cast<uint32_t>(d * sizeof(float)), 0,
                            static_cast<uint32_t>(copyOutDstStride * sizeof(float)), 0});
            }
            SetAtomicNone();
        }
    }
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::ComputeVecAdd(DBParams& dbParam)
{
    int64_t s1CalcInner = (s1CvInner + vecBlockNum - 1) / vecBlockNum;
    int64_t s2CalcInner = (s2CvInner + vecBlockNum - 1) / vecBlockNum;

    // When workspace is insufficient, fall back to a full-core reduction, and use
    // coreNum` as vecCalBlockNum to ensure global deterministic accumulation and stable numerical precision.
    if (unlikely(s1CalcInner * dAlign * sizeof(float) * 2 > TOTAL_SIZE ||
        s2CalcInner * dAlign * sizeof(float) * 2 > TOTAL_SIZE)) {
        CalcDqReduce(dbParam, dqDtmWsGm, dqWorkSpaceGm, d, dAlign, coreNum);
        CalcDkvReduce(dbParam, dkDtmWsGm, dkWorkSpaceGm, d, dAlign, coreNum);
        CalcDkvReduce(dbParam, dvDtmWsGm, dvWorkSpaceGm, value_d, value_dAlign, coreNum);
        if constexpr (HAS_ROPE == ENABLE) {
            CalcDqReduce(dbParam, dqRopeDtmWsGm, dqRopeWorkSpaceGm, rope_d, rope_dAlign, coreNum);
            CalcDkvReduce(dbParam, dkRopeDtmWsGm, dkRopeWorkSpaceGm, rope_d, rope_dAlign, coreNum);
        }
    } else {
        if (cBlockIdx < vecBlockNum) {
            CalcDqReduce(dbParam, dqDtmWsGm, dqWorkSpaceGm, d, dAlign, vecBlockNum);
            if constexpr (HAS_ROPE == ENABLE) {
               CalcDqReduce(dbParam, dqRopeDtmWsGm, dqRopeWorkSpaceGm, rope_d, rope_dAlign, vecBlockNum);
            }
        } else if (cBlockIdx < 2 * vecBlockNum) {
            CalcDkvReduce(dbParam, dkDtmWsGm, dkWorkSpaceGm, d, dAlign, vecBlockNum);
            if constexpr (HAS_ROPE == ENABLE) {
               CalcDkvReduce(dbParam, dkRopeDtmWsGm, dkRopeWorkSpaceGm, rope_d, rope_dAlign, vecBlockNum);
            }
        } else if (cBlockIdx < 3 * vecBlockNum) {
            CalcDkvReduce(dbParam, dvDtmWsGm, dvWorkSpaceGm, value_d, value_dAlign, vecBlockNum);
        }
    }
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::ProcessFirstMM()
{
    bool isFinish = false;
    int64_t startCoreId = 0;
    while (!isFinish) {
        isFinish = CalcValidBlock(blockStartIdx, startCoreId, dbParams[0]);
    }
    dbParams[0].taskId = 0;
    if (dbParams[0].blockId != -1) {
        if ASCEND_IS_AIC {
            ComputeMM1(dbParams[0]);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG[dbParams[0].taskId]);
        }
    }
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::Process()
{
    int64_t taskId = 1;
    int8_t extraLoopNum = 1;
    if constexpr (IS_DTM == ENABLE) {
        extraLoopNum = 2;
    }

    int64_t startOffset = cBlockIdx / 2 * 2 * 512;    // 软同步workspace偏移
    GM_ADDR startGmAddr = workspaceAddr;
    GroupBarrier<PipeMode::MTE3_MODE> blockBar(startGmAddr + startOffset, 2, 2);
    event_t eventIdMte1ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
    while (extraLoopNum >= 0) {
        bool isFinish = false;
        int64_t startCoreId = 0;
        while (!isFinish) {
            isFinish = CalcValidBlock(blockStartIdx, startCoreId, dbParams[taskId % 3]);
        }
        dbParams[taskId % 3].taskId = taskId;
        if (taskId > 0 && dbParams[(taskId - 1) % 3].blockId != -1) {
            if ASCEND_IS_AIV {
                CrossCoreWaitFlag(SYNC_C1_V1_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);
            }
        }

        if ASCEND_IS_AIC {
            if (dbParams[taskId % 3].blockId != -1) {
                ComputeMM1(dbParams[taskId % 3]);
                CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG[dbParams[taskId % 3].taskId % 2]);
            }
        }
        if (taskId > 0 && dbParams[(taskId - 1) % 3].blockId != -1) {
            if ASCEND_IS_AIV {
                ComputeVec(dbParams[(taskId - 1) % 3]);    // vec compute
                CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V1_C2_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);
            }
            // ---非确定性计算---
            if constexpr (IS_DTM != ENABLE) {
                if ASCEND_IS_AIC {
                    if constexpr(!(DTEMPLATETYPE == DTemplateType::Aligned128 || DTEMPLATETYPE == DTemplateType::Aligned192 || DTEMPLATETYPE == DTemplateType::Aligned64)) {
                        AscendC::SetFlag<HardEvent::MTE1_MTE2>(static_cast<int32_t>(eventIdMte1ToMte2));
                        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(static_cast<int32_t>(eventIdMte1ToMte2));
                    }
                    CrossCoreWaitFlag(SYNC_V1_C2_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);
                    if constexpr (DTEMPLATETYPE == DTemplateType::Aligned128) {
                        ComputeL1CustomMMDqkv(dbParams[(taskId - 1) % 3], dbParams[(taskId) % 3].blockId);
                    } else {
                        ComputeMMDqkv(dbParams[(taskId - 1) % 3], dbParams[(taskId) % 3].blockId);
                    }
                    if (dbParams[(taskId) % 3].blockId == -1) {
                        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C2_V1_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);
                    }
                }

                if ASCEND_IS_AIV {
                    if (dbParams[(taskId) % 3].blockId == -1) {
                        CrossCoreWaitFlag(SYNC_C2_V1_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);
                    }
                }
            }
        }

        // -----确定性计算-----
        if constexpr (IS_DTM == ENABLE) {
            if (taskId > 1 && dbParams[(taskId - 2) % 3].blockId != -1) {
                if ASCEND_IS_AIV {
                    CrossCoreWaitFlag(SYNC_C2_V1_FLAG[dbParams[(taskId - 2) % 3].taskId % 2]);
                }
            }
            if (taskId > 0 && dbParams[(taskId - 1) % 3].blockId != -1) {
                if ASCEND_IS_AIC {
                    AscendC::SetFlag<HardEvent::MTE1_MTE2>(static_cast<int32_t>(eventIdMte1ToMte2));
                    AscendC::WaitFlag<HardEvent::MTE1_MTE2>(static_cast<int32_t>(eventIdMte1ToMte2));
                    CrossCoreWaitFlag(SYNC_V1_C2_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);
                    DTMComputeMMDqkv(dbParams[(taskId - 1) % 3], dbParams[(taskId) % 3].blockId);
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C2_V1_FLAG[dbParams[(taskId - 1) % 3].taskId % 2]);
                }
            }

            if (taskId > 1) {
                SyncAll();
                if ASCEND_IS_AIV {
                    ComputeVecAdd(dbParams[(taskId - 2) % 3]);
                }
                SyncAll();
            }
        }

        taskId++;
        if (blockStartIdx == -1) {
            extraLoopNum -= 1;
        }
    }
    // -----非确定性计算，最后释放L1自主管理InitTscmBuffer分配的eventID-----
    if constexpr (IS_DTM != ENABLE) {
        if ASCEND_IS_AIC {
            if constexpr (DTEMPLATETYPE == DTemplateType::Aligned128) {
                ReleaseTSCMAllocEventID();
            }
        }
    }
    // -----确定性计算，最后做一次全核同步，保证所有核的atomicAdd做完-----
    if constexpr (IS_DTM == ENABLE) {
        SyncAll();
    }
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::UpdateToken(int64_t bIdx)
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
    } else if (sparseMode == 3 || sparseMode == 4 || (sparseMode == 7 && bIdx == bandIdx) || sparseMode == 6 ||
               (sparseMode == 8 && bIdx == bandIdx)) {
        GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
        actualCalcS1Token = s1Token + actualS1Len - actualS2Len;
        actualCalcS2Token = s2Token - actualS1Len + actualS2Len;
    }
}

template <typename FAGT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::CalcAttenMaskOffset(int64_t &attenMaskOffset, const int64_t delta,
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

template <typename FAGT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::CalcAttenMaskOffsetForPrefixCompressMode(int64_t &attenMaskOffset,
    int64_t &attenMaskOffset2, const int64_t delta, uint32_t s1VSize, uint32_t s2VSize, uint32_t s2VBegin,
    bool &canSimplify, DBParams& dbParam)
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
    uint32_t curBatchDimIdx = dbParam.bIdx;;
    if constexpr (INPUT_LAYOUT == TND) {
        S1 = dbParam.actualS1Len;
        S2 = dbParam.actualS2Len;
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

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::CalcAttenMaskOffsetWithSparseModeForUnpad(
    int64_t &attenMaskOffset, int64_t &attenMaskOffset2,
    uint32_t s1VSize, uint32_t s2VSize, int64_t curS1Idx,
    uint32_t s2VBegin, bool unpadUseBand, bool &canSimplify, DBParams &dbParam)
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    uint64_t compressMode = TilingData->s1s2BNGS1S2BaseParams.attenMaskCompressMode;
    int64_t causal_delta =
        static_cast<int64_t>(dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) - static_cast<int64_t>(s2VBegin);
    CalcAttenBandMode(compressMode, causal_delta, dbParam);
    if (compressMode == 1 || (sparseMode == 8 && dbParam.bIdx != bandIdx)) { // causal s1==s2
        CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1VSize, s2VSize);
        return;
    }

    if (compressMode == 2 || (sparseMode == 7 && dbParam.bIdx != bandIdx)) { // causal s1!=s2
        GetSeqQlenKvlenByBidx(dbParam.bIdx, actualS1Len, actualS2Len);
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
                                                 s2VBegin, canSimplify, dbParam);
        return;
    }

    if (TilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType == 0) {
        attenMaskDimS2 = (uint32_t)s2;
        attenMaskOffset += (static_cast<int64_t>(dbParam.s1oIdx) * s1CvInner + curS1Idx * s1VecSize) * s2 + s2VBegin;
    } else if (TilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType == 1) {
        attenMaskDimS2 = (uint32_t)dbParam.actualS2Len;
        for (uint32_t bidx = 0; bidx < dbParam.bIdx; bidx++) {
            GetSeqQlenKvlenByBidx(bidx, actualS1Len, actualS2Len);
            attenMaskOffset += actualS1Len * actualS2Len;
        }
        attenMaskOffset += (dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) *
                           dbParam.actualS2Len + s2VBegin;
    } else {
        attenMaskDimS2 = (uint32_t)dbParam.actualS2Len;
        for (uint32_t bidx = 0; bidx < dbParam.bIdx; bidx++) {
            GetSeqQlenKvlenByBidx(bidx, actualS1Len, actualS2Len);
            attenMaskOffset += static_cast<int64_t>(n2) * g * actualS1Len * actualS2Len;
        }
        attenMaskOffset += ((dbParam.n2Idx * g + dbParam.gIdx) * dbParam.actualS1Len +
                           dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) * dbParam.actualS2Len + s2VBegin;
    }
}

template <typename FAGT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::CalcAttenMaskOffsetWithSparseMode(int64_t &attenMaskOffset,
    int64_t &attenMaskOffset2, uint32_t s1VSize, uint32_t s2VSize, int64_t curS1Idx, uint32_t s2VBegin,
    bool &canSimplify, DBParams &dbParam)
{
    uint64_t compressMode = TilingData->s1s2BNGS1S2BaseParams.attenMaskCompressMode;
    int64_t causal_delta =
        static_cast<int64_t>(dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) - static_cast<int64_t>(s2VBegin);
    CalcAttenBandMode(compressMode, causal_delta, dbParam);
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
                                                 s2VBegin, canSimplify, dbParam);
        return;
    }

    if (TilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType == 0) {
        attenMaskOffset = (static_cast<int64_t>(dbParam.s1oIdx) * s1CvInner + curS1Idx * s1VecSize) * s2 + s2VBegin;
    } else if (TilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType == 1) {
        attenMaskOffset =
            (dbParam.bIdx * s1 + dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) * s2 + s2VBegin;
    } else {
        attenMaskOffset = (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1 +
                           dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) * s2 + s2VBegin;
    }
}

template <typename FAGT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::NZCopyIn(int64_t mmAddr, GlobalTensor<T2> &mmWspGm,
                                                          LocalTensor<T2> &mmTensorCurr, uint32_t s1VecSize,
                                                          uint32_t s2VecSize, uint32_t s1CvInner)
{
    /*
    Func:
    MM输出NZ数据，数据搬运进UB
    */
    DataCopyParams intriParams;
    intriParams.blockCount = s2VecSize / C0_SIZE;
    intriParams.blockLen = s1VecSize * C0_SIZE / cal_block_num;
    intriParams.srcStride = s1CvInner * C0_SIZE / cal_block_num - intriParams.blockLen;
    intriParams.dstStride = 1;
    DataCopy(mmTensorCurr, mmWspGm[mmAddr], intriParams);
}

template <typename FAGT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::NZ2ND(LocalTensor<T2> &mmTensorCurr, LocalTensor<T2> &tmpTensor,
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

template <typename FAGT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::ND2NZ(LocalTensor<T1> &mmTensorCurr, LocalTensor<T1> &tmpTensor,
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


template <typename FAGT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::CalcAttenBandMode(
    int64_t compressMode, int64_t causal_delta, DBParams &dbParam)
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    if (compressMode == 1 || compressMode == 2 || compressMode == 3 || sparseMode == 7 || sparseMode == 8) { // compress
        int64_t next_delta = causal_delta;
        int64_t pre_delta = causal_delta - INT32_MAX - 1;
        if (compressMode == 1 || (sparseMode == 8 && dbParam.bIdx != bandIdx)) {
        } else if (compressMode == 2) {
            if constexpr (INPUT_LAYOUT == TND) {
                next_delta = causal_delta - dbParam.actualS1Len + dbParam.actualS2Len;
            } else {
                next_delta = causal_delta - s1 + s2;
            }
        } else if (sparseMode == 7 && dbParam.bIdx != bandIdx) {
            next_delta = causal_delta - dbParam.actualS1Len + dbParam.actualS2Len;
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

template <typename FAGT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::DropOutCopy(LocalTensor<uint8_t> &vecInDropBuffer,
                                                             int64_t curS1Idx, int64_t s2VBegin)
{
    // for compute dropout mask offset
    dropMaskInfo.s2StartIdx = s2VBegin;
    // for copy in dropout mask
    dropMaskInfo.s2CopySize = s2Extend;

    CopyInDropMask<true>(vecInDropBuffer, maskWorkSpaceGm, dropMaskGm, this->dropMaskInfo);
}

template <typename FAGT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::SubGrapA(int64_t curIdx, int64_t curS1Idx, int64_t curS2Idx,
                                    DBParams& dbParam, event_t mte2WaitMte3A)
{
    pingpongIdx = dbParam.taskId % 2;
    s2Extend = (curS2Idx == s2VecLoop - 1) ? (dbParam.s2CvExtend - (s2VecLoop - 1) * s2VecSize) : s2VecSize;
    s2ExtendAlign = (s2Extend + 15) / 16 * 16;
    uint32_t s2VBegin = dbParam.s2oIdx * s2CvInner + curS2Idx * s2VecSize;

    uint32_t ubBufferOffset = 0;
    uint32_t ubTmpBufferOffset = 0;

    if (curIdx > 0) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3A));
    }

    LocalTensor<float> vecInBuffer3 =
        unifiedBuffer.GetWithOffset<float>(8 * 1024 / sizeof(float), ubBufferOffset + T2BlockBegin);
    int64_t softMaxOffset = 0;
    if constexpr (INPUT_LAYOUT == TND) {
        if(tndSoftmaxIn){
            int64_t innerRowOffsetLeft = unlikely(dbParam.bIdx == 0) ? 0 : ((__gm__ int64_t *)actual_seq_qlen_addr)[dbParam.bIdx - 1] * 32 / sizeof(float);
            int64_t originInnerBatchOffset = ((dbParam.n2Idx * g + dbParam.gIdx) * dbParam.actualS1Len +
                            dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) * 32 / sizeof(float);
            softMaxOffset = ((((__gm__ int64_t *)actual_seq_qlen_addr)[b - 1] * 32 / sizeof(float)) * (dbParam.n2Idx * g + dbParam.gIdx) + innerRowOffsetLeft + originInnerBatchOffset % (dbParam.actualS1Len * 32 / sizeof(float)));
        }else {
            if (dbParam.bIdx > 0) {
                softMaxOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[dbParam.bIdx - 1] * n2 * g * 32 / sizeof(float);
            }
            softMaxOffset += ((dbParam.n2Idx * g + dbParam.gIdx) * dbParam.actualS1Len +
                            dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) * 32 / sizeof(float);
        }
    } else {
        softMaxOffset = (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1 + dbParam.s1oIdx * s1CvInner +
                         curS1Idx * s1VecSize) * 32 / sizeof(float);
    }
    CopyInSoftMax(vecInBuffer3, s1ExtendSubGraph, softMaxOffset);

    LocalTensor<T1> pseUbT1 = unifiedBuffer.GetWithOffset<T1>(16 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
    LocalTensor<half> pseUb = pseUbT1.template ReinterpretCast<half>();
    if constexpr (IS_PSE == ENABLE) {
        pseInfo.bSSOffset = dbParam.bIdx * s1 * s2;
        pseInfo.s2SizeAcc = dbParam.bIdx * s2;
        pseInfo.boIdx = dbParam.bIdx;
        pseInfo.n2oIdx = dbParam.n2Idx;
        pseInfo.goIdx = dbParam.gIdx;
        pseInfo.s1oIdx = dbParam.s1oIdx;
        pseInfo.loopIdx = curS1Idx;
        pseInfo.vec1S1BaseSize = s1VecSize;
        pseInfo.vec1S1RealSize = s1ExtendSubGraph;
        pseInfo.s1BaseSize = s1CvInner;
        pseInfo.s2RealSize = s2Extend;
        pseInfo.s2AlignedSize = s2ExtendAlign;
        pseInfo.s2StartIdx = s2VBegin;
        LocalTensor<T2> noCastedPseUb = unifiedBuffer.GetWithOffset<T2>(0 / sizeof(T2), 0);
        if (pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
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
        unifiedBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + BoolBegin);
    bool unpadUseBand = (sparseMode == 7 && dbParam.bIdx == bandIdx) || (sparseMode == 8 && dbParam.bIdx == bandIdx);
    int64_t attenMaskOffsetPre = 0;
    bool prefixCompressCanSimplify = false;
    if constexpr (IS_ATTEN_MASK == ENABLE) {
        int64_t attenMaskOffset = 0;
        if constexpr(INPUT_LAYOUT == TND) {
            CalcAttenMaskOffsetWithSparseModeForUnpad(attenMaskOffset, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend,
                                                    curS1Idx, s2VBegin, unpadUseBand, prefixCompressCanSimplify, dbParam);
        } else {
            CalcAttenMaskOffsetWithSparseMode(attenMaskOffset, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend, curS1Idx,
                                            s2VBegin, prefixCompressCanSimplify, dbParam);
        }
        // uint8_t
        if (AttenBandMode == AttenMaskCompress::All || AttenBandMode == AttenMaskCompress::NextOnly) {
            CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffset, s1ExtendSubGraph, s2Extend);
        } else if (AttenBandMode == AttenMaskCompress::PreOnly) {
            CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend);
        }
    }

    LocalTensor<uint8_t> vecInDropBuffer =
        unifiedBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + U8Begin);
    if constexpr (IS_DROP == ENABLE) {
        if constexpr (IsSameType<T1, float>::value) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        DropOutCopy(vecInDropBuffer, curS1Idx, s2VBegin);
    }

    LocalTensor<float> vecClc2Buffer =
        unifiedBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), ubBufferOffset + T2Begin);
    if constexpr (MM_OUT_FORMAT == CubeFormat::ND) {
        if (s2VecLoop == 1) {
            DataCopy(vecClc2Buffer, mm2WorkspaceGm[pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * s2ExtendAlign],
                    s1ExtendSubGraph * s2ExtendAlign);
        } else {
            DataCopyPad(vecClc2Buffer, mm2WorkspaceGm[pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * dbParam.s2CvExtendAlign + curS2Idx * s2VecSize],
                        {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                         static_cast<uint16_t>((dbParam.s2CvExtendAlign - s2ExtendAlign) * sizeof(float)), 0},
                        {false, 0, 0, 0});
        }
        event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        AscendC::SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
        AscendC::WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    } else {
        int64_t mmAddr = pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * C0_SIZE + curS2Idx * dbParam.s1CvExtendAlign * s2VecSize;
        NZCopyIn(mmAddr, mm2WorkspaceGm, vecClc2Buffer, s1VecSize, s2ExtendAlign, dbParam.s1CvExtendAlign);
        event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        AscendC::SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
        AscendC::WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
        auto tmpTensor = unifiedBuffer.GetWithOffset<T2>(TMP_UB_SIZE / sizeof(T2), TMP_UB_OFFSET);
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
        LocalTensor<T2> castTensor = unifiedBuffer.GetWithOffset<T2>(TMP_UB_SIZE / sizeof(T2), TMP_UB_OFFSET);
        if (!(pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE)) {

            if constexpr (!IsSameType<T1, float>::value) {
                uint32_t calculateRowsAlign = (s2Extend + input_block_num - 1) / input_block_num * input_block_num;
                Cast(castTensor, pseUbT1, RoundMode::CAST_NONE, repeatTimes * calculateRowsAlign);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                event_t mte2WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                AscendC::SetFlag<HardEvent::V_MTE2>(static_cast<int32_t>(mte2WaitV));
                AscendC::WaitFlag<HardEvent::V_MTE2>(static_cast<int32_t>(mte2WaitV));
                if constexpr (INPUT_LAYOUT == TND) {
                    PseCopyIn<T1, T2, LayOutTypeEnum::LAYOUT_TND, true>(castTensor, castTensor, this->pseGm, pseInfo);
                } else {
                    PseCopyIn<T1, T2, LayOutTypeEnum::LAYOUT_BNSD, true>(castTensor, castTensor, this->pseGm, pseInfo);
                }
                event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                AscendC::SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
                AscendC::WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
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

        if (compressMode == 4) {   // 4: prefix compress
            if (prefixCompressCanSimplify == false) {
                LocalTensor<uint8_t> attenMaskUbPreuint8 =
                    unifiedBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), TMP_UB_OFFSET + ubTmpBufferOffset);
                uint32_t s2ExtendPadAlign = (s2Extend + 31) / 32 * 32; // attenmask做pad时会32对齐，故加31/32做ceil
                int32_t maskNum = s1ExtendSubGraph * s2ExtendPadAlign / 2; // 除2数据量按照uint16类型折半

                event_t mte2WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                AscendC::SetFlag<HardEvent::V_MTE2>(static_cast<int32_t>(mte2WaitV));
                AscendC::WaitFlag<HardEvent::V_MTE2>(static_cast<int32_t>(mte2WaitV));
                CopyInAttenMaskBool(attenMaskUbPreuint8, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend);

                event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                AscendC::SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
                AscendC::WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
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

        if ((compressMode == 3 || unpadUseBand) && AttenBandMode == AttenMaskCompress::All) {   // 3: band
            event_t mte2WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            AscendC::SetFlag<HardEvent::V_MTE2>(static_cast<int32_t>(mte2WaitV));
            AscendC::WaitFlag<HardEvent::V_MTE2>(static_cast<int32_t>(mte2WaitV));
            CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend);
            event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            AscendC::SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
            AscendC::WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
            CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8, s1ExtendSubGraph, s2ExtendAlign, 1);
        }
    }

    ///////////////////////////////////////////////////////////////
    // simpleSoftMax
    ///////////////////////////////////////////////////////////////
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<float> simpleSoftmaxResBuf = unifiedBuffer.GetWithOffset<float>(33 * 1024 / sizeof(T2), DbBegin);
    CalcSoftMax(simpleSoftmaxResBuf, vecClc2Buffer, vecInBuffer3, s1ExtendSubGraph, s2Extend, s2ExtendAlign, TilingData->softmaxTilingData);

    ///////////////////////////////////////////////////////////////
    // dropout
    ///////////////////////////////////////////////////////////////
    LocalTensor<T2> vecDropBuffer = simpleSoftmaxResBuf;
    if constexpr (IS_DROP == ENABLE) {
        vecDropBuffer = unifiedBuffer.GetWithOffset<T2>(33 * 1024 / sizeof(T2), TMP_UB_OFFSET);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<uint8_t> tmpDropBuffer =
            unifiedBuffer.GetWithOffset<uint8_t>(32 * 1024 / sizeof(uint8_t), ubBufferOffset + T1Begin);

        // for compute dropout mask
        dropMaskInfo.lstAxis = s2ExtendAlign;
        dropMaskInfo.maskLstAxis = s2ExtendAlign;
        ComputeDropMask<T2, true>(vecDropBuffer, simpleSoftmaxResBuf, vecInDropBuffer, tmpDropBuffer, this->dropMaskInfo);
        if constexpr (IsSameType<T1, float>::value) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    ///////////////////////////////////////////////////////////////
    // cast fp322bf16
    ///////////////////////////////////////////////////////////////
    LocalTensor<T1> vecCopyOutBuffer = vecDropBuffer.template ReinterpretCast<T1>();
    if constexpr (!IsSameType<T1, float>::value) {
        vecCopyOutBuffer = unifiedBuffer.GetWithOffset<T1>(17 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(vecCopyOutBuffer, vecDropBuffer, RoundMode::CAST_ROUND, s1ExtendSubGraph * s2ExtendAlign);
    }
    int64_t copyOutOffset = 0;
    DataCopyParams copyOutParam;
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<T1> tmpTensor = unifiedBuffer.GetWithOffset<T1>(TMP_UB_SIZE / sizeof(T1), TMP_UB_OFFSET);
        copyOutOffset = pingpongIdx * cubeBaseMN * DTYPE_FACTOR + curS1Idx * s1VecSize * C0_SIZE +
                        curS2Idx * dbParam.s1CvExtendAlign * DTYPE_FACTOR * s2VecSize;
        copyOutParam = {
            static_cast<uint16_t>(s2ExtendAlign / C0_SIZE),
            static_cast<uint16_t>(s1ExtendSubGraph * C0_SIZE * sizeof(T1)),
            1,
            static_cast<uint16_t>((dbParam.s1CvExtendAlign * DTYPE_FACTOR - s1ExtendSubGraph) * C0_SIZE * sizeof(T1))
        };
        DataCopy(tmpTensor, vecCopyOutBuffer, s1ExtendSubGraph * s2ExtendAlign);
        AscendC::PipeBarrier<PIPE_V>();
        ND2NZ(vecCopyOutBuffer, tmpTensor, s1ExtendSubGraph, s2ExtendAlign);
    } else {
        copyOutOffset = pingpongIdx * cubeBaseMN * DTYPE_FACTOR +
                        curS1Idx * s1VecSize * dbParam.s2CvExtendAlign * DTYPE_FACTOR + curS2Idx * s2VecSize;
        copyOutParam = {
            static_cast<uint16_t>(s1ExtendSubGraph),
            static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)),
            0,
            static_cast<uint16_t>((dbParam.s2CvExtendAlign * DTYPE_FACTOR - s2ExtendAlign) * sizeof(T1))
        };
    }
    event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    AscendC::SetFlag<HardEvent::V_MTE3>(static_cast<int32_t>(mte3WaitV));
    AscendC::WaitFlag<HardEvent::V_MTE3>(static_cast<int32_t>(mte3WaitV));
    DataCopyPad(dropWorkSpaceGm[copyOutOffset], vecCopyOutBuffer, copyOutParam);

    if (curIdx < vecLoopEnd - vecLoopStart - 1) {
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3A));
    }
}

template <typename FAGT>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::SubGrapB(int64_t curIdx, int64_t s1VecLoop, int64_t s2VecLoop,
                                     int64_t curS1Idx, int64_t curS2Idx, DBParams& dbParam, event_t mte2WaitMte3B, float* dsinkSumLocal)
{
    pingpongIdx = dbParam.taskId % 2;
    uint32_t ubBufferOffset = DbBegin;
    s2Extend = (curS2Idx == s2VecLoop -1) ? (dbParam.s2CvExtend - (s2VecLoop - 1) * s2VecSize) : s2VecSize;
    s2ExtendAlign = (s2Extend + 15) / 16 * 16;

    if (curIdx > 0) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3B));
    }

    if (preS1Idx != curS1Idx) {    // copyIn sfmg
        preS1Idx = curS1Idx;
        LocalTensor<float> sfmgClc3 = unifiedBuffer.GetWithOffset<float>(SFMG_UB_SIZE / sizeof(float), SFMG_UB_OFFSET);
        DataCopy(sfmgClc3, sfmgWorkspaceGm[sfmgOffset + curS1Idx * s1VecSize * 8], s1ExtendSubGraph * 8);
    }

    LocalTensor<uint8_t> vecInDropBuffer =
        unifiedBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + U8Begin);
    if constexpr (IS_DROP == ENABLE) {
        int64_t s2VBegin = dbParam.s2oIdx * s2CvInner + curS2Idx * s2VecSize;
        DropOutCopy(vecInDropBuffer, curS1Idx, s2VBegin);
        if constexpr (IsSameType<T1, float>::value) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    LocalTensor<T2> vecClc1Buffer = unifiedBuffer.GetWithOffset<T2>(33 * 1024 / sizeof(T2), ubBufferOffset + T1Begin);
    // copyIn dyv
    LocalTensor<T2> dyvBuffer = unifiedBuffer.GetWithOffset<T2>(33 * 1024 / sizeof(T2), TMP_UB_OFFSET);
    bool has_sink = (TilingData->s1s2BNGS1S2BaseParams.sink == 1);
    if constexpr (MM_OUT_FORMAT == CubeFormat::ND) {
        if (s2VecLoop == 1) {
            DataCopy(vecClc1Buffer, mm1WorkspaceGm[pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * s2ExtendAlign],
                     s1ExtendSubGraph * s2ExtendAlign);
        } else {
            DataCopyPad(vecClc1Buffer, mm1WorkspaceGm[pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * dbParam.s2CvExtendAlign + curS2Idx * s2VecSize],
                        {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                         static_cast<uint16_t>((dbParam.s2CvExtendAlign - s2ExtendAlign) * sizeof(float)), 0},
                        {false, 0, 0, 0});      
        }
        event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        AscendC::SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
        AscendC::WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
    } else {
        int64_t mmAddr = pingpongIdx * cubeBaseMN + curS1Idx * s1VecSize * C0_SIZE + curS2Idx * dbParam.s1CvExtendAlign * s2VecSize;
        NZCopyIn(mmAddr, mm1WorkspaceGm, vecClc1Buffer, s1VecSize, s2ExtendAlign, dbParam.s1CvExtendAlign);
        event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        AscendC::SetFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
        AscendC::WaitFlag<HardEvent::MTE2_V>(static_cast<int32_t>(vWaitMte2));
        auto tmpTensor = unifiedBuffer.GetWithOffset<T2>(TMP_UB_SIZE / sizeof(T2), TMP_UB_OFFSET);
        DataCopy(tmpTensor, vecClc1Buffer, s1VecSize * s2ExtendAlign + s2ExtendAlign / C0_SIZE * VEC_REPEAT);
        AscendC::PipeBarrier<PIPE_V>();
        NZ2ND(vecClc1Buffer, tmpTensor, s1VecSize, s2ExtendAlign);
    }

    ///////////////////////////////////////////////////////////////
    // ss
    ///////////////////////////////////////////////////////////////
    if constexpr (IS_DROP == ENABLE) {
        LocalTensor<uint8_t> tmpDropBuffer = unifiedBuffer.GetWithOffset<uint8_t>(32 * 1024 / sizeof(uint8_t), TMP_UB_OFFSET);

        // for compute dropout mask
        dropMaskInfo.lstAxis = s2ExtendAlign;
        dropMaskInfo.maskLstAxis = s2ExtendAlign;
        ComputeDropMask<T2, true>(vecClc1Buffer, vecClc1Buffer, vecInDropBuffer, tmpDropBuffer, this->dropMaskInfo);
    }
    AscendC::PipeBarrier<PIPE_V>();

    if (unlikely(has_sink)) {
        AscendC::PipeBarrier<PIPE_ALL>();
        DataCopy(dyvBuffer, vecClc1Buffer, s1ExtendSubGraph * s2ExtendAlign);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    //
    ///////////////////////////////////////////////////////////////
    // sub to improve
    ///////////////////////////////////////////////////////////////
    uint32_t sub_block_cout = (s2ExtendAlign + cal_repeat_num - 1) / cal_repeat_num;
    LocalTensor<float> sfmgClc3 = unifiedBuffer.GetWithOffset<float>(SFMG_UB_SIZE / sizeof(float), SFMG_UB_OFFSET);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t subIdx = 0; subIdx < sub_block_cout; subIdx++) {
        uint32_t subMaskCout =
            (subIdx == sub_block_cout - 1) ? (s2ExtendAlign - subIdx * cal_repeat_num) : cal_repeat_num;
        Sub(vecClc1Buffer[subIdx * cal_repeat_num], vecClc1Buffer[subIdx * cal_repeat_num], sfmgClc3,
            subMaskCout, s1ExtendSubGraph,
            {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
             static_cast<uint8_t>(s2ExtendAlign / 8), 1});
    }

    ///////////////////////////////////////////////////////////////
    // mul
    ///////////////////////////////////////////////////////////////
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<float> simpleSoftmaxResBuf = unifiedBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), DbBegin);
    Mul(vecClc1Buffer, vecClc1Buffer, simpleSoftmaxResBuf, s1ExtendSubGraph * s2ExtendAlign);
    LocalTensor<T1> vecCopyOutBuffer = vecClc1Buffer.template ReinterpretCast<T1>();
    if constexpr (!IsSameType<T1, float>::value) {
        vecCopyOutBuffer = unifiedBuffer.GetWithOffset<T1>(17 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(vecCopyOutBuffer, vecClc1Buffer, RoundMode::CAST_ROUND, s1ExtendSubGraph * s2ExtendAlign);
    }

    if (unlikely(has_sink)) {
        // SubGrapSink
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t tmpS1Idx = 0; tmpS1Idx < s1ExtendSubGraph; tmpS1Idx++) {
            for (int32_t tmpS2Idx = s2Extend; tmpS2Idx < s2ExtendAlign; tmpS2Idx++) {
                simpleSoftmaxResBuf.SetValue(tmpS1Idx * s2ExtendAlign + tmpS2Idx, static_cast<float>(0.0));
                dyvBuffer.SetValue(tmpS1Idx * s2ExtendAlign + tmpS2Idx, static_cast<float>(0.0));
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        Mul(dyvBuffer, dyvBuffer, simpleSoftmaxResBuf, s1ExtendSubGraph * s2ExtendAlign);
        AscendC::PipeBarrier<PIPE_V>();

        // Simple_softmax for sink
        LocalTensor<float> learnable_sink = unifiedBuffer.GetWithOffset<float>(s1ExtendSubGraph * 8, DbBegin);

        AscendC::PipeBarrier<PIPE_ALL>();
        float getsink = sinkGm.GetValue(dbParam.n2Idx * g + dbParam.gIdx);
        AscendC::PipeBarrier<PIPE_ALL>();

        Duplicate(learnable_sink, static_cast<float> (getsink), s1ExtendSubGraph * 8);
        AscendC::PipeBarrier<PIPE_V>();

        LocalTensor<float> vecInBuffer3 = unifiedBuffer.GetWithOffset<float>(8 * 1024 / sizeof(float), T2BlockBegin);
        int64_t softMaxOffset = 0;
        if constexpr (INPUT_LAYOUT == TND) {
            if(tndSoftmaxIn){
                int64_t innerRowOffsetLeft = unlikely(dbParam.bIdx == 0) ? 0 : ((__gm__ int64_t *)actual_seq_qlen_addr)[dbParam.bIdx - 1] * 32 / sizeof(float);
                int64_t originInnerBatchOffset = ((dbParam.n2Idx * g + dbParam.gIdx) * dbParam.actualS1Len +
                                dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) * 32 / sizeof(float);
                softMaxOffset = ((((__gm__ int64_t *)actual_seq_qlen_addr)[b - 1] * 32 / sizeof(float)) * (dbParam.n2Idx * g + dbParam.gIdx) + innerRowOffsetLeft + originInnerBatchOffset % (dbParam.actualS1Len * 32 / sizeof(float)));
            }else {
                if (dbParam.bIdx > 0) {
                    softMaxOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[dbParam.bIdx - 1] * n2 * g * 32 / sizeof(float);
                }
                softMaxOffset += ((dbParam.n2Idx * g + dbParam.gIdx) * dbParam.actualS1Len +
                                dbParam.s1oIdx * s1CvInner + curS1Idx * s1VecSize) * 32 / sizeof(float);
            }
        } else {
            softMaxOffset = (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1 + dbParam.s1oIdx * s1CvInner +
                            curS1Idx * s1VecSize) * 32 / sizeof(float);
        }
        CopyInSoftMax(vecInBuffer3, s1ExtendSubGraph, softMaxOffset);

        // simple softmax
        AscendC::PipeBarrier<PIPE_V>();
        Sub(learnable_sink, learnable_sink, vecInBuffer3[s1ExtendSubGraph*8], s1ExtendSubGraph*8);
        AscendC::PipeBarrier<PIPE_V>();
        Exp(learnable_sink, learnable_sink, s1ExtendSubGraph*8);
        AscendC::PipeBarrier<PIPE_V>();
        Div(learnable_sink, learnable_sink, vecInBuffer3, s1ExtendSubGraph*8);
        AscendC::PipeBarrier<PIPE_V>();

        for (int i =0; i < s2ExtendAlign / 8; i ++){
            uint8_t dstRepStride = s2ExtendAlign / 8;
            Mul(dyvBuffer[8*i], dyvBuffer[8*i], learnable_sink, 8, s1ExtendSubGraph, {1, 1, 1, dstRepStride, dstRepStride, 1});
            AscendC::PipeBarrier<PIPE_V>();
        }

        // Sum
        LocalTensor<float> localDsink = unifiedBuffer.GetWithOffset<float>(8, DbBegin + 1024);
        Duplicate(localDsink, static_cast<float> (0.0), 8);

        LocalTensor<float> localDsinkSum = unifiedBuffer.GetWithOffset<float>(s1ExtendSubGraph * s2ExtendAlign, DbBegin + 1024 + 8);
        AscendC::ReduceSum<float>(localDsink, dyvBuffer, localDsinkSum,  s1ExtendSubGraph * s2ExtendAlign);
        AscendC::PipeBarrier<PIPE_V>();

        int s1Pad = (TilingData->s1s2BNGS1S2BaseParams.s1 + 255)/256*256;
        int s2Pad = (TilingData->s1s2BNGS1S2BaseParams.s2 + 255)/256*256; 

        int dataSizePerN1 = b * s2Outer * s1Outer;

        AscendC::PipeBarrier<PIPE_ALL>();
        dsinksumDataSizeGm.SetValue(0, dataSizePerN1 * n2 * g);
        AscendC::PipeBarrier<PIPE_ALL>();
        *dsinkSumLocal += localDsink.GetValue(0);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    
    int64_t copyOutOffset = 0;
    DataCopyParams copyOutParam;
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        auto tmpTensor1 = unifiedBuffer.GetWithOffset<T1>(TMP_UB_SIZE / sizeof(T1), TMP_UB_OFFSET);
        AscendC::PipeBarrier<PIPE_V>();
        DataCopy(tmpTensor1, vecCopyOutBuffer, s1ExtendSubGraph * s2ExtendAlign);
        AscendC::PipeBarrier<PIPE_V>();
        ND2NZ(vecCopyOutBuffer, tmpTensor1, s1ExtendSubGraph, s2ExtendAlign);

        copyOutOffset = pingpongIdx * cubeBaseMN * DTYPE_FACTOR + curS1Idx * s1VecSize * C0_SIZE +
                        curS2Idx * dbParam.s1CvExtendAlign * DTYPE_FACTOR * s2VecSize;
        copyOutParam = {
            static_cast<uint16_t>(s2ExtendAlign / C0_SIZE),
            static_cast<uint16_t>(s1ExtendSubGraph * C0_SIZE * sizeof(T1)),
            1,
            static_cast<uint16_t>((dbParam.s1CvExtendAlign * DTYPE_FACTOR - s1ExtendSubGraph) * C0_SIZE * sizeof(T1))
        };
    } else {
        copyOutOffset = pingpongIdx * cubeBaseMN * DTYPE_FACTOR +
                        curS1Idx * s1VecSize * dbParam.s2CvExtendAlign * DTYPE_FACTOR + curS2Idx * s2VecSize;
        copyOutParam = {
            static_cast<uint16_t>(s1ExtendSubGraph),
            static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)),
            0,
            static_cast<uint16_t>((dbParam.s2CvExtendAlign * DTYPE_FACTOR - s2ExtendAlign) * sizeof(T1))
        };
    }
    event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    AscendC::SetFlag<HardEvent::V_MTE3>(static_cast<int32_t>(mte3WaitV));
    AscendC::WaitFlag<HardEvent::V_MTE3>(static_cast<int32_t>(mte3WaitV));

    DataCopyPad(mulWorkSpaceGm[copyOutOffset], vecCopyOutBuffer, copyOutParam);

    if (curIdx < vecLoopEnd - vecLoopStart - 1) {
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3B));
    }

}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::ComputeVec(DBParams& dbParam)
{
    int64_t actualS1Len;
    int64_t actualS2Len;

    s2VecSize = dbParam.s2CvExtend > VEC_S2_LEN ? VEC_S2_LEN : dbParam.s2CvExtend;
    s2VecLoop = s2VecSize == 0 ? 0 : CeilDiv(dbParam.s2CvExtend, s2VecSize);
   if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        if (dbParam.s2CvExtend < VEC_S2_LEN * 2) {
            s2VecSize = AlignUp(CeilDiv(dbParam.s2CvExtend, 2), C0_SIZE);
            s2VecLoop = 2;
        }
        if (dbParam.s2CvExtend <= C0_SIZE) {
            s2VecSize = dbParam.s2CvExtend;
            s2VecLoop = 1;
        }
    }
    uint32_t s2AlignFactor = BLOCK_SIZE / 2;   // float32 also align to 16.
    if constexpr (IS_DROP == ENABLE || IS_ATTEN_MASK == ENABLE) {
        // last dim 32B align
        s2AlignFactor = BLOCK_SIZE / sizeof(uint8_t);
    }

    s1VecSize = baseMN / AlignUp(s2VecSize, s2AlignFactor);
    s1VecSize = s1VecSize > dbParam.s1CvExtend ? dbParam.s1CvExtend : s1VecSize;
    s1VecSize = s1VecSize > 128 ? 128 : s1VecSize;
    s1VecLoop = s1VecSize == 0 ? 0 : CeilDiv(dbParam.s1CvExtend, s1VecSize);
    dropMaskInfo.splitS1BaseSize = s1VecSize;
    if constexpr (INPUT_LAYOUT == TND) {
        UpdateToken(dbParam.bIdx);
        GetSeqQlenKvlenByBidx(dbParam.bIdx, dbParam.actualS1Len, dbParam.actualS2Len);
        dropMaskInfo.s2TotalSize = dbParam.actualS2Len;
        int64_t bSSOffset = 0;
        int64_t s2Accu = 0;
        for (int64_t bidx = 0; bidx < dbParam.bIdx; bidx++) {
            GetSeqQlenKvlenByBidx(bidx, actualS1Len, actualS2Len);
            bSSOffset += actualS1Len * actualS2Len;
            s2Accu += actualS2Len;
        }
        dropMaskInfo.bSSOffset = bSSOffset;
        dropMaskInfo.s1Size = dbParam.actualS1Len;
        dropMaskInfo.s2Size = dbParam.actualS2Len;
        pseInfo.bSSOffset = bSSOffset;
        pseInfo.s2SizeAcc = s2Accu;
        pseInfo.s1Size = dropMaskInfo.s1Size;
        pseInfo.s2Size = dropMaskInfo.s2Size;
    } else {
        dropMaskInfo.s2TotalSize = s2;
        dropMaskInfo.bSSOffset = dbParam.bIdx * s1 * s2;
        pseInfo.s2SizeAcc = dbParam.bIdx * s2;
        pseInfo.bSSOffset = dropMaskInfo.bSSOffset;
    }
    // for compute dropout mask offset
    dropMaskInfo.gOutIdx = dbParam.gIdx;
    dropMaskInfo.n2OutIdx = dbParam.n2Idx;
    dropMaskInfo.s1OutIdx = dbParam.s1oIdx;

    ///////////////////////////////////////////////////////////////
    // SoftmaxGradFront
    ///////////////////////////////////////////////////////////////
    sfmgOffset = 0;
    if constexpr(INPUT_LAYOUT == TND) {
        if (dbParam.bIdx > 0) {
            sfmgOffset = n2 * g * ((__gm__ int64_t *)actual_seq_qlen_addr)[dbParam.bIdx - 1] * 8;
        }
        sfmgOffset += ((dbParam.n2Idx * g + dbParam.gIdx) * dbParam.actualS1Len + dbParam.s1oIdx * s1CvInner) * 8;
    } else {
        sfmgOffset = (((dbParam.bIdx * n2 + dbParam.n2Idx) * g + dbParam.gIdx) * s1 + dbParam.s1oIdx * s1CvInner) * 8;
    }
    int32_t loopSize = s1VecLoop * s2VecLoop;
    int32_t halfLoop = 0;
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        halfLoop = (s2VecLoop / 2) * s1VecLoop;
    } else {
        halfLoop = (s1VecLoop / 2) * s2VecLoop;
    }

    vecLoopStart = cSubIdx ? halfLoop : 0;
    vecLoopEnd = cSubIdx ? loopSize : halfLoop;
    preS1Idx = -1;
    event_t mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3));
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(static_cast<int32_t>(mte2WaitMte3));
    float dsinkSumLocal = 0.0f;
    for (int32_t i = vecLoopStart, loopCnt = 0; i < vecLoopEnd; i++, loopCnt++) {
        int32_t curS1Idx;
        int32_t curS2Idx;
        if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
            curS1Idx = i % s1VecLoop;
            curS2Idx = i / s1VecLoop;
        } else {
            curS1Idx = i / s2VecLoop;
            curS2Idx = i % s2VecLoop;
        }
        s1ExtendSubGraph = (curS1Idx == s1VecLoop - 1) ? (dbParam.s1CvExtend - (s1VecLoop - 1) * s1VecSize) : s1VecSize;
        dropMaskInfo.s1CopySize = s1ExtendSubGraph;
        // for compute dropout mask offset
        dropMaskInfo.s1InnerIdx = curS1Idx;
        // for compute dropout mask
        dropMaskInfo.firstAxis = s1ExtendSubGraph;

        
        event_t mte2WaitMte3A = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        event_t mte2WaitMte3B = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        SubGrapA(loopCnt, curS1Idx, curS2Idx, dbParam, mte2WaitMte3A);
        SubGrapB(loopCnt, s1VecLoop, s2VecLoop, curS1Idx, curS2Idx, dbParam, mte2WaitMte3B, &dsinkSumLocal);

        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(mte2WaitMte3A);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(mte2WaitMte3B);
    }

    // workspace is n1 * b * s1Outer * s2Outer * subcore
    if (unlikely(TilingData->s1s2BNGS1S2BaseParams.sink == 1)) {
        int32_t s1Pad = (TilingData->postTilingData.s1 + 255) / 256 * 256;
        int32_t s2Pad = (TilingData->postTilingData.s2 + 255) / 256 * 256;
        int32_t dataSizePerS1S2 = s1Pad * s2Pad / TilingData->postTilingData.baseMN;
        int32_t dataSizePerN1 = TilingData->postTilingData.b * dataSizePerS1S2;
        int32_t dsinksumLoc = cSubIdx;
        dsinksumLoc += dbParam.s2oIdx * 2;
        dsinksumLoc += dbParam.s1oIdx * s2Pad / 256 * 2;
        dsinksumLoc += (dbParam.gIdx + g * dbParam.n2Idx) * dataSizePerN1 + dbParam.bIdx * dataSizePerS1S2;

        LocalTensor<float> localDsink = unifiedBuffer.GetWithOffset<float>(8, DbBegin + 1024);
        AscendC::PipeBarrier<PIPE_ALL>();
        localDsink.SetValue(0, dsinkSumLocal);
        AscendC::PipeBarrier<PIPE_ALL>();
        DataCopyPad(dsinksumWorkSpaceGm[dsinksumLoc], localDsink, {1, sizeof(float), 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

template <typename FAGT>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2gs1s2SameAB<FAGT>::SyncALLCores()
{
    SyncAll();
}

#endif // _FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2GS1S2_SAMEAB_H_