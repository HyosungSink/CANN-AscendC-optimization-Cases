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
 * \file fused_floyd_attention_tiling.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include "kernel_tiling/kernel_tiling.h"
#include "tiling_base/data_copy_transpose_tiling_def.h"


namespace optiling {

BEGIN_TILING_DATA_DEF(BaseParams)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, headNumSize);
TILING_DATA_FIELD_DEF(uint32_t, seqSizeQ);
TILING_DATA_FIELD_DEF(uint32_t, seqSizeK);
TILING_DATA_FIELD_DEF(uint32_t, seqSizeV);
TILING_DATA_FIELD_DEF(uint32_t, seqInnerSize);
TILING_DATA_FIELD_DEF(uint32_t, headSize);
TILING_DATA_FIELD_DEF(float, keepProb);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, preTockens);
TILING_DATA_FIELD_DEF(uint32_t, nextTockens);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskSOuter);
TILING_DATA_FIELD_DEF(uint32_t, seqBaseSize);
TILING_DATA_FIELD_DEF(uint32_t, seqBaseRange);
TILING_DATA_FIELD_DEF(uint32_t, seqBasePadValue);
TILING_DATA_FIELD_DEF(uint32_t, seqSizeAlign);
TILING_DATA_FIELD_DEF(uint32_t, transType);
TILING_DATA_FIELD_DEF(uint32_t, pseShapeType);
TILING_DATA_FIELD_DEF(uint32_t, seqInnerSizeAlign);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskTypeIdx);
TILING_DATA_FIELD_DEF(uint32_t, attenBatchSize);
TILING_DATA_FIELD_DEF(uint32_t, attenHeadNum);
TILING_DATA_FIELD_DEF(uint32_t, isSD9k);
TILING_DATA_FIELD_DEF(uint32_t, remain);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(BaseParamsOp, BaseParams)

BEGIN_TILING_DATA_DEF(AttentionScoreCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, loopBatchNum);
TILING_DATA_FIELD_DEF(uint32_t, loopHeadNum);
TILING_DATA_FIELD_DEF(uint32_t, loopBatchNumTail);
TILING_DATA_FIELD_DEF(uint32_t, loopHeadNumTail);
TILING_DATA_FIELD_DEF(uint32_t, b1);
TILING_DATA_FIELD_DEF(uint32_t, n1);
TILING_DATA_FIELD_DEF(uint32_t, s1);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreBatchSize);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreBatchSizeTail);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreSeqSize);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreHeadNumSize);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreHeadNumSizeTail);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreSeqSizeTail);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreHeadSize);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 48, seqList);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 48, eachCoreSeqSize);
TILING_DATA_FIELD_DEF(uint32_t, formerNum);
TILING_DATA_FIELD_DEF(uint32_t, tailNum);
TILING_DATA_FIELD_DEF(uint32_t, formerHeadNum);
TILING_DATA_FIELD_DEF(uint32_t, tailHeadNum);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreDataSize);
TILING_DATA_FIELD_DEF(uint32_t, bnRange);
TILING_DATA_FIELD_DEF(uint32_t, bnRangeTail);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(AttentionScoreCoreParamsOp, AttentionScoreCoreParams)

BEGIN_TILING_DATA_DEF(AttentionScoreSingleCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, eachCoreBatchNum);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 48, sOuterLoopTimesList);
TILING_DATA_FIELD_DEF(uint32_t, sOuterLoopTimes);
TILING_DATA_FIELD_DEF(uint32_t, sInnerLoopTimes);
TILING_DATA_FIELD_DEF(uint32_t, headRange);
TILING_DATA_FIELD_DEF(uint32_t, singleProcessBatchSize);
TILING_DATA_FIELD_DEF(uint32_t, singleProcessHeadNumSize);
TILING_DATA_FIELD_DEF(uint32_t, singleProcessSOuterSize);
TILING_DATA_FIELD_DEF(uint32_t, singleProcessSOuterSizeTail);
TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSize);
TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSizeTail);
TILING_DATA_FIELD_DEF(uint32_t, singleProcessHeadSize);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreSingleProcessSOuterSize);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreSingleProcessSOuterSizeTail);
TILING_DATA_FIELD_DEF(uint32_t, lastCoreSingleProcessSOuterSize);
TILING_DATA_FIELD_DEF(uint32_t, lastCoreSingleProcessSOuterSizeTail);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreSOuterLoopTimes);
TILING_DATA_FIELD_DEF(uint32_t, lastCoreSOuterLoopTimes);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreSingleProcessSInnerSize);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreSInnerLoopTimes);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreSingleProcessSInnerSizeTail);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(AttentionScoreSingleCoreParamsOp, AttentionScoreSingleCoreParams)

BEGIN_TILING_DATA_DEF(AttentionScoreSingleCoreTensorSize)
TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, paddingMaskUbSize);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskUbSize);
TILING_DATA_FIELD_DEF(uint32_t, pseUbSize);
TILING_DATA_FIELD_DEF(uint32_t, maskSize);
TILING_DATA_FIELD_DEF(uint32_t, softmaxMaxSize);
TILING_DATA_FIELD_DEF(uint32_t, softmaxSumSize);
TILING_DATA_FIELD_DEF(uint32_t, softmaxExpSize);
TILING_DATA_FIELD_DEF(uint32_t, spmTmpSize);
TILING_DATA_FIELD_DEF(uint32_t, scmTmpSize);
TILING_DATA_FIELD_DEF(uint32_t, apiTmpUbSize);
TILING_DATA_FIELD_DEF(uint32_t, apiTmpUbSizeTail);
TILING_DATA_FIELD_DEF(uint32_t, mmResInUbSize);
TILING_DATA_FIELD_DEF(uint32_t, mmResInUbSizeTail);
TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, tmpMMResBmm2PreUbSize);
TILING_DATA_FIELD_DEF(uint32_t, tmpSoftmaxBmm2UbSize);
TILING_DATA_FIELD_DEF(uint32_t, mmResUbSizeTailLoop);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskUbSizeTailLoop);
TILING_DATA_FIELD_DEF(uint32_t, maskSizeTailLoop);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreMMResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, tailCorePaddingMaskUbSize);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreAttenMaskUbSize);
TILING_DATA_FIELD_DEF(uint32_t, tailCorePseUbSize);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreMaskSize);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreSoftmaxMaxSize);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreSoftmaxSumSize);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreSoftmaxExpSize);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreSpmTmpSize);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreScmTmpSize);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreBmm2ResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreTmpMMResBmm2PreUbSize);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreTmpSoftmaxBmm2UbSize);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreMMResUbSizeTailLoop);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreAttenMaskUbSizeTailLoop);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreMaskSizeTailLoop);
TILING_DATA_FIELD_DEF(uint32_t, softmaxTmpBuffer);
TILING_DATA_FIELD_DEF(uint32_t, dropOutTmpBuffer);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(AttentionScoreSingleCoreTensorSizeOp, AttentionScoreSingleCoreTensorSize)

BEGIN_TILING_DATA_DEF(AttentionScoreOffestStrideParams)
TILING_DATA_FIELD_DEF(uint32_t, typeByteNum);
TILING_DATA_FIELD_DEF(uint32_t, matmulHead);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(AttentionScoreOffestStrideParamsOp, AttentionScoreOffestStrideParams)

BEGIN_TILING_DATA_DEF(FusedFloydAttentionEmptyInputTilingData)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, attentionOutFormerNum);
TILING_DATA_FIELD_DEF(uint32_t, attentionOutTailNum);
TILING_DATA_FIELD_DEF(uint32_t, softmaxMaxFormerNum);
TILING_DATA_FIELD_DEF(uint32_t, softmaxMaxTailNum);
TILING_DATA_FIELD_DEF(uint32_t, reserved);
TILING_DATA_FIELD_DEF(uint64_t, attentionOutSingleCoreDataSize);
TILING_DATA_FIELD_DEF(uint64_t, attentionOutTailCoreDataSize);
TILING_DATA_FIELD_DEF(uint64_t, softmaxMaxSingleCoreDataSize);
TILING_DATA_FIELD_DEF(uint64_t, softmaxMaxTailCoreDataSize);
TILING_DATA_FIELD_DEF(uint64_t, attentionOutLastCoreDataSize);
TILING_DATA_FIELD_DEF(uint64_t, attentionOutLastCoreIndex);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FusedFloydAttentionEmptyInputTilingDataOp, FusedFloydAttentionEmptyInputTilingData)

BEGIN_TILING_DATA_DEF(InputParams)
TILING_DATA_FIELD_DEF(int64_t, bSize);
TILING_DATA_FIELD_DEF(int64_t, n2Size);
TILING_DATA_FIELD_DEF(int64_t, gSize);
TILING_DATA_FIELD_DEF(int64_t, s1Size);
TILING_DATA_FIELD_DEF(int64_t, s2Size);
TILING_DATA_FIELD_DEF(int64_t, alignedS2);
TILING_DATA_FIELD_DEF(int64_t, dSize);
TILING_DATA_FIELD_DEF(int64_t, attenbSize);
TILING_DATA_FIELD_DEF(float, keepProb);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(int64_t, preTokens);
TILING_DATA_FIELD_DEF(int64_t, nextTokens);
// in pse encoding scenes, s1 and s2 might not equal with s1, s2 in Q, K
TILING_DATA_FIELD_DEF(int64_t, pseS1Size);
TILING_DATA_FIELD_DEF(int64_t, pseS2Size);
TILING_DATA_FIELD_DEF(uint32_t, pseBSize);
TILING_DATA_FIELD_DEF(uint32_t, bandIndex);

// 1: BSH/BSND, 2: SBH, 3: BNSD
TILING_DATA_FIELD_DEF(uint8_t, layoutType);
// 0: (B,N2,G,S1,S2), 1: (B,N2,G,1,S2)
TILING_DATA_FIELD_DEF(uint8_t, pseShapeType);
// 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
TILING_DATA_FIELD_DEF(uint8_t, attenMaskShapeType);
// 0: fp16, 1: bool(uint8)
TILING_DATA_FIELD_DEF(uint8_t, attenMaskDataType);
// ALL: 0, NONE: 1, ANY: 2, CAUSAL: 3, BAND: 4 };
TILING_DATA_FIELD_DEF(uint8_t, attenMaskCompressMode);
// 0: high precise, 1: high performance, 2: invalid line high precise
TILING_DATA_FIELD_DEF(uint8_t, implMode);
TILING_DATA_FIELD_DEF(uint8_t, sparseType);
TILING_DATA_FIELD_DEF(uint8_t, needDropMaskOp);
TILING_DATA_FIELD_DEF(uint8_t, pseEncodeType);
TILING_DATA_FIELD_DEF(uint8_t, rsv);
TILING_DATA_FIELD_DEF(uint16_t, remain);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskS2Size);
TILING_DATA_FIELD_DEF(uint32_t, pseType);
TILING_DATA_FIELD_DEF(uint32_t, rsv1);
TILING_DATA_FIELD_DEF(int64_t, qStartIdx);
TILING_DATA_FIELD_DEF(int64_t, kvStartIdx);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(InputParamsOp, InputParams)

BEGIN_TILING_DATA_DEF(MultiCoreParams)
TILING_DATA_FIELD_DEF(int32_t, coreNum);
TILING_DATA_FIELD_DEF(int32_t, reserve);
// BN2GS1.o
TILING_DATA_FIELD_DEF(int64_t, totalSize);
// BN2GS1.o / core_num
TILING_DATA_FIELD_DEF(int64_t, splitFactorSize);
TILING_DATA_FIELD_DEF(int64_t, splitFactorTailSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, 48, sparseStartIdx);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MultiCoreParamsOp, MultiCoreParams)

BEGIN_TILING_DATA_DEF(CoreParams)
TILING_DATA_FIELD_DEF(int32_t, s1BaseSize);
TILING_DATA_FIELD_DEF(int32_t, s1BaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, s1OuterSize);
TILING_DATA_FIELD_DEF(int32_t, s1Vec2BaseSize);
TILING_DATA_FIELD_DEF(int32_t, s1Vec2BaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, s1Vec2OuterSize);
TILING_DATA_FIELD_DEF(int32_t, s2BaseSize);
TILING_DATA_FIELD_DEF(int32_t, s2BaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, s2OuterSize);
TILING_DATA_FIELD_DEF(int32_t, dBaseSize);
TILING_DATA_FIELD_DEF(int32_t, dBaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, dOuterSize);
TILING_DATA_FIELD_DEF(int32_t, bBaseSize);
TILING_DATA_FIELD_DEF(int32_t, bBaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, bOuterSize);
TILING_DATA_FIELD_DEF(int32_t, n2BaseSize);
TILING_DATA_FIELD_DEF(int32_t, bmm1Num);
TILING_DATA_FIELD_DEF(int32_t, bmmv2Num);
TILING_DATA_FIELD_DEF(int32_t, n2BaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, n2OuterSize);
TILING_DATA_FIELD_DEF(int32_t, gBaseSize);
TILING_DATA_FIELD_DEF(int32_t, gBaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, gOuterSize);
TILING_DATA_FIELD_DEF(int32_t, nRatio);
TILING_DATA_FIELD_DEF(int32_t, rsvd);
TILING_DATA_FIELD_DEF(int64_t, s1SparseValidSize);
TILING_DATA_FIELD_DEF(int64_t, s2SparseValidSize);
TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS1);
TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS2);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CoreParamsOp, CoreParams)

BEGIN_TILING_DATA_DEF(TensorSizeParams)
TILING_DATA_FIELD_DEF(int32_t, bmm1ResUbSize);
TILING_DATA_FIELD_DEF(int32_t, attenMaskUbSize);
TILING_DATA_FIELD_DEF(int32_t, pseUbSize);
TILING_DATA_FIELD_DEF(int32_t, dropMaskUbSize);
TILING_DATA_FIELD_DEF(int32_t, castUbSize);
TILING_DATA_FIELD_DEF(int32_t, softmaxMaxUbSize);
TILING_DATA_FIELD_DEF(int32_t, softmaxSumUbSize);
TILING_DATA_FIELD_DEF(int32_t, softmaxExpUbSize);
TILING_DATA_FIELD_DEF(int32_t, apiTmpBufferBytes);
TILING_DATA_FIELD_DEF(int32_t, bmm2ResUbSize);
TILING_DATA_FIELD_DEF(int32_t, inputQueBytes);
TILING_DATA_FIELD_DEF(int32_t, outputQueBytes);
// API buffer use remain space of ub
TILING_DATA_FIELD_DEF(int32_t, tmpBufBytes);
TILING_DATA_FIELD_DEF(int32_t, softmaxMaxOffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, softmaxSumOffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, maxSumApiOffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, customSoftmaxApiOffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, pseTbufOffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, dropoutApiOffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, maxSumApiSize);
TILING_DATA_FIELD_DEF(int32_t, customSoftmaxApiSize);
TILING_DATA_FIELD_DEF(int32_t, dropoutApiSize);
TILING_DATA_FIELD_DEF(int32_t, attenMaskApiSize);
TILING_DATA_FIELD_DEF(int32_t, attenMaskApiOffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, bmm1ProcessTInStage2Size);
TILING_DATA_FIELD_DEF(int32_t, bmm1ProcessTInStage2OffsetBytes);
// workspace
TILING_DATA_FIELD_DEF(int32_t, wkspSection1OffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, wkspSection2OffsetBytes);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(TensorSizeParamsOp, TensorSizeParams)

BEGIN_TILING_DATA_DEF(MaxSumTiling)
TILING_DATA_FIELD_DEF(uint32_t, srcM);
TILING_DATA_FIELD_DEF(uint32_t, srcK);
TILING_DATA_FIELD_DEF(uint32_t, srcSize);
TILING_DATA_FIELD_DEF(uint32_t, outMaxM);
TILING_DATA_FIELD_DEF(uint32_t, outMaxK);
TILING_DATA_FIELD_DEF(uint32_t, outMaxSize);
TILING_DATA_FIELD_DEF(uint32_t, splitM);
TILING_DATA_FIELD_DEF(uint32_t, splitK);
TILING_DATA_FIELD_DEF(uint32_t, splitSize);
TILING_DATA_FIELD_DEF(uint32_t, reduceM);
TILING_DATA_FIELD_DEF(uint32_t, reduceK);
TILING_DATA_FIELD_DEF(uint32_t, reduceSize);
TILING_DATA_FIELD_DEF(uint32_t, rangeM);
TILING_DATA_FIELD_DEF(uint32_t, tailM);
TILING_DATA_FIELD_DEF(uint32_t, tailSplitSize);
TILING_DATA_FIELD_DEF(uint32_t, tailReduceSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MaxSumTilingOp, MaxSumTiling)

BEGIN_TILING_DATA_DEF(DropmaskParams)
TILING_DATA_FIELD_DEF(int32_t, multiCoreFactorSize);
TILING_DATA_FIELD_DEF(int32_t, baseUbCalSize);
TILING_DATA_FIELD_DEF(int64_t, multiCoreTotalSize);
TILING_DATA_FIELD_DEF(int64_t, shapeTotalSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DropmaskParamsOp, DropmaskParams)

BEGIN_TILING_DATA_DEF(FusedFloydAttentionTilingData)
TILING_DATA_FIELD_DEF_STRUCT(BaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(AttentionScoreCoreParams, attentionScoreCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(AttentionScoreSingleCoreParams, attentionScoreSingleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(AttentionScoreSingleCoreTensorSize, attentionScoreSingleCoreTensorSize);
TILING_DATA_FIELD_DEF_STRUCT(AttentionScoreOffestStrideParams, attentionScoreOffestStrideParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingDataTailCore);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingDataTailCore);

TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxFlashTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingDataTailCore);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxFlashTilingDataTailCore);
TILING_DATA_FIELD_DEF_STRUCT(CopyTransposeTiling, transposeTilingData);
TILING_DATA_FIELD_DEF_STRUCT(CopyTransposeTiling, transposeTilingDataTailCore);
TILING_DATA_FIELD_DEF_STRUCT(FusedFloydAttentionEmptyInputTilingData, emptyInputTilingData);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(FusedFloydAttentionGeneralTilingData)
TILING_DATA_FIELD_DEF_STRUCT(InputParams, inputParams);
TILING_DATA_FIELD_DEF_STRUCT(MultiCoreParams, multiCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(CoreParams, coreParams);
TILING_DATA_FIELD_DEF_STRUCT(TensorSizeParams, tensorSizeParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1k1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2v2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxFlashTilingData);
TILING_DATA_FIELD_DEF_STRUCT(CopyTransposeTiling, transposeTilingData);
TILING_DATA_FIELD_DEF_STRUCT(CopyTransposeTiling, transposeTilingDataTailCore);
TILING_DATA_FIELD_DEF_STRUCT(DropmaskParams, dropmaskParams);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedFloydAttention_90, FusedFloydAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedFloydAttention_92, FusedFloydAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedFloydAttention_94, FusedFloydAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedFloydAttention, FusedFloydAttentionGeneralTilingData)

} // namespace optiling