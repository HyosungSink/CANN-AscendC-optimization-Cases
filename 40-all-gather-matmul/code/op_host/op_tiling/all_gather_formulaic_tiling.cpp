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
 * \file all_gather_formulaic_tiling.cc
 * \brief
 */
#include <iostream>
#include "../../common/inc/mc2_log.h"
#include "all_gather_formulaic_tiling.h"

void AllGatherPlusMM::PrintEstimateKernelTimeResult(double totalMatmulTime, double totalTpTime)
{
    OPS_LOG_D("AllGatherMatmul", "Input shape {M, N, K} = {%lu, %lu, %lu}, cubeUtil_ %f, "
        "totalMatmulTime %f, totalCommTime %f, minTileSize %lu, mAlignLen %lu, commTimeFactor_ %f, "
        "rankDim_ %lu, rankTile %lu",
        clusterInfo_.mValue, clusterInfo_.nValue, clusterInfo_.kValue, matmulPerf_.cubeUtil_,
        totalMatmulTime, totalTpTime, tilingM_.GetMinLen(), tilingM_.GetAlignLength(), commPerf_.commTimeFactor_,
        rankDim_, rankTileNum_);
}


void AllGatherPlusMM::EstimateKernelTime()
{
    SetCommTimeFactor();
    
    // 预测计算、通信任务耗时
    double totalMatmulTime = EstimateTotalMatmulTime();
    double totalTpTime = EstimateTotalCommTime();
    ratioCalcComm_ = (std::max(totalTpTime, totalMatmulTime) / std::min(totalTpTime, totalMatmulTime));
    double frontUtil = matmulPerf_.FindCubeUtilByL2Usage(clusterInfo_.mValue, 1); // Local部分matmul耗时预测
    frontMMTime_ = matmulPerf_.MatmulTime(clusterInfo_.mValue, 1) * matmulPerf_.cubeUtil_ / frontUtil;
    if (totalMatmulTime >= totalTpTime) {
        tilingM_.cutRes.shortTileAtBack = true;
    }

    // 根据通信、计算耗时比例，调整切分约束
    // 2x compute time
    strongTpBound_ = (totalTpTime > totalMatmulTime * 2U) && (clusterInfo_.kValue >= LARGE_K_BOUNDARY);
    // 2x matmulMinTileSize
    bool smallMFlag = (clusterInfo_.mValue < tilingM_.GetMinLen() * 2U) &&
                      (rankTileNum_ > MatmulPerformance::SMALL_RANKTILE);
    bool allowMoreCuts = (!tilingM_.cutRes.shortTileAtBack && smallMFlag) ||
        (strongTpBound_ && clusterInfo_.socType == SocVersion::SOC910_B);
    bool reduceAlignLen = clusterInfo_.nValue > SMALL_N_BOUNDARY && clusterInfo_.mValue <= TINY_M;
    if (allowMoreCuts) {
        if (reduceAlignLen) {
            tilingM_.SetAlignLength(tilingM_.GetAlignLength() / TWO);  // 0.5 is half of mAlignLen
        }
        tilingM_.SetMinLenByMin(tilingM_.GetAlignLength());
    }
    noCutFlag_ = totalTpTime < frontMMTime_ && clusterInfo_.mValue <= tilingM_.tileArgs.maxTileLen;
    if (clusterInfo_.kValue > HUGE_K_BOUNDARY) { // 特大shape切分后matmul性能好一点
        noCutFlag_ = false;
    }
    PrintEstimateKernelTimeResult(totalMatmulTime, totalTpTime);
}

void AllGatherPlusMM::SetCommTimeFactorForA5()
{
    commPerf_.ChangeCommTimeFactorByDivision(ALLGATHERMM_COMMTIME_FACTOR); // 2x time of factor
}

void AllGatherPlusMM::SetCommTimeFactorForOther()
{
    // 通算并行时通信有膨胀，大K大N场景膨胀明显，做特殊处理
    bool medianMFlag = (clusterInfo_.mValue > SMALL_M) && (clusterInfo_.mValue <= MEDIAN_M);
    bool bwGrowthByUtil = matmulPerf_.cubeUtil_ < PART_L2_UTIL;
    bool bwGrowthByShape = clusterInfo_.kValue >= LARGE_K_BOUNDARY && clusterInfo_.nValue > LARGE_N_BOUNDARY;
    bool smallDim = rankDim_ <= MIN_COMM_RANKDIM || rankTileNum_ <= MatmulPerformance::SMALL_RANKTILE;
    bwGrowthByUtil = bwGrowthByUtil && !smallDim;
    bwGrowthByShape = bwGrowthByShape && !smallDim;
    if (bwGrowthByUtil) { // 0.85 is max cube utilizaion rate
        if (!medianMFlag) {
            tilingM_.SetMaxTileCnt(8U); // 8 is max tile cnt of M
        }
        commPerf_.ChangeCommTimeFactorByDivision(gatherLargerNKCommGrowRatio1); // 3x time of factor
    } else if (bwGrowthByShape) {
        if (!medianMFlag) {
            tilingM_.SetMaxTileCnt(8); // 8 is max tile cnt of M
        }
        commPerf_.ChangeCommTimeFactorByDivision(gatherLargerNKCommGrowRatio2); // 1.5x time of factor
    }
    commPerf_.ChangeCommTimeFactorByDivision(commGrowRatio); // 1.15x time of factor
	if (clusterInfo_.socType == SocVersion::SOC910_93) {
		commPerf_.ChangeCommTimeFactorByDivision(0.6);   // 0.6x time of factor
	}
}

void AllGatherPlusMM::SetCommTimeFactor()
{
	if (clusterInfo_.socType == SocVersion::SOC950) {
		SetCommTimeFactorForA5();
	}else{
        SetCommTimeFactorForOther();
    }
}

void AllGatherPlusMM::SelectTilingMethod()
{    
    if (tilingM_.SetShortTileLen(noCutFlag_)) { // 如果shape太小就不切
        return;
    }
    // 流水配平，找到理论切分长度
    if (tilingM_.cutRes.shortTileAtBack) {
        bool smallFront = rankDim_ <= MIN_COMM_RANKDIM; // 2p时，front占了一半的计算量
        if (!smallFront) {
            tilingM_.cutRes.longTileLen = commPerf_.InverseCommTime(frontMMTime_);
        } else {
            ShortAtEndCalcBoundBalancing();
        }
    } else {
        ShortAtFrontCommBoundBalancing();
        if (strongTpBound_) {
            tilingM_.cutRes.longTileLen = std::min(tilingM_.cutRes.longTileLen, tilingM_.cutRes.shortTileLen);
        }
    }

    // 生成切分
    bool smallDimAlignUp = (rankDim_ <= MIN_COMM_RANKDIM) && tilingM_.cutRes.shortTileAtBack &&
        (clusterInfo_.nValue < SMALL_SHAPE_BAR || clusterInfo_.kValue < SMALL_SHAPE_BAR); // 2p，计算Bound，且N轴或者K轴很小
    bool goodLinearityShape = (clusterInfo_.kValue * clusterInfo_.nValue >= LARGE_NK_BAR_BASE * ONE_MBYTE);
    tilingM_.FitTileLengthDiscrete(smallDimAlignUp, goodLinearityShape, hasLocalAtFront_);
    OPS_LOG_D("AllGatherMatmul", "Final cut: shortTileAtBack %d, longTileLen %lu"
        ", numLongTile %lu, shortTileLen %lu, numShortTile %lu",
        tilingM_.cutRes.shortTileAtBack, tilingM_.cutRes.longTileLen, tilingM_.cutRes.numLongTile,
        tilingM_.cutRes.shortTileLen, tilingM_.cutRes.numShortTile);
}