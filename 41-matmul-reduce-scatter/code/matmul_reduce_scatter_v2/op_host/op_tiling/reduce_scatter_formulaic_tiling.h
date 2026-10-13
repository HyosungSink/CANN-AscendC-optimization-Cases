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
 * \file reduce_scatter_formulaic_tiling.h
 * \brief
 */
#ifndef __REDUCE_SCATTER_FORMULAIC_TILING_H__
#define __REDUCE_SCATTER_FORMULAIC_TILING_H__

#pragma once
#include "../../../common/inc/tiling/hccl_formulaic_tiling.h"
constexpr double TIME_LOWER_RATIO = 2.0;
constexpr double TIME_UPPER_RATIO = 3.0;
constexpr uint64_t LARGE_K_BOUNDARY = 8192;
constexpr uint64_t LARGE_N_BOUNDARY = 5120;
constexpr double scatterLargerNKCommGrowRatio1 = 1.5;
constexpr double scatterLargerNKCommGrowRatio2 = 1.2;
constexpr uint64_t ONE_KILO = 1024;
constexpr double UNBALANCE_RATIO = 1.5;

class MMPlusReduceScatter : public OneCalcOneCommBase {
    public:
        bool deterministicSoc910B_ = false; // 通信是否使用local reduce确定性算法
        // Constructor
        explicit MMPlusReduceScatter(const mc2tiling::TilingArgs& args, uint32_t inputRankDim,
            KernelType inputKernelType, SocVersion inputSocVersion = SocVersion::SOC910_B,
            bool deterministicFlag = false)
            : OneCalcOneCommBase(args, inputRankDim, inputKernelType, inputSocVersion),
            deterministicSoc910B_(deterministicFlag)
        {
            commPerf_.SetCommShapeLen(clusterInfo_.nValue);
            commPerf_.SetCommDTypeSize(clusterInfo_.outMatrixCDtypeSize);
            if (deterministicSoc910B_) { // 通信拟合适配确定性算法
                commPerf_.SetLocalReduceFactor();
            }
            rankTileNum_ = commPerf_.GetRankTileNum();
            tilingM_.SetMinLenByMax(commPerf_.GetLinearThresholdLen());
            tilingM_.SetMinLenByMax(matmulPerf_.GetLinearThresholdLen(rankTileNum_));
        }

        void EstimateKernelTime() override;
        void SelectTilingMethod() override;
    private:
        void SetCommTimeFactorForA5();
        void SetCommTimeFactorForOther();
        void SetCommTimeFactor();
};


#endif //__REDUCE_SCATTER_FORMULAIC_TILING_H__
