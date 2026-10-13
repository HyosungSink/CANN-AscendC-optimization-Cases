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
 * \file rms_norm_grad_tiling.h
 * \brief RmsNormGrad tiling file
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_RMS_NORM_GRAD_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_RMS_NORM_GRAD_H

#include "../common/stub/op_api/level0/log.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../common/inc/op_host/tiling_base.h"
#include "../common/inc/op_host/tiling_templates_registry.h"
#include "../common/inc/error_util.h"
#include "util/math_util.h"
#include "platform/platform_infos_def.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(RmsNormGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, row);
TILING_DATA_FIELD_DEF(uint32_t, col);
TILING_DATA_FIELD_DEF(float, avg_factor);
TILING_DATA_FIELD_DEF(uint32_t, data_type);
TILING_DATA_FIELD_DEF(uint32_t, block_factor);
TILING_DATA_FIELD_DEF(uint32_t, ub_split_dim);
TILING_DATA_FIELD_DEF(uint32_t, ub_factor);
TILING_DATA_FIELD_DEF(uint32_t, core_calc_num);
TILING_DATA_FIELD_DEF(uint32_t, core_calc_tail);
TILING_DATA_FIELD_DEF(uint32_t, block_dim);
TILING_DATA_FIELD_DEF(uint32_t, ub_calc_num);
TILING_DATA_FIELD_DEF(uint32_t, ub_calc_tail);
TILING_DATA_FIELD_DEF(uint32_t, ub_calc_loop);
TILING_DATA_FIELD_DEF(uint32_t, ub_calc_tail_num);
TILING_DATA_FIELD_DEF(uint32_t, ub_calc_tail_tail);
TILING_DATA_FIELD_DEF(uint32_t, ub_calc_tail_loop);
TILING_DATA_FIELD_DEF(uint32_t, fixed_output);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNormGrad, RmsNormGradTilingData);

BEGIN_TILING_DATA_DEF(RmsNormGradRegbaseDxTilingData)
TILING_DATA_FIELD_DEF(int64_t, rows);
TILING_DATA_FIELD_DEF(int64_t, cols);
TILING_DATA_FIELD_DEF(int64_t, blockFactorDx);
TILING_DATA_FIELD_DEF(int64_t, bodyPart);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNumDx);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNormGradRegbaseDxTilingDataOp, RmsNormGradRegbaseDxTilingData);

BEGIN_TILING_DATA_DEF(RmsNormGradRegbaseTilingData)
TILING_DATA_FIELD_DEF_STRUCT(RmsNormGradRegbaseDxTilingData, dxTilingData);
TILING_DATA_FIELD_DEF(int64_t, tailCoreNumDG);
TILING_DATA_FIELD_DEF(int64_t, colsPerCoreDG);
TILING_DATA_FIELD_DEF(int64_t, colsLastCoreDG);
TILING_DATA_FIELD_DEF(int64_t, colsPerTailCoreDG);
TILING_DATA_FIELD_DEF(int64_t, rowsPerUBDG);
TILING_DATA_FIELD_DEF(int32_t, powerofTwoValueDG);
TILING_DATA_FIELD_DEF(int32_t, rowsTailDG);
TILING_DATA_FIELD_DEF(int32_t, totalBlockCountDG);
TILING_DATA_FIELD_DEF(int32_t, mainBlockCountDG);
TILING_DATA_FIELD_DEF(int32_t, tailBlockCountwithPadDG);
TILING_DATA_FIELD_DEF(int32_t, powerOfTwoBlockCountDG);
TILING_DATA_FIELD_DEF(int32_t, tailBlockCountWithoutPadDG);
TILING_DATA_FIELD_DEF(int32_t, binaryAddKDG);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNumDG);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, vlFp32);
TILING_DATA_FIELD_DEF(uint32_t, isFullLoad);
TILING_DATA_FIELD_DEF(uint32_t, isMultiColset);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNormGrad_7000, RmsNormGradRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(RmsNormGrad_7001, RmsNormGradRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(RmsNormGrad_7010, RmsNormGradRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(RmsNormGrad_7011, RmsNormGradRegbaseTilingData);


BEGIN_TILING_DATA_DEF(RmsNormGradBigMTilingData)
TILING_DATA_FIELD_DEF_STRUCT(RmsNormGradRegbaseDxTilingData, dxTilingData);
TILING_DATA_FIELD_DEF(int64_t, dgammaUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, dgammaMPerBlock);
TILING_DATA_FIELD_DEF(int64_t, dgammaMReminder);
TILING_DATA_FIELD_DEF(int64_t, dgammaNloop);
TILING_DATA_FIELD_DEF(int64_t, dgammaNtail);
TILING_DATA_FIELD_DEF(int64_t, dgammaMfactorBlockAligned);
TILING_DATA_FIELD_DEF(int64_t, dgammaNfactorBlockAligned);
TILING_DATA_FIELD_DEF(int64_t, dgammaMToProcessMainBlock);
TILING_DATA_FIELD_DEF(int64_t, dgammaMLoopMainBlock);
TILING_DATA_FIELD_DEF(int64_t, dgammaMTotalLoopMainBlock);
TILING_DATA_FIELD_DEF(int64_t, dgammaMTailMainBlock);
TILING_DATA_FIELD_DEF(int64_t, dgammaBasicBlockLoopMainBlock);
TILING_DATA_FIELD_DEF(int64_t, dgammaMainFoldCountMainBlock);
TILING_DATA_FIELD_DEF(int64_t, dgammaCacheBufferCountMainBlock);
TILING_DATA_FIELD_DEF(int64_t, dgammaResultCacheIDMainBlock);
TILING_DATA_FIELD_DEF(int64_t, dgammaMToProcessTailBlock);
TILING_DATA_FIELD_DEF(int64_t, dgammaMLoopTailBlock);
TILING_DATA_FIELD_DEF(int64_t, dgammaMTotalLoopTailBlock);
TILING_DATA_FIELD_DEF(int64_t, dgammaMTailTailBlock);
TILING_DATA_FIELD_DEF(int64_t, dgammaBasicBlockLoopTailBlock);
TILING_DATA_FIELD_DEF(int64_t, dgammaMainFoldCountTailBlock);
TILING_DATA_FIELD_DEF(int64_t, dgammaCacheBufferCountTailBlock);
TILING_DATA_FIELD_DEF(int64_t, dgammaResultCacheIDTailBlock);
TILING_DATA_FIELD_DEF(int64_t, dgammaAInnerAlignedStg1);
TILING_DATA_FIELD_DEF(int64_t, dgammaAOuterStg1);
TILING_DATA_FIELD_DEF(int64_t, dgammaATailStg1);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNormGrad_9000, RmsNormGradBigMTilingData);
REGISTER_TILING_DATA_CLASS(RmsNormGrad_9010, RmsNormGradBigMTilingData);

BEGIN_TILING_DATA_DEF(RmsNormGradEmptyTilingData)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNumDG);
TILING_DATA_FIELD_DEF(uint64_t, colsPerCoreDG);
TILING_DATA_FIELD_DEF(uint64_t, cols);
TILING_DATA_FIELD_DEF(uint32_t, ubSize);
TILING_DATA_FIELD_DEF(uint64_t, colsPerUBDG);
TILING_DATA_FIELD_DEF(uint64_t, coreUbBlockCount);
TILING_DATA_FIELD_DEF(uint64_t, tailUbCols);
TILING_DATA_FIELD_DEF(uint64_t, lastCoreBlockCount);
TILING_DATA_FIELD_DEF(uint64_t, lastCoreTailUbCols);
TILING_DATA_FIELD_DEF(uint64_t, colsLastCoreDG);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RmsNormGrad_8000, RmsNormGradEmptyTilingData);

class RmsNormGradRegbaseTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit RmsNormGradRegbaseTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus CalcTilingDataDx();

protected:
    uint64_t ubSize_{0};
    uint32_t blockSize_{0};
    uint32_t vecRegSize_{0};
    uint32_t vlFp32_{0};
    uint32_t aivCoreNum_{0};
    uint32_t usedCoreNumDx_{0};
    int64_t rows_{0}; // A轴
    int64_t cols_{0}; // R轴
    int64_t blockFactorDx_{0};
    int64_t bodyPart_{0};
    int64_t colsPerCore_{0};
    int64_t rowsPerCore_{0};
    // params for dgamma
    int64_t usedCoreNumDG_{0};
    int64_t colsPerCoreDG_{0};
    int64_t colsPerTailCoreDG_{0};
    int64_t binaryAddNumDG_{0};
    int64_t colsPerLoopDG_{0};
    int64_t rowsPerUBDG_{0};
    int64_t cols_sets_{0};
    int64_t colsPerUBDG_{0};
    uint32_t isFullLoadDG_{0};
    uint32_t isMultiColset_{0};
    int64_t colsLastCoreDG_{0};
    int64_t isPowerofTwoDG_{0};
    int32_t powerofTwoValueDG_{0};
    int32_t rowsTailDG_{0};
    int32_t maxRowsNumDG_{0};
    int32_t totalBlockCountDG_{0};
    int32_t mainBlockCountDG_{0};
    int32_t tailBlockCountwithPadDG_{0};
    int32_t powerOfTwoBlockCountDG_{0};
    int32_t tailBlockCountWithoutPadDG_{0};
    int32_t binaryAddKDG_{0};
    int64_t tailCoreNumDG_{0};

    ge::DataType dyDtype_{ge::DataType::DT_FLOAT};
    uint32_t tilingKey_{0};
    RmsNormGradRegbaseTilingData tilingData_;

    ge::graphStatus CheckShapeAllPositive(gert::Shape& shape);
    ge::graphStatus CheckInputsShape();
    ge::graphStatus CheckInputsDtypeAndFormat();
    ge::graphStatus CheckShapesEqual(gert::Shape& shape0, gert::Shape& shape1);
    void CalcRowsAndCols(gert::Shape& xShape, gert::Shape& gammaShape);
    void CalcUsedCoreNumGamma();
    ge::graphStatus CalcUbBufferSizeDgamma();
    ge::graphStatus CalcTilingDataDgamma();
    int32_t CalcRowsTails();
    int64_t GetSizeOfBlockAlign(int64_t nonAlignSize);
    int32_t NearestLowerPowerOfTwo(int32_t temp);
    void LogTilingResult();
};

class RmsNormGradEmptyTiling : public Ops::NN::Optiling::TilingBaseClass  {
public:
    explicit RmsNormGradEmptyTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {}

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;

    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoLibApiTiling() override;
    const gert::Shape& EnsureNotScalar(const gert::Shape& inShape);
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    uint32_t aivCoreNum_{0};
    uint64_t rows_{0};
    uint64_t cols_{0};
    uint64_t usedCoreNumDG_{0};
    uint64_t colsPerCoreDG_{0};
    uint64_t colsPerUBDG_{0};
    uint64_t tailUbCols_{0};
    uint64_t lastCoreBlockCount_{0};
    uint64_t lastCoreTailUbCols_{0};
    uint64_t coreUbBlockCount_{0};
    uint64_t colsLastCoreDG_{0};
    uint64_t ubSize_{0};

    uint32_t tilingKey_{0};
    RmsNormGradEmptyTilingData tilingData_;

    ge::graphStatus CheckShapeAllPositive(gert::Shape& shape);
    ge::graphStatus CheckInputsShape();
    ge::graphStatus CheckInputsDtypeAndFormat();
    ge::graphStatus CheckShapesEqual(gert::Shape& shape0, gert::Shape& shape1);
    void CalcRowsAndCols(gert::Shape& gammaShape);
    ge::graphStatus CalcTilingDataDgamma();
    int32_t NearestLowerPowerOfTwo(int32_t temp);
    ge::graphStatus CalcUsedCoreNumGamma();
    void LogTilingResult();
};

class RmsNormGradBigMTiling : public RmsNormGradRegbaseTiling
{
public:
    explicit RmsNormGradBigMTiling(gert::TilingContext* context) : RmsNormGradRegbaseTiling(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    RmsNormGradBigMTilingData tilingData_;

    ge::graphStatus DgammaDoTiling();
    ge::graphStatus DgammaDoTilingStg0();
    ge::graphStatus DgammaDoTilingStg1();

    int64_t GetCacheID(const int64_t idx);
    int64_t FindNearestPower2(const int64_t value);

    int64_t usedCoreNumDgamma_{0};
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_RMS_NORM_GRAD_H
