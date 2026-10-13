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
 * \file add_layer_norm_tiling.h
 * \brief
 */
#ifndef ADD_LAYER_NORM_TILING_H
#define ADD_LAYER_NORM_TILING_H

#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "op_host/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_templates_registry.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(AddLayerNormTilingData)
TILING_DATA_FIELD_DEF(int64_t, numCore);
TILING_DATA_FIELD_DEF(int64_t, numLastDim);
TILING_DATA_FIELD_DEF(int64_t, numFirstDim);
TILING_DATA_FIELD_DEF(int64_t, firstDimPerCore);
TILING_DATA_FIELD_DEF(int64_t, firstDimPerCoreTail);
TILING_DATA_FIELD_DEF(int64_t, firstDimPerTime);
TILING_DATA_FIELD_DEF(int64_t, lastDimPerTime);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, aveFactor);
TILING_DATA_FIELD_DEF(int64_t, colMoveCnt);
TILING_DATA_FIELD_DEF(int64_t, colTail);
TILING_DATA_FIELD_DEF(int64_t, workspaceSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddLayerNorm, AddLayerNormTilingData);
REGISTER_TILING_DATA_CLASS(InplaceAddLayerNorm, AddLayerNormTilingData);

BEGIN_TILING_DATA_DEF(AddLayerNormRegbaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, blockSize);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, vlFp32);
TILING_DATA_FIELD_DEF(int64_t, tailCoreStartIndex);
TILING_DATA_FIELD_DEF(int64_t, rowsPerCore);
TILING_DATA_FIELD_DEF(int64_t, rowsPerTailCore);
TILING_DATA_FIELD_DEF(int64_t, rowsPerLoop);
TILING_DATA_FIELD_DEF(int64_t, cols);
TILING_DATA_FIELD_DEF(int64_t, colsPerLoop);
TILING_DATA_FIELD_DEF(int64_t, colsLoopCount);
TILING_DATA_FIELD_DEF(int64_t, colsTail);
TILING_DATA_FIELD_DEF(int64_t, binaryAddNum);
TILING_DATA_FIELD_DEF(int64_t, binaryAddK);
TILING_DATA_FIELD_DEF(int64_t, binaryAddLastNum);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(int64_t, outputX);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddLayerNorm_8000, AddLayerNormRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNorm_8001, AddLayerNormRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNorm_8002, AddLayerNormRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNorm_8100, AddLayerNormRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNorm_8101, AddLayerNormRegbaseTilingData);
REGISTER_TILING_DATA_CLASS(AddLayerNorm_8102, AddLayerNormRegbaseTilingData);

enum class BIAS {
    BIAS_NONE,
    BIAS_ELEWISE,
    BIAS_BRC
};

class AddLayerNormRegbaseTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit AddLayerNormRegbaseTiling(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
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

private:
    uint64_t ubSize_;
    int64_t blockSize_;
    int64_t vecRegSize_;
    int64_t vlFp32_;
    int64_t aivCoreNum_;
    int64_t tailCoreNum_;
    int64_t usedCoreNum_;
    int64_t tailCoreStartIndex_;
    int64_t rows_; // A轴
    int64_t cols_; // R轴
    int64_t rowsPerCore_;
    int64_t rowsPerTailCore_;
    int64_t rowsPerLoop_;
    int64_t colsPerLoop_;
    int64_t colsLoopCount_;
    int64_t colsTail_;
    int64_t binaryAddNum_;
    int64_t binaryAddK_;
    int64_t binaryAddLastNum_;
    float epsilon_;
    bool needOutputX_;
    bool isWelford_;
    bool isMix_;
    ge::DataType dataType_;
    BIAS biasType_;
    uint32_t tilingKey_;
    AddLayerNormRegbaseTilingData tilingData_;
    ge::graphStatus CheckDimNum(gert::Shape& shape) const;
    ge::graphStatus CheckShapeAllPositive(gert::Shape& shape);
    ge::graphStatus CheckInputsShape();
    ge::graphStatus CheckOutputsShape();
    ge::graphStatus CheckInputsDtype();
    ge::graphStatus CheckOutputsDtype() const;
    ge::graphStatus CheckShapesEqual(gert::Shape& shape0, gert::Shape& shape1);
    ge::graphStatus CalcRowsAndCols(gert::Shape& shapeX, gert::Shape& shapeGamma);
    ge::graphStatus BiasShapeProcess(gert::Shape& shapeX, gert::Shape& shapeGamma, gert::Shape& shapeBias);
    ge::graphStatus MeanRstdShapeProcess(
        gert::Shape& shapeX, gert::Shape& shapeGamma, gert::Shape& shapeMeanRstd) const;

    void CalcUsedCoreNum();
    ge::graphStatus CalcUbBufferSize();
    int64_t GetSizeOfBlockAlign(int64_t nonAlignSize);
    void LogTilingResult();
    const gert::Shape& EnsureNotScalar(const gert::Shape& inShape);
};

struct AddLayerNormCompileInfo {
    int64_t aivCoreNum_ = 0;
    int64_t sysWorkspaceSize_ = 0;
    uint64_t ubSize_ = 0;
    int64_t vecRegSize_ = 0;
    int64_t blockSize_ = 0;
    bool isAscend950_ = false;
};
} // namespace optiling

#endif // ADD_LAYER_NORM_TILING_H