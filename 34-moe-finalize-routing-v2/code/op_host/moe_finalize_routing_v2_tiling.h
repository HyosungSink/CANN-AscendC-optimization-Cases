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
 * \file moe_finalize_routing_v2_tiling.h
 * \brief
 */
#pragma once
#include <cstdint>
#include <limits>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "../common/include/tiling_base/tiling_base.h"
#include "../common/include/tiling_base/tiling_templates_registry.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {
const int64_t DTYPE_FLOAT_BIG_K_V2 = 20000;
const int64_t DTYPE_FLOAT16_BIG_K_V2 = 20001;
const int64_t DTYPE_BF16_BIG_K_V2 = 20002;
const int64_t DTYPE_BF16_V2 = 20003;
const int64_t DTYPE_FLOAT_DB_V2 = 20004;
const int64_t DTYPE_FLOAT16_DB_V2 = 20005;
const int64_t DTYPE_FLOAT_CUTH_K2_V2 = 20006;
const int64_t DTYPE_FLOAT16_CUTH_K2_V2 = 20007;
const int64_t DTYPE_BF16_CUTH_K2_V2 = 20008;
const int64_t DTYPE_FLOAT_CUTH_K4_V2 = 20009;
const int64_t DTYPE_FLOAT16_CUTH_K4_V2 = 20010;
const int64_t DTYPE_BF16_CUTH_K4_V2 = 20011;
const int64_t DTYPE_FLOAT_CUTH_V2 = 20012;
const int64_t DTYPE_FLOAT16_CUTH_V2 = 20013;
const int64_t DTYPE_BF16_CUTH_V2 = 20014;
const int64_t DTYPE_FLOAT_DB_ALL_BIAS_V2 = 20015;
const int64_t DTYPE_FLOAT16_DB_ALL_BIAS_V2 = 20016;
const int64_t DTYPE_BF16_ALL_BIAS_V2 = 20017;
const int64_t DTYPE_FLOAT_CUTH_NETWORK_V2 = 20018;
const int64_t DTYPE_FLOAT_BIG_K_V2_WITHOUT_BIAS = 20019;
const int64_t DTYPE_FLOAT16_BIG_K_V2_WITHOUT_BIAS = 20020;
const int64_t DTYPE_BF16_BIG_K_V2_WITHOUT_BIAS = 20021;
const int64_t DTYPE_BF16_V2_WITHOUT_BIAS = 20022;
const int64_t DTYPE_FLOAT_DB_V2_WITHOUT_BIAS = 20023;
const int64_t DTYPE_FLOAT16_DB_V2_WITHOUT_BIAS = 20024;
const int64_t DTYPE_FLOAT_CUTH_K2_V2_WITHOUT_BIAS = 20025;
const int64_t DTYPE_FLOAT16_CUTH_K2_V2_WITHOUT_BIAS = 20026;
const int64_t DTYPE_BF16_CUTH_K2_V2_WITHOUT_BIAS = 20027;
const int64_t DTYPE_FLOAT_CUTH_K4_V2_WITHOUT_BIAS = 20028;
const int64_t DTYPE_FLOAT16_CUTH_K4_V2_WITHOUT_BIAS = 20029;
const int64_t DTYPE_BF16_CUTH_K4_V2_WITHOUT_BIAS = 20030;
const int64_t DTYPE_FLOAT_CUTH_V2_WITHOUT_BIAS = 20031;
const int64_t DTYPE_FLOAT16_CUTH_V2_WITHOUT_BIAS = 20032;
const int64_t DTYPE_BF16_CUTH_V2_WITHOUT_BIAS = 20033;
const int64_t DTYPE_FLOAT_DB_ALL_BIAS_V2_WITHOUT_BIAS = 20034;
const int64_t DTYPE_FLOAT16_DB_ALL_BIAS_V2_WITHOUT_BIAS = 20035;
const int64_t DTYPE_BF16_ALL_BIAS_V2_WITHOUT_BIAS = 20036;
const int64_t DTYPE_FLOAT_CUTH_NETWORK_V2_WITHOUT_BIAS = 20037;
const int64_t DTYPE_BF16_BIG_K_V2_MIX_PRECISION = 20038;
const int64_t DTYPE_BF16_V2_MIX_PRECISION = 20039;
const int64_t DTYPE_BF16_CUTH_K2_V2_MIX_PRECISION = 20040;
const int64_t DTYPE_BF16_CUTH_K4_V2_MIX_PRECISION = 20041;
const int64_t DTYPE_BF16_CUTH_V2_MIX_PRECISION = 20042;
const int64_t DTYPE_BF16_ALL_BIAS_V2_MIX_PRECISION = 20043;
const int64_t DTYPE_BF16_BIG_K_V2_WITHOUT_BIAS_MIX_PRECISION = 20044;
const int64_t DTYPE_BF16_V2_WITHOUT_BIAS_MIX_PRECISION = 20045;
const int64_t DTYPE_BF16_CUTH_K2_V2_WITHOUT_BIAS_MIX_PRECISION = 20046;
const int64_t DTYPE_BF16_CUTH_K4_V2_WITHOUT_BIAS_MIX_PRECISION = 20047;
const int64_t DTYPE_BF16_CUTH_V2_WITHOUT_BIAS_MIX_PRECISION = 20048;
const int64_t DTYPE_BF16_ALL_BIAS_V2_WITHOUT_BIAS_MIX_PRECISION = 20049;

BEGIN_TILING_DATA_DEF(MoeFinalizeRoutingV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, skip2IsNull);
TILING_DATA_FIELD_DEF(int64_t, biasRowNum);          // bias的行数
TILING_DATA_FIELD_DEF(int64_t, totalRowNum);         // skip1的行数
TILING_DATA_FIELD_DEF(int64_t, H);                   // skip1的列数
TILING_DATA_FIELD_DEF(int64_t, normalH);             // skip1的列数切分后满载的列数大小
TILING_DATA_FIELD_DEF(int64_t, unnormalH);           // skip1的列数切分后满载后剩余的列数大小
TILING_DATA_FIELD_DEF(int64_t, hSliceNum);           // skip1的列数切分的次数
TILING_DATA_FIELD_DEF(int64_t, normalK);             // scales的列数切分后满载的列数大小
TILING_DATA_FIELD_DEF(int64_t, unnormalK);           // scales的列数切分后满载后剩余的列数大小
TILING_DATA_FIELD_DEF(int64_t, kSliceNum);           // scales的列数切片的数量
TILING_DATA_FIELD_DEF(int64_t, K);                   // scales的列数
TILING_DATA_FIELD_DEF(int64_t, normalCoreHandleNum); // 非尾核，每个核处理的skip1行数
TILING_DATA_FIELD_DEF(int64_t, normalCoreLoopNum);   // 非尾核，每个核需要的循环次数
TILING_DATA_FIELD_DEF(int64_t, normalCoreHandleNumPerLoop); // 非尾核，每个核，非尾Loop，每次loop需要处理的skip1行数
TILING_DATA_FIELD_DEF(int64_t, normalCoreHandleNumTailLoop); // 非尾核，每个核，尾Loop需要处理的skip1行数
TILING_DATA_FIELD_DEF(int64_t, tailCoreHandleNum);           // 尾核处理的skip1行数
TILING_DATA_FIELD_DEF(int64_t, tailCoreLoopNum);             // 尾核，每个核需要的循环次数
TILING_DATA_FIELD_DEF(int64_t, tailCoreHandleNumPerLoop);  // 尾核，每个核，非尾Loop需要处理的skip1行数
TILING_DATA_FIELD_DEF(int64_t, tailCoreHandleNumTailLoop); // 尾核，每个核，尾Loop需要处理的skip1行数
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(int64_t, skip1IsNull);
TILING_DATA_FIELD_DEF(int64_t, dropPadMode);
TILING_DATA_FIELD_DEF(int64_t, scalesIsNull);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeFinalizeRoutingV2, MoeFinalizeRoutingV2TilingData)

struct MoeFinalizeRoutingCompileInfoV2 {
    int32_t totalCoreNum = 0;
    uint64_t ubSize = 0;
};

struct ShapeParamsV2 {
    const gert::StorageShape* expandedXShape = nullptr;
    const gert::StorageShape* expandedRowIdxShape = nullptr;
    const gert::StorageShape* x1Shape = nullptr;
    const gert::StorageShape* x2Shape = nullptr;
    const gert::StorageShape* biasShape = nullptr;
    const gert::StorageShape* scalesShape = nullptr;
};

class MoeFinalizeRoutingTilingV2 : public Ops::Transformer::OpTiling::TilingBaseClass
{
public:
    explicit MoeFinalizeRoutingTilingV2(gert::TilingContext* context) : TilingBaseClass(context){};

protected:
    // 重载TilingBase基类方法
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    // 内部计算方法
    virtual ge::graphStatus DoGetPlatformInfo() = 0;
    virtual ge::graphStatus DoGetShapeAttrsInfo() = 0;
    virtual ge::graphStatus CalcOpTiling() = 0;
    virtual ge::graphStatus CalcTilingKey() = 0;
    virtual void DoPostTiling() = 0;
    virtual void PrintTilingData() = 0;

protected:
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    size_t offset_{0};
    int64_t totalCoreNum_{0};
    int64_t ubSize_{0};
    int64_t usedCoreNum_{0};
    int64_t skip2IsNull_{0};
    int64_t biasRowNum_{0};
    int64_t totalRowNum_{0};
    int64_t h_{0};
    int64_t normalH_{0};
    int64_t unnormalH_{0};
    int64_t hSliceNum_{0};
    int64_t normalK_{0};
    int64_t unnormalK_{0};
    int64_t kSliceNum_{0};
    int64_t k_{0};
    int64_t normalCoreHandleNum_{0};
    int64_t normalCoreLoopNum_{0};
    int64_t normalCoreHandleNumPerLoop_{0};
    int64_t normalCoreHandleNumTailLoop_{0};
    int64_t tailCoreHandleNum_{0};
    int64_t tailCoreLoopNum_{0};
    int64_t tailCoreHandleNumPerLoop_{0};
    int64_t tailCoreHandleNumTailLoop_{0};
    ge::DataType dataType_;
    ge::DataType scaleDataType_;
    int inputDataTypeSize_{0};
    int scaleInputDataTypeSize_{0};
    int64_t skip1IsNull_{0};
    int64_t biasIsNull_{0};
    bool isCanLoadH_{true};
    bool isCanLoadAllBias_{false};
    bool isOptimizedCutH_{false};
    int64_t scalesIsNull_{0};
    int64_t dropPadMode_{0};

    uint64_t tilingKey_ = 0;
};
} // namespace optiling