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
 * \file add_rms_norm_quant_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_QUANT_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_QUANT_H_

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
BEGIN_TILING_DATA_DEF(AddRMSNormQuantTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numRow);
TILING_DATA_FIELD_DEF(uint32_t, numCol);
TILING_DATA_FIELD_DEF(uint32_t, blockFactor);
TILING_DATA_FIELD_DEF(uint32_t, rowFactor);
TILING_DATA_FIELD_DEF(uint32_t, ubFactor);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avgFactor);
TILING_DATA_FIELD_DEF(uint32_t, hasZeroPoints1);
TILING_DATA_FIELD_DEF(uint32_t, hasBeta);
TILING_DATA_FIELD_DEF(uint32_t, divMode);
TILING_DATA_FIELD_DEF(uint32_t, hasScales2);
TILING_DATA_FIELD_DEF(uint32_t, hasZeroPoints2);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(AddRmsNormQuantRegbaseTilingData)
TILING_DATA_FIELD_DEF(uint64_t, numM);  //A
TILING_DATA_FIELD_DEF(uint64_t, numN);  //R
TILING_DATA_FIELD_DEF(uint64_t, baseM);  //ubfactor ub处理a的大小
TILING_DATA_FIELD_DEF(uint64_t, baseN);  //全载时=R
TILING_DATA_FIELD_DEF(uint64_t, baseNReduceAlign);
TILING_DATA_FIELD_DEF(uint64_t, baseNDtypeAlign);  //R对32B对齐的个数
TILING_DATA_FIELD_DEF(uint64_t, powerLoop);
TILING_DATA_FIELD_DEF(uint64_t, powerSplit);  //binaryAdd 二分折叠点
TILING_DATA_FIELD_DEF(uint64_t, mPerCore); //blockFactor 单核处理a的大小
TILING_DATA_FIELD_DEF(uint64_t, mLastCore); //blockTail 尾核处理a的大小
TILING_DATA_FIELD_DEF(float, avgFactor);
TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddRmsNormQuant, AddRMSNormQuantTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuantV2, AddRMSNormQuantTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1000, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1010, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1020, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1030, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1040, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1050, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1060, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1070, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1100, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1110, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1120, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1130, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1140, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1150, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1160, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1170, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1001, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1011, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1021, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1031, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1041, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1051, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1061, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1071, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1101, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1111, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1121, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1131, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1141, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1151, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1161, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1171, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1002, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1012, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1022, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1032, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1042, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1052, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1062, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1072, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1102, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1112, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1122, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1132, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1142, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1152, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1162, AddRmsNormQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormQuant_1172, AddRmsNormQuantRegbaseTilingData)
constexpr uint32_t TILING_TYPE_NORMAL = 0;
constexpr uint32_t TILING_TYPE_SPILT = 1;
constexpr uint32_t TILING_TYPE_PERF = 2;
constexpr uint32_t TILING_OFFSET_HAS_QUANT = 10;
constexpr uint32_t TILING_OFFSET_DIV_MODE = 100;
constexpr uint32_t TILING_OFFSET_REGBASE = 1000;
constexpr uint64_t X1_INDEX = 0;
constexpr uint64_t X2_INDEX = 1;
constexpr uint64_t GAMMA_INDEX = 2;
constexpr uint64_t SCALES1_INDEX = 3;
constexpr uint64_t SCALES2_INDEX = 4;
constexpr uint64_t ZERO_POINTS1_INDEX = 5;
constexpr uint64_t ZERO_POINTS2_INDEX = 6;
constexpr uint64_t BETA_INDEX = 7;
constexpr uint64_t Y1_INDEX = 0;
constexpr uint64_t Y2_INDEX = 1;
constexpr uint64_t X_INDEX = 2;
constexpr uint64_t EPS_ATTR_INDEX = 1;
constexpr uint64_t DIV_MODE_ATTR_INDEX = 2;

struct AddRmsNormQuantCompileInfo {
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t totalCoreNum = 0;
    uint64_t maxUbSize = 0;
};

struct AddRmsNormQuantRegbaseTilingParams {
    // Platform
    uint64_t maxUbSize{0};
    uint64_t totalCoreNum{0};
    uint64_t vecLength{0};
    // Input Info
    uint64_t numM{0};
    uint64_t numN{0};
    uint64_t xDtypeSize{0};
    uint64_t quantDtypeSize{0};
    uint64_t zeroPointDtypeSize{0};
    uint64_t xDtypeAlignNum{0};
    uint64_t xReduceAlignNum{0};
    uint64_t quantDtypeAlignNum{0};
    uint64_t zeroPointDtypeAlignNum{0};
    // Cal params
    uint64_t baseM{0};
    uint64_t baseN{0};
    uint64_t baseNB8Align{0};
    uint64_t baseNDtypeAlign{0};
    uint64_t baseNQuantAlign{0};
    uint64_t baseNReduceAlign{0};
    uint64_t reduceBufLenAlign{0};
    uint64_t powerLoop{0};
    uint64_t powerSplit{0};
    uint64_t mPerCore{0};
    uint64_t mLastCore{0};
    uint64_t usedCoreNum{0};
    // Workspace
    uint64_t workspaceSize{0};
    // Tiling key parmas
    uint64_t tilingType{0};

    float avgFactor{0};
    float epsilon{0};
    uint32_t quantBufCnt{0};
    bool divMode{false};
    bool hasScales2{false};
    bool hasZeroPoints1{false};
    bool hasZeroPoints2{false};
    bool hasY2{false};
    bool needGetCompileInfo{false};
};

class AddRmsNormQuantRegbaseTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit AddRmsNormQuantRegbaseTiling(gert::TilingContext* tilingContext) : Ops::NN::Optiling::TilingBaseClass(tilingContext)
    {}
    ~AddRmsNormQuantRegbaseTiling() override
    {}

    const string nodeName = "AddRmsNormQuantRegbase";
    AddRmsNormQuantRegbaseTilingData tilingData;
    AddRmsNormQuantRegbaseTilingParams tilingParams;

    bool CheckShapeBC(
        const gert::StorageShape* srcBcShape, const gert::StorageShape* srcShape, string srcBcName, string srcName);
    ge::graphStatus CheckDtypeVaild(
        ge::DataType& srcDtype, std::vector<ge::DataType>& supportDtypeList, string srcName);
    bool CheckShapeNull();
    void CheckOptionalInput();
    bool CheckInputShapeDim();
    bool CheckInputShapeValue();
    bool CheckInputDtype();
    bool CheckOutputDtype();
    ge::graphStatus SetInputParams();
    uint64_t CalUBTotalSize(uint64_t baseM, uint64_t baseN, const uint32_t tilingType);
    int64_t CalFullLoadBaseM(uint64_t baseN, int64_t& tmpPower);
    uint64_t CalUsedSize(uint64_t baseM, uint64_t baseNB8Align, uint64_t baseNB32Align, int64_t tmpPower, int64_t firstVcaddLength);
    ge::graphStatus SetTilingParams();
    void SetTilingData();
    void PrintTilingData();

protected:
    // Order: GetShapeAttrsInfo->GetPlatformInfo->
    //        IsCapable->DoOpTiling->DoLibApiTiling->
    //        GetWorkspaceSize->PostTiling->GetTilingKey
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
};

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_QUANT_H_
