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
 * \file grouped_matmul_finalize_routing_base_tiling.h
 * \brief
 */
#ifndef __OP_HOST_GROUPED_MATMUL_FINALIZE_ROUTING_BASE_TILING_H__
#define __OP_HOST_GROUPED_MATMUL_FINALIZE_ROUTING_BASE_TILING_H__

#include "grouped_matmul_finalize_routing_tiling.h"
#include "../common/include/tiling_base/tiling_base.h"
#include "../common/include/err/ops_err.h"

namespace optiling {
namespace grouped_matmul_finalize_routing {
class GroupedMatmulFinalizeRoutingBaseTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
 public:
    explicit GroupedMatmulFinalizeRoutingBaseTiling(gert::TilingContext* context) : Ops::Transformer::OpTiling::TilingBaseClass(context) {};

    ~GroupedMatmulFinalizeRoutingBaseTiling() override = default;

protected:
    bool IsCapable() override { 
        return true; 
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override {return ge::GRAPH_SUCCESS;};
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override{return ge::GRAPH_SUCCESS;};
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override{return ge::GRAPH_SUCCESS;};
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override{return ge::GRAPH_SUCCESS;};
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    ge::graphStatus GetInputShape();
    ge::graphStatus ParseInputAndAttr();
    ge::graphStatus ParseAttr();

    ge::graphStatus W4A8BaseTilingProcess();
    ge::graphStatus W4A8L1OptTilingProcess();
    ge::graphStatus W4A8TilingProcess();
    ge::graphStatus W8A8TilingProcess();
    void DeterministicTilingProcess();
    void OtherSettingTilingProcess();
    void FillTilingData();
    void FillTilingDataL1Opt();
    void PrintTilingData();
protected:

private:
    GroupMatmulFRTilingData tilingData_;
    bool failFlag_;
    uint32_t blockDim_;
    uint32_t groupNum_;
    uint32_t batch_;
    uint64_t k_;
    uint64_t m_;
    uint64_t n_;
    uint64_t vBaseM_;
    uint32_t sharedInputOffset_;
    uint32_t sharedInputLen_;
    float residualScale_;
    uint32_t quantGroupNum_;
    uint32_t withOffset_;
    uint64_t ubRestBytes_;
    uint64_t ubSize_;
    uint64_t l1Size_;
    uint64_t l0CSize_;
    int64_t tuningConfig_;
    uint32_t hasPertokenScale_;
    uint32_t hasBias_;
    uint32_t deterministicFlag_;
    uint32_t deterWorkspaceSize_;
    bool useL1OptKernel_;    
protected:
    matmul_tiling::MultiCoreMatmulTiling mm_;
};
}
}
#endif // __OP_HOST_GROUPED_MATMUL_FINALIZE_ROUTING_BASE_TILING_H__
