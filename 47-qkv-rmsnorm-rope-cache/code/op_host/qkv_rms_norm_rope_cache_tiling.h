/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qkv_rms_norm_rope_cache_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_QKV_RMS_NORM_ROPE_CACHE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_QKV_RMS_NORM_ROPE_CACHE_H_

#include "register/tilingdata_base.h"
#include "../common/include/tiling_base/tiling_base.h"
#include "../common/include/tiling_base/tiling_templates_registry.h"
#include <unordered_map>
namespace optiling {
// tiling结构体
BEGIN_TILING_DATA_DEF(QkvRmsNormRopeCacheTilingData)
TILING_DATA_FIELD_DEF(int64_t, batchSize); // Bqkv
TILING_DATA_FIELD_DEF(int64_t, seqLength); // Sqkv
TILING_DATA_FIELD_DEF(int64_t, numHead); // Nqkv
TILING_DATA_FIELD_DEF(int64_t, qkvDim); // D
TILING_DATA_FIELD_DEF(int64_t, ropeRange); // D
TILING_DATA_FIELD_DEF(int64_t, numHeadQ); // Nq
TILING_DATA_FIELD_DEF(int64_t, numHeadK); // Nk
TILING_DATA_FIELD_DEF(int64_t, numHeadV); //Nv
TILING_DATA_FIELD_DEF(int64_t, blockNum); // k_cache/v_cache的blockNum
TILING_DATA_FIELD_DEF(int64_t, blockSize); // k_cache/v_cache的blockSize
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, blockFactorQ);
TILING_DATA_FIELD_DEF(int64_t, blockFactorK);
TILING_DATA_FIELD_DEF(int64_t, blockFactorV);
TILING_DATA_FIELD_DEF(int64_t, blockDim);
TILING_DATA_FIELD_DEF(int64_t, blockDimQ);
TILING_DATA_FIELD_DEF(int64_t, blockDimK);
TILING_DATA_FIELD_DEF(int64_t, blockDimV);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, ubFactorQ);
TILING_DATA_FIELD_DEF(int64_t, ubFactorK);
TILING_DATA_FIELD_DEF(int64_t, ubFactorV);
TILING_DATA_FIELD_DEF(float, reciprocal);
TILING_DATA_FIELD_DEF(int64_t, isOutputQkv);
TILING_DATA_FIELD_DEF(int64_t, isKQuant);
TILING_DATA_FIELD_DEF(int64_t, isVQuant);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QkvRmsNormRopeCache, QkvRmsNormRopeCacheTilingData)

constexpr int32_t TEMPLATE_DS_PRIORITY = 1000;

struct QkvRmsNormRopeCacheCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

enum CacheMode
{
    Norm = 0,
    PA = 1,
    PA_NZ = 2,
    PA_BLK_BNSD = 3,
    PA_BLK_NZ = 4
};

constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t ONE_BUFFER = 1;
constexpr int64_t DIM_NUM_ONE = 1;
// Tensor Indices
constexpr int64_t QKV_INDEX = 0;
constexpr int64_t GAMMA_Q_INDEX = 1;
constexpr int64_t GAMMA_K_INDEX = 2;
constexpr int64_t COS_INDEX = 3;
constexpr int64_t SIN_INDEX = 4;
constexpr int64_t INDEX_INDEX = 5;
// In-place inputs
constexpr int64_t Q_OUT_INDEX = 6;
constexpr int64_t K_CACHE_INDEX = 7;     
constexpr int64_t V_CACHE_INDEX = 8;    
// Quant Scale
constexpr int64_t K_SCALE_IDX = 9; 
constexpr int64_t V_SCALE_IDX = 10;   
// Quant Offset
constexpr int64_t K_OFFSET_IDX = 11;   
constexpr int64_t V_OFFSET_IDX = 12;  
constexpr float FAC_K = 0.6;  // K路和V路计算量的比例
constexpr float NEAR_ONE = 0.999999f;

// Attr Indices
constexpr int64_t QKV_SIZE_IDX = 0;
constexpr int64_t HEAD_NUMS_IDX = 1;
constexpr int64_t EPSILON_IDX = 2;
constexpr int64_t CACHE_MODE_IDX = 3;
constexpr int64_t IS_OUTPUT_QKV_IDX = 4;

constexpr int64_t SHAPE_IDX_B = 0;
constexpr int64_t SHAPE_IDX_S = 1;
constexpr int64_t SHAPE_IDX_N = 2;
constexpr int64_t SHAPE_IDX_D = 3;
constexpr int64_t SHAPE_IDX_BS = 0;
constexpr int64_t SHAPE_IDX_ND = 1;
constexpr int64_t SHAPE_IDX_BLOCK_NUM = 0;
constexpr int64_t SHAPE_IDX_BLOCK_SIZE = 2;

constexpr int64_t FLOAT32_BYTES = 4;
constexpr int64_t FLOAT16_BYTES = 2;
constexpr int64_t INT8_BYTES = sizeof(int8_t);
constexpr int64_t FP32_BLOCK_ALIGN_NUM = 8;
constexpr int64_t FP16_BLOCK_ALIGN_NUM = 16;
constexpr int64_t INT8_BLOCK_ALIGN_NUM = 32;
constexpr int64_t BASE_BLOCK_SIZE = 32;

constexpr int64_t DIM_SIZE = 4;
constexpr int64_t DIM_ZERO = 0;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;
constexpr int64_t NUM_ONE = 1;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_THREE = 3;
constexpr int64_t NUM_FOUR = 4;
constexpr int64_t NUM_HUNDRED = 100;
constexpr int64_t NUM_CACHE_MODE_UNIT = 10;

constexpr int64_t BYTES_PER_KILO_BYTE = 1024;
static constexpr int64_t UB_RESERVED_BYTES = 1 * BYTES_PER_KILO_BYTE; // 多留1K

class QkvRmsNormRopeCacheTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit QkvRmsNormRopeCacheTilingBase(gert::TilingContext* tillingContext) : TilingBaseClass(tillingContext)
    {}
    ~QkvRmsNormRopeCacheTilingBase() override
    {}
    uint64_t tilingKey_{0};
    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    bool isRegbase_{false};
    int64_t rope_seq_ = 0; // cos/sin的S
    CacheMode currentCacheMode_ = CacheMode::PA_NZ;
    int64_t quantMode_ = 0; // 是否量化
    int64_t batchSize_ = 0;
    int64_t seqLength_ = 0;
    int64_t numHead_ = 0;
    int64_t qkvDim_ = 0;
    int64_t ropeRange_ = 0;
    int64_t numHeadQ_ = 0;
    int64_t numHeadK_ = 0;
    int64_t numHeadV_ = 0;
    int64_t blockNum_ = 0;
    int64_t blockSize_ = 0;
    float epsilon_ = 0.0;
    int64_t blockFactor_ = 0;
    int64_t blockFactorQ_ = 0;
    int64_t blockFactorK_ = 0;
    int64_t blockFactorV_ = 0;
    int64_t blockDim_ = 0;
    int64_t blockDimQ_ = 0;
    int64_t blockDimK_ = 0;
    int64_t blockDimV_ = 0;
    int64_t ubFactor_ = 0;
    int64_t ubFactorQ_ = 0;
    int64_t ubFactorK_ = 0;
    int64_t ubFactorV_ = 0;
    float reciprocal_ = 0.0;
    int64_t isOutputQkv_ = 0;
    int64_t isKQuant_ = 1;
    int64_t isVQuant_ = 1;

    ge::DataType qkvDtype_{ge::DataType::DT_FLOAT16};
    int64_t qkvDtypeSize_{0};// qkv数据类型所占字节数

protected:
    ge::graphStatus GetShapeAttrsInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override
    {
        return false;
    }
    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override {}

protected:
    std::tuple<int64_t, int64_t, int64_t, int64_t> GetShapeTuple(
        const gert::TilingContext* context, const int64_t index = 0);
    std::tuple<int64_t, int64_t> GetShapeTupleOfTH(
        const gert::TilingContext* context, const int64_t index = 0);
};

class QkvRmsNormRopeCacheTilingDs : virtual public QkvRmsNormRopeCacheTilingBase {
public:
    explicit QkvRmsNormRopeCacheTilingDs(gert::TilingContext* tillingContext) : QkvRmsNormRopeCacheTilingBase(tillingContext)
    {}
    ~QkvRmsNormRopeCacheTilingDs()
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

protected:
    ge::graphStatus GetShapeAttrsInfoInner();
    void CalUbTiling();
    ge::graphStatus CheckQkvValid();
    ge::graphStatus CheckGammaValid(int64_t gammaIdx);
    ge::graphStatus CheckCosSinValid();
    ge::graphStatus CheckIndexValid();
    ge::graphStatus CheckQOutValid();
    ge::graphStatus CheckKCacheValid();
    ge::graphStatus CheckVCacheValid();
    ge::graphStatus CheckKScaleValid();
    ge::graphStatus CheckVScaleValid();
    ge::graphStatus CheckKOffsetValid();
    ge::graphStatus CheckVOffsetValid();

private:
    QkvRmsNormRopeCacheTilingData tilingData_;
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_QKV_RMS_NORM_ROPE_CACHE_H_
