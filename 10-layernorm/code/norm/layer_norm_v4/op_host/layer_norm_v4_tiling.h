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
 * \file layer_norm_v4_tiling.h
 * \brief
 */

#ifndef LAYER_NORM_V4_TILING_H
#define LAYER_NORM_V4_TILING_H

#include "util/math_util.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {
constexpr uint64_t LN_TEMPLATE_KEY_WEIGHT = 100;

BEGIN_TILING_DATA_DEF(LayerNormV4TilingData)
TILING_DATA_FIELD_DEF(uint32_t, colSize);
TILING_DATA_FIELD_DEF(uint32_t, rowSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormV4, LayerNormV4TilingData)

BEGIN_TILING_DATA_DEF(LayerNormV4TilingDataSingleRead)
TILING_DATA_FIELD_DEF(uint32_t, numBlocks);
TILING_DATA_FIELD_DEF(uint32_t, colSize);
TILING_DATA_FIELD_DEF(uint32_t, rowSize);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, coefficient);
TILING_DATA_FIELD_DEF(uint32_t, rowAlign);
TILING_DATA_FIELD_DEF(uint32_t, nRow);
TILING_DATA_FIELD_DEF(uint32_t, tailNRow);
TILING_DATA_FIELD_DEF(uint32_t, loopCount);
TILING_DATA_FIELD_DEF(uint32_t, tailLoop);
TILING_DATA_FIELD_DEF(uint32_t, tileLength);
TILING_DATA_FIELD_DEF(uint32_t, blockLength);
TILING_DATA_FIELD_DEF(uint32_t, nullptrGamma);
TILING_DATA_FIELD_DEF(uint32_t, nullptrBeta);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormV4_100, LayerNormV4TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormV4_110, LayerNormV4TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormV4_111, LayerNormV4TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormV4_120, LayerNormV4TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormV4_122, LayerNormV4TilingDataSingleRead)

BEGIN_TILING_DATA_DEF(LayerNormV4TilingDataTranspose)
TILING_DATA_FIELD_DEF(uint64_t, col);                    // 输入tensor的行
TILING_DATA_FIELD_DEF(uint64_t, row);                    // 输入tensor的列，即reduce的轴
TILING_DATA_FIELD_DEF(uint64_t, numBlocks);               // 实际使用的core数量
TILING_DATA_FIELD_DEF(uint64_t, blockFormer);            // 整核处理的row大小
TILING_DATA_FIELD_DEF(uint64_t, blockTail);              // 尾核处理的row大小
TILING_DATA_FIELD_DEF(uint64_t, ubFormer);               // ub整循环处理的row大小
TILING_DATA_FIELD_DEF(uint64_t, ubLoopOfFormerBlock);    // 整核处理的ub循环次数
TILING_DATA_FIELD_DEF(uint64_t, ubLoopOfTailBlock);      // 尾核处理的ub循环次数
TILING_DATA_FIELD_DEF(uint64_t, ubTailOfFormerBlock);    // 整核ub尾循环处理的row大小
TILING_DATA_FIELD_DEF(uint64_t, ubTailOfTailBlock);      // 尾核ub尾循环处理的row大小
TILING_DATA_FIELD_DEF(uint64_t, bFormer);                // ubFormer借轴大小，ubFormer->16*bFormer
TILING_DATA_FIELD_DEF(uint64_t, dichotomizeAddDiffSize); // row与小于row的最近二次幂的差值
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, coefficient);
TILING_DATA_FIELD_DEF(uint32_t, nullptrGamma);
TILING_DATA_FIELD_DEF(uint32_t, nullptrBeta);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormV4_200, LayerNormV4TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormV4_210, LayerNormV4TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormV4_211, LayerNormV4TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormV4_220, LayerNormV4TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormV4_222, LayerNormV4TilingDataTranspose)

BEGIN_TILING_DATA_DEF(LayerNormV4TilingDataRegBaseTwoPass)
TILING_DATA_FIELD_DEF(int64_t, r);
TILING_DATA_FIELD_DEF(int64_t, rAlign);
TILING_DATA_FIELD_DEF(int64_t, a);
TILING_DATA_FIELD_DEF(int64_t, aFactor);
TILING_DATA_FIELD_DEF(int64_t, aBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, binaryAddQuotient);
TILING_DATA_FIELD_DEF(int64_t, binaryAddK);
TILING_DATA_FIELD_DEF(int64_t, binaryAddLastNum);
TILING_DATA_FIELD_DEF(int64_t, powerOfTwoForR);
TILING_DATA_FIELD_DEF(int64_t, tmpBufferSize);
TILING_DATA_FIELD_DEF(int64_t, nullptrGamma);
TILING_DATA_FIELD_DEF(int64_t, nullptrBeta);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF_STRUCT(LayerNormSeparateTiling, layerNormTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormV4_300, LayerNormV4TilingDataRegBaseTwoPass)
REGISTER_TILING_DATA_CLASS(LayerNormV4_310, LayerNormV4TilingDataRegBaseTwoPass)
REGISTER_TILING_DATA_CLASS(LayerNormV4_311, LayerNormV4TilingDataRegBaseTwoPass)
REGISTER_TILING_DATA_CLASS(LayerNormV4_320, LayerNormV4TilingDataRegBaseTwoPass)
REGISTER_TILING_DATA_CLASS(LayerNormV4_322, LayerNormV4TilingDataRegBaseTwoPass)

BEGIN_TILING_DATA_DEF(LayerNormV4TilingDataWelford)
TILING_DATA_FIELD_DEF(int64_t, M);                  // 输入tensor的行
TILING_DATA_FIELD_DEF(int64_t, N);                  // 输入tensor的列，即reduce的轴
TILING_DATA_FIELD_DEF(int64_t, rAlign);             // r对齐的大小
TILING_DATA_FIELD_DEF(int64_t, numBlocks);           // 实际使用的core数量
TILING_DATA_FIELD_DEF(int64_t, mainBlockCount);     // 整核的数量
TILING_DATA_FIELD_DEF(int64_t, mainBlockFactor);    // 整核处理的row大小
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor);    // 尾核处理的row大小
TILING_DATA_FIELD_DEF(int64_t, tileLength);         // tile块的元素个数
TILING_DATA_FIELD_DEF(int64_t, welfordTempSize);    // welford临时buffer的大小
TILING_DATA_FIELD_DEF(int64_t, welfordUpdateTimes); // welford update的次数
TILING_DATA_FIELD_DEF(int64_t, welfordUpdateTail);  // welford update的尾数
TILING_DATA_FIELD_DEF(int64_t, nullptrGamma);
TILING_DATA_FIELD_DEF(int64_t, nullptrBeta);
TILING_DATA_FIELD_DEF(int64_t, apiTempBufferSize);
TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormV4_400, LayerNormV4TilingDataWelford)
REGISTER_TILING_DATA_CLASS(LayerNormV4_410, LayerNormV4TilingDataWelford)
REGISTER_TILING_DATA_CLASS(LayerNormV4_411, LayerNormV4TilingDataWelford)
REGISTER_TILING_DATA_CLASS(LayerNormV4_420, LayerNormV4TilingDataWelford)
REGISTER_TILING_DATA_CLASS(LayerNormV4_422, LayerNormV4TilingDataWelford)

BEGIN_TILING_DATA_DEF(LayerNormV4TilingDataRegBaseTwoPassPerf)
TILING_DATA_FIELD_DEF(int64_t, a);
TILING_DATA_FIELD_DEF(int64_t, aBlockFactor);
TILING_DATA_FIELD_DEF(int32_t, aUbFactor);
TILING_DATA_FIELD_DEF(int32_t, aUbFactorAlignB32);
TILING_DATA_FIELD_DEF(int32_t, r);
TILING_DATA_FIELD_DEF(int32_t, rAlign);
TILING_DATA_FIELD_DEF(int32_t, formerBlockUbLoops);
TILING_DATA_FIELD_DEF(int32_t, tailBlockUbLoops);
TILING_DATA_FIELD_DEF(int32_t, powerOfTwoForR);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(int8_t, nullptrGamma);
TILING_DATA_FIELD_DEF(int8_t, nullptrBeta);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormV4_500, LayerNormV4TilingDataRegBaseTwoPassPerf)
REGISTER_TILING_DATA_CLASS(LayerNormV4_510, LayerNormV4TilingDataRegBaseTwoPassPerf)
REGISTER_TILING_DATA_CLASS(LayerNormV4_511, LayerNormV4TilingDataRegBaseTwoPassPerf)
REGISTER_TILING_DATA_CLASS(LayerNormV4_520, LayerNormV4TilingDataRegBaseTwoPassPerf)
REGISTER_TILING_DATA_CLASS(LayerNormV4_522, LayerNormV4TilingDataRegBaseTwoPassPerf)

BEGIN_TILING_DATA_DEF(LayerNormV4CommonTilingData)
TILING_DATA_FIELD_DEF(int64_t, numBlocks);
TILING_DATA_FIELD_DEF(uint32_t, colSize);
TILING_DATA_FIELD_DEF(uint32_t, rowSize);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(uint32_t, space);
TILING_DATA_FIELD_DEF(float, coefficient);
TILING_DATA_FIELD_DEF(uint32_t, nullptrGamma);
TILING_DATA_FIELD_DEF(uint32_t, nullptrBeta);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormV4_700, LayerNormV4CommonTilingData);
REGISTER_TILING_DATA_CLASS(LayerNormV4_701, LayerNormV4CommonTilingData);
REGISTER_TILING_DATA_CLASS(LayerNormV4_702, LayerNormV4CommonTilingData);

BEGIN_TILING_DATA_DEF(LayerNormV4MergeNTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numBlocks);
TILING_DATA_FIELD_DEF(uint32_t, colSize);
TILING_DATA_FIELD_DEF(uint32_t, rowSize);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, coefficient);
TILING_DATA_FIELD_DEF(uint32_t, rowAlign);
TILING_DATA_FIELD_DEF(uint32_t, nRow);
TILING_DATA_FIELD_DEF(uint32_t, ableNRow);
TILING_DATA_FIELD_DEF(uint32_t, loopCount);
TILING_DATA_FIELD_DEF(uint32_t, formerNum);
TILING_DATA_FIELD_DEF(uint32_t, formerRemainNum);
TILING_DATA_FIELD_DEF(uint32_t, tailNRow);
TILING_DATA_FIELD_DEF(uint32_t, tailLoop);
TILING_DATA_FIELD_DEF(uint32_t, tailNum);
TILING_DATA_FIELD_DEF(uint32_t, tailRemainNum);
TILING_DATA_FIELD_DEF(uint32_t, tileLength);
TILING_DATA_FIELD_DEF(uint32_t, blockLength);
TILING_DATA_FIELD_DEF(uint32_t, nullptrGamma);
TILING_DATA_FIELD_DEF(uint32_t, nullptrBeta);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormV4_600, LayerNormV4MergeNTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormV4_1600, LayerNormV4MergeNTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormV4_2600, LayerNormV4MergeNTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormV4_601, LayerNormV4MergeNTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormV4_1601, LayerNormV4MergeNTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormV4_2601, LayerNormV4MergeNTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormV4_602, LayerNormV4MergeNTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormV4_1602, LayerNormV4MergeNTilingData)
REGISTER_TILING_DATA_CLASS(LayerNormV4_2602, LayerNormV4MergeNTilingData)

struct ParamsLayerNomrV4 {
    uint64_t coreNum;
    uint64_t ubSizePlatForm;
    int64_t blockSize = 0;
    uint64_t colSize = 1;
    uint64_t rowSize = 1;
    float eps = 0;
    float coefficient = 0;
    uint64_t rowAlign;
    uint64_t gammaNullPtr;
    uint64_t betaNullPtr;
    uint64_t meanAndRstdNullPtr = 0;
    ge::DataType tensorDtype;
    ge::DataType paramDtype;
    int64_t dtypeKey = 0;
    bool isAscend310P = false;
    bool isRegBase = false;
    int64_t vlFp32 = 0;
    bool isV3 = false;
};

enum class LayerNormV4TilingKey : int64_t
{
    // FLOAT32/FLOAT16/BFLOAT16 -- 0/1/2
    // Single Read
    LAYER_NORM_SINGLE_READ_FLOAT32_FLOAT32 = 100,
    LAYER_NORM_SINGLE_READ_FLOAT16_FLOAT32 = 110,
    LAYER_NORM_SINGLE_READ_FLOAT16_FLOAT16 = 111,
    LAYER_NORM_SINGLE_READ_BFLOAT16_FLOAT32 = 120,
    LAYER_NORM_SINGLE_READ_BFLOAT16_BFLOAT16 = 122,
    // Transpose
    LAYER_NORM_TRANSPOSE_FLOAT32_FLOAT32 = 200,
    LAYER_NORM_TRANSPOSE_FLOAT16_FLOAT32 = 210,
    LAYER_NORM_TRANSPOSE_FLOAT16_FLOAT16 = 211,
    LAYER_NORM_TRANSPOSE_BFLOAT16_FLOAT32 = 220,
    LAYER_NORM_TRANSPOSE_BFLOAT16_BFLOAT16 = 222,
    // Regbase two pass
    LAYER_NORM_REGBASE_TWO_PASS_FLOAT32_FLOAT32 = 300,
    LAYER_NORM_REGBASE_TWO_PASS_FLOAT16_FLOAT32 = 310,
    LAYER_NORM_REGBASE_TWO_PASS_FLOAT16_FLOAT16 = 311,
    LAYER_NORM_REGBASE_TWO_PASS_BFLOAT16_FLOAT32 = 320,
    LAYER_NORM_REGBASE_TWO_PASS_BFLOAT16_BFLOAT16 = 322,
    // Regbase two pass perf
    LAYER_NORM_REGBASE_TWO_PASS_PERF_FLOAT32_FLOAT32 = 500,
    LAYER_NORM_REGBASE_TWO_PASS_PERF_FLOAT16_FLOAT32 = 510,
    LAYER_NORM_REGBASE_TWO_PASS_PERF_FLOAT16_FLOAT16 = 511,
    LAYER_NORM_REGBASE_TWO_PASS_PERF_BFLOAT16_FLOAT32 = 520,
    LAYER_NORM_REGBASE_TWO_PASS_PERF_BFLOAT16_BFLOAT16 = 522,

    LAYER_NORM_MERGE_N_FLOAT32_FLOAT32 = 600,

    LAYER_NORM_COMMON_BFLOAT16_BFLOAT16 = 700,
};

enum class LNTemplateKey : int
{
    SINGEL_READ = 1,
    TRANSPOSE = 2,
    SINGEL_READ_REGBASE = 3,
    WELFORD = 4
};

struct LayerNormV4CompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool isAscend310P = false;
    bool isRegBase = false;
    uint32_t vectorLength = 0;
    uint64_t blockSize = 0;
};

class LayerNormV4TilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit LayerNormV4TilingBase(gert::TilingContext* tilingContext) : TilingBaseClass(tilingContext)
    {}
    ~LayerNormV4TilingBase() override = default;
    ParamsLayerNomrV4 commonParams;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
};

class LayerNormV4SingleReadTiling : public LayerNormV4TilingBase {
public:
    explicit LayerNormV4SingleReadTiling(gert::TilingContext* tilingContext) : LayerNormV4TilingBase(tilingContext)
    {}
    ~LayerNormV4SingleReadTiling() override = default;
    LayerNormV4TilingDataSingleRead td_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
};

class LayerNormV4MergeNTiling : public LayerNormV4TilingBase {
public:
    explicit LayerNormV4MergeNTiling(gert::TilingContext* tilingContext) : LayerNormV4TilingBase(tilingContext)
    {}
    ~LayerNormV4MergeNTiling() override = default;
    LayerNormV4MergeNTilingData td_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void PrepareTiling(uint64_t ubSize, uint32_t& nRow, uint32_t tailNRow, uint32_t tailNum);
};

class LayerNormV4CommonTiling : public LayerNormV4TilingBase {
public:
    explicit LayerNormV4CommonTiling(gert::TilingContext* tilingContext) : LayerNormV4TilingBase(tilingContext)
    {}
    ~LayerNormV4CommonTiling() override = default;
    LayerNormV4CommonTilingData td_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void TilingDataPrint();
};

class LayerNormV4TransposeTiling : public LayerNormV4TilingBase {
public:
    explicit LayerNormV4TransposeTiling(gert::TilingContext* tilingContext) : LayerNormV4TilingBase(tilingContext)
    {}
    ~LayerNormV4TransposeTiling() override = default;
    LayerNormV4TilingDataTranspose td_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

private:
    struct BlockTilingData {
        uint64_t numBlocks;
        uint64_t blockFormer;
        uint64_t blockTail;
    };

    struct UbTilingData {
        uint64_t ubFormer;
        uint64_t bFormer;
        uint64_t ubLoopOfFormerBlock;
        uint64_t ubLoopOfTailBlock;
        uint64_t ubTailOfFormerBlock;
        uint64_t ubTailOfTailBlock;
    };
    void DoBlockTiling(BlockTilingData& blockTilingParams);
    void DoUbTiling(const BlockTilingData& blockTilingParams, UbTilingData& ubTilingParams);
    uint64_t CalcBorrowFactor(uint64_t oriFactor);
    uint32_t FindDichotomizeAddDiffSize();
};

class LayerNormV4RegBaseTwoPassTiling : public LayerNormV4TilingBase {
public:
    explicit LayerNormV4RegBaseTwoPassTiling(gert::TilingContext* tilingContext) : LayerNormV4TilingBase(tilingContext)
    {}
    ~LayerNormV4RegBaseTwoPassTiling() override = default;
    LayerNormV4TilingDataRegBaseTwoPass td_;

protected:
    bool CanFitInBuffer(int64_t curA, int64_t largeBufferMemPerA, int64_t baseMemSize, int64_t& tmpBufferUse);
    bool CanFitInBuffer(
        int64_t curA, int64_t largeBufferMemPerA, int64_t baseMemSize, int64_t& tmpBufferUse, int64_t xElemSize);
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;

    int64_t binaryAddQuotient;
    int64_t blockNum;
};

class LayerNormV4WelfordTiling : public LayerNormV4TilingBase {
public:
    explicit LayerNormV4WelfordTiling(gert::TilingContext* tilingContext) : LayerNormV4TilingBase(tilingContext)
    {}
    ~LayerNormV4WelfordTiling() override = default;
    LayerNormV4TilingDataWelford td_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;

protected:
    bool IsValidTileLength(int64_t tileLength);
};

class LayerNormV4RegBaseTwoPassPerfTiling : public LayerNormV4TilingBase {
public:
    explicit LayerNormV4RegBaseTwoPassPerfTiling(gert::TilingContext* tilingContext) :
        LayerNormV4TilingBase(tilingContext) {}
    ~LayerNormV4RegBaseTwoPassPerfTiling() override = default;
    LayerNormV4TilingDataRegBaseTwoPassPerf td_;

protected:
    int64_t GetUBCanUseSize();
    int64_t GetRowWeight();
    bool CanFitInBuffer(int64_t curA);
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;

    int64_t blockNum_;
};

int64_t GetDTypeKey(ge::DataType tensorDtype, ge::DataType paramDtype);
ge::graphStatus TilingPrepare4CompileInfo(gert::TilingParseContext* context, LayerNormV4CompileInfo* compileInfo);
ge::graphStatus GetCommonPlatformInfo(
    gert::TilingContext* context, const LayerNormV4CompileInfo* compileInfo, ParamsLayerNomrV4& commonParams);
ge::graphStatus GetCommonShapeAttrsInfo(
    gert::TilingContext* context, uint64_t colSize, uint64_t rowSize, ParamsLayerNomrV4& commonParams);
} // namespace optiling
#endif // LAYER_NORM_V4_TILING_H
