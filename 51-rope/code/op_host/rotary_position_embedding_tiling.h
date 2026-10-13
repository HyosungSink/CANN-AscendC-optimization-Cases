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
 * \file rotary_position_embedding.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ROTARY_POSITION_EMBEDDING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ROTARY_POSITION_EMBEDDING_H


#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_util.h"
#include "platform/platform_info.h"
#include "util/math_util.h"
namespace optiling {

BEGIN_TILING_DATA_DEF(RotateMatrixParams)
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
    TILING_DATA_FIELD_DEF(uint64_t, blockNum);           // baseMN num
    TILING_DATA_FIELD_DEF(uint64_t, blockNumM);
    TILING_DATA_FIELD_DEF(uint64_t, blockNumN);
    TILING_DATA_FIELD_DEF(uint64_t, cvParallNum);        // cv parall num
    TILING_DATA_FIELD_DEF(uint64_t, tilingMode);         // layout code
    TILING_DATA_FIELD_DEF(uint64_t, gmLength);           // x dim
    TILING_DATA_FIELD_DEF(uint64_t, broadcastFirstDim);  // broadcast or not
    TILING_DATA_FIELD_DEF(uint64_t, broadcastSecondDim); // broadcast or not
    TILING_DATA_FIELD_DEF(uint64_t, broadcastThirdDim);  // broadcast or not
    TILING_DATA_FIELD_DEF(uint64_t, dLength);            // D dim length
    TILING_DATA_FIELD_DEF(uint64_t, baseM);                 
    TILING_DATA_FIELD_DEF(uint64_t, baseN);                  
    TILING_DATA_FIELD_DEF(uint64_t, baseK);               
    TILING_DATA_FIELD_DEF(uint64_t, coreNum);           // baseMN num
    TILING_DATA_FIELD_DEF(uint64_t, m);
    TILING_DATA_FIELD_DEF(uint64_t, xFirstDim);              // xFirstDim
    TILING_DATA_FIELD_DEF(uint64_t, xSecondDim);             // xSecondDim
    TILING_DATA_FIELD_DEF(uint64_t, xThirdDim);              // xThirdDim
    TILING_DATA_FIELD_DEF(uint64_t, cosSinFirstDim);            // cosSinFirstDim
    TILING_DATA_FIELD_DEF(uint64_t, cosSinSecondDim);           // cosSinSecondDim
    TILING_DATA_FIELD_DEF(uint64_t, cosSinThirdDim);            // cosSinThirdDim
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RotateMatrixParamsOp, RotateMatrixParams)


BEGIN_TILING_DATA_DEF(RotateHalfParams)
TILING_DATA_FIELD_DEF(uint64_t, tilingMode); // layout code
TILING_DATA_FIELD_DEF(uint64_t, gmLength);
TILING_DATA_FIELD_DEF(uint64_t, broadcastFirstDim);  // B
TILING_DATA_FIELD_DEF(uint64_t, broadcastSecondDim); // N
TILING_DATA_FIELD_DEF(uint64_t, dLength);            // D dim length
TILING_DATA_FIELD_DEF(uint64_t, halfDLength);        // D/2
TILING_DATA_FIELD_DEF(uint64_t, halfDPadLength);     // D/2 pads block aligned length
TILING_DATA_FIELD_DEF(uint64_t, dPadLength);         // D pads block aligned length when D/2 is not aligned
TILING_DATA_FIELD_DEF(uint64_t, isAligned);          // is D/2 block aligned, 0: false; 1: true
TILING_DATA_FIELD_DEF(uint64_t, totalSLines);        // S dim length
TILING_DATA_FIELD_DEF(uint64_t, storeSLines);        // number of S lines ub can store
TILING_DATA_FIELD_DEF(uint64_t, storeDataLength);
TILING_DATA_FIELD_DEF(uint64_t, storePadDataLength);
// former cores tiling data
TILING_DATA_FIELD_DEF(uint64_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, formerSLines);           // number of S lines in each former core
TILING_DATA_FIELD_DEF(uint64_t, formerUbLoop);           // ub loop times in former cores
TILING_DATA_FIELD_DEF(uint64_t, formerUbLast);           // S lines left after ub loop
TILING_DATA_FIELD_DEF(uint64_t, formerXDataLength);      // x data length in each former core
TILING_DATA_FIELD_DEF(uint64_t, formerRDataLength);      // r data length in each former core
TILING_DATA_FIELD_DEF(uint64_t, formerXCoreOffset);      // x total data length former core(start offset of tail core)
TILING_DATA_FIELD_DEF(uint64_t, formerRCoreOffset);      // r total data length former core(start offset of tail core)
TILING_DATA_FIELD_DEF(uint64_t, formerUbLastDataLength); // ub last processed data length
TILING_DATA_FIELD_DEF(uint64_t, formerUbLastPadDataLength); // ub last processed pad data length
// tail cores tiling data
TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, tailSLines);
TILING_DATA_FIELD_DEF(uint64_t, tailUbLoop);
TILING_DATA_FIELD_DEF(uint64_t, tailUbLast);
TILING_DATA_FIELD_DEF(uint64_t, tailXDataLength);
TILING_DATA_FIELD_DEF(uint64_t, tailRDataLength);
TILING_DATA_FIELD_DEF(uint64_t, tailUbLastDataLength);
TILING_DATA_FIELD_DEF(uint64_t, tailUbLastPadDataLength);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RotateHalfParamsOp, RotateHalfParams)

BEGIN_TILING_DATA_DEF(RopeInterleavedParams)
TILING_DATA_FIELD_DEF(uint64_t, batchSize);
TILING_DATA_FIELD_DEF(uint64_t, seqLen);
TILING_DATA_FIELD_DEF(uint64_t, numHeads);
TILING_DATA_FIELD_DEF(uint64_t, headDim);
TILING_DATA_FIELD_DEF(uint64_t, frontCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, coreCalcNum);
TILING_DATA_FIELD_DEF(uint64_t, coreCalcTail);
TILING_DATA_FIELD_DEF(uint64_t, ubCalcNum);
TILING_DATA_FIELD_DEF(uint64_t, ubCalcLoop);
TILING_DATA_FIELD_DEF(uint64_t, ubCalcTail);
TILING_DATA_FIELD_DEF(uint64_t, ubCalcTailNum);
TILING_DATA_FIELD_DEF(uint64_t, ubCalcTailLoop);
TILING_DATA_FIELD_DEF(uint64_t, ubCalcTailTail);
TILING_DATA_FIELD_DEF(uint64_t, ubCalcBNum);
TILING_DATA_FIELD_DEF(uint64_t, ubCalcBLoop);
TILING_DATA_FIELD_DEF(uint64_t, ubCalcBTail);
TILING_DATA_FIELD_DEF(uint64_t, ubCalcNNum);
TILING_DATA_FIELD_DEF(uint64_t, ubCalcNLoop);
TILING_DATA_FIELD_DEF(uint64_t, ubCalcNTail);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RopeInterleavedParamsOp, RopeInterleavedParams)

BEGIN_TILING_DATA_DEF(RotaryPositionEmbeddingTilingData)
TILING_DATA_FIELD_DEF_STRUCT(RotateMatrixParams, rotateMatrixParams);
TILING_DATA_FIELD_DEF_STRUCT(RotateHalfParams, rotateHalfParams);
TILING_DATA_FIELD_DEF_STRUCT(RopeInterleavedParams, ropeInterleavedParams);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RotaryPositionEmbedding, RotaryPositionEmbeddingTilingData)

BEGIN_TILING_DATA_DEF(RopeRegbaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, B);
TILING_DATA_FIELD_DEF(int64_t, CosB);
TILING_DATA_FIELD_DEF(int64_t, S);
TILING_DATA_FIELD_DEF(int64_t, D);
TILING_DATA_FIELD_DEF(int64_t, N);
TILING_DATA_FIELD_DEF(int64_t, blockNumB);
TILING_DATA_FIELD_DEF(int64_t, blockFactorB);
TILING_DATA_FIELD_DEF(int64_t, blockNumS);
TILING_DATA_FIELD_DEF(int64_t, blockFactorS);
TILING_DATA_FIELD_DEF(int64_t, ubLoopNumS);
TILING_DATA_FIELD_DEF(int64_t, ubFactorS);
TILING_DATA_FIELD_DEF(int64_t, ubTailFactorS);
TILING_DATA_FIELD_DEF(int64_t, ubLoopNumB);
TILING_DATA_FIELD_DEF(int64_t, ubFactorB);
TILING_DATA_FIELD_DEF(int64_t, ubTailFactorB);
TILING_DATA_FIELD_DEF(int64_t, ubLoopNumN);
TILING_DATA_FIELD_DEF(int64_t, ubFactorN);
TILING_DATA_FIELD_DEF(int64_t, ubTailFactorN);
TILING_DATA_FIELD_DEF(int64_t, rotaryMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RotaryPositionEmbedding_20010, RopeRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(RotaryPositionEmbedding_20011, RopeRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(RotaryPositionEmbedding_20020, RopeRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(RotaryPositionEmbedding_20040, RopeRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(RotaryPositionEmbedding_20041, RopeRegbaseTilingData)

BEGIN_TILING_DATA_DEF(RopeRegbaseABTilingData)
TILING_DATA_FIELD_DEF(int64_t, B);
TILING_DATA_FIELD_DEF(int64_t, CosB);
TILING_DATA_FIELD_DEF(int64_t, S);
TILING_DATA_FIELD_DEF(int64_t, D);
TILING_DATA_FIELD_DEF(int64_t, N);
TILING_DATA_FIELD_DEF(int64_t, dAlign);
TILING_DATA_FIELD_DEF(int64_t, dSplitCoef);
TILING_DATA_FIELD_DEF(int64_t, blockNumBS);
TILING_DATA_FIELD_DEF(int64_t, blockFactorBS);
TILING_DATA_FIELD_DEF(int64_t, blockTailBS);
TILING_DATA_FIELD_DEF(int64_t, blockNumN);
TILING_DATA_FIELD_DEF(int64_t, blockFactorN);
TILING_DATA_FIELD_DEF(int64_t, blockTailN);
TILING_DATA_FIELD_DEF(int64_t, ubFactorBS);
TILING_DATA_FIELD_DEF(int64_t, ubFactorN);
TILING_DATA_FIELD_DEF(int64_t, rotaryMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RotaryPositionEmbedding_20030, RopeRegbaseABTilingData)

struct RotaryPositionEmbeddingCompileInfo {
    int64_t numBlocks;
    uint64_t ubSize;
    platform_ascendc::SocVersion socVersion;
};

enum class RopeLayout : uint8_t {
    NO_BROADCAST = 1,
    BROADCAST_BSN = 2,
    BSND = 3,
    SBND = 4,
    BNSD = 5
};

enum class RotaryPosEmbeddingMode : uint8_t {
    HALF = 0,
    INTERLEAVE = 1,
    QUARTER = 2,
    DEEPSEEK_INTERLEAVE = 3
};

class RotaryPosEmbeddingMembaseTilingClass : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit RotaryPosEmbeddingMembaseTilingClass(gert::TilingContext *context) : TilingBaseClass(context)
    {
    }

    void Reset(gert::TilingContext *context) override
    {
        RotaryPosEmbeddingMembaseTilingClass::Reset(context);
    }

protected:
    ge::graphStatus GetPlatformInfo() override;

    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    bool IsCapable() override
    {
        return true;
    }
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetShapeAttrsInfo() override;

    uint64_t GetTilingKey() const override
    {
        return context_->GetTilingKey();
    }

protected:
    static const uint32_t MODE_ROTATE_INTERLEAVED = 1;
    uint32_t inputMode_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
};

class RopeRegBaseTilingClass : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit RopeRegBaseTilingClass(gert::TilingContext *context) : TilingBaseClass(context)
    {
    }

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
    }

private:
    bool IsRotaryPosEmbeddingMode(const int32_t mode) const;
    ge::graphStatus CheckNullptr();
    ge::graphStatus CheckShape();
    ge::graphStatus CheckDtypeAndAttr();
    ge::graphStatus CheckParam();
    ge::graphStatus JudgeLayoutByShape(const gert::Shape &xShape, const gert::Shape &cosShape);
    ge::graphStatus CheckRotaryModeShapeRelation(const int64_t d);
    ge::graphStatus CheckShapeAllPositive(const int64_t idx) const;
    ge::graphStatus CheckShapeAllPositive() const;
    std::string rotaryModeStr_;

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override
    {
        return 0;
    }

    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    bool IsCapable() override
    {
        return true;
    }
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    int64_t b_{0};
    int64_t s_{0};
    int64_t n_{0};
    int64_t d_{0};
    int64_t cosb_{0};
    ge::DataType dtype_;
    RopeLayout layout_;
    RotaryPosEmbeddingMode rotaryMode_;

    int64_t blockSize_;
    int64_t dSplitCoef_;
    bool is1snd_ = false;
};

} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ROTARY_POSITION_EMBEDDING_H
