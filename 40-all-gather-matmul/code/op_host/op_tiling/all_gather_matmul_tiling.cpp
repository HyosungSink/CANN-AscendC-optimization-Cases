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
 * \file all_gather_matmul_tiling.cc
 * \brief
 */
#include "vector"
#include "../../common/inc/mc2_hcom_topo_info.h"
#include "../../common/inc/tiling/matmul_formulaic_tiling.h"
#include "all_gather_formulaic_tiling.h"
#include "../../common/inc/mc2_log.h"
#include "../../common/inc/ops_utils.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../../common/inc/tiling/mc2_tiling_utils.h"

#include "../../op_kernel/all_gather_matmul_tiling_key.h"
#include "../../op_kernel/all_gather_matmul_tiling.h"

using namespace AscendC;
using namespace ge;
using namespace all_gather_matmul_tiling_key;

namespace {
const std::map<uint32_t, std::vector<uint32_t>> VALID_RANK = {
    {0, {2, 4, 8}},
    {1, {2, 4, 8, 16, 32}}
    };

constexpr size_t OUTPUT_IDX = 0;
constexpr size_t GATHEROUT_IDX = 1;
constexpr size_t INPUT_X1_IDX = 0;
constexpr size_t INPUT_X2_IDX = 1;
constexpr size_t INPUT_BIAS_IDX = 2;
constexpr size_t DIM0_IDX = 0;
constexpr size_t DIM1_IDX = 1;
constexpr size_t GATHER_IDX = 3;
constexpr size_t GROUP_IDX = 0;
constexpr size_t IS_TRANS_A_IDX = 1;
static void PrintTilingData(::TCubeTiling& tiling)
{
    OP_LOGD("AllGatherMatmul", " tiling.usedCoreNum %d", tiling.usedCoreNum);
    OP_LOGD("AllGatherMatmul", " tiling.M %d", tiling.M);
    OP_LOGD("AllGatherMatmul", " tiling.N %d", tiling.N);
    OP_LOGD("AllGatherMatmul", " tiling.Ka %d", tiling.Ka);
    OP_LOGD("AllGatherMatmul", " tiling.Kb %d", tiling.Kb);
    OP_LOGD("AllGatherMatmul", " tiling.singleCoreM %d", tiling.singleCoreM);
    OP_LOGD("AllGatherMatmul", " tiling.singleCoreN %d", tiling.singleCoreN);
    OP_LOGD("AllGatherMatmul", " tiling.singleCoreK %d", tiling.singleCoreK);
    OP_LOGD("AllGatherMatmul", " tiling.baseM %d", tiling.baseM);
    OP_LOGD("AllGatherMatmul", " tiling.baseN %d", tiling.baseN);
    OP_LOGD("AllGatherMatmul", " tiling.baseK %d", tiling.baseK);
    OP_LOGD("AllGatherMatmul", " tiling.depthA1 %d", tiling.depthA1);
    OP_LOGD("AllGatherMatmul", " tiling.depthB1 %d", tiling.depthB1);
    OP_LOGD("AllGatherMatmul", " tiling.stepM %d", tiling.stepM);
    OP_LOGD("AllGatherMatmul", " tiling.stepN %d", tiling.stepN);
    OP_LOGD("AllGatherMatmul", " tiling.stepka %d", tiling.stepKa);
    OP_LOGD("AllGatherMatmul", " tiling.stepkb %d", tiling.stepKb);
    OP_LOGD("AllGatherMatmul", " tiling.isBias %d", tiling.isBias);
    OP_LOGD("AllGatherMatmul", " tiling.transLength %d", tiling.transLength);
    OP_LOGD("AllGatherMatmul", " tiling.iterateOrder %s", ((tiling.iterateOrder == 1) ? "orderM" : "orderN"));
    OP_LOGD("AllGatherMatmul", " tiling.usedL1Size %d", tiling.shareL1Size);
    OP_LOGD("AllGatherMatmul", " tiling.usedL0CSize %d", tiling.shareL0CSize);
    OP_LOGD("AllGatherMatmul", " tiling.dbL0C %d", tiling.dbL0C);
    OP_LOGD("AllGatherMatmul", " tiling.usedUBSize %d", tiling.shareUbSize);
    OP_LOGD("AllGatherMatmul", " tiling.batchM %d", tiling.batchM);
    OP_LOGD("AllGatherMatmul", " tiling.batchN %d", tiling.batchN);
    OP_LOGD("AllGatherMatmul", " tiling.singleBatchM %d", tiling.singleBatchM);
    OP_LOGD("AllGatherMatmul", " tiling.singleBatchN %d", tiling.singleBatchN);
}

static void PrintTilingData(Mc2Tiling::RCSTiling& rcsTiling)
{
    OP_LOGD("AllGatherMatmul", " rcsTiling.commtype %d", rcsTiling.commtype);
    OP_LOGD("AllGatherMatmul", " rcsTiling.subtype %d", rcsTiling.subtype);
    OP_LOGD("AllGatherMatmul", " rcsTiling.rankDim %d", rcsTiling.rankDim);
    OP_LOGD("AllGatherMatmul", " rcsTiling.rankID %d", rcsTiling.rankID);
    OP_LOGD("AllGatherMatmul", " rcsTiling.tileCnt %d", rcsTiling.tileCnt);
    OP_LOGD("AllGatherMatmul", " rcsTiling.tailM %d", rcsTiling.tailM);
    OP_LOGD("AllGatherMatmul", " rcsTiling.tailCnt %d", rcsTiling.tailCnt);
    OP_LOGD("AllGatherMatmul", " rcsTiling.isTransA %d", rcsTiling.isTransposeA);
    OP_LOGD("AllGatherMatmul", " rcsTiling.isTransB %d", rcsTiling.isTransposeB);
    OP_LOGD("AllGatherMatmul", " rcsTiling.rankM %d", rcsTiling.rankM);
    OP_LOGD("AllGatherMatmul", " rcsTiling.rankN %d", rcsTiling.rankN);
    OP_LOGD("AllGatherMatmul", " rcsTiling.rankK %d", rcsTiling.rankK);
    OP_LOGD("AllGatherMatmul", " rcsTiling.gatherIndex %d", rcsTiling.gatherIndex);
    OP_LOGD("AllGatherMatmul", " rcsTiling.cToFloatLen %lu", rcsTiling.cToFloatLen);
    OP_LOGD("AllGatherMatmul", " rcsTiling.nd2NzWorkLen %lu", rcsTiling.nd2NzWorkLen);
    OP_LOGD("AllGatherMatmul", " rcsTiling.gatherLen %lu", rcsTiling.gatherLen);
}

static void PrintTilingData(Mc2Tiling::TileL2Tiling& tileL2Tiling)
{
    OP_LOGD("AllGatherMatmul", " tileL2Tiling.mL2TileCnt %d", tileL2Tiling.mL2TileCnt);
    OP_LOGD("AllGatherMatmul", " tileL2Tiling.nL2TileCnt %d", tileL2Tiling.nL2TileCnt);
    OP_LOGD("AllGatherMatmul", " tileL2Tiling.mTileBlocks %d", tileL2Tiling.mTileBlocks);
    OP_LOGD("AllGatherMatmul", " tileL2Tiling.nTileBlocks %d", tileL2Tiling.nTileBlocks);
    OP_LOGD("AllGatherMatmul", " tileL2Tiling.mTailBlocks %d", tileL2Tiling.mTailBlocks);
    OP_LOGD("AllGatherMatmul", " tileL2Tiling.nTailBlocks %d", tileL2Tiling.nTailBlocks);
    OP_LOGD("AllGatherMatmul", " tileL2Tiling.rankTileNum %d", tileL2Tiling.rankTileNum);
    OP_LOGD("AllGatherMatmul", " tileL2Tiling.calcOrder %d", tileL2Tiling.calcOrder);
    OP_LOGD("AllGatherMatmul", " tileL2Tiling.enableL2Tile %d", tileL2Tiling.enableL2Tile);
}
}

namespace optiling {

static ge::graphStatus CalcMatmulTiling(mc2tiling::TilingArgs& args, ::TCubeTiling& cubeTiling,
                                        Mc2Tiling::TileL2Tiling &l2Tiling);

static ge::graphStatus MC2SetWorkspace(gert::TilingContext* context, Mc2Tiling::AllGatherMatmulTilingData& tilingData,
                                       mc2tiling::TilingArgs& args);

static uint32_t MC2_Splite(mc2tiling::TilingArgs& args, uint32_t maxTileCnt = 64)
{
    // 检查允许通信的最大次数
    if (args.commTurn >= maxTileCnt) {
        args.commTurn = maxTileCnt;
    }

    uint64_t tileLen = 1;
    if (args.mValue > args.commTurn) {
        tileLen = args.mValue/ args.commTurn;
    }

    if (args.inputDtypeSize == 2) { // 数据长度为2, 则向 2*64 = 128，则向128对齐
        tileLen = mc2tiling::AlignUp<uint64_t>(tileLen, 64); // align size
    } else if (args.inputDtypeSize == 4) { // 4 is float32 type size
        tileLen = mc2tiling::AlignUp<uint64_t>(tileLen, 32); // align size
    }
    if (args.mValue > tileLen) {
        return tileLen;
    }
    return args.mValue;
}

static bool CheckOutputParamDim0(gert::TilingContext* context)
{
    Mc2Tiling::AllGatherMatmulTilingData* tilingData = context->GetTilingData<Mc2Tiling::AllGatherMatmulTilingData>();
    auto outputShape = context->GetOutputShape(OUTPUT_IDX);
    uint64_t outputDim0 = outputShape->GetStorageShape().GetDim(DIM0_IDX);
    const gert::StorageShape* x1Shape = context->GetInputShape(INPUT_X1_IDX);
    uint64_t x1Dim0 = x1Shape->GetStorageShape().GetDim(DIM0_IDX);
    auto group = context->GetAttrs()->GetAttrPointer<char>(GROUP_IDX);
    auto rankSize = mc2tiling::MatmulFormulaicTiling::GetRankSize(group);
    uint64_t mValue = x1Dim0 * static_cast<uint64_t>(rankSize);

    OP_TILING_CHECK(outputDim0 != mValue,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
        "m-axis should be %lu, but output's m-axis is %lu", mValue, outputDim0),
        return false);

    return true;
}

static ge::graphStatus AllGatherParamsCheck(const gert::TilingContext* context)
{
    OP_TILING_CHECK(mc2tiling::Mc2TilingUtils::CommonParamCheck(context) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "common check failed"), return ge::GRAPH_FAILED);

    const gert::StorageShape* aShape = context->GetInputShape(INPUT_X1_IDX);
    uint64_t valueOne = aShape->GetStorageShape().GetDim(DIM0_IDX);
    uint64_t valueTwo = aShape->GetStorageShape().GetDim(DIM1_IDX);

    OP_TILING_CHECK(valueOne == 0 || valueTwo == 0,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "the value is invalid"), return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(!CheckOutputParamDim0(const_cast<gert::TilingContext*>(context)),
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "the output's dim0 is invalid"), return ge::GRAPH_FAILED);

    if (context->GetAttrs() == nullptr) {
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "get attrs failed");
    } else {
        auto gatherIndex = context->GetAttrs()->GetAttrPointer<int64_t>(GATHER_IDX);
        OP_TILING_CHECK(*gatherIndex != 0,
            VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
        "the gatherIndex should be 0, but real value is %d", *gatherIndex), return ge::GRAPH_FAILED);

        auto isTransA = context->GetAttrs()->GetAttrPointer<bool>(IS_TRANS_A_IDX);
        OP_TILING_CHECK(*isTransA != false,
            VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "the isTransA should be false, but real value is 1"), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((valueTwo < KVALUE_MIN || valueTwo >= KVALUE_MAX),
            VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
            "The k-axis should be in range[256, 65535), but it is: %lu.", valueTwo), return ge::GRAPH_FAILED);
    }
    auto group = context->GetAttrs()->GetAttrPointer<char>(static_cast<int>(GROUP_IDX));
    OP_TILING_CHECK(group == nullptr, VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "group is nullptr. "),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetCommAlg(Mc2Tiling::AllGatherMatmulTilingData &tilingData)
{
    tilingData.socParam.commAlg = COMM_ALG_FULL_MESH;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAllGatherFormulateTileCnt(const gert::TilingContext* ctx,
    Mc2Tiling::AllGatherMatmulTilingData& tilingData, mc2tiling::TilingArgs& args)
{
    if (ctx->GetAttrs() == nullptr) {
        OP_LOGW(ctx->GetNodeName(), " ctx->GetAttrs is nullptr.");
        return ge::GRAPH_FAILED;
    }

    SocVersion inputSocVersion = (tilingData.socParam.isA3 == 0) ? SocVersion::SOC910_B : SocVersion::SOC910_93;
    
    AllGatherPlusMM tileFormulate(args, args.rankDim, KernelType::ALL_GATHER, inputSocVersion);
    tileFormulate.GetTiling();
    CutResult mCutGather = tileFormulate.tilingM_.cutRes;
    tilingData.param.tileCnt = mCutGather.numLongTile;
    args.mValue = mCutGather.longTileLen;
    CalcMatmulTiling(args, tilingData.tileTiling, tilingData.tileL2Tiling);
    args.baseMLimit = mCutGather.longTileLen;
    args.mValue = mCutGather.longTileLen * args.rankTileNum;
    tilingData.param.tailM = mCutGather.shortTileLen;
    tilingData.param.tailCnt = 0;
    if (mCutGather.numShortTile > 0) {
        args.mValue = mCutGather.shortTileLen;
        tilingData.param.tailM = args.mValue;
        tilingData.param.tailCnt = mCutGather.numShortTile;
        CalcMatmulTiling(args, tilingData.tailTiling, tilingData.tailL2Tiling);
        args.baseMLimit = mCutGather.shortTileLen;
        args.mValue = mCutGather.shortTileLen * args.rankTileNum;
    }
    args.mValue = mCutGather.longTileLen;
    return ge::GRAPH_SUCCESS;
}

// 第一个参数m
static ge::graphStatus MCSpliteM(gert::TilingContext* ctx, Mc2Tiling::AllGatherMatmulTilingData& tilingData,
                                 mc2tiling::TilingArgs& args)
{
    args.rankTileNum = args.rankDim - 1;
    // cmdType = HCCL_CMD_ALLGATHER, 是允许切K
    if (args.enableSplitK) { // 只有1份
        tilingData.param.tileCnt = 1;
        tilingData.param.tailCnt = 0;
        tilingData.param.tailM = 0;

        CalcMatmulTiling(args, tilingData.tileTiling, tilingData.tileL2Tiling);
    } else if (args.commTurn != 0) {
        uint64_t splite = MC2_Splite(args);

        // 现在找到1个合适的切分
        auto tileCnt = args.mValue / splite; // 切的份数
        auto tileTail = args.mValue % splite; // 尾巴

        tilingData.param.tileCnt = tileCnt;
        args.mValue = splite;
        tilingData.param.tailCnt = 0;
        CalcMatmulTiling(args, tilingData.tileTiling, tilingData.tileL2Tiling);
        tilingData.param.tailM = tileTail;
        if (tileTail != 0) {
            args.mValue = tileTail;
            tilingData.param.tailCnt = 1;
            CalcMatmulTiling(args, tilingData.tailTiling, tilingData.tailL2Tiling);
        }
        args.mValue = splite;
    } else {
        GetAllGatherFormulateTileCnt(ctx, tilingData, args);
    }
    MC2SetWorkspace(ctx, tilingData, args);

    return ge::GRAPH_SUCCESS;
}

static void UpdateTilingKey(uint64_t& tilingKey, Mc2Tiling::AllGatherMatmulTilingData& tilingData, bool isBias)
{
    bool allGatherMatmulFullMesh = true;
    bool allGatherMatmulNd2nzOpt = false;
    bool allGatherMatmulBiasCast = false;

    if(isBias) {
        allGatherMatmulBiasCast = true;
    }
    else {
        allGatherMatmulBiasCast = false;
    }

    if(tilingData.socParam.isND2NZ == 1) {
        allGatherMatmulNd2nzOpt = true;
    }
    else {
        allGatherMatmulNd2nzOpt = false;
    }

    if (tilingData.socParam.commAlg == COMM_ALG_FULL_MESH){
        allGatherMatmulFullMesh = true;
    }
    else {
        allGatherMatmulFullMesh = false;
    }

    tilingKey = GET_TPL_TILING_KEY(allGatherMatmulFullMesh, allGatherMatmulNd2nzOpt, allGatherMatmulBiasCast);
}

static ge::graphStatus SetMatmulTilingAllGatherMatmul(gert::TilingContext* context,
                                                      Mc2Tiling::AllGatherMatmulTilingData& tilingData,
                                                      mc2tiling::TilingArgs& args)
{
    ge::DataType  biasType;
    bool isBias = true;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNumAic();
    auto aType = context->GetInputDesc(INPUT_X1_IDX)->GetDataType();
    auto bType = context->GetInputDesc(INPUT_X2_IDX)->GetDataType();
    auto cType = aType;
    const gert::StorageShape* matrixBias = context->GetOptionalInputShape(INPUT_BIAS_IDX);
    if (matrixBias == nullptr) {
        isBias = false;
        biasType = cType;
    }
    else {
        biasType = context->GetInputDesc(INPUT_BIAS_IDX)->GetDataType();
    }

    const gert::StorageShape* aShape = context->GetInputShape(INPUT_X1_IDX);
    const gert::StorageShape* bShape = context->GetInputShape(INPUT_X2_IDX);
    uint64_t mValue = aShape->GetStorageShape().GetDim(DIM0_IDX);
    uint64_t kValue = aShape->GetStorageShape().GetDim(DIM1_IDX);
    uint64_t nValue = bShape->GetStorageShape().GetDim(DIM1_IDX);

    if (aShape->GetStorageShape().GetDim(DIM1_IDX) != bShape->GetStorageShape().GetDim(DIM0_IDX)) {
        OP_LOGD(context->GetNodeName(), "A.shape(1) %lu B.shape(0) %lu, istransB = %d",
                aShape->GetStorageShape().GetDim(DIM1_IDX), bShape->GetStorageShape().GetDim(DIM0_IDX), args.isBTrans);
        nValue = bShape->GetStorageShape().GetDim(DIM0_IDX);
    }

    uint64_t inputDtypeSize = mc2tiling::D_TYPE_SIZE_MAP.at(aType);
    uint64_t outputDtypeSize = mc2tiling::D_TYPE_SIZE_MAP.at(cType);

    tilingData.param.rankM = mValue; // 存放用户原始输入的mValue
    tilingData.param.rankN = nValue; // 存放用户原始输入的nValue
    tilingData.param.rankK = kValue; // 存放用户原始输入的kValue
    tilingData.param.aicCoreNum = coreNum;

    args.orgMValue = mValue;
    args.orgNValue = nValue;
    args.orgKValue = kValue;
    args.mValue = mValue;
    args.nValue = nValue;
    args.kValue = kValue;
    args.baseMLimit = -1;
    args.inputDtypeSize = inputDtypeSize;
    args.outputDtypeSize = outputDtypeSize;
    args.aicCoreNum = coreNum;
    args.enablePad = false;
    args.enableSplitK = false;
    args.isBias = isBias;
    args.geAType = aType;
    args.geBType = bType;
    args.geCType = cType;
    args.geBiasType = biasType;
    args.aType = mc2tiling::D_TYPE_MAP.at(aType);
    args.bType = mc2tiling::D_TYPE_MAP.at(bType);
    args.cType = mc2tiling::D_TYPE_MAP.at(cType);
    args.biasType = mc2tiling::D_TYPE_MAP.at(biasType); // 因为bias可能不存在，先采用biasType规避

    // 为通信而进行调整搬运
    if (args.cmdType == mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER) {
        // 先计算出自己的Tiling
        args.rankTileNum = 1; // 1: local matrix not tile
        args.isLocal = true;
        CalcMatmulTiling(args, tilingData.localTiling, tilingData.localL2Tiling);
        if (tilingData.param.rankID == 0) {
            PrintTilingData(tilingData.localTiling);
            PrintTilingData(tilingData.localL2Tiling);
        }
    } else {
      OP_LOGE(context->GetNodeName(), "args.cmdType error %d", static_cast<int>(args.cmdType));
      return ge::GRAPH_FAILED;
    }

    // 本卡一次计算完,其他卡数据按照DR搬运
    if ((tilingData.socParam.commAlg == COMM_ALG_DOUBLE_RING) && (tilingData.socParam.isStep == 1)) {
        args.mValue /= DOUBLE_RING_FACTOR;
        OP_LOGI(context->GetNodeName(), " args.mValue is set to be %lu under double ring + step communication algorithm.",
            args.mValue);
    }

    args.isLocal = false;

    MCSpliteM(context, tilingData, args);

    uint64_t tilingKey = 0U;
    UpdateTilingKey(tilingKey, tilingData, isBias);     // 当前GetTilingKey函数中使用了Mc2Msg结构体，因而无法归一化，此处使用自己的tilingkey计算函数，确保计算逻辑与旧的key保持一致
    OP_LOGD(context->GetNodeName(), "tilingKey is %u, aicCoreNum is %lu.", tilingKey, args.aicCoreNum);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(args.aicCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcMatmulTiling(mc2tiling::TilingArgs& args, ::TCubeTiling& cubeTiling, Mc2Tiling::TileL2Tiling &l2Tiling)
{
    uint64_t mValue = args.mValue;
    uint64_t nValue = args.nValue;
    uint64_t kValue = args.kValue;

    matmul_tiling::MultiCoreMatmulTiling mm;
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args.aType, args.isATrans);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args.bType, args.isBTrans);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args.cType);
    if (args.isBias) {
        mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, args.biasType);
        mm.SetBias(true);
    }
    else {
        mm.SetBias(false);
    }
    mm.SetDim(args.aicCoreNum);
    mm.SetShape(mValue, nValue, kValue);
    mm.SetOrgShape(mValue, nValue, kValue);
    mm.SetBufferSpace(512 * 1024, -1, -1); // 512 * 1024 is buffer size
    mm.SetSingleShape(-1, -1, -1);
    if (nValue == 0) {
        cubeTiling.M = mValue;
        cubeTiling.N = nValue;
        cubeTiling.Ka = kValue;
        cubeTiling.Kb = kValue;
    } else {
        if (mm.GetTiling(cubeTiling) == -1) {
            OP_LOGE("AllGatherMatmul", "mValue %lu, nValue %lu, kValue %lu, aicCoreNum %lu",
                    mValue, nValue, kValue, args.aicCoreNum);
            return ge::GRAPH_FAILED;
        }
    }
    mc2tiling::MatmulFormulaicTiling gatherTiling("AllGatherMatmul");
    gatherTiling.GetCubeTiling(args, cubeTiling, l2Tiling);
    return ge::GRAPH_SUCCESS;
}

static uint64_t GetStorage_a(Mc2Tiling::AllGatherMatmulTilingData& tilingData, mc2tiling::TilingArgs& args)
{
    constexpr uint64_t alignAddrLen = 512;
    auto&& cfg = tilingData.param;
    uint32_t gatherIndex = cfg.gatherIndex;
    uint64_t nd2nzLen = 0;
    uint64_t storageA = 0;

    // step1: ND2NZ
    if (gatherIndex == 0) { // 转置B
        // 计算ND2NZ需使用空间方法保持与MMV3 tiling计算逻辑一致
        uint64_t alignByte = 256 / args.inputDtypeSize;  // 256B 对齐shape
        uint64_t kALign = OpsUtils::CeilAlign(static_cast<uint64_t>(cfg.rankK), alignByte);
        uint64_t nALign = OpsUtils::CeilAlign(static_cast<uint64_t>(cfg.rankN), alignByte);
        nd2nzLen = kALign * nALign * args.inputDtypeSize;
    }
    else {
        auto alignM = cfg.rankM + 16;
        auto alignK = cfg.rankK + 16;
        nd2nzLen = mc2tiling::AlignUp(alignM * alignK * args.inputDtypeSize, alignAddrLen);
    }

    if (args.cmdType == mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER) {
        uint64_t gmcFloat = 0; // allgatherMm 通信后数据只需放在gatherLen对应的workspace或者gatherout中，不需要gmcFloat
        uint64_t gatherLen = 0;
        if (args.isStorageGather == false) {
            if (gatherIndex == 0) { // A矩阵
                gatherLen = mc2tiling::AlignUp(cfg.rankM * cfg.rankK * args.inputDtypeSize, alignAddrLen);
            }
            else {
                gatherLen = mc2tiling::AlignUp(cfg.rankK * cfg.rankN * args.inputDtypeSize, alignAddrLen);
            }
            gatherLen *= cfg.rankDim;
        }

        tilingData.param.nd2NzWorkLen = nd2nzLen;
        tilingData.param.cToFloatLen = gmcFloat;
        tilingData.param.gatherLen = gatherLen;

        storageA = nd2nzLen + gmcFloat + gatherLen; // 需要计算存放的A矩阵
    }
    return storageA;
}

struct HcclAicpuOpParam {
    uint8_t res[64];
};

struct KFCMsgBody {
    // Rank* aiv * MsgSize * sizeof(消息)
    HcclAicpuOpParam msgSndArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
    HcclAicpuOpParam msgRcvArea[mc2tiling::AC_MAX_AIV][mc2tiling::AC_MSG_CNT];
};
struct KFCNotify {
    // 消息通信
    HcclAicpuOpParam msgSend[16]; // 填充16个
    HcclAicpuOpParam msgCnt[16];
};

static ge::graphStatus MC2SetWorkspace(gert::TilingContext* context, Mc2Tiling::AllGatherMatmulTilingData& tilingData,
                                       mc2tiling::TilingArgs& args)
{
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "get workspace failed"),
        return ge::GRAPH_FAILED);
    uint64_t storageA = GetStorage_a(tilingData, args);

    int biasLen = 0;
    if (args.isBias) {
        biasLen = mc2tiling::AlignUp(args.orgNValue, mc2tiling::SHAPE_ALIGN_SIZE) * sizeof(float);
    }
    tilingData.param.biasLen = biasLen;
    workspaces[0] = storageA + 16 * 1024 * 1024 + biasLen; // 16 mb, 1024 * 1024 is 1 mb
    OP_LOGD("AllGatherMatmul", "workspaces[0] size is %ld.", workspaces[0]);
    OP_LOGD("AllGatherMatmul", "biasLen is %d.", biasLen);

    tilingData.param.dataType = static_cast<uint8_t>(mc2tiling::Mc2TilingUtils::GetDataType(args.geAType));

    if (tilingData.param.rankID == 0) {
        OP_LOGD("AllGatherMatmul", "workspace size %ld.", workspaces[0]);

        PrintTilingData(tilingData.param);
        PrintTilingData(tilingData.tileTiling);
        PrintTilingData(tilingData.tileL2Tiling);
        if (tilingData.param.tailM != 0U) {
            OP_LOGD("AllGatherMatmul", "have tail.");
            PrintTilingData(tilingData.tailTiling);
            PrintTilingData(tilingData.tailL2Tiling);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static bool NeedGatherOut(const gert::TilingContext* context) {
  const gert::StorageShape* gatherOut = context->GetOutputShape(GATHEROUT_IDX);
  int64_t mulGatherShape = 1;
  if (gatherOut != nullptr) {
    for (unsigned int i = 0;i < gatherOut->GetStorageShape().GetDimNum(); i++) {
        mulGatherShape = mulGatherShape * gatherOut->GetStorageShape().GetDim(i);
        OP_LOGD("AllGatherMatmul", "gatherOut StorageShape[%d] is %ld", i, gatherOut->GetStorageShape().GetDim(i));
    }
  }

  if (gatherOut == nullptr || mulGatherShape == 0) {
    return false;
  } else {
    return true;
  }
}

static void SetSocParam(Mc2Tiling::AllGatherMatmulTilingData* tilingData, const char* group)
{
  auto commSets = mc2tiling::Mc2TilingUtils::GetCommSets(group);
  tilingData->socParam.isA3 = (commSets == mc2tiling::COMM_MESH) ? 0 : 1; 
  tilingData->socParam.isStep = 0U;   
  tilingData->socParam.isND2NZ = 1U; 
}

static ge::graphStatus InitHcclParam(gert::TilingContext *context, Mc2Tiling::AllGatherMatmulTilingData* tilingData,
                                     const char* group)
{
  std::string algConfig = (tilingData->socParam.isA3 == 0) ?
    "AllGather=level0:fullmesh" : "AllGather=level0:doublering";
  Mc2CcTilingConfig mc2CcTilingConfig(group, tilingData->param.commtype, algConfig);
  uint8_t skipBufferWindowCopy = (tilingData->param.gatherLen == 0) ? 
                                 static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_DEFAULT) :
                                 static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT);
  mc2CcTilingConfig.SetSkipBufferWindowCopy(skipBufferWindowCopy);
  OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
    OP_LOGE(context->GetNodeName(), "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"), return ge::GRAPH_FAILED);
  OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
    OP_LOGE(context->GetNodeName(), "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling failed"), return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AllGatherMatmulTilingFunc(gert::TilingContext *context) {
  // 对参数进行校验
  int index = 0;
  Mc2Tiling::AllGatherMatmulTilingData* tilingData = context->GetTilingData<Mc2Tiling::AllGatherMatmulTilingData>();
  mc2tiling::TilingArgs args;
  auto group = context->GetAttrs()->GetAttrPointer<char>(index++);
  OP_TILING_CHECK(AllGatherParamsCheck(context) != ge::GRAPH_SUCCESS,
                VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "param is invalid"), return ge::GRAPH_FAILED);

  auto isTransA = context->GetAttrs()->GetAttrPointer<bool>(index++);
  auto isTransB = context->GetAttrs()->GetAttrPointer<bool>(index++);
  auto gatherIndex = context->GetAttrs()->GetAttrPointer<int64_t>(index++);
  auto commTurn = *context->GetAttrs()->GetAttrPointer<int64_t>(index++);

  auto rankSize = mc2tiling::MatmulFormulaicTiling::GetRankSize(group);
  OP_TILING_CHECK(commTurn != 0, VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
      "commTurn should be 0, but the actual value is %d.", commTurn), return ge::GRAPH_FAILED);

  OP_LOGD("AllGatherMatmul"," group is %s, rankSize is %u, isTransA is %d, isTransB is %d, gatherIndex is %d,"
          "commTurn is %d.", group, rankSize, *isTransA, *isTransB, *gatherIndex, commTurn);
  tilingData->param.rankDim = rankSize;
  tilingData->param.isTransposeA = isTransA ? *isTransA : 0;
  tilingData->param.isTransposeB = isTransB ? *isTransB : 0;
  tilingData->param.gatherIndex = gatherIndex ? *gatherIndex : 0;
  tilingData->param.commtype = static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER);
  tilingData->param.subtype = 0;
  tilingData->param.storageGather = 0;
  SetSocParam(tilingData, group);

  OP_TILING_CHECK(SetCommAlg(*tilingData) != ge::GRAPH_SUCCESS,
    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), " Set comm algorithm failed."), return ge::GRAPH_FAILED);
  OP_LOGI(context->GetNodeName(), " Communication algorithm is %u.", tilingData->socParam.commAlg);
  
  // distinguish between 910A2 and 910A3
  auto it = std::find(VALID_RANK.at(tilingData->socParam.isA3).begin(),
  VALID_RANK.at(tilingData->socParam.isA3).end(), rankSize);
  OP_TILING_CHECK(it == VALID_RANK.at(tilingData->socParam.isA3).end(),
    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
    "world_size value is %u, which is illegal.", rankSize), return ge::GRAPH_FAILED);

  args.isATrans = isTransA ? *isTransA : 0;
  args.isBTrans = isTransB ? *isTransB : 0;
  args.cmdType = mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER;
  args.rankDim = rankSize;
  args.commTurn = commTurn;
  args.commAlg = tilingData->socParam.commAlg;

  if (NeedGatherOut(context)) {
    args.isStorageGather = true;
    tilingData->param.storageGather = 1;
  } else {
    args.isStorageGather = false;
  }

  SetMatmulTilingAllGatherMatmul(context, *tilingData, args);
  OP_TILING_CHECK(InitHcclParam(context, tilingData, group) != ge::GRAPH_SUCCESS,
    OP_LOGE(context->GetNodeName(), "Tiling InitHcclParam failed."), return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

struct AllGatherMatmulCompileInfo {};
static ge::graphStatus TilingParseForAllGatherMatmul([[maybe_unused]] gert::TilingParseContext *context) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(AllGatherMatmul)
    .Tiling(AllGatherMatmulTilingFunc)
    .TilingParse<AllGatherMatmulCompileInfo>(TilingParseForAllGatherMatmul);
}  // namespace optiling
