/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file grouped_matmul_finalize_routing_base_tiling.cc
 * \brief
 */
#include "grouped_matmul_finalize_routing_base_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"
#include "../common/include/err/ops_err.h"

namespace optiling {
namespace grouped_matmul_finalize_routing {
constexpr uint32_t BEST_BASE_M = 128;
constexpr uint32_t BEST_BASE_K = 128;
constexpr uint32_t BEST_BASE_N = 256;
constexpr uint32_t BEST_VBASE_M = 16;
constexpr uint32_t BEST_VBASE_BIG_M = 32;
constexpr uint32_t DETER_WORK_SPACE_SIZE = 96 * 1024 * 1024; // Deterministic worksize is 96M or 64M
constexpr uint32_t DETER_WORK_SPACE_LOWER_SIZE = 64 * 1024 * 1024;


constexpr uint64_t RPC_WORKSIZE = 20;
constexpr uint64_t MB_SIZE = 1024 * 1024;
constexpr uint64_t ONE_BATCH_DIM = 1;
constexpr uint64_t TWO_BATCH_DIM = 2;
constexpr uint64_t THREE_BATCH_DIM = 3;
constexpr uint64_t FOUR_BATCH_DIM = 4;
constexpr uint64_t GROUPLIST_INDEX = 5;
constexpr uint64_t SHARE_INPUT_INDEX = 6;
constexpr uint64_t LOGIT_INDEX = 7;
constexpr uint64_t ROW_INDEX_INDEX = 8;
constexpr uint64_t OFFSET_INDEX = 9;
constexpr uint64_t BATCH_INDEX = 5;
constexpr uint64_t ATTR_TUNINGCONFIG_INDEX = 7;
constexpr uint64_t SHARED_INPUT_OFFSET_INDEX = 2;
constexpr uint64_t PERTOKEN_SCALE_INDEX = 4;
constexpr uint64_t BIAS_SCALE_INDEX = 3;
constexpr uint64_t SCALE_INPUT_INDEX = 2;
constexpr uint32_t UBCALSIZE = 16 * 256;  // for vector compute
constexpr uint32_t MAX_K_A8W4_MSD = 18432;  // k is limited by pre process, a line of X should be able to put in UB
constexpr uint32_t A8W4_UBRESTBYTES = 9 * 32 * 256;  // for vector compute
constexpr uint32_t A8W8_UBRESTBYTES = 101376;  // for vector compute
constexpr uint32_t EIGHT = 8;
constexpr uint32_t AVG_M_THREHOLD = 128;
constexpr uint32_t AVG_M_BIG_THREHOLD = 256;
constexpr uint32_t A8W4_MSD_BASE_M_DEFAULT = 16;
constexpr uint32_t A8W4_MSD_BASE_N_DEFAULT = 512;
constexpr uint32_t A8W4_MSD_SMALLM_BASE_M = 64;
constexpr uint32_t A8W4_MSD_BIGM_BASE_M = 128;
constexpr uint32_t A8W4_MSD_SMALLM_BASE_N = 512;
constexpr uint32_t A8W4_MSD_BIGM_BASE_N = 256;
constexpr uint32_t A8W4_MSD_BASE_K = 256;
constexpr uint32_t CV_PARALL_NUM = 4;
constexpr uint32_t ONE_BLK_SIZE = 32;
constexpr uint32_t ROW_INDEX_FACTOR = 10;
constexpr uint32_t SCALE_FACTOR = 100;
constexpr uint32_t SHARED_INPUT_FACTOR = 10000;
constexpr uint32_t GROUP_LIST_TYPE_INDEX = 6;
constexpr uint32_t GROUP_LIST_TYPE_FACTOR = 1000;

constexpr uint32_t A8W4_L1OPT_MAX_K = 2048;
constexpr uint32_t A8W4_L1OPT_SMALLM_BASE_M = 96;
constexpr uint32_t A8W4_L1OPT_SMALLM_BASE_N = 128;
constexpr uint32_t A8W4_L1OPT_BIGM_BASE_M = 128;
constexpr uint32_t A8W4_L1OPT_BIGM_BASE_N = 128;
constexpr uint32_t A8W4_L1OPT_BASE_K = 512;

static ge::graphStatus GetInputDims(const gert::Shape &storageShape, ge::Format format, int64_t (&dims)[TWO_BATCH_DIM])
{
    const size_t dimNum = storageShape.GetDimNum();
    if (format == ge::FORMAT_ND) {
        if (dimNum < TWO_BATCH_DIM) {
            return ge::GRAPH_FAILED;
        }
        dims[0] = storageShape[dimNum - TWO_BATCH_DIM];
        dims[1] = storageShape[dimNum - ONE_BATCH_DIM];
    } else {
        if (dimNum < FOUR_BATCH_DIM) {
            return ge::GRAPH_FAILED;
        }
        dims[0] = storageShape[dimNum - THREE_BATCH_DIM] * storageShape[dimNum - TWO_BATCH_DIM];
        dims[1] = storageShape[dimNum - FOUR_BATCH_DIM] * storageShape[dimNum - ONE_BATCH_DIM];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulFinalizeRoutingBaseTiling::GetInputShape()
{
    auto inputXDesc = context_->GetInputDesc(0);
    auto inputWDesc = context_->GetInputDesc(1);
    int64_t mkDims[2];
    int64_t knDims[2];

    if (context_ == nullptr || context_->GetInputShape(0) == nullptr || context_->GetInputShape(1) == nullptr || inputXDesc == nullptr
        || inputWDesc == nullptr) {
        OP_LOGE(context_->GetNodeName(), "invalid input pointer");
        return ge::GRAPH_FAILED;
    }

    if ((GetInputDims(context_->GetInputShape(0)->GetStorageShape(), inputXDesc->GetStorageFormat(), mkDims) != ge::GRAPH_SUCCESS) ||
        (GetInputDims(context_->GetInputShape(1)->GetStorageShape(), inputWDesc->GetStorageFormat(), knDims) != ge::GRAPH_SUCCESS)) {
        OP_LOGE(context_->GetNodeName(), "invalid input dim num");
        return ge::GRAPH_FAILED;
    }

    if (context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX) == nullptr) {
        hasPertokenScale_ = 0;
    } else {
        hasPertokenScale_ = 1;
    }

    if (context_->GetOptionalInputDesc(BIAS_SCALE_INDEX) == nullptr) {
        hasBias_ = 0;
    } else {
        hasBias_ = 1;
    }

    k_ = static_cast<uint64_t>(mkDims[1]);
    m_ = static_cast<uint64_t>(mkDims[0]);
    n_ = static_cast<uint64_t>(knDims[1]);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulFinalizeRoutingBaseTiling::ParseAttr()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    // 存在sharedInput为空的情况
    if (context_->GetOptionalInputDesc(SHARE_INPUT_INDEX) != nullptr &&
        attrs->GetAttrPointer<int>(SHARED_INPUT_OFFSET_INDEX) != nullptr) {
        sharedInputOffset_ = *attrs->GetAttrPointer<int>(SHARED_INPUT_OFFSET_INDEX);
    } else {
        sharedInputOffset_ = 0;
    }
    
    if (attrs->GetAttrPointer<float>(1) != nullptr) {
        residualScale_ = *attrs->GetAttrPointer<float>(1);
    } else {
        OP_LOGE(context_->GetNodeName(), "Attr RESIDUAL_SCALE is None.");
        return ge::GRAPH_FAILED;
    }

    if (attrs->GetAttrPointer<int>(BATCH_INDEX) != nullptr) {
        batch_ = *attrs->GetAttrPointer<int>(BATCH_INDEX);
    } else {
        OP_LOGE(context_->GetNodeName(), "Attr BATCH is None.");
        return ge::GRAPH_FAILED;
    }

    useL1OptKernel_ = false; // 确定性场景默认不使用L1Opt
    if (context_->GetDeterministic() == 0) { // 非确定性场景进一步判断tuningConfig
        const auto tuningConfigPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_TUNINGCONFIG_INDEX);
        if (tuningConfigPtr != nullptr && tuningConfigPtr->GetSize() > 0) {
            tuningConfig_ = (reinterpret_cast<const int64_t *>(tuningConfigPtr->GetData()))[0];
            tuningConfig_ = (tuningConfig_ > 0) ? tuningConfig_ : 0;
            useL1OptKernel_ = (tuningConfigPtr->GetSize() > 1 && k_ <= A8W4_L1OPT_MAX_K) ? true : false;
        } else {
            tuningConfig_ = 0;
            useL1OptKernel_ = false;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulFinalizeRoutingBaseTiling::ParseInputAndAttr()
{
    uint64_t ubSize, l1Size, l0CSize;

    if (GetInputShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "get input shape failed");
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get platform info failed");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);
    mm_.SetBufferSpace(l1Size, l0CSize, ubSize);
    blockDim_ = ascendcPlatform.GetCoreNumAic();

    if (context_->GetOptionalInputDesc(SHARE_INPUT_INDEX) != nullptr) {
        sharedInputLen_ = context_->GetOptionalInputShape(SHARE_INPUT_INDEX)->GetStorageShape()[0];
    } else {
        sharedInputLen_ = 0;
    }

    withOffset_ = 0;
    if (context_->GetOptionalInputDesc(OFFSET_INDEX) != nullptr) {
        withOffset_ = 1U; // with offset set 1
    }

    if (ParseAttr() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Parse attr failed");
        return ge::GRAPH_FAILED;
    }

    groupNum_ = context_->GetInputShape(1)->GetStorageShape()[0];
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulFinalizeRoutingBaseTiling::W4A8BaseTilingProcess()
{
    uint32_t singleN = 256;
    uint32_t singleM = 128;
    size_t userWorkspaceSize = (CV_PARALL_NUM * blockDim_ * singleN * singleM * sizeof(int32_t) * EIGHT) + m_ * sizeof(float);
    size_t systemWorkspaceSize = RPC_WORKSIZE * MB_SIZE;

    auto wFormat0 = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(0)->GetStorageFormat()));
    bool wNZ = (wFormat0 == ge::FORMAT_FRACTAL_NZ);

    quantGroupNum_ = context_->GetOptionalInputShape(SCALE_INPUT_INDEX)->GetStorageShape()[1];
    ubRestBytes_ = A8W4_UBRESTBYTES;

    uint32_t baseM = A8W4_MSD_BASE_M_DEFAULT;
    uint32_t baseN = A8W4_MSD_BASE_N_DEFAULT;
    uint32_t avg_m = (tuningConfig_ != 0) ? tuningConfig_  : ((groupNum_ != 0) ? (m_ / groupNum_) : 1);
    OP_LOGD(context_->GetNodeName(), "GroupedMatmulFinalizeRoutingBaseTiling tuningConfig is %ld, avg_m is %u",
            tuningConfig_, avg_m);
    baseM = avg_m < AVG_M_THREHOLD ? A8W4_MSD_SMALLM_BASE_M : A8W4_MSD_BIGM_BASE_M;
    baseN = avg_m < AVG_M_THREHOLD ? A8W4_MSD_SMALLM_BASE_N : A8W4_MSD_BIGM_BASE_N;

    if (baseN > n_) {
        baseN = Ops::Base::CeilAlign(n_, uint64_t(ONE_BLK_SIZE));
    }

    if (baseN == 0) {
        OP_LOGE(context_->GetNodeName(), "GroupedMatmulFinalizeRoutingBaseTiling: baseN is 0! Tling Failed!");
        return ge::GRAPH_FAILED;
    }

    vBaseM_ = UBCALSIZE / baseN;
    mm_.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4, false);
    if (wNZ) {
        mm_.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT4, false);
    } else {
        mm_.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4, false);
    }
    mm_.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    mm_.SetBias(false);
    mm_.SetOrgShape(baseM, n_, k_);
    mm_.SetShape(baseM, n_, k_);
    mm_.SetFixSplit(baseM, baseN, A8W4_MSD_BASE_K);
    if (mm_.GetTiling(tilingData_.matmulTiling) == -1){
        OP_LOGE(context_->GetNodeName(), "GroupedMatmulFinalizeRoutingBaseTiling Get Tiling Failed!"
             "m, n, k: %lu, %lu, %lu", m_, n_, k_);
        return ge::GRAPH_FAILED;
    }

    if (k_ > MAX_K_A8W4_MSD) {
        OP_LOGE(context_->GetNodeName(), "GMM_tiling: K should be less than 18432 on the A8W4 scenario, but now is %lu", k_);
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context_->GetNodeName(), "GMM_tiling: baseM is %d, baseK is %d, baseN is %d.",
        baseM, A8W4_MSD_BASE_K, baseN);

    // key 11···UL for A8W4
    tilingKey_ = 11000000000000000011UL;

    workspaceSize_ = userWorkspaceSize + systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulFinalizeRoutingBaseTiling::W4A8L1OptTilingProcess()
{
    uint32_t singleN = 1024;
    uint32_t singleM = 128;
    size_t userWorkspaceSize = (blockDim_ * singleN * singleM * sizeof(int32_t) * EIGHT) + m_ * sizeof(float);
    size_t systemWorkspaceSize = RPC_WORKSIZE * MB_SIZE;

    auto wFormat0 = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(0)->GetStorageFormat()));
    bool wNZ = (wFormat0 == ge::FORMAT_FRACTAL_NZ);

    quantGroupNum_ = context_->GetOptionalInputShape(SCALE_INPUT_INDEX)->GetStorageShape()[1];
    ubRestBytes_ = A8W4_UBRESTBYTES;

    uint32_t avg_m = (tuningConfig_ != 0) ? tuningConfig_  : ((groupNum_ != 0) ? (m_ / groupNum_) : 1);
    OP_LOGD(context_->GetNodeName(), "GroupedMatmulFinalizeRoutingBaseTiling tuningConfig is %ld, avg_m is %u",
            tuningConfig_, avg_m);
    uint32_t baseM = avg_m < AVG_M_THREHOLD ? A8W4_L1OPT_SMALLM_BASE_M : A8W4_L1OPT_BIGM_BASE_M;
    uint32_t baseN = avg_m < AVG_M_THREHOLD ? A8W4_L1OPT_SMALLM_BASE_N : A8W4_L1OPT_BIGM_BASE_N;

    if (baseN > n_) {
        baseN = Ops::Base::CeilAlign(n_, uint64_t(ONE_BLK_SIZE));
    }

    if (baseN == 0) {
        OP_LOGE(context_->GetNodeName(), "GroupedMatmulFinalizeRoutingBaseTiling: baseN is 0! Tling Failed!");
        return ge::GRAPH_FAILED;
    }

    vBaseM_ = UBCALSIZE / baseN;
    mm_.SetAType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT4, false);
    if (wNZ) {
        mm_.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT4, false);
    } else {
        mm_.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4, false);
    }
    mm_.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    mm_.SetBias(false);
    mm_.SetOrgShape(baseM, n_, k_);
    mm_.SetShape(baseM, n_, k_);
    mm_.SetSingleShape(baseM, singleN, k_);
    mm_.SetFixSplit(baseM, baseN, A8W4_L1OPT_BASE_K);
    if (mm_.GetTiling(tilingData_.matmulTiling) == -1){
        OP_LOGE(context_->GetNodeName(), "GroupedMatmulFinalizeRoutingBaseTiling Get Tiling Failed!"
             "m, n, k: %lu, %lu, %lu", m_, n_, k_);
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context_->GetNodeName(), "GMM_tiling: baseM is %d, baseK is %d, baseN is %d.",
        baseM, A8W4_L1OPT_BASE_K, baseN);

    // key 11···UL for A8W4
    tilingKey_ = 11000000000000000111UL;

    workspaceSize_ = userWorkspaceSize + systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulFinalizeRoutingBaseTiling::W4A8TilingProcess()
{
    if (useL1OptKernel_) {
        W4A8L1OptTilingProcess();
    } else {
        W4A8BaseTilingProcess();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulFinalizeRoutingBaseTiling::W8A8TilingProcess()
{
    size_t userWorkspaceSize = CV_PARALL_NUM * 256 * 128 * sizeof(int32_t) * blockDim_;
    size_t systemWorkspaceSize = RPC_WORKSIZE * MB_SIZE;
    ubRestBytes_ = A8W8_UBRESTBYTES;
    
    uint32_t baseM = BEST_BASE_M;
    uint32_t baseN = BEST_BASE_N;
    // When n=7168/7680 and k=2048, we use tuningConfig_ to select more optimal baseM/baseN.
    if ((n_ == 7168 || n_ == 7680) && k_ == 2048) {
        uint32_t avg_m = (tuningConfig_ != 0) ? tuningConfig_  : ((groupNum_ != 0) ? (m_ / groupNum_) : 1);
        OP_LOGD(context_->GetNodeName(), "GroupedMatmulFinalizeRoutingBaseTiling(A8W8) tuningConfig is %ld, avg_m is %u",
                tuningConfig_, avg_m);
        baseM = (avg_m > AVG_M_THREHOLD && avg_m <= AVG_M_BIG_THREHOLD) ? BEST_BASE_N : BEST_BASE_M;
        baseN = (avg_m > AVG_M_THREHOLD && avg_m <= AVG_M_BIG_THREHOLD) ? BEST_BASE_M : BEST_BASE_N;
    }
    vBaseM_ = UBCALSIZE / baseN;
    mm_.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, false);
    mm_.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT8, false);
    mm_.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);

    mm_.SetBias(false);
    mm_.SetDim(1);
    mm_.SetShape(baseM, n_, k_);
    mm_.SetOrgShape(baseM, n_, k_);
    mm_.SetFixSplit(baseM, baseN, BEST_BASE_K);
    if (mm_.GetTiling(tilingData_.matmulTiling) == -1) {
        OP_LOGE(context_->GetNodeName(), "GroupedMatmulFinalizeRoutingBaseTiling Get Tiling Failed!, m, n, k: %lu, %lu, %lu", m_, n_, k_);
        return ge::GRAPH_FAILED;
    }
    
    // row_index类型
    auto rowIndexDesc = context_->GetOptionalInputDesc(ROW_INDEX_INDEX);
    auto rowIndexDtype = rowIndexDesc != nullptr ? rowIndexDesc->GetDataType() : ge::DT_INT64;
    tilingKey_ = 10000000000000000001UL;
    if (rowIndexDtype == ge::DT_INT32)
    {
        tilingKey_ += ROW_INDEX_FACTOR;
    }

    auto scaleDesc = context_->GetOptionalInputDesc(SCALE_INPUT_INDEX);
    auto scaleDtype = scaleDesc != nullptr ? scaleDesc->GetDataType() : ge::DT_FLOAT;
    if (scaleDtype == ge::DT_BF16)
    {
        tilingKey_ += SCALE_FACTOR;
    }

    workspaceSize_ = userWorkspaceSize + systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

void GroupedMatmulFinalizeRoutingBaseTiling::OtherSettingTilingProcess()
{
    // 判断sharedInput是否为空
    if (context_->GetOptionalInputDesc(SHARE_INPUT_INDEX) == nullptr) {
        tilingKey_ += SHARED_INPUT_FACTOR;
    }
    auto attrs = context_->GetAttrs();
    if (*attrs->GetAttrPointer<int64_t>(GROUP_LIST_TYPE_INDEX) == 0) {
        tilingKey_ += GROUP_LIST_TYPE_FACTOR;
    }
}

void GroupedMatmulFinalizeRoutingBaseTiling::DeterministicTilingProcess()
{
    if (context_->GetDeterministic() == 0) {
        deterministicFlag_ = 0;
        return;
    }
    deterministicFlag_ = 1;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t l2_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2_size);
    deterWorkspaceSize_ = l2_size > DETER_WORK_SPACE_SIZE ? DETER_WORK_SPACE_SIZE : DETER_WORK_SPACE_LOWER_SIZE;
    workspaceSize_ += deterWorkspaceSize_;
}

void GroupedMatmulFinalizeRoutingBaseTiling::FillTilingData()
{
    tilingData_.matmulTiling.set_dbL0C(1);
    tilingData_.matmulTiling.set_stepKa(4);  // 4: L1中左矩阵单次搬运基于baseK的4倍数据
    tilingData_.matmulTiling.set_stepKb(4);  // 4: L1中右矩阵单次搬运基于baseK的4倍数据
    tilingData_.matmulTiling.set_depthA1(8);  // 8: stepKa的两倍，开启double buffer
    tilingData_.matmulTiling.set_depthB1(8);  // 8: stepKb的两倍，开启double buffer
    tilingData_.matmulTiling.set_stepM(1);
    tilingData_.matmulTiling.set_stepN(1);

    tilingData_.set_coreNum(blockDim_);
    tilingData_.set_groupNum(groupNum_);
    tilingData_.set_batch(batch_);
    tilingData_.set_totalInGroup(m_);
    tilingData_.set_k(k_);
    tilingData_.set_n(n_);
    tilingData_.set_vBaseM(vBaseM_);           // 16: vBaseM
    tilingData_.set_ubCalSize(UBCALSIZE);  // 16: vector每次计算的行数，256: 每次计算的列数，与cube baseN保持一致
    tilingData_.set_parallNum(CV_PARALL_NUM);
    tilingData_.set_sharedInputOffset(sharedInputOffset_);
    tilingData_.set_sharedInputLen(sharedInputLen_);
    tilingData_.set_residualScale(residualScale_);
    tilingData_.set_quantGroupNum(quantGroupNum_);
    tilingData_.set_ubRestBytes(ubRestBytes_);  // 126976: 除分配给TQue外剩余给TBuf的大小为126976
    tilingData_.set_withOffset(withOffset_);
    tilingData_.set_hasPertokenScale(hasPertokenScale_);
    tilingData_.set_hasBias(hasBias_);
    tilingData_.set_deterministicFlag(deterministicFlag_);
    tilingData_.set_deterWorkspaceSize(deterWorkspaceSize_);
}

void GroupedMatmulFinalizeRoutingBaseTiling::FillTilingDataL1Opt()
{
    tilingData_.matmulTiling.set_dbL0C(2); // double buffer dbL0C 2
    tilingData_.matmulTiling.set_stepKa(1);
    tilingData_.matmulTiling.set_stepKb(4);  // 4: L1中右矩阵单次搬运基于baseK的4倍数据
    tilingData_.matmulTiling.set_depthA1(1);
    tilingData_.matmulTiling.set_depthB1(8);  // 8: stepKb的两倍，开启double buffer
    tilingData_.matmulTiling.set_stepM(1);
    tilingData_.matmulTiling.set_stepN(1);
}

void GroupedMatmulFinalizeRoutingBaseTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "blockDim: [%d]", tilingData_.get_coreNum());
    OP_LOGD(context_->GetNodeName(), "groupNum: [%u]", tilingData_.get_groupNum());
    OP_LOGD(context_->GetNodeName(), "batch: [%u]", tilingData_.get_batch());
    OP_LOGD(context_->GetNodeName(), "totalInGroup: [%u]", tilingData_.get_totalInGroup());
    OP_LOGD(context_->GetNodeName(), "k: [%u]", tilingData_.get_k());
    OP_LOGD(context_->GetNodeName(), "n: [%u]", tilingData_.get_n());
    OP_LOGD(context_->GetNodeName(), "vBaseM: [%u]", tilingData_.get_vBaseM());
    OP_LOGD(context_->GetNodeName(), "ubCalSize: [%u]", tilingData_.get_ubCalSize());
    OP_LOGD(context_->GetNodeName(), "parallNum: [%u]", tilingData_.get_parallNum());
    OP_LOGD(context_->GetNodeName(), "sharedInputOffset: [%u]", tilingData_.get_sharedInputOffset());
    OP_LOGD(context_->GetNodeName(), "sharedInputLen: [%u]", tilingData_.get_sharedInputLen());
    OP_LOGD(context_->GetNodeName(), "residualScale: [%.2f]", tilingData_.get_residualScale());
    OP_LOGD(context_->GetNodeName(), "quantGroupNum: [%u]", tilingData_.get_quantGroupNum());
    OP_LOGD(context_->GetNodeName(), "ubRestBytes: [%u]", tilingData_.get_ubRestBytes());
    OP_LOGD(context_->GetNodeName(), "hasPertokenScale: [%u]", tilingData_.get_hasPertokenScale());
    OP_LOGD(context_->GetNodeName(), "hasBias: [%u]", tilingData_.get_hasBias());
    OP_LOGD(context_->GetNodeName(), "deterministicFlag: [%u]", tilingData_.get_deterministicFlag());
    OP_LOGD(context_->GetNodeName(), "deterWorkspaceSize: [%u]", tilingData_.get_deterWorkspaceSize());
}

ge::graphStatus GroupedMatmulFinalizeRoutingBaseTiling::DoOpTiling()
{
    failFlag_ = false;
    auto inputXDesc = context_->GetInputDesc(0);
    auto inputWDesc = context_->GetInputDesc(1);
    if (inputXDesc == nullptr || inputWDesc == nullptr) {
        OP_LOGE(context_->GetNodeName(), "invalid input pointer");
        return ge::GRAPH_FAILED;
    }

    if (ParseInputAndAttr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (inputXDesc->GetDataType() == ge::DT_INT8 && inputWDesc->GetDataType() == ge::DT_INT4) {
        W4A8TilingProcess();
    } else {
        W8A8TilingProcess();
    }

    DeterministicTilingProcess();
    OtherSettingTilingProcess();

    FillTilingData();
    if (inputXDesc->GetDataType() == ge::DT_INT8 && inputWDesc->GetDataType() == ge::DT_INT4 && useL1OptKernel_) {
        FillTilingDataL1Opt();
    }
    PrintTilingData();

    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulFinalizeRoutingBaseTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus GroupedMatmulFinalizeRoutingBaseTiling::PostTiling()
{
    OP_CHECK_IF(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        OP_LOGE(context_->GetNodeName(), "tiling data size[%zu] is not aligned to 8", tilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(tilingData_.get_coreNum());
    context_->SetScheduleMode(1);
    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
        return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    if (failFlag_) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

}
}
