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
 * \file moe_gating_top_k_softmax_tiling_arch35.cpp
 * \brief
 */
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "../common/include/tiling_base/tiling_templates_registry.h"
#include "../common/include/tiling_base/tiling_type.h"
#include "tiling/tiling_api.h"
#include "moe_gating_top_k_softmax_tiling_base.h"
#include "log/log.h"
#include "../op_kernel/arch35/moe_gating_top_k_softmax_tiling_def.h"
#include "../common/include/tiling_base/tiling_util.h"

using namespace Ops::Transformer::OpTiling;
using namespace AscendC;
using namespace ge;

namespace optiling {

static const uint64_t TILINGKEY_BASE = 100000;
static const uint64_t TILINGKEY_WITH_FINISHED_BASE = 1;
static const uint64_t TILINGKEY_NEED_PAD_BASE = 10;
static const int64_t DEFAULT_WORKSPACE_SIZE = 16777216; // 预留16M空间
static const int64_t ONE_REPEAT_SORT_NUM = 32;
static const int64_t MAX_EXPERT_NUM = 2048;
static const int64_t BLOCK_BYTES = 32;
static const int64_t B32_BLOCK_COUNT = 8;
static const int64_t CONSTANT_TWO = 2;

class MoeGatingTopKSoftmaxRegbaseTiling : public MoeGatingTopKSoftmaxBaseTiling {
public:
    explicit MoeGatingTopKSoftmaxRegbaseTiling(gert::TilingContext *context) : MoeGatingTopKSoftmaxBaseTiling(context)
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    int64_t CalcMaxRowCountInUb();
    void SplitRowInUb(int64_t &perLoopRowCount, int64_t &lastLoopRowCount, int64_t &loopCount, int64_t rowCount,
                      int64_t maxRowCountInUb);

private:
    int64_t rows_{0};
    int64_t needCoreNum_{0};
    int64_t perCoreRowCount_{0};
    int64_t lastCoreRowCount_{0};
    int64_t loopCount_{0};
    int64_t perCoreLoopCount_{0};
    int64_t lastCoreLoopCount_{0};
    int64_t perCorePerLoopRowCount_{0};
    int64_t perCoreLastLoopRowCount_{0};
    int64_t lastCorePerLoopRowCount_{0};
    int64_t lastCoreLastLoopRowCount_{0};
    int64_t expertCount_{0};
    int64_t k_{0};
    int64_t kAlign_{0};
    int64_t expertCountAlign_{0};
    int64_t padNegInfCount_{0};
    int64_t sortRepeatTimes_{0};

    MoeGatingTopKSoftmaxRegbaseTilingData *tilingData_{nullptr};
};

bool MoeGatingTopKSoftmaxRegbaseTiling::IsCapable()
{
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        return false;
    }
    return true;
}


int64_t MoeGatingTopKSoftmaxRegbaseTiling::CalcMaxRowCountInUb()
{
    int64_t maxRowCountInUb = 0;
    int64_t inQueue = static_cast<int64_t>(expertCountAlign_ * sizeof(float) + BLOCK_BYTES);
    int64_t outQueue = static_cast<int64_t>(kAlign_ * sizeof(float) + kAlign_ * sizeof(float) + kAlign_ * sizeof(float));
    int64_t bufQueue = static_cast<int64_t>(expertCountAlign_ * sizeof(float) + expertCountAlign_ * sizeof(float) + expertCountAlign_ * sizeof(float));
    int64_t sortTmpQueue =  static_cast<int64_t>(expertCountAlign_ * sizeof(float) * CONSTANT_TWO * CONSTANT_TWO);
    int64_t sortedQueue = static_cast<int64_t>(expertCountAlign_ * sizeof(float) * CONSTANT_TWO);
    maxRowCountInUb = static_cast<int64_t>((ubSize - sortedQueue) / (inQueue + outQueue + bufQueue + sortTmpQueue));
    OP_LOGD(context_, "maxRowCountInUb is: %ld", maxRowCountInUb);
    return maxRowCountInUb;
}

void MoeGatingTopKSoftmaxRegbaseTiling::SplitRowInUb(int64_t &perLoopRowCount, int64_t &lastLoopRowCount,
                                                     int64_t &loopCount, int64_t rowCount, int64_t maxRowCountInUb)
{
    loopCount = (rowCount + maxRowCountInUb - 1) / maxRowCountInUb;
    perLoopRowCount = (rowCount + loopCount - 1) / loopCount;
    lastLoopRowCount = rowCount - (loopCount - 1) * perLoopRowCount;
    OP_LOGD(context_, "rowCount is: %ld, perLoopRowCount is: %ld, lastLoopRowCount is: %ld, loopCount is: %ld",
            rowCount, perLoopRowCount, lastLoopRowCount, loopCount);
}

ge::graphStatus MoeGatingTopKSoftmaxRegbaseTiling::DoOpTiling()
{
    OP_CHECK_IF((col > MAX_EXPERT_NUM),
                OP_LOGE(context_->GetNodeName(), "expert count (=%u) is too large, please check.", col),
                return ge::GRAPH_FAILED);
    rows_ = static_cast<int64_t>(row);
    perCoreRowCount_ = static_cast<int64_t>((rows_ + aivNum - 1) / aivNum);
    needCoreNum_ = (rows_ + perCoreRowCount_ - 1) / perCoreRowCount_;
    lastCoreRowCount_ = rows_ - (needCoreNum_ - 1) * perCoreRowCount_;
    expertCount_ = static_cast<int64_t>(col);
    k_ = k;
    kAlign_ = (k_ + B32_BLOCK_COUNT - 1) / B32_BLOCK_COUNT * B32_BLOCK_COUNT;
    expertCountAlign_ = (expertCount_ + ONE_REPEAT_SORT_NUM - 1) / ONE_REPEAT_SORT_NUM * ONE_REPEAT_SORT_NUM;
    padNegInfCount_ = expertCount_ % ONE_REPEAT_SORT_NUM;
    sortRepeatTimes_ = expertCountAlign_ / ONE_REPEAT_SORT_NUM;

    int64_t maxRowCountInUb = CalcMaxRowCountInUb();
    SplitRowInUb(perCorePerLoopRowCount_, perCoreLastLoopRowCount_, perCoreLoopCount_, perCoreRowCount_,
                 maxRowCountInUb);
    SplitRowInUb(lastCorePerLoopRowCount_, lastCoreLastLoopRowCount_, lastCoreLoopCount_, lastCoreRowCount_,
                 maxRowCountInUb);
    OP_LOGD(context_, "needCoreNum is: %ld", needCoreNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxRegbaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeGatingTopKSoftmaxRegbaseTiling::GetTilingKey() const
{
    uint64_t tilingKey = TILINGKEY_BASE;
    if (hasFinished) {
        tilingKey += TILINGKEY_WITH_FINISHED_BASE;
    }
    if (padNegInfCount_ > 0) {
        tilingKey += TILINGKEY_NEED_PAD_BASE;
    }
    return tilingKey;
}

ge::graphStatus MoeGatingTopKSoftmaxRegbaseTiling::GetWorkspaceSize()
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    OP_LOGD(context_, "workspace size is: %ld", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKSoftmaxRegbaseTiling::PostTiling()
{
    tilingData_ = context_->GetTilingData<MoeGatingTopKSoftmaxRegbaseTilingData>();
    tilingData_->rows = rows_;
    tilingData_->perCoreRowCount = perCoreRowCount_;
    tilingData_->lastCoreRowCount = lastCoreRowCount_;
    tilingData_->perCoreLoopCount = perCoreLoopCount_;
    tilingData_->lastCoreLoopCount = lastCoreLoopCount_;
    tilingData_->perCorePerLoopRowCount = perCorePerLoopRowCount_;
    tilingData_->perCoreLastLoopRowCount = perCoreLastLoopRowCount_;
    tilingData_->lastCorePerLoopRowCount = lastCorePerLoopRowCount_;
    tilingData_->lastCoreLastLoopRowCount = lastCoreLastLoopRowCount_;
    tilingData_->expertCount = expertCount_;
    tilingData_->k = k_;
    tilingData_->kAlign = kAlign_;
    tilingData_->expertCountAlign = expertCountAlign_;
    tilingData_->padNegInfCount = padNegInfCount_;
    tilingData_->sortRepeatTimes = sortRepeatTimes_;
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(needCoreNum_);

    OP_LOGD(context_,
            "tilingdata rows is: %ld, perCoreRowCount is: %ld, lastCoreRowCount is: %ld, perCoreLoopCount is: %ld, "
            "lastCoreLoopCount is: %ld, perCorePerLoopRowCount is: %ld, perCoreLastLoopRowCount is: %ld, "
            "lastCorePerLoopRowCount is: %ld, lastCoreLastLoopRowCount is: %ld, lastCoreLoopCountexpertCount is: %ld, "
            "k is: "
            "%ld, kAlign is: %ld, expertCountAlign is: %ld, padNegInfCount  is: %ld sortRepeatTimes is: %ld",
            tilingData_->rows, tilingData_->perCoreRowCount, tilingData_->lastCoreRowCount,
            tilingData_->perCoreLoopCount, tilingData_->lastCoreLoopCount, tilingData_->perCorePerLoopRowCount,
            tilingData_->perCoreLastLoopRowCount, tilingData_->lastCorePerLoopRowCount,
            tilingData_->lastCoreLastLoopRowCount, tilingData_->expertCount, tilingData_->k, tilingData_->kAlign,
            tilingData_->expertCountAlign, tilingData_->padNegInfCount, tilingData_->sortRepeatTimes);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MoeGatingTopKSoftmax, MoeGatingTopKSoftmaxRegbaseTiling, 100);
} // namespace optiling