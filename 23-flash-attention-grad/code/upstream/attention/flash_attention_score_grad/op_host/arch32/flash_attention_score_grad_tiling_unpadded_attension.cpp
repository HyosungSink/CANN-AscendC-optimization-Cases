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
 * \file flash_attention_score_grad_tiling_unpadded_attension.cc
 * \brief
 */

#include "flash_attention_score_grad_tiling_s1s2_bn2gs1s2.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {

class FlashAttentionScoreGradTilingUnpaddedAttension : public FlashAttentionScoreGradTilingS1s2Bn2gs1s2 {
public:
    explicit FlashAttentionScoreGradTilingUnpaddedAttension(gert::TilingContext *context)
        : FlashAttentionScoreGradTilingS1s2Bn2gs1s2(context)
    {
    }

    bool IsCapable() override
    {
        auto sinkShape = context_->GetOptionalInputShape(SINK_IN);
        if (sinkShape != nullptr && sinkShape->GetStorageShape().GetDimNum() == 1 ) {
            return false;
        }
        const char *tndSoftmaxIn = context_->GetAttrs()->GetAttrNum() > static_cast<size_t>(TND_SOFTMAX_IN) ? context_->GetAttrs()->GetAttrPointer<char>(TND_SOFTMAX_IN) : "";
        if (strcmp(tndSoftmaxIn, "") != 0) return false;

        if (tnd2bsh) {
            OP_LOGI(context_, "FlashAttentionScoreGradTilingUnpaddedAttension is not support tnd to bsh.");
            return false;
        }

        auto actualSeqQLenTensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_LEN);
        auto actualSeqKVLenTensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_KV_LEN);
        bool isTND = actualSeqQLenTensor != nullptr && actualSeqQLenTensor->GetShapeSize() > 0 &&
                     actualSeqKVLenTensor != nullptr && actualSeqKVLenTensor->GetShapeSize() > 0;
        if (isTND) {
            if (!isTndSABHit(context_)) {
                OP_LOGI(context_, "TND layout FlashAttentionScoreGradTilingUnpaddedAttension hit.");
                return true;
            }
        }

        return false;
    };
};

REGISTER_TILING_TEMPLATE_WITH_ARCH(
    FlashAttentionScoreGrad, FlashAttentionScoreGradTilingUnpaddedAttension,
    std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}),
    2000);

} // namespace optiling
