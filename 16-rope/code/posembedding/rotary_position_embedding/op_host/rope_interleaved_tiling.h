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
 * \file rope_interleaved.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ROPE_INTERLEAVED_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ROPE_INTERLEAVED_H
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "rotary_position_embedding_tiling.h"

namespace optiling {

class RopeInterLeavedTilingClass : public RotaryPosEmbeddingMembaseTilingClass {
public:
    explicit RopeInterLeavedTilingClass(gert::TilingContext *context) : RotaryPosEmbeddingMembaseTilingClass(context)
    {
    }

    void Reset(gert::TilingContext *context) override
    {
        RotaryPosEmbeddingMembaseTilingClass::Reset(context);
    }
protected:
    bool IsCapable() override
    {
        if ((socVersion_ == platform_ascendc::SocVersion::ASCEND910B ||
             socVersion_ == platform_ascendc::SocVersion::ASCEND310P ||
             socVersion_ == platform_ascendc::SocVersion::KIRINX90 ||
             socVersion_ == platform_ascendc::SocVersion::KIRIN9030) && inputMode_ == MODE_ROTATE_INTERLEAVED) {
            return true;
        }
        return false;
    }
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
};

} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ROPE_INTERLEAVED_H
