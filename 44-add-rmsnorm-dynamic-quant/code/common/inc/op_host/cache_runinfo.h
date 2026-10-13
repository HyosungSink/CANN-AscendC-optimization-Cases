/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_TILING_CACHE_RUN_INFO_H_
#define OPS_BUILT_IN_OP_TILING_CACHE_RUN_INFO_H_

#include <memory>
#include <vector>
#include "../op_api/runtime2_util.h"

namespace optiling {
class CacheTilingContext {
public:
    CacheTilingContext();
    ~CacheTilingContext();
    bool Save(gert::TilingContext& context);
    bool Load(gert::TilingContext& context) const;
    CacheTilingContext& operator=(const CacheTilingContext& rhs);
    CacheTilingContext(const CacheTilingContext& rhs);

private:
    uint64_t tilingKey{0};
    uint32_t numBlocks{0};
    bool atomicCleanFlag{false};
    std::shared_ptr<char> tilingData;
    uint32_t tilingDataSize{0};
    int32_t tilingCond{0};
    size_t workspaceNum{0};
    std::shared_ptr<size_t> workspace;
};

template <typename T>
class GenericHashItem {
public:
    GenericHashItem() {};
    GenericHashItem(const GenericHashItem& obj)
    {
        cacheTilingContext = obj.cacheTilingContext;
        hashInput = obj.hashInput;
    }
    bool SetContext(gert::TilingContext& context, const T& hi)
    {
        if (!cacheTilingContext.Save(context)) {
            return false;
        }
        hashInput = hi;
        return true;
    }
    bool GetContext(gert::TilingContext& context) const
    {
        return cacheTilingContext.Load(context);
    }
    const T& input() const
    {
        return hashInput;
    }
    GenericHashItem& operator=(const GenericHashItem& rhs)
    {
        cacheTilingContext = rhs.cacheTilingContext;
        hashInput = rhs.hashInput;
        return *this;
    }

private:
    CacheTilingContext cacheTilingContext;
    T hashInput;
};
} // namespace optiling
#endif
