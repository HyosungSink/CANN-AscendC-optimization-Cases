/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_FAI_QK_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_FAI_QK_TLA_HPP

#include "../../catlass.hpp"
#include "../../arch/resource.hpp"
#include "../../coord.hpp"
#include "../dispatch_policy.hpp"
#include "../helper.hpp"
#include "../../gemm_coord.hpp"
#include "../tile/tile_copy.hpp"
#include "../tile/tile_mmad.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"

////////////////////////////////////////////////////////////////////

namespace Catlass::Gemm::Block {
////////////////////////////////////////////////////////////////////

template <
    bool PAGED_CACHE_FLAG_,
    bool ENABLE_UNIT_FLAG_,

}
#endif