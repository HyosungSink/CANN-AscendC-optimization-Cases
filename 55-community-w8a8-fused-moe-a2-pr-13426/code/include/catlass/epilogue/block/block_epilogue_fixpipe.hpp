/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_FIXPIPE_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_FIXPIPE_HPP

#include "../../catlass.hpp"
#include "../../arch/resource.hpp"
#include "../dispatch_policy.hpp"
#include "../../gemm_coord.hpp"
#include "../../matrix_coord.hpp"
#include "../tile/tile_copy.hpp"
#include "../../gemm/helper.hpp"
#include "../tile/copy_ub_to_gm_tla.hpp"

namespace Catlass::Epilogue::Block {

template <
    class L0TileShape_,
    class ElementOut_,
    class ElementIn_,
    bool SPLIT_M_
>
class BlockEpilogue<
    EpilogueAscend950Fixpipe<SPLIT_M_>,
    L0TileShape_,
    ElementOut_,
    ElementIn_
> {
public:
    using DispatchPolicy = EpilogueAscend950Fixpipe<SPLIT_M_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOut = ElementOut_;
    using ElementIn = ElementIn_;

    static constexpr bool SPLIT_M = DispatchPolicy::SPLIT_M;
    static constexpr int64_t ML0_ = tla::get<0>(L0TileShape_{});
    static constexpr int64_t NL0_ = tla::get<1>(L0TileShape_{});

    CATLASS_DEVICE 
    BlockEpilogue() = default;

    template <class TensorOut, class TensorIn>
    CATLASS_DEVICE 
    void operator()(TensorOut& tensorOut, TensorIn& tensorIn)
    {
        if constexpr (!SPLIT_M && AscendC::GetSubBlockIdx() > 0) {
            return;
        }

        int64_t blockShapeM = tla::get<0>(tensorOut.shape());
        int64_t blockShapeN = tla::get<1>(tensorOut.shape());
        int64_t halfBlockShapeM = CeilDiv(blockShapeM, AscendC::GetTaskRation());
        if constexpr (SPLIT_M) {
            blockShapeM = (static_cast<uint64_t>(blockShapeM) & 1UL) > 0UL ? 
                (halfBlockShapeM - AscendC::GetSubBlockIdx()) : halfBlockShapeM;
        } 
        
        // real copy data size
        int64_t copySize = blockShapeM * blockShapeN;
        if (copySize <= 0) {
            return;
        }

        auto tensorTileOut = GetTile(
            tensorOut,
            tla::MakeCoord(halfBlockShapeM * (AscendC::GetSubBlockIdx() & 0x1), 0),
            tla::MakeShape(blockShapeM, blockShapeN)
        );

        using CopyUbToGm = Epilogue::Tile::CopyUb2GmTla<ArchTag, TensorIn, decltype(tensorTileOut)>;
        CopyUbToGm copyUbToGm;
        copyUbToGm(tensorTileOut, tensorIn);
    }
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_EPILOGUE_FIXPIPE_HPP
