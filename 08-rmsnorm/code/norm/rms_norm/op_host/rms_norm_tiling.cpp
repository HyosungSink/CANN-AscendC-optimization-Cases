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
 * \file rms_norm_tiling.cpp
 * \brief RmsNorm tiling file
 */
#include <iostream>
#include "op_host/tiling_util.h"
#include "rms_norm_tiling.h"

namespace optiling {
namespace {
constexpr uint32_t DTYPE_KEY_FP16 = 1;
constexpr uint32_t DTYPE_KEY_FP32 = 2;
constexpr uint32_t DTYPE_KEY_BF16 = 3;
constexpr uint32_t UB_FACTOR_B16 = 12288;
constexpr uint32_t GEMMA_UB_FACTOR_B16 = 10240;
constexpr uint32_t GEMMA_UB_FACTOR_B32 = 8192;
constexpr uint32_t UB_FACTOR_B32 = 10240;
constexpr uint32_t BLOCK_ALIGN_NUM = 16;
constexpr uint32_t FLOAT_BLOCK_ALIGN_NUM = 8;
constexpr uint32_t BYTE_SIZE_2_BLOCK_ALIGN_NUM = 16;
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t GAMMA_INDEX = 1;
constexpr uint32_t B16_BYTE_SIZE = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t UB_USED = 1024;
constexpr uint32_t BRCB_RESERVED_UB_USED = 8 * 1024;
constexpr uint32_t UB_COUNTS = 3;
constexpr uint32_t FLOAT_UB_COUNTS = 2;
constexpr uint32_t UB_COUNTS_X_SQX = 2;
constexpr uint32_t UB_COUNTS_GAMMA = 1;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t MASK_64 = 64;
constexpr uint32_t SOC_WEIGHT = 1000;
constexpr uint32_t ALIGN_WEIGHT = 100;
constexpr uint32_t DTYPE_WEIGHT = 10;
constexpr uint32_t DIV_TO_HALF = 2;
constexpr uint32_t SMALL_REDUCE_NUM = 2000;
constexpr uint32_t MODE_MERGE_N = 2;
constexpr uint32_t MODE_SINGLE_ROW = 3;
constexpr size_t MAX_DIM_NUM = 8;
constexpr size_t MIN_DIM_X = 1;
constexpr size_t MIN_DIM_GAMMA = 1;
constexpr size_t SYS_WORKSPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);
constexpr int32_t MOV_2 = 2;
constexpr int32_t MOV_4 = 4;
constexpr int32_t MOV_8 = 8;
constexpr int32_t MOV_16 = 16;
constexpr int32_t PERFORMANC_DIM_ZERO = 0;
constexpr int32_t PERFORMANC_DIM_ONE = 1;
constexpr int32_t PERFORMANC_DIM_TWO = 2;
constexpr int32_t PERFORMANC_DIM_THREE = 3;
constexpr int32_t PERFORMANC_DIM_ONE_MAX = 512;
constexpr int32_t PERFORMANC_DIM_TWO_MAX = 8;
constexpr int32_t PERFORMANC_DIM_THREE_MAX = 5120;

RMSNormTilingData tilingData;

void SetByDtype(ge::DataType dataType, uint32_t& dtypeKey, uint32_t& dataPerBlock)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            dtypeKey = DTYPE_KEY_FP16;
            dataPerBlock = BYTE_SIZE_2_BLOCK_ALIGN_NUM;
            break;
        case ge::DT_BF16:
            dtypeKey = DTYPE_KEY_BF16;
            dataPerBlock = BYTE_SIZE_2_BLOCK_ALIGN_NUM;
            break;
        default:
            dtypeKey = DTYPE_KEY_FP32;
            dataPerBlock = FLOAT_BLOCK_ALIGN_NUM;
            break;
    }
}

uint32_t FindPowerTwo(uint64_t n)
{
    // Set all the bits after the first 1 in the binary of n to 1,
    // then add 1 and shift one bit to the right to find max power of 2 no more than n (32 bit).
    n |= n >> 1; // Set the first digit of n's binary to 1
    n |= n >> MOV_2; // Set the first two bits of n's binary to 1
    n |= n >> MOV_4;
    n |= n >> MOV_8;
    n |= n >> MOV_16; // Set the first 32 bits of n's binary to 1
    n |= n >> BLOCK_SIZE;
    return static_cast<uint32_t>(n + 1) >> 1;
}

void OnceReduceMaxColsAlign64(uint64_t& onceReduceMaxCols, uint32_t& mask, uint32_t& leftNum)
{
    uint32_t nowCols = FindPowerTwo(onceReduceMaxCols);
    leftNum = static_cast<uint32_t>(onceReduceMaxCols) - nowCols;
    while (nowCols > FLOAT_BLOCK_ALIGN_NUM) {
        nowCols = nowCols / DIV_TO_HALF;
    }
    mask = nowCols;
}

} // namespace

static bool CheckInputShape4RmsNorm(const gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* gammaShape = context->GetInputShape(1);
    const gert::StorageShape* yShape = context->GetOutputShape(0);
    const gert::StorageShape* rstdShape = context->GetOutputShape(1);

    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    size_t yDimNum = yShape->GetStorageShape().GetDimNum();
    size_t rstdDimNum = rstdShape->GetStorageShape().GetDimNum();

    OP_TILING_CHECK(
        xDimNum > MAX_DIM_NUM || xDimNum < MIN_DIM_X,
        OP_LOGE(context, "Input x's dim num should not greater than 8 or smaller than 1."),
        return false);
    OP_TILING_CHECK(
        gammaDimNum > MAX_DIM_NUM || gammaDimNum < MIN_DIM_GAMMA,
        OP_LOGE(context, "Input gamma's dim num should not greater than 8 or smaller than 1."),
        return false);
    OP_TILING_CHECK(
        gammaDimNum > xDimNum,
        OP_LOGE(context, "Input gamma's dim num should not greater than input x's."), return false);
    OP_TILING_CHECK(
        xDimNum != yDimNum, OP_LOGE(context, "Input x's dim num must equal to output y's dim num."),
        return false);
    for (uint32_t i = 0; i < gammaDimNum; i++) {
        OP_TILING_CHECK(
            gammaShape->GetStorageShape().GetDim(i) == 0,
            OP_LOGE(context, "Input gamma shape can not be 0."), return false);
        OP_TILING_CHECK(
            gammaShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(xDimNum - gammaDimNum + i),
            OP_LOGE(context, "Input gamma shape invaild, gamma shape is not equal dy last few dim."),
            return false);
    }
    for (uint32_t i = 0; i < xDimNum; i++) {
        OP_TILING_CHECK(
            xShape->GetStorageShape().GetDim(i) == 0, OP_LOGE(context, "Input x shape can not be 0."),
            return false);
        OP_TILING_CHECK(
            yShape->GetStorageShape().GetDim(i) == 0, OP_LOGE(context, "Output y shape can not be 0."),
            return false);
        OP_TILING_CHECK(
            xShape->GetStorageShape().GetDim(i) != yShape->GetStorageShape().GetDim(i),
            OP_LOGE(context, "Output y shape must equal to input x."), return false);
    }
    for (uint32_t i = 0; i < rstdDimNum; i++) {
        OP_TILING_CHECK(
            rstdShape->GetStorageShape().GetDim(i) == 0, OP_LOGE(context, "rstdShape can not be 0."),
            return false);
        OP_TILING_CHECK(
            rstdShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i) &&
                rstdShape->GetStorageShape().GetDim(i) != 1,
            OP_LOGE(context, "Input rstd shape invaild, shape is not equal to xshape first few dim."),
            return false);
    }
    return true;
}

bool CalMixDtypeTiling(uint32_t& modeKey, uint64_t& rowFactor, uint64_t& ubFactor, const RMSNormTilingInfo& rmsTilInfo)
{
    uint64_t ubSize = rmsTilInfo.ubSize;
    uint64_t numColAlign =
        CeilDiv(rmsTilInfo.numCol, static_cast<uint64_t>(BYTE_SIZE_2_BLOCK_ALIGN_NUM)) * BYTE_SIZE_2_BLOCK_ALIGN_NUM;

    // Cal buffer size
    // bufsize(UB) = sizeof(dtype) * num * count
    uint64_t oneRowXYGammaBufSize = static_cast<uint64_t>(B16_BYTE_SIZE) * 2 * numColAlign +
                                    static_cast<uint64_t>(FLOAT_BYTE_SIZE) * 1 * numColAlign;
    uint64_t oneRowRstdBufSize =
        static_cast<uint64_t>(FLOAT_BYTE_SIZE) * 1 * static_cast<uint64_t>(FLOAT_BLOCK_ALIGN_NUM); // rstd
    uint64_t oneRowtmpBufSize = static_cast<uint64_t>(FLOAT_BYTE_SIZE) * 2 * numColAlign;
    uint64_t oneRowReduceBufSize =
        static_cast<uint64_t>(FLOAT_BYTE_SIZE) * 1 * static_cast<uint64_t>(FLOAT_PER_REAPEAT); // buffer for reduce

    uint64_t mutiRowRstdBufSize =
        static_cast<uint64_t>(FLOAT_BYTE_SIZE) * 1U *
        static_cast<uint64_t>(FLOAT_PER_REAPEAT); // rowFactor = FLOAT_PER_REAPEAT, bigger than oneRowRstdBufSize.

    // 1. Mode MergeN
    uint64_t oneRowBufSize = oneRowXYGammaBufSize + oneRowtmpBufSize + oneRowRstdBufSize + oneRowReduceBufSize;
    if (numColAlign <= SMALL_REDUCE_NUM && rmsTilInfo.isSoc910B) {
        modeKey = MODE_MERGE_N;
        rowFactor = ubSize / oneRowBufSize; // oneRowBufSize not be zero, div without check.
        ubFactor = rowFactor * numColAlign;
        return true;
    }

    // 2. Mode Normal
    uint64_t bufSizeNew = oneRowXYGammaBufSize + oneRowtmpBufSize + mutiRowRstdBufSize + oneRowReduceBufSize;
    if (bufSizeNew < ubSize) {
        modeKey = MODE_NORMAL;
        rowFactor = FLOAT_PER_REAPEAT;
        ubFactor = numColAlign;
        return true;
    }

    // 3. Mode SingleRow
    uint64_t oneRowXYGammaBufSizeNew = static_cast<uint64_t>(B16_BYTE_SIZE) * 1 * numColAlign +
                                       static_cast<uint64_t>(FLOAT_BYTE_SIZE) * 1 * numColAlign;
    uint64_t oneRowBufSizeNew = oneRowXYGammaBufSizeNew + oneRowtmpBufSize + mutiRowRstdBufSize + oneRowReduceBufSize;
    if (oneRowBufSizeNew < ubSize) {
        modeKey = MODE_SINGLE_ROW;
        rowFactor = FLOAT_PER_REAPEAT;
        ubFactor = numColAlign;
        return true;
    }

    // 4. Mode SplitD
    modeKey = MODE_SPLIT_D;
    rowFactor = FLOAT_PER_REAPEAT;
    uint64_t oneColSize = static_cast<uint64_t>(B16_BYTE_SIZE) * 2 + static_cast<uint64_t>(FLOAT_BYTE_SIZE) * 3;
    uint64_t mutiRowSumBufSize =
        static_cast<uint64_t>(FLOAT_BYTE_SIZE) * static_cast<uint64_t>(FLOAT_BLOCK_ALIGN_NUM) * rowFactor;
    uint64_t notColBufSizeSum = mutiRowRstdBufSize + oneRowReduceBufSize + mutiRowSumBufSize;
    uint64_t tmpCol;
    if (ubSize > notColBufSizeSum) {
        tmpCol = (ubSize - notColBufSizeSum) / oneColSize; // oneColSize not be zero, div without check.
    } else {
        OP_LOGE("[RmsNorm]", "Cal tiling failed, col less than 0.");
        return false;
    }
    tmpCol = (tmpCol / BLOCK_ALIGN_NUM) * BLOCK_ALIGN_NUM;
    ubFactor = tmpCol;
    return true;
}

static ge::graphStatus Tiling4RmsNorm(gert::TilingContext* context)
{
    OP_TILING_CHECK(
        !CheckInputShape4RmsNorm(context), OP_LOGE(context, "Input shape invalid."),
        return ge::GRAPH_FAILED);
    RMSNormTilingData tiling;
    OP_LOGD(context, " Tiling4RmsNorm");
    auto ptrCompileInfo = reinterpret_cast<const Tiling4RmsNormCompileInfo*>(context->GetCompileInfo());
    uint32_t numCore;
    uint64_t ubSize;
    platform_ascendc::SocVersion curSocVersion;
    if (nullptr == ptrCompileInfo) {
        auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        curSocVersion = ascendc_platform.GetSocVersion();
        numCore = ascendc_platform.GetCoreNumAiv();
        ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    } else {
        numCore = ptrCompileInfo->totalCoreNum;
        ubSize = ptrCompileInfo->totalUbSize;
        curSocVersion = ptrCompileInfo->curSocVersion;
    }
    bool isSoc910B = (curSocVersion == platform_ascendc::SocVersion::ASCEND910B);
    const gert::Shape x_shape = context->GetInputShape(X_INDEX)->GetStorageShape();

    const gert::Shape gamma_shape = context->GetInputShape(GAMMA_INDEX)->GetStorageShape();
    std::string opType(context->GetNodeType());
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* epsilon = attrs->GetFloat(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsilon);
    OP_TILING_CHECK(
        *epsilon < 0, OPS_REPORT_VECTOR_INNER_ERR(context, "Epsilon less than zero, please check."),
        return ge::GRAPH_FAILED);
    uint64_t numCol = gamma_shape.GetShapeSize();
    float avgFactor = (numCol == 0ULL) ? 0.0f : 1.0f / static_cast<float>(numCol);
    size_t xDimNum = x_shape.GetDimNum();
    size_t gammaDimNum = gamma_shape.GetDimNum();
    uint64_t numRow = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        numRow *= x_shape.GetDim(i);
    }

    bool isMixDtype = false;
    auto xDesc = context->GetInputDesc(0);
    auto gammaDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaDesc);
    auto xDataType = xDesc->GetDataType();
    auto gammaDataType = gammaDesc->GetDataType();
    uint32_t xDtypeKey = DTYPE_KEY_FP16;

    size_t usrSize = 256;
    size_t sysWorkspaceSize = 16UL * 1024UL * 1024UL;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    uint32_t SocVersion = 0;
    uint64_t numColAlign = 0;
    uint64_t onceReduceMaxCols;
    uint32_t mask{0};
    uint32_t lastMask{0};
    uint32_t leftNum{0};
    uint32_t lastLeftNum{0};
    uint64_t colAlign = 0;
    uint32_t rstdSize{0};
    uint64_t blockFactor;
    uint64_t latsBlockFactor{0};
    uint64_t ubFactor;
    uint64_t rowFactor;
    uint64_t rowLoop{0};
    uint64_t lastBlockRowLoop{0};
    uint64_t rowTail{0};
    uint64_t lastBlockRowTail{0};
    uint32_t modeKey = MODE_NORMAL;
    uint64_t ubLoop{0};
    uint64_t colBufferLength{0};
    uint32_t multiNNum = 0;
    uint32_t isNddma = 1;
    uint8_t isPerformance = 0;

    ubSize = ubSize - UB_USED;

    uint32_t dataPerBlock;
    SetByDtype(xDataType, xDtypeKey, dataPerBlock);
    if (xDataType != gammaDataType && gammaDataType == ge::DT_FLOAT) {
        // only support x fp16/bf16 gamma fp32
        isMixDtype = true;
    }

    numColAlign = CeilDiv(numCol, static_cast<uint64_t>(dataPerBlock)) * dataPerBlock;
    if (Ops::NN::OpTiling::IsRegbaseSocVersion(context) || curSocVersion == platform_ascendc::SocVersion::MC62CM12A) {
        RMSNormArch35TilingData arch35TilingData;
        return TilingArch354RmsNorm(
            context, numRow, numCol, numCore, ubSize, xDataType, gammaDataType, *epsilon, tiling, arch35TilingData);
    } else if (curSocVersion == platform_ascendc::SocVersion::ASCEND910) {
        SocVersion = 1U;
        colAlign = numColAlign == numCol ? 1ULL : 0ULL;
        int64_t ubDataNum = static_cast<int64_t>(ubSize) / static_cast<int64_t>(FLOAT_BYTE_SIZE);
        if (static_cast<uint64_t>(ubDataNum) > numColAlign) {
            rowFactor = (static_cast<uint64_t>(ubDataNum) - numColAlign) /
                         (static_cast<uint64_t>(BUFFER_NUM) * static_cast<uint64_t>(FLOAT_PER_REAPEAT) +
                          numColAlign * static_cast<uint64_t>(UB_COUNTS_X_SQX) * static_cast<uint64_t>(BUFFER_NUM));
        } else {
            rowFactor = 0ULL;
        }
        blockFactor = CeilDiv(numRow, static_cast<uint64_t>(numCore));
        blockFactor = CeilDiv(blockFactor, static_cast<uint64_t>(FLOAT_BLOCK_ALIGN_NUM)) * FLOAT_BLOCK_ALIGN_NUM;
        blockFactor = std::max(static_cast<uint64_t>(dataPerBlock), blockFactor);
        blockFactor = std::min(blockFactor, numRow);

        if (rowFactor > BYTE_SIZE_2_BLOCK_ALIGN_NUM) {
            rowFactor = rowFactor / BYTE_SIZE_2_BLOCK_ALIGN_NUM * BYTE_SIZE_2_BLOCK_ALIGN_NUM;
        }

        uint32_t useCoreNums = CeilDiv(numRow, blockFactor);
        context->SetBlockDim(useCoreNums);
        ubFactor = 1UL;
        modeKey = 0U;
        if (rowFactor == 0ULL) {
            modeKey = 1U;
            uint32_t bufferFactor = (dataPerBlock / FLOAT_BLOCK_ALIGN_NUM);

            onceReduceMaxCols = static_cast<uint64_t>(bufferFactor) *
                                   (static_cast<uint64_t>(ubDataNum) -
                                    static_cast<uint64_t>(MASK_64) * static_cast<uint64_t>(FLOAT_BLOCK_ALIGN_NUM) -
                                    static_cast<uint64_t>(MASK_64)) /
                                   (static_cast<uint64_t>(UB_COUNTS) +
                                    static_cast<uint64_t>(bufferFactor) * static_cast<uint64_t>(FLOAT_UB_COUNTS)) /
                                   static_cast<uint64_t>(dataPerBlock) * static_cast<uint64_t>(dataPerBlock);
            onceReduceMaxCols = onceReduceMaxCols / dataPerBlock * dataPerBlock;
            ubFactor = CeilDiv(numColAlign, static_cast<uint64_t>(onceReduceMaxCols));
            ubFactor = CeilDiv(numColAlign, static_cast<uint64_t>(ubFactor));
            ubFactor = (ubFactor + static_cast<uint64_t>(dataPerBlock) - 1UL) /
                        static_cast<uint64_t>(dataPerBlock) * static_cast<uint64_t>(dataPerBlock);
            OnceReduceMaxColsAlign64(ubFactor, mask, leftNum);
            uint32_t repeatTime = CeilDiv(numCol, static_cast<uint64_t>(ubFactor));
            uint64_t colTail = numCol - (static_cast<uint64_t>(repeatTime) - 1UL) * ubFactor;
            OnceReduceMaxColsAlign64(colTail, lastMask, lastLeftNum);
        } else {
            onceReduceMaxCols = numCol;
            OnceReduceMaxColsAlign64(onceReduceMaxCols, mask, leftNum);
        }
        rowFactor = std::min(rowFactor, blockFactor);
        rstdSize = static_cast<uint32_t>(rowFactor) * FLOAT_PER_REAPEAT * FLOAT_BYTE_SIZE;
    } else if (isMixDtype) {
        blockFactor = CeilDiv(numRow, static_cast<uint64_t>(numCore));
        uint32_t useCoreNum = CeilDiv(numRow, blockFactor);
        context->SetBlockDim(useCoreNum);
        latsBlockFactor = numRow - blockFactor * (useCoreNum - 1);

        RMSNormTilingInfo rmsNormTilingInfo;
        rmsNormTilingInfo.ubSize = ubSize;
        rmsNormTilingInfo.numCol = numCol;
        rmsNormTilingInfo.numRow = numRow;
        rmsNormTilingInfo.isSoc910B = isSoc910B;
        bool res = CalMixDtypeTiling(modeKey, rowFactor, ubFactor, rmsNormTilingInfo);
        OP_TILING_CHECK(
            !res, OP_LOGE(context, "CalMixDtypeTiling run failed."), return ge::GRAPH_FAILED);
    } else {
        bool dimOK = ((xDimNum == PERFORMANC_DIM_TWO || xDimNum == PERFORMANC_DIM_THREE) && gammaDimNum == PERFORMANC_DIM_ONE);
        bool sizeOk = numCol <= PERFORMANC_DIM_THREE_MAX && ((xDimNum == PERFORMANC_DIM_TWO && x_shape.GetDim(PERFORMANC_DIM_ZERO) <= PERFORMANC_DIM_ONE_MAX) || (xDimNum == PERFORMANC_DIM_THREE && x_shape.GetDim(PERFORMANC_DIM_ZERO) <= PERFORMANC_DIM_ONE_MAX && x_shape.GetDim(PERFORMANC_DIM_ONE) <= PERFORMANC_DIM_TWO_MAX));
        bool dtypeOk = (xDtypeKey == DTYPE_KEY_FP16 || xDtypeKey == DTYPE_KEY_BF16) && !isMixDtype;
        if(dimOK && sizeOk && dtypeOk && curSocVersion == platform_ascendc::SocVersion::ASCEND910B) {
          isPerformance = 1;
        }
        ubFactor = (xDtypeKey == DTYPE_KEY_FP32) ? UB_FACTOR_B32 : UB_FACTOR_B16;
        blockFactor = 1ULL;
        uint64_t tileNum = CeilDiv(numRow, numCore * blockFactor);
        blockFactor *= tileNum;
        uint32_t useCoreNum = CeilDiv(numRow, blockFactor);

        latsBlockFactor = numRow - blockFactor * (useCoreNum - 1);

        rowFactor = FLOAT_PER_REAPEAT;

        if (numColAlign > ubFactor) {
            modeKey = MODE_SPLIT_D;
        }

        if ((numColAlign <= SMALL_REDUCE_NUM && curSocVersion == platform_ascendc::SocVersion::ASCEND910B) || (xDtypeKey == DTYPE_KEY_FP16 && numCol == 128 && curSocVersion == platform_ascendc::SocVersion::ASCEND310P && numRow % BYTE_SIZE_2_BLOCK_ALIGN_NUM == 0)) {
            modeKey = MODE_MERGE_N;
        }

        if (modeKey == MODE_SPLIT_D) {
            uint64_t colTileNum = CeilDiv(numCol, ubFactor);
            ubFactor = CeilDiv(numCol, colTileNum * BLOCK_ALIGN_NUM) * BLOCK_ALIGN_NUM;
            while (numCol % ubFactor != 0UL && numCol % ubFactor < static_cast<uint64_t>(BLOCK_ALIGN_NUM)) {
                ubFactor -= BLOCK_ALIGN_NUM;
                OP_TILING_CHECK(
                    ubFactor == 0,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        context, "Tiling split last dim failed, please check."),
                    return ge::GRAPH_FAILED);
            }
        }
        if (modeKey == MODE_MERGE_N) {
            if (curSocVersion == platform_ascendc::SocVersion::ASCEND310P) {
                ubSize = ubSize - BRCB_RESERVED_UB_USED;
            }
            tiling.set_normal_flag(PERFORMANC_DIM_ZERO);
            uint64_t numColAlignWeight = 16UL;
            rowFactor = ubSize / (numColAlign * numColAlignWeight + 260UL);
            if (curSocVersion == platform_ascendc::SocVersion::ASCEND310P) {
                rowFactor = rowFactor / FLOAT_BLOCK_ALIGN_NUM * FLOAT_BLOCK_ALIGN_NUM; // BroadCast need 32B Align
                blockFactor = (blockFactor + FLOAT_BLOCK_ALIGN_NUM - 1) / FLOAT_BLOCK_ALIGN_NUM * FLOAT_BLOCK_ALIGN_NUM;
                useCoreNum = CeilDiv(numRow, blockFactor);
                latsBlockFactor = numRow - blockFactor * (useCoreNum - 1);
                isPerformance = 1;
            }
            ubFactor = rowFactor * numColAlign;
            OP_TILING_CHECK(
                ubFactor == 0,
                OPS_REPORT_VECTOR_INNER_ERR(context, "Tiling split last dim failed, please check."),
                return ge::GRAPH_FAILED);
        }
        context->SetBlockDim(useCoreNum);
    }
    rowLoop = CeilDiv(blockFactor, rowFactor);
    lastBlockRowLoop = CeilDiv(latsBlockFactor, rowFactor);
    rowTail = blockFactor - (rowLoop - 1) * rowFactor;
    lastBlockRowTail = latsBlockFactor - (lastBlockRowLoop - 1) * rowFactor;

    uint32_t mulLoop = numColAlign / 64;
    uint32_t mulTail = numColAlign - mulLoop * 64;
    uint8_t dstRepStride = numColAlign / 8;
    tiling.set_mul_loop(mulLoop);
    tiling.set_mul_tail(mulTail);
    tiling.set_dst_rep_stride(dstRepStride);

    uint32_t tilingKey = (modeKey == 0U) ? static_cast<uint32_t>(colAlign) * ALIGN_WEIGHT + SocVersion * SOC_WEIGHT +
                                        static_cast<uint32_t>(modeKey) :
                                        static_cast<uint32_t>(modeKey);
    if ((curSocVersion == platform_ascendc::SocVersion::ASCEND910) && (modeKey == 0)) {
        tilingKey = tilingKey + xDtypeKey * DTYPE_WEIGHT;
    }
    if (Ops::NN::OpTiling::IsRegbaseSocVersion(context) ||
        curSocVersion == platform_ascendc::SocVersion::MC62CM12A) {
        tilingKey = SocVersion * SOC_WEIGHT + modeKey;
    }
    context->SetTilingKey(tilingKey);

    tiling.set_is_performance(isPerformance);
    tiling.set_num_row(numRow);
    tiling.set_num_col(numCol);
    tiling.set_num_col_align(numColAlign);

    tiling.set_reduce_mask(mask);
    tiling.set_last_reduce_mask(lastMask);

    tiling.set_block_factor(blockFactor);
    tiling.set_last_block_factor(latsBlockFactor);
    tiling.set_row_loop(rowLoop);
    tiling.set_last_block_row_loop(lastBlockRowLoop);
    tiling.set_row_tail(rowTail);
    tiling.set_last_block_row_tail(lastBlockRowTail);

    tiling.set_row_factor(rowFactor);
    tiling.set_ub_factor(ubFactor);
    tiling.set_left_num(leftNum);
    tiling.set_last_left_num(lastLeftNum);

    tiling.set_epsilon(*epsilon);
    tiling.set_avg_factor(avgFactor);
    tiling.set_rstd_size(rstdSize);
    tiling.set_is_gemma(0);
    tiling.set_ub_loop(ubLoop);
    tiling.set_col_buffer_length(colBufferLength);
    tiling.set_multi_n_num(multiNNum);
    tiling.set_is_nddma(isNddma);
    OP_LOGD(
        context,
        "TilingData numCore: %u, ubSize: %lu, numRow: %lu, numCol: %lu, numColAlign: %lu, col_buffer_length: %u, "
        "blockFactor: %lu, rowFactor: %u, ubFactor: %u, epsilon: %f, avgFactor: %f, "
        "rstdSize: %u, ubLoop: %u, multi_n_num: %u, isNddma: %u.",
        numCore, ubSize, tiling.get_num_row(), tiling.get_num_col(), tiling.get_num_col_align(),
        tiling.get_col_buffer_length(), tiling.get_block_factor(), tiling.get_row_factor(), tiling.get_ub_factor(),
        tiling.get_epsilon(), tiling.get_avg_factor(), tiling.get_rstd_size(), tiling.get_ub_loop(),
        tiling.get_multi_n_num(), tiling.get_is_nddma());

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

void setPlateCoreInfo(gert::TilingContext* context, uint32_t& numCore, uint64_t& ubSize)
{
    auto ptrCompileInfo = reinterpret_cast<const Tiling4RmsNormCompileInfo*>(context->GetCompileInfo());
    if (nullptr == ptrCompileInfo) {
        auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        numCore = ascendc_platform.GetCoreNumAiv();
        ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    } else {
        numCore = ptrCompileInfo->totalCoreNum;
        ubSize = ptrCompileInfo->totalUbSize;
    }
}

static bool setEpsTiling(const gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* epsilon = attrs->GetFloat(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsilon);
    OP_TILING_CHECK(
        *epsilon < 0, OPS_REPORT_VECTOR_INNER_ERR(context, "Epsilon less than zero, please check."),
        return ge::GRAPH_FAILED);
    tilingData.set_epsilon(*epsilon);

    return true;
}

static bool getUbFactor(
    const uint32_t dataPerBlock, const uint32_t xDtypeKey, const uint32_t numCore, gert::TilingContext* context)
{
    const gert::Shape x_shape = context->GetInputShape(X_INDEX)->GetStorageShape();
    const gert::Shape gamma_shape = context->GetInputShape(GAMMA_INDEX)->GetStorageShape();
    std::string opType(context->GetNodeType());

    uint64_t numCol = gamma_shape.GetShapeSize();
    float avgFactor = (static_cast<float>(numCol) == 0.0f) ? 0.0f : (1.0f / static_cast<float>(numCol));
    size_t xDimNum = x_shape.GetDimNum();
    size_t gammaDimNum = gamma_shape.GetDimNum();
    uint64_t numRow = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        numRow *= x_shape.GetDim(i);
    }

    uint64_t numColAlign =
        CeilDiv(numCol, static_cast<uint64_t>(dataPerBlock)) * static_cast<uint64_t>(dataPerBlock);
    uint64_t blockFactor = 1;
    uint64_t ubFactor = (xDtypeKey == DTYPE_KEY_FP32) ? GEMMA_UB_FACTOR_B32 : GEMMA_UB_FACTOR_B16;
    uint32_t modeKey = MODE_NORMAL;

    uint64_t tileNum = CeilDiv(numRow, numCore * blockFactor);
    blockFactor *= tileNum;
    uint32_t useCoreNum = CeilDiv(numRow, blockFactor);

    if (numColAlign > ubFactor) {
        modeKey = MODE_SPLIT_D;
        uint64_t colTileNum = CeilDiv(numCol, ubFactor);
        ubFactor = CeilDiv(numCol, colTileNum * BLOCK_ALIGN_NUM) * BLOCK_ALIGN_NUM;
        while ((numCol % ubFactor) != (static_cast<uint64_t>(0)) &&
               (numCol % ubFactor) < (static_cast<uint64_t>(BLOCK_ALIGN_NUM))) {
            ubFactor -= (static_cast<uint64_t>(BLOCK_ALIGN_NUM));
            if (ubFactor == static_cast<uint64_t>(0)) {
                return false;
            }
        }
    }

    context->SetBlockDim(useCoreNum);
    context->SetTilingKey(modeKey);

    tilingData.set_num_col_align(numColAlign);
    tilingData.set_block_factor(blockFactor);
    tilingData.set_ub_factor(ubFactor);
    tilingData.set_num_row(numRow);
    tilingData.set_num_col(numCol);
    tilingData.set_avg_factor(avgFactor);
    return true;
}

void SetDefaultTiling()
{
    tilingData.set_is_gemma(1);
    tilingData.set_row_factor(FLOAT_PER_REAPEAT);
    tilingData.set_reduce_mask(0);
    tilingData.set_last_reduce_mask(0);
    tilingData.set_rstd_size(0);
    tilingData.set_left_num(0);
    tilingData.set_last_left_num(0);
}

void setWorkSize(gert::TilingContext* context)
{
    size_t usrSize = 256;
    size_t sysWorkspaceSize = SYS_WORKSPACE_SIZE;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
}

static ge::graphStatus Tiling4GemmaRmsNorm(gert::TilingContext* context)
{
    OP_TILING_CHECK(
        !CheckInputShape4RmsNorm(context), OP_LOGE(context, "Input shape invalid."),
        return ge::GRAPH_FAILED);
    OP_LOGD(context, " Tiling4RmsNorm");
    OP_TILING_CHECK(
        !setEpsTiling(context), OP_LOGE(context, "Get attr epsilon error."), return ge::GRAPH_FAILED);

    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto xDataType = xDesc->GetDataType();

    uint32_t numCore;
    uint32_t dataPerBlock;
    uint64_t ubSize;
    setPlateCoreInfo(context, numCore, ubSize);
    ubSize = ubSize - UB_USED;

    uint32_t xDtypeKey = DTYPE_KEY_FP16;
    SetByDtype(xDataType, xDtypeKey, dataPerBlock);
    OP_TILING_CHECK(
        !getUbFactor(dataPerBlock, xDtypeKey, numCore, context),
        OPS_REPORT_VECTOR_INNER_ERR(context, "Tiling split last dim failed, please check."),
        return ge::GRAPH_FAILED);
    SetDefaultTiling();
    setWorkSize(context);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4RmsNorm(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4RmsNorm running.");
    auto compileInfo = context->GetCompiledInfo<Tiling4RmsNormCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->curSocVersion = ascendcPlatform.GetSocVersion();
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->totalUbSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RmsNorm).Tiling(Tiling4RmsNorm).TilingParse<Tiling4RmsNormCompileInfo>(TilingPrepare4RmsNorm);

IMPL_OP_OPTILING(GemmaRmsNorm)
    .Tiling(Tiling4GemmaRmsNorm)
    .TilingParse<Tiling4RmsNormCompileInfo>(TilingPrepare4RmsNorm);

} // namespace optiling
