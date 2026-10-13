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
 * \file swi_glu_tiling.cc
 * \brief
 */
#include "swi_glu_tiling.h"
#include "error_util.h"
#include "op_host/tiling_util.h"
#include <chrono>
#include "../../swi_glu_grad/op_host/swi_glu_grad_tiling_regbase.h"
#include "register/op_impl_registry.h"

namespace optiling {
using namespace Ops::NN::OpTiling;

const uint32_t UB_RESERVED_BUFF = 0; // reserve 0k
const uint32_t L2_CACHE_LINE_SIZE = 512; // pack unit in cache 512B
const uint32_t UB_MIN_BLOCK_SIZE = 32; // align unit in cache 32B
const uint32_t BLOCK_SIZE_OF_64B = 64; // align unit in cache 64B
const uint32_t DEFAULT_BUFFER_NUM = 2;
const uint32_t MAX_BLOCK_COUNT = 4095; // datacopy指令包含的连续传输数据块的最大个数
const uint32_t MAX_BLOCK_LEN = static_cast<uint32_t>(65535 * 32); // datacopy指令每个连续传输数据块的最长长度为65535，单位为32bytes
const uint32_t MAX_UINT32 = 4294967295;
const uint16_t DISCONTINE_COPY_MAX_BLOCKCNT = 4095; // 非连续拷贝，blockCount最大值,AscendC接口限制
const uint16_t DISCONTINE_COPY_MAX_BLOCKLEN = 65535; // 非连续拷贝，blockLen最大值,AscendC接口限制
const uint16_t DISCONTINE_COPY_MAX_STRIDE = 65535; // 非连续拷贝，srcStride/dstStride最大值,AscendC接口限制

const uint32_t XXGLU_TQUE_NUM = 3;
const uint32_t SWIGLU_TBUF_NUM_HALF = 2;
const uint32_t SWIGLU_TBUF_NUM_BF16 = 2;
const uint32_t SWIGLU_TBUF_NUM_FLOAT = 1;
const uint32_t XXGLU_BW_TQUE_NUM = 5;
const uint32_t SWIGLU_BW_TBUF_NUM_FLOAT = 2;
const uint32_t SWIGLU_BW_TBUF_NUM_BF16 = 5;
const uint32_t SWIGLU_BW_TBUF_NUM_HALF = 5;
const int64_t WORKSPACE_BUFFER = static_cast<int64_t>(16 * 1024 * 1024);

const int64_t INPUT_OUTPUT_IDX = 0;
const int64_t TILING_KEY_EMPTY = 100;

template<typename T>
inline auto AlignUp(T num, T rnd) -> T
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}
// align num to multiples of rnd, round down
template<typename T>
inline auto AlignDown(T num, T rnd) -> T
{
    return ((((rnd) == 0) || ((num) < (rnd))) ? 0 : ((num) / (rnd) * (rnd)));
}
template<typename T>
inline auto DivCeil(T num, T div) -> T
{
    return (((div) == 0) ? 0 : (((num) + (div) - 1) / (div)));
}

enum GLU_FLAG : uint8_t {
    SWIGLU_SINGLE,
    SWIGLU_GRAD_SINGLE
};

// Tiling优选参数
struct GluSingleTilingOptParam {
    // Maximum amount of data that can be transferred by an operator UB at a time. Unit:element
    uint32_t maxTileLen = 0;
    uint32_t optBaseRowLen = 0; // 最优的BaseRowLen
    uint32_t optBaseColLen = 0; // 最优的BaseColLen
    uint64_t optTotalTileNum = 0; // 最优的分割后的数据块数量
    uint64_t optBaseSize = 0; // 最优的分割后的base shape数据块的大小， optBaseRowLen*optBaseColLen, Unit:element
    uint64_t optBaseTileNum = 0; // 最优的分割后的base shape数据块数量，不包含尾块

    uint32_t totalUsedCoreNum = 0; // 最终实际使用的核数
    uint64_t tileNumPerCore = 0; // 每个核需要处理的TileNum，如果不均匀，按照多的计算
};

struct GluSingleTilingCalculator {
public:
    explicit GluSingleTilingCalculator(SwiGluTilingData *iTilingData, const std::string& opName) : tilingData(iTilingData), opName_(opName) {}

    template<GLU_FLAG Glu_Flag>
    bool CalcTiling(uint32_t totalCore, uint64_t ubSize, int32_t dtype, NpuArch npuArch);

    template <GLU_FLAG Glu_Flag>
    bool SetTotalShape(const gert::Shape &inShape, const int32_t inDim);

    void SaveTilingData(gert::TilingContext *context);
    inline bool isSupportSocV(uint32_t dtype, NpuArch npuArch) const;
    SwiGluTilingData *tilingData;

    uint32_t totalUsedCoreNum_ = 0; // 最终实际使用的核数
private:
    template<GLU_FLAG Glu_Flag, uint16_t bufferNum>
    bool CalcOptTiling(uint64_t ubSize, int32_t dtype, GluSingleTilingOptParam &optTiling) const;

    template<GLU_FLAG Glu_Flag, uint16_t bufferNum>
    bool GetBufferNumAndDataLenPerUB(uint64_t ubSize, int32_t dtype, uint64_t &dataLenPerUB) const;

    template<GLU_FLAG Glu_Flag, uint16_t bufferNum>
    inline bool CalcUbMaxTileLen(uint64_t ubSize, int32_t dtype, GluSingleTilingOptParam &optTiling) const;

    inline void SaveOptBaseShape(uint32_t baseRowLen_, uint32_t baseColLen_, GluSingleTilingOptParam &optTiling) const;

    inline uint32_t getBaseColLenUpBound(GluSingleTilingOptParam &optTiling) const;

    inline uint32_t getBaseRowLenUpBound() const;

    inline bool MustBeSingleBaseRowLen(uint32_t baseColLen_) const;

    inline bool isInvalidBaseShape(uint32_t baseRowlen_, uint32_t baseColLen_) const;
    inline bool isValidTailCol(uint32_t baseRowlen_, uint32_t baseColLen_) const;

    template<GLU_FLAG Glu_Flag>
    inline bool CalcOptBaseShape(GluSingleTilingOptParam &optTiling) const;

    uint32_t inputDTypeLen = 2;
    // Indicates the minimum processing data unit of the UB. Unit:element.
    // Formula: 32B/sizeof(DType). For example, if Dtype is BF16, ubMinBlockLen = 32/2 = 16
    uint32_t ubMinBlockLen = 0;
    // Length of the L2 cache line. Unit:element.
    // Formula: 512B/sizeof(DType). For example, if the Dtype is BF16, cacheLineLen = 512/2 = 256
    uint32_t cacheLineLen = 0;
    // baseColLen aligned package Len. elelment:Unit. 512-byte alignment or 32-byte alignment
    uint32_t alignPackLen = 0;
    uint32_t totalAvailableCore = 0; // total avaliable core in device

    std::string opName_;
};

inline bool GetLengthByType(int32_t dtype, uint32_t& dsize) {
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
        case ge::DT_BF16:
            dsize = sizeof(int16_t);
            return true;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            dsize = sizeof(int32_t);
            return true;
        case ge::DT_DOUBLE:
        case ge::DT_INT64:
        case ge::DT_UINT64:
            dsize = sizeof(int64_t);
            return true;
        default:
            return false;
    }
}

    template <GLU_FLAG Glu_Flag>
    inline uint32_t GetSelfIdx()
    {
        auto idx = Glu_Flag == GLU_FLAG::SWIGLU_GRAD_SINGLE ? 1 : 0;
        return static_cast<uint32_t>(idx);
    }

    template <GLU_FLAG Glu_Flag>
    inline bool GluSingleTilingCalculator::SetTotalShape(const gert::Shape& inShape, const int32_t inDim)
    {
        int64_t shapeBefore = 1;
        int64_t shapeAfter = 1;
        int64_t dimNum = inShape.GetDimNum();
        if (inDim < -dimNum || inDim >= dimNum) {
            OP_LOGE(opName_, "SetTotalShape Unsupported inDim %d", inDim);
            return false;
        }
        int64_t splitDim = inDim < 0 ? dimNum + inDim : inDim; // inDim default -1
        for (int64_t i = 0; i < splitDim; i++) {
            shapeBefore *= inShape.GetDim(i);
        }
        for (int64_t j = splitDim; j < dimNum; j++) {
            shapeAfter *= inShape.GetDim(j);
        }
        // 如果shape不是2的倍数,返回
        if (shapeAfter % 2 != 0) {
            OP_LOGE(opName_, "SetTotalShape Unsupported inDim %d, shapeAfter %ld %% 2 != 0", inDim, shapeAfter);
            return false;
        }

        tilingData->set_rowLen(shapeBefore);
        // colLen为原shape除以2
        tilingData->set_colLen(shapeAfter / 2);
        return true;
    }

    template <GLU_FLAG Glu_Flag, uint16_t bufferNum>
    bool GluSingleTilingCalculator::GetBufferNumAndDataLenPerUB(uint64_t ubSize, int32_t dtype, uint64_t& dataLenPerUB) const
    {
        uint32_t singleDataSize = 0;
        switch (Glu_Flag) {
        case GLU_FLAG::SWIGLU_SINGLE:
            if (dtype == ge::DT_FLOAT16) {
                singleDataSize = static_cast<uint32_t>(static_cast<uint64_t>(bufferNum) * XXGLU_TQUE_NUM * sizeof(int16_t) + SWIGLU_TBUF_NUM_HALF * sizeof(int32_t));
            } else if (dtype == ge::DT_BF16) {
                singleDataSize = static_cast<uint32_t>(static_cast<uint64_t>(bufferNum) * XXGLU_TQUE_NUM * sizeof(int16_t) + SWIGLU_TBUF_NUM_BF16 * sizeof(int32_t));
            } else {
                singleDataSize = static_cast<uint32_t>(static_cast<uint64_t>(bufferNum) * XXGLU_TQUE_NUM * sizeof(int32_t) + SWIGLU_TBUF_NUM_FLOAT * sizeof(int32_t));
            }
            dataLenPerUB = ubSize / singleDataSize;
            return true;
        case GLU_FLAG::SWIGLU_GRAD_SINGLE:
            if (dtype == ge::DT_FLOAT16) {
            singleDataSize = static_cast<uint32_t>(static_cast<uint64_t>(bufferNum) * XXGLU_BW_TQUE_NUM * sizeof(int16_t) +
                             SWIGLU_BW_TBUF_NUM_HALF * sizeof(int32_t));
            } else if (dtype == ge::DT_BF16) {
            singleDataSize = static_cast<uint32_t>(static_cast<uint64_t>(bufferNum) * XXGLU_BW_TQUE_NUM * sizeof(int16_t) +
                             SWIGLU_BW_TBUF_NUM_BF16 * sizeof(int32_t));
            } else {
            singleDataSize = static_cast<uint32_t>(static_cast<uint64_t>(bufferNum) * XXGLU_BW_TQUE_NUM * sizeof(int32_t) +
                             SWIGLU_BW_TBUF_NUM_FLOAT * sizeof(int32_t));
            }
            dataLenPerUB = ubSize / singleDataSize;
            return true;
        default:
            return false;
    }
}

template <GLU_FLAG Glu_Flag, uint16_t bufferNum>
inline bool GluSingleTilingCalculator::CalcUbMaxTileLen(uint64_t ubSize, int32_t dtype, GluSingleTilingOptParam& optTiling) const
{
    // get buffernum and maxTileLen
    uint64_t maxTileLenPerUB = 1;
    if (!GetBufferNumAndDataLenPerUB<Glu_Flag, bufferNum>(ubSize, dtype, maxTileLenPerUB)) {
        OP_LOGE(opName_, "CalcTiling Get bufferNum %d and maxTileLenPerUB %lu failed", bufferNum, maxTileLenPerUB);
        return false;
    }
    optTiling.maxTileLen = AlignDown<uint64_t>(maxTileLenPerUB, ubMinBlockLen);
    OP_LOGI(opName_, "CalcTiling ubSize:%lu, bufferNum:%d, maxTileLenPerUB:%u", ubSize, bufferNum, optTiling.maxTileLen);
    return true;
}

inline void GluSingleTilingCalculator::SaveOptBaseShape(uint32_t baseRowLen_, uint32_t baseColLen_, GluSingleTilingOptParam& optTiling) const
{
    uint64_t totalTileNum = DivCeil<uint64_t>(tilingData->get_rowLen(), (baseRowLen_)) * DivCeil<uint64_t>(tilingData->get_colLen(), (baseColLen_));
    uint64_t baseSize = static_cast<uint64_t>(baseRowLen_) * baseColLen_;
    if (baseRowLen_ == static_cast<uint32_t>(0) || baseColLen_ == static_cast<uint32_t>(0)) {
        OP_LOGE(opName_, "SaveOptBaseShape devide by 0 baseRowLen_:%u baseColLen_:%u", baseRowLen_, baseColLen_);
        return;
    }
    uint64_t baseTileNum = (tilingData->get_rowLen() / baseRowLen_) * (tilingData->get_colLen() / baseColLen_);
    uint32_t totalUsedCoreNum = std::min(totalTileNum, static_cast<uint64_t>(totalAvailableCore));
    if ((optTiling.optTotalTileNum == static_cast<uint64_t>(0)) ||
        (totalUsedCoreNum > optTiling.totalUsedCoreNum) ||
        ((totalUsedCoreNum == optTiling.totalUsedCoreNum) && (totalTileNum < optTiling.optTotalTileNum)) ||
        ((totalUsedCoreNum == optTiling.totalUsedCoreNum) && (totalTileNum == optTiling.optTotalTileNum) && (baseSize > optTiling.optBaseSize)) ||
        ((totalUsedCoreNum == optTiling.totalUsedCoreNum) && (totalTileNum == optTiling.optTotalTileNum) && (baseSize == optTiling.optBaseSize) && (baseTileNum > optTiling.optBaseTileNum))) {
        optTiling.optBaseRowLen = baseRowLen_;
        optTiling.optBaseColLen = baseColLen_;
        optTiling.optTotalTileNum = totalTileNum;
        optTiling.optBaseSize = baseSize;
        optTiling.optBaseTileNum = baseTileNum;
        optTiling.totalUsedCoreNum = totalUsedCoreNum;
        optTiling.tileNumPerCore = DivCeil<uint64_t>(totalTileNum, totalUsedCoreNum);;
    }
}

inline uint32_t GluSingleTilingCalculator::getBaseColLenUpBound(GluSingleTilingOptParam& optTiling) const
{
    uint32_t upBound = std::min(tilingData->get_colLen(), static_cast<uint64_t>(optTiling.maxTileLen));
    if (tilingData->get_is32BAligned() == 1) {
        upBound = std::min(upBound, static_cast<uint32_t>(DISCONTINE_COPY_MAX_BLOCKLEN));
    } else {
        upBound = std::min(upBound, static_cast<uint32_t>(DISCONTINE_COPY_MAX_BLOCKLEN / inputDTypeLen));
    }

    if (upBound < tilingData->get_colLen() && upBound > cacheLineLen) {
        // 该种场景，每一个colLen至少被切割成2块，需要保证baseColLen为512B整数倍才高效
        return AlignDown<uint32_t>(upBound, cacheLineLen);
    } else {
        return upBound;
    }
}

inline uint32_t GluSingleTilingCalculator::getBaseRowLenUpBound() const
{
    return std::min(tilingData->get_rowLen(), static_cast<uint64_t>(DISCONTINE_COPY_MAX_BLOCKCNT));
}

/**
 * colLen 32B对齐时：若(colLen * 2 – baseCloLen) > 65535* ubMinBlockLen，则baseRowLen=1
 * colLen非32B对齐时：若(colLen * 2 – baseCloLen) * sizeof(Dtype) > 65535，则baseRowLen=1
 */
inline bool GluSingleTilingCalculator::MustBeSingleBaseRowLen(uint32_t baseColLen_) const
{
    if (tilingData->get_is32BAligned() == 1) {
        // colLen 32B对齐时：若(colLen * 2 – baseCloLen) > 65535* ubMinBlockLen，则baseRowLen=1
        return ((tilingData->get_colLen() * 2 - baseColLen_) > (DISCONTINE_COPY_MAX_STRIDE * ubMinBlockLen));
    } else {
        // colLen非32B对齐时：若(colLen * 2 – baseCloLen) * sizeof(Dtype) > 65535，则baseRowLen=1
        return (((tilingData->get_colLen() * 2 - baseColLen_) * inputDTypeLen) > DISCONTINE_COPY_MAX_STRIDE);
    }
}

/**
 * 若则baseRowLen大于1，且约束判决baseRowLen必须等于1时，则是不合法的
 * colLen 32B对齐时：若(colLen * 2 – baseCloLen) > 65535* ubMinBlockLen，则baseRowLen=1
 * colLen非32B对齐时：若(colLen * 2 – baseCloLen) * sizeof(Dtype) > 65535，则baseRowLen=1
 */
inline bool GluSingleTilingCalculator::isInvalidBaseShape(uint32_t baseRowlen_, uint32_t baseColLen_) const
{
    return ((baseRowlen_ < static_cast<uint32_t>(1)) || (baseRowlen_ > static_cast<uint32_t>(1) && MustBeSingleBaseRowLen(baseColLen_)));
}

inline bool GluSingleTilingCalculator::isValidTailCol(uint32_t baseRowlen_, uint32_t baseColLen_) const
{
    if (baseColLen_ == static_cast<uint32_t>(0)) {
        return false;
    }
    uint32_t tailColLen = tilingData->get_colLen() % baseColLen_;
    return !(baseRowlen_ > static_cast<uint32_t>(1) && MustBeSingleBaseRowLen(tailColLen));
}

inline bool GluSingleTilingCalculator::isSupportSocV(uint32_t dtype, NpuArch npuArch) const
{
    if ((npuArch == NpuArch::DAV_2002) && (dtype == ge::DT_BF16)) {
        return false; //310p dont support BF16
    } else {
        return true;
    }
}
template <GLU_FLAG Glu_Flag>
inline bool GluSingleTilingCalculator::CalcOptBaseShape(GluSingleTilingOptParam& optTiling) const
{
    uint32_t baseColLen_ = getBaseColLenUpBound(optTiling);
    OP_LOGI(opName_, "CalcOptBaseShape init baseColLen : %u", baseColLen_);
    if (MustBeSingleBaseRowLen(baseColLen_)) {
        SaveOptBaseShape(static_cast<uint32_t>(1), baseColLen_, optTiling);
        return true;
    }

    while(true) {
        // colLen非32B对齐时，数据copy到ub时，每一行的尾部会补齐32B
        uint32_t baseRowlen_ = std::min(optTiling.maxTileLen / AlignUp<uint32_t>(baseColLen_, ubMinBlockLen), getBaseRowLenUpBound());
        if (isInvalidBaseShape(baseRowlen_, baseColLen_)) {
            OP_LOGI(opName_, "CalcOptBaseShape baseRowln:%u or baseColLen:%u is invalid. optTotalTileNum:%lu end",
                    baseRowlen_, baseColLen_, optTiling.optTotalTileNum);
            // optTotalTileNum有效，则前面有最优解，返回true;否则返回false
            return (optTiling.optTotalTileNum > static_cast<uint64_t>(0));
        }
        // 保存较优的base shape
        if (isValidTailCol(baseRowlen_, baseColLen_)) {
            SaveOptBaseShape(baseRowlen_, baseColLen_, optTiling);
        }

        // baseColLen已经到达下限 或者 baseRowlen已经达到上限，无法继续调整，结束
        if (baseColLen_ <= alignPackLen || (baseRowlen_ >= getBaseRowLenUpBound())) {
            return true; // baseColLen无法继续调整了，结束
        }
        // 继续调整baseColLen
        // baseColLen为若alignPackLen的整数倍，则baseColLen减少1个alignPackLen的长度
        // 否则baseColLen减少到alignPackLen的整数倍（最接近的）
        if (baseColLen_ % alignPackLen == static_cast<uint32_t>(0)) {
            baseColLen_ -= alignPackLen;
        } else {
            baseColLen_ = AlignDown<uint32_t>(baseColLen_, alignPackLen);
        }
    }
}

// 如果开启double buffer，则bufferNum为2，否则为1
template <GLU_FLAG Glu_Flag, uint16_t bufferNum>
bool GluSingleTilingCalculator::CalcOptTiling(uint64_t ubSize, int32_t dtype, GluSingleTilingOptParam& optTiling) const
{
    // 计算maxTilingLen
    if (!CalcUbMaxTileLen<Glu_Flag, bufferNum>(ubSize, dtype, optTiling)) {
        return false;
    }
    // 计算最优的base块形状
    if (!CalcOptBaseShape<Glu_Flag>(optTiling)) {
        return false;
    }
    return true;
}

template <GLU_FLAG Glu_Flag>
bool GluSingleTilingCalculator::CalcTiling(uint32_t totalCore, uint64_t ubSize, int32_t dtype,  NpuArch npuArch)
{
    totalAvailableCore = totalCore;
    if (!GetLengthByType(dtype, inputDTypeLen)) {
        OP_LOGI(opName_, "CalcTiling Unsupported input data type %d", dtype);
        return false;
    }
    ubMinBlockLen = UB_MIN_BLOCK_SIZE / inputDTypeLen; // min block size
    cacheLineLen = L2_CACHE_LINE_SIZE / inputDTypeLen; // bandwidth max efficiency
    alignPackLen = cacheLineLen; // 默认512对齐，策略可调整
    OP_LOGI(opName_, "CalcTiling GetLengthByType:%u ubMinBlockLen:%u cacheLineLen:%u alignPackLen:%u", inputDTypeLen, ubMinBlockLen, cacheLineLen, alignPackLen);
    // Is 32-byte aligned for split colLen?
    tilingData->set_is32BAligned(tilingData->get_colLen() % ubMinBlockLen == 0);
    // 310p not support Non-64B
    uint32_t blockSizeOf64B = BLOCK_SIZE_OF_64B / inputDTypeLen;
    if (((npuArch == NpuArch::DAV_2002)) && (tilingData->get_colLen() % blockSizeOf64B != 0)) {
        OP_LOGE(opName_, "input shape is not support Non-64B aligned");
        return false;
    }
    // 先计算开启double buffer的tiling参数
    tilingData->set_isDoubleBuffer(1);
    GluSingleTilingOptParam optTilingDb;
    // 判断buffer = 2时是否计算成功
    if (!CalcOptTiling<Glu_Flag, 2>(ubSize, dtype, optTilingDb)) {
        return false;
    }
    GluSingleTilingOptParam *optTiling = &optTilingDb;
    // 如果double buffer开启的tiling参数中，每个核需要处理的tileNum等于2，尝试关闭double buffer;
    // 若关闭double buffer后只需要搬运1次数据，且使用的核没有减少, 则使用关闭double buffer的tiling
    // 判断tileNumPerCoer是否为2
    if (optTilingDb.tileNumPerCore == static_cast<uint64_t>(2)) {
        GluSingleTilingOptParam optTilingNoDb;
        if (CalcOptTiling<Glu_Flag, 1>(ubSize, dtype, optTilingNoDb) &&
            (optTilingNoDb.tileNumPerCore == static_cast<uint64_t>(1)) && (optTilingNoDb.totalUsedCoreNum >= optTilingDb.totalUsedCoreNum)) {
            optTiling = &optTilingNoDb;
            tilingData->set_isDoubleBuffer(0);
        }
    }
    // 记录最优的结果
    tilingData->set_baseRowLen(optTiling->optBaseRowLen);
    tilingData->set_baseColLen(optTiling->optBaseColLen);
    totalUsedCoreNum_ = optTiling->totalUsedCoreNum;
    OP_LOGI(opName_, "CalcTilingRES baseRowLen:%u baseColLen:%u", optTiling->optBaseRowLen, optTiling->optBaseColLen);
    return true;
}

inline static int64_t CeilDiv(int64_t value, int64_t factor) {
    int64_t valueNum = 0;
    if (factor == 0) {
        return value;
    }
    if (value % factor == 0) {
        valueNum = value / factor;
    } else {
        valueNum = value / factor + 1;
    }
    return valueNum;
}

static ge::graphStatus TilingPrepare4SwiGlu(gert::TilingParseContext* /*context*/) {
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus processEmptyTensor(gert::TilingContext* context, const uint32_t& totalCore)
{
    SwiGluTilingData tilingData;
    auto outputShape = context->GetOutputShape(INPUT_OUTPUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    auto yTotalLength = outputShape->GetStorageShape().GetShapeSize();
    OP_TILING_CHECK(
        yTotalLength != 0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context->GetNodeName(), "y must be an empty tensor when x is empty, but total num of y is %ld.",
            yTotalLength),
        return ge::GRAPH_FAILED);
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(totalCore);
    context->SetTilingKey(TILING_KEY_EMPTY);
    return ge::GRAPH_SUCCESS;
}

template <GLU_FLAG Glu_Flag>
ge::graphStatus Tiling4SwiGlu(gert::TilingContext* context) {
    OP_LOGD(context->GetNodeName(), "Tiling4SwiGlu enter.");
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t totalCore = platformInfo.GetCoreNumAiv();
    auto npuArch = platformInfo.GetCurNpuArch();
    if ((IsRegbaseSocVersion(context)) && GetSelfIdx<Glu_Flag>() == 1) {
        OP_LOGD(context->GetNodeName(), "Tiling SwiGluGrad RegBase start");
        GluBaseTiling4RegBase tilingObj(context);
        return tilingObj.DoTiling();
    }
    uint64_t ubSize = 0;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_TILING_CHECK((static_cast<int>(totalCore) < 0 || ubSize <= UB_RESERVED_BUFF),
        OP_LOGE(context->GetNodeName(), "Compile Info is invalid, coreNum:%u, ubSize:%lu", totalCore, ubSize),
        return ge::GRAPH_PARAM_INVALID);
    ubSize -= UB_RESERVED_BUFF;

    auto inputShape = context->GetInputShape(INPUT_OUTPUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto xTotalLength = inputShape->GetStorageShape().GetShapeSize();
    if (xTotalLength == 0 && IsRegbaseSocVersion(context) && Glu_Flag == GLU_FLAG::SWIGLU_SINGLE) {
        return processEmptyTensor(context, totalCore);
    }

    SwiGluTilingData tilingData;
    auto inputDesc = context->GetInputDesc(0);
    ge::DataType dataType = inputDesc->GetDataType();

    auto inTensor = context->GetInputTensor(GetSelfIdx<Glu_Flag>());
    OPS_CHECK_NULL_WITH_CONTEXT(context, inTensor);
    auto inShape = inTensor->GetOriginShape();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    OP_LOGE_IF(attrs == nullptr, false, context->GetNodeName(), "Get op attrs failed.");

    uint32_t inDim = *(attrs->GetInt(0));

    GluSingleTilingCalculator tilingCalculator(&tilingData, context->GetNodeName());
    if (!tilingCalculator.isSupportSocV(dataType, npuArch)) {
        return ge::GRAPH_FAILED;
    }

    if (!tilingCalculator.SetTotalShape<Glu_Flag>(inShape, inDim) ||
        !tilingCalculator.CalcTiling<Glu_Flag>(totalCore, ubSize, dataType, npuArch)) {
        return ge::GRAPH_FAILED;
    }

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(tilingCalculator.totalUsedCoreNum_);
    context->SetTilingKey(dataType);

    size_t *userWorkspaceSize = context->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, userWorkspaceSize);
    size_t workspaceSize = WORKSPACE_BUFFER;
    userWorkspaceSize[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

struct SwiGluCompileInfo {};

IMPL_OP_OPTILING(SwiGlu).Tiling(Tiling4SwiGlu<SWIGLU_SINGLE>).TilingParse<SwiGluCompileInfo>(TilingPrepare4SwiGlu);
IMPL_OP_OPTILING(SwiGluGrad).Tiling(Tiling4SwiGlu<SWIGLU_GRAD_SINGLE>).TilingParse<SwiGluCompileInfo>(TilingPrepare4SwiGlu);

}  // namespace optiling
