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
 * \file flash_attention_score_grad_tiling_common.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <vector>
#include <register/tilingdata_base.h>
#include <register/op_impl_registry.h>
#include <tiling/tiling_api.h>
#include "log/log.h"
#include "../../op_kernel/arch32/flash_attention_score_grad_tiling.h"
#include "../../op_kernel/arch32/flash_attention_score_grad_template_tiling_key.h"

namespace optiling {

constexpr int64_t BYTE_PER_BLOCK = 32; // 32 B in block
constexpr int HIGH_PRECISION = 0;
constexpr int HIGH_PERFORMANCE = 1;

constexpr const char *BSH_STR = "BSH";
constexpr const char *SBH_STR = "SBH";
constexpr const char *BNSD_STR = "BNSD";
constexpr const char *BSND_STR = "BSND";
constexpr const char *TND_STR = "TND";
constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;
constexpr size_t DIM_4 = 4;
constexpr size_t LAST_AXIS_IDX = 1;
constexpr size_t SEC_LAST_AXIS_IDX = 2;
constexpr uint32_t MULT_BASE = 2;
constexpr uint32_t SOFTMAX_REMAIN_SIZE = 8 * 1024;
constexpr uint32_t API_RSDV_BUFFER_SIZE = 4 * 1024;
constexpr int64_t SINGLE_VEC_INST_DATASIZE = 256;
constexpr uint32_t DEFAULT_DATA_TYPE_SIZE = 4;
constexpr uint32_t DEFAULT_MASK = 64;
constexpr uint32_t FP16_DATA_TYPE_SIZE = 2;
constexpr uint32_t BF16_DATA_TYPE_SIZE = 2;
constexpr uint32_t FP16_MASK = 128;
constexpr uint32_t BF16_MASK = 64;
constexpr int64_t FRACTAL_NUM = 16;     // 16 is 分形大小
constexpr uint32_t CUBE_ALIGN_NUM = 16; // 16 is cube align num
constexpr uint32_t BYTE_BLOCK = 32;     // 32 B in block
constexpr uint32_t BATCH_MAX_SIZE = 64;
constexpr uint32_t PREFIX_COMPRESS_S1_SIZE = 3072;
constexpr uint32_t ATTEN_MASK_COMPRESS_LIMIT = 2048;
constexpr uint32_t BOOL_BLOCK_NUMS = 32;
constexpr uint32_t DROPOUT4BIT_LEN = 16;
const int64_t UB_BASIC_LIMIT_SIZE = 8 * 1024;
const int64_t SAB_TND_SIZE = 1024;

enum TilingDataType {
    FP16 = 1,
    BF16 = 2,
    FP32 = 3,
    INHP
};

enum class InputLayout {
    BSH = 0,
    SBH = 1,
    BNSD = 2,
    BSND = 3,
    TND
};

enum AxisIdx {
    B = 0,
    S = 1,
    H = 2
};

enum Axis4Idx {
    AXIS4_B = B,
    AXIS4_S = S,
    AXIS4_N = 2,
    AXIS4_D = 3
};

/* layout和b,s,h三根轴的位置关系映射 */
const std::vector<std::vector<size_t>> LAYOUT_TO_AXIS{
    // 3根轴对应dimid
    {0, 1, 2}, // BSH
    {1, 0, 2}, // SBH
    // 4根轴对应dimid
    {0, 2, 1, 3}, // BNSD
    {0, 1, 2, 3}  // BSND
};

enum InputIndex {
    QUERY = 0,
    KEY,
    VALUE,
    DY,
    PSE_SHIFT,
    DROP_MASK,
    PADDING_MASK,
    ATTEN_MASK,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    SOFTMAX_IN,
    ATTENTION_IN,
    PREFIX_N,
    ACTUAL_SEQ_Q_LEN,
    ACTUAL_SEQ_KV_LEN,
    Q_START_IDX,
    KV_START_IDX,
    D_SCALE_Q,
    D_SCALE_K,
    D_SCALE_V,
    D_SCALE_DY,
    D_SCALE_0,
    QUERY_ROPE,
    KEY_ROPE,
    SINK_IN
};

enum AttenMaskCompressMode {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE,
    RIGHT_DOWN_CAUSAL_MODE,
    BAND_EQUAL_S_MODE,
    PREFIX_COMPRESS_MODE,
    RIGHT_DOWN_CASUAL_BAND_MODE,
    BAND_LEFT_UP_CASUAL_MODE
};

enum AttrIndex {
    SCALE_VALUE = 0,
    KEEP_PROB,
    PRE_TOKENS,
    NEXT_TOKENS,
    HEAD_NUM,
    INPUT_LAYOUT,
    INNER_PRECISE,
    SPARSE_MODE,
    PSETYPE,
    SEED,
    OFFSET,
    OUT_TYPE,
    TND_SOFTMAX_IN
};

enum PseShapeType {
    PSE_SHAPE_TYPE_BNSS,
    PSE_SHAPE_TYPE_BN1S,
    PSE_SHAPE_TYPE_1NSS,
    PSE_SHAPE_TYPE_BNHS,
    PSE_SHAPE_TYPE_1NHS,
    PSE_B_N2_G_SLOPE,
    PSE_1_N2_G_SLOPE
};

enum PseType : uint8_t {
    PSE_OUTER_MUL_ADD_TYPE = 0,
    PSE_OUTER_ADD_MUL_TYPE, // default
    PSE_INNER_MUL_ADD_TYPE,
    PSE_INNER_MUL_ADD_SQRT_TYPE,
    PSE_INVALID_TYPE
};

enum AttenDataType {
    ATTEN_MASK_TYPE_SAME = 0,   // 0 表示 AttenMask 数据类型与 qkv 一致
    ATTEN_MASK_TYPE_U8_BOOL = 1 // 1 表示 AttenMask 数据类型为 u8 bool
};

enum AttenShapeType {
    ATTEN_MASK_SHAPE_TYPE_SS,
    ATTEN_MASK_SHAPE_TYPE_B1SS,
    ATTEN_MASK_SHAPE_TYPE_BNSS
};

enum SparseMode {
    NO_MASK = 0, // 未传入attenmask，不做mask操作
    ALL_MASK,
    LEFT_UP_CAUSAL,        // 左上角点划分的三角部分
    RIGHT_DOWN_CAUSAL = 3, // 右下角点划分的下三角部分
    BAND = 4,
    PREFIX = 5,
    PREFIX_COMPRESS = 6,
    RIGHT_DOWN_CASUAL_BAND = 7,
    BAND_LEFT_UP_CASUAL = 8
};

constexpr uint32_t ATTEN_MASK_SHAPE_TEMP_DIMS = 0; // 0 是 B1SS 及 SS差异轴索引, S不可能为 1

struct TempParams { // 频繁使用的中间态临时变量
    uint32_t usedUBSize;
    uint32_t tilingKey;
    uint32_t apiClcQueueSize = 0;
};

inline uint32_t Gcd(uint32_t a, uint32_t b) // a >= b
{
    if (b > a) {
        return Gcd(b, a);
    }
    if (a % b == 0U) {
        return b;
    }
    return Gcd(b, a % b);
}

inline int64_t CeilCommon(int64_t num1, int64_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

inline int64_t Align(const int64_t n)
{
    return (n + BYTE_PER_BLOCK - 1) & (~(BYTE_PER_BLOCK - 1));
}

inline uint32_t AlignData(const uint32_t a, const uint32_t b)
{
    if (b == 0U) {
        return a;
    }
    return (a + b - 1U) / b * b;
}

template <class T> inline T AlignTo(const T n, const T alignSize)
{
    if (alignSize == 0) {
        return 0;
    }
    return (n + alignSize - 1) & (~(alignSize - 1));
}

template <typename T> inline T AlignUp(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}
ge::graphStatus CheckSoftmaxMaxShape(gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1);
ge::graphStatus CheckTndSoftmaxMaxShape(gert::TilingContext *context, int64_t t1, int64_t n1);
ge::graphStatus CheckSoftmaxSumShape(gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1);
ge::graphStatus CheckTndSoftmaxSumShape(gert::TilingContext *context, int64_t t1, int64_t n1);
ge::graphStatus CheckAttentionInShape(gert::TilingContext *context);
ge::graphStatus CheckSoftmaxDtype(gert::TilingContext *context);
ge::graphStatus CheckAttentionInDtype(gert::TilingContext *context);
ge::graphStatus CheckShapeValid(gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1, int64_t d);
ge::graphStatus CheckTndShapeValid(gert::TilingContext *context, int64_t t1, int64_t n1, int64_t d);
ge::graphStatus CheckDtypeValid(gert::TilingContext *context);
bool IsSameShape(const gert::StorageShape *aShape, const gert::StorageShape *bShape);
bool IsSameShapeButValueDLeEqD(const gert::StorageShape *aShape, const gert::StorageShape *bShape);
bool isTndSABHit(const gert::TilingContext *context);

// dq/dv/dk vdup and dropMask bit2bool
BEGIN_TILING_DATA_DEF(PreParams)
TILING_DATA_FIELD_DEF(uint64_t, maskPreBlockTotal);
TILING_DATA_FIELD_DEF(uint64_t, maskSingleCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, qPreBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, qPreBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, qPreBlockTail);
TILING_DATA_FIELD_DEF(uint32_t, qRopePreBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, qRopePreBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, qRopePreBlockTail);
TILING_DATA_FIELD_DEF(uint32_t, kvPreBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, kvPreBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, kvPreBlockTail);
TILING_DATA_FIELD_DEF(uint32_t, kRopePreBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, kRopePreBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, kRopePreBlockTail);
TILING_DATA_FIELD_DEF(uint32_t, vPreBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, vPreBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, vPreBlockTail); 
TILING_DATA_FIELD_DEF(uint32_t, maskCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, castBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, outputBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, inputBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, singleUBProcessNum);
TILING_DATA_FIELD_DEF(uint32_t, maskSingleCoreLoop);
TILING_DATA_FIELD_DEF(uint32_t, maskLastLoopNum);
TILING_DATA_FIELD_DEF(uint32_t, maskTailCoreLoop);
TILING_DATA_FIELD_DEF(uint32_t, maskTailCoreLastLoopNum);
TILING_DATA_FIELD_DEF(uint32_t, dropoutIsDivisibleBy8);
TILING_DATA_FIELD_DEF(uint64_t, dropBeginAddr);
TILING_DATA_FIELD_DEF(uint32_t, reserved); 
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PreParamsOp, PreParams)

// sfmg pre
BEGIN_TILING_DATA_DEF(PreSfmgParams)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, inputBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, castBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, outputBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, tempBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, rsv1);
TILING_DATA_FIELD_DEF(int64_t, singleLoopNBurstNum);
TILING_DATA_FIELD_DEF(int64_t, normalCoreLoopTimes);
TILING_DATA_FIELD_DEF(int64_t, tailCoreLoopTimes);
TILING_DATA_FIELD_DEF(int64_t, normalCoreLastLoopNBurstNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreLastLoopNBurstNum);
TILING_DATA_FIELD_DEF(int64_t, normalCoreNBurstNums);
TILING_DATA_FIELD_DEF(int64_t, sfmgPreBeginAddr);
TILING_DATA_FIELD_DEF(int64_t, b);
TILING_DATA_FIELD_DEF(int64_t, n2);
TILING_DATA_FIELD_DEF(int64_t, g);
TILING_DATA_FIELD_DEF(int64_t, s1);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, rope_d);
TILING_DATA_FIELD_DEF(int64_t, value_d);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PreSfmgParamsOp, PreSfmgParams)

// dq/dv/dk cast fp322T1
BEGIN_TILING_DATA_DEF(PostParams)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, postUbBaseSize);
TILING_DATA_FIELD_DEF(uint32_t, postMaxDataSize);
TILING_DATA_FIELD_DEF(uint32_t, nzReservedSize);
TILING_DATA_FIELD_DEF(uint32_t, qPostBlockFactor);
TILING_DATA_FIELD_DEF(uint64_t, qPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, qPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, qPostTailNum);
TILING_DATA_FIELD_DEF(uint32_t, qRopePostBlockFactor);
TILING_DATA_FIELD_DEF(uint64_t, qRopePostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, qRopePostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, qRopePostTailNum);
TILING_DATA_FIELD_DEF(uint32_t, kvPostBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, kvPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, kvPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, kvPostTailNum);
TILING_DATA_FIELD_DEF(uint32_t, kRopePostBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, kRopePostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, kRopePostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, kRopePostTailNum);
TILING_DATA_FIELD_DEF(uint32_t, vPostBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, vPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, vPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, vPostTailNum);
TILING_DATA_FIELD_DEF(uint64_t, qSizeAlign);
TILING_DATA_FIELD_DEF(uint64_t, kvSizeAlign);
TILING_DATA_FIELD_DEF(uint64_t, qRopeSizeAlign);
TILING_DATA_FIELD_DEF(uint64_t, kRopeSizeAlign);
TILING_DATA_FIELD_DEF(uint64_t, vSizeAlign);
TILING_DATA_FIELD_DEF(uint64_t, dqWorkSpaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, dqRopeWorkSpaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, dkWorkSpaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, dkRopeWorkSpaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, dvWorkSpaceOffset);
TILING_DATA_FIELD_DEF(int64_t, b);
TILING_DATA_FIELD_DEF(int64_t, n2);
TILING_DATA_FIELD_DEF(int64_t, g);
TILING_DATA_FIELD_DEF(int64_t, s1);
TILING_DATA_FIELD_DEF(int64_t, s2);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, rope_d);
TILING_DATA_FIELD_DEF(int64_t, value_d);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PostParamsOp, PostParams)

} // namespace optiling
