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
 * \file norm_tiling_check_common.h
 * \brief
 */
#ifndef NORM_COMMON_NORM_TILING_CHECK_COMMON_H_
#define NORM_COMMON_NORM_TILING_CHECK_COMMON_H_

#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "../../common/inc/error_util.h"

namespace NormCheck {

inline bool CheckShapeSame(const gert::StorageShape* src1Shape, const gert::StorageShape* src2Shape,
    string nodeName, string src1Name, string src2Name)
{
    size_t src1DimNum = src1Shape->GetStorageShape().GetDimNum();
    size_t src2DimNum = src2Shape->GetStorageShape().GetDimNum();

    OP_TILING_CHECK(
        (src1DimNum != src2DimNum),
        OP_LOGE(
            nodeName.c_str(), "Dim num check invalid, %s is %lu %s is %lu, not equal.", src1Name.c_str(), src1DimNum,
            src2Name.c_str(), src2DimNum),
        return false);
    for (size_t i = 0; i < src1DimNum; i++) {
        uint64_t src1DimValue = src1Shape->GetStorageShape().GetDim(i);
        uint64_t src2DimValue = src2Shape->GetStorageShape().GetDim(i);

        OP_TILING_CHECK(
            (src1DimValue != src2DimValue),
            OP_LOGE(
                nodeName.c_str(), "Dim value check invalid, %s[%lu] is %lu, %s[%lu] is %lu, not equal.",
                src1Name.c_str(), i, src1DimValue, src2Name.c_str(), i, src2DimValue),
            return false);
    }
    return true;
}

inline bool CheckShapeLastDim(const gert::StorageShape* src1Shape, const gert::StorageShape* src2Shape,
    string nodeName, string src1Name, string src2Name)
{
    size_t src1DimNum = src1Shape->GetStorageShape().GetDimNum();
    size_t src2DimNum = src2Shape->GetStorageShape().GetDimNum();
    uint64_t src1DimValue = src1Shape->GetStorageShape().GetDim(src1DimNum - 1);
    uint64_t src2DimValue = src2Shape->GetStorageShape().GetDim(src2DimNum - 1);

    OP_TILING_CHECK(
        (src1DimValue != src2DimValue),
        OP_LOGE(
            nodeName.c_str(), "Dim value check invalid, %s[%lu] is %lu, %s[%lu] is %lu, not equal.", src1Name.c_str(),
            src1DimNum - 1, src1DimValue, src2Name.c_str(), src2DimNum - 1, src2DimValue),
        return false);
    return true;
}

inline bool CheckShapeNumel(const gert::StorageShape* src1Shape, const gert::StorageShape* src2Shape,
    string nodeName, string src1Name, string src2Name)
{
    size_t src1DimNum = src1Shape->GetStorageShape().GetDimNum();
    size_t src2DimNum = src2Shape->GetStorageShape().GetDimNum();
    uint64_t src1Numel = 1;
    uint64_t src2Numel = 1;
    for (size_t i = 0; i < src1DimNum; i++) {
        src1Numel *= src1Shape->GetStorageShape().GetDim(i);
    }
    for (size_t i = 0; i < src2DimNum; i++) {
        src2Numel *= src2Shape->GetStorageShape().GetDim(i);
    }
    OP_TILING_CHECK(
        (src1Numel != src2Numel),
        OP_LOGE(
            nodeName.c_str(), "Shape Numel check invalid, %s is %lu, %s is %lu, not equal.", src1Name.c_str(),
            src1Numel, src2Name.c_str(), src2Numel),
        return false);
    return true;
}

inline bool CheckShapeBC(const gert::StorageShape* srcBcShape, const gert::StorageShape* srcShape,
    string nodeName, string srcBcName, string srcName, bool isBcHeader = true)
{
    size_t srcBcDimNum = srcBcShape->GetStorageShape().GetDimNum();
    size_t srcDimNum = srcShape->GetStorageShape().GetDimNum();

    OP_TILING_CHECK(
        (srcBcDimNum < srcDimNum),
        OP_LOGE(
            nodeName.c_str(), "Dim num check invalid, %s is %lu %s is %lu, not bigger.", srcBcName.c_str(), srcBcDimNum,
            srcName.c_str(), srcDimNum),
        return false);
    for (size_t i = 0; i < srcDimNum; i++) {
        uint64_t srcBcDimValue;
        if (isBcHeader) {
            srcBcDimValue = srcBcShape->GetStorageShape().GetDim(srcBcDimNum - srcDimNum + i);
        } else {
            srcBcDimValue = srcBcShape->GetStorageShape().GetDim(i);
        }
        uint64_t srcDimValue = srcShape->GetStorageShape().GetDim(i);

        OP_TILING_CHECK(
            (srcBcDimValue != srcDimValue),
            OP_LOGE(
                nodeName.c_str(), "Dim value check invalid, %s[%lu] is %lu, %s[%lu] is %lu, not equal.",
                srcBcName.c_str(), i, srcBcDimValue, srcName.c_str(), i, srcDimValue),
            return false);
    }
    return true;
}

inline bool CheckDimBiggerZero(const gert::StorageShape* srcShape, const uint32_t shapeLen,
    string nodeName, string srcName)
{
    // Support zero shape.
    for (uint32_t i = 0; i < shapeLen; i++) {
        OP_TILING_CHECK(
            srcShape->GetStorageShape().GetDim(i) < 0,
            OP_LOGE(nodeName.c_str(), "Input %s shape should bigger or equal than zero.", srcName.c_str()),
            return false);
    }
    return true;
}
} // namespace NormCheck

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_NORM_COMMON_NORM_TILING_CHECK_COMMON_H_
