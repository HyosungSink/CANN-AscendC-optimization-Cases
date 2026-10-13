/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_INFERSHAPE_UTIL_H_
#define COMMON_INFERSHAPE_UTIL_H_

#include "register/op_impl_registry.h"
#include <sstream>

namespace Ops {
namespace Nn {
inline bool IsConstTensor(const gert::Tensor* input_tensor)
{
    if (input_tensor != nullptr) {
        if (input_tensor->GetAddr() == nullptr) {
            // empty tensor
            return input_tensor->GetShapeSize() == 0;
        }
        return true;
    }
    return false;
}

template <typename T1, typename T2>
static inline bool IsDimValid(const T1 shape_size, const T2 dim_value)
{
    int64_t minimum_num = static_cast<int64_t>(shape_size) * (-1);
    int64_t maximum_num = static_cast<int64_t>(shape_size) - 1;

    return static_cast<int64_t>(dim_value) >= minimum_num && static_cast<int64_t>(dim_value) <= maximum_num;
}

template <typename T>
static inline std::string ConcatString(const T& arg)
{
    std::ostringstream oss;
    oss << arg;
    return oss.str();
}

template <typename T, typename... Ts>
static inline std::string ConcatString(const T& arg, const Ts&... arg_left)
{
    std::ostringstream oss;
    oss << arg;
    oss << ConcatString(arg_left...);
    return oss.str();
}

static inline std::string GetAttrValueErrMsg(
    const std::string& attr_name, const std::string& wrong_val, const std::string& correct_val)
{
    std::string msg =
        ConcatString("attr[", attr_name, "], has wrong value[", wrong_val, "], it should be ", correct_val);
    return msg;
}

template <typename T1, typename T2>
static inline std::string GenInvalidDimMsg(const std::string dim_name, const T1 shape_size, const T2 dim_value)
{
    std::string wrong_val = ConcatString(static_cast<int64_t>(dim_value));
    // will be "[-rank, rank)"
    std::string neg_rank = ConcatString(static_cast<int64_t>(shape_size) * (-1));
    std::string expect_val = ConcatString("[", neg_rank, ", ", ConcatString(static_cast<int64_t>(shape_size)), ")");

    return GetAttrValueErrMsg(dim_name, wrong_val, expect_val);
}

template <typename T1, typename T2>
static inline std::string GenInvalidDimMsg(
    const std::string dim_name, const size_t dim_idx, const T1 shape_size, const T2 dim_value)
{
    std::string invalid_dim_name = ConcatString(dim_name, "[", ConcatString(dim_idx), "]");

    return GenInvalidDimMsg(invalid_dim_name, shape_size, dim_value);
}
} // namespace Nn
} // namespace Ops
#endif // COMMON_INFERSHAPE_UTIL_H_
