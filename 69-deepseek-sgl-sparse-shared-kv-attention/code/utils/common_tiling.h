/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMMON_TILING_H
#define COMMMON_TILING_H

#include <cstdint>

namespace host_utils {

enum class DataType : int32_t {
    DT_FLOAT16 = 0,   // fp16 type
    DT_BFLOAT16 = 1,  // bfloat16 type
    DT_FLOAT = 2,     // float type
    DT_DOUBLE = 3,    // double type
    DT_BOOL = 4,      // bool type
    DT_INT8 = 5,      // int8 type
    DT_INT16,         // int16 type
    DT_INT32,         // int32 type
    DT_INT64,         // int64 type
    DT_UINT8,         // unsigned int8 type
    DT_UINT16,        // unsigned int16 type
    DT_UINT32,        // unsigned int32 type
    DT_UINT64,        // unsigned int64 type
    DT_COMPLEX64,     // complex64 type
    DT_COMPLEX128,    // complex128 type

    DT_UNDEFINED,
    DT_MAX  // max type
};

}  // namespace host_utils

#endif  // COMMMON_TILING_H
