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
 * \file runtime2_util.h
 * \brief runtime2 util
 */
#ifndef CANN_OPS_BUILT_IN_OP_TILING_RUNTIME2_UTIL_H_
#define CANN_OPS_BUILT_IN_OP_TILING_RUNTIME2_UTIL_H_

#include "register/op_impl_registry.h"
#include "op_util.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "util/fp16.h"

namespace optiling {
template <typename T>
static inline T* GetCompileInfoPtr(gert::TilingParseContext* context)
{
    return context->GetCompiledInfo<T>();
}

/*
 * @brief: Calculate reduce cof value
 * @param [in] input_shape: gert::Shape, the input shape for reduce
 * @param [in] reduce_axis: const std::vector<int32_t>, the reduce axes num
 * @param [out] reduce_mean_cof: the result of reduce cof value
 * @return bool: true or false;
 */
template <typename T>
static inline bool CalcReduceMeanCof(
    const gert::Shape& input_shape, const std::vector<T>& reduce_axis, float& reduce_mean_cof)
{
    const size_t dim_len = input_shape.GetDimNum();
    const size_t ori_reduce_axis_len = reduce_axis.size();
    // init reduce_mean_cof is 1.0
    reduce_mean_cof = 1.0;
    for (size_t i = 0; i < ori_reduce_axis_len; i++) {
        OP_CHECK_IF(
            !Ops::Nn::IsDimValid(dim_len, reduce_axis[i]),
            OP_LOGE(
                "CalcReduceMeanCof", "%s",
                Ops::Nn::GenInvalidDimMsg("reduce_axis", i, dim_len, reduce_axis[i]).c_str()),
            return false);

        // convert reduce axis (like: -1 -> (dim_len - 1))
        T single_reduce_axis = reduce_axis[i] < 0 ? reduce_axis[i] + dim_len : reduce_axis[i];

        int64_t reduce_dim = input_shape.GetDim(single_reduce_axis);
        OP_CHECK_IF(
            reduce_dim == 0, OP_LOGI("CalcReduceMeanCof", "the reduce dim is 0, will ignore reduce_mean_cof"),
            return true);
        if (reduce_dim != 0) {
            reduce_mean_cof = reduce_mean_cof / reduce_dim;
        }
    }
    OP_LOGD("CalcReduceMeanCof", "CalcReduceMeanCof cof is %1f", reduce_mean_cof);

    return true;
}

/*
 * @brief: add reduce cof value after the tiling data
 * @param [in] input_shape: gert::Shape, the input shape for reduce
 * @param [in] input_dtype: ge::DataType,  the input dtype for reduce
 * @param [in] reduce_axis: const std::vector<int32_t>, the reduce axes num
 * @param [out] tiling_data: gert::TilingData, the tiling data, will add the cof value to th lasy TilingData
 * @return bool: true or false;
 */
template <typename T>
static inline bool AddReduceMeanCof(
    const gert::Shape& input_shape, const ge::DataType input_dtype, const std::vector<T>& reduce_axis,
    gert::TilingData* tiling_data)
{
    float reduce_mean_cof = 1.0;
    bool calcu_flag = CalcReduceMeanCof(input_shape, reduce_axis, reduce_mean_cof);
    OP_LOGD("AddReduceMeanCof", "AddReduceMeanCof dtype is %s", Ops::Base::ToString(input_dtype).c_str());
    switch (input_dtype) {
        case ge::DT_FLOAT:
            tiling_data->Append((float)reduce_mean_cof);
            return calcu_flag;
        case ge::DT_FLOAT16:
            tiling_data->Append((Ops::Base::fp16_t)reduce_mean_cof);
            tiling_data->Append((uint16_t)0);
            return calcu_flag;
        default:
            OP_LOGW(
                "AddReduceMeanCof", "Only support [DT_FLOAT, DT_FLOAT16], but is [%s]",
                Ops::Base::ToString(input_dtype).c_str());
            return false;
    }
}

template <typename T>
class OpHashInput {
public:
    OpHashInput(const OpHashInput& rhs)
    {
        content = rhs.content;
    }
    OpHashInput()
    {}

    explicit OpHashInput(T& input)
    {
        content = input;
    }

    OpHashInput& operator=(const OpHashInput& rhs)
    {
        content = rhs.content;
        return *this;
    }

    bool operator==(const OpHashInput& input) const
    {
        if (std::memcmp(&content, &input.content, sizeof(content)) == 0) {
            return true;
        }
        return false;
    }

private:
    T content;
};
} // namespace optiling
#endif // CANN_OPS_BUILT_IN_OP_TILING_RUNTIME2_UTIL_H_