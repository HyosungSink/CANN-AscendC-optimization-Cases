/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

void PrintMat(std::vector<float> resultData, std::vector<int64_t> resultShape)
{
    int64_t m = resultShape[0];
    int64_t n = resultShape[1];
    for (size_t i = 0; i < m; i++) {
        printf(i == 0 ? "[[" : " [");
        for (size_t j = 0; j < n; j++) {
            printf(j == n - 1 ? "%.1f" : "%.1f, ", resultData[i * n + j]);
            if (j == 2 && j + 3 < n) {
                printf("..., ");
                j = n - 4;
            }
        }
        printf(i < m - 1 ? "],\n" : "]]\n");
        if (i == 2 && i + 3 < m) {
            printf(" ... \n");
            i = m - 4;
        }
    }
}

int AclnnWeightQuantBatchMatmulV2Test(int32_t deviceId, aclrtStream stream)
{
    int64_t m = 16;
    int64_t k = 32;
    int64_t n = 16;
    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> weightShape = {k, n};
    std::vector<int64_t> antiquantScaleShape = {n};
    std::vector<int64_t> yShape = {m, n};
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* antiquantScaleDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* antiquantScale = nullptr;
    aclTensor* y = nullptr;
    std::vector<uint16_t> xHostData(GetShapeSize(xShape), 0b0011110000000000); // fp16的1.0
    std::vector<int8_t> weightHostData(GetShapeSize(weightShape), 1);
    std::vector<uint16_t> antiquantScaleHostData(GetShapeSize(antiquantScaleShape), 0b0011110000000000);
    std::vector<float> yHostData(GetShapeSize(yShape), 0);

    // 创建x aclTensor
    auto ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建other aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y aclTensor
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建antiquantScale aclTensor
    ret = CreateAclTensor(
        antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT16,
        &antiquantScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleTensorPtr(
        antiquantScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建yFp16 aclTensor
    void* yFp16DeviceAddr = nullptr;
    aclTensor* yFp16 = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yFp16deviceAddrPtr(yFp16DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    // 调用aclnnWeightQuantBatchMatmulV2第一段接口
    ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
        x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 0, yFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnWeightQuantBatchMatmulV2第二段接口
    ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 将输出转为FP32
    workspaceSize = 0;
    executor = nullptr;
    ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceCastAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceCastAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceCastAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceCastAddrPtr.reset(workspaceCastAddr);
    }
    ret = aclnnCast(workspaceCastAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    PrintMat(resultData, yShape);
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    ret = AclnnWeightQuantBatchMatmulV2Test(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnWeightQuantBatchMatmulV2Test failed. ERROR: %d\n", ret);
                   return ret);

    Finalize(deviceId, stream);
    return 0;
}