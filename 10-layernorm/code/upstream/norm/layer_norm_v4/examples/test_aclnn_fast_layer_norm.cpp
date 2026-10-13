/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_layer_norm.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
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

template <typename T>
int CreateAclTensorMem(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
void aclCreateTensorP(const std::vector<T>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
}

template <typename T>
void aclCreateIntArrayP(const std::vector<T>& hostData, aclIntArray** intArray)
{
    *intArray = aclCreateIntArray(hostData.data(), hostData.size());
}

int main()
{
    // 1.（固定写法）device/stream初始化，参考acl API手册
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {1, 2, 32};
    std::vector<int64_t> normShape = {32};
    std::vector<int64_t> meanShape = {1, 2, 1};
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclIntArray* norm = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* out = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* rstd = nullptr;
    std::vector<float> xHostData(64, 2.0);
    std::vector<int64_t> normData = {32};
    std::vector<float> weightHostData(32, 1.0);
    std::vector<float> biasHostData(32, 0.0);
    std::vector<float> outHostData(64, 0.0);
    std::vector<float> meanHostData(2, 0.0);
    std::vector<float> rstdHostData(2, 0.0);
    double eps = 1e-5;

    ret = CreateAclTensorMem(xHostData, xShape, &xDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorMem(weightHostData, normShape, &weightDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorMem(biasHostData, normShape, &biasDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorMem(outHostData, xShape, &outDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorMem(meanHostData, meanShape, &meanDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorMem(rstdHostData, meanShape, &rstdDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclCreateTensorP(xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    aclCreateIntArrayP(normData, &norm);
    aclCreateTensorP(normShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    aclCreateTensorP(normShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    aclCreateTensorP(xShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    aclCreateTensorP(meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    aclCreateTensorP(meanShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnFastLayerNorm第一段接口
    ret = aclnnFastLayerNormGetWorkspaceSize(x, norm, weight, bias, eps, out, mean, rstd, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFastLayerNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnFastLayerNorm第二段接口
    ret = aclnnFastLayerNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFastLayerNorm failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(xShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy first result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out result[%ld] is: %f\n", i, resultData[i]);
    }

    auto size1 = GetShapeSize(meanShape);
    std::vector<float> resultData1(size1, 0);
    ret = aclrtMemcpy(
        resultData1.data(), resultData1.size() * sizeof(resultData1[0]), meanDeviceAddr, size1 * sizeof(resultData1[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy second result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("mean result[%ld] is: %f\n", i, resultData1[i]);
    }

    auto size2 = GetShapeSize(meanShape);
    std::vector<float> resultData2(size2, 0);
    ret = aclrtMemcpy(
        resultData2.data(), resultData2.size() * sizeof(resultData2[0]), rstdDeviceAddr, size2 * sizeof(resultData2[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy last result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size2; i++) {
        LOG_PRINT("rstd result[%ld] is: %f\n", i, resultData2[i]);
    }

    // 6. 释放aclTensor和aclIntArray，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyIntArray(norm);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(out);
    aclDestroyTensor(mean);
    aclDestroyTensor(rstd);

    // 7. 释放device资源
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(rstdDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}