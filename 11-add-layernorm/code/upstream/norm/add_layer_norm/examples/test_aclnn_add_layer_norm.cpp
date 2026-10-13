/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_layer_norm.h"

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

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2.
    // 构造输入与输出，需要根据API的接口自定义构造，本示例中将各调用一次不带biasOptional可选输入的和带biasOptional输入的用例
    float eps = 1e-6;
    bool additionalOutput = true;

    std::vector<int64_t> x1Shape = {1, 2, 8};
    std::vector<int64_t> x2Shape = {1, 2, 8};
    std::vector<int64_t> gammaShape = {8};
    std::vector<int64_t> betaShape = {8};
    std::vector<int64_t> biasOptionalShape = {8};

    std::vector<int64_t> outputYShape = {1, 2, 8};
    std::vector<int64_t> outputMeanShape = {1, 2, 1};
    std::vector<int64_t> outputRstdShape = {1, 2, 1};
    std::vector<int64_t> outputXShape = {1, 2, 8};

    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* biasOptionalDeviceAddr = nullptr;

    // 用于不带biasOptional的输出 Device 地址
    void* outputYDeviceAddr = nullptr;
    void* outputMeanDeviceAddr = nullptr;
    void* outputRstdDeviceAddr = nullptr;
    void* outputXDeviceAddr = nullptr;

    // 用于带biasOptional的输出 Device 地址
    void* outputYDeviceAddrbiasOptional = nullptr;
    void* outputMeanDeviceAddrbiasOptional = nullptr;
    void* outputRstdDeviceAddrbiasOptional = nullptr;
    void* outputXDeviceAddrbiasOptional = nullptr;

    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* biasOptional = nullptr;

    // 用于不带biasOptional的aclTensor
    aclTensor* outputY = nullptr;
    aclTensor* outputMean = nullptr;
    aclTensor* outputRstd = nullptr;
    aclTensor* outputX = nullptr;

    // 用于带biasOptional的aclTensor
    aclTensor* outputYbiasOptional = nullptr;
    aclTensor* outputMeanbiasOptional = nullptr;
    aclTensor* outputRstdbiasOptional = nullptr;
    aclTensor* outputXbiasOptional = nullptr;

    std::vector<float> x1HostData = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<float> x2HostData = {4, 4, 4, 4, 4, 4, 4, 4, -3, -3, -3, -3, -3, -3, -3, -3};
    std::vector<float> gammaHostData = {2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<float> betaHostData = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    std::vector<float> biasOptionalHostData = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

    // 用于不带biasOptional的HostData
    std::vector<float> outputYHostData(1 * 2 * 8);
    std::vector<float> outputMeanHostData(2);
    std::vector<float> outputRstdHostData(2);
    std::vector<float> outputXHostData(1 * 2 * 8);

    // 用于带biasOptional的HostData
    std::vector<float> outputYHostDatabiasOptional(1 * 2 * 8);
    std::vector<float> outputMeanHostDatabiasOptional(2);
    std::vector<float> outputRstdHostDatabiasOptional(2);
    std::vector<float> outputXHostDatabiasOptional(1 * 2 * 8);

    // 创建self aclTensor
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        biasOptionalHostData, biasOptionalShape, &biasOptionalDeviceAddr, aclDataType::ACL_FLOAT, &biasOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建不带 biasOptional 的 aclTensor
    ret = CreateAclTensor(outputYHostData, outputYShape, &outputYDeviceAddr, aclDataType::ACL_FLOAT, &outputY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputMeanHostData, outputMeanShape, &outputMeanDeviceAddr, aclDataType::ACL_FLOAT, &outputMean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputRstdHostData, outputRstdShape, &outputRstdDeviceAddr, aclDataType::ACL_FLOAT, &outputRstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outputXHostData, outputXShape, &outputXDeviceAddr, aclDataType::ACL_FLOAT, &outputX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建带 biasOptional 的 aclTensor
    ret = CreateAclTensor(
        outputYHostDatabiasOptional, outputYShape, &outputYDeviceAddrbiasOptional, aclDataType::ACL_FLOAT,
        &outputYbiasOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputMeanHostDatabiasOptional, outputMeanShape, &outputMeanDeviceAddrbiasOptional, aclDataType::ACL_FLOAT,
        &outputMeanbiasOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputRstdHostDatabiasOptional, outputRstdShape, &outputRstdDeviceAddrbiasOptional, aclDataType::ACL_FLOAT,
        &outputRstdbiasOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputXHostDatabiasOptional, outputXShape, &outputXDeviceAddrbiasOptional, aclDataType::ACL_FLOAT,
        &outputXbiasOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // aclnnAddLayerNorm接口调用示例，包含带biasOptional和不带biasOptional的各一次
    // 3. 调用CANN算子库API，需要修改为具体的API名称

    // 3.1 不带biasOptional可选输入的示例
    // 调用aclnnAddLayerNorm第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    LOG_PRINT("\nUse aclnnAddLayerNorm Non-biasOptional Port.");
    // biasOptional参数直接传入nullptr即可
    ret = aclnnAddLayerNormGetWorkspaceSize(
        x1, x2, gamma, beta, nullptr, eps, additionalOutput, outputY, outputMean, outputRstd, outputX, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnAddLayerNorm第二段接口
    ret = aclnnAddLayerNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNorm failed. ERROR: %d\n", ret); return ret);

    // 3.2 带biasOptional可选输入的示例
    // 调用aclnnAddLayerNorm第一段接口
    uint64_t workspaceSizebiasOptional = 0;
    aclOpExecutor* executorbiasOptional;
    LOG_PRINT("\nUse aclnnAddLayerNorm biasOptional Port.");
    // 正常传入biasOptional即可
    ret = aclnnAddLayerNormGetWorkspaceSize(
        x1, x2, gamma, beta, biasOptional, eps, additionalOutput, outputYbiasOptional, outputMeanbiasOptional,
        outputRstdbiasOptional, outputXbiasOptional, &workspaceSizebiasOptional, &executorbiasOptional);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddrbiasOptional = nullptr;
    if (workspaceSizebiasOptional > 0) {
        ret = aclrtMalloc(&workspaceAddrbiasOptional, workspaceSizebiasOptional, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnAddLayerNorm第二段接口
    ret = aclnnAddLayerNorm(workspaceAddrbiasOptional, workspaceSizebiasOptional, executorbiasOptional, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNorm failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改

    // 5.1 拷贝出不带biasOptional的输出
    auto outputYSize = GetShapeSize(outputYShape);
    std::vector<float> resultDataY(outputYSize, 0);
    ret = aclrtMemcpy(
        resultDataY.data(), resultDataY.size() * sizeof(resultDataY[0]), outputYDeviceAddr,
        outputYSize * sizeof(resultDataY[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm non-biasOptional: y output");
    for (int64_t i = 0; i < outputYSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataY[i]);
    }

    auto outputMeanSize = GetShapeSize(outputMeanShape);
    std::vector<float> resultDataMean(outputMeanSize, 0);
    ret = aclrtMemcpy(
        resultDataMean.data(), resultDataMean.size() * sizeof(resultDataMean[0]), outputMeanDeviceAddr,
        outputMeanSize * sizeof(resultDataMean[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm non-biasOptional: mean output");
    for (int64_t i = 0; i < outputMeanSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataMean[i]);
    }

    auto outputRstdSize = GetShapeSize(outputRstdShape);
    std::vector<float> resultDataRstd(outputRstdSize, 0);
    ret = aclrtMemcpy(
        resultDataRstd.data(), resultDataRstd.size() * sizeof(resultDataRstd[0]), outputRstdDeviceAddr,
        outputRstdSize * sizeof(resultDataRstd[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm non-biasOptional: rstd output");
    for (int64_t i = 0; i < outputRstdSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataRstd[i]);
    }

    auto outputXSize = GetShapeSize(outputXShape);
    std::vector<float> resultDataX(outputXSize, 0);
    ret = aclrtMemcpy(
        resultDataX.data(), resultDataX.size() * sizeof(resultDataX[0]), outputXDeviceAddr,
        outputXSize * sizeof(resultDataX[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm non-biasOptional: x output");
    for (int64_t i = 0; i < outputXSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataX[i]);
    }

    // 5.2 拷贝出带biasOptional的输出
    auto outputYSizebiasOptional = GetShapeSize(outputYShape);
    std::vector<float> resultDataYbiasOptional(outputYSizebiasOptional, 0);
    ret = aclrtMemcpy(
        resultDataYbiasOptional.data(), resultDataYbiasOptional.size() * sizeof(resultDataYbiasOptional[0]),
        outputYDeviceAddrbiasOptional, outputYSizebiasOptional * sizeof(resultDataYbiasOptional[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm biasOptional: y output");
    for (int64_t i = 0; i < outputYSizebiasOptional; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataYbiasOptional[i]);
    }

    auto outputMeanSizebiasOptional = GetShapeSize(outputMeanShape);
    std::vector<float> resultDataMeanbiasOptional(outputMeanSizebiasOptional, 0);
    ret = aclrtMemcpy(
        resultDataMeanbiasOptional.data(), resultDataMeanbiasOptional.size() * sizeof(resultDataMeanbiasOptional[0]),
        outputMeanDeviceAddrbiasOptional, outputMeanSizebiasOptional * sizeof(resultDataMeanbiasOptional[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm biasOptional: mean output");
    for (int64_t i = 0; i < outputMeanSizebiasOptional; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataMeanbiasOptional[i]);
    }

    auto outputRstdSizebiasOptional = GetShapeSize(outputRstdShape);
    std::vector<float> resultDataRstdbiasOptional(outputRstdSizebiasOptional, 0);
    ret = aclrtMemcpy(
        resultDataRstdbiasOptional.data(), resultDataRstdbiasOptional.size() * sizeof(resultDataRstdbiasOptional[0]),
        outputRstdDeviceAddrbiasOptional, outputRstdSizebiasOptional * sizeof(resultDataRstdbiasOptional[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm biasOptional: rstd output");
    for (int64_t i = 0; i < outputRstdSizebiasOptional; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataRstdbiasOptional[i]);
    }

    auto outputXSizebiasOptional = GetShapeSize(outputXShape);
    std::vector<float> resultDataXbiasOptional(outputXSizebiasOptional, 0);
    ret = aclrtMemcpy(
        resultDataXbiasOptional.data(), resultDataXbiasOptional.size() * sizeof(resultDataXbiasOptional[0]),
        outputXDeviceAddrbiasOptional, outputXSizebiasOptional * sizeof(resultDataXbiasOptional[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm biasOptional: x output");
    for (int64_t i = 0; i < outputXSizebiasOptional; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataXbiasOptional[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(beta);
    aclDestroyTensor(gamma);
    aclDestroyTensor(biasOptional);

    aclDestroyTensor(outputY);
    aclDestroyTensor(outputMean);
    aclDestroyTensor(outputRstd);
    aclDestroyTensor(outputX);

    aclDestroyTensor(outputYbiasOptional);
    aclDestroyTensor(outputMeanbiasOptional);
    aclDestroyTensor(outputRstdbiasOptional);
    aclDestroyTensor(outputXbiasOptional);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(biasOptionalDeviceAddr);

    aclrtFree(outputYDeviceAddr);
    aclrtFree(outputMeanDeviceAddr);
    aclrtFree(outputRstdDeviceAddr);
    aclrtFree(outputXDeviceAddr);

    aclrtFree(outputYDeviceAddrbiasOptional);
    aclrtFree(outputMeanDeviceAddrbiasOptional);
    aclrtFree(outputRstdDeviceAddrbiasOptional);
    aclrtFree(outputXDeviceAddrbiasOptional);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    if (workspaceSizebiasOptional > 0) {
        aclrtFree(workspaceAddrbiasOptional);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}