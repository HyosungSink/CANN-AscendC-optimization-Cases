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
#include "aclnnop/aclnn_deformable_conv2d.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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

    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    auto format = shape.size() == 1 ? ACL_FORMAT_ND : ACL_FORMAT_NCHW;
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化, 参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {1, 6, 2, 4};
    std::vector<int64_t> weightShape = {4, 3, 5, 5};
    std::vector<int64_t> offsetShape = {1, 75, 2, 4};
    std::vector<int64_t> biasShape = {4};
    std::vector<int64_t> outShape = {1, 4, 2, 4};
    std::vector<int64_t> deformOutShape = {1, 6, 10, 20};
    std::vector<int64_t> kernelSize = {5, 5};
    std::vector<int64_t> stride = {1, 1, 1, 1};
    std::vector<int64_t> padding = {2, 2, 2, 2};
    std::vector<int64_t> dilation = {1, 1, 1, 1};
    int64_t groups = 2;
    int64_t deformableGroups = 1;
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* deformOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* out = nullptr;
    aclTensor* deformOut = nullptr;
    std::vector<float> xHostData(1 * 6 * 2 * 4, 1);
    std::vector<float> weightHostData(4 * 3 * 5 * 5, 1);
    std::vector<float> offsetHostData(1 * 75 * 2 * 4, 1);
    std::vector<float> biasHostData(4, 0);
    std::vector<float> outHostData(1 * 4 * 2 * 4, 0);
    std::vector<float> deformOutHostData(1 * 6 * 10 * 20, 0);
    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建weight aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建offset aclTensor
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建bias aclTensor
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建deformOut aclTensor
    ret = CreateAclTensor(deformOutHostData, deformOutShape, &deformOutDeviceAddr, aclDataType::ACL_FLOAT, &deformOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建kernelSize aclIntArray
    const aclIntArray* kernelSizeArray = aclCreateIntArray(kernelSize.data(), kernelSize.size());
    CHECK_RET(kernelSizeArray != nullptr, return ret);
    // 创建stride aclIntArray
    const aclIntArray* strideArray = aclCreateIntArray(stride.data(), stride.size());
    CHECK_RET(strideArray != nullptr, return ret);
    // 创建padding aclIntArray
    const aclIntArray* paddingArray = aclCreateIntArray(padding.data(), padding.size());
    CHECK_RET(paddingArray != nullptr, return ret);
    // 创建dilation aclIntArray
    const aclIntArray* dilationArray = aclCreateIntArray(dilation.data(), dilation.size());
    CHECK_RET(dilationArray != nullptr, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnDeformableConv2d第一段接口
    ret = aclnnDeformableConv2dGetWorkspaceSize(
        x, weight, offset, bias, kernelSizeArray, strideArray, paddingArray, dilationArray, groups, deformableGroups,
        true, out, deformOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeformableConv2dGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnDeformableConv2d第二段接口
    ret = aclnnDeformableConv2d(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeformableConv2d failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(offset);
    aclDestroyTensor(bias);
    aclDestroyTensor(out);
    aclDestroyTensor(deformOut);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(deformOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}