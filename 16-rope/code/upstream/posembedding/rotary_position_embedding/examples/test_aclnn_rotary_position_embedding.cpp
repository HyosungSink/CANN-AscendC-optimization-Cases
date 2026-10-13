/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "aclnnop/aclnn_rotary_position_embedding.h"
#include <iostream>
#include <vector>

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. 固定写法，device/stream初始化, 参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> xShape = {1, 1, 1, 128};
    std::vector<int64_t> cosShape = {1, 1, 1, 128};
    std::vector<int64_t> sinShape = {1, 1, 1, 128};
    std::vector<int64_t> outShape = {1, 1, 1, 128};
    int64_t mode = 1;

    void* xDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> xHostData = {74, 54, 84, 125, 23, 78, 37, 72, 27, 98, 34, 107, 29, 23, 54, 60, 70, 49,
                                    119, 54, 29, 54, 41, 99, 27, 62, 5, 46, 108, 39, 24, 123, 33, 82, 6, 40, 88,
                                    24, 6, 116, 38, 119, 110, 5, 30, 79, 87, 18, 29, 100, 90, 24, 21, 93, 63, 68,
                                    34, 112, 119, 48, 74, 43, 85, 64, 14, 49, 128, 59, 18, 37, 123, 76, 14, 63, 10,
                                    39, 107, 124, 79, 16, 17, 76, 80, 47, 90, 41, 58, 82, 75, 80, 69, 37, 74, 36, 54,
                                    26, 32, 54, 13, 100, 105, 15, 13, 69, 122, 26, 94, 59, 29, 14, 60, 8, 24, 17, 45,
                                    33, 107, 122, 63, 111, 75, 128, 68, 31, 105, 6, 82, 99};
    std::vector<float> cosHostData = {41, 37, 17, 25, 49, 25, 22, 24, 110, 120, 107, 3, 82, 66, 75, 86, 85, 115, 110, 56, 52,
                                      39, 86, 23, 36, 71, 20, 73, 113, 25, 114, 56, 125, 80, 95, 82, 31, 63, 99, 62, 23, 55, 30,
                                      99, 42, 121, 15, 24, 97, 87, 81, 67, 43, 21, 13, 9, 33, 29, 117, 10, 114, 61, 98, 15, 78,
                                      108, 48, 97, 1, 3, 78, 109, 57, 46, 47, 56, 50, 66, 81, 77, 17, 128, 68, 121, 47, 91, 114,
                                      125, 51, 108, 31, 15, 47, 78, 109, 115, 113, 26, 53, 97, 1, 111, 103, 58, 106, 68, 11,
                                      104, 22, 79, 61, 127, 86, 39, 33, 123, 102, 39, 64, 41, 119, 120, 61, 29, 94, 68, 36, 12};
    std::vector<float> sinHostData = {46, 56, 56, 101, 66, 10, 96, 16, 86, 57, 102, 66, 12, 105, 76, 58, 90, 6, 79, 128, 126,
                                      82, 41, 3, 45, 7, 66, 4, 46, 22, 31, 26, 37, 63, 97, 84, 91, 90, 47, 77, 90, 34, 41, 83,
                                      91, 108, 120, 13, 90, 32, 85, 37, 119, 31, 51, 82, 122, 125, 7, 116, 121, 108, 38, 56,
                                      100, 20, 97, 119, 10, 4, 53, 13, 46, 82, 103, 119, 124, 80, 23, 67, 78, 56, 119, 122, 40,
                                      58, 128, 27, 30, 52, 71, 42, 123, 69, 4, 5, 116, 97, 38, 107, 8, 4, 65, 120, 40, 22, 60,
                                      44, 48, 66, 68, 125, 4, 93, 112, 112, 113, 90, 94, 23, 104, 39, 85, 84, 64, 128, 96, 119};
    std::vector<float> outHostData(128, 0);                                      

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建cos aclTensor
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建sin aclTensor
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);    

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnRotaryPositionEmbedding第一段接口
    ret = aclnnRotaryPositionEmbeddingGetWorkspaceSize(x, cos, sin, mode, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRotaryPositionEmbeddingGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnRotaryPositionEmbedding第二段接口
    ret = aclnnRotaryPositionEmbedding(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRotaryPositionEmbedding failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);
    aclDestroyTensor(out);

    // 7. 释放device 资源
    aclrtFree(xDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}