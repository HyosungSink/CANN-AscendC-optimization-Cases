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
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_permute.h"
#include "aclnnop/aclnn_quant_matmul_v3.h"
#include "aclnnop/aclnn_trans_matmul_weight.h"
#include "aclnnop/aclnn_trans_quant_param_v2.h"

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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensorX2(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = static_cast<uint64_t>(GetShapeSize(shape));

    const aclIntArray* mat2Size = aclCreateIntArray(shape.data(), shape.size());
    auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret); return ret);
    size *= sizeof(T);

    // 调用aclrtMalloc申请device侧内存
    ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    std::vector<int64_t> storageShape;
    storageShape.push_back(GetShapeSize(shape));

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, storageShape.data(),
        storageShape.size(), *deviceAddr);
    return 0;
}

int aclnnQuantMatmulV3Test(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> x1Shape = {5, 2};
    std::vector<int64_t> x2Shape = {2, 3};
    std::vector<int64_t> biasShape = {3};
    std::vector<int64_t> offsetShape = {3};
    std::vector<int64_t> scaleShape = {3};
    std::vector<int64_t> outShape = {5, 3};
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* quantParamDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* quantParam = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* out = nullptr;
    std::vector<int8_t> x1HostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<int8_t> x2HostData = {1, 1, 1, 1, 1, 1};
    std::vector<int32_t> biasHostData = {1, 1, 1};
    std::vector<float> scaleHostData = {1, 1, 1};
    std::vector<float> offsetHostData = {1, 1, 1};
    std::vector<uint16_t> outHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; // 实际上是float16半精度方式
    // 创建x1 aclTensor
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建AI处理器亲和数据排布格式的x2 aclTensor
    ret = CreateAclTensorX2(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2HPTensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2HPDeviceAddrPtr(x2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建scale aclTensor
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scaleTensorPtr(scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建quantParam aclTensor
    ret = CreateAclTensor(scaleHostData, scaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> quantParamTensorPtr(quantParam, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建offset aclTensor
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> offsetTensorPtr(offset, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建bias aclTensor
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> biasTensorPtr(bias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    bool transposeX1 = false;
    bool transposeX2 = false;

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    // 调用aclnnTransMatmulWeight第一段接口
    ret = aclnnTransMatmulWeightGetWorkspaceSize(x2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtrTrans(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtrTrans.reset(workspaceAddr);
    }
    // 调用aclnnTransMatmulWeight第二段接口
    ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

    // FLOAT数据类型的scale需要提前调用TransQuantParamV2算子的aclnn接口
    // 调用aclnnTransQuantParamV2第一段接口
    ret = aclnnTransQuantParamV2GetWorkspaceSize(scale, offset, quantParam, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtrV2(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtrV2.reset(workspaceAddr);
    }
    // 调用aclnnTransQuantParamV2第二段接口
    ret = aclnnTransQuantParamV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

    // 调用aclnnQuantMatmulV3第一段接口
    workspaceSize = 0;
    ret = aclnnQuantMatmulV3GetWorkspaceSize(
        x1, x2, quantParam, nullptr, bias, transposeX1, transposeX2, out, &workspaceSize, &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtrV3(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtrV3.reset(workspaceAddr);
    }
    // 调用aclnnQuantMatmulV3第二段接口
    ret = aclnnQuantMatmulV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV3 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(
        size, 0); // C语言中无法直接打印fp16的数据，需要用uint16读出来，自行通过二进制转成fp16
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnQuantMatmulV3Test(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV3Test failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}