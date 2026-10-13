# aclnnSwiGlu

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/activation/swi_glu)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：Swish门控线性单元激活函数，实现x的SwiGlu计算。  

- 计算公式：  

  <p style="text-align: center">
  out<sub>i</sub> = SwiGlu(x<sub>i</sub>)=Swish(A<sub>i</sub>)*B<sub>i</sub>
  </p>
  其中，A<sub>i</sub>表示x<sub>i</sub>按指定dim维度一分为二的前半部分张量，B<sub>i</sub>表示x<sub>i</sub>按指定dim维度一分为二的后半部分张量。

## 函数原型

每个算子分为[两段式接口](zh/context/两段式接口.md)，必须先调用“aclnnSwiGluGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSwiGlu”接口执行计算。

```Cpp
aclnnStatus aclnnSwiGluGetWorkspaceSize(
  const aclTensor *x,
  int64_t          dim,
  const aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSwiGlu(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnSwiGluGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 270px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>表示待计算的数据，公式中的x<sub>i</sub>。</td>
      <td><ul><li>支持空Tensor。</li><li>维度必须大于0且shape必须在入参dim对应维度上是偶数。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
      <tr>
      <td>dim（int64_t）</td>
      <td>输入</td>
      <td>需要进行切分的维度序号，对x相应轴进行对半切分。</td>
      <td>取值范围为[-x.dim(), x.dim()-1]。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>表示计算结果，公式中的out<sub>i</sub>。</td>
      <td>数据类型与计算输入x的类型一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
     <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
  
    - <term>Atlas 推理系列产品</term>：数据类型支持FLOAT16、FLOAT32。shape不支持非64字节对齐。不支持空Tensor。
  
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的x或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>x或out的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>dim不在取值范围内。</td>
    </tr>
  </tbody></table>

## aclnnSwiGlu

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSwiGluGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnSwiGlu默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swi_glu.h"

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
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，acl初始化
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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> xShape = {2, 32};
  std::vector<int64_t> outShape = {2, 16};
  void* xDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> xHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  int dim = -1;
  // 创建x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnSwiGlu第一段接口
  ret = aclnnSwiGluGetWorkspaceSize(x, dim, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwiGluGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnSwiGlu第二段接口
  ret = aclnnSwiGlu(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwiGlu failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(out);
  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(xDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
