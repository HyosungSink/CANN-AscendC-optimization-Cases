# aclnnFastLayerNorm

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/layer_norm_v4)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：对指定层进行均值为0、标准差为1的归一化计算。aclnnFastLayerNorm接口相比aclnnLayerNorm接口，整体性能提升了50%，内存与GPU保持一致，累加序优化导致精度存在差异。

- 计算公式：

  $$
  out = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + eps}} * weightOptional + biasOptional
  $$

  $$
  meanOutOptional = \mathrm{E}[x]
  $$

  $$
  rstdOutOptional = \frac{1}{ \sqrt{\mathrm{Var}[x] + eps}}
  $$

  其中，E[x]表示输入的均值，Var[x]表示输入的方差。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFastLayerNormGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFastLayerNorm”接口执行计算。

```Cpp
aclnnStatus aclnnFastLayerNormGetWorkspaceSize(
  const aclTensor   *input,
  const aclIntArray *normalizedShape,
  const aclTensor   *weightOptional,
  const aclTensor   *biasOptional,
  double             eps,
  aclTensor         *out,
  aclTensor         *meanOutOptional,
  aclTensor         *rstdOutOptional,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnFastLayerNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnFastLayerNormGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
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
      <td>input（aclTensor*）</td>
      <td>输入</td>
      <td>表示进行归一化计算的输入，对应公式中的`x`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape为[A1,...,Ai,R1,...,Rj]，其中A1至Ai表示无需norm的维度，R1至Rj表示需norm的维度。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>normalizedShape（aclIntArray*）</td>
      <td>输入</td>
      <td>表示需要进行norm计算的维度。</td>
      <td>值为[R1,...,Rj]，长度小于等于输入input的shape长度，不支持为空。且R1*R2*...*Rj小于等于583705600。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightOptional（aclTensor*）</td>
      <td>输入</td>
      <td>可选输入参数，表示进行归一化计算的权重。对应公式中的`weightOptional`。</td>
      <td><ul><li>支持空Tensor。</li><li>当`weightOptional`非空时：<ul><li>数据类型与输入`input`一致或为FLOAT类型，且当`biasOptional`存在时，`weightOptional`与`biasOptional`的数据类型相同。</li><li>shape与`normalizedShape`相等，为[R1,...,Rj]。</li></ul></li><li>当`weightOptional`为空时，接口内部会构造一个shape为[R1,...,Rj]，数据全为1的Tensor。<ul><li>当`biasOptional`存在时，`weightOptional`与`biasOptional`的数据类型相同。</li><li>当`biasOptional`不存在时，`weightOptional`与输入`input`的数据类型相同。</li></ul></li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional（aclTensor*）</td>
      <td>输入</td>
      <td>可选输入参数，表示进行归一化计算的偏移量。对应公式中的`biasOptional`。</td>
      <td><ul><li>支持空Tensor。</li><li>当`biasOptional`非空时：<ul><li>数据类型与输入`input`一致或为FLOAT类型，且当`weightOptional`存在时，`biasOptional`与`weightOptional`的数据类型相同。</li><li>shape与`normalizedShape`相等，为[R1,...,Rj]。</li></ul></li><li>当`biasOptional`为空时，接口内部会构造一个shape为[R1,...,Rj]，数据全为0的Tensor。<ul><li>当`weightOptional`存在时，`biasOptional`与`weightOptional`的数据类型相同。</li><li>当`weightOptional`不存在时，`biasOptional`与输入`input`的数据类型相同。</li></ul></li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>eps（double）</td>
      <td>输入</td>
      <td>表示添加到分母中的值，以确保数值稳定。对应公式中的`eps`。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>表示进行归一化计算的结果。对应公式中的`out`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`input`的数据类型保持一致。</li><li>shape需要与`input`的shape相等，为[A1,...,Ai,R1,...,Rj]。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>meanOutOptional（aclTensor*）</td>
      <td>输出</td>
      <td>可选输出，表示进行归一化后的均值。对应公式中的`meanOutOptional`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`input`的数据类型保持一致。</li><li>当`rstdOutOptional`存在时与`rstdOutOptional`的shape相同，shape为[A1,...,Ai,1,...,1]，Ai后共有j个1，与需要norm的轴长度保持相同。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOutOptional（aclTensor*）</td>
      <td>输出</td>
      <td>可选输出，表示进行归一化后的标准差倒数。对应公式中的`rstdOutOptional`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`input`的数据类型保持一致。</li><li>当`meanOutOptional`存在时与`meanOutOptional`的shape相同，shape为[A1,...,Ai,1,...,1]，Ai后共有j个1，与需要norm的轴长度保持相同。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td>传入的input、normalizedShape或out为空指针。</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>input、weightOptional（非空时）、biasOptional（非空时）、out、meanOutOptional（非空时）、rstdOutOptional（非空时），shape的维度超过8维。</td>
    </tr>
    <tr>
      <td>normalizedShape的元素个数超过8。</td>
    </tr>
    <tr>
      <td>input、weightOptional（非空时）、biasOptional（非空时）、out、meanOutOptional（非空时）、rstdOutOptional（非空时），数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>normalizedShape维度小于1维。</td>
    </tr>
    <tr>
      <td>weightOptional非空且shape与normalizedShape不相等。</td>
    </tr>
    <tr>
      <td>biasOptional非空且shape与normalizedShape不相等。</td>
    </tr>
    <tr>
      <td>input的维度小于normalizedShape的维度。</td>
    </tr>
    <tr>
      <td>input的shape与normalizedShape右对齐时对应维度shape不相等。</td>
    </tr>
    <tr>
      <td>input和out的shape不一致。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>normalizedShape的值为[R1,...,Rj]时，R1*R2*...*Rj的值大于583705600。</td>
    </tr>
  </tbody></table>

## aclnnFastLayerNorm

- **参数说明**

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFastLayerNormGetWorkspaceSize获取。</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- input、normalizedShape、weightOptional（非空时）、biasOptional（非空时）、out、meanOutOptional（非空时）或rstdOutOptional（非空时）的shape不超过8维。
- 确定性计算：
  - aclnnFastLayerNorm默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
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
```
