# aclnnRmsNormGrad

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/rms_norm_grad)

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

- 接口功能：[aclnnRmsNorm](../../rms_norm/docs/aclnnRmsNorm.md)的反向计算。用于计算RmsNorm的梯度，即在反向传播过程中计算输入张量的梯度。
- 算子公式：

  - 正向公式：

  $$
  \operatorname{RmsNorm}(x_i)=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  - 反向推导：

  $$
  dx_i= (dy_i * g_i - \frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * \operatorname{Mean}(\mathbf{y})) * \frac{1} {\operatorname{Rms}(\mathbf{x})},  \quad \text { where } \operatorname{Mean}(\mathbf{y}) = \frac{1}{n}\sum_{i=1}^n (dy_i * g_i * x_i * \frac{1}{\operatorname{Rms}(\mathbf{x})})
  $$

  $$
  dg_i = \frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * dy_i
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnRmsNormGradGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnRmsNormGrad`接口执行计算。

```Cpp
aclnnStatus aclnnRmsNormGradGetWorkspaceSize(
  const aclTensor *dy,
  const aclTensor *x,
  const aclTensor *rstd,
  const aclTensor *gamma,
  const aclTensor *dxOut,
  const aclTensor *dgammaOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnRmsNormGrad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnRmsNormGradGetWorkspaceSize

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
      <td>dy（aclTensor*）</td>
      <td>输入</td>
      <td>表示反向传回的梯度，对应公式中的`dy`。</td>
      <td>支持空Tensor。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>表示正向算子的输入，表示被标准化的数据，对应公式中的`x`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与入参`dy`的shape保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd（aclTensor*）</td>
      <td>输入</td>
      <td>表示正向算子的中间计算结果，对应公式中的`Rms(x)`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需要满足rstd_shape = x_shape[0:n]，n < x_shape.dims()，n与`gamma`的n一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma（aclTensor*）</td>
      <td>输入</td>
      <td>表示正向算子进行归一化计算的缩放因子（权重），对应公式中的`gamma`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需要满足gamma_shape = x_shape[n:], n < x_shape.dims()。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dxOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示输入`x`的梯度，对应公式中的`dx`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与入参`dy`的shape保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dgammaOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示`gamma`的梯度，对应公式中的`dg`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与入参`gamma`的shape保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
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

  - <term>Atlas 推理系列产品</term>：参数`dy`、`x`、`gamma`、`dxOut`的数据类型不支持BFLOAT16。
  
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
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="1">161002</td>
      <td>输入或输出的数据类型不在支持范围之内。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>参数不满足参数说明中的要求。</td>
    </tr>
  </tbody></table>

## aclnnRmsNormGrad

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnRmsNormGradGetWorkspaceSize获取。</td>
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

- <term>Atlas 推理系列产品</term>：`x`、`dy`、`gamma`输入的尾轴长度必须大于等于32Bytes。

- 各产品支持数据类型说明：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：

    | `dy`数据类型 | `x`数据类型 | `rstd`数据类型 | `gamma`数据类型 | `dxOut`数据类型 | `dgammaOut`数据类型 |
    | -------- | -------- | -------- | -------- | -------- | -------- |
    | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT16  | FLOAT32  |
    | BFLOAT16 | BFLOAT16 | FLOAT32  | FLOAT32  | BFLOAT16 | FLOAT32  |
    | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT16  | FLOAT16  | FLOAT32  |
    | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | BFLOAT16 | BFLOAT16 | FLOAT32  | BFLOAT16 | BFLOAT16 | FLOAT32  |

  - <term>Atlas 推理系列产品</term>：

    | `dy`数据类型 | `x`数据类型 | `rstd`数据类型 | `gamma`数据类型 | `dxOut`数据类型 | `dgammaOut`数据类型 |
    | -------- | -------- | -------- | -------- | -------- | -------- |
    | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT16  | FLOAT16  | FLOAT32  |
    | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |

- 确定性计算：
  - aclnnRmsNormGrad默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm_grad.h"
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

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW, shape.data(),
        shape.size(), *deviceAddr);
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
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> gradInputShape = {2, 1, 16};
    std::vector<int64_t> xInputShape = {2, 1, 16};
    std::vector<int64_t> rstdInputShape = {2};
    std::vector<int64_t> gammaInputShape = {16};
    std::vector<int64_t> dxOutputShape = {2, 1, 16};
    std::vector<int64_t> dgammaOutputShape = {16};

    void* gradInputDeviceAddr = nullptr;
    void* xInputDeviceAddr = nullptr;
    void* rstdInputDeviceAddr = nullptr;
    void* gammaInputDeviceAddr = nullptr;
    void* dxOutDeviceAddr = nullptr;
    void* dgammaOutDeviceAddr = nullptr;

    aclTensor* gradInput = nullptr;
    aclTensor* xInput = nullptr;
    aclTensor* rstdInput = nullptr;
    aclTensor* gammaInput = nullptr;
    aclTensor* dxOut = nullptr;
    aclTensor* dgammaOut = nullptr;

    std::vector<float> gradInputHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    std::vector<float> xInputHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    std::vector<float> rstdInputHostData = {1, 2};
    std::vector<float> gammaInputHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    std::vector<float> dxOutHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    std::vector<float> dgammaOutHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    std::vector<int64_t> output1SizeData = {2, 1, 16};
    std::vector<int64_t> output2SizeData = {16};
    std::vector<int64_t> input1SizeData = {2, 1, 16};
    std::vector<int64_t> input2SizeData = {2};
    std::vector<int64_t> input3SizeData = {16};

    ret = CreateAclTensor(gradInputHostData, input1SizeData, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(xInputHostData, input1SizeData, &xInputDeviceAddr, aclDataType::ACL_FLOAT, &xInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(rstdInputHostData, input2SizeData, &rstdInputDeviceAddr, aclDataType::ACL_FLOAT, &rstdInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret =
        CreateAclTensor(gammaInputHostData, input3SizeData, &gammaInputDeviceAddr, aclDataType::ACL_FLOAT, &gammaInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(dxOutHostData, output1SizeData, &dxOutDeviceAddr, aclDataType::ACL_FLOAT, &dxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(dgammaOutHostData, output2SizeData, &dgammaOutDeviceAddr, aclDataType::ACL_FLOAT, &dgammaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnRmsNormGrad第一段接口
    ret = aclnnRmsNormGradGetWorkspaceSize(
        gradInput, xInput, rstdInput, gammaInput, dxOut, dgammaOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnRmsNormGrad第二段接口
    ret = aclnnRmsNormGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormGrad failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    auto size_dx = GetShapeSize(gradInputShape);
    std::vector<float> resultData1(size_dx, 0);
    ret = aclrtMemcpy(
        resultData1.data(), resultData1.size() * sizeof(resultData1[0]), dxOutDeviceAddr, size_dx * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size_dx; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData1[i]);
    }
    auto size_dgamma = GetShapeSize(gammaInputShape);
    std::vector<float> resultData2(size_dgamma, 1);
    ret = aclrtMemcpy(
        resultData2.data(), resultData2.size() * sizeof(resultData2[0]), dgammaOutDeviceAddr,
        size_dgamma * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size_dgamma; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData2[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(gradInput);
    aclDestroyTensor(xInput);
    aclDestroyTensor(rstdInput);
    aclDestroyTensor(gammaInput);
    aclDestroyTensor(dxOut);
    aclDestroyTensor(dgammaOut);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(gradInputDeviceAddr);
    aclrtFree(xInputDeviceAddr);
    aclrtFree(rstdInputDeviceAddr);
    aclrtFree(gammaInputDeviceAddr);
    aclrtFree(dxOutDeviceAddr);
    aclrtFree(dgammaOutDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
