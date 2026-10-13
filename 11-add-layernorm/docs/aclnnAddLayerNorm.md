# aclnnAddLayerNorm

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/add_layer_norm)

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

- 接口功能：实现AddLayerNorm功能。
- 计算公式：

  $$
  x = x1 + x2 + biasOptional
  $$

  $$
  rstd = {{1}\over\sqrt {Var(x)+epsilon}}
  $$

  $$
  y = (x-E(x)) * rstd * gamma + beta
  $$

  其中，E(x)表示均值，Var(x)表示方差，均需要在算子内部计算得到。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnAddLayerNormGetWorkspaceSize`接口获取入参并根据计算流程所需workspace大小，再调用`aclnnAddLayerNorm`接口执行计算。

```Cpp
aclnnStatus aclnnAddLayerNormGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *gamma,
  const aclTensor *beta,
  const aclTensor *biasOptional,
  double           epsilon,
  bool             additionalOutput,
  const aclTensor *yOut,
  const aclTensor *meanOut,
  const aclTensor *rstdOut,
  const aclTensor *xOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddLayerNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddLayerNormGetWorkspaceSize

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
      <td>x1（aclTensor*）</td>
      <td>输入</td>
      <td>表示AddLayerNorm中加法计算的输入，将会在算子内做x1 + x2 + biasOptional的计算并对计算结果做层归一化。对应公式中的`x1`。</td>
      <td><ul><li>支持空Tensor。</li><li>不支持输入的某一维的值为0。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2（aclTensor*）</td>
      <td>输入</td>
      <td>表示AddLayerNorm中加法计算的输入，将会在算子内做x1 + x2 + biasOptional的计算并对计算结果做层归一化。对应公式中的`x2`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape和`x1`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta（aclTensor*）</td>
      <td>输入</td>
      <td>表示层归一化中的beta参数。对应公式中的`beta`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape的维度值与`x1`需要norm的维度值相同。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma（aclTensor*）</td>
      <td>输入</td>
      <td>表示层归一化中的gamma参数。对应公式中的`gamma`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape的维度值与`x1`需要norm的维度值相同。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional（aclTensor*）</td>
      <td>输入</td>
      <td>可选输入参数，表示AddLayerNorm中加法计算的输入，将会在算子内做x1 + x2 + biasOptional的计算并对计算结果做层归一化。对应公式中的`biasOptional`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape可以和`gamma`/`beta`或`x1`/`x2`一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon（double）</td>
      <td>输入</td>
      <td>表示添加到分母中的值，以确保数值稳定。对应公式中的`epsilon`。</td>
      <td>取值仅支持1e-5。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additionalOutput（bool）</td>
      <td>输入</td>
      <td>表示是否开启x=x1+x2+biasOptional的输出。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>meanOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示输出LayerNorm计算过程中（x1 + x2 + biasOptional）的结果的均值。对应公式中的`E(x)`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需要与`x1`满足<a href="../../../docs/zh/context/broadcast关系.md">broadcast关系</a>（前几维的维度和`x1`前几维的维度相同，后面的维度为1，总维度与`x1`维度相同，前几维指`x1`的维度减去gamma的维度，表示不需要norm的维度）。</li><li>当输入`x1`为空tensor时，`meanOut`也必须为空tensor。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示输出LayerNorm计算过程中`rstd`的结果。对应公式中的`rstd`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需要与`x1`满足<a href="../../../docs/zh/context/broadcast关系.md">broadcast关系</a>（前几维的维度和`x1`前几维的维度相同，后面的维度为1，总维度与`x1`维度相同，前几维指`x1`的维度减去gamma的维度，表示不需要norm的维度）。</li><li>当输入`x1`为空tensor时，`rstdOut`也必须为空tensor。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示LayerNorm的结果输出。对应公式中的`y`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需要与输入`x1`一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>xOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示Add的结果输出`x`。对应公式中的`x`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需要与输入`x1`一致。</li></ul></td>
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

  - <term>Atlas 推理系列产品</term>：
    - 参数`x1`、`x2`、`beta`、`gamma`、`biasOptional`、`yOut`、`xOut`的数据类型不支持BFLOAT16。
    - 参数`meanOut`、`rstdOut`在当前产品使用场景下无效。

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
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入或输出的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="9">561002</td>
      <td>tiling阶段（x1、x2、gamma、beta、yOut、meanOut、rstdOut、xOut）的shape获取失败。</td>
    </tr>
    <tr>
      <td>（x1、gamma）的shape维数大于8或小于0。</td>
    </tr>
    <tr>
      <td>（x1、x2、yOut、meanOut、rstdOut、xOut）的维数不一致。</td>
    </tr>
    <tr>
      <td>x1的维数小于gamma。</td>
    </tr>
    <tr>
      <td>（x1、gamma、meanOut）的任意一个维度等于0。</td>
    </tr>
    <tr>
      <td>（x1、x2、yOut、xOut）的shape不是完全相同的shape。</td>
    </tr>
    <tr>
      <td>（gamma、beta）的shape不是完全相同的shape。</td>
    </tr>
    <tr>
      <td>（meanOut、rstdOut）的shape不是完全相同的shape。</td>
    </tr>
    <tr>
      <td>gamma的维度和x的需要作norm的维度不相同，或meanOut的维度和x的不需要norm的维度不相同，或meanOut的需要norm的维度不为1。</td>
    </tr>
  </tbody></table>

## aclnnAddLayerNorm

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAddLayerNormGetWorkspaceSize获取。</td>
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

- **功能维度**
  - 数据类型支持
    - <term>Atlas 推理系列产品</term>：x1、x2、beta、gamma、biasOptional支持FLOAT32、FLOAT16。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：x1、x2、beta、gamma、biasOptional支持FLOAT32、FLOAT16、BFLOAT16。
    - rstdOut、meanOut支持：FLOAT32。
  - 数据格式支持：ND。
  - <term>Atlas 推理系列产品</term>：x1、x2、beta、gamma、biasOptional五个输入的尾轴长度必须大于等于32Bytes。
- **未支持类型说明**
  - DOUBLE：不支持DOUBLE。
- **边界值场景说明**
  - 当输入是Inf时，输出为Inf。
  - 当输入是NaN时，输出为NaN。
- **各产品支持数据类型说明**
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    
    | x1数据类型 | x2数据类型 | gamma数据类型 | beta数据类型 | biasOptional数据类型 | yOut数据类型 | meanOut数据类型 | rstdOut数据类型 | xOut数据类型 |
    | -------- | -------- | ------------- | ------------- | ----------- | --------- | --------- | --------- | :-------- |
    | FLOAT32  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | FLOAT32  | BFLOAT16 | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | BFLOAT16 | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT16  |
    | BFLOAT16 | BFLOAT16 | FLOAT32  | FLOAT32  | BFLOAT16 | BFLOAT16 | FLOAT32  | FLOAT32  | BFLOAT16 |
    | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT16  |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32  | FLOAT32  | BFLOAT16 |
    | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |

  - <term>Atlas 推理系列产品</term>：
    
    | x1数据类型 | x2数据类型 | gamma数据类型 | beta数据类型 | biasOptional数据类型 | yOut数据类型 | meanOut数据类型 | rstdOut数据类型 | xOut数据类型 |
    | -------- | -------- | ------------- | ------------- | ----------- | --------- | --------- | --------- | :-------- |
    | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | FLOAT16 |

- 确定性计算：
  - aclnnAddLayerNorm默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
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
```
