# aclnnRotaryPositionEmbeddingV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/posembedding/rotary_position_embedding)

## 产品支持情况

| 产品                                                                   | 是否支持 |
|:-------------------------------------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                                                |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                        |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                                         |    ×    |
| <term>Atlas 推理系列产品 </term>                                                |    ×    |
| <term>Atlas 训练系列产品</term>                                                 |    ×    |

## 功能说明

- 接口功能：执行单路旋转位置编码计算。本接口相较于[aclnnRotaryPositionEmbedding](aclnnRotaryPositionEmbedding.md)，新增入参rotate，在推荐场景中通过输入旋转编码矩阵获得性能收益，请根据实际情况选择合适的接口。

- rotate推荐使用场景
  - interleave模式，且B \* N \* S > 28800。
  - half模式仅在以下场景时推荐使用：输入矩阵x需要在最后一个维度切分多份时，每一份都需要调用aclnnRotaryPositionEmbedding接口进行旋转位置编码计算，可以通过构造旋转编码矩阵实现一次调用获得性能收益，以x的layout为BSND需要切分为3份为例：
     x切分为3份，$x = [x1|x2|x3]_{(dim=4)} ∈ R^{B×S×N×D}, x1 ∈ R^{B×S×N×D1},x2 ∈ R^{B×S×N×D2},x3 ∈ R^{B×S×N×D3}, 其中D = D1 + D2 + D3$，那么可以构造一个rotate矩阵，实现调用一次aclnnRotaryPositionEmbeddingV2接口完成x的旋转位置编码计算功能，rotate矩阵构造如下：
     $$rotate = diag(rotate1, rotate2, rotate3) = \begin{pmatrix}rotate1&0&0\\0&rotate2&0\\0&0&rotate3\\\end{pmatrix}$$
     其中rotate1、rotate2、rotate3分别为x1、x2、x3的旋转编码矩阵，单个旋转矩阵构建参考调用示例。

- 计算公式：

  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

    - 不传入rotate参数（推荐half模式1D使用）：

      （1）half模式（mode等于0）：

      $$
      x1 = x[..., : x.shape[-1] // 2]
      $$

      $$
      x2 = x[..., x.shape[-1] // 2 :]
      $$

      $$
      x\_rotate = torch.cat((-x2, x1), dim=-1)
      $$

      $$
      y = x * cos + x\_rotate * sin
      $$

      （2）interleave模式（mode等于1）：

      $$
      x1 = x[..., ::2].view(-1, 1)
      $$

      $$
      x2 = x[..., 1::2].view(-1, 1)
      $$

      $$
      x\_rotate = torch.cat((-x2, x1), dim=-1).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
      $$

      $$
      y = x * cos + x\_rotate * sin
      $$

    - 传入rotate参数时（由开发者生成rotate矩阵，参考调用示例生成部分）：

      $$
      x\_rotate = x  @ rotate
      $$

      $$
      y = x * cos + x\_rotate * sin
      $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnRotaryPositionEmbeddingV2GetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnRotaryPositionEmbeddingV2”接口执行计算。

```c++
aclnnStatus aclnnRotaryPositionEmbeddingV2GetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* cos,
    const aclTensor* sin,
    int64_t mode,
    const aclTensor* rotate,
    aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

```c++
aclnnStatus aclnnRotaryPositionEmbeddingV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnRotaryPositionEmbeddingV2GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1461px"><colgroup>
  <col style="width: 162px">
  <col style="width: 121px">
  <col style="width: 332px">
  <col style="width: 169px">
  <col style="width: 275px">
  <col style="width: 118px">
  <col style="width: 138px">
  <col style="width: 146px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待执行旋转位置编码的张量，公式中的x。</td>
      <td>-</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3或4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>输入</td>
      <td>位置编码张量，公式中的cos。</td>
      <td>与x数据类型一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3或4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>输入</td>
      <td>位置编码张量，公式中的sin。</td>
      <td>与x数据类型一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3或4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>输入</td>
      <td>旋转模式。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rotate</td>
      <td>输入</td>
      <td>旋转矩阵</td>
      <td>与x数据类型一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>旋转位置编码计算结果，公式中的y。</td>
      <td>与x数据类型一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - 参数mode约束：
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：0=half，1=interleave。V2接口不同mode参数约束和V1接口相同，开发者可以根据mode在调用示例的辅助矩阵rotate生成中选择合适的rotate生成方式。
    - <term>Ascend 950PR/Ascend 950DT</term>：2=quarter，3=interleave-half。
    
  - 参数rotate当前支持BFLOAT16、FLOAT16、FLOAT32类型。
  
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 288px">
  <col style="width: 125px">
  <col style="width: 742px">
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
      <td>传入的x、cos、sin或out是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>传入的x、cos、sin、out的数据类型和格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="2">561103</td>
      <td>传入的x、cos、sin、out的shape不匹配。</td>
    </tr>
    <tr>
      <td>传入的mode参数不在0、1、2、3范围内。 </td>
    </tr>
  </tbody>
  </table>

## aclnnRotaryPositionEmbeddingV2

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 849px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnRotaryPositionEmbeddingV2GetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream流。</td>
    </tr>
  </tbody>
  </table>
  
- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

  输入张量x支持BNSD、BSND、SBND排布。
  输入张量x、cos、sin及输出张量y的D维度大小必须相同，满足D<896，且必须为2的倍数。
  输入张量x和输出张量y的shape必须完全相同。
  输入张量cos和sin的shape必须完全相同.

  - half模式：
    - B，N < 1000;
    - 当x为BNSD时，cos、sin支持11SD、B1SD、BNSD
      - 当（D/2）% (32/inputDtypeSize) == 0时，需满足B \* N <= S * 8
      - 当（D/2）% (32/inputDtypeSize) != 0时，需满足B \* N * 2 <= (S + coreNum -1) / coreNum 或者 D >= 80
    - 当x为BSND时，cos、sin支持1S1D、BS1D、BSND
    - 当x为SBND时，cos、sin支持S11D、SB1D、SBND
  - interleave模式：
    - B * N < 1000
    - 当x为BNSD时，cos、sin支持11SD
    - 当x为BSND时，cos、sin支持1S1D
    - 当x为SBND时，cos、sin支持S11D

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
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

std::vector<float> get_interleave_matrix(int64_t n) {
    std::vector<float> matrix(n * n, 0.0f);
    for (int64_t i = 0; i < n; i += 2) {
        if (i + 1 >= n) {
            break;
        }
        matrix[i * n + (i + 1)] = 1.0f;
        matrix[(i + 1) * n + i] = 1.0f;
    }
    return matrix;
}

int main() {
    // 1. 固定写法，device/stream初始化, 参考acl API手册
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
    std::vector<int64_t> rotateShape = {128, 128};
    int64_t mode = 1;

    void* xDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    void* rotateDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;
    aclTensor* rotate = nullptr;
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
    std::vector<float> rotateHostData = get_interleave_matrix(128);
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
    // 创建rotate aclTensor
    ret = CreateAclTensor(rotateHostData, rotateShape, &rotateDeviceAddr, aclDataType::ACL_FLOAT, &rotate);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnRotaryPositionEmbeddingV2第一段接口
    ret = aclnnRotaryPositionEmbeddingGetWorkspaceSizeV2(x, cos, sin, mode, rotate, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRotaryPositionEmbeddingGetWorkspaceSizeV2 failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnRotaryPositionEmbeddingV2第二段接口
    ret = aclnnRotaryPositionEmbeddingV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRotaryPositionEmbeddingV2 failed. ERROR: %d\n", ret); return ret);
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
    aclDestroyTensor(rotate);
    aclDestroyTensor(out);

    // 7. 释放device 资源
    aclrtFree(xDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    aclrtFree(rotateDeviceAddr);
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

- 辅助矩阵rotate生成示例：

```python
import torch
import torch_npu

def get_interleave_matrix(n):
    matrix = torch.zeros(n, n, dtype=torch.bfloat16)
    for i in range(0, n, 2):
        matrix[i + 0, i + 1] = 1
        matrix[i + 1, i + 0] = -1
    return matrix

def get_half_matrix(n):
    matrix = torch.zeros(n, n, dtype=torch.bfloat16)
    half = n // 2
    matrix[:half, half:] = torch.eye(half)
    matrix[half:, :half] = -torch.eye(half)
    return matrix

def compose_2matrix(A, B):
    total_rows = A.size(0) + B.size(0)
    total_cols = A.size(1) + B.size(1)
    result = torch.zeros(total_rows, total_cols, dtype=torch.bfloat16)
    result[:A.size(0), :A.size(1)] = A
    b_row_start = A.size(0)
    b_col_start = A.size(1)
    result[b_row_start:b_row_start + B.size(0), 
           b_col_start:b_col_start + B.size(1)] = B
    return result

def main():
    # interleave
    inter_mat_128 = get_interleave_matrix(128)
    inter_mat_64 = get_interleave_matrix(64)
    # interleave 2D
    inter_mat_128_64 = compose_2matrix(inter_mat_128, inter_mat_64)

    x = torch.rand(2, 2, 5, 128).npu()
    r1 = torch.rand(1, 2, 1, 128).npu()
    r2 = torch.rand(1, 2, 1, 128).npu()
    out = torch_npu.npu_rotary_mul(x, r1, r2, "interleave", inter_mat_128.npu())
```
