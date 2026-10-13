# aclnnNsaCompress

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|     x      |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|     √      |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：训练场景下，使用NSA Compress算法减轻long-context的注意力计算，实现在KV序列维度进行压缩。

- 计算公式：

    Nsa Compress正向计算公式如下：

$$
\tilde{K}_t^{\text{cmp}} = f_K^{\text{cmp}}(k_{:t}) = \left\{ \varphi(k_{id+1:id+l}) \bigg| 0 \leq i \leq \left\lfloor \frac{t-l}{d} \right\rfloor \right\}
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnNsaCompressGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNsaCompress”接口执行计算。

```c++
aclnnStatus aclnnNsaCompressGetWorkspaceSize(
  const aclTensor   *input, 
  const aclTensor   *weight, 
  const aclIntArray *actSeqLenOptional, 
  char              *layoutOptional, 
  int64_t            compressBlockSize, 
  int64_t            compressStride, 
  int64_t            actSeqLenType, 
  aclTensor         *output, 
  uint64_t          *workspaceSize, 
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnNsaCompress(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnNsaCompressGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1565px"><colgroup>
    <col style="width: 146px">
    <col style="width: 135px">
    <col style="width: 326px">
    <col style="width: 246px">
    <col style="width: 275px">
    <col style="width: 101px">
    <col style="width: 190px">
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
      </tr></thead>
    <tbody>
      <tr>
        <td>input</td>
        <td>输入</td>
        <td>表示待压缩张量。</td>
        <td>
          <ul>
            <li>不支持空Tensor。</li>
            <li>数据类型与weight数据类型一致。</li>
            <li>shape支持[T, N, D]。</li>
          </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>输入</td>
        <td>表示压缩权重。</td>
        <td>
          <ul>
            <li>不支持空Tensor。</li>
            <li>数据类型与input数据类型一致。</li>
            <li>weight与input的shape满足broadcast关系。</li>
          </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>actSeqLenOptional</td>
        <td>输入</td>
        <td>描述每个Batch对应的S大小。</td>
        <td>
          <ul>
            <li>当前不能为空。</li>
          </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
      </tr>
      <tr>
        <td>layoutOptional</td>
        <td>输入</td>
        <td>代表输入input的数据排布格式。</td>
        <td>
          <ul>
            <li>支持BSH、SBH、BSND、BNSD、TND。</li>
            <li>当前仅支持TND。</li>
          </ul>
        </td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compressBlockSize</td>
        <td>输入</td>
        <td>压缩滑窗大小。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compressStride</td>
        <td>输入</td>
        <td>两次压缩滑窗间隔大小。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actSeqLenType</td>
        <td>输入</td>
        <td>描述actSeqLenOptional数值类型。</td>
        <td>
          <ul>
            <li>可取值0或1。</li>
            <li>0：数值为cumsum结果；1：数值为每个batch序列大小。</li>
            <li>当前仅支持0。</li>
          </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>output</td>
        <td>输出</td>
        <td>压缩后的结果。</td>
        <td>
          <ul>
            <li>不支持空Tensor。</li>
            <li>数据类型与input保持一致。</li>
            <li>shape支持[T, N, D]。</li>
          </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
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

  - input数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N； 其中T是B和S合轴紧密排列的数据（每个batch的actSeqLen）、B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入input、weight、actSeqLenOptional或output是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>input和weight的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>input和weight的shape无法做broadcast。</td>
    </tr>
    <tr>
      <td>layoutOptional不合法。</td>
    </tr>
  </tbody>
  </table>

## aclnnNsaCompress

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnNsaCompressGetWorkspaceSize获取。</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnNsaCompress默认确定性实现。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- input和weight需要满足broadcast关系，input.shape[1]=weight.shape[1]，不支持input、weight为空输入。
- actSeqLenType目前仅支持取值0，即actSeqLenOptional需要是前缀和模式。
- actSeqLenOptional目前不支持为空。
- layoutOptional目前仅支持TND，此时input.shape[0]必须等于actSeqLenOptional[-1]。
- input.shape[1]=weight.shape[1]，需要小于等于128。
- input.shape[2]必须是16的倍数，上限256。
- weight.shape[0]=compressBlockSize，必须是16的倍数，上限128。
- compressStride必须是16的整数倍，并且compressBlockSize>=compressStride。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_nsa_compress.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape)
    {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<aclFloat16> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++)
    {
        LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                            shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}
int main()
{
    // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    void *inputDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;

    aclTensor *input = nullptr;
    aclTensor *weight = nullptr;
    aclIntArray *actSeqLenOptional = nullptr;
    aclTensor *output = nullptr;

    // 自定义输入与属性
    int64_t compressBlockSize = 32;
    int64_t compressStride = 32;
    int64_t actSeqLenType = 0; // 0是前缀和模式，1是count计数模式
    char *layout = "TND";
    int32_t batchSize = 1;
    int32_t sampleLen = 64;
    int32_t headNum = 4;
    int32_t headDim = 32;

    std::vector<int64_t> inputShape = {batchSize * sampleLen, headNum, headDim};
    std::vector<int64_t> weightShape = {compressBlockSize, headNum};
    std::vector<int64_t> actSeqShape = {batchSize};
    std::vector<aclFloat16> inputHostData(batchSize * sampleLen * headNum * headDim);
    std::vector<aclFloat16> weightHostData(compressBlockSize * headNum);
    std::vector<int64_t> actSeqHostData(batchSize);

    for (int i = 0; i < inputHostData.size(); i++)
    {
        inputHostData[i] = aclFloatToFloat16(1.0);
    }
    for (int i = 0; i < weightHostData.size(); i++)
    {
        weightHostData[i] = aclFloatToFloat16(1.0);
    }

    int outputNum = 0;
    int preActSeqLen = 0;
    for (int i = 0; i < batchSize; i++)
    {
        if (actSeqLenType == 0)
        {
            actSeqHostData[i] = sampleLen + preActSeqLen;
            preActSeqLen = actSeqHostData[i];
        }
        else if (actSeqLenType == 1)
        {
            actSeqHostData[i] = sampleLen;
        }
        if (sampleLen >= compressBlockSize)
        {
            outputNum += (sampleLen - compressBlockSize) / compressStride + 1;
        }
    }

    std::vector<int64_t> outputShape = {outputNum, headNum, headDim};
    std::vector<aclFloat16> outputHostData(outputNum * headNum * headDim);

    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT16, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    actSeqLenOptional = aclCreateIntArray(actSeqHostData.data(), actSeqHostData.size());

    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    // 调用aclnnNsaCompressGetWorkspaceSize第一段接口
    ret = aclnnNsaCompressGetWorkspaceSize(input, weight, actSeqLenOptional, layout, compressBlockSize, compressStride,
                                        actSeqLenType, output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnNsaCompress第二段接口
    ret = aclnnNsaCompress(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompress failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(outputShape, &outputDeviceAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(input);
    aclDestroyTensor(weight);
    aclDestroyIntArray(actSeqLenOptional);
    aclDestroyTensor(output);

    // 7. 释放device资源
    aclrtFree(inputDeviceAddr);
    aclrtFree(weightDeviceAddr);
    // aclrtFree(actSeqDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
