
# aclnnFusedFloydAttention

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|     √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|     √     |
|<term>Atlas 200I/500 A2 推理产品</term>|     ×      |
|<term>Atlas  推理系列产品</term>|     ×     |
|<term>Atlas  训练系列产品</term>|     ×     |

## 功能说明

- 接口功能：训练场景下，使用FloydAttention算法实现多维自注意力的计算。

- 计算公式：

    注意力的正向计算公式如下：

    $$
    weights = Softmax(attenMask + scale*(einsum(query, key1^T) + einsum(query, key2^T)))
    $$

    $$
    attention\_out = einsum(weights, value1) + einsum(weights, value2)
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFusedFloydAttentionGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFusedFloydAttention”接口执行计算。

```Cpp
aclnnStatus aclnnFusedFloydAttentionGetWorkspaceSize(
    const aclTensor *query, 
    const aclTensor *key1, 
    const aclTensor *value1, 
    const aclTensor *key2, 
    const aclTensor *value2, 
    const aclTensor *attenMaskOptional, 
    double           scaleValueOptional, 
    const aclTensor *softmaxMaxOut, 
    const aclTensor *softmaxSumOut, 
    const aclTensor *attentionOutOut, 
    uint64_t        *workspaceSize, 
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnFusedFloydAttention(
    void             *workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor    *executor, 
    aclrtStream       stream)
```

## aclnnFusedFloydAttentionGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1565px"><colgroup>
      <col style="width: 146px">
      <col style="width: 135px">
      <col style="width: 326px">
      <col style="width: 246px">
      <col style="width: 275px">
      <col style="width: 120px">
      <col style="width: 171px">
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
          <td>query</td>
          <td>输入</td>
          <td>公式中的query。</td>
          <td>数据类型与key1/value1/key2/value2的数据类型一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>key1</td>
          <td>输入</td>
          <td>公式中的key1。</td>
          <td>数据类型与query/value1/key2/value2的数据类型一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,K,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>value1</td>
          <td>输入</td>
          <td>公式中的value1。</td>
          <td>数据类型与query/key1/key2/value2的数据类型一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,K,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>key2</td>
          <td>输入</td>
          <td>公式中的key2。</td>
          <td>数据类型与query/key1/value1/value2的数据类型一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,K,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>value2</td>
          <td>输入</td>
          <td>公式中的value2。</td>
          <td>数据类型与query/key1/value1/key2的数据类型一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,K,M,D]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>attenMaskOptional</td>
          <td>输入</td>
          <td>公式中的attenMask。</td>
          <td>取值为1代表该位不参与计算，为0代表该位参与计算。</td>
          <td>BOOL、UINT8</td>
          <td>ND</td>
          <td>[B,1,N,1,K]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>scaleValueOptional</td>
          <td>输入</td>
          <td>公式中的scale，代表缩放系数。</td>
          <td>-</td>
          <td>DOUBLE</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>softmaxMaxOut</td>
          <td>输出</td>
          <td>注意力正向计算的中间输出。</td>
          <td>输出的shape类型为[B,H,N,M,8]。</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>[B,H,N,M,8]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>softmaxSumOut</td>
          <td>输出</td>
          <td>注意力正向计算的中间输出。</td>
          <td>输出的shape类型为[B,H,N,M,8]。</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>[B,H,N,M,8]</td>
          <td>√</td>
        </tr>
        <tr>
          <td>attentionOutOut</td>
          <td>输出</td>
          <td>计算公式的最终输出。</td>
          <td>数据类型与query的数据类型一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>[B,H,N,M,D]</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1146px"><colgroup>
    <col style="width: 283px">
    <col style="width: 120px">
    <col style="width: 743px">
    </colgroup>
    <thead>
    <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
    </tr>
    <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>query、key1、value1、key2、value2、attenMaskOptional、softmaxMaxOut、softmaxSumOut、attentionOutOut的数据类型和数据格式不在支持的范围内。</td>
    </tr>
     <tr>
      <td rowspan="1">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="1">561002</td>
      <td>tiling发生异常，query、key1、value1、key2、value2、attenMaskOptional不符合约束说明。</td>
    </tr>
    </tbody>
    </table>

## aclnnFusedFloydAttention

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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnFusedFloydAttentionGetWorkspaceSize获取。</td>
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

## 约束说明<a name="1"></a>

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配
- 关于数据shape的约束，其中：
  - B：取值范围为1\~2K。
  - H：取值范围为1\~256。
  - N：取值范围为16\~1M且N%16==0。
  - M：取值范围为128\~1M且M%128==0。
  - K：取值范围为128\~1M且K%128==0。
  - D：取值范围为32/64/128。

- query与key1的第0/2/4轴需相同。
- key1与value1 shape需相同。
- key2与value2 shape需相同。
- softmaxMax与softmaxSum shape需相同。
- D只支持32/64/128。
- 由于底层指令限制，当M\*D>=65536或者K\*D>=65536时，会出现明显性能下降，此时建议使用小算子拼接替换实现。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
  
```c++
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_floyd_attention.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  size = size > 1000 ? 1000 : size; // 打印数据个数小于等于1000
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
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
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  int64_t B = 1;
  int64_t H = 32;
  int64_t N = 128;
  int64_t M = 128;
  int64_t K = 128;
  int64_t D = 32;
  double scaleValue = 1.0;

  int64_t q_size = B * H * N * M * D;
  int64_t kv_size = B * H * N * K * D;
  int64_t k1v1_size = B * H * K * M * D;
  int64_t atten_mask_size = B * H * N * M * K;

  std::vector<int64_t> qShape = {B, H, N, M, D};
  std::vector<int64_t> kShape = {B, H, N, K, D};
  std::vector<int64_t> k1Shape = {B, H, K, M, D};
  std::vector<int64_t> vShape = {B, H, N, K, D};
  std::vector<int64_t> v1Shape = {B, H, K, M, D};
  std::vector<int64_t> attenmaskShape = {B, H, N, M, K};
  std::vector<int64_t> attentionOutShape = {B, H, N, M, D};
  std::vector<int64_t> softmaxMaxShape = {B, H, N, M, 8};
  std::vector<int64_t> softmaxSumShape = {B, H, N, M, 8};

  void *qDeviceAddr = nullptr;
  void *kDeviceAddr = nullptr;
  void *vDeviceAddr = nullptr;
  void *k1DeviceAddr = nullptr;
  void *v1DeviceAddr = nullptr;
  void *attenmaskDeviceAddr = nullptr;
  void *attentionOutDeviceAddr = nullptr;
  void *softmaxMaxDeviceAddr = nullptr;
  void *softmaxSumDeviceAddr = nullptr;

  aclTensor *q = nullptr;
  aclTensor *k = nullptr;
  aclTensor *v = nullptr;
  aclTensor *k1 = nullptr;
  aclTensor *v1 = nullptr;
  aclTensor *attenMask = nullptr;
  aclTensor *softmaxMax = nullptr;
  aclTensor *softmaxSum = nullptr;
  aclTensor *attentionOut = nullptr;

  std::vector<float> qHostData(q_size, 1.0);
  std::vector<float> kHostData(kv_size, 1.0);
  std::vector<float> vHostData(kv_size, 1.0);
  std::vector<float> k1HostData(k1v1_size, 1.0);
  std::vector<float> v1HostData(k1v1_size, 1.0);
  std::vector<uint8_t> attenmaskHostData(atten_mask_size, 0);
  std::vector<float> attentionOutHostData(B*H*N*M*D, 0.0);
  std::vector<float> softmaxMaxHostData(B*H*N*M*8, 0.0);
  std::vector<float> softmaxSumHostData(B*H*N*M*8, 0.0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(k1HostData, k1Shape, &k1DeviceAddr, aclDataType::ACL_FLOAT16, &k1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(v1HostData, v1Shape, &v1DeviceAddr, aclDataType::ACL_FLOAT16, &v1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenMask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionOutHostData, attentionOutShape , &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16, &attentionOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  
  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  
  // 调用aclnnFusedFloydAttention第一段接口
  ret = aclnnFusedFloydAttentionGetWorkspaceSize(
      q, k, v, k1, v1, attenMask, scaleValue, softmaxMax, softmaxSum, attentionOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedFloydAttentionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  // 调用aclnnFusedFloydAttention第二段接口
  ret = aclnnFusedFloydAttention(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedFloydAttention failed. ERROR: %d\n", ret); return ret);
  
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(attentionOutShape, &attentionOutDeviceAddr);
  PrintOutResult(softmaxMaxShape, &softmaxMaxDeviceAddr);
  PrintOutResult(softmaxSumShape, &softmaxSumDeviceAddr);
  
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(k1);
  aclDestroyTensor(v1);
  aclDestroyTensor(attenMask);
  aclDestroyTensor(attentionOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  
  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(k1DeviceAddr);
  aclrtFree(v1DeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(attentionOutDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  
  return 0;
}
```
