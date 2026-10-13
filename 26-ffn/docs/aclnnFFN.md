# aclnnFFN

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/ffn/ffn)

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 接口功能：该FFN算子提供MoeFFN和FFN的计算功能。在没有专家分组（expertTokens为空）时是FFN，有专家分组时是MoeFFN，统称为FFN，属于Moe结构。MoE（Mixture-of-Experts，混合专家系统）是一种用于训练万亿参数量级模型的技术。MoE将预测建模任务分解为若干子任务，在每个子任务上训练一个专家模型（Expert Model），开发一个门控模型（Gating Model），该模型会根据输入数据分配一个或多个专家，最终综合多个专家计算结果作为预测结果。Mixture-of-Experts结构的模型是将输入数据分配给最相关的一个或者多个专家，综合涉及的所有专家的计算结果来确定最终结果。
- 计算公式：

  - **非量化场景：**
  $$
  y=activation(x * W1 + b1) * W2 + b2
  $$

  - **量化场景：**
  $$
  y=((activation((x * W1 + b1) * deqScale1) * scale + offset) * W2 + b2) * deqScale2
  $$

  - **伪量化场景：**
  $$
  y=activation(x * ((W1 + antiquantOffset1) * antiquantScale1) + b1) * ((W2 + antiquantOffset2) * antiquantScale2) + b2
  $$

**说明：**

FFN在无专家或单个专家场景是否有性能收益需要根据实际测试情况判断，当整网中FFN结构对应的小算子vector耗时超过30us，且在FFN结构中占比10%以上时，可以尝试使用该融合算子，若实际测试性能劣化则不使用。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFFNGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFFN”接口执行计算。

```Cpp
aclnnStatus aclnnFFNGetWorkspaceSize(
  const aclTensor*   x, 
  const aclTensor*   weight1, 
  const aclTensor*   weight2,
  const aclIntArray* expertTokens, 
  const aclTensor*   bias1, 
  const aclTensor*   bias2, 
  const aclTensor*   scale, 
  const aclTensor*   offset, 
  const aclTensor*   deqScale1, 
  const aclTensor*   deqScale2, 
  const aclTensor*   antiquantScale1, 
  const aclTensor*   antiquantScale2, 
  const aclTensor*   antiquantOffset1, 
  const aclTensor*   antiquantOffset2, 
  const char*        activation, 
  int64_t            innerPrecise, 
  const aclTensor*   y, 
  uint64_t*          workspaceSize, 
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnFFN(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnFFNGetWorkspaceSize

**说明：**  
M表示token个数，对应transform中的BS（B：Batch，表示输入样本批量大小，S：  Seq-Length，表示输入样本序列长度）；  
K1表示第一个matmul的输入通道数，对应transform中的H（Head-Size，表示隐藏层的大小）；  
N1表示第一个matmul的输出通道数；K2表示第二个matmul的输入通道数；  
N2表示第二个matmul的输出通道数，对应transform中的H；  
E表示有专家场景的专家数。  
G表示伪量化per-group场景下，antiquantOffset、antiquantScale的组数。

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1650px">
  <colgroup>
  <col style="width: 200px"> <!-- 参数名 -->
  <col style="width: 120px"> <!-- 输入/输出 -->
  <col style="width: 280px">  <!-- 描述 -->
  <col style="width: 300px">  <!-- 使用说明 -->
  <col style="width: 250px">  <!-- 数据类型 -->
  <col style="width: 120px">  <!-- 数据格式 -->
  <col style="width: 240px"> <!-- 维度(shape) -->
  <col style="width: 140px">  <!-- 非连续Tensor -->
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
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>计算输入，公式中的输入x。</td>
      <td>
        <ul>
          <li>不支持空Tensor。</li>
          <li>无Optional后缀，不可传空指针。</li>
        </ul>
      </td>
      <td>FLOAT16、BFLOAT16、INT8</td>
      <td>ND</td>
      <td>2-8维[M, K1]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight1（aclTensor*）</td>
      <td>输入</td>
      <td>专家的权重数据，公式中的W1。</td>
      <td>
        <ul>
          <li>不支持空Tensor。</li>
          <li>无Optional后缀，不可传空指针。</li>
        </ul>
      </td>
      <td>FLOAT16、BFLOAT16、INT8、INT4</td>
      <td>ND</td>
      <td>有专家[E,K1,N1]<br>无专家[K1,N1]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight2（aclTensor*）</td>
      <td>输入</td>
      <td>专家的权重数据，公式中的W2。</td>
      <td>
        <ul>
          <li>不支持空Tensor。</li>
          <li>无Optional后缀，不可传空指针。</li>
        </ul>
      </td>
      <td>FLOAT16、BFLOAT16、INT8、INT4</td>
      <td>ND</td>
      <td>有专家[E,K2,N2]<br>无专家[K2,N2]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expertTokens（aclIntArray*）</td>
      <td>可选输入</td>
      <td>各专家的token数。</td>
      <td>
        <ul>
          <li>支持传空指针（空Tensor）。</li>
          <li>无Optional后缀，传空指针无token数约束。</li>
          <li>非空时最大长度256。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>1维，最大长度256</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias1（aclTensor*）</td>
      <td>可选输入</td>
      <td>权重数据修正值，公式中的b1。</td>
      <td>
        <ul>
          <li>支持空Tensor，可传空指针。</li>
          <li>无Optional后缀，传空指针无偏置约束。</li>
        </ul>
      </td>
      <td>FLOAT16、FLOAT32、INT32</td>
      <td>ND</td>
      <td>有专家[E,N1]<br>无专家[N1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias2（aclTensor*）</td>
      <td>可选输入</td>
      <td>权重数据修正值，公式中的b2。</td>
      <td>
        <ul>
          <li>支持空Tensor，可传空指针。</li>
          <li>无Optional后缀，传空指针无偏置约束。</li>
        </ul>
      </td>
      <td>FLOAT16、FLOAT32、INT32</td>
      <td>ND</td>
      <td>有专家[E,N2]<br>无专家[N2]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale（aclTensor*）</td>
      <td>可选输入</td>
      <td>量化参数，量化缩放系数。</td>
      <td>
        <ul>
          <li>支持空Tensor，可传空指针。</li>
          <li>无Optional后缀，传空指针无缩放约束。</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>per-tensor 一维，有专家[E]，无专家[1]<br>per-channel 有专家[E,N1]，无专家[N1]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offset（aclTensor*）</td>
      <td>可选输入</td>
      <td>量化参数，量化偏移量。</td>
      <td>
        <ul>
          <li>支持空Tensor，可传空指针。</li>
          <li>无Optional后缀，传空指针无偏移约束。</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1维，有专家[E]<br>无专家[1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deqScale1（aclTensor*）</td>
      <td>可选输入</td>
      <td>量化参数，第一个matmul的反量化缩放系数。</td>
      <td>
        <ul>
          <li>支持空Tensor，可传空指针。</li>
          <li>无Optional后缀，传空指针无反量化约束。</li>
        </ul>
      </td>
      <td>UINT64、INT64、FLOAT32、BFLOAT16</td>
      <td>ND</td>
      <td>有专家[E,N1]<br>无专家[N1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deqScale2（aclTensor*）</td>
      <td>可选输入</td>
      <td>量化参数，第二个matmul的反量化缩放系数。</td>
      <td>
        <ul>
          <li>支持空Tensor，可传空指针。</li>
          <li>无Optional后缀，传空指针无反量化约束。</li>
        </ul>
      </td>
      <td>UINT64、INT64、FLOAT32、BFLOAT16</td>
      <td>ND</td>
      <td>有专家[E,N2]<br>无专家[N2]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantScale1（aclTensor*）</td>
      <td>可选输入</td>
      <td>伪量化参数，第一个matmul的缩放系数。</td>
      <td>
        <ul>
          <li>支持空Tensor，可传空指针。</li>
          <li>无Optional后缀，传空指针无伪量化约束。</li>
        </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>per-channel 有专家[E,N1]，无专家[N1]<br>per-group 有专家[E,G,N1]，无专家[G,N1]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>antiquantScale2（aclTensor*）</td>
      <td>可选输入</td>
      <td>伪量化参数，第二个matmul的缩放系数。</td>
      <td>
        <ul>
          <li>支持空Tensor，可传空指针。</li>
          <li>无Optional后缀，传空指针无伪量化约束。</li>
        </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>per-channel 有专家[E,N2]，无专家[N2]<br>per-group 有专家[E,G,N2]，无专家[G,N2]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>antiquantOffset1（aclTensor*）</td>
      <td>可选输入</td>
      <td>伪量化参数，第一个matmul的偏移量。</td>
      <td>
        <ul>
          <li>支持空Tensor，可传空指针。</li>
          <li>无Optional后缀，传空指针无伪量化约束。</li>
        </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>per-channel 有专家[E,N1]，无专家[N1]<br>per-group 有专家[E,G,N1]，无专家[G,N1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantOffset2（aclTensor*）</td>
      <td>可选输入</td>
      <td>伪量化参数，第二个matmul的偏移量。</td>
      <td>
        <ul>
          <li>支持空Tensor，可传空指针。</li>
          <li>无Optional后缀，传空指针无伪量化约束。</li>
        </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>per-channel 有专家[E,N2]，无专家[N2]<br>per-group 有专家[E,G,N2]，无专家[G,N2]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activation（char*）</td>
      <td>输入</td>
      <td>使用的激活函数，公式中的activation。</td>
      <td>
        <ul>
          <li>无空Tensor概念，不可传空指针。</li>
          <li>无Optional后缀，必须传有效值。</li>
          <li>取值支持fastgelu/gelu/relu/silu/geglu/swiglu/reglu。</li>
        </ul>
      </td>
      <td>CHAR</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPrecise（int64_t）</td>
      <td>可选输入</td>
      <td>高精度或者高性能选择。</td>
      <td>
        <ul>
          <li>无空Tensor概念，可传默认值。</li>
          <li>无Optional后缀，仅对FLOAT16生效。</li>
          <li>0=高精度（FLOAT32计算），1=高性能。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y（aclTensor*）</td>
      <td>输出</td>
      <td>计算输出，公式中的输出y。</td>
      <td>
        <ul>
          <li>不支持空Tensor。</li>
          <li>无Optional后缀，不可传空指针。</li>
        </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>与x维度一致</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回Device侧需申请的workspace大小。</td>
      <td>
        <ul>
          <li>无空Tensor概念，不可传空指针。</li>
          <li>无Optional后缀，返回非负整数。</li>
        </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含算子计算流程。</td>
      <td>
        <ul>
          <li>无空Tensor概念，不可传空指针。</li>
          <li>无Optional后缀，返回的执行器需释放资源。</li>
        </ul>
      </td>
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

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 287px">
  <col style="width: 119px">
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
      <td>x、weight1、weight2、activation、expertTokens、bias1、bias2、y的数据类型和数据格式不在支持的范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnFFN

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFFNGetWorkspaceSize获取。</td>
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
  - aclnnFFN默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
- 有专家时，专家数据的总数需要与x的M保持一致。
- 激活层为geglu/swiglu/reglu时，仅支持无专家分组时的FLOAT16高性能场景（FLOAT16场景指类型为aclTensor的必选参数数据类型都为FLOAT16的场景），且N1=2\*K2。
- 激活层为gelu/fastgelu/relu/silu时，支持有专家或无专家分组的FLOAT16高精度及高性能场景、BFLOAT16场景、量化场景及伪量化场景，且N1=K2。
- 所有场景下需满足K1=N2, K1<65536, K2<65536, M轴在32Byte对齐后小于INT32的最大值。
- 非量化场景不能输入量化参数和伪量化参数，量化场景不能输入伪量化参数，伪量化场景不能输入量化参数。
- 量化场景参数类型：x为INT8、weight为INT8、bias为INT32、scale为FLOAT32、offset为FLOAT32，其余参数类型根据y不同分两种情况：
  - y为FLOAT16，deqScale支持数据类型：UINT64、INT64、FLOAT32。
  - y为BFLOAT16，deqScale支持数据类型：BFLOAT16。
  - 要求deqScale1与deqScale2的数据类型保持一致。
- 量化场景支持scale的per-channel模式参数类型：x为INT8、weight为INT8、bias为INT32、scale为FLOAT32、offset为FLOAT32，其余参数类型根据y不同分两种情况：
  - y为FLOAT16，deqScale支持数据类型：UINT64、INT64。
  - y为BFLOAT16，deqScale支持数据类型：BFLOAT16。
  - 要求deqScale1与deqScale2的数据类型保持一致。
- 伪量化场景支持两种不同参数类型：
  - y为FLOAT16、x为FLOAT16、bias为FLOAT16，antiquantScale为FLOAT16、antiquantOffset为FLOAT16，weight支持数据类型INT8和INT4。
  - y为BFLOAT16、x为BFLOAT16、bias为FLOAT32，antiquantScale为BFLOAT16、antiquantOffset为BFLOAT16，weight支持数据类型INT8和INT4。
- 当weight1/weight2的数据类型为INT4时，其shape最后一维必须为偶数。
- 伪量化场景，per-group下，antiquantScale1和antiquantOffset1中的K1需要能整除组数G，antiquantScale2和antiquantOffset2中的K2需要能整除组数G。
- 伪量化场景，per-group下目前只支持weight是INT4数据类型的场景。
- innerPrecise参数在BFLOAT16非量化场景，只能配置为0；FLOAT16非量化场景，可以配置为0或者1；量化或者伪量化场景，0和1都可配置，但是配置后不生效。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ffn.h"
#include "aclnn/opdev/fp16_t.h"

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
  auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
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
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  std::vector<int64_t> weight1Shape = {2, 2};
  std::vector<int64_t> weight2Shape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* weight1DeviceAddr = nullptr;
  void* weight2DeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* weight1 = nullptr;
  aclTensor* weight2 = nullptr;
  std::vector<op::fp16_t> selfHostData = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  std::vector<op::fp16_t> outHostData = {0, 0, 0, 0};
  std::vector<op::fp16_t> weight1HostData = {0.1, 0.2, 0.3, 0.4};
  std::vector<op::fp16_t> weight2HostData = {0.4, 0.3, 0.2, 0.1};
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建weight1 aclTensor
  ret = CreateAclTensor(weight1HostData, weight1Shape, &weight1DeviceAddr, aclDataType::ACL_FLOAT16, &weight1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建weight2 aclTensor
  ret = CreateAclTensor(weight2HostData, weight2Shape, &weight2DeviceAddr, aclDataType::ACL_FLOAT16, &weight2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnFFN接口调用示例
  LOG_PRINT("test aclnnFFN\n");

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  // 调用aclnnFFN第一段接口
  ret = aclnnFFNGetWorkspaceSize(self, weight1, weight2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, "relu", 1, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFFNGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnFFN第二段接口
  ret = aclnnFFN(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFFN failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<op::fp16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(weight1);
  aclDestroyTensor(weight2);

  // 7. 释放device资源
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(weight1DeviceAddr);
  aclrtFree(weight2DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
