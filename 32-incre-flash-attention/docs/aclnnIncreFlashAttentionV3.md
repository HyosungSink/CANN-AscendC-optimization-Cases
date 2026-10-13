# aclnnIncreFlashAttentionV3

**须知：该接口后续版本会废弃，请使用最新接口[aclnnIncreFlashAttentionV4](aclnnIncreFlashAttentionV4.md)。**

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | -------- |
| <term>Ascend 950PR/Ascend 950DT</term>                             | ×        |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     | ×        |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √        |
| <term>Atlas 200I/500 A2 推理产品</term>                      | ×        |
| <term>Atlas 推理系列加速卡产品</term>                        | √        |
| <term>Atlas 训练系列产品</term>                              | ×        |

## 功能说明

- 接口功能：兼容（[aclnnIncreFlashAttentionV2](aclnnIncreFlashAttentionV2.md)）接口功能，在其基础上**新增位置编码，page attention，KV cache反量化特性**。

  对于自回归（Auto-regressive）的语言模型，随着新词的生成，推理输入长度不断增大。在原来全量推理的基础上**实现增量推理**，query的S轴固定为1，key和value是经过KV Cache后，将之前推理过的state信息，叠加在一起，每个Batch对应S轴的实际长度可能不一样，输入的数据是经过padding后的固定长度数据。

  相比全量场景的FlashAttention算子（[PromptFlashAttention](../../prompt_flash_attention/README.md)），增量推理的流程与正常全量推理并不完全等价，不过增量推理的精度并无明显劣化。

    **说明：**

    KV Cache是大模型推理性能优化的一个常用技术。采样时，Transformer模型会以给定的prompt/context作为初始输入进行推理（可以并行处理），随后逐一生成额外的token来继续完善生成的序列（体现了模型的自回归性质）。在采样过程中，Transformer会执行自注意力操作，为此需要给当前序列中的每个项目（无论是prompt/context还是生成的token）提取键值（KV）向量。这些向量存储在一个矩阵中，通常被称为kv缓存（KV Cache）。

- 计算公式：

  self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*$d的矩阵。

    self-attention的计算公式一般定义如下，其中$Q$、$K$、$V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。

  $$
     Attention(Q,K,V)=Score(Q,K)V
  $$

    本算子中Score函数采用Softmax函数，self-attention计算公式为：

  $$
   Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

    其中$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n*d$的矩阵。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnIncreFlashAttentionV3GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnIncreFlashAttentionV3”接口执行计算。

```cpp
aclnnStatus aclnnIncreFlashAttentionV3GetWorkspaceSize(
    const aclTensor     *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor     *pseShift,
    const aclTensor     *attenMask,
    const aclIntArray   *actualSeqLengths,
    const aclTensor     *dequantScale1,
    const aclTensor     *quantScale1,
    const aclTensor     *dequantScale2,
    const aclTensor     *quantScale2,
    const aclTensor     *quantOffset2,
    const aclTensor     *antiquantScale,
    const aclTensor     *antiquantOffset,
    const aclTensor     *blocktable,
    int64_t              numHeads,
    double               scaleValue,
    char                *inputLayout,
    int64_t              numKeyValueHeads,
    int64_t              blockSize,
    int64_t              innerPrecise,
    const aclTensor     *attentionOut,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```cpp
aclnnStatus aclnnIncreFlashAttentionV3(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnIncreFlashAttentionV3GetWorkspaceSize

- **参数说明**

  <div style="overflow-x: auto;">
  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup> 
  <col style="width: 190px"> 
  <col style="width: 120px"> 
  <col style="width: 300px"> 
  <col style="width: 330px"> 
  <col style="width: 212px"> 
  <col style="width: 100px">  
  <col style="width: 170px">  
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
        <td>query</td>
        <td>输入</td>
        <td>公式中的输入Q。</td>
        <td>query和attentionOut的shape需要完全一致。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td><ul><li>(B, N, S, D)</li><li>(B, S, N, D)</li><li>(B, S, H)</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>key</td>
        <td>输入</td>
        <td>公式中的输入K。</td>
        <td>key、value 中对应tensor的shape需要完全一致。</td>
        <td>FLOAT16、BFLOAT16、INT8</td>
        <td>ND</td>
        <td><ul><li>(B, N, S, D)</li><li>(B, S, N, D)</li><li>(B, S, H)</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>value</td>
        <td>输入</td>
        <td>公式中的输入V。</td>
        <td>key、value 中对应tensor的shape需要完全一致。</td>
        <td>FLOAT16、BFLOAT16、INT8</td>
        <td>ND</td>
        <td><ul><li>(B, N, S, D)</li><li>(B, S, N, D)</li><li>(B, S, H)</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>pseShift</td>
        <td>输入</td>
        <td>位置编码。</td>
        <td><ul><li>支持空Tensor。</li><li>pseShift数据类型需与query数据类型保持一致。</li></ul></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td><ul><li>(B, N, 1, S)</li><li>(1, N, 1, S)</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>attenMask</td>
        <td>输入</td>
        <td>attention掩码矩阵。</td>
        <td><ul><li>支持空Tensor。</li><li>当attenMask数据类型取INT8、UINT8时，其tensor中的值需要为0或1。</li></ul></td>
        <td>BOOL、INT8、UINT8</td>
        <td>ND</td>
        <td><ul><li>(B, S)</li><li>(B, 1, S)</li><li>(B, 1, 1, S)</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>actualSeqLengths</td>
        <td>输入</td>
        <td>key和value的S轴实际长度。</td>
        <td>综合约束请见<a href="#约束说明">约束说明</a>。</td>
        <td>INT64</td>
        <td>ND</td>
        <td>(B)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dequantScale1</td>
        <td>输入</td>
        <td>BMM1后面反量化的量化因子。</td>
        <td><ul><li>支持空Tensor。</li><li>支持per-tensor（scalar）。</li></ul></td>
        <td>UINT64、FLOAT32</td>
        <td>ND</td>
        <td>(1)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>quantScale1</td>
        <td>输入</td>
        <td>BMM2前面量化的量化因子。</td>
        <td><ul><li>支持空Tensor。</li><li>支持per-tensor（scalar）。</li></ul></td>
        <td>FLOAT32、BFLOAT16</td>
        <td>ND</td>
        <td>(1)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>dequantScale2</td>
        <td>输入</td>
        <td>BMM2后面反量化的量化因子。</td>
        <td><ul><li>支持空Tensor。</li><li>支持per-tensor（scalar）。</li></ul></td>
        <td>UINT64、FLOAT32</td>
        <td>ND</td>
        <td>(1)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>quantScale2</td>
        <td>输入</td>
        <td>输出量化的量化因子。</td>
        <td><ul><li>支持空Tensor。</li><li>支持per-tensor，per-channel。</li></ul></td>
        <td>FLOAT32、BFLOAT16</td>
        <td>ND</td>
        <td><ul><li>(1)</li><li>(D * N)，N为numKeyValueHeads</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>quantOffset2</td>
        <td>输入</td>
        <td>输出量化的量化偏移。</td>
        <td><ul><li>支持空Tensor。</li><li>支持per-tensor，per-channel。</li></ul></td>
        <td>FLOAT32、BFLOAT16</td>
        <td>ND</td>
        <td><ul><li>(1)</li><li>(D * N)，N为numKeyValueHeads</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>antiquantScale</td>
        <td>输入</td>
        <td>伪量化因子。</td>
        <td><ul><li>支持空Tensor。</li><li>支持per-channel（list），由shape决定，BNSD场景下shape为(2, N, 1, D)，BSH场景下shape为(2, H)，BSND场景下shape为(2, N, D)。</li></ul></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>-</td>
        <td><ul><li>(2, N, 1, D)</li><li>(2, H)</li><li>(2, N, D)</li><li>(2)</li></ul></td>
        <td>×</td>
      </tr>
        <tr>
        <td>antiquantOffset</td>
        <td>输入</td>
        <td>伪量化因子。</td>
        <td><ul><li>支持空Tensor。</li><li>支持per-channel（list），由shape决定，BNSD场景下shape为(2, N, 1, D)，BSH场景下shape为(2, H)，BSND场景下shape为(2, N, D)。</li></ul></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td><ul><li>(2, N, 1, D)</li><li>(2, H)</li><li>(2, N, D)</li><li>(2)</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>blocktable</td>
        <td>输入</td>
        <td>page attention中KV存储使用的block映射表。</td>
        <td><ul><li>支持空Tensor。</li><li>综合约束请见<a href="#约束说明">约束说明</a></li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B, maxBlockNumPerSeq)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>numHeads</td>
        <td>输入</td>
        <td>query的head个数。</td>
        <td>numHeads是numKeyValueHeads的倍数关系。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>输入</td>
        <td>公式中d开根号的倒数。</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>输入</td>
        <td>标识输入query、key、value的数据排布格式。</td>
        <td>当前支持BSH、BNSD、BSND。用户不特意指定时建议传入"BSH"。</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numKeyValueHeads</td>
        <td>输入</td>
        <td>key、value中head个数。</td>
        <td><ul><li>用于支持GQA（Grouped-Query Attention，分组查询注意力）场景，传入0表示和query的head个数相等。</li><li>综合约束请见<a href="#约束说明">约束说明</a>。</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
        <tr>
        <td>blockSize</td>
        <td>输入</td>
        <td>page attention中KV存储每个block中最大的token个数。</td>
        <td><ul><li>用户不特意指定时建议传入0。</li><li>综合约束请见<a href="#约束说明">约束说明</a>。</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
        <tr>
        <td>innerPrecise</td>
        <td>输入</td>
        <td>代表高精度/高性能选择。</td>
        <td>0代表高精度，1代表高性能，用户不特意指定时建议传入1。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attentionOut</td>
        <td>输出</td>
        <td>公式中的输出。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、INT8</td>
        <td>ND</td>
        <td><ul><li>(B, N, S, D)</li><li>(B, S, N, D)</li><li>(B, S, H)</li></ul></td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回用户需要在Device侧申请的workspace大小。</td>
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
    </tbody></table></div>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 280px">
  <col style="width: 119px">
  <col style="width: 751px">
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
      <td>query、key、value、pseShift、attenMask、attentionOut的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>API内存调用npu runtime的接口异常。</td>
    </tr>
  </tbody>
  </table>

## aclnnIncreFlashAttentionV3

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnIncreFlashAttentionV3GetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnIncreFlashAttentionV3默认确定性实现。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：
  - 支持B轴小于等于65536，N轴小于等于256，D轴小于等于512。
  - query数据类型支持FLOAT16、BFLOAT16，attentionOut、key和value数据类型支持FLOAT16、INT8、BFLOAT16。
  - dequantScale1、dequantScale2数据类型支持UINT64、FLOAT32。
  - quantScale1、quantScale2和quantOffset2数据类型支持FLOAT32
  - numKeyValueHeads数据类型支持INT64。
- <term>Atlas 推理系列加速卡产品</term>：
  - 支持B轴小于等于256，N轴小于等于256，D轴小于等于512。
  - 支持key、value的S轴小于等于65536。
  - query、key、value和attentionOut数据类型仅支持FLOAT16。
  - dequantScale1、quantScale1、dequantScale2、quantScale2和quantOffset2仅支持nullptr。
  - numKeyValueHeads仅支持取值0。
- 非连续场景下，参数key、value的tensorlist中tensor的个数等于query的B(由于tensorlist限制，非连续场景下B需要小于等于256)。shape除S外需要完全一致，且batch只能为1。
- 参数query中的N和numHeads值相等，key、value的N和numKeyValueHeads值相等，并且numHeads是numKeyValueHeads的倍数关系。
- 仅支持query的S轴等于1。
- 当attenMask数据类型取INT8、UINT8时，其tensor中的值需要为0或1。
- query、key、value输入均为INT8的场景暂不支持。
- query、key、value输入为FLOAT16，输出为INT8的场景：入参quantScale2必填，quantOffset2可选，不能传入dequantScale1、quantScale1、dequantScale2（即为nullptr）参数。

- antiquantScale和antiquantOffset参数约束：
  - per-channel模式：两个参数的shape可支持(2, N, 1, D)，(2, N, D)，(2, H)，N为numKeyValueHeads。参数数据类型和query数据类型相同。
  - per-tensor模式：两个参数的shape均为(2)，数据类型和query数据类型相同。

- 入参 quantScale2 和 quantOffset2 支持 per-tensor/per-channel 两种格式和 FLOAT32/BFLOAT16 两种数据类型。若传入 quantOffset2 ，需保证其类型和shape信息与 quantScale2 一致。当输入为BFLOAT16时，同时支持FLOAT32和BFLOAT16，否则仅支持FLOAT32 。per-channel 格式，当输出layout为BSH时，要求 quantScale2 所有维度的乘积等于H；其他layout要求乘积等于N*D。（建议输出layout为BSH时，quantScale2 shape传入[1,1,H]或[H]；输出为BNSD时，建议传入[1,N,1,D]或[N,D]；输出为BSND时，建议传入[1,1,N,D]或[N,D]）。

- page attention场景:
  - page attention的使能必要条件是blockTable存在且有效，同时key、value是按照blockTable中的索引在一片连续内存中排布，支持key、value dtype为FLOAT16/BFLOAT16/INT8，在该场景下key、value的inputLayout参数无效。
  - blockSize是用户自定义的参数，该参数的取值会影响page attention的性能，在使能page attention场景下，blockSize需要传入非0值, 且blocksize最大不超过512。key、value输入类型为FLOAT16/BFLOAT16时需要16对齐，key、value 输入类型为INT8时需要32对齐，推荐使用128。通常情况下，page attention可以提高吞吐量，但会带来性能上的下降。
  - page attention场景下，当query的inputLayout为BNSD时，kv cache排布支持（blocknum, blocksize, H）和（blocknum, KV_N, blocksize, D）两种格式，当query的inputLayout为BSH、BSND时，kv cache排布只支持（blocknum, blocksize, H）一种格式。blocknum不能小于根据actualSeqLengthsKv和blockSize计算的每个batch的block数量之和。且key和value的shape需保证一致。
  - page attention场景下，kv cache排布为（blocknum, KV_N, blocksize, D）时性能通常优于kv cache排布为（blocknum, blocksize, H）时的性能，建议优先选择（blocknum, KV_N, blocksize, D）格式。
  - page attention使能场景下，当输入kv cache排布格式为（blocknum, blocksize, H），且 numKeyValueHeads * headDim 超过64k时，受硬件指令约束，会被拦截报错。可通过使能GQA（减小 numKeyValueHeads）或调整kv cache排布格式为（blocknum, numKeyValueHeads, blocksize, D）解决。
  - page attention场景下，必须传入输入actualSeqLengths。
  - page attention场景下，blockTable必须为二维，第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大actualSeqLengthsKv对应的block数量）。
  - page attention使能场景下，以下场景输入S需要大于等于maxBlockNumPerSeq * blockSize。
  - 使能 Attention mask，例如 mask shape为 (B, 1, 1, S)。
  - 使能 pseShift，例如 pseShift shape为(B, N, 1, S)。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```cpp
#include <iostream>
#include <vector>
#include <math.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_incre_flash_attention_v3.h"

using namespace std;

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
  int32_t batchSize = 1;
  int32_t numHeads = 2;
  int32_t headDims = 16;
  int32_t keyNumHeads = 2;
  int32_t sequenceLengthKV = 16;
  std::vector<int64_t> queryShape = {batchSize, numHeads, 1, headDims}; // BNSD
  std::vector<int64_t> keyShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
  std::vector<int64_t> valueShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
  std::vector<int64_t> attenShape = {batchSize, 1, 1, sequenceLengthKV}; // B11S
  std::vector<int64_t> outShape = {batchSize, numHeads, 1, headDims}; // BNSD
  void *queryDeviceAddr = nullptr;
  void *keyDeviceAddr = nullptr;
  void *valueDeviceAddr = nullptr;
  void *attenDeviceAddr = nullptr;
  void *outDeviceAddr = nullptr;
  aclTensor *queryTensor = nullptr;
  aclTensor *keyTensor = nullptr;
  aclTensor *valueTensor = nullptr;
  aclTensor *attenTensor = nullptr;
  aclTensor *outTensor = nullptr;
  std::vector<float> queryHostData(batchSize * numHeads * headDims, 1.0f);
  std::vector<float> keyHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
  std::vector<float> valueHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
  std::vector<int8_t> attenHostData(batchSize * sequenceLengthKV, 0);
  std::vector<float> outHostData(batchSize * numHeads * headDims, 1.0f);

  // 创建query aclTensor
  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建key aclTensor
  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int kvTensorNum = 1;
  aclTensor *tensorsOfKey[kvTensorNum];
  tensorsOfKey[0] = keyTensor;
  auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
  // 创建value aclTensor
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor *tensorsOfValue[kvTensorNum];
  tensorsOfValue[0] = valueTensor;
  auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
  // 创建atten aclTensor
  ret = CreateAclTensor(attenHostData, attenShape, &attenDeviceAddr, aclDataType::ACL_INT8, &attenTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> actualSeqlenVector = {sequenceLengthKV};
  auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());

  int64_t numKeyValueHeads = numHeads;
  int64_t blockSize = 1;
  int64_t innerPrecise = 1;
  double scaleValue = 1 / sqrt(headDims); // 1/sqrt(d)
  string sLayerOut = "BNSD";
  char layerOut[sLayerOut.length()+1];
  strcpy(layerOut, sLayerOut.c_str());
  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用第一段接口
  ret = aclnnIncreFlashAttentionV3GetWorkspaceSize(queryTensor, tensorKeyList, tensorValueList, nullptr, attenTensor, actualSeqLengths, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                   nullptr, nullptr, nullptr, numHeads, scaleValue, layerOut, numKeyValueHeads, blockSize, innerPrecise, outTensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIncreFlashAttentionV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用第二段接口
  ret = aclnnIncreFlashAttentionV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIncreFlashAttentionV3 failed. ERROR: %d\n", ret); return ret);

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

  // 6. 释放资源
  aclDestroyTensor(queryTensor);
  aclDestroyTensor(keyTensor);
  aclDestroyTensor(valueTensor);
  aclDestroyTensor(attenTensor);
  aclDestroyTensor(outTensor);
  aclDestroyIntArray(actualSeqLengths);
  aclrtFree(queryDeviceAddr);
  aclrtFree(keyDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(attenDeviceAddr);
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
