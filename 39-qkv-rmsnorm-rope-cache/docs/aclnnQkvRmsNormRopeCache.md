# aclnnQkvRmsNormRopeCache

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>     |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：输入qkv融合张量，通过SplitVD拆分q、k、v张量，执行RmsNorm、ApplyRotaryPosEmb、Quant、Scatter融合操作，输出qOut、kCache、vCache、qBeforeQuant(可选)、kBeforeQuant(可选)、vBeforeQuant(可选)。
- 本接口目前支持的场景如下表：

  |场景类型|情况概要|
  |:---|:---|
  |<ul><li>cacheMode为PA_NZ</li><li>q无量化</li><li>k和v支持无量化、对称量化和非对称量化</li><li>qBeforeQuant/kBeforeQuant/vBeforeQuant不输出</li></ul>|qkv Shape为[$B_{qkv}$ * $S_{qkv}$, $N_{qkv}$ * $D_{qkv}$]，q、k、v具有完全相同的D维度。主要计算过程与输出对应关系：<br><ul><li>qkv 经过SplitVD->q、k、v</li><li>q经过RmsNorm、RoPE->qOut</li><li>k经过RmsNorm、RoPE、Quant(可选)、Scatter->kCache</li><li>v经过Quant(可选)、Scatter->vCache</li></ul>|

- 计算公式：

  (1) SplitVD:

  下式中，$N_q$、$N_k$、$N_v$分别表示 q、k、v 分量的注意力头数量，必须满足：

  $$
  \begin{cases}
  N_k = N_v \\
  N_{qkv} = N_k + N_v + N_q \\
  D_{qkv} = D_q = D_k = D_v
  \end{cases}
  $$

  $$
  \begin{aligned}
  q &= qkv[..., [:N_q] * D_{qkv}] \\
  k &= qkv[..., [N_q:-N_v] * D_{qkv}] \\
  v &= qkv[..., [-N_v:] * D_{qkv}]
  \end{aligned}
  $$

  (2) RmsNorm:

  此处x和y分别表示RmsNorm的输入张量和输出张量，归一化沿最后一维（feature dimension）进行，该计算规则通用于q、k分量。

  $$
  squareX = x * x
  $$

  $$
  meanSquareX = squareX.mean(dim = -1, keepdim = True)
  $$

  $$
  rms = \sqrt{meanSquareX + epsilon}
  $$

  $$
  y = (x / rms) * gamma
  $$

  (3) RoPE (Half-and-Half):

  此处的y指代完成RmsNorm计算的输出结果。

  $$
  y1 = y[\ldots, :d/2]
  $$

  $$
  y2 = y[\ldots, d/2:]
  $$

  $$
  y\_RoPE = torch.cat((-y2, y1), dim = -1)
  $$
  
  $$
  y\_embed = (y * cos) + y\_RoPE * sin
  $$

  (4) Quant:

  无量化：

  $$
  kQuant = kRoPE \\
  vQuant = v
  $$

  对称量化部分：

  $$
  kQuant = kRoPE / kScale \\
  vQuant = v / vScale
  $$

  非对称量化部分：
  
  $$
  kQuant = kRoPE / kScale + kOffset \\
  vQuant = v / vScale + vOffset
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnQkvRmsNormRopeCacheGetWorkspaceSize”接口获得入参并根据流程计算所需workspace大小，再调用“aclnnQkvRmsNormRopeCache”接口执行计算。

```Cpp
aclnnStatus aclnnQkvRmsNormRopeCacheGetWorkspaceSize(
  const aclTensor   *qkv,
  const aclTensor   *qGamma,
  const aclTensor   *kGamma,
  const aclTensor   *cos,
  const aclTensor   *sin,
  const aclTensor   *index,
  aclTensor         *qOut,
  aclTensor         *kCache,
  aclTensor         *vCache,
  const aclTensor   *kScaleOptional,
  const aclTensor   *vScaleOptional,
  const aclTensor   *kOffsetOptional,
  const aclTensor   *vOffsetOptional,
  const aclIntArray *qkvSize,
  const aclIntArray *headNums,
  double             epsilon,
  char              *cacheModeOptional,
  const aclTensor   *qOutBeforeQuant,
  const aclTensor   *kOutBeforeQuant,
  const aclTensor   *vOutBeforeQuant,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnQkvRmsNormRopeCache(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnQkvRmsNormRopeCacheGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1503px"><colgroup>
  <col style="width: 146px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 392px">
  <col style="width: 228px">
  <col style="width: 101px">
  <col style="width: 100px">
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
      <td>qkv</td>
      <td>输入</td>
      <td>用于切分出q、k、v的输入数据，对应计算公式中的qkv。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为[B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>qkv</sub> * D<sub>qkv</sub>]。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>qGamma</td>
      <td>输入</td>
      <td>用于q的rms_norm计算的输入数据，对应计算公式中的gamma。</td>
      <td><ul><li>不支持空Tensor。</li><li>与输入qkv的数据类型相同。</li><li>shape为[D<sub>qkv</sub>]。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kGamma</td>
      <td>输入</td>
      <td>用于k的rms_norm计算的输入数据，对应计算公式中的gamma。</td>
      <td><ul><li>不支持空Tensor。</li><li>与输入qkv的数据类型相同。</li><li>shape为[D<sub>qkv</sub>]。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>输入</td>
      <td>用于rope计算的输入数据，对输入张量进行余弦变换，对应计算公式中的cos。</td>
      <td><ul><li>不支持空Tensor。</li><li>与输入qkv的数据类型相同。</li><li>shape为[B<sub>qkv</sub> * S<sub>qkv</sub>, 1 * D_rope]，D_rope = D<sub>qkv</sub>且要求（D<sub>qkv</sub> * qkv数据类型所占字节数）可以被32整除。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>输入</td>
      <td>用于rope计算的输入数据，对输入张量进行正弦变换，对应计算公式中的sin。</td>
      <td><ul><li>不支持空Tensor。</li><li>与输入cos的数据类型、格式保持一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>index</td>
      <td>输入</td>
      <td>用于指定写入cache的具体索引位置，对应计算公式中的index。</td>
      <td><ul><li>不支持空Tensor。</li><li>与输入qkv的数据类型相同。</li><li>shape为[B<sub>qkv</sub> * S<sub>qkv</sub>]，要求index的value不能重复，value的数值范围为[-1, BlockNum * BlockSize]。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>qOut</td>
      <td>输入\输出</td>
      <td>提请申请的cache，输入输出同地址复用。</td>
      <td><ul><li>不支持空Tensor。</li><li>与输入qkv的数据类型相同。</li><li>shape为[B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>q</sub> * D<sub>qkv</sub>]。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kCache</td>
      <td>输入\输出</td>
      <td>提请申请的cache，输入输出同地址复用。</td>
      <td><ul><li>不支持空Tensor。</li><li>与输入qkv的数据类型相同（k不量化），或者INT8（k量化）。</li><li>shape为[BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 16, BlockSize, 16]（k不量化），或者[BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 32, BlockSize, 32]（k量化）。</li></ul></td>
      <td>FLOAT16、BFLOAT16、INT8</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vCache</td>
      <td>输入\输出</td>
      <td>提请申请的cache，输入输出同地址复用。</td>
      <td><ul><li>不支持空Tensor。</li><li>与输入qkv的数据类型相同（v不量化），或者INT8（v量化）。</li><li>shape为[BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 16, BlockSize, 16]（v不量化），或者[BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 32, BlockSize, 32]（v量化）。</li></ul></td>
      <td>FLOAT16、BFLOAT16、INT8</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kScaleOptional</td>
      <td>可选输入</td>
      <td>当kCache为INT8数据类型时需要存在。对应计算公式中的kScale。</td>
      <td>shape为[N<sub>k</sub>, D<sub>qkv</sub>]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vScaleOptional</td>
      <td>可选输入</td>
      <td>当vCache为INT8数据类型时需要存在。对应计算公式中的vScale。</td>
      <td>shape为[N<sub>v</sub>, D<sub>qkv</sub>]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kOffsetOptional</td>
      <td>可选输入</td>
      <td>对应计算公式中的kOffset。</td>
      <td><ul><li>当kCache数据类型为INT8且对应的kScaleOptional输入存在并量化场景为非对称量化时，需要此参数输入。</li><li>shape为[N<sub>k</sub>, D<sub>qkv</sub>]。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vOffsetOptional</td>
      <td>可选输入</td>
      <td>对应计算公式中的vOffset。</td>
      <td><ul><li>当vCache数据类型为INT8且对应的vScaleOptional输入存在并量化场景为非对称量化时，需要此参数输入。</li><li>shape为[N<sub>v</sub>, D<sub>qkv</sub>]。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>qkvSize</td>
      <td>输入</td>
      <td>按[B<sub>qkv</sub>, S<sub>qkv</sub>, N<sub>qkv</sub>, D<sub>qkv</sub>]顺序传入，提供输入参数qkv矩阵的B，S，N，D维度具体尺寸。</td>
      <td><ul><li>要求N<sub>qkv</sub> = N<sub>q</sub> + N<sub>k</sub> + N<sub>v</sub>。</li><li>元素数量严格等于4，各元素必须为非零有效正整数。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headNums</td>
      <td>输入</td>
      <td>按[N<sub>q</sub>, N<sub>k</sub>, N<sub>v</sub>]顺序传入，提供输入参数qkv矩阵中，qkv分量单元中分的N维度具体尺寸。</td>
      <td>[N<sub>q</sub>, N<sub>k</sub>, N<sub>v</sub>]必须为实际原始值，不可进行约化。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>输入</td>
      <td>RmsNorm计算中防止除0。对应计算公式中的epsilon。</td>
      <td>建议值为1e-6</td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cacheModeOptional</td>
      <td>输入</td>
      <td>表示cache的格式。</td>
      <td>建议值为"PA_NZ"。</td>
      <td>CHAR*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qOutBeforeQuant</td>
      <td>输出</td>
      <td>表示即将写入到qOut中的数据。</td>
      <td><ul><li>分型、数据类型，需要与输入参数qkv保持一致。</li><li>shape为[B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>q</sub> * D<sub>qkv</sub>]。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kOutBeforeQuant</td>
      <td>输出</td>
      <td>表示即将写入到vCache中的数据，在未经量化和Scatter前的中间计算结果。</td>
      <td><ul><li>分型、数据类型，需要与输入参数qkv保持一致。</li><li>shape为[B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>q</sub> * D<sub>qkv</sub>]。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vOutBeforeQuant</td>
      <td>输出</td>
      <td>表示即将写入到vCache中的数据，在未经量化和Scatter前的中间计算结果。</td>
      <td><ul><li>分型、数据类型，需要与输入参数qkv保持一致。</li><li>shape为[B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>q</sub> * D<sub>qkv</sub>]。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td rowspan="1">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="1">161001</td>
      <td>传入的qkv、qGamma、kGamma、cos、sin等为空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>输入或输出的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>输入或输出的参数维度不在支持的范围内。</td>
    </tr>
    <tr>
      <td>dim不在指定的取值范围内。</td>
    </tr>
  </tbody></table>

## aclnnQkvRmsNormRopeCache

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnQkvRmsNormRopeCacheGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnQkvRmsNormRopeCache默认确定性实现。

- 输入shape限制：
    * B<sub>qkv</sub>为输入qkv的batch_size，S<sub>qkv</sub>为输入qkv的sequence length，大小由qkvSize决定。
    * N<sub>qkv</sub>为输入qkv的head number。D<sub>qkv</sub>为输入qkv的head dim，目前仅支持128。D<sub>q</sub>、D<sub>k</sub>和D<sub>k</sub>分别为q、k、v的head dim，要求D<sub>qkv</sub> = D<sub>q</sub> = D<sub>k</sub> = D<sub>v</sub>，D<sub>qkv</sub>需要满足（D<sub>qkv</sub>*qkv数据类型占字节数）可以被32整除。
    * 根据rope规则，D<sub>k</sub>和D<sub>q</sub>为偶数。若cacheMode为PA_NZ场景下，D<sub>k</sub>、D<sub>q</sub>需32B对齐；BlockSize需32B对齐。
    * 关于上述32B对齐的情形，对齐值由cache的数据类型决定。以BlockSize为例，若cache的数据类型为int8，则需要满足BlockSize % 32 = 0；若cache的数据类型为float16，则需要满足BlockSize % 16 = 0；若kCache与vCache参数的dtype不一致，BlockSize需同时满足BlockSize % 32 = 0和BlockSize % 16 = 0。
    * BlockNum为写入cache的内存块数，大小由用户输入场景决定，要求BlockNum >= Ceil(S<sub>qkv</sub> / BlockSize) * B<sub>qkv</sub>。
    * 使用requireMemory表示存放数据所需的空间大小，需满足：requireMemory >= (B<sub>qkv</sub> * S<sub>qkv</sub> * N<sub>qkv</sub> * D<sub>qkv</sub> + 2 * D<sub>qkv</sub> + 2 * B<sub>qkv</sub> * S<sub>qkv</sub> * D<sub>qkv</sub> + B<sub>qkv</sub> * S<sub>qkv</sub> * N<sub>q</sub> * D<sub>qkv</sub> + BlockNum * BlockSize * N<sub>v</sub> * D<sub>qkv</sub> + BlockNum * BlockSize * N<sub>k</sub> * D<sub>qkv</sub>) * sizeof(FLOAT16) + B<sub>qkv</sub> * S<sub>qkv</sub> * sizeof(INT64) + (2 * N<sub>k</sub> * D<sub>qkv</sub> + 2 * N<sub>v</sub>) * sizeof(FLOAT)，当计算出requireMemory的大小超过当前AI处理器的GM空间总大小，不支持使用该接口。
- 其他限制：
    * 对于index，要求index的value值范围为[-1, BlockNum * BlockSize)。value数值不可以重复，index为-1时，代表跳过更新。
    * kScaleOptional, vScaleOptional表示对称量化的缩放因子，因此若传参，则值不能为0。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_qkv_rms_norm_rope_cache.h"

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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int8_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }
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
    // 1. 固定写法，device/stream初始化, 参考acl API
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> qkvShape = {48, 2304};
    std::vector<int64_t> qGammaShape = {128};
    std::vector<int64_t> kGammaShape = {128};
    std::vector<int64_t> cosShape = {48, 128};
    std::vector<int64_t> sinShape = {48, 128};
    std::vector<int64_t> indexShape = {48};
    std::vector<int64_t> qOutShape = {48, 2048};
    std::vector<int64_t> kCacheShape = {16, 4, 128, 32};
    std::vector<int64_t> vCacheShape = {16, 4, 128, 32};
    std::vector<int64_t> kScaleShape = {1, 128};
    std::vector<int64_t> vScaleShape = {1, 128};
    std::vector<int64_t> qkv_size_list = {16, 3, 18, 128};
    std::vector<int64_t> head_nums_list = {16, 1, 1};

    std::vector<int16_t> qkvHostData(48 * 2304, 0);
    std::vector<int16_t> qGammaHostData(128, 0);
    std::vector<int16_t> kGammaHostData(128, 0);
    std::vector<int16_t> cosHostData(48 * 128, 0);
    std::vector<int16_t> sinHostData(48 * 128, 0);
    std::vector<int64_t> indexHostData(48, 0);
    std::vector<int16_t> qOutHostData(48 * 2048, 0);
    std::vector<int16_t> kCacheHostData(16 * 4 * 128 * 32, 0);
    std::vector<int16_t> vCacheHostData(16 * 4 * 128 * 32, 0);
    std::vector<int16_t> kScaleHostData(1 * 128, 0);
    std::vector<int16_t> vScaleHostData(1 * 128, 0);

    void* qkvDeviceAddr = nullptr;
    void* qGammaDeviceAddr = nullptr;
    void* kGammaDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    void* indexDeviceAddr = nullptr;
    void* qOutDeviceAddr = nullptr;
    void* kCacheDeviceAddr = nullptr;
    void* vCacheDeviceAddr = nullptr;
    void* kScaleDeviceAddr = nullptr;
    void* vScaleDeviceAddr = nullptr;

    aclTensor* qkv = nullptr;
    aclTensor* qGamma = nullptr;
    aclTensor* kGamma = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;
    aclTensor* index = nullptr;
    aclTensor* qOut = nullptr;
    aclTensor* kCache = nullptr;
    aclTensor* vCache = nullptr;
    aclTensor* kScale = nullptr;
    aclTensor* vScale = nullptr;

    aclIntArray *qkv_size = aclCreateIntArray(qkv_size_list.data(), qkv_size_list.size());
    aclIntArray *head_nums = aclCreateIntArray(head_nums_list.data(), head_nums_list.size());

    double epsilon = 1e-6;
    char* cacheMode = "PA_NZ";
    bool isOutputQkv = false;

    ret = CreateAclTensor(qkvHostData, qkvShape, &qkvDeviceAddr, aclDataType::ACL_FLOAT16, &qkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qGammaHostData, qGammaShape, &qGammaDeviceAddr, aclDataType::ACL_FLOAT16, &qGamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kGammaHostData, kGammaShape, &kGammaDeviceAddr, aclDataType::ACL_FLOAT16, &kGamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT16, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT16, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qOutHostData, qOutShape, &qOutDeviceAddr, aclDataType::ACL_FLOAT16, &qOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kCacheHostData, kCacheShape, &kCacheDeviceAddr, aclDataType::ACL_INT8, &kCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vCacheHostData, vCacheShape, &vCacheDeviceAddr, aclDataType::ACL_INT8, &vCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kScaleHostData, kScaleShape, &kScaleDeviceAddr, aclDataType::ACL_FLOAT, &kScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vScaleHostData, vScaleShape, &vScaleDeviceAddr, aclDataType::ACL_FLOAT, &vScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnQkvRmsNormRopeCache第一段接口
    ret = aclnnQkvRmsNormRopeCacheGetWorkspaceSize(
        qkv, qGamma, kGamma, cos, sin, index, qOut, kCache, vCache, kScale, vScale, nullptr, nullptr, qkv_size,
        head_nums, epsilon, cacheMode, nullptr, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQkvRmsNormRopeCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnQkvRmsNormRopeCache第二段接口
    ret = aclnnQkvRmsNormRopeCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQkvRmsNormRopeCache failed. ERROR: %d\n", ret); return ret);

    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(kCacheShape, &kCacheDeviceAddr);
    PrintOutResult(vCacheShape, &vCacheDeviceAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(qkv);
    aclDestroyTensor(qGamma);
    aclDestroyTensor(kGamma);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);
    aclDestroyTensor(index);
    aclDestroyTensor(qOut);
    aclDestroyTensor(kCache);
    aclDestroyTensor(vCache);
    aclDestroyTensor(kScale);
    aclDestroyTensor(vScale);

    // 7. 释放device 资源
    aclrtFree(qkvDeviceAddr);
    aclrtFree(qGammaDeviceAddr);
    aclrtFree(kGammaDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    aclrtFree(indexDeviceAddr);
    aclrtFree(qOutDeviceAddr);
    aclrtFree(kCacheDeviceAddr);
    aclrtFree(vCacheDeviceAddr);
    aclrtFree(kScaleDeviceAddr);
    aclrtFree(vScaleDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```