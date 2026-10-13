# GroupNormSiluV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Kirin X90 处理器系列产品</term> | √ |
| <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：计算输入self的组归一化结果out，均值meanOut，标准差的倒数rstdOut，以及silu的输出。

- 计算公式：
  - **GroupNorm:**
  记 $E[x] = \bar{x}$代表$x$的均值，$Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$代表$x$的方差，则
  $$
  \left\{
  \begin{array} {rcl}
  out& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
  meanOut& &= E[x]\\
  rstdOut& &= \frac{1}{\sqrt{Var[x] + eps}}\\
  \end{array}
  \right.
  $$
  - **Silu:**
  $$
  out = \frac{x}{1+e^{-x}}
  $$
    当activateSilu为True时，会计算Silu， 此时Silu计算公式的x为GroupNorm公式得到的out。
      
## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>out计算公式中的x。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>out计算公式中的gamma。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>out计算公式中的beta。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group</td>
      <td>属性</td>
      <td>self的第1维度分为group组。</td>
      <td>INT32、INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td>out和rstdOut计算公式中的eps值。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activateSilu</td>
      <td>属性</td>
      <td>是否支持silu计算。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>out输出张量。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>meanOut</td>
      <td>输出</td>
      <td>mean输出张量。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstdOut</td>
      <td>输出</td>
      <td>rstd输出张量。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030处理器系列产品：所有输入、输出的数据类型不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_group_norm_silu.cpp](examples/test_aclnn_group_norm_silu.cpp) | 通过[aclnnGroupNormSilu.md](aclnnGroupNormSilu.md)接口方式调用GroupNormSilu算子。 |
| aclnn接口  | [test_aclnn_group_norm_silu_v2.cpp](examples/test_aclnn_group_norm_silu_v2.cpp) | 通过[aclnnGroupNormSiluV2.md](aclnnGroupNormSiluV2.md)接口方式调用GroupNormSilu算子。 |
| 图模式 | -  | 通过[算子IR](../code/op_graph/group_norm_silu_proto.h)构图方式调用GroupNormSilu算子。
