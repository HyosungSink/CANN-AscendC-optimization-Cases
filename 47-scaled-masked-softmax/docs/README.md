# ScaledMaskedSoftmaxV2

## 支持的产品型号

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Kirin X90 处理器系列产品</term> | √ |
|  <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：将输入的数据x先进行scale缩放和mask，然后执行softmax的输出。
- 计算公式：

  $$
  y = Softmax((scale * x) * mask, dim = -1)
  $$

  $$
  Softmax(X_i) ={e^{X_i - max(X, dim=-1)} \over \sum{e^{X_i - max(X, dim=-1)}}}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 181px">
  <col style="width: 144px">
  <col style="width: 273px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>x</td>
      <td>输入</td>
      <td>公式中的输入x。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输入</td>
      <td>公式中的输入mask。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>属性</td>
      <td>默认值为1.0。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fixTriuMask</td>
      <td>属性</td>
      <td>仅支持false</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的y。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 处理器系列产品: 不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_scaled_masked_softmax](./examples/test_aclnn_scaled_masked_softmax.cpp) | 通过[aclnnScaledMaskedSoftmax](aclnnScaledMaskedSoftmax.md)接口方式调用aclnnScaledMaskedSoftmax算子。    |
