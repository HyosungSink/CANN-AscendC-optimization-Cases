# RmsNormGrad

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

- 算子功能：[RmsNorm](../rms_norm/README.md)的反向计算。用于计算RmsNorm的梯度，即在反向传播过程中计算输入张量的梯度。
- 算子公式：

  - 正向公式：

  $$
  \operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  - 反向推导：

  $$
  dx_i= (dy_i * g_i - \frac{x_i}{\operatorname{Rms}(\mathbf{x})} * \operatorname{Mean}(\mathbf{y})) * \frac{1} {\operatorname{Rms}(\mathbf{x})},  \quad \text { where } \operatorname{Mean}(\mathbf{y}) = \frac{1}{n}\sum_{i=1}^n (dy_i * g_i * \frac{x_i}{\operatorname{Rms}(\mathbf{x})})
  $$

  $$
  dg_i = \frac{x_i}{\operatorname{Rms}(\mathbf{x})} dy_i
  $$

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
      <td>dy</td>
      <td>输入</td>
      <td>表示反向传回的梯度，对应公式中的`dy`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>正向算子的输入，表示被标准化的数据，对应公式中的`x`。shape与入参`dy`的shape一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输入</td>
      <td>正向算子的中间计算结果，对应公式中的`Rms(x)`的倒数。shape需要满足rstd_shape = x_shape\[0:n\]，n < x_shape.dims()，n与gamma一致。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示标准化过程中的缩放张量，对应公式中的`g`。shape需要满足gamma_shape = x_shape\[n:\], n < x_shape.dims()。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>表示输入`x`的梯度，对应公式中的`dx`。shape需要与输入`dy`一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgamma</td>
      <td>输出</td>
      <td>表示`gamma`的梯度，对应公式中的`dg`。shape需要与输入`gamma`一致。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas 推理系列产品</term>：输入参数`dy`、`x`、`gamma`和输出`dx`的数据类型不支持BFLOAT16。

## 约束说明

<term>Atlas 推理系列产品</term>：`x`、`dy`、`gamma`输入的尾轴长度必须大于等于32Bytes。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_rms_norm_grad](examples/test_aclnn_rms_norm_grad.cpp) | 通过[aclnnRmsNormGrad](aclnnRmsNormGrad.md)接口方式调用RmsNormGrad算子。 |
| 图模式 | -  | 通过[算子IR](../code/op_graph/rms_norm_grad_proto.h)构图方式调用RmsNormGrad算子。         |
