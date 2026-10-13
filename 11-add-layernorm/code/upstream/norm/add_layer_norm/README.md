# AddLayerNorm

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | √ |
|  <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：实现AddLayerNorm功能。
- 计算公式：

  $$
  x = x1 + x2 + bias
  $$

  $$
  rstd = {{1}\over\sqrt {Var(x)+epsilon}}
  $$

  $$
  y = (x-\bar{x}) * rstd * \gamma + \beta
  $$

  其中：

  - ${\bar{x}}$：

    $$
    \operatorname{\bar{x}}=\frac{1}{D}\sum_{1}^{D}{x_i}
    $$

    D为x中参加均值计算的数量。
    
  - $Var(x)$：
    $$
    Var(x) = E(x-{\bar{x}})^2
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
      <td>x1</td>
      <td>输入</td>
      <td>表示AddLayerNorm中加法计算的输入，对应公式中的`x1`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示AddLayerNorm中加法计算的输入，对应公式中的`x2`。shape需要与`x1`一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示层归一化中的`gamma`参数，对应公式中的`γ`。shape与`x1`的norm的维度值相同。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>表示层归一化中的`beta`参数，对应公式中的`β`。shape与`x1`的norm的维度值相同。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>可选输入</td>
      <td>表示AddLayerNorm中加法计算的输入，对应公式中的`bias`。shape可以和`gamma`/`beta`或`x1`/`x2`一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>添加到分母中的值，以确保数值稳定，对应公式中的`epsilon`。</li><li>默认值为1e-5f。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additional_output</td>
      <td>可选属性</td>
      <td><ul><li>表示是否开启x=x1+x2+bias的输出。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示LayerNorm的结果输出`y`，对应公式中的`y`。shape需要与输入`x1`一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输出</td>
      <td>输出LayerNorm算过程中（x1 + x2 + bias）的结果的均值，对应公式中的`x`的平均值。shape需要与`x1`满足broadcast关系（前几维的维度和`x1`前几维的维度相同，后面的维度为1，总维度与`x1`维度相同，前几维指`x1`的维度减去`gamma`的维度，表示不需要norm的维度）。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输出</td>
      <td>输出LayerNorm算过程中`rstd`的结果，对应公式中的`rstd`。shape需要与`x1`满足broadcast关系（前几维的维度和`x1`前几维的维度相同，后面的维度为1，总维度与`x1`维度相同，前几维指`x1`的维度减去gamma的维度，表示不需要norm的维度）。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输出</td>
      <td>表示LayerNorm的结果输出`x`，对应公式中的`x`。shape需要与输入`x1`一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term> ：
  - 所有的输入参数和输出参数`y`、`x`的数据类型不支持BFLOAT16。
  - 在当前产品下的使用场景下，输出参数`mean`、`rstd`为无效参数，输出的值不生效。
  - x1、x2、beta、gamma、bias五个输入的尾轴长度必须大于等于32Bytes。
- Kirin X90/Kirin 9030处理器系列产品：不支持BFLOAT16。

## 约束说明

- 是否支持空tensor：不支持空进空出。
- 是否支持非连续tensor：输入输出不支持非连续。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_add_layer_norm](examples/test_aclnn_add_layer_norm.cpp) | 通过[aclnnAddLayerNorm](docs/aclnnAddLayerNorm.md)接口方式调用AddLayerNorm算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/add_layer_norm_proto.h)构图方式调用AddLayerNorm算子。         |
