# SwiGlu

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

- 接口功能：Swish门控线性单元激活函数，实现x的SwiGlu计算。  

- 计算公式：  

  <p style="text-align: center">
  out<sub>i</sub> = SwiGlu(x<sub>i</sub>)=Swish(A<sub>i</sub>)*B<sub>i</sub>
  </p>
  其中，A<sub>i</sub>表示x<sub>i</sub>按指定dim维度一分为二的前半部分张量，B<sub>i</sub>表示x<sub>i</sub>按指定dim维度一分为二的后半部分张量。

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
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td><ul><li>需要进行切分的维度序号，对x相应轴进行对半切分。</li><li>取值范围为[-x.dim(), x.dim()-1]。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的out<sub>i</sub>。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas 推理系列产品</term>：数据类型支持FLOAT16、FLOAT32。

- Kirin X90/Kirin 9030处理器系列产品：不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_swi_glu](./examples/test_aclnn_swi_glu.cpp) | 通过[aclnnSwiGlu](./docs/aclnnSwiGlu.md)接口方式调用SwiGlu算子。    |
| 图模式调用 | -   | 通过[算子IR](./op_graph/swi_glu_proto.h)构图方式调用SwiGlu算子。 |
