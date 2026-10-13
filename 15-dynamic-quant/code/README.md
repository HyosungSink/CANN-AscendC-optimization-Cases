# DynamicQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Kirin X90 处理器系列产品</term> | √ |
| <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：为输入张量进行per-token对称动态量化。

- 计算公式：
  - 若不输入smoothScalesOptional，则

  $$
   scaleOut=row\_max(abs(x))/dtypeMax
  $$

  $$
   yOut=round(x/scaleOut)
  $$
  
  - 若输入smoothScalesOptional，则
  
  $$
  input = x\cdot smoothScalesOptional
  $$

  $$
   scaleOut=row\_max(abs(input))/dtypeMax
  $$

  $$
   yOut=round(input/scaleOut)
  $$
  
  其中row\_max代表每行求最大值，dtypeMax为输出数据类型的最大值。

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
      <td>x</td>
      <td>输入</td>
      <td><ul><li>算子输入的Tensor，对应公式中的`x`。</li><li>shape维度要大于1。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scales</td>
      <td>可选输入</td>
      <td><ul><li>算子输入的Tensor，对应公式中的`smoothScalesOptional`。</li><li>shape维度与`x`的最后一维相同。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>group_index</td>
      <td>可选输入</td>
      <td><ul><li>指定组的索引。</li><li>shape支持1D，为[E, ]，且shape跟输出`scale`的shape的第一维相同。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>可选属性</td>
      <td><ul><li>指定输出y的数据类型。支持DT_INT4(29)、DT_INT8(2)、DT_FLOAT8_E5M2(35)、DT_FLOAT8_E4M3FN(36)、DT_HIFLOAT8(34)。</li><li>默认值为DT_INT8(2)。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td><ul><li>量化后的输出张量，对应公式中的`yOut`。</li><li>数据类型与参数`dst_type`配置的数据类型一致，shape与输入`x`的shape保持一致。</li></ul></td>
      <td>INT8、INT4、FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输出</td>
      <td>量化后的缩放系数，对应公式中的`scaleOut`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term> ：
  - 输入`x`、`smooth_scales`：数据类型仅支持FLOAT16。
  - 输出`y`：数据类型仅支持INT8。
  - 输入`smooth_scales`为预留参数，当前版本不参与计算。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：输出`y`的数据类型仅支持INT8、INT4。

- Kirin X90/Kirin 9030 处理器系列产品：
  - 输入`x`：数据类型仅支持FLOAT16。
  - 可选输入`smooth_scales`：数据类型仅支持FLOAT16。
  - 输出`y`：数据类型仅支持INT8。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_dynamic_quant](examples/test_aclnn_dynamic_quant.cpp) | 通过[aclnnDynamicQuant](docs/aclnnDynamicQuant.md)接口方式调用DynamicQuant算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/dynamic_quant_proto.h)构图方式调用DynamicQuant算子。         |
