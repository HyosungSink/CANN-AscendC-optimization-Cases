# MoeInitRoutingQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- **算子功能**：MoE的routing计算，根据[aclnnMoeGatingTopKSoftmax](../moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax.md)的计算结果做routing处理，并对结果进行量化。
- **计算公式**：

  $$
  expandedExpertIdx,sortedRowIdx=keyValueSort(expertIdx,rowIdx)
  $$

  $$
  expandedRowIdx[sortedRowIdx[i]]=i
  $$

  $$
  expandedX[i]=quant[x[sortedRowIdx[i]\%numRows]]
  $$

## 参数说明

  <table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出/属性</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>MOE的输入即token特征输入，对应公式中的`x`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rowIdx</td>
      <td>输入</td>
      <td>指示每个位置对应的原始行位置，对应公式中的`rowIdx`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expertIdx</td>
      <td>输入</td>
      <td>aclnnMoeInitRoutingQuantSoftmax的输出每一行特征对应的K个处理专家，对应公式中的`expertIdx`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activeNum</td>
      <td>属性</td>
      <td>表示总的最大处理row数且大于等于0，expandedXOut只有这么多行是有效的。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>属性</td>
      <td>量化计算需要。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>属性</td>
      <td>量化计算需要。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedXOut</td>
      <td>输出</td>
      <td>根据expertIdx进行扩展过的特征，对应公式中的`expandedX`。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expandedRowIdxOut</td>
      <td>输出</td>
      <td>expandedX和x的映射关系，对应公式中的`expandedRowIdx`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expandedExpertIdxOut</td>
      <td>输出</td>
      <td>输出expertIdx排序后的结果，对应公式中的`expandedExpertIdx`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

  无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_init_routing_quant](examples/test_aclnn_moe_init_routing_quant.cpp) | 通过[aclnnMoeInitRoutingQuant](aclnnMoeInitRoutingQuant.md)接口方式调用MoeInitRoutingQuant算子。 |
