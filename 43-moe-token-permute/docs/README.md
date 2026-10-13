# MoeTokenPermute

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- **算子功能**：MoE的permute计算，根据索引indices将tokens广播并排序。
- **计算公式**：
  - paddedMode为`false`时
  
    $$
    sortedIndicesFirst=argSort(Indices)
    $$
  
    $$
    sortedIndicesOut=argSort(sortedIndicesFirst)
    $$
  
    $$
    permuteTokens[sortedIndicesFirst[i]]=tokens[i//topK]
    $$
  
  - paddedMode为`true`时
  
    $$
    permuteTokensOut[i]=tokens[Indices[i]]
    $$
  
    $$
    sortedIndicesOut=Indices
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
      <td>tokens</td>
      <td>输入</td>
      <td>输入token，对应公式中的`tokens`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>输入indices，对应公式中的`Indices`。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numOutTokens</td>
      <td>属性</td>
      <td>有效输出token数，设置为0时，表示不会删除任何token。不为0时，会按照numOutTokens进行切片丢弃按照indices排序好的token中超过numOutTokens的部分，为负数时按照切片索引为负数时处理。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddedMode</td>
      <td>属性</td>
      <td>paddedMode为true时表示indices已被填充为代表每个专家选中的token索引，此时不对indices进行排序。目前仅支持paddedMode为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>permuteTokensOut</td>
      <td>输出</td>
      <td>根据indices进行扩展并排序过的tokens，对应公式中的`permuteTokensOut`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sortedIndicesOut</td>
      <td>输出</td>
      <td>permuteTokensOut和tokens的映射关系，对应公式中的`sortedIndicesOut`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    
  </tbody></table>

## 约束说明

- indices 要求元素个数小于`16777215`，值大于等于`0`小于`16777215`(单点支持int32或int64的最大或最小值，其余值不在范围内排序结果不正确)。
- 不支持paddedMode为`True`。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：topK小于等于512。
- <term>Ascend 950PR/Ascend 950DT</term>：
  在调用本接口时，框架内部会转调用[aclnnMoeInitRoutingV2](../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md)接口，如果出现参数错误提示，请参考以下参数映射关系：
  - token输入等同于aclnnMoeInitRoutingV2接口的x输入。
  - indices输入等同于aclnnMoeInitRoutingV2接口的expertIdx输入。
  - numOutTokens输入等同于aclnnMoeInitRoutingV2接口的activeNum输入。
  - paddedMode输入等同于aclnnMoeInitRoutingV2接口的dropPadMode输入。
  - permuteTokensOut输出等同于aclnnMoeInitRoutingV2接口的expandedXOut输出。
  - sortedIndicesOut输出等同于aclnnMoeInitRoutingV2接口的expandedRowIdxOut输出。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_token_permute](examples/test_aclnn_moe_token_permute.cpp) | 通过[aclnnMoeTokenPermute](aclnnMoeTokenPermute.md)接口方式调用MoeTokenPermute算子。 |
