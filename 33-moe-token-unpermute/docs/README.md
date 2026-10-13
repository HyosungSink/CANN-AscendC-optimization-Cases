# MoeTokenUnpermute

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                      |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |
| <term>Kirin X90 处理器系列产品</term> | √ |
| <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：根据sortedIndices存储的下标，获取permutedTokens中存储的输入数据；如果存在probs数据，permutedTokens会与probs相乘；最后进行累加求和，并输出计算结果。
- 计算公式：

  - probs非None计算公式如下：
    
    $$
    T[k] = T[S[k]]
    $$
    
    $$
    T[k] = T[k] * P[i][j]
    $$

    $$
    O[i] = \sum_{k=i*topK}^{(i+1)*topK - 1 } T[k]
    $$
    
    其中$i \in {0,1,...,tokens-1}$；$j \in {0,1,...,topK-1}$；$k \in {0,1,...,tokens*topK-1}$；T表示permutedTokens；S表示sortedIndices；P表示probs；O表示out；topK表示topK\_num；tokens表示tokens_num。

  - probs为None时，此时topK\_num=1，计算公式如下：

    $$
    T[i] = T[S[i]]
    $$

    $$
    O[i] = T[i]
    $$

    其中 $i \in {0,1,...,tokens-1}$；T表示permutedTokens；S表示sortedIndices；O表示out；tokens表示tokens_num。

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
      <td>permutedTokens</td>
      <td>输入</td>
      <td>待计算输入，对应公式中的`T`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sortedIndices</td>
      <td>输入</td>
      <td>表示需要计算的数据在permutedTokens中的位置，对应公式中的`S`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>probsOptional</td>
      <td>可选输入</td>
      <td>对应公式中的`P`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>paddedMode</td>
      <td>属性</td>
      <td>目前仅支持false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>restoreShapeOptional</td>
      <td>属性</td>
      <td>目前仅支持nullptr。</td>
      <td>aclIntArray*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>对应公式中的`O`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：topK_num <= 512。
- <term>Ascend 950PR/Ascend 950DT</term>：
  在调用本接口时，框架内部会转调用[aclnnMoeFinalizeRoutingV2](../moe_finalize_routing_v2/docs/aclnnMoeFinalizeRoutingV2.md)接口，如果出现参数错误提示，请参考以下参数映射关系：
  - permutedTokens输入等同于aclnnMoeFinalizeRoutingV2接口的expandedX输入。
  - sortedIndices输入等同于aclnnMoeFinalizeRoutingV2接口的expandedRowIdx输入。
  - probsOptional输入等同于aclnnMoeFinalizeRoutingV2接口的scalesOptional输入。
  - paddedMode输入等同于aclnnMoeFinalizeRoutingV2接口的dropPadMode输入。
  - out输出等同于aclnnMoeFinalizeRoutingV2接口的out输出。
- <term>Atlas 推理系列产品</term>：
  - permutedTokens与probsOptional支持的数据类型为FLOAT16、FLOAT32。 
  - topK_num <= 512。
  - hiddensize是128的倍数且小于10240。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_token_unpermute](examples/test_aclnn_moe_token_unpermute.cpp) | 通过[aclnnMoeTokenUnpermute](aclnnMoeTokenUnpermute.md)接口方式调用MoeTokenUnpermute算子。 |
