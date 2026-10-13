# MoeGatingTopKSoftmax

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：MoE计算中，对x的输出做Softmax计算，取topk操作。其中yOut为softmax的topk结果；expertIdxOut为topk的indices结果即对应的专家序号；rowIdxOut为与expertIdxOut相同shape的列取值结果。如果对应的行finished为True，则expert序号直接填num\_expert值（即x的最后一个轴大小）。
- 计算公式：

  $$
  softmaxOut=softmax(x,axis=-1)
  $$

  $$
  yOut,expertIdxOut=topK(softmaxOut,k=k)
  $$

  $$
  rowIdxRange=arange(expertIdxOut.shape[0]*expertIdxOut.shape[1])
  $$

  $$
  rowIdxOut=rowIdxRange.reshape([expertIdxOut.shape[1],expertIdxOut.shape[0]]).transpose(1,0)
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
        <td>公式中的`x`。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>finishedOptional</td>
        <td>输入</td>
        <td>表示如果对应的行finished为True，则expert序号直接填num_expert值（即x的最后一个轴大小）。</td>
        <td>BOOL</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>k</td>
        <td>属性</td>
        <td>公式中的`k`，表示topk的k值。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yOut</td>
        <td>输出</td>
        <td>公式中的`yOut`，表示softmax的topk结果。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertIdxOut</td>
        <td>输出</td>
        <td>公式中的`expertIdxOut`，表示topk的indices结果即对应的专家序号。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>rowIdxOut</td>
        <td>输出</td>
        <td>公式中的`rowIdxOut`，指示每个位置对应的原始行位置。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- k的值不大于1024。
- x和finishedOptional的每一维大小应不大于int32的最大值2147483647。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_gating_top_k_softmax.cpp](examples/test_aclnn_moe_gating_top_k_softmax.cpp) | 通过[aclnnMoeGatingTopKSoftmax](aclnnMoeGatingTopKSoftmax.md)接口方式调用MoeGatingTopKSoftmax算子。 |
