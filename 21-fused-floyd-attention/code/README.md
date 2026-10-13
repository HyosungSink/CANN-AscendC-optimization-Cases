# FusedFloydAttention

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品</term>|      √     |
|<term>Atlas A2 推理系列产品</term>|      ×     |

## 功能说明

- 算子功能：训练推理场景下，使用FloydAttention算法实现多维自注意力的计算。

- 计算公式：

  注意力的正向计算公式如下：

  $$
  weights = Softmax(attenMask + scale*(einsum(query, key1^T) + einsum(query, key2^T)))
  $$
  
  $$
  attention\_out = einsum(weights, value1) + einsum(weights, value2)
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>query</td>
      <td>输入</td>
      <td>公式中的输入query。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key1</td>
      <td>输入</td>
      <td>公式中的输入key1。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value1</td>
      <td>输入</td>
      <td>公式中的输入value1。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>可选输入</td>
      <td>公式中的atten_mask，表示注意力掩码，取值为1代表该位不参与计算（不生效），为0代表该位参与计算。</td>
      <td>BOOL、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>公式中的scale，表示缩放系数，作为计算流中Muls的scalar值。</li>
          <li>默认值为1.0。</li>
        </ul>
      </td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxMaxOut</td>
      <td>输出</td>
      <td>Softmax计算的Max中间结果，用于反向计算。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmaxSumOut</td>
      <td>输出</td>
      <td>Softmax计算的Sum中间结果，用于反向计算。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attentionOut</td>
      <td>输出</td>
      <td>公式中的attention_out。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配
- 关于数据shape的约束，其中：
  - B：取值范围为1\~2K。
  - H：取值范围为1\~256。
  - N：取值范围为16\~1M且N%16==0。
  - M：取值范围为128\~1M且M%128==0。
  - K：取值范围为128\~1M且K%128==0。
  - D：取值范围为32/64/128。

- query与key1的第0/2/4轴需相同。
- key1与value1 shape需相同。
- key2与value2 shape需相同。
- softmaxMax与softmaxSum shape需相同。
- D只支持32/64/128。

## 调用说明

| 调用方式           | 调用样例                                                                                    | 说明                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_fused_floyd_attention](./examples/test_aclnn_fused_floyd_attention.cpp) | 通过[aclnnFusedFloydAttention](./docs/aclnnFusedFloydAttention.md)接口方式调用FusedFloydAttention算子。 |
