# FlashAttentionScoreGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品</term>|      √     |
|<term>Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品</term>|      √     |
|<term>Atlas A2 推理系列产品</term>|      ×     |

## 功能说明

- 算子功能：训练场景下计算注意力的反向输出，即FlashAttentionScore的反向计算：

  - pseType=1时，需要先add再mul。
  - pseType≠1时，需要先mul再add。

- 计算公式：

  已知注意力的正向计算公式为：
  - pseType=1时，公式如下：

    $$
    Y=Dropout(Softmax(Mask(\frac{QK^T+pse}{\sqrt{d}}),atten\_mask),keep\_prob)V
    $$
  - pseType≠1时，公式如下：

    $$
    Y=Dropout(Softmax(Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask),keep\_prob)V
    $$

  为方便表达，以变量$S$和$P$表示计算公式：

  $$
  S=Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask
  $$

  $$
  P=Dropout(Softmax(S),keep\_prob)
  $$

  $$
  Y=PV
  $$

  则注意力的反向计算公式为：

  $$
  dV=P^TdY
  $$

  $$
  dQ=\frac{((dS)*K)}{\sqrt{d}}
  $$

  $$
  dK=\frac{((dS)^T*Q)}{\sqrt{d}}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px">
  <colgroup>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入</td>
      <td>公式中的输入Q。</td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的输入K。</td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>公式中的输入V。</td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>公式中的输入dY。</td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pseShiftOptional</td>
      <td>可选输入</td>
      <td>公式中的pse，表示位置编码。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dropMaskOptional</td>
      <td>可选输入</td>
      <td>公式中的Dropout，表示数据丢弃掩码。取值为1代表保留该数据，为0代表丢弃该数据。</td>
      <td>UINT8</td>
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
      <td>keepProb</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>公式中的keep_prob，表示数据需要保留的概率。</li>
          <li>默认值为1.0。</li>
        </ul>
      </td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dqOut</td>
      <td>输出</td>
      <td>公式中的dQ，表示query的梯度。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dkOut</td>
      <td>输出</td>
      <td>公式中的dK，表示key的梯度。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dvOut</td>
      <td>输出</td>
      <td>公式中的dV，表示value的梯度。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>
<ul>
  <li><term>Atlas A2 训练产品</term>:不支持FLOAT8_E5M2、FLOAT8_E4M3FN。</li>
  <li><term>Atlas A3 训练产品</term>:不支持FLOAT8_E5M2、FLOAT8_E4M3FN。</li>
</ul>

## 约束说明

- 输入query、key、value、pseShiftOptional的数据类型必须一致。
- 输入query、key、value、dy的input_layout必须一致。
- 关于数据shape的约束，以inputLayout的BSND、BNSD为例（BSH、SBH下H=N\*D），其中：
  - B：取值范围为1\~2M。当prefixOptional的时候B最大支持2K。
  - N：取值范围为1\~256。
  - S：取值范围为1\~1M。
  - D：
      - Atlas A2 训练系列产品：取值范围为1\~512。
      - Atlas A3 训练系列产品：取值范围为1\~512。
      - Ascend 950PR/Ascend 950DT:取值范围为1\~768。
- keepProb的取值范围为(0, 1]。
- 部分场景下，如果计算量过大可能会导致算子执行超时（aicore error类型报错，errorStr为：timeout or trap error），此时建议做轴切分处理，注：这里的计算量会受B、S、N、D等参数的影响，值越大计算量越大。
- pseType为2或3的时候，当前只支持Sq和Skv等长。

## 调用说明

| 调用方式           | 调用样例                                                                                                              | 说明                                                                                                                    |
|----------------|-------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_flash_attention_score_grad_v2](./examples/test_aclnn_flash_attention_score_grad_v2.cpp)                     | 非TND场景，通过[aclnnFlashAttentionScoreGradV2](./docs/aclnnFlashAttentionScoreGradV2.md)接口方式调用FlashAttentionGrad算子。                   |

## 参考资源

- [FAG算子设计介绍](./docs/FAG算子设计介绍.md)
