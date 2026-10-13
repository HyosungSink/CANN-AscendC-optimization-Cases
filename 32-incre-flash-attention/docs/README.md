# IncreFlashAttention

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      √     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：对于自回归（Auto-regressive）的语言模型，随着新词的生成，推理输入长度不断增大。在原来全量推理的基础上**实现增量推理**，query的S轴固定为1，key和value是经过kvCache后，将之前推理过的state信息，叠加在一起，每个Batch对应S轴的实际长度可能不一样，输入的数据是经过padding后的固定长度数据。支持**量化，位置编码，page attention，kvCache反量化和KV左Padding特性。**

  相比全量场景的FlashAttention算子（[PromptFlashAttention](../prompt_flash_attention/README.md)），增量推理的流程与正常全量推理并不完全等价，不过增量推理的精度并无明显劣化。

  **说明：**

    kvCache是大模型推理性能优化的一个常用技术。采样时，Transformer模型会以给定的prompt/context作为初始输入进行推理（可以并行处理），随后逐一生成额外的token来继续完善生成的序列（体现了模型的自回归性质）。在采样过程中，Transformer会执行自注意力操作，为此需要给当前序列中的每个项目（无论是prompt/context还是生成的token）提取键值（KV）向量。这些向量存储在一个矩阵中，通常被称为kv缓存（kvCache）。

- 计算公式：

  self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。

    self-attention的计算公式一般定义如下，其中$Q$、$K$、$V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。

  $$
     Attention(Q,K,V)=Score(Q,K)V
  $$

    本算子中Score函数采用Softmax函数，self-attention计算公式为：

  $$
   Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

    其中$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n*d$的矩阵。

## 参数说明

<div style="overflow-x: auto;">
<table style="undefined;table-layout: fixed; width: 930px"><colgroup>
<col style="width: 150px">
<col style="width: 150px">
<col style="width: 250px">
<col style="width: 230px">
<col style="width: 150px">
</colgroup>
<thead>
  <tr>
    <th style="text-align: center;">参数名</th>
    <th style="text-align: center;">输入/输出/属性</th>
      <th style="text-align: center;">描述</th>
    <th style="text-align: center;">数据类型</th>
      <th style="text-align: center;">数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>query</td>
    <td >输入</td>
    <td >公式中的输入Q。</td>
    <td >FLOAT、FLOAT16</td>
    <td >ND</td>
  </tr>
  <tr>
    <td>key</td>
    <td >输入</td>
    <td >公式中的输入K。</td>
    <td >FLOAT、INT8、FLOAT16</td>
    <td >ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td >输入</td>
    <td >公式中的输入V。</td>
    <td >FLOAT、INT8、FLOAT16</td>
    <td >ND</td>
  </tr>
  <tr>
    <td>scaleValue</td>
    <td >属性</td>
    <td >公式中的d开根号的倒数。</td>
    <td >DOUBLE</td>
    <td >-</td>
  </tr>
  <tr>
    <td>attentionOut</td>
    <td >输出</td>
    <td >公式中的输出。</td>
    <td >FLOAT、INT8、FLOAT16</td>
    <td >ND</td>
  </tr>
</tbody>
</table></div>

## 约束说明

- 参数query和attentionOut的shape需要完全一致，参数key、value 中对应tensor的shape需要完全一致。
- 非连续场景下，参数key、value的tensorlist中tensor的个数等于query的B(由于tensorlist限制, 非连续场景下B需要小于等于256)。shape除S外需要完全一致，且batch只能为1。
- 参数query中的N和numHeads值相等，key、value的N和numKeyValueHeads值相等，并且numHeads是numKeyValueHeads的倍数关系。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：
  - 支持B轴小于等于65536，N轴小于等于256，D轴小于等于512。
- Atlas 推理系列加速卡产品：
  - 支持B轴小于等于256，N轴小于等于256，D轴小于等于512，key、value的S轴小于等于65536。
  - query、key、value和ttentionOut数据类型仅支持FLOAT16。
  - 在数据排布格式为BNSD时，需要满足numHeads与numKeyValueHeads的比值不大于8，其他情况仅支持取值0；
- INT8量化相关入参数量与输入、输出数据格式的综合限制：
  - query、key、value输入为FLOAT16，输出为INT8的场景：入参quantScale2必填，quantOffset2可选，不能传入dequantScale1、quantScale1、dequantScale2（即为nullptr）参数。

- pseShift数据类型需与query数据类型保持一致。

- antiquantScale和antiquantOffset参数约束：
    - per-channel模式：两个参数的shape可支持(2, N, 1, D)，(2, N, D)，(2, H)，N为numKeyValueHeads。参数数据类型和query数据类型相同。
    - per-tensor模式：两个参数的shape均为(2)，数据类型和query数据类型相同。

- 入参 quantScale2 和 quantOffset2 支持 per-tensor/per-channel 两种格式和 FLOAT32/BFLOAT16 两种数据类型。若传入 quantOffset2 ，需保证其类型和shape信息与 quantScale2 一致。当输入为BFLOAT16时，同时支持FLOAT32和BFLOAT16，否则仅支持FLOAT32 。per-channel 格式，当输出layout为BSH时，要求 quantScale2 所有维度的乘积等于H；其他layout要求乘积等于N*D。（建议输出layout为BSH时，quantScale2 shape传入[1,1,H]或[H]；输出为BNSD时，建议传入[1,N,1,D]或[N,D]；输出为BSND时，建议传入[1,1,N,D]或[N,D]）。

- page attention场景:

  - page attention的使能必要条件是blockTable存在且有效，同时key、value是按照blockTable中的索引在一片连续内存中排布，支持key、value dtype为FLOAT16/BFLOAT16/INT8，在该场景下key、value的inputLayout参数无效。
  - blockSize是用户自定义的参数，该参数的取值会影响page attention的性能，在使能page attention场景下，blockSize需要传入非0值, 且blocksize最大不超过512。key、value输入类型为FLOAT16/BFLOAT16时需要16对齐，key、value 输入类型为INT8时需要32对齐，推荐使用128。通常情况下，page attention可以提高吞吐量，但会带来性能上的下降。
  - page attention场景下，当query的inputLayout为BNSD时，kvCache排布支持（blocknum, blocksize, H）和（blocknum, KV_N, blocksize, D）两种格式，当query的inputLayout为BSH、BSND时，kvCache排布只支持（blocknum, blocksize, H）一种格式。blocknum不能小于根据actualSeqLengthsKv和blockSize计算的每个batch的block数量之和。且key和value的shape需保证一致。
  - page attention场景下，kvCache排布为（blocknum, KV_N, blocksize, D）时性能通常优于kvCache排布为（blocknum, blocksize, H）时的性能，建议优先选择（blocknum, KV_N, blocksize, D）格式。
  - page attention使能场景下，当输入kvCache排布格式为（blocknum, blocksize, H），且 numKvHeads * headDim 超过64k时，受硬件指令约束，会被拦截报错。可通过使能GQA（减小 numKvHeads）或调整kvCache排布格式为（blocknum, numKvHeads, blocksize, D）解决。
  - page attention场景下，必须传入输入actualSeqLengths。
  - page attention场景下，blockTable必须为二维，第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大actualSeqLengthsKv对应的block数量）。
  - page attention使能场景下，以下场景输入S需要大于等于maxBlockNumPerSeq * blockSize。
  - 使能 Attention Mask，例如 mask shape为 (B, 1, 1, S)。
  - 使能 pseShift，例如 pseShift shape为(B, N, 1, S)。
- kv左padding场景:
    - kvCache的搬运起点计算公式为：Smax - kvPaddingSize - actualSeqLengths；kvCache的搬运终点计算公式为：Smax - kvPaddingSize。其中kvCache的搬运起点或终点小于0时，返回数据结果为全0。
    - kvPaddingSize小于0时将被置为0。
    - 需要与actualSeqLengths参数一起使能，否则默认为kv右padding场景。
    - 与Attention Mask参数一起使能时，需要保证Attention Mask含义正确，即能够正确的对无效数据进行隐藏。否则将引入精度问题。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_IncreFlashAttentionV4](./examples/test_aclnn_incre_flash_attention.cpp) | 通过[aclnnIncreFlashAttentionV4](aclnnIncreFlashAttentionV4.md)调用IncreFlashAttentionV4算子。 |
