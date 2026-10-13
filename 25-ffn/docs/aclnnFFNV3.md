# aclnnFFNV3

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/ffn/ffn)

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      √     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 接口功能：该FFN算子提供MoeFFN和FFN的计算功能。在没有专家分组（expertTokens为空）时是FFN，有专家分组时是MoeFFN，统称为FFN，属于Moe结构。MoE（Mixture-of-Experts，混合专家系统）是一种用于训练万亿参数量级模型的技术。MoE将预测建模任务分解为若干子任务，在每个子任务上训练一个专家模型（Expert Model），开发一个门控模型（Gating Model），该模型会根据输入数据分配一个或多个专家，最终综合多个专家计算结果作为预测结果。Mixture-of-Experts结构的模型是将输入数据分配给最相关的一个或者多个专家，综合涉及的所有专家的计算结果来确定最终结果。

  相较于[FFNV2](aclnnFFNV2.md)接口，**此接口中expertTokens由数组改为Tensor输入。** 

  相较于[FFN](aclnnFFN.md)接口，**此接口新增支持expertTokens索引输入，用tokensIndexFlag区分。expertTokens由数组改为Tensor输入。**
- 计算公式：

  - **非量化场景：**

    $$
    y=activation(x * W1 + b1) * W2 + b2
    $$

  - **量化场景：**

    $$
    y=((activation((x * W1 + b1) * deqScale1) * scale + offset) * W2 + b2) * deqScale2
    $$

  - **伪量化场景：**

    $$
    y=activation(x * ((W1 + antiquantOffset1) * antiquantScale1) + b1) * ((W2 + antiquantOffset2) * antiquantScale2) + b2
    $$

  **说明：**

  FFN在无专家或单个专家场景是否有性能收益需要根据实际测试情况判断，当整网中FFN结构对应的小算子vector耗时超过30us，且在FFN结构中占比10%以上时，可以尝试使用该融合算子，若实际测试性能劣化则不使用。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFFNV3GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFFNV3”接口执行计算。

```Cpp
aclnnStatus aclnnFFNV3GetWorkspaceSize(
  const aclTensor* x, 
  const aclTensor* weight1, 
  const aclTensor* weight2, 
  const aclTensor* expertTokensOptional, 
  const aclTensor* bias1Optional, 
  const aclTensor* bias2Optional, 
  const aclTensor* scaleOptional, 
  const aclTensor* offsetOptional, 
  const aclTensor* deqScale1Optional, 
  const aclTensor* deqScale2Optional, 
  const aclTensor* antiquantScale1Optional, 
  const aclTensor* antiquantScale2Optional, 
  const aclTensor* antiquantOffset1Optional, 
  const aclTensor* antiquantOffset2Optional, 
  const char*      activation, 
  int64_t          innerPrecise, 
  bool             tokensIndexFlag, 
  const aclTensor* y, 
  uint64_t*        workspaceSize, 
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnFFNV3(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnFFNV3GetWorkspaceSize

**说明：** 下述参数说明中涉及到的变量说明  
  M表示token个数，对应transform中的BS（B：Batch，表示输入样本批量大小。   
  S：Seq-Length，表示输入样本序列长度）。  
  K1表示第一个matmul的输入通道数，对应transform中的H（Head-Size，表示隐藏层的大小）。  
  N1表示第一个matmul的输出通道数。  
  K2表示第二个matmul的输入通道数。  
  N2表示第二个matmul的输出通道数，对应transform中的H。    
  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：E表示有专家场景的专家数；G表示伪量化per-group场景下，antiquantOffset、antiquantScale的组数。

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1645px">
  <colgroup> <!--工具调整列宽时，所列宽总和最大不要超过1550--> 
  <col style="width: 180px"> <!-- 参数名：自行调整列宽，原则：不能换行展示--> 
  <col style="width: 120px"> <!-- 输入/输出：固定列宽--> 
  <col style="width: 280px"> <!-- 描述列宽：自行调整列宽--> 
  <col style="width: 300px"> <!-- 使用说明列宽：自行调整列宽--> 
  <col style="width: 300px"> <!-- 数据类型列宽：自行调整列宽--> 
  <col style="width: 120px"> <!--数据格式：自行调整列宽--> 
  <col style="width: 300px"> <!-- 维度(shape)：自行调整列宽，内容较少时可以适当缩小列宽--> 
  <col style="width: 145px"> <!--非连续Tensor：自行调整列宽，必须不要修改这个数值--> 
  </colgroup> 
  <thead> 
  <tr> 
  <th>参数名</th> 
  <th>输入/输出</th> 
  <th>描述</th> 
  <th>使用说明</th> 
  <th>数据类型</th> 
  <th>数据格式</th> 
  <th>维度(shape)</th> 
  <th>非连续Tensor</th> 
  </tr>
  </thead> 
  <tbody> 
  <tr> 
  <td>x（aclTensor*）</td> 
  <td>输入</td> 
  <td>计算输入，公式中的输入x。</td> 
  <td>
  <ul>
  <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持输入的维度最少是2维，最多是8维。</li>
  <li><term>Atlas 推理系列加速卡产品</term>：支持输入的维度是2维。</li>
  </ul>
  </td> 
  <td>
  <ul>
  <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：FLOAT16、BFLOAT16、INT8</li>
  <li><term>Atlas 推理系列加速卡产品</term>：FLOAT16</li>
  </ul>
  </td> 
  <td>ND</td> 
  <td>
  <ul>
  <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：[M, K1]</li>
  <li><term>Atlas 推理系列加速卡产品</term>：[M, K1]</li>
  </ul>
  </td> 
  <td>√</td> 
  </tr> 
  <tr> 
  <td>weight1（aclTensor*）</td> 
  <td>输入</td> 
  <td>专家的权重数据，公式中的W1。</td> 
  <td><term>Atlas 推理系列加速卡产品</term>：支持输入的维度是2维。</td> 
  <td>
  <ul>
  <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：FLOAT16、BFLOAT16、INT8、INT4</li>
  <li><term>Atlas 推理系列加速卡产品</term>：FLOAT16</li>
  </ul>
  </td> 
  <td>ND</td> 
  <td>
  <ul>
  <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：有专家[E, K1, N1]；无专家[K1, N1]</li>
  <li><term>Atlas 推理系列加速卡产品</term>：[K1, N1]</li>
  </ul>
  </td> 
  <td>√</td> 
  </tr> 
  <tr> 
  <td>weight2（aclTensor*）</td> 
  <td>输入</td> 
  <td>专家的权重数据，公式中的W2。</td> 
  <td><term>Atlas 推理系列加速卡产品</term>：支持输入的维度是2维。</td> 
  <td>
  <ul>
  <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：FLOAT16、BFLOAT16、INT8、INT4</li>
  <li><term>Atlas 推理系列加速卡产品</term>：FLOAT16</li>
  </ul>
  </td> 
  <td>ND</td> 
  <td>
  <ul>
  <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：有专家[E, K2, N2]；无专家[K2, N2]</li>
  <li><term>Atlas 推理系列加速卡产品</term>：[K2, N2]</li>
  </ul>
  </td> 
  <td>√</td> 
  </tr> 
  <tr> 
  <td>expertTokensOptional（aclTensor*）</td> 
  <td>可选输入</td> 
  <td>各专家的token数。</td> 
  <td>
  <ul>
  <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：若不为空时可支持的最大长度为256个。</li>
  <li><term>Atlas 推理系列加速卡产品</term>：只支持传空指针。</li>
  </ul>
  </td> 
  <td>INT64</td> 
  <td>ND</td> 
  <td>1维，最大长度256</td> 
  <td>-</td> 
  </tr> 
  <tr> 
  <td>bias1Optional（aclTensor*）</td> 
  <td>可选输入</td> 
  <td>权重数据修正值，公式中的b1。</td> 
  <td><term>Atlas 推理系列加速卡产品</term>：支持输入的维度是1维。</td> 
  <td>
  <ul>
  <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>Atlas 800I A2推理产品：FLOAT16、FLOAT32、INT32</li>
  <li><term>Atlas 推理系列加速卡产品</term>：FLOAT16</li>
  </ul>
  </td> 
  <td>ND</td> 
  <td>
  <ul>
  <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：有专家[E, N1]；无专家[N1]</li>
  <li><term>Atlas 推理系列加速卡产品</term>：[N1]</li>
  </ul>
  </td> 
  <td>-</td> 
  </tr> 
  <tr> 
  <td>bias2Optional（aclTensor*）</td> 
  <td>可选输入</td> 
  <td>权重数据修正值，公式中的b2。</td> 
  <td><term>Atlas 推理系列加速卡产品</term>：支持输入的维度是1维。</td> 
  <td>
  <ul>
  <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：FLOAT16、FLOAT32、INT32</li>
  <li><term>Atlas 推理系列加速卡产品</term>：FLOAT16</li>
  </ul>
  </td> 
  <td>ND</td> 
  <td>
  <ul>
  <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：有专家[E, N2]；无专家[N2]</li>
  <li><term>Atlas 推理系列加速卡产品</term>：[N2]</li>
  </ul>
  </td> 
  <td>-</td> 
  </tr> 
  <tr> 
  <td>scaleOptional（aclTensor*）</td> 
  <td>可选输入</td> 
  <td>量化参数，量化缩放系数。</td> 
  <td><term>Atlas 推理系列加速卡产品</term>：只支持传空指针。</td> 
  <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：FLOAT32</td> 
  <td>ND</td> 
  <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：<ul><li>per-tensor下输入在有/无专家时均为一维向量，输入元素个数在有/无专家时分别为[E]/[1]</li><li>per-channel下输入在有/无专家时为二维向量/一维向量，输入元素个数在有/无专家时分别为[E, N1]/[N1]</li></ul></td> 
  <td>√</td> 
  </tr> 
  <tr> 
  <td>offsetOptional（aclTensor*）</td> 
  <td>可选输入</td> 
  <td>量化参数，量化偏移量。</td> 
  <td><term>Atlas 推理系列加速卡产品</term>：只支持传空指针。</td> 
  <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：FLOAT32</td> 
  <td>ND</td> 
  <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：一维向量，输入元素个数在有/无专家时分别为[E]/[1]</td> 
  <td>-</td> 
  </tr> 
  <tr> 
  <td>deqScale1Optional（aclTensor*）</td> 
  <td>可选输入</td> 
  <td>量化参数，第一个matmul的反量化缩放系数。</td> 
  <td><term>Atlas 推理系列加速卡产品</term>：只支持传空指针。</td> 
  <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：UINT64、INT64、FLOAT32、BFLOAT16</td> 
  <td>ND</td> 
  <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：输入在有/无专家时分别为[E, N1]/[N1]</td> 
  <td>-</td> 
  </tr> 
  <tr> 
  <td>deqScale2Optional（aclTensor*）</td> 
  <td>可选输入</td> 
  <td>量化参数，第二个matmul的反量化缩放系数。</td> 
  <td><term>Atlas 推理系列加速卡产品</term>：只支持传空指针。</td> 
  <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：UINT64、INT64、FLOAT32、BFLOAT16</td> 
  <td>ND</td> 
  <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：输入在有/无专家时分别为[E, N2]/[N2]</td> 
  <td>-</td> 
  </tr> 
  <tr> 
  <td>antiquantScale1Optional（aclTensor*）</td> 
  <td>可选输入</td> 
  <td>伪量化参数，第一个matmul的缩放系数。</td> 
  <td><term>Atlas 推理系列加速卡产品</term>：只支持传空指针。</td> 
  <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：FLOAT16、BFLOAT16</td> 
  <td>ND</td> 
  <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：<ul><li>per-channel下输入在有/无专家时分别为[E, N1]/[N1]</li><li>per-group下输入在有/无专家时分别为[E, G, N1]/[G, N1]</li></ul></td> 
  <td>√</td> 
  </tr> 
  <tr> 
  <td>antiquantScale2Optional（aclTensor*）</td> 
  <td>可选输入</td> 
  <td>伪量化参数，第二个matmul的缩放系数。</td> 
  <td><term>Atlas 推理系列加速卡产品</term>：只支持传空指针。</td> 
  <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：FLOAT16、BFLOAT16</td> 
  <td>ND</td> 
  <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：<ul><li>per-channel下输入在有/无专家时分别为[E, N2]/[N2]</li><li>per-group下输入在有/无专家时分别为[E, G, N2]/[G, N2]</li></ul></td> 
  <td>√</td> 
  </tr> 
  <tr> 
  <td>antiquantOffset1Optional（aclTensor*）</td> 
  <td>可选输入</td> 
  <td>伪量化参数，第一个matmul的偏移量。</td> 
  <td><term>Atlas 推理系列加速卡产品</term>：只支持传空指针。</td> 
  <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：FLOAT16、BFLOAT16</td> 
  <td>ND</td> 
  <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：<ul><li>per-channel下输入在有/无专家时分别为[E, N1]/[N1]</li><li>per-group下输入在有/无专家时分别为[E, G, N1]/[G, N1]</li></ul></td> 
  <td>√</td> 
  </tr> 
  <tr> 
  <td>antiquantOffset2Optional（aclTensor*）</td> 
  <td>可选输入</td> 
  <td>伪量化参数，第二个matmul的偏移量。</td> 
  <td><term>Atlas 推理系列加速卡产品</term>：只支持传空指针。</td> 
  <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：FLOAT16、BFLOAT16</td> 
  <td>ND</td> 
  <td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：<ul><li>per-channel下输入在有/无专家时分别为[E, N2]/[N2]</li><li>per-group下输入在有/无专家时分别为[E, G, N2]/[G, N2]</li></ul></td> 
  <td>√</td> 
  </tr> 
  <tr> 
  <td>activation（char*）</td> 
  <td>输入</td> 
  <td>代表使用的激活函数，公式中的activation。</td> 
  <td>
  <ul>
  <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：当前支持fastgelu/gelu/relu/silu以及geglu/swiglu/reglu。</li>
  <li><term>Atlas 推理系列加速卡产品</term>：当前支持fastgelu/gelu/relu/silu。</li>
  </ul>
  </td> 
  <td>-</td> 
  <td>-</td> 
  <td>-</td> 
  <td>-</td> 
  </tr> 
  <tr> 
  <td>innerPrecise（int64_t）</td> 
  <td>可选输入</td> 
  <td>表示高精度或者高性能选择。</td> 
  <td>
  <ul>
  <li>innerPrecise为0时，代表开启高精度模式，非量化场景下必选参数都为FLOAT16时，算子内部激活层输入输出都采用FLOAT32数据类型计算。</li>
  <li>innerPrecise为1时，代表高性能模式。</li>
  <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：该参数仅在非量化场景下必选参数都为FLOAT16时生效，其余场景不区分高精度和高性能。</li>
  <li><term>Atlas 推理系列加速卡产品</term>：只支持传1。</li>
  </ul>
  </td> 
  <td>INT64</td> 
  <td>-</td> 
  <td>-</td> 
  <td>-</td> 
  </tr> 
  <tr> 
  <td>tokensIndexFlag（bool）</td> 
  <td>可选输入</td> 
  <td>指示expertTokens是否为索引值。</td> 
  <td>
  <ul>
  <li>tokensIndexFlag为true时，表示expertTokens为索引值。</li>
  <li>tokensIndexFlag为false时，表示expertTokens为各专家的token数。</li>
  </ul>
  </td> 
  <td>bool</td> 
  <td>-</td> 
  <td>-</td> 
  <td>-</td> 
  </tr> 
  <tr> 
  <td>y（aclTensor*）</td> 
  <td>输出</td> 
  <td>公式中的输出y。</td> 
  <td>
  <ul>
  <li>输出维度与x一致。</li>
  </ul>
  </td> 
  <td>
  <ul>
  <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：FLOAT16、BFLOAT16</li>
  <li><term>Atlas 推理系列加速卡产品</term>：FLOAT16</li>
  </ul>
  </td> 
  <td>ND</td> 
  <td>与x一致</td> 
  <td>√</td> 
  </tr> 
  <tr> 
  <td>workspaceSize（uint64_t*）</td> 
  <td>出参</td> 
  <td>返回用户需要在Device侧申请的workspace大小。</td> 
  <td>-</td> 
  <td>-</td> 
  <td>-</td> 
  <td>-</td> 
  <td>-</td> 
  </tr> 
  <tr> 
  <td>executor（aclOpExecutor**）</td> 
  <td>出参</td> 
  <td>返回op执行器，包含了算子计算流程。</td> 
  <td>-</td> 
  <td>-</td> 
  <td>-</td> 
  <td>-</td> 
  <td>-</td> 
  </tr> 
  </tbody>
  </table>

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

   第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 294px">
  <col style="width: 134px">
  <col style="width: 722px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>x、weight1、weight2、activation、expertTokensOptional、bias1Optional、bias2Optional、y的数据类型和数据格式不在支持的范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnFFNV3

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 848px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFFNV3GetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnFFNV3默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
- 所有场景下需满足K1=N2, K1<65536, K2<65536, M轴在32Byte对齐后小于INT32的最大值。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - 含BFLOAT16数据类型的场景仅支持Atlas 800I A2推理产品。
  - 有专家时，专家数据的总数需要与x的M保持一致。
  - 激活层为geglu/swiglu/reglu时，仅支持无专家分组时的FLOAT16高性能场景（FLOAT16场景指类型为aclTensor的必选参数数据类型都为FLOAT16的场景），且N1=2*K2。
  - 激活层为gelu/fastgelu/relu/silu时，支持有专家或无专家分组的FLOAT16高精度及高性能场景、BFLOAT16场景、量化场景及伪量化场景，且N1=K2。
  - 非量化场景不能输入量化参数和伪量化参数，量化场景不能输入伪量化参数，伪量化场景不能输入量化参数。
  - 量化场景参数类型：x为INT8、weight为INT8、biasOptional为INT32、scaleOptional为FLOAT32、offsetOptional为FLOAT32，其余参数类型根据y不同分两种情况：
    - y为FLOAT16，deqScaleOptional支持数据类型：UINT64、INT64、FLOAT32；
    - y为BFLOAT16，deqScaleOptional支持数据类型：BFLOAT16；
    - 要求deqScale1Optional与deqScale2Optional的数据类型保持一致。
  - 量化场景支持scale的per-channel模式参数类型：x为INT8、weight为INT8、biasOptional为INT32、scaleOptional为FLOAT32、offsetOptional为FLOAT32，其余参数类型根据y不同分两种情况：
    - y为FLOAT16，deqScaleOptional支持数据类型：UINT64、INT64；
    - y为BFLOAT16，deqScaleOptional支持数据类型：BFLOAT16；
    - 要求deqScale1Optional与deqScale2Optional的数据类型保持一致。
  - 伪量化场景支持两种不同参数类型：
    - y为FLOAT16、x为FLOAT16、biasOptional为FLOAT16，antiquantScaleOptional为FLOAT16、antiquantOffsetOptional为FLOAT16，weight支持数据类型INT8和INT4。
    - y为BFLOAT16、x为BFLOAT16、biasOptional为FLOAT32，antiquantScaleOptional为BFLOAT16、antiquantOffsetOptional为BFLOAT16，weight支持数据类型INT8和INT4。
  - 当weight1/weight2的数据类型为INT4时，其shape最后一维必须为偶数。
  - 伪量化场景，per-group下，antiquantScale1Optional和antiquantOffset1Optional中的组数G要能被K1整除，antiquantScale2Optional和antiquantOffset2Optional中的组数G要能被K2整除。
  - innerPrecise参数在BFLOAT16非量化场景，只能配置为0；FLOAT16非量化场景，可以配置为0或者1；量化或者伪量化场景，0和1都可配置，但是配置后不生效。
  - tokensIndexFlag为true且有专家（expertTokens不为空）时，expertTokens中的数值必须满足：如果i和j都是expertTokens中有效的数组索引，且j大于i，那么expertTokens中第j个元素的数值大于或者等于expertTokens中第i个元素的数值。

- <term>Atlas 推理系列加速卡产品</term>：
  - 只支持无专家场景。
  - 需满足N1=K2。

## 调用示例

该融合算子接口只支持aclnn单算子调用方式，调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ffn_v3.h"
#include "aclnn/opdev/fp16_t.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  std::vector<int64_t> weight1Shape = {2, 2};
  std::vector<int64_t> weight2Shape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* weight1DeviceAddr = nullptr;
  void* weight2DeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* weight1 = nullptr;
  aclTensor* weight2 = nullptr;
  std::vector<op::fp16_t> selfHostData = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  std::vector<op::fp16_t> outHostData = {0, 0, 0, 0};
  std::vector<op::fp16_t> weight1HostData = {0.1, 0.2, 0.3, 0.4};
  std::vector<op::fp16_t> weight2HostData = {0.4, 0.3, 0.2, 0.1};
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建weight1 aclTensor
  ret = CreateAclTensor(weight1HostData, weight1Shape, &weight1DeviceAddr, aclDataType::ACL_FLOAT16, &weight1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建weight2 aclTensor
  ret = CreateAclTensor(weight2HostData, weight2Shape, &weight2DeviceAddr, aclDataType::ACL_FLOAT16, &weight2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnFFNV3接口调用示例
  LOG_PRINT("test aclnnFFNV3\n");

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  // 调用aclnnFFNV3第一段接口
  ret = aclnnFFNV3GetWorkspaceSize(self, weight1, weight2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,   "relu", 1, false, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFFNV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnFFNV3第二段接口
  ret = aclnnFFNV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFFNV3 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<op::fp16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(weight1);
  aclDestroyTensor(weight2);

  // 7. 释放device资源
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(weight1DeviceAddr);
  aclrtFree(weight2DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
