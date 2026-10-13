# MoeInitRoutingV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：该算子对应MoE（Mixture of Experts，混合专家模型）中的**Routing计算**，以MoeGatingTopKSoftmax算子的输出x和expertIdx作为输入，并输出Routing矩阵expanded_x等结果供后续计算使用。本接口针对V1接口（MoeInitRouting，源码未开放）做了如下功能变更，请根据实际情况选择合适的接口：

  - 新增Drop模式，在该模式下输出内容会将每个专家需要处理的Token个数对齐为expertCapacity个，超过expertCapacity个的Token会被Drop，不足的会用0填充。
  - 新增Dropless模式下expertTokensCountOrCumsumOut可选输出，输出每个专家需要处理的累积Token个数（Cumsum），或每个专家需要处理的Token数（Count）。
  - 新增Drop模式下expertTokensBeforeCapacityOut可选输出，输出每个专家在Drop前应处理的Token个数。
  - 删除rowIdx输入。

  **说明：**
  Routing计算是MoE模型中的一个环节。MoE模型主要由一组专家模型和一个门控模型组成，在计算时，输入的数据会先根据门控网络（Gating Network，包含MoeGatingTopKSoftmax算子）计算出每个数据元素对应权重最高的k个专家，然后该结果会输入MoeInitRouting算子，生成Routing矩阵。在后续，模型中的每个专家会根据Routing矩阵处理其应处理的数据，产生相应的输出。各专家的输出最后与权重加权求和，形成最终的预测结果。

- 计算公式：  

  1.对输入expertIdx做排序，得出排序后的结果sortedExpertIdx和对应的序号sortedRowIdx：
  
  $$
  sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx)
  $$

  2.以sortedRowIdx做位置映射得出expandedRowIdxOut：
  
  $$
  expandedRowIdxOut[sortedRowIdx[i]]=i
  $$
  
  3.对x取前numRows个sortedRowIdx的对应位置的值，得出expandedXOut：
  
  $$
  expandedXOut[i]=x[sortedRowIdx[i]\%numRows]
  $$
  
  4.对sortedExpertIdx的每个专家统计直方图结果，再进行Cumsum，得出expertTokensCountOrCumsumOut：
  
  $$
  expertTokensCountOrCumsumOut[i]=Cumsum(Histogram(sortedExpertIdx))
  $$
  
  5.对sortedExpertIdx的每个专家统计直方图结果，得出expertTokensBeforeCapacityOut：
  
  $$
  expertTokensBeforeCapacityOut[i]=Histogram(sortedExpertIdx)
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
      <td>为MOE的输入，即token特征输入。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expertIdx</td>
      <td>输入</td>
      <td>为每个Token对应的k个处理专家的序号。</td>
      <td>Ascend 950PR/Ascend 950DT：INT32、INT64/其他处理器：INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activeNum</td>
      <td>属性</td>
      <td>表示是否为Active场景。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertCapacity</td>
      <td>属性</td>
      <td>表示每个专家能够处理的tokens数。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertNum</td>
      <td>属性</td>
      <td>表示专家数，值范围大于等于0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropPadMode</td>
      <td>属性</td>
      <td>表示是否为Drop/Pad场景。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumFlag</td>
      <td>属性</td>
      <td>取值为0、1和2。
        0：表示不输出expertTokensCountOrCumsumOut。
        1：表示输出的值为各个专家处理的token数量的累计值。
        2：表示输出的值为各个专家处理的token数量。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensBeforeCapacityFlag</td>
      <td>属性</td>
      <td>取值为false和true。
        false：表示不输出expertTokensBeforeCapacityOut。
        true：表示输出的值为在drop之前各个专家处理的token数量。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedXOut</td>
      <td>输出</td>
      <td>根据expertIdx进行扩展过的特征，在Dropless/Active场景下要求是一个2D的Tensor。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expandedRowIdxOut</td>
      <td>输出</td>
      <td>expandedXOut和x的索引映射关系， 要求是一个1D的Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumOut</td>
      <td>输出</td>
      <td>输出每个专家处理的token数量的统计结果及累加值，通过expertTokensCountOrCumsumFlag参数控制是否输出。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expertTokensBeforeCapacityOut</td>
      <td>输出</td>
      <td>输出drop之前每个专家处理的token数量的统计结果。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入x都必须为2维，且x的numRows等于expertIdx的numRows。
- dropPadMode为1时，expertNum和expertCapacity必须大于0。
- x的numRows轴必须大于expertCapacity。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_init_routing_v2](examples/test_aclnn_moe_init_routing_v2.cpp) | 通过[aclnnMoeInitRoutingV2](aclnnMoeInitRoutingV2.md)接口方式调用MoeInitRoutingV2算子。 |
