# GroupedMatmulFinalizeRouting

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |

## 功能说明

- 算子功能：GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 120px">
  <col style="width: 301px">
  <col style="width: 219px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 143px">
  <col style="width: 146px">
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
      <td>x1</td>
      <td>输入</td>
      <td>输入x(左矩阵)。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>输入weight(右矩阵)</td>
      <td>INT4、INT8</td>
      <td>ND、NZ</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>输入</td>
      <td>量化参数中的缩放因子，perchannel量化参数</td>
      <td>INT64、BF16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>biasOptional</td>
      <td>输入</td>
      <td>矩阵的偏移</td>
      <td>BF16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offsetOptional</td>
      <td>输入</td>
      <td>非对称量化的偏移量</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>antiquantScaleOptional</td>
      <td>输入</td>
      <td>伪量化的缩放因子</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>antiquantOffsetOptional</td>
      <td>输入</td>
      <td>伪量化的偏移量</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pertokenScaleOptional</td>
      <td>输入</td>
      <td>矩阵计算的反量化参数</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>groupListOptional</td>
      <td>输入</td>
      <td>输入和输出分组轴方向的matmul大小分布</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sharedInputOptional</td>
      <td>输入</td>
      <td>moe计算中共享专家的输出，需要与moe专家的输出进行combine操作</td>
      <td>BF16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>logitOptional</td>
      <td>输入</td>
      <td>moe专家对各个token的logit大小</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rowIndexOptional</td>
      <td>输入</td>
      <td>moe专家输出按照该rowIndex进行combine，其中的值即为combine做scatter add的索引</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>属性</td>
      <td>计算的输出类型：0：FLOAT32；1：FLOAT16；2：BFLOAT16。目前仅支持0。</td>
      <td>INT64</td>
      <td></td>
    </tr>
    <tr>
      <td>sharedInputWeight</td>
      <td>属性</td>
      <td>共享专家与moe专家进行combine的系数，sharedInput先与该参数乘，然后在和moe专家结果累加。</td>
      <td>FLOAT</td>
      <td></td>
    </tr>
    <tr>
      <td>sharedInputOffset</td>
      <td>属性</td>
      <td>共享专家输出的在总输出中的偏移。</td>
      <td>INT64</td>
      <td></td>
    </tr>
    <tr>
      <td>transposeX</td>
      <td>属性</td>
      <td>左矩阵是否转置，仅支持false。</td>
      <td>BOOL</td>
      <td></td>
    </tr>
    <tr>
      <td>transposeW</td>
      <td>属性</td>
      <td>右矩阵是否转置，仅支持false。</td>
      <td>BOOL</td>
      <td></td>
    </tr>
    <tr>
      <td>groupListType</td>
      <td>属性</td>
      <td>分组模式：配置为0：cumsum模式，即为前缀和；配置为1：count模式。</td>
      <td>INT64</td>
      <td></td>
    </tr>
    <tr>
      <td>tuningConfigOptional</td>
      <td>属性</td>
      <td>数组中的第一个元素表示各个专家处理的token数的预期值，算子tiling时会按照数组的第一个元素合理进行tiling切分，性能更优。从第二个元素开始预留，用户无须填写。未来会进行扩展。兼容历史版本，用户如不使用该参数，不传入（即为nullptr）即可。</td>
      <td>INT64</td>
      <td></td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出结果。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

## 约束说明

输入和输出支持以下数据类型组合：

| x1   | x2         | scaleOptional | biasOptional | offsetOptional | antiquantScaleOptional | antiquantOffsetOptional | pertokenScaleOptional | groupListOptional | sharedInputOptional | logitOptional | rowIndexOptional | out     |
| ---- | ---------- | ------------- | ------------ | -------------- | ---------------------- | ----------------------- | --------------------- | ----------------- | ------------------- | ------------- | ---------------- | ------- |
| INT8 | INT4       | INT64         | FLOAT32      | FLOAT32        | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT32 |
| INT8 | INT4       | INT64         | FLOAT32      | null           | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT32 |
| INT8 | INT8（NZ） | FLOAT32       | null         | null           | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT   |
| INT8 | INT8（NZ） | FLOAT32       | null         | null           | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT   |
| INT8 | INT4（NZ） | INT64         | FLOAT32      | FLOAT32        | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT   |
| INT8 | INT4（NZ） | INT64         | FLOAT32      | null           | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT   |

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
|aclnn调用|[test_aclnn_grouped_matmul_finalize_routing](examples/test_aclnn_grouped_matmul_finalize_routing.cpp)|通过[aclnnGroupedMatmulFinalizeRoutingV3](aclnnGroupedMatmulFinalizeRoutingV3.md)接口方式调用GroupedMatmulFinalizeRouting算子。|
|aclnn调用|[test_aclnn_grouped_matmul_finalize_routing_weight_nz](../grouped_matmul_finalize_routing/examples/arch35/test_aclnn_grouped_matmul_finalize_routing_weightnz.cpp)|通过[aclnnGroupedMatmulFinalizeRoutingWeightNzV2](aclnnGroupedMatmulFinalizeRoutingWeightNzV2.md)接口方式调用GroupedMatmulFinalizeRoutingWeightNz算子。 |
