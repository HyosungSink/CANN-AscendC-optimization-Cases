# WeightQuantBatchMatmulV2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend 950PR/Ascend 950DT|√|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

- **算子功能**：完成一个输入为伪量化场景的矩阵乘计算，并可以实现对于输出的量化计算。
- **计算公式**：

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

  公式中的$weight$为伪量化场景的输入，其反量化公式$ANTIQUANT(weight)$为

  $$
  ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
  $$

  当需要对输出进行量化处理时，其量化公式为

  $$
  \begin{aligned}
  y &= QUANT(x @ ANTIQUANT(weight) + bias) \\
  &= (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset \\
  \end{aligned}
  $$

  当不需要对输出再进行量化操作时，其计算公式为

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

## 参数说明

<table class="tg"><thead>
  <tr>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">参数名</span></th>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">输入/输出/属性</span></th>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">描述</span></th>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">数据类型</span></th>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">数据格式</span></th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">x</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">输入</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">矩阵乘运算中的左矩阵。</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT16, BF16</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">weight</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">输入</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">矩阵乘运算中的右矩阵。</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">INT8, INT4, INT32, FLOAT8_E4M3FN, HIFLOAT8, FLOAT4_E2M1</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND, FRACTAL_NZ</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">antiquant_scale</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">输入</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">反量化参数中的缩放因子，对应公式的antiquantScale。</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT16, BF16, UINT64, INT64, FLOAT8_E8M0</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">antiquant_offset</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">输入</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">反量化参数的偏置因子，对应公式的antiquantOffset。</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT16, BF16, INT32</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">quant_scale</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">输入</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">量化参数的缩放因子，对应公式的quantScale。</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT32, UINT64</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">quant_offset</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">输入</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">量化参数的偏置因子，对应公式的quantOffset。</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT32</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">bias</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">输入</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">矩阵乘运算后累加的偏置，对应公式中的bias。</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT16, FLOAT32, BF16</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">y</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">输出</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">矩阵乘运算的计算结果。</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT16, BF16, INT8</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
</tbody></table>

- Atlas A2 训练系列产品/Atlas A2 推理系列产品：
  - weight只支持INT8、INT4、INT32。
  - antiquant_scale只支持FLOAT16、BF16、UINT64、INT64。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：
  - weight只支持INT8、INT4、INT32。
  - antiquant_scale只支持FLOAT16、BF16、UINT64、INT64。
- Ascend 950PR/Ascend 950DT：quant_scale和quant_offset暂不支持。

## 约束说明

- 不支持空tensor。
- 支持连续tensor，[非连续tensor](../../docs/zh/context/非连续的Tensor.md)只支持转置场景。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_weight_quant_batch_matmul_v2](examples/test_aclnn_weight_quant_batch_matmul_v2.cpp) | 通过<br>[aclnnWeightQuantBatchMatmulV2](docs/aclnnWeightQuantBatchMatmulV2.md)<br>[aclnnWeightQuantBatchMatmulV3](docs/aclnnWeightQuantBatchMatmulV3.md)<br>[aclnnWeightQuantBatchMatmulNz](docs/aclnnWeightQuantBatchMatmulNz.md)<br>等方式调用WeightQuantBatchMatmulV2算子。|
