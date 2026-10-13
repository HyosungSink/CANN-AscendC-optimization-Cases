# AddRmsNormQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | √ |
|  <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：RmsNorm是大模型常用的标准化操作，相比LayerNorm，其去掉了减去均值的部分。AddRmsNormQuant算子将RmsNorm前的Add算子以及RmsNorm归一化的输出给到1个或2个Quantize算子融合起来，减少搬入搬出操作。

- 计算公式：

  $$
  x_i={x1}_i+{x2}_i
  $$

  $$
  y_i=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i + beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  - divMode为True时：

    $$
    y1=round((y/scales1)+zero\_points1)
    $$

    $$
    y2=round((y/scales2)+zero\_points2)
    $$

  - divMode为False时：

    $$
    y1=round((y*scales1)+zero\_points1)
    $$

    $$
    y2=round((y*scales2)+zero\_points2)
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td>x1</td>
      <td>输入</td>
      <td>表示标准化过程中的源数据张量，对应公式中的`x1`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示标准化过程中的源数据张量，对应公式中的`x2`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示标准化过程中的权重张量，对应公式中的`g`。shape与x1需要norm（层归一化）的维度保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales1</td>
      <td>输入</td>
      <td>表示量化过程中得到y1的scales张量，对应公式中的`scales1`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales2</td>
      <td>可选输入</td>
      <td>表示量化过程中得到y2的scales张量，对应公式中的`scales2`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points1</td>
      <td>可选输入</td>
      <td>表示量化过程中得到y1的offset张量，对应公式中的`zero_points1`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points2</td>
      <td>可选输入</td>
      <td>表示量化过程中得到y2的offset张量，对应公式中的`zero_points2`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>可选输入</td>
      <td>表示标准化过程中的偏置项，公式中的`beta`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>可选属性</td>
      <td><ul><li>表示需要进行量化的elewise轴，其他的轴做broadcast，指定的轴不能超过输入x的维度数。当前仅支持-1，传其他值均不生效。</li><li>默认值为-1。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>用于防止除0错误，对应公式中的`epsilon`。</li><li>默认值为1e-6。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>div_mode</td>
      <td>可选属性</td>
      <td><ul><li>公式中决定量化公式是否使用除法的参数，对应公式中的`divMode`。</li><li>默认值为true。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的`y1`。</td>
      <td>INT8、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的`y2`。</td>
      <td>INT8、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输出</td>
      <td>表示x1和x2的和，对应公式中的`x`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas 推理系列产品</term>：x1、x2、gamma、scales1、scales2、zero_points1、zero_points2、beta、x的数据类型不支持BFLOAT16，输出参数y1、y2的数据类型仅支持INT8。

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：x1、x2、gamma、x、beta的数据类型不支持FLOAT32，输出参数y1、y2的数据类型仅支持INT8。

  - <term>Ascend 950PR/Ascend 950DT</term>：可选输入beta不支持配置。
  - Kirin X90/Kirin 9030处理器系列产品：
    - x1、x2、gamma、beta和x的数据类型只支持FLOAT16。
    - scales1和scales2的数据类型只支持FLOAT32。
    - zero_points1和zero_points2的数据类型只支持INT32。
    - y1和y2的数据类型只支持INT8。

## 约束说明

<term>Atlas 推理系列产品</term>：`x1`、`x2`需要norm的维度数据个数不能小于32。`gamma`、`beta`、`scales1`、`scales2`、`zero_points1`、`zero_points2`的数据个数不能小于32。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_add_rms_norm_quant](examples/test_aclnn_add_rms_norm_quant.cpp) | 通过[aclnnAddRmsNormQuant](docs/aclnnAddRmsNormQuant.md)接口方式调用AddRmsNormQuant算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/add_rms_norm_quant_proto.h)构图方式调用AddRmsNormQuant算子。         |
