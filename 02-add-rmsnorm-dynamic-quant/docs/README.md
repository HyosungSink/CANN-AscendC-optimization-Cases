# AddRmsNormDynamicQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | √ |
|  <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。DynamicQuant算子则是为输入张量进行对称动态量化的算子。AddRmsNormDynamicQuantV2算子将RmsNorm前的Add算子和RmsNorm归一化输出给到的1个或2个DynamicQuant算子融合起来，减少搬入搬出操作。AddRmsNormDynamicQuant算子相较于AddRmsNormDynamicQuantV2在RmsNorm计算过程中增加了偏置项betaOptional参数，即计算对应公式中的beta，以及新增输出配置项output_mask参数，用于配置是否输出对应位置的量化结果。

- 计算公式：

  $$
  x=x_{1}+x_{2}
  $$

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  $$
  input1 =\begin{cases}
    y\cdot smoothScale1Optional & \ \ smoothScale1Optional\ != null \\
    y & \ \  smoothScale1Optional\ = null
    \end{cases}
  $$

  $$
  input2 =\begin{cases}
    y\cdot smoothScale2Optional & \ \ smoothScale2Optional\ != null  \\
    y & \ \ smoothScale2Optional\ = null
    \end{cases}
  $$
  
  $$
  scale1Out=\begin{cases}
    row\_max(abs(input1))/127 & outputMask[0]=True\ ||\ outputMask\ = null \\
    无效输出 & outputMask[0]=False
    \end{cases}
  $$

  $$
  y1Out=\begin{cases}
    round(input1/scale1Out) & outputMask[0]=True\ ||\ outputMask\ = null \\
    无效输出 & outputMask[0]=False
    \end{cases}
  $$
  
  $$
  scale2Out=\begin{cases}
    row\_max(abs(input2))/127 & outputMask[1]=True\ ||\ (outputMask\ = null\ \&\ smoothScale1Optional\ != null\ \&\ smoothScale2Optional\ != null) \\
    无效输出 & outputMask[1]=False\ ||\ (outputMask\ = null\ \&\ smoothScale1Optional\ != null\ \&\ smoothScale2Optional\ = null)
    \end{cases}
  $$
  
  $$
  y2Out=\begin{cases}
    round(input2/scale2Out) & outputMask[1]=True\ ||\ (outputMask\ = null\ \&\ smoothScale1Optional\ != null\ \&\ smoothScale2Optional\ != null)\\
    无效输出 & outputMask[1]=False\ ||\ (outputMask\ = null\ \&\ smoothScale1Optional\ != null\ \&\ smoothScale2Optional\ = null)
    \end{cases}
  $$
  
  公式中的row\_max代表每行求最大值。

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
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示标准化过程中的源数据张量，对应公式中的`x2`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示标准化过程中的权重张量，对应公式中的`gamma`。shape需要与x1最后一维一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scale1</td>
      <td>可选输入</td>
      <td>表示量化过程中得到y1使用的smoothScale张量，对应公式中的`smoothScale1Optional`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scale2</td>
      <td>可选输入</td>
      <td>表示量化过程中得到y2使用的smoothScale张量，对应公式中的`smoothScale2Optional`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>可选输入</td>
      <td>表示标准化过程中的偏置项，对应公式中的`beta`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>用于防止除0错误，对应公式中的`epsilon`。</li><li>默认值为1e-6。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_mask</td>
      <td>可选属性</td>
      <td><ul><li>表示输出的掩码，对应公式中的`outputMask`。只支持空指针，或者长度为2的数组。</li><li>默认值为nullptr。</li></ul></td>
      <td>LISTBOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的`y1Out`。</td>
      <td>INT8、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、INT4</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的`y2Out`。</td>
      <td>INT8、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、INT4</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输出</td>
      <td>表示x1和x2的和，对应公式中的`x`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale1</td>
      <td>输出</td>
      <td>第一路量化的输出，对应公式中的`scale1Out`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale2</td>
      <td>输出</td>
      <td>第二路量化的输出，对应公式中的`scale2Out`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Ascend 950PR/Ascend 950DT</term>：
    - 暂不支持入参`beta`和可选属性`output_mask`的配置。
    - 输出参数`y1`、`y2`的数据类型不支持INT4。

  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

    输出参数`y1`、`y2`的数据类型仅支持INT4、INT8。

  - Kirin X90/Kirin 9030处理器系列产品：x1、x2、gamma、smooth_scale1、smooth_scale2、beta和x的数据类型不支持BFLOAT16，y1和y2只支持INT8。

## 约束说明

- 当output_mask不为空时，参数smooth_scale1有值时，则output_mask[0]必须为True。参数smooth_scale2有值时，则output_mask[1]必须为True。

- 当output_mask为空时，参数smooth_scale2有值时，参数smooth_scale1不能为空。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_add_rms_norm_dynamic_quant](examples/test_aclnn_add_rms_norm_dynamic_quant.cpp) | 通过[aclnnAddRmsNormDynamicQuant](aclnnAddRmsNormDynamicQuant.md)接口方式调用AddRmsNormDynamicQuant算子。 |
| aclnn接口  | [test_aclnn_add_rms_norm_dynamic_quant_v2](examples/test_aclnn_add_rms_norm_dynamic_quant_v2.cpp) | 通过[aclnnAddRmsNormDynamicQuantV2](aclnnAddRmsNormDynamicQuantV2.md)接口方式调用AddRmsNormDynamicQuant算子。 |
| 图模式 | -  | 通过[算子IR](../code/op_graph/add_rms_norm_dynamic_quant_proto.h)构图方式调用AddRmsNormDynamicQuant算子。         |
