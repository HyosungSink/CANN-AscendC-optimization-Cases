# DequantSwigluQuant

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

- 算子功能：在Swish门控线性单元激活函数前后添加dequant和quant操作，实现x的DequantSwigluQuant计算。
- swiglu_mode为0时的计算公式：  

  $$
  dequantOut_i = Dequant(x_i)
  $$

  $$
  swigluOut_i = Swiglu(dequantOut_i)=Swish(A_i)*B_i
  $$

  $$
  out_i = Quant(swigluOut_i)
  $$

  其中，A<sub>i</sub>表示dequantOut<sub>i</sub>的前半部分，B<sub>i</sub>表示dequantOut<sub>i</sub>的后半部分。

- swiglu_mode为1时的计算公式：  

  $$
  dequantOut_i = Dequant(x_i)
  $$

  $$
  x\_glu = x\_glu.clamp(min=None, max=clamp\_limit)
  $$
  
  $$
  x\_linear = x\_linear.clamp(min=-clamp\_limit, max=clamp\_limit)
  $$

  $$
  out\_glu = x\_glu * sigmoid(glu\_alpha * x\_glu)
  $$

  $$
  swigluOut_i = out\_glu * (x\_linear + glu\_bias)
  $$

  $$
  out_i = Quant(swigluOut_i)
  $$

  其中，x\_glu表示dequantOut<sub>i</sub>的偶数索引部分，x\_linear表示dequantOut<sub>i</sub>的奇数索引部分。

## 参数说明

<table style="undefined;table-layout: fixed; width: 951px"><colgroup>
  <col style="width: 221px">
  <col style="width: 144px">
  <col style="width: 213px">
  <col style="width: 257px">
  <col style="width: 116px">
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
      <td>x</td>
      <td>输入</td>
      <td>输入待处理的数据，公式中的x。</td>
      <td>FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>weight_scale</td>
      <td>输入</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>activation_scale</td>
      <td>输入</td>
      <td>激活函数的反量化scale。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>bias</td>
      <td>输入</td>
      <td>Matmul的bias，公式中的bias。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_scale</td>
      <td>输入</td>
      <td>量化的scale，公式中的quant_scale。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>quant_offset</td>
      <td>输入</td>
      <td>量化的offset。</td>
      <td>FLOAT</td>
      <td>ND</td>
     </tr>
      <tr>
      <td>group_index</td>
      <td>输入</td>
      <td>MoE分组需要的group_index。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>activate_left</td>
      <td>属性</td>
      <td>表示是否对输入的左半部分做swiglu激活。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>属性</td>
      <td>表示使用动态量化。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
     <tr>
      <td>dst_type</td>
      <td>属性</td>
      <td>表示指定输出y的数据类型。</td>
      <td>INT64</td>
      <td>-</td>
     </tr>
     <tr>
      <td>round_mode</td>
      <td>属性</td>
      <td>表示对输出y结果的舍入模式。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
      <tr>
      <td>activate_dim</td>
      <td>属性</td>
      <td>表示进行swish计算时，选择的指定切分轴。</td>
      <td>INT64</td>
      <td>-</td>
     </tr>
       <tr>
      <td>swiglu_mode</td>
      <td>属性</td>
      <td>表示swiglu的计算模式。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
      <tr>
      <td>clamp_limit</td>
      <td>属性</td>
      <td>表示变体swiglu使用的门限值。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
      <tr>
      <td>glu_alpha</td>
      <td>属性</td>
      <td>表示变体swiglu使用的参数。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
      <tr>
      <td>glu_bias</td>
      <td>属性</td>
      <td>表示变体swiglu使用的偏差参数。</td>
      <td>FLOAT</td>
      <td>-</td>
     </tr>
       <tr>
      <td>y</td>
      <td>输出</td>
      <td>-</td>
      <td>INT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输出</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 处理器系列产品:
  - 输入`x`：数据类型不支持BFLOAT16。
  - 输入`biasOptional`：数据类型不支持BFLOAT16。
  - 输入`quantScaleOptional`：数据类型不支持FLOAT16。
  - 输出`y`：数据类型仅支持INT8。

## 约束说明

- <term>Ascend 950PR/Ascend 950DT</term>：
  - 输入x对应activate_dim的维度需要是2的倍数，且x的维数必须大于1维。
  - 当输入x的数据类型为INT32时，weight_scale不能为空；当输入x的数据类型不为INT32时，weight_scale不允许输入，传入空指针。
  - 当输入x的数据类型不为INT32时，activation_scale不允许输入，参数置为空指针。
  - 当输入x的数据类型不为INT32时，bias不允许输入，参数置为空指针。
  - 当输出y的数据类型为FLOAT4_E2M1、FLOAT4_E1M2时，y的最后一维需要是2的倍数。
  - 输出y的尾轴不超过5120.

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - swiglu_mode、clamp_limit、glu_alpha和glu_bias四个参数用于GPT-OSS变体SwiGLU的使用。
  - x的最后一维需要是2的倍数，且x的维数必须大于1维。
  - 当quant_mode为static时，quant_scale和quant_offset为1维，值为1；quant_mode为dynamic时，quant_scale和quant_offset
  - 算子支持的输入张量的内存大小有上限，校验公式：weight_scale张量内存大小+bias张量内存大小+quant_scale张量内存大小+quant_offset张量内存大小 + （activation_scale张量内存大小 + scale张量内存大小）/40  + x张量最后一维H内存大小 * 10 < 192KB。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_dequant_swiglu_quant](./examples/test_aclnn_dequant_swiglu_quant.cpp) | 通过[aclnnDequantSwigluQuant](aclnnDequantSwigluQuant.md)接口方式调用DequantSwigluQuant算子。    |
| aclnn调用 | [test_aclnn_dequant_swiglu_quant_v2](./examples/test_aclnn_dequant_swiglu_quant_v2.cpp) | 通过[aclnnDequantSwigluQuantV2](aclnnDequantSwigluQuantV2.md)接口方式调用DequantSwigluQuant算子。    |
| 图模式调用 | -   | 通过[算子IR](../code/op_graph/dequant_swiglu_quant_proto.h)构图方式调用DequantSwigluQuant算子。 |
