# LayerNormV4

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×   |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | √ |
|  <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：

  对指定层进行均值为0、标准差为1的归一化计算。
  - 归一化：对输入张量的每个样本进行归一化处理，使得每个样本的均值为0，方差为1。
  - 缩放和偏移：在归一化之后，可以通过缩放因子和偏移量进一步调整归一化后的输出，以适应不同的模型需求。

- 计算公式：

  $$
  mean = {E}[x]
  $$

  $$
  rstd = \frac{1}{ \sqrt{\mathrm{Var}[x] + eps}} 
  $$

  $$
  y = w*((x - mean) * rstd) + b
  $$

  其中，E[x]表示输入的均值，Var[x]表示输入的方差。

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
      <td>x</td>
      <td>输入</td>
      <td>表示进行归一化计算的输入，对应公式中的`x`。shape为[A1,...,Ai,R1,...,Rj]，其中A1至Ai表示无需norm的维度，R1至Rj表示需norm的维度。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normalized_shape</td>
      <td>输入</td>
      <td>表示需要进行norm计算的维度。值为[R1,...,Rj]，长度小于等于输入x的shape长度，不支持为空。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>可选输入</td>
      <td>表示进行归一化计算的权重，公式中的输入`w`。gamma非空时，数据类型与输入x一致或为FLOAT类型，且当beta存在时gamma与beta的数据类型相同。shape与normalized_shape相等，为[R1,...,Rj]。gamma为空时，接口内部会构造一个shape为[R1,...,Rj]，数据全为1的tensor，当beta存在时gamma与beta的数据类型相同，beta不存在时gamma与输入x的数据类型相同。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>可选输入</td>
      <td>表示进行归一化计算的偏移量，公式中的输入`b`。beta非空时，数据类型与输入x一致或为FLOAT类型，且当gamma存在时beta与gamma的数据类型相同。shape与normalized_shape相等，为[R1,...,Rj]。beta为空时，接口内部会构造一个shape为[R1,...,Rj]，数据全为0的tensor，当gamma存在时beta与gamma的数据类型相同，gamma不存在时beta与输入x的数据类型相同。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>添加到分母中的值，以确保数值稳定，对应公式中的eps。</li><li>默认值为0.00001f。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示进行归一化计算的结果，公式中的输出`y`。shape需要与x的shape相等，为[A1,...,Ai,R1,...,Rj]。数据类型与x的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输出</td>
      <td>表示进行归一化后的均值，对应公式中的`mean`。与rstd的shape相同，shape为[A1,...,Ai,1,...,1]，Ai后共有j个1，与需要norm的轴长度保持相同。数据类型与x的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输出</td>
      <td>表示进行归一化后的标准差倒数，对应公式中的`rstd`。与mean的shape相同，shape为[A1,...,Ai,1,...,1]，Ai后共有j个1，与需要norm的轴长度保持相同。数据类型与x的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas 推理系列产品</term>：输入参数（除了normalized_shape）和所有输出参数的数据类型不支持BFLOAT16。
  - Kirin X90/Kirin 9030处理器系列产品：
    - 所有输入、输出的数据类型不支持BFLOAT16。
    - normalized_shape的数据类型不支持INT64。
    - mean和rstd的数据类型不支持FLOAT16、BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_fast_layer_norm](examples/test_aclnn_fast_layer_norm.cpp) | 通过[aclnnFastLayerNorm](docs/aclnnFastLayerNorm.md)接口方式调用LayerNormV4算子。 |
| aclnn接口  | [test_aclnn_layer_norm_v4](examples/test_aclnn_layer_norm_v4.cpp) | 通过[aclnnLayerNorm](docs/aclnnLayerNorm&aclnnLayerNormWithImplMode.md)接口方式调用LayerNormV4算子。 |
| aclnn接口  | [test_aclnn_layer_norm_with_impl_mode](examples/test_aclnn_layer_norm_with_impl_mode.cpp) | 通过[aclnnLayerNormWithImplMode](docs/aclnnLayerNorm&aclnnLayerNormWithImplMode.md)接口方式调用LayerNormV4算子。 |
| 图模式 | - | 通过[算子IR](op_graph/layer_norm_v4_proto.h)构图方式调用LayerNormV4算子。         |
