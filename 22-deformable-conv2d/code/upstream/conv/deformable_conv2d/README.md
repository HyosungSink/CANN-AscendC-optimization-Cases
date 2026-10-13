# DeformableConv2d

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：实现卷积功能，支持2D卷积，同时支持可变形卷积、分组卷积。

- 计算公式：
  
  假定输入（input）的shape是[N, inH, inW, inC]，输出的（out）的shape为[N, outH, outC, outW]，根据已有参数计算outH、outW:
  
  $$
  outH = (inH + padding[0] + padding[1] - ((K_H - 1) * dilation[2] + 1)) // stride[2] + 1
  $$
  
  $$
  outW = (inW + padding[2] + padding[3] - ((K_W - 1) * dilation[3] + 1)) // stride[3] + 1
  $$
  
  标准卷积计算采样点下标：
  
  $$
  x = -padding[2] + ow*stride[3] + kw*dilation[3], kw的取值为(0, K\_W-1)
  $$
  
  $$
  y = -padding[0] + oh*stride[2] + kh*dilation[2], kh的取值为(0, K\_H-1)
  $$
  
  根据传入的offset，进行变形卷积，计算偏移后的下标：
  
  $$
  (x,y) = (x + offsetX, y + offsetY)
  $$

  使用双线性插值计算偏移后点的值：
  
  $$
  (x_{0}, y_{0}) = (int(x), int(y)) \\
  (x_{1}, y_{1}) = (x_{0} + 1, y_{0} + 1)
  $$
  
  $$
  weight_{00} = (x_{1} - x) * (y_{1} - y) \\
  weight_{01} = (x_{1} - x) * (y - y_{0}) \\ 
  weight_{10} = (x - x_{0}) * (y_{1} - y) \\ 
  weight_{11} = (x - x_{0}) * (y - y_{0}) \\ 
  $$
  
  $$
  deformOut(x, y) = weight_{00} * input(x0, y0) + weight_{01} * input(x0,y1) + weight_{10} * input(x1, y0) + weight_{11} * input(x1,y1)
  $$
  
  进行卷积计算得到最终输出：
  
  $$
  \text{out}(N_i, C_{\text{out}_j}) = \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{deformOut}(N_i, k)
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
      <td>x</td>
      <td>输入</td>
      <td>表示输入的原始数据。对应公式中的`input`。shape为[N, inH, inW, inC]，其中inH * inW不能超过2147483647。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
     </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>表示可学习过滤器的4D张量，对应公式中的`weight`。数据类型、数据格式与入参`x`保持一致。shape为[outC, K_H, K_W, inC/groups]。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>输入</td>
      <td>表示x-y坐标偏移和掩码的4D张量。对应公式中的`offset`。数据类型、数据格式与入参`x`保持一致。当`modulated`为true时，shape为[N, outH, outW, 3 * deformable_groups * K_H * K_W]；当`modulated`为false时，shape为[N, outH, outW, 2 * deformable_groups * K_H * K_W]。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>可选输入</td>
      <td>表示过滤器输出附加偏置的1D张量，预留参数。数据类型与入参`x`一致。不需要时为空指针，存在时shape为[outC]。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kernel_size</td>
      <td>可选属性</td>
      <td>表示卷积核大小，对应公式中的`K_H`、`K_W`。size为2(K_H, K_W)，各元素均大于零，K_H * K_W不能超过2048，K_H * K_W * inC/groups不能超过65535。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>可选属性</td>
      <td>表示每个输入维度的滑动窗口步长，对应公式中的`stride`。size为4，各元素均大于零，维度顺序根据`x`的数据格式解释。N维和C维必须设置为1。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>可选属性</td>
      <td>表示要添加到输入每侧（顶部、底部、左侧、右侧）的像素数，对应公式中的`padding`。size为4。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>属性</td>
      <td><ul><li>表示输入每个维度的膨胀系数，对应公式中的`dilation`。size为4，各元素均大于零，维度顺序根据x的数据格式解释。N维和C维必须设置为1。</li><li>默认值为{1, 1, 1, 1}。</li></ul></td>
      <td>LISTINT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groups</td>
      <td>属性</td>
      <td><ul><li>表示从输入通道到输出通道的分组连接数。inC和outC需都可被`groups`数整除，`groups`数大于零。</li><li>默认值为1。</li></ul></td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deformable_groups</td>
      <td>属性</td>
      <td><ul><li>表示可变形组分区的数量。inC需可被`deformable_groups`数整除，`deformable_groups`的值大于零。</li><li>默认值为1。</li></ul></td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>modulated</td>
      <td>属性</td>
      <td><ul><li>表示`offset`中是否包含掩码。若为true，`offset`中包含掩码；若为false，则不包含。</li><li>默认值为true，且当前仅支持配置为true。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示计算的输出张量。对应公式中的`out`。数据类型、数据格式与`x保持一致。shape为[N, outH, outC, outW]。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>deform_out</td>
      <td>输出</td>
      <td>表示计算的输出张量。对应公式中的`out`。数据类型、数据格式与`x保持一致。shape为[N, outH * K_H, outW * K_W, inC]。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_deformable_conv2d](examples/test_aclnn_deformable_conv2d.cpp) | 通过[aclnnDeformableConv2d](docs/aclnnDeformableConv2d.md)接口方式调用DeformableConv2d算子。 |
