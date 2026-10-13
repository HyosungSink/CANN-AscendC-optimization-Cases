# QkvRmsNormRopeCache

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

- 算子功能：输入qkv融合张量，通过SplitVD拆分q、k、v张量，执行RmsNorm、ApplyRotaryPosEmb、Quant、Scatter融合操作，输出q_out、k_cache、v_cache、q_out_before_quant(可选)、k_out_before_quant(可选)、v_out_before_quant(可选)。
- 本算子目前支持的场景如下表：

  |场景类型|情况概要|
  |:---|:---|
  |<ul><li>cache_mode为PA_NZ</li><li>q无量化</li><li>k和v支持无量化、对称量化和非对称量化</li><li>q_out_before_quant/k_out_before_quant/v_out_before_quant不输出</li></ul>|qkv Shape为[$B_{qkv}$ \* $S_{qkv}$, $N_{qkv}$ * $D_{qkv}$]，q、k、v具有完全相同的D维度。主要计算过程与输出对应关系：<br><ul><li>qkv 经过SplitVD->q、k、v</li><li>q经过RmsNorm、RoPE->q_out</li><li>k经过RmsNorm、RoPE、Quant(可选)、Scatter->k_cache</li><li>v经过Quant(可选)、Scatter->v_cache</li></ul>|

- 计算公式：

  (1) SplitVD:

  下式中，$N_q$、$N_k$、$N_v$分别表示q、k、v分量的注意力头数量，必须满足：

  $$
  \begin{cases}
  N_k = N_v \\
  N_{qkv} = N_k + N_v + N_q \\
  D_{qkv} = D_q = D_k = D_v
  \end{cases}
  $$

  $$
  \begin{aligned}
  q &= qkv[..., [:N_q] * D_{qkv}] \\
  k &= qkv[..., [N_q:-N_v] * D_{qkv}] \\
  v &= qkv[..., [-N_v:] * D_{qkv}]
  \end{aligned}
  $$

  (2) RmsNorm:

  此处x和y分别表示RmsNorm的输入张量和输出张量，归一化沿最后一维（feature dimension）进行，该计算规则通用于q、k分量。

  $$
  squareX = x * x
  $$

  $$
  meanSquareX = squareX.mean(dim = -1, keepdim = True)
  $$

  $$
  rms = \sqrt{meanSquareX + epsilon}
  $$

  $$
  y = (x / rms) * gamma
  $$

  (3) RoPE (Half-and-Half):

  此处的y指代完成RmsNorm计算的输出结果。

  $$
  y1 = y[\ldots, :d/2]
  $$

  $$
  y2 = y[\ldots, d/2:]
  $$

  $$
  y\_RoPE = torch.cat((-y2, y1), dim = -1)
  $$
  
  $$
  y\_embed = (y * cos) + y\_RoPE * sin
  $$

  (4) Quant:

  无量化：

  $$
  kQuant = kRoPE \\
  vQuant = v
  $$

  对称量化部分：

  $$
  kQuant = kRoPE / kScale \\
  vQuant = v / vScale
  $$

  非对称量化部分：
  
  $$
  kQuant = kRoPE / kScale + kOffset \\
  vQuant = v / vScale + vOffset
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 312px">
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
      <td>qkv</td>
      <td>输入</td>
      <td>用于切分出q、k、v的输入数据，对应公式中的qkv。shape为[B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>qkv</sub> * D<sub>qkv</sub>]。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>q_gamma</td>
      <td>输入</td>
      <td>用于q的rms_norm计算的输入数据，对应公式中的gamma。与输入qkv的数据类型相同，shape为[D<sub>qkv</sub>]。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_gamma</td>
      <td>输入</td>
      <td>用于k的rms_norm计算的输入数据，对应公式中的gamma。与输入qkv的数据类型相同，shape为[D<sub>qkv</sub>]。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>输入</td>
      <td>用于rope计算的输入数据，对输入张量进行余弦变换，对应公式中的cos。与输入qkv的数据类型相同，shape为[B<sub>qkv</sub> * S<sub>qkv</sub>, 1 * D_rope]，D_rope = D<sub>qkv</sub>且要求（D<sub>qkv</sub> * qkv数据类型所占字节数）可以被32整除。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>输入</td>
      <td>用于rope计算的输入数据，对输入张量进行正弦变换，对应公式中的sin。与输入cos的数据类型、格式保持一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>输入</td>
      <td>用于指定写入cache的具体索引位置。shape为[B<sub>qkv</sub> * S<sub>qkv</sub>]。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>q_out</td>
      <td>输入/输出</td>
      <td>提前申请的cache，输入输出同地址复用。与输入qkv的数据类型相同，shape为[B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>q</sub> * D<sub>qkv</sub>]。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_cache</td>
      <td>输入/输出</td>
      <td>提前申请的cache，输入输出同地址复用。与输入qkv的数据类型相同（k不量化），或者INT8（k量化）。shape为[BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 16, BlockSize, 16]（k不量化），或者[BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 32, BlockSize, 32]（k量化）。</td>
      <td>FLOAT16、BFLOAT16、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v_cache</td>
      <td>输入/输出</td>
      <td>提前申请的cache，输入输出同地址复用。与输入qkv的数据类型相同（v不量化），或者INT8（v量化）。shape为[BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 16, BlockSize, 16]（v不量化），或者[BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 32, BlockSize, 32]（v量化）。</td>
      <td>FLOAT16、BFLOAT16、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_scale</td>
      <td>可选输入</td>
      <td>当k_cache数据类型为INT8时需要此输入参数，对应公式中的kScale。shape为[N<sub>k</sub>, D<sub>qkv</sub>]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v_scale</td>
      <td>可选输入</td>
      <td>当v_cache数据类型为INT8时需要此输入参数，对应公式中的vScale。shape为[N<sub>v</sub>, D<sub>qkv</sub>]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_offset</td>
      <td>可选输入</td>
      <td>当k_cache数据类型为INT8且对应的k_scale输入存在并量化场景为非对称量化时，需要此参数输入，对应公式中的kOffset。shape为[N<sub>k</sub>, D<sub>qkv</sub>]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v_offset</td>
      <td>可选输入</td>
      <td>当v_cache数据类型为INT8且对应的v_scale输入存在并量化场景为非对称量化时，需要此参数输入，对应公式中的vOffset。shape为[N<sub>v</sub>, D<sub>qkv</sub>]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>q_out_before_quant</td>
      <td>可选输出</td>
      <td>即将写入到q_out中的数据。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_out_before_quant</td>
      <td>可选输出</td>
      <td>即将写入到k_cache中的数据，在未经量化和Scatter前的中间计算结果。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v_out_before_quant</td>
      <td>可选输出</td>
      <td>即将写入到v_cache中的数据，在未经量化和Scatter前的中间计算结果。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>qkv_size</td>
      <td>属性</td>
      <td>按[B<sub>qkv</sub>, S<sub>qkv</sub>, N<sub>qkv</sub>, D<sub>qkv</sub>]顺序传入，提供输入参数qkv矩阵的B，S，N，D维度具体尺寸。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_nums</td>
      <td>属性</td>
      <td>按[N<sub>q</sub>, N<sub>k</sub>, N<sub>v</sub>]顺序传入，提供输入参数qkv矩阵中，qkv分量单元中分的N维度具体尺寸。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul>
      <li>用于防止RmsNorm计算除0错误，对应公式中的epsilon。</li>
      <li>默认值为1e-6。</li>
      </ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cache_mode</td>
      <td>可选属性</td>
      <td><ul>
      <li>cache格式的选择标记，目前只支持PA_NZ。</li>
      <li>默认值为PA_NZ。</li>
      </ul></td>
      <td>CHAR*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_output_qkv</td>
      <td>可选属性</td>
      <td><ul>
      <li>表示是否需要输出各cache输出中对应内容在未经量化和Scatter前的原始值。</li>
      <li>默认值为false。</li>
      </ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>

  </tbody></table>

## 约束说明

* 输入shape限制：
  * B<sub>qkv</sub>为输入qkv的batch_size，S<sub>qkv</sub>为输入qkv的sequence length，大小由qkvSize决定。
  * N<sub>qkv</sub>为输入qkv的head number。D<sub>qkv</sub>为输入qkv的head dim，目前仅支持128。D<sub>q</sub>、D<sub>k</sub>和D<sub>k</sub>分别为q、k、v的head dim，要求D<sub>qkv</sub> = D<sub>q</sub> = D<sub>k</sub> = D<sub>v</sub>，D<sub>qkv</sub>需要满足（D<sub>qkv</sub>*qkv数据类型占字节数）可以被32整除。
  * 根据rope规则，D<sub>k</sub>和D<sub>q</sub>为偶数。若cache_mode为PA_NZ场景下，D<sub>k</sub>、D<sub>q</sub>需32B对齐；BlockSize需32B对齐。
  * 关于上述32B对齐的情形，对齐值由cache的数据类型决定。以BlockSize为例，若cache的数据类型为int8，则需要满足BlockSize % 32 = 0；若cache的数据类型为float16，则需要满足BlockSize % 16 = 0；若k_cache与v_cache参数的dtype不一致，BlockSize需同时满足BlockSize % 32 = 0和BlockSize % 16 = 0。
  * BlockNum为写入cache的内存块数，大小由用户输入场景决定，要求BlockNum >= Ceil(S<sub>qkv</sub> / BlockSize) * B<sub>qkv</sub>。
  * 使用requireMemory表示存放数据所需的空间大小，需满足：requireMemory >= (B<sub>qkv</sub> \* S<sub>qkv</sub> \* N<sub>qkv</sub> \* D<sub>qkv</sub> + 2 \* D<sub>qkv</sub> + 2 \* B<sub>qkv</sub> \* S<sub>qkv</sub> \* D<sub>qkv</sub> + B<sub>qkv</sub> \* S<sub>qkv</sub> \* N<sub>q</sub> \* D<sub>qkv</sub> + BlockNum \* BlockSize \* N<sub>v</sub> \* D<sub>qkv</sub> + BlockNum \* BlockSize \* N<sub>k</sub> \* D<sub>qkv</sub>) \* sizeof(FLOAT16) + B<sub>qkv</sub> \* S<sub>qkv</sub> \* sizeof(INT64) + (2 \* N<sub>k</sub> \* D<sub>qkv</sub> + 2 \* N<sub>v</sub>) \ sizeof(FLOAT)，当计算出requireMemory的大小超过当前AI处理器的GM空间总大小，不支持使用该算子。
* 其他限制：
  * 对于index，要求index的value值范围为[-1, BlockNum * BlockSize)。value数值不可以重复，index为-1时，代表跳过更新。
  * k_scale, v_scale表示对称量化的缩放因子，因此若传参，则值不能为0。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_qkv_rms_norm_rope_cache](examples/test_aclnn_qkv_rms_norm_rope_cache.cpp) | 通过[aclnnQkvRmsNormRopeCache](aclnnQkvRmsNormRopeCache.md)接口方式调用QkvRmsNormRopeCache算子。 |
