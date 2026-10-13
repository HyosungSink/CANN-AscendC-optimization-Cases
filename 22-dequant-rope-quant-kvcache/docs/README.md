# DequantRopeQuantKvcache

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | √ |
|  <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：对输入张量（x）进行dequant（可选）后，按`sizeSplits`（为切分的长度）对尾轴进行切分，划分为q、k、vOut，对q、k进行旋转位置编码，生成qOut和kOut，之后对kOut和vOut进行量化并按照`indices`更新到kCacheRef和vCacheRef上。

- 计算公式：
  
  $$
  dequantX = Dequant(x,weightScaleOptional,activationScaleOptional,biasOptional)
  $$
  
  $$
  q,k,vOut = SplitTensor(dequantX,dim=-1,`sizeSplits`)
  $$
  
  $$
  qOut,kOut = ApplyRotaryPosEmb(q,k,cos,sin)
  $$
  
  $$
  quantK = Quant(kOut,scaleK,offsetKOptional)
  $$
  
  $$
  quantV = Quant(vOut,scaleV,offsetVOptional)
  $$
  
  如果cacheModeOptional为contiguous则：
  
  $$
  kCacheRef[i][indices[i]]=quantK[i]
  $$
  
  $$
  vCacheRef[i][indices[i]]=quantV[i]
  $$
  
  如果cacheModeOptional为page则：
  
  $$
  kCacheRefView=kCacheRef.view(-1,kCacheRef[-2],kCacheRef[-1])
  $$
  
  $$
  vCacheRefView=vCacheRef.view(-1,vCacheRef[-2],vCacheRef[-1])
  $$
  
  $$
  kCacheRefView[indices[i]]=quantK[i]
  $$
  
  $$
  vCacheRefView[indices[i]]=quantV[i]
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
    </tr></thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap">x</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中的用于切分的输入`x`，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT16、INT32、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">cos</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中的用于位置编码的输入`cos`，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT16、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">sin</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中的用于位置编码的输入`sin`，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">BFLOAT16、FLOAT16、FLOAT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kCacheRef</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中用于缓存k的输入`kCacheRef`，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">INT8</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">vCacheRef</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中用于缓存v的输入`vCacheRef`，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">INT8</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">indices</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中表示Kvcache的token位置信息的输入`indices`。</td>
      <td style="white-space: nowrap">INT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">scaleK</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中的输入`scaleK`用于量化`k`的scale因子，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">scaleV</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中的输入`scaleV`用于量化`v`的scale因子，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">offsetKOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">公式中的输入`offsetKOptional`用于量化k的offset因子，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">offsetVOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">公式中的输入`offsetVoptional`用于量化的offset因子，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">weightScaleOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">公式中的输入`weightScaleoptional`用于反量化的权重scale因子，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">activationScaleOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">公式中的输入`activationScaleOptional`用于反量化的激活scale因子，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">biasOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">公式中的输入用于反量化的偏置`biasOptional`，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT、FLOAT16(HALF)、INT32、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">sizeSplits</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">表示输入的qkv进行切分的长度。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">quantModeOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">Host侧表达式字符串。表示支持的量化类型，目前仅支持static。</td>
      <td style="white-space: nowrap">String</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">layoutOptional</td>
      <td style="white-space: nowrap">可选输入</td>
      <td style="white-space: nowrap">Host侧表达式字符串。表示支持的数据格式，目前仅支持BSND。</td>
      <td style="white-space: nowrap">String</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kvOutput</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">Host侧表达式布尔值。表示是否输出kOut和vOut。</td>
      <td style="white-space: nowrap">BOOL</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">cacheModeOptional</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">Host侧表达式字符串。表示kCacheRef的更新方式，目前仅支持page和contiguous，默认为contiguous。</td>
      <td style="white-space: nowrap">String</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">qOut</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">表示经过处理的q，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT16、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kOut</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">表示输入的qkv进行切分的长度。</td>
      <td style="white-space: nowrap">FLOAT16、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">vOut</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">表示经过处理的v，Device侧的aclTensor。</td>
      <td style="white-space: nowrap">FLOAT16、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
  </tbody></table>

## 约束说明

- cacheModeOptional为contiguous时：kCacheRef的第0维大于x的第0维，indices数据值大于等于0且小于等于vCacheRef的第1维([b，s，n，d]格式中的s)减x的第1维；cacheModeOptional为page时：indices 数据值大于等于0，小于kCacheRef的第0维*第1维且不重复。
- x的尾轴小于等于4096，且按64对齐。
- 输入x不为int32时，x、cos、sin与输出qOut、kOut、vOut的数据类型保持一致，此时activationScaleOptional，weightScaleOptional、biasOptional不生效；x为int32时，cos、sin与输出qOut、kOut、vOut的数据类型保持一致，此时weightScaleOptional必选，activationScaleOptional、biasOptional可选（biasOptional不需要与其他输入类型一致）。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_dequant_rope_quant_kvcache](examples/test_aclnn_dequant_rope_quant_kvcache.cpp) | 通过接口方式调用[DequantRopeQuantKvcache](aclnnDequantRopeQuantKvcache.md)算子。 |
