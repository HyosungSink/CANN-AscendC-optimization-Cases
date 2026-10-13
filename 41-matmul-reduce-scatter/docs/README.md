# MatmulReduceScatter

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

**说明：** 使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本，否则将会引发报错，比如BUS ERROR等。

## 功能说明

算子功能：完成mm + reduce_scatter_base计算。

计算公式：

$$
output=reduce\_scatter\_base(x1@x2+bias)
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
 </colgroup>
 <thead>
  <tr>
   <th>参数名</th>
   <th>输入/输出</th>
   <th>描述</th>
   <th>数据类型</th>
   <th>数据格式</th>
  </tr></thead>
 <tbody>
  <tr>
   <td>x1</td>
   <td>输入</td>
   <td>即计算公式中的x1，数据类型与x2保持一致。当前版本仅支持二维输入，且仅支持不转置场景。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x2</td>
   <td>输入</td>
   <td>即计算公式中的x2，数据类型与x1保持一致。支持通过转置构造的非连续的Tensor，当前版本仅支持二维输入。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>bias</td>
   <td>输入</td>
   <td>即计算公式中的bias，支持传入空指针。当前版本仅支持一维输入，且暂不支持bias输入为非0的场景。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>group</td>
   <td>输入</td>
   <td>Host侧标识通信域的字符串，即通信域名称，通过Hccl接口HcclGetCommName获取commName作为该参数。</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>reduceOp</td>
   <td>输入</td>
   <td>Host侧的reduce操作类型，当前版本仅支持“sum”。</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>commTurn</td>
   <td>输入</td>
   <td>Host侧整型，通信数据切分数（总数据量/单次通信量），当前版本仅支持输入0。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>streamMode</td>
   <td>输入</td>
   <td>Host侧整型，流模式的枚举，当前只支持枚举值1。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>output</td>
   <td>输出</td>
   <td>mm计算+reducescatter通信的结果，数据类型与x1保持一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
 </tbody></table>

## 约束说明

- 输入x1为2维，其shape为(m, k)，m须为卡数rank_size的整数倍。
- 输入x2必须是2维，其shape为(k, n)，轴满足mm算子入参要求，k轴相等，且k轴取值范围为[256, 65535)。
- x1/x2支持的空tensor场景，m和n可以为空，k不可为空，且需满足以下条件：
  - m为空，k不为空，n不为空；
  - m不为空，k不为空，n为空；
  - m为空，k不为空，n为空。
- x2矩阵支持转置/不转置场景，x1矩阵只支持不转置场景。
- x1、x2计算输入的数据类型要和output计算输出的数据类型一致。
- bias暂不支持输入为非0的场景。
- 输出为2维，其shape为(m/rank_size, n), rank_size为卡数。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持2、4、8卡，并且仅支持HCCS链路all mesh组网。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持2、4、8、16、32卡，并且仅支持HCCS链路double ring组网。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：一个模型中的通算融合MC2算子，仅支持相同通信域。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_matmul_reduce_scatter.cpp](./examples/test_aclnn_matmul_reduce_scatter.cpp) | 通过[aclnnMatmulReduceScatter](aclnnMatmulReduceScatter.md)接口方式调用matmul_reduce_scatter算子。 |
