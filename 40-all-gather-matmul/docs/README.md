# AllGatherMatmul

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

- 算子功能：完成AllGather通信与MatMul计算融合。
- 计算公式：

    $$
    y=AllGather(x1)@x2+bias
    $$

    $$
    gatherOut=AllGather(x1)
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
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>公式中的输入x1。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入x2。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>可选输入</td>
      <td>公式中的输入bias。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出y。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gather_out</td>
      <td>输出</td>
      <td>公式中的输出gatherOut。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group</td>
      <td>属性</td>
      <td><li>Host侧标识通信域的字符串，通信域名称。</li><li>通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取，其中commName即为group。</li></td>
      <td>CHAR*、STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_trans_a</td>
      <td>可选属性</td>
      <td><li>决定x1是否执行矩阵乘前进行转置。</li><li>默认值为false。</li></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_trans_b</td>
      <td>可选属性</td>
      <td><li>决定x2是否执行矩阵乘前进行转置。</li><li>默认值为false。</li></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gather_index</td>
      <td>可选属性</td>
      <td><li>标识gather目标，当前版本仅支持输入0，表示目标为x1。</li><li>默认值为0。</li></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>commTurn</td>
      <td>可选属性</td>
      <td><li>通信数据切分数，即总数据量/单次通信量。</li><li>默认值为0。</li></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rank_size</td>
      <td>可选属性</td>
      <td><li>通信域里面的卡数。</li><li>默认值为0。</li></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_gather_out</td>
      <td>可选属性</td>
      <td><li>若为true，输出gather_out。</li><li>默认值为true。</li></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

* 当前版本中，输入x1为2维，其shape为(m, k)。x2必须是2维，其shape为(k, n)，轴满足MM算子入参要求，k轴相等，且k轴取值范围为[256, 65535)。
* x1/x2支持的空tensor场景，m和n可以为空，k不可为空，且需要满足以下条件：
    * m为空，k不为空，n不为空；
    * m不为空，k不为空，n为空；
    * m为空，k不为空，n为空。
* x1计算输入、x2计算输入、output计算输出、gather_out计算输出的数据类型均需保持一致。
* x2矩阵支持转置/不转置场景，x2矩阵支持通过转置构造的非连续的Tensor，x1矩阵只支持不转置场景。
* bias可选可为空，非空时，当前版本仅支持一维输入，且暂不支持bias输入为非0的场景。
* 输出为2维，其shape为(m*rank_size, n), rank_size为卡数。
* gather_index当前版本仅支持输入0。
* commTurn当前版本仅支持输入0。
* <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    * 支持2、4、8卡，并且仅支持HCCS链路all mesh组网。
    * 一个模型中的通算融合MC2算子，仅支持相同通信域。
* <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    * 支持2、4、8、16、32卡，并且仅支持HCCS链路double ring组网。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_all_gather_matmul](./examples/test_aclnn_all_gather_matmul.cpp) | 通过[aclnnAllGatherMatmul](aclnnAllGatherMatmul.md)接口方式调用AllGatherMatmul算子。 |
