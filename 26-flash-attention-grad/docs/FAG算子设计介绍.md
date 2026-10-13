## 1 计算过程

图1 训练计算流程图

![流程图](./fig/FlashAttentionScoreGrad.png)

按照FlashAttention反向计算流程实现，整体计算流程如下：

1. 重计算p，$p = SimpledSoftmax(Mask(Matmul(query, key^T) + pse\_shift) * scale)$，本步骤重计算了fa流程中的softmax结果p，计算结果保存的ub中。
2. 计算dp，$dp = Dropout(Matmul(dy, value^T))$，该计算包含matmul计算和dropout计算，matmul计算中，左矩阵为dy，右矩阵为转置后的value。
3. 计算ds，$ds = p * Sub(dp, FlashSoftmaxGrad(dy, attention\_in))$，本计算中，FlashSoftmaxGrad计算的入参为dy、正向输出attention\_in，该结果与dp做减操作，最终的结果与p相乘得到结果ds。
4. 计算dq，$dq = Matmul(ds, key) * scale$，本计算将ds结果与key做matmul计算，并将结果与scale相乘得到结果dq。
5. 计算dk，$dk = Matmul(ds^T, query) * scale$，本计算将转置后的ds结果与query做matmul计算，并将结果与scale相乘得到结果dk。
6. 计算dv，$dv = Matmul(DropOut(p)^T, dy)$；本计算p的结果做drop计算，转置后与dy做matmul计算。

## 2 每个计算阶段的入口

以sameAB模板为例

1. 得到每个核处理的任务块数后，调用函数：`Process`
2. Cube1阶段，流程中1，计算入口：`ComputeMM1`
3. Vector1阶段，流程中2-3，计算入口：`ComputeVec`
4. Cube2阶段，流程中4-6，计算入口：`ComputeMMDqkv`

## 3 Tiling设计

Tiling操作的目的是为了找到一种更高效的NPU执行方式，原始的数据量一般是非常大的，没有办法通过一次指令调用就完成所有计算，因此需要将数据量分到多个核上并行计算，且每个核上也需要考虑如何循环计算性能最优，不同的输入可能有不同的最优执行方式，所以需要通过tiling策略决定怎么将数据分配到各个核上进行计算。根据硬件架构特征，AI Core分成AIC和AIV两个独立的核，AIC和AIV核拥有自己独立的Scalar计算单元，能够独立加载自己的代码段，单独执行。 

### 3.1 <term>Atlas A2 训练系列产品</term>

<term>Atlas A2 训练系列产品</term>芯片AIC和AIV分离的架构可以使得AIC和AIV并行执行。AIC和AIV之间数据交互的通路是L2和GM（Global Memory，高带宽存储器），两者之间的交互次数对性能影响是比较大的，同时由于AIC和AIV算力差异，两者需要使用不同的基本块大小，本着尽量减少AIC和AIV通信次数和发挥最大算力的原则，CVtiling分离策略应运而生，可以有效地减少CV通信次数，同时根据不同单元的buffer特征，选择不同的基本块进行计算，从而提升算子性能。

 对于FAG算子，Vector计算涉及多个输入、输出、中间计算结果、double-buffer设计等，需要将buffer分配成多份，最优分配方案中最大一份为32KB，由于Vector计算使用的数据类型是float32，因此Vector的tiling基本块为8 \* 1024。为了充分发挥Cube的算力，在CV之间一轮计算的数据量进行了1:16的配比，又由于Cube侧的输入数据类型是float16，输出是float32，Cube的基本块为128 \* 128，所以通过nRatio=8配比出128 \* 1024的数据量。伪代码如下：

```c++
// C-Tiling: (S1_c_i,D)x(D,S2_c_i) => (S1_c_i, S2_c_i):(128,1024)
// V-Tiling: (S1_v_i, S2_v_i) => (8,1024)

// C侧 matmul计算
Bmm((S1_c_i,D)x(D,S2_c_i)) => 128*1024  // 输出结果128*1024，放到workspace上 
// V侧 Vector计算
for S1_c_i/S1_v_i=128/8:
  copy_gm_to_ub(S1_v_i*S2_v_i)  // 从bmm的workspace上拷入bmm结果数据
  Vector(S1_v_i,S2_v_i)         // 进行Vector计算
  copy_ub_to_gm(S1_v_i*S2_v_i)  // Vector计算结束，得到最终输出数据，拷贝到GM上

// 由于Cube侧计算数据比Vector侧大，因此，ub内需要再次进行Vector Tiling，从而产生了S1方向的配比：S1_c_i/S1_v_i
```

上述示例中，仅在S1方向开了配比，S2方向C/V计算的长度是一致的，当然，也可以在S1/S2方向均开启配比；这样做的好处是，Cube一次可以发射大块的数据，避免因为小块数据不断发射带来的通信开销，也能最大程度地使用Cube单元的buffer。

### 3.2 **<term>Ascend 950PR/Ascend 950DT</term>**

Ascend 950PR/Ascend 950DT同样是AIC和AIV分离的架构，保留了AIC AIV并行执行特性，同时新增了AIC和AIV之间的高速数据交互通路L0C->UB和UB->L1。降低了CV之间交互的成本，流水并行度更高，且相比于<term>Atlas A2 训练系列产品</term>复杂的tiling切块策略，在950上仅使用一种基本块就可以获得较好的性能。

 对于FAG算子，Vector计算涉及多个输入、输出、中间计算结果、double-buffer设计等，需要将buffer分配成多份，最优分配方案中最大一份为32KB，由于Vector计算使用的数据类型是float32，因此Vector的tiling基本块为64 \* 128，由于Cube与Vector核数为1：2的数量比，为了充分利用cube核的算力，Cube侧考虑采用128 \* 128的基本块，即每个cube核计算完128 \* 128的数据后，均分给两个vector核处理。伪代码如下：

```c++
// C-Tiling: (S1_i,D)x(D,S2_i) => (S1_i, S2_i):(128,128)
// V-Tiling: (S1_i / 2, S2_i) => (64,128)

// C侧 matmul计算
Bmm((S1_i,D)x(D,S2_i)) => 128*128  // value@dy,q@k输出结果128*128，放到L0C上
fixp_l0c_to_ub(S1_i / 2,S2_i); => v0: 64*128, v1: 64*128 // 通过CV通路L0C->UB，均分写到两个V核的UB上
// V侧 Vector计算
Vector(S1_i / 2,S2_i)       // 进行Vector计算
copy_ub_to_l1(S1_i/2*S2_i)  // Vector计算结束，得到输出数据，通过CV通路UB->L1，两个v核数据聚合拷贝到L1上
Bmm((S1_i,S2_i)*(S2_i,D));	// dq，最终结果输出到workspace
Bmm((S2_i,S1_i)*(S1_i,D));  // dkv，最终结果输出到workspace
```

上述示例中，通过两个拷贝指令fixp_l0c_to_ub和copy_ub_to_l1分别将cube侧计算的数据和vector侧计算的数据copy到vector侧和cube侧，这两个拷贝都是核内高速通道。

## 4 流水设计

为了追求极致性能，必须充分利用硬件资源，通常需要进行不同pipeline的流水设计。流水设计的宗旨是尽量使某一条pipeline达成bound效果，使硬件的某一个单元一直在工作，达到性能上限。

### 4.1 V侧流水

V侧流水设计需要考虑Vector的搬运及计算过程，实施的优化手段主要是double buffer。
以下面的流水任务示意图为例，Vec的功能被拆分成2个流水任务：subA、subB，每个任务专注于完成单一功能；需要处理的数据被切分成2片，使用ping-pong表示两个数据处理任务，每个任务需要依次搬运DataCopy与计算Clc（Clc表示Vector计算）操作。任务间的箭头表示数据间的依赖关系，比如subA处理完DataCopy之后，subB才能对Clc进行处理。
从图上可以看出，不进行流水设计时，搬运与计算任务之间是串行执行的，会出现断流现象，即第一次DataCopy完成之后的搬运流水就一直处于空闲状态，直到第一次搬入的数据计算完成并搬出之后搬运流水才会继续工作，进行第二次DataCopy（Vector计算和搬出流水也存在同样问题）。通常这种情况下，性能是极差的。

![设计图1](./fig/设计图1.png)

将上图的流水任务做ping-pong流水间的double buffer处理后，流水任务运行起来的示意图如下，从运行图中可以看出，对于同一片数据，搬运DataCopy与计算Clc之间的处理具有依赖关系，需要串行处理；不同的数据切片，同一时间点，可以有多个任务在并行处理，由此达到任务并行、提升性能的目的。

![设计图2](./fig/设计图2.png)

其中ping、pong两块计算数据所占用的内存资源均相互独立。
FAG融合算子V侧计算过程较多，情况也比较复杂，通常简单的double buffer是无法覆盖所有情况的，因此会出现不同的计算流水排布。不同的计算流水适用于不同类的shape特征，以达到在该类特征下最好的流水设计。

### 4.2 CV流水

融合算子通常包含了Vector计算和Cube计算，对于FA算子，V侧的计算是依赖C侧的计算结果的，如果只关注V侧流水，不关注C侧，则C侧与V侧很有可能是串行流水的效果，不能达到并行计算的目的，无法使得融合算子性能达到最优，从而有了CV流水设计。此外，CV流水在不同算子情况下，表现的现象也是不一致的，FA/FAG的Cube双发机制，又称为CV间preload流水，可实现两种场景下的流水优化：

- C侧总耗时 > V侧总耗时

  该场景流水特征下，Vector计算节点少，计算速度快，在<term>Atlas A2 训练系列产品</term> C:V=1:2的情况下，Cube的搬运时长足以覆盖Vector的计算时长，因此只要关注Cube的MTE2耗时即可，最终达成MTE2 bound。在Cube双发机制下，提前发射两块Cube计算，Cube1、Cube2计算可以衔接，使得Cube利用率最高，达成Cube bound。

![设计图3](./fig/设计图3.png)

- C侧总耗时 < V侧总耗时

  该场景流水特征下，Vector计算节点多，Vector计算是瓶颈，C侧的搬运不足以覆盖V侧的流水，因此需要进行CV流水排布，尽量达到CV并行的效果，最通用的优化手段是C侧提前发射。

![设计图4](./fig/设计图4.png)

C侧连续发射两块Cube计算，这样可以保证V侧计算完上一轮时，可以立马启动当前轮的计算，而不用等待Cube1的数据。这样可以使V侧一直在工作，达成Vector bound。

## 5 多模板设计

为了使不同的输入可以复用相同的tiling和流水，采用了模板的方式来实现融合算子，但是不同的输入全部使用同一套模板时又无法达到性能最优和功能泛化，因此需要根据输入shape的特征区分不同的模板来实现。
FAG（FlashAttentionScoreGrad，简称FAG）融合算子的多模板设计思路主要为：

- **根据核内及核间切分进行模板拆分**

  由于硬件buffer大小是有限的，而计算的数据量又是巨大的，无法一次计算完，那么就需要进行tiling切分，shape不同会导致算子的切分轴不同，而算子的切分轴，会影响模板的功能及性能。简单的elewise类算子，往往会将所有的轴fuse成一根轴进行切分，逻辑简单，因此模板也比较单一。而融合算子融合了elewise、broadcast、reduce及matmul等多类场景，功能复杂，为达到较高的性能要求，往往需要根据切分轴进行模板拆分，模板拆分时为了达到性能最优，需要考虑如下几个点：

  - a. 将核心的数量用满，防止部分核闲置

  - b. 每一个核心被分配的计算量相对均匀，避免出现某些核计算的数据量过大，其余核在围观的情况。

  - c. AIC和AIV之间处理的数据量要符合其对应的算力，避免AIC或AIV出现长时间的空闲。

  FAG算子包含B、N2(key和value的N)、G(query_N/kv_N)、S1(query的S)、S2(key和value的S)共5个轴，切分顺序是先核内再核间，核内切分依据基本块大小选择切分轴，核间切分是把核内切分后剩余的轴合并后依据AI Core核数再进行切分。由于shape的大小不同，切分轴会发生变化，从Vector视角，FAG算子划分为如下几类模板，模板按序号排优先级，序号越小，优先级越高，越先匹配。

- **FAG算子根据适用范围不同，划分成以下几类模板：**

**<term>Atlas A2 训练系列产品</term>**

<table style="undefined;table-layout: fixed; width: 1576px">
<colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
</colgroup>
<thead>
  <tr>
    <th>模板</th>
    <th>切分轴</th>
    <th>走入条件</th>
    <th>适用范围</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>B模板</td>
    <td>多核切B</td>
    <td>非FP32，bestBasicBlockNum = S1 >= 4 ? 64 * 128 / 4 * 3 : 64 * 128;
N1 * G * alignedS1 * alignedS2 <= bestBasicBlockNum。 </td>
    <td>普通场景</td>
  </tr>
  <tr>
    <td>N2模板</td>
    <td>多核切N2</td>
    <td>非FP32，S1 * S2 < 64 * 128。 </td>
    <td>普通场景</td>
  </tr>
  <tr>
    <td>SameAB模板</td>
    <td>UB切S1S2多核切S2</td>
    <td>确定性计算场景，非FP32或者S1 >= 1024 or S2 >= 1024，走SameAB模板；其余走1.2模板。<br>
        TND场景，平均S1 >= 1024 and 平均S2 >= 1024，走SameAB模板；其余走1.1模板。<br>
        普通场景，S1 >= 1024 or S2 >= 1024。
    </td>
    <td>确定性计算场景<br>TND场景<br>普通场景</td>
  </tr>
  <tr>
    <td>S1S2模板</td>
    <td>UB切S1S2多核切N2</td>
    <td>确定性计算场景，不满足SameAB模板条件。<br>
        普通场景，不满足其他模板条件。
    </td>
    <td>确定性计算场景<br>普通场景</td>
  </tr>
  <tr>
    <td>TND模板</td>
    <td>UB切S1S2多核切S2</td>
    <td>不满足SameAB模板条件。</td>
    <td>TND场景</td>
  </tr>
</tbody>
</table>

**Ascend 950PR/Ascend 950DT** 

<table style="undefined;table-layout: fixed; width: 1576px">
<colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
</colgroup>
<thead>
  <tr>
    <th>模板</th>
    <th>切分轴</th>
    <th>走入条件</th>
    <th>适用范围</th>
  </tr>
</thead>
<tbody>
 <tr>
    <td>BN2</td>
    <td>多核切BN2</td>
    <td>S1<=128 and S2 <= 128 and G=1，非FP32。</td>
    <td>普通场景</td>
  </tr>
  <tr>
    <td>BN2S2模板</td>
    <td>多核切BN2S2</td>
    <td>非FP32，TND，B * N2 * S2 > cubeCoreNum and  G = 1。 </td>
    <td>普通场景</td>
  </tr>
  <tr>
    <td>确定性计算模板</td>
    <td>多核切BN2GS1S2</td>
    <td>开启确定性计算</td>
    <td>确定性计算场景</td>
  </tr>
  <tr>
    <td>BN2GS1S2模板</td>
    <td>多核切BN2GS1S2</td>
    <td>上述场景都不满足</td>
    <td>普通场景</td>
  </tr>
</tbody>
</table>

每一类模板都有其独特的UB及Block切分轴，能处理某一类具备特定shape特征输入的场景，针对该类shape特征进行模板设计。

- **根据特殊场景及特定优化进行模板特化**

  每一个融合算子有一个基础模板，并辅以多个特化模板：

    - 基础模板覆盖功能以及大部分此类shape特征输入的性能。
    - 特化模板是覆盖特定场景的极致性能，主要根据某些适用于特定场景的特殊手段进行的优化，不适合进行泛化。

  例如，根据特殊场景空tensor特化而出的empty_input模板，基于角色的Cube核管理（RCM）优化方案设计的确定性计算模板等。

- **根据不同计算流水进行模板特化**

  为了充分发挥硬件优势，通常融合算子都需要进行流水设计，以提高融合算子性能，不同的流水设计对代码的架构影响非常大，为了提升代码的可维可测可读性，需要根据不同的计算流水进行模板特化，达到特定场景的极致性能。

### 5.1 多模板详细设计

- **基本概念：**

  **CV基本块:** 表示Cube或者Vector单次计算的数据量大小，用来描述一次完整的Cube和Vector交互的数据量，通常也等价于**核间基本块**。在芯片上由于Cube和Vector之间通信有一定开销，所以CV基本块设置的比较大，一般合适的数据量大小是512*1024（单位Bytes）。又由于Cube和Vector的核内Buffer有限，所以核间基本块可能要通过**多次核内计算**完成。

  **核内基本块：**

  如果CV基本块过大，Cube和Vector核内会将CV基本块进一步的切分，切分成适合核内L0A、L0B、L0C、UB等大小的基本块，这个就叫**核内基本块**；对于Cube侧，核内基本块一般在32KB（单位Bytes），这样可以让L0A、L0B的DoubleBuffer能力展开，同时算力和带宽也能尽可能用满；在Vector侧一般基本块大小是32KB（单位Bytes）。

  **xxx.i:** 表示经过切分后，CV基本块中的某根轴的大小, 一般基本块都是2维的，xxx.i 表示其中一个维度的大小。xxx可以是B、N2、G、S1、S2;
  例如，S1 = 512, S2 = 1024， S1.i = 64, S2.i = 128, 表示把[S1, S2]切分成大小是[64, 128]的基本块，S1轴的基本块大小是64，S2轴的基本块大小是128。

  **xxx.o**: 表示经过基本块切分后某根轴的分数，xxx可以是B、N2、G、S1、S2;

  例如，S1 = 512, S2 = 1024， S1.i = 64, S2.i = 128, 表示把[S1, S2]切分成大小是[64, 128]的基本块，S1.o = 512 / 64 = 8, S2.o = 1024 / 128 = 8 ，一共切分成8 * 8个基本块。

  - **<term>Atlas A2 训练系列产品</term>**  

  FAG算子的模板划分如下，以下模板，序号越大，模板的优先级越高，序号1的模板是泛化模板（支持所有shape）：

    > 1. 核间切分B、N2、G、S1轴，核内切分S1轴、S2轴模板：
    >
    >    tiling代码文件：ops-transformer-dev/attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling_s1s2_bn2gs1s2.cpp
    >
    >    tiling代码类：FlashAttentionScoreGradTilingS1s2Bn2gs1s2
    >
    >    kernel代码：ops-transformer-dev/attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad_s1s2_bn2gs1s2.h
    >
    >    条件：支持所有shape，其他模板如果不支持，就会走到这个模板
    >    依据：这个模板按照最通用的做法，可以支持所有的Shape。但是由于核内一次只能处理S1 * S2的大小，当S1和S2都比较小的时候，会有频繁的CV交互开销，性能较差。当S1和S2都比较小的时候会路由到下面的这些模板。
    >
    > 
    >
    > 2. 核间切分B、N2轴，核内切分G、S1、S2，该模板是通过单纯的分核改变来优化S1、S2都比较小且(B * N2比较大或者G = 1)场景下的性能
    >
    >    tiling代码文件：ops-transformer-dev/attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling_s1s2_bn2.cpp
    >
    >    tiling代码类：FlashAttentionScoreGradTilingS1s2Bn2
    >
    >    kernel代码：ops-transformer-dev/attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad_s1s2_bn2.h
    >
    >    条件：(S1 < 1024) && (S2 < 1024)  && (B \* N2 \* 2 > CoreNum || G == 1)
    >
    >    依据：S1和S2都比较小，且B和N2比较大的时候，这时候把B和N2用于分核，核内不切分N2.i，循环N2.i次进行Cube和Vector计算。
    >
    > 
    >
    > 3. 核间切分B、N2.o轴，核内切分N2.i、G、S1、S2轴, 该模板是为了优化G \* S1 \* S2都比较小的场景时的性能，把N2轴切分到核内，并且在核内也切分N2.i轴，用于加速Vector计算。相比于模板2, 模板3会更复杂一些，模板3在核内计算中也切分了N2.i轴，让每次的计算量更大。
    >
    >    tiling代码文件：ops-transformer-dev/attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling_ngs1s2_bn.cpp
    >
    >    tiling代码类：FlashAttentionScoreGradUngs1s2BbnTiling
    >
    >    kernel代码：FlashAttentionScoreGradUngs1s2Bbn
    >
    >    条件：G \* S1 \* S2 <= 32KB && G = 1 && S2 < 1536
    >
    >    依据：当G \* S1 \* S2小于32KB时，可以通过把N2轴切分一部分到核内，让CV基本块更大，同时在Vector核内，把N2.i的也进行切分，让单次Vector的计算量更大，提升Vector利用率。
    >
    > 
    >
    > 4. 核间切分B轴，核内计算B.i 、N2、 G、S1、S2轴，该模板是为了优化N2 \* G \* S1 \* S2比较小时的性能
    >
    >    tiling代码文件：ops-transformer-dev/attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling_bngs1s2_b.cpp
    >
    >    tiling代码类：FlashAttentionScoreGradUbngs1s2BbTiling
    >
    >    kernel代码：FlashAttentionScoreGradUngs1s2Bbn
    >
    > 条件：N2 \* G \* S1 \* S2 <= 64 \* 128
    >
    > 依据：如果希望单纯的把B.i放入CV基本块中，那么内层轴N2 \* G \* S1 \* S2就需要足够小，一般是根据这个只小于64KB的话，Bmm1和Bmm2的数据量一般不会超过L1的一半，那么B轴切分时有意义的，否则单个Matmul就把L1用满，多个Matmul之间的数据搬入没有办法和计算并行。

  - **<term>Ascend 950PR/Ascend 950DT</term>**  
  FAG算子的模板划分如下，以下模板，序号越大，模板的优先级越高，序号1的模板是泛化模板（支持所有shape），虽然存在多个模板，但在实现时仅存在两个模板文件，一个是确定性计算另一个是非确定性计算模板，其中非确定性计算模板包含了BN2，BN2S2，BN2GS1S2三种切分模板，在代码中通过模板参数隔离各自的实现逻辑：

    > 1. 核间切分B、N2、G、S1、S2轴模板：
    >
    >       tiling代码文件：ops-transformer-dev/attention/common/op_host/arch-310/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp
    >
    >       tiling代码类：FlashAttentionScoreGradTilingUs1s2Bs2Regbase
    >
    >       kernel代码：ops-transformer-dev/attention/common/op_kernel/arch-310/flash_attention_score_grad_kernel.h
    >
    >       条件：支持所有shape，其他模板如果不支持，就会走到这个模板
    >       依据：这个模板按照最通用的做法，可以支持所有的Shape。但是在部分场景下性能不是最优，此时根据不同场景判断路由到以下其他特化模板。
    >
    > 2. 核间切分B、N2轴模板：
    >
    >       tiling代码文件：ops-transformer-dev/attention/common/op_host/arch-310/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp
    >
    >       tiling代码类：FlashAttentionScoreGradTilingUs1s2Bs2Regbase
    >
    >       kernel代码：ops-transformer-dev/attention/common/op_kernel/arch-310/flash_attention_score_grad_kernel.h
    >
    >       条件：S1<=128 and S2 <= 128 and G=1，非FP32。
    >       依据：S1<=128 and S2 <= 128 and G=1时，不存在多核切分S1S2轴，即不存在多核累加，可以省去前置GM清零和后置cast和musl计算，性能更优。
    >
    > 3. 核间切分B、N2、S2轴模板：
    >
    >       tiling代码文件：ops-transformer-dev/attention/common/op_host/arch-310/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp
    >
    >       tiling代码类：FlashAttentionScoreGradTilingUs1s2Bs2Regbase
    >
    >       kernel代码：ops-transformer-dev/attention/common/op_kernel/arch-310/flash_attention_score_grad_kernel.h
    >
    >       条件：非FP32，TND，B \* N2 \* S2 > cubeCoreNum and  G = 1。
    >       依据：B \* N2 \* S2 > cubeCoreNum and  G = 1场景，dk和dv的计算无需多核累加，可以省去前置dk dv GM清零和后置cast和musl计算，性能更优。
    >
    > 4. 确定性计算模板
    >
    >       tiling代码文件：ops-transformer-dev/attention/common/op_host/arch-310/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp
    >
    >       tiling代码类：FlashAttentionScoreGradTilingUs1s2Bs2Regbase
    >
    >       kernel代码：ops-transformer-dev/attention/common/op_kernel/arch-310/flash_attention_score_grad_kernel_deter.h
    >
    >       条件：开启确定性计算。
    >       依据：确定性计算场景对分核有比较严格的要求，通过特定分核方式避免多核同地址累加，达到确定性计算的效果。

## 6 编程视角

### 6.1 AscendC高阶API

**<term>Atlas A2 训练系列产品</term>**

当前AscendC高阶API提供了两种编程模式，第一种是以Vector为主核，Cube为从核的视角，Vector0和Vector1会独立发起Matmul的任务，两者没有关联性。

第二种是以Cube为主核，Vector为从核，这时会由V0统一发起Matmul任务，这个任务的结果由V0和V1共同处理，一般是V0、V1各处理一半。当前如果某个模板的结尾是"_sab"，说明这个模板是一个以Cube为主核的模板。例如：

```c++
ops-transformer-dev/attention/flash_attention_score/op_kernel/flash_attention_score_s1s2_bn2gs1_sab.h
    
ops-transformer-dev/attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad_s1s2_bn2gs1s2_sab.h 
```

以Cube为主核对于FlashAttention来说由于V0、V1的Matmul任务可以复用左矩阵，且输出的部分结果可以在L0C累加，减少了对于带宽的依赖诉求，大部分场景性能会更优。

**<term>Ascend 950PR/Ascend 950DT</term>**

为了实现极致性能，除部分确定性计算场景，全部切换到AscendC低阶API实现，且无论是高阶还是低阶都是以Cube为主核实现。

### 6.2 AscendC低阶API

**<term>Atlas A2 训练系列产品</term>**

当前还存在一些没有使用AscendC高阶API的模板，例如：

```c++
ops-transformer-dev/attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad_s1s2_bn2gs1s2_basic.h
```

这个模板更加彻底地使用了以Cube为主核，Vector为从核，这时Matmul的任务都已经完全从Cube侧发起，通过同步通知Vector侧。

**<term>Ascend 950PR/Ascend 950DT</term>**

除部分确定性计算场景，其余场景全部采用低阶API实现，可以实现极致的内存复用。
