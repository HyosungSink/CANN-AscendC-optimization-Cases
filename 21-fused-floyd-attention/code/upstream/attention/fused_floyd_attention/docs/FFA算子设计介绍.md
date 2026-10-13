## 1 计算过程

按照FusedFloydAttention正向计算流程实现，整体计算流程如下：

1. <query, key_1>通过特定矩阵乘计算得到a_0与<query, key_2>通过特定矩阵乘计算得到的a_1相加，然后再乘以缩放系数scale_value。此时的结果通过atten_mask进行select操作，将atten_mask中为true的位置进行遮蔽，得到结果masked_attention_score，即atten_mask中为true的位置在select后结果为负的极小值，经过softmax计算之后变成0从而达到遮蔽效果。

2. 借鉴FlashAttention加速算法，FusedFloydAttention使用FlashSoftmax操作对masked_attention_score进行运算，用以代替原公式中的softmax运算，而后将结果与value做matmul运算。由于FlashSoftmax操作对masked_attention_score的Skv(输入key、value的sequence length)方向进行了切分，故实现过程中存在一个刷新流程，具体如下：
    1. 每次FlashSoftmax计算只对切分后的一个SkvSplit（SkvSplit是针对Skv轴进行切分之后的序列长度的简称）进行操作，并从第二次循环开始记录exp，其中 i 表示Skv切分后的循环变量，针对exp的i是从1开始 ，exp的计算公式如下：
       $$
       exp[i] = e^{max_{i - 1} - max_{i}}
       $$
    2. 当i = 0时，计算出的MM[PV]结果直接保存到ub_attention_out[0]的ub中。
    3. 从i = 1开始，需要增加Mul和Add操作，即将上一次的MM[PV]的结果和当前exp相乘，相乘完的结果和本次MM[PV]的结果相加得到的结果保存到ub_attention_out[1]的ub中。以此类推，遍历Skv计算完成。
    4. 由于FlashSoftmax计算中的除sum被后移到输出attention_out之前，因此最后需要将ub中的ub_attention_out按行除以softmax_sum并将最终完整的结果保存到输出内存attention_out(Final)上。

## 2 Tiling设计

Tiling操作的目的是为了找到一种更高效的NPU执行方式，原始的数据量一般是非常大的，没有办法通过一次指令调用就完成所有计算，因此需要将数据量分到多个核上并行计算，且每个核上也需要考虑如何循环计算性能最优，不同的输入可能有不同的最优执行方式，所以需要通过tiling策略决定怎么将数据分配到各个核上进行计算。

 根据硬件架构特征，AI Core分成AIC和AIV两个独立的核，AIC和AIV核拥有自己独立的Scalar计算单元，能够独立加载自己的代码段，单独执行。AIC和AIV分离的架构可以使得AIC和AIV并行执行。AIC和AIV之间数据交互的通路是L2和GM（Global Memory，高带宽存储器），两者之间的交互次数对性能影响是比较大的，同时由于AIC和AIV算力差异，两者需要使用不同的基本块大小，本着尽量减少AIC和AIV通信次数和发挥最大算力的原则，CVtiling分离策略应运而生，可以有效地减少CV通信次数，同时根据不同单元的buffer特征，选择不同的基本块进行计算，从而提升算子性能。

 对于FFA算子，Vector计算涉及多个输入、输出、中间计算结果、double-buffer设计等，需要将buffer分配成多份，最优分配方案中最大一份为32KB，由于Vector计算使用的数据类型是float32，因此Vector的tiling基本块为8 \* 1024。为了充分发挥Cube的算力，在CV之间一轮计算的数据量进行了4:1的配比，又由于Cube侧的输入数据类型是float16，输出是float32，Cube的基本块为32 \* 32，所以通过nRatio=32配比出32 \* 1024的数据量。伪代码如下：

```c++
// C-Tiling: (S1_c_i,D)x(D,S2_c_i) => (S1_c_i, S2_c_i):(32,1024)
// V-Tiling: (S1_v_i, S2_v_i) => (8,1024)

// C侧 matmul计算
Bmm((S1_c_i,D)x(D,S2_c_i)) => 32*1024  // 输出结果32*1024，放到workspace上
// V侧 Vector计算
for S1_c_i/S1_v_i=32/8:
	copy_gm_to_ub(S1_v_i*S2_v_i)  // 从bmm的workspace上拷入bmm结果数据
	Vector(S1_v_i,S2_v_i)         // 进行Vector计算
	copy_ub_to_gm(S1_v_i*S2_v_i)  // Vector计算结束，得到最终输出数据，拷贝到GM上

// 由于Cube侧计算数据比Vector侧大，因此，ub内需要再次进行Vector Tiling，从而产生了S1方向的配比：S1_c_i/S1_v_i
```

上述示例中，仅在S1方向开了配比，S2方向C/V计算的长度是一致的，当然，也可以在S1/S2方向均开启配比；这样做的好处是，Cube一次可以发射大块的数据，避免因为小块数据不断发射带来的通信开销，也能最大程度地使用Cube单元的buffer。

## 3 流水设计

为了追求极致性能，必须充分利用硬件资源，通常需要进行不同pipeline的流水设计。流水设计的宗旨是尽量使某一条pipeline达成bound效果，使硬件的某一个单元一直在工作，达到性能上限。

### 3.1  V侧流水

V侧流水设计需要考虑Vector内部的搬运及计算过程，实施的优化手段主要是double buffer。 Vec的功能被拆分成2个流水任务：Stage1、Stage2，每个任务专注于完成单一功能；需要处理的数据被切分成2片，使用ping-pong表示两个数据处理任务，每个任务需要依次搬运DataCopy与计算Clc操作。任务间的箭头表达数据间的依赖关系，比如Stage1处理完DataCopy之后，Stage2才能对Clc进行处理。 不进行流水设计时，搬运与计算任务之间是串行执行的，会出现断流现象，即第一次DataCopy完成之后的搬运流水就一直处于空闲状态，直到第一次搬入的数据计算完成并搬出之后搬运流水才会继续工作，进行第二次DataCopy（Vector计算和搬出流水也存在同样问题）。通常这种情况下，性能是极差的。

对ping-pong流水间的double buffer处理后，对于同一片数据，搬运DataCopy与计算Clc之间的处理具有依赖关系，需要串行处理；不同的数据切片，同一时间点，可以有多个任务在并行处理，由此达到任务并行、提升性能的目的。

其中ping、pong两块计算数据所占用的内存资源均相互独立。融合算子V侧计算过程较多，情况也比较复杂，通常简单的double buffer是无法覆盖所有情况的，因此会出现不同的计算流水排布。

### 3.2 CV流水

对于融合算子，V侧的计算是依赖C侧的计算结果的，如果只关注V侧流水，不关注C侧，则C侧与V侧很有可能是串行流水的效果，不能达到并行计算的目的，无法使得融合算子性能达到最优，从而有了CV流水设计。此外，CV流水在不同算子情况下，表现的现象也是不一致的，Cube双发机制可实现两种场景下的流水优化：

- C侧总耗时 > V侧总耗时

  该场景流水特征下，Vector计算节点少，计算速度快，在Atlas A2 训练系列产品 C:V=1:2的情况下，Cube的搬运时长足以掩盖Vector的计算时长，因此只要关注Cube的MTE2耗时即可，最终达成MTE2 bound。在Cube双发机制下，提前发射两块Cube计算，Cube1、Cube2计算可以衔接，使得Cube利用率最高，达成Cube bound。

- C侧总耗时 < V侧总耗时

  C侧连续发射两块Cube计算，这样可以保证V侧计算完上一轮时，可以立马启动当前轮的计算，而不用等待Cube1的数据。这样可以使V侧一直在工作，达成Vector bound。

## 4 优化效果

- 单算子推理耗时

| 输入shape（B, H, N, M, K, D） | Chunk（s） | AscendC（s） | 加速比 (Speedup) |
| :--- | :--- | :--- | :--- |
| (1, 32, 256, 256, 256, 32)    | 0.0282     | **0.0209** | **1.35x** |
| (1, 32, 384, 384, 384, 32)    | 0.0955     | **0.0604** | **1.58x** |
| (1, 32, 512, 512, 512, 32)    | 0.1941     | **0.1383** | **1.40x** |
| (1, 32, 768, 768, 768, 32)    | 0.7640     | **0.4563** | **1.67x** |
| (1, 16, 256, 256, 256, 64)    | 0.0158     | **0.0142** | **1.11x** |
| (1, 16, 384, 384, 384, 64)    | 0.0561     | **0.0316** | **1.78x** |
| (1, 16, 512, 512, 512, 64)    | 0.1205     | **0.1151** | **1.05x** |
| (1, 16, 768, 768, 768, 64)    | 0.4249     | **0.3302** | **1.29x** |

- 显存占用

| 输入shape（B, H, N, M, K, D） | Chunk（GB） | AscendC（GB） | 显存节省率 (Mem Saving) |
| :--- | :--- | :--- | :--- |
| (1, 32, 256, 256, 256, 32)    | 6.38        | 2.65          | **58.46%** |
| (1, 32, 384, 384, 384, 32)    | 19.97       | 5.51          | **72.41%** |
| (1, 32, 512, 512, 512, 32)    | 45.50       | 9.50          | **79.12%** |
| (1, 16, 256, 256, 256, 64)    | 3.88        | 2.40          | **38.14%** |
| (1, 16, 384, 384, 384, 64)    | 11.53       | 5.18          | **55.07%** |
| (1, 16, 512, 512, 512, 64)    | 25.50       | 9.13          | **64.20%** |

## 5 附录——FusedFloydAttention算子中IterateBatch配置

本方案中涉及两种类型的批量矩阵乘（数据格式ND）

第一种是: NMD @ NKD, 其中N是batch轴， D是K轴;

第二种是: NMD @ KMD, 其中M是batch轴， D是K轴

第一种方案中batch轴在最外轴，M和K可能有切片，这样每次单batch的ND矩阵是连续的，但是不同batch的ND矩阵间可能会存在间隔。

比如M=256, 假设切片为N[0:16]M[128:256]D[:], 每个batch为 128 \* D 的数据， batch数据起始地址之间间隔256 \* D;

针对这种情形我们使用IterateBatch的NORMAL数据排布，对于batch数据之间的地址间隔，需要配置matrixStrideA， matrixStrideB等参数；这两个参数能指定输入A矩阵和B矩阵的相邻batch的ND矩阵间的间隔。上述的例子中，将matrixStrideA设置为 256*D。

第二种方案中batch轴在中间，这种情形下每次单batch的ND是不连续的，跳跃的。一种方法是实现中自己控制数据搬运，使用低阶接口；或者使用AscendC中的高阶接IterateBatch。

使用IterateBatch，针对batch在中间轴的情况，可配置使用**BSNGD**的数据排布格式。其中每个batch计算一个SD数据。例如NMD @ KMD，可以把B矩阵KMD理解为[B=1,S=K,N=1,G=M, D=D]这种排布，这样就能使用IterateBatch高阶接口。

使用IterateBatch每次调用时的batch数据需要指定，但是不能任意指定，有硬件约束。主要是要求指定的计算batch数据能完全搬运进L1 cache(对于910B为512KB)，即计算数据量不能超过NPU L1 Cache。下面是相关的计算流程。

- **计算一个批次（Batch）矩阵乘计算的数据量 (Bytes)**

  oneBatchBytes = datatype Bytes \* D \* (K1 + K2)

  K1与K2取实际计算的矩阵的轴的切片长度，比如上面例子中M[128:256]对应的是128

- **实际最大的batch数量则如下**

  maxBatchNums = Floor(L1Size / oneBatchBytes)  # 向下取整

- **对应的循环次数如下**

  Loops = Ceil(TotalBatches, maxBatchNums)    # 向上取整
