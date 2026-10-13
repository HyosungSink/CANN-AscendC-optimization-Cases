# AddLayerNorm 如何用较少的 UB 缓冲保住整行计算

AddLayerNorm 将残差相加、可选 bias 和 LayerNorm 放在一个算子内。CANN 9.0.0 的非 950 实现展示了比“融合减少 Kernel launch”更具体的优化：Host 为多种存储组织计算 UB 预算，能批量处理时采用短行特殊归约，整行放不下时先尝试复用临时 tensor，最后才切分 D。

本文使用 `cann/ops-nn` 正式 `v9.0.0`，固定提交 `0413e303e6cd1986b592c3e039f2579ac0f10b62`。上游产品表支持 A2/A3，快照保留两种 SoC 配置与原有 `add_layer_norm.cpp` 及辅助 Kernel。`add_layer_norm_apt.cpp` 的 arch35 FullLoad/Welford 不用于本文结论。[版本配套](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)、[算子源码](https://gitcode.com/cann/ops-nn/tree/0413e303e6cd1986b592c3e039f2579ac0f10b62/norm/add_layer_norm)

## 额外输出与 bias 决定需要保存什么

基本计算是 `z=x1+x2+bias`，随后对最后若干归一化维度求 mean、rstd，输出 `(z-mean)*rstd*gamma+beta`。`enableAdditionalOutput` 控制是否还需要保存残差 z，bias 则分为没有、全量和按参数维度广播。它们决定 Kernel 能否提前释放输入与临时结果，也决定 Host 的 UB 公式。

Host 按归一化组折成 N×D，沿 N 分核，不同 dtype 的输入可能先提升到 FP32。默认 queue 的 `bufferNum=1`，这个案例不能简单归为 Double Buffer 优化。[Host 预算与分核](../code/upstream/norm/add_layer_norm/op_host/add_layer_norm_tiling.cpp)

## 短 D 全 FP16 的特殊归约

非 310P、x/参数均 FP16、bias 无或为广播，且对齐后的 D 不超过 `255*8=2040` 时，Host 优先尝试 `NORMAL_SPECIAL_REDUCE`。它把 gamma/beta 的 FP32 缓冲和可选 bias 当作固定成本，把输入输出、行统计及中间结果当作每行成本；只在 UB 能放超过一行时选择该路径。

`KernelAddLayerNormNormalSpecialReduce::PrecisionCompute` 先将残差乘 `1/D`，由 `ReduceSumForSmallReduceDim` 产生多行均值，再减均值、平方、第二次归约得到方差，最后 sqrt、求倒数并做 affine。其 Vector repeat 与 stride 按多行布局组织，reduce tail mask 对非对齐 D 很重要。

源码还针对大 N 保留广播或转置实现，使每条指令有更多有效工作。减少按行发指令不等于改变 LayerNorm 的统计定义，仍需每行独立 mean/rstd。可核对 [特殊归约 Kernel](../code/upstream/norm/add_layer_norm/op_kernel/add_layer_norm_normal_special_reduce.h) 和 [基础 Reduce 工具](../code/upstream/norm/add_layer_norm/op_kernel/add_layer_norm_base.h)。

## BetterUB 是一个有边界的容量优化

普通多行预算认为需要三份 FP32 中间结果，而 BetterUB 的广播 bias 全 FP16 特例用更少临时量重算一次可容纳行数。只有普通预算可放 0～2 行、特殊预算能放超过 2 行时，Host 选择 `NORMAL_SPECIAL`。这不是“参数越少就总选 BetterUB”，而是具体容量区间内才能增加每轮行数。

两条路径的比较应该固定 D、N、dtype、bias 和额外输出，只改模板选择。改变 `enableAdditionalOutput` 后 GM 写出和缓冲成本都变了，不能当作相同语义基线。

## 宽行先尝试缩短 tensor 生命周期

若 Normal 放不下多行，Host 依次尝试 SingleRow、共享输入空间的 SingleRowExt，然后是 `SINGLE_ROW_LESS_TENSOR`。最后一种适用于相应非 310P 的无 bias 全 FP32，以及低精度 x、FP32 参数的无/广播 bias 场景，还要求额外输出开启，并通过各自的 UB 预算检查。

`KernelAddLayerNormSingleRowLessTensor::Process` 每次先 `CopyInAdd`，需要残差输出就立刻 `CopyOutX`。接着只搬 gamma，执行 `ComputeFirstPart`；gamma 消费结束后再搬 beta，执行 `ComputeSecondPart` 和输出。`inputOutputQue` 被重复用于输入/参数/输出阶段，另外只保留两份 FP32 工作区。这样不用让 x1、x2、gamma、beta、y 的独立大缓冲同时存活，可让更多宽行在 UB 内完成归一化。[少 tensor 路径](../code/upstream/norm/add_layer_norm/op_kernel/add_layer_norm_single_row_less_tensor.h)

这里的“少”是峰值同时占用少，并不代表读取 gamma/beta 次数一定更少：该单行主循环会按行搬入参数，多行模板则可能在核内保留参数。容量与参数重读需要一起评估。使用 ReinterpretCast 和共址时，代码依赖 MTE2/V/MTE3 等事件确保上一消费者已经结束，不能只沿算术顺序证明复用安全。

## 最后才沿 D 切片

整行仍无法容纳时，Host 计算 `colPerTime`，检查最小列片限制 2048 和搬运块对齐，再选择 Slice 或 SliceExt。切片需要跨片统计，并在统计完成之后继续处理输出，意味着额外输入读取或中间数据存储。少 tensor 路径的价值之一，就是让部分 shape 留在单行阶段，避免进入这种更复杂的数据流。

`add_layer_norm.cpp` 按 key 把普通模板、BetterUB、SpecialReduce 与 LessTensor 实例连接起来。key 中还编码 bias 类型、额外输出等条件，因而同一个类名不同模板参数的存储/精度行为可能不同。只修改 Host key 而不保证 Kernel 模板支持对应输入会导致错误。

## 可执行的比较方式

短行实验可用全 FP16、D=1024、N 足够大、无 bias，原版会成为 SpecialReduce 候选。独立副本中只关闭相应的能力条件，检查普通路径能否容纳同一输入后比较。在宽行实验中先从 Host 日志找到真实触发 LessTensor 的 D，再仅关闭它，验证 fallback 切到 Slice；精度与额外输出相同，才有意义。

验证必须分别检查 y、mean、rstd 和额外残差 x；低方差、常数、混合 dtype、广播 bias、全量 bias、非对齐 D 和尾核行数都应覆盖。本文提供实际 Kernel、Tiling、源码哈希与 [复现路线](reproduce.md)，没有伪造 A/B 数值。这个案例能帮助判断究竟该缩小工作集、增加行批量，还是接受两阶段的 D 切片。
