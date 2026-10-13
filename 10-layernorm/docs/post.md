# LayerNormV4 的 Single Read 与短行转置归约

LayerNorm 比 RMSNorm 多了均值和减均值，输入与平方差要经历更多生命周期。CANN 9.0.0 的 LayerNormV4 提供了一个具体优化案例：在特定 BF16 行宽上，仅保留两份大 FP32 工作区，顺序复用已经完成归约的行来加载 gamma/beta；极短非对齐行则改为转置后批量归约。

本文固定 `cann/ops-nn` 正式标签 `v9.0.0`，提交 `0413e303e6cd1986b592c3e039f2579ac0f10b62`。选择 `norm/layer_norm_v4` 的非 RegBase 源码，A2/A3 产品表与 `ascend910b`、`ascend910_93` 配置都在该目录。`layer_norm_v3` 的同名新目录仅有 arch35 Kernel，不能据接口名称把它当成 A2/A3 实现。[版本映射](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)、[LayerNormV4](https://gitcode.com/cann/ops-nn/tree/0413e303e6cd1986b592c3e039f2579ac0f10b62/norm/layer_norm_v4)

## Host 使用能力检查而不是一个通用 Kernel

Host 的 `TilingRegistry` 逐个评估模板的 `IsCapable`。每个模板自己检查架构、dtype、行宽、UB 容量和 API 模式。源码中的 `rowSize` 指每个归一化组的长度 D，`colSize` 指组数 N，名字容易让人误读。本文统一用 N 行 D 列描述。

非 RegBase 路径包含 Transpose、SingleRead、MergeN 和 Common 等实现；950 RegBase 的 TwoPass/Welford 模板不是这里分析的对象。不能看到头文件定义了多种 dtype，就认定 Host 会为任意 dtype 选择它。

## SingleRead 当前仅对两个 BF16 行宽启用

`LayerNormV4SingleReadTiling::IsCapable` 明确要求 x 与参数均为 BF16，D 为 1024 或 4096，且不是 RegBase。`GetTilingKey` 虽然保留其他类型的 key，前面的条件已经排除了这些形状。

可用行容量为 `rowMax=((UB-8*BLOCK_SIZE)/4)/2`，代表两份 FP32 工作区。若 N 不超过核数，每核处理一行；否则以 `rowMax/rowAlign` 决定一轮行数，并限制到最多 32 行。Host 分别保存完整轮数、尾轮、尾部行数，Kernel 按这些参数处理，避免对齐填充部分进入有效统计。[SingleRead Tiling](../code/upstream/norm/layer_norm_v4/op_host/layer_norm_v4_single_read_tiling.cpp)

## 为什么输入只读一次却还能够加载权重

`LayerNormV4SingleRead::ProcessBasicBlock` 先把 x 搬到 `xLocal` 的低精度区域，Cast 到 FP32；`yLocal` 保存乘 `1/D` 后的数值，逐行归约出均值，再写成中心化后的 `x-mean`。随后 `xLocal` 被覆盖成平方差，归约计算方差与 rstd。

关键优化发生在方差的逐行循环中：第一行的平方差已经归约完成，就把 gamma 搬入该行原来的 xLocal 区；第二行完成后，将 beta 搬入第二行对应区域。没有额外的整行 gamma/beta FP32 Buffer，参数占用的是已经不再需要的中间结果空间。读入低精度参数时仍需等待 MTE2 后 Cast，写入前也需 `V_MTE2` 确认旧的 Vector 读者已经结束。

只有一行时没有第二行可借用，代码先乘 gamma，等其消费者结束后才把 beta 搬到同一区域。这个特殊分支解释了复用的条件：不是两个参数任意共址，而是先后消费使它们可以共享内存。最终所有行用 rstd 缩放，乘 gamma、加 beta，再 Cast 输出。[SingleRead 具体生命周期](../code/upstream/norm/layer_norm_v4/op_kernel/layer_norm_v4_single_read.h)

两份大工作区并不等于 Double Buffer。此实现的输入输出 queue 深度为 1，收益来自减少同时存活的大参数缓存和避免 x 的 GM 重读。对极小 N，逐行 Scalar 提取归约结果和事件同步仍然可能明显，不能从内存复用直接推导所有规格都更快。

## D 不超过 64 的非对齐行采用 Transpose

`LayerNormV4TransposeTiling::IsCapable` 接受 D≤64 且 D 与 rowAlign 不相等的非 RegBase 形状，并进一步检查 dtype key。比如 FP16 的 D=33，按常规逐行处理会有较多 padding，归约时一条指令的有效数据不多。

`LayerNormV4Transpose::Process` 先将 gamma/beta 搬入并保留。每个 `ProcessBasicBlock` 将一批 x 搬入带 padding 的 UB 区，`DoTranspose` 改变行列组织，随后 `DoReduce` 对排布后的数据做均值和平方差归约；结果经 `DoPostTranspose` 回到输出布局。mean 与 rstd 也有独立的结果重排与写出逻辑。

这个策略增加 UB 内转置，换取短行归约时更规则的 Vector 数据组织。它只在原布局造成浪费时启用，D 已对齐会被能力检查排除。测试必须包含 D=31/32/33/63/64 这样的边界，才能看到模板选择与尾部处理的变化。[Transpose Tiling](../code/upstream/norm/layer_norm_v4/op_host/layer_norm_v4_transpose_tiling.cpp)、[Transpose Kernel](../code/upstream/norm/layer_norm_v4/op_kernel/layer_norm_v4_transpose.h)

## MergeN 面向每核多行的工作集

`LayerNormV4MergeNTiling` 排除 RegBase、310P 与 V3 模式，只有每核预计行数超过 2 才考虑多行模板。它用可用 UB、D 与对齐行宽决定一次行数；D≤1280 还有不同的 align-limit key。因而 N 与 D 同时决定是否值得多行处理，不能只按总元素数合并。

一个直接消融是在独立工程中仅让 SingleRead 的 `IsCapable` 返回 false，固定 BF16 D=1024/4096，让注册表选择下一条原有有效路径。确认 Tiling key 改变后比较计时和 y/mean/rstd 精度，才能判断生命周期复用的作用。Transpose 也可用相同方法禁用单个能力检查，比较相同 D 的 fallback，不能通过把 D 改成 32 来构造“关闭转置”的基线。

所有修改应保存 diff 和模板选择日志。测试至少覆盖参数为空、N=1、N 小于核数、尾部不足一轮、近常数输入和低方差输入；方差是平方差均值，不可用 `E[x²]-E[x]²` 的消减误差充当这个实现的精度参考。固定构建路线见 [复现说明](reproduce.md)，本文没有已测性能表。
