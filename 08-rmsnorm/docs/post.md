# RmsNorm 在 A2 上如何按行长选择归约方式

RmsNorm 只有平方、求和、倒平方根和逐元素乘法，但同一个 Kernel 很难同时处理短行、大行和混合精度参数。CANN 9.0.0 的实现把这类问题落实为四种模板：Normal、MergeN、SingleRow 和 SplitD。最值得研究的是 Host 如何根据 UB 中实际存活的数据选择模板，以及 Kernel 如何复用 gamma 和归约缓冲。

本文使用 `cann/ops-nn` 正式标签 `v9.0.0`，提交 `0413e303e6cd1986b592c3e039f2579ac0f10b62`。产品表列出 A2/A3，源码快照保留 `ascend910b` 与 `ascend910_93` 配置。分析集中在原有非 RegBase 路径；`arch35` 的 950 SIMD 实现未用于这里的结论。[版本配套](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)、[算子源码](https://gitcode.com/cann/ops-nn/tree/0413e303e6cd1986b592c3e039f2579ac0f10b62/norm/rms_norm)

## 将输入折成 N 行 D 列

输入的归一化维度折成 D，其余维度折成 N。每行需要输出 `y=x*rsqrt(mean(x²)+eps)*gamma`，以及 FP32 的 rstd。Host 按行给 AIV 分配工作，普通核采用 `ceil(N/coreNum)` 行，尾核保存实际剩余行数。短行并行的基本问题是每条 Vector 指令覆盖的元素少；大行的问题则是 x、平方结果、输出与 FP32 转换缓冲无法同时放进 UB。

`rms_norm_tiling.cpp` 将默认 `ubFactor` 设为 FP16/BF16 的 12288 元素或 FP32 的 10240 元素。最后一维按块对齐后超过这个预算就选择 SplitD。A2 的短行还另有 `SMALL_REDUCE_NUM=2000` 分支：**比较的是 numColAlign，不是原始 D**，因此阈值附近必须先考虑对齐。

## MergeN 一次处理多行并保留 gamma

当 A2 的对齐 D 不超过 2000，Host 选择 MergeN，计算 `rowFactor = usableUB / (numColAlign*16+260)`，再令 `ubFactor=rowFactor*numColAlign`。16 和 260 来自这个模板的多份输入/输出、FP32 中间结果和行归约存储预算，不是单独 x 的字节数。

`KernelRmsNormMergeN::Process` 先 `CopyInGamma`，将其 DeQue 后在整段 row loop 中持有，最后才释放。非特定性能分支还会先 `BroadCastGamma`，使同一份参数适应多行运算。每轮 `DoMainCompute` 搬入多个 Query 行，用平方缓冲与归约临时空间产生各行 rstd，再乘 gamma 和写出 y。

这种安排减少了同核上相邻行重复读取 gamma 的次数；它并不意味着所有核共享 UB，gamma 仍由各核各搬一次。可检查的源码位置是 [MergeN 主循环](../code/upstream/norm/rms_norm/op_kernel/rms_norm_merge_n.h) 和 [Host 模式选择](../code/upstream/norm/rms_norm/op_host/rms_norm_tiling.cpp)。

## 混合精度下选择依据是完整 UB 账单

x 为低精度、gamma 为 FP32 时，`CalMixDtypeTiling` 逐步尝试：短行 MergeN、整行 Normal、减少输入输出空间的 SingleRow，最后才是 SplitD。预算包括低精度 x/y、FP32 gamma、两份 FP32 临时量、rstd 和 Reduce 空间。因此相同 D 的混合精度形状可能比统一 dtype 更早切换模板。

这里有一个有用的设计方法：先检查“能否通过缩短同时存活的缓冲保住整行”，再采用增加 GM 读取的拆分方案。SingleRow 的源文件展示了这个中间选项，不能将 Normal 与 SplitD 看成唯一的二选一。[混合精度预算](../code/upstream/norm/rms_norm/op_host/rms_norm_tiling.cpp)、[SingleRow Kernel](../code/upstream/norm/rms_norm/op_kernel/rms_norm_single_row.h)

## SplitD 用两遍换取较小工作集

`KernelRmsNormSplitD::SubProcess` 首先在各列分片上累计平方和，计算整行 rstd；之后再遍历列分片，搬入 x 与相应 gamma，完成归一化输出。非老架构分支使用 `ComputeFormerPrecision`，将列片归约组织成多个层级，而不是把全部 D 的 FP32 中间结果保留到最后。

这使 UB 使用量与列片大小相关，支持更宽的归一化维度。代价是 x 要读两遍，gamma 分片在输出阶段搬入；它属于容量约束下的实用选择，不应因为名字含 Split 就宣称一定更快。Host 会对 `ubFactor` 做块对齐，并调整过小的不对齐尾片，避免尾部搬运和归约无法满足要求。[SplitD 的两遍流程](../code/upstream/norm/rms_norm/op_kernel/rms_norm_split_d.h)

平方和在 FP32 中间量上累计。`reduce_common.h` 与 `rms_norm_base.h` 实现了分段加法、WholeReduceSum/BlockReduceSum 和尾部 mask，阅读时要跟随实际选择的架构宏。`BUFFER_NUM=1` 也明确提醒：这里的优化重点是分块、参数复用和归约组织，不能把队列默认理解成 Double Buffer。

## 怎样验证模式切换是否带来收益

可用 A2 上的 `N=4096` 配合 `D=1024、2048、8192、12288、16384` 观察不同分支。1024 是短行候选，2048 越过短行阈值，16384 在普通低精度预算下进入 SplitD；实际路径仍应以 Tiling 日志为准。再固定 x 为 BF16，分别使用 BF16 与 FP32 gamma，检查混合精度是否改变选择。

一个具体消融是独立副本中取消 A2 的短行 MergeN 条件，同时保持同一输入、gamma 与 epsilon，走已存在的 Normal 路径；只对能容纳整行的 D 比较。记录加载次数、Vector 归约时间与正确性，比只报告总调用速度更能解释 gamma 复用的收益。另一组比较拆分大小时，应保留尾块对齐和所有缓冲预算，不可只增大 12288 常量。

输出同时检查 y 与 rstd，特别覆盖 D 不对齐、N 小于核数、尾核少行、近零输入以及较大数值。这里提供固定源码和 [复现命令](reproduce.md)，没有公开 A/B 数字或本机 NPU 测量；可迁移的是源码中有明确阈值、缓冲与生命周期依据的选择过程。
